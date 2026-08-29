/*
 * h3_vae_decode.c — H3 pipeline M5: video VAE decoder (ViT3D) spike.
 *
 * Semantics pinned from comfy/ldm/minimax/vae.py (ViT3DDecoder +
 * MiniMaxH3VideoVAE.decode):
 *   z [24, T, H, W] --(*latents_std + latents_mean)--> post_quant_conv
 *   (1x1x1 conv 24->24) -> x_embedder (24->2048) per latent voxel
 *   -> append 4 register tokens (embed space, no x_embedder) + 1 zero
 *   token -> 36 x TransformerBlock:
 *       x += attn(rmsnorm(x, norm1)) * scale1
 *       x += ff(rmsnorm(x, norm2))  * scale2
 *     attn: to_qkv (out 6144 laid out per head [q64|k64|v64]x32),
 *       per-head qk rmsnorm (eps 1e-5, no affine), split-half rope over
 *       the first 48 of 64 dims: pair j couples (x[j], x[24+j]),
 *       angle = 2pi * coord[axis(j/8)] * inv_freq8[j%8],
 *       inv_freq8[i] = 100^(-0.125*i), coords = 2*((i+.5)/n)-1 per axis,
 *       suffix tokens (registers + zero) get coord 0 (identity rope).
 *     ff: w1 (2048->16384) chunk gate|val, silu(gate)*val, w2 (8192->2048).
 *   -> LayerNorm(2048, affine) -> proj_out (2048->3072) -> first n_patch
 *   tokens -> unpatchify [T,H,W,3,4,16,16] -> (3, 4T, 16H, 16W)
 *   -> pixel * IMAGENET_STD + IMAGENET_MEAN -> clamp [0,1].
 *
 * Weights: minimax_h3_video_vae_fp16.safetensors, mmap-streamed (page
 * cache, never wired). Anonymous memory = activations only (few MB).
 *
 * Usage: h3_vae_decode <vae.safetensors> <patch_latent.bin> <vt> <lat_h>
 *                      <lat_w> <outdir>
 *   patch_latent.bin: [vt*(lat_h/2)*(lat_w/2), 96] f32 rows (the H3_X_IN
 *   format the h3_forward/h3_euler pipeline produces), row order t,hh,ww,
 *   96 = 24ch x 2x2 (c,b,a) — unpacked here to the full latent grid.
 */
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/task_info.h>

#define DIM 2048
#define HDIM 64
#define HEADS 32
#define NLAYERS 36
#define NREG 4
#define ROT_PAIRS 24           /* 48 rotary dims = first 48 of 64 */
#define ROT_ANG_FREQ 8         /* inv_freq count per axis (3 axes -> 24 pairs) */
#define PROJ_OUT (3 * 4 * 16 * 16)
#define LATENTS_DIM 24

static const float PIXEL_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float PIXEL_STD[3] = { 0.229f, 0.224f, 0.225f };

/* ---------------- safetensors reader (mmap, zero copy) ---------------- */

typedef struct {
    char name[256];
    char dtype[16];
    uint64_t start, end;
    uint32_t ndim;
    uint64_t dims[8];
} SEntry;

typedef struct {
    int fd;
    const uint8_t* base;
    size_t size;
    uint64_t hlen;
    SEntry* entries;
    size_t n;
} SFile;

static int sopen_(const char* path, SFile* sf) {
    sf->fd = open(path, O_RDONLY);
    if (sf->fd < 0) return -1;
    struct stat st;
    if (fstat(sf->fd, &st) != 0) return -1;
    sf->size = (size_t)st.st_size;
    sf->base = mmap(NULL, sf->size, PROT_READ, MAP_SHARED, sf->fd, 0);
    if (sf->base == MAP_FAILED) return -1;
    memcpy(&sf->hlen, sf->base, 8);
    size_t cap = 64;
    sf->entries = malloc(sizeof(SEntry) * cap);
    sf->n = 0;
    const char* p = (const char*)sf->base + 8;
    const char* end = p + sf->hlen;
    while (p < end) {
        const char* q = memchr(p, '"', (size_t)(end - p));
        if (!q) break;
        const char* name0 = q + 1;
        const char* name1 = memchr(name0, '"', (size_t)(end - name0));
        if (!name1) break;
        size_t nlen = (size_t)(name1 - name0);
        const char* obj = memchr(name1, '{', (size_t)(end - name1));
        const char* objend = memchr(name1, '}', (size_t)(end - name1));
        if (!obj || !objend || obj > objend) { p = name1 + 1; continue; }
        if (sf->n >= cap) {
            cap *= 2;
            sf->entries = realloc(sf->entries, sizeof(SEntry) * cap);
        }
        SEntry* e = &sf->entries[sf->n];
        size_t cpy = nlen < sizeof(e->name) - 1 ? nlen : sizeof(e->name) - 1;
        memcpy(e->name, name0, cpy); e->name[cpy] = 0;
        const char* dt = strstr(obj, "\"dtype\"");
        if (dt && dt < objend) {
            dt = memchr(dt, ':', (size_t)(objend - dt));
            if (dt) { while (dt < objend && (*dt == ':' || *dt == ' ')) dt++; if (dt < objend && *dt == '"') dt++; }
        }
        if (dt && dt < objend) {
            const char* dte = memchr(dt, '"', (size_t)(objend - dt));
            size_t dl = (size_t)(dte - dt); if (dl > 15) dl = 15;
            memcpy(e->dtype, dt, dl); e->dtype[dl] = 0;
        } else e->dtype[0] = 0;
        e->ndim = 0;
        const char* sh = strstr(obj, "\"shape\"");
        if (sh && sh < objend) {
            sh = memchr(sh, '[', (size_t)(objend - sh));
            if (!sh || sh >= objend) sh = objend - 1; else sh++;
            while (sh < objend && *sh != ']' && e->ndim < 8) {
                char* e2;
                unsigned long long v = strtoull(sh, &e2, 10);
                if (e2 == sh) break;
                e->dims[e->ndim++] = v;
                sh = e2;
                while (sh < objend && (*sh == ',' || *sh == ' ')) sh++;
            }
        }
        e->start = e->end = 0;
        const char* of = strstr(obj, "\"data_offsets\"");
        if (of && of < objend) {
            of = memchr(of, '[', (size_t)(objend - of));
            if (!of || of >= objend) of = objend - 1; else of++;
            char* e2;
            e->start = strtoull(of, &e2, 10);
            of = e2;
            while (of < objend && (*of == ',' || *of == ' ')) of++;
            e->end = strtoull(of, NULL, 10);
        }
        sf->n++;
        p = objend + 1;
    }
    return 0;
}

static float f16v(uint16_t h) {
    /* IEEE half: rebuild exponent+mantissa into f32 (shift <<13 into f32
     * fields), not <<16 — <<16 is the BF16 expansion and misreads F16 */
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h & 0x7C00u);
    uint32_t man = ((uint32_t)h & 0x03FFu);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;               /* zero */
        else {                                    /* subnormal -> normal */
            int e = -1;
            uint32_t m = man;
            do { e++; m <<= 1; } while (!(m & 0x0400u));
            m &= 0x03FFu;
            bits = sign | ((127u - 15u - (uint32_t)e) << 23) | (m << 13);
        }
    } else if (exp == 0x7C00u) {
        bits = sign | 0x7F800000u | (man << 13);  /* inf/nan */
    } else {
        bits = sign | (((exp >> 10) + 112u) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

static const uint16_t* SW(const SFile* sf, const char* name, uint32_t want_ndim) {
    for (size_t i = 0; i < sf->n; i++)
        if (!strcmp(sf->entries[i].name, name)) {
            if (sf->entries[i].ndim != want_ndim) {
                fprintf(stderr, "tensor %s ndim %u != %u\n", name,
                        sf->entries[i].ndim, want_ndim);
                exit(1);
            }
            return (const uint16_t*)(sf->base + 8 + sf->hlen + sf->entries[i].start);
        }
    fprintf(stderr, "missing tensor %s\n", name);
    exit(1);
}

/* ---------------- timing ---------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------- thread pool (parallel-for) ---------------- */

static int g_nthreads = 8;

typedef struct { int lo, hi; void (*fn)(int, int, void*); void* arg; } PJob;

static void* pworker(void* p) {
    PJob* j = (PJob*)p;
    j->fn(j->lo, j->hi, j->arg);
    return NULL;
}

static void par_for(int n, void (*fn)(int, int, void*), void* arg) {
    if (g_nthreads <= 1 || n < 2 * g_nthreads) { fn(0, n, arg); return; }
    int nt = g_nthreads;
    pthread_t th[64];
    PJob jobs[64];
    if (nt > 64) nt = 64;
    int chunk = (n + nt - 1) / nt;
    int started = 0;
    for (int t = 0; t < nt; t++) {
        int lo = t * chunk, hi = lo + chunk;
        if (hi > n) hi = n;
        if (lo >= hi) break;
        if (t == nt - 1 || t == 63) { fn(lo, hi, arg); break; }
        jobs[t].lo = lo; jobs[t].hi = hi; jobs[t].fn = fn; jobs[t].arg = arg;
        if (pthread_create(&th[t], NULL, pworker, &jobs[t]) != 0) fn(lo, hi, arg);
        else started++;
    }
    for (int t = 0; t < started; t++) pthread_join(th[t], NULL);
}

/* ---------------- F16 GEMV (NEON, rows parallelized) ----------------
 * Weight [nout, nin] row-major F16 (safetensors layout), x f32, bias F16. */

typedef struct {
    const uint16_t* w;
    const uint16_t* b;
    int nin, nout;
    const float* x;
    float* y;
} GemvCtx;

static void gemv_rows(int lo, int hi, void* arg) {
    GemvCtx* c = (GemvCtx*)arg;
    const float* x = c->x;
    int nin = c->nin;
    for (int o = lo; o < hi; o++) {
        const uint16_t* wr = c->w + (size_t)o * nin;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= nin; i += 8) {
            float16x8_t hf = vreinterpretq_f16_u16(vld1q_u16(wr + i));
            float32x4_t wlo = vcvt_f32_f16(vget_low_f16(hf));
            float32x4_t whi = vcvt_f32_f16(vget_high_f16(hf));
            acc = vfmaq_f32(acc, wlo, vld1q_f32(x + i));
            acc = vfmaq_f32(acc, whi, vld1q_f32(x + i + 4));
        }
        float s = vaddvq_f32(acc);
        for (; i < nin; i++) s += f16v(wr[i]) * x[i];
        c->y[o] = s + (c->b ? f16v(c->b[o]) : 0.0f);
    }
}

static void f16_gemv(const SFile* sf, const char* wname, const char* bname,
                     int nin, int nout, const float* x, float* y) {
    GemvCtx c;
    c.w = SW(sf, wname, 2);
    if (bname) {   /* bias name: same prefix with .weight swapped for .bias */
        char bn[256];
        size_t l = strlen(wname);
        memcpy(bn, wname, l + 1);
        memcpy(bn + l - 6, "bias", 5);
        c.b = SW(sf, bn, 1);
    } else c.b = NULL;
    c.nin = nin; c.nout = nout; c.x = x; c.y = y;
    par_for(nout, gemv_rows, &c);
}

/* batched over tokens: x [S, nin], y [S, nout]; parallel over output rows,
 * each row loops all tokens (weights streamed once per row — mmap friendly) */
typedef struct {
    const uint16_t* w;
    const uint16_t* b;
    int nin, nout;
    long S;
    const float* x;
    float* y;
} GemvBatchCtx;

static void gemv_batch_rows(int lo, int hi, void* arg) {
    GemvBatchCtx* c = (GemvBatchCtx*)arg;
    long S = c->S;
    int nin = c->nin;
    for (int o = lo; o < hi; o++) {
        const uint16_t* wr = c->w + (size_t)o * nin;
        float bias = c->b ? f16v(c->b[o]) : 0.0f;
        for (long s = 0; s < S; s++) {
            const float* x = c->x + s * nin;
            float32x4_t acc = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 8 <= nin; i += 8) {
                float16x8_t hf = vreinterpretq_f16_u16(vld1q_u16(wr + i));
                acc = vfmaq_f32(acc, vcvt_f32_f16(vget_low_f16(hf)), vld1q_f32(x + i));
                acc = vfmaq_f32(acc, vcvt_f32_f16(vget_high_f16(hf)), vld1q_f32(x + i + 4));
            }
            float sum = vaddvq_f32(acc);
            for (; i < nin; i++) sum += f16v(wr[i]) * x[i];
            c->y[(size_t)s * c->nout + o] = sum + bias;
        }
    }
}

static void f16_gemv_batch(const SFile* sf, const char* wname,
                           int nin, int nout, long S,
                           const float* x, float* y) {
    char bn[256];
    size_t l = strlen(wname);
    memcpy(bn, wname, l + 1);
    memcpy(bn + l - 6, "bias", 5);
    GemvBatchCtx c;
    c.w = SW(sf, wname, 2);
    c.b = SW(sf, bn, 1);
    c.nin = nin; c.nout = nout; c.S = S; c.x = x; c.y = y;
    par_for(nout, gemv_batch_rows, &c);
}

static void f16_vec(const SFile* sf, const char* name, int n, float* out) {
    const uint16_t* p = SW(sf, name, 1);
    for (int i = 0; i < n; i++) out[i] = f16v(p[i]);
}

/* ---------------- math helpers ---------------- */

static void rmsnorm_affine(const float* x, float* o, const float* w,
                           long rows, int n, float eps) {
    for (long r = 0; r < rows; r++) {
        const float* xr = x + r * n;
        float* or_ = o + r * n;
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
        float sc = (float)(1.0 / sqrt(ss / n + eps));
        for (int i = 0; i < n; i++) or_[i] = xr[i] * sc * w[i];
    }
}

static void layernorm_affine(const float* x, float* o, const float* w,
                             const float* b, long rows, int n, float eps) {
    for (long r = 0; r < rows; r++) {
        const float* xr = x + r * n;
        float* or_ = o + r * n;
        double sm = 0;
        for (int i = 0; i < n; i++) sm += xr[i];
        float mu = (float)(sm / n);
        double sv = 0;
        for (int i = 0; i < n; i++) { float d = xr[i] - mu; sv += (double)d * d; }
        float inv = (float)(1.0 / sqrt(sv / n + eps));
        for (int i = 0; i < n; i++) or_[i] = (xr[i] - mu) * inv * w[i] + b[i];
    }
}

/* ---------------- attention ---------------- */

typedef struct {
    int S;
    const float* qkv;      /* [S, 6144] head h at [h*192 .. +192): q,k,v */
    const float* rc;       /* [S, ROT_PAIRS] cos */
    const float* rs;       /* [S, ROT_PAIRS] sin */
    float* out;            /* [S, 2048] */
    /* per-thread scratch */
    float* qb; float* kb; float* vb; float* sc;
} AttCtx;

/* per-head q/k rmsnorm (no affine, eps 1e-5) + split-half rope on first 48 */
static void norm_rope64(const float* src, float* dst, const float* rc,
                        const float* rs) {
    double ss = 0;
    for (int i = 0; i < HDIM; i++) ss += (double)src[i] * src[i];
    float inv = (float)(1.0 / sqrt(ss / HDIM + 1e-5));
    float t[HDIM];
    for (int i = 0; i < HDIM; i++) t[i] = src[i] * inv;
    for (int j = 0; j < ROT_PAIRS; j++) {
        float a = t[j], b = t[ROT_PAIRS + j];
        dst[j] = rc[j] * a - rs[j] * b;
        dst[ROT_PAIRS + j] = rs[j] * a + rc[j] * b;
    }
    memcpy(dst + 2 * ROT_PAIRS, t + 2 * ROT_PAIRS,
           (HDIM - 2 * ROT_PAIRS) * sizeof(float));
}

static void att_heads(int lo, int hi, void* arg) {
    AttCtx* c = (AttCtx*)arg;
    int S = c->S;
    for (int h = lo; h < hi; h++) {
        for (int a = 0; a < S; a++) {
            const float* r = c->qkv + (size_t)a * (3 * DIM) + h * (3 * HDIM);
            norm_rope64(r, c->qb + (size_t)a * HDIM,
                        c->rc + (size_t)a * ROT_PAIRS, c->rs + (size_t)a * ROT_PAIRS);
            norm_rope64(r + HDIM, c->kb + (size_t)a * HDIM,
                        c->rc + (size_t)a * ROT_PAIRS, c->rs + (size_t)a * ROT_PAIRS);
            memcpy(c->vb + (size_t)a * HDIM, r + 2 * HDIM, HDIM * sizeof(float));
        }
        for (int a = 0; a < S; a++) {
            const float* qa = c->qb + (size_t)a * HDIM;
            float mx = -1e30f;
            for (int b = 0; b < S; b++) {
                const float* kbp = c->kb + (size_t)b * HDIM;
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int i = 0; i < HDIM; i += 4)
                    acc = vfmaq_f32(acc, vld1q_f32(qa + i), vld1q_f32(kbp + i));
                float d = vaddvq_f32(acc) * 0.125f;   /* 1/sqrt(64) */
                c->sc[b] = d;
                if (d > mx) mx = d;
            }
            double sum = 0;
            for (int b = 0; b < S; b++) {
                float e = expf(c->sc[b] - mx);
                c->sc[b] = e; sum += e;
            }
            float invs = (float)(1.0 / sum);
            float oh[HDIM];
            memset(oh, 0, sizeof(oh));
            for (int b = 0; b < S; b++) {
                float pb = c->sc[b] * invs;
                const float* vbp = c->vb + (size_t)b * HDIM;
                for (int i = 0; i < HDIM; i++) oh[i] += pb * vbp[i];
            }
            memcpy(c->out + (size_t)a * DIM + h * HDIM, oh, HDIM * sizeof(float));
        }
    }
}

/* ---------------- block scratch ---------------- */

typedef struct {
    int S;
    float *hn;       /* [S, DIM] pre-normed */
    float *qkv;      /* [S, 6144] */
    float *attn;     /* [S, DIM] */
    float *ff1;      /* [S, 16384] */
    float *ffact;    /* [S, 8192] */
    float *ffout;    /* [S, DIM] */
    float norm1[DIM], norm2[DIM], scale1[DIM], scale2[DIM];
    AttCtx actx;
} BlockScratch;

static void thread_scratch_init(AttCtx* a, int S) {
    int nt = g_nthreads;
    a->qb = malloc(sizeof(float) * (size_t)nt * S * HDIM);
    a->kb = malloc(sizeof(float) * (size_t)nt * S * HDIM);
    a->vb = malloc(sizeof(float) * (size_t)nt * S * HDIM);
    a->sc = malloc(sizeof(float) * (size_t)nt * S);
}
/* par_for over heads shares one scratch per thread — restructure: pass
 * per-thread offsets. Simplest: head range maps 1:1 to thread slots. */

static void att_heads_tls(int lo, int hi, void* arg) {
    AttCtx* c = (AttCtx*)arg;
    /* each par_for worker gets a contiguous head range; derive its slot
     * from the range start (chunked evenly by par_for) */
    int nt = g_nthreads;
    int chunk = (HEADS + nt - 1) / nt;
    int slot = lo / chunk;
    if (slot >= nt) slot = nt - 1;
    AttCtx t = *c;
    t.qb = c->qb + (size_t)slot * c->S * HDIM;
    t.kb = c->kb + (size_t)slot * c->S * HDIM;
    t.vb = c->vb + (size_t)slot * c->S * HDIM;
    t.sc = c->sc + (size_t)slot * c->S;
    att_heads(lo, hi, &t);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <vae.safetensors> <patch_latent.bin> "
                        "<vt> <lat_h> <lat_w> [outdir=/tmp/h3_frames]\n", argv[0]);
        return 1;
    }
    const char* vae_path = argv[1];
    const char* x_path = argv[2];
    int vt = atoi(argv[3]);
    int lh = atoi(argv[4]);
    int lw = atoi(argv[5]);
    const char* outdir = argc > 6 ? argv[6] : "/tmp/h3_frames";
    {
        const char* e = getenv("STRATUM_VAE_THREADS");
        if (e) g_nthreads = atoi(e);
        if (g_nthreads < 1) g_nthreads = 1;
    }

    double t0 = now_s();
    SFile sf;
    if (sopen_(vae_path, &sf) != 0) {
        fprintf(stderr, "open %s failed\n", vae_path);
        return 1;
    }
    printf("vae: %s (%zu tensors, mmap)\n", vae_path, sf.n);

    /* --- unpack patch rows -> full latent [24][vt][lh][lw], unnormalize --- */
    int nh = lh / 2, nw = lw / 2;
    long n_video = (long)vt * nh * nw;
    float* prow_in = malloc(sizeof(float) * (size_t)n_video * 96);
    {
        FILE* f = fopen(x_path, "rb");
        if (!f || fread(prow_in, 4, (size_t)n_video * 96, f) != (size_t)n_video * 96) {
            fprintf(stderr, "patch latent %s unreadable (need %ld rows x 96)\n",
                    x_path, n_video);
            return 1;
        }
        fclose(f);
    }
    float lm[LATENTS_DIM], ls[LATENTS_DIM];
    f16_vec(&sf, "latents_mean", LATENTS_DIM, lm);
    f16_vec(&sf, "latents_std", LATENTS_DIM, ls);
    size_t lsz = (size_t)LATENTS_DIM * vt * lh * lw;
    float* lat = malloc(sizeof(float) * lsz);
    {
        long rr = 0;
        for (int t = 0; t < vt; t++)
            for (int hh = 0; hh < nh; hh++)
                for (int ww = 0; ww < nw; ww++) {
                    for (int c = 0; c < LATENTS_DIM; c++)
                        for (int b = 0; b < 2; b++)
                            for (int a = 0; a < 2; a++)
                                lat[(((size_t)c * vt + t) * lh + (2 * hh + b)) * lw
                                    + (2 * ww + a)] =
                                    prow_in[rr * 96 + c * 4 + b * 2 + a];
                    rr++;
                }
    }
    free(prow_in);
    for (size_t i = 0; i < lsz; i++) {
        int c = (int)(i / ((size_t)vt * lh * lw));
        lat[i] = lat[i] * ls[c] + lm[c];
    }
    double smin = 1e30, smax = -1e30;
    for (size_t i = 0; i < lsz; i++) {
        if (lat[i] < smin) smin = lat[i];
        if (lat[i] > smax) smax = lat[i];
    }
    printf("latent (unnormalized): n=%zu range [%+.4f, %+.4f]\n", lsz, smin, smax);

    /* --- post_quant_conv: 1x1x1 conv 24->24 --- */
    float* lat2 = malloc(sizeof(float) * lsz);
    {
        const uint16_t* w = SW(&sf, "post_quant_conv.weight", 5);
        const uint16_t* b = SW(&sf, "post_quant_conv.bias", 1);
        for (int t = 0; t < vt; t++)
            for (int y = 0; y < lh; y++)
                for (int x = 0; x < lw; x++)
                    for (int o = 0; o < LATENTS_DIM; o++) {
                        float s = f16v(b[o]);
                        for (int i = 0; i < LATENTS_DIM; i++)
                            s += f16v(w[o * LATENTS_DIM + i])
                                 * lat[(((size_t)i * vt + t) * lh + y) * lw + x];
                        lat2[(((size_t)o * vt + t) * lh + y) * lw + x] = s;
                    }
    }
    free(lat);

    /* --- x_embedder + register/zero suffix tokens --- */
    int np = vt * lh * lw;
    int S = np + NREG + 1;
    printf("tokens: patches=%d S=%d (grid t=%d h=%d w=%d)\n", np, S, vt, lh, lw);
    float* h = calloc((size_t)S * DIM, sizeof(float));
    {
        const uint16_t* w = SW(&sf, "decoder.x_embedder.weight", 2);
        const uint16_t* b = SW(&sf, "decoder.x_embedder.bias", 1);
        for (int s = 0; s < np; s++) {
            float xin[LATENTS_DIM];
            int t = s / (lh * lw), rem = s % (lh * lw);
            int y = rem / lw, x = rem % lw;
            for (int c = 0; c < LATENTS_DIM; c++)
                xin[c] = lat2[(((size_t)c * vt + t) * lh + y) * lw + x];
            float32x4_t acc = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 8 <= LATENTS_DIM; i += 8) { /* 24 = 3*8 */
                float16x8_t hf = vreinterpretq_f16_u16(vld1q_u16(w + i * DIM)); /* unused */
                (void)hf;
                break; /* 24 wide: scalar below for clarity */
            }
            float* orow = h + (size_t)s * DIM;
            for (int o = 0; o < DIM; o++) {
                float sum = f16v(b[o]);
                const uint16_t* wr = w + (size_t)o * LATENTS_DIM;
                for (int c = 0; c < LATENTS_DIM; c++) sum += f16v(wr[c]) * xin[c];
                orow[o] = sum;
            }
        }
        const uint16_t* reg = SW(&sf, "decoder.register_tokens", 3);
        for (int k = 0; k < NREG; k++) {
            float* orow = h + (size_t)(np + k) * DIM;
            for (int o = 0; o < DIM; o++) orow[o] = f16v(reg[k * DIM + o]);
        }
        /* last token stays zero */
    }
    free(lat2);

    /* --- rope tables: rc/rs [S, ROT_PAIRS] --- */
    float* rc = malloc(sizeof(float) * (size_t)S * ROT_PAIRS);
    float* rs = malloc(sizeof(float) * (size_t)S * ROT_PAIRS);
    {
        double ct[4096], ch[4096], cw[4096];
        for (int i = 0; i < vt; i++) ct[i] = 2.0 * ((i + 0.5) / vt) - 1.0;
        for (int i = 0; i < lh; i++) ch[i] = 2.0 * ((i + 0.5) / lh) - 1.0;
        for (int i = 0; i < lw; i++) cw[i] = 2.0 * ((i + 0.5) / lw) - 1.0;
        double invf[ROT_ANG_FREQ];
        for (int i = 0; i < ROT_ANG_FREQ; i++)
            invf[i] = pow(100.0, -(2.0 * 3.0 * i) / (double)(2 * ROT_PAIRS));
        for (int s = 0; s < S; s++) {
            double coord[3] = { 0, 0, 0 };
            if (s < np) {
                int t = s / (lh * lw), rem = s % (lh * lw);
                coord[0] = ct[t]; coord[1] = ch[rem / lw]; coord[2] = cw[rem % lw];
            }
            for (int j = 0; j < ROT_PAIRS; j++) {
                int axis = j / ROT_ANG_FREQ, fi = j % ROT_ANG_FREQ;
                double ang = 2.0 * M_PI * coord[axis] * invf[fi];
                rc[(size_t)s * ROT_PAIRS + j] = (float)cos(ang);
                rs[(size_t)s * ROT_PAIRS + j] = (float)sin(ang);
            }
        }
    }

    /* --- 36 transformer blocks --- */
    BlockScratch B;
    memset(&B, 0, sizeof(B));
    B.S = S;
    B.hn = malloc(sizeof(float) * (size_t)S * DIM);
    B.qkv = malloc(sizeof(float) * (size_t)S * 3 * DIM);
    B.attn = malloc(sizeof(float) * (size_t)S * DIM);
    B.ff1 = malloc(sizeof(float) * (size_t)S * 2 * DIM * 4);
    B.ffact = malloc(sizeof(float) * (size_t)S * DIM * 4);
    B.ffout = malloc(sizeof(float) * (size_t)S * DIM);
    B.actx.S = S;
    B.actx.qkv = B.qkv;
    B.actx.rc = rc; B.actx.rs = rs;
    B.actx.out = B.attn;
    thread_scratch_init(&B.actx, S);

    char nm[256];
    double t_blk = now_s();
    for (int li = 0; li < NLAYERS; li++) {
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.norm1.weight", li);
        f16_vec(&sf, nm, DIM, B.norm1);
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.norm2.weight", li);
        f16_vec(&sf, nm, DIM, B.norm2);
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.scale1", li);
        f16_vec(&sf, nm, DIM, B.scale1);
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.scale2", li);
        f16_vec(&sf, nm, DIM, B.scale2);

        /* attention branch */
        rmsnorm_affine(h, B.hn, B.norm1, S, DIM, 1e-5f);
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.attn.to_qkv.weight", li);
        f16_gemv_batch(&sf, nm, DIM, 3 * DIM, S, B.hn, B.qkv);
        par_for(HEADS, att_heads_tls, &B.actx);
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.attn.to_out.weight", li);
        f16_gemv_batch(&sf, nm, DIM, DIM, S, B.attn, B.ffout);
        for (long i = 0; i < (long)S * DIM; i++)
            h[i] += B.ffout[i] * B.scale1[i % DIM];

        /* ffn branch */
        rmsnorm_affine(h, B.hn, B.norm2, S, DIM, 1e-5f);
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.ff.w1.weight", li);
        f16_gemv_batch(&sf, nm, DIM, 2 * DIM * 4, S, B.hn, B.ff1);
        for (long s = 0; s < S; s++)
            for (int i = 0; i < DIM * 4; i++) {
                float g = B.ff1[s * 2 * DIM * 4 + i];
                B.ffact[s * DIM * 4 + i] =
                    (g / (1.0f + expf(-g))) * B.ff1[s * 2 * DIM * 4 + DIM * 4 + i];
            }
        snprintf(nm, sizeof nm, "decoder.transformer_blocks.%d.ff.w2.weight", li);
        f16_gemv_batch(&sf, nm, DIM * 4, DIM, S, B.ffact, B.ffout);
        for (long i = 0; i < (long)S * DIM; i++)
            h[i] += B.ffout[i] * B.scale2[i % DIM];

        if (li == 0 || li == NLAYERS - 1) {
            double ss = 0;
            for (long i = 0; i < (long)S * DIM; i++) ss += (double)h[i] * h[i];
            printf("block %2d done, |h| rms=%.4f  (%.1fs)\n", li,
                   sqrt(ss / ((long)S * DIM)), now_s() - t_blk);
        }
    }

    /* --- final norm + proj_out + unpatchify --- */
    {
        float nw_[DIM], nb_[DIM];
        f16_vec(&sf, "decoder.norm_out.weight", DIM, nw_);
        f16_vec(&sf, "decoder.norm_out.bias", DIM, nb_);
        float* ln = malloc(sizeof(float) * (size_t)S * DIM);
        layernorm_affine(h, ln, nw_, nb_, S, DIM, 1e-5f);
        float* pr = malloc(sizeof(float) * (size_t)S * PROJ_OUT);
        f16_gemv_batch(&sf, "decoder.proj_out.weight",
                       DIM, PROJ_OUT, S, ln, pr);
        free(ln); free(h);

        int T = vt * 4, H = lh * 16, Wd = lw * 16;
        size_t nsz = (size_t)3 * T * H * Wd;
        unsigned char* px = malloc(nsz);
        for (int s = 0; s < np; s++) {
            int t = s / (lh * lw), rem = s % (lh * lw);
            int y = rem / lw, x = rem % lw;
            const float* row = pr + (size_t)s * PROJ_OUT;
            for (int c = 0; c < 3; c++)
                for (int pt = 0; pt < 4; pt++)
                    for (int ph = 0; ph < 16; ph++)
                        for (int pw = 0; pw < 16; pw++) {
                            float v = row[c * 1024 + pt * 256 + ph * 16 + pw];
                            v = v * PIXEL_STD[c] + PIXEL_MEAN[c];
                            if (v < 0) v = 0;
                            if (v > 1) v = 1;
                            px[((size_t)(c * T + t * 4 + pt) * H + y * 16 + ph) * Wd
                               + x * 16 + pw] = (unsigned char)(v * 255.0f + 0.5f);
                        }
        }
        free(pr);

        char path[512];
        snprintf(path, sizeof path, "mkdir -p %s", outdir);
        if (system(path) != 0) { /* best effort */ }
        for (int f = 0; f < T; f++) {
            snprintf(path, sizeof path, "%s/frame_%03d.ppm", outdir, f);
            FILE* o = fopen(path, "wb");
            if (!o) { fprintf(stderr, "cannot write %s\n", path); return 1; }
            fprintf(o, "P6\n%d %d\n255\n", Wd, H);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < Wd; x++)
                    for (int c = 0; c < 3; c++)
                        fputc(px[(((size_t)c * T + f) * H + y) * Wd + x], o);
            fclose(o);
            double sm2 = 0; long n2 = (size_t)H * Wd;
            unsigned char* fp = px + (size_t)f * H * Wd * 3;
            for (long i = 0; i < n2 * 3; i++) sm2 += fp[i];
            printf("frame %d/%d -> %s  mean=%u/255  %dx%d\n", f + 1, T, path,
                   (unsigned)(sm2 / (n2 * 3)), Wd, H);
        }
        free(px);
    }

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    /* anonymous (non-reclaimable) footprint via task_vm_info — the number the
     * memory boundary cares about; resident includes the mmap'd weight pages
     * which are reclaimable page cache */
    mach_port_t task = mach_task_self();
    struct task_vm_info tvi;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    double anon_mb = -1;
    if (task_info(task, TASK_VM_INFO, (task_info_t)&tvi, &cnt) == KERN_SUCCESS)
        anon_mb = (double)tvi.phys_footprint / (1024.0 * 1024.0);
    /* phys_footprint includes file-backed; anonymous-only estimate:
       ledger entries aren't exported — fall back to malloc zones via
       phys_footprint - (resident - internal) is unreliable; report both numbers */
    (void)anon_mb;
    printf("done in %.1fs\n", now_s() - t0);
    printf("memory: resident-peak %.0f MB (includes touched weight pages — those are "
           "reclaimable page cache), phys_footprint %.0f MB\n",
           ru.ru_maxrss / (1024.0 * 1024.0), anon_mb);
    return 0;
}
