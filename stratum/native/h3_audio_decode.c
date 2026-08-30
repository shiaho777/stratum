/*
 * h3_audio_decode.c — H3 pipeline M5b: audio VAE decoder (BigVGAN).
 *
 * Semantics pinned from comfy/ldm/minimax/audio_vae.py:
 *   z [32, 2, T] (channel-major stereo, 40 latent frames/s) — each stereo
 *   channel decoded independently:
 *     z = z*latents_std + latents_mean
 *     x = dec_in_proj (1x1 conv 32->2048)
 *     BigVGAN: conv_pre (2048->1024, k7 pad3)
 *       7 upsample stages (rates 5,5,2,2,2,2,2 → ×800 total), each:
 *         ConvTranspose1d(ups[i]) then mean of 3 AMPBlock1 resblocks
 *         (k3 dilations 1,3,5; each conv followed by anti-aliased
 *         SnakeBeta: upsample x2 (kaiser sinc k12) -> snake ->
 *         downsample x2)
 *       activation_post (same anti-aliased SnakeBeta, ch=8)
 *     conv_post (8->1, k7, no bias), clamp [-1,1]
 *   → stereo waveform [2, T*800] at 32 kHz.
 *
 * Weights: minimax_h3_audio_vae_fp32.safetensors (F32, mmap).
 * Audio path input here: H3 audio-velocity patch rows [audio_t*2, 32]
 * (h3_forward H3_X_OUT companion dump /tmp/h3_full_out.bin audio half)
 * integrated over the same Euler trajectory as the video latent.
 *
 * Usage: h3_audio_decode <vae.safetensors> <audio_latent.bin> <audio_t>
 *                        [out.wav]
 *   audio_latent.bin: [audio_t*2, 32] f32 rows (row = (t, stereo c)),
 *   exactly the layout h3_forward writes for the audio segment.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <malloc/malloc.h>

#define LATENT_CH 32
#define DEC_IN 2048
#define UP0_CH 1024
#define NUPS 7
#define NRES 3              /* resblock_kernel_sizes = 3 */
#define NBLK 6              /* convs per AMPBlock1 (3 dilations x conv1+conv2) */
#define AA_KERNEL 12

static const int UP_RATES[NUPS] = { 5, 5, 2, 2, 2, 2, 2 };
static const int UP_KERNELS[NUPS] = { 9, 9, 4, 4, 4, 4, 4 };
static const int RB_DILATION[3] = { 1, 3, 5 };

/* ---------------- safetensors reader (F32) ---------------- */

typedef struct {
    char name[256];
    uint64_t start, end;
    uint32_t ndim;
    uint64_t dims[8];
} AEntry;

typedef struct {
    int fd;
    const uint8_t* base;
    size_t size;
    uint64_t hlen;
    AEntry* entries;
    size_t n;
} AFile;

static int aopen(const char* path, AFile* sf) {
    sf->fd = open(path, O_RDONLY);
    if (sf->fd < 0) return -1;
    struct stat st;
    if (fstat(sf->fd, &st) != 0) return -1;
    sf->size = (size_t)st.st_size;
    sf->base = mmap(NULL, sf->size, PROT_READ, MAP_SHARED, sf->fd, 0);
    if (sf->base == MAP_FAILED) return -1;
    memcpy(&sf->hlen, sf->base, 8);
    size_t cap = 256;
    sf->entries = malloc(sizeof(AEntry) * cap);
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
            sf->entries = realloc(sf->entries, sizeof(AEntry) * cap);
        }
        AEntry* e = &sf->entries[sf->n];
        size_t cpy = nlen < sizeof(e->name) - 1 ? nlen : sizeof(e->name) - 1;
        memcpy(e->name, name0, cpy); e->name[cpy] = 0;
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

static const float* AF(const AFile* sf, const char* name, uint32_t ndim) {
    for (size_t i = 0; i < sf->n; i++)
        if (!strcmp(sf->entries[i].name, name)) {
            if (sf->entries[i].ndim != ndim) {
                fprintf(stderr, "tensor %s ndim %u != %u\n", name,
                        sf->entries[i].ndim, ndim);
                exit(1);
            }
            return (const float*)(sf->base + 8 + sf->hlen + sf->entries[i].start);
        }
    fprintf(stderr, "missing tensor %s\n", name);
    exit(1);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------- conv1d (weight [out, in, k], x [in, L]) ---------------- */

static void conv1d(const float* w, const float* b, int in_c, int out_c,
                   int k, int pad, const float* x, long L, float* y) {
    for (int o = 0; o < out_c; o++) {
        const float* wo = w + (size_t)o * in_c * k;
        for (long t = 0; t < L; t++) {
            double s = b ? b[o] : 0.0;
            for (int i = 0; i < in_c; i++) {
                const float* xi = x + (size_t)i * L;
                const float* woi = wo + (size_t)i * k;
                for (int d = 0; d < k; d++) {
                    long ti = t + d - pad;
                    if (ti >= 0 && ti < L) s += woi[d] * xi[ti];
                }
            }
            y[(size_t)o * L + t] = (float)s;
        }
    }
}

/* dilated conv1d (AMPBlock): out[t] = sum w[i,d]*x[t + d*dilation - pad] */
static void conv1d_dil(const float* w, const float* b, int in_c, int out_c,
                       int k, int dilation, const float* x, long L, float* y) {
    int pad = (k * dilation - dilation) / 2;   /* get_padding(3, d) */
    for (int o = 0; o < out_c; o++) {
        const float* wo = w + (size_t)o * in_c * k;
        for (long t = 0; t < L; t++) {
            double s = b ? b[o] : 0.0;
            for (int i = 0; i < in_c; i++) {
                const float* xi = x + (size_t)i * L;
                const float* woi = wo + (size_t)i * k;
                for (int d = 0; d < k; d++) {
                    long ti = t + d * dilation - pad;
                    if (ti >= 0 && ti < L) s += woi[d] * xi[ti];
                }
            }
            y[(size_t)o * L + t] = (float)s;
        }
    }
}

/* ---------------- anti-aliased SnakeBeta activation ---------------- */

static double sinc(double x);
static void kaiser_sinc(double cutoff, double half_width, int kernel_size,
                        float* filt) {
    int even = kernel_size % 2 == 0;
    int half_size = kernel_size / 2;
    double delta_f = 4 * half_width;
    double A = 2.285 * (half_size - 1) * M_PI * delta_f + 7.95;
    double beta;
    if (A > 50.0) beta = 0.1102 * (A - 8.7);
    else if (A >= 21.0) beta = 0.5842 * pow(A - 21, 0.4) + 0.07886 * (A - 21.0);
    else beta = 0.0;
    double win[64];
    double sum = 0;
    for (int n = 0; n < kernel_size; n++) {
        /* torch.kaiser_window(periodic=False) = Kaiser-Bessel with the
         * standard symmetric denominator */
        double r = (double)n / (kernel_size - 1) * 2 - 1;   /* [-1, 1] */
        double t = beta * sqrt(fmax(0.0, 1.0 - r * r));
        win[n] = t != 0 ? sinhf(t) / sinhf(beta) : 1.0;
    }
    for (int n = 0; n < kernel_size; n++) {
        double time = even ? (n - half_size + 0.5) : (n - half_size);
        double f = 2 * cutoff * win[n] * sinc(2 * cutoff * time);
        filt[n] = (float)f;
        sum += f;
    }
    for (int n = 0; n < kernel_size; n++) filt[n] /= (float)sum;
}
/* math.h doesn't expose sinc; define as the normalized sinc */
static double sinc(double x) {
    if (fabs(x) < 1e-12) return 1.0;
    return sin(M_PI * x) / (M_PI * x);
}

/* upsample x2 with stored filter, per channel.
 * Torch semantics: replicate-pad 5, conv_transpose stride 2 (k=12), .mul_(2),
 * then trim [15 : end-15] — kept window is torch t in [15, 15+2L).
 * Folded into a direct gather: x index = (t + 15 - d)/2 - 5 clamped to
 * [0, L) (replicate), taps with odd (t + 15 - d) skipped. Verified exactly
 * equal to the torch window on random inputs. */
static void upsample2x(const float* filt, int C, const float* x, long L,
                       float* y) {
    long Lo = L * 2;
    for (int c = 0; c < C; c++) {
        const float* xc = x + (size_t)c * L;
        float* yc = y + (size_t)c * Lo;
        for (long t = 0; t < Lo; t++) {
            double s = 0;
            for (int d = 0; d < AA_KERNEL; d++) {
                long sidx = t + 15 - d;
                if (sidx < 0 || (sidx & 1)) continue;
                long xi = sidx / 2 - 5;
                if (xi < 0) xi = 0;
                if (xi >= L) xi = L - 1;
                s += xc[xi] * filt[d];
            }
            yc[t] = (float)(s * 2.0);          /* .mul_(ratio) */
        }
    }
}

static void downsample2x(const float* filt, int C, const float* x, long L,
                         float* y) {
    int pad_left = AA_KERNEL / 2 - 1;
    int pad_right = AA_KERNEL / 2;
    long Lo = L / 2;
    for (int c = 0; c < C; c++) {
        const float* xc = x + (size_t)c * L;
        float* yc = y + (size_t)c * Lo;
        for (long t = 0; t < Lo; t++) {
            double s = 0;
            for (int d = 0; d < AA_KERNEL; d++) {
                long ti = t * 2 + d - pad_left;    /* stride 2 conv */
                float xv;
                if (ti < 0) ti = 0;
                if (ti >= L) ti = L - 1;
                xv = xc[ti];
                s += xv * filt[d];
            }
            yc[t] = (float)s;
        }
    }
}

/* SnakeBeta on x [C, L] with exp(alpha), exp(beta) per channel */
static void snake_beta(const float* alpha, const float* beta, int C,
                       const float* x, long L, float* y) {
    for (int c = 0; c < C; c++) {
        float a = expf(alpha[c]), b = 1.0f / (expf(beta[c]) + 1e-9f);
        const float* xc = x + (size_t)c * L;
        float* yc = y + (size_t)c * L;
        for (long t = 0; t < L; t++) {
            float s = sinf(a * xc[t]);
            yc[t] = xc[t] + s * s * b;
        }
    }
}

static void aa_snake(const float* upf, const float* downf,
                     const float* alpha, const float* beta, int C,
                     const float* x, long L, float* scratch, float* y) {
    long L2 = L * 2;
    float* up = scratch;               /* [C, L2] */
    float* act = scratch + (size_t)C * L2;   /* [C, L2] */
    upsample2x(upf, C, x, L, up);
    snake_beta(alpha, beta, C, up, L2, act);
    downsample2x(downf, C, act, L2, y);     /* [C, L] */
}

/* ---------------- AMPBlock1 ---------------- */

typedef struct {
    const AFile* sf;
    int ch;
    int j;                /* resblock index within stage: kernel 3/7/11 */
    char base[160];       /* e.g. "decoder.resblocks.0" */
    float *c1a, *c1b, *c2a, *c2b;  /* activation scratch */
    float *conv_out, *sum;
} AMP;

static const int RB_KERNEL[3] = { 3, 7, 11 };

static void amp_forward(AMP* a, const float* x, long L, float* out,
                        float* tmp1, float* tmp2) {
    for (long i = 0; i < (size_t)a->ch * L; i++) out[i] = x[i];
    for (int d = 0; d < NRES; d++) {
        char nm[256];
        /* activation 2d: aa_snake -> conv1(d) */
        snprintf(nm, sizeof nm, "%s.activations.%d.act.alpha", a->base, d * 2);
        const float* al = AF(a->sf, nm, 1);
        snprintf(nm, sizeof nm, "%s.activations.%d.act.beta", a->base, d * 2);
        const float* be = AF(a->sf, nm, 1);
        snprintf(nm, sizeof nm, "%s.activations.%d.upsample.filter", a->base, d * 2);
        const float* upf = AF(a->sf, nm, 3);
        snprintf(nm, sizeof nm, "%s.activations.%d.downsample.lowpass.filter", a->base, d * 2);
        const float* dnf = AF(a->sf, nm, 3);
        aa_snake(upf, dnf, al, be, a->ch, out, L, tmp1, tmp2);
        snprintf(nm, sizeof nm, "%s.convs1.%d.weight", a->base, d);
        const float* w1 = AF(a->sf, nm, 3);
        snprintf(nm, sizeof nm, "%s.convs1.%d.bias", a->base, d);
        const float* b1 = AF(a->sf, nm, 1);
        conv1d_dil(w1, b1, a->ch, a->ch, RB_KERNEL[a->j], RB_DILATION[d],
                   tmp2, L, tmp1);
        for (long i = 0; i < (size_t)a->ch * L; i++) out[i] += tmp1[i];
        /* activation 2d+1 -> conv2 (dilation 1) */
        snprintf(nm, sizeof nm, "%s.activations.%d.act.alpha", a->base, d * 2 + 1);
        al = AF(a->sf, nm, 1);
        snprintf(nm, sizeof nm, "%s.activations.%d.act.beta", a->base, d * 2 + 1);
        be = AF(a->sf, nm, 1);
        snprintf(nm, sizeof nm, "%s.activations.%d.upsample.filter", a->base, d * 2 + 1);
        upf = AF(a->sf, nm, 3);
        snprintf(nm, sizeof nm, "%s.activations.%d.downsample.lowpass.filter", a->base, d * 2 + 1);
        dnf = AF(a->sf, nm, 3);
        aa_snake(upf, dnf, al, be, a->ch, out, L, tmp1, tmp2);
        snprintf(nm, sizeof nm, "%s.convs2.%d.weight", a->base, d);
        const float* w2 = AF(a->sf, nm, 3);
        snprintf(nm, sizeof nm, "%s.convs2.%d.bias", a->base, d);
        const float* b2 = AF(a->sf, nm, 1);
        conv1d_dil(w2, b2, a->ch, a->ch, RB_KERNEL[a->j], 1, tmp2, L, tmp1);
        for (long i = 0; i < (size_t)a->ch * L; i++) out[i] += tmp1[i];
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <audio_vae.safetensors> <audio_latent.bin> "
                        "<audio_t> [out.wav]\n", argv[0]);
        return 1;
    }
    double t0 = now_s();
    AFile sf;
    if (aopen(argv[1], &sf) != 0) {
        fprintf(stderr, "open %s failed\n", argv[1]);
        return 1;
    }
    int audio_t = atoi(argv[3]);
    const char* outw = argc > 4 ? argv[4] : "/tmp/h3_audio.wav";

    /* input rows [audio_t*2, 32] — the audio-velocity Euler-updated state */
    long T = audio_t;
    float zraw[LATENT_CH * 2 * 4096];
    size_t need = (size_t)T * 2 * LATENT_CH;
    {
        FILE* f = fopen(argv[2], "rb");
        if (!f || fread(zraw, 4, need, f) != need) {
            fprintf(stderr, "audio latent unreadable (need %zu floats)\n", need);
            return 1;
        }
        fclose(f);
    }
    /* rows: h3_forward emits (t, c) with row = t*2 + c; decode wants [32, 2, T] */
    float z[LATENT_CH * 2 * 4096];   /* [c][s][t] */
    {
        const float* lm = AF(&sf, "latents_mean", 1);
        const float* ls = AF(&sf, "latents_std", 1);
        for (int t = 0; t < T; t++)
            for (int s = 0; s < 2; s++)
                for (int c = 0; c < LATENT_CH; c++) {
                    float v = zraw[(size_t)(t * 2 + s) * LATENT_CH + c];
                    z[((size_t)c * 2 + s) * T + t] = v * ls[c] + lm[c];
                }
    }
    fprintf(stderr, "audio latent: %ld frames x2 stereo, unnormalized\n", T);

    /* per stereo channel */
    long L = T * 800;   /* 40 latent fps, 800 samples per frame */
    float* wav = malloc(sizeof(float) * 2 * L);
    float* buf1 = malloc(sizeof(float) * DEC_IN * T);
    float* buf2 = malloc(sizeof(float) * UP0_CH * T);
    float* wA = malloc(sizeof(float) * UP0_CH * 2 * L);
    float* wB = malloc(sizeof(float) * UP0_CH * 2 * L);
    float* sc1 = malloc(sizeof(float) * (size_t)UP0_CH * 2 * L * 2);
    float* sc2 = malloc(sizeof(float) * (size_t)UP0_CH * 2 * L);

    for (int s = 0; s < 2; s++) {
        /* dec_in_proj: 1x1 conv 32->2048 */
        {
            const float* w = AF(&sf, "dec_in_proj.weight", 3);
            const float* b = AF(&sf, "dec_in_proj.bias", 1);
            for (int o = 0; o < DEC_IN; o++) {
                double acc = b[o];
                for (int c = 0; c < LATENT_CH; c++)
                    acc += w[o * LATENT_CH + c] * z[(size_t)c * 2 + s] * 0 /* placeholder */;
                buf1[(size_t)o * T] = (float)acc;
            }
            /* real gather across T */
            for (int o = 0; o < DEC_IN; o++)
                for (int t = 0; t < T; t++) {
                    double acc = b[o];
                    for (int c = 0; c < LATENT_CH; c++)
                        acc += w[o * LATENT_CH + c] * z[((size_t)c * 2 + s) * T + t];
                    buf1[(size_t)o * T + t] = (float)acc;
                }
        }
        /* BigVGAN conv_pre 2048->1024 k7 pad3 */
        {
            const float* w = AF(&sf, "decoder.conv_pre.weight", 3);
            const float* b = AF(&sf, "decoder.conv_pre.bias", 1);
            conv1d(w, b, DEC_IN, UP0_CH, 7, 3, buf1, T, buf2);
        }
        long cl = T;      /* current length at stage input */
        float* curp = buf2;
        float* nxt = wA;
        for (int u = 0; u < NUPS; u++) {
            int cin = UP0_CH >> u, cout = UP0_CH >> (u + 1);
            /* ConvTranspose1d k, stride u_rate, padding (k-u)//2 */
            int rate = UP_RATES[u], k = UP_KERNELS[u];
            int pad = (k - rate) / 2;
            long Lo = cl * rate;
            char nm[256];
            snprintf(nm, sizeof nm, "decoder.ups.%d.0.weight", u);
            const float* w = AF(&sf, nm, 3);
            snprintf(nm, sizeof nm, "decoder.ups.%d.0.bias", u);
            const float* bb = AF(&sf, nm, 1);
            for (int o = 0; o < cout; o++)
                memset(nxt + (size_t)o * Lo, 0, sizeof(float) * Lo);
            for (int o = 0; o < cout; o++)
                for (long t = 0; t < Lo; t++) {
                    /* y[t] = sum_{i,d} x[i, u]*w[o, i, d] with t = u*rate + d - pad,
                     * x zero-padded left/right by pad (ConvTranspose padding) */
                    double acc = bb[o];
                    for (int d = 0; d < k; d++) {
                        long ts = t - d + pad;
                        if (ts < 0 || ts % rate != 0) continue;
                        long uu = ts / rate;
                        if (uu < 0 || uu >= cl) continue;
                        for (int i = 0; i < cin; i++)
                            acc += (double)w[((size_t)i * cout + o) * k + d]
                                   * curp[(size_t)i * cl + uu];
                    }
                    nxt[(size_t)o * Lo + t] = (float)acc;
                }
            /* mean of 3 resblocks over nxt */
            {
                float* rs = malloc(sizeof(float) * (size_t)cout * Lo);
                float* tmp1 = malloc(sizeof(float) * (size_t)cout * Lo * 4);
                float* tmp2 = malloc(sizeof(float) * (size_t)cout * Lo);
                memset(rs, 0, sizeof(float) * (size_t)cout * Lo);
                for (int j = 0; j < NRES; j++) {
                    AMP a;
                    a.sf = &sf; a.ch = cout; a.j = j;
                    snprintf(a.base, sizeof a.base, "decoder.resblocks.%d", u * NRES + j);
                    float* ob = malloc(sizeof(float) * (size_t)cout * Lo);
                    amp_forward(&a, nxt, Lo, ob, tmp1, tmp2);
                    for (long i = 0; i < (size_t)cout * Lo; i++) rs[i] += ob[i];
                    free(ob);
                }
                for (long i = 0; i < (size_t)cout * Lo; i++) nxt[i] = rs[i] / NRES;
                free(rs); free(tmp1); free(tmp2);
            }
            { double ss2=0; for (long i=0;i<(size_t)cout*Lo;i++) ss2+=(double)nxt[i]*nxt[i];
              fprintf(stderr, "  ups %d: ch %d->%d, len %ld, rms %.5f\n", u, cin, cout, Lo,
                      sqrt(ss2/((size_t)cout*Lo))); }
            /* ping-pong: nxt becomes the next stage's input */
            float* nw = (nxt == wA) ? wB : wA;
            curp = nxt;
            nxt = nw;
            cl = Lo;
        }
        /* activation_post (ch=8) then conv_post 8->1 k7, clamp [-1,1] */
        {
            const float* al = AF(&sf, "decoder.activation_post.act.alpha", 1);
            const float* be = AF(&sf, "decoder.activation_post.act.beta", 1);
            const float* upf = AF(&sf, "decoder.activation_post.upsample.filter", 3);
            const float* dnf = AF(&sf, "decoder.activation_post.downsample.lowpass.filter", 3);
            aa_snake(upf, dnf, al, be, 8, curp, cl, sc1, nxt);
            const float* w = AF(&sf, "decoder.conv_post.weight", 3);
            conv1d(w, NULL, 8, 1, 7, 3, nxt, cl, wav + (size_t)s * L);
            for (long t = 0; t < L; t++) {
                float v = wav[(size_t)s * L + t];
                if (v > 1) v = 1;
                if (v < -1) v = -1;
                wav[(size_t)s * L + t] = v;
            }
        }
    }

    /* write WAV (16-bit PCM 32 kHz stereo) */
    {
        FILE* o = fopen(outw, "wb");
        uint32_t sr = 32000;
        uint32_t dsz = (uint32_t)(L * 2 * 2);
        uint16_t fmt = 1, ch = 2, bits = 16, align = 4;
        uint32_t rate = sr, byte_rate = sr * 4, fsize = 16, dsz32 = dsz, riff = dsz + 36;
        fwrite("RIFF", 1, 4, o);
        fwrite(&riff, 4, 1, o);
        fwrite("WAVEfmt ", 1, 8, o);
        fwrite(&fsize, 4, 1, o);
        fwrite(&fmt, 2, 1, o);
        fwrite(&ch, 2, 1, o);
        fwrite(&rate, 4, 1, o);
        fwrite(&byte_rate, 4, 1, o);
        fwrite(&align, 2, 1, o);
        fwrite(&bits, 2, 1, o);
        fwrite("data", 1, 4, o);
        fwrite(&dsz32, 4, 1, o);
        for (long t = 0; t < L; t++)
            for (int s = 0; s < 2; s++) {
                float v = wav[(size_t)s * L + t];
                int iv = (int)(v * 32767.0f);
                if (iv > 32767) iv = 32767;
                if (iv < -32768) iv = -32768;
                int16_t sv = (int16_t)iv;
                fwrite(&sv, 2, 1, o);
            }
        fclose(o);
        printf("audio: %.1f s stereo 32kHz -> %s\n", (double)L / 32000.0, outw);
    }
    printf("done in %.1fs\n", now_s() - t0);
    return 0;
}
