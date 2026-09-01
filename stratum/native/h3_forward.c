/*
 * h3_forward.c — H3 pipeline M3: packed forward through the denoiser.
 *
 * Assembles the full conditioning stream and runs ONE denoiser pass
 * (t=0) over all 50 blocks:
 *
 *   [text rows]     <- token_refiner(condition_proj(text_states)) from
 *                      a Q3TE0001 dump (te_qwen3vl_step output)
 *   [audio rows]    <- synthetic stereo latent -> audio_patch_proj
 *   [video rows]    <- synthetic video latent -> patchify -> video_
 *                      patch_proj, rows in (t, h, w) order
 *
 * Positions come from the h3_pack semantics (verified separately):
 * text t=0..L-1; audio (t, 0, w_low|w_high) per channel; video rows on
 * the FRAME_RESCALE cumsum grid with area-normalized (h, w).
 *
 * Per-block AdaLN rows by modality tag: video=0, text=1, audio=2 —
 * adaln_proj output is [3][6*HID] at M=1 (single timestep t=0).
 *
 * Denoiser op order per block (validated in h3_step.c): norm1 -> AdaLN
 * scale/shift -> qkv -> per-head QK-norm -> split-half rope over pairs
 * 0..47 with angle = pos[axis(j)] * inv_freq[j%16] -> bidirectional
 * attention -> gate_msa residual; norm2 -> AdaLN -> swiglu MLP ->
 * gate_mlp residual. RMSNorm gains and condition/token_refiner norms
 * are BF16; patch/proj heads are F32; adaln_proj is F16.
 *
 * Output: final hidden stream stats + "H3PKT001" dump of the video
 * segment (what a sampler would iterate on).
 */
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include "stratum_q6k.h"
#include "stratum_q4k_neon.h"
#include "stratum_q6k_neon.h"
#include <pthread.h>
#include <mach/mach_time.h>
#include <mach/mach.h>
#ifdef STRATUM_USE_METAL
#include "stratum_metal.h"
#endif
#include "stratum_q6k.h"

/* phys_footprint probe: the honest wired-proxy number (counts Metal shared
 * buffers + anon; excludes reclaimable page cache backing the weights). */
static void h3_footprint(const char* tag) {
    task_vm_info_data_t info;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &cnt)
            == KERN_SUCCESS)
        fprintf(stderr, "  [mem] %-14s phys_footprint=%.1fMB\n", tag,
                info.phys_footprint / (1024.0 * 1024.0));
}
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LATENTS_DIM 24
#define AUDIO_DIM 32
#define VIDEO_ROW_DIM (LATENTS_DIM * 4)
#define AUDIO_ROW_DIM AUDIO_DIM
#define FRAME_RESCALE (5.0 / 3.0)
static const int FRAME_PER_TOKEN[5] = {1, 4, 4, 4, 4};
#define ROT_PAIRS 48
#define MAX_HID 8192

static Gguf G;

static const void* T(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "missing '%s'\n", name); exit(1); }
    return (const void*)(G.mmap_base + t->offset);
}
static const GgufTensor* TT(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "missing '%s'\n", name); exit(1); }
    return t;
}

static inline float f16v(uint16_t h) { return q4k_fp16_to_fp32(h); }
static inline float bfv(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

static void act_stats(const char* tag, const float* x, long n) {
    float mx = 0; double sm = 0; long nan = 0;
    for (long i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (isnan(x[i]) || isinf(x[i])) { nan++; continue; }
        if (a > mx) mx = a;
        sm += a;
    }
    fprintf(stderr, "    %-12s |x|mean=%.6g max=%.6g%s\n", tag,
            nan ? 0.0 : sm / n, mx,
            nan ? (nan == n ? " ALL-NAN/INF" : " (nan contaminated)") : "");
}

/* mixed-quant gemv (Q4_K/Q6_K), rows [out][K/256] */
/* ---- threaded row-parallel batched gemv (rows split across threads) ---- */

static int g_nt = 8;
static int g_nt_init = 0;

typedef struct {
    const GgufTensor* t;
    int in_dim, out_dim;
    const float* x;       /* [S, in_dim] */
    float* y;             /* [S, out_dim] */
    long S;
    int8_t* xq;           /* SDOT: [S, in_dim] int8 prequant */
    float* xs;            /* SDOT: [S, in_dim/32] scales */
    int xstride, ystride; /* row strides in floats; 0 = compact */
} BGMVCtx;

static int ty4k(const GgufTensor* t) { return (GgmlType)t->type == GGML_TYPE_Q4_K; }
static int g_h3_sdot = -1;    /* STRATUM_H3_SDOT=1: int8 x-prequant (boundary-1
                               * approved approximation; verified token-greedy
                               * identical in the engine gates) */

static void bgmv_range(int lo, int hi, void* arg) {
    BGMVCtx* c = (BGMVCtx*)arg;
    const void* base = (const void*)(G.mmap_base + c->t->offset);
    GgmlType ty = (GgmlType)c->t->type;
    long xr = c->xstride ? c->xstride : c->in_dim;
    long yr = c->ystride ? c->ystride : c->out_dim;
    for (long s = 0; s < c->S; s++) {
        const float* x = c->x + s * xr;
        float* y = c->y + s * yr;
        if (ty == GGML_TYPE_Q4_K) {
            const block_q4_K* brow = (const block_q4_K*)base;
            if (g_h3_sdot) {
#if defined(__ARM_FEATURE_DOTPROD)
                int nb32 = c->in_dim / 32;
                int8_t* xq = c->xq + (size_t)s * c->in_dim;
                const float* xs = c->xs + (size_t)s * nb32;
                for (int r = lo; r < hi; r++)
                    y[r] = q4k_dot_row_sdot(brow + (size_t)r * (c->in_dim / 256),
                                            c->in_dim, xq, xs);
#endif
            } else {
                for (int r = lo; r < hi; r++)
                    y[r] = q4k_dot_row_neon(brow + (size_t)r * (c->in_dim / 256),
                                            c->in_dim, x);
            }
        } else { /* Q6_K */
            const block_q6_K* brow = (const block_q6_K*)base;
            for (int r = lo; r < hi; r++)
                y[r] = q6k_dot_row_neon(brow + (size_t)r * (c->in_dim / 256),
                                        c->in_dim, x);
        }
    }
}

typedef struct { int lo, hi; void (*fn)(int, int, void*); void* arg; } HJob;

static void* hworker(void* p) {
    HJob* j = (HJob*)p;
    j->fn(j->lo, j->hi, j->arg);
    return NULL;
}

static void h3_par_for(int n, void (*fn)(int, int, void*), void* arg) {
    if (!g_nt_init) {
        g_nt_init = 1;
        const char* e = getenv("H3_THREADS");
        if (e) { g_nt = atoi(e); if (g_nt < 1) g_nt = 1; if (g_nt > 32) g_nt = 32; }
    }
    if (g_nt <= 1 || n < 2 * g_nt) { fn(0, n, arg); return; }
    int nt = g_nt > 32 ? 32 : g_nt;
    pthread_t th[32];
    HJob jobs[32];
    int chunk = (n + nt - 1) / nt;
    int started = 0;
    for (int t = 0; t < nt; t++) {
        int lo = t * chunk, hi = lo + chunk;
        if (hi > n) hi = n;
        if (lo >= hi) break;
        if (t == nt - 1 || t == 31) { fn(lo, hi, arg); break; }
        jobs[t].lo = lo; jobs[t].hi = hi; jobs[t].fn = fn; jobs[t].arg = arg;
        if (pthread_create(&th[t], NULL, hworker, &jobs[t]) != 0) fn(lo, hi, arg);
        else started++;
    }
    for (int t = 0; t < started; t++) pthread_join(th[t], NULL);
}

/* batched: y[s] = W @ x[s] for all S token rows, rows parallelized.
 * STRATUM_H3_NC=1 routes the gemv through the engine's per-tensor NoCopy
 * Metal path (boundary 2a: per-tensor <100MB windows, never whole-model).
 * add() failure (-1) falls back to the CPU NEON path transparently. */
static int g_h3_nc = -1;          /* -1 = unset */
static int g_metal_ready = 0;

#ifdef STRATUM_USE_METAL
static void h3_metal_init_once(void) {
    const char* mlpath = getenv("STRATUM_METALLIB");
    if (!mlpath) mlpath = "stratum_q4k.metallib";
    if (stratum_metal_init(mlpath, NULL, 0) != 0) {
        fprintf(stderr, "  H3 NC: metal init failed, CPU fallback\n");
        return;
    }
    g_metal_ready = 1;
}
#endif

#ifndef STRATUM_USE_METAL
static int h3_nc_gemv(const GgufTensor* t, int in_dim, int out_dim,
                      long S, const float* x, float* y) { (void)t;(void)in_dim;(void)out_dim;(void)S;(void)x;(void)y; return -1; }
static void h3_metal_init_once(void) { }
#else
static int h3_nc_gemv(const GgufTensor* t, int in_dim, int out_dim,
                      long S, const float* x, float* y) {
    size_t nbytes = (size_t)t->nbytes;
    int rc = stratum_metal_nc_batch_add((const void*)(G.mmap_base + t->offset),
                                        nbytes, t->type, x, y,
                                        out_dim, in_dim, (int)S);
    return rc == 0 ? 0 : -1;   /* -2 (sync-executed) also counts as done */
}
#endif

static void mixed_gemv_batch(const GgufTensor* t, int in_dim, int out_dim,
                             long S, const float* x, float* y) {
    if (g_h3_nc < 0) {
        const char* e = getenv("STRATUM_H3_NC");
        g_h3_nc = e && atoi(e) ? 1 : 0;
        const char* sd = getenv("STRATUM_H3_SDOT");
        g_h3_sdot = sd && atoi(sd) ? 1 : 0;
    }
#ifdef STRATUM_USE_METAL
    if (g_h3_nc && g_metal_ready) {
        stratum_metal_nc_batch_begin();
        int rc = h3_nc_gemv(t, in_dim, out_dim, S, x, y);
        stratum_metal_nc_batch_flush();
        if (rc == 0) return;
    }
#endif
    BGMVCtx c = { t, in_dim, out_dim, x, y, S, NULL, NULL, 0, 0 };
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_h3_sdot > 0 && ty4k(t)) {
        int nb32 = in_dim / 32;
        c.xq = malloc(sizeof(int8_t) * (size_t)S * in_dim);
        c.xs = malloc(sizeof(float) * (size_t)S * nb32);
        for (long s = 0; s < S; s++)
            q4k_quantize_x_q8(x + s * in_dim, in_dim,
                              c.xq + s * in_dim, c.xs + s * nb32);
    }
#endif
    h3_par_for(out_dim, bgmv_range, &c);
    free(c.xq); free(c.xs);
}

/* strided gemv: x/y rows inside wider caller buffers (H3 fused fbuf).
 * Metal: nc_batch_add_strided (staged x copy-in, staged y + scatter-out).
 * CPU: same row-parallel kernel with strided base pointers. */
static void mixed_gemv_batch_strided(const GgufTensor* t, int in_dim, int out_dim,
                                     long S, const float* x, float* y,
                                     int xstride, int ystride) {
    if (g_h3_nc < 0) {
        const char* e = getenv("STRATUM_H3_NC");
        g_h3_nc = e && atoi(e) ? 1 : 0;
        const char* sd = getenv("STRATUM_H3_SDOT");
        g_h3_sdot = sd && atoi(sd) ? 1 : 0;
    }
#ifdef STRATUM_USE_METAL
    if (g_h3_nc && g_metal_ready) {
        stratum_metal_nc_batch_begin();
        size_t nbytes = (size_t)t->nbytes;
        int rc = stratum_metal_nc_batch_add_strided(
            (const void*)(G.mmap_base + t->offset), nbytes, t->type,
            x, y, out_dim, in_dim, (int)S, xstride, ystride);
        stratum_metal_nc_batch_flush();
        if (rc == 0 || rc == -2) return;
    }
#endif
    BGMVCtx c = { t, in_dim, out_dim, x, y, S, NULL, NULL, xstride, ystride };
    h3_par_for(out_dim, bgmv_range, &c);
}

static double h3_now_s(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

/* H3_PROFILE=1: per-stage wall-clock accumulation across the 50 blocks.
 * Reveals where a step actually spends time (gemv vs attention vs norms
 * vs copies) without changing any numerics. */
static int g_h3_prof = -1;
static double g_prof[8];
static const char* g_prof_nm[8] = {
    "norm1+adaln", "qkv gemv", "qknorm+rope", "attention",
    "out_proj gemv", "norm2", "fc1 gemv", "fc2+resid"
};
static inline void prof_on(void) {
    if (g_h3_prof < 0) g_h3_prof = getenv("H3_PROFILE") ? 1 : 0;
}
#define PROF_ENTER(do_reset) \
    double h3p_t0 = 0; int h3p_fresh = 1; \
    if (g_h3_prof) { h3p_t0 = h3_now_s(); h3p_fresh = 0; } \
    (void)h3p_t0; (void)h3p_fresh; (void)do_reset
#define PROF_EXIT(slot) \
    if (g_h3_prof) g_prof[slot] += h3_now_s() - h3p_t0
#define PROF_BEGINSLOT(slot) \
    if (g_h3_prof) { h3p_t0 = h3_now_s(); h3p_fresh = 0; }
#define PROF_ENDSLOT(slot) \
    if (g_h3_prof) { g_prof[slot] += h3_now_s() - h3p_t0; h3p_fresh = 1; }

static void mixed_gemv(const GgufTensor* t, int in_dim, int out_dim,
                       const float* x, float* y) {
    int nbpr = in_dim / 256;
    float tmp[256];
    const void* base = (const void*)(G.mmap_base + t->offset);
    for (int r = 0; r < out_dim; r++) {
        double acc = 0;
        if ((GgmlType)t->type == GGML_TYPE_Q4_K) {
            const block_q4_K* row = (const block_q4_K*)base + (size_t)r * nbpr;
            for (int nb = 0; nb < nbpr; nb++) {
                q4k_dequant_block_scalar(&row[nb], tmp);
                for (int c = 0; c < 256; c++)
                    acc += (double)tmp[c] * x[nb*256+c];
            }
        } else if ((GgmlType)t->type == GGML_TYPE_Q6_K) {
            const block_q6_K* row = (const block_q6_K*)base + (size_t)r * nbpr;
            for (int nb = 0; nb < nbpr; nb++) {
                q6k_dequant_block_scalar(&row[nb], tmp);
                for (int c = 0; c < 256; c++)
                    acc += (double)tmp[c] * x[nb*256+c];
            }
        } else { fprintf(stderr, "bad quant %u\n", t->type); exit(1); }
        y[r] = (float)acc;
    }
}

/* F32 gemv [in, out] row-major (ne[0]=in contiguous, rows = out) */
static void f32_gemv(const void* w, int in_dim, int out_dim,
                     const float* x, float* y) {
    const float* wr = (const float*)w;
    for (int r = 0; r < out_dim; r++) {
        const float* row = wr + (size_t)r * in_dim;
        double acc = 0;
        for (int c = 0; c < in_dim; c++) acc += (double)row[c] * x[c];
        y[r] = (float)acc;
    }
}

static double g_sigma_v = 0.0;   /* M4 driver sets these per step */
static double g_sigma_a = 0.0;

/* page-aligned allocation for NC zero-copy x/y windows (needs page-rounded
 * size; used by stream + block scratch buffers) */
#define H3_ALIGNED_FLOATS(n) ((float*)aligned_alloc(4096, \
    (((size_t)(n) * sizeof(float) + 4095) / 4096) * 4096))

static double env_sigma_default(void) {
    const char* e = getenv("H3_SIGMA_V");
    double sv = e ? atof(e) : 0.0;
    if (sv < 1e-6) sv = 1e-6;
    if (sv > 1.0) sv = 1.0;
    return sv;
}

static double shift_map(double s, double f, double g) {
    double base = s / (f + s * (1.0 - f));
    return g * base / (1.0 + (g - 1.0) * base);
}

static double axis_val(int dim, int patch, int idx, double sqrt_area) {
    int n = dim / patch;
    double ratio = (double)dim / sqrt_area;
    return ((double)idx * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
}

static double video_t_at(int k, double origin) {
    double acc = 0;
    for (int i = 0; i < k; i++)
        acc += FRAME_RESCALE * FRAME_PER_TOKEN[i % 5];
    return origin + acc;
}

static float g_t_emb_m[2][16];   /* M=2 stream embeddings (m0: v/text, m1: audio) */
static long g_seq_len; static int g_tag_sel;

int run_h3_forward_main(int argc, char** argv);   /* old main */

/* ================= resident multi-step sampler =================
 * The Euler driver used to spawn ONE PROCESS PER STEP (h3_euler.sh ->
 * ./h3_forward), re-doing per step: gguf header parse, Metal device init,
 * condition_proj, the 2-block token refiner, both patch projections and
 * the pos/tag table — all of which are IDENTICAL across steps (only
 * sigma and the video latent change). With H3_SAMPLER_STEPS=N the same
 * process instead:
 *   1. runs the one-time conditioning once,
 *   2. loops N sigma steps: refresh AdaLN rows from the new sigma, run
 *      the 50-block denoiser, apply the Euler update in patch space.
 * Audio latent state is carried the same way (H3_A_IN -> per-step rows).
 * Numerics per step are identical to the single-step binary: same code
 * path, same fp order; the sampler only skips re-doing step-invariant
 * work. Sampling-loop timing now equals 50-block time + epsilon. */

/* state that is step-invariant, captured after one-time setup */
typedef struct {
    long seq_len; int s_text, n_audio, n_video;
    int HID, HD, HEADS, QKV, comp, FF1, FF2, NL, T_DIM;
    long TBL_ROWS;
    float* stream;      /* [seq_len, HID] — video/audio rows re-embedded per step */
    double* pos;        /* [seq_len, 3] */
    int* tag;           /* [seq_len] */
    float* text_cond;   /* refined text rows [s_text, HID] — copied into stream each step */
    float* astate;      /* audio rows [n_audio, 32] or NULL */
    float* xstate;      /* video patch rows [n_video, 96] (Euler state, lives here) */
    float* xnext;       /* video velocity rows [n_video, 96] from the last step */
    void* adaln6;       /* [6][6][HID] rows for ONE layer (lazy per-layer) */
    const float* inv_freq;
    float* xres; float* qkv; float* attn; float* fc1o; float* proj;
} H3SamState;

static int h3_sampler_step(H3SamState* st, double sigma_v, FILE* xsrc);
int h3_sampler_main(int argc, char** argv);
static int h3_sampler_setup(H3SamState* st, int argc, char** argv,
                            int vt, int lat_h, int lat_w, int audio_t);

/* M4 driver: Euler loop lives in h3_sample_main below; the packed
 * forward itself is parameterized by sigma through g_t_emb_*. */
static int g_resident_sampler = 0;
int main(int argc, char** argv) {
    if (getenv("H3_SAMPLER_STEPS")) g_resident_sampler = atoi(getenv("H3_SAMPLER_STEPS"));
    if (g_resident_sampler > 0) return h3_sampler_main(argc, argv);
    return run_h3_forward_main(argc, argv);
}

int run_h3_forward_main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <denoiser.gguf> <text_states.bin> "
                        "[vt=1 lat_h=32 lat_w=32 audio_t=4]\n", argv[0]);
        return 1;
    }
    int vt = argc > 3 ? atoi(argv[3]) : 1;
    int lat_h = argc > 4 ? atoi(argv[4]) : 32;
    int lat_w = argc > 5 ? atoi(argv[5]) : 32;
    int audio_t = argc > 6 ? atoi(argv[6]) : 4;

    if (gguf_open(argv[1], &G) != 0) return 1;

#ifdef STRATUM_USE_METAL
    {
        const char* nc = getenv("STRATUM_H3_NC");
        if (nc && atoi(nc)) {
            h3_metal_init_once();
            stratum_metal_set_model_base(G.mmap_base, G.mmap_size);
            {
                const char* ml = getenv("STRATUM_H3_ATTNLIB");
                stratum_metal_h3_attn_init(ml ? ml : "/tmp/h3_attn.metallib");
            }
        }
    }
#endif

    int HID = 0;
    { const GgufTensor* t = TT("final_layer.norm.weight"); HID = (int)t->dims[0]; }
    int HD = 0, HEADS, QKV, FF1, FF2, NL = 0;
    { const GgufTensor* t = TT("blocks.0.attn.q_norm.weight"); HD = (int)t->dims[0]; }
    { const GgufTensor* t = TT("blocks.0.attn.qkv_proj.weight");
      QKV = (int)t->dims[1]; HEADS = QKV / 3 / HD; }
    { const GgufTensor* t = TT("blocks.0.mlp.fc1.weight"); FF1 = (int)t->dims[1]; }
    { const GgufTensor* t = TT("blocks.0.mlp.fc2.weight"); FF2 = (int)t->dims[0]; }
    for (uint64_t i = 0; i < G.n_tensors; i++) {
        if (!strncmp(G.tensors[i].name, "blocks.", 7)) {
            int bi = atoi(G.tensors[i].name + 7);
            if (bi + 1 > NL) NL = bi + 1;
        }
    }
    int comp = QKV / 3;

    /* --- text_states from dump --- */
    long s_text = 0; int te_hid = 0;
    float* text_states = NULL;
    {
        FILE* f = fopen(argv[2], "rb");
        char magic[8]; uint32_t meta[3];
        if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "Q3TE0001", 8)) {
            fprintf(stderr, "bad text_states dump\n"); return 1;
        }
        if (fread(meta, 4, 3, f) != 3) return 1;
        s_text = meta[0]; te_hid = (int)meta[1];
        if (te_hid != 5120) { fprintf(stderr, "text_states hid %d?\n", te_hid); return 1; }
        text_states = malloc(sizeof(float) * s_text * te_hid);
        if (fread(text_states, 4, s_text * te_hid, f) != (size_t)(s_text * te_hid))
            return 1;
        fclose(f);
    }

    /* --- text conditioning: condition_proj (bf16, bias) + token_refiner --- */
    /* per-row out_proj scratch: indexed trow[s*HID] over ALL s_text rows —
     * sized for the full sequence (was MAX_HID: ASan-caught heap overflow
     * that silently stomped adjacent heap on every run) */
    float* trow = malloc(sizeof(float) * (size_t)s_text * HID);
    float* text_cond = malloc(sizeof(float) * s_text * HID);
    {
        const GgufTensor* w = TT("condition_proj.weight");
        const GgufTensor* b = TT("condition_proj.bias");
        const uint16_t* wr = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* br = (const uint16_t*)(G.mmap_base + b->offset);
        for (long s = 0; s < s_text; s++) {
            for (int r = 0; r < HID; r++) {
                const uint16_t* row = wr + (size_t)r * te_hid;
                double acc = (double)bfv(br[r]);
                for (int c = 0; c < te_hid; c++)
                    acc += (double)bfv(row[c]) * (double)text_states[s * te_hid + c];
                text_cond[s * HID + r] = (float)acc;
            }
        }
    }
    act_stats("cond_proj", text_cond, s_text * HID);

    /* token_refiner: 2 blocks, no adaln, rope identity? Reference
     * RefinerBlock calls self.attn(...) WITHOUT rope_freqs => the qk-norm
     * else-branch applies ONLY q_norm/k_norm, no rope. Bidirectional. */
    for (int li = 0; li < 2; li++) {
        char nm[160];
        float* xr = malloc(sizeof(float) * s_text * HID);
        memcpy(xr, text_cond, sizeof(float) * s_text * HID);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.norm1.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < s_text; s++) {
                float* r = &text_cond[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
            }
        }
        float* qkv = malloc(sizeof(float) * s_text * QKV);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.qkv_proj.weight", li);
        mixed_gemv_batch(TT(nm), HID, QKV, s_text, text_cond, qkv);
        if (li == 0 && getenv("H3_DBG_QKV")) {
            FILE* df = fopen(getenv("H3_DBG_QKV"), "wb");
            fwrite(qkv, 4, (size_t)s_text * QKV, df);
            fclose(df);
        }
        /* qk-norm only (no rope in refiner) */
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.q_norm.weight", li);
        const uint16_t* qw = (const uint16_t*)T(nm);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.k_norm.weight", li);
        const uint16_t* kw = (const uint16_t*)T(nm);
        for (long s = 0; s < s_text; s++) {
            for (int h = 0; h < HEADS; h++) {
                float* qp = &qkv[s * QKV + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)qp[d] * qp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) qp[d] *= sc * bfv(qw[d]);
            }
            for (int h = 0; h < HEADS; h++) {
                float* kp = &qkv[s * QKV + comp + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)kp[d] * kp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) kp[d] *= sc * bfv(kw[d]);
            }
        }
        /* bidirectional attention */
        float* attn = calloc((size_t)s_text * comp, sizeof(float));
        float scale = 1.0f / sqrtf((float)HD);
        for (int h = 0; h < HEADS; h++)
            for (int a = 0; a < s_text; a++) {
                const float* qh = &qkv[a * QKV + h * HD];
                float lg[4096];
                for (int b2 = 0; b2 < s_text; b2++) {
                    const float* kh = &qkv[b2 * QKV + comp + h * HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++) dot += (double)qh[d] * kh[d];
                    lg[b2] = (float)(dot * scale);
                }
                float mx = lg[0];
                for (int j = 1; j < s_text; j++) if (lg[j] > mx) mx = lg[j];
                double se = 0;
                for (int j = 0; j < s_text; j++) { lg[j] -= mx; se += exp((double)lg[j]); }
                float inv = (float)(1.0 / se);
                float* oh = &attn[a * comp + h * HD];
                for (int b2 = 0; b2 < s_text; b2++) {
                    float pv = expf(lg[b2]) * inv;
                    const float* vh = &qkv[b2 * QKV + 2 * comp + h * HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv * vh[d];
                }
            }
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.out_proj.weight", li);
        mixed_gemv_batch(TT(nm), comp, HID, s_text, attn, trow);
        for (long i = 0; i < s_text * HID; i++) text_cond[i] = xr[i] + trow[i];
        free(qkv); free(attn);

        /* mlp */
        memcpy(xr, text_cond, sizeof(float) * s_text * HID);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.norm2.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < s_text; s++) {
                float* r = &text_cond[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
            }
        }
        float* f1 = malloc(sizeof(float) * s_text * FF1);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.mlp.fc1.weight", li);
        mixed_gemv_batch(TT(nm), HID, FF1, s_text, text_cond, f1);
        float* fa = malloc(sizeof(float) * s_text * FF2);
        for (long s = 0; s < s_text; s++)
            for (int i = 0; i < FF2; i++) {
                float gv = f1[s * FF1 + i];
                fa[s * FF2 + i] = (gv / (1.0f + expf(-gv))) * f1[s * FF1 + FF2 + i];
            }
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.mlp.fc2.weight", li);
        mixed_gemv_batch(TT(nm), FF2, HID, s_text, fa, trow);
        for (long i = 0; i < s_text * HID; i++) text_cond[i] = xr[i] + trow[i];
        free(xr); free(f1); free(fa);
    }
    {   /* final_norm of the refiner */
        const uint16_t* g16 = (const uint16_t*)T("token_refiner.final_norm.weight");
        for (long s = 0; s < s_text; s++) {
            float* r = &text_cond[s * HID];
            double ss = 0;
            for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
            float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
            for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
        }
    }
    act_stats("refined-text", text_cond, s_text * HID);

    /* --- audio/video embeds (synthetic latents) --- */
    int nh = lat_h / 2, nw = lat_w / 2;
    int frame_rows = nh * nw;
    int n_audio = audio_t * 2;
    int n_video = vt * frame_rows;
    long seq_len = s_text + n_audio + n_video;
    fprintf(stderr, "packed: text=%ld audio=%d video=%d seq=%ld\n",
            s_text, n_audio, n_video, seq_len);

    float* stream = malloc(sizeof(float) * seq_len * HID);
    double* pos = calloc((size_t)seq_len * 3, sizeof(double));
    int* tag = malloc(sizeof(int) * seq_len);
    memcpy(stream, text_cond, sizeof(float) * s_text * HID);
    for (long s = 0; s < s_text; s++) {
        pos[s * 3 + 0] = (double)s;
        tag[s] = 1;
    }
    /* audio rows */
    {
        float* arow = malloc(sizeof(float) * AUDIO_ROW_DIM);
        const GgufTensor* w = TT("audio_patch_proj.weight");
        const GgufTensor* b = TT("audio_patch_proj.bias");
        double w_low = axis_val(lat_w, 2, 0, sqrt((double)lat_h * lat_w));
        double w_high = axis_val(lat_w, 2, nw - 1, sqrt((double)lat_h * lat_w));
        float* astate = NULL;
        {
            const char* ain = getenv("H3_A_IN");
            if (ain) {
                FILE* af = fopen(ain, "rb");
                if (!af || fread(arow, 4, (size_t)n_audio * AUDIO_ROW_DIM, af)
                        != (size_t)n_audio * AUDIO_ROW_DIM) {
                    fprintf(stderr, "H3_A_IN unreadable\n"); return 1;
                }
                fclose(af);
                astate = arow;   /* rows (t, c): [n_audio, 32] */
                arow = malloc(sizeof(float) * AUDIO_ROW_DIM);
            }
        }
        long r = s_text;
        for (int t = 0; t < audio_t; t++)
            for (int c = 0; c < 2; c++) {
                if (astate)
                    memcpy(arow, astate + (size_t)(t * 2 + c) * AUDIO_ROW_DIM,
                           sizeof(float) * AUDIO_ROW_DIM);
                else
                    for (int i = 0; i < AUDIO_ROW_DIM; i++)
                        arow[i] = (float)(((i * 7 + t * 3 + c) % 29) * 0.05 - 0.7);
                f32_gemv(T("audio_patch_proj.weight"), AUDIO_ROW_DIM, HID,
                         arow, &stream[r * HID]);
                for (int i = 0; i < HID; i++)
                    stream[r * HID + i] += ((const float*)T("audio_patch_proj.bias"))[i];
                pos[r * 3 + 0] = (double)s_text + t;
                pos[r * 3 + 2] = c == 0 ? w_low : w_high;
                tag[r] = 2;
                r++;
            }
        free(arow);
    }
    /* video rows: patchify synthetic [24][vt][lat_h][lat_w] */
    {
        size_t ln = (size_t)LATENTS_DIM * vt * lat_h * lat_w;
        float* latent = malloc(sizeof(float) * ln);
        const char* xin = getenv("H3_X_IN");
        if (xin) {
            /* patch-space state: [n_video rows, 96] — distribute row k to
             * its 24-channel patch */
            FILE* xf = fopen(xin, "rb");
            float* prow_in = malloc(sizeof(float) * (size_t)n_video * 96);
            if (!xf || fread(prow_in, sizeof(float),
                             (size_t)n_video * 96, xf)
                    != (size_t)n_video * 96) {
                fprintf(stderr, "H3_X_IN unreadable\n"); return 1;
            }
            fclose(xf);
            long rr = 0;
            for (int t = 0; t < vt; t++)
                for (int hh = 0; hh < nh; hh++)
                    for (int ww = 0; ww < nw; ww++) {
                        for (int c = 0; c < LATENTS_DIM; c++)
                            for (int b = 0; b < 2; b++)
                                for (int a = 0; a < 2; a++)
                                    latent[(((size_t)c * vt + t) * lat_h
                                            + (2 * hh + b)) * lat_w
                                           + (2 * ww + a)] =
                                        prow_in[rr * 96
                                                + c * 4 + b * 2 + a];
                        rr++;
                    }
            free(prow_in);
        } else {
            for (size_t i = 0; i < ln; i++)
                latent[i] = (float)(((i * 13) % 41) * 0.06 - 1.2);
        }
        float* prow = malloc(sizeof(float) * VIDEO_ROW_DIM);
        long r = s_text + n_audio;
        double sqrt_area = sqrt((double)lat_h * lat_w);
        for (int t = 0; t < vt; t++)
            for (int hh = 0; hh < nh; hh++)
                for (int ww = 0; ww < nw; ww++) {
                    int p = 0;
                    for (int c = 0; c < LATENTS_DIM; c++)
                        for (int b = 0; b < 2; b++)
                            for (int a = 0; a < 2; a++)
                                prow[p++] = latent[
                                    (((size_t)c * vt + t) * lat_h + (2 * hh + b))
                                    * lat_w + (2 * ww + a)];
                    f32_gemv(T("video_patch_proj.weight"), VIDEO_ROW_DIM, HID,
                             prow, &stream[r * HID]);
                    for (int i = 0; i < HID; i++)
                        stream[r * HID + i] +=
                            ((const float*)T("video_patch_proj.bias"))[i];
                    pos[r * 3 + 0] = video_t_at(t, (double)s_text);
                    pos[r * 3 + 1] = axis_val(lat_h, 2, hh, sqrt_area);
                    pos[r * 3 + 2] = axis_val(lat_w, 2, ww, sqrt_area);
                    tag[r] = 0;
                    r++;
                }
        free(latent); free(prow);
    }
    act_stats("packed-stream", stream, seq_len * HID);

    /* --- AdaLN at t=0: rows [3][6*HID], per-tag --- */
    int T_DIM = 0; long TBL_ROWS = 0;
    {
        const GgufTensor* t = TT("adaln_t_table");
        T_DIM = (int)t->dims[0]; TBL_ROWS = (long)t->dims[1];
    }
    /* M4: two stream timesteps. t_v = 1 - sigma_v (video/text);
     * t_a = 1 - shift_map(sigma_v, 12 -> 3) (audio). Table row =
     * fractional index of t over [0,1] grid (rows-1). */
    float t_emb[16];
    {
        double sigma_v = g_sigma_v, sigma_a = g_sigma_a;
        if (sigma_v <= 0.0) {
            sigma_v = env_sigma_default();
            sigma_a = shift_map(sigma_v, 12.0, 3.0);
        }
        double t_arr[2] = { 1.0 - sigma_v, 1.0 - sigma_a };  /* m=0, m=1 */
        const float* raw = (const float*)T("adaln_t_table");
        for (int m = 0; m < 2; m++) {
            double pv = t_arr[m] * ((double)TBL_ROWS - 1.0);
            int p0 = (int)pv; if (p0 > (int)TBL_ROWS - 2) p0 = (int)TBL_ROWS - 2;
            double pf = pv - p0;
            for (int k = 0; k < T_DIM; k++)
                g_t_emb_m[m][k] = (float)(
                    (1.0 - pf) * raw[(size_t)p0 * T_DIM + k]
                    + pf * raw[(size_t)(p0 + 1) * T_DIM + k]);
        }
        memcpy(t_emb, g_t_emb_m[0], sizeof(float) * T_DIM);
    }
    /* per-layer AdaLN rows computed lazily at the top of the block loop
     * (was: all NL layers up front = 6*6*MAX_HID*4*NL ≈ 56 MB anon).
     * Computing one layer just-in-time keeps only 0.74 MB live; numerics
     * are unchanged (same rows, same fp order, just deferred). */
    float (*adaln6)[6][6][HID] = malloc(sizeof(float[6][6][HID]));
    (void)0; /* g_adaln removed */
    double t_fw = h3_now_s();

    /* --- denoiser blocks --- */
    /* NOTE (measured, seq=276 M4 Pro, interleaved A/B): page-aligned
     * (aligned_alloc) scratch made the NC path ~12% SLOWER (36.5s vs
     * 32.3s), and NoCopy x/y windows over the activations were slower
     * still (39-46s). The staging xbuf/ybuf copies inside nc_batch_add
     * beat NoCopy windows on small buffers; malloc is intentional here. */
    float* xres = malloc(sizeof(float) * seq_len * HID);
    /* ONE fused activation buffer: rows of FF1 = QKV(21504) + comp(7168).
     *   attention phase: [qkv 21504 | attn-out 7168]
     *   mlp phase:       [fc1-out 28672] (swiglu compacts to [0,FF2))
     * qkv/attn-out/fc1 reuse the same memory in phases; only xres, stream
     * and proj stay live. (was 4 separate seq-sized buffers) */
    float* fbuf = malloc(sizeof(float) * (size_t)seq_len * FF1);
    if (((uintptr_t)fbuf & 16383) == 0 && g_metal_ready)
        stratum_metal_nc_attn_direct_register(fbuf,
            (size_t)seq_len * FF1 * sizeof(float));
    float* qkv = fbuf;            /* row stride FF1, q at +0 */
    float* attn = fbuf + QKV;     /* row stride FF1, region offset QKV */
    float* fc1o = fbuf;
    float* proj = malloc(sizeof(float) * seq_len * HID);
    const float* inv_freq = (const float*)T("rope.inv_freq");
    prof_on();
    memset(g_prof, 0, sizeof g_prof);
    clock_t t0 = clock();

    for (int li = 0; li < NL; li++) {
        char nm[160];
        /* AdaLN rows for THIS layer, computed lazily (0.74MB live table) */
        snprintf(nm, sizeof nm, "blocks.%d.adaln_proj.linear.weight", li);
        const GgufTensor* w = TT(nm);
        snprintf(nm, sizeof nm, "blocks.%d.adaln_proj.linear.bias", li);
        const GgufTensor* bts = TT(nm);
        const uint16_t* wb = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* bb = (const uint16_t*)(G.mmap_base + bts->offset);
        /* rows m*3+tag over M=2: 0=video(m0) 1=text(m0) 5=audio(m1) */
        for (int row = 0; row < 6; row++) {
            const float* emb_row = g_t_emb_m[row / 3];
            for (int cidx = 0; cidx < 6 * HID; cidx++) {
                const uint16_t* wrow = wb + (size_t)cidx * T_DIM;
                double acc = (double)f16v(bb[cidx]);
                for (int k = 0; k < T_DIM; k++)
                    acc += (double)f16v(wrow[k]) * (double)emb_row[k];
                int chunk = cidx / HID, i = cidx % HID;
                adaln6[0][row][chunk][i] = (float)acc;
            }
        }
        /* rows: video=0 (m0 tag0), text=1 (m0 tag1), audio=5 (m1 tag2) */
        double h3p_t0 = 0; int h3p_fresh = 1; (void)h3p_t0; (void)h3p_fresh;
        PROF_BEGINSLOT(0);
        float* shift_msa = adaln6[0][0][0];
        float* scale_msa = adaln6[0][0][1];
        float* gate_msa  = adaln6[0][0][2];
        float* shift_mlp = adaln6[0][0][3];
        float* scale_mlp = adaln6[0][0][4];
        float* gate_mlp  = adaln6[0][0][5];
        float* shift_msa_t = adaln6[0][1][0];
        float* scale_msa_t = adaln6[0][1][1];
        float* gate_msa_t  = adaln6[0][1][2];
        float* shift_mlp_t = adaln6[0][1][3];
        float* scale_mlp_t = adaln6[0][1][4];
        float* gate_mlp_t  = adaln6[0][1][5];
        float* shift_msa_a = adaln6[0][5][0];
        float* scale_msa_a = adaln6[0][5][1];
        float* gate_msa_a  = adaln6[0][5][2];
        float* shift_mlp_a = adaln6[0][5][3];
        float* scale_mlp_a = adaln6[0][5][4];
        float* gate_mlp_a  = adaln6[0][5][5];

        memcpy(xres, stream, sizeof(float) * seq_len * HID);
        PROF_ENDSLOT(0);
        snprintf(nm, sizeof nm, "blocks.%d.norm1.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < seq_len; s++) {
                float* r = &stream[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++)
                    r[i] = r[i] * sc * bfv(g16[i])
                         * (1.0f + (tag[s] == 0 ? scale_msa[i]
                                  : tag[s] == 1 ? scale_msa_t[i]
                                  : scale_msa_a[i]))
                         + (tag[s] == 0 ? shift_msa[i]
                            : tag[s] == 1 ? shift_msa_t[i] : shift_msa_a[i]);
            }
        }
        snprintf(nm, sizeof nm, "blocks.%d.attn.qkv_proj.weight", li);
        PROF_BEGINSLOT(1);
        mixed_gemv_batch_strided(TT(nm), HID, QKV, seq_len, stream, qkv,
                                 HID, FF1);
        PROF_ENDSLOT(1);
        /* fused qk-norm + split-half rope, per-tag AdaLN'd rows already in */
        snprintf(nm, sizeof nm, "blocks.%d.attn.q_norm.weight", li);
        const uint16_t* qw = (const uint16_t*)T(nm);
        snprintf(nm, sizeof nm, "blocks.%d.attn.k_norm.weight", li);
        const uint16_t* kw = (const uint16_t*)T(nm);
        PROF_BEGINSLOT(2);
        for (long s = 0; s < seq_len; s++) {
            for (int h = 0; h < HEADS; h++) {
                float* qp = &qkv[s * FF1 + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)qp[d] * qp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) qp[d] *= sc * bfv(qw[d]);
            }
            for (int h = 0; h < HEADS; h++) {
                float* kp = &qkv[s * FF1 + comp + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)kp[d] * kp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) kp[d] *= sc * bfv(kw[d]);
            }
            for (int j = 0; j < ROT_PAIRS; j++) {
                int axis = j / 16, kbase = j % 16;
                float ang = (float)pos[s * 3 + axis] * inv_freq[kbase];
                float c = cosf(ang), sn = sinf(ang);
                for (int h = 0; h < HEADS; h++) {
                    float* qp = &qkv[s * FF1 + h * HD];
                    float a0 = qp[j], a1 = qp[HD / 2 + j];
                    qp[j]        = a0 * c - a1 * sn;
                    qp[HD / 2 + j] = a0 * sn + a1 * c;
                    float* kp = &qkv[s * FF1 + comp + h * HD];
                    a0 = kp[j]; a1 = kp[HD / 2 + j];
                    kp[j]        = a0 * c - a1 * sn;
                    kp[HD / 2 + j] = a0 * sn + a1 * c;
                }
            }
        }

        /* bidirectional attention (full packed stream) */
        PROF_ENDSLOT(2);
        float scale2 = 1.0f / sqrtf((float)HD);
        float* lgd = NULL;   /* CPU-attention scratch; NULL on the GPU path */
#ifdef STRATUM_USE_METAL
        int attn_minseq = 512;
            { const char* e = getenv("H3_ATTN_MINSEQ"); if (e) attn_minseq = atoi(e); }
            if (g_h3_nc && g_metal_ready && seq_len >= attn_minseq) {
            /* GPU flash attention IN PLACE over fbuf: kernel reads the
             * q/k/v regions of each row and writes the attn-out region.
             * No gather, no staging (fbuf is registered when 16K-aligned). */
            static int attn_used = 0;
            int attn_layer_max = 1000000;
            { const char* e = getenv("H3_ATTN_LAYERS"); if (e) attn_layer_max = atoi(e); }
            if (attn_used < attn_layer_max) {
                attn_used++;
                stratum_metal_nc_batch_begin();
                int arc = stratum_metal_nc_batch_attn_packed(fbuf, FF1,
                                    0, comp, 2 * comp, QKV, attn,
                                    (int)seq_len, HEADS, HD, scale2);
                stratum_metal_nc_batch_flush();
                if (arc == 0)
                    goto attn_done;
            }
        }
#endif
        {   /* strided CPU attention over fbuf rows [q|k|v|out] */
            float* att_row0 = fbuf + QKV;
            for (long s = 0; s < seq_len; s++)
                memset(att_row0 + s * FF1, 0, sizeof(float) * comp);
        }
        lgd = malloc(sizeof(float) * seq_len);
        for (int h = 0; h < HEADS; h++)
            for (long a = 0; a < seq_len; a++) {
                const float* qh = &qkv[a * FF1 + h * HD];
                for (long b2 = 0; b2 < seq_len; b2++) {
                    const float* kh = &qkv[b2 * FF1 + comp + h * HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++)
                        dot += (double)qh[d] * kh[d];
                    lgd[b2] = (float)(dot * scale2);
                }
                float mx = lgd[0];
                for (long j = 1; j < seq_len; j++) if (lgd[j] > mx) mx = lgd[j];
                double se = 0;
                for (long j = 0; j < seq_len; j++) {
                    lgd[j] -= mx; se += exp((double)lgd[j]);
                }
                float inv = (float)(1.0 / se);
                float* oh = &attn[a * FF1 + h * HD];
                for (long b2 = 0; b2 < seq_len; b2++) {
                    float pv = expf(lgd[b2]) * inv;
                    const float* vh = &qkv[b2 * FF1 + 2 * comp + h * HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv * vh[d];
                }
            }

        attn_done:;
        PROF_ENDSLOT(3);
        snprintf(nm, sizeof nm, "blocks.%d.attn.out_proj.weight", li);
        PROF_BEGINSLOT(4);
        mixed_gemv_batch_strided(TT(nm), comp, HID, seq_len, attn, proj,
                                 FF1, HID);
        PROF_ENDSLOT(4);
        PROF_BEGINSLOT(0);
        if (getenv("H3_ATTN_PROBE") && li == 2) {
            long b1=0; double m1=0;
            for (long s = 0; s < seq_len; s++)
                for (int t = 0; t < comp; t++) {
                    float v = attn[s * FF1 + t];
                    if (!isfinite(v)) b1++; else if (fabs(v)>m1) m1=fabs(v);
                }
            long b2=0; double m2=0;
            for (long t = 0; t < seq_len * HID; t++)
                if (!isfinite(proj[t])) b2++; else if (fabs(proj[t])>m2) m2=fabs(proj[t]);
            fprintf(stderr, "  [L2 probe] attn nonfinite=%ld max=%.4g | proj nonfinite=%ld max=%.4g\n",
                    b1, m1, b2, m2);
        }
        for (long s = 0; s < seq_len; s++) {
            const float* g = tag[s] == 0 ? gate_msa
                           : tag[s] == 1 ? gate_msa_t : gate_msa_a;
            for (int i = 0; i < HID; i++)
                stream[s * HID + i] = xres[s * HID + i] + g[i] * proj[s * HID + i];
        }
        PROF_ENDSLOT(0);

        /* mlp */
        PROF_BEGINSLOT(0);
        memcpy(xres, stream, sizeof(float) * seq_len * HID);
        PROF_ENDSLOT(0);
        PROF_BEGINSLOT(5);
        snprintf(nm, sizeof nm, "blocks.%d.norm2.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < seq_len; s++) {
                float* r = &stream[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++)
                    r[i] = r[i] * sc * bfv(g16[i])
                         * (1.0f + (tag[s] == 0 ? scale_mlp[i]
                                  : tag[s] == 1 ? scale_mlp_t[i]
                                  : scale_mlp_a[i]))
                         + (tag[s] == 0 ? shift_mlp[i]
                            : tag[s] == 1 ? shift_mlp_t[i] : shift_mlp_a[i]);
            }
        }
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc1.weight", li);
        const GgufTensor* t_fc1 = TT(nm);
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc2.weight", li);
        const GgufTensor* t_fc2 = TT(nm);
        int mlp_fused = 0;
#ifdef STRATUM_USE_METAL
        static int fused_en = -1;
        if (fused_en < 0) {
            const char* e = getenv("H3_MLP_FUSED");
            fused_en = (e && atoi(e) == 0) ? 0 : 1;
        }
        if (fused_en && g_h3_nc && g_metal_ready &&
            (t_fc1->type == 12 || t_fc1->type == 14) &&
            t_fc2->type == t_fc1->type) {
            stratum_metal_nc_batch_begin();
            int rc = stratum_metal_nc_mlp_fused(
                (const void*)(G.mmap_base + t_fc1->offset), (size_t)t_fc1->nbytes,
                (int)t_fc1->type,
                (const void*)(G.mmap_base + t_fc2->offset), (size_t)t_fc2->nbytes,
                stream, fc1o, FF1, proj, (int)seq_len, HID, FF2);
            stratum_metal_nc_batch_flush();
            mlp_fused = (rc == 0);
        }
#endif
        if (!mlp_fused) {
        PROF_BEGINSLOT(6);
        mixed_gemv_batch_strided(t_fc1, HID, FF1, seq_len, stream, fc1o,
                                 HID, FF1);
        PROF_ENDSLOT(6);
        PROF_BEGINSLOT(7);
        for (long s = 0; s < seq_len; s++)
            for (int i = 0; i < FF2; i++) {
                float gv = fc1o[s * FF1 + i];
                fc1o[s * FF1 + i] =
                    (gv / (1.0f + expf(-gv))) * fc1o[s * FF1 + FF2 + i];
            }
        PROF_ENDSLOT(7);
        PROF_BEGINSLOT(7);
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc2.weight", li);
        mixed_gemv_batch_strided(TT(nm), FF2, HID, seq_len, fc1o, proj,
                                 FF1, HID);
        PROF_ENDSLOT(7);
        }
        if (getenv("H3_MLP_PROBE") && li == 0) {
            FILE* fp = fopen("/tmp/mlp_proj_probe.bin", "wb");
            if (fp) { fwrite(proj, 4, (size_t)seq_len * HID, fp); fclose(fp); }
        }
        PROF_BEGINSLOT(0);
        for (long s = 0; s < seq_len; s++) {
            const float* g = tag[s] == 0 ? gate_mlp
                           : tag[s] == 1 ? gate_mlp_t : gate_mlp_a;
            for (int i = 0; i < HID; i++)
                stream[s * HID + i] =
                    xres[s * HID + i] + g[i] * proj[s * HID + i];
        }
        PROF_ENDSLOT(0);

        free(lgd);
        lgd = NULL;
        if (getenv("H3_ATTN_CHK")) {
            long bad = 0; double bmax = 0;
            for (long t = 0; t < seq_len * HID; t++) {
                if (!isfinite(stream[t])) bad++;
                else if (fabs(stream[t]) > bmax) bmax = fabs(stream[t]);
            }
            fprintf(stderr, "  [li=%d] stream nonfinite=%ld max=%.4g\n", li, bad, bmax);
        }
        if (li % 10 == 0 || li == NL - 1 || getenv("H3_ATTN_CHK")) {
            fprintf(stderr, "  L%02d\n", li);
            act_stats("post-mlp", stream, seq_len * HID);
        }
    }
    double elapsed = h3_now_s() - t_fw;
    fprintf(stderr, "\n  packed forward (%d layers, seq=%ld): %.1fs wall\n",
            NL, seq_len, elapsed);
    h3_footprint("post-forward");
    if (g_h3_prof) {
        double acc = 0;
        for (int i = 1; i < 8; i++) acc += g_prof[i];
        g_prof[0] = elapsed - acc;   /* slot 0 = everything not named */
        fprintf(stderr, "  profile:");
        for (int i = 0; i < 8; i++)
            fprintf(stderr, " %s %.2fs", g_prof_nm[i], g_prof[i]);
        fprintf(stderr, "\n");
    }

    /* --- M4b: final_layer — norm + its own AdaLN (expand=2, modalities=1)
     * + video_out/audio_out F32 heads. THIS is the velocity in patch
     * space; the raw hidden state is not the model output. --- */
    {
        /* heap (was stack [2][MAX_HID]x2 = 128KB): sized by runtime HID */
        float (*fshift)[HID] = malloc(sizeof(float[2][HID]));
        float (*fscale)[HID] = malloc(sizeof(float[2][HID]));
        const GgufTensor* w = TT("final_layer.adaln_proj.linear.weight");
        const GgufTensor* bts = TT("final_layer.adaln_proj.linear.bias");
        const uint16_t* wb = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* bb = (const uint16_t*)(G.mmap_base + bts->offset);
        for (int m = 0; m < 2; m++) {
            for (int cidx = 0; cidx < 2 * HID; cidx++) {
                const uint16_t* wrow = wb + (size_t)cidx * T_DIM;
                double acc = (double)f16v(bb[cidx]);
                for (int k = 0; k < T_DIM; k++)
                    acc += (double)f16v(wrow[k]) * (double)g_t_emb_m[m][k];
                int chunk = cidx / HID, i = cidx % HID;
                if (chunk == 0) fshift[m][i] = (float)acc;
                else fscale[m][i] = (float)acc;
            }
        }
        const uint16_t* g16 = (const uint16_t*)T("final_layer.norm.weight");
        /* video rows (m=0) then audio rows (m=1) */
        float* vvel = malloc(sizeof(float) * (size_t)n_video * 96);
        float* avel = malloc(sizeof(float) * (size_t)n_audio * 32);
        long vrow = 0, arow2 = 0;
        for (long s = 0; s < seq_len; s++) {
            if (tag[s] != 0 && tag[s] != 2) continue;
            float* r = &stream[s * HID];
            double ss = 0;
            for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
            float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
            int m = tag[s] == 0 ? 0 : 1;
            for (int i = 0; i < HID; i++)
                r[i] = r[i] * sc * bfv(g16[i]) * (1.0f + fscale[m][i])
                     + fshift[m][i];
            if (tag[s] == 0) {
                f32_gemv(T("final_layer.video_out.weight"), HID, 96, r,
                         &vvel[vrow * 96]);
                for (int i = 0; i < 96; i++)
                    vvel[vrow * 96 + i] +=
                        ((const float*)T("final_layer.video_out.bias"))[i];
                vrow++;
            } else {
                f32_gemv(T("final_layer.audio_out.weight"), HID, 32, r,
                         &avel[arow2 * 32]);
                for (int i = 0; i < 32; i++)
                    avel[arow2 * 32 + i] +=
                        ((const float*)T("final_layer.audio_out.bias"))[i];
                arow2++;
            }
        }
        act_stats("video-velocity", vvel, (long)n_video * 96);
        act_stats("audio-velocity", avel, (long)n_audio * 32);

        FILE* out = fopen("/tmp/h3_packed_out.bin", "wb");
        fwrite("H3PKT001", 1, 8, out);
        uint32_t meta[4] = { (uint32_t)n_video, (uint32_t)96,
                             (uint32_t)n_audio, (uint32_t)32 };
        fwrite(meta, 4, 4, out);
        fwrite(vvel, sizeof(float), (size_t)n_video * 96, out);
        fwrite(avel, sizeof(float), (size_t)n_audio * 32, out);
        fclose(out);
        if (getenv("H3_X_OUT")) {
            FILE* xo = fopen(getenv("H3_X_OUT"), "wb");
            fwrite(vvel, sizeof(float), (size_t)n_video * 96, xo);
            fclose(xo);
        }
        if (getenv("H3_A_OUT")) {
            FILE* ao = fopen(getenv("H3_A_OUT"), "wb");
            fwrite(avel, sizeof(float), (size_t)n_audio * 32, ao);
            fclose(ao);
        }
        printf("velocity: video %dx96, audio %dx32 -> /tmp/h3_packed_out.bin\n",
               n_video, n_audio);
        free(vvel); free(avel); free(fshift); free(fscale);
    }

    return 0;
}

/* ================= resident multi-step sampler (H3_SAMPLER_STEPS) ================= */

static const char* g_te_dump_path = NULL;

/* One-time conditioning + buffers, shared verbatim with run_h3_forward_main:
 * text_states load -> condition_proj -> token_refiner(2 blocks) -> final_norm
 * -> audio/video patch projections (with initial or carried state) -> pos/tag.
 * Everything produced here is step-invariant except the latent state itself. */
static int h3_sampler_setup(H3SamState* st, int argc, char** argv,
                            int vt, int lat_h, int lat_w, int audio_t) {
    (void)argc; (void)argv;
    /* ---- dims (same probes as run_h3_forward_main) ---- */
    int HID = 0;
    { const GgufTensor* t = TT("final_layer.norm.weight"); HID = (int)t->dims[0]; }
    int HD = 0, HEADS, QKV, FF1, FF2, NL = 0;
    { const GgufTensor* t = TT("blocks.0.attn.q_norm.weight"); HD = (int)t->dims[0]; }
    { const GgufTensor* t = TT("blocks.0.attn.qkv_proj.weight");
      QKV = (int)t->dims[1]; HEADS = QKV / 3 / HD; }
    { const GgufTensor* t = TT("blocks.0.mlp.fc1.weight"); FF1 = (int)t->dims[1]; }
    { const GgufTensor* t = TT("blocks.0.mlp.fc2.weight"); FF2 = (int)t->dims[0]; }
    for (uint64_t i = 0; i < G.n_tensors; i++) {
        if (!strncmp(G.tensors[i].name, "blocks.", 7)) {
            int bi = atoi(G.tensors[i].name + 7);
            if (bi + 1 > NL) NL = bi + 1;
        }
    }
    int comp = QKV / 3;
    st->HID = HID; st->HD = HD; st->HEADS = HEADS; st->QKV = QKV;
    st->comp = comp; st->FF1 = FF1; st->FF2 = FF2; st->NL = NL;

    /* ---- text_states dump ---- */
    long s_text = 0; int te_hid = 0;
    float* text_states = NULL;
    {
        FILE* f = fopen(g_te_dump_path, "rb");
        char magic[8]; uint32_t meta[3];
        if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "Q3TE0001", 8)) {
            fprintf(stderr, "bad text_states dump\n"); return 1;
        }
        if (fread(meta, 4, 3, f) != 3) return 1;
        s_text = meta[0]; te_hid = (int)meta[1];
        if (te_hid != 5120) { fprintf(stderr, "text_states hid %d?\n", te_hid); return 1; }
        text_states = malloc(sizeof(float) * s_text * te_hid);
        if (fread(text_states, 4, s_text * te_hid, f) != (size_t)(s_text * te_hid))
            return 1;
        fclose(f);
    }
    st->s_text = (int)s_text;

    /* ---- condition_proj (bf16) ---- */
    float* trow = malloc(sizeof(float) * (size_t)s_text * HID);
    float* text_cond = malloc(sizeof(float) * s_text * HID);
    {
        const GgufTensor* w = TT("condition_proj.weight");
        const GgufTensor* b = TT("condition_proj.bias");
        const uint16_t* wr = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* br = (const uint16_t*)(G.mmap_base + b->offset);
        for (long s = 0; s < s_text; s++) {
            for (int r = 0; r < HID; r++) {
                const uint16_t* row = wr + (size_t)r * te_hid;
                double acc = (double)bfv(br[r]);
                for (int c = 0; c < te_hid; c++)
                    acc += (double)bfv(row[c]) * (double)text_states[s * te_hid + c];
                text_cond[s * HID + r] = (float)acc;
            }
        }
    }
    free(text_states);

    /* ---- token refiner (2 blocks) + final_norm: identical to
     * run_h3_forward_main's refiner section; runs ONCE for the whole
     * sampling loop (its output does not depend on sigma or the latent) ---- */
    for (int li = 0; li < 2; li++) {
        char nm[160];
        float* xr = malloc(sizeof(float) * s_text * HID);
        memcpy(xr, text_cond, sizeof(float) * s_text * HID);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.norm1.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < s_text; s++) {
                float* r = &text_cond[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
            }
        }
        float* qkv = malloc(sizeof(float) * s_text * QKV);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.qkv_proj.weight", li);
        mixed_gemv_batch(TT(nm), HID, QKV, s_text, text_cond, qkv);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.q_norm.weight", li);
        const uint16_t* qw = (const uint16_t*)T(nm);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.k_norm.weight", li);
        const uint16_t* kw = (const uint16_t*)T(nm);
        for (long s = 0; s < s_text; s++) {
            for (int h = 0; h < HEADS; h++) {
                float* qp = &qkv[s * QKV + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)qp[d] * qp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) qp[d] *= sc * bfv(qw[d]);
            }
            for (int h = 0; h < HEADS; h++) {
                float* kp = &qkv[s * QKV + comp + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)kp[d] * kp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) kp[d] *= sc * bfv(kw[d]);
            }
        }
        float* attn = calloc((size_t)s_text * comp, sizeof(float));
        float scale = 1.0f / sqrtf((float)HD);
        for (int h = 0; h < HEADS; h++)
            for (int a = 0; a < s_text; a++) {
                const float* qh = &qkv[a * QKV + h * HD];
                float lg[4096];
                for (int b2 = 0; b2 < s_text; b2++) {
                    const float* kh = &qkv[b2 * QKV + comp + h * HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++) dot += (double)qh[d] * kh[d];
                    lg[b2] = (float)(dot * scale);
                }
                float mx = lg[0];
                for (int j = 1; j < s_text; j++) if (lg[j] > mx) mx = lg[j];
                double se = 0;
                for (int j = 0; j < s_text; j++) { lg[j] -= mx; se += exp((double)lg[j]); }
                float inv = (float)(1.0 / se);
                float* oh = &attn[a * comp + h * HD];
                for (int b2 = 0; b2 < s_text; b2++) {
                    float pv = expf(lg[b2]) * inv;
                    const float* vh = &qkv[b2 * QKV + 2 * comp + h * HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv * vh[d];
                }
            }
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.out_proj.weight", li);
        mixed_gemv_batch(TT(nm), comp, HID, s_text, attn, trow);
        for (long i = 0; i < s_text * HID; i++) text_cond[i] = xr[i] + trow[i];
        free(qkv); free(attn);

        memcpy(xr, text_cond, sizeof(float) * s_text * HID);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.norm2.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < s_text; s++) {
                float* r = &text_cond[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
            }
        }
        float* f1 = malloc(sizeof(float) * s_text * FF1);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.mlp.fc1.weight", li);
        mixed_gemv_batch(TT(nm), HID, FF1, s_text, text_cond, f1);
        float* fa = malloc(sizeof(float) * s_text * FF2);
        for (long s = 0; s < s_text; s++)
            for (int i = 0; i < FF2; i++) {
                float gv = f1[s * FF1 + i];
                fa[s * FF2 + i] = (gv / (1.0f + expf(-gv))) * f1[s * FF1 + FF2 + i];
            }
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.mlp.fc2.weight", li);
        mixed_gemv_batch(TT(nm), FF2, HID, s_text, fa, trow);
        for (long i = 0; i < s_text * HID; i++) text_cond[i] = xr[i] + trow[i];
        free(xr); free(f1); free(fa);
    }
    {
        const uint16_t* g16 = (const uint16_t*)T("token_refiner.final_norm.weight");
        for (long s = 0; s < s_text; s++) {
            float* r = &text_cond[s * HID];
            double ss = 0;
            for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
            float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
            for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
        }
    }
    free(trow);
    st->text_cond = text_cond;

    /* ---- audio state ---- */
    int nh = lat_h / 2, nw = lat_w / 2;
    int frame_rows = nh * nw;
    int n_audio = audio_t * 2;
    int n_video = vt * frame_rows;
    long seq_len = s_text + n_audio + n_video;
    st->n_audio = n_audio; st->n_video = n_video; st->seq_len = seq_len;
    fprintf(stderr, "packed: text=%ld audio=%d video=%d seq=%ld\n",
            s_text, n_audio, n_video, seq_len);

    st->astate = NULL;
    {
        const char* ain = getenv("H3_A_IN");
        if (ain) {
            FILE* af = fopen(ain, "rb");
            st->astate = malloc(sizeof(float) * (size_t)n_audio * AUDIO_ROW_DIM);
            if (!af || fread(st->astate, 4, (size_t)n_audio * AUDIO_ROW_DIM, af)
                    != (size_t)n_audio * AUDIO_ROW_DIM) {
                fprintf(stderr, "H3_A_IN unreadable\n"); return 1;
            }
            fclose(af);
        }
    }

    /* ---- video latent state: initial noise or H3_X_IN, patch rows [n_video, 96] ---- */
    st->xstate = malloc(sizeof(float) * (size_t)n_video * 96);
    st->xnext = malloc(sizeof(float) * (size_t)n_video * 96);
    {
        const char* xin = getenv("H3_X_IN");
        if (xin) {
            FILE* xf = fopen(xin, "rb");
            if (!xf || fread(st->xstate, sizeof(float), (size_t)n_video * 96, xf)
                    != (size_t)n_video * 96) {
                fprintf(stderr, "H3_X_IN unreadable\n"); return 1;
            }
            fclose(xf);
        } else {
            /* deterministic pure noise in [-1,1) — same LCG as h3_euler.sh */
            unsigned state = 12345;
            long ln = (long)n_video * 96;
            for (long i = 0; i < ln; i++) {
                state = (1103515245u * state + 12345u) & 0x7FFFFFFFu;
                st->xstate[i] = (float)((double)state / 0x40000000 - 1.0);
            }
        }
    }

    /* ---- pos/tag tables (step-invariant) ---- */
    st->pos = calloc((size_t)seq_len * 3, sizeof(double));
    st->tag = malloc(sizeof(int) * seq_len);
    for (long s = 0; s < s_text; s++) {
        st->pos[s * 3 + 0] = (double)s;
        st->tag[s] = 1;
    }
    {
        double w_low = axis_val(lat_w, 2, 0, sqrt((double)lat_h * lat_w));
        double w_high = axis_val(lat_w, 2, nw - 1, sqrt((double)lat_h * lat_w));
        long r = s_text;
        for (int t = 0; t < audio_t; t++)
            for (int c = 0; c < 2; c++) {
                st->pos[r * 3 + 0] = (double)s_text + t;
                st->pos[r * 3 + 2] = c == 0 ? w_low : w_high;
                st->tag[r] = 2;
                r++;
            }
        double sqrt_area = sqrt((double)lat_h * lat_w);
        for (int t = 0; t < vt; t++)
            for (int hh = 0; hh < nh; hh++)
                for (int ww = 0; ww < nw; ww++) {
                    st->pos[r * 3 + 0] = video_t_at(t, (double)s_text);
                    st->pos[r * 3 + 1] = axis_val(lat_h, 2, hh, sqrt_area);
                    st->pos[r * 3 + 2] = axis_val(lat_w, 2, ww, sqrt_area);
                    st->tag[r] = 0;
                    r++;
                }
    }

    /* ---- per-step scratch ---- */
    st->stream = malloc(sizeof(float) * seq_len * HID);
    st->adaln6 = malloc(sizeof(float[6][6][HID]));   /* ONE layer (lazy per-layer) */
    st->inv_freq = (const float*)T("rope.inv_freq");
    st->xres = malloc(sizeof(float) * seq_len * HID);
    /* ONE fused buffer: rows of FF1 = [qkv 21504 | attn-out 7168] in the
     * attention phase, [fc1 28672] in the mlp phase. proj stays separate. */
    st->fc1o = malloc(sizeof(float) * (size_t)seq_len * FF1);
    if (((uintptr_t)st->fc1o & 16383) == 0 && g_metal_ready)
        stratum_metal_nc_attn_direct_register(st->fc1o,
            (size_t)seq_len * FF1 * sizeof(float));
    st->qkv = st->fc1o;              /* row stride FF1 */
    st->attn = st->fc1o + QKV;       /* row stride FF1, offset QKV */
    st->proj = malloc(sizeof(float) * seq_len * HID);

    /* ---- T_DIM / TBL_ROWS for the adaln table ---- */
    {
        const GgufTensor* t = TT("adaln_t_table");
        st->T_DIM = (int)t->dims[0]; st->TBL_ROWS = (long)t->dims[1];
    }
    prof_on();
    h3_footprint("post-setup");
    return 0;
}

int h3_sampler_main(int argc, char** argv) {
    int steps = atoi(getenv("H3_SAMPLER_STEPS"));
    if (argc < 3) {
        fprintf(stderr, "usage: %s <denoiser.gguf> <text_states.bin> "
                        "[vt=1 lat_h=32 lat_w=32 audio_t=4]\n", argv[0]);
        return 1;
    }
    int vt = argc > 3 ? atoi(argv[3]) : 1;
    int lat_h = argc > 4 ? atoi(argv[4]) : 32;
    int lat_w = argc > 5 ? atoi(argv[5]) : 32;
    int audio_t = argc > 6 ? atoi(argv[6]) : 4;

    if (gguf_open(argv[1], &G) != 0) return 1;
#ifdef STRATUM_USE_METAL
    {
        const char* nc = getenv("STRATUM_H3_NC");
        if (nc && atoi(nc)) {
            h3_metal_init_once();
            stratum_metal_set_model_base(G.mmap_base, G.mmap_size);
            {
                const char* ml = getenv("STRATUM_H3_ATTNLIB");
                stratum_metal_h3_attn_init(ml ? ml : "/tmp/h3_attn.metallib");
            }
        }
    }
#endif
    /* argv[2] is the text_states dump path used by setup */
    g_te_dump_path = argv[2];

    /* ---- one-time setup (identical code path to run_h3_forward_main) ---- */
    H3SamState st;
    memset(&st, 0, sizeof st);
    if (h3_sampler_setup(&st, argc, argv, vt, lat_h, lat_w, audio_t) != 0) return 1;

    setenv("STRATUM_NC_FREESTAGING", "1", 1);   /* don't hold 64MB staging across steps */
    /* ---- Euler loop ---- */
    double dt = 1.0 / steps;
    double t_all = h3_now_s();
    for (int i = 0; i < steps; i++) {
        double sigma = 1000.0 * (1.0 - (i + 0.5) / steps) / 1000.0;
        double t0 = h3_now_s();
        if (h3_sampler_step(&st, sigma, NULL) != 0) return 1;
        /* Euler update against the flow ODE: the model returns -v in x0
         * space, so x <- x - dt * out (h3_euler.sh contract). */
        for (long k = 0; k < (long)st.n_video * 96; k++)
            st.xstate[k] -= (float)(dt * st.xnext[k]);
        double t1 = h3_now_s();
        fprintf(stderr, "=== step %d/%d (sigma=%.4f) %.1fs\n", i + 1, steps, sigma, t1 - t0);
        h3_footprint("step-end");
    }
    fprintf(stderr, "sampler: %d steps in %.1fs\n", steps, h3_now_s() - t_all);

    /* final latent: video patch-space rows, same layout as h3_euler.sh output */
    const char* xout = getenv("H3_X_OUT");
    FILE* xo = xout ? fopen(xout, "wb") : fopen("/tmp/h3_x_final.bin", "wb");
    if (xo) {
        fwrite(st.xstate, sizeof(float), (size_t)st.n_video * 96, xo);
        fclose(xo);
        fprintf(stderr, "final latent: %s\n", xout ? xout : "/tmp/h3_x_final.bin");
    }
    return 0;
}

/* One denoiser pass at the given sigma, byte-for-byte the same op order and
 * fp sequence as the block loop in run_h3_forward_main — the sampler only
 * re-embeds the (changed) latent rows and refreshes the sigma-dependent
 * AdaLN rows. The velocity output lands in st->xnext (and st->anext). */
static int h3_sampler_step(H3SamState* st, double sigma_v, FILE* xsrc) {
    (void)xsrc;
    const int HID = st->HID, QKV = st->QKV, comp = st->comp;
    const int FF1 = st->FF1, FF2 = st->FF2, NL = st->NL;
    const int HEADS = st->HEADS, HD = st->HD;
    const long seq_len = st->seq_len;
    const long s_text = st->s_text;
    const int n_video = st->n_video, n_audio = st->n_audio;
    float* stream = st->stream;
    float* xres = st->xres;
    float* fbuf = st->fc1o;   /* fused [qkv|attn|fc1] rows of FF1 */
    float* qkv = st->qkv;
    float* attn = st->attn;
    float* fc1o = st->fc1o;
    float* proj = st->proj;
    const float* inv_freq = st->inv_freq;
    const double* pos = st->pos;
    const int* tag = st->tag;

    /* ---- stream: text rows from cached refiner output, audio rows from
     * astate (fixed), video rows re-projected from the Euler state ---- */
    memcpy(stream, st->text_cond, sizeof(float) * (size_t)s_text * HID);
    {
        float arow[AUDIO_ROW_DIM];
        long r = s_text;
        for (int t = 0; t < n_audio / 2; t++)
            for (int c = 0; c < 2; c++) {
                if (st->astate)
                    memcpy(arow, st->astate + (size_t)(t * 2 + c) * AUDIO_ROW_DIM,
                           sizeof(float) * AUDIO_ROW_DIM);
                else
                    for (int i = 0; i < AUDIO_ROW_DIM; i++)
                        arow[i] = (float)(((i * 7 + t * 3 + c) % 29) * 0.05 - 0.7);
                f32_gemv(T("audio_patch_proj.weight"), AUDIO_ROW_DIM, HID,
                         arow, &stream[r * HID]);
                for (int i = 0; i < HID; i++)
                    stream[r * HID + i] += ((const float*)T("audio_patch_proj.bias"))[i];
                r++;
            }
    }
    {
        float* prow = malloc(sizeof(float) * VIDEO_ROW_DIM);
        long r = s_text + n_audio;
        for (int k = 0; k < n_video; k++) {
            for (int i = 0; i < VIDEO_ROW_DIM; i++) prow[i] = st->xstate[k * 96 + i];
            f32_gemv(T("video_patch_proj.weight"), VIDEO_ROW_DIM, HID,
                     prow, &stream[r * HID]);
            for (int i = 0; i < HID; i++)
                stream[r * HID + i] += ((const float*)T("video_patch_proj.bias"))[i];
            r++;
        }
        free(prow);
    }

    /* ---- sigma-dependent t-embeddings + AdaLN rows ---- */
    {
        double sigma_a = shift_map(sigma_v, 12.0, 3.0);
        double t_arr[2] = { 1.0 - sigma_v, 1.0 - sigma_a };
        const float* raw = (const float*)T("adaln_t_table");
        int T_DIM = st->T_DIM; long TBL_ROWS = st->TBL_ROWS;
        for (int m = 0; m < 2; m++) {
            double pv = t_arr[m] * ((double)TBL_ROWS - 1.0);
            int p0 = (int)pv; if (p0 > (int)TBL_ROWS - 2) p0 = (int)TBL_ROWS - 2;
            double pf = pv - p0;
            for (int k = 0; k < T_DIM; k++)
                g_t_emb_m[m][k] = (float)(
                    (1.0 - pf) * raw[(size_t)p0 * T_DIM + k]
                    + pf * raw[(size_t)(p0 + 1) * T_DIM + k]);
        }
    }

    /* ---- 50-block loop: identical sequence to run_h3_forward_main ---- */
    float (*adaln6)[6][6][HID] = st->adaln6;
    const int T_DIM = st->T_DIM;
    for (int li = 0; li < NL; li++) {
        char nm[160];
        /* AdaLN rows for THIS layer, computed lazily (0.74MB live table) */
        snprintf(nm, sizeof nm, "blocks.%d.adaln_proj.linear.weight", li);
        const GgufTensor* w = TT(nm);
        snprintf(nm, sizeof nm, "blocks.%d.adaln_proj.linear.bias", li);
        const GgufTensor* bts = TT(nm);
        const uint16_t* wb = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* bb = (const uint16_t*)(G.mmap_base + bts->offset);
        for (int row = 0; row < 6; row++) {
            const float* emb_row = g_t_emb_m[row / 3];
            for (int cidx = 0; cidx < 6 * HID; cidx++) {
                const uint16_t* wrow = wb + (size_t)cidx * T_DIM;
                double acc = (double)f16v(bb[cidx]);
                for (int k = 0; k < T_DIM; k++)
                    acc += (double)f16v(wrow[k]) * (double)emb_row[k];
                int chunk = cidx / HID, i = cidx % HID;
                adaln6[0][row][chunk][i] = (float)acc;
            }
        }
        /* rows: video=0 (m0 tag0), text=1 (m0 tag1), audio=5 (m1 tag2) */
        double h3p_t0 = 0; int h3p_fresh = 1; (void)h3p_t0; (void)h3p_fresh;
        PROF_BEGINSLOT(0);
        float* shift_msa = adaln6[0][0][0];
        float* scale_msa = adaln6[0][0][1];
        float* gate_msa  = adaln6[0][0][2];
        float* shift_mlp = adaln6[0][0][3];
        float* scale_mlp = adaln6[0][0][4];
        float* gate_mlp  = adaln6[0][0][5];
        float* shift_msa_t = adaln6[0][1][0];
        float* scale_msa_t = adaln6[0][1][1];
        float* gate_msa_t  = adaln6[0][1][2];
        float* shift_mlp_t = adaln6[0][1][3];
        float* scale_mlp_t = adaln6[0][1][4];
        float* gate_mlp_t  = adaln6[0][1][5];
        float* shift_msa_a = adaln6[0][5][0];
        float* scale_msa_a = adaln6[0][5][1];
        float* gate_msa_a  = adaln6[0][5][2];
        float* shift_mlp_a = adaln6[0][5][3];
        float* scale_mlp_a = adaln6[0][5][4];
        float* gate_mlp_a  = adaln6[0][5][5];

        memcpy(xres, stream, sizeof(float) * seq_len * HID);
        PROF_ENDSLOT(0);
        snprintf(nm, sizeof nm, "blocks.%d.norm1.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < seq_len; s++) {
                float* r = &stream[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++)
                    r[i] = r[i] * sc * bfv(g16[i])
                         * (1.0f + (tag[s] == 0 ? scale_msa[i]
                                  : tag[s] == 1 ? scale_msa_t[i]
                                  : scale_msa_a[i]))
                         + (tag[s] == 0 ? shift_msa[i]
                            : tag[s] == 1 ? shift_msa_t[i] : shift_msa_a[i]);
            }
        }
        snprintf(nm, sizeof nm, "blocks.%d.attn.qkv_proj.weight", li);
        PROF_BEGINSLOT(1);
        mixed_gemv_batch_strided(TT(nm), HID, QKV, seq_len, stream, qkv,
                                 HID, FF1);
        PROF_ENDSLOT(1);
        /* fused qk-norm + split-half rope, per-tag AdaLN'd rows already in */
        snprintf(nm, sizeof nm, "blocks.%d.attn.q_norm.weight", li);
        const uint16_t* qw = (const uint16_t*)T(nm);
        snprintf(nm, sizeof nm, "blocks.%d.attn.k_norm.weight", li);
        const uint16_t* kw = (const uint16_t*)T(nm);
        PROF_BEGINSLOT(2);
        for (long s = 0; s < seq_len; s++) {
            for (int h = 0; h < HEADS; h++) {
                float* qp = &qkv[s * FF1 + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)qp[d] * qp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) qp[d] *= sc * bfv(qw[d]);
            }
            for (int h = 0; h < HEADS; h++) {
                float* kp = &qkv[s * FF1 + comp + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)kp[d] * kp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-5));
                for (int d = 0; d < HD; d++) kp[d] *= sc * bfv(kw[d]);
            }
            for (int j = 0; j < ROT_PAIRS; j++) {
                int axis = j / 16, kbase = j % 16;
                float ang = (float)pos[s * 3 + axis] * inv_freq[kbase];
                float c = cosf(ang), sn = sinf(ang);
                for (int h = 0; h < HEADS; h++) {
                    float* qp = &qkv[s * FF1 + h * HD];
                    float a0 = qp[j], a1 = qp[HD / 2 + j];
                    qp[j]        = a0 * c - a1 * sn;
                    qp[HD / 2 + j] = a0 * sn + a1 * c;
                    float* kp = &qkv[s * FF1 + comp + h * HD];
                    a0 = kp[j]; a1 = kp[HD / 2 + j];
                    kp[j]        = a0 * c - a1 * sn;
                    kp[HD / 2 + j] = a0 * sn + a1 * c;
                }
            }
        }

        /* bidirectional attention (full packed stream) */
        PROF_ENDSLOT(2);
        float scale2 = 1.0f / sqrtf((float)HD);
        float* lgd = NULL;   /* CPU-attention scratch; NULL on the GPU path */
#ifdef STRATUM_USE_METAL
        int attn_minseq = 512;
            { const char* e = getenv("H3_ATTN_MINSEQ"); if (e) attn_minseq = atoi(e); }
            if (g_h3_nc && g_metal_ready && seq_len >= attn_minseq) {
            /* GPU flash attention IN PLACE over fbuf (no gather/staging) */
            static int attn_used = 0;
            int attn_layer_max = 1000000;
            { const char* e = getenv("H3_ATTN_LAYERS"); if (e) attn_layer_max = atoi(e); }
            if (attn_used < attn_layer_max) {
                attn_used++;
                stratum_metal_nc_batch_begin();
                int arc = stratum_metal_nc_batch_attn_packed(fbuf, FF1,
                                    0, comp, 2 * comp, QKV, attn,
                                    (int)seq_len, HEADS, HD, scale2);
                stratum_metal_nc_batch_flush();
                if (arc == 0)
                    goto attn_done;
            }
        }
#endif
        {   /* strided CPU attention over fbuf rows [q|k|v|out] */
            float* att_row0 = fbuf + QKV;
            for (long s = 0; s < seq_len; s++)
                memset(att_row0 + s * FF1, 0, sizeof(float) * comp);
        }
        lgd = malloc(sizeof(float) * seq_len);
        for (int h = 0; h < HEADS; h++)
            for (long a = 0; a < seq_len; a++) {
                const float* qh = &qkv[a * FF1 + h * HD];
                for (long b2 = 0; b2 < seq_len; b2++) {
                    const float* kh = &qkv[b2 * FF1 + comp + h * HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++)
                        dot += (double)qh[d] * kh[d];
                    lgd[b2] = (float)(dot * scale2);
                }
                float mx = lgd[0];
                for (long j = 1; j < seq_len; j++) if (lgd[j] > mx) mx = lgd[j];
                double se = 0;
                for (long j = 0; j < seq_len; j++) {
                    lgd[j] -= mx; se += exp((double)lgd[j]);
                }
                float inv = (float)(1.0 / se);
                float* oh = &attn[a * FF1 + h * HD];
                for (long b2 = 0; b2 < seq_len; b2++) {
                    float pv = expf(lgd[b2]) * inv;
                    const float* vh = &qkv[b2 * FF1 + 2 * comp + h * HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv * vh[d];
                }
            }

        attn_done:;
        PROF_ENDSLOT(3);
        snprintf(nm, sizeof nm, "blocks.%d.attn.out_proj.weight", li);
        PROF_BEGINSLOT(4);
        mixed_gemv_batch_strided(TT(nm), comp, HID, seq_len, attn, proj,
                                 FF1, HID);
        PROF_ENDSLOT(4);
        PROF_BEGINSLOT(0);
        if (getenv("H3_ATTN_PROBE") && li == 2) {
            long b1=0; double m1=0;
            for (long s = 0; s < seq_len; s++)
                for (int t = 0; t < comp; t++) {
                    float v = attn[s * FF1 + t];
                    if (!isfinite(v)) b1++; else if (fabs(v)>m1) m1=fabs(v);
                }
            long b2=0; double m2=0;
            for (long t = 0; t < seq_len * HID; t++)
                if (!isfinite(proj[t])) b2++; else if (fabs(proj[t])>m2) m2=fabs(proj[t]);
            fprintf(stderr, "  [L2 probe] attn nonfinite=%ld max=%.4g | proj nonfinite=%ld max=%.4g\n",
                    b1, m1, b2, m2);
        }
        for (long s = 0; s < seq_len; s++) {
            const float* g = tag[s] == 0 ? gate_msa
                           : tag[s] == 1 ? gate_msa_t : gate_msa_a;
            for (int i = 0; i < HID; i++)
                stream[s * HID + i] = xres[s * HID + i] + g[i] * proj[s * HID + i];
        }
        PROF_ENDSLOT(0);

        /* mlp */
        PROF_BEGINSLOT(0);
        memcpy(xres, stream, sizeof(float) * seq_len * HID);
        PROF_ENDSLOT(0);
        PROF_BEGINSLOT(5);
        snprintf(nm, sizeof nm, "blocks.%d.norm2.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < seq_len; s++) {
                float* r = &stream[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++)
                    r[i] = r[i] * sc * bfv(g16[i])
                         * (1.0f + (tag[s] == 0 ? scale_mlp[i]
                                  : tag[s] == 1 ? scale_mlp_t[i]
                                  : scale_mlp_a[i]))
                         + (tag[s] == 0 ? shift_mlp[i]
                            : tag[s] == 1 ? shift_mlp_t[i] : shift_mlp_a[i]);
            }
        }
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc1.weight", li);
        const GgufTensor* t_fc1 = TT(nm);
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc2.weight", li);
        const GgufTensor* t_fc2 = TT(nm);
        int mlp_fused = 0;
#ifdef STRATUM_USE_METAL
        static int fused_en = -1;
        if (fused_en < 0) {
            const char* e = getenv("H3_MLP_FUSED");
            fused_en = (e && atoi(e) == 0) ? 0 : 1;
        }
        if (fused_en && g_h3_nc && g_metal_ready &&
            (t_fc1->type == 12 || t_fc1->type == 14) &&
            t_fc2->type == t_fc1->type) {
            stratum_metal_nc_batch_begin();
            int rc = stratum_metal_nc_mlp_fused(
                (const void*)(G.mmap_base + t_fc1->offset), (size_t)t_fc1->nbytes,
                (int)t_fc1->type,
                (const void*)(G.mmap_base + t_fc2->offset), (size_t)t_fc2->nbytes,
                stream, fc1o, FF1, proj, (int)seq_len, HID, FF2);
            stratum_metal_nc_batch_flush();
            mlp_fused = (rc == 0);
        }
#endif
        if (!mlp_fused) {
        PROF_BEGINSLOT(6);
        mixed_gemv_batch_strided(t_fc1, HID, FF1, seq_len, stream, fc1o,
                                 HID, FF1);
        PROF_ENDSLOT(6);
        PROF_BEGINSLOT(7);
        for (long s = 0; s < seq_len; s++)
            for (int i = 0; i < FF2; i++) {
                float gv = fc1o[s * FF1 + i];
                fc1o[s * FF1 + i] =
                    (gv / (1.0f + expf(-gv))) * fc1o[s * FF1 + FF2 + i];
            }
        PROF_ENDSLOT(7);
        PROF_BEGINSLOT(7);
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc2.weight", li);
        mixed_gemv_batch_strided(TT(nm), FF2, HID, seq_len, fc1o, proj,
                                 FF1, HID);
        PROF_ENDSLOT(7);
        }
        if (getenv("H3_MLP_PROBE") && li == 0) {
            FILE* fp = fopen("/tmp/mlp_proj_probe.bin", "wb");
            if (fp) { fwrite(proj, 4, (size_t)seq_len * HID, fp); fclose(fp); }
        }
        PROF_BEGINSLOT(0);
        for (long s = 0; s < seq_len; s++) {
            const float* g = tag[s] == 0 ? gate_mlp
                           : tag[s] == 1 ? gate_mlp_t : gate_mlp_a;
            for (int i = 0; i < HID; i++)
                stream[s * HID + i] =
                    xres[s * HID + i] + g[i] * proj[s * HID + i];
        }
        PROF_ENDSLOT(0);

        free(lgd);
        lgd = NULL;
        if (getenv("H3_ATTN_CHK")) {
            long bad = 0; double bmax = 0;
            for (long t = 0; t < seq_len * HID; t++) {
                if (!isfinite(stream[t])) bad++;
                else if (fabs(stream[t]) > bmax) bmax = fabs(stream[t]);
            }
            fprintf(stderr, "  [li=%d] stream nonfinite=%ld max=%.4g\n", li, bad, bmax);
        }
        if (li % 10 == 0 || li == NL - 1 || getenv("H3_ATTN_CHK")) {
            fprintf(stderr, "  L%02d\n", li);
            act_stats("post-mlp", stream, seq_len * HID);
        }
    }

    /* --- M4b: final_layer — norm + its own AdaLN (expand=2, modalities=1)
     * + video_out/audio_out F32 heads. THIS is the velocity in patch
     * space; the raw hidden state is not the model output. --- */
    {
        float (*fshift)[HID] = malloc(sizeof(float[2][HID]));
        float (*fscale)[HID] = malloc(sizeof(float[2][HID]));
        const GgufTensor* w = TT("final_layer.adaln_proj.linear.weight");
        const GgufTensor* bts = TT("final_layer.adaln_proj.linear.bias");
        const uint16_t* wb = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* bb = (const uint16_t*)(G.mmap_base + bts->offset);
        for (int m = 0; m < 2; m++) {
            for (int cidx = 0; cidx < 2 * HID; cidx++) {
                const uint16_t* wrow = wb + (size_t)cidx * T_DIM;
                double acc = (double)f16v(bb[cidx]);
                for (int k = 0; k < T_DIM; k++)
                    acc += (double)f16v(wrow[k]) * (double)g_t_emb_m[m][k];
                int chunk = cidx / HID, i = cidx % HID;
                if (chunk == 0) fshift[m][i] = (float)acc;
                else fscale[m][i] = (float)acc;
            }
        }
        const uint16_t* g16 = (const uint16_t*)T("final_layer.norm.weight");
        /* video rows (m=0) then audio rows (m=1) */
        float* vvel = malloc(sizeof(float) * (size_t)n_video * 96);
        float* avel = malloc(sizeof(float) * (size_t)n_audio * 32);
        long vrow = 0, arow2 = 0;
        for (long s = 0; s < seq_len; s++) {
            if (tag[s] != 0 && tag[s] != 2) continue;
            float* r = &stream[s * HID];
            double ss = 0;
            for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
            float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
            int m = tag[s] == 0 ? 0 : 1;
            for (int i = 0; i < HID; i++)
                r[i] = r[i] * sc * bfv(g16[i]) * (1.0f + fscale[m][i])
                     + fshift[m][i];
            if (tag[s] == 0) {
                f32_gemv(T("final_layer.video_out.weight"), HID, 96, r,
                         &vvel[vrow * 96]);
                for (int i = 0; i < 96; i++)
                    vvel[vrow * 96 + i] +=
                        ((const float*)T("final_layer.video_out.bias"))[i];
                vrow++;
            } else {
                f32_gemv(T("final_layer.audio_out.weight"), HID, 32, r,
                         &avel[arow2 * 32]);
                for (int i = 0; i < 32; i++)
                    avel[arow2 * 32 + i] +=
                        ((const float*)T("final_layer.audio_out.bias"))[i];
                arow2++;
            }
        }
        memcpy(st->xnext, vvel, sizeof(float) * (size_t)n_video * 96);
        free(vvel); free(avel); free(fshift); free(fscale);
    }

    return 0;
}
