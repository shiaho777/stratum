/*
 * stratum_linear.h — Generic quantized linear layer dispatch.
 *
 * Eliminates the duplication between la_linear_* and q35_linear_* functions.
 * All architectures share the same quantized matmul kernels, selected by
 * GGML tensor type. No model-specific logic here.
 *
 * Usage:
 *   1. Call stratum_linear_init() once at startup (sets nchunks, SDOT, GPU)
 *   2. Use st_linear_dispatch(w, x, y, N, K) for single-input matmul
 *   3. Use st_linear_q4k_multix() for batched (spec decode) matmul
 *   4. Use st_q4k_group() for fused multi-tensor matmul
 */
#ifndef STRATUM_LINEAR_H
#define STRATUM_LINEAR_H

#include <stdatomic.h>
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include "stratum_q5k.h"
#include "stratum_q5k_neon.h"
#include "stratum_q6k.h"
#include "stratum_q6k_neon.h"
#include "stratum_q3k.h"
#include "stratum_q3k_neon.h"
#include "stratum_q2k.h"
#include "stratum_q2k_neon.h"
#include "stratum_q8_0.h"
#include "stratum_bf16.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/sysctl.h>

#ifdef STRATUM_USE_METAL
#include "stratum_metal.h"
#endif

/* ------------------------------------------------------------------ */
/*  Engine state — shared across all architectures                     */
/* ------------------------------------------------------------------ */

typedef struct {
    /* GGUF model data (mmap base) */
    const uint8_t* mmap_base;
    size_t         mmap_size;

    /* CPU parallelism */
    int nchunks;        /* dispatch_apply thread count */

    /* SDOT (ARM dotprod) — default ON for Q4_K/Q6_K */
    int use_sdot;

    /* AMX W16 multiseq path (type-43 tensors); STRATUM_AMX_W16_OFF=1 disables */
    int amx_w16_off;

    /* GPU */
    int use_metal;
    int gpu2;           /* STRATUM_GPU_FULL=1 */
    int gpu3;           /* STRATUM_GPU_FULL=2 (qwen35 SSM dispatch) */
    int gpu2_minrows;

    /* Memory management */
    int keep_resident;  /* hot mode: skip madvise, page cache stays warm */
    int hot_enabled;

    /* Profiling */
    double  typesecs[GGML_TYPE_COUNT];
    long    typecalls[GGML_TYPE_COUNT];
} StratumLinearState;

/* Global state — accessible by all architectures */
extern StratumLinearState g_st;

/* ------------------------------------------------------------------ */
/*  Initialization                                                     */
/* ------------------------------------------------------------------ */

static inline void stratum_linear_init(const uint8_t* mmap_base, size_t mmap_size) {
    g_st.mmap_base = mmap_base;
    g_st.mmap_size = mmap_size;

    /* CPU detection */
    int ncpu = 0; size_t l = sizeof(ncpu);
    if (sysctlbyname("hw.physicalcpu", &ncpu, &l, NULL, 0) != 0 || ncpu < 1) ncpu = 8;
    int pcpu = 0; size_t pl = sizeof(pcpu);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcpu, &pl, NULL, 0) != 0 || pcpu < 1) pcpu = ncpu;

    const char* env_nc = getenv("STRATUM_NCHUNKS");
    if (env_nc) {
        g_st.nchunks = atoi(env_nc);
    } else {
        /* Measured on M4 Pro: models that fit the page cache hot saturate
         * unified-memory bandwidth at ~5 chunks — beyond that fork-join
         * overhead and contention lose (~112 vs ~79 tok/s on a 0.5 GB
         * model). Large models streamed cold still want max parallelism
         * for NVMe queue depth. STRATUM_NCHUNKS always wins. */
        g_st.nchunks = (mmap_size <= (size_t)2 * 1024 * 1024 * 1024) ? 5 : pcpu;
    }
    if (g_st.nchunks < 1) g_st.nchunks = 1;

    /* SDOT default ON (V25+). STRATUM_NO_SDOT (any set value) or
     * STRATUM_SDOT=0 disables; explicit STRATUM_SDOT=1 wins over both. */
    { const char* noe = getenv("STRATUM_NO_SDOT");
      const char* se  = getenv("STRATUM_SDOT");
      g_st.use_sdot = se ? (atoi(se) != 0) : (noe == NULL); }
    g_st.amx_w16_off = (getenv("STRATUM_AMX_W16_OFF") != NULL);

    /* GPU */
    g_st.use_metal = (getenv("STRATUM_GPU") != NULL) ? 1 : 0;
    g_st.gpu2 = 0; g_st.gpu3 = 0; g_st.gpu2_minrows = 0;
    const char* env_gf = getenv("STRATUM_GPU_FULL");
    if (env_gf) {
        int v = atoi(env_gf);
        if (v >= 1) g_st.gpu2 = 1;
        if (v >= 2) g_st.gpu3 = 1;
    }

    g_st.keep_resident = 0;
    g_st.hot_enabled = 0;

    memset(g_st.typesecs, 0, sizeof(g_st.typesecs));
    memset(g_st.typecalls, 0, sizeof(g_st.typecalls));
}

/* ------------------------------------------------------------------ */
/*  Tensor data access                                                 */
/* ------------------------------------------------------------------ */

/* Row pointer helpers — type-specific, shared by all architectures.
 * (qwen35 keeps its own q35_g_tensor_live override for staging; these
 * helpers always read the mmap directly.) */

static inline const block_q4_K* st_q4k_row_ptr(const GgufTensor* t, int K, int r) {
    return (const block_q4_K*)(g_st.mmap_base + t->offset)
         + (size_t)r * (K / 256);
}
static inline const block_q5_K* st_q5k_row_ptr(const GgufTensor* t, int K, int r) {
    return (const block_q5_K*)(g_st.mmap_base + t->offset)
         + (size_t)r * (K / 256);
}
static inline const block_q6_K* st_q6k_row_ptr(const GgufTensor* t, int K, int r) {
    return (const block_q6_K*)(g_st.mmap_base + t->offset)
         + (size_t)r * (K / 256);
}
static inline const block_q3_K* st_q3k_row_ptr(const GgufTensor* t, int K, int r) {
    return (const block_q3_K*)(g_st.mmap_base + t->offset)
         + (size_t)r * (K / 256);
}
static inline const block_q2_K* st_q2k_row_ptr(const GgufTensor* t, int K, int r) {
    return (const block_q2_K*)(g_st.mmap_base + t->offset)
         + (size_t)r * (K / 256);
}
static inline const block_q8_0* st_q8_0_row_ptr(const GgufTensor* t, int K, int r) {
    return (const block_q8_0*)(g_st.mmap_base + t->offset)
         + (size_t)r * (K / 32);
}
static inline const float* st_f32_tensor_ptr(const GgufTensor* t) {
    return (const float*)(g_st.mmap_base + t->offset);
}

/* ------------------------------------------------------------------ */
/*  Parallel row dispatch macro — shared by all architectures          */
/* ------------------------------------------------------------------ */

#define ST_PAR_ROWS(N, body) \
    do { \
        int _N = (N); \
        int _T = g_st.nchunks; \
        int _tmax = (_N + 255) / 256;   /* >=256 rows/chunk: measured peak \
                                         * at ~4-5 chunks on small hot \
                                         * models; >6 loses to fork-join \
                                         * overhead + bandwidth contention */ \
        if (_T > _tmax) _T = _tmax; \
        if (_T > _N) _T = _N; \
        if (_T < 1) _T = 1; \
        /* dynamic 64-row work tickets: a preempted worker no longer \
         * stalls the join — output is identical (per-row independence) */ \
        _Atomic int _next_store = 0; \
        _Atomic int* _next = &_next_store; \
        dispatch_apply(_T, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), \
            ^(size_t _t) { \
                (void)_t; \
                for (;;) { \
                    int _s = atomic_fetch_add_explicit(_next, 64, memory_order_relaxed); \
                    if (_s >= _N) break; \
                    int _e = _s + 64; \
                    if (_e > _N) _e = _N; \
                    for (int r = _s; r < _e; r++) { body; } \
                } \
            }); \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Single-input linear dispatch (one x → one y)                       */
/* ------------------------------------------------------------------ */

static inline void st_linear_q4k(const GgufTensor* w, const float* x, float* y, int N, int K) {
#ifdef STRATUM_USE_METAL
    if (g_st.use_metal && w->offset > 0 && N < 50000) {
        if (stratum_metal_q4k_sgemv(w->offset, x, y, N, K) == 0) return;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_st.use_sdot) {
        int nb = K / 32;
        int8_t* xq = (int8_t*)alloca((size_t)K);
        float*  xs = (float*)alloca((size_t)nb * sizeof(float));
        int32_t* xsum = (int32_t*)alloca((size_t)nb * sizeof(int32_t));
        q4k_quantize_x_q8(x, K, xq, xs);
        for (int g = 0; g < nb; g++) xsum[g] = q4k_sum_i8_32(xq + (size_t)g * 32);
        ST_PAR_ROWS(N, y[r] = q4k_dot_row_sdot_f(st_q4k_row_ptr(w, K, r), K, xq, xs, xsum));
        return;
    }
#endif
    ST_PAR_ROWS(N, y[r] = q4k_dot_row_neon(st_q4k_row_ptr(w, K, r), K, x));
}

static inline void st_linear_q6k(const GgufTensor* w, const float* x, float* y, int N, int K) {
#ifdef STRATUM_USE_METAL
    if (g_st.use_metal && w->offset > 0) {
        if (stratum_metal_q6k_sgemv(w->offset, x, y, N, K) == 0) return;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_st.use_sdot) {
        int ng = K / 16;
        int8_t* xq = (int8_t*)alloca((size_t)K);
        float*  xs = (float*)alloca((size_t)ng * sizeof(float));
        q6k_quantize_x_q8_g16(x, K, xq, xs);
        ST_PAR_ROWS(N, y[r] = q6k_dot_row_sdot_f(st_q6k_row_ptr(w, K, r), K, xq, xs));
        return;
    }
#endif
    ST_PAR_ROWS(N, y[r] = q6k_dot_row_neon(st_q6k_row_ptr(w, K, r), K, x));
}

static inline void st_linear_q5k(const GgufTensor* w, const float* x, float* y, int N, int K) {
#ifdef STRATUM_USE_METAL
    if (g_st.use_metal && w->offset > 0) {
        if (stratum_metal_q5k_sgemv(w->offset, x, y, N, K) == 0) return;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_st.use_sdot) {
        int nb = K / 32;
        int8_t* xq = (int8_t*)alloca((size_t)K);
        float*  xs = (float*)alloca((size_t)nb * sizeof(float));
        int32_t* xsum = (int32_t*)alloca((size_t)nb * sizeof(int32_t));
        q4k_quantize_x_q8(x, K, xq, xs);
        for (int g = 0; g < nb; g++) xsum[g] = q4k_sum_i8_32(xq + (size_t)g * 32);
        ST_PAR_ROWS(N, y[r] = q5k_dot_row_sdot_f(st_q5k_row_ptr(w, K, r), K, xq, xs, xsum));
        return;
    }
#endif
    ST_PAR_ROWS(N, y[r] = q5k_dot_row_neon(st_q5k_row_ptr(w, K, r), K, x));
}

static inline void st_linear_q3k(const GgufTensor* w, const float* x, float* y, int N, int K) {
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_st.use_sdot) {
        int nb = K / 32;
        int8_t* xq = (int8_t*)alloca((size_t)K);
        float*  xs = (float*)alloca((size_t)nb * sizeof(float));
        q4k_quantize_x_q8(x, K, xq, xs);
        ST_PAR_ROWS(N, y[r] = q3k_dot_row_sdot_f(st_q3k_row_ptr(w, K, r), K, xq, xs));
        return;
    }
#endif
    ST_PAR_ROWS(N, y[r] = q3k_dot_row_neon(st_q3k_row_ptr(w, K, r), K, x));
}

static inline void st_linear_q2k(const GgufTensor* w, const float* x, float* y, int N, int K) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    ST_PAR_ROWS(N, y[r] = q2k_dot_row_neon(st_q2k_row_ptr(w, K, r), K, x));
#else
    ST_PAR_ROWS(N, y[r] = q2k_dot_row_scalar(st_q2k_row_ptr(w, K, r), K, x));
#endif
}

#ifndef ST_B_MAX
#define ST_B_MAX 8
#endif

static inline void st_linear_q2k_multix(const GgufTensor* w,
                                        const float* const* xs,
                                        float* const* ys,
                                        int B, int N, int K) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float* const* xsl = xs;
    float* const* ysl = ys;
    ST_PAR_ROWS(N, {
        const float* xrow[ST_B_MAX];
        for (int s = 0; s < B; s++) xrow[s] = xsl[s];
        float out[ST_B_MAX];
        q2k_dot_row_neon_multix(st_q2k_row_ptr(w, K, r), K, xrow, B, out);
        for (int s = 0; s < B; s++) ysl[s][r] = out[s];
    });
#else
    for (int s = 0; s < B; s++) st_linear_q2k(w, xs[s], ys[s], N, K);
#endif
}

static inline void st_linear_q8_0(const GgufTensor* w, const float* x, float* y, int N, int K) {
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_st.use_sdot) {
        int nb = K / 32;
        int8_t* xq = (int8_t*)alloca((size_t)K);
        float*  xs = (float*)alloca((size_t)nb * sizeof(float));
        q4k_quantize_x_q8(x, K, xq, xs);
        ST_PAR_ROWS(N, y[r] = q8_0_dot_row_sdot_f(st_q8_0_row_ptr(w, K, r), K, xq, xs));
        return;
    }
#endif
    ST_PAR_ROWS(N, y[r] = q8_0_dot_row_neon(st_q8_0_row_ptr(w, K, r), K, x));
}

static inline void st_linear_f16(const GgufTensor* w, const float* x, float* y, int N, int K) {
    const uint16_t* raw = (const uint16_t*)(g_st.mmap_base + w->offset);
    ST_PAR_ROWS(N, {
        const uint16_t* row = raw + (size_t)r * K;
        double acc = 0.0;
        for (int c = 0; c < K; c++) acc += (double)q4k_fp16_to_fp32(row[c]) * x[c];
        y[r] = (float)acc;
    });
}

static inline void st_linear_f32(const GgufTensor* w, const float* x, float* y, int N, int K) {
    const float* raw = st_f32_tensor_ptr(w);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K,
                1.0f, raw, K, x, 1, 0.0f, y, 1);
}

/* Main dispatch — select kernel by tensor type. No model-specific logic. */
/* type-43 W16 engine lives in stratum_q4k_w16_amx.h (included at the bottom
 * of this file); forward-declare so the single-seq dispatch can reach it. */
static int  st_amx_available(void);
static void st_q4k_w16_matvec_fallback(const GgufTensor*, const float* const*,
                                       float* const*, int, int, int);
static inline void st_q4k_w16_amx_multix(const GgufTensor*, const float* const*,
                                         float* const*, int, int, int);

static inline int st_linear_dispatch(const GgufTensor* w, const float* x, float* y,
                                     int N, int K) {
    int _tm = (getenv("STRATUM_TYPETIME") != NULL);
    struct timespec _a, _b;
    if (_tm) clock_gettime(CLOCK_MONOTONIC, &_a);
    int _t = (int)w->type;
    int _rc = 0;
    switch ((GgmlType)w->type) {
        case GGML_TYPE_Q2_K: st_linear_q2k (w, x, y, N, K); break;
        case GGML_TYPE_Q3_K: st_linear_q3k (w, x, y, N, K); break;
        case GGML_TYPE_Q4_K: st_linear_q4k (w, x, y, N, K); break;
        case GGML_TYPE_Q5_K: st_linear_q5k (w, x, y, N, K); break;
        case GGML_TYPE_Q6_K: st_linear_q6k (w, x, y, N, K); break;
        case GGML_TYPE_Q8_0: st_linear_q8_0(w, x, y, N, K); break;
        case GGML_TYPE_F16:  st_linear_f16 (w, x, y, N, K); break;
        case GGML_TYPE_F32:  st_linear_f32 (w, x, y, N, K); break;
        case GGML_TYPE_Q4K_W16: {
            const float* x1[1]; float* y1[1];
            x1[0] = x; y1[0] = y;
            if (!g_st.amx_w16_off && (N % 32) == 0 && (K % 256) == 0
                && st_amx_available())
                st_q4k_w16_amx_multix(w, x1, y1, 1, N, K);
            else
                st_q4k_w16_matvec_fallback(w, x1, y1, 1, N, K);
            break;
        }
        default:
            fprintf(stderr, "st_linear_dispatch: unsupported type %s\n",
                    gguf_type_name((GgmlType)w->type));
            _rc = -1;
    }
    if (_tm && _rc == 0 && _t >= 0 && _t < GGML_TYPE_COUNT) {
        clock_gettime(CLOCK_MONOTONIC, &_b);
        g_st.typesecs[_t] += (_b.tv_sec-_a.tv_sec)+(_b.tv_nsec-_a.tv_nsec)/1e9;
        g_st.typecalls[_t]++;
    }
    return _rc;
}

/* ------------------------------------------------------------------ */
/*  Batched multi-input linear (spec decode — B inputs share weights)  */
/* ------------------------------------------------------------------ */

#if defined(__ARM_FEATURE_DOTPROD)
static inline void st_linear_q4k_multix(const GgufTensor* w,
                                         const float* const* xs,
                                         float* const* ys,
                                         int B, int N, int K) {
    if (g_st.use_sdot) {
        int nb = K / 32;
        int8_t** xqs = (int8_t**)alloca((size_t)B * sizeof(int8_t*));
        float**  xss = (float**)alloca((size_t)B * sizeof(float*));
        int32_t** xsums = (int32_t**)alloca((size_t)B * sizeof(int32_t*));
        for (int b = 0; b < B; b++) {
            xqs[b] = (int8_t*)malloc((size_t)K);
            xss[b] = (float*)malloc((size_t)nb * sizeof(float));
            xsums[b] = (int32_t*)malloc((size_t)nb * sizeof(int32_t));
            q4k_quantize_x_q8(xs[b], K, xqs[b], xss[b]);
            for (int g = 0; g < nb; g++)
                xsums[b][g] = q4k_sum_i8_32(xqs[b] + (size_t)g * 32);
        }
        ST_PAR_ROWS(N, {
            const block_q4_K* row = st_q4k_row_ptr(w, K, r);
            for (int b = 0; b < B; b++)
                ys[b][r] = q4k_dot_row_sdot_f(row, K, xqs[b], xss[b], xsums[b]);
        });
        for (int b = 0; b < B; b++) { free(xqs[b]); free(xss[b]); free(xsums[b]); }
        return;
    }
#endif
    /* Fallback: independent dispatches */
    for (int b = 0; b < B; b++)
        ST_PAR_ROWS(N, ys[b][r] = q4k_dot_row_neon(st_q4k_row_ptr(w, K, r), K, xs[b]));
}

#if defined(__ARM_FEATURE_DOTPROD)
static inline void st_linear_q6k_multix(const GgufTensor* w,
                                         const float* const* xs,
                                         float* const* ys,
                                         int B, int N, int K) {
    if (g_st.use_sdot) {
        int ng = K / 16;
        int8_t** xqs = (int8_t**)alloca((size_t)B * sizeof(int8_t*));
        float**  xss = (float**)alloca((size_t)B * sizeof(float*));
        for (int b = 0; b < B; b++) {
            xqs[b] = (int8_t*)malloc((size_t)K);
            xss[b] = (float*)malloc((size_t)ng * sizeof(float));
            q6k_quantize_x_q8_g16(xs[b], K, xqs[b], xss[b]);
        }
        ST_PAR_ROWS(N, {
            const block_q6_K* row = st_q6k_row_ptr(w, K, r);
            for (int b = 0; b < B; b++)
                ys[b][r] = q6k_dot_row_sdot_f(row, K, xqs[b], xss[b]);
        });
        for (int b = 0; b < B; b++) { free(xqs[b]); free(xss[b]); }
        return;
    }
#endif
    for (int b = 0; b < B; b++)
        ST_PAR_ROWS(N, ys[b][r] = q6k_dot_row_neon(st_q6k_row_ptr(w, K, r), K, xs[b]));
}

/* ------------------------------------------------------------------ */
/*  Unified batched dispatch — type-43 W16 goes to the AMX engine,      */
/*  the rest to their per-type multix (or per-seq fallback).            */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Generic batched multi-input linear — shared by all architectures    */
/*  Q4_K/Q6_K take the SDOT register-sharing pack path (activations     */
/*  quantized once per stream into [group][stream][32], each row's      */
/*  nibbles unpacked once for all B streams); other types fall back     */
/*  to per-stream st_linear_dispatch (numerics identical to single).    */
/* ------------------------------------------------------------------ */

#define ST_MS_BMAX 32

static inline void st_linear_multix(const GgufTensor* w,
                                    const float* const* xs,
                                    float* const* ys,
                                    int B, int N, int K) {
    if (w->type == GGML_TYPE_Q4K_W16) {
#if defined(__ARM_NEON) || defined(__aarch64__)
        if (!g_st.amx_w16_off && B >= 2 && (N % 32) == 0 && (K % 256) == 0
            && st_amx_available()) {
            st_q4k_w16_amx_multix(w, xs, ys, B, N, K);
            return;
        }
#endif
        st_q4k_w16_matvec_fallback(w, xs, ys, B, N, K);
        return;
    }
    if ((w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K)) {
#if defined(__ARM_FEATURE_DOTPROD)
        if (g_st.use_sdot && B >= 2 && (K % 256) == 0) {
            static int8_t*  st_xpack = NULL;
            static float*   st_scpack = NULL;
            static int32_t* st_sumpack = NULL;
            static size_t   st_pack_cap = 0;
            int ng = K / 32;
            int Bc_max = (B < 16) ? B : 16;
            size_t need_x = (size_t)ng * (size_t)Bc_max * 32;
            if (need_x > st_pack_cap) {
                free(st_xpack); free(st_scpack); free(st_sumpack);
                st_xpack   = (int8_t*) malloc(need_x);
                st_scpack  = (float*)  malloc((size_t)ng * Bc_max * sizeof(float));
                st_sumpack = (int32_t*)malloc((size_t)ng * Bc_max * sizeof(int32_t));
                st_pack_cap = (st_xpack && st_scpack && st_sumpack) ? need_x : 0;
            }
            if (st_pack_cap) {
                for (int s0 = 0; s0 < B; s0 += 16) {
                    int Bc = (B - s0 < 16) ? (B - s0) : 16;
                    for (int s = 0; s < Bc; s++)
                        for (int g = 0; g < ng; g++) {
                            int8_t* dst = st_xpack + ((size_t)g * Bc + s) * 32;
                            float scv;
                            q4k_quantize_x_q8_1b(xs[s0 + s] + (size_t)g * 32, dst, &scv);
                            st_scpack[(size_t)g * Bc + s] = scv;
                            st_sumpack[(size_t)g * Bc + s] = q4k_sum_i8_32(dst);
                        }
                    int np = N / 2, tail = N & 1;
                    int _T = g_st.nchunks; if (_T > np) _T = np; if (_T < 1) _T = 1;
                    int _chunk = (np + _T - 1) / _T;
                    int is_q4 = (w->type == GGML_TYPE_Q4_K);
                    dispatch_apply(_T, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                        ^(size_t _t) {
                            int _s = (int)_t * _chunk, _e = _s + _chunk;
                            if (_e > np) _e = np;
                            for (int p = _s; p < _e; p++) {
                                int r = p * 2;
                                float o0[ST_MS_BMAX], o1[ST_MS_BMAX];
                                if (is_q4)
                                    q4k_dot_rows2_sdot_multix_pack(
                                        st_q4k_row_ptr(w, K, r), st_q4k_row_ptr(w, K, r + 1), K,
                                        st_xpack, st_scpack, st_sumpack, Bc, o0, o1);
                                else
                                    q6k_dot_rows2_sdot_multix_pack(
                                        st_q6k_row_ptr(w, K, r), st_q6k_row_ptr(w, K, r + 1), K,
                                        st_xpack, st_scpack, Bc, o0, o1);
                                for (int s = 0; s < Bc; s++) {
                                    ys[s0 + s][r] = o0[s];
                                    ys[s0 + s][r + 1] = o1[s];
                                }
                            }
                        });
                    if (tail) {
                        int r = N - 1;
                        float out[ST_MS_BMAX];
                        if (is_q4)
                            q4k_dot_row_sdot_multix_pack(st_q4k_row_ptr(w, K, r), K,
                                                         st_xpack, st_scpack, st_sumpack, Bc, out);
                        else
                            q6k_dot_row_sdot_multix_pack(st_q6k_row_ptr(w, K, r), K,
                                                         st_xpack, st_scpack, Bc, out);
                        for (int s = 0; s < Bc; s++) ys[s0 + s][r] = out[s];
                    }
                }
                return;
            }
        }
#endif
    }
    for (int b = 0; b < B; b++)
        st_linear_dispatch(w, xs[b], ys[b], N, K);
}

/* ------------------------------------------------------------------ */
/*  Fused multi-tensor dispatch (Q4_K group — gate/up/down share x)    */
/* ------------------------------------------------------------------ */

#if defined(__ARM_FEATURE_DOTPROD)
static inline void st_q4k_fused_sdot(const float* x, int K,
                                     const GgufTensor* const ws[], float* const ys[],
                                     const int Ns[], int nw) {
    int nb = K / 32;
    int8_t* xq = (int8_t*)alloca((size_t)K);
    float*  xs = (float*)alloca((size_t)nb * sizeof(float));
    int32_t* xsum = (int32_t*)alloca((size_t)nb * sizeof(int32_t));
    q4k_quantize_x_q8(x, K, xq, xs);
    for (int g = 0; g < nb; g++) xsum[g] = q4k_sum_i8_32(xq + (size_t)g * 32);
    int R = 0;
    int* pref = (int*)alloca((size_t)nw * sizeof(int));
    for (int i = 0; i < nw; i++) { pref[i] = R; R += Ns[i]; }
    ST_PAR_ROWS(R, {
        int wi = nw - 1;
        while (wi > 0 && r < pref[wi]) wi--;
        int lr = r - pref[wi];
        ys[wi][lr] = q4k_dot_row_sdot_f(st_q4k_row_ptr(ws[wi], K, lr), K, xq, xs, xsum);
    });
}
#endif

static inline void st_q4k_group(const float* x, int K,
                                const GgufTensor* w0, float* y0, int N0,
                                const GgufTensor* w1, float* y1, int N1,
                                const GgufTensor* w2, float* y2, int N2) {
    const GgufTensor* ws[3]; float* ys[3]; int Ns[3]; int nw = 0;
    ws[nw]=w0; ys[nw]=y0; Ns[nw]=N0; nw++;
    ws[nw]=w1; ys[nw]=y1; Ns[nw]=N1; nw++;
    if (w2) { ws[nw]=w2; ys[nw]=y2; Ns[nw]=N2; nw++; }

#ifdef STRATUM_USE_METAL
    if (g_st.use_metal) {
        int all_q4k = 1;
        for (int i = 0; i < nw; i++)
            if (ws[i]->type != GGML_TYPE_Q4_K || ws[i]->offset == 0) all_q4k = 0;
        if (all_q4k) {
            uint64_t offs[3]; size_t tb[3];
            for (int i = 0; i < nw; i++) { offs[i] = ws[i]->offset; tb[i] = ws[i]->nbytes; }
            if (stratum_metal_q4k_sgemv_group(offs, tb, x, ys, Ns, nw, K) == 0) return;
        }
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_st.use_sdot && !g_st.use_metal) {
        int all_q4k = 1;
        for (int i = 0; i < nw; i++) if (ws[i]->type != GGML_TYPE_Q4_K) all_q4k = 0;
        if (all_q4k) { st_q4k_fused_sdot(x, K, ws, ys, Ns, nw); return; }
    }
#endif
    for (int i = 0; i < nw; i++) st_linear_dispatch(ws[i], x, ys[i], Ns[i], K);
}

/* ------------------------------------------------------------------ */
/*  F16→F32 tensor reader                                              */
/* ------------------------------------------------------------------ */

static inline int st_read_f16_to_f32(const GgufTensor* t, float* out, int n) {
    if ((int)t->nelem != n) {
        fprintf(stderr, "f16 read size mismatch: %d vs %lld\n", n, (long long)t->nelem);
        return -1;
    }
    const uint16_t* raw = (const uint16_t*)(g_st.mmap_base + t->offset);
    for (int i = 0; i < n; i++) out[i] = q4k_fp16_to_fp32(raw[i]);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  RMSNorm — universal (every Transformer architecture uses it)       */
/* ------------------------------------------------------------------ */

static inline void st_rmsnorm(float* x, const float* w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    ss = 1.0 / sqrt(ss / (double)n + (double)eps);
    for (int i = 0; i < n; i++) x[i] = (float)((double)x[i] * ss * (double)w[i]);
}

/* ------------------------------------------------------------------ */
/*  RoPE — universal half-rotation                                     */
/* ------------------------------------------------------------------ */

static inline void st_rope_half(float* x, int n, int rope_dim, int pos, float theta) {
    int half = rope_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2*i) / (float)rope_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = x[i];
        float x1 = x[i + half];
        x[i]       = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

/* Per-head RoPE (applies to each head independently) */
static inline void st_rope_half_heads(float* x, int nheads, int head_dim, int rope_dim,
                                       int pos, float theta) {
    for (int h = 0; h < nheads; h++) {
        st_rope_half(x + h * head_dim, head_dim, rope_dim, pos, theta);
    }
}

/* ------------------------------------------------------------------ */
/*  Q4K_W16 (type 43) AMX multiseq engine                              */
/* ------------------------------------------------------------------ */
#include "stratum_q4k_w16_amx.h"

#endif /* STRATUM_LINEAR_H */
