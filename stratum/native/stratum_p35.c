
/*
 * stratum_p35.c — HISTORICAL ARCHIVE ONLY (Phase 7, Qwen3.5-VL-0.8B bf16).
 * Not buildable against the current tree: it includes stratum_prefetch.h,
 * which no longer exists. Kept as source-level record of engine evolution.
 * All model paths are mandatory CLI arguments (no built-in defaults).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "stratum_st.h"
#include "stratum_algo.h"
#include "stratum_prefetch.h"
#include "stratum_neon.h"
#include "stratum_lin_attn_neon.h"
#include "stratum_sgemv_worker.h"

#include <pthread/qos.h>
#include <sys/mman.h>

static const uint8_t* g_mmap_base = NULL;
static size_t         g_mmap_size = 0;
#define NUM_LAYERS  24
#define H           1024
#define INTERM      3584
#define HEAD        256
#define NQ          8
#define NK_FULL     2
#define ROT_DIM     64
#define THETA       1e7f
#define EPS         1e-6f

#define NV          16
#define LIN_NK      16
#define HK          128
#define HV          128
#define KEY_DIM     (HK * LIN_NK)
#define VAL_DIM     (HV * NV)
#define CONV_DIM    (KEY_DIM*2 + VAL_DIM)
#define KERNEL      4

#define VOCAB       248320

#define MAX_KV      512
#define SLOT_BYTES  (2 * 1024 * 1024)

#define KV_BYTES    (MAX_KV * NK_FULL * HEAD * 2)
#define KV_HEADER   16
#define LIN_CONV_BYTES  (CONV_DIM * KERNEL * 4)
#define LIN_REC_BYTES   (NV * HK * HV * 4)

static const int LAYER_TYPES[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

static int g_state_fd = -1;

static uint8_t* g_state_mmap = NULL;
static size_t   g_state_mmap_size = 0;

static int open_state_file(const char* path) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { perror("open state"); return -1; }
    if (ftruncate(fd, (off_t)NUM_LAYERS * SLOT_BYTES) != 0) {
        perror("ftruncate"); close(fd); return -1;
    }
#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1) < 0) perror("fcntl F_NOCACHE state");
#endif
    return fd;
}

static int state_pwrite(off_t off, const void* buf, size_t bytes) {
    if (g_state_mmap != NULL) {
        memcpy(g_state_mmap + off, buf, bytes);
        return 0;
    }
    const char* p = (const char*)buf;
    while (bytes > 0) {
        ssize_t w = pwrite(g_state_fd, p, bytes, off);
        if (w < 0) {
            if (errno == EINTR) continue;
            perror("pwrite"); return -1;
        }
        p += w; off += w; bytes -= (size_t)w;
    }
    return 0;
}

static int state_pread(off_t off, void* buf, size_t bytes) {
    if (g_state_mmap != NULL) {
        memcpy(buf, g_state_mmap + off, bytes);
        return 0;
    }
    return st_pread_full(g_state_fd, buf, bytes, off);
}

static int g_fd = -1;
static TensorIndex g_ix = {0};

static int          g_mtp_fd = -1;
static TensorIndex  g_mtp_ix = {0};
static const uint8_t* g_mtp_mmap_base = NULL;
static size_t       g_mtp_mmap_size = 0;
static bool         g_mtp_enabled = false;

static int          g_bf16_fd = -1;
static TensorIndex  g_bf16_ix = {0};
static int64_t      g_bf16_embed_off = 0;

static uint16_t* g_blk_bf16 = NULL;
static uint8_t*  g_blk_u8   = NULL;
static float*    g_blk_fp32 = NULL;
#define BLOCK_ROWS_MAX 256

#define WORKER_BLOCK_ROWS_MAX 512

static uint8_t*  g_blk_alt  = NULL;

static Prefetch  g_pf;

#define MAX_SGEMV_WORKERS 7
static SgemvWorker g_sw[MAX_SGEMV_WORKERS];
static bool        g_sw_inited = false;
static int         g_n_workers = 2;

#define SW_SPLIT_THRESHOLD 96

#define EMBED_SCALE_FLOATS  VOCAB
static float g_embed_scale[EMBED_SCALE_FLOATS];

static float g_layer_scale[8 * 1024];

static float* g_scale_buf = NULL;
#define SCALE_BUF_FLOATS VOCAB

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

static inline float rms_scale(const float* x, int N, float eps) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i + 0);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);
        a0 = vfmaq_f32(a0, x0, x0);
        a1 = vfmaq_f32(a1, x1, x1);
        a2 = vfmaq_f32(a2, x2, x2);
        a3 = vfmaq_f32(a3, x3, x3);
    }
    for (; i + 4 <= N; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        a0 = vfmaq_f32(a0, xv, xv);
    }
    float ss = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < N; i++) ss += x[i] * x[i];
    return 1.0f / sqrtf(ss / (float)N + eps);
#else
    float ss = 0.0f;
    for (int i = 0; i < N; i++) ss += x[i] * x[i];
    return 1.0f / sqrtf(ss / (float)N + eps);
#endif
}

static inline void rms_apply(const float* x, const float* w, float scale,
                             int N, float* y) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t sv = vdupq_n_f32(scale);
    float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i + 0);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);
        float32x4_t w0 = vaddq_f32(vld1q_f32(w + i + 0), one);
        float32x4_t w1 = vaddq_f32(vld1q_f32(w + i + 4), one);
        float32x4_t w2 = vaddq_f32(vld1q_f32(w + i + 8), one);
        float32x4_t w3 = vaddq_f32(vld1q_f32(w + i + 12), one);
        vst1q_f32(y + i + 0,  vmulq_f32(vmulq_f32(x0, sv), w0));
        vst1q_f32(y + i + 4,  vmulq_f32(vmulq_f32(x1, sv), w1));
        vst1q_f32(y + i + 8,  vmulq_f32(vmulq_f32(x2, sv), w2));
        vst1q_f32(y + i + 12, vmulq_f32(vmulq_f32(x3, sv), w3));
    }
    for (; i + 4 <= N; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vaddq_f32(vld1q_f32(w + i), one);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(xv, sv), wv));
    }
    for (; i < N; i++) y[i] = x[i] * scale * (1.0f + w[i]);
#else
    for (int i = 0; i < N; i++) y[i] = x[i] * scale * (1.0f + w[i]);
#endif
}

static inline void rms_scale_inplace(float* x, float scale, int N) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t sv = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        vst1q_f32(x + i + 0,  vmulq_f32(vld1q_f32(x + i + 0),  sv));
        vst1q_f32(x + i + 4,  vmulq_f32(vld1q_f32(x + i + 4),  sv));
        vst1q_f32(x + i + 8,  vmulq_f32(vld1q_f32(x + i + 8),  sv));
        vst1q_f32(x + i + 12, vmulq_f32(vld1q_f32(x + i + 12), sv));
    }
    for (; i + 4 <= N; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), sv));
    }
    for (; i < N; i++) x[i] *= scale;
#else
    for (int i = 0; i < N; i++) x[i] *= scale;
#endif
}

typedef struct {
    float in_ln[H], post_ln[H];
    float q_norm[HEAD], k_norm[HEAD];
    float dt_bias[NV], A_log[NV], lin_norm[HV];
} LayerSmallWeights;

typedef struct {
    TensorMeta* w;
    TensorMeta* scale;
    const float* scale_cached;
} WHandle;

typedef struct {

    WHandle q_proj, k_proj, v_proj, o_proj;
    WHandle iqkv, iz, out_proj;
    WHandle gate_proj, up_proj, down_proj;

    TensorMeta *ib, *ia;
    TensorMeta *conv1d;
} LayerHandles;

static LayerSmallWeights g_lsw[NUM_LAYERS];
static LayerHandles      g_lh[NUM_LAYERS];

static float g_conv_state[CONV_DIM * KERNEL];
static float g_rec_state[NV * HK * HV];
static float g_kv_cache[MAX_KV * NK_FULL * HEAD * 2];

static float* g_conv_w_cache[NUM_LAYERS] = {0};
static float  g_conv_w_arena[18 * CONV_DIM * KERNEL];

static float* g_ib_cache[NUM_LAYERS] = {0};
static float* g_ia_cache[NUM_LAYERS] = {0};
static float  g_ib_arena[18 * NV * H];
static float  g_ia_arena[18 * NV * H];

static bool   g_cache_bf16 = true;

typedef struct {

    WHandle q_proj, k_proj, v_proj, o_proj;
    WHandle gate_proj, up_proj, down_proj;
    WHandle fc;

    float input_ln[H];
    float post_ln[H];
    float q_norm[HEAD], k_norm[HEAD];
    float final_norm[H];
    float pre_fc_e_norm[H];
    float pre_fc_h_norm[H];
} MtpHandles;

static MtpHandles g_mtp = {0};

static float g_mtp_kv_cache[MAX_KV * NK_FULL * HEAD * 2];
static int   g_mtp_kv_len = 0;

static float g_x[H], g_residual[H], g_xn[H];
static float g_mlp_gate[INTERM], g_mlp_up[INTERM], g_mlp_acc[INTERM];

static float g_q_full[NQ * HEAD * 2];
static float g_q[NQ][HEAD], g_kf[NK_FULL][HEAD], g_vf[NK_FULL][HEAD];
static float g_gate[NQ * HEAD], g_attn_concat[NQ * HEAD];

static float g_mixed_qkv[CONV_DIM], g_z_full[VAL_DIM], g_qkv_post[CONV_DIM];
static float g_b_v[NV], g_a_v[NV], g_g_v[NV], g_beta_v[NV];
static float g_qh[LIN_NK][HK], g_kh[LIN_NK][HK], g_vh[NV][HV];
static float g_core_attn[NV][HV], g_core_post[NV * HV];
static float g_attn_out[H], g_mlp_out[H];

static uint16_t g_conv_w_bf[CONV_DIM * KERNEL];
static float    g_conv_w_fp[CONV_DIM * KERNEL];

static void mkn(char* dst, size_t cap, int li, const char* leaf) {
    snprintf(dst, cap, "model.language_model.layers.%d.%s", li, leaf);
}

static TensorMeta* find_or_null(const char* name) {
    return st_index_lookup(&g_ix, name);
}

static TensorMeta* must_find(const char* name) {
    TensorMeta* m = find_or_null(name);
    if (!m) { fprintf(stderr, "missing tensor: %s\n", name); exit(2); }
    return m;
}

static float g_scale_arena[2 * 1024 * 1024 / 4];

static size_t g_scale_arena_used = 0;

static int cache_scale_into_arena(const TensorMeta* sm, const float** out_ptr) {
    if (!sm) { *out_ptr = NULL; return 0; }
    int64_t rows = sm->nbytes / 2;
    size_t cap = sizeof(g_scale_arena) / sizeof(float);
    if (g_scale_arena_used + (size_t)rows > cap) {
        fprintf(stderr, "scale arena overflow: %zu + %lld > %zu\n",
                g_scale_arena_used, (long long)rows, cap);
        return -1;
    }

    if (st_pread_full(g_fd, g_blk_bf16, (size_t)sm->nbytes, sm->abs_offset) < 0) return -1;
    float* dst = g_scale_arena + g_scale_arena_used;
    const uint16_t* raw = g_blk_bf16;
    for (int64_t i = 0; i < rows; i++) dst[i] = st_f16_to_f32(raw[i]);
    *out_ptr = dst;
    g_scale_arena_used += (size_t)rows;
    return 0;
}

static WHandle resolve_w(const char* name) {
    char nm[256];
    WHandle h = {0};
    h.w = must_find(name);
    h.scale_cached = NULL;
    snprintf(nm, sizeof nm, "%s.scale", name);
    h.scale = find_or_null(nm);
    if (h.scale != NULL) {
        if (cache_scale_into_arena(h.scale, &h.scale_cached) != 0) {
            fprintf(stderr, "fatal: failed to cache scale for %s\n", name);
            exit(2);
        }
    }
    return h;
}

static int load_bf16_meta(const TensorMeta* m, float* out, uint16_t* sc) {
    if (st_pread_full(g_fd, sc, (size_t)m->nbytes, m->abs_offset) < 0) return -1;
    st_bf16_to_fp32(sc, out, (size_t)(m->nbytes / 2));
    return 0;
}

static int cache_mtp_scale_into_arena(const TensorMeta* sm, const float** out_ptr) {
    if (!sm) { *out_ptr = NULL; return 0; }
    int64_t rows = sm->nbytes / 2;
    size_t cap = sizeof(g_scale_arena) / sizeof(float);
    if (g_scale_arena_used + (size_t)rows > cap) {
        fprintf(stderr, "scale arena overflow (mtp): %zu + %lld > %zu\n",
                g_scale_arena_used, (long long)rows, cap);
        return -1;
    }
    if (st_pread_full(g_mtp_fd, g_blk_bf16, (size_t)sm->nbytes, sm->abs_offset) < 0) return -1;
    float* dst = g_scale_arena + g_scale_arena_used;
    const uint16_t* raw = g_blk_bf16;
    for (int64_t i = 0; i < rows; i++) dst[i] = st_f16_to_f32(raw[i]);
    *out_ptr = dst;
    g_scale_arena_used += (size_t)rows;
    return 0;
}

static WHandle resolve_mtp_w(const char* name) {
    char nm[256];
    WHandle h = {0};
    h.w = st_index_lookup(&g_mtp_ix, name);
    if (!h.w) {
        fprintf(stderr, "missing MTP tensor: %s\n", name);
        exit(2);
    }
    h.scale_cached = NULL;
    snprintf(nm, sizeof nm, "%s.scale", name);
    h.scale = st_index_lookup(&g_mtp_ix, nm);
    if (h.scale != NULL) {
        if (cache_mtp_scale_into_arena(h.scale, &h.scale_cached) != 0) {
            fprintf(stderr, "fatal: failed to cache MTP scale for %s\n", name);
            exit(2);
        }
    }
    return h;
}

static int load_mtp_bf16_meta(const char* name, float* out, int N) {
    TensorMeta* m = st_index_lookup(&g_mtp_ix, name);
    if (!m) {
        fprintf(stderr, "missing MTP bf16 tensor: %s\n", name);
        return -1;
    }
    if ((size_t)m->nbytes != (size_t)N * 2) {
        fprintf(stderr, "MTP %s size mismatch: %lld vs %d*2\n",
                name, (long long)m->nbytes, N);
        return -1;
    }
    uint16_t scratch[H + 16];
    if ((size_t)N > sizeof(scratch)/2) {
        fprintf(stderr, "MTP bf16 vec too large: %d\n", N);
        return -1;
    }
    if (st_pread_full(g_mtp_fd, scratch, (size_t)m->nbytes, m->abs_offset) < 0) return -1;
    st_bf16_to_fp32(scratch, out, (size_t)N);
    return 0;
}

static int load_mtp_weights(void) {
    g_mtp.q_proj    = resolve_mtp_w("mtp.layers.0.self_attn.q_proj.weight");
    g_mtp.k_proj    = resolve_mtp_w("mtp.layers.0.self_attn.k_proj.weight");
    g_mtp.v_proj    = resolve_mtp_w("mtp.layers.0.self_attn.v_proj.weight");
    g_mtp.o_proj    = resolve_mtp_w("mtp.layers.0.self_attn.o_proj.weight");
    g_mtp.gate_proj = resolve_mtp_w("mtp.layers.0.mlp.gate_proj.weight");
    g_mtp.up_proj   = resolve_mtp_w("mtp.layers.0.mlp.up_proj.weight");
    g_mtp.down_proj = resolve_mtp_w("mtp.layers.0.mlp.down_proj.weight");
    g_mtp.fc        = resolve_mtp_w("mtp.fc.weight");
    if (load_mtp_bf16_meta("mtp.layers.0.input_layernorm.weight",
                            g_mtp.input_ln, H) != 0) return -1;
    if (load_mtp_bf16_meta("mtp.layers.0.post_attention_layernorm.weight",
                            g_mtp.post_ln, H) != 0) return -1;
    if (load_mtp_bf16_meta("mtp.layers.0.self_attn.q_norm.weight",
                            g_mtp.q_norm, HEAD) != 0) return -1;
    if (load_mtp_bf16_meta("mtp.layers.0.self_attn.k_norm.weight",
                            g_mtp.k_norm, HEAD) != 0) return -1;
    if (load_mtp_bf16_meta("mtp.norm.weight",
                            g_mtp.final_norm, H) != 0) return -1;
    if (load_mtp_bf16_meta("mtp.pre_fc_norm_embedding.weight",
                            g_mtp.pre_fc_e_norm, H) != 0) return -1;
    if (load_mtp_bf16_meta("mtp.pre_fc_norm_hidden.weight",
                            g_mtp.pre_fc_h_norm, H) != 0) return -1;

    if (getenv("STRATUM_MTP_DEBUG")) {

        float sum_e = 0, sum_h = 0, sum_in = 0, sum_post = 0, sum_fn = 0;
        for (int i = 0; i < H; i++) {
            sum_e += g_mtp.pre_fc_e_norm[i];
            sum_h += g_mtp.pre_fc_h_norm[i];
            sum_in += g_mtp.input_ln[i];
            sum_post += g_mtp.post_ln[i];
            sum_fn += g_mtp.final_norm[i];
        }
        fprintf(stderr, "[mtp] pre_fc_e_norm sum=%.3f, pre_fc_h_norm sum=%.3f\n", sum_e, sum_h);
        fprintf(stderr, "[mtp] input_ln sum=%.3f, post_ln sum=%.3f, final_norm sum=%.3f\n",
                sum_in, sum_post, sum_fn);
        fprintf(stderr, "[mtp] pre_fc_e_norm[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
                g_mtp.pre_fc_e_norm[0], g_mtp.pre_fc_e_norm[1],
                g_mtp.pre_fc_e_norm[2], g_mtp.pre_fc_e_norm[3],
                g_mtp.pre_fc_e_norm[4]);
        fprintf(stderr, "[mtp] q_proj.scale[0..4] = %.6f %.6f %.6f %.6f %.6f\n",
                g_mtp.q_proj.scale_cached[0], g_mtp.q_proj.scale_cached[1],
                g_mtp.q_proj.scale_cached[2], g_mtp.q_proj.scale_cached[3],
                g_mtp.q_proj.scale_cached[4]);
    }
    return 0;
}

static int load_f32_meta(const TensorMeta* m, float* out) {
    return st_pread_full(g_fd, out, (size_t)m->nbytes, m->abs_offset);
}

static int load_scale(const TensorMeta* sm, float* out) {
    if (!sm) return -1;
    int64_t rows = sm->nbytes / 2;

    if (st_pread_full(g_fd, g_blk_bf16, (size_t)sm->nbytes, sm->abs_offset) < 0) return -1;
    const uint16_t* raw = g_blk_bf16;
    for (int64_t i = 0; i < rows; i++) out[i] = st_f16_to_f32(raw[i]);
    return (int)rows;
}

static int sgemv_auto(const WHandle* h, const float* x, float* y,
                      int64_t N, int64_t K) {
    bool int4 = (h->scale != NULL);
    const float* scale_f32 = NULL;
    if (int4) {
        if (h->scale_cached) {
            scale_f32 = h->scale_cached;
        } else {

            int64_t rows = h->scale->nbytes / 2;
            if (rows > (int64_t)(sizeof(g_layer_scale) / sizeof(float))) {
                fprintf(stderr, "layer scale too large: %lld\n", (long long)rows);
                return -1;
            }
            uint16_t scratch[8 * 1024];
            if (st_pread_full(g_fd, scratch, (size_t)(rows * 2), h->scale->abs_offset) < 0) return -1;
            for (int64_t i = 0; i < rows; i++) g_layer_scale[i] = st_f16_to_f32(scratch[i]);
            scale_f32 = g_layer_scale;
        }
    }

    int64_t target_rows = ST_BLOCK_FP32_BYTES / (K * 4);
    if (target_rows < 1) target_rows = 1;
    if (target_rows > BLOCK_ROWS_MAX) target_rows = BLOCK_ROWS_MAX;

    memset(y, 0, (size_t)N * sizeof(float));
    int64_t row_bytes = int4 ? (K / 2) : (K * 2);
    int64_t off_base  = h->w->abs_offset;

    if (int4 && g_sw_inited && N >= SW_SPLIT_THRESHOLD) {
        int total = g_n_workers + 1;
        int64_t slab = N / total;
        int64_t rem  = N - slab * total;

        int64_t cursor = 0;

        int64_t main_start = cursor;
        int64_t main_count = slab + (rem > 0 ? 1 : 0);
        cursor += main_count;
        if (rem > 0) rem--;

        for (int wi = 0; wi < g_n_workers; wi++) {
            int64_t s = slab + (rem > 0 ? 1 : 0);
            if (rem > 0) rem--;
            SgemvJob job = {
                .fd = g_fd,
                .packed_off = off_base,
                .row_start = cursor,
                .row_count = s,
                .in_features = K,
                .scale_f32 = scale_f32,
                .x = x,
                .y = y,
                .block_rows_cap = WORKER_BLOCK_ROWS_MAX,
                .mmap_base = g_mmap_base,
            };
            sgemv_worker_dispatch(&g_sw[wi], &job);
            cursor += s;
        }

        int64_t main_N = main_count;
        int64_t nblocks_main = (main_N + target_rows - 1) / target_rows;
        if (nblocks_main > 0) {
            if (g_mmap_base != NULL && int4) {

                for (int64_t b = 0; b < nblocks_main; b++) {
                    int64_t row0 = b * target_rows;
                    int64_t rows = (row0 + target_rows <= main_N) ? target_rows : (main_N - row0);
                    const uint8_t* blk = g_mmap_base + off_base + (main_start + row0) * row_bytes;
                    st_int4_sgemv_fused_neon(blk, scale_f32 + main_start + row0,
                                             (int)rows, (int)K, x, y + main_start + row0);
                }
            } else {
            uint8_t* bufs[2] = { g_blk_u8, g_blk_alt };
            int cur = 0;
            int64_t rows0 = (target_rows <= main_N) ? target_rows : main_N;
            prefetch_request(&g_pf, g_fd, bufs[cur],
                             (size_t)(rows0 * row_bytes),
                             off_base + main_start * row_bytes);
            for (int64_t b = 0; b < nblocks_main; b++) {
                int64_t row0 = b * target_rows;
                int64_t rows = (row0 + target_rows <= main_N) ? target_rows : (main_N - row0);
                if (prefetch_wait(&g_pf) != 0) {
                    for (int wi = 0; wi < g_n_workers; wi++) sgemv_worker_wait(&g_sw[wi]);
                    return -1;
                }
                uint8_t* this_raw = bufs[cur];
                if (b + 1 < nblocks_main) {
                    int next = 1 - cur;
                    int64_t nrow0 = (b + 1) * target_rows;
                    int64_t nrows = (nrow0 + target_rows <= main_N) ? target_rows : (main_N - nrow0);
                    prefetch_request(&g_pf, g_fd, bufs[next],
                                     (size_t)(nrows * row_bytes),
                                     off_base + (main_start + nrow0) * row_bytes);
                }
                st_int4_sgemv_fused_neon(this_raw, scale_f32 + main_start + row0,
                                         (int)rows, (int)K, x, y + main_start + row0);
                cur = 1 - cur;
            }
            }
        }
        for (int wi = 0; wi < g_n_workers; wi++) {
            if (sgemv_worker_wait(&g_sw[wi]) != 0) return -1;
        }
        return 0;
    }

    int64_t nblocks   = (N + target_rows - 1) / target_rows;
    uint8_t* primary = int4 ? g_blk_u8 : (uint8_t*)g_blk_bf16;
    uint8_t* bufs[2] = { primary, g_blk_alt };
    int cur = 0;

    int64_t rows0 = (target_rows <= N) ? target_rows : N;
    prefetch_request(&g_pf, g_fd, bufs[cur],
                     (size_t)(rows0 * row_bytes), off_base);

    for (int64_t b = 0; b < nblocks; b++) {
        int64_t row0 = b * target_rows;
        int64_t rows = (row0 + target_rows <= N) ? target_rows : (N - row0);

        if (prefetch_wait(&g_pf) != 0) return -1;
        uint8_t* this_raw = bufs[cur];

        if (b + 1 < nblocks) {
            int next = 1 - cur;
            int64_t nrow0 = (b + 1) * target_rows;
            int64_t nrows = (nrow0 + target_rows <= N) ? target_rows : (N - nrow0);
            prefetch_request(&g_pf, g_fd, bufs[next],
                             (size_t)(nrows * row_bytes),
                             off_base + nrow0 * row_bytes);
        }

        if (int4) {
            st_int4_sgemv_fused_neon(this_raw, scale_f32 + row0,
                                     (int)rows, (int)K, x, y + row0);
        } else {
            st_bf16_to_fp32((uint16_t*)this_raw, g_blk_fp32, (size_t)(rows * K));
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        (int)rows, (int)K,
                        1.0f, g_blk_fp32, (int)K,
                        x, 1,
                        1.0f, y + row0, 1);
        }
        cur = 1 - cur;
    }
    return 0;
}

static int load_layer_weights(void) {
    char nm[256]; uint16_t sc[H];
    for (int li = 0; li < NUM_LAYERS; li++) {
        mkn(nm, sizeof nm, li, "input_layernorm.weight");
        if (load_bf16_meta(must_find(nm), g_lsw[li].in_ln, sc) != 0) return -1;
        mkn(nm, sizeof nm, li, "post_attention_layernorm.weight");
        if (load_bf16_meta(must_find(nm), g_lsw[li].post_ln, sc) != 0) return -1;

        mkn(nm, sizeof nm, li, "mlp.gate_proj.weight"); g_lh[li].gate_proj = resolve_w(nm);
        mkn(nm, sizeof nm, li, "mlp.up_proj.weight");   g_lh[li].up_proj   = resolve_w(nm);
        mkn(nm, sizeof nm, li, "mlp.down_proj.weight"); g_lh[li].down_proj = resolve_w(nm);

        if (LAYER_TYPES[li] == 1) {
            mkn(nm, sizeof nm, li, "self_attn.q_proj.weight"); g_lh[li].q_proj = resolve_w(nm);
            mkn(nm, sizeof nm, li, "self_attn.k_proj.weight"); g_lh[li].k_proj = resolve_w(nm);
            mkn(nm, sizeof nm, li, "self_attn.v_proj.weight"); g_lh[li].v_proj = resolve_w(nm);
            mkn(nm, sizeof nm, li, "self_attn.o_proj.weight"); g_lh[li].o_proj = resolve_w(nm);
            mkn(nm, sizeof nm, li, "self_attn.q_norm.weight");
            if (load_bf16_meta(must_find(nm), g_lsw[li].q_norm, sc) != 0) return -1;
            mkn(nm, sizeof nm, li, "self_attn.k_norm.weight");
            if (load_bf16_meta(must_find(nm), g_lsw[li].k_norm, sc) != 0) return -1;
        } else {
            mkn(nm, sizeof nm, li, "linear_attn.in_proj_qkv.weight"); g_lh[li].iqkv = resolve_w(nm);
            mkn(nm, sizeof nm, li, "linear_attn.in_proj_z.weight");   g_lh[li].iz   = resolve_w(nm);
            mkn(nm, sizeof nm, li, "linear_attn.in_proj_b.weight");
            g_lh[li].ib = must_find(nm);
            mkn(nm, sizeof nm, li, "linear_attn.in_proj_a.weight");
            g_lh[li].ia = must_find(nm);

            if (g_cache_bf16) {
                int slot = 0;
                for (int j = 0; j < li; j++) if (LAYER_TYPES[j] == 0) slot++;
                float* ib_slot = g_ib_arena + (size_t)slot * NV * H;
                float* ia_slot = g_ia_arena + (size_t)slot * NV * H;
                int64_t ia_bytes = (int64_t)NV * H * 2;
                if (st_pread_full(g_fd, g_blk_bf16, (size_t)ia_bytes,
                                  g_lh[li].ib->abs_offset) < 0) return -1;
                st_bf16_to_fp32(g_blk_bf16, ib_slot, NV * H);
                if (st_pread_full(g_fd, g_blk_bf16, (size_t)ia_bytes,
                                  g_lh[li].ia->abs_offset) < 0) return -1;
                st_bf16_to_fp32(g_blk_bf16, ia_slot, NV * H);
                g_ib_cache[li] = ib_slot;
                g_ia_cache[li] = ia_slot;
            }
            mkn(nm, sizeof nm, li, "linear_attn.conv1d.weight");
            g_lh[li].conv1d = must_find(nm);

            {
                int slot = 0;
                for (int j = 0; j < li; j++) if (LAYER_TYPES[j] == 0) slot++;
                float* slot_ptr = g_conv_w_arena + (size_t)slot * CONV_DIM * KERNEL;
                int64_t conv_bytes = (int64_t)CONV_DIM * KERNEL * 2;
                if (st_pread_full(g_fd, g_blk_bf16, (size_t)conv_bytes,
                                  g_lh[li].conv1d->abs_offset) < 0) return -1;
                st_bf16_to_fp32(g_blk_bf16, slot_ptr, CONV_DIM * KERNEL);
                g_conv_w_cache[li] = slot_ptr;
            }
            mkn(nm, sizeof nm, li, "linear_attn.out_proj.weight");
            g_lh[li].out_proj = resolve_w(nm);
            mkn(nm, sizeof nm, li, "linear_attn.dt_bias");
            { uint16_t s[NV]; if (load_bf16_meta(must_find(nm), g_lsw[li].dt_bias, s) != 0) return -1; }
            mkn(nm, sizeof nm, li, "linear_attn.A_log");
            if (load_f32_meta(must_find(nm), g_lsw[li].A_log) != 0) return -1;
            mkn(nm, sizeof nm, li, "linear_attn.norm.weight");
            if (load_f32_meta(must_find(nm), g_lsw[li].lin_norm) != 0) return -1;
        }
    }
    return 0;
}

static int sgemv_bf16(const TensorMeta* m, const float* x, float* y,
                      int64_t out_features, int64_t in_features) {
    StreamWeight w = {0};
    w.abs_offset = m->abs_offset;
    w.out_features = out_features;
    w.in_features  = in_features;
    strncpy(w.dtype, m->dtype, sizeof w.dtype);
    return st_linear_stream(g_fd, &w, x, NULL, y, false,
                            g_blk_bf16, g_blk_fp32, BLOCK_ROWS_MAX);
}

static int forward_full_attn(int li, int position, const float* x_in, float* y_out) {
    LayerHandles* lh = &g_lh[li];
    LayerSmallWeights* lsw = &g_lsw[li];

    if (sgemv_auto(&lh->q_proj, x_in, g_q_full,        NQ * HEAD * 2, H) != 0) return -1;
    if (sgemv_auto(&lh->k_proj, x_in, (float*)g_kf,    NK_FULL * HEAD, H) != 0) return -1;
    if (sgemv_auto(&lh->v_proj, x_in, (float*)g_vf,    NK_FULL * HEAD, H) != 0) return -1;

    for (int h = 0; h < NQ; h++) {
        const float* row = g_q_full + h * (HEAD * 2);
        memcpy(g_q[h], row, HEAD * sizeof(float));
        memcpy(g_gate + h * HEAD, row + HEAD, HEAD * sizeof(float));
    }
    for (int h = 0; h < NQ; h++) {
        float scale = rms_scale(g_q[h], HEAD, EPS);
        rms_apply(g_q[h], lsw->q_norm, scale, HEAD, g_q[h]);
    }
    for (int h = 0; h < NK_FULL; h++) {
        float scale = rms_scale(g_kf[h], HEAD, EPS);
        rms_apply(g_kf[h], lsw->k_norm, scale, HEAD, g_kf[h]);
    }
    for (int h = 0; h < NQ; h++) rope_apply_inplace_fp32(g_q[h], position, HEAD, ROT_DIM, THETA);
    for (int h = 0; h < NK_FULL; h++) rope_apply_inplace_fp32(g_kf[h], position, HEAD, ROT_DIM, THETA);

    int32_t header[KV_HEADER / 4];
    if (state_pread((off_t)li * SLOT_BYTES, header, KV_HEADER) != 0) return -1;
    int kv_len = header[0];

    float* K_arr = g_kv_cache;
    float* V_arr = g_kv_cache + (size_t)MAX_KV * NK_FULL * HEAD;
    if (kv_len > 0) {
        size_t row_count = (size_t)kv_len * NK_FULL * HEAD;
        const uint16_t* K_bf = (const uint16_t*)
            (g_state_mmap + (off_t)li * SLOT_BYTES + KV_HEADER);
        const uint16_t* V_bf = (const uint16_t*)
            (g_state_mmap + (off_t)li * SLOT_BYTES + KV_HEADER + KV_BYTES);
        if (g_state_mmap != NULL) {
            st_bf16_to_fp32(K_bf, K_arr, row_count);
            st_bf16_to_fp32(V_bf, V_arr, row_count);
        } else {

            static uint16_t staging[MAX_KV * NK_FULL * HEAD];
            size_t bytes = row_count * 2;
            if (state_pread((off_t)li * SLOT_BYTES + KV_HEADER,
                            staging, bytes) != 0) return -1;
            st_bf16_to_fp32(staging, K_arr, row_count);
            if (state_pread((off_t)li * SLOT_BYTES + KV_HEADER + KV_BYTES,
                            staging, bytes) != 0) return -1;
            st_bf16_to_fp32(staging, V_arr, row_count);
        }
    }

    for (int h = 0; h < NK_FULL; h++) {
        memcpy(K_arr + (size_t)kv_len * NK_FULL * HEAD + h * HEAD,
               g_kf[h], HEAD * sizeof(float));
        memcpy(V_arr + (size_t)kv_len * NK_FULL * HEAD + h * HEAD,
               g_vf[h], HEAD * sizeof(float));
    }
    int new_len = kv_len + 1;

    float scale = 1.0f / sqrtf((float)HEAD);

    const int GROUP = NQ / NK_FULL;
    const int SGEMM_THRESHOLD = 96;
    if (new_len >= SGEMM_THRESHOLD) {
        float Q_grp[GROUP * HEAD];
        float logits_mat[GROUP * MAX_KV];
        float out_grp[GROUP * HEAD];

        for (int kv = 0; kv < NK_FULL; kv++) {
            for (int gi = 0; gi < GROUP; gi++) {
                int h = kv * GROUP + gi;
                memcpy(Q_grp + gi * HEAD, g_q[h], HEAD * sizeof(float));
            }
            const float* K_kv = K_arr + kv * HEAD;
            const float* V_kv = V_arr + kv * HEAD;
            const int lda = NK_FULL * HEAD;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        GROUP, new_len, HEAD,
                        scale,
                        Q_grp, HEAD,
                        K_kv,  lda,
                        0.0f,
                        logits_mat, MAX_KV);

            for (int gi = 0; gi < GROUP; gi++) {
                softmax_fp32_inplace(logits_mat + gi * MAX_KV, new_len);
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        GROUP, HEAD, new_len,
                        1.0f,
                        logits_mat, MAX_KV,
                        V_kv, lda,
                        0.0f,
                        out_grp, HEAD);

            for (int gi = 0; gi < GROUP; gi++) {
                int h = kv * GROUP + gi;
                memcpy(g_attn_concat + h * HEAD, out_grp + gi * HEAD,
                       HEAD * sizeof(float));
            }
        }
    } else {

        float logits[MAX_KV];
        for (int h = 0; h < NQ; h++) {
            int kv = h / (NQ / NK_FULL);
            const float* K_kv = K_arr + kv * HEAD;
            const float* V_kv = V_arr + kv * HEAD;
            const int lda = NK_FULL * HEAD;
            const float* qh = g_q[h];
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        new_len, HEAD,
                        scale, K_kv, lda,
                        qh, 1,
                        0.0f, logits, 1);
            softmax_fp32_inplace(logits, new_len);
            float head_out[HEAD];
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        new_len, HEAD,
                        1.0f, V_kv, lda,
                        logits, 1,
                        0.0f, head_out, 1);
            memcpy(g_attn_concat + h * HEAD, head_out, HEAD * sizeof(float));
        }
    }
    for (int i = 0; i < NQ * HEAD; i++) {
        float gv = g_gate[i];
        g_attn_concat[i] *= 1.0f / (1.0f + expf(-gv));
    }
    if (sgemv_auto(&lh->o_proj, g_attn_concat, y_out, H, NQ * HEAD) != 0) return -1;

    header[0] = new_len;
    if (state_pwrite((off_t)li * SLOT_BYTES, header, KV_HEADER) != 0) return -1;

    size_t row_floats = (size_t)NK_FULL * HEAD;
    size_t row_bytes_bf16 = row_floats * 2;
    off_t k_off = (off_t)li * SLOT_BYTES + KV_HEADER + (off_t)kv_len * row_bytes_bf16;
    off_t v_off = (off_t)li * SLOT_BYTES + KV_HEADER + KV_BYTES + (off_t)kv_len * row_bytes_bf16;
    {
        uint16_t bf_buf[NK_FULL * HEAD];
        const float* k_src = K_arr + (size_t)kv_len * row_floats;
        for (size_t i = 0; i < row_floats; i++) bf_buf[i] = fp32_to_bf16(k_src[i]);
        if (state_pwrite(k_off, bf_buf, row_bytes_bf16) != 0) return -1;
        const float* v_src = V_arr + (size_t)kv_len * row_floats;
        for (size_t i = 0; i < row_floats; i++) bf_buf[i] = fp32_to_bf16(v_src[i]);
        if (state_pwrite(v_off, bf_buf, row_bytes_bf16) != 0) return -1;
    }
    return 0;
}

static int forward_linear_attn(int li, const float* x_in, float* y_out) {
    LayerHandles* lh = &g_lh[li];
    LayerSmallWeights* lsw = &g_lsw[li];

    if (sgemv_auto(&lh->iqkv, x_in, g_mixed_qkv, CONV_DIM, H) != 0) return -1;
    if (sgemv_auto(&lh->iz,   x_in, g_z_full,    VAL_DIM,  H) != 0) return -1;

    if (g_ib_cache[li] != NULL) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    NV, H, 1.0f, g_ib_cache[li], H, x_in, 1, 0.0f, g_b_v, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    NV, H, 1.0f, g_ia_cache[li], H, x_in, 1, 0.0f, g_a_v, 1);
    } else {
        if (sgemv_bf16(lh->ib, x_in, g_b_v, NV, H) != 0) return -1;
        if (sgemv_bf16(lh->ia, x_in, g_a_v, NV, H) != 0) return -1;
    }

    const float* conv_w = g_conv_w_cache[li];

    if (state_pread((off_t)li * SLOT_BYTES, g_conv_state, LIN_CONV_BYTES) != 0) return -1;

    float new_col[CONV_DIM];
    for (int c = 0; c < CONV_DIM; c++) {
        const float* old = g_conv_state + (size_t)c * KERNEL;
        const float* w   = conv_w  + (size_t)c * KERNEL;
        float new_v = g_mixed_qkv[c];
        float acc = old[1]*w[0] + old[2]*w[1] + old[3]*w[2] + new_v*w[3];
        g_qkv_post[c] = silu_f(acc);
        new_col[c] = new_v;
    }
    for (int c = 0; c < CONV_DIM; c++) {
        float* row = g_conv_state + (size_t)c * KERNEL;
        row[0] = row[1]; row[1] = row[2]; row[2] = row[3]; row[3] = new_col[c];
    }

    if (state_pread((off_t)li * SLOT_BYTES + LIN_CONV_BYTES,
                    g_rec_state, LIN_REC_BYTES) != 0) return -1;

    for (int h = 0; h < LIN_NK; h++) {
        memcpy(g_qh[h], g_qkv_post + h * HK, HK * sizeof(float));
        memcpy(g_kh[h], g_qkv_post + KEY_DIM + h * HK, HK * sizeof(float));
    }
    for (int h = 0; h < NV; h++)
        memcpy(g_vh[h], g_qkv_post + 2*KEY_DIM + h * HV, HV * sizeof(float));

    for (int h = 0; h < NV; h++) {
        g_beta_v[h] = 1.0f / (1.0f + expf(-g_b_v[h]));
        float ax = g_a_v[h] + lsw->dt_bias[h];
        float sp = (ax > 20.0f) ? ax : log1pf(expf(ax));
        g_g_v[h] = -expf(lsw->A_log[h]) * sp;
    }
    for (int h = 0; h < LIN_NK; h++) {

        float ss_q = rms_scale(g_qh[h], HK, 0.0f);
        (void)ss_q;

        float ssq = 0.0f, ssk = 0.0f;
#if defined(__ARM_NEON) || defined(__aarch64__)
        {
            float32x4_t a0 = vdupq_n_f32(0.0f);
            for (int c = 0; c + 4 <= HK; c += 4) {
                float32x4_t v = vld1q_f32(g_qh[h] + c);
                a0 = vfmaq_f32(a0, v, v);
            }
            ssq = vaddvq_f32(a0);
            float32x4_t b0 = vdupq_n_f32(0.0f);
            for (int c = 0; c + 4 <= HK; c += 4) {
                float32x4_t v = vld1q_f32(g_kh[h] + c);
                b0 = vfmaq_f32(b0, v, v);
            }
            ssk = vaddvq_f32(b0);
        }
#else
        for (int c = 0; c < HK; c++) ssq += g_qh[h][c] * g_qh[h][c];
        for (int c = 0; c < HK; c++) ssk += g_kh[h][c] * g_kh[h][c];
#endif
        float inv_q = 1.0f / sqrtf(ssq + 1e-6f);
        float inv_k = 1.0f / sqrtf(ssk + 1e-6f);
        rms_scale_inplace(g_qh[h], inv_q, HK);
        rms_scale_inplace(g_kh[h], inv_k, HK);
    }

    float scale = 1.0f / sqrtf((float)HK);

    for (int h = 0; h < NV; h++) {
        float* S = g_rec_state + (size_t)h * HK * HV;
        float gt = expf(g_g_v[h]);
        float bt = g_beta_v[h];
        const float* qt = g_qh[h];
        const float* kt = g_kh[h];
        const float* vt = g_vh[h];

        float b_pre[HV], a_pre[HV];
        st_lin_attn_dual_gemv_trans(S, kt, qt, HK, HV, b_pre, a_pre);

        float kq_dot = 0.0f;
        vDSP_dotpr(kt, 1, qt, 1, &kq_dot, HK);

        float delta[HV];
        for (int v = 0; v < HV; v++) {
            float bv = gt * b_pre[v];
            float dv = (vt[v] - bv) * bt;
            delta[v] = dv;
            g_core_attn[h][v] = (gt * a_pre[v] + kq_dot * dv) * scale;
        }

        st_lin_attn_fused_decay_outer(S, kt, delta, HK, HV, gt);
    }
    for (int h = 0; h < NV; h++) {
        const float* zhead = g_z_full + h * HV;
        const float* xhead = g_core_attn[h];
        float* yhead = g_core_post + h * HV;

        float inv = rms_scale(xhead, HV, EPS);
        for (int c = 0; c < HV; c++) {
            float n = xhead[c] * inv * lsw->lin_norm[c];
            yhead[c] = n * silu_f(zhead[c]);
        }
    }
    if (sgemv_auto(&lh->out_proj, g_core_post, y_out, H, VAL_DIM) != 0) return -1;

    if (state_pwrite((off_t)li * SLOT_BYTES, g_conv_state, LIN_CONV_BYTES) != 0) return -1;
    if (state_pwrite((off_t)li * SLOT_BYTES + LIN_CONV_BYTES,
                     g_rec_state, LIN_REC_BYTES) != 0) return -1;
    return 0;
}

static int forward_mlp(int li, const float* x_in, float* y_out) {
    LayerHandles* lh = &g_lh[li];
    if (sgemv_auto(&lh->gate_proj, x_in, g_mlp_gate, INTERM, H) != 0) return -1;
    if (sgemv_auto(&lh->up_proj,   x_in, g_mlp_up,   INTERM, H) != 0) return -1;
    swiglu_fp32(g_mlp_gate, g_mlp_up, g_mlp_acc, INTERM);
    if (sgemv_auto(&lh->down_proj, g_mlp_acc, y_out, H, INTERM) != 0) return -1;
    return 0;
}

static int int4_lookup_row(const TensorMeta* embed_w, const float* embed_scale_cached,
                           int token_id, float* out_row) {
    int half = H / 2;
    uint8_t row_packed[H / 2];
    int64_t off_w = embed_w->abs_offset + (int64_t)token_id * half;
    if (st_pread_full(g_fd, row_packed, (size_t)half, off_w) < 0) return -1;
    float s = embed_scale_cached[token_id];
    for (int i = 0; i < half; i++) {
        uint8_t b = row_packed[i];
        int lo = b & 0x0F;
        int hi = (b >> 4) & 0x0F;
        int lo_s = (lo >= 8) ? lo - 16 : lo;
        int hi_s = (hi >= 8) ? hi - 16 : hi;
        out_row[2*i]     = (float)lo_s * s;
        out_row[2*i + 1] = (float)hi_s * s;
    }
    return 0;
}

static uint64_t g_pcg_state = 0xC0DEC0DE;
static uint64_t g_pcg_inc   = 0x14057B7EF767814F;

static inline uint32_t pcg32(void) {
    uint64_t old = g_pcg_state;
    g_pcg_state = old * 6364136223846793005ULL + g_pcg_inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int)rot) & 31));
}

static inline float pcg32_uniform(void) {
    return (pcg32() >> 8) / (float)(1u << 24);
}

static void pcg32_seed(uint64_t seed) {
    g_pcg_state = 0;
    g_pcg_inc   = (seed << 1u) | 1u;
    pcg32();
    g_pcg_state += seed;
    pcg32();
}

typedef struct { float logit; int id; } TopEntry;

static inline void topk_swap(TopEntry* a, TopEntry* b) {
    TopEntry t = *a; *a = *b; *b = t;
}

static void topk_sift_down(TopEntry* heap, int n, int i) {
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, smallest = i;
        if (l < n && heap[l].logit < heap[smallest].logit) smallest = l;
        if (r < n && heap[r].logit < heap[smallest].logit) smallest = r;
        if (smallest == i) break;
        topk_swap(&heap[i], &heap[smallest]);
        i = smallest;
    }
}

static void topk_offer(TopEntry* heap, int K, int* size, float lv, int id) {
    if (*size < K) {
        heap[*size].logit = lv;
        heap[*size].id    = id;
        (*size)++;
        if (*size == K) {
            for (int i = K/2 - 1; i >= 0; i--) topk_sift_down(heap, K, i);
        }
    } else if (lv > heap[0].logit) {
        heap[0].logit = lv;
        heap[0].id    = id;
        topk_sift_down(heap, K, 0);
    }
}

static int topk_int4_lm_head(const TensorMeta* embed_w, const float* embed_scale_cached,
                              const float* x_in,
                              const int* recent_ids, int recent_count,
                              float rep_penalty,
                              TopEntry* out_top, int K) {
    int half = H / 2;
    int heap_size = 0;
    int64_t off_base = embed_w->abs_offset;

    if (g_sw_inited) {

        const int K_local = (K < 64) ? 64 : K;
        TopEntryW worker_heaps[MAX_SGEMV_WORKERS][256];
        TopEntryW main_heap[256];
        int main_heap_size = 0;

        int total = g_n_workers + 1;
        int64_t slab = VOCAB / total;
        int64_t rem  = VOCAB - slab * total;

        int64_t cursor = 0;
        int64_t main_count = slab + (rem > 0 ? 1 : 0);
        int64_t main_start = cursor;
        cursor += main_count;
        if (rem > 0) rem--;

        for (int wi = 0; wi < g_n_workers; wi++) {
            int64_t s = slab + (rem > 0 ? 1 : 0);
            if (rem > 0) rem--;
            SgemvJob job = {
                .fd = g_fd,
                .packed_off = off_base,
                .row_start = cursor,
                .row_count = s,
                .in_features = H,
                .scale_f32 = embed_scale_cached,
                .x = x_in,
                .y = NULL,
                .block_rows_cap = WORKER_BLOCK_ROWS_MAX,
                .mmap_base = g_mmap_base,
                .topk_out = worker_heaps[wi],
                .topk_K = K_local,
                .topk_size_out = 0,
            };
            sgemv_worker_dispatch(&g_sw[wi], &job);
            cursor += s;
        }

        int64_t target_rows = ST_BLOCK_FP32_BYTES / (H * 4);
        if (target_rows < 1) target_rows = 1;
        if (target_rows > BLOCK_ROWS_MAX) target_rows = BLOCK_ROWS_MAX;
        int64_t main_N = main_count;
        int64_t row_bytes = half;
        int64_t nblocks_main = (main_N + target_rows - 1) / target_rows;
        if (nblocks_main > 0) {
            float lblock[BLOCK_ROWS_MAX];
            if (g_mmap_base != NULL) {

                for (int64_t b = 0; b < nblocks_main; b++) {
                    int64_t row0 = b * target_rows;
                    int64_t rows = (row0 + target_rows <= main_N) ? target_rows : (main_N - row0);
                    const uint8_t* blk = g_mmap_base + off_base + (main_start + row0) * row_bytes;
                    memset(lblock, 0, (size_t)rows * sizeof(float));
                    st_int4_sgemv_fused_neon(blk,
                                             embed_scale_cached + main_start + row0,
                                             (int)rows, H, x_in, lblock);
                    for (int64_t i = 0; i < rows; i++) {
                        int id = (int)(main_start + row0 + i);
                        float lv = lblock[i];
                        topk_w_offer(main_heap, K_local, &main_heap_size, lv, id);
                    }
                }
            } else {
            uint8_t* bufs[2] = { g_blk_u8, g_blk_alt };
            int cur = 0;
            int64_t rows0 = (target_rows <= main_N) ? target_rows : main_N;
            prefetch_request(&g_pf, g_fd, bufs[cur],
                             (size_t)(rows0 * row_bytes),
                             off_base + main_start * row_bytes);
            for (int64_t b = 0; b < nblocks_main; b++) {
                int64_t row0 = b * target_rows;
                int64_t rows = (row0 + target_rows <= main_N) ? target_rows : (main_N - row0);
                if (prefetch_wait(&g_pf) != 0) {
                    for (int wi = 0; wi < g_n_workers; wi++) sgemv_worker_wait(&g_sw[wi]);
                    return -1;
                }
                uint8_t* this_raw = bufs[cur];
                if (b + 1 < nblocks_main) {
                    int next = 1 - cur;
                    int64_t nrow0 = (b + 1) * target_rows;
                    int64_t nrows = (nrow0 + target_rows <= main_N) ? target_rows : (main_N - nrow0);
                    prefetch_request(&g_pf, g_fd, bufs[next],
                                     (size_t)(nrows * row_bytes),
                                     off_base + (main_start + nrow0) * row_bytes);
                }
                memset(lblock, 0, (size_t)rows * sizeof(float));
                st_int4_sgemv_fused_neon(this_raw,
                                         embed_scale_cached + main_start + row0,
                                         (int)rows, H, x_in, lblock);
                for (int64_t i = 0; i < rows; i++) {
                    int id = (int)(main_start + row0 + i);
                    float lv = lblock[i];
                    topk_w_offer(main_heap, K_local, &main_heap_size, lv, id);
                }
                cur = 1 - cur;
            }
            }
        }
        for (int wi = 0; wi < g_n_workers; wi++) {
            if (sgemv_worker_wait(&g_sw[wi]) != 0) return -1;
        }

        for (int i = 0; i < main_heap_size; i++) {
            int id = main_heap[i].id;
            float lv = main_heap[i].logit;
            if (rep_penalty > 1.0f) {
                for (int r = 0; r < recent_count; r++) {
                    if (recent_ids[r] == id) {
                        lv = (lv > 0) ? (lv / rep_penalty) : (lv * rep_penalty);
                        break;
                    }
                }
            }
            topk_offer(out_top, K, &heap_size, lv, id);
        }
        for (int wi = 0; wi < g_n_workers; wi++) {
            int sz = g_sw[wi].job.topk_size_out;
            for (int i = 0; i < sz; i++) {
                int id = worker_heaps[wi][i].id;
                float lv = worker_heaps[wi][i].logit;
                if (rep_penalty > 1.0f) {
                    for (int r = 0; r < recent_count; r++) {
                        if (recent_ids[r] == id) {
                            lv = (lv > 0) ? (lv / rep_penalty) : (lv * rep_penalty);
                            break;
                        }
                    }
                }
                topk_offer(out_top, K, &heap_size, lv, id);
            }
        }
        return 0;
    }

    int64_t target_rows = ST_BLOCK_FP32_BYTES / (H * 4);
    if (target_rows < 1) target_rows = 1;
    if (target_rows > BLOCK_ROWS_MAX) target_rows = BLOCK_ROWS_MAX;
    float lblock[BLOCK_ROWS_MAX];

    int64_t nblocks = (VOCAB + target_rows - 1) / target_rows;
    uint8_t* bufs[2] = { g_blk_u8, g_blk_alt };
    int cur = 0;

    int64_t rows0 = (target_rows <= VOCAB) ? target_rows : VOCAB;
    prefetch_request(&g_pf, g_fd, bufs[cur],
                     (size_t)(rows0 * half), embed_w->abs_offset);

    for (int64_t b = 0; b < nblocks; b++) {
        int64_t row0 = b * target_rows;
        int64_t rows = (row0 + target_rows <= VOCAB) ? target_rows : (VOCAB - row0);

        if (prefetch_wait(&g_pf) != 0) return -1;
        uint8_t* this_u8 = bufs[cur];

        if (b + 1 < nblocks) {
            int next = 1 - cur;
            int64_t nrow0 = (b + 1) * target_rows;
            int64_t nrows = (nrow0 + target_rows <= VOCAB) ? target_rows : (VOCAB - nrow0);
            prefetch_request(&g_pf, g_fd, bufs[next],
                             (size_t)(nrows * half),
                             embed_w->abs_offset + nrow0 * half);
        }

        memset(lblock, 0, (size_t)rows * sizeof(float));
        st_int4_sgemv_fused_neon(this_u8, embed_scale_cached + row0,
                                 (int)rows, H, x_in, lblock);
        for (int64_t i = 0; i < rows; i++) {
            int id = (int)(row0 + i);
            float lv = lblock[i];
            if (rep_penalty > 1.0f) {
                for (int r = 0; r < recent_count; r++) {
                    if (recent_ids[r] == id) {
                        lv = (lv > 0) ? (lv / rep_penalty) : (lv * rep_penalty);
                        break;
                    }
                }
            }
            topk_offer(out_top, K, &heap_size, lv, id);
        }
        cur = 1 - cur;
    }
    return 0;
}

static int sample_top_k(TopEntry* top, int K, float temp) {
    if (temp <= 0.0f) {

        int best = 0;
        for (int i = 1; i < K; i++)
            if (top[i].logit > top[best].logit) best = i;
        return top[best].id;
    }

    float maxv = top[0].logit;
    for (int i = 1; i < K; i++) if (top[i].logit > maxv) maxv = top[i].logit;
    double sum = 0.0;
    float probs[256];
    for (int i = 0; i < K; i++) {
        probs[i] = expf((top[i].logit - maxv) / temp);
        sum += probs[i];
    }
    float u = pcg32_uniform() * (float)sum;
    float acc = 0.0f;
    for (int i = 0; i < K; i++) {
        acc += probs[i];
        if (acc >= u) return top[i].id;
    }
    return top[K-1].id;
}

static int sample_next_token(const TensorMeta* embed_w, const float* embed_scale_cached,
                             const float* x_in,
                             const int* recent_ids, int recent_count,
                             int top_k, float temp, float rep_penalty) {
    if (top_k > 256) top_k = 256;
    if (top_k < 1) top_k = 1;
    TopEntry top[256];
    if (topk_int4_lm_head(embed_w, embed_scale_cached, x_in,
                          recent_ids, recent_count, rep_penalty,
                          top, top_k) != 0) return -1;
    return sample_top_k(top, top_k, temp);
}

static int argmax_int4_lm_head(const TensorMeta* embed_w, const float* embed_scale_cached,
                               const float* x_in) {
    return sample_next_token(embed_w, embed_scale_cached, x_in, NULL, 0,
                             1, 0.0f, 1.0f);
}

static int forward_one_token(int input_token, int position,
                             const TensorMeta* embed_w, const float* embed_scale_cached,
                             const float* fnorm_w) {
    if (int4_lookup_row(embed_w, embed_scale_cached, input_token, g_x) < 0) return -1;
    for (int li = 0; li < NUM_LAYERS; li++) {
        memcpy(g_residual, g_x, sizeof g_x);
        {
            float scale = rms_scale(g_x, H, EPS);
            rms_apply(g_x, g_lsw[li].in_ln, scale, H, g_xn);
        }
        if (LAYER_TYPES[li] == 1) {
            if (forward_full_attn(li, position, g_xn, g_attn_out) != 0) return -1;
        } else {
            if (forward_linear_attn(li, g_xn, g_attn_out) != 0) return -1;
        }
        for (int i = 0; i < H; i++) g_x[i] = g_residual[i] + g_attn_out[i];

        memcpy(g_residual, g_x, sizeof g_x);
        {
            float scale = rms_scale(g_x, H, EPS);
            rms_apply(g_x, g_lsw[li].post_ln, scale, H, g_xn);
        }
        if (forward_mlp(li, g_xn, g_mlp_out) != 0) return -1;
        for (int i = 0; i < H; i++) g_x[i] = g_residual[i] + g_mlp_out[i];
    }
    {
        float scale = rms_scale(g_x, H, EPS);
        rms_apply(g_x, fnorm_w, scale, H, g_xn);
    }
    return 0;
}

static float g_mtp_x[H], g_mtp_xn[H], g_mtp_resid[H], g_mtp_attn_out[H];
static float g_mtp_mlp_out[H], g_mtp_mlp_gate[INTERM], g_mtp_mlp_up[INTERM];
static float g_mtp_mlp_acc[INTERM];
static float g_mtp_q_full[NQ * HEAD * 2];
static float g_mtp_q[NQ][HEAD], g_mtp_kf[NK_FULL][HEAD], g_mtp_vf[NK_FULL][HEAD];
static float g_mtp_gate[NQ * HEAD], g_mtp_attn_concat[NQ * HEAD];
static float g_mtp_fc_in[2 * H];

static int bf16_embed_lookup(int token_id, float* out_row);

static int forward_mtp_bf16(int prev_token_id, const float* h_main, int position,
                            const TensorMeta* embed_w, const float* embed_scale_cached,
                            int top_k_out, TopEntry* out_top);

static int mtp_full_attn(int position, const float* x_in, float* y_out) {
    if (sgemv_auto(&g_mtp.q_proj, x_in, g_mtp_q_full,        NQ * HEAD * 2, H) != 0) return -1;
    if (sgemv_auto(&g_mtp.k_proj, x_in, (float*)g_mtp_kf,    NK_FULL * HEAD, H) != 0) return -1;
    if (sgemv_auto(&g_mtp.v_proj, x_in, (float*)g_mtp_vf,    NK_FULL * HEAD, H) != 0) return -1;

    for (int h = 0; h < NQ; h++) {
        const float* row = g_mtp_q_full + h * (HEAD * 2);
        memcpy(g_mtp_q[h], row, HEAD * sizeof(float));
        memcpy(g_mtp_gate + h * HEAD, row + HEAD, HEAD * sizeof(float));
    }
    for (int h = 0; h < NQ; h++) {
        float scale = rms_scale(g_mtp_q[h], HEAD, EPS);
        rms_apply(g_mtp_q[h], g_mtp.q_norm, scale, HEAD, g_mtp_q[h]);
    }
    for (int h = 0; h < NK_FULL; h++) {
        float scale = rms_scale(g_mtp_kf[h], HEAD, EPS);
        rms_apply(g_mtp_kf[h], g_mtp.k_norm, scale, HEAD, g_mtp_kf[h]);
    }
    for (int h = 0; h < NQ; h++) rope_apply_inplace_fp32(g_mtp_q[h], position, HEAD, ROT_DIM, THETA);
    for (int h = 0; h < NK_FULL; h++) rope_apply_inplace_fp32(g_mtp_kf[h], position, HEAD, ROT_DIM, THETA);

    int kv_len = g_mtp_kv_len;
    if (kv_len + 1 > MAX_KV) {
        fprintf(stderr, "mtp kv overflow\n"); return -1;
    }
    float* K_arr = g_mtp_kv_cache;
    float* V_arr = g_mtp_kv_cache + (size_t)MAX_KV * NK_FULL * HEAD;
    size_t row_floats = (size_t)NK_FULL * HEAD;
    for (int h = 0; h < NK_FULL; h++) {
        memcpy(K_arr + (size_t)kv_len * row_floats + h * HEAD,
               g_mtp_kf[h], HEAD * sizeof(float));
        memcpy(V_arr + (size_t)kv_len * row_floats + h * HEAD,
               g_mtp_vf[h], HEAD * sizeof(float));
    }
    int new_len = kv_len + 1;

    float scale = 1.0f / sqrtf((float)HEAD);

    float logits[MAX_KV];
    for (int h = 0; h < NQ; h++) {
        int kv = h / (NQ / NK_FULL);
        const float* K_kv = K_arr + kv * HEAD;
        const float* V_kv = V_arr + kv * HEAD;
        const int lda = NK_FULL * HEAD;
        const float* qh = g_mtp_q[h];
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    new_len, HEAD,
                    scale, K_kv, lda,
                    qh, 1,
                    0.0f, logits, 1);
        softmax_fp32_inplace(logits, new_len);
        float head_out[HEAD];
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    new_len, HEAD,
                    1.0f, V_kv, lda,
                    logits, 1,
                    0.0f, head_out, 1);
        memcpy(g_mtp_attn_concat + h * HEAD, head_out, HEAD * sizeof(float));
    }

    for (int i = 0; i < NQ * HEAD; i++) {
        float gv = g_mtp_gate[i];
        g_mtp_attn_concat[i] *= 1.0f / (1.0f + expf(-gv));
    }
    if (sgemv_auto(&g_mtp.o_proj, g_mtp_attn_concat, y_out, H, NQ * HEAD) != 0) return -1;
    g_mtp_kv_len = new_len;
    return 0;
}

static int mtp_mlp(const float* x_in, float* y_out) {
    if (sgemv_auto(&g_mtp.gate_proj, x_in, g_mtp_mlp_gate, INTERM, H) != 0) return -1;
    if (sgemv_auto(&g_mtp.up_proj,   x_in, g_mtp_mlp_up,   INTERM, H) != 0) return -1;
    swiglu_fp32(g_mtp_mlp_gate, g_mtp_mlp_up, g_mtp_mlp_acc, INTERM);
    if (sgemv_auto(&g_mtp.down_proj, g_mtp_mlp_acc, y_out, H, INTERM) != 0) return -1;
    return 0;
}

static int forward_mtp(int prev_token_id, const float* h_main, int position,
                       const TensorMeta* embed_w, const float* embed_scale_cached,
                       int top_k_out, TopEntry* out_top) {

    float e[H];
    if (g_bf16_fd >= 0) {
        if (bf16_embed_lookup(prev_token_id, e) < 0) return -1;
    } else {
        if (int4_lookup_row(embed_w, embed_scale_cached, prev_token_id, e) < 0) return -1;
    }

    float e_n[H], h_n[H];
    {
        float s_e = rms_scale(e, H, EPS);
        rms_apply(e, g_mtp.pre_fc_e_norm, s_e, H, e_n);
        float s_h = rms_scale(h_main, H, EPS);
        rms_apply(h_main, g_mtp.pre_fc_h_norm, s_h, H, h_n);
        if (getenv("STRATUM_MTP_TRACE")) {
            fprintf(stderr, "[mtp_trace] e[0..7]    = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7]);
            fprintf(stderr, "[mtp_trace] e_n[0..7]  = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f (s_e=%.6f)\n",
                    e_n[0],e_n[1],e_n[2],e_n[3],e_n[4],e_n[5],e_n[6],e_n[7], s_e);
            fprintf(stderr, "[mtp_trace] h_main[0..7] = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    h_main[0],h_main[1],h_main[2],h_main[3],h_main[4],h_main[5],h_main[6],h_main[7]);
            fprintf(stderr, "[mtp_trace] h_n[0..7]    = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f (s_h=%.6f)\n",
                    h_n[0],h_n[1],h_n[2],h_n[3],h_n[4],h_n[5],h_n[6],h_n[7], s_h);
        }
    }

    memcpy(g_mtp_fc_in,        e_n, H * sizeof(float));
    memcpy(g_mtp_fc_in + H,    h_n, H * sizeof(float));

    const uint8_t* save_main_mmap = g_mmap_base;
    g_mmap_base = g_mtp_mmap_base;

    if (sgemv_auto(&g_mtp.fc, g_mtp_fc_in, g_mtp_x, H, 2 * H) != 0) {
        g_mmap_base = save_main_mmap; return -1;
    }

    if (g_bf16_fd >= 0 && getenv("STRATUM_MTP_BF16_FC") != NULL) {
        TensorMeta* fc_bf16 = st_index_lookup(&g_bf16_ix, "mtp.fc.weight");
        if (fc_bf16) {
            static uint16_t fc_bf[H * 2 * H];
            static float    fc_fp[H * 2 * H];
            if (st_pread_full(g_bf16_fd, fc_bf, fc_bf16->nbytes, fc_bf16->abs_offset) == 0) {
                st_bf16_to_fp32(fc_bf, fc_fp, (size_t)H * 2 * H);
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            H, 2 * H, 1.0f, fc_fp, 2 * H,
                            g_mtp_fc_in, 1, 0.0f, g_mtp_x, 1);
            }
        }
    }
    if (getenv("STRATUM_MTP_TRACE")) {
        fprintf(stderr, "[mtp_trace] after_fc: x[0..7] = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                g_mtp_x[0], g_mtp_x[1], g_mtp_x[2], g_mtp_x[3],
                g_mtp_x[4], g_mtp_x[5], g_mtp_x[6], g_mtp_x[7]);
    }

    memcpy(g_mtp_resid, g_mtp_x, sizeof g_mtp_x);
    {
        float s = rms_scale(g_mtp_x, H, EPS);
        rms_apply(g_mtp_x, g_mtp.input_ln, s, H, g_mtp_xn);
    }
    if (mtp_full_attn(position, g_mtp_xn, g_mtp_attn_out) != 0) {
        g_mmap_base = save_main_mmap; return -1;
    }
    for (int i = 0; i < H; i++) g_mtp_x[i] = g_mtp_resid[i] + g_mtp_attn_out[i];

    memcpy(g_mtp_resid, g_mtp_x, sizeof g_mtp_x);
    {
        float s = rms_scale(g_mtp_x, H, EPS);
        rms_apply(g_mtp_x, g_mtp.post_ln, s, H, g_mtp_xn);
    }
    if (mtp_mlp(g_mtp_xn, g_mtp_mlp_out) != 0) {
        g_mmap_base = save_main_mmap; return -1;
    }
    for (int i = 0; i < H; i++) g_mtp_x[i] = g_mtp_resid[i] + g_mtp_mlp_out[i];

    {
        float s = rms_scale(g_mtp_x, H, EPS);
        rms_apply(g_mtp_x, g_mtp.final_norm, s, H, g_mtp_xn);
    }

    g_mmap_base = save_main_mmap;

    int heap_size = 0;
    (void)heap_size;
    if (topk_int4_lm_head(embed_w, embed_scale_cached, g_mtp_xn,
                           NULL, 0, 1.0f,
                           out_top, top_k_out) != 0) return -1;
    return 0;
}

static int forward_mtp_bf16(int prev_token_id, const float* h_main, int position,
                            const TensorMeta* embed_w, const float* embed_scale_cached,
                            int top_k_out, TopEntry* out_top) {
    (void)embed_w; (void)embed_scale_cached;
    if (g_bf16_fd < 0) return -1;

#define READ_BF16(name, out, n_floats) do {                              \
        TensorMeta* _m = st_index_lookup(&g_bf16_ix, (name));           \
        if (!_m) { fprintf(stderr, "missing %s\n", (name)); return -1; }\
        if ((size_t)_m->nbytes != (size_t)(n_floats) * 2) {              \
            fprintf(stderr, "size mismatch for %s\n", (name)); return -1;\
        }                                                                \
        static uint16_t _scratch[16 * 1024 * 1024 / 2];                  \
        if ((size_t)_m->nbytes > sizeof(_scratch)) {                     \
            fprintf(stderr, "scratch overflow for %s\n", (name)); return -1; \
        }                                                                \
        if (st_pread_full(g_bf16_fd, _scratch, _m->nbytes, _m->abs_offset) < 0) return -1; \
        st_bf16_to_fp32(_scratch, (out), (size_t)(n_floats));            \
    } while (0)

    float e[H];
    if (bf16_embed_lookup(prev_token_id, e) < 0) return -1;

    float pre_e_w[H], pre_h_w[H];
    READ_BF16("mtp.pre_fc_norm_embedding.weight", pre_e_w, H);
    READ_BF16("mtp.pre_fc_norm_hidden.weight",    pre_h_w, H);
    float e_n[H], h_n[H];
    {
        float s_e = rms_scale(e, H, EPS);
        rms_apply(e, pre_e_w, s_e, H, e_n);
        float s_h = rms_scale(h_main, H, EPS);
        rms_apply(h_main, pre_h_w, s_h, H, h_n);
    }

    static float fc_fp[H * 2 * H];
    READ_BF16("mtp.fc.weight", fc_fp, H * 2 * H);
    float fc_in[2 * H];
    memcpy(fc_in, e_n, H * sizeof(float));
    memcpy(fc_in + H, h_n, H * sizeof(float));
    float x[H];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, H, 2 * H,
                1.0f, fc_fp, 2 * H, fc_in, 1, 0.0f, x, 1);

    float in_ln[H], post_ln[H], q_norm[HEAD], k_norm[HEAD], final_norm[H];
    READ_BF16("mtp.layers.0.input_layernorm.weight", in_ln, H);
    READ_BF16("mtp.layers.0.post_attention_layernorm.weight", post_ln, H);
    READ_BF16("mtp.layers.0.self_attn.q_norm.weight", q_norm, HEAD);
    READ_BF16("mtp.layers.0.self_attn.k_norm.weight", k_norm, HEAD);
    READ_BF16("mtp.norm.weight", final_norm, H);

    static float q_w[NQ * HEAD * 2 * H];
    static float k_w[NK_FULL * HEAD * H];
    static float v_w[NK_FULL * HEAD * H];
    static float o_w[H * NQ * HEAD];
    READ_BF16("mtp.layers.0.self_attn.q_proj.weight", q_w, (size_t)NQ * HEAD * 2 * H);
    READ_BF16("mtp.layers.0.self_attn.k_proj.weight", k_w, (size_t)NK_FULL * HEAD * H);
    READ_BF16("mtp.layers.0.self_attn.v_proj.weight", v_w, (size_t)NK_FULL * HEAD * H);
    READ_BF16("mtp.layers.0.self_attn.o_proj.weight", o_w, (size_t)H * NQ * HEAD);

    static float gate_w[INTERM * H];
    static float up_w[INTERM * H];
    static float down_w[H * INTERM];
    READ_BF16("mtp.layers.0.mlp.gate_proj.weight", gate_w, (size_t)INTERM * H);
    READ_BF16("mtp.layers.0.mlp.up_proj.weight",   up_w,   (size_t)INTERM * H);
    READ_BF16("mtp.layers.0.mlp.down_proj.weight", down_w, (size_t)H * INTERM);

    float resid[H], xn[H], q_full[NQ * HEAD * 2];
    float kf[NK_FULL][HEAD], vf[NK_FULL][HEAD];
    float q[NQ][HEAD], gate[NQ * HEAD];
    float attn_concat[NQ * HEAD];

    memcpy(resid, x, sizeof x);
    {
        float s = rms_scale(x, H, EPS);
        rms_apply(x, in_ln, s, H, xn);
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, NQ * HEAD * 2, H,
                1.0f, q_w, H, xn, 1, 0.0f, q_full, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, NK_FULL * HEAD, H,
                1.0f, k_w, H, xn, 1, 0.0f, (float*)kf, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, NK_FULL * HEAD, H,
                1.0f, v_w, H, xn, 1, 0.0f, (float*)vf, 1);
    for (int h = 0; h < NQ; h++) {
        const float* row = q_full + h * (HEAD * 2);
        memcpy(q[h], row, HEAD * sizeof(float));
        memcpy(gate + h * HEAD, row + HEAD, HEAD * sizeof(float));
    }
    for (int h = 0; h < NQ; h++) {
        float s = rms_scale(q[h], HEAD, EPS);
        rms_apply(q[h], q_norm, s, HEAD, q[h]);
    }
    for (int h = 0; h < NK_FULL; h++) {
        float s = rms_scale(kf[h], HEAD, EPS);
        rms_apply(kf[h], k_norm, s, HEAD, kf[h]);
    }
    for (int h = 0; h < NQ; h++) rope_apply_inplace_fp32(q[h], position, HEAD, ROT_DIM, THETA);
    for (int h = 0; h < NK_FULL; h++) rope_apply_inplace_fp32(kf[h], position, HEAD, ROT_DIM, THETA);

    for (int h = 0; h < NQ; h++) {
        int kv = h / (NQ / NK_FULL);
        memcpy(attn_concat + h * HEAD, vf[kv], HEAD * sizeof(float));
    }
    for (int i = 0; i < NQ * HEAD; i++) {
        float gv = gate[i];
        attn_concat[i] *= 1.0f / (1.0f + expf(-gv));
    }
    float attn_out[H];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, H, NQ * HEAD,
                1.0f, o_w, NQ * HEAD, attn_concat, 1, 0.0f, attn_out, 1);
    for (int i = 0; i < H; i++) x[i] = resid[i] + attn_out[i];

    memcpy(resid, x, sizeof resid);
    {
        float s = rms_scale(x, H, EPS);
        rms_apply(x, post_ln, s, H, xn);
    }
    static float gate_act[INTERM], up_act[INTERM], mlp_acc[INTERM];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, INTERM, H,
                1.0f, gate_w, H, xn, 1, 0.0f, gate_act, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, INTERM, H,
                1.0f, up_w, H, xn, 1, 0.0f, up_act, 1);
    swiglu_fp32(gate_act, up_act, mlp_acc, INTERM);
    float mlp_out[H];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, H, INTERM,
                1.0f, down_w, INTERM, mlp_acc, 1, 0.0f, mlp_out, 1);
    for (int i = 0; i < H; i++) x[i] = resid[i] + mlp_out[i];

    {
        float s = rms_scale(x, H, EPS);
        rms_apply(x, final_norm, s, H, xn);
    }

    if (topk_int4_lm_head(embed_w, embed_scale_cached, xn,
                           NULL, 0, 1.0f,
                           out_top, top_k_out) != 0) return -1;
    return 0;
#undef READ_BF16
}

static void mtp_reset_kv(void) {
    g_mtp_kv_len = 0;
}

static int bf16_embed_lookup(int token_id, float* out_row) {
    if (g_bf16_fd < 0 || g_bf16_embed_off == 0) return -1;
    uint16_t scratch[H];
    int64_t off = g_bf16_embed_off + (int64_t)token_id * H * 2;
    if (st_pread_full(g_bf16_fd, scratch, H * 2, off) < 0) return -1;
    st_bf16_to_fp32(scratch, out_row, H);
    return 0;
}

static int g_tok_pid = -1;
static int g_tok_in  = -1;
static FILE* g_tok_out = NULL;

static int tok_server_start(const char* python, const char* helper) {
    int to_py[2], from_py[2];
    if (pipe(to_py) != 0 || pipe(from_py) != 0) { perror("pipe"); return -1; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }
    if (pid == 0) {

        dup2(to_py[0], 0);
        dup2(from_py[1], 1);

        close(to_py[0]); close(to_py[1]);
        close(from_py[0]); close(from_py[1]);
        execl(python, python, helper, (char*)NULL);
        perror("execl");
        _exit(127);
    }

    close(to_py[0]); close(from_py[1]);
    g_tok_pid = pid;
    g_tok_in  = to_py[1];
    g_tok_out = fdopen(from_py[0], "r");
    if (!g_tok_out) { perror("fdopen"); return -1; }

    char line[256];
    if (!fgets(line, sizeof line, g_tok_out)) {
        fprintf(stderr, "tokenizer server failed to start\n");
        return -1;
    }
    if (strncmp(line, "READY", 5) != 0) {
        fprintf(stderr, "tokenizer server unexpected: %s", line);
        return -1;
    }
    return 0;
}

static void tok_server_stop(void) {
    if (g_tok_in >= 0) {
        const char* q = "QUIT\n";
        if (write(g_tok_in, q, strlen(q)) > 0) {
            char line[16];
            (void)fgets(line, sizeof line, g_tok_out);
        }
        close(g_tok_in); g_tok_in = -1;
    }
    if (g_tok_out) { fclose(g_tok_out); g_tok_out = NULL; }
    if (g_tok_pid > 0) {
        int status;
        waitpid(g_tok_pid, &status, 0);
        g_tok_pid = -1;
    }
}

static int tok_send(const char* s) {
    size_t n = strlen(s);
    while (n > 0) {
        ssize_t w = write(g_tok_in, s, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            perror("write tok"); return -1;
        }
        s += w; n -= (size_t)w;
    }
    return 0;
}

static int tokenize_encode(const char* prompt, int* out_ids, int max_ids) {

    size_t plen = strlen(prompt);
    char* msg = (char*)malloc(plen + 16);
    if (!msg) return -1;
    size_t pos = 0;
    memcpy(msg + pos, "ENCODE ", 7); pos += 7;
    for (size_t i = 0; i < plen; i++) {
        char c = prompt[i];
        if (c == '\n' || c == '\r') c = ' ';
        msg[pos++] = c;
    }
    msg[pos++] = '\n';
    msg[pos] = 0;
    if (tok_send(msg) != 0) { free(msg); return -1; }
    free(msg);

    char line[16384];
    if (!fgets(line, sizeof line, g_tok_out)) return -1;
    int n = 0;
    char* p = line;
    while (*p && n < max_ids) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
        if (!*p) break;
        out_ids[n++] = (int)strtol(p, &p, 10);
    }
    return n;
}

static char* tokenize_decode_one(int id, size_t* out_len) {
    char buf[64];
    int len = snprintf(buf, sizeof buf, "DECODE_NL %d\n", id);
    if (write(g_tok_in, buf, (size_t)len) < 0) return NULL;

    size_t cap = 64, n = 0;
    char* out = (char*)malloc(cap);
    if (!out) return NULL;
    int fd = fileno(g_tok_out);

    while (1) {
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r < 0) {
            if (errno == EINTR) continue;
            free(out); return NULL;
        }
        if (r == 0) break;
        if (c == '\x00') break;
        if (n + 1 >= cap) {
            cap *= 2;
            char* nb = (char*)realloc(out, cap);
            if (!nb) { free(out); return NULL; }
            out = nb;
        }
        out[n++] = c;
    }
    out[n] = 0;
    if (out_len) *out_len = n;
    return out;
}

int main(int argc, char** argv) {
    const char* model_path  = NULL;   /* required via --model */
    const char* state_path  = "/tmp/.stratum-runtime-state-p35";
    const char* python      = "python3";
    const char* helper      = "native/stratum_tokenize_server.py";

    int gen = 32;
    int top_k = 40;
    float temp = 0.8f;
    float rep_penalty = 1.15f;
    int rep_window = 64;
    uint64_t seed = 0xC0DE;
    bool use_mmap = true;
    bool cache_bf16 = false;

    bool enable_mtp = false;
    const char* mtp_model_path = NULL;   /* optional, via --mtp-model */
    const char* ids_arg = NULL;
    int agree_mode = 0;

    const char* bf16_model_path = NULL;  /* optional, via --bf16-model */
    const char* prompt = NULL;
    int ai = 1;
    while (ai < argc) {
        if (strcmp(argv[ai], "-n") == 0 && ai + 1 < argc) {
            gen = atoi(argv[ai + 1]); ai += 2;
        } else if (strcmp(argv[ai], "--python") == 0 && ai + 1 < argc) {
            python = argv[ai + 1]; ai += 2;
        } else if (strcmp(argv[ai], "--model") == 0 && ai + 1 < argc) {
            model_path = argv[ai + 1]; ai += 2;
        } else if (strcmp(argv[ai], "--mtp-model") == 0 && ai + 1 < argc) {
            mtp_model_path = argv[ai + 1]; ai += 2;
        } else if (strcmp(argv[ai], "--bf16-model") == 0 && ai + 1 < argc) {
            bf16_model_path = argv[ai + 1]; ai += 2;
        } else if (strcmp(argv[ai], "--temp") == 0 && ai + 1 < argc) {
            temp = (float)atof(argv[ai + 1]); ai += 2;
        } else if (strcmp(argv[ai], "--top-k") == 0 && ai + 1 < argc) {
            top_k = atoi(argv[ai + 1]); ai += 2;
        } else if (strcmp(argv[ai], "--rep") == 0 && ai + 1 < argc) {
            rep_penalty = (float)atof(argv[ai + 1]); ai += 2;
        } else if (strcmp(argv[ai], "--rep-window") == 0 && ai + 1 < argc) {
            rep_window = atoi(argv[ai + 1]); ai += 2;
        } else if (strcmp(argv[ai], "--seed") == 0 && ai + 1 < argc) {
            seed = (uint64_t)strtoull(argv[ai + 1], NULL, 0); ai += 2;
        } else if (strcmp(argv[ai], "--no-mmap") == 0) {
            use_mmap = false; ai++;
        } else if (strcmp(argv[ai], "--mmap") == 0) {
            use_mmap = true; ai++;
        } else if (strcmp(argv[ai], "--strict-10mb") == 0) {
            cache_bf16 = false; ai++;
        } else if (strcmp(argv[ai], "--cache-bf16") == 0) {
            cache_bf16 = true; ai++;
        } else if (strcmp(argv[ai], "--enable-mtp") == 0) {
            enable_mtp = true; ai++;
        } else if (strcmp(argv[ai], "--no-mtp") == 0) {
            enable_mtp = false; ai++;
        } else if (strcmp(argv[ai], "--ids") == 0 && ai + 1 < argc) {
            ids_arg = argv[ai + 1]; ai += 2;
        } else if (strcmp(argv[ai], "--agree") == 0) {
            agree_mode = 1; ai++;
        } else {
            prompt = argv[ai]; ai++;
        }
    }
    if (ids_arg && !prompt) prompt = "(ids)";
    if (!model_path) {
        fprintf(stderr, "usage: %s [-n GEN] --model PATH [--mtp-model PATH] [--bf16-model PATH] "
                        "[--temp T] [--top-k K] [--rep P] [--rep-window N] [--seed S] "
                        "\"prompt text\"  (no default model — pass your path)\n", argv[0]);
        return 1;
    }
    if (!prompt) {
        fprintf(stderr, "usage: %s [-n GEN] --model PATH [--temp T] "
                        "[--top-k K] [--rep P] [--rep-window N] [--seed S] "
                        "\"prompt text\"\n", argv[0]);
        return 1;
    }
    if (rep_window > 256) rep_window = 256;

    pcg32_seed(seed);

    fprintf(stderr, "== stratum_p35 — MTP draft head infrastructure ==\n");
    fprintf(stderr, "  bf16 cache: %s (%s)\n",
            cache_bf16 ? "ENABLED" : "DISABLED",
            cache_bf16 ? "fastest, ~10.6 MB anon" : "strict-10mb, ~8.5 MB anon");
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    fprintf(stderr, "  model     : %s\n", model_path);
    fprintf(stderr, "  top_k=%d  temp=%.2f  rep_penalty=%.2f  rep_window=%d  seed=0x%llx\n",
            top_k, temp, rep_penalty, rep_window, (unsigned long long)seed);
    fprintf(stderr, "  prompt    : %s\n", prompt);
    fprintf(stderr, "  generate  : %d tokens\n", gen);

    if (tok_server_start(python, helper) != 0) {
        if (!ids_arg) { fprintf(stderr, "tok server start failed\n"); return 1; }
        fprintf(stderr, "  tok server unavailable; --ids mode (no detokenize)\n");
    }

    int ids[2048];
    int n_ids = 0;
    if (ids_arg) {
        const char* p = ids_arg;
        while (*p && n_ids < 2048) {
            while (*p == ' ' || *p == ',') p++;
            if (!*p) break;
            ids[n_ids++] = (int)strtol(p, (char**)&p, 10);
        }
        fprintf(stderr, "  raw ids (%d) provided\n", n_ids);
    } else {
        n_ids = tokenize_encode(prompt, ids, 2048);
    }
    if (n_ids <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    fprintf(stderr, "  prompt ids (%d): ", n_ids);
    for (int i = 0; i < n_ids && i < 16; i++) fprintf(stderr, "%d ", ids[i]);
    fprintf(stderr, "%s\n", n_ids > 16 ? "..." : "");
    if (n_ids + gen > MAX_KV) {
        fprintf(stderr, "n_ids+gen exceeds MAX_KV (%d)\n", MAX_KV); return 1;
    }

    bool nocache = false;
    if (getenv("STRATUM_NOCACHE")) nocache = (atoi(getenv("STRATUM_NOCACHE")) != 0);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nocache") == 0) nocache = true;
        if (strcmp(argv[i], "--cache") == 0)   nocache = false;
    }
    fprintf(stderr, "  weight cache : %s\n", nocache ? "F_NOCACHE (RSS-bounded)" : "page cache (speed-optimized)");

    if (st_open_and_index(model_path, nocache, &g_fd, &g_ix) != 0) return 1;

    if (use_mmap) {
        struct stat st;
        if (fstat(g_fd, &st) != 0) { perror("fstat"); return 1; }
        g_mmap_size = (size_t)st.st_size;
        void* p = mmap(NULL, g_mmap_size, PROT_READ, MAP_PRIVATE, g_fd, 0);
        if (p == MAP_FAILED) {
            perror("mmap");
            fprintf(stderr, "  mmap failed; falling back to pread path\n");
            g_mmap_base = NULL;
        } else {
            g_mmap_base = (const uint8_t*)p;

            madvise(p, g_mmap_size, MADV_WILLNEED);
            volatile uint8_t sink = 0;
            const size_t PAGE = 16384;
            for (size_t off = 0; off < g_mmap_size; off += PAGE) {
                sink ^= ((const uint8_t*)p)[off];
            }
            (void)sink;
            fprintf(stderr, "  mmap prefault: %.1f MB swept\n",
                    g_mmap_size / 1024.0 / 1024.0);
        }
    }
    fprintf(stderr, "  mmap path: %s\n", g_mmap_base ? "enabled" : "disabled");

    if (enable_mtp) {
        if (st_open_and_index(mtp_model_path, nocache, &g_mtp_fd, &g_mtp_ix) != 0) {
            fprintf(stderr, "  failed to open MTP model %s — disabling MTP\n",
                    mtp_model_path);
            enable_mtp = false;
        } else if (use_mmap) {
            struct stat mst;
            if (fstat(g_mtp_fd, &mst) != 0) { perror("fstat mtp"); }
            else {
                g_mtp_mmap_size = (size_t)mst.st_size;
                void* mp = mmap(NULL, g_mtp_mmap_size, PROT_READ, MAP_PRIVATE,
                                g_mtp_fd, 0);
                if (mp == MAP_FAILED) {
                    perror("mmap mtp");
                    g_mtp_mmap_base = NULL;
                } else {
                    g_mtp_mmap_base = (const uint8_t*)mp;
                    madvise(mp, g_mtp_mmap_size, MADV_WILLNEED);
                    volatile uint8_t sink = 0;
                    const size_t PAGE = 16384;
                    for (size_t off = 0; off < g_mtp_mmap_size; off += PAGE) {
                        sink ^= ((const uint8_t*)mp)[off];
                    }
                    (void)sink;
                    fprintf(stderr, "  mtp mmap prefault: %.1f MB swept\n",
                            g_mtp_mmap_size / 1024.0 / 1024.0);
                }
            }
        }
        g_mtp_enabled = enable_mtp;

        if (g_mtp_enabled) {
            if (st_open_and_index(bf16_model_path, true, &g_bf16_fd, &g_bf16_ix) != 0) {
                fprintf(stderr, "  bf16 ref open failed (%s) — MTP embed lookup will fall back to INT4\n",
                        bf16_model_path);
                g_bf16_fd = -1;
            } else {
                TensorMeta* etok = st_index_lookup(&g_bf16_ix,
                    "model.language_model.embed_tokens.weight");
                if (etok && strcmp(etok->dtype, "BF16") == 0) {
                    g_bf16_embed_off = etok->abs_offset;
                    fprintf(stderr, "  bf16 embed offset: 0x%llx\n",
                            (unsigned long long)g_bf16_embed_off);
                } else {
                    fprintf(stderr, "  bf16 embed not found or wrong dtype\n");
                    g_bf16_embed_off = 0;
                }
            }
        }
    }
    fprintf(stderr, "  mtp: %s\n", g_mtp_enabled ? "ENABLED" : "disabled");
    g_state_fd = open_state_file(state_path);
    if (g_state_fd < 0) return 1;

    g_state_mmap_size = (size_t)NUM_LAYERS * SLOT_BYTES;
    void* sp = mmap(NULL, g_state_mmap_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, g_state_fd, 0);
    if (sp == MAP_FAILED) {
        perror("mmap state");
        fprintf(stderr, "  state mmap failed; pread/pwrite fallback\n");
        g_state_mmap = NULL;
    } else {
        g_state_mmap = (uint8_t*)sp;

        memset(g_state_mmap, 0, g_state_mmap_size);
        fprintf(stderr, "  state mmap: %zu MB\n",
                g_state_mmap_size / (1024 * 1024));
    }

    static uint8_t shared_raw[BLOCK_ROWS_MAX * H * 2];
    static uint8_t shared_alt[BLOCK_ROWS_MAX * H * 2];

    static uint8_t worker_bufs[MAX_SGEMV_WORKERS * 2][WORKER_BLOCK_ROWS_MAX * (H / 2)];
    g_blk_bf16 = (uint16_t*)shared_raw;
    g_blk_u8   = shared_raw;
    g_blk_alt  = shared_alt;
    g_blk_fp32 = (float*)   malloc((size_t)BLOCK_ROWS_MAX * H * 4);
    if (!g_blk_fp32) return 1;
    if (prefetch_init(&g_pf) != 0) {
        fprintf(stderr, "prefetch_init failed\n"); return 1;
    }

    g_n_workers = 7;
    if (getenv("STRATUM_PIPELINES")) {
        int p = atoi(getenv("STRATUM_PIPELINES"));
        if (p < 1) p = 1;
        if (p > 1 + MAX_SGEMV_WORKERS) p = 1 + MAX_SGEMV_WORKERS;
        g_n_workers = p - 1;
    }

    for (int wi = 0; wi < g_n_workers; wi++) {
        if (sgemv_worker_init(&g_sw[wi],
                              worker_bufs[wi * 2],
                              worker_bufs[wi * 2 + 1]) != 0) {
            fprintf(stderr, "sgemv_worker_init failed\n"); return 1;
        }
    }
    g_sw_inited = (g_n_workers > 0);
    fprintf(stderr, "  pipelines: %d (main + %d workers, threshold %d rows)\n",
            g_n_workers + 1, g_n_workers, SW_SPLIT_THRESHOLD);
    g_cache_bf16 = cache_bf16;
    if (load_layer_weights() != 0) return 1;
    fprintf(stderr, "  RSS @ smallW : %ld KB\n", st_peak_rss_kb());

    if (g_mtp_enabled) {
        if (load_mtp_weights() != 0) {
            fprintf(stderr, "  MTP load failed — disabling\n");
            g_mtp_enabled = false;
        } else {
            fprintf(stderr, "  MTP weights loaded (mmap %p)\n",
                    (const void*)g_mtp_mmap_base);
        }
    }
    fprintf(stderr, "  scale arena : %zu / %zu KB used\n",
            g_scale_arena_used * 4 / 1024,
            sizeof(g_scale_arena) / 1024);

    TensorMeta* embed_w = must_find("model.language_model.embed_tokens.weight");
    TensorMeta* embed_s = must_find("model.language_model.embed_tokens.weight.scale");

    {
        enum { CHUNK = 8192 };
        uint16_t scratch[CHUNK];
        int64_t off = embed_s->abs_offset;
        int64_t left = VOCAB;
        float* dst = g_embed_scale;
        while (left > 0) {
            int64_t n = (left < CHUNK) ? left : CHUNK;
            if (st_pread_full(g_fd, scratch, (size_t)(n * 2), off) < 0) return 1;
            for (int64_t i = 0; i < n; i++) dst[i] = st_f16_to_f32(scratch[i]);
            dst += n; off += n * 2; left -= n;
        }
    }
    const float* embed_scale_cached = g_embed_scale;
    fprintf(stderr, "  embed scale cache: %.2f MB\n",
            (double)EMBED_SCALE_FLOATS * 4.0 / 1024 / 1024);
    TensorMeta* fnorm = must_find("model.language_model.norm.weight");
    float fnorm_w[H];
    { uint16_t s[H]; if (load_bf16_meta(fnorm, fnorm_w, s) != 0) return 1; }

    if (agree_mode) {
        /* Teacher-force the provided sequence; at each position compare the
         * 0.8B greedy argmax to the actual next id (the 27B's token). The
         * match rate is the expected speculative-decode acceptance per token. */
        int matches = 0, total = 0;
        int pos = 0;
        for (int t = 0; t + 1 < n_ids; t++) {
            if (forward_one_token(ids[t], pos, embed_w, embed_scale_cached, fnorm_w) != 0) return 1;
            pos++;
            int pred = argmax_int4_lm_head(embed_w, embed_scale_cached, g_xn);
            int actual = ids[t + 1];
            if (pred == actual) matches++;
            total++;
            fprintf(stderr, "  pos %2d: ctx=%-7d pred=%-7d actual=%-7d %s\n",
                    t, ids[t], pred, actual, pred == actual ? "HIT" : "");
        }
        fprintf(stderr, "\n  [agree] 0.8B predicted 27B next-token %d/%d = %.1f%%\n",
                matches, total, total ? 100.0 * matches / total : 0.0);
        return 0;
    }

    double t_pre0 = st_now_seconds();
    int position = 0;
    for (int t = 0; t < n_ids; t++) {
        if (forward_one_token(ids[t], position, embed_w, embed_scale_cached, fnorm_w) != 0) return 1;
        position++;
        fprintf(stderr, "."); fflush(stderr);
    }
    fprintf(stderr, " prefill done in %.2fs\n\n", st_now_seconds() - t_pre0);

    fprintf(stderr, "%s", prompt);
    fflush(stderr);
    int gen_ids[2048]; int n_gen = 0;

    int recent[256];
    int recent_count = 0;
    int rstart = (n_ids > rep_window) ? (n_ids - rep_window) : 0;
    for (int i = rstart; i < n_ids; i++) recent[recent_count++] = ids[i];

    int next_tok = sample_next_token(embed_w, embed_scale_cached, g_xn,
                                     recent, recent_count,
                                     top_k, temp, rep_penalty);
    if (next_tok < 0) return 1;
    gen_ids[n_gen++] = next_tok;
    if (recent_count < rep_window) recent[recent_count++] = next_tok;
    else { memmove(recent, recent + 1, sizeof(int) * (rep_window - 1)); recent[rep_window - 1] = next_tok; }
    {
        size_t flen;
        char* frag = tokenize_decode_one(next_tok, &flen);
        if (frag) { fwrite(frag, 1, flen, stderr); fflush(stderr); free(frag); }
    }

    int mtp_draft = -1;
    int mtp_total = 0, mtp_top1_match = 0, mtp_top5_match = 0;
    bool use_bf16_mtp = (getenv("STRATUM_MTP_BF16") != NULL);
    if (g_mtp_enabled) {
        mtp_reset_kv();
        TopEntry mtp_top[8];
        if (getenv("STRATUM_MTP_DUMP_HIDDEN")) {
            fprintf(stderr, "[mtp_dump] prev_tok=%d position=%d\n", next_tok, position);
            fprintf(stderr, "[mtp_dump] hidden=");
            for (int i = 0; i < H; i++) fprintf(stderr, "%.6f ", g_xn[i]);
            fprintf(stderr, "\n");
        }
        int rc;
        if (use_bf16_mtp) {
            rc = forward_mtp_bf16(next_tok, g_xn, position,
                                  embed_w, embed_scale_cached, 5, mtp_top);
        } else {
            rc = forward_mtp(next_tok, g_xn, position,
                             embed_w, embed_scale_cached, 5, mtp_top);
        }
        if (rc == 0) {
            int best = 0;
            for (int i = 1; i < 5; i++) if (mtp_top[i].logit > mtp_top[best].logit) best = i;
            mtp_draft = mtp_top[best].id;
            if (getenv("STRATUM_MTP_DEBUG")) {
                fprintf(stderr, "[mtp_topk] ");
                for (int i = 0; i < 5; i++)
                    fprintf(stderr, "%d=%d(%.3f) ", i, mtp_top[i].id, mtp_top[i].logit);
                fprintf(stderr, "\n");
            }
        }
    }

    double t_gen0 = st_now_seconds();
    for (int g = 1; g < gen; g++) {
        if (forward_one_token(next_tok, position, embed_w, embed_scale_cached, fnorm_w) != 0) return 1;
        position++;
        int prev_for_mtp = next_tok;
        next_tok = sample_next_token(embed_w, embed_scale_cached, g_xn,
                                     recent, recent_count,
                                     top_k, temp, rep_penalty);
        gen_ids[n_gen++] = next_tok;

        if (g_mtp_enabled && mtp_draft >= 0) {
            mtp_total++;
            if (mtp_draft == next_tok) mtp_top1_match++;
            if (getenv("STRATUM_MTP_DEBUG")) {
                size_t fl1=0, fl2=0;
                char* main_tok = tokenize_decode_one(next_tok, &fl1);
                char* draft_tok = tokenize_decode_one(mtp_draft, &fl2);
                fprintf(stderr, "[MTP %d] main=%d(%s) draft=%d(%s) %s\n",
                        position, next_tok, main_tok ? main_tok : "?",
                        mtp_draft, draft_tok ? draft_tok : "?",
                        mtp_draft == next_tok ? "MATCH" : "miss");
                if (main_tok) free(main_tok);
                if (draft_tok) free(draft_tok);
            }

            (void)mtp_top5_match;
        }

        if (recent_count < rep_window) recent[recent_count++] = next_tok;
        else { memmove(recent, recent + 1, sizeof(int) * (rep_window - 1)); recent[rep_window - 1] = next_tok; }
        size_t flen;
        char* frag = tokenize_decode_one(next_tok, &flen);
        if (frag) { fwrite(frag, 1, flen, stderr); fflush(stderr); free(frag); }

        if (g_mtp_enabled) {
            mtp_reset_kv();
            TopEntry mtp_top[8];
            int rc;
            if (use_bf16_mtp) {
                rc = forward_mtp_bf16(next_tok, g_xn, position,
                                      embed_w, embed_scale_cached, 5, mtp_top);
            } else {
                rc = forward_mtp(next_tok, g_xn, position,
                                 embed_w, embed_scale_cached, 5, mtp_top);
            }
            if (rc == 0) {
                int best = 0;
                for (int i = 1; i < 5; i++) if (mtp_top[i].logit > mtp_top[best].logit) best = i;
                mtp_draft = mtp_top[best].id;
            } else {
                mtp_draft = -1;
            }
        }
        (void)prev_for_mtp;
    }
    double t_gen = st_now_seconds() - t_gen0;
    fprintf(stderr, "\n\n");
    if (g_mtp_enabled && mtp_total > 0) {
        fprintf(stderr, "  MTP top-1 accuracy: %d/%d = %.1f%%\n",
                mtp_top1_match, mtp_total,
                100.0 * mtp_top1_match / mtp_total);
    }
    fprintf(stderr, "  %d gen tokens in %.2fs (%.2f tok/s)\n",
            n_gen - 1, t_gen, n_gen > 1 ? (n_gen - 1) / t_gen : 0);
    fprintf(stderr, "  ids: ");
    for (int i = 0; i < n_gen && i < 32; i++) fprintf(stderr, "%d ", gen_ids[i]);
    fprintf(stderr, "%s\n", n_gen > 32 ? "..." : "");
    fprintf(stderr, "  peak RSS: %ld KB (%.2f MB)\n",
            st_peak_rss_kb(), st_peak_rss_kb()/1024.0);

    free(g_blk_fp32);
    if (g_sw_inited) {
        for (int wi = 0; wi < g_n_workers; wi++) sgemv_worker_shutdown(&g_sw[wi]);
    }
    prefetch_shutdown(&g_pf);
    tok_server_stop();
    if (g_mmap_base) munmap((void*)g_mmap_base, g_mmap_size);
    if (g_state_mmap) munmap((void*)g_state_mmap, g_state_mmap_size);
    if (g_mtp_mmap_base) munmap((void*)g_mtp_mmap_base, g_mtp_mmap_size);
    st_index_free(&g_ix); close(g_fd); close(g_state_fd);
    if (g_mtp_fd >= 0) { st_index_free(&g_mtp_ix); close(g_mtp_fd); }
    if (g_bf16_fd >= 0) { st_index_free(&g_bf16_ix); close(g_bf16_fd); }
    unlink(state_path);
    return 0;
}
