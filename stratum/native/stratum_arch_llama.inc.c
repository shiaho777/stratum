
#define _GNU_SOURCE

#include "stratum_arch.h"
#include "stratum_linear.h"
#include "stratum_engine.h"
#include "stratum_tokenize.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>

static Gguf         la_g_gguf;
static StratumConfig  la_g_cfg;

typedef struct {
    const GgufTensor* attn_norm;
    const GgufTensor* attn_q;
    const GgufTensor* attn_k;
    const GgufTensor* attn_v;
    const GgufTensor* attn_output;
    const GgufTensor* attn_q_norm;
    const GgufTensor* attn_k_norm;
    const GgufTensor* ffn_norm;
    const GgufTensor* ffn_gate;
    const GgufTensor* ffn_up;
    const GgufTensor* ffn_down;
} la_BlockTensors;

static la_BlockTensors* la_g_blocks = NULL;
static const GgufTensor* la_g_token_embd = NULL;
static const GgufTensor* la_g_output_norm = NULL;
static const GgufTensor* la_g_output_w = NULL;

static void la_rmsnorm(const float* x, const float* gain, int N, float eps, float* y) {
    /* f64x2 pairwise sum (same double domain, vector-pair accumulation
     * order), then the f32x4 scale/gain pass */
#if defined(__ARM_NEON) || defined(__aarch64__)
    float64x2_t vss = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 8 <= N; i += 8) {
        float32x4_t a = vld1q_f32(x + i), b = vld1q_f32(x + i + 4);
        vss = vfmaq_f64(vss, vcvt_f64_f32(vget_low_f32(a)),  vcvt_f64_f32(vget_low_f32(a)));
        vss = vfmaq_f64(vss, vcvt_f64_f32(vget_high_f32(a)), vcvt_f64_f32(vget_high_f32(a)));
        vss = vfmaq_f64(vss, vcvt_f64_f32(vget_low_f32(b)),  vcvt_f64_f32(vget_low_f32(b)));
        vss = vfmaq_f64(vss, vcvt_f64_f32(vget_high_f32(b)), vcvt_f64_f32(vget_high_f32(b)));
    }
    double ss = vaddvq_f64(vss);
    for (; i < N; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)N + (double)eps));
    i = 0;
    for (; i + 8 <= N; i += 8) {
        float32x4_t a = vld1q_f32(x + i), b = vld1q_f32(x + i + 4);
        float32x4_t ga = vld1q_f32(gain + i), gb = vld1q_f32(gain + i + 4);
        float32x4_t vs = vdupq_n_f32(scale);
        vst1q_f32(y + i,     vmulq_f32(vmulq_f32(a, vs), ga));
        vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(b, vs), gb));
    }
    for (; i < N; i++) y[i] = x[i] * scale * gain[i];
#else
    double ss = 0.0;
    for (int i = 0; i < N; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)N + (double)eps));
    for (int i = 0; i < N; i++) y[i] = x[i] * scale * gain[i];
#endif
}

static void la_swiglu(const float* g, const float* u, int N, float* y) {
    stratum_swiglu(g, u, N, y);
}

/* RoPE pair layout: ggml has two conventions and the GGUF keeps the
 * model's own — Llama-family weights are stored interleaved and rotated
 * as adjacent pairs (NORM); Qwen-family weights keep the original HF
 * layout and rotate (x[k], x[k+n/2]) pairs (NEOX). Selected from the
 * file's general.architecture — no model names, just arch ids. */
static int la_g_rope_neox = 0;

/* P2: cos/sin tables — powf/cosf/sinf were recomputed for every pair of
 * every head of every layer per token (756 rope calls x 64 powf on a
 * 42-layer model). The table caches cosf/sinf of the SAME angles, so the
 * rotated values are bit-identical to the scalar path. Positions grow
 * lazily up to the KV cap; memory is 2 * pos * n_pairs floats (<= 512 KB
 * at the 1024-token cap) — activation-scale, not weight-scale. */
static float*  la_g_rope_cos   = NULL;
static float*  la_g_rope_sin   = NULL;
static int     la_g_rope_pos_cap = 0;
static int     la_g_rope_npair = 0;
static float   la_g_rope_theta = -1.0f;

static void la_rope_tables_grow(int position, int n_pairs, float theta) {
    if (la_g_rope_theta != theta || la_g_rope_npair != n_pairs) {
        free(la_g_rope_cos); free(la_g_rope_sin);
        la_g_rope_cos = la_g_rope_sin = NULL;
        la_g_rope_pos_cap = 0;
        la_g_rope_theta = theta;
        la_g_rope_npair = n_pairs;
    }
    if (position < la_g_rope_pos_cap) return;
    int newcap = la_g_rope_pos_cap ? la_g_rope_pos_cap * 2 : 256;
    while (newcap <= position) newcap *= 2;
    float* nc = malloc(sizeof(float) * (size_t)newcap * n_pairs);
    float* ns = malloc(sizeof(float) * (size_t)newcap * n_pairs);
    for (int p = 0; p < newcap; p++) {
        for (int k = 0; k < n_pairs; k++) {
            float freq  = 1.0f / powf(theta, (float)(2 * k) / (float)(n_pairs * 2));
            float angle = (float)p * freq;
            nc[(size_t)p * n_pairs + k] = cosf(angle);
            ns[(size_t)p * n_pairs + k] = sinf(angle);
        }
    }
    if (la_g_rope_cos) {  /* copy overlapping prefix (bit-identical) */
        memcpy(nc, la_g_rope_cos, sizeof(float) * (size_t)la_g_rope_pos_cap * n_pairs);
        memcpy(ns, la_g_rope_sin, sizeof(float) * (size_t)la_g_rope_pos_cap * n_pairs);
        free(la_g_rope_cos); free(la_g_rope_sin);
    }
    la_g_rope_cos = nc; la_g_rope_sin = ns; la_g_rope_pos_cap = newcap;
}

static void la_rope(float* x, int head_dim, int rope_dim, int position, float theta) {
    int n_pairs = rope_dim / 2;
    la_rope_tables_grow(position, n_pairs, theta);
    const float* cs = la_g_rope_cos + (size_t)position * n_pairs;
    const float* sn = la_g_rope_sin + (size_t)position * n_pairs;
    if (la_g_rope_neox) {
        for (int k = 0; k < n_pairs; k++) {
            float c = cs[k], s = sn[k];
            float x0 = x[k], x1 = x[k + n_pairs];
            x[k]           = x0 * c - x1 * s;
            x[k + n_pairs] = x0 * s + x1 * c;
        }
    } else {
        for (int k = 0; k < n_pairs; k++) {
            float c = cs[k], s = sn[k];
            float x0 = x[2 * k], x1 = x[2 * k + 1];
            x[2 * k]     = x0 * c - x1 * s;
            x[2 * k + 1] = x0 * s + x1 * c;
        }
    }
    (void)head_dim;
}

static void la_softmax_inplace(float* x, int N) {
    stratum_softmax_inplace(x, N);
}

static float* la_g_x       = NULL;
static float* la_g_x_resid = NULL;
static float* la_g_xn      = NULL;
static float* la_g_q_buf   = NULL;
static float* la_g_k_buf   = NULL;
static float* la_g_v_buf   = NULL;
static float* la_g_attn_out= NULL;
static float* la_g_ff_g    = NULL;
static float* la_g_ff_u    = NULL;
static float* la_g_ff_a    = NULL;

static float* la_g_K_cache = NULL;
static float* la_g_V_cache = NULL;
#define la_MAX_KV 1024
static int    la_g_kv_len  = 0;

/* Multi-sequence KV: B independent streams, layout [L][B][MAX_KV][Nk*Hd].
 * Allocated lazily by the multi-seq generator. This is the unbounded
 * THROUGHPUT axis: B streams, one weight sweep serves all B, each
 * commits exactly one token per sweep (no accept-rate gamble). */
#define la_MS_MAX 16
static float* la_g_msK = NULL;
static float* la_g_msV = NULL;
static int    la_g_ms_B = 0;
static int    la_g_ms_maxkv = 0;

static float* la_g_logits  = NULL;

#define la_B_MAX 64
static float* la_gb_xn[la_B_MAX];
static float* la_gb_logits[la_B_MAX];

static int la_g_blas_batch = 0;

/* Dequantize one weight row (Q4_K/Q6_K) into fp32 dst. */
static inline int la_dequant_row(const GgufTensor* w, int r, int K, float* dst) {
    if (w->type == GGML_TYPE_Q4_K) {
        q4k_dequant_row_neon(st_q4k_row_ptr(w, K, r), K, dst); return 1;
    }
    if (w->type == GGML_TYPE_Q6_K) {
        const block_q6_K* row = st_q6k_row_ptr(w, K, r);
        for (int i = 0; i < K/256; i++) q6k_dequant_block_scalar(row + i, dst + i*256);
        return 1;
    }
    return 0;
}

/* BLAS-batched matmul: dequant W rows to fp32 (in tiles), then one
 * cblas_sgemm per tile computes all B columns. ~5x over handwritten
 * multix at B=8 (compute-bound GEMM vs memory-bound per-row dequant).
 * Y[s][r] = sum_k W[r][k] * xs[s][k]. */
static void la_linear_multix_blas(const GgufTensor* w, const float* const* xs,
                                  float* const* ys, int B, int N, int K) {
    const int TILE = 512;                 /* rows of W dequant'd at once */
    /* build Xt: K x B row-major (Xt[k*B + s] = xs[s][k]) */
    float* Xt = (float*)malloc((size_t)K * B * sizeof(float));
    for (int s = 0; s < B; s++)
        for (int k = 0; k < K; k++) Xt[(size_t)k*B + s] = xs[s][k];
    int ntile = (N + TILE - 1) / TILE;
    /* parallelize dequant+gemm across tiles */
    st_par_run(ntile,
        ^(int ti) {
            int r0 = (int)ti * TILE;
            int rows = (r0 + TILE <= N) ? TILE : (N - r0);
            float* Wf = (float*)malloc((size_t)rows * K * sizeof(float));
            float* Yt = (float*)malloc((size_t)rows * B * sizeof(float));
            for (int r = 0; r < rows; r++)
                la_dequant_row(w, r0 + r, K, Wf + (size_t)r * K);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        rows, B, K, 1.0f, Wf, K, Xt, B, 0.0f, Yt, B);
            for (int r = 0; r < rows; r++)
                for (int s = 0; s < B; s++) ys[s][r0 + r] = Yt[(size_t)r*B + s];
            free(Wf); free(Yt);
        });
    free(Xt);
}

static int la_g_gpu_batch = 0;

static void la_linear_multix(const GgufTensor* w, const float* const* xs,
                             float* const* ys, int B, int N, int K) {
#ifdef STRATUM_USE_METAL
    if (la_g_gpu_batch && w->type == GGML_TYPE_Q4_K && B >= 2 && B <= 16) {
        static float* xp = NULL; static float* yp = NULL;
        static size_t xpc = 0, ypc = 0;
        size_t xn = (size_t)B*K, yn = (size_t)B*N;
        if (xn > xpc) { free(xp); xp = malloc(xn*sizeof(float)); xpc = xn; }
        if (yn > ypc) { free(yp); yp = malloc(yn*sizeof(float)); ypc = yn; }
        for (int s = 0; s < B; s++) memcpy(xp + (size_t)s*K, xs[s], (size_t)K*sizeof(float));
        if (stratum_metal_q4k_sgemv_batched(w->offset, xp, yp, N, K, B) == 0) {
            for (int s = 0; s < B; s++)
                memcpy(ys[s], yp + (size_t)s*N, (size_t)N*sizeof(float));
            return;
        }
    }
#endif
    if (la_g_blas_batch &&
        (w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K)) {
        la_linear_multix_blas(w, xs, ys, B, N, K);
        return;
    }
    /* everything else (Q4_K/Q6_K pack, W16/AMX, per-format fallback) lives
     * in the shared dispatcher — identical semantics, one implementation */
    st_linear_multix(w, xs, ys, B, N, K);
}

/* Hidden-state export (STRATUM_HIDDEN_DUMP=<layer>:<path>): captures the
 * residual stream after layer <layer> for every processed token, f32 records
 * behind a "SHID0001" | u32 layer | u32 H header. Probe tooling for the
 * video-encoder phase (epic #35); single-stream paths only — MULTISEQ/spec
 * interleave streams and are not represented. */
static FILE* la_hidden_dump_fp = NULL;
static int   la_hidden_dump_layer = -1;

static void la_forward_block(int li, int position) {
    la_BlockTensors* b = &la_g_blocks[li];
    /* optional advisory prefetch of the NEXT layer's weights — measures
     * whether B=1 is page-in bound. Reclaimable hint only; off by default. */
    static int pf_next = -1;
    if (pf_next < 0) pf_next = getenv("STRATUM_PF_NEXT") ? 1 : 0;
    if (pf_next && li + 1 < la_g_cfg.n_layers) {
        const la_BlockTensors* nb = &la_g_blocks[li + 1];
        const GgufTensor* ts[] = { nb->attn_q, nb->attn_k, nb->attn_v,
                                   nb->attn_output, nb->ffn_gate,
                                   nb->ffn_up, nb->ffn_down };
        for (int i = 0; i < 7; i++)
            if (ts[i])
                madvise((void*)((const char*)g_st.mmap_base + ts[i]->offset),
                        (size_t)ts[i]->nbytes, MADV_WILLNEED);
    }
    int H  = la_g_cfg.n_embed;
    int Hd = la_g_cfg.head_dim;
    int Nq = la_g_cfg.n_q_heads;
    int Nk = la_g_cfg.n_kv_heads;
    int Ff = la_g_cfg.n_ff;

    int dbg = getenv("STRATUM_BLOCK_DBG") && position <= 2;
    memcpy(la_g_x_resid, la_g_x, sizeof(float) * H);
    {
        const float* gain = st_f32_tensor_ptr(b->attn_norm);
        la_rmsnorm(la_g_x, gain, H, la_g_cfg.rms_eps, la_g_xn);
    }

    st_q4k_group(la_g_xn, H,
                 b->attn_q, la_g_q_buf, Nq * Hd,
                 b->attn_k, la_g_k_buf, Nk * Hd,
                 b->attn_v, la_g_v_buf, Nk * Hd);

    /* Qwen3-style per-head q/k RMSNorm — optional tensors; applied
     * pre-rope with the same eps as the block norms. NULL for plain
     * Llama-family files, which skip this entirely. */
    if (b->attn_q_norm) {
        const float* gain = st_f32_tensor_ptr(b->attn_q_norm);
        for (int h = 0; h < Nq; h++)
            la_rmsnorm(la_g_q_buf + h * Hd, gain, Hd, la_g_cfg.rms_eps,
                       la_g_q_buf + h * Hd);
    }
    if (b->attn_k_norm) {
        const float* gain = st_f32_tensor_ptr(b->attn_k_norm);
        for (int h = 0; h < Nk; h++)
            la_rmsnorm(la_g_k_buf + h * Hd, gain, Hd, la_g_cfg.rms_eps,
                       la_g_k_buf + h * Hd);
    }

    for (int h = 0; h < Nq; h++) {
        la_rope(la_g_q_buf + h * Hd, Hd, la_g_cfg.rope_dim, position, la_g_cfg.rope_theta);
    }
    for (int h = 0; h < Nk; h++) {
        la_rope(la_g_k_buf + h * Hd, Hd, la_g_cfg.rope_dim, position, la_g_cfg.rope_theta);
    }
    if (dbg && li == 0)
        fprintf(stderr, "  [cL0] q0=%.5f q1=%.5f k0=%.5f k1=%.5f\n",
                la_g_q_buf[0], la_g_q_buf[1], la_g_k_buf[0], la_g_k_buf[1]);

    int kv_len_now = la_g_kv_len + 1;
    {
        size_t per_layer = (size_t)la_MAX_KV * Nk * Hd;
        size_t off = (size_t)li * per_layer + (size_t)la_g_kv_len * Nk * Hd;
        memcpy(la_g_K_cache + off, la_g_k_buf, sizeof(float) * Nk * Hd);
        memcpy(la_g_V_cache + off, la_g_v_buf, sizeof(float) * Nk * Hd);
    }

    float scale = 1.0f / sqrtf((float)Hd);
    /* heads are independent: parallelize when the serial loop is long
     * enough to matter (kv_len grows with context). Same math, same
     * output — only the head iteration order across threads changes. */
    int attn_par = (kv_len_now >= 32 && Nq >= 4);
    void (^attn_head)(int) = ^(int h) {
        int kv_h = h * Nk / Nq;
        const float* qh = la_g_q_buf + h * Hd;
        size_t per_layer = (size_t)la_MAX_KV * Nk * Hd;
        const float* K_layer = la_g_K_cache + (size_t)li * per_layer;
        const float* V_layer = la_g_V_cache + (size_t)li * per_layer;

        float logits[la_MAX_KV];
        for (int t = 0; t < kv_len_now; t++) {
            const float* kt = K_layer + (size_t)t * Nk * Hd + kv_h * Hd;
            float dot = 0.0f;
#if defined(__ARM_NEON) || defined(__aarch64__)
            /* P2b: QK dot was the last scalar hot loop — 16 q-heads x
             * kv_len x head_dim MACs per layer. f32x4 FMA accumulation
             * reorders the reduction; greedy pins are gate-verified. */
            float32x4_t acc = vdupq_n_f32(0.0f);
            int d = 0;
            for (; d + 16 <= Hd; d += 16) {
                acc = vfmaq_f32(acc, vld1q_f32(qh + d),      vld1q_f32(kt + d));
                acc = vfmaq_f32(acc, vld1q_f32(qh + d + 4),  vld1q_f32(kt + d + 4));
                acc = vfmaq_f32(acc, vld1q_f32(qh + d + 8),  vld1q_f32(kt + d + 8));
                acc = vfmaq_f32(acc, vld1q_f32(qh + d + 12), vld1q_f32(kt + d + 12));
            }
            for (; d + 4 <= Hd; d += 4)
                acc = vfmaq_f32(acc, vld1q_f32(qh + d), vld1q_f32(kt + d));
            dot = vaddvq_f32(acc);
            for (; d < Hd; d++) dot += qh[d] * kt[d];
#else
            for (int d = 0; d < Hd; d++) dot += qh[d] * kt[d];
#endif
            logits[t] = dot * scale;
        }
        la_softmax_inplace(logits, kv_len_now);

        float* head_out = la_g_attn_out + h * Hd;
        memset(head_out, 0, sizeof(float) * Hd);
        for (int t = 0; t < kv_len_now; t++) {
            const float* vt = V_layer + (size_t)t * Nk * Hd + kv_h * Hd;
            float p = logits[t];
#if defined(__ARM_NEON) || defined(__aarch64__)
            /* element-wise accumulate, same t order — bit-exact vs scalar */
            float32x4_t vp = vdupq_n_f32(p);
            int d = 0;
            for (; d + 16 <= Hd; d += 16) {
                vst1q_f32(head_out + d, vfmaq_f32(vld1q_f32(head_out + d),      vp, vld1q_f32(vt + d)));
                vst1q_f32(head_out + d + 4, vfmaq_f32(vld1q_f32(head_out + d + 4),  vp, vld1q_f32(vt + d + 4)));
                vst1q_f32(head_out + d + 8, vfmaq_f32(vld1q_f32(head_out + d + 8),  vp, vld1q_f32(vt + d + 8)));
                vst1q_f32(head_out + d + 12, vfmaq_f32(vld1q_f32(head_out + d + 12), vp, vld1q_f32(vt + d + 12)));
            }
            for (; d < Hd; d++) head_out[d] += p * vt[d];
#else
            for (int d = 0; d < Hd; d++) head_out[d] += p * vt[d];
#endif
        }
    };
    if (attn_par)
        st_par_run(Nq, attn_head);
    else
        for (int h = 0; h < Nq; h++) attn_head(h);

    static float attn_proj[8192];
    if (H > 8192) { fprintf(stderr, "H exceeds buffer\n"); exit(2); }
    if (dbg && li == 0)
        fprintf(stderr, "  [cL0] ao0=%.5f ao1=%.5f\n", la_g_attn_out[0], la_g_attn_out[1]);
    st_linear_dispatch(b->attn_output, la_g_attn_out, attn_proj, H, Nq * Hd);
    for (int i = 0; i < H; i++) la_g_x[i] = la_g_x_resid[i] + attn_proj[i];
    if (dbg && li == 0)
        fprintf(stderr, "  [cL0] x0=%.5f x1=%.5f (post-attn)\n", la_g_x[0], la_g_x[1]);

    memcpy(la_g_x_resid, la_g_x, sizeof(float) * H);
    {
        const float* gain = st_f32_tensor_ptr(b->ffn_norm);
        la_rmsnorm(la_g_x, gain, H, la_g_cfg.rms_eps, la_g_xn);
    }
    static float ff_out[8192];
#ifdef STRATUM_USE_METAL
    int ffn_gpu = 0;
    if (g_st.use_metal
        && b->ffn_gate->type == GGML_TYPE_Q4_K && b->ffn_gate->offset
        && b->ffn_up->type   == GGML_TYPE_Q4_K && b->ffn_up->offset
        && b->ffn_down->type == GGML_TYPE_Q6_K && b->ffn_down->offset) {
        if (stratum_metal_ffn(b->ffn_gate->offset, b->ffn_gate->nbytes,
                              b->ffn_up->offset,   b->ffn_up->nbytes,
                              b->ffn_down->offset, b->ffn_down->nbytes,
                              la_g_xn, ff_out, H, Ff) == 0) ffn_gpu = 1;
    }
    if (!ffn_gpu)
#endif
    {
        st_q4k_group(la_g_xn, H,
                     b->ffn_gate, la_g_ff_g, Ff,
                     b->ffn_up,   la_g_ff_u, Ff,
                     NULL, NULL, 0);
        la_swiglu(la_g_ff_g, la_g_ff_u, Ff, la_g_ff_a);
        st_linear_dispatch(b->ffn_down, la_g_ff_a, ff_out, H, Ff);
    }
    for (int i = 0; i < H; i++) la_g_x[i] = la_g_x_resid[i] + ff_out[i];

    if (li == la_hidden_dump_layer && la_hidden_dump_fp) {
        fwrite(la_g_x, sizeof(float), H, la_hidden_dump_fp);
        fflush(la_hidden_dump_fp);
    }
}

static void la_embed_lookup(int token_id, float* out) {
    int H = la_g_cfg.n_embed;

    if (la_g_token_embd->type == GGML_TYPE_Q4_K) {
        const block_q4_K* row = st_q4k_row_ptr(la_g_token_embd, H, token_id);
        q4k_dequant_row_scalar(row, H, out);
    } else if (la_g_token_embd->type == GGML_TYPE_Q6_K) {
        const block_q6_K* row = st_q6k_row_ptr(la_g_token_embd, H, token_id);
        int n_blocks = H / 256;
        for (int i = 0; i < n_blocks; i++) {
            q6k_dequant_block_scalar(row + i, out + i * 256);
        }
    } else if (la_g_token_embd->type == GGML_TYPE_Q8_0) {
        const block_q8_0* row = st_q8_0_row_ptr(la_g_token_embd, H, token_id);
        for (int i = 0; i < H / 32; i++)
            q8_0_dequant_block_scalar(row + i, out + i * 32);
    } else if (la_g_token_embd->type == GGML_TYPE_F16) {
        const uint16_t* raw = (const uint16_t*)(g_st.mmap_base + la_g_token_embd->offset)
                            + (size_t)token_id * H;
        for (int i = 0; i < H; i++) out[i] = q4k_fp16_to_fp32(raw[i]);
    } else if (la_g_token_embd->type == GGML_TYPE_F32) {
        const float* raw = (const float*)(g_st.mmap_base + la_g_token_embd->offset)
                         + (size_t)token_id * H;
        memcpy(out, raw, sizeof(float) * H);
    } else {
        fprintf(stderr, "unsupported embed type %s\n",
                gguf_type_name((GgmlType)la_g_token_embd->type));
        exit(2);
    }
}

static int la_discover_blocks(void) {
    la_g_blocks = (la_BlockTensors*)calloc(la_g_cfg.n_layers, sizeof(la_BlockTensors));
    if (!la_g_blocks) return -1;
    char nm[128];

#define FIND(field, name)                                              \
    snprintf(nm, sizeof nm, "blk.%d." name ".weight", li);             \
    la_g_blocks[li].field = gguf_find_tensor(&la_g_gguf, nm);                \
    if (!la_g_blocks[li].field) {                                         \
        fprintf(stderr, "missing %s\n", nm); return -1;                \
    }
#define FIND_OPTIONAL(field, name)                                     \
    snprintf(nm, sizeof nm, "blk.%d." name ".weight", li);             \
    la_g_blocks[li].field = gguf_find_tensor(&la_g_gguf, nm);

    for (int li = 0; li < la_g_cfg.n_layers; li++) {
        FIND(attn_norm,   "attn_norm")
        FIND(attn_q,      "attn_q")
        FIND(attn_k,      "attn_k")
        FIND(attn_v,      "attn_v")
        FIND(attn_output, "attn_output")
        FIND_OPTIONAL(attn_q_norm, "attn_q_norm")
        FIND_OPTIONAL(attn_k_norm, "attn_k_norm")
        FIND(ffn_norm,    "ffn_norm")
        FIND(ffn_gate,    "ffn_gate")
        FIND(ffn_up,      "ffn_up")
        FIND(ffn_down,    "ffn_down")
    }
    la_g_token_embd  = gguf_find_tensor(&la_g_gguf, "token_embd.weight");
    la_g_output_norm = gguf_find_tensor(&la_g_gguf, "output_norm.weight");
    la_g_output_w    = gguf_find_tensor(&la_g_gguf, "output.weight");

    if (!la_g_token_embd || !la_g_output_norm) {
        fprintf(stderr, "missing token_embd or output_norm\n"); return -1;
    }

#undef FIND
#undef FIND_OPTIONAL
    return 0;
}

static int la_allocate_state(void) {
    int H  = la_g_cfg.n_embed;
    int Hd = la_g_cfg.head_dim;
    int Nq = la_g_cfg.n_q_heads;
    int Nk = la_g_cfg.n_kv_heads;
    int Ff = la_g_cfg.n_ff;
    int V  = la_g_cfg.vocab_size;
    int L  = la_g_cfg.n_layers;

    la_g_x        = (float*)calloc(H, sizeof(float));
    la_g_x_resid  = (float*)calloc(H, sizeof(float));
    la_g_xn       = (float*)calloc(H, sizeof(float));
    la_g_q_buf    = (float*)calloc(Nq * Hd, sizeof(float));
    la_g_k_buf    = (float*)calloc(Nk * Hd, sizeof(float));
    la_g_v_buf    = (float*)calloc(Nk * Hd, sizeof(float));
    la_g_attn_out = (float*)calloc(Nq * Hd, sizeof(float));
    la_g_ff_g     = (float*)calloc(Ff, sizeof(float));
    la_g_ff_u     = (float*)calloc(Ff, sizeof(float));
    la_g_ff_a     = (float*)calloc(Ff, sizeof(float));
    la_g_logits   = (float*)calloc(V, sizeof(float));

    size_t kv_floats = (size_t)L * la_MAX_KV * Nk * Hd;
    la_g_K_cache = (float*)calloc(kv_floats, sizeof(float));
    la_g_V_cache = (float*)calloc(kv_floats, sizeof(float));

    if (!la_g_x || !la_g_K_cache || !la_g_logits) return -1;

    fprintf(stderr, "  KV cache: %.1f MB (anonymous)\n",
            (double)kv_floats * 2 * 4 / (1024.0 * 1024.0));
    fprintf(stderr, "  activations: %.1f KB anon\n",
            (double)(H * 3 + Nq * Hd * 2 + Nk * Hd * 2 + Ff * 3 + V) * 4 / 1024.0);
    return 0;
}

#ifdef STRATUM_USE_METAL
static int la_g_gpu_full;
static int la_forward_one_token_gpu(int token_id, int position);
#endif
static int la_forward_one_token(int token_id, int position) {
#ifdef STRATUM_USE_METAL
    if (la_g_gpu_full) return la_forward_one_token_gpu(token_id, position);
#endif
    int H = la_g_cfg.n_embed;
    int V = la_g_cfg.vocab_size;

    la_embed_lookup(token_id, la_g_x);

    int _nl = la_g_cfg.n_layers;
    { const char* e = getenv("STRATUM_FWD_NL"); if (e) { int v=atoi(e); if (v>=0 && v<_nl) _nl=v; } }
    for (int li = 0; li < _nl; li++) {
        la_forward_block(li, position);
        if (getenv("STRATUM_FWD_NL") && position == 0)
            fprintf(stderr, "  [cx] L%d x[0..3]=%.6f %.6f %.6f %.6f\n",
                    li, la_g_x[0], la_g_x[1], la_g_x[2], la_g_x[3]);
    }

    {
        const float* gain = st_f32_tensor_ptr(la_g_output_norm);
        la_rmsnorm(la_g_x, gain, H, la_g_cfg.rms_eps, la_g_xn);
    }

    if (getenv("STRATUM_MS_DUMPXN")) {
        FILE* df = fopen(getenv("STRATUM_MS_DUMPXN"), "ab");
        if (df) { fwrite(la_g_x, 4, H, df); fwrite(la_g_xn, 4, H, df); fclose(df); }
    }
    const GgufTensor* lm = la_g_output_w ? la_g_output_w : la_g_token_embd;
    if (st_linear_dispatch(lm, la_g_xn, la_g_logits, V, H) != 0) return -1;

    la_g_kv_len++;
    return 0;
}

#ifdef STRATUM_USE_METAL
/* Whole-token forward entirely on GPU (one command buffer). Requires
 * STRATUM_GPU (model wrapped as zero-copy chunks). KV cache lives on GPU. */
static int la_g_gpu_full = 0;
static StratumMetalLayer la_g_lys[128];
static int la_g_lys_built = 0;
static int la_forward_one_token_gpu(int token_id, int position) {
    int H = la_g_cfg.n_embed, V = la_g_cfg.vocab_size;
/* lys moved to global la_g_lys */
/* built moved to global la_g_lys_built */
    int nL = la_g_cfg.n_layers;
    if (!la_g_lys_built) {
        for (int i = 0; i < nL; i++) {
            la_BlockTensors* b = &la_g_blocks[i];
            la_g_lys[i].attn_norm_off = b->attn_norm->offset; la_g_lys[i].ffn_norm_off = b->ffn_norm->offset;
            la_g_lys[i].qnorm_off = b->attn_q_norm ? b->attn_q_norm->offset : 0;
            la_g_lys[i].knorm_off = b->attn_k_norm ? b->attn_k_norm->offset : 0;
            la_g_lys[i].q_off=b->attn_q->offset; la_g_lys[i].q_tb=b->attn_q->nbytes; la_g_lys[i].q_ty=b->attn_q->type;
            la_g_lys[i].k_off=b->attn_k->offset; la_g_lys[i].k_tb=b->attn_k->nbytes; la_g_lys[i].k_ty=b->attn_k->type;
            la_g_lys[i].v_off=b->attn_v->offset; la_g_lys[i].v_tb=b->attn_v->nbytes; la_g_lys[i].v_ty=b->attn_v->type;
            la_g_lys[i].o_off=b->attn_output->offset; la_g_lys[i].o_tb=b->attn_output->nbytes; la_g_lys[i].o_ty=b->attn_output->type;
            la_g_lys[i].gate_off=b->ffn_gate->offset; la_g_lys[i].gate_tb=b->ffn_gate->nbytes; la_g_lys[i].gate_ty=b->ffn_gate->type;
            la_g_lys[i].up_off=b->ffn_up->offset; la_g_lys[i].up_tb=b->ffn_up->nbytes; la_g_lys[i].up_ty=b->ffn_up->type;
            la_g_lys[i].down_off=b->ffn_down->offset; la_g_lys[i].down_tb=b->ffn_down->nbytes; la_g_lys[i].down_ty=b->ffn_down->type;
        }
        la_g_lys_built = 1;
    }
    la_embed_lookup(token_id, la_g_x);
    const GgufTensor* lm = la_g_output_w ? la_g_output_w : la_g_token_embd;
    /* V-opt: fused argmax on GPU when STRATUM_GPU_FUSED_ARGMAX is set.
     * Skips 128KB logits transfer per token. */
    float* logits_ptr = getenv("STRATUM_GPU_FUSED_ARGMAX") ? NULL : la_g_logits;
    int rc = stratum_metal_forward(la_g_lys, la_g_cfg.n_layers, la_g_output_norm->offset,
                                   lm->offset, lm->nbytes, lm->type == GGML_TYPE_Q6_K,
                                   la_g_x, logits_ptr,
                                   H, la_g_cfg.head_dim, la_g_cfg.n_q_heads, la_g_cfg.n_kv_heads,
                                   la_g_cfg.n_ff, V, la_g_cfg.rope_dim, position,
                                   la_g_cfg.rope_theta, la_g_cfg.rms_eps, la_g_kv_len, la_MAX_KV,
                                   la_g_rope_neox, 0, 0, -1);
    if (rc != 0) return -1;
    /* V-opt: if fused argmax, get token from GPU.
     * Write token to la_g_logits[0] as a sentinel — caller checks
     * STRATUM_GPU_FUSED_ARGMAX and uses stratum_metal_get_last_token(). */
    la_g_kv_len++;
    return 0;
}
#endif

static int la_forward_batch_ex(const int* tokens, const int* positions,
                               const int* parents, int B) {
    int H  = la_g_cfg.n_embed;
    int Hd = la_g_cfg.head_dim;
    int Nq = la_g_cfg.n_q_heads;
    int Nk = la_g_cfg.n_kv_heads;
    int Ff = la_g_cfg.n_ff;
    int V  = la_g_cfg.vocab_size;
    if (B > la_B_MAX) return -1;

    static float* x[la_B_MAX]; static float* xr[la_B_MAX]; static float* xn[la_B_MAX];
    static float* qb[la_B_MAX]; static float* kb[la_B_MAX]; static float* vb[la_B_MAX];
    static float* ao[la_B_MAX]; static float* ap[la_B_MAX];
    static float* fg[la_B_MAX]; static float* fu[la_B_MAX]; static float* fa[la_B_MAX];
    static int alloc_done = 0;
    if (!alloc_done) {
        for (int s = 0; s < la_B_MAX; s++) {
            x[s]=calloc(H,4); xr[s]=calloc(H,4); xn[s]=calloc(H,4);
            qb[s]=calloc(Nq*Hd,4); kb[s]=calloc(Nk*Hd,4); vb[s]=calloc(Nk*Hd,4);
            ao[s]=calloc(Nq*Hd,4); ap[s]=calloc(H,4);
            fg[s]=calloc(Ff,4); fu[s]=calloc(Ff,4); fa[s]=calloc(Ff,4);
            la_gb_logits[s]=calloc(V,4);
        }
        alloc_done = 1;
    }

    for (int s = 0; s < B; s++) la_embed_lookup(tokens[s], x[s]);

    const float* cxn[la_B_MAX]; float* cqb[la_B_MAX]; float* ckb[la_B_MAX];
    float* cvb[la_B_MAX]; float* cao[la_B_MAX]; float* cap[la_B_MAX];
    float* cfg[la_B_MAX]; float* cfu[la_B_MAX]; float* cfa[la_B_MAX];
    const float* cfa_in[la_B_MAX];
    for (int s = 0; s < B; s++) {
        cxn[s]=xn[s]; cqb[s]=qb[s]; ckb[s]=kb[s]; cvb[s]=vb[s];
        cao[s]=ao[s]; cap[s]=ap[s]; cfg[s]=fg[s]; cfu[s]=fu[s]; cfa[s]=fa[s];
    }

    float scale = 1.0f / sqrtf((float)Hd);
    size_t per_layer = (size_t)la_MAX_KV * Nk * Hd;

    for (int li = 0; li < la_g_cfg.n_layers; li++) {
        la_BlockTensors* b = &la_g_blocks[li];
        for (int s = 0; s < B; s++) {
            memcpy(xr[s], x[s], sizeof(float)*H);
            la_rmsnorm(x[s], st_f32_tensor_ptr(b->attn_norm), H, la_g_cfg.rms_eps, xn[s]);
        }
#ifdef STRATUM_USE_METAL
        if (la_g_gpu_batch && b->attn_q->type == GGML_TYPE_Q4_K
            && b->attn_k->type == GGML_TYPE_Q4_K && b->attn_v->type == GGML_TYPE_Q4_K) {
            /* fuse q,k,v into ONE command buffer / ONE sync */
            static float* xpk=NULL; static size_t xpkc=0;
            size_t xn_=(size_t)B*H; if(xn_>xpkc){free(xpk);xpk=malloc(xn_*4);xpkc=xn_;}
            for(int s=0;s<B;s++) memcpy(xpk+(size_t)s*H, xn[s], (size_t)H*4);
            uint64_t wo[3]={b->attn_q->offset,b->attn_k->offset,b->attn_v->offset};
            int Na[3]={Nq*Hd,Nk*Hd,Nk*Hd};
            static float* yqkv=NULL; static size_t yqkvc=0;
            size_t ytot=(size_t)B*(Nq*Hd+2*Nk*Hd); if(ytot>yqkvc){free(yqkv);yqkv=malloc(ytot*4);yqkvc=ytot;}
            size_t yoff[3]={0,(size_t)B*Nq*Hd*4,(size_t)B*(Nq*Hd+Nk*Hd)*4};
            if(stratum_metal_q4k_multi(wo,Na,3,xpk,yqkv,yoff,H,B)==0){
                for(int s=0;s<B;s++){
                    memcpy(qb[s], yqkv+(size_t)s*Nq*Hd, (size_t)Nq*Hd*4);
                    memcpy(kb[s], (char*)yqkv+yoff[1]+(size_t)s*Nk*Hd*4, (size_t)Nk*Hd*4);
                    memcpy(vb[s], (char*)yqkv+yoff[2]+(size_t)s*Nk*Hd*4, (size_t)Nk*Hd*4);
                }
            } else {
                la_linear_multix(b->attn_q, cxn, cqb, B, Nq*Hd, H);
                la_linear_multix(b->attn_k, cxn, ckb, B, Nk*Hd, H);
                la_linear_multix(b->attn_v, cxn, cvb, B, Nk*Hd, H);
            }
        } else
#endif
        {
        la_linear_multix(b->attn_q, cxn, cqb, B, Nq*Hd, H);
        la_linear_multix(b->attn_k, cxn, ckb, B, Nk*Hd, H);
        la_linear_multix(b->attn_v, cxn, cvb, B, Nk*Hd, H);
        }

        for (int s = 0; s < B; s++) {
            /* Qwen3-style per-head q/k RMSNorm — optional, pre-rope (same
             * as the single-token path in la_forward_block) */
            if (b->attn_q_norm) {
                const float* gain = st_f32_tensor_ptr(b->attn_q_norm);
                for (int h = 0; h < Nq; h++)
                    la_rmsnorm(qb[s] + h * Hd, gain, Hd, la_g_cfg.rms_eps,
                               qb[s] + h * Hd);
            }
            if (b->attn_k_norm) {
                const float* gain = st_f32_tensor_ptr(b->attn_k_norm);
                for (int h = 0; h < Nk; h++)
                    la_rmsnorm(kb[s] + h * Hd, gain, Hd, la_g_cfg.rms_eps,
                               kb[s] + h * Hd);
            }
            for (int h = 0; h < Nq; h++)
                la_rope(qb[s] + h*Hd, Hd, la_g_cfg.rope_dim, positions[s], la_g_cfg.rope_theta);
            for (int h = 0; h < Nk; h++)
                la_rope(kb[s] + h*Hd, Hd, la_g_cfg.rope_dim, positions[s], la_g_cfg.rope_theta);
            size_t off = (size_t)li*per_layer + (size_t)(la_g_kv_len+s)*Nk*Hd;
            memcpy(la_g_K_cache + off, kb[s], sizeof(float)*Nk*Hd);
            memcpy(la_g_V_cache + off, vb[s], sizeof(float)*Nk*Hd);
        }
        const float* K_layer = la_g_K_cache + (size_t)li*per_layer;
        const float* V_layer = la_g_V_cache + (size_t)li*per_layer;
        for (int s = 0; s < B; s++) {
            if (parents) {
                /* tree-verify: slot s attends the committed prefix plus its
                 * ancestor draft slots in path order plus itself. The visited
                 * sequence of KV entries is identical to sequential greedy
                 * for whichever path gets accepted (bit-exact). */
                int path[la_B_MAX]; int pn = 0;
                for (int a = s; a >= 0; a = parents[a]) { path[pn++] = a; }
                for (int h = 0; h < Nq; h++) {
                    int kv_h = h * Nk / Nq;
                    const float* qh = qb[s] + h*Hd;
                    float lg[la_MAX_KV];
                    int nl = 0;
                    for (int t = 0; t < la_g_kv_len; t++) {
                        const float* kt = K_layer + (size_t)t*Nk*Hd + kv_h*Hd;
                        float dot=0; for (int d=0;d<Hd;d++) dot+=qh[d]*kt[d];
                        lg[nl++]=dot*scale;
                    }
                    for (int a = pn - 1; a >= 0; a--) {
                        const float* kt = K_layer
                            + (size_t)(la_g_kv_len + path[a])*Nk*Hd + kv_h*Hd;
                        float dot=0; for (int d=0;d<Hd;d++) dot+=qh[d]*kt[d];
                        lg[nl++]=dot*scale;
                    }
                    la_softmax_inplace(lg, nl);
                    float* hd = ao[s] + h*Hd;
                    memset(hd,0,sizeof(float)*Hd);
                    nl = 0;
                    for (int t = 0; t < la_g_kv_len; t++) {
                        const float* vt=V_layer+(size_t)t*Nk*Hd+kv_h*Hd;
                        float p=lg[nl++]; for(int d=0;d<Hd;d++) hd[d]+=p*vt[d];
                    }
                    for (int a = pn - 1; a >= 0; a--) {
                        const float* vt = V_layer
                            + (size_t)(la_g_kv_len + path[a])*Nk*Hd + kv_h*Hd;
                        float p=lg[nl++]; for(int d=0;d<Hd;d++) hd[d]+=p*vt[d];
                    }
                }
            } else {
            int klen = la_g_kv_len + s + 1;
            for (int h = 0; h < Nq; h++) {
                int kv_h = h * Nk / Nq;
                const float* qh = qb[s] + h*Hd;
                float lg[la_MAX_KV];
                for (int t = 0; t < klen; t++) {
                    const float* kt = K_layer + (size_t)t*Nk*Hd + kv_h*Hd;
                    float dot=0; for (int d=0;d<Hd;d++) dot+=qh[d]*kt[d];
                    lg[t]=dot*scale;
                }
                la_softmax_inplace(lg, klen);
                float* hd = ao[s] + h*Hd;
                memset(hd,0,sizeof(float)*Hd);
                for (int t=0;t<klen;t++) {
                    const float* vt=V_layer+(size_t)t*Nk*Hd+kv_h*Hd;
                    float p=lg[t]; for(int d=0;d<Hd;d++) hd[d]+=p*vt[d];
                }
            }
            }
        }
        la_linear_multix(b->attn_output, (const float* const*)cao, cap, B, H, Nq*Hd);
        for (int s=0;s<B;s++) for(int i=0;i<H;i++) x[s][i]=xr[s][i]+ap[s][i];

        for (int s=0;s<B;s++) {
            memcpy(xr[s], x[s], sizeof(float)*H);
            la_rmsnorm(x[s], st_f32_tensor_ptr(b->ffn_norm), H, la_g_cfg.rms_eps, xn[s]);
        }
#ifdef STRATUM_USE_METAL
        if (la_g_gpu_batch && b->ffn_gate->type == GGML_TYPE_Q4_K
            && b->ffn_up->type == GGML_TYPE_Q4_K) {
            static float* xpk=NULL; static size_t xpkc=0;
            size_t xn_=(size_t)B*H; if(xn_>xpkc){free(xpk);xpk=malloc(xn_*4);xpkc=xn_;}
            for(int s=0;s<B;s++) memcpy(xpk+(size_t)s*H, xn[s], (size_t)H*4);
            uint64_t wo[2]={b->ffn_gate->offset,b->ffn_up->offset};
            int Na[2]={Ff,Ff};
            static float* ygu=NULL; static size_t yguc=0;
            size_t ytot=(size_t)B*Ff*2; if(ytot>yguc){free(ygu);ygu=malloc(ytot*4);yguc=ytot;}
            size_t yoff[2]={0,(size_t)B*Ff*4};
            if(stratum_metal_q4k_multi(wo,Na,2,xpk,ygu,yoff,H,B)==0){
                for(int s=0;s<B;s++){
                    memcpy(fg[s], ygu+(size_t)s*Ff, (size_t)Ff*4);
                    memcpy(fu[s], (char*)ygu+yoff[1]+(size_t)s*Ff*4, (size_t)Ff*4);
                }
            } else {
                la_linear_multix(b->ffn_gate, cxn, cfg, B, Ff, H);
                la_linear_multix(b->ffn_up,   cxn, cfu, B, Ff, H);
            }
        } else
#endif
        {
        la_linear_multix(b->ffn_gate, cxn, cfg, B, Ff, H);
        la_linear_multix(b->ffn_up,   cxn, cfu, B, Ff, H);
        }
        for (int s=0;s<B;s++) la_swiglu(fg[s], fu[s], Ff, fa[s]);
        for (int s=0;s<B;s++) cfa_in[s]=fa[s];
        la_linear_multix(b->ffn_down, cfa_in, cap, B, H, Ff);
        for (int s=0;s<B;s++) for(int i=0;i<H;i++) x[s][i]=xr[s][i]+ap[s][i];
    }

    const float* gain = st_f32_tensor_ptr(la_g_output_norm);
    for (int s=0;s<B;s++) la_rmsnorm(x[s], gain, H, la_g_cfg.rms_eps, xn[s]);
    const GgufTensor* lm = la_g_output_w ? la_g_output_w : la_g_token_embd;
    float* clog[la_B_MAX];
    for (int s=0;s<B;s++) clog[s]=la_gb_logits[s];
    la_linear_multix(lm, cxn, clog, B, V, H);
    return 0;
}

static int la_forward_batch(const int* tokens, const int* positions, int B) {
    return la_forward_batch_ex(tokens, positions, NULL, B);
}


/* ---- persistent worker pool for per-stream sections ---------------------
 * the pool itself lives in stratum_linear.h (st_par_run); la_par_run keeps
 * its per-stream threshold heuristic (serial below 6) */
static void la_par_run(int B, void (^blk)(int)) {
    if (B < 6) { for (int s = 0; s < B; s++) blk(s); return; }
    st_par_run(B, blk);
}

/* Multi-sequence forward: B INDEPENDENT streams, one token each, sharing
 * one weight load per matmul. Each slot s has its own position pos[s] and
 * its own KV stream in la_g_msK/msV at [L][s][kvlen[s]]. Per-slot logits
 * land in la_gb_logits[s]. tokens-per-sweep = B exactly, no accept gamble.
 * Bit-exact per stream vs running that stream single-token. */
static int la_forward_multiseq(const int* tokens, const int* pos,
                               const int* kvlen, int B) {
    int H=la_g_cfg.n_embed, Hd=la_g_cfg.head_dim, Nq=la_g_cfg.n_q_heads;
    int Nk=la_g_cfg.n_kv_heads, Ff=la_g_cfg.n_ff, V=la_g_cfg.vocab_size;
    if (B > la_B_MAX) return -1;
    static float* x[la_B_MAX]; static float* xr[la_B_MAX]; static float* xn[la_B_MAX];
    static float* qb[la_B_MAX]; static float* kb[la_B_MAX]; static float* vb[la_B_MAX];
    static float* ao[la_B_MAX]; static float* ap[la_B_MAX];
    static float* fg[la_B_MAX]; static float* fu[la_B_MAX]; static float* fa[la_B_MAX];
    static int done=0;
    if(!done){for(int s=0;s<la_B_MAX;s++){x[s]=calloc(H,4);xr[s]=calloc(H,4);xn[s]=calloc(H,4);
        qb[s]=calloc(Nq*Hd,4);kb[s]=calloc(Nk*Hd,4);vb[s]=calloc(Nk*Hd,4);
        ao[s]=calloc(Nq*Hd,4);ap[s]=calloc(H,4);fg[s]=calloc(Ff,4);fu[s]=calloc(Ff,4);
        fa[s]=calloc(Ff,4); if(!la_gb_logits[s])la_gb_logits[s]=calloc(V,4);} done=1;}
    for(int s=0;s<B;s++) la_embed_lookup(tokens[s], x[s]);
    const float* cxn[la_B_MAX]; float* cqb[la_B_MAX]; float* ckb[la_B_MAX];
    float* cvb[la_B_MAX]; float* cao[la_B_MAX]; float* cap[la_B_MAX];
    float* cfg[la_B_MAX]; float* cfu[la_B_MAX]; const float* cfa_in[la_B_MAX];
    for(int s=0;s<B;s++){cxn[s]=xn[s];cqb[s]=qb[s];ckb[s]=kb[s];cvb[s]=vb[s];
        cao[s]=ao[s];cap[s]=ap[s];cfg[s]=fg[s];cfu[s]=fu[s];}
    float scale=1.0f/sqrtf((float)Hd);
    /* per-slot KV stride: [L][B][MAX_KV][Nk*Hd] */
    size_t kv_seqstride=(size_t)la_g_ms_maxkv*Nk*Hd;
    size_t kv_laystride=(size_t)B*kv_seqstride;
    static double lt_acc=0; static int lt_n=0;
    double lt0=0; if(getenv("STRATUM_MSTIME")){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);lt0=ts.tv_sec*1e6+ts.tv_nsec/1e3;}
    for(int li=0;li<la_g_cfg.n_layers;li++){
        la_BlockTensors* b=&la_g_blocks[li];
        double aT=0; int _mt=!!getenv("STRATUM_MSTIME");
        if(_mt){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);aT=ts.tv_sec*1e6+ts.tv_nsec/1e3;}
        la_par_run(B, ^(int s){memcpy(xr[s],x[s],sizeof(float)*H);
            la_rmsnorm(x[s],st_f32_tensor_ptr(b->attn_norm),H,la_g_cfg.rms_eps,xn[s]);});
        if(_mt){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
            static double a=0;static int n=0;a+=ts.tv_sec*1e6+ts.tv_nsec/1e3-aT;n++;
            if(n%896==0)fprintf(stderr,"[mst] rmsA %.0f us\n",a/n);}
        la_linear_multix(b->attn_q,cxn,cqb,B,Nq*Hd,H);
        la_linear_multix(b->attn_k,cxn,ckb,B,Nk*Hd,H);
        la_linear_multix(b->attn_v,cxn,cvb,B,Nk*Hd,H);
        /* per-head q/k RMSNorm (Qwen3-style) — must match la_forward_block;
         * absent on plain Llama files (NULL tensor skips) */
        const float* qng = b->attn_q_norm ? st_f32_tensor_ptr(b->attn_q_norm) : NULL;
        const float* kng = b->attn_k_norm ? st_f32_tensor_ptr(b->attn_k_norm) : NULL;
        double rpT=0; if(_mt){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);rpT=ts.tv_sec*1e6+ts.tv_nsec/1e3;}
        { int mp=0; for(int s=0;s<B;s++) if(pos[s]>mp) mp=pos[s];
          la_rope_tables_grow(mp, la_g_cfg.rope_dim/2, la_g_cfg.rope_theta); }
        la_par_run(B, ^(int s){
            if (qng) for (int h=0;h<Nq;h++)
                la_rmsnorm(qb[s]+h*Hd, qng, Hd, la_g_cfg.rms_eps, qb[s]+h*Hd);
            if (kng) for (int h=0;h<Nk;h++)
                la_rmsnorm(kb[s]+h*Hd, kng, Hd, la_g_cfg.rms_eps, kb[s]+h*Hd);
            for(int h=0;h<Nq;h++) la_rope(qb[s]+h*Hd,Hd,la_g_cfg.rope_dim,pos[s],la_g_cfg.rope_theta);
            for(int h=0;h<Nk;h++) la_rope(kb[s]+h*Hd,Hd,la_g_cfg.rope_dim,pos[s],la_g_cfg.rope_theta);
            float* Kbase=la_g_msK+(size_t)li*kv_laystride+(size_t)s*kv_seqstride;
            float* Vbase=la_g_msV+(size_t)li*kv_laystride+(size_t)s*kv_seqstride;
            memcpy(Kbase+(size_t)kvlen[s]*Nk*Hd,kb[s],sizeof(float)*Nk*Hd);
            memcpy(Vbase+(size_t)kvlen[s]*Nk*Hd,vb[s],sizeof(float)*Nk*Hd);
            int klen=kvlen[s]+1;
            const float* Kb2=Kbase; const float* Vb2=Vbase; (void)Kb2;(void)Vb2;
            for(int h=0;h<Nq;h++){
                int kv_h=h*Nk/Nq; const float* qh=qb[s]+h*Hd;
                float lg[la_MAX_KV];
                for(int t=0;t<klen;t++){const float* kt=Kbase+(size_t)t*Nk*Hd+kv_h*Hd;
                    float dot=0;
#if defined(__ARM_NEON) || defined(__aarch64__)
                    /* same f32x4 reduction as the single-stream attn_head —
                     * MS stream logits stay bit-identical to single-stream */
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    int d = 0;
                    for (; d + 16 <= Hd; d += 16) {
                        acc = vfmaq_f32(acc, vld1q_f32(qh + d),      vld1q_f32(kt + d));
                        acc = vfmaq_f32(acc, vld1q_f32(qh + d + 4),  vld1q_f32(kt + d + 4));
                        acc = vfmaq_f32(acc, vld1q_f32(qh + d + 8),  vld1q_f32(kt + d + 8));
                        acc = vfmaq_f32(acc, vld1q_f32(qh + d + 12), vld1q_f32(kt + d + 12));
                    }
                    for (; d + 4 <= Hd; d += 4)
                        acc = vfmaq_f32(acc, vld1q_f32(qh + d), vld1q_f32(kt + d));
                    dot = vaddvq_f32(acc);
                    for (; d < Hd; d++) dot += qh[d] * kt[d];
#else
                    for(int d=0;d<Hd;d++)dot+=qh[d]*kt[d];
#endif
                    lg[t]=dot*scale;}
                la_softmax_inplace(lg,klen);
                float* hd=ao[s]+h*Hd;
                for (int d=0;d<Hd;d++) hd[d]=0.0f;
                for(int t=0;t<klen;t++){const float* vt=Vbase+(size_t)t*Nk*Hd+kv_h*Hd;
                    float p=lg[t];
#if defined(__ARM_NEON) || defined(__aarch64__)
                    /* NOTE: f32x4 lanewise accumulation changes fp add order vs
                     * scalar; identical across all streams/runs (deterministic) */
                    float32x4_t pv=vdupq_n_f32(p);
                    for(int d=0;d+4<=Hd;d+=4)
                        vst1q_f32(hd+d,vfmaq_f32(vld1q_f32(hd+d),pv,vld1q_f32(vt+d)));
                    for(int d=(Hd&~3);d<Hd;d++)hd[d]+=p*vt[d];
#else
                    for(int d=0;d<Hd;d++)hd[d]+=p*vt[d];
#endif
                }
            }
        });
        if(_mt){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
            static double a=0;static int n=0;a+=ts.tv_sec*1e6+ts.tv_nsec/1e3-rpT;n++;
            if(n%896==0)fprintf(stderr,"[mst] attn-region %.0f us\n",a/n);}
        la_linear_multix(b->attn_output,(const float* const*)cao,cap,B,H,Nq*Hd);
        la_par_run(B, ^(int s){for(int i=0;i<H;i++)x[s][i]=xr[s][i]+ap[s][i];
            memcpy(xr[s],x[s],sizeof(float)*H);
            la_rmsnorm(x[s],st_f32_tensor_ptr(b->ffn_norm),H,la_g_cfg.rms_eps,xn[s]);});
        if(_mt){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
            static double a=0;static int n=0;a+=ts.tv_sec*1e6+ts.tv_nsec/1e3-rpT;n++;
            if(n%896==0)fprintf(stderr,"[mst] resid+ffnnorm %.0f us\n",a/n);}
        la_linear_multix(b->ffn_gate,cxn,cfg,B,Ff,H);
        la_linear_multix(b->ffn_up,cxn,cfu,B,Ff,H);
        { const float** cfi=(const float**)cfa_in;
          la_par_run(B, ^(int s){la_swiglu(fg[s],fu[s],Ff,fa[s]);cfi[s]=fa[s];}); }
        la_linear_multix(b->ffn_down,cfa_in,cap,B,H,Ff);
        la_par_run(B, ^(int s){for(int i=0;i<H;i++)x[s][i]=xr[s][i]+ap[s][i];});
        if (li == la_hidden_dump_layer && la_hidden_dump_fp) {
            fwrite(x[0], sizeof(float), H, la_hidden_dump_fp);
            fflush(la_hidden_dump_fp);
        }
    }
    if(getenv("STRATUM_MSTIME")){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
        lt_acc+=ts.tv_sec*1e6+ts.tv_nsec/1e3-lt0; lt_n++;
        if(lt_n%32==0)fprintf(stderr,"[mstime] layers avg %.0f us over %d\n",lt_acc/lt_n,lt_n);}
    const float* gain=st_f32_tensor_ptr(la_g_output_norm);
    la_par_run(B, ^(int s){la_rmsnorm(x[s],gain,H,la_g_cfg.rms_eps,xn[s]);});
    if (getenv("STRATUM_MS_DUMPXN")) {
        FILE* df=fopen(getenv("STRATUM_MS_DUMPXN"),"ab");
        if (df) { fwrite(x[0],4,H,df); fwrite(xn[0],4,H,df); fclose(df); }
    }
    const GgufTensor* lm=la_g_output_w?la_g_output_w:la_g_token_embd;
    float* clog[la_B_MAX]; for(int s=0;s<B;s++)clog[s]=la_gb_logits[s];
    double t0=0; if(getenv("STRATUM_MSTIME")){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);t0=ts.tv_sec*1e6+ts.tv_nsec/1e3;}
    la_linear_multix(lm,cxn,clog,B,V,H);
    if(getenv("STRATUM_MSTIME")){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
        static double acc=0; static int n=0; acc+=ts.tv_sec*1e6+ts.tv_nsec/1e3-t0; n++;
        if(n%32==0)fprintf(stderr,"[mstime] lm_head avg %.0f us over %d calls\n",acc/n,n);}
    return 0;
}

#ifdef STRATUM_USE_METAL
static int la_g_gpu_batch_full = 0;
#else
static int la_g_gpu_batch_full = 0;   /* CPU-only build: GPU batch path never taken */
#endif
#ifdef STRATUM_USE_METAL
/* Full-GPU multi-sequence forward: B independent streams, ENTIRE forward in
 * one Metal command buffer (one sync/sweep). KV cache lives on GPU inside the
 * Metal module. Per-stream logits land in la_gb_logits[s]. Bit-exact per
 * stream vs the single-stream full-GPU path. */
static int la_forward_multiseq_gpu(const int* tokens, const int* pos,
                                   const int* kvlen, int B, int* next_out) {
    int H=la_g_cfg.n_embed, V=la_g_cfg.vocab_size, nL=la_g_cfg.n_layers;
/* lys moved to global la_g_lys */
    if (!la_g_lys_built) {
        for (int i=0;i<nL;i++){
            la_BlockTensors* b=&la_g_blocks[i];
            la_g_lys[i].attn_norm_off=b->attn_norm->offset; la_g_lys[i].ffn_norm_off=b->ffn_norm->offset;
            la_g_lys[i].q_off=b->attn_q->offset; la_g_lys[i].q_tb=b->attn_q->nbytes; la_g_lys[i].q_ty=b->attn_q->type;
            la_g_lys[i].k_off=b->attn_k->offset; la_g_lys[i].k_tb=b->attn_k->nbytes; la_g_lys[i].k_ty=b->attn_k->type;
            la_g_lys[i].v_off=b->attn_v->offset; la_g_lys[i].v_tb=b->attn_v->nbytes; la_g_lys[i].v_ty=b->attn_v->type;
            la_g_lys[i].o_off=b->attn_output->offset; la_g_lys[i].o_tb=b->attn_output->nbytes; la_g_lys[i].o_ty=b->attn_output->type;
            la_g_lys[i].gate_off=b->ffn_gate->offset; la_g_lys[i].gate_tb=b->ffn_gate->nbytes; la_g_lys[i].gate_ty=b->ffn_gate->type;
            la_g_lys[i].up_off=b->ffn_up->offset; la_g_lys[i].up_tb=b->ffn_up->nbytes; la_g_lys[i].up_ty=b->ffn_up->type;
            la_g_lys[i].down_off=b->ffn_down->offset; la_g_lys[i].down_tb=b->ffn_down->nbytes; la_g_lys[i].down_ty=b->ffn_down->type;
        }
        la_g_lys_built=1;
    }
    static float* xb=NULL; static size_t xbc=0;
    size_t need=(size_t)B*H; if(need>xbc){free(xb);xb=malloc(need*4);xbc=need;}
    for (int s=0;s<B;s++) la_embed_lookup(tokens[s], xb+(size_t)s*H);
    int need_logits = (next_out == NULL) || getenv("STRATUM_MS_VERIFY");
    static float* lg=NULL; static size_t lgc=0;
    size_t ln=(size_t)B*V; if(need_logits && ln>lgc){free(lg);lg=malloc(ln*4);lgc=ln;}
    if (need_logits) for (int s=0;s<B;s++) if(!la_gb_logits[s]) la_gb_logits[s]=calloc(V,4);
    const GgufTensor* lm = la_g_output_w ? la_g_output_w : la_g_token_embd;
    int rc = stratum_metal_forward_batched(la_g_lys, nL, la_g_output_norm->offset,
                 lm->offset, lm->nbytes, lm->type==GGML_TYPE_Q6_K, xb,
                 need_logits ? lg : NULL, next_out,
                 H, la_g_cfg.head_dim, la_g_cfg.n_q_heads, la_g_cfg.n_kv_heads,
                 la_g_cfg.n_ff, V, la_g_cfg.rope_dim, pos, la_g_cfg.rope_theta,
                 la_g_cfg.rms_eps, kvlen, la_g_ms_maxkv, B);
    if (rc != 0) return -1;
    if (need_logits) for (int s=0;s<B;s++) memcpy(la_gb_logits[s], lg+(size_t)s*V, (size_t)V*4);
    return 0;
}
#endif

/* stratum_argmax is provided by stratum_engine.h */

int run_llama_arch(int argc, char** argv) {
    stratum_enforce_boundaries();
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]\n"
                "  Reads a GGUF model, runs forward on the provided token IDs,\n"
                "  prints stratum_argmax of the resulting logits + top-1 token id.\n",
                argv[0]);
        return 1;
    }
    fprintf(stderr, "== stratum_v2 — universal GGUF inference (Phase 8.3) ==\n");
    fprintf(stderr, "  model : %s\n", argv[1]);

    if (gguf_open(argv[1], &la_g_gguf) != 0) return 1;
    fprintf(stderr, "  GGUF v%u, %llu tensors, body @ %llu\n",
            la_g_gguf.version,
            (unsigned long long)la_g_gguf.n_tensors,
            (unsigned long long)la_g_gguf.body_offset);
    {
        char* a = gguf_get_string_dup(&la_g_gguf, "general.architecture");
        if (a) {
            la_g_rope_neox = (strcmp(a, "qwen3") == 0);
            fprintf(stderr, "  rope layout: %s (arch %s)\n",
                    la_g_rope_neox ? "NEOX half-split" : "NORM adjacent-pairs", a);
            free(a);
        }
    }

    stratum_linear_init(la_g_gguf.mmap_base, la_g_gguf.mmap_size);
    stratum_engine_init(la_g_gguf.mmap_size);
    if (stratum_load_config(&la_g_gguf, &la_g_cfg) != 0) return 1;
    fprintf(stderr, "\n");
    stratum_print_config(&la_g_cfg);
    fprintf(stderr, "\n");

    if (la_discover_blocks() != 0) return 1;
    {
        const char* hd = getenv("STRATUM_HIDDEN_DUMP");
        if (hd && hd[0]) {
            char path[512];
            if (sscanf(hd, "%d:%511s", &la_hidden_dump_layer, path) == 2
                && la_hidden_dump_layer >= 0 && la_hidden_dump_layer < la_g_cfg.n_layers) {
                la_hidden_dump_fp = fopen(path, "wb");
                if (la_hidden_dump_fp) {
                    fwrite("SHID0001", 1, 8, la_hidden_dump_fp);
                    uint32_t meta[2] = { (uint32_t)la_hidden_dump_layer,
                                         (uint32_t)la_g_cfg.n_embed };
                    fwrite(meta, 4, 2, la_hidden_dump_fp);
                    fprintf(stderr, "  hidden dump: layer %d -> %s\n",
                            la_hidden_dump_layer, path);
                } else {
                    fprintf(stderr, "  hidden dump: cannot open %s\n", path);
                }
            } else {
                fprintf(stderr, "  hidden dump: bad spec '%s' (want <layer>:<path>)\n", hd);
            }
        }
    }
    if (la_allocate_state()  != 0) return 1;

    {
        int ncpu = 0; size_t l = sizeof(ncpu);
        if (sysctlbyname("hw.physicalcpu", &ncpu, &l, NULL, 0) != 0 || ncpu < 1) ncpu = 8;
        int pcpu = 0; size_t pl = sizeof(pcpu);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &pcpu, &pl, NULL, 0) != 0 || pcpu < 1) pcpu = ncpu;
        const char* env_nc = getenv("STRATUM_NCHUNKS");
        if (env_nc) {
            g_st.nchunks = atoi(env_nc);
        } else {
            g_st.nchunks = pcpu;
            /* the old hidden<=4096 -> 6 clamp predates the row-parallel
             * kernels; measured on MiniCPM5-2B Q4_K_M it costs ~15% decode
             * throughput. All P-cores now. */
        }
        if (g_st.nchunks < 1) g_st.nchunks = 1;
        fprintf(stderr, "  parallel matmul: %d chunks (%d P-cores, %d physical)\n",
                g_st.nchunks, pcpu, ncpu);
    }
    /* use_sdot is set once by stratum_linear_init (STRATUM_NO_SDOT /
     * STRATUM_SDOT both honored) — do not override it here. */
    if (g_st.use_sdot)
        fprintf(stderr, "  Q4_K/Q6_K SDOT: ENABLED (default; int8 activations, greedy bit-exact, +0.2%% ppl)\n");
    else
        fprintf(stderr, "  Q4_K/Q6_K SDOT: disabled (STRATUM_SDOT=0; exact-float path)\n");
    la_g_blas_batch = (getenv("STRATUM_BLAS_BATCH") != NULL) ? (atoi(getenv("STRATUM_BLAS_BATCH")) != 0) : 0;
    if (la_g_blas_batch)
        fprintf(stderr, "  batched matmul: BLAS sgemm (dequant+gemm tiles)\n");
    fprintf(stderr, "\n");

#ifdef STRATUM_USE_METAL
    if (getenv("STRATUM_GPU")) {
        const char* mlpath = getenv("STRATUM_METALLIB");
        if (!mlpath) {
            if (access("stratum_q4k.metallib", R_OK) == 0)
                mlpath = "stratum_q4k.metallib";
            else if (access("native/stratum_q4k.metallib", R_OK) == 0)
                mlpath = "native/stratum_q4k.metallib";
            else
                mlpath = "stratum_q4k.metallib";
        }
        if (stratum_metal_init(mlpath, g_st.mmap_base, g_st.mmap_size) == 0) {
            g_st.use_metal = 1;
            if (getenv("STRATUM_GPU_BATCH")) {
                la_g_gpu_batch = 1;
                fprintf(stderr, "  Metal GPU: batched matmul ENABLED (Q4_K, B>=2)\n");
            }
            fprintf(stderr, "  Metal GPU acceleration: ENABLED for Q4_K matmul\n\n");
            if (getenv("STRATUM_GPU_FULL")) {
                la_g_gpu_full = 1;
                fprintf(stderr, "  Metal GPU: FULL forward on-GPU (1 sync/token, KV cache on GPU)\n\n");
            }
            if (getenv("STRATUM_GPU_BATCH_FULL")) {
                la_g_gpu_batch_full = 1;
                fprintf(stderr, "  Metal GPU: FULL batched forward on-GPU (1 sync/sweep, B streams, KV on GPU)\n\n");
            }
        } else {
            fprintf(stderr, "  Metal init failed; falling back to NEON\n\n");
        }
    }
#endif

    madvise((void*)g_st.mmap_base, g_st.mmap_size, MADV_WILLNEED);

    /* V13: 不主动 touch 权重页。MADV_WILLNEED 提示内核异步预读。
     * 正常前向读取自然填充 page cache。不主动 touch 避免瞬间占用物理 RAM。 */

    int n_gen = (argc > 2) ? atoi(argv[2]) : 4;
    int prompt[2048];
    int n_prompt = 0;
    if (argc > 3) {
        for (int i = 3; i < argc && n_prompt < 2048; i++) {
            prompt[n_prompt++] = atoi(argv[i]);
        }
    } else {
        prompt[n_prompt++] = 1;
        prompt[n_prompt++] = 12968;
    }

    if (stratum_validate_prompt_ids(prompt, n_prompt, la_g_cfg.vocab_size) != 0) return 1;
    static StratumVocab la_vocab;
    stratum_vocab_init(&la_g_gguf, la_g_gguf.mmap_base, &la_vocab);
    if (la_vocab.available)
        fprintf(stderr, "  tokenizer: %u tokens loaded (text output enabled)\n", la_vocab.count);
    fprintf(stderr, "  prompt ids:");
    for (int i = 0; i < n_prompt; i++) fprintf(stderr, " %d", prompt[i]);
    fprintf(stderr, "\n  generating %d tokens\n\n", n_gen);

    /* Multi-sequence throughput mode: run B independent streams (here B
     * copies of the prompt) in lockstep. Each sweep commits B tokens.
     * Reports aggregate tok/s = B * per-stream rate. The unbounded
     * throughput axis: weight read once, serves all B streams. */
    int ms_B = 0;
    { const char* e = getenv("STRATUM_MULTISEQ"); if (e) ms_B = atoi(e); }
    if (ms_B >= 1) {
        if (ms_B > la_B_MAX) ms_B = la_B_MAX;
        la_g_ms_maxkv = n_prompt + n_gen + 2;
        size_t cells = (size_t)la_g_cfg.n_layers * ms_B * la_g_ms_maxkv
                     * la_g_cfg.n_kv_heads * la_g_cfg.head_dim;
        la_g_msK = (float*)calloc(cells, sizeof(float));
        la_g_msV = (float*)calloc(cells, sizeof(float));
        la_g_ms_B = ms_B;
        if (!la_g_msK || !la_g_msV) { fprintf(stderr, "  multiseq KV alloc failed\n"); return 1; }
        fprintf(stderr, "  MULTISEQ: %d independent streams, KV %.1f MB\n",
                ms_B, (double)cells*2*4/(1024.0*1024.0));
        int kvlen[la_B_MAX]={0}, pos[la_B_MAX], tok[la_B_MAX];
        int nxt[la_B_MAX];
        struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
        /* prefill all streams (identical prompt) position by position */
        for (int p = 0; p < n_prompt; p++) {
            for (int s=0;s<ms_B;s++){tok[s]=prompt[p];pos[s]=p;}
#ifdef STRATUM_USE_METAL
            if (la_g_gpu_batch_full) { if (la_forward_multiseq_gpu(tok,pos,kvlen,ms_B,nxt)!=0) return 1; }
            else
#endif
            if (la_forward_multiseq(tok,pos,kvlen,ms_B)!=0) return 1;
            for (int s=0;s<ms_B;s++) kvlen[s]++;
        }
        if (!la_g_gpu_batch_full) for (int s=0;s<ms_B;s++) nxt[s]=stratum_argmax(la_gb_logits[s],la_g_cfg.vocab_size);
        clock_gettime(CLOCK_MONOTONIC,&b);
        double pf=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
        if (getenv("STRATUM_MS_DUMP0")) {
            FILE* df = fopen(getenv("STRATUM_MS_DUMP0"), "wb");
            if (df) { fwrite(la_gb_logits[0], 4, la_g_cfg.vocab_size, df); fclose(df); }
        }
        fprintf(stderr,"  multiseq prefill %.3fs; stream0 first tok=%d\n", pf, nxt[0]);
        clock_gettime(CLOCK_MONOTONIC,&a);
        for (int g=0;g<n_gen;g++){
            for (int s=0;s<ms_B;s++){tok[s]=nxt[s];pos[s]=n_prompt+g;}
#ifdef STRATUM_USE_METAL
            if (la_g_gpu_batch_full) { if (la_forward_multiseq_gpu(tok,pos,kvlen,ms_B,nxt)!=0) return 1; }
            else
#endif
            if (la_forward_multiseq(tok,pos,kvlen,ms_B)!=0) return 1;
            if (la_g_gpu_batch_full) {
                for (int s=0;s<ms_B;s++) kvlen[s]++;
            } else {
                for (int s=0;s<ms_B;s++){kvlen[s]++;nxt[s]=stratum_argmax(la_gb_logits[s],la_g_cfg.vocab_size);}
            }
            if (getenv("STRATUM_MS_VERIFY")) {
                int V=la_g_cfg.vocab_size;
                for (int s=1;s<ms_B;s++) {
                    if (nxt[s]!=nxt[0]) { fprintf(stderr,"  [VERIFY] MISMATCH g=%d stream%d tok=%d != stream0 tok=%d\n",g,s,nxt[s],nxt[0]); }
                    if (memcmp(la_gb_logits[s], la_gb_logits[0], (size_t)V*4)!=0) {
                        int diffs=0; for(int i=0;i<V;i++) if(la_gb_logits[s][i]!=la_gb_logits[0][i]) diffs++;
                        fprintf(stderr,"  [VERIFY] g=%d stream%d logits differ from stream0 in %d/%d dims\n",g,s,diffs,V);
                    }
                }
            }
            if (g<8) fprintf(stderr,"  ms step %2d  stream0 argmax=%d\n",g,nxt[0]);
        }
        clock_gettime(CLOCK_MONOTONIC,&b);
        double gn=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
        fprintf(stderr,"\n  [MULTISEQ] %d streams x %d gen tok = %d tok in %.3fs\n",
                ms_B, n_gen, ms_B*n_gen, gn);
        fprintf(stderr,"  aggregate %.1f tok/s  (per-stream %.1f tok/s)\n",
                ms_B*n_gen/gn, n_gen/gn);
        return 0;
    }

    int position = 0;
    int last_tok = -1;

    /* PERPLEXITY mode: teacher-force the prompt and sum NLL of each true
     * next token from the ENGINE'S OWN logits. This is the capability the
     * project never had — it measures the engine's actual output QUALITY
     * (not just bit-equality vs fp32), opening the speed/quality tradeoff.
     * Run with the same env flags (STRATUM_SDOT etc.) to get that
     * config's ppl. STRATUM_PPL=1. */
    if (getenv("STRATUM_PPL")) {
        int V = la_g_cfg.vocab_size;
        double nll = 0.0; int ntok = 0;
        for (int t = 0; t < n_prompt; t++) {
            if (la_forward_one_token(prompt[t], position++) != 0) return 1;
            if (t + 1 < n_prompt) {
                int tgt = prompt[t+1];
                /* log-softmax: log p(tgt) = logit[tgt] - logsumexp(logits) */
                float mx = la_g_logits[0];
                for (int i = 1; i < V; i++) if (la_g_logits[i] > mx) mx = la_g_logits[i];
                double se = 0.0;
                for (int i = 0; i < V; i++) se += exp((double)(la_g_logits[i] - mx));
                double lse = mx + log(se);
                nll += lse - (double)la_g_logits[tgt];
                ntok++;
            }
        }
        double ppl = exp(nll / (ntok > 0 ? ntok : 1));
        fprintf(stderr, "\n  [PPL] %d scored tokens, mean NLL %.4f, perplexity %.4f\n",
                ntok, nll / (ntok>0?ntok:1), ppl);
        fprintf(stdout, "PPL %.6f %d\n", ppl, ntok);
        return 0;
    }

    int _timing = (getenv("STRATUM_TIMING") != NULL);
    struct timespec _tp0, _tp1, _tg0, _tg1;
    if (_timing) clock_gettime(CLOCK_MONOTONIC, &_tp0);
    int pf_B = 8;   /* batched prefill default (~3x on long prompts). Greedy-equal
                     * to per-token prefill, not bit-exact: B>=2 takes the SDOT
                     * pack path whose accumulation order differs from the
                     * single-stream kernels (~0.2 max logit drift on Qwen3-0.6B,
                     * hidden-state drift compounds ~5x/layer — verified same
                     * token sequence). pf_B=1 or STRATUM_SDOT=0 gives bit-exact. */
    /* W16 weights feed the 32-lane AMX tile — fill it (measured ~1.9x
     * prefill on Qwen3-0.6B-W16, same greedy sequence). */
    if (la_g_blocks && la_g_blocks[0].attn_q
        && (la_g_blocks[0].attn_q->type == 43 || la_g_blocks[0].attn_q->type == 44))
        pf_B = 32;
    if (la_hidden_dump_fp) pf_B = 1;  /* probe mode: capture every position */
    { const char* e = getenv("STRATUM_BATCH_PREFILL"); if (e) pf_B = atoi(e); }
#ifdef STRATUM_USE_METAL
    if (la_g_gpu_full) pf_B = 1;  /* full-GPU forward keeps KV on GPU; prefill must go token-by-token */
#endif
    if (pf_B > la_B_MAX) pf_B = la_B_MAX;
    if (pf_B >= 2 && n_prompt >= 2) {
        int t = 0;
        for (; t < n_prompt; ) {
            int Bk = n_prompt - t; if (Bk > pf_B) Bk = pf_B;
            if (Bk == 1) { if (la_forward_one_token(prompt[t], position++) != 0) return 1; t++; continue; }
            int btok[la_B_MAX], bpos[la_B_MAX];
            for (int s = 0; s < Bk; s++) { btok[s]=prompt[t+s]; bpos[s]=position+s; }
            if (la_forward_batch(btok, bpos, Bk) != 0) return 1;
            /* last slot's logits become the prefill output */
            memcpy(la_g_logits, la_gb_logits[Bk-1], sizeof(float)*la_g_cfg.vocab_size);
            la_g_kv_len += Bk;
            position += Bk;
            t += Bk;
        }
        last_tok = prompt[n_prompt-1];
    } else {
        for (int t = 0; t < n_prompt; t++) {
            if (la_forward_one_token(prompt[t], position++) != 0) return 1;
            last_tok = prompt[t];
        }
    }
    if (_timing) clock_gettime(CLOCK_MONOTONIC, &_tp1);
    int next_tok;
#ifdef STRATUM_USE_METAL
    if (la_g_gpu_full && getenv("STRATUM_GPU_FUSED_ARGMAX")) {
        next_tok = stratum_metal_get_last_token();
    } else
#endif
    next_tok = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
    if (getenv("STRATUM_MS_DUMP0")) {
        FILE* df = fopen(getenv("STRATUM_MS_DUMP0"), "wb");
        if (df) { fwrite(la_g_logits, 4, la_g_cfg.vocab_size, df); fclose(df); }
    }
    fprintf(stderr, "  after prefill, stratum_argmax = %d  (logit=%g)\n",
            next_tok, la_g_logits[next_tok]);
    if (_timing) clock_gettime(CLOCK_MONOTONIC, &_tg0);

    if (getenv("STRATUM_DEBUG")) {
        int V = la_g_cfg.vocab_size;

        float* tmp = malloc(sizeof(float) * V);
        memcpy(tmp, la_g_logits, sizeof(float) * V);
        fprintf(stderr, "  TOP-5 prefill logits:\n");
        for (int k = 0; k < 5; k++) {
            int id = stratum_argmax(tmp, V);
            fprintf(stderr, "    [%d] id=%d logit=%g\n", k, id, tmp[id]);
            tmp[id] = -1e30f;
        }
        free(tmp);

        char* ids_env = getenv("STRATUM_DEBUG_IDS");
        if (ids_env) {
            fprintf(stderr, "  Specific token logits at prefill:\n");
            char* s = strdup(ids_env);
            char* p = strtok(s, ", ");
            while (p) {
                int id = atoi(p);
                if (id >= 0 && id < V) {
                    fprintf(stderr, "    id=%d  logit=%g\n", id, la_g_logits[id]);
                }
                p = strtok(NULL, ", ");
            }
            free(s);
        }
    }

    int spec_k = 0;
    { const char* e = getenv("STRATUM_NGRAM_SPEC"); if (e) spec_k = atoi(e); }
    if (spec_k > la_B_MAX - 1) spec_k = la_B_MAX - 1;
    int tree_br = 0;
    { const char* e = getenv("STRATUM_NGRAM_TREE"); if (e) tree_br = atoi(e); }
    if (tree_br > 8) tree_br = 8;

    if (spec_k >= 1) {
        /* n-gram (prompt-lookup) speculative decoding.
         * Each step: propose up to spec_k draft tokens by looking up the
         * latest n-gram suffix in the token history; verify [next_tok,
         * draft...] in ONE batched forward; accept the longest prefix
         * whose argmax chain matches. Greedy-bit-exact: a draft is
         * accepted only if it equals the model's argmax at its parent,
         * so the emitted sequence is identical to single-token greedy. */
        int* hist = (int*)malloc(sizeof(int)*(n_prompt + n_gen + la_B_MAX + 2));
        int hlen = 0;
        for (int i = 0; i < n_prompt; i++) hist[hlen++] = prompt[i];
        int spec_calls = 0, spec_accepts = 0, spec_nodes = 0;
        int g = 0;
        while (g < n_gen) {
            /* build batch: slot0 = next_tok (known-correct), slots 1..d =
             * n-gram drafts continuing from the running hypothesis. */
            int btok[la_B_MAX], bpos[la_B_MAX];
            btok[0] = next_tok; bpos[0] = position;
            int B = 1;

            if (tree_br >= 2) {
                /* ---- n-gram TREE speculation ---------------------------
                 * Each node expands up to tree_br DISTINCT continuations
                 * found by prompt-lookup along its own ancestor context.
                 * One batched forward verifies the whole tree; argmax being
                 * single-valued means at most one child per node matches,
                 * so the accepted path is unique. Output = greedy exactly. */
                int n_tok[la_B_MAX], n_par[la_B_MAX], n_dep[la_B_MAX];
                int front[la_B_MAX], nfront[la_B_MAX];
                int nn = 1, nf = 1;
                n_tok[0] = next_tok; n_par[0] = -1; n_dep[0] = 0; front[0] = 0;
                static int* ctx = NULL; static size_t ctxcap = 0;
                size_t ctxneed = (size_t)hlen + la_B_MAX;
                if (ctxneed > ctxcap) {
                    free(ctx); ctx = malloc(ctxneed * sizeof(int));
                    ctxcap = ctxneed;
                }
                while (nn <= spec_k && nf > 0) {
                    int nnf = 0;
                    for (int f = 0; f < nf && nn <= spec_k; f++) {
                        int node = front[f];
                        /* ctx = committed hist + ancestor path tokens */
                        int clen = hlen, ch[la_B_MAX], cn = 0;
                        for (int a = node;; a = n_par[a]) {
                            ch[cn++] = a; if (a == 0) break;
                        }
                        memcpy(ctx, hist, sizeof(int) * hlen);
                        for (int a = cn - 1; a >= 0; a--)
                            ctx[clen++] = n_tok[ch[a]];
                        /* distinct continuations: try ng=3 then ng=2 */
                        int props[8], np = 0;
                        for (int ng = 3; ng >= 2 && np < tree_br; ng--) {
                            if (clen < ng + 1) continue;
                            const int* suf = ctx + clen - ng;
                            for (int st = clen - ng - 1;
                                 st >= 0 && np < tree_br; st--) {
                                int m = 1;
                                for (int i = 0; i < ng; i++)
                                    if (ctx[st + i] != suf[i]) { m = 0; break; }
                                if (!m) continue;
                                int cand = ctx[st + ng], seen = 0;
                                for (int i = 0; i < np; i++)
                                    if (props[i] == cand) seen = 1;
                                if (!seen) props[np++] = cand;
                            }
                        }
                        for (int i = 0; i < np && nn <= spec_k; i++) {
                            n_tok[nn] = props[i]; n_par[nn] = node;
                            n_dep[nn] = n_dep[node] + 1;
                            nfront[nnf++] = nn; nn++;
                        }
                    }
                    nf = nnf;
                    memcpy(front, nfront, sizeof(int) * nf);
                }
                B = nn;
                if (B > 1) {
                    int bpar[la_B_MAX];
                    for (int s = 0; s < B; s++) {
                        btok[s] = n_tok[s];
                        bpos[s] = position + n_dep[s];
                        bpar[s] = n_par[s];
                    }
                    if (la_forward_batch_ex(btok, bpos, bpar, B) != 0)
                        return 1;
                    spec_calls++;
                    spec_nodes += B;
                    /* unique accepted path: follow argmax-matching child */
                    int path[la_B_MAX]; int plen = 1, cur = 0;
                    path[0] = 0;
                    int final_argm;
                    for (;;) {
                        int argm = stratum_argmax(la_gb_logits[cur],
                                                  la_g_cfg.vocab_size);
                        int nxt = -1;
                        for (int s = 1; s < B; s++)
                            if (n_par[s] == cur && n_tok[s] == argm) {
                                nxt = s; break;
                            }
                        if (nxt < 0) { final_argm = argm; break; }
                        path[plen++] = nxt; cur = nxt;
                    }
                    for (int i = 0; i < plen && g < n_gen; i++) {
                        fprintf(stderr,
                            "  step %2d  in=%d  stratum_argmax=%d  (tspec)\n",
                            g, n_tok[path[i]],
                            (i + 1 < plen) ? n_tok[path[i + 1]] : final_argm);
                        hist[hlen++] = n_tok[path[i]];
                        g++;
                    }
                    spec_accepts += plen - 1;
                    next_tok = final_argm;
                    /* repack accepted-path KV into the committed prefix:
                     * node path[i] was written at kv slot kvlen+path[i] and
                     * must live at kvlen+i (path[i] >= i always in BFS
                     * order, so in-order moves never clobber a source). */
                    static float* reptmp = NULL;
                    static size_t  repcap = 0;
                    size_t kvblk = (size_t)la_g_cfg.n_kv_heads
                                 * la_g_cfg.head_dim;
                    if (kvblk > repcap) {
                        free(reptmp); reptmp = malloc(kvblk * sizeof(float));
                        repcap = kvblk;
                    }
                    for (int li = 0; li < la_g_cfg.n_layers; li++) {
                        float* Kl = la_g_K_cache
                            + (size_t)li * la_MAX_KV * kvblk;
                        float* Vl = la_g_V_cache
                            + (size_t)li * la_MAX_KV * kvblk;
                        for (int i = 1; i < plen; i++) {
                            if (path[i] == i) continue;
                            size_t so = (size_t)(la_g_kv_len + path[i]) * kvblk;
                            size_t do_ = (size_t)(la_g_kv_len + i) * kvblk;
                            memcpy(reptmp, Kl + so, kvblk * 4);
                            memcpy(Kl + do_, reptmp, kvblk * 4);
                            memcpy(reptmp, Vl + so, kvblk * 4);
                            memcpy(Vl + do_, reptmp, kvblk * 4);
                        }
                    }
                    la_g_kv_len += plen;
                    position    += plen;
                    continue;
                }
                /* B == 1: fall through to single-token step below */
            }
            int hyp[la_B_MAX]; hyp[0] = next_tok; int hn = 1;
            while (B <= spec_k) {
                /* propose token after the current hypothesis tail via
                 * n-gram lookup over hist + accepted hypothesis so far. */
                int prop = -1;
                for (int ng = 3; ng >= 2 && prop < 0; ng--) {
                    int total = hlen + hn;
                    if (total < ng + 1) continue;
                    /* suffix = last ng tokens of (hist + hyp) */
                    int suf[3];
                    for (int i = 0; i < ng; i++) {
                        int idx = total - ng + i;
                        suf[i] = (idx < hlen) ? hist[idx] : hyp[idx - hlen];
                    }
                    for (int start = total - ng - 1; start >= 0; start--) {
                        int match = 1;
                        for (int i = 0; i < ng; i++) {
                            int idx = start + i;
                            int tk = (idx < hlen) ? hist[idx] : hyp[idx - hlen];
                            if (tk != suf[i]) { match = 0; break; }
                        }
                        if (match) {
                            int nidx = start + ng;
                            prop = (nidx < hlen) ? hist[nidx] : hyp[nidx - hlen];
                            break;
                        }
                    }
                }
                if (prop < 0) break;
                btok[B] = prop; bpos[B] = position + B;
                hyp[hn++] = prop;
                B++;
            }

            if (B == 1) {
                /* no draft: plain single-token step */
                if (la_forward_one_token(next_tok, position) != 0) return 1;
#ifdef STRATUM_USE_METAL
                int argm;
                int have_logits = 1;
                if (la_g_gpu_full && getenv("STRATUM_GPU_FUSED_ARGMAX")) { argm = stratum_metal_get_last_token(); have_logits = 0; }
                else argm = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
#else
                int argm = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
                int have_logits = 1;
#endif
                if (have_logits) stratum_logits_dump_record(la_g_logits, la_g_cfg.vocab_size, argm);
                fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  logit=%g\n",
                        g, next_tok, argm, la_g_logits[argm]);
                if (la_vocab.available) {
                    char tok_text[256];
                    stratum_decode_token(&la_vocab, argm, tok_text, sizeof(tok_text));
                    fprintf(stdout, "%s", tok_text);
                    fflush(stdout);
                }
                hist[hlen++] = next_tok;
                position++; g++;
                next_tok = argm;
                continue;
            }

            /* batched verify of the B positions */
#ifdef STRATUM_USE_METAL
            if (la_g_gpu_full) {
                /* V5: GPU逐token verify. Each slot forwarded via
                 * stratum_metal_forward (1 dispatch/token). KV cache
                 * stays on GPU across all slots. */
                spec_calls++;
                int accepted = 0;
                int final_argm = 0;
                int saved_kv_len = la_g_kv_len;
                /* Ensure la_gb_logits is allocated (normally done by la_forward_batch) */
                for (int s = 0; s < B; s++) {
                    if (!la_gb_logits[s]) la_gb_logits[s] = (float*)calloc(la_g_cfg.vocab_size, sizeof(float));
                }
                for (int s = 0; s < B; s++) {
                    la_g_kv_len = saved_kv_len + s;
                    if (getenv("STRATUM_SPEC_DBG")) fprintf(stderr, "  [spec] s=%d tok=%d pos=%d kv=%d\n", s, btok[s], bpos[s], la_g_kv_len);
                    if (la_forward_one_token(btok[s], bpos[s]) != 0) return 1;
                    if (getenv("STRATUM_SPEC_DBG")) fprintf(stderr, "  [spec] s=%d fwd done, kv=%d logit[0]=%.4f\n", s, la_g_kv_len, la_g_logits ? la_g_logits[0] : -999.0f);
#ifdef STRATUM_USE_METAL
                    if (la_g_gpu_full && getenv("STRATUM_GPU_FUSED_ARGMAX")) final_argm = stratum_metal_get_last_token();
                    else final_argm = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
#else
                    final_argm = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
#endif
                    if (getenv("STRATUM_SPEC_DBG")) fprintf(stderr, "  [spec] s=%d argmax=%d\n", s, final_argm);
                    memcpy(la_gb_logits[s], la_g_logits, la_g_cfg.vocab_size * sizeof(float));
                    if (getenv("STRATUM_SPEC_DBG")) fprintf(stderr, "  [spec] s=%d copied\n", s);
                    if (s < B - 1 && final_argm != btok[s+1]) {
                        accepted = s;
                        break;
                    }
                    if (s == B - 1) {
                        /* Last slot: all drafts accepted */
                        accepted = B - 1;
                    } else {
                        accepted = s + 1;
                    }
                }
                /* Restore kv_len: la_forward_one_token already incremented
                 * kv_len for each slot forwarded (0..accepted inclusive).
                 * After loop, kv_len = saved + accepted + 1 (from the last
                 * la_forward_one_token call). But if rejected, the rejection
                 * slot's KV is invalid — only committed slots 0..accepted-1
                 * plus slot 'accepted' (whose prediction is the new token).
                 * So kv_len = saved + accepted + 1 is correct in both cases.
                 * However, la_forward_one_token already set kv_len correctly
                 * for the last executed slot, so we only need to fix it when
                 * break happened (accepted < B): kv_len was set to saved+s+1
                 * by the last la_forward_one_token, then we set la_g_kv_len =
                 * saved + s before the next iteration which didn't happen.
                 * The break means kv_len = saved + accepted + 1 (from the
                 * la_forward_one_token at slot 'accepted'). This is correct. */
                /* For all-accepted case: kv_len = saved + B (from B increments).
                 * For rejection at slot s: kv_len = saved + s + 1.
                 * Both equal saved + accepted + 1. So this is correct: */
                la_g_kv_len = saved_kv_len + accepted + 1;
                /* accepted = number of slots that matched (0..B-1) */
                /* The token predicted by slot 'accepted-1' is final_argm if
                 * accepted < B (rejection), or by slot B-1 if all accepted.
                 * But if accepted == B, we need argmax of last slot. */
                if (accepted < B) {
                    /* rejected at slot 'accepted': final_argm is from slot 'accepted' */
                    next_tok = final_argm;
                } else {
                    /* all B slots accepted: next_tok = argmax of last slot */
                    next_tok = final_argm;
                }
                /* Emit accepted+1 tokens (slot 0 + accepted drafts) */
                for (int ss = 0; ss <= accepted && g < n_gen; ss++) {
                    int emit_argm = (ss < accepted) ? btok[ss+1]
                                 : (ss == accepted) ? next_tok : next_tok;
                    fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  (spec)\n",
                            g, btok[ss], emit_argm);
                    hist[hlen++] = btok[ss];
                    g++;
                }
                spec_accepts += accepted;
                /* la_g_kv_len already set to saved_kv_len + accepted + 1 above */
                position    += accepted + 1;
                continue;
            }
#endif
            if (la_forward_batch(btok, bpos, B) != 0) return 1;
            spec_calls++;
            spec_nodes += B;
            /* slot s predicts the token AFTER btok[s]. Accept draft
             * btok[s+1] iff it equals argmax(slot s). */
            int accepted = 0;   /* number of drafts accepted */
            for (int s = 0; s < B - 1; s++) {
                int argm = stratum_argmax(la_gb_logits[s], la_g_cfg.vocab_size);
                if (argm == btok[s+1]) accepted++;
                else break;
            }
            /* commit: btok[0..accepted] are real tokens; emit them, and
             * the (accepted+1)-th token is argmax(slot accepted). */
            for (int s = 0; s <= accepted && g < n_gen; s++) {
                int emit_in = btok[s];
                fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  (spec)\n",
                        g, emit_in, (s < accepted) ? btok[s+1]
                                     : stratum_argmax(la_gb_logits[accepted], la_g_cfg.vocab_size));
                hist[hlen++] = btok[s];
                g++;
            }
            spec_accepts += accepted;
            /* KV: slots 0..accepted are committed; advance kv_len by
             * accepted+1 (slot0 + accepted drafts that became real). The
             * rejected slots' KV is overwritten next iteration. */
            la_g_kv_len += accepted + 1;
            position    += accepted + 1;
            next_tok = stratum_argmax(la_gb_logits[accepted], la_g_cfg.vocab_size);
        }
        if (getenv("STRATUM_TIMING") || getenv("STRATUM_SPEC_STATS"))
            fprintf(stderr, "\n  [ngram-spec] %d batched calls, %d drafts accepted "
                    "(%.2f tok/call)%s\n", spec_calls, spec_accepts,
                    spec_calls ? (double)(spec_accepts + spec_calls) / spec_calls : 0.0,
                    tree_br >= 2 ? "" : "");
            if (tree_br >= 2 && spec_calls)
                fprintf(stderr, "  [tree-spec] avg batch width %.1f nodes/call\n",
                        (double)spec_nodes / spec_calls);
        free(hist);
    } else
#ifdef STRATUM_USE_METAL
    /* Chained GPU decode: the token ring + embd gather live on GPU — all
     * n_gen command buffers are submitted back-to-back with no CPU wait
     * between tokens, removing the per-token sync gap. */
    if (la_g_gpu_full && getenv("STRATUM_GPU_CHAIN") && n_gen <= 1000
        && la_g_token_embd->type == GGML_TYPE_F32) {
        const GgufTensor* lm = la_g_output_w ? la_g_output_w : la_g_token_embd;
        stratum_metal_chain_seed(0, next_tok);
        int ok = 1;
        for (int s = 0; s < n_gen && ok; s++) {
            int rc = stratum_metal_forward(la_g_lys, la_g_cfg.n_layers, la_g_output_norm->offset,
                lm->offset, lm->nbytes, lm->type == GGML_TYPE_Q6_K,
                NULL, NULL,
                la_g_cfg.n_embed, la_g_cfg.head_dim, la_g_cfg.n_q_heads,
                la_g_cfg.n_kv_heads, la_g_cfg.n_ff, la_g_cfg.vocab_size,
                la_g_cfg.rope_dim, position + s, la_g_cfg.rope_theta,
                la_g_cfg.rms_eps, la_g_kv_len + s, la_MAX_KV,
                la_g_rope_neox,
                la_g_token_embd->offset, la_g_token_embd->nbytes, s);
            if (rc != 0) ok = 0;
        }
        if (!ok || stratum_metal_chain_wait() != 0) {
            fprintf(stderr, "  chain decode failed\n"); return 1;
        }
        const uint32_t* ring = stratum_metal_chain_ring();
        for (int s = 0; s < n_gen; s++) {
            int tok = (int)ring[(s + 1) & 255];
            fprintf(stderr, "  step %2d  stratum_argmax=%d  (chain)\n", s, tok);
            if (la_vocab.available) {
                char tok_text[256];
                stratum_decode_token(&la_vocab, tok, tok_text, sizeof(tok_text));
                fprintf(stdout, "%s", tok_text);
            }
        }
        fflush(stdout);
        next_tok = (int)ring[n_gen & 255];
        la_g_kv_len += n_gen;
        position    += n_gen;
    } else
#endif
    for (int g = 0; g < n_gen; g++) {
        last_tok = next_tok;
        if (la_forward_one_token(last_tok, position++) != 0) return 1;
        int fused = 0;
#ifdef STRATUM_USE_METAL
        if (la_g_gpu_full && getenv("STRATUM_GPU_FUSED_ARGMAX")) { next_tok = stratum_metal_get_last_token(); fused = 1; }
        else next_tok = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
#else
        next_tok = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
#endif
        if (!fused)
            stratum_logits_dump_record(la_g_logits, la_g_cfg.vocab_size, next_tok);
        fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  logit=%g\n",
                g, last_tok, next_tok, la_g_logits[next_tok]);
        if (la_vocab.available) {
            char tok_text[256];
            stratum_decode_token(&la_vocab, next_tok, tok_text, sizeof(tok_text));
            fprintf(stdout, "%s", tok_text);
            fflush(stdout);
        }
    }
    if (_timing) {
        clock_gettime(CLOCK_MONOTONIC, &_tg1);
        double pf = (_tp1.tv_sec-_tp0.tv_sec)+(_tp1.tv_nsec-_tp0.tv_nsec)/1e9;
        double gn = (_tg1.tv_sec-_tg0.tv_sec)+(_tg1.tv_nsec-_tg0.tv_nsec)/1e9;
        fprintf(stderr,
            "\n  [timing] prefill %d tok: %.3fs (%.1f ms/tok)  |  "
            "gen %d tok: %.3fs (%.1f ms/tok, %.1f tok/s)\n",
            n_prompt, pf, 1000.0*pf/n_prompt,
            n_gen, gn, 1000.0*gn/n_gen, n_gen/gn);
#ifdef STRATUM_USE_METAL
        if (g_st.use_metal) {
            long nd = 0; double ds = 0.0;
            stratum_metal_dispatch_stats(&nd, &ds);
            fprintf(stderr,
                "  [gpu] %ld dispatches, %.3fs total GPU round-trip "
                "(%.0f us/dispatch, %.1f dispatches/tok)\n",
                nd, ds, nd ? 1e6*ds/nd : 0.0, n_gen ? (double)nd/n_gen : 0.0);
        }
#endif
        if (getenv("STRATUM_TYPETIME")) {
            const char* nm[32] = {0};
            nm[GGML_TYPE_Q4_K&31]="Q4_K"; nm[GGML_TYPE_Q6_K&31]="Q6_K";
            nm[GGML_TYPE_Q5_K&31]="Q5_K"; nm[GGML_TYPE_Q8_0&31]="Q8_0";
            nm[GGML_TYPE_F16&31]="F16"; nm[GGML_TYPE_F32&31]="F32";
            nm[GGML_TYPE_Q2_K&31]="Q2_K"; nm[GGML_TYPE_Q3_K&31]="Q3_K";
            fprintf(stderr, "  [typetime] per-quant matmul totals:\n");
            for (int i=0;i<32;i++) if (g_st.typecalls[i])
                fprintf(stderr, "    %-5s %.3fs over %ld calls (%.3f ms/call)\n",
                    nm[i]?nm[i]:"?", g_st.typesecs[i], g_st.typecalls[i],
                    1000.0*g_st.typesecs[i]/g_st.typecalls[i]);
        }
    }

    gguf_close(&la_g_gguf);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Architecture registration — no model-specific names in stratum.c   */
/* ------------------------------------------------------------------ */

static const StratumArch stratum_arch_llama = {
    .arch_names   = "llama,qwen3",
    .description  = "Llama-family (Llama 1/2/3, TinyLlama, Mistral, Qwen2-dense, etc.)",
    .run          = run_llama_arch,
};

STRATUM_REGISTER_ARCH(stratum_arch_llama);
