
#define _GNU_SOURCE

#include "stratum_arch.h"
#include "stratum_linear.h"
#include "stratum_engine.h"

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
    double ss = 0.0;
    for (int i = 0; i < N; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)N + (double)eps));
    for (int i = 0; i < N; i++) y[i] = x[i] * scale * gain[i];
}

static void la_swiglu(const float* g, const float* u, int N, float* y) {
    for (int i = 0; i < N; i++) {
        float gv = g[i];
        float s  = gv / (1.0f + expf(-gv));
        y[i] = s * u[i];
    }
}

static void la_rope(float* x, int head_dim, int rope_dim, int position, float theta) {
    int n_pairs = rope_dim / 2;
    for (int k = 0; k < n_pairs; k++) {
        float freq  = 1.0f / powf(theta, (float)(2 * k) / (float)rope_dim);
        float angle = (float)position * freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = x[2 * k];
        float x1 = x[2 * k + 1];
        x[2 * k]     = x0 * c - x1 * s;
        x[2 * k + 1] = x0 * s + x1 * c;
    }
    (void)head_dim;
}

static void la_softmax_inplace(float* x, int N) {
    float maxv = x[0];
    for (int i = 1; i < N; i++) if (x[i] > maxv) maxv = x[i];
    double sum = 0.0;
    for (int i = 0; i < N; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < N; i++) x[i] *= inv;
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

#define la_B_MAX 32
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
    dispatch_apply(ntile, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t ti) {
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
    if (w->type == GGML_TYPE_Q4_K) {
        ST_PAR_ROWS(N, {
            const float* xrow[la_B_MAX];
            for (int s = 0; s < B; s++) xrow[s] = xs[s];
            float out[la_B_MAX];
            q4k_dot_row_neon_multix(st_q4k_row_ptr(w, K, r), K, xrow, B, out);
            for (int s = 0; s < B; s++) ys[s][r] = out[s];
        });
        return;
    }
    if (w->type == GGML_TYPE_Q6_K) {
        ST_PAR_ROWS(N, {
            const float* xrow[la_B_MAX];
            for (int s = 0; s < B; s++) xrow[s] = xs[s];
            float out[la_B_MAX];
            q6k_dot_row_neon_multix(st_q6k_row_ptr(w, K, r), K, xrow, B, out);
            for (int s = 0; s < B; s++) ys[s][r] = out[s];
        });
        return;
    }
    for (int s = 0; s < B; s++) st_linear_dispatch(w, xs[s], ys[s], N, K);
}

static void la_forward_block(int li, int position) {
    la_BlockTensors* b = &la_g_blocks[li];
    int H  = la_g_cfg.n_embed;
    int Hd = la_g_cfg.head_dim;
    int Nq = la_g_cfg.n_q_heads;
    int Nk = la_g_cfg.n_kv_heads;
    int Ff = la_g_cfg.n_ff;

    memcpy(la_g_x_resid, la_g_x, sizeof(float) * H);
    {
        const float* gain = st_f32_tensor_ptr(b->attn_norm);
        la_rmsnorm(la_g_x, gain, H, la_g_cfg.rms_eps, la_g_xn);
    }

    st_q4k_group(la_g_xn, H,
                 b->attn_q, la_g_q_buf, Nq * Hd,
                 b->attn_k, la_g_k_buf, Nk * Hd,
                 b->attn_v, la_g_v_buf, Nk * Hd);

    for (int h = 0; h < Nq; h++) {
        la_rope(la_g_q_buf + h * Hd, Hd, la_g_cfg.rope_dim, position, la_g_cfg.rope_theta);
    }
    for (int h = 0; h < Nk; h++) {
        la_rope(la_g_k_buf + h * Hd, Hd, la_g_cfg.rope_dim, position, la_g_cfg.rope_theta);
    }

    int kv_len_now = la_g_kv_len + 1;
    {
        size_t per_layer = (size_t)la_MAX_KV * Nk * Hd;
        size_t off = (size_t)li * per_layer + (size_t)la_g_kv_len * Nk * Hd;
        memcpy(la_g_K_cache + off, la_g_k_buf, sizeof(float) * Nk * Hd);
        memcpy(la_g_V_cache + off, la_g_v_buf, sizeof(float) * Nk * Hd);
    }

    float scale = 1.0f / sqrtf((float)Hd);
    for (int h = 0; h < Nq; h++) {
        int kv_h = h * Nk / Nq;
        const float* qh = la_g_q_buf + h * Hd;
        size_t per_layer = (size_t)la_MAX_KV * Nk * Hd;
        const float* K_layer = la_g_K_cache + (size_t)li * per_layer;
        const float* V_layer = la_g_V_cache + (size_t)li * per_layer;

        float logits[la_MAX_KV];
        for (int t = 0; t < kv_len_now; t++) {
            const float* kt = K_layer + (size_t)t * Nk * Hd + kv_h * Hd;
            float dot = 0.0f;
            for (int d = 0; d < Hd; d++) dot += qh[d] * kt[d];
            logits[t] = dot * scale;
        }
        la_softmax_inplace(logits, kv_len_now);

        float* head_out = la_g_attn_out + h * Hd;
        memset(head_out, 0, sizeof(float) * Hd);
        for (int t = 0; t < kv_len_now; t++) {
            const float* vt = V_layer + (size_t)t * Nk * Hd + kv_h * Hd;
            float p = logits[t];
            for (int d = 0; d < Hd; d++) head_out[d] += p * vt[d];
        }
    }

    static float attn_proj[8192];
    if (H > 8192) { fprintf(stderr, "H exceeds buffer\n"); exit(2); }
    st_linear_dispatch(b->attn_output, la_g_attn_out, attn_proj, H, Nq * Hd);
    for (int i = 0; i < H; i++) la_g_x[i] = la_g_x_resid[i] + attn_proj[i];

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

    for (int li = 0; li < la_g_cfg.n_layers; li++) {
        la_forward_block(li, position);
    }

    {
        const float* gain = st_f32_tensor_ptr(la_g_output_norm);
        la_rmsnorm(la_g_x, gain, H, la_g_cfg.rms_eps, la_g_xn);
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
    int rc = stratum_metal_forward(la_g_lys, nL, la_g_output_norm->offset,
                                   lm->offset, lm->nbytes, lm->type == GGML_TYPE_Q6_K,
                                   la_g_x, logits_ptr,
                                   H, la_g_cfg.head_dim, la_g_cfg.n_q_heads, la_g_cfg.n_kv_heads,
                                   la_g_cfg.n_ff, V, la_g_cfg.rope_dim, position,
                                   la_g_cfg.rope_theta, la_g_cfg.rms_eps, la_g_kv_len, la_MAX_KV);
    if (rc != 0) return -1;
    /* V-opt: if fused argmax, get token from GPU.
     * Write token to la_g_logits[0] as a sentinel — caller checks
     * STRATUM_GPU_FUSED_ARGMAX and uses stratum_metal_get_last_token(). */
    la_g_kv_len++;
    return 0;
}
#endif

static int la_forward_batch(const int* tokens, const int* positions, int B) {
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
    for(int li=0;li<la_g_cfg.n_layers;li++){
        la_BlockTensors* b=&la_g_blocks[li];
        for(int s=0;s<B;s++){memcpy(xr[s],x[s],sizeof(float)*H);
            la_rmsnorm(x[s],st_f32_tensor_ptr(b->attn_norm),H,la_g_cfg.rms_eps,xn[s]);}
        la_linear_multix(b->attn_q,cxn,cqb,B,Nq*Hd,H);
        la_linear_multix(b->attn_k,cxn,ckb,B,Nk*Hd,H);
        la_linear_multix(b->attn_v,cxn,cvb,B,Nk*Hd,H);
        for(int s=0;s<B;s++){
            for(int h=0;h<Nq;h++) la_rope(qb[s]+h*Hd,Hd,la_g_cfg.rope_dim,pos[s],la_g_cfg.rope_theta);
            for(int h=0;h<Nk;h++) la_rope(kb[s]+h*Hd,Hd,la_g_cfg.rope_dim,pos[s],la_g_cfg.rope_theta);
            float* Kbase=la_g_msK+(size_t)li*kv_laystride+(size_t)s*kv_seqstride;
            float* Vbase=la_g_msV+(size_t)li*kv_laystride+(size_t)s*kv_seqstride;
            memcpy(Kbase+(size_t)kvlen[s]*Nk*Hd,kb[s],sizeof(float)*Nk*Hd);
            memcpy(Vbase+(size_t)kvlen[s]*Nk*Hd,vb[s],sizeof(float)*Nk*Hd);
        }
        for(int s=0;s<B;s++){
            int klen=kvlen[s]+1;
            const float* Kbase=la_g_msK+(size_t)li*kv_laystride+(size_t)s*kv_seqstride;
            const float* Vbase=la_g_msV+(size_t)li*kv_laystride+(size_t)s*kv_seqstride;
            for(int h=0;h<Nq;h++){
                int kv_h=h*Nk/Nq; const float* qh=qb[s]+h*Hd;
                float lg[la_MAX_KV];
                for(int t=0;t<klen;t++){const float* kt=Kbase+(size_t)t*Nk*Hd+kv_h*Hd;
                    float dot=0;for(int d=0;d<Hd;d++)dot+=qh[d]*kt[d];lg[t]=dot*scale;}
                la_softmax_inplace(lg,klen);
                float* hd=ao[s]+h*Hd; memset(hd,0,sizeof(float)*Hd);
                for(int t=0;t<klen;t++){const float* vt=Vbase+(size_t)t*Nk*Hd+kv_h*Hd;
                    float p=lg[t];for(int d=0;d<Hd;d++)hd[d]+=p*vt[d];}
            }
        }
        la_linear_multix(b->attn_output,(const float* const*)cao,cap,B,H,Nq*Hd);
        for(int s=0;s<B;s++)for(int i=0;i<H;i++)x[s][i]=xr[s][i]+ap[s][i];
        for(int s=0;s<B;s++){memcpy(xr[s],x[s],sizeof(float)*H);
            la_rmsnorm(x[s],st_f32_tensor_ptr(b->ffn_norm),H,la_g_cfg.rms_eps,xn[s]);}
        la_linear_multix(b->ffn_gate,cxn,cfg,B,Ff,H);
        la_linear_multix(b->ffn_up,cxn,cfu,B,Ff,H);
        for(int s=0;s<B;s++)la_swiglu(fg[s],fu[s],Ff,fa[s]);
        for(int s=0;s<B;s++)cfa_in[s]=fa[s];
        la_linear_multix(b->ffn_down,cfa_in,cap,B,H,Ff);
        for(int s=0;s<B;s++)for(int i=0;i<H;i++)x[s][i]=xr[s][i]+ap[s][i];
    }
    const float* gain=st_f32_tensor_ptr(la_g_output_norm);
    for(int s=0;s<B;s++)la_rmsnorm(x[s],gain,H,la_g_cfg.rms_eps,xn[s]);
    const GgufTensor* lm=la_g_output_w?la_g_output_w:la_g_token_embd;
    float* clog[la_B_MAX]; for(int s=0;s<B;s++)clog[s]=la_gb_logits[s];
    la_linear_multix(lm,cxn,clog,B,V,H);
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

    stratum_linear_init(la_g_gguf.mmap_base, la_g_gguf.mmap_size);
    stratum_engine_init(la_g_gguf.mmap_size);
    if (stratum_load_config(&la_g_gguf, &la_g_cfg) != 0) return 1;
    fprintf(stderr, "\n");
    stratum_print_config(&la_g_cfg);
    fprintf(stderr, "\n");

    if (la_discover_blocks() != 0) return 1;
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
            if (la_g_cfg.n_embed <= 4096 && g_st.nchunks > 6) g_st.nchunks = 6;
        }
        if (g_st.nchunks < 1) g_st.nchunks = 1;
        fprintf(stderr, "  parallel matmul: %d chunks (%d P-cores, %d physical)\n",
                g_st.nchunks, pcpu, ncpu);
    }
    {
        const char* e_sdot = getenv("STRATUM_SDOT");
        g_st.use_sdot = (e_sdot != NULL) ? (atoi(e_sdot) != 0) : 1;
    }
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
    int pf_B = 8;   /* batched prefill default (bit-exact, ~3x on long prompts) */
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
        int spec_calls = 0, spec_accepts = 0;
        int g = 0;
        while (g < n_gen) {
            /* build batch: slot0 = next_tok (known-correct), slots 1..d =
             * n-gram drafts continuing from the running hypothesis. */
            int btok[la_B_MAX], bpos[la_B_MAX];
            btok[0] = next_tok; bpos[0] = position;
            int B = 1;
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
                    "(%.2f tok/call)\n", spec_calls, spec_accepts,
                    spec_calls ? (double)(spec_accepts + spec_calls) / spec_calls : 0.0);
        free(hist);
    } else
    for (int g = 0; g < n_gen; g++) {
        last_tok = next_tok;
        if (la_forward_one_token(last_tok, position++) != 0) return 1;
#ifdef STRATUM_USE_METAL
        if (la_g_gpu_full && getenv("STRATUM_GPU_FUSED_ARGMAX")) next_tok = stratum_metal_get_last_token();
        else next_tok = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
#else
        next_tok = stratum_argmax(la_g_logits, la_g_cfg.vocab_size);
#endif
        fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  logit=%g\n",
                g, last_tok, next_tok, la_g_logits[next_tok]);
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
    .arch_names   = "llama",
    .description  = "Llama-family (Llama 1/2/3, TinyLlama, Mistral, Qwen2-dense, etc.)",
    .run          = run_llama_arch,
};

STRATUM_REGISTER_ARCH(stratum_arch_llama);
