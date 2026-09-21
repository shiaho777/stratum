/*
 * stratum_arch_moe.inc.c — MoE architecture (llama.cpp expert naming).
 *
 * Dense self-attention exactly like the llama arch; the FFN block is a
 * mixture of experts: ffn_gate_inp router [H, E] -> softmax -> top-k
 * experts from stacked tensors (ffn_gate_exps / ffn_up_exps / ffn_down_exps,
 * each [K_in, N_out, E]) -> weighted sum.
 *
 * THE streaming point (epic: extend "weights are a stream" to MoE): only
 * k of E expert slices are touched per token. Each expert slice IS a
 * contiguous byte range inside its tensor (slice e = e*K*N bytes), so
 * after the router fires we madvise WILLNEED those ranges — the next
 * layer's dense work then overlaps the expert pages faulting in. No page
 * is ever locked; residency stays under OS control (memory boundary).
 *
 * Greedy contract: top-k selection breaks ties by smaller expert id;
 * routing depends only on activations, never on timing or threads, so
 * output is deterministic across runs and configs.
 *
 * Registered as arch string "llama-moe"; auto-collected by the Makefile,
 * nothing else in stratum.c changes.
 */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "stratum_arch.h"
#include "stratum_linear.h"
#include "stratum_engine.h"
#include "stratum_tokenize.h"

static Gguf moe_g_gguf;
static StratumConfig moe_g_cfg;

typedef struct {
    const GgufTensor* attn_norm;
    const GgufTensor* attn_q;
    const GgufTensor* attn_k;
    const GgufTensor* attn_v;
    const GgufTensor* attn_output;
    const GgufTensor* ffn_norm;
    const GgufTensor* ffn_gate_inp;
    const GgufTensor* ffn_gate_exps;
    const GgufTensor* ffn_up_exps;
    const GgufTensor* ffn_down_exps;
} moe_BlockTensors;

static moe_BlockTensors* moe_g_blocks = NULL;
static const GgufTensor* moe_g_token_embd = NULL;
static const GgufTensor* moe_g_output_norm = NULL;
static const GgufTensor* moe_g_output_w = NULL;

/* ---- math primitives (mirror la_* bit-for-bit) ----------------------- */

static void moe_rmsnorm(const float* x, const float* gain, int N, float eps, float* y) {
    double ss = 0.0;
    for (int i = 0; i < N; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)N + (double)eps));
    for (int i = 0; i < N; i++) y[i] = x[i] * scale * gain[i];
}

static void moe_swiglu(const float* g, const float* u, int N, float* y) {
    stratum_swiglu(g, u, N, y);
}

static void moe_rope(float* x, int head_dim, int rope_dim, int position, float theta) {
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

static void moe_softmax_inplace(float* x, int N) {
    stratum_softmax_inplace(x, N);
}

/* ---- state ------------------------------------------------------------ */

static float* moe_g_x       = NULL;
static float* moe_g_x_resid = NULL;
static float* moe_g_xn      = NULL;
static float* moe_g_q_buf   = NULL;
static float* moe_g_k_buf   = NULL;
static float* moe_g_v_buf   = NULL;
static float* moe_g_attn_out= NULL;
/* expert scratch: [E] selection + per-expert FFN buffers */
static float* moe_g_router  = NULL;   /* E softmax weights */
static int    moe_g_sel[64];          /* selected expert ids (k <= 64) */
static float  moe_g_selw[64];         /* their router weights */
static float* moe_g_ff_g    = NULL;
static float* moe_g_ff_u    = NULL;
static float* moe_g_ff_a    = NULL;
static float* moe_g_ff_acc  = NULL;   /* weighted accumulator [H] */
static float* moe_g_logits  = NULL;

#define moe_MAX_KV 1024
static float* moe_g_K_cache = NULL;
static float* moe_g_V_cache = NULL;
static int    moe_g_kv_len  = 0;

/* multiseq/spec-verify state: per-slot KV when running independent streams */
#define moe_B_MAX 32
static float* moe_g_msK = NULL;
static float* moe_g_msV = NULL;
static int    moe_g_ms_maxkv = 0;
static int    moe_g_ms_B = 0;
static float* moe_gb_logits[moe_B_MAX];

/* ---- routing diagnostics (STRATUM_MOE_STATS=1) ------------------------ *
 * oracle-LRU idea borrowed from FreeToken: track how many DISTINCT experts
 * fire per step vs per layer, so prefetch policy can be tuned against a
 * measured working set instead of guesswork. */
static long moe_stat_steps = 0, moe_stat_expert_touches = 0;
static long moe_stat_distinct_last_step = 0;
static unsigned long long moe_stat_mask[256]; /* bit per expert this step */
static int moe_stats_on(void) { return getenv("STRATUM_MOE_STATS") != NULL; }

static void moe_stats_begin_step(void) {
    if (!moe_stats_on()) return;
    memset(moe_stat_mask, 0, sizeof(moe_stat_mask));
}
static void moe_stats_end_step(void) {
    if (!moe_stats_on()) return;
    long d = 0;
    for (int w = 0; w < 256; w++)
        for (int b = 0; b < 64; b++) if ((moe_stat_mask[w] >> b) & 1) d++;
    moe_stat_distinct_last_step = d;
    moe_stat_steps++;
    fprintf(stderr, "  [moe-stats] step %ld: distinct experts=%ld/%d "
            "(accum touches=%ld)\n",
            moe_stat_steps - 1, d, moe_g_cfg.n_layers * moe_g_cfg.n_experts_used,
            moe_stat_expert_touches);
}

/* ---- embedding (F16/F32/Q4_K/Q6_K, mirrors la_embed_lookup) ---------- */

static void moe_embed_lookup(int token_id, float* out) {
    int H = moe_g_cfg.n_embed;
    if (moe_g_token_embd->type == GGML_TYPE_Q4_K) {
        const block_q4_K* row = st_q4k_row_ptr(moe_g_token_embd, H, token_id);
        q4k_dequant_row_scalar(row, H, out);
    } else if (moe_g_token_embd->type == GGML_TYPE_Q6_K) {
        const block_q6_K* row = st_q6k_row_ptr(moe_g_token_embd, H, token_id);
        for (int i = 0; i < H / 256; i++)
            q6k_dequant_block_scalar(row + i, out + i * 256);
    } else if (moe_g_token_embd->type == GGML_TYPE_F16) {
        const uint16_t* raw = (const uint16_t*)(g_st.mmap_base + moe_g_token_embd->offset)
                            + (size_t)token_id * H;
        for (int i = 0; i < H; i++) out[i] = q4k_fp16_to_fp32(raw[i]);
    } else if (moe_g_token_embd->type == GGML_TYPE_F32) {
        const float* raw = (const float*)(g_st.mmap_base + moe_g_token_embd->offset)
                         + (size_t)token_id * H;
        memcpy(out, raw, sizeof(float) * H);
    } else {
        fprintf(stderr, "unsupported embed type %s\n",
                gguf_type_name((GgmlType)moe_g_token_embd->type));
        exit(2);
    }
}

/* ---- MoE FFN ----------------------------------------------------------
 * weight(xn) @ router -> softmax over E -> top-k by weight (ties: lower id)
 * -> for each expert: gate/up SwiGLU + down, accumulate weight*result.
 * Expert slice e starts at byte offset t->offset + e*slice_bytes inside
 * the stacked tensor (offset is absolute post-load). st_linear_dispatch
 * walks rows via g_st.mmap_base + offset, so a shifted-offset descriptor
 * carrying ONE slice's nbytes routes the kernel at the right slice —
 * zero copies, the stream reads only the selected experts' pages. */

/* bytes of ONE expert slice within a stacked [K_in, N_out, E] tensor */
static int64_t moe_slice_bytes(const GgufTensor* t, int K_in, int N_out) {
    int64_t one = (int64_t)K_in * N_out;
    if (t->nbytes <= 0 || t->nelem <= 0) return gguf_tensor_bytes((GgmlType)t->type, one);
    int64_t e_all = t->nelem / one;
    if (e_all < 1) e_all = 1;
    return (t->nbytes - 0) / e_all;
}

/* WILLNEED the byte range of expert `e` inside a stacked tensor. Advisory
 * only — pages stay reclaimable; this overlaps future faults with compute
 * and never wires anything (memory boundary). */
static void moe_prefetch_slice(const GgufTensor* t, int64_t slice_bytes,
                               int e) {
    if (!t) return;
    size_t off = t->offset + (size_t)e * (size_t)slice_bytes;
    madvise((void*)((const char*)g_st.mmap_base + off),
            (size_t)slice_bytes, MADV_WILLNEED);
}

static void moe_dispatch_slice(const GgufTensor* t, int e,
                               int64_t slice_bytes,
                               const float* x, float* y, int N_out, int K_in) {
    GgufTensor tmp = *t;
    tmp.offset  = t->offset + (uint64_t)e * (uint64_t)slice_bytes;
    tmp.nelem   = (int64_t)K_in * N_out;
    tmp.nbytes  = slice_bytes;
    if (st_linear_dispatch(&tmp, x, y, N_out, K_in) != 0)
        fprintf(stderr, "moe: dispatch failed for expert %d\n", e);
}

static void moe_forward_block(int li, int position) {
    moe_BlockTensors* b = &moe_g_blocks[li];
    int H  = moe_g_cfg.n_embed;
    int Hd = moe_g_cfg.head_dim;
    int Nq = moe_g_cfg.n_q_heads;
    int Nk = moe_g_cfg.n_kv_heads;
    int Ff = moe_g_cfg.n_ff;
    int E  = moe_g_cfg.n_experts;
    int K_ = moe_g_cfg.n_experts_used;

    memcpy(moe_g_x_resid, moe_g_x, sizeof(float) * H);
    {
        const float* gain = st_f32_tensor_ptr(b->attn_norm);
        moe_rmsnorm(moe_g_x, gain, H, moe_g_cfg.rms_eps, moe_g_xn);
    }

    /* dense attention — same as llama arch */
    st_linear_dispatch(b->attn_q, moe_g_xn, moe_g_q_buf, Nq * Hd, H);
    st_linear_dispatch(b->attn_k, moe_g_xn, moe_g_k_buf, Nk * Hd, H);
    st_linear_dispatch(b->attn_v, moe_g_xn, moe_g_v_buf, Nk * Hd, H);

    for (int h = 0; h < Nq; h++)
        moe_rope(moe_g_q_buf + h * Hd, Hd, moe_g_cfg.rope_dim,
                 position, moe_g_cfg.rope_theta);
    for (int h = 0; h < Nk; h++)
        moe_rope(moe_g_k_buf + h * Hd, Hd, moe_g_cfg.rope_dim,
                 position, moe_g_cfg.rope_theta);

    int kv_len_now = moe_g_kv_len + 1;
    {
        size_t per_layer = (size_t)moe_MAX_KV * Nk * Hd;
        size_t off = (size_t)li * per_layer
                   + (size_t)moe_g_kv_len * Nk * Hd;
        memcpy(moe_g_K_cache + off, moe_g_k_buf, sizeof(float) * Nk * Hd);
        memcpy(moe_g_V_cache + off, moe_g_v_buf, sizeof(float) * Nk * Hd);
    }

    float scale = 1.0f / sqrtf((float)Hd);
    for (int h = 0; h < Nq; h++) {
        int kv_h = h * Nk / Nq;
        const float* qh = moe_g_q_buf + h * Hd;
        size_t per_layer = (size_t)moe_MAX_KV * Nk * Hd;
        const float* K_layer = moe_g_K_cache + (size_t)li * per_layer;
        const float* V_layer = moe_g_V_cache + (size_t)li * per_layer;

        float logits[moe_MAX_KV];
        for (int t = 0; t < kv_len_now; t++) {
            const float* kt = K_layer + (size_t)t * Nk * Hd + kv_h * Hd;
            float dot = 0.0f;
            for (int d = 0; d < Hd; d++) dot += qh[d] * kt[d];
            logits[t] = dot * scale;
        }
        moe_softmax_inplace(logits, kv_len_now);

        float* head_out = moe_g_attn_out + h * Hd;
        memset(head_out, 0, sizeof(float) * Hd);
        for (int t = 0; t < kv_len_now; t++) {
            const float* vt = V_layer + (size_t)t * Nk * Hd + kv_h * Hd;
            float p = logits[t];
            for (int d = 0; d < Hd; d++) head_out[d] += p * vt[d];
        }
    }

    st_linear_dispatch(b->attn_output, moe_g_attn_out, moe_g_x, H, Nq * Hd);
    for (int i = 0; i < H; i++) moe_g_x[i] += moe_g_x_resid[i];

    /* ---- MoE FFN ---- */
    memcpy(moe_g_x_resid, moe_g_x, sizeof(float) * H);
    {
        const float* gain = st_f32_tensor_ptr(b->ffn_norm);
        moe_rmsnorm(moe_g_x, gain, H, moe_g_cfg.rms_eps, moe_g_xn);
    }

    /* router: xn @ [H,E] -> softmax */
    st_linear_dispatch(b->ffn_gate_inp, moe_g_xn, moe_g_router, E, H);
    moe_softmax_inplace(moe_g_router, E);

    /* top-k by weight, ties broken toward the LOWER expert id (stable):
     * scan thresholds in descending weight order; equal weights pick the
     * smallest index first because we only replace a selection on strictly
     * greater weight or equal-weight-later-index eviction below. */
    for (int s = 0; s < K_; s++) { moe_g_sel[s] = -1; moe_g_selw[s] = -1.0f; }
    for (int e = 0; e < E; e++) {
        float w = moe_g_router[e];
        for (int s = 0; s < K_; s++) {
            if (w > moe_g_selw[s]) {
                for (int t2 = K_ - 1; t2 > s; t2--) {
                    moe_g_sel[t2] = moe_g_sel[t2 - 1];
                    moe_g_selw[t2] = moe_g_selw[t2 - 1];
                }
                moe_g_sel[s] = e;
                moe_g_selw[s] = w;
                break;
            }
        }
    }

    /* stream prefetch hint FIRST (the whole point): each selected expert's
     * gate/up/down slices are contiguous ranges — nudge the kernel to read
     * them ahead while dense work of later layers proceeds. Reclaimable
     * page-cache hints only; no locking (memory boundary). */
    {
        int64_t g_sb = moe_slice_bytes(b->ffn_gate_exps, H, Ff);
        int64_t u_sb = moe_slice_bytes(b->ffn_up_exps, H, Ff);
        int64_t d_sb = moe_slice_bytes(b->ffn_down_exps, Ff, H);
        for (int s = 0; s < K_; s++) {
            int e = moe_g_sel[s];
            if (e < 0) continue;
            moe_prefetch_slice(b->ffn_gate_exps, g_sb, e);
            moe_prefetch_slice(b->ffn_up_exps, u_sb, e);
            moe_prefetch_slice(b->ffn_down_exps, d_sb, e);
            if (moe_stats_on()) moe_stat_expert_touches++;
        }
    }

    /* accumulate weighted expert outputs */
    memset(moe_g_ff_acc, 0, sizeof(float) * H);
    for (int s = 0; s < K_; s++) {
        int e = moe_g_sel[s];
        if (e < 0) continue;
        float w = moe_g_selw[s];
        static float ff_out_tmp[8192];
        moe_dispatch_slice(b->ffn_gate_exps, e,
                           moe_slice_bytes(b->ffn_gate_exps, H, Ff),
                           moe_g_xn, moe_g_ff_g, Ff, H);
        moe_dispatch_slice(b->ffn_up_exps, e,
                           moe_slice_bytes(b->ffn_up_exps, H, Ff),
                           moe_g_xn, moe_g_ff_u, Ff, H);
        moe_swiglu(moe_g_ff_g, moe_g_ff_u, Ff, moe_g_ff_a);
        moe_dispatch_slice(b->ffn_down_exps, e,
                           moe_slice_bytes(b->ffn_down_exps, Ff, H),
                           moe_g_ff_a, ff_out_tmp, H, Ff);
        for (int i = 0; i < H; i++) moe_g_ff_acc[i] += w * ff_out_tmp[i];
    }
    for (int i = 0; i < H; i++) moe_g_x[i] = moe_g_x_resid[i] + moe_g_ff_acc[i];

    /* stats: mark this step's fired experts in the diagnostic mask */
    if (moe_stats_on())
        for (int s = 0; s < K_; s++) {
            int e = moe_g_sel[s] + li * moe_g_cfg.n_experts;
            if (moe_g_sel[s] >= 0 && e / 64 < 256)
                moe_stat_mask[e / 64] |= 1ULL << (e % 64);
        }
}

/* ---- single-token forward --------------------------------------------- */

static void moe_forward_one_token(int token_id, int position) {
    int H = moe_g_cfg.n_embed;
    int V = moe_g_cfg.vocab_size;

    moe_embed_lookup(token_id, moe_g_x);
    moe_stats_begin_step();

    for (int li = 0; li < moe_g_cfg.n_layers; li++)
        moe_forward_block(li, position);

    {
        const float* gain = st_f32_tensor_ptr(moe_g_output_norm);
        moe_rmsnorm(moe_g_x, gain, H, moe_g_cfg.rms_eps, moe_g_xn);
    }

    const GgufTensor* lm = moe_g_output_w ? moe_g_output_w : moe_g_token_embd;
    if (st_linear_dispatch(lm, moe_g_xn, moe_g_logits, V, H) != 0) {
        fprintf(stderr, "moe: lm_head dispatch failed\n");
        exit(2);
    }

    moe_stats_end_step();
    moe_g_kv_len++;
}

/* ---- batched / multiseq forward ----------------------------------------- *
 * moe_forward_b(tokens, pos, kvlen, B, shared_kv)
 *
 *   shared_kv=0 — B independent streams. Each slot owns a KV slice in
 *     moe_g_msK/msV laid out [layer][slot][ms_maxkv][Nk*Hd]; slot s writes
 *     its KV at position kvlen[s] and attends kvlen[s]+1 positions.
 *   shared_kv=1 — verify semantics: all slots share moe_g_K/V_cache; slot
 *     s writes at kvlen[0]+s and attends kvlen[0]+s+1 positions, so the
 *     chained drafts see each other's KV exactly like sequential steps.
 *
 * Expert-grouped batching: the router is a single B-stream matmul per
 * layer; streams selecting the same expert are matmul'd together through
 * st_linear_multix, so each expert slice's bytes are streamed once per
 * layer regardless of member count. Each stream's weighted accumulation
 * runs strictly in its own top-k rank order, preserving the exact FP
 * summation order of the single-stream path (greedy-token identical). */

static void moe_multix_slice(const GgufTensor* t, int e,
                             int64_t slice_bytes,
                             const float* const* xs, float* const* ys,
                             int B, int N_out, int K_in) {
    GgufTensor tmp = *t;
    tmp.offset  = t->offset + (uint64_t)e * (uint64_t)slice_bytes;
    tmp.nelem   = (int64_t)K_in * N_out;
    tmp.nbytes  = slice_bytes;
    st_linear_multix(&tmp, xs, ys, B, N_out, K_in);
}

static int moe_forward_b(const int* tokens, const int* pos, const int* kvlen,
                         int B, int shared_kv) {
    int H  = moe_g_cfg.n_embed;
    int Hd = moe_g_cfg.head_dim;
    int Nq = moe_g_cfg.n_q_heads;
    int Nk = moe_g_cfg.n_kv_heads;
    int Ff = moe_g_cfg.n_ff;
    int V  = moe_g_cfg.vocab_size;
    int E  = moe_g_cfg.n_experts;
    int K_ = moe_g_cfg.n_experts_used;
    if (B < 1 || B > moe_B_MAX || K_ < 1) return -1;

    static float* x[moe_B_MAX];  static float* xr[moe_B_MAX];
    static float* xn[moe_B_MAX];
    static float* qb[moe_B_MAX]; static float* kb[moe_B_MAX];
    static float* vb[moe_B_MAX];
    static float* ao[moe_B_MAX]; static float* ap[moe_B_MAX];
    static float* rt[moe_B_MAX]; static float* ac[moe_B_MAX];
    /* expert member staging (indexed by member slot j) + ordered outputs */
    static float* mg[moe_B_MAX]; static float* mu[moe_B_MAX];
    static float* ma[moe_B_MAX]; static float* mo[moe_B_MAX];
    static float* ffo = NULL;    static size_t ffo_cap = 0;   /* [B*K_][H] */
    static int*   sel_all = NULL; static float* selw_all = NULL;
    static size_t sel_cap = 0;
    /* per-expert member lists: memb[e*B + j] = s*K_+k — dynamic so E>256
     * (DeepSeek-class) never silently drops a routing */
    static int*   memb = NULL; static int* mcnt = NULL;
    static size_t memb_cap = 0, mcnt_cap = 0;
    static int alloc_done = 0;
    if (!alloc_done) {
        for (int s = 0; s < moe_B_MAX; s++) {
            x[s]  = calloc(H, 4);     xr[s] = calloc(H, 4);
            xn[s] = calloc(H, 4);
            qb[s] = calloc(Nq*Hd, 4); kb[s] = calloc(Nk*Hd, 4);
            vb[s] = calloc(Nk*Hd, 4);
            ao[s] = calloc(Nq*Hd, 4); ap[s] = calloc(H, 4);
            rt[s] = calloc(E, 4);     ac[s] = calloc(H, 4);
            mg[s] = calloc(Ff, 4);    mu[s] = calloc(Ff, 4);
            ma[s] = calloc(Ff, 4);    mo[s] = calloc(H, 4);
            if (!moe_gb_logits[s]) moe_gb_logits[s] = calloc(V, 4);
            if (!x[s]||!xr[s]||!xn[s]||!qb[s]||!kb[s]||!vb[s]||!ao[s]||
                !ap[s]||!rt[s]||!ac[s]||!mg[s]||!mu[s]||!ma[s]||!mo[s]||
                !moe_gb_logits[s]) return -1;
        }
        alloc_done = 1;
    }
    size_t need_sel = (size_t)B * K_;
    if (need_sel > sel_cap) {
        free(sel_all); free(selw_all);
        sel_all  = malloc(need_sel * 4);
        selw_all = malloc(need_sel * 4);
        sel_cap  = (sel_all && selw_all) ? need_sel : 0;
        if (!sel_cap) return -1;
    }
    size_t need_ffo = need_sel * (size_t)H;
    if (need_ffo > ffo_cap) {
        free(ffo); ffo = malloc(need_ffo * 4);
        ffo_cap = ffo ? need_ffo : 0;
        if (!ffo_cap) return -1;
    }
    if ((size_t)E * B > memb_cap) {
        free(memb); memb = malloc((size_t)E * B * 4);
        memb_cap = memb ? (size_t)E * B : 0;
        if (!memb_cap) return -1;
    }
    if ((size_t)E > mcnt_cap) {
        free(mcnt); mcnt = malloc((size_t)E * 4);
        mcnt_cap = mcnt ? (size_t)E : 0;
        if (!mcnt_cap) return -1;
    }

    for (int s = 0; s < B; s++) moe_embed_lookup(tokens[s], x[s]);

    const float scale = 1.0f / sqrtf((float)Hd);
    size_t kv_seqstride = (size_t)moe_g_ms_maxkv * Nk * Hd;
    size_t kv_laystride = (size_t)B * kv_seqstride;
    size_t sh_laystride = (size_t)moe_MAX_KV * Nk * Hd;

    const float* cxn[moe_B_MAX]; float* cqb[moe_B_MAX];
    float* ckb[moe_B_MAX]; float* cvb[moe_B_MAX];
    float* cao[moe_B_MAX]; float* cap[moe_B_MAX];
    float* crt[moe_B_MAX]; float* clog[moe_B_MAX];

    moe_stats_begin_step();
    for (int li = 0; li < moe_g_cfg.n_layers; li++) {
        moe_BlockTensors* b = &moe_g_blocks[li];
        for (int s = 0; s < B; s++) {
            memcpy(xr[s], x[s], sizeof(float) * H);
            const float* gain = st_f32_tensor_ptr(b->attn_norm);
            moe_rmsnorm(x[s], gain, H, moe_g_cfg.rms_eps, xn[s]);
            cxn[s] = xn[s];
        }
        for (int s = 0; s < B; s++) {
            cqb[s] = qb[s]; ckb[s] = kb[s]; cvb[s] = vb[s];
        }
        st_linear_multix(b->attn_q, cxn, cqb, B, Nq * Hd, H);
        st_linear_multix(b->attn_k, cxn, ckb, B, Nk * Hd, H);
        st_linear_multix(b->attn_v, cxn, cvb, B, Nk * Hd, H);

        for (int s = 0; s < B; s++) {
            for (int h = 0; h < Nq; h++)
                moe_rope(qb[s] + h * Hd, Hd, moe_g_cfg.rope_dim,
                         pos[s], moe_g_cfg.rope_theta);
            for (int h = 0; h < Nk; h++)
                moe_rope(kb[s] + h * Hd, Hd, moe_g_cfg.rope_dim,
                         pos[s], moe_g_cfg.rope_theta);
            if (shared_kv) {
                size_t off = (size_t)li * sh_laystride
                           + (size_t)(kvlen[0] + s) * Nk * Hd;
                memcpy(moe_g_K_cache + off, kb[s], sizeof(float) * Nk * Hd);
                memcpy(moe_g_V_cache + off, vb[s], sizeof(float) * Nk * Hd);
            } else {
                float* Kb = moe_g_msK + (size_t)li * kv_laystride
                          + (size_t)s * kv_seqstride;
                float* Vb = moe_g_msV + (size_t)li * kv_laystride
                          + (size_t)s * kv_seqstride;
                memcpy(Kb + (size_t)kvlen[s] * Nk * Hd, kb[s],
                       sizeof(float) * Nk * Hd);
                memcpy(Vb + (size_t)kvlen[s] * Nk * Hd, vb[s],
                       sizeof(float) * Nk * Hd);
            }
        }

        for (int s = 0; s < B; s++) {
            int klen = shared_kv ? kvlen[0] + s + 1 : kvlen[s] + 1;
            const float* K_layer; const float* V_layer;
            if (shared_kv) {
                K_layer = moe_g_K_cache + (size_t)li * sh_laystride;
                V_layer = moe_g_V_cache + (size_t)li * sh_laystride;
            } else {
                K_layer = moe_g_msK + (size_t)li * kv_laystride
                        + (size_t)s * kv_seqstride;
                V_layer = moe_g_msV + (size_t)li * kv_laystride
                        + (size_t)s * kv_seqstride;
            }
            for (int h = 0; h < Nq; h++) {
                int kv_h = h * Nk / Nq;
                const float* qh = qb[s] + h * Hd;
                float logits[moe_MAX_KV];
                for (int t = 0; t < klen; t++) {
                    const float* kt = K_layer + (size_t)t * Nk * Hd
                                    + kv_h * Hd;
                    float dot = 0.0f;
                    for (int d = 0; d < Hd; d++) dot += qh[d] * kt[d];
                    logits[t] = dot * scale;
                }
                moe_softmax_inplace(logits, klen);
                float* hd = ao[s] + h * Hd;
                memset(hd, 0, sizeof(float) * Hd);
                for (int t = 0; t < klen; t++) {
                    const float* vt = V_layer + (size_t)t * Nk * Hd
                                    + kv_h * Hd;
                    float p = logits[t];
                    for (int d = 0; d < Hd; d++) hd[d] += p * vt[d];
                }
            }
        }
        for (int s = 0; s < B; s++) { cao[s] = ao[s]; cap[s] = ap[s]; }
        st_linear_multix(b->attn_output, (const float* const*)cao, cap,
                         B, H, Nq * Hd);
        for (int s = 0; s < B; s++)
            for (int i = 0; i < H; i++) x[s][i] = xr[s][i] + ap[s][i];

        /* ---- MoE FFN, expert-grouped batching ---- */
        for (int s = 0; s < B; s++) {
            memcpy(xr[s], x[s], sizeof(float) * H);
            const float* gain = st_f32_tensor_ptr(b->ffn_norm);
            moe_rmsnorm(x[s], gain, H, moe_g_cfg.rms_eps, xn[s]);
            crt[s] = rt[s];
        }
        st_linear_multix(b->ffn_gate_inp, cxn, crt, B, E, H);

        /* per-stream softmax + top-k (same stable lower-id tie-break as
         * the single-stream path) */
        memset(mcnt, 0, sizeof(int) * (size_t)E);
        int* sel = sel_all; float* selw = selw_all;
        for (int s = 0; s < B; s++) {
            int* sel_s = sel + s * K_; float* selw_s = selw + s * K_;
            moe_softmax_inplace(rt[s], E);
            for (int k = 0; k < K_; k++) { sel_s[k] = -1; selw_s[k] = -1.0f; }
            for (int e = 0; e < E; e++) {
                float w = rt[s][e];
                for (int k = 0; k < K_; k++) {
                    if (w > selw_s[k]) {
                        for (int t2 = K_ - 1; t2 > k; t2--) {
                            sel_s[t2] = sel_s[t2 - 1];
                            selw_s[t2] = selw_s[t2 - 1];
                        }
                        sel_s[k] = e; selw_s[k] = w;
                        break;
                    }
                }
            }
            for (int k = 0; k < K_; k++)
                if (sel_s[k] >= 0)
                    memb[(size_t)sel_s[k] * B + mcnt[sel_s[k]]++] = s * K_ + k;
        }

        int64_t g_sb = moe_slice_bytes(b->ffn_gate_exps, H, Ff);
        int64_t u_sb = moe_slice_bytes(b->ffn_up_exps, H, Ff);
        int64_t d_sb = moe_slice_bytes(b->ffn_down_exps, Ff, H);
        /* prefetch each selected expert's slices once (reclaimable hints) */
        for (int e = 0; e < E; e++) {
            if (!mcnt[e]) continue;
            moe_prefetch_slice(b->ffn_gate_exps, g_sb, e);
            moe_prefetch_slice(b->ffn_up_exps, u_sb, e);
            moe_prefetch_slice(b->ffn_down_exps, d_sb, e);
            if (moe_stats_on()) moe_stat_expert_touches += mcnt[e];
        }

        /* per expert: batch its member streams through one multix */
        for (int e = 0; e < E; e++) {
            int M = mcnt[e];
            if (!M) continue;
            const float* xe[moe_B_MAX]; float* og[moe_B_MAX];
            float* ou[moe_B_MAX];     float* od[moe_B_MAX];
            const float* oa[moe_B_MAX];
            for (int j = 0; j < M; j++) {
                int s = memb[(size_t)e * B + j] / K_;
                xe[j] = xn[s]; og[j] = mg[j]; ou[j] = mu[j];
                oa[j] = ma[j];    od[j] = mo[j];
            }
            moe_multix_slice(b->ffn_gate_exps, e, g_sb, xe, og, M, Ff, H);
            moe_multix_slice(b->ffn_up_exps,   e, u_sb, xe, ou, M, Ff, H);
            for (int j = 0; j < M; j++)
                moe_swiglu(mg[j], mu[j], Ff, ma[j]);
            moe_multix_slice(b->ffn_down_exps, e, d_sb, oa, od, M, H, Ff);
            for (int j = 0; j < M; j++)
                memcpy(ffo + (size_t)memb[(size_t)e * B + j] * H, mo[j],
                       sizeof(float) * H);
            if (moe_stats_on())
                for (int j = 0; j < M; j++) {
                    int ee = e + li * E;
                    if (ee / 64 < 256) moe_stat_mask[ee / 64] |= 1ULL << (ee % 64);
                }
        }

        /* accumulate in each stream's own top-k rank order (FP order ==
         * single-stream path) */
        for (int s = 0; s < B; s++) {
            memset(ac[s], 0, sizeof(float) * H);
            const int* sel_s = sel + s * K_;
            const float* selw_s = selw + s * K_;
            for (int k = 0; k < K_; k++) {
                if (sel_s[k] < 0) continue;
                const float* o = ffo + (size_t)(s * K_ + k) * H;
                float w = selw_s[k];
                for (int i = 0; i < H; i++) ac[s][i] += w * o[i];
            }
            for (int i = 0; i < H; i++) x[s][i] = xr[s][i] + ac[s][i];
        }
    }
    moe_stats_end_step();

    {
        const float* gain = st_f32_tensor_ptr(moe_g_output_norm);
        for (int s = 0; s < B; s++) {
            moe_rmsnorm(x[s], gain, H, moe_g_cfg.rms_eps, xn[s]);
            clog[s] = moe_gb_logits[s];
        }
    }
    const GgufTensor* lm = moe_g_output_w ? moe_g_output_w : moe_g_token_embd;
    st_linear_multix(lm, cxn, clog, B, V, H);
    return 0;
}

static int moe_forward_multiseq(const int* tokens, const int* pos,
                                const int* kvlen, int B) {
    return moe_forward_b(tokens, pos, kvlen, B, 0);
}

static int moe_forward_batch(const int* tokens, const int* pos,
                             const int* kvlen, int B) {
    return moe_forward_b(tokens, pos, kvlen, B, 1);
}

/* ---- discovery + allocation ------------------------------------------- */

static int moe_discover_blocks(void) {
    if (moe_g_cfg.n_experts <= 0 || moe_g_cfg.n_experts_used <= 0) {
        fprintf(stderr, "moe: expert_count/expert_used_count missing or zero "
                        "(got %d/%d)\n",
                moe_g_cfg.n_experts, moe_g_cfg.n_experts_used);
        return -1;
    }
    if (moe_g_cfg.n_experts_used > 64) {
        fprintf(stderr, "moe: expert_used_count %d exceeds scratch cap 64\n",
                moe_g_cfg.n_experts_used);
        return -1;
    }
    moe_g_blocks = (moe_BlockTensors*)calloc(moe_g_cfg.n_layers,
                                             sizeof(moe_BlockTensors));
    if (!moe_g_blocks) return -1;
    char nm[128];

#define FIND(field, name)                                                  \
    do {                                                                   \
        snprintf(nm, sizeof nm, "blk.%d." name ".weight", li);             \
        moe_g_blocks[li].field = gguf_find_tensor(&moe_g_gguf, nm);        \
        if (!moe_g_blocks[li].field) {                                     \
            fprintf(stderr, "missing %s\n", nm);                           \
            return -1;                                                     \
        }                                                                  \
    } while (0)

    for (int li = 0; li < moe_g_cfg.n_layers; li++) {
        FIND(attn_norm,      "attn_norm");
        FIND(attn_q,         "attn_q");
        FIND(attn_k,         "attn_k");
        FIND(attn_v,         "attn_v");
        FIND(attn_output,    "attn_output");
        FIND(ffn_norm,       "ffn_norm");
        FIND(ffn_gate_inp,   "ffn_gate_inp");
        FIND(ffn_gate_exps,  "ffn_gate_exps");
        FIND(ffn_up_exps,    "ffn_up_exps");
        FIND(ffn_down_exps,  "ffn_down_exps");

    }
#undef FIND
    moe_g_token_embd  = gguf_find_tensor(&moe_g_gguf, "token_embd.weight");
    moe_g_output_norm = gguf_find_tensor(&moe_g_gguf, "output_norm.weight");
    moe_g_output_w    = gguf_find_tensor(&moe_g_gguf, "output.weight");
    if (!moe_g_token_embd || !moe_g_output_norm) {
        fprintf(stderr, "missing token_embd or output_norm\n");
        return -1;
    }
    return 0;
}

static int moe_allocate_state(void) {
    int H  = moe_g_cfg.n_embed;
    int Hd = moe_g_cfg.head_dim;
    int Nq = moe_g_cfg.n_q_heads;
    int Nk = moe_g_cfg.n_kv_heads;
    int Ff = moe_g_cfg.n_ff;
    int V  = moe_g_cfg.vocab_size;
    int L  = moe_g_cfg.n_layers;

    moe_g_x        = (float*)calloc(H, sizeof(float));
    moe_g_x_resid  = (float*)calloc(H, sizeof(float));
    moe_g_xn       = (float*)calloc(H, sizeof(float));
    moe_g_q_buf    = (float*)calloc(Nq * Hd, sizeof(float));
    moe_g_k_buf    = (float*)calloc(Nk * Hd, sizeof(float));
    moe_g_v_buf    = (float*)calloc(Nk * Hd, sizeof(float));
    moe_g_attn_out = (float*)calloc(Nq * Hd, sizeof(float));
    moe_g_router   = (float*)calloc(moe_g_cfg.n_experts, sizeof(float));
    moe_g_ff_g     = (float*)calloc(Ff, sizeof(float));
    moe_g_ff_u     = (float*)calloc(Ff, sizeof(float));
    moe_g_ff_a     = (float*)calloc(Ff, sizeof(float));
    moe_g_ff_acc   = (float*)calloc(H, sizeof(float));
    moe_g_logits   = (float*)calloc(V, sizeof(float));

    size_t kv_floats = (size_t)L * moe_MAX_KV * Nk * Hd;
    moe_g_K_cache = (float*)calloc(kv_floats, sizeof(float));
    moe_g_V_cache = (float*)calloc(kv_floats, sizeof(float));

    if (!moe_g_x || !moe_g_K_cache || !moe_g_logits || !moe_g_router ||
        !moe_g_ff_acc) return -1;

    fprintf(stderr, "  KV cache: %.1f MB (anonymous)\n",
            (double)kv_floats * 2 * 4 / (1024.0 * 1024.0));
    fprintf(stderr,
            "  MoE: E=%d used=%d per layer — streaming slices via WILLNEED\n",
            moe_g_cfg.n_experts, moe_g_cfg.n_experts_used);
    return 0;
}

/* ---- entry point -------------------------------------------------------
 * Minimal shape of the llama arch entry: dense prefill + greedy decode.
 * No spec/multiseq yet — Phase-3 optimization, correctness first. */
int run_moe_arch(int argc, char** argv) {
    stratum_enforce_boundaries();
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]\n",
                argv[0]);
        return 1;
    }
    fprintf(stderr, "== stratum — MoE architecture (streaming experts) ==\n");
    fprintf(stderr, "  model : %s\n", argv[1]);

    if (gguf_open(argv[1], &moe_g_gguf) != 0) return 1;
    fprintf(stderr, "  GGUF v%u, %llu tensors, body @ %llu\n",
            moe_g_gguf.version,
            (unsigned long long)moe_g_gguf.n_tensors,
            (unsigned long long)moe_g_gguf.body_offset);

    stratum_linear_init(moe_g_gguf.mmap_base, moe_g_gguf.mmap_size);
    stratum_engine_init(moe_g_gguf.mmap_size);
    if (stratum_load_config(&moe_g_gguf, &moe_g_cfg) != 0) return 1;
    fprintf(stderr, "\n");
    stratum_print_config(&moe_g_cfg);
    fprintf(stderr, "\n");

    if (moe_discover_blocks() != 0) return 1;
    if (moe_allocate_state()  != 0) return 1;

    {
        int ncpu = 0; size_t l = sizeof(ncpu);
        if (sysctlbyname("hw.physicalcpu", &ncpu, &l, NULL, 0) != 0 || ncpu < 1)
            ncpu = 8;
        const char* env_nc = getenv("STRATUM_NCHUNKS");
        g_st.nchunks = env_nc ? atoi(env_nc) : ncpu;
        if (g_st.nchunks < 1) g_st.nchunks = 1;
        fprintf(stderr, "  parallel matmul: %d chunks (%d physical cores)\n",
                g_st.nchunks, ncpu);
    }
    madvise((void*)g_st.mmap_base, g_st.mmap_size, MADV_WILLNEED);

    static StratumVocab moe_vocab;
    stratum_vocab_init(&moe_g_gguf, moe_g_gguf.mmap_base, &moe_vocab);

    int n_gen = (argc > 2) ? atoi(argv[2]) : 4;
    int prompt[2048];
    int n_prompt = 0;
    for (int i = 3; i < argc && n_prompt < 2048; i++)
        prompt[n_prompt++] = atoi(argv[i]);
    if (n_prompt == 0) prompt[n_prompt++] = 1;

    int ms_B = 0;
    { const char* e = getenv("STRATUM_MULTISEQ"); if (e) ms_B = atoi(e); }
    if (ms_B >= 1) {
        if (ms_B > moe_B_MAX) ms_B = moe_B_MAX;
        moe_g_ms_maxkv = n_prompt + n_gen + 2;
        if (moe_g_ms_maxkv > moe_MAX_KV) moe_g_ms_maxkv = moe_MAX_KV;
        size_t cells = (size_t)moe_g_cfg.n_layers * ms_B * moe_g_ms_maxkv
                     * moe_g_cfg.n_kv_heads * moe_g_cfg.head_dim;
        moe_g_msK = (float*)calloc(cells, sizeof(float));
        moe_g_msV = (float*)calloc(cells, sizeof(float));
        moe_g_ms_B = ms_B;
        if (!moe_g_msK || !moe_g_msV) {
            fprintf(stderr, "  multiseq KV alloc failed\n"); return 1;
        }
        fprintf(stderr, "  MULTISEQ: %d independent streams, KV %.1f MB\n",
                ms_B, (double)cells*2*4/(1024.0*1024.0));
        int kvlen[moe_B_MAX]={0}, pos[moe_B_MAX], tok[moe_B_MAX];
        int nxt[moe_B_MAX];
        struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
        for (int p = 0; p < n_prompt; p++) {
            for (int s=0;s<ms_B;s++){tok[s]=prompt[p];pos[s]=p;}
            if (moe_forward_multiseq(tok,pos,kvlen,ms_B)!=0) return 1;
            for (int s=0;s<ms_B;s++) kvlen[s]++;
        }
        for (int s=0;s<ms_B;s++)
            nxt[s]=stratum_argmax(moe_gb_logits[s],moe_g_cfg.vocab_size);
        clock_gettime(CLOCK_MONOTONIC,&b);
        double pf=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
        fprintf(stderr,"  multiseq prefill %.3fs; stream0 first tok=%d\n",
                pf, nxt[0]);
        clock_gettime(CLOCK_MONOTONIC,&a);
        for (int g=0;g<n_gen;g++){
            for (int s=0;s<ms_B;s++){tok[s]=nxt[s];pos[s]=n_prompt+g;}
            if (moe_forward_multiseq(tok,pos,kvlen,ms_B)!=0) return 1;
            for (int s=0;s<ms_B;s++){
                kvlen[s]++;
                nxt[s]=stratum_argmax(moe_gb_logits[s],moe_g_cfg.vocab_size);
            }
            if (getenv("STRATUM_MS_VERIFY")) {
                int V=moe_g_cfg.vocab_size;
                for (int s=1;s<ms_B;s++) {
                    if (nxt[s]!=nxt[0])
                        fprintf(stderr,"  [VERIFY] MISMATCH g=%d stream%d tok=%d != stream0 tok=%d\n",g,s,nxt[s],nxt[0]);
                    if (memcmp(moe_gb_logits[s],moe_gb_logits[0],(size_t)V*4)!=0){
                        int diffs=0;
                        for(int i=0;i<V;i++) if(moe_gb_logits[s][i]!=moe_gb_logits[0][i]) diffs++;
                        fprintf(stderr,"  [VERIFY] g=%d stream%d logits differ in %d/%d dims\n",g,s,diffs,V);
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
    for (int t = 0; t < n_prompt; t++) {
        moe_forward_one_token(prompt[t], position++);
        last_tok = prompt[t];
    }
    int next_tok = stratum_argmax(moe_g_logits, moe_g_cfg.vocab_size);
    fprintf(stderr, "  after prefill, sampled = %d  (logit=%g)\n",
            next_tok, moe_g_logits[next_tok]);

    int spec_k = 0;
    { const char* e = getenv("STRATUM_NGRAM_SPEC"); if (e) spec_k = atoi(e); }
    if (spec_k > moe_B_MAX - 1) spec_k = moe_B_MAX - 1;

    if (spec_k >= 1) {
        /* n-gram (prompt-lookup) speculative decoding — same acceptance
         * rule as llama: emit only tokens equal to the model's argmax at
         * their parent, so the sequence is greedy-identical. Drafts are
         * verified by one batched forward over the shared KV cache. */
        int* hist = (int*)malloc(sizeof(int)*(n_prompt+n_gen+moe_B_MAX+2));
        int hlen = 0;
        for (int i = 0; i < n_prompt; i++) hist[hlen++] = prompt[i];
        int spec_calls = 0, spec_accepts = 0;
        int g = 0;
        while (g < n_gen) {
            int btok[moe_B_MAX], bpos[moe_B_MAX];
            btok[0] = next_tok; bpos[0] = position;
            int B = 1;
            int hyp[moe_B_MAX]; hyp[0] = next_tok; int hn = 1;
            while (B <= spec_k) {
                int prop = -1;
                for (int ng = 3; ng >= 2 && prop < 0; ng--) {
                    int total = hlen + hn;
                    if (total < ng + 1) continue;
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
                hyp[hn++] = prop; B++;
            }

            if (B == 1) {
                /* no draft: single-token step (KV-full falls here too) */
                moe_forward_one_token(next_tok, position);
                int argm = stratum_argmax(moe_g_logits, moe_g_cfg.vocab_size);
                stratum_logits_dump_record(moe_g_logits, moe_g_cfg.vocab_size, argm);
                fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  logit=%g\n",
                        g, next_tok, argm, moe_g_logits[argm]);
                if (moe_vocab.available) {
                    char tok_text[256];
                    stratum_decode_token(&moe_vocab, argm, tok_text, sizeof(tok_text));
                    fprintf(stdout, "%s", tok_text); fflush(stdout);
                }
                hist[hlen++] = next_tok;
                position++; g++;
                next_tok = argm;
                continue;
            }

            /* batched verify over shared KV: slot s attends 0..kv_len+s */
            if (moe_g_kv_len + B >= moe_MAX_KV) {
                /* KV full: fall back to single-token step */
                moe_forward_one_token(next_tok, position);
                int argm = stratum_argmax(moe_g_logits, moe_g_cfg.vocab_size);
                hist[hlen++] = next_tok;
                position++; g++;
                next_tok = argm;
                continue;
            }
            spec_calls++;
            int bkv[moe_B_MAX];
            for (int s = 0; s < B; s++) bkv[s] = moe_g_kv_len;
            if (moe_forward_batch(btok, bpos, bkv, B) != 0) return 1;

            /* accept the longest argmax-consistent prefix */
            int accepted = 0;
            for (int s = 0; s + 1 < B; s++) {
                int argm = stratum_argmax(moe_gb_logits[s], moe_g_cfg.vocab_size);
                if (argm == btok[s + 1]) accepted++;
                else break;
            }
            int emit_n = accepted + 1;
            spec_accepts += accepted;
            for (int s = 0; s < emit_n; s++) {
                hist[hlen++] = btok[s];
                if (moe_vocab.available) {
                    char tok_text[256];
                    stratum_decode_token(&moe_vocab, btok[s], tok_text, sizeof(tok_text));
                    fprintf(stdout, "%s", tok_text); fflush(stdout);
                }
            }
            position += emit_n; moe_g_kv_len += emit_n; g += emit_n;
            next_tok = stratum_argmax(moe_gb_logits[accepted],
                                      moe_g_cfg.vocab_size);
            fprintf(stderr, "  spec: B=%d accepted=%d (cum %d/%d calls)\n",
                    B, accepted, spec_accepts, spec_calls);
        }
        fprintf(stderr, "  [NGRAM_SPEC] %d accepted drafts over %d batched calls"
                " (%.2f tok/call)\n",
                spec_accepts, spec_calls,
                spec_calls ? (double)(spec_accepts + spec_calls) / spec_calls : 0.0);
        return 0;
    }

    for (int g = 0; g < n_gen; g++) {
        last_tok = next_tok;
        moe_forward_one_token(last_tok, position++);
        next_tok = stratum_argmax(moe_g_logits, moe_g_cfg.vocab_size);
        stratum_logits_dump_record(moe_g_logits, moe_g_cfg.vocab_size, next_tok);
        fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  logit=%g\n",
                g, last_tok, next_tok, moe_g_logits[next_tok]);
        if (moe_vocab.available) {
            char tok_text[256];
            stratum_decode_token(&moe_vocab, next_tok, tok_text, sizeof(tok_text));
            fprintf(stdout, "%s", tok_text);
            fflush(stdout);
        }
    }

    gguf_close(&moe_g_gguf);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Architecture registration                                          */
/* ------------------------------------------------------------------ */

static const StratumArch stratum_arch_moe = {
    .arch_names   = "llama-moe,llama_moe,moe",
    .description  = "MoE (llama.cpp expert tensors; router top-k; streaming expert slices)",
    .run          = run_moe_arch,
};

STRATUM_REGISTER_ARCH(stratum_arch_moe);
