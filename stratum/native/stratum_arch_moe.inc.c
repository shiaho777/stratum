/*
 * stratum_arch_moe.inc.c — MoE architecture (llama.cpp expert naming).
 *
 * Dense self-attention exactly like the llama arch; the FFN block is a
 * mixture of experts: ffn_gate_inp router [H, E] -> softmax -> top-k
 * experts from stacked tensors (ffn_gate_exps / ffn_up_exps / ffn_down_exps,
 * each [K_in, N_out, E]) -> weighted sum.
 *
 * THE streaming point: only k of E expert slices are touched per token.
 * Three reclaimable-only mechanisms (no mlock, no copies, routing unchanged):
 *   1. STRATUM_MOE_PACK — sidecar that lays gate|up|down of one expert in
 *      one byte range (values copied, not requantized), so one WILLNEED
 *      covers the expert. Built by tools/moe_expert_pack.py.
 *   2. STRATUM_MOE_AHEAD — before attention, a hint router on the pre-attn
 *      residual prefetches likely experts so the fault overlaps attention.
 *      The real post-attn router still selects; misses are hinted then.
 *   3. STRATUM_MOE_ADAPT — decayed per-expert counts, WILLNEED the hottest
 *      experts that fit in STRATUM_MOE_ADAPT_MB. Hints only. DONTNEED is
 *      opt-in (STRATUM_MOE_ADAPT_DROP) because dropping a page the next
 *      token needs is a fault, not a saving.
 * No page is ever locked; residency stays under OS control.
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
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

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
    const GgufTensor* attn_q_norm;   /* optional QK-norm (OLMoE, Qwen-MoE) */
    const GgufTensor* attn_k_norm;
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

static int moe_g_rope_neox = 0;   /* NEOX half-split vs NORM adjacent pairs */
static int moe_g_rope_hold_unscaled = 0;
static void moe_rope(float* x, int head_dim, int rope_dim, int position, float theta) {
    int n_pairs = rope_dim / 2;
    const StratumRope* rp = &moe_g_cfg.rope;
    for (int k = 0; k < n_pairs; k++) {
        float c, s;
        if (moe_g_rope_hold_unscaled || rp->type == STRATUM_ROPE_NONE || !(rp->factor > 1.0f)) {
            float freq  = 1.0f / powf(theta, (float)(2 * k) / (float)rope_dim);
            float angle = (float)position * freq;
            c = cosf(angle);
            s = sinf(angle);
        } else {
            stratum_rope_cs(rp, k, n_pairs, position, &c, &s);
        }
        if (moe_g_rope_neox) {
            float x0 = x[k], x1 = x[k + n_pairs];
            x[k]           = x0 * c - x1 * s;
            x[k + n_pairs] = x0 * s + x1 * c;
        } else {
            float x0 = x[2 * k], x1 = x[2 * k + 1];
            x[2 * k]     = x0 * c - x1 * s;
            x[2 * k + 1] = x0 * s + x1 * c;
        }
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

static int    moe_g_maxkv = 0;
static float* moe_g_K_cache = NULL;
static float* moe_g_V_cache = NULL;
static int    moe_g_kv_len  = 0;
static int    moe_g_kv_stride = 0;   /* allocated per-layer KV positions */

static __thread float* moe_lg = NULL;
static __thread size_t moe_lg_cap = 0;
static float* moe_lg_need(int n) {
    if (n < 1) n = 1;
    if ((size_t)n > moe_lg_cap) {
        free(moe_lg);
        moe_lg = (float*)malloc((size_t)n * sizeof(float));
        moe_lg_cap = moe_lg ? (size_t)n : 0;
    }
    return moe_lg;
}

/* multiseq/spec-verify state: per-slot KV when running independent streams */
#define moe_B_MAX 64
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
    /* W16-converted experts: per-slice blob bytes are derivable from dims
     * (t->nbytes is only the nominal 2B/elem — understates the real blob) */
    if (t->type == 43)   /* GGML_TYPE_Q4K_W16: w16 + d/dm per 256 + m16 */
        return (int64_t)N_out * K_in * 2
             + (int64_t)N_out * (K_in / 256) * 8
             + (int64_t)N_out * (K_in / 32) * 2;
    if (t->type == 44)   /* GGML_TYPE_Q6K_W16: same minus the m16 section */
        return (int64_t)N_out * K_in * 2
             + (int64_t)N_out * (K_in / 256) * 8;
    int64_t one = (int64_t)K_in * N_out;
    if (t->nbytes <= 0 || t->nelem <= 0) return gguf_tensor_bytes((GgmlType)t->type, one);
    int64_t e_all = t->nelem / one;
    if (e_all < 1) e_all = 1;
    return (t->nbytes - 0) / e_all;
}

/* ---- pack sidecar + adaptive hints + ahead prefetch ------------------ *
 * Pack file (tools/moe_expert_pack.py), little-endian, payload page-aligned:
 *   magic "MOEP" | u32 version=1 | u32 n_layers | u32 n_experts
 *   | u32 n_embed | u32 n_ff | u32 gate_type | u32 up_type | u32 down_type
 *   | u64 gate_slice | u64 up_slice | u64 down_slice | u64 data_off
 *   | pad to data_off | for each layer, for each expert: gate|up|down bytes
 * The bytes are a copy of the GGUF slices. Kernels see the same type. */

typedef struct __attribute__((packed)) {
    char     magic[4];
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t n_embed;
    uint32_t n_ff;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint64_t gate_slice;
    uint64_t up_slice;
    uint64_t down_slice;
    uint64_t data_off;
} MoePackHdr;

static const uint8_t* moe_pack_base = NULL;
static size_t         moe_pack_size = 0;
static int            moe_pack_fd   = -1;
static uint64_t       moe_pack_data_off = 0;
static uint64_t       moe_pack_gate = 0, moe_pack_up = 0, moe_pack_down = 0;
static uint64_t       moe_pack_stride = 0;
static int            moe_pack_on = 0;

static float*  moe_adapt_hits = NULL;
static uint8_t* moe_adapt_hot = NULL;
static size_t  moe_adapt_budget = 0;
static int     moe_adapt_on = 0;
static int     moe_adapt_drop = 0;

static int moe_ahead_on(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("STRATUM_MOE_AHEAD");
        v = e ? (atoi(e) != 0) : 1;
    }
    return v;
}

static int moe_spec_conf_on(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("STRATUM_SPEC_CONF");
        v = e ? (atoi(e) != 0) : 1;
    }
    return v;
}

static int moe_page_resident(const void* p, size_t n) {
    if (!p || n == 0) return 0;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps < 4096) ps = 4096;
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(ps - 1);
    size_t len = ((uintptr_t)p + n) - a;
    size_t np = (len + (size_t)ps - 1) / (size_t)ps;
    if (np > 8) np = 8;                 /* sample the first pages only */
    char vec[8];
    if (mincore((void*)a, np * (size_t)ps, vec) != 0) return 0;
    for (size_t i = 0; i < np; i++) if ((vec[i] & 1) == 0) return 0;
    return 1;
}

static const uint8_t* moe_pack_which(int li, int e, int which) {
    uint64_t off = moe_pack_data_off
        + ((uint64_t)li * (uint64_t)moe_g_cfg.n_experts + (uint64_t)e) * moe_pack_stride;
    if (which == 1) off += moe_pack_gate;
    else if (which == 2) off += moe_pack_gate + moe_pack_up;
    return moe_pack_base + off;
}

static size_t moe_expert_span(int li) {
    if (moe_pack_on) return (size_t)moe_pack_stride;
    moe_BlockTensors* b = &moe_g_blocks[li];
    int H = moe_g_cfg.n_embed, Ff = moe_g_cfg.n_ff;
    return (size_t)moe_slice_bytes(b->ffn_gate_exps, H, Ff)
         + (size_t)moe_slice_bytes(b->ffn_up_exps, H, Ff)
         + (size_t)moe_slice_bytes(b->ffn_down_exps, Ff, H);
}

/* Bind one expert slice to a stack tensor whose bytes the linear kernels
 * read via g_st.mmap_base + offset. Pack pages live in a second mapping;
 * the offset is the pointer difference, so no weight bytes are copied. */
static void moe_bind_slice(GgufTensor* dst, int li, int which,
                           const GgufTensor* t, int e, int64_t slice,
                           int N_out, int K_in) {
    *dst = *t;
    dst->nelem  = (int64_t)K_in * N_out;
    dst->nbytes = slice;
    const uint8_t* p = moe_pack_on
        ? moe_pack_which(li, e, which)
        : g_st.mmap_base + t->offset + (uint64_t)e * (uint64_t)slice;
    dst->offset = (uint64_t)(p - g_st.mmap_base);
}

static void moe_advise_expert(int li, int e, int advice) {
    if (e < 0) return;
    if (moe_pack_on) {
        const uint8_t* p = moe_pack_which(li, e, 0);
        size_t n = (size_t)moe_pack_stride;
        if (advice == MADV_WILLNEED && moe_page_resident(p, n)) return;
        madvise((void*)p, n, advice);
        return;
    }
    moe_BlockTensors* b = &moe_g_blocks[li];
    int H = moe_g_cfg.n_embed, Ff = moe_g_cfg.n_ff;
    const GgufTensor* ts[3] = { b->ffn_gate_exps, b->ffn_up_exps, b->ffn_down_exps };
    int64_t sb[3] = {
        moe_slice_bytes(b->ffn_gate_exps, H, Ff),
        moe_slice_bytes(b->ffn_up_exps, H, Ff),
        moe_slice_bytes(b->ffn_down_exps, Ff, H)
    };
    for (int w = 0; w < 3; w++) {
        if (!ts[w]) continue;
        const uint8_t* p = g_st.mmap_base + ts[w]->offset + (uint64_t)e * (uint64_t)sb[w];
        if (advice == MADV_WILLNEED && moe_page_resident(p, (size_t)sb[w])) continue;
        madvise((void*)p, (size_t)sb[w], advice);
    }
}

static void moe_prefetch_expert(int li, int e) {
    moe_advise_expert(li, e, MADV_WILLNEED);
}

static void moe_topk(const float* w, int E, int K, int* sel, float* selw) {
    for (int s = 0; s < K; s++) { sel[s] = -1; selw[s] = -1.0f; }
    for (int e = 0; e < E; e++) {
        float ww = w[e];
        for (int s = 0; s < K; s++) {
            if (ww > selw[s]) {
                for (int t2 = K - 1; t2 > s; t2--) {
                    sel[t2] = sel[t2 - 1];
                    selw[t2] = selw[t2 - 1];
                }
                sel[s] = e;
                selw[s] = ww;
                break;
            }
        }
    }
}

/* Hint, not a decision: router on the pre-attention residual. The real
 * router runs again on the post-attention residual and is what gets used. */
static void moe_ahead_prefetch_one(int li, const float* x) {
    if (!moe_ahead_on()) return;
    moe_BlockTensors* b = &moe_g_blocks[li];
    int H = moe_g_cfg.n_embed;
    int E = moe_g_cfg.n_experts;
    int K = moe_g_cfg.n_experts_used;
    const float* gain = st_f32_tensor_ptr(b->ffn_norm);
    if (!gain || K > 64) return;
    moe_rmsnorm(x, gain, H, moe_g_cfg.rms_eps, moe_g_xn);
    if (st_linear_dispatch(b->ffn_gate_inp, moe_g_xn, moe_g_router, E, H) != 0)
        return;
    moe_softmax_inplace(moe_g_router, E);
    int sel[64];
    float selw[64];
    moe_topk(moe_g_router, E, K, sel, selw);
    for (int s = 0; s < K; s++) moe_prefetch_expert(li, sel[s]);
}

static void moe_adapt_touch(int li, int e) {
    if (!moe_adapt_on || e < 0 || !moe_adapt_hits) return;
    moe_adapt_hits[(size_t)li * (size_t)moe_g_cfg.n_experts + (size_t)e] += 1.0f;
}

static int moe_hit_cmp(const void* a, const void* b) {
    const int* ia = (const int*)a;
    const int* ib = (const int*)b;
    float ha = moe_adapt_hits[*ia];
    float hb = moe_adapt_hits[*ib];
    if (ha < hb) return 1;
    if (ha > hb) return -1;
    return *ia - *ib;
}

static void moe_adapt_rebalance(void) {
    if (!moe_adapt_on || !moe_adapt_hits) return;
    int L = moe_g_cfg.n_layers, E = moe_g_cfg.n_experts;
    int n = L * E;
    int* idx = (int*)malloc(sizeof(int) * (size_t)n);
    uint8_t* neu = (uint8_t*)calloc((size_t)n, 1);
    if (!idx || !neu) { free(idx); free(neu); return; }
    int m = 0;
    for (int i = 0; i < n; i++) if (moe_adapt_hits[i] > 0.0f) idx[m++] = i;
    qsort(idx, (size_t)m, sizeof(int), moe_hit_cmp);
    size_t used = 0;
    for (int j = 0; j < m; j++) {
        int i = idx[j];
        int li = i / E, e = i % E;
        size_t bytes = moe_expert_span(li);
        if (bytes == 0 || used + bytes > moe_adapt_budget) break;
        neu[i] = 1;
        used += bytes;
        if (!moe_adapt_hot[i]) moe_advise_expert(li, e, MADV_WILLNEED);
    }
    for (int i = 0; i < n; i++) {
        if (moe_adapt_hot[i] && !neu[i] && moe_adapt_drop)
            moe_advise_expert(i / E, i % E, MADV_DONTNEED);
        moe_adapt_hot[i] = neu[i];
        moe_adapt_hits[i] *= 0.5f;
        if (moe_adapt_hits[i] < 0.05f) moe_adapt_hits[i] = 0.0f;
    }
    free(idx);
    free(neu);
}

static int moe_pack_check_slice(int li, int e, int which, const GgufTensor* t,
                                int64_t slice) {
    const uint8_t* got = moe_pack_which(li, e, which);
    const uint8_t* exp = g_st.mmap_base + t->offset + (uint64_t)e * (uint64_t)slice;
    return memcmp(got, exp, (size_t)slice) == 0;
}

/* Returns 0 when the pack is in use, 1 when no pack was requested,
 * -1 when a requested pack cannot be used. */
static int moe_pack_load(const char* path) {
    if (!path || !path[0]) return 1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "  MOE_PACK: cannot open %s (%s)\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(MoePackHdr)) {
        fprintf(stderr, "  MOE_PACK: %s is not a pack file\n", path);
        close(fd);
        return -1;
    }
    size_t sz = (size_t)st.st_size;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps < 4096) ps = 4096;
    uintptr_t hint_a = ((uintptr_t)g_st.mmap_base + g_st.mmap_size
                        + (uintptr_t)ps + (uintptr_t)ps) & ~(uintptr_t)(ps - 1);
    void* reserved = mmap((void*)hint_a, sz, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    void* p = MAP_FAILED;
    if (reserved == (void*)hint_a) {
        munmap(reserved, sz);
        p = mmap((void*)hint_a, sz, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
    } else if (reserved != MAP_FAILED) {
        munmap(reserved, sz);
    }
    if (p == MAP_FAILED)
        p = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "  MOE_PACK: mmap %s failed (%s)\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    /* Kernels address weights as g_st.mmap_base + offset. A pack mapped
     * below the model cannot be expressed that way until the base moves
     * down with it and every tensor offset shifts by the same delta.
     * mmap_size stays the model length: it is only a chunking hint. */
    if ((const uint8_t*)p < g_st.mmap_base) {
        ptrdiff_t delta = g_st.mmap_base - (const uint8_t*)p;
        g_st.mmap_base = (const uint8_t*)p;
        for (uint64_t i = 0; i < moe_g_gguf.n_tensors; i++)
            moe_g_gguf.tensors[i].offset += (uint64_t)delta;
    }
    MoePackHdr hdr;
    memcpy(&hdr, p, sizeof hdr);
    int H = moe_g_cfg.n_embed, Ff = moe_g_cfg.n_ff, E = moe_g_cfg.n_experts;
    int L = moe_g_cfg.n_layers;
    moe_BlockTensors* b0 = &moe_g_blocks[0];
    int64_t gsb = moe_slice_bytes(b0->ffn_gate_exps, H, Ff);
    int64_t usb = moe_slice_bytes(b0->ffn_up_exps, H, Ff);
    int64_t dsb = moe_slice_bytes(b0->ffn_down_exps, Ff, H);
    int ok = memcmp(hdr.magic, "MOEP", 4) == 0 && hdr.version == 1
          && (int)hdr.n_layers == L && (int)hdr.n_experts == E
          && (int)hdr.n_embed == H && (int)hdr.n_ff == Ff
          && hdr.gate_type == b0->ffn_gate_exps->type
          && hdr.up_type == b0->ffn_up_exps->type
          && hdr.down_type == b0->ffn_down_exps->type
          && hdr.gate_slice == (uint64_t)gsb
          && hdr.up_slice == (uint64_t)usb
          && hdr.down_slice == (uint64_t)dsb
          && hdr.data_off >= sizeof(MoePackHdr)
          && hdr.data_off < sz;
    uint64_t stride = hdr.gate_slice + hdr.up_slice + hdr.down_slice;
    uint64_t need = hdr.data_off + (uint64_t)L * (uint64_t)E * stride;
    if (ok && need > sz) ok = 0;
    if (!ok) {
        fprintf(stderr, "  MOE_PACK: %s does not match this model\n", path);
        munmap(p, sz);
        close(fd);
        return -1;
    }
    moe_pack_base = (const uint8_t*)p;
    moe_pack_size = sz;
    moe_pack_fd = fd;
    moe_pack_data_off = hdr.data_off;
    moe_pack_gate = hdr.gate_slice;
    moe_pack_up = hdr.up_slice;
    moe_pack_down = hdr.down_slice;
    moe_pack_stride = stride;
    moe_pack_on = 1;

    const char* chk = getenv("STRATUM_MOE_PACK_CHECK");
    int full = chk && atoi(chk) != 0;
    int bad = 0, checked = 0;
    for (int li = 0; li < L && !bad; li++) {
        moe_BlockTensors* b = &moe_g_blocks[li];
        int64_t gs = moe_slice_bytes(b->ffn_gate_exps, H, Ff);
        int64_t us = moe_slice_bytes(b->ffn_up_exps, H, Ff);
        int64_t ds = moe_slice_bytes(b->ffn_down_exps, Ff, H);
        if (gs != gsb || us != usb || ds != dsb) { bad = 1; break; }
        for (int e = 0; e < E; e++) {
            int sample = full || e == 0 || e == E / 2 || e == E - 1;
            if (!sample) continue;
            checked++;
            if (!moe_pack_check_slice(li, e, 0, b->ffn_gate_exps, gs)
                || !moe_pack_check_slice(li, e, 1, b->ffn_up_exps, us)
                || !moe_pack_check_slice(li, e, 2, b->ffn_down_exps, ds)) {
                bad = 1;
                break;
            }
        }
    }
    if (bad) {
        fprintf(stderr, "  MOE_PACK: byte check failed — refusing the sidecar\n");
        munmap(p, sz);
        close(fd);
        moe_pack_base = NULL;
        moe_pack_on = 0;
        moe_pack_fd = -1;
        return -1;
    }
    fprintf(stderr, "  MOE_PACK: %s  stride %.1f KB  (%s check, %d experts)\n",
            path, (double)stride / 1024.0, full ? "full" : "sample", checked);
    return 0;
}

static int moe_stream_init(void) {
    int n = moe_g_cfg.n_layers * moe_g_cfg.n_experts;
    if (n < 1) return -1;
    moe_adapt_hits = (float*)calloc((size_t)n, sizeof(float));
    moe_adapt_hot  = (uint8_t*)calloc((size_t)n, 1);
    if (!moe_adapt_hits || !moe_adapt_hot) return -1;
    const char* ae = getenv("STRATUM_MOE_ADAPT");
    const char* mb = getenv("STRATUM_MOE_ADAPT_MB");
    long mib = mb ? atol(mb) : 256;
    if (mib < 0) mib = 0;
    moe_adapt_budget = (size_t)mib * 1024u * 1024u;
    moe_adapt_on = (ae ? atoi(ae) != 0 : 1) && mib > 0;
    const char* dr = getenv("STRATUM_MOE_ADAPT_DROP");
    moe_adapt_drop = dr && atoi(dr) != 0;
    fprintf(stderr, "  stream: ahead=%d adapt=%d budget=%ldMB drop=%d spec_conf=%d\n",
            moe_ahead_on(), moe_adapt_on, mib, moe_adapt_drop, moe_spec_conf_on());
    return 0;
}

static void moe_dispatch_slice(int li, int which, const GgufTensor* t, int e,
                               int64_t slice_bytes,
                               const float* x, float* y, int N_out, int K_in) {
    GgufTensor tmp;
    moe_bind_slice(&tmp, li, which, t, e, slice_bytes, N_out, K_in);
    if (st_linear_dispatch(&tmp, x, y, N_out, K_in) != 0)
        fprintf(stderr, "moe: dispatch failed for expert %d\n", e);
}

static int moe_forward_block(int li, int position) {
    moe_BlockTensors* b = &moe_g_blocks[li];
    moe_ahead_prefetch_one(li, moe_g_x);
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

    /* dense attention — QKV share xn; one x-quant, three weight rows */
    st_q4k_group(moe_g_xn, H,
                 b->attn_q, moe_g_q_buf, Nq * Hd,
                 b->attn_k, moe_g_k_buf, Nk * Hd,
                 b->attn_v, moe_g_v_buf, Nk * Hd);

    /* QK-norm (OLMoE/Qwen-MoE): one RMS norm over the WHOLE projected
     * vector, applied before the head split (llama.cpp olmoe semantics) */
    if (b->attn_q_norm)
        moe_rmsnorm(moe_g_q_buf, st_f32_tensor_ptr(b->attn_q_norm),
                    Nq * Hd, moe_g_cfg.rms_eps, moe_g_q_buf);
    if (b->attn_k_norm)
        moe_rmsnorm(moe_g_k_buf, st_f32_tensor_ptr(b->attn_k_norm),
                    Nk * Hd, moe_g_cfg.rms_eps, moe_g_k_buf);

    for (int h = 0; h < Nq; h++)
        moe_rope(moe_g_q_buf + h * Hd, Hd, moe_g_cfg.rope_dim,
                 position, moe_g_cfg.rope_theta);
    for (int h = 0; h < Nk; h++)
        moe_rope(moe_g_k_buf + h * Hd, Hd, moe_g_cfg.rope_dim,
                 position, moe_g_cfg.rope_theta);

    int kv_len_now = moe_g_kv_len + 1;
    {
        size_t per_layer = (size_t)moe_g_kv_stride * Nk * Hd;
        size_t off = (size_t)li * per_layer
                   + (size_t)moe_g_kv_len * Nk * Hd;
        memcpy(moe_g_K_cache + off, moe_g_k_buf, sizeof(float) * Nk * Hd);
        memcpy(moe_g_V_cache + off, moe_g_v_buf, sizeof(float) * Nk * Hd);
    }

    float scale = 1.0f / sqrtf((float)Hd);
    for (int h = 0; h < Nq; h++) {
        int kv_h = h * Nk / Nq;
        const float* qh = moe_g_q_buf + h * Hd;
        size_t per_layer = (size_t)moe_g_kv_stride * Nk * Hd;
        const float* K_layer = moe_g_K_cache + (size_t)li * per_layer;
        const float* V_layer = moe_g_V_cache + (size_t)li * per_layer;

        float* logits = moe_lg_need(kv_len_now);
        if (!logits) { fprintf(stderr, "moe attn scratch alloc failed\n"); return -1; }
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

    st_linear_dispatch_add(b->attn_output, moe_g_attn_out, moe_g_x_resid, moe_g_x,
                           H, Nq * Hd);

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

    /* True selection. Ahead may already have hinted a guess; this covers
     * the misses. One range when the pack sidecar is on. */
    for (int s = 0; s < K_; s++) {
        int e = moe_g_sel[s];
        if (e < 0) continue;
        moe_prefetch_expert(li, e);
        moe_adapt_touch(li, e);
        if (moe_stats_on()) moe_stat_expert_touches++;
    }

    /* accumulate weighted expert outputs */
    memset(moe_g_ff_acc, 0, sizeof(float) * H);
    for (int s = 0; s < K_; s++) {
        int e = moe_g_sel[s];
        if (e < 0) continue;
        float w = moe_g_selw[s];
        static float ff_out_tmp[8192];
        moe_dispatch_slice(li, 0, b->ffn_gate_exps, e,
                           moe_slice_bytes(b->ffn_gate_exps, H, Ff),
                           moe_g_xn, moe_g_ff_g, Ff, H);
        moe_dispatch_slice(li, 1, b->ffn_up_exps, e,
                           moe_slice_bytes(b->ffn_up_exps, H, Ff),
                           moe_g_xn, moe_g_ff_u, Ff, H);
        moe_swiglu(moe_g_ff_g, moe_g_ff_u, Ff, moe_g_ff_a);
        moe_dispatch_slice(li, 2, b->ffn_down_exps, e,
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
    return 0;
}

/* ---- single-token forward --------------------------------------------- */

static int moe_forward_one_token(int token_id, int position) {
    int H = moe_g_cfg.n_embed;
    int V = moe_g_cfg.vocab_size;

    if (moe_g_kv_len >= moe_g_kv_stride) {
        fprintf(stderr, "moe: context %d exceeds KV capacity %d "
                "(model context_length / STRATUM_MAX_KV)\n",
                moe_g_kv_len + 1, moe_g_maxkv);
        return -1;
    }

    moe_embed_lookup(token_id, moe_g_x);
    moe_stats_begin_step();

    for (int li = 0; li < moe_g_cfg.n_layers; li++)
        if (moe_forward_block(li, position) != 0) return -1;

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
    moe_adapt_rebalance();
    moe_g_kv_len++;
    return 0;
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

static void moe_multix_slice(int li, int which, const GgufTensor* t, int e,
                             int64_t slice_bytes,
                             const float* const* xs, float* const* ys,
                             int B, int N_out, int K_in) {
    GgufTensor tmp;
    moe_bind_slice(&tmp, li, which, t, e, slice_bytes, N_out, K_in);
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
    /* expert member staging indexed by GLOBAL member id (s*K_+k, up to
     * B*K_-1): per-expert tickets then write disjoint slots, and a single
     * expert can never overflow the scratch (the old local-j arrays were
     * only moe_B_MAX deep — an overflow when >moe_B_MAX members routed to
     * one expert). */
    static float* mgf = NULL; static float* muf = NULL;
    static float* maf = NULL; static size_t mem_cap = 0;
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
            if (!moe_gb_logits[s]) moe_gb_logits[s] = calloc(V, 4);
            if (!x[s]||!xr[s]||!xn[s]||!qb[s]||!kb[s]||!vb[s]||!ao[s]||
                !ap[s]||!rt[s]||!ac[s]||!moe_gb_logits[s]) return -1;
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
    size_t need_mem = need_sel * (size_t)Ff;
    if (need_mem > mem_cap) {
        free(mgf); free(muf); free(maf);
        mgf = malloc(need_mem * 4); muf = malloc(need_mem * 4);
        maf = malloc(need_mem * 4);
        mem_cap = (mgf && muf && maf) ? need_sel : 0;
        if (!mem_cap) return -1;
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

    if (shared_kv) {
        if (kvlen[0] + B > moe_g_kv_stride) {
            fprintf(stderr, "moe: batched forward needs %d > KV capacity %d\n",
                    kvlen[0] + B, moe_g_maxkv);
            return -1;
        }
    } else {
        for (int s = 0; s < B; s++) {
            if (kvlen[s] + 1 > moe_g_ms_maxkv) {
                fprintf(stderr, "moe: context %d exceeds KV capacity %d "
                        "(model context_length / STRATUM_MAX_KV)\n",
                        kvlen[s] + 1, moe_g_ms_maxkv);
                return -1;
            }
        }
    }

    for (int s = 0; s < B; s++) moe_embed_lookup(tokens[s], x[s]);

    const float scale = 1.0f / sqrtf((float)Hd);
    size_t kv_seqstride = (size_t)moe_g_ms_maxkv * Nk * Hd;
    size_t kv_laystride = (size_t)B * kv_seqstride;
    size_t sh_laystride = (size_t)moe_g_kv_stride * Nk * Hd;

    const float* cxn[moe_B_MAX]; float* cqb[moe_B_MAX];
    float* ckb[moe_B_MAX]; float* cvb[moe_B_MAX];
    float* cao[moe_B_MAX]; float* cap[moe_B_MAX];
    float* crt[moe_B_MAX]; float* clog[moe_B_MAX];

    /* section profiling (STRATUM_MOE_PROF): cumulative µs per section */
    static double prof_us[8];
    static int prof_on = -1;
    if (prof_on < 0) prof_on = getenv("STRATUM_MOE_PROF") ? 1 : 0;
#define MOE_PROF_T() ({ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); \
                        ts.tv_sec*1e6 + ts.tv_nsec*1e-3; })
#define MOE_PROF_ACC(i, t0) do { if (prof_on) prof_us[i] += MOE_PROF_T() - (t0); } while(0)
    double pt;

    moe_stats_begin_step();
    for (int li = 0; li < moe_g_cfg.n_layers; li++) {
        moe_BlockTensors* b = &moe_g_blocks[li];
        if (moe_ahead_on() && K_ <= 64) {
            const float* hgain = st_f32_tensor_ptr(b->ffn_norm);
            if (hgain) {
                const float* hx[moe_B_MAX];
                float* hy[moe_B_MAX];
                for (int s = 0; s < B; s++) {
                    moe_rmsnorm(x[s], hgain, H, moe_g_cfg.rms_eps, xn[s]);
                    hx[s] = xn[s];
                    hy[s] = rt[s];
                }
                st_linear_multix(b->ffn_gate_inp, hx, hy, B, E, H);
                int hsel[64];
                float hselw[64];
                for (int s = 0; s < B; s++) {
                    moe_softmax_inplace(rt[s], E);
                    moe_topk(rt[s], E, K_, hsel, hselw);
                    for (int k = 0; k < K_; k++) moe_prefetch_expert(li, hsel[k]);
                }
            }
        }
        pt = MOE_PROF_T();
        {
            const float* gain = st_f32_tensor_ptr(b->attn_norm);
            float eps = moe_g_cfg.rms_eps;
            const float** cxn_p = cxn;
            if (B >= 32) st_par_run(B, ^(int s) {
                memcpy(xr[s], x[s], sizeof(float) * H);
                moe_rmsnorm(x[s], gain, H, eps, xn[s]);
                cxn_p[s] = xn[s];
            }); else
            for (int s = 0; s < B; s++) {
                memcpy(xr[s], x[s], sizeof(float) * H);
                moe_rmsnorm(x[s], gain, H, eps, xn[s]);
                cxn[s] = xn[s];
            }
        }
        for (int s = 0; s < B; s++) {
            cqb[s] = qb[s]; ckb[s] = kb[s]; cvb[s] = vb[s];
        }
        MOE_PROF_ACC(0, pt); pt = MOE_PROF_T();
        st_linear_multix(b->attn_q, cxn, cqb, B, Nq * Hd, H);
        st_linear_multix(b->attn_k, cxn, ckb, B, Nk * Hd, H);
        st_linear_multix(b->attn_v, cxn, cvb, B, Nk * Hd, H);
        MOE_PROF_ACC(1, pt); pt = MOE_PROF_T();

        if (b->attn_q_norm) {
            const float* gq = st_f32_tensor_ptr(b->attn_q_norm);
            for (int s = 0; s < B; s++)
                moe_rmsnorm(qb[s], gq, Nq * Hd, moe_g_cfg.rms_eps, qb[s]);
        }
        if (b->attn_k_norm) {
            const float* gk = st_f32_tensor_ptr(b->attn_k_norm);
            for (int s = 0; s < B; s++)
                moe_rmsnorm(kb[s], gk, Nk * Hd, moe_g_cfg.rms_eps, kb[s]);
        }

        /* rope+KV write: pooled only at wide B — at ms16 the pool barrier
         * costs more than the ~50us serial pass saves */
        if (B >= 24) st_par_run(B, ^(int s) {
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
        }); else
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
        MOE_PROF_ACC(2, pt); pt = MOE_PROF_T();

        volatile int _lgfail = 0;
        st_par_run(B, ^(int s) {
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
                float* logits = moe_lg_need(klen);
                if (!logits) { *(int*)&_lgfail = 1; return; }
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
        });
        if (_lgfail) { fprintf(stderr, "moe attn scratch alloc failed\n"); return -1; }
        MOE_PROF_ACC(3, pt); pt = MOE_PROF_T();
        for (int s = 0; s < B; s++) { cao[s] = ao[s]; cap[s] = ap[s]; }
        st_linear_multix(b->attn_output, (const float* const*)cao, cap,
                         B, H, Nq * Hd);
        MOE_PROF_ACC(4, pt); pt = MOE_PROF_T();
        if (B >= 32) st_par_run(B, ^(int s) {
            for (int i = 0; i < H; i++) x[s][i] = xr[s][i] + ap[s][i];
        }); else
        for (int s = 0; s < B; s++)
            for (int i = 0; i < H; i++) x[s][i] = xr[s][i] + ap[s][i];

        /* ---- MoE FFN, expert-grouped batching ---- */
        {
            const float* gain = st_f32_tensor_ptr(b->ffn_norm);
            float eps = moe_g_cfg.rms_eps;
            float** crt_p = crt;
            if (B >= 32) st_par_run(B, ^(int s) {
                memcpy(xr[s], x[s], sizeof(float) * H);
                moe_rmsnorm(x[s], gain, H, eps, xn[s]);
                crt_p[s] = rt[s];
            }); else
            for (int s = 0; s < B; s++) {
                memcpy(xr[s], x[s], sizeof(float) * H);
                moe_rmsnorm(x[s], gain, H, eps, xn[s]);
                crt[s] = rt[s];
            }
        }
        st_linear_multix(b->ffn_gate_inp, cxn, crt, B, E, H);

        /* per-stream softmax + top-k (same stable lower-id tie-break as
         * the single-stream path). The top-k is per-stream private →
         * pooled at wide B; the memb append stays serial (E*B int writes
         * — cheaper than a barrier at any B). */
        memset(mcnt, 0, sizeof(int) * (size_t)E);
        int* sel = sel_all; float* selw = selw_all;
        int Kc = K_, Ec = E;
        float* const* rt_p = (float* const*)rt;
        void (^topk)(int) = ^(int s) {
            int* sel_s = sel + s * Kc; float* selw_s = selw + s * Kc;
            moe_softmax_inplace(rt_p[s], Ec);
            for (int k = 0; k < Kc; k++) { sel_s[k] = -1; selw_s[k] = -1.0f; }
            for (int e = 0; e < Ec; e++) {
                float w = rt_p[s][e];
                for (int k = 0; k < Kc; k++) {
                    if (w > selw_s[k]) {
                        for (int t2 = Kc - 1; t2 > k; t2--) {
                            sel_s[t2] = sel_s[t2 - 1];
                            selw_s[t2] = selw_s[t2 - 1];
                        }
                        sel_s[k] = e; selw_s[k] = w;
                        break;
                    }
                }
            }
        };
        if (B >= 32) st_par_run(B, topk);
        else for (int s = 0; s < B; s++) topk(s);
        for (int s = 0; s < B; s++) {
            int* sel_s = sel + s * K_;
            for (int k = 0; k < K_; k++)
                if (sel_s[k] >= 0)
                    memb[(size_t)sel_s[k] * B + mcnt[sel_s[k]]++] = s * K_ + k;
        }

        MOE_PROF_ACC(5, pt); pt = MOE_PROF_T();
        int64_t g_sb = moe_slice_bytes(b->ffn_gate_exps, H, Ff);
        int64_t u_sb = moe_slice_bytes(b->ffn_up_exps, H, Ff);
        int64_t d_sb = moe_slice_bytes(b->ffn_down_exps, Ff, H);
        /* prefetch each selected expert once (reclaimable hints) */
        for (int e = 0; e < E; e++) {
            if (!mcnt[e]) continue;
            moe_prefetch_expert(li, e);
            moe_adapt_touch(li, e);
            if (moe_stats_on()) moe_stat_expert_touches += mcnt[e];
        }

        /* measured: WILLNEED on next-layer attn tensors gains nothing —
         * those matmuls are AMX-compute-bound, not cold-read bound */

        /* per expert: batch its member streams through one multix.
         * Experts are independent — farm them across the persistent pool;
         * nested st_linear_multix calls serialize inside each ticket
         * (thread-local pack buffers) so the parallelism is at the expert
         * level. down writes straight into ffo[m] — no scatter pass. */
        /* chunked tickets: split hot experts' member lists into ~8-member
         * chunks so one overloaded expert can't straggle behind the pool */
        enum { CH = 8 }; int tk_cap = B * K_ / CH + E + 2;
        int* tk_e  = (int*)malloc(sizeof(int) * 3 * tk_cap);
        int* tk_j0 = tk_e + tk_cap;
        int* tk_j1 = tk_e + 2 * tk_cap;
        int nt = 0;
        for (int e = 0; e < E; e++) {
            int M = mcnt[e];
            for (int j0 = 0; j0 < M; j0 += CH) {
                tk_e[nt] = e; tk_j0[nt] = j0;
                tk_j1[nt] = (j0 + CH < M) ? j0 + CH : M;
                nt++;
            }
        }
        st_par_run(nt, ^(int t) {
            int e = tk_e[t], M = tk_j1[t] - tk_j0[t];
            const float* xe[CH]; float* og[CH];
            float* ou[CH];     float* od[CH];
            const float* oa[CH];
            for (int j = 0; j < M; j++) {
                int m = memb[(size_t)e * B + tk_j0[t] + j];
                int s = m / K_;
                xe[j] = xn[s];
                og[j] = mgf + (size_t)m * Ff;
                ou[j] = muf + (size_t)m * Ff;
                oa[j] = maf + (size_t)m * Ff;
                od[j] = ffo + (size_t)m * H;
            }
            moe_multix_slice(li, 0, b->ffn_gate_exps, e, g_sb, xe, og, M, Ff, H);
            moe_multix_slice(li, 1, b->ffn_up_exps,   e, u_sb, xe, ou, M, Ff, H);
            for (int j = 0; j < M; j++) {
                int m = memb[(size_t)e * B + tk_j0[t] + j];
                moe_swiglu(mgf + (size_t)m * Ff, muf + (size_t)m * Ff,
                           Ff, maf + (size_t)m * Ff);
            }
            moe_multix_slice(li, 2, b->ffn_down_exps, e, d_sb, oa, od, M, H, Ff);
            if (moe_stats_on())
                for (int j = 0; j < M; j++) {
                    int ee = e + li * E;
                    if (ee / 64 < 256)
                        __atomic_or_fetch(&moe_stat_mask[ee / 64],
                                          1ULL << (ee % 64),
                                          __ATOMIC_RELAXED);
                }
        });
        free(tk_e);

        /* accumulate in each stream's own top-k rank order (FP order ==
         * single-stream path) */
        MOE_PROF_ACC(6, pt); pt = MOE_PROF_T();
        st_par_run(B, ^(int s) {
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
        });
        MOE_PROF_ACC(7, pt);
    }
    moe_stats_end_step();

    if (prof_on) {
        static int pc = 0;
        if (++pc % 8 == 0) {
            fprintf(stderr, "  [moe-prof] rmsN=%.0f qkv=%.0f ropekv=%.0f "
                    "attn=%.0f attnO=%.0f rtr+topk=%.0f expert=%.0f accum=%.0f us\n",
                    prof_us[0]/pc*8, prof_us[1]/pc*8, prof_us[2]/pc*8,
                    prof_us[3]/pc*8, prof_us[4]/pc*8, prof_us[5]/pc*8,
                    prof_us[6]/pc*8, prof_us[7]/pc*8);
        }
    }

    double pt2 = MOE_PROF_T();
    {
        const float* gain = st_f32_tensor_ptr(moe_g_output_norm);
        float eps = moe_g_cfg.rms_eps;
        float** clog_p = clog;
        if (B >= 32) st_par_run(B, ^(int s) {
            moe_rmsnorm(x[s], gain, H, eps, xn[s]);
            clog_p[s] = moe_gb_logits[s];
        }); else
        for (int s = 0; s < B; s++) {
            moe_rmsnorm(x[s], gain, H, eps, xn[s]);
            clog_p[s] = moe_gb_logits[s];
        }
    }
    const GgufTensor* lm = moe_g_output_w ? moe_g_output_w : moe_g_token_embd;
    st_linear_multix(lm, cxn, clog, B, V, H);
    if (prof_on) {
        static double lm_us = 0; static int lc = 0;
        lm_us += MOE_PROF_T() - pt2;
        if (++lc % 8 == 0)
            fprintf(stderr, "  [moe-prof] lm_head=%.0f us\n", lm_us/lc*8);
    }
    moe_adapt_rebalance();
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

        /* QK-norm is optional (OLMoE, Qwen-MoE) — absent on llama-moe */
        snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", li);
        moe_g_blocks[li].attn_q_norm = gguf_find_tensor(&moe_g_gguf, nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", li);
        moe_g_blocks[li].attn_k_norm = gguf_find_tensor(&moe_g_gguf, nm);
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

static int moe_allocate_state(int kv_need) {
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

    /* KV sized to this request (prompt+gen+spec slack), clamped to the
     * shared engine cap. calloc is demand-zero; unused positions stay free. */
    if (kv_need < 1) kv_need = 1;
    if (kv_need > moe_g_maxkv) kv_need = moe_g_maxkv;
    moe_g_kv_stride = kv_need;
    size_t kv_floats = (size_t)L * moe_g_kv_stride * Nk * Hd;
    moe_g_K_cache = (float*)calloc(kv_floats, sizeof(float));
    moe_g_V_cache = (float*)calloc(kv_floats, sizeof(float));

    if (!moe_g_x || !moe_g_K_cache || !moe_g_logits || !moe_g_router ||
        !moe_g_ff_acc) return -1;

    fprintf(stderr, "  KV cache: %.1f MB anonymous, %d positions (cap %d)\n",
            (double)kv_floats * 2 * 4 / (1024.0 * 1024.0),
            moe_g_kv_stride, moe_g_maxkv);
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
    {
        /* rope layout: llama-arch GGUFs permute Q/K so adjacent-pair rope
         * reproduces HF rotate_half; OLMoE/Qwen-MoE store unpermuted and
         * need NEOX half-split rotation (same convention as qwen3). */
        char* a = gguf_get_string_dup(&moe_g_gguf, "general.architecture");
        if (a) {
            moe_g_rope_neox = (strcmp(a, "olmoe") == 0
                               || strcmp(a, "qwen3moe") == 0);
            fprintf(stderr, "  rope layout: %s (arch %s)\n",
                    moe_g_rope_neox ? "NEOX half-split" : "NORM adjacent-pairs", a);
            free(a);
        }
    }

    stratum_linear_init(moe_g_gguf.mmap_base, moe_g_gguf.mmap_size);
    stratum_engine_init(moe_g_gguf.mmap_size);
    if (stratum_load_config(&moe_g_gguf, &moe_g_cfg) != 0) return 1;
    moe_g_maxkv = stratum_resolve_kv_cap(moe_g_cfg.context_length,
                                        getenv("STRATUM_MAX_KV"));
    stratum_rope_bind_context(&moe_g_cfg.rope, moe_g_cfg.context_length, moe_g_maxkv);
    fprintf(stderr, "\n");
    stratum_print_config(&moe_g_cfg);
    stratum_rope_report(&moe_g_cfg.rope, moe_g_cfg.context_length, moe_g_maxkv);
    fprintf(stderr, "\n");

    if (moe_discover_blocks() != 0) return 1;
    if (moe_stream_init() != 0) return 1;

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
    if (moe_pack_load(getenv("STRATUM_MOE_PACK")) < 0) return 1;

    static StratumVocab moe_vocab;
    stratum_vocab_init(&moe_g_gguf, moe_g_gguf.mmap_base, &moe_vocab);

    int n_gen = (argc > 2) ? atoi(argv[2]) : 4;
    int* prompt = (int*)malloc((size_t)(moe_g_maxkv + 2) * sizeof(int));
    if (!prompt) { fprintf(stderr, "prompt alloc failed\n"); return 1; }
    int n_prompt = 0;
    if (argc > 3 && argv[3][0] == '@') {
        FILE* pf = fopen(argv[3] + 1, "r");
        if (!pf) {
            fprintf(stderr, "  prompt file %s: %s\n", argv[3] + 1, strerror(errno));
            return 1;
        }
        int id;
        while (fscanf(pf, "%d", &id) == 1) {
            if (n_prompt >= moe_g_maxkv) {
                fprintf(stderr, "  prompt truncated at KV capacity %d\n",
                        moe_g_maxkv);
                break;
            }
            prompt[n_prompt++] = id;
        }
        fclose(pf);
    } else {
        int more = 0;
        for (int i = 3; i < argc; i++) {
            if (n_prompt >= moe_g_maxkv) { more = 1; break; }
            prompt[n_prompt++] = atoi(argv[i]);
        }
        if (more)
            fprintf(stderr, "  prompt truncated at KV capacity %d\n",
                    moe_g_maxkv);
    }
    if (n_prompt == 0) prompt[n_prompt++] = 1;
    fprintf(stderr, "  prompt length: %d\n", n_prompt);
    moe_g_rope_hold_unscaled = stratum_rope_hold_unscaled(&moe_g_cfg.rope, n_prompt, n_gen);
    stratum_rope_reach_report(&moe_g_cfg.rope, n_prompt, n_gen);

    int spec_k = 0;
    { const char* e = getenv("STRATUM_NGRAM_SPEC"); if (e) spec_k = atoi(e); }
    if (moe_allocate_state(n_prompt + n_gen + spec_k + 4) != 0) return 1;

    int ms_B = 0;
    { const char* e = getenv("STRATUM_MULTISEQ"); if (e) ms_B = atoi(e); }
    if (ms_B >= 1) {
        if (ms_B > moe_B_MAX) ms_B = moe_B_MAX;
        moe_g_ms_maxkv = n_prompt + n_gen + 2;
        if (moe_g_ms_maxkv > moe_g_maxkv) moe_g_ms_maxkv = moe_g_maxkv;
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
    /* Batched prefill through the shared-KV verify path: each expert slice
     * read serves every routed position in the chunk (vs one weight+expert
     * scan per token). Greedy-equal, not bit-exact (batched SDOT order);
     * STRATUM_PREFILL_B overrides the chunk size, 1 = old serial path. */
    int pf_B = moe_B_MAX;
    { const char* e = getenv("STRATUM_PREFILL_B"); if (e) pf_B = atoi(e); }
    if (pf_B < 1) pf_B = 1;
    if (pf_B > moe_B_MAX) pf_B = moe_B_MAX;
    {
        struct timespec pf0, pf1; clock_gettime(CLOCK_MONOTONIC, &pf0);
        int btok[moe_B_MAX], bpos[moe_B_MAX], bkv[moe_B_MAX];
        for (int t = 0; t < n_prompt; ) {
            int Bk = n_prompt - t; if (Bk > pf_B) Bk = pf_B;
            if (Bk == 1) {
                if (moe_forward_one_token(prompt[t], position++) != 0) return 1;
                t++; last_tok = prompt[t - 1]; continue;
            }
            for (int s = 0; s < Bk; s++) {
                btok[s] = prompt[t + s]; bpos[s] = position + s;
                bkv[s] = position;
            }
            if (moe_forward_batch(btok, bpos, bkv, Bk) != 0) return 1;
            memcpy(moe_g_logits, moe_gb_logits[Bk - 1],
                   sizeof(float) * moe_g_cfg.vocab_size);
            last_tok = prompt[t + Bk - 1];
            t += Bk; position += Bk; moe_g_kv_len += Bk;
        }
        clock_gettime(CLOCK_MONOTONIC, &pf1);
        fprintf(stderr, "  prefill %d tok in %.3fs (B=%d)\n", n_prompt,
                (pf1.tv_sec-pf0.tv_sec)+(pf1.tv_nsec-pf0.tv_nsec)/1e9, pf_B);
    }
    int next_tok = stratum_argmax(moe_g_logits, moe_g_cfg.vocab_size);
    fprintf(stderr, "  after prefill, sampled = %d  (logit=%g)\n",
            next_tok, moe_g_logits[next_tok]);

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
                int conf = moe_spec_conf_on();
                for (int ng = 3; ng >= 2 && prop < 0; ng--) {
                    int total = hlen + hn;
                    if (total < ng + 1) continue;
                    int suf[3];
                    for (int i = 0; i < ng; i++) {
                        int idx = total - ng + i;
                        suf[i] = (idx < hlen) ? hist[idx] : hyp[idx - hlen];
                    }
                    int cand0 = -1, amb = 0;
                    for (int start = total - ng - 1; start >= 0; start--) {
                        int match = 1;
                        for (int i = 0; i < ng; i++) {
                            int idx = start + i;
                            int tk = (idx < hlen) ? hist[idx] : hyp[idx - hlen];
                            if (tk != suf[i]) { match = 0; break; }
                        }
                        if (!match) continue;
                        int nidx = start + ng;
                        int cand = (nidx < hlen) ? hist[nidx] : hyp[nidx - hlen];
                        if (cand0 < 0) cand0 = cand;
                        else if (cand != cand0 && conf) { amb = 1; break; }
                        if (!conf) break;   /* historical: first match wins */
                    }
                    /* Ambiguous n-gram: try the shorter suffix. A draft we
                     * do not propose is emitted by the next real step. */
                    if (cand0 >= 0 && !amb) prop = cand0;
                }
                if (prop < 0) break;
                btok[B] = prop; bpos[B] = position + B;
                hyp[hn++] = prop; B++;
            }

            if (B == 1) {
                /* no draft: single-token step (KV-full falls here too) */
                if (moe_forward_one_token(next_tok, position) != 0) return 1;
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
            if (moe_g_kv_len + B >= moe_g_kv_stride) {
                /* KV full: fall back to single-token step */
                if (moe_forward_one_token(next_tok, position) != 0) return 1;
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
            if (g + emit_n > n_gen) emit_n = n_gen - g;
            spec_accepts += emit_n - 1;
            for (int s = 0; s < emit_n; s++) {
                fprintf(stderr, "  step %2d  in=%d  (spec)\n", g + s, btok[s]);
                hist[hlen++] = btok[s];
                if (moe_vocab.available) {
                    char tok_text[256];
                    stratum_decode_token(&moe_vocab, btok[s], tok_text, sizeof(tok_text));
                    fprintf(stdout, "%s", tok_text); fflush(stdout);
                }
            }
            position += emit_n; moe_g_kv_len += emit_n; g += emit_n;
            next_tok = stratum_argmax(moe_gb_logits[emit_n - 1],
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

    struct timespec t_dec0, t_dec1;
    clock_gettime(CLOCK_MONOTONIC, &t_dec0);
    for (int g = 0; g < n_gen; g++) {
        last_tok = next_tok;
        if (moe_forward_one_token(last_tok, position++) != 0) return 1;
        next_tok = stratum_argmax(moe_g_logits, moe_g_cfg.vocab_size);
        stratum_logits_dump_record(moe_g_logits, moe_g_cfg.vocab_size, next_tok);
        fprintf(stderr, "  step %2d  in=%d  stratum_argmax=%d  logit=%g\n",
                g, last_tok, next_tok, moe_g_logits[next_tok]);
        if (moe_vocab.available) {
            char tok_text[256];
            stratum_decode_token(&moe_vocab, next_tok, tok_text, sizeof(tok_text));
            fflush(stdout);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_dec1);
    double dec_s = (t_dec1.tv_sec - t_dec0.tv_sec)
                 + (t_dec1.tv_nsec - t_dec0.tv_nsec) / 1e9;
    fprintf(stderr, "\n  decode: %d tokens in %.2fs = %.1f tok/s\n",
            n_gen, dec_s, n_gen / dec_s);

    gguf_close(&moe_g_gguf);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Architecture registration                                          */
/* ------------------------------------------------------------------ */

static const StratumArch stratum_arch_moe = {
    .arch_names   = "llama-moe,llama_moe,moe,olmoe",
    .description  = "MoE (llama.cpp expert tensors; router top-k; streaming expert slices)",
    .run          = run_moe_arch,
};

STRATUM_REGISTER_ARCH(stratum_arch_moe);
