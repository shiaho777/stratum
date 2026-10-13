/* RoPE frequencies and the KV position cap.
 *
 * Unscaled frequencies are 1/theta^(2i/dim). That is the whole story
 * until the reserved context grows past the length the model was
 * trained at. Qwen3.5's published extension past 262144 is static YaRN
 * (factor 4, original context 262144), which lands on this engine's
 * position cap of 1<<20 (1048576). YaRN is opt-in: applying it changes
 * short-context scores, so a file with no rope.scaling metadata keeps
 * the unscaled angles.
 *
 * The mix matches the Hugging Face YaRN used by the Qwen cards:
 * low pairs keep the original frequency, high pairs use frequency/factor,
 * and cos/sin are multiplied by 0.1*log(factor)+1 unless an attention
 * factor is set.
 */
#ifndef STRATUM_ROPE_H
#define STRATUM_ROPE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STRATUM_ROPE_NONE   0
#define STRATUM_ROPE_LINEAR 1
#define STRATUM_ROPE_YARN   2

#define STRATUM_KV_CAP_MIN 32
#define STRATUM_KV_CAP_MAX (1 << 20)   /* 1048576 positions */
#define STRATUM_KV_CAP_DEFAULT 4096

typedef struct {
    int   type;
    float theta;
    float factor;        /* 1 = no scale */
    float ext_factor;    /* 1 = full YaRN mix, 0 = pure interpolation */
    float attn_factor;   /* >0 uses this mscale; <0 derives it */
    float beta_fast;
    float beta_slow;
    int   orig_ctx;
} StratumRope;

static inline void stratum_rope_defaults(StratumRope* r, float theta) {
    r->type = STRATUM_ROPE_NONE;
    r->theta = theta;
    r->factor = 1.0f;
    r->ext_factor = 1.0f;
    r->attn_factor = -1.0f;
    r->beta_fast = 32.0f;
    r->beta_slow = 1.0f;
    r->orig_ctx = 0;
}

/* context_length 0 means the GGUF omitted it. env NULL or empty or <=0
 * leaves the file value (or 4096) in place. */
static inline int stratum_resolve_kv_cap(int context_length, const char* env) {
    int mk = context_length > 0 ? context_length : STRATUM_KV_CAP_DEFAULT;
    if (env && env[0]) {
        int v = atoi(env);
        if (v > 0) mk = v;
    }
    if (mk < STRATUM_KV_CAP_MIN) mk = STRATUM_KV_CAP_MIN;
    if (mk > STRATUM_KV_CAP_MAX) mk = STRATUM_KV_CAP_MAX;
    return mk;
}

/* Fill orig/factor once the reserved cap is known. An explicit factor
 * above 1 is left alone. Yarn with no factor and a cap above the
 * original context uses cap/orig. */
static inline void stratum_rope_bind_context(StratumRope* r, int context_length, int maxkv) {
    if (r->type != STRATUM_ROPE_YARN) return;
    if (r->orig_ctx <= 0)
        r->orig_ctx = context_length > 0 ? context_length : maxkv;
    if (!(r->factor > 1.0f) && r->orig_ctx > 0 && maxkv > r->orig_ctx)
        r->factor = (float)maxkv / (float)r->orig_ctx;
}

static inline float stratum_yarn_corr_dim(float dim, float n_rot, float base, float orig) {
    return (dim * logf(orig / (n_rot * 2.0f * 3.14159265358979323846f)))
         / (2.0f * logf(base));
}

static inline float stratum_rope_inv_freq(const StratumRope* r, int pair, int n_pairs) {
    float dim = (float)(n_pairs * 2);
    float base = 1.0f / powf(r->theta, (float)(2 * pair) / dim);
    if (r->type == STRATUM_ROPE_NONE || !(r->factor > 1.0f) || n_pairs < 1)
        return base;
    if (r->type == STRATUM_ROPE_LINEAR)
        return base / r->factor;
    if (r->orig_ctx <= 0 || !(r->theta > 1.0f) || !(r->beta_fast > 0.0f) || !(r->beta_slow > 0.0f))
        return base;
    float low = floorf(stratum_yarn_corr_dim(dim, r->beta_fast, r->theta, (float)r->orig_ctx));
    float high = ceilf(stratum_yarn_corr_dim(dim, r->beta_slow, r->theta, (float)r->orig_ctx));
    if (low < 0.0f) low = 0.0f;
    if (high > dim - 1.0f) high = dim - 1.0f;
    float denom = high - low;
    if (denom < 0.001f) denom = 0.001f;
    float y = ((float)pair - low) / denom;
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    float ramp = (1.0f - y) * r->ext_factor;
    if (ramp < 0.0f) ramp = 0.0f;
    if (ramp > 1.0f) ramp = 1.0f;
    float extra = base;
    float inter = base / r->factor;
    return inter * (1.0f - ramp) + extra * ramp;
}

static inline float stratum_rope_mscale(const StratumRope* r) {
    if (r->type != STRATUM_ROPE_YARN || !(r->factor > 1.0f)) return 1.0f;
    if (r->attn_factor > 0.0f) return r->attn_factor;
    return 0.1f * logf(r->factor) + 1.0f;
}

/* True when angles differ from the unscaled powf path. */
static inline int stratum_rope_scales(const StratumRope* r) {
    return r && r->type != STRATUM_ROPE_NONE && (r->factor > 1.0f);
}

/* YaRN multiplies every angle, including position 0. A request that never
 * leaves the trained length keeps the unscaled powf path, so its scores
 * match a run with scaling off. A request that will pass orig_ctx uses the
 * scaled frequencies from the first token, one frequency set for the whole
 * sequence. reach is n_prompt + n_gen (positions 0 .. reach-1). */
static inline int stratum_rope_hold_unscaled(const StratumRope* r, int n_prompt, int n_gen) {
    if (!stratum_rope_scales(r) || r->orig_ctx <= 0) return 0;
    int reach = n_prompt + n_gen;
    if (reach < 0) return 0;
    return reach <= r->orig_ctx;
}

static inline void stratum_rope_reach_report(const StratumRope* r, int n_prompt, int n_gen) {
    if (!stratum_rope_scales(r) || r->orig_ctx <= 0) return;
    int reach = n_prompt + n_gen;
    if (reach <= r->orig_ctx)
        fprintf(stderr, "  rope: yarn held (sequence %d <= orig %d, unscaled angles)\n",
                reach, r->orig_ctx);
    else
        fprintf(stderr, "  rope: yarn on (sequence %d > orig %d)\n",
                reach, r->orig_ctx);
}

/* GPU rope kernels recompute 1/theta^(2i/dim) and the full-GPU KV slab is
 * an eager allocation. Past this cap, or with scaling on, the forward
 * stays on the CPU cache (lazy commit, scaled angles). */
#define STRATUM_GPU_ROPE_MAX_KV 8192
static inline int stratum_gpu_full_ok(const StratumRope* r, int maxkv) {
    return !stratum_rope_scales(r) && maxkv > 0 && maxkv <= STRATUM_GPU_ROPE_MAX_KV;
}

static inline void stratum_rope_cs(const StratumRope* r, int pair, int n_pairs,
                                  int position, float* c, float* s) {
    float angle = (float)position * stratum_rope_inv_freq(r, pair, n_pairs);
    *c = cosf(angle);
    *s = sinf(angle);
    if (r->type == STRATUM_ROPE_YARN && r->factor > 1.0f) {
        float m = stratum_rope_mscale(r);
        *c *= m;
        *s *= m;
    }
}

static inline void stratum_rope_report(const StratumRope* r, int context_length, int maxkv) {
    const char* name = r->type == STRATUM_ROPE_YARN ? "yarn" :
                       r->type == STRATUM_ROPE_LINEAR ? "linear" : "none";
    fprintf(stderr, "  rope: %s factor=%.6g orig=%d mscale=%.6g\n",
            name, r->factor, r->orig_ctx, stratum_rope_mscale(r));
    if (r->type == STRATUM_ROPE_NONE && context_length > 0 && maxkv > context_length) {
        fprintf(stderr,
                "  rope: KV capacity %d exceeds context_length %d with unscaled RoPE. "
                "STRATUM_ROPE_SCALING=yarn STRATUM_YARN_ORIG=%d extends it "
                "(factor defaults to capacity/orig, cap %d).\n",
                maxkv, context_length, context_length, STRATUM_KV_CAP_MAX);
    }
}

#endif
