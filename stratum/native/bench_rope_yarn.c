/* Shipped YaRN and the KV cap.
 * Qwen3.5 native context is 262144. The published extension to the
 * engine cap (1048576, about 1M) is static YaRN with factor 4 and
 * original context 262144, theta 1e7, rotary dim 64.
 * No model file and no KV allocation.
 */
#include "stratum_rope.h"
#include <math.h>
#include <stdio.h>

static int fail = 0;

static void check(int cond, const char* msg) {
    if (!cond) {
        printf("rope_fail %s\n", msg);
        fail = 1;
    }
}

static float base_freq(float theta, int pair, int n_pairs) {
    return 1.0f / powf(theta, (float)(2 * pair) / (float)(n_pairs * 2));
}

int main(void) {
    const int n_pairs = 32;
    const float theta = 10000000.0f;

    StratumRope plain;
    stratum_rope_defaults(&plain, theta);
    float max_none = 0.0f;
    for (int k = 0; k < n_pairs; k++) {
        float d = fabsf(stratum_rope_inv_freq(&plain, k, n_pairs) - base_freq(theta, k, n_pairs));
        if (d > max_none) max_none = d;
    }
    check(max_none == 0.0f, "none freq");
    check(stratum_rope_mscale(&plain) == 1.0f, "none mscale");
    printf("rope_none_max_abs_diff=%.9g\n", max_none);

    StratumRope yarn;
    stratum_rope_defaults(&yarn, theta);
    yarn.type = STRATUM_ROPE_YARN;
    yarn.factor = 4.0f;
    yarn.orig_ctx = 262144;
    float f0 = stratum_rope_inv_freq(&yarn, 0, n_pairs);
    float fL = stratum_rope_inv_freq(&yarn, n_pairs - 1, n_pairs);
    float b0 = base_freq(theta, 0, n_pairs);
    float bL = base_freq(theta, n_pairs - 1, n_pairs);
    check(f0 == b0, "yarn pair0 stays extrapolated");
    check(fL == bL / 4.0f, "yarn last pair is interpolated");
    float mid = stratum_rope_inv_freq(&yarn, 18, n_pairs);
    float bM = base_freq(theta, 18, n_pairs);
    check(mid > bM / 4.0f && mid < bM, "yarn middle pair is mixed");
    float mscale = stratum_rope_mscale(&yarn);
    float want_m = 0.1f * logf(4.0f) + 1.0f;
    check(mscale == want_m, "yarn mscale");
    /* Published Qwen3.5 extension: theta 1e7, rotary dim 64, orig 262144.
     * corr(beta_fast=32) floors to 14, corr(beta_slow=1) ceils to 22. */
    float low = floorf(stratum_yarn_corr_dim(64.0f, 32.0f, theta, 262144.0f));
    float high = ceilf(stratum_yarn_corr_dim(64.0f, 1.0f, theta, 262144.0f));
    check(low == 14.0f && high == 22.0f, "qwen corr range 14..22");
    printf("rope_yarn_corr_low=%.0f rope_yarn_corr_high=%.0f\n", low, high);
    printf("rope_yarn_pair0_diff=%.9g\n", fabsf(f0 - b0));
    printf("rope_yarn_last_diff=%.9g\n", fabsf(fL - bL / 4.0f));
    printf("rope_yarn_mscale=%.9g\n", mscale);

    float c, s;
    /* Position 0 still scales every rotated component by mscale, which is
     * why YaRN stays opt-in: short-context scores move. */
    stratum_rope_cs(&yarn, 0, n_pairs, 0, &c, &s);
    check(fabsf(c - mscale) < 1e-6f && fabsf(s) < 1e-6f, "pos0 mscale");
    stratum_rope_cs(&yarn, 0, n_pairs, 1048575, &c, &s);
    check(isfinite(c) && isfinite(s), "pos 1048575 pair0");
    stratum_rope_cs(&yarn, n_pairs - 1, n_pairs, 1048575, &c, &s);
    check(isfinite(c) && isfinite(s), "pos 1048575 last");
    printf("rope_pos_1m_finite=%d\n", fail ? 0 : 1);

    StratumRope lin;
    stratum_rope_defaults(&lin, theta);
    lin.type = STRATUM_ROPE_LINEAR;
    lin.factor = 4.0f;
    float max_lin = 0.0f;
    for (int k = 0; k < n_pairs; k++) {
        float d = fabsf(stratum_rope_inv_freq(&lin, k, n_pairs) - base_freq(theta, k, n_pairs) / 4.0f);
        if (d > max_lin) max_lin = d;
    }
    check(max_lin == 0.0f, "linear");
    printf("rope_linear_max_abs_diff=%.9g\n", max_lin);

    StratumRope auto_y;
    stratum_rope_defaults(&auto_y, theta);
    auto_y.type = STRATUM_ROPE_YARN;
    stratum_rope_bind_context(&auto_y, 262144, 1048576);
    check(auto_y.orig_ctx == 262144, "auto orig");
    check(auto_y.factor == (float)1048576 / 262144.0f, "auto factor 4");

    check(stratum_resolve_kv_cap(0, NULL) == 4096, "default 4096");
    check(stratum_resolve_kv_cap(131072, NULL) == 131072, "128k");
    check(stratum_resolve_kv_cap(262144, NULL) == 262144, "256k");
    check(stratum_resolve_kv_cap(262144, "1048576") == 1048576, "1m");
    check(stratum_resolve_kv_cap(262144, "2000000") == 1048576, "clamp");
    check(stratum_resolve_kv_cap(0, "16") == 32, "floor");
    check(!stratum_rope_scales(&plain), "none does not scale");
    check(stratum_rope_scales(&yarn), "yarn scales");
    /* reach = n_prompt + n_gen. Inside orig the engine keeps powf angles.
     * Past orig it applies YaRN from token 0. */
    check(stratum_rope_hold_unscaled(&plain, 4, 8) == 0, "none does not hold");
    check(stratum_rope_hold_unscaled(&yarn, 4, 8) == 1, "yarn holds inside orig");
    StratumRope past = yarn;
    past.orig_ctx = 4;
    check(stratum_rope_hold_unscaled(&past, 4, 8) == 0, "yarn engages past orig");
    printf("rope_hold_inside=%d rope_hold_past=%d\n",
           stratum_rope_hold_unscaled(&yarn, 4, 8),
           stratum_rope_hold_unscaled(&past, 4, 8));
    check(stratum_gpu_full_ok(&plain, 4096), "gpu full at 4096");
    check(stratum_gpu_full_ok(&plain, 8192), "gpu full at 8192");
    check(!stratum_gpu_full_ok(&plain, 8193), "gpu full refuses 8193");
    check(!stratum_gpu_full_ok(&plain, 1048576), "gpu full refuses 1m");
    check(!stratum_gpu_full_ok(&yarn, 128), "gpu full refuses yarn");
    printf("rope_gpu_full_4096=%d rope_gpu_full_1m=%d rope_gpu_full_yarn=%d\n",
           stratum_gpu_full_ok(&plain, 4096),
           stratum_gpu_full_ok(&plain, 1048576),
           stratum_gpu_full_ok(&yarn, 128));
    printf("rope_kv_default=%d rope_kv_128k=%d rope_kv_256k=%d rope_kv_1m=%d\n",
           stratum_resolve_kv_cap(0, NULL),
           stratum_resolve_kv_cap(131072, NULL),
           stratum_resolve_kv_cap(262144, NULL),
           stratum_resolve_kv_cap(262144, "1048576"));

    if (!fail) printf("rope_yarn_ok\n");
    return fail ? 2 : 0;
}
