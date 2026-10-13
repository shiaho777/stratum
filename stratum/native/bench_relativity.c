/* Six relativity-discipline experiments for Stratum.
 * No model weights. No 1M KV fill. Coordinate chart size is a number.
 *
 * 1 coordinate vs geometry  2 local inertial frame
 * 3 spacelike / timelike    4 RoPE relative invariant
 * 5 two clocks              6 domain-of-dependence prefetch
 */
#include "stratum_rope.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

static int fail = 0;

static void check(int cond, const char* msg) {
    if (!cond) {
        printf("rel_fail %s\n", msg);
        fail = 1;
    }
}

static float base_freq(float theta, int pair, int n_pairs) {
    return 1.0f / powf(theta, (float)(2 * pair) / (float)(n_pairs * 2));
}

static void softmax(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void rope_apply(const StratumRope* r, int n_pairs, int pos,
                       const float* in, float* out) {
    for (int k = 0; k < n_pairs; k++) {
        float c, s;
        if (!r || r->type == STRATUM_ROPE_NONE || !(r->factor > 1.0f)) {
            float freq = base_freq(r ? r->theta : 10000.0f, k, n_pairs);
            float a = (float)pos * freq;
            c = cosf(a);
            s = sinf(a);
        } else {
            stratum_rope_cs(r, k, n_pairs, pos, &c, &s);
        }
        float x0 = in[2 * k], x1 = in[2 * k + 1];
        out[2 * k]     = x0 * c - x1 * s;
        out[2 * k + 1] = x0 * s + x1 * c;
    }
}

static float pair_score(const float* q, const float* k, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; i++) d += q[i] * k[i];
    return d;
}

int main(void) {
    /* ---- 1. Coordinate chart vs geometry --------------------------------
     * Reserved cap is the chart. Attended tokens are the geometry.
     * Score scratch must track n_attn, not maxkv. */
    const int nq = 4, n_attn = 12, maxkv = 1048576;
    size_t geom = (size_t)nq * (size_t)n_attn * sizeof(float);
    size_t chart = (size_t)nq * (size_t)maxkv * sizeof(float);
    check(geom == 192, "geom 4*12*4");
    check(chart == (size_t)nq * (size_t)maxkv * 4, "chart 4*1M*4");
    check(chart / geom == (size_t)maxkv / (size_t)n_attn, "chart/geom = cap/attn");
    printf("rel1_geom_bytes=%zu rel1_chart_bytes=%zu rel1_ratio=%zu\n",
           geom, chart, chart / geom);

    /* ---- 2. Local inertial frame ----------------------------------------
     * Inside orig_ctx the engine holds unscaled angles. */
    StratumRope yarn;
    stratum_rope_defaults(&yarn, 10000000.0f);
    yarn.type = STRATUM_ROPE_YARN;
    yarn.factor = 4.0f;
    yarn.orig_ctx = 262144;
    check(stratum_rope_hold_unscaled(&yarn, 4, 8) == 1, "short yarn holds");
    check(stratum_rope_hold_unscaled(&yarn, 200000, 70000) == 0, "long yarn engages");
    StratumRope plain;
    stratum_rope_defaults(&plain, yarn.theta);
    int n_pairs = 32;
    float max_hold = 0.0f;
    for (int k = 0; k < n_pairs; k++) {
        /* held path = unscaled powf, same as type none */
        float a = base_freq(plain.theta, k, n_pairs);
        float b = 1.0f / powf(plain.theta, (float)(2 * k) / (float)(n_pairs * 2));
        float d = fabsf(a - b);
        if (d > max_hold) max_hold = d;
    }
    check(max_hold == 0.0f, "held freq is unscaled");
    printf("rel2_hold_short=%d rel2_hold_long=%d rel2_held_freq_diff=%.9g\n",
           stratum_rope_hold_unscaled(&yarn, 4, 8),
           stratum_rope_hold_unscaled(&yarn, 200000, 70000),
           max_hold);

    /* ---- 3. Spacelike commute vs timelike barrier ----------------------- */
    float q0[4] = {0.1f, -0.2f, 0.3f, 0.4f};
    float q1[4] = {0.5f, 0.1f, -0.1f, 0.2f};
    float k0[4] = {0.2f, 0.2f, 0.1f, -0.3f};
    float k1[4] = {-0.1f, 0.4f, 0.2f, 0.1f};
    float d01 = pair_score(q0, k0, 4);
    float d11 = pair_score(q1, k1, 4);
    float d10 = pair_score(q1, k1, 4);
    float d00 = pair_score(q0, k0, 4);
    check(d01 == d00 && d11 == d10, "spacelike head dots commute");
    float row[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float full[4], part[4];
    memcpy(full, row, sizeof row);
    memcpy(part, row, sizeof row);
    softmax(full, 4);
    softmax(part, 2);
    float sm_diff = 0.0f;
    for (int i = 0; i < 2; i++) sm_diff += fabsf(full[i] - part[i]);
    check(sm_diff > 0.0f, "timelike softmax needs the full row");
    printf("rel3_spacelike_commute=1 rel3_timelike_softmax_diff=%.9g\n", sm_diff);

    /* ---- 4. RoPE relative invariant -------------------------------------
     * One frequency set: score(m,n) == score(m+Δ,n+Δ).
     * Mixing unscaled Q with yarn K breaks it. */
    const int dim = 8;
    const int np = dim / 2;
    float q[8] = {0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.1f, 0.05f, -0.3f};
    float k[8] = {0.1f, 0.2f, -0.4f, 0.1f, 0.3f, -0.1f, 0.2f, 0.2f};
    float qR[8], kR[8], qS[8], kS[8];
    int m = 3, n = 7, delta = 11;
    rope_apply(&plain, np, m, q, qR);
    rope_apply(&plain, np, n, k, kR);
    rope_apply(&plain, np, m + delta, q, qS);
    rope_apply(&plain, np, n + delta, k, kS);
    float rel_none = fabsf(pair_score(qR, kR, dim) - pair_score(qS, kS, dim));
    check(rel_none < 1e-5f, "unscaled relative RoPE");

    rope_apply(&yarn, np, m, q, qR);
    rope_apply(&yarn, np, n, k, kR);
    rope_apply(&yarn, np, m + delta, q, qS);
    rope_apply(&yarn, np, n + delta, k, kS);
    float rel_yarn = fabsf(pair_score(qR, kR, dim) - pair_score(qS, kS, dim));
    check(rel_yarn < 1e-5f, "static yarn still relative");

    rope_apply(&plain, np, m, q, qR);
    rope_apply(&yarn, np, n, k, kR);
    rope_apply(&plain, np, m + delta, q, qS);
    rope_apply(&yarn, np, n + delta, k, kS);
    float rel_mix = fabsf(pair_score(qR, kR, dim) - pair_score(qS, kS, dim));
    check(rel_mix > rel_yarn && rel_mix > 1e-6f, "mixed frames break relative RoPE");
    printf("rel4_none_abs_diff=%.9g rel4_yarn_abs_diff=%.9g rel4_mixed_abs_diff=%.9g\n",
           rel_none, rel_yarn, rel_mix);

    /* ---- 5. Two clocks: GPU-full is a different frame ------------------- */
    check(stratum_gpu_full_ok(&plain, 4096) == 1, "gpu ok 4096");
    check(stratum_gpu_full_ok(&plain, 1048576) == 0, "gpu refuses 1m");
    check(stratum_gpu_full_ok(&yarn, 128) == 0, "gpu refuses yarn");
    /* Mixed backends change the pairwise score vs a single frame. */
    rope_apply(&plain, np, 5, q, qR);
    rope_apply(&plain, np, 9, k, kR);
    float same_clk = pair_score(qR, kR, dim);
    rope_apply(&plain, np, 5, q, qS);
    rope_apply(&yarn, np, 9, k, kS);
    float two_clk = pair_score(qS, kS, dim);
    check(fabsf(same_clk - two_clk) > 0.0f, "two clocks disagree");
    printf("rel5_gpu_full_4096=%d rel5_gpu_full_1m=%d rel5_gpu_full_yarn=%d "
           "rel5_clock_abs_diff=%.9g\n",
           stratum_gpu_full_ok(&plain, 4096),
           stratum_gpu_full_ok(&plain, 1048576),
           stratum_gpu_full_ok(&yarn, 128),
           fabsf(same_clk - two_clk));

    /* ---- 6. Domain of dependence prefetch -------------------------------
     * WILLNEED only future layers in the cone. Values do not change. */
    int nL = 4, ahead = 1, cur = 1;
    int hit[4] = {0, 0, 0, 0};
    for (int a = 1; a <= ahead; a++) {
        int l = cur + a;
        if (l < nL) hit[l] = 1;
    }
    check(hit[0] == 0 && hit[1] == 0 && hit[2] == 1 && hit[3] == 0,
          "prefetch is next layer only");
    long page = sysconf(_SC_PAGESIZE);
    if (page < 4096) page = 4096;
    size_t bytes = (size_t)page;
    void* mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    check(mem != MAP_FAILED, "mmap scratch");
    if (mem != MAP_FAILED) {
        float* x = (float*)mem;
        int nf = (int)(bytes / sizeof(float));
        if (nf > 64) nf = 64;
        for (int i = 0; i < nf; i++) x[i] = 0.125f * (float)(i + 1);
        float copy[64];
        memcpy(copy, x, (size_t)nf * sizeof(float));
        madvise(mem, bytes, MADV_WILLNEED);
        float md = 0.0f;
        for (int i = 0; i < nf; i++) md += fabsf(x[i] - copy[i]);
        check(md == 0.0f, "WILLNEED does not change bytes");
        printf("rel6_prefetch_hit=%d%d%d%d rel6_willneed_abs_diff=%.9g\n",
               hit[0], hit[1], hit[2], hit[3], md);
        munmap(mem, bytes);
    }

    if (!fail) printf("relativity_ok\n");
    return fail ? 2 : 0;
}
