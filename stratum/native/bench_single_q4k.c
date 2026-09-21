/* Single-stream Q4_K kernel restructure probe.
 * Hypothesis: q4k_dot_row_sdot is serial-latency-bound (one double acc,
 * 2 vaddvq per 64 elems all feeding a single chain). Variant splits the
 * accumulation into 4 independent double accumulators.
 * Compare: correctness (max diff) + single-thread timing on real tensor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* v2: 4 parallel double accumulators + deferred vaddvq pairs */
static inline float q4k_dot_row_sdot_v2(const block_q4_K* row, int K,
                                        const int8_t* xq, const float* xscale,
                                        const int32_t* xsum) {
    int nb = K / 256;
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        const block_q4_K* b = row + i;
        float dd = q4k_fp16_to_fp32(b->d), dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2);
            uint8x16_t w0 = vld1q_u8(q);
            uint8x16_t w1 = vld1q_u8(q + 16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            const int8_t* xl = xq + (size_t)blk32 * 32;
            const int8_t* xh = xq + (size_t)(blk32 + 1) * 32;
            int8x16_t xl0 = vld1q_s8(xl), xl1 = vld1q_s8(xl + 16);
            int8x16_t xh0 = vld1q_s8(xh), xh1 = vld1q_s8(xh + 16);
            int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0), lo1, xl1);
            int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0), hi1, xh1);
            int32_t p_lo = vaddvq_s32(aq_lo);
            int32_t p_hi = vaddvq_s32(aq_hi);
            float scl_lo = xscale[blk32], scl_hi = xscale[blk32 + 1];
            /* same term structure, but fanned into independent chains */
            double t_lo = (double)dd * sc1 * ((double)p_lo * scl_lo)
                        - (double)dmin * m1 * ((double)xsum[blk32] * scl_lo);
            double t_hi = (double)dd * sc2 * ((double)p_hi * scl_hi)
                        - (double)dmin * m2 * ((double)xsum[blk32 + 1] * scl_hi);
            switch (is & 3) {
                case 0: d0 += t_lo; d1 += t_hi; break;
                case 2: d2 += t_lo; d3 += t_hi; break;
                case 4: d0 += t_lo; d1 += t_hi; break;
                default: d2 += t_lo; d3 += t_hi; break;
            }
            q += 32; is += 2; blk32 += 2;
        }
    }
    return (float)((d0 + d1) + (d2 + d3));
}

/* v3: also int32-accumulate the whole 256-block when scales are uniform
 * within the block? No — scales differ per 32-group, can't merge.
 * Instead: keep per-group scaling but accumulate in float per block-pair
 * to shorten chain (NOT bit-exact, argmax-risk — measure diff). */
static inline float q4k_dot_row_sdot_v3(const block_q4_K* row, int K,
                                        const int8_t* xq, const float* xscale,
                                        const int32_t* xsum) {
    int nb = K / 256;
    float32x4_t facc = vdupq_n_f32(0.0f);
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        const block_q4_K* b = row + i;
        float d = q4k_fp16_to_fp32(b->d), dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2);
            uint8x16_t w0 = vld1q_u8(q);
            uint8x16_t w1 = vld1q_u8(q + 16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            const int8_t* xl = xq + (size_t)blk32 * 32;
            const int8_t* xh = xq + (size_t)(blk32 + 1) * 32;
            int8x16_t xl0 = vld1q_s8(xl), xl1 = vld1q_s8(xl + 16);
            int8x16_t xh0 = vld1q_s8(xh), xh1 = vld1q_s8(xh + 16);
            int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0), lo1, xl1);
            int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0), hi1, xh1);
            float32x4_t coeff = { d * (float)sc1 * xscale[blk32],
                                  -dmin * (float)m1 * xscale[blk32],
                                  d * (float)sc2 * xscale[blk32 + 1],
                                  -dmin * (float)m2 * xscale[blk32 + 1] };
            int32x4_t terms = { vaddvq_s32(aq_lo), xsum[blk32],
                                vaddvq_s32(aq_hi), xsum[blk32 + 1] };
            facc = vfmaq_f32(facc, vcvtq_f32_s32(terms), coeff);
            q += 32; is += 2; blk32 += 2;
        }
    }
    return vaddvq_f32(facc);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    const char* wfile = argc > 1 ? argv[1] : "/tmp/wq0.bin";
    const char* xfile = argc > 2 ? argv[2] : "/tmp/xn0.bin";
    int K = 1024, N = 2048;
    FILE* f = fopen(wfile, "rb"); if (!f) { fprintf(stderr, "no W\n"); return 1; }
    block_q4_K* W = malloc((size_t)N * (K / 256) * sizeof(block_q4_K));
    fread(W, 1, (size_t)N * (K / 256) * sizeof(block_q4_K), f); fclose(f);
    f = fopen(xfile, "rb"); if (!f) { fprintf(stderr, "no x\n"); return 1; }
    float* x = malloc(K * sizeof(float));
    fread(x, 4, K, f); fclose(f);

    int nb32 = K / 32;
    int8_t* xq = malloc(K); float* xs = malloc(nb32 * 4);
    int32_t* xsum = malloc(nb32 * 4);
    q4k_quantize_x_q8(x, K, xq, xs);
    for (int g = 0; g < nb32; g++) xsum[g] = q4k_sum_i8_32(xq + (size_t)g * 32);

    float* yA = malloc(N * 4); float* yB = malloc(N * 4); float* yC = malloc(N * 4);
    const block_q4_K* rows = W; int nb = K / 256;

    /* correctness vs current kernel */
    for (int r = 0; r < N; r++) {
        yA[r] = q4k_dot_row_sdot(rows + (size_t)r * nb, K, xq, xs);
        yB[r] = q4k_dot_row_sdot_v2(rows + (size_t)r * nb, K, xq, xs, xsum);
        yC[r] = q4k_dot_row_sdot_v3(rows + (size_t)r * nb, K, xq, xs, xsum);
    }
    double m2 = 0, m3 = 0; int argdiff2 = 0, argdiff3 = 0;
    int amA = 0, amB = 0, amC = 0;
    for (int r = 0; r < N; r++) {
        double e2 = fabs(yA[r] - yB[r]); if (e2 > m2) m2 = e2;
        double e3 = fabs(yA[r] - yC[r]); if (e3 > m3) m3 = e3;
        if (yB[r] > yB[amB]) amB = r;
        if (yC[r] > yC[amC]) amC = r;
        if (yA[r] > yA[amA]) amA = r;
    }
    argdiff2 = (amA != amB); argdiff3 = (amA != amC);
    printf("v2 max|diff|=%.3g argmax_flip=%d   v3 max|diff|=%.3g argmax_flip=%d\n",
           m2, argdiff2, m3, argdiff3);

    /* timing: single thread, R rounds over all N rows */
    int R = 200;
    double t0, best[3] = {1e9, 1e9, 1e9};
    volatile float sink = 0;
    for (int rep = 0; rep < 5; rep++) {
        t0 = now_s();
        for (int it = 0; it < R; it++)
            for (int r = 0; r < N; r++)
                sink += q4k_dot_row_sdot(rows + (size_t)r * nb, K, xq, xs);
        double dt = now_s() - t0; if (dt < best[0]) best[0] = dt;
        t0 = now_s();
        for (int it = 0; it < R; it++)
            for (int r = 0; r < N; r++)
                sink += q4k_dot_row_sdot_v2(rows + (size_t)r * nb, K, xq, xs, xsum);
        dt = now_s() - t0; if (dt < best[1]) best[1] = dt;
        t0 = now_s();
        for (int it = 0; it < R; it++)
            for (int r = 0; r < N; r++)
                sink += q4k_dot_row_sdot_v3(rows + (size_t)r * nb, K, xq, xs, xsum);
        dt = now_s() - t0; if (dt < best[2]) best[2] = dt;
    }
    printf("current  : %7.2f ms/1000rows (%.1f GB/s weight)\n",
           best[0] / R * 1e3, (double)N * nb * sizeof(block_q4_K) * R / best[0] / 1e9);
    printf("v2 4xacc : %7.2f ms/1000rows (%.1f GB/s)  %.2fx\n",
           best[1] / R * 1e3, (double)N * nb * sizeof(block_q4_K) * R / best[1] / 1e9, best[0] / best[1]);
    printf("v3 fvec  : %7.2f ms/1000rows (%.1f GB/s)  %.2fx\n",
           best[2] / R * 1e3, (double)N * nb * sizeof(block_q4_K) * R / best[2] / 1e9, best[0] / best[2]);
    (void)sink;
    return 0;
}
