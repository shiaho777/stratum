/* Weighted vector accumulate. The 16-wide body matches the previous
 * llama single-token PV loop. The 4-wide tail writes every lane: the
 * old tail wrote only index d and then stepped by 4, so a head dim that
 * is not a multiple of 16 dropped three lanes.
 */
#ifndef STRATUM_AXPY_H
#define STRATUM_AXPY_H

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

static inline void stratum_axpy_f32(float* acc, float p, const float* v, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vp = vdupq_n_f32(p);
    int d = 0;
    for (; d + 16 <= n; d += 16) {
        vst1q_f32(acc + d,      vfmaq_f32(vld1q_f32(acc + d),      vp, vld1q_f32(v + d)));
        vst1q_f32(acc + d + 4,  vfmaq_f32(vld1q_f32(acc + d + 4),  vp, vld1q_f32(v + d + 4)));
        vst1q_f32(acc + d + 8,  vfmaq_f32(vld1q_f32(acc + d + 8),  vp, vld1q_f32(v + d + 8)));
        vst1q_f32(acc + d + 12, vfmaq_f32(vld1q_f32(acc + d + 12), vp, vld1q_f32(v + d + 12)));
    }
    for (; d + 4 <= n; d += 4)
        vst1q_f32(acc + d, vfmaq_f32(vld1q_f32(acc + d), vp, vld1q_f32(v + d)));
    for (; d < n; d++) acc[d] += p * v[d];
#else
    for (int d = 0; d < n; d++) acc[d] += p * v[d];
#endif
}

#endif
