
#ifndef STRATUM_LIN_ATTN_NEON_H
#define STRATUM_LIN_ATTN_NEON_H

#include <stdint.h>
#include <stddef.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define STRATUM_LIN_HAS_NEON 1
#else
#define STRATUM_LIN_HAS_NEON 0
#endif

static inline void st_lin_attn_dual_gemv_trans(
    const float* S, const float* kt, const float* qt,
    int HK, int HV,
    float* b, float* a)
{
#if STRATUM_LIN_HAS_NEON
    int v;

    for (v = 0; v + 16 <= HV; v += 16) {
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        for (int k = 0; k < HK; k++) {
            const float* row = S + (size_t)k * HV + v;
            float32x4_t s0 = vld1q_f32(row + 0);
            float32x4_t s1 = vld1q_f32(row + 4);
            float32x4_t s2 = vld1q_f32(row + 8);
            float32x4_t s3 = vld1q_f32(row + 12);
            b0 = vfmaq_n_f32(b0, s0, kt[k]);
            b1 = vfmaq_n_f32(b1, s1, kt[k]);
            b2 = vfmaq_n_f32(b2, s2, kt[k]);
            b3 = vfmaq_n_f32(b3, s3, kt[k]);
            a0 = vfmaq_n_f32(a0, s0, qt[k]);
            a1 = vfmaq_n_f32(a1, s1, qt[k]);
            a2 = vfmaq_n_f32(a2, s2, qt[k]);
            a3 = vfmaq_n_f32(a3, s3, qt[k]);
        }
        vst1q_f32(b + v + 0,  b0); vst1q_f32(b + v + 4,  b1);
        vst1q_f32(b + v + 8,  b2); vst1q_f32(b + v + 12, b3);
        vst1q_f32(a + v + 0,  a0); vst1q_f32(a + v + 4,  a1);
        vst1q_f32(a + v + 8,  a2); vst1q_f32(a + v + 12, a3);
    }

    for (; v + 4 <= HV; v += 4) {
        float32x4_t bv = vdupq_n_f32(0.0f), av = vdupq_n_f32(0.0f);
        for (int k = 0; k < HK; k++) {
            float32x4_t s = vld1q_f32(S + (size_t)k * HV + v);
            bv = vfmaq_n_f32(bv, s, kt[k]);
            av = vfmaq_n_f32(av, s, qt[k]);
        }
        vst1q_f32(b + v, bv);
        vst1q_f32(a + v, av);
    }

    for (; v < HV; v++) {
        float bv = 0.0f, av = 0.0f;
        for (int k = 0; k < HK; k++) {
            float s = S[(size_t)k * HV + v];
            bv += s * kt[k];
            av += s * qt[k];
        }
        b[v] = bv;
        a[v] = av;
    }
#else
    for (int v = 0; v < HV; v++) {
        float bv = 0.0f, av = 0.0f;
        for (int k = 0; k < HK; k++) {
            float s = S[(size_t)k * HV + v];
            bv += s * kt[k];
            av += s * qt[k];
        }
        b[v] = bv;
        a[v] = av;
    }
#endif
}

static inline void st_lin_attn_fused_decay_outer(
    float* S, const float* kt, const float* delta,
    int HK, int HV, float gt)
{
#if STRATUM_LIN_HAS_NEON
    float32x4_t gv = vdupq_n_f32(gt);
    for (int k = 0; k < HK; k++) {
        float kk = kt[k];
        float32x4_t kv = vdupq_n_f32(kk);
        float* row = S + (size_t)k * HV;
        int v;
        for (v = 0; v + 16 <= HV; v += 16) {
            float32x4_t s0 = vld1q_f32(row + v + 0);
            float32x4_t s1 = vld1q_f32(row + v + 4);
            float32x4_t s2 = vld1q_f32(row + v + 8);
            float32x4_t s3 = vld1q_f32(row + v + 12);
            float32x4_t d0 = vld1q_f32(delta + v + 0);
            float32x4_t d1 = vld1q_f32(delta + v + 4);
            float32x4_t d2 = vld1q_f32(delta + v + 8);
            float32x4_t d3 = vld1q_f32(delta + v + 12);

            s0 = vfmaq_f32(vmulq_f32(s0, gv), kv, d0);
            s1 = vfmaq_f32(vmulq_f32(s1, gv), kv, d1);
            s2 = vfmaq_f32(vmulq_f32(s2, gv), kv, d2);
            s3 = vfmaq_f32(vmulq_f32(s3, gv), kv, d3);
            vst1q_f32(row + v + 0,  s0);
            vst1q_f32(row + v + 4,  s1);
            vst1q_f32(row + v + 8,  s2);
            vst1q_f32(row + v + 12, s3);
        }
        for (; v + 4 <= HV; v += 4) {
            float32x4_t s = vld1q_f32(row + v);
            float32x4_t d = vld1q_f32(delta + v);
            s = vfmaq_f32(vmulq_f32(s, gv), kv, d);
            vst1q_f32(row + v, s);
        }
        for (; v < HV; v++) {
            row[v] = gt * row[v] + kk * delta[v];
        }
    }
#else
    for (int k = 0; k < HK; k++) {
        float kk = kt[k];
        float* row = S + (size_t)k * HV;
        for (int v = 0; v < HV; v++) {
            row[v] = gt * row[v] + kk * delta[v];
        }
    }
#endif
}

#endif
