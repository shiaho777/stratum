
#ifndef STRATUM_Q8_0_H
#define STRATUM_Q8_0_H

#include "stratum_q4k.h"
#include <stdint.h>

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} block_q8_0;
_Static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

static inline void q8_0_dequant_block_scalar(const block_q8_0* b, float* y) {
    float d = q4k_fp16_to_fp32(b->d);
    for (int k = 0; k < 32; k++) y[k] = d * (float)b->qs[k];
}

static inline float q8_0_dot_row_scalar(const block_q8_0* blocks, int K,
                                        const float* x) {
    int n_blocks = K / 32;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q8_0* b = blocks + i;
        float d = q4k_fp16_to_fp32(b->d);
        float acc = 0.0f;
        for (int k = 0; k < 32; k++) acc += (float)b->qs[k] * xp[k];
        dot += (double)(d * acc);
        xp += 32;
    }
    return (float)dot;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

static inline float q8_0_dot_row_neon(const block_q8_0* blocks, int K,
                                      const float* x) {
    int n_blocks = K / 32;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q8_0* b = blocks + i;
        float d = q4k_fp16_to_fp32(b->d);

        int8x16_t q0 = vld1q_s8(b->qs);
        int8x16_t q1 = vld1q_s8(b->qs + 16);

        int16x8_t q0_l = vmovl_s8(vget_low_s8(q0));
        int16x8_t q0_h = vmovl_s8(vget_high_s8(q0));
        int16x8_t q1_l = vmovl_s8(vget_low_s8(q1));
        int16x8_t q1_h = vmovl_s8(vget_high_s8(q1));

        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_l)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_l)));
        float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_h)));
        float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_h)));
        float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_l)));
        float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_l)));
        float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_h)));
        float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_h)));

        float32x4_t x0 = vld1q_f32(xp +  0);
        float32x4_t x1 = vld1q_f32(xp +  4);
        float32x4_t x2 = vld1q_f32(xp +  8);
        float32x4_t x3 = vld1q_f32(xp + 12);
        float32x4_t x4 = vld1q_f32(xp + 16);
        float32x4_t x5 = vld1q_f32(xp + 20);
        float32x4_t x6 = vld1q_f32(xp + 24);
        float32x4_t x7 = vld1q_f32(xp + 28);

        float32x4_t acc = vmulq_f32(f0, x0);
        acc = vfmaq_f32(acc, f1, x1);
        acc = vfmaq_f32(acc, f2, x2);
        acc = vfmaq_f32(acc, f3, x3);
        acc = vfmaq_f32(acc, f4, x4);
        acc = vfmaq_f32(acc, f5, x5);
        acc = vfmaq_f32(acc, f6, x6);
        acc = vfmaq_f32(acc, f7, x7);

        dot += (double)(d * vaddvq_f32(acc));
        xp += 32;
    }
    return (float)dot;
}

#endif

#endif
