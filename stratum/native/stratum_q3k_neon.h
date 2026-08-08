
#ifndef STRATUM_Q3K_NEON_H
#define STRATUM_Q3K_NEON_H

#include "stratum_q3k.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

static inline float q3k_neon_sub16(const uint8_t* q16, const uint8_t* hm16,
                                   int shift, uint8_t m, const float* x16) {
    uint8x16_t qb   = vld1q_u8(q16);

    uint8x16_t low2 = vandq_u8(vshlq_u8(qb, vdupq_n_s8(-(int8_t)shift)),
                               vdupq_n_u8(0x03));
    uint8x16_t hmb  = vld1q_u8(hm16);
    uint8x16_t hset = vandq_u8(hmb, vdupq_n_u8(m));
    uint8x16_t clr  = vceqq_u8(hset, vdupq_n_u8(0));
    uint8x16_t sub4 = vandq_u8(clr, vdupq_n_u8(0x04));
    int8x16_t  q3   = vsubq_s8(vreinterpretq_s8_u8(low2),
                              vreinterpretq_s8_u8(sub4));

    int16x8_t q3l = vmovl_s8(vget_low_s8(q3));
    int16x8_t q3h = vmovl_s8(vget_high_s8(q3));
    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q3l)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q3l)));
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q3h)));
    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q3h)));

    float32x4_t x0 = vld1q_f32(x16 + 0);
    float32x4_t x1 = vld1q_f32(x16 + 4);
    float32x4_t x2 = vld1q_f32(x16 + 8);
    float32x4_t x3 = vld1q_f32(x16 + 12);

    float32x4_t acc = vmulq_f32(f0, x0);
    acc = vfmaq_f32(acc, f1, x1);
    acc = vfmaq_f32(acc, f2, x2);
    acc = vfmaq_f32(acc, f3, x3);
    return vaddvq_f32(acc);
}

static inline float q3k_dot_row_neon(const block_q3_K* blocks, int K,
                                     const float* x) {
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q3_K* b = blocks + i;
        const float d_all = q4k_fp16_to_fp32(b->d);
        int8_t scales[16];
        q3k_unpack_scales(b->scales, scales);
        const uint8_t* hm = b->hmask;
        const uint8_t* q  = b->qs;
        uint8_t m = 1;
        int is = 0;
        const float* xpp = xp;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; j++) {
                int shift = j * 2;
                float accA = q3k_neon_sub16(q,      hm,      shift, m, xpp);
                dot += (double)(d_all * (float)(scales[is++] - 32) * accA);
                float accB = q3k_neon_sub16(q + 16, hm + 16, shift, m, xpp + 16);
                dot += (double)(d_all * (float)(scales[is++] - 32) * accB);
                m <<= 1;
                xpp += 32;
            }
            q += 32;
        }
        xp += 256;
    }
    return (float)dot;
}

#endif
#endif
