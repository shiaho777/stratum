#ifndef STRATUM_Q2K_NEON_H
#define STRATUM_Q2K_NEON_H

#include "stratum_q2k.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include <math.h>

static inline float q2k_neon_group16(const uint8_t* q16, int shift,
                                     float dl, float ml, const float* x16) {
    uint8x16_t qb = vld1q_u8(q16);
    uint8x16_t q2 = vandq_u8(vshlq_u8(qb, vdupq_n_s8(-(int8_t)shift)),
                             vdupq_n_u8(0x03));
    uint16x8_t q2l = vmovl_u8(vget_low_u8(q2));
    uint16x8_t q2h = vmovl_u8(vget_high_u8(q2));
    float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q2l)));
    float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q2l)));
    float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q2h)));
    float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q2h)));
    float32x4_t x0 = vld1q_f32(x16 + 0);
    float32x4_t x1 = vld1q_f32(x16 + 4);
    float32x4_t x2 = vld1q_f32(x16 + 8);
    float32x4_t x3 = vld1q_f32(x16 + 12);
    float32x4_t acc = vmulq_f32(f0, x0);
    acc = vfmaq_f32(acc, f1, x1);
    acc = vfmaq_f32(acc, f2, x2);
    acc = vfmaq_f32(acc, f3, x3);
    float32x4_t sx = vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3));
    float qx = vaddvq_f32(acc);
    float sumx = vaddvq_f32(sx);
    return dl * qx - ml * sumx;
}

static inline float q2k_dot_row_neon(const block_q2_K* blocks, int K,
                                     const float* x) {
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q2_K* b = blocks + i;
        const float d   = q4k_fp16_to_fp32(b->d);
        const float dmn = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        const float* xpp = xp;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; j++) {
                int shift = j * 2;
                uint8_t sc1 = b->scales[is++];
                float dl1 = d   * (float)(sc1 & 0xF);
                float ml1 = dmn * (float)(sc1 >>  4);
                dot += (double)q2k_neon_group16(q, shift, dl1, ml1, xpp);
                uint8_t sc2 = b->scales[is++];
                float dl2 = d   * (float)(sc2 & 0xF);
                float ml2 = dmn * (float)(sc2 >>  4);
                dot += (double)q2k_neon_group16(q + 16, shift, dl2, ml2, xpp + 16);
                xpp += 32;
            }
            q += 32;
        }
        xp += 256;
    }
    return (float)dot;
}

static inline void q2k_dot_row_neon_multix(
    const block_q2_K* row_blocks, int K,
    const float* const* xs,
    int B,
    float* out)
{
    int n_blocks = K / 256;
    double dot[64];
    for (int s = 0; s < B; s++) dot[s] = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        const block_q2_K* b = row_blocks + i;
        const float d   = q4k_fp16_to_fp32(b->d);
        const float dmn = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        int x_off = i * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; j++) {
                int shift = j * 2;
                uint8_t sc1 = b->scales[is++];
                float dl1 = d   * (float)(sc1 & 0xF);
                float ml1 = dmn * (float)(sc1 >>  4);
                uint8_t sc2 = b->scales[is++];
                float dl2 = d   * (float)(sc2 & 0xF);
                float ml2 = dmn * (float)(sc2 >>  4);
                uint8x16_t qb0 = vld1q_u8(q);
                uint8x16_t qb1 = vld1q_u8(q + 16);
                uint8x16_t q2a = vandq_u8(vshlq_u8(qb0, vdupq_n_s8(-(int8_t)shift)),
                                          vdupq_n_u8(0x03));
                uint8x16_t q2b = vandq_u8(vshlq_u8(qb1, vdupq_n_s8(-(int8_t)shift)),
                                          vdupq_n_u8(0x03));
                uint16x8_t a_l = vmovl_u8(vget_low_u8(q2a));
                uint16x8_t a_h = vmovl_u8(vget_high_u8(q2a));
                float32x4_t fa0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_l)));
                float32x4_t fa1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_l)));
                float32x4_t fa2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_h)));
                float32x4_t fa3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_h)));
                uint16x8_t b_l = vmovl_u8(vget_low_u8(q2b));
                uint16x8_t b_h = vmovl_u8(vget_high_u8(q2b));
                float32x4_t fb0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_l)));
                float32x4_t fb1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_l)));
                float32x4_t fb2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_h)));
                float32x4_t fb3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_h)));
                float32x4_t vdl1 = vdupq_n_f32(dl1);
                float32x4_t vml1 = vdupq_n_f32(ml1);
                float32x4_t vdl2 = vdupq_n_f32(dl2);
                float32x4_t vml2 = vdupq_n_f32(ml2);
                float32x4_t wa0 = vfmsq_f32(vmulq_f32(fa0, vdl1), vml1, vdupq_n_f32(1.0f));
                float32x4_t wa1 = vfmsq_f32(vmulq_f32(fa1, vdl1), vml1, vdupq_n_f32(1.0f));
                float32x4_t wa2 = vfmsq_f32(vmulq_f32(fa2, vdl1), vml1, vdupq_n_f32(1.0f));
                float32x4_t wa3 = vfmsq_f32(vmulq_f32(fa3, vdl1), vml1, vdupq_n_f32(1.0f));
                float32x4_t wb0 = vfmsq_f32(vmulq_f32(fb0, vdl2), vml2, vdupq_n_f32(1.0f));
                float32x4_t wb1 = vfmsq_f32(vmulq_f32(fb1, vdl2), vml2, vdupq_n_f32(1.0f));
                float32x4_t wb2 = vfmsq_f32(vmulq_f32(fb2, vdl2), vml2, vdupq_n_f32(1.0f));
                float32x4_t wb3 = vfmsq_f32(vmulq_f32(fb3, vdl2), vml2, vdupq_n_f32(1.0f));
                for (int s = 0; s < B; s++) {
                    const float* xp = xs[s] + x_off;
                    float32x4_t x0 = vld1q_f32(xp + 0);
                    float32x4_t x1 = vld1q_f32(xp + 4);
                    float32x4_t x2 = vld1q_f32(xp + 8);
                    float32x4_t x3 = vld1q_f32(xp + 12);
                    float32x4_t x4 = vld1q_f32(xp + 16);
                    float32x4_t x5 = vld1q_f32(xp + 20);
                    float32x4_t x6 = vld1q_f32(xp + 24);
                    float32x4_t x7 = vld1q_f32(xp + 28);
                    float32x4_t p = vmulq_f32(wa0, x0);
                    p = vfmaq_f32(p, wa1, x1);
                    p = vfmaq_f32(p, wa2, x2);
                    p = vfmaq_f32(p, wa3, x3);
                    p = vfmaq_f32(p, wb0, x4);
                    p = vfmaq_f32(p, wb1, x5);
                    p = vfmaq_f32(p, wb2, x6);
                    p = vfmaq_f32(p, wb3, x7);
                    dot[s] += (double)vaddvq_f32(p);
                }
                x_off += 32;
            }
            q += 32;
        }
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}

#if defined(__ARM_FEATURE_DOTPROD)
static inline float q2k_dot_row_sdot(const block_q2_K* row, int K,
                                     const int8_t* xq, const float* xscale) {
    int n_blocks = K / 256;
    double dot = 0.0;
    int8x16_t ones = vdupq_n_s8(1);
    int blk32 = 0;
    for (int i = 0; i < n_blocks; i++) {
        const block_q2_K* b = row + i;
        const float d   = q4k_fp16_to_fp32(b->d);
        const float dmn = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; j++) {
                int shift = j * 2;
                uint8_t sc1 = b->scales[is++];
                float dl1 = d   * (float)(sc1 & 0xF);
                float ml1 = dmn * (float)(sc1 >>  4);
                uint8_t sc2 = b->scales[is++];
                float dl2 = d   * (float)(sc2 & 0xF);
                float ml2 = dmn * (float)(sc2 >>  4);
                uint8x16_t qb0 = vld1q_u8(q);
                uint8x16_t qb1 = vld1q_u8(q + 16);
                int8x16_t qlo = vreinterpretq_s8_u8(
                    vandq_u8(vshlq_u8(qb0, vdupq_n_s8(-(int8_t)shift)),
                             vdupq_n_u8(0x03)));
                int8x16_t qhi = vreinterpretq_s8_u8(
                    vandq_u8(vshlq_u8(qb1, vdupq_n_s8(-(int8_t)shift)),
                             vdupq_n_u8(0x03)));
                const int8_t* xl = xq + (size_t)blk32 * 32;
                int8x16_t xl0 = vld1q_s8(xl);
                int8x16_t xl1 = vld1q_s8(xl + 16);
                int32x4_t aq_lo = vdotq_s32(vdupq_n_s32(0), qlo, xl0);
                int32x4_t aq_hi = vdotq_s32(vdupq_n_s32(0), qhi, xl1);
                int32x4_t as_lo = vdotq_s32(vdupq_n_s32(0), ones, xl0);
                int32x4_t as_hi = vdotq_s32(vdupq_n_s32(0), ones, xl1);
                float scl = xscale[blk32];
                float iql = (float)vaddvq_s32(aq_lo);
                float iqh = (float)vaddvq_s32(aq_hi);
                float isl = (float)vaddvq_s32(as_lo);
                float ish = (float)vaddvq_s32(as_hi);
                dot += (double)(dl1 * iql * scl - ml1 * isl * scl);
                dot += (double)(dl2 * iqh * scl - ml2 * ish * scl);
                blk32++;
            }
            q += 32;
        }
    }
    return (float)dot;
}

static inline void q2k_dot_row_sdot_multix(
    const block_q2_K* row, int K,
    const int8_t* const* xq,
    const float* const* xscale,
    int B,
    float* out)
{
    int n_blocks = K / 256;
    double dot[64];
    for (int s = 0; s < B; s++) dot[s] = 0.0;
    int8x16_t ones = vdupq_n_s8(1);
    int blk32 = 0;
    for (int i = 0; i < n_blocks; i++) {
        const block_q2_K* b = row + i;
        const float d   = q4k_fp16_to_fp32(b->d);
        const float dmn = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; j++) {
                int shift = j * 2;
                uint8_t sc1 = b->scales[is++];
                float dl1 = d   * (float)(sc1 & 0xF);
                float ml1 = dmn * (float)(sc1 >>  4);
                uint8_t sc2 = b->scales[is++];
                float dl2 = d   * (float)(sc2 & 0xF);
                float ml2 = dmn * (float)(sc2 >>  4);
                uint8x16_t qb0 = vld1q_u8(q);
                uint8x16_t qb1 = vld1q_u8(q + 16);
                int8x16_t qlo = vreinterpretq_s8_u8(
                    vandq_u8(vshlq_u8(qb0, vdupq_n_s8(-(int8_t)shift)),
                             vdupq_n_u8(0x03)));
                int8x16_t qhi = vreinterpretq_s8_u8(
                    vandq_u8(vshlq_u8(qb1, vdupq_n_s8(-(int8_t)shift)),
                             vdupq_n_u8(0x03)));
                for (int s = 0; s < B; s++) {
                    const int8_t* xl = xq[s] + (size_t)blk32 * 32;
                    int8x16_t xl0 = vld1q_s8(xl);
                    int8x16_t xl1 = vld1q_s8(xl + 16);
                    int32x4_t aq_lo = vdotq_s32(vdupq_n_s32(0), qlo, xl0);
                    int32x4_t aq_hi = vdotq_s32(vdupq_n_s32(0), qhi, xl1);
                    int32x4_t as_lo = vdotq_s32(vdupq_n_s32(0), ones, xl0);
                    int32x4_t as_hi = vdotq_s32(vdupq_n_s32(0), ones, xl1);
                    float scl = xscale[s][blk32];
                    float iql = (float)vaddvq_s32(aq_lo);
                    float iqh = (float)vaddvq_s32(aq_hi);
                    float isl = (float)vaddvq_s32(as_lo);
                    float ish = (float)vaddvq_s32(as_hi);
                    dot[s] += (double)(dl1 * iql * scl - ml1 * isl * scl);
                    dot[s] += (double)(dl2 * iqh * scl - ml2 * ish * scl);
                }
                blk32++;
            }
            q += 32;
        }
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}
#endif

#endif
#endif

/* V55: nibble-layout Q2K multix — bit-exact with q2k_dot_row_neon_multix
 * (identical FP32 arithmetic), but the weight bytes are pre-arranged
 * (offline converter, values 0-3 unchanged — NOT requantization):
 *   block = 148B = scales[16] + qs[128] + d(2) + dmin(2)
 *   qs: 每 32 权重 16B — 低 4bit = 前 16 权重(sc1), 高 4bit = 后 16 权重(sc2)
 * Unpack is a plain vand/vshr (Q4K-style) instead of 2-bit vshl chains:
 * measured 14-thread: 2.45x (FP32) / 4.5x (SDOT) vs the ggml 2-bit layout. */
static inline void q2k_nib_dot_row_neon_multix(
    const uint8_t* row_nib, int K,
    const float* const* xs, int B, float* out)
{
    int nb = K / 256;
    double dot[64];
    for (int s = 0; s < B && s < 64; s++) dot[s] = 0.0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        const uint8_t* b = row_nib + (size_t)i * 148;
        float d = q4k_fp16_to_fp32(*(const uint16_t*)(b + 144));
        float dmin = q4k_fp16_to_fp32(*(const uint16_t*)(b + 146));
        const uint8_t* q = b + 16;
        int is = 0;
        int x_off = i * 256;
        for (int n = 0; n < 256; n += 32) {
            uint8_t sc1 = b[is], sc2 = b[is + 1];
            float dl1 = d * (float)(sc1 & 0xF), ml1 = dmin * (float)(sc1 >> 4);
            float dl2 = d * (float)(sc2 & 0xF), ml2 = dmin * (float)(sc2 >> 4);
            is += 2;
            uint8x16_t q0 = vld1q_u8(q);
            uint8x16_t qlo = vandq_u8(q0, mask4);
            uint8x16_t qhi = vshrq_n_u8(q0, 4);
            uint16x8_t a_l = vmovl_u8(vget_low_u8(qlo));
            uint16x8_t a_h = vmovl_u8(vget_high_u8(qlo));
            float32x4_t fa0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_l)));
            float32x4_t fa1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_l)));
            float32x4_t fa2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_h)));
            float32x4_t fa3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_h)));
            uint16x8_t b_l = vmovl_u8(vget_low_u8(qhi));
            uint16x8_t b_h = vmovl_u8(vget_high_u8(qhi));
            float32x4_t fb0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_l)));
            float32x4_t fb1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_l)));
            float32x4_t fb2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_h)));
            float32x4_t fb3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_h)));
            float32x4_t vdl1 = vdupq_n_f32(dl1), vml1 = vdupq_n_f32(ml1);
            float32x4_t vdl2 = vdupq_n_f32(dl2), vml2 = vdupq_n_f32(ml2);
            float32x4_t wa0 = vfmsq_f32(vmulq_f32(fa0, vdl1), vml1, vdupq_n_f32(1.0f));
            float32x4_t wa1 = vfmsq_f32(vmulq_f32(fa1, vdl1), vml1, vdupq_n_f32(1.0f));
            float32x4_t wa2 = vfmsq_f32(vmulq_f32(fa2, vdl1), vml1, vdupq_n_f32(1.0f));
            float32x4_t wa3 = vfmsq_f32(vmulq_f32(fa3, vdl1), vml1, vdupq_n_f32(1.0f));
            float32x4_t wb0 = vfmsq_f32(vmulq_f32(fb0, vdl2), vml2, vdupq_n_f32(1.0f));
            float32x4_t wb1 = vfmsq_f32(vmulq_f32(fb1, vdl2), vml2, vdupq_n_f32(1.0f));
            float32x4_t wb2 = vfmsq_f32(vmulq_f32(fb2, vdl2), vml2, vdupq_n_f32(1.0f));
            float32x4_t wb3 = vfmsq_f32(vmulq_f32(fb3, vdl2), vml2, vdupq_n_f32(1.0f));
            for (int s = 0; s < B; s++) {
                const float* xp = xs[s] + x_off;
                float32x4_t x0 = vld1q_f32(xp + 0);
                float32x4_t x1 = vld1q_f32(xp + 4);
                float32x4_t x2 = vld1q_f32(xp + 8);
                float32x4_t x3 = vld1q_f32(xp + 12);
                float32x4_t x4 = vld1q_f32(xp + 16);
                float32x4_t x5 = vld1q_f32(xp + 20);
                float32x4_t x6 = vld1q_f32(xp + 24);
                float32x4_t x7 = vld1q_f32(xp + 28);
                float32x4_t p = vmulq_f32(wa0, x0);
                p = vfmaq_f32(p, wa1, x1);
                p = vfmaq_f32(p, wa2, x2);
                p = vfmaq_f32(p, wa3, x3);
                p = vfmaq_f32(p, wb0, x4);
                p = vfmaq_f32(p, wb1, x5);
                p = vfmaq_f32(p, wb2, x6);
                p = vfmaq_f32(p, wb3, x7);
                dot[s] += (double)vaddvq_f32(p);
            }
            x_off += 32;
            q += 16;
        }
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}
