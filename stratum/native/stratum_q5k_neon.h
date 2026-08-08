
#ifndef STRATUM_Q5K_NEON_H
#define STRATUM_Q5K_NEON_H

#include "stratum_q5k.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

static inline float q5k_neon_chunk_dot(
    const uint8_t* qs,
    const uint8_t* qh32,
    const float* xp,
    float d_sc1, float dmin_m1,
    float d_sc2, float dmin_m2,
    int shift)
{
    uint8x16_t b0 = vld1q_u8(qs);
    uint8x16_t b1 = vld1q_u8(qs + 16);
    uint8x16_t h0 = vld1q_u8(qh32);
    uint8x16_t h1 = vld1q_u8(qh32 + 16);
    uint8x16_t mask4 = vdupq_n_u8(0x0F);

    uint8x16_t lo0 = vandq_u8(b0, mask4);
    uint8x16_t lo1 = vandq_u8(b1, mask4);

    uint8x16_t hi0 = vshrq_n_u8(b0, 4);
    uint8x16_t hi1 = vshrq_n_u8(b1, 4);

    uint8x16_t m1 = vdupq_n_u8(1u);
    uint8x16_t m2 = vdupq_n_u8(2u);
    uint8x16_t hbit_lo0, hbit_lo1, hbit_hi0, hbit_hi1;

    int8x16_t neg_shift = vdupq_n_s8(-shift);
    uint8x16_t h0_shr = vshlq_u8(h0, neg_shift);
    uint8x16_t h1_shr = vshlq_u8(h1, neg_shift);
    hbit_lo0 = vshlq_n_u8(vandq_u8(h0_shr, m1), 4);
    hbit_lo1 = vshlq_n_u8(vandq_u8(h1_shr, m1), 4);
    hbit_hi0 = vshlq_n_u8(vshrq_n_u8(vandq_u8(h0_shr, m2), 1), 4);
    hbit_hi1 = vshlq_n_u8(vshrq_n_u8(vandq_u8(h1_shr, m2), 1), 4);

    uint8x16_t lo0c = vorrq_u8(lo0, hbit_lo0);
    uint8x16_t lo1c = vorrq_u8(lo1, hbit_lo1);
    uint8x16_t hi0c = vorrq_u8(hi0, hbit_hi0);
    uint8x16_t hi1c = vorrq_u8(hi1, hbit_hi1);

    #define W4(byte_vec, lane) ({                                            \
        uint16x8_t w16_l = vmovl_u8(vget_low_u8(byte_vec));                  \
        uint16x8_t w16_h = vmovl_u8(vget_high_u8(byte_vec));                 \
        uint32x4_t w32_a = vmovl_u16(vget_low_u16(w16_l));                   \
        uint32x4_t w32_b = vmovl_u16(vget_high_u16(w16_l));                  \
        uint32x4_t w32_c = vmovl_u16(vget_low_u16(w16_h));                   \
        uint32x4_t w32_d = vmovl_u16(vget_high_u16(w16_h));                  \
        (lane)[0] = vcvtq_f32_u32(w32_a);                                    \
        (lane)[1] = vcvtq_f32_u32(w32_b);                                    \
        (lane)[2] = vcvtq_f32_u32(w32_c);                                    \
        (lane)[3] = vcvtq_f32_u32(w32_d);                                    \
    })

    float32x4_t lo_f0[4], lo_f1[4], hi_f0[4], hi_f1[4];
    W4(lo0c, lo_f0);
    W4(lo1c, lo_f1);
    W4(hi0c, hi_f0);
    W4(hi1c, hi_f1);
    #undef W4

    float32x4_t xl0 = vld1q_f32(xp +  0);
    float32x4_t xl1 = vld1q_f32(xp +  4);
    float32x4_t xl2 = vld1q_f32(xp +  8);
    float32x4_t xl3 = vld1q_f32(xp + 12);
    float32x4_t xl4 = vld1q_f32(xp + 16);
    float32x4_t xl5 = vld1q_f32(xp + 20);
    float32x4_t xl6 = vld1q_f32(xp + 24);
    float32x4_t xl7 = vld1q_f32(xp + 28);
    float32x4_t xh0 = vld1q_f32(xp + 32);
    float32x4_t xh1 = vld1q_f32(xp + 36);
    float32x4_t xh2 = vld1q_f32(xp + 40);
    float32x4_t xh3 = vld1q_f32(xp + 44);
    float32x4_t xh4 = vld1q_f32(xp + 48);
    float32x4_t xh5 = vld1q_f32(xp + 52);
    float32x4_t xh6 = vld1q_f32(xp + 56);
    float32x4_t xh7 = vld1q_f32(xp + 60);

    float32x4_t qx_lo = vmulq_f32(lo_f0[0], xl0);
    qx_lo = vfmaq_f32(qx_lo, lo_f0[1], xl1);
    qx_lo = vfmaq_f32(qx_lo, lo_f0[2], xl2);
    qx_lo = vfmaq_f32(qx_lo, lo_f0[3], xl3);
    qx_lo = vfmaq_f32(qx_lo, lo_f1[0], xl4);
    qx_lo = vfmaq_f32(qx_lo, lo_f1[1], xl5);
    qx_lo = vfmaq_f32(qx_lo, lo_f1[2], xl6);
    qx_lo = vfmaq_f32(qx_lo, lo_f1[3], xl7);

    float32x4_t qx_hi = vmulq_f32(hi_f0[0], xh0);
    qx_hi = vfmaq_f32(qx_hi, hi_f0[1], xh1);
    qx_hi = vfmaq_f32(qx_hi, hi_f0[2], xh2);
    qx_hi = vfmaq_f32(qx_hi, hi_f0[3], xh3);
    qx_hi = vfmaq_f32(qx_hi, hi_f1[0], xh4);
    qx_hi = vfmaq_f32(qx_hi, hi_f1[1], xh5);
    qx_hi = vfmaq_f32(qx_hi, hi_f1[2], xh6);
    qx_hi = vfmaq_f32(qx_hi, hi_f1[3], xh7);

    float32x4_t xs_lo = vaddq_f32(vaddq_f32(vaddq_f32(xl0, xl1), vaddq_f32(xl2, xl3)),
                                  vaddq_f32(vaddq_f32(xl4, xl5), vaddq_f32(xl6, xl7)));
    float32x4_t xs_hi = vaddq_f32(vaddq_f32(vaddq_f32(xh0, xh1), vaddq_f32(xh2, xh3)),
                                  vaddq_f32(vaddq_f32(xh4, xh5), vaddq_f32(xh6, xh7)));

    float qx_lo_s = vaddvq_f32(qx_lo);
    float qx_hi_s = vaddvq_f32(qx_hi);
    float xs_lo_s = vaddvq_f32(xs_lo);
    float xs_hi_s = vaddvq_f32(xs_hi);

    return d_sc1 * qx_lo_s - dmin_m1 * xs_lo_s
         + d_sc2 * qx_hi_s - dmin_m2 * xs_hi_s;
}

static inline float q5k_dot_row_neon(
    const block_q5_K* row_blocks, int K, const float* x)
{
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q5_K* b = row_blocks + i;
        const float d    = q4k_fp16_to_fp32(b->d);
        const float dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* ql = b->qs;
        const uint8_t* qh = b->qh;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1b, sc2, m2b;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1b);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2b);
            float d_sc1   = d * (float)sc1;
            float dmin_m1 = dmin * (float)m1b;
            float d_sc2   = d * (float)sc2;
            float dmin_m2 = dmin * (float)m2b;

            dot += (double)q5k_neon_chunk_dot(ql, qh, xp + j,
                                              d_sc1, dmin_m1, d_sc2, dmin_m2, is);
            ql += 32;
            is += 2;
        }
        xp += 256;
    }
    return (float)dot;
}

#endif

#endif
