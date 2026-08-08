
#ifndef STRATUM_Q4K_NEON_H
#define STRATUM_Q4K_NEON_H

#include "stratum_q4k.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

/* NEON dequant of a full Q4_K row to fp32 (for the BLAS/sgemm batched
 * path). Mirrors q4k_dequant_block_scalar but vectorized: ~3-4x faster
 * than the scalar version, which is what made the first BLAS attempt
 * lose. y must hold K floats. */
static inline void q4k_dequant_row_neon(const block_q4_K* blocks, int K, float* y) {
    int nb = K / 256;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        const block_q4_K* b = blocks + i;
        const float d    = q4k_fp16_to_fp32(b->d);
        const float dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        float* yb = y + i * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1b, sc2, m2b;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1b);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2b);
            float d1 = d * (float)sc1, mm1 = dmin * (float)m1b;
            float d2 = d * (float)sc2, mm2 = dmin * (float)m2b;
            float32x4_t vd1 = vdupq_n_f32(d1), vm1 = vdupq_n_f32(mm1);
            float32x4_t vd2 = vdupq_n_f32(d2), vm2 = vdupq_n_f32(mm2);
            uint8x16_t b0 = vld1q_u8(q), b1 = vld1q_u8(q + 16);
            uint8x16_t lo0 = vandq_u8(b0, mask4), lo1 = vandq_u8(b1, mask4);
            uint8x16_t hi0 = vshrq_n_u8(b0, 4),   hi1 = vshrq_n_u8(b1, 4);
            #define DQ(src, vd, vm, base) do { \
                uint16x8_t _l = vmovl_u8(vget_low_u8(src)); \
                uint16x8_t _h = vmovl_u8(vget_high_u8(src)); \
                float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_l))); \
                float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_l))); \
                float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_h))); \
                float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_h))); \
                vst1q_f32(yb+(base)+0,  vsubq_f32(vmulq_f32(f0,vd),vm)); \
                vst1q_f32(yb+(base)+4,  vsubq_f32(vmulq_f32(f1,vd),vm)); \
                vst1q_f32(yb+(base)+8,  vsubq_f32(vmulq_f32(f2,vd),vm)); \
                vst1q_f32(yb+(base)+12, vsubq_f32(vmulq_f32(f3,vd),vm)); \
            } while(0)
            DQ(lo0, vd1, vm1, j + 0);
            DQ(lo1, vd1, vm1, j + 16);
            DQ(hi0, vd2, vm2, j + 32);
            DQ(hi1, vd2, vm2, j + 48);
            #undef DQ
            q += 32; is += 2;
        }
    }
}

static inline float q4k_neon_pair_dot(
    const uint8_t* qs,
    const float*   xp,
    float          d_sc1,
    float          dmin_m1,
    float          d_sc2,
    float          dmin_m2)
{

    uint8x16_t b0 = vld1q_u8(qs);
    uint8x16_t b1 = vld1q_u8(qs + 16);
    uint8x16_t mask4 = vdupq_n_u8(0x0F);

    uint8x16_t lo0 = vandq_u8(b0, mask4);
    uint8x16_t lo1 = vandq_u8(b1, mask4);

    uint8x16_t hi0 = vshrq_n_u8(b0, 4);
    uint8x16_t hi1 = vshrq_n_u8(b1, 4);

    uint16x8_t lo0_l = vmovl_u8(vget_low_u8(lo0));
    uint16x8_t lo0_h = vmovl_u8(vget_high_u8(lo0));
    uint16x8_t lo1_l = vmovl_u8(vget_low_u8(lo1));
    uint16x8_t lo1_h = vmovl_u8(vget_high_u8(lo1));
    float32x4_t lo_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo0_l)));
    float32x4_t lo_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo0_l)));
    float32x4_t lo_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo0_h)));
    float32x4_t lo_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo0_h)));
    float32x4_t lo_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo1_l)));
    float32x4_t lo_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo1_l)));
    float32x4_t lo_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo1_h)));
    float32x4_t lo_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo1_h)));

    uint16x8_t hi0_l = vmovl_u8(vget_low_u8(hi0));
    uint16x8_t hi0_h = vmovl_u8(vget_high_u8(hi0));
    uint16x8_t hi1_l = vmovl_u8(vget_low_u8(hi1));
    uint16x8_t hi1_h = vmovl_u8(vget_high_u8(hi1));
    float32x4_t hi_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi0_l)));
    float32x4_t hi_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi0_l)));
    float32x4_t hi_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi0_h)));
    float32x4_t hi_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi0_h)));
    float32x4_t hi_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi1_l)));
    float32x4_t hi_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi1_l)));
    float32x4_t hi_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi1_h)));
    float32x4_t hi_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi1_h)));

    float32x4_t xl0 = vld1q_f32(xp + 0);
    float32x4_t xl1 = vld1q_f32(xp + 4);
    float32x4_t xl2 = vld1q_f32(xp + 8);
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

    float32x4_t qx_lo = vmulq_f32(lo_f0, xl0);
    qx_lo = vfmaq_f32(qx_lo, lo_f1, xl1);
    qx_lo = vfmaq_f32(qx_lo, lo_f2, xl2);
    qx_lo = vfmaq_f32(qx_lo, lo_f3, xl3);
    qx_lo = vfmaq_f32(qx_lo, lo_f4, xl4);
    qx_lo = vfmaq_f32(qx_lo, lo_f5, xl5);
    qx_lo = vfmaq_f32(qx_lo, lo_f6, xl6);
    qx_lo = vfmaq_f32(qx_lo, lo_f7, xl7);

    float32x4_t qx_hi = vmulq_f32(hi_f0, xh0);
    qx_hi = vfmaq_f32(qx_hi, hi_f1, xh1);
    qx_hi = vfmaq_f32(qx_hi, hi_f2, xh2);
    qx_hi = vfmaq_f32(qx_hi, hi_f3, xh3);
    qx_hi = vfmaq_f32(qx_hi, hi_f4, xh4);
    qx_hi = vfmaq_f32(qx_hi, hi_f5, xh5);
    qx_hi = vfmaq_f32(qx_hi, hi_f6, xh6);
    qx_hi = vfmaq_f32(qx_hi, hi_f7, xh7);

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

static inline float q4k_dot_row_neon(
    const block_q4_K* row_blocks, int K, const float* x)
{
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q4_K* b = row_blocks + i;
        const float d    = q4k_fp16_to_fp32(b->d);
        const float dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1b, sc2, m2b;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1b);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2b);
            float d_sc1   = d * (float)sc1;
            float dmin_m1 = dmin * (float)m1b;
            float d_sc2   = d * (float)sc2;
            float dmin_m2 = dmin * (float)m2b;
            dot += (double)q4k_neon_pair_dot(q, xp, d_sc1, dmin_m1, d_sc2, dmin_m2);
            q += 32;
            xp += 64;
            is += 2;
        }
    }
    return (float)dot;
}
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#include <math.h>

static inline void q4k_quantize_x_q8_1b(const float* xp, int8_t* o, float* scale_out) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t a0 = vabsq_f32(vld1q_f32(xp + 0));
    float32x4_t a1 = vabsq_f32(vld1q_f32(xp + 4));
    float32x4_t a2 = vabsq_f32(vld1q_f32(xp + 8));
    float32x4_t a3 = vabsq_f32(vld1q_f32(xp + 12));
    float32x4_t a4 = vabsq_f32(vld1q_f32(xp + 16));
    float32x4_t a5 = vabsq_f32(vld1q_f32(xp + 20));
    float32x4_t a6 = vabsq_f32(vld1q_f32(xp + 24));
    float32x4_t a7 = vabsq_f32(vld1q_f32(xp + 28));
    float32x4_t m0 = vmaxq_f32(vmaxq_f32(a0, a1), vmaxq_f32(a2, a3));
    float32x4_t m1 = vmaxq_f32(vmaxq_f32(a4, a5), vmaxq_f32(a6, a7));
    float amax = vmaxvq_f32(vmaxq_f32(m0, m1));
    float scale = amax / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    *scale_out = scale;
    float32x4_t vinv = vdupq_n_f32(inv);
    for (int i = 0; i < 32; i += 4) {
        float32x4_t v = vmulq_f32(vld1q_f32(xp + i), vinv);
        int32x4_t iv = vcvtnq_s32_f32(v);
        iv = vmaxq_s32(vdupq_n_s32(-128), vminq_s32(vdupq_n_s32(127), iv));
        o[i+0] = (int8_t)vgetq_lane_s32(iv, 0);
        o[i+1] = (int8_t)vgetq_lane_s32(iv, 1);
        o[i+2] = (int8_t)vgetq_lane_s32(iv, 2);
        o[i+3] = (int8_t)vgetq_lane_s32(iv, 3);
    }
#else
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float v = xp[i] < 0 ? -xp[i] : xp[i];
        if (v > amax) amax = v;
    }
    float scale = amax / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    *scale_out = scale;
    for (int i = 0; i < 32; i++) {
        int iv = (int)lrintf(xp[i] * inv);
        if (iv < -128) iv = -128;
        if (iv > 127) iv = 127;
        o[i] = (int8_t)iv;
    }
#endif
}

static inline void q4k_quantize_x_q8(const float* x, int K,
                                     int8_t* xq, float* xscale) {
    int nb = K / 32;
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (int b = 0; b < nb; b++) {
        const float* xp = x + (size_t)b * 32;
        float32x4_t a0 = vabsq_f32(vld1q_f32(xp + 0));
        float32x4_t a1 = vabsq_f32(vld1q_f32(xp + 4));
        float32x4_t a2 = vabsq_f32(vld1q_f32(xp + 8));
        float32x4_t a3 = vabsq_f32(vld1q_f32(xp + 12));
        float32x4_t a4 = vabsq_f32(vld1q_f32(xp + 16));
        float32x4_t a5 = vabsq_f32(vld1q_f32(xp + 20));
        float32x4_t a6 = vabsq_f32(vld1q_f32(xp + 24));
        float32x4_t a7 = vabsq_f32(vld1q_f32(xp + 28));
        float32x4_t m0 = vmaxq_f32(vmaxq_f32(a0, a1), vmaxq_f32(a2, a3));
        float32x4_t m1 = vmaxq_f32(vmaxq_f32(a4, a5), vmaxq_f32(a6, a7));
        float32x4_t m = vmaxq_f32(m0, m1);
        float amax = vmaxvq_f32(m);
        float scale = amax / 127.0f;
        float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        xscale[b] = scale;
        float32x4_t vinv = vdupq_n_f32(inv);
        int8_t* o = xq + (size_t)b * 32;
        for (int i = 0; i < 32; i += 4) {
            float32x4_t v = vmulq_f32(vld1q_f32(xp + i), vinv);
            int32x4_t iv = vcvtnq_s32_f32(v);
            iv = vmaxq_s32(vdupq_n_s32(-128), vminq_s32(vdupq_n_s32(127), iv));
            o[i+0] = (int8_t)vgetq_lane_s32(iv, 0);
            o[i+1] = (int8_t)vgetq_lane_s32(iv, 1);
            o[i+2] = (int8_t)vgetq_lane_s32(iv, 2);
            o[i+3] = (int8_t)vgetq_lane_s32(iv, 3);
        }
    }
#else
    for (int b = 0; b < nb; b++) {
        const float* xp = x + (size_t)b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) { float a = fabsf(xp[i]); if (a > amax) amax = a; }
        float scale = amax / 127.0f;
        float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        xscale[b] = scale;
        for (int i = 0; i < 32; i++) {
            int v = (int)lrintf(xp[i] * inv);
            if (v > 127) v = 127; if (v < -128) v = -128;
            xq[(size_t)b * 32 + i] = (int8_t)v;
        }
    }
#endif
}

static inline float q4k_dot_row_sdot(const block_q4_K* row, int K,
                                     const int8_t* xq, const float* xscale) {
    int nb = K / 256;
    double dot = 0.0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t ones = vdupq_n_s8(1);
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
            int32x4_t as_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, xl0), ones, xl1);
            int32x4_t as_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, xh0), ones, xh1);
            float scl_lo = xscale[blk32], scl_hi = xscale[blk32 + 1];
            dot += (double)(d * sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo)
                          - dmin * m1 * ((float)vaddvq_s32(as_lo) * scl_lo));
            dot += (double)(d * sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi)
                          - dmin * m2 * ((float)vaddvq_s32(as_hi) * scl_hi));
            q += 32; is += 2; blk32 += 2;
        }
    }
    return (float)dot;
}
#endif

#if defined(__ARM_FEATURE_DOTPROD)
/* V25/V208: SDOT multix — shared weight unpack, B x int8 slots.
 * V208: prefetch next Q4_K block; rows2 reuses each x load across 2 output rows. */

static inline int32_t q4k_sum_i8_32(const int8_t* p) {
    int8x16_t a = vld1q_s8(p);
    int8x16_t b = vld1q_s8(p + 16);
    int8x16_t ones = vdupq_n_s8(1);
    int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, a), ones, b);
    return vaddvq_s32(s);
}

static inline void q4k_dot_row_sdot_multix(
    const block_q4_K* row_blocks, int K,
    const int8_t* const* xq, const float* const* xscale,
    const int32_t* const* xsum,
    int B, float* out)
{
    int nb = K / 256;
    double dot[16];
    for (int s = 0; s < B && s < 16; s++) dot[s] = 0.0;
    if (B > 16) { for (int s = 0; s < B; s++) out[s] = 0.0f; return; }
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t ones = vdupq_n_s8(1);
    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) __builtin_prefetch(row_blocks + i + 1, 0, 3);
        const block_q4_K* b = row_blocks + i;
        float d = q4k_fp16_to_fp32(b->d), dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2);
            float d_sc1 = d * (float)sc1, dmin_m1 = dmin * (float)m1;
            float d_sc2 = d * (float)sc2, dmin_m2 = dmin * (float)m2;
            uint8x16_t w0 = vld1q_u8(q);
            uint8x16_t w1 = vld1q_u8(q + 16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            for (int s = 0; s < B; s++) {
                const int8_t* xl = xq[s] + (size_t)blk32 * 32;
                const int8_t* xh = xq[s] + (size_t)(blk32 + 1) * 32;
                int8x16_t xl0 = vld1q_s8(xl), xl1 = vld1q_s8(xl + 16);
                int8x16_t xh0 = vld1q_s8(xh), xh1 = vld1q_s8(xh + 16);
                int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0), lo1, xl1);
                int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0), hi1, xh1);
                float scl_lo = xscale[s][blk32], scl_hi = xscale[s][blk32 + 1];
                if (xsum) {
                    int32_t s_lo = xsum[s][blk32];
                    int32_t s_hi = xsum[s][blk32 + 1];
                    dot[s] += (double)(d_sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo)
                                - dmin_m1 * ((float)s_lo * scl_lo));
                    dot[s] += (double)(d_sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi)
                                - dmin_m2 * ((float)s_hi * scl_hi));
                } else {
                    int32x4_t as_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, xl0), ones, xl1);
                    int32x4_t as_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, xh0), ones, xh1);
                    dot[s] += (double)(d_sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo)
                                - dmin_m1 * ((float)vaddvq_s32(as_lo) * scl_lo));
                    dot[s] += (double)(d_sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi)
                                - dmin_m2 * ((float)vaddvq_s32(as_hi) * scl_hi));
                }
            }
            q += 32; is += 2; blk32 += 2;
        }
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}

static inline void q4k_dot_row_sdot_pack_b7(
    const block_q4_K* row_blocks, int K,
    const int8_t* xpack, const float* scpack, const int32_t* sumpack,
    float* out)
{
    int nb = K / 256;
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) __builtin_prefetch(row_blocks + i + 1, 0, 3);
        if (i + 2 < nb) __builtin_prefetch(row_blocks + i + 2, 0, 1);
        if (i + 3 < nb) __builtin_prefetch(row_blocks + i + 3, 0, 0);
        const block_q4_K* b = row_blocks + i;
        float d = q4k_fp16_to_fp32(b->d), dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2);
            float d_sc1 = d * (float)sc1, dmin_m1 = dmin * (float)m1;
            float d_sc2 = d * (float)sc2, dmin_m2 = dmin * (float)m2;
            uint8x16_t w0 = vld1q_u8(q);
            uint8x16_t w1 = vld1q_u8(q + 16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            const int8_t* base_lo = xpack + (size_t)blk32 * 7 * 32;
            const int8_t* base_hi = xpack + (size_t)(blk32 + 1) * 7 * 32;
            const float* sc_lo = scpack + (size_t)blk32 * 7;
            const float* sc_hi = scpack + (size_t)(blk32 + 1) * 7;
            const int32_t* sm_lo = sumpack + (size_t)blk32 * 7;
            const int32_t* sm_hi = sumpack + (size_t)(blk32 + 1) * 7;
            if (j + 64 < 256) {
                __builtin_prefetch(xpack + (size_t)(blk32 + 2) * 7 * 32, 0, 3);
                __builtin_prefetch(scpack + (size_t)(blk32 + 2) * 7, 0, 1);
            } else if (i + 1 < nb) {
                __builtin_prefetch(xpack + (size_t)((i + 1) * 8) * 7 * 32, 0, 3);
            }
#define Q4K_B7_SLOT(s, acc) do { \
                int8x16_t xl0 = vld1q_s8(base_lo + (size_t)(s) * 32); \
                int8x16_t xl1 = vld1q_s8(base_lo + (size_t)(s) * 32 + 16); \
                int8x16_t xh0 = vld1q_s8(base_hi + (size_t)(s) * 32); \
                int8x16_t xh1 = vld1q_s8(base_hi + (size_t)(s) * 32 + 16); \
                int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0), lo1, xl1); \
                int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0), hi1, xh1); \
                float scl_lo = sc_lo[(s)], scl_hi = sc_hi[(s)]; \
                float s_lo = (float)sm_lo[(s)], s_hi = (float)sm_hi[(s)]; \
                (acc) += d_sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo) - dmin_m1 * (s_lo * scl_lo); \
                (acc) += d_sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi) - dmin_m2 * (s_hi * scl_hi); \
            } while (0)
            Q4K_B7_SLOT(0, d0);
            Q4K_B7_SLOT(1, d1);
            Q4K_B7_SLOT(2, d2);
            Q4K_B7_SLOT(3, d3);
            Q4K_B7_SLOT(4, d4);
            Q4K_B7_SLOT(5, d5);
            Q4K_B7_SLOT(6, d6);
#undef Q4K_B7_SLOT
            q += 32; is += 2; blk32 += 2;
        }
    }
    out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3;
    out[4]=d4; out[5]=d5; out[6]=d6;
}

static inline void q4k_dot_tile4_sdot_pack_b7(
    const block_q4_K* r0, const block_q4_K* r1,
    const block_q4_K* r2, const block_q4_K* r3, int K,
    const int8_t* xpack, const float* scpack, const int32_t* sumpack,
    float* o0, float* o1, float* o2, float* o3)
{
    int nb = K / 256;
    double a0[7]={0}, a1[7]={0}, a2[7]={0}, a3[7]={0};
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) {
            __builtin_prefetch(r0 + i + 1, 0, 3);
            __builtin_prefetch(r1 + i + 1, 0, 3);
            __builtin_prefetch(r2 + i + 1, 0, 3);
            __builtin_prefetch(r3 + i + 1, 0, 3);
        }
        if (i + 2 < nb) {
            __builtin_prefetch(r0 + i + 2, 0, 1);
            __builtin_prefetch(r1 + i + 2, 0, 1);
        }
        const block_q4_K* bs[4] = { r0 + i, r1 + i, r2 + i, r3 + i };
        double* accs[4] = { a0, a1, a2, a3 };
        int blk32_base = i * 8;
        for (int sub = 0; sub < 4; sub++) {
            int is = sub * 2;
            int blk32 = blk32_base + is;
            const int8_t* base_lo = xpack + (size_t)blk32 * 7 * 32;
            const int8_t* base_hi = xpack + (size_t)(blk32 + 1) * 7 * 32;
            const float* sc_lo = scpack + (size_t)blk32 * 7;
            const float* sc_hi = scpack + (size_t)(blk32 + 1) * 7;
            const int32_t* sm_lo = sumpack + (size_t)blk32 * 7;
            const int32_t* sm_hi = sumpack + (size_t)(blk32 + 1) * 7;
            int8x16_t xl0[7], xl1[7], xh0[7], xh1[7];
            float scl_lo[7], scl_hi[7], s_lo[7], s_hi[7];
            for (int s = 0; s < 7; s++) {
                const int8_t* xl = base_lo + (size_t)s * 32;
                const int8_t* xh = base_hi + (size_t)s * 32;
                xl0[s] = vld1q_s8(xl);
                xl1[s] = vld1q_s8(xl + 16);
                xh0[s] = vld1q_s8(xh);
                xh1[s] = vld1q_s8(xh + 16);
                scl_lo[s] = sc_lo[s];
                scl_hi[s] = sc_hi[s];
                s_lo[s] = (float)sm_lo[s];
                s_hi[s] = (float)sm_hi[s];
            }
            for (int rr = 0; rr < 4; rr++) {
                const block_q4_K* b = bs[rr];
                float d = q4k_fp16_to_fp32(b->d), dmin = q4k_fp16_to_fp32(b->dmin);
                const uint8_t* q = b->qs + (size_t)sub * 32;
                uint8_t sc1, m1, sc2, m2;
                q4k_get_scale_min(is + 0, b->scales, &sc1, &m1);
                q4k_get_scale_min(is + 1, b->scales, &sc2, &m2);
                float d_sc1 = d * (float)sc1, dmin_m1 = dmin * (float)m1;
                float d_sc2 = d * (float)sc2, dmin_m2 = dmin * (float)m2;
                uint8x16_t w0 = vld1q_u8(q);
                uint8x16_t w1 = vld1q_u8(q + 16);
                int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
                int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
                int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                double* acc = accs[rr];
                for (int s = 0; s < 7; s++) {
                    int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0[s]), lo1, xl1[s]);
                    int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0[s]), hi1, xh1[s]);
                    acc[s] += (double)(d_sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo[s])
                                - dmin_m1 * (s_lo[s] * scl_lo[s]));
                    acc[s] += (double)(d_sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi[s])
                                - dmin_m2 * (s_hi[s] * scl_hi[s]));
                }
            }
        }
    }
    for (int s = 0; s < 7; s++) {
        o0[s] = (float)a0[s]; o1[s] = (float)a1[s];
        o2[s] = (float)a2[s]; o3[s] = (float)a3[s];
    }
}

static inline void q4k_dot_rows2_sdot_pack_b7(
    const block_q4_K* row0, const block_q4_K* row1, int K,
    const int8_t* xpack, const float* scpack, const int32_t* sumpack,
    float* out0, float* out1)
{
    int nb = K / 256;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0;
    float b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) {
            __builtin_prefetch(row0 + i + 1, 0, 3);
            __builtin_prefetch(row1 + i + 1, 0, 3);
        }
        if (i + 2 < nb) {
            __builtin_prefetch(row0 + i + 2, 0, 1);
            __builtin_prefetch(row1 + i + 2, 0, 1);
        }
        const block_q4_K* br0 = row0 + i;
        const block_q4_K* br1 = row1 + i;
        float dA = q4k_fp16_to_fp32(br0->d), dmA = q4k_fp16_to_fp32(br0->dmin);
        float dB = q4k_fp16_to_fp32(br1->d), dmB = q4k_fp16_to_fp32(br1->dmin);
        const uint8_t* qA = br0->qs;
        const uint8_t* qB = br1->qs;
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2, sc3, m3, sc4, m4;
            q4k_get_scale_min(is + 0, br0->scales, &sc1, &m1);
            q4k_get_scale_min(is + 1, br0->scales, &sc2, &m2);
            q4k_get_scale_min(is + 0, br1->scales, &sc3, &m3);
            q4k_get_scale_min(is + 1, br1->scales, &sc4, &m4);
            float d_sc1 = dA * (float)sc1, dmin_m1 = dmA * (float)m1;
            float d_sc2 = dA * (float)sc2, dmin_m2 = dmA * (float)m2;
            float d_sc3 = dB * (float)sc3, dmin_m3 = dmB * (float)m3;
            float d_sc4 = dB * (float)sc4, dmin_m4 = dmB * (float)m4;
            uint8x16_t w0 = vld1q_u8(qA), w1 = vld1q_u8(qA + 16);
            uint8x16_t v0 = vld1q_u8(qB), v1 = vld1q_u8(qB + 16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            int8x16_t lo2 = vreinterpretq_s8_u8(vandq_u8(v0, mask4));
            int8x16_t lo3 = vreinterpretq_s8_u8(vandq_u8(v1, mask4));
            int8x16_t hi2 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
            int8x16_t hi3 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
            const int8_t* base_lo = xpack + (size_t)blk32 * 7 * 32;
            const int8_t* base_hi = xpack + (size_t)(blk32 + 1) * 7 * 32;
            const float* sc_lo = scpack + (size_t)blk32 * 7;
            const float* sc_hi = scpack + (size_t)(blk32 + 1) * 7;
            const int32_t* sm_lo = sumpack + (size_t)blk32 * 7;
            const int32_t* sm_hi = sumpack + (size_t)(blk32 + 1) * 7;
            if (j + 64 < 256)
                __builtin_prefetch(xpack + (size_t)(blk32 + 2) * 7 * 32, 0, 3);
#define Q4K_XSS(s, accA, accB) do { \
                int8x16_t xl0 = vld1q_s8(base_lo + (size_t)(s) * 32); \
                int8x16_t xl1 = vld1q_s8(base_lo + (size_t)(s) * 32 + 16); \
                int8x16_t xh0 = vld1q_s8(base_hi + (size_t)(s) * 32); \
                int8x16_t xh1 = vld1q_s8(base_hi + (size_t)(s) * 32 + 16); \
                int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0), lo1, xl1); \
                int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0), hi1, xh1); \
                int32x4_t bq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo2, xl0), lo3, xl1); \
                int32x4_t bq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi2, xh0), hi3, xh1); \
                float scl_lo = sc_lo[(s)], scl_hi = sc_hi[(s)]; \
                float s_lo = (float)sm_lo[(s)], s_hi = (float)sm_hi[(s)]; \
                (accA) += d_sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo) - dmin_m1 * (s_lo * scl_lo); \
                (accA) += d_sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi) - dmin_m2 * (s_hi * scl_hi); \
                (accB) += d_sc3 * ((float)vaddvq_s32(bq_lo) * scl_lo) - dmin_m3 * (s_lo * scl_lo); \
                (accB) += d_sc4 * ((float)vaddvq_s32(bq_hi) * scl_hi) - dmin_m4 * (s_hi * scl_hi); \
            } while (0)
            Q4K_XSS(0, a0, b0); Q4K_XSS(1, a1, b1); Q4K_XSS(2, a2, b2); Q4K_XSS(3, a3, b3);
            Q4K_XSS(4, a4, b4); Q4K_XSS(5, a5, b5); Q4K_XSS(6, a6, b6);
#undef Q4K_XSS
            qA += 32; qB += 32; is += 2; blk32 += 2;
        }
    }
    out0[0]=a0; out0[1]=a1; out0[2]=a2; out0[3]=a3; out0[4]=a4; out0[5]=a5; out0[6]=a6;
    out1[0]=b0; out1[1]=b1; out1[2]=b2; out1[3]=b3; out1[4]=b4; out1[5]=b5; out1[6]=b6;
}

static inline void q4k_dot_rows4_sdot_pack_b7(
    const block_q4_K* row0, const block_q4_K* row1,
    const block_q4_K* row2, const block_q4_K* row3, int K,
    const int8_t* xpack, const float* scpack, const int32_t* sumpack,
    float* o0, float* o1, float* o2, float* o3)
{
    int nb = K / 256;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0;
    float b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0;
    float c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0;
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) {
            __builtin_prefetch(row0 + i + 1, 0, 3);
            __builtin_prefetch(row1 + i + 1, 0, 3);
            __builtin_prefetch(row2 + i + 1, 0, 3);
            __builtin_prefetch(row3 + i + 1, 0, 3);
        }
        if (i + 2 < nb) {
            __builtin_prefetch(row0 + i + 2, 0, 1);
            __builtin_prefetch(row1 + i + 2, 0, 1);
        }
        const block_q4_K* br[4] = { row0 + i, row1 + i, row2 + i, row3 + i };
        float dA[4], dmA[4];
        const uint8_t* qs[4];
        for (int r = 0; r < 4; r++) {
            dA[r] = q4k_fp16_to_fp32(br[r]->d);
            dmA[r] = q4k_fp16_to_fp32(br[r]->dmin);
            qs[r] = br[r]->qs;
        }
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc[4][2], mn[4][2];
            for (int r = 0; r < 4; r++) {
                q4k_get_scale_min(is + 0, br[r]->scales, &sc[r][0], &mn[r][0]);
                q4k_get_scale_min(is + 1, br[r]->scales, &sc[r][1], &mn[r][1]);
            }
            float d_sc[4][2], dmin_m[4][2];
            for (int r = 0; r < 4; r++) {
                d_sc[r][0] = dA[r] * (float)sc[r][0];
                dmin_m[r][0] = dmA[r] * (float)mn[r][0];
                d_sc[r][1] = dA[r] * (float)sc[r][1];
                dmin_m[r][1] = dmA[r] * (float)mn[r][1];
            }
            int8x16_t lo[4], hi[4], loh[4], hih[4];
            for (int r = 0; r < 4; r++) {
                uint8x16_t w0 = vld1q_u8(qs[r]);
                uint8x16_t w1 = vld1q_u8(qs[r] + 16);
                lo[r]  = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
                loh[r] = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
                hi[r]  = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                hih[r] = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            }
            const int8_t* base_lo = xpack + (size_t)blk32 * 7 * 32;
            const int8_t* base_hi = xpack + (size_t)(blk32 + 1) * 7 * 32;
            const float* sc_lo = scpack + (size_t)blk32 * 7;
            const float* sc_hi = scpack + (size_t)(blk32 + 1) * 7;
            const int32_t* sm_lo = sumpack + (size_t)blk32 * 7;
            const int32_t* sm_hi = sumpack + (size_t)(blk32 + 1) * 7;
            if (j + 64 < 256)
                __builtin_prefetch(xpack + (size_t)(blk32 + 2) * 7 * 32, 0, 3);
#define Q4K_XSS4(s, A0, A1, A2, A3) do { \
                int8x16_t xl0 = vld1q_s8(base_lo + (size_t)(s) * 32); \
                int8x16_t xl1 = vld1q_s8(base_lo + (size_t)(s) * 32 + 16); \
                int8x16_t xh0 = vld1q_s8(base_hi + (size_t)(s) * 32); \
                int8x16_t xh1 = vld1q_s8(base_hi + (size_t)(s) * 32 + 16); \
                float scl_lo = sc_lo[(s)], scl_hi = sc_hi[(s)]; \
                float s_lo = (float)sm_lo[(s)], s_hi = (float)sm_hi[(s)]; \
                int32x4_t q0l = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[0], xl0), loh[0], xl1); \
                int32x4_t q0h = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi[0], xh0), hih[0], xh1); \
                int32x4_t q1l = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[1], xl0), loh[1], xl1); \
                int32x4_t q1h = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi[1], xh0), hih[1], xh1); \
                int32x4_t q2l = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[2], xl0), loh[2], xl1); \
                int32x4_t q2h = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi[2], xh0), hih[2], xh1); \
                int32x4_t q3l = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[3], xl0), loh[3], xl1); \
                int32x4_t q3h = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi[3], xh0), hih[3], xh1); \
                (A0) += d_sc[0][0] * ((float)vaddvq_s32(q0l) * scl_lo) - dmin_m[0][0] * (s_lo * scl_lo); \
                (A0) += d_sc[0][1] * ((float)vaddvq_s32(q0h) * scl_hi) - dmin_m[0][1] * (s_hi * scl_hi); \
                (A1) += d_sc[1][0] * ((float)vaddvq_s32(q1l) * scl_lo) - dmin_m[1][0] * (s_lo * scl_lo); \
                (A1) += d_sc[1][1] * ((float)vaddvq_s32(q1h) * scl_hi) - dmin_m[1][1] * (s_hi * scl_hi); \
                (A2) += d_sc[2][0] * ((float)vaddvq_s32(q2l) * scl_lo) - dmin_m[2][0] * (s_lo * scl_lo); \
                (A2) += d_sc[2][1] * ((float)vaddvq_s32(q2h) * scl_hi) - dmin_m[2][1] * (s_hi * scl_hi); \
                (A3) += d_sc[3][0] * ((float)vaddvq_s32(q3l) * scl_lo) - dmin_m[3][0] * (s_lo * scl_lo); \
                (A3) += d_sc[3][1] * ((float)vaddvq_s32(q3h) * scl_hi) - dmin_m[3][1] * (s_hi * scl_hi); \
            } while (0)
            Q4K_XSS4(0, a0, b0, c0, d0);
            Q4K_XSS4(1, a1, b1, c1, d1);
            Q4K_XSS4(2, a2, b2, c2, d2);
            Q4K_XSS4(3, a3, b3, c3, d3);
            Q4K_XSS4(4, a4, b4, c4, d4);
            Q4K_XSS4(5, a5, b5, c5, d5);
            Q4K_XSS4(6, a6, b6, c6, d6);
#undef Q4K_XSS4
            for (int r = 0; r < 4; r++) qs[r] += 32;
            is += 2; blk32 += 2;
        }
    }
    o0[0]=a0; o0[1]=a1; o0[2]=a2; o0[3]=a3; o0[4]=a4; o0[5]=a5; o0[6]=a6;
    o1[0]=b0; o1[1]=b1; o1[2]=b2; o1[3]=b3; o1[4]=b4; o1[5]=b5; o1[6]=b6;
    o2[0]=c0; o2[1]=c1; o2[2]=c2; o2[3]=c3; o2[4]=c4; o2[5]=c5; o2[6]=c6;
    o3[0]=d0; o3[1]=d1; o3[2]=d2; o3[3]=d3; o3[4]=d4; o3[5]=d5; o3[6]=d6;
}

static inline void q4k_dot_row_sdot_multix_pack(
    const block_q4_K* row_blocks, int K,
    const int8_t* xpack, const float* scpack, const int32_t* sumpack,
    int B, float* out)
{
    if (B == 7) { q4k_dot_row_sdot_pack_b7(row_blocks, K, xpack, scpack, sumpack, out); return; }
    int nb = K / 256;
    double dot[16];
    for (int s = 0; s < B && s < 16; s++) dot[s] = 0.0;
    if (B > 16) { for (int s = 0; s < B; s++) out[s] = 0.0f; return; }
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) __builtin_prefetch(row_blocks + i + 1, 0, 3);
        const block_q4_K* b = row_blocks + i;
        float d = q4k_fp16_to_fp32(b->d), dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2);
            float d_sc1 = d * (float)sc1, dmin_m1 = dmin * (float)m1;
            float d_sc2 = d * (float)sc2, dmin_m2 = dmin * (float)m2;
            uint8x16_t w0 = vld1q_u8(q);
            uint8x16_t w1 = vld1q_u8(q + 16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            const int8_t* base_lo = xpack + (size_t)blk32 * (size_t)B * 32;
            const int8_t* base_hi = xpack + (size_t)(blk32 + 1) * (size_t)B * 32;
            const float* sc_lo = scpack + (size_t)blk32 * (size_t)B;
            const float* sc_hi = scpack + (size_t)(blk32 + 1) * (size_t)B;
            const int32_t* sm_lo = sumpack + (size_t)blk32 * (size_t)B;
            const int32_t* sm_hi = sumpack + (size_t)(blk32 + 1) * (size_t)B;
            for (int s = 0; s < B; s++) {
                const int8_t* xl = base_lo + (size_t)s * 32;
                const int8_t* xh = base_hi + (size_t)s * 32;
                int8x16_t xl0 = vld1q_s8(xl), xl1 = vld1q_s8(xl + 16);
                int8x16_t xh0 = vld1q_s8(xh), xh1 = vld1q_s8(xh + 16);
                int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0), lo1, xl1);
                int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0), hi1, xh1);
                float scl_lo = sc_lo[s], scl_hi = sc_hi[s];
                dot[s] += (double)(d_sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo)
                            - dmin_m1 * ((float)sm_lo[s] * scl_lo));
                dot[s] += (double)(d_sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi)
                            - dmin_m2 * ((float)sm_hi[s] * scl_hi));
            }
            q += 32; is += 2; blk32 += 2;
        }
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}

static inline void q4k_dot_rows2_sdot_multix_pack(
    const block_q4_K* row0, const block_q4_K* row1, int K,
    const int8_t* xpack, const float* scpack, const int32_t* sumpack,
    int B, float* out0, float* out1)
{
    int nb = K / 256;
    double d0[16], d1[16];
    for (int s = 0; s < B && s < 16; s++) { d0[s] = 0.0; d1[s] = 0.0; }
    if (B > 16) {
        for (int s = 0; s < B; s++) { out0[s] = 0.0f; out1[s] = 0.0f; }
        return;
    }
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) {
            __builtin_prefetch(row0 + i + 1, 0, 3);
            __builtin_prefetch(row1 + i + 1, 0, 3);
        }
        const block_q4_K* b0 = row0 + i;
        const block_q4_K* b1 = row1 + i;
        float dA = q4k_fp16_to_fp32(b0->d), dmA = q4k_fp16_to_fp32(b0->dmin);
        float dB = q4k_fp16_to_fp32(b1->d), dmB = q4k_fp16_to_fp32(b1->dmin);
        const uint8_t* qA = b0->qs;
        const uint8_t* qB = b1->qs;
        int is = 0;
        int blk32 = i * 8;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2, sc3, m3, sc4, m4;
            q4k_get_scale_min(is + 0, b0->scales, &sc1, &m1);
            q4k_get_scale_min(is + 1, b0->scales, &sc2, &m2);
            q4k_get_scale_min(is + 0, b1->scales, &sc3, &m3);
            q4k_get_scale_min(is + 1, b1->scales, &sc4, &m4);
            float d_sc1 = dA * (float)sc1, dmin_m1 = dmA * (float)m1;
            float d_sc2 = dA * (float)sc2, dmin_m2 = dmA * (float)m2;
            float d_sc3 = dB * (float)sc3, dmin_m3 = dmB * (float)m3;
            float d_sc4 = dB * (float)sc4, dmin_m4 = dmB * (float)m4;
            uint8x16_t w0 = vld1q_u8(qA), w1 = vld1q_u8(qA + 16);
            uint8x16_t v0 = vld1q_u8(qB), v1 = vld1q_u8(qB + 16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0, mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1, mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            int8x16_t lo2 = vreinterpretq_s8_u8(vandq_u8(v0, mask4));
            int8x16_t lo3 = vreinterpretq_s8_u8(vandq_u8(v1, mask4));
            int8x16_t hi2 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
            int8x16_t hi3 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
            const int8_t* base_lo = xpack + (size_t)blk32 * (size_t)B * 32;
            const int8_t* base_hi = xpack + (size_t)(blk32 + 1) * (size_t)B * 32;
            const float* sc_lo = scpack + (size_t)blk32 * (size_t)B;
            const float* sc_hi = scpack + (size_t)(blk32 + 1) * (size_t)B;
            const int32_t* sm_lo = sumpack + (size_t)blk32 * (size_t)B;
            const int32_t* sm_hi = sumpack + (size_t)(blk32 + 1) * (size_t)B;
            for (int s = 0; s < B; s++) {
                const int8_t* xl = base_lo + (size_t)s * 32;
                const int8_t* xh = base_hi + (size_t)s * 32;
                int8x16_t xl0 = vld1q_s8(xl), xl1 = vld1q_s8(xl + 16);
                int8x16_t xh0 = vld1q_s8(xh), xh1 = vld1q_s8(xh + 16);
                float scl_lo = sc_lo[s], scl_hi = sc_hi[s];
                float s_lo = (float)sm_lo[s], s_hi = (float)sm_hi[s];
                int32x4_t aq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, xl0), lo1, xl1);
                int32x4_t aq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, xh0), hi1, xh1);
                d0[s] += (double)(d_sc1 * ((float)vaddvq_s32(aq_lo) * scl_lo)
                            - dmin_m1 * (s_lo * scl_lo));
                d0[s] += (double)(d_sc2 * ((float)vaddvq_s32(aq_hi) * scl_hi)
                            - dmin_m2 * (s_hi * scl_hi));
                int32x4_t bq_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo2, xl0), lo3, xl1);
                int32x4_t bq_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi2, xh0), hi3, xh1);
                d1[s] += (double)(d_sc3 * ((float)vaddvq_s32(bq_lo) * scl_lo)
                            - dmin_m3 * (s_lo * scl_lo));
                d1[s] += (double)(d_sc4 * ((float)vaddvq_s32(bq_hi) * scl_hi)
                            - dmin_m4 * (s_hi * scl_hi));
            }
            qA += 32; qB += 32; is += 2; blk32 += 2;
        }
    }
    for (int s = 0; s < B; s++) { out0[s] = (float)d0[s]; out1[s] = (float)d1[s]; }
}

#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
static inline void q4k_dot_row_neon_multix(
    const block_q4_K* row_blocks, int K,
    const float* const* xs,
    int B,
    float* out)
{
    int n_blocks = K / 256;
    double dot[256] = {0};
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int x_off = 0;
    for (int i = 0; i < n_blocks; i++) {
        const block_q4_K* b = row_blocks + i;
        const float d    = q4k_fp16_to_fp32(b->d);
        const float dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1b, sc2, m2b;
            q4k_get_scale_min(is + 0, b->scales, &sc1, &m1b);
            q4k_get_scale_min(is + 1, b->scales, &sc2, &m2b);
            float d_sc1   = d * (float)sc1;
            float dmin_m1 = dmin * (float)m1b;
            float d_sc2   = d * (float)sc2;
            float dmin_m2 = dmin * (float)m2b;

            uint8x16_t b0 = vld1q_u8(q);
            uint8x16_t b1 = vld1q_u8(q + 16);
            uint8x16_t lo0 = vandq_u8(b0, mask4);
            uint8x16_t lo1 = vandq_u8(b1, mask4);
            uint8x16_t hi0 = vshrq_n_u8(b0, 4);
            uint8x16_t hi1 = vshrq_n_u8(b1, 4);
            uint16x8_t lo0_l = vmovl_u8(vget_low_u8(lo0));
            uint16x8_t lo0_h = vmovl_u8(vget_high_u8(lo0));
            uint16x8_t lo1_l = vmovl_u8(vget_low_u8(lo1));
            uint16x8_t lo1_h = vmovl_u8(vget_high_u8(lo1));
            float32x4_t lo_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo0_l)));
            float32x4_t lo_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo0_l)));
            float32x4_t lo_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo0_h)));
            float32x4_t lo_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo0_h)));
            float32x4_t lo_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo1_l)));
            float32x4_t lo_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo1_l)));
            float32x4_t lo_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo1_h)));
            float32x4_t lo_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo1_h)));
            uint16x8_t hi0_l = vmovl_u8(vget_low_u8(hi0));
            uint16x8_t hi0_h = vmovl_u8(vget_high_u8(hi0));
            uint16x8_t hi1_l = vmovl_u8(vget_low_u8(hi1));
            uint16x8_t hi1_h = vmovl_u8(vget_high_u8(hi1));
            float32x4_t hi_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi0_l)));
            float32x4_t hi_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi0_l)));
            float32x4_t hi_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi0_h)));
            float32x4_t hi_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi0_h)));
            float32x4_t hi_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi1_l)));
            float32x4_t hi_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi1_l)));
            float32x4_t hi_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi1_h)));
            float32x4_t hi_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi1_h)));

            for (int s = 0; s < B; s++) {
                const float* xp = xs[s] + x_off;
                float32x4_t xl0 = vld1q_f32(xp + 0);
                float32x4_t xl1 = vld1q_f32(xp + 4);
                float32x4_t xl2 = vld1q_f32(xp + 8);
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

                float32x4_t qx_lo = vmulq_f32(lo_f0, xl0);
                qx_lo = vfmaq_f32(qx_lo, lo_f1, xl1);
                qx_lo = vfmaq_f32(qx_lo, lo_f2, xl2);
                qx_lo = vfmaq_f32(qx_lo, lo_f3, xl3);
                qx_lo = vfmaq_f32(qx_lo, lo_f4, xl4);
                qx_lo = vfmaq_f32(qx_lo, lo_f5, xl5);
                qx_lo = vfmaq_f32(qx_lo, lo_f6, xl6);
                qx_lo = vfmaq_f32(qx_lo, lo_f7, xl7);
                float32x4_t qx_hi = vmulq_f32(hi_f0, xh0);
                qx_hi = vfmaq_f32(qx_hi, hi_f1, xh1);
                qx_hi = vfmaq_f32(qx_hi, hi_f2, xh2);
                qx_hi = vfmaq_f32(qx_hi, hi_f3, xh3);
                qx_hi = vfmaq_f32(qx_hi, hi_f4, xh4);
                qx_hi = vfmaq_f32(qx_hi, hi_f5, xh5);
                qx_hi = vfmaq_f32(qx_hi, hi_f6, xh6);
                qx_hi = vfmaq_f32(qx_hi, hi_f7, xh7);
                float32x4_t xs_lo = vaddq_f32(vaddq_f32(vaddq_f32(xl0, xl1), vaddq_f32(xl2, xl3)),
                                              vaddq_f32(vaddq_f32(xl4, xl5), vaddq_f32(xl6, xl7)));
                float32x4_t xs_hi = vaddq_f32(vaddq_f32(vaddq_f32(xh0, xh1), vaddq_f32(xh2, xh3)),
                                              vaddq_f32(vaddq_f32(xh4, xh5), vaddq_f32(xh6, xh7)));
                float qx_lo_s = vaddvq_f32(qx_lo);
                float qx_hi_s = vaddvq_f32(qx_hi);
                float xs_lo_s = vaddvq_f32(xs_lo);
                float xs_hi_s = vaddvq_f32(xs_hi);
                dot[s] += (double)(d_sc1 * qx_lo_s - dmin_m1 * xs_lo_s
                                 + d_sc2 * qx_hi_s - dmin_m2 * xs_hi_s);
            }

            q += 32;
            x_off += 64;
            is += 2;
        }
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}

#endif

#endif
