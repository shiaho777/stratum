
#ifndef STRATUM_Q6K_NEON_H
#define STRATUM_Q6K_NEON_H

#include "stratum_q6k.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

static inline void q6k_neon_dequant_half(
    const uint8_t* ql_half, const uint8_t* qh_half, const int8_t* sc,
    float d, float* y128)
{

    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t  bias  = vdupq_n_s8(-32);

    for (int is = 0; is < 2; is++) {

        float ds1 = d * (float)sc[is + 0];
        float ds2 = d * (float)sc[is + 2];
        float ds3 = d * (float)sc[is + 4];
        float ds4 = d * (float)sc[is + 6];
        float32x4_t v_ds1 = vdupq_n_f32(ds1);
        float32x4_t v_ds2 = vdupq_n_f32(ds2);
        float32x4_t v_ds3 = vdupq_n_f32(ds3);
        float32x4_t v_ds4 = vdupq_n_f32(ds4);

        int loff = is * 16;
        const uint8_t* ql_a = ql_half + loff;
        const uint8_t* ql_b = ql_half + loff + 32;
        const uint8_t* qh   = qh_half + loff;

        uint8x16_t ql_a_v = vld1q_u8(ql_a);
        uint8x16_t ql_b_v = vld1q_u8(ql_b);
        uint8x16_t qh_v   = vld1q_u8(qh);

        uint8x16_t lo_a = vandq_u8(ql_a_v, mask4);
        uint8x16_t hi1  = vshlq_n_u8(vandq_u8(qh_v, vdupq_n_u8(0x03)), 4);
        int8x16_t  q1   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(lo_a, hi1)), bias);

        uint8x16_t lo_b = vandq_u8(ql_b_v, mask4);
        uint8x16_t hi2  = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), vdupq_n_u8(0x03)), 4);
        int8x16_t  q2   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(lo_b, hi2)), bias);

        uint8x16_t hi_a = vshrq_n_u8(ql_a_v, 4);
        uint8x16_t hi3  = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), vdupq_n_u8(0x03)), 4);
        int8x16_t  q3   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(hi_a, hi3)), bias);

        uint8x16_t hi_b = vshrq_n_u8(ql_b_v, 4);
        uint8x16_t hi4  = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 6), vdupq_n_u8(0x03)), 4);
        int8x16_t  q4   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(hi_b, hi4)), bias);

        #define EMIT(qv, dsv, base)                                                  \
            do {                                                                     \
                int16x8_t s_lo = vmovl_s8(vget_low_s8(qv));                          \
                int16x8_t s_hi = vmovl_s8(vget_high_s8(qv));                         \
                int32x4_t i0 = vmovl_s16(vget_low_s16(s_lo));                        \
                int32x4_t i1 = vmovl_s16(vget_high_s16(s_lo));                       \
                int32x4_t i2 = vmovl_s16(vget_low_s16(s_hi));                        \
                int32x4_t i3 = vmovl_s16(vget_high_s16(s_hi));                       \
                vst1q_f32(y128 + (base) + 0,  vmulq_f32(vcvtq_f32_s32(i0), dsv));    \
                vst1q_f32(y128 + (base) + 4,  vmulq_f32(vcvtq_f32_s32(i1), dsv));    \
                vst1q_f32(y128 + (base) + 8,  vmulq_f32(vcvtq_f32_s32(i2), dsv));    \
                vst1q_f32(y128 + (base) + 12, vmulq_f32(vcvtq_f32_s32(i3), dsv));    \
            } while (0)

        EMIT(q1, v_ds1, loff +  0);
        EMIT(q2, v_ds2, loff + 32);
        EMIT(q3, v_ds3, loff + 64);
        EMIT(q4, v_ds4, loff + 96);

        #undef EMIT
    }
}

static inline float q6k_dot_row_neon(const block_q6_K* row_blocks, int K,
                                     const float* x)
{
    int n_blocks = K / 256;

    float buf[256] __attribute__((aligned(16)));
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q6_K* b = row_blocks + i;
        const float d = q4k_fp16_to_fp32(b->d);

        q6k_neon_dequant_half(b->ql + 0,  b->qh + 0,  b->scales + 0, d, buf + 0);
        q6k_neon_dequant_half(b->ql + 64, b->qh + 32, b->scales + 8, d, buf + 128);

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        for (int k = 0; k < 256; k += 16) {
            float32x4_t y0 = vld1q_f32(buf + k +  0);
            float32x4_t y1 = vld1q_f32(buf + k +  4);
            float32x4_t y2 = vld1q_f32(buf + k +  8);
            float32x4_t y3 = vld1q_f32(buf + k + 12);
            float32x4_t v0 = vld1q_f32(xp  + k +  0);
            float32x4_t v1 = vld1q_f32(xp  + k +  4);
            float32x4_t v2 = vld1q_f32(xp  + k +  8);
            float32x4_t v3 = vld1q_f32(xp  + k + 12);
            a0 = vfmaq_f32(a0, y0, v0);
            a1 = vfmaq_f32(a1, y1, v1);
            a2 = vfmaq_f32(a2, y2, v2);
            a3 = vfmaq_f32(a3, y3, v3);
        }
        dot += (double)vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        xp += 256;
    }
    return (float)dot;
}

#if defined(__ARM_FEATURE_DOTPROD)
static inline void q6k_quantize_x_q8_g16(const float* x, int K,
                                         int8_t* xq, float* xscale) {
    int ng = K / 16;
    for (int g = 0; g < ng; g++) {
        const float* xp = x + (size_t)g * 16;
        float amax = 0.0f;
        for (int i = 0; i < 16; i++) { float a = fabsf(xp[i]); if (a > amax) amax = a; }
        float scale = amax / 127.0f;
        float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        xscale[g] = scale;
        for (int i = 0; i < 16; i++) {
            int v = (int)lrintf(xp[i] * inv);
            if (v > 127) v = 127; if (v < -128) v = -128;
            xq[(size_t)g * 16 + i] = (int8_t)v;
        }
    }
}

static inline void q6k_dequant_half_i8(
    const uint8_t* ql_half, const uint8_t* qh_half, int8_t* q256, int base)
{
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t  bias  = vdupq_n_s8(-32);
    for (int is = 0; is < 2; is++) {
        int loff = is * 16;
        const uint8_t* ql_a = ql_half + loff;
        const uint8_t* ql_b = ql_half + loff + 32;
        const uint8_t* qh   = qh_half + loff;
        uint8x16_t ql_a_v = vld1q_u8(ql_a);
        uint8x16_t ql_b_v = vld1q_u8(ql_b);
        uint8x16_t qh_v   = vld1q_u8(qh);
        uint8x16_t lo_a = vandq_u8(ql_a_v, mask4);
        uint8x16_t hi1  = vshlq_n_u8(vandq_u8(qh_v, vdupq_n_u8(0x03)), 4);
        int8x16_t  q1   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(lo_a, hi1)), bias);
        uint8x16_t lo_b = vandq_u8(ql_b_v, mask4);
        uint8x16_t hi2  = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), vdupq_n_u8(0x03)), 4);
        int8x16_t  q2   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(lo_b, hi2)), bias);
        uint8x16_t hi_a = vshrq_n_u8(ql_a_v, 4);
        uint8x16_t hi3  = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), vdupq_n_u8(0x03)), 4);
        int8x16_t  q3   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(hi_a, hi3)), bias);
        uint8x16_t hi_b = vshrq_n_u8(ql_b_v, 4);
        uint8x16_t hi4  = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 6), vdupq_n_u8(0x03)), 4);
        int8x16_t  q4   = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(hi_b, hi4)), bias);
        vst1q_s8(q256 + base + loff +  0, q1);
        vst1q_s8(q256 + base + loff + 32, q2);
        vst1q_s8(q256 + base + loff + 64, q3);
        vst1q_s8(q256 + base + loff + 96, q4);
    }
}

static inline float q6k_dot_row_sdot(const block_q6_K* row_blocks, int K,
                                     const int8_t* xq, const float* xscale)
{
    int n_blocks = K / 256;
    int8_t q256[256] __attribute__((aligned(16)));
    double dot = 0.0;
    int goff = 0;
    for (int i = 0; i < n_blocks; i++) {
        const block_q6_K* b = row_blocks + i;
        const float d = q4k_fp16_to_fp32(b->d);
        q6k_dequant_half_i8(b->ql + 0,  b->qh + 0,  q256, 0);
        q6k_dequant_half_i8(b->ql + 64, b->qh + 32, q256, 128);
        const int8_t* s = b->scales;
        for (int g = 0; g < 16; g++) {
            int8x16_t qv = vld1q_s8(q256 + g * 16);
            int8x16_t xv = vld1q_s8(xq + (size_t)(goff + g) * 16);
            int32x4_t acc = vdotq_s32(vdupq_n_s32(0), qv, xv);
            int isum = vaddvq_s32(acc);
            dot += (double)d * (double)s[g] * (double)xscale[(goff + g) / 2] * (double)isum;
        }
        goff += 16;
    }
    return (float)dot;
}
#endif

#endif

#if defined(__ARM_FEATURE_DOTPROD)
/* V26: SDOT multix for Q6_K — B slots share int8 weight unpacking.
 * ~2x faster than NEON float multix. */
static inline void q6k_dot_row_sdot_multix(
    const block_q6_K* row_blocks, int K,
    const int8_t* const* xq, const float* const* xscale,
    int B, float* out)
{
    int n_blocks = K / 256;
    double dot[16];
    for (int s = 0; s < B && s < 16; s++) dot[s] = 0.0;
    if (B > 16) { for (int s = 0; s < B; s++) out[s] = 0.0f; return; }
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t bias = vdupq_n_s8(-32);
    int goff = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (i + 1 < n_blocks) __builtin_prefetch(row_blocks + i + 1, 0, 3);
        const block_q6_K* b = row_blocks + i;
        const float d = q4k_fp16_to_fp32(b->d);
        const int8_t* sc = b->scales;
        for (int half = 0; half < 2; half++) {
            const uint8_t* ql_half = b->ql + half * 64;
            const uint8_t* qh_half = b->qh + half * 32;
            int gbase = half * 8;
            for (int is = 0; is < 2; is++) {
                int loff = is * 16;
                uint8x16_t ql_a_v = vld1q_u8(ql_half + loff);
                uint8x16_t ql_b_v = vld1q_u8(ql_half + loff + 32);
                uint8x16_t qh_v = vld1q_u8(qh_half + loff);
                int8x16_t qv[4];
                qv[0] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_a_v, mask4),
                    vshlq_n_u8(vandq_u8(qh_v, vdupq_n_u8(0x03)), 4))), bias);
                qv[1] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_b_v, mask4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), vdupq_n_u8(0x03)), 4))), bias);
                qv[2] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_a_v, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), vdupq_n_u8(0x03)), 4))), bias);
                qv[3] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_b_v, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 6), vdupq_n_u8(0x03)), 4))), bias);
                int gg[4] = { gbase + is, gbase + is + 2, gbase + is + 4, gbase + is + 6 };
                for (int t = 0; t < 4; t++) {
                    int g = gg[t];
                    double ds = (double)d * (double)sc[g];
                    int xs_idx = (goff + g) / 2;
                    int xoff = (goff + g) * 16;
                    int8x16_t qvv = qv[t];
                    for (int si = 0; si < B; si++) {
                        int8x16_t xv = vld1q_s8(xq[si] + (size_t)xoff);
                        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), qvv, xv);
                        dot[si] += ds * (double)xscale[si][xs_idx] * (double)vaddvq_s32(acc);
                    }
                }
            }
        }
        goff += 16;
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}

static inline void q6k_dot_row_sdot_pack_b7(
    const block_q6_K* row_blocks, int K,
    const int8_t* xpack, const float* scpack,
    float* out)
{
    int n_blocks = K / 256;
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t bias = vdupq_n_s8(-32);
    int goff = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (i + 1 < n_blocks) __builtin_prefetch(row_blocks + i + 1, 0, 3);
        const block_q6_K* b = row_blocks + i;
        const float d = q4k_fp16_to_fp32(b->d);
        const int8_t* sc = b->scales;
        for (int half = 0; half < 2; half++) {
            const uint8_t* ql_half = b->ql + half * 64;
            const uint8_t* qh_half = b->qh + half * 32;
            int gbase = half * 8;
            for (int is = 0; is < 2; is++) {
                int loff = is * 16;
                uint8x16_t ql_a_v = vld1q_u8(ql_half + loff);
                uint8x16_t ql_b_v = vld1q_u8(ql_half + loff + 32);
                uint8x16_t qh_v = vld1q_u8(qh_half + loff);
                int8x16_t q1 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_a_v, mask4),
                    vshlq_n_u8(vandq_u8(qh_v, vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t q2 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_b_v, mask4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t q3 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_a_v, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t q4 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_b_v, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 6), vdupq_n_u8(0x03)), 4))), bias);
                int gg[4] = { gbase + is, gbase + is + 2, gbase + is + 4, gbase + is + 6 };
                int8x16_t qv[4] = { q1, q2, q3, q4 };
                for (int t = 0; t < 4; t++) {
                    int g = gg[t];
                    float ds = d * (float)sc[g];
                    int gabs = goff + g;
                    int b32 = gabs / 2;
                    int hx = (gabs & 1) * 16;
                    const int8_t* base = xpack + (size_t)b32 * 7 * 32;
                    const float* scb = scpack + (size_t)b32 * 7;
                    int8x16_t qvv = qv[t];
#define Q6_B7(s, acc) do { \
                        int8x16_t xv = vld1q_s8(base + (size_t)(s) * 32 + hx); \
                        int32x4_t accv = vdotq_s32(vdupq_n_s32(0), qvv, xv); \
                        (acc) += ds * scb[(s)] * (float)vaddvq_s32(accv); \
                    } while (0)
                    Q6_B7(0, d0); Q6_B7(1, d1); Q6_B7(2, d2); Q6_B7(3, d3);
                    Q6_B7(4, d4); Q6_B7(5, d5); Q6_B7(6, d6);
#undef Q6_B7
                }
            }
        }
        goff += 16;
    }
    out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3;
    out[4]=d4; out[5]=d5; out[6]=d6;
}

static inline void q6k_dot_rows2_sdot_pack_b7(
    const block_q6_K* row0, const block_q6_K* row1, int K,
    const int8_t* xpack, const float* scpack,
    float* out0, float* out1)
{
    int n_blocks = K / 256;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0;
    float b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t bias = vdupq_n_s8(-32);
    int goff = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (i + 1 < n_blocks) {
            __builtin_prefetch(row0 + i + 1, 0, 3);
            __builtin_prefetch(row1 + i + 1, 0, 3);
        }
        const block_q6_K* bA = row0 + i;
        const block_q6_K* bB = row1 + i;
        const float dA = q4k_fp16_to_fp32(bA->d);
        const float dB = q4k_fp16_to_fp32(bB->d);
        const int8_t* scA = bA->scales;
        const int8_t* scB = bB->scales;
        for (int half = 0; half < 2; half++) {
            const uint8_t* qlA = bA->ql + half * 64;
            const uint8_t* qhA = bA->qh + half * 32;
            const uint8_t* qlB = bB->ql + half * 64;
            const uint8_t* qhB = bB->qh + half * 32;
            int gbase = half * 8;
            for (int is = 0; is < 2; is++) {
                int loff = is * 16;
                uint8x16_t ql_a = vld1q_u8(qlA + loff);
                uint8x16_t ql_b = vld1q_u8(qlA + loff + 32);
                uint8x16_t qh_a = vld1q_u8(qhA + loff);
                uint8x16_t ql_c = vld1q_u8(qlB + loff);
                uint8x16_t ql_d = vld1q_u8(qlB + loff + 32);
                uint8x16_t qh_b = vld1q_u8(qhB + loff);
                int8x16_t qa[4], qb[4];
                qa[0] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_a, mask4),
                    vshlq_n_u8(vandq_u8(qh_a, vdupq_n_u8(0x03)), 4))), bias);
                qa[1] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_b, mask4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 2), vdupq_n_u8(0x03)), 4))), bias);
                qa[2] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_a, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 4), vdupq_n_u8(0x03)), 4))), bias);
                qa[3] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_b, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 6), vdupq_n_u8(0x03)), 4))), bias);
                qb[0] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_c, mask4),
                    vshlq_n_u8(vandq_u8(qh_b, vdupq_n_u8(0x03)), 4))), bias);
                qb[1] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_d, mask4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 2), vdupq_n_u8(0x03)), 4))), bias);
                qb[2] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_c, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 4), vdupq_n_u8(0x03)), 4))), bias);
                qb[3] = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_d, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 6), vdupq_n_u8(0x03)), 4))), bias);
                int gg[4] = { gbase + is, gbase + is + 2, gbase + is + 4, gbase + is + 6 };
                for (int t = 0; t < 4; t++) {
                    int g = gg[t];
                    float dsA = dA * (float)scA[g];
                    float dsB = dB * (float)scB[g];
                    int gabs = goff + g;
                    int b32 = gabs / 2;
                    int hx = (gabs & 1) * 16;
                    const int8_t* base = xpack + (size_t)b32 * 7 * 32;
                    const float* scb = scpack + (size_t)b32 * 7;
                    int8x16_t qAv = qa[t];
                    int8x16_t qBv = qb[t];
#define Q6_XSS(s, accA, accB) do { \
                        int8x16_t xv = vld1q_s8(base + (size_t)(s) * 32 + hx); \
                        int32x4_t aa = vdotq_s32(vdupq_n_s32(0), qAv, xv); \
                        int32x4_t bb = vdotq_s32(vdupq_n_s32(0), qBv, xv); \
                        float scv = scb[(s)]; \
                        (accA) += dsA * scv * (float)vaddvq_s32(aa); \
                        (accB) += dsB * scv * (float)vaddvq_s32(bb); \
                    } while (0)
                    Q6_XSS(0, a0, b0); Q6_XSS(1, a1, b1); Q6_XSS(2, a2, b2); Q6_XSS(3, a3, b3);
                    Q6_XSS(4, a4, b4); Q6_XSS(5, a5, b5); Q6_XSS(6, a6, b6);
#undef Q6_XSS
                }
            }
        }
        goff += 16;
    }
    out0[0]=a0; out0[1]=a1; out0[2]=a2; out0[3]=a3; out0[4]=a4; out0[5]=a5; out0[6]=a6;
    out1[0]=b0; out1[1]=b1; out1[2]=b2; out1[3]=b3; out1[4]=b4; out1[5]=b5; out1[6]=b6;
}

static inline void q6k_dot_rows4_sdot_pack_b7(
    const block_q6_K* row0, const block_q6_K* row1,
    const block_q6_K* row2, const block_q6_K* row3, int K,
    const int8_t* xpack, const float* scpack,
    float* o0, float* o1, float* o2, float* o3)
{
    q6k_dot_rows2_sdot_pack_b7(row0, row1, K, xpack, scpack, o0, o1);
    q6k_dot_rows2_sdot_pack_b7(row2, row3, K, xpack, scpack, o2, o3);
}

static inline void q6k_dot_row_sdot_multix_pack(
    const block_q6_K* row_blocks, int K,
    const int8_t* xpack, const float* scpack,
    int B, float* out)
{
    if (B == 7) { q6k_dot_row_sdot_pack_b7(row_blocks, K, xpack, scpack, out); return; }
    int n_blocks = K / 256;
    double dot[16];
    for (int s = 0; s < B && s < 16; s++) dot[s] = 0.0;
    if (B > 16) { for (int s = 0; s < B; s++) out[s] = 0.0f; return; }
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t bias = vdupq_n_s8(-32);
    int goff = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (i + 1 < n_blocks) __builtin_prefetch(row_blocks + i + 1, 0, 3);
        const block_q6_K* b = row_blocks + i;
        const float d = q4k_fp16_to_fp32(b->d);
        const int8_t* sc = b->scales;
        for (int half = 0; half < 2; half++) {
            const uint8_t* ql_half = b->ql + half * 64;
            const uint8_t* qh_half = b->qh + half * 32;
            int gbase = half * 8;
            for (int is = 0; is < 2; is++) {
                int loff = is * 16;
                uint8x16_t ql_a_v = vld1q_u8(ql_half + loff);
                uint8x16_t ql_b_v = vld1q_u8(ql_half + loff + 32);
                uint8x16_t qh_v = vld1q_u8(qh_half + loff);
                int8x16_t q1 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_a_v, mask4),
                    vshlq_n_u8(vandq_u8(qh_v, vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t q2 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_b_v, mask4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t q3 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_a_v, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t q4 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_b_v, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 6), vdupq_n_u8(0x03)), 4))), bias);
                int g1 = gbase + is;
                int g2 = gbase + is + 2;
                int g3 = gbase + is + 4;
                int g4 = gbase + is + 6;
                int8x16_t qv[4] = { q1, q2, q3, q4 };
                int gg[4] = { g1, g2, g3, g4 };
                for (int t = 0; t < 4; t++) {
                    int g = gg[t];
                    double ds = (double)d * (double)sc[g];
                    int gabs = goff + g;
                    int b32 = gabs / 2;
                    int hx = (gabs & 1) * 16;
                    const int8_t* base = xpack + (size_t)b32 * (size_t)B * 32;
                    const float* scb = scpack + (size_t)b32 * (size_t)B;
                    int8x16_t qvv = qv[t];
                    for (int si = 0; si < B; si++) {
                        int8x16_t xv = vld1q_s8(base + (size_t)si * 32 + hx);
                        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), qvv, xv);
                        dot[si] += ds * (double)scb[si] * (double)vaddvq_s32(acc);
                    }
                }
            }
        }
        goff += 16;
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}

static inline void q6k_dot_rows2_sdot_multix_pack(
    const block_q6_K* row0, const block_q6_K* row1, int K,
    const int8_t* xpack, const float* scpack,
    int B, float* out0, float* out1)
{
    int n_blocks = K / 256;
    double d0[16], d1[16];
    for (int s = 0; s < B && s < 16; s++) { d0[s] = 0.0; d1[s] = 0.0; }
    if (B > 16) {
        for (int s = 0; s < B; s++) { out0[s] = 0.0f; out1[s] = 0.0f; }
        return;
    }
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    int8x16_t bias = vdupq_n_s8(-32);
    int goff = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (i + 1 < n_blocks) {
            __builtin_prefetch(row0 + i + 1, 0, 3);
            __builtin_prefetch(row1 + i + 1, 0, 3);
        }
        const block_q6_K* b0 = row0 + i;
        const block_q6_K* b1 = row1 + i;
        const float dA = q4k_fp16_to_fp32(b0->d);
        const float dB = q4k_fp16_to_fp32(b1->d);
        const int8_t* scA = b0->scales;
        const int8_t* scB = b1->scales;
        for (int half = 0; half < 2; half++) {
            const uint8_t* qlA = b0->ql + half * 64;
            const uint8_t* qhA = b0->qh + half * 32;
            const uint8_t* qlB = b1->ql + half * 64;
            const uint8_t* qhB = b1->qh + half * 32;
            int gbase = half * 8;
            for (int is = 0; is < 2; is++) {
                int loff = is * 16;
                uint8x16_t ql_a = vld1q_u8(qlA + loff);
                uint8x16_t ql_b = vld1q_u8(qlA + loff + 32);
                uint8x16_t qh_a = vld1q_u8(qhA + loff);
                uint8x16_t ql_c = vld1q_u8(qlB + loff);
                uint8x16_t ql_d = vld1q_u8(qlB + loff + 32);
                uint8x16_t qh_b = vld1q_u8(qhB + loff);
                int8x16_t a1 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_a, mask4),
                    vshlq_n_u8(vandq_u8(qh_a, vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t a2 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_b, mask4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 2), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t a3 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_a, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 4), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t a4 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_b, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 6), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t c1 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_c, mask4),
                    vshlq_n_u8(vandq_u8(qh_b, vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t c2 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql_d, mask4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 2), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t c3 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_c, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 4), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t c4 = vaddq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql_d, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 6), vdupq_n_u8(0x03)), 4))), bias);
                int8x16_t qA[4] = { a1, a2, a3, a4 };
                int8x16_t qB[4] = { c1, c2, c3, c4 };
                int gg[4] = { gbase + is, gbase + is + 2, gbase + is + 4, gbase + is + 6 };
                for (int t = 0; t < 4; t++) {
                    int g = gg[t];
                    double dsA = (double)dA * (double)scA[g];
                    double dsB = (double)dB * (double)scB[g];
                    int gabs = goff + g;
                    int b32 = gabs / 2;
                    int hx = (gabs & 1) * 16;
                    const int8_t* base = xpack + (size_t)b32 * (size_t)B * 32;
                    const float* scb = scpack + (size_t)b32 * (size_t)B;
                    int8x16_t qa = qA[t];
                    int8x16_t qb = qB[t];
                    for (int si = 0; si < B; si++) {
                        int8x16_t xv = vld1q_s8(base + (size_t)si * 32 + hx);
                        int32x4_t accA = vdotq_s32(vdupq_n_s32(0), qa, xv);
                        int32x4_t accB = vdotq_s32(vdupq_n_s32(0), qb, xv);
                        float scv = scb[si];
                        d0[si] += dsA * (double)scv * (double)vaddvq_s32(accA);
                        d1[si] += dsB * (double)scv * (double)vaddvq_s32(accB);
                    }
                }
            }
        }
        goff += 16;
    }
    for (int s = 0; s < B; s++) { out0[s] = (float)d0[s]; out1[s] = (float)d1[s]; }
}
#endif

static inline void q6k_dot_row_neon_multix(const block_q6_K* row_blocks, int K,
                                           const float* const* xs, int B,
                                           float* out)
{
    int n_blocks = K / 256;
    float buf[256] __attribute__((aligned(16)));
    double dot[256] = {0};
    int x_off = 0;
    for (int i = 0; i < n_blocks; i++) {
        const block_q6_K* b = row_blocks + i;
        const float d = q4k_fp16_to_fp32(b->d);
        q6k_neon_dequant_half(b->ql + 0,  b->qh + 0,  b->scales + 0, d, buf + 0);
        q6k_neon_dequant_half(b->ql + 64, b->qh + 32, b->scales + 8, d, buf + 128);
        for (int s = 0; s < B; s++) {
            const float* xp = xs[s] + x_off;
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            for (int k = 0; k < 256; k += 16) {
                a0 = vfmaq_f32(a0, vld1q_f32(buf + k +  0), vld1q_f32(xp + k +  0));
                a1 = vfmaq_f32(a1, vld1q_f32(buf + k +  4), vld1q_f32(xp + k +  4));
                a2 = vfmaq_f32(a2, vld1q_f32(buf + k +  8), vld1q_f32(xp + k +  8));
                a3 = vfmaq_f32(a3, vld1q_f32(buf + k + 12), vld1q_f32(xp + k + 12));
            }
            dot[s] += (double)vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        }
        x_off += 256;
    }
    for (int s = 0; s < B; s++) out[s] = (float)dot[s];
}

#endif
