
#ifndef STRATUM_NEON_H
#define STRATUM_NEON_H

#include <stdint.h>
#include <stddef.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define STRATUM_HAS_NEON 1
#else
#define STRATUM_HAS_NEON 0
#endif

static inline void st_int4_block_to_fp32_neon(
    const uint8_t* packed, const float* scale,
    int rows, int in_features, float* dst)
{
    int half = in_features >> 1;
#if STRATUM_HAS_NEON
    for (int r = 0; r < rows; r++) {
        const uint8_t* row = packed + (size_t)r * half;
        float*         drow = dst + (size_t)r * in_features;
        float32x4_t    sv  = vdupq_n_f32(scale[r]);

        int i = 0;
        for (; i + 16 <= half; i += 16) {
            uint8x16_t b = vld1q_u8(row + i);

            int8x16_t lo_s = vshrq_n_s8(
                vreinterpretq_s8_u8(vshlq_n_u8(b, 4)),
                4);

            int8x16_t hi_s = vshrq_n_s8(vreinterpretq_s8_u8(b), 4);

            int8x16x2_t z = vzipq_s8(lo_s, hi_s);

            for (int k = 0; k < 2; k++) {
                int8x16_t zk = z.val[k];

                int16x8_t i16_lo = vmovl_s8(vget_low_s8(zk));
                int16x8_t i16_hi = vmovl_s8(vget_high_s8(zk));

                int32x4_t i32_0 = vmovl_s16(vget_low_s16(i16_lo));
                int32x4_t i32_1 = vmovl_s16(vget_high_s16(i16_lo));
                int32x4_t i32_2 = vmovl_s16(vget_low_s16(i16_hi));
                int32x4_t i32_3 = vmovl_s16(vget_high_s16(i16_hi));

                float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(i32_0), sv);
                float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(i32_1), sv);
                float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(i32_2), sv);
                float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(i32_3), sv);

                float* outp = drow + i * 2 + k * 16;
                vst1q_f32(outp + 0,  f0);
                vst1q_f32(outp + 4,  f1);
                vst1q_f32(outp + 8,  f2);
                vst1q_f32(outp + 12, f3);
            }
        }

        for (; i < half; i++) {
            uint8_t b = row[i];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            int lo_s = (lo >= 8) ? lo - 16 : lo;
            int hi_s = (hi >= 8) ? hi - 16 : hi;
            float s = scale[r];
            drow[2*i]     = (float)lo_s * s;
            drow[2*i + 1] = (float)hi_s * s;
        }
    }
#else

    for (int r = 0; r < rows; r++) {
        const uint8_t* row = packed + (size_t)r * half;
        float* drow = dst + (size_t)r * in_features;
        float s = scale[r];
        for (int i = 0; i < half; i++) {
            uint8_t b = row[i];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            int lo_s = (lo >= 8) ? lo - 16 : lo;
            int hi_s = (hi >= 8) ? hi - 16 : hi;
            drow[2*i]     = (float)lo_s * s;
            drow[2*i + 1] = (float)hi_s * s;
        }
    }
#endif
}

static inline void st_int4_sgemv_fused_neon(
    const uint8_t* packed, const float* scale,
    int rows, int in_features,
    const float* x, float* y)
{
    int half = in_features >> 1;
#if STRATUM_HAS_NEON
    int tail_start = (half / 16) * 16;
    int tail = half - tail_start;
    for (int r = 0; r < rows; r++) {
        const uint8_t* row = packed + (size_t)r * half;
        const float* xp = x;

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f);
        float32x4_t b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f);
        float32x4_t b3 = vdupq_n_f32(0.0f);

        for (int i = 0; i < tail_start; i += 16) {
            uint8x16_t b = vld1q_u8(row + i);
            int8x16_t lo_s = vshrq_n_s8(
                vreinterpretq_s8_u8(vshlq_n_u8(b, 4)),
                4);
            int8x16_t hi_s = vshrq_n_s8(vreinterpretq_s8_u8(b), 4);
            int8x16x2_t z = vzipq_s8(lo_s, hi_s);

            int8x16_t  zk = z.val[0];
            int16x8_t  iL = vmovl_s8(vget_low_s8(zk));
            int16x8_t  iH = vmovl_s8(vget_high_s8(zk));
            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iL)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(iL)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iH)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(iH)));
            float32x4_t x0 = vld1q_f32(xp + 0);
            float32x4_t x1 = vld1q_f32(xp + 4);
            float32x4_t x2 = vld1q_f32(xp + 8);
            float32x4_t x3 = vld1q_f32(xp + 12);
            a0 = vfmaq_f32(a0, f0, x0);
            a1 = vfmaq_f32(a1, f1, x1);
            a2 = vfmaq_f32(a2, f2, x2);
            a3 = vfmaq_f32(a3, f3, x3);

            zk = z.val[1];
            iL = vmovl_s8(vget_low_s8(zk));
            iH = vmovl_s8(vget_high_s8(zk));
            f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iL)));
            f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(iL)));
            f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iH)));
            f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(iH)));
            x0 = vld1q_f32(xp + 16);
            x1 = vld1q_f32(xp + 20);
            x2 = vld1q_f32(xp + 24);
            x3 = vld1q_f32(xp + 28);
            b0 = vfmaq_f32(b0, f0, x0);
            b1 = vfmaq_f32(b1, f1, x1);
            b2 = vfmaq_f32(b2, f2, x2);
            b3 = vfmaq_f32(b3, f3, x3);

            xp += 32;
        }

        float32x4_t s0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float32x4_t s1 = vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3));
        float32x4_t sv = vaddq_f32(s0, s1);
        float dot = vaddvq_f32(sv);

        for (int i = tail_start; i < half; i++) {
            uint8_t b = row[i];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            int lo_s = (lo >= 8) ? lo - 16 : lo;
            int hi_s = (hi >= 8) ? hi - 16 : hi;
            dot += (float)lo_s * xp[0] + (float)hi_s * xp[1];
            xp += 2;
        }

        y[r] += dot * scale[r];
        (void)tail;
    }
#else

    for (int r = 0; r < rows; r++) {
        const uint8_t* row = packed + (size_t)r * half;
        const float* xp = x;
        float dot = 0.0f;
        for (int i = 0; i < half; i++) {
            uint8_t b = row[i];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            int lo_s = (lo >= 8) ? lo - 16 : lo;
            int hi_s = (hi >= 8) ? hi - 16 : hi;
            dot += (float)lo_s * xp[0] + (float)hi_s * xp[1];
            xp += 2;
        }
        y[r] += dot * scale[r];
    }
#endif
}

#endif