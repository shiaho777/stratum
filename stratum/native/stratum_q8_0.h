
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

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>

/* SDOT variants: weights are already i8 — the int8 activation dots are
 * exact; only the block scale product is float. */
static inline float q8_0_dot_row_sdot(const block_q8_0* blocks, int K,
                                      const int8_t* xq, const float* xscale) {
    int nb = K / 32;
    double dot = 0.0;
    for (int i = 0; i < nb; i++) {
        const block_q8_0* b = blocks + i;
        float d = q4k_fp16_to_fp32(b->d);
        const int8_t* xv = xq + (size_t)i * 32;
        int32x4_t acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
            vld1q_s8(b->qs), vld1q_s8(xv)),
            vld1q_s8(b->qs + 16), vld1q_s8(xv + 16));
        dot += (double)(d * xscale[i] * (float)vaddvq_s32(acc));
    }
    return (float)dot;
}

/* f32x4 accumulate variant: four block terms per vfmaq. */
static inline float q8_0_dot_row_sdot_f(const block_q8_0* blocks, int K,
                                        const int8_t* xq, const float* xscale) {
    int nb = K / 32;
    float32x4_t facc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= nb; i += 4) {
        int32_t tv[4]; float cv[4];
        for (int t = 0; t < 4; t++) {
            const block_q8_0* b = blocks + i + t;
            const int8_t* xv = xq + (size_t)(i + t) * 32;
            int32x4_t acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                vld1q_s8(b->qs), vld1q_s8(xv)),
                vld1q_s8(b->qs + 16), vld1q_s8(xv + 16));
            tv[t] = vaddvq_s32(acc);
            cv[t] = q4k_fp16_to_fp32(b->d) * xscale[i + t];
        }
        int32x4_t  terms = { tv[0], tv[1], tv[2], tv[3] };
        float32x4_t coeff = { cv[0], cv[1], cv[2], cv[3] };
        facc = vfmaq_f32(facc, vcvtq_f32_s32(terms), coeff);
    }
    for (; i < nb; i++) {
        const block_q8_0* b = blocks + i;
        const int8_t* xv = xq + (size_t)i * 32;
        int32x4_t acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
            vld1q_s8(b->qs), vld1q_s8(xv)),
            vld1q_s8(b->qs + 16), vld1q_s8(xv + 16));
        facc = vfmaq_f32(facc,
            vcvtq_f32_s32(acc),
            vdupq_n_f32(q4k_fp16_to_fp32(b->d) * xscale[i]));
    }
    return vaddvq_f32(facc);
}

/* Multi-sequence Q8_0: 4 rows share each activation load. xpack/scpack use
 * the shared [g][Bc][32] / [g][Bc] layout (same pack the Q4_K multix path
 * produces via q4k_quantize_x_q8). Per-(row,seq) summation order is kept
 * identical to q8_0_dot_row_sdot_f (f32 lanes of 4 consecutive group terms)
 * so results are bit-exact vs the single-stream SDOT path. */
static inline void q8_0_dot_rows4_sdot_multix_pack(
    const block_q8_0* r0, const block_q8_0* r1,
    const block_q8_0* r2, const block_q8_0* r3, int K,
    const int8_t* xpack, const float* scpack, int B,
    float* out0, float* out1, float* out2, float* out3)
{
    int nb = K / 32;
    for (int s = 0; s < B && s < 16; s++) {
        float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0),
                    f2 = vdupq_n_f32(0), f3 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 4 <= nb; i += 4) {
            int32_t tv0[4], tv1[4], tv2[4], tv3[4];
            float   cv0[4], cv1[4], cv2[4], cv3[4];
            for (int t = 0; t < 4; t++) {
                const block_q8_0* b0 = r0 + i + t;
                const block_q8_0* b1 = r1 + i + t;
                const block_q8_0* b2 = r2 + i + t;
                const block_q8_0* b3 = r3 + i + t;
                const int8_t* xv = xpack + ((size_t)(i + t) * B + s) * 32;
                int8x16_t x0 = vld1q_s8(xv), x1 = vld1q_s8(xv + 16);
                float xs = scpack[(size_t)(i + t) * B + s];
                tv0[t] = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                    vld1q_s8(b0->qs), x0), vld1q_s8(b0->qs + 16), x1));
                tv1[t] = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                    vld1q_s8(b1->qs), x0), vld1q_s8(b1->qs + 16), x1));
                tv2[t] = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                    vld1q_s8(b2->qs), x0), vld1q_s8(b2->qs + 16), x1));
                tv3[t] = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                    vld1q_s8(b3->qs), x0), vld1q_s8(b3->qs + 16), x1));
                cv0[t] = q4k_fp16_to_fp32(b0->d) * xs;
                cv1[t] = q4k_fp16_to_fp32(b1->d) * xs;
                cv2[t] = q4k_fp16_to_fp32(b2->d) * xs;
                cv3[t] = q4k_fp16_to_fp32(b3->d) * xs;
            }
            int32x4_t t0 = { tv0[0], tv0[1], tv0[2], tv0[3] };
            int32x4_t t1 = { tv1[0], tv1[1], tv1[2], tv1[3] };
            int32x4_t t2 = { tv2[0], tv2[1], tv2[2], tv2[3] };
            int32x4_t t3 = { tv3[0], tv3[1], tv3[2], tv3[3] };
            float32x4_t c0 = { cv0[0], cv0[1], cv0[2], cv0[3] };
            float32x4_t c1 = { cv1[0], cv1[1], cv1[2], cv1[3] };
            float32x4_t c2 = { cv2[0], cv2[1], cv2[2], cv2[3] };
            float32x4_t c3 = { cv3[0], cv3[1], cv3[2], cv3[3] };
            f0 = vfmaq_f32(f0, vcvtq_f32_s32(t0), c0);
            f1 = vfmaq_f32(f1, vcvtq_f32_s32(t1), c1);
            f2 = vfmaq_f32(f2, vcvtq_f32_s32(t2), c2);
            f3 = vfmaq_f32(f3, vcvtq_f32_s32(t3), c3);
        }
        for (; i < nb; i++) {
            const int8_t* xv = xpack + ((size_t)i * B + s) * 32;
            int8x16_t x0 = vld1q_s8(xv), x1 = vld1q_s8(xv + 16);
            float xs = scpack[(size_t)i * B + s];
            const block_q8_0* b0 = r0 + i;
            const block_q8_0* b1 = r1 + i;
            const block_q8_0* b2 = r2 + i;
            const block_q8_0* b3 = r3 + i;
            int32x4_t a0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                vld1q_s8(b0->qs), x0), vld1q_s8(b0->qs + 16), x1);
            int32x4_t a1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                vld1q_s8(b1->qs), x0), vld1q_s8(b1->qs + 16), x1);
            int32x4_t a2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                vld1q_s8(b2->qs), x0), vld1q_s8(b2->qs + 16), x1);
            int32x4_t a3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                vld1q_s8(b3->qs), x0), vld1q_s8(b3->qs + 16), x1);
            f0 = vfmaq_f32(f0, vcvtq_f32_s32(a0),
                 vdupq_n_f32(q4k_fp16_to_fp32(b0->d) * xs));
            f1 = vfmaq_f32(f1, vcvtq_f32_s32(a1),
                 vdupq_n_f32(q4k_fp16_to_fp32(b1->d) * xs));
            f2 = vfmaq_f32(f2, vcvtq_f32_s32(a2),
                 vdupq_n_f32(q4k_fp16_to_fp32(b2->d) * xs));
            f3 = vfmaq_f32(f3, vcvtq_f32_s32(a3),
                 vdupq_n_f32(q4k_fp16_to_fp32(b3->d) * xs));
        }
        out0[s] = vaddvq_f32(f0); out1[s] = vaddvq_f32(f1);
        out2[s] = vaddvq_f32(f2); out3[s] = vaddvq_f32(f3);
    }
    for (int s = 16; s < B; s++) {   /* pack callers cap Bc<=16; defensive */
        out0[s] = out1[s] = out2[s] = out3[s] = 0.0f;
    }
}

/* 1-row tail variant (same summation order). */
static inline void q8_0_dot_row_sdot_multix_pack(
    const block_q8_0* row, int K,
    const int8_t* xpack, const float* scpack, int B, float* out)
{
    int nb = K / 32;
    for (int s = 0; s < B && s < 16; s++) {
        float32x4_t facc = vdupq_n_f32(0);
        int i = 0;
        for (; i + 4 <= nb; i += 4) {
            int32_t tv[4]; float cv[4];
            for (int t = 0; t < 4; t++) {
                const block_q8_0* b = row + i + t;
                const int8_t* xv = xpack + ((size_t)(i + t) * B + s) * 32;
                int32x4_t acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                    vld1q_s8(b->qs), vld1q_s8(xv)),
                    vld1q_s8(b->qs + 16), vld1q_s8(xv + 16));
                tv[t] = vaddvq_s32(acc);
                cv[t] = q4k_fp16_to_fp32(b->d) * scpack[(size_t)(i + t) * B + s];
            }
            int32x4_t  terms = { tv[0], tv[1], tv[2], tv[3] };
            float32x4_t coeff = { cv[0], cv[1], cv[2], cv[3] };
            facc = vfmaq_f32(facc, vcvtq_f32_s32(terms), coeff);
        }
        for (; i < nb; i++) {
            const block_q8_0* b = row + i;
            const int8_t* xv = xpack + ((size_t)i * B + s) * 32;
            int32x4_t acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                vld1q_s8(b->qs), vld1q_s8(xv)),
                vld1q_s8(b->qs + 16), vld1q_s8(xv + 16));
            facc = vfmaq_f32(facc, vcvtq_f32_s32(acc),
                vdupq_n_f32(q4k_fp16_to_fp32(b->d) * scpack[(size_t)i * B + s]));
        }
        out[s] = vaddvq_f32(facc);
    }
}
#endif

#endif
