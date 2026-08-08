
#ifndef STRATUM_LIN_ATTN_V3_H
#define STRATUM_LIN_ATTN_V3_H

#include <stdint.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

static inline void ssm_v3_dual_gemv_trans_hv128(
    const float* __restrict__ S,
    const float* __restrict__ kt,
    const float* __restrict__ qt,
    int HK,
    float* __restrict__ b_pre,
    float* __restrict__ a_pre)
{

    float32x4_t b0  = vdupq_n_f32(0), b1  = vdupq_n_f32(0), b2  = vdupq_n_f32(0), b3  = vdupq_n_f32(0);
    float32x4_t b4  = vdupq_n_f32(0), b5  = vdupq_n_f32(0), b6  = vdupq_n_f32(0), b7  = vdupq_n_f32(0);
    float32x4_t b8  = vdupq_n_f32(0), b9  = vdupq_n_f32(0), b10 = vdupq_n_f32(0), b11 = vdupq_n_f32(0);
    float32x4_t b12 = vdupq_n_f32(0), b13 = vdupq_n_f32(0), b14 = vdupq_n_f32(0), b15 = vdupq_n_f32(0);
    float32x4_t b16 = vdupq_n_f32(0), b17 = vdupq_n_f32(0), b18 = vdupq_n_f32(0), b19 = vdupq_n_f32(0);
    float32x4_t b20 = vdupq_n_f32(0), b21 = vdupq_n_f32(0), b22 = vdupq_n_f32(0), b23 = vdupq_n_f32(0);
    float32x4_t b24 = vdupq_n_f32(0), b25 = vdupq_n_f32(0), b26 = vdupq_n_f32(0), b27 = vdupq_n_f32(0);
    float32x4_t b28 = vdupq_n_f32(0), b29 = vdupq_n_f32(0), b30 = vdupq_n_f32(0), b31 = vdupq_n_f32(0);

    float32x4_t a0  = vdupq_n_f32(0), a1  = vdupq_n_f32(0), a2  = vdupq_n_f32(0), a3  = vdupq_n_f32(0);
    float32x4_t a4  = vdupq_n_f32(0), a5  = vdupq_n_f32(0), a6  = vdupq_n_f32(0), a7  = vdupq_n_f32(0);
    float32x4_t a8  = vdupq_n_f32(0), a9  = vdupq_n_f32(0), a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0);
    float32x4_t a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0), a14 = vdupq_n_f32(0), a15 = vdupq_n_f32(0);
    float32x4_t a16 = vdupq_n_f32(0), a17 = vdupq_n_f32(0), a18 = vdupq_n_f32(0), a19 = vdupq_n_f32(0);
    float32x4_t a20 = vdupq_n_f32(0), a21 = vdupq_n_f32(0), a22 = vdupq_n_f32(0), a23 = vdupq_n_f32(0);
    float32x4_t a24 = vdupq_n_f32(0), a25 = vdupq_n_f32(0), a26 = vdupq_n_f32(0), a27 = vdupq_n_f32(0);
    float32x4_t a28 = vdupq_n_f32(0), a29 = vdupq_n_f32(0), a30 = vdupq_n_f32(0), a31 = vdupq_n_f32(0);

    for (int k = 0; k < HK; k++) {
        const float* row = S + k * 128;
        float32x4_t kv = vdupq_n_f32(kt[k]);
        float32x4_t qv = vdupq_n_f32(qt[k]);

        float32x4_t r0  = vld1q_f32(row +   0);
        float32x4_t r1  = vld1q_f32(row +   4);
        float32x4_t r2  = vld1q_f32(row +   8);
        float32x4_t r3  = vld1q_f32(row +  12);
        float32x4_t r4  = vld1q_f32(row +  16);
        float32x4_t r5  = vld1q_f32(row +  20);
        float32x4_t r6  = vld1q_f32(row +  24);
        float32x4_t r7  = vld1q_f32(row +  28);
        float32x4_t r8  = vld1q_f32(row +  32);
        float32x4_t r9  = vld1q_f32(row +  36);
        float32x4_t r10 = vld1q_f32(row +  40);
        float32x4_t r11 = vld1q_f32(row +  44);
        float32x4_t r12 = vld1q_f32(row +  48);
        float32x4_t r13 = vld1q_f32(row +  52);
        float32x4_t r14 = vld1q_f32(row +  56);
        float32x4_t r15 = vld1q_f32(row +  60);
        float32x4_t r16 = vld1q_f32(row +  64);
        float32x4_t r17 = vld1q_f32(row +  68);
        float32x4_t r18 = vld1q_f32(row +  72);
        float32x4_t r19 = vld1q_f32(row +  76);
        float32x4_t r20 = vld1q_f32(row +  80);
        float32x4_t r21 = vld1q_f32(row +  84);
        float32x4_t r22 = vld1q_f32(row +  88);
        float32x4_t r23 = vld1q_f32(row +  92);
        float32x4_t r24 = vld1q_f32(row +  96);
        float32x4_t r25 = vld1q_f32(row + 100);
        float32x4_t r26 = vld1q_f32(row + 104);
        float32x4_t r27 = vld1q_f32(row + 108);
        float32x4_t r28 = vld1q_f32(row + 112);
        float32x4_t r29 = vld1q_f32(row + 116);
        float32x4_t r30 = vld1q_f32(row + 120);
        float32x4_t r31 = vld1q_f32(row + 124);

        b0  = vfmaq_f32(b0,  r0,  kv); a0  = vfmaq_f32(a0,  r0,  qv);
        b1  = vfmaq_f32(b1,  r1,  kv); a1  = vfmaq_f32(a1,  r1,  qv);
        b2  = vfmaq_f32(b2,  r2,  kv); a2  = vfmaq_f32(a2,  r2,  qv);
        b3  = vfmaq_f32(b3,  r3,  kv); a3  = vfmaq_f32(a3,  r3,  qv);
        b4  = vfmaq_f32(b4,  r4,  kv); a4  = vfmaq_f32(a4,  r4,  qv);
        b5  = vfmaq_f32(b5,  r5,  kv); a5  = vfmaq_f32(a5,  r5,  qv);
        b6  = vfmaq_f32(b6,  r6,  kv); a6  = vfmaq_f32(a6,  r6,  qv);
        b7  = vfmaq_f32(b7,  r7,  kv); a7  = vfmaq_f32(a7,  r7,  qv);
        b8  = vfmaq_f32(b8,  r8,  kv); a8  = vfmaq_f32(a8,  r8,  qv);
        b9  = vfmaq_f32(b9,  r9,  kv); a9  = vfmaq_f32(a9,  r9,  qv);
        b10 = vfmaq_f32(b10, r10, kv); a10 = vfmaq_f32(a10, r10, qv);
        b11 = vfmaq_f32(b11, r11, kv); a11 = vfmaq_f32(a11, r11, qv);
        b12 = vfmaq_f32(b12, r12, kv); a12 = vfmaq_f32(a12, r12, qv);
        b13 = vfmaq_f32(b13, r13, kv); a13 = vfmaq_f32(a13, r13, qv);
        b14 = vfmaq_f32(b14, r14, kv); a14 = vfmaq_f32(a14, r14, qv);
        b15 = vfmaq_f32(b15, r15, kv); a15 = vfmaq_f32(a15, r15, qv);
        b16 = vfmaq_f32(b16, r16, kv); a16 = vfmaq_f32(a16, r16, qv);
        b17 = vfmaq_f32(b17, r17, kv); a17 = vfmaq_f32(a17, r17, qv);
        b18 = vfmaq_f32(b18, r18, kv); a18 = vfmaq_f32(a18, r18, qv);
        b19 = vfmaq_f32(b19, r19, kv); a19 = vfmaq_f32(a19, r19, qv);
        b20 = vfmaq_f32(b20, r20, kv); a20 = vfmaq_f32(a20, r20, qv);
        b21 = vfmaq_f32(b21, r21, kv); a21 = vfmaq_f32(a21, r21, qv);
        b22 = vfmaq_f32(b22, r22, kv); a22 = vfmaq_f32(a22, r22, qv);
        b23 = vfmaq_f32(b23, r23, kv); a23 = vfmaq_f32(a23, r23, qv);
        b24 = vfmaq_f32(b24, r24, kv); a24 = vfmaq_f32(a24, r24, qv);
        b25 = vfmaq_f32(b25, r25, kv); a25 = vfmaq_f32(a25, r25, qv);
        b26 = vfmaq_f32(b26, r26, kv); a26 = vfmaq_f32(a26, r26, qv);
        b27 = vfmaq_f32(b27, r27, kv); a27 = vfmaq_f32(a27, r27, qv);
        b28 = vfmaq_f32(b28, r28, kv); a28 = vfmaq_f32(a28, r28, qv);
        b29 = vfmaq_f32(b29, r29, kv); a29 = vfmaq_f32(a29, r29, qv);
        b30 = vfmaq_f32(b30, r30, kv); a30 = vfmaq_f32(a30, r30, qv);
        b31 = vfmaq_f32(b31, r31, kv); a31 = vfmaq_f32(a31, r31, qv);
    }

    vst1q_f32(b_pre +   0, b0 ); vst1q_f32(a_pre +   0, a0 );
    vst1q_f32(b_pre +   4, b1 ); vst1q_f32(a_pre +   4, a1 );
    vst1q_f32(b_pre +   8, b2 ); vst1q_f32(a_pre +   8, a2 );
    vst1q_f32(b_pre +  12, b3 ); vst1q_f32(a_pre +  12, a3 );
    vst1q_f32(b_pre +  16, b4 ); vst1q_f32(a_pre +  16, a4 );
    vst1q_f32(b_pre +  20, b5 ); vst1q_f32(a_pre +  20, a5 );
    vst1q_f32(b_pre +  24, b6 ); vst1q_f32(a_pre +  24, a6 );
    vst1q_f32(b_pre +  28, b7 ); vst1q_f32(a_pre +  28, a7 );
    vst1q_f32(b_pre +  32, b8 ); vst1q_f32(a_pre +  32, a8 );
    vst1q_f32(b_pre +  36, b9 ); vst1q_f32(a_pre +  36, a9 );
    vst1q_f32(b_pre +  40, b10); vst1q_f32(a_pre +  40, a10);
    vst1q_f32(b_pre +  44, b11); vst1q_f32(a_pre +  44, a11);
    vst1q_f32(b_pre +  48, b12); vst1q_f32(a_pre +  48, a12);
    vst1q_f32(b_pre +  52, b13); vst1q_f32(a_pre +  52, a13);
    vst1q_f32(b_pre +  56, b14); vst1q_f32(a_pre +  56, a14);
    vst1q_f32(b_pre +  60, b15); vst1q_f32(a_pre +  60, a15);
    vst1q_f32(b_pre +  64, b16); vst1q_f32(a_pre +  64, a16);
    vst1q_f32(b_pre +  68, b17); vst1q_f32(a_pre +  68, a17);
    vst1q_f32(b_pre +  72, b18); vst1q_f32(a_pre +  72, a18);
    vst1q_f32(b_pre +  76, b19); vst1q_f32(a_pre +  76, a19);
    vst1q_f32(b_pre +  80, b20); vst1q_f32(a_pre +  80, a20);
    vst1q_f32(b_pre +  84, b21); vst1q_f32(a_pre +  84, a21);
    vst1q_f32(b_pre +  88, b22); vst1q_f32(a_pre +  88, a22);
    vst1q_f32(b_pre +  92, b23); vst1q_f32(a_pre +  92, a23);
    vst1q_f32(b_pre +  96, b24); vst1q_f32(a_pre +  96, a24);
    vst1q_f32(b_pre + 100, b25); vst1q_f32(a_pre + 100, a25);
    vst1q_f32(b_pre + 104, b26); vst1q_f32(a_pre + 104, a26);
    vst1q_f32(b_pre + 108, b27); vst1q_f32(a_pre + 108, a27);
    vst1q_f32(b_pre + 112, b28); vst1q_f32(a_pre + 112, a28);
    vst1q_f32(b_pre + 116, b29); vst1q_f32(a_pre + 116, a29);
    vst1q_f32(b_pre + 120, b30); vst1q_f32(a_pre + 120, a30);
    vst1q_f32(b_pre + 124, b31); vst1q_f32(a_pre + 124, a31);
}

static inline void ssm_v3_fused_decay_outer_hv128(
    float* __restrict__ S,
    const float* __restrict__ kt,
    const float* __restrict__ delta,
    int HK,
    float gt)
{

    float32x4_t d0  = vld1q_f32(delta +   0);
    float32x4_t d1  = vld1q_f32(delta +   4);
    float32x4_t d2  = vld1q_f32(delta +   8);
    float32x4_t d3  = vld1q_f32(delta +  12);
    float32x4_t d4  = vld1q_f32(delta +  16);
    float32x4_t d5  = vld1q_f32(delta +  20);
    float32x4_t d6  = vld1q_f32(delta +  24);
    float32x4_t d7  = vld1q_f32(delta +  28);
    float32x4_t d8  = vld1q_f32(delta +  32);
    float32x4_t d9  = vld1q_f32(delta +  36);
    float32x4_t d10 = vld1q_f32(delta +  40);
    float32x4_t d11 = vld1q_f32(delta +  44);
    float32x4_t d12 = vld1q_f32(delta +  48);
    float32x4_t d13 = vld1q_f32(delta +  52);
    float32x4_t d14 = vld1q_f32(delta +  56);
    float32x4_t d15 = vld1q_f32(delta +  60);
    float32x4_t d16 = vld1q_f32(delta +  64);
    float32x4_t d17 = vld1q_f32(delta +  68);
    float32x4_t d18 = vld1q_f32(delta +  72);
    float32x4_t d19 = vld1q_f32(delta +  76);
    float32x4_t d20 = vld1q_f32(delta +  80);
    float32x4_t d21 = vld1q_f32(delta +  84);
    float32x4_t d22 = vld1q_f32(delta +  88);
    float32x4_t d23 = vld1q_f32(delta +  92);
    float32x4_t d24 = vld1q_f32(delta +  96);
    float32x4_t d25 = vld1q_f32(delta + 100);
    float32x4_t d26 = vld1q_f32(delta + 104);
    float32x4_t d27 = vld1q_f32(delta + 108);
    float32x4_t d28 = vld1q_f32(delta + 112);
    float32x4_t d29 = vld1q_f32(delta + 116);
    float32x4_t d30 = vld1q_f32(delta + 120);
    float32x4_t d31 = vld1q_f32(delta + 124);

    float32x4_t gtv = vdupq_n_f32(gt);

    for (int k = 0; k < HK; k++) {
        float* row = S + k * 128;
        float32x4_t kv = vdupq_n_f32(kt[k]);

        float32x4_t s0  = vld1q_f32(row +   0); s0  = vfmaq_f32(vmulq_f32(s0 , gtv), kv, d0 ); vst1q_f32(row +   0, s0 );
        float32x4_t s1  = vld1q_f32(row +   4); s1  = vfmaq_f32(vmulq_f32(s1 , gtv), kv, d1 ); vst1q_f32(row +   4, s1 );
        float32x4_t s2  = vld1q_f32(row +   8); s2  = vfmaq_f32(vmulq_f32(s2 , gtv), kv, d2 ); vst1q_f32(row +   8, s2 );
        float32x4_t s3  = vld1q_f32(row +  12); s3  = vfmaq_f32(vmulq_f32(s3 , gtv), kv, d3 ); vst1q_f32(row +  12, s3 );
        float32x4_t s4  = vld1q_f32(row +  16); s4  = vfmaq_f32(vmulq_f32(s4 , gtv), kv, d4 ); vst1q_f32(row +  16, s4 );
        float32x4_t s5  = vld1q_f32(row +  20); s5  = vfmaq_f32(vmulq_f32(s5 , gtv), kv, d5 ); vst1q_f32(row +  20, s5 );
        float32x4_t s6  = vld1q_f32(row +  24); s6  = vfmaq_f32(vmulq_f32(s6 , gtv), kv, d6 ); vst1q_f32(row +  24, s6 );
        float32x4_t s7  = vld1q_f32(row +  28); s7  = vfmaq_f32(vmulq_f32(s7 , gtv), kv, d7 ); vst1q_f32(row +  28, s7 );
        float32x4_t s8  = vld1q_f32(row +  32); s8  = vfmaq_f32(vmulq_f32(s8 , gtv), kv, d8 ); vst1q_f32(row +  32, s8 );
        float32x4_t s9  = vld1q_f32(row +  36); s9  = vfmaq_f32(vmulq_f32(s9 , gtv), kv, d9 ); vst1q_f32(row +  36, s9 );
        float32x4_t s10 = vld1q_f32(row +  40); s10 = vfmaq_f32(vmulq_f32(s10, gtv), kv, d10); vst1q_f32(row +  40, s10);
        float32x4_t s11 = vld1q_f32(row +  44); s11 = vfmaq_f32(vmulq_f32(s11, gtv), kv, d11); vst1q_f32(row +  44, s11);
        float32x4_t s12 = vld1q_f32(row +  48); s12 = vfmaq_f32(vmulq_f32(s12, gtv), kv, d12); vst1q_f32(row +  48, s12);
        float32x4_t s13 = vld1q_f32(row +  52); s13 = vfmaq_f32(vmulq_f32(s13, gtv), kv, d13); vst1q_f32(row +  52, s13);
        float32x4_t s14 = vld1q_f32(row +  56); s14 = vfmaq_f32(vmulq_f32(s14, gtv), kv, d14); vst1q_f32(row +  56, s14);
        float32x4_t s15 = vld1q_f32(row +  60); s15 = vfmaq_f32(vmulq_f32(s15, gtv), kv, d15); vst1q_f32(row +  60, s15);
        float32x4_t s16 = vld1q_f32(row +  64); s16 = vfmaq_f32(vmulq_f32(s16, gtv), kv, d16); vst1q_f32(row +  64, s16);
        float32x4_t s17 = vld1q_f32(row +  68); s17 = vfmaq_f32(vmulq_f32(s17, gtv), kv, d17); vst1q_f32(row +  68, s17);
        float32x4_t s18 = vld1q_f32(row +  72); s18 = vfmaq_f32(vmulq_f32(s18, gtv), kv, d18); vst1q_f32(row +  72, s18);
        float32x4_t s19 = vld1q_f32(row +  76); s19 = vfmaq_f32(vmulq_f32(s19, gtv), kv, d19); vst1q_f32(row +  76, s19);
        float32x4_t s20 = vld1q_f32(row +  80); s20 = vfmaq_f32(vmulq_f32(s20, gtv), kv, d20); vst1q_f32(row +  80, s20);
        float32x4_t s21 = vld1q_f32(row +  84); s21 = vfmaq_f32(vmulq_f32(s21, gtv), kv, d21); vst1q_f32(row +  84, s21);
        float32x4_t s22 = vld1q_f32(row +  88); s22 = vfmaq_f32(vmulq_f32(s22, gtv), kv, d22); vst1q_f32(row +  88, s22);
        float32x4_t s23 = vld1q_f32(row +  92); s23 = vfmaq_f32(vmulq_f32(s23, gtv), kv, d23); vst1q_f32(row +  92, s23);
        float32x4_t s24 = vld1q_f32(row +  96); s24 = vfmaq_f32(vmulq_f32(s24, gtv), kv, d24); vst1q_f32(row +  96, s24);
        float32x4_t s25 = vld1q_f32(row + 100); s25 = vfmaq_f32(vmulq_f32(s25, gtv), kv, d25); vst1q_f32(row + 100, s25);
        float32x4_t s26 = vld1q_f32(row + 104); s26 = vfmaq_f32(vmulq_f32(s26, gtv), kv, d26); vst1q_f32(row + 104, s26);
        float32x4_t s27 = vld1q_f32(row + 108); s27 = vfmaq_f32(vmulq_f32(s27, gtv), kv, d27); vst1q_f32(row + 108, s27);
        float32x4_t s28 = vld1q_f32(row + 112); s28 = vfmaq_f32(vmulq_f32(s28, gtv), kv, d28); vst1q_f32(row + 112, s28);
        float32x4_t s29 = vld1q_f32(row + 116); s29 = vfmaq_f32(vmulq_f32(s29, gtv), kv, d29); vst1q_f32(row + 116, s29);
        float32x4_t s30 = vld1q_f32(row + 120); s30 = vfmaq_f32(vmulq_f32(s30, gtv), kv, d30); vst1q_f32(row + 120, s30);
        float32x4_t s31 = vld1q_f32(row + 124); s31 = vfmaq_f32(vmulq_f32(s31, gtv), kv, d31); vst1q_f32(row + 124, s31);
    }
}

static inline float ssm_v3_l2norm_inplace(float* x, int HK, float eps) {
    float32x4_t acc = vdupq_n_f32(0);
    for (int c = 0; c + 4 <= HK; c += 4) {
        float32x4_t v = vld1q_f32(x + c);
        acc = vfmaq_f32(acc, v, v);
    }
    float ss = vaddvq_f32(acc);
    float inv = 1.0f / sqrtf(ss + eps);
    float32x4_t invv = vdupq_n_f32(inv);
    for (int c = 0; c + 4 <= HK; c += 4) {
        float32x4_t v = vld1q_f32(x + c);
        vst1q_f32(x + c, vmulq_f32(v, invv));
    }
    return inv;
}

#endif

#endif
