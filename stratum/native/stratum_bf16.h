
#ifndef STRATUM_BF16_H
#define STRATUM_BF16_H

#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

static inline uint16_t stratum_fp32_to_bf16(float x) {

    uint32_t u;
    memcpy(&u, &x, 4);
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    return (uint16_t)((u + rounding_bias) >> 16);
}

static inline float stratum_bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

#if defined(__ARM_NEON) || defined(__aarch64__)

static inline float32x4_t stratum_bf16x4_to_fp32(uint16x4_t b) {
    uint32x4_t u = vshlq_n_u32(vmovl_u16(b), 16);
    return vreinterpretq_f32_u32(u);
}

static inline uint16x4_t stratum_fp32x4_to_bf16(float32x4_t f) {
    uint32x4_t u = vreinterpretq_u32_f32(f);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
    uint32x4_t rounded = vaddq_u32(u, bias);
    return vmovn_u32(vshrq_n_u32(rounded, 16));
}
#endif

#endif
