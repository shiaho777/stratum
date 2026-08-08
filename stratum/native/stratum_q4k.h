
#ifndef STRATUM_Q4K_H
#define STRATUM_Q4K_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define STRATUM_Q4K_HAS_NEON 1
#else
#define STRATUM_Q4K_HAS_NEON 0
#endif

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} block_q4_K;
_Static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

static inline float q4k_fp16_to_fp32(uint16_t h) {

#if defined(__ARM_FP16_FORMAT_IEEE) && (defined(__ARM_NEON) || defined(__aarch64__))
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
#else

    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {

            while ((mant & 0x400) == 0) { mant <<= 1; exp -= 1; }
            mant &= 0x3FF;
            exp += 1;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
#endif
}

static inline void q4k_get_scale_min(int j, const uint8_t* s, uint8_t* sc, uint8_t* m) {
    if (j < 4) {
        *sc = s[j]   & 63;
        *m  = s[j+4] & 63;
    } else {
        *sc = (s[j+4] & 0x0F) | ((s[j-4] >> 6) << 4);
        *m  = (s[j+4] >>   4) | ((s[j]   >> 6) << 4);
    }
}

static inline void q4k_dequant_block_scalar(const block_q4_K* b, float* y) {
    const float d   = q4k_fp16_to_fp32(b->d);
    const float dmin = q4k_fp16_to_fp32(b->dmin);
    const uint8_t* q = b->qs;
    int is = 0;

    for (int j = 0; j < 256; j += 64) {
        uint8_t sc1, m1b, sc2, m2b;
        q4k_get_scale_min(is + 0, b->scales, &sc1, &m1b);
        q4k_get_scale_min(is + 1, b->scales, &sc2, &m2b);
        const float d1 = d * sc1, mm1 = dmin * m1b;
        const float d2 = d * sc2, mm2 = dmin * m2b;
        for (int l = 0; l < 32; l++) y[j + l]      = d1 * (q[l] & 0xF) - mm1;
        for (int l = 0; l < 32; l++) y[j + l + 32] = d2 * (q[l] >>  4) - mm2;
        q += 32;
        is += 2;
    }
}

static inline void q4k_dequant_row_scalar(
    const block_q4_K* blocks, int K, float* y_row)
{
    int n_blocks = K / 256;
    for (int i = 0; i < n_blocks; i++) {
        q4k_dequant_block_scalar(blocks + i, y_row + i * 256);
    }
}

static inline float q4k_dot_row_scalar(
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
            const float d1 = d * sc1, mm1 = dmin * m1b;
            const float d2 = d * sc2, mm2 = dmin * m2b;
            float a1 = 0.0f, a2 = 0.0f;
            float xs1 = 0.0f, xs2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                a1 += (float)(q[l] & 0xF) * xp[l];
                xs1 += xp[l];
            }
            for (int l = 0; l < 32; l++) {
                a2 += (float)(q[l] >> 4) * xp[l + 32];
                xs2 += xp[l + 32];
            }
            dot += (double)(d1 * a1 - mm1 * xs1);
            dot += (double)(d2 * a2 - mm2 * xs2);
            q += 32;
            xp += 64;
            is += 2;
        }
    }
    return (float)dot;
}

#endif
