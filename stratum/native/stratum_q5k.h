
#ifndef STRATUM_Q5K_H
#define STRATUM_Q5K_H

#include "stratum_q4k.h"
#include <stdint.h>
#include <string.h>

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
} block_q5_K;
_Static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");

static inline void q5k_dequant_block_scalar(const block_q5_K* b, float* y) {
    const float d    = q4k_fp16_to_fp32(b->d);
    const float dmin = q4k_fp16_to_fp32(b->dmin);
    const uint8_t* ql = b->qs;
    const uint8_t* qh = b->qh;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        q4k_get_scale_min(is + 0, b->scales, &sc, &m);
        const float d1 = d * (float)sc;  const float m1 = dmin * (float)m;
        q4k_get_scale_min(is + 1, b->scales, &sc, &m);
        const float d2 = d * (float)sc;  const float m2 = dmin * (float)m;
        for (int l = 0; l < 32; ++l) {
            int q = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            y[j + l] = d1 * (float)q - m1;
        }
        for (int l = 0; l < 32; ++l) {
            int q = (ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0);
            y[j + l + 32] = d2 * (float)q - m2;
        }
        ql += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

static inline float q5k_dot_row_scalar(const block_q5_K* blocks, int K,
                                       const float* x) {
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q5_K* b = blocks + i;
        const float d    = q4k_fp16_to_fp32(b->d);
        const float dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* ql = b->qs;
        const uint8_t* qh = b->qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            q4k_get_scale_min(is + 0, b->scales, &sc, &m);
            const float d1 = d * (float)sc;  const float m1 = dmin * (float)m;
            q4k_get_scale_min(is + 1, b->scales, &sc, &m);
            const float d2 = d * (float)sc;  const float m2 = dmin * (float)m;

            float qx1 = 0.f, sx1 = 0.f;
            for (int l = 0; l < 32; ++l) {
                int q = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
                qx1 += (float)q * xp[j + l];
                sx1 += xp[j + l];
            }
            float qx2 = 0.f, sx2 = 0.f;
            for (int l = 0; l < 32; ++l) {
                int q = (ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0);
                qx2 += (float)q * xp[j + l + 32];
                sx2 += xp[j + l + 32];
            }
            dot += (double)(d1 * qx1 - m1 * sx1 + d2 * qx2 - m2 * sx2);
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
        xp += 256;
    }
    return (float)dot;
}

#endif
