
#ifndef STRATUM_Q6K_H
#define STRATUM_Q6K_H

#include "stratum_q4k.h"
#include <stdint.h>
#include <string.h>

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} block_q6_K;
_Static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

static inline void q6k_dequant_block_scalar(const block_q6_K* b, float* y) {
    const float d = q4k_fp16_to_fp32(b->d);

    for (int n = 0; n < 256; n += 128) {
        const uint8_t* ql = b->ql + n / 2;
        const uint8_t* qh = b->qh + n / 4;
        const int8_t*  s  = b->scales + n / 16;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l]      & 0x0F) | ((qh[l] & 0x03) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
            int q3 = (int)((ql[l]      >>  4)  | (((qh[l] >> 4) & 0x03) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >>  4)  | (((qh[l] >> 6) & 0x03) << 4)) - 32;
            y[n + l +  0] = d * (float)s[is + 0] * (float)q1;
            y[n + l + 32] = d * (float)s[is + 2] * (float)q2;
            y[n + l + 64] = d * (float)s[is + 4] * (float)q3;
            y[n + l + 96] = d * (float)s[is + 6] * (float)q4;
        }
    }
}

static inline float q6k_dot_row_scalar(const block_q6_K* blocks, int K,
                                       const float* x) {
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q6_K* b = blocks + i;
        const float d = q4k_fp16_to_fp32(b->d);
        for (int n = 0; n < 256; n += 128) {
            const uint8_t* ql = b->ql + n / 2;
            const uint8_t* qh = b->qh + n / 4;
            const int8_t*  s  = b->scales + n / 16;

            float a1[2] = {0, 0}, a2[2] = {0, 0}, a3[2] = {0, 0}, a4[2] = {0, 0};
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l]      & 0x0F) | ((qh[l] & 0x03) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
                int q3 = (int)((ql[l]      >>  4)  | (((qh[l] >> 4) & 0x03) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >>  4)  | (((qh[l] >> 6) & 0x03) << 4)) - 32;
                a1[is] += (float)q1 * xp[n + l +  0];
                a2[is] += (float)q2 * xp[n + l + 32];
                a3[is] += (float)q3 * xp[n + l + 64];
                a4[is] += (float)q4 * xp[n + l + 96];
            }
            for (int is = 0; is < 2; is++) {
                dot += (double)d * (double)s[is + 0] * a1[is];
                dot += (double)d * (double)s[is + 2] * a2[is];
                dot += (double)d * (double)s[is + 4] * a3[is];
                dot += (double)d * (double)s[is + 6] * a4[is];
            }
        }
        xp += 256;
    }
    return (float)dot;
}

#endif
