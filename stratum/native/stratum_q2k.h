
#ifndef STRATUM_Q2K_H
#define STRATUM_Q2K_H

#include "stratum_q4k.h"
#include <stdint.h>

typedef struct {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;
_Static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");

static inline void q2k_dequant_block_scalar(const block_q2_K* b, float* y) {
    const float d   = q4k_fp16_to_fp32(b->d);
    const float dmn = q4k_fp16_to_fp32(b->dmin);
    const uint8_t* q = b->qs;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc = b->scales[is++];
            float dl  = d   * (float)(sc & 0xF);
            float ml  = dmn * (float)(sc >>  4);
            for (int l = 0; l < 16; l++) {
                int q2 = (int)((q[l] >> shift) & 3);
                *y++ = dl * (float)q2 - ml;
            }
            sc = b->scales[is++];
            dl = d   * (float)(sc & 0xF);
            ml = dmn * (float)(sc >>  4);
            for (int l = 0; l < 16; l++) {
                int q2 = (int)((q[l + 16] >> shift) & 3);
                *y++ = dl * (float)q2 - ml;
            }
            shift += 2;
        }
        q += 32;
    }
}

static inline float q2k_dot_row_scalar(const block_q2_K* blocks, int K,
                                       const float* x) {
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q2_K* b = blocks + i;
        const float d   = q4k_fp16_to_fp32(b->d);
        const float dmn = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is = 0;
        const float* xpp = xp;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc1 = b->scales[is++];
                float dl1 = d   * (float)(sc1 & 0xF);
                float ml1 = dmn * (float)(sc1 >>  4);
                float qx1 = 0, sx1 = 0;
                for (int l = 0; l < 16; l++) {
                    int q2 = (int)((q[l] >> shift) & 3);
                    qx1 += (float)q2 * xpp[l];
                    sx1 += xpp[l];
                }
                dot += (double)(dl1 * qx1 - ml1 * sx1);

                uint8_t sc2 = b->scales[is++];
                float dl2 = d   * (float)(sc2 & 0xF);
                float ml2 = dmn * (float)(sc2 >>  4);
                float qx2 = 0, sx2 = 0;
                for (int l = 0; l < 16; l++) {
                    int q2 = (int)((q[l + 16] >> shift) & 3);
                    qx2 += (float)q2 * xpp[l + 16];
                    sx2 += xpp[l + 16];
                }
                dot += (double)(dl2 * qx2 - ml2 * sx2);

                shift += 2;
                xpp += 32;
            }
            q += 32;
        }
        xp += 256;
    }
    return (float)dot;
}

#endif
