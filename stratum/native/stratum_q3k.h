
#ifndef STRATUM_Q3K_H
#define STRATUM_Q3K_H

#include "stratum_q4k.h"
#include <stdint.h>
#include <string.h>

typedef struct {
    uint8_t  hmask[32];
    uint8_t  qs[64];
    uint8_t  scales[12];
    uint16_t d;
} block_q3_K;
_Static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");

static inline void q3k_unpack_scales(const uint8_t* in12, int8_t* scales_out) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, in12, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    memcpy(scales_out, aux, 16);
}

static inline void q3k_dequant_block_scalar(const block_q3_K* b, float* y) {
    const float d_all = q4k_fp16_to_fp32(b->d);
    int8_t scales[16];
    q3k_unpack_scales(b->scales, scales);
    const uint8_t* q  = b->qs;
    const uint8_t* hm = b->hmask;
    uint8_t m = 1;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl1 = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
                int q3 = (int)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4);
                *y++ = dl1 * (float)q3;
            }
            float dl2 = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
                int q3 = (int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                *y++ = dl2 * (float)q3;
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static inline float q3k_dot_row_scalar(const block_q3_K* blocks, int K,
                                       const float* x) {
    int n_blocks = K / 256;
    double dot = 0.0;
    const float* xp = x;
    for (int i = 0; i < n_blocks; i++) {
        const block_q3_K* b = blocks + i;
        const float d_all = q4k_fp16_to_fp32(b->d);
        int8_t scales[16];
        q3k_unpack_scales(b->scales, scales);
        const uint8_t* q  = b->qs;
        const uint8_t* hm = b->hmask;
        uint8_t m = 1;
        int is = 0;
        const float* xpp = xp;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl1 = d_all * (float)(scales[is++] - 32);
                float acc1 = 0.0f;
                for (int l = 0; l < 16; l++) {
                    int q3 = (int)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4);
                    acc1 += (float)q3 * xpp[l];
                }
                dot += (double)(dl1 * acc1);
                float dl2 = d_all * (float)(scales[is++] - 32);
                float acc2 = 0.0f;
                for (int l = 0; l < 16; l++) {
                    int q3 = (int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                    acc2 += (float)q3 * xpp[l + 16];
                }
                dot += (double)(dl2 * acc2);
                shift += 2;
                m <<= 1;
                xpp += 32;
            }
            q += 32;
        }
        xp += 256;
    }
    return (float)dot;
}

#endif
