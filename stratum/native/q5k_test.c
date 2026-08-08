
#define _GNU_SOURCE
#include "stratum_gguf.h"
#include "stratum_q5k.h"
#include "stratum_q5k_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) return 1;

    const GgufTensor* t = NULL;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        if ((GgmlType)g.tensors[i].type == GGML_TYPE_Q5_K) {
            t = &g.tensors[i];
            break;
        }
    }
    if (!t) { fprintf(stderr, "no Q5_K tensor found\n"); return 1; }

    int K = (int)t->dims[0];
    int N = (int)t->dims[1];
    fprintf(stderr, "Q5_K tensor: %s  dims=[%d,%d]\n", t->name, K, N);

    srand(7);
    float* x = malloc(sizeof(float) * K);
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 65535.0f - 0.5f;

    int r = 5;
    int blocks_per_row = K / 256;
    const block_q5_K* row = (const block_q5_K*)(g.mmap_base + t->offset)
                          + (size_t)r * blocks_per_row;

    float a = q5k_dot_row_scalar(row, K, x);

    double b = 0.0;
    {
        float buf[256];
        for (int i = 0; i < blocks_per_row; i++) {
            q5k_dequant_block_scalar(row + i, buf);
            for (int k = 0; k < 256; k++) b += (double)buf[k] * x[i*256 + k];
        }
    }

    float cn = q5k_dot_row_neon(row, K, x);

    float rel = fabsf(a - (float)b) / fmaxf(fabsf((float)b), 1e-6f);
    float rel_n = fabsf(cn - (float)b) / fmaxf(fabsf((float)b), 1e-6f);
    fprintf(stderr, "  fused dot   = %g\n", a);
    fprintf(stderr, "  dequant+dot = %g  (rel err scalar %g)\n", (float)b, rel);
    fprintf(stderr, "  neon  dot   = %g  (rel err neon   %g)\n", cn, rel_n);
    if (rel > 1e-4f || rel_n > 1e-4f) { fprintf(stderr, "FAIL\n"); return 1; }
    fprintf(stderr, "OK Q5_K scalar+neon agree, rel err scalar %g, neon %g\n", rel, rel_n);

    free(x);
    gguf_close(&g);
    return 0;
}
