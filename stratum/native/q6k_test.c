
#define _GNU_SOURCE
#include "stratum_gguf.h"
#include "stratum_q6k.h"
#include "stratum_q6k_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) return 1;

    const GgufTensor* t = NULL;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        if ((GgmlType)g.tensors[i].type == GGML_TYPE_Q6_K) {
            t = &g.tensors[i];
            break;
        }
    }
    if (!t) { fprintf(stderr, "no Q6_K tensor found\n"); return 1; }

    int K = (int)t->dims[0];
    int N = (int)t->dims[1];
    fprintf(stderr, "Q6_K tensor: %s  dims=[%d,%d]\n", t->name, K, N);

    srand(42);
    float* x = malloc(sizeof(float) * K);
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 65535.0f - 0.5f;

    int r = 7;
    int blocks_per_row = K / 256;
    const block_q6_K* row = (const block_q6_K*)(g.mmap_base + t->offset)
                          + (size_t)r * blocks_per_row;

    float ref = q6k_dot_row_scalar(row, K, x);

    float opt = q6k_dot_row_neon(row, K, x);

    double check = 0.0;
    {
        float buf[256];
        for (int i = 0; i < blocks_per_row; i++) {
            q6k_dequant_block_scalar(row + i, buf);
            for (int k = 0; k < 256; k++) check += (double)buf[k] * x[i*256 + k];
        }
    }

    float rel_err_neon = fabsf(opt - ref) / fmaxf(fabsf(ref), 1e-6f);
    float rel_err_chk  = fabsf((float)check - ref) / fmaxf(fabsf(ref), 1e-6f);

    fprintf(stderr, "  scalar dot       = %g\n", ref);
    fprintf(stderr, "  neon   dot       = %g  (rel err %g)\n", opt,  rel_err_neon);
    fprintf(stderr, "  scalar dequant+. = %g  (rel err %g)\n", (float)check, rel_err_chk);

    if (rel_err_neon > 1e-4f) {
        fprintf(stderr, "FAIL: neon vs scalar mismatch\n");
        return 1;
    }
    fprintf(stderr, "OK  Q6_K neon agrees with scalar to rel err %g\n", rel_err_neon);

    free(x);
    gguf_close(&g);
    return 0;
}
