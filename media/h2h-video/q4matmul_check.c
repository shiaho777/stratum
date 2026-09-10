/* q4matmul_check.c — end-to-end single-matrix check: stratum's full
 * st_linear_dispatch on a real GGUF tensor vs a reference (x = ones,
 * so y[r] = row sum of dequantized row r).
 * usage: ./q4matmul_check <model.gguf> <tensor-name> <N> <K> */
#include <stdio.h>
#include <stdlib.h>
#include "stratum_gguf.h"
#include "stratum_linear.h"

StratumLinearState g_st;

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s model tensor N K\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    const GgufTensor* t = gguf_find_tensor(&g, argv[2]);
    if (!t) { fprintf(stderr, "tensor not found\n"); return 1; }
    int N = atoi(argv[3]), K = atoi(argv[4]);
    g_st.mmap_base = g.mmap_base;
    g_st.nchunks = 1;
    g_st.use_sdot = 0;
    g_st.use_metal = 0;
    float* x = malloc(sizeof(float) * K);
    float* y = malloc(sizeof(float) * N);
    for (int i = 0; i < K; i++) x[i] = 1.0f;
    int rc = st_linear_dispatch(t, x, y, N, K);
    printf("dispatch rc=%d type=%d\ny[0..15]:", rc, (int)t->type);
    for (int r = 0; r < 16; r++) printf(" %.5f", y[r]);
    printf("\n");
    return 0;
}
