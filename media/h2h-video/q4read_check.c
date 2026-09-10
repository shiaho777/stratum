/* q4read_check.c — dump row r of a Q4_K tensor via stratum's own reader.
 * usage: ./q4read_check <model.gguf> <tensor-name> <K> <row> */
#include <stdio.h>
#include "stratum_gguf.h"
#include "stratum_q4k.h"

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s model.gguf tensor K row\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    const GgufTensor* t = gguf_find_tensor(&g, argv[2]);
    if (!t) { fprintf(stderr, "tensor %s not found\n", argv[2]); return 1; }
    int K = atoi(argv[3]);
    long r = atol(argv[4]);
    printf("tensor=%s type=%d offset=%llu nelem=%lld K=%d row=%ld\n",
           argv[2], (int)t->type, (unsigned long long)t->offset,
           (long long)t->nelem, K, r);
    const block_q4_K* row = (const block_q4_K*)(g.mmap_base + t->offset)
                          + (size_t)r * (K / 256);
    float out[1024];
    q4k_dequant_row_scalar(row, K, out);
    for (int i = 0; i < 16; i++) printf("%.5f ", out[i]);
    printf("\n");
    return 0;
}
