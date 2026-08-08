
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int read_file(const char* path, void* buf, size_t expected) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    size_t got = fread(buf, 1, expected, f);
    fclose(f);
    if (got != expected) {
        fprintf(stderr, "%s: expected %zu bytes, got %zu\n", path, expected, got);
        return -1;
    }
    return 0;
}

int main(void) {
    const int N_BLOCKS = 4;
    const int N_FLOATS = N_BLOCKS * 256;

    block_q4_K blocks[N_BLOCKS];
    if (read_file("/tmp/q4k_test/blocks.bin", blocks, sizeof(blocks)) != 0) return 1;

    float expected[N_FLOATS];
    if (read_file("/tmp/q4k_test/expected.f32", expected, sizeof(expected)) != 0) return 1;

    float out[N_FLOATS];
    q4k_dequant_row_scalar(blocks, N_FLOATS, out);

    int n_mismatch = 0;
    float max_abs_diff = 0.0f;
    for (int i = 0; i < N_FLOATS; i++) {
        float diff = fabsf(out[i] - expected[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (diff > 1e-7f) {
            if (n_mismatch < 10) {
                fprintf(stderr, "  mismatch [%d]: got %.10f expected %.10f diff %.2e\n",
                        i, out[i], expected[i], diff);
            }
            n_mismatch++;
        }
    }
    printf("compared %d floats: %d mismatches, max abs diff = %.2e\n",
           N_FLOATS, n_mismatch, max_abs_diff);

    if (n_mismatch == 0) {
        printf("PASS scalar Q4_K dequant matches Python reference exactly.\n");
    } else {
        return 1;
    }

    float x[N_FLOATS];
    for (int i = 0; i < N_FLOATS; i++) {

        x[i] = sinf((float)i * 0.013f) * 1.7f;
    }

    double ref_dot = 0.0;
    for (int i = 0; i < N_FLOATS; i++) ref_dot += (double)expected[i] * x[i];

    float scalar_dot = q4k_dot_row_scalar(blocks, N_FLOATS, x);
    float neon_dot   = q4k_dot_row_neon(blocks, N_FLOATS, x);

    double rel_scalar = fabs((double)scalar_dot - ref_dot) / (fabs(ref_dot) + 1e-12);
    double rel_neon   = fabs((double)neon_dot   - ref_dot) / (fabs(ref_dot) + 1e-12);
    printf("\nDot product check (K=%d):\n", N_FLOATS);
    printf("  reference (sum expected*x) = %.10g\n", ref_dot);
    printf("  scalar fused dot           = %.10g  (rel err %.2e)\n", scalar_dot, rel_scalar);
    printf("  NEON   fused dot           = %.10g  (rel err %.2e)\n", neon_dot, rel_neon);

    if (rel_scalar > 1e-5 || rel_neon > 1e-5) {
        printf("FAIL: dot products disagree.\n");
        return 1;
    }
    printf("PASS scalar + NEON Q4_K dot products match reference.\n");
    return 0;
}
