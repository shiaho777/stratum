
#define _GNU_SOURCE

#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include "stratum_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) return 1;

    const GgufTensor* t = NULL;
    uint64_t best_bytes = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        if ((GgmlType)g.tensors[i].type != GGML_TYPE_Q4_K) continue;
        if (g.tensors[i].nbytes > best_bytes) {
            best_bytes = g.tensors[i].nbytes;
            t = &g.tensors[i];
        }
    }
    if (!t) { fprintf(stderr, "no Q4_K tensor found\n"); return 1; }

    int K = (int)t->dims[0];
    int N = (int)t->dims[1];
    fprintf(stderr, "Q4_K tensor: %s  dims=[%d,%d]  bytes=%llu  offset=%llu\n",
            t->name, K, N, (unsigned long long)t->nbytes,
            (unsigned long long)t->offset);

    srand(42);
    float* x = malloc(sizeof(float) * K);
    for (int i = 0; i < K; i++)
        x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;

    float* y_neon = malloc(sizeof(float) * N);
    double t0 = now_sec();
    int blocks_per_row = K / 256;
    for (int r = 0; r < N; r++) {
        const block_q4_K* row = (const block_q4_K*)((const uint8_t*)g.mmap_base + t->offset)
                              + (size_t)r * blocks_per_row;
        y_neon[r] = q4k_dot_row_neon(row, K, x);
    }
    double t1 = now_sec();
    fprintf(stderr, "  NEON  : %.4f s\n", t1 - t0);

    char metallib_path[1024];
    snprintf(metallib_path, sizeof metallib_path, "%s/stratum_q4k.metallib",
             "native");
    if (stratum_metal_init(metallib_path, g.mmap_base, g.mmap_size) != 0) return 1;

    float* y_metal = malloc(sizeof(float) * N);

    stratum_metal_q4k_sgemv(t->offset, x, y_metal, N, K);
    double t2 = now_sec();
    stratum_metal_q4k_sgemv(t->offset, x, y_metal, N, K);
    double t3 = now_sec();
    fprintf(stderr, "  Metal : %.4f s (after warmup)\n", t3 - t2);

    double max_abs = 0, max_rel = 0;
    int    bad_rows = 0;
    for (int r = 0; r < N; r++) {
        double a = (double)y_neon[r], b = (double)y_metal[r];
        double da = fabs(a - b);
        if (da > max_abs) max_abs = da;
        if (fabs(a) > 1e-4) {
            double dr = da / fabs(a);
            if (dr > max_rel) max_rel = dr;
            if (dr > 1e-3) bad_rows++;
        }
    }
    fprintf(stderr, "  max_abs_err = %g\n", max_abs);
    fprintf(stderr, "  max_rel_err (|a|>1e-4)  = %g\n", max_rel);
    fprintf(stderr, "  bad_rows (rel > 1e-3)   = %d / %d\n", bad_rows, N);

    if (max_abs > 1e-3) {
        fprintf(stderr, "FAIL: Metal abs err too large\n");
        return 1;
    }
    fprintf(stderr, "OK: Metal Q4_K sgemv matches NEON to abs err %g rel err %g\n",
            max_abs, max_rel);

    int reps = 20;
    double tA = now_sec();
    for (int i = 0; i < reps; i++) stratum_metal_q4k_sgemv(t->offset, x, y_metal, N, K);
    double tB = now_sec();
    double tC = now_sec();
    for (int i = 0; i < reps; i++) {
        for (int r = 0; r < N; r++) {
            const block_q4_K* row = (const block_q4_K*)((const uint8_t*)g.mmap_base + t->offset)
                                  + (size_t)r * blocks_per_row;
            y_neon[r] = q4k_dot_row_neon(row, K, x);
        }
    }
    double tD = now_sec();
    fprintf(stderr, "\nThroughput on this tensor (over %d reps):\n", reps);
    fprintf(stderr, "  Metal: %.1f sgemvs/s  (%.4f s/op)\n",
            reps / (tB - tA), (tB - tA) / reps);
    fprintf(stderr, "  NEON : %.1f sgemvs/s  (%.4f s/op)\n",
            reps / (tD - tC), (tD - tC) / reps);
    fprintf(stderr, "  speedup: %.1fx\n", (tD - tC) / (tB - tA));

    stratum_metal_shutdown();
    free(x); free(y_neon); free(y_metal);
    gguf_close(&g);
    return 0;
}
