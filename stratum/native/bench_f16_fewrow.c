/* llama.cpp #29869 on CPU F16: one weight-row convert, B independent
 * double accumulators. Must match B serial dots (max abs 0). */
#include "stratum_q4k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 256
#define K 64
#define B 8

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float serial_row(const uint16_t* row, const float* x, int k) {
    double acc = 0.0;
    for (int c = 0; c < k; c++)
        acc += (double)q4k_fp16_to_fp32(row[c]) * (double)x[c];
    return (float)acc;
}

static void fewrow(const uint16_t* W, const float* const* xs, float* const* ys,
                   int n, int k, int b) {
    for (int r = 0; r < n; r++) {
        const uint16_t* row = W + (size_t)r * k;
        double acc[16];
        for (int s = 0; s < b; s++) acc[s] = 0.0;
        for (int c = 0; c < k; c++) {
            double wv = (double)q4k_fp16_to_fp32(row[c]);
            for (int s = 0; s < b; s++)
                acc[s] += wv * (double)xs[s][c];
        }
        for (int s = 0; s < b; s++) ys[s][r] = (float)acc[s];
    }
}

int main(void) {
    uint16_t* W = malloc((size_t)N * K * 2);
    float* xflat = malloc((size_t)B * K * 4);
    float* yflat = malloc((size_t)B * N * 4);
    float* rflat = malloc((size_t)B * N * 4);
    const float* xs[B];
    float* ys[B];
    float* rs[B];
    for (int i = 0; i < N * K; i++) {
        float f = (float)sin(0.17 * i);
        uint32_t u;
        memcpy(&u, &f, 4);
        W[i] = (uint16_t)(u >> 16);
    }
    for (int s = 0; s < B; s++) {
        xs[s] = xflat + s * K;
        ys[s] = yflat + s * N;
        rs[s] = rflat + s * N;
        for (int c = 0; c < K; c++)
            xflat[s * K + c] = (float)(cos(0.11 * c + s) + 0.05 * s);
    }
    for (int s = 0; s < B; s++)
        for (int r = 0; r < N; r++)
            rs[s][r] = serial_row(W + (size_t)r * K, xs[s], K);
    fewrow(W, xs, ys, N, K, B);
    float md = 0.0f;
    for (int s = 0; s < B; s++)
        for (int r = 0; r < N; r++) {
            float d = fabsf(ys[s][r] - rs[s][r]);
            if (d > md) md = d;
        }
    printf("f16_fewrow_max_abs_diff=%.9g\n", md);
    if (md != 0.0f) { printf("f16_fewrow_fail\n"); return 2; }

    volatile float sink = 0;
    double best_s = 1e9, best_f = 1e9;
    for (int rep = 0; rep < 8; rep++) {
        double t0 = now_s();
        for (int it = 0; it < 40; it++)
            for (int s = 0; s < B; s++)
                for (int r = 0; r < N; r++)
                    sink += serial_row(W + (size_t)r * K, xs[s], K);
        double dt = now_s() - t0; if (dt < best_s) best_s = dt;
        t0 = now_s();
        for (int it = 0; it < 40; it++) {
            fewrow(W, xs, ys, N, K, B);
            sink += ys[0][0];
        }
        dt = now_s() - t0; if (dt < best_f) best_f = dt;
    }
    printf("f16_fewrow_serial_us=%.3f f16_fewrow_fused_us=%.3f f16_fewrow_speedup=%.3f sink=%.3g\n",
           best_s * 1e6 / 40, best_f * 1e6 / 40, best_s / best_f, (double)sink);
    printf("f16_fewrow_ok\n");
    free(W); free(xflat); free(yflat); free(rflat);
    return 0;
}
