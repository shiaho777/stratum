/* Batched-decode Q4_K kernel shootout (B independent streams sharing one
 * weight stream — the MULTISEQ workload). All contenders include their own
 * prep/unpack costs so the numbers are end-to-end per-stream cost.
 *
 *   A  — single-stream SDOT kernel, one stream at a time (baseline)
 *   M1 — q4k_dot_row_neon_multix (float activations, per-stream NEON)
 *   M2 — q4k_dot_row_sdot_multix_pack (int8-packed [group][stream][32])
 *   M3 — dequant weights to f32 + Accelerate cblas_sgemm (AMX-fp32 path)
 *
 * The BNNS int8/fp16 arms (M4/M5) were measured in the original probe and
 * structurally lost — per-group scales force prep+unpack+accum around the
 * FC call that cannot amortize across streams. Numbers and verdict are
 * recorded in docs/PERF-ROADMAP.md.
 *
 * Usage: bench_batch_q4k [weights.bin] [x.bin]   (same format as
 * bench_single_q4k; falls back to deterministic synthetic data if absent)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"

#define B 16

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    const char* wfile = argc > 1 ? argv[1] : NULL;
    const char* xfile = argc > 2 ? argv[2] : NULL;
    int K = 1024, N = 2048, nb = K / 256, ng = K / 32;

    block_q4_K* W = malloc((size_t)N * nb * sizeof(block_q4_K));
    float* x = malloc((size_t)K * sizeof(float));
    float* xs_flat = malloc((size_t)B * K * sizeof(float));
    const float* xs[B];

    if (wfile) {
        FILE* f = fopen(wfile, "rb");
        if (!f) { fprintf(stderr, "no W\n"); return 1; }
        fread(W, 1, (size_t)N * nb * sizeof(block_q4_K), f); fclose(f);
    } else {
        /* deterministic synthetic weights: valid Q4_K blocks, varied scales */
        srand(42);
        for (size_t i = 0; i < (size_t)N * nb; i++) {
            block_q4_K* b = W + i;
            for (int j = 0; j < 12; j++) b->scales[j] = (uint8_t)(32 + rand() % 96);
            for (int j = 0; j < 128; j++) b->qs[j] = (uint8_t)rand();
            uint16_t d16 = 0x3400 + (uint16_t)(rand() & 0x3FF);   /* ~0.5-1.0 */
            uint16_t m16 = 0x3000 + (uint16_t)(rand() & 0x1FF);
            memcpy(&b->d, &d16, 2); memcpy(&b->dmin, &m16, 2);
        }
    }
    if (xfile) {
        FILE* f = fopen(xfile, "rb");
        if (!f) { fprintf(stderr, "no x\n"); return 1; }
        fread(x, 4, K, f); fclose(f);
        for (int s = 0; s < B; s++) memcpy(xs_flat + (size_t)s * K, x, K * 4);
    } else {
        for (int s = 0; s < B; s++)
            for (int i = 0; i < K; i++)
                xs_flat[(size_t)s * K + i] = (float)(sin(i * 0.37 + s) + 0.1 * sin(i * 1.7));
    }
    for (int s = 0; s < B; s++) xs[s] = xs_flat + (size_t)s * K;

    /* per-stream quantized activations (A) and packed layout (M2) */
    int8_t* xq = malloc((size_t)B * K);
    float* xsc = malloc((size_t)B * ng * 4);
    int32_t* xsum = malloc((size_t)B * ng * 4);
    for (int s = 0; s < B; s++) {
        q4k_quantize_x_q8(xs[s], K, xq + (size_t)s * K, xsc + (size_t)s * ng);
        for (int g = 0; g < ng; g++)
            xsum[(size_t)s * ng + g] = q4k_sum_i8_32(xq + (size_t)s * K + (size_t)g * 32);
    }
    int8_t* xpack = malloc((size_t)ng * B * 32);
    float* scpack = malloc((size_t)ng * B * 4);
    int32_t* sumpack = malloc((size_t)ng * B * 4);
    for (int s = 0; s < B; s++)
        for (int g = 0; g < ng; g++) {
            int8_t* dst = xpack + ((size_t)g * B + s) * 32;
            float scv;
            q4k_quantize_x_q8_1b(xs[s] + (size_t)g * 32, dst, &scv);
            scpack[(size_t)g * B + s] = scv;
            sumpack[(size_t)g * B + s] = q4k_sum_i8_32(dst);
        }

    /* M3: dequant whole weight to f32 once (page-cost amortized in timing) */
    float* Wf = malloc((size_t)N * K * sizeof(float));
    float* XB = malloc((size_t)K * B * sizeof(float));  /* column-major for sgemm */
    for (int r = 0; r < N; r++) q4k_dequant_row_neon(W + (size_t)r * nb, K, Wf + (size_t)r * K);
    for (int s = 0; s < B; s++) for (int i = 0; i < K; i++)
        XB[(size_t)s * K + i] = xs[s][i];

    float* out = malloc((size_t)B * N * sizeof(float));
    float* ref = malloc((size_t)B * N * sizeof(float));

    /* ---- correctness: all batch arms vs single-stream SDOT reference ---- */
    for (int s = 0; s < B; s++)
        for (int r = 0; r < N; r++)
            ref[(size_t)s * N + r] = q4k_dot_row_sdot(W + (size_t)r * nb, K,
                                                    xq + (size_t)s * K,
                                                    xsc + (size_t)s * ng);
    for (int r = 0; r < N; r++)
        q4k_dot_row_neon_multix(W + (size_t)r * nb, K, xs, B, out + (size_t)r * B);
    double m1 = 0;
    for (int s = 0; s < B; s++) for (int r = 0; r < N; r++) {
        double e = fabs(out[(size_t)r * B + s] - ref[(size_t)s * N + r]);
        if (e > m1) m1 = e;
    }
    for (int r = 0; r < N; r++)
        q4k_dot_row_sdot_multix_pack(W + (size_t)r * nb, K, xpack, scpack, sumpack,
                                     B, out + (size_t)r * B);
    double m2 = 0;
    for (int s = 0; s < B; s++) for (int r = 0; r < N; r++) {
        double e = fabs(out[(size_t)r * B + s] - ref[(size_t)s * N + r]);
        if (e > m2) m2 = e;
    }
    /* M3 sgemm: C[N,B] = W[N,K] * X[K,B], column-major */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N, B, K, 1.0f, Wf, N, XB, K, 0.0f, out, N);
    double m3 = 0;
    for (int s = 0; s < B; s++) for (int r = 0; r < N; r++) {
        double e = fabs(out[(size_t)s * N + r] - ref[(size_t)s * N + r]);
        if (e > m3) m3 = e;
    }
    printf("correctness vs single-SDOT ref:  M1 max|d|=%.3g  M2=%.3g  M3=%.3g\n",
           m1, m2, m3);

    /* ---- timing: R full rounds over N rows, per-stream cost ---- */
    int R = 60;
    volatile float sink = 0;
    double t0, bestA = 1e9, best1 = 1e9, best2 = 1e9, best3 = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        t0 = now_s();
        for (int it = 0; it < R; it++)
            for (int s = 0; s < B; s++)
                for (int r = 0; r < N; r++)
                    sink += q4k_dot_row_sdot(W + (size_t)r * nb, K,
                                             xq + (size_t)s * K,
                                             xsc + (size_t)s * ng);
        double dt = now_s() - t0; if (dt < bestA) bestA = dt;

        t0 = now_s();
        for (int it = 0; it < R; it++)
            for (int r = 0; r < N; r++)
                q4k_dot_row_neon_multix(W + (size_t)r * nb, K, xs, B,
                                        out + (size_t)r * B);
        dt = now_s() - t0; if (dt < best1) best1 = dt;

        t0 = now_s();
        for (int it = 0; it < R; it++)
            for (int r = 0; r < N; r++)
                q4k_dot_row_sdot_multix_pack(W + (size_t)r * nb, K,
                                             xpack, scpack, sumpack, B,
                                             out + (size_t)r * B);
        dt = now_s() - t0; if (dt < best2) best2 = dt;

        t0 = now_s();
        for (int it = 0; it < R; it++) {
            /* dequant cost is per-weight-pass; batch sgemm amortizes it */
            for (int r = 0; r < N; r++)
                q4k_dequant_row_neon(W + (size_t)r * nb, K, Wf + (size_t)r * K);
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        N, B, K, 1.0f, Wf, N, XB, K, 0.0f, out, N);
            sink += out[0];
        }
        dt = now_s() - t0; if (dt < best3) best3 = dt;
    }

    double usA = bestA / R * 1e6 / B;
    double us1 = best1 / R * 1e6 / B;
    double us2 = best2 / R * 1e6 / B;
    double us3 = best3 / R * 1e6 / B;
    printf("\n=== B=%d  %dx%d  per-stream cost (lower is better) ===\n", B, N, K);
    printf("A  single-stream SDOT   : %6.1f us/stream\n", usA);
    printf("M1 neon_multix          : %6.1f us/stream  (%.2fx vs A)\n", us1, usA / us1);
    printf("M2 sdot_multix_pack     : %6.1f us/stream  (%.2fx vs A)\n", us2, usA / us2);
    printf("M3 dequant+sgemm (AMX)  : %6.1f us/stream  (%.2fx vs A)\n", us3, usA / us3);
    (void)sink;
    free(W); free(x); free(xs_flat); free(xq); free(xsc); free(xsum);
    free(xpack); free(scpack); free(sumpack); free(Wf); free(XB); free(out); free(ref);
    return 0;
}
