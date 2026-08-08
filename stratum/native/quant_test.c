
#define _GNU_SOURCE

#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"

static inline uint16_t q4k_fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

#include "stratum_q5k.h"
#include "stratum_q5k_neon.h"
#include "stratum_q6k.h"
#include "stratum_q6k_neon.h"
#include "stratum_q3k.h"
#include "stratum_q3k_neon.h"
#include "stratum_q2k.h"
#include "stratum_q8_0.h"
#include "stratum_bf16.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int errors = 0;

#define ASSERT_REL(name, a, b, eps) do {                                     \
    float _ra = (a), _rb = (b);                                              \
    float _re = fabsf(_ra - _rb) / fmaxf(fabsf(_rb), 1e-6f);                 \
    if (_re > (eps)) {                                                       \
        fprintf(stderr, "FAIL %s: a=%g b=%g rel_err=%g\n",                   \
                name, _ra, _rb, _re); errors++;                              \
    } else {                                                                 \
        fprintf(stderr, "OK   %s: rel_err=%g\n", name, _re);                 \
    }                                                                        \
} while (0)

static void rand_blocks_q8_0(block_q8_0* b, int n) {
    for (int i = 0; i < n; i++) {
        b[i].d = q4k_fp32_to_fp16(0.01f + (rand() & 0xFFFF) / 65535.0f);
        for (int k = 0; k < 32; k++) b[i].qs[k] = (int8_t)(rand() & 0xFF);
    }
}

static void rand_bytes(void* p, size_t n) {
    uint8_t* b = (uint8_t*)p;
    for (size_t i = 0; i < n; i++) b[i] = rand() & 0xFF;
}

static void test_q8_0(void) {
    int n_blocks = 4;
    int K = n_blocks * 32;
    block_q8_0* blocks = malloc(n_blocks * sizeof(block_q8_0));
    rand_blocks_q8_0(blocks, n_blocks);

    float* x = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;

    double ref = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        float buf[32];
        q8_0_dequant_block_scalar(blocks + i, buf);
        for (int k = 0; k < 32; k++) ref += (double)buf[k] * x[i*32 + k];
    }
    float fs = q8_0_dot_row_scalar(blocks, K, x);
    float fn = q8_0_dot_row_neon(blocks, K, x);
    ASSERT_REL("q8_0 scalar fused", fs, (float)ref, 1e-5f);
    ASSERT_REL("q8_0 neon   fused", fn, (float)ref, 1e-5f);

    free(blocks); free(x);
}

static void test_q4k(void) {
    int n_blocks = 4;
    int K = n_blocks * 256;
    block_q4_K* blocks = malloc(n_blocks * sizeof(block_q4_K));
    rand_bytes(blocks, n_blocks * sizeof(block_q4_K));

    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d    = q4k_fp32_to_fp16(0.01f);
        blocks[i].dmin = q4k_fp32_to_fp16(0.001f);
    }

    float* x = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;

    double ref = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        float buf[256];
        q4k_dequant_row_scalar(blocks + i, 256, buf);
        for (int k = 0; k < 256; k++) ref += (double)buf[k] * x[i*256 + k];
    }
    float fn = q4k_dot_row_neon(blocks, K, x);
    ASSERT_REL("q4_K neon fused", fn, (float)ref, 1e-3f);
#if defined(__ARM_FEATURE_DOTPROD)

    {
        int8_t* xq = malloc(K);
        float*  xs = malloc((K/32) * sizeof(float));
        q4k_quantize_x_q8(x, K, xq, xs);
        float fsd = q4k_dot_row_sdot(blocks, K, xq, xs);
        ASSERT_REL("q4_K sdot  approx", fsd, (float)ref, 5e-2f);
        free(xq); free(xs);
    }
#endif

    free(blocks); free(x);
}

static void test_q5k(void) {
    int n_blocks = 4;
    int K = n_blocks * 256;
    block_q5_K* blocks = malloc(n_blocks * sizeof(block_q5_K));
    rand_bytes(blocks, n_blocks * sizeof(block_q5_K));
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d    = q4k_fp32_to_fp16(0.01f);
        blocks[i].dmin = q4k_fp32_to_fp16(0.001f);
    }

    float* x = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;

    double ref = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        float buf[256];
        q5k_dequant_block_scalar(blocks + i, buf);
        for (int k = 0; k < 256; k++) ref += (double)buf[k] * x[i*256 + k];
    }
    float fs = q5k_dot_row_scalar(blocks, K, x);
    float fn = q5k_dot_row_neon(blocks, K, x);
    ASSERT_REL("q5_K scalar fused", fs, (float)ref, 1e-3f);
    ASSERT_REL("q5_K neon   fused", fn, (float)ref, 1e-3f);

    free(blocks); free(x);
}

static void test_q6k(void) {
    int n_blocks = 4;
    int K = n_blocks * 256;
    block_q6_K* blocks = malloc(n_blocks * sizeof(block_q6_K));
    rand_bytes(blocks, n_blocks * sizeof(block_q6_K));
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d = q4k_fp32_to_fp16(0.01f);
    }

    float* x = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;

    double ref = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        float buf[256];
        q6k_dequant_block_scalar(blocks + i, buf);
        for (int k = 0; k < 256; k++) ref += (double)buf[k] * x[i*256 + k];
    }
    float fs = q6k_dot_row_scalar(blocks, K, x);
    float fn = q6k_dot_row_neon(blocks, K, x);
    ASSERT_REL("q6_K scalar fused", fs, (float)ref, 1e-3f);
    ASSERT_REL("q6_K neon   fused", fn, (float)ref, 1e-3f);

    free(blocks); free(x);
}

static void test_q3k(void) {
    int n_blocks = 4;
    int K = n_blocks * 256;
    block_q3_K* blocks = malloc(n_blocks * sizeof(block_q3_K));
    rand_bytes(blocks, n_blocks * sizeof(block_q3_K));
    for (int i = 0; i < n_blocks; i++) blocks[i].d = q4k_fp32_to_fp16(0.01f);

    float* x = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;

    double ref = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        float buf[256];
        q3k_dequant_block_scalar(blocks + i, buf);
        for (int k = 0; k < 256; k++) ref += (double)buf[k] * x[i*256 + k];
    }
    float fs = q3k_dot_row_scalar(blocks, K, x);
    ASSERT_REL("q3_K scalar fused", fs, (float)ref, 1e-3f);
#if defined(__ARM_NEON) || defined(__aarch64__)
    float fn = q3k_dot_row_neon(blocks, K, x);
    ASSERT_REL("q3_K neon   fused", fn, (float)ref, 1e-3f);
#endif

    free(blocks); free(x);
}

static void test_q2k(void) {
    int n_blocks = 4;
    int K = n_blocks * 256;
    block_q2_K* blocks = malloc(n_blocks * sizeof(block_q2_K));
    rand_bytes(blocks, n_blocks * sizeof(block_q2_K));
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d    = q4k_fp32_to_fp16(0.01f);
        blocks[i].dmin = q4k_fp32_to_fp16(0.001f);
    }

    float* x = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;

    double ref = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        float buf[256];
        q2k_dequant_block_scalar(blocks + i, buf);
        for (int k = 0; k < 256; k++) ref += (double)buf[k] * x[i*256 + k];
    }
    float fs = q2k_dot_row_scalar(blocks, K, x);
    ASSERT_REL("q2_K scalar fused", fs, (float)ref, 1e-3f);

    free(blocks); free(x);
}

static void test_bf16(void) {

    float xs[] = {0.0f, 1.0f, -1.0f, 0.5f, 3.14159f, -2.71828f, 1e-3f, 1e3f};
    int n = sizeof(xs)/sizeof(xs[0]);
    float maxerr = 0;
    for (int i = 0; i < n; i++) {
        uint16_t b = stratum_fp32_to_bf16(xs[i]);
        float r = stratum_bf16_to_fp32(b);
        float e = fabsf(r - xs[i]) / fmaxf(fabsf(xs[i]), 1e-6f);
        if (e > maxerr) maxerr = e;
    }
    if (maxerr > 1e-2f) {
        fprintf(stderr, "FAIL bf16 round-trip: max_rel_err=%g\n", maxerr);
        errors++;
    } else {
        fprintf(stderr, "OK   bf16 round-trip:    max_rel_err=%g\n", maxerr);
    }
}

int main(void) {
    srand(42);
    fprintf(stderr, "=== quant kernel cross-validation ===\n");
    test_bf16();
    test_q8_0();
    test_q4k();
    test_q5k();
    test_q6k();
    test_q3k();
    test_q2k();
    if (errors) {
        fprintf(stderr, "\n%d failure(s)\n", errors);
        return 1;
    }
    fprintf(stderr, "\nALL %d tests passed.\n", 8);
    return 0;
}
