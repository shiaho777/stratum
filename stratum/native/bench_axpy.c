/* Calls the shipped stratum_axpy_f32 and compares every lane with a
 * scalar axpy. Hd=16 is the tiny-llama width. Hd=20 is the width the
 * old one-lane tail dropped. A few dozen floats, no model file.
 */
#include "stratum_axpy.h"
#include <math.h>
#include <stdio.h>

static void axpy_scalar(float* acc, float p, const float* v, int n) {
    for (int d = 0; d < n; d++) acc[d] += p * v[d];
}

static void run_one(int n, float* maxd, int* missed) {
    float a[32], b[32], v[32];
    for (int i = 0; i < n; i++) {
        a[i] = 0.1f * (float)(i + 1);
        b[i] = a[i];
        v[i] = 0.01f * (float)(i + 3);
    }
    float p = 0.5f;
    stratum_axpy_f32(a, p, v, n);
    axpy_scalar(b, p, v, n);
    *maxd = 0.0f;
    *missed = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > *maxd) *maxd = d;
        if (a[i] != b[i]) (*missed)++;
    }
}

int main(void) {
    float d16 = 0, d20 = 0;
    int m16 = 0, m20 = 0;
    run_one(16, &d16, &m16);
    run_one(20, &d20, &m20);
    printf("axpy_hd16_max_abs_diff=%.9g missed=%d\n", d16, m16);
    printf("axpy_hd20_max_abs_diff=%.9g missed=%d\n", d20, m20);
    if (d16 == 0.0f && d20 == 0.0f) printf("axpy_ok\n");
    else printf("axpy_mismatch\n");
    return (d16 == 0.0f && d20 == 0.0f) ? 0 : 2;
}
