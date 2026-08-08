
#ifndef STRATUM_ALGO_H
#define STRATUM_ALGO_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <Accelerate/Accelerate.h>

static inline float bf16_to_fp32(uint16_t b) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

static inline uint16_t fp32_to_bf16(float f) {
    union { uint32_t u; float f; } v; v.f = f;

    uint32_t rounding_bias = 0x7FFF + ((v.u >> 16) & 1);
    return (uint16_t)((v.u + rounding_bias) >> 16);
}

static inline void bf16v_to_fp32v(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(src[i]);
}

static inline void rmsnorm_fp32(
    const float* x, const float* weight, float eps, size_t d, float* y)
{
    double ss = 0.0;
    for (size_t i = 0; i < d; ++i) ss += (double)x[i] * (double)x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)d + (double)eps));
    for (size_t i = 0; i < d; ++i) y[i] = x[i] * scale * weight[i];
}

static inline void linear_fp32(
    const float* W, const float* x, const float* bias,
    size_t out_features, size_t in_features, float* y)
{
    if (bias) {
        memcpy(y, bias, out_features * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)out_features, (int)in_features,
                    1.0f, W, (int)in_features,
                    x, 1,
                    1.0f, y, 1);
    } else {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)out_features, (int)in_features,
                    1.0f, W, (int)in_features,
                    x, 1,
                    0.0f, y, 1);
    }
}

static inline void silu_fp32(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

static inline void swiglu_fp32(
    const float* gate, const float* up, float* y, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));
        y[i] = s * up[i];
    }
}

static inline void softmax_fp32_inplace(float* x, size_t n) {
    if (n == 0) return;
    float m = x[0];
    for (size_t i = 1; i < n; ++i) if (x[i] > m) m = x[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - m);
        sum += (double)x[i];
    }
    float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

static inline void rope_apply_inplace_fp32(
    float* x, int64_t position, size_t head_dim, size_t rot_dim, float theta)
{
    (void)head_dim;
    size_t half = rot_dim / 2;
    for (size_t k = 0; k < half; ++k) {
        float inv_freq = 1.0f / powf(theta, (float)(2 * k) / (float)rot_dim);
        float angle = (float)position * inv_freq;
        float c = cosf(angle), s = sinf(angle);
        float a = x[k];
        float b = x[k + half];

        x[k]        = a * c - b * s;
        x[k + half] = b * c + a * s;
    }
}

static inline void causal_mask_add_fp32(
    float* logits, size_t kv_len, int64_t q_pos)
{
    for (size_t j = 0; j < kv_len; ++j) {
        if ((int64_t)j > q_pos) logits[j] = -INFINITY;
    }
}

#endif
