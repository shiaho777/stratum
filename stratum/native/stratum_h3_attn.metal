#include <metal_stdlib>
using namespace metal;

// Flash-style prefill attention for H3 denoiser: per-head, streaming KV.
// Q,K,V layout: [S, H*Hd] token-major (token s, head h at s*H*Hd + h*Hd).
// One threadgroup (64 threads) per (head, q-tile of QT=64 rows).
// Online softmax over KV tiles of KT=256.

kernel void h3_attn_prefill(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device float*       Out [[buffer(3)]],
    constant uint& S            [[buffer(4)]],
    constant uint& H            [[buffer(5)]],
    constant uint& Hd           [[buffer(6)]],
    constant float& scale       [[buffer(7)]],
    uint hg [[threadgroup_position_in_grid]],   // h * q_tiles + qt
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    constexpr uint QT = 64;   // queries per tile
    uint H_ = H;
    uint qtiles = (S + QT - 1) / QT;
    uint h = hg / qtiles;
    uint qt = hg % qtiles;
    uint q0 = qt * QT;

    device const float* Qp = Q + (size_t)q0 * H_ * Hd + (size_t)h * Hd;

    // per-thread partial output row and online softmax state, one query row per thread
    if (tid >= QT) return;   // 64 threads, one query each
    uint qi = tid;           // query index within tile
    uint qs = q0 + qi;
    if (qs >= S) return;

    float acc[128];          // Hd <= 128 (H3 denoiser: 128)
    for (uint d = 0; d < Hd; d++) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    device const float* qh = Qp + (size_t)qi * H_ * Hd;

    for (uint k0 = 0; k0 < S; k0 += 256) {
        uint ke = min(k0 + 256u, S);
        // pass 1: scores for this KV tile
        float sc[256];
        float tmax = -INFINITY;
        for (uint t = k0; t < ke; t++) {
            device const float* kt = K + (size_t)t * H_ * Hd + (size_t)h * Hd;
            float dot = 0.0f;
            for (uint d = 0; d < Hd; d++) dot += qh[d] * kt[d];
            dot *= scale;
            sc[t - k0] = dot;
            if (dot > tmax) tmax = dot;
        }
        float mnew = max(m, tmax);
        float corr = exp(m - mnew);
        float lnew = l * corr;
        for (uint t = k0; t < ke; t++) {
            float p = exp(sc[t - k0] - mnew);
            lnew += p;
            device const float* vt = V + (size_t)t * H_ * Hd + (size_t)h * Hd;
            for (uint d = 0; d < Hd; d++) acc[d] = acc[d] * corr + p * vt[d];
        }
        m = mnew; l = lnew;
    }
    float inv = 1.0f / l;
    device float* op = Out + (size_t)qs * H_ * Hd + (size_t)h * Hd;
    for (uint d = 0; d < Hd; d++) op[d] = acc[d] * inv;
    if (!isfinite(l) || !isfinite(m)) {
        // leave a marker: m, l stamped into out row (caller inspects)
        op[0] = m; op[1] = l;
    }
}

kernel void h3_trivial(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    out[tid] = in[tid] * 2.0f;
}
