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

// Strided variant: Q/K/V/out are region pointers inside ONE packed activation
// buffer (fbuf). Token row r sits at base + r*rowstride; the q/k/v/out
// sub-regions start at qoff/koff/voff/ooff floats into the row:
//   geom [buffer(8)]  = (rowstride, qoff, koff, voff)
//   geom2[buffer(9)]  = (orowstride, ooff, 0, 0)
// Enables zero-copy in-place attention over the fused [q|k|v|attn-out] rows:
// the out region of a row never overlaps its own or any other row's q/k/v.
kernel void h3_attn_prefill_strided(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device float*       Out [[buffer(3)]],
    constant uint& S            [[buffer(4)]],
    constant uint& H            [[buffer(5)]],
    constant uint& Hd           [[buffer(6)]],
    constant float& scale       [[buffer(7)]],
    constant uint4* geom        [[buffer(8)]],
    constant uint4* geom2       [[buffer(9)]],
    uint hg [[threadgroup_position_in_grid]],   // h * q_tiles + qt
    uint tid [[thread_position_in_threadgroup]])
{
    constexpr uint QT = 64;
    uint H_ = H;
    uint qtiles = (S + QT - 1) / QT;
    uint h = hg / qtiles;
    uint qt = hg % qtiles;
    uint q0 = qt * QT;

    uint rowstride = geom[0].x, qoff = geom[0].y, koff = geom[0].z, voff = geom[0].w;
    uint orowstride = geom2[0].x, ooff = geom2[0].y;

    if (tid >= QT) return;
    uint qi = tid;
    uint qs = q0 + qi;
    if (qs >= S) return;

    device const float* qh = Q + (size_t)qs * rowstride + qoff + (size_t)h * Hd;

    float acc[128];          // Hd <= 128 (H3 denoiser: 128)
    for (uint d = 0; d < Hd; d++) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (uint k0 = 0; k0 < S; k0 += 256) {
        uint ke = min(k0 + 256u, S);
        float sc[256];
        float tmax = -INFINITY;
        for (uint t = k0; t < ke; t++) {
            device const float* kt = K + (size_t)t * rowstride + koff + (size_t)h * Hd;
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
            device const float* vt = V + (size_t)t * rowstride + voff + (size_t)h * Hd;
            for (uint d = 0; d < Hd; d++) acc[d] = acc[d] * corr + p * vt[d];
        }
        m = mnew; l = lnew;
    }
    float inv = 1.0f / l;
    device float* op = Out + (size_t)qs * orowstride + ooff + (size_t)h * Hd;
    for (uint d = 0; d < Hd; d++) op[d] = acc[d] * inv;
    if (!isfinite(l) || !isfinite(m)) {
        op[0] = m; op[1] = l;
    }
}

/* Two-pass (non-online) softmax attention, fp32 throughout — the standard
 * PyTorch numeric contract, unlike the online-softmax kernel above. The dot
 * accumulates in float4 (4 partial sums, matching the CPU -ffast-math float4
 * vectorization — measured dot error ~4.5e-5 vs scalar float's ~5.3e-4 on the
 * deep-layer activation spikes) and the softmax uses precise::exp so the deep
 * spikes never overflow the fast-exp path. Scores are stored ONCE in a
 * threadgroup lg[] and exp() is cached in p[], so the working set is O(S+Hd)
 * (no [S,S] score matrix) — which is what lets a long sequence (768p, S~7.4k)
 * run without the O(S^2) memory blowup.
 *
 * One threadgroup per (head, query). grid = H*S, 64 threads.
 *
 * NOTE on the reductions: the max / lse are done by tid 0 scanning lg[]/p[]
 * serially and broadcasting via sm[0]. This is deliberate — the classic
 * cross-thread partial reduction (sm[tid]=x; barrier; tid0 reduces sm[0..63])
 * is miscompiled by the Metal backend when this kernel is inlined with the
 * float4 dot and the exp/acc loops, producing a wrong max (too small by up to
 * ~185) that overflows exp() into inf→NaN. Reading the stored lg[]/p[] values
 * back (a single source of truth) sidesteps it. tid-0 serial scan costs O(S)
 * additions per threadgroup — negligible next to the O(S·Hd) dot.
 */
kernel void h3_attn_two_pass(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device float*       Out [[buffer(3)]],
    constant uint& S            [[buffer(4)]],
    constant uint& H            [[buffer(5)]],
    constant uint& Hd           [[buffer(6)]],
    constant float& scale       [[buffer(7)]],
    constant uint4* geom        [[buffer(8)]],
    constant uint4* geom2       [[buffer(9)]],
    uint hg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    uint rowstride = geom[0].x, qoff = geom[0].y, koff = geom[0].z, voff = geom[0].w;
    uint orowstride = geom2[0].x, ooff = geom2[0].y;
    uint h = hg / S;
    uint a = hg % S;
    const device float* qh = Q + (size_t)a * rowstride + qoff + (size_t)h * Hd;

    threadgroup float sm[2];
    threadgroup float lg[512];
    threadgroup float p[512];

    /* 1) scores. Hd is the H3 denoiser's fixed 128 (divisible by 4). */
    for (uint b2 = tid; b2 < S; b2 += tg) {
        const device float* kh = K + (size_t)b2 * rowstride + koff + (size_t)h * Hd;
        float4 ss = float4(0.0f);
        for (uint d = 0; d < Hd; d += 4)
            ss += float4(qh[d], qh[d+1], qh[d+2], qh[d+3]) * float4(kh[d], kh[d+1], kh[d+2], kh[d+3]);
        lg[b2] = ((ss.x + ss.y) + (ss.z + ss.w)) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* 2) max (tid 0 serial scan) */
    if (tid == 0) { float mx = -INFINITY; for (uint b2 = 0; b2 < S; b2++) if (lg[b2] > mx) mx = lg[b2]; sm[0] = mx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mx = sm[0];

    /* 3) exp once per key -> p */
    for (uint b2 = tid; b2 < S; b2 += tg) p[b2] = precise::exp(lg[b2] - mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* 4) lse (tid 0 serial scan) -> inv */
    if (tid == 0) { float se = 0.0f; for (uint b2 = 0; b2 < S; b2++) se += p[b2]; sm[0] = 1.0f / se; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = sm[0];

    /* 5) weighted sum, reusing the cached p[] */
    device float* op = Out + (size_t)a * orowstride + ooff + (size_t)h * Hd;
    for (uint d = tid; d < Hd; d += tg) {
        float acc = 0.0f;
        for (uint b2 = 0; b2 < S; b2++)
            acc += p[b2] * V[(size_t)b2 * rowstride + voff + (size_t)h * Hd + d];
        op[d] = acc * inv;
    }
}

