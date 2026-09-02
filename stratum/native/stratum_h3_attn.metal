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

/* Float4 dot shared by both passes of the two-pass kernel. Kept as a single
 * static inline so the Metal backend emits IDENTICAL code for the pass-1 max
 * pass and the pass-2 sum/PV pass — the score must be bit-identical on both
 * passes or a max found on pass 1 can be exceeded by a re-computed score on
 * pass 2, overflowing exp() into inf→NaN. */
static inline float qk_dot4(const device float* qh, const device float* kh, uint Hd) {
    float4 ss = float4(0.0f);
    for (uint d = 0; d < Hd; d += 4)
        ss += float4(qh[d], qh[d+1], qh[d+2], qh[d+3]) * float4(kh[d], kh[d+1], kh[d+2], kh[d+3]);
    return (ss.x + ss.y) + (ss.z + ss.w);
}

/* Two-pass (non-online) softmax attention, fp32 throughout — the standard
 * PyTorch numeric contract, unlike the online-softmax kernel above. The dot
 * accumulates in float4 (4 partial sums, matching the CPU -ffast-math float4
 * vectorization — measured dot error ~4.5e-5 vs scalar float's ~5.3e-4 on the
 * deep-layer activation spikes) and the softmax uses precise::exp so the deep
 * spikes never overflow the fast-exp path.
 *
 * Flash-style: the KV range is split into blocks of BLOCK=512 keys so a long
 * sequence (768p, S~7.4k) runs with an O(S) working set — no [S,S] score
 * matrix. Two passes: pass 1 finds the global max per (head,query); pass 2
 * re-computes the (bit-identical) scores and accumulates sum + PV. The score
 * is re-computed rather than stored, since storing all H·S·S scores would be
 * O(S^2) memory.
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
    constexpr uint BLOCK = 512;
    uint rowstride = geom[0].x, qoff = geom[0].y, koff = geom[0].z, voff = geom[0].w;
    uint orowstride = geom2[0].x, ooff = geom2[0].y;
    uint h = hg / S;
    uint a = hg % S;
    const device float* qh = Q + (size_t)a * rowstride + qoff + (size_t)h * Hd;

    threadgroup float sm[2];
    threadgroup float lg[BLOCK];
    threadgroup float p[BLOCK];

    /* Pass 1: block-wise global max. */
    float gmax = -INFINITY;
    for (uint bs = 0; bs < S; bs += BLOCK) {
        uint be = min(bs + BLOCK, S);
        uint len = be - bs;
        for (uint b2 = bs + tid; b2 < be; b2 += tg)
            lg[b2 - bs] = qk_dot4(qh, K + (size_t)b2 * rowstride + koff + (size_t)h * Hd, Hd) * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float m = -INFINITY; for (uint i = 0; i < len; i++) if (lg[i] > m) m = lg[i]; if (m > gmax) gmax = m; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) sm[0] = gmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gmax = sm[0];

    /* Pass 2: re-compute (bit-identical) scores, exp once per key into p,
     * then accumulate the softmax denominator (lse) and the weighted sum.
     * Hd is the H3 denoiser's fixed 128 with tg=64, so each thread owns the
     * two output dims d=tid and d=tid+64. */
    const uint d0 = tid, d1 = tid + 64;
    float lse = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint bs = 0; bs < S; bs += BLOCK) {
        uint be = min(bs + BLOCK, S);
        uint len = be - bs;
        for (uint b2 = bs + tid; b2 < be; b2 += tg)
            lg[b2 - bs] = qk_dot4(qh, K + (size_t)b2 * rowstride + koff + (size_t)h * Hd, Hd) * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < len; i += tg) p[i] = precise::exp(lg[i] - gmax);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float s = 0.0f; for (uint i = 0; i < len; i++) s += p[i]; lse += s; }
        for (uint i = 0; i < len; i++) {
            float pp = p[i];
            acc0 += pp * V[(size_t)(bs + i) * rowstride + voff + (size_t)h * Hd + d0];
            acc1 += pp * V[(size_t)(bs + i) * rowstride + voff + (size_t)h * Hd + d1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) sm[0] = 1.0f / lse;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = sm[0];

    device float* op = Out + (size_t)a * orowstride + ooff + (size_t)h * Hd;
    op[d0] = acc0 * inv;
    op[d1] = acc1 * inv;
}

