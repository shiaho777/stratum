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
    constant uint& h0           [[buffer(10)]],
    uint hg [[threadgroup_position_in_grid]],   // h * q_tiles + qt
    uint tid [[thread_position_in_threadgroup]])
{
    constexpr uint QT = 64;
    uint H_ = H;
    uint qtiles = (S + QT - 1) / QT;
    uint h = h0 + hg / qtiles;
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
    constant uint& h0           [[buffer(10)]],
    uint hg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    constexpr uint BLOCK = 512;
    uint rowstride = geom[0].x, qoff = geom[0].y, koff = geom[0].z, voff = geom[0].w;
    uint orowstride = geom2[0].x, ooff = geom2[0].y;
    uint h = h0 + hg / S;
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

/* ---------------------------------------------------------------------------
 * V2 two-pass kernel (2026-09): same numeric contract as h3_attn_two_pass,
 * three structural fixes. All scores still re-computed with ONE shared
 * static-inline dot (bit-identical pass 1 / pass 2), softmax still
 * precise::exp, accumulation still fp32 — max|d| contract unchanged.
 *
 * 1. qk_dot4_4x: 4 independent float4 accumulators — the old single-chain
 *    dot serialized 32 FMAs (4-deep dependency); 4 chains restore ILP.
 * 2. Private max (pass 1) and private partial sums (pass 2) per thread,
 *    reduced by simd shuffle + a 1-row cross-simd sum instead of the tid-0
 *    serial O(BLOCK) scans. The tid-0 threadgroup-array scan was ~30k idle
 *    cycles per 512-key block (62 of 64 threads waiting at the barrier).
 *    The pass-1 max needs a max reduction (simd_max), pass 2 a sum
 *    reduction (simd_sum) — same stored-scores source of truth as V1
 *    (lg[] written by all threads, then reduced), so the miscompile V1
 *    documented for inline cross-thread partial reductions is not in play.
 * 3. PV accumulate over float4 V rows: each thread owns output dims
 *    (d0 = tid, d1 = tid + 64) as before, but loads V as two float4s per
 *    step and unrolls the key loop ×2 for load pairing.
 * --------------------------------------------------------------------------- */

static inline float qk_dot4_4x(const device float* qh, const device float* kh, uint Hd) {
    float4 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint d = 0;
    for (; d + 16 <= Hd; d += 16) {
        s0 += float4(qh[d+0],  qh[d+1],  qh[d+2],  qh[d+3])  * float4(kh[d+0],  kh[d+1],  kh[d+2],  kh[d+3]);
        s1 += float4(qh[d+4],  qh[d+5],  qh[d+6],  qh[d+7])  * float4(kh[d+4],  kh[d+5],  kh[d+6],  kh[d+7]);
        s2 += float4(qh[d+8],  qh[d+9],  qh[d+10], qh[d+11]) * float4(kh[d+8],  kh[d+9],  kh[d+10], kh[d+11]);
        s3 += float4(qh[d+12], qh[d+13], qh[d+14], qh[d+15]) * float4(kh[d+12], kh[d+13], kh[d+14], kh[d+15]);
    }
    for (; d + 4 <= Hd; d += 4)
        s0 += float4(qh[d], qh[d+1], qh[d+2], qh[d+3]) * float4(kh[d], kh[d+1], kh[d+2], kh[d+3]);
    float4 t = (s0 + s1) + (s2 + s3);
    for (; d < Hd; d++) t.x += qh[d] * kh[d];
    return (t.x + t.y) + (t.z + t.w);
}

/* warp-level reductions via simd shuffle (no shared-memory round trip) */
static inline float simd_reduce_max(float v) {
    v = max(v, simd_shuffle_xor(v, 16));
    v = max(v, simd_shuffle_xor(v, 8));
    v = max(v, simd_shuffle_xor(v, 4));
    v = max(v, simd_shuffle_xor(v, 2));
    v = max(v, simd_shuffle_xor(v, 1));
    return v;
}
static inline float simd_reduce_sum(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

kernel void h3_attn_two_pass_v2(
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
    constant uint& h0           [[buffer(10)]],
    uint hg [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    constexpr uint BLOCK = 512;
    uint rowstride = geom[0].x, qoff = geom[0].y, koff = geom[0].z, voff = geom[0].w;
    uint orowstride = geom2[0].x, ooff = geom2[0].y;
    uint h = h0 + hg / S;
    uint a = hg % S;
    const device float* qh = Q + (size_t)a * rowstride + qoff + (size_t)h * Hd;

    threadgroup float smax[2];   /* cross-simd broadcast: [max, 1/lse] */
    threadgroup float ssum[2];

    /* Pass 1: global max. Each thread strides keys, keeps a private max;
     * simd_reduce_max folds lanes, one cross-simd max folds simdgroups. */
    float pmax = -INFINITY;
    for (uint b2 = simd_group * 32 + simd_lane; b2 < S; b2 += tg)
        pmax = max(pmax, qk_dot4_4x(qh, K + (size_t)b2 * rowstride + koff + (size_t)h * Hd, Hd) * scale);
    pmax = simd_reduce_max(pmax);
    if (simd_lane == 0) smax[simd_group] = pmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float m = (tg > 32 && simd_lane < tg / 32) ? smax[simd_lane] : smax[0];
        m = max(m, simd_reduce_max(m));
        if (simd_lane == 0) smax[1] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gmax = smax[1];

    /* Pass 2: re-compute (bit-identical) scores, accumulate per-thread
     * partial lse and PV. Each thread owns output dims d0=tid, d1=tid+64.
     * V rows loaded as float4 pairs; key loop unrolled ×2 for load pairing. */
    const uint d0 = (simd_group * 32 + simd_lane), d1 = d0 + 64;
    float lse = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint bs = 0; bs < S; bs += BLOCK) {
        uint be = min(bs + BLOCK, S);
        for (uint b2 = bs + tid; b2 < be; b2 += tg) {
            float p = precise::exp(
                qk_dot4_4x(qh, K + (size_t)b2 * rowstride + koff + (size_t)h * Hd, Hd) * scale
                - gmax);
            lse += p;
            const device float* vh = V + (size_t)b2 * rowstride + voff + (size_t)h * Hd;
            for (uint d = d0; d < d0 + 64; d += 4) {
                acc0 += p * (vh[d] + vh[d+1] + vh[d+2] + vh[d+3]);
            }
            acc1 += p * (vh[d1] + vh[d1+1] + vh[d1+2] + vh[d1+3]);
        }
    }
    lse = simd_reduce_sum(lse);
    if (simd_lane == 0) ssum[simd_group] = lse;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float s = (tg > 32 && simd_lane < tg / 32) ? ssum[simd_lane] : ssum[0];
        s = simd_reduce_sum(s);
        if (simd_lane == 0) ssum[1] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* op = Out + (size_t)a * orowstride + ooff + (size_t)h * Hd;
    if (tid == 0) smax[0] = 1.0f / ssum[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = smax[0];
    op[d0] = acc0 * inv;
    op[d1] = acc1 * inv;
}

/* ---------------------------------------------------------------------------
 * Tiled kernel (2026-09, Phase 2): Bq=16 queries per threadgroup x Bk=256-key
 * KV tiles with per-tile online softmax. Motivated by measurement: at
 * S=7400 the v2 kernel demands ~2.6 TB/s of L2 (every query re-reads the
 * full K/V stream); tiling K/V across Bq queries divides that traffic by
 * Bq (16x -> ~160 GB/s) and turns the QK^T inner loop into a register-block
 * GEMM shape. Grid stays H-major so concurrent threadgroups share one
 * head's K/V slice (7.6 MB) in L2.
 *
 * Layout: 64 threads per threadgroup; thread tid owns dims [dq0,dq0+32)
 * (dq0 = (tid%4)*32) of exactly one query (qa = tid/4) and accumulates its
 * complete (l, PV) serially. Scores for the tile (16x256) live in
 * threadgroup memory (16 KB of the 32 KB budget). Per-thread running state
 * is one query's (m, l) + 8 float4 accumulators.
 *
 * Numeric contract: fp32 throughout, precise::exp, fp32 accumulation —
 * same as the two-pass kernels. The softmax is per-tile online (one
 * max/rescale per 256-key tile instead of one global max): each tile
 * rescales the previous tiles' (l, vacc) by exp(m_old - m_new) with m
 * the RUNNING max (flash-style; omitting this is a math error, not fp
 * noise — caught by the bench_h3_oracle RAMP synth). Exp arguments are
 * always <= 0 — no overflow path. Remaining differences vs the
 * global-max form are fp32 sum order only (same class as the v1/v2
 * dot-order drift); bench_h3_oracle asserts tiled-vs-v2 stays within
 * 10x of the v1-vs-v2 baseline.
 * --------------------------------------------------------------------------- */

static inline float qk_dot16(const device float* qh, const device float* kh, uint Hd) {
    /* same 4x-ILP shape as qk_dot4_4x; separate symbol so the tiled kernel
     * can evolve independently */
    float4 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint d = 0;
    for (; d + 16 <= Hd; d += 16) {
        s0 += float4(qh[d+0],  qh[d+1],  qh[d+2],  qh[d+3])  * float4(kh[d+0],  kh[d+1],  kh[d+2],  kh[d+3]);
        s1 += float4(qh[d+4],  qh[d+5],  qh[d+6],  qh[d+7])  * float4(kh[d+4],  kh[d+5],  kh[d+6],  kh[d+7]);
        s2 += float4(qh[d+8],  qh[d+9],  qh[d+10], qh[d+11]) * float4(kh[d+8],  kh[d+9],  kh[d+10], kh[d+11]);
        s3 += float4(qh[d+12], qh[d+13], qh[d+14], qh[d+15]) * float4(kh[d+12], kh[d+13], kh[d+14], kh[d+15]);
    }
    for (; d + 4 <= Hd; d += 4)
        s0 += float4(qh[d], qh[d+1], qh[d+2], qh[d+3]) * float4(kh[d], kh[d+1], kh[d+2], kh[d+3]);
    float4 t = (s0 + s1) + (s2 + s3);
    for (; d < Hd; d++) t.x += qh[d] * kh[d];
    return (t.x + t.y) + (t.z + t.w);
}

kernel void h3_attn_tiled(
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
    constant uint& h0           [[buffer(10)]],
    uint hg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    constexpr uint BQ = 16, BK = 256;
    uint rowstride = geom[0].x, qoff = geom[0].y, koff = geom[0].z, voff = geom[0].w;
    uint orowstride = geom2[0].x, ooff = geom2[0].y;
    uint qtiles = (S + BQ - 1) / BQ;
    uint h = h0 + hg / qtiles;
    uint qt = hg % qtiles;
    uint q0 = qt * BQ;

    threadgroup float sc[BQ * BK];

    /* each thread owns output dims [dq0, dq0+32) of exactly ONE query
     * (qa = tid/4) and accumulates its COMPLETE (l, PV) serially over all
     * tile keys. Stage 1 collaborates on all queries' scores into shared
     * sc[] (the L2-traffic amortization); stage 2 is per-thread complete,
     * so no cross-thread reduction exists anywhere (a strided stage-2
     * partition leaves each private copy holding a fraction of the mass). */
    const uint qa = tid / 4;              /* the one query this thread owns */
    const uint dq0 = (tid % 4) * 32;      /* first of 32 owned output dims */
    const bool qvalid = (q0 + qa) < S;

    const device float* qptr[BQ];
    for (uint i = 0; i < BQ; i++) {
        uint qs = q0 + i;
        qptr[i] = (qs < S) ? Q + (size_t)qs * rowstride + qoff + (size_t)h * Hd : Q;
    }

    /* per-thread running state for the owned query only. */
    float m_qa = -INFINITY, mo_qa = -INFINITY, l_qa = 0.0f;
    float4 vacc_qa[8];
    for (uint j = 0; j < 8; j++) vacc_qa[j] = 0.0f;

    for (uint k0 = 0; k0 < S; k0 += BK) {
        uint ke = min(k0 + BK, S);
        uint len = ke - k0;

        /* stage 1: scores for the whole tile (all 16 queries),
         * collaborative strided dots into shared sc[]. */
        for (uint i = 0; i < BQ; i++) {
            uint qs = q0 + i;
            if (qs >= S) break;
            for (uint b2 = k0 + tid; b2 < ke; b2 += tg) {
                float s = qk_dot16(qptr[i], K + (size_t)b2 * rowstride + koff + (size_t)h * Hd, Hd) * scale;
                sc[i * BK + (b2 - k0)] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* running max for the owned query: serial scan of its score row. */
        mo_qa = m_qa;
        if (qvalid) {
            float tm = mo_qa;
            for (uint k = 0; k < len; k++) tm = max(tm, sc[qa * BK + k]);
            m_qa = tm;
        }

        /* stage 2: complete (l, PV) for the owned query over all tile
         * keys, serially. Flash rescale rebases the previous tiles to the
         * new running max; when the max did not move corr is exactly 1.0. */
        if (qvalid) {
            float mi = m_qa;
            float corr = precise::exp(mo_qa - mi);
            const device float* vh0 = V + (size_t)k0 * rowstride + voff + (size_t)h * Hd + dq0;
            float lnew = l_qa * corr;
            for (uint j = 0; j < 8; j++) vacc_qa[j] *= corr;
            /* serial over keys (an x2 unroll was measured: no gain —
             * exp/V-load bound, not ILP bound — so keep the simple form). */
            for (uint k = 0; k < len; k++) {
                float p = precise::exp(sc[qa * BK + k] - mi);
                lnew += p;
                const device float* vhr = vh0 + (size_t)k * rowstride;
                for (uint j = 0; j < 8; j++)
                    vacc_qa[j] += p * float4(vhr[4*j], vhr[4*j+1], vhr[4*j+2], vhr[4*j+3]);
            }
            l_qa = lnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* write out: owned query, owned dims. */
    {
        uint qs = q0 + qa;
        if (qs < S) {
            float inv = 1.0f / l_qa;
            device float* op = Out + (size_t)qs * orowstride + ooff + (size_t)h * Hd + dq0;
            for (uint j = 0; j < 8; j++) {
                float4 v = vacc_qa[j] * inv;
                op[4*j] = v.x; op[4*j+1] = v.y; op[4*j+2] = v.z; op[4*j+3] = v.w;
            }
        }
    }
}

