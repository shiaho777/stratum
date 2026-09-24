
#include <metal_stdlib>
using namespace metal;

struct block_q4_K {
    half     d;
    half     dmin;
    uchar    scales[12];
    uchar    qs[128];
};

inline void unpack_scale_min(int j, const device uchar* scales, thread uchar& sc_out, thread uchar& m_out) {
    if (j < 4) {
        sc_out = scales[j]   & 63;
        m_out  = scales[j+4] & 63;
    } else {
        sc_out = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
        m_out  = (scales[j+4] >>  4) | ((scales[j  ] >> 6) << 4);
    }
}

kernel void q4k_sgemv_row(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    uint                     row         [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]],
    uint                     tg_size     [[threads_per_threadgroup]])
{
    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + (uint)row * blocks_per_row;
    const uint total_sb = blocks_per_row * 8;   /* 32-element sub-blocks */

    float partial = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;          /* 256-block index */
        uint j = sb & 7;           /* sub-block 0..7 */
        const device block_q4_K& b = row_blocks[i];
        const float d    = float(b.d);
        const float dmin = float(b.dmin);
        uchar sc, m;
        unpack_scale_min(j, b.scales, sc, m);
        float d_sc   = d    * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = b.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float qx = 0.0f, xs = 0.0f;
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
            float4 xv  = *(device const float4*)(x + elem_offset + l);
            /* dot() uses hardware dot product — fewer ALU instructions
             * than float4(nib) conversion + element-wise FMA */
            qx += dot(float4(nib), xv);
            xs += xv.x + xv.y + xv.z + xv.w;
        }
        partial += d_sc * qx - dmin_m * xs;
    }

    threadgroup float sdata[256];
    /* simd-level reduction first (register, no barrier), then combine the
     * few simdgroups via one barrier — much cheaper than a 6-step tree. */
    float sg = simd_sum(partial);
    uint simd_id = tid / 32u;
    uint lane    = tid % 32u;
    if (lane == 0) sdata[simd_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot = 0.0f;
        for (uint s = 0; s < nsimd; s++) tot += sdata[s];
        y[row] = tot;
    }
}

/* V12: Block-coalesced Q4_K sgemv — 32 threads per block, shared weight load.
 * Instead of 1 thread per sub-block (64 threads, 64 different blocks),
 * this kernel uses 32 threads per block: 4 threads per sub-block × 8 sub-blocks.
 * The 144-byte block is loaded once and shared across all 32 threads via
 * threadgroup memory. This maximizes memory coalescing and cache reuse.
 *
 * Threadgroup: 256 threads = 8 blocks processed in parallel.
 * Grid: ceil(N_rows / 8) threadgroups.
 */
kernel void q4k_sgemv_row_coalesced(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N_total     [[buffer(4)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint blocks_per_row = K / 256;
    const uint ROWS_PER_TG = 8;
    uint row_base = tgid * ROWS_PER_TG;
    uint local_row = tid / 32;   /* 0..7 → which row in this threadgroup */
    uint lane = tid % 32;        /* 0..31 → which element in sub-block */
    uint sub_block = lane / 4;   /* 0..7 → which 32-elem sub-block */
    uint elem = lane % 4;        /* 0..3 → which 4-element chunk */
    uint row = row_base + local_row;
    if (row >= N_total) return;

    device const block_q4_K* row_blocks = W + (uint)row * blocks_per_row;

    float partial = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; blk++) {
        const device block_q4_K& b = row_blocks[blk];
        const float d    = float(b.d);
        const float dmin = float(b.dmin);

        /* Each sub-block's scale/min */
        uchar sc, m;
        unpack_scale_min(sub_block, b.scales, sc, m);
        float d_sc   = d    * float(sc);
        float dmin_m = dmin * float(m);

        const device uchar* qs_pair = b.qs + (sub_block / 2) * 32;
        uint shift = (sub_block & 1) ? 4u : 0u;
        uint elem_offset = blk * 256 + sub_block * 32;

        /* 4 elements per thread x 2 loads at stride 16: threads (elem=0..3)
         * cover byte offsets {0,16},{4,20},{8,24},{12,28} = full 32-byte window.
         * (An earlier version looped l<4 — one load, only bytes 0..15: it read
         * HALF the sub-block. Wrong results; inflated bandwidth. Fixed.) */
        float qx = 0.0f, xs = 0.0f;
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 16) {
            uchar4 raw = *(device const uchar4*)(qs_pair + elem * 4 + l);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
            float4 nf  = float4(nib);
            float4 xv  = *(device const float4*)(x + elem_offset + elem * 4 + l);
            qx += dot(nf, xv);
            xs += xv.x + xv.y + xv.z + xv.w;
        }
        partial += d_sc * qx - dmin_m * xs;

        /* No barrier needed — each thread reads its own weight from device memory.
         * The coalescing benefit comes from 4 threads in the same sub-block
         * reading adjacent bytes from the same qs_pair, which is a single
         * 128-byte cache line fetch. */
    }

    /* Reduce 4 threads per sub-block → 8 partial sums per block → 1 per row */
    threadgroup float sdata[256];
    /* First: reduce within sub-block (4 threads → 1) */
    float v = partial;
    /* simd_sum across 4 threads in same sub-block is not directly available.
     * Use shuffle or shared memory. Simplest: write to shared and reduce. */
    sdata[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Reduce: 32 threads per row → 1 value.
     * 8 sub-blocks × 4 threads each = 32 threads.
     * Step 1: reduce 4→1 within each sub-block */
    if (lane % 4 == 0) {
        sdata[tid] = sdata[tid] + sdata[tid+1] + sdata[tid+2] + sdata[tid+3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Step 2: reduce 8 sub-block values → 1 */
    if (lane < 8 && lane % 4 == 0) {
        uint sb_idx = lane / 4;
        float sum = sdata[local_row * 32 + sb_idx * 4];
        for (int s = 1; s < 8; s++) {
            sum += sdata[local_row * 32 + s * 4];
        }
        if (sb_idx == 0) y[row] = sum;
    }
}

/* V15: Q4_K sgemv coalesced16 — 16 rows per threadgroup (vs V12's 8).
 * 256 threads = 16 rows x 16 threads/row. lane in [0,16): sub_block = lane>>1,
 * elem = lane&1 — 2 threads per sub-block, each covers 4 uchar4 loads at
 * stride 8 (byte offsets {0,8,16,24}+{4,12,20,28} = full 32-byte window).
 * Butterfly reduction over the 16 lanes of each row (rows are 16-aligned
 * within the 32-lane simd group, so masks 8/4/2/1 stay in-row).
 * Lane 0 writes y[row]. Tail-safe: rr clamp + write guarded by row < N_total.
 *
 * Opt-in via STRATUM_Q4K_COAL=1, dispatched for gguf_type 12, B==1, N >= 4096.
 * Probe (metal_q4k_coalesced16_probe.m, batched x8 submission): 166-184 GB/s
 * vs q4k_sgemv_row's 105-175 GB/s at N in [6144, 51200], K in {5120, 17408}
 * (1.02-1.59x); parity at N=2048. max|d| vs engine scalar ~1.7e-5 (same as
 * production kernels' FP reassociation noise). */
kernel void q4k_sgemv_row_coalesced16(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N_total     [[buffer(4)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_size != 256) return;
    const uint blocks_per_row = K / 256;
    const uint local_row = tid >> 4;    /* 0..15 */
    const uint lane      = tid & 15;    /* 0..15 */
    const uint row = tgid * 16u + local_row;
    const uint rr  = min(row, N_total - 1u);
    device const block_q4_K* row_blocks = W + (uint)rr * blocks_per_row;

    const uint sub_block = lane >> 1;   /* 0..7 */
    const uint elem      = lane & 1;    /* 0..1 */
    const uint shift     = (sub_block & 1) ? 4u : 0u;

    float partial = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; blk++) {
        const device block_q4_K& b = row_blocks[blk];
        const float d    = float(b.d);
        const float dmin = float(b.dmin);
        uchar sc, m;
        unpack_scale_min(sub_block, b.scales, sc, m);
        const float d_sc   = d    * float(sc);
        const float dmin_m = dmin * float(m);

        const device uchar* qp = b.qs + (sub_block / 2) * 32 + elem * 4;
        const uint xoff = blk * 256 + sub_block * 32 + elem * 4;

        float qx = 0.0f, xs = 0.0f;
        #pragma unroll
        for (int l4 = 0; l4 < 4; l4++) {
            uchar4 raw = *(device const uchar4*)(qp + 8u*l4);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & uchar4(0xF);
            float4 xv  = *(device const float4*)(x + xoff + 8u*l4);
            qx += dot(float4(nib), xv);
            xs += xv.x + xv.y + xv.z + xv.w;
        }
        partial += d_sc * qx - dmin_m * xs;
    }

    /* 16-lane reduction within each row group */
    float tot = partial;
    tot += simd_shuffle_xor(tot, 8);
    tot += simd_shuffle_xor(tot, 4);
    tot += simd_shuffle_xor(tot, 2);
    tot += simd_shuffle_xor(tot, 1);
    if (lane == 0 && row < N_total) y[row] = tot;
}

/* V16: multi-stream coalesced16 — fixes the B>1 sweep collapse on the
 * multiseq / best-of-N compute path. The per-row batched_b family
 * collapses under register pressure (B=8 sweep 14.9 GB/s vs 170.8 at B=1)
 * and bparallel re-reads weights per stream (19.6 GB/s). This keeps the
 * coalesced16 mapping (16 rows/tg, 16 thr/row, full 32-byte window) and
 * hoists the dequant OUT of the stream loop: nib4 registers are computed
 * once per weight chunk, all BC streams' FMA chains run against the
 * L2-resident x. Per-BC literal generation keeps partial[BC] in registers.
 * Probe (N=6144 K=5120, batched x8): B=4 56.5 GB/s, B=8 37.8 GB/s sweep
 * — 1.3x/2.5x over the incumbent kernels. Numeric: max|d| 3.4e-3 vs
 * engine scalar (same FP-noise magnitude as production kernels).
 * Opt-in STRATUM_Q4K_COAL=1 (same flag as the B=1 coalesced16), B 2..8,
 * nc_batch_add_streams path only (where qwen35 multiseq dispatches). */
#define DEFINE_Q4K_COAL_MB(BC) \
kernel void q4k_sgemv_coal16_mb_b##BC( \
    device const block_q4_K* W           [[buffer(0)]], \
    device const float*      x           [[buffer(1)]], \
    device float*            y           [[buffer(2)]], \
    constant uint&           K           [[buffer(3)]], \
    constant uint&           N_total     [[buffer(4)]], \
    constant uint&           B           [[buffer(5)]], \
    uint tgid    [[threadgroup_position_in_grid]], \
    uint tid     [[thread_position_in_threadgroup]], \
    uint tg_size [[threads_per_threadgroup]]) \
{ \
    if (tg_size != 256) return; \
    const uint blocks_per_row = K / 256; \
    const uint local_row = tid >> 4; \
    const uint lane      = tid & 15; \
    const uint row = tgid * 16u + local_row; \
    const uint rr  = min(row, N_total - 1u); \
    device const block_q4_K* row_blocks = W + (uint)rr * blocks_per_row; \
    const uint sub_block = lane >> 1; \
    const uint elem      = lane & 1; \
    const uint shift     = (sub_block & 1) ? 4u : 0u; \
    float partial[BC]; \
    for (uint s = 0; s < BC; s++) partial[s] = 0.0f; \
    for (uint blk = 0; blk < blocks_per_row; blk++) { \
        const device block_q4_K& b = row_blocks[blk]; \
        const float d    = float(b.d); \
        const float dmin = float(b.dmin); \
        uchar sc, m; \
        unpack_scale_min(sub_block, b.scales, sc, m); \
        const float d_sc   = d    * float(sc); \
        const float dmin_m = dmin * float(m); \
        const device uchar* qp = b.qs + (sub_block / 2) * 32 + elem * 4; \
        const uint xoff = blk * 256 + sub_block * 32 + elem * 4; \
        float4 nib4[4]; \
        _Pragma("unroll") \
        for (int l4 = 0; l4 < 4; l4++) { \
            uchar4 raw = *(device const uchar4*)(qp + 8u*l4); \
            nib4[l4] = float4((raw >> uchar4((uchar)shift)) & uchar4(0xF)); \
        } \
        for (uint s = 0; s < BC; s++) { \
            device const float* xs = x + (size_t)s * K + xoff; \
            float4 qx4 = float4(0.0f), xs4 = float4(0.0f); \
            _Pragma("unroll") \
            for (int l4 = 0; l4 < 4; l4++) { \
                float4 xv = *(device const float4*)(xs + 8u*l4); \
                qx4 += nib4[l4] * xv; \
                xs4 += xv; \
            } \
            partial[s] += d_sc * (qx4.x + qx4.y + qx4.z + qx4.w) \
                        - dmin_m * (xs4.x + xs4.y + xs4.z + xs4.w); \
        } \
    } \
    threadgroup float tgp[16][16]; \
    for (uint s = 0; s < BC; s++) { \
        float tot = partial[s]; \
        tot += simd_shuffle_xor(tot, 8); \
        tot += simd_shuffle_xor(tot, 4); \
        tot += simd_shuffle_xor(tot, 2); \
        tot += simd_shuffle_xor(tot, 1); \
        if (lane == 0) tgp[local_row][s] = tot; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0 && row < N_total) { \
        for (uint s = 0; s < BC; s++) y[(size_t)s * N_total + row] = tgp[local_row][s]; \
    } \
}

DEFINE_Q4K_COAL_MB(2)
DEFINE_Q4K_COAL_MB(3)
DEFINE_Q4K_COAL_MB(4)
DEFINE_Q4K_COAL_MB(5)
DEFINE_Q4K_COAL_MB(6)
DEFINE_Q4K_COAL_MB(7)
DEFINE_Q4K_COAL_MB(8)

/* V7: Q4_K sgemv with sparse block skip.
 * If the input x's 32-element sub-block max|x| < threshold * global_max_x,
 * skip that sub-block entirely — saves weight reads + computation.
 * The caller pre-computes x_block_max[K/32] and x_global_max. */
kernel void q4k_sgemv_row_sparse(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    device const float*      x_block_max [[buffer(4)]],  /* [K/32] */
    constant float&          x_global_max [[buffer(5)]],
    constant float&          threshold    [[buffer(6)]],  /* e.g. 0.01 */
    uint                     row         [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]],
    uint                     tg_size     [[threads_per_threadgroup]])
{
    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + (uint)row * blocks_per_row;
    const uint total_sb = blocks_per_row * 8;

    float skip_thresh = x_global_max * threshold;

    float partial = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        uint elem_offset = i * 256 + j * 32;

        /* Sparse skip: if this sub-block of x is negligible, skip it */
        if (x_block_max[elem_offset / 32] < skip_thresh) continue;

        const device block_q4_K& b = row_blocks[i];
        const float d    = float(b.d);
        const float dmin = float(b.dmin);
        uchar sc, m;
        unpack_scale_min(j, b.scales, sc, m);
        float d_sc   = d    * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = b.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        float4 qx4 = float4(0.0f), xs4 = float4(0.0f);
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
            float4 nf  = float4(nib);
            float4 xv  = *(device const float4*)(x + elem_offset + l);
            qx4 += nf * xv;
            xs4 += xv;
        }
        float qx = qx4.x + qx4.y + qx4.z + qx4.w;
        float xs = xs4.x + xs4.y + xs4.z + xs4.w;
        partial += d_sc * qx - dmin_m * xs;
    }

    threadgroup float sdata[256];
    float sg = simd_sum(partial);
    uint simd_id = tid / 32u;
    uint lane    = tid % 32u;
    if (lane == 0) sdata[simd_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot = 0.0f;
        for (uint s = 0; s < nsimd; s++) tot += sdata[s];
        y[row] = tot;
    }
}

/* Compute x_block_max[K/32] and x_global_max in one pass.
 * One threadgroup, each thread handles multiple 32-element blocks. */
kernel void compute_x_block_max(
    device const float* x     [[buffer(0)]],
    device float*       block_max [[buffer(1)]],  /* [K/32] */
    device float*       global_max [[buffer(2)]],
    constant uint&      K      [[buffer(3)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint nblocks = K / 32;
    float local_max = 0.0f;
    for (uint b = tid; b < nblocks; b += tg_size) {
        float bm = 0.0f;
        for (int i = 0; i < 32; i++) {
            float v = fabs(x[b * 32 + i]);
            if (v > bm) bm = v;
        }
        block_max[b] = bm;
        if (bm > local_max) local_max = bm;
    }
    /* Reduce global max */
    threadgroup float sdata[256];
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) *global_max = sdata[0];
}

/* Batched: one threadgroup per row computes BC output columns. BC is a
 * literal in each generated kernel, not a function constant, so the compiler
 * allocates exactly BC accumulators and unrolls the batch loop. */
#define DEFINE_Q4K_BATCH_KERNEL(BC) \
kernel void q4k_sgemv_row_batched_b##BC( \
    device const block_q4_K* W           [[buffer(0)]], \
    device const float*      x           [[buffer(1)]], \
    device float*            y           [[buffer(2)]], \
    constant uint&           K           [[buffer(3)]], \
    constant uint&           N           [[buffer(4)]], \
    constant uint&           B           [[buffer(5)]], \
    uint                     row         [[threadgroup_position_in_grid]], \
    uint                     tid         [[thread_position_in_threadgroup]], \
    uint                     tg_size     [[threads_per_threadgroup]]) \
{ \
    if (B != BC) return; \
    const uint blocks_per_row = K / 256; \
    device const block_q4_K* row_blocks = W + (uint)row * blocks_per_row; \
    const uint total_sb = blocks_per_row * 8; \
    float partial[BC]; \
    for (uint s = 0; s < BC; s++) partial[s] = 0.0f; \
    for (uint sb = tid; sb < total_sb; sb += tg_size) { \
        uint i = sb >> 3; \
        uint j = sb & 7; \
        const device block_q4_K& b = row_blocks[i]; \
        const float d = float(b.d); \
        const float dmin = float(b.dmin); \
        uchar sc, m; \
        unpack_scale_min(j, b.scales, sc, m); \
        float d_sc = d * float(sc); \
        float dmin_m = dmin * float(m); \
        const device uchar* qs_pair = b.qs + (j / 2) * 32; \
        uint shift = (j & 1) ? 4u : 0u; \
        uint elem_offset = i * 256 + j * 32; \
        float4 nib4[8]; \
        uchar4 mask = uchar4(0xF); \
        for (int l = 0; l < 32; l += 4) { \
            uchar4 raw = *(device const uchar4*)(qs_pair + l); \
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask; \
            nib4[l >> 2] = float4(nib); \
        } \
        for (uint s = 0; s < BC; s++) { \
            device const float* xs_ptr = x + s * K + elem_offset; \
            float4 qx4 = float4(0.0f), xs4 = float4(0.0f); \
            for (int l = 0; l < 32; l += 4) { \
                float4 xv = *(device const float4*)(xs_ptr + l); \
                qx4 += nib4[l >> 2] * xv; \
                xs4 += xv; \
            } \
            float qx = qx4.x + qx4.y + qx4.z + qx4.w; \
            float xs = xs4.x + xs4.y + xs4.z + xs4.w; \
            partial[s] += d_sc * qx - dmin_m * xs; \
        } \
    } \
    threadgroup float sdata[8 * BC]; \
    uint simd_id = tid / 32u; \
    uint lane = tid % 32u; \
    uint nsimd = (tg_size + 31u) / 32u; \
    for (uint s = 0; s < BC; s++) { \
        float v = simd_sum(partial[s]); \
        if (lane == 0) sdata[simd_id * BC + s] = v; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid < BC) { \
        float tot = 0.0f; \
        for (uint g = 0; g < nsimd; g++) tot += sdata[g * BC + tid]; \
        y[tid * N + row] = tot; \
    } \
}

DEFINE_Q4K_BATCH_KERNEL(1)
DEFINE_Q4K_BATCH_KERNEL(2)
DEFINE_Q4K_BATCH_KERNEL(3)
DEFINE_Q4K_BATCH_KERNEL(4)
DEFINE_Q4K_BATCH_KERNEL(5)
DEFINE_Q4K_BATCH_KERNEL(6)
DEFINE_Q4K_BATCH_KERNEL(7)
DEFINE_Q4K_BATCH_KERNEL(8)
DEFINE_Q4K_BATCH_KERNEL(9)
DEFINE_Q4K_BATCH_KERNEL(10)
DEFINE_Q4K_BATCH_KERNEL(11)
DEFINE_Q4K_BATCH_KERNEL(12)
DEFINE_Q4K_BATCH_KERNEL(13)
DEFINE_Q4K_BATCH_KERNEL(14)
DEFINE_Q4K_BATCH_KERNEL(15)
DEFINE_Q4K_BATCH_KERNEL(16)
DEFINE_Q4K_BATCH_KERNEL(17)
DEFINE_Q4K_BATCH_KERNEL(18)
DEFINE_Q4K_BATCH_KERNEL(19)
DEFINE_Q4K_BATCH_KERNEL(20)
DEFINE_Q4K_BATCH_KERNEL(21)
DEFINE_Q4K_BATCH_KERNEL(22)
DEFINE_Q4K_BATCH_KERNEL(23)
DEFINE_Q4K_BATCH_KERNEL(24)
DEFINE_Q4K_BATCH_KERNEL(25)
DEFINE_Q4K_BATCH_KERNEL(26)
DEFINE_Q4K_BATCH_KERNEL(27)
DEFINE_Q4K_BATCH_KERNEL(28)
DEFINE_Q4K_BATCH_KERNEL(29)
DEFINE_Q4K_BATCH_KERNEL(30)
DEFINE_Q4K_BATCH_KERNEL(31)
DEFINE_Q4K_BATCH_KERNEL(32)
#undef DEFINE_Q4K_BATCH_KERNEL

/* Batch-parallel fallback: one threadgroup per (row, stream). This rereads the
 * weight rows for each stream, but it restores full grid parallelism and keeps
 * register pressure identical to the single-stream kernel. It is useful when
 * serial-in-threadgroup B amortization loses occupancy at larger B. */
kernel void q4k_sgemv_row_bparallel(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint bidx = tgid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bidx >= B) return;
    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_sb = blocks_per_row * 8;

    float partial = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const float dmin = float(bl.dmin);
        uchar sc, m;
        unpack_scale_min(j, bl.scales, sc, m);
        float d_sc = d * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = bl.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float4 qx4 = float4(0.0f), xs4 = float4(0.0f);
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
            float4 nf = float4(nib);
            float4 xv = *(device const float4*)(xb + elem_offset + l);
            qx4 += nf * xv;
            xs4 += xv;
        }
        float qx = qx4.x + qx4.y + qx4.z + qx4.w;
        float xs = xs4.x + xs4.y + xs4.z + xs4.w;
        partial += d_sc * qx - dmin_m * xs;
    }

    threadgroup float sdata[256];
    float sg = simd_sum(partial);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) sdata[simd_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot = 0.0f;
        for (uint s = 0; s < nsimd; s++) tot += sdata[s];
        y[(size_t)bidx * N + row] = tot;
    }
}

/* Q4_K single-matrix bparallel for two adjacent streams per threadgroup.
 * Reuses the decoded weight row across two x vectors. */
kernel void q4k_sgemv_row_bparallel_g2(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint b0idx = tgid.y * 2u;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || b0idx >= B) return;
    uint b1idx = b0idx + 1u;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0idx * K;
    device const float* xb1 = x + (size_t)b1idx * K;
    const uint total_sb = blocks_per_row * 8;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const float dmin = float(bl.dmin);
        uchar sc, m;
        unpack_scale_min(j, bl.scales, sc, m);
        float d_sc = d * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = bl.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float4 qx0 = float4(0.0f), xs0 = float4(0.0f);
        float4 qx1 = float4(0.0f), xs1 = float4(0.0f);
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            float4 nf = float4((raw >> uchar4((uchar)shift)) & mask);
            float4 xv0 = *(device const float4*)(xb0 + elem_offset + l);
            qx0 += nf * xv0;
            xs0 += xv0;
            if (b1idx < B) {
                float4 xv1 = *(device const float4*)(xb1 + elem_offset + l);
                qx1 += nf * xv1;
                xs1 += xv1;
            }
        }
        float sx0 = xs0.x + xs0.y + xs0.z + xs0.w;
        float sx1 = xs1.x + xs1.y + xs1.z + xs1.w;
        partial0 += d_sc * (qx0.x + qx0.y + qx0.z + qx0.w) - dmin_m * sx0;
        partial1 += d_sc * (qx1.x + qx1.y + qx1.z + qx1.w) - dmin_m * sx1;
    }

    threadgroup float sdata0[256];
    threadgroup float sdata1[256];
    float sg0 = simd_sum(partial0);
    float sg1 = simd_sum(partial1);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        sdata0[simd_id] = sg0;
        sdata1[simd_id] = sg1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot0 = 0.0f, tot1 = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            tot0 += sdata0[s];
            tot1 += sdata1[s];
        }
        y[(size_t)b0idx * N + row] = tot0;
        if (b1idx < B) y[(size_t)b1idx * N + row] = tot1;
    }
}

/* Q4_K single-matrix g2 fused with residual add. This targets attention
 * output projection while retaining two-stream weight reuse. */
kernel void q4k_sgemv_row_bparallel_g2_add(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            residual    [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint b0idx = tgid.y * 2u;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || b0idx >= B) return;
    uint b1idx = b0idx + 1u;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0idx * K;
    device const float* xb1 = x + (size_t)b1idx * K;
    const uint total_sb = blocks_per_row * 8;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const float dmin = float(bl.dmin);
        uchar sc, m;
        unpack_scale_min(j, bl.scales, sc, m);
        float d_sc = d * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = bl.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float4 qx0 = float4(0.0f), xs0 = float4(0.0f);
        float4 qx1 = float4(0.0f), xs1 = float4(0.0f);
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            float4 nf = float4((raw >> uchar4((uchar)shift)) & mask);
            float4 xv0 = *(device const float4*)(xb0 + elem_offset + l);
            qx0 += nf * xv0;
            xs0 += xv0;
            if (b1idx < B) {
                float4 xv1 = *(device const float4*)(xb1 + elem_offset + l);
                qx1 += nf * xv1;
                xs1 += xv1;
            }
        }
        float sx0 = xs0.x + xs0.y + xs0.z + xs0.w;
        float sx1 = xs1.x + xs1.y + xs1.z + xs1.w;
        partial0 += d_sc * (qx0.x + qx0.y + qx0.z + qx0.w) - dmin_m * sx0;
        partial1 += d_sc * (qx1.x + qx1.y + qx1.z + qx1.w) - dmin_m * sx1;
    }

    threadgroup float sdata0[256];
    threadgroup float sdata1[256];
    float sg0 = simd_sum(partial0);
    float sg1 = simd_sum(partial1);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        sdata0[simd_id] = sg0;
        sdata1[simd_id] = sg1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot0 = 0.0f, tot1 = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            tot0 += sdata0[s];
            tot1 += sdata1[s];
        }
        residual[(size_t)b0idx * N + row] += tot0;
        if (b1idx < B) residual[(size_t)b1idx * N + row] += tot1;
    }
}

/* Q4_K bparallel matmul fused with residual add. This targets attention
 * output projection, replacing temp write + add kernel with one write. */
kernel void q4k_sgemv_row_bparallel_add(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            residual    [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint bidx = tgid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bidx >= B) return;
    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_sb = blocks_per_row * 8;

    float partial = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const float dmin = float(bl.dmin);
        uchar sc, m;
        unpack_scale_min(j, bl.scales, sc, m);
        float d_sc = d * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = bl.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float4 qx4 = float4(0.0f), xs4 = float4(0.0f);
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
            float4 nf = float4(nib);
            float4 xv = *(device const float4*)(xb + elem_offset + l);
            qx4 += nf * xv;
            xs4 += xv;
        }
        float qx = qx4.x + qx4.y + qx4.z + qx4.w;
        float xs = xs4.x + xs4.y + xs4.z + xs4.w;
        partial += d_sc * qx - dmin_m * xs;
    }

    threadgroup float sdata[256];
    float sg = simd_sum(partial);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) sdata[simd_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot = 0.0f;
        for (uint s = 0; s < nsimd; s++) tot += sdata[s];
        residual[(size_t)bidx * N + row] += tot;
    }
}

/* Dual Q4_K bparallel matmul for FFN gate/up. One threadgroup computes the
 * same (row, stream) for two same-shaped weight matrices, sharing x loads. */
kernel void q4k_dual_sgemv_row_bparallel(
    device const block_q4_K* W0          [[buffer(0)]],
    device const block_q4_K* W1          [[buffer(1)]],
    device const float*      x           [[buffer(2)]],
    device float*            y0          [[buffer(3)]],
    device float*            y1          [[buffer(4)]],
    constant uint&           K           [[buffer(5)]],
    constant uint&           N           [[buffer(6)]],
    constant uint&           B           [[buffer(7)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint bidx = tgid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bidx >= B) return;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row0 = W0 + row * blocks_per_row;
    device const block_q4_K* row1 = W1 + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_sb = blocks_per_row * 8;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& b0 = row0[i];
        const device block_q4_K& b1 = row1[i];

        uchar sc0, m0, sc1, m1;
        unpack_scale_min(j, b0.scales, sc0, m0);
        unpack_scale_min(j, b1.scales, sc1, m1);
        float d0_sc = float(b0.d) * float(sc0);
        float d0_min = float(b0.dmin) * float(m0);
        float d1_sc = float(b1.d) * float(sc1);
        float d1_min = float(b1.dmin) * float(m1);

        const device uchar* q0_pair = b0.qs + (j / 2) * 32;
        const device uchar* q1_pair = b1.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        uchar4 mask = uchar4(0xF);
        float4 qx0 = float4(0.0f), qx1 = float4(0.0f), xs4 = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw0 = *(device const uchar4*)(q0_pair + l);
            uchar4 raw1 = *(device const uchar4*)(q1_pair + l);
            float4 n0 = float4((raw0 >> uchar4((uchar)shift)) & mask);
            float4 n1 = float4((raw1 >> uchar4((uchar)shift)) & mask);
            float4 xv = *(device const float4*)(xb + elem_offset + l);
            qx0 += n0 * xv;
            qx1 += n1 * xv;
            xs4 += xv;
        }
        float sx = xs4.x + xs4.y + xs4.z + xs4.w;
        partial0 += d0_sc * (qx0.x + qx0.y + qx0.z + qx0.w) - d0_min * sx;
        partial1 += d1_sc * (qx1.x + qx1.y + qx1.z + qx1.w) - d1_min * sx;
    }

    threadgroup float sdata0[256];
    threadgroup float sdata1[256];
    float sg0 = simd_sum(partial0);
    float sg1 = simd_sum(partial1);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        sdata0[simd_id] = sg0;
        sdata1[simd_id] = sg1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot0 = 0.0f, tot1 = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            tot0 += sdata0[s];
            tot1 += sdata1[s];
        }
        size_t out = (size_t)bidx * N + row;
        y0[out] = tot0;
        y1[out] = tot1;
    }
}

/* Dual Q4_K FFN gate/up for two adjacent streams per threadgroup. This reuses
 * each decoded gate/up weight row across two x vectors, aiming at B=24/32. */
kernel void q4k_dual_sgemv_row_bparallel_g2(
    device const block_q4_K* W0          [[buffer(0)]],
    device const block_q4_K* W1          [[buffer(1)]],
    device const float*      x           [[buffer(2)]],
    device float*            y0          [[buffer(3)]],
    device float*            y1          [[buffer(4)]],
    constant uint&           K           [[buffer(5)]],
    constant uint&           N           [[buffer(6)]],
    constant uint&           B           [[buffer(7)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint b0idx = tgid.y * 2u;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || b0idx >= B) return;
    uint b1idx = b0idx + 1u;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row0 = W0 + row * blocks_per_row;
    device const block_q4_K* row1 = W1 + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0idx * K;
    device const float* xb1 = x + (size_t)b1idx * K;
    const uint total_sb = blocks_per_row * 8;

    float4 partial = float4(0.0f);
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& b0 = row0[i];
        const device block_q4_K& b1 = row1[i];

        uchar sc0, m0, sc1, m1;
        unpack_scale_min(j, b0.scales, sc0, m0);
        unpack_scale_min(j, b1.scales, sc1, m1);
        float d0_sc = float(b0.d) * float(sc0);
        float d0_min = float(b0.dmin) * float(m0);
        float d1_sc = float(b1.d) * float(sc1);
        float d1_min = float(b1.dmin) * float(m1);

        const device uchar* q0_pair = b0.qs + (j / 2) * 32;
        const device uchar* q1_pair = b1.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        uchar4 mask = uchar4(0xF);

        float4 qx00 = float4(0.0f), qx01 = float4(0.0f);
        float4 qx10 = float4(0.0f), qx11 = float4(0.0f);
        float4 xs0 = float4(0.0f), xs1 = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw0 = *(device const uchar4*)(q0_pair + l);
            uchar4 raw1 = *(device const uchar4*)(q1_pair + l);
            float4 n0 = float4((raw0 >> uchar4((uchar)shift)) & mask);
            float4 n1 = float4((raw1 >> uchar4((uchar)shift)) & mask);
            float4 xv0 = *(device const float4*)(xb0 + elem_offset + l);
            float4 xv1 = *(device const float4*)(xb1 + elem_offset + l);
            qx00 += n0 * xv0;
            qx01 += n1 * xv0;
            qx10 += n0 * xv1;
            qx11 += n1 * xv1;
            xs0 += xv0;
            xs1 += xv1;
        }
        float sx0 = xs0.x + xs0.y + xs0.z + xs0.w;
        float sx1 = xs1.x + xs1.y + xs1.z + xs1.w;
        partial.x += d0_sc * (qx00.x + qx00.y + qx00.z + qx00.w) - d0_min * sx0;
        partial.y += d1_sc * (qx01.x + qx01.y + qx01.z + qx01.w) - d1_min * sx0;
        partial.z += d0_sc * (qx10.x + qx10.y + qx10.z + qx10.w) - d0_min * sx1;
        partial.w += d1_sc * (qx11.x + qx11.y + qx11.z + qx11.w) - d1_min * sx1;
    }

    threadgroup float sdata0[256];
    threadgroup float sdata1[256];
    threadgroup float sdata2[256];
    threadgroup float sdata3[256];
    float sg0 = simd_sum(partial.x);
    float sg1 = simd_sum(partial.y);
    float sg2 = simd_sum(partial.z);
    float sg3 = simd_sum(partial.w);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        sdata0[simd_id] = sg0;
        sdata1[simd_id] = sg1;
        sdata2[simd_id] = sg2;
        sdata3[simd_id] = sg3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot0 = 0.0f, tot1 = 0.0f, tot2 = 0.0f, tot3 = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            tot0 += sdata0[s];
            tot1 += sdata1[s];
            tot2 += sdata2[s];
            tot3 += sdata3[s];
        }
        size_t out0 = (size_t)b0idx * N + row;
        y0[out0] = tot0;
        y1[out0] = tot1;
        if (b1idx < B) {
            size_t out1 = (size_t)b1idx * N + row;
            y0[out1] = tot2;
            y1[out1] = tot3;
        }
    }
}

/* Dual Q4_K FFN gate/up for two adjacent streams, fused with SwiGLU.
 * Writes out[b, row] = silu(gate[b,row]) * up[b,row], replacing the
 * gate/up temp writes plus the separate swiglu_inplace pass. */
kernel void q4k_dual_swiglu_row_bparallel_g2(
    device const block_q4_K* W0          [[buffer(0)]],
    device const block_q4_K* W1          [[buffer(1)]],
    device const float*      x           [[buffer(2)]],
    device float*            out         [[buffer(3)]],
    constant uint&           K           [[buffer(4)]],
    constant uint&           N           [[buffer(5)]],
    constant uint&           B           [[buffer(6)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint b0idx = tgid.y * 2u;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || b0idx >= B) return;
    uint b1idx = b0idx + 1u;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row0 = W0 + row * blocks_per_row;
    device const block_q4_K* row1 = W1 + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0idx * K;
    device const float* xb1 = x + (size_t)b1idx * K;
    const uint total_sb = blocks_per_row * 8;

    float4 partial = float4(0.0f);
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& b0 = row0[i];
        const device block_q4_K& b1 = row1[i];

        uchar sc0, m0, sc1, m1;
        unpack_scale_min(j, b0.scales, sc0, m0);
        unpack_scale_min(j, b1.scales, sc1, m1);
        float d0_sc = float(b0.d) * float(sc0);
        float d0_min = float(b0.dmin) * float(m0);
        float d1_sc = float(b1.d) * float(sc1);
        float d1_min = float(b1.dmin) * float(m1);

        const device uchar* q0_pair = b0.qs + (j / 2) * 32;
        const device uchar* q1_pair = b1.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        uchar4 mask = uchar4(0xF);

        float4 qx00 = float4(0.0f), qx01 = float4(0.0f);
        float4 qx10 = float4(0.0f), qx11 = float4(0.0f);
        float4 xs0 = float4(0.0f), xs1 = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw0 = *(device const uchar4*)(q0_pair + l);
            uchar4 raw1 = *(device const uchar4*)(q1_pair + l);
            float4 n0 = float4((raw0 >> uchar4((uchar)shift)) & mask);
            float4 n1 = float4((raw1 >> uchar4((uchar)shift)) & mask);
            float4 xv0 = *(device const float4*)(xb0 + elem_offset + l);
            float4 xv1 = *(device const float4*)(xb1 + elem_offset + l);
            qx00 += n0 * xv0;
            qx01 += n1 * xv0;
            qx10 += n0 * xv1;
            qx11 += n1 * xv1;
            xs0 += xv0;
            xs1 += xv1;
        }
        float sx0 = xs0.x + xs0.y + xs0.z + xs0.w;
        float sx1 = xs1.x + xs1.y + xs1.z + xs1.w;
        partial.x += d0_sc * (qx00.x + qx00.y + qx00.z + qx00.w) - d0_min * sx0;
        partial.y += d1_sc * (qx01.x + qx01.y + qx01.z + qx01.w) - d1_min * sx0;
        partial.z += d0_sc * (qx10.x + qx10.y + qx10.z + qx10.w) - d0_min * sx1;
        partial.w += d1_sc * (qx11.x + qx11.y + qx11.z + qx11.w) - d1_min * sx1;
    }

    threadgroup float sdata0[256];
    threadgroup float sdata1[256];
    threadgroup float sdata2[256];
    threadgroup float sdata3[256];
    float sg0 = simd_sum(partial.x);
    float sg1 = simd_sum(partial.y);
    float sg2 = simd_sum(partial.z);
    float sg3 = simd_sum(partial.w);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        sdata0[simd_id] = sg0;
        sdata1[simd_id] = sg1;
        sdata2[simd_id] = sg2;
        sdata3[simd_id] = sg3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float gate0 = 0.0f, up0 = 0.0f, gate1 = 0.0f, up1 = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            gate0 += sdata0[s];
            up0   += sdata1[s];
            gate1 += sdata2[s];
            up1   += sdata3[s];
        }
        out[(size_t)b0idx * N + row] = (gate0 / (1.0f + exp(-gate0))) * up0;
        if (b1idx < B) {
            out[(size_t)b1idx * N + row] = (gate1 / (1.0f + exp(-gate1))) * up1;
        }
    }
}

/* Dual Q4_K FFN gate/up for four adjacent streams per threadgroup. This is a
 * higher reuse experiment for B>=16; it may trade lower weight traffic for
 * higher register pressure. */
kernel void q4k_dual_sgemv_row_bparallel_g4(
    device const block_q4_K* W0          [[buffer(0)]],
    device const block_q4_K* W1          [[buffer(1)]],
    device const float*      x           [[buffer(2)]],
    device float*            y0          [[buffer(3)]],
    device float*            y1          [[buffer(4)]],
    constant uint&           K           [[buffer(5)]],
    constant uint&           N           [[buffer(6)]],
    constant uint&           B           [[buffer(7)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint bbase = tgid.y * 4u;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bbase >= B) return;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row0 = W0 + row * blocks_per_row;
    device const block_q4_K* row1 = W1 + row * blocks_per_row;
    const uint total_sb = blocks_per_row * 8;

    float p00 = 0.0f, p01 = 0.0f;
    float p10 = 0.0f, p11 = 0.0f;
    float p20 = 0.0f, p21 = 0.0f;
    float p30 = 0.0f, p31 = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& b0 = row0[i];
        const device block_q4_K& b1 = row1[i];

        uchar sc0, m0, sc1, m1;
        unpack_scale_min(j, b0.scales, sc0, m0);
        unpack_scale_min(j, b1.scales, sc1, m1);
        float d0_sc = float(b0.d) * float(sc0);
        float d0_min = float(b0.dmin) * float(m0);
        float d1_sc = float(b1.d) * float(sc1);
        float d1_min = float(b1.dmin) * float(m1);

        const device uchar* q0_pair = b0.qs + (j / 2) * 32;
        const device uchar* q1_pair = b1.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        uchar4 mask = uchar4(0xF);

        for (uint ss = 0; ss < 4; ss++) {
            uint bidx = bbase + ss;
            if (bidx >= B) break;
            device const float* xb = x + (size_t)bidx * K;
            float4 qx0 = float4(0.0f), qx1 = float4(0.0f), xs = float4(0.0f);
            for (int l = 0; l < 32; l += 4) {
                uchar4 raw0 = *(device const uchar4*)(q0_pair + l);
                uchar4 raw1 = *(device const uchar4*)(q1_pair + l);
                float4 n0 = float4((raw0 >> uchar4((uchar)shift)) & mask);
                float4 n1 = float4((raw1 >> uchar4((uchar)shift)) & mask);
                float4 xv = *(device const float4*)(xb + elem_offset + l);
                qx0 += n0 * xv;
                qx1 += n1 * xv;
                xs += xv;
            }
            float sx = xs.x + xs.y + xs.z + xs.w;
            float v0 = d0_sc * (qx0.x + qx0.y + qx0.z + qx0.w) - d0_min * sx;
            float v1 = d1_sc * (qx1.x + qx1.y + qx1.z + qx1.w) - d1_min * sx;
            if (ss == 0) { p00 += v0; p01 += v1; }
            else if (ss == 1) { p10 += v0; p11 += v1; }
            else if (ss == 2) { p20 += v0; p21 += v1; }
            else { p30 += v0; p31 += v1; }
        }
    }

    threadgroup float sdata0[256];
    threadgroup float sdata1[256];
    threadgroup float sdata2[256];
    threadgroup float sdata3[256];
    threadgroup float sdata4[256];
    threadgroup float sdata5[256];
    threadgroup float sdata6[256];
    threadgroup float sdata7[256];
    float sg0 = simd_sum(p00);
    float sg1 = simd_sum(p01);
    float sg2 = simd_sum(p10);
    float sg3 = simd_sum(p11);
    float sg4 = simd_sum(p20);
    float sg5 = simd_sum(p21);
    float sg6 = simd_sum(p30);
    float sg7 = simd_sum(p31);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        sdata0[simd_id] = sg0;
        sdata1[simd_id] = sg1;
        sdata2[simd_id] = sg2;
        sdata3[simd_id] = sg3;
        sdata4[simd_id] = sg4;
        sdata5[simd_id] = sg5;
        sdata6[simd_id] = sg6;
        sdata7[simd_id] = sg7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            if (tid == 0) tot += sdata0[s];
            else if (tid == 1) tot += sdata1[s];
            else if (tid == 2) tot += sdata2[s];
            else if (tid == 3) tot += sdata3[s];
            else if (tid == 4) tot += sdata4[s];
            else if (tid == 5) tot += sdata5[s];
            else if (tid == 6) tot += sdata6[s];
            else tot += sdata7[s];
        }
        uint stream = bbase + (tid >> 1);
        if (stream < B) {
            size_t out = (size_t)stream * N + row;
            if ((tid & 1u) == 0) y0[out] = tot;
            else y1[out] = tot;
        }
    }
}

/* Variant of the dual Q4_K FFN gate/up kernel with a shorter inner dependency
 * chain. It computes both rows with shared x loads and uses one float4 x-sum
 * per 32-element sub-block. */
kernel void q4k_dual_sgemv_row_bparallel_v2(
    device const block_q4_K* W0          [[buffer(0)]],
    device const block_q4_K* W1          [[buffer(1)]],
    device const float*      x           [[buffer(2)]],
    device float*            y0          [[buffer(3)]],
    device float*            y1          [[buffer(4)]],
    constant uint&           K           [[buffer(5)]],
    constant uint&           N           [[buffer(6)]],
    constant uint&           B           [[buffer(7)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint bidx = tgid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bidx >= B) return;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row0 = W0 + row * blocks_per_row;
    device const block_q4_K* row1 = W1 + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_sb = blocks_per_row * 8;
    uchar4 mask = uchar4(0xF);

    float2 partial = float2(0.0f);
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& b0 = row0[i];
        const device block_q4_K& b1 = row1[i];

        uchar sc0, m0, sc1, m1;
        unpack_scale_min(j, b0.scales, sc0, m0);
        unpack_scale_min(j, b1.scales, sc1, m1);
        float2 dsc = float2(float(b0.d) * float(sc0), float(b1.d) * float(sc1));
        float2 dmn = float2(float(b0.dmin) * float(m0), float(b1.dmin) * float(m1));

        const device uchar* q0 = b0.qs + (j / 2) * 32;
        const device uchar* q1 = b1.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float2 qx = float2(0.0f);
        float xs = 0.0f;
        for (int l = 0; l < 32; l += 4) {
            float4 xv = *(device const float4*)(xb + elem_offset + l);
            uchar4 raw0 = *(device const uchar4*)(q0 + l);
            uchar4 raw1 = *(device const uchar4*)(q1 + l);
            float4 n0 = float4((raw0 >> uchar4((uchar)shift)) & mask);
            float4 n1 = float4((raw1 >> uchar4((uchar)shift)) & mask);
            qx.x += dot(n0, xv);
            qx.y += dot(n1, xv);
            xs += xv.x + xv.y + xv.z + xv.w;
        }
        partial += dsc * qx - dmn * xs;
    }

    threadgroup float sdata0[256];
    threadgroup float sdata1[256];
    float sg0 = simd_sum(partial.x);
    float sg1 = simd_sum(partial.y);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        sdata0[simd_id] = sg0;
        sdata1[simd_id] = sg1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float tot0 = 0.0f, tot1 = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            tot0 += sdata0[s];
            tot1 += sdata1[s];
        }
        size_t out = (size_t)bidx * N + row;
        y0[out] = tot0;
        y1[out] = tot1;
    }
}

/* SIMD-batch: one threadgroup per row, 64 threads per stream. This keeps row
 * group count low while making B parallel inside the threadgroup. B<=16. */
kernel void q4k_sgemv_row_simdb_batched(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint                     row         [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]],
    uint                     tg_size     [[threads_per_threadgroup]])
{
    if (row >= N || B < 1 || B > 16) return;
    uint stream = tid >> 6;
    uint local_tid = tid & 63u;
    if (stream >= B) return;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)stream * K;
    const uint total_sb = blocks_per_row * 8;

    float partial = 0.0f;
    for (uint sb = local_tid; sb < total_sb; sb += 64u) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const float dmin = float(bl.dmin);
        uchar sc, m;
        unpack_scale_min(j, bl.scales, sc, m);
        float d_sc = d * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = bl.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float4 qx4 = float4(0.0f), xs4 = float4(0.0f);
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
            float4 nf = float4(nib);
            float4 xv = *(device const float4*)(xb + elem_offset + l);
            qx4 += nf * xv;
            xs4 += xv;
        }
        float qx = qx4.x + qx4.y + qx4.z + qx4.w;
        float xs = xs4.x + xs4.y + xs4.z + xs4.w;
        partial += d_sc * qx - dmin_m * xs;
    }

    threadgroup float sdata[32]; /* 2 simdgroups per stream, B<=16 */
    uint lane = tid & 31u;
    uint local_simd = (local_tid >> 5);
    float sg = simd_sum(partial);
    if (lane == 0) sdata[stream * 2u + local_simd] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < B) {
        y[(size_t)tid * N + row] = sdata[tid * 2u] + sdata[tid * 2u + 1u];
    }
    (void)tg_size;
}

kernel void q4k_argmax_batched(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device uint*             out         [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           V           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint                     bidx        [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]],
    uint                     tg_size     [[threads_per_threadgroup]])
{
    if (bidx >= B) return;
    const uint blocks_per_row = K / 256;
    device const float* xb = x + (size_t)bidx * K;
    float best = -INFINITY;
    uint best_i = 0;

    for (uint row = tid; row < V; row += tg_size) {
        device const block_q4_K* row_blocks = W + (size_t)row * blocks_per_row;
        const uint total_sb = blocks_per_row * 8;
        float acc = 0.0f;
        for (uint sb = 0; sb < total_sb; sb++) {
            uint i = sb >> 3;
            uint j = sb & 7;
            const device block_q4_K& bl = row_blocks[i];
            const float d = float(bl.d);
            const float dmin = float(bl.dmin);
            uchar sc, m;
            unpack_scale_min(j, bl.scales, sc, m);
            float d_sc = d * float(sc);
            float dmin_m = dmin * float(m);
            const device uchar* qs_pair = bl.qs + (j / 2) * 32;
            uint shift = (j & 1) ? 4u : 0u;
            uint elem_offset = i * 256 + j * 32;
            float4 qx4 = float4(0.0f), xs4 = float4(0.0f);
            uchar4 mask = uchar4(0xF);
            for (int l = 0; l < 32; l += 4) {
                uchar4 raw = *(device const uchar4*)(qs_pair + l);
                uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
                float4 nf = float4(nib);
                float4 xv = *(device const float4*)(xb + elem_offset + l);
                qx4 += nf * xv;
                xs4 += xv;
            }
            float qx = qx4.x + qx4.y + qx4.z + qx4.w;
            float xs = xs4.x + xs4.y + xs4.z + xs4.w;
            acc += d_sc * qx - dmin_m * xs;
        }
        if (acc > best) {
            best = acc;
            best_i = row;
        }
    }

    threadgroup float vals[256];
    threadgroup uint idxs[256];
    vals[tid] = best;
    idxs[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = vals[tid + s];
            uint oi = idxs[tid + s];
            if (ov > vals[tid]) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[bidx] = idxs[0];
}

/* Row-tiled batched Q4_K GEMM. Each threadgroup (256 threads = 8 simdgroups)
 * computes ROWS_TILE=8 output rows for all B columns. The B input columns for
 * the current 32-element sub-block are staged ONCE into threadgroup memory and
 * shared across all 8 rows -> global x traffic drops 8x vs the per-row kernel,
 * which is what lets aggregate throughput keep scaling with B. One simdgroup
 * per output row; lane L owns element L of each sub-block; per-column simd_sum.
 * B=1 bit-exact to q4k_sgemv_row. */
kernel void q4k_sgemv_rowtiled_batched(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint                     tgid        [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]])
{
    const uint ROWS_TILE = 8;
    const uint blocks_per_row = K / 256;
    uint simd_id = tid >> 5;     /* 0..7 -> row within tile */
    uint lane    = tid & 31u;
    uint row     = tgid * ROWS_TILE + simd_id;
    device const block_q4_K* row_blocks = W + (size_t)row * blocks_per_row;

    float acc[32];
    for (uint s = 0; s < B; s++) acc[s] = 0.0f;

    threadgroup float xt[16*256];   /* [B][256] for the current 256-block, B<=16 */

    for (uint i = 0; i < blocks_per_row; i++) {
        uint base = i*256;
        /* stage the full 256-block of all B columns ONCE, shared by 8 rows */
        for (uint idx = tid; idx < B*256u; idx += 256u) {
            uint s = idx >> 8, l = idx & 255u;
            xt[s*256 + l] = x[(size_t)s*K + base + l];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < N) {
            const device block_q4_K& b = row_blocks[i];
            float d = float(b.d), dmin = float(b.dmin);
            for (uint j = 0; j < 8; j++) {
                uchar sc, m;
                unpack_scale_min(j, b.scales, sc, m);
                float d_sc = d*float(sc), dmin_m = dmin*float(m);
                const device uchar* qs_pair = b.qs + (j/2)*32;
                uint shift = (j & 1) ? 4u : 0u;
                float nib = float((qs_pair[lane] >> shift) & 0xF);
                float coef = d_sc*nib - dmin_m;   /* (d_sc*nib - dmin_m)*xv per elem */
                uint eoff = j*32 + lane;
                for (uint s = 0; s < B; s++) acc[s] += coef * xt[s*256 + eoff];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < N) {
        for (uint s = 0; s < B; s++) {
            float v = simd_sum(acc[s]);
            if (lane == 0) y[s*N + row] = v;
        }
    }
}


/* SwiGLU: a[i] = silu(gate[i]) * up[i], silu(z) = z / (1 + exp(-z)).
 * Runs on the GPU so the FFN tail (gate->up->swiglu->down) stays resident
 * in GPU buffers across the whole block — one command buffer, one sync. */
kernel void swiglu_inplace(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint                gid  [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    float s = g / (1.0f + exp(-g));
    out[gid] = s * up[gid];
}


/* RMSNorm matching CPU: scale = 1/sqrt(mean(x^2)+eps) (double accum), y=x*scale*gain. */
kernel void rmsnorm_f32(
    device const float* x    [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    constant float&     eps  [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float sdata[256];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg) { float v = x[i]; local += v*v; }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(sdata[0]/float(n) + eps);
    for (uint i = tid; i < n; i += tg) y[i] = x[i]*scale*gain[i];
}

/* Fused residual-add + rmsnorm: x[i] += r[i], then y = rms(x)*gain.
 * Replaces a separate add_inplace + rmsnorm pair (one dispatch). Same math
 * order as the CPU path (residual update, then normalize). */
kernel void rmsnorm_resid_f32(
    device float*       x    [[buffer(0)]],   /* residual accumulator, updated */
    device const float* r    [[buffer(1)]],   /* addend */
    device const float* gain [[buffer(2)]],
    device float*       y    [[buffer(3)]],   /* normalized output */
    constant uint&      n    [[buffer(4)]],
    constant float&     eps  [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float sdata[256];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg) { float v = x[i] + r[i]; x[i] = v; local += v*v; }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(sdata[0]/float(n) + eps);
    for (uint i = tid; i < n; i += tg) y[i] = x[i]*scale*gain[i];
}

/* Fused per-head qk-norm + RoPE, in-place on a [nheads*hd] buffer.
 * One threadgroup per head: rmsnorm(xh)*gain then rotate pairs.
 * neox (qwen3): half-split pairs (k, k+pairs); else adjacent (2k, 2k+1).
 * Elements past rope_dim get the norm only. */
kernel void qknorm_rope_f32(
    device float*       x    [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    constant uint&      hd   [[buffer(2)]],
    constant float&     eps  [[buffer(3)]],
    constant uint&      nheads [[buffer(4)]],
    constant uint&      rope_dim [[buffer(5)]],
    constant int&       position [[buffer(6)]],
    constant float&     theta    [[buffer(7)]],
    constant uint&      neox     [[buffer(8)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    if (h >= nheads) return;
    device float* xh = x + (size_t)h * hd;
    threadgroup float sdata[256];
    float local = 0.0f;
    for (uint i = tid; i < hd; i += tg) { float v = xh[i]; local += v*v; }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(sdata[0]/float(hd) + eps);
    uint pairs = rope_dim/2;
    for (uint kk = tid; kk < pairs; kk += tg) {
        uint i0 = neox ? kk : 2*kk;
        uint i1 = neox ? kk + pairs : 2*kk + 1;
        float v0 = xh[i0]*scale*gain[i0];
        float v1 = xh[i1]*scale*gain[i1];
        float freq  = 1.0f / pow(theta, float(2*kk)/float(rope_dim));
        float angle = float(position) * freq;
        float c = cos(angle), s = sin(angle);
        xh[i0] = v0*c - v1*s;
        xh[i1] = v0*s + v1*c;
    }
    for (uint i = rope_dim + tid; i < hd; i += tg)
        xh[i] = xh[i]*scale*gain[i];
}

/* RoPE (interleaved pairs), matching CPU la_rope. Applies to all heads of a
 * [n_heads * head_dim] buffer; rope_dim pairs per head. */
kernel void rope_f32(
    device float*   x        [[buffer(0)]],
    constant uint&  head_dim [[buffer(1)]],
    constant uint&  rope_dim [[buffer(2)]],
    constant int&   position [[buffer(3)]],
    constant float& theta    [[buffer(4)]],
    constant uint&  neox     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint pairs = rope_dim/2;
    uint head = gid / pairs;
    uint k    = gid % pairs;
    device float* xh = x + head*head_dim;
    float freq  = 1.0f / pow(theta, float(2*k)/float(rope_dim));
    float angle = float(position) * freq;
    float c = cos(angle), s = sin(angle);
    /* neox (qwen3): half-split pairs (k, k+pairs); else adjacent (2k, 2k+1) */
    uint i0 = neox ? k : 2*k;
    uint i1 = neox ? k + pairs : 2*k + 1;
    float x0 = xh[i0], x1 = xh[i1];
    xh[i0] = x0*c - x1*s;
    xh[i1] = x0*s + x1*c;
}

/* Decode attention, one threadgroup per query head. GQA: kv_h = h*Nk/Nq.
 * K/V cache laid out [t * Nk*Hd + kv_h*Hd + d]. Softmax matches CPU
 * (max-subtract, exp, normalize). */
kernel void attn_decode_f32(
    device const float* q       [[buffer(0)]],   // [Nq*Hd]
    device const float* Kc      [[buffer(1)]],
    device const float* Vc      [[buffer(2)]],
    device float*       out     [[buffer(3)]],   // [Nq*Hd]
    constant uint&      Hd      [[buffer(4)]],
    constant uint&      Nq      [[buffer(5)]],
    constant uint&      Nk      [[buffer(6)]],
    constant uint&      kvlen   [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    uint kv_h = h * Nk / Nq;
    device const float* qh = q + h*Hd;
    threadgroup float sc[512];
    threadgroup float red[64];
    // scores
    for (uint t = tid; t < kvlen; t += tg) {
        device const float* kt = Kc + (size_t)t*Nk*Hd + kv_h*Hd;
        float dot = 0.0f;
        for (uint d = 0; d < Hd; d++) dot += qh[d]*kt[d];
        sc[t] = dot*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // max
    float m = -INFINITY;
    for (uint t = tid; t < kvlen; t += tg) m = max(m, sc[t]);
    red[tid] = m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s>0; s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float maxv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // exp + sum
    float lsum = 0.0f;
    for (uint t = tid; t < kvlen; t += tg) { float e = exp(sc[t]-maxv); sc[t]=e; lsum+=e; }
    red[tid]=lsum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s>0; s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv = 1.0f/red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // weighted sum over V, each thread handles some dims
    device float* oh = out + h*Hd;
    for (uint d = tid; d < Hd; d += tg) {
        float acc = 0.0f;
        for (uint t = 0; t < kvlen; t++) {
            device const float* vt = Vc + (size_t)t*Nk*Hd + kv_h*Hd;
            acc += sc[t]*inv*vt[d];
        }
        oh[d] = acc;
    }
}


/* Residual add: x[i] += y[i]. */
kernel void add_inplace_f32(
    device float*       x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] += y[gid];
}

/* Copy n floats src->dst (for writing k/v into the KV cache slot). */
kernel void copy_f32(
    device float*       dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) dst[gid] = src[gid];
}


/* ===== Batched (B independent sequences) elementwise/attention kernels =====
 * Each operates on B sequence-major rows. One weight stream / one command
 * buffer serves all B streams (the throughput axis). B=1 is bit-exact to the
 * single-stream kernels above. swiglu_inplace and add_inplace_f32 are reused
 * flat with n = B*Ff / B*H (no per-row state), so no batched variant needed. */

/* RMSNorm, one threadgroup per sequence b. x,y laid out [B][n] sequence-major;
 * gain[n] is shared across all sequences. Matches rmsnorm_f32 per row. */
kernel void rmsnorm_f32_batched(
    device const float* x    [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    constant float&     eps  [[buffer(4)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    device const float* xb = x + (size_t)b * n;
    device float*       yb = y + (size_t)b * n;
    threadgroup float sdata[256];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg) { float v = xb[i]; local += v*v; }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(sdata[0]/float(n) + eps);
    for (uint i = tid; i < n; i += tg) yb[i] = xb[i]*scale*gain[i];
}

/* RoPE for B sequences, each with its own position. x laid out [B][nheads*Hd].
 * positions[b] gives the rotation position for sequence b. Matches rope_f32. */
kernel void rope_f32_batched(
    device float*       x         [[buffer(0)]],
    constant uint&      head_dim  [[buffer(1)]],
    constant uint&      rope_dim  [[buffer(2)]],
    device const int*   positions [[buffer(3)]],
    constant float&     theta     [[buffer(4)]],
    constant uint&      nheads    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint pairs = rope_dim/2;
    uint per_seq = nheads * pairs;
    uint b    = gid / per_seq;
    uint rem  = gid % per_seq;
    uint head = rem / pairs;
    uint k    = rem % pairs;
    device float* xh = x + (size_t)b*nheads*head_dim + head*head_dim;
    float freq  = 1.0f / pow(theta, float(2*k)/float(rope_dim));
    float angle = float(positions[b]) * freq;
    float c = cos(angle), s = sin(angle);
    float x0 = xh[2*k], x1 = xh[2*k+1];
    xh[2*k]   = x0*c - x1*s;
    xh[2*k+1] = x0*s + x1*c;
}

/* Scatter B freshly computed k (or v) vectors [B][Nk*Hd] into a per-sequence
 * KV cache laid out [B][max_kv][Nk*Hd] (one layer's slab). Sequence b writes
 * into slot slots[b]. grid = B*Nk*Hd threads. */
kernel void kv_scatter_f32(
    device const float* src     [[buffer(0)]],   // [B][Nk*Hd]
    device float*       cache   [[buffer(1)]],   // [B][max_kv][Nk*Hd]
    device const int*   slots   [[buffer(2)]],   // [B]
    constant uint&      row     [[buffer(3)]],   // Nk*Hd
    constant uint&      max_kv  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint b = gid / row;
    uint d = gid % row;
    size_t dst = (size_t)b*max_kv*row + (size_t)slots[b]*row + d;
    cache[dst] = src[(size_t)b*row + d];
}

/* Fuse K RoPE and K/V cache writes for decode. The rotated K is written
 * directly to cache to avoid a pair-level read/write race inside ksrc. */
kernel void kv_rope_scatter_f32(
    device const float* ksrc      [[buffer(0)]],   // [B][Nk*Hd]
    device const float* vsrc      [[buffer(1)]],   // [B][Nk*Hd]
    device float*       kcache    [[buffer(2)]],   // [B][max_kv][Nk*Hd]
    device float*       vcache    [[buffer(3)]],
    device const int*   slots     [[buffer(4)]],
    device const int*   positions [[buffer(5)]],
    constant uint&      head_dim  [[buffer(6)]],
    constant uint&      rope_dim  [[buffer(7)]],
    constant uint&      nheads    [[buffer(8)]],
    constant uint&      row       [[buffer(9)]],
    constant uint&      max_kv    [[buffer(10)]],
    constant float&     theta     [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint b = gid / row;
    uint d = gid % row;
    uint head = d / head_dim;
    uint hd_i = d - head * head_dim;
    device const float* kb = ksrc + (size_t)b * row;

    float kv = kb[d];
    if (head < nheads && hd_i < rope_dim) {
        uint pair_base = hd_i & ~1u;
        uint k = pair_base >> 1;
        float freq = 1.0f / pow(theta, float(2*k)/float(rope_dim));
        float angle = float(positions[b]) * freq;
        float c = cos(angle), s = sin(angle);
        float x0 = kb[head*head_dim + pair_base];
        float x1 = kb[head*head_dim + pair_base + 1u];
        kv = (hd_i & 1u) ? (x0*s + x1*c) : (x0*c - x1*s);
    }

    size_t dst = (size_t)b*max_kv*row + (size_t)slots[b]*row + d;
    kcache[dst] = kv;
    vcache[dst] = vsrc[(size_t)b*row + d];
}

/* Decode attention for B sequences, one threadgroup per (sequence, query head).
 * q laid out [B][Nq*Hd]; per-sequence KV cache Kc/Vc laid out
 * [B][max_kv][Nk*Hd]; kvlens[b] is the number of valid (already-written) KV
 * slots for sequence b INCLUDING the just-scattered current token. out [B][Nq*Hd].
 * Softmax matches attn_decode_f32. */
kernel void attn_decode_f32_batched(
    device const float* q       [[buffer(0)]],   // [B][Nq*Hd]
    device const float* Kc      [[buffer(1)]],   // [B][max_kv][Nk*Hd]
    device const float* Vc      [[buffer(2)]],
    device float*       out     [[buffer(3)]],   // [B][Nq*Hd]
    constant uint&      Hd      [[buffer(4)]],
    constant uint&      Nq      [[buffer(5)]],
    constant uint&      Nk      [[buffer(6)]],
    device const int*   kvlens  [[buffer(7)]],   // [B]
    constant float&     scale   [[buffer(8)]],
    constant uint&      max_kv  [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg   [[threads_per_threadgroup]])
{
    uint b = tgid / Nq;
    uint h = tgid % Nq;
    uint kvlen = (uint)kvlens[b];
    uint kv_h = h * Nk / Nq;
    device const float* qh = q + (size_t)b*Nq*Hd + h*Hd;
    device const float* Kb = Kc + (size_t)b*max_kv*Nk*Hd;
    device const float* Vb = Vc + (size_t)b*max_kv*Nk*Hd;
    threadgroup float sc[512];
    threadgroup float red[64];
    for (uint t = tid; t < kvlen; t += tg) {
        device const float* kt = Kb + (size_t)t*Nk*Hd + kv_h*Hd;
        float dot = 0.0f;
        for (uint d = 0; d < Hd; d++) dot += qh[d]*kt[d];
        sc[t] = dot*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint t = tid; t < kvlen; t += tg) m = max(m, sc[t]);
    red[tid] = m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s>0; s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float maxv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lsum = 0.0f;
    for (uint t = tid; t < kvlen; t += tg) { float e = exp(sc[t]-maxv); sc[t]=e; lsum+=e; }
    red[tid]=lsum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s>0; s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv = 1.0f/red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* oh = out + (size_t)b*Nq*Hd + h*Hd;
    for (uint d = tid; d < Hd; d += tg) {
        float acc = 0.0f;
        for (uint t = 0; t < kvlen; t++) {
            device const float* vt = Vb + (size_t)t*Nk*Hd + kv_h*Hd;
            acc += sc[t]*inv*vt[d];
        }
        oh[d] = acc;
    }
}

/* Stage-1 tiled argmax over a logits row: each tg scans a contiguous
 * slice and emits one (val, idx) pair. top1_reduce_tiles merges. */
kernel void argmax_f32_tiles(
    device const float* logits [[buffer(0)]],
    device float*       vals   [[buffer(1)]],
    device uint*        idxs   [[buffer(2)]],
    constant uint&      V      [[buffer(3)]],
    uint t   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    const uint per = (V + 63u) / 64u;
    const uint beg = t * per;
    const uint end = min(V, beg + per);
    threadgroup float tv[256];
    threadgroup uint  ti[256];
    float best = -INFINITY;
    uint best_i = 0;
    for (uint i = beg + tid; i < end; i += tg) {
        float v = logits[i];
        if (v > best) { best = v; best_i = i; }
    }
    tv[tid] = best;
    ti[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = tg / 2; st > 0; st >>= 1) {
        if (tid < st && tv[tid + st] > tv[tid]) {
            tv[tid] = tv[tid + st];
            ti[tid] = ti[tid + st];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { vals[t] = tv[0]; idxs[t] = ti[0]; }
}

kernel void argmax_f32_batched(
    device const float* logits [[buffer(0)]],
    device uint*        out    [[buffer(1)]],
    constant uint&      V      [[buffer(2)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    device const float* row = logits + (size_t)b * V;
    threadgroup float vals[256];
    threadgroup uint  idxs[256];
    float best = row[0];
    uint best_i = 0;
    for (uint i = tid; i < V; i += tg) {
        float v = row[i];
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = vals[tid + s];
            uint oi = idxs[tid + s];
            if (ov > vals[tid]) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[b] = idxs[0];
}

kernel void top1_reduce_tiles(
    device const float* vals [[buffer(0)]],
    device const uint*  idxs [[buffer(1)]],
    device uint*        out  [[buffer(2)]],
    constant uint&      nt   [[buffer(3)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float tv[256];
    threadgroup uint ti[256];
    float best = -INFINITY;
    uint best_i = 0;
    for (uint i = tid; i < nt; i += tg) {
        float v = vals[(size_t)b * nt + i];
        uint ix = idxs[(size_t)b * nt + i];
        if (v > best) {
            best = v;
            best_i = ix;
        }
    }
    tv[tid] = best;
    ti[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2; s > 0; s >>= 1) {
        if (tid < s && tv[tid + s] > tv[tid]) {
            tv[tid] = tv[tid + s];
            ti[tid] = ti[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[b] = ti[0];
}

/* ===== Qwen3.5-specific kernels for GPU-resident full-attention layer =====
 *
 * These enable a full layer forward with zero CPU↔GPU round-trips within
 * a layer.  Used by stratum_metal_qwen35_forward_full_attn().
 */

/* Half-RoPE (rotary embedding): pairs are (x[i], x[i+half]) not (x[2i], x[2i+1]).
 * Qwen3.5 uses this layout. Applies to nheads * head_dim elements, rotating
 * only the first rope_dim dimensions. */
kernel void rope_half_f32(
    device float*       x         [[buffer(0)]],
    constant uint&      head_dim  [[buffer(1)]],
    constant uint&      rope_dim  [[buffer(2)]],
    constant int&       position  [[buffer(3)]],
    constant float&     theta     [[buffer(4)]],
    constant uint&      nheads    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = rope_dim / 2;
    uint per_head = head_dim;
    uint head = gid / half_dim;
    uint k    = gid % half_dim;
    if (head >= nheads) return;
    device float* xh = x + (size_t)head * per_head;
    float freq  = 1.0f / pow(theta, float(2*k)/float(rope_dim));
    float angle = float(position) * freq;
    float c = cos(angle), s = sin(angle);
    float x0 = xh[k];       /* first half */
    float x1 = xh[k + half_dim]; /* second half */
    xh[k]            = x0 * c - x1 * s;
    xh[k + half_dim] = x0 * s + x1 * c;
}

/* Per-head RMSNorm for Q or K. Qwen3.5 applies RMSNorm to each head
 * independently (not the full vector). x is [nheads * head_dim], gain is
 * [head_dim] (shared across heads). One threadgroup per head. */
kernel void rmsnorm_per_head_f32(
    device const float* x    [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      hd   [[buffer(3)]],   /* head_dim */
    constant float&     eps  [[buffer(4)]],
    constant uint&      nheads [[buffer(5)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    if (h >= nheads) return;
    device const float* xh = x + (size_t)h * hd;
    device float*       yh = y + (size_t)h * hd;
    threadgroup float sdata[256];
    float local = 0.0f;
    for (uint i = tid; i < hd; i += tg) { float v = xh[i]; local += v*v; }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(sdata[0]/float(hd) + eps);
    for (uint i = tid; i < hd; i += tg) yh[i] = xh[i]*scale*gain[i];
}

/* Sigmoid output gate: y[i] = x[i] * sigmoid(gate[i]).
 * Qwen3.5 applies this to attention output before o_proj. */
kernel void sigmoid_gate_inplace_f32(
    device float*       x    [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint&      n    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    float s = 1.0f / (1.0f + exp(-g));
    x[gid] *= s;
}

/* Split Q+gate concatenated buffer: qkv_proj outputs [2*Nq*Hd] where first
 * half is Q, second half is gate. This kernel deinterleaves into separate
 * q_buf[Nq*Hd] and gate_buf[Nq*Hd]. */
kernel void split_qgate_f32(
    device const float* qgate [[buffer(0)]],   /* [Nq * 2 * Hd] */
    device float*       q_out [[buffer(1)]],    /* [Nq * Hd] */
    device float*       g_out [[buffer(2)]],    /* [Nq * Hd] */
    constant uint&      NqHd  [[buffer(3)]],    /* Nq * Hd */
    uint gid [[thread_position_in_grid]])
{
    if (gid >= NqHd) return;
    q_out[gid] = qgate[gid];
    g_out[gid] = qgate[gid + NqHd];
}

/* Decode attention with output gate for Qwen3.5.
 * Same as attn_decode_f32 but applies sigmoid gate after weighted sum.
 * q is [Nq*Hd], gate is [Nq*Hd], Kc/Vc are [kvlen*Nk*Hd], out is [Nq*Hd]. */
kernel void attn_decode_gated_f32(
    device const float* q       [[buffer(0)]],
    device const float* gate    [[buffer(1)]],
    device const float* Kc      [[buffer(2)]],
    device const float* Vc      [[buffer(3)]],
    device float*       out     [[buffer(4)]],
    constant uint&      Hd      [[buffer(5)]],
    constant uint&      Nq      [[buffer(6)]],
    constant uint&      Nk      [[buffer(7)]],
    constant uint&      kvlen   [[buffer(8)]],
    constant float&     scale   [[buffer(9)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    uint kv_h = h * Nk / Nq;
    device const float* qh = q + h*Hd;
    threadgroup float sc[512];
    threadgroup float red[64];
    for (uint t = tid; t < kvlen; t += tg) {
        device const float* kt = Kc + (size_t)t*Nk*Hd + kv_h*Hd;
        float dot = 0.0f;
        for (uint d = 0; d < Hd; d++) dot += qh[d]*kt[d];
        sc[t] = dot*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint t = tid; t < kvlen; t += tg) m = max(m, sc[t]);
    red[tid] = m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s>0; s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float maxv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lsum = 0.0f;
    for (uint t = tid; t < kvlen; t += tg) { float e = exp(sc[t]-maxv); sc[t]=e; lsum+=e; }
    red[tid]=lsum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s>0; s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv = 1.0f/red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* oh = out + h*Hd;
    for (uint d = tid; d < Hd; d += tg) {
        float acc = 0.0f;
        for (uint t = 0; t < kvlen; t++) {
            device const float* vt = Vc + (size_t)t*Nk*Hd + kv_h*Hd;
            acc += sc[t]*inv*vt[d];
        }
        /* apply sigmoid gate */
        float g = gate[h*Hd + d];
        float sig = 1.0f / (1.0f + exp(-g));
        oh[d] = acc * sig;
    }
}


/* H3 fused MLP stage 1: fc1 gate/up rows of ONE fused Q4_K weight
 * (rows [0,N) = gate, [N,2N) = up), swiglu, output written with a caller
 * row stride (h rows live inside the wider fbuf rows). Grid: (N, ceil(B/2));
 * one threadgroup handles two adjacent tokens — the dual_swiglu_g2 scheme. */
kernel void q4k_h3_mlp1_swiglu(
    device const block_q4_K* Wg         [[buffer(0)]],   // gate rows
    device const block_q4_K* Wu         [[buffer(1)]],   // up rows
    device const float*      x          [[buffer(2)]],   // [B, K] compact
    device float*            h          [[buffer(3)]],   // rows of hstride
    constant uint&           K          [[buffer(4)]],   // in_dim (HID)
    constant uint&           N          [[buffer(5)]],   // FF2
    constant uint&           B          [[buffer(6)]],
    constant uint&           hstride    [[buffer(7)]],
    uint2                    tgid       [[threadgroup_position_in_grid]],
    uint2                    tid2       [[thread_position_in_threadgroup]],
    uint2                    tg2        [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint b0 = tgid.y * 2u;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || b0 >= B) return;
    uint b1 = b0 + 1u;

    const uint blocks_per_row = K / 256;
    device const block_q4_K* rg = Wg + row * blocks_per_row;
    device const block_q4_K* ru = Wu + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0 * K;
    device const float* xb1 = x + (size_t)b1 * K;
    const uint total_sb = blocks_per_row * 8;

    float4 partial = float4(0.0f);
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& bg = rg[i];
        const device block_q4_K& bu = ru[i];
        uchar scg, mg, scu, mu;
        unpack_scale_min(j, bg.scales, scg, mg);
        unpack_scale_min(j, bu.scales, scu, mu);
        float g_sc = float(bg.d) * float(scg);
        float g_min = float(bg.dmin) * float(mg);
        float u_sc = float(bu.d) * float(scu);
        float u_min = float(bu.dmin) * float(mu);
        const device uchar* qg = bg.qs + (j / 2) * 32;
        const device uchar* qu = bu.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint eo = i * 256 + j * 32;
        uchar4 mask = uchar4(0xF);
        float4 qgg0 = 0, qgu0 = 0, qgg1 = 0, qgu1 = 0;
        float4 xs0 = 0, xs1 = 0;
        for (int l = 0; l < 32; l += 4) {
            uchar4 rawg = *(device const uchar4*)(qg + l);
            uchar4 rawu = *(device const uchar4*)(qu + l);
            float4 ng = float4((rawg >> uchar4((uchar)shift)) & mask);
            float4 nu = float4((rawu >> uchar4((uchar)shift)) & mask);
            float4 xv0 = *(device const float4*)(xb0 + eo + l);
            float4 xv1 = *(device const float4*)(xb1 + eo + l);
            qgg0 += ng * xv0; qgu0 += nu * xv0;
            qgg1 += ng * xv1; qgu1 += nu * xv1;
            xs0 += xv0; xs1 += xv1;
        }
        float sx0 = xs0.x + xs0.y + xs0.z + xs0.w;
        float sx1 = xs1.x + xs1.y + xs1.z + xs1.w;
        partial.x += g_sc * (qgg0.x+qgg0.y+qgg0.z+qgg0.w) - g_min * sx0;
        partial.y += u_sc * (qgu0.x+qgu0.y+qgu0.z+qgu0.w) - u_min * sx0;
        partial.z += g_sc * (qgg1.x+qgg1.y+qgg1.z+qgg1.w) - g_min * sx1;
        partial.w += u_sc * (qgu1.x+qgu1.y+qgu1.z+qgu1.w) - u_min * sx1;
    }
    threadgroup float sd0[256], sd1[256], sd2[256], sd3[256];
    float sg0 = simd_sum(partial.x), sg1 = simd_sum(partial.y);
    float sg2 = simd_sum(partial.z), sg3 = simd_sum(partial.w);
    uint simd_id = tid / 32u, lane = tid % 32u;
    if (lane == 0) { sd0[simd_id]=sg0; sd1[simd_id]=sg1; sd2[simd_id]=sg2; sd3[simd_id]=sg3; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint ns = (tg_size + 31u) / 32u;
        float g0=0,u0=0,g1=0,u1=0;
        for (uint s = 0; s < ns; s++) { g0+=sd0[s]; u0+=sd1[s]; g1+=sd2[s]; u1+=sd3[s]; }
        h[(size_t)b0 * hstride + row] = (g0 / (1.0f + exp(-g0))) * u0;
        if (b1 < B) h[(size_t)b1 * hstride + row] = (g1 / (1.0f + exp(-g1))) * u1;
    }
}

/* H3 fused MLP stage 2: fc2 Q4_K gemv reading h rows at stride hstride
 * (in fbuf), writing proj compact [B, N]. Grid: (N, B). */
kernel void q4k_h3_mlp2_ostride(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      h           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],   // FF2
    constant uint&           N           [[buffer(4)]],   // HID
    constant uint&           B           [[buffer(5)]],
    constant uint&           hstride     [[buffer(6)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint bidx = tgid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bidx >= B) return;
    const uint blocks_per_row = K / 256;
    device const block_q4_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = h + (size_t)bidx * hstride;
    const uint total_sb = blocks_per_row * 8;
    float partial = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 3;
        uint j = sb & 7;
        const device block_q4_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const float dmin = float(bl.dmin);
        uchar sc, m;
        unpack_scale_min(j, bl.scales, sc, m);
        float d_sc = d * float(sc);
        float dmin_m = dmin * float(m);
        const device uchar* qs_pair = bl.qs + (j / 2) * 32;
        uint shift = (j & 1) ? 4u : 0u;
        uint elem_offset = i * 256 + j * 32;
        float4 qx4 = float4(0.0f), xs4 = float4(0.0f);
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 32; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            uchar4 nib = (raw >> uchar4((uchar)shift)) & mask;
            float4 nf = float4(nib);
            float4 xv = *(device const float4*)(xb + elem_offset + l);
            qx4 += nf * xv;
            xs4 += xv;
        }
        partial += d_sc * (qx4.x+qx4.y+qx4.z+qx4.w) - dmin_m * (xs4.x+xs4.y+xs4.z+xs4.w);
    }
    threadgroup float sdata[256];
    float sg = simd_sum(partial);
    uint simd_id = tid / 32u, lane = tid % 32u;
    if (lane == 0) sdata[simd_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint ns = (tg_size + 31u) / 32u;
        float tot = 0.0f;
        for (uint s = 0; s < ns; s++) tot += sdata[s];
        y[(size_t)bidx * N + row] = tot;
    }
}

/* Tiled Q4_K batch GEMM: C[M,N] = A[M,K] x W^T[N,K], one threadgroup per
 * 64x64 output tile. Beats the per-row bparallel (~0.34 vs ~1.9 TFLOP/s at
 * seq=276) because each threadgroup reuses its A/B slabs across an 8x8
 * register tile instead of streaming one row per threadgroup. Weights are
 * dequantized into the B slab in-place (nib*d_sc - dmin_m); this reorders the
 * fp32 math vs the per-row d_sc*sum - dmin_m*sum form, so it matches to ~2e-3
 * (fp32 associativity), not bit-exact. Requires K a multiple of 256 and a
 * 64-aligned N (both true for every H3 GEMV). */
kernel void q4k_tile_gemm(
    device const block_q4_K* W [[buffer(0)]],
    device const float*      A [[buffer(1)]],
    device float*            C [[buffer(2)]],
    constant uint&           M [[buffer(3)]],
    constant uint&           N [[buffer(4)]],
    constant uint&           K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgsz [[threads_per_threadgroup]])
{
    const uint BM = 64, BN = 128, BK = 32;
    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];
    const uint bpr = K / 256;
    const uint m0 = tgid.y * BM, n0 = tgid.x * BN;
    const uint tx = tid.x, ty = tid.y;

    float acc[8][8];
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    for (uint kk = 0; kk < K; kk += BK) {
        for (uint i = tx + ty * 16; i < BM * BK / 4; i += 128) {
            uint r = i / (BK / 4), c4 = i % (BK / 4);
            uint mr = m0 + r;
            float4 v = (mr < M && kk + c4 * 4 < K)
                ? ((device const float4*)(A + mr * K + kk))[c4] : float4(0.0f);
            ((threadgroup float4*)As[r])[c4] = v;
        }
        uint j0 = (kk % 256) / 32;
        for (uint i = tx + ty * 16; i < BN; i += 128) {
            uint c = i, nc = n0 + c;
            if (nc < N) {
                const device block_q4_K& bl = W[nc * bpr + kk / 256];
                float d = float(bl.d), dmin = float(bl.dmin);
                uchar sc, m;
                unpack_scale_min((int)j0, bl.scales, sc, m);
                float d_sc = d * (float)sc, dmin_m = dmin * (float)m;
                const device uchar* qs = bl.qs + (j0 / 2) * 32;
                uint shift = (j0 & 1) ? 4u : 0u;
                for (uint l = 0; l < 32; l++)
                    Bs[l][c] = (float)((qs[l] >> shift) & 0xF) * d_sc - dmin_m;
            } else {
                for (uint l = 0; l < 32; l++) Bs[l][c] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k2 = 0; k2 < BK; k2++)
            for (int i = 0; i < 8; i++) {
                float a = As[ty * 8 + i][k2];
                for (int j = 0; j < 8; j++) acc[i][j] += a * Bs[k2][tx * 8 + j];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
        uint mr = m0 + ty * 8 + i, nc = n0 + tx * 8 + j;
        if (mr < M && nc < N) C[mr * N + nc] = acc[i][j];
    }
}


/* ===== Whole-layer fusion kernels (single-token decode) =====
 * coal16 mapping (16 rows/tg, 16 thr/row) with a small-op stage folded
 * into the GEMV prologue/epilogue, so a transformer layer runs in ~6
 * dispatches instead of ~15. The norm scale is recomputed redundantly
 * per threadgroup (a few hundred FLOPs on an L2-hot x) — cheaper than a
 * dispatch boundary + pipeline drain. */

struct block_q6_K_l {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    half  d;
};

/* Fused QKV projection with rmsnorm prologue: y = W * rmsnorm(x)*gain.
 * One dispatch covers q (Q4_K rows [0,nq)), k (Q4_K rows [nq,nq+nk)) and
 * v (Q6_K rows [nq+nk,ntot)) — each 16-thread row-group picks its tensor
 * independently, so no range alignment is required. */
kernel void qkv_coal16_norm(
    device const block_q4_K*   Wq   [[buffer(0)]],
    device const block_q4_K*   Wk   [[buffer(1)]],
    device const block_q6_K_l* Wv   [[buffer(2)]],
    device const float*        x    [[buffer(3)]],
    device const float*        gain [[buffer(4)]],
    device float*              yq   [[buffer(5)]],
    device float*              yk   [[buffer(6)]],
    device float*              yv   [[buffer(7)]],
    constant uint&             K    [[buffer(8)]],
    constant float&            eps  [[buffer(9)]],
    constant uint&             nq   [[buffer(10)]],
    constant uint&             nk   [[buffer(11)]],
    constant uint&             ntot [[buffer(12)]],
    constant uint&             vq6  [[buffer(13)]],
    device float*              kraw [[buffer(14)]],  /* raw current-k shadow for attn prologue */
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_size != 256) return;
    threadgroup float tg_red[8];
    threadgroup float tg_scale;
    {
        float ss = 0.0f;
        for (uint i = tid; i < K; i += 256u) { float v = x[i]; ss += v*v; }
        ss += simd_shuffle_xor(ss, 16);
        ss += simd_shuffle_xor(ss, 8);
        ss += simd_shuffle_xor(ss, 4);
        ss += simd_shuffle_xor(ss, 2);
        ss += simd_shuffle_xor(ss, 1);
        if ((tid & 31u) == 0u) tg_red[tid >> 5] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float t = 0.0f;
            for (int i = 0; i < 8; i++) t += tg_red[i];
            tg_scale = 1.0f / sqrt(t / float(K) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float nrm = tg_scale;

    const uint local_row = tid >> 4;
    const uint lane      = tid & 15;
    const uint grow      = tgid * 16u + local_row;
    if (grow >= ntot) return;
    const uint blocks_per_row = K / 256;

    if (grow < nq + nk) {
        device const block_q4_K* W; device float* y; uint r;
        if (grow < nq) { W = Wq; y = yq; r = grow; }
        else           { W = Wk; y = yk; r = grow - nq; }
        device const block_q4_K* row_blocks = W + (uint)r * blocks_per_row;
        const uint sub_block = lane >> 1;
        const uint elem      = lane & 1;
        const uint shift     = (sub_block & 1) ? 4u : 0u;
        float partial = 0.0f;
        for (uint blk = 0; blk < blocks_per_row; blk++) {
            const device block_q4_K& b = row_blocks[blk];
            const float d = float(b.d), dmin = float(b.dmin);
            uchar sc, m;
            unpack_scale_min(sub_block, b.scales, sc, m);
            const float d_sc = d * float(sc), dmin_m = dmin * float(m);
            const device uchar* qp = b.qs + (sub_block / 2) * 32 + elem * 8;
            const uint xoff = blk * 256 + sub_block * 32 + elem * 8;
            float qx = 0.0f, xs = 0.0f;
            #pragma unroll
            for (int l2 = 0; l2 < 2; l2++) {
                uint2  w2  = *(device const uint2*)(qp + 16u*l2);
                uchar4 na  = (as_type<uchar4>(w2.x) >> uchar4((uchar)shift)) & uchar4(0xF);
                uchar4 nb  = (as_type<uchar4>(w2.y) >> uchar4((uchar)shift)) & uchar4(0xF);
                float4 xa  = *(device const float4*)(x    + xoff + 16u*l2);
                float4 ga  = *(device const float4*)(gain + xoff + 16u*l2);
                xa = (xa * nrm) * ga;
                float4 xb  = *(device const float4*)(x    + xoff + 16u*l2 + 4);
                float4 gb  = *(device const float4*)(gain + xoff + 16u*l2 + 4);
                xb = (xb * nrm) * gb;
                qx += dot(float4(na), xa) + dot(float4(nb), xb);
                xs += xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
            }
            partial += d_sc * qx - dmin_m * xs;
        }
        float tot = partial;
        tot += simd_shuffle_xor(tot, 8);
        tot += simd_shuffle_xor(tot, 4);
        tot += simd_shuffle_xor(tot, 2);
        tot += simd_shuffle_xor(tot, 1);
        if (lane == 0) { y[r] = tot; if (grow >= nq) kraw[r] = tot; }
    } else if (!vq6) {
        /* v stored as Q4_K on some layers */
        const uint r = grow - nq - nk;
        device const block_q4_K* row_blocks =
            (device const block_q4_K*)Wv + (uint)r * blocks_per_row;
        const uint sub_block = lane >> 1;
        const uint elem      = lane & 1;
        const uint shift     = (sub_block & 1) ? 4u : 0u;
        float partial = 0.0f;
        for (uint blk = 0; blk < blocks_per_row; blk++) {
            const device block_q4_K& b = row_blocks[blk];
            const float d = float(b.d), dmin = float(b.dmin);
            uchar sc, m;
            unpack_scale_min(sub_block, b.scales, sc, m);
            const float d_sc = d * float(sc), dmin_m = dmin * float(m);
            const device uchar* qp = b.qs + (sub_block / 2) * 32 + elem * 8;
            const uint xoff = blk * 256 + sub_block * 32 + elem * 8;
            float qx = 0.0f, xs = 0.0f;
            #pragma unroll
            for (int l2 = 0; l2 < 2; l2++) {
                uint2  w2  = *(device const uint2*)(qp + 16u*l2);
                uchar4 na  = (as_type<uchar4>(w2.x) >> uchar4((uchar)shift)) & uchar4(0xF);
                uchar4 nb  = (as_type<uchar4>(w2.y) >> uchar4((uchar)shift)) & uchar4(0xF);
                float4 xa  = *(device const float4*)(x    + xoff + 16u*l2);
                float4 ga  = *(device const float4*)(gain + xoff + 16u*l2);
                xa = (xa * nrm) * ga;
                float4 xb  = *(device const float4*)(x    + xoff + 16u*l2 + 4);
                float4 gb  = *(device const float4*)(gain + xoff + 16u*l2 + 4);
                xb = (xb * nrm) * gb;
                qx += dot(float4(na), xa) + dot(float4(nb), xb);
                xs += xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
            }
            partial += d_sc * qx - dmin_m * xs;
        }
        float tot = partial;
        tot += simd_shuffle_xor(tot, 8);
        tot += simd_shuffle_xor(tot, 4);
        tot += simd_shuffle_xor(tot, 2);
        tot += simd_shuffle_xor(tot, 1);
        if (lane == 0) yv[r] = tot;
    } else {
        const uint r = grow - nq - nk;
        device const block_q6_K_l* row_blocks = Wv + (uint)r * blocks_per_row;
        const uint half_idx = lane >> 3;
        const uint t8 = lane & 7;
        const uint l0 = t8 * 4;
        const uint is = t8 >> 2;
        float partial = 0.0f;
        for (uint blk = 0; blk < blocks_per_row; blk++) {
            const device block_q6_K_l& b = row_blocks[blk];
            const float d = float(b.d);
            const uint n = half_idx * 128;
            const device uchar* ql = b.ql + n / 2;
            const device uchar* qh = b.qh + n / 4;
            const device char*  s  = b.scales + n / 16;
            const uint base = blk * 256 + n;
            /* vectorized: ql/qh bytes gathered via uchar4, x/gain via float4 */
            float a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f;
            const uchar4 ql_lo4 = *(device const uchar4*)(ql + l0);
            const uchar4 ql_hi4 = *(device const uchar4*)(ql + l0 + 32);
            const uchar4 qh4    = *(device const uchar4*)(qh + l0);
            const int4 q1v = int4(ql_lo4 & uchar4(0xF)) | (int4((qh4 >> uchar4(0)) & uchar4(3)) << 4);
            const int4 q2v = int4(ql_hi4 & uchar4(0xF)) | (int4((qh4 >> uchar4(2)) & uchar4(3)) << 4);
            const int4 q3v = int4(ql_lo4 >> uchar4(4))    | (int4((qh4 >> uchar4(4)) & uchar4(3)) << 4);
            const int4 q4v = int4(ql_hi4 >> uchar4(4))    | (int4((qh4 >> uchar4(6)) & uchar4(3)) << 4);
            const float4 xv1 = (*(device const float4*)(x + base + l0)) * nrm;
            const float4 gv1 = *(device const float4*)(gain + base + l0);
            const float4 xv2 = (*(device const float4*)(x + base + l0 + 32)) * nrm;
            const float4 gv2 = *(device const float4*)(gain + base + l0 + 32);
            const float4 xv3 = (*(device const float4*)(x + base + l0 + 64)) * nrm;
            const float4 gv3 = *(device const float4*)(gain + base + l0 + 64);
            const float4 xv4 = (*(device const float4*)(x + base + l0 + 96)) * nrm;
            const float4 gv4 = *(device const float4*)(gain + base + l0 + 96);
            a1 = dot(float4(q1v - 32), xv1 * gv1);
            a2 = dot(float4(q2v - 32), xv2 * gv2);
            a3 = dot(float4(q3v - 32), xv3 * gv3);
            a4 = dot(float4(q4v - 32), xv4 * gv4);
            partial += d * float(s[is + 0]) * a1;
            partial += d * float(s[is + 2]) * a2;
            partial += d * float(s[is + 4]) * a3;
            partial += d * float(s[is + 6]) * a4;
        }
        float tot = partial;
        tot += simd_shuffle_xor(tot, 8);
        tot += simd_shuffle_xor(tot, 4);
        tot += simd_shuffle_xor(tot, 2);
        tot += simd_shuffle_xor(tot, 1);
        if (lane == 0) yv[r] = tot;
    }
}

/* Fused gate+up projection with rmsnorm prologue (both Q4_K):
 * rows [0,nf) -> yg = Wg*xn, rows [nf,2*nf) -> yu = Wu*xn. */
kernel void gateup_coal16_norm(
    device const block_q4_K*   Wg   [[buffer(0)]],
    device const block_q4_K*   Wu   [[buffer(1)]],
    device const float*        x    [[buffer(2)]],
    device const float*        gain [[buffer(3)]],
    device float*              yg   [[buffer(4)]],
    device float*              yu   [[buffer(5)]],
    constant uint&             K    [[buffer(6)]],
    constant float&            eps  [[buffer(7)]],
    constant uint&             nf   [[buffer(8)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_size != 256) return;
    threadgroup float tg_red[8];
    threadgroup float tg_scale;
    {
        float ss = 0.0f;
        for (uint i = tid; i < K; i += 256u) { float v = x[i]; ss += v*v; }
        ss += simd_shuffle_xor(ss, 16);
        ss += simd_shuffle_xor(ss, 8);
        ss += simd_shuffle_xor(ss, 4);
        ss += simd_shuffle_xor(ss, 2);
        ss += simd_shuffle_xor(ss, 1);
        if ((tid & 31u) == 0u) tg_red[tid >> 5] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float t = 0.0f;
            for (int i = 0; i < 8; i++) t += tg_red[i];
            tg_scale = 1.0f / sqrt(t / float(K) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float nrm = tg_scale;

    /* row-pair variant: each lane computes TWO adjacent rows sharing one
     * x/gain slice — halves x-load issue. 32 rows per threadgroup. */
    const uint local_row = tid >> 4;
    const uint lane      = tid & 15;
    const uint r0        = tgid * 32u + local_row * 2u;
    if (r0 >= 2u * nf) return;
    const uint r1        = r0 + 1u;
    const bool has1      = r1 < 2u * nf;
    device const block_q4_K* W0; device float* y0; uint rr0;
    if (r0 < nf) { W0 = Wg; y0 = yg; rr0 = r0; }
    else         { W0 = Wu; y0 = yu; rr0 = r0 - nf; }
    device const block_q4_K* W1; device float* y1; uint rr1;
    if (r1 < nf) { W1 = Wg; y1 = yg; rr1 = r1; }
    else         { W1 = Wu; y1 = yu; rr1 = r1 - nf; }
    const uint blocks_per_row = K / 256;
    device const block_q4_K* rb0 = W0 + (uint)rr0 * blocks_per_row;
    device const block_q4_K* rb1 = W1 + (uint)(has1 ? rr1 : rr0) * blocks_per_row;
    const uint sub_block = lane >> 1;
    const uint elem      = lane & 1;
    const uint shift     = (sub_block & 1) ? 4u : 0u;
    float partial0 = 0.0f, partial1 = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; blk++) {
        const device block_q4_K& b0 = rb0[blk];
        const device block_q4_K& b1 = rb1[blk];
        const float d0 = float(b0.d), dmin0 = float(b0.dmin);
        const float d1 = float(b1.d), dmin1 = float(b1.dmin);
        uchar sc0, m0, sc1, m1;
        unpack_scale_min(sub_block, b0.scales, sc0, m0);
        unpack_scale_min(sub_block, b1.scales, sc1, m1);
        const float d_sc0 = d0 * float(sc0), dmin_m0 = dmin0 * float(m0);
        const float d_sc1 = d1 * float(sc1), dmin_m1 = dmin1 * float(m1);
        const device uchar* qp0 = b0.qs + (sub_block / 2) * 32 + elem * 8;
        const device uchar* qp1 = b1.qs + (sub_block / 2) * 32 + elem * 8;
        const uint xoff = blk * 256 + sub_block * 32 + elem * 8;
        float qx0 = 0.0f, qx1 = 0.0f, xs = 0.0f;
        #pragma unroll
        for (int l2 = 0; l2 < 2; l2++) {
            uint2  w20 = *(device const uint2*)(qp0 + 16u*l2);
            uint2  w21 = *(device const uint2*)(qp1 + 16u*l2);
            uchar4 na0 = (as_type<uchar4>(w20.x) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 nb0 = (as_type<uchar4>(w20.y) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 na1 = (as_type<uchar4>(w21.x) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 nb1 = (as_type<uchar4>(w21.y) >> uchar4((uchar)shift)) & uchar4(0xF);
            float4 xa  = *(device const float4*)(x    + xoff + 16u*l2);
            float4 ga  = *(device const float4*)(gain + xoff + 16u*l2);
            xa = (xa * nrm) * ga;
            float4 xb  = *(device const float4*)(x    + xoff + 16u*l2 + 4);
            float4 gb  = *(device const float4*)(gain + xoff + 16u*l2 + 4);
            xb = (xb * nrm) * gb;
            qx0 += dot(float4(na0), xa) + dot(float4(nb0), xb);
            qx1 += dot(float4(na1), xa) + dot(float4(nb1), xb);
            xs += xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
        }
        partial0 += d_sc0 * qx0 - dmin_m0 * xs;
        partial1 += d_sc1 * qx1 - dmin_m1 * xs;
    }
    float tot0 = partial0, tot1 = partial1;
    tot0 += simd_shuffle_xor(tot0, 8); tot1 += simd_shuffle_xor(tot1, 8);
    tot0 += simd_shuffle_xor(tot0, 4); tot1 += simd_shuffle_xor(tot1, 4);
    tot0 += simd_shuffle_xor(tot0, 2); tot1 += simd_shuffle_xor(tot1, 2);
    tot0 += simd_shuffle_xor(tot0, 1); tot1 += simd_shuffle_xor(tot1, 1);
    if (lane == 0) { y0[rr0] = tot0; if (has1) y1[rr1] = tot1; }
}

/* coal16 GEMV with residual-accumulate epilogue: y[row] += dot(W_row, x).
 * The owning thread reads y[row] after its dot — safe in place. */
kernel void q4k_sgemv_row_coal16_accum(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N_total     [[buffer(4)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_size != 256) return;
    const uint blocks_per_row = K / 256;
    /* row-pair: lane covers rows r0,r0+1 sharing one x slice (32 rows/tg) */
    const uint local_row = tid >> 4;
    const uint lane      = tid & 15;
    const uint r0 = tgid * 32u + local_row * 2u;
    if (r0 >= N_total) return;
    const uint r1 = min(r0 + 1u, N_total - 1u);
    device const block_q4_K* rb0 = W + (uint)r0 * blocks_per_row;
    device const block_q4_K* rb1 = W + (uint)r1 * blocks_per_row;

    const uint sub_block = lane >> 1;
    const uint elem      = lane & 1;
    const uint shift     = (sub_block & 1) ? 4u : 0u;

    float partial0 = 0.0f, partial1 = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; blk++) {
        const device block_q4_K& b0 = rb0[blk];
        const device block_q4_K& b1 = rb1[blk];
        uchar sc0, m0, sc1, m1;
        unpack_scale_min(sub_block, b0.scales, sc0, m0);
        unpack_scale_min(sub_block, b1.scales, sc1, m1);
        const float d_sc0 = float(b0.d) * float(sc0), dmin_m0 = float(b0.dmin) * float(m0);
        const float d_sc1 = float(b1.d) * float(sc1), dmin_m1 = float(b1.dmin) * float(m1);
        const device uchar* qp0 = b0.qs + (sub_block / 2) * 32 + elem * 8;
        const device uchar* qp1 = b1.qs + (sub_block / 2) * 32 + elem * 8;
        const uint xoff = blk * 256 + sub_block * 32 + elem * 8;
        float qx0 = 0.0f, qx1 = 0.0f, xs = 0.0f;
        #pragma unroll
        for (int l2 = 0; l2 < 2; l2++) {
            uint2 w20 = *(device const uint2*)(qp0 + 16u*l2);
            uint2 w21 = *(device const uint2*)(qp1 + 16u*l2);
            uchar4 na0 = (as_type<uchar4>(w20.x) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 nb0 = (as_type<uchar4>(w20.y) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 na1 = (as_type<uchar4>(w21.x) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 nb1 = (as_type<uchar4>(w21.y) >> uchar4((uchar)shift)) & uchar4(0xF);
            float4 xa = *(device const float4*)(x + xoff + 16u*l2);
            float4 xb = *(device const float4*)(x + xoff + 16u*l2 + 4);
            qx0 += dot(float4(na0), xa) + dot(float4(nb0), xb);
            qx1 += dot(float4(na1), xa) + dot(float4(nb1), xb);
            xs += xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
        }
        partial0 += d_sc0 * qx0 - dmin_m0 * xs;
        partial1 += d_sc1 * qx1 - dmin_m1 * xs;
    }

    float tot0 = partial0, tot1 = partial1;
    tot0 += simd_shuffle_xor(tot0, 8); tot1 += simd_shuffle_xor(tot1, 8);
    tot0 += simd_shuffle_xor(tot0, 4); tot1 += simd_shuffle_xor(tot1, 4);
    tot0 += simd_shuffle_xor(tot0, 2); tot1 += simd_shuffle_xor(tot1, 2);
    tot0 += simd_shuffle_xor(tot0, 1); tot1 += simd_shuffle_xor(tot1, 1);
    if (lane == 0) {
        y[r0] += tot0;
        if (r1 != r0) y[r1] += tot1;
    }
}

/* Per-head qk-norm + rope for q and k in ONE dispatch:
 * threadgroup h<nq handles q head h, else k head h-nq. */
kernel void qknorm_rope_dual_f32(
    device float*       xq       [[buffer(0)]],
    device float*       xk       [[buffer(1)]],
    device const float* gq       [[buffer(2)]],
    device const float* gk       [[buffer(3)]],
    constant uint&      hd       [[buffer(4)]],
    constant float&     eps      [[buffer(5)]],
    constant uint&      nq       [[buffer(6)]],
    constant uint&      nk       [[buffer(7)]],
    constant uint&      rope_dim [[buffer(8)]],
    constant int&       position [[buffer(9)]],
    constant float&     theta    [[buffer(10)]],
    constant uint&      neox     [[buffer(11)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    device float* xh; device const float* gain;
    if (h < nq) { xh = xq + (size_t)h * hd; gain = gq; }
    else {
        uint hk = h - nq;
        if (hk >= nk) return;
        xh = xk + (size_t)hk * hd; gain = gk;
    }
    threadgroup float sdata[256];
    float local = 0.0f;
    for (uint i = tid; i < hd; i += tg) { float v = xh[i]; local += v*v; }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(sdata[0]/float(hd) + eps);
    uint pairs = rope_dim/2;
    for (uint kk = tid; kk < pairs; kk += tg) {
        uint i0 = neox ? kk : 2*kk;
        uint i1 = neox ? kk + pairs : 2*kk + 1;
        float v0 = xh[i0]*scale*gain[i0];
        float v1 = xh[i1]*scale*gain[i1];
        float freq  = 1.0f / pow(theta, float(2*kk)/float(rope_dim));
        float angle = float(position) * freq;
        float c = cos(angle), s = sin(angle);
        xh[i0] = v0*c - v1*s;
        xh[i1] = v0*s + v1*c;
    }
    for (uint i = rope_dim + tid; i < hd; i += tg)
        xh[i] = xh[i]*scale*gain[i];
}


/* Q4_K coal16 GEMV with fused SwiGLU prologue and residual epilogue:
 * fa[i] = silu(g[i]) * u[i] staged once per threadgroup (all 16 rows
 * share it);  y[row] += dot(W_row, fa).  Companion to the Q6_K variant
 * for layers whose down_proj is Q4_K. */
kernel void q4k_sgemv_row_coal16_swires(
    device const block_q4_K* W           [[buffer(0)]],
    device const float*      g           [[buffer(1)]],
    device const float*      u           [[buffer(2)]],
    device float*            y           [[buffer(3)]],
    constant uint&           K           [[buffer(4)]],
    constant uint&           N_total     [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_size != 256) return;
    if (K > 4096u) return;
    threadgroup float fa_s[4096];
    for (uint i = tid; i < K; i += 256u) {
        float gi = g[i];
        fa_s[i] = gi / (1.0f + exp(-gi)) * u[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint blocks_per_row = K / 256;
    const uint local_row = tid >> 4;
    const uint lane      = tid & 15;
    const uint r0 = tgid * 32u + local_row * 2u;
    if (r0 >= N_total) return;
    const uint r1 = min(r0 + 1u, N_total - 1u);
    device const block_q4_K* rb0 = W + (uint)r0 * blocks_per_row;
    device const block_q4_K* rb1 = W + (uint)r1 * blocks_per_row;

    const uint sub_block = lane >> 1;
    const uint elem      = lane & 1;
    const uint shift     = (sub_block & 1) ? 4u : 0u;

    float partial0 = 0.0f, partial1 = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; blk++) {
        const device block_q4_K& b0 = rb0[blk];
        const device block_q4_K& b1 = rb1[blk];
        uchar sc0, m0, sc1, m1;
        unpack_scale_min(sub_block, b0.scales, sc0, m0);
        unpack_scale_min(sub_block, b1.scales, sc1, m1);
        const float d_sc0 = float(b0.d) * float(sc0), dmin_m0 = float(b0.dmin) * float(m0);
        const float d_sc1 = float(b1.d) * float(sc1), dmin_m1 = float(b1.dmin) * float(m1);
        const device uchar* qp0 = b0.qs + (sub_block / 2) * 32 + elem * 8;
        const device uchar* qp1 = b1.qs + (sub_block / 2) * 32 + elem * 8;
        const uint xoff = blk * 256 + sub_block * 32 + elem * 8;
        float qx0 = 0.0f, qx1 = 0.0f, xs = 0.0f;
        #pragma unroll
        for (int l2 = 0; l2 < 2; l2++) {
            uint2 w20 = *(device const uint2*)(qp0 + 16u*l2);
            uint2 w21 = *(device const uint2*)(qp1 + 16u*l2);
            uchar4 na0 = (as_type<uchar4>(w20.x) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 nb0 = (as_type<uchar4>(w20.y) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 na1 = (as_type<uchar4>(w21.x) >> uchar4((uchar)shift)) & uchar4(0xF);
            uchar4 nb1 = (as_type<uchar4>(w21.y) >> uchar4((uchar)shift)) & uchar4(0xF);
            float4 xa = *(threadgroup const float4*)(fa_s + xoff + 16u*l2);
            float4 xb = *(threadgroup const float4*)(fa_s + xoff + 16u*l2 + 4);
            qx0 += dot(float4(na0), xa) + dot(float4(nb0), xb);
            qx1 += dot(float4(na1), xa) + dot(float4(nb1), xb);
            xs += xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
        }
        partial0 += d_sc0 * qx0 - dmin_m0 * xs;
        partial1 += d_sc1 * qx1 - dmin_m1 * xs;
    }

    float tot0 = partial0, tot1 = partial1;
    tot0 += simd_shuffle_xor(tot0, 8); tot1 += simd_shuffle_xor(tot1, 8);
    tot0 += simd_shuffle_xor(tot0, 4); tot1 += simd_shuffle_xor(tot1, 4);
    tot0 += simd_shuffle_xor(tot0, 2); tot1 += simd_shuffle_xor(tot1, 2);
    tot0 += simd_shuffle_xor(tot0, 1); tot1 += simd_shuffle_xor(tot1, 1);
    if (lane == 0) {
        y[r0] += tot0;
        if (r1 != r0) y[r1] += tot1;
    }
}

/* Shared flash-decode scan: simd sid sweeps positions [tbeg,tend) with
 * stride nsimd, batching 4 positions per iteration so the four xor-reduce
 * chains pipeline instead of serializing. Lane l32 owns dims [d4,d4+4)
 * (inactive when d4 >= Hd). Maintains running (m,l,acc4); caller merges
 * across simds. kh_s holds the roped+normed current-token K (position
 * tcur) — its K row is read from shared, its V row from device. */
static inline void flash_scan4(
    threadgroup const float* qh_s,
    threadgroup const float* kh_s,
    device const float* Kc,
    device const float* Vc,
    const uint   kv_h, const uint Hd, const uint Nk,
    const uint   tbeg, const uint tend, const uint tcur,
    const uint   sid,  const uint nsimd, const uint d4,
    const float  scale,
    thread float& m_run, thread float& l_run, thread float4& acc4)
{
    const size_t row = (size_t)Nk * Hd;
    const bool   act = (d4 < Hd);
    threadgroup const float4* q4 = (threadgroup const float4*)(qh_s + d4);
    threadgroup const float4* k4 = (threadgroup const float4*)(kh_s + d4);
    uint t = tbeg + sid;
    for (; t + 3u*nsimd < tend; t += 4u*nsimd) {
        const uint t0 = t, t1 = t + nsimd, t2 = t + 2u*nsimd, t3 = t + 3u*nsimd;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        float4 v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if (act) {
            d0 = dot(*q4, (t0 == tcur) ? *k4 : *(device const float4*)(Kc + (size_t)t0*row + kv_h*Hd + d4));
            d1 = dot(*q4, (t1 == tcur) ? *k4 : *(device const float4*)(Kc + (size_t)t1*row + kv_h*Hd + d4));
            d2 = dot(*q4, (t2 == tcur) ? *k4 : *(device const float4*)(Kc + (size_t)t2*row + kv_h*Hd + d4));
            d3 = dot(*q4, (t3 == tcur) ? *k4 : *(device const float4*)(Kc + (size_t)t3*row + kv_h*Hd + d4));
            v0 = *(device const float4*)(Vc + (size_t)t0*row + kv_h*Hd + d4);
            v1 = *(device const float4*)(Vc + (size_t)t1*row + kv_h*Hd + d4);
            v2 = *(device const float4*)(Vc + (size_t)t2*row + kv_h*Hd + d4);
            v3 = *(device const float4*)(Vc + (size_t)t3*row + kv_h*Hd + d4);
        }
        d0 += simd_shuffle_xor(d0, 16); d1 += simd_shuffle_xor(d1, 16);
        d2 += simd_shuffle_xor(d2, 16); d3 += simd_shuffle_xor(d3, 16);
        d0 += simd_shuffle_xor(d0, 8);  d1 += simd_shuffle_xor(d1, 8);
        d2 += simd_shuffle_xor(d2, 8);  d3 += simd_shuffle_xor(d3, 8);
        d0 += simd_shuffle_xor(d0, 4);  d1 += simd_shuffle_xor(d1, 4);
        d2 += simd_shuffle_xor(d2, 4);  d3 += simd_shuffle_xor(d3, 4);
        d0 += simd_shuffle_xor(d0, 2);  d1 += simd_shuffle_xor(d1, 2);
        d2 += simd_shuffle_xor(d2, 2);  d3 += simd_shuffle_xor(d3, 2);
        d0 += simd_shuffle_xor(d0, 1);  d1 += simd_shuffle_xor(d1, 1);
        d2 += simd_shuffle_xor(d2, 1);  d3 += simd_shuffle_xor(d3, 1);
        const float s0 = d0*scale, s1 = d1*scale, s2 = d2*scale, s3 = d3*scale;
        const float m_new = max(m_run, max(max(s0, s1), max(s2, s3)));
        const float resc  = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
        const float e0 = exp(s0 - m_new), e1 = exp(s1 - m_new);
        const float e2 = exp(s2 - m_new), e3 = exp(s3 - m_new);
        l_run = l_run * resc + ((e0 + e1) + (e2 + e3));
        acc4  = acc4 * resc + ((e0*v0 + e1*v1) + (e2*v2 + e3*v3));
        m_run = m_new;
    }
    for (; t < tend; t += nsimd) {
        float dp = 0.0f;
        float4 vv4 = 0.0f;
        if (act) {
            dp = dot(*q4, (t == tcur) ? *k4
                     : *(device const float4*)(Kc + (size_t)t*row + kv_h*Hd + d4));
            vv4 = *(device const float4*)(Vc + (size_t)t*row + kv_h*Hd + d4);
        }
        dp += simd_shuffle_xor(dp, 16);
        dp += simd_shuffle_xor(dp, 8);
        dp += simd_shuffle_xor(dp, 4);
        dp += simd_shuffle_xor(dp, 2);
        dp += simd_shuffle_xor(dp, 1);
        const float score = dp * scale;
        const float m_new = max(m_run, score);
        const float resc  = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
        const float e     = exp(score - m_new);
        l_run = l_run * resc + e;
        acc4  = acc4 * resc + e * vv4;
        m_run = m_new;
    }
}

/* Attention decode with fused per-head qk-norm + rope (qwen3-style).
 * One threadgroup per q head; the prologue normalizes+rotates this head's
 * q into shared memory and does the same for the current-position k row
 * (written back to the KV slot by the first q head of each kv group, so
 * later tokens read roped k). Replaces a separate qknorm+rope dispatch. */
kernel void attn_decode_qkr_f32(
    device const float* q       [[buffer(0)]],   // [Nq*Hd] raw (un-normed)
    device float*       Kc      [[buffer(1)]],   // KV cache; current k slot updated in place
    device const float* Vc      [[buffer(2)]],
    device float*       out     [[buffer(3)]],   // [Nq*Hd]
    device const float* qgain   [[buffer(4)]],   // [Hd]
    device const float* kgain   [[buffer(5)]],   // [Hd]
    constant uint&      Hd      [[buffer(6)]],
    constant uint&      Nq      [[buffer(7)]],
    constant uint&      Nk      [[buffer(8)]],
    constant uint&      kvlen   [[buffer(9)]],
    constant float&     scale   [[buffer(10)]],
    constant uint&      rope_dim [[buffer(11)]],
    constant int&       position [[buffer(12)]],
    constant float&     theta    [[buffer(13)]],
    constant uint&      neox     [[buffer(14)]],
    constant float&     eps      [[buffer(15)]],
    device const float* cs       [[buffer(16)]],
    device const float* kraw     [[buffer(17)]],   /* raw current-k (unroped) shadow */
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    if (Hd > 128u || tg > 512u) return;
    const uint kv_h = h * Nk / Nq;
    device const float* qh_raw = q + h*Hd;
    device float* kslot = Kc + (size_t)(kvlen-1)*Nk*Hd + kv_h*Hd;
    device const float* kr = kraw + kv_h*Hd;
    threadgroup float qh_s[128];
    threadgroup float kh_s[128];
    threadgroup float red[256];
    threadgroup float ssum;

    /* ---- q head + current k row: rmsnorm scales in ONE pass ----
     * simd_xor reduction + single cross-simd combine: 2 barriers total
     * instead of two ~log2(tg) tree reductions. */
    float lq = 0.0f, lk = 0.0f;
    for (uint i = tid; i < Hd; i += tg) {
        float v = qh_raw[i]; lq += v*v;
        float w = kr[i];     lk += w*w;
    }
    lq += simd_shuffle_xor(lq, 16); lk += simd_shuffle_xor(lk, 16);
    lq += simd_shuffle_xor(lq, 8);  lk += simd_shuffle_xor(lk, 8);
    lq += simd_shuffle_xor(lq, 4);  lk += simd_shuffle_xor(lk, 4);
    lq += simd_shuffle_xor(lq, 2);  lk += simd_shuffle_xor(lk, 2);
    lq += simd_shuffle_xor(lq, 1);  lk += simd_shuffle_xor(lk, 1);
    const uint nsimd0 = tg >> 5;
    if ((tid & 31u) == 0u) { red[tid >> 5] = lq; red[16u + (tid >> 5)] = lk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float a = 0.0f, b = 0.0f;
        for (uint i = 0; i < nsimd0; i++) { a += red[i]; b += red[16u + i]; }
        red[32] = a; red[33] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qscale = 1.0f / sqrt(red[32]/float(Hd) + eps);
    const float kscale = 1.0f / sqrt(red[33]/float(Hd) + eps);
    (void)ssum;

    /* ---- apply norm gain + rope into shared ---- */
    const uint pairs = rope_dim/2;
    for (uint kk = tid; kk < pairs; kk += tg) {
        uint i0 = neox ? kk : 2*kk;
        uint i1 = neox ? kk + pairs : 2*kk + 1;
        float c = cs[kk], s = cs[128u + kk];
        { float v0 = qh_raw[i0]*qscale*qgain[i0];
          float v1 = qh_raw[i1]*qscale*qgain[i1];
          qh_s[i0] = v0*c - v1*s;
          qh_s[i1] = v0*s + v1*c; }
        { float v0 = kr[i0]*kscale*kgain[i0];
          float v1 = kr[i1]*kscale*kgain[i1];
          kh_s[i0] = v0*c - v1*s;
          kh_s[i1] = v0*s + v1*c; }
    }
    for (uint i = rope_dim + tid; i < Hd; i += tg) {
        qh_s[i] = qh_raw[i]*qscale*qgain[i];
        kh_s[i] = kr[i]*kscale*kgain[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* write roped+normed current k back so future tokens see it;
     * only the first q head of each kv group writes (identical value). */
    if ((h % (Nq/Nk)) == 0) {
        for (uint i = tid; i < Hd; i += tg) kslot[i] = kh_s[i];
    }
    /* ---- single-pass flash-decode over [0, kvlen): each simd keeps a
     * running (m, l, acc4); lane l32 owns dims [l32*4, l32*4+4). One
     * merge barrier instead of score-buffer + tiled-V phases. ---- */
    const uint sid = tid >> 5, l32 = tid & 31u;
    const uint nsimd = tg >> 5;
    const uint d4 = l32 * 4u;
    float m_run = -INFINITY, l_run = 0.0f;
    float4 acc4 = 0.0f;
    flash_scan4(qh_s, kh_s, Kc, Vc, kv_h, Hd, Nk,
                0u, kvlen, kvlen - 1u, sid, nsimd, d4, scale,
                m_run, l_run, acc4);
    threadgroup float ms[16], ls[16];
    threadgroup float accm[16][128];
    if (l32 == 0u) { ms[sid] = m_run; ls[sid] = l_run; }
    if (d4 < Hd) *(threadgroup float4*)(accm[sid] + d4) = acc4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_st = ms[0], l_st = 0.0f;
    for (uint i = 1; i < nsimd; i++) m_st = max(m_st, ms[i]);
    for (uint i = 0; i < nsimd; i++) l_st += ls[i] * exp(ms[i] - m_st);
    const float inv = 1.0f / l_st;
    device float* oh = out + h*Hd;
    if (tid < Hd) {
        float a = 0.0f;
        for (uint i = 0; i < nsimd; i++)
            a += accm[i][tid] * exp(ms[i] - m_st);
        oh[tid] = a * inv;
    }

}

/* Chained-decode embedding gather: x[i] = embd[tok*H+i] where tok is read
 * from the GPU token ring (written by the previous token's argmax). Lets
 * the next token's command buffer run without a CPU round-trip. */
kernel void embd_gather_f32(
    device float*       x    [[buffer(0)]],
    device const float* embd [[buffer(1)]],
    device const uint*  tokp [[buffer(2)]],
    constant uint&      H    [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    device const float* src = embd + (size_t)(*tokp) * H;
    for (uint i = tid; i < H; i += tg) x[i] = src[i];
}

/* Flash-decoding split-KV variant of attn_decode_qkr_f32 for long contexts.
 * Grid = (Nq heads) x (nsplit seq tiles); each tg repeats the fused
 * qk-norm+rope prologue (cheap, identical results) and computes an
 * UNNORMALIZED partial attention over its position range:
 *   part[(h*nsplit+s)] = { m_s, l_s, acc_s[Hd] }
 * where acc_s = sum_t e^(score_t - m_s) * V_t  (softmax numerator).
 * attn_combine_f32 then merges partials across splits. The roped current-k
 * is written back by the split tg that owns position kvlen-1 (the last
 * split), once per kv group — same rule as the single-tg kernel. */
kernel void attn_decode_qkr_split_f32(
    device const float* q       [[buffer(0)]],
    device float*       Kc      [[buffer(1)]],
    device const float* Vc      [[buffer(2)]],
    device float*       part    [[buffer(3)]],   // [Nq*nsplit*(Hd+2)]
    device const float* qgain   [[buffer(4)]],
    device const float* kgain   [[buffer(5)]],
    constant uint&      Hd      [[buffer(6)]],
    constant uint&      Nq      [[buffer(7)]],
    constant uint&      Nk      [[buffer(8)]],
    constant uint&      kvlen   [[buffer(9)]],
    constant float&     scale   [[buffer(10)]],
    constant uint&      rope_dim [[buffer(11)]],
    constant int&       position [[buffer(12)]],
    constant float&     theta    [[buffer(13)]],
    constant uint&      neox     [[buffer(14)]],
    constant float&     eps      [[buffer(15)]],
    constant uint&      nsplit   [[buffer(16)]],
    device const float* cs       [[buffer(17)]],
    device atomic_uint* cnt      [[buffer(18)]],  // [Nq] self-resetting tickets
    device float*       out      [[buffer(19)]],  // [Nq*Hd] combine target
    device const float* kraw     [[buffer(20)]],   /* raw current-k (unroped) shadow */
    uint    tgid [[threadgroup_position_in_grid]],
    uint    tid  [[thread_position_in_threadgroup]],
    uint    tg   [[threads_per_threadgroup]])
{
    if (Hd > 128u || tg != 256u) return;
    const uint h = tgid / nsplit, s = tgid % nsplit;
    const uint per   = (kvlen + nsplit - 1u) / nsplit;
    const uint tbeg  = s * per;
    const uint tend  = min(kvlen, tbeg + per);
    const uint kv_h  = h * Nk / Nq;
    device const float* qh_raw = q + h*Hd;
    device float* kslot = Kc + (size_t)(kvlen-1)*Nk*Hd + kv_h*Hd;
    device const float* kr = kraw + kv_h*Hd;
    device float* pout  = part + (size_t)(h*nsplit + s)*(Hd + 2u);
    const uint nsimd0 = tg >> 5;
    threadgroup float qh_s[128];
    threadgroup float kh_s[128];
    threadgroup float red[256];
    threadgroup uint  tk;

    if (tbeg < kvlen) {
    /* ---- q head + current k row: rmsnorm scales in ONE pass ---- */
    float lq = 0.0f, lk = 0.0f;
    for (uint i = tid; i < Hd; i += tg) {
        float v = qh_raw[i]; lq += v*v;
        float w = kr[i];     lk += w*w;
    }
    lq += simd_shuffle_xor(lq, 16); lk += simd_shuffle_xor(lk, 16);
    lq += simd_shuffle_xor(lq, 8);  lk += simd_shuffle_xor(lk, 8);
    lq += simd_shuffle_xor(lq, 4);  lk += simd_shuffle_xor(lk, 4);
    lq += simd_shuffle_xor(lq, 2);  lk += simd_shuffle_xor(lk, 2);
    lq += simd_shuffle_xor(lq, 1);  lk += simd_shuffle_xor(lk, 1);
    if ((tid & 31u) == 0u) { red[tid >> 5] = lq; red[16u + (tid >> 5)] = lk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float a = 0.0f, b = 0.0f;
        for (uint i = 0; i < nsimd0; i++) { a += red[i]; b += red[16u + i]; }
        red[32] = a; red[33] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qscale = 1.0f / sqrt(red[32]/float(Hd) + eps);
    const float kscale = 1.0f / sqrt(red[33]/float(Hd) + eps);

    /* ---- apply norm gain + rope into shared ---- */
    const uint pairs = rope_dim/2;
    for (uint kk = tid; kk < pairs; kk += tg) {
        uint i0 = neox ? kk : 2*kk;
        uint i1 = neox ? kk + pairs : 2*kk + 1;
        float c = cs[kk], sn = cs[128u + kk];
        { float v0 = qh_raw[i0]*qscale*qgain[i0];
          float v1 = qh_raw[i1]*qscale*qgain[i1];
          qh_s[i0] = v0*c - v1*sn;
          qh_s[i1] = v0*sn + v1*c; }
        { float v0 = kr[i0]*kscale*kgain[i0];
          float v1 = kr[i1]*kscale*kgain[i1];
          kh_s[i0] = v0*c - v1*sn;
          kh_s[i1] = v0*sn + v1*c; }
    }
    for (uint i = rope_dim + tid; i < Hd; i += tg) {
        qh_s[i] = qh_raw[i]*qscale*qgain[i];
        kh_s[i] = kr[i]*kscale*kgain[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* write roped current k back: only the split owning kvlen-1 (the last
     * split) and only the first q head of each kv group. */
    if (s == nsplit - 1u && (h % (Nq/Nk)) == 0) {
        for (uint i = tid; i < Hd; i += tg) kslot[i] = kh_s[i];
    }

    /* ---- single-pass flash-decode over [tbeg, tend): each simd keeps a
     * running (m, l, acc4) — lane l32 owns dims [l32*4, l32*4+4). No score
     * buffer, no V tile: ~3 barriers instead of ~18. ---- */
    const uint sid  = tid >> 5, l32 = tid & 31u;
    const uint nsimd = tg >> 5;
    const uint d4   = l32 * 4u;
    float m_run = -INFINITY, l_run = 0.0f;
    float4 acc4 = 0.0f;
    flash_scan4(qh_s, kh_s, Kc, Vc, kv_h, Hd, Nk,
                tbeg, tend, kvlen - 1u, sid, nsimd, d4, scale,
                m_run, l_run, acc4);
    /* merge nsimd partials in shared: ms/ls per simd + acc row each */
    threadgroup float ms[8], ls[8];
    threadgroup float accm[8][128];
    if (l32 == 0u) { ms[sid] = m_run; ls[sid] = l_run; }
    if (d4 < Hd) *(threadgroup float4*)(accm[sid] + d4) = acc4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_tg = ms[0], l_tg = 0.0f;
    for (uint i = 1; i < nsimd; i++) m_tg = max(m_tg, ms[i]);
    for (uint i = 0; i < nsimd; i++) l_tg += ls[i] * exp(ms[i] - m_tg);
    if (tid == 0u) { pout[0] = m_tg; pout[1] = l_tg; }
    if (tid < Hd) {
        float a = 0.0f;
        for (uint i = 0; i < nsimd; i++)
            a += accm[i][tid] * exp(ms[i] - m_tg);
        pout[2u + tid] = a;
    }

    } else {                      /* empty split: neutral partial */
        if (tid == 0u) { pout[0] = -INFINITY; pout[1] = 0.0f; }
        for (uint i = tid; i < Hd; i += tg) pout[2u + i] = 0.0f;
    }

#if defined(__HAVE_ATOMIC_FENCE__)
    /* ---- last-arriving tg for head h merges all splits inline ----
     * Device-scope release fence + ticket: the tg that observes
     * ticket == nsplit-1 knows every sibling's partial writes are
     * visible and performs the softmax merge for this head. The
     * counter self-resets so the next layer/token reuses it.
     * Requires MSL >= 3.2 device fences (__HAVE_ATOMIC_FENCE__); older
     * toolchains compile this kernel without the merge and the host
     * dispatches attn_combine_f32 instead (see atomic_fence_probe). */
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (tid == 0u)
        tk = atomic_fetch_add_explicit(&cnt[h], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tk != nsplit - 1u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (tid == 0u) atomic_store_explicit(&cnt[h], 0u, memory_order_relaxed);

    device const float* pb = part + (size_t)h * nsplit * (Hd + 2u);
    float m2 = -INFINITY;
    for (uint s2 = tid; s2 < nsplit; s2 += tg) m2 = max(m2, pb[s2*(Hd+2u)]);
    m2 = max(m2, simd_shuffle_xor(m2, 16));
    m2 = max(m2, simd_shuffle_xor(m2, 8));
    m2 = max(m2, simd_shuffle_xor(m2, 4));
    m2 = max(m2, simd_shuffle_xor(m2, 2));
    m2 = max(m2, simd_shuffle_xor(m2, 1));
    if ((tid & 31u) == 0u) red[tid >> 5] = m2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float mm = -INFINITY;
        for (uint i = 0; i < nsimd0; i++) mm = max(mm, red[i]);
        red[32] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gm = red[32];
    float l2 = 0.0f;
    for (uint s2 = tid; s2 < nsplit; s2 += tg)
        l2 += pb[s2*(Hd+2u) + 1u] * exp(pb[s2*(Hd+2u)] - gm);
    l2 += simd_shuffle_xor(l2, 16);
    l2 += simd_shuffle_xor(l2, 8);
    l2 += simd_shuffle_xor(l2, 4);
    l2 += simd_shuffle_xor(l2, 2);
    l2 += simd_shuffle_xor(l2, 1);
    if ((tid & 31u) == 0u) red[16u + (tid >> 5)] = l2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sv = 0.0f;
        for (uint i = 0; i < nsimd0; i++) sv += red[16u + i];
        red[33] = sv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / red[33];
    for (uint d4 = tid*4u; d4 < Hd; d4 += tg*4u) {
        float4 a4 = 0.0f;
        for (uint s2 = 0u; s2 < nsplit; s2++) {
            device const float* pp = pb + s2*(Hd+2u);
            a4 += exp(pp[0] - gm) * (*(device const float4*)(pp + 2u + d4));
        }
        *(device float4*)(out + (size_t)h*Hd + d4) = a4 * inv;
    }
#endif
}

#if defined(__HAVE_ATOMIC_FENCE__)
/* Presence probe: this kernel exists only when the toolchain provides
 * device-scope atomic fences (MSL >= 3.2). The host loads it to decide
 * between the fused in-kernel combine and a separate attn_combine_f32
 * dispatch. Never dispatched. */
kernel void atomic_fence_probe() {}
#endif

/* Split-KV partial merge for toolchains without device atomic fences
 * (MSL < 3.2): one threadgroup per head reduces the nsplit partials
 * written by attn_decode_qkr_split_f32. A dispatch boundary supplies the
 * device-scope ordering the fused path gets from atomic_thread_fence. */
kernel void attn_combine_f32(
    device const float* part   [[buffer(0)]],   // [Nq*nsplit*(Hd+2)]
    device float*       out    [[buffer(1)]],   // [Nq*Hd]
    constant uint&      Hd     [[buffer(2)]],
    constant uint&      nsplit [[buffer(3)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    device const float* pb = part + (size_t)h * nsplit * (Hd + 2u);
    threadgroup float red[64];
    const uint nsimd = tg >> 5;
    float m2 = -INFINITY;
    for (uint s2 = tid; s2 < nsplit; s2 += tg) m2 = max(m2, pb[s2*(Hd+2u)]);
    m2 = max(m2, simd_shuffle_xor(m2, 16));
    m2 = max(m2, simd_shuffle_xor(m2, 8));
    m2 = max(m2, simd_shuffle_xor(m2, 4));
    m2 = max(m2, simd_shuffle_xor(m2, 2));
    m2 = max(m2, simd_shuffle_xor(m2, 1));
    if ((tid & 31u) == 0u) red[tid >> 5] = m2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float mm = -INFINITY;
        for (uint i = 0; i < nsimd; i++) mm = max(mm, red[i]);
        red[32] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gm = red[32];
    float l2 = 0.0f;
    for (uint s2 = tid; s2 < nsplit; s2 += tg)
        l2 += pb[s2*(Hd+2u) + 1u] * exp(pb[s2*(Hd+2u)] - gm);
    l2 += simd_shuffle_xor(l2, 16);
    l2 += simd_shuffle_xor(l2, 8);
    l2 += simd_shuffle_xor(l2, 4);
    l2 += simd_shuffle_xor(l2, 2);
    l2 += simd_shuffle_xor(l2, 1);
    if ((tid & 31u) == 0u) red[16u + (tid >> 5)] = l2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sv = 0.0f;
        for (uint i = 0; i < nsimd; i++) sv += red[16u + i];
        red[33] = sv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / red[33];
    for (uint d4 = tid*4u; d4 < Hd; d4 += tg*4u) {
        float4 a4 = 0.0f;
        for (uint s2 = 0u; s2 < nsplit; s2++) {
            device const float* pp = pb + s2*(Hd+2u);
            a4 += exp(pp[0] - gm) * (*(device const float4*)(pp + 2u + d4));
        }
        *(device float4*)(out + (size_t)h*Hd + d4) = a4 * inv;
    }
}

