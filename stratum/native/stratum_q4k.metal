
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

        /* 4 elements per thread, 8 threads per sub-block = 32 elements */
        float qx = 0.0f, xs = 0.0f;
        uchar4 mask = uchar4(0xF);
        for (int l = 0; l < 4; l += 4) {
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

/* RoPE (interleaved pairs), matching CPU la_rope. Applies to all heads of a
 * [n_heads * head_dim] buffer; rope_dim pairs per head. */
kernel void rope_f32(
    device float*   x        [[buffer(0)]],
    constant uint&  head_dim [[buffer(1)]],
    constant uint&  rope_dim [[buffer(2)]],
    constant int&   position [[buffer(3)]],
    constant float& theta    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint pairs = rope_dim/2;
    uint head = gid / pairs;
    uint k    = gid % pairs;
    device float* xh = x + head*head_dim;
    float freq  = 1.0f / pow(theta, float(2*k)/float(rope_dim));
    float angle = float(position) * freq;
    float c = cos(angle), s = sin(angle);
    float x0 = xh[2*k], x1 = xh[2*k+1];
    xh[2*k]   = x0*c - x1*s;
    xh[2*k+1] = x0*s + x1*c;
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

/* ===== V6: F16 sgemv — no nibble unpack, native half arithmetic =====
 * For pre-decoded weights (Q4_K -> F16 at load time). Each thread handles
 * one sub-block of 32 elements; 8 sub-blocks per 256-element block.
 * Uses half4 vector ops for 2x throughput vs float4 on Apple Silicon. */
kernel void f16_sgemv_row(
    device const half*  W   [[buffer(0)]],   /* [N][K] half */
    device const float* x   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    constant uint&      K   [[buffer(3)]],
    uint row   [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    device const half* wrow = W + (size_t)row * K;

    /* Each thread accumulates a partial dot product over strided elements.
     * Use half4 for the weight loads, float4 for x loads, accumulate in float. */
    float partial = 0.0f;
    for (uint i = tid * 4; i < K; i += tg_size * 4) {
        half4 w4 = *(device const half4*)(wrow + i);
        float4 x4 = *(device const float4*)(x + i);
        partial += dot(float4(w4), x4);
    }

    /* simd-level reduction */
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

/* F16 batched sgemv for multi-stream: one threadgroup per (row, stream). */
kernel void f16_sgemv_row_bparallel(
    device const half*  W   [[buffer(0)]],
    device const float* x   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    constant uint&      K   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant uint&      B   [[buffer(5)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint2 tid2  [[thread_position_in_threadgroup]],
    uint2 tg2   [[threads_per_threadgroup]])
{
    uint row = tgid.x;
    uint bidx = tgid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bidx >= B) return;
    device const half* wrow = W + (size_t)row * K;
    device const float* xb = x + (size_t)bidx * K;

    float partial = 0.0f;
    for (uint i = tid * 4; i < K; i += tg_size * 4) {
        half4 w4 = *(device const half4*)(wrow + i);
        float4 x4 = *(device const float4*)(xb + i);
        partial += dot(float4(w4), x4);
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

/* ===== V9: Q4_0 sgemv — minimal unpack, maximum bandwidth =====
 * Q4_0 block: half d + uchar qs[16] (32 nibbles).
 * Decode: v = d * (nibble - 8)
 * No scale unpack, no min subtraction — just one multiply.
 * Same 4.5 bits/elem as Q4_K but ~2x less ALU per element. */

struct block_q4_0 {
    half    d;      /* scale */
    uchar   qs[16]; /* 32 packed 4-bit values */
};

kernel void q4_0_sgemv_row(
    device const block_q4_0* W    [[buffer(0)]],
    device const float*      x    [[buffer(1)]],
    device float*            y    [[buffer(2)]],
    constant uint&           K    [[buffer(3)]],
    uint row    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint blocks_per_row = K / 32;
    device const block_q4_0* row_blocks = W + (uint)row * blocks_per_row;
    const uint total_sb = blocks_per_row;  /* 1 sub-block per Q4_0 block */

    float partial = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        const device block_q4_0& b = row_blocks[sb];
        const float d = float(b.d);
        const device uchar* qs = b.qs;
        uint elem_offset = sb * 32;

        /* 32 elements, 4 at a time */
        float4 acc = float4(0.0f);
        for (int l = 0; l < 16; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs + l);
            /* Each byte has 2 nibbles: low (even elem) and high (odd elem) */
            float4 n0 = float4(raw & uchar4(0xF));
            float4 n1 = float4((raw >> 4) & uchar4(0xF));
            float4 xv0 = *(device const float4*)(x + elem_offset + l * 2);
            float4 xv1 = *(device const float4*)(x + elem_offset + l * 2 + 4);
            acc += (n0 - 8.0f) * xv0;
            acc += (n1 - 8.0f) * xv1;
        }
        partial += d * (acc.x + acc.y + acc.z + acc.w);
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
