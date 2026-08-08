
#include <metal_stdlib>
using namespace metal;

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    half  d;
};

kernel void q6k_sgemv_row(
    device const block_q6_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    uint                     row         [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]],
    uint                     tg_size     [[threads_per_threadgroup]])
{
    const uint blocks_per_row = K / 256;
    device const block_q6_K* row_blocks = W + (uint)row * blocks_per_row;
    const uint total_half = blocks_per_row * 2;   /* one 128-element half per thread */

    float partial = 0.0f;
    for (uint u = tid; u < total_half; u += tg_size) {
        uint i = u >> 1;
        uint n = (u & 1) * 128;
        const device block_q6_K& b = row_blocks[i];
        const float d = float(b.d);
        {
            const device uchar* ql = b.ql + n / 2;
            const device uchar* qh = b.qh + n / 4;
            const device char*  s  = b.scales + n / 16;

            uint base = i * 256 + n;

            float a1[2] = {0, 0};
            float a2[2] = {0, 0};
            float a3[2] = {0, 0};
            float a4[2] = {0, 0};

            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = int((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = int((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = int((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                a1[is] += float(q1) * x[base + l +  0];
                a2[is] += float(q2) * x[base + l + 32];
                a3[is] += float(q3) * x[base + l + 64];
                a4[is] += float(q4) * x[base + l + 96];
            }
            for (int is = 0; is < 2; is++) {
                partial += d * float(s[is + 0]) * a1[is];
                partial += d * float(s[is + 2]) * a2[is];
                partial += d * float(s[is + 4]) * a3[is];
                partial += d * float(s[is + 6]) * a4[is];
            }
        }
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

/* Batched Q6_K, generated per literal BC for exact register allocation. */
#define DEFINE_Q6K_BATCH_KERNEL(BC) \
kernel void q6k_sgemv_row_batched_b##BC( \
    device const block_q6_K* W           [[buffer(0)]], \
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
    device const block_q6_K* row_blocks = W + (uint)row * blocks_per_row; \
    const uint total_half = blocks_per_row * 2; \
    float partial[BC]; \
    for (uint s = 0; s < BC; s++) partial[s] = 0.0f; \
    for (uint u = tid; u < total_half; u += tg_size) { \
        uint i = u >> 1; \
        uint n = (u & 1) * 128; \
        const device block_q6_K& b = row_blocks[i]; \
        const float d = float(b.d); \
        const device uchar* ql = b.ql + n / 2; \
        const device uchar* qh = b.qh + n / 4; \
        const device char* sc = b.scales + n / 16; \
        uint base = i * 256 + n; \
        for (int l = 0; l < 32; l++) { \
            int is = l / 16; \
            float q1 = float(int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32); \
            float q2 = float(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32); \
            float q3 = float(int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32); \
            float q4 = float(int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32); \
            float c1 = d * float(sc[is + 0]) * q1; \
            float c2 = d * float(sc[is + 2]) * q2; \
            float c3 = d * float(sc[is + 4]) * q3; \
            float c4 = d * float(sc[is + 6]) * q4; \
            for (uint s = 0; s < BC; s++) { \
                const device float* xs = x + s * K + base; \
                partial[s] += c1 * xs[l + 0] + c2 * xs[l + 32] + c3 * xs[l + 64] + c4 * xs[l + 96]; \
            } \
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

DEFINE_Q6K_BATCH_KERNEL(1)
DEFINE_Q6K_BATCH_KERNEL(2)
DEFINE_Q6K_BATCH_KERNEL(3)
DEFINE_Q6K_BATCH_KERNEL(4)
DEFINE_Q6K_BATCH_KERNEL(5)
DEFINE_Q6K_BATCH_KERNEL(6)
DEFINE_Q6K_BATCH_KERNEL(7)
DEFINE_Q6K_BATCH_KERNEL(8)
DEFINE_Q6K_BATCH_KERNEL(9)
DEFINE_Q6K_BATCH_KERNEL(10)
DEFINE_Q6K_BATCH_KERNEL(11)
DEFINE_Q6K_BATCH_KERNEL(12)
DEFINE_Q6K_BATCH_KERNEL(13)
DEFINE_Q6K_BATCH_KERNEL(14)
DEFINE_Q6K_BATCH_KERNEL(15)
DEFINE_Q6K_BATCH_KERNEL(16)
DEFINE_Q6K_BATCH_KERNEL(17)
DEFINE_Q6K_BATCH_KERNEL(18)
DEFINE_Q6K_BATCH_KERNEL(19)
DEFINE_Q6K_BATCH_KERNEL(20)
DEFINE_Q6K_BATCH_KERNEL(21)
DEFINE_Q6K_BATCH_KERNEL(22)
DEFINE_Q6K_BATCH_KERNEL(23)
DEFINE_Q6K_BATCH_KERNEL(24)
DEFINE_Q6K_BATCH_KERNEL(25)
DEFINE_Q6K_BATCH_KERNEL(26)
DEFINE_Q6K_BATCH_KERNEL(27)
DEFINE_Q6K_BATCH_KERNEL(28)
DEFINE_Q6K_BATCH_KERNEL(29)
DEFINE_Q6K_BATCH_KERNEL(30)
DEFINE_Q6K_BATCH_KERNEL(31)
DEFINE_Q6K_BATCH_KERNEL(32)
#undef DEFINE_Q6K_BATCH_KERNEL

kernel void q6k_sgemv_row_bparallel(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_half = blocks_per_row * 2;

    float partial = 0.0f;
    for (uint u = tid; u < total_half; u += tg_size) {
        uint i = u >> 1;
        uint n = (u & 1) * 128;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const device uchar* ql = bl.ql + n / 2;
        const device uchar* qh = bl.qh + n / 4;
        const device char* sc = bl.scales + n / 16;
        uint base = i * 256 + n;

        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            float q1 = float(int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            float q2 = float(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            float q3 = float(int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            float q4 = float(int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            partial += d * (
                float(sc[is + 0]) * q1 * xb[base + l + 0] +
                float(sc[is + 2]) * q2 * xb[base + l + 32] +
                float(sc[is + 4]) * q3 * xb[base + l + 64] +
                float(sc[is + 6]) * q4 * xb[base + l + 96]);
        }
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

/* Q6_K bparallel matmul fused with residual add: residual[b][row] += W*x[b].
 * Used for FFN down_proj, where the unfused path writes a temp H-vector then
 * launches a second add kernel. */
kernel void q6k_sgemv_row_bparallel_add(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_half = blocks_per_row * 2;

    float partial = 0.0f;
    for (uint u = tid; u < total_half; u += tg_size) {
        uint i = u >> 1;
        uint n = (u & 1) * 128;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const device uchar* ql = bl.ql + n / 2;
        const device uchar* qh = bl.qh + n / 4;
        const device char* sc = bl.scales + n / 16;
        uint base = i * 256 + n;

        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            float q1 = float(int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            float q2 = float(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            float q3 = float(int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            float q4 = float(int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            partial += d * (
                float(sc[is + 0]) * q1 * xb[base + l + 0] +
                float(sc[is + 2]) * q2 * xb[base + l + 32] +
                float(sc[is + 4]) * q3 * xb[base + l + 64] +
                float(sc[is + 6]) * q4 * xb[base + l + 96]);
        }
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

/* Vectorized Q6_K bparallel variant. Same math/order at the 32-lane chunk
 * granularity as q6k_sgemv_row_bparallel, but uses float4 accumulators to
 * shorten scalar dependency chains in the 128-element half-block loop. */
kernel void q6k_sgemv_row_bparallel_v4(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_half = blocks_per_row * 2;

    float partial = 0.0f;
    for (uint u = tid; u < total_half; u += tg_size) {
        uint i = u >> 1;
        uint n = (u & 1) * 128;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const device uchar* ql = bl.ql + n / 2;
        const device uchar* qh = bl.qh + n / 4;
        const device char* sc = bl.scales + n / 16;
        uint base = i * 256 + n;

        float4 acc = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            int is = l / 16;
            float s0 = d * float(sc[is + 0]);
            float s1 = d * float(sc[is + 2]);
            float s2 = d * float(sc[is + 4]);
            float s3 = d * float(sc[is + 6]);
            uchar4 ql0 = *(device const uchar4*)(ql + l);
            uchar4 ql1 = *(device const uchar4*)(ql + l + 32);
            uchar4 qh4 = *(device const uchar4*)(qh + l);
            float4 q1 = float4((ql0 & uchar4(0xF)) | (((qh4 >> uchar4(0)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            float4 q2 = float4((ql1 & uchar4(0xF)) | (((qh4 >> uchar4(2)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            float4 q3 = float4((ql0 >> uchar4(4)) | (((qh4 >> uchar4(4)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            float4 q4 = float4((ql1 >> uchar4(4)) | (((qh4 >> uchar4(6)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            acc += s0 * q1 * *(device const float4*)(xb + base + l + 0);
            acc += s1 * q2 * *(device const float4*)(xb + base + l + 32);
            acc += s2 * q3 * *(device const float4*)(xb + base + l + 64);
            acc += s3 * q4 * *(device const float4*)(xb + base + l + 96);
        }
        partial += acc.x + acc.y + acc.z + acc.w;
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

/* Q6_K bparallel for two adjacent streams per threadgroup. It reuses each
 * decoded weight row across two x vectors; intended for FFN down_proj. */
kernel void q6k_sgemv_row_bparallel_g2(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0idx * K;
    device const float* xb1 = x + (size_t)b1idx * K;
    const uint total_half = blocks_per_row * 2;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint u = tid; u < total_half; u += tg_size) {
        uint i = u >> 1;
        uint n = (u & 1) * 128;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const device uchar* ql = bl.ql + n / 2;
        const device uchar* qh = bl.qh + n / 4;
        const device char* sc = bl.scales + n / 16;
        uint base = i * 256 + n;

        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            int is = l / 16;
            float s0 = d * float(sc[is + 0]);
            float s1 = d * float(sc[is + 2]);
            float s2 = d * float(sc[is + 4]);
            float s3 = d * float(sc[is + 6]);
            uchar4 ql0 = *(device const uchar4*)(ql + l);
            uchar4 ql1 = *(device const uchar4*)(ql + l + 32);
            uchar4 qh4 = *(device const uchar4*)(qh + l);
            float4 q1 = float4((ql0 & uchar4(0xF)) | (((qh4 >> uchar4(0)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            float4 q2 = float4((ql1 & uchar4(0xF)) | (((qh4 >> uchar4(2)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            float4 q3 = float4((ql0 >> uchar4(4)) | (((qh4 >> uchar4(4)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            float4 q4 = float4((ql1 >> uchar4(4)) | (((qh4 >> uchar4(6)) & uchar4(3)) << uchar4(4))) - float4(32.0f);
            float4 xv00 = *(device const float4*)(xb0 + base + l + 0);
            float4 xv01 = *(device const float4*)(xb0 + base + l + 32);
            float4 xv02 = *(device const float4*)(xb0 + base + l + 64);
            float4 xv03 = *(device const float4*)(xb0 + base + l + 96);
            acc0 += s0 * q1 * xv00;
            acc0 += s1 * q2 * xv01;
            acc0 += s2 * q3 * xv02;
            acc0 += s3 * q4 * xv03;
            if (b1idx < B) {
                float4 xv10 = *(device const float4*)(xb1 + base + l + 0);
                float4 xv11 = *(device const float4*)(xb1 + base + l + 32);
                float4 xv12 = *(device const float4*)(xb1 + base + l + 64);
                float4 xv13 = *(device const float4*)(xb1 + base + l + 96);
                acc1 += s0 * q1 * xv10;
                acc1 += s1 * q2 * xv11;
                acc1 += s2 * q3 * xv12;
                acc1 += s3 * q4 * xv13;
            }
        }
        partial0 += acc0.x + acc0.y + acc0.z + acc0.w;
        partial1 += acc1.x + acc1.y + acc1.z + acc1.w;
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

/* Finer-grained Q6_K bparallel variant. One work item covers one 32-element
 * scale group instead of a full 128-element half-block, keeping all 64 lanes
 * busy for common K=2048/5632 projections. */
kernel void q6k_sgemv_row_bparallel_v5(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_groups = blocks_per_row * 8;

    float partial = 0.0f;
    for (uint gidx = tid; gidx < total_groups; gidx += tg_size) {
        uint i = gidx >> 3;
        uint g = gidx & 7u;
        uint hpart = g >> 2;
        uint sub = g & 3u;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);

        uint ql_base = hpart * 64u + (sub & 1u) * 32u;
        uint qh_base = hpart * 32u;
        uint x_base = i * 256u + hpart * 128u + sub * 32u;
        uint sc_base = hpart * 8u + sub * 2u;
        uchar4 qh_shift = uchar4((uchar)(sub * 2u));
        uchar4 qmask = uchar4(0xFu);

        float4 acc = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 ql4 = *(device const uchar4*)(bl.ql + ql_base + l);
            uchar4 qh4 = *(device const uchar4*)(bl.qh + qh_base + l);
            uchar4 lo = (sub < 2u) ? (ql4 & qmask) : (ql4 >> uchar4(4u));
            uchar4 hi = ((qh4 >> qh_shift) & uchar4(3u)) << uchar4(4u);
            float4 qv = float4(lo | hi) - float4(32.0f);
            float s = d * float(bl.scales[sc_base + (uint)l / 16u]);
            acc += s * qv * *(device const float4*)(xb + x_base + l);
        }
        partial += acc.x + acc.y + acc.z + acc.w;
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

/* Q6_K v5 for two adjacent streams per threadgroup. This keeps v5's
 * 32-value work split, but reuses the decoded weight group for two x rows. */
kernel void q6k_sgemv_row_bparallel_v5_g2(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0idx * K;
    device const float* xb1 = x + (size_t)b1idx * K;
    const uint total_groups = blocks_per_row * 8;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint gidx = tid; gidx < total_groups; gidx += tg_size) {
        uint i = gidx >> 3;
        uint g = gidx & 7u;
        uint hpart = g >> 2;
        uint sub = g & 3u;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);

        uint ql_base = hpart * 64u + (sub & 1u) * 32u;
        uint qh_base = hpart * 32u;
        uint x_base = i * 256u + hpart * 128u + sub * 32u;
        uint sc_base = hpart * 8u + sub * 2u;
        uchar4 qh_shift = uchar4((uchar)(sub * 2u));
        uchar4 qmask = uchar4(0xFu);

        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 ql4 = *(device const uchar4*)(bl.ql + ql_base + l);
            uchar4 qh4 = *(device const uchar4*)(bl.qh + qh_base + l);
            uchar4 lo = (sub < 2u) ? (ql4 & qmask) : (ql4 >> uchar4(4u));
            uchar4 hi = ((qh4 >> qh_shift) & uchar4(3u)) << uchar4(4u);
            float4 qv = float4(lo | hi) - float4(32.0f);
            float s = d * float(bl.scales[sc_base + (uint)l / 16u]);
            acc0 += s * qv * *(device const float4*)(xb0 + x_base + l);
            if (b1idx < B) {
                acc1 += s * qv * *(device const float4*)(xb1 + x_base + l);
            }
        }
        partial0 += acc0.x + acc0.y + acc0.z + acc0.w;
        partial1 += acc1.x + acc1.y + acc1.z + acc1.w;
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

/* Q6_K v5_g2 fused with residual add for FFN down_proj. */
kernel void q6k_sgemv_row_bparallel_v5_g2_add(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb0 = x + (size_t)b0idx * K;
    device const float* xb1 = x + (size_t)b1idx * K;
    const uint total_groups = blocks_per_row * 8;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint gidx = tid; gidx < total_groups; gidx += tg_size) {
        uint i = gidx >> 3;
        uint g = gidx & 7u;
        uint hpart = g >> 2;
        uint sub = g & 3u;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);

        uint ql_base = hpart * 64u + (sub & 1u) * 32u;
        uint qh_base = hpart * 32u;
        uint x_base = i * 256u + hpart * 128u + sub * 32u;
        uint sc_base = hpart * 8u + sub * 2u;
        uchar4 qh_shift = uchar4((uchar)(sub * 2u));
        uchar4 qmask = uchar4(0xFu);

        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 ql4 = *(device const uchar4*)(bl.ql + ql_base + l);
            uchar4 qh4 = *(device const uchar4*)(bl.qh + qh_base + l);
            uchar4 lo = (sub < 2u) ? (ql4 & qmask) : (ql4 >> uchar4(4u));
            uchar4 hi = ((qh4 >> qh_shift) & uchar4(3u)) << uchar4(4u);
            float4 qv = float4(lo | hi) - float4(32.0f);
            float s = d * float(bl.scales[sc_base + (uint)l / 16u]);
            acc0 += s * qv * *(device const float4*)(xb0 + x_base + l);
            if (b1idx < B) {
                acc1 += s * qv * *(device const float4*)(xb1 + x_base + l);
            }
        }
        partial0 += acc0.x + acc0.y + acc0.z + acc0.w;
        partial1 += acc1.x + acc1.y + acc1.z + acc1.w;
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

/* Q6_K v5 for four adjacent streams per threadgroup. High-B experiment:
 * reuse the decoded 32-value weight group across four x rows. */
kernel void q6k_sgemv_row_bparallel_v5_g4(
    device const block_q6_K* W           [[buffer(0)]],
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
    uint bbase = tgid.y * 4u;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    if (row >= N || bbase >= B) return;

    const uint blocks_per_row = K / 256;
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb0 = x + (size_t)(bbase + 0u) * K;
    device const float* xb1 = x + (size_t)(bbase + 1u) * K;
    device const float* xb2 = x + (size_t)(bbase + 2u) * K;
    device const float* xb3 = x + (size_t)(bbase + 3u) * K;
    const uint total_groups = blocks_per_row * 8;

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    for (uint gidx = tid; gidx < total_groups; gidx += tg_size) {
        uint i = gidx >> 3;
        uint g = gidx & 7u;
        uint hpart = g >> 2;
        uint sub = g & 3u;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);

        uint ql_base = hpart * 64u + (sub & 1u) * 32u;
        uint qh_base = hpart * 32u;
        uint x_base = i * 256u + hpart * 128u + sub * 32u;
        uint sc_base = hpart * 8u + sub * 2u;
        uchar4 qh_shift = uchar4((uchar)(sub * 2u));
        uchar4 qmask = uchar4(0xFu);

        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        float4 acc3 = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 ql4 = *(device const uchar4*)(bl.ql + ql_base + l);
            uchar4 qh4 = *(device const uchar4*)(bl.qh + qh_base + l);
            uchar4 lo = (sub < 2u) ? (ql4 & qmask) : (ql4 >> uchar4(4u));
            uchar4 hi = ((qh4 >> qh_shift) & uchar4(3u)) << uchar4(4u);
            float4 qv = float4(lo | hi) - float4(32.0f);
            float s = d * float(bl.scales[sc_base + (uint)l / 16u]);
            acc0 += s * qv * *(device const float4*)(xb0 + x_base + l);
            if (bbase + 1u < B) acc1 += s * qv * *(device const float4*)(xb1 + x_base + l);
            if (bbase + 2u < B) acc2 += s * qv * *(device const float4*)(xb2 + x_base + l);
            if (bbase + 3u < B) acc3 += s * qv * *(device const float4*)(xb3 + x_base + l);
        }
        p0 += acc0.x + acc0.y + acc0.z + acc0.w;
        p1 += acc1.x + acc1.y + acc1.z + acc1.w;
        p2 += acc2.x + acc2.y + acc2.z + acc2.w;
        p3 += acc3.x + acc3.y + acc3.z + acc3.w;
    }

    threadgroup float s0[256];
    threadgroup float s1[256];
    threadgroup float s2[256];
    threadgroup float s3[256];
    float sg0 = simd_sum(p0);
    float sg1 = simd_sum(p1);
    float sg2 = simd_sum(p2);
    float sg3 = simd_sum(p3);
    uint simd_id = tid / 32u;
    uint lane = tid % 32u;
    if (lane == 0) {
        s0[simd_id] = sg0;
        s1[simd_id] = sg1;
        s2[simd_id] = sg2;
        s3[simd_id] = sg3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint nsimd = (tg_size + 31u) / 32u;
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        for (uint s = 0; s < nsimd; s++) {
            t0 += s0[s];
            t1 += s1[s];
            t2 += s2[s];
            t3 += s3[s];
        }
        y[(size_t)(bbase + 0u) * N + row] = t0;
        if (bbase + 1u < B) y[(size_t)(bbase + 1u) * N + row] = t1;
        if (bbase + 2u < B) y[(size_t)(bbase + 2u) * N + row] = t2;
        if (bbase + 3u < B) y[(size_t)(bbase + 3u) * N + row] = t3;
    }
}

/* Q6_K bparallel variant with 64 values per work item. This is the midpoint
 * between v4's 128-value half-block and v5's 32-value scale group: enough
 * row-local work to keep 64 lanes occupied on down_proj, with less per-group
 * indexing overhead than v5. */
kernel void q6k_sgemv_row_bparallel_v6(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_pairs = blocks_per_row * 4;

    float partial = 0.0f;
    for (uint pidx = tid; pidx < total_pairs; pidx += tg_size) {
        uint i = pidx >> 2;
        uint pair_in_block = pidx & 3u;
        uint hpart = pair_in_block >> 1;
        uint pair = pair_in_block & 1u;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);

        uint ql_base = hpart * 64u;
        uint qh_base = hpart * 32u;
        uint x_base = i * 256u + hpart * 128u + pair * 64u;
        uint sc_base = hpart * 8u + pair * 4u;

        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        if (pair == 0u) {
            for (int l = 0; l < 32; l += 4) {
                uchar4 ql0 = *(device const uchar4*)(bl.ql + ql_base + l);
                uchar4 ql1 = *(device const uchar4*)(bl.ql + ql_base + 32u + l);
                uchar4 qh4 = *(device const uchar4*)(bl.qh + qh_base + l);
                float4 q0 = float4((ql0 & uchar4(0xFu)) | (((qh4 >> uchar4(0u)) & uchar4(3u)) << uchar4(4u))) - float4(32.0f);
                float4 q1 = float4((ql1 & uchar4(0xFu)) | (((qh4 >> uchar4(2u)) & uchar4(3u)) << uchar4(4u))) - float4(32.0f);
                float s0 = d * float(bl.scales[sc_base + (uint)l / 16u]);
                float s1 = d * float(bl.scales[sc_base + 2u + (uint)l / 16u]);
                acc0 += s0 * q0 * *(device const float4*)(xb + x_base + l);
                acc1 += s1 * q1 * *(device const float4*)(xb + x_base + 32u + l);
            }
        } else {
            for (int l = 0; l < 32; l += 4) {
                uchar4 ql0 = *(device const uchar4*)(bl.ql + ql_base + l);
                uchar4 ql1 = *(device const uchar4*)(bl.ql + ql_base + 32u + l);
                uchar4 qh4 = *(device const uchar4*)(bl.qh + qh_base + l);
                float4 q0 = float4((ql0 >> uchar4(4u)) | (((qh4 >> uchar4(4u)) & uchar4(3u)) << uchar4(4u))) - float4(32.0f);
                float4 q1 = float4((ql1 >> uchar4(4u)) | (((qh4 >> uchar4(6u)) & uchar4(3u)) << uchar4(4u))) - float4(32.0f);
                float s0 = d * float(bl.scales[sc_base + (uint)l / 16u]);
                float s1 = d * float(bl.scales[sc_base + 2u + (uint)l / 16u]);
                acc0 += s0 * q0 * *(device const float4*)(xb + x_base + l);
                acc1 += s1 * q1 * *(device const float4*)(xb + x_base + 32u + l);
            }
        }
        partial += acc0.x + acc0.y + acc0.z + acc0.w;
        partial += acc1.x + acc1.y + acc1.z + acc1.w;
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

/* Q6_K v5 fused with residual add for FFN down_proj. Same work split as v5,
 * but writes residual[b][row] += W*x[b] to skip temp write + add kernel. */
kernel void q6k_sgemv_row_bparallel_v5_add(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)bidx * K;
    const uint total_groups = blocks_per_row * 8;

    float partial = 0.0f;
    for (uint gidx = tid; gidx < total_groups; gidx += tg_size) {
        uint i = gidx >> 3;
        uint g = gidx & 7u;
        uint hpart = g >> 2;
        uint sub = g & 3u;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);

        uint ql_base = hpart * 64u + (sub & 1u) * 32u;
        uint qh_base = hpart * 32u;
        uint x_base = i * 256u + hpart * 128u + sub * 32u;
        uint sc_base = hpart * 8u + sub * 2u;
        uchar4 qh_shift = uchar4((uchar)(sub * 2u));
        uchar4 qmask = uchar4(0xFu);

        float4 acc = float4(0.0f);
        for (int l = 0; l < 32; l += 4) {
            uchar4 ql4 = *(device const uchar4*)(bl.ql + ql_base + l);
            uchar4 qh4 = *(device const uchar4*)(bl.qh + qh_base + l);
            uchar4 lo = (sub < 2u) ? (ql4 & qmask) : (ql4 >> uchar4(4u));
            uchar4 hi = ((qh4 >> qh_shift) & uchar4(3u)) << uchar4(4u);
            float4 qv = float4(lo | hi) - float4(32.0f);
            float s = d * float(bl.scales[sc_base + (uint)l / 16u]);
            acc += s * qv * *(device const float4*)(xb + x_base + l);
        }
        partial += acc.x + acc.y + acc.z + acc.w;
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

/* Row-tiled batched Q6_K GEMM for B<=16. One 256-thread threadgroup computes
 * 8 output rows; the current x[B][256] block is staged once and shared across
 * those rows. This is an opt-in experiment for FFN down_proj activation reuse. */
kernel void q6k_sgemv_rowtiled_batched_b16(
    device const block_q6_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint                     tgid        [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]])
{
    if (B < 1 || B > 16) return;
    const uint ROWS_TILE = 8;
    const uint blocks_per_row = K / 256;
    uint simd_id = tid >> 5;
    uint lane = tid & 31u;
    uint row = tgid * ROWS_TILE + simd_id;
    device const block_q6_K* row_blocks = W + (size_t)row * blocks_per_row;

    float acc[16];
    for (uint s = 0; s < 16; s++) acc[s] = 0.0f;

    threadgroup float xt[16 * 256];

    for (uint i = 0; i < blocks_per_row; i++) {
        uint base = i * 256u;
        for (uint idx = tid; idx < B * 256u; idx += 256u) {
            uint s = idx >> 8;
            uint l = idx & 255u;
            xt[s * 256u + l] = x[(size_t)s * K + base + l];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < N) {
            const device block_q6_K& bl = row_blocks[i];
            float d = float(bl.d);
            for (uint g = 0; g < 8; g++) {
                uint hpart = g >> 2;
                uint sub = g & 3u;
                uint ql_base = hpart * 64u + (sub & 1u) * 32u;
                uint qh_base = hpart * 32u;
                uint xoff = hpart * 128u + sub * 32u + lane;
                uint sc_base = hpart * 8u + sub * 2u + (lane >> 4);
                uchar ql = bl.ql[ql_base + lane];
                uchar qh = bl.qh[qh_base + lane];
                uchar lo = (sub < 2u) ? (ql & 0xFu) : (ql >> 4u);
                uchar hi = ((qh >> (sub * 2u)) & 3u) << 4u;
                float qv = float(lo | hi) - 32.0f;
                float coef = d * float(bl.scales[sc_base]) * qv;
                for (uint s = 0; s < B; s++) {
                    acc[s] += coef * xt[s * 256u + xoff];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < N) {
        for (uint s = 0; s < B; s++) {
            float v = simd_sum(acc[s]);
            if (lane == 0) y[(size_t)s * N + row] = v;
        }
    }
}

/* Row/stream tiled Q6_K down_proj experiment for B>=8. One 256-thread group
 * computes 8 output rows for an 8-stream tile, staging x[8][256] once per
 * block. This cuts weight rereads versus per-stream-pair v5_g2 at high B. */
kernel void q6k_sgemv_rowtiled_s8_batched(
    device const block_q6_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    constant uint&           N           [[buffer(4)]],
    constant uint&           B           [[buffer(5)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]])
{
    uint tid = tid2.x;
    const uint ROWS_TILE = 8;
    const uint STREAMS_TILE = 8;
    const uint blocks_per_row = K / 256;
    uint row_lane = tid >> 5;
    uint lane = tid & 31u;
    uint row = tgid.x * ROWS_TILE + row_lane;
    uint bbase = tgid.y * STREAMS_TILE;
    device const block_q6_K* row_blocks = W + (size_t)row * blocks_per_row;

    float acc[8];
    for (uint s = 0; s < STREAMS_TILE; s++) acc[s] = 0.0f;

    threadgroup float xt[8 * 256];

    for (uint i = 0; i < blocks_per_row; i++) {
        uint base = i * 256u;
        for (uint idx = tid; idx < STREAMS_TILE * 256u; idx += 256u) {
            uint s = idx >> 8;
            uint l = idx & 255u;
            uint bidx = bbase + s;
            xt[s * 256u + l] = (bidx < B) ? x[(size_t)bidx * K + base + l] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < N) {
            const device block_q6_K& bl = row_blocks[i];
            float d = float(bl.d);
            for (uint g = 0; g < 8; g++) {
                uint hpart = g >> 2;
                uint sub = g & 3u;
                uint ql_base = hpart * 64u + (sub & 1u) * 32u;
                uint qh_base = hpart * 32u;
                uint xoff = hpart * 128u + sub * 32u + lane;
                uint sc_base = hpart * 8u + sub * 2u + (lane >> 4);
                uchar ql = bl.ql[ql_base + lane];
                uchar qh = bl.qh[qh_base + lane];
                uchar lo = (sub < 2u) ? (ql & 0xFu) : (ql >> 4u);
                uchar hi = ((qh >> (sub * 2u)) & 3u) << 4u;
                float coef = d * float(bl.scales[sc_base]) * (float(lo | hi) - 32.0f);
                for (uint s = 0; s < STREAMS_TILE; s++) {
                    acc[s] += coef * xt[s * 256u + xoff];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < N) {
        for (uint s = 0; s < STREAMS_TILE; s++) {
            uint bidx = bbase + s;
            float v = simd_sum(acc[s]);
            if (lane == 0 && bidx < B) y[(size_t)bidx * N + row] = v;
        }
    }
}

kernel void q6k_sgemv_row_simdb_batched(
    device const block_q6_K* W           [[buffer(0)]],
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
    device const block_q6_K* row_blocks = W + row * blocks_per_row;
    device const float* xb = x + (size_t)stream * K;
    const uint total_half = blocks_per_row * 2;
    float partial = 0.0f;

    for (uint u = local_tid; u < total_half; u += 64u) {
        uint i = u >> 1;
        uint n = (u & 1) * 128;
        const device block_q6_K& bl = row_blocks[i];
        const float d = float(bl.d);
        const device uchar* ql = bl.ql + n / 2;
        const device uchar* qh = bl.qh + n / 4;
        const device char* sc = bl.scales + n / 16;
        uint base = i * 256 + n;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            float q1 = float(int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            float q2 = float(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            float q3 = float(int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            float q4 = float(int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            partial += d * (
                float(sc[is + 0]) * q1 * xb[base + l + 0] +
                float(sc[is + 2]) * q2 * xb[base + l + 32] +
                float(sc[is + 4]) * q3 * xb[base + l + 64] +
                float(sc[is + 6]) * q4 * xb[base + l + 96]);
        }
    }

    threadgroup float sdata[32];
    uint lane = tid & 31u;
    uint local_simd = local_tid >> 5;
    float sg = simd_sum(partial);
    if (lane == 0) sdata[stream * 2u + local_simd] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < B) {
        y[(size_t)tid * N + row] = sdata[tid * 2u] + sdata[tid * 2u + 1u];
    }
    (void)tg_size;
}

kernel void q6k_argmax_batched(
    device const block_q6_K* W           [[buffer(0)]],
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
        device const block_q6_K* row_blocks = W + (size_t)row * blocks_per_row;
        const uint total_half = blocks_per_row * 2;
        float acc = 0.0f;
        for (uint u = 0; u < total_half; u++) {
            uint i = u >> 1;
            uint n = (u & 1) * 128;
            const device block_q6_K& bl = row_blocks[i];
            const float d = float(bl.d);
            const device uchar* ql = bl.ql + n / 2;
            const device uchar* qh = bl.qh + n / 4;
            const device char* sc = bl.scales + n / 16;
            uint base = i * 256 + n;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                float q1 = float(int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                float q2 = float(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                float q3 = float(int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                float q4 = float(int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                acc += d * (
                    float(sc[is + 0]) * q1 * xb[base + l + 0] +
                    float(sc[is + 2]) * q2 * xb[base + l + 32] +
                    float(sc[is + 4]) * q3 * xb[base + l + 64] +
                    float(sc[is + 6]) * q4 * xb[base + l + 96]);
            }
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

kernel void q6k_top1_tiles_batched(
    device const block_q6_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            vals        [[buffer(2)]],
    device uint*             idxs        [[buffer(3)]],
    constant uint&           K           [[buffer(4)]],
    constant uint&           V           [[buffer(5)]],
    constant uint&           B           [[buffer(6)]],
    constant uint&           tile_rows   [[buffer(7)]],
    uint2                    tgid        [[threadgroup_position_in_grid]],
    uint2                    tid2        [[thread_position_in_threadgroup]],
    uint2                    tg2         [[threads_per_threadgroup]])
{
    uint tile = tgid.x;
    uint bidx = tgid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;
    uint nt = (V + tile_rows - 1) / tile_rows;
    if (bidx >= B || tile >= nt) return;
    uint row0 = tile * tile_rows;
    uint row1 = min(row0 + tile_rows, V);
    const uint blocks_per_row = K / 256;
    device const float* xb = x + (size_t)bidx * K;
    float best = -INFINITY;
    uint best_i = row0;

    for (uint row = row0 + tid; row < row1; row += tg_size) {
        device const block_q6_K* row_blocks = W + (size_t)row * blocks_per_row;
        const uint total_half = blocks_per_row * 2;
        float acc = 0.0f;
        for (uint u = 0; u < total_half; u++) {
            uint i = u >> 1;
            uint n = (u & 1) * 128;
            const device block_q6_K& bl = row_blocks[i];
            const float d = float(bl.d);
            const device uchar* ql = bl.ql + n / 2;
            const device uchar* qh = bl.qh + n / 4;
            const device char* sc = bl.scales + n / 16;
            uint base = i * 256 + n;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                float q1 = float(int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                float q2 = float(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                float q3 = float(int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                float q4 = float(int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                acc += d * (
                    float(sc[is + 0]) * q1 * xb[base + l + 0] +
                    float(sc[is + 2]) * q2 * xb[base + l + 32] +
                    float(sc[is + 4]) * q3 * xb[base + l + 64] +
                    float(sc[is + 6]) * q4 * xb[base + l + 96]);
            }
        }
        if (acc > best) {
            best = acc;
            best_i = row;
        }
    }

    threadgroup float tv[256];
    threadgroup uint ti[256];
    tv[tid] = best;
    ti[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && tv[tid + s] > tv[tid]) {
            tv[tid] = tv[tid + s];
            ti[tid] = ti[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        vals[(size_t)bidx * nt + tile] = tv[0];
        idxs[(size_t)bidx * nt + tile] = ti[0];
    }
}
