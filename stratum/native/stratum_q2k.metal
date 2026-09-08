#include <metal_stdlib>
using namespace metal;

/* Q2_K: 256 weights per super-block, 84 bytes.
 * File layout (matches stratum_q2k.h, verified bit-exact on CPU):
 *   scales[16] + qs[64] + d (fp16) + dmin (fp16)
 * qs: 2 bits/weight, 4 weights per byte. scales: 16 sub-blocks of 16
 * weights; low nibble = scale 0..15, high nibble = min 0..15.
 * Sub-block s (0..15): n = s>>3, j = (s>>1)&3, half = s&1.
 *   qs_pair = qs + 32n + 16*half  (n-stride 32: j reuses one 32-byte window), shift = 2j,
 *   scale byte = scales[8n + 2j + half],
 *   x offset = 128n + 32j + 16*half.
 * dot = d*sc * sum(q*x) - dmin*m * sum(x), q in {0,1,2,3}.
 */

struct block_q2_K {
    uchar  scales[16];
    uchar  qs[64];
    half   d;
    half   dmin;
};

kernel void q2k_sgemv_row(
    device const block_q2_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    uint                     row         [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]],
    uint                     tg_size     [[threads_per_threadgroup]])
{
    const uint blocks_per_row = K / 256;
    device const block_q2_K* row_blocks = W + (uint)row * blocks_per_row;
    const uint total_sb = blocks_per_row * 16;   /* 16-weight sub-blocks */

    float partial = 0.0f;
    for (uint sb = tid; sb < total_sb; sb += tg_size) {
        uint i = sb >> 4;            /* 256-block index */
        uint r = sb & 15;            /* sub-block 0..15 */
        uint n = r >> 3;
        uint j = (r >> 1) & 3;
        uint hb = r & 1;
        const device block_q2_K& b = row_blocks[i];
        const float d    = float(b.d);
        const float dmin = float(b.dmin);
        uchar scb = b.scales[8u*n + 2u*j + hb];
        float dl = d    * float(scb & 0xF);
        float ml = dmin * float(scb >> 4);
        const device uchar* qs_pair = b.qs + 32u*n + 16u*hb;
        uint shift = 2u*j;
        uint elem_offset = i*256u + 128u*n + 32u*j + 16u*hb;
        float qx = 0.0f, xs = 0.0f;
        uchar4 mask = uchar4(0x3);
        for (int l = 0; l < 16; l += 4) {
            uchar4 raw = *(device const uchar4*)(qs_pair + l);
            uchar4 wb  = (raw >> uchar4((uchar)shift)) & mask;
            float4 xv  = *(device const float4*)(x + elem_offset + l);
            qx += dot(float4(wb), xv);
            xs += xv.x + xv.y + xv.z + xv.w;
        }
        partial += dl * qx - ml * xs;
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

/* Batched Q2_K sgemv: one threadgroup per row, weights unpacked once and
 * reused across all BC streams. Mirrors the q4k/q6k batched kernels. */
#define DEFINE_Q2K_BATCH_KERNEL(BC) \
kernel void q2k_sgemv_row_batched_b##BC( \
    device const block_q2_K* W           [[buffer(0)]], \
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
    device const block_q2_K* row_blocks = W + (uint)row * blocks_per_row; \
    const uint total_sb = blocks_per_row * 16; \
    float partial[BC]; \
    for (uint s = 0; s < BC; s++) partial[s] = 0.0f; \
    for (uint sb = tid; sb < total_sb; sb += tg_size) { \
        uint i = sb >> 4; \
        uint r = sb & 15; \
        uint n = r >> 3; \
        uint j = (r >> 1) & 3; \
        uint hb = r & 1; \
        const device block_q2_K& b = row_blocks[i]; \
        const float d = float(b.d); \
        const float dmin = float(b.dmin); \
        uchar scb = b.scales[8u*n + 2u*j + hb]; \
        float dl = d * float(scb & 0xF); \
        float ml = dmin * float(scb >> 4); \
        const device uchar* qs_pair = b.qs + 32u*n + 16u*hb; \
        uint shift = 2u*j; \
        uint elem_offset = i*256u + 128u*n + 32u*j + 16u*hb; \
        uchar4 mask = uchar4(0x3); \
        float4 w4[4]; \
        for (int l = 0; l < 16; l += 4) { \
            uchar4 raw = *(device const uchar4*)(qs_pair + l); \
            w4[l >> 2] = float4((raw >> uchar4((uchar)shift)) & mask); \
        } \
        for (uint s = 0; s < BC; s++) { \
            device const float* xs_ptr = x + s * K + elem_offset; \
            float4 qx4 = float4(0.0f), xs4 = float4(0.0f); \
            for (int l = 0; l < 16; l += 4) { \
                float4 xv = *(device const float4*)(xs_ptr + l); \
                qx4 += w4[l >> 2] * xv; \
                xs4 += xv; \
            } \
            float qx = qx4.x + qx4.y + qx4.z + qx4.w; \
            float xs = xs4.x + xs4.y + xs4.z + xs4.w; \
            partial[s] += dl * qx - ml * xs; \
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

DEFINE_Q2K_BATCH_KERNEL(1)
DEFINE_Q2K_BATCH_KERNEL(2)
DEFINE_Q2K_BATCH_KERNEL(3)
DEFINE_Q2K_BATCH_KERNEL(4)
DEFINE_Q2K_BATCH_KERNEL(5)
DEFINE_Q2K_BATCH_KERNEL(6)
DEFINE_Q2K_BATCH_KERNEL(7)
DEFINE_Q2K_BATCH_KERNEL(8)
DEFINE_Q2K_BATCH_KERNEL(9)
DEFINE_Q2K_BATCH_KERNEL(10)
DEFINE_Q2K_BATCH_KERNEL(11)
DEFINE_Q2K_BATCH_KERNEL(12)
DEFINE_Q2K_BATCH_KERNEL(13)
DEFINE_Q2K_BATCH_KERNEL(14)
DEFINE_Q2K_BATCH_KERNEL(15)
DEFINE_Q2K_BATCH_KERNEL(16)
DEFINE_Q2K_BATCH_KERNEL(17)
DEFINE_Q2K_BATCH_KERNEL(18)
DEFINE_Q2K_BATCH_KERNEL(19)
DEFINE_Q2K_BATCH_KERNEL(20)
DEFINE_Q2K_BATCH_KERNEL(21)
DEFINE_Q2K_BATCH_KERNEL(22)
DEFINE_Q2K_BATCH_KERNEL(23)
DEFINE_Q2K_BATCH_KERNEL(24)
DEFINE_Q2K_BATCH_KERNEL(25)
DEFINE_Q2K_BATCH_KERNEL(26)
DEFINE_Q2K_BATCH_KERNEL(27)
DEFINE_Q2K_BATCH_KERNEL(28)
DEFINE_Q2K_BATCH_KERNEL(29)
DEFINE_Q2K_BATCH_KERNEL(30)
DEFINE_Q2K_BATCH_KERNEL(31)
DEFINE_Q2K_BATCH_KERNEL(32)
#undef DEFINE_Q2K_BATCH_KERNEL

/* q2k_sgemv_row_coalesced32 (2026-09-07, opt-in via STRATUM_Q2K_COAL=1)
 * Byte-optimal coalesced Q2_K GEMV: 8 threads per row, 32 rows/tg (256 thr).
 * Thread t8 covers (n=t8>>2, hb=(t8>>1)&1, hsel=t8&1): one 8-byte qs window
 * read ONCE per 256-block, all four 2-bit shift groups extracted in-thread —
 * kills the 4x qs re-read of the per-row kernel. Butterfly reduce over the
 * 8-lane row group (rows 8-aligned within each 32-lane simd).
 * Micro-bench (metal_q2k_coalesced_probe, N=51200 K=5120, M4 Pro):
 *   q2k_sgemv_row 51-73 GB/s -> coalesced32 90-107 GB/s (1.46-1.77x),
 *   max|d| vs CPU scalar reference 1.2e-3, 0 bad rows.
 * Tail-safe: rows past N_total compute on a clamped row and are not written;
 * all lanes still execute the shuffle (no divergent-exit UB). */
kernel void q2k_sgemv_row_coalesced32(
    device const block_q2_K* W           [[buffer(0)]],
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
    const uint local_row = tid >> 3;    /* 0..31 */
    const uint t8        = tid & 7;     /* 0..7  */
    const uint row = tgid * 32u + local_row;
    const uint rr  = min(row, N_total - 1u);   /* clamp: OOB rows read row N-1 */
    device const block_q2_K* row_blocks = W + (uint)rr * blocks_per_row;

    const uint n    = t8 >> 2;
    const uint hb   = (t8 >> 1) & 1;
    const uint hsel = t8 & 1;

    float partial = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; blk++) {
        const device block_q2_K& b = row_blocks[blk];
        const float d    = float(b.d);
        const float dmin = float(b.dmin);
        const device uchar* qp = b.qs + 32u*n + 16u*hb + hsel*8u;
        const uint xbase = blk*256u + 128u*n + 16u*hb + hsel*8u;

        float qx[4], xs[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) { qx[j] = 0.0f; xs[j] = 0.0f; }

        #pragma unroll
        for (int l4 = 0; l4 < 2; l4++) {
            uchar4 raw = *(device const uchar4*)(qp + 4u*l4);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uchar4 w = (raw >> uchar4((uchar)(2u*j))) & uchar4(0x3);
                float4 xv = *(device const float4*)(x + xbase + 32u*j + 4u*l4);
                qx[j] += dot(float4(w), xv);
                xs[j] += xv.x + xv.y + xv.z + xv.w;
            }
        }

        const uint sbase = 8u*n + hb;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uchar scb = b.scales[sbase + 2u*j];
            const float dl = d    * float(scb & 0xF);
            const float ml = dmin * float(scb >> 4);
            partial += dl * qx[j] - ml * xs[j];
        }
    }

    /* 8-lane butterfly reduction (rows are 8-aligned in each simd) */
    float tot = partial;
    tot += simd_shuffle_xor(tot, 4);
    tot += simd_shuffle_xor(tot, 2);
    tot += simd_shuffle_xor(tot, 1);
    if (t8 == 0 && row < N_total) y[row] = tot;
}
