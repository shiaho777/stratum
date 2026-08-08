
#include <metal_stdlib>
using namespace metal;

struct block_q5_K {
    half     d;
    half     dmin;
    uchar    scales[12];
    uchar    qh[32];
    uchar    qs[128];
};

inline void unpack_scale_min_q5(int j, const device uchar* scales, thread uchar& sc_out, thread uchar& m_out) {
    if (j < 4) {
        sc_out = scales[j]   & 63;
        m_out  = scales[j+4] & 63;
    } else {
        sc_out = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
        m_out  = (scales[j+4] >>  4) | ((scales[j  ] >> 6) << 4);
    }
}

kernel void q5k_sgemv_row(
    device const block_q5_K* W           [[buffer(0)]],
    device const float*      x           [[buffer(1)]],
    device float*            y           [[buffer(2)]],
    constant uint&           K           [[buffer(3)]],
    uint                     row         [[threadgroup_position_in_grid]],
    uint                     tid         [[thread_position_in_threadgroup]],
    uint                     tg_size     [[threads_per_threadgroup]])
{
    const uint blocks_per_row = K / 256;
    device const block_q5_K* row_blocks = W + (uint)row * blocks_per_row;

    float partial = 0.0f;

    for (uint i = tid; i < blocks_per_row; i += tg_size) {
        const device block_q5_K& b = row_blocks[i];
        const float d    = float(b.d);
        const float dmin = float(b.dmin);

        for (uint c = 0; c < 4; c++) {
            const device uchar* qs_chunk = b.qs + c * 32;
            uint shift_lo = 2 * c;
            uint shift_hi = 2 * c + 1;

            uchar sc1, m1b;
            uchar sc2, m2b;
            unpack_scale_min_q5(int(2*c + 0), b.scales, sc1, m1b);
            unpack_scale_min_q5(int(2*c + 1), b.scales, sc2, m2b);

            float d_sc1   = d    * float(sc1);
            float dmin_m1 = dmin * float(m1b);
            float d_sc2   = d    * float(sc2);
            float dmin_m2 = dmin * float(m2b);

            float qx1 = 0.0f, sx1 = 0.0f;
            float qx2 = 0.0f, sx2 = 0.0f;
            uint base = i * 256 + c * 64;

            for (int l = 0; l < 32; l++) {
                uchar qbyte = qs_chunk[l];
                uchar hbit_lo = (b.qh[l] >> shift_lo) & 1;
                uchar hbit_hi = (b.qh[l] >> shift_hi) & 1;
                uint  q_lo = (qbyte & 0xF) | (hbit_lo ? 16u : 0u);
                uint  q_hi = (qbyte >>  4) | (hbit_hi ? 16u : 0u);

                float xv_lo = x[base + l];
                float xv_hi = x[base + l + 32];
                qx1 += float(q_lo) * xv_lo;
                sx1 += xv_lo;
                qx2 += float(q_hi) * xv_hi;
                sx2 += xv_hi;
            }
            partial += d_sc1 * qx1 - dmin_m1 * sx1
                     + d_sc2 * qx2 - dmin_m2 * sx2;
        }
    }

    threadgroup float sdata[64];
    sdata[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[row] = sdata[0];
}
