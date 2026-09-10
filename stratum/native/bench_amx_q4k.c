/* bench_amx_q4k.c — architecture probe: Q4_K matvec via
 * (A) the current SDOT kernel vs (B) lossless nibble->int8 unpack into
 * [group][row] transposed layout + per-32-column contiguous BNNS group
 * calls (AMX) + scale accumulation.
 * Runs on the REAL blk.0.attn_q tensor (2048x1024) of MiniCPM5-2B Q4_K_M.
 * Numerics: yB is checked against the SDOT kernel (same quantized x).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include "stratum_linear.h"

StratumLinearState g_st;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    const char* wfile = argc > 1 ? argv[1] : "/tmp/wq0.bin";
    const char* xfile = argc > 2 ? argv[2] : "/tmp/xn0.bin";
    int N = 2048, K = 1024;
    int nb = K / 256, ng = K / 32;

    FILE* f = fopen(wfile, "rb");
    size_t wbytes = (size_t)N * nb * sizeof(block_q4_K);
    block_q4_K* W = malloc(wbytes);
    if (fread(W, 1, wbytes, f) != wbytes) { fprintf(stderr, "short W\n"); return 1; }
    fclose(f);
    f = fopen(xfile, "rb");
    float* x = malloc(sizeof(float) * K);
    if (fread(x, 4, K, f) != K) { fprintf(stderr, "short x\n"); return 1; }
    fclose(f);

    /* ---------- A) current SDOT kernel (6 chunks like the engine) ---------- */
    GgufTensor t = {0};
    t.type = GGML_TYPE_Q4_K; t.nelem = (long long)N * K; t.offset = 0;
    g_st.mmap_base = (const uint8_t*)W;
    g_st.nchunks = 6; g_st.use_metal = 0;
    float* yA = malloc(sizeof(float) * N);
    int iters = 200;
    double t0 = now_s();
    for (int it = 0; it < iters; it++)
        st_linear_dispatch(&t, x, yA, N, K);
    double dtA = now_s() - t0;
    printf("A) SDOT kernel         : %6.1f GB/s  (%5.0f us/matmul)\n",
           (double)wbytes * iters / dtA / 1e9, dtA / iters * 1e6);

    /* ---------- shared prep ---------- */
    int8_t* Wu = malloc((size_t)N * K);            /* [group][row*32] transposed */
    int8_t* xq = malloc(K);
    float*   xs = malloc(sizeof(float) * ng);
    int32_t* xsum = malloc(sizeof(int) * ng);
    q4k_quantize_x_q8(x, K, xq, xs);
    for (int g = 0; g < ng; g++) {
        int s = 0;
        for (int i = 0; i < 32; i++) s += xq[g * 32 + i];
        xsum[g] = s;
    }
    /* per-(row,group) scales: scmat[r*ng+g] = (d*sc), mnmat[r*ng+g] = (dmin*m) */
    float* scmat = malloc(sizeof(float) * (size_t)N * ng);
    float* mnmat = malloc(sizeof(float) * (size_t)N * ng);
    for (int r = 0; r < N; r++) {
        const block_q4_K* row = W + (size_t)r * nb;
        for (int b = 0; b < nb; b++) {
            float d = q4k_fp16_to_fp32(row[b].d);
            float dmin = q4k_fp16_to_fp32(row[b].dmin);
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                q4k_get_scale_min(is + 0, row[b].scales, &sc1, &m1);
                q4k_get_scale_min(is + 1, row[b].scales, &sc2, &m2);
                int g0 = (b * 256 + j) / 32;
                scmat[(size_t)r * ng + g0]     = d * sc1;
                mnmat[(size_t)r * ng + g0]     = dmin * m1;
                scmat[(size_t)r * ng + g0 + 1] = d * sc2;
                mnmat[(size_t)r * ng + g0 + 1] = dmin * m2;
                is += 2;
            }
        }
    }

    /* B1: transposed unpack — group g's row-r chunk at Wu[g*N*32 + r*32] */
    double dtU = 0;
    {
        uint8x16_t mask = vdupq_n_u8(0x0F);
        t0 = now_s();
        for (int it = 0; it < iters; it++) {
            for (int r = 0; r < N; r++) {
                const block_q4_K* row = W + (size_t)r * nb;
                for (int b = 0; b < nb; b++) {
                    const uint8_t* q = row[b].qs;
                    for (int rep = 0; rep < 4; rep++) {
                        int gLo = b * 8 + rep * 2;        /* elems [rep*64, +32) */
                        int gHi = gLo + 1;               /* elems [rep*64+32, +64) */
                        int8_t* dstLo = Wu + (size_t)gLo * N * 32 + (size_t)r * 32;
                        int8_t* dstHi = Wu + (size_t)gHi * N * 32 + (size_t)r * 32;
                        uint8x16_t v0 = vld1q_u8(q);
                        uint8x16_t v1 = vld1q_u8(q + 16);
                        vst1q_u8((uint8_t*)dstLo,      vandq_u8(v0, mask));
                        vst1q_u8((uint8_t*)dstLo + 16, vandq_u8(v1, mask));
                        vst1q_u8((uint8_t*)dstHi,      vshrq_n_u8(v0, 4));
                        vst1q_u8((uint8_t*)dstHi + 16, vshrq_n_u8(v1, 4));
                        q += 32;
                    }
                }
            }
        }
        dtU = now_s() - t0;
    }
    printf("B1) NEON unpack (T)    : %6.1f GB/s  (%5.0f us/matmul)\n",
           (double)N * K * iters / dtU / 1e9, dtU / iters * 1e6);

    /* B2: BNNS filters — contiguous (N x 32) per group on Wu */
    BNNSFilter* filters = malloc(sizeof(BNNSFilter) * ng);
    float* partial = malloc(sizeof(float) * N);
    for (int g = 0; g < ng; g++) {
        BNNSNDArrayDescriptor xd, yd, wd;
        memset(&xd, 0, sizeof xd); memset(&yd, 0, sizeof yd); memset(&wd, 0, sizeof wd);
        xd.data_type = BNNSDataTypeInt8; xd.size[0] = 32; xd.stride[0] = 1;
        xd.data = xq + g * 32; xd.layout = BNNSDataLayoutVector;
        yd.data_type = BNNSDataTypeFloat32; yd.size[0] = N; yd.stride[0] = 1;
        yd.data = partial; yd.layout = BNNSDataLayoutVector;
        wd.data_type = BNNSDataTypeInt8; wd.size[0] = 32; wd.size[1] = N;
        wd.stride[0] = 1; wd.stride[1] = 32;
        wd.data = Wu + (size_t)g * N * 32;
        wd.layout = BNNSDataLayoutRowMajorMatrix;
        BNNSLayerParametersFullyConnected fc;
        memset(&fc, 0, sizeof fc);
        fc.i_desc = xd; fc.o_desc = yd; fc.w_desc = wd;
        fc.bias.data_type = BNNSDataTypeFloat32;
        fc.bias.size[0] = N; fc.bias.stride[0] = 1;
        fc.activation.function = BNNSActivationFunctionIdentity;
        filters[g] = BNNSFilterCreateLayerFullyConnected(&fc, NULL);
        if (!filters[g]) { printf("filter[%d] FAILED\n", g); return 1; }
    }

    /* B2+B3 fused: per group, one BNNS call + scale accumulation */
    float* yB = malloc(sizeof(float) * N);
    double dtBN = 0, dtAcc = 0;
    for (int it = 0; it < iters; it++) {
        memset(yB, 0, sizeof(float) * N);
        for (int g = 0; g < ng; g++) {
            t0 = now_s();
            BNNSFilterApply(filters[g], xq + g * 32, partial);
            dtBN += now_s() - t0;
            t0 = now_s();
            float xsg = xs[g], xsumg = (float)xsum[g];
            const float* scr = scmat + g;
            const float* mnr = mnmat + g;
            for (int r = 0; r < N; r++) {
                yB[r] += scr[(size_t)r * ng] * xsg * partial[r]
                       - mnr[(size_t)r * ng] * xsg * xsumg;
            }
            dtAcc += now_s() - t0;
        }
    }
    printf("B2) BNNS group calls   : %6.1f GB/s  (%5.0f us/matmul)\n",
           (double)N * K * iters / dtBN / 1e9, dtBN / iters * 1e6);
    printf("B3) scale accumulation :           (%5.0f us/matmul)\n", dtAcc / iters * 1e6);
    double dtB = dtU + dtBN + dtAcc;
    printf("B) TOTAL               : %6.1f GB/s  (%5.0f us/matmul)  vs A: %.2fx\n",
           (double)wbytes * iters / dtB / 1e9, dtB / iters * 1e6, dtA / dtB);

    /* numeric verification vs the SDOT kernel */
    double maxd = 0; int worst = -1;
    for (int r = 0; r < N; r++) {
        double d = fabs((double)yA[r] - yB[r]);
        if (d > maxd) { maxd = d; worst = r; }
    }
    printf("verify: max|yB - yA| = %.6f (row %d); yA[0]=%.4f yB[0]=%.4f\n",
           maxd, worst, yA[0], yB[0]);
    return 0;
}
