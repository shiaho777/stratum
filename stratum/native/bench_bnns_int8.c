/* bench_bnns_int8.c — probe the AMX hardware ceiling via BNNS int8
 * fully-connected (BNNS dispatches to Apple's AMX matrix unit on
 * Apple Silicon for int8 math). Measures effective weight-read GB/s
 * for a decode-shaped GEMV (N=2048/6144, K=2048) and compares with
 * the hand-written SDOT kernel ceiling. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int try_combo(BNNSDataType xt, BNNSDataType wt, BNNSDataType yt, const char* tag,
                     int N, int K, int iters) {
    void* xp_ = NULL; (void)xp_;
    int8_t* w8 = malloc((size_t)N * K);
    float* wf = malloc(sizeof(float) * (size_t)N * K);
    int8_t* x8 = malloc(K);
    float* xf = malloc(sizeof(float) * K);
    float* y = malloc(sizeof(float) * N);
    for (int i = 0; i < N * K; i++) { w8[i] = (int8_t)(i % 7); wf[i] = (float)(i % 7); }
    for (int i = 0; i < K; i++) { x8[i] = (int8_t)(i % 5); xf[i] = (float)(i % 5); }
    xp_ = (xt == BNNSDataTypeInt8) ? (void*)x8 : (void*)xf;

    BNNSNDArrayDescriptor xd, yd, wd;
    memset(&xd, 0, sizeof xd); memset(&yd, 0, sizeof yd); memset(&wd, 0, sizeof wd);
    xd.data_type = xt; xd.size[0] = K; xd.stride[0] = 1; xd.data = xp_;
    xd.layout = BNNSDataLayoutVector;
    yd.data_type = yt; yd.size[0] = N; yd.stride[0] = 1; yd.data = y;
    yd.layout = BNNSDataLayoutVector;
    wd.data_type = wt; wd.size[0] = K; wd.size[1] = N; wd.stride[0] = 1; wd.stride[1] = K;
    wd.layout = BNNSDataLayoutRowMajorMatrix;
    wd.data = (wt == BNNSDataTypeInt8) ? (void*)w8 : (void*)wf;

    BNNSLayerParametersFullyConnected fc;
    memset(&fc, 0, sizeof fc);
    fc.i_desc = xd; fc.o_desc = yd; fc.w_desc = wd;
    fc.bias.data_type = BNNSDataTypeFloat32;   /* no bias: data stays NULL */
    fc.bias.size[0] = N; fc.bias.stride[0] = 1;
    fc.activation.function = BNNSActivationFunctionIdentity;

    BNNSFilter filter = BNNSFilterCreateLayerFullyConnected(&fc, NULL);
    if (!filter) { printf("%-28s: create FAILED\n", tag); return 1; }
    void *xp = (xt == BNNSDataTypeInt8) ? (void*)x8 : (void*)xf;
    int rc = BNNSFilterApply(filter, xp, y);
    printf("  apply rc=%d, y[0]=%.2f\n", rc, y[0]);
    if (rc != 0) { printf("%-28s: apply FAILED\n", tag); return 1; }
    double t0 = now_s();
    for (int i = 0; i < iters; i++)
        BNNSFilterApply(filter, xp, y);
    double dt = now_s() - t0;
    double gbs = (double)N * K * iters / dt / 1e9;
    {
        double exact = 0;
        for (int i = 0; i < K; i++) exact += (double)((int8_t)(i%7)) * (double)((int8_t)(i%5));
        printf("%-28s: %7.1f GB/s  y[0]=%.2f (expected %.0f) y[1]=%.2f y[2047]=%.2f\n",
               tag, gbs, y[0], exact, y[1], y[N-1]);
    }
    return 0;
}

int main(void) {
    const int N = 2048, K = 2048;
    int iters = 2000;
    try_combo(BNNSDataTypeInt8, BNNSDataTypeInt8, BNNSDataTypeFloat32, "K=2048 full row", N, 2048, iters);
    try_combo(BNNSDataTypeInt8, BNNSDataTypeInt8, BNNSDataTypeFloat32, "K=32 group",     N, 32,    iters);
    try_combo(BNNSDataTypeInt8, BNNSDataTypeInt8, BNNSDataTypeFloat32, "K=256 superblock", N, 256, iters);
    try_combo(BNNSDataTypeInt8, BNNSDataTypeInt8, BNNSDataTypeFloat32, "K=2048 N=6144",   6144, 2048, iters);
    return 0;
}
