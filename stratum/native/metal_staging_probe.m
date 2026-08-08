
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

typedef struct { unsigned char ql[128]; unsigned char qh[64]; signed char scales[16]; unsigned short d; } block_q6_K;

int main(int argc, char** argv) {

    int N = (argc > 1) ? atoi(argv[1]) : 5120;
    int K = (argc > 2) ? atoi(argv[2]) : 5120;
    int blocks_per_row = K / 256;
    size_t wbytes = (size_t)N * blocks_per_row * sizeof(block_q6_K);
    fprintf(stderr, "Q6_K staging probe: N=%d K=%d  staging=%.1f MB\n",
            N, K, wbytes / 1e6);

    void* staging = NULL;
    if (posix_memalign(&staging, 16384, wbytes) != 0) { perror("memalign"); return 1; }
    srand(1);
    unsigned char* p = (unsigned char*)staging;
    for (size_t i = 0; i < wbytes; i++) p[i] = rand() & 0xFF;

    block_q6_K* blk = (block_q6_K*)staging;
    size_t nblk = (size_t)N * blocks_per_row;
    for (size_t i = 0; i < nblk; i++) blk[i].d = 0x3C00;

    float* x = malloc(sizeof(float) * K);
    for (int i = 0; i < K; i++) x[i] = (float)(rand() & 0xFFFF) / 32768.0f - 0.5f;
    float* y = malloc(sizeof(float) * N);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:
            [NSURL fileURLWithPath:@"native/stratum_q4k.metallib"] error:&err];
        if (!lib) { fprintf(stderr, "lib load fail: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"q6k_sgemv_row"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso fail: %s\n", [[err localizedDescription] UTF8String]); return 1; }

        double tw0 = now_sec();
        id<MTLBuffer> wbuf = [dev newBufferWithBytesNoCopy:staging length:wbytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        double tw1 = now_sec();
        if (!wbuf) { fprintf(stderr, "newBufferWithBytesNoCopy FAILED on host staging\n"); return 1; }
        fprintf(stderr, "wrap host staging buffer: %.4f ms (this is per-layer cost if we re-wrap)\n",
                (tw1 - tw0) * 1e3);

        id<MTLBuffer> xbuf = [dev newBufferWithLength:sizeof(float)*K options:MTLResourceStorageModeShared];
        id<MTLBuffer> ybuf = [dev newBufferWithLength:sizeof(float)*N options:MTLResourceStorageModeShared];
        memcpy([xbuf contents], x, sizeof(float)*K);
        uint32_t Ku = (uint32_t)K;

        {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf offset:0 atIndex:0];
            [enc setBuffer:xbuf offset:0 atIndex:1];
            [enc setBuffer:ybuf offset:0 atIndex:2];
            [enc setBytes:&Ku length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        }

        int reps = 50;
        double tA = now_sec();
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf offset:0 atIndex:0];
            [enc setBuffer:xbuf offset:0 atIndex:1];
            [enc setBuffer:ybuf offset:0 atIndex:2];
            [enc setBytes:&Ku length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        }
        double tB = now_sec();
        double gpu_per = (tB - tA) / reps;
        double gflop = 2.0 * N * K / 1e9;
        fprintf(stderr, "GPU Q6_K sgemv: %.4f ms/op  (%.1f effective GFLOP/s, %.1f GB-weights/s)\n",
                gpu_per*1e3, gflop/gpu_per, (wbytes/1e9)/gpu_per);
        fprintf(stderr, "  => one 27B-layer-worth of matmul (~250 MB) would take ~%.1f ms on GPU\n",
                gpu_per * (250e6 / (double)wbytes) * 1e3);
        fprintf(stderr, "  => 65 layers ~= %.2f s/token of GPU matmul (vs ~4.4s NEON)\n",
                gpu_per * (16.0e9 / (double)wbytes));
    }
    free(staging); free(x); free(y);
    return 0;
}
