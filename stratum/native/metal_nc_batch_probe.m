// metal_nc_batch_probe.m
// Quantify the per-matmul wait cost in the NC (NoCopy) path.
// A: per-matmul wrap+encode+commit+wait  (current nc_sgemv2 behavior)
// B: all matmuls encoded into ONE command buffer, single wait
// Uses REAL Q2_K weights from the 27B GGUF (mmap'd, NoCopy).
//
// usage: ./metal_nc_batch_probe <model.gguf> [nmatmul]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include "stratum_gguf.h"
#include "stratum_q2k.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static Gguf g;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [nmatmul]\n", argv[0]); return 1; }
    if (gguf_open(argv[1], &g) != 0) { fprintf(stderr, "gguf open fail\n"); return 1; }
    const int NM = argc > 2 ? atoi(argv[2]) : 480;

    /* collect the largest Q2_K tensors as GEMV workloads */
    typedef struct { const GgufTensor* t; uint64_t bytes; } Job;
    Job jobs[512]; int njobs = 0;
    for (uint64_t i = 0; i < g.n_tensors && njobs < 512; i++) {
        const GgufTensor* t = &g.tensors[i];
        if ((int)t->type == 10) { jobs[njobs].t = t; jobs[njobs].bytes = t->nbytes; njobs++; }
    }
    if (njobs == 0) { fprintf(stderr, "no Q2_K tensors\n"); return 1; }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    /* load q2k_sgemv from the existing metallib */
    NSString* mlpath = nil;
    if (access("stratum_q4k.metallib", R_OK) == 0) mlpath = @"stratum_q4k.metallib";
    else if (access("native/stratum_q4k.metallib", R_OK) == 0) mlpath = @"native/stratum_q4k.metallib";
    if (!mlpath) { fprintf(stderr, "no metallib\n"); return 1; }
    id<MTLLibrary> lib = [dev newLibraryWithFile:mlpath error:&err];
    if (!lib) { fprintf(stderr, "lib: %s\n", err.description.UTF8String); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"q2k_sgemv_row"];
    if (!fn) { fprintf(stderr, "no q2k_sgemv_row\n"); return 1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso fail\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    /* one x / y pair (K=5120-ish largest) */
    const GgufTensor* t0 = jobs[0].t;
    int K = (int)t0->dims[0], N = (int)t0->dims[1];
    (void)N;
    if (K > 8192) K = 8192;
    float* x = malloc((size_t)K * sizeof(float));
    float* y = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = 0.01f * (i % 7) - 0.03f;
    id<MTLBuffer> xb = [dev newBufferWithBytes:x length:(size_t)K*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [dev newBufferWithLength:(size_t)N*4 options:MTLResourceStorageModeShared];
    uint32_t K_u = (uint32_t)K;

    /* --- A: per-matmul wait --- */
    double s0 = now_s();
    double sink = 0;
    for (int m = 0; m < NM; m++) {
        const GgufTensor* t = jobs[m % njobs].t;
        const void* wptr = g.mmap_base + t->offset;
        int n = (int)t->dims[1];
        id<MTLBuffer> wb = [dev newBufferWithBytesNoCopy:(void*)wptr length:(size_t)t->nbytes
                                options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:wb offset:0 atIndex:0];
        [enc setBuffer:xb offset:0 atIndex:1];
        [enc setBuffer:yb offset:0 atIndex:2];
        [enc setBytes:&K_u length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        sink += ((float*)[yb contents])[0];
    }
    double tA = now_s() - s0;
    printf("A per-matmul wait:  %d matmuls %8.1f ms  (%.2f ms/matmul)  sink=%f\n",
           NM, tA * 1e3, tA / NM * 1e3, sink);

    /* --- B: single command buffer, all matmuls, one wait --- */
    s0 = now_s();
    sink = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        for (int m = 0; m < NM; m++) {
            const GgufTensor* t = jobs[m % njobs].t;
            const void* wptr = g.mmap_base + t->offset;
            int n = (int)t->dims[1];
            id<MTLBuffer> wb = [dev newBufferWithBytesNoCopy:(void*)wptr length:(size_t)t->nbytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wb offset:0 atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:yb offset:0 atIndex:2];
            [enc setBytes:&K_u length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        sink += ((float*)[yb contents])[0];
    }
    double tB = now_s() - s0;
    printf("B one-buffer all:   %d matmuls %8.1f ms  (%.2f ms/matmul)  sink=%f\n",
           NM, tB * 1e3, tB / NM * 1e3, sink);
    printf("SPEEDUP: %.1fx\n", tA / tB);
    return 0;
}
