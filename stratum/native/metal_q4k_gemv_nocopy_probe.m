// metal_q4k_gemv_nocopy_probe.m
// 真实 Q4_K GEMV kernel (q4k_sgemv_row_coalesced) 在 NoCopy mmap 直读下的带宽
// 对比同 tensor 的 CPU 14 核 multix 路径
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach.h>
#import <sys/mman.h>
#import <fcntl.h>
#import <unistd.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <time.h>
#import <dispatch/dispatch.h>
#include "stratum_gguf.h"
#include "stratum_q2k.h"
#include "stratum_q2k_neon.h"

static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static double wired_gb(void){
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t v;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &cnt) != KERN_SUCCESS) return -1;
    return (double)v.wire_count * vm_kernel_page_size / 1e9;
}

int main(int argc, char** argv){
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) { fprintf(stderr, "gguf_open fail\n"); return 1; }
    printf("tensors=%llu file=%.1f GB\n", (unsigned long long)g.n_tensors, g.mmap_size/1e9);

    /* 找最大的 Q2_K ffn_down tensor（FFN 是 decode 主力，占模型大头） */
    const GgufTensor* best = NULL;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        if ((GgmlType)t->type != GGML_TYPE_Q2_K) continue;
        if (!strstr(t->name, "ffn_down")) continue;
        if (!best || (t->dims[1] * t->dims[0] > best->dims[1] * best->dims[0])) best = t;
    }
    if (!best) { fprintf(stderr, "no Q2_K ffn_down found\n"); return 1; }
    int K = (int)best->dims[0], N = (int)best->dims[1];
    int nb = K / 256;
    size_t wbytes = (size_t)N * nb * sizeof(block_q2_K);
    printf("tensor %s [%d,%d] type=Q2_K  wt=%.1f MB\n", best->name, K, N, wbytes/1e6);

    /* mmap */
    int fd = open(argv[1], O_RDONLY);
    void* base = mmap(NULL, best->offset + wbytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    madvise((char*)base + best->offset, wbytes, MADV_WILLNEED);
    usleep(200000);
    const void* wt = (const char*)base + best->offset;

    printf("wired=%.3f GB\n", wired_gb());

    /* CPU 14 核 multix 路径（与引擎相同 kernel） */
    {
        int B = 1;
        float* x = malloc(sizeof(float)*K); for(int i=0;i<K;i++) x[i]=0.01f*(i%7)-0.03f;
        float* y = malloc(sizeof(float)*N);
        double best = 1e30;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            dispatch_apply(14, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^(size_t tid){
                int per = (N + 13) / 14, s = (int)tid*per, e = s+per; if(e>N) e=N;
                for (int i = s; i < e; i++) y[i] = q2k_dot_row_neon((const block_q2_K*)wt + (size_t)i*nb, K, x);
            });
            double t = now_s()-t0; if (t < best) best = t;
        }
        printf("CPU 14-thread q2k GEMV:     %6.1f GB/s   (%6.2f ms)\n", wbytes/best/1e9, best*1e3);
        free(x); free(y);
    }

    /* GPU NoCopy */
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithFile:@"stratum_q4k.metallib" error:&err];
    if (!lib) { fprintf(stderr, "metallib load fail: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"q2k_sgemv_row"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso fail: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    {
        @autoreleasepool {
            id<MTLBuffer> wbuf = [dev newBufferWithBytesNoCopy:(void*)wt length:wbytes
                                                       options:MTLResourceStorageModeShared deallocator:nil];
            uint32_t x_bytes = (uint32_t)(K*sizeof(float)), y_bytes = (uint32_t)(N*sizeof(float));
            id<MTLBuffer> xbuf = [dev newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> ybuf = [dev newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
            float* xp = (float*)[xbuf contents];
            for (int i=0;i<K;i++) xp[i] = 0.01f*(i%7)-0.03f;
            uint32_t K_u = (uint32_t)K, N_u = (uint32_t)N;

            printf("wired after NoCopy create: %.3f GB\n", wired_gb());
            double best = 1e30;
            for (int r = 0; r < 5; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:wbuf offset:0 atIndex:0];
                [enc setBuffer:xbuf offset:0 atIndex:1];
                [enc setBuffer:ybuf offset:0 atIndex:2];
                [enc setBytes:&K_u length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                     threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding];
                [cb commit];
                double t0 = now_s();
                [cb waitUntilCompleted];
                double t = now_s()-t0; if (t < best) best = t;
            }
            printf("GPU NoCopy q2k GEMV:       %6.1f GB/s   (%6.2f ms)   wired=%.3f GB\n",
                   wbytes/best/1e9, best*1e3, wired_gb());

            /* 校验 GPU vs CPU 输出一致性（bit-exact 粗查） */
            float* yg = (float*)[ybuf contents];
            float* yc = malloc(sizeof(float)*N);
            for (int i = 0; i < N; i++) yc[i] = q2k_dot_row_neon((const block_q2_K*)wt + (size_t)i*nb, K, xp);
            int bad = 0; float maxd = 0;
            for (int i = 0; i < N; i++) { float d = fabsf(yg[i]-yc[i]); if (d > maxd) maxd = d; if (d > 1e-3f) bad++; }
            printf("GPU-vs-CPU: maxdiff=%.3e  bad_rows(>1e-3)=%d / %d\n", maxd, bad, N);
            free(yc);
        }
    }
    printf("wired after release: %.3f GB\n", wired_gb());
    munmap(base, best->offset + wbytes);
    close(fd);
    return 0;
}
