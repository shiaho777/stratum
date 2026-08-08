// metal_nocopy_window_probe.m
// 实证"GPU 零拷贝窗口"可行性：
//   1) newBufferWithBytesNoCopy 指向 mmap 页 → 创建/访问时 wired 多少
//   2) 释放 buffer 后 wired 是否回落（页恢复 reclaimable）
//   3) GPU 直读 mmap 带宽 vs CPU 直读带宽
// 用法: ./metal_nocopy_window_probe <model.gguf> [windowMB]
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

static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static double wired_gb(void){
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t v;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &cnt) != KERN_SUCCESS) return -1;
    return (double)v.wire_count * vm_kernel_page_size / 1e9;
}
static double anon_gb(void){
    struct mach_task_basic_info info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) != KERN_SUCCESS) return -1;
    return info.resident_size / 1e9;
}

int main(int argc, char** argv){
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [windowMB]\n", argv[0]); return 1; }
    const size_t win_mb = argc > 2 ? (size_t)atoll(argv[2]) : 256;
    const size_t win = win_mb << 20;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    off_t fsz = lseek(fd, 0, SEEK_END);
    if (fsz < (off_t)(win * 2)) { fprintf(stderr, "file too small\n"); return 1; }
    const size_t off = (size_t)fsz / 4;              /* 中段偏移，避免文件头 */

    void* base = mmap(NULL, off + win, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(base + off, win, MADV_WILLNEED);         /* 预取到 page cache */
    usleep(200000);

    printf("=== 0. baseline (file %lld MB, window %zu MB @ off %zu MB) ===\n",
           (long long)(fsz >> 20), win_mb, off >> 20);
    printf("wired=%.3f GB  anon_rss=%.3f GB\n", wired_gb(), anon_gb());

    /* CPU 直读带宽 */
    {
        volatile uint64_t sink = 0;
        double best = 1e30;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            const uint32_t* p = (const uint32_t*)((char*)base + off);
            uint64_t acc = 0;
            for (size_t i = 0; i < win / 4; i++) acc += p[i];
            sink += acc;
            double t = now_s() - t0;
            if (t < best) best = t;
        }
        printf("CPU  seq-read bandwidth: %8.1f GB/s   (sink=%llu)\n", win / best / 1e9, (unsigned long long)sink);
    }

    printf("wired=%.3f GB  anon_rss=%.3f GB\n", wired_gb(), anon_gb());

    /* 14 线程 CPU 直读 */
    {
        const uint32_t* src = (const uint32_t*)((char*)base + off);
        size_t per = win / 4 / 14;
        double best = 1e30;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            dispatch_apply(14, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t id){
                volatile uint64_t a0 = 0;
                const uint32_t* p = src + id * per;
                for (size_t i = 0; i < per; i++) a0 += p[i];
                (void)a0;
            });
            double t = now_s() - t0;
            if (t < best) best = t;
        }
        printf("CPU  14-thread read bandwidth: %6.1f GB/s\n", win / best / 1e9);
    }
    printf("wired=%.3f GB  anon_rss=%.3f GB\n", wired_gb(), anon_gb());

    /* GPU 零拷贝窗口 */
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s\n", dev.name.UTF8String);

    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "kernel void read_bw(const device uint* src [[buffer(0)]],\n"
         "                    device uint* dst [[buffer(1)]],\n"
         "                    constant uint& n [[buffer(2)]],\n"
         "                    uint tid [[thread_position_in_grid]]) {\n"
         "  uint acc = 0;\n"
         "  for (uint i = tid; i < n; i += 4096) acc += src[i];\n"
         "  if (acc == 0xDEADBEEF) dst[0] = acc;\n"
         "}\n"
        options:nil error:&err];
    if (!lib) { fprintf(stderr, "kernel compile fail: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"read_bw"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso fail\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    const uint32_t n_u32 = (uint32_t)(win / 4);
    {
        @autoreleasepool {
            id<MTLBuffer> wbuf = [dev newBufferWithBytesNoCopy:(void*)((char*)base + off)
                                                        length:win
                                                       options:MTLResourceStorageModeShared
                                                       deallocator:nil];
            printf("=== 1. after NoCopy buffer CREATE (win %zu MB) ===\n", win_mb);
            printf("wired=%.3f GB  anon_rss=%.3f GB   (delta wired=%.3f)\n",
                   wired_gb(), anon_gb(), wired_gb() - 0);

            /* GPU 读 3 轮计时 */
            double best = 1e30;
            for (int r = 0; r < 3; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:wbuf offset:0 atIndex:0];
                [enc setBuffer:wbuf offset:0 atIndex:1];
                [enc setBytes:&n_u32 length:4 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(4096, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                double t0 = now_s();
                [cb waitUntilCompleted];
                double t = now_s() - t0;
                if (t < best) best = t;
            }
            printf("=== 2. GPU read mmap (NoCopy): %6.1f GB/s ===\n", win / best / 1e9);
            printf("wired=%.3f GB  anon_rss=%.3f GB\n", wired_gb(), anon_gb());

            wbuf = nil;
        }
    }
    printf("=== 3. after NoCopy buffer RELEASE ===\n");
    printf("wired=%.3f GB  anon_rss=%.3f GB\n", wired_gb(), anon_gb());
    usleep(200000);
    printf("wired after 200ms settle: %.3f GB\n", wired_gb());

    munmap(base, off + win);
    close(fd);
    return 0;
}
