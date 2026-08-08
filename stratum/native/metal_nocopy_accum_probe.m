// metal_nocopy_accum_probe.m
// DECISIVE experiment: does per-window NoCopy (tessera pattern) accumulate wired
// memory across sequential GPU reads of the WHOLE model, or does it stay flat?
//
// tessera (M4 Pro) wraps each page-aligned payload in newBufferWithBytesNoCopy
// (MTLResourceStorageModeShared), keeps buffers alive, and reports private=5MiB.
// Stratum V54 observed 11GB wired on the whole-model NoCopy chunk. This probe
// isolates the variable: windowed NoCopy + release, sequential across the file,
// monitoring vm_stat wired AND task_vm_info internal (private).
//
// usage: ./metal_nocopy_accum_probe <file> [windowMB] [hold=0|1]
//   hold=0: release buffer after each window (tessera streaming pattern)
//   hold=1: keep ALL buffers alive (tessera non-streaming pattern)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double vm_wired_gb(void) {
    /* use host_statistics for wired */
    vm_statistics64_data_t vs;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &cnt);
    return (double)vs.wire_count * 4096.0 / 1e9;
}

static double internal_gb(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &cnt);
    return (double)info.internal * 4096.0 / 1e9;   /* private pages */
}

static double phys_footprint_gb(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &cnt);
    return (double)info.phys_footprint / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [windowMB] [hold=0|1]\n", argv[0]); return 1; }
    const size_t win_mb = argc > 2 ? (size_t)atoll(argv[2]) : 100;
    const int hold = argc > 3 ? atoi(argv[3]) : 0;
    const size_t win = win_mb << 20;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    off_t fsz = st.st_size;
    printf("file=%s size=%.2f GB  window=%zu MB  hold=%d\n",
           argv[1], (double)fsz / 1e9, win_mb, hold);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal\n"); return 1; }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "kernel void read_bw(const device uint* src [[buffer(0)]],\n"
         "                    device uint* dst [[buffer(1)]],\n"
         "                    const device uint* n [[buffer(2)]],\n"
         "                    uint tid [[thread_position_in_grid]]) {\n"
         "  uint acc = 0;\n"
         "  uint nn = n[0];\n"
         "  for (uint i = tid; i < nn; i += 8192) acc += src[i];\n"
         "  if (acc == 0xDEADBEEF) dst[0] = acc;\n"
         "}\n"
        options:nil error:&err];
    if (!lib) { fprintf(stderr, "lib fail: %s\n", err.description.UTF8String); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"read_bw"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso fail\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    uint32_t dummy = 0;
    id<MTLBuffer> dstbuf = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    memcpy([dstbuf contents], &dummy, 4);
    uint32_t ndummy = 0;
    id<MTLBuffer> nbuf = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    memcpy([nbuf contents], &ndummy, 4);

    double w0 = vm_wired_gb(), i0 = internal_gb(), f0 = phys_footprint_gb();
    printf("baseline: wired=%.3f GB  private=%.3f GB  footprint=%.3f GB\n", w0, i0, f0);

    const int nwin = (int)((fsz + (off_t)win - 1) / (off_t)win);
    const size_t pgsz = 4096;
    NSMutableArray* held = hold ? [NSMutableArray arrayWithCapacity:nwin] : nil;
    double t0 = now_s();
    double sum_wired = 0, sum_priv = 0;
    for (int i = 0; i < nwin; i++) {
        off_t off = (off_t)i * (off_t)win;
        size_t len = win;
        if (off + (off_t)len > fsz) len = (size_t)(fsz - off);
        /* mmap this window */
        void* base = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, off);
        if (base == MAP_FAILED) { perror("mmap"); break; }
        id<MTLBuffer> wbuf = [dev newBufferWithBytesNoCopy:base
                               length:len
                               options:MTLResourceStorageModeShared
                               deallocator:nil];
        if (!wbuf) { fprintf(stderr, "wrap fail at %d\n", i); munmap(base, len); break; }
        if (hold) [held addObject:wbuf];
        uint32_t n_u = (uint32_t)(len / 4);
        [nbuf contents]; memcpy([nbuf contents], &n_u, 4);
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:wbuf offset:0 atIndex:0];
        [enc setBuffer:dstbuf offset:0 atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(8192,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (!hold) { wbuf = nil; /* release */ munmap(base, len); }
        if (i % 5 == 4 || i == nwin - 1) {
            double w = vm_wired_gb(), p = internal_gb(), f = phys_footprint_gb();
            printf("  win %3d/%-3d  wired=%+.3f GB  private=%+.3f GB  footprint=%+.3f GB\n",
                   i + 1, nwin, w - w0, p - i0, f - f0);
            sum_wired = w - w0; sum_priv = p - i0;
        }
    }
    double t1 = now_s();
    double w1 = vm_wired_gb(), i1 = internal_gb(), f1 = phys_footprint_gb();
    printf("=== FINAL: wired %+.3f GB | private %+.3f GB | footprint %+.3f GB | %.2fs ===\n",
           w1 - w0, i1 - i0, f1 - f0, t1 - t0);
    printf("wired_delta_per_win_avg = %+.4f GB\n", (w1 - w0) / nwin);
    printf("VERDICT: %s\n",
           (w1 - w0) > 0.5 ? "WIRED ACCUMULATES (V54 confirmed)" :
           (i1 - i0) > 0.5 ? "PRIVATE ACCUMULATES (bad for multitenant)" :
           "NO ACCUMULATION (tessera pattern is safe)");
    return 0;
}
