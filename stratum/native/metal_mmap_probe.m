
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "stratum_gguf.h"
#include "stratum_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mach/mach.h>

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static double rss_gb(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return -1;
    return info.resident_size / (1024.0*1024.0*1024.0);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) return 1;
    fprintf(stderr, "opened, RSS=%.2f GB\n", rss_gb());

    char ml[1024]; snprintf(ml, sizeof ml, "native/stratum_q4k.metallib");
    if (stratum_metal_init(ml, g.mmap_base, g.mmap_size) != 0) { fprintf(stderr,"metal init fail\n"); return 1; }
    fprintf(stderr, "after metal chunk registration, RSS=%.2f GB\n", rss_gb());

    float* x = NULL; float* y = NULL;
    int done = 0;
    for (uint64_t i = 0; i < g.n_tensors && done < 6; i++) {
        const GgufTensor* t = &g.tensors[i];
        if (strstr(t->name, "ffn_down") == NULL) continue;
        if ((GgmlType)t->type != GGML_TYPE_Q6_K) continue;
        int K = (int)t->dims[0], N = (int)t->dims[1];
        if (!x) { x = malloc(sizeof(float)*K); for (int j=0;j<K;j++) x[j]=0.01f*(j%7); }
        if (!y) y = malloc(sizeof(float)*N);
        double a = now_sec();
        int rc = stratum_metal_q6k_sgemv(t->offset, x, y, N, K);
        double b = now_sec();
        fprintf(stderr, "  %s [%d,%d] rc=%d %.1fms  RSS=%.2f GB\n",
                t->name, K, N, rc, (b-a)*1e3, rss_gb());
        done++;
    }
    fprintf(stderr, "peak RSS=%.2f GB (model on disk = %.1f GB)\n",
            rss_gb(), g.mmap_size/1e9);
    stratum_metal_shutdown();
    gguf_close(&g);
    return 0;
}
