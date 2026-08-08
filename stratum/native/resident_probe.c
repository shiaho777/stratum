
#define _GNU_SOURCE
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <dispatch/dispatch.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s model.gguf\n",argv[0]);return 1;}
    int fd=open(argv[1],O_RDONLY); if(fd<0){perror("open");return 1;}
    struct stat st; fstat(fd,&st); size_t fsize=st.st_size;
    const uint8_t* base=mmap(NULL,fsize,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){perror("mmap");return 1;}

    int K=5120;

    size_t region = 3ULL*1024*1024*1024;
    size_t rowbytes=(size_t)(K/256)*144;
    int N=(int)(region/rowbytes);
    region = (size_t)N*rowbytes;

    size_t hot_off = 1ULL*1024*1024*1024;
    size_t cold_off= 8ULL*1024*1024*1024;
    hot_off  -= hot_off % rowbytes;
    cold_off -= cold_off % rowbytes;

    const block_q4_K* hot =(const block_q4_K*)(base+hot_off);
    const block_q4_K* cold=(const block_q4_K*)(base+cold_off);

    float* x=malloc(K*sizeof(float)); for(int i=0;i<K;i++)x[i]=0.01f*(i%17)-0.05f;
    float* y=malloc(N*sizeof(float));
    double bytes=(double)N*rowbytes;

    if(mlock(hot,region)!=0) perror("mlock(hot)");
    volatile uint64_t sink=0; const volatile uint8_t* hp=(const volatile uint8_t*)hot;
    for(size_t o=0;o<region;o+=16384) sink+=hp[o];
    (void)sink;

    madvise((void*)cold, region, MADV_DONTNEED);

    int T=8;
    #define RUNMM(W) dispatch_apply(T,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t t){\
        int chunk=(N+T-1)/T,s=(int)t*chunk,e=s+chunk; if(e>N)e=N;\
        for(int r=s;r<e;r++) y[r]=q4k_dot_row_neon((W)+(size_t)r*(K/256),K,x); })

    RUNMM(hot);
    double t0=now(); RUNMM(hot); double dt_hot=now()-t0;

    t0=now(); RUNMM(cold); double dt_cold=now()-t0;

    t0=now(); RUNMM(cold); double dt_cold2=now()-t0;

    printf("region=%.2f GB, N=%d rows\n", bytes/1e9, N);
    printf("HOT  (mlock resident)   %.3f s  %.1f GB/s\n", dt_hot,  bytes/dt_hot/1e9);
    printf("COLD (1st, from SSD)    %.3f s  %.1f GB/s\n", dt_cold, bytes/dt_cold/1e9);
    printf("COLD (2nd, now cached)  %.3f s  %.1f GB/s\n", dt_cold2,bytes/dt_cold2/1e9);
    return 0;
}
