
#define _GNU_SOURCE
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static block_q4_K* make_w(int N,int K){
    int nb = (N*K)/256;
    block_q4_K* w = malloc((size_t)nb*sizeof(block_q4_K));
    for(int i=0;i<nb;i++){
        for(int j=0;j<(int)sizeof(w[i].qs);j++) ((uint8_t*)w[i].qs)[j]=rand()&0xFF;
        for(int j=0;j<(int)sizeof(w[i].scales);j++) w[i].scales[j]=rand()&0xFF;
        w[i].d = 0x2400; w[i].dmin = 0x1c00;
    }
    return w;
}

static inline const block_q4_K* row_ptr(const block_q4_K* w,int K,int r){
    return w + (size_t)r*(K/256);
}

int main(int argc,char**argv){
    int K = 5120;
    srand(1);
    float* x = malloc(K*sizeof(float));
    for(int i=0;i<K;i++) x[i]=0.01f*(i%17)-0.05f;

    int ncpu=0; size_t l=sizeof(ncpu); sysctlbyname("hw.physicalcpu",&ncpu,&l,NULL,0);
    printf("physical cores: %d\n\n", ncpu);

    int sizes[] = {48, 1024, 5120, 17408, 248320};
    const char* names[] = {"ssm_beta(48)","attn_v(1024)","attn_o(5120)","ffn(17408)","lm_head(248320)"};
    int reps[]  = {2000, 500, 200, 80, 6};

    for(int si=0; si<5; si++){
        int N=sizes[si];
        block_q4_K* w = make_w(N,K);
        float* y = malloc(N*sizeof(float));
        int R = reps[si];

        double t0=now();
        for(int r=0;r<R;r++){
            dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),
                ^(size_t i){ y[i]=q4k_dot_row_neon(row_ptr(w,K,(int)i),K,x); });
        }
        double dt_da=(now()-t0)/R;

        t0=now();
        for(int r=0;r<(R<20?R:20);r++){
            for(int i=0;i<N;i++) y[i]=q4k_dot_row_neon(row_ptr(w,K,i),K,x);
        }
        double dt_st=(now()-t0)/(R<20?R:20);

        double bytes = (double)N*(K/256)*144;
        printf("%-18s N=%-7d  1thread %.3f ms (%.1f GB/s)  dispatch_apply %.3f ms (%.1f GB/s)  speedup %.2fx\n",
               names[si], N, dt_st*1e3, bytes/dt_st/1e9, dt_da*1e3, bytes/dt_da/1e9, dt_st/dt_da);
        free(w); free(y);
    }
    return 0;
}
