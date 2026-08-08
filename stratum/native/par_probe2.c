
#define _GNU_SOURCE
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static block_q4_K* make_w(int N,int K){
    int nb=(N*K)/256; block_q4_K* w=malloc((size_t)nb*sizeof(block_q4_K));
    for(int i=0;i<nb;i++){ for(int j=0;j<(int)sizeof(w[i].qs);j++)((uint8_t*)w[i].qs)[j]=rand()&0xFF;
        for(int j=0;j<(int)sizeof(w[i].scales);j++)w[i].scales[j]=rand()&0xFF; w[i].d=0x2400; w[i].dmin=0x1c00; }
    return w;
}
static inline const block_q4_K* row_ptr(const block_q4_K* w,int K,int r){ return w+(size_t)r*(K/256); }

static const block_q4_K* g_w; static const float* g_x; static float* g_y; static int g_N,g_K,g_nth;
typedef struct{int id;} targ;
static void* tworker(void* a){
    int id=((targ*)a)->id;
    int chunk=(g_N+g_nth-1)/g_nth; int s=id*chunk; int e=s+chunk; if(e>g_N)e=g_N;
    for(int i=s;i<e;i++) g_y[i]=q4k_dot_row_neon(row_ptr(g_w,g_K,i),g_K,g_x);
    return NULL;
}

int main(void){
    int K=5120; srand(1);
    float* x=malloc(K*sizeof(float)); for(int i=0;i<K;i++)x[i]=0.01f*(i%17)-0.05f;
    int N=17408; block_q4_K* w=make_w(N,K); float* y=malloc(N*sizeof(float));
    g_w=w; g_x=x; g_y=y; g_N=N; g_K=K;
    double bytes=(double)N*(K/256)*144;
    int R=100;

    double t0=now();
    for(int r=0;r<R;r++) dispatch_apply(N,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),
        ^(size_t i){ y[i]=q4k_dot_row_neon(row_ptr(w,K,(int)i),K,x); });
    double dt=(now()-t0)/R; printf("dispatch_apply(N)       %.3f ms  %.1f GB/s\n",dt*1e3,bytes/dt/1e9);

    for(int T=8;T<=14;T+=2){
        t0=now();
        for(int r=0;r<R;r++){
            dispatch_apply(T,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t t){
                int chunk=(N+T-1)/T; int s=(int)t*chunk; int e=s+chunk; if(e>N)e=N;
                for(int i=s;i<e;i++) y[i]=q4k_dot_row_neon(row_ptr(w,K,i),K,x);
            });
        }
        dt=(now()-t0)/R; printf("dispatch_apply(%2d chunks) %.3f ms  %.1f GB/s\n",T,dt*1e3,bytes/dt/1e9);
    }

    for(int T=8;T<=14;T+=2){
        g_nth=T;
        t0=now();
        for(int r=0;r<R;r++){
            pthread_t th[16]; targ ta[16];
            for(int i=0;i<T;i++){ta[i].id=i; pthread_create(&th[i],NULL,tworker,&ta[i]);}
            for(int i=0;i<T;i++) pthread_join(th[i],NULL);
        }
        dt=(now()-t0)/R; printf("pthreads(%2d)            %.3f ms  %.1f GB/s\n",T,dt*1e3,bytes/dt/1e9);
    }
    return 0;
}
