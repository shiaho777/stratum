/* bench_w16_amx.c — measure the ENGINE's AMX w16 multiseq path at real dims.
 * Builds a synthetic w16 blob (valid layout, random values), a fake
 * GgufTensor, B f32 activations, then times st_linear_multix (type 43).
 * Also times the portable fallback and reports per-seq cost.
 * Correctness gate: FULL-matrix AMX-vs-fallback compare + argmax match. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include "stratum_linear.h"

StratumLinearState g_st;

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc, char** argv){
    int N = argc>1 ? atoi(argv[1]) : 3584;
    int K = argc>2 ? atoi(argv[2]) : 1024;
    int B = argc>3 ? atoi(argv[3]) : 64;
    int iters = argc>4 ? atoi(argv[4]) : 20;
    if (N%32||K%256){ fprintf(stderr,"need N%%32==0 && K%%256==0\n"); return 1; }
    int NG=K/32, NB=K/256;
    size_t blob = (size_t)N*K*2 + (size_t)N*NB*8 + (size_t)N*NG*2;
    uint8_t* buf = aligned_alloc(128, (blob+127)&~127ull);
    int16_t* w16 = (int16_t*)buf;
    float* dv = (float*)(buf + (size_t)N*K*2);
    float* dmv = dv + (size_t)N*NB;
    int16_t* m16 = (int16_t*)(dmv + (size_t)N*NB);   /* [rt][g][32] = m[rt*32+j][g] */
    srand(1);
    for (size_t i=0;i<(size_t)N*K;i++) w16[i] = (int16_t)(rand()%946);   /* n*sc <=945 */
    for (int i=0;i<N*NB;i++){ dv[i]=0.05f+0.001f*(i%13); dmv[i]=0.004f+0.0002f*(i%7); }
    for (int rt=0;rt<N/32;rt++) for (int g=0;g<NG;g++) for (int j=0;j<32;j++)
        m16[(size_t)rt*NG*32 + (size_t)g*32 + j]=(int16_t)(rand()%64);

    float** xs = malloc(sizeof(float*)*B);
    float** ys = malloc(sizeof(float*)*B);
    float** yr = malloc(sizeof(float*)*B);
    for(int b=0;b<B;b++){ xs[b]=malloc(K*4); ys[b]=calloc(N,4); yr[b]=calloc(N,4);
        for(int k=0;k<K;k++) xs[b][k]=(float)(rand()%41-20)/7.f; }

    GgufTensor w; memset(&w,0,sizeof w);
    w.type = GGML_TYPE_Q4K_W16; w.offset = 0; w.n_dims=2;
    w.dims[0]=K; w.dims[1]=N; w.nelem=(int64_t)N*K;
    g_st.mmap_base = buf; g_st.mmap_size = blob;
    g_st.amx_w16_off = 0; g_st.use_sdot = 1; g_st.nchunks = 1;

    if(!st_amx_available()){ printf("AMX unavailable\n"); return 1; }

    /* warm */
    st_linear_multix(&w,(const float*const*)xs,ys,B,N,K);
    double t0=now_us();
    for(int i=0;i<iters;i++) st_linear_multix(&w,(const float*const*)xs,ys,B,N,K);
    double us=(now_us()-t0)/iters;
    printf("AMX w16 multix: N=%d K=%d B=%d -> %.1f us/matmul  %.2f us/seq  %.1f GMAC/s\n",
           N,K,B,us,us/B,(double)N*B*K*iters/(us*iters/1e6)/1e9);

    /* fallback (portable, single timed call) */
    g_st.amx_w16_off = 1;
    t0=now_us();
    st_linear_multix(&w,(const float*const*)xs,yr,B,N,K);
    printf("fallback       : %.1f us/matmul (%.2f us/seq)\n", now_us()-t0, (now_us()-t0)/B);

    /* full-matrix check: AMX ys vs fallback yr — argmax per seq + max|d| */
    double md=0, se=0; long cnt=0; int amis=0;
    for(int b=0;b<B;b++){
        int a1=0,a2=0;
        for(int r=0;r<N;r++){
            double d=fabs((double)ys[b][r]-(double)yr[b][r]);
            if(d>md)md=d; se+=d*d; cnt++;
            if(ys[b][r]>ys[b][a1])a1=r; if(yr[b][r]>yr[b][a2])a2=r;
        }
        if(a1!=a2)amis++;
    }
    printf("full check: max|d|=%.5f rms=%.5f argmax_mismatch=%d/%d\n",
           md,sqrt(se/cnt),amis,B);
    return amis!=0;
}
