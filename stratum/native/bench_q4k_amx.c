/* bench_q4k_amx.c — differential test for the runtime Q4_K->AMX path.
 * Builds a synthetic Q4_K tensor (valid fp16 d/dmin, random scales/qs),
 * runs st_linear_multix (type Q4_K, B>=16 -> AMX unpack path), and compares
 * against the single-stream Q4_K reference (st_linear_q4k). */
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

static uint16_t f2h(float f){ __fp16 h=(__fp16)f; uint16_t u; memcpy(&u,&h,2); return u; }

int main(int argc, char** argv){
    int N = argc>1 ? atoi(argv[1]) : 1024;
    int K = argc>2 ? atoi(argv[2]) : 1024;
    int B = argc>3 ? atoi(argv[3]) : 32;
    if (N%32||K%256){ fprintf(stderr,"need N%%32==0 && K%%256==0\n"); return 1; }
    int nb=K/256;
    size_t bytes=(size_t)N*nb*144;
    uint8_t* buf = aligned_alloc(128, (bytes+127)&~127ull);
    srand(7);
    for (int64_t r=0;r<N;r++) for(int i=0;i<nb;i++){
        block_q4_K* b=(block_q4_K*)(buf+(size_t)r*nb*144+(size_t)i*144);
        b->d=f2h(0.03f+0.001f*(r%11)); b->dmin=f2h(0.002f+0.0001f*(r%5));
        for(int j=0;j<12;j++) b->scales[j]=rand()&0x3F;
        for(int j=0;j<128;j++) b->qs[j]=rand()&0xFF;
    }
    float** xs=malloc(sizeof(float*)*B); float** ys=malloc(sizeof(float*)*B);
    float** yr=malloc(sizeof(float*)*B);
    for(int b=0;b<B;b++){ xs[b]=malloc(K*4); ys[b]=calloc(N,4); yr[b]=calloc(N,4);
        for(int k=0;k<K;k++) xs[b][k]=(float)(rand()%41-20)/7.f; }

    GgufTensor w; memset(&w,0,sizeof w);
    w.type=GGML_TYPE_Q4_K; w.offset=0; w.n_dims=2;
    w.dims[0]=K; w.dims[1]=N; w.nelem=(int64_t)N*K;
    g_st.mmap_base=buf; g_st.mmap_size=bytes;
    g_st.amx_w16_off=0; g_st.use_sdot=1; g_st.nchunks=1;
    setenv("STRATUM_AMX_Q4K","1",0);   /* path under test is opt-in */

    st_linear_multix(&w,(const float* const*)xs,ys,B,N,K);
    for(int b=0;b<B;b++) st_linear_q4k(&w,xs[b],yr[b],N,K);

    double mx=0,rms=0; int bad=0;
    for(int b=0;b<B;b++){
        int am=0,rm=0;
        for(int r=0;r<N;r++){ double d=fabs(ys[b][r]-yr[b][r]); mx=fmax(mx,d); rms+=d*d;
            if(ys[b][r]>ys[b][am])am=r; if(yr[b][r]>yr[b][rm])rm=r; }
        if(am!=rm)bad++;
    }
    printf("B=%d N=%d K=%d  max|d|=%.5f rms=%.5f argmax_mismatch=%d/%d\n",
           B,N,K,mx,sqrt(rms/((double)B*N)),bad,B);
    return bad?1:0;
}
