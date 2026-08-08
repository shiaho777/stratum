
#define _GNU_SOURCE
#include "stratum_q4k.h"
#include "stratum_q4k_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static block_q4_K* make_w(int N,int K){
    int nb=(N*K)/256; block_q4_K* w=malloc((size_t)nb*sizeof(block_q4_K));
    for(int i=0;i<nb;i++){ for(int j=0;j<(int)sizeof(w[i].qs);j++)((uint8_t*)w[i].qs)[j]=rand()&0xFF;
        for(int j=0;j<(int)sizeof(w[i].scales);j++)w[i].scales[j]=rand()&0xFF; w[i].d=0x2c00; w[i].dmin=0x2400; }
    return w;
}
static inline const block_q4_K* row_ptr(const block_q4_K* w,int K,int r){ return w+(size_t)r*(K/256); }

#if defined(__ARM_FEATURE_DOTPROD)

static void quantize_x_q8(const float* x, int K, int8_t* xq, float* xscale){
    int nb = K/32;
    for(int b=0;b<nb;b++){
        const float* xp = x + b*32;
        float amax=0; for(int i=0;i<32;i++){ float a=fabsf(xp[i]); if(a>amax)amax=a; }
        float scale = amax/127.0f; float inv = scale>0? 1.0f/scale : 0.0f;
        xscale[b]=scale;
        for(int i=0;i<32;i++){ int v=(int)lrintf(xp[i]*inv); if(v>127)v=127; if(v<-128)v=-128; xq[b*32+i]=(int8_t)v; }
    }
}

static float q4k_dot_row_sdot(const block_q4_K* row, int K,
                              const int8_t* xq, const float* xscale){
    int nb = K/256;
    double dot = 0.0;
    uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for(int i=0;i<nb;i++){
        const block_q4_K* b = row+i;
        float d = q4k_fp16_to_fp32(b->d), dmin = q4k_fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        int is=0;
        int blk32 = i*8;
        for(int j=0;j<256;j+=64){
            uint8_t sc1,m1,sc2,m2;
            q4k_get_scale_min(is+0,b->scales,&sc1,&m1);
            q4k_get_scale_min(is+1,b->scales,&sc2,&m2);

            uint8x16_t w0 = vld1q_u8(q);
            uint8x16_t w1 = vld1q_u8(q+16);
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(w0,mask4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(w1,mask4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(w0,4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(w1,4));

            const int8_t* xl = xq + (size_t)(blk32)*32;
            const int8_t* xh = xq + (size_t)(blk32+1)*32;
            int8x16_t xl0=vld1q_s8(xl), xl1=vld1q_s8(xl+16);
            int8x16_t xh0=vld1q_s8(xh), xh1=vld1q_s8(xh+16);

            int32x4_t accq_lo = vdupq_n_s32(0);
            accq_lo = vdotq_s32(accq_lo, lo0, xl0);
            accq_lo = vdotq_s32(accq_lo, lo1, xl1);
            int32x4_t accq_hi = vdupq_n_s32(0);
            accq_hi = vdotq_s32(accq_hi, hi0, xh0);
            accq_hi = vdotq_s32(accq_hi, hi1, xh1);

            int8x16_t ones = vdupq_n_s8(1);
            int32x4_t accs_lo = vdupq_n_s32(0);
            accs_lo = vdotq_s32(accs_lo, ones, xl0);
            accs_lo = vdotq_s32(accs_lo, ones, xl1);
            int32x4_t accs_hi = vdupq_n_s32(0);
            accs_hi = vdotq_s32(accs_hi, ones, xh0);
            accs_hi = vdotq_s32(accs_hi, ones, xh1);

            float qx_lo = (float)vaddvq_s32(accq_lo);
            float qx_hi = (float)vaddvq_s32(accq_hi);
            float sx_lo = (float)vaddvq_s32(accs_lo);
            float sx_hi = (float)vaddvq_s32(accs_hi);
            float scl_lo = xscale[blk32], scl_hi = xscale[blk32+1];
            dot += (double)(d*sc1*(qx_lo*scl_lo) - dmin*m1*(sx_lo*scl_lo));
            dot += (double)(d*sc2*(qx_hi*scl_hi) - dmin*m2*(sx_hi*scl_hi));
            q += 32; is += 2; blk32 += 2;
        }
    }
    return (float)dot;
}
#endif

int main(void){
    int K=5120, N=8192; srand(1);
    float* x=malloc(K*sizeof(float)); for(int i=0;i<K;i++)x[i]=0.02f*((i%23)-11);
    block_q4_K* w=make_w(N,K); float* y=malloc(N*sizeof(float));
    double bytes=(double)N*(K/256)*144;
    int R=200;

    volatile float sink=0;
    double t0=now();
    for(int r=0;r<R;r++){ for(int i=0;i<N;i++) y[i]=q4k_dot_row_neon(row_ptr(w,K,i),K,x); sink+=y[r%N]; }
    double dt=(now()-t0)/R; printf("fp32-widening  %.3f ms  %.1f GB/s (sink=%.1f)\n",dt*1e3,bytes/dt/1e9,sink);

#if defined(__ARM_FEATURE_DOTPROD)
    int8_t* xq=malloc(K); float* xs=malloc((K/32)*sizeof(float));

    sink=0;
    t0=now();
    for(int r=0;r<R;r++){
        quantize_x_q8(x,K,xq,xs);
        for(int i=0;i<N;i++) y[i]=q4k_dot_row_sdot(row_ptr(w,K,i),K,xq,xs);
        sink+=y[r%N];
    }
    dt=(now()-t0)/R; printf("sdot-int8      %.3f ms  %.1f GB/s (sink=%.1f)\n",dt*1e3,bytes/dt/1e9,sink);

    float* yref=malloc(N*sizeof(float));
    for(int i=0;i<N;i++) yref[i]=q4k_dot_row_neon(row_ptr(w,K,i),K,x);
    quantize_x_q8(x,K,xq,xs);
    double maxrel=0,sumref=0,sumabs=0,maxabs=0;
    for(int i=0;i<N;i++){ float a=q4k_dot_row_sdot(row_ptr(w,K,i),K,xq,xs);
        double e=fabs(a-yref[i]); double r=e/(fabs(yref[i])+1e-6);
        if(r>maxrel)maxrel=r; if(e>maxabs)maxabs=e; sumabs+=e; sumref+=fabs(yref[i]); }
    printf("max rel err: %.4f  max abs err: %.3f  mean abs err: %.4f  mean|ref|: %.3f  (aggregate rel %.4f)\n",
           maxrel,maxabs,sumabs/N,sumref/N, sumabs/sumref);
#else
    printf("DOTPROD not available\n");
#endif
    return 0;
}
