
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

static uint64_t g_state = 0x853c49e6748fea9bULL;
static uint64_t g_inc   = 0xda3e39cb94b95bdbULL;
static uint32_t pcg32(void){
    uint64_t old=g_state; g_state=old*6364136223846793005ULL+g_inc;
    uint32_t xs=(uint32_t)(((old>>18u)^old)>>27u); uint32_t rot=(uint32_t)(old>>59u);
    return (xs>>rot)|(xs<<((-(int)rot)&31));
}
static void seed(uint64_t s){ g_state=0; g_inc=(s<<1)|1; pcg32(); g_state+=0x853c49e6748fea9bULL; pcg32(); }
static float uniform(void){ return (float)(pcg32()>>8)*(1.0f/16777216.0f); }

#define N 8

static int sample_p(const float* p){
    float r=uniform(); double c=0;
    for(int i=0;i<N;i++){ c+=p[i]; if(r<c) return i; }
    return N-1;
}
static int sample_residual(const float* p,const float* q){
    float resid[N]; double sum=0;
    for(int i=0;i<N;i++){ float r=p[i]-q[i]; if(r<0)r=0; resid[i]=r; sum+=r; }
    if(sum<=0) return sample_p(p);
    float rr=uniform()*(float)sum; double c=0;
    for(int i=0;i<N;i++){ c+=resid[i]; if(rr<c) return i; }
    return N-1;
}

static int spec_emit_k1(const float* p,const float* q){
    int x=sample_p(q);
    float acc = (q[x]>0.0f)?(p[x]/q[x]):1.0f;
    if(acc>1.0f)acc=1.0f;
    if(uniform()<acc) return x;
    return sample_residual(p,q);
}

static void run_case(const char* name,const float* p,const float* q,int trials){
    long hp[N]={0}, hs[N]={0};
    seed(12345);
    for(int t=0;t<trials;t++) hp[sample_p(p)]++;
    seed(12345);
    for(int t=0;t<trials;t++) hs[spec_emit_k1(p,q)]++;
    double maxabs=0;
    printf("  case %-18s  tok:   p_hat    spec_hat   |diff|\n",name);
    for(int i=0;i<N;i++){
        double a=(double)hp[i]/trials, b=(double)hs[i]/trials, d=fabs(a-b);
        if(d>maxabs)maxabs=d;
        printf("                        %3d  %7.4f   %7.4f   %7.4f%s\n",
               i,a,b,d, d>0.02?"  <-- ":"");
    }
    printf("    -> max|p_hat - spec_hat| = %.4f  (%s)\n\n",
           maxabs, maxabs<0.02?"PASS (within MC noise)":"FAIL");
}

int main(void){
    int trials=2000000;

    float p[N]={0.40f,0.25f,0.15f,0.10f,0.05f,0.03f,0.015f,0.005f};

    float q1[N]; memcpy(q1,p,sizeof p);

    float q2[N]={0.05f,0.05f,0.10f,0.10f,0.20f,0.20f,0.15f,0.15f};

    float q3[N]={0.00f,0.40f,0.30f,0.20f,0.10f,0.00f,0.00f,0.00f};

    float q4[N]; for(int i=0;i<N;i++)q4[i]=1.0f/N;

    printf("Leviathan-Chen rejection sampling exactness (%d trials/case)\n",trials);
    printf("Claim: emitted-token distribution == target p, for ANY drafter q.\n\n");
    run_case("q==p",        p,q1,trials);
    run_case("q wrong",     p,q2,trials);
    run_case("q disjoint",  p,q3,trials);
    run_case("q uniform",   p,q4,trials);
    return 0;
}
