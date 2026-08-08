
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

#define LAYER_BYTES (256ull*1024*1024)
#define NBUF 3
#define NSEG 4
static size_t g_pg;
static int    g_fd_nocache;
static uint8_t* g_ring[NBUF];

static int    g_K = 5120;
static float* g_x;
static float* g_y;
static double neon_over(const uint8_t* data, size_t len){
    size_t rowbytes = (size_t)(g_K/256)*144;
    int N = (int)(len/rowbytes);
    int T = 10;
    double t0 = now();
    dispatch_apply(T, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^(size_t t){
        int chunk=(N+T-1)/T, s=(int)t*chunk, e=s+chunk; if(e>N)e=N;
        for(int r=s;r<e;r++)
            g_y[r] = q4k_dot_row_neon((const block_q4_K*)data + (size_t)r*(g_K/256), g_K, g_x);
    });
    double dt = now()-t0;
    return (double)N*rowbytes/dt/1e9;
}

static void pread_layer(int fd, uint8_t* buf, off_t off, size_t len){
    __block int ok=1;
    size_t seg=(len+NSEG-1)/NSEG; seg=(seg+g_pg-1)&~(g_pg-1);
    dispatch_apply(NSEG, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^(size_t s){
        size_t o=s*seg; if(o>=len) return;
        size_t want=seg; if(o+want>len) want=len-o; size_t got=0;
        while(got<want){ ssize_t n=pread(fd,buf+o+got,want-got,off+(off_t)(o+got)); if(n<=0){ok=0;return;} got+=(size_t)n; }
    });
    (void)ok;
}

static double resident_frac(const uint8_t* base,size_t off,size_t len){
    size_t np=(len+g_pg-1)/g_pg; char* v=malloc(np); if(!v)return -1;
    double f=-1; if(mincore((void*)(base+off),len,v)==0){ size_t r=0; for(size_t i=0;i<np;i++) if(v[i]&1)r++; f=(double)r/np; }
    free(v); return f;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [hot_gb] [cold_gb] [sweeps]\n",argv[0]); return 1; }
    const char* path=argv[1];
    double hot_gb=(argc>2)?atof(argv[2]):4.0;
    double cold_gb=(argc>3)?atof(argv[3]):6.0;
    int sweeps=(argc>4)?atoi(argv[4]):4;
    g_pg=(size_t)sysconf(_SC_PAGESIZE);

    struct stat st; if(stat(path,&st)){perror("stat");return 1;} size_t fsize=st.st_size;
    int n_hot=(int)(hot_gb*1024/256), n_cold=(int)(cold_gb*1024/256);
    off_t hot_base=1ull*1024*1024*1024;
    off_t cold_base=hot_base+(off_t)n_hot*LAYER_BYTES;
    if(cold_base+(off_t)n_cold*LAYER_BYTES>(off_t)fsize){ fprintf(stderr,"file too small\n"); return 1; }

    int fd=open(path,O_RDONLY); g_fd_nocache=open(path,O_RDONLY);
    if(fd<0||g_fd_nocache<0){perror("open");return 1;}
    fcntl(g_fd_nocache,F_NOCACHE,1);
    const uint8_t* base=mmap(NULL,fsize,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){perror("mmap");return 1;}

    g_x=malloc(g_K*sizeof(float)); for(int i=0;i<g_K;i++) g_x[i]=0.01f*(i%17)-0.05f;
    g_y=malloc((LAYER_BYTES/((g_K/256)*144)+16)*sizeof(float));
    for(int b=0;b<NBUF;b++){ posix_memalign((void**)&g_ring[b],g_pg,LAYER_BYTES); memset(g_ring[b],0,LAYER_BYTES); mlock(g_ring[b],LAYER_BYTES); }

    printf("file=%.1f GB  hot=%d L (%.1f GB)  cold=%d L (%.1f GB)  sweeps=%d\n\n",
           fsize/1e9,n_hot,n_hot*256/1024.0,n_cold,n_cold*256/1024.0,sweeps);

    const char* names[3]={"1 zerocopy-nolock","2 zerocopy-mlock","3 pread-ring(F_NOCACHE)"};
    for(int mode=0;mode<3;mode++){

        madvise((void*)(base+hot_base),(size_t)n_hot*LAYER_BYTES,MADV_DONTNEED);
        madvise((void*)(base+cold_base),(size_t)n_cold*LAYER_BYTES,MADV_DONTNEED);

        if(mode==1){

            if(mlock(base+hot_base,(size_t)n_hot*LAYER_BYTES)!=0) perror("mlock hot");
        } else {
            madvise((void*)(base+hot_base),(size_t)n_hot*LAYER_BYTES,MADV_WILLNEED);
        }
        printf("=== mode %s ===\n",names[mode]);
        for(int sw=0;sw<sweeps;sw++){
            double hot_sum=0;
            double t0=now();
            for(int i=0;i<n_hot;i++){
                off_t off=hot_base+(off_t)i*LAYER_BYTES;
                if(mode==2){
                    pread_layer(g_fd_nocache,g_ring[i%NBUF],off,LAYER_BYTES);
                    hot_sum+=neon_over(g_ring[i%NBUF],LAYER_BYTES);
                } else {
                    hot_sum+=neon_over(base+off,LAYER_BYTES);
                }
            }
            double t_hot=now()-t0;

            for(int i=0;i<n_cold;i++)
                pread_layer(g_fd_nocache,g_ring[i%NBUF],cold_base+(off_t)i*LAYER_BYTES,LAYER_BYTES);
            double res=resident_frac(base,(size_t)hot_base,(size_t)n_hot*LAYER_BYTES);
            printf("  sweep %d: HOT-consume %.1f GB/s (avg/layer %.1f, resident %.0f%%)\n",
                   sw,(n_hot*256/1024.0)/t_hot,hot_sum/n_hot,res*100);
        }
        if(mode==1) munlock(base+hot_base,(size_t)n_hot*LAYER_BYTES);
        printf("\n");
    }
    return 0;
}
