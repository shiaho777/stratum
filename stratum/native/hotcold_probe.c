
#define _GNU_SOURCE
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

static int    g_fd_cached;
static int    g_fd_nocache;
static uint8_t* g_ring[NBUF];
static size_t g_pg;

static double pread_layer(int fd, uint8_t* buf, off_t off, size_t len){
    __block int ok = 1;
    size_t seg = (len + NSEG - 1)/NSEG;
    seg = (seg + g_pg-1) & ~(g_pg-1);
    double t0 = now();
    dispatch_apply(NSEG, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^(size_t s){
        size_t o = s*seg; if (o>=len) return;
        size_t want = seg; if (o+want>len) want = len-o;
        size_t got=0;
        while(got<want){
            ssize_t n = pread(fd, buf+o+got, want-got, off+(off_t)(o+got));
            if(n<=0){ ok=0; return; }
            got+=(size_t)n;
        }
    });
    double dt = now()-t0;
    if(!ok) return -1.0;

    volatile uint64_t sink=0;
    for(size_t o=0;o<len;o+=g_pg) sink += buf[o];
    (void)sink;
    return (double)len/dt/1e9;
}

static double resident_frac(const uint8_t* base, size_t off, size_t len){
    size_t np = (len + g_pg - 1)/g_pg;
    char* vec = malloc(np);
    if(!vec) return -1.0;
    double frac = -1.0;
    if(mincore((void*)(base+off), len, vec)==0){
        size_t res=0; for(size_t i=0;i<np;i++) if(vec[i]&1) res++;
        frac = (double)res/(double)np;
    }
    free(vec);
    return frac;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [hot_gb] [cold_gb] [sweeps]\n",argv[0]); return 1; }
    const char* path = argv[1];
    double hot_gb  = (argc>2)? atof(argv[2]) : 6.0;
    double cold_gb = (argc>3)? atof(argv[3]) : 6.0;
    int    sweeps  = (argc>4)? atoi(argv[4]) : 3;
    g_pg = (size_t)sysconf(_SC_PAGESIZE);

    struct stat st; if(stat(path,&st)!=0){ perror("stat"); return 1; }
    size_t fsize = (size_t)st.st_size;

    int n_hot  = (int)(hot_gb *1024/256);
    int n_cold = (int)(cold_gb*1024/256);

    off_t hot_base  = 1ull*1024*1024*1024;
    off_t cold_base = hot_base + (off_t)n_hot*LAYER_BYTES;
    if(cold_base + (off_t)n_cold*LAYER_BYTES > (off_t)fsize){
        fprintf(stderr,"file too small for hot+cold; shrink args\n"); return 1;
    }

    g_fd_cached  = open(path,O_RDONLY);
    g_fd_nocache = open(path,O_RDONLY);
    if(g_fd_cached<0||g_fd_nocache<0){ perror("open"); return 1; }
    fcntl(g_fd_nocache, F_NOCACHE, 1);

    const uint8_t* base = mmap(NULL,fsize,PROT_READ,MAP_PRIVATE,g_fd_cached,0);
    if(base==MAP_FAILED){ perror("mmap"); return 1; }

    madvise((void*)(base+hot_base),  (size_t)n_hot*LAYER_BYTES,  MADV_DONTNEED);
    madvise((void*)(base+cold_base), (size_t)n_cold*LAYER_BYTES, MADV_DONTNEED);

    for(int b=0;b<NBUF;b++){
        if(posix_memalign((void**)&g_ring[b],g_pg,LAYER_BYTES)!=0){ perror("memalign"); return 1; }
        memset(g_ring[b],0,LAYER_BYTES);
        mlock(g_ring[b],LAYER_BYTES);
    }

    printf("file=%.1f GB  hot=%d layers (%.1f GB)  cold=%d layers (%.1f GB)  sweeps=%d  page=%zuK\n",
           fsize/1e9, n_hot, n_hot*256/1024.0, n_cold, n_cold*256/1024.0, sweeps, g_pg/1024);
    printf("ring=%d x 256MB = %.0f MB wired\n\n", NBUF, NBUF*256.0);

    off_t coff = (argc>5)? (off_t)atoi(argv[5]) : 0;
    if (coff) printf("[cold reads offset by +%lld bytes (unaligned test)]\n", (long long)coff);

    const char* names[3] = {"A cached-cold (dead path)","B nocache-cold (NEW)","C nocache-all (stable stream)"};
    for(int mode=0; mode<3; mode++){

        madvise((void*)(base+hot_base),  (size_t)n_hot*LAYER_BYTES,  MADV_DONTNEED);
        madvise((void*)(base+cold_base), (size_t)n_cold*LAYER_BYTES, MADV_DONTNEED);
        printf("=== mode %s ===\n", names[mode]);
        for(int sw=0; sw<sweeps; sw++){
            int hot_fd  = (mode==2)? g_fd_nocache : g_fd_cached;
            int cold_fd = (mode==0)? g_fd_cached  : g_fd_nocache;
            double hot_gbps_sum=0; int hb=0;
            double t_hot0=now();
            for(int i=0;i<n_hot;i++){
                off_t off = hot_base + (off_t)i*LAYER_BYTES;
                double g = pread_layer(hot_fd, g_ring[i%NBUF], off, LAYER_BYTES);
                if(g>0){ hot_gbps_sum+=g; hb++; }
            }
            double t_hot = now()-t_hot0;
            double t_cold0=now();
            for(int i=0;i<n_cold;i++){
                off_t off = cold_base + (off_t)i*LAYER_BYTES + coff;
                pread_layer(cold_fd, g_ring[i%NBUF], off, LAYER_BYTES - (coff?g_pg:0));
            }
            double t_cold = now()-t_cold0;
            double hot_res  = resident_frac(base,(size_t)hot_base,(size_t)n_hot*LAYER_BYTES);
            double cold_res = resident_frac(base,(size_t)cold_base,(size_t)n_cold*LAYER_BYTES);
            double hot_agg  = (n_hot *256/1024.0)/t_hot;
            double cold_agg = (n_cold*256/1024.0)/t_cold;
            printf("  sweep %d: HOT %.2f GB/s (resident %.0f%%)   COLD %.2f GB/s (resident %.0f%%)\n",
                   sw, hot_agg, hot_res*100, cold_agg, cold_res*100);
        }
        printf("\n");
    }
    return 0;
}
