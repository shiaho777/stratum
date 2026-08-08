
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

static const uint8_t* g_base;
static size_t g_size;
static int g_nthreads;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static void* worker(void* arg){
    long id=(long)arg;
    size_t chunk=g_size/g_nthreads;
    size_t start=id*chunk;
    size_t end=(id==g_nthreads-1)?g_size:start+chunk;
    volatile uint64_t acc=0;
    const uint64_t* p=(const uint64_t*)(g_base+start);
    size_t n=(end-start)/8;
    for(size_t i=0;i<n;i+=8){
        acc+=p[i];
    }
    uint64_t* ret=malloc(sizeof(uint64_t)); *ret=acc; return ret;
}

static double sweep(void){
    pthread_t th[64];
    double t0=now();
    for(long i=0;i<g_nthreads;i++) pthread_create(&th[i],NULL,worker,(void*)i);
    for(int i=0;i<g_nthreads;i++){ void* r; pthread_join(th[i],&r); free(r); }
    double t1=now();
    return t1-t0;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <file> [n_threads]\n",argv[0]); return 1; }
    g_nthreads = (argc>2)?atoi(argv[2]):8;
    if(g_nthreads<1)g_nthreads=1; if(g_nthreads>64)g_nthreads=64;
    int fd=open(argv[1],O_RDONLY); if(fd<0){perror("open");return 1;}
    struct stat st; fstat(fd,&st); g_size=st.st_size;
    g_base=mmap(NULL,g_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(g_base==MAP_FAILED){perror("mmap");return 1;}
    double gb=g_size/1e9;
    printf("file %.1f GB, %d threads\n",gb,g_nthreads);
    for(int pass=1;pass<=3;pass++){
        double dt=sweep();
        printf("  pass %d: %.3f s  ->  %.1f GB/s\n",pass,dt,gb/dt);
    }
    munmap((void*)g_base,g_size); close(fd);
    return 0;
}
