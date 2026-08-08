
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static const char* g_path;
static size_t g_bs;
static off_t  g_start[8];
static size_t g_len;
static int    g_nocache;

static void* worker(void* a){
    long id=(long)a;
    int fd=open(g_path,O_RDONLY);
    if(fd<0){perror("open");return NULL;}
    if(g_nocache) fcntl(fd,F_NOCACHE,1);
    void* buf=NULL;
    if(posix_memalign(&buf,16384,g_bs)!=0){close(fd);return NULL;}
    size_t got=0; off_t off=g_start[id];
    while(got<g_len){
        size_t want=g_bs; if(got+want>g_len)want=g_len-got;
        ssize_t n=pread(fd,buf,want,off+got);
        if(n<=0)break;
        got+=n;
    }
    free(buf); close(fd);
    return NULL;
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s file [nocache=1]\n",argv[0]);return 1;}
    g_path=argv[1];
    g_nocache = (argc>2)? atoi(argv[2]) : 1;
    struct stat st; stat(g_path,&st); size_t fsize=st.st_size;

    size_t testbytes = fsize; if(testbytes > 12ULL*1024*1024*1024) testbytes=12ULL*1024*1024*1024;

    printf("file=%.1f GB  test=%.1f GB  nocache=%d\n",
           fsize/1e9, testbytes/1e9, g_nocache);
    printf("%-8s", "bs\\thr");
    int threads[]={1,2,4,8}; int nt=4;
    for(int t=0;t<nt;t++) printf("  %dthr", threads[t]);
    printf("   (GB/s)\n");

    size_t bss[]={256*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024};
    for(int bi=0; bi<5; bi++){
        g_bs=bss[bi];
        printf("%-6zuK", g_bs/1024);
        for(int ti=0;ti<nt;ti++){
            int T=threads[ti];
            g_len = testbytes/T;
            for(int i=0;i<T;i++) g_start[i]=(off_t)i*g_len;
            double t0=now();
            pthread_t th[8];
            for(long i=0;i<T;i++) pthread_create(&th[i],NULL,worker,(void*)i);
            for(int i=0;i<T;i++) pthread_join(th[i],NULL);
            double dt=now()-t0;
            printf("  %5.1f", testbytes/dt/1e9);
            fflush(stdout);
        }
        printf("\n");
    }
    return 0;
}
