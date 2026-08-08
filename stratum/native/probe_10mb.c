
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <time.h>
#include <errno.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static long peak_rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);

#ifdef __APPLE__
    return ru.ru_maxrss / 1024;
#else
    return ru.ru_maxrss;
#endif
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <safetensors-file>\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); return 1; }
    long file_size = (long)st.st_size;
    fprintf(stderr, "file size: %ld bytes (%.2f MB)\n",
            file_size, file_size / 1024.0 / 1024.0);

#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1) < 0) {
        perror("fcntl(F_NOCACHE)");

    } else {
        fprintf(stderr, "F_NOCACHE enabled — reads will bypass page cache\n");
    }
#endif

    const size_t BUF = 4096;
    unsigned char *buf = (unsigned char *)malloc(BUF);
    if (!buf) { perror("malloc"); return 1; }

    long total_read = 0;
    unsigned long xor_sentinel = 0;
    long peak_at_start = peak_rss_kb();

    double t0 = now_seconds();
    while (total_read < file_size) {
        size_t want = (file_size - total_read) > (long)BUF
                      ? BUF : (size_t)(file_size - total_read);
        ssize_t got = read(fd, buf, want);
        if (got <= 0) {
            if (got < 0 && errno == EINTR) continue;
            perror("read");
            break;
        }

        for (ssize_t i = 0; i < got; i += 64) {
            xor_sentinel ^= buf[i];
        }
        total_read += got;
    }
    double t1 = now_seconds();

    long peak_at_end = peak_rss_kb();
    double mb_read = total_read / 1024.0 / 1024.0;

    fprintf(stderr, "\n=== probe_10mb result ===\n");
    fprintf(stderr, "  bytes read       : %ld (%.2f MB)\n", total_read, mb_read);
    fprintf(stderr, "  wall time        : %.2f s\n", t1 - t0);
    fprintf(stderr, "  read bandwidth   : %.2f MB/s\n", mb_read / (t1 - t0));
    fprintf(stderr, "  peak RSS at start: %ld KB (%.2f MB)\n",
            peak_at_start, peak_at_start / 1024.0);
    fprintf(stderr, "  peak RSS at end  : %ld KB (%.2f MB)\n",
            peak_at_end, peak_at_end / 1024.0);
    fprintf(stderr, "  xor sentinel     : 0x%lx (forces compiler not to elide reads)\n",
            xor_sentinel);

    free(buf);
    close(fd);
    return 0;
}
