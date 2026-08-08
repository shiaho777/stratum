
#ifndef STRATUM_SGEMV_WORKER_H
#define STRATUM_SGEMV_WORKER_H

#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "stratum_st.h"
#include "stratum_neon.h"

typedef struct { float logit; int id; } TopEntryW;

typedef struct {
    int             fd;
    int64_t         packed_off;
    int64_t         row_start;
    int64_t         row_count;
    int64_t         in_features;
    const float*    scale_f32;
    const float*    x;
    float*          y;
    int64_t         block_rows_cap;

    TopEntryW*      topk_out;
    int             topk_K;
    int             topk_size_out;

    const uint8_t*  mmap_base;
} SgemvJob;

typedef struct {
    pthread_t       thread;
    pthread_mutex_t m;
    pthread_cond_t  cv_go;
    pthread_cond_t  cv_done;
    int             state;
    int             err;

    SgemvJob        job;

    uint8_t*        buf_a;
    uint8_t*        buf_b;
} SgemvWorker;

static int sgemv_run_pipeline(int fd, int64_t packed_off,
                              int64_t row_start, int64_t row_count,
                              int64_t in_features,
                              const float* scale_f32,
                              const float* x, float* y,
                              int64_t block_rows_cap,
                              uint8_t* buf_a, uint8_t* buf_b,
                              const uint8_t* mmap_base);

static int sgemv_run_pipeline_topk(int fd, int64_t packed_off,
                                    int64_t row_start, int64_t row_count,
                                    int64_t in_features,
                                    const float* scale_f32,
                                    const float* x,
                                    TopEntryW* heap, int K, int* heap_size_io,
                                    int64_t block_rows_cap,
                                    uint8_t* buf_a, uint8_t* buf_b,
                                    const uint8_t* mmap_base);

static void* sgemv_worker_main(void* arg) {
    SgemvWorker* w = (SgemvWorker*)arg;

#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    pthread_mutex_lock(&w->m);
    while (1) {
        while (w->state == 0) pthread_cond_wait(&w->cv_go, &w->m);
        if (w->state == 2) { pthread_mutex_unlock(&w->m); return NULL; }
        SgemvJob job = w->job;
        pthread_mutex_unlock(&w->m);

        int err;
        if (job.topk_out != NULL) {
            int heap_size = 0;
            err = sgemv_run_pipeline_topk(
                job.fd, job.packed_off,
                job.row_start, job.row_count, job.in_features,
                job.scale_f32, job.x,
                job.topk_out, job.topk_K, &heap_size,
                job.block_rows_cap,
                w->buf_a, w->buf_b,
                job.mmap_base);

            pthread_mutex_lock(&w->m);
            w->job.topk_size_out = heap_size;
            pthread_mutex_unlock(&w->m);
        } else {
            err = sgemv_run_pipeline(
                job.fd, job.packed_off,
                job.row_start, job.row_count, job.in_features,
                job.scale_f32, job.x, job.y,
                job.block_rows_cap,
                w->buf_a, w->buf_b,
                job.mmap_base);
        }

        pthread_mutex_lock(&w->m);
        w->err = err;
        w->state = 0;
        pthread_cond_signal(&w->cv_done);
    }
}

static int sgemv_worker_init(SgemvWorker* w,
                             uint8_t* buf_a, uint8_t* buf_b) {
    w->state = 0;
    w->err = 0;
    w->buf_a = buf_a;
    w->buf_b = buf_b;
    pthread_mutex_init(&w->m, NULL);
    pthread_cond_init(&w->cv_go, NULL);
    pthread_cond_init(&w->cv_done, NULL);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 64 * 1024);
    int rc = pthread_create(&w->thread, &attr, sgemv_worker_main, w);
    pthread_attr_destroy(&attr);
    return rc;
}

static void sgemv_worker_shutdown(SgemvWorker* w) {
    pthread_mutex_lock(&w->m);
    w->state = 2;
    pthread_cond_broadcast(&w->cv_go);
    pthread_mutex_unlock(&w->m);
    pthread_join(w->thread, NULL);
    pthread_mutex_destroy(&w->m);
    pthread_cond_destroy(&w->cv_go);
    pthread_cond_destroy(&w->cv_done);
}

static void sgemv_worker_dispatch(SgemvWorker* w, const SgemvJob* job) {
    pthread_mutex_lock(&w->m);
    w->job = *job;
    w->state = 1;
    pthread_cond_signal(&w->cv_go);
    pthread_mutex_unlock(&w->m);
}

static int sgemv_worker_wait(SgemvWorker* w) {
    pthread_mutex_lock(&w->m);
    while (w->state == 1) pthread_cond_wait(&w->cv_done, &w->m);
    int err = w->err;
    pthread_mutex_unlock(&w->m);
    return err;
}

static int sgemv_run_pipeline(int fd, int64_t packed_off,
                              int64_t row_start, int64_t row_count,
                              int64_t in_features,
                              const float* scale_f32,
                              const float* x, float* y,
                              int64_t block_rows_cap,
                              uint8_t* buf_a, uint8_t* buf_b,
                              const uint8_t* mmap_base)
{

    int64_t target_rows = block_rows_cap;
    if (target_rows < 1) target_rows = 1;
    int64_t row_bytes = in_features / 2;
    int64_t nblocks = (row_count + target_rows - 1) / target_rows;

    if (nblocks == 0) return 0;

    if (mmap_base != NULL) {
        for (int64_t b = 0; b < nblocks; b++) {
            int64_t r_off = b * target_rows;
            int64_t r_n   = (r_off + target_rows <= row_count) ? target_rows : (row_count - r_off);
            const uint8_t* blk = mmap_base + packed_off + (row_start + r_off) * row_bytes;
            st_int4_sgemv_fused_neon(blk,
                                     scale_f32 + row_start + r_off,
                                     (int)r_n, (int)in_features,
                                     x, y + row_start + r_off);
        }
        return 0;
    }

    uint8_t* bufs[2] = { buf_a, buf_b };
    int cur = 0;

    int64_t r0 = (target_rows <= row_count) ? target_rows : row_count;
    int64_t off0 = packed_off + (row_start) * row_bytes;
    if (st_pread_full(fd, bufs[cur], (size_t)(r0 * row_bytes), off0) < 0) return -1;

    for (int64_t b = 0; b < nblocks; b++) {
        int64_t r_off = b * target_rows;
        int64_t r_n   = (r_off + target_rows <= row_count) ? target_rows : (row_count - r_off);

        st_int4_sgemv_fused_neon(bufs[cur],
                                 scale_f32 + row_start + r_off,
                                 (int)r_n, (int)in_features,
                                 x, y + row_start + r_off);

        if (b + 1 < nblocks) {
            int next = 1 - cur;
            int64_t nr_off = (b + 1) * target_rows;
            int64_t nr_n   = (nr_off + target_rows <= row_count) ? target_rows : (row_count - nr_off);
            int64_t off_n  = packed_off + (row_start + nr_off) * row_bytes;
            if (st_pread_full(fd, bufs[next], (size_t)(nr_n * row_bytes), off_n) < 0) return -1;
            cur = 1 - cur;
        }
    }
    return 0;
}

static inline void topk_w_swap(TopEntryW* a, TopEntryW* b) {
    TopEntryW t = *a; *a = *b; *b = t;
}
static inline void topk_w_sift_down(TopEntryW* heap, int n, int i) {
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, smallest = i;
        if (l < n && heap[l].logit < heap[smallest].logit) smallest = l;
        if (r < n && heap[r].logit < heap[smallest].logit) smallest = r;
        if (smallest == i) break;
        topk_w_swap(&heap[i], &heap[smallest]);
        i = smallest;
    }
}
static inline void topk_w_offer(TopEntryW* heap, int K, int* size, float lv, int id) {
    if (*size < K) {
        heap[*size].logit = lv;
        heap[*size].id    = id;
        (*size)++;
        if (*size == K) {
            for (int i = K/2 - 1; i >= 0; i--) topk_w_sift_down(heap, K, i);
        }
    } else if (lv > heap[0].logit) {
        heap[0].logit = lv;
        heap[0].id    = id;
        topk_w_sift_down(heap, K, 0);
    }
}

static int sgemv_run_pipeline_topk(int fd, int64_t packed_off,
                                    int64_t row_start, int64_t row_count,
                                    int64_t in_features,
                                    const float* scale_f32,
                                    const float* x,
                                    TopEntryW* heap, int K, int* heap_size_io,
                                    int64_t block_rows_cap,
                                    uint8_t* buf_a, uint8_t* buf_b,
                                    const uint8_t* mmap_base)
{
    int64_t target_rows = block_rows_cap;
    if (target_rows < 1) target_rows = 1;
    int64_t row_bytes = in_features / 2;
    int64_t nblocks = (row_count + target_rows - 1) / target_rows;
    if (nblocks == 0) { *heap_size_io = 0; return 0; }

    float lblock[512];
    int heap_size = *heap_size_io;

    if (mmap_base != NULL) {
        for (int64_t b = 0; b < nblocks; b++) {
            int64_t r_off = b * target_rows;
            int64_t r_n   = (r_off + target_rows <= row_count) ? target_rows : (row_count - r_off);
            const uint8_t* blk = mmap_base + packed_off + (row_start + r_off) * row_bytes;
            memset(lblock, 0, (size_t)r_n * sizeof(float));
            st_int4_sgemv_fused_neon(blk,
                                     scale_f32 + row_start + r_off,
                                     (int)r_n, (int)in_features,
                                     x, lblock);
            for (int64_t i = 0; i < r_n; i++) {
                int id = (int)(row_start + r_off + i);
                float lv = lblock[i];
                topk_w_offer(heap, K, &heap_size, lv, id);
            }
        }
        *heap_size_io = heap_size;
        return 0;
    }

    uint8_t* bufs[2] = { buf_a, buf_b };
    int cur = 0;

    int64_t r0 = (target_rows <= row_count) ? target_rows : row_count;
    int64_t off0 = packed_off + row_start * row_bytes;
    if (st_pread_full(fd, bufs[cur], (size_t)(r0 * row_bytes), off0) < 0) return -1;

    for (int64_t b = 0; b < nblocks; b++) {
        int64_t r_off = b * target_rows;
        int64_t r_n   = (r_off + target_rows <= row_count) ? target_rows : (row_count - r_off);

        memset(lblock, 0, (size_t)r_n * sizeof(float));
        st_int4_sgemv_fused_neon(bufs[cur],
                                 scale_f32 + row_start + r_off,
                                 (int)r_n, (int)in_features,
                                 x, lblock);
        for (int64_t i = 0; i < r_n; i++) {
            int id = (int)(row_start + r_off + i);
            float lv = lblock[i];
            topk_w_offer(heap, K, &heap_size, lv, id);
        }

        if (b + 1 < nblocks) {
            int next = 1 - cur;
            int64_t nr_off = (b + 1) * target_rows;
            int64_t nr_n   = (nr_off + target_rows <= row_count) ? target_rows : (row_count - nr_off);
            int64_t off_n  = packed_off + (row_start + nr_off) * row_bytes;
            if (st_pread_full(fd, bufs[next], (size_t)(nr_n * row_bytes), off_n) < 0) return -1;
            cur = 1 - cur;
        }
    }
    *heap_size_io = heap_size;
    return 0;
}

#endif
