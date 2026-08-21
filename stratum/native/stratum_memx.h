#ifndef STRATUM_MEMX_H
#define STRATUM_MEMX_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>

#ifdef STRATUM_USE_MEMX
#include "memx_runtime.h"

static memx_runtime_context_t* g_memx_buf_ctx = NULL;
static memx_runtime_context_t* g_memx_kv_ctx = NULL;
static int g_memx_active = 0;
static void* g_memx_weight_base = NULL;
static size_t g_memx_weight_size = 0;
static int g_memx_epoch_infer = 0;
static int g_memx_hard_seal = 0;
static long g_memx_stage_hits = 0;
static long g_memx_stage_cold = 0;
static long g_memx_compress_pages = 0;
static long g_memx_kv_hits = 0;
static long g_memx_donate_pages = 0;
static long g_memx_donate_n = 0;
static double g_memx_donate_s = 0.0;

/* Quota override (MB) for safe repro of MemX pressure scenarios: shrinking a
 * context's quota raises the pool pressure its adaptive paths react to,
 * without stressing the host machine. 0 / unset = compiled default. */
static inline size_t stratum_memx_quota_mb(const char* env, size_t def) {
    const char* e = getenv(env);
    if (!e || !e[0]) return def;
    long mb = atol(e);
    if (mb <= 0) return def;
    return (size_t)mb * 1024 * 1024;
}

static inline int stratum_memx_init(void) {
    if (g_memx_active) return 0;
    {
        const char* e = getenv("STRATUM_MEMX");
        if (e && atoi(e) == 0) {
            fprintf(stderr, "  MEMX: disabled by STRATUM_MEMX=0\n");
            return -1;
        }
    }
    if (memx_runtime_init() != 0) {
        fprintf(stderr, "  MEMX: runtime_init failed\n");
        return -1;
    }
    if (memx_runtime_context_create("stratum-bufs", &g_memx_buf_ctx) != 0) {
        fprintf(stderr, "  MEMX: buffer context create failed\n");
        memx_runtime_shutdown();
        return -1;
    }
    memx_runtime_context_set_quota(g_memx_buf_ctx, stratum_memx_quota_mb("STRATUM_MEMX_BUF_QUOTA_MB", 3ULL * 1024 * 1024 * 1024));
    if (memx_runtime_context_create("stratum-kv", &g_memx_kv_ctx) != 0) {
        fprintf(stderr, "  MEMX: kv context create failed\n");
        memx_runtime_context_destroy(g_memx_buf_ctx);
        g_memx_buf_ctx = NULL;
        memx_runtime_shutdown();
        return -1;
    }
    memx_runtime_context_set_quota(g_memx_kv_ctx, stratum_memx_quota_mb("STRATUM_MEMX_KV_QUOTA_MB", 2ULL * 1024 * 1024 * 1024));
    g_memx_active = 1;
    fprintf(stderr,
        "  MEMX dual-plane: ON | weights=mmap | stage=MemX-pinned | KV=MemX-compress\n");
    return 0;
}

static inline void stratum_memx_shutdown(void) {
    if (!g_memx_active) return;
    if (g_memx_epoch_infer) {
        if (g_memx_buf_ctx) memx_runtime_context_end_epoch(g_memx_buf_ctx, 1);
        if (g_memx_kv_ctx) memx_runtime_context_end_epoch(g_memx_kv_ctx, 1);
        g_memx_epoch_infer = 0;
    }
    if (g_memx_kv_ctx) {
        memx_runtime_context_destroy(g_memx_kv_ctx);
        g_memx_kv_ctx = NULL;
    }
    if (g_memx_buf_ctx) {
        memx_runtime_context_destroy(g_memx_buf_ctx);
        g_memx_buf_ctx = NULL;
    }
    memx_runtime_shutdown();
    g_memx_active = 0;
    g_memx_weight_base = NULL;
    g_memx_weight_size = 0;
}

static inline void* stratum_memx_weight_mmap(int fd, size_t size) {
    (void)fd; (void)size;
    return MAP_FAILED;
}

static inline int stratum_memx_weight_munmap(void* ptr, size_t size) {
    (void)ptr; (void)size;
    return -1;
}

static inline void stratum_memx_bind_weight_mmap(const void* base, size_t size) {
    g_memx_weight_base = (void*)base;
    g_memx_weight_size = size;
}

static inline void stratum_memx_weight_window(
    size_t hot_offset, size_t hot_length,
    size_t prefetch_offset, size_t prefetch_length)
{
    if (!g_memx_weight_base || !g_memx_weight_size) return;
    size_t pgsz = 16384;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps > 0) pgsz = (size_t)ps;
    const char* base = (const char*)g_memx_weight_base;
    if (hot_length > 0 && hot_offset < g_memx_weight_size) {
        size_t a = hot_offset - (hot_offset % pgsz);
        size_t end = hot_offset + hot_length;
        if (end > g_memx_weight_size) end = g_memx_weight_size;
        size_t b = (end + pgsz - 1) / pgsz * pgsz;
        if (b > g_memx_weight_size) b = g_memx_weight_size;
        if (b > a) madvise((void*)(base + a), b - a, MADV_WILLNEED);
    }
    if (prefetch_length > 0 && prefetch_offset < g_memx_weight_size) {
        size_t a = prefetch_offset - (prefetch_offset % pgsz);
        size_t end = prefetch_offset + prefetch_length;
        if (end > g_memx_weight_size) end = g_memx_weight_size;
        size_t b = (end + pgsz - 1) / pgsz * pgsz;
        if (b > g_memx_weight_size) b = g_memx_weight_size;
        if (b > a) madvise((void*)(base + a), b - a, MADV_WILLNEED);
    }
}

static inline void stratum_memx_set_hard_seal(int hard) {
    g_memx_hard_seal = hard ? 1 : 0;
}

static inline void stratum_memx_begin_infer(size_t hot_budget) {
    if (!g_memx_active || g_memx_epoch_infer) return;
    if (hot_budget < (size_t)(64ULL << 20))
        hot_budget = (size_t)(64ULL << 20);
    if (g_memx_buf_ctx)
        memx_runtime_context_begin_epoch(g_memx_buf_ctx, MEMX_EPOCH_INFER, hot_budget);
    if (g_memx_kv_ctx)
        memx_runtime_context_begin_epoch(g_memx_kv_ctx, MEMX_EPOCH_INFER, hot_budget);
    g_memx_epoch_infer = 1;
}

static inline void stratum_memx_end_infer(void) {
    if (!g_memx_active || !g_memx_epoch_infer) return;
    if (g_memx_buf_ctx)
        memx_runtime_context_end_epoch(g_memx_buf_ctx, 1);
    if (g_memx_kv_ctx)
        memx_runtime_context_end_epoch(g_memx_kv_ctx, 1);
    g_memx_epoch_infer = 0;
}

static inline void* stratum_memx_buf_alloc(size_t size) {
    if (!g_memx_active || !g_memx_buf_ctx) return NULL;
    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.role = MEMX_TENSOR_ROLE_TEMPORARY;
    desc.dtype = MEMX_TENSOR_DTYPE_UINT8;
    desc.layout = MEMX_TENSOR_LAYOUT_BLOCKED;
    desc.flags = MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS
               | MEMX_TENSOR_FLAG_SEQUENTIAL;
    desc.rank = 1;
    desc.shape[0] = size;
    void* ptr = memx_runtime_context_malloc_tensor(g_memx_buf_ctx, size, &desc);
    if (!ptr) {
        fprintf(stderr, "  MEMX: stage alloc %zu failed\n", size);
        return NULL;
    }
    g_memx_stage_hits++;
    return ptr;
}

static inline void stratum_memx_buf_free(void* ptr) {
    if (!g_memx_active || !ptr || !g_memx_buf_ctx) return;
    memx_runtime_context_free(g_memx_buf_ctx, ptr);
}

static inline int stratum_memx_stage_ws_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("STRATUM_MEMX_STAGE_WS");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v;
}

static inline void stratum_memx_buf_set_hot(void* ptr, size_t size) {
    if (!g_memx_active || !ptr || !g_memx_buf_ctx) return;
    if (!stratum_memx_stage_ws_enabled()) return;
    memx_runtime_context_update_tensor_flags_range(
        g_memx_buf_ctx, ptr, 0, size,
        MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS);
    memx_runtime_context_ws_advance(
        g_memx_buf_ctx, ptr, 0, size, 0,
        MEMX_WS_FLAG_HOT | MEMX_WS_FLAG_NO_ASYNC);
}

static inline void stratum_memx_buf_set_cold(void* ptr, size_t size) {
    if (!g_memx_active || !ptr || !g_memx_buf_ctx) return;
    if (!stratum_memx_stage_ws_enabled()) return;
    memx_runtime_context_update_tensor_flags_range(
        g_memx_buf_ctx, ptr, 0, size, MEMX_TENSOR_FLAG_COLD);
    memx_runtime_context_ws_advance(
        g_memx_buf_ctx, ptr, 0, 0, 0, MEMX_WS_FLAG_RETIRE);
    g_memx_stage_cold++;
}

static inline void stratum_memx_buf_seal(void* ptr, size_t size) {
    if (!g_memx_active || !ptr || !g_memx_buf_ctx || size == 0) return;
    memx_runtime_context_update_tensor_flags_range(
        g_memx_buf_ctx, ptr, 0, size, MEMX_TENSOR_FLAG_COLD);
    memx_runtime_context_ws_advance(
        g_memx_buf_ctx, ptr, 0, 0, 0,
        MEMX_WS_FLAG_RETIRE | MEMX_WS_FLAG_RETIRE_SYNC | MEMX_WS_FLAG_NO_ASYNC);
    uint64_t n = 0;
    if (memx_runtime_context_force_compress_range(
            g_memx_buf_ctx, ptr, 0, size, &n) == 0)
        g_memx_compress_pages += (long)n;
}

static inline int stratum_memx_pool_pressure(void) {
    if (!g_memx_active) return 0;
    memx_runtime_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    memx_runtime_get_stats(&stats);
    return (int)stats.pool_pressure_percent;
}

static inline void* stratum_memx_kv_alloc(size_t size_bytes,
                                          int n_layers, int n_heads, int head_dim,
                                          int max_kv) {
    if (!g_memx_active || !g_memx_kv_ctx) return NULL;
    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.role = MEMX_TENSOR_ROLE_KV_CACHE;
    desc.dtype = MEMX_TENSOR_DTYPE_FP32;
    desc.layout = MEMX_TENSOR_LAYOUT_BLOCKED;
    desc.flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY
               | MEMX_TENSOR_FLAG_SEQUENTIAL;
    desc.rank = 4;
    desc.shape[0] = (uint64_t)n_layers;
    desc.shape[1] = (uint64_t)max_kv;
    desc.shape[2] = (uint64_t)n_heads;
    desc.shape[3] = (uint64_t)head_dim;
    desc.stride[3] = 1;
    desc.stride[2] = (uint64_t)head_dim;
    desc.stride[1] = (uint64_t)n_heads * (uint64_t)head_dim;
    desc.stride[0] = (uint64_t)max_kv * (uint64_t)n_heads * (uint64_t)head_dim;
    void* ptr = memx_runtime_context_malloc_tensor(
        g_memx_kv_ctx, size_bytes, &desc);
    if (!ptr) {
        fprintf(stderr, "  MEMX: KV alloc %zu failed\n", size_bytes);
        return NULL;
    }
    g_memx_kv_hits++;
    return ptr;
}

static inline void stratum_memx_kv_free(void* ptr) {
    if (!g_memx_active || !ptr || !g_memx_kv_ctx) return;
    memx_runtime_context_free(g_memx_kv_ctx, ptr);
}

static inline void stratum_memx_kv_window(
    void* kv_ptr, size_t kv_size,
    size_t hot_offset, size_t hot_length,
    size_t prefetch_offset, size_t prefetch_length)
{
    if (!g_memx_active || !kv_ptr || !g_memx_kv_ctx) return;
    memx_runtime_kv_cache_window_t window;
    memset(&window, 0, sizeof(window));
    window.struct_size = sizeof(window);
    window.managed_offset = 0;
    window.managed_length = kv_size;
    window.hot_offset = hot_offset;
    window.hot_length = hot_length;
    window.prefetch_offset = prefetch_offset;
    window.prefetch_length = prefetch_length;
    memx_runtime_context_update_kv_cache_window(g_memx_kv_ctx, kv_ptr, &window);
    if (hot_length > 0) {
        memx_runtime_context_ws_advance(
            g_memx_kv_ctx, kv_ptr, hot_offset, hot_length,
            prefetch_length,
            MEMX_WS_FLAG_HOT | MEMX_WS_FLAG_PREFETCH);
    }
}


static inline long stratum_memx_donate_kv(void* ptr, size_t offset, size_t size) {
    if (!g_memx_active || !ptr || !g_memx_kv_ctx || size == 0) return 0;
    memx_runtime_context_update_tensor_flags_range(
        g_memx_kv_ctx, ptr, offset, size,
        MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY);
    uint64_t n = 0;
    if (memx_runtime_context_force_compress_range(
            g_memx_kv_ctx, ptr, offset, size, &n) == 0) {
        g_memx_donate_pages += (long)n;
        g_memx_donate_n++;
        return (long)n;
    }
    return 0;
}

static inline long stratum_memx_donate_stage(void* ptr, size_t size) {
    if (!g_memx_active || !ptr || !g_memx_buf_ctx || size == 0) return 0;
    memx_runtime_context_update_tensor_flags_range(
        g_memx_buf_ctx, ptr, 0, size, MEMX_TENSOR_FLAG_COLD);
    uint64_t n = 0;
    if (memx_runtime_context_force_compress_range(
            g_memx_buf_ctx, ptr, 0, size, &n) == 0) {
        g_memx_donate_pages += (long)n;
        g_memx_donate_n++;
        memx_runtime_context_update_tensor_flags_range(
            g_memx_buf_ctx, ptr, 0, size,
            MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS);
        return (long)n;
    }
    memx_runtime_context_update_tensor_flags_range(
        g_memx_buf_ctx, ptr, 0, size,
        MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS);
    return 0;
}

static inline void stratum_memx_print_stats(void) {
    if (!g_memx_active) return;
    memx_runtime_stats_t stats;
    memx_runtime_context_stats_t b_stats, k_stats;
    memset(&stats, 0, sizeof(stats));
    memset(&b_stats, 0, sizeof(b_stats));
    memset(&k_stats, 0, sizeof(k_stats));
    memx_runtime_get_stats(&stats);
    if (g_memx_buf_ctx) memx_runtime_context_get_stats(g_memx_buf_ctx, &b_stats);
    if (g_memx_kv_ctx) memx_runtime_context_get_stats(g_memx_kv_ctx, &k_stats);
    fprintf(stderr, "  MEMX dual-plane stats:\n");
    fprintf(stderr, "    stage: managed=%lluMB live=%llu hits=%ld cold=%ld cpages=%ld\n",
            (unsigned long long)(b_stats.bytes_in_use / (1024 * 1024)),
            (unsigned long long)b_stats.allocations_live,
            g_memx_stage_hits, g_memx_stage_cold, g_memx_compress_pages);
    fprintf(stderr, "    kv: managed=%lluMB live=%llu hits=%ld\n",
            (unsigned long long)(k_stats.bytes_in_use / (1024 * 1024)),
            (unsigned long long)k_stats.allocations_live, g_memx_kv_hits);
    fprintf(stderr, "    donate: n=%ld pages=%ld time=%.3fs\n",
            g_memx_donate_n, g_memx_donate_pages, g_memx_donate_s);
    fprintf(stderr, "    pool: %llu/%llu MB pressure=%u%% compressed_pages=%llu saved=%lluMB\n",
            (unsigned long long)(stats.pool_used_bytes / (1024 * 1024)),
            (unsigned long long)(stats.pool_capacity_bytes / (1024 * 1024)),
            stats.pool_pressure_percent,
            (unsigned long long)stats.compressed_pages,
            (unsigned long long)(stats.bytes_saved / (1024 * 1024)));
}

#define stratum_memx_is_active() (g_memx_active)

#else

#define stratum_memx_init() (0)
#define stratum_memx_shutdown() do {} while(0)
#define stratum_memx_weight_mmap(fd, size) (MAP_FAILED)
#define stratum_memx_weight_munmap(ptr, size) (-1)
#define stratum_memx_bind_weight_mmap(base, size) do {} while(0)
#define stratum_memx_weight_window(hot_off, hot_len, pf_off, pf_len) do {} while(0)
#define stratum_memx_set_hard_seal(hard) do {} while(0)
#define stratum_memx_begin_infer(hot_budget) do {} while(0)
#define stratum_memx_end_infer() do {} while(0)
#define stratum_memx_buf_alloc(size) ((void*)NULL)
#define stratum_memx_buf_free(ptr) do {} while(0)
#define stratum_memx_buf_set_hot(ptr, size) do {} while(0)
#define stratum_memx_buf_set_cold(ptr, size) do {} while(0)
#define stratum_memx_buf_seal(ptr, size) do {} while(0)
#define stratum_memx_donate_kv(ptr, off, size) (0L)
#define stratum_memx_donate_stage(ptr, size) (0L)
#define stratum_memx_pool_pressure() (0)
#define stratum_memx_kv_alloc(size, nl, nh, hd, mkv) ((void*)NULL)
#define stratum_memx_kv_free(ptr) do {} while(0)
#define stratum_memx_kv_window(ptr, size, hot_off, hot_len, pf_off, pf_len) do {} while(0)
#define stratum_memx_print_stats() do {} while(0)
#define stratum_memx_is_active() (0)

#endif

#endif
