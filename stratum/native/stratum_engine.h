/*
 * stratum_engine.h — Universal engine infrastructure.
 *
 * Contains all the shared engine logic that was previously duplicated
 * between run_llama_arch and run_qwen35_arch:
 *   - Hard-boundary enforcement (forbidden env vars refuse to run)
 *   - RAM detection
 *   - madvise layer prefetch
 *   - GPU initialization
 *   - Sliding window KV cache with attention sink
 *   - Speculative decoding infrastructure
 *   - Memory reporting
 *
 * No model-specific logic. All parameters come from StratumConfig.
 */
#ifndef STRATUM_ENGINE_H
#define STRATUM_ENGINE_H

#include "stratum_arch.h"
#include "stratum_linear.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <time.h>
#include <errno.h>

/* ------------------------------------------------------------------ */
/*  Hard boundary enforcement                                          */
/* ------------------------------------------------------------------ */
/* The project's three boundaries (quality, wired memory, machine load)
 * are enforced in code, not by convention: setting any forbidden
 * variable aborts with an explicit message before any work is done.
 * Both architecture run() entries call this first. */

static inline void stratum_enforce_boundaries(void) {
    static const char* forbidden[] = {
        /* quality boundary — re-encoding weights */
        "STRATUM_Q4_0",
        "STRATUM_PREDECODE",
        /* memory boundary — locking / pinning weights */
        "STRATUM_MLOCK_ALL",
        "STRATUM_HOT_GB",
        "STRATUM_KEEP_RESIDENT",
    };
    for (size_t i = 0; i < sizeof(forbidden) / sizeof(forbidden[0]); i++) {
        const char* v = getenv(forbidden[i]);
        if (v && atoi(v) != 0) {
            fprintf(stderr, "stratum: ERROR: %s is forbidden by project boundary"
                            " (quality / wired-memory) — refusing to run.\n",
                    forbidden[i]);
            exit(1);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Engine configuration (runtime-tunable, not model-specific)         */
/* ------------------------------------------------------------------ */

typedef struct {
    /* KV cache */
    int  max_kv;            /* ring buffer window size */
    int  kv_sink;           /* attention sink tokens (always kept) */

    /* Speculative decoding */
    int  spec_k;            /* n-gram spec: K candidate tokens */
    int  spec_b;            /* batch size for spec forward (B_MAX) */

    /* Prefetch */
    int  prefetch_ahead;    /* madvise layers ahead */
    int  release_behind;    /* madvise DONTNEED behind (0 = let OS manage) */

    /* Derived at init */
    size_t total_ram;       /* hw.memsize */
    size_t model_size;      /* mmap_size */
} StratumEngineConfig;

static StratumEngineConfig g_eng;

/* ------------------------------------------------------------------ */
/*  Initialization                                                     */
/* ------------------------------------------------------------------ */

static inline void stratum_engine_init(size_t mmap_size) {
    g_eng.max_kv        = 4096;
    g_eng.kv_sink       = 4;
    g_eng.spec_k        = 0;
    g_eng.spec_b        = 5;
    g_eng.prefetch_ahead = 1;
    g_eng.release_behind = 0;
    g_eng.model_size    = mmap_size;

    /* Detect RAM */
    g_eng.total_ram = 0;
    size_t len = sizeof(g_eng.total_ram);
    if (sysctlbyname("hw.memsize", &g_eng.total_ram, &len, NULL, 0) != 0)
        g_eng.total_ram = 0;

    /* Override from env */
    const char* e;
    if ((e = getenv("STRATUM_MAX_KV")))      g_eng.max_kv = atoi(e);
    if ((e = getenv("STRATUM_KV_SINK")))     g_eng.kv_sink = atoi(e);
    if ((e = getenv("STRATUM_NGRAM_SPEC")))  g_eng.spec_k = atoi(e);
    if ((e = getenv("STRATUM_B_MAX")))       g_eng.spec_b = atoi(e);
    if ((e = getenv("STRATUM_PREFETCH")))    g_eng.prefetch_ahead = atoi(e);
    if ((e = getenv("STRATUM_RELEASE")))     g_eng.release_behind = atoi(e);
}

/* ------------------------------------------------------------------ */
/*  GPU initialization (Metal)                                         */
/* ------------------------------------------------------------------ */

static inline int stratum_engine_init_gpu(const char* mmap_base, size_t mmap_size) {
#ifdef STRATUM_USE_METAL
    if (!getenv("STRATUM_GPU")) return 0;

    const char* mlpath = getenv("STRATUM_METALLIB");
    if (!mlpath) mlpath = "native/stratum_q4k.metallib";
    if (stratum_metal_init(mlpath, NULL, 0) == 0) {
        stratum_metal_set_model_base(mmap_base, mmap_size);
        fprintf(stderr, "  Metal GPU: initialized (staging-only, model NOT on GPU)\n");
        return 0;
    }
    fprintf(stderr, "  Metal GPU: init failed, falling back to CPU\n");
    g_st.use_metal = 0;
    return -1;
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------ */
/*  Sliding window KV cache with attention sink                        */
/*  Universal — works for any Transformer architecture with attention  */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Ring buffer: [sink(0..SINK-1)] [ring(SINK..MAX_KV-1)] */
    int  max_kv;        /* total buffer size */
    int  sink;          /* sink token count (immutable) */
    int  write_pos;     /* next write position in ring region */
    int  n_filled;      /* total tokens currently in cache */

    /* Per-layer storage (allocated by architecture) */
    /* The architecture owns the actual K/V float buffers; */
    /* this struct only manages the index mapping. */
} StratumKVCache;

static inline void stratum_kv_init(StratumKVCache* kv, int max_kv, int sink) {
    kv->max_kv = max_kv;
    kv->sink = sink;
    kv->write_pos = 0;
    kv->n_filled = 0;
}

/* Map a logical token position to a physical KV slot.
 * Tokens [0..sink-1] are always at physical slots [0..sink-1].
 * Tokens [sink..] go into the ring buffer starting at physical slot [sink]. */
static inline int stratum_kv_slot(const StratumKVCache* kv, int logical_pos) {
    if (logical_pos < kv->sink)
        return logical_pos;

    int ring_pos = logical_pos - kv->sink;
    int ring_size = kv->max_kv - kv->sink;
    int physical = kv->sink + (ring_pos % ring_size);
    return physical;
}

/* Get the number of valid KV entries to attend to (for causal mask) */
static inline int stratum_kv_n_valid(const StratumKVCache* kv) {
    if (kv->n_filled < kv->max_kv)
        return kv->n_filled;
    return kv->max_kv;
}

/* Advance write position after writing a new token */
static inline void stratum_kv_advance(StratumKVCache* kv) {
    kv->n_filled++;
    int ring_size = kv->max_kv - kv->sink;
    if (kv->n_filled > kv->sink) {
        kv->write_pos = (kv->write_pos + 1) % ring_size;
    }
}

/* Get the physical slot for the current write position */
static inline int stratum_kv_write_slot(const StratumKVCache* kv) {
    return stratum_kv_slot(kv, kv->n_filled);
}

/* Get the logical positions currently in the cache (for attention) */
static inline int stratum_kv_logical_pos(const StratumKVCache* kv, int physical_slot) {
    if (physical_slot < kv->sink)
        return physical_slot;

    int ring_size = kv->max_kv - kv->sink;
    int ring_idx = physical_slot - kv->sink;

    if (kv->n_filled <= kv->max_kv) {
        /* Cache not yet full: logical = physical */
        return physical_slot;
    }

    /* Cache is full: oldest ring entry has been overwritten.
     * The oldest surviving ring entry is at write_pos (next to be overwritten).
     * So the newest entry is at write_pos - 1 (mod ring_size).
     * logical_pos = n_filled - ring_size + (ring_idx - write_pos + ring_size) % ring_size
     */
    int offset = (ring_idx - kv->write_pos + ring_size) % ring_size;
    return kv->n_filled - ring_size + offset;
}

/* ------------------------------------------------------------------ */
/*  madvise layer prefetch — universal                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    const Gguf* gguf;
    const uint8_t* mmap_base;
    int n_layers;
    int prefetch_ahead;
    int release_behind;
    int keep_resident;
} StratumPrefetchCtx;

static inline void stratum_prefetch_layer(StratumPrefetchCtx* ctx, int layer) {
    if (ctx->keep_resident) return;  /* hot mode: no madvise needed */
    if (ctx->prefetch_ahead <= 0) return;

    /* Prefetch layers [layer+1 .. layer+prefetch_ahead] */
    for (int ahead = 1; ahead <= ctx->prefetch_ahead; ahead++) {
        int l = layer + ahead;
        if (l >= ctx->n_layers) break;

        /* Find all tensors for this layer and madvise WILLNEED */
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "blk.%d.", l);
        /* The GGUF tensor array is sorted; we scan for matching prefixes.
         * This is generic — no assumption about which tensors exist. */
        for (uint32_t t = 0; t < ctx->gguf->n_tensors; t++) {
            const GgufTensor* gt = &ctx->gguf->tensors[t];
            if (strncmp(gt->name, prefix, strlen(prefix)) == 0) {
                const uint8_t* addr = ctx->mmap_base + gt->offset;
                size_t sz = gt->nbytes;
                /* Page-align */
                uintptr_t page_start = (uintptr_t)addr & ~0x3FFFUL;
                size_t page_end = ((uintptr_t)addr + sz + 0x3FFF) & ~0x3FFFUL;
                madvise((void*)page_start, page_end - page_start, MADV_WILLNEED);
            }
        }
    }

    /* Release layers far behind */
    if (ctx->release_behind > 0) {
        int l = layer - ctx->release_behind;
        if (l >= 0) {
            char prefix[32];
            snprintf(prefix, sizeof(prefix), "blk.%d.", l);
            for (uint32_t t = 0; t < ctx->gguf->n_tensors; t++) {
                const GgufTensor* gt = &ctx->gguf->tensors[t];
                if (strncmp(gt->name, prefix, strlen(prefix)) == 0) {
                    const uint8_t* addr = ctx->mmap_base + gt->offset;
                    size_t sz = gt->nbytes;
                    uintptr_t page_start = (uintptr_t)addr & ~0x3FFFUL;
                    size_t page_end = ((uintptr_t)addr + sz + 0x3FFF) & ~0x3FFFUL;
                    madvise((void*)page_start, page_end - page_start, MADV_DONTNEED);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Memory reporting — universal                                       */
/* ------------------------------------------------------------------ */

static inline void stratum_report_memory(void) {
    mach_port_t task = mach_task_self();
    mach_vm_size_t rss = 0;
    mach_vm_size_t vsz = 0;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_vm_info_data_t info;
    if (task_info(task, TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        rss = info.phys_footprint;
        vsz = info.virtual_size;
    }
    fprintf(stderr, "  memory: RSS=%.1fMB (phys_footprint), VM=%.1fGB\n",
            rss / (1024.0*1024.0),
            vsz / (1024.0*1024.0*1024.0));
}

/* ------------------------------------------------------------------ */
/*  Argmax + softmax — universal sampling primitives                   */
/* ------------------------------------------------------------------ */

static inline int stratum_argmax(const float* logits, int n) {
    int best = 0;
    float maxv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > maxv) { maxv = logits[i]; best = i; }
    }
    return best;
}

static inline float stratum_softmax(float* x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    if (sum > 0) for (int i = 0; i < n; i++) x[i] /= sum;
    return sum;
}

/* ------------------------------------------------------------------ */
/*  Optional per-step logits dump (STRATUM_LOGITS_DUMP=<path>)         */
/*                                                                     */
/*  Opt-in measurement hook for offline backend/config comparison:
 *  run the same prompt twice with different STRATUM_* settings, then
 *  compare the two dumps with tools_logit_compare.c (KL(base||Q),
 *  top-1 agreement, max|Δ|). Pure measurement — no effect on output. */
/*  Format: "SLOG0001" | u32 n_vocab | repeat{ u32 token; f32[n_vocab] }*/
/*  Captured at the greedy sampler, so plain greedy runs (what the     */
/*  gates exercise) record every emitted token; speculative paths      */
/*  that bypass the sampler are not represented. Zero cost when the    */
/*  env var is unset.                                                  */
/* ------------------------------------------------------------------ */

static inline void stratum_logits_dump_record(const float* logits, int n_vocab, int token) {
    static FILE* fp;
    static int initialized;
    if (!initialized) {
        initialized = 1;
        const char* path = getenv("STRATUM_LOGITS_DUMP");
        if (path && path[0]) {
            fp = fopen(path, "wb");
            if (fp) {
                fwrite("SLOG0001", 1, 8, fp);
                uint32_t nv = (uint32_t)n_vocab;
                fwrite(&nv, 4, 1, fp);
            } else {
                fprintf(stderr, "logits dump: cannot open %s\n", path);
            }
        }
    }
    if (!fp) return;
    uint32_t t = (uint32_t)token;
    fwrite(&t, 4, 1, fp);
    fwrite(logits, sizeof(float), (size_t)n_vocab, fp);
    fflush(fp);
}

#endif /* STRATUM_ENGINE_H */
