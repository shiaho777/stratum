
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "stratum_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static id<MTLDevice>              g_device     = nil;
static id<MTLCommandQueue>        g_queue      = nil;
static id<MTLLibrary>             g_lib        = nil;
static id<MTLComputePipelineState> g_q4k_sgemv = nil;
static id<MTLComputePipelineState> g_q4k_sgemv_b[33] = {nil};
static id<MTLComputePipelineState> g_q4k_sgemv_bp = nil;
static id<MTLComputePipelineState> g_q4k_sgemv_bp_g2 = nil;
static id<MTLComputePipelineState> g_q4k_sgemv_bp_g2_add = nil;
static id<MTLComputePipelineState> g_q4k_sgemv_bp_add = nil;
static id<MTLComputePipelineState> g_q4k_dual_sgemv_bp = nil;
static id<MTLComputePipelineState> g_q4k_dual_sgemv_bp_g2 = nil;
static id<MTLComputePipelineState> g_q4k_dual_swiglu_bp_g2 = nil;
static id<MTLComputePipelineState> g_q4k_dual_sgemv_bp_g4 = nil;
static id<MTLComputePipelineState> g_q4k_dual_sgemv_bp_v2 = nil;
static id<MTLComputePipelineState> g_q4k_sgemv_simdb = nil;
static id<MTLComputePipelineState> g_q4k_argmax_b = nil;
static id<MTLComputePipelineState> g_q4k_rowtiled_b = nil;
static id<MTLComputePipelineState> g_q5k_sgemv = nil;
static id<MTLComputePipelineState> g_q6k_sgemv = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_b[33] = {nil};
static id<MTLComputePipelineState> g_q2k_sgemv = nil;
static id<MTLComputePipelineState> g_q2k_sgemv_b[33] = {nil};
static id<MTLComputePipelineState> g_q6k_sgemv_bp = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_add = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_v4 = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_v5 = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_v5_g2 = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_v5_g2_add = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_v5_g4 = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_v6 = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_v5_add = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_bp_g2 = nil;
static id<MTLComputePipelineState> g_q6k_sgemv_simdb = nil;
static id<MTLComputePipelineState> g_q6k_rowtiled_b16 = nil;
static id<MTLComputePipelineState> g_q6k_rowtiled_s8 = nil;
static id<MTLComputePipelineState> g_q6k_argmax_b = nil;
static id<MTLComputePipelineState> g_q6k_top1_tiles_b = nil;
static id<MTLComputePipelineState> g_top1_reduce_tiles = nil;
static id<MTLComputePipelineState> g_swiglu = nil;
static id<MTLComputePipelineState> g_rmsnorm = nil;
static id<MTLComputePipelineState> g_rmsnorm_b = nil;
/* Qwen3.5-specific kernels */
static id<MTLComputePipelineState> g_rope_half = nil;
static id<MTLComputePipelineState> g_rmsnorm_ph = nil;   /* per-head */
static id<MTLComputePipelineState> g_sigmoid_gate = nil;
static id<MTLComputePipelineState> g_split_qgate = nil;
static id<MTLComputePipelineState> g_attn_gated = nil;
static id<MTLComputePipelineState> g_rope = nil;
static id<MTLComputePipelineState> g_rope_b = nil;
static id<MTLComputePipelineState> g_attn = nil;
static id<MTLComputePipelineState> g_attn_b = nil;
static id<MTLComputePipelineState> g_q4k_coalesced = nil;  /* V12 */
static id<MTLComputePipelineState> g_kv_scatter = nil;
static id<MTLComputePipelineState> g_kv_rope_scatter = nil;
static id<MTLComputePipelineState> g_argmax_b = nil;
static id<MTLComputePipelineState> g_add = nil;
static id<MTLComputePipelineState> g_copy = nil;
static id<MTLBuffer> g_ffn_gate = nil, g_ffn_up = nil, g_ffn_a = nil, g_ffn_out = nil;
static size_t g_ffn_ff_cap = 0, g_ffn_h_cap = 0;

static id<MTLBuffer>              g_xbuf      = nil;
static id<MTLBuffer>              g_ybuf      = nil;
static size_t                     g_xbuf_size = 0;
static size_t                     g_ybuf_size = 0;

#define STRATUM_METAL_MAX_CHUNKS 16
#define STRATUM_METAL_CHUNK_SIZE  ((size_t)10 * 1024 * 1024 * 1024)
#define STRATUM_METAL_STEP        ((size_t) 8 * 1024 * 1024 * 1024)
static id<MTLBuffer>              g_model_chunks[STRATUM_METAL_MAX_CHUNKS] = {nil};
static int                        g_model_n_chunks = 0;
static const void*                g_model_base = NULL;
static size_t                     g_model_size = 0;

#define STRATUM_METAL_MAX_STAGING 8
static id<MTLBuffer> g_staging[STRATUM_METAL_MAX_STAGING] = {nil};
static int           g_n_staging = 0;

static id<MTLBuffer> g_xbatch = nil;
static id<MTLBuffer> g_ybatch = nil;
static size_t        g_xbatch_size = 0;
static size_t        g_ybatch_size = 0;

static long   g_n_dispatch = 0;
static double g_dispatch_secs = 0.0;
static id<MTLBuffer> g_ygroup = nil;
static size_t        g_ygroup_size = 0;

enum {
    BPROF_ATTN_NORM = 0,
    BPROF_QKV,
    BPROF_ROPE_KV,
    BPROF_ATTN,
    BPROF_O_PROJ,
    BPROF_FFN_NORM,
    BPROF_GATE_UP,
    BPROF_SWIGLU,
    BPROF_DOWN,
    BPROF_OUT_NORM,
    BPROF_LM_HEAD,
    BPROF_ARGMAX,
    BPROF_COUNT
};

static const char* g_bprof_names[BPROF_COUNT] = {
    "attn_norm", "qkv", "rope_kv", "attention", "o_proj",
    "ffn_norm", "gate_up", "swiglu", "down", "out_norm",
    "lm_head", "argmax"
};
static double g_bprof_secs[BPROF_COUNT] = {0};
static long   g_bprof_calls[BPROF_COUNT] = {0};
static long   g_bprof_sweeps = 0;
static int    g_bprof_enabled_ever = 0;
static int    g_bprof_atexit = 0;

static double elapsed_secs(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

static void stratum_metal_bprofile_print(void) {
    if (!g_bprof_enabled_ever) return;
    double total = 0.0;
    for (int i = 0; i < BPROF_COUNT; i++) total += g_bprof_secs[i];
    fprintf(stderr, "\n  [gpu batch profile] %ld sweeps, %.3fs staged GPU time\n",
            g_bprof_sweeps, total);
    for (int i = 0; i < BPROF_COUNT; i++) {
        if (!g_bprof_calls[i]) continue;
        double pct = total > 0.0 ? 100.0 * g_bprof_secs[i] / total : 0.0;
        fprintf(stderr, "    %-10s %.3fs  %6.2f%%  %.3f ms/sweep\n",
                g_bprof_names[i], g_bprof_secs[i], pct,
                1000.0 * g_bprof_secs[i] / (double)g_bprof_calls[i]);
    }
}

static int load_batched_psos(const char* prefix, int target) {
    /* target: 0 = q4k, 1 = q6k, 2 = q2k */
    int loaded = 0;
    for (int b = 1; b <= 32; b++) {
        @autoreleasepool {
            NSError* err = nil;
            char fname[96];
            snprintf(fname, sizeof fname, "%s_b%d", prefix, b);
            NSString* name = [NSString stringWithUTF8String:fname];
            id<MTLFunction> fn = [g_lib newFunctionWithName:name];
            if (!fn) {
                if (b == 1) {
                    fprintf(stderr, "Metal: specialized %s not found\n", fname);
                }
                continue;
            }
            id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) {
                fprintf(stderr, "Metal: specialized %s pipeline failed: %s\n",
                        fname, [[err localizedDescription] UTF8String]);
                continue;
            }
            if (target == 2)      g_q2k_sgemv_b[b] = pso;
            else if (target == 1) g_q6k_sgemv_b[b] = pso;
            else                  g_q4k_sgemv_b[b] = pso;
            loaded++;
        }
    }
    return loaded;
}

int stratum_metal_init(const char* metallib_path,
                       const void* model_base,
                       size_t model_size) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "Metal: no default device\n");
            return -1;
        }
        fprintf(stderr, "  Metal device: %s (max threads/tg = %lu)\n",
                [[g_device name] UTF8String],
                (unsigned long)g_device.maxThreadsPerThreadgroup.width);

        g_queue = [g_device newCommandQueue];
        if (!g_queue) { fprintf(stderr, "Metal: queue creation failed\n"); return -1; }
        if (!g_bprof_atexit) {
            atexit(stratum_metal_bprofile_print);
            g_bprof_atexit = 1;
        }

        NSError* err = nil;
        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        g_lib = [g_device newLibraryWithURL:url error:&err];
        if (!g_lib) {
            fprintf(stderr, "Metal: failed to load library at %s: %s\n",
                    metallib_path, [[err localizedDescription] UTF8String]);
            return -1;
        }

        id<MTLFunction> fn = [g_lib newFunctionWithName:@"q4k_sgemv_row"];
        if (!fn) {
            fprintf(stderr, "Metal: kernel q4k_sgemv_row not found in library\n");
            return -1;
        }
        g_q4k_sgemv = [g_device newComputePipelineStateWithFunction:fn error:&err];
        if (!g_q4k_sgemv) {
            fprintf(stderr, "Metal: pipeline creation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            return -1;
        }

        int q4b = load_batched_psos("q4k_sgemv_row_batched", 0);
        if (q4b == 0) fprintf(stderr, "Metal: no specialized q4k batched pipelines loaded\n");
        id<MTLFunction> fnbp = [g_lib newFunctionWithName:@"q4k_sgemv_row_bparallel"];
        if (fnbp) {
            g_q4k_sgemv_bp = [g_device newComputePipelineStateWithFunction:fnbp error:&err];
            if (!g_q4k_sgemv_bp)
                fprintf(stderr, "Metal: q4k batch-parallel pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fnbpg2 = [g_lib newFunctionWithName:@"q4k_sgemv_row_bparallel_g2"];
        if (fnbpg2) {
            g_q4k_sgemv_bp_g2 = [g_device newComputePipelineStateWithFunction:fnbpg2 error:&err];
            if (!g_q4k_sgemv_bp_g2)
                fprintf(stderr, "Metal: q4k g2 batch-parallel pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fnbpg2add = [g_lib newFunctionWithName:@"q4k_sgemv_row_bparallel_g2_add"];
        if (fnbpg2add) {
            g_q4k_sgemv_bp_g2_add = [g_device newComputePipelineStateWithFunction:fnbpg2add error:&err];
            if (!g_q4k_sgemv_bp_g2_add)
                fprintf(stderr, "Metal: q4k g2 batch-parallel add pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fnbpadd = [g_lib newFunctionWithName:@"q4k_sgemv_row_bparallel_add"];
        if (fnbpadd) {
            g_q4k_sgemv_bp_add = [g_device newComputePipelineStateWithFunction:fnbpadd error:&err];
            if (!g_q4k_sgemv_bp_add)
                fprintf(stderr, "Metal: q4k batch-parallel add pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fndual = [g_lib newFunctionWithName:@"q4k_dual_sgemv_row_bparallel"];
        if (fndual) {
            g_q4k_dual_sgemv_bp = [g_device newComputePipelineStateWithFunction:fndual error:&err];
            if (!g_q4k_dual_sgemv_bp)
                fprintf(stderr, "Metal: q4k dual batch-parallel pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fndualg2 = [g_lib newFunctionWithName:@"q4k_dual_sgemv_row_bparallel_g2"];
        if (fndualg2) {
            g_q4k_dual_sgemv_bp_g2 = [g_device newComputePipelineStateWithFunction:fndualg2 error:&err];
            if (!g_q4k_dual_sgemv_bp_g2)
                fprintf(stderr, "Metal: q4k dual g2 batch-parallel pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fndualsgg2 = [g_lib newFunctionWithName:@"q4k_dual_swiglu_row_bparallel_g2"];
        if (fndualsgg2) {
            g_q4k_dual_swiglu_bp_g2 = [g_device newComputePipelineStateWithFunction:fndualsgg2 error:&err];
            if (!g_q4k_dual_swiglu_bp_g2)
                fprintf(stderr, "Metal: q4k dual swiglu g2 batch-parallel pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fndualg4 = [g_lib newFunctionWithName:@"q4k_dual_sgemv_row_bparallel_g4"];
        if (fndualg4) {
            g_q4k_dual_sgemv_bp_g4 = [g_device newComputePipelineStateWithFunction:fndualg4 error:&err];
            if (!g_q4k_dual_sgemv_bp_g4)
                fprintf(stderr, "Metal: q4k dual g4 batch-parallel pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fndual2 = [g_lib newFunctionWithName:@"q4k_dual_sgemv_row_bparallel_v2"];
        if (fndual2) {
            g_q4k_dual_sgemv_bp_v2 = [g_device newComputePipelineStateWithFunction:fndual2 error:&err];
            if (!g_q4k_dual_sgemv_bp_v2)
                fprintf(stderr, "Metal: q4k dual v2 batch-parallel pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fnsb = [g_lib newFunctionWithName:@"q4k_sgemv_row_simdb_batched"];
        if (fnsb) {
            g_q4k_sgemv_simdb = [g_device newComputePipelineStateWithFunction:fnsb error:&err];
            if (!g_q4k_sgemv_simdb)
                fprintf(stderr, "Metal: q4k simd-batch pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        id<MTLFunction> fnqa = [g_lib newFunctionWithName:@"q4k_argmax_batched"];
        if (fnqa) {
            g_q4k_argmax_b = [g_device newComputePipelineStateWithFunction:fnqa error:&err];
            if (!g_q4k_argmax_b)
                fprintf(stderr, "Metal: q4k argmax pipeline failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }

        id<MTLFunction> fn5 = [g_lib newFunctionWithName:@"q5k_sgemv_row"];
        if (fn5) {
            g_q5k_sgemv = [g_device newComputePipelineStateWithFunction:fn5 error:&err];
            if (!g_q5k_sgemv) {
                fprintf(stderr, "Metal: q5k pipeline creation failed: %s\n",
                        [[err localizedDescription] UTF8String]);
            }
        }
        id<MTLFunction> fn6 = [g_lib newFunctionWithName:@"q6k_sgemv_row"];
        if (fn6) {
            g_q6k_sgemv = [g_device newComputePipelineStateWithFunction:fn6 error:&err];
            if (!g_q6k_sgemv) {
                fprintf(stderr, "Metal: q6k pipeline creation failed: %s\n",
                        [[err localizedDescription] UTF8String]);
            }
        }
        id<MTLFunction> fnq2 = [g_lib newFunctionWithName:@"q2k_sgemv_row"];
        if (fnq2) {
            g_q2k_sgemv = [g_device newComputePipelineStateWithFunction:fnq2 error:&err];
            if (!g_q2k_sgemv) {
                fprintf(stderr, "Metal: q2k pipeline creation failed: %s\n",
                        [[err localizedDescription] UTF8String]);
            }
        }
        id<MTLFunction> fnsg = [g_lib newFunctionWithName:@"swiglu_inplace"];
        if (fnsg) {
            g_swiglu = [g_device newComputePipelineStateWithFunction:fnsg error:&err];
            if (!g_swiglu)
                fprintf(stderr, "Metal: swiglu pipeline creation failed: %s\n",
                        [[err localizedDescription] UTF8String]);
        }
        #define LOAD_PSO(var, name) do { \
            id<MTLFunction> _f = [g_lib newFunctionWithName:@name]; \
            if (_f) { var = [g_device newComputePipelineStateWithFunction:_f error:&err]; \
                if (!var) fprintf(stderr, "Metal: " name " pipeline failed: %s\n", [[err localizedDescription] UTF8String]); } \
        } while(0)
        LOAD_PSO(g_rmsnorm, "rmsnorm_f32");
        LOAD_PSO(g_rope,    "rope_f32");
        LOAD_PSO(g_attn,    "attn_decode_f32");
        LOAD_PSO(g_add,     "add_inplace_f32");
        LOAD_PSO(g_copy,    "copy_f32");
        LOAD_PSO(g_top1_reduce_tiles, "top1_reduce_tiles");
        /* batched (B independent sequences) variants */
        int q6b = load_batched_psos("q6k_sgemv_row_batched", 1);
        if (q6b == 0) fprintf(stderr, "Metal: no specialized q6k batched pipelines loaded\n");
        int q2b = load_batched_psos("q2k_sgemv_row_batched", 2);
        if (q2b == 0) fprintf(stderr, "Metal: no specialized q2k batched pipelines loaded\n");
        LOAD_PSO(g_q6k_sgemv_bp, "q6k_sgemv_row_bparallel");
        LOAD_PSO(g_q6k_sgemv_bp_add, "q6k_sgemv_row_bparallel_add");
        LOAD_PSO(g_q6k_sgemv_bp_v4, "q6k_sgemv_row_bparallel_v4");
        LOAD_PSO(g_q6k_sgemv_bp_v5, "q6k_sgemv_row_bparallel_v5");
        LOAD_PSO(g_q6k_sgemv_bp_v5_g2, "q6k_sgemv_row_bparallel_v5_g2");
        LOAD_PSO(g_q6k_sgemv_bp_v5_g2_add, "q6k_sgemv_row_bparallel_v5_g2_add");
        LOAD_PSO(g_q6k_sgemv_bp_v5_g4, "q6k_sgemv_row_bparallel_v5_g4");
        LOAD_PSO(g_q6k_sgemv_bp_v6, "q6k_sgemv_row_bparallel_v6");
        LOAD_PSO(g_q6k_sgemv_bp_v5_add, "q6k_sgemv_row_bparallel_v5_add");
        LOAD_PSO(g_q6k_sgemv_bp_g2, "q6k_sgemv_row_bparallel_g2");
        LOAD_PSO(g_q6k_sgemv_simdb, "q6k_sgemv_row_simdb_batched");
        LOAD_PSO(g_q6k_rowtiled_b16, "q6k_sgemv_rowtiled_batched_b16");
        LOAD_PSO(g_q6k_rowtiled_s8, "q6k_sgemv_rowtiled_s8_batched");
        LOAD_PSO(g_q6k_argmax_b, "q6k_argmax_batched");
        LOAD_PSO(g_q6k_top1_tiles_b, "q6k_top1_tiles_batched");
        LOAD_PSO(g_q4k_rowtiled_b, "q4k_sgemv_rowtiled_batched");
        LOAD_PSO(g_rmsnorm_b,   "rmsnorm_f32_batched");
        LOAD_PSO(g_rope_b,      "rope_f32_batched");
        LOAD_PSO(g_attn_b,      "attn_decode_f32_batched");
        LOAD_PSO(g_kv_scatter,  "kv_scatter_f32");
        LOAD_PSO(g_kv_rope_scatter, "kv_rope_scatter_f32");
        LOAD_PSO(g_argmax_b,    "argmax_f32_batched");
        #undef LOAD_PSO

        /* V12: coalesced Q4_K pipeline */
        { id<MTLFunction> fn = [g_lib newFunctionWithName:@"q4k_sgemv_row_coalesced"];
          if (fn) { g_q4k_coalesced = [g_device newComputePipelineStateWithFunction:fn error:&err];
            if (g_q4k_coalesced) fprintf(stderr, "  Metal: loaded q4k_sgemv_row_coalesced\n");
          }
        }

        /* Qwen3.5-specific kernels */
        #define LOAD_PSO_Q(var, name) do { \
            id<MTLFunction> _f = [g_lib newFunctionWithName:@name]; \
            if (_f) { var = [g_device newComputePipelineStateWithFunction:_f error:&err]; \
                if (!var) fprintf(stderr, "Metal: " name " pipeline failed: %s\n", [[err localizedDescription] UTF8String]); \
                else fprintf(stderr, "  Metal: loaded %s\n", name); \
            } else fprintf(stderr, "  Metal: %s not found in library\n", name); \
        } while(0)
        LOAD_PSO_Q(g_rope_half,    "rope_half_f32");
        LOAD_PSO_Q(g_rmsnorm_ph,   "rmsnorm_per_head_f32");
        LOAD_PSO_Q(g_sigmoid_gate, "sigmoid_gate_inplace_f32");
        LOAD_PSO_Q(g_split_qgate,  "split_qgate_f32");
        LOAD_PSO_Q(g_attn_gated,   "attn_decode_gated_f32");
        #undef LOAD_PSO_Q

        if (model_base && model_size) {
            g_model_base = model_base;
            g_model_size = model_size;

            size_t maxBuf = g_device.maxBufferLength;
            const uint8_t* p = (const uint8_t*)model_base;
            int idx = 0;
            if (model_size <= maxBuf) {
                g_model_chunks[0] = [g_device
                    newBufferWithBytesNoCopy:(void*)p
                                      length:model_size
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
                if (!g_model_chunks[0]) {
                    fprintf(stderr, "Metal: failed to wrap single buffer (%zu bytes)\n", model_size);
                    return -1;
                }
                idx = 1;
            } else {
                for (size_t base_off = 0; base_off < model_size; base_off += STRATUM_METAL_STEP) {
                    if (idx >= STRATUM_METAL_MAX_CHUNKS) {
                        fprintf(stderr, "Metal: model exceeds %d chunks; raise MAX_CHUNKS\n",
                                STRATUM_METAL_MAX_CHUNKS);
                        return -1;
                    }
                    size_t this_size = STRATUM_METAL_CHUNK_SIZE;
                    if (base_off + this_size > model_size) this_size = model_size - base_off;
                    if (this_size > maxBuf) this_size = maxBuf;
                    g_model_chunks[idx] = [g_device
                        newBufferWithBytesNoCopy:(void*)(p + base_off)
                                          length:this_size
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
                    if (!g_model_chunks[idx]) {
                        fprintf(stderr, "Metal: failed to wrap chunk %d (off=%zu size=%zu)\n",
                                idx, base_off, this_size);
                        return -1;
                    }
                    idx++;

                    if (base_off + this_size >= model_size) break;
                }
            }
            g_model_n_chunks = idx;
            fprintf(stderr,
                "  Metal: registered %.1f GB mmap'd model as %d zero-copy chunk%s "
                "(maxBufferLength=%.1f GB)\n",
                (double)model_size / (1024.0 * 1024.0 * 1024.0),
                g_model_n_chunks,
                g_model_n_chunks == 1 ? "" : "s",
                (double)maxBuf / (1024.0 * 1024.0 * 1024.0));
        }
    }
    return 0;
}

void stratum_metal_shutdown(void) {
    stratum_metal_staged_wait();

    g_q4k_sgemv = nil;
    g_q5k_sgemv = nil;
    g_q6k_sgemv = nil;
    g_xbuf = nil;
    g_ybuf = nil;
    g_xbuf_size = 0;
    g_ybuf_size = 0;
    g_lib = nil;
    for (int i = 0; i < g_model_n_chunks; i++) g_model_chunks[i] = nil;
    g_model_n_chunks = 0;
    for (int i = 0; i < g_n_staging; i++) g_staging[i] = nil;
    g_n_staging = 0;
    g_xbatch = nil; g_ybatch = nil; g_xbatch_size = 0; g_ybatch_size = 0;
    g_queue = nil;
    g_device = nil;
}

static int find_chunk(uint64_t file_off, size_t tensor_size,
                      id<MTLBuffer>* buf_out, uint64_t* off_out) {

    if (g_model_n_chunks == 1) {
        if (file_off + tensor_size > g_model_size) return -1;
        *buf_out = g_model_chunks[0];
        *off_out = file_off;
        return 0;
    }

    for (int i = g_model_n_chunks - 1; i >= 0; i--) {
        uint64_t base   = (uint64_t)i * STRATUM_METAL_STEP;
        uint64_t length = (uint64_t)g_model_chunks[i].length;
        if (file_off >= base && file_off + tensor_size <= base + length) {
            *buf_out = g_model_chunks[i];
            *off_out = file_off - base;
            return 0;
        }
    }
    fprintf(stderr, "Metal: no chunk contains tensor [off=%llu, size=%zu]\n",
            (unsigned long long)file_off, tensor_size);
    return -1;
}

static int dispatch_sgemv(id<MTLComputePipelineState> pso,
                          uint64_t weight_offset,
                          size_t   tensor_bytes,
                          const float* x, float* y,
                          int N, int K) {
    @autoreleasepool {
        if (!pso) return -1;
        id<MTLBuffer> wbuf;
        uint64_t      woff;
        if (find_chunk(weight_offset, tensor_bytes, &wbuf, &woff) != 0) {
            return -1;
        }

        size_t x_bytes = (size_t)K * sizeof(float);
        size_t y_bytes = (size_t)N * sizeof(float);

        if (x_bytes > g_xbuf_size) {
            g_xbuf = [g_device newBufferWithLength:x_bytes
                                           options:MTLResourceStorageModeShared];
            g_xbuf_size = x_bytes;
        }
        if (y_bytes > g_ybuf_size) {
            g_ybuf = [g_device newBufferWithLength:y_bytes
                                           options:MTLResourceStorageModeShared];
            g_ybuf_size = y_bytes;
        }
        if (!g_xbuf || !g_ybuf) return -1;
        memcpy([g_xbuf contents], x, x_bytes);

        uint32_t K_u32 = (uint32_t)K;
        id<MTLCommandBuffer>      cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:wbuf    offset:woff atIndex:0];
        [enc setBuffer:g_xbuf  offset:0    atIndex:1];
        [enc setBuffer:g_ybuf  offset:0    atIndex:2];
        [enc setBytes:&K_u32   length:sizeof(uint32_t) atIndex:3];

        MTLSize grid_size = MTLSizeMake((NSUInteger)N, 1, 1);
        MTLSize tg_size   = MTLSizeMake(64, 1, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
        [enc endEncoding];
        struct timespec _d0, _d1;
        clock_gettime(CLOCK_MONOTONIC, &_d0);
        [cmd commit];
        [cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &_d1);
        g_n_dispatch++;
        g_dispatch_secs += (_d1.tv_sec-_d0.tv_sec)+(_d1.tv_nsec-_d0.tv_nsec)/1e9;

        memcpy(y, [g_ybuf contents], y_bytes);
        return 0;
    }
}

void stratum_metal_dispatch_stats(long* n_out, double* secs_out) {
    if (n_out) *n_out = g_n_dispatch;
    if (secs_out) *secs_out = g_dispatch_secs;
}

/* V15: Set model base for ephemeral dispatch (no chunk registration) */
void stratum_metal_set_model_base(const void* model_base, size_t model_size) {
    g_model_base = model_base;
    g_model_size = model_size;
}

/* Multiple Q4_K matmuls sharing input x, encoded into ONE command buffer
 * with ONE wait — collapses N dispatch round-trips into 1. All same K. */
int stratum_metal_q4k_sgemv_group(const uint64_t* offsets, const size_t* tbytes,
                                  const float* x, float* const* ys,
                                  const int* Ns, int count, int K) {
    if (!g_device || !g_q4k_sgemv || count < 1) return -1;
    @autoreleasepool {
        size_t total_N = 0;
        for (int i = 0; i < count; i++) total_N += (size_t)Ns[i];
        size_t x_bytes = (size_t)K * sizeof(float);
        size_t y_bytes = total_N * sizeof(float);
        if (x_bytes > g_xbuf_size) {
            g_xbuf = [g_device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            g_xbuf_size = x_bytes;
        }
        if (y_bytes > g_ygroup_size) {
            g_ygroup = [g_device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
            g_ygroup_size = y_bytes;
        }
        if (!g_xbuf || !g_ygroup) return -1;
        memcpy([g_xbuf contents], x, x_bytes);
        uint32_t K_u32 = (uint32_t)K;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        size_t yoff = 0;
        for (int i = 0; i < count; i++) {
            id<MTLBuffer> wbuf; uint64_t woff;
            if (find_chunk(offsets[i], tbytes[i], &wbuf, &woff) != 0) return -1;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_q4k_sgemv];
            [enc setBuffer:wbuf    offset:woff atIndex:0];
            [enc setBuffer:g_xbuf  offset:0    atIndex:1];
            [enc setBuffer:g_ygroup offset:yoff atIndex:2];
            [enc setBytes:&K_u32   length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Ns[i],1,1)
                 threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
            yoff += (size_t)Ns[i] * sizeof(float);
        }
        struct timespec _d0, _d1;
        clock_gettime(CLOCK_MONOTONIC, &_d0);
        [cmd commit];
        [cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &_d1);
        g_n_dispatch++;
        g_dispatch_secs += (_d1.tv_sec-_d0.tv_sec)+(_d1.tv_nsec-_d0.tv_nsec)/1e9;

        const uint8_t* yb = (const uint8_t*)[g_ygroup contents];
        size_t off = 0;
        for (int i = 0; i < count; i++) {
            memcpy(ys[i], yb + off, (size_t)Ns[i] * sizeof(float));
            off += (size_t)Ns[i] * sizeof(float);
        }
        return 0;
    }
}

/* Multi-type group: multiple matmuls with different quantization types,
 * sharing the same input x, in ONE command buffer with ONE wait.
 * types[i]: 12=Q4_K, 13=Q5_K, 14=Q6_K.
 * This is the V2 double-buffer enabler for SSM layers. */
int stratum_metal_ssm_group(const uint64_t* offsets, const size_t* tbytes,
                            const int* types, const float* x, float* const* ys,
                            const int* Ns, int count, int K) {
    if (!g_device || !g_q4k_sgemv || count < 1) {
        if (getenv("STRATUM_GPU_DEBUG"))
            fprintf(stderr, "  [GPU_DEBUG] ssm_group early exit: device=%d q4k=%d count=%d\n",
                    g_device!=nil, g_q4k_sgemv!=nil, count);
        return -1;
    }
    @autoreleasepool {
        size_t total_N = 0;
        for (int i = 0; i < count; i++) total_N += (size_t)Ns[i];
        size_t x_bytes = (size_t)K * sizeof(float);
        size_t y_bytes = total_N * sizeof(float);
        if (x_bytes > g_xbuf_size) {
            g_xbuf = [g_device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            g_xbuf_size = x_bytes;
        }
        if (y_bytes > g_ygroup_size) {
            g_ygroup = [g_device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
            g_ygroup_size = y_bytes;
        }
        if (!g_xbuf || !g_ygroup) return -1;
        memcpy([g_xbuf contents], x, x_bytes);
        uint32_t K_u32 = (uint32_t)K;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        size_t yoff = 0;
        /* V15: ephemeral weight buffers — auto-release at end of autoreleasepool.
         * Uses newBufferWithBytes (copy) to avoid wiring mmap pages. */
        __block id<MTLBuffer> ephemeral_wbufs[8] = {nil};
        for (int i = 0; i < count; i++) {
            id<MTLBuffer> wbuf; uint64_t woff;
            if (find_chunk(offsets[i], tbytes[i], &wbuf, &woff) != 0) {
                /* V15: copy weight data into GPU buffer (no mmap page wiring) */
                if (!g_model_base || offsets[i] + tbytes[i] > g_model_size) {
                    if (getenv("STRATUM_GPU_DEBUG"))
                        fprintf(stderr, "  [GPU_DEBUG] ssm_group ephemeral: model_base=%p size=%zu off=%llu tb=%zu\n",
                                g_model_base, g_model_size,
                                (unsigned long long)offsets[i], tbytes[i]);
                    return -1;
                }
                const void* src = (const uint8_t*)g_model_base + offsets[i];
                wbuf = [g_device newBufferWithBytes:src length:tbytes[i]
                              options:MTLResourceStorageModeShared];
                if (!wbuf) {
                    if (getenv("STRATUM_GPU_DEBUG"))
                        fprintf(stderr, "  [GPU_DEBUG] ssm_group ephemeral buf failed: off=%llu size=%zu\n",
                                (unsigned long long)offsets[i], tbytes[i]);
                    return -1;
                }
                ephemeral_wbufs[i] = wbuf;
                woff = 0;
            }
            id<MTLComputePipelineState> pso;
            switch (types[i]) {
                case 12: pso = g_q4k_sgemv; break;
                case 13: pso = g_q5k_sgemv; break;
                case 14: pso = g_q6k_sgemv; break;
                default:
                    if (getenv("STRATUM_GPU_DEBUG"))
                        fprintf(stderr, "  [GPU_DEBUG] ssm_group unsupported type %d at i=%d\n", types[i], i);
                    return -1;
            }
            if (!pso) {
                if (getenv("STRATUM_GPU_DEBUG"))
                    fprintf(stderr, "  [GPU_DEBUG] ssm_group pso nil for type %d\n", types[i]);
                return -1;
            }
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf    offset:woff atIndex:0];
            [enc setBuffer:g_xbuf  offset:0    atIndex:1];
            [enc setBuffer:g_ygroup offset:yoff atIndex:2];
            [enc setBytes:&K_u32   length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Ns[i],1,1)
                 threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
            yoff += (size_t)Ns[i] * sizeof(float);
        }
        struct timespec _d0, _d1;
        clock_gettime(CLOCK_MONOTONIC, &_d0);
        [cmd commit];
        [cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &_d1);
        g_n_dispatch++;
        g_dispatch_secs += (_d1.tv_sec-_d0.tv_sec)+(_d1.tv_nsec-_d0.tv_nsec)/1e9;

        const uint8_t* yb = (const uint8_t*)[g_ygroup contents];
        size_t off = 0;
        for (int i = 0; i < count; i++) {
            memcpy(ys[i], yb + off, (size_t)Ns[i] * sizeof(float));
            off += (size_t)Ns[i] * sizeof(float);
        }
        return 0;
    }
}

/* ===== V6: Q4_K -> F16 pre-decode ===== */
#include "stratum_q4k.h"
#include "stratum_q6k.h"

static inline uint16_t float_to_fp16(float v) {
#if defined(__ARM_FP16_FORMAT_IEEE) && (defined(__ARM_NEON) || defined(__aarch64__))
    __fp16 h = (__fp16)v;
    uint16_t r; memcpy(&r, &h, 2); return r;
#else
    /* Software fp32->fp16 (IEEE 754) */
    uint32_t f; memcpy(&f, &v, 4);
    uint16_t sign = (f >> 31) & 1;
    int32_t exp = (f >> 23) & 0xFF;
    uint32_t mant = f & 0x7FFFFF;
    uint16_t h_sign = sign << 15;
    if (exp == 0xFF) {
        return h_sign | 0x7C00 | (mant ? 0x200 : 0);
    }
    exp -= 127;
    if (exp >= 16) return h_sign | 0x7C00;
    if (exp < -14) {
        int shift = -14 - exp;
        if (shift < 24) mant = mant | 0x800000;
        mant >>= shift;
        return h_sign | (mant & 0x3FF);
    }
    uint16_t h_exp = (exp + 15) << 10;
    return h_sign | h_exp | (mant >> 13);
#endif
}

/* g_flog_ref points to the real g_flog once it is allocated */
/* (V6 pre-decode / V9 Q4_0 conversion machinery removed — forbidden by
 * project boundary: pre-decoding pins GPU weight buffers and Q4_K→Q4_0
 * re-encoding loses precision.) */

static __unsafe_unretained id<MTLBuffer> g_flog_ref = nil;

int stratum_metal_get_last_token(void) {
    if (!g_flog_ref) return -1;
    uint32_t* p = (uint32_t*)[g_flog_ref contents];
    return (int)p[0];
}

static id<MTLBuffer> g_fx=nil, g_fxn=nil, g_fq=nil, g_fk=nil, g_fv=nil,
                     g_fattn=nil, g_ftmp=nil, g_fg=nil, g_fu=nil, g_fa=nil, g_flog=nil;
static size_t g_fcap_h=0, g_fcap_ff=0, g_fcap_v=0, g_fcap_qh=0, g_fcap_kh=0;
static id<MTLBuffer> g_fkv_k=nil, g_fkv_v=nil;
static size_t g_fkv_cap=0;

/* Whole-token forward on GPU: one command buffer, x stays resident across all
 * layers, KV cache on GPU. Returns logits to CPU (1 sync/token). */
int stratum_metal_forward(const StratumMetalLayer* layers, int n_layers,
                          unsigned long long out_norm_off,
                          unsigned long long lm_off, unsigned long lm_tb, int lm_is_q6,
                          const float* x_in, float* logits_out,
                          int H, int Hd, int Nq, int Nk, int Ff, int V,
                          int rope_dim, int position, float rope_theta,
                          float rms_eps, int kv_len, int max_kv) {
    if (!g_device || !g_q4k_sgemv || !g_q6k_sgemv || !g_swiglu || !g_rmsnorm
        || !g_rope || !g_attn || !g_add || !g_copy) return -1;
    @autoreleasepool {
        size_t hb=(size_t)H*4, ffb=(size_t)Ff*4, vb=(size_t)V*4,
               qhb=(size_t)Nq*Hd*4, khb=(size_t)Nk*Hd*4;
        if (hb > g_fcap_h) {
            g_fx   = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_fxn  = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_ftmp = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_fcap_h = hb;
        }
        if (qhb > g_fcap_qh) {
            g_fq    = [g_device newBufferWithLength:qhb options:MTLResourceStorageModeShared];
            g_fattn = [g_device newBufferWithLength:qhb options:MTLResourceStorageModeShared];
            g_fcap_qh = qhb;
        }
        if (khb > g_fcap_kh) {
            g_fk = [g_device newBufferWithLength:khb options:MTLResourceStorageModeShared];
            g_fv = [g_device newBufferWithLength:khb options:MTLResourceStorageModeShared];
            g_fcap_kh = khb;
        }
        if (ffb > g_fcap_ff) {
            g_fg = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_fu = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_fa = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_fcap_ff = ffb;
        }
        if (vb > g_fcap_v) {
            g_flog = [g_device newBufferWithLength:vb options:MTLResourceStorageModeShared];
            g_flog_ref = g_flog;
            g_fcap_v = vb;
        }
        size_t kvb = (size_t)n_layers*max_kv*Nk*Hd*4;
        if (kvb > g_fkv_cap) {
            g_fkv_k = [g_device newBufferWithLength:kvb options:MTLResourceStorageModeShared];
            g_fkv_v = [g_device newBufferWithLength:kvb options:MTLResourceStorageModeShared];
            g_fkv_cap = kvb;
        }
        if (!g_fx||!g_fxn||!g_ftmp||!g_fq||!g_fattn||!g_fk||!g_fv||!g_fg||!g_fu||!g_fa||!g_flog||!g_fkv_k||!g_fkv_v) return -1;
        memcpy([g_fx contents], x_in, hb);

        uint32_t Hu=(uint32_t)H, Ffu=(uint32_t)Ff, Vu=(uint32_t)V, Hdu=(uint32_t)Hd,
                 Nqu=(uint32_t)Nq, Nku=(uint32_t)Nk, rdu=(uint32_t)rope_dim;
        uint32_t kvn=(uint32_t)(kv_len+1);
        float scale = 1.0f/sqrt((float)Hd);
        int posi = position;
        static uint s_tg_size = 0;
        if (!s_tg_size) { const char* e=getenv("STRATUM_GPU_TG"); s_tg_size = e ? (uint)atoi(e) : 64; if (s_tg_size < 32) s_tg_size = 32; if (s_tg_size > 256) s_tg_size = 256; }
        size_t kv_layer_stride = (size_t)max_kv*Nk*Hd;
        size_t kv_slot = (size_t)kv_len*Nk*Hd;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        #define PSO_TY(ty) ((ty)==14 ? g_q6k_sgemv : (ty)==13 ? g_q5k_sgemv : g_q4k_sgemv)
        #define MM(ty, woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            if (g_q4k_coalesced && (ty)==12 && getenv("STRATUM_COALESCE") && (Nrows) >= 8) { \
                id<MTLBuffer> _wb; uint64_t _wo; \
                if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
                id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
                [_e setComputePipelineState:g_q4k_coalesced]; \
                [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
                [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
                uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
                [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; \
                [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(((Nrows)+7)/8),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; \
                [_e endEncoding]; \
            } else { \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:PSO_TY(ty)]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; uint32_t _K=(uint32_t)(Kdim); [_e setBytes:&_K length:4 atIndex:3]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),1,1) threadsPerThreadgroup:MTLSizeMake(s_tg_size,1,1)]; \
            [_e endEncoding]; } } while(0)
        #define MMO(ty, woff, wtb, xbuf, ybuf, yoff, Nrows, Kdim) do { \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:PSO_TY(ty)]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:(yoff) atIndex:2]; uint32_t _K=(uint32_t)(Kdim); [_e setBytes:&_K length:4 atIndex:3]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),1,1) threadsPerThreadgroup:MTLSizeMake(s_tg_size,1,1)]; \
            [_e endEncoding]; } while(0)
        #define NORM(gain_off, xbuf, ybuf) do { \
            id<MTLBuffer> _gb; uint64_t _go; if (find_chunk((gain_off),(size_t)H*4,&_gb,&_go)!=0) return -1; \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_rmsnorm]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:0]; [_e setBuffer:_gb offset:_go atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; [_e setBytes:&Hu length:4 atIndex:3]; [_e setBytes:&rms_eps length:4 atIndex:4]; \
            [_e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [_e endEncoding]; } while(0)
        #define ROPE(buf, nheads) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:g_rope]; \
            [_e setBuffer:(buf) offset:0 atIndex:0]; [_e setBytes:&Hdu length:4 atIndex:1]; [_e setBytes:&rdu length:4 atIndex:2]; \
            [_e setBytes:&posi length:4 atIndex:3]; [_e setBytes:&rope_theta length:4 atIndex:4]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)((nheads)*(rope_dim/2)),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [_e endEncoding]; } while(0)
        #define ROPEO(buf, boff, nheads) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:g_rope]; \
            [_e setBuffer:(buf) offset:(boff) atIndex:0]; [_e setBytes:&Hdu length:4 atIndex:1]; [_e setBytes:&rdu length:4 atIndex:2]; \
            [_e setBytes:&posi length:4 atIndex:3]; [_e setBytes:&rope_theta length:4 atIndex:4]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)((nheads)*(rope_dim/2)),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [_e endEncoding]; } while(0)
        #define ELEM(pso, abuf, aoff, bbuf, boff, nn) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:(pso)]; \
            [_e setBuffer:(abuf) offset:(aoff) atIndex:0]; [_e setBuffer:(bbuf) offset:(boff) atIndex:1]; \
            uint32_t _n=(uint32_t)(nn); [_e setBytes:&_n length:4 atIndex:2]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)(nn),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [_e endEncoding]; } while(0)

        { const char* e = getenv("STRATUM_FWD_NL"); if (e) { int v=atoi(e); if (v>=0 && v<n_layers) n_layers=v; } }
        for (int L=0; L<n_layers; L++) {
            const StratumMetalLayer* ly=&layers[L];
            NORM(ly->attn_norm_off, g_fx, g_fxn);
            MM(ly->q_ty, ly->q_off, ly->q_tb, g_fxn, g_fq, Nq*Hd, H);
            /* k,v written straight into the KV cache slot (no copy encoder) */
            size_t koff = ((size_t)L*kv_layer_stride + kv_slot)*4;
            MMO(ly->k_ty, ly->k_off, ly->k_tb, g_fxn, g_fkv_k, koff, Nk*Hd, H);
            MMO(ly->v_ty, ly->v_off, ly->v_tb, g_fxn, g_fkv_v, koff, Nk*Hd, H);
            ROPE(g_fq, Nq); ROPEO(g_fkv_k, koff, Nk);
            /* attention */
            { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:g_attn];
              [e setBuffer:g_fq offset:0 atIndex:0];
              [e setBuffer:g_fkv_k offset:(size_t)L*kv_layer_stride*4 atIndex:1];
              [e setBuffer:g_fkv_v offset:(size_t)L*kv_layer_stride*4 atIndex:2];
              [e setBuffer:g_fattn offset:0 atIndex:3];
              [e setBytes:&Hdu length:4 atIndex:4]; [e setBytes:&Nqu length:4 atIndex:5];
              [e setBytes:&Nku length:4 atIndex:6]; [e setBytes:&kvn length:4 atIndex:7];
              [e setBytes:&scale length:4 atIndex:8];
              [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Nq,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [e endEncoding]; }
            MM(ly->o_ty, ly->o_off, ly->o_tb, g_fattn, g_ftmp, H, Nq*Hd);
            ELEM(g_add, g_fx, 0, g_ftmp, 0, H);
            if (L==0 && getenv("STRATUM_FWDBG")) {
                const float* a=(const float*)[g_fattn contents];
                const float* t=(const float*)[g_ftmp contents];
                const float* x=(const float*)[g_fx contents];
                const float* kk=(const float*)[g_fk contents];
                const float* vv=(const float*)[g_fv contents];
                fprintf(stderr,"  [L0] kvn=%u k[0]=%.3f v[0]=%.3f attn[0]=%.4f o[0]=%.4f x[0]=%.4f\n",
                        kvn, kk[0], vv[0], a[0],t[0],x[0]);
            }
            /* FFN */
            NORM(ly->ffn_norm_off, g_fx, g_fxn);
            MM(ly->gate_ty, ly->gate_off, ly->gate_tb, g_fxn, g_fg, Ff, H);
            MM(ly->up_ty,   ly->up_off,   ly->up_tb,   g_fxn, g_fu, Ff, H);
            { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:g_swiglu];
              [e setBuffer:g_fg offset:0 atIndex:0]; [e setBuffer:g_fu offset:0 atIndex:1];
              [e setBuffer:g_fa offset:0 atIndex:2]; [e setBytes:&Ffu length:4 atIndex:3];
              [e dispatchThreads:MTLSizeMake((NSUInteger)Ff,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding]; }

            /* V7: Sparse down_proj — skip blocks where |fa| < threshold */
            static id<MTLComputePipelineState> s_sparse_sgemv = nil;
            static id<MTLComputePipelineState> s_blockmax = nil;
            static id<MTLBuffer> s_blockmax_buf = nil;
            static id<MTLBuffer> s_globalmax_buf = nil;
            static size_t s_blockmax_cap = 0;
            static int s_sparse_inited = 0;
            static float s_sparse_thresh = 0.01f;
            if (!s_sparse_inited) {
                s_sparse_inited = 1;
                const char* se = getenv("STRATUM_SPARSE");
                if (se) s_sparse_thresh = (float)atof(se);
                if (getenv("STRATUM_SPARSE")) {
                    id<MTLFunction> fn1 = [g_lib newFunctionWithName:@"q4k_sgemv_row_sparse"];
                    if (fn1) s_sparse_sgemv = [g_device newComputePipelineStateWithFunction:fn1 error:nil];
                    id<MTLFunction> fn2 = [g_lib newFunctionWithName:@"compute_x_block_max"];
                    if (fn2) s_blockmax = [g_device newComputePipelineStateWithFunction:fn2 error:nil];
                    if (s_sparse_sgemv && s_blockmax)
                        fprintf(stderr, "  V7: sparse down_proj enabled (threshold=%.3f)\n", s_sparse_thresh);
                }
            }

            if (s_sparse_sgemv && s_blockmax && ly->down_ty == 12) {
                /* Compute block max of fa */
                size_t bm_bytes = (size_t)(Ff / 32) * sizeof(float);
                if (bm_bytes > s_blockmax_cap) {
                    s_blockmax_buf = [g_device newBufferWithLength:bm_bytes options:MTLResourceStorageModeShared];
                    s_globalmax_buf = [g_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
                    s_blockmax_cap = bm_bytes;
                }
                { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
                  [e setComputePipelineState:s_blockmax];
                  [e setBuffer:g_fa offset:0 atIndex:0];
                  [e setBuffer:s_blockmax_buf offset:0 atIndex:1];
                  [e setBuffer:s_globalmax_buf offset:0 atIndex:2];
                  [e setBytes:&Ffu length:4 atIndex:3];
                  [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                  [e endEncoding]; }
                /* Sparse sgemv */
                id<MTLBuffer> _wb; uint64_t _wo;
                if (find_chunk(ly->down_off, ly->down_tb, &_wb, &_wo) == 0) {
                    id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
                    [e setComputePipelineState:s_sparse_sgemv];
                    [e setBuffer:_wb offset:_wo atIndex:0];
                    [e setBuffer:g_fa offset:0 atIndex:1];
                    [e setBuffer:g_ftmp offset:0 atIndex:2];
                    uint32_t _K=(uint32_t)Ff; [e setBytes:&_K length:4 atIndex:3];
                    [e setBuffer:s_blockmax_buf offset:0 atIndex:4];
                    /* read global max from buffer */
                    float gm = *(float*)[s_globalmax_buf contents];
                    [e setBytes:&gm length:4 atIndex:5];
                    [e setBytes:&s_sparse_thresh length:4 atIndex:6];
                    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)H,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                    [e endEncoding];
                } else {
                    MM(ly->down_ty, ly->down_off, ly->down_tb, g_fa, g_ftmp, H, Ff);
                }
            } else {
                MM(ly->down_ty, ly->down_off, ly->down_tb, g_fa, g_ftmp, H, Ff);
            }
            ELEM(g_add, g_fx, 0, g_ftmp, 0, H);
        }
        NORM(out_norm_off, g_fx, g_fxn);

        /* V-opt: fused argmax on GPU — skip 128KB logits transfer to CPU.
         * Only when caller passes logits_out=NULL (greedy decode). */
        if (!logits_out && g_q4k_argmax_b) {
            id<MTLBuffer> _wb; uint64_t _wo;
            if (find_chunk(lm_off, lm_tb, &_wb, &_wo) == 0) {
                id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
                [e setComputePipelineState:(lm_is_q6 ? g_q6k_argmax_b : g_q4k_argmax_b)];
                [e setBuffer:_wb offset:_wo atIndex:0];
                [e setBuffer:g_fxn offset:0 atIndex:1];
                [e setBuffer:g_flog offset:0 atIndex:2]; /* reuse as token output */
                uint32_t _K=(uint32_t)H, _V=(uint32_t)V, _B=1;
                [e setBytes:&_K length:4 atIndex:3]; [e setBytes:&_V length:4 atIndex:4]; [e setBytes:&_B length:4 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            } else {
                MM(lm_is_q6 ? 14 : 12, lm_off, lm_tb, g_fxn, g_flog, V, H);
            }
        } else {
            MM(lm_is_q6 ? 14 : 12, lm_off, lm_tb, g_fxn, g_flog, V, H);
        }

        struct timespec _d0,_d1; clock_gettime(CLOCK_MONOTONIC,&_d0);
        [cmd commit]; [cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC,&_d1);
        g_n_dispatch++; g_dispatch_secs += (_d1.tv_sec-_d0.tv_sec)+(_d1.tv_nsec-_d0.tv_nsec)/1e9;
        if (logits_out) {
            memcpy(logits_out, [g_flog contents], vb);
        } else if (g_q4k_argmax_b) {
            /* Fused argmax: g_flog[0] contains the token id (uint) */
            /* Caller reads it via stratum_metal_get_last_token() */
        }
        if (getenv("STRATUM_FWDBG")) {
            const float* xx = (const float*)[g_fx contents];
            const float* lg = logits_out;
            float mx=lg[0]; int mi=0;
            for (int i=1;i<V;i++) if(lg[i]>mx){mx=lg[i];mi=i;}
            fprintf(stderr, "  [fwdbg] x[0..2]=%.4f %.4f %.4f  logits[0..2]=%.4f %.4f %.4f  argmax=%d(%.4f)\n",
                    xx[0],xx[1],xx[2], lg[0],lg[1],lg[2], mi, mx);
        }
        #undef MM
        #undef NORM
        #undef ROPE
        #undef ELEM
        return 0;
    }
}

/* ===== Batched whole-token forward: B INDEPENDENT sequences, one command
 * buffer, one sync. One weight stream per matmul serves all B streams ->
 * aggregate throughput scales with B (the dimension llama.cpp can't touch at
 * low binding RAM). KV cache laid out [n_layers][B][max_kv][Nk*Hd]. Each
 * sequence has its own position[b] and kvlen[b]. B=1 is bit-exact to
 * stratum_metal_forward. */

static id<MTLBuffer> g_bx=nil, g_bxn=nil, g_btmp=nil, g_bq=nil, g_battn=nil,
                     g_bk=nil, g_bv=nil, g_bg=nil, g_bu=nil, g_ba=nil, g_blog=nil;
static id<MTLBuffer> g_bkv_k=nil, g_bkv_v=nil;
static id<MTLBuffer> g_bpos=nil, g_bslot=nil, g_bkvp1=nil;
static id<MTLBuffer> g_btok=nil;
static id<MTLBuffer> g_btile_vals=nil, g_btile_idxs=nil;
static size_t g_bcap_h=0, g_bcap_ff=0, g_bcap_qh=0, g_bcap_kh=0,
              g_bcap_v=0, g_bkv_cap=0, g_bcap_idx=0, g_bcap_tiles=0;

int stratum_metal_forward_batched(const StratumMetalLayer* layers, int n_layers,
                          unsigned long long out_norm_off,
                          unsigned long long lm_off, unsigned long lm_tb, int lm_is_q6,
                          const float* x_in, float* logits_out, int* tokens_out,
                          int H, int Hd, int Nq, int Nk, int Ff, int V,
                          int rope_dim, const int* positions, float rope_theta,
                          float rms_eps, const int* kvlens, int max_kv, int B) {
    if (!g_device || B < 1 || B > 32 || !g_q4k_sgemv_b[B]
        || !g_q6k_sgemv_b[B] || !g_swiglu || !g_rmsnorm_b
            || !g_rope_b || !g_attn_b || !g_kv_scatter || !g_add) return -1;
    if (tokens_out && !g_argmax_b) return -1;
    if (B < 1 || B > 32) return -1;
    @autoreleasepool {
        size_t hb=(size_t)B*H*4, ffb=(size_t)B*Ff*4, vb=(size_t)B*V*4,
               qhb=(size_t)B*Nq*Hd*4, khb=(size_t)B*Nk*Hd*4, idxb=(size_t)B*4;
        if (hb > g_bcap_h) {
            g_bx   = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_bxn  = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_btmp = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_bcap_h = hb;
        }
        if (qhb > g_bcap_qh) {
            g_bq    = [g_device newBufferWithLength:qhb options:MTLResourceStorageModeShared];
            g_battn = [g_device newBufferWithLength:qhb options:MTLResourceStorageModeShared];
            g_bcap_qh = qhb;
        }
        if (khb > g_bcap_kh) {
            g_bk = [g_device newBufferWithLength:khb options:MTLResourceStorageModeShared];
            g_bv = [g_device newBufferWithLength:khb options:MTLResourceStorageModeShared];
            g_bcap_kh = khb;
        }
        if (ffb > g_bcap_ff) {
            g_bg = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_bu = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_ba = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_bcap_ff = ffb;
        }
        if (vb > g_bcap_v) {
            g_blog = [g_device newBufferWithLength:vb options:MTLResourceStorageModeShared];
            g_bcap_v = vb;
        }
        if (idxb > g_bcap_idx) {
            g_bpos  = [g_device newBufferWithLength:idxb options:MTLResourceStorageModeShared];
            g_bslot = [g_device newBufferWithLength:idxb options:MTLResourceStorageModeShared];
            g_bkvp1 = [g_device newBufferWithLength:idxb options:MTLResourceStorageModeShared];
            g_btok  = [g_device newBufferWithLength:idxb options:MTLResourceStorageModeShared];
            g_bcap_idx = idxb;
        }
        size_t kvb = (size_t)n_layers*B*max_kv*Nk*Hd*4;
        if (kvb > g_bkv_cap) {
            g_bkv_k = [g_device newBufferWithLength:kvb options:MTLResourceStorageModeShared];
            g_bkv_v = [g_device newBufferWithLength:kvb options:MTLResourceStorageModeShared];
            g_bkv_cap = kvb;
        }
        uint32_t tile_rows = 256;
        const char* _top1_tile_env = getenv("STRATUM_GPU_TOP1_TILE_ROWS");
        if (_top1_tile_env) {
            unsigned long _v = strtoul(_top1_tile_env, NULL, 10);
            if (_v == 64 || _v == 128 || _v == 256 || _v == 512 || _v == 1024)
                tile_rows = (uint32_t)_v;
        }
        uint32_t ntile = (uint32_t)((V + tile_rows - 1) / tile_rows);
        size_t tileb = (size_t)B * ntile * 4;
        if (tileb > g_bcap_tiles) {
            g_btile_vals = [g_device newBufferWithLength:tileb options:MTLResourceStorageModeShared];
            g_btile_idxs = [g_device newBufferWithLength:tileb options:MTLResourceStorageModeShared];
            g_bcap_tiles = tileb;
        }
        if (!g_bx||!g_bxn||!g_btmp||!g_bq||!g_battn||!g_bk||!g_bv||!g_bg||!g_bu||!g_ba
            ||!g_blog||!g_bkv_k||!g_bkv_v||!g_bpos||!g_bslot||!g_bkvp1||!g_btok
            ||!g_btile_vals||!g_btile_idxs) return -1;

        memcpy([g_bx contents], x_in, hb);
        int* posp  = (int*)[g_bpos contents];
        int* slotp = (int*)[g_bslot contents];
        int* kvp1  = (int*)[g_bkvp1 contents];
        for (int s=0;s<B;s++){ posp[s]=positions[s]; slotp[s]=kvlens[s]; kvp1[s]=kvlens[s]+1; }

        uint32_t Hu=(uint32_t)H, Ffu=(uint32_t)Ff, Vu=(uint32_t)V, Hdu=(uint32_t)Hd,
                 Nqu=(uint32_t)Nq, Nku=(uint32_t)Nk, rdu=(uint32_t)rope_dim,
                 Bu=(uint32_t)B, mku=(uint32_t)max_kv, row_kv=(uint32_t)(Nk*Hd);
        float scale = 1.0f/sqrt((float)Hd);
        size_t kv_layer_stride = (size_t)B*max_kv*Nk*Hd;
        int use_simdb = getenv("STRATUM_GPU_BATCH_SIMDB") != NULL && B <= 16;
        int use_bparallel;
        if (use_simdb) {
            use_bparallel = 0;
        } else if (getenv("STRATUM_GPU_BATCH_PAR")) {
            use_bparallel = 1;
        } else if (getenv("STRATUM_GPU_BATCH_SERIAL")) {
            use_bparallel = 0;
        } else {
            use_bparallel = (B >= 5);
        }
        int use_bparallel_saved = use_bparallel;

        int bprof = getenv("STRATUM_GPU_BATCH_PROFILE") != NULL;
        if (bprof) g_bprof_enabled_ever = 1;
        g_bprof_sweeps++;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        NSUInteger ffn_tg = (B >= 24 && !getenv("STRATUM_GPU_BATCH_FFN_TG32_DISABLE")) ? 32 : 64;
        const char* _ffn_tg_env = getenv("STRATUM_GPU_BATCH_FFN_TG");
        if (_ffn_tg_env) {
            unsigned long _v = strtoul(_ffn_tg_env, NULL, 10);
            if (_v == 32 || _v == 64 || _v == 128 || _v == 256) ffn_tg = (NSUInteger)_v;
        }
        NSUInteger q4_dual_tg = 64;
        NSUInteger kv_tg = 64;
        const char* _kv_tg_env = getenv("STRATUM_GPU_BATCH_KV_TG");
        if (_kv_tg_env) {
            unsigned long _v = strtoul(_kv_tg_env, NULL, 10);
            if (_v == 32 || _v == 64 || _v == 128 || _v == 256) kv_tg = (NSUInteger)_v;
        }
        NSUInteger down_q6_tg = (B == 24 && !getenv("STRATUM_GPU_BATCH_DOWN_Q6_TG32_DISABLE")) ? 32 : 64;
        const char* _down_q6_tg_env = getenv("STRATUM_GPU_BATCH_DOWN_Q6_TG");
        if (_down_q6_tg_env) {
            unsigned long _v = strtoul(_down_q6_tg_env, NULL, 10);
            if (_v == 32 || _v == 64 || _v == 128 || _v == 256) down_q6_tg = (NSUInteger)_v;
        }
        NSUInteger q6_generic_tg = 64;
        const char* _q6_tg_env = getenv("STRATUM_GPU_BATCH_Q6_TG");
        if (_q6_tg_env) {
            unsigned long _v = strtoul(_q6_tg_env, NULL, 10);
            if (_v == 32 || _v == 64 || _v == 128 || _v == 256) q6_generic_tg = (NSUInteger)_v;
        }
        NSUInteger q4_g2_tg = (B >= 24 && !getenv("STRATUM_GPU_BATCH_Q4_G2_TG32_DISABLE")) ? 32 : 64;
        const char* _q4_g2_tg_env = getenv("STRATUM_GPU_BATCH_Q4_G2_TG");
        if (_q4_g2_tg_env) {
            unsigned long _v = strtoul(_q4_g2_tg_env, NULL, 10);
            if (_v == 32 || _v == 64 || _v == 128 || _v == 256) q4_g2_tg = (NSUInteger)_v;
        }

        #define PROFILE_CUT(stage_id) do { \
            if (bprof) { \
                struct timespec _p0, _p1; \
                clock_gettime(CLOCK_MONOTONIC, &_p0); \
                [cmd commit]; [cmd waitUntilCompleted]; \
                clock_gettime(CLOCK_MONOTONIC, &_p1); \
                double _dt = elapsed_secs(_p0, _p1); \
                g_bprof_secs[(stage_id)] += _dt; \
                g_bprof_calls[(stage_id)]++; \
                g_n_dispatch++; \
                g_dispatch_secs += _dt; \
                cmd = [g_queue commandBuffer]; \
            } \
        } while(0)

        #define PSO_TYB(ty) ((ty)==14 ? g_q6k_sgemv_b[B] : (ty)==12 ? g_q4k_sgemv_b[B] : nil)
        #define PSO_BPAR(ty) ((ty)==14 ? (getenv("STRATUM_GPU_BATCH_Q6_OLD") ? g_q6k_sgemv_bp : (getenv("STRATUM_GPU_BATCH_Q6_V4") ? (g_q6k_sgemv_bp_v4 ? g_q6k_sgemv_bp_v4 : g_q6k_sgemv_bp) : (((getenv("STRATUM_GPU_BATCH_Q6_V5") || (B == 24)) && g_q6k_sgemv_bp_v5) ? g_q6k_sgemv_bp_v5 : (g_q6k_sgemv_bp_v4 ? g_q6k_sgemv_bp_v4 : g_q6k_sgemv_bp)))) : (ty)==12 ? g_q4k_sgemv_bp : nil)
        #define PSO_SIMDB(ty) ((ty)==14 ? g_q6k_sgemv_simdb : (ty)==12 ? g_q4k_sgemv_simdb : nil)
        /* batched matmul: x[B][Kdim] -> y[B][Nrows], weight read once per row,
         * dequant amortized across B columns. */
        #define MMB(ty, woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            int _q4g2 = (ty)==12 && g_q4k_sgemv_bp_g2 && !getenv("STRATUM_GPU_BATCH_Q4_G2_DISABLE") && (B >= 5 || getenv("STRATUM_GPU_BATCH_Q4_G2")); \
            id<MTLComputePipelineState> _p=use_simdb ? PSO_SIMDB(ty) : (use_bparallel ? (_q4g2 ? g_q4k_sgemv_bp_g2 : PSO_BPAR(ty)) : PSO_TYB(ty)); if(!_p) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:_p]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; \
            [_e setBytes:&Bu length:4 atIndex:5]; \
            if (use_simdb) \
                [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)(B*64),1,1)]; \
            else if (use_bparallel) { \
                NSUInteger _gy = _q4g2 ? (NSUInteger)((B + 1) / 2) : (NSUInteger)B; \
                NSUInteger _tg = _q4g2 ? q4_g2_tg : ((ty)==14 ? q6_generic_tg : 64); \
                [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),_gy,1) threadsPerThreadgroup:MTLSizeMake(_tg,1,1)]; \
            } \
            else \
                [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_DUAL_Q4(woff0, wtb0, woff1, wtb1, xbuf, ybuf0, ybuf1, Nrows, Kdim) do { \
            id<MTLComputePipelineState> _pdual = (getenv("STRATUM_GPU_BATCH_FFN_FUSED_V2") && g_q4k_dual_sgemv_bp_v2) ? g_q4k_dual_sgemv_bp_v2 : g_q4k_dual_sgemv_bp; \
            if (!_pdual) return -1; \
            id<MTLBuffer> _wb0; uint64_t _wo0; \
            id<MTLBuffer> _wb1; uint64_t _wo1; \
            if (find_chunk((woff0),(wtb0),&_wb0,&_wo0)!=0) return -1; \
            if (find_chunk((woff1),(wtb1),&_wb1,&_wo1)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:_pdual]; \
            [_e setBuffer:_wb0 offset:_wo0 atIndex:0]; [_e setBuffer:_wb1 offset:_wo1 atIndex:1]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:2]; \
            [_e setBuffer:(ybuf0) offset:0 atIndex:3]; [_e setBuffer:(ybuf1) offset:0 atIndex:4]; \
            [_e setBytes:&_K length:4 atIndex:5]; [_e setBytes:&_N length:4 atIndex:6]; [_e setBytes:&Bu length:4 atIndex:7]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)B,1) threadsPerThreadgroup:MTLSizeMake(q4_dual_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_DUAL_Q4_G2(woff0, wtb0, woff1, wtb1, xbuf, ybuf0, ybuf1, Nrows, Kdim, tgval) do { \
            if (!g_q4k_dual_sgemv_bp_g2) return -1; \
            id<MTLBuffer> _wb0; uint64_t _wo0; \
            id<MTLBuffer> _wb1; uint64_t _wo1; \
            if (find_chunk((woff0),(wtb0),&_wb0,&_wo0)!=0) return -1; \
            if (find_chunk((woff1),(wtb1),&_wb1,&_wo1)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q4k_dual_sgemv_bp_g2]; \
            [_e setBuffer:_wb0 offset:_wo0 atIndex:0]; [_e setBuffer:_wb1 offset:_wo1 atIndex:1]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:2]; \
            [_e setBuffer:(ybuf0) offset:0 atIndex:3]; [_e setBuffer:(ybuf1) offset:0 atIndex:4]; \
            [_e setBytes:&_K length:4 atIndex:5]; [_e setBytes:&_N length:4 atIndex:6]; [_e setBytes:&Bu length:4 atIndex:7]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 1) / 2),1) threadsPerThreadgroup:MTLSizeMake((tgval),1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_DUAL_SWIGLU_Q4_G2(woff0, wtb0, woff1, wtb1, xbuf, ybuf, Nrows, Kdim) do { \
            if (!g_q4k_dual_swiglu_bp_g2) return -1; \
            id<MTLBuffer> _wb0; uint64_t _wo0; \
            id<MTLBuffer> _wb1; uint64_t _wo1; \
            if (find_chunk((woff0),(wtb0),&_wb0,&_wo0)!=0) return -1; \
            if (find_chunk((woff1),(wtb1),&_wb1,&_wo1)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q4k_dual_swiglu_bp_g2]; \
            [_e setBuffer:_wb0 offset:_wo0 atIndex:0]; [_e setBuffer:_wb1 offset:_wo1 atIndex:1]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:2]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:3]; \
            [_e setBytes:&_K length:4 atIndex:4]; [_e setBytes:&_N length:4 atIndex:5]; [_e setBytes:&Bu length:4 atIndex:6]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 1) / 2),1) threadsPerThreadgroup:MTLSizeMake(q4_dual_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_DUAL_Q4_G4(woff0, wtb0, woff1, wtb1, xbuf, ybuf0, ybuf1, Nrows, Kdim) do { \
            if (!g_q4k_dual_sgemv_bp_g4) return -1; \
            id<MTLBuffer> _wb0; uint64_t _wo0; \
            id<MTLBuffer> _wb1; uint64_t _wo1; \
            if (find_chunk((woff0),(wtb0),&_wb0,&_wo0)!=0) return -1; \
            if (find_chunk((woff1),(wtb1),&_wb1,&_wo1)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q4k_dual_sgemv_bp_g4]; \
            [_e setBuffer:_wb0 offset:_wo0 atIndex:0]; [_e setBuffer:_wb1 offset:_wo1 atIndex:1]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:2]; \
            [_e setBuffer:(ybuf0) offset:0 atIndex:3]; [_e setBuffer:(ybuf1) offset:0 atIndex:4]; \
            [_e setBytes:&_K length:4 atIndex:5]; [_e setBytes:&_N length:4 atIndex:6]; [_e setBytes:&Bu length:4 atIndex:7]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 3) / 4),1) threadsPerThreadgroup:MTLSizeMake(q4_dual_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_ADD(woff, wtb, xbuf, rbuf, Nrows, Kdim) do { \
            if (!g_q6k_sgemv_bp_add) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_sgemv_bp_add]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(rbuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)B,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_V5_ADD(woff, wtb, xbuf, rbuf, Nrows, Kdim) do { \
            if (!g_q6k_sgemv_bp_v5_add) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_sgemv_bp_v5_add]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(rbuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)B,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_G2(woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            if (!g_q6k_sgemv_bp_g2) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_sgemv_bp_g2]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 1) / 2),1) threadsPerThreadgroup:MTLSizeMake(down_q6_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_V5_G2(woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            if (!g_q6k_sgemv_bp_v5_g2) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_sgemv_bp_v5_g2]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 1) / 2),1) threadsPerThreadgroup:MTLSizeMake(down_q6_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_V5_G2_ADD(woff, wtb, xbuf, rbuf, Nrows, Kdim) do { \
            if (!g_q6k_sgemv_bp_v5_g2_add) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_sgemv_bp_v5_g2_add]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(rbuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 1) / 2),1) threadsPerThreadgroup:MTLSizeMake(down_q6_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_V5_G4(woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            if (!g_q6k_sgemv_bp_v5_g4) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_sgemv_bp_v5_g4]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 3) / 4),1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_ROWTILED_B16(woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            if (!g_q6k_rowtiled_b16 || B > 16) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_rowtiled_b16]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(((Nrows) + 7) / 8),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_ROWTILED_S8(woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            if (!g_q6k_rowtiled_s8 || B < 8) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q6k_rowtiled_s8]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(((Nrows) + 7) / 8),(NSUInteger)((B + 7) / 8),1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q6_PSO(pso, woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            id<MTLComputePipelineState> _p=(pso); if (!_p) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            NSUInteger _tg = down_q6_tg; \
            const char* _tg_env = getenv("STRATUM_GPU_BATCH_DOWN_Q6_TG"); \
            if (_tg_env) { \
                unsigned long _v = strtoul(_tg_env, NULL, 10); \
                if (_v == 32 || _v == 64 || _v == 128 || _v == 256) _tg = (NSUInteger)_v; \
            } \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:_p]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)B,1) threadsPerThreadgroup:MTLSizeMake(_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q4_ADD(woff, wtb, xbuf, rbuf, Nrows, Kdim) do { \
            if (!g_q4k_sgemv_bp_add) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q4k_sgemv_bp_add]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(rbuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)B,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define MMB_Q4_G2_ADD(woff, wtb, xbuf, rbuf, Nrows, Kdim) do { \
            if (!g_q4k_sgemv_bp_g2_add) return -1; \
            id<MTLBuffer> _wb; uint64_t _wo; \
            if (find_chunk((woff),(wtb),&_wb,&_wo)!=0) return -1; \
            uint32_t _K=(uint32_t)(Kdim), _N=(uint32_t)(Nrows); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_q4k_sgemv_bp_g2_add]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(rbuf) offset:0 atIndex:2]; \
            [_e setBytes:&_K length:4 atIndex:3]; [_e setBytes:&_N length:4 atIndex:4]; [_e setBytes:&Bu length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),(NSUInteger)((B + 1) / 2),1) threadsPerThreadgroup:MTLSizeMake(q4_g2_tg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define NORMB(gain_off, xbuf, ybuf) do { \
            id<MTLBuffer> _gb; uint64_t _go; if (find_chunk((gain_off),(size_t)H*4,&_gb,&_go)!=0) return -1; \
            NSUInteger _ntg = 128; \
            const char* _ntg_env = getenv("STRATUM_GPU_BATCH_NORM_TG"); \
            if (_ntg_env) { \
                unsigned long _v = strtoul(_ntg_env, NULL, 10); \
                if (_v == 64 || _v == 128 || _v == 256) _ntg = (NSUInteger)_v; \
            } \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_rmsnorm_b]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:0]; [_e setBuffer:_gb offset:_go atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; [_e setBytes:&Hu length:4 atIndex:3]; [_e setBytes:&rms_eps length:4 atIndex:4]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)B,1,1) threadsPerThreadgroup:MTLSizeMake(_ntg,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define ROPEB(buf, nheads) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:g_rope_b]; \
            [_e setBuffer:(buf) offset:0 atIndex:0]; [_e setBytes:&Hdu length:4 atIndex:1]; [_e setBytes:&rdu length:4 atIndex:2]; \
            [_e setBuffer:g_bpos offset:0 atIndex:3]; [_e setBytes:&rope_theta length:4 atIndex:4]; \
            uint32_t _nh=(uint32_t)(nheads); [_e setBytes:&_nh length:4 atIndex:5]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)((size_t)B*(nheads)*(rope_dim/2)),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define SCATTER(src, cache, layoff) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:g_kv_scatter]; \
            [_e setBuffer:(src) offset:0 atIndex:0]; [_e setBuffer:(cache) offset:(layoff) atIndex:1]; \
            [_e setBuffer:g_bslot offset:0 atIndex:2]; [_e setBytes:&row_kv length:4 atIndex:3]; [_e setBytes:&mku length:4 atIndex:4]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)((size_t)B*Nk*Hd),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define KVROPE_SCATTER(layoff) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:g_kv_rope_scatter]; \
            [_e setBuffer:g_bk offset:0 atIndex:0]; [_e setBuffer:g_bv offset:0 atIndex:1]; \
            [_e setBuffer:g_bkv_k offset:(layoff) atIndex:2]; [_e setBuffer:g_bkv_v offset:(layoff) atIndex:3]; \
            [_e setBuffer:g_bslot offset:0 atIndex:4]; [_e setBuffer:g_bpos offset:0 atIndex:5]; \
            [_e setBytes:&Hdu length:4 atIndex:6]; [_e setBytes:&rdu length:4 atIndex:7]; [_e setBytes:&Nku length:4 atIndex:8]; \
            [_e setBytes:&row_kv length:4 atIndex:9]; [_e setBytes:&mku length:4 atIndex:10]; [_e setBytes:&rope_theta length:4 atIndex:11]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)((size_t)B*Nk*Hd),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; \
            [_e endEncoding]; \
        } while(0)
        #define ELEMB(pso, abuf, bbuf, nn) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:(pso)]; \
            [_e setBuffer:(abuf) offset:0 atIndex:0]; [_e setBuffer:(bbuf) offset:0 atIndex:1]; \
            uint32_t _n=(uint32_t)(nn); [_e setBytes:&_n length:4 atIndex:2]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)(nn),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; \
            [_e endEncoding]; \
        } while(0)

        int use_fused_kv = (g_kv_rope_scatter != nil) && getenv("STRATUM_GPU_FUSED_KV") != NULL;
        { const char* e = getenv("STRATUM_FWD_NL"); if (e) { int v=atoi(e); if (v>=0 && v<n_layers) n_layers=v; } }
        for (int L=0; L<n_layers; L++) {
            const StratumMetalLayer* ly=&layers[L];
            size_t layoff = (size_t)L*kv_layer_stride*4;
            NORMB(ly->attn_norm_off, g_bx, g_bxn);
            PROFILE_CUT(BPROF_ATTN_NORM);
            MMB(ly->q_ty, ly->q_off, ly->q_tb, g_bxn, g_bq, Nq*Hd, H);
            int use_kv_fused =
                !getenv("STRATUM_GPU_BATCH_KV_FUSED_DISABLE")
                && (getenv("STRATUM_GPU_BATCH_KV_FUSED") || B < 32);
            if (use_kv_fused
                && !use_simdb && use_bparallel && g_q4k_dual_sgemv_bp_g2
                && ly->k_ty == 12 && ly->v_ty == 12) {
                MMB_DUAL_Q4_G2(ly->k_off, ly->k_tb, ly->v_off, ly->v_tb, g_bxn, g_bk, g_bv, Nk*Hd, H, kv_tg);
            } else {
                MMB(ly->k_ty, ly->k_off, ly->k_tb, g_bxn, g_bk, Nk*Hd, H);
                MMB(ly->v_ty, ly->v_off, ly->v_tb, g_bxn, g_bv, Nk*Hd, H);
            }
            PROFILE_CUT(BPROF_QKV);
            ROPEB(g_bq, Nq);
            if (use_fused_kv) {
                KVROPE_SCATTER(layoff);
            } else {
                ROPEB(g_bk, Nk);
                SCATTER(g_bk, g_bkv_k, layoff);
                SCATTER(g_bv, g_bkv_v, layoff);
            }
            PROFILE_CUT(BPROF_ROPE_KV);
            /* batched attention: one threadgroup per (sequence, query head) */
            { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:g_attn_b];
              [e setBuffer:g_bq offset:0 atIndex:0];
              [e setBuffer:g_bkv_k offset:layoff atIndex:1];
              [e setBuffer:g_bkv_v offset:layoff atIndex:2];
              [e setBuffer:g_battn offset:0 atIndex:3];
              [e setBytes:&Hdu length:4 atIndex:4]; [e setBytes:&Nqu length:4 atIndex:5];
              [e setBytes:&Nku length:4 atIndex:6]; [e setBuffer:g_bkvp1 offset:0 atIndex:7];
              [e setBytes:&scale length:4 atIndex:8]; [e setBytes:&mku length:4 atIndex:9];
              [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((size_t)B*Nq),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [e endEncoding]; }
            PROFILE_CUT(BPROF_ATTN);
            if (getenv("STRATUM_GPU_BATCH_O_FUSED_G2") && !use_simdb && use_bparallel
                && ly->o_ty == 12 && g_q4k_sgemv_bp_g2_add) {
                MMB_Q4_G2_ADD(ly->o_off, ly->o_tb, g_battn, g_bx, H, Nq*Hd);
            } else if (getenv("STRATUM_GPU_BATCH_O_FUSED") && !use_simdb && use_bparallel
                && ly->o_ty == 12 && g_q4k_sgemv_bp_add) {
                MMB_Q4_ADD(ly->o_off, ly->o_tb, g_battn, g_bx, H, Nq*Hd);
            } else {
                MMB(ly->o_ty, ly->o_off, ly->o_tb, g_battn, g_btmp, H, Nq*Hd);
                ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
            }
            PROFILE_CUT(BPROF_O_PROJ);
            NORMB(ly->ffn_norm_off, g_bx, g_bxn);
            PROFILE_CUT(BPROF_FFN_NORM);
            if (getenv("STRATUM_GPU_BATCH_FFN_SERIAL")) use_bparallel = 0;
            else if (getenv("STRATUM_GPU_BATCH_FFN_PAR")) use_bparallel = 1;
            int ffn_swiglu_fused = 0;
            if (!getenv("STRATUM_GPU_BATCH_FFN_NO_FUSED")
                && ly->gate_ty == 12 && ly->up_ty == 12) {
                q4_dual_tg = ffn_tg;
                if (!use_simdb && use_bparallel && g_q4k_dual_sgemv_bp_g2
                    && !getenv("STRATUM_GPU_BATCH_FFN_FUSED_V1")
                    && !getenv("STRATUM_GPU_BATCH_FFN_FUSED_V2")) {
                    if (getenv("STRATUM_GPU_BATCH_FFN_FUSED_G4") && g_q4k_dual_sgemv_bp_g4) {
                        MMB_DUAL_Q4_G4(ly->gate_off, ly->gate_tb, ly->up_off, ly->up_tb, g_bxn, g_bg, g_bu, Ff, H);
                    } else if ((getenv("STRATUM_GPU_BATCH_FFN_SWIGLU_FUSED")
                        || (B == 16 && !getenv("STRATUM_GPU_BATCH_FFN_SWIGLU_FUSED_DISABLE")))
                        && g_q4k_dual_swiglu_bp_g2) {
                        MMB_DUAL_SWIGLU_Q4_G2(ly->gate_off, ly->gate_tb, ly->up_off, ly->up_tb, g_bxn, g_ba, Ff, H);
                        ffn_swiglu_fused = 1;
                    } else {
                        MMB_DUAL_Q4_G2(ly->gate_off, ly->gate_tb, ly->up_off, ly->up_tb, g_bxn, g_bg, g_bu, Ff, H, q4_dual_tg);
                    }
                } else if (g_q4k_dual_sgemv_bp) {
                    MMB_DUAL_Q4(ly->gate_off, ly->gate_tb, ly->up_off, ly->up_tb, g_bxn, g_bg, g_bu, Ff, H);
                } else {
                    MMB(ly->gate_ty, ly->gate_off, ly->gate_tb, g_bxn, g_bg, Ff, H);
                    MMB(ly->up_ty,   ly->up_off,   ly->up_tb,   g_bxn, g_bu, Ff, H);
                }
                q4_dual_tg = 64;
            } else {
                MMB(ly->gate_ty, ly->gate_off, ly->gate_tb, g_bxn, g_bg, Ff, H);
                MMB(ly->up_ty,   ly->up_off,   ly->up_tb,   g_bxn, g_bu, Ff, H);
            }
            use_bparallel = use_bparallel_saved;
            PROFILE_CUT(BPROF_GATE_UP);
            if (!ffn_swiglu_fused) {
                { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:g_swiglu];
                  [e setBuffer:g_bg offset:0 atIndex:0]; [e setBuffer:g_bu offset:0 atIndex:1];
                  [e setBuffer:g_ba offset:0 atIndex:2]; uint32_t _nbff=(uint32_t)((size_t)B*Ff); [e setBytes:&_nbff length:4 atIndex:3];
                  [e dispatchThreads:MTLSizeMake((NSUInteger)((size_t)B*Ff),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding]; }
                PROFILE_CUT(BPROF_SWIGLU);
            }
            if (getenv("STRATUM_GPU_BATCH_DOWN_SERIAL")) use_bparallel = 0;
            else if (getenv("STRATUM_GPU_BATCH_DOWN_PAR")) use_bparallel = 1;
            if (getenv("STRATUM_GPU_BATCH_DOWN_V5_FUSED") && !use_simdb && use_bparallel
                && ly->down_ty == 14 && g_q6k_sgemv_bp_v5_add) {
                MMB_Q6_V5_ADD(ly->down_off, ly->down_tb, g_ba, g_bx, H, Ff);
            } else if (getenv("STRATUM_GPU_BATCH_DOWN_FUSED") && !use_simdb && use_bparallel
                && ly->down_ty == 14 && g_q6k_sgemv_bp_add) {
                MMB_Q6_ADD(ly->down_off, ly->down_tb, g_ba, g_bx, H, Ff);
            } else if (getenv("STRATUM_GPU_BATCH_DOWN_G2") && !use_simdb && use_bparallel
                && ly->down_ty == 14 && g_q6k_sgemv_bp_g2) {
                MMB_Q6_G2(ly->down_off, ly->down_tb, g_ba, g_btmp, H, Ff);
                ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
            } else if (getenv("STRATUM_GPU_BATCH_DOWN_Q6_V5_G4") && !use_simdb && use_bparallel
                && ly->down_ty == 14 && g_q6k_sgemv_bp_v5_g4) {
                MMB_Q6_V5_G4(ly->down_off, ly->down_tb, g_ba, g_btmp, H, Ff);
                ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
            } else if (((B >= 24 && !getenv("STRATUM_GPU_BATCH_DOWN_Q6_ROWTILED_S8_DISABLE"))
                    || getenv("STRATUM_GPU_BATCH_DOWN_Q6_ROWTILED_S8"))
                && !use_simdb && use_bparallel
                && ly->down_ty == 14 && g_q6k_rowtiled_s8 && B >= 8) {
                MMB_Q6_ROWTILED_S8(ly->down_off, ly->down_tb, g_ba, g_btmp, H, Ff);
                ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
            } else if (!getenv("STRATUM_GPU_BATCH_DOWN_Q6_V5_G2_DISABLE") && !use_simdb && use_bparallel
                && ly->down_ty == 14 && g_q6k_sgemv_bp_v5_g2
                && (B >= 6 || getenv("STRATUM_GPU_BATCH_DOWN_Q6_V5_G2"))) {
                if (getenv("STRATUM_GPU_BATCH_DOWN_Q6_V5_G2_ADD") && g_q6k_sgemv_bp_v5_g2_add) {
                    MMB_Q6_V5_G2_ADD(ly->down_off, ly->down_tb, g_ba, g_bx, H, Ff);
                } else {
                    MMB_Q6_V5_G2(ly->down_off, ly->down_tb, g_ba, g_btmp, H, Ff);
                    ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
                }
            } else if (getenv("STRATUM_GPU_BATCH_DOWN_Q6_ROWTILED") && !use_simdb && use_bparallel
                && ly->down_ty == 14 && g_q6k_rowtiled_b16 && B <= 16) {
                MMB_Q6_ROWTILED_B16(ly->down_off, ly->down_tb, g_ba, g_btmp, H, Ff);
                ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
            } else if (ly->down_ty == 14 && !use_simdb && use_bparallel
                && (B >= 16 ||
                    getenv("STRATUM_GPU_BATCH_DOWN_Q6_OLD") ||
                    getenv("STRATUM_GPU_BATCH_DOWN_Q6_V4") ||
                    getenv("STRATUM_GPU_BATCH_DOWN_Q6_V5") ||
                    getenv("STRATUM_GPU_BATCH_DOWN_Q6_V6"))) {
                id<MTLComputePipelineState> _dp =
                    getenv("STRATUM_GPU_BATCH_DOWN_Q6_OLD") ? g_q6k_sgemv_bp :
                    getenv("STRATUM_GPU_BATCH_DOWN_Q6_V4")  ? (g_q6k_sgemv_bp_v4 ? g_q6k_sgemv_bp_v4 : g_q6k_sgemv_bp) :
                    getenv("STRATUM_GPU_BATCH_DOWN_Q6_V6")  ? (g_q6k_sgemv_bp_v6 ? g_q6k_sgemv_bp_v6 : (g_q6k_sgemv_bp_v5 ? g_q6k_sgemv_bp_v5 : g_q6k_sgemv_bp)) :
                    (g_q6k_sgemv_bp_v5 ? g_q6k_sgemv_bp_v5 : (g_q6k_sgemv_bp_v4 ? g_q6k_sgemv_bp_v4 : g_q6k_sgemv_bp));
                MMB_Q6_PSO(_dp, ly->down_off, ly->down_tb, g_ba, g_btmp, H, Ff);
                ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
            } else {
                MMB(ly->down_ty, ly->down_off, ly->down_tb, g_ba, g_btmp, H, Ff);
                ELEMB(g_add, g_bx, g_btmp, (size_t)B*H);
            }
            use_bparallel = use_bparallel_saved;
            PROFILE_CUT(BPROF_DOWN);
        }
        NORMB(out_norm_off, g_bx, g_bxn);
        PROFILE_CUT(BPROF_OUT_NORM);
        if (tokens_out && !logits_out && lm_is_q6 && getenv("STRATUM_GPU_TOP1_TILED")) {
            if (!g_q6k_top1_tiles_b || !g_top1_reduce_tiles) return -1;
            id<MTLBuffer> _wb; uint64_t _wo;
            if (find_chunk(lm_off, lm_tb, &_wb, &_wo) != 0) return -1;
            uint32_t _K=(uint32_t)H, _V=(uint32_t)V;
            { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:g_q6k_top1_tiles_b];
              [e setBuffer:_wb offset:_wo atIndex:0];
              [e setBuffer:g_bxn offset:0 atIndex:1];
              [e setBuffer:g_btile_vals offset:0 atIndex:2];
              [e setBuffer:g_btile_idxs offset:0 atIndex:3];
              [e setBytes:&_K length:4 atIndex:4];
              [e setBytes:&_V length:4 atIndex:5];
              [e setBytes:&Bu length:4 atIndex:6];
              [e setBytes:&tile_rows length:4 atIndex:7];
              [e dispatchThreadgroups:MTLSizeMake((NSUInteger)ntile,(NSUInteger)B,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding]; }
            { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:g_top1_reduce_tiles];
              [e setBuffer:g_btile_vals offset:0 atIndex:0];
              [e setBuffer:g_btile_idxs offset:0 atIndex:1];
              [e setBuffer:g_btok offset:0 atIndex:2];
              [e setBytes:&ntile length:4 atIndex:3];
              [e dispatchThreadgroups:MTLSizeMake((NSUInteger)B,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding]; }
            PROFILE_CUT(BPROF_LM_HEAD);
        } else if (tokens_out && !logits_out && getenv("STRATUM_GPU_FUSED_ARGMAX")) {
            id<MTLComputePipelineState> ap = lm_is_q6 ? g_q6k_argmax_b : g_q4k_argmax_b;
            if (!ap) return -1;
            id<MTLBuffer> _wb; uint64_t _wo;
            if (find_chunk(lm_off, lm_tb, &_wb, &_wo) != 0) return -1;
            uint32_t _K=(uint32_t)H, _V=(uint32_t)V;
            id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:ap];
            [e setBuffer:_wb offset:_wo atIndex:0];
            [e setBuffer:g_bxn offset:0 atIndex:1];
            [e setBuffer:g_btok offset:0 atIndex:2];
            [e setBytes:&_K length:4 atIndex:3];
            [e setBytes:&_V length:4 atIndex:4];
            [e setBytes:&Bu length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)B,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
            PROFILE_CUT(BPROF_LM_HEAD);
        } else {
            MMB(lm_is_q6 ? 14 : 12, lm_off, lm_tb, g_bxn, g_blog, V, H);
            PROFILE_CUT(BPROF_LM_HEAD);
            if (tokens_out) {
                id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; [e setComputePipelineState:g_argmax_b];
                [e setBuffer:g_blog offset:0 atIndex:0];
                [e setBuffer:g_btok offset:0 atIndex:1];
                [e setBytes:&Vu length:4 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger)B,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
                PROFILE_CUT(BPROF_ARGMAX);
            }
        }

        struct timespec _d0,_d1; clock_gettime(CLOCK_MONOTONIC,&_d0);
        [cmd commit]; [cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC,&_d1);
        if (!bprof) {
            g_n_dispatch++;
            g_dispatch_secs += elapsed_secs(_d0, _d1);
        }
        if (tokens_out) {
            const uint32_t* outp = (const uint32_t*)[g_btok contents];
            for (int s=0;s<B;s++) tokens_out[s] = (int)outp[s];
        } else if (logits_out) {
            memcpy(logits_out, [g_blog contents], vb);
        }
        #undef PSO_TYB
        #undef PSO_BPAR
        #undef PSO_SIMDB
        #undef MMB
        #undef MMB_DUAL_Q4
        #undef MMB_DUAL_Q4_G2
        #undef MMB_DUAL_Q4_G4
        #undef MMB_Q4_G2_ADD
        #undef MMB_Q4_ADD
        #undef MMB_Q6_PSO
        #undef MMB_Q6_ROWTILED_B16
        #undef MMB_Q6_ROWTILED_S8
        #undef MMB_Q6_V5_G4
        #undef MMB_Q6_V5_G2_ADD
        #undef MMB_Q6_V5_G2
        #undef MMB_Q6_G2
        #undef MMB_Q6_V5_ADD
        #undef MMB_Q6_ADD
        #undef NORMB
        #undef ROPEB
        #undef SCATTER
        #undef KVROPE_SCATTER
        #undef ELEMB
        #undef PROFILE_CUT
        return 0;
    }
}

int stratum_metal_ffn(uint64_t gate_off, size_t gate_tb,
                      uint64_t up_off,   size_t up_tb,
                      uint64_t down_off, size_t down_tb,
                      const float* x, float* y, int H, int Ff) {
    if (!g_device || !g_q4k_sgemv || !g_q6k_sgemv || !g_swiglu) return -1;
    @autoreleasepool {
        id<MTLBuffer> gw, uw, dw; uint64_t go, uo, do_;
        if (find_chunk(gate_off, gate_tb, &gw, &go) != 0) return -1;
        if (find_chunk(up_off,   up_tb,   &uw, &uo) != 0) return -1;
        if (find_chunk(down_off, down_tb, &dw, &do_) != 0) return -1;

        size_t x_bytes = (size_t)H  * sizeof(float);
        size_t ff_bytes = (size_t)Ff * sizeof(float);
        size_t h_bytes  = (size_t)H  * sizeof(float);
        if (x_bytes > g_xbuf_size) {
            g_xbuf = [g_device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            g_xbuf_size = x_bytes;
        }
        if (ff_bytes > g_ffn_ff_cap) {
            g_ffn_gate = [g_device newBufferWithLength:ff_bytes options:MTLResourceStorageModePrivate];
            g_ffn_up   = [g_device newBufferWithLength:ff_bytes options:MTLResourceStorageModePrivate];
            g_ffn_a    = [g_device newBufferWithLength:ff_bytes options:MTLResourceStorageModePrivate];
            g_ffn_ff_cap = ff_bytes;
        }
        if (h_bytes > g_ffn_h_cap) {
            g_ffn_out = [g_device newBufferWithLength:h_bytes options:MTLResourceStorageModeShared];
            g_ffn_h_cap = h_bytes;
        }
        if (!g_xbuf || !g_ffn_gate || !g_ffn_up || !g_ffn_a || !g_ffn_out) return -1;
        memcpy([g_xbuf contents], x, x_bytes);

        uint32_t Hk = (uint32_t)H, Ffk = (uint32_t)Ff, Ffn = (uint32_t)Ff;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        { id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
          [e setComputePipelineState:g_q4k_sgemv];
          [e setBuffer:gw offset:go atIndex:0]; [e setBuffer:g_xbuf offset:0 atIndex:1];
          [e setBuffer:g_ffn_gate offset:0 atIndex:2]; [e setBytes:&Hk length:4 atIndex:3];
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Ff,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
          [e endEncoding]; }
        { id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
          [e setComputePipelineState:g_q4k_sgemv];
          [e setBuffer:uw offset:uo atIndex:0]; [e setBuffer:g_xbuf offset:0 atIndex:1];
          [e setBuffer:g_ffn_up offset:0 atIndex:2]; [e setBytes:&Hk length:4 atIndex:3];
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Ff,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
          [e endEncoding]; }
        { id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
          [e setComputePipelineState:g_swiglu];
          [e setBuffer:g_ffn_gate offset:0 atIndex:0]; [e setBuffer:g_ffn_up offset:0 atIndex:1];
          [e setBuffer:g_ffn_a offset:0 atIndex:2]; [e setBytes:&Ffn length:4 atIndex:3];
          NSUInteger tg = 256;
          [e dispatchThreadgroups:MTLSizeMake((Ff+tg-1)/tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
          [e endEncoding]; }
        { id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
          [e setComputePipelineState:g_q6k_sgemv];
          [e setBuffer:dw offset:do_ atIndex:0]; [e setBuffer:g_ffn_a offset:0 atIndex:1];
          [e setBuffer:g_ffn_out offset:0 atIndex:2]; [e setBytes:&Ffk length:4 atIndex:3];
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)H,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
          [e endEncoding]; }

        struct timespec _d0, _d1;
        clock_gettime(CLOCK_MONOTONIC, &_d0);
        [cmd commit];
        [cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &_d1);
        g_n_dispatch++;
        g_dispatch_secs += (_d1.tv_sec-_d0.tv_sec)+(_d1.tv_nsec-_d0.tv_nsec)/1e9;

        memcpy(y, [g_ffn_out contents], h_bytes);
        return 0;
    }
}

int stratum_metal_ephemeral_sgemv(const void* weight_ptr, size_t weight_bytes,
                                  int gguf_type,
                                  const float* x, float* y,
                                  int N, int K, int B) {
    if (!g_device) return -1;
    id<MTLComputePipelineState> pso;
    switch (gguf_type) {
        case 12: pso = g_q4k_sgemv; break;
        case 13: pso = g_q5k_sgemv; break;
        case 14: pso = g_q6k_sgemv; break;
        default: return -1;
    }
    if (!pso) return -1;
    @autoreleasepool {

        /* V15: newBufferWithBytes (copy) — avoids wiring mmap pages.
         * One-time copy at unified-memory bandwidth (270GB/s). */
        id<MTLBuffer> wbuf = [g_device newBufferWithBytes:weight_ptr
                                                   length:weight_bytes
                                                  options:MTLResourceStorageModeShared];
        if (!wbuf) return -1;

        uint32_t K_u32 = (uint32_t)K;
        size_t xb_bytes = (size_t)K * B * sizeof(float);
        size_t yb_bytes = (size_t)N * B * sizeof(float);
        if (xb_bytes > g_xbatch_size) {
            g_xbatch = [g_device newBufferWithLength:xb_bytes options:MTLResourceStorageModeShared];
            g_xbatch_size = xb_bytes;
        }
        if (yb_bytes > g_ybatch_size) {
            g_ybatch = [g_device newBufferWithLength:yb_bytes options:MTLResourceStorageModeShared];
            g_ybatch_size = yb_bytes;
        }
        if (!g_xbatch || !g_ybatch) return -1;
        memcpy([g_xbatch contents], x, xb_bytes);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        for (int b = 0; b < B; b++) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf     offset:0                        atIndex:0];
            [enc setBuffer:g_xbatch offset:(size_t)b*K*sizeof(float) atIndex:1];
            [enc setBuffer:g_ybatch offset:(size_t)b*N*sizeof(float) atIndex:2];
            [enc setBytes:&K_u32    length:sizeof(uint32_t)         atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(y, [g_ybatch contents], yb_bytes);

        return 0;
    }
}

/* Fused multi-matmul: up to 4 weight tensors sharing the same B-column
 * input, dispatched into ONE command buffer with ONE sync. Eliminates
 * per-matmul commit/wait latency (the bottleneck). Each output i goes to
 * y_i = ys + yoff[i], shape [B][N_i]. All share x [B][K]. */
int stratum_metal_q4k_multi(const uint64_t* woff_arr, const int* N_arr,
                            int nmat, const float* x, float* ys,
                            const size_t* yoff, int K, int B) {
    @autoreleasepool {
        if (B < 1 || B > 32) return -1;
        id<MTLComputePipelineState> bpso = g_q4k_sgemv_b[B];
        if (!bpso || nmat < 1 || nmat > 8) return -1;
        size_t x_bytes = (size_t)B * K * sizeof(float);
        if (x_bytes > g_xbuf_size) {
            g_xbuf = [g_device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            g_xbuf_size = x_bytes;
        }
        /* total y bytes across all matmuls */
        size_t ytot = 0;
        for (int i = 0; i < nmat; i++) ytot += (size_t)B * N_arr[i] * sizeof(float);
        if (ytot > g_ybuf_size) {
            g_ybuf = [g_device newBufferWithLength:ytot options:MTLResourceStorageModeShared];
            g_ybuf_size = ytot;
        }
        if (!g_xbuf || !g_ybuf) return -1;
        memcpy([g_xbuf contents], x, x_bytes);
        uint32_t K_u32 = (uint32_t)K, B_u32 = (uint32_t)B;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        size_t ybyte_off = 0;
        for (int i = 0; i < nmat; i++) {
            int N = N_arr[i];
            size_t tensor_bytes = ((size_t)N * (size_t)K / 256) * 144;
            id<MTLBuffer> wbuf; uint64_t wo;
            if (find_chunk(woff_arr[i], tensor_bytes, &wbuf, &wo) != 0) return -1;
            uint32_t N_u32 = (uint32_t)N;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:bpso];
            [enc setBuffer:wbuf   offset:wo  atIndex:0];
            [enc setBuffer:g_xbuf offset:0   atIndex:1];
            [enc setBuffer:g_ybuf offset:ybyte_off atIndex:2];
            [enc setBytes:&K_u32 length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N_u32 length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&B_u32 length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                  threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
            ybyte_off += (size_t)B * N * sizeof(float);
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        /* copy out each matmul's [B][N_i] block to ys+yoff[i] */
        const char* yc = (const char*)[g_ybuf contents];
        size_t cur = 0;
        for (int i = 0; i < nmat; i++) {
            size_t blk = (size_t)B * N_arr[i] * sizeof(float);
            memcpy((char*)ys + yoff[i], yc + cur, blk);
            cur += blk;
        }
        return 0;
    }
}

int stratum_metal_q4k_sgemv(uint64_t weight_offset,
                            const float* x, float* y,
                            int N, int K) {

    size_t bytes = ((size_t)N * (size_t)K / 256) * 144;
    return dispatch_sgemv(g_q4k_sgemv, weight_offset, bytes, x, y, N, K);
}

/* Batched: B input columns in one dispatch. x = B*K floats (x[s*K+k]),
 * y = B*N floats (y[s*N+row]). Amortizes dispatch over B. */
int stratum_metal_q4k_sgemv_batched(uint64_t weight_offset,
                                    const float* x, float* y,
                                    int N, int K, int B) {
    @autoreleasepool {
        if (B < 1 || B > 32) return -1;
        id<MTLComputePipelineState> bpso = g_q4k_sgemv_b[B];
        if (!bpso) return -1;
        size_t tensor_bytes = ((size_t)N * (size_t)K / 256) * 144;
        id<MTLBuffer> wbuf; uint64_t woff;
        if (find_chunk(weight_offset, tensor_bytes, &wbuf, &woff) != 0) return -1;
        size_t x_bytes = (size_t)B * K * sizeof(float);
        size_t y_bytes = (size_t)B * N * sizeof(float);
        if (x_bytes > g_xbuf_size) {
            g_xbuf = [g_device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            g_xbuf_size = x_bytes;
        }
        if (y_bytes > g_ybuf_size) {
            g_ybuf = [g_device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
            g_ybuf_size = y_bytes;
        }
        if (!g_xbuf || !g_ybuf) return -1;
        memcpy([g_xbuf contents], x, x_bytes);
        uint32_t K_u32 = (uint32_t)K, N_u32 = (uint32_t)N, B_u32 = (uint32_t)B;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:bpso];
        [enc setBuffer:wbuf   offset:woff atIndex:0];
        [enc setBuffer:g_xbuf offset:0    atIndex:1];
        [enc setBuffer:g_ybuf offset:0    atIndex:2];
        [enc setBytes:&K_u32 length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&N_u32 length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&B_u32 length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
              threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(y, [g_ybuf contents], y_bytes);
        return 0;
    }
}

int stratum_metal_q5k_sgemv(uint64_t weight_offset,
                            const float* x, float* y,
                            int N, int K) {

    size_t bytes = ((size_t)N * (size_t)K / 256) * 176;
    return dispatch_sgemv(g_q5k_sgemv, weight_offset, bytes, x, y, N, K);
}

int stratum_metal_q6k_sgemv(uint64_t weight_offset,
                            const float* x, float* y,
                            int N, int K) {

    size_t bytes = ((size_t)N * (size_t)K / 256) * 210;
    return dispatch_sgemv(g_q6k_sgemv, weight_offset, bytes, x, y, N, K);
}

int stratum_metal_register_staging(void* host_ptr, size_t nbytes) {
    if (!g_device) return -1;
    if (g_n_staging >= STRATUM_METAL_MAX_STAGING) return -1;
    @autoreleasepool {
        id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:host_ptr
                                                        length:nbytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (!buf) return -1;
        int handle = g_n_staging++;
        g_staging[handle] = buf;
        return handle;
    }
}

int stratum_metal_staged_sgemv(int handle, uint64_t woff, int gguf_type,
                               const float* x, float* y,
                               int N, int K, int B) {
    if (handle != STRATUM_METAL_NC_HANDLE && (handle < 0 || handle >= g_n_staging)) return -1;
    id<MTLComputePipelineState> pso, bpso;
    switch (gguf_type) {
        case 10: pso = g_q2k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q2k_sgemv_b[B] : nil; break;
        case 12: pso = g_q4k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q4k_sgemv_b[B] : nil; break;
        case 13: pso = g_q5k_sgemv;  bpso = nil; break;   /* no batched q5k */
        case 14: pso = g_q6k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q6k_sgemv_b[B] : nil; break;
        default: return -1;
    }
    if (!pso) return -1;
    @autoreleasepool {
        id<MTLBuffer> wbuf; uint64_t woff2;
        if (handle == STRATUM_METAL_NC_HANDLE) {
            /* V54: zero-copy direct read — woff is a FILE offset into the
             * mmap'd model; find_chunk resolves the pre-registered NoCopy
             * chunk buffer (no staging copy, no pread, no extra wiring —
             * verified: GPU reads of file-backed mmap pages add ~0 wired). */
            int blk = (gguf_type == 10) ? 84 : (gguf_type == 12) ? 144
                    : (gguf_type == 13) ? 176 : 210;
            size_t need = (size_t)N * ((size_t)(K / 256)) * (size_t)blk;
            if (find_chunk(woff, need, &wbuf, &woff2) != 0) return -1;
        } else {
            wbuf = g_staging[handle];
            woff2 = woff;
        }
        uint32_t K_u32 = (uint32_t)K;
        size_t xb_bytes = (size_t)K * B * sizeof(float);
        size_t yb_bytes = (size_t)N * B * sizeof(float);
        if (xb_bytes > g_xbatch_size) {
            g_xbatch = [g_device newBufferWithLength:xb_bytes options:MTLResourceStorageModeShared];
            g_xbatch_size = xb_bytes;
        }
        if (yb_bytes > g_ybatch_size) {
            g_ybatch = [g_device newBufferWithLength:yb_bytes options:MTLResourceStorageModeShared];
            g_ybatch_size = yb_bytes;
        }
        if (!g_xbatch || !g_ybatch) return -1;
        memcpy([g_xbatch contents], x, xb_bytes);
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        if (B > 1 && bpso) {
            /* V51: batched single-encoder dispatch — weights unpacked once
             * per row and reused across all B streams (vs B separate
             * single-stream kernels that each re-read the weight rows). */
            uint32_t N_u32 = (uint32_t)N, B_u32 = (uint32_t)B;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:bpso];
            [enc setBuffer:wbuf     offset:woff2 atIndex:0];
            [enc setBuffer:g_xbatch offset:0    atIndex:1];
            [enc setBuffer:g_ybatch offset:0    atIndex:2];
            [enc setBytes:&K_u32    length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N_u32    length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&B_u32    length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        } else {
        for (int b = 0; b < B; b++) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf     offset:woff2                    atIndex:0];
            [enc setBuffer:g_xbatch offset:(size_t)b*K*sizeof(float) atIndex:1];
            [enc setBuffer:g_ybatch offset:(size_t)b*N*sizeof(float) atIndex:2];
            [enc setBytes:&K_u32    length:sizeof(uint32_t)         atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        }
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(y, [g_ybatch contents], yb_bytes);
        return 0;
    }
}

/* V54.1: per-tensor zero-copy dispatch — SAFE form. Each call wraps ONLY
 * the given tensor's bytes in a NoCopy buffer (≤ tensor size, typically
 * <100MB), dispatches, waits, and releases. The GPU therefore touches only
 * the pages of the in-flight tensor; wired memory stays bounded by the
 * largest single tensor instead of the whole model (the whole-model NoCopy
 * chunk form wires everything on first GPU touch -> OOM kill, seen as
 * EXIT=137). Per AGENTS.md this is the allowed "per-tensor dispatch
 * (<50MB, release after use)" GPU mode. */
int stratum_metal_nc_sgemv2(const void* wptr, size_t nbytes, int gguf_type,
                            const float* x, float* y, int N, int K, int B) {
    if (!wptr || nbytes == 0 || !g_device) return -1;
    id<MTLComputePipelineState> pso, bpso;
    switch (gguf_type) {
        case 10: pso = g_q2k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q2k_sgemv_b[B] : nil; break;
        case 12: pso = g_q4k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q4k_sgemv_b[B] : nil; break;
        case 13: pso = g_q5k_sgemv;  bpso = nil; break;
        case 14: pso = g_q6k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q6k_sgemv_b[B] : nil; break;
        default: return -1;
    }
    if (!pso) return -1;
    @autoreleasepool {
        id<MTLBuffer> wbuf = [g_device newBufferWithBytesNoCopy:(void*)wptr
                                                         length:nbytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!wbuf) return -1;
        uint32_t K_u32 = (uint32_t)K;
        size_t xb_bytes = (size_t)K * B * sizeof(float);
        size_t yb_bytes = (size_t)N * B * sizeof(float);
        if (xb_bytes > g_xbatch_size) {
            g_xbatch = [g_device newBufferWithLength:xb_bytes options:MTLResourceStorageModeShared];
            g_xbatch_size = xb_bytes;
        }
        if (yb_bytes > g_ybatch_size) {
            g_ybatch = [g_device newBufferWithLength:yb_bytes options:MTLResourceStorageModeShared];
            g_ybatch_size = yb_bytes;
        }
        if (!g_xbatch || !g_ybatch) return -1;
        memcpy([g_xbatch contents], x, xb_bytes);
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        if (B > 1 && bpso) {
            uint32_t N_u32 = (uint32_t)N, B_u32 = (uint32_t)B;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:bpso];
            [enc setBuffer:wbuf     offset:0 atIndex:0];
            [enc setBuffer:g_xbatch offset:0 atIndex:1];
            [enc setBuffer:g_ybatch offset:0 atIndex:2];
            [enc setBytes:&K_u32    length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N_u32    length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&B_u32    length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        } else {
        for (int b = 0; b < B; b++) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf     offset:0                           atIndex:0];
            [enc setBuffer:g_xbatch offset:(size_t)b*K*sizeof(float)   atIndex:1];
            [enc setBuffer:g_ybatch offset:(size_t)b*N*sizeof(float)   atIndex:2];
            [enc setBytes:&K_u32    length:sizeof(uint32_t)            atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        }
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(y, [g_ybatch contents], yb_bytes);
        return 0;
    }
}

static id<MTLCommandBuffer> g_async_cmd = nil;
static float* g_async_y = NULL;
static size_t g_async_ybytes = 0;
static id<MTLBuffer> g_async_ybuf = nil;

int stratum_metal_staged_wait(void) {
    if (!g_async_cmd) return 0;
    @autoreleasepool {
        [g_async_cmd waitUntilCompleted];
        if (g_async_y && g_async_ybuf && g_async_ybytes)
            memcpy(g_async_y, [g_async_ybuf contents], g_async_ybytes);
        g_async_cmd = nil;
        g_async_y = NULL;
        g_async_ybytes = 0;
        g_async_ybuf = nil;
        return 0;
    }
}

int stratum_metal_staged_sgemv_async(int handle, uint64_t woff, int gguf_type,
                                     const float* x, float* y,
                                     int N, int K, int B) {
    if (handle < 0 || handle >= g_n_staging) return -1;
    if (B < 1) return -1;
    stratum_metal_staged_wait();
    id<MTLComputePipelineState> pso, bpso;
    switch (gguf_type) {
        case 10: pso = g_q2k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q2k_sgemv_b[B] : nil; break;
        case 12: pso = g_q4k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q4k_sgemv_b[B] : nil; break;
        case 13: pso = g_q5k_sgemv;  bpso = nil; break;
        case 14: pso = g_q6k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q6k_sgemv_b[B] : nil; break;
        default: return -1;
    }
    if (!pso) return -1;
    @autoreleasepool {
        id<MTLBuffer> wbuf = g_staging[handle];
        uint32_t K_u32 = (uint32_t)K;
        size_t xb_bytes = (size_t)K * B * sizeof(float);
        size_t yb_bytes = (size_t)N * B * sizeof(float);
        if (xb_bytes > g_xbatch_size) {
            g_xbatch = [g_device newBufferWithLength:xb_bytes options:MTLResourceStorageModeShared];
            g_xbatch_size = xb_bytes;
        }
        if (yb_bytes > g_ybatch_size) {
            g_ybatch = [g_device newBufferWithLength:yb_bytes options:MTLResourceStorageModeShared];
            g_ybatch_size = yb_bytes;
        }
        if (!g_xbatch || !g_ybatch) return -1;
        memcpy([g_xbatch contents], x, xb_bytes);
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        if (B > 1 && bpso) {
            /* V51: batched single-encoder dispatch — weight reuse across B */
            uint32_t N_u32 = (uint32_t)N, B_u32 = (uint32_t)B;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:bpso];
            [enc setBuffer:wbuf     offset:woff atIndex:0];
            [enc setBuffer:g_xbatch offset:0    atIndex:1];
            [enc setBuffer:g_ybatch offset:0    atIndex:2];
            [enc setBytes:&K_u32    length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N_u32    length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&B_u32    length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        } else {
        for (int b = 0; b < B; b++) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf     offset:woff                     atIndex:0];
            [enc setBuffer:g_xbatch offset:(size_t)b*K*sizeof(float) atIndex:1];
            [enc setBuffer:g_ybatch offset:(size_t)b*N*sizeof(float) atIndex:2];
            [enc setBytes:&K_u32    length:sizeof(uint32_t)         atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        }
        }
        [cmd commit];
        g_async_cmd = cmd;
        g_async_y = y;
        g_async_ybytes = yb_bytes;
        g_async_ybuf = g_ybatch;
        return 0;
    }
}

int stratum_metal_staged_group(const int* handles, const uint64_t* woffs,
                               const int* types, const int* Ns, int nmat,
                               const float* x, float* const* ys, int K) {
    if (!g_device || nmat < 1 || nmat > 8) return -1;
    stratum_metal_staged_wait();
    @autoreleasepool {
        size_t total_N = 0;
        for (int i = 0; i < nmat; i++) {
            if (handles[i] < 0 || handles[i] >= g_n_staging) return -1;
            total_N += (size_t)Ns[i];
        }
        size_t x_bytes = (size_t)K * sizeof(float);
        size_t y_bytes = total_N * sizeof(float);
        if (x_bytes > g_xbatch_size) {
            g_xbatch = [g_device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            g_xbatch_size = x_bytes;
        }
        if (y_bytes > g_ybatch_size) {
            g_ybatch = [g_device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
            g_ybatch_size = y_bytes;
        }
        if (!g_xbatch || !g_ybatch) return -1;
        memcpy([g_xbatch contents], x, x_bytes);
        uint32_t K_u32 = (uint32_t)K;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        size_t yoff = 0;
        for (int i = 0; i < nmat; i++) {
            id<MTLComputePipelineState> pso;
            switch (types[i]) {
                case 10: pso = g_q2k_sgemv; break;
                case 12: pso = g_q4k_sgemv; break;
                case 13: pso = g_q5k_sgemv; break;
                case 14: pso = g_q6k_sgemv; break;
                default: return -1;
            }
            if (!pso) return -1;
            id<MTLBuffer> wbuf = g_staging[handles[i]];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:wbuf     offset:woffs[i] atIndex:0];
            [enc setBuffer:g_xbatch offset:0        atIndex:1];
            [enc setBuffer:g_ybatch offset:yoff     atIndex:2];
            [enc setBytes:&K_u32 length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Ns[i],1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
            yoff += (size_t)Ns[i] * sizeof(float);
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        const uint8_t* yb = (const uint8_t*)[g_ybatch contents];
        size_t off = 0;
        for (int i = 0; i < nmat; i++) {
            memcpy(ys[i], yb + off, (size_t)Ns[i] * sizeof(float));
            off += (size_t)Ns[i] * sizeof(float);
        }
        return 0;
    }
}

/* V53: batched staged group (B streams, one input x). Each mat dispatched
 * with its type's batched PSO into one command buffer. ys is [nmat][B]
 * pointers: ys[i][s] = stream-s output for matrix i (length Ns[i]). */
int stratum_metal_staged_group_b(const int* handles, const uint64_t* woffs,
                                 const int* types, const int* Ns, int nmat,
                                 const float* x, float** const* ys, int K, int B) {
    if (!g_device || nmat < 1 || nmat > 8 || B < 1 || B > 32) return -1;
    stratum_metal_staged_wait();
    @autoreleasepool {
        size_t total_N = 0;
        for (int i = 0; i < nmat; i++) {
            if (handles[i] < 0 || handles[i] >= g_n_staging) return -1;
            total_N += (size_t)Ns[i] * (size_t)B;
        }
        size_t x_bytes = (size_t)K * (size_t)B * sizeof(float);
        size_t y_bytes = total_N * sizeof(float);
        if (x_bytes > g_xbatch_size) {
            g_xbatch = [g_device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
            g_xbatch_size = x_bytes;
        }
        if (y_bytes > g_ybatch_size) {
            g_ybatch = [g_device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
            g_ybatch_size = y_bytes;
        }
        if (!g_xbatch || !g_ybatch) return -1;
        memcpy([g_xbatch contents], x, x_bytes);
        uint32_t K_u32 = (uint32_t)K, B_u32 = (uint32_t)B;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        size_t yoff = 0;
        for (int i = 0; i < nmat; i++) {
            id<MTLComputePipelineState> bpso = nil, pso = nil;
            switch (types[i]) {
                case 10: pso = g_q2k_sgemv;  bpso = g_q2k_sgemv_b[B]; break;
                case 12: pso = g_q4k_sgemv;  bpso = g_q4k_sgemv_b[B]; break;
                case 13: pso = g_q5k_sgemv;  break;
                case 14: pso = g_q6k_sgemv;  bpso = g_q6k_sgemv_b[B]; break;
                default: return -1;
            }
            if (!pso && !bpso) return -1;
            id<MTLBuffer> wbuf = g_staging[handles[i]];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (B > 1 && bpso) {
                uint32_t Nm_u32 = (uint32_t)Ns[i];
                [enc setComputePipelineState:bpso];
                [enc setBuffer:wbuf     offset:woffs[i]        atIndex:0];
                [enc setBuffer:g_xbatch offset:0              atIndex:1];
                [enc setBuffer:g_ybatch offset:yoff           atIndex:2];
                [enc setBytes:&K_u32    length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&Nm_u32   length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&B_u32    length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Ns[i],1,1)
                    threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            } else {
                if (!pso) return -1;
                [enc setComputePipelineState:pso];
                [enc setBuffer:wbuf     offset:woffs[i]        atIndex:0];
                [enc setBuffer:g_xbatch offset:0              atIndex:1];
                [enc setBuffer:g_ybatch offset:yoff           atIndex:2];
                [enc setBytes:&K_u32    length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Ns[i],1,1)
                    threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            }
            [enc endEncoding];
            yoff += (size_t)Ns[i] * (size_t)B * sizeof(float);
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        /* scatter ybatch back to per-stream pointers: layout in ybatch is
         * [mat][B][N] interleaved as (mat, stream) blocks of N each. */
        const uint8_t* yb = (const uint8_t*)[g_ybatch contents];
        size_t off = 0;
        for (int i = 0; i < nmat; i++) {
            for (int s = 0; s < B; s++) {
                memcpy(ys[i][s], yb + off, (size_t)Ns[i] * sizeof(float));
                off += (size_t)Ns[i] * sizeof(float);
            }
        }
        return 0;
    }
}


/* ===== Qwen3.5 GPU-resident full-attention layer forward =====
 *
 * Persistent GPU buffers for Qwen3.5 full-attn layers. Allocated lazily.
 * These are separate from the Llama-path buffers (g_fx etc.) to avoid
 * conflicts when both architectures are loaded.
 */
static id<MTLBuffer> g_q35_x=nil, g_q35_xn=nil, g_q35_xresid=nil, g_q35_tmp=nil;
static id<MTLBuffer> g_q35_qgate=nil, g_q35_q=nil, g_q35_gate=nil;
static id<MTLBuffer> g_q35_k=nil, g_q35_v=nil, g_q35_attn=nil;
static id<MTLBuffer> g_q35_fg=nil, g_q35_fu=nil, g_q35_fa=nil;
static id<MTLBuffer> g_q35_kv_k=nil, g_q35_kv_v=nil;
static size_t g_q35_cap_h=0, g_q35_cap_qh=0, g_q35_cap_kh=0, g_q35_cap_ff=0;
static size_t g_q35_kv_cap=0;

int stratum_metal_qwen35_forward_full_attn(
    const Qwen35MetalAttnLayer* ly,
    int layer_idx,
    int H, int Hd, int Nq, int Nk, int Ff,
    int rope_dim, int position, float rope_theta,
    float rms_eps,
    int kv_slot, int kv_len, int max_kv,
    void* x_gpu, int use_internal_buffers)
{
    if (!g_device || !g_q4k_sgemv || !g_q6k_sgemv || !g_swiglu || !g_rmsnorm
        || !g_rope_half || !g_rmsnorm_ph || !g_sigmoid_gate
        || !g_split_qgate || !g_attn_gated || !g_add || !g_copy) return -1;
    @autoreleasepool {
        size_t hb  = (size_t)H*4;
        size_t qhb = (size_t)Nq*Hd*4;
        size_t khb = (size_t)Nk*Hd*4;
        size_t qghb = (size_t)2*Nq*Hd*4;  /* Q+gate concatenated */
        size_t ffb = (size_t)Ff*4;

        /* Lazily allocate persistent GPU buffers */
        if (hb > g_q35_cap_h) {
            g_q35_x      = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_q35_xn     = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_q35_xresid = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_q35_tmp    = [g_device newBufferWithLength:hb options:MTLResourceStorageModeShared];
            g_q35_cap_h = hb;
        }
        if (qghb > g_q35_cap_qh) {
            g_q35_qgate = [g_device newBufferWithLength:qghb options:MTLResourceStorageModeShared];
            g_q35_q     = [g_device newBufferWithLength:qhb options:MTLResourceStorageModeShared];
            g_q35_gate  = [g_device newBufferWithLength:qhb options:MTLResourceStorageModeShared];
            g_q35_attn  = [g_device newBufferWithLength:qhb options:MTLResourceStorageModeShared];
            g_q35_cap_qh = qghb;
        }
        if (khb > g_q35_cap_kh) {
            g_q35_k = [g_device newBufferWithLength:khb options:MTLResourceStorageModeShared];
            g_q35_v = [g_device newBufferWithLength:khb options:MTLResourceStorageModeShared];
            g_q35_cap_kh = khb;
        }
        if (ffb > g_q35_cap_ff) {
            g_q35_fg = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_q35_fu = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_q35_fa = [g_device newBufferWithLength:ffb options:MTLResourceStorageModeShared];
            g_q35_cap_ff = ffb;
        }
        /* KV cache: [max_kv][Nk*Hd] per full-attn layer slab.
         * V28: dynamic slab count based on max kv_slot seen + margin.
         * Was hardcoded 8 (V15 bug), then 32 (V16 fix). Now grows on demand. */
        size_t kvb = (size_t)max_kv * Nk * Hd * 4;
        size_t n_slabs_needed = (size_t)(kv_slot + 1);
        if (n_slabs_needed < 32) n_slabs_needed = 32;  /* minimum */
        if (n_slabs_needed > 128) n_slabs_needed = 128; /* cap for safety */
        size_t kv_total = kvb * n_slabs_needed;
        if (kv_total > g_q35_kv_cap) {
            /* First allocation or growth — zero-fill to avoid garbage */
            g_q35_kv_k = [g_device newBufferWithLength:kv_total options:MTLResourceStorageModeShared];
            g_q35_kv_v = [g_device newBufferWithLength:kv_total options:MTLResourceStorageModeShared];
            g_q35_kv_cap = kv_total;
        }
        if (!g_q35_x||!g_q35_xn||!g_q35_xresid||!g_q35_tmp||!g_q35_qgate||!g_q35_q
            ||!g_q35_gate||!g_q35_k||!g_q35_v||!g_q35_attn||!g_q35_fg||!g_q35_fu
            ||!g_q35_fa||!g_q35_kv_k||!g_q35_kv_v) return -1;

        /* Copy x into GPU buffer if using internal buffers */
        if (use_internal_buffers && x_gpu) {
            memcpy([g_q35_x contents], x_gpu, hb);
        }

        uint32_t Hu=(uint32_t)H, Hdu=(uint32_t)Hd, Nqu=(uint32_t)Nq, Nku=(uint32_t)Nk,
                 Ffu=(uint32_t)Ff, rdu=(uint32_t)rope_dim;
        uint32_t kvn=(uint32_t)(kv_len+1);
        float scale = 1.0f/sqrtf((float)Hd);
        int posi = position;
        static uint s_q35_tg = 0;
        if (!s_q35_tg) { const char* e=getenv("STRATUM_GPU_TG"); s_q35_tg = e ? (uint)atoi(e) : 64; if (s_q35_tg < 32) s_q35_tg = 32; if (s_q35_tg > 256) s_q35_tg = 256; }
        size_t kv_layer_stride = (size_t)max_kv * Nk * Hd;
        size_t kv_slot_off = (size_t)kv_slot * kv_layer_stride * 4;  /* byte offset */
        size_t kv_write_off = kv_slot_off + (size_t)kv_len * Nk * Hd * 4;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        /* Helper macros for this function */
        /* V15: ephemeral weight buffer — copies weight data into a private GPU
         * buffer using newBufferWithBytes (NOT NoCopy). This avoids wiring
         * mmap pages — the copy is a one-time CPU→GPU transfer at 270GB/s.
         * Buffer auto-releases at end of autoreleasepool. Peak extra memory
         * = one tensor (<100MB). Zero mmap page wiring. */
        __block id<MTLBuffer> ewbufs[32] = {nil};
        #define EPHEMERAL_LOOKUP(_woff, _wtb, _wb, _wo) do { \
            if (find_chunk((_woff),(_wtb),&(_wb),&(_wo)) != 0) { \
                if (!g_model_base || (_woff)+(_wtb) > g_model_size) return -1; \
                const void* _src = (const uint8_t*)g_model_base + (_woff); \
                (_wb) = [g_device newBufferWithBytes:_src length:(_wtb) \
                               options:MTLResourceStorageModeShared]; \
                if (!(_wb)) return -1; \
                for (int _ei = 0; _ei < 16; _ei++) { if (!ewbufs[_ei]) { ewbufs[_ei] = (_wb); break; } } \
                (_wo) = 0; \
            } \
        } while(0)
        #define Q35_MM(ty, woff, wtb, xbuf, ybuf, Nrows, Kdim) do { \
            id<MTLBuffer> _wb; uint64_t _wo; \
            EPHEMERAL_LOOKUP((woff),(wtb),_wb,_wo); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:((ty)==14 ? g_q6k_sgemv : (ty)==13 ? g_q5k_sgemv : g_q4k_sgemv)]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; uint32_t _K=(uint32_t)(Kdim); [_e setBytes:&_K length:4 atIndex:3]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),1,1) threadsPerThreadgroup:MTLSizeMake(s_q35_tg,1,1)]; \
            [_e endEncoding]; } while(0)
        #define Q35_MMO(ty, woff, wtb, xbuf, ybuf, yoff, Nrows, Kdim) do { \
            id<MTLBuffer> _wb; uint64_t _wo; \
            EPHEMERAL_LOOKUP((woff),(wtb),_wb,_wo); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:((ty)==14 ? g_q6k_sgemv : (ty)==13 ? g_q5k_sgemv : g_q4k_sgemv)]; \
            [_e setBuffer:_wb offset:_wo atIndex:0]; [_e setBuffer:(xbuf) offset:0 atIndex:1]; \
            [_e setBuffer:(ybuf) offset:(yoff) atIndex:2]; uint32_t _K=(uint32_t)(Kdim); [_e setBytes:&_K length:4 atIndex:3]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(Nrows),1,1) threadsPerThreadgroup:MTLSizeMake(s_q35_tg,1,1)]; \
            [_e endEncoding]; } while(0)
        #define Q35_NORM(gain_off, xbuf, ybuf) do { \
            id<MTLBuffer> _gb; uint64_t _go; EPHEMERAL_LOOKUP((gain_off),(size_t)H*4,_gb,_go); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_rmsnorm]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:0]; [_e setBuffer:_gb offset:_go atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; [_e setBytes:&Hu length:4 atIndex:3]; [_e setBytes:&rms_eps length:4 atIndex:4]; \
            [_e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [_e endEncoding]; } while(0)
        #define Q35_NORM_PH(gain_off, xbuf, ybuf, nheads) do { \
            id<MTLBuffer> _gb; uint64_t _go; EPHEMERAL_LOOKUP((gain_off),(size_t)Hd*4,_gb,_go); \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; \
            [_e setComputePipelineState:g_rmsnorm_ph]; \
            [_e setBuffer:(xbuf) offset:0 atIndex:0]; [_e setBuffer:_gb offset:_go atIndex:1]; \
            [_e setBuffer:(ybuf) offset:0 atIndex:2]; [_e setBytes:&Hdu length:4 atIndex:3]; [_e setBytes:&rms_eps length:4 atIndex:4]; \
            uint32_t _nh=(uint32_t)(nheads); [_e setBytes:&_nh length:4 atIndex:5]; \
            [_e dispatchThreadgroups:MTLSizeMake((NSUInteger)(nheads),1,1) threadsPerThreadgroup:MTLSizeMake(s_q35_tg,1,1)]; [_e endEncoding]; } while(0)
        #define Q35_ROPE_HALF(buf, nheads) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:g_rope_half]; \
            [_e setBuffer:(buf) offset:0 atIndex:0]; [_e setBytes:&Hdu length:4 atIndex:1]; [_e setBytes:&rdu length:4 atIndex:2]; \
            [_e setBytes:&posi length:4 atIndex:3]; [_e setBytes:&rope_theta length:4 atIndex:4]; \
            uint32_t _nh=(uint32_t)(nheads); [_e setBytes:&_nh length:4 atIndex:5]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)((nheads)*(rope_dim/2)),1,1) threadsPerThreadgroup:MTLSizeMake(s_q35_tg,1,1)]; [_e endEncoding]; } while(0)
        #define Q35_ELEM(pso, abuf, aoff, bbuf, boff, nn) do { \
            id<MTLComputeCommandEncoder> _e=[cmd computeCommandEncoder]; [_e setComputePipelineState:(pso)]; \
            [_e setBuffer:(abuf) offset:(aoff) atIndex:0]; [_e setBuffer:(bbuf) offset:(boff) atIndex:1]; \
            uint32_t _n=(uint32_t)(nn); [_e setBytes:&_n length:4 atIndex:2]; \
            [_e dispatchThreads:MTLSizeMake((NSUInteger)(nn),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [_e endEncoding]; } while(0)

        /* 1. Save residual: xresid = x */
        Q35_ELEM(g_copy, g_q35_xresid, 0, g_q35_x, 0, H);

        /* 2. attn_norm(x) -> xn */
        Q35_NORM(ly->attn_norm_off, g_q35_x, g_q35_xn);

        /* 3. q_proj(xn) -> qgate [2*Nq*Hd] */
        Q35_MM(ly->q_ty, ly->q_off, ly->q_tb, g_q35_xn, g_q35_qgate, 2*Nq*Hd, H);

        /* 4. split qgate -> q[Nq*Hd] + gate[Nq*Hd] */
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_split_qgate];
          [e setBuffer:g_q35_qgate offset:0 atIndex:0];
          [e setBuffer:g_q35_q offset:0 atIndex:1];
          [e setBuffer:g_q35_gate offset:0 atIndex:2];
          uint32_t NqHd = (uint32_t)(Nq*Hd); [e setBytes:&NqHd length:4 atIndex:3];
          [e dispatchThreads:MTLSizeMake((NSUInteger)(Nq*Hd),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [e endEncoding]; }

        /* 5. k_proj(xn) -> k[Nk*Hd], v_proj(xn) -> v[Nk*Hd] */
        Q35_MM(ly->k_ty, ly->k_off, ly->k_tb, g_q35_xn, g_q35_k, Nk*Hd, H);
        Q35_MM(ly->v_ty, ly->v_off, ly->v_tb, g_q35_xn, g_q35_v, Nk*Hd, H);

        /* 6. q_norm per-head (if present) */
        if (ly->q_norm_off) {
            Q35_NORM_PH(ly->q_norm_off, g_q35_q, g_q35_q, Nq);
        }
        /* 7. k_norm per-head (if present) */
        if (ly->k_norm_off) {
            Q35_NORM_PH(ly->k_norm_off, g_q35_k, g_q35_k, Nk);
        }

        /* 8. rope_half on q and k */
        Q35_ROPE_HALF(g_q35_q, Nq);
        Q35_ROPE_HALF(g_q35_k, Nk);

        /* 9. Scatter k,v into KV cache at slot kv_len */
        Q35_ELEM(g_copy, g_q35_kv_k, kv_write_off, g_q35_k, 0, Nk*Hd);
        Q35_ELEM(g_copy, g_q35_kv_v, kv_write_off, g_q35_v, 0, Nk*Hd);

        /* 10. Gated attention: q @ K^T -> softmax -> @ V -> * sigmoid(gate) */
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_attn_gated];
          [e setBuffer:g_q35_q offset:0 atIndex:0];
          [e setBuffer:g_q35_gate offset:0 atIndex:1];
          [e setBuffer:g_q35_kv_k offset:kv_slot_off atIndex:2];
          [e setBuffer:g_q35_kv_v offset:kv_slot_off atIndex:3];
          [e setBuffer:g_q35_attn offset:0 atIndex:4];
          [e setBytes:&Hdu length:4 atIndex:5]; [e setBytes:&Nqu length:4 atIndex:6];
          [e setBytes:&Nku length:4 atIndex:7]; [e setBytes:&kvn length:4 atIndex:8];
          [e setBytes:&scale length:4 atIndex:9];
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Nq,1,1) threadsPerThreadgroup:MTLSizeMake(s_q35_tg,1,1)];
          [e endEncoding]; }

        /* 11. o_proj(attn) -> tmp */
        Q35_MM(ly->o_ty, ly->o_off, ly->o_tb, g_q35_attn, g_q35_tmp, H, Nq*Hd);

        /* 12. residual: x = xresid + tmp */
        Q35_ELEM(g_add, g_q35_x, 0, g_q35_tmp, 0, H);

        /* 13. post_attn_norm(x) -> xn */
        Q35_NORM(ly->post_attn_norm_off, g_q35_x, g_q35_xn);

        /* 14. FFN: gate_proj + up_proj + swiglu + down_proj */
        Q35_MM(ly->gate_ty, ly->gate_off, ly->gate_tb, g_q35_xn, g_q35_fg, Ff, H);
        Q35_MM(ly->up_ty,   ly->up_off,   ly->up_tb,   g_q35_xn, g_q35_fu, Ff, H);
        /* swiglu: fa = silu(fg) * fu */
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_swiglu];
          [e setBuffer:g_q35_fg offset:0 atIndex:0];
          [e setBuffer:g_q35_fu offset:0 atIndex:1];
          [e setBuffer:g_q35_fa offset:0 atIndex:2];
          [e setBytes:&Ffu length:4 atIndex:3];
          [e dispatchThreads:MTLSizeMake((NSUInteger)Ff,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [e endEncoding]; }
        /* down_proj: tmp = down @ fa */
        Q35_MM(ly->down_ty, ly->down_off, ly->down_tb, g_q35_fa, g_q35_tmp, H, Ff);

        /* 15. residual: x = xresid + tmp (but xresid was saved at step 1,
         *     and x has been modified since. We need to re-save before FFN
         *     or use a different approach. Actually looking at the CPU code:
         *     x_resid is saved before attn, then x = x_resid + o_proj.
         *     Then x_resid = x (re-saved), then x = x_resid + ffn_out.
         *     So we need to save xresid again after step 12. */
        /* Fix: re-save residual after attention residual add */
        /* Actually, the macros above already do step 12 as x = xresid + tmp.
         * But xresid still holds the pre-attention value. We need:
         *   after step 12: xresid = x  (re-save)
         *   after step 15: x = xresid + tmp
         * Let me redo: step 12 should be:
         *   x = xresid + tmp  (xresid is pre-attn)
         *   xresid = x        (re-save for FFN residual)
         *   ... FFN ...
         *   x = xresid + tmp2
         */
        /* The Q35_ELEM(g_add, ...) above did x[0..H] += tmp[0..H].
         * But g_add is add_inplace: x[i] += y[i]. Since xresid was copied
         * into x at step 1 via g_copy, then x was overwritten by norm+attn.
         * Wait — step 1 copies x->xresid, step 2 reads x and writes xn.
         * x still holds original value. Step 12 does x += tmp, which is
         * x = x_original + o_proj. That's correct for attention residual.
         * But then xresid still holds x_original, not x_after_attn.
         * We need xresid = x_after_attn before FFN.
         */
        /* Re-save residual: xresid = x (after attention residual) */
        Q35_ELEM(g_copy, g_q35_xresid, 0, g_q35_x, 0, H);
        /* Now do FFN residual: x = xresid + tmp (tmp = down_proj output) */
        Q35_ELEM(g_add, g_q35_x, 0, g_q35_tmp, 0, H);

        /* Commit and wait */
        struct timespec _d0,_d1; clock_gettime(CLOCK_MONOTONIC,&_d0);
        [cmd commit]; [cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC,&_d1);
        g_n_dispatch++; g_dispatch_secs += (_d1.tv_sec-_d0.tv_sec)+(_d1.tv_nsec-_d0.tv_nsec)/1e9;

        /* Copy x back if using internal buffers */
        if (use_internal_buffers && x_gpu) {
            memcpy(x_gpu, [g_q35_x contents], hb);
        }

        #undef Q35_MM
        #undef Q35_MMO
        #undef Q35_NORM
        #undef Q35_NORM_PH
        #undef Q35_ROPE_HALF
        #undef Q35_ELEM
        #undef EPHEMERAL_LOOKUP
        return 0;
    }
}

/* ================= V54.4: batched NC submission ================= */
static id<MTLCommandBuffer> g_ncb_cmd = nil;
static id<MTLBuffer> g_ncb_xbuf = nil, g_ncb_ybuf = nil;
static size_t g_ncb_xcap = 0, g_ncb_ycap = 0, g_ncb_xpos = 0, g_ncb_ypos = 0;
typedef struct { float* dst; float* const* dsts; float* dst_ptrs[32]; size_t off, bytes; int B, N; int is_streams; } NCBatchYTask;
static NCBatchYTask g_ncb_ytasks[512];
static int g_ncb_ny = 0;
static id<MTLBuffer> g_ncb_wbufs[512];
static int g_ncb_nw = 0;   /* NoCopy weight buffers must outlive autoreleasepool: held until flush */

int stratum_metal_nc_batch_begin(void) {
    if (!g_ncb_cmd) g_ncb_cmd = [g_queue commandBuffer];
    g_ncb_ny = 0;
    g_ncb_xpos = 0;
    g_ncb_ypos = 0;
    return 0;
}

int stratum_metal_nc_batch_add(const void* wptr, size_t nbytes, int gguf_type,
                               const float* x, float* y, int N, int K, int B) {
    if (!wptr || nbytes == 0 || !g_device) return -1;
    if (g_ncb_ny >= 512) return -1;
    if (getenv("STRATUM_NC_DEBUG")) fprintf(stderr, "  [nc] add type=%d N=%d K=%d B=%d cmd=%p\n", gguf_type, N, K, B, g_ncb_cmd);
    if (!g_ncb_cmd) {
        /* no batch open (e.g. single-stream path without begin/flush):
         * fall back to synchronous per-matmul execution; caller must
         * treat this as "done now" (-2). */
        return stratum_metal_nc_sgemv2(wptr, nbytes, gguf_type, x, y, N, K, B) == 0 ? -2 : -1;
    }
    id<MTLComputePipelineState> pso, bpso;
    switch (gguf_type) {
        case 10: pso = g_q2k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q2k_sgemv_b[B] : nil; break;
        case 12: pso = g_q4k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q4k_sgemv_b[B] : nil; break;
        case 13: pso = g_q5k_sgemv;  bpso = nil; break;
        case 14: pso = g_q6k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q6k_sgemv_b[B] : nil; break;
        default: return -1;
    }
    if (!pso) return -1;
    @autoreleasepool {
        id<MTLBuffer> wbuf = [g_device newBufferWithBytesNoCopy:(void*)wptr
                                                         length:nbytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!wbuf) return -1;
        if (g_ncb_nw < 512) g_ncb_wbufs[g_ncb_nw++] = wbuf;   /* hold until flush */
        uint32_t K_u32 = (uint32_t)K;
        /* x region: always copy fresh (caller may reuse one static buffer
         * with different contents across calls — pointer identity is NOT
         * a valid reuse key) */
        int xoff;
        {
            size_t xb = (size_t)K * B * sizeof(float);
            if (g_ncb_xpos + xb > g_ncb_xcap) {
                g_ncb_xcap = (g_ncb_xcap ? g_ncb_xcap * 2 : (4u << 20));
                while (g_ncb_xpos + xb > g_ncb_xcap) g_ncb_xcap *= 2;
                g_ncb_xbuf = [g_device newBufferWithLength:g_ncb_xcap options:MTLResourceStorageModeShared];
                g_ncb_xpos = 0;
            }
            xoff = (int)g_ncb_xpos;
            memcpy((char*)[g_ncb_xbuf contents] + xoff, x, xb);
            g_ncb_xpos += xb;
        }
        /* y region */
        size_t yb = (size_t)N * B * sizeof(float);
        if (g_ncb_ypos + yb > g_ncb_ycap) {
            g_ncb_ycap = (g_ncb_ycap ? g_ncb_ycap * 2 : (4u << 20));
            while (g_ncb_ypos + yb > g_ncb_ycap) g_ncb_ycap *= 2;
            g_ncb_ybuf = [g_device newBufferWithLength:g_ncb_ycap options:MTLResourceStorageModeShared];
            g_ncb_ypos = 0;
        }
        int yoff = (int)g_ncb_ypos;
        g_ncb_ypos += yb;
        g_ncb_ytasks[g_ncb_ny].dst = y;
        g_ncb_ytasks[g_ncb_ny].dsts = NULL;
        g_ncb_ytasks[g_ncb_ny].off = (size_t)yoff;
        g_ncb_ytasks[g_ncb_ny].bytes = yb;
        g_ncb_ytasks[g_ncb_ny].B = B;
        g_ncb_ytasks[g_ncb_ny].N = N;
        g_ncb_ytasks[g_ncb_ny].is_streams = 0;
        g_ncb_ny++;

        if (B > 1 && bpso) {
            uint32_t N_u32 = (uint32_t)N, B_u32 = (uint32_t)B;
            id<MTLComputeCommandEncoder> enc = [g_ncb_cmd computeCommandEncoder];
            [enc setComputePipelineState:bpso];
            [enc setBuffer:wbuf      offset:0  atIndex:0];
            [enc setBuffer:g_ncb_xbuf offset:xoff atIndex:1];
            [enc setBuffer:g_ncb_ybuf offset:yoff atIndex:2];
            [enc setBytes:&K_u32     length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N_u32     length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&B_u32     length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        } else {
            for (int b = 0; b < B; b++) {
                id<MTLComputeCommandEncoder> enc = [g_ncb_cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:wbuf      offset:0 atIndex:0];
                [enc setBuffer:g_ncb_xbuf offset:(size_t)xoff + (size_t)b*K*sizeof(float) atIndex:1];
                [enc setBuffer:g_ncb_ybuf offset:(size_t)yoff + (size_t)b*N*sizeof(float) atIndex:2];
                [enc setBytes:&K_u32     length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                    threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                [enc endEncoding];
            }
        }
        return 0;
    }
}

int stratum_metal_nc_batch_add_streams(const void* wptr, size_t nbytes, int gguf_type,
                                      const float* x, float* const* ys, int N, int K, int B) {
    if (!wptr || nbytes == 0 || !g_device) return -1;
    if (g_ncb_ny >= 512) return -1;
    if (!g_ncb_cmd) return -2;   /* caller falls back to sync nc_sgemv2 */
    id<MTLComputePipelineState> pso, bpso;
    switch (gguf_type) {
        case 10: pso = g_q2k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q2k_sgemv_b[B] : nil; break;
        case 12: pso = g_q4k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q4k_sgemv_b[B] : nil; break;
        case 13: pso = g_q5k_sgemv;  bpso = nil; break;
        case 14: pso = g_q6k_sgemv;  bpso = (B >= 1 && B <= 32) ? g_q6k_sgemv_b[B] : nil; break;
        default: return -1;
    }
    if (!pso) return -1;
    @autoreleasepool {
        id<MTLBuffer> wbuf = [g_device newBufferWithBytesNoCopy:(void*)wptr
                                                         length:nbytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!wbuf) return -1;
        if (g_ncb_nw < 512) g_ncb_wbufs[g_ncb_nw++] = wbuf;   /* hold until flush */
        uint32_t K_u32 = (uint32_t)K;
        int xoff;
        {
            size_t xb = (size_t)K * B * sizeof(float);
            if (g_ncb_xpos + xb > g_ncb_xcap) {
                g_ncb_xcap = (g_ncb_xcap ? g_ncb_xcap * 2 : (4u << 20));
                while (g_ncb_xpos + xb > g_ncb_xcap) g_ncb_xcap *= 2;
                g_ncb_xbuf = [g_device newBufferWithLength:g_ncb_xcap options:MTLResourceStorageModeShared];
                g_ncb_xpos = 0;
            }
            xoff = (int)g_ncb_xpos;
            memcpy((char*)[g_ncb_xbuf contents] + xoff, x, xb);
            g_ncb_xpos += xb;
        }
        size_t yb = (size_t)N * B * sizeof(float);
        if (g_ncb_ypos + yb > g_ncb_ycap) {
            g_ncb_ycap = (g_ncb_ycap ? g_ncb_ycap * 2 : (4u << 20));
            while (g_ncb_ypos + yb > g_ncb_ycap) g_ncb_ycap *= 2;
            g_ncb_ybuf = [g_device newBufferWithLength:g_ncb_ycap options:MTLResourceStorageModeShared];
            g_ncb_ypos = 0;
        }
        int yoff = (int)g_ncb_ypos;
        g_ncb_ypos += yb;
        g_ncb_ytasks[g_ncb_ny].dst = NULL;
        g_ncb_ytasks[g_ncb_ny].dsts = NULL;
        for (int si = 0; si < B && si < 32; si++)
            g_ncb_ytasks[g_ncb_ny].dst_ptrs[si] = ys[si];   /* snapshot: ys array is reused */
        g_ncb_ytasks[g_ncb_ny].off = (size_t)yoff;
        g_ncb_ytasks[g_ncb_ny].bytes = yb;
        g_ncb_ytasks[g_ncb_ny].B = B;
        g_ncb_ytasks[g_ncb_ny].N = N;
        g_ncb_ytasks[g_ncb_ny].is_streams = 1;
        g_ncb_ny++;

        if (B > 1 && bpso) {
            uint32_t N_u32 = (uint32_t)N, B_u32 = (uint32_t)B;
            id<MTLComputeCommandEncoder> enc = [g_ncb_cmd computeCommandEncoder];
            [enc setComputePipelineState:bpso];
            [enc setBuffer:wbuf      offset:0  atIndex:0];
            [enc setBuffer:g_ncb_xbuf offset:xoff atIndex:1];
            [enc setBuffer:g_ncb_ybuf offset:yoff atIndex:2];
            [enc setBytes:&K_u32     length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N_u32     length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&B_u32     length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
        } else {
            for (int b = 0; b < B; b++) {
                id<MTLComputeCommandEncoder> enc = [g_ncb_cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:wbuf      offset:0 atIndex:0];
                [enc setBuffer:g_ncb_xbuf offset:(size_t)xoff + (size_t)b*K*sizeof(float) atIndex:1];
                [enc setBuffer:g_ncb_ybuf offset:(size_t)yoff + (size_t)b*N*sizeof(float) atIndex:2];
                [enc setBytes:&K_u32     length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,1,1)
                    threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                [enc endEncoding];
            }
        }
        return 0;
    }
}

int stratum_metal_nc_batch_flush(void) {
    if (!g_ncb_cmd) return 0;
    if (getenv("STRATUM_NC_DEBUG")) fprintf(stderr, "  [nc] flush cmd ny=%d\n", g_ncb_ny);
    [g_ncb_cmd commit];
    [g_ncb_cmd waitUntilCompleted];
    const char* ybase = (const char*)[g_ncb_ybuf contents];
    for (int i = 0; i < g_ncb_ny; i++) {
        const NCBatchYTask* t = &g_ncb_ytasks[i];
        if (t->is_streams) {
            for (int s2 = 0; s2 < t->B && s2 < 32; s2++)
                memcpy(t->dst_ptrs[s2], ybase + t->off + (size_t)s2 * t->N * 4, (size_t)t->N * 4);
        } else {
            memcpy(t->dst, ybase + t->off, t->bytes);
        }
    }
    g_ncb_cmd = nil;
    g_ncb_ny = 0;
    g_ncb_nw = 0;   /* release weight buffers (command buffer done) */
    return 0;
}
