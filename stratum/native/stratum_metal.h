
#ifndef STRATUM_METAL_H
#define STRATUM_METAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int stratum_metal_init(const char* metallib_path,
                       const void* model_base,
                       size_t model_size);

/* V15: Set model base/size for ephemeral dispatch without registering
 * zero-copy chunks. Allows ephemeral sgemv to compute host pointers
 * from file offsets without wiring the entire mmap. */
void stratum_metal_set_model_base(const void* model_base, size_t model_size);

void stratum_metal_shutdown(void);
void stratum_metal_dispatch_stats(long* n_out, double* secs_out);
int stratum_metal_q4k_sgemv_group(const uint64_t* offsets, const size_t* tbytes,
                                  const float* x, float* const* ys,
                                  const int* Ns, int count, int K);
int stratum_metal_ffn(uint64_t gate_off, size_t gate_tb,
                      uint64_t up_off,   size_t up_tb,
                      uint64_t down_off, size_t down_tb,
                      const float* x, float* y, int H, int Ff);

typedef struct {
    unsigned long long attn_norm_off, ffn_norm_off;
    unsigned long long q_off, k_off, v_off, o_off;
    unsigned long long gate_off, up_off, down_off;
    unsigned long      q_tb, k_tb, v_tb, o_tb, gate_tb, up_tb, down_tb;
    int q_ty, k_ty, v_ty, o_ty, gate_ty, up_ty, down_ty;
} StratumMetalLayer;

int stratum_metal_forward(const StratumMetalLayer* layers, int n_layers,
                          unsigned long long out_norm_off,
                          unsigned long long lm_off, unsigned long lm_tb, int lm_is_q6,
                          const float* x_in, float* logits_out,
                          int H, int Hd, int Nq, int Nk, int Ff, int V,
                          int rope_dim, int position, float rope_theta,
                          float rms_eps, int kv_len, int max_kv);

int stratum_metal_forward_batched(const StratumMetalLayer* layers, int n_layers,
                          unsigned long long out_norm_off,
                          unsigned long long lm_off, unsigned long lm_tb, int lm_is_q6,
                          const float* x_in, float* logits_out, int* tokens_out,
                          int H, int Hd, int Nq, int Nk, int Ff, int V,
                          int rope_dim, const int* positions, float rope_theta,
                          float rms_eps, const int* kvlens, int max_kv, int B);

int stratum_metal_q4k_sgemv(uint64_t weight_offset,
                            const float* x, float* y,
                            int N, int K);
int stratum_metal_q4k_sgemv_batched(uint64_t weight_offset,
                                    const float* x, float* y,
                                    int N, int K, int B);
int stratum_metal_q4k_multi(const uint64_t* woff_arr, const int* N_arr,
                            int nmat, const float* x, float* ys,
                            const size_t* yoff, int K, int B);
int stratum_metal_q5k_sgemv(uint64_t weight_offset,
                            const float* x, float* y,
                            int N, int K);
int stratum_metal_q6k_sgemv(uint64_t weight_offset,
                            const float* x, float* y,
                            int N, int K);

/* V6: Pre-decode Q4_K weights to F16 for faster GPU matmul.
 * Decodes the specified weight tensor from Q4_K to half (F16) and stores
 * it in a GPU buffer. Returns a handle for use with f16_sgemv.
 * Returns -1 on failure. */
int stratum_metal_predecode_q4k(uint64_t weight_offset, size_t weight_bytes,
                                 int N, int K);
/* Check if a weight tensor (by offset) has been pre-decoded. */
int stratum_metal_is_predecoded(uint64_t weight_offset);

/* V-opt: Get last token id from fused argmax (after stratum_metal_forward
 * with logits_out=NULL). Returns -1 if not available. */
int stratum_metal_get_last_token(void);

/* V9: Convert Q4_K weights to Q4_0 format for faster GPU decode.
 * Q4_0 has simpler unpacking (1 multiply vs scale-unpack+multiply+subtract).
 * Same 4.5 bits/elem density. Returns 0 on success.
 * The converted weights replace Q4_K in GPU dispatch when STRATUM_Q4_0=1. */
int stratum_metal_convert_q4k_to_q4_0(uint64_t weight_offset, size_t weight_bytes,
                                       int N, int K);

/* V2: Multi-type group dispatch — multiple matmuls with different quant
 * types sharing the same input x, ONE command buffer, ONE wait.
 * types[i]: 12=Q4_K, 13=Q5_K, 14=Q6_K. */
int stratum_metal_ssm_group(const uint64_t* offsets, const size_t* tbytes,
                            const int* types, const float* x, float* const* ys,
                            const int* Ns, int count, int K);

/* V54: zero-copy direct-read marker. Pass as `handle` to
 * stratum_metal_staged_sgemv(_async); `woff` is then interpreted as a FILE
 * offset into the mmap'd model and resolved via the pre-registered NoCopy
 * chunk (no staging copy / no pread). GPU reads of file-backed mmap pages
 * were verified to add ~0 wired memory (windowed NoCopy does NOT wire). */
#define STRATUM_METAL_NC_HANDLE (-2)

/* V54.1: per-tensor zero-copy dispatch (SAFE form). Wraps ONLY the given
 * tensor's bytes in a NoCopy buffer, dispatches, waits, releases — wired
 * memory stays bounded by the largest single tensor, NOT the whole model
 * (the whole-model chunk form wires everything -> OOM/EXIT=137). */
int stratum_metal_nc_sgemv2(const void* wptr, size_t nbytes, int gguf_type,
                            const float* x, float* y, int N, int K, int B);

int stratum_metal_register_staging(void* host_ptr, size_t nbytes);

int stratum_metal_staged_sgemv(int handle, uint64_t woff, int gguf_type,
                               const float* x, float* y,
                               int N, int K, int B);

int stratum_metal_staged_sgemv_async(int handle, uint64_t woff, int gguf_type,
                                     const float* x, float* y,
                                     int N, int K, int B);
int stratum_metal_staged_wait(void);

int stratum_metal_staged_group(const int* handles, const uint64_t* woffs,
                               const int* types, const int* Ns, int nmat,
                               const float* x, float* const* ys, int K);

/* V53: batched staged group — like staged_group but for B>1 streams sharing
 * one input x. Each matrix uses its type's batched PSO (weight reuse across
 * B); all nmat dispatches share one command buffer (one commit/wait, not
 * nmat). x layout [B][K]; ys is [nmat][B] pointers (ys[i][s] = stream s
 * output for matrix i, length Ns[i]). Falls back to single-stream when a
 * batched PSO is unavailable. */
int stratum_metal_staged_group_b(const int* handles, const uint64_t* woffs,
                                 const int* types, const int* Ns, int nmat,
                                 const float* x, float** const* ys, int K, int B);

int stratum_metal_ephemeral_sgemv(const void* weight_ptr, size_t weight_bytes,
                                  int gguf_type,
                                  const float* x, float* y,
                                  int N, int K, int B);

/* ===== Qwen3.5 GPU-resident full-attention layer forward =====
 *
 * Executes one full-attention layer entirely on GPU: rmsnorm -> q_proj ->
 * split_qgate -> k_proj -> v_proj -> q_norm -> k_norm -> rope_half ->
 * kv_scatter -> gated_attention -> o_proj -> residual_add ->
 * rmsnorm -> gate_proj -> up_proj -> swiglu -> down_proj -> residual_add.
 *
 * All intermediate buffers stay GPU-resident. One command buffer, one sync.
 * Returns 0 on success.
 */

typedef struct {
    /* weight offsets + sizes + types for all tensors in this layer */
    unsigned long long attn_norm_off;
    unsigned long long post_attn_norm_off;
    unsigned long long q_off;   unsigned long q_tb;  int q_ty;
    unsigned long long k_off;   unsigned long k_tb;  int k_ty;
    unsigned long long v_off;   unsigned long v_tb;  int v_ty;
    unsigned long long o_off;   unsigned long o_tb;  int o_ty;
    unsigned long long gate_off; unsigned long gate_tb; int gate_ty;
    unsigned long long up_off;   unsigned long up_tb;   int up_ty;
    unsigned long long down_off; unsigned long down_tb; int down_ty;
    /* Q/K norm gains (F32, head_dim elements each) */
    unsigned long long q_norm_off;   /* 0 if no q_norm */
    unsigned long long k_norm_off;   /* 0 if no k_norm */
} Qwen35MetalAttnLayer;

int stratum_metal_qwen35_forward_full_attn(
    const Qwen35MetalAttnLayer* ly,
    /* GPU-resident state (persisted across calls) */
    int layer_idx,
    int H, int Hd, int Nq, int Nk, int Ff,
    int rope_dim, int position, float rope_theta,
    float rms_eps,
    /* KV cache: [n_full_attn_layers][max_kv][Nk*Hd], GPU-resident */
    int kv_slot,           /* which layer's KV slab */
    int kv_len,            /* current length BEFORE this token */
    int max_kv,
    /* x_in/out: GPU buffer (shared), H floats */
    /* Called per-layer; x is read and written in-place via GPU buffers */
    void* x_gpu,           /* MTLBuffer contents mapped to float* */
    /* If NULL, internal persistent buffers are used */
    int use_internal_buffers
);

#ifdef __cplusplus
}
#endif

#endif

/* V54.4: batched NC (zero-copy) submission — multiple independent matmuls
 * encoded into ONE command buffer, ONE wait (measured 17.3x vs per-matmul
 * wait: 480 matmuls 3.71s -> 0.21s on M4 Pro). Engine groups layer-internal
 * independent matmuls (attn_q/k/v share the same input) into one batch.
 * begin -> add* -> flush. flush copies all y results back. */
int stratum_metal_nc_batch_begin(void);
int stratum_metal_nc_batch_add(const void* wptr, size_t nbytes, int gguf_type,
                               const float* x, float* y, int N, int K, int B);
int stratum_metal_nc_batch_flush(void);

/* V54.4b: batch_add with per-stream y targets (multix ys[] array). The
 * continuous-buffer form (batch_add) breaks when several matmuls in one
 * batch share one caller-side scratch buffer (static ypack overwrite).
 * flush() copies each stream's result straight into ys[s]. */
int stratum_metal_nc_batch_add_streams(const void* wptr, size_t nbytes, int gguf_type,
                                       const float* x, float* const* ys, int N, int K, int B);
