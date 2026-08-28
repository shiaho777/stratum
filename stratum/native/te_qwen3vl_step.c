/* te_qwen3vl_step.c — H3 pipeline M1: run the bundled Qwen3-VL-32B
 * text-encoder checkpoint (unsloth qwen3vl_32b_minimax_h3-Q4_K_M.gguf)
 * through Stratum primitives, text tokens in -> text_states [L, 5120] out.
 *
 * Checkpoint semantics (from ComfyUI Qwen3VL_32BConfig):
 *   - first 50 of the original 64 layers, NO final norm, NO lm_head;
 *     output = hidden state straight out of layer 50
 *   - hidden=5120, ffn 25600, 64 heads x 128, GQA n_kv=? (probe at load)
 *   - rope theta 5e6, QK-norm (per-head RMSNorm) BEFORE split-half rope,
 *     causal attention, RMS eps 1e-6 (Qwen3 family)
 *   - spike passes plain sequential position ids (pure text — the H3
 *     denoiser conditions on text only; visual towers are separate)
 *
 * Weights are streamed via mmap exactly like h3_step; only activations
 * are anonymous. Diagnostics: per-layer |x| stats, final text_states
 * dumped in "Q3TE0001" format.
 */
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include "stratum_q6k.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static Gguf G;

static const void* T(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "te: missing '%s'\n", name); exit(1); }
    return (const void*)(G.mmap_base + t->offset);
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}
static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

static void act_stats(const char* tag, const float* x, int n) {
    float mx = 0; double sm = 0; int nan = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (isnan(x[i]) || isinf(x[i])) { nan++; continue; }
        if (a > mx) mx = a;
        sm += a;
    }
    fprintf(stderr, "    %-10s |x|mean=%.6g max=%.6g %s\n",
            tag, nan ? 0.0 : sm / n, mx,
            nan ? (nan == n ? "ALL-NAN/INF" : "(nan contaminated)") : "");
}

/* per-token RMSNorm with BF16 gain (Qwen3: eps 1e-6, gains type=30) */
static void rmsnorm_rows(float* x, const uint16_t* g16, int S, int n) {
    for (int s = 0; s < S; s++) {
        float* r = x + (size_t)s * n;
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)r[i]*r[i];
        float sc = (float)(1.0/sqrt(ss/n + 1e-6));
        for (int i = 0; i < n; i++) r[i] *= sc * bf16_to_f32(g16[i]);
    }
}

/* mixed-quant matvec: unsloth mixed Q4_K (q/k/o/gate/up) with Q6_K
 * (v_proj/down_proj — 210 B per 256, type 14); dispatch on tensor type.
 * Rows are [out][K/256 blocks] in both layouts. */
static void mixed_gemv(const GgufTensor* t, int in_dim, int out_dim,
                       const float* x, float* y) {
    int nbpr = in_dim / 256;
    float tmp[256];
    const void* base = (const void*)(G.mmap_base + t->offset);
    for (int r = 0; r < out_dim; r++) {
        double acc = 0;
        if ((GgmlType)t->type == GGML_TYPE_Q4_K) {
            const block_q4_K* row = (const block_q4_K*)base + (size_t)r * nbpr;
            for (int nb = 0; nb < nbpr; nb++) {
                q4k_dequant_block_scalar(&row[nb], tmp);
                for (int c = 0; c < 256; c++)
                    acc += (double)tmp[c] * x[nb*256+c];
            }
        } else if ((GgmlType)t->type == GGML_TYPE_Q6_K) {
            const block_q6_K* row = (const block_q6_K*)base + (size_t)r * nbpr;
            for (int nb = 0; nb < nbpr; nb++) {
                q6k_dequant_block_scalar(&row[nb], tmp);
                for (int c = 0; c < 256; c++)
                    acc += (double)tmp[c] * x[nb*256+c];
            }
        } else {
            fprintf(stderr, "te: unsupported quant %u on r=%d\n", t->type, r);
            exit(1);
        }
        y[r] = (float)acc;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <te.gguf> [seq=16] [tok0 tok1 ...]\n", argv[0]);
        return 1;
    }
    int S = (argc > 2) ? atoi(argv[2]) : 16;
    if (S < 1 || S > 4096) S = 16;

    if (gguf_open(argv[1], &G) != 0) return 1;

    /* probe geometry from tensors */
    int HID, FF, HD, NQ, NK, NL;
    { const GgufTensor* t = gguf_find_tensor(&G, "model.embed_tokens.weight");
      if (!t) return 1; HID = (int)t->dims[0]; }
    { const GgufTensor* t = gguf_find_tensor(&G, "model.layers.0.mlp.gate_proj.weight");
      if (!t) return 1; FF = (int)t->dims[1]; }
    { const GgufTensor* t = gguf_find_tensor(&G, "model.layers.0.self_attn.q_proj.weight");
      if (!t) return 1; NQ = (int)t->dims[1] / 128; HD = 128; }
    { const GgufTensor* t = gguf_find_tensor(&G, "model.layers.0.self_attn.k_proj.weight");
      if (!t) return 1; NK = (int)t->dims[1] / 128; }
    NL = 0;
    for (uint64_t i = 0; i < G.n_tensors; i++) {
        const char* n = G.tensors[i].name;
        if (!strncmp(n, "model.layers.", 13)) {
            int bi = atoi(n + 13);
            if (bi + 1 > NL) NL = bi + 1;
        }
    }
    printf("Qwen3VL-TE: hidden=%d ffn=%d heads=%dx%d kv=%d layers=%d seq=%d\n",
           HID, FF, NQ, HD, NK, NL, S);

    /* token ids: argv or deterministic */
    int* toks = malloc(sizeof(int)*S);
    if (argc > 3) {
        for (int i = 3; i < argc && i - 3 < S; i++) toks[i-3] = atoi(argv[i]);
        for (int i = argc - 3; i < S; i++) toks[i] = toks[0];
    } else {
        for (int i = 0; i < S; i++) toks[i] = (i % 151936);
    }

    /* embedding: bf16 rows [vocab, HID] */
    float* x = malloc(sizeof(float)*S*HID);
    { const uint16_t* emb = (const uint16_t*)T("model.embed_tokens.weight");
      for (int s = 0; s < S; s++)
        for (int i = 0; i < HID; i++)
            x[(size_t)s*HID + i] = bf16_to_f32(emb[(size_t)toks[s]*HID + i]); }
    act_stats("embed", x, S*HID);

    float* qkv = malloc(sizeof(float)*S*(NQ+NK+NK)*HD);
    float* xres = malloc(sizeof(float)*S*HID);
    float* attnout = malloc(sizeof(float)*S*NQ*HD);
    float* ffn = malloc(sizeof(float)*S*FF);
    clock_t t0 = clock();

    for (int li = 0; li < NL; li++) {
        char nm[160];
        /* --- attention --- */
        memcpy(xres, x, sizeof(float)*S*HID);
        snprintf(nm, sizeof nm, "model.layers.%d.input_layernorm.weight", li);
        rmsnorm_rows(x, (const uint16_t*)T(nm), S, HID);

        snprintf(nm, sizeof nm, "model.layers.%d.self_attn.q_proj.weight", li);
        { const GgufTensor* w = gguf_find_tensor(&G, nm);
          for (int s = 0; s < S; s++)
            mixed_gemv(w, HID, NQ*HD, &x[s*HID], &qkv[s*(NQ+2*NK)*HD]); }
        snprintf(nm, sizeof nm, "model.layers.%d.self_attn.k_proj.weight", li);
        { const GgufTensor* w = gguf_find_tensor(&G, nm);
          for (int s = 0; s < S; s++)
            mixed_gemv(w, HID, NK*HD, &x[s*HID], &qkv[s*(NQ+2*NK)*HD + NQ*HD]); }
        snprintf(nm, sizeof nm, "model.layers.%d.self_attn.v_proj.weight", li);
        { const GgufTensor* w = gguf_find_tensor(&G, nm);
          for (int s = 0; s < S; s++)
            mixed_gemv(w, HID, NK*HD, &x[s*HID], &qkv[s*(NQ+2*NK)*HD + (NQ+NK)*HD]); }

        /* per-head QK norm BEFORE rope (q_norm/k_norm are BF16 [128]) */
        snprintf(nm, sizeof nm, "model.layers.%d.self_attn.q_norm.weight", li);
        const uint16_t* qw = (const uint16_t*)T(nm);
        snprintf(nm, sizeof nm, "model.layers.%d.self_attn.k_norm.weight", li);
        const uint16_t* kw = (const uint16_t*)T(nm);
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < NQ; h++) {
                float* qp = &qkv[s*(NQ+2*NK)*HD + h*HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)qp[d]*qp[d];
                float sc = (float)(1.0/sqrt(ss/HD + 1e-6));
                for (int d = 0; d < HD; d++) qp[d] *= sc * bf16_to_f32(qw[d]);
            }
            for (int h = 0; h < NK; h++) {
                float* kp = &qkv[s*(NQ+2*NK)*HD + NQ*HD + h*HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)kp[d]*kp[d];
                float sc = (float)(1.0/sqrt(ss/HD + 1e-6));
                for (int d = 0; d < HD; d++) kp[d] *= sc * bf16_to_f32(kw[d]);
            }
        }

        /* split-half rope, theta 5e6, sequential position = s
         * (pure-text spike: the interleaved-MRoPE T/H/W mixing reduces to
         * position s on every pair because text positions are equal on all
         * three axes) */
        for (int s = 0; s < S; s++) {
            float base_q[(NQ+2*NK)*HD];
            (void)base_q;
            for (int h = 0; h < NQ; h++) {
                float* q = &qkv[s*(NQ+2*NK)*HD + h*HD];
                for (int k = 0; k < HD/2; k++) {
                    float inv = powf(5000000.0f, -2.0f*k/(float)HD);
                    float ang = (float)s * inv;
                    float c = cosf(ang), sn = sinf(ang);
                    float a0 = q[k], a1 = q[HD/2 + k];
                    q[k]        = a0*c - a1*sn;
                    q[HD/2 + k] = a0*sn + a1*c;
                }
            }
            for (int h = 0; h < NK; h++) {
                float* kp = &qkv[s*(NQ+2*NK)*HD + NQ*HD + h*HD];
                for (int k = 0; k < HD/2; k++) {
                    float inv = powf(5000000.0f, -2.0f*k/(float)HD);
                    float ang = (float)s * inv;
                    float c = cosf(ang), sn = sinf(ang);
                    float a0 = kp[k], a1 = kp[HD/2 + k];
                    kp[k]        = a0*c - a1*sn;
                    kp[HD/2 + k] = a0*sn + a1*c;
                }
            }
        }

        /* causal attention (prefill: token s attends [0..s]) */
        float scale = 1.0f/sqrtf((float)HD);
        memset(attnout, 0, sizeof(float)*S*NQ*HD);
        for (int h = 0; h < NQ; h++) {
            int kvh = h * NK / NQ;
            for (int s = 0; s < S; s++) {
                const float* q = &qkv[s*(NQ+2*NK)*HD + h*HD];
                float lg[4096];
                for (int t = 0; t <= s; t++) {
                    const float* k = &qkv[t*(NQ+2*NK)*HD + NQ*HD + kvh*HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++) dot += (double)q[d]*k[d];
                    lg[t] = (float)(dot*scale);
                }
                float mx = lg[0];
                for (int j = 1; j <= s; j++) if (lg[j] > mx) mx = lg[j];
                double se = 0;
                for (int j = 0; j <= s; j++) { lg[j] -= mx; se += exp((double)lg[j]); }
                float inv = (float)(1.0/se);
                float* oh = &attnout[s*NQ*HD + h*HD];
                for (int t = 0; t <= s; t++) {
                    float pv = expf(lg[t])*inv;
                    const float* v = &qkv[t*(NQ+2*NK)*HD + (NQ+NK)*HD + kvh*HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv*v[d];
                }
            }
        }

        snprintf(nm, sizeof nm, "model.layers.%d.self_attn.o_proj.weight", li);
        { const GgufTensor* w = gguf_find_tensor(&G, nm);
          for (int s = 0; s < S; s++)
            mixed_gemv(w, NQ*HD, HID, &attnout[s*NQ*HD], &x[s*HID]); }
        for (int i = 0; i < S*HID; i++) x[i] += xres[i];

        /* --- mlp (swiglu: gate then up, comma layout from ComfyUI) --- */
        memcpy(xres, x, sizeof(float)*S*HID);
        snprintf(nm, sizeof nm, "model.layers.%d.post_attention_layernorm.weight", li);
        rmsnorm_rows(x, (const uint16_t*)T(nm), S, HID);

        snprintf(nm, sizeof nm, "model.layers.%d.mlp.gate_proj.weight", li);
        { const GgufTensor* w = gguf_find_tensor(&G, nm);
          for (int s = 0; s < S; s++)
            mixed_gemv(w, HID, FF, &x[s*HID], &ffn[s*FF]); }
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.up_proj.weight", li);
        { const GgufTensor* w = gguf_find_tensor(&G, nm);
          for (int s = 0; s < S; s++)
            mixed_gemv(w, HID, FF, &x[s*HID], &ffn[s*FF + FF]); }
        for (int s = 0; s < S; s++) {
            float* f0 = &ffn[s*FF];
            float* up = &ffn[s*FF + FF];
            for (int i = 0; i < FF; i++) {
                float gv = f0[i];
                f0[i] = (gv/(1.0f+expf(-gv))) * up[i];
            }
        }
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.down_proj.weight", li);
        { const GgufTensor* w = gguf_find_tensor(&G, nm);
          for (int s = 0; s < S; s++)
            mixed_gemv(w, FF, HID, &ffn[s*FF], &x[s*HID]); }
        for (int i = 0; i < S*HID; i++) x[i] += xres[i];

        if (li % 10 == 0 || li == NL-1)
            act_stats("post-mlp", x, S*HID);
    }

    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
    fprintf(stderr, "\n  TE forward (%d layers): %.1fs\n", NL, elapsed);

    /* NO final norm by design (Qwen3VL_32BConfig): dump raw hidden */
    const char* out_path = "/tmp/h3_text_states.bin";
    FILE* out = fopen(out_path, "wb");
    fwrite("Q3TE0001", 1, 8, out);
    uint32_t meta[3] = { (uint32_t)S, (uint32_t)HID, (uint32_t)NL };
    fwrite(meta, 4, 3, out);
    fwrite(x, sizeof(float), S*HID, out);
    fclose(out);
    printf("text_states: [S=%d, HID=%d] -> %s\n", S, HID, out_path);
    act_stats("final", x, S*HID);

    free(x); free(xres); free(qkv); free(attnout); free(ffn); free(toks);
    gguf_close(&G);
    return 0;
}
