/* h3_step.c — Phase 2: run real H3 weights through Stratum primitives.
 *
 * Reads the Q4_K quantized MiniMax-H3 GGUF, runs ONE transformer block
 * forward with production dimensions (hidden=5376, heads=56×128), and
 * reports timing + output statistics.
 */
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static Gguf G;

static const void* T(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "h3: missing '%s'\n", name); exit(1); }
    return (const void*)(G.mmap_base + t->offset);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

static void rmsnorm(float* x, const float* gain, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float sc = (float)(1.0/sqrt(ss/n + 1e-6));
    for (int i = 0; i < n; i++) x[i] *= sc * gain[i];
}

/* Q4_K matvec: y[r] = sum_c dequant(W[r,c])*x[c]
 * Weight layout: [in/256 Q4_K blocks per output] × [out outputs] */
static void q4k_gemv(const void* w, int in_dim, int out_dim,
                     const float* x, float* y) {
    int nbpr = in_dim / 256;
    const block_q4_K* blk = (const block_q4_K*)w;
    float tmp[256];
    for (int r = 0; r < out_dim; r++) {
        double acc = 0;
        const block_q4_K* row = &blk[r * nbpr];
        for (int nb = 0; nb < nbpr; nb++) {
            q4k_dequant_block_scalar(&row[nb], tmp);
            for (int c = 0; c < 256; c++)
                acc += (double)tmp[c] * x[nb*256+c];
        }
        y[r] = (float)acc;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <h3.gguf> [seq=64]\n", argv[0]); return 1; }
    int S = (argc > 2) ? atoi(argv[2]) : 64;
    if (S < 1 || S > 4096) S = 64;

    if (gguf_open(argv[1], &G) != 0) return 1;

    /* derive architecture from tensor shapes */
    int HID, HD, HEADS, QKV, FF1, FF2, NL;
    { const GgufTensor* t = gguf_find_tensor(&G, "token_refiner.final_norm.weight");
      if (!t) return 1; HID = (int)t->dims[0]; }
    { const GgufTensor* t = gguf_find_tensor(&G, "blocks.0.attn.q_norm.weight");
      if (!t) return 1; HD = (int)t->dims[0]; }
    { const GgufTensor* t = gguf_find_tensor(&G, "blocks.0.attn.qkv_proj.weight");
      if (!t) return 1; QKV = (int)t->dims[1]; HEADS = QKV / 3 / HD; }
    { const GgufTensor* t = gguf_find_tensor(&G, "blocks.0.mlp.fc1.weight");
      if (!t) return 1; FF1 = (int)t->dims[1]; }
    { const GgufTensor* t = gguf_find_tensor(&G, "blocks.0.mlp.fc2.weight");
      if (!t) return 1; FF2 = (int)t->dims[0]; }

    /* count blocks */
    NL = 0;
    for (uint64_t i = 0; i < G.n_tensors; i++) {
        if (strncmp(G.tensors[i].name, "blocks.", 7) == 0) {
            int bi = atoi(G.tensors[i].name + 7);
            if (bi + 1 > NL) NL = bi + 1;
        }
    }

    printf("H3: hidden=%d heads=%d×%d qkv=%d ff1=%d ff2=%d layers=%d seq=%d\n",
           HID, HEADS, HD, QKV, FF1, FF2, NL, S);

    /* allocate */
    size_t xsz = sizeof(float)*S*HID;
    float* x     = malloc(xsz);
    float* xres  = malloc(xsz);
    float* xn    = malloc(xsz);
    float* qkv   = malloc(sizeof(float)*S*QKV);
    float* attn  = malloc(sizeof(float)*S*HEADS*HD);
    float* proj  = malloc(xsz);
    float* mlp   = malloc(sizeof(float)*S*FF2);
    float* te    = calloc(HID, sizeof(float));

    /* deterministic input */
    srand(42);
    for (int i = 0; i < S*HID; i++)
        x[i] = (float)((rand()/(double)RAND_MAX)*2.0 - 1.0);

    char nm[160];
    clock_t t0 = clock();

    /* ===== run ALL 50 blocks ===== */
    for (int li = 0; li < NL; li++) {
        /* attention sub-block */
        memcpy(xres, x, xsz);
        snprintf(nm, sizeof nm, "blocks.%d.norm1.weight", li);
        rmsnorm(x, (const float*)T(nm), HID);

        snprintf(nm, sizeof nm, "blocks.%d.attn.qkv_proj.weight", li);
        for (int s = 0; s < S; s++)
            q4k_gemv(T(nm), HID, QKV, &x[s*HID], &qkv[s*QKV]);

        /* split Q,K,V */
        float* Qh = malloc(sizeof(float)*S*(QKV/3));
        float* Kh = malloc(sizeof(float)*S*(QKV/3));
        float* Vh = malloc(sizeof(float)*S*(QKV/3));
        int comp = QKV / 3; /* per-component dim */
        for (int s = 0; s < S; s++) {
            memcpy(&Qh[s*comp], &qkv[s*QKV], sizeof(float)*comp);
            memcpy(&Kh[s*comp], &qkv[s*QKV+comp], sizeof(float)*comp);
            memcpy(&Vh[s*comp], &qkv[s*QKV+2*comp], sizeof(float)*comp);
        }

        /* rope using inv_freq from model */
        { const float* inv_freq = (const float*)T("rope.inv_freq");
          for (int s = 0; s < S; s++)
            for (int hh = 0; hh < HEADS; hh++) {
                float* qp = &Qh[s*comp + hh*HD];
                float* kp = &Kh[s*comp + hh*HD];
                for (int k = 0; k < HD/2; k++) {
                    float ang = (float)s * inv_freq[k];
                    float c = cosf(ang), sn = sinf(ang);
                    float a0 = qp[2*k], a1 = qp[2*k+1];
                    qp[2*k] = a0*c-a1*sn; qp[2*k+1] = a0*sn+a1*c;
                    a0 = kp[2*k]; a1 = kp[2*k+1];
                    kp[2*k] = a0*c-a1*sn; kp[2*k+1] = a0*sn+a1*c;
                }
            } }

        /* QK norm */
        snprintf(nm, sizeof nm, "blocks.%d.attn.q_norm.weight", li);
        { const uint16_t* qw = (const uint16_t*)T(nm);
          for (int s = 0; s < S; s++) for (int hh = 0; hh < HEADS; hh++) {
              float* qp = &Qh[s*comp + hh*HD];
              float ss = 0; for (int d = 0; d < HD; d++) ss += qp[d]*qp[d];
              float sc = 1.0f/sqrtf(ss/HD + 1e-6f);
              for (int d = 0; d < HD; d++) qp[d] *= sc * bf16_to_f32(qw[d]);
          } }
        snprintf(nm, sizeof nm, "blocks.%d.attn.k_norm.weight", li);
        { const uint16_t* kw = (const uint16_t*)T(nm);
          for (int s = 0; s < S; s++) for (int hh = 0; hh < HEADS; hh++) {
              float* kp = &Kh[s*comp + hh*HD];
              float ss = 0; for (int d = 0; d < HD; d++) ss += kp[d]*kp[d];
              float sc = 1.0f/sqrtf(ss/HD + 1e-6f);
              for (int d = 0; d < HD; d++) kp[d] *= sc * bf16_to_f32(kw[d]);
          } }

        /* bidirectional attention */
        float scale = 1.0f/sqrtf((float)HD);
        memset(attn, 0, sizeof(float)*S*HEADS*HD);
        for (int hh = 0; hh < HEADS; hh++) {
            for (int qi = 0; qi < S; qi++) {
                const float* qh = &Qh[qi*comp + hh*HD];
                float logits[4096];
                for (int kj = 0; kj < S; kj++) {
                    const float* kh = &Kh[kj*comp + hh*HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++) dot += (double)qh[d]*kh[d];
                    logits[kj] = (float)(dot*scale);
                }
                float mx = logits[0];
                for (int j = 1; j < S; j++) if (logits[j]>mx) mx=logits[j];
                double se = 0;
                for (int j = 0; j < S; j++) { logits[j]-=mx; se+=exp((double)logits[j]); }
                float inv = (float)(1.0/se);
                float* oh = &attn[qi*HEADS*HD + hh*HD];
                for (int kj = 0; kj < S; kj++) {
                    float pv = expf(logits[kj])*inv;
                    const float* vh = &Vh[kj*comp + hh*HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv*vh[d];
                }
            }
        }

        /* output projection */
        snprintf(nm, sizeof nm, "blocks.%d.attn.out_proj.weight", li);
        for (int s = 0; s < S; s++)
            q4k_gemv(T(nm), comp, HID, &attn[s*comp], &proj[s*HID]);
        for (int i = 0; i < S*HID; i++) x[i] = xres[i] + proj[i];

        free(Qh); free(Kh); free(Vh);

        /* MLP sub-block */
        memcpy(xres, x, xsz);
        snprintf(nm, sizeof nm, "blocks.%d.norm2.weight", li);
        rmsnorm(x, (const float*)T(nm), HID);

        /* fc1: [S,HID] → [S,FF1] (fused gate+up) */
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc1.weight", li);
        { float* fc1_out = malloc(sizeof(float)*S*FF1);
          for (int s = 0; s < S; s++)
            q4k_gemv(T(nm), HID, FF1, &x[s*HID], &fc1_out[s*FF1]);

        /* split fused output: gate = first FF2, up = next FF2 */
        for (int s = 0; s < S; s++) {
            const float* fco = &fc1_out[s * FF1];
            for (int i = 0; i < FF2; i++) {
                float gv = fco[i];
                float uv = fco[FF2 + i];
                mlp[s*FF2 + i] = (gv/(1.0f+expf(-gv))) * uv;
            }
        }
        free(fc1_out); }

        /* fc2: [S,FF2] → [S,HID] */
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc2.weight", li);
        { float* proj2 = malloc(sizeof(float)*S*HID);
          for (int s = 0; s < S; s++)
            q4k_gemv(T(nm), FF2, HID, &mlp[s*FF2], &proj2[s*HID]);
          for (int i = 0; i < S*HID; i++) x[i] = xres[i] + proj2[i];
          free(proj2); }

        if (li % 10 == 0 || li == NL-1)
            fprintf(stderr, "  L%02d/%d done (%.1fs elapsed)\n",
                    li, NL-1, (double)(clock()-t0)/CLOCKS_PER_SEC);
    }

    /* final norm */
    snprintf(nm, sizeof nm, "final_layer.norm.weight");
    rmsnorm(x, (const float*)T("final_layer.norm.weight"), HID);

    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
    fprintf(stderr, "\n  H3 full forward (%d layers): %.1fs\n", NL, elapsed);

    /* stats */
    float mn = x[0], mx = x[0];
    double sum = 0;
    for (int i = 0; i < S*HID; i++) {
        if (x[i]<mn) mn=x[i]; if (x[i]>mx) mx=x[i]; sum += x[i];
    }
    printf("output: min=%.4f max=%.4f mean=%.4f\n", mn, mx, sum/(S*HID));
    printf("dumped to %s\n", "/tmp/h3_full_out.bin");

    FILE* out = fopen("/tmp/h3_full_out.bin","wb");
    fwrite("H3ALL01",1,8,out);
    uint32_t meta[2]={S,HID};
    fwrite(meta,4,2,out);
    fwrite(x,sizeof(float),S*HID,out);
    fclose(out);

    free(x);free(xres);free(xn);free(qkv);free(attn);free(proj);
    free(mlp);free(te);

    gguf_close(&G);
    return 0;
}
