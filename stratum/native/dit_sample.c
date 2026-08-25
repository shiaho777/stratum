/*
 * dit_sample.c — Phase-2 spike: flow-matching sampling loop for DiT models.
 *
 * Implements:
 *   - Flow-matching Euler ODE solver: x_{t+1} = x_t + dt * v_θ(x_t, t)
 *     where dt = 1/n_steps, t goes from 0 to 1.
 *   - Tiled bidirectional attention: processes Q,K,V in blocks of
 *     ATTN_BLOCK to bound memory. Never materializes full S×S matrix.
 *   - Sequence-batched linear: loops over tokens calling GEMV (correctness-
 *     first; cblas_sgemm upgrade is a Phase-3 optimization).
 *
 * Usage: ./dit_sample <model.gguf> <output.bin> [steps] [timestep_start]
 *
 * Validation strategy:
 *   With steps=1, output should equal a single forward pass (the Phase-1
 *   dit_probe result). With steps>1, each intermediate x should have
 *   monotonically decreasing norm (flow-matching trajectories converge).
 */
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- config from GGUF ---- */
static Gguf G;
static int S, H, HD, HEADS, FF, NL;
static double THETA;

static const float* ftensor(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(1); }
    return (const float*)(G.mmap_base + t->offset);
}
static const float* W(const char* fmt, int li) {
    char nm[128];
    int nw = snprintf(nm, sizeof nm, fmt, li);
    if (nw < 0 || nw >= 128) { fprintf(stderr, "W: snprintf failed (%d) fmt=%s li=%d\n", nw, fmt, li); exit(1); }
    const GgufTensor* t = gguf_find_tensor(&G, nm);
    if (!t) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    return (const float*)(G.mmap_base + t->offset);
}

/* F16 matrix-vector: y[r] = sum_c w[r][c]*x[c], double accumulation */
static void gemv_f16(const float* w, int out, int in, const float* x, float* y) {
    const uint16_t* raw = (const uint16_t*)w;
    for (int r = 0; r < out; r++) {
        double acc = 0.0;
        const uint16_t* row = raw + (size_t)r * in;
        for (int c = 0; c < in; c++)
            acc += (double)q4k_fp16_to_fp32(row[c]) * (double)x[c];
        y[r] = (float)acc;
    }
}

/* Sequence matmul: Y[s][out] = W @ X[s][in] for all s tokens */
static void seq_linear(const float* w, int out, int in,
                       const float* X, float* Y, int n_tokens) {
    for (int s = 0; s < n_tokens; s++)
        gemv_f16(w, out, in, &X[s * in], &Y[s * out]);
}

static void rmsnorm_seq(float* x, const float* gain, int n_tokens, int dim, float eps) {
    for (int s = 0; s < n_tokens; s++) {
        float* xs = &x[s * dim];
        double ss = 0.0;
        for (int i = 0; i < dim; i++) ss += (double)xs[i] * xs[i];
        float scale = (float)(1.0 / sqrt(ss / (double)dim + (double)eps));
        for (int i = 0; i < dim; i++) xs[i] *= scale * gain[i];
    }
}

/* MM-RoPE: pair 0←t, pair 1←h, pairs 2..←w; angle = coord*theta^(-p/pairs) */
static void mm_rope(float* vec, int hd, int t, int h, int w) {
    int pairs = hd / 2;
    for (int p = 0; p < pairs; p++) {
        int coord = (p == 0) ? t : (p == 1) ? h : w;
        double angle = (double)coord * pow(THETA, -(double)p / (double)pairs);
        float c = (float)cos(angle), s = (float)sin(angle);
        float x0 = vec[2*p], x1 = vec[2*p+1];
        vec[2*p]   = x0*c - x1*s;
        vec[2*p+1] = x0*s + x1*c;
    }
}

/* Tiled bidirectional attention: never materializes S×S matrix.
 * Processes query blocks of ATTN_BLOCK rows at a time.
 * For each block: computes scores against ALL keys, softmaxes the block,
 * accumulates weighted values. Memory: O(ATTN_BLOCK × S) per head. */
#define ATTN_BLOCK 256

static void attention_tiled(const float* Q, const float* K, const float* V,
                            float* Out, int n_tokens, int n_heads, int hd) {
    int H = n_heads * hd;
    float scale = 1.0f / sqrtf((float)hd);
    float* scores = malloc(sizeof(float) * ATTN_BLOCK * n_tokens);
    float* probs  = malloc(sizeof(float) * ATTN_BLOCK * n_tokens);

    for (int hh = 0; hh < n_heads; hh++) {
        for (int qb = 0; qb < n_tokens; qb += ATTN_BLOCK) {
            int qe = qb + ATTN_BLOCK > n_tokens ? n_tokens : qb + ATTN_BLOCK;
            int qlen = qe - qb;

            /* scores[qlen][n_tokens] = Q[qb:qe] @ K^T for this head */
            for (int qi = 0; qi < qlen; qi++) {
                const float* qh = &Q[(qb+qi)*H + hh*hd];
                for (int kj = 0; kj < n_tokens; kj++) {
                    const float* kh = &K[kj*H + hh*hd];
                    double dot = 0.0;
                    for (int d = 0; d < hd; d++) dot += (double)qh[d]*kh[d];
                    scores[qi*n_tokens+kj] = (float)(dot*scale);
                }
            }

            /* row-wise softmax */
            for (int qi = 0; qi < qlen; qi++) {
                float* row = &scores[qi*n_tokens];
                float mx = row[0];
                for (int j=1;j<n_tokens;j++) if(row[j]>mx) mx=row[j];
                double se = 0.0;
                for (int j=0;j<n_tokens;j++){row[j]-=mx;se+=exp((double)row[j]);}
                float inv=(float)(1.0/se);
                for (int j=0;j<n_tokens;j++) probs[qi*n_tokens+j]=expf(row[j])*inv;
            }

            /* weighted sum: Out[qb:qe, head] = probs @ V[:, head] */
            for (int qi = 0; qi < qlen; qi++) {
                float* oh = &Out[(qb+qi)*H + hh*hd];
                memset(oh, 0, sizeof(float)*HD);
                for (int kj = 0; kj < n_tokens; kj++) {
                    float pv = probs[qi*n_tokens+kj];
                    const float* vh = &V[kj*H + hh*HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv*vh[d];
                }
            }
        }
    }
    free(scores); free(probs);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> [steps=20] [tau=0.25]\n", argv[0]);
        return 1;
    }
    int n_steps = (argc > 3) ? atoi(argv[3]) : 20;
    double tau  = (argc > 4) ? atof(argv[4]) : 0.25;

    if (gguf_open(argv[1], &G) != 0) return 1;

    /* read config */
    uint32_t Su=0,Hu=0,NQu=0,HDu=0,NL=0,FFu=0,GT=0,GH=0,GW=0;
    gguf_get_u32(&G, "dit.sequence_length", &Su);
    gguf_get_u32(&G, "dit.embedding_length", &Hu);
    gguf_get_u32(&G, "dit.attention.head_count", &NQu);
    gguf_get_u32(&G, "dit.attention.key_length", &HDu);
    gguf_get_u32(&G, "dit.block_count", &NL);
    gguf_get_u32(&G, "dit.feed_forward_length", &FFu);
    gguf_get_u32(&G, "dit.grid_t", &GT);
    gguf_get_u32(&G, "dit.grid_h", &GH);
    gguf_get_u32(&G, "dit.grid_w", &GW);
    gguf_get_f32(&G, "dit.rope.freq_base", &THETA);

    S=(int)Su; H=(int)Hu; HD=(int)HDu; HEADS=(int)NQu; FF=(int)FFu; NL=NL;

    fprintf(stderr, "  dit-sample: S=%d H=%d heads=%d HD=%d layers=%u steps=%d tau=%.3f\n",
            S,H,HEADS,HD,NL,n_steps,tau);

    /* allocate state */
    size_t xsz = sizeof(float)*S*H;
    float* x     = malloc(xsz);      /* current latent */
    float* x0    = malloc(xsz);      /* noise init */
    float* xres  = malloc(xsz);
    float* xn    = malloc(xsz);
    float* q     = malloc(xsz);
    float* k     = malloc(xsz);
    float* v     = malloc(xsz);
    float* attn  = malloc(xsz);
    float* proj  = malloc(xsz);
    float* te    = calloc(H, sizeof(float));
    float* adsh  = calloc(H, sizeof(float));
    float* adga  = calloc(H, sizeof(float));
    float* gb    = malloc(sizeof(float)*S*FF);
    float* ub    = malloc(sizeof(float)*S*FF);
    char nm[128];

    /* deterministic noise init (seeded) */
    srand(42);
    for (int i = 0; i < S*H; i++) x0[i] = (float)((rand()/(double)RAND_MAX)*2.0 - 1.0);

    /* grid coords for MM-RoPE */
    int* coords_t = malloc(sizeof(int)*S);
    int* coords_h = malloc(sizeof(int)*S);
    int* coords_w = malloc(sizeof(int)*S);
    for (int s = 0; s < S; s++) {
        coords_t[s] = s / ((int)GH*(int)GW);
        coords_h[s] = (s/(int)GW) % (int)GH;
        coords_w[s] = s % (int)GW;
    }

    /* timestep embedding function */
    #define TIME_EMB(t_out, t_val) do { \
        int half = H/2; \
        for (int i = 0; i < half && i < 256; i++) { \
            double omega = pow(THETA, -(double)i/(double)half); \
            (t_out)[i]        = (float)sin(omega*(t_val)); \
            (t_out)[i + half] = (float)cos(omega*(t_val)); \
        } \
    } while(0)

    /* ===== flow-matching sampling loop ===== */
    memcpy(x, x0, xsz);
    double dt = 1.0 / n_steps;
    clock_t t_start = clock();

    for (int step = 0; step < n_steps; step++) {
        double t_curr = (double)step / n_steps;
        TIME_EMB(te, t_curr);

        /* ----- transformer forward (single step) ----- */
        for (int li = 0; li < NL; li++) {
            /* attention block */
            memcpy(xres, x, xsz);
            rmsnorm_seq(x, W("blk.%d.attn_norm.weight", li), S, H, 1e-5f);

            /* AdaLN-lite */
            snprintf(nm, sizeof nm, "blk.%d.ada_shift.weight", li);
            seq_linear(ftensor(nm), H, H, te, adsh, 1);
            snprintf(nm, sizeof nm, "blk.%d.ada_gate.weight", li);
            seq_linear(ftensor(nm), H, H, te, adga, 1);
            for (int i = 0; i < S*H; i++)
                x[i] = x[i]*(1.0f+adga[i%H]) + adsh[i%H];

            /* qkv projection */
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", li);
            seq_linear(ftensor(nm), H, H, x, q, S);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", li);
            seq_linear(ftensor(nm), H, H, x, k, S);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", li);
            seq_linear(ftensor(nm), H, H, x, v, S);

            /* MM-RoPE */
            for (int s = 0; s < S; s++) {
                mm_rope(&q[s*H], HD, coords_t[s], coords_h[s], coords_w[s]);
                mm_rope(&k[s*H], HD, coords_t[s], coords_h[s], coords_w[s]);
            }

            /* bidirectional tiled attention */
            attention_tiled(q, k, v, attn, S, HEADS, HD);

            /* output projection + residual */
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", li);
            seq_linear(ftensor(nm), H, H, attn, proj, S);
            for (int i = 0; i < S*H; i++) x[i] = xres[i] + proj[i];

            /* MLP block */
            memcpy(xres, x, xsz);
            rmsnorm_seq(x, W("blk.%d.mlp_norm.weight", li), S, H, 1e-5f);

            snprintf(nm, sizeof nm, "blk.%d.mlp_ada_shift.weight", li);
            seq_linear(ftensor(nm), H, H, te, adsh, 1);
            snprintf(nm, sizeof nm, "blk.%d.mlp_ada_gate.weight", li);
            seq_linear(ftensor(nm), H, H, te, adga, 1);
            for (int i = 0; i < S*H; i++)
                x[i] = x[i]*(1.0f+adga[i%H]) + adsh[i%H];

            snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", li);
            seq_linear(ftensor(nm), FF, H, x, gb, S);
            snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", li);
            seq_linear(ftensor(nm), FF, H, x, ub, S);
            for (int i = 0; i < S*FF; i++) {
                float gv = gb[i];
                gb[i] = (gv/(1.0f+expf(-gv))) * ub[i];
            }
            snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", li);
            seq_linear(ftensor(nm), H, FF, gb, proj, S);
            for (int i = 0; i < S*H; i++) x[i] = xres[i] + proj[i];
        }

        /* ----- final norm + head → velocity prediction ----- */
        rmsnorm_seq(x, ftensor("final_norm.weight"), S, H, 1e-5f);
        seq_linear(ftensor("output_head.weight"), H, H, x, proj, S);

        /* ----- Euler step: x += dt * v_pred ----- */
        for (int i = 0; i < S*H; i++) x[i] = x0[i] + (float)(dt*(step+1)) * proj[i];

        /* progress */
        double l2 = 0; for (int i = 0; i < S*H; i++) l2 += (double)x[i]*x[i];
        fprintf(stderr, "  step %d/%d t=%.3f |x|²=%.3f (%.1fs elapsed)\n",
                step+1, n_steps, t_curr+dt, l2,
                (double)(clock()-t_start)/CLOCKS_PER_SEC);
    }

    /* ===== dump final latent ===== */
    FILE* out = fopen(argv[2], "wb");
    fwrite("SDIT0001", 1, 8, out);
    uint32_t meta[2] = {(uint32_t)S, (uint32_t)H};
    fwrite(meta, 4, 2, out);
    fwrite(proj, sizeof(float), S*H, out);
    fclose(out);

    double elapsed = (double)(clock()-t_start)/CLOCKS_PER_SEC;
    fprintf(stderr, "  done: %d steps in %.1fs → %s\n", n_steps, elapsed, argv[2]);

    /* cleanup */
    free(x);free(x0);free(xres);free(xn);free(q);free(k);free(v);
    free(attn);free(proj);free(te);free(adsh);free(adga);free(gb);free(ub);
    free(coords_t);free(coords_h);free(coords_w);
    return 0;
}
