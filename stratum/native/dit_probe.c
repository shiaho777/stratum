/*
 * dit_probe.c — Phase-1 spike for epic #35: sequence-shaped forward through
 * Stratum's weight-streaming primitives, shaped like a miniature video-DiT
 * block stack (packed (t,h,w) token grid, MM-RoPE, bidirectional attention,
 * timestep AdaLN-lite conditioning).
 *
 * Validates the four riskiest pieces of a video-model port:
 *   1. sequence-batched compute (per-token GEMV loops here; GEMM later),
 *   2. MM-RoPE semantics (pair 0 rotates by t, pair 1 by h, pairs 2-7 by w;
 *      angle = coord * theta^(-p/TOTAL_PAIRS)),
 *   3. bidirectional (non-causal) attention over the packed sequence,
 *   4. timestep AdaLN-lite (x_mod = rmsnorm(x)*(1+gate)+shift).
 *
 * Standalone tool on purpose (bench_/tools_ precedent): no registry ceremony
 * until the mechanism is proven. Oracle: tools/dit_oracle.py must agree.
 *
 * Usage: ./dit_probe <model.gguf> <out.bin> [timestep]
 */
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Gguf G;

static const float* ftensor(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "dit_probe: missing %s\n", name); exit(1); }
    return (const float*)(G.mmap_base + t->offset);
}

/* F16-stored matrix [out,in], row-major: y[r] = sum_c w[r,c]*x[c] */
static void lin_f16(const float* wbase, int out, int in,
                    const float* x, float* y) {
    const uint16_t* raw = (const uint16_t*)wbase;
    for (int r = 0; r < out; r++) {
        double acc = 0.0;
        const uint16_t* row = raw + (size_t)r * in;
        for (int c = 0; c < in; c++)
            acc += (double)q4k_fp16_to_fp32(row[c]) * (double)x[c];
        y[r] = (float)acc;
    }
}

static void rmsnorm(const float* x, const float* gain, int n, float eps,
                    float* y) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++) y[i] = x[i] * scale * gain[i];
}

/* MM-RoPE: pairs = hd/2; pair 0 <- t, pair 1 <- h, pairs 2..pairs-1 <- w.
 * angle(pair) = coord * theta^(-pair/pairs). Interleaved rotation. */
static void mm_rope(float* vec, int hd, int t, int h, int w, double theta) {
    int pairs = hd / 2;
    for (int p = 0; p < pairs; p++) {
        int coord = (p == 0) ? t : (p == 1) ? h : w;
        double angle = (double)coord * pow(theta, -(double)p / (double)pairs);
        float c = (float)cos(angle), s = (float)sin(angle);
        float x0 = vec[2 * p], x1 = vec[2 * p + 1];
        vec[2 * p]     = x0 * c - x1 * s;
        vec[2 * p + 1] = x0 * s + x1 * c;
    }
}

#define NAME(buf, fmt, li) (snprintf(buf, sizeof buf, fmt, li), buf)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <dit.gguf> <out.bin> [timestep]\n", argv[0]);
        return 1;
    }
    double tau = (argc > 3) ? atof(argv[3]) : 0.25;
    char nm[128];

    if (gguf_open(argv[1], &G) != 0) return 1;
    uint32_t S = 0, GT = 0, GH = 0, GW = 0, Hu = 0, NQ = 0, HDu = 0, NL = 0, FFu = 0;
    gguf_get_u32(&G, "dit.sequence_length", &S);
    gguf_get_u32(&G, "dit.grid_t", &GT);
    gguf_get_u32(&G, "dit.grid_h", &GH);
    gguf_get_u32(&G, "dit.grid_w", &GW);
    gguf_get_u32(&G, "dit.embedding_length", &Hu);
    gguf_get_u32(&G, "dit.attention.head_count", &NQ);
    gguf_get_u32(&G, "dit.attention.key_length", &HDu);
    gguf_get_u32(&G, "dit.block_count", &NL);
    gguf_get_u32(&G, "dit.feed_forward_length", &FFu);
    float theta = 10000.0f;
    gguf_get_f32(&G, "dit.rope.freq_base", &theta);

    int S_i = (int)S, H = (int)Hu, HD = (int)HDu, FF = (int)FFu, NLn = (int)NL;
    if ((int)GT * (int)GH * (int)GW != S_i || !S_i || S_i > 4096) {
        fprintf(stderr, "dit_probe: grid %ux%ux%u inconsistent with S=%u\n",
                GT, GH, GW, S);
        return 1;
    }
    fprintf(stderr,
            "  dit-probe: S=%d (%ux%ux%u) H=%d heads=%u HD=%d layers=%u tau=%.3f\n",
            S_i, GT, GH, GW, H, NQ, HD, NLn, tau);

    /* timestep embedding: omega_i = theta^(-i/(H/2)) */
    int half = H / 2;
    float te[512];
    for (int i = 0; i < half && i < 256; i++) {
        double omega = pow((double)theta, -(double)i / (double)half);
        te[i]        = (float)sin(omega * tau);
        te[i + half] = (float)cos(omega * tau);
    }

    float* x     = malloc(sizeof(float) * S_i * H);
    float* xres  = malloc(sizeof(float) * S_i * H);
    float* xn    = malloc(sizeof(float) * S_i * H);
    float* q     = malloc(sizeof(float) * S_i * H);
    float* k     = malloc(sizeof(float) * S_i * H);
    float* v     = malloc(sizeof(float) * S_i * H);
    float* attn  = malloc(sizeof(float) * S_i * H);
    float* proj  = malloc(sizeof(float) * S_i * H);
    float* gbuf  = malloc(sizeof(float) * S_i * FF);
    float* ubuf  = malloc(sizeof(float) * S_i * FF);
    float* adsh  = malloc(sizeof(float) * H);
    float* adga  = malloc(sizeof(float) * H);

    /* deterministic deterministic seed state from grid coords — no embedding
     * table in the probe; both implementations start identically */
    for (int s = 0; s < S_i; s++) {
        int t = s / ((int)GH * (int)GW);
        int h = (s / (int)GW) % (int)GH;
        int w = s % (int)GW;
        for (int d = 0; d < H; d++) {
            double ph = 0.6180339887 * (s + 1) + 0.3819660113 * d;
            x[s * H + d] = (float)(0.5 * sin(ph)) +
                           0.02f * (float)(((t * 13 + h * 7 + w * 3 + d) % 11) - 5);
        }
    }

    for (int li = 0; li < NLn; li++) {
        /* ---- attention block ---- */
        memcpy(xres, x, sizeof(float) * S_i * H);
        for (int s = 0; s < S_i; s++)
            rmsnorm(&x[s * H], ftensor(NAME(nm, "blk.%d.attn_norm.weight", li)),
                    H, 1e-5f, &xn[s * H]);
        lin_f16(ftensor(NAME(nm, "blk.%d.ada_shift.weight", li)), H, H, te, adsh);
        lin_f16(ftensor(NAME(nm, "blk.%d.ada_gate.weight", li)), H, H, te, adga);
        for (int s = 0; s < S_i * H; s++)
            xn[s] = xn[s] * (1.0f + adga[s % H]) + adsh[s % H];
        

        for (int s = 0; s < S_i; s++) {
            lin_f16(ftensor(NAME(nm, "blk.%d.attn_q.weight", li)), H, H, &xn[s * H], &q[s * H]);
            lin_f16(ftensor(NAME(nm, "blk.%d.attn_k.weight", li)), H, H, &xn[s * H], &k[s * H]);
            lin_f16(ftensor(NAME(nm, "blk.%d.attn_v.weight", li)), H, H, &xn[s * H], &v[s * H]);
        }
                for (int s = 0; s < S_i; s++) {
            int t = s / ((int)GH * (int)GW);
            int h = (s / (int)GW) % (int)GH;
            int w = s % (int)GW;
            mm_rope(&q[s * H], HD, t, h, w, (double)theta);
            mm_rope(&k[s * H], HD, t, h, w, (double)theta);
        }
                        
        /* bidirectional attention, all heads, full S×S */
        float scale = 1.0f / sqrtf((float)HD);
        float* logits = malloc(sizeof(float) * S_i);
        for (int hh = 0; hh < (int)NQ; hh++) {
            for (int qi = 0; qi < S_i; qi++) {
                const float* qh = &q[qi * H + hh * HD];
                for (int kj = 0; kj < S_i; kj++) {
                    const float* kh = &k[kj * H + hh * HD];
                    double dot = 0.0;
                    for (int d = 0; d < HD; d++) dot += (double)qh[d] * kh[d];
                    logits[kj] = (float)(dot * scale);
                }
                float mx = logits[0];
                for (int kj = 1; kj < S_i; kj++) if (logits[kj] > mx) mx = logits[kj];
                double se = 0.0;
                for (int kj = 0; kj < S_i; kj++) { logits[kj] -= mx; se += exp((double)logits[kj]); }
                float inv = (float)(1.0 / se);
                float* oh = &attn[qi * H + hh * HD];
                memset(oh, 0, sizeof(float) * HD);
                for (int kj = 0; kj < S_i; kj++) {
                    float p = expf(logits[kj]) * inv;
                    const float* vh = &v[kj * H + hh * HD];
                    for (int d = 0; d < HD; d++) oh[d] += p * vh[d];
                }
            }
        }
        free(logits);
                for (int s = 0; s < S_i; s++)
            lin_f16(ftensor(NAME(nm, "blk.%d.attn_output.weight", li)),
                    H, H, &attn[s * H], &proj[s * H]);
        for (int s = 0; s < S_i * H; s++) x[s] = xres[s] + proj[s];
        
        /* ---- MLP block ---- */
        memcpy(xres, x, sizeof(float) * S_i * H);
        for (int s = 0; s < S_i; s++)
            rmsnorm(&x[s * H], ftensor(NAME(nm, "blk.%d.mlp_norm.weight", li)),
                    H, 1e-5f, &xn[s * H]);
        lin_f16(ftensor(NAME(nm, "blk.%d.mlp_ada_shift.weight", li)), H, H, te, adsh);
        lin_f16(ftensor(NAME(nm, "blk.%d.mlp_ada_gate.weight", li)), H, H, te, adga);
                        for (int s = 0; s < S_i * H; s++)
            xn[s] = xn[s] * (1.0f + adga[s % H]) + adsh[s % H];
        
        for (int s = 0; s < S_i; s++) {
            lin_f16(ftensor(NAME(nm, "blk.%d.ffn_gate.weight", li)),
                    FF, H, &xn[s * H], &gbuf[s * FF]);
            lin_f16(ftensor(NAME(nm, "blk.%d.ffn_up.weight", li)),
                    FF, H, &xn[s * H], &ubuf[s * FF]);
        }
        for (int s = 0; s < S_i * FF; s++) {
            float gv = gbuf[s];
            gbuf[s] = (gv / (1.0f + expf(-gv))) * ubuf[s];
        }
                for (int s = 0; s < S_i; s++)
            lin_f16(ftensor(NAME(nm, "blk.%d.ffn_down.weight", li)),
                    H, FF, &gbuf[s * FF], &proj[s * H]);
        for (int s = 0; s < S_i * H; s++) x[s] = xres[s] + proj[s];
            }

    /* final norm + output head; dump all S rows */
    for (int s = 0; s < S_i; s++)
        rmsnorm(&x[s * H], ftensor("final_norm.weight"), H, 1e-5f, &xn[s * H]);
    for (int s = 0; s < S_i; s++)
        lin_f16(ftensor("output_head.weight"), H, H, &xn[s * H], &proj[s * H]);

    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror("fopen"); return 1; }
    fwrite("SDIT0001", 1, 8, out);
    uint32_t meta[2] = { S, Hu };
    fwrite(meta, 4, 2, out);
    fwrite(proj, sizeof(float), (size_t)S_i * H, out);
    fclose(out);
    fprintf(stderr, "  dit-probe: wrote %s (%d tokens x %d dims)\n", argv[2], S_i, H);
    return 0;
}
