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

/* V221 diagnostics: activation stats after every sub-block, so the first
 * layer that goes non-finite is named instead of inferred from a final
 * NaN soup. |x| mean + max keeps the growth curve visible BEFORE the
 * explosion (a slowly rising max foretells the layer). */
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

/* V221: H3 rotates only the first 48 pairs of each 128-dim head
 * (rope_inv_freq_len=16 per axis, t/h/w concatenated); head_dim 128 ->
 * rot region is dims [0..48) and [64..112), tail 32 dims pass through.
 * (rot_dim = 96 in reference terms.) */
#define ROT_PAIRS 48

static void rmsnorm(float* x, const float* gain, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float sc = (float)(1.0/sqrt(ss/n + 1e-6));
    for (int i = 0; i < n; i++) x[i] *= sc * gain[i];
}

/* V221: H3 stores ALL RMSNorm gains as BF16 (type=30): norm1/norm2/
 * final_layer.norm included — the original spike read them as native
 * fp32, so every channel multiplied a garbage reinterpretation
 * (observed h(max)=17 vs correct upper bound ~3). Decode per element. */
static void rmsnorm_bf16(float* x, const uint16_t* g16, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float sc = (float)(1.0/sqrt(ss/n + 1e-5));   /* norm_eps=1e-5 */
    for (int i = 0; i < n; i++) x[i] *= sc * bf16_to_f32(g16[i]);
}

/* V221: per-token RMSNorm over an [S, n] row-major stream (the reference
 * normalizes each token independently; the batch-wide variant was a
 * semantic bug mixing energy across rows). */
static void rmsnorm_bf16_rows(float* x, const uint16_t* g16, int S, int n) {
    for (int s = 0; s < S; s++)
        rmsnorm_bf16(x + (size_t)s * n, g16, n);
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

    /* V221: AdaLN conditioning — THE missing piece of the old spike.
     * Reference: h = norm1(x)*(1+scale_msa)+shift_msa; residual absorbs
     * gate_msa*attn(h) (and the mlp trio likewise). Without gates the
     * full-magnitude branch outputs stack unattenuated for 50 layers ->
     * measured blowup L0 ->1.8e7, L1 ->4.9e19, inf at L2, NaN after.
     *
     * Curve-form checkpoint: adaln_t_table is fp32, ggml dims
     * [ne0=T_DIM basis cols (contiguous), ne1=1025 rows]; the producer
     * held ROWS of an 8-dim basis — t in [0,1] lerps two adjacent rows
     * into t_emb[8] (apply_silu=False for curve checkpoints). Spike is
     * all-video: modality tag(video)=0. Row stride is ne0 (8), NOT
     * dims[1]. */
    int T_DIM = 0;
    int64_t TBL_ROWS = 0;
    { const GgufTensor* t = gguf_find_tensor(&G, "adaln_t_table");
      if (!t || t->n_dims != 2 || (GgmlType)t->type != GGML_TYPE_F32) {
          fprintf(stderr, "bad/missing adaln_t_table\n"); return 1; }
      T_DIM = (int)t->dims[0];
      TBL_ROWS = (int64_t)t->dims[1]; }
    float* t_emb = malloc(sizeof(float)*T_DIM);
    {
        const GgufTensor* tbl = gguf_find_tensor(&G, "adaln_t_table");
        const float* raw = (const float*)(G.mmap_base + tbl->offset);
        double t_val = 0.0;                     /* sampler schedule start */
        double posf = t_val * ((double)TBL_ROWS - 1.0);
        int i0 = (int)posf;
        if (i0 > (int)TBL_ROWS - 2) i0 = (int)TBL_ROWS - 2;
        double frac = posf - (double)i0;
        for (int k = 0; k < T_DIM; k++) {
            float a = raw[(size_t)i0 * T_DIM + k];
            float b2 = raw[(size_t)(i0 + 1) * T_DIM + k];
            t_emb[k] = (float)((1.0 - frac) * a + frac * b2);
        }
    }

    char nm[160];
    clock_t t0 = clock();

    /* ===== run ALL 50 blocks ===== */
    for (int li = 0; li < NL; li++) {
        /* V221: per-block AdaLN modulation.
         * adaln_proj.linear is F16 [K_in=T_DIM(8), N_out=6*HID*3],
         * bias F16 [6*HID*3]. Reference: linear(t_emb) (no silu for
         * curve checkpoints) -> .view(M*modalities, expand*HID) ->
         * chunk(expand). Flat index = row*(6*HID) + c*HID + i where
         * row = m*3 + tag, video tag=0; chunk order = shift_msa,
         * scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp. */
        if (HID > 8192) { fprintf(stderr, "HID too big\n"); return 1; }
        float shift_msa[8192], scale_msa[8192], gate_msa[8192];
        float shift_mlp[8192], scale_mlp[8192], gate_mlp[8192];
        {
            char nmw[160];
            snprintf(nmw, sizeof nmw, "blocks.%d.adaln_proj.linear.weight", li);
            const GgufTensor* w = gguf_find_tensor(&G, nmw);
            snprintf(nm, sizeof nm, "blocks.%d.adaln_proj.linear.bias", li);
            const GgufTensor* bts = gguf_find_tensor(&G, nm);
            if (!w || !bts) { fprintf(stderr, "missing adaln_proj\n"); return 1; }
            const int OUT_COLS = 6 * HID;
            const uint16_t* wb = (const uint16_t*)(G.mmap_base + w->offset);
            const uint16_t* bb = (const uint16_t*)(G.mmap_base + bts->offset);
            /* weight layout: ggml dims reversed => ne[0]=T_DIM contiguous
             * per output row. Row r at base + r*T_DIM. */
            static float ap_out[6 * 3 * 8192];
            for (int cidx = 0; cidx < OUT_COLS; cidx++) {
                const uint16_t* wrow =
                    wb + (size_t)cidx * T_DIM;
            /* NOTE: type=1 is IEEE F16 — decode with q4k_fp16_to_fp32,
             * NOT bf16 bit-shift semantics; mixing poisons acc. */
                double acc = (double)q4k_fp16_to_fp32(bb[cidx]);
                for (int k = 0; k < T_DIM; k++)
                    acc += (double)q4k_fp16_to_fp32(wrow[k])
                         * (double)t_emb[k];
                ap_out[cidx] = (float)acc;      /* row 0 = video */
            }
            const float* row0 = ap_out;
            for (int i = 0; i < HID; i++) {
                shift_msa[i] = row0[0 * HID + i];
                scale_msa[i] = row0[1 * HID + i];
                gate_msa[i]  = row0[2 * HID + i];
                shift_mlp[i] = row0[3 * HID + i];
                scale_mlp[i] = row0[4 * HID + i];
                gate_mlp[i]  = row0[5 * HID + i];
            }
        }

        /* attention sub-block */
        memcpy(xres, x, xsz);
        snprintf(nm, sizeof nm, "blocks.%d.norm1.weight", li);
        rmsnorm_bf16_rows(x, (const uint16_t*)T(nm), S, HID);
        /* V221: AdaLN scale/shift on the normed stream BEFORE qkv */
        for (int i = 0; i < S*HID; i++)
            x[i] = x[i] * (1.0f + scale_msa[i % HID]) + shift_msa[i % HID];

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

        /* V221: per-head RMSNorm (q_norm/k_norm gains) applied BEFORE the
         * split-half rope, matching rms_rope_split_half. */
        /* V221: fused norm+rope per the reference implementation.
         * 1) per-head RMSNorm over full head_dim with q_norm/k_norm gains
         *    (eps = qk_norm_eps 1e-6)
         * 2) SPLIT-HALF rope on the first ROT_PAIRS pairs only:
         *    pair j couples (x[j], x[HD/2+j]); angle base =
         *    s * inv_freq[axis*16 + j%16], axis = j/16 in {t,h,w}.
         *    Dims >= ROT_PAIRS pass through unrotated. */
        snprintf(nm, sizeof nm, "blocks.%d.attn.q_norm.weight", li);
        { const float* inv_freq = (const float*)T("rope.inv_freq");
          const uint16_t* qw = (const uint16_t*)T(nm);
          snprintf(nm, sizeof nm, "blocks.%d.attn.k_norm.weight", li);
          const uint16_t* kw = (const uint16_t*)T(nm);
          for (int s = 0; s < S; s++)
            for (int hh = 0; hh < HEADS; hh++) {
                float* qp = &Qh[s*comp + hh*HD];
                float* kp = &Kh[s*comp + hh*HD];
                float ssq = 0;
                for (int d = 0; d < HD; d++) ssq += qp[d]*qp[d];
                float scq = 1.0f/sqrtf(ssq/HD + 1e-6f);
                for (int d = 0; d < HD; d++) qp[d] *= scq * bf16_to_f32(qw[d]);
                ssq = 0;
                for (int d = 0; d < HD; d++) ssq += kp[d]*kp[d];
                float sck = 1.0f/sqrtf(ssq/HD + 1e-6f);
                for (int d = 0; d < HD; d++) kp[d] *= sck * bf16_to_f32(kw[d]);

                for (int j = 0; j < ROT_PAIRS; j++) {
                    /* GGUF stores ONE inv_freq vector of 16 entries shared
                     * by all three axes (dims=[16], F32); reference builds
                     * per-axis angles as pos_axis * inv_freq[j%16]. Spike
                     * uses t=h=w=s, so angle = s * inv_freq[j%16] for
                     * every pair j. */
                    float ang = (float)s * inv_freq[j % 16];
                    float c = cosf(ang), sn = sinf(ang);
                    float a0 = qp[j], a1 = qp[HD/2 + j];
                    qp[j]        = a0*c - a1*sn;
                    qp[HD/2 + j] = a0*sn + a1*c;
                    a0 = kp[j]; a1 = kp[HD/2 + j];
                    kp[j]        = a0*c - a1*sn;
                    kp[HD/2 + j] = a0*sn + a1*c;
                }
            }
        }

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
        /* V221: residual absorbs gate_msa * attn_out */
        for (int i = 0; i < S*HID; i++)
            x[i] = xres[i] + gate_msa[i % HID] * proj[i];

        free(Qh); free(Kh); free(Vh);
        act_stats("post-attn", x, S*HID);

        /* MLP sub-block */
        memcpy(xres, x, xsz);
        snprintf(nm, sizeof nm, "blocks.%d.norm2.weight", li);
        rmsnorm_bf16_rows(x, (const uint16_t*)T(nm), S, HID);
        for (int i = 0; i < S*HID; i++)
            x[i] = x[i] * (1.0f + scale_mlp[i % HID]) + shift_mlp[i % HID];

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
        act_stats("swiglu", mlp, S*FF2);

        /* fc2: [S,FF2] → [S,HID] */
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc2.weight", li);
        { float* proj2 = malloc(sizeof(float)*S*HID);
          for (int s = 0; s < S; s++)
            q4k_gemv(T(nm), FF2, HID, &mlp[s*FF2], &proj2[s*HID]);
          /* V221: residual absorbs gate_mlp * mlp_out */
          for (int i = 0; i < S*HID; i++)
            x[i] = xres[i] + gate_mlp[i % HID] * proj2[i];
          free(proj2); }

        act_stats("post-mlp", x, S*HID);
        if (li % 10 == 0 || li == NL-1)
            fprintf(stderr, "  L%02d/%d done (%.1fs elapsed)\n",
                    li, NL-1, (double)(clock()-t0)/CLOCKS_PER_SEC);
    }

    /* final norm */
    snprintf(nm, sizeof nm, "final_layer.norm.weight");
    rmsnorm_bf16(x, (const uint16_t*)T("final_layer.norm.weight"), HID);

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
