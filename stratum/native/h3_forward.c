/*
 * h3_forward.c — H3 pipeline M3: packed forward through the denoiser.
 *
 * Assembles the full conditioning stream and runs ONE denoiser pass
 * (t=0) over all 50 blocks:
 *
 *   [text rows]     <- token_refiner(condition_proj(text_states)) from
 *                      a Q3TE0001 dump (te_qwen3vl_step output)
 *   [audio rows]    <- synthetic stereo latent -> audio_patch_proj
 *   [video rows]    <- synthetic video latent -> patchify -> video_
 *                      patch_proj, rows in (t, h, w) order
 *
 * Positions come from the h3_pack semantics (verified separately):
 * text t=0..L-1; audio (t, 0, w_low|w_high) per channel; video rows on
 * the FRAME_RESCALE cumsum grid with area-normalized (h, w).
 *
 * Per-block AdaLN rows by modality tag: video=0, text=1, audio=2 —
 * adaln_proj output is [3][6*HID] at M=1 (single timestep t=0).
 *
 * Denoiser op order per block (validated in h3_step.c): norm1 -> AdaLN
 * scale/shift -> qkv -> per-head QK-norm -> split-half rope over pairs
 * 0..47 with angle = pos[axis(j)] * inv_freq[j%16] -> bidirectional
 * attention -> gate_msa residual; norm2 -> AdaLN -> swiglu MLP ->
 * gate_mlp residual. RMSNorm gains and condition/token_refiner norms
 * are BF16; patch/proj heads are F32; adaln_proj is F16.
 *
 * Output: final hidden stream stats + "H3PKT001" dump of the video
 * segment (what a sampler would iterate on).
 */
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include "stratum_q6k.h"
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LATENTS_DIM 24
#define AUDIO_DIM 32
#define VIDEO_ROW_DIM (LATENTS_DIM * 4)
#define AUDIO_ROW_DIM AUDIO_DIM
#define FRAME_RESCALE (5.0 / 3.0)
static const int FRAME_PER_TOKEN[5] = {1, 4, 4, 4, 4};
#define ROT_PAIRS 48
#define MAX_HID 8192

static Gguf G;

static const void* T(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "missing '%s'\n", name); exit(1); }
    return (const void*)(G.mmap_base + t->offset);
}
static const GgufTensor* TT(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "missing '%s'\n", name); exit(1); }
    return t;
}

static inline float f16v(uint16_t h) { return q4k_fp16_to_fp32(h); }
static inline float bfv(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

static void act_stats(const char* tag, const float* x, long n) {
    float mx = 0; double sm = 0; long nan = 0;
    for (long i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (isnan(x[i]) || isinf(x[i])) { nan++; continue; }
        if (a > mx) mx = a;
        sm += a;
    }
    fprintf(stderr, "    %-12s |x|mean=%.6g max=%.6g%s\n", tag,
            nan ? 0.0 : sm / n, mx,
            nan ? (nan == n ? " ALL-NAN/INF" : " (nan contaminated)") : "");
}

/* mixed-quant gemv (Q4_K/Q6_K), rows [out][K/256] */
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
        } else { fprintf(stderr, "bad quant %u\n", t->type); exit(1); }
        y[r] = (float)acc;
    }
}

/* F32 gemv [in, out] row-major (ne[0]=in contiguous, rows = out) */
static void f32_gemv(const void* w, int in_dim, int out_dim,
                     const float* x, float* y) {
    const float* wr = (const float*)w;
    for (int r = 0; r < out_dim; r++) {
        const float* row = wr + (size_t)r * in_dim;
        double acc = 0;
        for (int c = 0; c < in_dim; c++) acc += (double)row[c] * x[c];
        y[r] = (float)acc;
    }
}

static double g_sigma_v = 0.0;   /* M4 driver sets these per step */
static double g_sigma_a = 0.0;

static double env_sigma_default(void) {
    const char* e = getenv("H3_SIGMA_V");
    double sv = e ? atof(e) : 0.0;
    if (sv < 1e-6) sv = 1e-6;
    if (sv > 1.0) sv = 1.0;
    return sv;
}

static double shift_map(double s, double f, double g) {
    double base = s / (f + s * (1.0 - f));
    return g * base / (1.0 + (g - 1.0) * base);
}

static double axis_val(int dim, int patch, int idx, double sqrt_area) {
    int n = dim / patch;
    double ratio = (double)dim / sqrt_area;
    return ((double)idx * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
}

static double video_t_at(int k, double origin) {
    double acc = 0;
    for (int i = 0; i < k; i++)
        acc += FRAME_RESCALE * FRAME_PER_TOKEN[i % 5];
    return origin + acc;
}

static float (*g_adaln)[3][6][MAX_HID] = NULL;
static float g_t_emb_v[16], g_t_emb_a[16];
static long g_seq_len; static int g_tag_sel;

int run_h3_forward_main(int argc, char** argv);   /* old main */

/* M4 driver: Euler loop lives in h3_sample_main below; the packed
 * forward itself is parameterized by sigma through g_t_emb_*. */
int main(int argc, char** argv) { return run_h3_forward_main(argc, argv); }

int run_h3_forward_main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <denoiser.gguf> <text_states.bin> "
                        "[vt=1 lat_h=32 lat_w=32 audio_t=4]\n", argv[0]);
        return 1;
    }
    int vt = argc > 3 ? atoi(argv[3]) : 1;
    int lat_h = argc > 4 ? atoi(argv[4]) : 32;
    int lat_w = argc > 5 ? atoi(argv[5]) : 32;
    int audio_t = argc > 6 ? atoi(argv[6]) : 4;

    if (gguf_open(argv[1], &G) != 0) return 1;

    int HID = 0;
    { const GgufTensor* t = TT("final_layer.norm.weight"); HID = (int)t->dims[0]; }
    int HD = 0, HEADS, QKV, FF1, FF2, NL = 0;
    { const GgufTensor* t = TT("blocks.0.attn.q_norm.weight"); HD = (int)t->dims[0]; }
    { const GgufTensor* t = TT("blocks.0.attn.qkv_proj.weight");
      QKV = (int)t->dims[1]; HEADS = QKV / 3 / HD; }
    { const GgufTensor* t = TT("blocks.0.mlp.fc1.weight"); FF1 = (int)t->dims[1]; }
    { const GgufTensor* t = TT("blocks.0.mlp.fc2.weight"); FF2 = (int)t->dims[0]; }
    for (uint64_t i = 0; i < G.n_tensors; i++) {
        if (!strncmp(G.tensors[i].name, "blocks.", 7)) {
            int bi = atoi(G.tensors[i].name + 7);
            if (bi + 1 > NL) NL = bi + 1;
        }
    }
    int comp = QKV / 3;

    /* --- text_states from dump --- */
    long s_text = 0; int te_hid = 0;
    float* text_states = NULL;
    {
        FILE* f = fopen(argv[2], "rb");
        char magic[8]; uint32_t meta[3];
        if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "Q3TE0001", 8)) {
            fprintf(stderr, "bad text_states dump\n"); return 1;
        }
        if (fread(meta, 4, 3, f) != 3) return 1;
        s_text = meta[0]; te_hid = (int)meta[1];
        if (te_hid != 5120) { fprintf(stderr, "text_states hid %d?\n", te_hid); return 1; }
        text_states = malloc(sizeof(float) * s_text * te_hid);
        if (fread(text_states, 4, s_text * te_hid, f) != (size_t)(s_text * te_hid))
            return 1;
        fclose(f);
    }

    /* --- text conditioning: condition_proj (bf16, bias) + token_refiner --- */
    float* trow = malloc(sizeof(float) * MAX_HID);
    float* text_cond = malloc(sizeof(float) * s_text * HID);
    {
        const GgufTensor* w = TT("condition_proj.weight");
        const GgufTensor* b = TT("condition_proj.bias");
        const uint16_t* wr = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* br = (const uint16_t*)(G.mmap_base + b->offset);
        for (long s = 0; s < s_text; s++) {
            for (int r = 0; r < HID; r++) {
                const uint16_t* row = wr + (size_t)r * te_hid;
                double acc = (double)bfv(br[r]);
                for (int c = 0; c < te_hid; c++)
                    acc += (double)bfv(row[c]) * (double)text_states[s * te_hid + c];
                text_cond[s * HID + r] = (float)acc;
            }
        }
    }
    act_stats("cond_proj", text_cond, s_text * HID);

    /* token_refiner: 2 blocks, no adaln, rope identity? Reference
     * RefinerBlock calls self.attn(...) WITHOUT rope_freqs => the qk-norm
     * else-branch applies ONLY q_norm/k_norm, no rope. Bidirectional. */
    for (int li = 0; li < 2; li++) {
        char nm[160];
        float* xr = malloc(sizeof(float) * s_text * HID);
        memcpy(xr, text_cond, sizeof(float) * s_text * HID);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.norm1.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < s_text; s++) {
                float* r = &text_cond[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-6));
                for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
            }
        }
        float* qkv = malloc(sizeof(float) * s_text * QKV);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.qkv_proj.weight", li);
        for (long s = 0; s < s_text; s++)
            mixed_gemv(TT(nm), HID, QKV, &text_cond[s * HID], &qkv[s * QKV]);
        /* qk-norm only (no rope in refiner) */
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.q_norm.weight", li);
        const uint16_t* qw = (const uint16_t*)T(nm);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.k_norm.weight", li);
        const uint16_t* kw = (const uint16_t*)T(nm);
        for (long s = 0; s < s_text; s++) {
            for (int h = 0; h < HEADS; h++) {
                float* qp = &qkv[s * QKV + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)qp[d] * qp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-6));
                for (int d = 0; d < HD; d++) qp[d] *= sc * bfv(qw[d]);
            }
            for (int h = 0; h < HEADS; h++) {
                float* kp = &qkv[s * QKV + comp + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)kp[d] * kp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-6));
                for (int d = 0; d < HD; d++) kp[d] *= sc * bfv(kw[d]);
            }
        }
        /* bidirectional attention */
        float* attn = calloc((size_t)s_text * comp, sizeof(float));
        float scale = 1.0f / sqrtf((float)HD);
        for (int h = 0; h < HEADS; h++)
            for (int a = 0; a < s_text; a++) {
                const float* qh = &qkv[a * QKV + h * HD];
                float lg[4096];
                for (int b2 = 0; b2 < s_text; b2++) {
                    const float* kh = &qkv[b2 * QKV + comp + h * HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++) dot += (double)qh[d] * kh[d];
                    lg[b2] = (float)(dot * scale);
                }
                float mx = lg[0];
                for (int j = 1; j < s_text; j++) if (lg[j] > mx) mx = lg[j];
                double se = 0;
                for (int j = 0; j < s_text; j++) { lg[j] -= mx; se += exp((double)lg[j]); }
                float inv = (float)(1.0 / se);
                float* oh = &attn[a * comp + h * HD];
                for (int b2 = 0; b2 < s_text; b2++) {
                    float pv = expf(lg[b2]) * inv;
                    const float* vh = &qkv[b2 * QKV + 2 * comp + h * HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv * vh[d];
                }
            }
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.attn.out_proj.weight", li);
        for (long s = 0; s < s_text; s++)
            mixed_gemv(TT(nm), comp, HID, &attn[s * comp], &trow[s * HID]);
        for (long i = 0; i < s_text * HID; i++) text_cond[i] = xr[i] + trow[i];
        free(qkv); free(attn);

        /* mlp */
        memcpy(xr, text_cond, sizeof(float) * s_text * HID);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.norm2.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < s_text; s++) {
                float* r = &text_cond[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-6));
                for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
            }
        }
        float* f1 = malloc(sizeof(float) * s_text * FF1);
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.mlp.fc1.weight", li);
        for (long s = 0; s < s_text; s++)
            mixed_gemv(TT(nm), HID, FF1, &text_cond[s * HID], &f1[s * FF1]);
        float* fa = malloc(sizeof(float) * s_text * FF2);
        for (long s = 0; s < s_text; s++)
            for (int i = 0; i < FF2; i++) {
                float gv = f1[s * FF1 + i];
                fa[s * FF2 + i] = (gv / (1.0f + expf(-gv))) * f1[s * FF1 + FF2 + i];
            }
        snprintf(nm, sizeof nm, "token_refiner.blocks.%d.mlp.fc2.weight", li);
        for (long s = 0; s < s_text; s++)
            mixed_gemv(TT(nm), FF2, HID, &fa[s * FF2], &trow[s * HID]);
        for (long i = 0; i < s_text * HID; i++) text_cond[i] = xr[i] + trow[i];
        free(xr); free(f1); free(fa);
    }
    {   /* final_norm of the refiner */
        const uint16_t* g16 = (const uint16_t*)T("token_refiner.final_norm.weight");
        for (long s = 0; s < s_text; s++) {
            float* r = &text_cond[s * HID];
            double ss = 0;
            for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
            float sc = (float)(1.0 / sqrt(ss / HID + 1e-6));
            for (int i = 0; i < HID; i++) r[i] *= sc * bfv(g16[i]);
        }
    }
    act_stats("refined-text", text_cond, s_text * HID);

    /* --- audio/video embeds (synthetic latents) --- */
    int nh = lat_h / 2, nw = lat_w / 2;
    int frame_rows = nh * nw;
    int n_audio = audio_t * 2;
    int n_video = vt * frame_rows;
    long seq_len = s_text + n_audio + n_video;
    fprintf(stderr, "packed: text=%ld audio=%d video=%d seq=%ld\n",
            s_text, n_audio, n_video, seq_len);

    float* stream = malloc(sizeof(float) * seq_len * HID);
    double* pos = calloc((size_t)seq_len * 3, sizeof(double));
    int* tag = malloc(sizeof(int) * seq_len);
    memcpy(stream, text_cond, sizeof(float) * s_text * HID);
    for (long s = 0; s < s_text; s++) {
        pos[s * 3 + 0] = (double)s;
        tag[s] = 1;
    }
    /* audio rows */
    {
        float* arow = malloc(sizeof(float) * AUDIO_ROW_DIM);
        const GgufTensor* w = TT("audio_patch_proj.weight");
        const GgufTensor* b = TT("audio_patch_proj.bias");
        double w_low = axis_val(lat_w, 2, 0, sqrt((double)lat_h * lat_w));
        double w_high = axis_val(lat_w, 2, nw - 1, sqrt((double)lat_h * lat_w));
        long r = s_text;
        for (int t = 0; t < audio_t; t++)
            for (int c = 0; c < 2; c++) {
                for (int i = 0; i < AUDIO_ROW_DIM; i++)
                    arow[i] = (float)(((i * 7 + t * 3 + c) % 29) * 0.05 - 0.7);
                f32_gemv(T("audio_patch_proj.weight"), AUDIO_ROW_DIM, HID,
                         arow, &stream[r * HID]);
                for (int i = 0; i < HID; i++)
                    stream[r * HID + i] += ((const float*)T("audio_patch_proj.bias"))[i];
                pos[r * 3 + 0] = (double)s_text + t;
                pos[r * 3 + 2] = c == 0 ? w_low : w_high;
                tag[r] = 2;
                r++;
            }
        free(arow);
    }
    /* video rows: patchify synthetic [24][vt][lat_h][lat_w] */
    {
        size_t ln = (size_t)LATENTS_DIM * vt * lat_h * lat_w;
        float* latent = malloc(sizeof(float) * ln);
        const char* xin = getenv("H3_X_IN");
        if (xin) {
            FILE* xf = fopen(xin, "rb");
            if (!xf || fread(latent, sizeof(float), ln, xf) != ln) {
                fprintf(stderr, "H3_X_IN unreadable\n"); return 1;
            }
            fclose(xf);
        } else {
            for (size_t i = 0; i < ln; i++)
                latent[i] = (float)(((i * 13) % 41) * 0.06 - 1.2);
        }
        float* prow = malloc(sizeof(float) * VIDEO_ROW_DIM);
        long r = s_text + n_audio;
        double sqrt_area = sqrt((double)lat_h * lat_w);
        for (int t = 0; t < vt; t++)
            for (int hh = 0; hh < nh; hh++)
                for (int ww = 0; ww < nw; ww++) {
                    int p = 0;
                    for (int c = 0; c < LATENTS_DIM; c++)
                        for (int b = 0; b < 2; b++)
                            for (int a = 0; a < 2; a++)
                                prow[p++] = latent[
                                    (((size_t)c * vt + t) * lat_h + (2 * hh + b))
                                    * lat_w + (2 * ww + a)];
                    f32_gemv(T("video_patch_proj.weight"), VIDEO_ROW_DIM, HID,
                             prow, &stream[r * HID]);
                    for (int i = 0; i < HID; i++)
                        stream[r * HID + i] +=
                            ((const float*)T("video_patch_proj.bias"))[i];
                    pos[r * 3 + 0] = video_t_at(t, (double)s_text);
                    pos[r * 3 + 1] = axis_val(lat_h, 2, hh, sqrt_area);
                    pos[r * 3 + 2] = axis_val(lat_w, 2, ww, sqrt_area);
                    tag[r] = 0;
                    r++;
                }
        free(latent); free(prow);
    }
    act_stats("packed-stream", stream, seq_len * HID);

    /* --- AdaLN at t=0: rows [3][6*HID], per-tag --- */
    int T_DIM = 0; long TBL_ROWS = 0;
    {
        const GgufTensor* t = TT("adaln_t_table");
        T_DIM = (int)t->dims[0]; TBL_ROWS = (long)t->dims[1];
    }
    /* M4: two stream timesteps. t_v = 1 - sigma_v (video/text);
     * t_a = 1 - shift_map(sigma_v, 12 -> 3) (audio). Table row =
     * fractional index of t over [0,1] grid (rows-1). */
    float t_emb[16];
    {
        double sigma_v = g_sigma_v, sigma_a = g_sigma_a;
        if (sigma_v <= 0.0) {
            sigma_v = env_sigma_default();
            sigma_a = shift_map(sigma_v, 12.0, 3.0);
        }
        double t_v = 1.0 - sigma_v, t_a = 1.0 - sigma_a;
        const float* raw = (const float*)T("adaln_t_table");
        for (int k = 0; k < T_DIM; k++) {
            double pv = t_v * ((double)TBL_ROWS - 1.0);
            int p0 = (int)pv; if (p0 > (int)TBL_ROWS - 2) p0 = (int)TBL_ROWS - 2;
            double pf = pv - p0;
            g_t_emb_v[k] = (float)((1.0 - pf) * raw[(size_t)p0 * T_DIM + k]
                                   + pf * raw[(size_t)(p0 + 1) * T_DIM + k]);
            double pa = t_a * ((double)TBL_ROWS - 1.0);
            int a0 = (int)pa; if (a0 > (int)TBL_ROWS - 2) a0 = (int)TBL_ROWS - 2;
            double af = pa - a0;
            g_t_emb_a[k] = (float)((1.0 - af) * raw[(size_t)a0 * T_DIM + k]
                                   + af * raw[(size_t)(a0 + 1) * T_DIM + k]);
        }
        memcpy(t_emb, g_t_emb_v, sizeof(float) * T_DIM);
    }
    float (*adaln)[3][6][MAX_HID] =
        malloc(sizeof(float[3][6][MAX_HID]) * NL);
    g_adaln = adaln;
    for (int li = 0; li < NL; li++) {
        char nm[160];
        snprintf(nm, sizeof nm, "blocks.%d.adaln_proj.linear.weight", li);
        const GgufTensor* w = TT(nm);
        snprintf(nm, sizeof nm, "blocks.%d.adaln_proj.linear.bias", li);
        const GgufTensor* bts = TT(nm);
        const uint16_t* wb = (const uint16_t*)(G.mmap_base + w->offset);
        const uint16_t* bb = (const uint16_t*)(G.mmap_base + bts->offset);
        for (int row = 0; row < 3; row++) {
            (void)row;
            const float* emb_row = row == 2 ? g_t_emb_a : g_t_emb_v;
            for (int cidx = 0; cidx < 6 * HID; cidx++) {
                const uint16_t* wrow = wb + (size_t)cidx * T_DIM;
                double acc = (double)f16v(bb[cidx]);
                for (int k = 0; k < T_DIM; k++)
                    acc += (double)f16v(wrow[k]) * (double)emb_row[k];
                int chunk = cidx / HID, i = cidx % HID;
                adaln[li][row][chunk][i] = (float)acc;
            }
        }
    }

    /* --- denoiser blocks --- */
    float* xres = malloc(sizeof(float) * seq_len * HID);
    float* qkv = malloc(sizeof(float) * seq_len * QKV);
    float* attn = malloc(sizeof(float) * seq_len * comp);
    float* fc1o = malloc(sizeof(float) * seq_len * FF1);
    float* proj = malloc(sizeof(float) * seq_len * HID);
    const float* inv_freq = (const float*)T("rope.inv_freq");
    clock_t t0 = clock();

    for (int li = 0; li < NL; li++) {
        char nm[160];
        float* shift_msa = adaln[li][0][0];
        float* scale_msa = adaln[li][0][1];
        float* gate_msa  = adaln[li][0][2];
        float* shift_mlp = adaln[li][0][3];
        float* scale_mlp = adaln[li][0][4];
        float* gate_mlp  = adaln[li][0][5];
        float* shift_msa_t = adaln[li][1][0];
        float* scale_msa_t = adaln[li][1][1];
        float* gate_msa_t  = adaln[li][1][2];
        float* shift_mlp_t = adaln[li][1][3];
        float* scale_mlp_t = adaln[li][1][4];
        float* gate_mlp_t  = adaln[li][1][5];
        float* shift_msa_a = adaln[li][2][0];
        float* scale_msa_a = adaln[li][2][1];
        float* gate_msa_a  = adaln[li][2][2];
        float* shift_mlp_a = adaln[li][2][3];
        float* scale_mlp_a = adaln[li][2][4];
        float* gate_mlp_a  = adaln[li][2][5];

        memcpy(xres, stream, sizeof(float) * seq_len * HID);
        snprintf(nm, sizeof nm, "blocks.%d.norm1.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < seq_len; s++) {
                float* r = &stream[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++)
                    r[i] = r[i] * sc * bfv(g16[i])
                         * (1.0f + (tag[s] == 0 ? scale_msa[i]
                                  : tag[s] == 1 ? scale_msa_t[i]
                                  : scale_msa_a[i]))
                         + (tag[s] == 0 ? shift_msa[i]
                            : tag[s] == 1 ? shift_msa_t[i] : shift_msa_a[i]);
            }
        }
        snprintf(nm, sizeof nm, "blocks.%d.attn.qkv_proj.weight", li);
        for (long s = 0; s < seq_len; s++)
            mixed_gemv(TT(nm), HID, QKV, &stream[s * HID], &qkv[s * QKV]);

        /* fused qk-norm + split-half rope, per-tag AdaLN'd rows already in */
        snprintf(nm, sizeof nm, "blocks.%d.attn.q_norm.weight", li);
        const uint16_t* qw = (const uint16_t*)T(nm);
        snprintf(nm, sizeof nm, "blocks.%d.attn.k_norm.weight", li);
        const uint16_t* kw = (const uint16_t*)T(nm);
        for (long s = 0; s < seq_len; s++) {
            for (int h = 0; h < HEADS; h++) {
                float* qp = &qkv[s * QKV + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)qp[d] * qp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-6));
                for (int d = 0; d < HD; d++) qp[d] *= sc * bfv(qw[d]);
            }
            for (int h = 0; h < HEADS; h++) {
                float* kp = &qkv[s * QKV + comp + h * HD];
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)kp[d] * kp[d];
                float sc = (float)(1.0 / sqrt(ss / HD + 1e-6));
                for (int d = 0; d < HD; d++) kp[d] *= sc * bfv(kw[d]);
            }
            for (int j = 0; j < ROT_PAIRS; j++) {
                int axis = j / 16, kbase = j % 16;
                float ang = (float)pos[s * 3 + axis] * inv_freq[kbase];
                float c = cosf(ang), sn = sinf(ang);
                for (int h = 0; h < HEADS; h++) {
                    float* qp = &qkv[s * QKV + h * HD];
                    float a0 = qp[j], a1 = qp[HD / 2 + j];
                    qp[j]        = a0 * c - a1 * sn;
                    qp[HD / 2 + j] = a0 * sn + a1 * c;
                    float* kp = &qkv[s * QKV + comp + h * HD];
                    a0 = kp[j]; a1 = kp[HD / 2 + j];
                    kp[j]        = a0 * c - a1 * sn;
                    kp[HD / 2 + j] = a0 * sn + a1 * c;
                }
            }
        }

        /* bidirectional attention (full packed stream) */
        float scale2 = 1.0f / sqrtf((float)HD);
        memset(attn, 0, sizeof(float) * seq_len * comp);
        float* lgd = malloc(sizeof(float) * seq_len);
        for (int h = 0; h < HEADS; h++)
            for (long a = 0; a < seq_len; a++) {
                const float* qh = &qkv[a * QKV + h * HD];
                float lg[4096];
                (void)lg;
                for (long b2 = 0; b2 < seq_len; b2++) {
                    const float* kh = &qkv[b2 * QKV + comp + h * HD];
                    double dot = 0;
                    for (int d = 0; d < HD; d++)
                        dot += (double)qh[d] * kh[d];
                    lgd[b2] = (float)(dot * scale2);
                }
                float mx = lgd[0];
                for (long j = 1; j < seq_len; j++) if (lgd[j] > mx) mx = lgd[j];
                double se = 0;
                for (long j = 0; j < seq_len; j++) {
                    lgd[j] -= mx; se += exp((double)lgd[j]);
                }
                float inv = (float)(1.0 / se);
                float* oh = &attn[a * comp + h * HD];
                for (long b2 = 0; b2 < seq_len; b2++) {
                    float pv = expf(lgd[b2]) * inv;
                    const float* vh = &qkv[b2 * QKV + 2 * comp + h * HD];
                    for (int d = 0; d < HD; d++) oh[d] += pv * vh[d];
                }
            }

        snprintf(nm, sizeof nm, "blocks.%d.attn.out_proj.weight", li);
        for (long s = 0; s < seq_len; s++)
            mixed_gemv(TT(nm), comp, HID, &attn[s * comp], &proj[s * HID]);
        for (long s = 0; s < seq_len; s++) {
            const float* g = tag[s] == 0 ? gate_msa
                           : tag[s] == 1 ? gate_msa_t : gate_msa_a;
            for (int i = 0; i < HID; i++)
                stream[s * HID + i] = xres[s * HID + i] + g[i] * proj[s * HID + i];
        }

        /* mlp */
        memcpy(xres, stream, sizeof(float) * seq_len * HID);
        snprintf(nm, sizeof nm, "blocks.%d.norm2.weight", li);
        {
            const uint16_t* g16 = (const uint16_t*)T(nm);
            for (long s = 0; s < seq_len; s++) {
                float* r = &stream[s * HID];
                double ss = 0;
                for (int i = 0; i < HID; i++) ss += (double)r[i] * r[i];
                float sc = (float)(1.0 / sqrt(ss / HID + 1e-5));
                for (int i = 0; i < HID; i++)
                    r[i] = r[i] * sc * bfv(g16[i])
                         * (1.0f + (tag[s] == 0 ? scale_mlp[i]
                                  : tag[s] == 1 ? scale_mlp_t[i]
                                  : scale_mlp_a[i]))
                         + (tag[s] == 0 ? shift_mlp[i]
                            : tag[s] == 1 ? shift_mlp_t[i] : shift_mlp_a[i]);
            }
        }
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc1.weight", li);
        for (long s = 0; s < seq_len; s++)
            mixed_gemv(TT(nm), HID, FF1, &stream[s * HID], &fc1o[s * FF1]);
        for (long s = 0; s < seq_len; s++)
            for (int i = 0; i < FF2; i++) {
                float gv = fc1o[s * FF1 + i];
                fc1o[s * FF1 + i] =
                    (gv / (1.0f + expf(-gv))) * fc1o[s * FF1 + FF2 + i];
            }
        snprintf(nm, sizeof nm, "blocks.%d.mlp.fc2.weight", li);
        for (long s = 0; s < seq_len; s++)
            mixed_gemv(TT(nm), FF2, HID, &fc1o[s * FF2], &proj[s * HID]);
        for (long s = 0; s < seq_len; s++) {
            const float* g = tag[s] == 0 ? gate_mlp
                           : tag[s] == 1 ? gate_mlp_t : gate_mlp_a;
            for (int i = 0; i < HID; i++)
                stream[s * HID + i] =
                    xres[s * HID + i] + g[i] * proj[s * HID + i];
        }

        free(lgd);
        if (li % 10 == 0 || li == NL - 1) {
            fprintf(stderr, "  L%02d\n", li);
            act_stats("post-mlp", stream, seq_len * HID);
        }
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "\n  packed forward (%d layers, seq=%ld): %.1fs\n",
            NL, seq_len, elapsed);

    /* dump video segment (sampler input) */
    FILE* out = fopen("/tmp/h3_packed_out.bin", "wb");
    fwrite("H3PKT001", 1, 8, out);
    uint32_t meta[4] = { (uint32_t)n_video, (uint32_t)HID,
                         (uint32_t)s_text, (uint32_t)n_audio };
    fwrite(meta, 4, 4, out);
    fwrite(stream + (size_t)(s_text + n_audio) * HID, sizeof(float),
           (size_t)n_video * HID, out);
    fclose(out);
    if (getenv("H3_X_OUT")) {
        FILE* xo = fopen(getenv("H3_X_OUT"), "wb");
        fwrite(stream + (size_t)(s_text + n_audio) * HID, sizeof(float),
               (size_t)n_video * HID, xo);
        fclose(xo);
    }
    printf("video rows: %d x %d -> /tmp/h3_packed_out.bin\n", n_video, HID);
    act_stats("final-video", stream + (size_t)(s_text + n_audio) * HID,
              (long)n_video * HID);

    return 0;
}
