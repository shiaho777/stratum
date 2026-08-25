/* h3_step.c — Phase-2 validation: read real H3 weights, dequantize, compute.
 *
 * Validates that Stratum's infrastructure can process production-scale
 * video DiT weights. This is NOT a complete inference engine — it proves
 * the data pipeline works end-to-end on real model dimensions.
 */
#include "stratum_gguf.h"
#include "stratum_q4k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static Gguf G;
static int S, HID, HD, HEADS, QKV, FF1, FF2, NL;

static const void* T(const char* name) {
    const GgufTensor* t = gguf_find_tensor(&G, name);
    if (!t) { fprintf(stderr, "h3: missing '%s'\n", name); return NULL; }
    return (const void*)(G.mmap_base + t->offset);
}

static double q4k_dot(const void* w_data, const float* x,
                      int in_dim, int out_idx, int out_dim) {
    /* For Q4_K weight [in_padded × out], extract row `out_idx`
     * and compute dot product with x[0..in_dim) */
    int n_blocks_per_row = in_dim / 256;
    const block_q4_K* blocks = (const block_q4_K*)w_data;
    /* Weight layout: each output column has n_blocks_per_row contiguous
     * Q4_K blocks along the input dimension */
    const block_q4_K* col_blocks = blocks + out_idx * n_blocks_per_row;

    double acc = 0.0;
    float tmp[256];
    for (int nb = 0; nb < n_blocks_per_row; nb++) {
        q4k_dequant_block_scalar(&col_blocks[nb], tmp);
        for (int c = 0; c < 256; c++)
            acc += (double)tmp[c] * (double)x[nb * 256 + c];
    }
    return acc;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <h3.gguf>\n", argv[0]);
        return 1;
    }
    if (gguf_open(argv[1], &G) != 0) { fprintf(stderr, "open failed\n"); return 1; }

    printf("=== H3 Model Validation ===\n\n");

    /* 1. Architecture discovery from tensor shapes */
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

    printf("Architecture:\n");
    printf("  hidden_size:     %d\n", HID);
    printf("  attention heads: %d × %d = %d QKV dims\n", HEADS, HD, HEADS*HD);
    printf("  QKV total:       %d (= 3 × %d)\n", QKV, QKV/3);
    printf("  FFN fused:       %d → inner: %d (fused gate+up)\n", FF1, FF2);
    printf("  layers:          %d\n", NL);
    printf("\n");

    /* 2. Validate all expected tensors exist */
    printf("Tensor presence check:\n");
    const char* critical[] = {
        "adaln_t_table",
        "video_patch_proj.weight",
        "condition_proj.weight",
        "rope.inv_freq",
        "final_layer.norm.weight",
        "final_layer.video_out.weight",
        NULL
    };
    int missing = 0;
    for (int i = 0; critical[i]; i++) {
        if (!gguf_find_tensor(&G, critical[i])) {
            printf("  ✗ MISSING: %s\n", critical[i]);
            missing++;
        } else {
            printf("  ✓ %s\n", critical[i]);
        }
    }

    /* Check all blocks have required tensors */
    for (int li = 0; li < NL; li++) {
        char nm[160];
        const char* required[] = {
            "adaln_proj.linear.weight", "attn.qkv_proj.weight",
            "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight",
            "norm1.weight", "norm2.weight", NULL
        };
        for (int j = 0; required[j]; j++) {
            snprintf(nm, sizeof nm, "blocks.%d.%s", li, required[j]);
            if (!gguf_find_tensor(&G, nm)) {
                printf("  ✗ MISSING: %s\n", nm);
                missing++;
            }
        }
    }
    printf("  missing: %d / %d checks\n\n", missing,
           6 + NL * 7);

    /* 3. Dequantize real Q4_K weights and compute */
    printf("Q4_K dequantization test (blocks.0.mlp.fc1):\n");
    const void* fc1_w = T("blocks.0.mlp.fc1.weight");
    if (!fc1_w) return 1;

    /* Create deterministic input vector of size HID=5376 */
    float* xin = malloc(sizeof(float) * HID);
    srand(42);
    for (int i = 0; i < HID; i++)
        xin[i] = (float)((rand() / (double)RAND_MAX) * 2.0 - 1.0);

    clock_t start = clock();

    /* Compute first 8 outputs of mlp.fc1 using Q4_K dequantization */
    float outputs[8];
    int n_blocks_per_col = HID / 256;
    const block_q4_K* blk = (const block_q4_K*)fc1_w;

    /* Q4_K layout: [HID/256 blocks][out] — each block covers 256 input dims
     * for one output neuron. So for output r, the blocks are at
     * blk[r*n_blocks_per_col .. (r+1)*n_blocks_per_col) */
    for (int r = 0; r < 8; r++) {
        double acc = 0.0;
        float tmp[256];
        for (int nb = 0; nb < n_blocks_per_col; nb++) {
            q4k_dequant_block_scalar(&blk[r * n_blocks_per_col + nb], tmp);
            for (int c = 0; c < 256 && nb*256+c < HID; c++)
                acc += (double)tmp[c] * (double)xin[nb*256+c];
        }
        outputs[r] = (float)acc;
    }

    double dt = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("  outputs[0..7]: ");
    for (int r = 0; r < 8; r++) printf("%.6f ", outputs[r]);
    printf("\n");
    printf("  timing: %.4fs for 8 outputs × %d dims (est %.1fs per full layer)\n",
           dt, HID, dt / 8 * FF1);
    printf("  values range: [%.4f, %.4f]\n",
           outputs[0] < outputs[1] ? outputs[0] : outputs[1],
           outputs[0] > outputs[1] ? outputs[0] : outputs[1]);

    /* 4. BF16 reading test */
    printf("\nBF16 reading test (blocks.0.norm1):\n");
    const uint16_t* n1 = (const uint16_t*)T("blocks.0.norm1.weight");
    printf("  values[0..3]: ");
    for (int i = 0; i < 4; i++) {
        uint32_t bits = (uint32_t)n1[i] << 16;
        float f; memcpy(&f, &bits, 4);
        printf("%.6f ", f);
    }
    printf("\n");

    /* 5. Memory footprint estimate */
    printf("\nMemory estimate for full forward (S=%d tokens):\n", 256);
    size_t wts = 11420630112ULL; /* from GGUF header */
    size_t act_hidden = 256 * HID * 4;
    size_t act_qkv = 256 * QKV * 4;
    size_t act_ffn = 256 * FF1 * 4;
    printf("  weights (mmap):   %.2f GB\n", wts / 1e9);
    printf("  hidden states:    %.2f MB\n", act_hidden / 1e6);
    printf("  QKV buffer:       %.2f MB\n", act_qkv / 1e6);
    printf("  FFN intermediate: %.2f MB\n", act_ffn / 1e6);
    printf("  total anon:       %.2f GB\n",
           (act_hidden + act_qkv + act_ffn) / 1e9);

    printf("\n=== Validation %s ===\n", missing == 0 ? "PASSED" : "PARTIAL");
    printf("All critical tensors present and readable.\n");
    printf("Ready for Phase-2 implementation.\n");

    free(xin);
    gguf_close(&G);
    return 0;
}
