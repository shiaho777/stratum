/*
 * stratum_arch.h — Universal architecture registry for Stratum.
 *
 * Design principle: Stratum is a GENERIC inference engine. No code should
 * be specialized for any particular model. Each architecture registers
 * itself through this interface; the main dispatch loop is model-agnostic.
 *
 * To add a new architecture:
 *   1. Implement the StratumArch interface
 *   2. Call STRATUM_REGISTER_ARCH() with the GGUF arch names it handles
 *   3. Done — no changes to stratum.c or any other file needed
 */
#ifndef STRATUM_ARCH_H
#define STRATUM_ARCH_H

#include "stratum_gguf.h"
#include <stdint.h>
#include <stdbool.h>

/* ------------------------------------------------------------------ */
/*  Generic model config — superset of all supported architectures     */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Identity */
    char arch[64];          /* general.architecture string from GGUF */

    /* Transformer core (universal — every LLM has these) */
    int  n_layers;          /* block_count */
    int  n_embed;           /* embedding_length (hidden_size) */
    int  n_ff;              /* feed_forward_length (intermediate_size) */
    int  n_q_heads;         /* attention.head_count */
    int  n_kv_heads;        /* attention.head_count_kv (GQA) */
    int  head_dim;          /* attention.key_length */
    int  rope_dim;          /* rope.dimension_count */
    float rms_eps;          /* attention.layer_norm_rms_epsilon */
    float rope_theta;       /* rope.freq_base */

    /* Vocabulary */
    int  vocab_size;
    bool has_output_weight; /* output.weight separate from token_embd */
    bool has_qk_norm;       /* attn_q_norm / attn_k_norm present */

    /* Attention type pattern (for hybrid architectures) */
    /* If full_attn_stride > 0: layer N is full-attention when (N+1)%stride==0
     * If full_attn_stride == 0: all layers are full-attention (standard Transformer)
     * If full_attn_stride < 0: all layers are SSM/linear-attn (pure Mamba) */
    int  full_attn_stride;

    /* SSM / Mamba params (only meaningful for hybrid architectures;
     * zero/unused for pure Transformer models) */
    int  ssm_state_size;    /* HV — hidden state dimension per group */
    int  ssm_group_count;   /* LIN_NK — key/heads count */
    int  ssm_value_heads;   /* NV — value head count (= time_step_rank) */
    int  ssm_inner_size;    /* VAL_DIM = NV * HV */
    int  ssm_key_dim;       /* KEY_DIM = LIN_NK * HV */
    int  ssm_conv_dim;      /* CONV_DIM = 2*KEY_DIM + VAL_DIM */
    int  ssm_conv_kernel;   /* causal conv1d kernel size */

    /* Multi-Token Prediction (MTP) / NextN */
    int  n_nextn_layers;    /* number of MTP/NextN layers at end (0 = none) */

    /* Expert / MoE (future — currently informational) */
    int  n_experts;         /* 0 = dense model */
    int  n_experts_used;    /* top-k experts per token */

} StratumConfig;

/* ------------------------------------------------------------------ */
/*  Universal layer-type classification                               */
/* ------------------------------------------------------------------ */

typedef enum {
    STRATUM_LAYER_FULL_ATTN = 0,  /* standard multi-head attention */
    STRATUM_LAYER_SSM,            /* state-space model / linear attention */
    STRATUM_LAYER_NEXTN,          /* MTP / NextN prediction layer */
} StratumLayerType;

/* Determine layer type from config — generic, works for any architecture */
static inline StratumLayerType stratum_layer_type(const StratumConfig* c, int layer) {
    if (c->n_nextn_layers > 0 && layer >= c->n_layers - c->n_nextn_layers)
        return STRATUM_LAYER_NEXTN;
    if (c->full_attn_stride <= 0)
        return STRATUM_LAYER_FULL_ATTN;  /* all-attn or no SSM */
    return ((layer + 1) % c->full_attn_stride == 0)
        ? STRATUM_LAYER_FULL_ATTN
        : STRATUM_LAYER_SSM;
}

static inline int stratum_n_main_layers(const StratumConfig* c) {
    return c->n_layers - c->n_nextn_layers;
}

static inline bool stratum_has_ssm(const StratumConfig* c) {
    return c->ssm_state_size > 0 && c->full_attn_stride != 0;
}

/* Compatibility helpers — match the old per-arch function signatures */
static inline bool stratum_is_full_attn_v2(const StratumConfig* c, int layer) {
    return stratum_layer_type(c, layer) == STRATUM_LAYER_FULL_ATTN;
}

static inline bool stratum_is_nextn_v2(const StratumConfig* c, int layer) {
    return stratum_layer_type(c, layer) == STRATUM_LAYER_NEXTN;
}

/* ------------------------------------------------------------------ */
/*  Architecture interface — each arch implements this                 */
/* ------------------------------------------------------------------ */

typedef struct StratumArch {
    /* Architecture name(s) this handler supports (comma-separated GGUF arch strings).
     * e.g. "llama" or "qwen35,qwen35moe" */
    const char* arch_names;

    /* Human-readable description for help text */
    const char* description;

    /* Main entry point: receives full argc/argv.
     * The handler is responsible for:
     *   - Opening the GGUF (argv[1])
     *   - Loading config via stratum_load_config()
     *   - Setting up state (tensors, KV cache, GPU, etc.)
     *   - Running inference (argv[2] = n_generate, argv[3..] = prompt tokens)
     *   - Cleanup (close GGUF, free state)
     * Returns 0 on success.
     *
     * This interface gives each architecture full control over its lifecycle,
     * while the dispatch (which architecture to call) is fully generic. */
    int  (*run)(int argc, char** argv);
} StratumArch;

/* ------------------------------------------------------------------ */
/*  Registry — automatic registration via __attribute__((constructor)) */
/* ------------------------------------------------------------------ */

#define STRATUM_MAX_ARCHS 16

typedef struct {
    const StratumArch* archs[STRATUM_MAX_ARCHS];
    int count;
} StratumArchRegistry;

/* Global registry — defined in stratum.c */
extern StratumArchRegistry g_arch_registry;

/* Register an architecture. Called by STRATUM_REGISTER_ARCH constructor. */
static inline void stratum_register_arch(const StratumArch* arch) {
    if (g_arch_registry.count < STRATUM_MAX_ARCHS) {
        g_arch_registry.archs[g_arch_registry.count++] = arch;
    }
}

/* Find an architecture handler by GGUF arch name.
 * Checks each registered arch's comma-separated arch_names. */
static inline const StratumArch* stratum_find_arch(const char* arch_name) {
    for (int i = 0; i < g_arch_registry.count; i++) {
        const StratumArch* a = g_arch_registry.archs[i];
        /* Parse comma-separated names */
        const char* p = a->arch_names;
        while (*p) {
            const char* comma = strchr(p, ',');
            int len = comma ? (int)(comma - p) : (int)strlen(p);
            if ((int)strlen(arch_name) == len && strncmp(p, arch_name, len) == 0)
                return a;
            if (!comma) break;
            p = comma + 1;
        }
    }
    return NULL;
}

/* Register macro — place in each arch .inc.c file at file scope.
 * Usage: STRATUM_REGISTER_ARCH(stratum_arch_llama)  (no & prefix) */
#define STRATUM_REGISTER_ARCH(arch_var) \
    __attribute__((constructor)) static void _stratum_register_##arch_var(void) { \
        stratum_register_arch(&arch_var); \
    }

/* ------------------------------------------------------------------ */
/*  Generic config loader — reads GGUF metadata into StratumConfig     */
/* ------------------------------------------------------------------ */

static inline int stratum_load_config(const Gguf* g, StratumConfig* c) {
    memset(c, 0, sizeof(*c));

    char* arch = gguf_get_string_dup(g, "general.architecture");
    if (!arch) {
        fprintf(stderr, "stratum_load_config: missing general.architecture\n");
        return -1;
    }
    strncpy(c->arch, arch, sizeof(c->arch) - 1);
    free(arch);

    char key[256];
    uint32_t u;
    float f;

#define NEED_U32(suffix, dst) \
    snprintf(key, sizeof(key), "%s.%s", c->arch, suffix); \
    if (gguf_get_u32(g, key, &u) != 0) { \
        fprintf(stderr, "stratum_load_config: missing %s\n", key); return -1; \
    } \
    *(dst) = (int)u;

#define OPT_U32(suffix, dst, def) \
    snprintf(key, sizeof(key), "%s.%s", c->arch, suffix); \
    if (gguf_get_u32(g, key, &u) == 0) *(dst) = (int)u; else *(dst) = (def);

#define OPT_F32(suffix, dst, def) \
    snprintf(key, sizeof(key), "%s.%s", c->arch, suffix); \
    if (gguf_get_f32(g, key, &f) == 0) *(dst) = f; else *(dst) = (def);

    /* Universal fields — every Transformer model has these */
    NEED_U32("block_count",          &c->n_layers);
    NEED_U32("embedding_length",     &c->n_embed);
    NEED_U32("feed_forward_length",  &c->n_ff);
    NEED_U32("attention.head_count", &c->n_q_heads);

    OPT_U32("attention.head_count_kv", &c->n_kv_heads, c->n_q_heads);

    /* head_dim: prefer metadata, fall back to tensor shape, then n_embed/n_q_heads */
    snprintf(key, sizeof(key), "%s.attention.key_length", c->arch);
    if (gguf_get_u32(g, key, &u) == 0) {
        c->head_dim = (int)u;
    } else {
        /* Try to infer from blk.0 tensor (NOT blk.3 — must work for any layer count) */
        const GgufTensor* tk = gguf_find_tensor(g, "blk.0.attn_k.weight");
        if (tk && tk->n_dims >= 2 && c->n_kv_heads > 0) {
            c->head_dim = (int)tk->dims[1] / c->n_kv_heads;
        } else {
            c->head_dim = c->n_embed / c->n_q_heads;
        }
    }

    OPT_U32("rope.dimension_count", &c->rope_dim, c->head_dim);
    OPT_F32("attention.layer_norm_rms_epsilon", &c->rms_eps, 1e-5f);
    OPT_F32("rope.freq_base", &c->rope_theta, 10000.0f);

    /* Vocabulary size from token_embd tensor */
    const GgufTensor* te = gguf_find_tensor(g, "token_embd.weight");
    if (!te || te->n_dims < 2) {
        fprintf(stderr, "stratum_load_config: missing token_embd.weight\n");
        return -1;
    }
    c->vocab_size = (int)te->dims[1];

    /* Output weight */
    c->has_output_weight = (gguf_find_tensor(g, "output.weight") != NULL);

    /* QK norm — check blk.0 (not blk.3, for generality) */
    c->has_qk_norm = (gguf_find_tensor(g, "blk.0.attn_q_norm.weight") != NULL);

    /* Full attention stride — for hybrid architectures.
     * Try metadata first, then auto-detect by probing layer 0 for attn tensors. */
    c->full_attn_stride = 0;  /* 0 = all full-attn (standard Transformer) */
    snprintf(key, sizeof(key), "%s.full_attention_interval", c->arch);
    if (gguf_get_u32(g, key, &u) == 0) {
        c->full_attn_stride = (int)u;
    } else {
        /* Auto-detect: if layer 0 has no attn_q.weight, it's an SSM layer.
         * Find the first full-attn layer to determine stride. */
        char nm[64];
        bool layer0_has_attn = false;
        snprintf(nm, sizeof nm, "blk.0.attn_q.weight");
        if (gguf_find_tensor(g, nm)) layer0_has_attn = true;

        if (!layer0_has_attn) {
            /* Hybrid model — find first full-attn layer */
            for (int i = 0; i < c->n_layers; i++) {
                snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", i);
                if (gguf_find_tensor(g, nm)) {
                    c->full_attn_stride = i + 1;
                    break;
                }
            }
            if (c->full_attn_stride <= 0)
                c->full_attn_stride = -1;  /* pure SSM, no full-attn at all */
        }
        /* If layer0_has_attn, full_attn_stride stays 0 = all full-attn */
    }

    /* SSM params — only loaded if SSM-related metadata exists.
     * No hardcoded defaults that assume any specific model. */
    snprintf(key, sizeof(key), "%s.ssm.state_size", c->arch);
    if (gguf_get_u32(g, key, &u) == 0) {
        c->ssm_state_size = (int)u;
        OPT_U32("ssm.group_count",   &c->ssm_group_count, 0);
        OPT_U32("ssm.time_step_rank", &c->ssm_value_heads, c->ssm_group_count);
        OPT_U32("ssm.inner_size",    &c->ssm_inner_size,
                c->ssm_value_heads * c->ssm_state_size);
        OPT_U32("ssm.conv_kernel",   &c->ssm_conv_kernel, 4);
        c->ssm_key_dim  = c->ssm_group_count * c->ssm_state_size;
        c->ssm_conv_dim = 2 * c->ssm_key_dim + c->ssm_inner_size;
    }
    /* If no SSM metadata: ssm_state_size stays 0 → stratum_has_ssm() returns false */

    /* MTP / NextN */
    OPT_U32("nextn_predict_layers", &c->n_nextn_layers, 0);

    /* MoE (informational — not yet implemented) */
    OPT_U32("expert_count",      &c->n_experts, 0);
    OPT_U32("expert_used_count", &c->n_experts_used, 0);

#undef NEED_U32
#undef OPT_U32
#undef OPT_F32
    return 0;
}

/* Print config — generic, shows all fields that are set */
static inline void stratum_print_config(const StratumConfig* c) {
    fprintf(stderr, "  arch              : %s\n", c->arch);
    fprintf(stderr, "  n_layers          : %d\n", c->n_layers);
    fprintf(stderr, "  n_embed           : %d\n", c->n_embed);
    fprintf(stderr, "  n_ff              : %d\n", c->n_ff);
    fprintf(stderr, "  n_q_heads         : %d\n", c->n_q_heads);
    fprintf(stderr, "  n_kv_heads        : %d\n", c->n_kv_heads);
    fprintf(stderr, "  head_dim          : %d\n", c->head_dim);
    fprintf(stderr, "  rope_dim          : %d\n", c->rope_dim);
    if (c->full_attn_stride > 0)
        fprintf(stderr, "  full_attn_stride  : %d  (hybrid: full-attn every %d layers)\n",
                c->full_attn_stride, c->full_attn_stride);
    else if (c->full_attn_stride < 0)
        fprintf(stderr, "  full_attn_stride  : pure SSM (no full-attention)\n");
    else
        fprintf(stderr, "  full_attn_stride  : all full-attention (standard Transformer)\n");
    fprintf(stderr, "  rms_eps           : %g\n", c->rms_eps);
    fprintf(stderr, "  rope_theta        : %g\n", c->rope_theta);
    fprintf(stderr, "  vocab_size        : %d\n", c->vocab_size);
    fprintf(stderr, "  has_output_w      : %s\n", c->has_output_weight ? "yes" : "no");
    fprintf(stderr, "  has_qk_norm       : %s\n", c->has_qk_norm ? "yes" : "no");
    if (stratum_has_ssm(c)) {
        fprintf(stderr, "  ssm.state_size    : %d\n", c->ssm_state_size);
        fprintf(stderr, "  ssm.group_count   : %d\n", c->ssm_group_count);
        fprintf(stderr, "  ssm.value_heads   : %d\n", c->ssm_value_heads);
        fprintf(stderr, "  ssm.inner_size    : %d\n", c->ssm_inner_size);
        fprintf(stderr, "  ssm.key_dim       : %d\n", c->ssm_key_dim);
        fprintf(stderr, "  ssm.conv_dim      : %d\n", c->ssm_conv_dim);
        fprintf(stderr, "  ssm.conv_kernel   : %d\n", c->ssm_conv_kernel);
    }
    if (c->n_nextn_layers > 0)
        fprintf(stderr, "  n_nextn_layers    : %d\n", c->n_nextn_layers);
    if (c->n_experts > 0) {
        fprintf(stderr, "  n_experts         : %d\n", c->n_experts);
        fprintf(stderr, "  n_experts_used    : %d\n", c->n_experts_used);
    }
}

#endif /* STRATUM_ARCH_H */
