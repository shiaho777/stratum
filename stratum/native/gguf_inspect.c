
#include "stratum_gguf.h"

#include <errno.h>
#include <inttypes.h>

static void dump_arch_metadata(const Gguf* g) {
    char* arch = gguf_get_string_dup(g, "general.architecture");
    char* name = gguf_get_string_dup(g, "general.name");
    printf("  general.architecture = %s\n", arch ? arch : "?");
    printf("  general.name         = %s\n", name ? name : "?");

    char buf[256];
    uint32_t u; uint64_t u64; float f;

    const char* keys[] = {
        "context_length", "block_count", "embedding_length",
        "feed_forward_length", "attention.head_count",
        "attention.head_count_kv", "rope.dimension_count",
        NULL
    };
    for (int i = 0; keys[i]; i++) {
        if (!arch) break;
        snprintf(buf, sizeof buf, "%s.%s", arch, keys[i]);
        if (gguf_get_u32(g, buf, &u) == 0) {
            printf("  %-40s = %u\n", buf, u);
        } else if (gguf_get_u64(g, buf, &u64) == 0) {
            printf("  %-40s = %" PRIu64 "\n", buf, u64);
        }
    }
    if (arch) {
        snprintf(buf, sizeof buf, "%s.attention.layer_norm_rms_epsilon", arch);
        if (gguf_get_f32(g, buf, &f) == 0) {
            printf("  %-40s = %g\n", buf, f);
        }
        snprintf(buf, sizeof buf, "%s.rope.freq_base", arch);
        if (gguf_get_f32(g, buf, &f) == 0) {
            printf("  %-40s = %g\n", buf, f);
        }
    }
    free(arch);
    free(name);
}

static void dump_tensor_summary(const Gguf* g) {

    int64_t per_type_count[GGML_TYPE_COUNT] = {0};
    int64_t per_type_bytes[GGML_TYPE_COUNT] = {0};
    int64_t total_bytes = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const GgufTensor* t = &g->tensors[i];
        if (t->type < GGML_TYPE_COUNT) {
            per_type_count[t->type]++;
            per_type_bytes[t->type] += t->nbytes;
        }
        total_bytes += t->nbytes;
    }
    printf("  total tensors        = %" PRIu64 "\n", g->n_tensors);
    printf("  total body bytes     = %" PRId64 " (%.2f GB)\n",
           total_bytes, total_bytes / 1e9);
    printf("  body offset in file  = %" PRIu64 "\n", g->body_offset);
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        if (per_type_count[t] == 0) continue;
        printf("    %-8s %4" PRId64 " tensors  %.2f GB\n",
               gguf_type_name((GgmlType)t),
               per_type_count[t],
               per_type_bytes[t] / 1e9);
    }
}

static void dump_sample_tensors(const Gguf* g, int n) {
    int dumped = 0;
    for (uint64_t i = 0; i < g->n_tensors && dumped < n; i++) {
        const GgufTensor* t = &g->tensors[i];
        printf("    %s  dims=[", t->name);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            printf("%" PRIu64 "%s", t->dims[d], d + 1 < t->n_dims ? "," : "");
        }
        printf("]  type=%s  off=%" PRIu64 "  bytes=%" PRId64 "\n",
               gguf_type_name((GgmlType)t->type),
               t->offset,
               t->nbytes);
        dumped++;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s path/to/file.gguf\n", argv[0]);
        return 1;
    }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) {
        return 1;
    }
    printf("GGUF v%u  tensors=%" PRIu64 "  metadata_pairs=%" PRIu64 "\n",
           g.version, g.n_tensors, g.n_kv);
    printf("  alignment            = %" PRIu64 "\n", g.alignment);

    printf("\n=== Metadata ===\n");
    dump_arch_metadata(&g);

    printf("\n=== Tensor types ===\n");
    dump_tensor_summary(&g);

    printf("\n=== Sample tensors (first 10) ===\n");
    dump_sample_tensors(&g, 10);

    gguf_close(&g);
    return 0;
}
