
#include "stratum_llama.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s path/to/file.gguf\n", argv[0]);
        return 1;
    }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) return 1;

    printf("== GGUF ==\n");
    printf("  version : %u\n", g.version);
    printf("  tensors : %llu\n", (unsigned long long)g.n_tensors);
    printf("  body off: %llu\n", (unsigned long long)g.body_offset);
    printf("\n== Llama config ==\n");
    LlamaConfig cfg;
    if (llama_load_config(&g, &cfg) != 0) {
        gguf_close(&g);
        return 1;
    }
    llama_print_config(&cfg);

    char nm[128];
    const GgufTensor* t;
    printf("\n== Sanity check ==\n");
    t = gguf_find_tensor(&g, "token_embd.weight");
    if (t) printf("  token_embd:    dims=[%llu,%llu] type=%s\n",
                  (unsigned long long)t->dims[0],
                  (unsigned long long)t->dims[1],
                  gguf_type_name((GgmlType)t->type));
    snprintf(nm, sizeof nm, "blk.0.attn_q.weight");
    t = gguf_find_tensor(&g, nm);
    if (t) printf("  blk.0.attn_q:  dims=[%llu,%llu] type=%s\n",
                  (unsigned long long)t->dims[0],
                  (unsigned long long)t->dims[1],
                  gguf_type_name((GgmlType)t->type));
    snprintf(nm, sizeof nm, "blk.0.attn_k.weight");
    t = gguf_find_tensor(&g, nm);
    if (t) printf("  blk.0.attn_k:  dims=[%llu,%llu] type=%s\n",
                  (unsigned long long)t->dims[0],
                  (unsigned long long)t->dims[1],
                  gguf_type_name((GgmlType)t->type));
    snprintf(nm, sizeof nm, "blk.0.ffn_gate.weight");
    t = gguf_find_tensor(&g, nm);
    if (t) printf("  blk.0.ffn_gate: dims=[%llu,%llu] type=%s\n",
                  (unsigned long long)t->dims[0],
                  (unsigned long long)t->dims[1],
                  gguf_type_name((GgmlType)t->type));

    gguf_close(&g);
    return 0;
}
