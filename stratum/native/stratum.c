/*
 * stratum.c — Universal GGUF inference engine.
 *
 * Stratum is a GENERIC engine. It does not hardcode any model names,
 * architecture parameters, or model-specific logic. Each architecture
 * registers itself via STRATUM_REGISTER_ARCH(); this main file simply
 * reads general.architecture from the GGUF and dispatches to the
 * registered handler.
 *
 * To add a new architecture:
 *   1. Create stratum_arch_<name>.inc.c implementing StratumArch
 *   2. #include it below
 *   3. That's it — no changes to this file.
 */
#define _GNU_SOURCE
#include "stratum_arch.h"
#include "stratum_linear.h"
#include "stratum_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Global registry — referenced by stratum_arch.h */
StratumArchRegistry g_arch_registry = {0};

/* Global linear engine state — referenced by stratum_linear.h */
StratumLinearState g_st = {0};

/* Read general.architecture from a GGUF file */
static int read_arch(const char* path, char* arch_out, size_t cap) {
    Gguf g;
    if (gguf_open(path, &g) != 0) return -1;
    char* arch = gguf_get_string_dup(&g, "general.architecture");
    if (!arch) {
        gguf_close(&g);
        return -1;
    }
    strncpy(arch_out, arch, cap - 1);
    arch_out[cap - 1] = '\0';
    free(arch);
    gguf_close(&g);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "stratum — universal GGUF inference\n"
                "\n"
                "usage: %s <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]\n"
                "\n"
                "Reads the GGUF file's `general.architecture` metadata key\n"
                "and dispatches to the appropriate registered handler.\n"
                "\n"
                "Registered architectures (%d):\n",
                argv[0], g_arch_registry.count);

        /* Dynamically list registered architectures — no hardcoded names */
        for (int i = 0; i < g_arch_registry.count; i++) {
            fprintf(stderr, "  %-16s %s\n",
                    g_arch_registry.archs[i]->arch_names,
                    g_arch_registry.archs[i]->description ?: "");
        }
        return 1;
    }

    char arch[64] = {0};
    if (read_arch(argv[1], arch, sizeof arch) != 0) {
        fprintf(stderr, "could not read architecture from %s\n", argv[1]);
        return 1;
    }

    fprintf(stderr, "== stratum — universal GGUF inference ==\n");
    fprintf(stderr, "  detected architecture: %s\n", arch);

    /* Find registered handler — no hardcoded strcmp chains */
    const StratumArch* handler = stratum_find_arch(arch);
    if (!handler) {
        fprintf(stderr, "  ERROR: architecture '%s' is not registered.\n", arch);
        fprintf(stderr, "  Registered architectures (%d):\n", g_arch_registry.count);
        for (int i = 0; i < g_arch_registry.count; i++) {
            fprintf(stderr, "    %-16s %s\n",
                    g_arch_registry.archs[i]->arch_names,
                    g_arch_registry.archs[i]->description ?: "");
        }
        return 1;
    }

    fprintf(stderr, "  handler: %s\n", handler->description ?: handler->arch_names);

    /* Dispatch to architecture handler — no hardcoded logic here.
     * The handler owns its full lifecycle (init → run → cleanup).
     * In STRATUM_SERVER mode, the handler may loop internally on stdin. */
    return handler->run(argc, argv);
}

/* Architecture implementations — each registers itself via constructor.
 * To add a new architecture, add its #include here. */
#include "stratum_arch_llama.inc.c"
#include "stratum_arch_qwen35.inc.c"
