/*
 * input_guards_test.c — model-free regression for GGUF parser guards and
 * KV-cache geometry. Every case builds its own bytes; no weights needed.
 *
 * Covers:
 *   - tensor offset + body_offset must not wrap uint64
 *   - an offset must be inside the file even for an unknown quant type
 *     (whose byte size is deliberately "not computable")
 *   - implausible n_tensors / n_kv counts are rejected before calloc/loops
 *   - the intentional size-not-computable path still accepts a valid offset
 *   - STRATUM_MAX_KV / STRATUM_KV_SINK cannot produce ring_size <= 0
 *
 * A well-formed baseline is included first: these guards must reject corrupt
 * files without rejecting ordinary ones. Run standalone or via `make tests`.
 */
#include "stratum_gguf.h"
#include "stratum_engine.h"

static int failures = 0;

#define CHECK(cond, ...) do {                                        \
    if (!(cond)) { fprintf(stderr, "FAIL "); failures++; }          \
    else         { fprintf(stderr, "ok   "); }                       \
    fprintf(stderr, __VA_ARGS__);                                    \
    fprintf(stderr, "\n");                                           \
} while (0)

static void put_u32(FILE* f, uint32_t v) { fwrite(&v, sizeof v, 1, f); }
static void put_u64(FILE* f, uint64_t v) { fwrite(&v, sizeof v, 1, f); }
static void put_str(FILE* f, const char* s) {
    uint64_t n = (uint64_t)strlen(s);
    put_u64(f, n);
    fwrite(s, 1, (size_t)n, f);
}

/* Minimal GGUF v3: one KV entry, `n_tensors` tensor infos in the header.
 * Only a handful of infos are actually written even when the header claims
 * a huge count — that is exactly the corrupt-input shape under test. */
static void write_case(const char* path, uint64_t n_tensors, uint64_t n_kv,
                       uint32_t type, uint64_t offset, size_t body_bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen"); exit(2); }
    fwrite("GGUF", 1, 4, f);
    put_u32(f, 3);
    put_u64(f, n_tensors);
    put_u64(f, n_kv);
    put_str(f, "general.alignment");
    put_u32(f, GGUF_VAL_UINT32);
    put_u32(f, 32);
    uint64_t written = n_tensors > 4 ? 4 : n_tensors;
    for (uint64_t i = 0; i < written; i++) {
        put_str(f, "token_embd.weight");
        put_u32(f, 2);              /* n_dims */
        put_u64(f, 4);
        put_u64(f, 4);              /* nelem = 16 -> F32 nbytes = 64 */
        put_u32(f, type);
        put_u64(f, offset);
    }
    long pos = ftell(f);
    long pad = (32 - (pos % 32)) % 32;
    while (pad-- > 0) fputc(0, f);
    for (size_t i = 0; i < body_bytes; i++) fputc(0, f);
    fclose(f);
}

static int open_expect(const char* label, uint64_t n_tensors, uint64_t n_kv,
                       uint32_t type, uint64_t offset, size_t body, int want) {
    const char* path = "/tmp/stratum_input_guards.gguf";
    write_case(path, n_tensors, n_kv, type, offset, body);
    Gguf g;
    memset(&g, 0, sizeof g);
    int rc = gguf_open(path, &g);
    int got = (rc == 0);
    if (got) gguf_close(&g);
    unlink(path);
    CHECK(got == want, "%s: gguf_open=%d (expected %s)",
          label, rc, want ? "success" : "rejection");
    return got == want;
}

int main(void) {
    fprintf(stderr, "== input guards (model-free) ==\n");

    /* No false positives: an ordinary small file must still open. */
    open_expect("baseline well-formed", 1, 1, GGML_TYPE_F32, 0, 512, 1);

    /* offset near UINT64_MAX wraps to a small in-range value when
     * body_offset is added, which used to slip past the size guard. */
    open_expect("offset overflow wraps", 1, 1, GGML_TYPE_F32,
                UINT64_MAX - 8, 512, 0);

    /* Unknown quant type -> nbytes = -1. The size check is skipped by
     * design, but the pointer still has to name a byte inside the file. */
    open_expect("unknown type, offset outside file", 1, 1, 100,
                1ULL << 40, 512, 0);
    open_expect("unknown type, valid offset accepted", 1, 1, 100, 0, 512, 1);

    /* Corrupt counts must be refused before a giant calloc or a long loop. */
    open_expect("implausible n_tensors", 1ULL << 40, 1,
                GGML_TYPE_F32, 0, 512, 0);
    open_expect("implausible n_kv", 1, 1ULL << 40,
                GGML_TYPE_F32, 0, 512, 0);

    /* KV ring geometry: max_kv <= sink used to hit % 0 / % negative. */
    StratumKVCache kv;
    stratum_kv_init(&kv, 8, 4);
    CHECK(stratum_kv_slot(&kv, 9) == 5, "normal ring slot 9 -> 5");
    stratum_kv_init(&kv, 4, 4);
    CHECK(stratum_kv_slot(&kv, 9) == 0, "zero ring_size returns a safe slot");
    stratum_kv_init(&kv, 2, 8);
    CHECK(stratum_kv_slot(&kv, 9) == 0, "negative ring_size returns a safe slot");

    unsetenv("STRATUM_MAX_KV");
    unsetenv("STRATUM_KV_SINK");
    stratum_engine_init(1024);
    CHECK(g_eng.max_kv == 4096 && g_eng.kv_sink == 4,
          "default KV geometry 4096/4");

    setenv("STRATUM_MAX_KV", "4", 1);
    setenv("STRATUM_KV_SINK", "4", 1);
    stratum_engine_init(1024);
    CHECK(g_eng.max_kv == 4096 && g_eng.kv_sink == 4,
          "invalid env KV geometry ignored, defaults kept");

    setenv("STRATUM_MAX_KV", "64", 1);
    setenv("STRATUM_KV_SINK", "8", 1);
    stratum_engine_init(1024);
    CHECK(g_eng.max_kv == 64 && g_eng.kv_sink == 8,
          "valid env KV geometry accepted 64/8");
    unsetenv("STRATUM_MAX_KV");
    unsetenv("STRATUM_KV_SINK");

    fprintf(stderr, "== input guards: %d failure(s) ==\n", failures);
    return failures ? 1 : 0;
}
