/* stratum_load_config must reject shapes that would divide by zero, wrap
 * a uint32 into a negative int, or overflow an SSM dimension product.
 * A normal tiny geometry still loads. */
#include "stratum_arch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_fail;

static void expect_int(const char* name, int got, int want) {
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %d want %d\n", name, got, want);
        g_fail++;
    } else {
        printf("PASS %s\n", name);
    }
}

typedef struct { uint8_t* p; size_t n, cap; } Buf;
static void b_add(Buf* b, const void* src, size_t n) {
    if (b->n + n > b->cap) {
        size_t c = b->cap ? b->cap * 2 : 256;
        while (c < b->n + n) c *= 2;
        uint8_t* q = (uint8_t*)realloc(b->p, c);
        if (!q) exit(1);
        b->p = q; b->cap = c;
    }
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void b_u32(Buf* b, uint32_t v) { b_add(b, &v, 4); }
static void b_u64(Buf* b, uint64_t v) { b_add(b, &v, 8); }
static void b_str(Buf* b, const char* s) {
    b_u64(b, (uint64_t)strlen(s));
    b_add(b, s, strlen(s));
}
static void b_kv_u32(Buf* b, const char* k, uint32_t v) {
    b_str(b, k); b_u32(b, 4); b_u32(b, v);
}
static void b_kv_str(Buf* b, const char* k, const char* v) {
    b_str(b, k); b_u32(b, 8); b_str(b, v);
}

typedef struct {
    uint32_t layers, embed, ff, nq, nkv, vocab;
    int has_nextn; uint32_t nextn;
    int has_experts; uint32_t nexp, nused;
    int has_ssm; uint32_t state, groups;
} Spec;

static int load_spec(const Spec* s, StratumConfig* cfg) {
    Buf kv = {0};
    b_kv_str(&kv, "general.architecture", "llama");
    b_kv_u32(&kv, "general.alignment", 32);
    b_kv_u32(&kv, "llama.block_count", s->layers);
    b_kv_u32(&kv, "llama.embedding_length", s->embed);
    b_kv_u32(&kv, "llama.feed_forward_length", s->ff);
    b_kv_u32(&kv, "llama.attention.head_count", s->nq);
    b_kv_u32(&kv, "llama.attention.head_count_kv", s->nkv);
    b_kv_u32(&kv, "llama.attention.key_length", 8);
    if (s->has_nextn) b_kv_u32(&kv, "llama.nextn_predict_layers", s->nextn);
    if (s->has_experts) {
        b_kv_u32(&kv, "llama.expert_count", s->nexp);
        b_kv_u32(&kv, "llama.expert_used_count", s->nused);
    }
    if (s->has_ssm) {
        b_kv_u32(&kv, "llama.ssm.state_size", s->state);
        b_kv_u32(&kv, "llama.ssm.group_count", s->groups);
    }
    uint64_t n_kv = 8;
    if (s->has_nextn) n_kv++;
    if (s->has_experts) n_kv += 2;
    if (s->has_ssm) n_kv += 2;

    Buf idx = {0};
    b_str(&idx, "token_embd.weight");
    b_u32(&idx, 2);
    b_u64(&idx, s->embed ? s->embed : 1);
    b_u64(&idx, s->vocab);
    b_u32(&idx, 0); /* F32 */
    b_u64(&idx, 0);

    Buf file = {0};
    b_add(&file, "GGUF", 4);
    b_u32(&file, 3);
    b_u64(&file, 1);
    b_u64(&file, n_kv);
    b_add(&file, kv.p, kv.n);
    b_add(&file, idx.p, idx.n);
    while (file.n & 31) { uint8_t z = 0; b_add(&file, &z, 1); }
    uint64_t elems = (uint64_t)(s->embed ? s->embed : 1) * s->vocab;
    for (uint64_t i = 0; i < elems * 4; i++) { uint8_t z = 0; b_add(&file, &z, 1); }

    char path[] = "/tmp/stratum-cfg-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) exit(1);
    if (write(fd, file.p, file.n) != (ssize_t)file.n) exit(1);
    close(fd);
    free(kv.p); free(idx.p); free(file.p);

    Gguf g;
    int orc = gguf_open(path, &g);
    unlink(path);
    if (orc != 0) return -2;
    int rc = stratum_load_config(&g, cfg);
    gguf_close(&g);
    return rc;
}

static Spec base(void) {
    Spec s;
    memset(&s, 0, sizeof s);
    s.layers = 2; s.embed = 32; s.ff = 64; s.nq = 4; s.nkv = 2; s.vocab = 16;
    return s;
}

int main(void) {
    StratumConfig c;
    Spec s = base();
    int rc = load_spec(&s, &c);
    expect_int("sane geometry loads", rc, 0);
    if (rc == 0) {
        expect_int("layers", c.n_layers, 2);
        expect_int("vocab", c.vocab_size, 16);
        expect_int("head_dim", c.head_dim, 8);
    }

    s = base(); s.nq = 0; s.nkv = 0;
    expect_int("zero head count rejected", load_spec(&s, &c), -1);

    s = base(); s.layers = 3000000000u;
    expect_int("block_count above INT_MAX rejected", load_spec(&s, &c), -1);

    s = base(); s.layers = 0;
    expect_int("zero block_count rejected", load_spec(&s, &c), -1);

    s = base(); s.vocab = 0;
    expect_int("zero vocab rejected", load_spec(&s, &c), -1);

    s = base(); s.has_nextn = 1; s.nextn = 9;
    expect_int("nextn past layer count rejected", load_spec(&s, &c), -1);

    s = base(); s.has_experts = 1; s.nexp = 2; s.nused = 4;
    expect_int("expert_used above expert_count rejected", load_spec(&s, &c), -1);

    s = base(); s.has_ssm = 1; s.state = 100000; s.groups = 100000;
    expect_int("ssm dimension product rejected", load_spec(&s, &c), -1);

    if (g_fail) {
        fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    printf("config_geometry_test: all passed\n");
    return 0;
}
