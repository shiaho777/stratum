/* Hostile-GGUF cases for gguf_open. A file that used to come back as a
 * successful open (negative nbytes, skipped bounds check, or a stack dive
 * through nested arrays) must now fail before any tensor pointer is used.
 */
#include "stratum_gguf.h"

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
        if (!q) { perror("realloc"); exit(1); }
        b->p = q;
        b->cap = c;
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
static void b_pad32(Buf* b) {
    while (b->n & 31) {
        uint8_t z = 0;
        b_add(b, &z, 1);
    }
}

/* One tensor. data_len bytes are appended after 32-byte alignment.
 * offset_field is the relative offset stored in the index. */
static Buf make_file(uint32_t n_dims, const uint64_t* dims, uint32_t type,
                     uint64_t offset_field, const void* data, size_t data_len,
                     int with_nested, int nest_depth) {
    Buf kv = {0};
    b_str(&kv, "general.alignment");
    b_u32(&kv, 4);          /* UINT32 */
    b_u32(&kv, 32);
    uint64_t n_kv = 1;
    if (with_nested) {
        b_str(&kv, "test.nested");
        b_u32(&kv, 9);      /* ARRAY */
        /* value: etype + count + payload, `nest_depth` arrays deep,
         * innermost an empty UINT8 array. */
        for (int d = 0; d < nest_depth; d++) {
            b_u32(&kv, 9);
            b_u64(&kv, 1);
        }
        b_u32(&kv, 0);      /* UINT8 */
        b_u64(&kv, 0);
        n_kv = 2;
    }

    Buf idx = {0};
    const char* name = "w";
    b_str(&idx, name);
    b_u32(&idx, n_dims);
    for (uint32_t d = 0; d < n_dims; d++) b_u64(&idx, dims[d]);
    b_u32(&idx, type);
    b_u64(&idx, offset_field);

    Buf file = {0};
    b_add(&file, "GGUF", 4);
    b_u32(&file, 3);
    b_u64(&file, 1);        /* n_tensors */
    b_u64(&file, n_kv);
    b_add(&file, kv.p, kv.n);
    b_add(&file, idx.p, idx.n);
    b_pad32(&file);
    if (data_len) b_add(&file, data, data_len);
    free(kv.p);
    free(idx.p);
    return file;
}

static int open_buf(const Buf* b, Gguf* g) {
    char path[] = "/tmp/stratum-gguf-bounds-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); exit(1); }
    if (write(fd, b->p, b->n) != (ssize_t)b->n) { perror("write"); exit(1); }
    close(fd);
    int rc = gguf_open(path, g);
    unlink(path);
    return rc;
}

int main(void) {
    uint8_t sixteen[16];
    memset(sixteen, 0xab, sizeof sixteen);

    /* 4 f32 elements, fully inside the file. */
    {
        uint64_t dims[1] = {4};
        Buf f = make_file(1, dims, 0 /* F32 */, 0, sixteen, sizeof sixteen, 0, 0);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("f32 x4 opens", rc, 0);
        if (rc == 0) {
            expect_int("f32 nelem", (int)g.tensors[0].nelem, 4);
            expect_int("f32 nbytes", (int)g.tensors[0].nbytes, 16);
            gguf_close(&g);
        }
        free(f.p);
    }

    /* 2^62 * 8 overflows int64. Previously nbytes went negative and open
     * returned success. */
    {
        uint64_t dims[2] = {1ULL << 62, 8};
        Buf f = make_file(2, dims, 0, 0, NULL, 0, 0, 0);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("overflowing dims rejected", rc, -1);
        free(f.p);
    }

    /* Single dim that fits in int64 but whose f32 byte size does not. */
    {
        uint64_t dims[1] = {(uint64_t)(INT64_MAX / 4) + 1};
        Buf f = make_file(1, dims, 0, 0, NULL, 0, 0, 0);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("overflowing byte size rejected", rc, -1);
        free(f.p);
    }

    /* Q4_K block is 256 elements. 100 is not a multiple, so a truncated
     * nbytes would under-state the read. */
    {
        uint64_t dims[1] = {100};
        Buf f = make_file(1, dims, 12 /* Q4_K */, 0, NULL, 0, 0, 0);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("partial q4k block rejected", rc, -1);
        free(f.p);
    }

    /* A dim that does not fit in a positive int64. */
    {
        uint64_t dims[1] = {~0ULL};
        Buf f = make_file(1, dims, 0, 0, NULL, 0, 0, 0);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("dim above int64 rejected", rc, -1);
        free(f.p);
    }

    /* Unknown type stays accepted: the engine does not know its size and
     * the existing guard intentionally skips it. */
    {
        uint64_t dims[1] = {4};
        Buf f = make_file(1, dims, 99, 0, NULL, 0, 0, 0);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("unknown type still opens", rc, 0);
        if (rc == 0) {
            expect_int("unknown nbytes", (int)g.tensors[0].nbytes, -1);
            gguf_close(&g);
        }
        free(f.p);
    }

    /* Shallow array metadata still parses. */
    {
        uint64_t dims[1] = {4};
        Buf f = make_file(1, dims, 0, 0, sixteen, sizeof sixteen, 1, 2);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("array depth 2 opens", rc, 0);
        if (rc == 0) gguf_close(&g);
        free(f.p);
    }

    /* Deeper than GGUF_VALUE_MAX_DEPTH must fail closed, not crash. */
    {
        uint64_t dims[1] = {4};
        Buf f = make_file(1, dims, 0, 0, sixteen, sizeof sixteen, 1, 64);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("array depth 64 rejected", rc, -1);
        free(f.p);
    }

    /* Tensor that claims 16 bytes but the file ends first. */
    {
        uint64_t dims[1] = {4};
        Buf f = make_file(1, dims, 0, 0, NULL, 0, 0, 0);
        Gguf g;
        int rc = open_buf(&f, &g);
        expect_int("tensor past eof rejected", rc, -1);
        free(f.p);
    }

    if (g_fail) {
        fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    printf("gguf_bounds_test: all passed\n");
    return 0;
}
