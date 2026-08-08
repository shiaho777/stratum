
#ifndef STRATUM_GGUF_H
#define STRATUM_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* MemX integration — include before gguf_open so it can override mmap */
#include "stratum_memx.h"

typedef enum {
    GGML_TYPE_F32      = 0,
    GGML_TYPE_F16      = 1,
    GGML_TYPE_Q4_0     = 2,
    GGML_TYPE_Q4_1     = 3,
    GGML_TYPE_Q5_0     = 6,
    GGML_TYPE_Q5_1     = 7,
    GGML_TYPE_Q8_0     = 8,
    GGML_TYPE_Q8_1     = 9,
    GGML_TYPE_Q2_K     = 10,
    GGML_TYPE_Q3_K     = 11,
    GGML_TYPE_Q4_K     = 12,
    GGML_TYPE_Q5_K     = 13,
    GGML_TYPE_Q6_K     = 14,
    GGML_TYPE_Q8_K     = 15,
    GGML_TYPE_IQ4_NL   = 20,
    GGML_TYPE_IQ4_XS   = 23,
    GGML_TYPE_BF16     = 30,
    GGML_TYPE_COUNT,
} GgmlType;

typedef struct {
    int     elements_per_block;
    int     bytes_per_block;
    const char* name;
} GgmlTypeInfo;

static const GgmlTypeInfo gguf_type_info[] = {
    [GGML_TYPE_F32]    = { 1,   4,   "F32"   },
    [GGML_TYPE_F16]    = { 1,   2,   "F16"   },
    [GGML_TYPE_Q4_0]   = { 32,  18,  "Q4_0"  },
    [GGML_TYPE_Q4_1]   = { 32,  20,  "Q4_1"  },
    [GGML_TYPE_Q5_0]   = { 32,  22,  "Q5_0"  },
    [GGML_TYPE_Q5_1]   = { 32,  24,  "Q5_1"  },
    [GGML_TYPE_Q8_0]   = { 32,  34,  "Q8_0"  },
    [GGML_TYPE_Q8_1]   = { 32,  40,  "Q8_1"  },
    [GGML_TYPE_Q2_K]   = { 256, 84,  "Q2_K"  },
    [42]               = { 256, 148, "Q2K_NIB" },   /* V55: nibble 布局 Q2K (数值不变) */
    [GGML_TYPE_Q3_K]   = { 256, 110, "Q3_K"  },
    [GGML_TYPE_Q4_K]   = { 256, 144, "Q4_K"  },
    [GGML_TYPE_Q5_K]   = { 256, 176, "Q5_K"  },
    [GGML_TYPE_Q6_K]   = { 256, 210, "Q6_K"  },
    [GGML_TYPE_Q8_K]   = { 256, 292, "Q8_K"  },
    [GGML_TYPE_IQ4_NL] = { 32,  18,  "IQ4_NL"},
    [GGML_TYPE_IQ4_XS] = { 256, 136, "IQ4_XS"},
    [GGML_TYPE_BF16]   = { 1,   2,   "BF16"  },
};

static inline int64_t gguf_tensor_bytes(GgmlType t, int64_t nelem) {
    if (t >= GGML_TYPE_COUNT || gguf_type_info[t].name == NULL) return -1;
    int eb = gguf_type_info[t].elements_per_block;
    int bb = gguf_type_info[t].bytes_per_block;
    return (nelem / eb) * bb;
}

static inline const char* gguf_type_name(GgmlType t) {
    if (t >= GGML_TYPE_COUNT || gguf_type_info[t].name == NULL) return "?";
    return gguf_type_info[t].name;
}

typedef enum {
    GGUF_VAL_UINT8   = 0,
    GGUF_VAL_INT8    = 1,
    GGUF_VAL_UINT16  = 2,
    GGUF_VAL_INT16   = 3,
    GGUF_VAL_UINT32  = 4,
    GGUF_VAL_INT32   = 5,
    GGUF_VAL_FLOAT32 = 6,
    GGUF_VAL_BOOL    = 7,
    GGUF_VAL_STRING  = 8,
    GGUF_VAL_ARRAY   = 9,
    GGUF_VAL_UINT64  = 10,
    GGUF_VAL_INT64   = 11,
    GGUF_VAL_FLOAT64 = 12,
} GgufValType;

typedef struct {
    char*   key;
    uint32_t vtype;

    const uint8_t* bytes;
    size_t  bytes_len;
} GgufKV;

typedef struct {
    char*    name;
    uint32_t n_dims;
    uint64_t dims[8];
    uint32_t type;
    uint64_t offset;
    int64_t  nelem;
    int64_t  nbytes;
} GgufTensor;

typedef struct {
    int             fd;
    const uint8_t*  mmap_base;
    size_t          mmap_size;

    uint32_t        version;
    uint64_t        n_tensors;
    uint64_t        n_kv;

    GgufKV*         kv;
    GgufTensor*     tensors;

    uint64_t        body_offset;
    uint64_t        alignment;
} Gguf;

typedef struct { const uint8_t* p; const uint8_t* end; } GgufCur;

static inline int gguf_take(GgufCur* c, void* dst, size_t n) {
    if ((size_t)(c->end - c->p) < n) return -1;
    memcpy(dst, c->p, n);
    c->p += n;
    return 0;
}
static inline int gguf_skip(GgufCur* c, size_t n) {
    if ((size_t)(c->end - c->p) < n) return -1;
    c->p += n;
    return 0;
}
static inline int gguf_take_u32(GgufCur* c, uint32_t* v) { return gguf_take(c, v, 4); }
static inline int gguf_take_u64(GgufCur* c, uint64_t* v) { return gguf_take(c, v, 8); }

static char* gguf_take_string_dup(GgufCur* c) {
    uint64_t n;
    if (gguf_take_u64(c, &n) != 0) return NULL;
    if ((size_t)(c->end - c->p) < n) return NULL;
    char* s = (char*)malloc(n + 1);
    if (!s) return NULL;
    memcpy(s, c->p, n);
    s[n] = 0;
    c->p += n;
    return s;
}

static int gguf_skip_value(GgufCur* c, uint32_t vtype) {
    switch (vtype) {
        case GGUF_VAL_UINT8:
        case GGUF_VAL_INT8:
        case GGUF_VAL_BOOL:    return gguf_skip(c, 1);
        case GGUF_VAL_UINT16:
        case GGUF_VAL_INT16:   return gguf_skip(c, 2);
        case GGUF_VAL_UINT32:
        case GGUF_VAL_INT32:
        case GGUF_VAL_FLOAT32: return gguf_skip(c, 4);
        case GGUF_VAL_UINT64:
        case GGUF_VAL_INT64:
        case GGUF_VAL_FLOAT64: return gguf_skip(c, 8);
        case GGUF_VAL_STRING: {
            uint64_t n;
            if (gguf_take_u64(c, &n) != 0) return -1;
            return gguf_skip(c, n);
        }
        case GGUF_VAL_ARRAY: {
            uint32_t et;
            uint64_t cnt;
            if (gguf_take_u32(c, &et) != 0) return -1;
            if (gguf_take_u64(c, &cnt) != 0) return -1;
            for (uint64_t i = 0; i < cnt; i++) {
                if (gguf_skip_value(c, et) != 0) return -1;
            }
            return 0;
        }
    }
    return -1;
}

static int gguf_open(const char* path, Gguf* gguf) {
    memset(gguf, 0, sizeof(*gguf));
    gguf->fd = open(path, O_RDONLY);
    if (gguf->fd < 0) {
        fprintf(stderr, "gguf_open: open(%s): %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(gguf->fd, &st) != 0) {
        perror("gguf_open: fstat");
        close(gguf->fd); gguf->fd = -1;
        return -1;
    }
    gguf->mmap_size = (size_t)st.st_size;
    void* p = MAP_FAILED;

    /* Try MemX-managed mmap first (if STRATUM_USE_MEMX compiled + activated) */
    if (stratum_memx_is_active()) {
        p = stratum_memx_weight_mmap(gguf->fd, gguf->mmap_size);
    }
    /* Fallback to system mmap */
    if (p == MAP_FAILED) {
        p = mmap(NULL, gguf->mmap_size, PROT_READ, MAP_PRIVATE, gguf->fd, 0);
    }
    if (p == MAP_FAILED) {
        perror("gguf_open: mmap");
        close(gguf->fd); gguf->fd = -1;
        return -1;
    }
    gguf->mmap_base = (const uint8_t*)p;

    GgufCur c = { gguf->mmap_base, gguf->mmap_base + gguf->mmap_size };

    if ((size_t)(c.end - c.p) < 4 || memcmp(c.p, "GGUF", 4) != 0) {
        fprintf(stderr, "gguf_open: bad magic\n");
        munmap(p, gguf->mmap_size); close(gguf->fd);
        return -1;
    }
    c.p += 4;

    if (gguf_take_u32(&c, &gguf->version) != 0) return -1;
    if (gguf->version != 3) {
        fprintf(stderr, "gguf_open: unsupported version %u (only v3)\n", gguf->version);
        munmap(p, gguf->mmap_size); close(gguf->fd);
        return -1;
    }
    if (gguf_take_u64(&c, &gguf->n_tensors) != 0) return -1;
    if (gguf_take_u64(&c, &gguf->n_kv) != 0) return -1;

    gguf->kv = (GgufKV*)calloc(gguf->n_kv, sizeof(GgufKV));
    if (!gguf->kv) return -1;
    gguf->alignment = 32;
    for (uint64_t i = 0; i < gguf->n_kv; i++) {
        GgufKV* kv = &gguf->kv[i];
        kv->key = gguf_take_string_dup(&c);
        if (!kv->key) return -1;
        if (gguf_take_u32(&c, &kv->vtype) != 0) return -1;
        kv->bytes = c.p;
        if (gguf_skip_value(&c, kv->vtype) != 0) return -1;
        kv->bytes_len = (size_t)(c.p - kv->bytes);

        if (strcmp(kv->key, "general.alignment") == 0 && kv->vtype == GGUF_VAL_UINT32) {
            uint32_t a;
            memcpy(&a, kv->bytes, 4);
            gguf->alignment = a;
        }
    }

    gguf->tensors = (GgufTensor*)calloc(gguf->n_tensors, sizeof(GgufTensor));
    if (!gguf->tensors) return -1;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        GgufTensor* t = &gguf->tensors[i];
        t->name = gguf_take_string_dup(&c);
        if (!t->name) return -1;
        if (gguf_take_u32(&c, &t->n_dims) != 0) return -1;
        if (t->n_dims > 8) {
            fprintf(stderr, "gguf_open: tensor %s has %u dims (max 8)\n", t->name, t->n_dims);
            return -1;
        }
        t->nelem = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (gguf_take_u64(&c, &t->dims[d]) != 0) return -1;
            t->nelem *= (int64_t)t->dims[d];
        }
        if (gguf_take_u32(&c, &t->type) != 0) return -1;
        if (gguf_take_u64(&c, &t->offset) != 0) return -1;
        t->nbytes = gguf_tensor_bytes((GgmlType)t->type, t->nelem);
    }

    uint64_t after_index = (uint64_t)(c.p - gguf->mmap_base);
    uint64_t aligned = (after_index + gguf->alignment - 1) & ~(gguf->alignment - 1);
    gguf->body_offset = aligned;

    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        gguf->tensors[i].offset += gguf->body_offset;
    }

    return 0;
}

static void gguf_close(Gguf* gguf) {
    if (gguf->mmap_base) {
        /* Try MemX munmap first; falls back to system munmap if not MemX-owned */
        if (stratum_memx_weight_munmap((void*)gguf->mmap_base, gguf->mmap_size) != 0) {
            munmap((void*)gguf->mmap_base, gguf->mmap_size);
        }
    }
    if (gguf->fd >= 0) close(gguf->fd);
    if (gguf->kv) {
        for (uint64_t i = 0; i < gguf->n_kv; i++) free(gguf->kv[i].key);
        free(gguf->kv);
    }
    if (gguf->tensors) {
        for (uint64_t i = 0; i < gguf->n_tensors; i++) free(gguf->tensors[i].name);
        free(gguf->tensors);
    }
    memset(gguf, 0, sizeof(*gguf));
}

static const GgufKV* gguf_find_kv(const Gguf* gguf, const char* key) {
    for (uint64_t i = 0; i < gguf->n_kv; i++) {
        if (strcmp(gguf->kv[i].key, key) == 0) return &gguf->kv[i];
    }
    return NULL;
}

static const GgufTensor* gguf_find_tensor(const Gguf* gguf, const char* name) {
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name, name) == 0) return &gguf->tensors[i];
    }
    return NULL;
}

static int gguf_get_u32(const Gguf* gguf, const char* key, uint32_t* out) {
    const GgufKV* kv = gguf_find_kv(gguf, key);
    if (!kv) return -1;
    if (kv->vtype == GGUF_VAL_UINT32 || kv->vtype == GGUF_VAL_INT32) {
        memcpy(out, kv->bytes, 4); return 0;
    }
    if (kv->vtype == GGUF_VAL_UINT64 || kv->vtype == GGUF_VAL_INT64) {
        uint64_t v; memcpy(&v, kv->bytes, 8); *out = (uint32_t)v; return 0;
    }
    return -1;
}

static int gguf_get_u64(const Gguf* gguf, const char* key, uint64_t* out) {
    const GgufKV* kv = gguf_find_kv(gguf, key);
    if (!kv) return -1;
    if (kv->vtype == GGUF_VAL_UINT64 || kv->vtype == GGUF_VAL_INT64) {
        memcpy(out, kv->bytes, 8); return 0;
    }
    if (kv->vtype == GGUF_VAL_UINT32 || kv->vtype == GGUF_VAL_INT32) {
        uint32_t v; memcpy(&v, kv->bytes, 4); *out = v; return 0;
    }
    return -1;
}

static int gguf_get_f32(const Gguf* gguf, const char* key, float* out) {
    const GgufKV* kv = gguf_find_kv(gguf, key);
    if (!kv) return -1;
    if (kv->vtype == GGUF_VAL_FLOAT32) { memcpy(out, kv->bytes, 4); return 0; }
    return -1;
}

static char* gguf_get_string_dup(const Gguf* gguf, const char* key) {
    const GgufKV* kv = gguf_find_kv(gguf, key);
    if (!kv || kv->vtype != GGUF_VAL_STRING) return NULL;
    uint64_t n;
    memcpy(&n, kv->bytes, 8);
    char* s = (char*)malloc(n + 1);
    if (!s) return NULL;
    memcpy(s, kv->bytes + 8, n);
    s[n] = 0;
    return s;
}

#endif
