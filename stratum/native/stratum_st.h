
#ifndef STRATUM_ST_H
#define STRATUM_ST_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <time.h>
#include <math.h>

#include <Accelerate/Accelerate.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

static inline double st_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static inline long st_peak_rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return (long)(ru.ru_maxrss / 1024);
#else
    return (long)ru.ru_maxrss;
#endif
}

#define ST_MAX_DIMS  8
#define ST_MAX_NAME  192

typedef struct {
    char     name[ST_MAX_NAME];
    char     dtype[8];
    int      ndim;
    int64_t  shape[ST_MAX_DIMS];
    int64_t  data_start;
    int64_t  data_end;
    int64_t  abs_offset;
    int64_t  nbytes;
} TensorMeta;

typedef struct {
    int          count;
    int          cap;
    TensorMeta  *items;
} TensorIndex;

static int st_index_init(TensorIndex *ix, int cap) {
    ix->count = 0;
    ix->cap   = cap;
    ix->items = (TensorMeta*)calloc(cap, sizeof(TensorMeta));
    return ix->items ? 0 : -1;
}

static void st_index_free(TensorIndex *ix) {
    free(ix->items);
    ix->items = NULL;
    ix->count = ix->cap = 0;
}

static TensorMeta *st_index_lookup(const TensorIndex *ix, const char *name) {
    for (int i = 0; i < ix->count; i++) {
        if (strcmp(ix->items[i].name, name) == 0) return &ix->items[i];
    }
    return NULL;
}

typedef struct { const char *p, *end; } StCur;

static void st_skip_ws(StCur *c) {
    while (c->p < c->end) {
        char ch = *c->p;
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') c->p++;
        else break;
    }
}

static bool st_expect_char(StCur *c, char want) {
    st_skip_ws(c);
    if (c->p < c->end && *c->p == want) { c->p++; return true; }
    return false;
}

static bool st_peek_char(StCur *c, char want) {
    st_skip_ws(c);
    return (c->p < c->end && *c->p == want);
}

static bool st_parse_string(StCur *c, char *out, size_t out_cap) {
    st_skip_ws(c);
    if (c->p >= c->end || *c->p != '"') return false;
    c->p++;
    size_t i = 0;
    while (c->p < c->end && *c->p != '"') {
        char ch = *c->p++;
        if (ch == '\\' && c->p < c->end) {
            char esc = *c->p++;
            switch (esc) {
                case '"':  ch = '"';  break;
                case '\\': ch = '\\'; break;
                case '/':  ch = '/';  break;
                case 'n':  ch = '\n'; break;
                case 't':  ch = '\t'; break;
                case 'r':  ch = '\r'; break;
                case 'b':  ch = '\b'; break;
                case 'f':  ch = '\f'; break;
                default:   ch = esc;  break;
            }
        }
        if (i + 1 < out_cap) out[i++] = ch;
    }
    if (i < out_cap) out[i] = 0;
    if (c->p >= c->end || *c->p != '"') return false;
    c->p++;
    return true;
}

static bool st_parse_int64(StCur *c, int64_t *out) {
    st_skip_ws(c);
    int sign = 1;
    if (c->p < c->end && *c->p == '-') { sign = -1; c->p++; }
    if (c->p >= c->end || *c->p < '0' || *c->p > '9') return false;
    int64_t v = 0;
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
        v = v * 10 + (*c->p - '0');
        c->p++;
    }
    *out = sign * v;
    return true;
}

static bool st_parse_int_array(StCur *c, int64_t *out, int max, int *count_out) {
    if (!st_expect_char(c, '[')) return false;
    int n = 0;
    st_skip_ws(c);
    if (c->p < c->end && *c->p == ']') { c->p++; *count_out = 0; return true; }
    while (1) {
        int64_t v;
        if (!st_parse_int64(c, &v)) return false;
        if (n < max) out[n++] = v;
        st_skip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == ']') { c->p++; break; }
        return false;
    }
    *count_out = n;
    return true;
}

static bool st_skip_value(StCur *c) {
    st_skip_ws(c);
    if (c->p >= c->end) return false;
    char ch = *c->p;
    if (ch == '"') {
        char tmp[256];
        return st_parse_string(c, tmp, sizeof tmp);
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
        int64_t v; return st_parse_int64(c, &v);
    }
    if (ch == 't' || ch == 'f' || ch == 'n') {
        while (c->p < c->end && (*c->p >= 'a' && *c->p <= 'z')) c->p++;
        return true;
    }
    if (ch == '[' || ch == '{') {
        char open = ch, close = (ch == '[') ? ']' : '}';
        c->p++;
        int depth = 1;
        bool in_str = false, esc = false;
        while (c->p < c->end && depth > 0) {
            char x = *c->p++;
            if (esc) { esc = false; continue; }
            if (in_str) {
                if (x == '\\') esc = true;
                else if (x == '"') in_str = false;
                continue;
            }
            if (x == '"')        in_str = true;
            else if (x == open)  depth++;
            else if (x == close) depth--;
        }
        return depth == 0;
    }
    return false;
}

static bool st_parse_tensor_value(StCur *c, TensorMeta *m) {
    if (!st_expect_char(c, '{')) return false;
    m->ndim = 0; m->dtype[0] = 0;
    m->data_start = m->data_end = -1;
    while (1) {
        st_skip_ws(c);
        if (st_peek_char(c, '}')) { c->p++; break; }
        char key[64];
        if (!st_parse_string(c, key, sizeof key)) return false;
        if (!st_expect_char(c, ':')) return false;
        if (strcmp(key, "dtype") == 0) {
            if (!st_parse_string(c, m->dtype, sizeof m->dtype)) return false;
        } else if (strcmp(key, "shape") == 0) {
            int n = 0;
            if (!st_parse_int_array(c, m->shape, ST_MAX_DIMS, &n)) return false;
            m->ndim = n;
        } else if (strcmp(key, "data_offsets") == 0) {
            int64_t off[2]; int n = 0;
            if (!st_parse_int_array(c, off, 2, &n) || n != 2) return false;
            m->data_start = off[0];
            m->data_end   = off[1];
        } else {
            if (!st_skip_value(c)) return false;
        }
        st_skip_ws(c);
        if (st_peek_char(c, ',')) { c->p++; continue; }
        if (st_peek_char(c, '}')) { c->p++; break; }
        return false;
    }
    return true;
}

static int st_parse_header(const char *header, size_t hlen,
                           int64_t header_byte_count, TensorIndex *ix) {
    StCur c = { header, header + hlen };
    if (!st_expect_char(&c, '{')) return -1;
    while (1) {
        st_skip_ws(&c);
        if (st_peek_char(&c, '}')) { c.p++; break; }
        char key[ST_MAX_NAME];
        if (!st_parse_string(&c, key, sizeof key)) return -1;
        if (!st_expect_char(&c, ':')) return -1;
        if (strcmp(key, "__metadata__") == 0) {
            if (!st_skip_value(&c)) return -1;
        } else {
            if (ix->count >= ix->cap) {
                int newcap = ix->cap * 2;
                TensorMeta *p = (TensorMeta*)realloc(ix->items,
                                    (size_t)newcap * sizeof(TensorMeta));
                if (!p) return -1;
                memset(p + ix->cap, 0,
                       (size_t)(newcap - ix->cap) * sizeof(TensorMeta));
                ix->items = p;
                ix->cap   = newcap;
            }
            TensorMeta *m = &ix->items[ix->count];
            strncpy(m->name, key, ST_MAX_NAME - 1);
            m->name[ST_MAX_NAME - 1] = 0;
            if (!st_parse_tensor_value(&c, m)) return -1;
            if (m->data_start < 0 || m->data_end < m->data_start) return -1;
            m->abs_offset = 8 + header_byte_count + m->data_start;
            m->nbytes     = m->data_end - m->data_start;
            ix->count++;
        }
        st_skip_ws(&c);
        if (st_peek_char(&c, ',')) { c.p++; continue; }
        if (st_peek_char(&c, '}')) { c.p++; break; }
        return -1;
    }
    return 0;
}

static int st_open(const char *path, bool nocache, int *out_fd,
                   int64_t *out_file_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }
    *out_file_size = st.st_size;
#ifdef __APPLE__
    if (nocache) {
        if (fcntl(fd, F_NOCACHE, 1) < 0) perror("fcntl(F_NOCACHE)");
    }
#endif
    *out_fd = fd;
    return 0;
}

static int st_pread_full(int fd, void *buf, size_t n, off_t off) {
    char *p = (char*)buf;
    size_t left = n;
    while (left > 0) {
        ssize_t got = pread(fd, p, left, off);
        if (got < 0) {
            if (errno == EINTR) continue;
            perror("pread");
            return -1;
        }
        if (got == 0) {
            fprintf(stderr, "pread short read\n");
            return -1;
        }
        p   += got;
        off += got;
        left -= (size_t)got;
    }
    return 0;
}

static int st_open_and_index(const char *path, bool nocache,
                             int *out_fd, TensorIndex *ix) {
    int64_t file_size = 0;
    if (st_open(path, nocache, out_fd, &file_size) < 0) return -1;
    uint64_t header_len = 0;
    if (st_pread_full(*out_fd, &header_len, 8, 0) < 0) {
        close(*out_fd); return -1;
    }
    if (header_len == 0 || (int64_t)header_len > file_size - 8) {
        fprintf(stderr, "implausible header_len=%llu\n",
                (unsigned long long)header_len);
        close(*out_fd); return -1;
    }
    char *hdr = (char*)malloc((size_t)header_len + 1);
    if (!hdr) { close(*out_fd); return -1; }
    if (st_pread_full(*out_fd, hdr, header_len, 8) < 0) {
        free(hdr); close(*out_fd); return -1;
    }
    hdr[header_len] = 0;
    if (st_index_init(ix, 512) < 0) {
        free(hdr); close(*out_fd); return -1;
    }
    if (st_parse_header(hdr, (size_t)header_len, (int64_t)header_len, ix) < 0) {
        st_index_free(ix); free(hdr); close(*out_fd); return -1;
    }
    free(hdr);
    return 0;
}

static inline void st_bf16_to_fp32(const uint16_t *src, float *dst, size_t n) {
#if defined(__ARM_NEON) || defined(__aarch64__)

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vld1q_u16(src + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(v), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(v), 16);
        vst1q_u32((uint32_t*)(dst + i),     lo);
        vst1q_u32((uint32_t*)(dst + i + 4), hi);
    }
    for (; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &bits, 4);
    }
#else
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &bits, 4);
    }
#endif
}

#define ST_BLOCK_FP32_BYTES (1 * 1024 * 1024)

typedef struct {
    int64_t   abs_offset;
    int64_t   out_features;
    int64_t   in_features;
    char      dtype[8];
} StreamWeight;

static int st_resolve(const TensorIndex *ix, const char *name,
                      const char *want_dtype, int want_ndim,
                      StreamWeight *w) {
    TensorMeta *m = st_index_lookup(ix, name);
    if (!m) {
        fprintf(stderr, "tensor not found: %s\n", name);
        return -1;
    }
    if (strcmp(m->dtype, want_dtype) != 0) {
        fprintf(stderr, "%s: expected %s, got %s\n",
                name, want_dtype, m->dtype);
        return -1;
    }
    if (m->ndim != want_ndim) {
        fprintf(stderr, "%s: expected ndim=%d, got %d\n",
                name, want_ndim, m->ndim);
        return -1;
    }
    w->abs_offset   = m->abs_offset;
    w->out_features = (want_ndim >= 1) ? m->shape[0] : 0;
    w->in_features  = (want_ndim >= 2) ? m->shape[1] : 1;
    strncpy(w->dtype, m->dtype, sizeof w->dtype);
    return 0;
}

static int st_linear_stream(int fd,
                            const StreamWeight *w,
                            const float *x,
                            const float *bias,
                            float       *y,
                            bool         accumulate,
                            uint16_t    *blk_bf16,
                            float       *blk_fp32,
                            size_t       block_rows_cap)
{
    int64_t N = w->out_features;
    int64_t K = w->in_features;

    int64_t target_rows = ST_BLOCK_FP32_BYTES / (K * 4);
    if (target_rows < 1) target_rows = 1;
    if (target_rows > (int64_t)block_rows_cap) target_rows = (int64_t)block_rows_cap;
    int64_t block_rows = target_rows;

    if (!accumulate) {
        if (bias) memcpy(y, bias, (size_t)N * sizeof(float));
        else      memset(y, 0,    (size_t)N * sizeof(float));
    } else if (bias) {
        for (int64_t i = 0; i < N; i++) y[i] += bias[i];
    }

    int64_t nblocks = (N + block_rows - 1) / block_rows;
    for (int64_t b = 0; b < nblocks; b++) {
        int64_t row0 = b * block_rows;
        int64_t rows = (row0 + block_rows <= N) ? block_rows : (N - row0);
        int64_t bytes = rows * K * 2;
        int64_t off   = w->abs_offset + row0 * K * 2;
        if (st_pread_full(fd, blk_bf16, (size_t)bytes, off) < 0) return -1;
        st_bf16_to_fp32(blk_bf16, blk_fp32, (size_t)(rows * K));
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)rows, (int)K,
                    1.0f, blk_fp32, (int)K,
                    x, 1,
                    1.0f, y + row0, 1);
    }
    return 0;
}

static int st_load_bf16_vector(int fd, const StreamWeight *w,
                               float *out, uint16_t *scratch) {
    int64_t N = w->out_features;
    int64_t bytes = N * 2;
    if (st_pread_full(fd, scratch, (size_t)bytes, w->abs_offset) < 0) return -1;
    st_bf16_to_fp32(scratch, out, (size_t)N);
    return 0;
}

static int st_load_f32_vector(int fd, const StreamWeight *w, float *out) {
    int64_t N = w->out_features;
    int64_t bytes = N * 4;
    return st_pread_full(fd, out, (size_t)bytes, w->abs_offset);
}

static inline float st_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            bits = sign | ((uint32_t)(127 + e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v; v.u = bits;
    return v.f;
}

static inline void st_int4_block_to_fp32(
    const uint8_t* packed, const float* scale,
    int rows, int in_features, float* dst)
{
    int half = in_features >> 1;
    for (int r = 0; r < rows; r++) {
        const uint8_t* row = packed + (size_t)r * half;
        float* drow = dst + (size_t)r * in_features;
        float s = scale[r];
        for (int i = 0; i < half; i++) {
            uint8_t b = row[i];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            int lo_s = (lo >= 8) ? lo - 16 : lo;
            int hi_s = (hi >= 8) ? hi - 16 : hi;
            drow[2*i]     = (float)lo_s * s;
            drow[2*i + 1] = (float)hi_s * s;
        }
    }
}

static int st_linear_int4_stream(
    int fd,
    int64_t abs_offset_packed,
    int64_t N, int64_t K,
    const float* scale_f32,
    const float* x,
    float*       y,
    uint8_t*     blk_packed,
    float*       blk_fp32,
    int64_t      block_rows_cap)
{
    if (K & 1) { fprintf(stderr, "int4 K must be even\n"); return -1; }
    int64_t target_rows = ST_BLOCK_FP32_BYTES / (K * 4);
    if (target_rows < 1) target_rows = 1;
    if (target_rows > block_rows_cap) target_rows = block_rows_cap;

    memset(y, 0, (size_t)N * sizeof(float));
    int64_t row_packed_bytes = K / 2;
    int64_t nblocks = (N + target_rows - 1) / target_rows;
    for (int64_t b = 0; b < nblocks; b++) {
        int64_t row0 = b * target_rows;
        int64_t rows = (row0 + target_rows <= N) ? target_rows : (N - row0);
        int64_t bytes = rows * row_packed_bytes;
        int64_t off = abs_offset_packed + row0 * row_packed_bytes;
        if (st_pread_full(fd, blk_packed, (size_t)bytes, off) < 0) return -1;
        st_int4_block_to_fp32(blk_packed, scale_f32 + row0,
                              (int)rows, (int)K, blk_fp32);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)rows, (int)K,
                    1.0f, blk_fp32, (int)K,
                    x, 1,
                    1.0f, y + row0, 1);
    }
    return 0;
}

#endif
