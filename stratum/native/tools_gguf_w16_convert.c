// tools_gguf_w16_convert.c — Q4_K -> Q4K_W16 (type 43) GGUF converter.
// w16 stores the exact integer n*sc (nibble * 6-bit scale) in i16 lanes,
// k-major within 32-row tiles: this is the value the Q4_K dequant formula
// already computes (d*(sc*n) - dm*m), NOT a requantization — same class of
// transform as the type-42 nibble layout (values unchanged).
//
// Per-tensor blob layout (aligned to 32B):
//   [0              : N*K*2)   w16 tiles: w16[rt][k][32] i16, rt in [0,N/32)
//   [+N*K*2         : +4N )    d[N]   f32  (block super-scale)
//   [+4N            : +4N )    dm[N]  f32  (block super-min scale)
//   [+8N            : +2N*NG)  m16[rt][g][32] i16 = m[r][g] tile-transposed —
//                              the minterm runs on MATINT (i16, 4x MATFP
//                              density) with i16 activation group sums
//
// Requires N % 32 == 0 and K % 256 == 0 (inherent to Q4_K).
// Usage: gguf_w16_convert <in.gguf> <out.gguf>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "stratum_gguf.h"

#define GGML_TYPE_Q4K_W16 43

typedef struct { uint8_t d[2], dm[2]; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;

static inline void get_scale_min(int j, const uint8_t* s, int* sc, int* m) {
    if (j < 4) { *sc = s[j] & 63; *m = s[j+4] & 63; }
    else { *sc = (s[j+4] & 0x0F) | ((s[j-4] >> 6) << 4);
           *m  = (s[j+4] >> 4)   | ((s[j]   >> 6) << 4); }
}
static float h2f(uint16_t h) {
    uint32_t e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) { if (!m) f = 0; else { while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; e++; f = ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) f = 0x7F800000u | (m << 13);
    else f = ((e + 112) << 23) | (m << 13);
    if (h >> 15) f |= 0x80000000u;
    float o; memcpy(&o, &f, 4); return o;
}

static uint64_t w16_blob_bytes(int64_t N, int64_t K) {
    return (uint64_t)N * K * 2 + (uint64_t)N * 8 + (uint64_t)N * (K / 32) * 2;
}

/* convert one tensor: src = packed Q4_K rows (N rows, K cols, nb=K/256 blocks) */
static void conv_tensor(const uint8_t* src, int64_t N, int64_t K, uint8_t* dst) {
    int64_t nb = K / 256, NG = K / 32, NT = N / 32;
    int16_t* w16 = (int16_t*)dst;
    float* dv = (float*)(dst + (size_t)N * K * 2);
    float* dmv = dv + N;
    int16_t* m16 = (int16_t*)(dmv + N);   /* [rt][g][32] i16 = m[rt*32+j][g] */
    (void)NT;
    for (int64_t r = 0; r < N; r++) {
        int64_t rt = r / 32, ri = r % 32;
        for (int64_t i = 0; i < nb; i++) {
            const block_q4_K* b = (const block_q4_K*)(src + (size_t)r * nb * 144 + i * 144);
            dv[r] = h2f(*(const uint16_t*)b->d);
            dmv[r] = h2f(*(const uint16_t*)b->dm);
            for (int js = 0; js < 256; js += 64) {
                int is_ = js / 32, sc1, m1, sc2, m2;
                get_scale_min(is_,     b->scales, &sc1, &m1);
                get_scale_min(is_ + 1, b->scales, &sc2, &m2);
                int g0 = i * 8 + is_, g1 = g0 + 1;
                m16[(size_t)rt * NG * 32 + (size_t)g0 * 32 + ri] = (int16_t)m1;
                m16[(size_t)rt * NG * 32 + (size_t)g1 * 32 + ri] = (int16_t)m2;
                const uint8_t* q = b->qs + (js / 64) * 32;
                int kbase = i * 256 + js;
                for (int j = 0; j < 32; j++) {
                    w16[(size_t)rt * K * 32 + (size_t)(kbase + j) * 32 + ri]      = (int16_t)((q[j] & 0xF) * sc1);
                    w16[(size_t)rt * K * 32 + (size_t)(kbase + j + 32) * 32 + ri] = (int16_t)((q[j] >> 4) * sc2);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <in.gguf> <out.gguf>\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) { fprintf(stderr, "gguf_open fail\n"); return 1; }

    /* locate type/offset field file positions (same walk as nib converter) */
    struct { uint64_t type_off, off_off; } desc[8192];
    {
        const uint8_t* p = g.mmap_base;
        uint64_t pos = 4 + 4 + 8 + 8;
        for (uint64_t i = 0; i < g.n_kv; i++) {
            uint64_t len; memcpy(&len, p + pos, 8); pos += 8 + len;
            pos += 4;
            pos += g.kv[i].bytes_len;
        }
        for (uint64_t i = 0; i < g.n_tensors; i++) {
            uint64_t len; memcpy(&len, p + pos, 8); pos += 8 + len;
            uint32_t nd; memcpy(&nd, p + pos, 4); pos += 4;
            pos += 8 * nd;
            desc[i].type_off = pos; pos += 4;
            desc[i].off_off = pos; pos += 8;
        }
    }

    /* new offsets: converted tensors need w16 blob size */
    uint64_t body = g.body_offset, total = 0;
    uint64_t* newoff = malloc(sizeof(uint64_t) * g.n_tensors);
    int skipped = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        newoff[i] = total;
        /* token_embd is a row gather (not a matvec) and output/lm_head keeps
         * the SDOT path in v1 — both stay Q4_K. */
        int keep_q4k = t->name && (strncmp(t->name, "token_embd", 10) == 0 ||
                                   strncmp(t->name, "output", 6) == 0);
        if ((GgmlType)t->type == GGML_TYPE_Q4_K && !keep_q4k) {
            int64_t K = t->dims[0], N = t->nelem / K;   /* dims[0] = innermost = K */
            if (t->n_dims == 2 && N % 32 == 0 && K % 256 == 0)
                total += w16_blob_bytes(N, K);
            else { total += (uint64_t)t->nbytes; skipped++; }
        } else total += (uint64_t)t->nbytes;
        total = (total + 31) & ~31ull;
    }

    int fd = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open out"); return 1; }
    uint64_t out_size = body + total;
    if (ftruncate(fd, (off_t)out_size) != 0) { perror("ftruncate"); return 1; }
    uint8_t* out = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (out == MAP_FAILED) { perror("mmap out"); return 1; }

    memcpy(out, g.mmap_base, body);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        if ((GgmlType)t->type == GGML_TYPE_Q4_K) {
            int64_t K = t->dims[0], N = t->nelem / K;
            int keep_q4k = t->name && (strncmp(t->name, "token_embd", 10) == 0 ||
                                       strncmp(t->name, "output", 6) == 0);
            if (!keep_q4k && t->n_dims == 2 && N % 32 == 0 && K % 256 == 0) {
                uint32_t nt = GGML_TYPE_Q4K_W16;
                memcpy(out + desc[i].type_off, &nt, 4);
            }
        }
        memcpy(out + desc[i].off_off, &newoff[i], 8);
    }

    uint64_t wpos = body, n_conv = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        const uint8_t* src = g.mmap_base + t->offset;
        if ((GgmlType)t->type == GGML_TYPE_Q4_K) {
            int64_t K = t->dims[0], N = t->nelem / K;
            int keep_q4k = t->name && (strncmp(t->name, "token_embd", 10) == 0 ||
                                       strncmp(t->name, "output", 6) == 0);
            if (!keep_q4k && t->n_dims == 2 && N % 32 == 0 && K % 256 == 0) {
                conv_tensor(src, N, K, out + wpos);
                wpos += w16_blob_bytes(N, K);
                n_conv++;
            } else { memcpy(out + wpos, src, (size_t)t->nbytes); wpos += (uint64_t)t->nbytes; }
        } else { memcpy(out + wpos, src, (size_t)t->nbytes); wpos += (uint64_t)t->nbytes; }
        wpos = (wpos + 31) & ~31ull;
    }
    msync(out, out_size, MS_SYNC);
    munmap(out, out_size);
    close(fd);
    printf("wrote %s: %.2f GB (%llu Q4K->Q4K_W16, %d skipped, total %llu tensors)\n",
           argv[2], (double)out_size / 1e9, (unsigned long long)n_conv, skipped,
           (unsigned long long)g.n_tensors);
    return 0;
}
