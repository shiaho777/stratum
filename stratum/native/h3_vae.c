/*
 * h3_vae.c — H3 pipeline M5: safetensors reader + VAE decode skeleton.
 *
 * Reads the Comfy-Org safetensors VAE checkpoints:
 *   minimax_h3_video_vae_fp16.safetensors (5.2 GB)
 *   minimax_h3_audio_vae_fp32.safetensors (0.6 GB)
 *
 * safetensors format: 8-byte LE u64 header length, JSON header
 * ({"tensor_name": {"dtype": "F16", "shape": [...], "data_offsets":
 * [start, end]}, ...}, then raw data. Reader mmaps the file and builds
 * a name -> (offset, dtype, shape) index with zero copies.
 *
 * This first stage: open both VAEs, inventory the tensor names and
 * shapes (they reveal the decoder architecture: conv layers, mid/res
 * blocks, upsamplers), and verify finite stats on a sample. The decode
 * graph itself (latent 24ch -> RGB frames; 32ch -> stereo PCM) is
 * wired next once the block inventory is mapped to the reference
 * AutoencoderKL-like structure in Comfy-Org's VAE config.
 */
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char name[256];
    char dtype[16];
    uint64_t start, end;
    uint32_t ndim;
    uint64_t dims[8];
} SEntry;

typedef struct {
    int fd;
    const uint8_t* base;
    size_t size;
    SEntry* entries;
    size_t n;
} SFile;

static const char* DT_NAME[] = { "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3",
    "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "F64" };
static const size_t DT_SIZE[] = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 8 };

static int sopen_(const char* path, SFile* sf) {
    sf->fd = open(path, O_RDONLY);
    if (sf->fd < 0) return -1;
    struct stat st;
    if (fstat(sf->fd, &st) != 0) return -1;
    sf->size = (size_t)st.st_size;
    sf->base = mmap(NULL, sf->size, PROT_READ, MAP_SHARED, sf->fd, 0);
    if (sf->base == MAP_FAILED) return -1;
    uint64_t hlen;
    memcpy(&hlen, sf->base, 8);
    /* parse JSON header: minimal scanner for name/dtype/shape/offsets */
    size_t cap = 64;
    sf->entries = malloc(sizeof(SEntry) * cap);
    sf->n = 0;
    const char* p = (const char*)sf->base + 8;
    const char* end = p + hlen;
    /* walk JSON: crude but sufficient for safetensors structure */
    while (p < end) {
        const char* q = memchr(p, '"', (size_t)(end - p));
        if (!q) break;
        const char* name0 = q + 1;
        const char* name1 = memchr(name0, '"', (size_t)(end - name0));
        if (!name1) break;
        size_t nlen = (size_t)(name1 - name0);
        const char* obj = memchr(name1, '{', (size_t)(end - name1));
        const char* objend = memchr(name1, '}', (size_t)(end - name1));
        if (!obj || !objend || obj > objend) { p = name1 + 1; continue; }
        if (sf->n >= cap) {
            cap *= 2;
            sf->entries = realloc(sf->entries, sizeof(SEntry) * cap);
        }
        SEntry* e = &sf->entries[sf->n];
        size_t cpy = nlen < sizeof(e->name) - 1 ? nlen : sizeof(e->name) - 1;
        memcpy(e->name, name0, cpy); e->name[cpy] = 0;
        /* dtype */
        const char* dt = strstr(obj, "\"dtype\"");
        if (dt && dt < objend) {
            dt = memchr(dt, ':', (size_t)(objend - dt));
            if (dt) {
                while (dt < objend && (*dt == ':' || *dt == ' ')) dt++;
                if (dt < objend && *dt == '"') dt++;
            }
        }
        if (dt && dt < objend) {
            const char* dte = memchr(dt, '"', (size_t)(objend - dt));
            size_t dl = (size_t)(dte - dt);
            if (dl > 15) dl = 15;
            memcpy(e->dtype, dt, dl); e->dtype[dl] = 0;
        } else e->dtype[0] = 0;
        /* shape */
        e->ndim = 0;
        const char* sh = strstr(obj, "\"shape\"");
        if (sh && sh < objend) {
            sh = memchr(sh, '[', (size_t)(objend - sh));
            if (!sh || sh >= objend) sh = objend - 1; else sh++;
            while (sh < objend && *sh != ']' && e->ndim < 8) {
                char* e2;
                unsigned long long v = strtoull(sh, &e2, 10);
                if (e2 == sh) break;
                e->dims[e->ndim++] = v;
                sh = e2;
                while (sh < objend && (*sh == ',' || *sh == ' ')) sh++;
            }
        }
        /* data_offsets */
        e->start = e->end = 0;
        const char* of = strstr(obj, "\"data_offsets\"");
        if (of && of < objend) {
            of = memchr(of, '[', (size_t)(objend - of));
            if (!of || of >= objend) of = objend - 1; else of++;
            char* e2;
            e->start = strtoull(of, &e2, 10);
            of = e2;
            while (of < objend && (*of == ',' || *of == ' ')) of++;
            e->end = strtoull(of, NULL, 10);
        }
        sf->n++;
        p = objend + 1;
    }
    return 0;
}

static float f16v(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static float bfv(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

static void stat_tensor(const SFile* sf, const SEntry* e) {
    size_t dti = 0;
    for (size_t i = 0; i < 13; i++)
        if (!strcmp(e->dtype, DT_NAME[i])) { dti = i; break; }
    size_t esz = DT_SIZE[dti];
    size_t n = (e->end - e->start) / (esz ? esz : 1);
    const uint8_t* d = sf->base + 8 + /* hdr */ 0;
    /* data begins after header: base+8+hlen aligned to 8 */
    uint64_t hlen; memcpy(&hlen, sf->base, 8);
    d = sf->base + 8 + hlen + e->start;
    float mn = 0, mx = 0; double sm = 0; long nan = 0;
    size_t step = n > 4096 ? n / 4096 : 1;
    int first = 1;
    for (size_t i = 0; i < n; i += step) {
        float v = 0;
        if (!strcmp(e->dtype, "F32")) memcpy(&v, d + i * 4, 4);
        else if (!strcmp(e->dtype, "F16")) v = f16v(*(const uint16_t*)(d + i * 2));
        else if (!strcmp(e->dtype, "BF16")) v = bfv(*(const uint16_t*)(d + i * 2));
        if (isnan(v) || isinf(v)) { nan++; continue; }
        if (first) { mn = mx = v; first = 0; }
        if (v < mn) mn = v; if (v > mx) mx = v;
        sm += v;
    }
    printf("  %-64s %-6s [", e->name, e->dtype);
    for (uint32_t i = 0; i < e->ndim; i++)
        printf("%llu%s", (unsigned long long)e->dims[i],
               i + 1 < e->ndim ? "," : "");
    printf("] n=%zu sampled: mean=%.4g [%+.4g, %+.4g]%s\n",
           n, nan ? 0.0 : sm / (double)(n / step ? n / step : 1), mn, mx,
           nan ? " NAN!" : "");
}

int main(int argc, char** argv) {
    for (int a = 1; a < argc; a++) {
        SFile sf;
        if (sopen_(argv[a], &sf) != 0) {
            printf("%s: open failed\n", argv[a]);
            continue;
        }
        printf("== %s: %zu tensors ==\n", argv[a], sf.n);
        for (size_t i = 0; i < sf.n; i++)
            stat_tensor(&sf, &sf.entries[i]);
        munmap((void*)sf.base, sf.size);
        close(sf.fd);
    }
    return 0;
}
