// gguf_nib_convert.c — V55 完整 GGUF 转换器：Q2_K tensor 数据改为 nibble 布局
// (148B/block, x 位置序, 数值不变), type 字段改为自定义 42 (Q2K_NIB)。
// 其他 tensor 原样。输出单一 GGUF, 无 sidecar (24GB 机器可整体热 cache)。
// 用法: gguf_nib_convert <in.gguf> <out.gguf>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "stratum_gguf.h"

#define GGML_TYPE_Q2K_NIB 42

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <in.gguf> <out.gguf>\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) { fprintf(stderr, "gguf_open fail\n"); return 1; }

    /* 1. 解析 header, 记录每个 tensor 描述中 type/offset 字段的文件偏移 */
    struct { uint64_t type_off, off_off; } desc[8192];
    {
        const uint8_t* p = g.mmap_base;
        uint64_t pos = 4 + 4 + 8 + 8;          /* magic+version+n_tensors+n_kv */
        fprintf(stderr, "DBG n_kv=%llu n_tensors=%llu body_off=%llu align=%llu\n",
                (unsigned long long)g.n_kv, (unsigned long long)g.n_tensors,
                (unsigned long long)g.body_offset, (unsigned long long)g.alignment);
        for (uint64_t i = 0; i < g.n_kv && i < 5; i++) {
            uint64_t len; memcpy(&len, p + pos, 8);
            fprintf(stderr, "  DBG kv[%llu] pos=%llu keylen=%llu bytes_len=%zu\n",
                    (unsigned long long)i, (unsigned long long)pos,
                    (unsigned long long)len, g.kv[i].bytes_len);
            pos += 8 + len;
            pos += 4;
            const GgufKV* kv = &g.kv[i];
            pos += kv->bytes_len;
        }
        for (uint64_t i = 5; i < g.n_kv; i++) {
            uint64_t len; memcpy(&len, p + pos, 8); pos += 8 + len;
            pos += 4;
            pos += g.kv[i].bytes_len;
        }
        for (uint64_t i = 0; i < g.n_tensors; i++) {
            const GgufTensor* t = &g.tensors[i];
            uint64_t len; memcpy(&len, p + pos, 8); pos += 8 + len;   /* name */
            uint32_t nd; memcpy(&nd, p + pos, 4); pos += 4;           /* n_dims */
            pos += 8 * nd;                                           /* dims */
            desc[i].type_off = pos; pos += 4;                        /* type */
            desc[i].off_off = pos; pos += 8;                         /* offset */
            /* GGUF v3 规范有 size 字段, 但 gguf_open 不消费它 —— 实际文件
             * offset 后紧跟下一个 name 的 len? 实测: 不跳, 与 gguf_open 一致 */
        }
        /* 对齐检查 */
    }

    /* 2. 计算新数据偏移 (相对 body 的偏移; gguf_open 会再加 body_offset) */
    uint64_t body = g.body_offset;
    uint64_t total = 0;
    uint64_t* newoff = malloc(sizeof(uint64_t) * g.n_tensors);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        newoff[i] = total;
        uint64_t sz = (GgmlType)t->type == GGML_TYPE_Q2_K
                    ? (uint64_t)(t->nelem / 256) * 148
                    : (uint64_t)t->nbytes;
        total += sz;
    }

    /* 3. 写新文件 */
    int fd = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open out"); return 1; }
    /* Output = header region (body bytes) + converted data. Sizing the file
     * to `total` alone silently drops the last `body` bytes of tensor data
     * (writes past EOF land in the mmap page rounding and never hit disk). */
    uint64_t out_size = body + total;
    if (ftruncate(fd, (off_t)out_size) != 0) { perror("ftruncate"); return 1; }
    uint8_t* out = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (out == MAP_FAILED) { perror("mmap out"); return 1; }

    /* header: 复制原 header, 原位改 Q2K 的 type(42) 和 offset */
    memcpy(out, g.mmap_base, body);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        if ((GgmlType)t->type == GGML_TYPE_Q2_K) {
            uint32_t nt = GGML_TYPE_Q2K_NIB;
            memcpy(out + desc[i].type_off, &nt, 4);
        }
        memcpy(out + desc[i].off_off, &newoff[i], 8);
    }

    /* 数据区: Q2K 转 nib, 其他原样 */
    uint64_t wpos = body;
    uint64_t n_q2k = 0, bad = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        const uint8_t* src = g.mmap_base + t->offset;
        if ((GgmlType)t->type == GGML_TYPE_Q2_K) {
            int64_t nblk = t->nelem / 256;
            for (int64_t b = 0; b < nblk; b++) {
                const uint8_t* ob = src + (size_t)b * 84;
                uint8_t* nb = out + wpos + (size_t)b * 148;
                memcpy(nb, ob, 16);
                memcpy(nb + 144, ob + 80, 4);
                const uint8_t* qs = ob + 16;
                uint8_t* qn = nb + 16;
                for (int p = 0; p < 256; p++) {
                    int s2 = p >> 7, tt = p & 127;
                    int bb = tt & 31, j = tt >> 5;
                    int ii = s2 * 128 + 4 * bb + j;
                    uint8_t w = (uint8_t)((qs[ii >> 2] >> (2 * (ii & 3))) & 3);
                    int half = p & 31;
                    uint8_t* tgt = qn + (p >> 5) * 16 + (half & 15);
                    if (half < 16) *tgt = (uint8_t)((*tgt & 0xF0) | w);
                    else           *tgt = (uint8_t)((*tgt & 0x0F) | (w << 4));
                }
            }
            wpos += (uint64_t)nblk * 148;
            n_q2k++;
        } else {
            memcpy(out + wpos, src, (size_t)t->nbytes);
            wpos += (uint64_t)t->nbytes;
        }
    }
    msync(out, out_size, MS_SYNC);
    munmap(out, out_size);
    close(fd);
    printf("wrote %s: %.2f GB (%llu Q2K->Q2K_NIB, total %llu tensors)\n",
           argv[2], (double)out_size / 1e9, (unsigned long long)n_q2k,
           (unsigned long long)g.n_tensors);
    return bad ? 2 : 0;
}
