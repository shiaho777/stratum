// q2k_nibble_convert.c — Q2K 位布局重排转换器（方向 A 落地第一步）
// 读 GGUF，把 Q2_K tensor 的 qs 从"每字节4个2bit"重排为"每字节2个(4bit槽)"，
// 写入 sidecar 文件。数值序列 0-3 完全不变（非重量化）。
// sidecar 格式: [magic 8B "Q2KNIB01"][u32 n_q2k][u32 idx→off 表 n_q2k 项]
//               [tensor 数据段: 每 Q2K tensor 的 nibble block 序列]
// 用法: q2k_nibble_convert <model.gguf> <out.sidecar>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "stratum_gguf.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <out.sidecar>\n", argv[0]); return 1; }
    Gguf g;
    if (gguf_open(argv[1], &g) != 0) { fprintf(stderr, "gguf_open fail\n"); return 1; }

    /* 统计 Q2K tensor 与总数据量 */
    uint64_t n_q2k = 0, total_bytes = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++)
        if ((GgmlType)g.tensors[i].type == GGML_TYPE_Q2_K) { n_q2k++; total_bytes += g.tensors[i].nbytes * 2; }

    /* sidecar: 8 magic + 4 n + 4*n 索引 + 数据 */
    size_t hdr = 16 + 8 * (size_t)n_q2k;
    int fd = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open out"); return 1; }
    if (ftruncate(fd, (off_t)(hdr + total_bytes)) != 0) { perror("ftruncate"); return 1; }  /* 预留，稍后按实际截断 */
    uint8_t* base = mmap(NULL, hdr + total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    memcpy(base, "Q2KNIB01", 8);
    uint32_t* p = (uint32_t*)(base + 8);
    p[0] = (uint32_t)n_q2k;
    uint64_t* idx = (uint64_t*)(base + 12);   /* [n] 项, u64 (sidecar 可 >4GB) */
    uint8_t* dst = base + hdr;

    uint64_t done = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        if ((GgmlType)t->type != GGML_TYPE_Q2_K) continue;
        int64_t nblk = t->nelem / 256;               /* block 数 */
        const uint8_t* src = g.mmap_base + t->offset;
        /* 原 block: 84B = scales[16]+qs[64]+d+dmin; nibble: 16+128+2+2=148B */
        idx[done] = (uint64_t)(dst - base);
        for (int64_t b = 0; b < nblk; b++) {
            const uint8_t* ob = src + (size_t)b * 84;
            uint8_t* nb = dst + (size_t)b * 148;
            memcpy(nb, ob, 16);                       /* scales */
            memcpy(nb + 144, ob + 80, 4);             /* d(2)+dmin(2) 在尾部 */
            const uint8_t* qs = ob + 16;              /* 64B, 每字节 4 个 2bit */
            uint8_t* qn = nb + 16;                    /* 128B, 对齐 Q4K 布局 */
            /* 布局: 256 权重 = 128B, 按 X 位置序排列（Q2K 的 scale 与 x 位置绑定）:
             * 原 kernel: 权重 i=4b+j (字节b,组j) 配 x[32j+b]; scale 每 16 x 位置 1 个
             * 位置 p: 段 s=p/128, 段内 t=p%128, b=t%32, j=t/32 → 权重序号 i=s*128+4b+j
             * 每 32 位置 16B: 低 4bit = 前 16 位置, 高 4bit = 后 16 */
            for (int p = 0; p < 256; p++) {
                int s = p >> 7, t = p & 127;
                int b = t & 31, j = t >> 5;
                int i = s * 128 + 4 * b + j;
                uint8_t w = (uint8_t)((qs[i >> 2] >> (2 * (i & 3))) & 3);
                int half = p & 31;
                uint8_t* tgt = qn + (p >> 5) * 16 + (half & 15);
                if (half < 16) *tgt = (uint8_t)((*tgt & 0xF0) | w);
                else           *tgt = (uint8_t)((*tgt & 0x0F) | (w << 4));
            }
        }
        dst += (size_t)nblk * 148;
        done++;
    }
    msync(base, hdr + total_bytes, MS_SYNC);
    munmap(base, hdr + total_bytes);
    size_t used = (size_t)(dst - base);
    if (ftruncate(fd, (off_t)used) != 0) perror("ftruncate final");
    close(fd);
    printf("converted %llu Q2K tensors, 5.70 GB orig -> %.2f GB sidecar\n",
           (unsigned long long)n_q2k, (double)used / 1e9);

    /* 验证: 重开 sidecar, 解 nibble 与原 gguf 对比 */
    int vfd = open(argv[2], O_RDONLY);
    if (vfd < 0) { perror("reopen"); return 3; }
    uint8_t* rb = mmap(NULL, used, PROT_READ, MAP_PRIVATE, vfd, 0);
    if (rb == MAP_FAILED) { perror("verify mmap"); close(vfd); return 3; }
    uint32_t* rp = (uint32_t*)(rb + 8);
    uint64_t* ridx = (uint64_t*)(rb + 12);
    uint64_t bad = 0, checked = 0;
    uint64_t vi = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const GgufTensor* t = &g.tensors[i];
        if ((GgmlType)t->type != GGML_TYPE_Q2_K) continue;
        const uint8_t* src = g.mmap_base + t->offset;
        const uint8_t* nb = rb + ridx[vi++];
        int64_t nblk = t->nelem / 256;
        for (int64_t b = 0; b < nblk && checked < 200000; b++) {
            const uint8_t* qs = src + (size_t)b * 84 + 16;
            const uint8_t* qn = nb + (size_t)b * 148 + 16;
            for (int p = 0; p < 256; p++) {
                int s = p >> 7, t = p & 127;
                int b = t & 31, j = t >> 5;
                int i = s * 128 + 4 * b + j;
                uint8_t e = (uint8_t)((qs[i >> 2] >> (2 * (i & 3))) & 3);
                int half = p & 31;
                uint8_t bv = qn[(p >> 5) * 16 + (half & 15)];
                uint8_t got = (uint8_t)((half < 16) ? (bv & 0xF) : (bv >> 4));
                if (got != e) bad++;
                checked++;
            }
        }
    }
    munmap(rb, used);
    close(vfd);
    printf("verify: checked %llu weights, mismatches=%llu %s\n",
           (unsigned long long)checked, (unsigned long long)bad, bad == 0 ? "OK" : "FAIL");
    return bad == 0 ? 0 : 2;
}
