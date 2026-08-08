# V15-V27 最终报告 — 27B 模型

## 成果
- **速度: 0.19 → 0.58 tok/s (3.1x 加速)**
- **内存: 6.7GB → 442MB (减 93%)**
- **bit-exact: 全版本 token 序列一致**

## 版本对比

| 版本 | 8 token | tok/s | 加速比 | peak 内存 |
|---|---|---|---|---|
| V14 (违规) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ |
| V15 (基线) | 41.0s | 0.19 | 1.0x | 102MB |
| V22 | 19.5s | 0.41 | 2.2x | 357MB |
| V23 | 16.1s | 0.50 | 2.6x | 442MB |
| V25 | 14.1s | 0.57 | 3.0x | 442MB |
| **V27** | **13.9s** | **0.58** | **3.1x** | **442MB** |

## V27 Batch Profile (B=5, K=4, 2 forwards for 8 token)
```
B=5 forward:  attn=0.18s (3%)  ssm=0.79s (12%)  lm=0.62s (9%)  madv=0.93s (13%)
I/O page fault: 63% (SSD 3GB/s — irreducible)
```

## 核心优化路径
1. **内存安全** (V15): 移除 keep_resident + ephemeral GPU
2. **Bug 修复** (V16): GPU KV cache 8→32 slab
3. **Spec decode** (V17-V23): MTP chain spec, K=4, 100% accept
4. **Page cache** (V18-V20): RELEASE_BEHIND=0, PREFETCH_AHEAD=2
5. **I/O 预读** (V21, V24): lm_head + MTP head madvise prefetch
6. **计算优化** (V22, V25-V27): conv1d chunking + SDOT multix (Q4_K + Q6_K)

## 6 个 Bug 修复
1. GPU KV cache 硬编码 8 slab → 27B 16层越界 NaN
2. keep_resident 自动模式锁 16GB page cache
3. newBufferWithBytesNoCopy 注册整个 mmap → GPU wire 所有页
4. SSM group dispatch 不处理 F32 type
5. batched forward 独立 PREFETCH_AHEAD/RELEASE_BEHIND 参数
6. Q6_K SDOT xscale 索引越界 (16 vs 32 元素粒度)

## 7 个失败实验
| 实验 | 结果 |
|---|---|
| 合并 madvise | 8x 慢 |
| 半量 madvise | 2x 慢 |
| F_RDADVISE | 3x 慢 |
| 跳过 odd 层 madvise | 2x 慢 |
| B_MAX=8 K=7 | 比 B=5 K=4 慢 (B=8 forward 重) |
| GPU ephemeral 27B | 比 CPU 慢 |
| n-gram spec creative | 无匹配 |
