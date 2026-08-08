# V15-V25 版本迭代对比报告 — 27B 模型

## 测试环境
- 模型: Qwen3.6-27B Q4_K_M (16GB GGUF)
- 机器: Apple M4 Pro, 24GB RAM
- 测试: 8 token 生成, prompt [1, 12968]

## 版本对比总表

| 版本 | 8 token | tok/s | 加速比 | peak 内存 | bit-exact |
|---|---|---|---|---|---|
| V14 (违规) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | ✅ |
| V15 (基线) | 41.0s | 0.19 | 1.0x | 102MB ✅ | ✅ |
| V17 | 26.1s | 0.31 | 1.6x | ~600MB ✅ | ✅ |
| V18 | 23.3s | 0.34 | 1.8x | 697MB ✅ | ✅ |
| V19 | 22.8s | 0.35 | 1.8x | 356MB ✅ | ✅ |
| V20 | 21.1s | 0.38 | 2.0x | 357MB ✅ | ✅ |
| V21 | 20.2s | 0.40 | 2.1x | 357MB ✅ | ✅ |
| V22 | 19.5s | 0.41 | 2.2x | 357MB ✅ | ✅ |
| V23 | 16.1s | 0.50 | 2.6x | 442MB ✅ | ✅ |
| V24 | 15.5s | 0.52 | 2.7x | 442MB ✅ | ✅ |
| **V25** | **14.1s** | **0.57** | **3.0x** | **442MB** ✅ | **✅** |

## V25 Batch Profile (B=5 spec step)
```
attn=0.19s (3%)   ssm=0.83s (12%)   lm=0.64s (9%)   madv=0.91s (13%)
I/O page fault: 63% (SSD 3GB/s bottleneck)
Total compute: 2.57s (was 3.57s in V24, 28% faster)
```

## 逐版核心改动

| 版本 | 改动 | 速度 | 内存 |
|---|---|---|---|
| V15 | 移除内存炸弹 (keep_resident + NoCopy) | 基线 | 6.7GB→102MB |
| V16 | 修复 GPU KV cache 越界 | — | — |
| V17 | n-gram spec + MTP spec decode | +63% | +498MB |
| V18 | page cache 优化 (RELEASE_BEHIND=0) | +10% | +101MB |
| V19 | B_MAX 8→4 | +3% | -341MB |
| V20 | PREFETCH_AHEAD 1→2 | +9% | 0 |
| V21 | lm_head madvise + batched madvise fix | +5% | 0 |
| V22 | conv1d dispatch chunking | +3% | 0 |
| V23 | B_MAX=5 K=4 (100% accept step 1) | +22% | +85MB |
| V24 | MTP head + lm_head prefetch | +4% | 0 |
| **V25** | **SDOT multix (int8 dotprod)** | **+10%** | **0** |

## 修复的 Bug (6个)
1. GPU KV cache 硬编码 8 slab → 27B 16层越界 NaN
2. keep_resident 自动模式锁 16GB page cache
3. newBufferWithBytesNoCopy 注册整个 mmap → GPU wire 所有页
4. SSM group dispatch 不处理 F32 type
5. batched forward 独立 PREFETCH_AHEAD/RELEASE_BEHIND 参数
6. n-gram spec 每次 malloc 52MB

## 失败实验 (7个)
| 实验 | 结果 |
|---|---|
| 合并 2 层 madvise | 8x 慢 |
| 半量 madvise | 2x 慢 |
| F_RDADVISE 替代 madvise | 3x 慢 |
| 跳过 odd 层 madvise | 2x 慢 |
| B_MAX=6 K=5 | 同速 +85MB |
| GPU ephemeral for 27B | 比 CPU 慢 |
| n-gram spec creative text | 无匹配 |

## 性能进展图
```
tok/s
 0.57 |                                                        ★ V25
 0.50 |                                          ★ V23 ★ V24
 0.45 |
 0.40 |                                    ★ V21 ★ V22
 0.35 |                          ★ V19 ★ V20
 0.30 |               ★ V17
 0.25 |
 0.20 | ★ V15
 0.15 |
 0.10 |
 0.00 +--V15--V16--V17--V18--V19--V20--V21--V22--V23--V24--V25-->
```
