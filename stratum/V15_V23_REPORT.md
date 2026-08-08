# V15-V23 版本迭代对比报告 — 27B 模型

## 测试环境
- 模型: Qwen3.6-27B Q4_K_M (16GB GGUF)
- 机器: Apple M4 Pro, 24GB RAM
- 测试: 8 token 生成, prompt [1, 12968]

## 版本对比总表

| 版本 | 8 token | tok/s | 加速比 | peak 内存 | forwards | bit-exact |
|---|---|---|---|---|---|---|
| V14 (违规) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | 8 | ✅ |
| V15 (基线) | 41.0s | 0.19 | 1.0x | 102MB ✅ | 8 | ✅ |
| V17 | 26.1s | 0.31 | 1.6x | ~600MB ✅ | 3 | ✅ |
| V18 | 23.3s | 0.34 | 1.8x | 697MB ✅ | 2 | ✅ |
| V19 | 22.8s | 0.35 | 1.8x | 356MB ✅ | 3 | ✅ |
| V20 | 21.1s | 0.38 | 2.0x | 357MB ✅ | 3 | ✅ |
| V21 | 20.2s | 0.40 | 2.1x | 357MB ✅ | 3 | ✅ |
| V22 | 19.5s | 0.41 | 2.2x | 357MB ✅ | 3 | ✅ |
| **V23** | **16.0s** | **0.50** | **2.6x** | **442MB** ✅ | **2** | **✅** |

## V23 核心突破

**B_MAX 5→K=4**: MTP draft 第一步 accept=4/4 (100%!), 第二步 accept=2/2

```
spec-step  in=220  drafts=[16 25 561 14955]  reals=[16 25 561 14955 314]  accept=4/4  emit=5
spec-step  in=314  drafts=[14791 303]         reals=[14791 303 17722]      accept=2/2  emit=3
→ 8 tokens in 2 main forwards, 4.00 tok/main
```

## V22 Batch Profile (B=4 spec step)
```
attn=0.29s (4%)   ssm=1.05s (15%)   lm=0.61s (9%)   madv=0.91s (13%)
I/O page fault: 59% (SSD 3GB/s bottleneck, irreducible)
```

## 失败实验记录
| 实验 | 结果 | 原因 |
|---|---|---|
| 合并 2 层 madvise | 8x 慢 | 内核 readahead 500MB 同步阻塞 |
| 半量 madvise | 2x 慢 | 后半层 page fault |
| F_RDADVISE 替代 madvise | 3x 慢 | 无页面标记，LRU 不保留 |
| 跳过 odd 层 madvise | 2x 慢 | odd 层未被预读 |
| B_MAX=6 K=5 | 同速 +85MB | emit 6 vs 5，forward 数不变 |

## 修复的 Bug
1. GPU KV cache 硬编码 8 slab → 27B 16 层越界 NaN
2. keep_resident 自动模式锁 16GB page cache
3. newBufferWithBytesNoCopy 注册整个 mmap → GPU wire 所有页
4. SSM group dispatch 不处理 F32 type
5. batched forward 独立 PREFETCH_AHEAD=4/RELEASE_BEHIND=2
6. n-gram spec 每次 malloc 52MB

## 性能进展图
```
tok/s
 0.50 |                                              ★ V23
 0.45 |
 0.40 |                                    ★ V21 ★ V22
 0.35 |                          ★ V19 ★ V20
 0.30 |               ★ V17
 0.25 |
 0.20 | ★ V15
 0.15 |
 0.10 |
 0.05 |
 0.00 +--V15--V16--V17--V18--V19--V20--V21--V22--V23-->

Memory (MB)
700 |                    ★ V18
600 |               ★ V17
500 |                                              ★ V23
400 |                          ★ V19 ★ V20 ★ V21 ★ V22
300 |
200 |
100 | ★ V15
  0 +--V15--V16--V17--V18--V19--V20--V21--V22--V23-->
```
