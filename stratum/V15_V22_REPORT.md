# V15-V22 版本迭代对比报告 — 27B 模型

## 测试环境
- 模型: Qwen3.6-27B Q4_K_M (16GB GGUF)
- 机器: Apple M4 Pro, 24GB RAM
- 测试: 8 token 生成, prompt [1, 12968]

## 版本对比总表

| 版本 | 8 token 总时间 | tok/s | 加速比 | peak 内存 | bit-exact |
|---|---|---|---|---|---|
| V14 (违规) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | ✅ |
| V15 (基线) | 41.0s | 0.19 | 1.0x | 102MB ✅ | ✅ |
| V17 (MTP K=3) | 26.1s | 0.31 | 1.6x | ~600MB ✅ | ✅ |
| V18 (K=7 B=8) | 23.3s | 0.34 | 1.8x | 697MB ✅ | ✅ |
| V19 (B_MAX=4) | 22.8s | 0.35 | 1.8x | 356MB ✅ | ✅ |
| V20 (PF=2) | 21.1s | 0.38 | 2.0x | 357MB ✅ | ✅ |
| V21 (lm_head pf) | 20.2s | 0.40 | 2.1x | 357MB ✅ | ✅ |
| **V22 (conv1d chunk)** | **19.5s** | **0.41** | **2.2x** | **357MB** ✅ | **✅** |

## V22 Batch Profile (B=4 spec step)
```
attn=0.29s (4%)   — 16 full-attn layers
ssm=1.05s (15%)   — 48 SSM layers (conv1d + recursion)
lm=0.61s (9%)     — lm_head (1GB Q6_K)
madv=0.91s (13%)  — madvise prefetch system calls
I/O wait=59%      — SSD page fault latency (irreducible)
```

## 逐版改进

| 版本 | 改动 | 速度提升 | 内存变化 |
|---|---|---|---|
| V15 | 移除 keep_resident + ephemeral GPU | 基线 | 6.7GB→102MB |
| V16 | 修复 GPU KV cache 越界 (8→32 slab) | — | — |
| V17 | n-gram spec + MTP 验证 | +63% | +498MB |
| V18 | RELEASE_BEHIND=0, PREFETCH_AHEAD=1 | +10% | +101MB |
| V19 | B_MAX 8→4 | +3% | -341MB |
| V20 | PREFETCH_AHEAD 1→2 | +9% | 0 |
| V21 | lm_head madvise prefetch + batched madvise fix | +5% | 0 |
| V22 | conv1d dispatch chunking (10240→160 tasks) | +3% | 0 |

## 失败的实验
- 合并 2 层 madvise 为 1 次调用: 8x 慢 (内核 readahead 500MB → 同步阻塞)
- 半量 madvise (只预读前半层): 2x 慢 (后半层 page fault)
- GPU ephemeral for 27B: 比 CPU 慢 (copy 开销 > 计算节省, SSD I/O 不变)
- n-gram spec on creative text: 无匹配 (需要重复文本)

## 修复的 Bug
1. GPU KV cache 硬编码 8 slab, 27B 需要 16 → 越界 NaN
2. keep_resident 自动模式锁 16GB page cache = 变相 mlock
3. newBufferWithBytesNoCopy 注册整个 mmap → GPU wire 所有物理页
4. SSM group dispatch 不处理 F32 type → GPU fallback 失败
5. batched forward 独立的 PREFETCH_AHEAD=4/RELEASE_BEHIND=2 (未用 V18 优化值)
6. n-gram spec 每次 malloc 52MB → 改为预分配复用
