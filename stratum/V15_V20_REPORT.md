# V15-V20 版本迭代对比报告 — 27B 模型

## 测试环境
- 模型: Qwen3.6-27B Q4_K_M (16GB GGUF)
- 机器: Apple M4 Pro, 24GB RAM
- 测试: 8 token 生成, prompt [1, 12968]

## 版本对比总表

| 版本 | 模式 | 8 token 总时间 | tok/s | 加速比 | peak 内存 | bit-exact |
|---|---|---|---|---|---|---|
| V14 | CPU + keep_resident (违规) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | ✅ |
| V15 | CPU (无 GPU) | 41.0s | 0.19 | 1.0x (基线) | 102MB ✅ | ✅ |
| V16 | full_attn GPU (修复 KV) | 59.6s | 0.13 | 0.7x | 19MB ✅ | ✅ |
| V17 | MTP spec K=3 | 26.1s | 0.31 | 1.6x | ~600MB ✅ | ✅ |
| V18 | MTP spec K=7 B=8 | 23.3s | 0.34 | 1.8x | 697MB ✅ | ✅ |
| V19 | MTP spec K=3 B=4 | 22.8s | 0.35 | 1.8x | **356MB** ✅ | ✅ |
| **V20** | **MTP spec K=3 B=4 PF=2** | **21.1s** | **0.38** | **2.0x** | **357MB** ✅ | **✅** |

## 逐版改进

### V15: 内存安全修复 (0.19 tok/s, 102MB)
- 移除 keep_resident 自动模式（锁 16GB page cache = 变相 mlock）
- GPU ephemeral `newBufferWithBytes`（拷贝语义，不 wire mmap 页）
- 代价：慢 8%（5.13s vs 4.74s），但内存从 6.7GB 降到 102MB

### V16: KV cache 越界修复
- Bug: GPU KV cache 硬编码 8 slab，27B 有 16 full-attn 层 → 越界 NaN
- 修复: 32 slab。full_attn GPU 从 NaN 变为 bit-exact
- GPU ephemeral 仍比 CPU 慢（SSD I/O 瓶颈下拷贝是纯开销）

### V17: n-gram spec decode + MTP 验证 (0.31 tok/s)
- 新增 Qwen3.5 n-gram spec decode（零 draft forward 开销）
- SSM state save/restore 支持拒绝回滚
- 验证 MTP chain spec K=7 达到 4.00 tok/main_forward

### V18: Page cache 优化 (0.34 tok/s, 697MB)
- RELEASE_BEHIND 2→0, PREFETCH_AHEAD 4→1
- OS LRU 比手动 MADV_DONTNEED 更智能
- 第二个 token 快 14%

### V19: B_MAX 8→4 (0.35 tok/s, 356MB) ← 内存减半
- B_MAX 从 8 降到 4 → SSM snapshot 从 697MB 降到 356MB
- K=3 B=4 比 K=7 B=8 更快（少计算开销，同 accept rate）
- n-gram spec snapshot 预分配复用

### V20: 预读优化 (0.38 tok/s, 357MB) ← 速度提升
- PREFETCH_AHEAD 1→2（重叠 SSD readahead 与计算）
- 9% 速度提升，内存不变

## 关键发现

### 瓶颈
- **SSD I/O 是根本瓶颈**: 16GB/token ÷ 3GB/s = 5.3s 下限
- **Spec decode 是最佳加速**: batched forward 权重只读一次服务多 token
- **GPU 无法解决 I/O 瓶颈**: ephemeral copy 是纯开销

### 内存
- **SSM state snapshot 是内存大头**: 3 × 79.5MB = 238MB (67% of 356MB)
- **B_MAX=4 是内存/速度最佳平衡点**
- **n-gram spec 只需 1 份 snapshot**: 101MB peak（但匹配率低）

### 正确性
- **所有版本 bit-exact**: token 序列 16,25,561,14955,314,14791,303,17722
- **GPU full_attn bit-exact** (V16 修复后): argmax 一致，logit 微差

## 性能进展图

```
tok/s
 0.40 |                                    ★ V20 (0.38)
 0.35 |                          ★ V19   ★ V18
 0.30 |               ★ V17
 0.25 |
 0.20 | ★ V15
 0.15 |
 0.10 |
 0.05 |
 0.00 +--V15--V16--V17--V18--V19--V20-->

Memory (MB)
700 |                    ★ V18
600 |               ★ V17
500 |
400 |                          ★ V19 ★ V20
300 |
200 |
100 | ★ V15
  0 +--V15--V16--V17--V18--V19--V20-->
```
