# V15-V18 版本迭代对比报告 — 27B 模型

## 测试环境
- 模型: Qwen3.6-27B Q4_K_M (16GB GGUF)
- 机器: Apple M4 Pro, 24GB RAM
- 测试: 1-8 token 生成, prompt [1, 12968]

## 版本对比

| 版本 | 模式 | 单 token | 8 token 总时间 | tok/s | 加速比 | peak 内存 | bit-exact |
|---|---|---|---|---|---|---|---|
| V14 | CPU + keep_resident (违规) | 4.74s | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | ✅ |
| V15 | CPU (无 GPU) | 5.13s | 41.0s | 0.19 | 1.0x (基线) | 102MB ✅ | ✅ |
| V15 | SSM GPU only | 5.81s | 46.5s | 0.17 | 0.9x | 353MB ✅ | ✅ |
| V15 | full_attn GPU | NaN | — | — | — | — | ❌ KV cache bug |
| V16 | full_attn GPU (修复 KV) | 7.45s | 59.6s | 0.13 | 0.7x | 19MB ✅ | ✅ |
| V17 | MTP spec K=3 | — | 26.1s | 0.31 | 1.6x | ~600MB ✅ | ✅ |
| V17 | n-gram spec | 无匹配 | 41.0s | 0.19 | 1.0x | 102MB ✅ | ✅ |
| **V18** | **MTP spec K=7** | **4.93s** | **19.9s** | **0.40** | **2.1x** | **697MB ✅** | **✅** |

## 逐版改进分析

### V15: 内存安全修复
- **问题**: `newBufferWithBytesNoCopy` 注册 16GB mmap → GPU wire 所有物理页 = 变相 mlock
- **修复**: ephemeral `newBufferWithBytes`（拷贝语义），不 wire mmap 页
- **代价**: 速度慢 8%（5.13s vs 4.74s），但内存从 6.7GB 降到 102MB
- **发现**: GPU ephemeral 在 SSD I/O 瓶颈下比 CPU 慢（拷贝开销 > 计算节省）

### V16: KV cache 越界修复
- **Bug**: GPU KV cache 硬编码 8 slab，27B 有 16 个 full-attn 层 → slot 8-15 越界 → NaN
- **修复**: 分配 32 slab
- **结果**: full_attn GPU 从 NaN 变为 bit-exact（argmax=16, 25）
- **发现**: GPU ephemeral 仍比 CPU 慢（7.45s vs 5.13s），但基础设施修复为未来 GPU 优化铺路

### V17: n-gram spec decode + MTP 验证
- **新增**: Qwen3.5 n-gram spec decode（`STRATUM_NGRAM_SPEC=K`）
  - 零 draft forward 开销，用 token 历史 n-gram 匹配生成 draft
  - SSM state save/restore 支持拒绝回滚
- **验证**: MTP chain spec K=7 达到 4.00 tok/main_forward
- **发现**: n-gram 在创造性文本上匹配率低（需重复文本），MTP draft 质量更高

### V18: Page cache 管理优化
- **修改**: `RELEASE_BEHIND` 2→0, `PREFETCH_AHEAD` 4→1
- **原理**: 让 OS LRU 管理 page cache 比手动 `MADV_DONTNEED` 更智能
- **效果**: 第二个 token 快 14%（4.93s vs 5.73s），spec decode 快 16%

## 瓶颈分析

### 当前瓶颈: SSD I/O
- 16GB 模型 / 3GB/s SSD = 5.3s/token（I/O 下限）
- 计算只占 35%（matmul 1.78s），I/O 等待占 65%（other 3.44s）
- GPU 无法解决 I/O 瓶颈——权重仍需从 SSD 读到 page cache

### Spec decode 是当前最佳加速手段
- B=8 batched forward 权重只读一次，服务 4+ token
- MTP draft accept rate: 71%（K=7 第一步 5/7 接受）
- 有效速度 0.40 tok/s = 2.1x 加速

## 下一版方向 (V19+)

1. **提高 MTP draft accept rate**: 更大的 draft vocab、更好的 tree 结构
2. **Batched forward 计算优化**: B=8 的 SSM 递归是串行的，可并行化
3. **Full_attn GPU 优化**: 减少 ephemeral copy 开销（chunked staging）
4. **多序列并行**: 同时跑多个独立序列，权重复用
