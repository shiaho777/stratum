# V15 对比报告 — 27B 模型低内存 GPU 安全模式

## 核心问题

V13 引入的 `keep_resident` 自动模式和 `newBufferWithBytesNoCopy` 整模型注册，会在 GPU 访问 mmap 页面时 **wire（锁定）所有物理页**，等同于变相 mlock 16GB 模型——直接干爆系统内存。

## V15 修复

| 修复项 | 说明 |
|---|---|
| 移除 `keep_resident` 自动模式 | 不再在模型 <75% RAM 时锁定 page cache |
| GPU init 不注册整个 mmap | 传 `NULL, 0`，只加载 kernel |
| Ephemeral buffer (copy 语义) | `newBufferWithBytes` 逐 tensor 拷贝，不 wire mmap 页 |
| 大 tensor 跳过 GPU | N>=50000（如 lm_head 1GB）走 CPU NEON |
| full_attn GPU 默认禁用 | ephemeral copy 在 27B 上有精度问题，待修复 |

## 27B Q4_K_M 测试结果（1 token 生成）

| 版本 | 模式 | token 时间 | 绑定内存 (anon) | peak footprint | bit-exact | 内存安全 |
|---|---|---|---|---|---|---|
| V14 | CPU + keep_resident | **4.74s** | 87MB | ~6.7GB RSS | ✅ argmax=16 | ❌ 锁 16GB page cache |
| V15 | CPU (无 GPU) | **5.13s** | 59MB | **102MB** | ✅ argmax=16 | ✅ |
| V15 | SSM GPU only | 5.81s | 93MB | 353MB | ✅ argmax=16 | ✅ |
| V15 | full_attn GPU | — | — | — | ❌ NaN | 精度问题 |

## 对比分析

### V15 vs V14 速度
- V15 CPU: 5.13s vs V14 CPU: 4.74s → **慢 8%**
- 原因：移除 keep_resident 后 page cache 被 OS 回收，需重新从 SSD 读
- 这是**正确代价**——V14 的速度是以 16GB 内存锁定换来的违规速度

### V15 SSM GPU vs CPU
- SSM GPU: 5.81s vs CPU: 5.13s → **GPU 反而慢 13%**
- 原因：SSD I/O (3GB/s) 是瓶颈，不是计算。GPU ephemeral 拷贝增加开销但不减少 I/O
- GPU 计算虽快（270GB/s），但权重仍需从 SSD 读到 page cache

### 内存安全
- V15 peak footprint = **102MB**（仅激活 + KV cache + SSM state）
- V14 RSS = **6.7GB**（16GB 模型被 wire 在物理 RAM）
- V15 完全合规：不 wire、不 mlock、不锁 page cache

## 瓶颈分析

27B 模型 5.13s/token 的时间分解：
- matmul: 1.78s (35%) — CPU NEON 从 mmap 读权重计算
- other: 3.44s (67%) — **主要是 SSD I/O page fault 等待**
- attn: 0.15s (3%) — 16 层 full attention
- ssm: 0.53s (10%) — 48 层 SSM 递归
- lm_head: 1.01s (20%) — 248320×5120 大矩阵

**核心瓶颈是 SSD I/O（3GB/s），16GB 模型每 token 需读全部权重。**

## 下一版方向（V16）

1. **后台 pread 预取**：在计算当前层时，后台线程 pread 下一层权重到 page cache
2. **层间权重复用**：如果 page cache 未被回收，第二个 token 可跳过 SSD 读
3. **修复 full_attn GPU 精度**：排查 ephemeral copy 在 full_attn 层的精度问题
4. **GPU 计算重叠 I/O**：GPU 计算当前 tensor 时，CPU 预读下一个 tensor
