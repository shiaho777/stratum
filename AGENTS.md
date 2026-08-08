# AGENTS.md — Stratum 推理引擎开发指南

## 项目身份

Stratum 是一个在 Apple Silicon 上以极致低绑定内存运行 Transformer 模型的推理引擎。核心优势是 **mmap 流式架构**——绑定内存与模型大小解耦，7MB 匿名内存可运行任意大小模型。速度是次要目标，内存优势是命脉。

## 三条硬边界

以下三条边界不可违反，不可为了速度而妥协，不可在"测试"或"实验"中绕过：

### 1. 不得影响模型智商

- 禁止使用低于 Q4_K 的量化格式（Q2_K、Q3_K）来"提速"大模型
- 禁止重新量化已有权重（如 Q4_K → Q4_0 重编码）——这会引入精度损失
- 禁止跳过层计算（layer skip）来省 IO——这改变模型输出
- 禁止近似计算（如降低 softmax 精度、截断 attention）
- 唯一允许的"近似"是已有的 int8 dotprod（Q4_K SDOT），因为它已被验证为 greedy bit-exact

### 2. 不得增加绑定内存

- 绑定内存（anonymous, non-reclaimable）必须保持在 ~7MB 级别
- 禁止 mlock 整个模型或大块权重到物理 RAM
- 禁止预分配大 buffer 缓存权重（GPU staging buffer 除外，必须受控）
- 权重必须通过 mmap 从 OS page cache 流式读取，page cache 是可回收的
- 激活、KV cache、SSM state 的匿名分配是允许的（这些是模型本身需要的）

### 2a. Metal GPU 零拷贝陷阱（致命边界）

**`newBufferWithBytesNoCopy` 对 mmap 页面的 GPU 访问会 wire（锁定）物理页，使其变为 non-reclaimable。** 这等同于变相 mlock 整个模型：

- 对 16GB 模型使用 `newBufferWithBytesNoCopy` → 16GB 物理页被 wire → 内存干爆
- 即使代码不显式 mlock，GPU 首次访问 mmap 页时内核会自动 wire
- **禁止对大模型（>1GB）使用 `newBufferWithBytesNoCopy` 注册整个 mmap**
- **禁止 `keep_resident` 自动模式**（V13 引入的 auto-detect 会在模型 <75% RAM 时锁定 page cache）
- 允许的 GPU 模式：**受控 staging buffer** — 每次只拷贝当前层权重到 GPU（<100MB），计算完释放
- 允许的 GPU 模式：**逐 tensor dispatch** — 单个 tensor（<50MB）临时注册为 GPU buffer，用完释放

### 3. 测试时不得占用超过 1GB 常驻内存

- 测试任何推理引擎时，进程的常驻物理内存不得超过 1GB
- 大模型（27B）测试只允许 1-2 token 生成，不得长时间运行
- 禁止在 24GB 机器上同时运行模型 + 大量后台进程
- 测试前必须检查 `memory_pressure`，如果 free pages 不足则不得启动大模型测试
- 小模型（<1GB GGUF）是主要的测试对象

## 开发优先级

1. **正确性** — greedy decoding 必须与参考实现 bit-exact（token 序列一致）
2. **内存** — 绑定内存不随模型大小增长
3. **速度** — 在不违反 1 和 2 的前提下追求极致性能

## 速度优化方向（符合边界的）

### 允许的优化

- GPU kernel 效率优化（Q4_K 解包、SIMD、fused ops）
- 多序列批处理（batched decode，权重复用）
- 推测解码（n-gram spec、MTP）——不改变输出，只加速
- 后台 pread 预取（零额外绑定内存，page cache 可回收）
- SSM group dispatch（多 matmul 合并到一个 command buffer）
- Metal GPU **受控 staging**（逐层/tensor 拷贝权重到 GPU，<100MB/tensor，用完释放）
- Metal GPU 逐 tensor 临时 buffer（单 tensor <50MB，dispatch 后立即释放）

### 禁止的优化

- ❌ 降低量化精度（Q2_K、Q3_K、INT4 重编码）
- ❌ mlock 整模型或大块权重
- ❌ 预分配 GPU buffer 缓存所有权重（如 F16 预解码存 GPU）
- ❌ 层跳过、块跳过（改变模型输出）
- ❌ 降低 attention 精度、截断 KV cache
- ❌ `newBufferWithBytesNoCopy` 注册整个大模型 mmap（GPU 会 wire 所有物理页）
- ❌ `keep_resident` 自动模式（锁定 page cache = 变相 mlock）
- ❌ 任何在 GPU 上保留 >200MB 权重 buffer 的方案

## 架构概要

### 第四条硬边界：通用性

Stratum 针对的是**任意模型**，不是某个特定模型。以下原则不可违反：

- **禁止在 `stratum.c` 中硬编码任何模型名称或架构名称**——分发逻辑必须通过注册表 (`stratum_arch.h`) 完成
- **禁止在配置加载中使用特定模型假设**——如用 `blk.3` 而非 `blk.0` 探测张量存在性；SSM 默认值不能假设是某个模型的参数
- **新增架构只需**：创建 `stratum_arch_<name>.inc.c`，实现 `StratumArch` 接口，调用 `STRATUM_REGISTER_ARCH()` 注册，在 `stratum.c` 末尾 `#include`——**不修改任何现有文件**
- **通用基础设施（量化线性层、CPU/GPU 初始化、madvise、KV cache、spec decode）必须放在 `stratum_linear.h` / `stratum_engine.h` 中共享**，不得在各架构文件中复制粘贴
- **环境变量必须是通用的**——不得有 `STRATUM_QWEN35_XXX` 这样的模型特定变量

### C 原生引擎（`native/`）

- `stratum.c` — 主入口，通过注册表分发（**无任何硬编码架构名称**）
- `stratum_arch.h` — 通用架构注册表接口 + 通用配置加载器 (`StratumConfig`)
- `stratum_linear.h` — 通用量化线性层（Q4_K/Q5_K/Q6_K/Q3_K/Q2_K/Q8_0/F16/F32 dispatch，消除 `la_`/`q35_` 重复）
- `stratum_engine.h` — 通用引擎基础设施（CPU检测/GPU初始化/madvise/keep_resident/滑动窗口KV/spec decode/内存报告）
- `stratum_arch_qwen35.inc.c` — Qwen3.5 混合架构（Gated DeltaNet + full attention），通过注册表自注册
- `stratum_arch_llama.inc.c` — Llama 架构，通过注册表自注册
- `stratum_metal.m/.h` — Metal GPU 加速层
- `stratum_q4k.metal` — Q4_K/Q5_K/Q6_K + 自定义 kernel 的 Metal 着色器
- `stratum_q4k.h` 等 — 量化内核（NEON + scalar）

### Python 参考实现（`stratum/`）

- `adapters/qwen3_5.py` — meta device 模型构建 + mmap 权重注入
- `sparse/` — StreamingLLM KV cache、稀疏 attention、快速递归步
- `research/` — 稀疏性分析、低秩谱、预测器
- `backends/` — mmap、HTTP Range、量化后端

### 环境变量

| 变量 | 作用 | 边界状态 |
|---|---|---|
| `STRATUM_GPU=1` | 启用 GPU matmul（零拷贝 mmap） | ⚠️ 仅小模型；大模型会 wire 内存 |
| `STRATUM_GPU_FULL=1` | 整层 GPU 前向（Llama 架构） | ⚠️ 仅小模型 |
| `STRATUM_GPU_FULL=2` | Qwen3.5 GPU full-attn + SSM GPU dispatch | ⚠️ 仅小模型 |
| `STRATUM_GPU_BATCH_FULL=1` | 批量多序列 GPU 前向 | ⚠️ 仅小模型 |
| `STRATUM_MULTISEQ=N` | N 个独立序列并行 | ✅ 允许 |
| `STRATUM_NGRAM_SPEC=K` | n-gram 推测解码 | ✅ 允许 |
| `STRATUM_ASYNC_PREFETCH=1` | 后台线程 pread 预取 | ✅ 允许 |
| `STRATUM_SPARSE=0.001` | FFN down_proj 块跳过 | ⚠️ 需验证 bit-exact |
| `STRATUM_PREDECODE=1` | Q4_K→F16 预解码存 GPU | ❌ 违反内存边界 |
| `STRATUM_Q4_0=1` | Q4_K→Q4_0 重编码 | ❌ 违反智商边界 |
| `STRATUM_MLOCK_ALL=1` | mlock 整模型 | ❌ 违反内存边界 |
| `STRATUM_HOT_GB=N` | pin 前 N GB 权重 | ❌ 违反内存边界 |
| `STRATUM_KEEP_RESIDENT=1` | 锁定 page cache | ❌ 违反内存边界（V15 移除自动模式） |

## 测试规范

1. **默认用 TinyLlama 1.1B（638MB）或 Llama 3.2 3B（1.9GB）测试**
2. **27B 模型测试只跑 1 token，不得连续运行**
3. **测试前检查内存状态**：`memory_pressure` 或 `vm.swapusage`
4. **bit-exact 验证**：CPU 和 GPU 路径的 token 输出必须完全一致
5. **性能报告**：必须同时报告绑定内存（anon）和权重 cache（mapped）用量

## Git 规范

- 每个 Phase（V1-V11...）一个提交
- 提交信息包含：做了什么、为什么、bit-exact 验证结果、性能数据
- 禁止提交二进制文件（GGUF、safetensors、metallib）
