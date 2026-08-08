# Stratum

[English](README.md) | [简体中文](README_CN.md)

一个纯 C 实现的 Transformer 推理引擎，拥有所有引擎中最低的绑定内存占用：**权重是流，不是驻留物**。约 7 MB 匿名内存即可运行 27B dense 模型——因为权重通过 `mmap` 直接从操作系统页缓存读取，绑定内存与模型大小解耦。

## 为什么存在

传统引擎假设权重必须放进 RAM/VRAM。Stratum 假设相反：每个 token 都完整消费一次权重流，因此工作集与模型大小无关、恒定不变。

- **绑定内存**：约 7 MB 匿名内存（对比 llama.cpp 的数百 MB）——实测数据见下文
- **大于内存的模型**：能跑到完成，而 llama.cpp 会抖动（thrash）或 OOM——这是无人竞争的赛道
- **Bit-exact**：贪心解码与标量参考实现逐 token 完全一致
- **引擎不做量化**：原样消费 Q2_K–Q6_K / Q8_0 / F16 / F32，绝不降低精度

## 特性

- 单一二进制从 GGUF 元数据自动识别架构：**Llama 家族**（Llama 1/2/3、TinyLlama、Mistral、Qwen2-dense）与 **Qwen3.5/3.6 混合架构**（Gated DeltaNet SSM + 全注意力）
- NEON 向量化量化 kernel（Q2_K–Q6_K）、int8 dotprod（SDOT）路径
- **MTP 树解码**：单次 forward 出 8 token，100% 接受率
- **MULTISEQ**：N 条流共享一次权重读取——聚合吞吐近似线性增长（27B 上实测 64 流达 130 tok/s）
- **Q2_K nibble 布局**（type-42 GGUF）：纯字节重排，解包指令减半（2.2×，数值完全不变）
- 可选的有界 buffer Metal GPU 计算：含逐 tensor 零拷贝直读（`STRATUM_GPU_NC`）与批量多 matmul 提交（对比逐 matmul 等待 17.3×）

## 构建

```sh
cd stratum/native
make stratum          # 通用 GGUF 运行时
make tests            # 量化 kernel 交叉验证 + 采样器精确性
```

## 运行

```sh
./stratum <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]
STRATUM_NO_GPU=1 /usr/bin/time -p ./stratum <model.gguf> 64 0 1   # 纯 CPU 基准
```

环境变量参考与开发边界见 `AGENTS.md`。

## 实测基准

Apple M4 Pro（14 核），24 GB 统一内存，macOS，贪心解码，CPU 路径（`STRATUM_NO_GPU=1`）。方法：`/usr/bin/time -p ./stratum <model.gguf> 64 0 1`（wall 时间含加载 + prefill + 64 个生成 token）。

### 小模型档——感知即时的正常速度（0.5B–1B）

这是 Stratum 最擅长的场景：小模型以**正常、可交互的速度运行，且资源占用近乎为零**——生成过程中机器对其它一切依然响应如常。资源占用**与模型大小无关**：无论 0.5B 还是 27B，绑定内存都保持在 ~7 MB，因为权重从页缓存流式读取。

| 模型 | 格式 | 体积 | tok/s | 体验 |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | Q4_K | 398 MB | **125** | 即时 |
| Qwen3-0.6B | Q4_K | 484 MB | **89** | 即时 |
| MiniCPM5-1B-Base | Q4_K | 688 MB | **68** | 即时 |
| Qwen2.5-Coder-0.5B | F16 | 988 MB | 51 | 即时 |
| Qwen3-0.6B | F16 | 1.5 GB | 40 | 即时 |
| MiniCPM5-1B-Base | F16 | 2.16 GB | 30 | 即时 |

### 大模型档——研究前沿（27B）

| 模型 | 格式 | 体积 | tok/s | 备注 |
|---|---|---|---|---|
| Qwen3.6-27B-mixed | Q2_K/Q4_K/Q6_K 混合 | 11.98 GB | **5.73**（8-token 短测） | 热缓存；长生成持续 ~1.4–2 tok/s |

大模型是权衡最明显的地方：它能跑起来（这台硬件上其它引擎做不到——llama.cpp 要么抖动要么 OOM），但解码受带宽限制，吞吐有限。**在保持资源占用无感的前提下，缩小"小模型速度"与"大模型吞吐"之间的差距——正是本项目当前的研究方向。**

### 资源占用——与模型大小无关

**模型参数量理论上无上限。** 无论多大的模型，绑定内存都保持 ~7 MB，总物理占用只与操作系统选择保留的页缓存成正比——引擎本身从不把模型放进匿名内存。实际限制只有两个：磁盘空间（存放文件）和每 token 的耐心（等待时间）。

| 模型 | 参数量 | 绑定内存（anon） |
|---|---|---|
| Qwen2.5-Coder-0.5B | 0.5B | ~7 MB |
| MiniCPM5-1B-Base | 1B | ~7 MB |
| Qwen3.6-27B-mixed | 27B | **~77 MB**（含 KV/SSM 状态） |

### 内存——核心主张（TinyLlama 1.1B Q4_K_M，64 token）

| | stratum | llama.cpp | 比率 |
|---|---|---|---|
| 匿名（绑定内存） | **7.4 MB** | 634.9 MB | **低 85.7×** |
| 总物理内存 | 645.2 MB | 1300.5 MB | 低 2.0× |

### 规模——24 GB 机器上的 27B

stratum 以约 **77 MB 匿名内存**将 27B 跑到完成（权重经 mmap 从页缓存流式读取）。llama.cpp 在这里无法可用地运行：`-ngl 0` 抖动，`-ngl 99` 需要 16 GB 权重驻留统一内存。Stratum 是唯一能产出 token 的引擎。

## 机制极限

- 单流解码受带宽限制：每个 token 完整读取一次权重流（本机 11.98 GB @ ~27 GB/s 热 ≈ 0.44 s/token）
- Q2_K 解包受计算限制（14 核约 7 GB/s）；nibble 布局打破它（2.2×），代价是文件体积 +50%——需 ≥32 GB 内存保持热
- 长生成树效率 2.46 tok/main——瓶颈是 draft 质量，不是树参数

## 硬件需求

**CPU——必须 Apple Silicon（ARM64）**。引擎的热路径是手写 NEON SIMD 指令，仅存在于 Apple Silicon。Intel 版 Mac（x86_64）不支持。任何 M 系列芯片都可以：M1 / M2 / M3 / M4，含基础款、Pro、Max、Ultra 各变体。

**GPU——不需要买，也不需要装**。Apple Silicon 没有独立显卡，GPU 集成在 SoC 里、通过 Metal 暴露。引擎完全在 CPU 上运行，集成 GPU 只用于可选的加速（`STRATUM_GPU_NC` 等）。只要你有 M 系列 Mac，就已经有 GPU 了，无需任何配置。

**内存——统一内存架构**。CPU 与 GPU 共享同一块 RAM（统一内存）。这正是 24 GB 机器能跑 27B 模型的原因：权重从页缓存流式读取、可回收，真正绑定的只有 ~7 MB。内存大小买的是"模型有多少能保持热"：8 GB 全程从 SSD 流式，24 GB 能保持很大一部分热，≥ 模型体积（27B 需 32 GB+）则完全驻留页缓存。

**内存带宽比核心数更重要**。解码受带宽限制，而 Apple Silicon 的内存带宽随芯片档次增长——这是最大的性能杠杆：

| 芯片档次 | 内存带宽（约） |
|---|---|
| M1 / M2 / M3 基础款 | ~70–100 GB/s |
| M1 Pro / M2 Pro | ~200 GB/s |
| M4 / M4 Pro | ~120 / 273 GB/s |
| M3 Max / M4 Max | ~400 / 546 GB/s |
| Ultra（双 Max） | ~800+ GB/s |

**磁盘——推荐 NVMe SSD**。冷权重从磁盘流式读取；顺序读 ~3 GB/s 的 SSD 能让冷路径保持可用，热后 OS 页缓存会把重复读取变成内存速度。

| | 最低 | 推荐 | 理想 |
|---|---|---|---|
| CPU | 任意 Apple Silicon（M1，8 GB） | M4 Pro，24 GB | M4 Max/Ultra，≥48 GB |
| GPU | 集成（无需购买） | 集成（可选 Metal） | 集成 |
| 内存 | 8 GB | 16–24 GB | ≥ 模型体积（全热） |
| 磁盘 | ≥ 模型文件体积，SSD | NVMe SSD | NVMe SSD |

**性能如何随硬件变化**：每个 token 都完整消费一次权重流，所以每 token 耗时 = `W_bytes × (f_hot / BW_hot + f_cold / BW_cold)`——内存决定模型有多少能热驻留在页缓存（f_hot），内存带宽决定 BW_hot，SSD 决定冷流式速度。27B 预估表现：

| 硬件 | 内存 | 预期（估算） |
|---|---|---|
| M1 基础款，8 GB | 全冷流式 | ~0.2–0.3 tok/s（SSD ~3 GB/s） |
| M4 Pro，24 GB | 部分热 | **5.73 tok/s 热短测 / 持续 ~1.4–2**（实测） |
| M4 Max，48 GB+ | 全热 | ~8–10 tok/s（更高带宽） |
| ≥128 GB 工作站 | 全热 + nibble 布局 | 12+ tok/s（Q2_K 解包 2.2×） |

## 致谢

本项目站在以下项目的肩膀上：

- **[AirLLM](https://github.com/lyogavin/airllm)** —— 工作集应随层大小而非模型大小扩展。我们将其推进到张量粒度。
- **[ds4 / DwarfStar](https://github.com/antirez/ds4)** —— 窄专业引擎、解码图捕获、专家 LRU、"关键处精确"。
- **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** —— 8 GB 跑 2.78T；O_DIRECT、打包数据直算、位级确定性契约。我们的 nibble 布局质疑格式本身，正如 k3 质疑每一个字节。
- **[flash-moe (Alexintosh)](https://github.com/Alexintosh/flash-moe)** —— Apple Silicon 上的纯 C/Metal MoE；SSD 专家流式、FMA 融合解包、"信任 OS 页缓存"。
- **[HuggingFace transformers](https://github.com/huggingface/transformers)** —— 参考框架。
- 另参考：[**tessera**](https://github.com/geoph9/tessera)（NoCopy + `MADV_DONTNEED` 页驱逐）与 ggml/llama.cpp（量化格式与工具链）。

## 仓库布局

```
├── README.md          ← 本文件（英文）
├── README_CN.md       ← 中文版
├── AGENTS.md          ← 开发指南、边界、环境变量参考
└── stratum/
    ├── native/        ← 引擎（C/Metal，约 2.5 万行）、Makefile、gate 脚本
    ├── docs/          ← 实验证据库（159 次运行）+ README
    ├── tools/         ← 独立 GGUF 工具
    └── benchmarks/    ← shell/python 基准脚本
```

## 许可证

Apache-2.0（占位——v0.1 前最终确定）。
