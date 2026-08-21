# Stratum

[English](README.md) | [简体中文](README_CN.md)

一个面向 Apple Silicon 的纯 C Transformer 推理引擎，其**驻留内存占用与模型大小无关**。权重是流，不是驻留物：引擎通过 `mmap` 映射模型并经由 OS 页缓存读取，因此匿名内存无论是 0.5B 还是 27B 模型都保持在 ~7 MB。

| 指标 | 数值 |
|---|---|
| 实测模型 | 27B dense（Qwen3.6，11.98 GB GGUF），小至 0.5B |
| 27B 的匿名（wired）内存 | **~77 MB**（含 KV/SSM 状态） |
| 0.5–1B 模型的匿名内存 | ~7 MB |
| 对比 llama.cpp（TinyLlama 1.1B） | 匿名内存**低 85.7×** |
| 引擎体积 | ~790 KB 二进制，~2.5 万行 C/Metal |
| GPU 需求 | 无（集成 GPU，可选 Metal 加速） |
| 架构支持 | Llama 家族 + Qwen3.5/3.6 混合（Gated DeltaNet SSM + attention） |

真正的约束是磁盘和带宽，而不是内存：只要模型文件放得下，模型就能跑完。内存买的是"模型在页缓存中有多少保持热"——内存越多，每个 token 的 SSD 读取越少，吞吐越高。除此之外一切不变。

---

## Part I — 快速上手

### 硬件需求

| 组件 | 要求 |
|---|---|
| CPU | **必须是 Apple Silicon（ARM64）**。热路径是手写 NEON SIMD，只存在于 Apple Silicon。Intel Mac（x86_64）不支持。M1/M2/M3/M4 任意型号均可，含 base/Pro/Max/Ultra。 |
| GPU | 无需购买。GPU 集成在 SoC 中，通过 Metal 暴露；引擎完全在 CPU 上运行，GPU 仅用于可选加速。 |
| 内存 | **几乎没有最低限制——任何现代设备都能跑。** 引擎的匿名需求只有 ~7 MB（小模型）/ ~77 MB（27B，含 KV/SSM 状态）；权重放在可回收的页缓存里，所以内存不决定模型能不能跑，只决定跑多快。1 GB 是现代设备的基线；而 Apple Silicon Mac 实际起步就是 8 GB。 |
| 磁盘 | ≥ 模型文件大小，推荐 NVMe SSD。冷权重从磁盘流式读取；~3 GB/s 顺序读即可保证冷路径可用。 |

内存带宽是主导性能的杠杆，而不是核心数。Apple Silicon 带宽随芯片档位增长：

| 芯片档位 | 内存带宽（约） |
|---|---|
| M1 / M2 / M3 base | ~70–100 GB/s |
| M1 Pro / M2 Pro | ~200 GB/s |
| M4 / M4 Pro | ~120 / 273 GB/s |
| M3 Max / M4 Max | ~400 / 546 GB/s |
| Ultra（双 Max） | ~800+ GB/s |

### 达到 5 tok/s 的最低配置

"能跑"和"跑得快"是两个问题。以下是各模型档位达到最低 **5 token/s** 的配置，由每 token 时间公式 `W × (f_hot / BW_hot + f_cold / BW_cold)` 推导，并以实测硬件锚定：

| 模型 | 达到 5 tok/s 的最低配置 | 为什么 |
|---|---|---|
| 1B（Q4_K，~688 MB） | **任意 Apple Silicon（M1 base 及以上）+ ≥1 GB 内存 + SSD** | 5 tok/s = 200 ms/token = 688 MB / 0.2 s ≈ **3.4 GB/s** 有效带宽。任何 M 芯片（~70 GB/s+）和任何 SSD（~3 GB/s+）都超出一个数量级——M4 Pro 实测 68 tok/s，即 1B 规模下 ~20× 余量。 |
| 27B（Q2_K/Q4_K/Q6_K，11.98 GB） | **M4 Pro 级（~250 GB/s）+ ≥16 GB 内存（模型大部分保持热）+ NVMe SSD** | 5 tok/s = 200 ms/token = 11.98 GB / 0.2 s ≈ **60 GB/s** 有效带宽。解码只用到标称带宽的 ~25–30%，所以芯片需要 ~250 GB/s 标称——即 M4 Pro 档（273 GB/s）。低于 M4 Pro，即使全热，M2 Pro（~200 GB/s）也只能到 ~4 tok/s。内存 ≥16 GB 让模型大部分驻留页缓存（实测锚点：M4 Pro + 24 GB = 5.73 tok/s 热短测）。 |

模型*能跑*所需的配置远低于此：同样的 27B 在 M1 base + 8 GB 上全冷流式也有 ~0.2–0.3 tok/s。上表是吞吐不再是"等待游戏"的门槛。

### 快速开始

```sh
cd stratum/native
make                  # 构建 ./stratum（+ GPU 路径用的 stratum_q4k.metallib）
make tests            # 量化 kernel 交叉验证 + 采样器精确性

# 冒烟测试 —— 任意小 GGUF（Qwen、Llama、TinyLlama……）：
./stratum <model.gguf> 2 1 450 7483 310 3444 338
```

仓库不附带、不内置、不假设任何模型：引擎和所有脚本都要求把模型路径作为显式参数传入。二进制从 GGUF 元数据读取 `general.architecture` 并分发到已注册的处理器——没有任何模型名被硬编码。

构建选项：MemX（`github.com/shiaho777/memx`，MIT）是可选的内存压缩运行时，承载暂存缓冲与 KV/SSM 状态——**首次使用时自动拉取**（`make deps` 克隆/更新；直接 `make` 会在需要时自动拉取），可用 `make USE_MEMX=0` 禁用。`make USE_METAL=0` 构建纯 CPU 版（不含 Metal shader 库）。

### 用法

```
stratum <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]
```

- `<model.gguf>` — 受支持 GGUF 的路径（必填）
- `N_GENERATE` — 生成的 token 数（默认 32）
- `PROMPT_TOKEN_ID...` — 预分词 prompt；省略则从 stdin 读原始 token

示例：

```sh
# 纯 CPU、贪心、基准风格（wall = 加载 + prefill + 64 tokens）：
STRATUM_NO_GPU=1 /usr/bin/time -p ./stratum <model.gguf> 64 0 1

# GPU 路径（自动从 CWD 或 native/ 探测 stratum_q4k.metallib）：
STRATUM_GPU=1 ./stratum <model.gguf> 32 1 450 7483 310 3444 338
```

Metal 库自动检测（当前目录，然后 `native/`）；Metal 设备自动检测（`MTLCreateSystemDefaultDevice`）。完整开关见 [Part V — 环境变量](#part-v--参考)。

### 读懂运行报告

引擎向 stderr 打印诊断日志。三个关键数字：

- **`wired` / 匿名内存** —— 核心主张。小模型 ~7 MB，27B ~77 MB（含 KV/SSM 状态）。这个数字**不会**随模型大小增长。
- **`tok/s`** —— 解码吞吐。`MULTISEQ` 模式下聚合行显示 `aggregate X tok/s (per-stream Y tok/s)`。
- **`argmax` 序列** —— 贪心 token id；gate 脚本用它断言 bit-exact。

### 常见问题

- **为什么我的机器在交换/变慢？** 模型可能是冷的；第一遍从磁盘流式读取。大跑之前先查可用内存（`vm_stat`、`sysctl vm.swapusage`）。24 GB 机器上不要在有重型后台进程时跑 12 GB 模型。
- **它会用我的 GPU 吗？** 只有设置了 `STRATUM_GPU` / `STRATUM_GPU_NC` 才会。默认纯 CPU。
- **能跑比内存大的模型吗？** 能——这正是设计点。它需要的是磁盘空间和耐心，不是内存。
- **为什么大模型解码慢？** 每个 token 都要读一遍完整权重流（11.98 GB @ ~27 GB/s 热 ≈ M4 Pro 上 0.44 s/token）。解码按构造就是带宽受限的；见 [Part IV — 机制极限](#机制极限)。
- **一个进程能跑多个模型吗？** 不能——引擎按设计是单模型、单进程：状态是文件级全局变量，一个进程只能跑一个模型（见 AGENTS.md）。

---

## Part II — 工作原理

### 问题：dense 权重是一条流

MoE 模型每个 token 只激活一部分参数——kimi-k3 激活 896 个专家中的 16 个（~3.7%），所以它可以把 93% 的权重流式化。**dense** 模型没有这种运气：每个 token 消耗*每一个*权重，一次。传统引擎的应对是整模型载入 RAM/VRAM，这就是为什么 llama.cpp 跑 27B 需要 ~16 GB 统一内存，没有时只能颠簸。

Stratum 持相反立场：**工作集与模型大小无关**。一个 token = 一次完整权重流扫描，而这条流从来不需要驻留。结果就是 27B 在 24 GB 机器上以 ~77 MB 匿名内存跑完——换到 4 GB 机器也一样跑，只是更慢。

### 削减 1 — `mmap`：权重从不进入匿名内存

模型文件只读映射，每次权重读取都直接经过 OS 页缓存：

- 页缓存**可回收**——内存压力下 OS 驱逐权重页而不是杀死进程，与任何普通文件读取完全相同。
- 匿名内存只为模型运行真正需要的东西保留：激活、KV 缓存、SSM 状态。这就是 ~7 MB / ~77 MB 数字的来源。
- 预取是建议性的，不是驻留：`madvise(MADV_WILLNEED)` + `F_RDADVISE` 提示内核提前载入（逐 tensor 粒度，50–100 MB 窗口），随即释放。这些页仍然是普通页缓存。

这就是为什么内存下限是一个"刻度盘"而不是悬崖：内存大小决定 `f_hot`（页缓存能装下的权重比例），吞吐遵循 [Part IV](#性能如何缩放) 的公式。

### 削减 2 — 流式调度器：热检测，冷预取

不是所有层都平等。24 GB 机器 + 12 GB 模型时，第一遍扫描后大部分权重在页缓存中保持热。调度器测量而非猜测：

- **热检测** —— `mincore()` 对每层采样 3–12 页；若 ≥80% 驻留则判定为"热"，作为纯计算运行，无 I/O 等待。
- **冷路径** —— 冷层获得合并的 `madvise`/`F_RDADVISE` 突发（覆盖接下来几层，24 MB 间隙合并为一次调用），SSD 持续流式读取的同时计算继续推进。
- **确定性模式**（`STRATUM_STREAM_DET=1`）完全跳过 mincore 开销，一律按冷处理，保证可复现。

实测 mincore 采样本身在热路径上的成本约为零——赢的不是检测，而是知道什么时候*不用*预取。

### 削减 3 — MULTISEQ：一次权重扫描，N 个逻辑流

这是摊薄打法：`STRATUM_MULTISEQ=N` 让 N 条独立解码流（相同 prompt，各自采样后分叉）每一步共享**一次权重扫描**。N 条流共享同一批 tensor 读取；计算在其上批量执行。

- 聚合吞吐随 N 近似线性增长：27B 上 **64 流 130 tok/s**（实测），单流 ~2.0 tok/s。
- 运行报告打印 `weight-scans` 与 `main-scans`——比值就是摊薄因子。理论行：*一次冷扫描服务 N 个逻辑流*。
- 同 prompt 克隆流折叠是现实场景：N 份同一对话 130 tok/s 聚合 = 每份 130/N，仍然只有一次权重读取。

### Q2K nibble 布局（GGUF type 42）

Q2_K 解包是计算受限的（14 核 ~7 GB/s）——瓶颈是反量化指令而不是内存。nibble 布局是**字节重排**，不是重新量化：

- 每个 2-bit 权重值不变（0–3）；只置换字节打包方式，使 NEON 解包 kernel 的指令数减半。
- 实测**解包快 2.2×**，与原始布局 bit-exact。
- 代价：Q2_K tensor 文件体积 +50%（27B 的 sidecar 16.3 GB），需要 ≥32 GB 内存保持热——因此 nibble 路径通过 `STRATUM_Q2K_NIB=<path>` 或内嵌 type-42 GGUF 显式启用。

转换器（`tools_gguf_nib_convert.c`）输出 sidecar 或内嵌 type-42 GGUF，引擎在使用前会验证 nibble 与原值一致（`[NIB-DBG]` 诊断）。

### MTP 树解码

多 token 预测：小 draft head 每次 forward 提出 8-token 树；主模型一次验证整棵树。

- **draft 一致时每次 forward 8 个 token，100% 接受**——无额外 pass，输出与贪心完全相同。
- 27B 上的墙是 draft 的*质量*而不是树参数：实测树效率 2.46 tok/main，树深实验（`STRATUM_TREE_EXTEND_K`）超过 4 反而更差。树机制本身没问题；draft head 才是约束。

### Metal GPU：有界缓冲，逐 tensor 零拷贝

GPU 可选且严格有界：

- **有界 staging** —— GPU 缓冲有上限；引擎永远不会把模型镜像到 GPU。
- **逐 tensor NoCopy**（`STRATUM_GPU_NC=1`）——每个 matmul 只把自己的 tensor（<100 MB）用 `newBufferWithBytesNoCopy` 视图包住 mmap，分发，释放。wired 内存保持平稳（实测 11.98 GB 顺序 GPU 读取仅 +0.04 GB）。
- **批量提交**（`stratum_metal_nc_batch_*`）——多个独立 matmul 共享一个 command buffer：**比逐 matmul 等待快 17.3×**（480 个 matmul：3.71 s → 0.21 s）。
- **失败模式是整模型注册**——对整个 mmap（或 >~1 GB 的块）建一个 NoCopy 缓冲会把整个模型 wire 住并 OOM。这正是逐 tensor 粒度要避免的。此外任何地方都不锁页缓存：`keep_resident` 被禁止，`STRATUM_HOT_FAST` 是不锁页缓存的热模式调度器。

既然 CPU 能跑，为什么还要 GPU？内存层面的答案才是重点：GPU 加速是*有界*资源，所以它永远不会改变内存故事——在一个带宽受限的流水线里，它只能加速计算部分。

### KV 缓存与 SSM 状态

运行状态是模型唯一需要的匿名内存：

- 全注意力层保留 KV 缓存；qwen35 混合架构的 SSM 层（Gated DeltaNet）保留固定大小的 delta-rule 状态，**与上下文长度无关**——每个 token 原位更新的有界矩阵，类似 kimi 的 KDA。
- 27B 的 ~77 MB 匿名 = KV/SSM 状态 + 激活 + 小块 scratch；权重本身的贡献为 ~0。

### 代码结构

```
stratum/native/
├── stratum.c                    ← 入口：读 general.architecture，分发
├── stratum_arch.h               ← 通用架构注册表 + 配置加载
├── stratum_linear.h             ← 通用量化线性层（Q2_K…Q8_0/F16/F32）
├── stratum_engine.h             ← CPU/GPU 初始化、madvise、投机解码、内存报告
├── stratum_arch_llama.inc.c     ← Llama/Qwen2/Qwen3 dense 架构（自注册）
├── stratum_arch_qwen35.inc.c    ← qwen35 混合架构：Gated DeltaNet + attention（~2.4 万行）
├── stratum_metal.m/.h           ← Metal 层：GEMV kernel、batched-B、group dispatch、NC
├── stratum_q{k}_*.{h,neon.h,metal} ← 量化 kernel（scalar + NEON + Metal）
├── v199–v217_gate.sh            ← bit-exact 回归 gate
└── Makefile                     ← 构建 ./stratum（+ metallib）
```

新增架构 = 写 `stratum_arch_<name>.inc.c`，实现 `StratumArch` 接口，注册——Makefile 会自动把 `*.inc.c` 收集进 `stratum_archs.gen.h`，因此不需要改任何现有文件。对模型也是同理：没有任何逐模型硬编码。

### 不变量

引擎绝不越过的三条硬边界——它们保证数字始终可信：

1. **永不损害质量** —— 不重新量化现有权重（Q4_K→Q4_0 被禁止），不跳层，不近似计算。唯一允许的"近似"是 int8 SDOT，已验证贪心 bit-exact。nibble 布局是字节置换，不是重新量化。
2. **永不增加 wired 内存** —— 不整模型 mlock，不用 GPU 缓冲缓存权重，不注册 >1 GB 的 NoCopy。权重流式；只有 KV/SSM 状态驻留。
3. **测试不得耗尽机器** —— 大跑之前检查内存；kernel 实验用微基准，不用 27B 端到端。

---

## Part III — 验证

引擎把正确性当作契约而不是希望：

- **`quant_test`** —— 每个量化 kernel 与标量参考交叉验证。
- **`spec_sample_test`** —— Leviathan-Chen 拒绝采样精确性。
- **19 个 gate 脚本（`v199`–`v217`）** —— qwen35 架构 + 27B 的全模型贪心回归：断言精确 argmax 序列 `[2, 220, 16, 13]` 与 `tok/main ≥ 8.0`。任何引擎改动必须让所有 gate 保持通过。
- **双路径纪律** —— CPU（NEON）与 GPU（Metal）路径都被覆盖；逐 tensor NoCopy 在 27B 上与 CPU 路径验证 bit-exact 后才被允许。

---

## Part IV — 实测数据

所有数字实测于 Apple M4 Pro（14 核）、24 GB 统一内存、macOS、贪心解码、CPU 路径（`STRATUM_NO_GPU=1`），方法 `/usr/bin/time -p ./stratum <model.gguf> 64 0 1`。

### 小模型 — 感知即时（0.5B–1B）

这是 Stratum 的最佳舞台：小模型以正常、交互级速度运行，资源占用近乎为零——生成时机器对其它一切照常响应。wired 内存无论 0.5B 还是 27B 都保持 ~7 MB。

| 模型 | 格式 | 大小 | tok/s | 感知 |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | Q4_K | 398 MB | **125** | 即时 |
| Qwen3-0.6B | Q4_K | 484 MB | **89** | 即时 |
| MiniCPM5-1B-Base | Q4_K | 688 MB | **68** | 即时 |
| Qwen2.5-Coder-0.5B | F16 | 988 MB | 51 | 即时 |
| Qwen3-0.6B | F16 | 1.5 GB | 40 | 即时 |
| MiniCPM5-1B-Base | F16 | 2.16 GB | 30 | 即时 |

### 大模型 — 研究前沿（27B）

| 模型 | 格式 | 大小 | tok/s | 备注 |
|---|---|---|---|---|
| Qwen3.6-27B-mixed | Q2_K/Q4_K/Q6_K 混合 | 11.98 GB | **5.73**（8-token 短测） | 热缓存；长生成持续 ~1.4–2 tok/s |

大模型是权衡最明显的地方：它能跑（llama.cpp 在这里颠簸或 OOM），但解码受带宽限制。**在保持资源占用无感的前提下缩小小模型速度与大模型吞吐之间的差距，是本项目当前的研究方向。**

### 资源占用 — 与模型大小无关

| 模型 | 参数量 | Wired（匿名） |
|---|---|---|
| Qwen2.5-Coder-0.5B | 0.5B | ~7 MB |
| MiniCPM5-1B-Base | 1B | ~7 MB |
| Qwen3.6-27B-mixed | 27B | **~77 MB**（含 KV/SSM 状态） |

**模型大小理论上无上限。**无论参数量多少，wired 内存都保持在 ~7 MB 量级，总物理占用与 OS 选择保留的页缓存成正比——引擎本身从不把模型放进匿名内存。唯一的实际限制是磁盘空间和每个 token 的耐心。

### 内存 — 核心主张（TinyLlama 1.1B Q4_K_M，64 tokens）

| | stratum | llama.cpp | 比值 |
|---|---|---|---|
| 匿名（绑定 RAM） | **7.4 MB** | 634.9 MB | **低 85.7×** |
| 总物理 | 645.2 MB | 1300.5 MB | 低 2.0× |

### 规模 — 24 GB 机器上的 27B

stratum 以 ~77 MB 匿名内存跑完 27B。llama.cpp 在这里无法可用地运行它：`-ngl 0` 颠簸，`-ngl 99` 需要 16 GB 统一内存。Stratum 是唯一能产出 token 的引擎。

### 性能如何缩放

每个 token 消耗一次完整权重流，因此每 token 时间为 `W_bytes × (f_hot / BW_hot + f_cold / BW_cold)`——RAM 决定有多少保持热（`f_hot`），内存带宽决定 `BW_hot`，SSD 决定冷流式速率。27B 预估表现：

| 硬件 | 内存 | 预期（估算） |
|---|---|---|
| M1 base，8 GB | 全冷流式 | ~0.2–0.3 tok/s（SSD ~3 GB/s） |
| M4 Pro，24 GB | 部分热 | **5.73 tok/s 热短测 / ~1.4–2 持续**（实测） |
| M4 Max，48 GB+ | 全热 | ~8–10 tok/s（更高带宽） |
| ≥128 GB 工作站 | 全热 + nibble 布局 | 12+ tok/s（Q2_K 2.2× 解包） |

### 机制极限

- **单流解码是带宽受限的**：每个 token 读一遍完整权重流（本机 11.98 GB @ ~27 GB/s 热 ≈ 0.44 s/token）。
- **Q2_K 解包是计算受限的**（14 核 ~7 GB/s）；nibble 布局突破这一点（2.2×），代价是文件体积 +50%——需要 ≥32 GB 内存保持热。
- **长生成树效率 2.46 tok/main**——墙是 draft 质量，不是树参数。

---

## Part V — 参考

### 环境变量

引擎有 200+ 环境变量（多数是实验遗留的 GPU kernel 变体开关）。关键项：

| 变量 | 用途 | 边界 |
|---|---|---|
| `STRATUM_NO_GPU=1` | 强制纯 CPU（默认基准配置） | ✅ |
| `STRATUM_GPU=1` / `STRATUM_GPU_NC=1` | GPU 路径；NC = 逐 tensor 零拷贝 | ✅（NC bit-exact） |
| `STRATUM_MULTISEQ=N` | N 条流共享一次权重扫描 | ✅（64 流 130 tok/s） |
| `STRATUM_MTP` / `STRATUM_TREE_*` | MTP 树 / 投机解码 | ✅ |
| `STRATUM_ASYNC_PREFETCH=1` | 后台 pread 预取 | ✅ |
| `STRATUM_HOT_FAST=1` | 热缓存纯计算模式（不锁页缓存） | ✅ |
| `STRATUM_STREAM_DET=1` | 确定性流式（跳过 mincore 检测） | ✅ |
| `STRATUM_Q2K_NIB=<path>` | Q2K nibble 布局模型（type 42） | ✅（字节重排） |
| `STRATUM_TREE_EXTEND_K=N` | 树链深度（上限 12，默认 4） | ✅ |
| `STRATUM_GPU2=1` | 冷权重 staging 流水线 | ⚠️ 热权重时比 CPU 慢 |
| `STRATUM_Q2K_SDOT=1` | Q2K 的 int8 SDOT | ⚠️ 14 核无增益 |
| `STRATUM_PREDECODE=1` | Q4_K→F16 预解码到 GPU | ❌ 内存边界 |
| `STRATUM_Q4_0=1` | Q4_K→Q4_0 重编码 | ❌ 质量边界 |
| `STRATUM_MLOCK_ALL` / `STRATUM_KEEP_RESIDENT` | 锁定/钉住权重 | ❌ 内存边界 |

### 仓库布局

```
├── README.md          ← 本文件（英文）
├── README_CN.md       ← 简体中文
├── AGENTS.md          ← 开发指南、边界、完整 env 参考
└── stratum/
    ├── native/        ← 引擎（C/Metal）、Makefile、19 个 gate 脚本
    ├── docs/          ← 实测证据（基准记录）+ README
    ├── tools/         ← 独立 GGUF 工具（量化、检查、解码）
    └── benchmarks/    ← 基准脚本（headtohead、manifesto、GPU 扫描）
```

### 参考

本项目参考过的项目，以及它们的思路在 Stratum 中的体现：

- **[AirLLM](https://github.com/lyogavin/airllm)** —— 工作集应随层大小而非模型大小缩放；Stratum 将同一思路应用到 tensor 粒度。
- **[ds4 / DwarfStar](https://github.com/antirez/ds4)** —— 窄专业引擎、decode 图捕获、专家 LRU、在关键处保持精确。
- **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** —— 8 GB 跑 2.78T；O_DIRECT 读取、打包数据直读、位级确定性契约；nibble 布局本身就在质疑权重格式。
- **[flash-moe (Alexintosh)](https://github.com/Alexintosh/flash-moe)** —— Apple Silicon 上的纯 C/Metal MoE；SSD 专家流式、FMA 融合反量化 kernel、信任 OS 页缓存。
- **[tessera](https://github.com/geoph9/tessera)** —— NoCopy GPU 缓冲 + `MADV_DONTNEED` 页驱逐；为逐 tensor NoCopy 边界提供了依据。（该仓库已不再公开。）
- **[HuggingFace transformers](https://github.com/huggingface/transformers)** —— 用于输出对比的参考框架。
- **[ggml / llama.cpp](https://github.com/ggerganov/llama.cpp)** —— GGUF 量化格式与模型转换工具链。

### 许可证

Apache-2.0。
