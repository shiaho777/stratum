# Stratum

[English](README.md) | [简体中文](README_CN.md)

![CI](https://github.com/shiaho777/stratum/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)

**27B 模型用大约 77 MB 匿名内存就能跑完。** 权重留在文件里：引擎 `mmap` GGUF，经页缓存读取。0.5B 到 1B 的匿名内存大约 7 MB；27B 把 KV 和 SSM 状态算进去大约 77 MB。内存决定有多少文件留在热页里，也就决定下一个 token 有多快。它不决定模型能不能跑完。

<p align="center"><img src="docs/assets/hero-stream.svg" alt="权重从磁盘经页缓存流入引擎，匿名内存保持一条平线" width="760"></p>

实测机器：M4 Pro、24 GB、贪心。除注明外走 CPU。

| | |
|---|---|
| 27B（Qwen3.8，11.98 GB） | 热缓存短跑 **5.73 tok/s**，长生成约 1.4–2 tok/s，匿名内存 **约 77 MB** |
| 0.5B–1B Q4_K | **68–125 tok/s**，匿名内存 **约 7 MB** |
| 对比 llama.cpp，TinyLlama 1.1B | 匿名内存 **7.4 MB 对 634.9 MB**（**85.7×**） |
| 对比 llama.cpp Metal，Qwen3-0.6B Q4_K_M | 融合 GPU **247–252 tok/s** 对 224–231（在回合结束处停下，需要 `STRATUM_MAX_KV=8192`） |
| 引擎 | 二进制约 752 KB。仅 Apple Silicon。GPU 可选 |
| 架构 | Llama / Qwen2 / Qwen3 dense · Qwen3.5/3.8 混合（`qwen35`）· MoE（`llama-moe`，实验性）· DiT 与 H3 视频探针（独立程序） |

## 快速开始

需要 Apple Silicon。Intel Mac 不在范围内。热路径是手写 NEON。

```sh
cd stratum/native
make                         # ./stratum（含 Metal）。不要 MemX 或 Metal：make USE_MEMX=0 或 USE_METAL=0
make tests

./stratum <model.gguf> 8 1 2 3          # 生成 N 个 token，后面是 prompt 的 token id
STRATUM_NO_GPU=1 ./stratum <model.gguf> 64 0 1
```

仓库不带权重。一个确定性小 GGUF 就够看一次真正的贪心解码：

```sh
python3 ../tools/make_tiny_model.py --arch llama --weights q4k --out /tmp/tinyq.gguf
STRATUM_NO_GPU=1 ./stratum /tmp/tinyq.gguf 8 1 2 3 4 5 6 7 8
```

`general.architecture` 决定交给哪个架构。一个进程一个模型。stderr 里要看的是匿名内存，旁边是 tok/s。

## 怎么工作

一个 token 就是对权重流的一次扫描。文件只读映射。操作系统留住的页是白来的速度；丢掉的页再从 SSD 读。匿名内存是激活、KV cache 和 SSM 状态。

流上面有三层调度。热层纯计算。冷层在当前层计算时，以 50–100 MB 的窗口预取。`STRATUM_MULTISEQ=N` 让 N 条解码流共享一次权重扫描（27B 上 64 流实测聚合 **130 tok/s**）。

在这之上，权重不重量化，页也不锁：

- **Q2_K nibble 布局**（type 42）重排同样的 2-bit 值。解包实测 2.2×。用 `STRATUM_Q2K_NIB` 打开。
- **MoE** 每个 token 碰到 E 个专家里的 k 个。`moe_expert_pack.py` 把一个专家的 gate、up、down 排成一段连续字节。引擎可以在 attention 期间预取一个猜测，并在页缓存预算内维持一个衰减热集。真正做选择的仍是 router。
- **MTP** 一次 forward 验证一棵 draft 树，未命中就回滚，发出的 token 与贪心相同。低置信度的 draft 可以在付一次权重读取之前停掉。
- **Metal** 按 tensor、有上界。把整个模型注册成一块 GPU 缓冲是失败模式。

这些数字站得住，是因为有三条边界：不重量化、不 `mlock`、不对整份模型做 NoCopy。写在 [AGENTS.md](AGENTS.md)。

## 接着读

| | |
|---|---|
| [工作原理](docs/HOW_IT_WORKS_CN.md) | 权重流、MoE pack、MTP、Metal，以及实测表 |
| [验证](stratum/docs/VALIDATION.md) | v199–v217 覆盖什么，以及 MoE oracle |
| [环境变量](stratum/docs/ENVVARS.md) | 每一个 `STRATUM_*` 开关 |
| [AGENTS.md](AGENTS.md) | 边界与确定性约定 |
| [对照录像](media/h2h-video/) | stratum 与 llama.cpp，附原始 JSON |

## 许可证

Apache-2.0.
