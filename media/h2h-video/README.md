# Stratum vs llama.cpp — 流式生成对比视频集

同一份 GGUF、同一段提示词、同为 CPU-only 贪心解码,两个引擎并排流式输出,画面右上角悬浮实时 HUD(生成速度、匿名内存、页缓存、token 进度),底部为匿名内存对比条 —— 类似游戏跑分视频里的帧数悬浮窗。

视频可直接用于推文 / 短视频传播,配合下方脚本与日志可完整复现。

---

## 测试环境(全部视频统一)

| 项目 | 值 |
|---|---|
| 机器 | Apple M4 Pro,14 核,24 GB 统一内存 |
| 系统 | macOS(ARM64) |
| stratum | 本仓库 `stratum/native/stratum`,含本次 4 处引擎修复(见附录 A);构建选项默认(MemX + Metal 编入,CPU 运行) |
| llama.cpp | Homebrew `llama-simple`,build b9180(255582687),CPU-only(`-ngl 0`) |
| 模型 | **两个引擎共用同一 GGUF 文件**(由本地 HF 权重转换,见附录 C) |
| 解码 | 贪心(温度 0);llama.cpp 侧 `--ignore-eos` 保证跑满 192 token |
| prompt | Qwen3 chat 模板:`<\|im_start\|>user\nWhat is the capital of France? Answer in one short sentence.<\|im_end\|>\n<\|im_start\|>assistant\n`(两引擎 token 序列完全一致) |
| 生成长度 | 192 token(预热轮后取热轮数据) |
| **内存口径** | `vmmap --summary` 每 0.2s 采样:**匿名内存 = Physical footprint**(约束性占用,即机器真正被"钉住"的部分);**页缓存 = mapped file resident**(可回收,不构成约束)。与仓库 `stratum/benchmarks/headtohead.sh` 的官方方法论一致 |
| 测量方法 | 两遍法:速度跑(无 vmmap 采样,避免采样器挂起进程拖慢计时)+ 内存跑(带 vmmap 采样,wall 时间不可比);每配置跑 2 轮,报告热轮(run 2) |
| 温度状态 | 无热降频(macOS `pmset` 无 thermal 警告) |

**为什么速度跑和内存跑分开**:`vmmap` 采样会短暂挂起目标进程,对 1–5 秒的短跑动辄引入数秒误差,同轮测出的速度不可信。分离后各自干净。

---

## 视频 1:Qwen3-0.6B Q4_K_M(主打)

**[videos/qwen3-0.6b-q4km-stratum-vs-llamacpp.mp4](videos/qwen3-0.6b-q4km-stratum-vs-llamacpp.mp4)**(1280×720, ~6.3s)

<video src="videos/qwen3-0.6b-q4km-stratum-vs-llamacpp.mp4" controls muted loop style="max-width:100%"></video>

| 指标 | stratum | llama.cpp | 差距 |
|---|---|---|---|
| 生成速度(热) | 56.4 tok/s | 228.9 tok/s | llama.cpp 快 4.1× |
| **匿名内存(峰值)** | **58 MB** | **475 MB** | **stratum 低 8.2×** |
| 页缓存(峰值,可回收) | 462 MB | 411–540 MB | 相当 |
| 总 wall(含加载) | 3.60 s | 1.60 s | — |

### 复现脚本

```sh
# 1) 测量(每 case 跑两轮取热轮;速度/内存分离采样)
cd media/h2h-video && ./run_all_cases.sh

# 2) 渲染带 HUD 的对比视频
python3 render_h2h.py \
  results/q4km_stratum_speed_run2.json \
  results/q4km_llamacpp_speed_run2.json \
  videos/qwen3-0.6b-q4km-stratum-vs-llamacpp.mp4 \
  --title "Qwen3-0.6B Q4_K_M — stratum vs llama.cpp (CPU, greedy, same GGUF)"
```

### 测试日志(原始数据摘录)

完整 JSON:`results/q4km_{stratum,llamacpp}_{speed,mem}_run2.json`

```text
[stratum]  Qwen3-0.6B.q4km.llamacpp.gguf  wall=3.60s  tok/s(stream)=56.43
           peak_anon=58 MB    peak_file=462 MB   (mem run: wall=4.9s, 23 samples)
[llamacpp] Qwen3-0.6B.q4km.llamacpp.gguf  wall=1.60s  tok/s(stream)=228.93
           peak_anon=475 MB   peak_file=411 MB   (mem run: wall=3.6s, 17 samples)
llama.cpp 变体:--no-mmap 下 316 tok/s / 477 MB anon(速度更快,内存相同量级)
stratum 变体:STRATUM_HOT_FAST=1 下 55.3 tok/s / 58 MB(热模式无额外收益)
输出正确性:两引擎前 12 token 逐字一致("Okay, the user is asking for the
capital of France in one short sentence. "),其后因 f16/f32 累积路径差异
在接近 logits 上分道,语义平行;stratum 侧与 numpy/f64 权威前向逐 logit
一致(见附录 A 验证方法)。
```

---

## 视频 2:Qwen3-0.6B F16

**[videos/qwen3-0.6b-f16-stratum-vs-llamacpp.mp4](videos/qwen3-0.6b-f16-stratum-vs-llamacpp.mp4)**(1280×720, ~7.5s)

<video src="videos/qwen3-0.6b-f16-stratum-vs-llamacpp.mp4" controls muted loop style="max-width:100%"></video>

| 指标 | stratum | llama.cpp | 差距 |
|---|---|---|---|
| 生成速度(热) | 44.2 tok/s | 200.8 tok/s | llama.cpp 快 4.5× |
| **匿名内存(峰值)** | **57 MB** | **127 MB** | **stratum 低 2.2×** |
| 页缓存(峰值,可回收) | 1434 MB | 1536 MB | 相当 |
| 总 wall(含加载) | 4.81 s | 2.08 s | — |

### 复现脚本

```sh
python3 render_h2h.py \
  results/f16_stratum_speed_run2.json \
  results/f16_llamacpp_speed_run2.json \
  videos/qwen3-0.6b-f16-stratum-vs-llamacpp.mp4 \
  --title "Qwen3-0.6B F16 — stratum vs llama.cpp (CPU, greedy, same GGUF)"
```

### 测试日志(原始数据摘录)

完整 JSON:`results/f16_{stratum,llamacpp}_{speed,mem}_run2.json`

```text
[stratum]  Qwen3-0.6B.f16.qwen3.gguf  wall=4.81s  tok/s(stream)=44.16
           peak_anon=57 MB   peak_file=1434 MB
[llamacpp] Qwen3-0.6B.f16.qwen3.gguf  wall=2.08s  tok/s(stream)=200.79
           peak_anon=127 MB  peak_file=1536 MB
说明:llama.cpp 对 F16 权重保持 mmap 纯读(页干净,不进 footprint),
对 Q4_K_M 则在加载时改写权重页(变 dirty 进 footprint)—— 这解释了
Q4KM 视频 475 MB 与 F16 视频 127 MB 的差异;两种行为均为默认配置实测。
```

---

## 视频 3:Qwen3.6-27B 混合架构(待测)

11.98 GB 的 `qwen3.6-27b-mixed.gguf`。v217 bit-exact gate 已在当前引擎上
PASS(8.00 tok/main,1 main forward,argmax `[2, 220, 16, 13]` 保持)。双引擎
对比测量**等机器空闲时执行**(11 GB 模型测量需要干净的内存环境,且 llama.cpp
对 qwen35 架构的加载行为需要单独摸底 —— 初次尝试时其默认 Metal 模式一次性
占用了 20+ GB 统一内存)。

---

## 附录 A:本次为对比视频完成的引擎修复(全部已验证)

为让 stratum 正确运行标准 Qwen3 GGUF,修复了 `stratum_arch_llama.inc.c`
中四处真实缺陷。修复前 stratum 从未正确运行过 Qwen3 结构的模型:

1. **per-head q/k RMSNorm 未参与计算**(`attn_q_norm`/`attn_k_norm` 被解析
   但前向从未应用)—— 在单 token 路径与 batched prefill 路径分别补上;
2. **RoPE 布局错误**:llama handler 只实现了相邻对(NORM)旋转,而 Qwen 系
   GGUF 使用 NEOX(half-split)布局 —— 按文件 `general.architecture` 选择;
3. **注册 `qwen3` 架构**(`arch_names = "llama,qwen3"`),stratum 从此直接
   消费标准 Qwen3 GGUF,无需任何改名变体;
4. 附带:llama handler 的 SDOT 开关为 `STRATUM_SDOT`(与 linear 层的
   `STRATUM_NO_SDOT` 是两个变量)。

**验证**(AGENTS.md 确定性契约):
- numpy/f64 权威前向(直读 HF safetensors,独立实现):28 层逐层
  max|diff| ≤ 5e-5,首 token argmax 与 logit 逐位一致(15846 / 10.0025);
- 42-token teacher-forcing PPL 与 numpy 权威值一致(1.5018 vs 1.5018);
- 与 llama.cpp 同文件贪心输出前 12 token 逐字一致;
- 纯 llama 回归:`make tests`(quant kernel 8/8)、`spec_sample_test`、
  `dit_probe` + oracle(max|diff| 5e-7)、tiny-llama 基线 argmax 序列不变;
- **27B(qwen35 路径)`v217_gate.sh` PASS**(本修复不触及 q35 handler,
  gate 确认 bit-exact 无回归)。

## 附录 B:已知限制(如实记录)

- **SDOT(int8 x 预量化)在 Qwen3-0.6B 上数值爆炸**(PPL 18 万级):Qwen3
  的残差流范数(数千)远超 llama 家族(数十),超出该近似的适用域。本对比
  全程 `STRATUM_SDOT=0`(精确浮点路径)。SDOT 在 27B(qwen35)上依旧
  bit-exact,为已知悬案,留待专项调查。
- 小模型热缓存下 stratum 生成速度低于 llama.cpp 4–5×:流式架构每 token
  扫描全模型并携带调度开销(mincore/预取/MemX stage),在 0.6B 这种
  计算量极小的模型上占比显著。这是当前实测现状;内存与可运行模型规模才是
  stratum 的设计点(27B 上对比更公平,见视频 3)。

## 附录 C:测试资产的来历(无下载、无网络模型)

- `Qwen3-0.6B.f16.qwen3.gguf`:本地 HF 权重 `~/Desktop/0-/Qwen3-0.6B/model.safetensors`
  经 llama.cpp `convert_hf_to_gguf.py` 转换(标准 qwen3 架构,311 张量);
- `Qwen3-0.6B.q4km.llamacpp.gguf`:上一文件经 `llama-quantize ... Q4_K_M`。
  两个引擎**共用这两份文件**(字节级相同,双引擎交叉验证);
- 设备上原有的 `Qwen3-0.6B.q4k.gguf` / `Qwen3-0.6B.llama.f16.gguf` 为
  stratum 旧转换产物(缺 q/k_norm、llama.cpp 无法加载),已从测试矩阵
  排除并在本次会话中删除;
- prompt 的 token 序列由本地 `tokenizer.json` 离线编码,两引擎完全一致。

## 文件清单

```
media/h2h-video/
├── README.md                 ← 本文档
├── videos/                   ← 成品视频(两个,27B 待补)
├── results/                  ← 全部原始测量 JSON(速度跑 + 内存跑 × 2 轮)
├── run_h2h.py                ← 单 case 测量器(时间戳 token 流 + vmmap 采样)
├── run_all_cases.sh          ← 全矩阵测量 driver(速度/内存两遍法)
├── render_h2h.py              ← HUD 对比视频渲染器(PIL + ffmpeg)
├── gguf_arch_rename.py       ← (已完成历史使命)qwen3→llama 改名工具,引擎注册 qwen3 后退役
├── q4read_check.c            ← Q4_K 张量读取验证器(调试期证据)
└── q4matmul_check.c          ← 端到端单矩阵 dispatch 对拍器(调试期证据)
```
