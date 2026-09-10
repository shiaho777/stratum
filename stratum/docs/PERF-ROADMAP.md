# 性能超越作战计划(执行中)

目标:小模型热解码速度追平并反超 llama.cpp(M4 Pro / 24GB / CPU-only 同台),
同时保持内存优势(匿名内存不随模型大小增长)与全部 bit-exact gate。

基线(192-token 热轮,SDOT 修复后):

| 模型 | stratum | llama.cpp | 差距 |
|---|---|---|---|
| Qwen3-0.6B Q4_K_M | 46.0 tok/s | 230.2 tok/s | 5.0× |
| MiniCPM5-2B Q4_K_M | 31.8 tok/s | 236.4 tok/s | 7.4× |

侦察结论:llama.cpp 的 AMX 路径是 x86 专属(`__AMX_INT8__`+AVX512VNNI),
Apple Silicon 上他们用的是与我们同级的 NEON/dotprod —— 差距全部是软件层,
可达。另:每核 kernel 效率 11 vs 23 GB/s;我们 10 P 核 vs 他们默认 14 逻辑核。

## 阶段计划(每步:微基准先行 → 端到端测量 → gate 全绿 → 独立 PR)

### P0 — 重画像(SDOT 修复后)
TYPETIME 分解 Q4_K/Q6_K 耗时占比;nchunks 10 vs 14 在 SDOT=1 下重扫
(SDOT=0 时代的扫描结论已过期)。产出:剩余差距的构成账本。
验收:差距归因到 {kernel每核效率, 线程数, 非matmul开销} 三桶。

### P1 — 全核参战
`nchunks` 默认改用全部物理核(含 E 核,带宽瓶颈下 E 核有效)。
验收:tok/s 提升 ≥10% 且 gate 全绿;若 <10% 记录数据后回滚默认值。

### P2 — kernel 升级:复用 27B 路线的已验证快核
仓库里为 qwen35(27B)写的 rows2/pack_b7/multix 内核(rows2: 一次 x 加载
喂两行权重;pack: 打包激活)从未接到小模型 llama 路径。评估移植:
la_forward_block 的 Q4_K/Q6_K dispatch 接 rows2 变体。
验收:微基准每核 ≥16 GB/s;端到端 ≥1.3×;数值 bit-exact(PPL 对照)。

### P3 — 常驻线程池
若 P0 显示调度占比显著:pthread 池替代 per-matmul `dispatch_apply`
(每 token ~300 次带屏障提交)。每行计算独立 → 调度方式不影响 bit-exact。
验收:微基准无屏障开销;端到端再 +15% 以上才保留。

### P4 — 苹果 AMX int8(月度级研究,时间盒控制)
M4 原生矩阵单元,int8 吞吐数倍于 dotprod;无人区(公开无 GGUF 实现)。
阶段1:独立子进程 probe 指令编码(SIGILL 安全);阶段2:Q8×Q8 dot 原型。
验收:probe 通过才继续;任何阶段可无损放弃,不影响 P1-P3 成果。

### P5 — 汇总
修复前后总表 + 新视频 + README 更新 + 最终 PR。

## 纪律
- microbench(bench_smallm_decode)不碰 27B;
- 每个引擎改动过:tiny 基线不变 / make tests / spec_sample / dit oracle /
  27B v217 gate / PPL 数值对照;
- 数值正确性优先于速度:任何 kernel 改动先过真实张量对拍
  (blk.0.attn_q 2048 行 + lm_head V=151936 行,SDOT/NEON/scalar 三方一致)。

## AMX 探测结果(P4 阶段完成)

BNNS int8(Accelerate 内部走 Apple AMX 矩阵单元)探测结论(工具:
`stratum/native/bench_bnns_int8.c`):

| 测量 | 吞吐 | 正确性 |
|---|---|---|
| i8×i8→f32, K=2048, N=2048(单线程) | **260-280 GB/s** | 逐位正确 |
| i8×i8→f32, K=32 分组调用, N=2048 | 83.9 GB/s(0.8µs/call) | 逐位正确 |
| 对照:手写 SDOT kernel(单线程) | ~20 GB/s | — |
| 对照:llama.cpp 全核 | ~230 GB/s | — |

结论:AMX 硬件吞吐是手写 NEON/dotprod 内核的 ~13×,单线程即打平 llama.cpp
全核。**"超越"的硬件路线成立。**

集成约束:k-quant 的 per-row × per-group scale 无法直接映射到 BNNS 的
per-tensor data_scale。可行架构(下一阶段):
1. nibble→int8 无损解包(Q4_K 的 4bit 值 0..15 直存 int8,scale 分离 —— 值
   不变,合规)+ 每 32 列一组调 BNNS + NEON 侧乘 per-row scale 累加;
2. 或手写 AMX 指令(corsix/amx 风格编码)获得完全 scale 自由。
临时 int8 缓冲属激活级内存(用完即弃),不触犯内存边界。

## 本轮已完成(已验证)
- rmsnorm NEON(f64x2):tiny gate OK,PPL 不变
- attention QK/V NEON、rope 查表、Q6_K fused、SDOT xscale 索引修复(前轮)
- harness EOS 口径 bug 修复 + 全矩阵重测(前轮)
