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
| BNNS int8 FC 单线程 | ~13× 手写 NEON/dotprod | 数值正确 |

结论:AMX 硬件吞吐是手写 NEON/dotprod 内核的 ~13×,单线程即打平 llama.cpp。

集成约束:k-quant 的 per-row × per-group scale 无法直接映射到 BNNS 的
单层 FC 接口。两条可行路线:
1. 保持权重逐字节不变(Q4_K 原样读,页缓存/流式特性全部
   不变,合规)+ 每 32 列一组调 BNNS + NEON 侧乘 per-row scale 累加;
2. 或手写 AMX 指令(corsix/amx 风格编码)获得完全 scale 自由。

## AMX 集成原型裁决(bench_amx_q4k.c,真实张量验证)

per-32 列连续 BNNS 分组调用(AMX)+ scale 累加。三个关键发现:

1. **BNNS 批入口被绕开时吞吐兑现**:原型管线达到 ~5× SDOT 单线程
   (对比 SDOT 单线程 ~324µs)。
2. **BNNS 跨步(strided)权重视图是坏的**:stride[1]≠size[0] 时结果错位
   —— 必须转置布局给 BNNS 连续视图(已在本原型中解决)。
3. **瓶颈是 B3(scale 累加)而非 AMX**:稳定测量下 B2(BNNS 分组调用)
   只占小头,逐组 scale 的 NEON 侧乘累加吃掉大部分收益。

下一步(下一会话):
1. B3 NEON 化(转置 partial + row-dot)→ 重测
2. BNNSFilterApplyBatch B=8/16 批摊销测量 → MULTISEQ 集成决策
3. 若批摊销成立:prefill/MULTISEQ 走 AMX 路径,单流 SDOT 保留

## 批处理路径裁决(bench_batch_q4k.c,B=16,每流成本)

```
A  单流 SDOT (T=10)         : 61.8µs
M1 neon_multix(现行默认)     : 21.2µs/stream
M2 sdot_multix_pack(未接入)  : 13.4µs/stream  ← 比现行快 1.6×
M3 blas fp32 dequant+sgemm  : 11.8µs/stream  ← 现有 AMX-fp32 标杆
M4 int8 unpack+BNNS+scale   : 27.4µs/stream
M5 fp16 dequant+BNNS        : 20.1µs/stream
```

裁决:
1. **BNNS int8 的原始吞吐是真的**(146µs/16流 = 9.1µs/stream 的纯 AMX
   段),但 Q4_K 的逐组 scale 机制迫使 prep(60µs)+unpack(56µs)
   +accum(~177µs) 环绕它——累加是逐流成本,摊不薄,结构性输给 sgemm
   (sgemm 把 scale 直接折进 GEMM)。
2. **fp16 路径同理**(dequant 138µs + BNNS 183µs)——BNNS fp16 FC 甚至
   不如 fp32 sgemm 快。
3. **真正免费的大餐是 M2**:`q4k_dot_row_sdot_multix_pack` 已存在、已
   验证(qwen35 在用)、数值与 SDOT 同源——接入 `la_linear_multix`
   (后来的 `st_linear_multix` 共享派发)后即落地。
4. BNNSFilterApplyBatch 要求批输入连续(in_stride==单输入大小),跨步
   输入直接 abort;且批模式下有连续性约束/状态 bug——**BNNS int8/fp16
   批处理端到端输给 pack+sgemm,路线证伪,不再投入**。

## nchunks 自适应(Sep-12 结论)

热权重下 ~5 线程就饱和统一内存带宽(floor-256 得 ~95 tok/s,不如固定 5
的 112);冷盘大模型需要更多线程拉满 NVMe 队列。现行默认:模型 ≤2GB 用
5,否则物理核数(`stratum_linear_init`)。
