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

## 第四轮(SDOT 覆盖扩展 + Q4_K 重排证伪)

- **Q8_0/Q5_K/Q3_K int8 SDOT 核**(新增):全部复用 q4k_quantize_x_q8
  per-32 激活量化;Q5_K 走 q4k 骨架 + qh 高位面 OR 进 u5(≤31 仍 s8),
  Q3_K 无 min 项直接 signed q3∈[-4,3] vdotq。quant_test 相对误差
  2e-5~2e-3 与 q4k sdot 同级;tiny llama --weights q8_0 端到端
  SDOT=0/1 argmax 逐位一致(顺带补了引擎的 Q8_0 embed 分支)。
  make_tiny_model.py 新增 --weights q8_0 编码器。
- **Q4_K 字节重排(type-43 设想)证伪**:微基准模拟两种布局——
  (a) 148B 块仅预解码 sc/m:v6 与 _f 完全同速(1.33×)——get_scale_min
  标量解码本来就不是热点;(b) 276B 块 nibble 预分裂:v5 名义 1.51×
  但字节 +92%,冷流式 SSD 多读一倍 → 净亏 ~40%,热场景也仅比 _f
  快 ~14%(kernel 级)。144B 约束内 Q4_K 无油水,不建 sidecar 设施。

## 第五轮(SDOT 覆盖 + MoE MULTISEQ)

- **st_linear_multix 提升为共享入口**(stratum_linear.h):llama 的
  CPU pack 路径(rows2 行对 + ≤16 流分块 + neon 回退)原样上提,
  la_linear_multix 保留 Metal/BLAS 前置分支后委托;MS16 回归
  ~264-281 tok/s 零失配。
- **MoE 架构 MULTISEQ 落地**(stratum_arch_moe.inc.c):
  moe_forward_multiseq 每流独立 KV([L][B][maxkv][Nk*Hd]),dense
  部分全走 st_linear_multix;**expert 分组批处理**——每层每 expert
  只读一次,成员流共享 pack 核;累加严格按各流 top-k 权序
  (out_all[s][rank])→ 与单流浮点序一致。E>256 动态分组表。
  验证:tiny-moe f16/q4k 两种权重,MS4/MS8/MS32 stream0 argmax
  与单流逐位一致,MS_VERIFY 全流零失配;B=1 边界正常。
- **MoE ngram spec 解码**(STRATUM_NGRAM_SPEC):moe_forward_multiseq
  泛化为 moe_forward_b(shared_kv 开关)——batch 模式 B 个 token 延伸
  同一序列(slot s 写 kv_len+s、看 kv_len+s+1,spec 验证语义);
  multiseq 模式不变。驱动循环移植自 llama ngram-spec(后缀查找
  起草 + argmax 链验证,greedy bit-exact by construction)。验证:
  tiny-moe f16/q4k spec 序列与单流逐位一致;重复 prompt 上草稿
  命中率 2.0 tok/call;KV 满自动回退单 token。

## 第六轮(qwen35 单流缺口补齐)

排查发现 27B 主架构的单流路径没吃到这轮核红利——它走 `multix_preq`
B=1(pack 打包开销 + 旧逐流核)。已修:

| 改动 | 内容 | 验证 |
|---|---|---|
| qwen35 B==1 快道 | q4k→`_sdot_f`(1.32× 向量累加核直连,保留预取);q6k 走 pack 布局(见下) | tiny_q35_q4k argmax+logits 逐位一致 |
| qwen35 q5k/q3k/q8_0 | 单流从纯 neon 接上新 SDOT `_f` 核 | tiny_q35_q80 端到端 SDOT=0/1 一致 |
| qwen35 embed | 补 Q8_0 分支(真实模型会踩到) | 同上 |
| 单流 rows2 交错核 | **证伪**:18.7 vs `_f` 19.9 GB/s——vdotq 计算受限非 load 受限 | 微基准 |

**Q6_K 尺度布局守卫**(本轮排掉的 NaN 根因):prequant 池产 per-32
激活尺度,而 `q6k_dot_row_sdot*` 按 per-16 索引——B==1 直连
`_sdot_fused` 会越界读 scale 产生 NaN。最终处置:q6k 的两个调用方
(`q35_linear_q6k`/`q35_linear_q6k_multix`)只在 `q35_g_xq_pack_ready &&
xq_B==B && xq_K==K` 时进 preq——pack 核内部 b32=g/2 映射与 per-32 池
兼容;pack 不可用时回退 NEON,绝不把 per-32 池喂给 per-16 核。

quant_test 16/16、spec_sample 4/4 全绿。
