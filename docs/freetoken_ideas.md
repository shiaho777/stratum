# FreeToken → Stratum 借鉴改进清单

> 来源:对 `~/Downloads/FreeToken-main`(FlashML-org,Apache-2.0,arXiv:2608.16157)的代码级研读。
> 本文每一条都标注:FreeToken 里的原始机制(文件引用)、Stratum 现状、建议做法、硬边界合规性、工作量(S/M/L)。
> FreeToken 面向 NVIDIA 消费卡(CUDA/Triton/PCIe),**没有一行代码能直接复用**;可借鉴的全是机制与教训。

---

## FreeToken 是什么(30 秒版)

边端 MoE 服务引擎:290B+ MoE 跑在游戏 PC 上。三大支柱:
1. **带宽自适应 CPU–GPU 协同执行**(q★ 策略):一个 decode step 里未命中的专家,按" PCIe 取回一部分上 GPU 算 + 其余交给 CPU GEMV,两边同时完工"切分;
2. **全局 LRU 专家缓存**(GPU slot cache + pinned host banks)+ **全层双缓冲 prefill 流水**;
3. **FTW 权重格式**(4096 对齐、O_DIRECT 友好)+ 工具调用锚点 KV 复用 + OpenAI 兼容服务层。

成熟度:权重流式栈与 offload/hybrid 机制是生产级(注释里全是实测数字);但无 PR CI、MTP 解析了却没用、深度依赖 vllm/flashinfer 私有符号。

---

## P0 — 值得马上排期的三件事

### 1. MoE 流式架构支持(旗舰能力)— Phase A 已落地:tiny 生成器 + arch + oracle 全对齐

- **来源**:FreeToken 的核心前提——MoE 的"活跃专家"天然是流式单位;其 LRU 专家缓存、路由统计(oracle-LRU 上界思想)、双缓冲全部围绕这一点。
- **已做 (2026-08-27)**:
  - **`make_tiny_model.py --arch moe`**(llama.cpp 专家命名:`ffn_gate_inp` / `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`,堆叠张量 `[K_in,N_out,E]`):3 MB Q4_K / 1.2 MB F16,门禁不依赖大模型;
  - **`stratum_arch_moe.inc.c`**(新注册 `llama-moe`,零改动进 Makefile 自动收集):dense 注意力与 llama arch 同构;FFN = router softmax → top-k(tie 取低 id,确定性)→ 加权专家和。**流式机制**:选中专家的连续字节切片在 dispatch 前发 MADV_WILLNEED(page-cache 提示、可回收、不锁页——合规),后续层稠密计算与缺页重叠;
  - **`STRATUM_MOE_STATS=1`** 路由诊断探针(FreeToken decode_routing_stats 的借鉴):每步去重专家数/累计触达,为预读策略调优提供测量基础;
  - **独立 oracle**(`tools/tiny_moe_oracle.py`,numpy 从文件权重复算):Q4_K 与 F16 两条路径的贪心序列均与引擎**逐位一致**(q4k:`987 647 804 804 804 16`;f16:`63 69 192 133`)。
- **验证中的语义发现**:① GGUF F32 路由按 cblas RowMajor [E][H] 读,生成器 flat 写入即此布局;② `-O3` 会把探针 sink 当死代码消除(volatile 全局救回);③ f16 模型的专家也是 F16(不是 Q4_K)。
- **下一步(Phase B)**:真实 MoE 模型(Qwen3-30B-A3B 级)接入验证;top-k>2/MULTISEQ 批量路径;专家命中率先行统计 → 决定是否值得把 #3 的 qstar 校准接到专家预读;v-gate 门禁脚本新增 moe 场景。

### 2. 工具调用锚点 + recurrent-state 快照 ✅ 已落地 (2026-08-26)

- **来源**:FreeToken special-token checkpointing(`scheduler/cache.py:144-169`):agent 改写上下文时 tool-call 开启 token 之前的前缀不变,在该点快照 = 改写后仍可复用的最深点。GDN/recurrent 状态无法按 token 截断重算,快照是 hybrid 模型唯一复用手段。
- **Stratum 现状(实现前)**:已有 Phase 3b `STRATUM_PRESERVE_KV` 单槽机制——prefill 结束存 SSM(conv/rec)快照,下请求按 `N_REUSE` 回拨复用;KV 本体不拷贝(回拨 kv_len,prefill 覆写后缀自愈)。缺"回到更早时点"能力。
- **已做(V220)**:
  - **多槽锚点环**(8 槽,`q35_ANCHOR_SLOTS`):槽 0 保留 Phase 3b 原语义;槽 1–7 由新服务器动词 **`ANCHOR STAMP=<n>`** 盖章(把请求解码后的活状态+kv_len 存入指定槽,匿名内存代价 = 8×SSM 状态量,有界);
  - **锚点恢复严格相等**:`N_GEN N_REUSE:SLOT=<n> tok1 ...` 只允许恢复到盖章时的精确长度且必须带 ≥1 个新 token(SSM 状态位置绑定不可截断;零新 token 会采样到置零 logits——防护因测试中实踩而加);
  - **顺手修了一个继承性 bug**:prefill 尾部 logits 取槽用绝对下标 `(n_prompt-1)%pf_B`,带 n_reuse 的请求只跑了后缀批、相对槽位错开 → 恢复场景采到全零旧缓冲(sampled=0)。改为按本次实际执行的批计算。
- **验证**:等价性 harness(临时脚本)证明锚点恢复续跑与从头单发贪心序列逐位一致(两个探针 seed);slot-0 遗留契约同样逐位一致(修复受益);tiny 默认路径不变;27B v217 门禁 GATE PASS。语义注记:盖章时点的已提交序列 = 该请求 prompt+[prefill 采样]+argmax[:-1](最后一个 argmax 未回喂),kv_len 与之严格相等——上层按此构造前缀。
- **未做**:tool-call opener 的自动检测(FreeToken 从 HF tokenizer/模板读单 token opener id)。这属于服务层职责(边界 4:引擎不内嵌 tokenizer 特判);当前协议是显式动词,由上层(Python backend / 未来 HTTP 层)在正确时机发送。

### 3. 开机带宽校准 + 自适应执行策略(q★ 思想)— 探针+引擎校准已落地,路由消费待 A/B 数据

- **来源**:FreeToken `moe/benchbw.py` + `engine.py:641-670`。要点三个:
  1. 切分比例 q★ 用**并发竞争下测得**的带宽对(pcie_ov/(pcie_ov+cpu_ov)),不是各自单独测——单独测会高估(DMA 与 CPU 抢 DRAM);
  2. 校准一次、按 GPU UUID 存 profile,生产路径零开销;
  3. 整数取整陷阱:3 个 miss 的 41.5% 应取 1 不是 2(ceil 过取会让慢侧慢 1.6×)。
- **Stratum 现状**:hot/cold 用 mincore 检测,但 CPU vs GPU2 vs GPU-NC 的路由是**静态规则**(`q35_gpu2_should_use`:mincore 6 页采样 + 60% mostly-hot 阈值);没有"测量驱动的运行时路由"。
- **已完成 (2026-08-26)**:
  - 探针 `stratum/native/bench_bandwidth.c`(gitignore 按仓库惯例不收 bench_*)。方法修正:① hot 必须逐缓存行触摸;② F_NOCACHE 在 darwin 不可靠(18 GB/s 假数),真冷读=msync(MS_INVALIDATE) 驱逐后 pread。512 MB 实测:hot 85–91 GB/s,cold 4.19–4.50 GB/s,2 GB 并发对测双方 −22%/−31%,qstar≈0.04–0.05。
  - **引擎集成第一阶段**:`stratum_engine.h` 新增 `stratum_adaptive_calibrate()`(V219,共享基建),qwen35 init 挂点在 RAM 横幅之后;`STRATUM_ADAPTIVE=1` 时跑 ~1.5 s 短校准(128 MB/0.4 s 相位,TMPDIR 临时文件用完即删),把 `ADAPTIVE_PROFILE v1 ...` 行打进每次运行的配置记录。实测 hot 105 / cold 3.9 / pair 97/3.5 GB/s、qstar=0.035,与独立探针一致(128 MB 工作集略偏热)。踩坑记录:校准循环的 sink 被 -O3 当死代码消除,必须落 volatile 全局。
  - 验证:tiny 默认路径逐位一致;ADAPTIVE=1 输出不变;27B v217 门禁带 ADAPTIVE=1 通过;ENVVARS.md 重生成(202 变量)。
- **刻意不做**:路由消费(`q35_gpu2_should_use` 的阈值/HOT_Q2K 门改由测量驱动)**没有实现**——没有端到端 A/B 数据就改路由违背本项目"实测而非假设"的纪律。设计已在上面写明:每层可并流字节 ≤ qstar×层字节数;需要 MULTISEQ 冷热混合场景的 e2e 对比先行。

---

## P1 — 明确有收益、可跟进

### 4. 对齐权重布局(Stratum 版 FTW)

- **来源**:FTW 格式的本质是**让每个张量 4096 对齐、shard 按 16KB/4096 边界切割**,从而使任意张量可零拷贝直读(`checkpoint/ftw.py:52-66`);"flat 区间用其对齐包络窗口+头偏移视图读"(`ftw.py:507-526`)。
- **Stratum 现状**:GGUF 张量偏移只保证 GGUF 自身 alignment(常为 32);GPU-NC 的 per-tensor NoCopy 对地址/长度有页对齐要求,不对齐的张量要么降级要么处理头部偏移。
- **建议做法**:可选转换工具(与 `tools_gguf_nib_convert.c` 同类):所有张量起点 16384 对齐、pad 到边界、**同一层的张量连续摆放**(单层=一段顺序字节,冷读局部性最优)。与 type-42 同属"字节重排、值不变",显式合法。
- **工作量**:S–M。

### 5. Prefill 双缓冲推广

- **来源**:FreeToken 全层双缓冲(`offload_cache.py:563-798`):缓冲区直接借自 LRU 缓存的头 2·E 个槽位(**零额外显存**);prefetch(layer)、prefetch(layer+1)、事件等待/释放的三步舞;命中部分 D2D 收拢、只有 miss 过 PCIe 一次批量拷贝;<256KB 的小块整层搬。
- **Stratum 现状**:ASYNC_PREFETCH 是 decode 路径机制(qwen35 arch 内 5 处调用点);madvise WILLNEED 有层预读但未见 prefill chunk 边界的 next-chunk 重叠编排。
- **建议做法**:把"处理第 i 块时预热第 i+1 块"的编排加到 prefill 循环;"缓冲区借用主缓存"的思想对应到我们是**零成本**的(预热的就是 page cache);hit/miss 分离对应"已热区间跳过 madvise,mincore 先查"。
- **工作量**:S–M。收益取决于 prefill 在目标负载里的占比,先用探针量一下再动手。

### 6. 性能报告规范补丁(功率墙陷阱)✅ 已落地 (2026-08-25)

- **来源**:FreeToken 血泪注释:spin-wait 内核把利用率打到 99%,**功率耦合的笔记本 CPU 被压频,净负优化**(`cpu_executor.py:40-43`);worker 绑物理核(SMT 不加带宽);协调者独占一核。
- **已做**:`stratum/tools/thermal_report.sh`(免 sudo 读 `pmset -g therm`)接入 `run_all_gates.sh` 头部;AGENTS.md 测试节 6 增加热状态记录要求。引擎侧发现 parpref 线程已有 UTILITY QoS(V37),补齐了缺失的 ASYNC_PREFETCH 线程(`q35_prefetch_worker`,V218 注释);tiny 模型 A/B 贪心序列逐位一致,且 **v217 门禁在真实参考模型(model-27b-gguf/qwen3.6-27b-mixed.gguf,已确认就是钉死 [2,220,16,13] 的模型)上完整通过**(bitexact ✓、tok/main=8.0、mains=1)。
- **建议做法**:① `run_all_gates.sh`/perf 报告前记录 thermal 状态(`powermetrics -n1` 或 `sysctl machdep.xcpm`);② ASYNC_PREFETCH/预读线程用 `QOS_CLASS_UTILITY`,计算线程保持默认(等价于 P/E-core 分工);③ 写进 AGENTS.md 测试节。
- **工作量**:S。零风险,防止未来"看起来快了其实被压频骗了"。

### 7. fadvise DONTNEED 归还工具 ✅ 已落地 (2026-08-25/26)

- **来源**:FreeToken 每个 shard 读完后 `posix_fadvise(DONTNEED)` 防止 checkpoint 页挤掉驻留专家(`models/loader.py:52-62`)。
- **已做**:`stratum/tools/drop_page_cache.py`。实测发现 macOS 不导出 `posix_fadvise`、MADV_DONTNEED 对共享文件映射是空操作;**真正生效的 darwin 逐出机制是 `msync(MS_INVALIDATE|MS_SYNC)`**(受控实验:2 GB 驻留 → 0)。mincore() 前后计数如实报告,不臆断生效。已写入 AGENTS.md 测试节 3 作为大模型测试收尾步骤;27B 门禁跑完实测 2065 MB → 0。

### 4. 对齐权重布局(Stratum 版 FTW)✅ 已落地 (2026-08-26)

- **来源**:FTW 格式的本质是**让每个张量页对齐、shard 按边界切割**,使任意张量可零拷贝直读(`checkpoint/ftw.py:52-66`)。
- **已做**:`stratum/tools/gguf_align.py` — 输出仍是合法 GGUF(更新 `general.alignment`,读取器原生支持);张量按 token_embd → blk.N 升序 → 其他 分组,层=连续字节段;payload 原样拷贝(字节重排,type-42 同类,显式合法)。写后自动 sha256 全量比对每个张量 + 重解析校验。验证:tiny(25 张量)与真实 Qwen3-0.6B Q4_K(311 张量,481 MB,0.87 s 转换)均 payload byte-identical、引擎贪心序列与原文件完全一致。为 GPU-NC 页对齐直读和 MoE 专家粒度预读(#1)铺路。
- **Stratum 现状**:反向问题——27B 跑完 12 GB 驻留 page cache,后续测试环境变差(AGENTS.md 边界 3 要求测前查内存,但没有主动归还手段)。
- **建议做法**:`tools/drop_model_cache.sh <model.gguf>`:对模型文件区间 fadvise DONTNEED;大模型测试收尾自动调用。
- **工作量**:S。

---

## P2 — 方向正确,但前置条件未熟

### 8. OpenAI 兼容 HTTP 服务层
FreeToken 的价值一半在生态(API 直接接 Codex/Claude Code)。Stratum 的对应路线:串行调度循环 + MULTISEQ N 流,HTTP 只是把 stdin 换成 socket。**必须尊重非重入设计**(static 全局),daemon 化参考 FreeToken 的 control-message + idle-safe-point 架构。工作量 L;做完之后 #2/#9 才有真实流量。

### 9. Radix 前缀 KV 复用(跨请求)
FreeToken `kernel/radix.py` + page-aligned token-id key。多轮对话/agent 场景省整个 prompt 的 prefill。依赖 #8;离线 benchmark 可先行验证收益上限。

### 10. 运行时弹性内存重排
FreeToken 的 idle-safe-point rebuild(engine.py:769-913,先校验几何再破坏性释放,值得学的是**顺序**)。Stratum 的 KV/anon 预算本来就近乎常数,收益有限;仅在 #8 之后、KV 预算成为变量时才有意义。

---

## P3 — 记录备查,暂不动手

| 点 | 来源 | 备注 |
|---|---|---|
| DFloat11 无损 BF16 压缩(~11 bit/weight,bit-exact) | `kernel/triton/df11.py` | 值不变→合法,但 Huffman Metal decoder 成本高;BF16 非 Stratum 主打格式 |
| 多后端运行时探测+诚实降级阶梯 | `nvfp4_backends.py:183-272` | "probe 到符号才承诺"模式可用于未来 Metal 特性开关 |
| H3 Phase 3 稀疏注意力 | `attention/dsa_indexer.py`、`dsv4_sparse.py` | Triton 实现,算法可移植 Metal;H3 NaN 解决后再看 |
| 内核缓存构建印章配对 | `kernel/utils.py:89-103` | metallib 随二进制走,暂无 JIT 问题 |

---

## 明确不借(及原因)

- **pinned host banks / cudaHostRegister / mlock 驻留分级**:违反边界 2。我们的"host 缓存"就是 page cache,免费且可回收。
- **O_DIRECT 读栈**:mmap 流式架构不需要;FTW 对齐思想的用途收敛到 #4(NC 对齐+局部性)。
- **CUDA-graph 安全性约束**(Q16 定点分数、固定 shape 集):Metal 无此问题。
- **FreeToken 的 CI 现状**(无 PR CI)与其 MTP(parsed-but-ignored):反例,勿学;Stratum 的门禁文化是优势,任何新特性(#1/#2/#3)落地时同步进 `run_all_gates.sh`。

---

## 落地顺序建议

```
#6 功率墙规范 (S) ──┐
#7 fadvise 工具 (S) ──┴─ 立即可做,顺手合入门禁脚本
#4 对齐布局工具 (S-M)     ← type-42 同类,独立可交付
#3 带宽校准 (M)           ← 探针先行,数据驱动
#2 锚点快照 (M)           ← hybrid SSM 复用刚需
#1 MoE 架构 (L)           ← 旗舰项,tiny-MoE generator 先行
#8 服务层 (L)             ← 生态入口,最后
```
