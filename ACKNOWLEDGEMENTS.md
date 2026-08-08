# Acknowledgements & Full-Dimension Comparison

> **Date**: 2026-08-08
> **Purpose**: This project did not emerge from a vacuum. The ideas, engineering patterns, and honest failures of several outstanding open-source projects shaped its design. We owe them credit — and we publish this full-dimension comparison so that both the debts and the gaps are visible. Without comparison there is no honest measurement of difference.
>
> **Methodology note**: this comparison is about **technical mechanisms**, not about which model each engine happened to be benchmarked on. Where we state our own capability, we extrapolate from the measured anchor (27B dense, Apple M4 Pro 24 GB) through the engine's **mechanism parameters** — working-set independence, per-byte cost, multi-stream sharing — because those parameters are model-size invariant. Any machine with enough disk can run any model size through this engine; the table below says *how the mechanism behaves*, not what we happened to run.

---

## 1. Acknowledgements — Projects That Inspired This Work

We are grateful to the following projects for their ideas, engineering, and willingness to publish both results and failures. Each one taught us something concrete, and our implementation is heavily indebted to them.

### [AirLLM](https://github.com/lyogavin/airllm) — Layer-by-layer streaming inference
- **The lesson we took**: working-set requirements should scale with the *layer size*, not the *model size*. AirLLM proved a 70B model runs on a 4 GB GPU by streaming one layer at a time — memory is a time-shifting problem, not a capacity problem.
- **What we built on it**: our engine is the same philosophy pushed to its limit: instead of streaming *one layer* through a staging buffer, every tensor is read directly from the mapped file (double-buffered prefetch, I/O overlapped with compute). Working set is ~7 MB wired regardless of model size.

### [ds4 / DwarfStar](https://github.com/antirez/ds4) — Narrow, specialized inference engines
- **The lesson we took**: a specialist engine narrow by design beats a generalist runner; decode-graph capture eliminates per-token scheduling; routed-expert LRU turns "capacity mode" into a dial; **exactness where it matters** is a design principle.
- **What we built on it**: MTP-tree decode batching (8 tokens per forward at full acceptance), the deterministic bit-exact gate, and the discipline of rejecting paths that fail measurement.

### [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) — 2.78T parameters on 8 GB RAM
- **The lesson we took**: the deepest expression of "weights are a stream, not a resident" — O_DIRECT bypassing the page cache, per-layer contiguous packing with a single pread, consuming packed data without dequantizing (MXFP4 LUT), and a **bit-level determinism contract** (double accumulation, no FMA, fixed reduction order).
- **What we built on it**: our own **Q2_K nibble layout** (type-42 GGUF) — we questioned the 2-bit packing format itself instead of polishing a kernel on top of it, exactly as k3 questions every byte it reads. Our `F_NOCACHE` cold-layer path and k3's O_DIRECT converged independently on the same insight: **page cache is an asset when hot, a tax when cold**.

### [flash-moe (Alexintosh)](https://github.com/Alexintosh/flash-moe) — Pure C/Metal MoE on Apple Silicon
- **The lesson we took**: a hand-tuned Metal pipeline — SSD expert streaming with `pread()` + GCD, FMA-optimized dequant kernels, fused SwiGLU/RMSNorm/attention/rope shaders — is what "small laptop, giant model" actually requires.
- **What we built on it**: our Metal layer (q2k/q4k/q5k/q6k kernels, batched-`B` variants, group dispatch for SSM projections and FFN gate/up, batched zero-copy submission measured at **17.3×** vs per-matmul waits), and our zero-copy direct-read investigation (V54, revisited) — the same mmap-NoCopy trade-offs flash-moe navigates.

### [HuggingFace transformers](https://github.com/huggingface/transformers) — The reference framework
- **The lesson we took**: a clean unified model-definition API is how a framework becomes ubiquitous; studied as the reference for architecture handling and checkpoint conventions.

### Also consulted
- **[tessera](https://github.com/geoph9/tessera)** (local copy studied): proved with `newBufferWithBytesNoCopy` + `madvise(MADV_DONTNEED)` eviction that Metal zero-copy reads of file-backed mmap pages do **not** necessarily wire memory — overturning our earlier V54 conclusion and re-opening the GPU direct-read path.
- **ggml / llama.cpp** (tooling): the block quantization formats (Q2_K/Q4_K/Q5_K/Q6_K) and the conversion/quantization toolchain used for benchmarks.

---

## 2. Full-Dimension Comparison — Technical Mechanisms

### 2.1 Memory Architecture (the defining axis)

Every engine falls into one of three families on this axis. **This is the axis that decides everything else.**

| Family | Working set scales with… | Engines |
|---|---|---|
| **Resident** | model size (must fit in RAM/VRAM) | llama.cpp full-load, MLX, vLLM GPU-resident |
| **Layer/step-wise streaming** | one layer's size | AirLLM (one layer on GPU), ds4 (non-routed resident + expert LRU), flash-moe (K active experts per layer) |
| **Whole-model streaming** | **nothing** — every byte streams per token, working set = fixed overhead | **Stratum**, kimi-k3-in-c |

| Dimension | **Stratum** | **AirLLM** | **ds4** | **kimi-k3-in-c** | **flash-moe** |
|---|---|---|---|---|---|
| Working-set model | **Model-size invariant** (~7 MB wired overhead + KV state; measured on 27B dense) | One layer on GPU at a time | Non-routed weights resident; expert slots = a memory dial | Peak RSS 8.24 GB at any model size (dial up to 224 GB) | K active experts per layer (~6.75 MB each) |
| Theoretical ceiling | **Any size, any machine with disk** — same mechanism runs 27B and 2.78T; only per-token time grows with model bytes | Any size on a low-end GPU (layer-granular streaming) | Model must fit non-routed part + budget; distributed for larger | 2.78T proven (1.56 TB disk, 8 GB RAM) | MoE-only; up to 397B shown |
| Granularity of streaming | **Tensor** (finest: each matmul reads its own bytes) | Layer | Expert (routed) | Layer (packed trunk, 1 read/layer) + expert on demand | Expert |
| I/O path | `mmap` page cache for hot; **`F_NOCACHE` + pread** for cold | Disk→GPU staging | SSD streaming + LRU | **O_DIRECT** (bypasses page cache) | `pread()` + GCD, trusts OS cache |

**The shared insight (Stratum × k3)**: both engines decouple working set from model size *completely*. k3 proves 2.78T on 8 GB; our mechanism has the same property — the reason we state our 27B run is that it is the *anchor*, not the ceiling.

### 2.2 Streaming Throughput — Theoretical Model

The mechanism's per-token cost is: **every token consumes the whole weight stream once**.

```
t_per_token ≈ W_bytes × ( f_hot / BW_hot + f_cold / BW_cold )
tok/s        = 1 / t_per_token
aggregate    = tok/s × N_streams        (N identical streams share one weight read)
```

where `W_bytes` = model weight bytes (quantization-dependent), `f_hot/f_cold` = fraction of weights resident in page cache, `BW_hot` = hot effective bandwidth (measured anchor: ~27 GB/s on M4 Pro 24 GB), `BW_cold` = cold streaming bandwidth (SSD / O_DIRECT path).

| Scenario (mechanism behavior, extrapolated) | Stratum (this engine) | kimi-k3-in-c (reference) |
|---|---|---|
| 27B dense mixed-quant (11.98 GB) hot | **~0.44 s/token** (measured anchor) | — (n/a) |
| 2.78T (≈1.2 TB mixed-quant, hot where memory allows) | ~44 s/token (linear extrapolation) | 26.5 s/token @ 8 GB RAM (their measured cold-disk case) |
| 2.78T on an 8 GB machine (cold disk streaming) | same order as k3: whole stream from disk per token | 26.5 s/token (measured) |
| 64 identical streams | **130 tok/s aggregate measured** (weight stream read once, folded) | n/a (single stream) |

The point is not that our numbers are better — it is that **both engines obey the same law**: per-token time ∝ weight bytes, working set ∝ 0. Where they differ is the cold-path mechanism (O_DIRECT vs `F_NOCACHE`-pread) and the multi-stream folding, which is a mechanism k3 does not have.

### 2.3 Kernel & Compute Design

| Dimension | **Stratum** | **AirLLM** | **ds4** | **kimi-k3-in-c** | **flash-moe** |
|---|---|---|---|---|---|
| Instruction path | NEON: Q2_K/Q4_K/Q5_K/Q6_K; **type-42 Q2_K nibble layout** (vand-4bit unpack, 2.2× measured) | torch ops (not hand-tuned) | MXFP4 CUDA, asymmetric expert quant, tensor/pipeline parallel | MXFP4 LUT — packed data consumed **without dequantization** | Hand-tuned Metal dequant GEMV, FMA-fused (+12–34%) |
| Unpack strategy | Re-arranged bit layout to halve unpack instructions (values unchanged, not requantization) | n/a | n/a | LUT, byte-frugal | FMA fusion of dequant+multiply |
| Decode batching | **MTP tree: 8 tokens/forward @ 100% acceptance**; MULTISEQ stream folding | none | decode-graph capture (scheduling eliminated) | single token | deferred expert compute (GPU overlaps CPU) |
| Parallelism | 14-core NEON (Q4_K 11.7× scaling), Metal batched-B group dispatch | single GPU stream | CUDA TP + pipeline over TCP | CPU threads | Metal GPU |
| Determinism | **Bit-exact gate** (greedy path identical to reference, verified) | not guaranteed | "exactness where it matters" | **Bit-level contract**: byte-identical output on any machine (double accum, no FMA) | no formal contract |

**Where we differ from each one, in mechanism terms**:
- vs **AirLLM**: we stream at tensor granularity, not layer granularity — our working set is smaller (no per-layer staging buffer) and I/O overlaps at finer resolution.
- vs **ds4**: we have no decode-graph capture (we eliminate scheduling via the MTP tree instead) and no distributed backend — ds4 is ahead on both.
- vs **k3**: same whole-model-streaming law, but we add (a) tree batching, (b) multi-stream folding, (c) format-level unpack reduction (nibble) where k3 uses LUT — and we are behind on cross-machine bit-identical determinism (our gate is same-machine) and on O_DIRECT cold-path independence.
- vs **flash-moe**: we are behind on GPU-layer fusion completeness (their attention/rope/norm are fully on-device; our SSM recursion GPU-ization is unfinished); ahead on dense-model handling and tree decode.

### 2.4 Correctness & Quality Philosophy

| Dimension | **Stratum** | **AirLLM** | **ds4** | **kimi-k3-in-c** | **flash-moe** |
|---|---|---|---|---|---|
| Quality stance | No engine-side quantization or approximation; nibble = byte reorder (values 0–3 unchanged) | No quantization by default (optional, weights-only) | Asymmetric: quality-critical parts untouched | Full-precision math on packed reads | 4-bit production; 2-bit only for cold experts |
| Determinism | Bit-exact gate (same machine) | none | exact decode batching / tool replay | **cross-machine byte-identical** | none |

---

## 3. Honest Gap Analysis — Mechanism-Level

| Dimension | Who is ahead | The mechanism gap |
|---|---|---|
| **Scheduling elimination** | ds4 | Decode-graph capture removes per-token scheduling entirely; we reduce it via the MTP tree but still pay per-layer dispatch. |
| **Distributed inference** | ds4 | CUDA tensor/pipeline parallelism over TCP; we are single-device (single process). |
| **Cold-path page-cache independence** | kimi-k3-in-c | O_DIRECT gives predictable cold bandwidth regardless of system memory pressure; our cold path uses `F_NOCACHE`-pread but the hot path still leans on page cache. |
| **Cross-machine determinism** | kimi-k3-in-c | Bit-level contract makes output byte-identical on any hardware; our gate is same-machine bit-exact. |
| **GPU-layer fusion completeness** | flash-moe, ds4 | Fully fused on-device layers (attention/rope/norm/activation, one launch); our Metal path covers matmuls + group dispatch — the "whole layer, one wait" pipeline is unfinished (SSM recursion is the hard part). |
| **Packed-data direct consumption** | kimi-k3-in-c (LUT, no dequant) | We halve unpack cost via nibble layout but still dequantize to FP32; LUT-based consumption is a step further. |

### Where we believe we lead (mechanism-level)

- **Working-set independence at tensor granularity**: ~7 MB wired for any dense model size — the finest streaming granularity in the set (others are layer- or expert-granular).
- **Tree-structured decode**: 8 tokens per forward at full acceptance — a batching mechanism the comparison set lacks.
- **Multi-stream weight-stream folding**: N identical streams share one weight read (130 tok/s aggregate measured at N=64) — a throughput dimension none of the others expose.
- **Format-level unpack reduction**: the type-42 Q2_K nibble layout is, to our knowledge, the first re-arrangement of a 2-bit packing format (values unchanged) that halves unpack instructions — attacking the format rather than polishing kernels on it.
- **Questioning the hot-path assumption**: like k3's O_DIRECT, our hot/cold split (`F_NOCACHE` for cold) treats page cache as an asset when hot and a tax when cold — measured and documented.

---

*This document will be kept up to date as the project evolves. If you believe a comparison entry is inaccurate or a credit is missing, please open an issue — accurate attribution and honest measurement are part of the project's contract.*
