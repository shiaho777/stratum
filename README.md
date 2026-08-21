# Stratum

[English](README.md) | [简体中文](README_CN.md)

A pure-C transformer inference engine for Apple Silicon with a wired-memory footprint that is **independent of model size**. Weights are a stream, not a resident: the engine `mmap`s the model and reads it through the OS page cache, so anonymous memory stays ~7 MB whether the model is 0.5B or 27B.

| Metric | Value |
|---|---|
| Models measured | 27B dense (Qwen3.6, 11.98 GB GGUF), down to 0.5B |
| Anonymous (wired) RAM for 27B | **~77 MB** (incl. KV/SSM state) |
| Anonymous RAM for 0.5–1B | ~7 MB |
| vs llama.cpp (TinyLlama 1.1B) | **85.7× lower** anonymous RAM |
| Engine size | ~790 KB binary, ~25k lines C/Metal |
| GPU required | none (integrated, optional Metal accel) |
| Architecture support | Llama family + Qwen3.5/3.6 hybrid (Gated DeltaNet SSM + attention) |

The binding constraint is disk and bandwidth, not RAM: a model runs to completion as long as it fits on disk. What RAM buys is *how much of the model stays hot in page cache* — more RAM, fewer SSD reads per token, higher throughput. Nothing else changes.

---

## Part I — Getting started

### Requirements

| Component | Requirement |
|---|---|
| CPU | **Apple Silicon (ARM64) required.** The hot path is hand-written NEON SIMD, which exists only on Apple Silicon. Intel Macs (x86_64) are not supported. Any M1/M2/M3/M4 chip works, including base/Pro/Max/Ultra. |
| GPU | None to buy. The GPU is integrated into the SoC and exposed through Metal; the engine runs fully on CPU and uses it only for optional acceleration. |
| RAM | **No practical minimum — any modern device runs it.** The engine's anonymous need is ~7 MB (small models) / ~77 MB (27B, incl. KV/SSM state); weights live in reclaimable page cache, so RAM does not decide whether a model runs, only how fast. 1 GB is the modern-device baseline; every Apple Silicon Mac ships with ≥8 GB. |
| Disk | ≥ model file size, NVMe SSD recommended. Cold weights stream from disk; ~3 GB/s sequential read keeps the cold path usable. |

Memory bandwidth is the dominant performance lever, not core count. Apple Silicon bandwidth grows with chip tier:

| Chip tier | Memory bandwidth (approx.) |
|---|---|
| M1 / M2 / M3 base | ~70–100 GB/s |
| M1 Pro / M2 Pro | ~200 GB/s |
| M4 / M4 Pro | ~120 / 273 GB/s |
| M3 Max / M4 Max | ~400 / 546 GB/s |
| Ultra (dual Max) | ~800+ GB/s |

### Minimum hardware for 5 tok/s

"Runs" and "runs fast" are different questions. Minimum **5 tokens/s** per model tier, derived from the per-token time formula `W × (f_hot / BW_hot + f_cold / BW_cold)` and anchored to measured hardware:

| Model | Minimum for 5 tok/s | Why |
|---|---|---|
| 1B (Q4_K, ~688 MB) | **Any Apple Silicon (M1 base or newer) + ≥1 GB RAM + SSD** | 5 tok/s = 200 ms/token = 688 MB / 0.2 s ≈ **3.4 GB/s** effective bandwidth. Every M-chip (~70 GB/s+) and every SSD (~3 GB/s+) clears this by an order of magnitude — measured 68 tok/s on M4 Pro, i.e. ~20× headroom at 1B scale. |
| 27B (Q2_K/Q4_K/Q6_K, 11.98 GB) | **M4 Pro-class (~250 GB/s) + ≥16 GB RAM (model mostly hot) + NVMe SSD** | 5 tok/s = 200 ms/token = 11.98 GB / 0.2 s ≈ **60 GB/s** effective bandwidth. Decode uses ~25–30% of nominal bandwidth, so the chip needs ~250 GB/s nominal — the M4 Pro tier (273 GB/s). Below M4 Pro, even fully hot, M2 Pro (~200 GB/s) tops out near 4 tok/s. RAM ≥16 GB keeps the model mostly resident in page cache (measured anchor: M4 Pro + 24 GB = 5.73 tok/s hot short-run). |

A model *runs* on far less than this: on the same machine, the 27B streams fully cold at ~0.2–0.3 tok/s from an M1 base with 8 GB. The table above is the threshold where throughput stops being a waiting game.

### Quick start

```sh
cd stratum/native
make                  # builds ./stratum (+ stratum_q4k.metallib for GPU paths)
make tests            # quant-kernel cross-validation + sampler exactness

# smoke test — any small GGUF (Qwen, Llama, TinyLlama, ...):
./stratum <model.gguf> 2 1 450 7483 310 3444 338
```

No model is shipped, bundled, or assumed: the engine and every script require a model path as an explicit argument. The binary reads `general.architecture` from the GGUF metadata and dispatches to a registered handler — no model names are hardcoded.

Build options: MemX (`github.com/shiaho777/memx`, MIT) is an optional compressed-memory runtime backing staging buffers and KV/SSM state — it is **fetched automatically** on first use (`make deps` clones/pulls it; a plain `make` does so when needed) and can be disabled with `make USE_MEMX=0`. `make USE_METAL=0` builds CPU-only without the Metal shader library.

### Usage

```
stratum <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]
```

- `<model.gguf>` — path to any supported GGUF (required)
- `N_GENERATE` — number of tokens to generate (default 32)
- `PROMPT_TOKEN_ID...` — pre-tokenized prompt; omit to read raw tokens from stdin

Examples:

```sh
# CPU-only, greedy, benchmark style (wall = load + prefill + 64 tokens):
STRATUM_NO_GPU=1 /usr/bin/time -p ./stratum <model.gguf> 64 0 1

# GPU path (auto-detects stratum_q4k.metallib from CWD or native/):
STRATUM_GPU=1 ./stratum <model.gguf> 32 1 450 7483 310 3444 338
```

The Metal library is auto-detected (current dir, then `native/`); the Metal device is auto-detected (`MTLCreateSystemDefaultDevice`). See [Part V — Environment variables](#part-v--reference) for the full switchboard.

### Reading the run report

The engine prints a diagnostic log to stderr. The three numbers that matter:

- **`wired` / anonymous memory** — the core claim. ~7 MB for small models, ~77 MB for 27B (includes KV/SSM state). This is the number that does **not** grow with model size.
- **`tok/s`** — decode throughput. For `MULTISEQ`, the aggregate line reads `aggregate X tok/s (per-stream Y tok/s)`.
- **`argmax` sequences** — greedy token ids; used by the gate scripts to assert bit-exactness.

### Common questions

- **Why is my machine swapping / slow?** The model may be cold; the first pass streams from disk. Check free memory before big runs (`vm_stat`, `sysctl vm.swapusage`). On a 24 GB machine, do not run a 12 GB model while heavy background processes are active.
- **Does it use my GPU?** Only if `STRATUM_GPU` / `STRATUM_GPU_NC` is set. Default is CPU-only.
- **Can it run a model bigger than RAM?** Yes — that is the design point. It needs disk space and patience, not RAM.
- **Why is large-model decode slow?** Every token reads the whole weight stream once (11.98 GB @ ~27 GB/s hot ≈ 0.44 s/token on M4 Pro). Decode is bandwidth-bound by construction; see [Part IV — Mechanism limits](#mechanism-limits).
- **Can one process run multiple models?** No — the engine is single-model, single-process by design: state is file-scope global, so one model per process (see AGENTS.md).

---

## Part II — How it works

### The problem: dense weights are a stream

A MoE model activates a fraction of its parameters per token — kimi-k3 activates 16 of 896 experts (~3.7%), which is why it can stream 93% of its weights. A **dense** model has no such luck: every token consumes *every* weight, once. Traditional engines respond by loading the whole model into RAM/VRAM, which is why llama.cpp needs ~16 GB unified memory for a 27B and thrashes when it doesn't have it.

Stratum takes the opposite position: **working set is constant regardless of model size**. One token = one full sweep of the weight stream, and the stream never has to be resident. The result is that a 27B runs in ~77 MB anonymous RAM on a 24 GB machine — and would run identically on a 4 GB one, only slower.

### Reduction 1 — `mmap`: the weights never enter anonymous memory

The model file is mapped read-only and every weight read goes straight through the OS page cache:

- Page cache is **reclaimable** — under memory pressure the OS evicts weight pages instead of killing the process, exactly like any other file read.
- Anonymous memory is reserved only for what the model genuinely needs at runtime: activations, KV cache, SSM state. That is the ~7 MB / ~77 MB number.
- Prefetch is advisory, not resident: `madvise(MADV_WILLNEED)` + `F_RDADVISE` hint the kernel to load pages ahead (per-tensor granularity, 50–100 MB windows), then release. The pages are still just page cache.

This is why the memory floor is a *dial*, not a cliff: RAM size chooses `f_hot` (the fraction of weights the page cache can hold), and throughput follows the formula in [Part IV](#how-performance-scales).

### Reduction 2 — the streaming scheduler: hot detection, cold prefetch

Not all layers are equal. On a 24 GB machine with a 12 GB model, most weights stay hot in page cache after the first sweep. The scheduler measures this instead of guessing:

- **Hot detection** — `mincore()` samples 3–12 pages per layer; if ≥80% are resident, the layer is "hot" and runs as pure compute with no I/O waits.
- **Cold path** — cold layers get a coalesced `madvise`/`F_RDADVISE` burst for the next few layers (per-tensor windows, 24 MB gaps coalesced into one call), so the SSD keeps streaming while compute proceeds.
- **Deterministic mode** (`STRATUM_STREAM_DET=1`) skips the mincore tax entirely and treats everything as cold, for reproducibility.

The measured cost of the mincore sampling itself was ~zero on the hot path — the win is not the detection, it is knowing when *not* to prefetch.

### Reduction 3 — MULTISEQ: one weight scan, N logical streams

This is the amortization play: `STRATUM_MULTISEQ=N` runs N independent decode streams (same prompt, diverging as each samples) through **one weight scan per step**. N streams share the same tensor reads; the compute is batched on top.

- Aggregate throughput scales ~linearly with N: **130 tok/s @ 64 streams on the 27B** (measured), ~2.0 tok/s single-stream.
- The run report prints `weight-scans` vs `main-scans` — the ratio is the amortization factor. Theory line: *one cold scan serves N logical streams*.
- Same-prompt clone folding is the realistic case: N copies of one conversation at 130 tok/s aggregate = 130/N per copy, still one weight read.

### The Q2K nibble layout (GGUF type 42)

Q2_K unpack is compute-bound (~7 GB/s at 14 cores) — the dequant instructions, not the memory, are the wall. The nibble layout is a **byte re-arrangement**, not a requantization:

- Each 2-bit weight value is unchanged (0–3); only the byte packing is permuted so the NEON unpack kernel halves its instruction count.
- Measured **2.2× faster unpack**, bit-exact with the original layout.
- Cost: +50% file size for the Q2_K tensors (a 16.3 GB sidecar for the 27B), which needs ≥32 GB RAM to stay hot — hence the nibble path is opt-in via `STRATUM_Q2K_NIB=<path>` or an embedded type-42 GGUF.

The converter (`tools_gguf_nib_convert.c`) emits either a sidecar or an embedded type-42 GGUF, and the engine verifies nibble vs original values match (`[NIB-DBG]` diagnostics) before trusting it.

### MTP tree decode

Multi-token prediction: a small draft head proposes an 8-token tree per forward; the main model verifies the whole tree at once.

- **8 tokens per forward at 100% acceptance** when the draft agrees — no extra passes, output identical to greedy.
- On the 27B the wall is draft *quality*, not tree parameters: measured tree efficiency 2.46 tok/main, and tree-depth experiments (`STRATUM_TREE_EXTEND_K`) beyond 4 made it worse. The tree machinery works; the draft head is the binding constraint.

### Metal GPU: bounded buffers, per-tensor zero-copy

GPU is optional and strictly bounded:

- **Bounded staging** — GPU buffers are capped; the engine never mirrors the model onto the GPU.
- **Per-tensor NoCopy** (`STRATUM_GPU_NC=1`) — each matmul wraps *only its own tensor* (<100 MB) in a `newBufferWithBytesNoCopy` view over the mmap, dispatches, releases. Wired memory stays flat (measured +0.04 GB over an 11.98 GB sequential GPU read).
- **Batched submission** (`stratum_metal_nc_batch_*`) — independent matmuls share one command buffer: **17.3× vs per-matmul waits** (480 matmuls: 3.71 s → 0.21 s).
- **The failure mode is whole-model registration** — one NoCopy buffer over the entire mmap (or chunks > ~1 GB) wires the whole model and OOMs. That is exactly what per-tensor granularity avoids. There is also no page-cache locking anywhere: `keep_resident` is forbidden, `STRATUM_HOT_FAST` is the sanctioned hot-mode scheduler that does not lock.

Why the GPU at all, if CPU works? The memory answer is the point: GPU acceleration is a *bounded* resource, so it never changes the memory story — it can only speed up the compute portion of a bandwidth-bound pipeline.

### KV cache and SSM state

Runtime state is the only anonymous memory the model needs:

- Full-attention layers keep KV cache; the qwen35 hybrid's SSM layers (Gated DeltaNet) keep a fixed-size delta-rule state that is **independent of context length** — a bounded matrix updated in place per token, like kimi's KDA.
- The 27B's ~77 MB anon = KV/SSM state + activations + small scratch; the weights themselves contribute ~0.

### The codebase

```
stratum/native/
├── stratum.c                    ← entry: reads general.architecture, dispatches
├── stratum_arch.h               ← generic arch registry + config loader
├── stratum_linear.h             ← generic quantized linear layers (Q2_K…Q8_0/F16/F32)
├── stratum_engine.h             ← CPU/GPU init, madvise, spec decode, memory report
├── stratum_arch_llama.inc.c     ← Llama/Qwen2/Qwen3 dense arch (self-registers)
├── stratum_arch_qwen35.inc.c    ← qwen35 hybrid arch: Gated DeltaNet + attention (~24k lines)
├── stratum_metal.m/.h           ← Metal layer: GEMV kernels, batched-B, group dispatch, NC
├── stratum_q{k}_*.{h,neon.h,metal} ← quantized kernels (scalar + NEON + Metal)
├── v199–v217_gate.sh            ← bit-exact regression gates
└── Makefile                     ← builds ./stratum (+ metallib)
```

Adding an architecture = write `stratum_arch_<name>.inc.c`, implement the `StratumArch` interface, register it — the Makefile auto-collects `*.inc.c` into `stratum_archs.gen.h`, so no existing file changes. The same rule applies to models: nothing is hardcoded per-model.

### Invariants

Three hard boundaries the engine never crosses — they are the reason numbers stay honest:

1. **Never harm quality** — no requantization of existing weights (Q4_K→Q4_0 is forbidden), no skipped layers, no approximate compute. The only allowed "approximation" is int8 SDOT, verified greedy bit-exact. The nibble layout is a byte permutation, not requantization.
2. **Never grow wired memory** — no whole-model mlock, no GPU buffers caching weights, no >1 GB NoCopy registrations. Weights stream; only KV/SSM state is resident.
3. **Testing must not exhaust the machine** — memory checked before big runs; kernel experiments use micro-benchmarks, not the 27B end-to-end.

---

## Part III — Validation

The engine treats correctness as a contract, not a hope:

- **CI on every PR (3 jobs)** — build + quant kernel cross-validation and sampler exactness; the same tests under **ASan + UBSan**; and a **real end-to-end inference smoke**: deterministic tiny models are *generated* at test time (no weights in the repo) and driven through the full decode loop on **both architectures**, with the greedy sequences pinned as hard regression assertions.
- **`quant_test`** — every quantized kernel cross-validated against a scalar reference.
- **`spec_sample_test`** — Leviathan-Chen rejection-sampling exactness.
- **19 gate scripts (`v199`–`v217`)** — full-model greedy regressions on the qwen35 architecture + the 27B: assert the exact argmax sequence `[2, 220, 16, 13]` and `tok/main ≥ 8.0`. Any engine change must keep every gate passing.
- **Distribution-level regression** — `STRATUM_LOGITS_DUMP=<path>` records per-step logits; `logit_compare` reports KL(base‖candidate), top-1 agreement, and max |Δ| between any two runs. The gates pin argmax over a handful of tokens; this sees the whole distribution. (It is also how the MemX run-to-run variance documented in AGENTS.md was found.)
- **Dual-path discipline** — CPU (NEON) and GPU (Metal) paths are both exercised; per-tensor NoCopy was verified bit-exact against the CPU path on the 27B before it was allowed.

What "bit-exact" means, what is exempt (`-ffast-math` contraction across toolchains, int8 SDOT, MemX backing flips), and what re-validates it: see the determinism contract in `AGENTS.md`, and `stratum/docs/VALIDATION.md` for the full coverage matrix.

---

## Part IV — Measurements

All numbers measured on Apple M4 Pro (14 cores), 24 GB unified memory, macOS, greedy decoding, CPU path (`STRATUM_NO_GPU=1`), method `/usr/bin/time -p ./stratum <model.gguf> 64 0 1`.

### Small models — perceptually instant (0.5B–1B)

This is where Stratum is the pick of the litter: small models run at normal, interactive speed with near-zero resource footprint — the machine stays responsive to everything else while generating. Wired memory stays ~7 MB whether the model is 0.5B or 27B.

| Model | Format | Size | tok/s | Perceived |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | Q4_K | 398 MB | **125** | instant |
| Qwen3-0.6B | Q4_K | 484 MB | **89** | instant |
| MiniCPM5-1B-Base | Q4_K | 688 MB | **68** | instant |
| Qwen2.5-Coder-0.5B | F16 | 988 MB | 51 | instant |
| Qwen3-0.6B | F16 | 1.5 GB | 40 | instant |
| MiniCPM5-1B-Base | F16 | 2.16 GB | 30 | instant |

### Large models — the research frontier (27B)

| Model | Format | Size | tok/s | Notes |
|---|---|---|---|---|
| Qwen3.6-27B-mixed | Q2_K/Q4_K/Q6_K mixed | 11.98 GB | **5.73** (8-token short run) | hot cache; sustained ~1.4–2 tok/s long generation |

Large models are where the trade-off is visible: they run — where llama.cpp thrashes or OOMs — but decode is bandwidth-bound. **Closing the gap between small-model speed and large-model throughput, while keeping the resource footprint imperceptible, is the active research direction of this project.**

### Resource footprint — independent of model size

| Model | Params | Wired (anon) |
|---|---|---|
| Qwen2.5-Coder-0.5B | 0.5B | ~7 MB |
| MiniCPM5-1B-Base | 1B | ~7 MB |
| Qwen3.6-27B-mixed | 27B | **~77 MB** (incl. KV/SSM state) |

**Model size is theoretically unbounded.** Whatever the parameter count, wired memory stays at the ~7 MB level and total physical footprint stays proportional to the page cache the OS chooses to keep — the engine itself never holds the model in anonymous RAM. The only practical limits are disk space and patience per token.

### Memory — the core claim (TinyLlama 1.1B Q4_K_M, 64 tokens)

| | stratum | llama.cpp | ratio |
|---|---|---|---|
| anonymous (binding RAM) | **7.4 MB** | 634.9 MB | **85.7× lower** |
| total physical | 645.2 MB | 1300.5 MB | 2.0× lower |

### Scale — 27B on a 24 GB machine

stratum runs the 27B to completion in ~77 MB anonymous RAM. llama.cpp cannot run it usably here: `-ngl 0` thrashes, `-ngl 99` needs 16 GB in unified memory. Stratum is the only engine that produces tokens.

### How performance scales

Every token consumes the whole weight stream once, so per-token time is `W_bytes × (f_hot / BW_hot + f_cold / BW_cold)` — RAM decides how much stays hot in page cache (`f_hot`), memory bandwidth sets `BW_hot`, the SSD sets the cold-streaming rate. Estimated 27B behavior:

| Hardware | RAM | Expected (estimate) |
|---|---|---|
| M1 base, 8 GB | fully cold streaming | ~0.2–0.3 tok/s (SSD ~3 GB/s) |
| M4 Pro, 24 GB | partially hot | **5.73 tok/s hot short-run / ~1.4–2 sustained** (measured) |
| M4 Max, 48 GB+ | fully hot | ~8–10 tok/s (higher bandwidth) |
| ≥128 GB workstation | fully hot + nibble layout | 12+ tok/s (Q2_K 2.2× unpack) |

### Mechanism limits

- **Single-stream decode is bandwidth-bound**: every token reads the whole weight stream once (11.98 GB @ ~27 GB/s hot ≈ 0.44 s/token on this machine).
- **Q2_K unpack is compute-bound** (~7 GB/s at 14 cores); the nibble layout breaks this (2.2×) at the cost of +50% file size — needs ≥32 GB RAM to stay hot.
- **Long-generation tree efficiency is 2.46 tok/main** — draft quality, not tree parameters, is the wall.

---

## Part V — Reference

### Environment variables

The engine has 200+ env vars (mostly GPU kernel-variant toggles from experiments). The key ones:

| Variable | Purpose | Boundary |
|---|---|---|
| `STRATUM_NO_GPU=1` | Force CPU-only (default benchmark config) | ✅ |
| `STRATUM_GPU=1` / `STRATUM_GPU_NC=1` | GPU paths; NC = per-tensor zero-copy | ✅ (NC bit-exact) |
| `STRATUM_MULTISEQ=N` | N streams sharing one weight scan | ✅ (130 tok/s @ 64) |
| `STRATUM_MTP` / `STRATUM_TREE_*` | MTP tree / speculative decode | ✅ |
| `STRATUM_ASYNC_PREFETCH=1` | Background pread prefetch | ✅ |
| `STRATUM_HOT_FAST=1` | Hot-cache pure-compute mode (no page-cache lock) | ✅ |
| `STRATUM_STREAM_DET=1` | Deterministic streaming (skip mincore detection) | ✅ |
| `STRATUM_Q2K_NIB=<path>` | Q2K nibble-layout model (type 42) | ✅ (byte re-arrangement) |
| `STRATUM_TREE_EXTEND_K=N` | Tree chain depth (cap 12, default 4) | ✅ |
| `STRATUM_GPU2=1` | Cold-weight staging pipeline | ⚠️ slower than CPU for hot weights |
| `STRATUM_Q2K_SDOT=1` | int8 SDOT for Q2K | ⚠️ no gain at 14 cores |
| `STRATUM_PREDECODE=1` | Q4_K→F16 predecode to GPU | ❌ memory boundary |
| `STRATUM_Q4_0=1` | Q4_K→Q4_0 re-encode | ❌ quality boundary |
| `STRATUM_MLOCK_ALL` / `STRATUM_KEEP_RESIDENT` | Lock/pin weights | ❌ memory boundary |

### Repository layout

```
├── README.md          ← this file (English)
├── README_CN.md       ← 简体中文
├── AGENTS.md          ← development guide, boundaries, full env reference
└── stratum/
    ├── native/        ← the engine (C/Metal), Makefile, 19 gate scripts
    ├── docs/          ← measured evidence (benchmark records) + README
    ├── tools/         ← standalone GGUF utilities (quantize, inspect, decode)
    └── benchmarks/    ← benchmark scripts (headtohead, manifesto, GPU sweeps)
```

### References

Projects consulted and where their ideas surface in Stratum:

- **[AirLLM](https://github.com/lyogavin/airllm)** — working set should scale with layer size rather than model size; Stratum applies the same idea at tensor granularity.
- **[ds4 / DwarfStar](https://github.com/antirez/ds4)** — narrow specialist engines, decode-graph capture, expert LRU, exactness where it matters.
- **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** — 2.78T on 8 GB; O_DIRECT reads, direct consumption of packed data, a bit-level determinism contract; the nibble layout questions the weight format itself.
- **[flash-moe (Alexintosh)](https://github.com/Alexintosh/flash-moe)** — pure C/Metal MoE on Apple Silicon; SSD expert streaming, FMA-fused dequant kernels, trusting the OS page cache.
- **[tessera](https://github.com/geoph9/tessera)** — NoCopy GPU buffers with `MADV_DONTNEED` page eviction; informed the per-tensor NoCopy boundary. (Repository no longer publicly available.)
- **[HuggingFace transformers](https://github.com/huggingface/transformers)** — the reference framework used for output comparison.
- **[ggml / llama.cpp](https://github.com/ggerganov/llama.cpp)** — GGUF quantization formats and the model conversion toolchain.

### License

Apache-2.0.
