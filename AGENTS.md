# AGENTS.md — Stratum Inference Engine Development Guide

## Project Identity

Stratum is an inference engine for Transformer models on Apple Silicon that runs with an extremely low wired-memory footprint. Its core advantage is the **mmap streaming architecture** — wired memory is decoupled from model size: ~7 MB anonymous memory runs a 27B dense model. Speed is a secondary goal; the memory advantage is the lifeblood.

The engine is a **native C/Metal codebase** (`stratum/native/`). An early Python reference implementation existed but was removed (2026-08) — the native engine is the only maintained line. Python scripts under `stratum/tools/` are standalone GGUF utilities, not part of the engine.

## Three Hard Boundaries

The following boundaries must not be violated, must not be traded away for speed, and must not be bypassed "for testing" or "for an experiment":

### 1. Never harm model quality

- Never requantize existing weights (e.g. re-encoding Q4_K → Q4_0) — that introduces precision loss
- Never skip layer computation to save I/O — that changes model output
- Never approximate compute (e.g. lowering softmax precision, truncating attention)
- Quantization formats (Q2_K/Q4_K/Q6_K) in a model file are the model's own choice; the engine consumes them as-is and never degrades them
- The only allowed "approximation" is int8 dotprod (Q4_K SDOT), verified greedy bit-exact
- **The type-42 Q2_K nibble layout is a byte re-arrangement, not requantization**: the 0–3 weight values are unchanged; it only changes how bytes are unpacked. It is allowed.

### 2. Never increase wired memory

- Wired (anonymous, non-reclaimable) memory must stay at the ~7 MB level
- Never mlock the whole model or large weight blocks into physical RAM
- Never pre-allocate large buffers to cache weights (GPU staging buffers excepted, must be bounded)
- Weights must be streamed via mmap from the OS page cache; page cache is reclaimable
- Anonymous allocation for activations, KV cache, and SSM state is allowed (the model needs it)

### 2a. Metal GPU access — measured, not assumed

**History**: V54 observed wired-memory blowup with `newBufferWithBytesNoCopy` over a whole-model mmap and this boundary was written as "NoCopy is forbidden". That conclusion was **overturned by controlled measurement** (2026-08): sequentially GPU-reading the whole 11.98 GB model through per-tensor NoCopy windows added only +0.04 GB wired (hold=0) / +0.2 GB (hold=1). The V54 blowup was a compound of *whole-model single-chunk registration* + a system already near OOM — not an inherent property of NoCopy.

Current rules:

- **Allowed**: per-tensor NoCopy direct-read (`STRATUM_GPU_NC=1`) — each matmul wraps only its own tensor (<100 MB) in a NoCopy buffer, dispatches, and releases. Measured safe and bit-exact on the 27B model.
- **Allowed**: batched NC submission (`stratum_metal_nc_batch_*`) — multiple independent matmuls share one command buffer (17.3× vs per-matmul waits).
- **Forbidden**: registering the whole model (or chunks > ~1 GB) as one NoCopy buffer — this is the V54 failure mode.
- **Forbidden**: `keep_resident` auto mode (locks page cache = implicit mlock). `STRATUM_HOT_FAST` is the sanctioned replacement: it reuses the resident fast path for scheduling but does **not** lock page cache.
- GPU2 (staging) remains the cold-weight path; measured slower than CPU for hot weights — do not route hot weights to staging.

### 3. Testing must not exhaust the machine

- Before any big-model test, check memory state (`vm_stat` free pages, `sysctl vm.swapusage`); do not start if the system is under pressure
- 27B tests: prefer 1–2 tokens for correctness checks; a full 8-token speed run is acceptable **only** when the model is already hot in page cache and free memory allows
- Kernel-level experiments must use the lightweight micro-benchmarks (`bench_*.c`, tens of MB) — not the 27B end-to-end
- Never run the model plus heavy background processes simultaneously on the 24 GB machine
- Small models (0.5B–1B GGUF) are the primary test objects

## Development Priorities

1. **Correctness** — greedy decoding must stay bit-exact (token sequence identical) with the reference implementation
2. **Memory** — wired memory must not grow with model size
3. **Speed** — pursue performance only within boundaries 1 and 2

## Speed Optimization Directions (within boundaries)

### Allowed

- GPU kernel efficiency (Q4_K unpack, SIMD, fused ops)
- Multi-stream batching (batched decode, weight reuse) — `STRATUM_MULTISEQ=N` measured: aggregate throughput scales ~linearly with N (130 tok/s @ 64 streams on 27B)
- Speculative decoding (n-gram spec, MTP tree) — changes nothing about output, only speed. MTP tree: 8 tokens/forward at full acceptance
- Background pread prefetch (zero extra wired memory, page cache is reclaimable)
- SSM group dispatch (merge multiple matmuls into one command buffer)
- Metal per-tensor NoCopy direct-read + batched NC submission (see 2a)
- Q2K nibble layout (type-42): byte re-arrangement that halves unpack instructions (measured 2.2×) — allowed because values are unchanged

### Forbidden

- ❌ Requantizing to lower precision (Q4_K → Q4_0 re-encoding)
- ❌ mlock of the whole model or large weight blocks
- ❌ Pre-allocating GPU buffers to cache all weights
- ❌ Layer/block skipping (changes output)
- ❌ Lowering attention precision, truncating KV cache
- ❌ `newBufferWithBytesNoCopy` over whole-model/chunk >1 GB mmap (V54 failure mode)
- ❌ `keep_resident` auto mode (page-cache lock = implicit mlock)
- ❌ Any scheme keeping >200 MB of weight buffers on the GPU

## Architecture Overview

### Fourth hard boundary: generality

Stratum targets **any model**, not a specific one. These rules must not be violated:

- **Never hardcode a model or architecture name in `stratum.c`** — dispatch must go through the registry (`stratum_arch.h`)
- **Never assume specific-model properties in config loading** (e.g. probing tensor existence via `blk.3` instead of `blk.0`; SSM defaults must not assume one model's parameters)
- **Adding an architecture = create `stratum_arch_<name>.inc.c`, implement the `StratumArch` interface, call `STRATUM_REGISTER_ARCH()`** — the Makefile auto-collects `stratum_arch_*.inc.c` into `stratum_archs.gen.h` (included by `stratum.c`), so no existing file is modified
- **Shared infrastructure** (quantized linear layers, CPU/GPU init, madvise, KV cache, spec decode) lives in `stratum_linear.h` / `stratum_engine.h` — never copy-pasted per architecture
- **Env vars must be generic** — no model-specific `STRATUM_QWEN35_*` variables

### Native C engine (`native/`)

- `stratum.c` — entry point, dispatch through registry (no hardcoded architecture names)
- `stratum_arch.h` — generic registry interface + generic config loader
- `stratum_linear.h` — generic quantized linear layers (Q4_K/Q5_K/Q6_K/Q3_K/Q2_K/Q8_0/F16/F32 dispatch)
- `stratum_engine.h` — generic engine infrastructure (CPU detect/GPU init/madvise/spec decode/memory reporting)
- `stratum_arch_qwen35.inc.c` — Qwen3.5 hybrid architecture (Gated DeltaNet + full attention), self-registers (~24k lines, the main engine)
- `stratum_arch_llama.inc.c` — Llama/Qwen2/Qwen3 dense architecture, self-registers
- `stratum_metal.m/.h` — Metal GPU acceleration layer (GEMV kernels, batched-B, group dispatch, NC zero-copy)
- `stratum_q2k/q3k/q4k/q5k/q6k.{h,neon.h,metal}` — quantized kernels (NEON + scalar + Metal shaders)
- `v199~v217_gate.sh` — bit-exact regression gates (see Testing)
- `Makefile` — builds `stratum`; also `tools_gguf_nib_convert.c` (Q2K nibble converter)

### Optional runtime dependency (MemX)

The engine's staging buffers and KV/SSM state can live in the MemX
compressed-memory plane (`github.com/shiaho777/memx`, MIT). It is an
**external downloaded dependency, never vendored**: `fetch_memx.sh` clones
or pulls the upstream repo into `stratum/native/memx/` (gitignored), and
upstream updates flow in via `make deps` — do not copy its code into this
repository. The engine builds and runs without it (`make USE_MEMX=0`); the
README's ~77 MB 27B wired-memory figure is measured with MemX enabled.

### Standalone tools (`stratum/tools/`)

Python GGUF utilities independent of the engine: `decode_gguf_ids.py`, `encode_prompt.py`, `gguf_inspect.py`, `quantize_qwen3_5.py`, `lowrank_decompose.py`, etc.

### Experiment evidence (`stratum/docs/`)

A small set of measured evidence files backing the README's claims (see `docs/README.md`): head-to-head memory comparisons, memory floor, pressure test. The ~150 per-kernel tuning JSONs from the development log were pruned in the 2026-08 cleanup and survive only in git history. These are historical evidence — treat as records, not necessarily reproducible against today's code.

### Runtime model & limitations

The engine is **single-model, single-process, non-reentrant by design**:

- Every architecture keeps its state in file-scope `static` globals (`q35_g_*`,
  `la_g_*`, the shared `g_st` linear state). One process = one model; there is
  no isolation between concurrent invocations in the same address space.
- `STRATUM_SERVER` mode (if used) is a serial stdin loop, not a concurrent
  server — do not extend it to concurrent requests without first isolating
  this state.
- Testing consequence: one engine instance per test process; do not initialize
  two models in one process and expect clean teardown.

**Recording experiment configuration**: results are only comparable when the
configuration is recorded. For any perf/bit-exact claim, log at least:
architecture (from `general.architecture`), all relevant `STRATUM_*` vars
(see `docs/ENVVARS.md`), CPU/Metal/SDOT state, hot vs cold page-cache state,
and the command line. `run_all_gates.sh` and `verify_backends.sh` pin most of
this for their own runs.

## Environment Variables

The full map of every `STRATUM_*` switch, its call sites, and its status is
`docs/ENVVARS.md`, regenerated by `stratum/tools/env_census.py` — keep it in
sync when a switch is added/removed. The forbidden switches in the table are
**enforced at startup**: setting a non-zero value makes the engine refuse to
run (`stratum_enforce_boundaries()` in `stratum_engine.h`). Key ones:

| Variable | Purpose | Boundary status |
|---|---|---|
| `STRATUM_NO_GPU=1` | Force CPU-only (the default benchmark config) | ✅ allowed |
| `STRATUM_GPU_NC=1` | Per-tensor NoCopy direct-read (zero-copy) | ✅ allowed, bit-exact (2a) |
| `STRATUM_GPU2=1` | Cold-weight staging pipeline | ✅ allowed; measured slower than CPU for hot weights |
| `STRATUM_GPU=1` / `STRATUM_GPU_FULL` | Legacy GPU paths | ⚠️ small models only |
| `STRATUM_MULTISEQ=N` | N parallel sequences sharing one weight stream | ✅ allowed (130 tok/s @ 64) |
| `STRATUM_NGRAM_SPEC` / `STRATUM_TREE_*` / `STRATUM_MTP` | Speculative decoding / MTP tree | ✅ allowed |
| `STRATUM_ASYNC_PREFETCH=1` | Background pread prefetch | ✅ allowed |
| `STRATUM_HOT_FAST=1` | Hot-cache pure-compute mode (no page-cache lock) | ✅ allowed (replaces keep_resident) |
| `STRATUM_STREAM_DET=1` | Deterministic streaming (skip mincore detection) | ✅ allowed (measured: no gain, kept as option) |
| `STRATUM_Q2K_NIB=<path>` | Q2K nibble-layout model | ✅ allowed (byte re-arrangement) |
| `STRATUM_Q2K_NIB_OFF=1` | Disable nibble path | ✅ allowed |
| `STRATUM_TREE_EXTEND_K=N` | Tree chain depth (cap 12, default 4) | ✅ allowed |
| `STRATUM_Q2K_SDOT=1` | int8 SDOT for Q2K (opt-in) | ⚠️ measured no gain at 14 cores |
| `STRATUM_PREDECODE=1` | Q4_K→F16 predecode to GPU | ❌ violates memory boundary |
| `STRATUM_Q4_0=1` | Q4_K→Q4_0 re-encode | ❌ violates quality boundary |
| `STRATUM_MLOCK_ALL=1` / `STRATUM_HOT_GB=N` / `STRATUM_KEEP_RESIDENT=1` | Lock/pin weights | ❌ violates memory boundary |

## Testing

1. **Default test objects**: small models (0.5B–1B Q4_K GGUF, e.g. Qwen2.5-Coder-0.5B, Qwen3-0.6B); the 27B model only for specific streaming/wired-memory validation
2. **Kernel experiments**: use `bench_*.c` micro-benchmarks (tens of MB), never the 27B end-to-end
3. **Check memory before big tests**: `vm_stat` free pages and `sysctl vm.swapusage`; abort if the system is under pressure
4. **bit-exact validation**: run `v217_gate.sh` (or the relevant `v*_gate.sh`) — asserts greedy argmax sequence `[2, 220, 16, 13]` and `tok/main >= 8.0`. Any engine change must keep these gates passing. `run_all_gates.sh <model.gguf>` runs every gate in sequence.
5. **Backend consistency**: `verify_backends.sh <small.gguf>` asserts CPU / GPU-NC / GPU2 greedy sequences are identical (needs metallib). See `docs/VALIDATION.md` for the full matrix — what the gates cover, what they don't, and the per-kernel-change checklist.
6. **Performance reporting**: always report wired (anon) memory alongside wall time and tok/s

## Git

- The repository was rebuilt (2026-08) as a clean two-commit history: initial engine + path cleanup. Keep it clean.
- Commit in logical units with a message covering: what, why, bit-exact gate result, perf data when relevant
- Never commit: model weights (GGUF/safetensors), binaries, `*.metallib`, `.venv`, `__pycache__`, the fetched MemX dependency dir (`stratum/native/memx/`) — all covered by `.gitignore`
- Never commit machine-specific absolute paths (`/Users/...`); use relative paths or placeholders
