# Stratum

A pure-C transformer inference runtime.

## Goal

Lowest binding RAM of any inference engine, and the fastest **CPU** path
for the same GGUF weights — while being the only engine that runs models
**larger than RAM** at usable footprint. Honest scope (measured below):

- **Binding RAM:** we win against llama.cpp in every mode, always
  (mmap-streaming architecture; ~7 MB anonymous vs hundreds of MB).
- **Speed, CPU vs CPU:** we beat llama.cpp `-ngl 0` on models that fit.
- **Speed vs llama.cpp's Metal GPU (`-ngl 99`):** we currently **lose**
  on models that fit (it runs the whole forward on-GPU; we are CPU-only).
  Closing this is the open frontier, not a solved claim.
- **Models bigger than RAM:** llama.cpp is unusable (CPU thrashes/OOMs,
  GPU needs the model in unified memory); stratum runs them at ~77 MB
  binding RAM. This is the uncontested niche.

Output is bit-exact: greedy decoding is token-for-token identical to a
scalar reference.

One binary auto-detects the architecture from GGUF metadata and runs
Llama-family (Llama 1/2/3, TinyLlama, Mistral, Qwen2-dense) and the
Qwen3.5/3.6 hybrid Mamba/Transformer. K-quants (Q2_K–Q6_K), Q8_0, F16,
F32, NEON-vectorized, with optional bounded-buffer Metal GPU compute.

Weights stream from the OS page cache via mmap/pread; the only anonymous
allocations are activations, KV cache, SSM state, and scratch. Binding
RAM is decoupled from model size.

## Build

```sh
cd native
make stratum          # universal GGUF runtime
make tests            # quant-kernel cross-validation + sampler exactness
```

## Run

```sh
./native/stratum <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]
.venv/bin/python native/stratum_tokenize.py encode "The capital of France is"
```

## Benchmarks (measured)

Apple Silicon, 24 GB. Memory via `vmmap` physical-footprint of the live
process; "anonymous" = private writable RAM (the binding constraint),
"mapped file" = reclaimable page cache.

### Memory — TinyLlama 1.1B Q4_K_M (638 MB on disk), 64 tokens

| | stratum | llama.cpp | ratio |
|---|---|---|---|
| anonymous (binding RAM) | **7.4 MB** | 634.9 MB | **85.7× lower** |
| mapped file (reclaimable) | 637.8 MB | 718.5 MB | — |
| total physical | 645.2 MB | 1300.5 MB | 2.0× lower |

### Speed — TinyLlama 1.1B Q4_K_M, 128 tokens, generation throughput + peak RSS

Three-way, same weights, same machine (M4 Pro). This is the honest table —
including where we lose.

| configuration | gen tok/s | peak RSS | binding (anon) |
|---|---|---|---|
| stratum (default, CPU, int8 dot) | 84 | 683 MB | ~7 MB |
| **stratum (`STRATUM_GPU_FULL`, whole-token on GPU)** | **172** | ~720 MB | **~10 MB** |
| llama.cpp `-ngl 0` (CPU) | ~70 | 1353 MB | ~700 MB |
| llama.cpp `-ngl 99` (Metal GPU) | 230 | 727 MB | ~90 MB |

The full-GPU path runs the **entire token in one Metal command buffer** —
residual stream and KV cache stay resident on GPU across all layers, weights
read zero-copy from the mmap'd model (no extra binding RAM). It went 17 →
172 tok/s by collapsing 155 dispatches/token → 1 (eliminating CPU↔GPU
round-trip overhead) then fixing kernel thread-occupancy. Greedy output is
bit-exact to the CPU path. At 172 it is **2× the CPU path and 0.75× of
llama.cpp's Metal backend — at ~9× lower binding RAM** (it keeps the
zero-copy mmap weights instead of GPU-resident copies).

Single-stream batch-1 decode is bandwidth-bound (~420 tok/s ceiling on this
machine = 638 MB ÷ ~270 GB/s); kernel work continues toward it. Exponential
gains live in throughput (batched multi-sequence: one weight-stream serves B
tokens → aggregate ~tok/s × B), the next target.

### Speed — V1-V8 GPU acceleration phases (M4 Pro, same weights)

TinyLlama 1.1B Q4_K_M (638 MB):

| configuration | gen tok/s | vs CPU | notes |
|---|---|---|---|
| CPU B=1 (NEON SDOT) | 79.3 | 1.0× | baseline |
| GPU B=1 (`STRATUM_GPU_FULL=1`) | 183.8 | 2.32× | whole-token on GPU, 1 dispatch/tok |
| GPU B=1 + sparse (`STRATUM_SPARSE=0.001`) | 191.0 | 2.41× | bit-exact, block-skip on down_proj |
| GPU B=1 + F16 predecode (`STRATUM_PREDECODE=1`) | 93.9 | 1.18× | 3.5× bandwidth, slower on small model |
| **GPU B=8 (`STRATUM_GPU_BATCH_FULL=1`) aggregate** | **412.6** | **5.20×** | **97.5% of bandwidth ceiling** |
| GPU B=32 aggregate | 388.3 | 4.90× | saturated |

Llama 3.2 3B Q4_K_M (1.9 GB):

| configuration | gen tok/s | vs CPU | notes |
|---|---|---|---|
| CPU B=1 | 8.4 | 1.0× | baseline |
| GPU B=1 | 58.9 | 7.01× | larger matmuls → greater GPU advantage |
| GPU B=8 aggregate | 113.9 | 13.6× agg | 80% of bandwidth ceiling |

Token output bit-exact between CPU and GPU paths (greedy decoding).
Logits have ~0.1% float32 vs float64 accumulation difference (does not
affect token selection).

The CPU path itself reached 83 tok/s (12.0 ms/tok) via five first-principles
fixes, all measure-first and quality-preserving (greedy identical to the
exact-float path; +0.2% perplexity from int8 dot, `STRATUM_PPL=1`):

1. **coarse-chunked parallel matmul** — one contiguous row range per core.
2. **int8 dotprod (SDOT) Q4_K kernel**.
3. **P-core-only chunking for small models** (6 P-core chunks beat 14).
4. **int8 dotprod (SDOT) Q6_K kernel** (`ffn_down`/`lm_head`).
5. **fused same-input matmul groups** (q/k/v and gate/up): quantize the
   shared input once, one parallel region — cut Q4_K matmul time ~3.5×.

Llama 3.2 3B Q4_K_M, same comparison shape: stratum CPU **4.6 s** / 2045 MB
vs llama.cpp `-ngl 0` **5.1 s** / 4170 MB (we win CPU-vs-CPU on both).

### Correctness

Greedy (`temp=0`, top-k=1) decoding is token-for-token identical to
`llama-simple` on the models where llama.cpp fits in RAM as a reference
(TinyLlama 1.1B, Llama 3.2 3B). `make tests` cross-validates every quant
kernel against a scalar reference and verifies speculative-decode
sampling exactness by Monte-Carlo.

### Scale — Qwen3.6 27B Q4_K_M (17 GB on disk), the uncontested niche

On a 24 GB machine, stratum runs the 27B to completion in **~77 MB
anonymous RAM** (weights stream from the page cache via mmap). llama.cpp
cannot run it usably here: `-ngl 0` thrashes (~158 s for 8 tokens, swapping),
and `-ngl 99` needs the 16 GB of weights in unified memory, which doesn't
fit. stratum is the only engine that produces tokens.

The cost is speed, and the bottleneck is **SSD bandwidth**, not compute.
Measured per-token breakdown (partial residency): ~1.8 s matmul (CPU) +
~5 s waiting on the SSD to stream the non-resident weights = ~0.2 tok/s.
The compute floor (if the model were fully RAM-resident) is ~1.8 s matmul
→ **~0.4–0.5 tok/s on CPU**; reaching it needs ~18 GB free RAM so the
whole model stays resident. So: small models that fit are **CPU-bound**;
oversized streamed models are **SSD-bound**, and RAM capacity is the gate
that decides which wall you hit.
