# Stratum

[English](README.md) | [简体中文](README_CN.md)

A pure-C transformer inference engine with the lowest wired-memory footprint of any engine: **weights are a stream, not a resident**. ~7 MB anonymous RAM runs a 27B dense model, because weights are read directly from the OS page cache via `mmap` — wired memory is decoupled from model size.

## Why it exists

Traditional engines assume weights must fit in RAM/VRAM. Stratum assumes the opposite: every token consumes the whole weight stream once, so working set stays constant regardless of model size.

- **Binding RAM**: ~7 MB anonymous (vs hundreds of MB for llama.cpp) — measured, see below
- **Models larger than RAM**: runs to completion where llama.cpp thrashes or OOMs — the uncontested niche
- **Bit-exact**: greedy decoding is token-for-token identical to a scalar reference
- **No quantization by the engine**: it consumes Q2_K–Q6_K / Q8_0 / F16 / F32 as-is, never degrades them

## Features

- One binary auto-detects architecture from GGUF metadata: **Llama-family** (Llama 1/2/3, TinyLlama, Mistral, Qwen2-dense) and the **Qwen3.5/3.6 hybrid** (Gated DeltaNet SSM + full attention)
- NEON-vectorized quantized kernels (Q2_K–Q6_K), int8 dotprod (SDOT) paths
- **MTP tree decode**: 8 tokens per forward at 100% acceptance
- **MULTISEQ**: N streams share one weight read — aggregate throughput scales ~linearly (130 tok/s @ 64 streams on 27B, measured)
- **Q2_K nibble layout** (type-42 GGUF): byte re-arrangement that halves unpack instructions (2.2×, values unchanged)
- Optional bounded-buffer Metal GPU compute, including per-tensor zero-copy direct-read (`STRATUM_GPU_NC`) and batched multi-matmul submission (17.3× vs per-matmul waits)

## Build

```sh
cd stratum/native
make stratum          # universal GGUF runtime
make tests            # quant-kernel cross-validation + sampler exactness
```

## Run

```sh
./stratum <model.gguf> [N_GENERATE] [PROMPT_TOKEN_ID...]
STRATUM_NO_GPU=1 /usr/bin/time -p ./stratum <model.gguf> 64 0 1   # CPU-only benchmark
```

See `AGENTS.md` for the environment-variable reference and development boundaries.

## Measured benchmarks

Apple M4 Pro (14 cores), 24 GB unified memory, macOS, greedy decoding, CPU path (`STRATUM_NO_GPU=1`). Method: `/usr/bin/time -p ./stratum <model.gguf> 64 0 1` (wall includes load + prefill + 64 generated tokens).

| Model | Architecture | Format | Size | Params | tok/s | Notes |
|---|---|---|---|---|---|---|
| Qwen3.6-27B-mixed | qwen35 (SSM+attn) | Q2_K/Q4_K/Q6_K mixed | 11.98 GB | 27B | **5.73** (8-token short run) | hot cache; sustained ~1.4–2 tok/s long generation |
| Qwen2.5-Coder-0.5B | llama | Q4_K | 398 MB | 0.5B | **125** | |
| Qwen3-0.6B | llama | Q4_K | 484 MB | 0.6B | **89** | has_qk_norm=yes |
| MiniCPM5-1B-Base | llama | Q4_K | 688 MB | 1B | **68** | |
| Qwen2.5-Coder-0.5B | llama | F16 | 988 MB | 0.5B | **51** | |
| Qwen3-0.6B | llama | F16 | 1.5 GB | 0.6B | **40** | |
| MiniCPM5-1B-Base | llama | F16 | 2.16 GB | 1B | **30** | |

### Memory — the core claim (TinyLlama 1.1B Q4_K_M, 64 tokens)

| | stratum | llama.cpp | ratio |
|---|---|---|---|
| anonymous (binding RAM) | **7.4 MB** | 634.9 MB | **85.7× lower** |
| total physical | 645.2 MB | 1300.5 MB | 2.0× lower |

### Scale — 27B on a 24 GB machine

stratum runs the 27B to completion in **~77 MB anonymous RAM** (weights stream from page cache via mmap). llama.cpp cannot run it usably here: `-ngl 0` thrashes, `-ngl 99` needs 16 GB in unified memory. Stratum is the only engine that produces tokens.

### Mechanism limits (honest)

- Single-stream decode is bandwidth-bound: every token reads the whole weight stream once (11.98 GB @ ~27 GB/s hot ≈ 0.44 s/token on this machine)
- Q2_K unpack is compute-bound (~7 GB/s at 14 cores); the nibble layout breaks this (2.2×) at the cost of +50% file size — needs ≥32 GB RAM to stay hot
- The long-generation tree efficiency is 2.46 tok/main — draft quality, not tree parameters, is the wall

## Acknowledgements

This project stands on the shoulders of:

- **[AirLLM](https://github.com/lyogavin/airllm)** — working set should scale with layer size, not model size. We pushed the same idea to tensor granularity.
- **[ds4 / DwarfStar](https://github.com/antirez/ds4)** — narrow specialist engines, decode-graph capture, expert LRU, "exactness where it matters".
- **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** — 2.78T on 8 GB; O_DIRECT, packed-data direct consumption, bit-level determinism contract. Our nibble layout questions the format itself, as k3 questions every byte.
- **[flash-moe (Alexintosh)](https://github.com/Alexintosh/flash-moe)** — pure C/Metal MoE on Apple Silicon; SSD expert streaming, FMA-fused dequant, "trust the OS page cache".
- **[HuggingFace transformers](https://github.com/huggingface/transformers)** — the reference framework.
- Also consulted: **[tessera](https://github.com/geoph9/tessera)** (NoCopy + `MADV_DONTNEED` — overturned our earlier "NoCopy wires memory" conclusion) and ggml/llama.cpp (quant formats + toolchain).

## Repository layout

```
├── README.md          ← this file (English)
├── README_CN.md       ← 中文版
├── AGENTS.md          ← development guide, boundaries, env-var reference
└── stratum/
    ├── native/        ← the engine (C/Metal, 25k lines), Makefile, gate scripts
    ├── docs/          ← experiment evidence archive (159 runs) + README
    ├── tools/         ← standalone GGUF utilities
    └── benchmarks/    ← shell/python benchmark scripts
```

## License

Apache-2.0 (placeholder — finalize before v0.1).
