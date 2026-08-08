# Stratum Model Benchmarks

> Benchmark date: 2026-08-08
> Hardware: Apple M4 Pro (14 cores), 24 GB unified memory, macOS
> Engine: Stratum (`native/stratum`), CPU path (`STRATUM_NO_GPU=1`), greedy decoding (temp=0)
> Method: `/usr/bin/time -p ./stratum <model.gguf> 64 0 1`, wall time includes model load + prefill + 64 generated tokens
> Conversion: small models converted from safetensors with llama.cpp `convert_hf_to_gguf.py` (F16, lossless) + `llama-quantize` (Q4_K). F16 = lossless conversion, Q4_K = quantized.

## Results

| Model | Architecture | Format | File Size | Params | Speed (tok/s) | Wall (incl. prefill) | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3.6-27B-mixed | qwen35 (SSM+attn) | Q2_K/Q4_K/Q6_K mixed | 11.98 GB | 27B | **5.73** (8-token short run) | 23.3s (32-token long run) | Hot cache; long-generation tree efficiency 2.46 tok/main, sustained ~1.4-2 tok/s |
| Qwen2.5-Coder-0.5B | llama (Qwen2 dense) | **Q4_K** | 398 MB | 0.5B | **125** | 0.51s | 24 layers, embeddings kept F16 |
| Qwen2.5-Coder-0.5B | llama | F16 (lossless) | 988 MB | 0.5B | **51** | 1.25s | |
| Qwen3-0.6B | llama (Qwen3 dense) | **Q4_K** | 484 MB | 0.6B | **89** | 0.72s | 28 layers, has_qk_norm=yes |
| Qwen3-0.6B | llama | F16 (lossless) | 1.5 GB | 0.6B | **40** | 1.61s | |
| MiniCPM5-1B-Base | llama (Llama) | **Q4_K** | 688 MB | 1B | **68** | 0.94s | 24 layers |
| MiniCPM5-1B-Base | llama | F16 (lossless) | 2.16 GB | 1B | **30** | 2.11s | |

## Model Paths

| Model | GGUF path |
|---|---|
| Qwen3.6-27B-mixed | `/Users/shiaho/Desktop/Qwen3.5-0.8B-hf/model-27b-gguf/qwen3.6-27b-mixed.gguf` |
| Qwen2.5-Coder-0.5B (Q4_K) | `/Users/shiaho/Desktop/0-/Qwen2.5-Coder-0.5B/Qwen2.5-Coder-0.5B.q4k.gguf` |
| Qwen2.5-Coder-0.5B (F16) | `/Users/shiaho/Desktop/0-/Qwen2.5-Coder-0.5B/Qwen2.5-Coder-0.5B.llama.f16.gguf` |
| Qwen3-0.6B (Q4_K) | `/Users/shiaho/Desktop/0-/Qwen3-0.6B/Qwen3-0.6B.q4k.gguf` |
| Qwen3-0.6B (F16) | `/Users/shiaho/Desktop/0-/Qwen3-0.6B/Qwen3-0.6B.llama.f16.gguf` |
| MiniCPM5-1B-Base (Q4_K) | `/Users/shiaho/Desktop/MiniCPM5-1B-Base/MiniCPM5-1B-Base.q4k.gguf` |
| MiniCPM5-1B-Base (F16) | `/Users/shiaho/Desktop/MiniCPM5-1B-Base/MiniCPM5-1B-Base.f16.gguf` |

## Detailed Notes

### Qwen3.6-27B (mixed quantization, streaming inference)

- **Short run (8 tokens)**: 1 forward produces the full 8 tokens (MTP tree aema=8.00), 5.73 tok/s with hot cache.
- **Real long generation (32+ tokens)**: tree draft quality degrades after 8 tokens; measured 2.46 tok/main, 23.3s/32 tokens ≈ 1.37 tok/s (half-hot environment); clean-state estimate ~2 tok/s.
- **Memory characteristics**: mmap streaming, wired memory ~7 MB; the 11 GB of weights live in page cache (reclaimable).
- **Boundaries**: Q2_K makes up 48% of weights (that is the model file's own precision — the engine performs no precision reduction); the nibble-layout variant (16.3 GB) cannot stay fully hot in page cache on a 24 GB machine — needs ≥32 GB RAM.

### Small models (llama architecture)

- All run through the `llama` architecture registry (Stratum natively supports Qwen2-dense / Qwen3 dense / Llama family).
- Qwen3-0.6B's qk-norm is correctly detected by the engine (has_qk_norm=yes); the output path is complete.
- Purely autoregressive (no MTP tree for llama architecture), so speed = one forward per token.
- Q4_K variants are significantly faster than F16 (more weights per byte, better bandwidth utilization).

### Not tested / unsupported

| Model | Reason |
|---|---|
| Qwen3.5-0.8B (`model/` dir) | Directory contains no weight files (config only) |
| model-int4 / int4-awq (Qwen3.5 0.8B GPTQ/AWQ) | llama.cpp converter does not support GPTQ/AWQ weight mapping |
| model-mlx-4bit / 8bit / g32 | MLX quantization format (scale tensors), not supported by the converter |
| 27B nibble variant (16.3 GB) | Insufficient RAM on the 24 GB machine (page cache cannot hold it) |

## Reproduction

```bash
cd /Users/shiaho/Desktop/Qwen3.5-0.8B-hf/stratum/native
STRATUM_NO_GPU=1 /usr/bin/time -p ./stratum <model.gguf> 64 0 1
```
