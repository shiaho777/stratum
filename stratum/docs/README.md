# Experiment Data (docs/)

This directory is the **evidence archive** of the engine's development: every JSON file is the recorded output of a real experiment that decided a kernel or scheduling choice. The guiding principle of this project is *decisions backed by data* — these files are that data.

## File naming convention

Most names follow the pattern:

```
<feature>_<variant>_<batch-sizes>_<context-length>_r<repeats>.json
```

- `feature` — what was being tested (e.g. `q4dual_swiglu_fused`, `q6_down_v5_g2`, `rmsnorm_tg128`)
- `variant` — the specific configuration (`default`, `optin`, `vs_disable`, `b16_b32`, `tg32_tg128`…)
- `b16_b32` — batch sizes compared (MTP-tree batch, GPU batch)
- `96` / `64` / `128` — generated-token count of the run
- `r5` / `r10` — repeats (variance check)

## What each family recorded

| Prefix | What was being decided |
|---|---|
| `q4dual_swiglu_fused_*` | Whether fusing the Q4_K dual-path SwiGLU (gate+up) kernel wins vs unfused; threadgroup size and batch width sweeps |
| `q6_down_*`, `q6v*` | Q6_K down-projection kernel evolution (v4→v6), row-tiling, group-2/4, threadgroup sizes |
| `batch_gpu_sweep_*` | GPU batched-matmul parameter sweeps: aggregate tok/s, wall time, and golden token sequences (correctness guard) |
| `rmsnorm_*` | RMSNorm threadgroup-size choice |
| `q4_kv_fused_*`, `kv_rope_*` | KV-cache quantization and fused KV+RoPE layouts |
| `lmhead_top1_*`, `q4_o_fused_*` | LM-head top-1 path and Q4 output-projection fusion |
| `headtohead_*` | Stratum vs llama.cpp: anonymous-memory and speed comparison (the "7MB vs 635MB" evidence) |
| `manifesto_*`, `memory_floor_*`, `pressure_test_*` | Whole-engine memory-floor and pressure validation |
| `exp*` (exp1…exp19) | Early research-stage experiments: sparsity, gate skipping, token stability, predictors, long-context floors |

## Re-running

The JSON files are outputs; the scripts that produced them lived alongside earlier engine revisions and are not all preserved. Where a script still exists (e.g. `native/bench_*`, `stratum/tools/*`), it can be rebuilt with the included `Makefile`. For historical `exp*` files the numbers are recorded as-is — treat them as historical evidence, not necessarily reproducible against today's code.

## Honest note

File names use internal shorthand (e.g. `tg32`, `b24`, `interleaved_96`) from the development log. They were written for the developer's own bookkeeping; we keep them verbatim rather than rename, to preserve traceability to the experiments they document.
