# How Stratum works

[Back to the README](../README.md) · [中文](HOW_IT_WORKS_CN.md)

One token reads the weight file once. Time per token is

```
t = W × (f_hot / BW_hot + f_cold / BW_cold)
```

`W` is the file. `f_hot` is the fraction the page cache still holds. RAM moves `f_hot`. It does not change the arithmetic.

## The stream

The GGUF is mapped read-only. Anonymous memory is the activation scratch, the KV cache, and SSM state: about 7 MB on a 0.5B–1B model, about 77 MB on the 27B. Weight pages are ordinary file cache. Under pressure the OS drops them and the next token reads them again.

The scheduler asks `mincore` per layer. A hot layer is compute. A cold layer gets `madvise` / `F_RDADVISE` on a 50–100 MB window while the current layer runs. `STRATUM_HOT_FAST=1` keeps that hot path and does not lock the cache.

`STRATUM_MULTISEQ=N` runs N sequences through one scan of the file. On the 27B, 64 streams measured 130 tok/s aggregate.

<p align="center"><img src="assets/memory-bars.svg" alt="Anonymous RAM: stratum against llama.cpp" width="720"></p>

## Layouts that do not change values

**Q2_K nibble (type 42).** The 2-bit weights stay 0–3. The bytes are repacked so the NEON unpack does half the work (measured 2.2×). The file grows about 50%. Enable with `STRATUM_Q2K_NIB=<path>` or an embedded type-42 GGUF. The engine checks the values before it trusts the layout.

**Q4K_W16 (type 43)** is the same idea for the AMX multi-sequence path: Q4_K values, retiled. `STRATUM_AMX_W16_OFF=1` forces the portable fallback.

## MoE

Registered for `llama-moe`, `llama_moe`, `moe`, `olmoe`. Attention matches the llama path. The FFN is a router plus top-k over stacked `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`. Ties take the lower expert id.

Each token touches k of E slices. Three schedulers sit on `madvise`. None of them picks a different expert, and none of them locks a page.

| Switch | Default | What it does |
|---|---|---|
| `STRATUM_MOE_PACK=<path>` | off | Sidecar from `stratum/tools/moe_expert_pack.py`. Gate, up, and down of one expert are one byte range, copied from the GGUF. One `WILLNEED` covers the expert. A mismatch at load refuses the file. |
| `STRATUM_MOE_PACK_CHECK=1` | sample | Compare every expert, not a sample. |
| `STRATUM_MOE_AHEAD=1` | on | Before attention, a hint router on the incoming residual faults likely pages. The post-attention router still selects. |
| `STRATUM_MOE_ADAPT=1` | on | Decayed hit counts. `WILLNEED` the hottest experts that fit in the budget. |
| `STRATUM_MOE_ADAPT_MB` | 256 | Budget in MiB. `0` turns the hot set off. |
| `STRATUM_MOE_ADAPT_DROP=1` | off | `DONTNEED` experts that fell out of the budget. |
| `STRATUM_PREFILL_B=N` | batch cap | One chunk reads each selected expert once. `1` is token-at-a-time. |

```sh
python3 stratum/tools/moe_expert_pack.py model.gguf model.moep
STRATUM_NO_GPU=1 STRATUM_MOE_PACK=model.moep STRATUM_MOE_PACK_CHECK=1 \
  stratum/native/stratum model.gguf 8 1 2 3
```

Check it with `tiny_moe_oracle.py`. The procedure is in [VALIDATION.md](../stratum/docs/VALIDATION.md). F16 tiny models match the oracle. Q4_K tiny models can differ from the oracle on a late near-tie; the engine modes still have to match each other. There is no CI gate and no full-size MoE run yet.

## MTP and n-gram drafts

The draft proposes. The main model verifies. A miss rolls state back, so the tokens that leave are the greedy tokens.

MoE n-gram (`STRATUM_NGRAM_SPEC`) extends only when every match of the suffix continues with the same token (`STRATUM_SPEC_CONF=1`, the default). An ambiguous suffix tries the shorter n-gram. On the Qwen3.5 chain, `STRATUM_SPEC_MIN_P=<p>` stops another extension when the leaf top-1 probability is below `p`. Leave it unset and the chain is unchanged. The draft is never longer than the number of tokens you asked for.

On the 27B the measured tree efficiency is 2.46 tok/main. Depth past 4 (`STRATUM_TREE_EXTEND_K`) got worse. The limit is the draft, not the tree.

<p align="center"><img src="assets/mtp-tree.svg" alt="MTP draft tree: one verify, exact rollback" width="720"></p>

## Metal and MemX

Metal is optional. Each matmul may wrap its own tensor, under about 100 MB, in a NoCopy view, then release it. A sequential read of the 11.98 GB model this way added about 0.04 GB wired. Registering the whole file, or a chunk past about 1 GB, is the path that wires the model. Batched submission of many matmuls on one command buffer measured 17.3× against waiting for each one (480 matmuls, 3.71 s to 0.21 s).

Fused full-GPU decode (`STRATUM_GPU=1 STRATUM_GPU_FULL=1` plus `STRATUM_MAX_KV=8192`) on Qwen3-0.6B Q4_K_M measured 247–252 tok/s against llama.cpp Metal at 224–231, stopped at the end of the turn. The eager GPU KV slab is about 1.84 GB. Weights stay file-backed.

[MemX](https://github.com/shiaho777/memx) can hold staging buffers and KV/SSM state. The ~77 MB 27B figure is with MemX on. `make USE_MEMX=0` builds without it. With MemX, logits can move by mean KL about 1e-4 between two identical runs; the token sequence stays put. `USE_MEMX=0` is the byte-reproducible build.

## What runs

| GGUF `general.architecture` | Code |
|---|---|
| `llama`, `qwen3` | `stratum_arch_llama.inc.c` |
| `qwen35`, `qwen35moe`, `qwen38` | `stratum_arch_qwen35.inc.c` — Gated DeltaNet plus full attention, and the MTP head |
| `llama-moe`, `llama_moe`, `moe`, `olmoe` | `stratum_arch_moe.inc.c` |
| mini-DiT, H3 video | `dit_probe`, `h3_*.c` — separate binaries, not `./stratum` |

Add an architecture by dropping `stratum_arch_<name>.inc.c` next to the others. The Makefile picks it up. `stratum.c` does not grow a name check.

## Measurements

M4 Pro, 14 cores, 24 GB, macOS, greedy, `STRATUM_NO_GPU=1`, `/usr/bin/time -p ./stratum <model.gguf> 64 0 1`, unless a row says otherwise.

| Model | Format | File | tok/s | Anonymous |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | Q4_K | 398 MB | 125 | ~7 MB |
| Qwen3-0.6B | Q4_K | 484 MB | 89 | ~7 MB |
| MiniCPM5-1B | Q4_K | 688 MB | 68 | ~7 MB |
| Qwen2.5-Coder-0.5B | F16 | 988 MB | 51 | ~7 MB |
| Qwen3-0.6B | F16 | 1.5 GB | 40 | ~7 MB |
| MiniCPM5-1B | F16 | 2.16 GB | 30 | ~7 MB |
| Qwen3.8-27B mixed | Q2_K/Q4_K/Q6_K | 11.98 GB | 5.73 hot short run; ~1.4–2 sustained | ~77 MB |

TinyLlama 1.1B Q4_K_M, 64 tokens: anonymous 7.4 MB against llama.cpp 634.9 MB. Physical 645 MB against 1300 MB.

| Machine | 27B, this file |
|---|---|
| M1 base, 8 GB, cold SSD | ~0.2–0.3 tok/s (estimate from ~3 GB/s) |
| M4 Pro, 24 GB | **5.73 tok/s** hot short run, ~1.4–2 sustained (measured) |
| M4 Max, 48 GB+, hot | ~8–10 tok/s (estimate) |
| ≥128 GB, hot, nibble layout | 12+ tok/s (estimate; Q2_K unpack 2.2×) |

Head-to-head clips and the JSON behind them live in [media/h2h-video/](../media/h2h-video/).

| Clip | stratum | llama.cpp |
|---|---|---|
| Qwen3-0.6B Q4_K_M, CPU | 254.8 tok/s, footprint 46 MB | 184.0 tok/s, footprint 4915 MB |
| same, fused GPU, `MAX_KV=8192` | 251.7 tok/s, footprint 1843 MB | 223.9 tok/s, footprint 4608 MB |
| MiniCPM5-2B Q4_K_M | 89–92 tok/s, footprint 29 MB | does not load this GGUF |
| MiniCPM5-2B F16 | 19.0–19.9 tok/s, footprint 32 MB | does not load this GGUF |

The 1843 MB GPU footprint is the eager KV slab. Weights stay in the file.

## Limits

- One process, one model. `STRATUM_SERVER` is a serial stdin loop.
- Dense decode stays on the formula above. MULTISEQ raises aggregate tokens, not the latency of one stream.
- MoE and H3 have no v199–v217 gate and no memory-floor claim. H3 activations at 768p are gigabytes.
- Apple Silicon only.
