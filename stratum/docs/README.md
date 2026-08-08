# Measured Evidence (docs/)

This directory holds the **measured evidence behind the claims in the root README**. Every file here is a real benchmark run — the numbers in the README (memory ratio, pressure resilience, memory floor) are backed by these records.

| File | What it evidences |
|---|---|
| `headtohead_tinyllama-1.1b-chat-q4km.json` | Stratum vs llama.cpp on TinyLlama 1.1B: anonymous memory (7.4 MB vs 634.9 MB = **85.7× lower**) and wall time |
| `headtohead_llama-3.2-3b-instruct-q4km.json` | Same comparison on Llama 3.2 3B |
| `manifesto_results.json` | Whole-engine anonymous-memory measurement |
| `manifesto_v2_tinyllama-1.1b-chat-q4km.json` | Re-measured manifesto with the current engine revision |
| `memory_floor.json` | Memory-floor experiment: wired memory as a function of model size |
| `pressure_test_results.txt` | Pressure test: how tok/s degrades when sibling processes force page-cache eviction |

## Reproducing

- **headtohead**: `stratum/benchmarks/headtohead.sh <model.gguf> [N_GEN]`
- **manifesto**: `stratum/benchmarks/manifesto.sh <model.gguf>`
- **memory floor**: `stratum/benchmarks/memory_floor.py <model.gguf>`
- **pressure test**: `stratum/benchmarks/pressure_test.sh <model.gguf>`

All run on the CPU path (`STRATUM_NO_GPU=1`), greedy decoding.

## What was removed

The original archive contained ~150 additional JSON files recording per-kernel parameter sweeps (threadgroup sizes, batch widths, fused-op variants) from the development log. They were internal tuning records — not reproducible against the current code and not meaningful to readers without the surrounding development context. They are preserved in the git history; only the externally-facing evidence is kept in the tree.
