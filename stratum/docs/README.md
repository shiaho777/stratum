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

The scripts below are the only required entry points. **None of them hardcode a model path — pass yours as an argument.**

- **headtohead**: `stratum/benchmarks/headtohead.sh <model.gguf> [N_GEN]`
- **manifesto (current revision)**: `stratum/benchmarks/manifesto_v2.sh <model.gguf> [N_GEN] [PROMPT_IDS...]`

All run on the CPU path (`STRATUM_NO_GPU=1`), greedy decoding.

The other records (`manifesto_results.json`, `memory_floor.json`, `pressure_test_results.txt`) were captured against earlier engine revisions whose harnesses lived in the removed Python-era line; the data is kept as historical evidence, but those harnesses are not part of the current tree.

## What was removed

- ~150 additional JSON files recording per-kernel parameter sweeps (threadgroup sizes, batch widths, fused-op variants) from the development log. They were internal tuning records — not reproducible against the current code and not meaningful to readers without the surrounding development context. They are preserved in the git history; only the externally-facing evidence is kept in the tree.
- Python-era benchmark harnesses (`run_all.py`, `memory_floor*.py`, `*_worker.py`, `manifesto.sh`, `pressure_test.sh`, `tests/baseline_truth.json`) that referenced deleted model directories or the removed `stratum_p35` binary. Removed in the same cleanup.
