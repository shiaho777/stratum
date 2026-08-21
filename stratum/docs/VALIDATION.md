# Stratum validation matrix

How the engine is verified, what the gates do and do not cover, and what to
run before shipping a change. Keep this in sync with AGENTS.md (Testing).

## Quick commands

```sh
cd stratum/native
make tests                                  # quant kernels + sampler (no model needed)
make check MODEL=/path/to/model.gguf        # tests + one llama-arch smoke
./run_all_gates.sh /path/to/qwen35.gguf     # every v*_gate.sh in sequence
./verify_backends.sh /path/to/small.gguf    # CPU vs GPU-NC vs GPU2 argmax identity
python3 stratum/tools/env_census.py         # regenerate docs/ENVVARS.md
python3 stratum/tools/make_tiny_model.py --out /tmp/tiny.gguf
STRATUM_NO_GPU=1 ./stratum /tmp/tiny.gguf 4 1 2 3 4 5 6 7 8   # llama smoke
```

## What the gates cover (v199–v217)

Every gate is a full-model greedy run on the **qwen35 architecture** and
asserts at least:

- the exact argmax sequence `[2, 220, 16, 13]` (bit-exact decode contract),
- `tok/main >= 8.0` and related tree/MTP acceptance statistics,
- scheduler/IO counters expected for that gate's specific scenario.

They all run with `STRATUM_NO_GPU=1` (CPU path), `STRATUM_KEEP_RESIDENT=0`,
`STRATUM_SOFT_WARM=0`, `STRATUM_NO_PARTIAL_WARM=1`, and per-gate speculative
decode settings. **Any engine change must keep every gate passing.**

## What the gates do NOT cover

- the **llama** architecture at full scale: CI runs a generated deterministic
  tiny model (`stratum/tools/make_tiny_model.py`, no weights in the repo) and
  pins its greedy sequence — real end-to-end coverage, but at toy scale only;
  a real llama-family GGUF still has no gate,
- **small models** (gates target the 27B; small-model argmax identity is
  checked by `verify_backends.sh` on whatever small GGUF you have),
- **GPU paths** (Metal, GPU2 staging, GPU-NC) — gates are CPU-only,
- every quant format combo (covered by `quant_test`, model-free),
- **MULTISEQ** stream-count variants, long-context KV ring wrap,
- Q2K nibble sidecar vs embedded type-42,
- MemX-enabled builds (`USE_MEMX=1`),
- memory-pressure / swap behavior.

"Gates pass" therefore means *the default CPU qwen35 path is unregressed*, not
that every backend is safe.

## Per-kernel change checklist

A change to a quantized kernel or its dispatch must be verified through the
full chain, not just the default path:

1. `make tests` — quant_test cross-validates scalar/NEON/SDOT kernels against
   each other; spec_sample_test checks sampler exactness.
2. `./verify_backends.sh <small.gguf>` — CPU vs GPU-NC vs GPU2 argmax identity
   (needs `stratum_q4k.metallib`).
3. If Q2K touched: nibble layout (`STRATUM_Q2K_NIB=<sidecar>`, embedded
   type-42) both must match the original layout bit-exactly.
4. If scheduling/IO touched: run the v*_gate.sh for the affected scenario
   (`run_all_gates.sh`).

## Before large-model runs

Per AGENTS.md: check `vm_stat` free pages and `sysctl vm.swapusage` first;
do not start a 27B run when the system is under pressure or when heavy
background processes are active. Prefer 1–2 token correctness checks; a full
8-token speed run is only acceptable when the model is already hot in page
cache and free memory allows.

## Env-var map

`stratum/tools/env_census.py` regenerates `docs/ENVVARS.md` — the single map
of every `STRATUM_*` switch, its call sites, and its
sanctioned/forbidden/experimental status. Re-run it whenever a switch is
added, removed, or reclassified.
