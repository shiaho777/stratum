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
./h3_test_gate.sh check                     # boundary-3 pre-run gate for large-model tests
./h3_test_gate.sh clean <model.gguf>        # release page cache after a large run
python3 stratum/tools/env_census.py         # regenerate docs/ENVVARS.md (preserves the H3 appendix)
python3 stratum/tools/make_tiny_model.py --arch llama --out /tmp/tiny.gguf
STRATUM_NO_GPU=1 ./stratum /tmp/tiny.gguf 4 1 2 3 4 5 6 7 8   # llama smoke
python3 stratum/tools/make_tiny_model.py --arch moe --out /tmp/tinymoe.gguf
python3 stratum/tools/tiny_moe_oracle.py /tmp/tinymoe.gguf 0 1   # MoE numpy cross-check
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

## What the gates cover beyond v199–v217

- **mini-DiT sequence forward** (`dit_probe`, a `make` target): CI generates
  a mini-DiT model (`make_tiny_model.py --arch dit`: packed (t,h,w) grid,
  MM-RoPE, bidirectional attention, AdaLN-lite) and runs the probe against
  the torch oracle (`dit_oracle.py`) at max|diff| ≤ 5e-7. `dit_sample.c`
  adds the flow-matching Euler sampler loop on top (steps=1 must equal one
  probe forward).
- **MoE tiny cross-check** (local, not CI): `make_tiny_model.py --arch moe`
  + `tiny_moe_oracle.py` reimplement the `llama-moe` arithmetic (dense
  attention + router top-k expert FFN, deterministic tie-break) in numpy
  and compare greedy sequences with the engine.
- **H3 large-run hygiene** (`h3_test_gate.sh`): `check` refuses to start
  when available memory is < 1 GB or a sibling large-model process is
  running; `clean <model…>` drops the used files from page cache
  (mincore-verified) so the next test starts from a known state.

## What the gates do NOT cover

- the **llama** architecture at full scale: CI runs generated deterministic
  tiny models (`stratum/tools/make_tiny_model.py`, no weights in the repo) and
  pins their greedy sequences — real end-to-end coverage for llama, qwen35
  (pure full-attn), and the **qwen35 hybrid layout** (Gated DeltaNet SSM
  layers + full attention), but at toy scale only; real-family GGUFs of these
  still have no gate,
- the **MoE** architecture beyond the tiny oracle (no CI gate, no
  full-scale validation),
- the **H3 video pipeline** (denoiser / VAE / sampler spikes): validated by
  dump identity (md5 / max|d| between paths), never by token argmax; int8
  SDOT is killed there (diffusion has no argmax to hide behind), and the
  kernels default to the conservative path pending gate decisions — see the
  `ENVVARS.md` H3 appendix for the current per-kernel status,
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
5. If the H3/DiT spikes touched: re-run the oracle comparison for that
   spike (`dit_probe` vs `dit_oracle.py`; `tiny_moe_oracle.py`;
   H3 packed-dump md5 / max|d| between the affected paths). H3 numerics
   have their own rule: no int8 prequant in the denoiser, and e2e A/B
   under swap pressure swings ±9% run-to-run — sub-second effects are only
   resolvable in micro-benchmarks, never in e2e wall time.

## Before large-model runs

Per AGENTS.md: check `vm_stat` free pages and `sysctl vm.swapusage` first
(`./h3_test_gate.sh check` automates the refusal conditions); do not start
a 27B run when the system is under pressure or when heavy background
processes are active. Prefer 1–2 token correctness checks; a full 8-token
speed run is only acceptable when the model is already hot in page cache
and free memory allows. After the run, `./h3_test_gate.sh clean <model…>`
returns the resident weight pages so the next test starts from a known
cache state.

## Env-var map

`stratum/tools/env_census.py` regenerates `docs/ENVVARS.md` — the single map
of every `STRATUM_*` switch, its call sites, and its
sanctioned/forbidden/experimental status. Re-run it whenever a switch is
added, removed, or reclassified. The script preserves the hand-maintained
H3 appendix below the generated table (H3 spike switches are mostly `H3_*`
without the `STRATUM_` prefix, so the scan cannot see them — edit that
section by hand, never by re-running the census).
