# Contributing to Stratum

## Quick start

```sh
git clone https://github.com/shiaho777/stratum.git
cd stratum/stratum/native
make              # builds ./stratum + metallib
make check        # auto-generates a test model and runs inference
```

No weights needed — `make check` generates deterministic tiny models.

## Development workflow

1. **Open an issue** describing what you want to change
2. **Branch** from `main` (`git checkout -b your-branch`)
3. **Make changes**, run `make tests` (quant kernels + sampler)
4. **Push and open a PR** into `main`
5. **CI must pass** (build, sanitizers, inference smoke) before merge

## Key files

| File | Purpose |
|---|---|
| `AGENTS.md` | Development guide: boundaries, env vars, determinism contract |
| `stratum/docs/VALIDATION.md` | What the gates cover and don't |
| `stratum/docs/ENVVARS.md` | All STRATUM_* switches (generated table + hand-maintained H3 appendix) — regenerate with `python3 stratum/tools/env_census.py` |

## Hard boundaries

These are enforced in code at startup:

1. **Never harm quality** — no requantization, no skipped layers, no approximate compute
2. **Never grow wired memory** — weights stream via mmap; only KV/SSM state is resident
3. **Testing must not exhaust the machine** — check `vm_stat` before big runs
4. **Generality** — no model-specific names in dispatch code

## Adding an architecture

Create `stratum_arch_<name>.inc.c`, implement the `StratumArch` interface,
call `STRATUM_REGISTER_ARCH()`. The Makefile auto-collects `*.inc.c` files.
See `stratum_arch_llama.inc.c` (~1.3k lines) for a minimal example,
`stratum_arch_moe.inc.c` for an experimental one.

Standalone spikes (`h3_*.c`, `dit_*.c`, `te_*.c`) are NOT architectures —
they are separate binaries with their own mains. Do not register them;
document their switches in the `ENVVARS.md` H3 appendix by hand.

## Code style

- C11, compiled with clang `-O2/-O3`, no C++
- Hot path uses NEON intrinsics (`__ARM_NEON`)
- All public functions `static inline` in headers (single-TU compilation)
- Comments explain *why*, not *what*
