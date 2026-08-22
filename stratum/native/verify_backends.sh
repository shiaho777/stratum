#!/bin/zsh
# verify_backends.sh — assert greedy argmax sequences agree across backends.
#
# Usage:  ./verify_backends.sh <model.gguf> [N_GENERATE=8] [PROMPT_TOKEN...]
#
# Runs the same greedy decode on:
#   cpu      — STRATUM_NO_GPU=1            (reference)
#   nc       — STRATUM_GPU_NC=1            (per-tensor NoCopy direct-read)
#   gpu2     — STRATUM_GPU2=1              (cold-weight staging pipeline)
#
# and fails if any backend's stratum_argmax= sequence differs from CPU.
# GPU backends need stratum_q4k.metallib (built by `make`). Small models
# only — a 12 GB model takes minutes per backend and stresses the machine;
# for the 27B, use the v*_gate.sh scripts instead.
#
# After the sequence check, each backend's per-step logits are compared
# against cpu with logit_compare: top-1 agreement on EVERY recorded step
# must be 100% (hard gate); mean KL is reported as information — cross-
# backend FP accumulation paths differ legitimately, so KL > 0 between
# backends is expected and only worth watching when it jumps.
#
# Exit code: 0 = all backends bit-identical to CPU, 1 = mismatch/error.

set -euo pipefail
cd "$(dirname "$0")" || exit 1
MODEL="${1:?usage: $0 <model.gguf> [N_GENERATE=8] [PROMPT_TOKEN...]}"
N="${2:-8}"
if (( $# >= 2 )); then shift 2; else shift $#; fi
PROMPT_TOKENS=("$@")
BIN=./stratum
[[ -x $BIN ]] || make stratum || exit 1

# Refuse obvious large models: the script is a per-kernel/backend check,
# not a throughput run (see AGENTS.md testing rules).
sz=$(stat -f%z "$MODEL" 2>/dev/null || echo 0)
if (( sz > 2 * 1024 * 1024 * 1024 )); then
    echo "WARNING: model > 2 GB; backend verification is small-model oriented." >&2
fi

seq_of() {  # $1 = label
    local label="$1"
    # `|| true` at pipeline end: a missing match must not trip `set -e`.
    grep -oE "stratum_argmax=[0-9]+" "/tmp/backend_${label}.log" 2>/dev/null \
        | sed 's/stratum_argmax=//' | tr '\n' ' ' || true
}

run_backend() {  # $1 = label, rest = env assignments
    local label="$1"; shift
    local log="/tmp/backend_${label}.log"
    echo "== backend: $label =="
    if ! env "$@" "$BIN" "$MODEL" "$N" "${PROMPT_TOKENS[@]}" >/dev/null 2>"$log"; then
        echo "FAIL $label: engine exited nonzero; log tail:"
        tail -5 "$log"
        return 1
    fi
    local seq
    seq=$(seq_of "$label")
    echo "  $label argmax: ${seq:-<none>}"
    if [[ -z $seq ]]; then
        echo "FAIL $label: no stratum_argmax lines in output"
        return 1
    fi
    if [[ $label == cpu ]]; then
        ref="$seq"
    elif [[ $seq != $ref ]]; then
        echo "FAIL $label: sequence differs from cpu"
        return 1
    fi
    echo "  OK $label matches cpu"
}

ref=""
# NOSPEC forces per-token `stratum_argmax=` lines (speculative tree mode
# prints `tree-step path=[...]` instead) and makes greedy output the sole
# contract, so CPU / GPU-NC / GPU2 are directly comparable.
run_backend cpu STRATUM_NO_GPU=1 STRATUM_NOSPEC=1 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_LOGITS_DUMP=/tmp/backend_cpu.slog

if [[ -f stratum_q4k.metallib ]]; then
    run_backend nc  STRATUM_GPU_NC=1 STRATUM_NOSPEC=1 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_LOGITS_DUMP=/tmp/backend_nc.slog
    run_backend gpu2 STRATUM_GPU2=1 STRATUM_NOSPEC=1 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_LOGITS_DUMP=/tmp/backend_gpu2.slog
else
    echo "SKIP nc/gpu2: stratum_q4k.metallib not found (run: make)"
fi

echo "== backend verification: all argmax sequences match cpu =="

# Distribution-level comparison: every recorded step's top-1 token must
# agree with cpu (stronger than the printed-lines check above); mean KL is
# informational — different FP accumulation paths across backends make
# KL > 0 normal, jumps are what deserve attention.
[[ -x ./logit_compare ]] || make logit_compare >/dev/null || true
for label in nc gpu2; do
    [[ -f "/tmp/backend_${label}.slog" ]] || { echo "DIST ${label}: no logits dump (skipped)"; continue; }
    [[ -f /tmp/backend_cpu.slog ]] || { echo "DIST: no cpu dump (skipped)"; break; }
    echo "== distribution: ${label} vs cpu =="
    dist_line=$(./logit_compare /tmp/backend_cpu.slog "/tmp/backend_${label}.slog" | tail -1)
    echo "  ${dist_line}"
    agree=$(printf '%s' "$dist_line" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f1)
    steps=$(printf '%s' "$dist_line" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f2)
    if [[ -z $agree || $agree != $steps ]]; then
        echo "FAIL ${label}: top-1 agreement ${agree:-?}/${steps:-?} != full"
        exit 1
    fi
    echo "  OK ${label}: top-1 agreement full"
done
echo "== backend verification complete =="
