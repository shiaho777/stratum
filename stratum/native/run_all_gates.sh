#!/bin/zsh
# run_all_gates.sh — run every v*_gate.sh in this directory against one model.
#
# Usage:  ./run_all_gates.sh <model.gguf>
#
# Each gate asserts bit-exact greedy output (argmax [2,220,16,13] and related
# tree/stat invariants) on the qwen35 architecture. All gates take the model
# as $1 and set their own STRATUM_* environment, so no configuration is needed
# here beyond the model path.
#
# Exit code: 0 if every gate passed, 1 if any failed. Failed gates print the
# tail of their log so a "feature missing on this model" failure (e.g. no MTP
# head) is distinguishable from a real regression.

set -u
cd "$(dirname "$0")" || exit 1
MODEL="${1:?usage: $0 <model.gguf>}"
BIN=./stratum
[[ -x $BIN ]] || make stratum || exit 1

pass=0
fail=0
for g in v*_gate.sh; do
    [[ -x $g ]] || continue
    log="/tmp/gate_${g}_$(basename "$MODEL").log"
    printf "=== %s ===\n" "$g"
    if ./"$g" "$MODEL" >"$log" 2>&1; then
        echo "PASS $g"
        pass=$((pass + 1))
    else
        rc=$?
        echo "FAIL $g (rc=$rc) — log tail:"
        tail -8 "$log"
        fail=$((fail + 1))
    fi
done

echo "== gates: $pass passed, $fail failed =="
[[ $fail -eq 0 ]]
