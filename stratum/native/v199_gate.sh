#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-./qwen3.6-27b-mixed.gguf}"
BIN=./stratum
if [[ ! -x $BIN ]]; then make stratum; fi

export STRATUM_KEEP_RESIDENT=0 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_NO_GPU=1

echo "== NOSPEC bitexact gen4 =="
STRATUM_NOSPEC=1 lldb -b -o run -o quit -- $BIN "$MODEL" 4 0 1 > /tmp/stratum_gate_bitexact.log 2>&1
rg -n "stratum_argmax=" /tmp/stratum_gate_bitexact.log | tee /tmp/stratum_gate_tokens.txt
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_bitexact.log").read()
got=re.findall(r"stratum_argmax=(\d+)", t)
exp=["2","220","16","13"]
if got[:4]!=exp:
    raise SystemExit(f"bitexact FAIL got={got[:4]} want={exp}")
print("bitexact OK", got[:4])
PY

echo "== thrash gen8 =="
STRATUM_SPINE_DEBUG=1 lldb -b -o run -o quit -- $BIN "$MODEL" 8 0 1 > /tmp/stratum_gate_gen8.log 2>&1
rg -n "tree-step|generated|WALL|LWC|path-leaf" /tmp/stratum_gate_gen8.log | tee /tmp/stratum_gate_summary.txt
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_gen8.log").read()
m=re.search(r"generated (\d+) tokens with (\d+) main forwards \(([0-9.]+) tok/main\)", t)
if not m:
    raise SystemExit("missing generated line")
n, mains, tpm=float(m.group(1)), float(m.group(2)), float(m.group(3))
if tpm + 1e-9 < 2.67:
    raise SystemExit(f"tok/main FAIL {tpm} < 2.67")
if "path=[198 2 220 16]" not in t and "emit=4" not in t:
    print("WARN: expected emit=4 path through 16 not found")
print(f"thrash OK tok/main={tpm} mains={int(mains)} tokens={int(n)}")
lm=re.search(r"V199 LWC: ok=(\d+) repair=(\d+) miss=(\d+)", t)
if lm:
    print(f"LWC ok={lm.group(1)} repair={lm.group(2)} miss={lm.group(3)}")
print("GATE PASS")
PY
