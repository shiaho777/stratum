#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-/Users/shiaho/Downloads/llm/models/qwen3.6-27b-mixed.gguf}"
BIN=./stratum
[[ -x $BIN ]] || make stratum
export STRATUM_KEEP_RESIDENT=0 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_NO_GPU=1

echo "== NOSPEC bitexact gen4 =="
STRATUM_NOSPEC=1 lldb -b -o run -o quit -- $BIN "$MODEL" 4 0 1 > /tmp/stratum_gate_bitexact.log 2>&1
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_bitexact.log").read()
got=re.findall(r"stratum_argmax=(\d+)", t)
if got[:4]!=["2","220","16","13"]:
    raise SystemExit(f"bitexact FAIL {got[:4]}")
print("bitexact OK", got[:4])
PY

echo "== thrash gen8 =="
STRATUM_SPINE_DEBUG=1 lldb -b -o run -o quit -- $BIN "$MODEL" 8 0 1 > /tmp/stratum_gate_gen8.log 2>&1
rg -n "tree-step|generated|WALL|LWC|path-gsoed|spine-slim|spine-free" /tmp/stratum_gate_gen8.log | tee /tmp/stratum_gate_summary.txt
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_gen8.log").read()
m=re.search(r"generated (\d+) tokens with (\d+) main forwards \(([0-9.]+) tok/main\)", t)
if not m: raise SystemExit("missing generated")
tpm=float(m.group(3))
if tpm+1e-9 < 4.0: raise SystemExit(f"tok/main FAIL {tpm} < 4.0")
print(f"thrash OK tok/main={tpm} mains={m.group(2)} tokens={m.group(1)}")
w=re.search(r"WALL-CLOCK: ([0-9.]+)s  =>  ([0-9.]+) tok/s", t)
if w: print(f"wall {w.group(2)} tok/s ({w.group(1)}s)")
if "spine-slim" in t: print("slim active")
if "path=[198 2 220 16 13 220]" in t or "emit=6" in t: print("deep path ok")
lm=re.search(r"V203 LWC:.*?slim=(\d+)/(\d+)", t)
if lm: print(f"slim={lm.group(1)}/{lm.group(2)}")
print("GATE PASS")
PY
