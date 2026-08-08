#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-./qwen3.6-27b-mixed.gguf}"
BIN=./stratum
if [[ ! -x $BIN ]]; then make stratum; fi
export STRATUM_KEEP_RESIDENT=0 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_NO_GPU=1

echo "== NOSPEC bitexact gen4 =="
STRATUM_NOSPEC=1 lldb -b -o run -o quit -- $BIN "$MODEL" 4 0 1 > /tmp/stratum_gate_bitexact.log 2>&1
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_bitexact.log").read()
got=re.findall(r"stratum_argmax=(\d+)", t)
exp=["2","220","16","13"]
if got[:4]!=exp: raise SystemExit(f"bitexact FAIL got={got[:4]}")
print("bitexact OK", got[:4])
PY

echo "== thrash gen8 =="
STRATUM_SPINE_DEBUG=1 lldb -b -o run -o quit -- $BIN "$MODEL" 8 0 1 > /tmp/stratum_gate_gen8.log 2>&1
rg -n "tree-step|generated|WALL|LWC|path-gsoed|path-soed|spine-free" /tmp/stratum_gate_gen8.log | tee /tmp/stratum_gate_summary.txt
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_gen8.log").read()
m=re.search(r"generated (\d+) tokens with (\d+) main forwards \(([0-9.]+) tok/main\)", t)
if not m: raise SystemExit("missing generated")
n,mains,tpm=float(m.group(1)),float(m.group(2)),float(m.group(3))
if tpm+1e-9 < 4.0: raise SystemExit(f"tok/main FAIL {tpm} < 4.0")
print(f"thrash OK tok/main={tpm} mains={int(mains)} tokens={int(n)}")
if "emit=6" not in t and "path=[198 2 220 16 13" not in t:
    print("WARN: expected deep SOED path")
lm=re.search(r"V202 LWC:.*?soed=(\d+)/(\d+) teb=(\d+) gsoed_lv=(\d+) preio=(\d+)", t)
if lm: print(f"soed={lm.group(1)}/{lm.group(2)} teb={lm.group(3)} gsoed_lv={lm.group(4)} preio={lm.group(5)}")
else: print("WARN: V202 LWC missing")
w=re.search(r"WALL-CLOCK: ([0-9.]+)s  =>  ([0-9.]+) tok/s", t)
if w: print(f"wall {w.group(2)} tok/s ({w.group(1)}s)")
print("GATE PASS")
PY
