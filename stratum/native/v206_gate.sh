#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-/Users/shiaho/Downloads/llm/models/qwen3.6-27b-mixed.gguf}"
BIN=./stratum
[[ -x $BIN ]] || make stratum
export STRATUM_KEEP_RESIDENT=0 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_NO_GPU=1
export STRATUM_TREE_B=7 STRATUM_TREE_MAXW=2 STRATUM_TREE_BRANCH=2 STRATUM_TREE_CHAIN_D2=1
export STRATUM_PTM_PATH=/tmp/stratum_ptm_v206_gate.bin

echo "== NOSPEC bitexact gen4 =="
STRATUM_NOSPEC=1 $BIN "$MODEL" 4 0 1 > /tmp/stratum_v206_bitexact.log 2>&1
python3 - <<'PY'
import re
t=open("/tmp/stratum_v206_bitexact.log").read()
got=re.findall(r"stratum_argmax=(\d+)", t)
if got[:4]!=["2","220","16","13"]:
    raise SystemExit(f"bitexact FAIL {got[:4]}")
print("bitexact OK", got[:4])
PY

echo "== gen8 floor =="
STRATUM_SPINE_DEBUG=1 $BIN "$MODEL" 8 0 1 > /tmp/stratum_v206_gen8.log 2>&1
rg -n "tree-step|generated|WALL|LWC|path-osf|path-ste|spine-slim|sd-bonus|path-osf-seal|path-ste-seal|tree-slots" /tmp/stratum_v206_gen8.log | tee /tmp/stratum_v206_summary.txt
python3 - <<'PY'
import re
t=open("/tmp/stratum_v206_gen8.log").read()
m=re.search(r"generated (\d+) tokens with (\d+) main forwards \(([0-9.]+) tok/main\)", t)
if not m: raise SystemExit("missing generated")
tpm=float(m.group(3)); mains=int(m.group(2)); toks=int(m.group(1))
if tpm+1e-9 < 8.0: raise SystemExit(f"tok/main FAIL {tpm} < 8.0 (V206 floor)")
if mains != 1: raise SystemExit(f"mains FAIL {mains} != 1")
print(f"gen8 OK tok/main={tpm} mains={mains} tokens={toks}")
w=re.search(r"WALL-CLOCK: ([0-9.]+)s  =>  ([0-9.]+) tok/s", t)
if w: print(f"wall {w.group(2)} tok/s ({w.group(1)}s)")
if "135051" not in t: raise SystemExit("missing title token 135051 in debug path")
if "path=[198 2 220 16 13 220 135051]" not in t and "220 135051" not in t:
    print("WARN path marker soft-check")
lm=re.search(r"V206 LWC:.*?bonus=(\d+) ste=(\d+)/(\d+) ossg=(\d+)/(\d+) ptm=(\d+)/(\d+) tfan=(\d+)/(\d+) hit=(\d+) osf=(\d+)/(\d+)", t)
if lm:
    print(f"bonus={lm.group(1)} ste={lm.group(2)}/{lm.group(3)} ossg={lm.group(4)}/{lm.group(5)} tfan={lm.group(8)}/{lm.group(9)} hit={lm.group(10)} osf={lm.group(11)}/{lm.group(12)}")
    if int(lm.group(10)) < 1: raise SystemExit("tfan hit FAIL")
print("BREAKTHROUGH 8/1")
print("GATE PASS")
PY
