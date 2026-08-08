#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-/Users/shiaho/Downloads/llm/models/qwen3.6-27b-mixed.gguf}"
BIN=./stratum
[[ -x $BIN ]] || make stratum
export STRATUM_KEEP_RESIDENT=0 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_NO_GPU=1
export STRATUM_TREE_B=7 STRATUM_TREE_MAXW=2 STRATUM_TREE_BRANCH=2 STRATUM_TREE_CHAIN_D2=1
export STRATUM_PTM_PATH=/tmp/stratum_ptm_v205_gate.bin

echo "== NOSPEC bitexact gen4 =="
STRATUM_NOSPEC=1 $BIN "$MODEL" 4 0 1 > /tmp/stratum_gate_bitexact.log 2>&1
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_bitexact.log").read()
got=re.findall(r"stratum_argmax=(\d+)", t)
if got[:4]!=["2","220","16","13"]:
    raise SystemExit(f"bitexact FAIL {got[:4]}")
print("bitexact OK", got[:4])
PY

echo "== thrash gen8 =="
STRATUM_SPINE_DEBUG=1 $BIN "$MODEL" 8 0 1 > /tmp/stratum_gate_gen8.log 2>&1
rg -n "tree-step|generated|WALL|LWC|path-gsoed|path-ste|spine-slim|spine-free|sd-bonus|V205 memory|tree-slots" /tmp/stratum_gate_gen8.log | tee /tmp/stratum_gate_summary.txt
python3 - <<'PY'
import re
t=open("/tmp/stratum_gate_gen8.log").read()
m=re.search(r"generated (\d+) tokens with (\d+) main forwards \(([0-9.]+) tok/main\)", t)
if not m: raise SystemExit("missing generated")
tpm=float(m.group(3)); mains=int(m.group(2)); toks=int(m.group(1))
if tpm+1e-9 < 4.0: raise SystemExit(f"tok/main FAIL {tpm} < 4.0")
print(f"thrash OK tok/main={tpm} mains={mains} tokens={toks}")
w=re.search(r"WALL-CLOCK: ([0-9.]+)s  =>  ([0-9.]+) tok/s", t)
if w: print(f"wall {w.group(2)} tok/s ({w.group(1)}s)")
for k in ["path-ste","sd-bonus","path-gsoed","135051","ossg_seeds"]:
    if k in t: print("marker", k)
lm=re.search(r"V205 LWC:.*?bonus=(\d+) ste=(\d+)/(\d+) ossg=(\d+)/(\d+) ptm=(\d+)/(\d+) tfan=(\d+)/(\d+) hit=(\d+)", t)
if lm:
    print(f"bonus={lm.group(1)} ste={lm.group(2)}/{lm.group(3)} ossg={lm.group(4)}/{lm.group(5)} tfan={lm.group(8)}/{lm.group(9)} hit={lm.group(10)}")
if tpm >= 7.99 and mains == 1:
    print("BREAKTHROUGH 8/1")
elif tpm > 4.0 + 1e-9:
    print("IMPROVED", tpm)
print("GATE PASS")
PY
