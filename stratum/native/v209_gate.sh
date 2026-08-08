#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-/Users/shiaho/Downloads/llm/models/qwen3.6-27b-mixed.gguf}"
BIN=./stratum
[[ -x $BIN ]] || make stratum
export STRATUM_KEEP_RESIDENT=0 STRATUM_SOFT_WARM=0 STRATUM_NO_PARTIAL_WARM=1 STRATUM_NO_GPU=1
export STRATUM_TREE_B=7 STRATUM_TREE_MAXW=2 STRATUM_TREE_BRANCH=2 STRATUM_TREE_CHAIN_D2=1
export STRATUM_USER_TREE_B=1
export STRATUM_PTM_PATH=/tmp/stratum_ptm_v209_gate.bin

echo "== NOSPEC bitexact gen4 =="
STRATUM_NOSPEC=1 $BIN "$MODEL" 4 0 1 > /tmp/stratum_v209_bitexact.log 2>&1
python3 - <<'PY'
import re
t=open("/tmp/stratum_v209_bitexact.log").read()
got=re.findall(r"stratum_argmax=(\d+)", t)
if got[:4]!=["2","220","16","13"]:
    raise SystemExit(f"bitexact FAIL {got[:4]}")
print("bitexact OK", got[:4])
PY

echo "== gen8 floor =="
STRATUM_SPINE_DEBUG=1 $BIN "$MODEL" 8 0 1 > /tmp/stratum_v209_gen8.log 2>&1
rg -n "tree-step|generated|WALL|LWC|reseed|path-osf|path-ste|sd-bonus|ofp=" /tmp/stratum_v209_gen8.log | tee /tmp/stratum_v209_summary.txt
python3 - <<'PY'
import re
t=open("/tmp/stratum_v209_gen8.log").read()
m=re.search(r"generated (\d+) tokens with (\d+) main forwards \(([0-9.]+) tok/main\)", t)
if not m: raise SystemExit("missing generated")
tpm=float(m.group(3)); mains=int(m.group(2)); toks=int(m.group(1))
if tpm+1e-9 < 8.0: raise SystemExit(f"tok/main FAIL {tpm} < 8.0")
if mains != 1: raise SystemExit(f"mains FAIL {mains} != 1")
print(f"gen8 OK tok/main={tpm} mains={mains} tokens={toks}")
w=re.search(r"WALL-CLOCK: ([0-9.]+)s  =>  ([0-9.]+) tok/s", t)
if w: print(f"wall {w.group(2)} tok/s ({w.group(1)}s)")
if "path=[198 2 220 16 13 220 135051]" not in t:
    raise SystemExit("path FAIL")
ofp=re.search(r"ofp=(\d+)", t)
if not ofp or int(ofp.group(1)) < 1:
    raise SystemExit("ofp FAIL expected >=1")
print(f"ofp={ofp.group(1)}")
drafts=re.search(r"(\d+) drafts", t)
if drafts: print(f"drafts={drafts.group(1)}")
print("GATE PASS")
PY
