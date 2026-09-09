#!/usr/bin/env zsh
# hot_ab_27b.sh — the definitive hot-regime single-stream A/B, one command.
#
# Why this exists: the whole V57/COAL kernel stack (PRs #46-#60) is
# probe-validated at 170-245 GB/s, but the end-to-end hot measurement
# kept getting blocked by machine memory state (swap 87-97% all session;
# "hot" runs turned out to stream at ~5 GB/s because residency doesn't
# survive other processes' paging — see docs/hot_ab_and_nc_async_bug.txt).
# This script encodes the full protocol so the measurement is one command
# the moment the machine is clean, and REFUSES to produce a misleading
# number when it isn't.
#
# Usage: ./hot_ab_27b.sh <model.gguf> [nib.gguf]
#        (no defaults — pass your model path; the optional nib file adds
#         the Q2K-nibble-layout arm, see docs Part 7: hot-only ~1.27x
#         projected, needs its full 1.76x-expanded Q2K bytes resident)
#
# Protocol:
#   1. boundary-3 gate (h3_test_gate.sh check)
#   2. warm until weight-cache residency >= 90% AND swap < 60% — a "hot"
#      number measured while the machine swaps is a streaming number in
#      disguise (measured: ~5.3 GB/s effective vs ~200 GB/s true hot)
#   3. timed A/B: CPU vs NC+COALs vs NC+COALs+NC_ASYNC (vs CPU+nib when
#      a nib file is given) at n=4 and n=8; marginal per-token =
#      (t8 - t4)/4 strips the startup constant
#   4. sequence invariant: all configs must emit identical argmax
#   5. page-cache clean (gate protocol) on exit
set -uo pipefail
cd "$(dirname "$0")/../native" || exit 1
MODEL="${1:?usage: $0 <model.gguf> [nib.gguf]  (no default — pass your model path)}"
NIB="${2:-}"
# resolve relative model paths against the ORIGINAL cwd before we moved
case "$MODEL" in
    /*) ;;
    *) MODEL="$OLDPWD/$MODEL" ;;
esac
if [[ -n "$NIB" ]]; then
    case "$NIB" in
        /*) ;;
        *) NIB="$OLDPWD/$NIB" ;;
    esac
    [[ -r "$NIB" ]] || { echo "nib file not readable: $NIB"; exit 1; }
fi
BIN=./stratum
[[ -x $BIN ]] || { echo "build first: (cd ../native && make)"; exit 1; }

P="1 2 3 4 5 6 7 8"
COALS="STRATUM_Q2K_COAL=1 STRATUM_Q4K_COAL=1 STRATUM_Q6K_COAL=1"
TMP=$(mktemp -d /tmp/hot_ab.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

swap_pct() {
    local u t
    u=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
    t=$(sysctl -n vm.swapusage | sed 's/.*total = \([0-9.]*\)M.*/\1/')
    python3 -c "print(int(float('$u')/max(float('$t'),1)*100))"
}
now() { python3 -c "import time;print(time.time())"; }

echo "== hot_ab: $MODEL =="
./h3_test_gate.sh check || exit 1

# --- step 2: warm until genuinely hot -------------------------------------
SP=$(swap_pct)
if (( SP >= 60 )); then
    echo "❌ swap ${SP}% >= 60% — a 'hot' run now would be streaming in disguise"
    echo "   (measured: residency doesn't survive other processes' paging)."
    echo "   Free memory (close Java/emulator/browser tabs) and re-run."
    exit 2
fi
resident=0
for attempt in 1 2 3 4; do
    env STRATUM_NO_GPU=1 STRATUM_NOSPEC=1 $BIN "$MODEL" 2 ${=P} > "$TMP/warm.log" 2>&1
    resident=$(grep -oE 'resident [0-9]+%' "$TMP/warm.log" | tail -1 | grep -oE '[0-9]+')
    resident=${resident:-0}
    echo "  warm attempt $attempt: resident=${resident}%"
    if (( resident >= 90 )); then break; fi
done
if (( resident < 90 )); then
    echo "❌ residency stuck at ${resident}% (<90) — machine can't hold the model."
    echo "   This measurement would be streaming, not hot. Aborting (no fake number)."
    ./h3_test_gate.sh clean "$MODEL" > /dev/null 2>&1
    exit 3
fi
echo "  ✅ resident ${resident}% — hot regime confirmed"
if [[ -n "$NIB" ]]; then
    # the nib arm reads Q2K tensors from the nib file (1.76x-expanded) and
    # the rest from the main model — warm the nib pages too. The engine's
    # resident% only reports the MAIN model, so this warm is best-effort.
    env STRATUM_NO_GPU=1 STRATUM_Q2K_NIB="$NIB" STRATUM_NOSPEC=1 $BIN "$MODEL" 2 ${=P} > "$TMP/warm_nib.log" 2>&1
    echo "  nib warm done: $NIB"
fi

# --- step 3: timed A/B ------------------------------------------------------
run_timed() {  # $1=tag $2=env-string $3=n
    local best=999999 S E i
    for i in 1 2; do
        S=$(now)
        # ${=2}/${=P}: zsh does NOT word-split unquoted parameters — without
        # the split flag the whole env string becomes ONE assignment and the
        # prompt collapses to its first token (this exact bug silently faked
        # an entire manual A/B session before it was caught).
        env ${=2} STRATUM_NOSPEC=1 $BIN "$MODEL" $3 ${=P} > "$TMP/${1}_${3}.log" 2>&1
        E=$(now)
        best=$(python3 -c "print(min($best, $E-$S))")
    done
    echo "$best"
}
echo "== timing (best of 2 per point) =="
t_cpu4=$(run_timed cpu "STRATUM_NO_GPU=1" 4)
t_cpu8=$(run_timed cpu "STRATUM_NO_GPU=1" 8)
t_nc4=$(run_timed nc "STRATUM_GPU_NC=1 $COALS" 4)
t_nc8=$(run_timed nc "STRATUM_GPU_NC=1 $COALS" 8)
t_na4=$(run_timed ncasync "STRATUM_GPU_NC=1 $COALS STRATUM_NC_ASYNC=1" 4)
t_na8=$(run_timed ncasync "STRATUM_GPU_NC=1 $COALS STRATUM_NC_ASYNC=1" 8)
t_nib4=0; t_nib8=0
if [[ -n "$NIB" ]]; then
    t_nib4=$(run_timed nib "STRATUM_NO_GPU=1 STRATUM_Q2K_NIB=$NIB" 4)
    t_nib8=$(run_timed nib "STRATUM_NO_GPU=1 STRATUM_Q2K_NIB=$NIB" 8)
fi

python3 - "$t_cpu4" "$t_cpu8" "$t_nc4" "$t_nc8" "$t_na4" "$t_na8" "$t_nib4" "$t_nib8" << 'PY'
import sys
t = [float(x) for x in sys.argv[1:9]]
def marg(t4, t8): return (t8 - t4) / 4.0
mc, mn, ma = marg(t[0],t[1]), marg(t[2],t[3]), marg(t[4],t[5])
print(f"  CPU            marginal {mc*1000:7.1f} ms/token  ({1/mc:5.2f} tok/s)")
print(f"  NC+COALs       marginal {mn*1000:7.1f} ms/token  ({1/mn:5.2f} tok/s)  -> {mc/mn:.2f}x vs CPU")
print(f"  NC+COALs+ASYNC marginal {ma*1000:7.1f} ms/token  ({1/ma:5.2f} tok/s)  -> {mc/ma:.2f}x vs CPU")
if t[6] > 0 and t[7] > 0:
    mb = marg(t[6], t[7])
    print(f"  CPU+Q2K-NIB    marginal {mb*1000:7.1f} ms/token  ({1/mb:5.2f} tok/s)  -> {mc/mb:.2f}x vs CPU")
PY

# --- step 4: sequence invariant --------------------------------------------
seq_of() { grep -oE "stratum_argmax=[0-9]+" "$TMP/$1_8.log" | tr '\n' ' '; }
s_cpu=$(seq_of cpu); s_nc=$(seq_of nc); s_na=$(seq_of ncasync)
echo "== sequence invariant =="
echo "  cpu : $s_cpu"
echo "  nc  : $s_nc"
echo "  async: $s_na"
ok=1
if [[ "$s_cpu" != "$s_nc" || "$s_nc" != "$s_na" ]]; then ok=0; fi
if [[ -n "$NIB" ]]; then
    s_nib=$(seq_of nib)
    echo "  nib : $s_nib"
    # nib is a byte re-arrangement (values unchanged) — must be bit-identical
    [[ "$s_nib" == "$s_cpu" ]] || ok=0
fi
if (( ok )); then
    echo "  ✅ all configs bit-identical"
else
    echo "  ⚠️ divergence — on real-weight models this is a red flag; on tiny"
    echo "     random-weight models near-tie FP drift is documented (see"
    echo "     docs/hot_ab_and_nc_async_bug.txt). Check which model you passed."
fi

# --- step 5: clean ----------------------------------------------------------
./h3_test_gate.sh clean "$MODEL" 2>&1 | tail -1
[[ -n "$NIB" ]] && ./h3_test_gate.sh clean "$NIB" 2>&1 | tail -1
echo "== done. Record: machine swap was ${SP}% at start; residency ${resident}% =="
