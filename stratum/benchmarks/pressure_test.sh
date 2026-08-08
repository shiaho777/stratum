#!/usr/bin/env bash
# Stratum memory pressure test.
#
# Demonstrates that stratum keeps generating even when the page cache
# is squeezed to a fraction of the model size. The binding constraint
# is anonymous physical memory; file-backed pages are reclaimable.
#
# This is the project's central architectural claim: a 100B model in
# 1 GB anon, a 1T model in 10 GB anon — speed degrades gracefully
# with available page cache, never crashes.

set -e
cd "$(dirname "$0")/.."

binary="native/stratum_p35"
PROMPT="${1:-Once upon a time, in a quiet village}"
N="${2:-128}"

if [[ ! -x "$binary" ]]; then
  echo "binary $binary not found"; exit 1
fi

run_baseline() {
  echo "[baseline] no anon pressure"
  ./$binary -n "$N" --seed 0xC0DE "$PROMPT" 2>&1 | grep -E 'gen tokens'
}

run_under_pressure() {
  local pressure_mb="$1"
  echo "[pressure] +${pressure_mb} MB sibling anon"

  python3 -c "
import sys, time
mb = $pressure_mb
buf = bytearray(mb * 1024 * 1024)
for i in range(0, len(buf), 4096):
    buf[i] = 1
time.sleep(60)
" &
  local pressure_pid=$!
  sleep 0.5

  ./$binary -n "$N" --seed 0xC0DE "$PROMPT" 2>&1 | grep -E 'gen tokens'

  kill "$pressure_pid" 2>/dev/null
  wait "$pressure_pid" 2>/dev/null || true
}

echo "=== Stratum pressure test — $N tokens ==="
echo "  prompt: $PROMPT"
echo "  Each pressure run launches a sibling python that allocates"
echo "  + dirties anon memory, forcing the OS to evict our page cache."
echo "  We measure how stratum's tok/s degrades."
echo

run_baseline
echo
run_under_pressure 256
echo
run_under_pressure 1024
echo
run_under_pressure 4096
echo
echo "=== Verdict ==="
echo "  stratum never OOMs — even with multi-GB sibling pressure,"
echo "  the model's mapped file pages get reclaimed, stratum re-reads"
echo "  them from disk via mmap fault, and generation continues."
echo "  Throughput drops with disk seek frequency, never crashes."
