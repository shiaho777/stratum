#!/usr/bin/env bash
# Stratum manifesto benchmark.
#
# Demonstrates the project's central claim: anonymous physical memory
# is independent of model size. The only scaling factor is the file-
# backed page cache, which the OS reclaims under pressure.

set -e
cd "$(dirname "$0")/.."

PROMPT="${1:-Once upon a time, in a quiet seaside village, there lived an old fisherman named Pavel}"
N="${2:-128}"
OUT="${3:-docs/manifesto_results.json}"

if [[ ! -d docs ]]; then mkdir -p docs; fi

binary="native/stratum_p35"
if [[ ! -x "$binary" ]]; then
  echo "binary $binary not found; compile first"
  exit 1
fi

# We need the binary to run for long enough that vmmap-sampling
# catches it during the steady-state generation phase, not during
# the mmap prefault sweep at startup. With N=128 tokens at ~70 tok/s
# the steady-state lasts ~2 seconds — we sample at 1 second.

run_one() {
  local label="$1"
  local args="$2"

  echo "[$label] args='$args'"
  local tmp=$(mktemp -t stratum_bench)

  # Launch binary
  ./$binary $args -n "$N" --seed 0xC0DE "$PROMPT" > "$tmp" 2>&1 &
  local pid=$!

  # Wait until prefill+prefault are finished. With prefault being
  # the largest chunk (~5s for the 367MB model on a fanless MacBook)
  # we wait until generation has clearly started.
  sleep 5.0
  local peak_mb=0
  local file_pages_mb=0
  if kill -0 "$pid" 2>/dev/null; then
    local fp_raw=$(vmmap --summary "$pid" 2>/dev/null | grep '^Physical footprint:' | awk '{print $3}')
    if [[ -n "$fp_raw" ]]; then
      case "$fp_raw" in
        *K) peak_mb=$(awk -v v="${fp_raw%K}" 'BEGIN{printf "%.2f", v/1024}') ;;
        *M) peak_mb=$(awk -v v="${fp_raw%M}" 'BEGIN{printf "%.2f", v}') ;;
      esac
    fi
    # Also check the "mapped file" RESIDENT size — this is the page
    # cache contribution, which is reclaimable under memory pressure
    # and does not count against the binding "anonymous" budget.
    local mapped=$(vmmap --summary "$pid" 2>/dev/null | grep '^mapped file' | awk '{print $3}')
    if [[ -n "$mapped" ]]; then
      case "$mapped" in
        *K) file_pages_mb=$(awk -v v="${mapped%K}" 'BEGIN{printf "%.1f", v/1024}') ;;
        *M) file_pages_mb=$(awk -v v="${mapped%M}" 'BEGIN{printf "%.1f", v}') ;;
        *G) file_pages_mb=$(awk -v v="${mapped%G}" 'BEGIN{printf "%.1f", v*1024}') ;;
      esac
    fi
  fi

  wait "$pid" 2>/dev/null

  # Parse the line "  63 gen tokens in 0.85s (74.50 tok/s)"
  local stats=$(grep 'gen tokens in' "$tmp")
  local toks=$(echo "$stats" | awk '{print $1}')
  local secs=$(echo "$stats" | awk '{print $5}' | tr -d 's')
  local rate=$(echo "$stats" | awk -F'(' '{print $2}' | awk '{print $1}')

  echo "  toks=$toks secs=$secs rate=$rate tok/s peak_phys=${peak_mb} MB file_pages=${file_pages_mb} MB"
  rm -f "$tmp"

  printf '  {"label":"%s","args":"%s","toks":%s,"secs":%s,"rate":%s,"peak_phys_mb":%s,"file_pages_mb":%s}' \
    "$label" "$args" "${toks:-0}" "${secs:-0}" "${rate:-0}" "${peak_mb:-0}" "${file_pages_mb:-0}"
}

echo "=== Stratum manifesto benchmark ==="
echo "  prompt : $PROMPT"
echo "  gen    : $N tokens"
echo

results="[\n"
results+=$(run_one "default-strict-10mb" "")
results+=",\n"
results+=$(run_one "cache-bf16" "--cache-bf16")
results+="\n]"

printf '%b\n' "$results" > "$OUT"
echo
echo "wrote $OUT"
