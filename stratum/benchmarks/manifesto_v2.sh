#!/usr/bin/env bash
# Stratum manifesto benchmark — v2 (universal GGUF).
#
# Same idea as benchmarks/manifesto.sh but for stratum_v2, which
# accepts ANY Llama-architecture GGUF. We sample vmmap during the
# steady-state generation phase to read:
#
#   peak_phys_mb   physical footprint as reported by vmmap (anon + file)
#   anon_mb        anonymous bytes only (the binding "RAM cost")
#   file_pages_mb  resident bytes for the mmap'ed model file
#                  (reclaimable under memory pressure — NOT a real cost)
#
# Output: docs/manifesto_v2_<modelname>.json
#
# usage: benchmarks/manifesto_v2.sh <model.gguf> [N_GEN] [PROMPT_IDS...]

set -e
cd "$(dirname "$0")/.."

MODEL="${1:?model.gguf path required}"
shift || true
N="${1:-12}"
shift || true
# Default prompt ids = "The capital of France is" for TinyLlama
if [[ $# -eq 0 ]]; then
  PROMPT_IDS=(1 450 7483 310 3444 338)
else
  PROMPT_IDS=("$@")
fi

binary="native/stratum"
if [[ ! -x "$binary" ]]; then
  echo "binary $binary not found; compile first"
  exit 1
fi

modelname=$(basename "$MODEL" .gguf)
mkdir -p docs
out="docs/manifesto_v2_${modelname}.json"

echo "=== Stratum v2 manifesto benchmark ==="
echo "  model : $MODEL"
echo "  size  : $(ls -lh "$MODEL" | awk '{print $5}')"
echo "  N gen : $N"
echo "  prompt: ${PROMPT_IDS[*]}"
echo

# Launch the binary, then sample vmmap during steady-state.
tmp=$(mktemp -t stratum_v2_bench)
$binary "$MODEL" "$N" "${PROMPT_IDS[@]}" > "$tmp" 2>&1 &
pid=$!

# Sample at 0.5s intervals until process exits, take peak.
peak_phys_mb=0
peak_anon_mb=0
peak_file_mb=0
samples=0

parse_size() {
  # vmmap --summary prints values like "8192K" "23M" "1.5G"
  local v="$1"
  case "$v" in
    *K) awk -v x="${v%K}" 'BEGIN{printf "%.3f", x/1024}' ;;
    *M) awk -v x="${v%M}" 'BEGIN{printf "%.3f", x}' ;;
    *G) awk -v x="${v%G}" 'BEGIN{printf "%.3f", x*1024}' ;;
    *)  echo "0" ;;
  esac
}

max() { awk -v a="$1" -v b="$2" 'BEGIN{print (a>b)?a:b}'; }

while kill -0 "$pid" 2>/dev/null; do
  summary=$(vmmap --summary "$pid" 2>/dev/null || true)
  if [[ -n "$summary" ]]; then
    # macOS vmmap reports two key numbers in --summary:
    #   "Physical footprint" — anonymous + writable resident bytes
    #     (this IS the binding RAM cost — it's what "ps -o rss" wishes
    #     it could be, and what shows up in the "Memory" column of
    #     Activity Monitor as "Memory" minus "Compressed Memory" minus
    #     "Cached Files".)
    #   "mapped file"        — file-backed page-cache resident bytes
    #     (reclaimable under pressure — NOT a binding cost.)
    fp=$(echo "$summary" | awk '/^Physical footprint:/ {print $3; exit}')
    mapped=$(echo "$summary" | awk '/^mapped file/ && !/SM=COW/ {print $3; exit}')
    fp_mb=$(parse_size "${fp:-0K}")
    file_mb=$(parse_size "${mapped:-0K}")
    # On macOS, "Physical footprint" already excludes mapped files,
    # so it is itself the anonymous total.
    anon_mb="$fp_mb"
    total_mb=$(awk -v a="$fp_mb" -v b="$file_mb" 'BEGIN{printf "%.3f", a+b}')
    peak_phys_mb=$(max "$peak_phys_mb" "$total_mb")
    peak_anon_mb=$(max "$peak_anon_mb" "$anon_mb")
    peak_file_mb=$(max "$peak_file_mb" "$file_mb")
    samples=$((samples + 1))
  fi
  sleep 0.5
done

wait "$pid" 2>/dev/null || true

echo "=== Output (last 8 lines) ==="
tail -8 "$tmp"
echo
echo "=== Footprint (peak across $samples samples) ==="
printf "  physical footprint : %8.2f MB\n" "$peak_phys_mb"
printf "  mapped file (page cache, reclaimable) : %8.2f MB\n" "$peak_file_mb"
printf "  anonymous (binding RAM cost)          : %8.2f MB\n" "$peak_anon_mb"

model_size_mb=$(awk -v b="$(stat -f%z "$MODEL")" 'BEGIN{printf "%.2f", b/1024/1024}')

cat > "$out" <<EOF
{
  "model": "$MODEL",
  "model_size_mb": $model_size_mb,
  "n_gen": $N,
  "samples": $samples,
  "peak_physical_mb": $peak_phys_mb,
  "peak_mapped_file_mb": $peak_file_mb,
  "peak_anonymous_mb": $peak_anon_mb,
  "ratio_anon_to_model": $(awk -v a="$peak_anon_mb" -v b="$model_size_mb" 'BEGIN{printf "%.4f", a/b}')
}
EOF

echo
echo "wrote $out"
rm -f "$tmp"
