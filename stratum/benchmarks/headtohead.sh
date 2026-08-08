#!/usr/bin/env bash
# Head-to-head benchmark: stratum_v2 vs llama.cpp on the same GGUF model.
#
# Measures the manifesto's central claim: anonymous physical memory.
# Anonymous = "private writable RAM" = the binding cost. mmap'ed file
# pages are reclaimable under pressure and don't count.
#
# usage: benchmarks/headtohead.sh <model.gguf> [N_GEN]

set -e
cd "$(dirname "$0")/.."

MODEL="${1:?model.gguf path required}"
N="${2:-64}"
# Default prompt IDs are TinyLlama's "The capital of France is".
# For Llama 3, pass `--ids "128000 791 6864 315 9822 374"` explicitly.
PROMPT_TEXT="${3:-The capital of France is}"
PROMPT_IDS="${4:-1 450 7483 310 3444 338}"

modelname=$(basename "$MODEL" .gguf)
mkdir -p docs
out="docs/headtohead_${modelname}.json"

if [[ ! -x native/stratum ]]; then
  echo "native/stratum not built; please compile first."
  exit 1
fi
if ! command -v llama-simple >/dev/null 2>&1; then
  echo "llama-simple not found in PATH; please install llama.cpp."
  exit 1
fi

parse_size() {
  local v="$1"
  case "$v" in
    *K) awk -v x="${v%K}" 'BEGIN{printf "%.3f", x/1024}' ;;
    *M) awk -v x="${v%M}" 'BEGIN{printf "%.3f", x}' ;;
    *G) awk -v x="${v%G}" 'BEGIN{printf "%.3f", x*1024}' ;;
    *)  echo "0" ;;
  esac
}

max() { awk -v a="$1" -v b="$2" 'BEGIN{print (a>b)?a:b}'; }

sample_pid() {
  # Polls vmmap on $1 every 0.2s until the process exits.
  # Echos "peak_phys_mb peak_anon_mb peak_file_mb samples"
  local pid=$1
  local peak_phys=0 peak_anon=0 peak_file=0 samples=0
  while kill -0 "$pid" 2>/dev/null; do
    local s=$(vmmap --summary "$pid" 2>/dev/null || true)
    if [[ -n "$s" ]]; then
      local fp=$(echo "$s" | awk '/^Physical footprint:/ {print $3; exit}')
      local mf=$(echo "$s" | awk '/^mapped file/ && !/SM=COW/ {print $3; exit}')
      local fp_mb=$(parse_size "${fp:-0K}")
      local mf_mb=$(parse_size "${mf:-0K}")
      # On macOS, "Physical footprint" excludes mapped files. So:
      #   anon  = Physical footprint
      #   total = anon + mapped file resident
      local total=$(awk -v a="$fp_mb" -v b="$mf_mb" 'BEGIN{printf "%.3f", a+b}')
      peak_phys=$(max "$peak_phys" "$total")
      peak_anon=$(max "$peak_anon" "$fp_mb")
      peak_file=$(max "$peak_file" "$mf_mb")
      samples=$((samples + 1))
    fi
    sleep 0.2
  done
  echo "$peak_phys $peak_anon $peak_file $samples"
}

echo "=== Head-to-head: stratum_v2 vs llama.cpp ==="
echo "  model : $MODEL ($(ls -lh "$MODEL" | awk '{print $5}'))"
echo "  N gen : $N"
echo

# ---- stratum_v2 ----
echo "[1/2] stratum_v2 ..."
t0=$(date +%s.%N)
./native/stratum "$MODEL" "$N" $PROMPT_IDS > /tmp/strat_out 2>&1 &
spid=$!
read s_total s_anon s_file s_samples < <(sample_pid "$spid")
wait "$spid" 2>/dev/null || true
t1=$(date +%s.%N)
s_secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

echo "  total wall: ${s_secs}s, samples: $s_samples"
echo "  anonymous (binding RAM)   : ${s_anon} MB"
echo "  mapped file (reclaimable) : ${s_file} MB"
echo "  total physical            : ${s_total} MB"
echo

# ---- llama.cpp ----
echo "[2/2] llama-simple (CPU only) ..."
t0=$(date +%s.%N)
llama-simple -m "$MODEL" -n "$N" -ngl 0 "$PROMPT_TEXT" > /tmp/llamacpp_out 2>&1 &
lpid=$!
read l_total l_anon l_file l_samples < <(sample_pid "$lpid")
wait "$lpid" 2>/dev/null || true
t1=$(date +%s.%N)
l_secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

echo "  total wall: ${l_secs}s, samples: $l_samples"
echo "  anonymous (binding RAM)   : ${l_anon} MB"
echo "  mapped file (reclaimable) : ${l_file} MB"
echo "  total physical            : ${l_total} MB"
echo

ratio=$(awk -v a="$l_anon" -v b="$s_anon" 'BEGIN{if(b>0) printf "%.1f", a/b; else print "inf"}')
echo "=== Anonymous RAM: stratum / llama.cpp ratio = 1 / ${ratio}x ==="

cat > "$out" <<EOF
{
  "model": "$MODEL",
  "n_gen": $N,
  "stratum": {
    "wall_secs": $s_secs,
    "peak_anon_mb": $s_anon,
    "peak_mapped_file_mb": $s_file,
    "peak_total_phys_mb": $s_total,
    "samples": $s_samples
  },
  "llamacpp": {
    "wall_secs": $l_secs,
    "peak_anon_mb": $l_anon,
    "peak_mapped_file_mb": $l_file,
    "peak_total_phys_mb": $l_total,
    "samples": $l_samples
  },
  "anon_ratio_llamacpp_over_stratum": $ratio
}
EOF
echo "wrote $out"
