#!/bin/zsh
# run_all_cases.sh — measured head-to-head: stratum vs llama.cpp.
# Each (model x engine) case runs twice; run 2 (hot page cache) is kept.
# Results -> results/<model>_<engine>_run{1,2}.json
# 2026-refresh: llama side uses stock `llama-completion` (b9180, honors
# --ignore-eos natively, non-interactive). stratum GPU pass added.
set -u
cd "$(dirname "$0")"

STRATUM=/Users/shiaho/Desktop/Qwen3.5-0.8B-hf/stratum/native/stratum
LLAMA=/opt/homebrew/bin/llama-completion
Q4=/Users/shiaho/Desktop/0-/Qwen3-0.6B/Qwen3-0.6B.q4km.llamacpp.gguf
PROMPT_TEXT='<|im_start|>user
What is the capital of France? Answer in one short sentence.<|im_end|>
<|im_start|>assistant
'
PROMPT_IDS="151644,872,198,3838,374,279,6722,315,9625,30,21806,304,825,2805,11652,13,151645,198,151644,77091,198"
NGEN=192

mkdir -p results
for run in 1 2; do
  TAG=q4km; MODEL=$Q4
  # speed pass (no vmmap — sampling suspends the target and skews timing)
  python3 run_h2h.py $LLAMA  $MODEL llamacpp    "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_speed_run${run}.json --nomen
  python3 run_h2h.py $LLAMA  $MODEL llamacpp    "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_gpu_speed_run${run}.json --nomen --ngl=99
  python3 run_h2h.py $STRATUM $MODEL stratum    "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_speed_run${run}.json --nomen
  python3 run_h2h.py $STRATUM $MODEL stratum_gpu "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_gpu_speed_run${run}.json --nomen
  python3 run_h2h.py $STRATUM $MODEL stratum_gpu "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_gpu_chain_speed_run${run}.json --nomen --chain
  # memory pass (vmmap sampling; wall time here is not comparable)
  python3 run_h2h.py $LLAMA  $MODEL llamacpp    "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_mem_run${run}.json
  python3 run_h2h.py $LLAMA  $MODEL llamacpp    "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_gpu_mem_run${run}.json --ngl=99
  python3 run_h2h.py $STRATUM $MODEL stratum    "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_mem_run${run}.json
  python3 run_h2h.py $STRATUM $MODEL stratum_gpu "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_gpu_mem_run${run}.json
done
echo "=== all cases done ==="
