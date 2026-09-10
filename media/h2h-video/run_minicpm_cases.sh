#!/bin/zsh
# run_minicpm_cases.sh — MiniCPM5-2B head-to-head: stratum vs llama.cpp
# (llama.cpp built from current master: official GGUF needs the new
#  'minicpm5' pre-tokenizer, which Homebrew b9180 rejects).
# Same two-pass, two-run methodology as run_all_cases.sh; run 2 (hot) kept.
set -u
cd "$(dirname "$0")"

STRATUM=/Users/shiaho/Desktop/Qwen3.5-0.8B-hf/stratum/native/stratum
LLAMA=/tmp/llama.cpp/build/bin/llama-simple
Q4=/Users/shiaho/Desktop/0-/MiniCPM5-2B/MiniCPM5-2B-Q4_K_M.gguf
F16=/Users/shiaho/Desktop/0-/MiniCPM5-2B/MiniCPM5-2B-F16.gguf
PROMPT_TEXT='<|im_start|>user
What is the capital of France? Answer in one short sentence.<|im_end|>
<|im_start|>assistant
'
PROMPT_IDS="130072,8448,220,2928,357,285,4894,304,6918,52,10893,310,678,2871,9622,35,130073,220,130072,130071,220"
NGEN=192

mkdir -p results
for run in 1 2; do
  for m in q4km f16; do
    case $m in
      q4km) MODEL=$Q4; TAG=mcpm5_q4km ;;
      f16)  MODEL=$F16; TAG=mcpm5_f16 ;;
    esac
    # speed pass
    python3 run_h2h.py $LLAMA  $MODEL llamacpp "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_speed_run${run}.json --nomen
    python3 run_h2h.py $STRATUM $MODEL stratum  "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_speed_run${run}.json --nomen
    # memory pass
    python3 run_h2h.py $LLAMA  $MODEL llamacpp "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_mem_run${run}.json
    python3 run_h2h.py $STRATUM $MODEL stratum  "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_mem_run${run}.json 
  done
done
echo "=== minicpm cases done ==="