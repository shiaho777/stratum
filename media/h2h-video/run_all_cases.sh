#!/bin/zsh
# run_all_cases.sh — measured head-to-head: stratum vs llama.cpp.
# Each (model x engine) case runs twice; run 2 (hot page cache) is kept.
# Results -> results/<model>_<engine>_run{1,2}.json
set -u
cd "$(dirname "$0")"

STRATUM=/Users/shiaho/Desktop/Qwen3.5-0.8B-hf/stratum/native/stratum
LLAMA=/tmp/llama.cpp/build/bin/llama-simple
Q4=/Users/shiaho/Desktop/0-/Qwen3-0.6B/Qwen3-0.6B.q4km.llamacpp.gguf
F16=/Users/shiaho/Desktop/0-/Qwen3-0.6B/Qwen3-0.6B.f16.qwen3.gguf
PROMPT_TEXT='<|im_start|>user
What is the capital of France? Answer in one short sentence.<|im_end|>
<|im_start|>assistant
'
PROMPT_IDS="151644,872,198,3838,374,279,6722,315,9625,30,21806,304,825,2805,11652,13,151645,198,151644,77091,198"
NGEN=192

mkdir -p results
for run in 1 2; do
  for m in q4km f16; do
    case $m in
      q4km) MODEL=$Q4; EXTRA=--sdot0; TAG=q4km ;;
      f16)  MODEL=$F16; EXTRA="";    TAG=f16 ;;
    esac
    # speed pass (no vmmap — sampling suspends the target and skews timing)
    python3 run_h2h.py $LLAMA  $MODEL llamacpp "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_speed_run${run}.json --nomen
    python3 run_h2h.py $STRATUM $MODEL stratum  "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_speed_run${run}.json $EXTRA --nomen
    python3 run_h2h.py $STRATUM $MODEL stratum  "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_hotfast_speed_run${run}.json $EXTRA --hotfast --nomen
    # memory pass (vmmap sampling; wall time here is not comparable)
    python3 run_h2h.py $LLAMA  $MODEL llamacpp "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_llamacpp_mem_run${run}.json
    python3 run_h2h.py $STRATUM $MODEL stratum  "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_mem_run${run}.json $EXTRA
    python3 run_h2h.py $STRATUM $MODEL stratum  "$PROMPT_TEXT" "$PROMPT_IDS" $NGEN results/${TAG}_stratum_hotfast_mem_run${run}.json $EXTRA --hotfast
  done
done
echo "=== all cases done ==="