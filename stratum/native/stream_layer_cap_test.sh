#!/bin/sh
# Streaming pread keeps a fixed 128-slot layer index. A model with more
# main layers must fail with that message instead of writing off the array.
# A 4-layer model must not trip the cap, and must still decode.
set -eu
cd "$(dirname "$0")"

if [ ! -x ./stratum ]; then
    echo "stream_layer_cap_test: ./stratum is not built" >&2
    exit 1
fi

DEEP="${TMPDIR:-/tmp}/stratum_stream_cap.gguf"
SMALL="${TMPDIR:-/tmp}/stratum_stream_small.gguf"

STRATUM_TINY_N_LAYERS=129 python3 ../tools/make_tiny_model.py --arch qwen35 --out "$DEEP" >/dev/null
set +e
DEEP_LOG=$(STRATUM_NO_GPU=1 STRATUM_STREAM=1 ./stratum "$DEEP" 1 1 2>&1)
DEEP_RC=$?
set -e
printf '%s\n' "$DEEP_LOG"
printf '%s\n' "$DEEP_LOG" | grep -q "streaming layer table holds 1..128"
if [ "$DEEP_RC" -eq 0 ]; then
    echo "stream_layer_cap_test: 129-layer model exited 0" >&2
    exit 1
fi

python3 ../tools/make_tiny_model.py --arch qwen35 --out "$SMALL" >/dev/null
set +e
SMALL_LOG=$(STRATUM_NO_GPU=1 STRATUM_STREAM=1 ./stratum "$SMALL" 1 1 2>&1)
SMALL_RC=$?
set -e
printf '%s\n' "$SMALL_LOG"
if printf '%s\n' "$SMALL_LOG" | grep -q "streaming layer table holds 1..128"; then
    echo "stream_layer_cap_test: 4-layer model tripped the layer cap" >&2
    exit 1
fi
if [ "$SMALL_RC" -ne 0 ]; then
    echo "stream_layer_cap_test: 4-layer streaming decode failed" >&2
    exit 1
fi

echo "stream_layer_cap_test: ok"
