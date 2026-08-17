#!/bin/sh
# fetch_memx.sh — fetch the optional MemX runtime dependency.
#
# MemX (github.com/shiaho777/memx, MIT) is an optional compressed-memory
# runtime that backs the engine's staging buffers and KV/SSM state. It is a
# DOWNLOADED dependency, never vendored: this script clones (or pulls) the
# upstream repo into MEMX_HOME, so upstream updates flow in via `make deps`
# without replicating its code in this repository. The engine builds and
# runs without it (`make USE_MEMX=0`); this script only runs when a
# MemX-enabled build needs the source.
#
# Usage:  ./fetch_memx.sh [MEMX_HOME]     (default: <this dir>/memx)
# Env:    MEMX_REPO — upstream URL (default https://github.com/shiaho777/memx.git)

set -eu

MEMX_REPO="${MEMX_REPO:-https://github.com/shiaho777/memx.git}"
MEMX_HOME="${1:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/memx}"

if [ -d "$MEMX_HOME/.git" ]; then
    echo "memx: updating dependency in $MEMX_HOME"
    if ! git -C "$MEMX_HOME" pull --ff-only --quiet; then
        echo "memx: pull failed (shallow history moved?), re-cloning" >&2
        rm -rf "$MEMX_HOME"
        git clone --depth 1 "$MEMX_REPO" "$MEMX_HOME"
    fi
else
    echo "memx: cloning dependency into $MEMX_HOME"
    mkdir -p "$(dirname "$MEMX_HOME")"
    git clone --depth 1 "$MEMX_REPO" "$MEMX_HOME"
fi

echo "memx: at $(git -C "$MEMX_HOME" rev-parse --short HEAD)"
