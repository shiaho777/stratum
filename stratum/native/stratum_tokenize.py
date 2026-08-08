#!/usr/bin/env python3
"""
Stratum tokenizer helper.

Two modes invoked by the C runtime:
  - encode <text...>     prints space-separated token ids
  - decode <id> <id> ... prints the decoded string (no trailing newline)

We use the local model directory's tokenizer.json so this never depends on
network access.
"""
from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent / "model"


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: stratum_tokenize.py encode <text> | decode <id> ...\n")
        sys.exit(2)

    mode = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=False)

    if mode == "encode":
        text = " ".join(sys.argv[2:])
        ids = tok.encode(text, add_special_tokens=False)
        sys.stdout.write(" ".join(str(i) for i in ids))
        sys.stdout.write("\n")
    elif mode == "decode":
        ids = [int(x) for x in sys.argv[2:]]
        out = tok.decode(ids, skip_special_tokens=False)
        sys.stdout.write(out)
    else:
        sys.stderr.write(f"unknown mode: {mode}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
