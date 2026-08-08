#!/usr/bin/env python3
"""
Persistent tokenizer server. Reads commands from stdin, writes responses
to stdout. Used by stratum_p10 for streaming output.

Protocol (each request and response is one line, newline-terminated):
  -> "ENCODE <text>"        <- "<id> <id> ..." or "ERR <msg>"
  -> "DECODE <id> <id> ..." <- "<text>" (no newlines in text replaced with U+FFFC)
  -> "DECODE_NL <id> <id>"  <- same as DECODE but allows raw newlines, terminated with NUL byte
  -> "QUIT"                 <- "BYE"

Newlines inside decoded text would break the line protocol. We use NUL
termination for the streaming-decode case via DECODE_NL.
"""
from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent / "model"


def main():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=False)
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    out = sys.stdout.buffer

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        if line == "QUIT":
            sys.stdout.write("BYE\n"); sys.stdout.flush()
            break
        if line.startswith("ENCODE "):
            text = line[len("ENCODE "):]
            ids = tok.encode(text, add_special_tokens=False)
            sys.stdout.write(" ".join(str(i) for i in ids))
            sys.stdout.write("\n"); sys.stdout.flush()
        elif line.startswith("DECODE_NL "):
            ids = [int(x) for x in line[len("DECODE_NL "):].split()]
            s = tok.decode(ids, skip_special_tokens=False)
            out.write(s.encode("utf-8"))
            out.write(b"\x00")
            out.flush()
        elif line.startswith("DECODE "):
            ids = [int(x) for x in line[len("DECODE "):].split()]
            s = tok.decode(ids, skip_special_tokens=False)
            s = s.replace("\n", "\ufffc").replace("\r", "")
            sys.stdout.write(s)
            sys.stdout.write("\n"); sys.stdout.flush()
        else:
            sys.stdout.write("ERR unknown command\n"); sys.stdout.flush()


if __name__ == "__main__":
    main()
