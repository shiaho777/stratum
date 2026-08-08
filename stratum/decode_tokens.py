#!/usr/bin/env python3
"""Decode token ids from a GGUF's embedded vocab."""
import sys
from gguf import GGUFReader

if len(sys.argv) < 3:
    print("usage: decode_tokens.py <model.gguf> <id> [<id>...]", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]

r = GGUFReader(path)
toks_field = r.fields['tokenizer.ggml.tokens']
toks = toks_field.contents()

print("--- per-id mapping ---")
for i, tid in enumerate(ids):
    t = toks[tid]
    if isinstance(t, bytes):
        s = t.decode('utf-8', errors='replace')
    else:
        s = str(t)
    print("  step %2d  id=%6d  tok=%r" % (i, tid, s))

print("--- concatenated (SentencePiece U+2581 = space) ---")
parts = []
for tid in ids:
    t = toks[tid]
    if isinstance(t, bytes):
        s = t.decode('utf-8', errors='replace')
    else:
        s = str(t)
    s = s.replace('\u2581', ' ')
    parts.append(s)
print(repr(''.join(parts)))
