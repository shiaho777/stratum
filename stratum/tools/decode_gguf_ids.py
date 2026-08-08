#!/usr/bin/env python3
"""Decode token IDs using a GGUF model's embedded tokenizer.

usage: decode_gguf_ids.py <model.gguf> <id1> <id2> ...
"""
import sys
import gguf

if len(sys.argv) < 3:
    print("usage: decode_gguf_ids.py <model.gguf> <id1> <id2> ...", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
ids = [int(s) for s in sys.argv[2:]]

r = gguf.GGUFReader(path)
tf = r.fields.get('tokenizer.ggml.tokens')
if tf is None:
    print("no tokenizer.ggml.tokens field", file=sys.stderr)
    sys.exit(1)
parts = tf.parts
data = tf.data

pieces = []
for tid in ids:
    idx = data[tid]
    bs = bytes(parts[idx]).decode('utf-8', errors='replace')
    bs2 = bs.replace('\u0120', ' ').replace('\u010a', '\n').replace('Ġ', ' ').replace('Ċ', '\n')
    pieces.append(bs2)
    print(f"  {tid}: {bs!r} -> {bs2!r}")
print()
print(repr(''.join(pieces)))
