#!/usr/bin/env python3
"""Decode TinyLlama token IDs to verify stratum_v2 output."""
import sys
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

ids = [int(x) for x in sys.argv[1:]]
print(f'IDs: {ids}')
print()
for tid in ids:
    s = t.decode([tid])
    print(f'  {tid}: {repr(s)}')
print()
print('Joined:', repr(t.decode(ids)))
