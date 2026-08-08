#!/usr/bin/env python3
"""Encode a prompt into TinyLlama token IDs for use with stratum_v2."""
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
prompt = ' '.join(sys.argv[1:])
ids = t.encode(prompt, add_special_tokens=True)
print(' '.join(str(i) for i in ids))
