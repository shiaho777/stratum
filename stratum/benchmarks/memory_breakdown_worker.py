#!/usr/bin/env python3
"""Trace where RSS goes during generation by sampling at named breakpoints."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from stratum.adapters.qwen3_5 import load_qwen3_5_text


def rss():
    return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())]).strip()) / 1024


def main():
    repo_root = Path(__file__).resolve().parent.parent
    md = (repo_root.parent / "model-int4-awq").resolve()

    samples = []
    def mark(label):
        samples.append((label, rss()))

    mark("python_started")

    tok = AutoTokenizer.from_pretrained(md)
    mark("tokenizer_loaded")

    model, provider = load_qwen3_5_text(
        md, device="cpu", dtype=torch.bfloat16,
        max_layers_in_ram=2, quantized=True,
    )
    mark("model_loaded")

    inputs = tok("Hello, my name is", return_tensors="pt")
    mark("inputs_tokenized")

    with torch.no_grad():
        ids = inputs.input_ids
        emb = model.model.embed_tokens(ids)
        mark("embedding_materialized_lazy")
        del emb
        import gc; gc.collect()
        mark("after_gc")

        out = model.generate(
            **inputs, max_new_tokens=8,
            do_sample=False, num_beams=1, use_cache=True,
        )
    mark("generation_done")

    print(f"\n{'stage':<35s} {'RSS (MB)':>10s} {'delta':>10s}")
    prev = samples[0][1]
    for label, val in samples:
        d = val - prev
        sign = "+" if d >= 0 else ""
        print(f"  {label:<33s}  {val:>8.1f}  {sign}{d:>8.1f}")
        prev = val

    total_pin = 0
    total_stream = 0
    for n, t in provider._pin_cache.items():
        total_pin += t.numel() * t.element_size()
    for lid, bucket in provider._stream_cache.items():
        for t in bucket.values():
            total_stream += t.numel() * t.element_size()
    print(f"\nProvider cache: pin={total_pin/1e6:.1f} MB ({len(provider._pin_cache)} tensors), "
          f"stream={total_stream/1e6:.1f} MB ({sum(len(b) for b in provider._stream_cache.values())} tensors)")


if __name__ == "__main__":
    main()
