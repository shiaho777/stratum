#!/usr/bin/env python3
"""Locate the peak RSS during forward by hooking every decoder layer."""

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
    md = (repo_root.parent / Path(os.environ.get("STRATUM_QUANT_DIR", "model-int4-awq")).name).resolve()
    cache = int(os.environ.get("STRATUM_CACHE", "2"))

    tok = AutoTokenizer.from_pretrained(md)
    model, provider = load_qwen3_5_text(
        md, device="cpu", dtype=torch.bfloat16,
        max_layers_in_ram=cache, quantized=True,
    )
    print(f"after load: {rss():.1f} MB  (cache={cache})")

    samples = []
    def make_pre(lid, kind):
        def hook(module, args):
            samples.append((kind, lid, rss()))
        return hook
    def make_post(lid, kind):
        def hook(module, args, output):
            samples.append((kind, lid, rss()))
        return hook

    import re
    pat = re.compile(r"^model\.layers\.(\d+)$")
    handles = []
    for name, mod in model.named_modules():
        m = pat.match(name)
        if m:
            lid = int(m.group(1))
            handles.append(mod.register_forward_pre_hook(make_pre(lid, "layer_pre")))
            handles.append(mod.register_forward_hook(make_post(lid, "layer_post")))

    if hasattr(model, "lm_head"):
        handles.append(model.lm_head.register_forward_pre_hook(make_pre(-1, "lm_head_pre")))
        handles.append(model.lm_head.register_forward_hook(make_post(-1, "lm_head_post")))

    inputs = tok("Hello, my name is", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    for h in handles: h.remove()

    print(f"\n  {'kind':<14s}  {'lid':>4s}  {'RSS':>8s}  {'delta':>8s}")
    prev = None
    peak_kind, peak_lid, peak_rss = None, None, 0
    for kind, lid, val in samples:
        d = val - prev if prev is not None else 0
        print(f"  {kind:<14s}  {lid:>4d}  {val:>6.1f}  {d:>+7.1f}")
        if val > peak_rss:
            peak_rss = val
            peak_kind = kind
            peak_lid = lid
        prev = val
    print(f"\nPEAK during single forward: {peak_kind} layer={peak_lid} RSS={peak_rss:.1f} MB")
    print(f"after forward: {rss():.1f} MB")


if __name__ == "__main__":
    main()
