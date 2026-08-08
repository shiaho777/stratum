#!/usr/bin/env python3
"""Per-config worker for memory_floor.py. Runs ONE inference + reports JSON."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def rss_now_mb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out.strip()) / 1024


class RSSWatcher:
    """Samples RSS every `interval` seconds in a thread, records the peak."""
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                v = rss_now_mb()
                if v > self.peak:
                    self.peak = v
                self._stop.wait(self.interval)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["transformers", "stratum-bf16", "stratum-int4"])
    ap.add_argument("--quant-dir", default="../model-int4-awq")
    ap.add_argument("--cache", type=int, default=2)
    ap.add_argument("--prompt", default="Hello, my name is")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    model_dir = (repo_root.parent / "model").resolve()

    rss0 = rss_now_mb()

    import torch
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    if args.mode == "transformers":
        from transformers import AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_dir)
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        model.eval()
        load_s = time.perf_counter() - t0
    else:
        from stratum.adapters.qwen3_5 import load_qwen3_5_text
        is_quant = args.mode == "stratum-int4"
        if is_quant:
            md = (repo_root.parent / Path(args.quant_dir).name).resolve()
        else:
            md = model_dir
        tok = AutoTokenizer.from_pretrained(md)
        t0 = time.perf_counter()
        model, provider = load_qwen3_5_text(
            md, device="cpu", dtype=torch.bfloat16,
            max_layers_in_ram=args.cache, quantized=is_quant,
        )
        load_s = time.perf_counter() - t0

    rss_after_load = rss_now_mb()

    inputs = tok(args.prompt, return_tensors="pt")

    with RSSWatcher() as w:
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True,
            )
        gen_s = time.perf_counter() - t0

    new_ids = out[0, inputs.input_ids.shape[-1]:].tolist()
    text = tok.decode(out[0], skip_special_tokens=False)

    result = {
        "rss_at_start_mb":   rss0,
        "rss_after_load_mb": rss_after_load,
        "rss_peak_mb":       max(w.peak, rss_after_load),
        "load_s":            load_s,
        "gen_s":             gen_s,
        "tok_per_s":         len(new_ids) / gen_s if gen_s > 0 else 0,
        "new_ids":           new_ids,
        "text":              text,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
