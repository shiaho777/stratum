#!/usr/bin/env python3
"""Stable RSS measurement: run each config N times, report median + min."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    ("transformers-baseline", ["--mode", "transformers"]),
    ("bf16-cache2",            ["--mode", "stratum-bf16", "--cache", "2"]),
    ("int4-vanilla-cache2",    ["--mode", "stratum-int4", "--cache", "2", "--quant-dir", "../model-int4"]),
    ("int4-awq-cache2",        ["--mode", "stratum-int4", "--cache", "2", "--quant-dir", "../model-int4-awq"]),
    ("int4-awq-cache1",        ["--mode", "stratum-int4", "--cache", "1", "--quant-dir", "../model-int4-awq"]),
]


def run_once(label, worker_args, max_new_tokens):
    cmd = [sys.executable, str(Path(__file__).parent / "memory_floor_worker.py"),
           "--max-new-tokens", str(max_new_tokens), *worker_args]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if p.returncode != 0:
        return None
    last = [l for l in p.stdout.strip().split("\n") if l.startswith("{")]
    if not last: return None
    return json.loads(last[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    print(f"== Stable memory floor — {args.max_new_tokens} new tokens, "
          f"{args.repeats} repeats per config ==\n")
    print(f"  {'config':<24s}  {'rss min':>9s}  {'rss median':>11s}  "
          f"{'rss max':>9s}  {'tok/s':>6s}  output")

    rows = []
    for label, worker_args in CONFIGS:
        if args.only and args.only not in label:
            continue
        peaks = []
        toks = []
        text = ""
        for _ in range(args.repeats):
            r = run_once(label, worker_args, args.max_new_tokens)
            if r is None: continue
            peaks.append(r["rss_peak_mb"])
            toks.append(r["tok_per_s"])
            text = r["text"]
        if not peaks: continue
        peaks.sort()
        rss_min = peaks[0]; rss_max = peaks[-1]; rss_med = peaks[len(peaks)//2]
        tps = sum(toks)/len(toks)
        rows.append((label, rss_min, rss_med, rss_max, tps, text))
        print(f"  {label:<24s}  {rss_min:>7.0f} MB  {rss_med:>9.0f} MB  "
              f"{rss_max:>7.0f} MB  {tps:>5.2f}  {text!r}")


if __name__ == "__main__":
    main()
