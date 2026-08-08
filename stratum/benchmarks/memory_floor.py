#!/usr/bin/env python3
"""
Phase 4-eng — Memory floor benchmark harness.

Measures peak RSS across all current Stratum configurations on the same
prompt. RSS is sampled in a background thread every 50 ms during generation,
not just at the end (ru_maxrss is process-lifetime peak which is meaningless
for cross-config comparisons in the same process).

Each config is run in a *fresh subprocess* so PyTorch allocator state and
mmap'd pages from previous runs don't contaminate measurements.
"""

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
]


def run(label, worker_args, max_new_tokens):
    cmd = [sys.executable, str(Path(__file__).parent / "memory_floor_worker.py"),
           "--max-new-tokens", str(max_new_tokens), *worker_args]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if p.returncode != 0:
        print(f"!! {label} failed (exit={p.returncode})")
        print(p.stdout[-400:])
        print(p.stderr[-400:])
        return None
    last = [l for l in p.stdout.strip().split("\n") if l.startswith("{")][-1]
    result = json.loads(last)
    result["wall_total"] = time.perf_counter() - t0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default="docs/memory_floor.json")
    ap.add_argument("--only", default=None,
                    help="run only the config whose label contains this string")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"== Phase 4-eng memory floor — {args.max_new_tokens} new tokens ==\n")
    print(f"  {'config':<28s}  {'load(s)':>8s}  {'gen(s)':>8s}  "
          f"{'RSS load':>10s}  {'RSS peak':>10s}  {'tok/s':>6s}  output")

    results = {}
    for label, worker_args in CONFIGS:
        if args.only and args.only not in label:
            continue
        r = run(label, worker_args, args.max_new_tokens)
        if r is None:
            continue
        results[label] = r
        print(f"  {label:<28s}  {r['load_s']:>7.2f}s  {r['gen_s']:>7.2f}s  "
              f"{r['rss_after_load_mb']:>8.1f} MB  {r['rss_peak_mb']:>8.1f} MB  "
              f"{r['tok_per_s']:>5.2f}  {r['text']!r}")

    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
