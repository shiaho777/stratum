#!/usr/bin/env python3
"""
Stratum — comprehensive benchmark.

Runs the same prompt + same decoding through every Stratum mode + the
transformers baseline, recording RSS, latency, output ids, and a
disk-footprint figure.

Each scenario runs in a *fresh subprocess* so RSS measurements aren't
contaminated by allocator state from the previous run.

Output: benchmarks/results.json (consumed by README and the blog post).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RUNNER = ROOT / "benchmarks" / "_run_one.py"
PROMPT = "Hello, my name is"
N_NEW = 8


SCENARIOS = [
    {
        "name": "transformers-baseline",
        "kind": "transformers",
        "model_dir": "../model",
        "label": "transformers, full bf16 load",
        "disk_bytes": None,
    },
    {
        "name": "stratum-bf16",
        "kind": "stratum",
        "model_dir": "../model",
        "max_layers": None,
        "quantized": False,
        "label": "Stratum, bf16, no streaming",
    },
    {
        "name": "stratum-bf16-stream-2",
        "kind": "stratum",
        "model_dir": "../model",
        "max_layers": 2,
        "quantized": False,
        "label": "Stratum, bf16, stream cache=2",
    },
    {
        "name": "stratum-int4-stream-2",
        "kind": "stratum",
        "model_dir": "../model-int4",
        "max_layers": 2,
        "quantized": True,
        "label": "Stratum, INT4 (vanilla), stream cache=2",
    },
    {
        "name": "stratum-int4-awq-stream-2",
        "kind": "stratum",
        "model_dir": "../model-int4-awq",
        "max_layers": 2,
        "quantized": True,
        "label": "Stratum, INT4+AWQ, stream cache=2",
    },
    {
        "name": "stratum-int4-awq-remote",
        "kind": "stratum-remote",
        "model_dir": "../model-int4-awq",
        "max_layers": 2,
        "quantized": True,
        "rtt_ms": 0,
        "port": 19101,
        "cache_root": "/tmp/stratum-bench-remote",
        "label": "Stratum, INT4+AWQ, remote (local server, 0ms RTT)",
    },
    {
        "name": "stratum-int4-awq-remote-50ms",
        "kind": "stratum-remote",
        "model_dir": "../model-int4-awq",
        "max_layers": 2,
        "quantized": True,
        "rtt_ms": 50,
        "port": 19102,
        "cache_root": "/tmp/stratum-bench-remote-50",
        "label": "Stratum, INT4+AWQ, remote (50ms RTT, simulated WAN)",
    },
]


def run_scenario(s: dict) -> dict:
    print(f"\n=== {s['label']} ===")
    cmd = [sys.executable, str(SCRIPT_RUNNER),
           "--name",      s["name"],
           "--kind",      s["kind"],
           "--model-dir", s["model_dir"],
           "--prompt",    PROMPT,
           "--max-new",   str(N_NEW)]
    if "max_layers" in s and s["max_layers"] is not None:
        cmd += ["--max-layers", str(s["max_layers"])]
    if s.get("quantized"):
        cmd += ["--quantized"]
    if s["kind"] == "stratum-remote":
        cmd += ["--rtt-ms",     str(s.get("rtt_ms", 0)),
                "--port",       str(s.get("port", 19100)),
                "--cache-root", s.get("cache_root", "/tmp/stratum-bench-remote")]

    env = os.environ.copy()
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        print(out.stdout[-2000:])
        print(out.stderr[-2000:])
        return {**s, "error": out.stderr[-500:], "ok": False}

    try:
        result_line = next(
            ln for ln in reversed(out.stdout.splitlines())
            if ln.startswith("RESULT_JSON ")
        )
        rec = json.loads(result_line[len("RESULT_JSON "):])
    except Exception as e:
        print(f"  parse failure: {e}")
        print(out.stdout[-1000:])
        return {**s, "error": "parse_failure", "ok": False}

    print(f"  load wall    : {rec['load_s']:.2f}s")
    print(f"  RSS post-load: {rec['rss_post_load_mb']:.1f} MB")
    print(f"  gen wall     : {rec['gen_s']:.2f}s ({rec['tok_per_s']:.2f} tok/s)")
    print(f"  RSS peak     : {rec['rss_peak_mb']:.1f} MB")
    print(f"  ids          : {rec['new_ids']}")
    return {**s, **rec, "ok": True, "subprocess_wall_s": elapsed}


def disk_footprint(model_dir_rel: str) -> int:
    p = (ROOT / model_dir_rel).resolve()
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results.json"))
    ap.add_argument("--filter", default=None,
                    help="substring filter on scenario name")
    args = ap.parse_args()

    results = []
    for s in SCENARIOS:
        if args.filter and args.filter not in s["name"]:
            continue
        s = {**s, "disk_bytes": disk_footprint(s["model_dir"])}
        results.append(run_scenario(s))

    out = {
        "prompt": PROMPT,
        "max_new_tokens": N_NEW,
        "scenarios": results,
        "ts": time.time(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")

    print("\n=== Summary ===")
    fmt = ("{:<46} {:>10} {:>10} {:>10} {:>9} {:>9}")
    print(fmt.format("scenario", "disk", "RSS load", "RSS peak", "load", "tok/s"))
    print(fmt.format("-"*46, "-"*10, "-"*10, "-"*10, "-"*9, "-"*9))
    for r in results:
        if not r.get("ok"):
            print(fmt.format(r["label"][:46], "—", "—", "—", "—", "FAIL"))
            continue
        disk_str = f"{r.get('disk_bytes',0)/1e6:7.0f} MB" if r.get("disk_bytes") else "   —   "
        print(fmt.format(
            r["label"][:46],
            disk_str,
            f"{r['rss_post_load_mb']:7.1f} MB",
            f"{r['rss_peak_mb']:7.1f} MB",
            f"{r['load_s']:6.2f}s",
            f"{r['tok_per_s']:6.2f}",
        ))


if __name__ == "__main__":
    main()
