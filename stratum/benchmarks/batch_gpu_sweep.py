#!/usr/bin/env python3
"""Sweep native full-GPU multiseq decode throughput.

This benchmark is intentionally narrow: it exercises the throughput axis of
the native runtime, B independent decode streams in one GPU sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGG_RE = re.compile(r"aggregate\s+([0-9.]+)\s+tok/s\s+\(per-stream\s+([0-9.]+)\s+tok/s\)")
TOK_RE = re.compile(r"ms step\s+([0-9]+)\s+stream0 argmax=([0-9]+)")
FIRST_RE = re.compile(r"stream0 first tok=([0-9]+)")


def run_one(model: Path, n_gen: int, b: int, repeats: int, extra_env: dict[str, str]) -> dict:
    prompt = ["1", "450", "7483", "310", "3444", "338"]
    best = None
    runs = []
    for _ in range(repeats):
        env = os.environ.copy()
        env.update(extra_env)
        env.update({
            "STRATUM_GPU": "1",
            "STRATUM_GPU_BATCH_FULL": "1",
            "STRATUM_METALLIB": "stratum_q4k.metallib",
            "STRATUM_MULTISEQ": str(b),
        })
        cmd = ["./stratum", str(model), str(n_gen), *prompt]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=ROOT / "native", env=env,
                           text=True, capture_output=True)
        wall = time.perf_counter() - t0
        out = p.stdout + p.stderr
        if p.returncode != 0:
            raise RuntimeError(out[-4000:])
        m = AGG_RE.search(out)
        if not m:
            raise RuntimeError("aggregate line missing:\n" + out[-2000:])
        first = FIRST_RE.search(out)
        steps = [int(x.group(2)) for x in TOK_RE.finditer(out)]
        rec = {
            "aggregate_tok_s": float(m.group(1)),
            "per_stream_tok_s": float(m.group(2)),
            "wall_s": wall,
            "first_token": int(first.group(1)) if first else None,
            "first_steps": steps[:8],
        }
        runs.append(rec)
        if best is None or rec["aggregate_tok_s"] > best["aggregate_tok_s"]:
            best = rec
    return {"B": b, "best": best, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a GGUF model (no default)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "batch_gpu_sweep.json"))
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--b", default="1,2,4,8,16,24,32")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "serial", "parallel", "simdb", "top1_tiled", "fused_argmax"])
    ap.add_argument("--env", action="append", default=[],
                    help="Extra environment override, KEY=VALUE. May be repeated.")
    args = ap.parse_args()

    extra_env: dict[str, str] = {}
    if args.mode == "serial":
        extra_env["STRATUM_GPU_BATCH_SERIAL"] = "1"
    elif args.mode == "parallel":
        extra_env["STRATUM_GPU_BATCH_PAR"] = "1"
    elif args.mode == "simdb":
        extra_env["STRATUM_GPU_BATCH_SIMDB"] = "1"
    elif args.mode == "top1_tiled":
        extra_env["STRATUM_GPU_TOP1_TILED"] = "1"
    elif args.mode == "fused_argmax":
        extra_env["STRATUM_GPU_FUSED_ARGMAX"] = "1"
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        extra_env[k] = v

    bs = [int(x) for x in args.b.split(",") if x.strip()]
    model = Path(args.model).resolve()
    results = []
    for b in bs:
        rec = run_one(model, args.n_gen, b, args.repeats, extra_env)
        results.append(rec)
        best = rec["best"]
        print(f"B={b:<2} aggregate={best['aggregate_tok_s']:.1f} tok/s "
              f"per_stream={best['per_stream_tok_s']:.1f}")

    out = {
        "model": str(model),
        "n_gen": args.n_gen,
        "repeats": args.repeats,
        "mode": args.mode,
        "extra_env": dict(sorted(extra_env.items())),
        "golden_first_steps": [29889, 13, 13, 29906, 29889, 350, 29889, 29907],
        "results": results,
        "ts": time.time(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
