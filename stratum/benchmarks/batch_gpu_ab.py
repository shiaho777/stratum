#!/usr/bin/env python3
"""Interleaved AB benchmark for native full-GPU multiseq decode."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGG_RE = re.compile(r"aggregate\s+([0-9.]+)\s+tok/s\s+\(per-stream\s+([0-9.]+)\s+tok/s\)")
TOK_RE = re.compile(r"ms step\s+([0-9]+)\s+stream0 argmax=([0-9]+)")
FIRST_RE = re.compile(r"stream0 first tok=([0-9]+)")
GOLDEN = [29889, 13, 13, 29906, 29889, 350, 29889, 29907]


def parse_env(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"env override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def parse_variant(spec: str) -> dict:
    if ":" in spec:
        name, raw = spec.split(":", 1)
    else:
        name, raw = spec, ""
    env = parse_env([x for x in raw.split(",") if x])
    return {"name": name, "env": env}


def run_one(model: Path, n_gen: int, b: int, variant: dict) -> dict:
    prompt = ["1", "450", "7483", "310", "3444", "338"]
    env = os.environ.copy()
    env.update(variant["env"])
    env.update({
        "STRATUM_GPU": "1",
        "STRATUM_GPU_BATCH_FULL": "1",
        "STRATUM_METALLIB": "stratum_q4k.metallib",
        "STRATUM_MULTISEQ": str(b),
    })
    cmd = ["./stratum", str(model), str(n_gen), *prompt]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT / "native", env=env,
                          text=True, capture_output=True)
    wall = time.perf_counter() - t0
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(out[-4000:])
    agg = AGG_RE.search(out)
    if not agg:
        raise RuntimeError("aggregate line missing:\n" + out[-2000:])
    first = FIRST_RE.search(out)
    steps = [int(m.group(2)) for m in TOK_RE.finditer(out)]
    prefix_ok = steps[:len(GOLDEN)] == GOLDEN
    if not prefix_ok:
        raise RuntimeError(
            f"golden prefix mismatch for {variant['name']} B={b}: {steps[:len(GOLDEN)]}"
        )
    return {
        "variant": variant["name"],
        "B": b,
        "aggregate_tok_s": float(agg.group(1)),
        "per_stream_tok_s": float(agg.group(2)),
        "wall_s": wall,
        "first_token": int(first.group(1)) if first else None,
        "first_steps": steps[:len(GOLDEN)],
        "prefix_ok": prefix_ok,
    }


def summarize(rows: list[dict]) -> dict:
    vals = [r["aggregate_tok_s"] for r in rows]
    median = statistics.median(vals)
    low_outliers = [
        r for r in rows
        if median > 0.0 and r["aggregate_tok_s"] < 0.5 * median
    ]
    return {
        "n": len(vals),
        "median": median,
        "mean": statistics.fmean(vals),
        "best": max(vals),
        "worst": min(vals),
        "low_outliers": low_outliers,
        "runs": rows,
    }


def paired_summary(raw_runs: list[dict], b: int, baseline: str, variant: str,
                   skip: set[tuple[int, str]] | None = None) -> dict:
    by_round: dict[int, dict[str, dict]] = {}
    for row in raw_runs:
        if row["B"] != b:
            continue
        by_round.setdefault(row["round"], {})[row["variant"]] = row

    pairs = []
    ratios = []
    wins = 0
    for round_i in sorted(by_round):
        pair = by_round[round_i]
        if baseline not in pair or variant not in pair:
            continue
        if skip and ((round_i, baseline) in skip or (round_i, variant) in skip):
            continue
        base = pair[baseline]["aggregate_tok_s"]
        other = pair[variant]["aggregate_tok_s"]
        ratio = other / base - 1.0
        ratios.append(ratio)
        if other > base:
            wins += 1
        pairs.append({
            "round": round_i,
            baseline: base,
            variant: other,
            "ratio": ratio,
        })

    if not ratios:
        return {"n": 0, "wins": 0, "median": None, "mean": None, "pairs": []}
    return {
        "n": len(ratios),
        "wins": wins,
        "median": statistics.median(ratios),
        "mean": statistics.fmean(ratios),
        "best": max(ratios),
        "worst": min(ratios),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a GGUF model (no default)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-gen", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--b", default="16,24,32")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle B order and variant order per round with --seed.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="Seconds to sleep after each run.")
    parser.add_argument("--variant", action="append", required=True,
                        help="NAME or NAME:KEY=VALUE,KEY=VALUE. Repeat twice or more.")
    args = parser.parse_args()

    variants = [parse_variant(v) for v in args.variant]
    bs = [int(x) for x in args.b.split(",") if x.strip()]
    model = Path(args.model).resolve()
    raw_runs: list[dict] = []
    rng = random.Random(args.seed)

    for r in range(args.rounds):
        b_order = list(bs)
        if args.shuffle:
            rng.shuffle(b_order)
        for b in b_order:
            order = list(variants if r % 2 == 0 else reversed(variants))
            if args.shuffle:
                rng.shuffle(order)
            for variant in order:
                rec = run_one(model, args.n_gen, b, variant)
                rec["round"] = r
                raw_runs.append(rec)
                print(f"r={r} B={b:<2} {variant['name']:<12} "
                      f"{rec['aggregate_tok_s']:.1f} tok/s", flush=True)
                if args.cooldown > 0:
                    time.sleep(args.cooldown)

    by_b_variant: dict[str, dict[str, dict]] = {}
    for b in bs:
        key = str(b)
        by_b_variant[key] = {}
        for variant in variants:
            rows = [r for r in raw_runs if r["B"] == b and r["variant"] == variant["name"]]
            by_b_variant[key][variant["name"]] = summarize(rows)

    baseline = variants[0]["name"]
    comparisons: dict[str, dict[str, float]] = {}
    paired_comparisons: dict[str, dict[str, dict]] = {}
    filtered_paired_comparisons: dict[str, dict[str, dict]] = {}
    for b in bs:
        bkey = str(b)
        base_med = by_b_variant[bkey][baseline]["median"]
        comparisons[bkey] = {}
        paired_comparisons[bkey] = {}
        filtered_paired_comparisons[bkey] = {}
        low_outlier_keys = {
            (row["round"], name)
            for name, summary in by_b_variant[bkey].items()
            for row in summary["low_outliers"]
        }
        for variant in variants[1:]:
            med = by_b_variant[bkey][variant["name"]]["median"]
            comparisons[bkey][variant["name"]] = med / base_med - 1.0
            paired = paired_summary(raw_runs, b, baseline, variant["name"])
            paired_comparisons[bkey][variant["name"]] = paired
            filtered = paired_summary(raw_runs, b, baseline, variant["name"], low_outlier_keys)
            filtered_paired_comparisons[bkey][variant["name"]] = filtered
            print(f"B={b:<2} {variant['name']} vs {baseline}: "
                  f"{med:.1f} / {base_med:.1f} = {100.0 * (med / base_med - 1.0):+.1f}%",
                  flush=True)
            if paired["n"]:
                print(f"B={b:<2} {variant['name']} paired vs {baseline}: "
                      f"median {100.0 * paired['median']:+.1f}% "
                      f"wins {paired['wins']}/{paired['n']}",
                      flush=True)
            if filtered["n"] != paired["n"] and filtered["n"]:
                print(f"B={b:<2} {variant['name']} filtered paired vs {baseline}: "
                      f"median {100.0 * filtered['median']:+.1f}% "
                      f"wins {filtered['wins']}/{filtered['n']}",
                      flush=True)

    low_outlier_count = 0
    for bkey, by_variant in by_b_variant.items():
        for name, summary in by_variant.items():
            for row in summary["low_outliers"]:
                low_outlier_count += 1
                print(f"LOW OUTLIER B={bkey} {name} r={row['round']} "
                      f"{row['aggregate_tok_s']:.1f} tok/s "
                      f"(median {summary['median']:.1f})",
                      flush=True)

    out = {
        "model": str(model),
        "n_gen": args.n_gen,
        "rounds": args.rounds,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "cooldown": args.cooldown,
        "variants": variants,
        "golden_first_steps": GOLDEN,
        "summary": by_b_variant,
        "comparisons_vs_first_variant": comparisons,
        "paired_comparisons_vs_first_variant": paired_comparisons,
        "paired_comparisons_filtered_low_outliers_vs_first_variant": filtered_paired_comparisons,
        "low_outlier_count": low_outlier_count,
        "runs": raw_runs,
        "ts": time.time(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
