#!/usr/bin/env python3
"""Engine-side half of the six relativity experiments.

The C microbench (bench_relativity.c) covers the identities.
This script checks the same claims on a generated tiny llama:

1. Chart size (MAX_KV) does not move short-request logits
2. YaRN held inside orig matches unscaled logits
3. Spacelike pool (PAR_FORCE) does not change greedy ids
5. Two-clock GPU-full gate is already in the C bench (CPU build)
6. Next-layer WILLNEED (STRATUM_PF_NEXT) does not change logits
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
ENGINE = NATIVE / "stratum"
GEN = ROOT / "tools" / "make_tiny_model.py"
CC = "/usr/bin/clang" if Path("/usr/bin/clang").is_file() else "clang"


def run(cmd, env=None):
    e = os.environ.copy()
    e["STRATUM_NO_GPU"] = "1"
    if env:
        e.update(env)
    return subprocess.run(
        cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def expect(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)
    print("ok:", msg)


def argmax_ids(stderr):
    ids = []
    for line in stderr.splitlines():
        if "stratum_argmax=" in line and "step" in line:
            ids.append(line.split("stratum_argmax=", 1)[1].split()[0])
    return " ".join(ids)


def logit_vals(stderr):
    return tuple(re.findall(r"logit=([-+0-9.eE]+)", stderr))


def main():
    built = run(
        ["make", "-C", str(NATIVE), "USE_MEMX=0", "USE_METAL=0", f"CC={CC}", "stratum"]
    )
    expect(built.returncode == 0, f"build\n{built.stderr[-400:]}")

    bench = Path(tempfile.mkdtemp(prefix="stratum-rel-")) / "bench_relativity"
    c = run(
        [
            CC, "-O3", "-ffast-math", "-mcpu=apple-m1",
            "-I", str(NATIVE), "-o", str(bench),
            str(NATIVE / "bench_relativity.c"),
        ]
    )
    expect(c.returncode == 0, f"compile bench\n{c.stderr}")
    b = run([str(bench)])
    expect(b.returncode == 0 and "relativity_ok" in b.stdout, f"bench\n{b.stdout}{b.stderr}")
    expect("rel1_ratio=87381" in b.stdout, "chart/geom ratio")
    expect("rel2_hold_short=1" in b.stdout and "rel2_hold_long=0" in b.stdout, "local frame")
    expect("rel3_spacelike_commute=1" in b.stdout, "spacelike")
    expect("rel4_mixed_abs_diff=" in b.stdout, "mixed frames printed")
    expect("rel5_gpu_full_4096=1 rel5_gpu_full_1m=0 rel5_gpu_full_yarn=0" in b.stdout,
           "two-clock gpu gate")
    expect("rel6_prefetch_hit=0010" in b.stdout, "prefetch cone")
    expect("rel6_willneed_abs_diff=0" in b.stdout, "WILLNEED is advisory")
    print(b.stdout.strip())

    with tempfile.TemporaryDirectory(prefix="stratum-rel-mod-") as td:
        tdir = Path(td)
        llama = tdir / "tiny-llama.gguf"
        g = run([sys.executable, str(GEN), "--arch", "llama", "--out", str(llama)])
        expect(g.returncode == 0 and llama.is_file(), f"generate llama\n{g.stderr}")
        cmd = [str(ENGINE), str(llama), "8", "1", "2", "3", "4"]

        a = run(cmd, env={"STRATUM_MAX_KV": "128"})
        b128 = run(cmd, env={"STRATUM_MAX_KV": "4096"})
        expect(a.returncode == 0 and b128.returncode == 0, "cap runs exit 0")
        expect(logit_vals(a.stderr) == logit_vals(b128.stderr),
               "1 chart size does not move logits")
        expect(argmax_ids(a.stderr) == argmax_ids(b128.stderr),
               "1 chart size does not move ids")

        y = run(cmd, env={
            "STRATUM_ROPE_SCALING": "yarn",
            "STRATUM_ROPE_SCALE": "4",
            "STRATUM_YARN_ORIG": "262144",
            "STRATUM_MAX_KV": "128",
        })
        expect(y.returncode == 0, "yarn held exit 0")
        expect("rope: yarn held" in y.stderr, "2 yarn held banner")
        expect(logit_vals(y.stderr) == logit_vals(a.stderr),
               "2 local frame matches unscaled")

        serial = run(cmd)
        forced = run(cmd, env={"STRATUM_PAR_FORCE": "1"})
        expect(serial.returncode == 0 and forced.returncode == 0, "par runs exit 0")
        expect(argmax_ids(serial.stderr) == argmax_ids(forced.stderr),
               "3 spacelike pool does not change ids")
        expect(logit_vals(serial.stderr) == logit_vals(forced.stderr),
               "3 spacelike pool does not change logits")

        pf = run(cmd, env={"STRATUM_PF_NEXT": "1"})
        expect(pf.returncode == 0, "prefetch run exit 0")
        expect(logit_vals(pf.stderr) == logit_vals(serial.stderr),
               "6 next-layer WILLNEED does not change logits")
        expect(argmax_ids(pf.stderr) == argmax_ids(serial.stderr),
               "6 next-layer WILLNEED does not change ids")

    print("relativity_experiment_ok")


if __name__ == "__main__":
    main()
