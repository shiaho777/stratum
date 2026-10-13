#!/usr/bin/env python3
"""Cold-start checks for the behaviors this campaign shipped.

Builds the CPU engine if needed, generates a tiny llama and a tiny
qwen35-hybrid in a temporary directory, and asserts:

- short-prompt greedy ids, twice, on each model
- the small-model speed behavior (5 matmul chunks on qwen35, serial
  matmul under a 4 MB mmap)
- the long-context hard stop at STRATUM_MAX_KV=32
- opt-in YaRN on llama and qwen35-hybrid: a request inside the
  trained length keeps unscaled logits; a request past orig engages
  YaRN from the first token and moves logits
- the attendance microbench and the full-lane axpy microbench
- a llama with 80 query heads and a prompt of length >= 2, so batched
  prefill runs a GQA group whose first head is >= 64, exits 0, and
  prints a non-empty argmax on two runs
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
# PATH clang on this host is a swiftly shim whose toolchain is missing.
CC = "/usr/bin/clang" if Path("/usr/bin/clang").is_file() else "clang"

LLAMA_IDS = "142 22 185 32 22 22 22 22"
HYBRID_IDS = "136 62 122 62 69 62 240 248"


def run(cmd, env=None, cwd=None):
    e = os.environ.copy()
    e["STRATUM_NO_GPU"] = "1"
    if env:
        e.update(env)
    return subprocess.run(
        cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def argmax_ids(stderr):
    ids = []
    for line in stderr.splitlines():
        if "stratum_argmax=" in line and "step" in line:
            ids.append(line.split("stratum_argmax=", 1)[1].split()[0])
    return " ".join(ids)


def logit_vals(stderr):
    return tuple(re.findall(r"logit=([-+0-9.eE]+)", stderr))


def expect(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)
    print("ok:", msg)


def compile_bench(src, out):
    p = run(
        [
            CC, "-O3", "-ffast-math", "-mcpu=apple-m1",
            "-I", str(NATIVE), "-o", str(out), str(src),
        ]
    )
    expect(p.returncode == 0, f"compile {src.name}\n{p.stderr}")


def main():
    built = run(
        ["make", "-C", str(NATIVE), "USE_MEMX=0", "USE_METAL=0", f"CC={CC}", "stratum"]
    )
    expect(built.returncode == 0, f"build stratum\n{built.stderr}")
    expect(ENGINE.is_file(), "stratum binary exists")

    with tempfile.TemporaryDirectory(prefix="stratum-campaign-") as td:
        tdir = Path(td)
        llama = tdir / "tiny-llama.gguf"
        hybrid = tdir / "tiny-hybrid.gguf"
        for arch, dest in (("llama", llama), ("qwen35-hybrid", hybrid)):
            g = run([sys.executable, str(GEN), "--arch", arch, "--out", str(dest)])
            expect(g.returncode == 0 and dest.is_file(), f"generate {arch}\n{g.stderr}")

        unscaled_logits = {}
        for label, model, want in (
            ("llama", llama, LLAMA_IDS),
            ("hybrid", hybrid, HYBRID_IDS),
        ):
            seqs = []
            for rep in (1, 2):
                p = run([str(ENGINE), str(model), "8", "1", "2", "3", "4"])
                expect(p.returncode == 0, f"{label} run {rep} exit {p.returncode}")
                got = argmax_ids(p.stderr)
                expect(got == want, f"{label} run {rep} ids {got}")
                seqs.append(got)
                if label == "hybrid":
                    expect("parallel matmul: 5 chunks" in p.stderr, "hybrid 5 chunks")
                expect("small-model matmul: serial" in p.stderr, f"{label} serial matmul")
                expect("rope: none" in p.stderr, f"{label} unscaled rope")
                if label == "llama":
                    expect("rope freq: 32 bytes (inv-freq only)" in p.stderr, "llama inv-freq cache")
                if rep == 1:
                    unscaled_logits[label] = logit_vals(p.stderr)
            expect(seqs[0] == seqs[1] and seqs[0] != "", f"{label} two runs match")

        for label, model in (("llama", llama), ("hybrid", hybrid)):
            p = run(
                [str(ENGINE), str(model), "40", "1", "2", "3", "4"],
                env={"STRATUM_MAX_KV": "32"},
            )
            expect(p.returncode != 0, f"{label} capacity 32 exits")
            expect("exceeds KV capacity" in p.stderr, f"{label} capacity message")
            expect("32 positions" in p.stderr, f"{label} reserved 32")

        forced = run(
            [str(ENGINE), str(hybrid), "8", "1", "2", "3", "4"],
            env={"STRATUM_NCHUNKS": "14"},
        )
        expect(forced.returncode == 0, "forced 14 chunks exits")
        expect("parallel matmul: 14 chunks" in forced.stderr, "14-chunk override")
        expect(argmax_ids(forced.stderr) == HYBRID_IDS, "14-chunk argmax")

        held_line = "rope: yarn held (sequence 12 <= orig 262144, unscaled angles)"
        on_line = "rope: yarn on (sequence 12 > orig 4)"
        yarn_logits = {}
        yarn_env = {
            "STRATUM_ROPE_SCALING": "yarn",
            "STRATUM_ROPE_SCALE": "4",
            "STRATUM_YARN_ORIG": "262144",
            "STRATUM_MAX_KV": "128",
        }
        on_env = dict(yarn_env)
        on_env["STRATUM_YARN_ORIG"] = "4"
        for label, model, want in (
            ("llama", llama, LLAMA_IDS),
            ("hybrid", hybrid, HYBRID_IDS),
        ):
            base128 = run(
                [str(ENGINE), str(model), "8", "1", "2", "3", "4"],
                env={"STRATUM_MAX_KV": "128"},
            )
            expect(base128.returncode == 0, f"{label} unscaled 128 exit")
            expect(argmax_ids(base128.stderr) == want, f"{label} unscaled 128 ids")
            expect("rope: none" in base128.stderr, f"{label} unscaled 128 rope")
            base128_logits = logit_vals(base128.stderr)
            expect(base128_logits == unscaled_logits[label], f"{label} kv cap does not move logits")
            yarn_seqs = []
            for rep in (1, 2):
                yarn = run(
                    [str(ENGINE), str(model), "8", "1", "2", "3", "4"],
                    env=yarn_env,
                )
                if yarn.returncode != 0:
                    expect(False, f"{label} yarn held run {rep} exit {yarn.returncode}\n{yarn.stderr[-600:]}")
                expect(yarn.returncode == 0, f"{label} yarn held run {rep} exit 0")
                expect("rope: yarn factor=4 orig=262144" in yarn.stderr, f"{label} yarn banner {rep}")
                expect(held_line in yarn.stderr, f"{label} yarn held line {rep}")
                got = argmax_ids(yarn.stderr)
                expect(got == want, f"{label} yarn held ids {got}")
                yl = logit_vals(yarn.stderr)
                expect(yl == base128_logits, f"{label} yarn held logits match unscaled")
                expect("128 positions" in yarn.stderr, f"{label} yarn kv cap")
                yarn_seqs.append((got, yl))
            expect(yarn_seqs[0] == yarn_seqs[1], f"{label} yarn held two runs match")
            yarn_logits[label] = yarn_seqs[0][1]
            on_seqs = []
            for rep in (1, 2):
                on = run(
                    [str(ENGINE), str(model), "8", "1", "2", "3", "4"],
                    env=on_env,
                )
                if on.returncode != 0:
                    expect(False, f"{label} yarn on run {rep} exit {on.returncode}\n{on.stderr[-600:]}")
                expect(on.returncode == 0, f"{label} yarn on run {rep} exit 0")
                expect("rope: yarn factor=4 orig=4" in on.stderr, f"{label} yarn on banner {rep}")
                expect(on_line in on.stderr, f"{label} yarn on line {rep}")
                got = argmax_ids(on.stderr)
                expect(got != "" and len(got.split()) == 8, f"{label} yarn on ids {got!r}")
                ol = logit_vals(on.stderr)
                expect(len(ol) == len(base128_logits) and ol != base128_logits,
                       f"{label} yarn on logits moved")
                on_seqs.append((got, ol))
            expect(on_seqs[0] == on_seqs[1], f"{label} yarn on two runs match")

        warned = tdir / "tiny-llama-ctx64.gguf"
        g = run(
            [sys.executable, str(GEN), "--arch", "llama", "--out", str(warned),
             "--context-length", "64"],
        )
        expect(g.returncode == 0 and warned.is_file(), f"generate ctx64\n{g.stderr}")
        wrun = run(
            [str(ENGINE), str(warned), "8", "1", "2", "3", "4"],
            env={"STRATUM_MAX_KV": "128"},
        )
        if wrun.returncode != 0:
            expect(False, f"ctx64 exit {wrun.returncode}\n{wrun.stderr[-400:]}")
        expect(wrun.returncode == 0, "ctx64 exit 0")
        expect("rope: none" in wrun.stderr, "ctx64 stays unscaled")
        expect("exceeds context_length 64" in wrun.stderr, "ctx64 overshoot warning")
        expect(argmax_ids(wrun.stderr) == LLAMA_IDS, "ctx64 ids")
        expect(logit_vals(wrun.stderr) == unscaled_logits["llama"], "ctx64 logits")
        expect("128 positions" in wrun.stderr, "ctx64 keeps the override cap")

        yfile = tdir / "tiny-llama-yarn.gguf"
        g = run(
            [sys.executable, str(GEN), "--arch", "llama", "--out", str(yfile),
             "--context-length", "64", "--rope-scaling", "yarn",
             "--rope-scale", "4", "--yarn-orig", "262144"],
        )
        expect(g.returncode == 0 and yfile.is_file(), f"generate yarn file\n{g.stderr}")
        frun = run(
            [str(ENGINE), str(yfile), "8", "1", "2", "3", "4"],
            env={"STRATUM_MAX_KV": "128"},
        )
        if frun.returncode != 0:
            expect(False, f"yarn file exit {frun.returncode}\n{frun.stderr[-400:]}")
        expect(frun.returncode == 0, "yarn file exit 0")
        expect("rope: yarn factor=4 orig=262144" in frun.stderr, "yarn file banner")
        expect(held_line in frun.stderr, "yarn file held")
        expect("exceeds context_length" not in frun.stderr, "yarn file does not warn")
        expect(logit_vals(frun.stderr) == yarn_logits["llama"], "yarn file matches held env")
        expect(logit_vals(frun.stderr) == unscaled_logits["llama"], "yarn file matches unscaled")

        wide = tdir / "tiny-llama-nq80.gguf"
        g = run(
            [sys.executable, str(GEN), "--arch", "llama", "--out", str(wide)],
            env={
                "STRATUM_TINY_NQ": "80",
                "STRATUM_TINY_NK": "8",
                "STRATUM_TINY_LAYERS": "2",
            },
        )
        expect(g.returncode == 0 and wide.is_file(), f"generate nq80 llama\n{g.stderr}")
        expect(wide.stat().st_size < 4_000_000, f"nq80 file bytes {wide.stat().st_size}")
        wide_seqs = []
        for rep in (1, 2):
            p = run(
                [str(ENGINE), str(wide), "4", "1", "2", "3", "4"],
                env={"STRATUM_MAX_KV": "128"},
            )
            if p.returncode != 0:
                expect(False, f"nq80 run {rep} exit {p.returncode}\n{p.stderr[-800:]}")
            expect(p.returncode == 0, f"nq80 run {rep} exit 0")
            expect(re.search(r"n_q_heads\s+:\s+80\b", p.stderr) is not None, "nq80 query heads")
            expect(re.search(r"n_kv_heads\s+:\s+8\b", p.stderr) is not None, "nq80 kv heads")
            expect("128 positions" in p.stderr, "nq80 kv cap")
            got = argmax_ids(p.stderr)
            expect(got != "" and len(got.split()) == 4, f"nq80 run {rep} ids {got!r}")
            wide_seqs.append(got)
        expect(wide_seqs[0] == wide_seqs[1], "nq80 two runs match")

        longp = tdir / "prompt2050.txt"
        longp.write_text(" ".join(["1"] * 2050))
        for label, model in (("llama", llama), ("hybrid", hybrid)):
            p = run(
                [str(ENGINE), str(model), "1", f"@{longp}"],
                env={"STRATUM_MAX_KV": "4096"},
            )
            if p.returncode != 0:
                expect(False, f"{label} long prompt exit {p.returncode}\n{p.stderr[-800:]}")
            expect(p.returncode == 0, f"{label} long prompt exit 0")
            expect("prompt length: 2050" in p.stderr, f"{label} accepted 2050")
            expect("prompt truncated" not in p.stderr, f"{label} long prompt not truncated")
            expect("exceeds KV capacity" not in p.stderr, f"{label} long prompt within cap")
            got = argmax_ids(p.stderr)
            expect(len(got.split()) == 1, f"{label} long prompt one token {got!r}")

        moe = tdir / "tiny-moe.gguf"
        g = run([sys.executable, str(GEN), "--arch", "moe", "--out", str(moe)])
        expect(g.returncode == 0 and moe.is_file(), f"generate moe\n{g.stderr}")
        mp = tdir / "prompt1100.txt"
        mp.write_text(" ".join(["1"] * 1100))
        p = run(
            [str(ENGINE), str(moe), "1", f"@{mp}"],
            env={"STRATUM_MAX_KV": "2048"},
        )
        if p.returncode != 0:
            expect(False, f"moe long prompt exit {p.returncode}\n{p.stderr[-800:]}")
        expect(p.returncode == 0, "moe long prompt exit 0")
        expect("prompt length: 1100" in p.stderr, "moe accepted 1100")
        expect("(cap 2048)" in p.stderr, "moe cap 2048")
        expect("prompt truncated" not in p.stderr, "moe long prompt not truncated")
        expect("exceeds KV capacity" not in p.stderr, "moe long prompt within cap")
        got = argmax_ids(p.stderr)
        expect(len(got.split()) == 1, f"moe long prompt one token {got!r}")
        p32 = run(
            [str(ENGINE), str(moe), "40", "1", "2", "3", "4"],
            env={"STRATUM_MAX_KV": "32"},
        )
        expect(p32.returncode != 0, "moe capacity 32 exits")
        expect("exceeds KV capacity" in p32.stderr, "moe capacity message")

        aw = tdir / "bench_attn_window"
        ax = tdir / "bench_axpy"
        lg = tdir / "bench_attn_lg"
        ry = tdir / "bench_rope_yarn"
        rel = tdir / "bench_relativity"
        fr = tdir / "bench_f16_fewrow"
        compile_bench(NATIVE / "bench_attn_window.c", aw)
        compile_bench(NATIVE / "bench_axpy.c", ax)
        compile_bench(NATIVE / "bench_attn_lg.c", lg)
        compile_bench(NATIVE / "bench_rope_yarn.c", ry)
        compile_bench(NATIVE / "bench_relativity.c", rel)
        compile_bench(NATIVE / "bench_f16_fewrow.c", fr)
        w = run([str(aw)])
        expect(w.returncode == 0 and "attn_window_ok" in w.stdout, "attn window")
        expect("legacy_cap4096_visible=4096 legacy_has_100=0" in w.stdout, "legacy ring")
        expect("shipped_cap8192_visible=5000 shipped_has_100=1" in w.stdout, "full prefix")
        expect("shipped_short16_visible=16" in w.stdout, "short prefix")
        x = run([str(ax)])
        expect(x.returncode == 0 and "axpy_ok" in x.stdout, "axpy")
        expect("axpy_hd16_max_abs_diff=0" in x.stdout, "axpy hd16")
        expect("axpy_hd20_max_abs_diff=0" in x.stdout, "axpy hd20")
        g = run([str(lg)])
        expect(g.returncode == 0 and "attn_lg_ok" in g.stdout, f"attn lg\n{g.stdout}{g.stderr}")
        expect("nq=80 nk=8" in g.stdout and "h0_ge_64=1" in g.stdout, "attn lg geometry")
        yb = run([str(ry)])
        expect(yb.returncode == 0 and "rope_yarn_ok" in yb.stdout, f"rope yarn\n{yb.stdout}{yb.stderr}")
        expect("rope_yarn_corr_low=14 rope_yarn_corr_high=22" in yb.stdout, "yarn corr range")
        expect("rope_none_max_abs_diff=0" in yb.stdout, "unscaled freq")
        expect("rope_gpu_full_4096=1 rope_gpu_full_1m=0 rope_gpu_full_yarn=0" in yb.stdout,
               "gpu full gate")
        expect("rope_hold_inside=1 rope_hold_past=0" in yb.stdout, "yarn hold predicate")
        rb = run([str(rel)])
        expect(rb.returncode == 0 and "relativity_ok" in rb.stdout,
               f"relativity\n{rb.stdout}{rb.stderr}")
        expect("rel5_gpu_full_4096=1 rel5_gpu_full_1m=0 rel5_gpu_full_yarn=0" in rb.stdout,
               "relativity two-clock gate")
        expect("rel6_willneed_abs_diff=0" in rb.stdout, "relativity WILLNEED")
        fb = run([str(fr)])
        expect(fb.returncode == 0 and "f16_fewrow_ok" in fb.stdout,
               f"f16 fewrow\n{fb.stdout}{fb.stderr}")
        expect("f16_fewrow_max_abs_diff=0" in fb.stdout, "f16 fewrow exact")

    print("campaign_tiny_test_ok")


if __name__ == "__main__":
    main()
