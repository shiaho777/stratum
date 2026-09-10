#!/usr/bin/env python3
"""run_h2h.py — head-to-head measurement harness: stratum vs llama.cpp.

Runs one (model x engine) case: spawns the engine as a subprocess,
captures per-chunk stdout timestamps (token stream timing), samples
vmmap memory every 0.2s (anonymous = Physical footprint; mapped file =
reclaimable weight pages), and writes a JSON record.

usage: run_h2h.py <stratum_bin|llama_simple_bin> <model.gguf> <engine:stratum|llamacpp>
                  <prompt_text> <prompt_ids_csv> <n_gen> <out.json> [--sdot0]
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

VM_INTERVAL = 0.2


def parse_mb(v):
    """'123.4M' / '12K' / '1.2G' -> MB float"""
    try:
        if v.endswith("K"):
            return float(v[:-1]) / 1024.0
        if v.endswith("M"):
            return float(v[:-1])
        if v.endswith("G"):
            return float(v[:-1]) * 1024.0
    except ValueError:
        pass
    return 0.0


def mem_sampler(pid, stop, rec):
    """Poll vmmap --summary until stop; record anon/file MB series."""
    while not stop.is_set() and pid > 0:
        try:
            out = subprocess.run(["vmmap", "--summary", str(pid)],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            break
        fp = mf = None
        for line in out.splitlines():
            if fp is None and line.startswith("Physical footprint:"):
                parts = line.split()
                if len(parts) >= 3:
                    fp = parse_mb(parts[2])
            if line.startswith("mapped file") and "SM=COW" not in line:
                parts = line.split()
                # "mapped file  <virtual>  <resident>  ..." -> resident is col 3
                if len(parts) >= 3 and mf is None:
                    mf = parse_mb(parts[2])
                break  # only the first mapped-file line
        if fp is not None:
            rec["mem_series"].append({
                "t": round(time.time() - rec["t0"], 3),
                "anon_mb": round(fp, 2),
                "file_mb": round(mf or 0.0, 2),
            })
            rec["peak_anon_mb"] = max(rec["peak_anon_mb"], fp)
            rec["peak_file_mb"] = max(rec["peak_file_mb"], mf or 0.0)
        time.sleep(VM_INTERVAL)


def stdout_reader(fd, rec):
    """Timestamp every arriving stdout chunk."""
    while True:
        try:
            data = os.read(fd, 65536)
        except (OSError, ValueError):
            break
        if not data:
            break
        rec["stdout_events"].append({
            "t": round(time.time() - rec["t0"], 4),
            "bytes": len(data),
            "text": data.decode("utf-8", errors="replace"),
        })


def main():
    (binpath, model, engine, prompt_text, prompt_ids_csv, n_gen, outjson) = sys.argv[1:8]
    sdot0 = "--sdot0" in sys.argv
    prompt_ids = [int(x) for x in prompt_ids_csv.split(",") if x.strip()]
    n_gen = int(n_gen)

    if engine == "stratum":
        env = dict(os.environ, STRATUM_NO_GPU="1")
        if sdot0:
            env["STRATUM_SDOT"] = "0"
        if "--hotfast" in sys.argv:
            env["STRATUM_HOT_FAST"] = "1"
        cmd = [binpath, model, str(n_gen)] + [str(i) for i in prompt_ids]
        prompt_echo_len = 0            # stratum prints only generated tokens
    elif engine == "llamacpp":
        env = dict(os.environ)
        # llama-simple's argv parser treats the first unknown flag as the
        # start of the prompt — anything else silently becomes prompt text.
        # It understands exactly: -m, -n, -ngl, --ignore-eos (patched in
        # our local build to honor it). Sampler defaults to greedy.
        cmd = [binpath, "-m", model, "-n", str(n_gen), "-ngl", "0",
               "--ignore-eos", prompt_text]
        prompt_echo_len = len(prompt_text)   # llama-simple echoes the prompt
    else:
        raise SystemExit(f"unknown engine {engine}")

    rec = {
        "engine": engine, "model_path": model,
        "model_size_mb": round(os.path.getsize(model) / 1024 / 1024, 1),
        "n_prompt_tokens": len(prompt_ids), "n_gen": n_gen,
        "sdot0": sdot0,
        "prompt_text": prompt_text,
        "t0": time.time(),
        "stdout_events": [], "mem_series": [],
        "peak_anon_mb": 0.0, "peak_file_mb": 0.0,
        "llama_no_mmap": "--nommap" in sys.argv,
    }

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env)
    stop = threading.Event()
    t_mem = threading.Thread(target=mem_sampler, args=(proc.pid, stop, rec), daemon=True)
    t_out = threading.Thread(target=stdout_reader, args=(proc.stdout.fileno(), rec), daemon=True)
    if "--nomen" not in sys.argv:      # speed run: vmmap suspends the target
        t_mem.start()                  # and would skew short-run timings
    t_out.start()

    stderr_data = proc.stderr.read()   # serializes wait; reader drains stdout
    rc = proc.wait()
    stop.set()
    t_out.join(timeout=2)
    rec["exit_code"] = rc
    rec["wall_s"] = round(time.time() - rec["t0"], 3)
    rec["stderr_tail"] = stderr_data.decode("utf-8", errors="replace")[-4000:]

    # derived: generation stream = stdout after the prompt echo
    gen_events = []
    echoed = 0
    for ev in rec["stdout_events"]:
        if echoed < prompt_echo_len:
            echoed += ev["bytes"]
            continue
        gen_events.append(ev)
    rec["gen_chunks"] = len(gen_events)
    rec["gen_bytes"] = sum(e["bytes"] for e in gen_events)
    if gen_events:
        rec["t_first_gen_chunk_s"] = round(gen_events[0]["t"], 3)
        rec["t_last_gen_chunk_s"] = round(gen_events[-1]["t"], 3)
        span = gen_events[-1]["t"] - gen_events[0]["t"]
        rec["gen_span_s"] = round(span, 3)
        rec["tok_per_s_stream"] = round(n_gen / span, 2) if span > 0 else None
    rec["tok_per_s_wall"] = round(n_gen / rec["wall_s"], 2)
    rec["mem_samples"] = len(rec["mem_series"])

    with open(outjson, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[{engine}] {os.path.basename(model)} wall={rec['wall_s']}s "
          f"tok/s(wall)={rec['tok_per_s_wall']} "
          f"tok/s(stream)={rec.get('tok_per_s_stream')} "
          f"peak_anon={rec['peak_anon_mb']:.0f}MB peak_file={rec['peak_file_mb']:.0f}MB "
          f"-> {outjson}")


if __name__ == "__main__":
    main()
