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
# First of these ends the recorded answer. Later copies are the cap padding.
TURN_END_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")


def clip_at_turn_end(events):
    """Return events through the first end-of-turn marker, inclusive."""
    kept = []
    acc = ""
    for ev in events:
        start = len(acc)
        piece = ev.get("text") or ""
        acc += piece
        cut = None
        for marker in TURN_END_MARKERS:
            i = acc.find(marker)
            if i >= 0 and (cut is None or i < cut):
                cut = i + len(marker)
        if cut is None:
            kept.append(ev)
            continue
        take = cut - start
        if take > 0:
            text = piece[:take]
            kept.append({
                "t": ev["t"],
                "bytes": len(text.encode("utf-8")),
                "text": text,
            })
        return kept, _marker_at(acc, cut)
    return kept, None


def llama_generation_events(events):
    """Drop the echoed prompt. llama prints it, then `assistant:` and the tokens."""
    acc = ""
    spans = []
    for ev in events:
        start = len(acc)
        acc += ev.get("text") or ""
        spans.append((start, len(acc), ev))
    header = "<|im_start|>assistant"
    i = acc.rfind(header)
    if i < 0:
        return events
    j = i + len(header)
    if j < len(acc) and acc[j] == ":":
        j += 1
    if j < len(acc) and acc[j] == "\n":
        j += 1
    out = []
    for start, end, ev in spans:
        if end <= j:
            continue
        text = ev.get("text") or ""
        if start < j:
            text = text[j - start:]
        if not text:
            continue
        out.append({
            "t": ev["t"],
            "text": text,
            "bytes": len(text.encode("utf-8")),
        })
    return out


def _marker_at(acc, cut):
    for marker in TURN_END_MARKERS:
        if acc.rfind(marker, 0, cut) == cut - len(marker):
            return marker
    return None


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
    elif engine == "stratum_gpu":
        env = dict(os.environ, STRATUM_GPU="1", STRATUM_GPU_FULL="1",
                   STRATUM_GPU_FUSED_ARGMAX="1")
        if "--chain" in sys.argv:
            env["STRATUM_GPU_CHAIN"] = "1"
        cmd = [binpath, model, str(n_gen)] + [str(i) for i in prompt_ids]
        prompt_echo_len = 0
    elif engine == "llamacpp":
        env = dict(os.environ)
        # llama-completion (stock b9180): honors --ignore-eos natively, raw
        # completion mode, exits on stdin EOF. -ngl from --ngl flag.
        ngl = "0"
        for a in sys.argv:
            if a.startswith("--ngl="):
                ngl = a.split("=", 1)[1]
        # -no-cnv: conversation mode re-wraps an already-templated prompt.
        # -sp prints special tokens the way stratum does. temp 0 is greedy;
        # the binary defaults to temperature 0.80, which made the two
        # panels think in different words. EOS stays enabled.
        prompt_echo_len = 0
        cmd = [binpath, "-m", model, "-n", str(n_gen), "-ngl", ngl,
               "-no-cnv", "-sp", "--temp", "0", "--top-k", "1",
               "--top-p", "1", "--min-p", "0", "-p", prompt_text]
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
        "prompt_echo_len": prompt_echo_len,
    }

    # stratum auto-detects stratum_q4k.metallib from CWD then native/ —
    # run from the binary's own directory so GPU paths actually engage.
    cwd = os.path.dirname(os.path.abspath(binpath)) if engine.startswith("stratum") else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, env=env, cwd=cwd)
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
    # llama-completion appends "\n> EOF by user" after the last token — cut it
    for ev in reversed(gen_events):
        m = ev["text"].find("> EOF by user")
        if m >= 0:
            ev["text"] = ev["text"][:m]
            ev["bytes"] = len(ev["text"].encode("utf-8"))
            break
        if ev["bytes"] > 0:
            break
    # Keep the turn, drop the padding. Both engines are asked for n_gen
    # as a cap. stratum does not stop on EOS, so the text after the first
    # end-of-turn marker is the same answer repeated as special tokens or
    # a new prompt. The recorded answer stops at that marker.
    if engine == "llamacpp":
        gen_events = llama_generation_events(gen_events)
    answer_events, stop_marker = clip_at_turn_end(gen_events)
    rec["answer_events"] = answer_events
    rec["stop_marker"] = stop_marker
    rec["n_shown"] = len(answer_events)
    rec["gen_chunks"] = len(answer_events)
    rec["gen_bytes"] = sum(e["bytes"] for e in answer_events)
    shown = rec["n_shown"] or n_gen
    if answer_events:
        rec["t_first_gen_chunk_s"] = round(answer_events[0]["t"], 3)
        rec["t_last_gen_chunk_s"] = round(answer_events[-1]["t"], 3)
        rec["t_answer_end_s"] = rec["t_last_gen_chunk_s"]
        span = answer_events[-1]["t"] - answer_events[0]["t"]
        rec["gen_span_s"] = round(span, 3)
        rec["tok_per_s_stream"] = round(shown / span, 2) if span > 0 else None
    rec["tok_per_s_wall"] = round(shown / rec["wall_s"], 2) if rec["wall_s"] else None
    rec["mem_samples"] = len(rec["mem_series"])

    with open(outjson, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[{engine}] {os.path.basename(model)} wall={rec['wall_s']}s "
          f"shown={rec['n_shown']} stop={rec['stop_marker']} "
          f"tok/s(stream)={rec.get('tok_per_s_stream')} "
          f"peak_anon={rec['peak_anon_mb']:.0f}MB peak_file={rec['peak_file_mb']:.0f}MB "
          f"-> {outjson}")


if __name__ == "__main__":
    main()
