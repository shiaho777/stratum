#!/usr/bin/env python3
"""env_census.py — Stratum STRATUM_* environment-variable census.

Scans stratum/native/ for every getenv/setenv of a "STRATUM_*" variable,
aggregates per-variable call sites, classifies each variable against the
project's sanctioned / forbidden lists (AGENTS.md), and writes a markdown
table to stratum/docs/ENVVARS.md.

This is the single map for the engine's 200+ runtime switches — when a
switch's meaning is in doubt, this table plus the call sites it links to
is the source of truth, not the README table.

Usage:
    python3 stratum/tools/env_census.py          # write docs/ENVVARS.md
    python3 stratum/tools/env_census.py --stdout # print instead of write
"""

import os
import re
import sys
from collections import defaultdict

NATIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native")
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
OUT = os.path.join(DOCS, "ENVVARS.md")

GETENV_RE = re.compile(r'getenv\("(STRATUM_[A-Z0-9_]+)"')
SETENV_RE = re.compile(r'setenv\("(STRATUM_[A-Z0-9_]+)",')

# Status classification — keep in sync with AGENTS.md.
FORBIDDEN = {
    # quality boundary
    "STRATUM_Q4_0": "requantizes Q4_K→Q4_0 (precision loss)",
    "STRATUM_PREDECODE": "pre-decodes weights to GPU F16 buffers (quality + memory)",
    # memory boundary
    "STRATUM_MLOCK_ALL": "mlock whole model",
    "STRATUM_HOT_GB": "hot-set budget tied to pinning path",
    "STRATUM_KEEP_RESIDENT": "page-cache lock (implicit mlock)",
}
SANCTIONED = {
    "STRATUM_NO_GPU": "force CPU-only",
    "STRATUM_GPU_NC": "per-tensor NoCopy direct-read",
    "STRATUM_GPU2": "cold-weight staging pipeline",
    "STRATUM_GPU": "legacy GPU paths (small models only)",
    "STRATUM_GPU_FULL": "legacy GPU full paths (small models only)",
    "STRATUM_MULTISEQ": "N sequences share one weight scan",
    "STRATUM_NGRAM_SPEC": "n-gram speculative decoding",
    "STRATUM_MTP": "MTP tree speculative decoding",
    "STRATUM_ASYNC_PREFETCH": "background pread prefetch",
    "STRATUM_HOT_FAST": "hot-cache pure-compute mode (no page-cache lock)",
    "STRATUM_STREAM_DET": "deterministic streaming (skip mincore)",
    "STRATUM_Q2K_NIB": "Q2K nibble-layout model path",
    "STRATUM_Q2K_NIB_OFF": "disable nibble path",
    "STRATUM_TREE_EXTEND_K": "tree chain depth (cap 12)",
    "STRATUM_Q2K_SDOT": "int8 SDOT for Q2K (opt-in)",
}


def classify(name):
    if name in FORBIDDEN:
        return "forbidden", FORBIDDEN[name]
    if name in SANCTIONED:
        return "sanctioned", SANCTIONED[name]
    return "experimental", ""  # historical tuning / variant toggles


def nearest_comment(lines, idx, max_back=3):
    """Return the first comment line found within max_back lines above idx."""
    for j in range(idx - 1, max(idx - 1 - max_back, -1), -1):
        line = lines[j].strip()
        if line.startswith("/*") or line.startswith("//"):
            return line.lstrip("/").strip()
    return ""


def main():
    call_sites = defaultdict(list)  # var -> [(file, line, kind, comment)]
    for root, _dirs, files in os.walk(NATIVE):
        for fn in sorted(files):
            if not fn.endswith((".c", ".h", ".m", ".metal")):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            rel = os.path.relpath(path, os.path.dirname(NATIVE))
            for i, line in enumerate(lines):
                for m in GETENV_RE.finditer(line):
                    call_sites[m.group(1)].append(
                        (rel, i + 1, "read", nearest_comment(lines, i)))
                for m in SETENV_RE.finditer(line):
                    call_sites[m.group(1)].append(
                        (rel, i + 1, "write", nearest_comment(lines, i)))

    rows = []
    for name in sorted(call_sites):
        sites = call_sites[name]
        status, note = classify(name)
        n_read = sum(1 for s in sites if s[2] == "read")
        n_write = sum(1 for s in sites if s[2] == "write")
        # first non-empty comment, else first comment
        comments = [s[3] for s in sites if s[3]]
        ctx = comments[0] if comments else ""
        locs = ", ".join(f"{s[0]}:{s[1]}" for s in sites[:4])
        if len(sites) > 4:
            locs += f" (+{len(sites) - 4} more)"
        rows.append((name, status, n_read, n_write, ctx, locs, note))

    # Forbidden vars that survive only as string literals inside
    # stratum_enforce_boundaries() have no getenv call sites — list them
    # anyway so the doc is a complete map of the boundary surface.
    scanned = set(call_sites.keys())
    for name in sorted(FORBIDDEN):
        if name not in scanned:
            status, note = classify(name)
            rows.append((name, status, 0, 0,
                         "string-literal in stratum_enforce_boundaries()",
                         "native/stratum_engine.h", note))

    lines_out = [
        "# Stratum environment variables (generated by `stratum/tools/env_census.py`)",
        "",
        "Every `STRATUM_*` variable referenced in `stratum/native/`. This is the",
        "single map for the engine's runtime switches — when in doubt about what a",
        "switch does, follow its call sites rather than the README table.",
        "",
        "| Variable | Status | reads | writes | Context | Call sites |",
        "|---|---|---|---|---|---|",
    ]
    for name, status, nr, nw, ctx, locs, note in rows:
        ctx_c = ctx.replace("|", "/").replace("`", "").strip()
        note_c = note.replace("|", "/")
        if note_c:
            status_disp = f"{status} — {note_c}"
        else:
            status_disp = status
        lines_out.append(
            f"| `{name}` | {status_disp} | {nr} | {nw} | {ctx_c} | {locs} |"
        )

    lines_out += [
        "",
        "## Status legend",
        "",
        "- **forbidden** — violates a project hard boundary; setting a non-zero",
        "  value is refused at startup by `stratum_enforce_boundaries()`.",
        "- **sanctioned** — listed in AGENTS.md as an allowed path.",
        "- **experimental** — historical tuning / kernel-variant toggles from the",
        "  V-series experiments. They are read by live code but are not part of",
        "  any documented configuration surface; treat their defaults as measured",
        "  choices, not API.",
    ]

    text = "\n".join(lines_out) + "\n"
    if "--stdout" in sys.argv:
        print(text)
    else:
        os.makedirs(DOCS, exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {OUT} ({len(rows)} variables)")


if __name__ == "__main__":
    main()
