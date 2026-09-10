#!/usr/bin/env python3
"""render_h2h.py — render a side-by-side engine comparison video with a
floating HUD (real-time tok/s, anonymous memory, page cache), styled like a
game FPS overlay.

usage: render_h2h.py <left.json> <right.json> <out.mp4> [--title "..."] [--fps 30]
The two JSON records come from run_h2h.py (speed pass for timing, memory
pass for the memory curves — pass the memory JSON and we merge by wall-clock
if a matching *_mem_run2.json exists next to the speed JSON).
"""
import json
import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 30
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_SANS = "/System/Library/Fonts/Helvetica.ttc"

BG = (16, 18, 24)
PANEL = (24, 28, 36)
PANEL_EDGE = (48, 54, 66)
STRATUM_ACCENT = (52, 211, 153)     # green
LLAMA_ACCENT = (96, 165, 250)      # blue
TEXT = (226, 232, 240)
TEXT_DIM = (120, 130, 145)
WARN = (251, 146, 60)


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def parse_size(v):
    v = v.strip()
    if v.endswith("K"):
        return float(v[:-1]) / 1024.0
    if v.endswith("M"):
        return float(v[:-1])
    if v.endswith("G"):
        return float(v[:-1]) * 1024.0
    return 0.0


PROMPT_TEXT = '''<|im_start|>user
What is the capital of France? Answer in one short sentence.<|im_end|>
<|im_start|>assistant
'''


class Case:
    def __init__(self, path):
        self.rec = json.load(open(path))
        # merge the memory pass if present (same dir, *_speed_ -> *_mem_)
        mem = path.replace("_speed_", "_mem_")
        if "_speed_" in path and os.path.exists(mem):
            self.mem = json.load(open(mem))
        else:
            self.mem = self.rec
        self.is_stratum = self.rec["engine"] == "stratum"
        self.accent = STRATUM_ACCENT if self.is_stratum else LLAMA_ACCENT
        self.name = "stratum" if self.is_stratum else "llama.cpp"
        # generation text timeline: [(t, text)]
        # llama-simple echoes the prompt first — drop those leading bytes.
        # prompt text: prefer the record's own (multi-model harness);
        # fall back to the legacy Qwen3 template for the first video set.
        prompt_text = self.rec.get("prompt_text") or PROMPT_TEXT
        events = []
        if self.is_stratum:
            events = self.rec["stdout_events"]
        else:
            skip = len(prompt_text)
            acc = ""
            for ev in self.rec["stdout_events"]:
                if skip > 0:
                    take = ev["text"][skip:]
                    skip -= len(ev["text"])
                    if not take and skip > 0:
                        continue
                    if skip <= 0:
                        events.append({"t": ev["t"], "text": take})
                    continue
                events.append(ev)
        self.events = events
        self.wall = self.rec["wall_s"]
        self.n_gen = self.rec["n_gen"]
        self.total_chunks = max(1, len(self.events))
        self.t_first = self.events[0]["t"] if self.events else 0.0
        # drop memory samples taken at/after process exit (the sampler races
        # with teardown and reports a bogus near-zero footprint there)
        self.mem["mem_series"] = [p for p in self.mem["mem_series"]
                                   if p["t"] < self.mem["wall_s"] * 0.95]
        self.peak_anon = self.mem.get("peak_anon_mb", 0.0)
        self.peak_file = self.mem.get("peak_file_mb", 0.0)

    def text_at(self, t):
        out = []
        for ev in self.events:
            if ev["t"] <= t:
                out.append(ev["text"])
        return "".join(out)

    def progress_at(self, t):
        """fraction of generated tokens that have arrived by t (0..1)"""
        n = sum(1 for ev in self.events if ev["t"] <= t)
        return n / self.total_chunks

    def chunks_before(self, t):
        return sum(1 for ev in self.events if ev["t"] <= t)

    def _mem_time(self, t):
        """remap video time (speed-run axis) onto the memory-run axis"""
        mw = self.mem.get("wall_s") or 1.0
        return t / max(self.wall, 1e-3) * mw

    def anon_at(self, t):
        s = self.mem["mem_series"]
        tn = self._mem_time(t)
        v = 0.0
        for pt in s:
            if pt["t"] <= tn:
                v = pt["anon_mb"]
        return v

    def file_at(self, t):
        s = self.mem["mem_series"]
        tn = self._mem_time(t)
        v = 0.0
        for pt in s:
            if pt["t"] <= tn:
                v = pt["file_mb"]
        return v

    def tokps_at(self, t):
        """instantaneous tok/s: token count (progress * n_gen) over elapsed
        time since generation started, smoothed by requiring some progress."""
        if t <= self.t_first + 0.05:
            return 0.0
        prog = self.progress_at(t)
        if prog <= 0:
            return 0.0
        return min(prog * self.n_gen / (t - self.t_first), 999.0)


def wrap(draw, text, fnt, maxw):
    """Wrap text into lines of maxw pixels. Returns list of lines."""
    lines = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        trial = cur + ch
        if draw.textlength(trial, font=fnt) > maxw:
            lines.append(cur)
            cur = ch
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_case_panel(draw, case, x, y, w, h, t, fnts):
    f_title, f_mono, f_hud, f_big, f_small = fnts
    # panel
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=PANEL,
                            outline=PANEL_EDGE, width=2)
    # header
    draw.text((x + 16, y + 14), case.name, font=f_title, fill=case.accent)
    sub = os.path.basename(case.rec["model_path"])
    draw.text((x + 16 + draw.textlength(case.name, font=f_title) + 12, y + 18),
              sub, font=f_small, fill=TEXT_DIM)
    # HUD (top-right, floating like an FPS counter)
    done = t >= case.wall
    tokps = case.tokps_at(t)
    anon = case.anon_at(t) if not done else case.peak_anon
    filemb = case.file_at(t) if not done else case.peak_file
    n_tok = int(round(case.progress_at(min(t, case.wall)) * case.n_gen))
    hud_x = x + w - 190
    draw.rounded_rectangle([hud_x, y + 10, x + w - 12, y + 108], radius=8,
                           fill=(10, 12, 16), outline=case.accent, width=1)
    draw.text((hud_x + 10, y + 14), f"{tokps:5.1f} tok/s", font=f_hud,
              fill=case.accent if not done else TEXT_DIM)
    draw.text((hud_x + 10, y + 42), f"MEM  {anon:7.1f} MB", font=f_mono,
              fill=WARN if anon > 100 else TEXT)
    draw.text((hud_x + 10, y + 64), f"PAGE {filemb:7.0f} MB", font=f_mono,
              fill=TEXT_DIM)
    draw.text((hud_x + 10, y + 86), f"TOK  {n_tok:3d}/{case.n_gen}",
              font=f_mono, fill=TEXT_DIM)
    # memory bar (anon) under HUD
    bar_y = y + h - 64
    draw.text((x + 16, bar_y - 22), "anonymous (wired) memory",
              font=f_small, fill=TEXT_DIM)
    # scale: log-ish bar, 512MB full-scale reference
    frac = min(1.0, math.log10(max(anon, 1.0)) / math.log10(1024.0))
    draw.rounded_rectangle([x + 16, bar_y, x + w - 16, bar_y + 14], radius=4,
                           fill=(34, 38, 46))
    if anon > 0:
        draw.rounded_rectangle([x + 16, bar_y, x + 16 + int((w - 32) * frac), bar_y + 14],
                                radius=4, fill=case.accent if anon < 100 else WARN)
    # text area
    txt = case.text_at(min(t, case.wall + 0.05))
    ty = y + 130
    lines = wrap(draw, txt, f_mono, w - 40)[:18]
    for i, ln in enumerate(lines):
        draw.text((x + 16, ty + i * 22), ln, font=f_mono, fill=TEXT)
    # status line
    st = "running..." if t < case.wall else f"done in {case.wall:.2f}s"
    draw.text((x + 16, y + h - 30), st, font=f_small,
              fill=TEXT_DIM if done else case.accent)


def draw_cannot_run_panel(draw, x, y, w, h, model_name, reasons, fnts):
    """Right-side panel: llama.cpp cannot run this model."""
    f_title, f_mono, f_hud, f_big, f_small = fnts
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=PANEL,
                            outline=PANEL_EDGE, width=2)
    draw.text((x + 16, y + 14), "llama.cpp", font=f_title, fill=LLAMA_ACCENT)
    draw.text((x + 16 + draw.textlength("llama.cpp", font=f_title) + 12, y + 18),
              model_name, font=f_small, fill=TEXT_DIM)
    # crossed-circle mark
    cx, cy, r = x + w // 2, y + h // 2 - 60, 44
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(239, 68, 68), width=6)
    draw.line([cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7],
              fill=(239, 68, 68), width=6)
    draw.text((x + w // 2 - draw.textlength("CANNOT RUN", font=f_big) // 2,
              cy + r + 18), "CANNOT RUN", font=f_big, fill=(239, 68, 68))
    ty = cy + r + 70
    for line in reasons:
        draw.text((x + w // 2 - draw.textlength(line, font=f_mono) // 2, ty),
                  line, font=f_mono, fill=TEXT_DIM)
        ty += 26


CANNOT_RUN_27B = [
    "27B mixed-quant GGUF: 11.98 GB weights",
    "Metal (default): weights resident in unified",
    "  memory — measured 20+ GB on this machine",
    "  (24 GB total) -> swap storm, machine stalls",
    "CPU (-ngl 0): still needs ~12 GB of weight",
    "  pages resident while you are using the machine",
    "",
    "stratum: streams the same file with ~77 MB",
    "anonymous memory. Weights are a stream,",
    "not a resident.",
]


def render(left, right, out_path, title, fps=FPS, right_cannot_run=None):
    lc = Case(left) if left else None
    rc = Case(right) if right else None
    right_wall = rc.wall if rc else (lc.wall if lc else 1.0)
    LEADIN = 1.5                      # title card before the race starts
    total = LEADIN + right_wall + 1.2
    frames = int(total * fps)
    fnts = (
        font(FONT_SANS, 26),
        font(FONT_MONO, 16),
        font(FONT_MONO, 20),
        font(FONT_MONO, 28),
        font(FONT_SANS, 14),
    )
    f_title, f_mono, f_hud, f_big, f_small = fnts
    f_card = font(FONT_SANS, 40)

    tmpdir = "/tmp/h2h_frames"
    os.makedirs(tmpdir, exist_ok=True)
    for i in range(frames):
        t = i / fps - LEADIN          # negative during the title card
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        if t < 0:
            d.text((W // 2 - d.textlength("stratum vs llama.cpp", font=f_card) // 2,
                    H // 2 - 60), "stratum vs llama.cpp", font=f_card, fill=TEXT)
            sub = "same GGUF · same prompt · greedy · CPU-only"
            d.text((W // 2 - d.textlength(sub, font=f_title) // 2, H // 2 + 10),
                   sub, font=f_title, fill=TEXT_DIM)
            d.text((W // 2 - d.textlength(title, font=f_small) // 2, H // 2 + 70),
                   title, font=f_small, fill=STRATUM_ACCENT)
        else:
            d.text((24, 14), title, font=f_title, fill=TEXT)
            if lc:
                ratio = f"anon ratio  {max(lc.peak_anon,0.1)/max(rc.peak_anon if rc else 1.0,0.1):.1f}x less RAM" if rc else \
                       f"anonymous memory  {lc.peak_anon:.0f} MB total"
                d.text((W - 24 - d.textlength(ratio, font=f_small), 22), ratio,
                       font=f_small, fill=TEXT_DIM)
                draw_case_panel(d, lc, 24, 60, 606, 640, t, fnts)
            if rc:
                draw_case_panel(d, rc, 650, 60, 606, 640, t, fnts)
            elif right_cannot_run:
                draw_cannot_run_panel(d, 650, 60, 606, 640,
                                      right_cannot_run.get("model", ""),
                                      right_cannot_run.get("reasons", CANNOT_RUN_27B),
                                      fnts)
            d.text((24, H - 24), f"t = {t:5.2f}s", font=f_small, fill=TEXT_DIM)
        img.save(f"{tmpdir}/f{i:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", f"{tmpdir}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", out_path], check=True)
    print(f"rendered {frames} frames -> {out_path}")


def main():
    left, right, out = sys.argv[1:4]
    title = "stratum vs llama.cpp — same GGUF, same prompt, greedy"
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]
    rcn = None
    if "--right-cannot-run" in sys.argv:
        rcn = {"model": "qwen3.6-27b-mixed.gguf (11.98 GB)",
               "reasons": CANNOT_RUN_27B}
    render(left if left != "-" else None, right if right != "-" else None,
           out, title, right_cannot_run=rcn)


if __name__ == "__main__":
    main()
