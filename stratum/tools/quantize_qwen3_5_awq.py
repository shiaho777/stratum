#!/usr/bin/env python3
"""
Stratum — AWQ-calibrated INT4 quantizer for Qwen3.5 text-only.

Pipeline:
  1) Run calibration: load bf16 model + push N WikiText-2 texts through it,
     capturing per-Linear input absmax + a sample of inputs per layer.
  2) For each Linear weight that we'd quantize:
       a) compute vanilla INT4 quant + measure output MSE on captured samples
       b) AWQ search alpha in [0, 1], pick best by output MSE
       c) if AWQ reduces MSE by >= MIN_GAIN, store with awq scale; else
          fall back to vanilla
  3) Write Stratum quant format with optional `awq_s` per tensor.

Embedding gets vanilla INT4 (no input statistics — embedding is a lookup,
not a matmul on activations).

Usage:
  python tools/quantize_qwen3_5_awq.py --src <model-dir> --dst <out-dir> \
        --num-calib 32 --max-tokens 256
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from stratum.backends import MmapBackend
from stratum.quantization import (
    quantize_int4_per_row, dequantize_int4_per_row,
)
from stratum.quantization.format import (
    QuantManifest, QuantTensorMeta, write_manifest,
)


MIN_AWQ_GAIN = 0.03
GRID = tuple(i / 20 for i in range(0, 21))
SAMPLE_CAP = 512


def pick_scheme(name: str) -> str:
    if name.startswith("model.visual."):  return "DROP"
    if name.startswith("mtp."):           return "DROP"
    if "norm" in name and ("weight" in name or "bias" in name): return "bf16"
    if "conv1d.weight" in name:          return "bf16"
    if name.endswith(".A_log") or name.endswith(".dt_bias"): return "bf16"
    if "linear_attn.in_proj_a.weight" in name: return "bf16"
    if "linear_attn.in_proj_b.weight" in name: return "bf16"
    if name.endswith(".weight"):         return "int4"
    return "bf16"


def ckpt_to_runtime_modulename(ckpt_w: str) -> str:
    if not ckpt_w.endswith(".weight"):
        return ckpt_w
    mod = ckpt_w[:-len(".weight")]
    if mod.startswith("model.language_model."):
        return "model." + mod[len("model.language_model."):]
    return mod


class CalibCapture:
    """Per-Linear hook that records absmax + a few input samples."""
    def __init__(self, sample_cap: int):
        self.absmax = None
        self.samples = []
        self.sample_cap = sample_cap
        self.token_count = 0

    def __call__(self, module, args):
        if not args:
            return
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        cur = x.abs().amax(dim=0)
        self.absmax = cur if self.absmax is None else torch.maximum(self.absmax, cur)
        if self.token_count < self.sample_cap:
            take = min(self.sample_cap - self.token_count, x.shape[0])
            self.samples.append(x[:take].cpu())
            self.token_count += take


def search_awq_alpha(W: torch.Tensor, A: torch.Tensor, X_sample: torch.Tensor):
    """Return (best_s, best_alpha, vanilla_mse, best_mse)."""
    W = W.float()
    A = A.float().clamp(min=1e-5)
    A_norm = A / A.mean()
    X = X_sample.float()
    Y_true = X @ W.t()

    q, scl, last_dim = quantize_int4_per_row(W)
    Wdq = dequantize_int4_per_row(q, scl, last_dim, out_dtype=torch.float32)
    vanilla_mse = (X @ Wdq.t() - Y_true).pow(2).mean().item()

    best_alpha = 0.0
    best_s    = torch.ones_like(A)
    best_mse  = vanilla_mse
    for alpha in GRID:
        if alpha == 0.0:
            continue
        s = A_norm.pow(alpha).clamp(min=1e-5)
        Ws = W * s
        q, scl, last_dim = quantize_int4_per_row(Ws)
        Wdq = dequantize_int4_per_row(q, scl, last_dim, out_dtype=torch.float32)
        Wrec = Wdq.float() / s
        mse = (X @ Wrec.t() - Y_true).pow(2).mean().item()
        if mse < best_mse:
            best_mse, best_alpha, best_s = mse, alpha, s.clone()
    return best_s, best_alpha, vanilla_mse, best_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--num-calib", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--awq-min-gain", type=float, default=MIN_AWQ_GAIN)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print("== Stratum AWQ quantizer ==")
    print(f"src     : {src}")
    print(f"dst     : {dst}")
    print(f"calib   : {args.num_calib} texts, up to {args.max_tokens} tokens each")
    print(f"min gain: {args.awq_min_gain*100:.0f}% MSE reduction to keep AWQ scale")

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\n[1/3] Loading calibration data ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 200][:args.num_calib]
    print(f"      using {len(texts)} non-trivial texts")

    print("[2/3] Loading bf16 model + capturing activations ...")
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()

    captures = {}
    handles  = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            cap = CalibCapture(sample_cap=SAMPLE_CAP)
            captures[name] = cap
            handles.append(mod.register_forward_pre_hook(cap))

    print(f"      hooked {len(captures)} Linear modules")
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
            model(**ids, use_cache=False)
            if (i+1) % 4 == 0 or i+1 == len(texts):
                print(f"      [{i+1}/{len(texts)}]")
    for h in handles: h.remove()
    print(f"      calibration done in {time.perf_counter()-t0:.1f}s")

    print("\n[3/3] Quantizing weights ...")
    backend = MmapBackend(src)
    manifest = QuantManifest(notes=f"AWQ-INT4, {len(texts)} calib texts, "
                                   f"min_gain={args.awq_min_gain}")
    out_tensors: dict[str, torch.Tensor] = {}
    n_kept = n_int4_vanilla = n_int4_awq = n_dropped = 0
    n_awq_tried = 0
    bytes_in = bytes_out = 0
    t0 = time.perf_counter()

    for name in backend.all_names():
        scheme = pick_scheme(name)
        if scheme == "DROP":
            n_dropped += 1
            continue

        meta = backend.metadata(name)
        bytes_in += meta.nbytes
        W = backend.view(name).clone()

        if scheme == "bf16" or W.dim() < 2:
            out_tensors[name] = W
            manifest.tensors[name] = QuantTensorMeta(
                name=name, scheme="bf16",
                orig_shape=list(W.shape), orig_dtype="BF16",
                last_dim=W.shape[-1] if W.dim() else 0, has_scale=False,
            )
            n_kept += 1
            bytes_out += W.numel() * W.element_size()
            continue

        runtime_mod = ckpt_to_runtime_modulename(name)
        cap = captures.get(runtime_mod)

        try_awq = (cap is not None and cap.absmax is not None and cap.samples)

        if try_awq:
            n_awq_tried += 1
            X_sample = torch.cat(cap.samples, dim=0)[:SAMPLE_CAP]
            best_s, best_alpha, van_mse, best_mse = search_awq_alpha(
                W, cap.absmax, X_sample,
            )
            gain = (van_mse - best_mse) / max(van_mse, 1e-12)
            if gain >= args.awq_min_gain and best_alpha > 0:
                Ws = W.float() * best_s
                qp, sp, cols = quantize_int4_per_row(Ws.to(W.dtype))
                out_tensors[name] = qp
                out_tensors[f"{name}.scale"] = sp
                out_tensors[f"{name}.awq_s"] = best_s.to(torch.float16)
                manifest.tensors[name] = QuantTensorMeta(
                    name=name, scheme="int4",
                    orig_shape=list(W.shape), orig_dtype="BF16",
                    last_dim=cols, has_scale=True,
                )
                n_int4_awq += 1
                bytes_out += qp.numel() + sp.numel() * 2 + best_s.numel() * 2
                if n_int4_awq <= 5 or n_int4_awq % 30 == 0:
                    print(f"      AWQ: {name}  alpha={best_alpha:.2f} "
                          f"gain={gain*100:.1f}%")
                continue

        qp, sp, cols = quantize_int4_per_row(W)
        out_tensors[name] = qp
        out_tensors[f"{name}.scale"] = sp
        manifest.tensors[name] = QuantTensorMeta(
            name=name, scheme="int4",
            orig_shape=list(W.shape), orig_dtype="BF16",
            last_dim=cols, has_scale=True,
        )
        n_int4_vanilla += 1
        bytes_out += qp.numel() + sp.numel() * 2

    dt = time.perf_counter() - t0
    print(f"\n  quantization loop: {dt:.1f}s")
    print(f"  kept bf16        : {n_kept}")
    print(f"  INT4 (AWQ)       : {n_int4_awq} / {n_awq_tried} tried "
          f"({100*n_int4_awq/max(n_awq_tried,1):.0f}% kept AWQ)")
    print(f"  INT4 (vanilla)   : {n_int4_vanilla}")
    print(f"  dropped          : {n_dropped}")
    print(f"  bytes in : {bytes_in/1e9:.3f} GB")
    print(f"  bytes out: {bytes_out/1e9:.3f} GB ({100*bytes_out/bytes_in:.1f}%)")

    print("\nSaving safetensors ...")
    from safetensors.torch import save_file
    out_tensors = {k: v.contiguous() for k, v in out_tensors.items()}
    save_file(out_tensors, str(dst / "model.safetensors"))
    index = {
        "metadata": {"total_size": sum(t.numel()*t.element_size() for t in out_tensors.values())},
        "weight_map": {k: "model.safetensors" for k in out_tensors.keys()},
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    write_manifest(dst / "stratum_quant.json", manifest)

    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja",
              "preprocessor_config.json"]:
        src_f = src / f
        if src_f.exists():
            shutil.copy2(src_f, dst / f)

    print(f"\n✅ wrote AWQ-quantized model to {dst}")


if __name__ == "__main__":
    main()
