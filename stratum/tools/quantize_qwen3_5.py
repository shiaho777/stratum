#!/usr/bin/env python3
"""
Stratum — offline quantizer for Qwen3.5 text-only.

Strategy (Phase 3a):
  - Embedding (`embed_tokens.weight`) -> INT4 per-row
  - All Linear weights in language_model.layers.*  -> INT4 per-row
  - Norms, conv1d, A_log, dt_bias, in_proj_a, in_proj_b -> kept bf16
  - MTP and visual.* tensors not in the text path are *dropped* entirely
    (the runtime never reads them)

Output:
  <out_dir>/model.safetensors
  <out_dir>/model.safetensors.index.json
  <out_dir>/stratum_quant.json
  + copied config.json, tokenizer.json, etc.

Usage:
  python tools/quantize_qwen3_5.py --src ../model --dst ../model-int4
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from stratum.backends import MmapBackend
from stratum.quantization import (
    quantize_int4_per_row, quantize_int8_per_row,
)
from stratum.quantization.format import (
    QuantManifest, QuantTensorMeta, write_manifest,
)


def pick_scheme(name: str, default_linear: str = "int4") -> str:
    if name.startswith("model.visual."):  return "DROP"
    if name.startswith("mtp."):           return "DROP"

    if "norm" in name and "weight" in name:               return "bf16"
    if "norm" in name and "bias"   in name:               return "bf16"
    if "conv1d.weight" in name:                            return "bf16"
    if name.endswith(".A_log") or name.endswith(".dt_bias"):
        return "bf16"
    if "linear_attn.in_proj_a.weight" in name:             return "bf16"
    if "linear_attn.in_proj_b.weight" in name:             return "bf16"

    if "embed_tokens.weight" in name:                      return default_linear

    if name.endswith(".weight"):
        return default_linear

    return "bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source model dir (bf16 safetensors)")
    ap.add_argument("--dst", required=True, help="destination dir for quantized output")
    ap.add_argument("--scheme", default="int4", choices=["int4", "int8"],
                    help="default scheme for streamable Linear + embedding")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"== Stratum quantizer ==")
    print(f"src    : {src}")
    print(f"dst    : {dst}")
    print(f"scheme : {args.scheme} (where applicable)")

    backend = MmapBackend(src)
    print(f"loaded source index: {len(backend.all_names())} tensors, "
          f"{backend.total_bytes()/1e9:.2f} GB")

    manifest = QuantManifest(notes=f"Phase 3a quantization, default={args.scheme}")
    out_tensors: dict[str, torch.Tensor] = {}
    total_in_bytes = 0
    total_out_bytes = 0
    n_kept_bf16 = n_int8 = n_int4 = n_dropped = 0
    t0 = time.perf_counter()

    for name in backend.all_names():
        scheme = pick_scheme(name, default_linear=args.scheme)
        if scheme == "DROP":
            n_dropped += 1
            continue

        meta = backend.metadata(name)
        total_in_bytes += meta.nbytes

        W = backend.view(name).clone()

        if scheme == "bf16" or W.dim() < 2:
            out_tensors[name] = W
            qmeta = QuantTensorMeta(
                name=name, scheme="bf16",
                orig_shape=list(W.shape), orig_dtype="BF16",
                last_dim=W.shape[-1] if W.dim() > 0 else 0,
                has_scale=False,
            )
            manifest.tensors[name] = qmeta
            n_kept_bf16 += 1
            total_out_bytes += W.numel() * W.element_size()
            continue

        if scheme == "int8":
            q, s = quantize_int8_per_row(W)
            out_tensors[name]              = q
            out_tensors[f"{name}.scale"]   = s
            qmeta = QuantTensorMeta(
                name=name, scheme="int8",
                orig_shape=list(W.shape), orig_dtype="BF16",
                last_dim=W.shape[-1], has_scale=True,
            )
            manifest.tensors[name] = qmeta
            n_int8 += 1
            total_out_bytes += q.numel() + s.numel() * 2
            continue

        if scheme == "int4":
            qp, sp, cols = quantize_int4_per_row(W)
            out_tensors[name]            = qp
            out_tensors[f"{name}.scale"] = sp
            qmeta = QuantTensorMeta(
                name=name, scheme="int4",
                orig_shape=list(W.shape), orig_dtype="BF16",
                last_dim=cols, has_scale=True,
            )
            manifest.tensors[name] = qmeta
            n_int4 += 1
            total_out_bytes += qp.numel() + sp.numel() * 2
            continue

    print(f"\nQuantization done in {time.perf_counter()-t0:.1f}s")
    print(f"  kept bf16: {n_kept_bf16}")
    print(f"  int8     : {n_int8}")
    print(f"  int4     : {n_int4}")
    print(f"  dropped  : {n_dropped}")
    print(f"  bytes in : {total_in_bytes/1e9:.3f} GB")
    print(f"  bytes out: {total_out_bytes/1e9:.3f} GB  "
          f"({total_out_bytes/total_in_bytes*100:.1f}% of in)")

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
    print(f"  manifest: {dst / 'stratum_quant.json'}")

    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja",
              "preprocessor_config.json"]:
        src_f = src / f
        if src_f.exists():
            shutil.copy2(src_f, dst / f)

    print(f"\n✅ wrote quantized model to {dst}")


if __name__ == "__main__":
    main()
