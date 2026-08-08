#!/usr/bin/env python3
"""
Stratum P35 — quantize MTP block to INT4 (vanilla per-row).

The official transformers reference *ignores* the mtp.* keys, so there's
no calibration path for them. We do the simplest thing that works:

  - Each MTP linear weight: quantize_int4_per_row -> packed U8 + F16 scale
  - Norms / small vectors: keep bf16
  - Output a separate safetensors file at <dst> containing all MTP weights
    in the same naming/format the existing stratum_p* loaders use:
       <name>            : packed INT4 (U8) shape [out, in/2]
       <name>.scale      : per-row F16 scale shape [out]

That file mmap-loads cleanly as a second TensorIndex source for the MTP
forward path in stratum_p35.c.
"""

import argparse
import json
import struct
from pathlib import Path

import torch

from stratum.quantization import (
    quantize_int4_per_row,
    dequantize_int4_per_row,
)


MTP_INT4_KEYS = {
    "mtp.fc.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
}

MTP_BF16_KEYS = {
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def read_safetensors(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    hdr_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + hdr_len])
    body_off = 8 + hdr_len
    return header, raw[body_off:]


def get_tensor(header: dict, body: bytes, key: str) -> torch.Tensor:
    m = header[key]
    off, end = m["data_offsets"]
    raw = body[off:end]
    dtype_map = {
        "BF16": torch.bfloat16,
        "F16":  torch.float16,
        "F32":  torch.float32,
        "U8":   torch.uint8,
    }
    dtype = dtype_map[m["dtype"]]
    t = torch.frombuffer(bytearray(raw), dtype=dtype)
    return t.reshape(m["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="path to BF16 model.safetensors-* file")
    ap.add_argument("--dst", required=True,
                    help="path to output mtp.safetensors")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    header, body = read_safetensors(src)

    out_tensors: dict[str, tuple[str, list[int], bytes]] = {}

    for key in MTP_INT4_KEYS:
        if key not in header:
            print(f"WARN: missing {key}, skipping")
            continue
        W = get_tensor(header, body, key).to(torch.float32)
        q, scl, last_dim = quantize_int4_per_row(W)
        scl_f16 = scl.to(torch.float16)
        out_tensors[key] = (
            "U8", list(q.shape), q.contiguous().numpy().tobytes()
        )
        out_tensors[key + ".scale"] = (
            "F16", [int(scl.shape[0])], scl_f16.contiguous().numpy().tobytes()
        )
        Wdq = dequantize_int4_per_row(q, scl, last_dim, out_dtype=torch.float32)
        mse = (Wdq - W).pow(2).mean().item()
        print(f"  {key:<60s} shape={list(W.shape)} MSE={mse:.6e}")

    for key in MTP_BF16_KEYS:
        if key not in header:
            print(f"WARN: missing {key}, skipping")
            continue
        W = get_tensor(header, body, key)
        bf16_t = W.to(torch.bfloat16).contiguous()
        raw = bytearray(bf16_t.numel() * 2)
        out_tensors[key] = (
            "BF16", list(W.shape), bf16_t.view(torch.uint8).numpy().tobytes()
        )

    new_header = {}
    cursor = 0
    for name in sorted(out_tensors):
        dtype_str, shape, data = out_tensors[name]
        size = len(data)
        new_header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size

    new_header["__metadata__"] = {"format": "stratum-mtp-vanilla-int4"}

    hdr_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hdr_json)) % 8
    hdr_json += b" " * pad

    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_json)))
        f.write(hdr_json)
        for name in sorted(out_tensors):
            if name == "__metadata__":
                continue
            f.write(out_tensors[name][2])

    print(f"\nWrote {dst}")
    print(f"  total tensors: {len(out_tensors)}")
    print(f"  body bytes: {cursor}")


if __name__ == "__main__":
    main()
