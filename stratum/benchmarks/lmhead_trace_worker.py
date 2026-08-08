#!/usr/bin/env python3
"""Find the +128 MB inside lm_head with cycle-by-cycle RSS tracing."""
import os, subprocess
from pathlib import Path
import torch
from transformers import AutoTokenizer
from stratum.adapters.qwen3_5 import load_qwen3_5_text
from stratum.modules.linear import StratumLinear


def rss():
    return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())]).strip()) / 1024


orig_blocked = StratumLinear._forward_blocked
def traced_blocked(self, x):
    from stratum.quantization import dequantize_int4_per_row, dequantize_int8_per_row
    self._ensure_packed()
    print(f"  >>> lm_head input: {tuple(x.shape)} {x.dtype}, RSS={rss():.1f}")

    vocab = self._packed.shape[0]
    *batch_shape, hidden = x.shape
    if x.dtype != torch.bfloat16:
        x_flat = x.reshape(-1, hidden).to(torch.bfloat16)
    else:
        x_flat = x.reshape(-1, hidden)
    n = x_flat.shape[0]

    print(f"      x_flat n={n} hidden={hidden}, RSS={rss():.1f}")
    out = torch.empty((n, vocab), dtype=torch.bfloat16, device=x.device)
    print(f"      out alloc ({n}x{vocab}) = {n*vocab*2/1e6:.2f} MB, RSS={rss():.1f}")

    block_rows = self._block_rows
    iters = 0
    for start in range(0, vocab, block_rows):
        end = min(start + block_rows, vocab)
        packed_block = self._packed[start:end]
        scale_block = self._scale[start:end]
        if self._qm.scheme == "int4":
            W_block = dequantize_int4_per_row(
                packed_block.unsqueeze(0), scale_block.unsqueeze(0),
                last_dim=self._qm.last_dim, out_dtype=torch.bfloat16,
                high_precision=False,
            ).squeeze(0)
        else:
            W_block = dequantize_int8_per_row(
                packed_block.unsqueeze(0), scale_block.unsqueeze(0),
                out_dtype=torch.bfloat16, high_precision=False,
            ).squeeze(0)
        if iters in (0, 1, 5, 60, 121):
            print(f"      iter {iters}: dequant W_block {tuple(W_block.shape)} = "
                  f"{W_block.numel()*2/1e6:.2f} MB, RSS={rss():.1f}")
        out[:, start:end] = x_flat @ W_block.t()
        if iters in (0, 1, 5, 60, 121):
            print(f"      iter {iters}: matmul done, RSS={rss():.1f}")
        del W_block
        iters += 1
    print(f"      done {iters} iters, final RSS={rss():.1f}")
    return out.reshape(*batch_shape, vocab)

StratumLinear._forward_blocked = traced_blocked

md = (Path.cwd().parent / "model-int4-awq").resolve()
tok = AutoTokenizer.from_pretrained(md)
model, prov = load_qwen3_5_text(md, device="cpu", dtype=torch.bfloat16,
                                max_layers_in_ram=1, quantized=True)
print(f"after load: {rss():.1f}")
inputs = tok("Hello, my name is", return_tensors="pt")
with torch.no_grad():
    _ = model(**inputs, use_cache=True)
print(f"after forward: {rss():.1f}")
