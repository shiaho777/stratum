"""
V8: Low-rank weight decomposition for FFN down_proj.

Takes a safetensors model, decomposes each down_proj weight matrix
W [H, Ff] via SVD into U [H, r] @ V [r, Ff], and writes a new
safetensors with the factored weights.

For Qwen3.5-0.8B: H=1024, Ff=3584.
  rank 256: 1.18M params (32% of 3.67M dense)
  rank 512: 2.36M params (64% of dense)

At inference: y = x @ W^T  =>  y = x @ V^T @ U^T
  Two matmuls: x[Ff] @ V^T[r,Ff] -> [r], then [r] @ U^T[H,r] -> [H]
  Compute: r*Ff + r*H = r*(Ff+H) vs dense Ff*H
  For r=256: 256*4608 = 1.18M vs 3.67M (32% compute)

Usage:
  python lowrank_decompose.py --model-dir ../model --rank 256 --output-dir ../model-lr256
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def decompose_weight(W: np.ndarray, rank: int):
    """SVD decompose W [out, in] -> U [out, rank], V [rank, in].
    Returns (U, V, energy_captured)."""
    U, s, Vh = np.linalg.svd(W, full_matrices=False)
    total_energy = (s ** 2).sum()
    cum = np.cumsum(s ** 2) / total_energy
    r = min(rank, len(s))
    U_r = U[:, :r] * s[:r]  # fold singular values into U
    V_r = Vh[:r, :]
    energy = float(cum[r - 1])
    return U_r.astype(np.float32), V_r.astype(np.float32), energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="down_proj",
                        help="Which weights to decompose (substring match)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find safetensors files
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        idx = json.load(open(index_file))
        weight_map = idx["weight_map"]
        st_files = sorted(set(weight_map.values()))
    else:
        st_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]

    # Collect all tensor names
    all_tensors = {}
    for stf in st_files:
        with safe_open(model_dir / stf, framework="pt") as f:
            for key in f.keys():
                all_tensors[key] = stf

    # Find target weights
    targets = sorted([k for k in all_tensors if args.target in k and "mtp" not in k.lower()])
    print(f"# Found {len(targets)} '{args.target}' tensors to decompose")

    # Decompose and collect new tensors
    new_tensors = {}
    stats = []
    for stf in st_files:
        with safe_open(model_dir / stf, framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key in targets:
                    W = t.to(torch.float32).numpy()
                    U, V, energy = decompose_weight(W, args.rank)
                    new_tensors[f"{key}.U"] = torch.from_numpy(U)
                    new_tensors[f"{key}.V"] = torch.from_numpy(V)
                    stats.append({
                        "name": key,
                        "shape": list(W.shape),
                        "rank": args.rank,
                        "energy": round(energy, 4),
                        "orig_params": W.size,
                        "factored_params": U.size + V.size,
                        "ratio": round((U.size + V.size) / W.size, 3),
                    })
                    print(f"  {key}: rank={args.rank} energy={energy:.4f} "
                          f"ratio={((U.size+V.size)/W.size):.3f}")
                else:
                    new_tensors[key] = t

    # Save
    out_file = out_dir / "model.safetensors"
    save_file(new_tensors, str(out_file))
    print(f"\n# Saved {len(new_tensors)} tensors to {out_file}")

    # Copy config files
    import shutil
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja"]:
        src = model_dir / f
        if src.exists():
            shutil.copy2(src, out_dir / f)

    # Write decomposition manifest
    manifest = {"rank": args.rank, "target": args.target, "tensors": stats}
    with open(out_dir / "lowrank_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"# Manifest written to {out_dir / 'lowrank_manifest.json'}")

    # Summary
    orig = sum(s["orig_params"] for s in stats)
    factored = sum(s["factored_params"] for s in stats)
    print(f"\n# === SUMMARY ===")
    print(f"# {len(stats)} tensors, rank={args.rank}")
    print(f"# original params: {orig:,}")
    print(f"# factored params: {factored:,} ({factored/orig:.1%})")
    print(f"# IO reduction: {(1-factored/orig)*100:.1f}%")
    print(f"# median energy: {np.median([s['energy'] for s in stats]):.4f}")


if __name__ == "__main__":
    main()
