#!/usr/bin/env python3
"""make_tiny_model.py — generate a deterministic tiny GGUF for engine smoke tests.

The repo never ships weights (AGENTS.md); this script *generates* one at test
time from a fixed seed, so CI can run real inference (forward pass, KV cache,
sampler, logits dump) without downloading or committing a model. Output is
byte-identical across machines and CPython versions (random.Random is a
documented stable API).

Usage:
    python3 make_tiny_model.py --arch llama  --out /tmp/tiny.gguf
    python3 make_tiny_model.py --arch qwen35 --out /tmp/tiny35.gguf
    STRATUM_NO_GPU=1 ./stratum /tmp/tiny.gguf 4 1 2 3 4 5 6 7 8
"""
import argparse
import random
import struct

# Fixed geometry — small enough to build in milliseconds, big enough to
# exercise GQA attention, RoPE, FFN, and the F16 dequant matmul path.
N_LAYERS = 4
H = 64            # embedding_length
NQ = 4            # query heads
NK = 2            # kv heads (GQA)
HD = 16           # head_dim
FF = 176          # feed_forward_length
V = 256           # vocab

GGML_F32 = 0
GGML_F16 = 1


def build_entries(arch, rng):
    def mat(k, n):
        scale = 1.0 / (k ** 0.5)
        return b''.join(struct.pack('<e', rng.gauss(0.0, scale)) for _ in range(k * n))

    def norm_vec(n):
        return b''.join(struct.pack('<f', 0.5 + rng.random() * 0.5) for _ in range(n))

    entries = []  # (name, dims, ggml_type, data)
    for li in range(N_LAYERS):
        entries.append((f'blk.{li}.attn_norm.weight', (H,), GGML_F32, norm_vec(H)))
        if arch == 'qwen35':
            # qwen35 gated attention: attn_q emits (q, gate) per head, and the
            # layer norm before FFN is post_attention_norm (no ffn_norm).
            entries.append((f'blk.{li}.attn_q.weight', (H, 2 * NQ * HD), GGML_F16, mat(H, 2 * NQ * HD)))
            entries.append((f'blk.{li}.attn_q_norm.weight', (HD,), GGML_F32, norm_vec(HD)))
            entries.append((f'blk.{li}.attn_k_norm.weight', (HD,), GGML_F32, norm_vec(HD)))
            entries.append((f'blk.{li}.post_attention_norm.weight', (H,), GGML_F32, norm_vec(H)))
        else:
            entries.append((f'blk.{li}.attn_q.weight', (H, NQ * HD), GGML_F16, mat(H, NQ * HD)))
        # keep the llama RNG consumption order identical to the original
        # generator (PR #15 pinned its sequence): attn_norm, attn_q, attn_k,
        # attn_v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down
        entries.append((f'blk.{li}.attn_k.weight', (H, NK * HD), GGML_F16, mat(H, NK * HD)))
        entries.append((f'blk.{li}.attn_v.weight', (H, NK * HD), GGML_F16, mat(H, NK * HD)))
        entries.append((f'blk.{li}.attn_output.weight', (NQ * HD, H), GGML_F16, mat(NQ * HD, H)))
        if arch != 'qwen35':
            entries.append((f'blk.{li}.ffn_norm.weight', (H,), GGML_F32, norm_vec(H)))
        entries.append((f'blk.{li}.ffn_gate.weight', (H, FF), GGML_F16, mat(H, FF)))
        entries.append((f'blk.{li}.ffn_up.weight', (H, FF), GGML_F16, mat(H, FF)))
        entries.append((f'blk.{li}.ffn_down.weight', (FF, H), GGML_F16, mat(FF, H)))
    entries.append(('token_embd.weight', (H, V), GGML_F16, mat(H, V)))
    entries.append(('output_norm.weight', (H,), GGML_F32, norm_vec(H)))
    entries.append(('output.weight', (H, V), GGML_F16, mat(H, V)))
    return entries


def kv_pairs(arch):
    p = arch
    kvs = [kv_pair('general.architecture', p),
           kv_pair(f'{p}.block_count', N_LAYERS),
           kv_pair(f'{p}.embedding_length', H),
           kv_pair(f'{p}.feed_forward_length', FF),
           kv_pair(f'{p}.attention.head_count', NQ),
           kv_pair(f'{p}.attention.head_count_kv', NK),
           kv_pair(f'{p}.attention.key_length', HD),
           kv_pair(f'{p}.attention.layer_norm_rms_epsilon',
                   struct.unpack('<I', struct.pack('<f', 1e-5))[0]),
           kv_pair(f'{p}.rope.freq_base', 10000),
           kv_pair(f'{p}.rope.dimension_count', HD)]
    if arch == 'qwen35':
        # every layer full-attention: no SSM state, no MTP
        kvs.append(kv_pair(f'{p}.full_attention_interval', 1))
    kvs.append(kv_pair('general.alignment', 32))
    return b''.join(kvs)


def kv_pair(key, val):
    if isinstance(val, str):
        vb = struct.pack('<Q', len(val)) + val.encode()
        vt = 8
    else:
        vb = struct.pack('<I', val)
        vt = 4
    return struct.pack('<Q', len(key)) + key.encode() + struct.pack('<I', vt) + vb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='llama', choices=['llama', 'qwen35'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20260821)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    entries = build_entries(args.arch, rng)
    kvs = kv_pairs(args.arch)
    n_kv = 12 if args.arch == 'qwen35' else 11

    header = b'GGUF' + struct.pack('<IQQ', 3, len(entries), n_kv)

    def record(name, dims, ty, off):
        r = struct.pack('<Q', len(name)) + name.encode()
        r += struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims)
        return r + struct.pack('<I', ty) + struct.pack('<Q', off)

    index_len = len(header) + len(kvs)
    for name, dims, ty, data in entries:
        index_len += 8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8
    body_off = (index_len + 31) & ~31

    out = bytearray(header + kvs)
    off = 0
    for name, dims, ty, data in entries:
        out += record(name, dims, ty, off)
        off += len(data)

    out += b'\0' * (body_off - len(out))
    for _, _, _, data in entries:
        out += data

    with open(args.out, 'wb') as f:
        f.write(out)
    print(f'wrote {args.out}: arch={args.arch} layers={N_LAYERS} H={H} V={V} '
          f'size={len(out)} bytes')


if __name__ == '__main__':
    main()
