#!/usr/bin/env python3
"""make_tiny_model.py — generate a deterministic tiny GGUF for engine smoke tests.

The repo never ships weights (AGENTS.md); this script *generates* one at test
time from a fixed seed, so CI can run real inference (forward pass, KV cache,
sampler, logits dump) without downloading or committing a model. Output is
byte-identical across machines and CPython versions (random.Random is a
documented stable API).

Usage:
    python3 make_tiny_model.py --arch llama --out /tmp/tiny.gguf
    python3 make_tiny_model.py --arch qwen35 --out /tmp/tiny35.gguf
    python3 make_tiny_model.py --arch qwen35-hybrid --out /tmp/tinyh.gguf
    python3 make_tiny_model.py --arch llama --weights q4k --out /tmp/tinyq.gguf

--weights q4k emits real Q4_K blocks (engine layout, written against
stratum_q4k.h's reader) so the model routes through the quantized GEMV
kernels — what `./verify_backends.sh <model>` needs to exercise the
Metal/NC/GPU2 paths. Q4_K requires K % 256 == 0, so the q4k geometry is
larger (H=256); hybrid is not supported yet (ssm_out has K=128).
"""
import argparse
import random
import struct

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_K = 12

# Base geometry — small enough to build in milliseconds, big enough to
# exercise GQA attention, RoPE, FFN, and the F16 dequant matmul path.
BASE = dict(N_LAYERS=4, H=64, NQ=4, NK=2, HD=16, FF=176, V=256)
# Q4_K needs every matmul input dim (K) to be a multiple of 256.
Q4K = dict(N_LAYERS=2, H=256, NQ=8, NK=2, HD=32, FF=512, V=1024)

# Hybrid (Gated DeltaNet) SSM geometry — scaled-down from the real 27B
# (HV=128, NK=16, NV=48, conv_dim=10240) by the same factor as H.
SSM_HV = 16          # state_size
SSM_NK = 4           # group_count
SSM_NV = 8           # time_step_rank
SSM_KEY_DIM = SSM_NK * SSM_HV          # 64
SSM_INNER = SSM_NV * SSM_HV            # 128
SSM_CONV_DIM = 2 * SSM_KEY_DIM + SSM_INNER   # 256
SSM_CONV_K = 4
FULL_ATTN_INTERVAL = 4   # layers where (i+1)%4==0 are full-attn, rest SSM

# Mini-DiT probe geometry (epic #35 Phase-1 spike): packed token sequence on
# a tiny (t,h,w) grid for MM-RoPE, timestep-conditioned blocks (AdaLN-lite).
DIT = dict(S=32, T=2, HG=4, WG=4,   # 32 tokens on a 2x4x4 grid
           H=64, HEADS=4, HD=16, FF=176, NL=2)


def q4k_encode_mat(vals, k, n):
    """Encode a [k, n] row-major f32 matrix as Q4_K blocks.

    Layout matches stratum_q4k.h exactly: per 256-element block,
    d(f16) dmin(f16) scales[12] qs[128]; sub-block g of 32 keeps 6-bit sc/m;
    qs packs two sub-blocks per 32 bytes (low nibble = first sub-block,
    high nibble = second); dequant: w = d*sc*q - dmin*m.
    """
    assert k % 256 == 0, 'Q4_K requires k % 256 == 0'
    out = bytearray()
    nb = k // 256
    for r in range(n):
        row = vals[r * k:(r + 1) * k]
        for b in range(nb):
            blk = row[b * 256:(b + 1) * 256]
            s_g, off_g = [], []
            for g in range(8):
                grp = blk[g * 32:(g + 1) * 32]
                mn, mx = min(grp), max(grp)
                if mn >= 0:
                    s_g.append((mx / 15.0) if mx > 0 else 1.0)
                    off_g.append(0.0)
                else:
                    s_g.append((mx - mn) / 15.0)
                    off_g.append(-mn)
            d = max(max(s_g) / 63.0, 1e-8)
            dmin = max(max(off_g) / 63.0, 0.0)
            sc = [max(1, min(63, int(round(s / d)))) for s in s_g]
            ms = [min(63, int(round(o / dmin))) if dmin > 0 else 0 for o in off_g]

            scales = [0] * 12
            qs = bytearray(128)
            for gi in range(8):
                seff = d * sc[gi]
                oeff = dmin * ms[gi]
                grp = blk[gi * 32:(gi + 1) * 32]
                qv = [max(0, min(15, int(round((x + oeff) / seff)))) if seff > 0 else 0
                      for x in grp]
                if gi < 4:
                    scales[gi] = sc[gi] & 63                    # sc: 6 bits
                    scales[gi + 4] |= ms[gi] & 63               # m:  bits 0-5
                else:
                    scales[gi + 4] |= sc[gi] & 0x0F             # sc: bits 0-3
                    scales[gi - 4] |= ((sc[gi] >> 4) & 3) << 6  # sc: bits 4-5
                    scales[gi + 4] |= (ms[gi] & 0x0F) << 4      # m:  bits 0-3
                    scales[gi] |= ((ms[gi] >> 4) & 3) << 6      # m:  bits 4-5
                qb = (gi // 2) * 32
                if gi % 2 == 0:
                    for l in range(32):
                        qs[qb + l] |= qv[l]
                else:
                    for l in range(32):
                        qs[qb + l] |= qv[l] << 4
            out += struct.pack('<e', d)
            out += struct.pack('<e', dmin)
            out += bytes(scales)
            out += bytes(qs)
    return bytes(out)


def build_entries(arch, weights, rng):
    G = Q4K if weights == 'q4k' else BASE
    NL, H, NQ, NK, HD, FF, V = (G[k] for k in
                                ('N_LAYERS', 'H', 'NQ', 'NK', 'HD', 'FF', 'V'))
    wt = GGML_Q4_K if weights == 'q4k' else GGML_F16

    def mat(k, n):
        scale = 1.0 / (k ** 0.5)
        vals = [rng.gauss(0.0, scale) for _ in range(k * n)]
        if weights == 'q4k':
            return wt, q4k_encode_mat(vals, k, n)
        return GGML_F16, b''.join(struct.pack('<e', v) for v in vals)

    def norm_vec(n):
        return GGML_F32, b''.join(struct.pack('<f', 0.5 + rng.random() * 0.5)
                                  for _ in range(n))

    entries = []  # (name, dims, ggml_type, data)
    if arch == 'dit':
        G = DIT
        Hb, FFb, NLb = G['H'], G['FF'], G['NL']

        def norm_vec(n):
            return b''.join(struct.pack('<f', 0.5 + rng.random() * 0.5)
                            for _ in range(n))

        def lin(k, n):
            scale = 1.0 / (k ** 0.5)
            return b''.join(struct.pack('<e', rng.gauss(0.0, scale))
                            for _ in range(k * n))

        for li in range(NLb):
            entries.append((f'blk.{li}.attn_norm.weight', (Hb,), GGML_F32,
                            norm_vec(Hb)))
            entries.append((f'blk.{li}.ada_shift.weight', (Hb, Hb), GGML_F16,
                            lin(Hb, Hb)))
            entries.append((f'blk.{li}.ada_gate.weight', (Hb, Hb), GGML_F16,
                            lin(Hb, Hb)))
            entries.append((f'blk.{li}.attn_q.weight', (Hb, Hb), GGML_F16,
                            lin(Hb, Hb)))
            entries.append((f'blk.{li}.attn_k.weight', (Hb, Hb), GGML_F16,
                            lin(Hb, Hb)))
            entries.append((f'blk.{li}.attn_v.weight', (Hb, Hb), GGML_F16,
                            lin(Hb, Hb)))
            entries.append((f'blk.{li}.attn_output.weight', (Hb, Hb), GGML_F16,
                            lin(Hb, Hb)))
            entries.append((f'blk.{li}.mlp_norm.weight', (Hb,), GGML_F32,
                            norm_vec(Hb)))
            entries.append((f'blk.{li}.mlp_ada_shift.weight', (Hb, Hb),
                            GGML_F16, lin(Hb, Hb)))
            entries.append((f'blk.{li}.mlp_ada_gate.weight', (Hb, Hb),
                            GGML_F16, lin(Hb, Hb)))
            entries.append((f'blk.{li}.ffn_gate.weight', (Hb, FFb), GGML_F16,
                            lin(Hb, FFb)))
            entries.append((f'blk.{li}.ffn_up.weight', (Hb, FFb), GGML_F16,
                            lin(Hb, FFb)))
            entries.append((f'blk.{li}.ffn_down.weight', (FFb, Hb), GGML_F16,
                            lin(FFb, Hb)))
        entries.append(('final_norm.weight', (Hb,), GGML_F32, norm_vec(Hb)))
        entries.append(('output_head.weight', (Hb, Hb), GGML_F16, lin(Hb, Hb)))
        return entries

    for li in range(NL):
        entries.append((f'blk.{li}.attn_norm.weight', (H,), *norm_vec(H)))
        if arch == 'qwen35-hybrid' and (li + 1) % FULL_ATTN_INTERVAL != 0:
            entries.append((f'blk.{li}.post_attention_norm.weight', (H,),
                            *norm_vec(H)))
            # Gated DeltaNet SSM layer — tensor names/shapes mirror the real
            # model's blk.0 (see issue #21): fused attn_qkv over conv_dim,
            # gate over inner, decay/beta per time_step_rank, conv1d weight
            # [kernel, conv_dim], out [inner, H].
            entries.append((f'blk.{li}.attn_qkv.weight', (H, SSM_CONV_DIM),
                            *mat(H, SSM_CONV_DIM)))
            entries.append((f'blk.{li}.attn_gate.weight', (H, SSM_INNER),
                            *mat(H, SSM_INNER)))
            entries.append((f'blk.{li}.ssm_a', (SSM_NV,), GGML_F32,
                            b''.join(struct.pack('<f', 0.5 + rng.random())
                                     for _ in range(SSM_NV))))
            entries.append((f'blk.{li}.ssm_alpha.weight', (H, SSM_NV),
                            *mat(H, SSM_NV)))
            entries.append((f'blk.{li}.ssm_beta.weight', (H, SSM_NV),
                            *mat(H, SSM_NV)))
            entries.append((f'blk.{li}.ssm_conv1d.weight',
                            (SSM_CONV_K, SSM_CONV_DIM), GGML_F32,
                            b''.join(struct.pack('<f', rng.gauss(0.0, 0.5))
                                     for _ in range(SSM_CONV_K * SSM_CONV_DIM))))
            entries.append((f'blk.{li}.ssm_dt.bias', (SSM_NV,), GGML_F32,
                            b''.join(struct.pack('<f', rng.gauss(0.0, 0.25))
                                     for _ in range(SSM_NV))))
            entries.append((f'blk.{li}.ssm_norm.weight', (SSM_HV,),
                            *norm_vec(SSM_HV)))
            entries.append((f'blk.{li}.ssm_out.weight', (SSM_INNER, H),
                            *mat(SSM_INNER, H)))
        elif arch in ('qwen35', 'qwen35-hybrid'):
            # full-attention layer (both qwen35 variants): gated attention,
            # attn_q emits (q, gate) per head; FFN norm is post_attention_norm
            entries.append((f'blk.{li}.attn_q.weight', (H, 2 * NQ * HD),
                            *mat(H, 2 * NQ * HD)))
            entries.append((f'blk.{li}.attn_q_norm.weight', (HD,),
                            *norm_vec(HD)))
            entries.append((f'blk.{li}.attn_k_norm.weight', (HD,),
                            *norm_vec(HD)))
            entries.append((f'blk.{li}.post_attention_norm.weight', (H,),
                            *norm_vec(H)))
        else:
            entries.append((f'blk.{li}.attn_q.weight', (H, NQ * HD),
                            *mat(H, NQ * HD)))
        # keep the llama RNG consumption order identical to the original
        # generator (PR #15 pinned its sequence): attn_norm, attn_q, attn_k,
        # attn_v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down
        entries.append((f'blk.{li}.attn_k.weight', (H, NK * HD),
                        *mat(H, NK * HD)))
        entries.append((f'blk.{li}.attn_v.weight', (H, NK * HD),
                        *mat(H, NK * HD)))
        entries.append((f'blk.{li}.attn_output.weight', (NQ * HD, H),
                        *mat(NQ * HD, H)))
        if arch == 'llama':
            entries.append((f'blk.{li}.ffn_norm.weight', (H,), *norm_vec(H)))
        entries.append((f'blk.{li}.ffn_gate.weight', (H, FF), *mat(H, FF)))
        entries.append((f'blk.{li}.ffn_up.weight', (H, FF), *mat(H, FF)))
        entries.append((f'blk.{li}.ffn_down.weight', (FF, H), *mat(FF, H)))
    entries.append(('token_embd.weight', (H, V), *mat(H, V)))
    entries.append(('output_norm.weight', (H,), *norm_vec(H)))
    entries.append(('output.weight', (H, V), *mat(H, V)))
    return entries


def kv_pair(key, val):
    if isinstance(val, str):
        vb = struct.pack('<Q', len(val)) + val.encode()
        vt = 8
    elif isinstance(val, float):
        vb = struct.pack('<f', val)
        vt = 6
    else:
        vb = struct.pack('<I', val)
        vt = 4
    return struct.pack('<Q', len(key)) + key.encode() + struct.pack('<I', vt) + vb


def kv_pairs(arch, weights):
    if arch == 'dit':
        G = DIT
        kvs = [
            kv_pair('general.architecture', 'dit-probe'),
            kv_pair('dit.sequence_length', G['S']),
            kv_pair('dit.grid_t', G['T']),
            kv_pair('dit.grid_h', G['HG']),
            kv_pair('dit.grid_w', G['WG']),
            kv_pair('dit.embedding_length', G['H']),
            kv_pair('dit.attention.head_count', G['HEADS']),
            kv_pair('dit.attention.key_length', G['HD']),
            kv_pair('dit.block_count', G['NL']),
            kv_pair('dit.feed_forward_length', G['FF']),
            kv_pair('dit.rope.freq_base', 10000.0),
            kv_pair('general.alignment', 32),
        ]
        return b''.join(kvs), len(kvs)

    # both qwen35 variants share the registered arch string "qwen35";
    # hybrid-ness comes from full_attention_interval + ssm.* metadata
    p = 'llama' if arch == 'llama' else 'qwen35'
    G = Q4K if weights == 'q4k' else BASE
    kvs = [
        kv_pair('general.architecture', p),
        kv_pair(f'{p}.block_count', G['N_LAYERS']),
        kv_pair(f'{p}.embedding_length', G['H']),
        kv_pair(f'{p}.feed_forward_length', G['FF']),
        kv_pair(f'{p}.attention.head_count', G['NQ']),
        kv_pair(f'{p}.attention.head_count_kv', G['NK']),
        kv_pair(f'{p}.attention.key_length', G['HD']),
        kv_pair(f'{p}.attention.layer_norm_rms_epsilon', 1e-5),
        kv_pair(f'{p}.rope.freq_base', 10000.0),
        kv_pair(f'{p}.rope.dimension_count', G['HD']),
    ]
    if arch == 'qwen35':
        kvs.append(kv_pair(f'{p}.full_attention_interval', 1))
    if arch == 'qwen35-hybrid':
        kvs += [
            kv_pair(f'{p}.full_attention_interval', FULL_ATTN_INTERVAL),
            kv_pair(f'{p}.ssm.state_size', SSM_HV),
            kv_pair(f'{p}.ssm.group_count', SSM_NK),
            kv_pair(f'{p}.ssm.time_step_rank', SSM_NV),
            kv_pair(f'{p}.ssm.inner_size', SSM_INNER),
            kv_pair(f'{p}.ssm.conv_kernel', SSM_CONV_K),
        ]
    kvs.append(kv_pair('general.alignment', 32))
    return (b''.join(kvs), (17 if arch == 'qwen35-hybrid' else
                            (12 if arch == 'qwen35' else 11)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='llama',
                    choices=['llama', 'qwen35', 'qwen35-hybrid', 'dit'])
    ap.add_argument('--weights', default='f16', choices=['f16', 'q4k'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20260821)
    args = ap.parse_args()
    if args.weights == 'q4k' and args.arch == 'qwen35-hybrid':
        raise SystemExit('q4k weights not supported for hybrid yet '
                         '(ssm_out K=128 is not a multiple of 256)')

    rng = random.Random(args.seed)
    entries = build_entries(args.arch, args.weights, rng)
    kvs, n_kv = kv_pairs(args.arch, args.weights)

    header = b'GGUF' + struct.pack('<IQQ', 3, len(entries), n_kv)

    def record(name, dims, ty, off):
        r = struct.pack('<Q', len(name)) + name.encode()
        r += struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d)
                                                     for d in dims)
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
    print(f'wrote {args.out}: arch={args.arch} weights={args.weights} '
          f'size={len(out)} bytes')


if __name__ == '__main__':
    main()
