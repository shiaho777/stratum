#!/usr/bin/env python3
"""hidden_oracle.py — independent reference for Stratum llama-arch hidden states.

Reads a llama-arch GGUF with F16/F32 tensors, runs the prompt (+ optional
greedy continuation) through an independent PyTorch re-implementation of the
engine's exact semantics, and dumps post-layer-L residual streams in the same
"SHID0001" format the engine writes under STRATUM_HIDDEN_DUMP=<layer>:<path>.

Phase-0 tooling for epic #35: two independent code paths (C NEON vs torch)
over identical weights must agree to within op-order noise.

Semantics mirrored from stratum_arch_llama.inc.c:
  - RMSNorm: double-accumulated mean-square, y = x * scale * gain
  - RoPE: interleaved pairs (x[2k], x[2k+1]), theta^(-2k/rope_dim)
  - GQA: kv_h = h * Nk / Nq; attention scale = 1/sqrt(Hd); causal
  - FFN: down(silu(gate) * up)
  - capture point: residual stream AFTER the FFN residual add of layer L

Usage:
    python3 hidden_oracle.py --gguf /tmp/tiny.gguf --ids 1 2 3 4 5 6 7 8 \
        --layer 1 --generate 4 --out /tmp/oracle.bin --compare /tmp/engine.bin
"""
import argparse
import mmap
import struct
import sys

import torch


def read_gguf(path):
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def u32(pos):
        return struct.unpack('<I', mm[pos:pos + 4])[0], pos + 4

    def u64(pos):
        return struct.unpack('<Q', mm[pos:pos + 8])[0], pos + 8

    assert mm[0:4] == b'GGUF'
    ver, pos = u32(4)
    assert ver == 3, ver
    n_tensors, pos = u64(pos)
    n_kv, pos = u64(pos)

    SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def rd_string(pos):
        n, pos = u64(pos)
        return mm[pos:pos + n].decode('utf-8', 'replace'), pos + n

    def skip_value(vt, pos):
        if vt == 8:
            n, pos = u64(pos)
            return pos + n
        if vt == 9:
            et, pos = u32(pos)
            cnt, pos = u64(pos)
            if et == 8:
                for _ in range(cnt):
                    n, pos = u64(pos)
                    pos += n
            else:
                pos += SIZES[et] * cnt
            return pos
        return pos + SIZES[vt]

    kv = {}
    for _ in range(n_kv):
        key, pos = rd_string(pos)
        vt, pos = u32(pos)
        start = pos
        pos = skip_value(vt, pos)
        kv[key] = (vt, mm[start:pos])

    align_vt, align_vb = kv.get('general.alignment', (4, struct.pack('<I', 32)))
    align = struct.unpack('<I', align_vb)[0]

    records = []
    for _ in range(n_tensors):
        name, pos = rd_string(pos)
        nd, pos = u32(pos)
        dims = []
        for _ in range(nd):
            d, pos = u64(pos)
            dims.append(d)
        ty, pos = u32(pos)
        off, pos = u64(pos)
        records.append((name, dims, ty, off))
    body_off = (pos + align - 1) & ~(align - 1)

    tensors = {name: (dims, ty, body_off + off)
               for name, dims, ty, off in records}

    cfg = {}
    for key, (vt, vb) in kv.items():
        if vt == 4:
            cfg[key] = struct.unpack('<I', vb)[0]
        elif vt == 6:
            cfg[key] = struct.unpack('<f', vb)[0]
        elif vt == 8:
            cfg[key] = vb.decode('utf-8', 'replace')

    def tensor(name):
        dims, ty, off = tensors[name]
        els = 1
        for d in dims:
            els *= d
        raw = mm[off:off + els * (2 if ty == 1 else 4)]
        if ty == 0:
            t = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        elif ty == 1:
            t = torch.frombuffer(bytearray(raw), dtype=torch.float16).clone().float()
        else:
            raise ValueError(f'unsupported type {ty} for {name}')
        if len(dims) == 1:
            return t.view(-1)
        return t.view(dims[1], dims[0])  # [out_rows, in_cols]

    return cfg, tensor


def make_forward(cfg, T, dump_layer):
    p = 'llama.'
    NL = cfg[f'{p}block_count']
    HD = cfg[f'{p}attention.key_length']
    eps = cfg.get(f'{p}attention.layer_norm_rms_epsilon', 1e-5)
    theta = float(cfg.get(f'{p}rope.freq_base', 10000.0))
    rope_dim = cfg.get(f'{p}rope.dimension_count', HD)

    def rmsnorm(x, gain):
        ss = 0.0
        for v in x.tolist():
            ss += float(v) * float(v)
        scale = 1.0 / (ss / x.shape[0] + eps) ** 0.5
        return x * scale * gain

    def rope(vec, position_):
        out = vec.clone()
        for k in range(rope_dim // 2):
            angle = float(position_) * (theta ** (-(2.0 * k) / rope_dim))
            c = torch.cos(torch.tensor(angle)).item()
            s = torch.sin(torch.tensor(angle)).item()
            x0, x1 = float(out[2 * k]), float(out[2 * k + 1])
            out[2 * k] = x0 * c - x1 * s
            out[2 * k + 1] = x0 * s + x1 * c
        return out

    NQ = cfg[f'{p}attention.head_count']
    NK = cfg[f'{p}attention.head_count_kv']
    HD = cfg[f'{p}attention.key_length']

    import os as _os
    dbg_on = _os.environ.get('STRATUM_BLOCK_DBG')

    def step(token_id, kv_len, K_cache, V_cache):
        """One token forward. Returns (x_out, captures, kv_len+1, caches)."""
        captures = []
        x = T('token_embd.weight')[token_id].clone()
        for li in range(NL):
            dbg = dbg_on and kv_len == 1
            resid = x.clone()
            xn = rmsnorm(x, T(f'blk.{li}.attn_norm.weight'))
            if dbg:
                print(f"[DBG] embed={x[0]:.6f} {x[1]:.6f} | xn={xn[0]:.6f} {xn[1]:.6f}")

            q = xn @ T(f'blk.{li}.attn_q.weight').T
            k = xn @ T(f'blk.{li}.attn_k.weight').T
            v = xn @ T(f'blk.{li}.attn_v.weight').T
            q = torch.stack([rope(q[h * HD:(h + 1) * HD], kv_len)
                             for h in range(NQ)]).flatten()
            k = torch.stack([rope(k[h * HD:(h + 1) * HD], kv_len)
                             for h in range(NK)]).flatten()
            if dbg:
                print(f"[DBG] q_rope[0..3]={q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                      f"k_rope[0..3]={k[0]:.6f} {k[1]:.6f} {k[2]:.6f} {k[3]:.6f}")
            kr = k.reshape(NK, HD).unsqueeze(0)
            vr = v.reshape(NK, HD).unsqueeze(0)
            K_cache[li] = kr if K_cache[li] is None else torch.cat([K_cache[li], kr], 0)
            V_cache[li] = vr if V_cache[li] is None else torch.cat([V_cache[li], vr], 0)

            attn = torch.zeros(NQ, HD)
            scale = 1.0 / (HD ** 0.5)
            for h in range(NQ):
                kv_h = h * NK // NQ
                qh = q[h * HD:(h + 1) * HD]
                scores = torch.zeros(kv_len + 1)
                for t in range(kv_len + 1):
                    scores[t] = torch.dot(K_cache[li][t, kv_h], qh).item() * scale
                probs = torch.softmax(scores, dim=0)
                attn[h] = probs @ V_cache[li][:, kv_h]
                if dbg and h == 0:
                    print(f"[DBG] Kcache t0[0..2]={K_cache[li][0,kv_h][0]:.6f} {K_cache[li][0,kv_h][1]:.6f} {K_cache[li][0,kv_h][2]:.6f}")
                    print(f"[DBG] Vcache t0[0..2]={V_cache[li][0,kv_h][0]:.6f}")
            proj = attn.flatten() @ T(f'blk.{li}.attn_output.weight').T
            x = resid + proj
            if dbg:
                print(f"[DBG] q0={q[0]:.6f} v00={v.reshape(NK, HD)[0,0]:.6f} "
                      f"attn00={attn[0,0]:.6f} proj0={proj[0]:.6f} post_attn0={x[0]:.6f}")

            resid = x.clone()
            xn = rmsnorm(x, T(f'blk.{li}.ffn_norm.weight'))
            g = xn @ T(f'blk.{li}.ffn_gate.weight').T
            u = xn @ T(f'blk.{li}.ffn_up.weight').T
            a = (g / (1.0 + torch.exp(-g))) * u
            ff = a @ T(f'blk.{li}.ffn_down.weight').T
            x = resid + ff
            if dbg:
                print(f"[DBG] L{li} post_ffn x[0]={x[0]:.6f}")

            # capture AFTER the FFN residual add — same point as the engine
            if li == dump_layer:
                captures.append(x.clone())
        H = x.shape[0]
        return x, captures, kv_len + 1, K_cache, V_cache, H

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--ids', nargs='+', type=int, required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--generate', type=int, default=0,
                    help='greedy tokens to continue after the prompt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare')
    args = ap.parse_args()

    cfg, T = read_gguf(args.gguf)
    p = 'llama.'
    H = cfg[f'{p}embedding_length']
    step = make_forward(cfg, T, args.layer)

    K_cache = [None] * cfg[f'{p}block_count']
    V_cache = [None] * cfg[f'{p}block_count']
    captures = []
    x, kv_len = None, 0

    for token_id in args.ids:
        x, caps, kv_len, K_cache, V_cache, _ = step(
            token_id, kv_len, K_cache, V_cache)
        captures.extend(caps)

    for _ in range(args.generate):
        xn = rmsnorm_local(x, T('output_norm.weight'),
                           cfg.get(f'{p}attention.layer_norm_rms_epsilon', 1e-5))
        next_id = int(torch.argmax(xn @ T('output.weight').T))
        x, caps, kv_len, K_cache, V_cache, _ = step(
            next_id, kv_len, K_cache, V_cache)
        captures.extend(caps)

    with open(args.out, 'wb') as f:
        f.write(b'SHID0001')
        f.write(struct.pack('<II', args.layer, H))
        for cap in captures:
            f.write(cap.numpy().astype('<f4').tobytes())
    print(f'wrote {args.out}: layer={args.layer} tokens={len(captures)} H={H}')

    if args.compare:
        ref = open(args.compare, 'rb').read()
        assert ref[:8] == b'SHID0001'
        r_layer, r_H = struct.unpack('<II', ref[8:16])
        mine = open(args.out, 'rb').read()
        m_layer, m_H = struct.unpack('<II', mine[8:16])
        assert (r_layer, r_H) == (m_layer, m_H), (r_layer, r_H, m_layer, m_H)
        a = torch.frombuffer(bytearray(ref[16:]), dtype=torch.float32)
        b = torch.frombuffer(bytearray(mine[16:]), dtype=torch.float32)
        assert a.shape == b.shape, (a.shape, b.shape)
        diff = (a - b).abs()
        print(f'compare: max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e}')
        verdict = 'PASS' if diff.max() < 5e-3 else 'FAIL'
        print(f'verdict: {verdict} (tolerance 5e-03)')
        sys.exit(0 if diff.max() < 5e-3 else 1)


def rmsnorm_local(x, gain, eps):
    ss = 0.0
    for v in x.tolist():
        ss += float(v) * float(v)
    scale = 1.0 / (ss / x.shape[0] + eps) ** 0.5
    return x * scale * gain


if __name__ == '__main__':
    main()
