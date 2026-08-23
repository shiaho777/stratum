#!/usr/bin/env python3
"""dit_oracle.py — independent torch reference for the Mini-DiT probe
(dit_probe.c, epic #35 Phase-1 spike).

Mirrors the C implementation exactly:
  - deterministic seed state from (t,h,w) grid coords,
  - timestep embedding omega_i = theta^(-i/(H/2)), [sin; cos] concat,
  - AdaLN-lite: xn = rmsnorm(x)*(1+gate)+shift (gate/shift are te@W),
  - MM-RoPE: 8 pairs; pair0 <- t, pair1 <- h, pairs 2.. <- w;
    angle = coord * theta^(-p/pairs), interleaved rotation,
  - bidirectional full attention (scale 1/sqrt(Hd), max-sub softmax),
  - FFN: down(silu(gate)*up),
  - final norm + output head; dumps SDIT0001 format.

Usage:
    python3 dit_oracle.py --gguf /tmp/dit.gguf --out /tmp/py_dit.bin \
        [--timestep 0.25] [--compare /tmp/c_dit.bin]
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

    cfg = {}
    for key, (vt, vb) in kv.items():
        if vt == 4:
            cfg[key] = struct.unpack('<I', vb)[0]
        elif vt == 6:
            cfg[key] = struct.unpack('<f', vb)[0]
        elif vt == 8:
            cfg[key] = vb.decode('utf-8', 'replace')

    tensors = {name: (dims, ty, body_off + off)
               for name, dims, ty, off in records}

    def load(name):
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

    weights = {name: load(name) for name, _, _, _ in records}
    return cfg, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--timestep', type=float, default=0.25)
    ap.add_argument('--compare')
    args = ap.parse_args()

    cfg, gt = read_gguf(args.gguf)
    p = 'dit.'
    S = cfg[f'{p}sequence_length']
    GT = cfg[f'{p}grid_t']
    GH = cfg[f'{p}grid_h']
    GW = cfg[f'{p}grid_w']
    H = cfg[f'{p}embedding_length']
    NQ = cfg[f'{p}attention.head_count']
    HD = cfg[f'{p}attention.key_length']
    NL = cfg[f'{p}block_count']
    FF = cfg[f'{p}feed_forward_length']
    theta = float(cfg.get(f'{p}rope.freq_base', 10000.0))
    tau = args.timestep
    heads = NQ

    # deterministic seed state — mirrors dit_probe.c exactly
    x = torch.zeros(S, H, dtype=torch.float32)
    for s in range(S):
        t = s // (GH * GW)
        h = (s // GW) % GH
        w = s % GW
        for d in range(H):
            ph = 0.6180339887 * (s + 1) + 0.3819660113 * d
            x[s, d] = float(0.5 * __import__('math').sin(ph)) + \
                0.02 * (((t * 13 + h * 7 + w * 3 + d) % 11) - 5)

    # timestep embedding
    half = H // 2
    omegas = torch.tensor([theta ** (-(i / half)) for i in range(half)],
                          dtype=torch.float64)
    angles = omegas * tau
    te = torch.cat([torch.sin(angles), torch.cos(angles)]).float()

    def lin(name, inp):
        w = gt[f'{name}.weight']  # [out, in] f32 (from f16)
        return (inp.double() @ w.T.double()).float()

    def rmsnorm(xv, gain):
        ss = torch.sum(xv.double() * xv.double()).item()
        scale = float(1.0 / ((ss / xv.shape[0] + 1e-5) ** 0.5))
        return xv * scale * gain

    def mm_rope(vec, t, h, w):
        out = vec.clone()
        pairs = HD // 2
        for p_i in range(pairs):
            coord = t if p_i == 0 else (h if p_i == 1 else w)
            angle = coord * (theta ** (-(p_i / pairs)))
            c = float(torch.cos(torch.tensor(angle)))
            s = float(torch.sin(torch.tensor(angle)))
            x0, x1 = float(out[2 * p_i]), float(out[2 * p_i + 1])
            out[2 * p_i] = x0 * c - x1 * s
            out[2 * p_i + 1] = x0 * s + x1 * c
        return out

    coords = [(s // (GH * GW), (s // GW) % GH, s % GW) for s in range(S)]
    import os as _os
    dbg = _os.environ.get('STRATUM_BLOCK_DBG')

    for li in range(NL):
        resid = x.clone()
        xn = torch.stack([rmsnorm(x[s], gt[f'blk.{li}.attn_norm.weight'])
                          for s in range(S)])
        sh = lin(f'blk.{li}.ada_shift', te)
        ga = lin(f'blk.{li}.ada_gate', te)
        xn = xn * (1.0 + ga.unsqueeze(0)) + sh.unsqueeze(0)
        if dbg and li == 0:
            xn.detach().numpy().tofile('/tmp/py_xn.bin')

        q = torch.stack([mm_rope(lin(f'blk.{li}.attn_q', xn[s]), *coords[s])
                         for s in range(S)])
        k = torch.stack([mm_rope(lin(f'blk.{li}.attn_k', xn[s]), *coords[s])
                         for s in range(S)])
        v = torch.stack([lin(f'blk.{li}.attn_v', xn[s]) for s in range(S)])
        if dbg and li == 0:
            print(f"[DBG] q_rope[0..2]={float(q[0,0]):.6f} {float(q[0,1]):.6f} {float(q[0,2]):.6f} k_rope[0..2]={float(k[0,0]):.6f} {float(k[0,1]):.6f} {float(k[0,2]):.6f}")
        if dbg and li == 0:
            print(f"[DBG] v00[0..2]={float(v[0,0]):.6f} {float(v[0,1]):.6f} {float(v[0,2]):.6f}")
            print(f"[DBG] k_tok1_rope[0..2]={float(k[1,0]):.6f} {float(k[1,1]):.6f} {float(k[1,2]):.6f}")

        attn = torch.zeros(S, H)
        scale = 1.0 / (HD ** 0.5)
        for hh in range(heads):
            qs = q[:, hh * HD:(hh + 1) * HD]          # [S, HD]
            ks = k[:, hh * HD:(hh + 1) * HD]
            vs = v[:, hh * HD:(hh + 1) * HD]
            logits = (qs @ ks.T) * scale               # [S, S], float32
            logits = logits - logits.max(dim=1, keepdim=True).values
            probs = torch.softmax(logits, dim=1)
            attn[:, hh * HD:(hh + 1) * HD] = probs @ vs
        if dbg and li == 0:
            print(f"[DBG] raw_attn[0..2]={float(attn[0,0]):.6f} {float(attn[0,1]):.6f} {float(attn[0,2]):.6f}")
            for hh in range(heads):
                print(f"[DBG] head{hh} out[0..1]={float(attn[0,hh*HD]):.6f} {float(attn[0,hh*HD+1]):.6f}")
        proj = torch.stack([lin(f'blk.{li}.attn_output', attn[s])
                            for s in range(S)])
        x = resid + proj

        resid = x.clone()
        xn = torch.stack([rmsnorm(x[s], gt[f'blk.{li}.mlp_norm.weight'])
                          for s in range(S)])
        sh = lin(f'blk.{li}.mlp_ada_shift', te)
        ga = lin(f'blk.{li}.mlp_ada_gate', te)
        xn = xn * (1.0 + ga.unsqueeze(0)) + sh.unsqueeze(0)
        g = torch.stack([lin(f'blk.{li}.ffn_gate', xn[s]) for s in range(S)])
        u = torch.stack([lin(f'blk.{li}.ffn_up', xn[s]) for s in range(S)])
        a = (g / (1.0 + torch.exp(-g))) * u
        ff = torch.stack([lin(f'blk.{li}.ffn_down', a[s]) for s in range(S)])
        x = resid + ff
    if dbg and li == 0:
        for s2 in range(0, S, 8):
            print(f"[DBG] PA s{s2}: {float(x[s2,0]):.6f} {float(x[s2,1]):.6f} {float(x[s2,2]):.6f}")

    xn = torch.stack([rmsnorm(x[s], gt['final_norm.weight']) for s in range(S)])
    out = torch.stack([lin('output_head', xn[s]) for s in range(S)])

    with open(args.out, 'wb') as f:
        f.write(b'SDIT0001')
        f.write(struct.pack('<II', S, H))
        f.write(out.flatten().numpy().astype('<f4').tobytes())
    print(f'wrote {args.out}: tokens={S} H={H}')

    if args.compare:
        ref = open(args.compare, 'rb').read()
        assert ref[:8] == b'SDIT0001'
        r_S, r_H = struct.unpack('<II', ref[8:16])
        mine = open(args.out, 'rb').read()
        m_S, m_H = struct.unpack('<II', mine[8:16])
        assert (r_S, r_H) == (m_S, m_H), (r_S, r_H, m_S, m_H)
        aa = torch.frombuffer(bytearray(ref[16:]), dtype=torch.float32)
        bb = torch.frombuffer(bytearray(mine[16:]), dtype=torch.float32)
        diff = (aa - bb).abs()
        print(f'compare: max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e}')
        verdict = 'PASS' if diff.max() < 5e-3 else 'FAIL'
        print(f'verdict: {verdict} (tolerance 5e-03)')
        sys.exit(0 if diff.max() < 5e-3 else 1)


if __name__ == '__main__':
    main()
