#!/usr/bin/env python3
"""tiny_moe_oracle.py — independent numpy reference for the tiny MoE GGUF.

Reimplements the moe arch's arithmetic (dense attention + router top-k
expert FFN) directly from the file's weights, mirroring the engine's
deterministic float behavior:
  * rmsnorm: double-precision sum, fp32 scale (as la_rmsnorm/moe_rmsnorm)
  * softmax: max-subtract in fp32
  * RoPE: pair rotation over rope_dim/2 pairs
  * Q4_K dequant: d*sc*q - dmin*m per sub-block (stratum_q4k.h formula,
    q4k_get_scale_min bit unpack exactly)
  * top-k routing: descending weight, ties -> LOWER expert id

Prints the greedy sequence for a prompt; compare with engine output.
"""
import math
import struct
import sys

import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinymoe.gguf"
PROMPT = [int(x) for x in sys.argv[2:]] or [0, 1]
GEN = 6


def load(path):
    data = open(path, "rb").read()
    assert data[:4] == b"GGUF"
    _ver, n_tensors, n_kv = struct.unpack_from("<IQQ", data, 4)
    pos = 24

    def rd_str():
        nonlocal pos
        (n,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        s = data[pos:pos + n].decode()
        pos += n
        return s

    def rd_val(vt):
        nonlocal pos
        if vt == 4:
            v = struct.unpack_from("<I", data, pos)[0]; pos += 4; return v
        if vt == 5:
            v = struct.unpack_from("<i", data, pos)[0]; pos += 4; return v
        if vt == 6:
            v = struct.unpack_from("<f", data, pos)[0]; pos += 4; return v
        if vt == 8:
            return rd_str()
        if vt == 9:
            et = struct.unpack_from("<I", data, pos)[0]; pos += 4
            (n,) = struct.unpack_from("<Q", data, pos); pos += 8
            return [rd_val(et) for _ in range(n)]
        raise ValueError(f"kv type {vt}")

    kv = {}
    for _ in range(n_kv):
        k = rd_str()
        vt = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        kv[k] = rd_val(vt)

    tensors = {}
    for _ in range(n_tensors):
        name = rd_str()
        nd = struct.unpack_from("<I", data, pos)[0]; pos += 4
        dims = list(struct.unpack_from(f"<{'Q' * nd}", data, pos)); pos += 8 * nd
        tt = struct.unpack_from("<I", data, pos)[0]; pos += 4
        off = struct.unpack_from("<Q", data, pos)[0]; pos += 8
        tensors[name] = dict(dims=dims, type=tt, off=off)

    align = kv.get("general.alignment", 32)
    body = (pos + align - 1) // align * align
    return kv, tensors, body


class Reader:
    def __init__(self, data, tensors, body):
        self.data, self.tensors, self.body = data, tensors, body

    def raw(self, name):
        t = self.tensors[name + ".weight"] if name not in self.tensors \
            and (name + ".weight") in self.tensors else self.tensors[name]
        nelem = 1
        for d in t["dims"]:
            nelem *= d
        ty = t["type"]
        nb = {0: 4 * nelem, 1: 2 * nelem}.get(ty)
        if ty == 12:                        # Q4_K, 144 B per 256
            nb = ((nelem + 255) // 256) * 144
        if nb is None:
            raise ValueError(f"type {ty}")
        o = self.body + t["off"]
        return self.data[o:o + nb], dims_ok(t["dims"])

    def f32(self, name, shape):
        b, _ = self.raw(name)
        return np.frombuffer(b, dtype="<f4").reshape(shape).astype(np.float64)

    def f16(self, name, shape):
        # ggml stores dims reversed; caller passes the logical (row-major
        # [N_out, K_in]) shape of the ORIGINAL python matrix.
        b, dims = self.raw(name)
        nelem = len(b) // 2
        a = np.frombuffer(b, dtype="<f2").astype(np.float32)
        return a.reshape(nelem if False else shape)

    def mat_any(self, name, logical_shape):
        """Dequant any weight to f64 [N_out, K_in] = logical_shape.

        GGUF dims list is the reverse of the python builder's tuple. Handles
        F32/F16/Q4_K."""
        t = self.tensors[name]
        ty = t["type"]
        N_out, K_in = logical_shape
        if ty in (0, 1):
            b, _ = self.raw(name)
            dt = "<f4" if ty == 0 else "<f2"
            flat = np.frombuffer(b, dtype=dt).astype(np.float64)
            rows = []
            nb_row = ((K_in + 255) // 256) * 144 if ty == 12 else None
            return flat.reshape(N_out, K_in)
        if ty == 12:
            E_extra = 1
            blob, _ = self.raw(name)
            row_blocks = K_in // 256
            per_row = row_blocks * 144
            out = np.empty((N_out, K_in), dtype=np.float32)
            for r in range(N_out):
                ob = r * per_row
                y = out[r]
                for bi in range(row_blocks):
                    o = ob + bi * 144
                    dd = struct.unpack_from("<e", blob, o)[0]
                    dm = struct.unpack_from("<e", blob, o + 2)[0]
                    s = blob[o + 4:o + 16]
                    qs = np.frombuffer(blob[o + 16:o + 144], dtype=np.uint8)
                    seg = y[bi * 256:(bi + 1) * 256]
                    for half in range(4):
                        j = half * 2
                        if j < 4:
                            sc1 = s[j] & 63; m1 = s[j + 4] & 63
                        else:
                            sc1 = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)
                            m1 = (s[j + 4] >> 4) | ((s[j] >> 6) << 4)
                        j2 = j + 1
                        if j2 < 4:
                            sc2 = s[j2] & 63; m2 = s[j2 + 4] & 63
                        else:
                            sc2 = (s[j2 + 4] & 0x0F) | ((s[j2 - 4] >> 6) << 4)
                            m2 = (s[j2 + 4] >> 4) | ((s[j2] >> 6) << 4)
                        base = half * 64
                        lo = qs[half * 32:(half + 1) * 32]
                        seg[base:base + 32] = (lo & 0xF).astype(np.float32)
                        seg[base + 32:base + 64] = (lo >> 4).astype(np.float32)
                        seg[base:base + 32] = seg[base:base + 32] \
                            * np.float32(dd * sc1) - np.float32(dm * m1)
                        seg[base + 32:base + 64] = seg[base + 32:base + 64] \
                            * np.float32(dd * sc2) - np.float32(dm * m2)
            return out.astype(np.float64)
        raise ValueError(f"type {ty}")

    def exp_rows(self, name, N_out, K_in, E=1):
        """Stacked expert tensor -> [E][N_out][K_in] f64, dispatch on type
        (F16 build writes F16 experts; q4k build writes Q4_K)."""
        tname = name if name in self.tensors else name + ".weight"
        ty = self.tensors[tname]["type"]
        if ty == 1:
            b, _ = self.raw(name)
            flat = np.frombuffer(b, dtype="<f2").astype(np.float64)
            return flat.reshape(E, N_out, K_in)
        return self.q4k_rows(name, N_out, K_in, E)

    def q4k_rows(self, name, N_out, K_in, E=1):
        """Dequant a stacked tensor to f64 [E][N_out][K_in].

        Row r of each expert slice is K_in elements = K_in/256 blocks;
        rows are contiguous. Mirrors q4k_dequant_block_scalar +
        q4k_get_scale_min exactly."""
        blob, _ = self.raw(name)
        row_blocks = K_in // 256
        out = np.empty((E, N_out, K_in), dtype=np.float32)
        per_row_bytes = row_blocks * 144
        e_stride = N_out * per_row_bytes
        for e in range(E):
            base_e = e * e_stride
            for r in range(N_out):
                ob = base_e + r * per_row_bytes
                y = out[e, r]
                for bi in range(row_blocks):
                    o = ob + bi * 144
                    dd = struct.unpack_from("<e", blob, o)[0]
                    dm = struct.unpack_from("<e", blob, o + 2)[0]
                    s = blob[o + 4:o + 16]
                    qs = blob[o + 16:o + 144]
                    seg = y[bi * 256:(bi + 1) * 256]
                    qoff = 0
                    for half in range(4):
                        j = half * 2
                        if j < 4:
                            sc1 = s[j] & 63
                            m1 = s[j + 4] & 63
                        else:
                            sc1 = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)
                            m1 = (s[j + 4] >> 4) | ((s[j] >> 6) << 4)
                        j2 = j + 1
                        if j2 < 4:
                            sc2 = s[j2] & 63
                            m2 = s[j2 + 4] & 63
                        else:
                            sc2 = (s[j2 + 4] & 0x0F) | ((s[j2 - 4] >> 6) << 4)
                            m2 = (s[j2 + 4] >> 4) | ((s[j2] >> 6) << 4)
                        d1 = dd * sc1
                        mm1 = dm * m1
                        d2 = dd * sc2
                        mm2 = dm * m2
                        base = half * 64
                        seg[base:base + 32] = np.float32(d1) * (
                            qs[qoff:qoff + 32].numpy_view_low() if False
                            else (np.frombuffer(qs[qoff:qoff + 32],
                                                dtype=np.uint8) & 0xF))
                        seg[base + 32:base + 64] = np.float32(d2) * (
                            np.frombuffer(qs[qoff:qoff + 32],
                                          dtype=np.uint8) >> 4)
                        seg[base:base + 32] -= np.float32(mm1)
                        seg[base + 32:base + 64] -= np.float32(mm2)
                        qoff += 32
        return out.astype(np.float64)


def dims_ok(dims):
    return dims


def main():
    kv, tensors, body = load(MODEL)
    R = Reader(open(MODEL, "rb").read(), tensors, body)
    arch = kv["general.architecture"]
    L = kv[f"{arch}.block_count"]
    H = kv[f"{arch}.embedding_length"]
    NQ = kv[f"{arch}.attention.head_count"]
    NK = kv[f"{arch}.attention.head_count_kv"]
    HD = kv[f"{arch}.attention.key_length"]
    FF = kv[f"{arch}.feed_forward_length"]
    # vocab is not in the kv for our generator; read it off the embedding
    V = tensors["token_embd.weight"]["dims"][1]
    eps = kv[f"{arch}.attention.layer_norm_rms_epsilon"]
    theta = kv[f"{arch}.rope.freq_base"]
    rope_dim = kv[f"{arch}.rope.dimension_count"]
    E = kv[f"{arch}.expert_count"]
    K_ = kv[f"{arch}.expert_used_count"]

    def rms(x, g):
        return x * (g / math.sqrt(float(np.mean(x * x)) + eps))

    def softmax(v):
        e = np.exp(v.astype(np.float32) - np.max(v))
        return (e / np.sum(e)).astype(np.float64)

    def rope(x, pos):
        x = x.copy()
        for k in range(rope_dim // 2):
            ang = pos / theta ** (2 * k / rope_dim)
            c, s = math.cos(ang), math.sin(ang)
            x0, x1 = x[2 * k], x[2 * k + 1]
            x[2 * k], x[2 * k + 1] = x0 * c - x1 * s, x0 * s + x1 * c
        return x

    def lin(w, x):                 # w [N,K] logical, x [K]
        return w @ x

    def attn_norm_w(li):
        return R.f32(f"blk.{li}.attn_norm.weight", (H,))

    def ffn_norm_w(li):
        return R.f32(f"blk.{li}.ffn_norm.weight", (H,))

    Wq = {li: R.mat_any(f"blk.{li}.attn_q.weight", (NQ * HD, H)) for li in range(L)}
    Wk = {li: R.mat_any(f"blk.{li}.attn_k.weight", (NK * HD, H)) for li in range(L)}
    Wv = {li: R.mat_any(f"blk.{li}.attn_v.weight", (NK * HD, H)) for li in range(L)}
    Wo = {li: R.mat_any(f"blk.{li}.attn_output.weight", (H, NQ * HD))
          for li in range(L)}
    # router stored [H, E] ggml-dims (= python (H, E)); logical load [E, H]
    Rout = {li: R.f32(f"blk.{li}.ffn_gate_inp.weight", (E, H))
            for li in range(L)}   # row-major [E][H], matches cblas NoTrans
    Gate = {li: R.exp_rows(f"blk.{li}.ffn_gate_exps", FF, H, E)
            for li in range(L)}
    Up = {li: R.exp_rows(f"blk.{li}.ffn_up_exps", FF, H, E) for li in range(L)}
    Down = {li: R.exp_rows(f"blk.{li}.ffn_down_exps", H, FF, E)
            for li in range(L)}

    def embed(tok):
        t = tensors["token_embd.weight"]
        if t["type"] == 0:
            return R.f32("token_embd.weight", (V, H))[tok]
        return R.mat_any("token_embd.weight", (V, H))[tok]

    def forward_all(tok, pos, cache):
        x = embed(tok)
        for li in range(L):
            res = x
            xn = rms(x, attn_norm_w(li))
            q = (Wq[li] @ xn).astype(np.float32).reshape(NQ, HD)
            k = (Wk[li] @ xn).astype(np.float32).reshape(NK, HD)
            v = (Wv[li] @ xn).astype(np.float32).reshape(NK, HD)
            for h in range(NQ):
                q[h] = rope(q[h], pos)
            for h in range(NK):
                k[h] = rope(k[h], pos)
            cache.setdefault(li, {"k": [], "v": []})
            cache[li]["k"].append(k)
            cache[li]["v"].append(v)
            out = np.zeros(NQ * HD, dtype=np.float64)
            scale = 1.0 / math.sqrt(HD)
            for h in range(NQ):
                kvh = h * NK // NQ
                ks = np.stack([kk[kvh] for kk in cache[li]["k"]])
                vs = np.stack([vv[kvh] for vv in cache[li]["v"]])
                att = softmax((ks @ q[h]) * scale)
                out[h * HD:(h + 1) * HD] = att @ vs
            x = res + Wo[li] @ out
            res = x
            xn = rms(x, ffn_norm_w(li))
            logits_r = softmax(Rout[li] @ xn.astype(np.float32))
            idx = np.argsort(-logits_r, kind="stable")[:K_]
            acc = np.zeros(H, dtype=np.float64)
            for e in idx:
                wgt = logits_r[e]
                g = Gate[li][e] @ xn
                u = Up[li][e] @ xn
                sw = (g / (1 + np.exp(-g))) * u
                acc += wgt * (Down[li][e] @ sw)
            x = res + acc
        xn = rms(x, R.f32("output_norm.weight", (H,)))
        lm = R.output_weight
        return lm @ xn

    def output_head():
        if "output.weight" in tensors:
            R.output_weight = R.mat_any("output.weight", (V, H))
        else:
            R.output_weight = R.mat_any("token_embd.weight", (V, H))

    output_head()
    cache = {}
    toks = []
    seq = list(PROMPT)
    first_sample = None
    for i, t in enumerate(seq):
        lg = forward_all(t, i, cache)
    a = int(np.argmax(lg))
    print("argmax:", [a])
    for _ in range(GEN):
        lg2 = forward_all(a, len(seq), cache)
        seq.append(a)
        a = int(np.argmax(lg2))
        print("argmax:", [a])
    return 0


if __name__ == "__main__":
    sys.exit(main())
