#!/usr/bin/env python3
"""Pack MoE expert slices so gate, up and down of one expert are contiguous.

The sidecar is a byte copy. Nothing is requantized. The engine reads it
through STRATUM_MOE_PACK=<path> and checks the bytes against the GGUF.

    python3 stratum/tools/moe_expert_pack.py <model.gguf> <out.moep>

Layout (little-endian):
    magic "MOEP", u32 version=1, u32 n_layers, u32 n_experts,
    u32 n_embed, u32 n_ff, u32 gate_type, u32 up_type, u32 down_type,
    u64 gate_slice, u64 up_slice, u64 down_slice, u64 data_off,
    pad to data_off (4096),
    then for layer in order, for expert in order: gate bytes | up | down.
"""

import struct
import sys
from pathlib import Path

BLOCK = {
    0: (4, 1), 1: (2, 1), 30: (2, 1),
    2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32),
    8: (34, 32), 9: (40, 32),
    10: (84, 256), 11: (110, 256), 12: (144, 256),
    13: (176, 256), 14: (210, 256), 15: (292, 256),
    20: (18, 32), 23: (136, 256),
}


def slice_bytes(ty, k_in, n_out):
    if ty == 43:  # Q4K_W16, same formula as moe_slice_bytes
        return n_out * k_in * 2 + n_out * (k_in // 256) * 8 + n_out * (k_in // 32) * 2
    if ty == 44:
        return n_out * k_in * 2 + n_out * (k_in // 256) * 8
    if ty not in BLOCK:
        raise SystemExit(f'unsupported expert ggml type {ty}')
    bb, eb = BLOCK[ty]
    ne = k_in * n_out
    if ne % eb != 0:
        raise SystemExit(f'type {ty}: {ne} elements is not a multiple of {eb}')
    return (ne // eb) * bb


class R:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise SystemExit('truncated GGUF header')
        s = self.b[self.p:self.p + n]
        self.p += n
        return s

    def u32(self):
        return struct.unpack_from('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack_from('<Q', self.take(8))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode('utf-8', errors='replace')

    def skip_value(self):
        t = self.u32()
        if t == 8:
            self.string()
        elif t == 9:
            et = self.u32()
            n = self.u64()
            for _ in range(n):
                if et == 8:
                    self.string()
                else:
                    self._skip_scalar(et)
        else:
            self._skip_scalar(t)

    def _skip_scalar(self, t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
              10: 8, 11: 8, 12: 8}.get(t)
        if sz is None:
            raise SystemExit(f'bad gguf value type {t}')
        self.take(sz)


def load_index(path):
    data = Path(path).read_bytes()
    r = R(data)
    if r.take(4) != b'GGUF':
        raise SystemExit('not a GGUF file')
    version = r.u32()
    if version not in (2, 3):
        raise SystemExit(f'GGUF version {version} is not supported')
    n_tensors = r.u64()
    n_kv = r.u64()
    align = 32
    arch = None
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        # peek type by parsing; keep a few ints/strings we need
        pos = r.p
        vt = r.u32()
        if vt == 8:
            meta[key] = r.string()
        elif vt == 4:
            meta[key] = r.u32()
        elif vt == 5:
            meta[key] = struct.unpack_from('<i', r.take(4))[0]
        elif vt == 10:
            meta[key] = r.u64()
        else:
            r.p = pos
            r.skip_value()
        if key == 'general.alignment' and isinstance(meta.get(key), int):
            align = meta[key]
        if key == 'general.architecture' and isinstance(meta.get(key), str):
            arch = meta[key]
    tensors = {}
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ty = r.u32()
        off = r.u64()
        tensors[name] = (dims, ty, off)
    body = (r.p + align - 1) & ~(align - 1)
    return data, arch, meta, tensors, body


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: moe_expert_pack.py <model.gguf> <out.moep>')
    src, dst = sys.argv[1], sys.argv[2]
    data, arch, meta, tensors, body = load_index(src)
    if not arch:
        raise SystemExit('missing general.architecture')
    n_layers = int(meta.get(f'{arch}.block_count', 0))
    n_exp = int(meta.get(f'{arch}.expert_count', 0))
    n_emb = int(meta.get(f'{arch}.embedding_length', 0))
    n_ff = int(meta.get(f'{arch}.feed_forward_length', 0))
    if n_layers < 1 or n_exp < 1:
        raise SystemExit(f'{arch}: block_count={n_layers} expert_count={n_exp}')

    def tens(li, kind):
        name = f'blk.{li}.ffn_{kind}_exps.weight'
        if name not in tensors:
            raise SystemExit(f'missing {name}')
        return tensors[name]

    dims, gty, _ = tens(0, 'gate')
    _, uty, _ = tens(0, 'up')
    _, dty, _ = tens(0, 'down')
    if len(dims) < 3 or int(dims[2]) != n_exp:
        raise SystemExit(f'gate experts dim {dims} != expert_count {n_exp}')
    gsb = slice_bytes(gty, int(dims[0]), int(dims[1]))
    udims = tens(0, 'up')[0]
    usb = slice_bytes(uty, int(udims[0]), int(udims[1]))
    ddims = tens(0, 'down')[0]
    dsb = slice_bytes(dty, int(ddims[0]), int(ddims[1]))

    data_off = 4096
    hdr = struct.pack('<4s8I4Q', b'MOEP', 1, n_layers, n_exp, n_emb, n_ff,
                      gty, uty, dty, gsb, usb, dsb, data_off)
    if len(hdr) > data_off:
        raise SystemExit('header exceeds data_off')
    out = bytearray(hdr + b'\0' * (data_off - len(hdr)))

    for li in range(n_layers):
        parts = []
        for kind, sb in (('gate', gsb), ('up', usb), ('down', dsb)):
            dims, ty, off = tens(li, kind)
            if kind == 'gate' and (ty != gty or slice_bytes(ty, int(dims[0]), int(dims[1])) != sb):
                raise SystemExit(f'layer {li} {kind} layout differs from layer 0')
            if kind == 'up' and (ty != uty or slice_bytes(ty, int(dims[0]), int(dims[1])) != sb):
                raise SystemExit(f'layer {li} {kind} layout differs from layer 0')
            if kind == 'down' and (ty != dty or slice_bytes(ty, int(dims[0]), int(dims[1])) != sb):
                raise SystemExit(f'layer {li} {kind} layout differs from layer 0')
            base = body + off
            blob = data[base:base + sb * n_exp]
            if len(blob) != sb * n_exp:
                raise SystemExit(f'short read of blk.{li}.ffn_{kind}_exps')
            parts.append(blob)
        gate, up, down = parts
        for e in range(n_exp):
            out += gate[e * gsb:(e + 1) * gsb]
            out += up[e * usb:(e + 1) * usb]
            out += down[e * dsb:(e + 1) * dsb]

    Path(dst).write_bytes(out)
    print(f'wrote {dst}: {n_layers} layers x {n_exp} experts, '
          f'stride {gsb + usb + dsb} bytes, file {len(out)} bytes')


if __name__ == '__main__':
    main()
