#!/usr/bin/env python3
"""gguf_align.py <in.gguf> <out.gguf> [--alignment N] [--no-verify]

Rewrite a GGUF with a streaming-friendly layout:
  * every tensor payload starts at a page-aligned offset (default 16 KiB)
  * tensors are grouped by layer (token_embd* first, then blk.0..N in numeric
    order, then everything else), so one layer = one contiguous byte range

Payload bytes are copied verbatim — this is a byte re-arrangement, not
requantization (same class as the type-42 nibble layout). The output stays a
valid GGUF: `general.alignment` is updated/added, which stratum's reader
honors (stratum_gguf.h accepts powers of two in [16, 65536]).

After writing, every tensor's payload is sha256-compared between source and
output unless --no-verify is given, so "values are unchanged" is verified,
not assumed.
"""
import argparse
import hashlib
import os
import re
import struct
import sys

GGUF_MAGIC = 0x46554747          # "GGUF"
DEFAULT_ALIGNMENT = 16384
T_UINT32, T_STRING, T_ARRAY = 4, 8, 9

# ggml type -> (bytes_per_block, elements_per_block); same table as
# tools/gguf_inspect.py extended with the scalar types.
TYPE_SIZES = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32),
    7: (24, 32), 8: (33, 32), 9: (34, 32), 10: (84, 256), 11: (110, 256),
    12: (144, 256), 13: (176, 256), 14: (210, 256), 15: (196, 256),
    16: (35, 256), 17: (37, 256), 18: (50, 256), 19: (50, 256),
    20: (32, 32), 21: (72, 256), 22: (42, 256), 23: (40, 32),
    24: (1, 1), 25: (2, 1), 26: (4, 1), 27: (8, 1), 28: (8, 1), 30: (2, 1),
}
KV_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                   10: 8, 11: 8, 12: 8}


class ParseError(Exception):
    pass


def _read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise ParseError("unexpected EOF")
    return b


def _rd_u32(f):
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _rd_u64(f):
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _skip_value(f, vtype):
    if vtype == T_STRING:
        f.seek(_rd_u64(f), 1)
    elif vtype == T_ARRAY:
        elem = _rd_u32(f)
        for _ in range(_rd_u64(f)):
            _skip_value(f, elem)
    elif vtype in KV_SCALAR_SIZES:
        f.seek(KV_SCALAR_SIZES[vtype], 1)
    else:
        raise ParseError(f"unknown kv type {vtype}")


class TensorInfo:
    __slots__ = ("name", "dims", "type", "rel_off", "nbytes")

    def __init__(self, name, dims, ttype, rel_off, nbytes):
        self.name, self.dims, self.type = name, dims, ttype
        self.rel_off, self.nbytes = rel_off, nbytes


class Gguf:
    def __init__(self):
        self.version = 3
        self.n_kv = 0
        self.kv_region = b""        # header..end-of-KVs, verbatim bytes
        self.align_val_span = None  # span of the alignment value inside it
        self.alignment = None       # None => no explicit general.alignment
        self.tensors = []
        self.src_data_start = 0


def parse(path):
    g = Gguf()
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        if _rd_u32(f) != GGUF_MAGIC:
            raise ParseError(f"{path}: not a GGUF file")
        g.version = _rd_u32(f)
        if g.version < 2:
            raise ParseError(f"{path}: GGUF v{g.version} unsupported")
        n_tensors, g.n_kv = _rd_u64(f), _rd_u64(f)

        kv_start = f.tell()
        for _ in range(g.n_kv):
            klen = _rd_u64(f)
            key = _read_exact(f, klen).decode("utf-8")
            vtype = _rd_u32(f)
            if key == "general.alignment" and vtype == T_UINT32:
                g.alignment = _rd_u32(f)
                g.align_val_span = (f.tell() - 4 - kv_start,
                                    f.tell() - kv_start)
            else:
                _skip_value(f, vtype)
        kv_end = f.tell()
        f.seek(kv_start)
        g.kv_region = _read_exact(f, kv_end - kv_start)

        seen = set()
        for _ in range(n_tensors):
            nlen = _rd_u64(f)
            name = _read_exact(f, nlen).decode("utf-8")
            if name in seen:
                raise ParseError(f"duplicate tensor name {name}")
            seen.add(name)
            ndims = _rd_u32(f)
            dims = [_rd_u64(f) for _ in range(ndims)]
            ttype = _rd_u32(f)
            rel = _rd_u64(f)
            if ttype not in TYPE_SIZES:
                raise ParseError(f"{name}: unknown ggml type {ttype}")
            bpw, epb = TYPE_SIZES[ttype]
            nelem = 1
            for d in dims:
                nelem *= d
            nbytes = -(-nelem // epb) * bpw
            g.tensors.append(TensorInfo(name, dims, ttype, rel, nbytes))

        eff_align = g.alignment if g.alignment else 32
        after_index = f.tell()
        g.src_data_start = ((after_index + eff_align - 1)
                            // eff_align * eff_align)

    for t in g.tensors:
        abs_off = g.src_data_start + t.rel_off
        if abs_off + t.nbytes > fsize:
            raise ParseError(
                f"{t.name} [{abs_off}..{abs_off + t.nbytes}) "
                f"exceeds file size {fsize}")
    return g


BLK_RE = re.compile(r"^blk\.(\d+)\.")


def ordered(g):
    """token_embd* first, then blk.N ascending, then the rest (stable)."""

    def key(i):
        t = g.tensors[i]
        m = BLK_RE.match(t.name)
        if m:
            return (1, int(m.group(1)), 0, i)
        if t.name.startswith("token_embd"):
            return (0, 0, 0, i)
        return (2, 0, 0, i)

    return sorted(range(len(g.tensors)), key=key)


def serialize_tensor_info(t, new_rel_off):
    nb = t.name.encode("utf-8")
    out = struct.pack("<Q", len(nb)) + nb
    out += struct.pack("<I", len(t.dims))
    out += b"".join(struct.pack("<Q", d) for d in t.dims)
    out += struct.pack("<IQ", t.type, new_rel_off)
    return out


def convert(src_path, out_path, alignment, verify=True):
    g = parse(src_path)
    src_align = g.alignment if g.alignment else 32
    order = ordered(g)

    # Header KV region: copied verbatim, only general.alignment changes.
    kv_region = bytearray(g.kv_region)
    n_kv_extra = 0
    if g.align_val_span is not None:
        s, e = g.align_val_span
        kv_region[s:e] = struct.pack("<I", alignment)
    else:
        kb = b"general.alignment"
        kv_region += struct.pack("<Q", len(kb)) + kb
        kv_region += struct.pack("<II", T_UINT32, alignment)
        n_kv_extra = 1

    # Tensor infos with fresh aligned offsets.
    infos = []
    layout = []                      # (TensorInfo, new_rel_off)
    cursor = 0
    for i in order:
        t = g.tensors[i]
        cursor = -(-cursor // alignment) * alignment
        layout.append((t, cursor))
        infos.append(serialize_tensor_info(t, cursor))
        cursor += t.nbytes
    total_payload = cursor

    # GGUF v3 header: magic u32, version u32, tensor_count u64, kv_count u64
    head = struct.pack("<IIQQ", GGUF_MAGIC, max(g.version, 2),
                       len(g.tensors), g.n_kv + n_kv_extra)
    pos = len(head) + len(kv_region) + sum(len(x) for x in infos)
    data_start = -(-pos // alignment) * alignment

    with open(src_path, "rb") as src, open(out_path, "wb") as out:
        out.write(head)
        out.write(kv_region)
        for info in infos:
            out.write(info)
        out.write(b"\0" * (data_start - pos))
        chunk = 4 * 1024 * 1024
        for t, new_rel in layout:
            src.seek(g.src_data_start + t.rel_off)
            left = t.nbytes
            while left:
                n = min(chunk, left)
                out.write(_read_exact(src, n))
                left -= n
            end = new_rel + t.nbytes
            gap = -(-end // alignment) * alignment - end
            if gap:
                out.write(b"\0" * gap)

    print(f"{src_path} -> {out_path}")
    print(f"  alignment : {src_align} -> {alignment}")
    nl = [t.name for t, _ in layout]
    emb = sum(1 for n in nl if n.startswith("token_embd"))
    blk = sum(1 for n in nl if BLK_RE.match(n))
    print(f"  tensors   : {len(nl)} ({emb} embedding, {blk} blk.*, "
          f"{len(nl) - emb - blk} other)")
    print(f"  size      : {os.path.getsize(out_path)/(1024**3):.3f} GB "
          f"(was {os.path.getsize(src_path)/(1024**3):.3f} GB)")

    if not verify:
        return True

    g2 = parse(out_path)             # full re-parse validates readability
    problems = []
    if g2.alignment != alignment:
        problems.append(f"output alignment reads back as {g2.alignment}")
    if [t.name for t in g2.tensors] != nl:
        problems.append("tensor list differs after round-trip")
    chunk = 4 * 1024 * 1024
    with open(src_path, "rb") as src, open(out_path, "rb") as out:
        for (t, new_rel), t2 in zip(layout, g2.tensors):
            if t2.rel_off % alignment != 0:
                problems.append(f"{t.name}: offset +{t2.rel_off} unaligned")
            h = [[hashlib.sha256(), g.src_data_start + t.rel_off, src],
                 [hashlib.sha256(), g2.src_data_start + t2.rel_off, out]]
            for hh, off, fh in h:
                fh.seek(off)
                left = t.nbytes
                while left:
                    k = min(chunk, left)
                    hh.update(_read_exact(fh, k))
                    left -= k
            if h[0][0].hexdigest() != h[1][0].hexdigest():
                problems.append(f"{t.name}: payload hash mismatch")
    if problems:
        print("  VERIFY FAILED:")
        for p in problems[:10]:
            print(f"    {p}")
        return False
    print(f"  verified  : all {len(layout)} payloads byte-identical, "
          f"offsets aligned to {alignment}")
    return True


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    a = args.alignment
    if a < 16 or a > 65536 or (a & (a - 1)) != 0:
        print(f"alignment must be a power of two in [16, 65536], got {a}",
              file=sys.stderr)
        return 2
    try:
        ok = convert(args.src, args.out, a, verify=not args.no_verify)
    except ParseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
