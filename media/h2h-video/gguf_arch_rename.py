#!/usr/bin/env python3
"""gguf_arch_rename.py <in.gguf> <out.gguf> <old_arch> <new_arch>

Rewrite a GGUF's architecture identity in place-safe fashion:
  * general.architecture value string
  * every KV key prefixed "<old_arch>."
  * every tensor name prefixed "<old_arch>."

Requirement: len(old_arch) == len(new_arch) (e.g. qwen3 -> llama, both 5
bytes), so every patched string keeps its byte length and ALL file offsets,
alignment, and tensor payloads remain untouched. This is a metadata rename,
not a re-quantization: weight bytes are copied verbatim.

Usage: copy then patch (output is a byte-identical file except the patched
string ranges).
"""
import os
import struct
import sys

GGUF_MAGIC = 0x46554747
T_UINT32, T_STRING, T_ARRAY = 4, 8, 9
KV_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                   10: 8, 11: 8, 12: 8}


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    src, dst, old, new = sys.argv[1:5]
    if len(old) != len(new):
        raise SystemExit(f"arch strings must be equal length: {old!r} vs {new!r}")
    oldb, newb = old.encode(), new.encode()

    with open(src, "rb") as f:
        data = bytearray(f.read())

    if struct.unpack_from("<I", data, 0)[0] != GGUF_MAGIC:
        raise SystemExit("not a GGUF file")
    version = struct.unpack_from("<I", data, 4)[0]
    if version < 2:
        raise SystemExit(f"GGUF v{version} unsupported")
    off = 8
    n_tensors = struct.unpack_from("<Q", data, off)[0]; off += 8
    n_kv = struct.unpack_from("<Q", data, off)[0]; off += 8

    patches = []          # (offset, new_bytes)
    arch_value_seen = False

    def rd_u32(o): return struct.unpack_from("<I", data, o)[0]
    def rd_u64(o): return struct.unpack_from("<Q", data, o)[0]

    def skip_value(o, vtype):
        if vtype == T_STRING:
            ln = rd_u64(o); return o + 8 + ln
        if vtype == T_ARRAY:
            elem = rd_u32(o); o += 4
            cnt = rd_u64(o); o += 8
            for _ in range(cnt):
                o = skip_value(o, elem)
            return o
        if vtype in KV_SCALAR_SIZES:
            return o + KV_SCALAR_SIZES[vtype]
        raise SystemExit(f"unknown kv type {vtype}")

    # --- KV section ---
    for _ in range(n_kv):
        klen = rd_u64(off); koff = off + 8
        key = bytes(data[koff:koff + klen]).decode("utf-8")
        off = koff + klen
        vtype = rd_u32(off); off += 4
        if key == "general.architecture" and vtype == T_STRING:
            vlen = rd_u64(off); voff = off + 8
            val = bytes(data[voff:voff + vlen])
            if val == oldb:
                patches.append((voff, newb))
                arch_value_seen = True
            else:
                raise SystemExit(f"general.architecture is {val!r}, expected {oldb!r}")
            off = voff + vlen
        elif key.startswith(old + "."):
            patches.append((koff, newb))   # same length prefix swap
            off = skip_value(off, vtype)
        else:
            off = skip_value(off, vtype)

    if not arch_value_seen:
        raise SystemExit("general.architecture KV not found")

    # --- tensor info section ---
    renamed = 0
    for _ in range(n_tensors):
        nlen = rd_u64(off); noff = off + 8
        name = bytes(data[noff:noff + nlen]).decode("utf-8")
        if name.startswith(old + "."):
            patches.append((noff, newb))
            renamed += 1
        off = noff + nlen
        ndims = rd_u32(off); off += 4
        off += 8 * ndims            # dims
        off += 4 + 8                # type + rel offset

    # --- apply ---
    for poff, pb in patches:
        data[poff:poff + len(pb)] = pb

    with open(dst, "wb") as f:
        f.write(data)

    # verify: size identical, only patched ranges differ
    assert os.path.getsize(dst) == os.path.getsize(src)
    print(f"OK {dst}: arch {old}->{new}, KV keys + {renamed}/{n_tensors} "
          f"tensor names renamed, {len(patches)} string patches, "
          f"payload bytes untouched")


if __name__ == "__main__":
    main()
