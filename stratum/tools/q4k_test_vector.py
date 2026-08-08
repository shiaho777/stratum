#!/usr/bin/env python3
"""
Generate a Q4_K test vector and dump expected fp32 dequant.

Strategy: extract a real Q4_K block from TinyLlama, write it to disk,
and write the expected fp32 dequant to a sibling file. The C scalar
dequant must match this fp32 byte-for-byte (modulo expected fp32 rounding).
"""

import struct, sys
from pathlib import Path

def fp16_to_fp32(h):
    sign = (h >> 15) & 1
    exp  = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            f = 0
        else:
            while (mant & 0x400) == 0:
                mant <<= 1
                exp -= 1
            mant &= 0x3FF
            exp += 1
            f = ((exp + (127 - 15)) << 23) | (mant << 13)
    elif exp == 31:
        f = 0x7F800000 | (mant << 13)
    else:
        f = ((exp + (127 - 15)) << 23) | (mant << 13)
    if sign: f |= 0x80000000
    return struct.unpack('<f', struct.pack('<I', f))[0]


def get_scale_min(j, scales):
    if j < 4:
        sc = scales[j]   & 63
        m  = scales[j+4] & 63
    else:
        sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        m  = (scales[j+4] >>   4) | ((scales[j]   >> 6) << 4)
    return sc, m


def dequant_block(block_bytes):
    """Decode one 144-byte Q4_K block to a list of 256 floats."""
    d_raw    = struct.unpack('<H', block_bytes[0:2])[0]
    dmin_raw = struct.unpack('<H', block_bytes[2:4])[0]
    scales = block_bytes[4:16]
    qs = block_bytes[16:144]
    d    = fp16_to_fp32(d_raw)
    dmin = fp16_to_fp32(dmin_raw)
    out = [0.0] * 256
    for js in range(0, 256, 64):
        is_ = js // 32
        sc1, m1 = get_scale_min(is_,     scales)
        sc2, m2 = get_scale_min(is_ + 1, scales)
        d1 = d * sc1; mm1 = dmin * m1
        d2 = d * sc2; mm2 = dmin * m2
        q_off = (js // 64) * 32
        for l in range(32):
            out[js + l]      = d1 * (qs[q_off + l] & 0xF) - mm1
            out[js + l + 32] = d2 * (qs[q_off + l] >> 4)  - mm2
    return out


def find_q4k_tensor(gguf_path):
    """Return (offset, n_blocks) for the first Q4_K tensor we find."""
    raw = Path(gguf_path).read_bytes()
    pos = 0
    assert raw[:4] == b'GGUF'
    pos = 4
    version = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    n_tensors = struct.unpack_from('<Q', raw, pos)[0]; pos += 8
    n_kv      = struct.unpack_from('<Q', raw, pos)[0]; pos += 8

    def skip_value(vt):
        nonlocal pos
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(vt)
        if sz is not None:
            pos += sz; return
        if vt == 8:
            n = struct.unpack_from('<Q', raw, pos)[0]; pos += 8 + n; return
        if vt == 9:
            et = struct.unpack_from('<I', raw, pos)[0]; pos += 4
            n  = struct.unpack_from('<Q', raw, pos)[0]; pos += 8
            for _ in range(n): skip_value(et)
            return
        raise ValueError(f"bad vtype {vt}")

    for _ in range(n_kv):
        klen = struct.unpack_from('<Q', raw, pos)[0]; pos += 8 + klen
        vt   = struct.unpack_from('<I', raw, pos)[0]; pos += 4
        skip_value(vt)

    body_start = None
    align = 32
    target = None
    for i in range(n_tensors):
        nlen = struct.unpack_from('<Q', raw, pos)[0]; pos += 8
        name = raw[pos:pos+nlen].decode(); pos += nlen
        ndims = struct.unpack_from('<I', raw, pos)[0]; pos += 4
        dims = []
        for _ in range(ndims):
            dims.append(struct.unpack_from('<Q', raw, pos)[0])
            pos += 8
        ttype = struct.unpack_from('<I', raw, pos)[0]; pos += 4
        offset = struct.unpack_from('<Q', raw, pos)[0]; pos += 8
        nelem = 1
        for d in dims: nelem *= d
        if ttype == 12 and target is None:
            target = (name, offset, nelem, dims)

    body_start = (pos + align - 1) & ~(align - 1)
    if target is None:
        return None
    name, offset, nelem, dims = target
    abs_off = body_start + offset
    n_blocks = nelem // 256
    return name, abs_off, n_blocks, dims, raw


def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / "Desktop/Qwen3.5-0.8B-hf/gguf-test/tinyllama-1.1b-chat-q4km.gguf")
    out_dir = Path("/tmp/q4k_test")
    out_dir.mkdir(exist_ok=True)

    info = find_q4k_tensor(gguf_path)
    if info is None:
        print("No Q4_K tensor found"); return
    name, abs_off, n_blocks, dims, raw = info
    print(f"First Q4_K tensor: {name}  dims={dims}  blocks={n_blocks}")
    print(f"  abs_off={abs_off}")

    n_test = 4
    block_bytes_total = n_test * 144
    blocks = raw[abs_off : abs_off + block_bytes_total]
    (out_dir / "blocks.bin").write_bytes(blocks)
    print(f"  wrote {n_test} blocks ({block_bytes_total} bytes) -> /tmp/q4k_test/blocks.bin")

    out_floats = []
    for i in range(n_test):
        b = blocks[i * 144 : (i + 1) * 144]
        out_floats.extend(dequant_block(b))
    assert len(out_floats) == n_test * 256

    fp32_bytes = struct.pack(f'<{len(out_floats)}f', *out_floats)
    (out_dir / "expected.f32").write_bytes(fp32_bytes)
    print(f"  wrote {len(out_floats)} fp32 values -> /tmp/q4k_test/expected.f32")
    print()
    print("First 8 expected values:", out_floats[:8])
    print("Min/max:", min(out_floats), max(out_floats))


if __name__ == '__main__':
    main()
