#!/usr/bin/env python3
"""
GGUF file inspector. Read header + KV metadata + tensor index.
Used to validate our understanding of the format before writing
the C reader.

GGUF v3 format:
  magic       : "GGUF" (4 bytes)
  version     : u32 (= 3)
  tensor_cnt  : u64
  kv_cnt      : u64
  kv_pairs    : kv_cnt * (string key, type, value)
  tensors     : tensor_cnt * (string name, u32 n_dims, dims[n_dims], type, offset)
  body        : (aligned to alignment specified in metadata)

Types (subset we care about):
  0  = F32      (4 bytes per element)
  1  = F16      (2 bytes)
  2  = Q4_0     (block of 32 elements: 1 fp16 scale + 16 packed bytes)
  3  = Q4_1     (block of 32: 1 fp16 scale + 1 fp16 min + 16 bytes)
  6  = Q5_0     (block of 32: 1 fp16 scale + 4 high-bits + 16 packed bytes)
  7  = Q5_1     (block of 32: 1 fp16 scale + 1 fp16 min + 4 high + 16 bytes)
  8  = Q8_0     (block of 32: 1 fp16 scale + 32 int8)
  10 = Q2_K     (super-block of 256 elements: 16 quants per sub-block)
  11 = Q3_K     (super-block of 256, 3.4 bit/elem)
  12 = Q4_K     (super-block of 256, 4.5 bit/elem)
  13 = Q5_K     (super-block of 256, 5.5 bit/elem)
  14 = Q6_K     (super-block of 256, 6.5625 bit/elem)
  15 = Q8_K     (super-block of 256, 8.5 bit/elem, used for intermediate)
  17 = IQ2_XXS  ... (i-quants, super-blocks of 256)
  ...
"""

import struct
import sys
from pathlib import Path

GGML_TYPE = {
    0: 'F32', 1: 'F16',
    2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
    10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
    16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL',
    21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32',
    27: 'I64', 28: 'F64',
    30: 'BF16',
}
GGML_BYTES_PER_BLOCK = {
    'F32': (4, 1), 'F16': (2, 1), 'BF16': (2, 1),
    'Q4_0': (18, 32), 'Q4_1': (20, 32),
    'Q5_0': (22, 32), 'Q5_1': (24, 32),
    'Q8_0': (34, 32), 'Q8_1': (40, 32),
    'Q2_K': (84, 256),
    'Q3_K': (110, 256),
    'Q4_K': (144, 256),
    'Q5_K': (176, 256),
    'Q6_K': (210, 256),
    'Q8_K': (292, 256),
    'IQ4_NL': (18, 32),
    'IQ4_XS': (136, 256),
}


GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def u32(self):
        v = struct.unpack_from('<I', self.buf, self.pos)[0]; self.pos += 4; return v
    def u64(self):
        v = struct.unpack_from('<Q', self.buf, self.pos)[0]; self.pos += 8; return v
    def i32(self):
        v = struct.unpack_from('<i', self.buf, self.pos)[0]; self.pos += 4; return v
    def i64(self):
        v = struct.unpack_from('<q', self.buf, self.pos)[0]; self.pos += 8; return v
    def f32(self):
        v = struct.unpack_from('<f', self.buf, self.pos)[0]; self.pos += 4; return v
    def f64(self):
        v = struct.unpack_from('<d', self.buf, self.pos)[0]; self.pos += 8; return v
    def u8(self):
        v = self.buf[self.pos]; self.pos += 1; return v

    def string(self):
        n = self.u64()
        s = self.buf[self.pos:self.pos+n].decode('utf-8', errors='replace')
        self.pos += n
        return s

    def value(self, vtype=None):
        if vtype is None:
            vtype = self.u32()
        if vtype == GGUF_TYPE_UINT8:   return self.u8()
        if vtype == GGUF_TYPE_INT8:    v = struct.unpack_from('<b', self.buf, self.pos)[0]; self.pos += 1; return v
        if vtype == GGUF_TYPE_UINT16:  v = struct.unpack_from('<H', self.buf, self.pos)[0]; self.pos += 2; return v
        if vtype == GGUF_TYPE_INT16:   v = struct.unpack_from('<h', self.buf, self.pos)[0]; self.pos += 2; return v
        if vtype == GGUF_TYPE_UINT32:  return self.u32()
        if vtype == GGUF_TYPE_INT32:   return self.i32()
        if vtype == GGUF_TYPE_FLOAT32: return self.f32()
        if vtype == GGUF_TYPE_BOOL:    v = self.u8(); return bool(v)
        if vtype == GGUF_TYPE_STRING:  return self.string()
        if vtype == GGUF_TYPE_UINT64:  return self.u64()
        if vtype == GGUF_TYPE_INT64:   return self.i64()
        if vtype == GGUF_TYPE_FLOAT64: return self.f64()
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            return [self.value(elem_type) for _ in range(n)]
        raise ValueError(f"unknown gguf type {vtype}")


def main():
    path = Path(sys.argv[1])
    print(f"Inspecting {path}  ({path.stat().st_size / 1e9:.2f} GB)")
    print()

    with open(path, 'rb') as f:
        header_buf = f.read(64 * 1024 * 1024)

    r = Reader(header_buf)
    magic = r.buf[r.pos:r.pos+4]
    r.pos += 4
    if magic != b'GGUF':
        print(f"!! not a GGUF file (magic={magic})")
        return
    version = r.u32()
    tensor_cnt = r.u64()
    kv_cnt = r.u64()
    print(f"GGUF v{version}  tensors={tensor_cnt}  metadata_pairs={kv_cnt}")

    metadata = {}
    for i in range(kv_cnt):
        key = r.string()
        val = r.value()
        metadata[key] = val

    print("\n=== Architecture metadata ===")
    for k in sorted(metadata):
        if k.startswith('general.'):
            v = metadata[k]
            if isinstance(v, list) and len(v) > 5:
                print(f"  {k}: <list of {len(v)} items>")
            else:
                print(f"  {k}: {v}")

    print("\n=== Model dimensions ===")
    arch = metadata.get('general.architecture', '?')
    keys_of_interest = [
        f'{arch}.context_length',
        f'{arch}.block_count',
        f'{arch}.embedding_length',
        f'{arch}.feed_forward_length',
        f'{arch}.attention.head_count',
        f'{arch}.attention.head_count_kv',
        f'{arch}.attention.layer_norm_rms_epsilon',
        f'{arch}.rope.freq_base',
        f'{arch}.rope.dimension_count',
        f'{arch}.vocab_size',
        'tokenizer.ggml.model',
    ]
    for k in keys_of_interest:
        v = metadata.get(k, '?')
        print(f"  {k}: {v}")

    print("\n=== Tensor types ===")
    type_counts = {}
    type_bytes = {}
    type_floats = {}
    sample_tensors = []
    for i in range(tensor_cnt):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        ttype_id = r.u32()
        offset = r.u64()
        ttype = GGML_TYPE.get(ttype_id, f'?{ttype_id}')
        nelem = 1
        for d in dims: nelem *= d
        bpb, ebpb = GGML_BYTES_PER_BLOCK.get(ttype, (None, None))
        if bpb:
            nbytes = (nelem // ebpb) * bpb
        else:
            nbytes = -1
        type_counts[ttype] = type_counts.get(ttype, 0) + 1
        type_bytes[ttype] = type_bytes.get(ttype, 0) + max(nbytes, 0)
        type_floats[ttype] = type_floats.get(ttype, 0) + nelem
        if i < 5 or 'token_embd' in name or 'output' in name or name.startswith('blk.0.'):
            sample_tensors.append((name, dims, ttype, offset, nbytes))

    print(f"  Distinct types: {len(type_counts)}")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        nb = type_bytes[t]
        nf = type_floats[t]
        print(f"    {t}: {c} tensors, {nb/1e9:.2f} GB, {nf/1e9:.2f}G floats")

    print("\n=== Sample tensors ===")
    for name, dims, ttype, offset, nbytes in sample_tensors[:30]:
        print(f"  {name}: dims={dims} type={ttype} off={offset} bytes={nbytes}")


if __name__ == '__main__':
    main()
