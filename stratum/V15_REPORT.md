# V15 Comparison Report — 27B Model Low-Memory GPU-Safe Mode

## Core Problem

The `keep_resident` auto mode introduced in V13, together with whole-model `newBufferWithBytesNoCopy` registration, causes the GPU to **wire (lock) all physical pages** when it touches mmap pages — effectively an implicit mlock of the 16GB model — which exhausts system memory.

## V15 Fixes

| Fix | Description |
|---|---|
| Removed `keep_resident` auto mode | No longer locks page cache when model < 75% RAM |
| GPU init no longer registers the whole mmap | Passes `NULL, 0`, loads kernels only |
| Ephemeral buffer (copy semantics) | `newBufferWithBytes` per-tensor copy; does not wire mmap pages |
| Large tensors skip GPU | N >= 50000 (e.g. 1GB lm_head) goes through CPU NEON |
| full_attn GPU disabled by default | Ephemeral copy has precision issues on 27B; to be fixed |

## 27B Q4_K_M Test Results (1-token generation)

| Version | Mode | token time | wired (anon) | peak footprint | bit-exact | memory-safe |
|---|---|---|---|---|---|---|
| V14 | CPU + keep_resident | **4.74s** | 87MB | ~6.7GB RSS | ✅ argmax=16 | ❌ locks 16GB page cache |
| V15 | CPU (no GPU) | **5.13s** | 59MB | **102MB** | ✅ argmax=16 | ✅ |
| V15 | SSM GPU only | 5.81s | 93MB | 353MB | ✅ argmax=16 | ✅ |
| V15 | full_attn GPU | — | — | — | ❌ NaN | precision issue |

## Comparison Analysis

### V15 vs V14 speed
- V15 CPU: 5.13s vs V14 CPU: 4.74s → **8% slower**
- Cause: after removing keep_resident, the OS reclaims page cache and weights must be re-read from SSD
- This is the **correct cost** — V14's speed was bought by locking 16GB of RAM

### V15 SSM GPU vs CPU
- SSM GPU: 5.81s vs CPU: 5.13s → **GPU 13% slower**
- Cause: SSD I/O (3GB/s) is the bottleneck, not compute. GPU ephemeral copy adds overhead without reducing I/O
- GPU compute is fast (270GB/s), but weights still must be read from SSD into page cache

### Memory safety
- V15 peak footprint = **102MB** (activations + KV cache + SSM state only)
- V14 RSS = **6.7GB** (16GB model wired in physical RAM)
- V15 fully compliant: no wire, no mlock, no page-cache lock

## Bottleneck Analysis

27B model 5.13s/token time breakdown:
- matmul: 1.78s (35%) — CPU NEON reads weights from mmap
- other: 3.44s (67%) — **mostly SSD I/O page-fault waiting**
- attn: 0.15s (3%) — 16 full-attention layers
- ssm: 0.53s (10%) — 48-layer SSM recursion
- lm_head: 1.01s (20%) — 248320×5120 large matrix

**The core bottleneck is SSD I/O (3GB/s): the 16GB model must be read in full every token.**

## Next-Version Direction (V16)

1. **Background pread prefetch**: while computing the current layer, a background thread preads the next layer's weights into page cache
2. **Inter-layer weight reuse**: if page cache is not reclaimed, the second token can skip SSD reads
3. **Fix full_attn GPU precision**: investigate the ephemeral-copy precision issue in full-attn layers
4. **Overlap GPU compute with I/O**: while the GPU computes the current tensor, the CPU preads the next
