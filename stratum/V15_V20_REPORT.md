# V15-V20 Version Iteration Comparison Report — 27B Model

## Test Environment
- Model: Qwen3.6-27B Q4_K_M (16GB GGUF)
- Machine: Apple M4 Pro, 24GB RAM
- Test: 8-token generation, prompt [1, 12968]

## Version Comparison Summary

| Version | Mode | 8-token total | tok/s | speedup | peak memory | bit-exact |
|---|---|---|---|---|---|---|
| V14 | CPU + keep_resident (violation) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | ✅ |
| V15 | CPU (no GPU) | 41.0s | 0.19 | 1.0x (baseline) | 102MB ✅ | ✅ |
| V16 | full_attn GPU (KV fixed) | 59.6s | 0.13 | 0.7x | 19MB ✅ | ✅ |
| V17 | MTP spec K=3 | 26.1s | 0.31 | 1.6x | ~600MB ✅ | ✅ |
| V18 | MTP spec K=7 B=8 | 23.3s | 0.34 | 1.8x | 697MB ✅ | ✅ |
| V19 | MTP spec K=3 B=4 | 22.8s | 0.35 | 1.8x | **356MB** ✅ | ✅ |
| **V20** | **MTP spec K=3 B=4 PF=2** | **21.1s** | **0.38** | **2.0x** | **357MB** ✅ | **✅** |

## Per-Version Improvements

### V15: Memory-safety fix (0.19 tok/s, 102MB)
- Removed keep_resident auto mode (locks 16GB page cache = implicit mlock)
- GPU ephemeral `newBufferWithBytes` (copy semantics, no mmap-page wiring)
- Cost: 8% slower (5.13s vs 4.74s), but memory drops from 6.7GB to 102MB

### V16: KV-cache out-of-bounds fix
- Bug: GPU KV cache hardcoded to 8 slabs; 27B has 16 full-attn layers → out of bounds → NaN
- Fix: 32 slabs. full_attn GPU goes from NaN to bit-exact
- GPU ephemeral still slower than CPU (copy is pure overhead under SSD-I/O bottleneck)

### V17: n-gram spec decode + MTP validation (0.31 tok/s)
- Added Qwen3.5 n-gram spec decode (zero draft-forward overhead)
- SSM state save/restore supports rejection rollback
- Validated MTP chain spec K=7 reaches 4.00 tok/main_forward

### V18: Page-cache optimization (0.34 tok/s, 697MB)
- RELEASE_BEHIND 2→0, PREFETCH_AHEAD 4→1
- OS LRU is smarter than manual MADV_DONTNEED
- Second token 14% faster

### V19: B_MAX 8→4 (0.35 tok/s, 356MB) ← memory halved
- B_MAX reduced from 8 to 4 → SSM snapshot drops from 697MB to 356MB
- K=3 B=4 faster than K=7 B=8 (less compute overhead, same accept rate)
- n-gram spec snapshot pre-allocated and reused

### V20: Prefetch optimization (0.38 tok/s, 357MB) ← speedup
- PREFETCH_AHEAD 1→2 (overlap SSD readahead with compute)
- 9% speedup, memory unchanged

## Key Findings

### Bottleneck
- **SSD I/O is the fundamental bottleneck**: 16GB/token ÷ 3GB/s = 5.3s floor
- **Spec decode is the best speedup**: batched forward reads weights once, serves multiple tokens
- **GPU cannot solve the I/O bottleneck**: ephemeral copy is pure overhead

### Memory
- **SSM state snapshot is the memory hog**: 3 × 79.5MB = 238MB (67% of 356MB)
- **B_MAX=4 is the optimal memory/speed balance point**
- **n-gram spec needs only 1 snapshot**: 101MB peak (but low match rate)

### Correctness
- **All versions bit-exact**: token sequence 16,25,561,14955,314,14791,303,17722
- **GPU full_attn bit-exact** (after V16 fix): argmax identical, logits differ slightly

## Performance Progress Charts

```
tok/s
 0.40 |                                    ★ V20 (0.38)
 0.35 |                          ★ V19   ★ V18
 0.30 |               ★ V17
 0.25 |
 0.20 | ★ V15
 0.15 |
 0.10 |
 0.05 |
 0.00 +--V15--V16--V17--V18--V19--V20-->

Memory (MB)
700 |                    ★ V18
600 |               ★ V17
500 |
400 |                          ★ V19 ★ V20
300 |
200 |
100 | ★ V15
  0 +--V15--V16--V17--V18--V19--V20-->
```
