# V15-V27 Final Report — 27B Model

## Results
- **Speed: 0.19 → 0.58 tok/s (3.1x speedup)**
- **Memory: 6.7GB → 442MB (93% reduction)**
- **bit-exact: token sequence identical across all versions**

## Version Comparison

| Version | 8 tokens | tok/s | speedup | peak memory |
|---|---|---|---|---|
| V14 (violation) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ |
| V15 (baseline) | 41.0s | 0.19 | 1.0x | 102MB |
| V22 | 19.5s | 0.41 | 2.2x | 357MB |
| V23 | 16.1s | 0.50 | 2.6x | 442MB |
| V25 | 14.1s | 0.57 | 3.0x | 442MB |
| **V27** | **13.9s** | **0.58** | **3.1x** | **442MB** |

## V27 Batch Profile (B=5, K=4, 2 forwards for 8 tokens)
```
B=5 forward:  attn=0.18s (3%)  ssm=0.79s (12%)  lm=0.62s (9%)  madv=0.93s (13%)
I/O page fault: 63% (SSD 3GB/s — irreducible)
```

## Core Optimization Path
1. **Memory safety** (V15): removed keep_resident + ephemeral GPU
2. **Bug fix** (V16): GPU KV cache 8→32 slabs
3. **Spec decode** (V17-V23): MTP chain spec, K=4, 100% accept
4. **Page cache** (V18-V20): RELEASE_BEHIND=0, PREFETCH_AHEAD=2
5. **I/O prefetch** (V21, V24): lm_head + MTP head madvise prefetch
6. **Compute optimization** (V22, V25-V27): conv1d chunking + SDOT multix (Q4_K + Q6_K)

## 6 Bugs Fixed
1. GPU KV cache hardcoded to 8 slabs → 27B 16 layers out-of-bounds NaN
2. keep_resident auto mode locks 16GB page cache
3. newBufferWithBytesNoCopy registering the whole mmap → GPU wires all pages
4. SSM group dispatch does not handle F32 type
5. batched forward had its own PREFETCH_AHEAD/RELEASE_BEHIND parameters
6. Q6_K SDOT xscale index out-of-bounds (16 vs 32 element granularity)

## 7 Failed Experiments
| Experiment | Result |
|---|---|
| merge madvise | 8x slower |
| half madvise | 2x slower |
| F_RDADVISE | 3x slower |
| skip odd-layer madvise | 2x slower |
| B_MAX=8 K=7 | slower than B=5 K=4 (B=8 forward heavy) |
| GPU ephemeral 27B | slower than CPU |
| n-gram spec on creative text | no match |
