# V15-V25 Version Iteration Comparison Report — 27B Model

## Test Environment
- Model: Qwen3.6-27B Q4_K_M (16GB GGUF)
- Machine: Apple M4 Pro, 24GB RAM
- Test: 8-token generation, prompt [1, 12968]

## Version Comparison Summary

| Version | 8 tokens | tok/s | speedup | peak memory | bit-exact |
|---|---|---|---|---|---|
| V14 (violation) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | ✅ |
| V15 (baseline) | 41.0s | 0.19 | 1.0x | 102MB ✅ | ✅ |
| V17 | 26.1s | 0.31 | 1.6x | ~600MB ✅ | ✅ |
| V18 | 23.3s | 0.34 | 1.8x | 697MB ✅ | ✅ |
| V19 | 22.8s | 0.35 | 1.8x | 356MB ✅ | ✅ |
| V20 | 21.1s | 0.38 | 2.0x | 357MB ✅ | ✅ |
| V21 | 20.2s | 0.40 | 2.1x | 357MB ✅ | ✅ |
| V22 | 19.5s | 0.41 | 2.2x | 357MB ✅ | ✅ |
| V23 | 16.1s | 0.50 | 2.6x | 442MB ✅ | ✅ |
| V24 | 15.5s | 0.52 | 2.7x | 442MB ✅ | ✅ |
| **V25** | **14.1s** | **0.57** | **3.0x** | **442MB** ✅ | **✅** |

## V25 Batch Profile (B=5 spec step)
```
attn=0.19s (3%)   ssm=0.83s (12%)   lm=0.64s (9%)   madv=0.91s (13%)
I/O page fault: 63% (SSD 3GB/s bottleneck)
Total compute: 2.57s (was 3.57s in V24, 28% faster)
```

## Per-Version Core Changes

| Version | Change | speed | memory |
|---|---|---|---|
| V15 | removed memory bombs (keep_resident + NoCopy) | baseline | 6.7GB→102MB |
| V16 | fixed GPU KV-cache out-of-bounds | — | — |
| V17 | n-gram spec + MTP spec decode | +63% | +498MB |
| V18 | page-cache optimization (RELEASE_BEHIND=0) | +10% | +101MB |
| V19 | B_MAX 8→4 | +3% | -341MB |
| V20 | PREFETCH_AHEAD 1→2 | +9% | 0 |
| V21 | lm_head madvise + batched-madvise fix | +5% | 0 |
| V22 | conv1d dispatch chunking | +3% | 0 |
| V23 | B_MAX=5 K=4 (100% accept step 1) | +22% | +85MB |
| V24 | MTP head + lm_head prefetch | +4% | 0 |
| **V25** | **SDOT multix (int8 dotprod)** | **+10%** | **0** |

## Bugs Fixed (6)
1. GPU KV cache hardcoded to 8 slabs → 27B 16 layers out-of-bounds NaN
2. keep_resident auto mode locks 16GB page cache
3. newBufferWithBytesNoCopy registering the whole mmap → GPU wires all pages
4. SSM group dispatch does not handle F32 type
5. batched forward had its own PREFETCH_AHEAD/RELEASE_BEHIND parameters
6. n-gram spec malloc'd 52MB every time

## Failed Experiments (7)
| Experiment | Result |
|---|---|
| merge 2 layers' madvise | 8x slower |
| half madvise | 2x slower |
| F_RDADVISE instead of madvise | 3x slower |
| skip odd-layer madvise | 2x slower |
| B_MAX=6 K=5 | same speed +85MB |
| GPU ephemeral for 27B | slower than CPU |
| n-gram spec on creative text | no match |

## Performance Progress Chart
```
tok/s
 0.57 |                                                        ★ V25
 0.50 |                                          ★ V23 ★ V24
 0.45 |
 0.40 |                                    ★ V21 ★ V22
 0.35 |                          ★ V19 ★ V20
 0.30 |               ★ V17
 0.25 |
 0.20 | ★ V15
 0.15 |
 0.10 |
 0.00 +--V15--V16--V17--V18--V19--V20--V21--V22--V23--V24--V25-->
```
