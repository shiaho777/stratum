# V15-V23 Version Iteration Comparison Report — 27B Model

## Test Environment
- Model: Qwen3.6-27B Q4_K_M (16GB GGUF)
- Machine: Apple M4 Pro, 24GB RAM
- Test: 8-token generation, prompt [1, 12968]

## Version Comparison Summary

| Version | 8 tokens | tok/s | speedup | peak memory | forwards | bit-exact |
|---|---|---|---|---|---|---|
| V14 (violation) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | 8 | ✅ |
| V15 (baseline) | 41.0s | 0.19 | 1.0x | 102MB ✅ | 8 | ✅ |
| V17 | 26.1s | 0.31 | 1.6x | ~600MB ✅ | 3 | ✅ |
| V18 | 23.3s | 0.34 | 1.8x | 697MB ✅ | 2 | ✅ |
| V19 | 22.8s | 0.35 | 1.8x | 356MB ✅ | 3 | ✅ |
| V20 | 21.1s | 0.38 | 2.0x | 357MB ✅ | 3 | ✅ |
| V21 | 20.2s | 0.40 | 2.1x | 357MB ✅ | 3 | ✅ |
| V22 | 19.5s | 0.41 | 2.2x | 357MB ✅ | 3 | ✅ |
| **V23** | **16.0s** | **0.50** | **2.6x** | **442MB** ✅ | **2** | **✅** |

## V23 Core Breakthrough

**B_MAX 5→K=4**: MTP draft first step accept=4/4 (100%!), second step accept=2/2

```
spec-step  in=220  drafts=[16 25 561 14955]  reals=[16 25 561 14955 314]  accept=4/4  emit=5
spec-step  in=314  drafts=[14791 303]         reals=[14791 303 17722]      accept=2/2  emit=3
→ 8 tokens in 2 main forwards, 4.00 tok/main
```

## V22 Batch Profile (B=4 spec step)
```
attn=0.29s (4%)   ssm=1.05s (15%)   lm=0.61s (9%)   madv=0.91s (13%)
I/O page fault: 59% (SSD 3GB/s bottleneck, irreducible)
```

## Failed-Experiment Log
| Experiment | Result | Cause |
|---|---|---|
| merge 2 layers' madvise | 8x slower | kernel readahead 500MB synchronously blocks |
| half madvise | 2x slower | second half of layer page-faults |
| F_RDADVISE instead of madvise | 3x slower | no page marking; LRU does not retain |
| skip odd-layer madvise | 2x slower | odd layers not prefetched |
| B_MAX=6 K=5 | same speed +85MB | emits 6 vs 5, forward count unchanged |

## Bugs Fixed
1. GPU KV cache hardcoded to 8 slabs → 27B 16 layers out-of-bounds NaN
2. keep_resident auto mode locks 16GB page cache
3. newBufferWithBytesNoCopy registering the whole mmap → GPU wires all pages
4. SSM group dispatch does not handle F32 type
5. batched forward had its own PREFETCH_AHEAD=4/RELEASE_BEHIND=2
6. n-gram spec malloc'd 52MB every time

## Performance Progress Charts
```
tok/s
 0.50 |                                              ★ V23
 0.45 |
 0.40 |                                    ★ V21 ★ V22
 0.35 |                          ★ V19 ★ V20
 0.30 |               ★ V17
 0.25 |
 0.20 | ★ V15
 0.15 |
 0.10 |
 0.05 |
 0.00 +--V15--V16--V17--V18--V19--V20--V21--V22--V23-->

Memory (MB)
700 |                    ★ V18
600 |               ★ V17
500 |                                              ★ V23
400 |                          ★ V19 ★ V20 ★ V21 ★ V22
300 |
200 |
100 | ★ V15
  0 +--V15--V16--V17--V18--V19--V20--V21--V22--V23-->
```
