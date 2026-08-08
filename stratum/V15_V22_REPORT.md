# V15-V22 Version Iteration Comparison Report — 27B Model

## Test Environment
- Model: Qwen3.6-27B Q4_K_M (16GB GGUF)
- Machine: Apple M4 Pro, 24GB RAM
- Test: 8-token generation, prompt [1, 12968]

## Version Comparison Summary

| Version | 8-token total | tok/s | speedup | peak memory | bit-exact |
|---|---|---|---|---|---|
| V14 (violation) | 37.9s | 0.21 | 1.1x | ~6.7GB ❌ | ✅ |
| V15 (baseline) | 41.0s | 0.19 | 1.0x | 102MB ✅ | ✅ |
| V17 (MTP K=3) | 26.1s | 0.31 | 1.6x | ~600MB ✅ | ✅ |
| V18 (K=7 B=8) | 23.3s | 0.34 | 1.8x | 697MB ✅ | ✅ |
| V19 (B_MAX=4) | 22.8s | 0.35 | 1.8x | 356MB ✅ | ✅ |
| V20 (PF=2) | 21.1s | 0.38 | 2.0x | 357MB ✅ | ✅ |
| V21 (lm_head pf) | 20.2s | 0.40 | 2.1x | 357MB ✅ | ✅ |
| **V22 (conv1d chunk)** | **19.5s** | **0.41** | **2.2x** | **357MB** ✅ | **✅** |

## V22 Batch Profile (B=4 spec step)
```
attn=0.29s (4%)   — 16 full-attn layers
ssm=1.05s (15%)   — 48 SSM layers (conv1d + recursion)
lm=0.61s (9%)     — lm_head (1GB Q6_K)
madv=0.91s (13%)  — madvise prefetch system calls
I/O wait=59%      — SSD page-fault latency (irreducible)
```

## Per-Version Improvements

| Version | Change | speed | memory |
|---|---|---|---|
| V15 | removed keep_resident + ephemeral GPU | baseline | 6.7GB→102MB |
| V16 | fixed GPU KV-cache out-of-bounds (8→32 slab) | — | — |
| V17 | n-gram spec + MTP validation | +63% | +498MB |
| V18 | RELEASE_BEHIND=0, PREFETCH_AHEAD=1 | +10% | +101MB |
| V19 | B_MAX 8→4 | +3% | -341MB |
| V20 | PREFETCH_AHEAD 1→2 | +9% | 0 |
| V21 | lm_head madvise prefetch + batched-madvise fix | +5% | 0 |
| V22 | conv1d dispatch chunking (10240→160 tasks) | +3% | 0 |

## Failed Experiments
- Merging 2 layers' madvise into 1 call: 8x slower (kernel readahead 500MB → synchronous block)
- Half madvise (prefetch only first half of layer): 2x slower (second half page-faults)
- GPU ephemeral for 27B: slower than CPU (copy overhead > compute savings, SSD I/O unchanged)
- n-gram spec on creative text: no match (needs repetitive text)

## Bugs Fixed
1. GPU KV cache hardcoded to 8 slabs; 27B needs 16 → out-of-bounds NaN
2. keep_resident auto mode locks 16GB page cache = implicit mlock
3. newBufferWithBytesNoCopy registering the whole mmap → GPU wires all physical pages
4. SSM group dispatch does not handle F32 type → GPU fallback fails
5. batched forward had its own PREFETCH_AHEAD=4/RELEASE_BEHIND=2 (not using V18's optimized values)
6. n-gram spec malloc'd 52MB every time → changed to pre-allocated reuse
