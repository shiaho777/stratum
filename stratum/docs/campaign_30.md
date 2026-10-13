# Campaign log — thirty measured rounds

Each round names one next action chosen after the previous round's captured output (round 1 from the in-repo bottleneck notes), one target metric, and a cross-check. A round is recorded only when a command's output shows a behavior change or a falsified hypothesis. Scratch captures are the files named below in the campaign scratch directory. Commands use `$SCRATCH` for that directory and `$ENGINE` for `stratum/native/stratum`. No 27B run, no mlock, no whole-model NoCopy, no multi-GB allocation.

Bottleneck notes this campaign starts from (in-repo, not re-measured on a large model): single-stream large-model decode is weight-scan bound; hybrid full attention past 4096 drops middle tokens; CPU full attention is a two-pass score scan.

## Round 1 — short tiny-hybrid prompt does not reach the 4096 cliff

- Next action: On a generated tiny qwen35-hybrid, ask whether a short decode shows the long-context attendance cliff. A 27B weight-scan run is refused under the machine-safety rule, so the safe reading of the bottleneck notes is the 4096 attendance claim.
- Why this step: The three bottleneck notes are the only prior evidence. The attendance cliff is the one that a tiny model can touch without a multi-GB weight file.
- Target metric: long-context attendance (tokens visible to full attention). Wall time is recorded only as context.
- Hypothesis: `./stratum` on the tiny hybrid, 16 new tokens, will print a ring/truncation signal or a KV window smaller than the tokens written.
- Command: `STRATUM_NO_GPU=1 /usr/bin/time -p $ENGINE $SCRATCH/tinyh.gguf 16 1 2 3 4 5 6 7`
- Result: **FALSIFIED.** No ring message. KV line is `KV cache (full-attn, lazy-anon): 1.0 MB anon (calloc, lazy)`. 16 steps print. `real 0.15`. Greedy argmax `145 96 131 82 122 75 160 157 205 116 38 46 195 248 155 220` after prefill sample 48. The run also prints `parallel matmul: 14 chunks (14 physical cores)` and `anchored 135 MB` with `lm-stage: 2 x 64 MB`. Context stays far under 4096, so short-prompt tok/s is not the attendance metric.
- Evidence: `r1_hybrid.txt`
- Cross-check: the same capture contains both the 1.0 MB KV line and 16 `stratum_argmax` lines, and it does not contain `exceeds` or a sink-ring drop. The 14-chunk line is the fact the next round measures.
- Kept code: none.

## Round 2 — small-model matmul chunks 14 → 5

- Next action: Stop overriding `g_st.nchunks` to every physical core when the mmap is at most 2GB. `STRATUM_NCHUNKS` still wins. An mmap above 2GB still uses the physical-core count (that path is not measured here and is not changed).
- Why this step: Round 1 printed `parallel matmul: 14 chunks` on a model far under 2GB. `stratum_linear.h` already uses 5 chunks for mmap ≤ 2GB, because about five threads saturate hot bandwidth.
- Target metric: single-stream wall time on `tinyh.gguf`, 8 generated tokens, prompt ids `1 2 3 4`.
- Hypothesis: default 5 chunks has a lower warm wall than `STRATUM_NCHUNKS=14` on this input, and the greedy argmax stays the same.
- Commands (warm process, Python `perf_counter` around the binary, `STRATUM_NO_GPU=1`):
  - baseline, forced old policy: `STRATUM_NCHUNKS=14 $ENGINE $SCRATCH/tinyh.gguf 8 1 2 3 4`
  - shipped default: `$ENGINE $SCRATCH/tinyh.gguf 8 1 2 3 4`
- Result: **KEEP.**
  - 14 chunks warm wall: 0.026476s, 0.020518s, 0.030282s. Line: `parallel matmul: 14 chunks`.
  - 5 chunks warm wall: 0.013956s, 0.014458s, 0.014942s. Line: `parallel matmul: 5 chunks`.
  - Both print argmax `136 62 122 62 69 62 240 248` and anchored anon 135 MB.
  - About 8/0.0145s ≈ 550 tok/s versus 8/0.026s ≈ 310 tok/s, startup included.
- Evidence: `r2_warm_default_rep1.txt`, `r2_warm_default_rep2.txt`, `r2_warm_default_rep3.txt`, `r2_warm_nc14_rep1.txt`, `r2_warm_nc14_rep2.txt`, `r2_warm_nc14_rep3.txt`
- Cross-check: three reps each way, argmax identical, the chunks line is the only scheduling difference. The before number stays runnable on the shipped binary with `STRATUM_NCHUNKS=14`.
- Kept code: the ≤2GB default in `stratum_arch_qwen35.inc.c`.

## Round 3 — clamp the lm-stage window to a smaller tensor

- Next action: The stage buffer is 64MB×2 even when the lm_head tensor is far smaller. First measure `STRATUM_LM_STAGE_MB=16` against the default 64MB window. If wall and anchored anon drop and the argmax holds, clamp the window to the page-aligned tensor when the tensor is smaller than the window. A tensor at or above the window keeps 64MB.
- Why this step: Round 2's faster 5-chunk runs still print `lm-stage: 2 x 64 MB` and `anchored 135 MB`. That anonymous memset is not weight cache and it dominates startup on this tiny input.
- Target metric: overall wall time and anchored anonymous memory on the same 8-token tiny-hybrid command.
- Hypothesis: a smaller stage lowers both numbers and does not change the greedy ids.
- Commands (`STRATUM_NO_GPU=1`, same prompt):
  - `STRATUM_LM_STAGE_MB=64` (the pre-clamp default window)
  - `STRATUM_LM_STAGE_MB=16` (env probe, before the code clamp)
  - shipped clamp, `STRATUM_LM_STAGE_MB` unset
- Result: **KEEP.**
  - 64MB window: wall 0.013991s, 0.013159s; anchored 135 MB; argmax unchanged.
  - 16MB env: wall 0.008633s, 0.008939s; anchored 39 MB; argmax unchanged.
  - shipped clamp: wall 0.008047s, 0.006692s, 0.007455s; anchored 7 MB; argmax `136 62 122 62 69 62 240 248`. The binary that produced this capture printed `lm-stage: 2 x 0 MB` because the format was `%.0f` (32KB rounds to 0). The format is now `%.3f`; the byte clamp is the behavior change.
- Evidence: `r3_stage64_rep1.txt`, `r3_stage64_rep2.txt`, `r3_stage16_rep1.txt`, `r3_stage16_rep2.txt`, `r3_after_default_rep1.txt`, `r3_after_default_rep2.txt`, `r3_after_default_rep3.txt`
- Cross-check: argmax matches round 2 on the same ids. Anchored anon moved 135 → 7 in the same `[mem after prefill]` line. The 27B-sized window is unchanged when the tensor is larger than the cap.
- Kept code: `q35_lm_stage_init` tensor clamp.
- Next action this result selects: short prompts never reach 4096, so the attendance cliff is still unmeasured. The following round measures full-prefix visibility on the shipped predicate, then checks the short-prompt argmax on the rebuilt engine.

## Round 4 — full prefix inside capacity, hard stop past it

- Next action: Replace the qwen35 sink+ring with a full prefix while `last_written < capacity`, and refuse the step past capacity. Default capacity stays 4096 when the GGUF has no `context_length`. `STRATUM_MAX_KV` overrides it (clamp 32..1<<20).
- Why this step: Rounds 1–3 never left the short-prompt regime. Round 1 showed a 16-token run cannot see the cliff, so the long-context metric had not moved.
- Target metric: long-context attendance (tokens visible, and whether token 100 survives).
- Hypothesis: the old ring at capacity 4096 and last=4999 keeps 4096 tokens and drops token 100; the shipped predicate at capacity 8192 keeps 5000 and includes token 100. On the engine, a short prompt keeps argmax `136 62 122 62 69 62 240 248`, `STRATUM_MAX_KV=8192` attends index 100 once it has been written, and `STRATUM_MAX_KV=32` exits non-zero instead of wrapping.
- Commands:
  - `$SCRATCH/bench_attn_window` (compiled from `stratum/native/bench_attn_window.c`, calls `stratum_attn_window.h`)
  - `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyh.gguf 8 1 2 3 4`
  - `STRATUM_NO_GPU=1 STRATUM_MAX_KV=8192 STRATUM_ATTN_AUDIT=1 $ENGINE $SCRATCH/tinyh.gguf 110 1 2 3 4`
  - `STRATUM_NO_GPU=1 STRATUM_MAX_KV=32 $ENGINE $SCRATCH/tinyh.gguf 40 1 2 3 4`
- Result: **KEEP.**
  - Bench: `legacy_cap4096_visible=4096 legacy_has_100=0`, `shipped_cap8192_visible=5000 shipped_has_100=1`, `shipped_short16_visible=16`, `attn_window_ok`.
  - Short engine run: `4096 positions`, argmax unchanged, `lm-stage: 2 x 0.031 MB`, anchored 7 MB, 5 chunks.
  - Audit at capacity 8192: `n_attn=100 has_100=0 kv_len=99`, then `n_attn=101 has_100=1 kv_len=100`, ending `n_attn=114 has_100=1 kv_len=113 capacity=8192`. Visible count is the full prefix. First eight argmax values match the short run.
  - Capacity 32: exit 1, `qwen35: context 33 exceeds KV capacity 32 (full prefix, no sink ring)`. Tokens that fit were produced; nothing wrapped.
- Evidence: `r4_bench.txt`, `r4_default_rep1.txt`, `r4_default_rep2.txt`, `r4_maxkv8192.txt`, `r4_maxkv32.txt`, `r4_audit110.txt`, `r4_llama_rep1.txt`, `r4_llama_rep2.txt`
- Cross-check: the same short argmax appears with capacity 4096 and 8192, logits included (`sampled = 193`, `logit=2.17287` on step 0). A temporary rebuild of `HEAD` (no attendance edit, 14 chunks, 64MB stage) on 80 tokens also reaches `logit=nan` at step 72 and the same first-eight argmax (`r4_head_80.txt`). The late NaN is this untrained tiny model, not the attendance edit. Llama on the new binary still prints `142 22 185 32 22 22 22 22` on two runs. KV virtual at 8192 positions is 2.0 MB.
- Kept code: `stratum_attn_window.h`, the qwen35 capacity path, `bench_attn_window.c`.
- Next action this result selects: llama still prints `10 chunks` (P-cores). Round 2's 5-chunk win was qwen35-only. Measure whether llama's tiny-model wall drops at 5 chunks with the same argmax.

## Round 5 — llama 5 chunks vs the default 10

- Next action: On the tiny llama, compare warm wall of the default 10 P-core chunks against `STRATUM_NCHUNKS=5`. Do not edit unless the two sets separate.
- Why this step: Round 4's llama cross-check still printed `parallel matmul: 10 chunks`. The qwen35 14→5 win does not say the llama default is wrong.
- Target metric: single-stream wall time on `tinyl.gguf`, 8 tokens, prompt `1 2 3 4`.
- Hypothesis: 5 chunks has a lower warm wall than 10, with the same argmax.
- Command: interleaved 8+8 warm runs, `STRATUM_NO_GPU=1`, with and without `STRATUM_NCHUNKS=5`.
- Result: **FALSIFIED.** Argmax is `142 22 185 32 22 22 22 22` both ways. Walls (seconds):
  - 10 chunks: 0.009268 0.007897 0.005679 0.005593 0.004813 0.004925 0.004242 0.004355 (median 0.0053)
  - 5 chunks: 0.005922 0.005773 0.005122 0.004711 0.004348 0.004146 0.003793 0.004393 (median 0.0046)
  - The warm tail overlaps (both hit 0.0044s). A few tenths of a millisecond on a 4ms process start is not a stable win. No llama edit.
- Evidence: `r5_summary.txt`, `r5_nc10_rep1.txt`, `r5_nc5_rep1.txt`
- Cross-check: both configs print their chunk count, and the argmax strings are identical across all 16 runs.
- Kept code: none.
- Next action this result selects: the llama argmax ends in repeating `22`. Measure whether `STRATUM_NGRAM_SPEC` raises tokens per forward on a longer llama run without changing the greedy sequence.

## Round 6 — n-gram spec on the repeating tiny llama

- Next action: Run 32 tokens on `tinyl.gguf` with `STRATUM_NGRAM_SPEC=4` and `STRATUM_SPEC_STATS=1`, against greedy. Keep a default change only if wall time drops and the argmax sequence matches.
- Why this step: Round 5 left the chunk count alone. The greedy sequence ends in a run of token 22, which is the case prompt-lookup speculation is built for.
- Target metric: overall wall time, with tokens per forward as the mechanism check.
- Hypothesis: speculation keeps the greedy sequence and lowers wall time, because one batched verify replaces several serial forwards.
- Commands (`STRATUM_NO_GPU=1`, prompt `1 2 3 4`, 32 tokens):
  - greedy: `$ENGINE $SCRATCH/tinyl.gguf 32 1 2 3 4`
  - spec: `STRATUM_NGRAM_SPEC=4 STRATUM_SPEC_STATS=1` on the same argv
- Result: **FALSIFIED** for wall time. Sequence matches on both reps: `142 22 185 32 22 22 22 22 22 22 22 180 22 180 22 180 22 180 180 22 180 22 180 22 180 180 180 22 180 180 180 180`. Spec reports `7 batched calls, 15 drafts accepted (3.14 tok/call)`. Wall: greedy 0.0068s and 0.0066s; spec 0.0074s and 0.0070s. Tokens per forward rose, and the wall did not. No default change.
- Evidence: `r6_greedy_rep1.txt`, `r6_greedy_rep2.txt`, `r6_ngram_rep1.txt`, `r6_ngram_rep2.txt`
- Cross-check: the two greedy runs match each other and both spec runs, token for token.
- Kept code: none.
- Next action this result selects: the verify batch did not pay for itself on this tiny model. The worker pool behind `st_par_run` still starts `nchunks-1` spinners. Measure `STRATUM_PAR_OFF=1` against the 5-chunk default on the tiny hybrid.

## Round 7 — skip the worker pool for mmaps at or under 4 MB

- Next action: Measure `STRATUM_PAR_OFF=1` against the pool on the tiny hybrid. Ship a serial path only where that measurement separates, and keep the pool for larger mmaps.
- Why this step: Round 6 showed a batched verify losing on wall time. `st_par_run` still wakes `nchunks-1` workers for every parallel call, and these models are under 1 MB.
- Target metric: single-stream wall time on the 8-token tiny prompts.
- Hypothesis: serial execution has a lower warm wall than the pool, and the greedy ids do not change.
- Commands: `STRATUM_NO_GPU=1`, 8 tokens, prompt `1 2 3 4`, interleaved. Before the edit, `STRATUM_PAR_OFF=1` versus unset. After the edit, the shipped default versus `STRATUM_PAR_FORCE=1` (pool restored).
- Result: **KEEP.**
  - Hybrid, pool vs `STRATUM_PAR_OFF`, 8 interleaved reps, argmax `136 62 122 62 69 62 240 248` both ways. Pool walls 0.0059 0.0064 0.0079 0.0061 0.0064 0.0063 0.0071 0.0056 (median 0.0064). Serial walls 0.0054 0.0049 0.0053 0.0050 0.0048 0.0050 0.0048 0.0048 (median 0.0049). No overlap.
  - Llama cross-check before the edit, 6 reps: pool median 0.0059 (min 0.0055), serial median 0.0045 (max 0.0051), argmax `142 22 185 32 22 22 22 22`.
  - Shipped rule: mmap ≤ 4 MB runs serial unless `STRATUM_PAR_FORCE=1`. Larger mmaps are unchanged. Hybrid after the edit, 6 reps: serial median 0.0055, forced pool median 0.0063, same argmax, serial runs print `small-model matmul: serial`. Llama after the edit, 6 reps: serial median 0.0044 (max 0.0047), forced pool median 0.0057, same argmax.
- Evidence: `r7_summary.txt`, `r7_def_rep1.txt`, `r7_off_rep1.txt`, `r7_ship_summary.txt`, `r7_ship_ship_rep1.txt`, `r7_ship_force_rep1.txt`, `r7_llama_summary.txt`, `r7_llama_ship.txt`, `r7_llama_force.txt`
- Cross-check: both architectures, both the env probe and the shipped switch, same argmax. `STRATUM_PAR_FORCE=1` does not print the serial line. Models above 4 MB never enter the new branch.
- Kept code: `st_par_run` in `stratum_linear.h`.
- Next action this result selects: the pool is no longer the tiny-model tax. Llama groups GQA only past `STRATUM_ATTN_GRP_MIN` (default 4096). Measure whether dropping that threshold to 1 changes the short-prompt argmax or the wall.

## Round 8 — GQA grouping at a threshold of 1

- Next action: On the tiny llama (GQA group size 2), compare `STRATUM_ATTN_GRP_MIN=1` with the default 4096 over 32 tokens.
- Why this step: Round 7 removed the pool tax. The grouped K/V stream is documented as bit-exact and as a loss below about 4096 positions, but that crossover was not measured on this binary.
- Target metric: single-stream wall time. Argmax identity is the quality check.
- Hypothesis: threshold 1 lowers wall and keeps the greedy sequence.
- Command: interleaved 6+6, `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyl.gguf 32 1 2 3 4`, with and without `STRATUM_ATTN_GRP_MIN=1`.
- Result: **FALSIFIED.** Both sequences are `142 22 185 32 22 22 22 22 22 22 22 180 22 180 22 180 22 180 180 22 180 22 180 22 180 180 180 22 180 180 180 180`. Median wall is 0.0050s both ways (default 0.0045–0.0057, grouped 0.0046–0.0053). Grouping at this length is greedy-identical here and does not move the wall. The 4096 default stays.
- Evidence: `r8_summary.txt`, `r8_def.txt`, `r8_g1.txt`
- Cross-check: six runs each, one shared argmax string.
- Kept code: none.
- Next action this result selects: the f16 tiny matmul never takes the int8 SDOT path. Build a Q4_K tiny llama (K is a multiple of 256) and measure `STRATUM_NO_SDOT` against the default SDOT path.

## Round 9 — Q4_K SDOT versus the exact-float path

- Next action: Generate a Q4_K tiny llama (`make_tiny_model.py --arch llama --weights q4k`, 928416 bytes) and compare the default SDOT path with `STRATUM_NO_SDOT=1`.
- Why this step: Round 8's f16 llama never executes int8 SDOT. The approved approximation is only visible on Q4_K/Q6_K weights.
- Target metric: single-stream wall time on 8 tokens, prompt `1 2 3 4`. Argmax identity decides whether the default may change.
- Hypothesis: the exact-float path is as fast as SDOT on this small matrix, or the argmax differs and the default should move.
- Command: interleaved 6+6, `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyq.gguf 8 1 2 3 4`.
- Result: **FALSIFIED.** Argmax matches: `418 432 418 769 851 769 221 432`. SDOT walls 0.0052 0.0051 0.0046 0.0055 0.0056 0.0054 (median 0.0053). Exact-float walls 0.0063 0.0062 0.0062 0.0064 0.0062 0.0053 (median 0.0062). The default stays. The file is under 4 MB, so both sides already use the round-7 serial pool.
- Evidence: `r9_summary.txt`, `r9_sdot.txt`, `r9_off.txt`, model `tinyq.gguf`
- Cross-check: the SDOT line reads `ENABLED` versus `disabled`, and the argmax string is shared.
- Kept code: none.
- Next action this result selects: the bottleneck note still says CPU full attention is a two-pass score scan. Measure whether an online-softmax fuse matches the two-pass scores on a fixed vector. Ship it into the engine only if the max absolute difference is 0.

## Round 10 — online softmax versus the shipped two-pass

- Next action: Compare an online softmax against `stratum_softmax_inplace` on one fixed 128-vector. Do not edit the engine unless the max absolute difference is 0.
- Why this step: Round 9 left the SDOT default in place. The remaining note from the bottleneck list is the two-pass CPU attention score scan. A fuse is legal only if the scores do not move.
- Target metric: long-context attention numerics (max abs diff of the softmax output). This is the gate in front of any scan fusion.
- Hypothesis: online softmax matches the shipped two-pass exactly on `sin(i * 0.37) * 8`.
- Command: `clang -O3 -ffast-math ... -o $SCRATCH/bench_online_softmax stratum/native/bench_online_softmax.c` then run that binary. Capture `r10_online.txt` is from the first form of the bench, before the near-tie sweep was added.
- Result: **FALSIFIED.** `online_softmax_max_abs_diff=7.4505806e-09`, argmax 123 both ways, `online_softmax_not_exact`. The quality boundary forbids shipping a lower-precision softmax. No engine edit.
- Evidence: `r10_online.txt`
- Cross-check: the argmax of this one vector matches, and the values still differ, so argmax agreement is not exactness.
- Kept code: the bench only. The engine softmax is unchanged.
- Next action this result selects: the single-vector argmax matched. Measure whether near-tie scores flip the argmax across a deterministic sweep. Ship only if every trial is exact.

## Round 11 — near-tie argmax under online softmax

- Next action: On 2000 deterministic vectors with a 1e-7 logit tie, count argmax flips between the shipped softmax and the online fuse.
- Why this step: Round 10's values differed by 7.5e-9 and the argmax still matched. A flip would be the concrete reason the fuse can change greedy output.
- Target metric: long-context attention numerics (argmax flips).
- Hypothesis: at least one near-tie flips.
- Command: rebuild and run `bench_online_softmax` (same source, sweep added). Output includes the round-10 line plus the flip count.
- Result: **FALSIFIED.** `online_softmax_near_tie_flips=0/2000`. The argmax held on this sweep. The values are still not exact (`7.4505806e-09` on the fixed vector printed above the flip line), so the fuse stays out of the engine.
- Evidence: `r11_online_ties.txt`
- Cross-check: the same process prints both the non-zero diff and the zero flip count.
- Kept code: none in the engine.
- Next action this result selects: stop chasing a softmax fuse. Measure `STRATUM_NO_BATCH_PREFILL=1` against the default prefill on the tiny hybrid. The prompt is 4 tokens, so a batched prefill may be pure overhead.

## Round 12 — batched prefill on a 4-token prompt

- Next action: Compare the default prefill with `STRATUM_NO_BATCH_PREFILL=1` on the tiny hybrid, 8 generated tokens.
- Why this step: Round 11 closed the softmax fuse. The 4-token prompt is the next fixed cost that might be paying for a batch.
- Target metric: single-stream wall time.
- Hypothesis: turning the batch off lowers wall, argmax unchanged.
- Command: interleaved 6+6, `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyh.gguf 8 1 2 3 4`.
- Result: **FALSIFIED.** Argmax `136 62 122 62 69 62 240 248` both ways. Default walls 0.0050 0.0046 0.0052 0.0041 0.0047 0.0048 (median 0.0048). No-batch walls 0.0053 0.0048 0.0048 0.0043 0.0053 0.0045 (median 0.0048).
- Evidence: `r12_summary.txt`, `r12_def.txt`, `r12_nob.txt`
- Cross-check: medians match and the ranges sit on top of each other.
- Kept code: none.
- Next action this result selects: the KV capacity is 4096 while this prompt uses about 12 positions. Measure `STRATUM_MAX_KV=128` against the default. A smaller score stride might cut wall without changing ids, because 128 still holds the prompt.

## Round 13 — KV capacity 128 versus 4096 on a short prompt

- Next action: `STRATUM_MAX_KV=128` versus the default 4096 on the tiny hybrid, 8 tokens.
- Why this step: Round 12's prefill batch did not move the wall. The score row stride is the capacity, and this run only writes about 12 positions.
- Target metric: single-stream wall time.
- Hypothesis: capacity 128 lowers wall and keeps the argmax, because the prompt fits.
- Command: interleaved 6+6, `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyh.gguf 8 1 2 3 4`.
- Result: **FALSIFIED.** Argmax `136 62 122 62 69 62 240 248` both ways. Default (4096 positions, 1.0 MB virtual) walls 0.0061 0.0057 0.0059 0.0060 0.0058 0.0060 (median 0.0059). Capacity 128 (0.0 MB printed, 128 positions) walls 0.0063 0.0055 0.0063 0.0058 0.0062 0.0058 (median 0.0060).
- Evidence: `r13_summary.txt`, `r13_def.txt`, `r13_m128.txt`
- Cross-check: the KV line changes and the argmax does not, so the capacity edit from round 4 is doing what it says, and it is not a short-prompt speed knob.
- Kept code: none.
- Next action this result selects: short-prompt strides are not the cost. The Q4_K SDOT dot is one running double. Measure a 4-accumulator reorder against the shipped `q4k_dot_row_sdot` on a few dozen KB of random blocks. Ship it only if the max absolute difference is 0 and the reorder is faster.

## Round 14 — four Q4_K accumulators versus one running double

- Next action: Time and diff `q4k_dot_row_sdot` against a 4-double reorder on 32 rows, K=1024, deterministic blocks (about 32 * 4 * sizeof(block_q4_K) bytes).
- Why this step: Round 13 showed KV capacity is not the short-prompt wall. The SDOT kernel's latency is the serial double add.
- Target metric: single-stream kernel time, gated by max abs diff.
- Hypothesis: the reorder is bit-exact and faster.
- Command: `clang -O3 -ffast-math -mcpu=apple-m1 -Istratum/native -o $SCRATCH/bench_q4k_acc4 stratum/native/bench_q4k_acc4.c` and run it.
- Result: **FALSIFIED.** `q4k_acc4_max_abs_diff=0.0009765625`, argmax row 16 both ways, `q4k_acc4_not_exact`. Reported times 0.004 ms vs 0.003 ms are below the noise of a claim. The reorder does not enter the kernel.
- Evidence: `r14_q4k_acc4.txt`
- Cross-check: one process prints both the diff and the argmax of the 32 dots.
- Kept code: the bench only.
- Next action this result selects: the kernel reorder is illegal. Measure n-gram spec on the tiny hybrid for 32 tokens, the same question round 6 asked of llama.

## Round 15 — n-gram spec on the tiny hybrid

- Next action: 32 tokens on `tinyh.gguf` with `STRATUM_NGRAM_SPEC=4` and `STRATUM_SPEC_STATS=1`, against greedy.
- Why this step: Round 14 blocked the accumulator reorder. Round 6's llama spec raised tok/call only because the sequence repeated. The hybrid sequence does not look repetitive, so the same flag may accept nothing.
- Target metric: overall wall time, with tokens per forward as the mechanism check.
- Hypothesis: the flag lowers wall or accepts at least one draft, and the emitted ids match greedy.
- Commands: `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyh.gguf 32 1 2 3 4`, twice greedy and twice with the flag.
- Result: **FALSIFIED.** Greedy ids (two runs): `136 62 122 62 69 62 240 248 247 160 134 165 122 247 165 152 80 78 131 232 120 147 165 215 189 25 247 3 52 127 48 238`. Spec emits the same ids via `ngram-step` `out=` and reports `9 batched calls, 0 drafts accepted (1.00 tok/call)`. Wall: greedy 0.0065s and 0.0064s; spec 0.0092s and 0.0090s. No default change.
- Evidence: `r15_greedy.txt`, `r15_greedy2.txt`, `r15_ngram.txt`, `r15_ngram2.txt`
- Cross-check: the two greedy runs match, and every spec `out=` equals the greedy argmax in order. No `logit=nan` in these 32-token runs.
- Kept code: none.
- Next action this result selects: spec added overhead for zero accepts. Soft-warm still runs on every launch (`STRATUM_SOFT_WARM` defaults on). Measure `STRATUM_SOFT_WARM=0` against the default on the tiny hybrid.

## Round 16 — soft-warm on a sub-megabyte file

- Next action: `STRATUM_SOFT_WARM=0` versus the default (soft-warm on) on the tiny hybrid, 8 tokens.
- Why this step: Round 15's spec path was slower. Soft-warm still touches the mmap and runs the free-RAM checks on every launch.
- Target metric: single-stream wall time.
- Hypothesis: disabling soft-warm lowers wall and keeps the argmax.
- Command: interleaved 6+6, `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyh.gguf 8 1 2 3 4`.
- Result: **FALSIFIED.** Argmax `136 62 122 62 69 62 240 248` both ways. Default walls 0.0045 0.0044 0.0045 0.0044 0.0042 0.0040 (median 0.0044), and the capture contains `SOFT_WARM`. Off walls 0.0051 0.0045 0.0049 0.0046 0.0044 0.0047 (median 0.0046), no `SOFT_WARM` line. Disabling it does not help. The default stays.
- Evidence: `r16_summary.txt`, `r16_def.txt`, `r16_off.txt`
- Cross-check: the soft-warm line is present only on the default runs, and the argmax strings match.
- Kept code: none.
- Next action this result selects: the launch still opens several FDs (`V56 multi-FD`). Measure `STRATUM_IO_FDS=1` against the default fanout on the tiny hybrid.

## Round 17 — one I/O fd versus four

- Next action: `STRATUM_IO_FDS=1` versus the default fanout on the tiny hybrid, 8 tokens.
- Why this step: Round 16 kept soft-warm. The banner still opens multiple fds for a sub-megabyte file.
- Target metric: single-stream wall time.
- Hypothesis: one fd lowers wall and keeps the argmax.
- Command: interleaved 6+6, `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyh.gguf 8 1 2 3 4`.
- Result: **FALSIFIED.** Argmax `136 62 122 62 69 62 240 248` both ways. Default prints `V56 multi-FD I/O: 4 independent FDs` and walls 0.0062 0.0063 0.0061 0.0062 0.0060 0.0058 (median 0.0062). One fd prints no multi-FD line and walls 0.0063 0.0061 0.0061 0.0058 0.0057 0.0062 (median 0.0061). The ranges sit on top of each other.
- Evidence: `r17_summary.txt`, `r17_def.txt`, `r17_one.txt`
- Cross-check: the fd line changes and the argmax does not.
- Kept code: none.
- Next action this result selects: eight-token walls are inside a few tenths of a millisecond. One `STRATUM_TIMING=1` run of 32 hybrid tokens should show whether attention, SSM, or the lm head holds the time. The next edit follows that split.

## Round 18 — STRATUM_TIMING on 32 hybrid tokens

- Next action: One run, `STRATUM_TIMING=1`, 32 tokens, tiny hybrid. Read the per-token `[section]` lines.
- Why this step: Rounds 16 and 17 moved startup knobs by less than the run-to-run spread. The next edit needs to know which section holds the time.
- Target metric: single-stream section time (attention vs SSM vs lm head).
- Hypothesis: the printed section times are non-zero and one section is larger than the others.
- Command: `STRATUM_NO_GPU=1 STRATUM_TIMING=1 $ENGINE $SCRATCH/tinyh.gguf 32 1 2 3 4`
- Result: **FALSIFIED.** Process wall 0.0075s, exit 0, first argmax still 136. All 32 `[section]` lines are `attn 0.00s ssm 0.00s lm_head 0.00s` because the printer uses `%.2f`. Sums are 0. The timer cannot steer a kernel edit on this model.
- Evidence: `r18_timing.txt`
- Cross-check: the same file has 32 section lines and a finite step-0 logit (`2.17287`).
- Kept code: none.
- Next action this result selects: a 32-token process is still too short for that clock. Retry the round-8 grouping question at 128 llama tokens, where attention has more positions to scan.

## Round 19 — GQA grouping at 128 llama tokens

- Next action: Retry the round-8 grouping question at 128 llama tokens. `STRATUM_ATTN_GRP_MIN=1` versus the default 4096.
- Why this step: Round 18 showed `STRATUM_TIMING` prints `%.2f` zeros on a 32-token run. Attention has more positions at 128, so a grouping win could clear the noise that hid it at 32 tokens.
- Target metric: single-stream warm wall time, with greedy identity as the cross-check.
- Hypothesis: grouped attention (threshold 1) has a lower warm wall than the default, and the 128-id sequence stays the same.
- Command: interleaved 4+4 after one warmup, `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyl.gguf 128 1 2 3 4`, with and without `STRATUM_ATTN_GRP_MIN=1`.
- Result: **FALSIFIED.** Default walls 0.008147 0.008192 0.008582 0.008275. Grouped walls 0.007837 0.008562 0.008415 0.008420. The ranges overlap (default 0.008147–0.008582, grouped 0.007837–0.008562). Both exit 0. The full 128-id sequences match.
- Evidence: `r19_summary.txt`, `r19_def.txt`, `r19_g1.txt`
- Cross-check: `r19_summary.txt` prints `match True` and one shared arg line. `r19_def.txt` is the default run (10 chunks, the serial small-model line, step 0 argmax 142).
- Kept code: none. The 4096 default stays.
- Next action this result selects: 128-token process walls still sit on each other near 8 ms, so another scheduling flag on this tiny llama cannot show a kernel effect. Single-stream Q4_K still calls `q4k_dot_row_sdot_f` once per row and reloads x every row. The next measurement is a microbench of a two-row kernel that loads each x chunk once and keeps each row's float32x4 reduction identical, at K=1024 N=512, against two calls of the shipped function. Ship it only if the max abs diff is 0 and the timed reps separate.

## Round 20 — two-row Q4_K SDOT that shares the x load

- Next action: Microbench a two-row kernel against `q4k_dot_row_sdot_f` at K=1024, N=512. The candidate loads each x chunk once and keeps each row's float32x4 reduction identical. Do not edit the engine until the diff and the times are both in hand.
- Why this step: Round 19's 128-token walls overlap near 8 ms. A scheduling flag on the tiny llama cannot show a kernel effect. Single-stream Q4_K is still one `q4k_dot_row_sdot_f` per row.
- Target metric: kernel time on that matrix, and max abs diff versus the shipped function.
- Hypothesis: the two-row kernel has max abs diff 0 and a lower time than two shipped calls, with the six interleaved trials separating.
- Command: `clang -O3 -ffast-math -mcpu=apple-m1 -I stratum/native -o $SCRATCH/bench_q4k_rows2 stratum/native/bench_q4k_rows2.c && $SCRATCH/bench_q4k_rows2`
- Result: **FALSIFIED.** `q4k_rows2_max_abs_diff=0`, both argmax 339, `q4k_rows2_exact`. Row-at-a-time times ms: 0.552 0.546 0.509 0.495 0.523 0.495. Two-row times ms: 0.618 0.624 0.547 0.548 0.618 0.557. The two-row median is higher and the ranges overlap, so the shared x load is not a shippable win.
- Evidence: `r20_rows2.txt` (same numbers as this command's stdout).
- Cross-check: exactness and the slowdown are in the same print. The kernel stays in the bench only.
- Kept code: none in the engine. `bench_q4k_rows2.c` records the disproof.
- Next action this result selects: widening the single-stream SDOT dot is the wrong direction. The 8 ms llama run is F16, not Q4_K. One `STRATUM_TYPETIME=1` pass on 128 tiny-llama tokens should show whether F16 matmul is at least half the process wall. The following edit follows that split.

## Round 21 — F16 matmul's share of the 128-token wall

- Next action: One warm run of the tiny llama, 128 tokens, with `STRATUM_TIMING=1` and `STRATUM_TYPETIME=1`. The typetime printer lives inside the timing block, so the timing flag has to be set for the line to appear.
- Why this step: Round 20 left the Q4_K single-row kernel in place. The tiny llama that produced the 8 ms wall is F16. The next kernel edit needs to know whether that matmul is at least half the process wall.
- Target metric: F16 matmul seconds versus process wall on the same run.
- Hypothesis: the F16 typetime total is at least half the process wall.
- Command: `STRATUM_NO_GPU=1 STRATUM_TIMING=1 STRATUM_TYPETIME=1 $ENGINE $SCRATCH/tinyl.gguf 128 1 2 3 4` after one warmup. Python `perf_counter` around the process.
- Result: **FALSIFIED.** Process wall 0.008761s, exit 0. Gen timer `gen 128 tok: 0.005s`. Typetime `F16 0.003s over 3825 calls (0.001 ms/call)`. 0.003 / 0.008761 is about a third of the process wall, not half. The 0.003s also includes one `clock_gettime` pair per call.
- Evidence: `r21_typetime.txt`
- Cross-check: the same file has the prefill argmax 22 and logit 1.93117, matching the earlier llama runs, plus the timing line and the single F16 typetime line.
- Kept code: none.
- Next action this result selects: F16 is the only matmul time, about 0.003s, on the tiny-llama FFN shape. The next measurement is a microbench of that shape (N=176, K=64): NEON fp16-to-fp32 conversion followed by the same left-to-right double sum as `st_linear_f16`. Ship it only if the max abs diff is 0 and the trial times separate downward.

## Round 22 — NEON fp16 convert, same double sum

- Next action: Microbench `gemv_f16_neon_f64` against the scalar double loop copied from `st_linear_f16`, shape N=176 K=64, 4000 reps, 6 interleaved trials.
- Why this step: Round 21 put about 0.003s in F16 matmul. A conversion speedup that keeps the left-to-right double sum would be inside the quality boundary.
- Target metric: kernel time and max abs diff on that shape.
- Hypothesis: max abs diff is 0 and the NEON-convert trials are lower than the scalar trials, without overlap.
- Command: `clang -O3 -ffast-math -mcpu=apple-m1 -I stratum/native -o $SCRATCH/bench_f16_cvt stratum/native/bench_f16_cvt.c && $SCRATCH/bench_f16_cvt`
- Result: **FALSIFIED.** `f16_cvt_max_abs_diff=0`, both argmax 12, `f16_cvt_exact`. Scalar ms: 10.712 8.364 7.228 5.162 4.784 4.149. NEON-convert ms: 11.096 10.383 6.698 6.261 4.944 4.687. Scalar is faster on 5 of the 6 paired trials. Lane extracts do not beat the scalar convert at K=64. Both series fall together as the core ramps; the pairing is the comparison.
- Evidence: `r22_f16.txt`
- Cross-check: exactness and the times are one print. Nothing in the engine changes.
- Kept code: none. `bench_f16_cvt.c` records the disproof.
- Next action this result selects: the same-order double sum is not sped up by a NEON convert. The remaining F16 lever is an f32x4 FMA accumulator. That changes the sum, so the next measurement is the max abs diff against the double loop on the same shape. Ship only if that diff is 0.

## Round 23 — f32x4 F16 GEMV versus the double sum

- Next action: Microbench an f32x4 FMA GEMV against the scalar double loop, same N=176 K=64 shape, 4000 reps, 6 interleaved trials.
- Why this step: Round 22 kept the double sum exact and did not make it faster. An f32 accumulator is the remaining way to shorten that 0.003s, and it is only shippable if the values do not move.
- Target metric: max abs diff, then kernel time.
- Hypothesis: max abs diff is 0.
- Command: `clang -O3 -ffast-math -mcpu=apple-m1 -I stratum/native -o $SCRATCH/bench_f16_f32acc stratum/native/bench_f16_f32acc.c && $SCRATCH/bench_f16_f32acc`
- Result: **FALSIFIED.** `f16_f32_max_abs_diff=1.90734863e-06`, 123 of 176 rows differ, `f16_f32_not_exact`. Argmax happens to match on this draw (both 12). Scalar ms: 10.856 8.882 7.587 6.054 5.713 5.072. f32 ms: 7.149 4.739 2.942 2.735 2.181 2.074. The f32 kernel is faster on every paired trial and is not the shipped sum. It stays out of the engine.
- Evidence: `r23_f16_f32.txt`
- Cross-check: the diff, the neq count, and both time rows are one print.
- Kept code: none. `bench_f16_f32acc.c` records the disproof.
- Next action this result selects: the speed came from a second independent chain, and the double sum is one chain deep. The next measurement pairs two rows, each with its own left-to-right double sum, on the same shape. Ship only if the max abs diff is 0 and the paired trials separate downward.

## Round 24 — two F16 rows, each with its own double sum

- Next action: Microbench a two-row F16 GEMV whose per-row sum is the same left-to-right double chain, N=176 K=64, after a 200-call warmup, 2000 reps, 6 interleaved trials.
- Why this step: Round 23's f32 kernel was faster and not exact. A second double chain is the exact form of that extra issue width.
- Target metric: max abs diff and kernel time.
- Hypothesis: max abs diff is 0 and the two-row trials sit below the single-row trials with no overlap.
- Command: `clang -O3 -ffast-math -mcpu=apple-m1 -I stratum/native -o $SCRATCH/bench_f16_rows2 stratum/native/bench_f16_rows2.c && $SCRATCH/bench_f16_rows2`
- Result: **FALSIFIED.** `f16_rows2_max_abs_diff=0`, neq=0, both argmax 12, `f16_rows2_exact`. Single-row ms: 7.777 4.755 4.530 3.765 3.570 3.005. Two-row ms: 5.201 4.787 4.395 3.877 3.293 3.007. The ranges sit on top of each other. Pairing does not clear a win at K=64.
- Evidence: `r24_f16_rows2.txt`
- Cross-check: exactness and both time rows are one print. The engine keeps the single-row double sum.
- Kept code: none. `bench_f16_rows2.c` records the disproof.
- Next action this result selects: stop chasing the K=64 F16 GEMV. The llama single-token PV tail is still `head_out[d] += p * vt[d]` with `d += 4` after the 16-wide loop. The next measurement compares that pattern with a scalar axpy at Hd=20 and at Hd=16.

## Round 25 — llama PV tail versus a scalar axpy

- Next action: Reproduce the single-token PV loops at Hd=16 and Hd=20 and compare each lane with a scalar axpy.
- Why this step: Round 24 closed the K=64 F16 search. The leftover tail steps `d` by 4 and writes one lane.
- Target metric: lanes that disagree with the scalar axpy.
- Hypothesis: both widths match the scalar axpy.
- Command: `clang -O3 -ffast-math -mcpu=apple-m1 -o $SCRATCH/bench_pv_tail stratum/native/bench_pv_tail.c && $SCRATCH/bench_pv_tail`
- Result: **FALSIFIED.** Hd=16: `pv_tail_hd16_max_abs_diff=0 missed=0`. Hd=20: `pv_tail_hd20_max_abs_diff=0.109999895 missed=3`. `pv_tail_mismatch`. The 16-wide body is fine. The 4-wide tail drops three lanes, so a head dim that is not a multiple of 16 changes the attention output.
- Evidence: `r25_pv_tail.txt`
- Cross-check: the same print has both widths. Tiny llama and the Q4_K tiny use Hd=16 and 32, so their short prompts never entered the tail.
- Kept code: none yet. `bench_pv_tail.c` is the reproduction of the old pattern.
- Next action this result selects: replace that tail with a 4-wide store that writes every lane, keep the 16-wide body, and check Hd=20 against the scalar axpy plus the short-prompt argmax on the rebuilt engine.

## Round 26 — full-lane PV tail

- Next action: Put the 16-wide body and a full 4-wide tail in `stratum_axpy_f32`, call it from the llama single-token PV loop, and measure Hd=16 and Hd=20 against the scalar axpy. Then rebuild the CPU engine and rerun the short prompts.
- Why this step: Round 25 showed the one-lane tail drops 3 lanes at Hd=20 and matches at Hd=16.
- Target metric: lanes that disagree with the scalar axpy. Short-prompt greedy ids are the cross-check that Hd=16 did not move.
- Hypothesis: both widths match the scalar axpy, and the tiny llama, tiny hybrid, and tiny Q4_K llama keep their 8-token argmax.
- Commands:
  - `clang -O3 -ffast-math -mcpu=apple-m1 -I stratum/native -o $SCRATCH/bench_axpy stratum/native/bench_axpy.c && $SCRATCH/bench_axpy`
  - `make -C stratum/native USE_MEMX=0 USE_METAL=0 stratum`
  - `STRATUM_NO_GPU=1 $ENGINE $SCRATCH/tinyl.gguf 8 1 2 3 4` and the same for `tinyh.gguf` and `tinyq.gguf`
- Result: **KEEP.** `axpy_hd16_max_abs_diff=0 missed=0`, `axpy_hd20_max_abs_diff=0 missed=0`, `axpy_ok`. The old pattern's Hd=20 diff was 0.109999895 with 3 missed lanes. Short prompts: llama `142 22 185 32 22 22 22 22`, hybrid `136 62 122 62 69 62 240 248`, Q4_K llama `418 432 418 769 851 769 221 432`.
- Evidence: `r26_axpy.txt`, `r26_llama.txt`, `r26_hybrid.txt`, `r26_q4k.txt`
- Cross-check: the bench calls `stratum_axpy_f32`, which is the function the single-token loop now calls. The three argmax strings match the sequences pinned before this edit.
- Kept code: `stratum_axpy.h` and the llama single-token call.
- Next action this result selects: the other llama PV sites still call `la_axpy_hd`, a 4-wide loop. The next measurement asks whether the shipped 16-wide axpy is faster at Hd=128 over 2048 steps, and whether every lane still matches the 4-wide loop.

## Round 27 — 16-wide axpy versus the 4-wide helper

- Next action: Time `stratum_axpy_f32` against a copy of `la_axpy_hd` at Hd=128, 2048 steps, 40 reps, 6 interleaved trials. Compare every lane.
- Why this step: Round 26 left the other PV sites on the 4-wide helper. A faster exact 16-wide loop would be worth switching them.
- Target metric: kernel time, with max abs diff as the gate.
- Hypothesis: the lanes match and the 16-wide trials sit below the 4-wide trials with no overlap.
- Command: `clang -O3 -ffast-math -mcpu=apple-m1 -I stratum/native -o $SCRATCH/bench_axpy_width stratum/native/bench_axpy_width.c && $SCRATCH/bench_axpy_width`
- Result: **FALSIFIED.** `axpy_width_max_abs_diff=0 neq=0`, `axpy_width_exact`. 4-wide ms: 0.360 0.371 0.414 0.363 0.359 0.411. 16-wide ms: 0.380 0.403 0.362 0.422 0.384 0.396. The ranges overlap. The other call sites stay on `la_axpy_hd`.
- Evidence: `r27_axpy_width.txt`
- Cross-check: exactness and both time rows are one print.
- Kept code: none beyond round 26.
- Next action this result selects: axpy width is not the lever. Llama's `STRATUM_MAX_KV` floor is 256, while the qwen35 floor is 32. The next measurement asks whether `STRATUM_MAX_KV=32` on the tiny llama is honored or raised, using a 40-token run.

## Round 28 — llama STRATUM_MAX_KV floor

- Next action: Run the tiny llama for 40 tokens with `STRATUM_MAX_KV=32`.
- Why this step: Round 27 left axpy width alone. The llama capacity clamp is `if (mk < 256) mk = 256`, and the qwen35 clamp is 32. A requested capacity of 32 may be silently raised.
- Target metric: long-context capacity (the positions line, and whether token 33 is refused).
- Hypothesis: the run prints 32 positions and exits non-zero once the context passes 32.
- Command: `STRATUM_NO_GPU=1 STRATUM_MAX_KV=32 $ENGINE $SCRATCH/tinyl.gguf 40 1 2 3 4`
- Result: **FALSIFIED.** Exit 0. `KV cache: 0.2 MB virtual (lazy-commit, 256 positions)`. 40 `stratum_argmax=` lines. The requested 32 was raised to 256, so the run never hit the hard stop.
- Evidence: `r28_maxkv.txt`
- Cross-check: the positions line and the 40th step line are in the same capture. No `exceeds` line.
- Kept code: none yet.
- Next action this result selects: lower the llama floor from 256 to 32, the same floor qwen35 already uses. Then a 40-token run at `STRATUM_MAX_KV=32` must refuse, and the default 8-token argmax must stay `142 22 185 32 22 22 22 22`.

## Round 29 — llama capacity floor 256 → 32

- Next action: Change the llama floor from 256 to 32. Rerun `STRATUM_MAX_KV=32` for 40 tokens, the default 8-token prompt, and `STRATUM_MAX_KV=256` for 8 tokens.
- Why this step: Round 28 showed a request of 32 was stored as 256 and the hard stop never ran.
- Target metric: long-context capacity. The short-prompt argmax is the cross-check.
- Hypothesis: capacity 32 exits non-zero at context 33, the default run stays at 4096 with argmax `142 22 185 32 22 22 22 22`, and an explicit 256 still allocates 256.
- Commands: `make -C stratum/native USE_MEMX=0 USE_METAL=0 stratum`, then `STRATUM_NO_GPU=1` with `STRATUM_MAX_KV=32` on `tinyl.gguf 40 1 2 3 4`, the same binary with the env unset on `8 1 2 3 4`, and `STRATUM_MAX_KV=256` on `8 1 2 3 4`.
- Result: **KEEP.**
  - `STRATUM_MAX_KV=32`: exit 1, `32 positions`, `llama: context 33 exceeds KV capacity 32`. 28 generated ids. The first eight are `142 22 185 32 22 22 22 22`.
  - Default: exit 0, `4096 positions`, the same eight ids.
  - `STRATUM_MAX_KV=256`: exit 0, `256 positions`, the same eight ids.
- Evidence: `r29_maxkv32.txt`, `r29_default.txt`, `r29_maxkv256.txt`
- Cross-check: the 28 ids produced before the stop match the first 28 ids of round 28's 256-position run, so the tokens that fit did not move. The default capacity line is still 4096.
- Kept code: the llama floor in `la` allocate.
- Next action this result selects: the new floor refuses 32 and still accepts 256. A request below the floor is unmeasured. Run `STRATUM_MAX_KV=16` for 8 tokens and read the positions line.

## Round 30 — a request below the llama floor

- Next action: Run 8 tokens with `STRATUM_MAX_KV=16` on the tiny llama.
- Why this step: Round 29 honored 32 and 256. The new code raises anything below 32 up to 32. That edge has no capture yet.
- Target metric: long-context capacity (positions actually reserved) and the short-prompt argmax.
- Hypothesis: the positions line says 16.
- Command: `STRATUM_NO_GPU=1 STRATUM_MAX_KV=16 $ENGINE $SCRATCH/tinyl.gguf 8 1 2 3 4`
- Result: **FALSIFIED.** Exit 0. `KV cache: 0.0 MB virtual (lazy-commit, 32 positions)`. The request 16 was raised to the floor. Argmax `142 22 185 32 22 22 22 22`. No `exceeds` line, because 4 prompt tokens plus 8 generated tokens fit in 32.
- Evidence: `r30_maxkv16.txt`
- Cross-check: the same file has the 32-position line and eight `stratum_argmax=` lines, and the eight ids match the default run in round 29.
- Kept code: none. The floor stays 32.
- Next action this result selects: the thirty measurements are in. The proof pass is an in-repo cold start of `./stratum` on generated tiny models, plus the attendance and axpy benches, plus a second run of the shipped speed comparisons.

## Shipped-binary re-run

Not a round. Two fresh comparisons on the final CPU binary, tiny hybrid, 8 tokens, 6 interleaved reps each, done twice.

`STRATUM_PAR_FORCE=1` versus the default serial path. Trial 1 medians 0.005125s versus 0.003971s, no overlap. Trial 2 medians 0.004957s versus 0.004108s, no overlap. The default is faster both times. Argmax stays `136 62 122 62 69 62 240 248`. Evidence: `verify_speed.txt`.

`STRATUM_NCHUNKS=14` versus the default 5 still changes the chunks line. The walls overlap (trial 1 medians 0.003931s and 0.004091s, trial 2 0.004023s and 0.003978s). Mmaps at or under 4 MB no longer enter the pool, so the chunk count is not the wall on this input.

Two consecutive short runs match: llama `142 22 185 32 22 22 22 22`, hybrid `136 62 122 62 69 62 240 248`. Evidence: `verify_llama_rep1.txt`, `verify_llama_rep2.txt`, `verify_hybrid_rep1.txt`, `verify_hybrid_rep2.txt`. The cold-start suite is `stratum/tools/campaign_tiny_test.py`; its capture is `verify_test.txt`.
