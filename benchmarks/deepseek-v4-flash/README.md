# DeepSeek V4 Flash on Strix Halo (gfx1151)

Status: live. This page is the working performance card and experiment log for
the `rocm-strix-halo-release` branch. `ROCM_STRIX_HALO_OPTIMIZATION.md` remains
the narrative worklog; this card is the measured ledger.

Goal for the current session: **400 t/s prompt processing at the 4K frontier.**

## Model

| Field | Value |
| --- | --- |
| Repository | `antirez/deepseek-v4-gguf` |
| Snapshot | `1cd7b564460821938add0475a60b942c409295e0` |
| Artifact | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| Quantization | IQ2_XXS gate/up, Q2_K down, Q8 A-proj / shared expert / output |
| Size | 80.76 GiB resident |
| Layers | 43 |
| Branch | `rocm-strix-halo-release`, baseline `76a7755` |

## Hardware

| Field | Value |
| --- | --- |
| GPU | Radeon 8060S, `gfx1151`, RDNA 3.5, 40 CU |
| Memory | 128 GiB LPDDR5X unified, 256-bit, 16 channels x 16-bit |
| Measured ceilings | ~242 GB/s DRAM, ~59 TFLOP/s FP16 WMMA, 32 MB MALL |
| Per-CU limits | 64 KiB LDS, 1,536 VGPRs, 2 GB/s+ scalar |
| Toolchain | ROCm 7.2.3, pinned `nixos-unstable` flake |

## Run

```sh
MODEL=/var/llms/huggingface/hub/models--antirez--deepseek-v4-gguf/snapshots/1cd7b564460821938add0475a60b942c409295e0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf

nix develop --command make rocm -j"$(nproc)"

# Canonical two-frontier prefill measurement. Read the 4096 row as "4K".
./ds4-bench -m "$MODEL" --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 8192 --step-incr 4096 --gen-tokens 0

# Parity and quality gates
nix develop --command make test-mmq-parity-rocm -j"$(nproc)"

# Kernel trace
nix develop --command rocprofv3 --kernel-trace -d /tmp/prof -f csv -- \
  ./ds4-bench -m "$MODEL" --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 4096 --ctx-max 8192 --step-incr 4096 --gen-tokens 0
```

### Measurement rules

1. **Use the two-frontier form.** A single `--ctx-start 4096 --ctx-max 4096`
   run reads 281.50 t/s against 327.77 for the same binary in the two-frontier
   form, because `--ctx-alloc` derives from `--ctx-max` and changes the raw KV
   ring size. Never compare across the two forms.
2. **Alternate builds, never sessions.** The APU throttles across a sweep. Keep
   both binaries and interleave A/B/A/B/A/B in one script; a model load is 22 s,
   so rebuilding between measurements is pure waste.
3. **Probe before building.** Bound the payoff with a throwaway or a counter
   read first. Verify the probe is live: a corrupting probe *must* move the
   logit hash. Use `HIPCC_COMPILE_FLAGS_APPEND` to inject a define; overriding
   `ROCM_CFLAGS` drops the include paths Nix injects.
4. **State the memory delta with every proposal.** 80.76 GiB is resident on a
   128 GiB box. A sub-1% gain does not justify hundreds of MiB.
5. **An isolated GEMM harness under-predicts, and not uniformly.** With three
   rotating operand copies, `rocm/tools/f16_gemm_bench.cpp` scored the attention
   output B kernel at 1.34x over hipBLASLt; in the engine it measured **1.97x**,
   because Tensile degrades 1.62x from isolated to in-app while a low-traffic
   kernel degrades 1.10x. The correction is real but it did **not** carry across
   the transpose: the same reasoning applied to `opA=T` predicted a win and
   measured +456 ms. Always confirm with a per-kernel trace of the real run
   before an end-to-end A/B, which is both cheaper and less noisy.
6. **Prefer a trace to an A/B while iterating.** A two-chunk kernel trace
   attributes every millisecond per kernel in about four minutes and costs
   nothing in variance; the 4K A/B row has a ±1.4% baseline spread that hid a
   -1,109 ms win and a +987 ms regression cancelling each other exactly.

## Current Results

Clean runs of the canonical two-frontier command, mean of three alternating
rounds against the same baseline binary.

| Frontier | Baseline `76a7755` | Current | Gain |
| ---: | ---: | ---: | ---: |
| 4K | 328.72 t/s | **340.55 t/s** | **+3.6%** |
| 8K | 362.40 t/s | **380.69 t/s** | **+5.1%** |

Means of three alternating pairs in one script; all six pairs positive. Baseline
8K spread 362.24-362.65, candidate 379.43-381.98.

**The 400 t/s target was not reached.** The 8K steady-state rate is 380.69, so
19.3 t/s short, which is another 5.1% or about 425 ms off a 10,600 ms frontier.
See "Where the remaining 5% is" below: the BLAS/projection seam that produced
this gain is now closed, and every remaining candidate is either at the memory
roofline or inside the two kernels the previous worklog already mined hardest.

### Quality

| Gate | Result |
| --- | --- |
| HIP MMQ parity suite | **ALL PASS** |
| Argmax, 2K and 4K frontiers | **identical to baseline** |
| Non-finite logits / null logits | **0 / 0** at both frontiers |
| Top-20 overlap | **19/20** at both frontiers |
| Max / RMS logit delta, 2K | 2.822 / 0.465 |
| Max / RMS logit delta, 4K | 1.077 / 0.255 |
| Greedy `temp=0` continuation | ` Paris. It is located in the north-central part of the` |

Exact logit hashes do **not** hold and are not expected to: the retained change
replaces a Tensile GEMM with a kernel whose accumulation order differs. The
right comparison is the drift this project already accepted for its own
baseline-to-current series, which was 1.886/0.396 at 2K and 1.385/0.271 at 4K.
The 4K figures here are inside that envelope on both metrics; the 2K max delta
(2.822 against 1.886) is outside it, on a single changed GEMM whose per-output
accumulation is a plain ascending walk over K. Argmax, top-20 and the greedy
continuation are unaffected. A multi-prompt teacher-forced likelihood corpus
remains the missing gate and is the right next quality investment.

**Use the 8K row.** Both frontiers prefill exactly 4,096 tokens
(`prefill_tokens = frontier - previous`), but the 4K row is the first one and
therefore pays every first-use cost inside its measured window. A phase split of
the kernel trace prices that at **1.2 s of host stalls**, 11% of the frontier:
hipBLASLt plans are cached per `(m, n, k, opA, output type)` and `n` is the token
count, so the 2,048-token warmup creates plans the 4,096-token chunk cannot use,
and the routed-MoE and indexed-attention route messages also print inside
frontier one. The 8K row is the same 4,096 tokens with all of that already paid,
which is why it is both higher and six times tighter. It is the honest
steady-state prefill rate and the number to drive to 400.

The GPU is **99.8% busy** inside the measured prefill (union of dispatch
intervals over the phase span), so prefill is kernel-bound and there is no
launch-gap to recover.

## Baseline Kernel Profile

Two-chunk trace (`4096 -> 8192`) at `76a7755`, 7,093 dispatches, 25,306 ms of
GPU time. Unlike the earlier worklog table this one **includes** the startup
transpose, because the 4K frontier pays it inside the measured window.

| Kernel | ms | Calls | ms/call | Share | VGPR | LDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mul_mat_q<IQ2_XXS,80>` gate/up | 4,431.6 | 172 | 25.77 | 17.5% | 168 | 30.8K dyn |
| `moe_down_q2K_hotlist_wmma_wide` | 3,764.1 | 86 | 43.77 | 14.9% | 80 | 8K dyn |
| `attention_mixed_heads32_wmma<true,true>` | 2,582.1 | 42 | 61.48 | 10.2% | 80 | 58,368 |
| `dequant_q8_0_to_f16_transpose` | 2,258.3 | 43 | 52.52 | 8.9% | 16 | 0 |
| Tensile `Cijk_Ailk_Bljk_HSS_MT96x96x32_MI16x16x1` | 2,225.5 | 87 | 25.58 | 8.8% | 256 | 30,720 |
| `mul_mat_q<Q8_0,80>` dense | 1,866.3 | 430 | 4.34 | 7.4% | 192 | 0 |
| Tensile `Cijk_Alik_Bljk_HHS_MT128x128x32_MI16x16x16x1` | 1,126.5 | 87 | 12.95 | 4.5% | 256 | 51,200 |
| `attention_static_mixed_heads8_online` | 905.0 | 22 | 41.14 | 3.6% | 48 | 32,768 |
| Tensile `Cijk_Alik_Bljk_HSS_MT96x96x32_MI16x16x1` | 707.4 | 467 | 1.52 | 2.8% | 256 | 31,744 |
| `indexer_scores_wmma128` | 666.9 | 42 | 15.88 | 2.6% | 72 | 45,056 |
| `head_rms_norm_rope_tail` | 625.4 | 86 | 7.27 | 2.5% | 24 | 1,024 |
| Tensile `Cijk_Alik_Bljk_HSS_MT32x32x32_MI16x16x1` | 259.4 | 387 | 0.67 | 1.0% | 128 | 13,312 |
| all Tensile BLAS | **4,318.8** | 1,028 | - | **17.1%** | - | - |

Two findings the previous worklog did not have, both large:

- **Two of the four Tensile kernels do not use the matrix cores at all.**
  `MI16x16x1` is Tensile's non-WMMA source kernel; only the `MI16x16x16x1`
  variant issues `v_wmma`. The single largest BLAS shape, the attention output
  B GEMM (`m=4096 n=4096 k=8192`, `opA=N`, F16 in / F32 out), runs at 25.58 ms
  per call for 274.9 GFLOP, i.e. **10.6 TFLOP/s, 18% of the WMMA ceiling**, on
  the vector ALU. The `HHS` shape at 12.95 ms per call is real WMMA and still
  only reaches about 5.3 TFLOP/s.
- **`dequant_q8_0_to_f16_transpose` is ~25x off roofline.** It moves 67 MiB out
  plus 36 MiB in per call in 52.5 ms, an effective **1.9 GB/s**. See the
  experiment log.

hipBLASLt shapes reached at the 4K frontier, from the plan-selection log:

| Label | opA | m | n | k | Output |
| --- | --- | ---: | ---: | ---: | --- |
| attention output b | N | 4096 | 4096 | 8192 | F32 |
| q8_0 | T | 4096 | 4096 | 2048 | F16 |
| f16 projection | T | 8192 | 4096 | 1024 | F32 |
| f16 projection | T | 1024 | 4096 | 4096 | F32 |
| f16 projection | T | 512 | 4096 | 4096 | F32 |
| f16 projection | T | 256 | 4096 | 4096 | F32 |
| f16 projection | T | 64 | 4096 | 4096 | F32 |
| f16 projection | T | 24 | 4096 | 16384 | F32 |
| f16 projection | T | 1024 / 256 | 4 | 4096 | F32 |

`hipblaslt_gemm_plan_tune` pins candidate 4, 5 or 6 per shape and requests
**zero workspace**. Those indices were chosen for *determinism*, not speed, so
the pinning is itself an untested performance decision.

## Optimization Experiment Log

Ordered newest first. Every entry records the hypothesis, what was measured,
and the decision, so rejected directions are not retried.

| ID | Experiment | Result | Status |
| :--- | :--- | :--- | :--- |
| `ds4-b06-batched-a-gemm` | The attention output A GEMM is the last large kernel still on a BLAS: `Cijk_Alik_Bljk_HHS_MT128x128x32`, 1,120 ms per two-chunk trace, reached through `cublasGemmStridedBatchedEx` so it never even saw the hipBLASLt plan cache. Added a strided-batched form of the WMMA kernel (`blockIdx.z` over problems, plus an `ldc` because each group writes its own `rank` block of a `low_dim` row) | **Closed by arithmetic, and the eligibility check correctly declined it.** The dims are not what the low_dim=8192 arithmetic suggested: the trace's `grid=1024x32x8` gives `n_groups = 8`, hence `rank = 1024` and `group_dim = 4096`, so the call is 275 GFLOP in 12.88 ms = **21.4 TFLOP/s**, not the 2.7 an assumed `rank=128` implied. Our own 256x128 kernel reaches only 10.1 TFLOP/s on the near-identical `m=1024 n=4096 k=4096` shape. rocBLAS is already about 2x better here. The batched capability is retained because it cost nothing and is correct, but nothing selects it | **Rejected** |
| `ds4-b05-opa-t-revert` | Even with the grid gate, the `opA=T` family was still net-negative: the gate left 128 calls on the WMMA kernel at 592.3 ms while Tensile handled the rest, for 1,173 ms against the baseline's 967 ms. Remove the `opA=T` routing entirely and keep only `opA=N` | **-202 ms** (WMMA -592.3, Tensile +389.8). Confirms the harness's original verdict: Tensile genuinely wins every `opA=T` shape on this GPU, and the in-app/isolated correction that made `opA=N` a 1.97x win does **not** generalise across the transpose | **Retained** |
| `ds4-m01-mmq-x` | The `mmq_x` selector minimises `ceil(ncols_max / x)` and the routed path passes the whole chunk width as `ncols_max`, so it always lands on the 80 cap. Per expert only about `n_tokens * 6 / 256` = 96 columns are real, so the second 80-column tile should be a fifth full, and `x=48` would give two exactly-full tiles for the same number of weight passes. Made the cap tunable on HIP and swept it against a real chunk | **Rejected, and it closes the fill hypothesis.** 8K, two rounds each: X=80 **377.5**, X=64 374.8, X=48 372.1, X=40 334.1, X=32 334.6. Fill is monotonically *worse* below 80, and there is an 11% cliff at 40 where the tile granularity changes. So the measured MoE sub-linearity in `ds4-m02` is not column-tile fill; it is the 64-row destination granularity and per-tile activation amortisation, as the original worklog said. The env knob is kept as a diagnostic; the default is unchanged | **Rejected** |
| `ds4-b04-wmma-grid-gate` | `ds4-b03` routed *every* eligible shape through the 256x128 kernel and cost +456 ms, because at `m=256` and `m=512` the tile leaves 32 and 64 workgroups for 40 CUs while Tensile's MT96x96 and MT32x32 keep hundreds in flight. Require `(m/256) * (n/128) >= 240`, six workgroups per CU, so only `m=4096` and `m=8192` qualify. Also route the F16-result twin of the attention output B GEMM, which reaches hipBLAS directly and so never even saw the plan cache: Tensile's `MT128x128` prices it at 12.87 ms per call for 68.7 GFLOP, **2.9 TFLOP/s** | **4K 329.61 -> 343.80 (+4.3%), 8K 363.98 -> 377.03 (+3.6%)**, three alternating pairs, all six positive, candidate spread ±0.04% at 8K. Quality: argmax identical at 2K and 4K, zero non-finite and zero null logits, max/RMS delta 2.129/0.402 at 2K and 1.319/0.266 at 4K, i.e. inside the envelope this project already accepted for its own baseline-to-current drift (1.886/0.396 and 1.385/0.271) | **Retained** |
| `ds4-b03-proj-wmma` | Route all remaining Tensile shapes through the kernel, templated on A layout and output type | Mixed, and the trace separated it cleanly: the `opA=N` attention output B kept its **-1,121 ms** while the `opA=T` projection family went **707.4 ms over 467 calls to 1,163.3 over 462, +456 ms**. Superseded by the grid gate above | **Partly retained** |
| `ds4-b02-attn-out-b-wmma` | Take the attention output B GEMM (`m=4096 n=chunk k=8192`, `opA=N`, F16 in / F32 out) off hipBLASLt. Tensile picks `MT96x96x32`, which re-reads each panel `m/96` and `n/96` times, about 5.7 GB per call; at the traced 25.58 ms that is 223 GB/s, essentially DRAM peak, so the tile size *is* the cost. A 256x128 macro tile moves 3.22 GB. Four m-fragments and two n-fragments per wave, 512 threads, `col_major` LDS A so the transposed-weight read stays contiguous in `m`, two-column swizzle | **25.58 -> 12.98 ms per call, 1.97x.** Kernel total 2,225.5 -> 1,116.4 ms over the two-frontier trace, **-1,109 ms**. 104 VGPRs, 24,576 B LDS, no scratch. Note the microbench under-predicted this: isolated it was only 1.34x, because Tensile degrades 1.62x from isolated to in-app while this kernel degrades 1.10x — the low-traffic kernel is the one that keeps its speed when the caches are cold | **Retained** |
| `ds4-e01-rmsnorm-wave` | `rms_norm_plain_f16_kernel` runs one 256-thread workgroup per row with an eight-level shared-memory tree, re-reads the row for the output pass, and stores single halves. Give each row one wave, eight rows per workgroup, `warp_sum_f32`, and packed `half2` stores, with the slot tree arranged to reproduce the original association bit-for-bit | **Rejected: 2.50 -> 8.20 ms per call, 3.3x slower** (431.9 -> 1,418.9 ms, +987 ms), which exactly cancelled the win above and is why the first A/B read flat. The row is 4,096 long, so one wave per row cuts the thread count 8x, from 1,048,576 to 131,072 against ~82,000 concurrent lanes — there is no longer enough parallelism to cover the load latency. Wave-per-row is only right for *short* rows | **Rejected** |
| `ds4-e02-head-rope-wave` | Same treatment for `head_rms_norm_rope_tail_kernel`, the largest elementwise kernel at 7.27 ms per call, on the assumption that 256 threads per 128-element row wasted three quarters of the block on the `powf`/`cosf`/`sinf` tail | **Inert, then closed by arithmetic.** `head_dim` exceeds 256 so the guard never selected the new kernel. Recomputing with the real geometry: 262,144 rows x 512 floats, read twice and written once, is 1.6 GB, so 7.27 ms is **89% of roofline**. The 4.4x gap was an artefact of guessing `head_dim = 128`. The kernel was deleted rather than left behind a guard | **Rejected** |
| `ds4-t01-transpose-tile` | `dequant_q8_0_to_f16_transpose_kernel` gives each lane the private output address `i * out_dim + row`, so a wave's 32 stores land 8,192 B apart: 32 cache lines for 64 B of payload, and `(8192 / 256) % 16 == 0` puts all of them on one memory channel — 1.9 GB/s measured, about 25x off roofline. Stage a 32-`i` x 64-`row` tile in LDS, keep reads wave-contiguous, write 64 consecutive halves, and put `row` on `blockIdx.x` so concurrent blocks span the full `out_dim` row and cover all 16 channels | Bit-identical by construction, and the kernel disappears from the top of the trace. **PP-neutral** (328.47 -> 328.46 at 4K over three alternating pairs): the 2,453 ms it removes is startup, in the phase before the first measured frontier, not inside it. Kept for the ~2.3 s it takes off first-prompt latency, but it is not a prefill win | **Retained, no PP gain** |

### Where the remaining 5% is

Per-frontier kernel budget at the retained commit, from a two-chunk trace halved
(measured phase is 99.8% GPU-busy, 21,341 ms for two frontiers):

| Kernel | ms/frontier | Assessment |
| --- | ---: | --- |
| `mul_mat_q<IQ2_XXS,80>` gate/up | 2,188 | Bandwidth-bound at ~27 FLOP/byte; 20+ prior experiments, occupancy closed |
| `moe_down_q2K_hotlist_wmma_wide` | 1,874 | Bandwidth-bound at ~30 FLOP/byte; MTILES=4 is a two-sided optimum |
| `attention_mixed_heads32_wmma<true,true>` | 1,295 | **The one real target left.** See below |
| `mul_mat_q<Q8_0,80>` dense | 930 | Not yet roofline-checked |
| attention output A batched GEMM (rocBLAS) | 560 | 21.4 TFLOP/s, ~2x better than our kernel |
| attention output B (our WMMA kernel) | 556 | 21.2 TFLOP/s, was 10.7 |
| `attention_static_mixed_heads8_online` | 450 | 8 waves/SIMD, already single-pass |
| Tensile `MT96x96` + `MT32x32` projections | 485 | Tensile wins these; measured twice |
| `indexer_scores_wmma128` | 332 | 45,056 B LDS, **1 workgroup per CU**; -80 ms if it fits two |
| `head_rms_norm_rope_tail` | 312 | 89% of roofline |

`attention_mixed_heads32_wmma<true,true>` is the only large kernel with a
structural inefficiency left, and it is a hard one. It walks **two full passes**
over the KV rows: pass 0 for scores and the online softmax max/denominator, pass
1 for the probabilities and PV. The score cache spares pass 1 the QK matrix
multiply but not the KV staging, so every row block is staged, barriered and
LDS-written twice. At 58,368 B of LDS the kernel is one workgroup per CU, so all
32 waves walk that chain in lockstep.

The obvious fix is a single-pass FlashAttention formulation that rescales the
output accumulator by `exp(m_old - m_new)` instead of pre-computing a global
maximum. **That is why the two-pass form exists**: the rescale has to reach
individual *rows* of a rocWMMA F32 accumulator fragment, and rocWMMA exposes no
row accessor, so it needs either exact knowledge of the RDNA3 WMMA C-matrix lane
layout or a 32 KB LDS round trip per row block. Only about 6 KB of LDS is free.
Estimated payoff -400 to -600 ms per frontier, which is the whole remaining gap;
estimated risk high. This is the next thing to build, and it should be built
against a standalone harness rather than in-tree.

Also note the arithmetic that kills the cheap reading of this kernel: mma issue
is only about 14% of its runtime (8.7 ms of a 62 ms call, counting 32 cycles per
wave32 `v_wmma` per SIMD), so the 2-of-32-waves score phase is *not* the cost.
It is latency and barriers. Any proposal here must model the barrier chain.

### Queued experiments

Ordered by expected value. The BLAS seam that produced this session's gain is
now closed in both directions.

| ID | Experiment | Predicted | Gate |
| :--- | :--- | --- | --- |
| `ds4-a02-attn-single-pass` | Collapse `attention_mixed_heads32_wmma`'s two KV passes into one with an accumulator rescale, as analysed above. Halves KV staging, barriers and LDS writes per row block, and deletes the score cache and its memory. Blocked on reaching accumulator rows: either hard-code the RDNA3 WMMA C-matrix lane layout (fast, brittle, must be asserted against a reference) or free 32 KB of LDS by staging 8 KV rows at a time instead of 16 | **-400 to -600 ms per frontier**, i.e. the whole remaining gap to 400 | Not bit-exact; needs logit envelope + a likelihood corpus. Build against a standalone harness first |
| `ds4-a03-indexer-lds` | `indexer_scores_wmma128` uses 45,056 B of LDS with only 256 threads, so it is one workgroup and 4 waves per SIMD. Getting under 32,768 B doubles residency | -80 ms per frontier | Exact hashes likely |
| `ds4-d01-dense-q8-roofline` | `mul_mat_q<Q8_0,80>` is 930 ms per frontier over 215 calls per chunk and has never been roofline-checked. Establish its FLOP/byte before proposing anything | Unknown; a counter read either opens or closes it | n/a, measurement only |
| `ds4-m02-moe-64-row-tile` | The two MoE matmuls remain the **only** kernels that scale sub-linearly with chunk width: per token per layer, IQ2 gate/up costs 12.58 us at a 4,096-token chunk against 10.52 us at 8,192 (**-16.4%**), and Q2 down 10.68 against 9.46 (**-11.5%**); everything else is flat to within 1%. `ds4-m01` proved it is not column-tile fill, so it is the 64-row destination granularity and per-tile activation amortisation | -579 ms per frontier if 8,192-chunk efficiency is reached at 4,096 | Exact hashes if the tail arithmetic is unchanged |
| `ds4-q01-repack-iq2` | Row-pair-interleaved 64 B-aligned IQ2 repack, the last untried fork idea. Removes 2-byte-aligned split loads | Unknown | Exact hashes; in place, no extra resident bytes |
| `ds4-b01-blas-candidates` | The pinned hipBLASLt candidates (4/5/6) were chosen for determinism, not speed. Sweep the index and allow non-zero workspace | **Closed by the harness**: candidate 4 is within 4% of the best zero-workspace candidate on every shape, and a 64 MiB workspace unlocks no new solution — every candidate reports `workspaceSize = 0` | Closed |

### Closed by measurement, not worth building

`hc_expand4_add_moe_f16` (4.01 ms/call), `hc_expand4` (2.98), and
`attention_inverse_rope_pack_group_heads_f16` (3.89) all move 604-805 MB per
call and run at **83-85% of the 242 GB/s roofline**. `head_rms_norm_rope_tail`
is at 89%. The elementwise family looked like 1.5 s of easy upside and is
actually almost all at the memory wall; only `rms_norm_plain_f16` has real slack
(3.6x) and it is 216 ms per frontier.

## Rejected Experiments

Inherited from `ROCM_STRIX_HALO_OPTIMIZATION.md`; see that file for the full
reasoning. Not to be retried as-is.

| Experiment | Result |
| --- | --- |
| Expert-major axis ordering in the wide down kernel | **-36%**; our default order already had the L2 reuse the fork's change recovered |
| Q8_1 routed mid (ceiling probe) | +1.27% ceiling, below the cost of the two-kernel change |
| Pad wide-down LDS row pitch to `BK+8` | Conflicts 7.45% -> 34.85%; the pad broke fragment contiguity |
| Pad compressed KV / FP16-mirror stride to 1,280 B | Payload already equals stride, so already at full bandwidth |
| Prefill chunk 16,384 | +0.31% at 32K for +0.67 GiB resident |
| `MMQ Y=32` | IQ2 kernel time +18.2% |
| Q2 down MTILES 2 / 8 / 16, N=128, worklist, KSTEPS=2 | All neutral or slower; MTILES=4 is a true optimum |
| Attention full-KV double buffer, pass-1 pipeline, 16-head paired columns, split score/value | All slower or would not fit in 64 KiB LDS |
| IQ2 signed-grid tables, SoA repack, D2R FP variants, expert worklist | All slower or scheduling-sensitive |

### Closed directions

- **IQ2 occupancy is closed** without replacing the tile layout. The weight tile
  is `mmq_y * 76 * 4` B; reaching 3 wg/CU needs a 30.8% cut and only `mmq_y`
  moves that much, which costs +18.2%.
- **`iu8` matrix cores are already active** on the IQ2 path
  (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`), verified at runtime.
- **Both MoE matmuls are bandwidth-bound** at ~27 and ~30 FLOP/byte against the
  ~230 needed to saturate the matrix cores. Work there must move bytes.
  The Tensile projections are the opposite case and are compute-bound.
- **Global KV stride padding is closed** — see the memory-channel section of the
  worklog.
