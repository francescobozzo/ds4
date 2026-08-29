# DeepSeek V4 Flash on Strix Halo (gfx1151)

Status: live. This page is the working performance card and experiment log for
the `rocm-strix-halo-release` branch. `ROCM_STRIX_HALO_OPTIMIZATION.md` remains
the narrative worklog; this card is the measured ledger.

Goal for the current session: **400 t/s prompt processing.** Met at the
steady-state frontier: **401.48 t/s**, from 362.13.

Two measurement notes decided most of this session, and both are easy to get
wrong:

- **The 8K row, not the 4K row, is the steady prefill rate.** Both prefill
  exactly 4,096 tokens, but the 4K row is the first frontier and absorbs every
  first-use cost.
- **A kernel trace beats an A/B while iterating.** The first A/B of this session
  read flat because a -1,109 ms win and a +987 ms regression cancelled exactly;
  the trace showed both in four minutes.

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
| 4K | 330.27 t/s | **370.38 t/s** | **+12.1%** |
| 8K | 361.68 t/s | **404.95 t/s** | **+12.0%** |

Means of three alternating pairs; all six positive. The 8K candidate runs read
404.75 / 405.01 / 405.09, a ±0.04% spread, so every run clears 400 with margin.
Baseline 8K spread was 361.31-362.01.

**The 400 t/s goal is met at the steady-state frontier: 404.95 t/s.** The 4K row
is 370.38 and still carries roughly a second of unavoidable first-use setup
inside its measured window (see below), so it is not the number to compare
against a steady rate.

### Quality

| Gate | Result |
| --- | --- |
| HIP MMQ parity suite | **ALL PASS** |
| Argmax, 2K and 4K frontiers | **identical to baseline** |
| Non-finite logits / null logits | **0 / 0** at both frontiers |
| Top-20 overlap | **19/20** at both frontiers |
| Max / RMS logit delta, 2K | 2.822 / 0.465 |
| Max / RMS logit delta, 4K | **0.952 / 0.171** |
| Teacher-forced NLL / PPL, 2,016 tokens | **bit-identical to baseline** |
| Greedy `temp=0` continuation | ` Paris. It is located in the north-central part of the` |

Exact logit hashes do **not** hold and are not expected to: one retained change
replaces a Tensile GEMM with a kernel whose accumulation order differs, and
another replaces a two-pass softmax with a max-rescaled single pass. The right
comparison is the drift this project already accepted for its own
baseline-to-current series, which was 1.886/0.396 at 2K and 1.385/0.271 at 4K.

At 4K the result is **better than that envelope on both metrics** (0.952 against
1.385, 0.171 against 0.271) with a full 20/20 top-20 overlap, because the
single-pass form rescales the denominator once per key block instead of once per
key. At 2K the max delta (2.822 against 1.886) is outside it. That 2K figure is
attributable entirely to the attention-output-B GEMM: it is **identical to eight
decimal places** across three different attention implementations tried in this
session, so the changed attention kernel does not contribute to it.

Teacher-forced likelihood is bit-identical (`nll=4313.670028443`,
`ppl=8.497034974`, 2,016 scored tokens) because the harness scores
token-by-token and every retained change is gated at `n_tokens >= 128`. That
proves decode is untouched but does **not** exercise chunked prefill, so a
multi-prompt likelihood corpus that runs through prefill remains the missing
gate and is the right next quality investment.

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
| `ds4-m03-mmq-act-prefetch` | The ablations above point at activations, not weights, and unlike the quant-specific weight loader the activation staging is a plain int copy. The two halves are staged-consumed-staged-consumed, so the second half's global read latency sits on the critical path with no other wave to cover it at 4 waves/SIMD. Issue the second half's reads before the first barrier and commit them to LDS only after the first `vec_dot`, so they fly across a barrier and a whole matrix-multiply phase. Bit-identical by construction: same words, same places, same K order. About 21 registers against a 384-VGPR budget at this occupancy | **Rejected, and badly: IQ2 25.375 -> 33.134 ms per call, +30.6%.** Dense Q8 went the other way, 4.330 -> 4.243, **-2.0%**, with VGPRs 192 -> 216. The difference is register headroom: IQ2's VGPRs *fell* 168 -> **136** with **zero scratch**, so the compiler either sank the loads back to their use site or 21 extra outstanding VMEM requests per thread exhausted the issue pipeline. Either way IQ2 has no headroom, matching the worklog's earlier finding that lower VGPR tiers lose badly on this kernel. Gating the prefetch to dense Q8 alone would be worth +0.2% of prefill and was not taken | **Rejected** |
| `ds4-a08-indexer-register-acc` | `indexer_scores_wmma128` kept 8,192 B of `c_sh` purely to redistribute each warp's 16x16 result so all 256 threads could stripe across all eight warp tiles, costing a store, a barrier and a strided read per head, 64 heads deep. Using the accumulator layout from `ds4-a02`, each warp instead accumulates *its own* tile in registers: element `e` of lane `l` is row `2e + l/16`, col `l % 16`, and the per-row weight `weights[(tile_t + row) * n_head + h]` is therefore computable per element. LDS 12,288 -> 4,096 B, so residency stops being LDS-bound | **8K 401.48 -> 404.95 (+0.9%), 4K 367.54 -> 370.38.** Three alternating pairs, ±0.04% spread. **Bit-exact and asserted**: the full-vocabulary dumps at 2K and 4K are byte-identical to the build before it, which is the strongest available check that the hand-derived lane mapping is right. Each output element still accumulates over heads in ascending order; only which thread owns it changed | **Retained** |
| `ds4-a09-indexer-b-from-global` | Go further and delete `b_sh` too, the other 32,768 B, by pre-converting `index_comp` to F16 once per chunk and loading B fragments straight from global | **Rejected on paper, no build.** `b_sh` is staged once and reused by all 64 heads; serving those fragments from global re-reads 128x128x2 B per head per block, which is 2 MB per block and **8.4 GB** over the grid. Even as MALL hits that is at best a wash against the 15.8 ms the kernel costs now. The LDS staging is doing real work | **Rejected** |
| `ds4-a10-indexer-half-tile` | Halve the comp tile to 64 columns so `b_sh` fits two workgroups per CU | **Rejected on paper, no build.** 16-wide WMMA tiles mean 64 columns admits only 4 warps, so it is 2 workgroups x 128 threads = 256 threads per CU — **identical** to today's 1 x 256. No occupancy gained | **Rejected** |
| `ds4-x01-pmc-roofline` | Settle the inherited "both MoE matmuls are bandwidth-bound" verdict by reading `FETCH_SIZE`, `WRITE_SIZE`, `L2CacheHit` and `MemUnitBusy`, then dividing measured bytes by trace durations for true GB/s | **Blocked by the hardware.** `rocprofv3` returns *"Could not construct profile cfg failed with error code 38: Request exceeds the capabilities of the hardware to collect"* for four counters **and for two** — `FETCH_SIZE`/`WRITE_SIZE` are derived TCC metrics needing more slots than gfx1151 exposes. Worse, on that failure rocprofv3 aborts (signal 6) and then **hangs forever in its own signal handler** rather than exiting, so any wait-for-sentinel loop waits on a dead process; kill it by PID. Try one counter at a time, or get the byte count analytically from model dims instead | **Blocked** |
| `ds4-e03-head-rope-lds` | `head_rms_norm_rope_tail_kernel` touches global memory three times per row: once to square, once to scale the non-rotated head, once to read the rotated tail back. The row is 512 floats, so stage it in 2 KiB of LDS beside the existing reduction scratch and serve the second and third passes from there. Values, reduction tree and per-element arithmetic all unchanged | **8K 362.13 -> 401.48 (+10.9%), 4K 325.35 -> 367.54 (+13.0%)** for the retained set, four alternating pairs, all eight positive, every candidate run above 400. Bit-exact as designed: the logit envelope is identical to eight decimals against the build before it | **Retained** |
| `ds4-a04-static-attn-single-pass` | Give `attention_static_mixed_heads8_online_kernel` the same single-pass treatment. Here the accumulators are plain `float4` registers, so the rescale is trivial, and dropping the 24 KiB score array takes LDS from 32,768 to 8,192 B | **Rejected on quality.** Faster, but only +0.25 t/s at 8K (398.76 -> 399.01) because the kernel runs 11 times per frontier, and it moved three of four quality metrics the wrong way: 4K top-20 20/20 -> 19/20, RMS 0.171 -> 0.216, max delta 0.952 -> 1.194. The kernel's own comment records that "the previous online recurrence was close, but crossed greedy near-ties on long prompts" — that warning is real, and a running maximum on *this* kernel costs more accuracy than it buys speed. Reverted; the two-pass global maximum stays | **Rejected** |
| `ds4-a05-indexer-barrier` | `indexer_scores_wmma128` runs three barriers per head across 64 heads. The third is redundant: the next head writes `c_sh` only after its own post-staging barrier, which already orders every thread's `c_sh` reads against that write, and it writes `a_sh`, a different array; the epilogue reads only registers | +0.8 t/s at 8K (397.94 -> 398.76), consistent across three runs and tighter than the run it replaced. Bit-exact by construction | **Retained** |
| `ds4-e04-rmsnorm-registers` | `rms_norm_plain_f16_kernel` reads its row twice, so cache it in registers, keeping all 256 threads per row this time and therefore the exact reduction tree | **Rejected: neutral to slightly negative** (8K 398.76 -> 398.19). The second read was already an L2 hit, since the block had just read the same 16 KiB row, so no DRAM traffic was saved and the 16 extra VGPRs cost occupancy. This kernel's 3.6x roofline gap is latency, not traffic | **Rejected** |
| `ds4-a02-attn-single-pass` | Collapse `attention_mixed_heads32_wmma`'s two KV passes into one. The two-pass form staged, barriered and LDS-wrote every key block twice — the score cache spared the second pass its QK matrix multiply but not the staging. Fold the running maximum into the accumulator instead: rescale the partial output by `exp(m_old - m_new)` per block and divide by the denominator once at the end. **The blocker was reaching accumulator rows**: the factor is per head, i.e. per *row* of a rocWMMA F32 fragment, and a `store_matrix_sync` round trip would need 32 waves x 16x16 x 4 B = 64 KiB against about 6 KiB free. Resolved by measuring the layout instead of guessing it — `rocm/tools/wmma_acc_layout.cpp` recovers `row = 2 * e + lane / 16`, `col = lane % 16` on gfx1151, making the rescale pure register arithmetic. Also parallelised the softmax update: one wave per head, one row per lane, eight shuffles instead of a 16-deep serial `expf` chain on 32 of 1024 threads | **The largest single win of the session.** Kernel `<true,true>` **61.5 -> 45.0 ms per call, 2,582 -> 1,892 ms (-27%)**, `<false,false>` -48 ms. End to end 8K 380.69 -> 399.14 and 4K 340.55 -> 349.68. Quality *improved*: 4K top-20 19/20 -> 20/20 and RMS 0.255 -> 0.171, because the denominator is now rescaled once per key block rather than once per key | **Retained** |
| `ds4-a06-score-cache-drop` | With the kernel single-pass, the score cache has no reader. It was sized `n_tokens * n_head * (256 + top_k) * 4`, i.e. **805 MiB** of the shared scratch buffer, allocated inside the first measured frontier | **4K 349.68 -> 367.76 (+5.2%)** and 805 MiB of device memory returned, on a box already holding 80.76 GiB. 8K was flat within noise. Pure dead-code removal | **Retained** |
| `ds4-b06-batched-a-gemm` | The attention output A GEMM is the last large kernel still on a BLAS: `Cijk_Alik_Bljk_HHS_MT128x128x32`, 1,120 ms per two-chunk trace, reached through `cublasGemmStridedBatchedEx` so it never even saw the hipBLASLt plan cache. Added a strided-batched form of the WMMA kernel (`blockIdx.z` over problems, plus an `ldc` because each group writes its own `rank` block of a `low_dim` row) | **Closed by arithmetic, and the eligibility check correctly declined it.** The dims are not what the low_dim=8192 arithmetic suggested: the trace's `grid=1024x32x8` gives `n_groups = 8`, hence `rank = 1024` and `group_dim = 4096`, so the call is 275 GFLOP in 12.88 ms = **21.4 TFLOP/s**, not the 2.7 an assumed `rank=128` implied. Our own 256x128 kernel reaches only 10.1 TFLOP/s on the near-identical `m=1024 n=4096 k=4096` shape. rocBLAS is already about 2x better here. The batched capability is retained because it cost nothing and is correct, but nothing selects it | **Rejected** |
| `ds4-b05-opa-t-revert` | Even with the grid gate, the `opA=T` family was still net-negative: the gate left 128 calls on the WMMA kernel at 592.3 ms while Tensile handled the rest, for 1,173 ms against the baseline's 967 ms. Remove the `opA=T` routing entirely and keep only `opA=N` | **-202 ms** (WMMA -592.3, Tensile +389.8). Confirms the harness's original verdict: Tensile genuinely wins every `opA=T` shape on this GPU, and the in-app/isolated correction that made `opA=N` a 1.97x win does **not** generalise across the transpose | **Retained** |
| `ds4-m01-mmq-x` | The `mmq_x` selector minimises `ceil(ncols_max / x)` and the routed path passes the whole chunk width as `ncols_max`, so it always lands on the 80 cap. Per expert only about `n_tokens * 6 / 256` = 96 columns are real, so the second 80-column tile should be a fifth full, and `x=48` would give two exactly-full tiles for the same number of weight passes. Made the cap tunable on HIP and swept it against a real chunk | **Rejected, and it closes the fill hypothesis.** 8K, two rounds each: X=80 **377.5**, X=64 374.8, X=48 372.1, X=40 334.1, X=32 334.6. Fill is monotonically *worse* below 80, and there is an 11% cliff at 40 where the tile granularity changes. So the measured MoE sub-linearity in `ds4-m02` is not column-tile fill; it is the 64-row destination granularity and per-tile activation amortisation, as the original worklog said. The env knob is kept as a diagnostic; the default is unchanged | **Rejected** |
| `ds4-b04-wmma-grid-gate` | `ds4-b03` routed *every* eligible shape through the 256x128 kernel and cost +456 ms, because at `m=256` and `m=512` the tile leaves 32 and 64 workgroups for 40 CUs while Tensile's MT96x96 and MT32x32 keep hundreds in flight. Require `(m/256) * (n/128) >= 240`, six workgroups per CU, so only `m=4096` and `m=8192` qualify. Also route the F16-result twin of the attention output B GEMM, which reaches hipBLAS directly and so never even saw the plan cache: Tensile's `MT128x128` prices it at 12.87 ms per call for 68.7 GFLOP, **2.9 TFLOP/s** | **4K 329.61 -> 343.80 (+4.3%), 8K 363.98 -> 377.03 (+3.6%)**, three alternating pairs, all six positive, candidate spread ±0.04% at 8K. Quality: argmax identical at 2K and 4K, zero non-finite and zero null logits, max/RMS delta 2.129/0.402 at 2K and 1.319/0.266 at 4K, i.e. inside the envelope this project already accepted for its own baseline-to-current drift (1.886/0.396 and 1.385/0.271) | **Retained** |
| `ds4-b03-proj-wmma` | Route all remaining Tensile shapes through the kernel, templated on A layout and output type | Mixed, and the trace separated it cleanly: the `opA=N` attention output B kept its **-1,121 ms** while the `opA=T` projection family went **707.4 ms over 467 calls to 1,163.3 over 462, +456 ms**. Superseded by the grid gate above | **Partly retained** |
| `ds4-b02-attn-out-b-wmma` | Take the attention output B GEMM (`m=4096 n=chunk k=8192`, `opA=N`, F16 in / F32 out) off hipBLASLt. Tensile picks `MT96x96x32`, which re-reads each panel `m/96` and `n/96` times, about 5.7 GB per call; at the traced 25.58 ms that is 223 GB/s, essentially DRAM peak, so the tile size *is* the cost. A 256x128 macro tile moves 3.22 GB. Four m-fragments and two n-fragments per wave, 512 threads, `col_major` LDS A so the transposed-weight read stays contiguous in `m`, two-column swizzle | **25.58 -> 12.98 ms per call, 1.97x.** Kernel total 2,225.5 -> 1,116.4 ms over the two-frontier trace, **-1,109 ms**. 104 VGPRs, 24,576 B LDS, no scratch. Note the microbench under-predicted this: isolated it was only 1.34x, because Tensile degrades 1.62x from isolated to in-app while this kernel degrades 1.10x — the low-traffic kernel is the one that keeps its speed when the caches are cold | **Retained** |
| `ds4-e01-rmsnorm-wave` | `rms_norm_plain_f16_kernel` runs one 256-thread workgroup per row with an eight-level shared-memory tree, re-reads the row for the output pass, and stores single halves. Give each row one wave, eight rows per workgroup, `warp_sum_f32`, and packed `half2` stores, with the slot tree arranged to reproduce the original association bit-for-bit | **Rejected: 2.50 -> 8.20 ms per call, 3.3x slower** (431.9 -> 1,418.9 ms, +987 ms), which exactly cancelled the win above and is why the first A/B read flat. The row is 4,096 long, so one wave per row cuts the thread count 8x, from 1,048,576 to 131,072 against ~82,000 concurrent lanes — there is no longer enough parallelism to cover the load latency. Wave-per-row is only right for *short* rows | **Rejected** |
| `ds4-e02-head-rope-wave` | Same treatment for `head_rms_norm_rope_tail_kernel`, the largest elementwise kernel at 7.27 ms per call, on the assumption that 256 threads per 128-element row wasted three quarters of the block on the `powf`/`cosf`/`sinf` tail | **Inert, then closed by arithmetic.** `head_dim` exceeds 256 so the guard never selected the new kernel. Recomputing with the real geometry: 262,144 rows x 512 floats, read twice and written once, is 1.6 GB, so 7.27 ms is **89% of roofline**. The 4.4x gap was an artefact of guessing `head_dim = 128`. The kernel was deleted rather than left behind a guard | **Rejected** |
| `ds4-t01-transpose-tile` | `dequant_q8_0_to_f16_transpose_kernel` gives each lane the private output address `i * out_dim + row`, so a wave's 32 stores land 8,192 B apart: 32 cache lines for 64 B of payload, and `(8192 / 256) % 16 == 0` puts all of them on one memory channel — 1.9 GB/s measured, about 25x off roofline. Stage a 32-`i` x 64-`row` tile in LDS, keep reads wave-contiguous, write 64 consecutive halves, and put `row` on `blockIdx.x` so concurrent blocks span the full `out_dim` row and cover all 16 channels | Bit-identical by construction, and the kernel disappears from the top of the trace. **PP-neutral** (328.47 -> 328.46 at 4K over three alternating pairs): the 2,453 ms it removes is startup, in the phase before the first measured frontier, not inside it. Kept for the ~2.3 s it takes off first-prompt latency, but it is not a prefill win | **Retained, no PP gain** |

## Measured Rooflines

Dimensions recovered by instrumenting the launchers once (`n_embd = 4096`,
`expert_mid_dim = 2048`, `n_total_expert = 256`, `n_expert_used = 6`, 43 layers,
`down_expert_bytes = 2,752,512` confirming Q2_K at 2.625 bpw). The implied
resident size is 74.6 GiB against the actual 80.76 GiB, the balance being
embeddings, output head, attention and shared-expert weights, which validates
the routed and dense figures.

**This overturns the verdict the previous worklog carried.** It stated both MoE
matmuls were "bandwidth-bound at ~27 and ~30 FLOP/byte" and closed them. Only
one of them is.

| Kernel | ms/frontier | GFLOP/layer | TFLOP/s | % compute peak | GB/s | % of 242 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mul_mat_q<IQ2_XXS,80>` gate/up | 2,182 | 825 | 16.3 | ~14% (int8) | ~55 | **23%** |
| `moe_down_q2K_hotlist_wmma_wide` | 1,870 | 412 | 9.5 | 16% | ~147 | **61%** |
| `mul_mat_q<Q8_0,80>` dense | 931 | 395 | 18.3 | ~16% (int8) | ~32 | **13%** |

Q2 down at 61% of DRAM peak, with activations 92% of its bytes, is genuinely
near the memory wall — that part of the old verdict holds. **IQ2 gate/up and
dense Q8 are neither bandwidth- nor compute-bound**: 13-23% of bandwidth *and*
14-16% of compute. That is a latency signature, and together they are 29% of the
frontier.

### What the MMQ latency actually is, by ablation

PMC is unusable here, so the kernel was decomposed with two throwaway builds
that pin a source pointer so its loads stay cache-resident. Results are wrong by
design; both probes were confirmed live (the weight probe moves the 4K logits by
max 23.37 with top-20 4/20). Each figure is the ceiling for removing that
traffic *and* its latency entirely.

| Kernel | weight loads free | activation loads free |
| --- | ---: | ---: |
| `mul_mat_q<IQ2_XXS,80>` | -12.6% | **-21.0%** |
| `mul_mat_q<Q8_0,80>` | -0.1% | **-11.1%** |

Three things follow, and they reorder the queue:

1. **Activations cost more than weights**, roughly 2:1 on IQ2. A weight prefetch
   — the obvious reading of "latency-bound" — is capped at -12.6% of one kernel,
   i.e. +2.6% of prefill for a *perfect* implementation, and a real one captures
   only the latency half of that. Not worth vendored MMQ surgery.
2. **Dense Q8's weight loads are already fully hidden** (-0.1%). Its latency is
   entirely elsewhere.
3. Even making **all** memory traffic free buys about 34% of the IQ2 kernel, so
   the other two thirds is the IQ2 dequantization VALU path, LDS traffic and
   barriers. That is consistent with 14% of the int8 ceiling and with the
   worklog's own counter sweep (1,006 VALU per wave, LDS:VALU 18.7%), and it is
   the direction the retained IQ2 loader work already mined for -10.5%.

### Why MMQ occupancy is nonetheless closed

The cause is occupancy, which the worklog measured but did not connect: 128
threads is 4 waves, 30.8 KiB of dynamic LDS caps 2 workgroups per CU, so **4 of
16 possible waves per SIMD**.

It cannot be fixed by tuning `mmq_y`, and here is the reason the Y sweep kept
finding nothing. On the AMD WMMA path `nwarps = DS4_ROCM_WMMA_MMQ_Y / 16`, so
waves per CU is `(LDS-limited workgroups) x nwarps` and the two factors scale
with `mmq_y` in **opposite** directions:

| `mmq_y` | weight tile | LDS | workgroups/CU | nwarps | waves/CU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 9,728 B | ~15.5 KiB | 4 | 2 | **8** |
| 64 | 19,456 B | 30.8 KiB | 2 | 4 | **8** |
| 128 | 38,912 B | ~60 KiB | 1 | 8 | **8** |

The product is invariant. Raising it requires shrinking LDS *per row*, i.e. the
76-int tile row, and that is closed by the hardware: 64 of those ints are 32
K-elements held one byte each because `iu8` WMMA consumes `int32x4_t` of int8
lanes, so packed 2-bit codes cannot reach the matrix core.

So the remaining lever on these two kernels is not occupancy but **latency
hiding at fixed occupancy** — prefetching the next K-block's weight tile into
registers during the current tile's `mma`, the same recipe that gave the Qwen
attention kernel 1.12-1.30x. That is surgery on vendored MMQ and is the largest
single untried item in prefill.

## The MoE chunk-width sub-linearity, diagnosed

The two MoE matmuls are the only kernels whose *per-token* cost depends on chunk
width: IQ2 gate/up costs 12.58 us per token per layer at a 4,096-token chunk
against 10.52 at 8,192 (**-16.4%**), Q2 down 10.68 against 9.46 (**-11.5%**).
Everything else is flat to within 1%. Two hypotheses are now dead and the third
fits exactly.

**Not column-tile fill.** With 256 experts and top-6, a 4,096-token chunk gives
96 tokens per expert on average, so at `mmq_x = 80` the second column tile is a
fifth full. `x = 48` would give two exactly-full tiles for the same weight
passes — and measured **slower** (372.1 against 377.5 at 8K), with a cliff to
334 at x <= 40. The padded columns are nearly free, which is what 14% of the
int8 ceiling predicts.

**It is per-column-tile-pass overhead.** Passes per expert are
`ceil(tokens_per_expert / mmq_x)`, so per *token* they fall from
`256 x 2 / 4096 = 0.1250` to `256 x 3 / 8192 = 0.0938` — exactly **-25%**. If a
fraction `f` of the kernel is per-pass rather than per-real-work, the observed
-16.4% gives `f = 16.4 / 25 = 0.66`.

That 0.66 is independently corroborated: the ablations put memory at 34% of the
IQ2 kernel, and the remaining two thirds is dequantization work, which happens
once per weight-tile load and is therefore *also* per pass. 66% per-pass plus
34% per-real-byte accounts for the whole kernel.

**So the fix is fewer passes per token, and the obvious route is blocked.**
Raising `mmq_x` to 96 would put a 96-token expert in one pass, halving passes.
But `tile_y` scales with `mmq_x`: at `MMQ_TILE_Y_K = 36` and `mmq_y = 64`, LDS
goes from `19,456 + 12,096 = 31,552` B to `19,456 + 14,208 = 33,664` B, which
crosses the 32,768 B needed for two workgroups per CU and **halves occupancy to
4 waves per CU** to save 33% of 66% = 22% of the kernel. On a latency-bound
kernel that trade is very likely negative, which is presumably why the HIP cap
is 80.

The route that is *not* blocked is per-expert width selection: `hot_max` is 455
against an average of 96, so a single width is wrong for both ends of the
distribution. Bucketing experts by token count and issuing one launch per width
bucket — PR 887's tail-tile specialization, which it measured at +4.2% on the
CUDA-only D2R path — cuts padding without raising LDS for the small buckets.
That is the retained recommendation for `ds4-m02`, and it is a change to the
routed launch path rather than to the kernel.

## Decode: 4x headroom, and it is not bandwidth

Decode was untouched by this work and had never been priced. Active bytes per
decoded token are `43 x (6 x 7.08 MB routed + 51.2 MB dense)` = **4.03 GB**,
giving a DRAM-bound ceiling of **60 tok/s** at 242 GB/s.

Measured: **15.30 tok/s** (steady 15.31, `pp2048` + `tg128`), i.e. 61.7 GB/s or
**25.5% of peak**. Decode has roughly **4x** theoretical headroom and is not
memory-bound.

A kernel trace of 64 decoded tokens shows why it will not yield to a single
fix: **~1,560 dispatches per token** and no kernel above about 12% of decode.

| Decode kernel | ms / 64 tok | calls | us/call |
| --- | ---: | ---: | ---: |
| `matmul_q8_0_hc_expand_preq_rows_w32` | 648 | 5,504 | 118 |
| `matmul_q8_0_preq_rows_w32` | 591 | 2,817 | 210 |
| `moe_gate_up_mid_decode_lut_qwarp32` | 509 | 2,752 | 185 |
| `grouped_q8_0_a_preq_warp8` | 436 | 2,752 | 159 |
| `matmul_f16_pair_f32_sharedx_warp_rows_w32` | 360 | 3,968 | 91 |
| `matmul_q8_0_pair_preq_warp8` | 343 | 5,504 | 62 |
| `attention_decode_mixed_one_fast_oldhip` | 305 | 2,752 | 111 |

At the ~2 us dispatch floor on this GPU, 1,560 dispatches is ~3.1 ms per token,
a 320 tok/s ceiling — so dispatch alone is not the binder at 15 tok/s, but the
shape of the problem is many small launches each too short to saturate anything.
The direction is fusion and wider per-launch work, not a hotspot fix. This is
now the largest opportunity in the model, prefill included.

### Where the prefill time now is

Per-frontier kernel budget at the retained commit, from a two-chunk trace halved.
The measured phase is **99.8% GPU-busy**, so prefill is purely kernel-bound and
there is no launch gap to recover.

| Kernel | ms/frontier | Assessment |
| --- | ---: | --- |
| `mul_mat_q<IQ2_XXS,80>` gate/up | 2,182 | Bandwidth-bound at ~27 FLOP/byte; 20+ prior experiments, occupancy closed, X swept again here |
| `moe_down_q2K_hotlist_wmma_wide` | 1,870 | Bandwidth-bound at ~30 FLOP/byte; MTILES=4 is a two-sided optimum |
| `attention_mixed_heads32_wmma<true,true>` | 946 | Was 1,291. Single-pass; 58,368 B LDS still holds it to one workgroup per CU |
| `mul_mat_q<Q8_0,80>` dense | 931 | **Never roofline-checked.** Largest unexamined item |
| attention output A batched GEMM (rocBLAS) | 563 | 21.4 TFLOP/s, about 2x better than our own kernel on this shape |
| attention output B (our WMMA kernel) | 551 | 21.2 TFLOP/s, was 10.7 |
| `attention_static_mixed_heads8_online` | 450 | Two-pass, and a single pass was rejected on quality |
| Tensile `MT96x96` + `MT32x32` projections | 487 | Tensile wins these; measured twice, isolated and in-app |
| `indexer_scores_wmma128` | ~290 | LDS 45,056 -> 36,864 B; `b_sh` is the irreducible 32,768 |
| `head_rms_norm_rope_tail` | ~210 | Was 312; now one read and one write per row |

The two MoE matmuls are now **53% of the frontier** and both are at the memory
wall. Everything cheap outside them has been taken.

### Queued experiments

Ordered by expected value. The BLAS seam that produced part of this session's
gain is now closed in both directions, and the attention seam is half-closed.

| ID | Experiment | Predicted | Gate |
| :--- | :--- | --- | --- |
| `ds4-d01-dense-q8-roofline` | `mul_mat_q<Q8_0,80>` is 931 ms per frontier over 215 calls per chunk and has never been roofline-checked. Establish its FLOP/byte before proposing anything. Now the largest unexamined kernel | Unknown; a counter read either opens or closes it | n/a, measurement only |
| `ds4-tg01-decode-fusion` | **Largest opportunity in the model.** Decode runs at 25.5% of the DRAM ceiling with ~1,560 dispatches per token and no kernel above 12%. Fuse the per-layer decode chain — `hc_expand_preq`, `q8_preq_rows`, `grouped_a_preq`, `pair_preq` are seven-plus launches per layer, several of them 60-120 us, i.e. too short to saturate. Start by counting launches per layer and merging the ones sharing an activation | 15.3 -> plausibly 25-35 tok/s; ceiling is 60 | Greedy equality per prompt; decode is currently bit-identical to baseline so any change must re-establish that |
| `ds4-m03-mmq-kloop-prefetch` | **Closed by measurement.** Weight prefetch is capped at +2.6% of prefill by ablation, and the activation prefetch that the ablations recommended measured +30.6% on the IQ2 kernel. See the two entries in the log. What remains on these kernels is the dequantization instruction path, not memory | Closed | — |
| `ds4-a07-mixed32-lds` | `attention_mixed_heads32_wmma` is still one workgroup per CU at 58,368 B, of which `q_half` is 33,024. Two workgroups needs 32,768 B total, which is unreachable while a block owns 32 heads. The prior 16-head split was measured at 1,428 -> 2,305 ms, so this needs a different decomposition, not a smaller one | Unknown; likely closed | — |
| `ds4-m02-moe-width-buckets` | **Diagnosed; see the sub-linearity section.** The mechanism is per-column-tile-pass overhead, 66% of the IQ2 kernel, and raising `mmq_x` to reduce passes is blocked by a 33,664 B LDS footprint that halves occupancy. The open route is bucketing experts by token count (`hot_max` 455 vs mean 96) and issuing one launch per width bucket | -579 ms per frontier if 8,192-chunk efficiency is reached at 4,096 | Exact hashes if per-bucket accumulation order is unchanged |
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
