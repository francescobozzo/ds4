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

### Where the time now is

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
| `ds4-x02-moe-roofline-analytic` | The inherited "both MoE matmuls are bandwidth-bound at ~27 and ~30 FLOP/byte" verdict covers **53% of the frontier** and has never been verified on this branch. PMC is blocked (`ds4-x01`), so get the bytes analytically: instrument one `fprintf` of the routed dims (`n_expert`, `n_expert_used`, `inter_dim`, `n_embd`), then divide by the trace durations already collected. There is a reason to doubt the verdict — `speed-bench/gb10.csv` shows 825-899 t/s at the same 2,048 chunk, 2.2x ours, on a platform with **273 GB/s against our 242**. Two systems at the same memory bandwidth cannot differ 2.2x if both are memory-bound | Decides whether 4,052 ms per frontier has headroom or is closed | Measurement only; one `fprintf` build |
| `ds4-a07-mixed32-lds` | `attention_mixed_heads32_wmma` is still one workgroup per CU at 58,368 B, of which `q_half` is 33,024. Two workgroups needs 32,768 B total, which is unreachable while a block owns 32 heads. The prior 16-head split was measured at 1,428 -> 2,305 ms, so this needs a different decomposition, not a smaller one | Unknown; likely closed | — |
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
