/* Standalone gfx1151 F16 GEMM harness for the DS4 prefill projection shapes.
 *
 * The production ROCm path sends every F16 projection and the attention output
 * B GEMM to hipBLASLt with a pinned heuristic candidate and zero workspace.
 * A kernel trace shows two of the four Tensile kernels it selects are the
 * non-WMMA `MI16x16x1` source kernels, so those shapes run on the vector ALU
 * at roughly a fifth of the matrix-core ceiling.
 *
 * This harness measures, per shape:
 *   - every hipBLASLt heuristic candidate, with and without workspace,
 *   - the production pinned candidate, and
 *   - blocked rocWMMA kernels at several tile geometries,
 * and reports the error of each against candidate 0 so a faster kernel is only
 * accepted when it is also numerically sane.
 *
 * Build:
 *   hipcc -O3 -std=c++17 --offload-arch=gfx1151 \
 *     -I<rocwmma>/include -I<hipblaslt>/include -I<hipblas-common>/include \
 *     -o /tmp/f16_gemm_bench rocm/tools/f16_gemm_bench.cpp \
 *     -L<hipblaslt>/lib -lhipblaslt
 *
 * Layout convention, matching ds4_rocm_hipblaslt.cuh exactly:
 *   opA=T  A is W[m][k]   row-major, out[n][m] = sum_k W[m][k]  * X[n][k]
 *   opA=N  A is Wt[k][m]  row-major, out[n][m] = sum_k Wt[k][m] * X[n][k]
 *   B      X[n][k]        row-major
 *   C      out[n][m]      row-major F32  (= col-major [m][n], ld = m)
 */

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                          \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__,      \
                    hipGetErrorString(e_));                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* ---------------------------------------------------------------- kernels */

/* Blocked F16 GEMM.
 *
 * A and B are staged so that both the global read and the LDS write are
 * contiguous in the source's fast axis: for a row-major W[m][k] the LDS tile is
 * [m][k] and the fragment is row_major, for a row-major Wt[k][m] it is [k][m]
 * and col_major. B is always X[n][k] row-major, so its tile is [n][k] with a
 * col_major matrix_b fragment. PAD adds slack halves to each LDS row pitch.
 *
 * Each wave owns a WMF x WNF grid of 16x16 accumulators, held in registers for
 * the whole K loop, so the accumulation order per output element is a plain
 * ascending walk over K and does not depend on the tile geometry. */
template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WMF, uint32_t WNF,
          bool A_ROWMAJOR, uint32_t PAD, uint32_t SWZ>
__global__ __launch_bounds__((BM / 16u / WMF) * (BN / 16u / WNF) * 32u) void
gemm_f16_wmma_kernel(float *__restrict__ C, const __half *__restrict__ A,
                     const __half *__restrict__ B, uint32_t M, uint32_t N,
                     uint32_t K) {
    constexpr uint32_t WAVES_M = BM / 16u / WMF;
    constexpr uint32_t WAVES_N = BN / 16u / WNF;
    constexpr uint32_t WAVES = WAVES_M * WAVES_N;
    constexpr uint32_t THREADS = WAVES * 32u;
    /* A tile pitch is along the source's contiguous axis. */
    constexpr uint32_t A_PITCH = (A_ROWMAJOR ? BK : BM) + PAD;
    constexpr uint32_t A_LINES = A_ROWMAJOR ? BM : BK;
    constexpr uint32_t B_PITCH = BK + PAD;

    __shared__ __half shA[A_LINES * A_PITCH];
    __shared__ __half shB[BN * B_PITCH];

    /* Swizzle the block index so that SWZ x SWZ neighbourhoods of output tiles
     * are resident together, which is what keeps the A and B panels in the
     * 32 MB MALL instead of re-streaming them per block row. */
    uint32_t bm, bn;
    if (SWZ <= 1u) {
        bm = blockIdx.x;
        bn = blockIdx.y;
    } else {
        const uint32_t tiles_m = gridDim.x;
        const uint32_t group = SWZ * tiles_m;
        const uint32_t linear = blockIdx.y * tiles_m + blockIdx.x;
        const uint32_t gid = linear / group;
        const uint32_t within = linear - gid * group;
        const uint32_t rows = min(SWZ, gridDim.y - gid * SWZ);
        bm = within / rows;
        bn = gid * SWZ + (within - bm * rows);
    }

    const uint32_t m0 = bm * BM;
    const uint32_t n0 = bn * BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t wm = wave % WAVES_M;
    const uint32_t wn = wave / WAVES_M;

    using frag_a = rocwmma::fragment<
            rocwmma::matrix_a, 16, 16, 16, __half,
            typename std::conditional<A_ROWMAJOR, rocwmma::row_major,
                                      rocwmma::col_major>::type>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half,
                                     rocwmma::col_major>;
    using frag_c =
            rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

    frag_c acc[WMF][WNF];
    for (uint32_t i = 0; i < WMF; i++)
        for (uint32_t j = 0; j < WNF; j++) rocwmma::fill_fragment(acc[i][j], 0.0f);

    /* Eight halves per thread per group keeps every global load a 16-byte
     * dwordx4 and every LDS write contiguous across the wave. */
    constexpr uint32_t A_GROUPS = BM * BK / 8u;
    constexpr uint32_t B_GROUPS = BN * BK / 8u;
    /* Contiguous run length in the source, in units of 8 halves. */
    constexpr uint32_t A_RUN = (A_ROWMAJOR ? BK : BM) / 8u;
    constexpr uint32_t B_RUN = BK / 8u;

    for (uint32_t k0 = 0; k0 < K; k0 += BK) {
        for (uint32_t g = tid; g < A_GROUPS; g += THREADS) {
            const uint32_t run = g % A_RUN;
            const uint32_t line = g / A_RUN;
            uint64_t src;
            if (A_ROWMAJOR) {
                src = (uint64_t)(m0 + line) * K + k0 + run * 8u;
            } else {
                src = (uint64_t)(k0 + line) * M + m0 + run * 8u;
            }
            *reinterpret_cast<uint4 *>(&shA[line * A_PITCH + run * 8u]) =
                    *reinterpret_cast<const uint4 *>(&A[src]);
        }
        for (uint32_t g = tid; g < B_GROUPS; g += THREADS) {
            const uint32_t run = g % B_RUN;
            const uint32_t line = g / B_RUN;
            *reinterpret_cast<uint4 *>(&shB[line * B_PITCH + run * 8u]) =
                    *reinterpret_cast<const uint4 *>(
                            &B[(uint64_t)(n0 + line) * K + k0 + run * 8u]);
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < BK; kk += 16u) {
            frag_a fa[WMF];
            frag_b fb[WNF];
            for (uint32_t i = 0; i < WMF; i++) {
                const uint32_t m = (wm * WMF + i) * 16u;
                if (A_ROWMAJOR) {
                    rocwmma::load_matrix_sync(fa[i], &shA[m * A_PITCH + kk],
                                              A_PITCH);
                } else {
                    rocwmma::load_matrix_sync(fa[i], &shA[kk * A_PITCH + m],
                                              A_PITCH);
                }
            }
            for (uint32_t j = 0; j < WNF; j++) {
                const uint32_t n = (wn * WNF + j) * 16u;
                rocwmma::load_matrix_sync(fb[j], &shB[n * B_PITCH + kk],
                                          B_PITCH);
            }
            for (uint32_t i = 0; i < WMF; i++)
                for (uint32_t j = 0; j < WNF; j++)
                    rocwmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (uint32_t i = 0; i < WMF; i++) {
        for (uint32_t j = 0; j < WNF; j++) {
            const uint32_t m = m0 + (wm * WMF + i) * 16u;
            const uint32_t n = n0 + (wn * WNF + j) * 16u;
            rocwmma::store_matrix_sync(C + m + (uint64_t)n * M, acc[i][j], M,
                                       rocwmma::mem_col_major);
        }
    }
}

__global__ static void fill_kernel(__half *p, uint64_t n, uint32_t seed) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s = (uint32_t)(i * 2654435761u) ^ seed;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    /* Small symmetric values so a K=16384 reduction stays inside F32. */
    p[i] = __float2half(((float)(s & 0xFFFFu) / 65535.0f - 0.5f) * 0.25f);
}

/* --------------------------------------------------------------- harness */

struct Shape {
    const char *label;
    uint32_t m, n, k;
    bool op_t; /* true: A is W[m][k]; false: A is Wt[k][m] */
    uint32_t pinned;
};

static const char *g_filter = nullptr;

/* COPIES independent A/B pairs, cycled once per timed iteration.
 *
 * With a single pair the 32 MB MALL retains a large share of a 67 MB operand
 * between iterations, so an isolated measurement rewards small macro tiles that
 * re-read their panels many times. In the engine those panels are cold: the
 * attention output B GEMM measures 15.47 ms here and 25.58 ms in a kernel trace
 * of the real run. Cycling defeats the reuse and reproduces the traced cost. */
#define GEMM_BENCH_COPIES 3u

struct Buffers {
    __half *A[GEMM_BENCH_COPIES] = {};
    __half *B[GEMM_BENCH_COPIES] = {};
    float *C = nullptr, *Ref = nullptr;
    uint32_t turn = 0;
    __half *a() { return A[turn]; }
    __half *b() { return B[turn]; }
    void next() { turn = (turn + 1u) % GEMM_BENCH_COPIES; }
};

static double time_ms(void (*fn)(void *), void *ctx, int iters) {
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    fn(ctx); /* warm */
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(a, 0));
    for (int i = 0; i < iters; i++) fn(ctx);
    HIP_CHECK(hipEventRecord(b, 0));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return (double)ms / (double)iters;
}

__global__ static void diff_kernel(const float *a, const float *b, uint64_t n,
                                   float *out_max, float *out_sq) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = fabsf(a[i] - b[i]);
    atomicMax((int *)out_max, __float_as_int(d));
    atomicAdd(out_sq, d * d);
}

static void report_error(const float *c, const float *ref, uint64_t n,
                         double *max_abs, double *rms) {
    float *dm = nullptr, *ds = nullptr;
    HIP_CHECK(hipMalloc(&dm, sizeof(float)));
    HIP_CHECK(hipMalloc(&ds, sizeof(float)));
    HIP_CHECK(hipMemset(dm, 0, sizeof(float)));
    HIP_CHECK(hipMemset(ds, 0, sizeof(float)));
    diff_kernel<<<(n + 255) / 256, 256>>>(c, ref, n, dm, ds);
    HIP_CHECK(hipDeviceSynchronize());
    float hm = 0.0f, hs = 0.0f;
    HIP_CHECK(hipMemcpy(&hm, dm, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hs, ds, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dm));
    HIP_CHECK(hipFree(ds));
    *max_abs = hm;
    *rms = sqrt((double)hs / (double)n);
}

/* ---------------------------------------------------------- hipBLASLt run */

struct LtPlan {
    hipblasLtHandle_t h;
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t a, b, c, d;
    std::vector<hipblasLtMatmulHeuristicResult_t> cands;
};

static void lt_build(LtPlan &p, const Shape &s, size_t max_ws) {
    hipblasOperation_t op_a = s.op_t ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t op_b = HIPBLAS_OP_N;
    if (hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
        HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "desc create failed\n");
        exit(1);
    }
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                    sizeof(op_a));
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                    sizeof(op_b));
    const uint32_t a_rows = s.op_t ? s.k : s.m;
    const uint32_t a_cols = s.op_t ? s.m : s.k;
    hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16F, a_rows, a_cols, a_rows);
    hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16F, s.k, s.n, s.k);
    hipblasLtMatrixLayoutCreate(&p.c, HIP_R_32F, s.m, s.n, s.m);
    hipblasLtMatrixLayoutCreate(&p.d, HIP_R_32F, s.m, s.n, s.m);
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
            sizeof(max_ws));
    hipblasLtMatmulHeuristicResult_t heur[32];
    int got = 0;
    hipblasLtMatmulAlgoGetHeuristic(p.h, p.desc, p.a, p.b, p.c, p.d, pref, 32,
                                    heur, &got);
    hipblasLtMatmulPreferenceDestroy(pref);
    p.cands.assign(heur, heur + got);
}

struct LtCtx {
    LtPlan *p;
    Buffers *buf;
    const hipblasLtMatmulAlgo_t *algo;
    void *ws;
    size_t ws_bytes;
    float *out;
};

static void lt_launch(void *ctx) {
    LtCtx *c = (LtCtx *)ctx;
    const float alpha = 1.0f, beta = 0.0f;
    hipblasLtMatmul(c->p->h, c->p->desc, &alpha, c->buf->a(), c->p->a,
                    c->buf->b(), c->p->b, &beta, c->out, c->p->c, c->out,
                    c->p->d, c->algo, c->ws, c->ws_bytes, 0);
    c->buf->next();
}

/* -------------------------------------------------------------- wmma run */

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WMF, uint32_t WNF,
          uint32_t PAD, uint32_t SWZ>
static void run_wmma(const Shape &s, Buffers &buf, int iters, double ref_gflop,
                     const char *name) {
    if (s.m % BM || s.n % BN || s.k % BK) return;
    constexpr uint32_t THREADS = (BM / 16u / WMF) * (BN / 16u / WNF) * 32u;
    if (THREADS > 1024u) return;
    const dim3 grid(s.m / BM, s.n / BN, 1);

    struct Ctx {
        const Shape *s;
        Buffers *buf;
        dim3 grid;
    } ctx{&s, &buf, grid};

    auto launch = [](void *p) {
        Ctx *c = (Ctx *)p;
        if (c->s->op_t) {
            gemm_f16_wmma_kernel<BM, BN, BK, WMF, WNF, true, PAD, SWZ>
                    <<<c->grid, THREADS>>>(c->buf->C, c->buf->a(), c->buf->b(),
                                           c->s->m, c->s->n, c->s->k);
        } else {
            gemm_f16_wmma_kernel<BM, BN, BK, WMF, WNF, false, PAD, SWZ>
                    <<<c->grid, THREADS>>>(c->buf->C, c->buf->a(), c->buf->b(),
                                           c->s->m, c->s->n, c->s->k);
        }
        c->buf->next();
    };
    launch(&ctx);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("  %-34s LAUNCH FAIL %s\n", name, hipGetErrorString(err));
        (void)hipGetLastError();
        return;
    }
    const double ms = time_ms(launch, &ctx, iters);
    double mx = 0.0, rms = 0.0;
    report_error(buf.C, buf.Ref, (uint64_t)s.m * s.n, &mx, &rms);
    printf("  %-34s %8.3f ms  %7.2f TFLOP/s  max %.4g rms %.4g\n", name, ms,
           ref_gflop / ms, mx, rms);
}

#define WMMA_CASE(BM, BN, BK, WMF, WNF, PAD, SWZ)                             \
    run_wmma<BM, BN, BK, WMF, WNF, PAD, SWZ>(                                 \
            s, buf, iters, gflop,                                             \
            "wmma " #BM "x" #BN "x" #BK " f" #WMF "x" #WNF " p" #PAD " s" #SWZ)

int main(int argc, char **argv) {
    int iters = 20;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) g_filter = argv[++i];
    }

    /* n = 4096 is the production prefill chunk at the 4K frontier. */
    const Shape shapes[] = {
            {"attn-out-b", 4096, 4096, 8192, false, 4},
            {"q8-f16", 4096, 4096, 2048, true, 4},
            {"proj-8192", 8192, 4096, 1024, true, 4},
            {"proj-1024", 1024, 4096, 4096, true, 4},
            {"proj-512", 512, 4096, 4096, true, 5},
            {"proj-256", 256, 4096, 4096, true, 6},
            {"proj-64", 64, 4096, 4096, true, 4},
    };

    hipblasLtHandle_t lt = nullptr;
    if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "hipblasLtCreate failed\n");
        return 1;
    }

    for (const Shape &s : shapes) {
        if (g_filter && !strstr(s.label, g_filter)) continue;
        const double gflop = 2.0 * s.m * s.n * s.k / 1e9;
        printf("\n== %s  m=%u n=%u k=%u opA=%c  %.1f GFLOP ==\n", s.label, s.m,
               s.n, s.k, s.op_t ? 'T' : 'N', gflop);

        Buffers buf;
        const uint64_t an = (uint64_t)s.m * s.k, bn = (uint64_t)s.n * s.k;
        for (uint32_t c = 0; c < GEMM_BENCH_COPIES; c++) {
            HIP_CHECK(hipMalloc(&buf.A[c], an * sizeof(__half)));
            HIP_CHECK(hipMalloc(&buf.B[c], bn * sizeof(__half)));
            /* Same seed in every copy, so rotating changes only which bytes are
             * cached and never the result. */
            fill_kernel<<<(an + 255) / 256, 256>>>(buf.A[c], an, 12345u);
            fill_kernel<<<(bn + 255) / 256, 256>>>(buf.B[c], bn, 67890u);
        }
        HIP_CHECK(hipMalloc(&buf.C, (size_t)s.m * s.n * sizeof(float)));
        HIP_CHECK(hipMalloc(&buf.Ref, (size_t)s.m * s.n * sizeof(float)));
        HIP_CHECK(hipDeviceSynchronize());

        /* Candidate sweep at zero workspace, which is what production asks
         * for, then again with a real workspace. */
        for (int pass = 0; pass < 2; pass++) {
            const size_t max_ws = pass == 0 ? 0u : (64u << 20);
            LtPlan p;
            p.h = lt;
            lt_build(p, s, max_ws);
            void *ws = nullptr;
            if (max_ws) HIP_CHECK(hipMalloc(&ws, max_ws));
            printf(" hipBLASLt workspace<=%zu MiB, %zu candidates\n",
                   max_ws >> 20, p.cands.size());
            for (size_t i = 0; i < p.cands.size(); i++) {
                if (p.cands[i].state != HIPBLAS_STATUS_SUCCESS) continue;
                if (p.cands[i].workspaceSize > max_ws) continue;
                LtCtx c{&p, &buf, &p.cands[i].algo, ws,
                        (size_t)p.cands[i].workspaceSize, buf.C};
                const float alpha = 1.0f, beta = 0.0f;
                if (hipblasLtMatmul(p.h, p.desc, &alpha, buf.A[0], p.a,
                                    buf.B[0], p.b, &beta, buf.C, p.c, buf.C,
                                    p.d, &p.cands[i].algo, ws,
                                    (size_t)p.cands[i].workspaceSize,
                                    0) != HIPBLAS_STATUS_SUCCESS) {
                    continue;
                }
                HIP_CHECK(hipDeviceSynchronize());
                if (pass == 0 && i == 0) {
                    HIP_CHECK(hipMemcpy(buf.Ref, buf.C,
                                        (size_t)s.m * s.n * sizeof(float),
                                        hipMemcpyDeviceToDevice));
                }
                const double ms = time_ms(lt_launch, &c, iters);
                double mx = 0.0, rms = 0.0;
                report_error(buf.C, buf.Ref, (uint64_t)s.m * s.n, &mx, &rms);
                printf("  cand %-2zu ws %6zu KiB %19s %8.3f ms  %7.2f TFLOP/s"
                       "  max %.4g rms %.4g%s\n",
                       i, (size_t)p.cands[i].workspaceSize >> 10, "", ms,
                       gflop / ms, mx, rms,
                       (pass == 0 && i == s.pinned) ? "   <== PRODUCTION" : "");
            }
            if (ws) HIP_CHECK(hipFree(ws));
            hipblasLtMatrixLayoutDestroy(p.d);
            hipblasLtMatrixLayoutDestroy(p.c);
            hipblasLtMatrixLayoutDestroy(p.b);
            hipblasLtMatrixLayoutDestroy(p.a);
            hipblasLtMatmulDescDestroy(p.desc);
        }

        printf(" rocWMMA candidates\n");
        WMMA_CASE(128, 128, 32, 2, 2, 0, 4);
        WMMA_CASE(256, 128, 32, 4, 2, 0, 1);
        WMMA_CASE(256, 128, 32, 4, 2, 0, 2);
        WMMA_CASE(256, 128, 32, 4, 2, 0, 4);
        WMMA_CASE(256, 128, 32, 4, 2, 0, 8);
        WMMA_CASE(256, 128, 32, 4, 2, 8, 4);
        WMMA_CASE(256, 128, 16, 4, 2, 0, 4);
        WMMA_CASE(256, 256, 32, 4, 4, 0, 2);
        WMMA_CASE(256, 256, 32, 4, 4, 0, 4);
        WMMA_CASE(256, 256, 16, 4, 4, 0, 4);
        WMMA_CASE(256, 256, 32, 8, 2, 0, 4);
        WMMA_CASE(512, 128, 32, 8, 2, 0, 4);
        WMMA_CASE(128, 256, 32, 2, 4, 0, 4);
        WMMA_CASE(64, 128, 32, 1, 2, 0, 4);
        WMMA_CASE(64, 64, 32, 1, 1, 0, 1);

        HIP_CHECK(hipFree(buf.Ref));
        HIP_CHECK(hipFree(buf.C));
        for (uint32_t c = 0; c < GEMM_BENCH_COPIES; c++) {
            HIP_CHECK(hipFree(buf.B[c]));
            HIP_CHECK(hipFree(buf.A[c]));
        }
    }
    hipblasLtDestroy(lt);
    return 0;
}
