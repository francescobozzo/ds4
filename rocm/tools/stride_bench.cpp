// Standalone gather-bandwidth probe for the memory-channel-camping hypothesis.
//
// Claim under test: traffic is interleaved across the 16 LPDDR5X channels in
// 256 B blocks, so a row stride that is an even multiple of 256 B reaches only
// 16/gcd(stride/256,16) channels and caps achieved bandwidth.
//
// Each workgroup gathers ROWS_PER_WG rows through an index array and reads the
// first `payload` bytes of each. Bytes read are held constant across strides;
// only the row pitch changes, so any bandwidth step is attributable to the
// pitch alone.
//
// Build: hipcc --offload-arch=gfx1151 -O3 -o stride_bench stride_bench.cpp

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                              \
    do {                                                                             \
        hipError_t err_ = (expr);                                                    \
        if (err_ != hipSuccess) {                                                     \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err_),       \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                            \
    } while (0)

static constexpr uint32_t THREADS     = 256u;
static constexpr uint32_t ROWS_PER_WG = 64u;

__global__ __launch_bounds__(THREADS) void gather_kernel(const float *__restrict__ base,
                                                         const uint32_t *__restrict__ idx,
                                                         uint32_t stride_floats,
                                                         uint32_t payload_floats,
                                                         float *__restrict__ sink) {
    const uint32_t first = blockIdx.x * ROWS_PER_WG;
    const uint32_t total = ROWS_PER_WG * payload_floats;
    float acc            = 0.0f;
    // Flattened (row, element) walk: every thread stays busy no matter how small
    // the payload, and a wave keeps reading contiguously within one row.
    for (uint32_t f = threadIdx.x; f < total; f += THREADS) {
        const uint32_t r = f / payload_floats;
        const uint32_t d = f - r * payload_floats;
        acc += base[(uint64_t)idx[first + r] * stride_floats + d];
    }
    // Never taken: keeps every load live without adding a store to the timed path.
    if (acc == 1.2345678e30f) {
        sink[blockIdx.x] = acc;
    }
}

struct Result {
    uint32_t stride_bytes;
    double   gbps;
    double   ms;
};

static Result run_case(uint32_t stride_bytes, uint32_t payload_bytes, uint32_t rows,
                       bool shuffled, const uint32_t *d_idx_seq, const uint32_t *d_idx_shuf,
                       float *sink) {
    const uint32_t stride_floats  = stride_bytes / 4u;
    const uint32_t payload_floats = payload_bytes / 4u;
    const uint64_t bytes_alloc    = (uint64_t)rows * stride_bytes;

    float *d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, bytes_alloc));
    HIP_CHECK(hipMemset(d_buf, 0x3c, bytes_alloc));

    const uint32_t blocks = rows / ROWS_PER_WG;
    const uint32_t *d_idx = shuffled ? d_idx_shuf : d_idx_seq;

    // Warmup.
    for (int i = 0; i < 2; ++i) {
        hipLaunchKernelGGL(gather_kernel, dim3(blocks), dim3(THREADS), 0, 0, d_buf, d_idx,
                           stride_floats, payload_floats, sink);
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t beg, end;
    HIP_CHECK(hipEventCreate(&beg));
    HIP_CHECK(hipEventCreate(&end));

    const int iters = 10;
    HIP_CHECK(hipEventRecord(beg, 0));
    for (int i = 0; i < iters; ++i) {
        hipLaunchKernelGGL(gather_kernel, dim3(blocks), dim3(THREADS), 0, 0, d_buf, d_idx,
                           stride_floats, payload_floats, sink);
    }
    HIP_CHECK(hipEventRecord(end, 0));
    HIP_CHECK(hipEventSynchronize(end));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
    ms /= (float)iters;

    HIP_CHECK(hipEventDestroy(beg));
    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipFree(d_buf));

    const double bytes_read = (double)rows * (double)payload_bytes;
    return Result{stride_bytes, bytes_read / (ms * 1.0e-3) / 1.0e9, ms};
}

int main(int argc, char **argv) {
    // Rows are fixed so that bytes-read is constant across the sweep; only the
    // allocation grows with stride.
    uint32_t rows          = 262144u;  // 4096 workgroups
    uint32_t payload_bytes = 1024u;    // compressed-KV row: 512 halves
    if (argc > 1) payload_bytes = (uint32_t)atoi(argv[1]);
    if (argc > 2) rows = (uint32_t)atoi(argv[2]);

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("device            : %s (%s)\n", prop.name, prop.gcnArchName);
    printf("rows              : %u  (%u workgroups x %u rows)\n", rows, rows / ROWS_PER_WG,
           ROWS_PER_WG);
    printf("payload per row   : %u B\n", payload_bytes);
    printf("bytes read / pass : %.1f MiB\n", (double)rows * payload_bytes / (1024.0 * 1024.0));

    std::vector<uint32_t> seq(rows);
    std::iota(seq.begin(), seq.end(), 0u);
    std::vector<uint32_t> shuf = seq;
    std::mt19937 rng(12345u);
    std::shuffle(shuf.begin(), shuf.end(), rng);

    uint32_t *d_seq = nullptr, *d_shuf = nullptr;
    float    *sink  = nullptr;
    HIP_CHECK(hipMalloc(&d_seq, rows * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_shuf, rows * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&sink, (rows / ROWS_PER_WG) * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_seq, seq.data(), rows * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_shuf, shuf.data(), rows * sizeof(uint32_t), hipMemcpyHostToDevice));

    // Strides that are even multiples of 256 B (predicted to camp) interleaved
    // with nearby strides that are not.
    const uint32_t strides[] = {1024u, 1088u, 1152u, 1280u, 1536u, 1792u,
                                2048u, 2176u, 2304u, 4096u, 4224u};

    for (int pass = 0; pass < 2; ++pass) {
        const bool shuffled = (pass == 1);
        printf("\n%s row order\n", shuffled ? "shuffled" : "sequential");
        printf("  stride    /256   channels   GB/s     ms\n");
        for (uint32_t s : strides) {
            if (s < payload_bytes) continue;
            Result r = run_case(s, payload_bytes, rows, shuffled, d_seq, d_shuf, sink);
            // Predicted distinct channels touched by row starts.
            char ch[16];
            if (s % 256u == 0u) {
                uint32_t q = s / 256u, g = q, b = 16u;
                while (b) { uint32_t t = g % b; g = b; b = t; }
                snprintf(ch, sizeof(ch), "%u", 16u / g);
            } else {
                snprintf(ch, sizeof(ch), "16*");
            }
            printf("  %5u B  %5.2f   %8s   %6.1f  %6.3f\n", r.stride_bytes,
                   (double)s / 256.0, ch, r.gbps, r.ms);
        }
    }

    HIP_CHECK(hipFree(d_seq));
    HIP_CHECK(hipFree(d_shuf));
    HIP_CHECK(hipFree(sink));
    printf("\n* stride is not a multiple of 256 B, so row starts walk every block offset\n");
    return 0;
}
