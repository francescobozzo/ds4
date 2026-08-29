/* Probe the rocWMMA F32 accumulator lane layout on gfx1151.
 *
 * A single-pass (FlashAttention-style) rewrite of
 * `attention_mixed_heads32_wmma_kernel` has to rescale the output accumulator by
 * `exp(m_old - m_new)`, and that factor is per attention head, i.e. per *row* of
 * a 16x16 `rocwmma::fragment<accumulator, 16, 16, 16, float>`. rocWMMA exposes
 * no row accessor, and the two portable alternatives both fail here: a
 * `store_matrix_sync` round trip needs 32 waves x 16x16 x 4 B = 64 KiB of LDS
 * against about 6 KiB free, and the kernel's 58,368 B already hold it to one
 * workgroup per CU.
 *
 * So the question is whether each lane's elements sit at a fixed, computable
 * row. Rather than trust a guess about the RDNA3 `v_wmma_f32_16x16x16_f16` D
 * layout, recover it: store a fragment whose value at (row, col) is
 * `row * 100 + col`, then have every lane report the (row, col) of each of its
 * elements. If the row is a closed form in (lane, element index), the rescale is
 * a few lines of register arithmetic and the rewrite is unblocked.
 *
 * Build:
 *   hipcc -O3 -std=c++17 --offload-arch=gfx1151 -I<rocwmma>/include \
 *     -o /tmp/wmma_acc_layout rocm/tools/wmma_acc_layout.cpp
 */

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdio>
#include <cstdlib>

__global__ void probe_kernel(int *out_row, int *out_col, int *out_n) {
    using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    frag_c acc;

    /* Build the marker matrix in LDS, load it into the accumulator, and then
     * read the fragment's own registers: whatever value a lane holds names the
     * (row, col) it owns. */
    __shared__ float marker[256];
    const unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < 256u; i += blockDim.x) {
        const unsigned r = i / 16u;
        const unsigned c = i % 16u;
        marker[i] = (float)(r * 100u + c);
    }
    __syncthreads();

    rocwmma::load_matrix_sync(acc, marker, 16, rocwmma::mem_row_major);

    const unsigned lane = tid & 31u;
    const int n = (int)acc.num_elements;
    if (lane == 0u) *out_n = n;
    for (int e = 0; e < n; e++) {
        const int v = (int)acc.x[e];
        out_row[lane * 16 + e] = v / 100;
        out_col[lane * 16 + e] = v % 100;
    }
}

int main() {
    int *drow = nullptr, *dcol = nullptr, *dn = nullptr;
    if (hipMalloc(&drow, 32 * 16 * sizeof(int)) != hipSuccess ||
        hipMalloc(&dcol, 32 * 16 * sizeof(int)) != hipSuccess ||
        hipMalloc(&dn, sizeof(int)) != hipSuccess) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    hipMemset(drow, -1, 32 * 16 * sizeof(int));
    hipMemset(dcol, -1, 32 * 16 * sizeof(int));
    probe_kernel<<<1, 32>>>(drow, dcol, dn);
    if (hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, "launch failed: %s\n",
                hipGetErrorString(hipGetLastError()));
        return 1;
    }
    int hrow[32 * 16], hcol[32 * 16], hn = 0;
    hipMemcpy(hrow, drow, sizeof(hrow), hipMemcpyDeviceToHost);
    hipMemcpy(hcol, dcol, sizeof(hcol), hipMemcpyDeviceToHost);
    hipMemcpy(&hn, dn, sizeof(int), hipMemcpyDeviceToHost);

    printf("wave32 accumulator num_elements = %d\n\n", hn);
    printf("lane |            (row, col) per element\n");
    for (int l = 0; l < 32; l++) {
        printf("%4d |", l);
        for (int e = 0; e < hn; e++) {
            printf(" (%2d,%2d)", hrow[l * 16 + e], hcol[l * 16 + e]);
        }
        printf("\n");
    }

    /* Test the two closed forms worth having: row depending only on the element
     * index, or only on the lane. Either one makes a per-row rescale trivial. */
    bool row_from_elem = true, row_from_lane = true, col_from_lane = true;
    for (int l = 0; l < 32; l++) {
        for (int e = 0; e < hn; e++) {
            if (hrow[l * 16 + e] != hrow[e]) row_from_elem = false;
            if (hrow[l * 16 + e] != hrow[l * 16]) row_from_lane = false;
            if (hcol[l * 16 + e] != hcol[l * 16]) col_from_lane = false;
        }
    }
    printf("\nrow is a function of the element index alone: %s\n",
           row_from_elem ? "YES" : "no");
    printf("row is a function of the lane alone:         %s\n",
           row_from_lane ? "YES" : "no");
    printf("col is a function of the lane alone:         %s\n",
           col_from_lane ? "YES" : "no");
    if (row_from_elem) {
        printf("\nrow(e) =");
        for (int e = 0; e < hn; e++) printf(" %d", hrow[e]);
        printf("\n  -> a per-row rescale is register-only: scale element e by\n"
               "     the factor for row(e). Single-pass rewrite is unblocked.\n");
    }
    if (row_from_lane) {
        printf("\nrow(lane) =");
        for (int l = 0; l < 32; l++) printf(" %d", hrow[l * 16]);
        printf("\n  -> broadcast the factor per lane; also unblocked.\n");
    }
    return 0;
}
