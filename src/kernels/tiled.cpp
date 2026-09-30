#include "gemm/kernels.hpp"

#include <algorithm>

void gemm_tiled(const float* A, const float* B, float* C, int N, int tile_size)
{
    for (int i = 0; i < N * N; i++) C[i] = 0;
    for (int i = 0; i < N; i += tile_size) {
        int i_end = std::min(i+tile_size,N);
        for (int k = 0; k < N; k += tile_size) {
            int k_end = std::min(k + tile_size, N);
            for (int j = 0; j < N; j += tile_size) {
                int j_end = std::min(j+tile_size,N);
                for (int ii = i; ii < i_end ; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        float temp = A[ii * N + kk];
                        for (int jj = j; jj < j_end ; jj++) {
                            C[ii * N + jj] += temp * B[kk * N + jj];
                        }
                    }
                }
            }
        }
}
}
