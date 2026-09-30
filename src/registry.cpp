#include "gemm/kernels.hpp"

void gemm_tiled_64(const float* A, const float* B, float* C, int N)
{
    gemm_tiled(A, B, C, N, 64);
}

const std::vector<KernelInfo>& all_kernels()
{
    static const std::vector<KernelInfo> kernels = {
        {"naive", gemm_naive, true},
        {"register", gemm_register, true},
        {"ikj", gemm_ikj, false},
        {"tiled64", gemm_tiled_64, false},
        {"avx2_ikj", gemm_avx2, false},
        {"micro4x8", gemm_blocked_4x8, false},
        {"packed4x8", gemm_blocked_4x8_packed, false},
        {"packed4x8_prefetch", gemm_blocked_4x8_packed_prefetch, false},
        {"packed4x8_omp", gemm_blocked_4x8_packed_omp, false},
        {"packed4x8_prefetch_omp", gemm_blocked_4x8_packed_prefetch_omp, false},
    };
    return kernels;
}
