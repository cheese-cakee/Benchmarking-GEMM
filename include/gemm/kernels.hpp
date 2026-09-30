#pragma once

#include <vector>

// Every kernel computes C = A * B for square, row-major N x N float matrices
// and overwrites C.
using GemmKernel = void (*)(const float* A, const float* B, float* C, int N);

void gemm_naive(const float* A, const float* B, float* C, int N);
void gemm_register(const float* A, const float* B, float* C, int N);
void gemm_ikj(const float* A, const float* B, float* C, int N);
void gemm_tiled(const float* A, const float* B, float* C, int N, int tile_size);
void gemm_tiled_64(const float* A, const float* B, float* C, int N);
void gemm_avx2(const float* A, const float* B, float* C, int N);
void gemm_blocked_4x8(const float* A, const float* B, float* C, int N);
void gemm_blocked_4x8_packed(const float* A, const float* B, float* C, int N);
void gemm_blocked_4x8_packed_omp(const float* A, const float* B, float* C, int N);
void gemm_blocked_4x8_packed_prefetch(const float* A, const float* B, float* C, int N);
void gemm_blocked_4x8_packed_prefetch_omp(const float* A, const float* B, float* C, int N);

struct KernelInfo {
    const char* name;
    GemmKernel fn;
    bool slow;  // O(N^3) with poor locality; skipped at large N unless requested
};

// Kernels in the order of the optimization guide.
const std::vector<KernelInfo>& all_kernels();
