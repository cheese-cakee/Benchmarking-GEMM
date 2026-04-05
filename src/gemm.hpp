#ifndef GEMM_HPP
#define GEMM_HPP

void gemm_naive(const float* A, const float* B, float* C, int N);
void gemm_register(const float* A, const float* B, float* C, int N);
void gemm_ikj(const float* A, const float* B, float* C, int N);

#endif