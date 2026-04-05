#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

void gemm_ikj(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float temp = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += temp * B[k * N + j];
            }
        }
    }
}

void init_matrix(std::vector<float>& mat, int N) {
    for (int i = 0; i < N * N; i++) mat[i] = static_cast<float>((i % 17) * 0.1f);
}

int main() {
    for (int N : {64, 128, 256, 512, 1024}) {
        std::vector<float> A(N*N), B(N*N), C(N*N);
        init_matrix(A, N); init_matrix(B, N);
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_ikj(A.data(), B.data(), C.data(), N);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = 2.0 * N * N * N / ms / 1e6;
        std::cout << "N=" << N << ": " << std::fixed << std::setprecision(1) << ms << " ms, " << gflops << " GFLOPS\n";
    }
    return 0;
}
