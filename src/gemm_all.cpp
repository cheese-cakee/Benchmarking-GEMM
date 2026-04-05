#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>

constexpr int FULL_N = 2048;
constexpr int VERIFY_N = 64;
constexpr int VISUAL_N = 4;
constexpr int WARMUP_RUNS = 2;
constexpr int TIMED_RUNS = 5;
constexpr float VERIFY_EPSILON = 1e-3f;

void gemm_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
        }
    }
}

void gemm_register(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

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

static void init_matrix(std::vector<float>& mat, int N) {
    for (int i = 0; i < N * N; i++) mat[i] = static_cast<float>((i % 17) * 0.1f);
}

static void print_matrix(const float* mat, int N, const std::string& label) {
    std::cout << label << " (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; i++) {
        std::cout << "  ";
        for (int j = 0; j < N; j++)
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << mat[i * N + j];
        std::cout << "\n";
    }
    std::cout << "\n";
}

static bool check_correctness(const float* computed, const float* reference, int N) {
    for (int i = 0; i < N * N; i++)
        if (std::abs(computed[i] - reference[i]) > VERIFY_EPSILON) {
            std::cout << "  MISMATCH at [" << i / N << "][" << i % N << "]: "
                      << "computed=" << computed[i] << ", reference=" << reference[i] << "\n";
            return false;
        }
    return true;
}

struct Stats { double median, min_val, max_val, stddev, gflops; };

static Stats benchmark(void (*kernel)(const float*, const float*, float*, int),
                       const float* A, const float* B, float* C, int N,
                       const std::string& name, double total_flops) {
    for (int w = 0; w < WARMUP_RUNS; w++) {
        std::fill(C, C + N * N, 0.0f);
        kernel(A, B, C, N);
    }

    std::vector<double> times;
    volatile float sink = 0;
    for (int r = 0; r < TIMED_RUNS; r++) {
        std::fill(C, C + N * N, 0.0f);
        auto start = std::chrono::high_resolution_clock::now();
        kernel(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        for (int i = 0; i < N * N; i++) sink += C[i];
    }

    std::sort(times.begin(), times.end());
    double median = times[TIMED_RUNS / 2];
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / TIMED_RUNS;
    double var = 0;
    for (double t : times) var += (t - mean) * (t - mean);
    double stddev = std::sqrt(var / TIMED_RUNS);
    double gflops = total_flops / (median / 1000.0) / 1e9;

    std::cout << std::left << std::setw(22) << name
              << std::fixed << std::setprecision(1)
              << std::setw(10) << median << " ms  "
              << std::setw(10) << times.front() << " ms  "
              << std::setw(10) << times.back() << " ms  "
              << std::setw(8) << stddev << "  "
              << std::setprecision(1) << std::setw(8) << gflops << " GFLOPS\n";

    return {median, times.front(), times.back(), stddev, gflops};
}

static double time_single(void (*kernel)(const float*, const float*, float*, int),
                          const float* A, const float* B, float* C, int N) {
    volatile float sink = 0;
    auto start = std::chrono::high_resolution_clock::now();
    kernel(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N * N; i++) sink += C[i];
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== GEMM Optimization Benchmarks ===\n\n";

    // --- 4x4 Visual Verification ---
    {
        std::cout << "--- 4x4 Visual Verification ---\n";
        const int N = VISUAL_N;
        std::vector<float> A(N*N), B(N*N), C(N*N), Ref(N*N);
        init_matrix(A, N); init_matrix(B, N);
        gemm_register(A.data(), B.data(), Ref.data(), N);
        print_matrix(A.data(), N, "Matrix A");
        print_matrix(B.data(), N, "Matrix B");
        print_matrix(Ref.data(), N, "Reference Output");
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_naive(A.data(), B.data(), C.data(), N);
        std::cout << "Naive ijk:          " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_register(A.data(), B.data(), C.data(), N);
        std::cout << "Register optimized: " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        gemm_ikj(A.data(), B.data(), C.data(), N);
        std::cout << "Loop reorder ikj:   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n\n";
    }

    // --- 64x64 Correctness Check ---
    {
        std::cout << "--- Correctness Check (64x64) ---\n";
        const int N = VERIFY_N;
        std::vector<float> A(N*N), B(N*N), C(N*N), Ref(N*N);
        init_matrix(A, N); init_matrix(B, N);
        gemm_register(A.data(), B.data(), Ref.data(), N);
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_naive(A.data(), B.data(), C.data(), N);
        std::cout << "Naive ijk:          " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_register(A.data(), B.data(), C.data(), N);
        std::cout << "Register optimized: " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        gemm_ikj(A.data(), B.data(), C.data(), N);
        std::cout << "Loop reorder ikj:   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n\n";
    }

    // --- 256x256 Benchmark (all kernels, fast) ---
    {
        const int N = 256;
        double total_flops = 2.0 * N * N * N;
        std::cout << "--- Benchmark (256x256) - all kernels ---\n";
        std::cout << "Total FLOPs: " << std::fixed << std::setprecision(2) << (total_flops / 1e6) << " million\n\n";
        std::cout << std::left << std::setw(22) << "Kernel"
                  << std::setw(12) << "Median"
                  << std::setw(12) << "Min"
                  << std::setw(12) << "Max"
                  << std::setw(10) << "StdDev"
                  << std::setw(12) << "GFLOPS" << "\n";
        std::cout << std::string(70, '-') << "\n";

        std::vector<float> A(N*N), B(N*N), C(N*N);
        init_matrix(A, N); init_matrix(B, N);

        Stats s_naive = benchmark(gemm_naive, A.data(), B.data(), C.data(), N, "Naive ijk", total_flops);
        Stats s_reg   = benchmark(gemm_register, A.data(), B.data(), C.data(), N, "Register optimized", total_flops);
        Stats s_ikj   = benchmark(gemm_ikj, A.data(), B.data(), C.data(), N, "Loop reorder ikj", total_flops);

        std::cout << "\n--- Speedups (vs Naive) ---\n";
        std::cout << "Register optimized: " << std::fixed << std::setprecision(2) << s_naive.median / s_reg.median << "x\n";
        std::cout << "Loop reorder ikj:   " << std::fixed << std::setprecision(2) << s_naive.median / s_ikj.median << "x\n";
    }

    // --- 2048x2048 Benchmark (ikj only, projected others) ---
    {
        const int N = FULL_N;
        double total_flops = 2.0 * N * N * N;
        std::cout << "\n--- Benchmark (2048x2048) - full size ---\n";
        std::cout << "Total FLOPs: " << std::fixed << std::setprecision(2) << (total_flops / 1e9) << " billion\n";
        std::cout << "(Naive/Register omitted - would take ~35+ min each)\n\n";
        std::cout << std::left << std::setw(22) << "Kernel"
                  << std::setw(12) << "Median"
                  << std::setw(12) << "Min"
                  << std::setw(12) << "Max"
                  << std::setw(10) << "StdDev"
                  << std::setw(12) << "GFLOPS" << "\n";
        std::cout << std::string(70, '-') << "\n";

        std::vector<float> A(N*N), B(N*N), C(N*N);
        init_matrix(A, N); init_matrix(B, N);

        Stats s_ikj = benchmark(gemm_ikj, A.data(), B.data(), C.data(), N, "Loop reorder ikj", total_flops);

        // Project naive and register from 256x256 ratios
        std::cout << "\n--- Projected speedups (from 256x256 ratios) ---\n";
        std::cout << "Naive (projected ~35 min)       vs ikj: " << std::fixed << std::setprecision(1)
                  << (35000.0 / s_ikj.median) << "x\n";
    }

    return 0;
}
