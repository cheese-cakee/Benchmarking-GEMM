#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>
#include <omp.h>
#include <immintrin.h>

constexpr int FULL_N = 2048;
constexpr int VERIFY_N = 64;
constexpr int VISUAL_N = 4;
constexpr int WARMUP_RUNS = 2;
constexpr int TIMED_RUNS = 5;
constexpr float VERIFY_EPSILON = 1e-3f;

void pack_A(const float* A, float* packed, int N, int i_start, int k_start, int mc, int kc)
{
    for(int kk = 0;kk < kc; kk++)
    {
        for(int ii = 0; ii < mc;ii++)
        {
            packed[kk * mc + ii] = A[(i_start + ii) * N + (k_start + kk)];
        }
    }
}

void pack_B(const float* B, float* packed, int N, int k_start, int j_start, int kc, int nr)
{
    for(int kk = 0;kk < kc;kk++)
    {
        for(int jj = 0; jj < nr; jj++)
        {
            packed[kk * nr + jj] = B[(k_start + kk)* N + (j_start + jj)];
        }
    }
}

void gemm_packed_4x8(const float* A_packed, const float* B_packed, float* C, int N, int i, int j, int kc, int mc)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for(int kk = 0; kk < kc; kk++){
        __m256 b = _mm256_loadu_ps(&B_packed[kk * 8]);

        __m256 a0 = _mm256_broadcast_ss(&A_packed[kk * mc + 0]);
        acc0 = _mm256_fmadd_ps(a0,b,acc0);
        __m256 a1 = _mm256_broadcast_ss(&A_packed[kk * mc + 1]);
        acc1 = _mm256_fmadd_ps(a1,b,acc1);
        __m256 a2 = _mm256_broadcast_ss(&A_packed[kk * mc + 2]);
        acc2 = _mm256_fmadd_ps(a2,b,acc2);
        __m256 a3 = _mm256_broadcast_ss(&A_packed[kk * mc + 3]);
        acc3 = _mm256_fmadd_ps(a3,b,acc3);
    }

    _mm256_storeu_ps(&C[(i+0)*N + j], acc0);
    _mm256_storeu_ps(&C[(i+1)*N + j], acc1);
    _mm256_storeu_ps(&C[(i+2)*N + j], acc2);
    _mm256_storeu_ps(&C[(i+3)*N + j], acc3);

}
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


void gemm_avx2(const float* A, const float* B, float* C, int N)
{
    for(int i =0;i<N * N;i++)C[i] = 0;
    for(int i =0;i<N;i++){
        for(int k = 0;k<N;k++){
            __m256 a_vec = _mm256_broadcast_ss(&A[i * N + k]);
            int j =0;
            for(;j+8 <= N; j += 8)
            {
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], c_vec);

            }
            for(;j<N;j++){
                C[i * N + j] += A[i * N + k]*B[k * N + j];
            }
        }
    }
}

void gemm_blocked_4x8(const float* A, const float* B, float* C,int N)
{
    for(int i =0;i<N*N;i++)C[i] = 0;

    int i =0;
    for(; i + 4 <= N;i += 4)
    {
        int j =0;
        for(;j +8 <= N; j+= 8)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for(int k =0; k < N; k++)
            {
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);


                __m256 a0 = _mm256_broadcast_ss(&A[(i+0)*N + k ]);
                acc0 = _mm256_fmadd_ps(a0,b,acc0);
                __m256 a1 = _mm256_broadcast_ss(&A[(i+1)*N + k ]);
                acc1 = _mm256_fmadd_ps(a1,b,acc1);
                __m256 a2 = _mm256_broadcast_ss(&A[(i+2)*N + k ]);
                acc2 = _mm256_fmadd_ps(a2,b,acc2);
                __m256 a3 = _mm256_broadcast_ss(&A[(i+3)*N + k ]);
                acc3 = _mm256_fmadd_ps(a3,b,acc3);
            }

            _mm256_storeu_ps(&C[(i+0)*N + j], acc0);
            _mm256_storeu_ps(&C[(i+1)*N + j], acc1);
            _mm256_storeu_ps(&C[(i+2)*N + j], acc2);
            _mm256_storeu_ps(&C[(i+3)*N + j], acc3);
        }

        if(j<N){
            for(int k = 0;k<N;k++){
                float a0 = A[(i+0)*N + k];
                float a1 = A[(i+1)*N + k];
                float a2 = A[(i+2)*N + k];
                float a3 = A[(i+3)*N + k];
                for(int jj = j;jj < N;jj++){
                    float bval = B[k*N + jj];
                    C[(i+0)*N+jj] += a0 * bval;
                    C[(i+1)*N+jj] += a1 * bval;
                    C[(i+2)*N+jj] += a2 * bval;
                    C[(i+3)*N+jj] += a3 * bval;
                }
            }
        }
    }

    for(; i< N;i++)
    {
        for(int k =0;k<N;k++)
        {
            float a = A[i*N + k];
            for( int j =0;j<N;j++){
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void gemm_blocked_4x8_packed(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N * N; i++) C[i] = 0;

    const int TILE = 64;
    const int MR = 4;
    const int NR = 8;

    for (int i0 = 0; i0 < N; i0 += TILE) {
        int mc = std::min(TILE, N - i0);
        int mc_main = mc - (mc % MR);

        for (int k0 = 0; k0 < N; k0 += TILE) {
            int kc = std::min(TILE, N - k0);
            std::vector<float> packed_A(mc_main * kc);
            pack_A(A, packed_A.data(), N, i0, k0, mc_main, kc);

            for (int j0 = 0; j0 < N; j0 += TILE) {
                int nc = std::min(TILE, N - j0);

                for (int jj = 0; jj < nc; jj += NR) {
                    int nr = std::min(NR, nc - jj);

                    std::vector<float> packed_B(kc * NR);
                    pack_B(B, packed_B.data(), N, k0, j0 + jj, kc, NR);

                    for (int ii = 0; ii < mc_main; ii += MR) {
                        gemm_packed_4x8(
                            packed_A.data() + ii,
                            packed_B.data(),
                            C, N,
                            i0 + ii, j0 + jj,
                            kc, mc_main
                        );
                    }

                    for (int r = mc_main; r < mc; r++) {
                        for (int kk = 0; kk < kc; kk++) {
                            float a_val = A[(i0 + r) * N + (k0 + kk)];
                            for (int c = 0; c < nr; c++) {
                                C[(i0 + r) * N + (j0 + jj + c)] += a_val * B[(k0 + kk) * N + (j0 + jj + c)];
                            }
                        }
                    }
                }

            }
        }

    }
}

void gemm_packed_4x8_prefetch(const float* A_packed, const float* B_packed, float* C,
                             int N, int i, int j, int kc, int mc)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    _mm_prefetch((const char*)&B_packed[0 * 8], _MM_HINT_NTA);
    _mm_prefetch((const char*)&B_packed[1 * 8], _MM_HINT_NTA);
    _mm_prefetch((const char*)&A_packed[0 * mc], _MM_HINT_NTA);
    _mm_prefetch((const char*)&A_packed[1 * mc], _MM_HINT_NTA);

    const int PREFETCH_DIST = 2;
    for (int kk = 0; kk < kc; kk++) {

        if (kk + PREFETCH_DIST < kc) {
            _mm_prefetch((const char*)&B_packed[(kk + PREFETCH_DIST) * 8], _MM_HINT_NTA);
            _mm_prefetch((const char*)&A_packed[(kk + PREFETCH_DIST) * mc], _MM_HINT_NTA);
        }

        __m256 b = _mm256_loadu_ps(&B_packed[kk * 8]);

        __m256 a0 = _mm256_broadcast_ss(&A_packed[kk * mc + 0]);
        acc0 = _mm256_fmadd_ps(a0, b, acc0);
        __m256 a1 = _mm256_broadcast_ss(&A_packed[kk * mc + 1]);
        acc1 = _mm256_fmadd_ps(a1, b, acc1);
        __m256 a2 = _mm256_broadcast_ss(&A_packed[kk * mc + 2]);
        acc2 = _mm256_fmadd_ps(a2, b, acc2);
        __m256 a3 = _mm256_broadcast_ss(&A_packed[kk * mc + 3]);
        acc3 = _mm256_fmadd_ps(a3, b, acc3);
    }

    _mm256_storeu_ps(&C[(i + 0) * N + j], acc0);
    _mm256_storeu_ps(&C[(i + 1) * N + j], acc1);
    _mm256_storeu_ps(&C[(i + 2) * N + j], acc2);
    _mm256_storeu_ps(&C[(i + 3) * N + j], acc3);
}

void gemm_blocked_4x8_packed_prefetch(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N * N; i++) C[i] = 0;

    const int TILE = 64;
    const int MR = 4;
    const int NR = 8;

    for (int i0 = 0; i0 < N; i0 += TILE) {
        int mc = std::min(TILE, N - i0);
        int mc_main = mc - (mc % MR);

        for (int k0 = 0; k0 < N; k0 += TILE) {
            int kc = std::min(TILE, N - k0);
            std::vector<float> packed_A(mc_main * kc);
            pack_A(A, packed_A.data(), N, i0, k0, mc_main, kc);

            for (int j0 = 0; j0 < N; j0 += TILE) {
                int nc = std::min(TILE, N - j0);

                for (int jj = 0; jj < nc; jj += NR) {
                    int nr = std::min(NR, nc - jj);

                    std::vector<float> packed_B(kc * NR);
                    pack_B(B, packed_B.data(), N, k0, j0 + jj, kc, NR);

                    for (int ii = 0; ii < mc_main; ii += MR) {
                        gemm_packed_4x8_prefetch(
                            packed_A.data() + ii,
                            packed_B.data(),
                            C, N,
                            i0 + ii, j0 + jj,
                            kc, mc_main
                        );
                    }

                    for (int r = mc_main; r < mc; r++) {
                        for (int kk = 0; kk < kc; kk++) {
                            float a_val = A[(i0 + r) * N + (k0 + kk)];
                            for (int c = 0; c < nr; c++) {
                                C[(i0 + r) * N + (j0 + jj + c)] += a_val * B[(k0 + kk) * N + (j0 + jj + c)];
                            }
                        }
                    }
                }

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
              << std::setprecision(1) << std::setw(8) << gflops << " GFLOPS";
    std::cout << "\n";

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
    std::cout.setf(std::ios::unitbuf);
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
        std::cout << "Loop reorder ikj:   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_tiled(A.data(), B.data(), C.data(), N, 64);
        std::cout << "Tiled (64x64):   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_avx2(A.data(), B.data(), C.data(), N);
        std::cout << "AVX2 (64x64):   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_blocked_4x8(A.data(), B.data(), C.data(), N);
        std::cout << "4X8 Microkernel(64x64):   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_blocked_4x8_packed(A.data(), B.data(), C.data(), N);
        std::cout << "4X8 Packed (64x64):   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_blocked_4x8_packed_prefetch(A.data(), B.data(), C.data(), N);
        std::cout << "4X8 Pack+Prefetch:    " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n\n";
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
        std::cout << "Loop reorder ikj:   " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_tiled(A.data(), B.data(), C.data(), N, 64);
        std::cout << "Tiled (64x64):      " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_avx2(A.data(), B.data(), C.data(), N);
        std::cout << "AVX2 (64x64):      " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        gemm_blocked_4x8(A.data(), B.data(), C.data(), N);
        std::cout << "4X8 Micorkernel(64x64):      " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        gemm_blocked_4x8_packed(A.data(), B.data(), C.data(), N);
        std::cout << "4X8 Packed (64x64):      " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n";
        gemm_blocked_4x8_packed_prefetch(A.data(), B.data(), C.data(), N);
        std::cout << "4X8 Pack+Prefetch:       " << (check_correctness(C.data(), Ref.data(), N) ? "PASS" : "FAIL") << "\n\n";
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

        auto tiled_64 = [](const float* a, const float* b, float* c, int n){ gemm_tiled(a, b, c, n, 64); };
        Stats s_naive = benchmark(gemm_naive, A.data(), B.data(), C.data(), N, "Naive ijk", total_flops);
        Stats s_reg   = benchmark(gemm_register, A.data(), B.data(), C.data(), N, "Register optimized", total_flops);
        Stats s_ikj   = benchmark(gemm_ikj, A.data(), B.data(), C.data(), N, "Loop reorder ikj", total_flops);
        Stats s_tiled = benchmark(tiled_64, A.data(), B.data(), C.data(), N, "Tiled 64x64", total_flops);
        Stats s_avx2 = benchmark(gemm_avx2,A.data(),B.data(), C.data(), N, "AVX2 ikj", total_flops);
        Stats s_blocked = benchmark(gemm_blocked_4x8, A.data(), B.data(), C.data(), N, "4X8 Microkernel", total_flops);
        Stats s_packed = benchmark(gemm_blocked_4x8_packed, A.data(),B.data(), C.data(), N, "4X8 Packed", total_flops);
        Stats s_prefetch = benchmark(gemm_blocked_4x8_packed_prefetch, A.data(),B.data(), C.data(), N, "4X8+Prefetch", total_flops);

        std::cout << "\n--- Speedups (vs Naive) ---\n";
        std::cout << "Register optimized: " << std::fixed << std::setprecision(2) << s_naive.median / s_reg.median << "x\n";
        std::cout << "Loop reorder ikj:   " << std::fixed << std::setprecision(2) << s_naive.median / s_ikj.median << "x\n";
        std::cout << "Tiled 64x64:        " << std::fixed << std::setprecision(2) << s_naive.median / s_tiled.median << "x\n";
        std::cout << "AVX2:               " << std::fixed << std::setprecision(2) << s_naive.median / s_avx2.median << "x\n";
        std::cout << "4X8 Microkernel:    " << std::fixed << std::setprecision(2) << s_naive.median / s_blocked.median << "x\n";
        std::cout << "4X8 Packed:         " << std::fixed << std::setprecision(2) << s_naive.median / s_packed.median << "x\n";
        std::cout << "4X8+Prefetch:       " << std::fixed << std::setprecision(2) << s_naive.median / s_prefetch.median << "x\n";
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

        auto tiled_64 = [](const float* a, const float* b, float* c, int n){ gemm_tiled(a, b, c, n, 64); };
        Stats s_ikj = benchmark(gemm_ikj, A.data(), B.data(), C.data(), N, "Loop reorder ikj", total_flops);
        double tiled_time = time_single(tiled_64, A.data(), B.data(), C.data(), N);
        Stats s_avx2 = benchmark(gemm_avx2, A.data(), B.data(), C.data(), N, "AVX2 ikj", total_flops);
        Stats s_blocked = benchmark(gemm_blocked_4x8, A.data(), B.data(), C.data(), N, "4X8 Microkernel", total_flops);
        Stats s_packed = benchmark(gemm_blocked_4x8_packed,A.data(), B.data() ,C.data(),N, "Packed 4x8", total_flops);
        Stats s_prefetch = benchmark(gemm_blocked_4x8_packed_prefetch,A.data(), B.data() ,C.data(),N, "4X8+Prefetch", total_flops);

        // Project naive and register from 256x256 ratios
        std::cout << "\n--- Projected speedups (from 256x256 ratios) ---\n";
        std::cout << "Naive (projected ~35 min)       vs ikj: " << std::fixed << std::setprecision(1)
                  << (35000.0 / s_ikj.median) << "x\n";
        double tiled_gflops = total_flops / (tiled_time / 1000.0) / 1e9;
        std::cout << "Tiled 64x64 actual: " << std::fixed << std::setprecision(1)
                  << tiled_time << " ms  " << tiled_gflops << " GFLOPS\n";
        std::cout << "Tiled vs ikj speedup: " << std::fixed << std::setprecision(2)
                  << s_ikj.median / tiled_time << "x\n";
    }

    return 0;
}
