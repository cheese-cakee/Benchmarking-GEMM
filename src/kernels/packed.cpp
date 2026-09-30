#include "gemm/kernels.hpp"

#include <algorithm>
#include <vector>

#include <immintrin.h>

static void pack_A(const float* A, float* packed, int N, int i_start, int k_start, int mc, int kc)
{
    for(int kk = 0;kk < kc; kk++)
    {
        for(int ii = 0; ii < mc;ii++)
        {
            packed[kk * mc + ii] = A[(i_start + ii) * N + (k_start + kk)];
        }
    }
}

static void pack_B(const float* B, float* packed, int N, int k_start, int j_start, int kc, int nr)
{
    for(int kk = 0;kk < kc;kk++)
    {
        for(int jj = 0; jj < nr; jj++)
        {
            packed[kk * nr + jj] = (j_start + jj < N) ? B[(k_start + kk)* N + (j_start + jj)] : 0.0f;
        }
    }
}

// Lanes [0, nr) enabled: selects the valid columns of a partial 8-wide tile.
static __m256i tail_mask(int nr)
{
    return _mm256_setr_epi32(
        nr > 0 ? -1 : 0, nr > 1 ? -1 : 0, nr > 2 ? -1 : 0, nr > 3 ? -1 : 0,
        nr > 4 ? -1 : 0, nr > 5 ? -1 : 0, nr > 6 ? -1 : 0, nr > 7 ? -1 : 0
    );
}

static void gemm_packed_4x8(const float* A_packed, const float* B_packed, float* C, int N, int i, int j, int kc, int mc, int nr)
{
    // C already holds the partial sums of earlier k-tiles.
    __m256i mask = tail_mask(nr);
    __m256 acc0 = _mm256_maskload_ps(&C[(i + 0) * N + j], mask);
    __m256 acc1 = _mm256_maskload_ps(&C[(i + 1) * N + j], mask);
    __m256 acc2 = _mm256_maskload_ps(&C[(i + 2) * N + j], mask);
    __m256 acc3 = _mm256_maskload_ps(&C[(i + 3) * N + j], mask);

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

    if (nr == 8) {
        _mm256_storeu_ps(&C[(i+0)*N + j], acc0);
        _mm256_storeu_ps(&C[(i+1)*N + j], acc1);
        _mm256_storeu_ps(&C[(i+2)*N + j], acc2);
        _mm256_storeu_ps(&C[(i+3)*N + j], acc3);
    } else {
        _mm256_maskstore_ps(&C[(i+0)*N + j], mask, acc0);
        _mm256_maskstore_ps(&C[(i+1)*N + j], mask, acc1);
        _mm256_maskstore_ps(&C[(i+2)*N + j], mask, acc2);
        _mm256_maskstore_ps(&C[(i+3)*N + j], mask, acc3);
    }

}

static void gemm_packed_4x8_prefetch(const float* A_packed, const float* B_packed, float* C,
                             int N, int i, int j, int kc, int mc, int nr)
{
    // C already holds the partial sums of earlier k-tiles.
    __m256i mask = tail_mask(nr);
    __m256 acc0 = _mm256_maskload_ps(&C[(i + 0) * N + j], mask);
    __m256 acc1 = _mm256_maskload_ps(&C[(i + 1) * N + j], mask);
    __m256 acc2 = _mm256_maskload_ps(&C[(i + 2) * N + j], mask);
    __m256 acc3 = _mm256_maskload_ps(&C[(i + 3) * N + j], mask);

    if (kc > 1) {
        _mm_prefetch((const char*)&B_packed[1 * 8], _MM_HINT_NTA);
        _mm_prefetch((const char*)&A_packed[1 * mc], _MM_HINT_NTA);
    }

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

    if (nr == 8) {
        _mm256_storeu_ps(&C[(i + 0) * N + j], acc0);
        _mm256_storeu_ps(&C[(i + 1) * N + j], acc1);
        _mm256_storeu_ps(&C[(i + 2) * N + j], acc2);
        _mm256_storeu_ps(&C[(i + 3) * N + j], acc3);
    } else {
        _mm256_maskstore_ps(&C[(i + 0) * N + j], mask, acc0);
        _mm256_maskstore_ps(&C[(i + 1) * N + j], mask, acc1);
        _mm256_maskstore_ps(&C[(i + 2) * N + j], mask, acc2);
        _mm256_maskstore_ps(&C[(i + 3) * N + j], mask, acc3);
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
                            kc, mc_main, nr
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

void gemm_blocked_4x8_packed_omp(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N * N; i++) C[i] = 0;

    const int TILE = 64;
    const int MR = 4;
    const int NR = 8;

    #pragma omp parallel for schedule(dynamic)
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
                            kc, mc_main, nr
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
                            kc, mc_main, nr
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

void gemm_blocked_4x8_packed_prefetch_omp(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N * N; i++) C[i] = 0;

    const int TILE = 64;
    const int MR = 4;
    const int NR = 8;

    #pragma omp parallel for schedule(dynamic)
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
                            kc, mc_main, nr
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
