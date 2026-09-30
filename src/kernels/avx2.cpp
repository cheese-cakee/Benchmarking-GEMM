#include "gemm/kernels.hpp"

#include <immintrin.h>

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
