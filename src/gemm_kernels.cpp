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

void gemm_tiled(const float* A, const float* B, float* C, int N, int tile_size) {
    for (int i = 0; i < N * N; i++) C[i] = 0;
    for (int i = 0; i < N; i += tile_size) {
        for (int k = 0; k < N; k += tile_size) {
            for (int j = 0; j < N; j += tile_size) {
                for (int ii = i; ii < (i + tile_size < N ? i + tile_size : N); ii++) {
                    for (int kk = k; kk < (k + tile_size < N ? k + tile_size : N); kk++) {
                        float temp = A[ii * N + kk];
                        for (int jj = j; jj < (j + tile_size < N ? j + tile_size : N); jj++) {
                            C[ii * N + jj] += temp * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}