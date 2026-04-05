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