#include "gemm/kernels.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

// Sizes cover: tiny, one short of / exactly / one past the 4-row, 8-column and
// 64-wide tile edges, and several 64-deep k-tiles.
static const int kSizes[] = {1, 3, 4, 7, 8, 9, 31, 63, 64, 65, 100, 127, 128, 129, 200};

struct Reference {
    std::vector<double> value;    // exact-ish product in double precision
    std::vector<double> abs_dot;  // sum_k |A_ik| * |B_kj|, scales the error bound
};

static Reference reference_product(const std::vector<float>& A, const std::vector<float>& B, int n)
{
    Reference ref{std::vector<double>(n * n), std::vector<double>(n * n)};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0, abs_sum = 0;
            for (int k = 0; k < n; k++) {
                double product = double(A[i * n + k]) * B[k * n + j];
                sum += product;
                abs_sum += std::fabs(product);
            }
            ref.value[i * n + j] = sum;
            ref.abs_dot[i * n + j] = abs_sum;
        }
    return ref;
}

// Standard float dot-product error bound (n * eps * sum|a||b|), doubled for
// the reassociation that -ffast-math and FMA contraction allow.
static bool within_bound(float got, double want, double abs_dot, int n)
{
    return std::fabs(got - want) <= 2.0 * n * FLT_EPSILON * abs_dot;
}

static bool check_kernel(const KernelInfo& kernel, int n, const std::vector<float>& A,
                         const std::vector<float>& B, const Reference& ref)
{
    // NaN catches kernels that read C before writing it.
    std::vector<float> C(n * n, std::numeric_limits<float>::quiet_NaN());
    kernel.fn(A.data(), B.data(), C.data(), n);

    for (int idx = 0; idx < n * n; idx++)
        if (!within_bound(C[idx], ref.value[idx], ref.abs_dot[idx], n)) {
            std::printf("FAIL %-24s N=%-4d C[%d][%d] = %.9g, expected %.9g\n", kernel.name, n,
                        idx / n, idx % n, C[idx], ref.value[idx]);
            return false;
        }
    return true;
}

int main()
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int failures = 0, checks = 0;

    for (int n : kSizes) {
        std::vector<float> A(n * n), B(n * n);
        for (float& v : A) v = dist(rng);
        for (float& v : B) v = dist(rng);
        const Reference ref = reference_product(A, B, n);

        for (const KernelInfo& kernel : all_kernels()) {
            checks++;
            failures += !check_kernel(kernel, n, A, B, ref);
        }
    }

    std::printf("%d/%d kernel-size checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
