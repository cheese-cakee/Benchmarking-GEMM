#include "gemm/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cpuid.h>
#include <omp.h>

#ifndef GEMM_BUILD_DESCRIPTION
#define GEMM_BUILD_DESCRIPTION "unknown"
#endif

struct Options {
    std::vector<int> sizes = {256, 2048};
    std::vector<std::string> kernels;  // empty: every kernel, minus slow ones above 512
    int warmup = 2;
    int reps = 5;
    int threads = 0;  // 0: OpenMP default
    int cooldown_ms = 0;
    std::string csv_path;
};

static void usage(const char* program)
{
    std::printf(
        "Usage: %s [options]\n"
        "  --sizes 256,2048       matrix sizes N (default 256,2048)\n"
        "  --kernels a,b          kernel names (default: all; naive/register only for N <= 512)\n"
        "  --warmup W             untimed runs per kernel (default 2)\n"
        "  --reps R               timed runs per kernel (default 5)\n"
        "  --threads T            OpenMP threads (default: OpenMP's choice)\n"
        "  --cooldown-ms MS       sleep between timed runs to limit thermal drift (default 0)\n"
        "  --csv PATH             write every sample to PATH\n"
        "  --list                 print kernel names and exit\n",
        program);
}

static std::vector<std::string> split(const std::string& text)
{
    std::vector<std::string> parts;
    std::stringstream stream(text);
    for (std::string part; std::getline(stream, part, ',');)
        if (!part.empty()) parts.push_back(part);
    return parts;
}

static Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; i++) {
        const std::string flag = argv[i];
        if (flag == "--list") {
            for (const KernelInfo& kernel : all_kernels()) std::printf("%s\n", kernel.name);
            std::exit(0);
        }
        if (flag == "--help" || i + 1 >= argc) {
            usage(argv[0]);
            std::exit(flag == "--help" ? 0 : 2);
        }
        const std::string value = argv[++i];
        if (flag == "--sizes") {
            options.sizes.clear();
            for (const std::string& size : split(value)) options.sizes.push_back(std::stoi(size));
        } else if (flag == "--kernels") {
            options.kernels = split(value);
        } else if (flag == "--warmup") {
            options.warmup = std::stoi(value);
        } else if (flag == "--reps") {
            options.reps = std::stoi(value);
        } else if (flag == "--threads") {
            options.threads = std::stoi(value);
        } else if (flag == "--cooldown-ms") {
            options.cooldown_ms = std::stoi(value);
        } else if (flag == "--csv") {
            options.csv_path = value;
        } else {
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (options.reps < 1 || options.warmup < 0) {
        usage(argv[0]);
        std::exit(2);
    }
    return options;
}

static std::vector<KernelInfo> select_kernels(const Options& options, int n)
{
    std::vector<KernelInfo> selected;
    for (const KernelInfo& kernel : all_kernels()) {
        const bool requested = std::find(options.kernels.begin(), options.kernels.end(), kernel.name) != options.kernels.end();
        if (options.kernels.empty() ? !(kernel.slow && n > 512) : requested) selected.push_back(kernel);
    }
    return selected;
}

static std::string cpu_brand()
{
    unsigned regs[12] = {};
    for (unsigned leaf = 0; leaf < 3; leaf++)
        if (!__get_cpuid(0x80000002 + leaf, &regs[leaf * 4], &regs[leaf * 4 + 1], &regs[leaf * 4 + 2], &regs[leaf * 4 + 3]))
            return "unknown";
    char brand[49] = {};
    std::memcpy(brand, regs, 48);
    std::string text = brand;
    text.erase(0, text.find_first_not_of(' '));
    return text;
}

// Spot-checks sampled entries against a double-precision dot product so that
// no benchmark number comes from a wrong result.
static bool spot_check(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, int n)
{
    std::mt19937 rng(n);
    std::uniform_int_distribution<int> index(0, n - 1);
    for (int sample = 0; sample < 64; sample++) {
        const int i = index(rng), j = index(rng);
        double sum = 0, abs_sum = 0;
        for (int k = 0; k < n; k++) {
            const double product = double(A[i * n + k]) * B[k * n + j];
            sum += product;
            abs_sum += std::fabs(product);
        }
        if (!(std::fabs(C[i * n + j] - sum) <= 2.0 * n * FLT_EPSILON * abs_sum)) return false;
    }
    return true;
}

static double median_of(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

static double median_absolute_deviation(const std::vector<double>& values)
{
    const double center = median_of(values);
    std::vector<double> deviations;
    for (double v : values) deviations.push_back(std::fabs(v - center));
    return median_of(deviations);
}

static std::string utc_now()
{
    char text[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    return text;
}

int main(int argc, char** argv)
{
    const Options options = parse_options(argc, argv);
    if (options.threads > 0) omp_set_num_threads(options.threads);
    const int threads = omp_get_max_threads();

    FILE* csv = nullptr;
    if (!options.csv_path.empty()) {
        csv = std::fopen(options.csv_path.c_str(), "w");
        if (!csv) {
            std::perror(options.csv_path.c_str());
            return 2;
        }
        std::fprintf(csv, "# cpu: %s\n# compiler: %s\n# build: %s\n# threads: %d\n# warmup: %d, reps: %d, cooldown_ms: %d\n# started_utc: %s\n",
                     cpu_brand().c_str(), __VERSION__, GEMM_BUILD_DESCRIPTION, threads, options.warmup, options.reps,
                     options.cooldown_ms, utc_now().c_str());
        std::fprintf(csv, "n,kernel,rep,ms,gflops\n");
    }

    std::printf("CPU: %s | %s | %s | %d threads\n", cpu_brand().c_str(), __VERSION__, GEMM_BUILD_DESCRIPTION, threads);
    bool all_verified = true;

    for (int n : options.sizes) {
        const std::vector<KernelInfo> kernels = select_kernels(options, n);
        const double flops = 2.0 * n * n * n;
        std::vector<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& v : A) v = dist(rng);
        for (float& v : B) v = dist(rng);

        for (const KernelInfo& kernel : kernels)
            for (int w = 0; w < options.warmup; w++) kernel.fn(A.data(), B.data(), C.data(), n);

        // Interleave kernels within each repetition so slow drift (clocks,
        // temperature) is spread across all of them instead of biasing one.
        std::vector<std::vector<double>> times(kernels.size());
        std::vector<bool> verified(kernels.size(), true);
        for (int rep = 0; rep < options.reps; rep++)
            for (size_t k = 0; k < kernels.size(); k++) {
                if (options.cooldown_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(options.cooldown_ms));
                const auto start = std::chrono::steady_clock::now();
                kernels[k].fn(A.data(), B.data(), C.data(), n);
                const auto stop = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
                times[k].push_back(ms);
                verified[k] = verified[k] && spot_check(A, B, C, n);
                if (csv) std::fprintf(csv, "%d,%s,%d,%.4f,%.3f\n", n, kernels[k].name, rep, ms, flops / (ms * 1e6));
            }

        std::printf("\nN = %d (%.3g GFLOP per multiply)\n", n, flops / 1e9);
        std::printf("%-24s %12s %10s %12s %10s %9s\n", "kernel", "median ms", "MAD ms", "min ms", "GFLOPS", "verified");
        for (size_t k = 0; k < kernels.size(); k++) {
            const double median = median_of(times[k]);
            std::printf("%-24s %12.3f %10.3f %12.3f %10.1f %9s\n", kernels[k].name, median,
                        median_absolute_deviation(times[k]), *std::min_element(times[k].begin(), times[k].end()),
                        flops / (median * 1e6), verified[k] ? "yes" : "NO");
            all_verified = all_verified && verified[k];
        }
        std::fflush(stdout);
        if (csv) std::fflush(csv);
    }

    if (csv) std::fclose(csv);
    return all_verified ? 0 : 1;
}
