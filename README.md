# GEMM Optimization Benchmarks in C++

This repository explores the step-by-step optimization of **General Matrix Multiplication (GEMM)** in C++. GEMM is the core mathematical operation behind nearly all modern deep learning models, including Large Language Models and Vision Transformers.

This project starts with a mathematically correct but highly inefficient "naive" implementation and gradually applies memory, caching, and compiler optimizations to achieve massive performance gains.

---

## What is GEMM and FLOPs?

At its core, GEMM computes each output element of a matrix as a dot product:

```
C_ij = sum from k=0 to N-1 of (A_ik * B_kj)
```

**FLOPs (Floating Point Operations)** is the metric used to measure computational cost.

For an N x N matrix multiplication:
- The output matrix has N² elements
- Calculating a single element requires N multiplications and N-1 additions
- Total operations per element: N + (N-1) approximately equals 2N
- Total FLOPs = N² x 2N = **2N³**

Because matrix multiplication scales at **O(N³)**, doubling the matrix size increases the computational workload by a factor of **8**.

---

## The Optimization Journey

All benchmarks were run on an **Intel i5-13450HX** (13th Gen) multiplying two **2048x2048** floating-point matrices.

### 1. The Naive Implementation (ijk loop)

The most natural way to write matrix multiplication is a triple-nested loop corresponding to the mathematical formula:

```cpp
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum; 
        }
    }
}
```

**The Flaw:** We constantly write to memory (`C[i * N + j]`) inside the innermost loop. Writing to RAM is incredibly slow.

### 2. Register Optimization

We can easily speed this up by accumulating the dot product inside a local variable (which the compiler places in a high-speed CPU register) and only writing to memory once per output element.

```cpp
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum; // Moved OUTSIDE the k-loop!
    }
}
```

**The Flaw:** We are still thrashing the CPU cache. In C++, matrices are stored in row-major order. Accessing `B[k * N + j]` inside the k loop forces the CPU to jump forward in memory by N elements every iteration, missing the cache entirely.

### 3. Loop Reordering (ikj Loop) - The Cache Magic

By swapping the two inner loops, we fundamentally change the memory access pattern:

```cpp
for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
        float temp = A[i * N + k]; // Load once
        for (int j = 0; j < N; j++) {
            C[i * N + j] += temp * B[k * N + j]; // Sequential access!
        }
    }
}
```

**The Fix:** Now the innermost loop iterates over j. Both C and B are accessed sequentially (+1 offset in memory). The CPU can load entire 64-byte cache lines at once, eliminating RAM bottlenecking. This simple change yields massive speedups without altering the math.

### 4. Compiler Flags

Writing cache-friendly code is only half the battle. Unleashing the compiler pushes it to the limit:

- `-O3`: Enables aggressive optimizations (loop unrolling, function inlining, vectorization)
- `-march=native`: Uses CPU-specific instructions for your architecture
- `-ffast-math`: Enables faster (though sometimes less precise) mathematical operations

---

## Benchmark Results

Our benchmarking framework validates the math on small matrices before running heavy loads.

### Visual & Correctness Verification (4x4 & 64x64)

![4x4 Visual Verification - Before Optimization](./images/before-4x4-visual.png)
![64x64 Correctness Check - Before Optimization](./images/before-64x64-check.png)

Verifying that all kernels produce identical, mathematically correct results within a strict epsilon tolerance.

### The Final Showdown (2048x2048 Full Size)

![Benchmark Results - After Optimization](./images/after-2048-benchmark.png)
![Speedup Comparison - After Optimization](./images/after-speedup-comparison.png)

With compiler optimizations, the ikj kernel completes **17.18 Billion FLOPs** in just **5.5 seconds**, achieving **~3.2 GFLOPS**. The naive implementation is projected to be nearly **28x slower**.

| Kernel | 256x256 Time | 256x256 GFLOPS | Speedup |
|--------|---------------|----------------|---------|
| Naive ijk | ~12 ms | ~2.9 | 1x |
| Register | ~11 ms | ~3.1 | 1.1x |
| **ikj** | **~1.2 ms** | **~28** | **9.8x** |

---

## How to Build and Run

To run these benchmarks on your own machine, clone the repository and compile with aggressive optimization flags:

```bash
# Clone the repo
git clone https://github.com/cheese-cakee/GEMM-Benchmarking.git
cd GEMM-Benchmarking

# Compile with GCC/Clang (Highly Recommended Flags)
g++ -std=c++17 -O3 -march=native -ffast-math -static -o optimized.exe src/gemm_all.cpp

# Run
./optimized.exe
```

> **Note:** The `-static` flag is required on Windows/MSYS2 to work around linker issues.

---

## Project Structure

```
├── src/
│   ├── gemm.hpp          # Header with all kernel declarations
│   ├── gemm_kernels.cpp # Kernel implementations  
│   ├── gemm_all.cpp     # Main benchmark suite (all-in-one)
│   └── main.cpp         # Original benchmark harness
├── Makefile             # Build automation
├── baseline.exe        # Pre-built naive binary (no optimization)
└── optimized.exe       # Pre-built optimized binary (all kernels)
```

---

## Key Takeaways

This project demonstrates a fundamental truth of high-performance computing:

> **Understanding computer hardware (memory and caches) is just as important as understanding Big O notation.**

Simple loop reordering — without changing the math at all — can yield **~10x speedups**. The key insight is that **memory access patterns matter more than raw algorithmic complexity** when data doesn't fit in cache.

| Optimization | Speedup |
|--------------|---------|
| Register optimization | ~1.3x |
| Loop reordering (ikj) | ~10-20x |
| Compiler flags (-O3 -march=native -ffast-math) | Up to 65x |
