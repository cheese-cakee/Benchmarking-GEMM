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

All benchmarks were run on an **Intel i5-13450HX** (13th Gen, 10 cores) multiplying two **2048x2048** floating-point matrices.

### 1. The Naive Implementation (ijk loop)

The most natural way to write matrix multiplication is a triple-nested loop corresponding to the mathematical formula:

```cpp
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum; // Writes to RAM every inner iteration!
        }
    }
}
```

**The Flaw:** We constantly write to memory (`C[i * N + j]`) inside the innermost loop. Writing to RAM is incredibly slow. On large matrices this collapses performance due to write-through traffic saturating memory bandwidth.

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

**The Flaw:** We are still thrashing the CPU cache. In C++, matrices are stored in row-major order. Accessing `B[k * N + j]` inside the k loop forces the CPU to jump forward in memory by N elements every iteration, missing the cache entirely. At small sizes the compiler may auto-correct the naive version, but on large matrices the write-through penalty is severe.

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

**The Fix:** Now the innermost loop iterates over j. Both C and B are accessed sequentially (+1 offset in memory). The CPU can load entire 64-byte cache lines at once, eliminating RAM bottlenecking.

### 4. Tiled (Blocked) Optimization

Even with loop reordering, large matrices can't fit in cache. We divide matrices into tiles that DO fit in cache:

```cpp
for (int i = 0; i < N; i += tile_size) {
    for (int k = 0; k < N; k += tile_size) {
        for (int j = 0; j < N; j += tile_size) {
            // Process tile with ikj inside
            for (int ii = i; ii < min(i+tile_size, N); ii++) {
                for (int kk = k; kk < min(k+tile_size, N); kk++) {
                    float temp = A[ii * N + kk];
                    for (int jj = j; jj < min(j+tile_size, N); jj++) {
                        C[ii * N + jj] += temp * B[kk * N + jj];
                    }
                }
            }
        }
    }
}
```

### 5. Compiler Flags

Writing cache-friendly code is only half the battle. Unleashing the compiler pushes it to the limit:

- `-O3`: Enables aggressive optimizations (loop unrolling, function inlining, vectorization)
- `-march=native`: Uses CPU-specific instructions for your architecture
- `-ffast-math`: Enables faster (though sometimes less precise) mathematical operations
- `-static`: Required on Windows to avoid DLL issues

---

## Benchmark Results

*Run on Intel i5-13450HX, Windows 11, MinGW-w64 GCC 14.2.0*  
*Compile flags: `-O3 -march=native -ffast-math -static -lpdh`*

### 256x256 Matrix (33.55 Million FLOPs)

| Kernel | Median Time | GFLOPS | Speedup vs Naive |
|--------|-------------|--------|-----------------|
| Naive ijk | 8.2 ms | 4.1 | 1.00x |
| Register optimized | 7.6 ms | 4.4 | 1.08x |
| **Loop reorder (ikj)** | **0.8 ms** | **40.2** | **9.77x** |
| Tiled 64x64 | 1.3 ms | 26.5 | 6.43x |
| **AVX2 ikj** | **0.8 ms** | **40.5** | **10.03x** |

> **Note:** The gap between Naive and Register is small here because `-O3` auto-optimizes the repeated memory write. The true penalty of the naive approach appears at large sizes where write-through saturates memory bandwidth (hence omitted at 2048×2048).

> **Why is tiled slower than ikj here?** At 256×256, total matrix size is ~768KB — the full working set fits in L2 cache. Tiling adds loop overhead with no cache benefit. The 64×64 tile size is also at the edge of L1 capacity (48KB on i5-13450HX), causing thrashing when processing three tiles simultaneously.

### 2048x2048 Matrix (17.18 Billion FLOPs)

| Kernel | Median Time | GFLOPS | Notes |
|--------|-------------|--------|-------|
| **Loop reorder (ikj)** | **817.0 ms** | **21.0** | Baseline for large matrices |
| AVX2 ikj | 920.2 ms | 18.7 | No gain — memory bandwidth bottleneck, same access pattern as ikj |
| **Tiled 64x64** | **549.6 ms** | **31.3** | **1.49× faster than ikj** |

> **Why does tiled win at 2048×2048 but lose at 256×256?** At 2048×2048 the working set (~48MB) exceeds L3 cache. Tiling keeps active data in L1/L2, whereas `ikj` streams through the entire matrix set repeatedly. Tiling is not universally better — it is a **size-dependent optimization**.

### Key Results

- **ikj loop reorder**: ~9.8x speedup over naive at 256×256 — achieves 40.2 GFLOPS
- **Tiling**: Slightly slower than `ikj` at 256×256 (26.5 GFLOPS), but **1.49× faster** at 2048×2048 (31.3 GFLOPS)
- **AVX2**: Same speed as `ikj` at 256×256 (40.5 vs 40.2 GFLOPS) because the compiler already auto-vectorized the same loop. Slower at 2048×2048 (18.7 vs 19.6 GFLOPS) — **naive SIMD without register-blocking is useless when memory-bound**
- The key insight: **there is no single best algorithm — optimization is context-dependent. Cache-friendly access matters, but so does matching tile size to your memory hierarchy.**

---

## Technical Details

- **Language**: C++17
- **Compiler**: GCC (MinGW-w64) 14.2.0
- **Compile flags**: `-O3 -march=native -ffast-math -static`
- **Link flags**: `-lpdh` (required for Windows performance counters)
- **Build command**: `g++ -std=c++17 -Wall -Wextra -O3 -march=native -ffast-math -static -o gemm_bench.exe src/gemm_all.cpp -lpdh`
- **Platform**: Windows 11
- **CPU**: Intel i5-13450HX (10 cores, 2.4 GHz base, 48KB L1d, 1.25MB L2, 20MB L3)
- **Matrix sizes tested**: 4×4, 64×64, 256×256, 2048×2048

### Theoretical Context

For reference, OpenBLAS on the same hardware achieves ~180 GFLOPS on this operation. Our current best result (40.2 GFLOPS single-threaded at 256×256, 31.3 GFLOPS at 2048×2048) represents a fraction of peak. The remaining gap comes from:
- **SIMD vectorization** (AVX2/AVX-512) — next step in this project
- **Multi-threading** — utilizing all 10 cores
- **Prefetching** — hiding memory latency
- **Micro-kernel architecture** — register-blocking like BLIS/OpenBLAS

---

## License

MIT License — Feel free to use this for learning or as a starting point for your own optimization projects!