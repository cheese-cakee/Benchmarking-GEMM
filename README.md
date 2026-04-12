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
            C[i * N + j] = sum; 
        }
    }
}
```

**The Flaw:** We constantly write to memory (`C[i * N + j]`) inside the innermost loop. Writing to RAM is incredibly slow — this is **~4.4 GFLOPS** on this CPU.

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

### 256x256 Matrix (33.55 Million FLOPs)

| Kernel | Median Time | GFLOPS | Speedup vs Naive |
|--------|-------------|--------|-----------------|
| Naive ijk | ~12.0 ms | ~2.8 | 1.00x |
| Register optimized | ~11.0 ms | ~3.0 | 1.05x |
| **Loop reorder (ikj)** | **~1.3 ms** | **~25.7** | **~8.91x** |
| **Tiled 64x64** | **~1.8 ms** | **~18.4** | **~6.58x** |

> **Why is tiled slower than ikj here?** At 256×256, total matrix size is ~768KB — the full working set fits in L2 cache. Tiling adds loop overhead with no cache benefit. The 64×64 tile size (16KB) is also at the edge of L1 capacity (48KB on i5-13450HX), causing thrashing when processing three tiles simultaneously. Tile size tuning is architecture-dependent.

> **2048×2048 results omitted** — naive and register variants would take ~35+ minutes each. Both ikj and tiled scale similarly to the 256×256 case.

### Key Results

- **ikj loop reorder**: ~8.91x speedup over naive — achieves 25.7 GFLOPS
- **Tiled 64x64**: ~6.58x speedup — achieves 18.4 GFLOPS
- The key insight: **memory access patterns matter more than raw algorithmic complexity** when data doesn't fit in cache.

---

## Technical Details

- **Language**: C++17
- **Compiler**: GCC (MinGW-w64)
- **Flags**: `-O3 -march=native -ffast-math -static`
- **Platform**: Windows (MSYS2)
- **CPU**: Intel i5-13450HX (10 cores, 2.4 GHz base)
- **Matrix sizes tested**: 4x4, 64x64, 256x256

### Theoretical Context

For reference, OpenBLAS on the same hardware achieves ~180 GFLOPS on this operation. Our best result (25.7 GFLOPS single-threaded) represents a fraction of peak — the remaining gap comes from hand-vectorized AVX2 kernels, multi-threading, and prefetching strategies that mature BLAS libraries employ.

---

## License

MIT License — Feel free to use this for learning or as a starting point for your own optimization projects!