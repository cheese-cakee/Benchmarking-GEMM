# Optimizing SGEMM on a CPU, from first principles

This guide walks through the ten kernels in [`src/kernels/`](../src/kernels) in the order they
were written. Each step names the bottleneck it removes and the one it leaves behind. Measured
numbers are in the [README](../README.md#results); this document explains why they look the way
they do.

All kernels compute `C = A × B` for square, row-major `N × N` single-precision matrices.

## 1. What we are counting

Each output element is a dot product of a row of `A` and a column of `B`:

```
C[i][j] = Σ_k A[i][k] · B[k][j]
```

That is `N` multiplies and `N` adds per element, `2N` floating-point operations (FLOPs), and
`N²` elements, so one multiply costs `2N³` FLOPs. At `N = 2048` that is 17.2 GFLOP. Throughput
is reported as GFLOPS = `2N³ / seconds / 10⁹`.

The data is only `3N²` floats (48 MiB at `N = 2048`), but the work is `2N³`. Every float of `A`
and `B` is used `N` times. **Arithmetic intensity**, FLOPs per byte moved from memory, can
therefore be very high, but only if the kernel actually reuses data while it is close to the
core. The whole story below is about creating that reuse.

## 2. The machine

The reference machine is an Intel Core i5-13450HX: 6 performance cores (P-cores, two hardware
threads each) and 4 efficiency cores (E-cores), 16 threads in total.

| Resource | Size (per P-core unless noted) | Rough latency |
| --- | --- | --- |
| Vector registers | 16 × 256-bit YMM (8 floats each) | 0 cycles |
| L1 data cache | 48 KiB | ~5 cycles |
| L2 cache | 1.25 MiB | ~15 cycles |
| L3 cache | 20 MiB, shared | ~50 cycles |
| DRAM | DDR5 | ~100 ns |

Memory moves in 64-byte **cache lines**. Walking memory sequentially is cheap because every
line fetched is fully used and the hardware prefetcher streams the next lines ahead of time.

**Peak compute.** A P-core has two FMA units. Each executes a fused multiply-add on 8 floats per
cycle, and an FMA counts as 2 FLOPs, so one core peaks at `2 × 8 × 2 = 32` FLOPs per cycle.

| Assumption | Peak |
| --- | --- |
| One P-core at 4.6 GHz (single-core turbo) | 147 GFLOPS |
| Six P-cores at 4.0 GHz (a typical sustained all-core AVX2 clock) | 768 GFLOPS |
| Six P-cores at 4.6 GHz (upper bound, not sustainable) | 883 GFLOPS |

E-cores add some throughput at lower clocks and with narrower execution, so the six-P-core
figure is the honest yardstick. Laptop power and thermal limits move the real sustained clock,
which is why results are reported with the date, machine, and power state. The measured
`packed4x8_omp` result, 323 GFLOPS, is about 42% of the 768 GFLOPS six-P-core figure.

## 3. The kernels

### 3.1 `naive`: the formula as written

```cpp
for i, for j:
    sum = 0
    for k: sum += A[i][k] * B[k][j];  C[i][j] = sum;   // store inside the k loop
```

Two problems. The store to `C` inside the innermost loop turns every iteration into a memory
write. Worse, `B[k][j]` walks down a column: consecutive `k` values are `N` floats apart, so
every access touches a different cache line and uses 4 bytes of the 64 fetched.

### 3.2 `register`: accumulate in a register

Moving the store out of the `k` loop lets the compiler keep `sum` in a register. It removes the
store traffic but not the column walk over `B`, so the gain is small.

### 3.3 `ikj`: reorder the loops

```cpp
for i, for k:
    a = A[i][k]
    for j: C[i][j] += a * B[k][j];      // both C and B walk along a row
```

Same arithmetic, different order. The innermost loop now reads `B` and updates `C` sequentially,
so every cache line is fully used, the prefetcher can stream, and the compiler can vectorize the
`j` loop. This single change is the largest jump at small sizes.

What is left: for each `(i, k)` the kernel streams a whole row of `B` and a whole row of `C`
(16 KiB at `N = 2048`) and does only one FMA per loaded float. Once the matrices no longer fit in
cache, the kernel is bandwidth-bound.

### 3.4 `tiled64`: blocking for the cache

Split the loops into 64 × 64 tiles so that the pieces of `A`, `B`, and `C` in use stay resident
in L1/L2 while they are reused. Tiling helps at large `N`, where `ikj` falls out of cache, but it
adds loop overhead that costs a little at small `N`.

### 3.5 `avx2_ikj`: explicit SIMD

The `ikj` loop written with AVX2 intrinsics: broadcast `A[i][k]` into all 8 lanes, load 8 floats
of `B`, fused multiply-add into 8 floats of `C`. With `-O3 -march=native` the compiler already
vectorizes `ikj`, so this is roughly a tie. It confirms that the bottleneck is memory traffic,
not instruction selection.

### 3.6 `micro4x8`: register blocking

The key idea of every fast GEMM: compute a small block of `C` entirely in registers.

```cpp
for each 4-row, 8-column block of C:
    acc0..acc3 = 0                          // 4 YMM registers = 4 × 8 outputs
    for k:
        b = load 8 floats of B[k][j..j+7]   // one load
        acc0 += broadcast(A[i+0][k]) * b    // four FMAs reuse b
        acc1 += broadcast(A[i+1][k]) * b
        acc2 += broadcast(A[i+2][k]) * b
        acc3 += broadcast(A[i+3][k]) * b
    store acc0..acc3 to C                   // one store per output
```

Each loaded vector of `B` now feeds 4 FMAs, and `C` is written once instead of `N` times. At small
`N` this is the fastest single-threaded kernel. At large `N` it collapses: the `k` loop walks down
columns of `A` and rows of `B` with an `N`-float stride, which misses the cache on almost every
iteration.

### 3.7 `packed4x8`: packing feeds the micro-kernel

Copy a 64-deep slice of `A` and an 8-wide slice of `B` into small contiguous buffers first, in
exactly the order the micro-kernel will read them. The copy costs `O(N²)` per tile; the reuse is
`O(N³)`, so it pays for itself. Every load inside the micro-kernel is now sequential and hits L1.

Two correctness details matter here:

- **Accumulating across k-tiles.** The `k` loop is split into 64-deep tiles, so each call adds a
  partial sum. The micro-kernel therefore starts from the current value of `C`, not from zero.
  (An early version started from zero and overwrote `C`; a test that used only `N = 64`, a
  single k-tile, could not catch it. The test suite now covers sizes that span several tiles.)
- **Ragged edges.** When the last tile is narrower than 8 columns, a lane mask loads and stores
  only the valid columns. Leftover rows (when `N` is not a multiple of 4) use a scalar loop.

### 3.8 `packed4x8_prefetch`: when prefetching hurts

Adding `_mm_prefetch` two iterations ahead, with the non-temporal hint, made the kernel
**slower**. The packed buffers are small (2 KiB of `B`, 16 KiB of `A`) and already in L1, and the
access is sequential, which the hardware prefetcher handles on its own. Software prefetches add
instructions to the tightest loop, and the non-temporal hint asks the cache to avoid keeping
data that we are about to reuse. Prefetching helps for irregular access or data that is far
away, not for a sequential stream that already fits in L1.

### 3.9 `packed4x8_omp`: using every core

`#pragma omp parallel for schedule(dynamic)` over 64-row strips of `C`. Each thread owns its
strip, so no two threads write the same output and no locking is needed. Dynamic scheduling
matters on a hybrid CPU: a strip on an E-core takes longer, and dynamic scheduling hands the next
strip to whichever thread is free.

At small `N` the parallel speedup is limited: 256 / 64 = 4 strips means at most four threads
have work, and waking the thread team is a noticeable share of a sub-millisecond multiply. At
`N = 2048` there are 32 strips and ten threads give about a 5.6× speedup over `packed4x8`.

## 4. Where the remaining gap is

Production BLAS libraries (OpenBLAS, MKL, BLIS) reach roughly 80-90% of peak. The biggest
differences from this project, in the order they would likely pay off:

1. **Larger register block.** A 4 × 8 block uses 4 of 16 YMM registers for accumulators and does
   4 FMAs per load of `B`. A 6 × 16 block uses 12 accumulators and does 12 FMAs per two loads,
   which is enough to keep both FMA units busy.
2. **Three-level blocking (the BLIS loop order).** Choose the k-depth so a sliver of `B` fits L1,
   the row block so the packed `A` fits L2, and the column block so the packed `B` fits L3.
3. **Pack once, reuse many times.** The kernels here repack each `B` sliver for every 64-row strip
   and allocate the buffers inside the loop. A shared packed `B` block per thread team, and
   buffers allocated once and aligned to 64 bytes, remove that overhead.
4. **Hybrid-aware scheduling.** Give P-cores and E-cores work in proportion to their speed, or
   pin threads to P-cores only.

These are listed as next steps in the README; the current kernels are kept as they are because
they illustrate one idea per step.
