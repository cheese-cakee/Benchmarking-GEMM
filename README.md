# Benchmarking-GEMM

[![CI](https://github.com/cheese-cakee/Benchmarking-GEMM/actions/workflows/ci.yml/badge.svg)](https://github.com/cheese-cakee/Benchmarking-GEMM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Single-precision matrix multiplication (SGEMM) in C++17, optimized one step at a time: from the
textbook triple loop to a packed, register-blocked AVX2 micro-kernel running on every core.
Each kernel isolates one idea, and every result is verified against a double-precision
reference before it is reported.

<!-- RESULTS -->

## How it works

Ten kernels, each adding one technique to the previous ones:

| Kernel | Idea | Source |
| --- | --- | --- |
| `naive` | Textbook `i, j, k` loops | [`scalar.cpp`](src/kernels/scalar.cpp) |
| `register` | Accumulate the dot product in a register | [`scalar.cpp`](src/kernels/scalar.cpp) |
| `ikj` | Reorder loops so `B` and `C` are read along rows | [`scalar.cpp`](src/kernels/scalar.cpp) |
| `tiled64` | 64 × 64 cache blocking | [`tiled.cpp`](src/kernels/tiled.cpp) |
| `avx2_ikj` | `ikj` with explicit AVX2 FMA intrinsics | [`avx2.cpp`](src/kernels/avx2.cpp) |
| `micro4x8` | 4 × 8 block of `C` held in registers | [`avx2.cpp`](src/kernels/avx2.cpp) |
| `packed4x8` | Pack tiles of `A` and `B` so the micro-kernel reads contiguous memory | [`packed.cpp`](src/kernels/packed.cpp) |
| `packed4x8_prefetch` | Add software prefetch (it hurts; see the guide) | [`packed.cpp`](src/kernels/packed.cpp) |
| `packed4x8_omp` | Parallelize row strips with OpenMP | [`packed.cpp`](src/kernels/packed.cpp) |
| `packed4x8_prefetch_omp` | Both of the above | [`packed.cpp`](src/kernels/packed.cpp) |

[**The optimization guide**](docs/optimization-guide.md) explains each step from first
principles: FLOP counting, the cache hierarchy, arithmetic intensity, register blocking,
packing, why prefetching made things worse, and where the remaining gap to BLAS comes from.

## Build and run

Requirements: a C++17 compiler (GCC 9+ or Clang 10+; MinGW-w64 on Windows), CMake 3.16+, OpenMP,
and an x86-64 CPU with AVX2 and FMA.

```bash
cmake -S . -B build                 # Release, tuned with -march=native
cmake --build build -j
ctest --test-dir build              # correctness tests
./build/gemm_bench                  # benchmark N = 256 and N = 2048
```

On Windows with MinGW-w64, add `-G "MinGW Makefiles"` to the first command.

`gemm_bench` options:

```text
--sizes 256,2048       matrix sizes
--kernels a,b          subset of kernels (see --list); naive/register run only for N <= 512 by default
--warmup W --reps R    untimed and timed runs per kernel (default 2 and 5)
--threads T            OpenMP threads
--cooldown-ms MS       pause between timed runs to limit thermal drift
--csv PATH             write every sample, with CPU, compiler, and flags in the header
```

Kernels are interleaved within each repetition, so slow drift in clocks or temperature spreads
across all kernels instead of favouring whichever ran first. Every timed run is spot-checked
against a double-precision dot product; `gemm_bench` exits non-zero if any check fails.

To regenerate the charts from a CSV: `python scripts/plot.py results/<file>.csv`.

## Testing

`ctest` runs two tests:

- **correctness**: every kernel at 15 sizes from 1 to 200, including one short of, exactly at,
  and one past each tile edge (4 rows, 8 columns, 64-wide tiles), and sizes spanning several
  64-deep k-tiles. Results are compared with a double-precision reference using the standard
  floating-point dot-product error bound. `C` is pre-filled with NaN so a kernel that reads `C`
  before writing it fails.
- **bench_smoke**: one short benchmark run, which must pass its own spot checks.

CI runs both with GCC and Clang, plus a Debug build under AddressSanitizer and
UndefinedBehaviorSanitizer.

## Repository layout

```text
include/gemm/kernels.hpp     kernel declarations and the kernel registry
src/kernels/                 the kernels, grouped by technique
src/registry.cpp             kernel names in guide order
tests/test_correctness.cpp   correctness tests against a double-precision reference
bench/gemm_bench.cpp         benchmark driver
scripts/plot.py              CSV to charts
results/                     raw benchmark samples with machine metadata
docs/                        optimization guide and charts
```

## Limitations and next steps

- Square matrices only, row-major, single precision.
- The kernels are teaching steps, not a BLAS replacement. The largest remaining gains, in the
  order they would likely pay off: a 6 × 16 micro-kernel (12 accumulators, enough independent
  FMAs to keep both FMA units busy), BLIS-style three-level blocking sized to L1/L2/L3, packing
  each `B` block once and sharing it across threads, allocating aligned packing buffers once
  instead of inside the loop, and P-core/E-core-aware scheduling.
- Numbers come from one laptop. Laptop power limits change sustained clocks; compare runs only
  when they share a machine, power state, and build.

## License

MIT
