#!/usr/bin/env python3
"""Plot median GFLOPS per kernel from gemm_bench CSVs, one chart per matrix size."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_samples(paths):
    samples = defaultdict(list)  # (n, kernel) -> [gflops]
    order = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows = csv.DictReader(line for line in handle if not line.startswith("#"))
            for row in rows:
                key = (int(row["n"]), row["kernel"])
                if key not in samples:
                    order.append(key)
                samples[key].append(float(row["gflops"]))
    return samples, order


def plot_size(n, kernels, samples, output):
    medians = [statistics.median(samples[(n, k)]) for k in kernels]
    lows = [m - min(samples[(n, k)]) for m, k in zip(medians, kernels)]
    highs = [max(samples[(n, k)]) - m for m, k in zip(medians, kernels)]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(kernels) + 1.2))
    bars = ax.barh(kernels, medians, xerr=[lows, highs], color="#3b6ea5", ecolor="#555555", capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("GFLOPS (median; whiskers show min-max of samples)")
    ax.set_title(f"SGEMM, N = {n}")
    ax.grid(axis="x", alpha=0.3)
    ax.bar_label(bars, labels=[f"{m:.1f}" for m in medians], padding=4, fontsize=8)
    ax.set_xlim(0, max(m + h for m, h in zip(medians, highs)) * 1.15)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    args = parser.parse_args()

    samples, order = read_samples(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in sorted({n for n, _ in order}):
        kernels = [k for size, k in order if size == n]
        output = args.out_dir / f"gflops-n{n}.png"
        plot_size(n, kernels, samples, output)
        print(output)


if __name__ == "__main__":
    main()
