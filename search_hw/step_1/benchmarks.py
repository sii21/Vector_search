"""
Benchmark utilities to compare NumPy and pure-Python distance implementations
"""

from __future__ import annotations

import time
import numpy as np

from search_hw.step_1 import distances_python as dp, distances_numpy as dn


def benchmark_once(
    n_vectors: int,
    dim: int,
) -> None:
    """Run a single timing comparison for given size and dimensionality

    Args:
        n_vectors: number of vectors in the dataset
        dim: dimensionality of each vector
    """
    rng = np.random.default_rng(2025)  # reproducible RNG

    # Generate data and a single query
    data = rng.random((n_vectors, dim))
    query = rng.random(dim)

    data_list = data.tolist()
    query_list = query.tolist()

    print(f"\nN = {n_vectors}, dim = {dim}")

    t0 = time.perf_counter()
    _ = dn.euclidean_distance(query, data)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    _ = dp.euclidean_distance(query_list, data_list)
    t3 = time.perf_counter()

    print(f"numpy    : {t1 - t0:.6f} s")
    print(f"python   : {t3 - t2:.6f} s")
    if (t3 - t2) > 0:
        print(f"speedup  : {(t3 - t2) / (t1 - t0):.1f}x slower (python vs numpy)")


def main() -> None:
    """Run benchmarks for a few preset configurations"""
    configs = [
        (1_000, 16),
        (10_000, 16),
        (10_000, 128),
        (50_000, 128),
    ]
    for n, d in configs:
        benchmark_once(n, d)


if __name__ == "__main__":
    main()
