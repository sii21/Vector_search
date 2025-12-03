"""
Checks to validate distance implementations match between modules
"""

from __future__ import annotations

from typing import Any

import numpy as np

from search_hw.step_1 import distances_python as dp, distances_numpy as dn


def _random_data(n: int, d: int) -> Any:
    """Generate a random dataset and a single query for tests"""
    rng = np.random.default_rng(2025)
    data = rng.random((n, d))
    return data, data[0]


def check_all_close(n: int = 100, d: int = 16) -> bool:
    """Compare NumPy and Python distances; return True if close"""
    data, query = _random_data(n, d)
    data_list = data.tolist()
    query_list = query.tolist()

    a = dn.euclidean_distance(query, data)
    b = dp.euclidean_distance(query_list, data_list)

    return np.allclose(a, b)


if __name__ == "__main__":
    ok = check_all_close()
    print("All close:", ok)
