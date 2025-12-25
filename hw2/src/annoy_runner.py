from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from annoy import AnnoyIndex  # type: ignore[import]

from .metrics import load_ground_truth, precision_at_ks


def load_vectors(data_dir: Path) -> np.ndarray:
    vectors_path = data_dir / "vectors.npy"
    if not vectors_path.exists():
        msg = f"vectors.npy not found at {vectors_path}"
        raise FileNotFoundError(msg)
    vectors = np.load(vectors_path)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    return vectors


def build_annoy_index(
    vectors: np.ndarray,
    n_trees: int,
    metric: str = "euclidean",
) -> tuple[AnnoyIndex, float]:
    n, d = vectors.shape
    index = AnnoyIndex(d, metric)
    for i in range(n):
        index.add_item(i, vectors[i])

    start = time.perf_counter()
    index.build(n_trees)
    indexing_time = time.perf_counter() - start
    return index, indexing_time


def search_annoy(
    index: AnnoyIndex,
    n_items: int,
    k: int,
    search_k: int,
) -> tuple[Dict[int, List[int]], float]:
    preds: Dict[int, List[int]] = {}

    start = time.perf_counter()
    for i in range(n_items):
        nn = index.get_nns_by_item(i, k + 1, search_k=search_k, include_distances=False)
        nn = [idx for idx in nn if idx != i]
        preds[i] = nn[:k]
    total_search_time = time.perf_counter() - start

    return preds, total_search_time


def run_grid_annoy(
    vectors: np.ndarray,
    ground_truth_path: Path,
    results_path: Path,
    n_trees_list: Iterable[int],
    search_k_list: Iterable[int],
    k: int = 10,
) -> None:
    ground_truth = load_ground_truth(ground_truth_path)
    n_queries = len(ground_truth)
    if n_queries != vectors.shape[0]:
        msg = f"ground_truth size {n_queries} != number of vectors {vectors.shape[0]}"
        raise ValueError(msg)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f_out:
        for n_trees in n_trees_list:
            print(f"\n=== ANNOY: n_trees={n_trees} ===")
            index, indexing_time = build_annoy_index(
                vectors, n_trees=n_trees, metric="euclidean"
            )

            for search_k in search_k_list:
                print(f"  search_k={search_k} ...")
                preds, total_search_time = search_annoy(
                    index=index,
                    n_items=vectors.shape[0],
                    k=k,
                    search_k=search_k,
                )

                metrics = precision_at_ks(
                    ground_truth=ground_truth,
                    predictions=preds,
                    ks=(1, 3, 5, 10),
                )

                qps = n_queries / total_search_time if total_search_time > 0 else 0.0

                record = {
                    "n_trees": n_trees,
                    "search_k": search_k,
                    "Precision@1": metrics[1],
                    "Precision@3": metrics[3],
                    "Precision@5": metrics[5],
                    "Precision@10": metrics[10],
                    "indexing_time": indexing_time,
                    "total_search_time": total_search_time,
                    "QPS": qps,
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "hw2" / "data"
    results_dir = repo_root / "hw2" / "results"

    vectors = load_vectors(data_dir)
    ground_truth_path = results_dir / "ground_truth.jsonl"
    results_path = results_dir / "annoy_results.jsonl"

    n_trees_list = [10, 25, 50, 100, 200]
    search_k_list = [100, 500, 1000, 5000]

    run_grid_annoy(
        vectors=vectors,
        ground_truth_path=ground_truth_path,
        results_path=results_path,
        n_trees_list=n_trees_list,
        search_k_list=search_k_list,
        k=10,
    )


if __name__ == "__main__":
    main()
