from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss  # type: ignore[import]
import numpy as np

from .metrics import load_ground_truth, precision_at_ks


def load_vectors(data_dir: Path) -> np.ndarray:
    """Load vectors.npy from hw2/data as float32 array"""
    vectors_path = data_dir / "vectors.npy"
    if not vectors_path.exists():
        msg = (
            f"vectors.npy not found at {vectors_path}. Run hw2/download_data.sh first."
        )
        raise FileNotFoundError(msg)
    vectors = np.load(vectors_path)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    return vectors


def build_hnsw_index(
    vectors: np.ndarray,
    m: int,
    ef_construction: int,
) -> Tuple[faiss.Index, float]:
    """Build HNSW index (IndexHNSWFlat) and measure indexing time"""
    n, d = vectors.shape
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efConstruction = ef_construction

    start = time.perf_counter()
    index.add(vectors)
    indexing_time = time.perf_counter() - start

    if index.ntotal != n:
        msg = f"Index size mismatch: expected {n}, got {index.ntotal}"
        raise RuntimeError(msg)

    return index, indexing_time


def search_with_index(
    index: faiss.Index,
    vectors: np.ndarray,
    ef_search: int,
    k: int = 10,
) -> Tuple[Dict[int, List[int]], float]:
    """Search for k nearest neighbours for each vector and measure total search time"""
    n, _ = vectors.shape
    index.hnsw.efSearch = ef_search

    start = time.perf_counter()
    _, indices = index.search(vectors, k + 1)  # k+1 to drop self
    total_search_time = time.perf_counter() - start

    preds: Dict[int, List[int]] = {}
    for i in range(n):
        neighs = [int(idx) for idx in indices[i] if idx != i]
        preds[i] = neighs[:k]

    return preds, total_search_time


def run_grid_search_hnsw(
    vectors: np.ndarray,
    ground_truth_path: Path,
    results_path: Path,
    m_values: Iterable[int],
    ef_construction_values: Iterable[int],
    ef_search_values: Iterable[int],
    k: int = 10,
) -> None:
    """Run HNSW over a grid of hyperparameters and write results to JSONL."""
    ground_truth = load_ground_truth(ground_truth_path)
    n_queries = len(ground_truth)
    if n_queries != vectors.shape[0]:
        msg = f"ground_truth size {n_queries} != number of vectors {vectors.shape[0]}"
        raise ValueError(msg)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f_out:
        for m in m_values:
            for ef_c in ef_construction_values:
                print(f"\n=== HNSW config: M={m}, efConstruction={ef_c} ===")
                index, indexing_time = build_hnsw_index(
                    vectors, m=m, ef_construction=ef_c
                )

                for ef_s in ef_search_values:
                    print(f"  efSearch={ef_s} ...")
                    preds, total_search_time = search_with_index(
                        index=index,
                        vectors=vectors,
                        ef_search=ef_s,
                        k=k,
                    )

                    metrics = precision_at_ks(
                        ground_truth=ground_truth,
                        predictions=preds,
                        ks=(1, 3, 5, 10),
                    )

                    qps = (
                        n_queries / total_search_time if total_search_time > 0 else 0.0
                    )

                    record = {
                        "M": m,
                        "efConstruction": ef_c,
                        "efSearch": ef_s,
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
    n, d = vectors.shape
    print(f"Loaded vectors: shape={vectors.shape} (n={n}, dim={d})")

    ground_truth_path = results_dir / "ground_truth.jsonl"
    results_path = results_dir / "hnsw_results.jsonl"

    m_values = [8, 16, 32, 64]
    ef_construction_values = [32, 64, 100, 128, 256]
    ef_search_values = [32, 64, 100, 128, 256]

    run_grid_search_hnsw(
        vectors=vectors,
        ground_truth_path=ground_truth_path,
        results_path=results_path,
        m_values=m_values,
        ef_construction_values=ef_construction_values,
        ef_search_values=ef_search_values,
        k=10,
    )


if __name__ == "__main__":
    main()
