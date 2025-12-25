from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss  # type: ignore[import]
import numpy as np

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


def build_ivfpq_index(
    vectors: np.ndarray,
    nlist: int,
    m: int,
    nbits: int,
) -> Tuple[faiss.IndexIVFPQ, float]:
    """Build IVFPQ index and return (index, indexing_time)"""
    n, d = vectors.shape

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # training
    start = time.perf_counter()
    index.train(vectors)
    index.add(vectors)
    indexing_time = time.perf_counter() - start

    if index.ntotal != n:
        msg = f"Index size mismatch: expected {n}, got {index.ntotal}"
        raise RuntimeError(msg)

    return index, indexing_time


def search_ivfpq(
    index: faiss.IndexIVFPQ,
    vectors: np.ndarray,
    nprobe: int,
    k: int,
) -> Tuple[Dict[int, List[int]], float]:
    index.nprobe = nprobe
    n, _ = vectors.shape

    start = time.perf_counter()
    _, indices = index.search(vectors, k + 1)
    total_search_time = time.perf_counter() - start

    preds: Dict[int, List[int]] = {}
    for i in range(n):
        neighs = [int(idx) for idx in indices[i] if idx != i]
        preds[i] = neighs[:k]

    return preds, total_search_time


def run_grid_ivfpq(
    vectors: np.ndarray,
    ground_truth_path: Path,
    results_path: Path,
    nlist_values: Iterable[int],
    m_values: Iterable[int],
    nbits_values: Iterable[int],
    nprobe_values: Iterable[int],
    k: int = 10,
) -> None:
    ground_truth = load_ground_truth(ground_truth_path)
    n_queries = len(ground_truth)
    if n_queries != vectors.shape[0]:
        msg = f"ground_truth size {n_queries} != number of vectors {vectors.shape[0]}"
        raise ValueError(msg)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f_out:
        for nlist in nlist_values:
            for m in m_values:
                for nbits in nbits_values:
                    print(f"\n=== IVFPQ: nlist={nlist}, m={m}, nbits={nbits} ===")
                    index, indexing_time = build_ivfpq_index(
                        vectors=vectors,
                        nlist=nlist,
                        m=m,
                        nbits=nbits,
                    )

                    for nprobe in nprobe_values:
                        print(f"  nprobe={nprobe} ...")
                        preds, total_search_time = search_ivfpq(
                            index=index,
                            vectors=vectors,
                            nprobe=nprobe,
                            k=k,
                        )

                        metrics = precision_at_ks(
                            ground_truth=ground_truth,
                            predictions=preds,
                            ks=(1, 3, 5, 10),
                        )

                        qps = (
                            n_queries / total_search_time
                            if total_search_time > 0
                            else 0.0
                        )

                        record = {
                            "nlist": nlist,
                            "m": m,
                            "nbits": nbits,
                            "nprobe": nprobe,
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
    results_path = results_dir / "ivfpq_results.jsonl"

    nlist_values = [64, 128, 256, 512, 1024]
    m_values = [16, 32]
    nbits_values = [8]
    nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128]

    run_grid_ivfpq(
        vectors=vectors,
        ground_truth_path=ground_truth_path,
        results_path=results_path,
        nlist_values=nlist_values,
        m_values=m_values,
        nbits_values=nbits_values,
        nprobe_values=nprobe_values,
        k=10,
    )


if __name__ == "__main__":
    main()
