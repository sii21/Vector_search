from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import faiss  # type: ignore[import]
import numpy as np
from tqdm import tqdm


def load_vectors(data_dir: Path) -> np.ndarray:
    """Load vectors.npy from hw2/data as float32"""
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


def iter_batches(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) ranges for batching."""
    start = 0
    while start < n_items:
        end = min(start + batch_size, n_items)
        yield start, end
        start = end


def compute_ground_truth(
    vectors: np.ndarray,
    k: int,
    batch_size: int = 512,
) -> Dict[int, List[int]]:
    """Compute k exact nearest neighbours for each vector via brute-force L2"""
    n, d = vectors.shape
    index = faiss.IndexFlatL2(d)  # exact L2
    index.add(vectors)
    if index.ntotal != n:
        msg = f"Index size mismatch: expected {n}, got {index.ntotal}"
        raise RuntimeError(msg)

    ground_truth: Dict[int, List[int]] = {}

    for start, end in tqdm(
        iter_batches(n, batch_size),
        total=(n + batch_size - 1) // batch_size,
        desc="computing ground truth",
    ):
        xq = vectors[start:end]
        # k+1 to remove self match
        _, indices = index.search(xq, k + 1)
        for i in range(end - start):
            qid = start + i
            neighs = [int(idx) for idx in indices[i] if idx != qid]
            ground_truth[qid] = neighs[:k]

    return ground_truth


def save_ground_truth_jsonl(
    ground_truth: Dict[int, List[int]], output_path: Path
) -> None:
    """Save ground truth mapping {query_id: [nn_ids]} to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for qid, nns in ground_truth.items():
            record = {str(qid): nns}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "hw2" / "data"
    results_dir = repo_root / "hw2" / "results"
    output_path = results_dir / "ground_truth.jsonl"

    print(f"Loading vectors from: {data_dir}")
    vectors = load_vectors(data_dir)
    n, d = vectors.shape
    print(f"Loaded vectors: shape={vectors.shape} (n={n}, dim={d})")

    k = 10
    batch_size = 512

    start_time = time.perf_counter()
    gt = compute_ground_truth(vectors=vectors, k=k, batch_size=batch_size)
    elapsed = time.perf_counter() - start_time
    print(f"Computed ground truth for {n} vectors in {elapsed:.2f} s")

    print(f"Saving ground truth to: {output_path}")
    save_ground_truth_jsonl(gt, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
