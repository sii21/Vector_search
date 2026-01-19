"""Perform search experiments and evaluate precision"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_ground_truth(ground_truth_path: Path) -> Dict[int, List[int]]:
    """Load ground truth from jsonl file

    Args:
        ground_truth_path: Path to ground truth file

    Returns:
        Dictionary mapping query index to list of true neighbor ids
    """
    ground_truth = {}
    with ground_truth_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                for key, value in record.items():
                    ground_truth[int(key)] = value
    return ground_truth


def calculate_precision_at_k(
    predicted: List[int], ground_truth: List[int], k: int
) -> float:
    """Calculate precision@k for a single query

    Args:
        predicted: List of predicted neighbor ids
        ground_truth: List of true neighbor ids
        k: Number of top results to consider

    Returns:
        Precision@k score
    """
    predicted_at_k = set(predicted[:k])
    ground_truth_at_k = set(ground_truth[:k])

    if len(predicted_at_k) == 0:
        return 0.0

    return len(predicted_at_k & ground_truth_at_k) / k


def search_collection(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    vector_name: str | None = None,
    ef: int | None = None,
    top_k: int = 10,
    batch_size: int = 100,
) -> Dict:
    """Search collection and calculate precision metrics using batched requests

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to search
        vectors: Array of query vectors
        ground_truth: Dictionary mapping query index to true neighbor ids
        vector_name: Name of the vector field (for named vectors)
        ef: ef_search parameter for HNSW
        top_k: Number of nearest neighbors to retrieve
        batch_size: Number of queries to process in parallel batch

    Returns:
        Dictionary with precision metrics and performance stats
    """
    vector_label = f"{collection_name}{f'/{vector_name}' if vector_name else ''}"
    logger.info(
        f"Searching '{vector_label}' with ef={ef if ef else 'default'} (top_k={top_k}, batch_size={batch_size})"
    )

    info = client.get_collection(collection_name)
    if info.status != models.CollectionStatus.GREEN:
        logger.warning(
            f"Collection status is {info.status}, waiting for index to become GREEN"
        )
        while True:
            wait_start = time.time()
            info = client.get_collection(collection_name)
            if info.status == models.CollectionStatus.GREEN:
                break
            time.sleep(1)
        wait_time = time.time() - wait_start
        logger.info(f"Index is ready after waiting {wait_time:.1f}s")

    if ef is not None:
        search_params = models.SearchParams(hnsw_ef=ef, exact=False)
    else:
        search_params = models.SearchParams(exact=False)

    start_time = time.time()

    # Process in batches for speed
    precision_1_scores = []
    precision_3_scores = []
    precision_5_scores = []
    precision_10_scores = []

    for batch_start in range(0, len(vectors), batch_size):
        batch_end = min(batch_start + batch_size, len(vectors))
        batch_requests = []

        for idx in range(batch_start, batch_end):
            query_vector = vectors[idx].tolist()

            if vector_name is None:
                query_request = models.QueryRequest(
                    query=query_vector,
                    limit=top_k,
                    params=search_params,
                )
            else:
                query_request = models.QueryRequest(
                    query=query_vector,
                    using=vector_name,
                    limit=top_k,
                    params=search_params,
                )
            batch_requests.append(query_request)

        # Send batch request
        batch_results = client.query_batch_points(
            collection_name=collection_name,
            requests=batch_requests,
        )

        # Process batch results
        for local_idx, query_response in enumerate(batch_results):
            idx = batch_start + local_idx
            predicted_ids = [point.id for point in query_response.points]
            true_ids = ground_truth[idx]

            precision_1_scores.append(
                calculate_precision_at_k(predicted_ids, true_ids, 1)
            )
            precision_3_scores.append(
                calculate_precision_at_k(predicted_ids, true_ids, 3)
            )
            precision_5_scores.append(
                calculate_precision_at_k(predicted_ids, true_ids, 5)
            )
            precision_10_scores.append(
                calculate_precision_at_k(predicted_ids, true_ids, 10)
            )

        if batch_end % 10000 < batch_size or batch_end == len(vectors):
            logger.info(
                f"  Progress: searched {batch_end}/{len(vectors)} queries for {vector_label}"
            )

    end_time = time.time()
    total_time = end_time - start_time
    qps = len(vectors) / total_time

    config_info = client.get_collection(collection_name)

    if vector_name and hasattr(config_info.config.params, "vectors"):
        vector_config = config_info.config.params.vectors[vector_name]
        hnsw_config = vector_config.hnsw_config
    else:
        hnsw_config = config_info.config.params.vectors.hnsw_config

    m = hnsw_config.m if hnsw_config and hnsw_config.m else 16
    ef_construct = (
        hnsw_config.ef_construct if hnsw_config and hnsw_config.ef_construct else 100
    )

    results = {
        "m": m,
        "ef_construct": ef_construct,
        "Precision@1": np.mean(precision_1_scores),
        "Precision@3": np.mean(precision_3_scores),
        "Precision@5": np.mean(precision_5_scores),
        "Precision@10": np.mean(precision_10_scores),
        "total_search_time": total_time,
        "QPS": qps,
    }

    if ef is not None:
        results["ef_search"] = ef

    logger.info(
        f"  Finished '{vector_label}': Precision@1={results['Precision@1']:.5f}, "
        f"Precision@10={results['Precision@10']:.5f}, QPS={results['QPS']:.2f}, "
        f"total_time={results['total_search_time']:.1f}s"
    )

    return results


def save_results(results: List[Dict], output_path: Path) -> None:
    """Save search results to jsonl file

    Args:
        results: List of result dictionaries
        output_path: Path to save results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """Main entry point for search experiments"""
    logger.info("Starting search experiments")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    vectors = np.load(data_dir / "vectors.npy")
    ground_truth = load_ground_truth(base_dir / "ground_truth.jsonl")

    logger.info(f"Loaded {len(vectors)} vectors and ground truth")

    client = QdrantClient("http://localhost:6333")
    logger.info("Connected to Qdrant at http://localhost:6333")

    if not client.collection_exists("single_unnamed"):
        raise RuntimeError(
            "Collection 'single_unnamed' does not exist. Run `uv run python -m hw3.qdrant_collections` and `uv run python -m hw3.upload_data` first."
        )

    if not client.collection_exists("multiple_named"):
        raise RuntimeError(
            "Collection 'multiple_named' does not exist. Run `uv run python -m hw3.qdrant_collections` and `uv run python -m hw3.upload_data` first."
        )

    results_single = search_collection(client, "single_unnamed", vectors, ground_truth)
    save_results([results_single], base_dir / "single_unnamed_results.jsonl")

    results_clip_default = search_collection(
        client, "multiple_named", vectors, ground_truth, vector_name="clip_default"
    )
    save_results([results_clip_default], base_dir / "clip_default_results.jsonl")

    results_clip_tuned = search_collection(
        client, "multiple_named", vectors, ground_truth, vector_name="clip_tuned"
    )
    save_results([results_clip_tuned], base_dir / "clip_tuned_results.jsonl")

    logger.info("\n" + "=" * 40)
    logger.info("Running search with ef=50...")
    results_ef50 = search_collection(
        client, "multiple_named", vectors, ground_truth, vector_name="clip_tuned", ef=50
    )
    save_results([results_ef50], base_dir / "clip_tuned_ef_search_50_results.jsonl")

    logger.info("All search experiments completed!")


if __name__ == "__main__":
    main()
