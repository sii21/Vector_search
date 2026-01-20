"""Filtered search with HNSW (HW4 task 2.2)"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_tests(data_dir: Path) -> List[dict]:
    """Load test queries from tests.jsonl"""
    tests_path = data_dir / "tests.jsonl"

    if not tests_path.exists():
        msg = f"Tests file not found: {tests_path}"
        raise FileNotFoundError(msg)

    tests = []
    with tests_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))

    return tests


def build_filter(conditions: dict) -> models.Filter | None:
    """Build Qdrant filter from conditions dictionary"""
    if not conditions:
        return None

    must_conditions = []
    should_conditions = []
    must_not_conditions = []

    if "must" in conditions:
        for cond in conditions["must"]:
            must_conditions.append(parse_condition(cond))

    if "should" in conditions:
        for cond in conditions["should"]:
            should_conditions.append(parse_condition(cond))

    if "must_not" in conditions:
        for cond in conditions["must_not"]:
            must_not_conditions.append(parse_condition(cond))

    return models.Filter(
        must=must_conditions if must_conditions else None,
        should=should_conditions if should_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def parse_condition(cond: dict) -> models.Condition:
    """Parse a single condition into Qdrant Condition"""
    if "must" in cond or "should" in cond or "must_not" in cond:
        return build_filter(cond)

    if "range" in cond:
        range_cond = cond["range"]
        key = range_cond["key"]
        range_params = {}

        if "gte" in range_cond:
            range_params["gte"] = range_cond["gte"]
        if "lte" in range_cond:
            range_params["lte"] = range_cond["lte"]
        if "gt" in range_cond:
            range_params["gt"] = range_cond["gt"]
        if "lt" in range_cond:
            range_params["lt"] = range_cond["lt"]

        return models.FieldCondition(key=key, range=models.Range(**range_params))

    if "match" in cond:
        match_cond = cond["match"]
        return models.FieldCondition(
            key=match_cond["key"],
            match=models.MatchValue(value=match_cond["value"]),
        )

    msg = f"Unknown condition type: {cond}"
    raise ValueError(msg)


def calculate_precision_at_k(
    predicted_ids: List[int], expected_ids: List[int], k: int
) -> float:
    """Calculate Precision@k metric"""
    if k == 0:
        return 0.0

    predicted_k = set(predicted_ids[:k])
    expected_k = set(expected_ids[:k])

    if not expected_k:
        return 0.0

    intersection = predicted_k & expected_k
    return len(intersection) / k


def run_filtered_search_hnsw(
    client: QdrantClient,
    collection_name: str,
    tests: List[dict],
    limit: int = 10,
    batch_size: int = 100,
) -> dict:
    """Run filtered search experiments with HNSW using batch API"""
    logger.info(f"Running filtered search with HNSW on {len(tests)} queries...")

    precisions_at_1 = []
    precisions_at_3 = []
    precisions_at_5 = []
    precisions_at_10 = []

    start_time = time.time()

    for batch_start in range(0, len(tests), batch_size):
        batch_end = min(batch_start + batch_size, len(tests))
        batch_requests = []

        for idx in range(batch_start, batch_end):
            test = tests[idx]
            query_vector = test["query"]
            conditions = test.get("conditions", {})

            query_filter = build_filter(conditions)

            query_request = models.QueryRequest(
                query=query_vector,
                limit=limit,
                filter=query_filter,
            )
            batch_requests.append(query_request)

        batch_results = client.query_batch_points(
            collection_name=collection_name,
            requests=batch_requests,
        )

        for local_idx, query_response in enumerate(batch_results):
            idx = batch_start + local_idx
            test = tests[idx]
            expected_ids = test["closest_ids"]

            predicted_ids = [point.id for point in query_response.points]

            precisions_at_1.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 1)
            )
            precisions_at_3.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 3)
            )
            precisions_at_5.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 5)
            )
            precisions_at_10.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 10)
            )

        if batch_end % 1000 == 0 or batch_end == len(tests):
            logger.info(f"  Processed {batch_end}/{len(tests)} queries")

    end_time = time.time()
    total_time = end_time - start_time

    metrics = {
        "Precision@1": np.mean(precisions_at_1),
        "Precision@3": np.mean(precisions_at_3),
        "Precision@5": np.mean(precisions_at_5),
        "Precision@10": np.mean(precisions_at_10),
        "total_search_time": total_time,
        "QPS": len(tests) / total_time,
    }

    logger.info("HNSW search completed")
    logger.info(f"  Precision@1: {metrics['Precision@1']:.5f}")
    logger.info(f"  Precision@3: {metrics['Precision@3']:.5f}")
    logger.info(f"  Precision@5: {metrics['Precision@5']:.5f}")
    logger.info(f"  Precision@10: {metrics['Precision@10']:.5f}")
    logger.info(f"  Total time: {metrics['total_search_time']:.2f}s")
    logger.info(f"  QPS: {metrics['QPS']:.2f}")

    return metrics


def run_filtered_search_acorn(
    client: QdrantClient,
    collection_name: str,
    tests: List[dict],
    limit: int = 10,
    batch_size: int = 100,
) -> dict:
    """Run filtered search experiments with ACORN using batch API"""
    logger.info(f"Running filtered search with ACORN on {len(tests)} queries...")

    precisions_at_1 = []
    precisions_at_3 = []
    precisions_at_5 = []
    precisions_at_10 = []

    start_time = time.time()

    for batch_start in range(0, len(tests), batch_size):
        batch_end = min(batch_start + batch_size, len(tests))
        batch_requests = []

        for idx in range(batch_start, batch_end):
            test = tests[idx]
            query_vector = test["query"]
            conditions = test.get("conditions", {})

            query_filter = build_filter(conditions)

            query_request = models.QueryRequest(
                query=query_vector,
                limit=limit,
                filter=query_filter,
                params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
            )
            batch_requests.append(query_request)

        batch_results = client.query_batch_points(
            collection_name=collection_name,
            requests=batch_requests,
        )

        for local_idx, query_response in enumerate(batch_results):
            idx = batch_start + local_idx
            test = tests[idx]
            expected_ids = test["closest_ids"]

            predicted_ids = [point.id for point in query_response.points]

            precisions_at_1.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 1)
            )
            precisions_at_3.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 3)
            )
            precisions_at_5.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 5)
            )
            precisions_at_10.append(
                calculate_precision_at_k(predicted_ids, expected_ids, 10)
            )

        if batch_end % 1000 == 0 or batch_end == len(tests):
            logger.info(f"  Processed {batch_end}/{len(tests)} queries")

    end_time = time.time()
    total_time = end_time - start_time

    metrics = {
        "Precision@1": np.mean(precisions_at_1),
        "Precision@3": np.mean(precisions_at_3),
        "Precision@5": np.mean(precisions_at_5),
        "Precision@10": np.mean(precisions_at_10),
        "total_search_time": total_time,
        "QPS": len(tests) / total_time,
    }

    logger.info("ACORN search completed")
    logger.info(f"  Precision@1: {metrics['Precision@1']:.5f}")
    logger.info(f"  Precision@3: {metrics['Precision@3']:.5f}")
    logger.info(f"  Precision@5: {metrics['Precision@5']:.5f}")
    logger.info(f"  Precision@10: {metrics['Precision@10']:.5f}")
    logger.info(f"  Total time: {metrics['total_search_time']:.2f}s")
    logger.info(f"  QPS: {metrics['QPS']:.2f}")

    return metrics


def save_results(results: dict, output_path: Path) -> None:
    """Save results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """Main entry point for filtered search experiments"""
    logger.info("Running filtered search experiments (HW4 task 2.2)")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    tests = load_tests(data_dir)
    logger.info(f"Loaded {len(tests)} test queries")

    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=300)

    collection_name = "hw4"

    hnsw_metrics = run_filtered_search_hnsw(client, collection_name, tests)
    save_results(hnsw_metrics, results_dir / "filtered_search_results.jsonl")
    logger.info("Running ACORN experiments (HW4 task 2.3)")

    acorn_metrics = run_filtered_search_acorn(client, collection_name, tests)
    save_results(acorn_metrics, results_dir / "filtered_search_acorn_results.jsonl")
    logger.info("All experiments completed")


if __name__ == "__main__":
    main()
