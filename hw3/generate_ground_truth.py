"""Generate ground truth for nearest neighbor search using Qdrant brute-force search"""

from __future__ import annotations

import argparse
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
# Silence verbose HTTP logs from qdrant-client (PUT/POST spam)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_data(data_dir: Path) -> tuple[np.ndarray, List[dict]]:
    """Load vectors and payloads from data directory

    Args:
        data_dir: Path to the data directory

    Returns:
        Tuple of (vectors array, list of payloads)
    """
    vectors_path = data_dir / "vectors.npy"
    payloads_path = data_dir / "payloads.jsonl"

    if not vectors_path.exists():
        msg = f"Vectors file not found: {vectors_path}. Run download_data.sh first"
        raise FileNotFoundError(msg)

    if not payloads_path.exists():
        msg = f"Payloads file not found: {payloads_path}. Run download_data.sh first"
        raise FileNotFoundError(msg)

    vectors = np.load(vectors_path)
    logger.info(f"Loaded vectors with shape: {vectors.shape}")

    payloads = []
    with payloads_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payloads.append(json.loads(line))

    logger.info(f"Loaded {len(payloads)} payloads")

    return vectors, payloads


def create_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    """Create Qdrant collection for ground truth generation

    Args:
        client: Qdrant client instance
        collection_name: Name of collection to create
        vector_size: Dimensionality of vectors
    """
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info(f"Created collection: {collection_name}")


def upload_data(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    payloads: List[dict],
    batch_size: int = 500,
    parallel: int = 4,
) -> None:
    """Upload vectors and payloads to Qdrant

    Args:
        client: Qdrant client instance
        collection_name: Name of collection
        vectors: Array of vectors
        payloads: List of payloads
        batch_size: Batch size for upload
        parallel: Number of parallel workers
    """
    points = [
        models.PointStruct(id=idx, vector=vector.tolist(), payload=payload)
        for idx, (vector, payload) in enumerate(zip(vectors, payloads))
    ]

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=batch_size,
        parallel=parallel,
    )
    logger.info(
        f"Uploaded {len(points)} points (batch_size={batch_size}, parallel={parallel})"
    )


def generate_ground_truth(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    output_path: Path,
    top_k: int = 10,
    sample_size: int | None = None,
) -> None:
    """Generate ground truth using Qdrant brute-force search

    Args:
        client: Qdrant client instance
        collection_name: Name of collection to search
        vectors: Array of all vectors
        output_path: Path to save ground truth results
        top_k: Number of nearest neighbors to find
        sample_size: Number of queries to process (None = all)
    """
    num_queries = sample_size if sample_size else len(vectors)
    logger.info(f"Generating ground truth for {num_queries} queries...")

    start_time = time.time()
    all_results = []

    for idx in range(num_queries):
        query_vector = vectors[idx].tolist()

        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k + 1,
            search_params=models.SearchParams(exact=True),
        )

        neighbor_ids = [int(hit.id) for hit in results.points if hit.id != idx][:top_k]
        all_results.append({str(idx): neighbor_ids})

        if (idx + 1) % 1000 == 0 or idx == num_queries - 1:
            elapsed = time.time() - start_time
            qps = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Processed {idx + 1}/{num_queries} queries ({qps:.1f} QPS, {elapsed:.1f}s)"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_results:
            f.write(json.dumps(record) + "\n")

    total_time = time.time() - start_time
    logger.info(f"Ground truth saved to {output_path}")
    logger.info(f"Total time: {total_time:.1f}s ({num_queries/total_time:.1f} QPS)")


def main() -> None:
    """Main entry point for ground truth generation"""
    parser = argparse.ArgumentParser(
        description="Generate ground truth for nearest neighbor search using NumPy"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of queries to process (default: all vectors)",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant URL (default: http://localhost:6333, use :memory: for in-memory)",
    )
    parser.add_argument(
        "--reuse-collection",
        action="store_true",
        help="Skip collection recreation/upload if data already exists",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for search (default: 4, max recommended: 8)",
    )
    args = parser.parse_args()

    logger.info("Starting ground truth generation (Qdrant brute-force)")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_path = data_dir / "ground_truth.jsonl"

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Qdrant URL: {args.qdrant_url}")
    if args.sample_size:
        logger.info(f"Sample size: {args.sample_size} queries")
    else:
        logger.info("Sample size: ALL queries")

    vectors, payloads = load_data(data_dir)

    client = QdrantClient(args.qdrant_url)
    logger.info(f"Connected to Qdrant at {args.qdrant_url}")

    collection_name = "ground_truth_collection"
    if args.reuse_collection:
        logger.info("Skipping collection recreation and upload (--reuse-collection)")
    else:
        create_collection(client, collection_name, vectors.shape[1])
        logger.info("Uploading data to Qdrant...")
        upload_data(
            client, collection_name, vectors, payloads, batch_size=500, parallel=4
        )
        logger.info("Data uploaded successfully")

    info = client.get_collection(collection_name)
    logger.info(f"Collection status: {info.status}")

    generate_ground_truth(
        client,
        collection_name,
        vectors,
        output_path,
        top_k=args.top_k,
        sample_size=args.sample_size,
    )

    logger.info("Ground truth generation complete!")


if __name__ == "__main__":
    main()
