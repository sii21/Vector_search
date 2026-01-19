"""Upload data to Qdrant collections using different methods"""

from __future__ import annotations

import json
import logging
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
        msg = f"Vectors file not found: {vectors_path}"
        raise FileNotFoundError(msg)

    if not payloads_path.exists():
        msg = f"Payloads file not found: {payloads_path}"
        raise FileNotFoundError(msg)

    vectors = np.load(vectors_path)

    payloads = []
    with payloads_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payloads.append(json.loads(line))

    return vectors, payloads


def upload_with_upload_points(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    payloads: List[dict],
    batch_size: int = 1000,
    parallel: int = 1,
) -> None:
    """Upload data using upload_points method

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vectors: Array of vectors to upload
        payloads: List of payload dictionaries
        batch_size: Number of points per batch
        parallel: Number of parallel workers
    """
    logger.info(
        f"Uploading {len(vectors)} points to '{collection_name}' using upload_points (batch_size={batch_size}, parallel={parallel})..."
    )

    points = [
        models.PointStruct(
            id=idx,
            vector=vectors[idx].tolist(),
            payload=payloads[idx],
        )
        for idx in range(len(vectors))
    ]

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=batch_size,
        parallel=parallel,
    )

    logger.info(f"Uploaded all {len(vectors)} points")


def upload_with_upload_collection(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    payloads: List[dict],
    vector_name: str,
    batch_size: int = 1000,
    parallel: int = 1,
) -> None:
    """Upload data using upload_points method for named vectors

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vectors: Array of vectors to upload
        payloads: List of payload dictionaries
        vector_name: Name of the vector field
        batch_size: Number of points per batch
        parallel: Number of parallel workers
    """
    logger.info(
        f"Uploading {len(vectors)} points to '{collection_name}' ({vector_name}) using upload_points (batch_size={batch_size}, parallel={parallel})..."
    )

    points = [
        models.PointStruct(
            id=idx,
            vector={vector_name: vectors[idx].tolist()},
            payload=payloads[idx],
        )
        for idx in range(len(vectors))
    ]

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=batch_size,
        parallel=parallel,
    )

    logger.info(f"Uploaded all {len(vectors)} points for {vector_name}")


def main() -> None:
    """Main entry point for data upload experiments"""
    logger.info("Starting data upload experiments")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    vectors, payloads = load_data(data_dir)
    logger.info(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")

    client = QdrantClient("http://localhost:6333")
    logger.info("Connected to Qdrant at http://localhost:6333")

    if not client.collection_exists("single_unnamed"):
        raise RuntimeError(
            "Collection 'single_unnamed' does not exist. Run `uv run python -m hw3.qdrant_collections` first."
        )

    upload_with_upload_points(
        client, "single_unnamed", vectors, payloads, batch_size=1000, parallel=4
    )

    if not client.collection_exists("multiple_named"):
        raise RuntimeError(
            "Collection 'multiple_named' does not exist. Run `uv run python -m hw3.qdrant_collections` first."
        )

    upload_with_upload_collection(
        client,
        "multiple_named",
        vectors,
        payloads,
        "clip_default",
        batch_size=1000,
        parallel=4,
    )
    upload_with_upload_collection(
        client,
        "multiple_named",
        vectors,
        payloads,
        "clip_tuned",
        batch_size=100,
        parallel=4,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Upload summary:")
    for collection_name in ["single_unnamed", "multiple_named"]:
        info = client.get_collection(collection_name)
        logger.info(
            f"  {collection_name}: {info.points_count} points, status={info.status}"
        )


if __name__ == "__main__":
    main()
