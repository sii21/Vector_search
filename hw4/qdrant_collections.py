"""Create and configure Qdrant collections"""

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
    """Load vectors and payloads for upload"""
    vectors_path = data_dir / "vectors.npy"
    payloads_path = data_dir / "payloads.jsonl"

    if not vectors_path.exists() or not payloads_path.exists():
        msg = f"Dataset is missing. Expected files in {data_dir}"
        raise FileNotFoundError(msg)

    vectors = np.load(vectors_path)

    payloads: List[dict] = []
    with payloads_path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                payloads.append(json.loads(line))

    return vectors, payloads


def create_single_unnamed_collection(
    client: QdrantClient, collection_name: str = "hw4"
) -> None:
    """Create a collection with unnamed vectors and default HNSW parameters

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
    """
    logger.info(f"Creating collection '{collection_name}'...")

    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' exists, deleting...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info(
        f"Created collection '{collection_name}' with default HNSW parameters (m=16, ef_construct=100)"
    )


def upload_with_upload_points(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    payloads: List[dict],
    batch_size: int = 1_000,
) -> None:
    """Upload data using the upload_points helper from HW3"""
    total = len(vectors)
    logger.info(
        f"Uploading {total} points to '{collection_name}' via upload_points (batch={batch_size})"
    )

    points = [
        models.PointStruct(id=idx, vector=vectors[idx].tolist(), payload=payloads[idx])
        for idx in range(total)
    ]

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=batch_size,
    )
    logger.info("Upload finished")


def main() -> None:
    """Create hw4 collection and upload data (HW4 task 2.1)"""
    logger.info("Starting collection creation test")

    client = QdrantClient("http://localhost:6333")
    logger.info("Connected to Qdrant")

    create_single_unnamed_collection(client, collection_name="hw4")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    vectors, payloads = load_data(data_dir)
    upload_with_upload_points(
        client,
        collection_name="hw4",
        vectors=vectors,
        payloads=payloads,
        batch_size=1_000,
    )

    collections = client.get_collections()
    logger.info(f"\nSuccessfully created {len(collections.collections)} collections:")
    for col in collections.collections:
        logger.info(f"  - {col.name}")


if __name__ == "__main__":
    main()
