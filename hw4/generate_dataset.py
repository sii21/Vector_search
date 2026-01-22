"""Generate random dataset and create collection for filter experiments (HW4 task 2.4)"""

from __future__ import annotations

import logging

import numpy as np
from qdrant_client import QdrantClient, models

from hw4.payload_generator import generate_full_payload

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def generate_random_vectors(n_points: int, dim: int = 512) -> np.ndarray:
    """Generate random normalized vectors"""
    vectors = np.random.randn(n_points, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors


def create_filter_experiments_collection(
    client: QdrantClient,
    collection_name: str = "filter_experiments",
    vector_dim: int = 512,
) -> None:
    """Create collection for filter experiments"""
    logger.info(f"Creating collection '{collection_name}'...")

    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' exists, deleting...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info(f"Collection '{collection_name}' created")


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes for filterable fields"""
    logger.info("Creating payload indexes...")

    indexes = [
        ("color", models.PayloadSchemaType.KEYWORD),
        ("category", models.PayloadSchemaType.KEYWORD),
        ("price", models.PayloadSchemaType.FLOAT),
        ("quantity", models.PayloadSchemaType.INTEGER),
        ("rating", models.PayloadSchemaType.FLOAT),
        ("in_stock", models.PayloadSchemaType.BOOL),
        ("location", models.PayloadSchemaType.GEO),
        ("tags", models.PayloadSchemaType.KEYWORD),
        ("brand", models.PayloadSchemaType.KEYWORD),
    ]

    for field_name, schema_type in indexes:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
        logger.info(f"Created index for '{field_name}' ({schema_type})")

    logger.info("All payload indexes created")


def upload_data(
    client: QdrantClient,
    collection_name: str,
    n_points: int,
    vector_dim: int,
    batch_size: int = 1000,
) -> None:
    """Generate and upload random data"""
    logger.info(f"Generating and uploading {n_points} points...")

    vectors = generate_random_vectors(n_points, vector_dim)

    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)

        points = []
        for idx in range(batch_start, batch_end):
            payload = generate_full_payload()
            point = models.PointStruct(
                id=idx,
                vector=vectors[idx].tolist(),
                payload=payload,
            )
            points.append(point)

        client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=batch_size,
        )

        if (batch_end % 10000 == 0) or (batch_end == n_points):
            logger.info(f"  Uploaded {batch_end}/{n_points} points")

    logger.info("Upload completed")


def wait_for_indexes(client: QdrantClient, collection_name: str) -> None:
    """Wait for all indexes to be built"""
    logger.info("Waiting for indexes to be built...")

    import time

    while True:
        info = client.get_collection(collection_name)
        if info.status == models.CollectionStatus.GREEN:
            logger.info("All indexes are ready (status: GREEN)")
            break
        logger.info(f"  Status: {info.status}, waiting...")
        time.sleep(2)


def main() -> None:
    """Generate dataset and create collection for filter experiments"""
    logger.info("Generating random dataset for filter experiments (HW4 task 2.4)")

    n_points = 200_000
    vector_dim = 512
    collection_name = "filter_experiments"

    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=300)

    create_filter_experiments_collection(client, collection_name, vector_dim)
    create_payload_indexes(client, collection_name)
    upload_data(client, collection_name, n_points, vector_dim)
    wait_for_indexes(client, collection_name)

    info = client.get_collection(collection_name)
    logger.info(f"Collection ready: {info.points_count} points")

    logger.info("Dataset generation completed")


if __name__ == "__main__":
    main()
