"""Create and configure Qdrant collections"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def create_single_unnamed_collection(
    client: QdrantClient, collection_name: str = "single_unnamed"
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


def create_multiple_named_collection(
    client: QdrantClient, collection_name: str = "multiple_named"
) -> None:
    """Create a collection with multiple named vectors and custom HNSW parameters

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
    """
    logger.info(f"Creating collection '{collection_name}' with named vectors...")

    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' exists, deleting...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "clip_default": models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=32,
                    ef_construct=256,
                ),
            ),
            "clip_tuned": models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=36,
                    ef_construct=300,
                ),
            ),
        },
    )
    logger.info(f"Created collection '{collection_name}':")
    logger.info("  - clip_default: m=32, ef_construct=256")
    logger.info("  - clip_tuned: m=36, ef_construct=300")


def main() -> None:
    """Create both required collections"""
    logger.info("Starting collection creation test")

    client = QdrantClient("http://localhost:6333")
    logger.info("Connected to Qdrant at http://localhost:6333")

    create_single_unnamed_collection(client)
    create_multiple_named_collection(client)

    collections = client.get_collections()
    logger.info(f"\nSuccessfully created {len(collections.collections)} collections:")
    for col in collections.collections:
        logger.info(f"  - {col.name}")


if __name__ == "__main__":
    main()
