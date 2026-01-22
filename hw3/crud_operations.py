"""Demonstrate CRUD operations in Qdrant"""

from __future__ import annotations

import logging
from typing import List

from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def read_with_scroll(client: QdrantClient, collection_name: str) -> List[int]:
    """Read 10 points using scroll with specific parameters

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection

    Returns:
        List of point IDs that were read
    """
    logger.info(f"Reading 10 points from '{collection_name}' with scroll")

    scroll_result = client.scroll(
        collection_name=collection_name,
        limit=10,
        with_payload=False,
        with_vectors=True,
    )

    points, _ = scroll_result
    point_ids = [point.id for point in points]

    logger.info(f"Read {len(points)} points with IDs: {point_ids}")
    logger.info(
        f"  First point has vector shape: {len(points[0].vector) if points else 'N/A'}"
    )

    return point_ids


def retrieve_payloads(
    client: QdrantClient, collection_name: str, point_ids: List[int]
) -> None:
    """Retrieve payloads for specific point IDs

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        point_ids: List of point IDs to retrieve
    """
    logger.info(f"Retrieving payloads for {len(point_ids)} points")

    retrieved_points = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_vectors=False,
        with_payload=True,
    )

    logger.info(f"Retrieved {len(retrieved_points)} points with payloads")
    if retrieved_points:
        logger.info(
            f"  Sample payload keys: {list(retrieved_points[0].payload.keys())}"
        )


def read_with_pagination(client: QdrantClient, collection_name: str) -> None:
    """Read 100 points using pagination with limit=10

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
    """
    logger.info(f"Reading 100 points from '{collection_name}' with pagination")

    all_points = []
    offset = None

    for page in range(10):
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=10,
            offset=offset,
            with_payload=False,
            with_vectors=["clip_default"],
        )

        points, next_offset = scroll_result
        all_points.extend(points)

        logger.info(f"  Page {page + 1}: Read {len(points)} points")

        if next_offset is None:
            break

        offset = next_offset

    logger.info(f"Total points read: {len(all_points)}")


def main() -> None:
    """Main entry point for CRUD operations demo"""
    logger.info("Starting CRUD operations demo")

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

    point_ids = read_with_scroll(client, "single_unnamed")

    retrieve_payloads(client, "single_unnamed", point_ids)

    read_with_pagination(client, "multiple_named")

    logger.info("CRUD operations demo completed!")


if __name__ == "__main__":
    main()
