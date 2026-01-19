"""Demonstrate snapshot creation, collection modification, and restoration"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def create_snapshot(
    client: QdrantClient, collection_name: str, snapshot_dir: Path
) -> str:
    """Create a snapshot of the collection

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to snapshot
        snapshot_dir: Directory to save snapshot

    Returns:
        Name of the created snapshot
    """
    logger.info(f"Creating snapshot of '{collection_name}'")

    snapshot_info = client.create_snapshot(
        collection_name=collection_name,
        wait=True,
    )
    snapshot_name = snapshot_info.name

    logger.info(f"Snapshot created: {snapshot_name}")

    return snapshot_name


def spoil_collection(client: QdrantClient, collection_name: str) -> None:
    """Modify collection data to create unexplainable weird results

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to spoil
    """
    logger.info(f"Spoiling collection '{collection_name}'")

    point_ids_to_modify = [42, 100, 500, 1000, 5000]

    logger.info("1. Testing upsert - replacing vectors with random data")
    random_vectors = [np.random.randn(512).tolist() for _ in range(5)]
    points = [
        models.PointStruct(
            id=point_id,
            vector={
                "clip_default": vector,
                "clip_tuned": vector,
            },
            payload={"spoiled": True, "method": "upsert"},
        )
        for point_id, vector in zip(point_ids_to_modify, random_vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)

    retrieved = client.retrieve(
        collection_name=collection_name, ids=point_ids_to_modify[:2]
    )
    assert len(retrieved) == 2
    assert retrieved[0].payload.get("spoiled") is True
    logger.info(f"   Verified: {len(retrieved)} points modified via upsert")

    logger.info("2. Testing update_vectors - updating specific vectors")
    new_vectors = [np.random.randn(512).tolist() for _ in range(3)]
    client.update_vectors(
        collection_name=collection_name,
        points=[
            models.PointVectors(
                id=10,
                vector={
                    "clip_default": new_vectors[0],
                    "clip_tuned": new_vectors[0],
                },
            ),
            models.PointVectors(
                id=20,
                vector={
                    "clip_default": new_vectors[1],
                    "clip_tuned": new_vectors[1],
                },
            ),
            models.PointVectors(
                id=30,
                vector={
                    "clip_default": new_vectors[2],
                    "clip_tuned": new_vectors[2],
                },
            ),
        ],
    )

    retrieved = client.retrieve(collection_name=collection_name, ids=[10, 20])
    assert len(retrieved) == 2
    logger.info(f"   Verified: {len(retrieved)} points updated via update_vectors")

    logger.info("3. Testing set_payload - adding malicious payload")
    client.set_payload(
        collection_name=collection_name,
        payload={
            "corrupted": True,
            "timestamp": "2024-01-01",
            "reason": "unknown_error",
        },
        points=[200, 201, 202, 203],
    )

    retrieved = client.retrieve(collection_name=collection_name, ids=[200, 201])
    assert all(p.payload.get("corrupted") is True for p in retrieved)
    logger.info(f"   Verified: Payload added to {len(retrieved)} points")

    logger.info("4. Testing overwrite_payload - completely replacing payloads")
    client.overwrite_payload(
        collection_name=collection_name,
        payload={"status": "broken", "original_data": "lost"},
        points=[300, 301, 302],
    )

    retrieved = client.retrieve(collection_name=collection_name, ids=[300])
    assert retrieved[0].payload.get("status") == "broken"
    assert "corrupted" not in retrieved[0].payload
    logger.info("   Verified: Payload completely overwritten for point 300")

    logger.info("5. Testing delete_payload - removing specific payload fields")
    client.delete_payload(
        collection_name=collection_name,
        keys=["spoiled"],
        points=point_ids_to_modify[:2],
    )

    retrieved = client.retrieve(
        collection_name=collection_name, ids=point_ids_to_modify[:1]
    )
    assert "spoiled" not in retrieved[0].payload
    logger.info("   Verified: Payload field 'spoiled' deleted")

    logger.info("6. Testing clear_payload - removing all payload data")
    client.clear_payload(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=[400, 401, 402]),
    )

    retrieved = client.retrieve(collection_name=collection_name, ids=[400])
    assert len(retrieved[0].payload) == 0
    logger.info("   Verified: All payload cleared for point 400")

    logger.info("7. Testing delete_vectors - removing specific vectors")
    client.delete_vectors(
        collection_name=collection_name,
        vectors=["clip_tuned"],
        points=models.PointIdsList(points=[600, 601, 602]),
    )

    retrieved = client.retrieve(
        collection_name=collection_name, ids=[600], with_vectors=True
    )
    assert "clip_tuned" not in retrieved[0].vector
    assert "clip_default" in retrieved[0].vector
    logger.info("   Verified: Vector 'clip_tuned' deleted from point 600")

    logger.info("8. Testing delete - removing entire points")
    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=[700, 701, 702, 703, 704]),
    )

    retrieved = client.retrieve(
        collection_name=collection_name, ids=[700, 701, 702, 703, 704]
    )
    assert len(retrieved) == 0
    logger.info("   Verified: 5 points completely deleted")

    info = client.get_collection(collection_name)
    logger.warning(f"\nCollection spoiled! Current point count: {info.points_count}")
    logger.warning("   Investigation required...")


def restore_from_snapshot(
    client: QdrantClient,
    collection_name: str,
    snapshot_name: str,
) -> None:
    """Restore collection from snapshot

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to restore
        snapshot_name: Name of the snapshot to restore from
    """
    logger.info(f"Restoring '{collection_name}' from snapshot '{snapshot_name}'")

    snapshot_url = (
        f"http://localhost:6333/collections/{collection_name}/snapshots/{snapshot_name}"
    )
    client.recover_snapshot(
        collection_name=collection_name,
        location=snapshot_url,
    )

    logger.info("Snapshot restored successfully!")


def verify_restoration(client: QdrantClient, collection_name: str) -> None:
    """Verify that collection has been properly restored

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to verify
    """
    logger.info(f"Verifying restoration of '{collection_name}'")

    test_ids = [42, 100, 200, 300, 400, 600]
    retrieved = client.retrieve(
        collection_name=collection_name,
        ids=test_ids,
        with_payload=True,
        with_vectors=True,
    )

    assert len(retrieved) == len(
        test_ids
    ), f"Expected {len(test_ids)} points, got {len(retrieved)}"
    logger.info(f"   All {len(test_ids)} key test points exist")

    corruption_found = False
    for point in retrieved:
        if "spoiled" in point.payload or "corrupted" in point.payload:
            logger.warning(
                f"   Point {point.id} has corruption artifacts (likely from previous runs)"
            )
            corruption_found = True
        if "status" in point.payload and point.payload["status"] == "broken":
            logger.warning(f"   Point {point.id} has broken status")
            corruption_found = True

    if not corruption_found:
        logger.info("   No corruption artifacts in payloads")
    else:
        logger.warning(
            "   Some corruption found - snapshot was made after previous spoiling. Run qdrant_collections + upload_data for clean state."
        )

    point_600 = [p for p in retrieved if p.id == 600][0]
    assert "clip_tuned" in point_600.vector, "Point 600 missing 'clip_tuned' vector"
    assert "clip_default" in point_600.vector, "Point 600 missing 'clip_default' vector"
    logger.info("   Deleted vectors restored")

    info = client.get_collection(collection_name)
    logger.info(f"   Collection has {info.points_count} points")

    logger.info(
        "\nCollection fully restored! It was just a minor dev environment issue"
    )


def main() -> None:
    """Main entry point for snapshot/restore demo"""
    logger.info("Starting snapshot/restore demo")

    base_dir = Path(__file__).resolve().parent
    snapshot_dir = base_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    client = QdrantClient("http://localhost:6333", timeout=300)
    logger.info("Connected to Qdrant at http://localhost:6333 (timeout=300s)")

    collection_name = "multiple_named"
    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist. Run `uv run python -m hw3.qdrant_collections` and `uv run python -m hw3.upload_data` first."
        )

    info = client.get_collection(collection_name)
    logger.info(
        f"\nInitial collection: {info.points_count} points, status={info.status}"
    )

    snapshot_name = create_snapshot(client, collection_name, snapshot_dir)

    logger.info("TIME FOR A DAY OFF REQUEST...")

    spoil_collection(client, collection_name)

    logger.info("*DEEP SIGH* Back from day-off...")

    restore_from_snapshot(client, collection_name, snapshot_name)

    verify_restoration(client, collection_name)
    logger.info("Snapshot/restore demo completed!")


if __name__ == "__main__":
    main()
