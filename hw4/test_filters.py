"""Test different filter types on filter_experiments collection (HW4 task 2.4)"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def test_filter(
    client: QdrantClient,
    collection_name: str,
    filter_name: str,
    query_filter: models.Filter,
    limit: int = 5,
) -> None:
    """Test a single filter and display results"""
    logger.info(f"Filter: {filter_name}")

    results = client.query_points(
        collection_name=collection_name,
        query=[0.0] * 512,
        query_filter=query_filter,
        limit=limit,
    )

    logger.info(f"Found {len(results.points)} results")
    for idx, point in enumerate(results.points[:3], 1):
        logger.info(f"  Result {idx}: ID={point.id}, payload={point.payload}")


def main() -> None:
    """Test various filter types"""
    logger.info("Testing different filter types (HW4 task 2.4)")

    collection_name = "filter_experiments"
    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=300)

    if not client.collection_exists(collection_name):
        logger.error(
            f"Collection '{collection_name}' does not exist. "
            "Run 'python -m hw4.generate_dataset' first."
        )
        return

    # Filter 1: Match - exact keyword match
    test_filter(
        client,
        collection_name,
        "Filter 1: Match (keyword) - color='red'",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value="red"),
                )
            ]
        ),
    )

    # Filter 2: Match - integer match
    test_filter(
        client,
        collection_name,
        "Filter 2: Match (integer) - quantity=50",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="quantity",
                    match=models.MatchValue(value=50),
                )
            ]
        ),
    )

    # Filter 3: Match - boolean match
    test_filter(
        client,
        collection_name,
        "Filter 3: Match (bool) - in_stock=true",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="in_stock",
                    match=models.MatchValue(value=True),
                )
            ]
        ),
    )

    # Filter 4: Match Any - keyword in list
    test_filter(
        client,
        collection_name,
        "Filter 4: Match Any - color in ['red', 'blue']",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="color",
                    match=models.MatchAny(any=["red", "blue"]),
                )
            ]
        ),
    )

    # Filter 5: Match Except - keyword not in list
    test_filter(
        client,
        collection_name,
        "Filter 5: Match Except - color not in ['red', 'blue']",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="color",
                    match=models.MatchExcept(**{"except": ["red", "blue"]}),
                )
            ]
        ),
    )

    # Filter 6: Range - float range
    test_filter(
        client,
        collection_name,
        "Filter 6: Range (float) - 100 <= price <= 500",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="price",
                    range=models.Range(gte=100.0, lte=500.0),
                )
            ]
        ),
    )

    # Filter 7: Range - integer range with gt/lt
    test_filter(
        client,
        collection_name,
        "Filter 7: Range (integer) - 20 < quantity < 80",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="quantity",
                    range=models.Range(gt=20, lt=80),
                )
            ]
        ),
    )

    # Filter 8: Geo Radius - location within radius
    test_filter(
        client,
        collection_name,
        "Filter 8: Geo Radius - within 1000km of (52.52, 13.40)",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="location",
                    geo_radius=models.GeoRadius(
                        center=models.GeoPoint(lat=52.52, lon=13.40),
                        radius=1_000_000.0,
                    ),
                )
            ]
        ),
    )

    # Filter 9: Geo Bounding Box - location in rectangle
    test_filter(
        client,
        collection_name,
        "Filter 9: Geo Bounding Box - Europe region",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="location",
                    geo_bounding_box=models.GeoBoundingBox(
                        top_left=models.GeoPoint(lat=60.0, lon=-10.0),
                        bottom_right=models.GeoPoint(lat=35.0, lon=40.0),
                    ),
                )
            ]
        ),
    )

    # Filter 10: Is Null - field is null
    test_filter(
        client,
        collection_name,
        "Filter 10: Is Null - discount is null",
        models.Filter(
            must=[
                models.IsNullCondition(
                    is_null=models.PayloadField(key="discount"),
                )
            ]
        ),
    )

    # Filter 11: Is Empty - field is empty/null/missing
    test_filter(
        client,
        collection_name,
        "Filter 11: Is Empty - tags is empty",
        models.Filter(
            must=[
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="tags"),
                )
            ]
        ),
    )

    # Filter 12: Combined with must_not - exclude condition
    test_filter(
        client,
        collection_name,
        "Filter 12: Must Not - color != 'red' AND in_stock=true",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="in_stock",
                    match=models.MatchValue(value=True),
                )
            ],
            must_not=[
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value="red"),
                )
            ],
        ),
    )

    # Filter 13: Combined with should - OR logic
    test_filter(
        client,
        collection_name,
        "Filter 13: Should - color='red' OR color='blue'",
        models.Filter(
            should=[
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value="red"),
                ),
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value="blue"),
                ),
            ]
        ),
    )

    # Filter 14: Complex combination - must + should + must_not
    test_filter(
        client,
        collection_name,
        "Filter 14: Complex - (category='electronics' OR 'books') AND price<200 AND color!='black'",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="price",
                    range=models.Range(lt=200.0),
                )
            ],
            should=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value="electronics"),
                ),
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value="books"),
                ),
            ],
            must_not=[
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value="black"),
                )
            ],
        ),
    )

    # Filter 15: Nested filter - array field contains value
    test_filter(
        client,
        collection_name,
        "Filter 15: Array Match - 'sale' in tags",
        models.Filter(
            must=[
                models.FieldCondition(
                    key="tags",
                    match=models.MatchValue(value="sale"),
                )
            ]
        ),
    )
    logger.info("All filter tests completed")


if __name__ == "__main__":
    main()
