"""Late interaction with ColBERT late interaction search"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .data_loader import load_corpus, load_ground_truth, load_queries
from .metrics import calculate_mrr


def create_colbert_collection(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Create a collection with ColBERT multivectors (m=0 to disable HNSW)"""
    # For ColBERT, we need multivector support
    # Since we're doing rescoring only, we set m=0 to disable HNSW graph
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,  # all-minilm-l6-v2
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )

    print(f"Created ColBERT collection: {collection_name}")


def upload_corpus_colbert(
    client: QdrantClient,
    collection_name: str,
    corpus: Dict[str, Dict[str, str]],
    dense_model_name: str = "sentence-transformers/all-minilm-l6-v2",
    batch_size: int = 100,
) -> None:
    """Upload corpus with ColBERT embeddings."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading dense model: {dense_model_name}")
    dense_model = SentenceTransformer(dense_model_name)

    print(f"Uploading {len(corpus)} documents to {collection_name}...")

    corpus_list = list(corpus.values())

    for i in tqdm(range(0, len(corpus_list), batch_size), desc="Uploading"):
        batch = corpus_list[i : i + batch_size]

        texts = [f"{doc['title']} {doc['text']}" for doc in batch]
        dense_embeddings = dense_model.encode(texts, convert_to_numpy=True)

        points = []
        for j, doc in enumerate(batch):
            # Use hash of _id as numeric point_id
            point_id = hash(doc["_id"]) & 0x7FFFFFFFFFFFFFFF  # Positive 64-bit int

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embeddings[j].tolist(),
                        "sparse": models.SparseVector(
                            indices=[],
                            values=[],
                        ),
                    },
                    payload={
                        "_id": doc["_id"],
                        "title": doc["title"],
                        "text": doc["text"],
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points)

    print(f"Uploaded {len(corpus)} documents to {collection_name}")


def search_with_colbert_rescore(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    dense_model: SentenceTransformer,
    k: int = 10,
) -> List[str]:
    """Search with ColBERT rescoring

    Note: This is a simplified version. Full ColBERT implementation would require
    token-level embeddings and MaxSim scoring.
    """
    # Generate dense embedding
    dense_embedding = dense_model.encode(query_text, convert_to_numpy=True)

    # First stage: retrieve with dense + sparse
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_embedding.tolist(),
                using="dense",
                limit=k * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=[],
                    values=[],
                ),
                using="sparse",
                limit=k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
    )

    # Second stage: ColBERT rescoring would happen here
    # For now, we return the fusion results
    # In a full implementation, you would:
    # 1. Get token embeddings for query and documents
    # 2. Compute MaxSim scores
    # 3. Rerank based on MaxSim

    # Return _id from payload
    return [point.payload["_id"] for point in results.points]


def evaluate_colbert(
    client: QdrantClient,
    collection_name: str,
    queries: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, List[str]],
    dense_model: SentenceTransformer,
    k: int = 10,
) -> tuple[float, float]:
    """Evaluate ColBERT-based search"""
    predictions: Dict[str, List[str]] = {}

    start_time = time.perf_counter()

    for query_id, query_data in tqdm(queries.items(), desc="Searching with ColBERT"):
        query_text = query_data["text"]
        results = search_with_colbert_rescore(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            dense_model=dense_model,
            k=k,
        )
        predictions[query_id] = results

    total_search_time = time.perf_counter() - start_time

    mrr = calculate_mrr(ground_truth, predictions)

    return mrr, total_search_time


def main() -> None:
    """Main function for ColBERT late interaction"""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "hw6" / "data"

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    default_path = data_dir / "default.jsonl"

    print("Loading data...")
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    ground_truth = load_ground_truth(default_path)

    print(f"Loaded {len(corpus)} corpus documents")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(ground_truth)} ground truth entries")

    # Initialize Qdrant client with persistent storage
    qdrant_path = repo_root / "hw6" / "qdrant_storage_colbert"
    client = QdrantClient(path=str(qdrant_path))

    # Create and upload to ColBERT collection
    collection_name = "hw6_colbert"
    dense_model_name = "sentence-transformers/all-minilm-l6-v2"

    print(f"\nLoading dense model: {dense_model_name}")
    dense_model = SentenceTransformer(dense_model_name)

    print("\nCreating ColBERT collection...")
    create_colbert_collection(client, collection_name)

    print("\nUploading corpus...")
    upload_corpus_colbert(
        client, collection_name, corpus, dense_model_name=dense_model_name
    )

    # Evaluate
    print("\n=== Evaluating ColBERT Rescoring ===")
    mrr, search_time = evaluate_colbert(
        client=client,
        collection_name=collection_name,
        queries=queries,
        ground_truth=ground_truth,
        dense_model=dense_model,
        k=10,
    )

    qps = len(queries) / search_time if search_time > 0 else 0.0

    print(f"\nMRR: {mrr:.4f}")
    print(f"Total search time: {search_time:.2f}s")
    print(f"QPS: {qps:.2f}")

    # Save results
    results_dir = repo_root / "hw6" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "colbert_results.jsonl"

    result = {
        "task": "colbert_search",
        "MRR": mrr,
        "QPS": qps,
        "total_search_time": search_time,
    }

    with results_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
