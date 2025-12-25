"""Hybrid search with RRF and DBSF Fusion-based hybrid search using RRF."""

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
from .upload_data import create_hybrid_collection, upload_corpus_hybrid


def search_with_fusion(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    dense_model: SentenceTransformer,
    k: int = 10,
    rrf_k: int = 60,
) -> List[str]:
    """Search using RRF fusion of dense and sparse vectors

    Args:
        client: Qdrant client
        collection_name: Name of the collection
        query_text: Query text
        dense_model: Dense embedding model
        k: Number of results to return
        rrf_k: RRF parameter k

    Returns:
        List of corpus IDs
    """
    # Generate dense embedding
    dense_embedding = dense_model.encode(query_text, convert_to_numpy=True)

    # Search with RRF fusion
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

    # Return _id from payload
    return [point.payload["_id"] for point in results.points]


def evaluate_fusion(
    client: QdrantClient,
    collection_name: str,
    queries: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, List[str]],
    dense_model: SentenceTransformer,
    k: int = 10,
    rrf_k: int = 60,
) -> tuple[float, float]:
    """Evaluate fusion-based hybrid search

    Returns:
        (MRR, total_search_time)
    """
    predictions: Dict[str, List[str]] = {}

    start_time = time.perf_counter()

    for query_id, query_data in tqdm(queries.items(), desc="Searching"):
        query_text = query_data["text"]
        results = search_with_fusion(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            dense_model=dense_model,
            k=k,
            rrf_k=rrf_k,
        )
        predictions[query_id] = results

    total_search_time = time.perf_counter() - start_time

    mrr = calculate_mrr(ground_truth, predictions)

    return mrr, total_search_time


def main() -> None:
    """Main function for fusion-based hybrid search"""
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
    qdrant_path = repo_root / "hw6" / "qdrant_storage_fusion"
    client = QdrantClient(path=str(qdrant_path))

    # Create and upload to collection
    collection_name = "hw6"
    dense_model_name = "sentence-transformers/all-minilm-l6-v2"

    print(f"\nLoading dense model: {dense_model_name}")
    dense_model = SentenceTransformer(dense_model_name)

    print("\nCreating collection...")
    create_hybrid_collection(client, collection_name)

    print("\nUploading corpus...")
    upload_corpus_hybrid(
        client, collection_name, corpus, dense_model_name=dense_model_name
    )

    # Evaluate with different RRF k values
    print("\n=== Evaluating RRF Fusion ===")
    rrf_k_values = [10, 30, 60, 100]

    # Save results
    results_dir = repo_root / "hw6" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "fusion_results.jsonl"

    with results_path.open("w", encoding="utf-8") as f:
        for rrf_k in rrf_k_values:
            print(f"\nRRF k={rrf_k}")
            mrr, search_time = evaluate_fusion(
                client=client,
                collection_name=collection_name,
                queries=queries,
                ground_truth=ground_truth,
                dense_model=dense_model,
                k=10,
                rrf_k=rrf_k,
            )

            qps = len(queries) / search_time if search_time > 0 else 0.0

            result = {
                "task": "fusion_search",
                "rrf_k": rrf_k,
                "MRR": mrr,
                "QPS": qps,
                "total_search_time": search_time,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"MRR: {mrr:.4f}")
            print(f"Total search time: {search_time:.2f}s")
            print(f"QPS: {qps:.2f}")

    print(f"\nResults saved to: {results_path}")
    # MRR: Expected around 0.5-0.65


if __name__ == "__main__":
    main()
