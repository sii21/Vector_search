"""Cross-encoder reranking search"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

from .data_loader import load_corpus, load_ground_truth, load_queries
from .metrics import calculate_mrr
from .upload_data import create_hybrid_collection, upload_corpus_hybrid


def search_with_sample(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    dense_model: SentenceTransformer,
    sample_size: int = 100,
) -> List[Tuple[str, float]]:
    """Search using SampleQuery to get candidates for reranking

    Returns:
        List of (corpus_id, initial_score) tuples
    """
    # Generate dense embedding
    dense_embedding = dense_model.encode(query_text, convert_to_numpy=True)

    # Use SampleQuery to retrieve candidates
    # We'll use fusion first to get better candidates
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_embedding.tolist(),
                using="dense",
                limit=sample_size,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=[],
                    values=[],
                ),
                using="sparse",
                limit=sample_size,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=sample_size,
    )

    # Return _id from payload with score
    return [(point.payload["_id"], point.score) for point in results.points]


def rerank_with_crossencoder(
    query_text: str,
    candidates: List[Tuple[str, float]],
    corpus: Dict[str, Dict[str, str]],
    crossencoder: CrossEncoder,
    k: int = 10,
) -> List[str]:
    """Rerank candidates using cross-encoder

    Args:
        query_text: Query text
        candidates: List of (corpus_id, initial_score)
        corpus: Corpus dictionary
        crossencoder: Cross-encoder model
        k: Number of results to return

    Returns:
        List of top-k corpus IDs after reranking
    """
    if not candidates:
        return []

    # Prepare pairs for cross-encoder
    pairs = []
    corpus_ids = []

    for corpus_id, _ in candidates:
        if corpus_id in corpus:
            doc = corpus[corpus_id]
            doc_text = f"{doc['title']} {doc['text']}"
            pairs.append([query_text, doc_text])
            corpus_ids.append(corpus_id)

    if not pairs:
        return []

    # Score with cross-encoder
    scores = crossencoder.predict(pairs)

    # Sort by score (descending)
    ranked = sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True)

    # Return top-k
    return [corpus_id for corpus_id, _ in ranked[:k]]


def evaluate_crossencoder(
    client: QdrantClient,
    collection_name: str,
    queries: Dict[str, Dict[str, str]],
    corpus: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, List[str]],
    dense_model: SentenceTransformer,
    crossencoder: CrossEncoder,
    k: int = 10,
    sample_size: int = 100,
) -> tuple[float, float]:
    """Evaluate cross-encoder reranking"""
    predictions: Dict[str, List[str]] = {}

    start_time = time.perf_counter()

    for query_id, query_data in tqdm(
        queries.items(), desc="Searching with cross-encoder"
    ):
        query_text = query_data["text"]

        # Get candidates
        candidates = search_with_sample(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            dense_model=dense_model,
            sample_size=sample_size,
        )

        # Rerank with cross-encoder
        results = rerank_with_crossencoder(
            query_text=query_text,
            candidates=candidates,
            corpus=corpus,
            crossencoder=crossencoder,
            k=k,
        )

        predictions[query_id] = results

    total_search_time = time.perf_counter() - start_time

    mrr = calculate_mrr(ground_truth, predictions)

    return mrr, total_search_time


def main() -> None:
    """Main function for cross-encoder reranking"""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "hw6" / "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    qdrant_path = repo_root / "hw6" / "qdrant_storage_crossencoder"
    client = QdrantClient(path=str(qdrant_path))

    # Create and upload to collection
    collection_name = "hw6"
    dense_model_name = "sentence-transformers/all-minilm-l6-v2"
    crossencoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    print(f"\nDevice: {device}")
    print(f"Loading dense model: {dense_model_name} on {device}")
    dense_model = SentenceTransformer(dense_model_name, device=device)

    print(f"Loading cross-encoder: {crossencoder_model_name} on {device}")
    crossencoder = CrossEncoder(crossencoder_model_name, device=device)

    print("\nCreating collection...")
    create_hybrid_collection(client, collection_name)

    print("\nUploading corpus...")
    upload_corpus_hybrid(
        client, collection_name, corpus, dense_model_name=dense_model_name
    )

    # Evaluate
    print("\nEvaluating Cross-Encoder Reranking")
    mrr, search_time = evaluate_crossencoder(
        client=client,
        collection_name=collection_name,
        queries=queries,
        corpus=corpus,
        ground_truth=ground_truth,
        dense_model=dense_model,
        crossencoder=crossencoder,
        k=10,
        sample_size=100,
    )

    qps = len(queries) / search_time if search_time > 0 else 0.0

    print(f"\nMRR: {mrr:.4f}")
    print(f"Total search time: {search_time:.2f}s")
    print(f"QPS: {qps:.2f}")

    # Save results
    results_dir = repo_root / "hw6" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "crossencoder_results.jsonl"

    result = {
        "task": "crossencoder_search",
        "model": crossencoder_model_name,
        "MRR": mrr,
        "QPS": qps,
        "total_search_time": search_time,
    }

    with results_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
