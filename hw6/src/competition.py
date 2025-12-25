"""Competition: best pipeline with BGE-base-en-v1.5, GPU encoding, CPU search"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .data_loader import load_corpus, load_ground_truth, load_queries
from .metrics import calculate_mrr


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dense_search(
    corpus_vectors: np.ndarray,
    corpus_ids: List[str],
    query_vectors: np.ndarray,
    k: int = 10,
) -> List[List[str]]:
    scores = np.dot(query_vectors, corpus_vectors.T)
    top_idx = np.argsort(-scores, axis=1)[:, :k]
    return [[corpus_ids[i] for i in row] for row in top_idx]


def run_best_pipeline() -> Dict:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "hw6" / "data"
    results_dir = repo_root / "hw6" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading data...")
    corpus = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / "queries.jsonl")
    ground_truth = load_ground_truth(data_dir / "default.jsonl")

    corpus_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    corpus_texts = [
        f"{corpus[cid]['title']} {corpus[cid]['text']}" for cid in corpus_ids
    ]
    query_texts = [queries[qid]["text"] for qid in query_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    t0 = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    t_load = time.perf_counter() - t0
    print(f"Downloading model: {t_load:.2f}s")

    print("Encoding corpus на BGE-base-en-v1.5...")
    t0 = time.perf_counter()
    corpus_vecs = model.encode(
        corpus_texts,
        batch_size=256 if device == "cuda" else 256,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    t_corpus = time.perf_counter() - t0
    print(f"Corpus encoding: {t_corpus:.1f}s")

    print("Encoding queries...")
    t0 = time.perf_counter()
    query_vecs = model.encode(
        query_texts,
        batch_size=256 if device == "cuda" else 256,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    t_queries = time.perf_counter() - t0
    print(f"Queries encoding: {t_queries:.2f}s")

    print("Search top-10 (numpy)...")
    t0 = time.perf_counter()
    top_results = dense_search(corpus_vecs, corpus_ids, query_vecs, k=10)
    t_search = time.perf_counter() - t0
    print(f"Search time: {t_search:.3f}s")

    predictions = {qid: top_results[i] for i, qid in enumerate(query_ids)}
    mrr = calculate_mrr(ground_truth, predictions)
    qps = len(queries) / t_search if t_search > 0 else 0.0

    result = {
        "experiment": "bge_base_best",
        "model": "BAAI/bge-base-en-v1.5",
        "MRR": mrr,
        "search_time": t_search,
        "total_search_time": t_search,
        "QPS": qps,
        "encoding_time": t_corpus + t_queries,
        "target_reached": mrr >= 0.69,
    }

    results_path = results_dir / "final_hybrid_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"MRR: {mrr:.4f}")
    print(f"Search: {t_search:.3f}s (CPU numpy)")
    print(f"QPS: {qps:.0f}")
    print(f"Encoding total: {t_corpus + t_queries:.1f}s")
    print(f"Results saved {results_path}")

    return result


def main() -> None:
    run_best_pipeline()


if __name__ == "__main__":
    main()
