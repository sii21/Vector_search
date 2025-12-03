from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from search_hw.step_1.distances_numpy import cosine_similarity
from search_hw.step_2.embeddings import load_sentences


def load_dataset_and_embeddings(
    base_dir: Path,
) -> Tuple[List[str], np.ndarray]:
    """Load sentences and their embeddings from data/ directory"""
    data_dir = base_dir / "data"
    text_path = data_dir / "sentences.txt"
    emb_path = data_dir / "sentences_embeddings.npy"

    sentences = load_sentences(text_path)

    if not emb_path.exists():
        msg = f"Embeddings file not found: {emb_path}. Run 'python -m search_hw.step_2.embeddings' first."
        raise FileNotFoundError(msg)

    embeddings = np.load(emb_path, mmap_mode="r")
    if embeddings.shape[0] != len(sentences):
        msg = f"Number of embeddings {embeddings.shape[0]} does not match number of sentences {len(sentences)}"
        raise ValueError(msg)

    return sentences, embeddings


def find_nearest_neighbors(
    embeddings: np.ndarray,
    query_index: int,
    top_k: int = 5,
) -> np.ndarray:
    """Find indices of nearest neighbors by cosine similarity for a given sentence index"""
    query_vec = embeddings[query_index]
    sims = cosine_similarity(query_vec, embeddings)  # shape (N,)

    # Higher cosine similarity means closer neighbor
    order = np.argsort(-sims)

    # Exclude the query itself
    order = order[order != query_index]

    return order[:top_k]


def demo_search(n_examples: int = 10, top_k: int = 5) -> None:
    """Run a simple nearest neighbor search demo on random sentences"""
    base_dir = Path(__file__).resolve().parents[2]
    sentences, embeddings = load_dataset_and_embeddings(base_dir)

    n_sentences = len(sentences)
    rng = np.random.default_rng(42)
    n_examples = min(n_examples, n_sentences)

    indices = rng.choice(n_sentences, size=n_examples, replace=False)

    for idx in indices:
        print("=" * 80)
        print(f"Query index: {idx}")
        print(f"Query sentence: {sentences[idx]}")
        print()

        neighbors = find_nearest_neighbors(embeddings, idx, top_k=top_k)

        print(f"Top {top_k} nearest neighbors:")
        for rank, nb_idx in enumerate(neighbors, start=1):
            print(f"{rank}. [#{nb_idx}] {sentences[nb_idx]}")


def main() -> None:
    demo_search(n_examples=10, top_k=5)


if __name__ == "__main__":
    main()
