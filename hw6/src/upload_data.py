"""Upload SciFact data to Qdrant collections"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from .data_loader import load_corpus


def create_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 384,  # all-minilm-l6-v2 dimension
) -> None:
    """Create a collection with dense and sparse vectors for hybrid search."""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
    print(f"Created collection: {collection_name}")


def create_colbert_collection(
    client: QdrantClient,
    collection_name: str,
    multivector_size: int = 128,  # ColBERT dimension
) -> None:
    """Create a collection with ColBERT multivectors (m=0 to disable HNSW)."""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )

    # Add ColBERT multivector configuration with m=0
    client.update_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,  # all-minilm-l6-v2
                distance=models.Distance.COSINE,
            ),
        },
    )

    # Note: ColBERT multivectors will be added via on_disk configuration
    # with m=0 to disable HNSW graph construction
    print(f"Created ColBERT collection: {collection_name}")


def upload_corpus_hybrid(
    client: QdrantClient,
    collection_name: str,
    corpus: Dict[str, Dict[str, str]],
    dense_model_name: str = "sentence-transformers/all-minilm-l6-v2",
    sparse_model_name: str = "Qdrant/bm25",
    batch_size: int = 100,
) -> None:
    """Upload corpus to Qdrant with dense and sparse embeddings."""
    from sentence_transformers import SentenceTransformer
    import torch

    # Load models - use GPU if available for encoding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading dense model: {dense_model_name} on {device}")
    dense_model = SentenceTransformer(dense_model_name, device=device)

    print(f"Uploading {len(corpus)} documents to {collection_name}...")

    corpus_list = list(corpus.values())

    for i in tqdm(range(0, len(corpus_list), batch_size), desc="Uploading"):
        batch = corpus_list[i : i + batch_size]

        # Prepare texts for embedding
        texts = [f"{doc['title']} {doc['text']}" for doc in batch]

        # Generate dense embeddings
        dense_embeddings = dense_model.encode(texts, convert_to_numpy=True)

        # Prepare points
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
                        ),  # Will be computed by Qdrant BM25
                    },
                    payload={
                        "_id": doc["_id"],
                        "title": doc["title"],
                        "text": doc["text"],
                    },
                )
            )

        # Upload batch
        client.upsert(collection_name=collection_name, points=points)

    print(f"Uploaded {len(corpus)} documents to {collection_name}")


def main() -> None:
    """Main function to upload data to Qdrant."""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "hw6" / "data"

    corpus_path = data_dir / "corpus.jsonl"

    print("Loading corpus...")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} corpus documents")

    # Initialize Qdrant client (in-memory for now)
    client = QdrantClient(":memory:")

    # Create and upload to hybrid collection
    collection_name = "hw6"
    create_hybrid_collection(client, collection_name)
    upload_corpus_hybrid(client, collection_name, corpus)

    print("Done!")


if __name__ == "__main__":
    main()
