from __future__ import annotations

import re
from pathlib import Path
from typing import List

import gensim.downloader as api
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

TOKEN_RE = re.compile(r"[A-Za-z]+")


def load_sentences(path: Path) -> List[str]:
    """Load sentences from a text file, one sentence per line"""
    if not path.exists():
        msg = f"Text file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if not sentences:
        msg = f"No sentences found in {path}"
        raise ValueError(msg)

    return sentences


def load_word_vectors(name: str = "glove-wiki-gigaword-50") -> KeyedVectors:
    """Load pre-trained English word vectors via gensim downloader"""
    model: KeyedVectors = api.load(name)  # type: ignore[assignment]
    return model


def sentence_to_vector(sentence: str, model: KeyedVectors) -> np.ndarray:
    """Compute sentence embedding as an average of word vectors

    Unknown words are skipped. If no tokens are known, returns a zero vector
    """
    tokens = TOKEN_RE.findall(sentence.lower())
    dim = model.vector_size

    vectors = []
    for token in tokens:
        if token in model.key_to_index:
            vectors.append(model.get_vector(token))

    if not vectors:
        return np.zeros(dim, dtype=np.float32)

    arr = np.stack(vectors, axis=0)
    return arr.mean(axis=0).astype(np.float32)


def build_embeddings(
    text_path: Path,
    output_path: Path,
    vectors_name: str = "glove-wiki-gigaword-50",
) -> None:
    """Compute sentence embeddings and save them to a .npy file

    The resulting file can be loaded later with np.load(..., mmap_mode="r")
    """
    sentences = load_sentences(text_path)
    model = load_word_vectors(vectors_name)

    n_sentences = len(sentences)
    dim = model.vector_size

    embeddings = np.zeros((n_sentences, dim), dtype=np.float32)

    for idx, sent in enumerate(sentences):
        embeddings[idx] = sentence_to_vector(sent, model)
        if (idx + 1) % 1000 == 0 or idx == n_sentences - 1:
            print(f"Encoded {idx + 1}/{n_sentences} sentences")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path} with shape {embeddings.shape}")


def main() -> None:
    """Entry point for building embeddings

    Expects data/sentences.txt and writes data/sentences_embeddings.npy
    """
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"

    text_path = data_dir / "sentences.txt"
    output_path = data_dir / "sentences_embeddings.npy"

    build_embeddings(text_path, output_path)


if __name__ == "__main__":
    main()
