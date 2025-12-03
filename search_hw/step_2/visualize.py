"""
Visualize sentence embeddings using PCA, t-SNE and UMAP
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap  # type: ignore[import]

plt.style.use("seaborn-v0_8")


def load_embeddings(base_dir: Path) -> np.ndarray:
    """Load embeddings from data/sentences_embeddings.npy"""
    emb_path = base_dir / "data" / "sentences_embeddings.npy"
    if not emb_path.exists():
        msg = f"Embeddings file not found: {emb_path}. Run 'python -m search_hw.embeddings' first."
        raise FileNotFoundError(msg)

    embeddings = np.load(emb_path, mmap_mode="r")
    return embeddings


def make_topic_labels(n_sentences: int, n_topics: int = 4) -> np.ndarray:
    """Create topic labels assuming blocks of equal size in the dataset

    For example, for n_sentences = 200 and n_topics = 4:
    - indices 0..49   -> topic 0
    - indices 50..99  -> topic 1
    - indices 100..149-> topic 2
    - indices 150..199-> topic 3
    """
    if n_sentences < n_topics:
        # fall back: assign each sentence to its own topic
        return np.arange(n_sentences)

    block = n_sentences // n_topics
    labels = np.repeat(np.arange(n_topics), block)

    # if n_sentences is not exactly divisible, assign the rest to the last topic
    if labels.shape[0] < n_sentences:
        extra = n_sentences - labels.shape[0]
        labels = np.concatenate([labels, np.full(extra, n_topics - 1)])

    return labels[:n_sentences]


def sample_indices(n_total: int, n_samples: int, random_state: int = 42) -> np.ndarray:
    """Sample indices without replacement"""
    if n_total == 0:
        raise ValueError("No embeddings available for sampling")

    if n_samples > n_total:
        n_samples = n_total

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_total, size=n_samples, replace=False)
    return indices


def run_pca(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run PCA to 2D and return (embedding, elapsed_seconds)"""
    start = time.perf_counter()
    model = PCA(n_components=2, random_state=42)
    Y = model.fit_transform(X)
    elapsed = time.perf_counter() - start
    return Y, elapsed


def run_tsne(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run t-SNE to 2D and return (embedding, elapsed_seconds)

    Perplexity is adjusted to be valid for the given number of samples
    """
    n_samples = X.shape[0]
    if n_samples < 3:
        raise ValueError(f"t-SNE requires at least 3 samples, got {n_samples}")

    raw_perplexity = min(30.0, (n_samples - 1) / 3.0)
    perplexity = max(2.0, raw_perplexity)

    start = time.perf_counter()
    model = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=42,
        perplexity=perplexity,
    )
    Y = model.fit_transform(X)
    elapsed = time.perf_counter() - start
    return Y, elapsed


def run_umap(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run UMAP to 2D and return (embedding, elapsed_seconds)"""
    n_neighbors = max(2, min(X.shape[0] - 1, 15))

    start = time.perf_counter()
    model = umap.UMAP(
        n_components=2,
        random_state=42,
        n_jobs=1,
        n_neighbors=n_neighbors,
    )
    Y = model.fit_transform(X)
    elapsed = time.perf_counter() - start
    return Y, elapsed


def _scatter_with_topics(
    ax: plt.Axes,
    Y: np.ndarray,
    topics: np.ndarray,
    title: str,
) -> None:
    """Scatter plot where color encodes topic id"""
    unique_topics = np.unique(topics)
    cmap = plt.get_cmap("tab10")

    for i, topic_id in enumerate(unique_topics):
        mask = topics == topic_id
        ax.scatter(
            Y[mask, 0],
            Y[mask, 1],
            s=20,
            alpha=0.9,
            color=cmap(i),
            label=f"topic {int(topic_id) + 1}",
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_single_embedding(
    Y: np.ndarray,
    topics: np.ndarray,
    method_name: str,
    n_samples: int,
    elapsed: float,
    output_path: Path,
) -> None:
    """Plot a single 2D embedding and save to a file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_with_topics(
        ax,
        Y,
        topics,
        title=f"{method_name} (n={n_samples}, {elapsed:.2f}s)",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_three_embeddings(
    Y_pca: np.ndarray,
    Y_tsne: np.ndarray,
    Y_umap: np.ndarray,
    topics: np.ndarray,
    times: Tuple[float, float, float],
    n_samples: int,
    output_path: Path,
) -> None:
    """Plot PCA, t-SNE and UMAP embeddings on a single figure"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_pca, t_tsne, t_umap = times

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    _scatter_with_topics(
        axes[0],
        Y_pca,
        topics,
        title=f"PCA (n={n_samples}, {t_pca:.2f}s)",
    )
    _scatter_with_topics(
        axes[1],
        Y_tsne,
        topics,
        title=f"t-SNE (n={n_samples}, {t_tsne:.2f}s)",
    )
    _scatter_with_topics(
        axes[2],
        Y_umap,
        topics,
        title=f"UMAP (n={n_samples}, {t_umap:.2f}s)",
    )

    fig.suptitle(f"Sentence embeddings by topic, sample size {n_samples}", fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def visualize_for_sample_sizes(
    sample_sizes: Iterable[int] = (500, 5_000, 25_000),
) -> None:
    """Run PCA, t-SNE and UMAP for several sample sizes and save plots"""
    base_dir = Path(__file__).resolve().parents[2]
    embeddings = load_embeddings(base_dir)
    figures_dir = base_dir / "figures"

    n_total = embeddings.shape[0]
    if n_total == 0:
        raise ValueError("No embeddings found")

    topics_all = make_topic_labels(n_total, n_topics=4)

    actual_sizes = sorted({min(size, n_total) for size in sample_sizes})

    print(f"Total embeddings available: {n_total}")
    print(f"Sample sizes to use: {actual_sizes}")

    for n in actual_sizes:
        indices = sample_indices(n_total, n_samples=n)
        X = np.asarray(embeddings[indices])
        topics = topics_all[indices]

        print(f"\nSample size: {n}")

        Y_pca, t_pca = run_pca(X)
        print(f"PCA   time: {t_pca:.3f} s")

        Y_tsne, t_tsne = run_tsne(X)
        print(f"t-SNE time: {t_tsne:.3f} s")

        Y_umap, t_umap = run_umap(X)
        print(f"UMAP  time: {t_umap:.3f} s")

        plot_single_embedding(
            Y_pca,
            topics=topics,
            method_name="PCA",
            n_samples=n,
            elapsed=t_pca,
            output_path=figures_dir / f"pca_n{n}.png",
        )
        plot_single_embedding(
            Y_tsne,
            topics=topics,
            method_name="t-SNE",
            n_samples=n,
            elapsed=t_tsne,
            output_path=figures_dir / f"tsne_n{n}.png",
        )
        plot_single_embedding(
            Y_umap,
            topics=topics,
            method_name="UMAP",
            n_samples=n,
            elapsed=t_umap,
            output_path=figures_dir / f"umap_n{n}.png",
        )

        plot_three_embeddings(
            Y_pca=Y_pca,
            Y_tsne=Y_tsne,
            Y_umap=Y_umap,
            topics=topics,
            times=(t_pca, t_tsne, t_umap),
            n_samples=n,
            output_path=figures_dir / f"embeddings_n{n}.png",
        )


def main() -> None:
    visualize_for_sample_sizes(sample_sizes=(500, 5_000, 25_000))


if __name__ == "__main__":
    main()
