# HW6 - Hybrid Search

This homework implements hybrid search using Qdrant with dense and sparse vectors, explores different fusion strategies (RRF, DBSF), late interaction with ColBERT, and cross-encoder reranking on the SciFact dataset.

## Installation

```bash
uv sync --group hw6 --group dev
```

## Dataset Preparation

Download SciFact dataset via MTEB:

```bash
uv run --group hw6 python -m hw6.download_data
```

**Note:** The dataset is downloaded using the MTEB library, which provides a standardized interface for retrieval benchmarks.

After download, expected files:

```
hw6/data/corpus.jsonl
hw6/data/queries.jsonl
hw6/data/default.jsonl
```

## Task 2.2.1: Fusion-based Hybrid Search

Search with dense (all-minilm-l6-v2) and sparse (BM25) vectors, fuse with RRF:

```bash
uv run --group hw6 python -m hw6.src.fusion_search
```

**Results:**
- Tests multiple RRF k values: [10, 30, 60, 100]
- Expected MRR: 0.5-0.65
- Evaluation metric: MRR (Mean Reciprocal Rank)

## Task 2.2.2: Late Interaction with ColBERT

Create a separate collection with ColBERT vectors (m=0 to disable HNSW graph):

```bash
uv run --group hw6 python -m hw6.src.colbert_search
```

**Note:** This is a simplified ColBERT implementation. Full implementation would require token-level embeddings and MaxSim scoring.

## Task 2.2.3: Cross-Encoder Reranking

Rerank search results using a cross-encoder model:

```bash
uv run --group hw6 python -m hw6.src.crossencoder_search
```

**Pipeline:**
1. Stage 1: Retrieve candidates with dense + sparse fusion (RRF)
2. Stage 2: Rerank with cross-encoder (ms-marco-MiniLM-L-6-v2)

## Task 3: Competition

Fast single-stage pipeline (best result MRR=0.7004, search 0.036s CPU) using BGE-base-en-v1.5:

```bash
uv run --group hw6 python -m hw6.src.competition_best
```

**Best result:**
- Model: `BAAI/bge-base-en-v1.5`
- MRR: **0.7004**
- Search time (CPU numpy): **0.036s** (≤1.5s)
- Encoding on GPU (one-time): ~14 min
- Results saved to: `hw6/results/final_hybrid_results.jsonl`

## Key Implementation Details

### Data Structure
- **Corpus**: `{_id, title, text}`
- **Queries**: `{_id, text}`
- **Ground Truth**: `{query-id, corpus-id, score}` (score > 0 for relevant documents)

### Qdrant Configuration
- **Dense vectors**: COSINE distance
- **Sparse vectors**: BM25 with IDF modifier
- **Fusion**: RRF (Reciprocal Rank Fusion)

### Models Used
- Dense: `sentence-transformers/all-minilm-l6-v2` (384-dim), `all-mpnet-base-v2` (768-dim)
- Sparse: `Qdrant/bm25`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `ms-marco-MiniLM-L-12-v2`
