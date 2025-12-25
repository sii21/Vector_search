# HW1 – Vector Search

This homework implements distance functions, benchmarks them, builds sentence embeddings, searches for nearest neighbours, and visualizes embeddings.

## Installation

```bash
uv sync --group hw1 --group dev
```

## Distances

Check correctness:

```bash
uv run --group hw1 python -m search_hw.check_distances
```

Run benchmarks:

```bash
uv run --group hw1 python -m search_hw.benchmarks
```

## Task 2. Embeddings

First of all prepare dataset:

```
data/sentences.txt # one sentence per line
```

Then build embeddings via:

```bash
uv run --group hw1 python -m search_hw.step_2.embeddings
```

Final step is nearest neighbour search:

```bash
uv run --group hw1 python -m search_hw.step_2.search
```

## Visualization

PCA, t-SNE, UMAP:

```bash
uv run --group hw1 python -m search_hw.step_2.visualize
```

Images will appear in:

```
figures/
```

## Key points

### Task 1:

```
N = 1000, dim = 16
numpy    : 0.000106 s
python   : 0.001010 s
speedup  : 9.5x slower (python vs numpy)

N = 10000, dim = 16
numpy    : 0.000712 s
python   : 0.011992 s
speedup  : 16.8x slower (python vs numpy)

N = 10000, dim = 128
numpy    : 0.004155 s
python   : 0.073935 s
speedup  : 17.8x slower (python vs numpy)

N = 50000, dim = 128
numpy    : 0.030456 s
python   : 0.428129 s
speedup  : 14.1x slower (python vs numpy)
```

**NumPy > pure Python in performance**

### Task 2:

Got 4 clusters in the embeddings:
* Star Wars
* Taxi Uber
* Food Reviews
* Random Sentences


```
Total embeddings available: 201
Sample sizes to use: [201]

Sample size: 201
PCA   time: 0.003 s
t-SNE time: 1.601 s
UMAP  time: 5.668 s
```

* **PCA = fastest**
* **t-SNE = compromise between speed and structure**
* **UMAP = slowest, best cluster separation**

Result:

![embeddings_n201.png](figures%2Fembeddings_n201.png)


# HW2 – ANN Exploration

This homework evaluates three ANN algorithms (ANNOY, HNSW, IVFPQ) on CLIP embeddings from a LAION-small subset.  
The workflow includes dataset download, ground-truth generation, ANN search, and Precision@k evaluation.

## Installation

```bash
uv sync --group hw2 --group dev
```

## Dataset Preparation
Download and prepare dataset:

```bash
./hw2/download_data.sh
```

After extraction expected files:

```
hw2/data/vectors.npy
hw2/data/payloads.jsonl
hw2/data/tests.jsonl
```

## Ground-Truth Generation

Compute exact 10 nearest neighbours for all vectors using FAISS (IndexFlatL2):

Run: 
```bash
uv run --group hw2 python -m hw2.src.ground_truth
```

## ANNOY

Parameters tested:

* n_trees = [10, 25, 50, 100, 200]
* search_k = [100, 500, 1000, 5000]

Run:
```bash
uv run --group hw2 python -m hw2.src.annoy_runner
```

## HNSW (FAISS)

Parameters tested:

* M = [8, 16, 32, 64]
* efConstruction = [32, 64, 100, 128, 256]
* efSearch = [32, 64, 100, 128, 256]

Run:
```bash
uv run --group hw2 python -m hw2.src.hnsw
```

## IVFPQ (FAISS)

Parameters tested:
* nlist = [64, 128, 256, 512, 1024]
* m = [16, 32]
* nbits = [8]
* nprobe = [1, 2, 4, 8, 16, 32, 64, 128]

Run:
```bash
uv run --group hw2 python -m hw2.src.ivfpq
```

# HW6 – Hybrid Search

This homework implements hybrid search using Qdrant with dense and sparse vectors, explores different fusion strategies (RRF, DBSF), late interaction with ColBERT,
and cross-encoder reranking on the SciFact dataset

## Installation

```bash
uv sync --group hw6 --group dev
```

## Dataset Preparation

Download SciFact dataset:

```bash
uv run --group hw6 python -m hw6.download_data
```

## Task 2.2.1: Fusion-based Hybrid Search

```bash
uv run --group hw6 python -m hw6.src.fusion_search
```

Expected MRR: 0.5-0.65

## Task 2.2.2: Late Interaction with ColBERT

```bash
uv run --group hw6 python -m hw6.src.colbert_search
```

## Task 2.2.3: Cross-Encoder Reranking

```bash
uv run --group hw6 python -m hw6.src.crossencoder_search
```

## Task 3: Competition (MRR > 0.69)

```bash
uv run --group hw6 python -m hw6.src.competition
```

Results saved to: `hw6/results/final_hybrid_results.jsonl`

See [hw6/README.md](hw6/README.md) for detailed documentation.