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
