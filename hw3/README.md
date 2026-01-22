# Homework 3: Vector Search Engines with Qdrant

## Installation

Install dependencies for this homework:

```bash
uv sync --group hw3 --group dev
```

## Data Preparation

### 1. Download Dataset

Download and extract the laion-small-clip dataset:

```bash
bash hw3/download_data.sh
```

This script:
- Downloads the dataset (100,000 CLIP embeddings, 512 dimensions)
- Extracts `vectors.npy` and `payloads.jsonl`
- Skips download if data already exists

### 2. Generate Ground Truth

**IMPORTANT:** Start Qdrant

```bash
docker run --name qdrant-hw3 -p 6333:6333 -p 6334:6334 -v ${PWD}/hw3/snapshots:/qdrant/snapshots qdrant/qdrant:v1.16.2
```

Then generate ground truth using Qdrant brute-force search:

```bash
# Full dataset
uv run python -m hw3.generate_ground_truth

# test with subset
uv run python -m hw3.generate_ground_truth --sample-size 10000
```

**Available options:**
- `--sample-size N`: Process only first N queries (default: all)
- `--qdrant-url URL`: Qdrant URL (default: `http://localhost:6333`, use `:memory:` for in-memory but it's SLOW)
- `--reuse-collection`: Skip recreation/upload if data already in Qdrant (saves time on reruns)
- `--top-k K`: Number of nearest neighbors (default: 10)

This creates `hw3/data/ground_truth.jsonl` with 10 true nearest neighbors for each query vector.


## Tasks

### Task 2.1: Collection Configuration

Create collections with different configurations:

```bash
uv run python -m hw3.qdrant_collections
```

Creates:
- `single_unnamed`: Unnamed vectors, default HNSW parameters
- `multiple_named`: Named vectors (`clip_default` and `clip_tuned`) with custom HNSW parameters

### Task 2.2: Upload Data

Upload data using different methods and compare performance:

```bash
uv run python -m hw3.upload_data
```

Tests:
- `upsert` method with batching
- `upload_points` with parallelization
- `upload_collection` for named vectors

### Task 2.3: Search Experiments

Run search experiments and evaluate precision:

```bash
uv run python -m hw3.search
```

Outputs:
- `single_unnamed_results.jsonl`
- `clip_default_results.jsonl`
- `clip_tuned_results.jsonl`
- `clip_tuned_ef_search_50_results.jsonl`

Metrics calculated:
- Precision@1, @3, @5, @10
- QPS (queries per second)
- Total search time

### Task 2.4.1: CRUD Operations

Demonstrate read operations using `scroll` and `retrieve`:

```bash
uv run python -m hw3.crud_operations
```

Operations:
- Read with `scroll` (with_payload=False, with_vectors=True)
- Retrieve payloads by IDs
- Paginated reading (100 points, limit=10)

### Task 2.4.2: Snapshot & Restore

Test snapshot creation, collection modification, and restoration:

```bash
uv run python -m hw3.snapshot_restore
```

Demonstrates:
- Creating collection snapshots
- Spoiling data with various modification methods:
  - `upsert`, `update_vectors`, `set_payload`
  - `overwrite_payload`, `delete_payload`, `clear_payload`
  - `delete_vectors`, `delete`
- Restoring from snapshot
- Verification of restoration

## Results

After running the experiments, the following files will be generated:

- `hw3/ground_truth.jsonl` - Ground truth nearest neighbors
- `hw3/*_results.jsonl` - Search performance metrics
- `hw3/snapshots/` - Collection snapshots (created during snapshot_restore.py)
