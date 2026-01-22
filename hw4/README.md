# Homework 4: Vector Search Engines - Part 2 (Filtered Search)

## Prerequisites

- Python 3.10+
- uv package manager
- Qdrant server running on `http://localhost:6333` (or use local file storage)

## Installation

1. Install dependencies using uv:

```bash
uv sync --group hw4
```

## Data Preparation

1. Download and extract the dataset:

```bash
bash hw4/download_data.sh
```

This will download the `laion-small-clip.tgz` dataset and extract:
- `vectors.npy` - 512-dimensional vectors
- `payloads.jsonl` - payload data with fields (field_a, field_b, field_c, field_d)
- `tests.jsonl` - test queries with filtering conditions and expected results

The data will be placed in `hw4/data/` directory.

## Running the Experiments

### Task 2.1: Preparing the Collection

Create the `hw4` collection and upload data:

```bash
python -m hw4.qdrant_collections
```

This script will:
- Create a collection named `hw4` with 512-dimensional vectors and cosine similarity
- Upload all vectors and payloads from `hw4/data/`
- Use default HNSW parameters (m=16, ef_construct=100)

### Task 2.2: Filtrable HNSW & Task 2.3: ACORN

Run filtered search experiments with both HNSW and ACORN:

```bash
python -m hw4.filtered_search
```

This script will:
- Load 5,000 test queries from `tests.jsonl`
- Run filtered search using HNSW (default parameters)
- Run filtered search using ACORN (with quantization: `ignore=False`, `rescore=True`, `oversampling=2.0`)
- Build filters from conditions (supports `must`, `should`, `must_not`, `range`, `match`)
- Calculate Precision@1, Precision@3, Precision@5, Precision@10
- Measure QPS (queries per second) and total search time
- Save results to:
  - `hw4/results/filtered_search_results.jsonl` (HNSW)
  - `hw4/results/filtered_search_acorn_results.jsonl` (ACORN)

### Task 2.4: Other Filter Types

#### Step 1: Generate Random Dataset

Create a collection with 200,000 random points:

```bash
python -m hw4.generate_dataset
```

This script will:
- Generate 200,000 random vectors (512-dimensional)
- Generate random payloads with various field types:
  - **Keyword**: `color`, `category`, `brand`
  - **Integer**: `quantity`
  - **Float**: `price`, `rating`, `discount`
  - **Boolean**: `in_stock`
  - **Geo Point**: `location`
  - **Array**: `tags`
  - **Nullable**: `discount`, `brand`
- Create a collection named `filter_experiments`
- Create payload indexes for all filterable fields
- Wait for indexes to be built

#### Step 2: Test Different Filter Types

Test 15 different filter types:

```bash
python -m hw4.test_filters
```

This script tests:
1. **Match (keyword)** - exact string match
2. **Match (integer)** - exact integer match
3. **Match (bool)** - boolean value
4. **Match Any** - IN operator for keywords
5. **Match Except** - NOT IN operator
6. **Range (float)** - range with gte/lte
7. **Range (integer)** - range with gt/lt
8. **Geo Radius** - geographic radius search
9. **Geo Bounding Box** - geographic rectangle
10. **Is Null** - field is NULL
11. **Is Empty** - field is empty/missing
12. **Must Not** - exclude condition
13. **Should** - logical OR
14. **Complex** - combination of must + should + must_not
15. **Array Match** - search in array fields

## Results

Results are saved in `hw4/results/` directory:

- `filtered_search_results.jsonl` - HNSW filtered search metrics
- `filtered_search_acorn_results.jsonl` - ACORN filtered search metrics

Each result file contains:
- Precision@1, Precision@3, Precision@5, Precision@10
- Total search time (seconds)
- QPS (queries per second)
