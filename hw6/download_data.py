"""Download SciFact dataset using MTEB."""

from __future__ import annotations

import json
from pathlib import Path


def download_scifact(output_dir: Path) -> None:
    """Download SciFact dataset and save to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading SciFact dataset via MTEB...")

    # Use MTEB to load SciFact dataset
    import mteb

    # Load the task
    print("Loading SciFact task...")
    task = mteb.get_tasks(tasks=["SciFact"])[0]

    # Load data
    print("Loading data...")
    task.load_data()

    # Access the data correctly from dataset['default']['test']
    print("Extracting corpus...")
    corpus_dataset = task.dataset["default"]["test"]["corpus"]

    print("Extracting queries...")
    queries_dataset = task.dataset["default"]["test"]["queries"]

    print("Extracting ground truth...")
    qrels_data = task.dataset["default"]["test"]["relevant_docs"]

    # Save corpus
    corpus_path = output_dir / "corpus.jsonl"
    print(f"Saving corpus to {corpus_path} ({len(corpus_dataset)} documents)")
    with corpus_path.open("w", encoding="utf-8") as f:
        for item in corpus_dataset:
            record = {
                "_id": str(item["id"]),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save queries
    queries_path = output_dir / "queries.jsonl"
    print(f"Saving queries to {queries_path} ({len(queries_dataset)} queries)")
    with queries_path.open("w", encoding="utf-8") as f:
        for item in queries_dataset:
            record = {
                "_id": str(item["id"]),
                "text": item.get("text", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save ground truth (qrels)
    default_path = output_dir / "default.jsonl"
    print(f"Saving ground truth to {default_path}")
    gt_count = 0
    with default_path.open("w", encoding="utf-8") as f:
        for query_id, doc_scores in qrels_data.items():
            for doc_id, score in doc_scores.items():
                if score > 0:  # Only save relevant documents
                    record = {
                        "query-id": query_id,
                        "corpus-id": doc_id,
                        "score": score,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    gt_count += 1

    print(f"\nDownloaded {len(corpus_dataset)} corpus records")
    print(f"Downloaded {len(queries_dataset)} queries")
    print(f"Downloaded {gt_count} ground truth entries")
    print("Done!")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    download_scifact(data_dir)


if __name__ == "__main__":
    main()
