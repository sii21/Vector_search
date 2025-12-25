"""Load SciFact dataset from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_corpus(corpus_path: Path) -> Dict[str, Dict[str, str]]:
    """Load corpus from JSONL file

    Returns:
        Dict mapping corpus_id -> {_id, title, text}
    """
    if not corpus_path.exists():
        msg = f"Corpus file not found: {corpus_path}"
        raise FileNotFoundError(msg)

    corpus = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            corpus_id = str(record["_id"])
            corpus[corpus_id] = {
                "_id": corpus_id,
                "title": record.get("title", ""),
                "text": record.get("text", ""),
            }
    return corpus


def load_queries(queries_path: Path) -> Dict[str, Dict[str, str]]:
    """Load queries from JSONL file

    Returns:
        Dict mapping query_id -> {_id, text}
    """
    if not queries_path.exists():
        msg = f"Queries file not found: {queries_path}"
        raise FileNotFoundError(msg)

    queries = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            query_id = str(record["_id"])
            queries[query_id] = {
                "_id": query_id,
                "text": record.get("text", ""),
            }
    return queries


def load_ground_truth(default_path: Path) -> Dict[str, List[str]]:
    """Load ground truth from default JSONL file

    Returns:
        Dict mapping query_id -> [corpus_ids with score > 0]
    """
    if not default_path.exists():
        msg = f"Default file not found: {default_path}"
        raise FileNotFoundError(msg)

    ground_truth: Dict[str, List[str]] = {}
    with default_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            query_id = str(record["query-id"])
            corpus_id = str(record["corpus-id"])
            score = record.get("score", 0)

            # Include all relevant documents (score > 0)
            if score > 0:
                if query_id not in ground_truth:
                    ground_truth[query_id] = []
                ground_truth[query_id].append(corpus_id)

    return ground_truth
