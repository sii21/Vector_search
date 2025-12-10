from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


GroundTruth = Dict[int, List[int]]


def load_ground_truth(path: Path) -> GroundTruth:
    """Load ground_truth.jsonl as {qid: [nn_ids]} dict"""
    if not path.exists():
        msg = f"ground_truth.jsonl not found at {path}"
        raise FileNotFoundError(msg)

    result: GroundTruth = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # single key per line: {"0": [...]}
            for k, v in record.items():
                qid = int(k)
                result[qid] = [int(x) for x in v]
    return result


def precision_at_k_for_query(
    true_neighbors: Sequence[int],
    retrieved: Sequence[int],
    k: int,
) -> float:
    """Precision@k for a single query"""
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for idx in top_k if idx in true_neighbors)
    return hits / float(k)


def precision_at_ks(
    ground_truth: GroundTruth,
    predictions: Dict[int, Sequence[int]],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Dict[int, float]:
    """Global Precision@k averaged over all queries"""
    ks = list(ks)
    n_queries = len(ground_truth)
    if n_queries == 0:
        return {k: 0.0 for k in ks}

    sums = {k: 0.0 for k in ks}

    for qid, true_nn in ground_truth.items():
        pred = predictions.get(qid, [])
        for k in ks:
            sums[k] += precision_at_k_for_query(true_nn, pred, k)

    return {k: sums[k] / float(n_queries) for k in ks}
