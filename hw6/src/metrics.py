"""Evaluation metrics for hybrid search"""

from __future__ import annotations

from typing import Dict, List


def calculate_mrr(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR)

    Args:
        ground_truth: Dict mapping query_id -> [relevant_corpus_ids]
        predictions: Dict mapping query_id -> [predicted_corpus_ids]

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for query_id, relevant_ids in ground_truth.items():
        if query_id not in predictions:
            reciprocal_ranks.append(0.0)
            continue

        predicted_ids = predictions[query_id]
        relevant_set = set(relevant_ids)

        # Find the rank of the first relevant document
        for rank, pred_id in enumerate(predicted_ids, start=1):
            if pred_id in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant document found
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
