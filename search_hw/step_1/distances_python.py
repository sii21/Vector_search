"""
Pure-Python (list-based) distance and similarity functions
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]

SingleOrMultiVector = Union[Vector, Matrix]
Result = Union[List[float], List[List[float]]]


def _ensure_2d(queries: SingleOrMultiVector) -> Tuple[List[List[float]], bool]:
    """Normalize queries to a 2D list and indicate if input was 1D

    Returns a tuple (rows, was_1d)
    """
    try:
        first = queries[0]  # type: ignore[index]
    except Exception as exc:  # pragma: no cover - protect against empty input
        raise ValueError("queries must be non-empty") from exc

    if isinstance(first, (list, tuple)):
        return [list(row) for row in queries], False  # type: ignore[list-item]

    return [list(queries)], True  # type: ignore[arg-type]


def _matrix_to_list(matrix: Matrix) -> List[List[float]]:
    """Convert matrix-like input to a non-empty list of lists"""
    rows: List[List[float]] = [list(row) for row in matrix]
    if not rows:
        raise ValueError("matrix must be non-empty")
    return rows


def _dot(u: Vector, v: Vector) -> float:
    """Dot product of two vectors (raises on length mismatch)"""
    if len(u) != len(v):
        msg = f"Vectors must have same length, got {len(u)} and {len(v)}"
        raise ValueError(msg)
    return sum(float(a) * float(b) for a, b in zip(u, v))


def _norm(u: Vector) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(float(a) * float(a) for a in u))


def dot_product(queries: SingleOrMultiVector, matrix: Matrix) -> Result:
    """Compute dot products between queries and matrix rows"""
    q2d, was_1d = _ensure_2d(queries)
    m2d = _matrix_to_list(matrix)

    result: List[List[float]] = []
    for q in q2d:
        row: List[float] = []
        for m in m2d:
            row.append(_dot(q, m))
        result.append(row)

    if was_1d:
        return result[0]
    return result


def cosine_similarity(queries: SingleOrMultiVector, matrix: Matrix) -> Result:
    """Compute cosine similarity between queries and matrix rows"""
    q2d, was_1d = _ensure_2d(queries)
    m2d = _matrix_to_list(matrix)

    m_norms: List[float] = []
    for m in m2d:
        n = _norm(m)
        if n == 0.0:
            n = 1.0
        m_norms.append(n)

    result: List[List[float]] = []
    for q in q2d:
        qn = _norm(q)
        if qn == 0.0:
            qn = 1.0
        row: List[float] = []
        for m, mn in zip(m2d, m_norms):
            row.append(_dot(q, m) / (qn * mn))
        result.append(row)

    if was_1d:
        return result[0]
    return result


def euclidean_distance(queries: SingleOrMultiVector, matrix: Matrix) -> Result:
    """Compute Euclidean (L2) distances between queries and matrix rows"""
    q2d, was_1d = _ensure_2d(queries)
    m2d = _matrix_to_list(matrix)

    result: List[List[float]] = []
    for q in q2d:
        row: List[float] = []
        for m in m2d:
            if len(q) != len(m):
                msg = f"Vectors must have same length, got {len(q)} and {len(m)}"
                raise ValueError(msg)
            s = 0.0
            for a, b in zip(q, m):
                diff = float(a) - float(b)
                s += diff * diff
            row.append(math.sqrt(s))
        result.append(row)

    if was_1d:
        return result[0]
    return result


def manhattan_distance(queries: SingleOrMultiVector, matrix: Matrix) -> Result:
    """Compute Manhattan (L1) distances between queries and matrix rows"""
    q2d, was_1d = _ensure_2d(queries)
    m2d = _matrix_to_list(matrix)

    result: List[List[float]] = []
    for q in q2d:
        row: List[float] = []
        for m in m2d:
            if len(q) != len(m):
                msg = f"Vectors must have same length, got {len(q)} and {len(m)}"
                raise ValueError(msg)
            s = 0.0
            for a, b in zip(q, m):
                s += abs(float(a) - float(b))
            row.append(s)
        result.append(row)

    if was_1d:
        return result[0]
    return result
