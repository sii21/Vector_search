"""
NumPy-backed vector distance and similarity functions
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _ensure_2d(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Normalize input to a 2D NumPy array

    Args:
        arr: Input array-like to validate and convert

    Returns:
        A tuple (array_2d, was_1d)

    Raises:
        ValueError: If the input has dimensionality other than 1 or 2
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    msg = f"Expected 1D or 2D array, got shape {arr.shape}"
    raise ValueError(msg)


def dot_product(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute dot products between queries and rows of a matrix.

    Args:
        queries: Array of shape (D,) or (Q, D) containing one or more query vectors
        matrix: Array of shape (N, D) containing N candidate vectors

    Returns:
        If queries is 1D, returns a 1D array of shape (N,) with the dot
        product between the query and each row of the matrix. If queries
        is 2D, returns a 2D array of shape (Q, N) with results for each
        query
    """
    q2d, was_1d = _ensure_2d(queries)
    m2d, _ = _ensure_2d(matrix)

    result = q2d @ m2d.T
    if was_1d:
        return result[0]
    return result


def cosine_similarity(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between queries and rows of a matrix

    Args:
        queries: Array of shape (D,) or (Q, D)
        matrix: Array of shape (N, D)

    Returns:
        Array of cosine similarities with shape (N,) for a single query
        or (Q, N) for multiple queries
    """
    q2d, was_1d = _ensure_2d(queries)
    m2d, _ = _ensure_2d(matrix)

    q_norm = np.linalg.norm(q2d, axis=1, keepdims=True)
    m_norm = np.linalg.norm(m2d, axis=1, keepdims=True)

    # Protect against division by zero
    q_norm[q_norm == 0.0] = 1.0
    m_norm[m_norm == 0.0] = 1.0

    q2d_norm = q2d / q_norm
    m2d_norm = m2d / m_norm

    result = q2d_norm @ m2d_norm.T
    if was_1d:
        return result[0]
    return result


def euclidean_distance(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute Euclidean (L2) distance between queries and rows of a matrix

    Args:
        queries: Array of shape (D,) or (Q, D)
        matrix: Array of shape (N, D)

    Returns:
        Array of Euclidean distances with shape (N,) for a single query or
        (Q, N) for multiple queries
    """
    q2d, was_1d = _ensure_2d(queries)
    m2d, _ = _ensure_2d(matrix)

    diff = q2d[:, None, :] - m2d[None, :, :]
    result = np.sqrt(np.sum(diff**2, axis=2))

    if was_1d:
        return result[0]
    return result


def manhattan_distance(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute Manhattan (L1) distance between queries and rows of a matrix

    Args:
        queries: Array of shape (D,) or (Q, D)
        matrix: Array of shape (N, D)

    Returns:
        Array of Manhattan distances with shape (N,) for a single query or
        (Q, N) for multiple queries
    """
    q2d, was_1d = _ensure_2d(queries)
    m2d, _ = _ensure_2d(matrix)

    diff = np.abs(q2d[:, None, :] - m2d[None, :, :])
    result = np.sum(diff, axis=2)

    if was_1d:
        return result[0]
    return result
