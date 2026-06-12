"""
cosine_sim.py
-------------
Tính Cosine Similarity thuần túy bằng numpy (không phụ thuộc DB).

Công thức:
    cos(A, B) = (A · B) / (||A|| × ||B||)

    Kết quả ∈ [-1, 1]:
        1.0  = hai vector cùng hướng hoàn toàn (giống hệt)
        0.0  = vuông góc (không liên quan)
       -1.0  = ngược chiều hoàn toàn

Dùng để verify kết quả từ pgvector hoặc tính similarity ngoài DB.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity giữa 2 vector.

    Args:
        a, b : np.ndarray shape (N,)

    Returns:
        float ∈ [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity giữa 1 query vector và nhiều vectors.

    Args:
        query  : shape (N,)
        matrix : shape (M, N) — M vectors cần so sánh

    Returns:
        np.ndarray shape (M,) — similarity của query với từng vector
    """
    query_norm  = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    if query_norm == 0:
        return np.zeros(len(matrix))
    dots  = matrix @ query
    denom = np.where(matrix_norm * query_norm == 0, 1e-10, matrix_norm * query_norm)
    return dots / denom
