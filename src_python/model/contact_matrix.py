from __future__ import annotations

import numpy as np


def reciprocity_error(contact_matrix: np.ndarray, population: np.ndarray) -> float:
    """Return the largest pairwise population-weighted reciprocity error."""
    matrix = np.asarray(contact_matrix, dtype=float)
    pop = np.asarray(population, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != pop.size:
        raise ValueError("Contact matrix and population dimensions do not match.")

    max_error = 0.0
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            flow_ij = matrix[i, j] * pop[i]
            flow_ji = matrix[j, i] * pop[j]
            denom = max(abs(flow_ij), abs(flow_ji), 1e-12)
            max_error = max(max_error, abs(flow_ij - flow_ji) / denom)
    return float(max_error)


def balance_reciprocity(contact_matrix: np.ndarray, population: np.ndarray) -> np.ndarray:
    """Population-weighted reciprocity correction for age contact matrices."""
    matrix = np.asarray(contact_matrix, dtype=float).copy()
    pop = np.asarray(population, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != pop.size:
        raise ValueError("Contact matrix and population dimensions do not match.")
    if np.any(pop <= 0.0):
        raise ValueError("Population values must be positive for reciprocity correction.")

    balanced = matrix.copy()
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            shared_contact_total = 0.5 * (matrix[i, j] * pop[i] + matrix[j, i] * pop[j])
            balanced[i, j] = shared_contact_total / pop[i]
            balanced[j, i] = shared_contact_total / pop[j]
    return np.clip(balanced, 0.0, None)
