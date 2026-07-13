"""Numerical utilities for conditional state-space uncertainty propagation."""

from __future__ import annotations

import numpy as np


def bounded_multivariate_normal_draws(
    mean: np.ndarray,
    covariance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n: int,
    seed: int,
    max_proposals_per_draw: int = 500,
) -> tuple[np.ndarray, float]:
    """Rejection-sample a Gaussian approximation inside fitted bounds."""

    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    dimension = len(mean)
    if n < 1 or covariance.shape != (dimension, dimension):
        raise ValueError("Invalid bounded Gaussian draw dimensions")
    if lower.shape != mean.shape or upper.shape != mean.shape:
        raise ValueError("Bounded Gaussian limits do not match the mean vector")
    if not (
        np.isfinite(mean).all()
        and np.isfinite(covariance).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
    ):
        raise ValueError("Bounded Gaussian inputs must be finite")
    if np.any(lower >= upper) or np.any(mean < lower) or np.any(mean > upper):
        raise ValueError("Bounded Gaussian mean must lie inside ordered limits")

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-10, 1e-14)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Laplace covariance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, tolerance)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    accepted_count = 0
    proposed_count = 0
    in_bounds_count = 0
    proposal_limit = int(max(n * max_proposals_per_draw, n))
    while accepted_count < n and proposed_count < proposal_limit:
        batch_size = min(max(4 * (n - accepted_count), 256), proposal_limit - proposed_count)
        proposed = mean + rng.standard_normal((batch_size, dimension)) @ factor.T
        keep = np.logical_and(proposed >= lower, proposed <= upper).all(axis=1)
        in_bounds_count += int(np.count_nonzero(keep))
        if np.any(keep):
            block = proposed[keep][: n - accepted_count]
            accepted.append(block)
            accepted_count += len(block)
        proposed_count += batch_size
    if accepted_count < n:
        raise RuntimeError(
            "State-space Laplace approximation has negligible support inside fitted bounds: "
            f"accepted={accepted_count}, requested={n}, proposed={proposed_count}"
        )
    draws = np.vstack(accepted)[:n]
    rejection_fraction = float(1.0 - in_bounds_count / max(proposed_count, 1))
    return draws, rejection_fraction
