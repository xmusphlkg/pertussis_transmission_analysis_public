"""Stage age-scoped hospital cost allocation independently of model source.

Missing age ranges fail closed unless an explicitly supplied fallback is used.
Bin overlap assumes events are uniform within each model age bin.
"""
import numpy as np


class MissingAgeCostError(ValueError):
    """A cost is missing for at least one part of the model's age range."""


def allocate_hospital_cost(bin_lower, bin_upper, episode_entries, fallback=None):
    """Return (draw x age-bin costs, evidence coverage).

    episode_entries contain age_start_years, age_end_years, and one-dimensional
    draw arrays. fallback, when given, has one cost per draw and age bin. It is
    an explicit extrapolation and must be labelled by the caller.
    """
    lower = np.asarray(bin_lower, dtype=float)
    upper = np.asarray(bin_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower):
        raise ValueError('finite increasing age-bin bounds required')
    segments = []
    n_draws = None
    for entry in episode_entries:
        start = float(entry['age_start_years'])
        end_value = entry['age_end_years']
        end = np.inf if end_value is None or str(end_value).strip().lower() in {'inf', 'infinity', 'null', 'none', ''} else float(end_value)
        draws = np.asarray(entry['draws'], dtype=float)
        if not np.isfinite(start) or np.isnan(end) or start < 0 or end <= start or draws.ndim != 1 or not len(draws) or not np.isfinite(draws).all() or np.any(draws < 0):
            raise ValueError('invalid age-scoped hospital cost')
        if n_draws is not None and n_draws != len(draws):
            raise ValueError('draw counts differ')
        n_draws = len(draws)
        segments.append((start, end, draws))
    segments.sort(key=lambda item: item[0])
    if any(a[1] > b[0] for a, b in zip(segments, segments[1:])):
        raise ValueError('overlapping age-specific episode costs')
    if n_draws is None:
        if fallback is None:
            raise ValueError('no hospital costs')
        n_draws = np.asarray(fallback).shape[0]
    coverage = np.zeros(len(lower))
    costs = np.zeros((n_draws, len(lower)))
    for start, end, draws in segments:
        weight = np.maximum(0., np.minimum(upper, end) - np.maximum(lower, start)) / (upper - lower)
        coverage += weight
        costs += draws[:, None] * weight[None, :]
    uncovered = np.maximum(0., 1. - coverage)
    if np.any(uncovered > 1e-12):
        if fallback is None:
            raise MissingAgeCostError('age ranges lack a hospital cost; explicit fallback required')
        values = np.asarray(fallback, dtype=float)
        if values.shape != costs.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('invalid age-specific fallback')
        costs += values * uncovered[None, :]
    return costs, coverage
