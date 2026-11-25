"""Joint age/reporting allocation of monthly onset-bin projections.

The input is onset incidence accumulated by model bin during complete calendar
months.  The model's cohort-age composition is evaluated at each month midpoint;
events are not reconstructed daily and the cohort's within-bin age distribution
is assumed to follow ``AgeGrid.completed_age_weights``.
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from ..economics.evaluate import BAND_BOUNDS
from .aggregation import AGE_GROUPS


def monthly_reporting_weights(grid, logits, month_starts, month_ends, *, groups=AGE_GROUPS):
    """Return joint reported-case/group weights, shape (month, bin, group)."""
    if not groups:
        raise ValueError('at least one age group is required')
    for lo, hi in groups.values():
        if (not np.isfinite(lo) or lo < 0 or lo > 70 or lo != int(lo)
                or (np.isfinite(hi) and (hi > 70 or hi != int(hi)))
                or not lo < hi):
            raise ValueError('groups require completed-year boundaries at or below 70 or an open upper bound')
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (len(BAND_BOUNDS),) or not np.isfinite(logits).all():
        raise ValueError('six finite reporting logits are required')
    starts = pd.DatetimeIndex(month_starts)
    ends = pd.DatetimeIndex(month_ends)
    if (len(starts) == 0 or len(starts) != len(ends) or starts.isna().any() or ends.isna().any()
            or starts.tz is not None or ends.tz is not None or (starts.day != 1).any()
            or not (starts + pd.offsets.MonthBegin(1)).equals(ends)
            or not starts[1:].equals(ends[:-1])):
        raise ValueError('consecutive complete calendar months are required')
    ages = np.arange(101)
    report = np.column_stack([(ages >= lo) & (ages < hi) for lo, hi in BAND_BOUNDS]) @ expit(logits)
    memberships = np.column_stack([(ages >= lo) & (ages < hi) for lo, hi in groups.values()])
    per_age = report[:, None] * memberships
    weights = []
    for start, end in zip(starts, ends):
        first = pd.Timestamp(year=start.year, month=1, day=1)
        following = pd.Timestamp(year=start.year + 1, month=1, day=1)
        fraction = ((start - first).days + (end - start).days / 2) / (following - first).days
        weights.append(grid.completed_age_weights(fraction) @ per_age)
    return np.stack(weights)


def monthly_reported_cases(typical, mild, mild_reporting_ratio, weights):
    """Map nonnegative monthly bin onsets to reported cases by age group."""
    typical, mild, weights = np.asarray(typical, float), np.asarray(mild, float), np.asarray(weights, float)
    if (typical.ndim != 2 or mild.shape != typical.shape or weights.ndim != 3
            or weights.shape[:2] != typical.shape or weights.shape[2] == 0
            or not np.isfinite(typical).all() or not np.isfinite(mild).all()
            or not np.isfinite(weights).all() or (typical < 0).any() or (mild < 0).any()
            or (weights < 0).any() or (weights > 1).any()
            or not np.isfinite(mild_reporting_ratio) or not 0 <= mild_reporting_ratio <= 1):
        raise ValueError('invalid monthly onsets, reporting ratio or weights')
    return np.einsum('mb,mbg->mg', typical + mild_reporting_ratio * mild, weights)
