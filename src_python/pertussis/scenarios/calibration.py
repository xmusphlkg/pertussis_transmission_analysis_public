"""Observation-aware scoring for prospective scenario ensembles.

The transmission model works with latent typical and mild onsets, whereas the
available surveillance streams contain reported cases.  This module keeps
that distinction explicit while providing a lightweight calibration screen:
prior draws are scored through the existing core2026 NB2 and age-composition
observation model, then optionally reweighted with a tempered likelihood.

The screen is deliberately not a substitute for a converged joint posterior.
It is useful when the prospective intervention comparison is much cheaper than
refitting a full state-space model for every country.  The same parameter draw
is retained across all paired intervention arms, so the score changes the
ensemble weighting without breaking the counterfactual pairing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..observation import dirichlet_multinomial_nll, negative_binomial_nll


@dataclass(frozen=True)
class ObservationScore:
    """Training/holdout score for one natural-scale parameter draw."""

    total_nll: float
    age_nll: float
    train_nll: float
    holdout_nll: float
    train_case_years: tuple[int, ...]
    holdout_case_years: tuple[int, ...]
    train_age_years: tuple[int, ...]
    holdout_age_years: tuple[int, ...]

    @property
    def log_likelihood_train(self) -> float:
        return float(-self.train_nll)


def _years(values: Iterable[int] | None) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in (values or ())}))


def score_observation_model(
    inference: Any,
    values: dict[str, float],
    *,
    train_start_year: int | None = None,
    train_end_year: int | None = None,
    holdout_years: Iterable[int] = (2019,),
) -> ObservationScore:
    """Score a draw using the existing core2026 observation process.

    ``inference._components`` supplies the full expected case and age-group
    means.  The reported case stream is split by calendar year for a simple
    temporal holdout.  The fit-period score is used for optional weighting;
    held-out years are retained as an out-of-sample diagnostic and never enter
    that weight.
    """

    sim = inference._simulation(values)
    parts = inference._components(values, sim)
    case_years = np.asarray(inference._interval_years, dtype=int)
    holdout = set(int(year) for year in holdout_years)
    lower = -np.inf if train_start_year is None else int(train_start_year)
    upper = np.inf if train_end_year is None else int(train_end_year)
    train_mask = (case_years >= lower) & (case_years <= upper) & ~np.isin(case_years, list(holdout))
    holdout_mask = np.isin(case_years, list(holdout))
    observed = np.asarray(inference._observed_cases, dtype=float)
    expected = np.asarray(parts["expected_intervals"], dtype=float)
    size = float(values["nb_size"]) * np.asarray(inference._nb_exposure, dtype=float)
    if train_mask.any():
        train_case_nll = negative_binomial_nll(observed[train_mask], expected[train_mask], size[train_mask])
    else:
        train_case_nll = 0.0
    if holdout_mask.any():
        holdout_case_nll = negative_binomial_nll(observed[holdout_mask], expected[holdout_mask], size[holdout_mask])
    else:
        holdout_case_nll = 0.0
    age_train_nll = 0.0
    age_holdout_nll = 0.0
    train_age_years: list[int] = []
    holdout_age_years: list[int] = []
    for item, expected_counts in zip(inference.age_years, parts["expected_age"]):
        year = int(item.year)
        observed_counts = np.asarray(item.counts, dtype=float)
        probability = np.asarray(expected_counts, dtype=float)
        probability = probability / max(float(probability.sum()), 1e-300)
        nll = dirichlet_multinomial_nll(observed_counts, probability, values["dirichlet_concentration"])
        if year in holdout:
            age_holdout_nll += float(nll)
            holdout_age_years.append(year)
        elif lower <= year <= upper:
            age_train_nll += float(nll)
            train_age_years.append(year)
    train_nll = float(train_case_nll + age_train_nll)
    holdout_nll = float(holdout_case_nll + age_holdout_nll)
    return ObservationScore(
        total_nll=float(parts["total_nll"]),
        age_nll=float(parts["age_nll"]),
        train_nll=train_nll,
        holdout_nll=holdout_nll,
        train_case_years=tuple(sorted(set(case_years[train_mask].tolist()))),
        holdout_case_years=tuple(sorted(set(case_years[holdout_mask].tolist()))),
        train_age_years=tuple(sorted(set(train_age_years))),
        holdout_age_years=tuple(sorted(set(holdout_age_years))),
    )


def normalized_tempered_weights(scores: Iterable[ObservationScore], power: float = 0.5) -> np.ndarray:
    """Return normalized likelihood weights for a prior ensemble.

    ``power=1`` is ordinary prior-to-posterior importance weighting.  A value
    below one is a tempered calibration screen, useful when a small prior
    ensemble would otherwise collapse onto one draw.  The power is recorded in
    the run manifest and has no interpretation as a posterior probability.
    """

    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be positive and finite")
    values = np.asarray([-power * float(score.train_nll) for score in scores], dtype=float)
    if values.size == 0:
        raise ValueError("at least one score is required")
    if not np.isfinite(values).any():
        raise ValueError("no finite likelihood score")
    finite = np.isfinite(values)
    weights = np.zeros(values.size, dtype=float)
    weights[finite] = np.exp(values[finite] - logsumexp(values[finite]))
    return weights


def effective_sample_size(weights: Iterable[float]) -> float:
    values = np.asarray(list(weights), dtype=float)
    if values.size == 0 or np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    normalized = values / total
    return float(1.0 / np.sum(normalized**2))


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if x.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    return float(np.sum(x * w) / total)

