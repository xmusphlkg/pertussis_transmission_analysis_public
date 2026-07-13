"""Leakage-safe rolling validation for the semi-mechanistic panel POMP.

The deterministic compartment model supplies interval-specific count offsets.
A country static source scale and an irregular-time latent discrepancy are then
fit only to observations strictly before each forecast origin. Candidate
process specifications and ensemble weights are selected on the three
preceding untouched calendar years; the outer test year is never used for
fitting or selection.

This is a notification-index forecast, not a full stochastic compartment model
and not an absolute infection-burden estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
from math import exp, log
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import nbinom, norm

from src_python.calibration.calibrate_baseline import (
    _calibration_predicted_means,
    calibration_runtime_config,
    observed_annual_case_frame,
)
from src_python.calibration.panel_pomp import (
    CountrySeries,
    DiscrepancyProcessSpec,
    ObservationSpec,
    PanelCandidate,
    ParticleFilterConfig,
    ScaleSearchConfig,
    bootstrap_particle_filter,
    fit_country_static_log_scale,
    nb2_logpmf,
    select_panel_candidate,
)
from src_python.simulation.common import (
    current_run_metadata,
    file_sha256,
    make_config,
    publication_country_names,
    write_run_metadata,
)
from src_python.utils.io import ensure_output_dirs, project_path, write_dataframe
from src_python.validation.audit_observation_sources import (
    build_observation_source_audit,
    country_source_summary,
)


PREQUENTIAL_RUN_STEM = "panel_pomp_rolling_hindcast"
BLOCK_STRESS_RUN_STEM = "panel_pomp_block_stress"
MINIMUM_TRAINING_YEARS = 5
MINIMUM_COUNTRIES = 8
MINIMUM_FOLDS_PER_COUNTRY = 3
MINIMUM_PREDICTIVE_COVERAGE = 0.85
MAXIMUM_PREDICTIVE_COVERAGE = 0.995
MAXIMUM_MC_LOG_LIKELIHOOD_SD_PER_INTERVAL = 0.15
MAXIMUM_MEAN_PREDICTIVE_LOG1P_WIDTH = float(np.log(100.0))
COUNTRY_CANDIDATE_PRIOR_FOLDS = 3.0
# Reserved for the audited PIT recalibration experiment. The main hindcast
# intentionally leaves this layer disabled because it failed the outer folds.
PIT_COUNTRY_PRIOR_INTERVALS = 24.0
PIT_GLOBAL_PRIOR_COUNTRIES = 3.0
COMPONENT_NAMES = ("pomp", "seasonal_naive", "ew_recent_rate", "damped_log_trend")
BASELINE_OBSERVATION = ObservationSpec(dispersion_per_reference_exposure=30.0)


@lru_cache(maxsize=1)
def _mechanistic_offset_source_fingerprint() -> str:
    """Hash only code that can change deterministic mechanistic offsets."""

    paths = [
        project_path("src_python/simulation/common.py"),
        project_path("src_python/calibration/calibrate_baseline.py"),
        *sorted(project_path("src_python/model").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_path())).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def default_candidates() -> tuple[PanelCandidate, ...]:
    search = ScaleSearchConfig(
        method="count_ratio",
        lower=-10.0,
        upper=10.0,
        grid_points=7,
        refinement_steps=1,
        center_on_training_log_ratio=True,
        local_half_width=2.0,
    )
    processes = {
        "regular_ar1": DiscrepancyProcessSpec(rho=0.50, innovation_sd=0.45),
        "persistent_ar1": DiscrepancyProcessSpec(rho=0.80, innovation_sd=0.50),
        "robust_ar1": DiscrepancyProcessSpec(
                rho=0.70,
                innovation_sd=0.50,
                robust_mixture_probability=0.05,
                robust_mixture_scale=4.0,
        ),
        "jump_ar1": DiscrepancyProcessSpec(
                rho=0.70,
                innovation_sd=0.35,
                jump_rate_per_year=0.35,
                jump_sd=1.25,
        ),
        "damped_trend": DiscrepancyProcessSpec(
                rho=0.80,
                innovation_sd=0.35,
                trend_enabled=True,
                trend_damping=0.50,
                trend_innovation_sd=0.20,
                trend_coupling=0.50,
        ),
    }
    return tuple(
        PanelCandidate(
            name=f"{process_name}__k{int(dispersion)}",
            process=process,
            observation=ObservationSpec(
                dispersion_per_reference_exposure=float(dispersion)
            ),
            scale_search=search,
        )
        for process_name, process in processes.items()
        for dispersion in (10.0, 30.0, 100.0, 300.0, 1000.0)
    )


def _native_fold_observations(
    country: str,
    test_year: int,
    *,
    minimum_training_years: int,
) -> pd.DataFrame | None:
    observed = observed_annual_case_frame(country).copy()
    required = {"period_start", "period_end", "reported_cases", "observed_interval_id"}
    if observed.empty or not required.issubset(observed.columns):
        return None
    observed["period_start"] = pd.to_datetime(observed["period_start"], errors="coerce")
    observed["period_end"] = pd.to_datetime(observed["period_end"], errors="coerce")
    observed["reported_cases"] = pd.to_numeric(observed["reported_cases"], errors="coerce")
    observed = observed.dropna(subset=["period_start", "period_end", "reported_cases"]).copy()
    cutoff = pd.Timestamp(year=int(test_year), month=1, day=1)
    next_cutoff = pd.Timestamp(year=int(test_year) + 1, month=1, day=1)
    training = observed.loc[observed["period_end"].le(cutoff)].copy()
    holdout = observed.loc[
        observed["period_start"].ge(cutoff)
        & observed["period_end"].le(next_cutoff)
    ].copy()
    if training.empty or holdout.empty:
        return None
    training_midpoint = training["period_start"] + (
        training["period_end"] - training["period_start"]
    ) / 2
    if int(training_midpoint.dt.year.nunique()) < int(minimum_training_years):
        return None
    training["training_interval"] = True
    holdout["training_interval"] = False
    combined = pd.concat([training, holdout], ignore_index=True)
    combined = combined.sort_values(["period_start", "period_end"]).reset_index(drop=True)
    combined["series_year"] = np.arange(len(combined), dtype=int)
    combined["observed_year"] = (
        combined["period_start"] + (combined["period_end"] - combined["period_start"]) / 2
    ).dt.year.astype(int)
    combined["interval_days"] = (
        combined["period_end"] - combined["period_start"]
    ).dt.days.astype(float)
    return combined


def _build_country_series(
    country: str,
    test_year: int,
    *,
    minimum_training_years: int,
    mechanistic_source_hash: str,
) -> tuple[CountrySeries, dict[str, Any]] | None:
    observed = _native_fold_observations(
        country,
        test_year,
        minimum_training_years=minimum_training_years,
    )
    if observed is None:
        return None
    cutoff = f"{int(test_year):04d}-01-01"
    base = make_config(
        country_profile=country,
        load_calibration=False,
        evidence_cutoff_date=cutoff,
        config_overrides={
            # Retrospectively coded diagnostic regimes lack available-from
            # dates and are not admissible in a genuine forecast.
            "observation_model": {"diagnostic_standards": {"enabled": False}},
            "transmission": {
                "npi_contact_reduction": {
                    "enabled": True,
                    "reduction_column": "contact_reduction_mean",
                }
            },
        },
    )
    runtime = calibration_runtime_config(base, observed)
    cache_payload = {
        "country": country,
        "test_year": int(test_year),
        "runtime": runtime,
        "intervals": observed[
            ["observed_interval_id", "period_start", "period_end"]
        ].astype(str).to_dict(orient="records"),
        "mechanistic_source_code_hash": str(mechanistic_source_hash),
    }
    cache_key = hashlib.sha256(
        json.dumps(
            cache_payload,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_directory = project_path("outputs/metadata/panel_pomp_offset_cache")
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"{country}_{int(test_year)}_{cache_key}.npz"
    cache_hit = cache_path.exists()
    if cache_hit:
        with np.load(cache_path, allow_pickle=False) as cached:
            offsets = np.asarray(cached["offsets"], dtype=float)
    else:
        offsets = np.asarray(
            _calibration_predicted_means(runtime, observed, country), dtype=float
        )
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, offsets=offsets)
        temporary.replace(cache_path)
    if offsets.shape != (len(observed),) or not np.isfinite(offsets).all():
        raise RuntimeError(f"Invalid mechanistic offsets for {country} {test_year}")
    offsets = np.maximum(offsets, 1e-8)
    series = CountrySeries(
        country=country,
        interval_ids=observed["observed_interval_id"].astype(str).tolist(),
        interval_start=observed["period_start"].tolist(),
        interval_end=observed["period_end"].tolist(),
        observed_counts=observed["reported_cases"].to_numpy(dtype=float),
        mechanistic_offset=offsets,
        exposure_days=observed["interval_days"].to_numpy(dtype=float),
        training_mask=observed["training_interval"].to_numpy(dtype=bool),
    )
    metadata = base.get("metadata", {})
    audit = {
        "country": country,
        "test_year": int(test_year),
        "evidence_cutoff_date": cutoff,
        "training_intervals": int(series.n_training),
        "holdout_intervals": int(series.n_intervals - series.n_training),
        "training_years": int(observed.loc[observed["training_interval"], "observed_year"].nunique()),
        "future_evidence_used": False,
        "outcome_derived_profile_terms_disabled": bool(
            metadata.get("outcome_derived_profile_terms_disabled", False)
        ),
        "historical_dtp_coverage_year": int(metadata["historical_dtp_coverage_year"]),
        "historical_maternal_coverage_excluded": bool(
            metadata.get("historical_maternal_coverage_excluded", False)
        ),
        "npi_reduction_column": str(
            metadata.get("npi_contact_reduction_reduction_column", "")
        ),
        "diagnostic_timeline_enabled": bool(
            runtime.get("observation_model", {})
            .get("diagnostic_standards", {})
            .get("enabled", False)
        ),
        "mechanistic_offset_cache_key": cache_key,
        "mechanistic_offset_cache_hit": bool(cache_hit),
    }
    return series, audit


def _log_rate_baselines(
    series: CountrySeries,
    *,
    prequential: bool = False,
) -> dict[str, np.ndarray]:
    training = np.asarray(series.training_mask, dtype=bool)
    midpoint = np.asarray(series.midpoint_days, dtype=float)
    cases = np.asarray(series.observed_counts, dtype=float)
    exposure = np.asarray(series.exposure_days, dtype=float)
    test_indices = np.flatnonzero(~training)
    output = {
        "seasonal_naive": np.zeros(len(test_indices), dtype=float),
        "ew_recent_rate": np.zeros(len(test_indices), dtype=float),
        "damped_log_trend": np.zeros(len(test_indices), dtype=float),
    }
    for position, index in enumerate(test_indices):
        history = (
            np.arange(len(midpoint)) < int(index)
            if prequential
            else training
        )
        train_mid = midpoint[history]
        train_rate = (cases[history] + 0.5) / exposure[history]
        age = midpoint[index] - train_mid
        seasonal_index = int(np.argmin(np.abs(age - 365.2425)))
        output["seasonal_naive"][position] = (
            train_rate[seasonal_index] * exposure[index]
        )

        recent = age <= 3.0 * 365.2425
        if not bool(recent.any()):
            recent = np.ones(len(age), dtype=bool)
        recent_age = age[recent]
        recent_log_rate = np.log(train_rate[recent])
        weights = np.exp(-np.log(2.0) * recent_age / 180.0)
        level = float(np.sum(weights * recent_log_rate) / np.sum(weights))
        output["ew_recent_rate"][position] = exp(level) * exposure[index]

        x = (train_mid[recent] - train_mid[recent].max()) / 365.2425
        if len(x) >= 3 and float(np.ptp(x)) > 0.0:
            design = np.column_stack([np.ones(len(x)), x])
            root_w = np.sqrt(weights / np.max(weights))
            coefficient, *_ = np.linalg.lstsq(
                design * root_w[:, None], recent_log_rate * root_w, rcond=None
            )
            slope = float(np.clip(coefficient[1], -2.0, 2.0))
            horizon = float((midpoint[index] - train_mid[recent].max()) / 365.2425)
            trend_log_rate = float(coefficient[0] + 0.5 * slope * horizon)
        else:
            trend_log_rate = level
        output["damped_log_trend"][position] = exp(trend_log_rate) * exposure[index]
    return {name: np.maximum(values, 1e-8) for name, values in output.items()}


def _nb_forecast_metrics(observed: float, mean: float, size: float) -> tuple[float, float, float]:
    probability = float(size / (size + mean))
    lower = float(nbinom.ppf(0.025, size, probability))
    upper = float(nbinom.ppf(0.975, size, probability))
    score = float(nb2_logpmf(observed, mean, size))
    return score, lower, upper


def _candidate_by_name(
    candidates: Iterable[PanelCandidate], name: str
) -> PanelCandidate:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise KeyError(name)


def optimize_country_balanced_stacking_weights(
    log_scores: np.ndarray,
    countries: np.ndarray,
    *,
    minimum_weight: float = 0.02,
) -> dict[str, float]:
    """Fit a regularized linear pool using only inner-validation outcomes."""

    scores = np.asarray(log_scores, dtype=float)
    labels = np.asarray(countries, dtype=str)
    if scores.ndim != 2 or scores.shape[1] != len(COMPONENT_NAMES):
        raise ValueError("log_scores must have one column per ensemble component")
    if labels.shape != (scores.shape[0],) or not np.isfinite(scores).all():
        raise ValueError("stacking inputs are incomplete")
    floor = float(minimum_weight)
    if not 0.0 <= floor < 1.0 / len(COMPONENT_NAMES):
        raise ValueError("minimum_weight is incompatible with the component count")
    country_indices = [np.flatnonzero(labels == label) for label in sorted(set(labels))]

    def objective(weight: np.ndarray) -> float:
        mixture = logsumexp(np.log(weight)[None, :] + scores, axis=1)
        return -float(np.mean([np.mean(mixture[index]) for index in country_indices]))

    n = len(COMPONENT_NAMES)
    starts = [np.full(n, 1.0 / n)]
    for index in range(n):
        start = np.full(n, floor)
        start[index] = 1.0 - floor * (n - 1)
        starts.append(start)
    results = [
        minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(floor, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda weight: float(np.sum(weight) - 1.0)},
            options={"ftol": 1e-10, "maxiter": 500},
        )
        for start in starts
    ]
    valid = [result for result in results if result.success and np.isfinite(result.fun)]
    if not valid:
        raise RuntimeError("Country-balanced stacking optimization failed")
    selected = min(valid, key=lambda result: float(result.fun))
    weight = np.asarray(selected.x, dtype=float)
    weight /= float(np.sum(weight))
    return {name: float(value) for name, value in zip(COMPONENT_NAMES, weight)}


def optimize_country_balanced_point_weights(
    component_predictions: np.ndarray,
    observed: np.ndarray,
    countries: np.ndarray,
) -> dict[str, float]:
    """Fit a geometric pool for absolute log1p error."""

    predictions = np.asarray(component_predictions, dtype=float)
    outcome = np.asarray(observed, dtype=float)
    labels = np.asarray(countries, dtype=str)
    if predictions.ndim != 2 or predictions.shape[1] != len(COMPONENT_NAMES):
        raise ValueError("component_predictions have the wrong shape")
    if outcome.shape != (len(predictions),) or labels.shape != outcome.shape:
        raise ValueError("point-stacking inputs are incomplete")
    if np.any(predictions <= 0.0) or np.any(outcome < 0.0):
        raise ValueError("point-stacking counts must be non-negative")
    country_indices = [np.flatnonzero(labels == label) for label in sorted(set(labels))]

    def objective(weight: np.ndarray) -> float:
        point_log = np.log1p(predictions) @ weight
        errors = np.abs(point_log - np.log1p(outcome))
        return float(np.mean([np.mean(errors[index]) for index in country_indices]))

    n = len(COMPONENT_NAMES)
    starts = [np.full(n, 1.0 / n), *np.eye(n)]
    results = [
        minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda weight: float(np.sum(weight) - 1.0)},
            options={"ftol": 1e-10, "maxiter": 500},
        )
        for start in starts
    ]
    valid = [result for result in results if result.success and np.isfinite(result.fun)]
    if not valid:
        raise RuntimeError("Country-balanced point stacking optimization failed")
    weight = np.asarray(min(valid, key=lambda result: float(result.fun)).x, dtype=float)
    weight /= float(np.sum(weight))
    return {name: float(value) for name, value in zip(COMPONENT_NAMES, weight)}


def select_shrunk_country_candidates(
    selection: Any,
    *,
    prior_folds: float = COUNTRY_CANDIDATE_PRIOR_FOLDS,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Select country candidates with explicit shrinkage to the panel score.

    Each validation year contributes one country-level mean log score, so
    weekly sources cannot dominate monthly sources merely by reporting more
    often. The pseudo-fold prior stabilizes selection when only a few rolling
    validation years are available.
    """

    prior = float(prior_folds)
    if not np.isfinite(prior) or prior < 0.0:
        raise ValueError("prior_folds must be finite and non-negative")
    candidate_country_scores: dict[str, dict[str, tuple[float, int, int]]] = {}
    countries: set[str] = set()
    for result in selection.candidates:
        grouped: dict[str, list[tuple[float, int]]] = {}
        for fit in result.country_fits:
            base_country = str(fit.country).split("__validation_", 1)[0]
            forecasts = [
                forecast
                for forecast in fit.filter_result.forecasts
                if not forecast.training_interval
            ]
            scores = np.asarray(
                [forecast.predictive_log_score for forecast in forecasts],
                dtype=float,
            )
            if scores.size < 1 or not np.isfinite(scores).all():
                raise ValueError("Country candidate validation scores are incomplete")
            grouped.setdefault(base_country, []).append(
                (float(np.mean(scores)), int(scores.size))
            )
            countries.add(base_country)
        candidate_country_scores[result.name] = {
            country: (
                float(np.mean([item[0] for item in values])),
                int(len(values)),
                int(sum(item[1] for item in values)),
            )
            for country, values in grouped.items()
        }

    selected: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    for country in sorted(countries):
        country_rows: list[dict[str, Any]] = []
        for result in selection.candidates:
            local_score, fold_count, interval_count = candidate_country_scores[
                result.name
            ][country]
            local_weight = float(fold_count / (fold_count + prior))
            shrunk_score = float(
                local_weight * local_score
                + (1.0 - local_weight)
                * result.country_balanced_validation_log_score
            )
            country_rows.append(
                {
                    "country": country,
                    "candidate": result.name,
                    "local_validation_log_score": local_score,
                    "global_validation_log_score": float(
                        result.country_balanced_validation_log_score
                    ),
                    "country_validation_folds": fold_count,
                    "country_validation_intervals": interval_count,
                    "local_score_weight": local_weight,
                    "shrunk_selection_score": shrunk_score,
                }
            )
        winner = max(
            country_rows,
            key=lambda row: (float(row["shrunk_selection_score"]), row["candidate"]),
        )["candidate"]
        selected[country] = str(winner)
        for row in country_rows:
            row["selected"] = bool(row["candidate"] == winner)
            details.append(row)
    return selected, details


def _inner_stacking_weights(
    panel: list[CountrySeries],
    selection: Any,
    *,
    prequential: bool,
    selected_by_country: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    candidate_results = {result.name: result for result in selection.candidates}
    if selected_by_country is None:
        selected_by_country = {
            series.country.split("__validation_", 1)[0]: selection.selected_name
            for series in panel
        }
    fit_by_candidate_and_series = {
        result.name: {fit.country: fit for fit in result.country_fits}
        for result in selection.candidates
    }
    score_rows: list[list[float]] = []
    country_rows: list[str] = []
    point_rows: list[list[float]] = []
    point_observed: list[float] = []
    point_countries: list[str] = []
    for series in panel:
        base_country = series.country.split("__validation_", 1)[0]
        candidate_name = selected_by_country[base_country]
        if candidate_name not in candidate_results:
            raise KeyError(f"Unknown selected candidate {candidate_name!r}")
        fit = fit_by_candidate_and_series[candidate_name][series.country]
        forecasts = [item for item in fit.filter_result.forecasts if not item.training_interval]
        baselines = _log_rate_baselines(series, prequential=prequential)
        holdout = np.flatnonzero(~np.asarray(series.training_mask, dtype=bool))
        annual_observed = 0.0
        annual_components = {name: 0.0 for name in COMPONENT_NAMES}
        for position, (index, forecast) in enumerate(zip(holdout, forecasts)):
            observed = float(series.observed_counts[index])
            baseline_size = float(
                BASELINE_OBSERVATION.interval_size(series.exposure_days[index])
            )
            scores = [float(forecast.predictive_log_score)]
            for name in COMPONENT_NAMES[1:]:
                scores.append(
                    float(nb2_logpmf(observed, baselines[name][position], baseline_size))
                )
            score_rows.append(scores)
            country_rows.append(base_country)
            annual_observed += observed
            annual_components["pomp"] += float(forecast.expected_mean)
            for name in COMPONENT_NAMES[1:]:
                annual_components[name] += float(baselines[name][position])
        point_rows.append([annual_components[name] for name in COMPONENT_NAMES])
        point_observed.append(annual_observed)
        point_countries.append(base_country)

    scores_array = np.asarray(score_rows, dtype=float)
    score_country_array = np.asarray(country_rows, dtype=str)
    points_array = np.asarray(point_rows, dtype=float)
    point_observed_array = np.asarray(point_observed, dtype=float)
    point_country_array = np.asarray(point_countries, dtype=str)
    global_probability = optimize_country_balanced_stacking_weights(
        scores_array, score_country_array
    )
    global_point = optimize_country_balanced_point_weights(
        points_array, point_observed_array, point_country_array
    )
    probability_weights = {"__global__": global_probability}
    point_weights = {"__global__": global_point}
    for country in sorted(set(point_country_array)):
        score_mask = score_country_array == country
        point_mask = point_country_array == country
        local_probability = optimize_country_balanced_stacking_weights(
            scores_array[score_mask], score_country_array[score_mask]
        )
        local_point = optimize_country_balanced_point_weights(
            points_array[point_mask],
            point_observed_array[point_mask],
            point_country_array[point_mask],
        )
        probability_weights[country] = {
            name: 0.5 * global_probability[name] + 0.5 * local_probability[name]
            for name in COMPONENT_NAMES
        }
        point_weights[country] = {
            name: 0.5 * global_point[name] + 0.5 * local_point[name]
            for name in COMPONENT_NAMES
        }
    return probability_weights, point_weights


def _ensemble_predictive_draws(
    *,
    pomp_draws: np.ndarray,
    baseline_means: dict[str, float],
    component_sizes: dict[str, float],
    weights: dict[str, float],
    draws: int,
    seed: int,
    component_indices: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    names = list(COMPONENT_NAMES)
    probabilities = np.asarray([weights[name] for name in names], dtype=float)
    if component_indices is None:
        component = rng.choice(len(names), size=int(draws), p=probabilities)
    else:
        component = np.asarray(component_indices, dtype=np.int64)
        if component.shape != (int(draws),) or np.any(
            (component < 0) | (component >= len(names))
        ):
            raise ValueError("ensemble component indices are invalid")
    output = np.empty(int(draws), dtype=float)
    for component_index, name in enumerate(names):
        selected = np.flatnonzero(component == component_index)
        if not len(selected):
            continue
        if name == "pomp":
            source = np.asarray(pomp_draws, dtype=float)
            if source.size < 1:
                raise ValueError("POMP predictive draws are required for stacking")
            output[selected] = rng.choice(source, size=len(selected), replace=True)
        else:
            mean = float(baseline_means[name])
            size = float(component_sizes[name])
            probability = float(size / (size + mean))
            output[selected] = rng.negative_binomial(size, probability, size=len(selected))
    return output


def _randomized_empirical_pit(
    draws: np.ndarray,
    observed: float,
    *,
    seed: int,
) -> float:
    """Return a randomized PIT for a discrete empirical predictive law."""

    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Predictive draws are required for PIT calibration")
    if not np.isfinite(observed) or observed < 0.0:
        raise ValueError("PIT observations must be finite and non-negative")
    rng = np.random.default_rng(int(seed))
    less = float(np.count_nonzero(values < observed))
    equal = float(np.count_nonzero(values == observed))
    pit = (less + float(rng.random()) * equal) / float(values.size)
    return float(np.clip(pit, 1.0 / (2.0 * values.size), 1.0 - 1.0 / (2.0 * values.size)))


def calibrated_pit_quantile_levels(
    country_pits: dict[str, np.ndarray],
    *,
    country_prior_intervals: float = PIT_COUNTRY_PRIOR_INTERVALS,
    global_prior_countries: float = PIT_GLOBAL_PRIOR_COUNTRIES,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Estimate country PIT quantiles with two-level empirical shrinkage."""

    target = np.asarray([0.025, 0.5, 0.975], dtype=float)
    if not country_pits:
        raise ValueError("At least one country PIT sample is required")
    local_prior = float(country_prior_intervals)
    global_prior = float(global_prior_countries)
    if local_prior < 0.0 or global_prior < 0.0:
        raise ValueError("PIT shrinkage priors must be non-negative")
    raw: dict[str, np.ndarray] = {}
    for country, values in country_pits.items():
        pits = np.asarray(values, dtype=float)
        if pits.ndim != 1 or pits.size < 1 or not np.isfinite(pits).all():
            raise ValueError(f"Invalid PIT values for {country}")
        if np.any((pits <= 0.0) | (pits >= 1.0)):
            raise ValueError(f"PIT values for {country} must lie in (0, 1)")
        raw[country] = np.quantile(pits, target)
    country_mean = np.mean(np.vstack([raw[country] for country in sorted(raw)]), axis=0)
    global_weight = float(len(raw) / (len(raw) + global_prior))
    global_levels = global_weight * country_mean + (1.0 - global_weight) * target

    calibrated: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for country in sorted(raw):
        interval_count = int(np.asarray(country_pits[country]).size)
        local_weight = float(interval_count / (interval_count + local_prior))
        levels = local_weight * raw[country] + (1.0 - local_weight) * global_levels
        levels = np.clip(levels, 0.001, 0.999)
        if not bool(np.all(np.diff(levels) > 0.0)):
            raise RuntimeError(f"PIT quantile levels are not ordered for {country}")
        calibrated[country] = levels
        audit.append(
            {
                "country": country,
                "calibration_intervals": interval_count,
                "local_weight": local_weight,
                "global_weight": global_weight,
                "raw_lower_probability": float(raw[country][0]),
                "raw_median_probability": float(raw[country][1]),
                "raw_upper_probability": float(raw[country][2]),
                "global_lower_probability": float(global_levels[0]),
                "global_median_probability": float(global_levels[1]),
                "global_upper_probability": float(global_levels[2]),
                "calibrated_lower_probability": float(levels[0]),
                "calibrated_median_probability": float(levels[1]),
                "calibrated_upper_probability": float(levels[2]),
            }
        )
    return calibrated, audit


def _fit_probability_interval_calibration(
    panel: list[CountrySeries],
    *,
    candidates: tuple[PanelCandidate, ...],
    selected_by_country: dict[str, str],
    stacking_weights: dict[str, dict[str, float]],
    filter_config: ParticleFilterConfig,
    predictive_draws: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Fit PIT interval calibration on a year excluded from model selection."""

    country_pits: dict[str, list[float]] = {}
    for series in panel:
        country = series.country
        selected = _candidate_by_name(candidates, selected_by_country[country])
        fit = fit_country_static_log_scale(
            series,
            process=selected.process,
            observation=selected.observation,
            filter_config=replace(
                filter_config,
                seed=_stable_seed(seed, f"pit-fit::{country}"),
                predictive_draws=int(predictive_draws),
            ),
            search=selected.scale_search,
        )
        baselines = _log_rate_baselines(
            series,
            prequential=filter_config.assimilate_holdout_observations,
        )
        holdout_indices = np.flatnonzero(
            ~np.asarray(series.training_mask, dtype=bool)
        )
        forecasts = [
            forecast
            for forecast in fit.filter_result.forecasts
            if not forecast.training_interval
        ]
        weights = stacking_weights.get(country, stacking_weights["__global__"])
        pits: list[float] = []
        for position, (index, forecast) in enumerate(
            zip(holdout_indices, forecasts)
        ):
            exposure = float(series.exposure_days[index])
            baseline_means = {
                name: float(baselines[name][position])
                for name in COMPONENT_NAMES[1:]
            }
            draws = _ensemble_predictive_draws(
                pomp_draws=forecast.observation_draws,
                baseline_means=baseline_means,
                component_sizes={
                    "pomp": float(selected.observation.interval_size(exposure)),
                    **{
                        name: float(BASELINE_OBSERVATION.interval_size(exposure))
                        for name in COMPONENT_NAMES[1:]
                    },
                },
                weights=weights,
                draws=int(predictive_draws),
                seed=_stable_seed(
                    seed, f"pit-draws::{country}::{forecast.interval_id}"
                ),
            )
            pits.append(
                _randomized_empirical_pit(
                    draws,
                    float(series.observed_counts[index]),
                    seed=_stable_seed(
                        seed, f"pit-randomization::{country}::{forecast.interval_id}"
                    ),
                )
            )
        country_pits[country] = pits
    return calibrated_pit_quantile_levels(
        {country: np.asarray(values, dtype=float) for country, values in country_pits.items()}
    )


def _selection_rows(
    selection: Any,
    outer_test_year: int,
    stacking_weights: dict[str, float],
    point_weights: dict[str, float],
    inner_validation_years: list[int],
    country_candidate_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in selection.candidates:
        rows.append(
            {
                "outer_test_year": int(outer_test_year),
                "selection_scope": "panel_global",
                "country": "all_countries",
                "inner_validation_years": ";".join(
                    str(year) for year in inner_validation_years
                ),
                "candidate": result.name,
                "selected": result.name == selection.selected_name,
                "training_log_likelihood": result.training_log_likelihood,
                "validation_log_score": result.validation_log_score,
                "country_balanced_validation_log_score": (
                    result.country_balanced_validation_log_score
                ),
                "validation_interval_count": result.validation_interval_count,
                "selection_score": result.selection_score,
                "selection_criterion": result.selection_criterion,
                **{
                    f"stacking_weight_{name}": float(stacking_weights[name])
                    for name in COMPONENT_NAMES
                },
                **{
                    f"point_weight_{name}": float(point_weights[name])
                    for name in COMPONENT_NAMES
                },
            }
        )
    for detail in country_candidate_details:
        rows.append(
            {
                "outer_test_year": int(outer_test_year),
                "selection_scope": "country_shrunk_to_panel",
                "inner_validation_years": ";".join(
                    str(year) for year in inner_validation_years
                ),
                "training_log_likelihood": float("nan"),
                "validation_log_score": detail["local_validation_log_score"],
                "country_balanced_validation_log_score": detail[
                    "global_validation_log_score"
                ],
                "validation_interval_count": detail[
                    "country_validation_intervals"
                ],
                "selection_score": detail["shrunk_selection_score"],
                "selection_criterion": "country_mean_log_score_shrunk_to_panel",
                **detail,
                **{
                    f"stacking_weight_{name}": float(stacking_weights[name])
                    for name in COMPONENT_NAMES
                },
                **{
                    f"point_weight_{name}": float(point_weights[name])
                    for name in COMPONENT_NAMES
                },
            }
        )
    return rows


def run_hindcast(
    countries: list[str],
    test_years: list[int],
    *,
    particles: int,
    predictive_draws: int,
    seed: int,
    mc_replicates: int,
    n_jobs: int,
    forecast_mode: str,
    minimum_training_years: int = MINIMUM_TRAINING_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = default_candidates()
    forecast_mode = str(forecast_mode).lower()
    if forecast_mode not in {"prequential", "block"}:
        raise ValueError("forecast_mode must be prequential or block")
    prequential = forecast_mode == "prequential"
    mechanistic_source_hash = _mechanistic_offset_source_fingerprint()
    cache: dict[tuple[str, int], tuple[CountrySeries, dict[str, Any]] | None] = {}

    def ensure_year(year: int) -> None:
        missing = [country for country in countries if (country, int(year)) not in cache]
        if not missing:
            return
        print(
            f"[panel-pomp] origin {int(year)}: resolving {len(missing)} country offsets "
            f"with n_jobs={int(n_jobs)}",
            flush=True,
        )
        if int(n_jobs) == 1:
            results = [
                _build_country_series(
                    country,
                    int(year),
                    minimum_training_years=minimum_training_years,
                    mechanistic_source_hash=mechanistic_source_hash,
                )
                for country in missing
            ]
        else:
            results = Parallel(
                n_jobs=int(n_jobs),
                backend="loky",
                inner_max_num_threads=1,
            )(
                delayed(_build_country_series)(
                    country,
                    int(year),
                    minimum_training_years=minimum_training_years,
                    mechanistic_source_hash=mechanistic_source_hash,
                )
                for country in missing
            )
        for country, result in zip(missing, results):
            cache[(country, int(year))] = result

    interval_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    base_filter = ParticleFilterConfig(
        n_particles=int(particles),
        predictive_draws=int(predictive_draws),
        seed=int(seed),
        ess_resample_fraction=0.5,
        assimilate_holdout_observations=prequential,
    )

    for test_year in sorted(set(int(year) for year in test_years)):
        selection_years = list(range(test_year - 3, test_year))
        for inner_year in selection_years:
            ensure_year(inner_year)
        inner_panel: list[CountrySeries] = []
        for inner_year in selection_years:
            for country in countries:
                built = cache[(country, inner_year)]
                if built is not None:
                    inner_panel.append(
                        replace(
                            built[0],
                            country=f"{country}__validation_{inner_year}",
                        )
                    )
        if len(inner_panel) < MINIMUM_COUNTRIES:
            raise RuntimeError(
                f"Only {len(inner_panel)} countries support inner validation for {test_year}"
            )
        selection = select_panel_candidate(
            inner_panel,
            candidates,
            filter_config=replace(
                base_filter,
                seed=_stable_seed(seed, f"selection::{test_year}"),
                predictive_draws=0,
            ),
            selection_criterion="country_balanced_validation_log_score",
            n_jobs=n_jobs,
        )
        selected_by_country, country_candidate_details = (
            select_shrunk_country_candidates(selection)
        )
        probability_weights, point_weights = _inner_stacking_weights(
            inner_panel,
            selection,
            prequential=prequential,
            selected_by_country=selected_by_country,
        )
        global_stacking_weights = probability_weights["__global__"]
        global_point_weights = point_weights["__global__"]
        print(
            f"[panel-pomp] outer {test_year}: selected {selected_by_country}; "
            f"stacking={global_stacking_weights}; point={global_point_weights}",
            flush=True,
        )
        selection_rows.extend(
            _selection_rows(
                selection,
                test_year,
                global_stacking_weights,
                global_point_weights,
                selection_years,
                country_candidate_details,
            )
        )

        ensure_year(test_year)
        for country in countries:
            built = cache[(country, test_year)]
            if built is None:
                continue
            series, series_audit = built
            selected = _candidate_by_name(
                candidates, selected_by_country[country]
            )
            stacking_weights = probability_weights.get(
                country, global_stacking_weights
            )
            country_point_weights = point_weights.get(
                country, global_point_weights
            )
            audit_rows.append(series_audit)
            country_filter = replace(
                base_filter,
                seed=_stable_seed(seed, f"outer::{country}::{test_year}"),
            )
            fit = fit_country_static_log_scale(
                series,
                process=selected.process,
                observation=selected.observation,
                filter_config=country_filter,
                search=selected.scale_search,
            )
            replicate_ll = [float(fit.training_log_likelihood)]
            for replicate in range(1, int(mc_replicates)):
                replicate_result = bootstrap_particle_filter(
                    series,
                    static_log_scale=fit.static_log_scale,
                    process=selected.process,
                    observation=selected.observation,
                    config=replace(
                        country_filter,
                        seed=_stable_seed(
                            seed, f"mc::{country}::{test_year}::{replicate}"
                        ),
                        predictive_draws=0,
                    ),
                )
                replicate_ll.append(float(replicate_result.training_log_likelihood))
            mc_sd = float(np.std(replicate_ll, ddof=1)) if len(replicate_ll) > 1 else 0.0
            mc_sd_per_interval = mc_sd / max(series.n_training, 1)
            baselines = _log_rate_baselines(series, prequential=prequential)
            holdout_indices = np.flatnonzero(~np.asarray(series.training_mask, dtype=bool))
            forecasts = [
                forecast
                for forecast in fit.filter_result.forecasts
                if not forecast.training_interval
            ]
            if len(forecasts) != len(holdout_indices):
                raise RuntimeError("Particle forecast and holdout interval counts differ")

            annual_observed = 0.0
            annual_model = 0.0
            annual_model_point = 0.0
            annual_pomp = 0.0
            annual_baselines = {name: 0.0 for name in baselines}
            model_log_score = 0.0
            pomp_log_score = 0.0
            baseline_log_scores = {name: 0.0 for name in baselines}
            coverage_values: list[float] = []
            predictive_widths: list[float] = []
            annual_predictive_draws = np.zeros(int(predictive_draws), dtype=float)
            component_rng = np.random.default_rng(
                _stable_seed(seed, f"ensemble-path::{country}::{test_year}")
            )
            ensemble_component_indices = component_rng.choice(
                len(COMPONENT_NAMES),
                size=int(predictive_draws),
                p=np.asarray(
                    [stacking_weights[name] for name in COMPONENT_NAMES],
                    dtype=float,
                ),
            )
            for position, (index, forecast) in enumerate(zip(holdout_indices, forecasts)):
                observed = float(series.observed_counts[index])
                exposure = float(series.exposure_days[index])
                annual_observed += observed
                annual_pomp += float(forecast.expected_mean)
                pomp_log_score += float(forecast.predictive_log_score)
                pomp_interval_size = float(
                    selected.observation.interval_size(exposure)
                )
                baseline_interval_size = float(
                    BASELINE_OBSERVATION.interval_size(exposure)
                )
                baseline_means: dict[str, float] = {}
                baseline_metrics: dict[str, tuple[float, float, float]] = {}
                for name, predictions in baselines.items():
                    mean = float(predictions[position])
                    baseline_means[name] = mean
                    baseline_metrics[name] = _nb_forecast_metrics(
                        observed, mean, baseline_interval_size
                    )
                component_means = {
                    "pomp": float(forecast.expected_mean),
                    **baseline_means,
                }
                component_scores = {
                    "pomp": float(forecast.predictive_log_score),
                    **{
                        name: float(metrics[0])
                        for name, metrics in baseline_metrics.items()
                    },
                }
                model_mean = float(
                    sum(
                        stacking_weights[name] * component_means[name]
                        for name in COMPONENT_NAMES
                    )
                )
                model_point = float(
                    np.expm1(
                        sum(
                            country_point_weights[name]
                            * np.log1p(component_means[name])
                            for name in COMPONENT_NAMES
                        )
                    )
                )
                model_score = float(
                    logsumexp(
                        [
                            np.log(stacking_weights[name]) + component_scores[name]
                            for name in COMPONENT_NAMES
                        ]
                    )
                )
                ensemble_draws = _ensemble_predictive_draws(
                    pomp_draws=forecast.observation_draws,
                    baseline_means=baseline_means,
                    component_sizes={
                        "pomp": pomp_interval_size,
                        **{
                            name: baseline_interval_size
                            for name in COMPONENT_NAMES[1:]
                        },
                    },
                    weights=stacking_weights,
                    draws=int(predictive_draws),
                    seed=_stable_seed(
                        seed, f"ensemble::{country}::{test_year}::{forecast.interval_id}"
                    ),
                    component_indices=ensemble_component_indices,
                )
                model_q025, model_median, model_q975 = np.quantile(
                    ensemble_draws, (0.025, 0.5, 0.975)
                )
                annual_predictive_draws += ensemble_draws
                annual_model += model_mean
                annual_model_point += model_point
                model_log_score += model_score
                covered = float(model_q025 <= observed <= model_q975)
                coverage_values.append(covered)
                predictive_widths.append(
                    float(np.log1p(model_q975) - np.log1p(model_q025))
                )
                row: dict[str, Any] = {
                    "country": country,
                    "test_year": int(test_year),
                    "interval_id": forecast.interval_id,
                    "observed_cases": observed,
                    "exposure_days": exposure,
                    "model_expected_cases": model_mean,
                    "model_point_cases": model_point,
                    "model_observation_q025": float(model_q025),
                    "model_observation_median": float(model_median),
                    "model_observation_q975": float(model_q975),
                    "model_log_score": model_score,
                    "model_95_covered": covered,
                    "pomp_expected_cases": float(forecast.expected_mean),
                    "pomp_observation_q025": float(forecast.observation_q025),
                    "pomp_observation_median": float(forecast.observation_median),
                    "pomp_observation_q975": float(forecast.observation_q975),
                    "pomp_log_score": float(forecast.predictive_log_score),
                    "selected_candidate": selected.name,
                    "static_log_source_scale": float(fit.static_log_scale),
                    "scale_at_search_boundary": bool(fit.at_search_boundary),
                    "mc_log_likelihood_sd_per_training_interval": mc_sd_per_interval,
                    "future_evidence_used": False,
                    "claim_scope": "notification_index_forecast",
                    "forecast_mode": forecast_mode,
                    **{
                        f"stacking_weight_{name}": float(stacking_weights[name])
                        for name in COMPONENT_NAMES
                    },
                    **{
                        f"point_weight_{name}": float(country_point_weights[name])
                        for name in COMPONENT_NAMES
                    },
                }
                for name, mean in baseline_means.items():
                    score, lower, upper = baseline_metrics[name]
                    annual_baselines[name] += mean
                    baseline_log_scores[name] += score
                    row[f"{name}_expected_cases"] = mean
                    row[f"{name}_q025"] = lower
                    row[f"{name}_q975"] = upper
                    row[f"{name}_log_score"] = score
                interval_rows.append(row)

            if prequential:
                # Each forecast conditions on earlier observations from the
                # same holdout year. The sequential scores obey the chain
                # rule, but their interval draws are not a joint forecast
                # issued at the start of the year.
                annual_q025 = annual_median = annual_q975 = float("nan")
                annual_covered = float("nan")
                annual_width = float("nan")
                annual_prediction_scope = "not_applicable_prequential_sequence"
            else:
                annual_q025, annual_median, annual_q975 = np.quantile(
                    annual_predictive_draws, (0.025, 0.5, 0.975)
                )
                annual_covered = bool(
                    annual_q025 <= annual_observed <= annual_q975
                )
                annual_width = float(
                    np.log1p(annual_q975) - np.log1p(annual_q025)
                )
                annual_prediction_scope = "joint_one_year_block_forecast"
            fold: dict[str, Any] = {
                "country": country,
                "test_year": int(test_year),
                "selected_candidate": selected.name,
                "training_intervals": int(series.n_training),
                "holdout_intervals": int(len(holdout_indices)),
                "observed_cases": annual_observed,
                "model_expected_cases": annual_model,
                "model_point_cases": annual_model_point,
                "model_absolute_log1p_error": abs(
                    log(annual_model_point + 1.0) - log(annual_observed + 1.0)
                ),
                "model_log_score": model_log_score,
                "pomp_expected_cases": annual_pomp,
                "pomp_absolute_log1p_error": abs(
                    log(annual_pomp + 1.0) - log(annual_observed + 1.0)
                ),
                "pomp_log_score": pomp_log_score,
                "model_interval_95_coverage": float(np.mean(coverage_values)),
                "model_mean_predictive_log1p_width": float(np.mean(predictive_widths)),
                "model_annual_q025": float(annual_q025),
                "model_annual_median": float(annual_median),
                "model_annual_q975": float(annual_q975),
                "model_annual_95_covered": annual_covered,
                "model_annual_predictive_log1p_width": annual_width,
                "annual_prediction_scope": annual_prediction_scope,
                "static_log_source_scale": float(fit.static_log_scale),
                "scale_at_search_boundary": bool(fit.at_search_boundary),
                "mc_log_likelihood_sd_per_training_interval": mc_sd_per_interval,
                "future_evidence_used": False,
                "forecast_mode": forecast_mode,
                **{
                    f"stacking_weight_{name}": float(stacking_weights[name])
                    for name in COMPONENT_NAMES
                },
                **{
                    f"point_weight_{name}": float(country_point_weights[name])
                    for name in COMPONENT_NAMES
                },
            }
            for name in baselines:
                mean = annual_baselines[name]
                fold[f"{name}_expected_cases"] = mean
                fold[f"{name}_absolute_log1p_error"] = abs(
                    log(mean + 1.0) - log(annual_observed + 1.0)
                )
                fold[f"{name}_log_score"] = baseline_log_scores[name]
                fold[f"{name}_mean_log_score_per_interval"] = (
                    baseline_log_scores[name] / max(len(holdout_indices), 1)
                )
                fold[f"model_log_score_delta_vs_{name}"] = (
                    model_log_score - baseline_log_scores[name]
                )
                fold[f"model_log1p_error_delta_vs_{name}"] = (
                    fold[f"{name}_absolute_log1p_error"]
                    - fold["model_absolute_log1p_error"]
                )
            fold["model_mean_log_score_per_interval"] = (
                model_log_score / max(len(holdout_indices), 1)
            )
            fold_rows.append(fold)

    return (
        pd.DataFrame(interval_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(audit_rows),
    )


def summarize_and_gate(
    intervals: pd.DataFrame,
    folds: pd.DataFrame,
    audits: pd.DataFrame,
    source_summary: pd.DataFrame,
    *,
    mc_replicates: int,
    forecast_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_names = ("seasonal_naive", "ew_recent_rate", "damped_log_trend")
    summaries: list[dict[str, Any]] = []
    for scope, frame in [("overall", folds), *folds.groupby("country", sort=True)]:
        name = "all_countries" if scope == "overall" else str(scope)
        row: dict[str, Any] = {
            "scope": "overall" if scope == "overall" else "country",
            "country": name,
            "fold_count": int(len(frame)),
            "model_mean_absolute_log1p_error": float(frame["model_absolute_log1p_error"].mean()),
            "model_mean_log_score_per_interval": float(
                frame["model_mean_log_score_per_interval"].mean()
            ),
        }
        for baseline in baseline_names:
            row[f"{baseline}_mean_absolute_log1p_error"] = float(
                frame[f"{baseline}_absolute_log1p_error"].mean()
            )
            row[f"{baseline}_mean_log_score_per_interval"] = float(
                frame[f"{baseline}_mean_log_score_per_interval"].mean()
            )
            row[f"model_log_score_win_fraction_vs_{baseline}"] = float(
                frame[f"model_log_score_delta_vs_{baseline}"].gt(0.0).mean()
            )
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    overall = summary.loc[summary["scope"].eq("overall")].iloc[0]
    fold_counts = folds.groupby("country").size()
    data_integrity = bool(
        not audits.empty
        and not audits["future_evidence_used"].astype(bool).any()
        and audits["outcome_derived_profile_terms_disabled"].astype(bool).all()
        and audits["historical_maternal_coverage_excluded"].astype(bool).all()
        and audits["npi_reduction_column"].eq("contact_reduction_mean").all()
        and not audits["diagnostic_timeline_enabled"].astype(bool).any()
    )
    beats_all_log_scores = all(
        float(overall["model_mean_log_score_per_interval"])
        > float(overall[f"{baseline}_mean_log_score_per_interval"])
        for baseline in baseline_names
    )
    beats_all_point_errors = all(
        float(overall["model_mean_absolute_log1p_error"])
        < float(overall[f"{baseline}_mean_absolute_log1p_error"])
        for baseline in baseline_names
    )
    interval_coverage = float(intervals["model_95_covered"].mean())
    country_coverage = intervals.groupby("country")["model_95_covered"].agg(
        ["sum", "count", "mean"]
    )
    country_balanced_coverage = float(country_coverage["mean"].mean())
    country_log1p_width = (
        np.log1p(intervals["model_observation_q975"])
        - np.log1p(intervals["model_observation_q025"])
    ).groupby(intervals["country"]).mean()
    country_balanced_log1p_width = float(country_log1p_width.mean())
    simultaneous_alpha = 0.05 / max(len(country_coverage), 1)
    simultaneous_z = float(norm.ppf(1.0 - simultaneous_alpha / 2.0))
    coverage_probability = country_coverage["mean"].to_numpy(dtype=float)
    coverage_count = country_coverage["count"].to_numpy(dtype=float)
    denominator = 1.0 + simultaneous_z**2 / coverage_count
    center = (
        coverage_probability + simultaneous_z**2 / (2.0 * coverage_count)
    ) / denominator
    half_width = (
        simultaneous_z
        * np.sqrt(
            coverage_probability * (1.0 - coverage_probability) / coverage_count
            + simultaneous_z**2 / (4.0 * coverage_count**2)
        )
        / denominator
    )
    simultaneous_upper = center + half_width
    undercalibrated_countries = country_coverage.index[
        simultaneous_upper < 0.95
    ].tolist()
    annual_coverage_values = pd.to_numeric(
        folds["model_annual_95_covered"], errors="coerce"
    ).dropna()
    annual_coverage = (
        float(annual_coverage_values.mean())
        if not annual_coverage_values.empty
        else float("nan")
    )
    if str(forecast_mode) == "prequential":
        coverage = country_balanced_coverage
        mean_log1p_width = country_balanced_log1p_width
        coverage_scope = "country_balanced_one_interval_ahead_prequential"
    else:
        coverage = annual_coverage
        mean_log1p_width = float(
            folds["model_annual_predictive_log1p_width"].mean()
        )
        coverage_scope = "one_year_block"
    mc_stable = bool(
        int(mc_replicates) >= 3
        and folds["mc_log_likelihood_sd_per_training_interval"].max()
        <= MAXIMUM_MC_LOG_LIKELIHOOD_SD_PER_INTERVAL
    )
    predictive_gate = bool(
        len(fold_counts) >= MINIMUM_COUNTRIES
        and int(fold_counts.min()) >= MINIMUM_FOLDS_PER_COUNTRY
        and beats_all_log_scores
        and MINIMUM_PREDICTIVE_COVERAGE <= coverage <= MAXIMUM_PREDICTIVE_COVERAGE
        and mean_log1p_width <= MAXIMUM_MEAN_PREDICTIVE_LOG1P_WIDTH
        and mc_stable
    )
    absolute_burden_all = bool(
        not source_summary.empty
        and source_summary["absolute_burden_claim_allowed"].astype(bool).all()
    )
    gate = pd.DataFrame(
        [
            {
                "execution_complete": True,
                "data_integrity_gate_pass": data_integrity,
                "countries_completed": int(len(fold_counts)),
                "minimum_folds_per_country": int(fold_counts.min()),
                "model_beats_all_baselines_log_score": beats_all_log_scores,
                "model_beats_all_baselines_log1p_error": beats_all_point_errors,
                "model_interval_95_coverage_diagnostic": interval_coverage,
                "model_country_balanced_interval_95_coverage": (
                    country_balanced_coverage
                ),
                "minimum_country_interval_95_coverage": float(
                    country_coverage["mean"].min()
                ),
                "countries_incompatible_with_nominal_95_coverage": ";".join(
                    undercalibrated_countries
                ),
                "all_country_calibration_claim_gate_pass": bool(
                    not undercalibrated_countries
                ),
                "model_annual_95_coverage_diagnostic": annual_coverage,
                "primary_predictive_95_coverage": coverage,
                "primary_predictive_mean_log1p_width": mean_log1p_width,
                "primary_predictive_scope": coverage_scope,
                "predictive_width_gate_pass": bool(
                    mean_log1p_width <= MAXIMUM_MEAN_PREDICTIVE_LOG1P_WIDTH
                ),
                "mc_replicates": int(mc_replicates),
                "monte_carlo_stability_pass": mc_stable,
                "predictive_gate_pass": predictive_gate,
                "absolute_burden_claim_gate_pass": absolute_burden_all,
                "publication_gate_pass": bool(data_integrity and predictive_gate),
                "claim_scope": "notification-index forecasting and conditional scenario projections",
                "full_compartment_pomp_claimed": False,
                "posterior_parameter_uncertainty_claimed": False,
            }
        ]
    )
    return summary, gate


def _parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    if not years:
        raise ValueError("At least one test year is required")
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-years", default="2023-2026")
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--predictive-draws", type=int, default=2048)
    parser.add_argument("--mc-replicates", type=int, default=3)
    parser.add_argument(
        "--forecast-mode",
        choices=("prequential", "block"),
        default="prequential",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Country-level worker processes; -1 uses all available cores.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--countries", default="")
    parser.add_argument("--require-predictive-gate", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    all_countries = publication_country_names()
    countries = (
        [item.strip() for item in args.countries.split(",") if item.strip()]
        if args.countries
        else all_countries
    )
    unknown = sorted(set(countries).difference(all_countries))
    if unknown:
        raise ValueError(f"Unknown/non-publication countries requested: {unknown}")
    test_years = _parse_years(args.test_years)
    intervals, folds, selections, audits = run_hindcast(
        countries,
        test_years,
        particles=args.particles,
        predictive_draws=args.predictive_draws,
        seed=args.seed,
        mc_replicates=args.mc_replicates,
        n_jobs=args.n_jobs,
        forecast_mode=args.forecast_mode,
    )
    source_audit = build_observation_source_audit()
    source_summary = country_source_summary(source_audit)
    summary, gate = summarize_and_gate(
        intervals,
        folds,
        audits,
        source_summary,
        mc_replicates=args.mc_replicates,
        forecast_mode=args.forecast_mode,
    )

    if args.forecast_mode == "prequential":
        run_stem = PREQUENTIAL_RUN_STEM
        outputs = {
            "panel_pomp_rolling_hindcast_intervals": intervals,
            "panel_pomp_rolling_hindcast_folds": folds,
            "panel_pomp_rolling_hindcast_summary": summary,
            "panel_pomp_candidate_selection": selections,
            "panel_pomp_evidence_cutoff_audit": audits,
            "panel_pomp_rolling_hindcast_gate": gate,
            "observation_source_reconciliation": source_audit,
            "observation_source_reconciliation_summary": source_summary,
        }
    else:
        run_stem = BLOCK_STRESS_RUN_STEM
        outputs = {
            "panel_pomp_block_stress_intervals": intervals,
            "panel_pomp_block_stress_folds": folds,
            "panel_pomp_block_stress_summary": summary,
            "panel_pomp_block_stress_candidate_selection": selections,
            "panel_pomp_block_stress_evidence_cutoff_audit": audits,
            "panel_pomp_block_stress_gate": gate,
            "observation_source_reconciliation": source_audit,
            "observation_source_reconciliation_summary": source_summary,
        }
    for stem, frame in outputs.items():
        write_dataframe(frame, project_path(f"outputs/tables/{stem}.csv"))
    metadata = current_run_metadata(
        run_stem,
        row_counts={stem: int(len(frame)) for stem, frame in outputs.items()},
    ) | {
        "countries_included": countries,
        "countries_requested": countries,
        "test_years": test_years,
        "fold_counts": {
            str(country): int(count)
            for country, count in folds.groupby("country").size().items()
        },
        "fold_test_years": {
            str(country): sorted(group["test_year"].astype(int).unique().tolist())
            for country, group in folds.groupby("country", sort=True)
        },
        "particles": int(args.particles),
        "predictive_draws": int(args.predictive_draws),
        "mc_replicates": int(args.mc_replicates),
        "n_jobs": int(args.n_jobs),
        "forecast_mode": str(args.forecast_mode),
        "seed": int(args.seed),
        "candidate_specs": [
            {
                "name": candidate.name,
                "process": asdict(candidate.process),
                "observation": asdict(candidate.observation),
                "scale_search": asdict(candidate.scale_search),
            }
            for candidate in default_candidates()
        ],
        "data_integrity_gate_pass": bool(gate.iloc[0]["data_integrity_gate_pass"]),
        "predictive_gate_pass": bool(gate.iloc[0]["predictive_gate_pass"]),
        "publication_gate_pass": bool(gate.iloc[0]["publication_gate_pass"]),
        "all_country_calibration_claim_gate_pass": bool(
            gate.iloc[0]["all_country_calibration_claim_gate_pass"]
        ),
        "absolute_burden_claim_gate_pass": bool(
            gate.iloc[0]["absolute_burden_claim_gate_pass"]
        ),
        "full_compartment_pomp_claimed": False,
        "posterior_parameter_uncertainty_claimed": False,
        "input_artifact_sha256": {
            "pertussis_incidence_timeseries": file_sha256(
                project_path("data/processed/pertussis_incidence_timeseries.csv")
            ),
        },
        "output_artifact_sha256": {
            stem: file_sha256(project_path(f"outputs/tables/{stem}.csv"))
            for stem in outputs
        },
        "scope": "semi_mechanistic_discrepancy_pomp_not_full_compartment_pomp",
    }
    write_run_metadata(run_stem, metadata)
    if args.require_predictive_gate and not bool(gate.iloc[0]["publication_gate_pass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
