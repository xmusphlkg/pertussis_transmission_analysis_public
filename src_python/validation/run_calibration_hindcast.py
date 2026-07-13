"""Rolling-origin validation for the annual latent-transmission calibration.

The calibration fit metrics are in-sample state-reconstruction diagnostics.
This module performs the separate predictive check: fit only through year t,
propagate the AR(1) conditional mean into year t+1, and score the untouched
integer surveillance windows in that year.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import nbinom

from src_python.calibration.calibrate_baseline import (
    _calibration_predicted_means,
    aggregate_observed_case_intervals,
    calibration_runtime_config,
    extend_annual_log_beta_conditional_mean,
    grouped_likelihood_observations,
    observed_annual_case_frame,
    retain_recent_observed_window,
    state_space_map_calibration,
)
from src_python.calibration.likelihood import negative_binomial_nll
from src_python.calibration.state_space_uncertainty import bounded_multivariate_normal_draws
from src_python.simulation.common import (
    current_run_metadata,
    load_configs,
    make_config,
    publication_country_names,
    write_run_metadata,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import parallel_map


MINIMUM_FOLDS_FOR_PREDICTIVE_ADEQUACY = 3
MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE = 0.80
# A nominal 95% interval spanning more than 100-fold on the log1p count scale
# is non-discriminating even when it happens to cover the observation.
MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH = float(np.log(100.0))
CALIBRATION_HINDCAST_STEM = "calibration_rolling_hindcast"
PUBLICATION_STATE_UNCERTAINTY_MODE = "map_conditioned"
PUBLICATION_PREDICTIVE_UNCERTAINTY_SCOPE = (
    "conditional_on_MAP_state_with_future_AR1_process_and_NB2_measurement"
)

HINDCAST_FINITE_COLUMNS = (
    "observed_cases",
    "predicted_cases",
    "naive_predicted_cases",
    "absolute_log1p_error",
    "naive_absolute_log1p_error",
    "negative_binomial_nll",
    "naive_negative_binomial_nll",
    "absolute_log1p_error_improvement_over_naive",
    "negative_binomial_nll_improvement_over_naive",
    "process_mixture_negative_binomial_nll",
    "process_mixture_negative_binomial_nll_improvement_over_naive",
    "smape",
    "nb95_interval_coverage",
    "naive_nb95_interval_coverage",
    "nb95_interval_mean_width",
    "naive_nb95_interval_mean_width",
    "nb95_interval_mean_relative_width",
    "naive_nb95_interval_mean_relative_width",
    "process_predictive_mean_q025",
    "process_predictive_mean_median",
    "process_predictive_mean_q975",
    "observation_predictive_q025",
    "observation_predictive_median",
    "observation_predictive_q975",
    "observation_predictive_95_width",
    "observation_predictive_95_relative_width",
    "observation_predictive_95_log1p_width",
    "laplace_boundary_rejection_fraction",
    "predictive_process_draws",
    "predictive_measurement_draws",
    "posterior_rank",
    "posterior_dimension",
    "posterior_condition_number",
    "process_nfev",
)


def rolling_year_splits(
    observed: pd.DataFrame,
    *,
    minimum_training_years: int = 4,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return expanding-window, one-calendar-year-ahead splits."""

    if "observed_year" not in observed.columns:
        raise ValueError("Rolling hindcast observations require observed_year")
    years = sorted(pd.to_numeric(observed["observed_year"], errors="raise").astype(int).unique())
    if minimum_training_years < 2:
        raise ValueError("minimum_training_years must be at least two")
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for test_position in range(minimum_training_years, len(years)):
        training_years = set(years[:test_position])
        test_year = years[test_position]
        training = observed.loc[observed["observed_year"].isin(training_years)].copy()
        test = observed.loc[observed["observed_year"].eq(test_year)].copy()
        if not training.empty and not test.empty:
            splits.append((training.reset_index(drop=True), test.reset_index(drop=True)))
    return splits


def extend_ar1_conditional_mean_path(
    calibrated: dict[str, Any],
    *,
    through_year: int,
) -> dict[str, Any]:
    """Extend a fitted annual path with E[x[t+h] | x[t]] under its AR(1)."""
    return extend_annual_log_beta_conditional_mean(
        calibrated,
        through_year=through_year,
        interpretation="latent_AR1_process_MAP_hindcast_conditional_mean",
    )


def _forecast_runtime(
    country: str,
    calibrated: dict[str, Any],
    test: pd.DataFrame,
    *,
    evidence_cutoff_date: str,
) -> dict[str, Any]:
    configs = load_configs()
    runtime = calibration_runtime_config(
        make_config(
            vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
            resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
            country_profile=country,
            load_calibration=False,
            evidence_cutoff_date=evidence_cutoff_date,
        ),
        test,
    )
    runtime["transmission"]["beta_S"] = float(calibrated["transmission"]["beta_S"])
    runtime["reporting_multiplier"] = float(calibrated.get("reporting_multiplier", 1.0))
    runtime["transmission"]["log_beta_time_variation"] = deepcopy(
        calibrated["transmission"]["log_beta_time_variation"]
    )
    return runtime


def _nb_interval(mean: np.ndarray, dispersion: float) -> tuple[np.ndarray, np.ndarray]:
    mu = np.maximum(np.asarray(mean, dtype=float), 1e-12)
    k = float(dispersion)
    probability = k / (k + mu)
    return (
        nbinom.ppf(0.025, k, probability).astype(float),
        nbinom.ppf(0.975, k, probability).astype(float),
    )


def last_year_naive_means(
    training: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    """Exposure-adjust the most recent observed annual rate to test windows.

    This is the seasonal-naive comparator available for annual likelihood
    groups.  Scaling by observed surveillance days avoids treating a partial
    final year (or a year containing reporting gaps) as a complete year.
    """

    grouped_training = grouped_likelihood_observations(training)
    grouped_test = grouped_likelihood_observations(test)
    required = {"observed_year", "reported_cases", "interval_days"}
    for label, frame in (("training", grouped_training), ("test", grouped_test)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Last-year naive {label} observations are missing {missing}")
    if grouped_training.empty or grouped_test.empty:
        raise ValueError("Last-year naive forecast requires non-empty train and test data")
    last_year = int(
        pd.to_numeric(grouped_training["observed_year"], errors="raise").max()
    )
    previous = grouped_training.loc[
        pd.to_numeric(grouped_training["observed_year"], errors="raise").eq(last_year)
    ]
    previous_cases = float(
        pd.to_numeric(previous["reported_cases"], errors="raise").sum()
    )
    previous_days = float(
        pd.to_numeric(previous["interval_days"], errors="raise").sum()
    )
    test_days = pd.to_numeric(
        grouped_test["interval_days"], errors="raise"
    ).to_numpy(dtype=float)
    if previous_days <= 0.0 or np.any(test_days <= 0.0):
        raise ValueError("Last-year naive forecast requires positive surveillance exposure")
    return np.maximum(previous_cases / previous_days * test_days, 1e-12)


def _mean_interval_width(
    low: np.ndarray,
    high: np.ndarray,
) -> float:
    return float(np.mean(np.asarray(high, dtype=float) - np.asarray(low, dtype=float)))


def run_country_hindcast(
    country: str,
    *,
    minimum_training_years: int = 4,
    max_folds: int | None = None,
    predictive_draws: int = 64,
    dispersion: float | None = None,
    process_overrides: dict[str, Any] | None = None,
    state_coordinate_uncertainty_mode: str = "map_conditioned",
) -> pd.DataFrame:
    configs = load_configs()
    calibration = configs["baseline"].get("calibration", {})
    dispersion = float(
        calibration.get("dispersion", 50.0) if dispersion is None else dispersion
    )
    if dispersion <= 0.0:
        raise ValueError("Hindcast negative-binomial dispersion must be positive")
    resolved_process = {
        **calibration.get("process_model", {}),
        **(process_overrides or {}),
    }
    ar1_rho = float(resolved_process.get("ar1_rho", 0.5))
    innovation_sd = float(
        resolved_process.get("log_beta_innovation_sd", 0.6)
    )
    if state_coordinate_uncertainty_mode not in {
        "gaussian_approximation",
        "map_conditioned",
    }:
        raise ValueError(
            "state_coordinate_uncertainty_mode must be gaussian_approximation "
            "or map_conditioned"
        )
    recent_years = int(calibration.get("recent_years", 0))
    native = retain_recent_observed_window(observed_annual_case_frame(country), recent_years)
    observed = aggregate_observed_case_intervals(native, "annual")
    splits = rolling_year_splits(
        observed,
        minimum_training_years=minimum_training_years,
    )
    if max_folds is not None:
        if int(max_folds) < 1:
            raise ValueError("max_folds must be at least one when specified")
        splits = splits[-int(max_folds) :]
    rows: list[dict[str, Any]] = []
    for fold, (training, test) in enumerate(splits, start=1):
        test_year = int(
            pd.to_numeric(test["observed_year"], errors="raise").iloc[0]
        )
        evidence_cutoff_date = f"{test_year:04d}-01-01"
        training_base = make_config(
            country_profile=country,
            load_calibration=False,
            evidence_cutoff_date=evidence_cutoff_date,
        )
        training_runtime = calibration_runtime_config(
            training_base,
            training,
        )
        fitted, result = state_space_map_calibration(
            training_runtime,
            country,
            training,
            dispersion=dispersion,
            maxiter=int(calibration.get("maxiter", 12)),
            process_overrides=process_overrides,
        )
        extended = extend_ar1_conditional_mean_path(fitted, through_year=test_year)
        forecast = _forecast_runtime(
            country,
            extended,
            test,
            evidence_cutoff_date=evidence_cutoff_date,
        )
        predicted = _calibration_predicted_means(forecast, test, country)
        grouped_test = grouped_likelihood_observations(test)
        actual = pd.to_numeric(
            grouped_test["reported_cases"], errors="raise"
        ).to_numpy(dtype=float)
        naive_predicted = last_year_naive_means(training, test)
        if len(naive_predicted) != len(actual):
            raise RuntimeError("Last-year naive prediction/observation dimension mismatch")
        low, high = _nb_interval(predicted, dispersion)
        naive_low, naive_high = _nb_interval(naive_predicted, dispersion)
        coordinate_map = np.asarray(result.posterior_map_vector, dtype=float)
        if state_coordinate_uncertainty_mode == "map_conditioned":
            # Hyperparameter scoring must not treat the local Gauss-Newton
            # covariance as an exact posterior.  Repeating the MAP coordinate
            # retains independent future AR(1) innovations and NB2 measurement
            # draws while making the conditional scope explicit.
            coordinate_draws = np.repeat(
                coordinate_map[None, :],
                repeats=int(max(predictive_draws, 8)),
                axis=0,
            )
            boundary_rejection = 0.0
        else:
            coordinate_draws, boundary_rejection = bounded_multivariate_normal_draws(
                coordinate_map,
                np.asarray(result.posterior_covariance, dtype=float),
                np.asarray(result.posterior_lower_bounds, dtype=float),
                np.asarray(result.posterior_upper_bounds, dtype=float),
                n=int(max(predictive_draws, 8)),
                seed=20260712
                + sum(
                    (index + 1) * ord(char)
                    for index, char in enumerate(country)
                )
                + fold,
            )
        process_rng = np.random.default_rng(
            20260713 + sum((index + 3) * ord(char) for index, char in enumerate(country)) + fold
        )
        measurement_rng = np.random.default_rng(
            20260714
            + sum(
                (index + 5) * ord(char)
                for index, char in enumerate(country)
            )
            + fold
        )
        process_mean_totals: list[float] = []
        process_log_likelihoods: list[float] = []
        observation_predictive_totals: list[float] = []
        fitted_period_count = len(fitted["transmission"]["log_beta_time_variation"]["periods"])
        for coordinate in coordinate_draws:
            draw_config = deepcopy(fitted)
            draw_config["transmission"]["beta_S"] = float(np.exp(coordinate[0]))
            draw_config["reporting_multiplier"] = float(np.exp(coordinate[1]))
            draw_periods = draw_config["transmission"]["log_beta_time_variation"]["periods"]
            if len(coordinate) != fitted_period_count + 2:
                raise RuntimeError("Hindcast state coordinate/path dimension mismatch")
            for period, value in zip(draw_periods, coordinate[2:]):
                period["log_multiplier"] = float(value)
            last_year = int(pd.Timestamp(draw_periods[-1]["start_date"]).year)
            next_state = float(coordinate[-1])
            rho = float(
                draw_config["transmission"]["log_beta_time_variation"]["ar1_rho"]
            )
            innovation_sd = float(
                draw_config["transmission"]["log_beta_time_variation"]["innovation_sd"]
            )
            for _year in range(last_year + 1, test_year + 1):
                next_state = rho * next_state + float(process_rng.normal(0.0, innovation_sd))
            draw_extended = extend_ar1_conditional_mean_path(
                draw_config,
                through_year=test_year,
            )
            draw_extended["transmission"]["log_beta_time_variation"]["periods"][-1][
                "log_multiplier"
            ] = next_state
            draw_forecast = _forecast_runtime(
                country,
                draw_extended,
                test,
                evidence_cutoff_date=evidence_cutoff_date,
            )
            draw_mean = _calibration_predicted_means(draw_forecast, test, country)
            process_mean_totals.append(float(draw_mean.sum()))
            probability = dispersion / (dispersion + np.maximum(draw_mean, 1e-12))
            process_log_likelihoods.append(
                float(np.sum(nbinom.logpmf(actual, dispersion, probability)))
            )
            measurement_draw = nbinom.rvs(
                dispersion,
                probability,
                size=(4, len(draw_mean)),
                random_state=measurement_rng,
            )
            observation_predictive_totals.extend(
                np.asarray(measurement_draw, dtype=float).sum(axis=1).tolist()
            )
        process_q025, process_median, process_q975 = np.quantile(
            process_mean_totals,
            [0.025, 0.5, 0.975],
        )
        predictive_q025, predictive_median, predictive_q975 = np.quantile(
            observation_predictive_totals,
            [0.025, 0.5, 0.975],
        )
        observed_total = float(actual.sum())
        predicted_total = float(predicted.sum())
        naive_predicted_total = float(naive_predicted.sum())
        predictive_width = float(predictive_q975 - predictive_q025)
        process_mixture_nll = float(
            -logsumexp(process_log_likelihoods)
            + np.log(len(process_log_likelihoods))
        )
        naive_nll = float(
            negative_binomial_nll(actual, naive_predicted, dispersion)
        )
        rows.append(
            {
                "country": country,
                "process_ar1_rho": ar1_rho,
                "process_innovation_sd": innovation_sd,
                "observation_dispersion": dispersion,
                "state_coordinate_uncertainty_mode": (
                    state_coordinate_uncertainty_mode
                ),
                "predictive_uncertainty_scope": (
                    "conditional_on_MAP_state_with_future_AR1_process_and_NB2_measurement"
                    if state_coordinate_uncertainty_mode == "map_conditioned"
                    else "local_gaussian_state_approximation_with_future_AR1_process_and_NB2_measurement"
                ),
                "posterior_coverage_claimed": False,
                "fold": fold,
                "training_start_year": int(training["observed_year"].min()),
                "training_end_year": int(training["observed_year"].max()),
                "test_year": test_year,
                "evidence_cutoff_date": evidence_cutoff_date,
                "resistance_evidence_anchor_year": int(
                    training_base.get("metadata", {}).get(
                        "resistance_timeline_anchor_year",
                        test_year - 1,
                    )
                ),
                "resistance_timeline_applied": bool(
                    training_base.get("metadata", {}).get(
                        "resistance_timeline_applied",
                        False,
                    )
                ),
                "resistance_evidence_years": str(
                    training_base.get("metadata", {}).get(
                        "resistance_timeline_evidence_years",
                        "",
                    )
                ),
                "diagnostic_periods_known_at_origin": int(
                    training_base.get("metadata", {}).get(
                        "diagnostic_standard_periods_loaded",
                        0,
                    )
                ),
                "future_evidence_used": False,
                "training_observation_intervals": int(len(training)),
                "test_observation_intervals": int(len(test)),
                "observed_cases": observed_total,
                "predicted_cases": predicted_total,
                "naive_predicted_cases": naive_predicted_total,
                "naive_method": "last_year_exposure_adjusted",
                "absolute_log1p_error": float(
                    abs(np.log1p(predicted_total) - np.log1p(observed_total))
                ),
                "naive_absolute_log1p_error": float(
                    abs(np.log1p(naive_predicted_total) - np.log1p(observed_total))
                ),
                "smape": float(
                    2.0
                    * abs(predicted_total - observed_total)
                    / max(predicted_total + observed_total, 1e-12)
                ),
                "negative_binomial_nll": float(
                    negative_binomial_nll(actual, predicted, dispersion)
                ),
                "naive_negative_binomial_nll": float(
                    naive_nll
                ),
                "process_mixture_negative_binomial_nll": process_mixture_nll,
                "nb95_interval_coverage": float(
                    np.mean((actual >= low) & (actual <= high))
                ),
                "naive_nb95_interval_coverage": float(
                    np.mean((actual >= naive_low) & (actual <= naive_high))
                ),
                "nb95_interval_mean_width": _mean_interval_width(low, high),
                "naive_nb95_interval_mean_width": _mean_interval_width(
                    naive_low, naive_high
                ),
                "nb95_interval_mean_relative_width": float(
                    np.mean((high - low) / np.maximum(predicted, 1.0))
                ),
                "naive_nb95_interval_mean_relative_width": float(
                    np.mean(
                        (naive_high - naive_low) / np.maximum(naive_predicted, 1.0)
                    )
                ),
                "process_predictive_mean_q025": float(process_q025),
                "process_predictive_mean_median": float(process_median),
                "process_predictive_mean_q975": float(process_q975),
                "observation_predictive_q025": float(predictive_q025),
                "observation_predictive_median": float(predictive_median),
                "observation_predictive_q975": float(predictive_q975),
                "observation_predictive_95_coverage": bool(
                    predictive_q025 <= observed_total <= predictive_q975
                ),
                "observation_predictive_95_width": predictive_width,
                "observation_predictive_95_relative_width": float(
                    predictive_width / max(observed_total, 1.0)
                ),
                "observation_predictive_95_log1p_width": float(
                    np.log1p(predictive_q975) - np.log1p(predictive_q025)
                ),
                "absolute_log1p_error_improvement_over_naive": float(
                    abs(np.log1p(naive_predicted_total) - np.log1p(observed_total))
                    - abs(np.log1p(predicted_total) - np.log1p(observed_total))
                ),
                "negative_binomial_nll_improvement_over_naive": float(
                    naive_nll
                    - negative_binomial_nll(actual, predicted, dispersion)
                ),
                "process_mixture_negative_binomial_nll_improvement_over_naive": float(
                    naive_nll - process_mixture_nll
                ),
                "predictive_process_draws": int(len(coordinate_draws)),
                "predictive_measurement_draws": int(
                    len(observation_predictive_totals)
                ),
                "laplace_boundary_rejection_fraction": float(boundary_rejection),
                "optimizer_success": bool(result.success),
                "posterior_rank": int(getattr(result, "posterior_rank", 0)),
                "posterior_dimension": int(
                    len(getattr(result, "posterior_map_vector", []))
                ),
                "posterior_condition_number": float(
                    getattr(result, "posterior_condition_number", np.nan)
                ),
                "process_nfev": int(getattr(result, "process_nfev", 0)),
                "validation_type": "rolling_origin_one_year_ahead",
            }
        )
    return pd.DataFrame(rows)


def run_hindcasts(
    countries: Iterable[str],
    *,
    n_jobs: int | None = None,
    minimum_training_years: int = 4,
    max_folds: int | None = None,
    predictive_draws: int = 64,
    dispersion: float | None = None,
    process_overrides: dict[str, Any] | None = None,
    state_coordinate_uncertainty_mode: str = "map_conditioned",
) -> pd.DataFrame:
    frames = parallel_map(
        lambda country: run_country_hindcast(
            country,
            minimum_training_years=minimum_training_years,
            max_folds=max_folds,
            predictive_draws=predictive_draws,
            dispersion=dispersion,
            process_overrides=process_overrides,
            state_coordinate_uncertainty_mode=state_coordinate_uncertainty_mode,
        ),
        list(countries),
        desc="calibration_hindcast",
        n_jobs=n_jobs,
    )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def expected_hindcast_fold_counts(
    countries: Iterable[str],
    *,
    minimum_training_years: int = 4,
    max_folds: int | None = None,
) -> dict[str, int]:
    """Count every data-supported fold before fitting, for completeness checks."""

    configs = load_configs()
    recent_years = int(
        configs["baseline"].get("calibration", {}).get("recent_years", 0)
    )
    expected: dict[str, int] = {}
    for country in countries:
        native = retain_recent_observed_window(
            observed_annual_case_frame(country), recent_years
        )
        observed = aggregate_observed_case_intervals(native, "annual")
        count = len(
            rolling_year_splits(
                observed,
                minimum_training_years=minimum_training_years,
            )
        )
        if max_folds is not None:
            if int(max_folds) < 1:
                raise ValueError("max_folds must be at least one when specified")
            count = min(count, int(max_folds))
        expected[str(country)] = int(count)
    return expected


def _is_boolean_like(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return True
    return isinstance(value, str) and value.strip().lower() in {"true", "false"}


def _boolean_value(value: Any) -> bool:
    if not _is_boolean_like(value):
        raise ValueError(f"Expected a complete Boolean value, received {value!r}")
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def hindcast_execution_failures(
    result: pd.DataFrame,
    expected_fold_counts: dict[str, int],
) -> list[str]:
    """Return hard failures that make the predictive assessment unauditable."""

    failures: list[str] = []
    if result.empty and sum(expected_fold_counts.values()) == 0:
        return failures
    required = {
        "country",
        "fold",
        "training_end_year",
        "test_year",
        "evidence_cutoff_date",
        "resistance_evidence_anchor_year",
        "resistance_timeline_applied",
        "resistance_evidence_years",
        "diagnostic_periods_known_at_origin",
        "future_evidence_used",
        "optimizer_success",
        "observation_predictive_95_coverage",
        "state_coordinate_uncertainty_mode",
        "predictive_uncertainty_scope",
        "posterior_coverage_claimed",
        *HINDCAST_FINITE_COLUMNS,
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        return [f"hindcast output is missing required columns: {missing}"]

    observed_countries = set(result["country"].astype(str))
    unexpected = sorted(observed_countries.difference(expected_fold_counts))
    if unexpected:
        failures.append(f"unexpected countries in hindcast output: {unexpected}")
    for country, expected_count in expected_fold_counts.items():
        country_rows = result.loc[result["country"].astype(str).eq(str(country))]
        if len(country_rows) != int(expected_count):
            failures.append(
                f"{country}: expected {expected_count} folds but obtained {len(country_rows)}"
            )
        if country_rows.empty:
            continue
        if country_rows[["fold", "test_year"]].duplicated().any():
            failures.append(f"{country}: duplicate fold/test-year rows")
        optimizer_success = country_rows["optimizer_success"]
        if not bool(optimizer_success.map(_is_boolean_like).all()):
            failures.append(f"{country}: optimizer success flags are incomplete")
        elif not bool(optimizer_success.map(_boolean_value).all()):
            failures.append(f"{country}: one or more fold optimizers did not succeed")

    training_end_year = pd.to_numeric(
        result["training_end_year"], errors="coerce"
    ).to_numpy(dtype=float)
    test_year = pd.to_numeric(
        result["test_year"], errors="coerce"
    ).to_numpy(dtype=float)
    resistance_anchor = pd.to_numeric(
        result["resistance_evidence_anchor_year"], errors="coerce"
    ).to_numpy(dtype=float)
    if not (
        np.isfinite(training_end_year).all()
        and np.isfinite(test_year).all()
        and np.isfinite(resistance_anchor).all()
    ):
        failures.append("historical evidence-cutoff years must be finite")
    else:
        if np.any(training_end_year >= test_year):
            failures.append("hindcast training years must precede test years")
        if np.any(resistance_anchor > training_end_year):
            failures.append(
                "resistance evidence anchor exceeds the training-data horizon"
            )
    expected_cutoff = pd.to_datetime(
        pd.Series(test_year).astype("Int64").astype(str) + "-01-01",
        errors="coerce",
    )
    observed_cutoff = pd.to_datetime(
        result["evidence_cutoff_date"], errors="coerce"
    ).reset_index(drop=True)
    if observed_cutoff.isna().any() or not bool(
        observed_cutoff.eq(expected_cutoff).all()
    ):
        failures.append(
            "hindcast evidence cutoffs must equal the start of each untouched test year"
        )
    future_evidence = result["future_evidence_used"]
    if not bool(future_evidence.map(_is_boolean_like).all()):
        failures.append("future-evidence audit flags are incomplete")
    elif bool(future_evidence.map(_boolean_value).any()):
        failures.append("one or more hindcast folds used future evidence")
    for index, raw_years in enumerate(result["resistance_evidence_years"].fillna("")):
        parsed = [
            int(value)
            for value in str(raw_years).split(";")
            if value.strip()
        ]
        if parsed and max(parsed) > int(training_end_year[index]):
            failures.append(
                "one or more resistance evidence rows postdate the training horizon"
            )
            break
    diagnostic_periods = pd.to_numeric(
        result["diagnostic_periods_known_at_origin"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(diagnostic_periods).all() or np.any(
        diagnostic_periods < 0.0
    ):
        failures.append(
            "diagnostic periods known at the forecast origin must be nonnegative"
        )

    for column in HINDCAST_FINITE_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            failures.append(f"{column}: contains non-finite values")
    for column in ("nb95_interval_coverage", "naive_nb95_interval_coverage"):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            failures.append(f"{column}: coverage must lie in [0, 1]")
    q025 = pd.to_numeric(
        result["observation_predictive_q025"], errors="coerce"
    ).to_numpy(dtype=float)
    median = pd.to_numeric(
        result["observation_predictive_median"], errors="coerce"
    ).to_numpy(dtype=float)
    q975 = pd.to_numeric(
        result["observation_predictive_q975"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any((q025 > median) | (median > q975)):
        failures.append("observation predictive quantiles are not ordered")
    process_q025 = pd.to_numeric(
        result["process_predictive_mean_q025"], errors="coerce"
    ).to_numpy(dtype=float)
    process_median = pd.to_numeric(
        result["process_predictive_mean_median"], errors="coerce"
    ).to_numpy(dtype=float)
    process_q975 = pd.to_numeric(
        result["process_predictive_mean_q975"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any((process_q025 > process_median) | (process_median > process_q975)):
        failures.append("process predictive quantiles are not ordered")
    predictive_coverage = result["observation_predictive_95_coverage"]
    if not bool(predictive_coverage.map(_is_boolean_like).all()):
        failures.append("observation predictive coverage flags are incomplete")
    modes = result["state_coordinate_uncertainty_mode"]
    if modes.isna().any() or not bool(
        modes.astype(str).eq(PUBLICATION_STATE_UNCERTAINTY_MODE).all()
    ):
        failures.append(
            "state coordinate uncertainty mode must be complete and map_conditioned"
        )
    scopes = result["predictive_uncertainty_scope"]
    if scopes.isna().any() or not bool(
        scopes.astype(str).eq(PUBLICATION_PREDICTIVE_UNCERTAINTY_SCOPE).all()
    ):
        failures.append(
            "predictive uncertainty scope must match the publication contract"
        )
    posterior_claim = result["posterior_coverage_claimed"]
    if not bool(posterior_claim.map(_is_boolean_like).all()):
        failures.append("posterior coverage claim flags are incomplete")
    elif bool(posterior_claim.map(_boolean_value).any()):
        failures.append(
            "conditional hindcasts must not claim posterior interval coverage"
        )
    nonnegative_columns = (
        "observed_cases",
        "predicted_cases",
        "naive_predicted_cases",
        "nb95_interval_mean_width",
        "naive_nb95_interval_mean_width",
        "observation_predictive_95_width",
        "observation_predictive_95_relative_width",
        "observation_predictive_95_log1p_width",
    )
    for column in nonnegative_columns:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if np.any(values < 0.0):
            failures.append(f"{column}: contains negative values")
    for column in (
        "predictive_process_draws",
        "predictive_measurement_draws",
        "posterior_rank",
        "posterior_dimension",
        "process_nfev",
    ):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if np.any(values <= 0.0):
            failures.append(f"{column}: must be positive for every fold")
    ranks = pd.to_numeric(result["posterior_rank"], errors="coerce").to_numpy(dtype=float)
    dimensions = pd.to_numeric(
        result["posterior_dimension"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(ranks != dimensions):
        failures.append("one or more fold covariance approximations are rank deficient")
    rejection = pd.to_numeric(
        result["laplace_boundary_rejection_fraction"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any((rejection < 0.0) | (rejection > 1.0)):
        failures.append("Laplace boundary rejection fractions must lie in [0, 1]")
    return failures


def _aggregate_hindcast_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute paired model-vs-naive and calibration/sharpness summaries."""

    if frame.empty:
        return {
            "fold_count": 0,
            "mean_absolute_log1p_error": np.nan,
            "mean_naive_absolute_log1p_error": np.nan,
            "absolute_log1p_error_improvement_over_naive": np.nan,
            "total_negative_binomial_nll": np.nan,
            "total_naive_negative_binomial_nll": np.nan,
            "negative_binomial_nll_improvement_over_naive": np.nan,
            "point_error_fold_win_fraction": np.nan,
            "nb_score_fold_win_fraction": np.nan,
            "mean_nb95_interval_coverage": np.nan,
            "mean_naive_nb95_interval_coverage": np.nan,
            "observation_predictive_95_coverage": np.nan,
            "median_observation_predictive_95_width": np.nan,
            "median_observation_predictive_95_relative_width": np.nan,
            "median_observation_predictive_95_log1p_width": np.nan,
        }
    model_error = pd.to_numeric(frame["absolute_log1p_error"], errors="raise")
    naive_error = pd.to_numeric(
        frame["naive_absolute_log1p_error"], errors="raise"
    )
    model_score_column = (
        "process_mixture_negative_binomial_nll"
        if "process_mixture_negative_binomial_nll" in frame.columns
        else "negative_binomial_nll"
    )
    model_nll = pd.to_numeric(frame[model_score_column], errors="raise")
    naive_nll = pd.to_numeric(
        frame["naive_negative_binomial_nll"], errors="raise"
    )
    mean_model_error = float(model_error.mean())
    mean_naive_error = float(naive_error.mean())
    total_model_nll = float(model_nll.sum())
    total_naive_nll = float(naive_nll.sum())
    return {
        "fold_count": int(len(frame)),
        "mean_absolute_log1p_error": mean_model_error,
        "mean_naive_absolute_log1p_error": mean_naive_error,
        "absolute_log1p_error_improvement_over_naive": float(
            mean_naive_error - mean_model_error
        ),
        "total_negative_binomial_nll": total_model_nll,
        "total_naive_negative_binomial_nll": total_naive_nll,
        "negative_binomial_nll_improvement_over_naive": float(
            total_naive_nll - total_model_nll
        ),
        "point_error_fold_win_fraction": float(np.mean(model_error < naive_error)),
        "nb_score_fold_win_fraction": float(np.mean(model_nll < naive_nll)),
        "mean_nb95_interval_coverage": float(
            pd.to_numeric(frame["nb95_interval_coverage"], errors="raise").mean()
        ),
        "mean_naive_nb95_interval_coverage": float(
            pd.to_numeric(
                frame["naive_nb95_interval_coverage"], errors="raise"
            ).mean()
        ),
        "observation_predictive_95_coverage": float(
            frame["observation_predictive_95_coverage"].map(_boolean_value).mean()
        ),
        "median_observation_predictive_95_width": float(
            pd.to_numeric(
                frame["observation_predictive_95_width"], errors="raise"
            ).median()
        ),
        "median_observation_predictive_95_relative_width": float(
            pd.to_numeric(
                frame["observation_predictive_95_relative_width"], errors="raise"
            ).median()
        ),
        "median_observation_predictive_95_log1p_width": float(
            pd.to_numeric(
                frame["observation_predictive_95_log1p_width"], errors="raise"
            ).median()
        ),
    }


def _predictive_adequacy_reasons(
    metrics: dict[str, Any],
    *,
    execution_complete: bool,
    expected_fold_count: int,
) -> list[str]:
    reasons: list[str] = []
    if not execution_complete:
        return ["execution_incomplete"]
    if expected_fold_count < MINIMUM_FOLDS_FOR_PREDICTIVE_ADEQUACY:
        return ["insufficient_folds_for_predictive_assessment"]
    if metrics["absolute_log1p_error_improvement_over_naive"] < 0.0:
        reasons.append("absolute_log1p_error_not_better_than_naive")
    if metrics["negative_binomial_nll_improvement_over_naive"] < 0.0:
        reasons.append("negative_binomial_score_not_better_than_naive")
    if (
        metrics["observation_predictive_95_coverage"]
        < MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE
    ):
        reasons.append("observation_predictive_undercoverage")
    if (
        metrics["median_observation_predictive_95_log1p_width"]
        > MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH
    ):
        reasons.append("observation_predictive_interval_non_discriminating")
    return reasons


def _uncertainty_scope_reasons(frame: pd.DataFrame) -> list[str]:
    """Reject an uncorrected local Gaussian approximation as publication evidence."""

    if frame.empty:
        return []
    required = {
        "state_coordinate_uncertainty_mode",
        "predictive_uncertainty_scope",
        "posterior_coverage_claimed",
    }
    if not required.issubset(frame.columns):
        return ["predictive_uncertainty_scope_metadata_missing"]
    modes = set(frame["state_coordinate_uncertainty_mode"].dropna().astype(str))
    if "gaussian_approximation" in modes:
        return ["uncorrected_gaussian_state_approximation_not_publication_valid"]
    unexpected = sorted(modes.difference({"map_conditioned"}))
    if unexpected:
        return ["unknown_predictive_uncertainty_scope"]
    scopes = set(frame["predictive_uncertainty_scope"].dropna().astype(str))
    if scopes != {PUBLICATION_PREDICTIVE_UNCERTAINTY_SCOPE}:
        return ["predictive_uncertainty_scope_contract_mismatch"]
    claims = frame["posterior_coverage_claimed"]
    if (
        not bool(claims.map(_is_boolean_like).all())
        or bool(claims.map(_boolean_value).any())
    ):
        return ["conditional_interval_mislabelled_as_posterior_coverage"]
    return []


def summarize_hindcasts(
    result: pd.DataFrame,
    expected_fold_counts: dict[str, int],
) -> pd.DataFrame:
    """Create country and overall predictive-adequacy assessments.

    Coverage is necessary but never sufficient: a pass also requires both
    paired scores to beat the exposure-adjusted last-year forecast and the
    median 95% interval to remain informative on the log1p scale.
    """

    rows: list[dict[str, Any]] = []
    country_passes: list[bool] = []
    for country, expected_count in expected_fold_counts.items():
        frame = (
            result.loc[result["country"].astype(str).eq(str(country))].copy()
            if "country" in result.columns
            else result.iloc[0:0].copy()
        )
        execution_failures = hindcast_execution_failures(frame, {country: expected_count})
        execution_complete = not execution_failures
        metrics = _aggregate_hindcast_metrics(frame)
        reasons = _predictive_adequacy_reasons(
            metrics,
            execution_complete=execution_complete,
            expected_fold_count=expected_count,
        )
        reasons.extend(_uncertainty_scope_reasons(frame))
        reasons = list(dict.fromkeys(reasons))
        adequate = not reasons
        country_passes.append(adequate)
        rows.append(
            {
                "assessment_scope": "country",
                "country": country,
                "expected_fold_count": int(expected_count),
                "execution_complete": bool(execution_complete),
                "execution_failures": "; ".join(execution_failures),
                **metrics,
                "negative_binomial_score_type": (
                    "future_process_mixture_NB2_log_score"
                    if "process_mixture_negative_binomial_nll" in frame.columns
                    else "conditional_mean_NB2_log_score"
                ),
                "minimum_folds_required": MINIMUM_FOLDS_FOR_PREDICTIVE_ADEQUACY,
                "minimum_predictive_coverage_required": MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE,
                "maximum_predictive_log1p_width_allowed": MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH,
                "predictive_adequacy_pass": bool(adequate),
                "predictive_adequacy_status": (
                    "pass_limited_fold_evidence"
                    if adequate and expected_count < 5
                    else "pass"
                    if adequate
                    else "fail"
                ),
                "predictive_adequacy_reasons": "; ".join(reasons),
                "evidence_strength": "limited" if expected_count < 5 else "moderate",
            }
        )

    all_execution_failures = hindcast_execution_failures(result, expected_fold_counts)
    all_metrics = _aggregate_hindcast_metrics(result)
    total_expected = int(sum(expected_fold_counts.values()))
    overall_reasons = _predictive_adequacy_reasons(
        all_metrics,
        execution_complete=not all_execution_failures,
        expected_fold_count=total_expected,
    )
    overall_reasons.extend(_uncertainty_scope_reasons(result))
    if not all(country_passes):
        overall_reasons.append("one_or_more_country_assessments_failed")
    overall_reasons = list(dict.fromkeys(overall_reasons))
    overall_adequate = not overall_reasons
    rows.append(
        {
            "assessment_scope": "overall",
            "country": "__overall__",
            "expected_fold_count": total_expected,
            "execution_complete": not all_execution_failures,
            "execution_failures": "; ".join(all_execution_failures),
            **all_metrics,
            "negative_binomial_score_type": (
                "future_process_mixture_NB2_log_score"
                if "process_mixture_negative_binomial_nll" in result.columns
                else "conditional_mean_NB2_log_score"
            ),
            "minimum_folds_required": MINIMUM_FOLDS_FOR_PREDICTIVE_ADEQUACY,
            "minimum_predictive_coverage_required": MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE,
            "maximum_predictive_log1p_width_allowed": MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH,
            "predictive_adequacy_pass": bool(overall_adequate),
            "predictive_adequacy_status": "pass" if overall_adequate else "fail",
            "predictive_adequacy_reasons": "; ".join(overall_reasons),
            "evidence_strength": (
                "limited"
                if total_expected < 10
                else "moderate_with_limited_country_level_folds"
            ),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-origin calibration hindcasts")
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--minimum-training-years", type=int, default=4)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--predictive-draws", type=int, default=64)
    parser.add_argument(
        "--state-coordinate-uncertainty-mode",
        choices=("map_conditioned", "gaussian_approximation"),
        default="map_conditioned",
        help=(
            "Use conditional-on-MAP predictive simulation by default. The local "
            "Gaussian option is diagnostic only and cannot pass the publication gate."
        ),
    )
    parser.add_argument(
        "--require-predictive-adequacy",
        action="store_true",
        help=(
            "Exit non-zero unless every country passes the prespecified "
            "model-vs-naive, coverage, and interval-sharpness criteria."
        ),
    )
    args = parser.parse_args()

    configs = load_configs()
    publication_countries = publication_country_names(configs)
    countries = args.countries or publication_countries
    outside_publication_scope = sorted(set(countries) - set(publication_countries))
    if outside_publication_scope:
        raise ValueError(
            "Calibration publication hindcasts exclude countries outside the "
            "prespecified publication set: " + ", ".join(outside_publication_scope)
        )
    expected_fold_counts = expected_hindcast_fold_counts(
        countries,
        minimum_training_years=args.minimum_training_years,
        max_folds=args.max_folds,
    )
    result = run_hindcasts(
        countries,
        n_jobs=args.n_jobs,
        minimum_training_years=args.minimum_training_years,
        max_folds=args.max_folds,
        predictive_draws=args.predictive_draws,
        state_coordinate_uncertainty_mode=args.state_coordinate_uncertainty_mode,
    )
    write_dataframe(
        result,
        project_path("outputs", "tables", "calibration_rolling_hindcast.csv"),
    )
    summary = summarize_hindcasts(result, expected_fold_counts)
    write_dataframe(
        summary,
        project_path(
            "outputs", "tables", "calibration_rolling_hindcast_summary.csv"
        ),
    )
    execution_failures = hindcast_execution_failures(result, expected_fold_counts)
    if sum(expected_fold_counts.values()) < 1:
        execution_failures.append("no data-supported rolling-origin folds are available")
    overall = summary.loc[summary["assessment_scope"].eq("overall")].iloc[0]
    predictive_adequacy_pass = bool(overall["predictive_adequacy_pass"])
    gate = pd.DataFrame(
        [
            {
                "execution_gate_pass": not execution_failures,
                "predictive_adequacy_pass": predictive_adequacy_pass,
                "publication_gate_pass": bool(
                    not execution_failures and predictive_adequacy_pass
                ),
                "gate_status": (
                    "execution_failed"
                    if execution_failures
                    else "predictive_adequacy_passed"
                    if predictive_adequacy_pass
                    else "predictive_adequacy_failed"
                ),
                "execution_failures": "; ".join(execution_failures),
                "predictive_adequacy_reasons": str(
                    overall["predictive_adequacy_reasons"]
                ),
                "expected_fold_count": int(sum(expected_fold_counts.values())),
                "completed_fold_count": int(len(result)),
                # Publication always requires predictive adequacy.  The CLI
                # switch below controls only whether this diagnostic command
                # exits non-zero; it must not change the scientific release
                # contract recorded in the gate artifact.
                "publication_requires_predictive_adequacy": True,
                "cli_exit_requires_predictive_adequacy": bool(
                    args.require_predictive_adequacy
                ),
                "state_coordinate_uncertainty_mode": (
                    args.state_coordinate_uncertainty_mode
                ),
                "predictive_uncertainty_scope": (
                    PUBLICATION_PREDICTIVE_UNCERTAINTY_SCOPE
                ),
                "posterior_coverage_claimed": False,
            }
        ]
    )
    write_dataframe(
        gate,
        project_path("outputs", "tables", "calibration_rolling_hindcast_gate.csv"),
    )
    fold_years_by_country = {
        str(country): sorted(
            pd.to_numeric(
                result.loc[result["country"].astype(str).eq(str(country)), "test_year"],
                errors="raise",
            )
            .astype(int)
            .unique()
            .tolist()
        )
        for country in countries
    }
    write_run_metadata(
        CALIBRATION_HINDCAST_STEM,
        current_run_metadata(
            CALIBRATION_HINDCAST_STEM,
            row_counts={
                "fold_scores": int(len(result)),
                "country_summaries": int(
                    summary["assessment_scope"].eq("country").sum()
                ),
                "gate_rows": int(len(gate)),
            },
        )
        | {
            "countries_included": list(countries),
            "expected_fold_counts": {
                str(country): int(count)
                for country, count in expected_fold_counts.items()
            },
            "fold_test_years": fold_years_by_country,
            "execution_gate_pass": bool(not execution_failures),
            "predictive_adequacy_pass": bool(predictive_adequacy_pass),
            "publication_gate_pass": bool(
                not execution_failures and predictive_adequacy_pass
            ),
            "state_coordinate_uncertainty_mode": str(
                args.state_coordinate_uncertainty_mode
            ),
            "predictive_uncertainty_scope": (
                PUBLICATION_PREDICTIVE_UNCERTAINTY_SCOPE
            ),
            "posterior_coverage_claimed": False,
            "minimum_folds_for_predictive_adequacy": (
                MINIMUM_FOLDS_FOR_PREDICTIVE_ADEQUACY
            ),
            "minimum_observation_predictive_coverage": (
                MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE
            ),
            "maximum_observation_predictive_log1p_width": (
                MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH
            ),
        },
    )
    if execution_failures:
        raise RuntimeError(
            "Calibration hindcast execution gate failed: "
            + "; ".join(execution_failures)
        )
    if args.require_predictive_adequacy and not predictive_adequacy_pass:
        raise RuntimeError(
            "Calibration hindcast predictive-adequacy gate failed: "
            + str(overall["predictive_adequacy_reasons"])
        )


if __name__ == "__main__":
    main()
