from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src_python.calibration.likelihood import negative_binomial_nll
from src_python.simulation.common import (
    calibration_config_fingerprint,
    calibrated_country_artifact_path,
    config_fingerprint,
    load_configs,
    make_config,
    run_prepared_config,
)
from src_python.utils.io import project_path, write_dataframe, write_yaml


CALIBRATION_BOUNDS: list[tuple[float, float]] = [
    (np.log(0.002), np.log(0.2)),
    (np.log(0.01), np.log(10.0)),
    (-8.0, 8.0),
    (np.log(0.01), np.log(2.0)),
    (-8.0, 8.0),
]


def _logit(x: float) -> float:
    x = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    return float(np.log(x / (1.0 - x)))


def _inv_logit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _clip_to_bounds(vector: np.ndarray) -> np.ndarray:
    lower = np.array([bound[0] for bound in CALIBRATION_BOUNDS], dtype=float)
    upper = np.array([bound[1] for bound in CALIBRATION_BOUNDS], dtype=float)
    return np.clip(np.asarray(vector, dtype=float), lower, upper)


def apply_calibration_vector(config: dict[str, Any], vector: np.ndarray) -> dict[str, Any]:
    beta_s = float(np.exp(vector[0]))
    reporting_multiplier = float(np.exp(vector[1]))
    seasonal_amplitude = float(0.35 * _inv_logit(vector[2]))
    importation_rate = float(np.exp(vector[3]))
    importation_fraction = float(_inv_logit(vector[4]))

    out = deepcopy(config)
    out["transmission"]["beta_S"] = beta_s
    out["transmission"]["seasonal_amplitude"] = seasonal_amplitude
    out["reporting_multiplier"] = reporting_multiplier
    out["importation"]["rate_per_100k_per_year"] = importation_rate
    out.setdefault("resistance", {})["importation_fraction"] = importation_fraction
    out["importation"]["resistant_fraction"] = importation_fraction
    return out


def initial_calibration_vector(config: dict[str, Any]) -> np.ndarray:
    return _clip_to_bounds(
        np.array(
            [
                np.log(float(config["transmission"]["beta_S"])),
                np.log(float(config.get("reporting_multiplier", 1.0))),
                _logit(float(config["transmission"].get("seasonal_amplitude", 0.0)) / 0.35),
                np.log(float(config.get("importation", {}).get("rate_per_100k_per_year", 0.2))),
                _logit(float(config.get("importation", {}).get("resistant_fraction", 0.3))),
            ],
            dtype=float,
        )
    )


def calibration_start_vectors(config: dict[str, Any], *, n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [initial_calibration_vector(config)]
    jitter_scales = np.array([0.25, 0.35, 0.8, 0.35, 0.8], dtype=float)
    for _ in range(max(0, n_starts - 1)):
        jitter = rng.normal(loc=0.0, scale=jitter_scales, size=len(jitter_scales))
        starts.append(_clip_to_bounds(starts[0] + jitter))
    return starts


def reporting_rate_prior_penalty(config: dict[str, Any]) -> float:
    prior = config.get("reporting_rate_prior", {})
    if prior.get("method") != "literature_range":
        return 0.0

    age_bands = prior.get("age_groups", {})
    weight = float(prior.get("weight", 1.0))
    multipliers = _reporting_prior_endpoint_multipliers(config)

    penalty = 0.0
    for record in config.get("age_groups", []):
        label = record.get("label", "")
        band = age_bands.get(label)
        if not band:
            continue

        base_rate = float(record.get("reporting_rate", 0.0))
        lower = float(band.get("lower", base_rate))
        upper = float(band.get("upper", base_rate))
        width = max(upper - lower, 1e-6)
        for multiplier in multipliers:
            target_rate = float(np.clip(base_rate * multiplier, 0.0, 1.0))
            if lower <= target_rate <= upper:
                continue
            if target_rate < lower:
                penalty += ((lower - target_rate) / width) ** 2
            else:
                penalty += ((target_rate - upper) / width) ** 2

    return float(weight * penalty)


def _reporting_prior_endpoint_multipliers(config: dict[str, Any]) -> list[float]:
    base_multiplier = float(config.get("reporting_multiplier", 1.0))
    variation = config.get("reporting_time_variation", {})
    if not isinstance(variation, dict) or not variation:
        return [base_multiplier]
    endpoints = [
        float(variation.get("start_multiplier", 1.0)),
        float(variation.get("end_multiplier", 1.0)),
    ]
    values = sorted({round(base_multiplier * value, 12) for value in endpoints})
    return [float(value) for value in values]


def reporting_multiplier_prior_bounds(config: dict[str, Any]) -> tuple[float, float]:
    prior = config.get("reporting_rate_prior", {})
    if prior.get("method") != "literature_range":
        return 0.1, 10.0
    lower_bound = 0.1
    upper_bound = 10.0
    age_bands = prior.get("age_groups", {})
    for record in config.get("age_groups", []):
        band = age_bands.get(record.get("label", ""))
        if not band:
            continue
        base_rate = max(float(record.get("reporting_rate", 0.0)), 1e-9)
        lower_bound = max(lower_bound, float(band.get("lower", 0.0)) / base_rate)
        upper_bound = min(upper_bound, float(band.get("upper", 1.0)) / base_rate)
    return float(lower_bound), float(max(lower_bound, upper_bound))


def observed_annual_case_frame(country: str) -> pd.DataFrame:
    configs = load_configs()
    incidence_observed = _observed_incidence_case_frame(country)
    if not incidence_observed.empty:
        return incidence_observed

    surveillance_year = _surveillance_year_for_legacy_sources(configs)
    who_path = project_path("data/processed/who_pertussis_reported_cases.csv")
    if who_path.exists():
        who = pd.read_csv(who_path)
        who_country = who.loc[who["config_key"].eq(country)].copy()
        if not who_country.empty:
            who_country["reported_cases"] = pd.to_numeric(who_country["reported_cases"], errors="coerce")
            if surveillance_year is not None:
                who_country = who_country.loc[who_country["year"].le(surveillance_year)].copy()
            observed = (
                who_country.dropna(subset=["reported_cases"])
                .sort_values("year")
                .reset_index(drop=True)
            )
            if not observed.empty:
                observed["series_year"] = np.arange(len(observed), dtype=int)
                observed = observed.rename(columns={"year": "observed_year"})
                return observed.loc[:, ["series_year", "observed_year", "reported_cases"]]

    path = project_path("data/processed/pertussis_incidence_timeseries.csv")
    observed = pd.read_csv(path)
    observed = observed.loc[observed["config_key"].eq(country)].copy()
    if surveillance_year is not None:
        observed = observed.loc[observed["Year"].le(surveillance_year)].copy()
    if observed.empty:
        incidence = configs["countries"][country].get("observed_incidence", {})
        population = float(configs["countries"][country]["total_population"])
        mean_incidence = float(incidence.get("observed_mean_annual_reported_incidence_per_100k", 0.0))
        fallback_year = int(configs["data_sources"].get("surveillance_year", configs["data_sources"].get("analysis_year", 2023)))
        return pd.DataFrame(
            {
                "series_year": [0],
                "observed_year": [fallback_year],
                "reported_cases": [mean_incidence * population / 100_000.0],
            }
        )

    observed["Cases"] = pd.to_numeric(observed["Cases"], errors="coerce").fillna(0.0)
    observed = (
        observed.groupby("Year", as_index=False)["Cases"]
        .sum()
        .sort_values("Year")
        .reset_index(drop=True)
        .rename(columns={"Year": "observed_year", "Cases": "reported_cases"})
    )
    observed["series_year"] = np.arange(len(observed), dtype=int)
    return observed.loc[:, ["series_year", "observed_year", "reported_cases"]]


def observed_annual_cases(country: str) -> np.ndarray:
    return observed_annual_case_frame(country)["reported_cases"].to_numpy(dtype=float)


def _surveillance_year_for_legacy_sources(configs: dict[str, Any]) -> int | None:
    data_sources = configs["data_sources"]
    if str(data_sources.get("incidence_cutoff_policy", "surveillance_year")) == "all_records":
        return None
    return int(data_sources.get("surveillance_year", data_sources.get("analysis_year", 2023)))


def _observed_incidence_case_frame(country: str) -> pd.DataFrame:
    path = project_path("data/processed/pertussis_incidence_timeseries.csv")
    if not path.exists():
        return pd.DataFrame()
    observed = pd.read_csv(path)
    if "config_key" not in observed.columns:
        return pd.DataFrame()
    observed = observed.loc[observed["config_key"].eq(country)].copy()
    if observed.empty:
        return pd.DataFrame()

    configs = load_configs()
    surveillance_year = _surveillance_year_for_legacy_sources(configs)
    observed["Year"] = pd.to_numeric(observed["Year"], errors="coerce")
    if surveillance_year is not None:
        observed = observed.loc[observed["Year"].le(surveillance_year)].copy()
    observed["reported_cases"] = pd.to_numeric(observed["Cases"], errors="coerce").fillna(0.0)

    if {"period_start", "period_end"}.issubset(observed.columns):
        observed["period_start"] = pd.to_datetime(observed["period_start"], errors="coerce")
        observed["period_end"] = pd.to_datetime(observed["period_end"], errors="coerce")
        observed = observed.dropna(subset=["period_start", "period_end", "reported_cases"]).copy()
        observed = observed.loc[observed["period_end"].gt(observed["period_start"])].copy()
        if not observed.empty:
            observed = observed.sort_values(["period_start", "period_end"]).reset_index(drop=True)
            observed["series_year"] = np.arange(len(observed), dtype=int)
            observed["observed_year"] = observed["period_start"].dt.year.astype(int)
            observed["interval_days"] = (observed["period_end"] - observed["period_start"]).dt.days.astype(float)
            observed["observed_interval_id"] = (
                observed["period_start"].dt.strftime("%Y-%m-%d")
                + "_"
                + observed["period_end"].dt.strftime("%Y-%m-%d")
            )
            columns = [
                "series_year",
                "observed_year",
                "observed_interval_id",
                "period_start",
                "period_end",
                "interval_days",
                "reported_cases",
            ]
            if "reporting_frequency" in observed.columns:
                columns.append("reporting_frequency")
            return observed.loc[:, columns]

    observed = (
        observed.dropna(subset=["Year"])
        .groupby("Year", as_index=False)["reported_cases"]
        .sum()
        .sort_values("Year")
        .reset_index(drop=True)
        .rename(columns={"Year": "observed_year"})
    )
    observed["observed_year"] = observed["observed_year"].astype(int)
    observed["series_year"] = np.arange(len(observed), dtype=int)
    return observed.loc[:, ["series_year", "observed_year", "reported_cases"]]


def annual_reported_cases(timeseries: pd.DataFrame) -> pd.DataFrame:
    out = timeseries.copy()
    if "calendar_date" in out.columns and pd.to_datetime(out["calendar_date"], errors="coerce").notna().any():
        annual = _annual_reported_cases_by_calendar_interval(out)
        if not annual.empty:
            return annual
    if "calendar_year" in out.columns and pd.to_numeric(out["calendar_year"], errors="coerce").notna().any():
        out["observed_year"] = pd.to_numeric(out["calendar_year"], errors="coerce").astype("Int64")
        annual = (
            out.dropna(subset=["observed_year"])
            .groupby("observed_year", as_index=False)
            .agg(reported_cases=("reported_cases", "sum"))
            .sort_values("observed_year")
            .reset_index(drop=True)
        )
        annual["series_year"] = np.arange(len(annual), dtype=int)
        return annual.loc[:, ["series_year", "observed_year", "reported_cases"]]
    out["series_year"] = np.floor((out["time"] - float(out["time"].min())) / 365.0).astype(int)
    annual = (
        out.groupby("series_year", as_index=False)
        .agg(reported_cases=("reported_cases", "sum"))
        .sort_values("series_year")
        .reset_index(drop=True)
    )
    annual["observed_year"] = annual["series_year"]
    return annual.loc[:, ["series_year", "observed_year", "reported_cases"]]


def reported_cases_for_observed_intervals(timeseries: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    if not {"observed_interval_id", "period_start", "period_end"}.issubset(observed.columns):
        return annual_reported_cases(timeseries)
    out = timeseries.copy()
    if "calendar_date" not in out.columns:
        return annual_reported_cases(out)
    out["calendar_date_value"] = pd.to_datetime(out["calendar_date"], errors="coerce")
    out = out.dropna(subset=["calendar_date_value", "time"])
    if out.empty:
        return pd.DataFrame(columns=["series_year", "observed_year", "observed_interval_id", "reported_cases"])

    interval_frame = observed.copy()
    interval_frame["period_start"] = pd.to_datetime(interval_frame["period_start"], errors="coerce")
    interval_frame["period_end"] = pd.to_datetime(interval_frame["period_end"], errors="coerce")
    interval_frame = interval_frame.dropna(subset=["period_start", "period_end"]).reset_index(drop=True)
    interval_starts = interval_frame["period_start"].to_numpy(dtype="datetime64[ns]")
    interval_ends = interval_frame["period_end"].to_numpy(dtype="datetime64[ns]")
    allocations = np.zeros(len(interval_frame), dtype=float)

    group_cols = [
        col
        for col in ("analysis", "scenario", "vaccine_scenario", "resistance_scenario", "intervention", "age_group", "strain")
        if col in out.columns
    ]
    grouped = out.sort_values(group_cols + ["time"]).groupby(group_cols, dropna=False) if group_cols else [((), out)]
    for _, group in grouped:
        previous_date: pd.Timestamp | None = None
        for row in group.sort_values("time").itertuples(index=False):
            current_date = pd.Timestamp(getattr(row, "calendar_date_value")).normalize()
            count = float(getattr(row, "reported_cases", 0.0))
            if previous_date is None:
                previous_date = current_date
                continue
            if current_date <= previous_date:
                previous_date = current_date
                continue
            span_days = float((current_date - previous_date).days)
            segment_start = np.datetime64(previous_date.to_datetime64())
            segment_end = np.datetime64(current_date.to_datetime64())
            overlap_start = np.maximum(interval_starts, segment_start)
            overlap_end = np.minimum(interval_ends, segment_end)
            overlap_days = (overlap_end - overlap_start).astype("timedelta64[D]").astype(float)
            mask = overlap_days > 0.0
            if mask.any():
                allocations[mask] += count * overlap_days[mask] / span_days
            previous_date = current_date

    predicted = interval_frame.loc[:, ["series_year", "observed_year", "observed_interval_id"]].copy()
    predicted["reported_cases"] = allocations
    return predicted


def _annual_reported_cases_by_calendar_interval(timeseries: pd.DataFrame) -> pd.DataFrame:
    out = timeseries.copy()
    out["calendar_date_value"] = pd.to_datetime(out["calendar_date"], errors="coerce").dt.date
    out = out.dropna(subset=["calendar_date_value", "time"])
    if out.empty:
        return pd.DataFrame(columns=["series_year", "observed_year", "reported_cases"])

    group_cols = [
        col
        for col in ("analysis", "scenario", "vaccine_scenario", "resistance_scenario", "intervention", "age_group", "strain")
        if col in out.columns
    ]
    if not group_cols:
        group_cols = ["age_group", "strain"] if {"age_group", "strain"}.issubset(out.columns) else []

    allocations: dict[int, float] = {}
    grouped = out.sort_values(group_cols + ["time"]).groupby(group_cols, dropna=False) if group_cols else [((), out)]
    for _, group in grouped:
        previous_date: date | None = None
        for row in group.sort_values("time").itertuples(index=False):
            current_date = getattr(row, "calendar_date_value")
            count = float(getattr(row, "reported_cases", 0.0))
            if previous_date is None:
                previous_date = current_date
                continue
            if current_date <= previous_date:
                allocations[current_date.year] = allocations.get(current_date.year, 0.0) + count
                previous_date = current_date
                continue

            span_days = float((current_date - previous_date).days)
            for year in range(previous_date.year, current_date.year + 1):
                year_start = date(year, 1, 1)
                year_end = date(year + 1, 1, 1)
                segment_start = max(previous_date, year_start)
                segment_end = min(current_date, year_end)
                segment_days = float(max(0, (segment_end - segment_start).days))
                if segment_days > 0.0:
                    allocations[year] = allocations.get(year, 0.0) + count * segment_days / span_days
            previous_date = current_date

    if not allocations:
        return pd.DataFrame(columns=["series_year", "observed_year", "reported_cases"])
    annual = pd.DataFrame(
        {
            "observed_year": sorted(allocations),
            "reported_cases": [allocations[year] for year in sorted(allocations)],
        }
    )
    annual["series_year"] = np.arange(len(annual), dtype=int)
    return annual.loc[:, ["series_year", "observed_year", "reported_cases"]]


def align_annual_case_series(predicted: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    if "observed_interval_id" in predicted.columns and "observed_interval_id" in observed.columns:
        aligned = (
            predicted.rename(columns={"reported_cases": "predicted_reported_cases"})
            .merge(
                observed.rename(columns={"reported_cases": "observed_reported_cases"}),
                on="observed_interval_id",
                how="inner",
                suffixes=("_predicted", "_observed"),
            )
            .sort_values(["period_start", "observed_interval_id"] if "period_start" in observed.columns else ["observed_interval_id"])
            .reset_index(drop=True)
        )
        if "series_year_predicted" in aligned.columns:
            aligned["series_year"] = aligned["series_year_predicted"]
        if "observed_year_predicted" in aligned.columns:
            aligned["observed_year"] = aligned["observed_year_predicted"]
        if aligned.empty:
            raise ValueError("No overlapping observed reporting intervals were available for calibration.")
        return aligned

    if "observed_year" in predicted.columns and "observed_year" in observed.columns:
        aligned = (
            predicted.rename(columns={"reported_cases": "predicted_reported_cases"})
            .merge(
                observed.rename(columns={"reported_cases": "observed_reported_cases"}),
                on="observed_year",
                how="inner",
                suffixes=("_predicted", "_observed"),
            )
            .sort_values("observed_year")
            .reset_index(drop=True)
        )
        if "series_year_predicted" in aligned.columns:
            aligned["series_year"] = aligned["series_year_predicted"]
        if aligned.empty:
            raise ValueError("No overlapping annual case series were available for calibration.")
        return aligned

    aligned = (
        predicted.rename(columns={"reported_cases": "predicted_reported_cases"})
        .merge(observed.rename(columns={"reported_cases": "observed_reported_cases"}), on="series_year", how="inner")
        .sort_values("series_year")
        .reset_index(drop=True)
    )
    if aligned.empty:
        raise ValueError("No overlapping annual case series were available for calibration.")
    return aligned


def calibration_runtime_config(base_config: dict[str, Any], observed: pd.DataFrame) -> dict[str, Any]:
    out = deepcopy(base_config)
    calibration_settings = load_configs()["baseline"].get("calibration", {})
    sim_overrides = calibration_settings.get("simulation_overrides", {})
    out.setdefault("calendar", {})["enabled"] = True
    if {"period_start", "period_end"}.issubset(observed.columns):
        period_start = pd.to_datetime(observed["period_start"], errors="coerce").min()
        period_end = pd.to_datetime(observed["period_end"], errors="coerce").max()
        if pd.isna(period_start) or pd.isna(period_end) or period_end <= period_start:
            raise ValueError("Observed interval calibration data must include valid period_start and period_end values.")
        out["calendar"]["analysis_start_date"] = pd.Timestamp(period_start).date().isoformat()
        end_time = float((pd.Timestamp(period_end) - pd.Timestamp(period_start)).days)
    else:
        first_year = int(observed["observed_year"].min())
        last_year = int(observed["observed_year"].max())
        n_years = max(1, last_year - first_year + 1)
        out["calendar"]["analysis_start_date"] = f"{first_year}-01-01"
        end_time = float(365 * n_years)
    out["simulation"]["start_time"] = 0
    out["simulation"]["end_time"] = end_time
    out["simulation"]["output_time_step"] = float(sim_overrides.get("output_time_step", 30))
    for key in ("burn_in_years", "rtol", "atol"):
        if key in sim_overrides:
            out["simulation"][key] = float(sim_overrides[key])
    if "solver_method" in sim_overrides:
        out["simulation"]["solver_method"] = str(sim_overrides["solver_method"])
    return out


def predicted_case_frame(timeseries: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    if {"observed_interval_id", "period_start", "period_end"}.issubset(observed.columns):
        return reported_cases_for_observed_intervals(timeseries, observed)
    return annual_reported_cases(timeseries)


def retain_recent_observed_window(observed: pd.DataFrame, recent_years: int) -> pd.DataFrame:
    if recent_years <= 0 or observed.empty:
        return observed
    if {"period_start", "period_end"}.issubset(observed.columns):
        out = observed.copy()
        out["period_start"] = pd.to_datetime(out["period_start"], errors="coerce")
        max_year = int(out["period_start"].dt.year.max())
        min_year = max_year - int(recent_years) + 1
        out = out.loc[out["period_start"].dt.year.ge(min_year)].copy()
    else:
        out = observed.tail(recent_years).copy()
    out = out.reset_index(drop=True)
    out["series_year"] = np.arange(len(out), dtype=int)
    return out


def annualized_observed_incidence(observed: pd.DataFrame, population: float) -> float:
    if observed.empty:
        return 0.0
    cases = float(pd.to_numeric(observed["reported_cases"], errors="coerce").fillna(0.0).sum())
    if "interval_days" in observed.columns:
        days = float(pd.to_numeric(observed["interval_days"], errors="coerce").fillna(0.0).sum())
        annual_cases = cases / max(days, 1.0) * 365.0
    else:
        annual_cases = cases / max(float(len(observed)), 1.0)
    return float(annual_cases / max(population, 1e-9) * 100_000.0)


def interval_fit_diagnostics(aligned: pd.DataFrame) -> dict[str, float]:
    observed = aligned["observed_reported_cases"].to_numpy(dtype=float)
    predicted = aligned["predicted_reported_cases"].to_numpy(dtype=float)
    denominator = np.maximum(observed + predicted, 1e-9)
    smape = float(np.mean(2.0 * np.abs(predicted - observed) / denominator)) if len(aligned) else np.inf
    if len(aligned) >= 3 and np.std(np.log1p(observed)) > 0.0 and np.std(np.log1p(predicted)) > 0.0:
        log_corr = float(np.corrcoef(np.log1p(observed), np.log1p(predicted))[0, 1])
    else:
        log_corr = np.nan
    peak_ratio = float(np.max(predicted) / max(float(np.max(observed)), 1e-9)) if len(aligned) else np.nan
    return {
        "interval_smape": smape,
        "log1p_correlation": log_corr,
        "peak_ratio": peak_ratio,
    }


def calibration_acceptance_checks(aligned: pd.DataFrame, calibration: dict[str, Any]) -> dict[str, Any]:
    settings = calibration.get("acceptance", {}) if isinstance(calibration.get("acceptance", {}), dict) else {}
    min_intervals = int(settings.get("min_overlap_intervals", 5))
    max_smape = float(settings.get("max_interval_smape", 1.0))
    min_corr = float(settings.get("min_log1p_correlation", 0.0))
    peak_min = float(settings.get("peak_ratio_min", 0.25))
    peak_max = float(settings.get("peak_ratio_max", 4.0))
    diagnostics = interval_fit_diagnostics(aligned)
    n_intervals = int(len(aligned))
    corr = float(diagnostics["log1p_correlation"])
    corr_ok = bool(np.isnan(corr) or corr >= min_corr)
    checks = {
        "calibration_interval_smape": float(diagnostics["interval_smape"]),
        "calibration_log1p_correlation": corr,
        "calibration_peak_ratio": float(diagnostics["peak_ratio"]),
        "calibration_min_overlap_intervals": min_intervals,
        "calibration_max_interval_smape": max_smape,
        "calibration_min_log1p_correlation": min_corr,
        "calibration_peak_ratio_min": peak_min,
        "calibration_peak_ratio_max": peak_max,
        "calibration_shape_fit_passed": bool(
            n_intervals >= min_intervals
            and diagnostics["interval_smape"] <= max_smape
            and corr_ok
            and peak_min <= diagnostics["peak_ratio"] <= peak_max
        ),
    }
    return checks


def calibration_objective(vector: np.ndarray, base_config: dict[str, Any], country: str, dispersion: float) -> float:
    try:
        config = apply_calibration_vector(base_config, vector)
        timeseries, _ = run_prepared_config(
            config,
            analysis="calibration",
            scenario="candidate",
            vaccine_scenario=base_config["baseline_vaccine_scenario"],
            resistance_scenario=base_config["baseline_resistance_scenario"],
            metadata={"country": country},
        )
        observed = observed_annual_case_frame(country)
        recent_years = int(load_configs()["baseline"].get("calibration", {}).get("recent_years", 0))
        observed = retain_recent_observed_window(observed, recent_years)
        predicted = predicted_case_frame(timeseries, observed)
        aligned = align_annual_case_series(predicted, observed)
        if len(aligned) < 3:
            return 1e18
        nll = negative_binomial_nll(
            aligned["observed_reported_cases"].to_numpy(dtype=float),
            aligned["predicted_reported_cases"].to_numpy(dtype=float),
            dispersion=dispersion,
        )
        return float(nll + reporting_rate_prior_penalty(config))
    except Exception:
        return 1e18


def optimize_calibration(
    base_config: dict[str, Any],
    country: str,
    *,
    dispersion: float,
    maxiter: int,
    n_starts: int,
    seed: int,
) -> tuple[Any, int]:
    starts = calibration_start_vectors(base_config, n_starts=n_starts, seed=seed)
    best_result: Any | None = None
    best_fun = np.inf

    for start in starts:
        result = minimize(
            lambda x: calibration_objective(x, base_config, country, dispersion),
            start,
            method="L-BFGS-B",
            bounds=CALIBRATION_BOUNDS,
            options={"maxiter": maxiter, "ftol": 1e-6},
        )
        objective = float(result.fun) if np.isfinite(result.fun) else np.inf
        if best_result is None or objective < best_fun:
            best_result = result
            best_fun = objective

    if best_result is None:
        raise RuntimeError(f"Calibration failed to produce a candidate for {country}.")
    return best_result, len(starts)


def staged_fast_calibration(
    base_config: dict[str, Any],
    country: str,
    observed: pd.DataFrame,
    *,
    dispersion: float,
    maxiter: int,
) -> tuple[dict[str, Any], Any]:
    best_config = deepcopy(base_config)
    best_score = np.inf
    best_message = "staged calibration did not evaluate"
    observed_mean = float(observed["reported_cases"].mean())
    calibration_settings = load_configs()["baseline"].get("calibration", {})
    mean_fit_penalty_weight = float(calibration_settings.get("mean_fit_penalty_weight", 50000.0))

    def evaluate(candidate: dict[str, Any], label: str) -> tuple[float, float]:
        nonlocal best_config, best_score, best_message
        timeseries, _ = run_prepared_config(
            candidate,
            analysis="calibration",
            scenario="staged_candidate",
            vaccine_scenario=base_config["baseline_vaccine_scenario"],
            resistance_scenario=base_config["baseline_resistance_scenario"],
            metadata={"country": country},
        )
        predicted = predicted_case_frame(timeseries, observed)
        aligned = align_annual_case_series(predicted, observed)
        predicted_mean = float(aligned["predicted_reported_cases"].mean())
        data_fit_score = negative_binomial_nll(
            aligned["observed_reported_cases"].to_numpy(dtype=float),
            aligned["predicted_reported_cases"].to_numpy(dtype=float),
            dispersion=dispersion,
        )
        mean_log_error = np.log(max(predicted_mean, 1e-9) / max(observed_mean, 1e-9))
        score = float(
            data_fit_score
            + mean_fit_penalty_weight * mean_log_error**2
            + reporting_rate_prior_penalty(candidate)
        )
        if score < best_score:
            best_score = score
            best_config = deepcopy(candidate)
            best_message = label
        return predicted_mean, score

    beta_low = float(CALIBRATION_BOUNDS[0][0])
    beta_high = float(CALIBRATION_BOUNDS[0][1])
    lower = deepcopy(base_config)
    upper = deepcopy(base_config)
    lower["transmission"]["beta_S"] = float(np.exp(beta_low))
    upper["transmission"]["beta_S"] = float(np.exp(beta_high))

    try:
        lower_mean, _ = evaluate(lower, "staged_fast lower bracket")
        upper_mean, _ = evaluate(upper, "staged_fast upper bracket")
        if lower_mean > observed_mean:
            best_config = lower
        elif upper_mean < observed_mean:
            best_config = upper
        else:
            low_log = beta_low
            high_log = beta_high
            for iteration in range(max(1, int(maxiter))):
                mid_log = 0.5 * (low_log + high_log)
                candidate = deepcopy(base_config)
                candidate["transmission"]["beta_S"] = float(np.exp(mid_log))
                predicted_mean, _ = evaluate(candidate, f"staged_fast beta bisection {iteration + 1}")
                if predicted_mean > observed_mean:
                    high_log = mid_log
                else:
                    low_log = mid_log
                if abs(np.log(max(predicted_mean, 1e-9) / max(observed_mean, 1e-9))) < 0.05:
                    break

        final_candidate = deepcopy(best_config)
        try:
            predicted_mean, _ = evaluate(final_candidate, "staged_fast reporting check")
            ratio = observed_mean / max(predicted_mean, 1e-9)
            lower_multiplier = float(np.exp(CALIBRATION_BOUNDS[1][0]))
            upper_multiplier = float(np.exp(CALIBRATION_BOUNDS[1][1]))
            final_candidate["reporting_multiplier"] = float(
                np.clip(
                    float(final_candidate.get("reporting_multiplier", 1.0)) * ratio,
                    lower_multiplier,
                    upper_multiplier,
                )
            )
            evaluate(final_candidate, "staged_fast final reporting adjustment")
        except Exception:
            pass
    except Exception as exc:
        best_message = f"staged_fast failed: {exc}"

    result = SimpleNamespace(
        x=initial_calibration_vector(best_config),
        fun=float(best_score),
        success=bool(np.isfinite(best_score)),
        message=best_message,
    )
    return best_config, result


def _artifact_metadata(
    *,
    country: str,
    base_config: dict[str, Any],
    result: Any,
    accepted: bool,
    fit_score: float,
    data_fit_score: float,
    n_starts: int,
    maxiter: int,
) -> dict[str, Any]:
    return {
        "country": country,
        "accepted": bool(accepted),
        "optimizer_success": bool(result.success),
        "calibration_status": "accepted" if accepted else "rejected",
        "fit_score": float(fit_score),
        "data_fit_score": float(data_fit_score),
        "n_starts": int(n_starts),
        "maxiter": int(maxiter),
        "config_hash": config_fingerprint(),
        "calibration_config_hash": calibration_config_fingerprint(),
        "baseline_vaccine_scenario": base_config["baseline_vaccine_scenario"],
        "baseline_resistance_scenario": base_config["baseline_resistance_scenario"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_calibration_artifacts(country: str, calibrated: dict[str, Any], summary: pd.DataFrame, result: Any) -> None:
    artifact_path = calibrated_country_artifact_path(country)
    accepted = bool(summary["calibration_accepted"].iloc[0])
    if accepted:
        payload = {
            "config": calibrated,
            "metadata": _artifact_metadata(
                country=country,
                base_config=calibrated,
                result=result,
                accepted=accepted,
                fit_score=float(summary["fit_score"].iloc[0]),
                data_fit_score=float(summary["data_fit_score"].iloc[0]),
                n_starts=int(summary["calibration_n_starts"].iloc[0]),
                maxiter=int(summary["calibration_maxiter"].iloc[0]),
            ),
        }
        write_yaml(payload, artifact_path)
    elif artifact_path.exists():
        artifact_path.unlink()


def calibrate_country(country: str, *, maxiter: int | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    configs = load_configs()
    calibration = configs["baseline"].get("calibration", {})
    dispersion = float(calibration.get("dispersion", 50.0))
    n_starts = int(calibration.get("n_starts", 6))
    seed = int(calibration.get("random_seed", 20260430))
    maxiter = int(maxiter or calibration.get("maxiter", 30))

    observed = observed_annual_case_frame(country)
    recent_years = int(calibration.get("recent_years", 0))
    observed = retain_recent_observed_window(observed, recent_years)
    recent_observed_incidence = annualized_observed_incidence(
        observed,
        float(configs["countries"][country]["total_population"]),
    )
    config = make_config(
        vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
        resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
        country_profile=country,
        load_calibration=False,
    )
    config = calibration_runtime_config(config, observed)
    method = str(calibration.get("method", "staged_fast"))
    if method == "staged_fast":
        calibrated, result = staged_fast_calibration(
            config,
            country,
            observed,
            dispersion=dispersion,
            maxiter=maxiter,
        )
        n_starts_used = 1
    else:
        result, n_starts_used = optimize_calibration(
            config,
            country,
            dispersion=dispersion,
            maxiter=maxiter,
            n_starts=n_starts,
            seed=seed,
        )
        calibrated = apply_calibration_vector(config, result.x)
    timeseries, summary = run_prepared_config(
        calibrated,
        analysis="calibration",
        scenario=f"{country}_calibrated",
        vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
        resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
        metadata={
            "country": country,
            "observed_mean_annual_reported_incidence_per_100k": recent_observed_incidence,
        },
    )

    predicted = predicted_case_frame(timeseries, observed)
    aligned = align_annual_case_series(predicted, observed)
    predicted_mean = float(aligned["predicted_reported_cases"].mean())
    predicted_sd = float(max(aligned["predicted_reported_cases"].std(ddof=0), np.sqrt(max(predicted_mean, 1.0))))
    data_fit_score = float(
        negative_binomial_nll(
            aligned["observed_reported_cases"].to_numpy(dtype=float),
            aligned["predicted_reported_cases"].to_numpy(dtype=float),
            dispersion=dispersion,
        )
    )

    reporting_by_age = ";".join(
        f"{record['label']}={min(1.0, float(record.get('reporting_rate', 0.0)) * float(calibrated.get('reporting_multiplier', 1.0))):.4f}"
        for record in calibrated["age_groups"]
    )
    reporting_prior = calibrated.get("reporting_rate_prior", {})
    reporting_prior_groups = reporting_prior.get("age_groups", {})
    reporting_prior_by_age = ";".join(
        f"{record['label']}={float(record.get('reporting_rate', 0.0)):.4f}["
        f"{float(reporting_prior_groups.get(record['label'], {}).get('lower', np.nan)):.4f},"
        f"{float(reporting_prior_groups.get(record['label'], {}).get('upper', np.nan)):.4f}]"
        for record in calibrated["age_groups"]
    )

    fit_checks = calibration_acceptance_checks(aligned, calibration)
    calibration_accepted = bool(
        result.success
        and summary["absolute_fit_status"].iloc[0] == "calibrated_to_reported_cases"
        and fit_checks["calibration_shape_fit_passed"]
    )
    summary["data_fit_score"] = data_fit_score
    summary["reporting_rate_prior_penalty"] = float(reporting_rate_prior_penalty(calibrated))
    summary["fit_score"] = float(result.fun)
    summary["calibrated_beta"] = float(calibrated["transmission"]["beta_S"])
    summary["reporting_multiplier_by_age"] = reporting_by_age
    summary["reporting_rate_prior_by_age"] = reporting_prior_by_age
    summary["reporting_rate_prior_method"] = str(reporting_prior.get("method", ""))
    summary["reporting_rate_prior_evidence_class"] = str(reporting_prior.get("evidence_class", ""))
    summary["posterior_interval_low"] = max(0.0, predicted_mean - 1.96 * predicted_sd)
    summary["posterior_interval_high"] = predicted_mean + 1.96 * predicted_sd
    for key, value in fit_checks.items():
        summary[key] = value
    summary["calibration_accepted"] = calibration_accepted
    summary["calibration_success"] = calibration_accepted
    summary["optimizer_success"] = bool(result.success)
    summary["calibration_status"] = "accepted" if calibration_accepted else "rejected"
    summary["calibration_message"] = str(result.message)
    summary["calibration_n_starts"] = int(n_starts_used)
    summary["calibration_maxiter"] = int(maxiter)
    summary["calibration_data_overlap_years"] = int(aligned["observed_year"].nunique()) if "observed_year" in aligned else int(len(aligned))
    summary["calibration_data_overlap_intervals"] = int(len(aligned))
    summary["calibration_objective"] = float(result.fun)

    save_calibration_artifacts(country, calibrated, summary, result)
    return calibrated, summary


def main() -> None:
    configs = load_configs()
    default_country = configs["baseline"].get("calibration", {}).get(
        "default_country_profile",
        configs["baseline"].get("baseline_country_profile", "Australia"),
    )
    parser = argparse.ArgumentParser(description="Calibrate country-level pertussis model parameters.")
    parser.add_argument("--country", default=default_country)
    parser.add_argument("--maxiter", type=int, default=None)
    args = parser.parse_args()

    _, summary = calibrate_country(args.country, maxiter=args.maxiter)
    write_dataframe(summary, project_path(f"outputs/tables/calibration_{args.country}.csv"))


if __name__ == "__main__":
    main()
