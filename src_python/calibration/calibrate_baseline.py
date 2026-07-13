from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from src_python.calibration.likelihood import negative_binomial_nll
from src_python.model.observables import project_reported_cases
from src_python.simulation.common import (
    calibration_config_fingerprint,
    calibration_source_code_fingerprint,
    calibrated_country_artifact_path,
    config_fingerprint,
    load_configs,
    make_config,
    run_prepared_case_exposure,
    run_prepared_config,
    source_code_fingerprint,
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


def aggregate_observed_case_intervals(
    observed: pd.DataFrame,
    interval: str,
) -> pd.DataFrame:
    """Aggregate whole surveillance windows without fabricating fractional counts.

    A source reporting window is assigned by its midpoint to a calendar month
    or year and is never split.  Within each target period, only exactly
    contiguous source windows are collapsed; gaps start a new likelihood row.
    This preserves integer count support for the negative-binomial observation
    model and avoids integrating model incidence over unobserved gaps.
    """

    interval = str(interval or "native").lower()
    if "observed_interval_id" in observed.columns and observed[
        "observed_interval_id"
    ].astype(str).duplicated().any():
        raise ValueError("Observed surveillance interval IDs must be unique before aggregation")
    if interval in {"native", "none", "reporting_interval"}:
        return observed.copy()
    if interval in {"monthly", "month"}:
        period_frequency = "M"
        reporting_frequency = "monthly_aggregated"
    elif interval in {"annual", "year", "yearly"}:
        period_frequency = "Y"
        reporting_frequency = "annual_aggregated"
    else:
        raise ValueError(f"Unsupported likelihood observation interval: {interval}")
    if not {"period_start", "period_end"}.issubset(observed.columns) or observed.empty:
        return observed.copy()

    frame = observed.copy()
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce")
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["reported_cases"] = pd.to_numeric(frame["reported_cases"], errors="coerce")
    if frame[["period_start", "period_end", "reported_cases"]].isna().any().any():
        raise ValueError("Observed surveillance intervals contain invalid dates or cases")
    frame = frame.sort_values(["period_start", "period_end"]).reset_index(drop=True)
    if np.any(frame["period_end"] <= frame["period_start"]):
        raise ValueError("Observed surveillance interval end must follow start")
    if len(frame) > 1 and np.any(
        frame["period_start"].iloc[1:].to_numpy()
        < frame["period_end"].iloc[:-1].to_numpy()
    ):
        raise ValueError("Observed surveillance intervals must not overlap")

    midpoint = frame["period_start"] + (frame["period_end"] - frame["period_start"]) / 2
    frame["_target_period"] = midpoint.dt.to_period(period_frequency)
    rows: list[dict[str, Any]] = []
    for period, period_frame in frame.groupby("_target_period", sort=True):
        period_frame = period_frame.sort_values(["period_start", "period_end"])
        block_id = (
            period_frame["period_start"]
            .ne(period_frame["period_end"].shift(1))
            .cumsum()
        )
        for block_index, block in period_frame.groupby(block_id, sort=True):
            period_start = pd.Timestamp(block["period_start"].iloc[0])
            period_end = pd.Timestamp(block["period_end"].iloc[-1])
            reported = float(block["reported_cases"].sum())
            start_date = period_start.date().isoformat()
            end_date = period_end.date().isoformat()
            rows.append(
                {
                    "series_year": len(rows),
                    "observed_year": int(period.start_time.year),
                    "observed_interval_id": (
                        f"{start_date}_{end_date}_{period}_{int(block_index)}"
                    ),
                    "period_start": period_start,
                    "period_end": period_end,
                    "interval_days": float(
                        (block["period_end"] - block["period_start"])
                        .dt.total_seconds()
                        .sum()
                        / 86400.0
                    ),
                    "reported_cases": reported,
                    "likelihood_group_id": str(period),
                    "source_interval_count": int(len(block)),
                    "aggregation_assignment": "whole_interval_midpoint",
                    "reporting_frequency": reporting_frequency,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["likelihood_group_interval_count"] = (
            out.groupby("likelihood_group_id")["likelihood_group_id"].transform("size").astype(int)
        )
    return out


def grouped_likelihood_observations(observed: pd.DataFrame) -> pd.DataFrame:
    """Collapse disjoint exposure blocks only after exact interval projection."""

    if "likelihood_group_id" not in observed.columns:
        return observed.reset_index(drop=True).copy()
    frame = observed.copy()
    frame["reported_cases"] = pd.to_numeric(frame["reported_cases"], errors="raise")
    rows: list[dict[str, Any]] = []
    for series_index, (group_id, group) in enumerate(
        frame.groupby("likelihood_group_id", sort=False)
    ):
        rows.append(
            {
                "series_year": series_index,
                "observed_year": int(group["observed_year"].iloc[0]),
                "observed_interval_id": str(group_id),
                "likelihood_group_id": str(group_id),
                "period_start": pd.to_datetime(group["period_start"]).min(),
                "period_end": pd.to_datetime(group["period_end"]).max(),
                "interval_days": float(
                    pd.to_numeric(group["interval_days"], errors="raise").sum()
                ),
                "reported_cases": float(group["reported_cases"].sum()),
                "source_exposure_blocks": int(len(group)),
                "reporting_frequency": str(
                    group.get("reporting_frequency", pd.Series(["grouped"])).iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_projected_likelihood_means(
    observed: pd.DataFrame,
    interval_means: np.ndarray,
) -> np.ndarray:
    """Sum exact block-level means using the observation likelihood groups."""

    values = np.asarray(interval_means, dtype=float)
    if len(values) != len(observed):
        raise ValueError("Projected interval means do not match observation blocks")
    if "likelihood_group_id" not in observed.columns:
        return values
    frame = pd.DataFrame(
        {
            "likelihood_group_id": observed["likelihood_group_id"].astype(str).to_numpy(),
            "predicted_mean": values,
        }
    )
    return (
        frame.groupby("likelihood_group_id", sort=False)["predicted_mean"]
        .sum()
        .to_numpy(dtype=float)
    )


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
        out["calendar"]["analysis_start_date"] = f"{first_year}-01-01"
        end_time = float(
            (date(last_year + 1, 1, 1) - date(first_year, 1, 1)).days
        )
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


def predicted_case_frame_from_interval_means(
    predicted_means: np.ndarray,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Build an alignment frame from exact observed-interval counter means."""

    values = np.asarray(predicted_means, dtype=float)
    if values.shape != (len(observed),):
        raise ValueError("Predicted interval means do not match observed rows")
    key_columns = [
        column
        for column in (
            "series_year",
            "observed_year",
            "observed_interval_id",
        )
        if column in observed.columns
    ]
    predicted = observed.loc[:, key_columns].copy()
    predicted["reported_cases"] = values
    return predicted


def retain_recent_observed_window(observed: pd.DataFrame, recent_years: int) -> pd.DataFrame:
    if recent_years <= 0 or observed.empty:
        return observed
    if {"period_start", "period_end"}.issubset(observed.columns):
        out = observed.copy()
        out["period_start"] = pd.to_datetime(out["period_start"], errors="coerce")
        out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
        if out[["period_start", "period_end"]].isna().any().any():
            raise ValueError("Observed intervals contain invalid dates")
        midpoint_year = (
            out["period_start"] + (out["period_end"] - out["period_start"]) / 2
        ).dt.year
        max_year = int(midpoint_year.max())
        min_year = max_year - int(recent_years) + 1
        # Match the midpoint assignment used by annual likelihood aggregation.
        # This retains a cross-year epiweek when its midpoint belongs to the
        # retained target year instead of dropping it at the lookback boundary.
        out = out.loc[midpoint_year.ge(min_year)].copy()
    else:
        out = observed.tail(recent_years).copy()
    out = out.reset_index(drop=True)
    out["series_year"] = np.arange(len(out), dtype=int)
    return out


def calibration_observed_case_frame(country: str) -> pd.DataFrame:
    """Return surveillance counts on the prespecified calibration time scale."""

    calibration = load_configs()["baseline"].get("calibration", {})
    observed = retain_recent_observed_window(
        observed_annual_case_frame(country),
        int(calibration.get("recent_years", 0)),
    )
    return aggregate_observed_case_intervals(
        observed,
        str(calibration.get("likelihood_observation_frequency", "annual")),
    )


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


def _log_beta_period_key(config: dict[str, Any]) -> tuple[tuple[str, str, float], ...]:
    variation = config.get("transmission", {}).get("log_beta_time_variation", {})
    return tuple(
        (
            str(period.get("start_date", "")),
            str(period.get("end_date", "")),
            float(period.get("log_multiplier", 0.0)),
        )
        for period in variation.get("periods", [])
        if isinstance(period, dict)
    )


def _calibration_biological_key(config: dict[str, Any]) -> tuple[Any, ...]:
    """Calibration coordinates that require a new biological trajectory."""

    return (
        float(config["transmission"]["beta_S"]),
        float(config["transmission"].get("seasonal_amplitude", 0.0)),
        float(config["transmission"].get("seasonal_phase", 0.0)),
        float(config["transmission"].get("multi_year_amplitude", 0.0)),
        float(config["transmission"].get("multi_year_period_years", 4.0)),
        float(config["transmission"].get("multi_year_phase", 0.0)),
        float(config.get("importation", {}).get("rate_per_100k_per_year", 0.0)),
        float(config.get("importation", {}).get("resistant_fraction", 0.0)),
        float(config.get("resistance", {}).get("importation_fraction", 0.0)),
        _log_beta_period_key(config),
    )


def _calibration_predicted_means(
    config: dict[str, Any],
    observed: pd.DataFrame,
    country: str,
    *,
    exposure_cache: dict[tuple[Any, ...], tuple[Any, np.ndarray]] | None = None,
) -> np.ndarray:
    biological_key = _calibration_biological_key(config)
    cached = exposure_cache.get(biological_key) if exposure_cache is not None else None
    if cached is None:
        exposure, base_reporting_rates = run_prepared_case_exposure(
            config,
            observed,
            analysis="calibration",
            scenario="candidate",
            vaccine_scenario=config["baseline_vaccine_scenario"],
            resistance_scenario=config["baseline_resistance_scenario"],
            metadata={"country": country},
        )
        if exposure_cache is not None:
            exposure_cache.pop(biological_key, None)
            exposure_cache[biological_key] = (exposure, base_reporting_rates)
            while len(exposure_cache) > 32:
                exposure_cache.pop(next(iter(exposure_cache)))
    else:
        exposure, base_reporting_rates = cached
        if exposure_cache is not None:
            exposure_cache.pop(biological_key, None)
            exposure_cache[biological_key] = (exposure, base_reporting_rates)
    interval_means = project_reported_cases(
        exposure,
        base_reporting_rates,
        float(config.get("reporting_multiplier", 1.0)),
    ).interval_total_mean
    return aggregate_projected_likelihood_means(observed, interval_means)


def calibration_objective(
    vector: np.ndarray,
    base_config: dict[str, Any],
    country: str,
    dispersion: float,
    *,
    observed: pd.DataFrame | None = None,
    exposure_cache: dict[tuple[Any, ...], tuple[Any, np.ndarray]] | None = None,
) -> float:
    try:
        config = apply_calibration_vector(base_config, vector)
        if observed is None:
            observed = calibration_observed_case_frame(country)
        predicted = _calibration_predicted_means(
            config,
            observed,
            country,
            exposure_cache=exposure_cache,
        )
        likelihood_observed = grouped_likelihood_observations(observed)
        observed_values = pd.to_numeric(
            likelihood_observed["reported_cases"], errors="coerce"
        ).to_numpy(dtype=float)
        if len(observed_values) < 3 or len(predicted) != len(observed_values):
            return 1e18
        nll = negative_binomial_nll(
            observed_values,
            predicted,
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
    observed = calibration_observed_case_frame(country)
    best_result: Any | None = None
    best_fun = np.inf

    for start in starts:
        exposure_cache: dict[tuple[Any, ...], tuple[Any, np.ndarray]] = {}
        result = minimize(
            lambda x: calibration_objective(
                x,
                base_config,
                country,
                dispersion,
                observed=observed,
                exposure_cache=exposure_cache,
            ),
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
    likelihood_observed = grouped_likelihood_observations(observed)
    observed_mean = float(likelihood_observed["reported_cases"].mean())
    calibration_settings = load_configs()["baseline"].get("calibration", {})
    mean_fit_penalty_weight = float(calibration_settings.get("mean_fit_penalty_weight", 50000.0))
    exposure_cache: dict[tuple[Any, ...], tuple[Any, np.ndarray]] = {}

    def evaluate(candidate: dict[str, Any], label: str) -> tuple[float, float]:
        nonlocal best_config, best_score, best_message
        predicted_values = _calibration_predicted_means(
            candidate,
            observed,
            country,
            exposure_cache=exposure_cache,
        )
        observed_values = pd.to_numeric(
            likelihood_observed["reported_cases"], errors="coerce"
        ).to_numpy(dtype=float)
        if len(predicted_values) != len(observed_values):
            raise ValueError("Predicted and observed calibration intervals do not align")
        predicted_mean = float(np.mean(predicted_values))
        data_fit_score = negative_binomial_nll(
            observed_values,
            predicted_values,
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

    try:
        beta_low = float(CALIBRATION_BOUNDS[0][0])
        beta_high = float(CALIBRATION_BOUNDS[0][1])
        beta_points = int(max(5, calibration_settings.get("beta_grid_points", 11)))
        phase_points = int(max(1, calibration_settings.get("initial_phase_grid_points", 9)))
        period_days = 365.0 * float(
            base_config["transmission"].get("multi_year_period_years", 4.0)
        )
        base_phase = float(base_config["transmission"].get("multi_year_phase", 0.0))
        phases = (
            [base_phase]
            if (
                phase_points == 1
                or period_days <= 0.0
                or float(base_config["transmission"].get("multi_year_amplitude", 0.0)) == 0.0
            )
            else [
                float((base_phase + offset) % period_days)
                for offset in np.linspace(0.0, period_days, phase_points, endpoint=False)
            ]
        )
        beta_grid = np.linspace(beta_low, beta_high, beta_points)
        for phase_index, phase in enumerate(phases):
            for beta_index, log_beta in enumerate(beta_grid):
                candidate = deepcopy(base_config)
                candidate["transmission"]["multi_year_phase"] = phase
                candidate["transmission"]["beta_S"] = float(np.exp(log_beta))
                evaluate(
                    candidate,
                    f"staged phase {phase_index + 1}/{len(phases)} "
                    f"beta {beta_index + 1}/{len(beta_grid)}",
                )

        # Locally refine beta at the best phase without assuming that endemic
        # cycle amplitude is monotone over the entire beta support.
        refinements = int(max(0, calibration_settings.get("beta_grid_refinements", 2)))
        spacing = float(beta_grid[1] - beta_grid[0])
        for refinement in range(refinements):
            centre = float(np.log(best_config["transmission"]["beta_S"]))
            local_grid = np.linspace(
                max(beta_low, centre - spacing),
                min(beta_high, centre + spacing),
                5,
            )
            for beta_index, log_beta in enumerate(local_grid):
                candidate = deepcopy(best_config)
                candidate["transmission"]["beta_S"] = float(np.exp(log_beta))
                evaluate(
                    candidate,
                    f"staged beta refinement {refinement + 1}.{beta_index + 1}",
                )
            spacing *= 0.5

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


def _negative_binomial_deviance_residuals(
    observed: np.ndarray,
    mean: np.ndarray,
    dispersion: float,
) -> np.ndarray:
    """Signed square-root NB2 deviance residuals."""

    y = np.asarray(observed, dtype=float)
    mu = np.maximum(np.asarray(mean, dtype=float), 1e-12)
    k = max(float(dispersion), 1e-12)
    first = np.zeros_like(y)
    positive = y > 0.0
    first[positive] = y[positive] * np.log(y[positive] / mu[positive])
    second = (y + k) * np.log((y + k) / (mu + k))
    deviance = np.maximum(2.0 * (first - second), 0.0)
    return np.sign(y - mu) * np.sqrt(deviance)


def _annual_ar1_transition_scales(
    midpoints: pd.Series,
    *,
    rho: float,
    innovation_sd: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Convert a one-year AR(1) specification to irregular calendar steps."""

    if not 0.0 <= float(rho) < 0.99 or float(innovation_sd) <= 0.0:
        raise ValueError("Invalid annual AR(1) specification")
    values = pd.to_datetime(midpoints, errors="raise").reset_index(drop=True)
    stationary_sd = float(innovation_sd) / np.sqrt(1.0 - float(rho) ** 2)
    if len(values) < 2:
        return stationary_sd, np.empty(0, dtype=float), np.empty(0, dtype=float)
    delta_years = (
        values.iloc[1:].reset_index(drop=True)
        - values.iloc[:-1].reset_index(drop=True)
    ).dt.total_seconds().to_numpy(dtype=float) / (365.2425 * 86400.0)
    if np.any(delta_years <= 0.0):
        raise ValueError("AR(1) interval midpoints must be strictly increasing")
    transition_rho = (
        np.power(float(rho), delta_years)
        if float(rho) > 0.0
        else np.zeros_like(delta_years)
    )
    transition_sd = stationary_sd * np.sqrt(
        np.maximum(1.0 - transition_rho**2, 1e-12)
    )
    return stationary_sd, transition_rho, transition_sd


def _gauss_newton_covariance(
    jacobian: np.ndarray,
    *,
    relative_singular_value_floor: float = 1e-8,
) -> tuple[np.ndarray, int, float]:
    """Return the unscaled Gauss-Newton covariance and rank diagnostics.

    Residuals in the state-space fit are already normalized likelihood- or
    prior-scale residuals, so no ad-hoc residual-variance multiplier is used.
    A rank-deficient approximation is reported to callers rather than hidden
    by a large ridge term.
    """

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Gauss-Newton Jacobian must be a finite two-dimensional matrix")
    _u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    if len(singular_values) != matrix.shape[1]:
        raise ValueError("Gauss-Newton Jacobian has fewer rows than fitted coordinates")
    largest = float(singular_values[0])
    threshold = max(largest * float(relative_singular_value_floor), np.finfo(float).eps)
    retained = singular_values > threshold
    rank = int(np.count_nonzero(retained))
    inverse_squared = np.zeros_like(singular_values)
    inverse_squared[retained] = 1.0 / singular_values[retained] ** 2
    covariance = (vt.T * inverse_squared) @ vt
    covariance = 0.5 * (covariance + covariance.T)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    return covariance, rank, condition_number


def extend_annual_log_beta_conditional_mean(
    config: dict[str, Any],
    *,
    through_year: int,
    interpretation: str = "latent_AR1_process_MAP_with_conditional_mean_forecast",
) -> dict[str, Any]:
    """Extend a fitted annual latent path by its AR(1) conditional mean."""

    out = deepcopy(config)
    variation = out.get("transmission", {}).get("log_beta_time_variation")
    if not isinstance(variation, dict) or not variation.get("periods"):
        raise ValueError("Annual log-beta extension requires a fitted process path")
    rho = float(variation.get("ar1_rho", np.nan))
    if not 0.0 <= rho < 1.0:
        raise ValueError("Annual log-beta extension requires 0 <= ar1_rho < 1")
    periods = sorted(variation["periods"], key=lambda row: str(row["start_date"]))
    years: list[int] = []
    for period in periods:
        start = pd.Timestamp(period["start_date"])
        end = pd.Timestamp(period["end_date"])
        if (
            start.month != 1
            or start.day != 1
            or end.month != 12
            or end.day != 31
            or start.year != end.year
        ):
            raise ValueError("Annual log-beta extension found a non-annual period")
        years.append(int(start.year))
    if any(current != previous + 1 for previous, current in zip(years, years[1:])):
        raise ValueError("Annual log-beta periods must be consecutive")
    last_year = years[-1]
    last_value = float(periods[-1]["log_multiplier"])
    if int(through_year) < last_year:
        raise ValueError("through_year precedes the fitted process path")
    historical_end_year = int(
        variation.get("historical_state_end_year", last_year)
    )
    for year in range(last_year + 1, int(through_year) + 1):
        last_value = rho * last_value
        periods.append(
            {
                "start_date": f"{year:04d}-01-01",
                "end_date": f"{year:04d}-12-31",
                "log_multiplier": float(last_value),
                "state_origin": "AR1_conditional_mean_forecast",
            }
        )
    variation["periods"] = periods
    variation["historical_state_end_year"] = historical_end_year
    variation["forecast_process_contract"] = "AR1_conditional_mean"
    variation["interpretation"] = interpretation
    return out


def state_space_map_calibration(
    base_config: dict[str, Any],
    country: str,
    observed: pd.DataFrame,
    *,
    dispersion: float,
    maxiter: int,
    process_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Regularized annual transmission-process MAP approximation.

    This is a deterministic Laplace/MAP bridge toward a full POMP analysis:
    calendar-interval log transmission deviations form a latent AR(1) process,
    while reported counts retain the NB2 measurement model.  The fitted
    process path conditions the hidden state passed into forecasting; it is not
    mislabeled as a mechanistic country parameter.
    """

    if not {"period_start", "period_end"}.issubset(observed.columns):
        raise ValueError("state_space_map calibration requires dated observation intervals")
    calibration = load_configs()["baseline"].get("calibration", {})
    process = {
        **calibration.get("process_model", {}),
        **(process_overrides or {}),
    }
    rho = float(process.get("ar1_rho", 0.5))
    innovation_sd = float(process.get("log_beta_innovation_sd", 0.6))
    if not 0.0 <= rho < 0.99 or innovation_sd <= 0.0:
        raise ValueError("Invalid calibration process-model AR(1) parameters")
    initial_config, initial_result = staged_fast_calibration(
        base_config,
        country,
        observed,
        dispersion=dispersion,
        maxiter=maxiter,
    )
    n_observations = len(observed)
    if n_observations < 1:
        raise ValueError("state_space_map calibration requires at least one interval")
    likelihood_observed = grouped_likelihood_observations(observed)
    observed_values = pd.to_numeric(
        likelihood_observed["reported_cases"], errors="raise"
    ).to_numpy(dtype=float)
    starts = pd.to_datetime(observed["period_start"], errors="raise")
    ends = pd.to_datetime(observed["period_end"], errors="raise")
    if bool((ends <= starts).any()):
        raise ValueError("state_space_map observation intervals must have positive duration")
    if n_observations > 1 and bool(
        (starts.iloc[1:].reset_index(drop=True) < ends.iloc[:-1].reset_index(drop=True)).any()
    ):
        raise ValueError("state_space_map observation intervals must be ordered and non-overlapping")
    first_state_year = int(starts.dt.year.min())
    last_state_year = int((ends - pd.Timedelta(nanoseconds=1)).dt.year.max())
    state_years = np.arange(first_state_year, last_state_year + 1, dtype=int)
    n_states = len(state_years)
    state_starts = pd.Series(
        pd.to_datetime([f"{year:04d}-01-01" for year in state_years])
    )
    state_ends = pd.Series(
        pd.to_datetime([f"{year + 1:04d}-01-01" for year in state_years])
    )
    exposure_cache: dict[tuple[Any, ...], tuple[Any, np.ndarray]] = {}

    def configured(vector: np.ndarray) -> dict[str, Any]:
        # ``initial_config`` was fitted to these same observations and is used
        # only as an optimizer warm start.  Building the statistical target
        # from ``base_config`` prevents that preliminary fit from silently
        # becoming a data-dependent prior.
        candidate = deepcopy(base_config)
        candidate["transmission"]["beta_S"] = float(np.exp(vector[0]))
        candidate["reporting_multiplier"] = float(np.exp(vector[1]))
        candidate["transmission"]["log_beta_time_variation"] = {
            "enabled": True,
            "interpretation": "latent_AR1_process_MAP_point_estimate",
            "ar1_rho": rho,
            "innovation_sd": innovation_sd,
            "periods": [
                {
                    "start_date": pd.Timestamp(start).date().isoformat(),
                    # Observation intervals are half-open; the transmission
                    # schedule contract uses an inclusive calendar end.
                    "end_date": (
                        pd.Timestamp(end) - pd.Timedelta(days=1)
                    ).date().isoformat(),
                    "log_multiplier": float(log_multiplier),
                }
                for start, end, log_multiplier in zip(
                    state_starts,
                    state_ends,
                    vector[2:],
                )
            ],
        }
        return candidate

    # These centres come from the pre-surveillance country configuration.
    # The staged fit below supplies only the numerical starting point.
    beta_centre = float(np.log(base_config["transmission"]["beta_S"]))
    reporting_centre = float(np.log(base_config.get("reporting_multiplier", 1.0)))
    beta_prior_sd = float(process.get("log_beta_prior_sd", 0.5))
    reporting_prior_sd = float(process.get("log_reporting_prior_sd", 0.5))
    if beta_prior_sd <= 0.0 or reporting_prior_sd <= 0.0:
        raise ValueError("state_space_map prior scales must be positive")

    # Latent transmission states cover complete calendar years, including the
    # unobserved remainder of a partially reported final year.  Treat ``rho``
    # as a one-year correlation and still derive transition scales from the
    # calendar state midpoints so missing years are handled correctly.
    midpoints = state_starts + (state_ends - state_starts) / 2
    stationary_sd, transition_rho, transition_sd = _annual_ar1_transition_scales(
        midpoints,
        rho=rho,
        innovation_sd=innovation_sd,
    )

    def residuals(vector: np.ndarray) -> np.ndarray:
        candidate = configured(vector)
        prediction = _calibration_predicted_means(
            candidate,
            observed,
            country,
            exposure_cache=exposure_cache,
        )
        data_residual = _negative_binomial_deviance_residuals(
            observed_values,
            prediction,
            dispersion,
        )
        latent = np.asarray(vector[2:], dtype=float)
        process_residual = np.empty(n_states, dtype=float)
        process_residual[0] = latent[0] / stationary_sd
        if n_states > 1:
            process_residual[1:] = (
                latent[1:] - transition_rho * latent[:-1]
            ) / transition_sd
        prior_penalty = reporting_rate_prior_penalty(candidate)
        return np.concatenate(
            (
                data_residual,
                process_residual,
                np.asarray(
                    [
                        (vector[0] - beta_centre) / beta_prior_sd,
                        (vector[1] - reporting_centre) / reporting_prior_sd,
                        np.sqrt(max(2.0 * prior_penalty, 0.0)),
                    ]
                ),
            )
        )

    optimizer_beta_start = float(np.log(initial_config["transmission"]["beta_S"]))
    optimizer_reporting_start = float(
        np.log(initial_config.get("reporting_multiplier", 1.0))
    )
    start_vector = np.concatenate(
        (
            np.asarray([optimizer_beta_start, optimizer_reporting_start]),
            np.zeros(n_states, dtype=float),
        )
    )
    process_bound = float(process.get("max_abs_log_beta_deviation", 2.5))
    beta_value_bounds = np.asarray(
        process.get("beta_S_bounds", [np.exp(CALIBRATION_BOUNDS[0][0]), np.exp(CALIBRATION_BOUNDS[0][1])]),
        dtype=float,
    )
    reporting_value_bounds = np.asarray(
        process.get(
            "reporting_multiplier_bounds",
            [np.exp(CALIBRATION_BOUNDS[1][0]), np.exp(CALIBRATION_BOUNDS[1][1])],
        ),
        dtype=float,
    )
    if (
        beta_value_bounds.shape != (2,)
        or reporting_value_bounds.shape != (2,)
        or np.any(beta_value_bounds <= 0.0)
        or np.any(reporting_value_bounds <= 0.0)
        or beta_value_bounds[0] >= beta_value_bounds[1]
        or reporting_value_bounds[0] >= reporting_value_bounds[1]
    ):
        raise ValueError("Invalid state_space_map beta/reporting bounds")
    lower = np.concatenate(
        (
            np.log([beta_value_bounds[0], reporting_value_bounds[0]]),
            np.full(n_states, -process_bound),
        )
    )
    upper = np.concatenate(
        (
            np.log([beta_value_bounds[1], reporting_value_bounds[1]]),
            np.full(n_states, process_bound),
        )
    )
    start_vector = np.clip(start_vector, lower + 1e-10, upper - 1e-10)
    fit = least_squares(
        residuals,
        start_vector,
        bounds=(lower, upper),
        max_nfev=int(process.get("max_nfev", max(40, maxiter * 8))),
        xtol=float(process.get("xtol", 1e-4)),
        ftol=float(process.get("ftol", 1e-4)),
        gtol=float(process.get("gtol", 1e-4)),
    )
    calibrated = configured(fit.x)
    posterior_covariance, posterior_rank, posterior_condition = _gauss_newton_covariance(
        fit.jac
    )
    prediction = _calibration_predicted_means(
        calibrated,
        observed,
        country,
        exposure_cache=exposure_cache,
    )
    latent = np.asarray(fit.x[2:], dtype=float)
    process_penalty = 0.5 * (
        (latent[0] / stationary_sd) ** 2
        + np.sum(
            ((latent[1:] - transition_rho * latent[:-1]) / transition_sd) ** 2
        )
        + ((fit.x[0] - beta_centre) / beta_prior_sd) ** 2
        + ((fit.x[1] - reporting_centre) / reporting_prior_sd) ** 2
    )
    score = float(
        negative_binomial_nll(observed_values, prediction, dispersion)
        + process_penalty
        + reporting_rate_prior_penalty(calibrated)
    )
    result = SimpleNamespace(
        x=initial_calibration_vector(calibrated),
        fun=score,
        success=bool(fit.success and np.isfinite(score)),
        message=(
            "state_space_map: " + str(fit.message)
            + f"; staged_start={initial_result.message}"
        ),
        process_vector=latent,
        process_cost=float(fit.cost),
        process_optimality=float(fit.optimality),
        process_nfev=int(fit.nfev),
        process_transition_rho=transition_rho,
        process_transition_sd=transition_sd,
        prior_center_source="pre_surveillance_country_config",
        posterior_approximation="gauss_newton_laplace_conditional",
        posterior_coordinate_names=(
            ["log_beta_S", "log_reporting_multiplier"]
            + [f"log_beta_process_{year}" for year in state_years]
        ),
        posterior_map_vector=np.asarray(fit.x, dtype=float),
        posterior_covariance=posterior_covariance,
        posterior_rank=int(posterior_rank),
        posterior_condition_number=float(posterior_condition),
        posterior_lower_bounds=lower,
        posterior_upper_bounds=upper,
        measurement_dispersion=float(dispersion),
        likelihood_observation_frequency=str(
            calibration.get("likelihood_observation_frequency", "annual")
        ),
        process_ar1_rho=float(rho),
        process_innovation_sd=float(innovation_sd),
        beta_prior_log_sd=float(beta_prior_sd),
        reporting_prior_log_sd=float(reporting_prior_sd),
    )
    return calibrated, result


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
    metadata = {
        "country": country,
        "accepted": bool(accepted),
        "optimizer_success": bool(result.success),
        "calibration_status": (
            "state_reconstruction_accepted" if accepted else "state_reconstruction_rejected"
        ),
        "fit_score": float(fit_score),
        "data_fit_score": float(data_fit_score),
        "n_starts": int(n_starts),
        "maxiter": int(maxiter),
        "config_hash": config_fingerprint(),
        "calibration_config_hash": calibration_config_fingerprint(),
        "source_code_hash": source_code_fingerprint(),
        "calibration_source_code_hash": calibration_source_code_fingerprint(),
        "baseline_vaccine_scenario": base_config["baseline_vaccine_scenario"],
        "baseline_resistance_scenario": base_config["baseline_resistance_scenario"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if hasattr(result, "posterior_map_vector"):
        metadata["state_posterior_approximation"] = {
            "method": str(result.posterior_approximation),
            "scope": (
                "conditional_on_fixed_rho_innovation_sd_dispersion_and_reference_structure"
            ),
            "coordinate_names": list(result.posterior_coordinate_names),
            "map_vector": np.asarray(result.posterior_map_vector, dtype=float).tolist(),
            "covariance": np.asarray(result.posterior_covariance, dtype=float).tolist(),
            "lower_bounds": np.asarray(result.posterior_lower_bounds, dtype=float).tolist(),
            "upper_bounds": np.asarray(result.posterior_upper_bounds, dtype=float).tolist(),
            "rank": int(result.posterior_rank),
            "condition_number": float(result.posterior_condition_number),
            "prior_center_source": str(result.prior_center_source),
            "measurement_dispersion": float(result.measurement_dispersion),
            "likelihood_observation_frequency": str(
                result.likelihood_observation_frequency
            ),
            "observation_aggregation_contract": (
                "whole_integer_source_intervals_projected_exactly_then_grouped_by_midpoint_year"
            ),
            "process_ar1_rho": float(result.process_ar1_rho),
            "process_innovation_sd": float(result.process_innovation_sd),
            "beta_prior_log_sd": float(result.beta_prior_log_sd),
            "reporting_prior_log_sd": float(result.reporting_prior_log_sd),
        }
    return metadata


def save_calibration_artifacts(country: str, calibrated: dict[str, Any], summary: pd.DataFrame, result: Any) -> None:
    artifact_path = calibrated_country_artifact_path(country)
    accepted = bool(summary["calibration_accepted"].iloc[0])
    if accepted:
        artifact_config = deepcopy(calibrated)
        if "log_beta_time_variation" in artifact_config.get("transmission", {}):
            production_end = pd.Timestamp(
                load_configs()["baseline"]["calendar"]["analysis_end_date"]
            )
            artifact_config = extend_annual_log_beta_conditional_mean(
                artifact_config,
                through_year=int(production_end.year),
            )
        payload = {
            "config": artifact_config,
            "metadata": _artifact_metadata(
                country=country,
                base_config=artifact_config,
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

    observed = calibration_observed_case_frame(country)
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
    if method == "state_space_map":
        calibrated, result = state_space_map_calibration(
            config,
            country,
            observed,
            dispersion=dispersion,
            maxiter=maxiter,
        )
        n_starts_used = 1
    elif method == "staged_fast":
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

    exact_predicted_means = _calibration_predicted_means(
        calibrated,
        observed,
        country,
    )
    likelihood_observed = grouped_likelihood_observations(observed)
    predicted = predicted_case_frame_from_interval_means(
        exact_predicted_means,
        likelihood_observed,
    )
    aligned = align_annual_case_series(predicted, likelihood_observed)
    fitted_reported_cases = aligned["predicted_reported_cases"].to_numpy(dtype=float)
    fitted_temporal_mean = float(np.mean(fitted_reported_cases))
    fitted_temporal_sd = float(np.std(fitted_reported_cases, ddof=0))
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
    exact_total_ratio = float(
        aligned["predicted_reported_cases"].sum()
        / max(float(aligned["observed_reported_cases"].sum()), 1e-12)
    )
    relative_tolerance = float(calibration.get("relative_incidence_tolerance", 0.25))
    exact_absolute_fit_passed = bool(
        (1.0 - relative_tolerance)
        <= exact_total_ratio
        <= (1.0 + relative_tolerance)
    )
    posterior_dimension = int(
        len(getattr(result, "posterior_map_vector", []))
    )
    posterior_rank = int(getattr(result, "posterior_rank", posterior_dimension))
    posterior_condition_number = float(
        getattr(result, "posterior_condition_number", 1.0)
    )
    process_settings = calibration.get("process_model", {})
    max_posterior_condition = float(
        process_settings.get("max_posterior_condition_number", 1e8)
    )
    calibration_identifiability_passed = bool(
        posterior_dimension == 0
        or (
            posterior_rank == posterior_dimension
            and np.isfinite(posterior_condition_number)
            and posterior_condition_number <= max_posterior_condition
        )
    )
    calibration_accepted = bool(
        result.success
        and exact_absolute_fit_passed
        and fit_checks["calibration_shape_fit_passed"]
        and calibration_identifiability_passed
    )
    summary["model_to_observed_reported_incidence_ratio"] = exact_total_ratio
    summary["absolute_fit_status"] = (
        "calibrated_to_reported_cases"
        if exact_absolute_fit_passed
        else "outside_calibration_tolerance"
    )
    summary["absolute_fit_source"] = "exact_observed_interval_case_counters"
    summary["data_fit_score"] = data_fit_score
    summary["reporting_rate_prior_penalty"] = float(reporting_rate_prior_penalty(calibrated))
    summary["fit_score"] = float(result.fun)
    summary["calibrated_beta"] = float(calibrated["transmission"]["beta_S"])
    summary["calibrated_multi_year_phase_days"] = float(
        calibrated["transmission"].get("multi_year_phase", 0.0)
    )
    process_variation = calibrated["transmission"].get(
        "log_beta_time_variation",
        {},
    )
    process_periods = process_variation.get("periods", [])
    process_values = np.asarray(
        [float(period.get("log_multiplier", 0.0)) for period in process_periods],
        dtype=float,
    )
    summary["calibration_process_model"] = str(
        process_variation.get("interpretation", "none")
    )
    summary["calibrated_log_beta_process_sd"] = (
        float(np.std(process_values, ddof=0)) if len(process_values) else 0.0
    )
    summary["calibrated_log_beta_process_max_abs"] = (
        float(np.max(np.abs(process_values))) if len(process_values) else 0.0
    )
    summary["calibrated_log_beta_process_path"] = ";".join(
        f"{period.get('start_date','')}:{float(period.get('log_multiplier', 0.0)):.6f}"
        for period in process_periods
    )
    summary["calibration_process_nfev"] = int(getattr(result, "process_nfev", 0))
    summary["calibration_posterior_approximation"] = str(
        getattr(result, "posterior_approximation", "none")
    )
    summary["calibration_posterior_dimension"] = posterior_dimension
    summary["calibration_posterior_rank"] = posterior_rank
    summary["calibration_posterior_condition_number"] = posterior_condition_number
    summary["calibration_max_posterior_condition_number"] = max_posterior_condition
    summary["calibration_identifiability_passed"] = calibration_identifiability_passed
    summary["reporting_multiplier_by_age"] = reporting_by_age
    summary["reporting_rate_prior_by_age"] = reporting_prior_by_age
    summary["reporting_rate_prior_method"] = str(reporting_prior.get("method", ""))
    summary["reporting_rate_prior_evidence_class"] = str(reporting_prior.get("evidence_class", ""))
    # These are descriptive extrema across fitted surveillance intervals, not
    # posterior or predictive uncertainty bounds.  Keeping that distinction in
    # both the column names and the method metadata prevents downstream figures
    # from presenting temporal variation as inferential uncertainty.
    summary["fitted_temporal_mean_reported_cases"] = fitted_temporal_mean
    summary["fitted_temporal_sd_reported_cases"] = fitted_temporal_sd
    summary["fitted_temporal_range_low"] = float(np.min(fitted_reported_cases))
    summary["fitted_temporal_range_high"] = float(np.max(fitted_reported_cases))
    summary["fitted_temporal_range_method"] = (
        "minimum_and_maximum_fitted_reported_cases_across_likelihood_intervals"
    )
    for key, value in fit_checks.items():
        summary[key] = value
    summary["calibration_accepted"] = calibration_accepted
    summary["calibration_success"] = calibration_accepted
    summary["optimizer_success"] = bool(result.success)
    summary["calibration_status"] = (
        "state_reconstruction_accepted"
        if calibration_accepted
        else "state_reconstruction_rejected"
    )
    summary["calibration_validation_scope"] = (
        "in_sample_regularized_state_reconstruction; predictive validity requires "
        "rolling_origin_hindcast"
    )
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
