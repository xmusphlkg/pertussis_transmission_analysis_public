from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src_python.calibration.calibrate_baseline import (
    align_annual_case_series,
    observed_annual_case_frame,
    predicted_case_frame,
    retain_recent_observed_window,
)
from src_python.simulation.common import (
    calibrated_country_artifact_path,
    current_run_metadata,
    load_configs,
    run_prepared_config,
    write_run_metadata,
)
from src_python.utils.io import load_yaml, project_path, write_dataframe


def _period_label(year: int) -> str:
    if year < 2020:
        return "pre_pandemic"
    if year <= 2021:
        return "pandemic_npi"
    return "post_pandemic"


def _relative_error(predicted: pd.Series, observed: pd.Series) -> pd.Series:
    return (predicted - observed) / observed.replace(0, np.nan)


def _safe_mape(predicted: pd.Series, observed: pd.Series) -> float:
    mask = observed.astype(float).gt(0)
    if not bool(mask.any()):
        return np.nan
    return float(((predicted[mask] - observed[mask]).abs() / observed[mask]).mean())


def _run_country(country: str, recent_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_path = calibrated_country_artifact_path(country)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing calibrated artifact for {country}: {artifact_path}")
    artifact = load_yaml(artifact_path)
    config = artifact["config"]
    metadata = artifact.get("metadata", {})

    observed = observed_annual_case_frame(country)
    observed = retain_recent_observed_window(observed, recent_years)
    timeseries, _ = run_prepared_config(
        config,
        analysis="calibration_diagnostic",
        scenario=f"{country}_calibrated_window",
        vaccine_scenario=metadata.get("baseline_vaccine_scenario", ""),
        resistance_scenario=metadata.get("baseline_resistance_scenario", ""),
        metadata={"country": country},
    )
    predicted = predicted_case_frame(timeseries, observed)
    aligned = align_annual_case_series(predicted, observed).copy()
    aligned["country"] = country
    aligned["period"] = aligned["observed_year"].astype(int).map(_period_label)
    aligned["absolute_error"] = (
        aligned["predicted_reported_cases"] - aligned["observed_reported_cases"]
    ).abs()
    aligned["relative_error"] = _relative_error(
        aligned["predicted_reported_cases"], aligned["observed_reported_cases"]
    )
    aligned["absolute_percentage_error"] = aligned["relative_error"].abs()

    peak_observed = aligned.loc[aligned["observed_reported_cases"].idxmax()]
    peak_predicted = aligned.loc[aligned["predicted_reported_cases"].idxmax()]
    summary_rows = []
    total_predicted = aligned["predicted_reported_cases"]
    total_observed = aligned["observed_reported_cases"]
    summary_rows.append(
        {
            "country": country,
            "period": "overall",
            "n_intervals": int(len(aligned)),
            "observed_total_reported_cases": float(total_observed.sum()),
            "predicted_total_reported_cases": float(total_predicted.sum()),
            "mean_observed_reported_cases": float(total_observed.mean()),
            "mean_predicted_reported_cases": float(total_predicted.mean()),
            "mean_absolute_percentage_error": _safe_mape(total_predicted, total_observed),
            "peak_observed_year": int(peak_observed["observed_year"]),
            "peak_predicted_year": int(peak_predicted["observed_year"]),
            "peak_timing_error_years": int(peak_predicted["observed_year"])
            - int(peak_observed["observed_year"]),
            "observed_peak_reported_cases": float(peak_observed["observed_reported_cases"]),
            "predicted_peak_reported_cases": float(peak_predicted["predicted_reported_cases"]),
            "peak_magnitude_ratio": float(
                peak_predicted["predicted_reported_cases"]
                / max(float(peak_observed["observed_reported_cases"]), 1e-9)
            ),
        }
    )
    for period, group in aligned.groupby("period", sort=False):
        predicted_period = group["predicted_reported_cases"]
        observed_period = group["observed_reported_cases"]
        summary_rows.append(
            {
                "country": country,
                "period": period,
                "n_intervals": int(len(group)),
                "observed_total_reported_cases": float(observed_period.sum()),
                "predicted_total_reported_cases": float(predicted_period.sum()),
                "mean_observed_reported_cases": float(observed_period.mean()),
                "mean_predicted_reported_cases": float(predicted_period.mean()),
                "mean_absolute_percentage_error": _safe_mape(predicted_period, observed_period),
                "peak_observed_year": "",
                "peak_predicted_year": "",
                "peak_timing_error_years": "",
                "observed_peak_reported_cases": "",
                "predicted_peak_reported_cases": "",
                "peak_magnitude_ratio": "",
            }
        )
    return aligned, pd.DataFrame(summary_rows)


def run(countries: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    recent_years = int(configs["baseline"].get("calibration", {}).get("recent_years", 0))
    countries = list(countries or configs["countries"].keys())
    aligned_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for country in countries:
        aligned, summary = _run_country(country, recent_years)
        aligned_frames.append(aligned)
        summary_frames.append(summary)

    interval_table = pd.concat(aligned_frames, ignore_index=True)
    summary_table = pd.concat(summary_frames, ignore_index=True)
    write_dataframe(
        interval_table,
        project_path("outputs", "tables", "calibration_interval_diagnostics.csv"),
    )
    write_dataframe(
        summary_table,
        project_path("outputs", "tables", "calibration_fit_diagnostics_summary.csv"),
    )
    metadata = current_run_metadata(
        "calibration_diagnostics",
        row_counts={
            "interval_diagnostics": int(len(interval_table)),
            "fit_diagnostics_summary": int(len(summary_table)),
        },
    )
    metadata.update(
        {
            "countries": countries,
            "recent_years": int(recent_years),
        }
    )
    write_run_metadata("calibration_diagnostics", metadata)
    return interval_table, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Export interval-level calibration diagnostics.")
    parser.add_argument("--country", action="append", default=None, help="Country key to run; repeatable.")
    args = parser.parse_args()
    run(args.country)


if __name__ == "__main__":
    main()
