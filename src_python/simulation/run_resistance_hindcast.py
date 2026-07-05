"""Resistance hindcast validation.

For countries with multiple time-point resistance observations (China, Japan,
Australia), this module initializes the model at an earlier resistance prevalence
and simulates forward to check whether the model can reproduce the observed
trajectory under plausible fitness/importation/treatment assumptions.

This addresses the concern that near-fixation in the model may be a structural
artifact rather than a mechanistically supported outcome.

Usage:
    python -m src_python.simulation.run_resistance_hindcast
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    current_run_metadata,
    make_config,
    run_prepared_config,
    write_run_metadata,
)
from src_python.utils.parallel import parallel_map
from src_python.utils.io import project_path, write_dataframe


# Countries with at least 2 time-point resistance observations
HINDCAST_COUNTRIES = {
    "China": {
        "init_year": 2016,
        "init_resistance": 0.364,
        "observations": [
            {"year": 2022, "fraction": 0.972, "lower": 0.94, "upper": 0.99},
            {"year": 2024, "fraction": 0.997, "lower": 0.986, "upper": 1.0},
        ],
        "hindcast_years": 9,  # 2016 -> 2025
    },
    "Japan": {
        "init_year": 2024,
        "init_resistance": 0.875,
        "observations": [
            {"year": 2025, "fraction": 0.827, "lower": 0.697, "upper": 0.918},
        ],
        "hindcast_years": 2,  # 2024 -> 2026
    },
    "Australia": {
        "init_year": 2024,
        "init_resistance": 0.043,
        "observations": [
            # Only one time point available; use as endpoint check
            {"year": 2024, "fraction": 0.043, "lower": 0.019, "upper": 0.082},
        ],
        "hindcast_years": 3,  # Short forward projection to check stability
    },
}

# Fitness values to sweep for each country
FITNESS_VALUES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]


def _make_hindcast_config(
    country: str,
    init_resistance: float,
    hindcast_years: int,
    fitness_R: float,
    init_year: int,
) -> dict:
    """Build a config for hindcast validation with shortened horizon."""
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile=country,
        load_calibration=True,
        config_overrides={
            "resistance": {
                "target_prevalence_at_analysis_start": init_resistance,
                "importation_fraction": init_resistance,
                "rebalance_after_burn_in": True,
                "prevalence_anchor_rate_per_year": 2.0,
                "anchor_during_dynamics": False,
            },
            "importation": {
                "resistant_fraction": init_resistance,
            },
            "initial_conditions": {
                "initial_resistance_prevalence": init_resistance,
            },
            "transmission": {
                "fitness_R": fitness_R,
            },
            "simulation": {
                "end_time": hindcast_years * 365,
                "burn_in_years": 10,
                "output_time_step": 30,
            },
            "calendar": {
                "enabled": True,
                "analysis_start_date": f"{init_year}-01-01",
                "analysis_end_date": f"{init_year + hindcast_years}-12-31",
            },
        },
    )
    return config


def run_country_hindcast(country: str) -> pd.DataFrame:
    """Run hindcast for a single country across fitness values."""
    tasks = [{"country": country, "fitness_R": fitness_R} for fitness_R in FITNESS_VALUES]
    df = _run_hindcast_tasks(tasks)
    return _attach_observations(df, [country])


def _run_hindcast_tasks(tasks: list[dict[str, Any]], *, n_jobs: int | None = None) -> pd.DataFrame:
    frames = parallel_map(
        _run_hindcast_task,
        tasks,
        desc="resistance_hindcast",
        n_jobs=n_jobs,
    )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _run_hindcast_task(task: dict[str, Any]) -> pd.DataFrame:
    country = str(task["country"])
    fitness_R = float(task["fitness_R"])
    spec = HINDCAST_COUNTRIES[country]

    config = _make_hindcast_config(
        country=country,
        init_resistance=spec["init_resistance"],
        hindcast_years=spec["hindcast_years"],
        fitness_R=fitness_R,
        init_year=spec["init_year"],
    )

    ts, _summary = run_prepared_config(
        config,
        analysis="resistance_hindcast",
        scenario=f"fitness_{fitness_R:.2f}",
        vaccine_scenario="symptom_protective",
        resistance_scenario="hindcast",
        metadata={"country": country, "fitness_R": fitness_R},
    )

    if "calendar_year" in ts.columns and "resistant_fraction" in ts.columns:
        annual = ts.groupby("calendar_year")["resistant_fraction"].mean().reset_index()
    elif "time" in ts.columns and "resistant_fraction" in ts.columns:
        ts = ts.copy()
        ts["sim_year"] = spec["init_year"] + ts["time"] / 365.0
        annual = (
            ts.assign(calendar_year=ts["sim_year"].astype(int))
            .groupby("calendar_year")["resistant_fraction"]
            .mean()
            .reset_index()
        )
    else:
        annual = pd.DataFrame(columns=["calendar_year", "resistant_fraction"])

    if annual.empty:
        return pd.DataFrame()

    out = annual.rename(columns={"resistant_fraction": "model_resistant_fraction"}).copy()
    out["country"] = country
    out["fitness_R"] = fitness_R
    out["calendar_year"] = out["calendar_year"].astype(int)
    out["model_resistant_fraction"] = out["model_resistant_fraction"].astype(float)
    out["init_year"] = spec["init_year"]
    out["init_resistance"] = spec["init_resistance"]
    return out[
        [
            "country",
            "fitness_R",
            "calendar_year",
            "model_resistant_fraction",
            "init_year",
            "init_resistance",
        ]
    ]


def _attach_observations(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    obs_rows = []
    for country in countries:
        spec = HINDCAST_COUNTRIES[country]
        for obs in spec["observations"]:
            obs_rows.append({
                "country": country,
                "year": obs["year"],
                "observed_fraction": obs["fraction"],
                "observed_lower": obs["lower"],
                "observed_upper": obs["upper"],
            })
    obs_df = pd.DataFrame(obs_rows)
    if obs_df.empty:
        return df
    return df.merge(
        obs_df.rename(columns={"year": "calendar_year"}),
        on=["country", "calendar_year"],
        how="left",
    )


def _score_hindcast(df: pd.DataFrame) -> pd.DataFrame:
    """Score each fitness scenario by how well it matches observations."""
    scored = df.dropna(subset=["observed_fraction"]).copy()
    if scored.empty:
        return pd.DataFrame()

    scored["absolute_error"] = (
        scored["model_resistant_fraction"] - scored["observed_fraction"]
    ).abs()
    scored["within_ci"] = (
        (scored["model_resistant_fraction"] >= scored["observed_lower"])
        & (scored["model_resistant_fraction"] <= scored["observed_upper"])
    )

    summary = (
        scored.groupby(["country", "fitness_R"])
        .agg(
            mean_absolute_error=("absolute_error", "mean"),
            max_absolute_error=("absolute_error", "max"),
            observations_within_ci=("within_ci", "sum"),
            total_observations=("within_ci", "count"),
        )
        .reset_index()
    )
    summary["fraction_within_ci"] = (
        summary["observations_within_ci"] / summary["total_observations"]
    )
    return summary.sort_values(["country", "mean_absolute_error"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resistance hindcast validation.")
    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Subset of countries to hindcast (default: all available).",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel worker count.")
    args = parser.parse_args()

    countries = args.countries or list(HINDCAST_COUNTRIES.keys())
    countries = [c for c in countries if c in HINDCAST_COUNTRIES]

    tasks = [
        {"country": country, "fitness_R": fitness_R}
        for country in countries
        for fitness_R in FITNESS_VALUES
    ]
    combined = _attach_observations(
        _run_hindcast_tasks(tasks, n_jobs=args.n_jobs),
        countries,
    )

    if not combined.empty:
        output_path = project_path("outputs/tables/resistance_hindcast_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_dataframe(combined, output_path)

        # Score and write summary
        scores = _score_hindcast(combined)
        if not scores.empty:
            score_path = project_path("outputs/tables/resistance_hindcast_scores.csv")
            write_dataframe(scores, score_path)
            print("\nHindcast scoring summary:")
            print(scores.to_string(index=False))

    else:
        raise RuntimeError("No hindcast results generated.")

    write_run_metadata(
        "resistance_hindcast",
        current_run_metadata(
            "resistance_hindcast",
            row_counts={
                "results": int(len(combined)),
                "scores": int(len(scores)) if not combined.empty and not scores.empty else 0,
            },
        ),
    )


if __name__ == "__main__":
    main()
