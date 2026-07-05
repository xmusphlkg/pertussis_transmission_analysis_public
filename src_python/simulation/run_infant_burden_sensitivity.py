"""Infant burden sensitivity analysis.

Exploratory simulation script: not part of the current publication pipeline.

Targeted sensitivity analysis for parameters with the highest influence on
infant pertussis cases, based on LHS screening results showing:
  - relative_infectiousness_asymptomatic: PRCC ~0.59 with infant cases
  - infectious_duration_symptomatic: PRCC ~0.45 with infant cases
  - infectious_duration_asymptomatic: PRCC ~0.40 with infant cases
  - VE_inf: PRCC ~-0.35 with infant cases (protective)

These parameters control the "silent reservoir" of adult/adolescent infections
that maintain transmission chains reaching unvaccinated infants. The analysis
produces:
  1. Two-dimensional grids crossing the most influential parameter pairs
  2. Infant-specific outcome metrics (cases/100k, infections/100k)
  3. Threshold analysis: at what parameter values does infant burden exceed
     WHO-relevant thresholds?

References:
  - Warfel et al. 2014 (baboon model: aP prevents disease but not colonization)
  - Althouse & Scarpino 2015 (asymptomatic transmission drives pertussis resurgence)
  - WHO pertussis position paper 2015 (subclinical infection role)
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    execute_scenario_list,
    load_configs,
    make_config,
    write_outputs,
)
from src_python.utils.io import project_path, write_dataframe


# Grid values for the two most influential parameters
RELATIVE_INFECTIOUSNESS_VALUES = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
INFECTIOUS_DURATION_SYMPTOMATIC_VALUES = [14.0, 17.0, 21.0, 25.0, 28.0]
VE_INF_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]


def _build_grid_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build 2D grid: relative_infectiousness_asymptomatic × VE_inf."""
    scenarios = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")

    # Select a subset of countries for the grid (computationally intensive)
    grid_countries = ["Australia", "China", "United_Kingdom"]
    available = [c for c in grid_countries if c in configs["countries"]]

    for country in available:
        for rel_inf in RELATIVE_INFECTIOUSNESS_VALUES:
            for ve_inf in VE_INF_VALUES:
                scenario_name = f"relinf_{rel_inf:.2f}_VEinf_{ve_inf:.2f}"
                config = make_config(
                    vaccine_scenario="symptom_protective",
                    resistance_scenario=resistance_name,
                    country_profile=country,
                    load_calibration=False,
                )
                config["transmission"]["relative_infectiousness_asymptomatic"] = float(rel_inf)
                config["vaccine"]["VE_inf"] = float(ve_inf)

                scenarios.append({
                    "config": config,
                    "analysis": "infant_burden_sensitivity",
                    "scenario": scenario_name,
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": resistance_name,
                    "metadata": {
                        "country": country,
                        "relative_infectiousness_asymptomatic": float(rel_inf),
                        "VE_inf": float(ve_inf),
                    },
                })

    return scenarios


def _build_duration_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build grid: infectious_duration_symptomatic × relative_infectiousness."""
    scenarios = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")

    grid_countries = ["Australia", "China", "United_Kingdom"]
    available = [c for c in grid_countries if c in configs["countries"]]

    for country in available:
        for duration in INFECTIOUS_DURATION_SYMPTOMATIC_VALUES:
            for rel_inf in [0.25, 0.45, 0.65]:
                scenario_name = f"dur_{duration:.0f}_relinf_{rel_inf:.2f}"
                config = make_config(
                    vaccine_scenario="symptom_protective",
                    resistance_scenario=resistance_name,
                    country_profile=country,
                    load_calibration=False,
                )
                config["natural_history"]["infectious_duration_symptomatic"] = float(duration)
                config["transmission"]["relative_infectiousness_asymptomatic"] = float(rel_inf)

                scenarios.append({
                    "config": config,
                    "analysis": "infant_burden_duration_sensitivity",
                    "scenario": scenario_name,
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": resistance_name,
                    "metadata": {
                        "country": country,
                        "infectious_duration_symptomatic": float(duration),
                        "relative_infectiousness_asymptomatic": float(rel_inf),
                    },
                })

    return scenarios


def _compute_infant_sensitivity_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract infant-specific metrics and compute sensitivity indices."""
    rows: list[dict[str, Any]] = []

    for _, row in summary.iterrows():
        entry = {
            "country": row.get("country", ""),
            "scenario": row.get("scenario", ""),
        }
        # Extract grid parameters from metadata columns
        for param in [
            "relative_infectiousness_asymptomatic",
            "VE_inf",
            "infectious_duration_symptomatic",
        ]:
            if param in row.index:
                entry[param] = float(pd.to_numeric(pd.Series([row[param]]), errors="coerce").iloc[0])

        # Infant outcomes
        for outcome in [
            "annualized_infant_cases_per_100k",
            "annualized_infant_infections_per_100k",
            "annualized_reported_cases_per_100k",
            "total_infant_cases",
        ]:
            if outcome in row.index:
                entry[outcome] = float(pd.to_numeric(pd.Series([row[outcome]]), errors="coerce").iloc[0])

        rows.append(entry)

    return pd.DataFrame(rows)


def main(n_jobs: int | None = None, skip_duration_grid: bool = False):
    """Run infant burden sensitivity analysis.

    Produces:
      - 2D grid: relative_infectiousness × VE_inf → infant cases
      - 2D grid: infectious_duration × relative_infectiousness → infant cases
      - Summary with sensitivity metrics
    """
    configs = load_configs()

    # Phase 1: Main grid (relative_infectiousness × VE_inf)
    grid_scenarios = _build_grid_scenarios(configs)
    print(f"Running {len(grid_scenarios)} grid scenarios (rel_inf × VE_inf)...")
    ts_grid, summary_grid = execute_scenario_list(
        grid_scenarios, stem="infant_burden_grid", n_jobs=n_jobs
    )
    write_outputs(ts_grid, summary_grid, "infant_burden_grid")

    infant_summary = _compute_infant_sensitivity_summary(summary_grid)
    write_dataframe(
        infant_summary,
        project_path("outputs/summaries/infant_burden_sensitivity_grid.csv"),
    )

    # Phase 2: Duration grid
    if not skip_duration_grid:
        duration_scenarios = _build_duration_scenarios(configs)
        print(f"Running {len(duration_scenarios)} duration grid scenarios...")
        ts_dur, summary_dur = execute_scenario_list(
            duration_scenarios, stem="infant_burden_duration", n_jobs=n_jobs
        )
        write_outputs(ts_dur, summary_dur, "infant_burden_duration")

        duration_summary = _compute_infant_sensitivity_summary(summary_dur)
        write_dataframe(
            duration_summary,
            project_path("outputs/summaries/infant_burden_duration_sensitivity.csv"),
        )

    return ts_grid, summary_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run infant burden sensitivity analysis."
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--skip-duration-grid", action="store_true")
    args = parser.parse_args()
    main(n_jobs=args.n_jobs, skip_duration_grid=args.skip_duration_grid)
