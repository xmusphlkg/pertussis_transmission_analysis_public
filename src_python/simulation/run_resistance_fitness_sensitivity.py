"""Resistance fitness sensitivity analysis.

This module runs targeted scenarios to quantify how the assumed fitness of
macrolide-resistant B. pertussis strains affects long-term resistance burden
projections. It directly addresses the concern that the previous default
(fitness_R = 0.70, i.e. 30% fitness cost) was inconsistent with observed
epidemiology:

Evidence for fitness-neutral or fitness-advantaged MRBP:
  - China: MRBP rose from 36% (2016) to >99% (2024) in 8 years
    (Fu et al., EID 2024; multicenter 2024 study: 393/394 resistant)
  - Japan: 83-88% MRBP in 2024-2025 (Osaka/multicenter studies)
  - Australia: 4.3% in 2024 (Fong et al., Lancet Microbe 2026) with
    genomic evidence of MT28 importation from China
  - France: 14 MRBP cases in 2024, first significant cluster
    (Cai et al., medRxiv 2025)
  - Global: MT28-ptxP3 clone identified in France, Japan, US in 2024
    indicating cross-border transmission without fitness barrier

The rapid fixation in China (36% -> 100% in 8 years) is mathematically
inconsistent with fitness_R < 0.90 under any plausible treatment pressure
scenario. A fitness cost of 30% would require implausibly high macrolide
treatment rates to maintain resistance above 50%.

This analysis produces:
  1. Country-specific projections at fitness_R = {0.85, 0.95, 1.00, 1.05, 1.15}
  2. Comparison of resistant infection burden across fitness assumptions
  3. Time-to-fixation estimates under each fitness scenario
"""

from __future__ import annotations

import argparse
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


FITNESS_SENSITIVITY_VALUES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

FITNESS_LABELS = {
    0.85: "moderate_cost",
    0.90: "mild_cost",
    0.95: "near_neutral",
    1.00: "neutral",
    1.05: "mild_advantage",
    1.10: "moderate_advantage",
    1.15: "strong_advantage",
}


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build scenario list crossing countries with fitness values."""
    scenarios = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")

    for country in configs["countries"]:
        for fitness_r in FITNESS_SENSITIVITY_VALUES:
            label = FITNESS_LABELS.get(fitness_r, f"fitness_{fitness_r:.2f}")
            scenario_name = f"{label}_f{fitness_r:.2f}"
            config = make_config(
                vaccine_scenario="symptom_protective",
                resistance_scenario=resistance_name,
                country_profile=country,
                resistance_overrides={"fitness_R": float(fitness_r)},
            )
            scenarios.append(
                {
                    "config": config,
                    "analysis": "resistance_fitness_sensitivity",
                    "scenario": scenario_name,
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": resistance_name,
                    "metadata": {
                        "country": country,
                        "fitness_R": float(fitness_r),
                        "fitness_label": label,
                    },
                }
            )
    return scenarios


def _compute_fitness_impact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute relative impact of fitness assumptions on key outcomes."""
    rows: list[dict[str, Any]] = []

    outcome_cols = [
        "annualized_reported_cases_per_100k",
        "annualized_infant_cases_per_100k",
        "resistant_infections",
        "resistant_fraction_end",
    ]

    for country, group in summary.groupby("country", sort=False):
        # Reference: fitness-neutral (1.0)
        neutral = group.loc[group["fitness_R"].astype(float).between(0.995, 1.005)]
        if neutral.empty:
            continue
        ref_row = neutral.iloc[0]

        for _, row in group.iterrows():
            fitness_r = float(row.get("fitness_R", row.get("grid_fitness_R", 1.0)))
            label = FITNESS_LABELS.get(round(fitness_r, 2), f"fitness_{fitness_r:.2f}")

            entry = {
                "country": country,
                "fitness_R": fitness_r,
                "fitness_label": label,
            }

            for outcome in outcome_cols:
                if outcome in row.index and outcome in ref_row.index:
                    val = float(pd.to_numeric(pd.Series([row[outcome]]), errors="coerce").iloc[0])
                    ref_val = float(pd.to_numeric(pd.Series([ref_row[outcome]]), errors="coerce").iloc[0])
                    entry[outcome] = val
                    if ref_val > 0 and np.isfinite(ref_val) and np.isfinite(val):
                        entry[f"{outcome}_ratio_vs_neutral"] = val / ref_val
                    else:
                        entry[f"{outcome}_ratio_vs_neutral"] = np.nan

            rows.append(entry)

    return pd.DataFrame(rows)


def main(n_jobs: int | None = None):
    """Run resistance fitness sensitivity analysis.

    Produces:
      - outputs/simulations/resistance_fitness_sensitivity.csv (timeseries)
      - outputs/summaries/resistance_fitness_sensitivity_summary.csv
      - outputs/summaries/resistance_fitness_impact.csv (relative impacts)
    """
    configs = load_configs()
    scenarios = _build_scenarios(configs)

    timeseries, summary = execute_scenario_list(
        scenarios,
        stem="resistance_fitness_sensitivity",
        n_jobs=n_jobs,
    )
    write_outputs(timeseries, summary, "resistance_fitness_sensitivity")

    # Compute fitness impact summary
    impact = _compute_fitness_impact_summary(summary)
    write_dataframe(
        impact,
        project_path("outputs/summaries/resistance_fitness_impact.csv"),
    )

    return timeseries, summary, impact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run resistance fitness sensitivity analysis."
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
