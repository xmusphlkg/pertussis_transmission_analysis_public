"""Generate one locked deterministic parent for all Figure 2 programme options.

The routine-timeliness sensitivity run intentionally uses a coarser diagnostic
runtime.  Its current-practice denominator must therefore not be spliced into
the production intervention comparison.  This runner evaluates current
practice and all six programme-only strategies together with the production
runtime so that burden, relative reduction, leader, and runner-up are defined
against one within-profile comparator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import (
    add_relative_reductions,
    current_run_metadata,
    enforce_calibration_status,
    execute_scenario_list,
    load_configs,
    make_intervention_config,
    publication_country_names,
    validate_calibration_artifacts,
    write_run_metadata,
)
from src_python.simulation.run_joint_psa_rank_acceptability import (
    PROGRAMME_ONLY_STRATEGIES,
)
from src_python.simulation.run_routine_timeliness_sensitivity import (
    _apply_timeliness,
)
from src_python.utils.io import project_path, write_dataframe


STEM = "figure2_programme_reference"
OUTPUT_PATH = project_path(
    "outputs", "summaries", "figure2_programme_reference_summary.csv"
)
TIMESERIES_PATH = project_path(
    "outputs", "simulations", "figure2_programme_reference.parquet"
)
REFERENCE_SCENARIO = "current"
STRATEGIES = (REFERENCE_SCENARIO, *PROGRAMME_ONLY_STRATEGIES)
PRIMARY_RATE = "annualized_child_adolescent_cases_per_100k"
PRIMARY_TOTAL = "total_child_adolescent_cases"
PRIMARY_REDUCTION = "relative_reduction_child_adolescent_cases"


def _strategy_config(strategy: str, country: str) -> tuple[dict[str, Any], str]:
    if strategy == "timeliness_only":
        config, vaccine_name = make_intervention_config(
            REFERENCE_SCENARIO, country_profile=country
        )
        return _apply_timeliness(config), vaccine_name
    return make_intervention_config(strategy, country_profile=country)


def _build_scenarios(countries: tuple[str, ...]) -> list[dict[str, Any]]:
    configs = load_configs()
    resistance_name = configs["baseline"].get(
        "baseline_resistance_scenario", "country_timeline"
    )
    scenarios: list[dict[str, Any]] = []
    for country in countries:
        for strategy in STRATEGIES:
            config, vaccine_name = _strategy_config(strategy, country)
            scenarios.append(
                {
                    "config": config,
                    "analysis": STEM,
                    "scenario": strategy,
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": strategy,
                    "metadata": {
                        "country": country,
                        "strategy": strategy,
                        "figure2_reference_parent": True,
                        "common_current_practice_denominator": True,
                    },
                }
            )
    return scenarios


def _canonicalise_and_validate(
    summary: pd.DataFrame,
    *,
    countries: tuple[str, ...],
) -> pd.DataFrame:
    required = {
        "country",
        "scenario",
        "strategy",
        PRIMARY_RATE,
        PRIMARY_TOTAL,
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise KeyError(f"Figure 2 reference summary is missing columns: {missing}")

    out = summary.copy()
    out["country"] = out["country"].astype(str).str.replace(" ", "_", regex=False)
    out["strategy"] = out["strategy"].astype(str)
    expected_rows = len(countries) * len(STRATEGIES)
    if (
        len(out) != expected_rows
        or out[["country", "strategy"]].duplicated().any()
        or set(out["country"]) != set(countries)
        or set(out["strategy"]) != set(STRATEGIES)
    ):
        raise ValueError(
            "Figure 2 reference run requires exactly one row per profile and strategy"
        )

    out[PRIMARY_RATE] = pd.to_numeric(out[PRIMARY_RATE], errors="raise")
    out[PRIMARY_TOTAL] = pd.to_numeric(out[PRIMARY_TOTAL], errors="raise")
    if (
        not np.isfinite(out[[PRIMARY_RATE, PRIMARY_TOTAL]].to_numpy()).all()
        or out[PRIMARY_RATE].le(0.0).any()
        or out[PRIMARY_TOTAL].le(0.0).any()
    ):
        raise ValueError("Figure 2 reference burdens must be finite and positive")

    current = out.loc[
        out["strategy"].eq(REFERENCE_SCENARIO),
        ["country", PRIMARY_RATE, PRIMARY_TOTAL],
    ].rename(
        columns={
            PRIMARY_RATE: "current_primary_cases_per_100k",
            PRIMARY_TOTAL: "current_primary_total_cases",
        }
    )
    out = out.drop(
        columns=[
            column
            for column in (
                "current_primary_cases_per_100k",
                "current_primary_total_cases",
            )
            if column in out.columns
        ]
    ).merge(current, on="country", how="left", validate="many_to_one")
    out[PRIMARY_REDUCTION] = 1.0 - (
        out[PRIMARY_RATE] / out["current_primary_cases_per_100k"]
    )
    out["figure2_reference_estimand"] = (
        "within-profile annualised <18 symptomatic-case index versus the same "
        "production-runtime current-practice row"
    )

    programme = out.loc[out["strategy"].isin(PROGRAMME_ONLY_STRATEGIES)].copy()
    programme["burden_rank"] = programme.groupby("country")[PRIMARY_RATE].rank(
        method="first", ascending=True
    )
    programme["reduction_rank"] = programme.groupby("country")[
        PRIMARY_REDUCTION
    ].rank(method="first", ascending=False)
    if not np.array_equal(
        programme["burden_rank"].to_numpy(),
        programme["reduction_rank"].to_numpy(),
    ):
        raise AssertionError(
            "Figure 2 argmin burden and argmax common-denominator reduction disagree"
        )

    algebraic = 1.0 - out[PRIMARY_RATE] / out["current_primary_cases_per_100k"]
    if not np.allclose(
        out[PRIMARY_REDUCTION].to_numpy(),
        algebraic.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("Figure 2 relative reductions failed the denominator audit")
    return out.sort_values(["country", "strategy"]).reset_index(drop=True)


def generate(*, n_jobs: int | None = None) -> pd.DataFrame:
    countries = tuple(publication_country_names(load_configs()))
    validate_calibration_artifacts(
        countries,
        context="Figure 2 programme reference preflight",
    )
    timeseries, summary = execute_scenario_list(
        _build_scenarios(countries),
        stem=STEM,
        n_jobs=n_jobs,
    )
    summary = add_relative_reductions(
        summary,
        reference_scenario=REFERENCE_SCENARIO,
    )
    enforce_calibration_status(summary, stem=STEM)
    output = _canonicalise_and_validate(summary, countries=countries)
    write_dataframe(timeseries, TIMESERIES_PATH)
    write_dataframe(output, OUTPUT_PATH)
    metadata = current_run_metadata(
        STEM,
        row_counts={
            "timeseries": int(len(timeseries)),
            "summary": int(len(output)),
        },
    )
    metadata.update(
        {
            "countries": list(countries),
            "strategies": list(STRATEGIES),
            "programme_only_strategies": list(PROGRAMME_ONLY_STRATEGIES),
            "primary_endpoint": PRIMARY_RATE,
            "reference_scenario": REFERENCE_SCENARIO,
            "common_current_practice_denominator": True,
            "runtime_contract": "production runtime; no diagnostic solver coarsening",
        }
    )
    write_run_metadata(STEM, metadata)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    generate(n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
