from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import pandas as pd

from src_python.simulation.common import (
    execute_scenario_list,
    load_configs,
    make_intervention_config,
    write_outputs,
)
from src_python.utils.io import project_path, write_dataframe


NEAR_TERM_END_DATE = "2029-12-31"
NEAR_TERM_END_DAYS = 365.0 * 5.0
MATERNAL_PROTECTION_DURATIONS = (90.0, 180.0, 270.0)
STRATEGIES = (
    "current",
    "maternal_direct_antibody_only",
    "maternal_immunization",
)


def _set_near_term_runtime(config: dict[str, Any]) -> None:
    config.setdefault("calendar", {})["analysis_end_date"] = NEAR_TERM_END_DATE
    config.setdefault("simulation", {})["end_time"] = NEAR_TERM_END_DAYS
    config["simulation"]["output_time_step"] = 30.0
    config["simulation"]["rtol"] = max(float(config["simulation"].get("rtol", 1e-5)), 1e-4)
    config["simulation"]["atol"] = max(float(config["simulation"].get("atol", 1e-7)), 1e-6)


def _apply_maternal_duration(config: dict[str, Any], duration_days: float) -> dict[str, Any]:
    out = deepcopy(config)
    out.setdefault("natural_history", {})["maternal_protection_duration"] = float(duration_days)
    return out


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    for country in configs["countries"]:
        for duration_days in MATERNAL_PROTECTION_DURATIONS:
            for strategy in STRATEGIES:
                config, vaccine_name = make_intervention_config(strategy, country_profile=country)
                config = _apply_maternal_duration(config, duration_days)
                _set_near_term_runtime(config)
                scenarios.append(
                    {
                        "config": config,
                        "analysis": "maternal_duration_sensitivity",
                        "scenario": f"{strategy}_maternal_duration_{duration_days:.0f}d",
                        "vaccine_scenario": vaccine_name,
                        "resistance_scenario": resistance_name,
                        "intervention": strategy,
                        "metadata": {
                            "country": country,
                            "strategy": strategy,
                            "maternal_protection_duration_days": float(duration_days),
                        },
                    }
                )
    return scenarios


def _summarize(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    data["strategy"] = data.get("strategy", data["scenario"])
    data["maternal_protection_duration_days"] = pd.to_numeric(
        data["maternal_protection_duration_days"],
        errors="raise",
    )
    data["annualized_infant_cases_per_100k"] = pd.to_numeric(
        data["annualized_infant_cases_per_100k"],
        errors="coerce",
    )

    current = data[data["strategy"] == "current"][
        ["country", "maternal_protection_duration_days", "annualized_infant_cases_per_100k"]
    ].rename(columns={"annualized_infant_cases_per_100k": "current_infant_cases_per_100k"})
    merged = data.merge(current, on=["country", "maternal_protection_duration_days"], how="left")
    merged["relative_reduction_vs_duration_matched_current"] = (
        1.0
        - merged["annualized_infant_cases_per_100k"]
        / merged["current_infant_cases_per_100k"]
    )

    rows: list[dict[str, Any]] = []
    for (strategy, duration_days), group in merged.groupby(
        ["strategy", "maternal_protection_duration_days"],
        sort=False,
    ):
        infant = group["annualized_infant_cases_per_100k"]
        reduction = group["relative_reduction_vs_duration_matched_current"]
        rows.append(
            {
                "strategy": strategy,
                "maternal_protection_duration_days": float(duration_days),
                "median_infant_cases_per_100k_5y": float(infant.median(skipna=True)),
                "iqr_infant_cases_per_100k_5y": f"{infant.quantile(0.25):.4g}-{infant.quantile(0.75):.4g}",
                "median_infant_case_reduction_vs_current_5y": float(reduction.median(skipna=True)),
                "iqr_infant_case_reduction_vs_current_5y": f"{reduction.quantile(0.25):.4g}-{reduction.quantile(0.75):.4g}",
                "countries_with_positive_reduction": int((reduction > 0).sum()),
                "countries": int(group["country"].nunique()),
                "interpretation": "Near-term sensitivity varying passive maternal antibody duration while holding adult boosting and close-contact exposure-reduction assumptions fixed within each strategy.",
            }
        )
    return pd.DataFrame(rows)


def main(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    timeseries, summary = execute_scenario_list(
        _build_scenarios(configs),
        stem="maternal_duration_sensitivity",
        n_jobs=n_jobs,
    )
    write_outputs(timeseries, summary, "maternal_duration_sensitivity")
    table = _summarize(summary)
    write_dataframe(table, project_path("outputs", "tables", "maternal_duration_sensitivity.csv"))
    return timeseries, summary, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run near-term maternal passive-protection duration sensitivity scenarios."
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
