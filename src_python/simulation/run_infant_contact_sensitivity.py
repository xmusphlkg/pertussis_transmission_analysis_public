from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
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
INFANT_TARGETS = ("infant_0_2m", "infant_3_11m")
HOUSEHOLD_LIKE_SOURCES = (
    "child_1_4y",
    "child_5_9y",
    "adolescent_10_17y",
    "young_adult_18_39y",
    "middle_adult_40_64y",
)
SELECTED_STRATEGIES = ("current", "maternal_immunization")


def _set_near_term_runtime(config: dict[str, Any]) -> None:
    config.setdefault("calendar", {})["analysis_end_date"] = NEAR_TERM_END_DATE
    config.setdefault("simulation", {})["end_time"] = NEAR_TERM_END_DAYS
    config["simulation"]["output_time_step"] = 30.0
    config["simulation"]["rtol"] = max(float(config["simulation"].get("rtol", 1e-5)), 1e-4)
    config["simulation"]["atol"] = max(float(config["simulation"].get("atol", 1e-7)), 1e-6)


def _apply_infant_contact_multiplier(config: dict[str, Any], multiplier: float) -> dict[str, Any]:
    out = deepcopy(config)
    labels = [record["label"] for record in out["age_groups"]]
    rows = out["contact_matrix"]["rows"]
    for target in INFANT_TARGETS:
        if target not in labels:
            continue
        target_idx = labels.index(target)
        for source in HOUSEHOLD_LIKE_SOURCES:
            if source not in labels:
                continue
            source_idx = labels.index(source)
            rows[target_idx][source_idx] = float(rows[target_idx][source_idx]) * float(multiplier)
    return out


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    multipliers = (0.75, 1.00, 1.25, 1.50)
    for country in configs["countries"]:
        for strategy in SELECTED_STRATEGIES:
            for multiplier in multipliers:
                config, vaccine_name = make_intervention_config(strategy, country_profile=country)
                config = _apply_infant_contact_multiplier(config, multiplier)
                _set_near_term_runtime(config)
                scenarios.append(
                    {
                        "config": config,
                        "analysis": "infant_contact_sensitivity",
                        "scenario": f"{strategy}_infant_contact_x{multiplier:.2f}",
                        "vaccine_scenario": vaccine_name,
                        "resistance_scenario": configs["baseline"].get("baseline_resistance_scenario", "country_timeline"),
                        "intervention": strategy,
                        "metadata": {
                            "country": country,
                            "strategy": strategy,
                            "infant_contact_multiplier": float(multiplier),
                            "target_ages": ",".join(INFANT_TARGETS),
                            "source_ages": ",".join(HOUSEHOLD_LIKE_SOURCES),
                        },
                    }
                )
    return scenarios


def _summarize(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, multiplier), group in summary.groupby(["strategy", "infant_contact_multiplier"], sort=False):
        infant = pd.to_numeric(group["annualized_infant_cases_per_100k"], errors="coerce")
        infections = pd.to_numeric(group["annualized_infections_per_100k"], errors="coerce")
        rows.append(
            {
                "strategy": strategy,
                "infant_contact_multiplier": float(multiplier),
                "median_infant_cases_per_100k": float(infant.median(skipna=True)),
                "iqr_infant_cases_per_100k": f"{infant.quantile(0.25):.4g}-{infant.quantile(0.75):.4g}",
                "median_all_infections_per_100k": float(infections.median(skipna=True)),
                "countries": int(group["country"].nunique()),
                "interpretation": "Multiplier applied to contact-matrix entries from child/adolescent/adult sources into infant target age groups.",
            }
        )
    return pd.DataFrame(rows)


def main(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    timeseries, summary = execute_scenario_list(
        _build_scenarios(configs),
        stem="infant_contact_sensitivity",
        n_jobs=n_jobs,
    )
    write_outputs(timeseries, summary, "infant_contact_sensitivity")
    table = _summarize(summary)
    write_dataframe(table, project_path("outputs", "tables", "infant_contact_sensitivity.csv"))
    return timeseries, summary, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infant contact-matrix sensitivity scenarios.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
