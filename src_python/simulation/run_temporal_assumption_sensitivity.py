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


def _set_near_term_runtime(config: dict[str, Any]) -> None:
    config.setdefault("calendar", {})["analysis_end_date"] = NEAR_TERM_END_DATE
    config.setdefault("simulation", {})["end_time"] = NEAR_TERM_END_DAYS
    config["simulation"]["output_time_step"] = 30.0
    config["simulation"]["rtol"] = max(float(config["simulation"].get("rtol", 1e-5)), 1e-4)
    config["simulation"]["atol"] = max(float(config["simulation"].get("atol", 1e-7)), 1e-6)


def _scale_npi_reductions(config: dict[str, Any], scale: float | None) -> None:
    periods = config.get("transmission", {}).get("npi_contact_reduction_periods")
    if not periods:
        return
    if scale is None:
        config["transmission"].pop("npi_contact_reduction_periods", None)
        return
    scaled = []
    for period in periods:
        period = deepcopy(period)
        period["reduction"] = float(period.get("reduction", 0.0)) * float(scale)
        scaled.append(period)
    config["transmission"]["npi_contact_reduction_periods"] = scaled


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")

    for country in configs["countries"]:
        for burn_in_years in (10, 15, 30):
            config, vaccine_name = make_intervention_config("current", country_profile=country)
            _set_near_term_runtime(config)
            config["simulation"]["burn_in_years"] = float(burn_in_years)
            scenarios.append(
                {
                    "config": config,
                    "analysis": "temporal_assumption_sensitivity",
                    "scenario": f"burnin_{burn_in_years}y",
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": "current",
                    "metadata": {
                        "country": country,
                        "temporal_dimension": "burn_in",
                        "burn_in_years": float(burn_in_years),
                        "npi_scenario": "baseline_contact_reduction",
                        "npi_reduction_scale": 1.0,
                        "implementation_note": "Near-term current-practice run varying pre-analysis burn-in duration.",
                    },
                }
            )

    npi_overrides = {
        "transmission": {
            "npi_contact_reduction": {
                "enabled": True,
                "reduction_column": "contact_reduction_mean",
            }
        }
    }
    for country in configs["countries"]:
        for scenario_name, scale in (
            ("npi_country_profile", 1.0),
            ("npi_reduction_half", 0.5),
            ("npi_none", None),
        ):
            config, vaccine_name = make_intervention_config(
                "current",
                country_profile=country,
                config_overrides=npi_overrides,
            )
            _set_near_term_runtime(config)
            _scale_npi_reductions(config, scale)
            scenarios.append(
                {
                    "config": config,
                    "analysis": "temporal_assumption_sensitivity",
                    "scenario": scenario_name,
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": "current",
                    "metadata": {
                        "country": country,
                        "temporal_dimension": "npi_contact_shock",
                        "burn_in_years": float(config["simulation"].get("burn_in_years", np.nan)),
                        "npi_scenario": scenario_name,
                        "npi_reduction_scale": np.nan if scale is None else float(scale),
                        "implementation_note": "Near-term current-practice run varying COVID-19 NPI contact-reduction assumptions for countries with explicit NPI periods.",
                    },
                }
            )
    return scenarios


def _summarize(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dimension, scenario), group in summary.groupby(["temporal_dimension", "scenario"], sort=False):
        infant = pd.to_numeric(group["annualized_infant_cases_per_100k"], errors="coerce")
        infections = pd.to_numeric(group["annualized_infections_per_100k"], errors="coerce")
        resistant_end = pd.to_numeric(group["resistant_fraction_end"], errors="coerce")
        first = group.iloc[0]
        rows.append(
            {
                "temporal_dimension": dimension,
                "scenario": scenario,
                "countries": int(group["country"].nunique()),
                "burn_in_years": first.get("burn_in_years", np.nan),
                "npi_reduction_scale": first.get("npi_reduction_scale", np.nan),
                "median_infant_cases_per_100k_5y": float(infant.median(skipna=True)),
                "iqr_infant_cases_per_100k_5y": f"{infant.quantile(0.25):.4g}-{infant.quantile(0.75):.4g}",
                "median_all_infections_per_100k_5y": float(infections.median(skipna=True)),
                "median_end_resistant_fraction_5y": float(resistant_end.median(skipna=True)),
                "implementation_note": first.get("implementation_note", ""),
            }
        )
    return pd.DataFrame(rows)


def main(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    scenarios = _build_scenarios(configs)
    timeseries, summary = execute_scenario_list(
        scenarios,
        stem="temporal_assumption_sensitivity",
        n_jobs=n_jobs,
    )
    write_outputs(timeseries, summary, "temporal_assumption_sensitivity")
    table = _summarize(summary)
    write_dataframe(table, project_path("outputs", "tables", "temporal_assumption_sensitivity.csv"))
    return timeseries, summary, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run near-term burn-in and COVID-19 NPI temporal assumption sensitivity.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
