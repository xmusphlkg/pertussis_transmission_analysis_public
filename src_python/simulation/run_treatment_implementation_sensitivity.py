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


def _interpolate(base: float, target: float, fraction: float) -> float:
    fraction = min(1.0, max(0.0, float(fraction)))
    return float(base + fraction * (target - base))


def _guided_config(
    country: str,
    *,
    uptake: float,
    pep_restored: bool,
    pep_coverage_multiplier: float = 1.0,
) -> tuple[dict[str, Any], str]:
    base, vaccine_name = make_intervention_config("current", country_profile=country)
    full, _ = make_intervention_config("resistance_guided_treatment", country_profile=country)
    config = deepcopy(base)

    for key in ("infectious_duration_reduction", "infectiousness_reduction"):
        config["treatment"]["resistant"][key] = _interpolate(
            float(base["treatment"]["resistant"][key]),
            float(full["treatment"]["resistant"][key]),
            uptake,
        )
    config["treatment"]["treatment_rate_symptomatic"] = _interpolate(
        float(base["treatment"]["treatment_rate_symptomatic"]),
        float(full["treatment"]["treatment_rate_symptomatic"]),
        uptake,
    )

    if pep_restored:
        config["PEP"]["effectiveness_resistant"] = _interpolate(
            float(base["PEP"]["effectiveness_resistant"]),
            float(full["PEP"]["effectiveness_resistant"]),
            uptake,
        )
    else:
        config["PEP"]["effectiveness_resistant"] = float(base["PEP"]["effectiveness_resistant"])
    config["PEP"]["coverage_household_contacts"] = float(base["PEP"]["coverage_household_contacts"]) * float(
        pep_coverage_multiplier
    )
    return config, vaccine_name


def _scenario_specs() -> list[dict[str, Any]]:
    return [
        {
            "scenario": "current_near_term",
            "uptake": 0.0,
            "pep_restored": "baseline",
            "pep_coverage_multiplier": 1.0,
            "implementation_note": "Current macrolide treatment and baseline resistant-strain PEP effectiveness.",
        },
        {
            "scenario": "guided_uptake_25_pep_restored",
            "uptake": 0.25,
            "pep_restored": "yes",
            "pep_coverage_multiplier": 1.0,
            "implementation_note": "Quarter uptake of resistance-guided treatment with partial restoration of resistant-strain PEP effectiveness.",
        },
        {
            "scenario": "guided_uptake_50_pep_restored",
            "uptake": 0.50,
            "pep_restored": "yes",
            "pep_coverage_multiplier": 1.0,
            "implementation_note": "Half uptake of resistance-guided treatment with partial restoration of resistant-strain PEP effectiveness.",
        },
        {
            "scenario": "guided_uptake_75_pep_restored",
            "uptake": 0.75,
            "pep_restored": "yes",
            "pep_coverage_multiplier": 1.0,
            "implementation_note": "Three-quarter uptake of resistance-guided treatment with partial restoration of resistant-strain PEP effectiveness.",
        },
        {
            "scenario": "guided_uptake_100_pep_restored",
            "uptake": 1.00,
            "pep_restored": "yes",
            "pep_coverage_multiplier": 1.0,
            "implementation_note": "Full resistance-guided treatment scenario used in the main analysis.",
        },
        {
            "scenario": "guided_uptake_50_no_pep_restoration",
            "uptake": 0.50,
            "pep_restored": "no",
            "pep_coverage_multiplier": 1.0,
            "implementation_note": "Half uptake of resistance-guided treatment; resistant-strain PEP effectiveness remains at baseline.",
        },
        {
            "scenario": "guided_uptake_100_no_pep_restoration",
            "uptake": 1.00,
            "pep_restored": "no",
            "pep_coverage_multiplier": 1.0,
            "implementation_note": "Full treatment restoration but no restoration of resistant-strain PEP effectiveness.",
        },
        {
            "scenario": "guided_uptake_50_low_pep_reach",
            "uptake": 0.50,
            "pep_restored": "yes",
            "pep_coverage_multiplier": 0.50,
            "implementation_note": "Half uptake and half baseline PEP reach, approximating delayed activation, lower adherence, or household-only reach.",
        },
    ]


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for country in configs["countries"]:
        for spec in _scenario_specs():
            if spec["scenario"] == "current_near_term":
                config, vaccine_name = make_intervention_config("current", country_profile=country)
            else:
                config, vaccine_name = _guided_config(
                    country,
                    uptake=float(spec["uptake"]),
                    pep_restored=spec["pep_restored"] == "yes",
                    pep_coverage_multiplier=float(spec["pep_coverage_multiplier"]),
                )
            _set_near_term_runtime(config)
            scenarios.append(
                {
                    "config": config,
                    "analysis": "treatment_implementation_sensitivity",
                    "scenario": spec["scenario"],
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": configs["baseline"].get("baseline_resistance_scenario", "country_timeline"),
                    "intervention": "resistance_guided_treatment" if spec["scenario"] != "current_near_term" else "current",
                    "metadata": {
                        "country": country,
                        "implementation_uptake": float(spec["uptake"]),
                        "pep_restored": spec["pep_restored"],
                        "pep_coverage_multiplier": float(spec["pep_coverage_multiplier"]),
                        "implementation_note": spec["implementation_note"],
                    },
                }
            )
    return scenarios


def _summarize(summary: pd.DataFrame) -> pd.DataFrame:
    current = summary.loc[summary["scenario"].eq("current_near_term"), ["country", "total_infant_cases"]].rename(
        columns={"total_infant_cases": "current_total_infant_cases"}
    )
    out = summary.merge(current, on="country", how="left")
    out["relative_reduction_infant_cases_vs_current_near_term"] = 1.0 - (
        out["total_infant_cases"] / out["current_total_infant_cases"].replace(0, np.nan)
    )
    grouped = []
    for scenario, group in out.groupby("scenario", sort=False):
        first = group.iloc[0]
        reductions = pd.to_numeric(group["relative_reduction_infant_cases_vs_current_near_term"], errors="coerce")
        grouped.append(
            {
                "scenario": scenario,
                "implementation_uptake": first.get("implementation_uptake", np.nan),
                "pep_restored": first.get("pep_restored", ""),
                "pep_coverage_multiplier": first.get("pep_coverage_multiplier", np.nan),
                "median_infant_case_reduction_vs_current_5y": float(reductions.median(skipna=True)),
                "iqr_infant_case_reduction_vs_current_5y": f"{reductions.quantile(0.25):.4g}-{reductions.quantile(0.75):.4g}",
                "countries_with_positive_reduction": int((reductions > 0).sum()),
                "median_infant_cases_per_100k": float(
                    pd.to_numeric(group["annualized_infant_cases_per_100k"], errors="coerce").median(skipna=True)
                ),
                "implementation_note": first.get("implementation_note", ""),
            }
        )
    return pd.DataFrame(grouped)


def main(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    timeseries, summary = execute_scenario_list(
        _build_scenarios(configs),
        stem="treatment_implementation_sensitivity",
        n_jobs=n_jobs,
    )
    write_outputs(timeseries, summary, "treatment_implementation_sensitivity")
    table = _summarize(summary)
    write_dataframe(
        table,
        project_path("outputs", "tables", "treatment_implementation_sensitivity.csv"),
    )
    return timeseries, summary, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run near-term resistance-guided treatment implementation sensitivity.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
