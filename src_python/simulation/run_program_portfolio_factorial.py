from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import pandas as pd

from src_python.simulation.common import (
    apply_intervention_definition,
    load_configs,
    make_config,
    run_scenario_list,
)
from src_python.simulation.run_routine_timeliness_sensitivity import _apply_timeliness, _set_summary_runtime
from src_python.utils.io import project_path, write_dataframe


PORTFOLIO_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "portfolio": "current",
        "layers": (),
        "timeliness": False,
        "resistance_guided": False,
        "implementation_intensity": 0,
        "description": "Current practice.",
    },
    {
        "portfolio": "routine_timeliness",
        "layers": (),
        "timeliness": True,
        "resistance_guided": False,
        "implementation_intensity": 1,
        "description": "Routine timeliness improvement only.",
    },
    {
        "portfolio": "infant_exposure",
        "layers": ("maternal_immunization",),
        "timeliness": False,
        "resistance_guided": False,
        "implementation_intensity": 3,
        "description": "Infant-exposure reduction composite without routine timeliness changes.",
    },
    {
        "portfolio": "timeliness_pregnancy_tdap",
        "layers": ("pregnancy_tdap_scaleup",),
        "timeliness": True,
        "resistance_guided": False,
        "implementation_intensity": 2,
        "description": "Routine timeliness plus pregnancy Tdap scale-up.",
    },
    {
        "portfolio": "timeliness_targeted_pep",
        "layers": ("targeted_pep_high_risk",),
        "timeliness": True,
        "resistance_guided": False,
        "implementation_intensity": 2,
        "description": "Routine timeliness plus targeted high-risk PEP.",
    },
    {
        "portfolio": "timeliness_infant_exposure",
        "layers": ("maternal_immunization",),
        "timeliness": True,
        "resistance_guided": False,
        "implementation_intensity": 4,
        "description": "Routine timeliness plus infant-exposure reduction composite.",
    },
    {
        "portfolio": "timeliness_infant_exposure_targeted_pep",
        "layers": ("maternal_immunization", "targeted_pep_high_risk"),
        "timeliness": True,
        "resistance_guided": False,
        "implementation_intensity": 5,
        "description": "Routine timeliness plus infant-exposure reduction and targeted high-risk PEP.",
    },
    {
        "portfolio": "routine_timeliness_resistance_guided",
        "layers": ("resistance_guided_treatment",),
        "timeliness": True,
        "resistance_guided": True,
        "implementation_intensity": 3,
        "description": "Routine timeliness plus resistance-guided management.",
    },
    {
        "portfolio": "infant_exposure_resistance_guided",
        "layers": ("maternal_immunization", "resistance_guided_treatment"),
        "timeliness": False,
        "resistance_guided": True,
        "implementation_intensity": 5,
        "description": "Infant-exposure reduction plus resistance-guided management.",
    },
    {
        "portfolio": "timeliness_infant_exposure_resistance_guided",
        "layers": ("maternal_immunization", "resistance_guided_treatment"),
        "timeliness": True,
        "resistance_guided": True,
        "implementation_intensity": 6,
        "description": "Routine timeliness plus infant-exposure reduction and resistance-guided management.",
    },
    {
        "portfolio": "timeliness_infant_exposure_targeted_pep_resistance_guided",
        "layers": ("maternal_immunization", "targeted_pep_high_risk", "resistance_guided_treatment"),
        "timeliness": True,
        "resistance_guided": True,
        "implementation_intensity": 7,
        "description": "Routine timeliness plus infant-exposure reduction, targeted PEP, and resistance-guided management.",
    },
)


def _build_portfolio_config(configs: dict[str, Any], country: str, definition: dict[str, Any]) -> dict[str, Any]:
    config = make_config(
        vaccine_scenario=configs["baseline"].get("baseline_vaccine_scenario", "symptom_protective"),
        resistance_scenario=configs["baseline"].get("baseline_resistance_scenario", "country_timeline"),
        country_profile=country,
    )
    interventions = configs["interventions"]
    for layer in definition["layers"]:
        config = apply_intervention_definition(config, interventions[layer])
    if definition["timeliness"]:
        config = _apply_timeliness(config)
    return _set_summary_runtime(config)


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    vaccine_name = configs["baseline"].get("baseline_vaccine_scenario", "symptom_protective")
    for country in configs["countries"]:
        for definition in PORTFOLIO_DEFINITIONS:
            portfolio = definition["portfolio"]
            layers = ";".join(definition["layers"])
            config = _build_portfolio_config(configs, country, definition)
            scenarios.append(
                {
                    "config": config,
                    "analysis": "program_portfolio_factorial",
                    "scenario": portfolio,
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": portfolio,
                    "metadata": {
                        "country": country,
                        "portfolio": portfolio,
                        "program_layers": layers,
                        "timeliness_applied": bool(definition["timeliness"]),
                        "resistance_guided": bool(definition["resistance_guided"]),
                        "implementation_intensity": int(definition["implementation_intensity"]),
                        "implementation_note": definition["description"],
                    },
                }
            )
    return scenarios


def _resistant_infections_per_100k(df: pd.DataFrame) -> pd.Series:
    denominator = df["analysis_years"].replace(0, pd.NA) * df["total_population"].replace(0, pd.NA)
    return df["resistant_infections"] / denominator * 100_000.0


def _summarize(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = summary.copy()
    data["portfolio"] = data.get("portfolio", data["scenario"])
    data["annualized_resistant_infections_per_100k"] = _resistant_infections_per_100k(data)
    current = data.loc[
        data["portfolio"].eq("current"),
        ["country", "annualized_infant_cases_per_100k", "annualized_resistant_infections_per_100k"],
    ].rename(
        columns={
            "annualized_infant_cases_per_100k": "current_infant_cases_per_100k",
            "annualized_resistant_infections_per_100k": "current_resistant_infections_per_100k",
        }
    )
    country = data.merge(current, on="country", how="left")
    country["relative_reduction_resistant_infections"] = (
        1.0
        - country["annualized_resistant_infections_per_100k"]
        / country["current_resistant_infections_per_100k"].replace(0, pd.NA)
    )
    zero_current = country["current_resistant_infections_per_100k"].fillna(0).abs() <= 1e-12
    zero_scenario = country["annualized_resistant_infections_per_100k"].fillna(0).abs() <= 1e-12
    country.loc[zero_current & zero_scenario, "relative_reduction_resistant_infections"] = 0.0
    country["infant_case_rank_within_portfolio_set"] = country.groupby("country")[
        "annualized_infant_cases_per_100k"
    ].rank(method="min", ascending=True)

    rows: list[dict[str, Any]] = []
    for portfolio, group in country.groupby("portfolio", sort=False):
        infant_reduction = pd.to_numeric(group["relative_reduction_infant_cases"], errors="coerce")
        resistant_reduction = pd.to_numeric(group["relative_reduction_resistant_infections"], errors="coerce")
        cases = pd.to_numeric(group["annualized_infant_cases_per_100k"], errors="coerce")
        first = group.iloc[0]
        rows.append(
            {
                "portfolio": portfolio,
                "timeliness_applied": bool(first["timeliness_applied"]),
                "resistance_guided": bool(first["resistance_guided"]),
                "program_layers": first.get("program_layers", ""),
                "implementation_intensity": int(first["implementation_intensity"]),
                "median_infant_cases_per_100k": float(cases.median(skipna=True)),
                "median_relative_reduction_infant_cases": float(infant_reduction.median(skipna=True)),
                "iqr_relative_reduction_infant_cases": (
                    f"{infant_reduction.quantile(0.25):.4g} to {infant_reduction.quantile(0.75):.4g}"
                ),
                "countries_with_positive_infant_reduction": int((infant_reduction > 0).sum()),
                "median_relative_reduction_resistant_infections": float(resistant_reduction.median(skipna=True)),
                "countries_ranked_first_infant_cases": int(
                    (group["infant_case_rank_within_portfolio_set"].to_numpy(dtype=float) == 1).sum()
                ),
                "countries": int(group["country"].nunique()),
                "implementation_note": first.get("implementation_note", ""),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["countries_ranked_first_infant_cases", "median_relative_reduction_infant_cases"],
        ascending=[False, False],
    )
    country = country.sort_values(["country", "implementation_intensity", "portfolio"])
    return table, country


def main(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    timeseries, summary = run_scenario_list(
        _build_scenarios(configs),
        stem="program_portfolio_factorial",
        reference_scenario="current",
        n_jobs=n_jobs,
    )
    table, country = _summarize(summary)
    write_dataframe(table, project_path("outputs", "tables", "program_portfolio_factorial_summary.csv"))
    write_dataframe(country, project_path("outputs", "tables", "program_portfolio_factorial_country.csv"))
    return timeseries, summary, table, country


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run supplementary program portfolio factorial scenarios.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
