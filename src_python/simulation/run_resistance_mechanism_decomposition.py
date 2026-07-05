from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import execute_scenario_list, load_configs, make_config, write_outputs
from src_python.utils.io import project_path, write_dataframe


def _set_equal_treatment(config: dict[str, Any]) -> None:
    sensitive = deepcopy(config.get("treatment", {}).get("sensitive", {}))
    config.setdefault("treatment", {})["resistant"] = sensitive


def _set_equal_pep(config: dict[str, Any]) -> None:
    pep = config.setdefault("PEP", {})
    pep["effectiveness_resistant"] = float(pep.get("effectiveness_sensitive", pep.get("effectiveness_resistant", 0.0)))


def _set_no_resistant_importation(config: dict[str, Any]) -> None:
    config.setdefault("importation", {})["resistant_fraction"] = 0.0
    config.setdefault("resistance", {})["importation_fraction"] = 0.0


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_specs = [
        (
            "baseline_full_mechanism",
            "yes",
            "yes",
            "yes",
            1.00,
            "Full baseline mechanism: country anchor, resistant importation, strain-specific treatment and PEP, neutral fitness.",
            (),
        ),
        (
            "no_resistant_importation",
            "no",
            "yes",
            "yes",
            1.00,
            "Tests dependence on ongoing resistant-strain importation after the analysis-start anchor.",
            (_set_no_resistant_importation,),
        ),
        (
            "equal_treatment_effect",
            "yes",
            "no",
            "yes",
            1.00,
            "Tests treatment-mediated selection by making resistant treatment effects equal to sensitive-strain effects.",
            (_set_equal_treatment,),
        ),
        (
            "equal_pep_effect",
            "yes",
            "yes",
            "no",
            1.00,
            "Tests PEP-mediated selection by making resistant PEP effectiveness equal to sensitive-strain PEP effectiveness.",
            (_set_equal_pep,),
        ),
        (
            "no_treatment_or_pep_differential",
            "yes",
            "no",
            "no",
            1.00,
            "Tests neutral strain competition under importation when treatment and PEP do not favor resistant strains.",
            (_set_equal_treatment, _set_equal_pep),
        ),
        (
            "fitness_cost",
            "yes",
            "yes",
            "yes",
            0.85,
            "Fitness-cost stress test retaining baseline importation and management assumptions.",
            (),
        ),
    ]

    scenarios: list[dict[str, Any]] = []
    for country in configs["countries"]:
        for name, importation, treatment_diff, pep_diff, fitness, interpretation, mutators in scenario_specs:
            config = make_config(
                vaccine_scenario="symptom_protective",
                resistance_scenario="country_timeline",
                country_profile=country,
            )
            config["transmission"]["fitness_R"] = float(fitness)
            for mutator in mutators:
                mutator(config)

            scenarios.append(
                {
                    "config": config,
                    "analysis": "resistance_mechanism_decomposition",
                    "scenario": name,
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": "country_timeline",
                    "metadata": {
                        "country": country,
                        "mechanism_scenario": name,
                        "resistant_importation": importation,
                        "treatment_differential": treatment_diff,
                        "pep_differential": pep_diff,
                        "fitness_R": float(fitness),
                        "mechanism_interpretation": interpretation,
                    },
                }
            )
    return scenarios


def _summarize_decomposition(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, group in summary.groupby("mechanism_scenario", sort=False):
        end_fraction = pd.to_numeric(group["resistant_fraction_end"], errors="coerce")
        infant = pd.to_numeric(group["annualized_infant_cases_per_100k"], errors="coerce")
        resistant = pd.to_numeric(group["annualized_infections_per_100k"], errors="coerce") * pd.to_numeric(
            group["resistant_fraction"], errors="coerce"
        )
        first = group.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "importation": first.get("resistant_importation", ""),
                "treatment_differential": first.get("treatment_differential", ""),
                "pep_differential": first.get("pep_differential", ""),
                "fitness_R": first.get("fitness_R", np.nan),
                "median_end_resistant_fraction": float(end_fraction.median(skipna=True)),
                "iqr_end_resistant_fraction": f"{end_fraction.quantile(0.25):.4g}-{end_fraction.quantile(0.75):.4g}",
                "median_infant_cases_per_100k": float(infant.median(skipna=True)),
                "median_resistant_infections_per_100k": float(resistant.median(skipna=True)),
                "interpretation": first.get("mechanism_interpretation", ""),
            }
        )
    return pd.DataFrame(rows)


def main(n_jobs: int | None = None):
    configs = load_configs()
    scenarios = _build_scenarios(configs)
    timeseries, summary = execute_scenario_list(
        scenarios,
        stem="resistance_mechanism_decomposition",
        n_jobs=n_jobs,
    )
    write_outputs(timeseries, summary, "resistance_mechanism_decomposition")
    table = _summarize_decomposition(summary)
    write_dataframe(
        table,
        project_path("outputs", "tables", "resistance_mechanism_decomposition.csv"),
    )
    return timeseries, summary, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run resistance mechanism decomposition scenarios.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
