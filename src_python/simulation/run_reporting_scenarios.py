from __future__ import annotations

from copy import deepcopy

from src_python.simulation.common import load_configs, make_config, run_scenario_list
from src_python.utils.io import deep_update


def _apply_reporting_scenario(config: dict, scenario_def: dict) -> dict:
    """Apply a reporting-rate scenario to the observation layer ONLY.

    Since the decoupling of diagnosis_probability from reporting_rate,
    these scenarios now ONLY change how many true cases appear in
    surveillance data. They do NOT alter treatment rates, resistance
    selection pressure, or any other transmission dynamics.

    To test scenarios where diagnosis/treatment access changes, use
    the diagnosis_probability config key or treatment_rate scenarios.
    """
    out = deepcopy(config)
    out["reporting_multiplier"] = float(scenario_def.get("multiplier", 1.0))

    age_multipliers = scenario_def.get("age_multipliers", {})
    if age_multipliers:
        for record in out["age_groups"]:
            multiplier = float(age_multipliers.get(record["label"], 1.0))
            record["reporting_rate"] = min(1.0, max(0.0, float(record["reporting_rate"]) * multiplier))

    time_variation = scenario_def.get("time_variation")
    if time_variation:
        out["reporting_time_variation"] = deep_update(
            {
                "start_time": out["simulation"]["start_time"],
                "end_time": out["simulation"]["end_time"],
                "start_multiplier": 1.0,
                "end_multiplier": 1.0,
            },
            time_variation,
        )
    return out


def main():
    configs = load_configs()
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    scenarios = []
    for country in configs["countries"]:
        for name, scenario_def in configs["baseline"]["reporting_rate_sensitivity"].items():
            config = make_config(
                vaccine_scenario="symptom_protective",
                resistance_scenario=resistance_name,
                country_profile=country,
            )
            config = _apply_reporting_scenario(config, scenario_def)
            scenarios.append(
                {
                    "config": config,
                    "analysis": "reporting_rate_sensitivity",
                    "scenario": name,
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": resistance_name,
                    "metadata": {"reporting_scenario": name, "country": country},
                }
            )
    return run_scenario_list(scenarios, stem="reporting_scenarios", reference_scenario="medium")


if __name__ == "__main__":
    main()
