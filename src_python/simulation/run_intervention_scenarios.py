from __future__ import annotations

from src_python.simulation.common import load_configs, make_intervention_config, run_scenario_list


def main():
    configs = load_configs()
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    scenarios = []
    for country in configs["countries"]:
        for name in configs["interventions"]:
            config, vaccine_name = make_intervention_config(name, country_profile=country)
            scenarios.append(
                {
                    "config": config,
                    "analysis": "intervention_comparison",
                    "scenario": name,
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": name,
                    "metadata": {"country": country},
                }
            )
    return run_scenario_list(scenarios, stem="intervention_scenarios", reference_scenario="current")


if __name__ == "__main__":
    main()
