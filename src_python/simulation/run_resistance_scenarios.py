from __future__ import annotations

from src_python.simulation.common import load_configs, make_config, run_scenario_list


def main():
    configs = load_configs()
    scenarios = []
    for country in configs["countries"]:
        for name in configs["resistance"]:
            scenarios.append(
                {
                    "config": make_config(
                        vaccine_scenario="symptom_protective",
                        resistance_scenario=name,
                        country_profile=country,
                    ),
                    "analysis": "macrolide_resistance",
                    "scenario": name,
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": name,
                    "metadata": {"country": country},
                }
            )
    return run_scenario_list(scenarios, stem="resistance_scenarios", reference_scenario="low")


if __name__ == "__main__":
    main()
