from __future__ import annotations

from src_python.simulation.common import load_configs, make_config, run_scenario_list


def main():
    configs = load_configs()
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    scenarios = []
    for country in configs["countries"]:
        for name in configs["vaccines"]:
            scenarios.append(
                {
                    "config": make_config(
                        vaccine_scenario=name,
                        resistance_scenario=resistance_name,
                        country_profile=country,
                    ),
                    "analysis": "vaccine_mechanism",
                    "scenario": name,
                    "vaccine_scenario": name,
                    "resistance_scenario": resistance_name,
                    "metadata": {"country": country},
                }
            )
    return run_scenario_list(scenarios, stem="vaccine_scenarios", reference_scenario="no_vaccine")


if __name__ == "__main__":
    main()
