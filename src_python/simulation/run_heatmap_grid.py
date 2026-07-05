from __future__ import annotations

import numpy as np

from src_python.simulation.common import (
    load_configs,
    make_config,
    run_scenario_list,
)


def main():
    configs = load_configs()
    ve_inf_max = float(
        configs["sensitivity"]["parameters"].get("VE_inf", {}).get("max", 0.75)
    )
    ve_inf_values = np.round(np.linspace(0.0, ve_inf_max, 7), 2)
    resistance_values = np.round(np.linspace(0.0, 1.0, 7), 2)
    scenarios = []

    for country in configs["countries"]:
        for ve_inf in ve_inf_values:
            for resistance in resistance_values:
                scenario = f"VEinf_{ve_inf:.2f}_res_{resistance:.2f}"
                config = make_config(
                    vaccine_scenario="symptom_protective",
                    resistance_scenario="moderate",
                    country_profile=country,
                    vaccine_overrides={"VE_inf": float(ve_inf)},
                    resistance_overrides={
                        "target_prevalence_at_analysis_start": float(resistance),
                        "initial_resistance_prevalence": float(resistance),
                        "importation_fraction": float(resistance),
                    },
                )
                scenarios.append(
                    {
                        "config": config,
                        "analysis": "veinf_resistance_grid",
                        "scenario": scenario,
                        "vaccine_scenario": "symptom_protective",
                        "resistance_scenario": "custom_grid",
                        "metadata": {
                            "grid_VE_inf": float(ve_inf),
                            "grid_resistance_prevalence": float(resistance),
                            "country": country,
                        },
                    }
                )
    return run_scenario_list(
        scenarios,
        stem="veinf_resistance_grid",
        reference_scenario="VEinf_0.00_res_0.00",
    )


if __name__ == "__main__":
    main()
