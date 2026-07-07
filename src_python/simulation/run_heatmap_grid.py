from __future__ import annotations

import os

import numpy as np

from src_python.simulation.common import (
    add_relative_reductions,
    current_run_metadata,
    enforce_calibration_status,
    execute_scenario_summary_list,
    load_configs,
    make_config,
    write_run_metadata,
)
from src_python.utils.io import project_path, write_dataframe


def _values_from_env(name: str, default: np.ndarray) -> np.ndarray:
    raw = os.environ.get(name)
    if not raw:
        return default
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} was provided but did not contain any numeric values.")
    return np.unique(np.round(np.array(values, dtype=float), 2))


def main():
    configs = load_configs()
    ve_inf_max = float(
        configs["sensitivity"]["parameters"].get("VE_inf", {}).get("max", 0.75)
    )
    ve_inf_points = int(os.environ.get("PERTUSSIS_VEINF_GRID_POINTS", "13"))
    resistance_points = int(os.environ.get("PERTUSSIS_RESISTANCE_GRID_POINTS", "21"))
    ve_inf_values = _values_from_env(
        "PERTUSSIS_VEINF_GRID_VALUES",
        np.unique(np.round(np.linspace(0.0, ve_inf_max, ve_inf_points), 2)),
    )
    resistance_values = _values_from_env(
        "PERTUSSIS_RESISTANCE_GRID_VALUES",
        np.unique(np.round(np.linspace(0.0, 1.0, resistance_points), 2)),
    )
    stem = os.environ.get("PERTUSSIS_HEATMAP_GRID_STEM", "veinf_resistance_grid")
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
                        "analysis": stem,
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
    summary = execute_scenario_summary_list(scenarios, stem=stem)
    summary = add_relative_reductions(summary, reference_scenario="VEinf_0.00_res_0.00")
    enforce_calibration_status(summary, stem=stem)
    write_dataframe(summary, project_path("outputs", "summaries", f"{stem}_summary.csv"))
    metadata = current_run_metadata(
        stem,
        row_counts={"summary": int(len(summary))},
    )
    metadata.update(
        {
            "output_scope": "summary_only",
            "grid_VE_inf_values": [float(value) for value in ve_inf_values],
            "grid_resistance_prevalence_values": [float(value) for value in resistance_values],
        }
    )
    write_run_metadata(stem, metadata)
    return summary


if __name__ == "__main__":
    main()
