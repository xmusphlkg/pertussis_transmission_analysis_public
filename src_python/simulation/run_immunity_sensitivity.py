"""Immunity structure sensitivity analysis.

Compares the SIRWS (Susceptible-Infected-Recovered-Waned-Susceptible) boosting
model against alternative immunity structures:

1. Baseline SIRWS boosting: R→W (5y) → boosted back to R by re-exposure, or W→S (10y)
2. Strong boosting (efficiency=0.90): robust natural immune maintenance
3. Weak boosting (efficiency=0.40): limited natural immune maintenance
4. No boosting (disabled): equivalent to simple waning-only model
5. Long natural immunity: extended R→W duration (15 years)
6. Short vaccine immunity: 3-year vaccine waning (faster than baseline 5y)
7. Fast W→S: rapid loss of waned immunity (3 years) — amplifies immunity debt

The SIRWS model (Lavine et al. 2011 PNAS; Wearing & Rohani 2009) explains:
- Why pertussis persists at low levels in highly vaccinated populations
  (natural boosting maintains herd immunity)
- Why COVID-19 NPIs caused post-pandemic pertussis surges globally
  (reduced circulation → no boosting → W accumulates → S increases)
- Why China's 2024 surge was so explosive (3 years of zero-COVID
  eliminated natural boosting, creating massive immunity debt)

Usage:
    python -m src_python.simulation.run_immunity_sensitivity
"""
from __future__ import annotations

from copy import deepcopy

from src_python.simulation.common import load_configs, make_config, run_scenario_list


IMMUNITY_STRUCTURES = {
    "baseline_sirws_boosting": {
        "description": (
            "Current model: SIRWS with immune boosting. R→W in 5 years, "
            "W→S in 10 years if not boosted. Boosting efficiency 0.70. "
            "Natural boosting by circulating pathogen maintains population immunity."
        ),
        "overrides": {},
    },
    "sirws_strong_boosting": {
        "description": "High boosting efficiency (0.90) — strong natural immune maintenance.",
        "overrides": {
            "immunity_model": {
                "boosting_efficiency": 0.90,
            },
        },
    },
    "sirws_weak_boosting": {
        "description": "Low boosting efficiency (0.40) — weak natural immune maintenance.",
        "overrides": {
            "immunity_model": {
                "boosting_efficiency": 0.40,
            },
        },
    },
    "sirws_no_boosting": {
        "description": (
            "Boosting disabled: equivalent to simple waning-only model. "
            "R→W→S without any re-exposure boosting. Tests the null hypothesis "
            "that boosting is not needed to explain observed dynamics."
        ),
        "overrides": {
            "immunity_model": {
                "boosting_enabled": False,
            },
        },
    },
    "long_natural_immunity": {
        "description": "Extended R→W duration (15 years) simulating intrinsically durable immunity.",
        "overrides": {
            "natural_history": {
                "R_to_W_duration": 5475.0,  # 15 years
            },
        },
    },
    "short_vaccine_immunity": {
        "description": "Faster vaccine waning (3 years) representing rapid aP decline.",
        "overrides": {
            "natural_history": {
                "vaccine_protection_duration": 1095.0,  # 3 years
            },
        },
    },
    "fast_W_to_S": {
        "description": (
            "Rapid W→S transition (3 years). Tests scenario where unboosted "
            "individuals lose immunity quickly, amplifying immunity debt effects."
        ),
        "overrides": {
            "natural_history": {
                "W_to_S_duration": 1095.0,  # 3 years
            },
        },
    },
}


def main():
    configs = load_configs()
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    scenarios = []

    for country in configs["countries"]:
        for name, spec in IMMUNITY_STRUCTURES.items():
            config = make_config(
                vaccine_scenario="symptom_protective",
                resistance_scenario=resistance_name,
                country_profile=country,
            )
            # Apply immunity structure overrides
            overrides = spec.get("overrides", {})
            if "natural_history" in overrides:
                for key, value in overrides["natural_history"].items():
                    config["natural_history"][key] = value
            if "immunity_model" in overrides:
                for key, value in overrides["immunity_model"].items():
                    config.setdefault("immunity_model", {})[key] = value

            scenarios.append(
                {
                    "config": config,
                    "analysis": "immunity_structure_sensitivity",
                    "scenario": name,
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": resistance_name,
                    "metadata": {
                        "country": country,
                        "immunity_structure": name,
                        "description": spec["description"],
                    },
                }
            )

    return run_scenario_list(
        scenarios,
        stem="immunity_sensitivity",
        reference_scenario="baseline_sirws_boosting",
    )


if __name__ == "__main__":
    main()
