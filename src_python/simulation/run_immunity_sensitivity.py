"""Immunity structure sensitivity analysis.

Compares the SIRWS (Susceptible-Infected-Recovered-Waned-Susceptible) boosting
model against alternative immunity structures:

1. Baseline SIRWS boosting: literature-centred equal R and W stages
2. Strong boosting (efficiency=0.90): robust natural immune maintenance
3. Weak boosting (efficiency=0.40): limited natural immune maintenance
4. No boosting (disabled): single-stage R→S with the same unboosted mean
5. Long natural immunity: extended R→W duration (15 years)
6. Short vaccine immunity: total 3-year two-stage vaccine protection
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

from src_python.simulation.common import (
    load_configs,
    make_config,
    publication_country_names,
    run_scenario_list,
)


IMMUNITY_STRUCTURES = {
    "baseline_sirws_boosting": {
        "description": (
            "Current model: SIRWS with literature-centred R and W stages. "
            "Boosting efficiency 0.70. "
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
            "Boosting disabled: the implemented single-stage R→S pathway uses "
            "the same mean duration as the baseline R-plus-W pathway. This "
            "changes residence-time structure while holding the unboosted mean fixed."
        ),
        "overrides": {
            "immunity_model": {
                "boosting_enabled": False,
            },
        },
        "match_no_boosting_mean": True,
    },
    "long_natural_immunity": {
        "description": "Extended R→W stage (15 years) with the baseline W→S stage retained.",
        "overrides": {
            "natural_history": {
                "R_to_W_duration": 5475.0,  # 15 years
            },
        },
    },
    "short_vaccine_immunity": {
        "description": "Three-year total two-stage vaccine protection, split equally across recent and waned stages.",
        "overrides": {},
        "total_vaccine_protection_duration": 1095.0,
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


def _apply_immunity_structure(config: dict, name: str, spec: dict) -> dict:
    """Apply one structure while keeping compared mean durations explicit."""

    out = deepcopy(config)
    overrides = spec.get("overrides", {})
    for section in ("natural_history", "immunity_model"):
        for key, value in overrides.get(section, {}).items():
            out.setdefault(section, {})[key] = value

    if bool(spec.get("match_no_boosting_mean", False)):
        natural_history = out["natural_history"]
        natural_history["recovered_immunity_duration"] = float(
            natural_history["R_to_W_duration"] + natural_history["W_to_S_duration"]
        )

    total_vaccine_duration = spec.get("total_vaccine_protection_duration")
    if total_vaccine_duration is not None:
        total = float(total_vaccine_duration)
        if total <= 0.0:
            raise ValueError(f"{name}: total vaccine protection duration must be positive")
        half = 0.5 * total
        out["natural_history"]["vaccine_protection_duration"] = half
        out.setdefault("immunity_model", {})["waned_vaccine_duration"] = half
    return out


def main():
    configs = load_configs()
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    scenarios = []

    for country in publication_country_names(configs):
        for name, spec in IMMUNITY_STRUCTURES.items():
            config = make_config(
                vaccine_scenario="symptom_protective",
                resistance_scenario=resistance_name,
                country_profile=country,
            )
            config = _apply_immunity_structure(config, name, spec)

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
                        "resolved_recovered_immunity_duration": float(
                            config["natural_history"]["recovered_immunity_duration"]
                        ),
                        "resolved_R_to_W_duration": float(
                            config["natural_history"]["R_to_W_duration"]
                        ),
                        "resolved_W_to_S_duration": float(
                            config["natural_history"]["W_to_S_duration"]
                        ),
                        "resolved_vaccine_recent_duration": float(
                            config["natural_history"]["vaccine_protection_duration"]
                        ),
                        "resolved_vaccine_waned_duration": float(
                            config["immunity_model"]["waned_vaccine_duration"]
                        ),
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
