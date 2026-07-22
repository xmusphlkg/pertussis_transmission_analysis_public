from __future__ import annotations

from src_python.simulation.common import (
    attach_prospective_policy_history,
    load_configs,
    make_config,
    publication_country_names,
    run_scenario_list,
)


VACCINE_TRANSITION_DESIGN = "immediate_full_mechanism_replacement_at_policy_t0"


def build_scenarios(configs=None):
    """Build paired vaccine-mechanism policies from one current-practice t0.

    Each profile first follows the baseline current aP-like mechanism through
    the complete historical integration.  All five settings then branch from
    the resulting common analysis-start state.  The post-t0 parameter switch
    applies immediately to every vaccine-origin compartment, so these are
    mechanism-target contrasts rather than cohort-rollout forecasts.
    """

    configs = configs or load_configs()
    baseline_vaccine_name = str(
        configs["baseline"].get("baseline_vaccine_scenario", "symptom_protective")
    )
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    scenarios = []
    for country in publication_country_names(configs):
        history_config = make_config(
            vaccine_scenario=baseline_vaccine_name,
            resistance_scenario=resistance_name,
            country_profile=country,
        )
        for name in configs["vaccines"]:
            policy_config = make_config(
                vaccine_scenario=name,
                resistance_scenario=resistance_name,
                country_profile=country,
            )
            policy_config = attach_prospective_policy_history(
                policy_config,
                history_config,
                history_vaccine_scenario=baseline_vaccine_name,
                history_resistance_scenario=resistance_name,
            )
            scenarios.append(
                {
                    "config": policy_config,
                    "analysis": "prospective_vaccine_mechanism",
                    "scenario": name,
                    "vaccine_scenario": name,
                    "resistance_scenario": resistance_name,
                    "metadata": {
                        "country": country,
                        "history_vaccine_scenario": baseline_vaccine_name,
                        "vaccine_transition_design": VACCINE_TRANSITION_DESIGN,
                        "vaccine_transition_interpretation": "mechanism_target_not_cohort_rollout",
                    },
                }
            )
    return scenarios


def main():
    configs = load_configs()
    scenarios = build_scenarios(configs)
    baseline_vaccine_name = str(
        configs["baseline"].get("baseline_vaccine_scenario", "symptom_protective")
    )
    return run_scenario_list(
        scenarios,
        stem="vaccine_scenarios",
        reference_scenario=baseline_vaccine_name,
    )


if __name__ == "__main__":
    main()
