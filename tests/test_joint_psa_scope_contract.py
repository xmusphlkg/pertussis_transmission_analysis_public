from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src_python.simulation import run_joint_psa_rank_acceptability as joint_runner
from src_python.simulation.common import (
    PROSPECTIVE_POLICY_KEY,
    _history_config_key,
    load_configs,
    make_intervention_config,
)
from src_python.simulation.resistance_management_uptake import (
    RESISTANCE_MANAGEMENT_PARAMETER_NAMES,
    apply_resistance_management_sample,
)
from src_python.simulation.run_joint_psa_rank_acceptability import (
    EXPECTED_PARAMETER_NAMES,
    FIGURE2B_PARAMETER_NAMES,
    OBSERVATION_ONLY,
    PARAMETER_CONSUMERS,
    PARAMETER_TIME_SCOPES,
    PROSPECTIVE_IMPLEMENTATION,
    SELECTED_STRATEGIES,
    STRUCTURAL_ALL_TIME,
    _apply_psa_sample,
    _build_scenarios_for_sample,
    _make_strategy_config,
    _model_config_targets_for_scope,
    _sample_table,
)


def _figure2b_sample() -> dict[str, float]:
    return {
        "psa_sample_id": 7,
        "infant_contact_multiplier": 1.3,
        "VE_inf_baseline": 0.4,
        "relative_infectiousness_asymptomatic": 0.73,
        "infectious_duration_asymptomatic": 23.0,
        "fitness_R": 1.17,
        "PEP_coverage_multiplier": 1.4,
    }


def _figure2b_specs() -> dict[str, dict[str, float | str]]:
    return {
        name: {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.9,
            "time_scope": PARAMETER_TIME_SCOPES[name],
            "consumers": list(PARAMETER_CONSUMERS[name]),
        }
        for name in FIGURE2B_PARAMETER_NAMES
    }


def test_structural_draws_update_policy_and_attached_history() -> None:
    config, _ = _make_strategy_config("targeted_pep_high_risk", country="China")
    current, _ = make_intervention_config("current", country_profile="China")
    before = deepcopy(config)

    updated = _apply_psa_sample(
        config,
        current,
        strategy="targeted_pep_high_risk",
        sample=_figure2b_sample(),
        parameter_specs=_figure2b_specs(),
    )

    history = updated[PROSPECTIVE_POLICY_KEY]["history_config"]
    assert updated["transmission"]["relative_infectiousness_asymptomatic"] == 0.73
    assert history["transmission"]["relative_infectiousness_asymptomatic"] == 0.73
    assert updated["natural_history"]["infectious_duration_asymptomatic"] == 23.0
    assert history["natural_history"]["infectious_duration_asymptomatic"] == 23.0
    assert updated["transmission"]["fitness_R"] == 1.17
    assert history["transmission"]["fitness_R"] == 1.17

    labels = [row["label"] for row in updated["age_groups"]]
    target = labels.index("infant_0_2m")
    source = labels.index("young_adult_18_39y")
    before_policy_contact = before["contact_matrix"]["rows"][target][source]
    before_history_contact = before[PROSPECTIVE_POLICY_KEY]["history_config"]["contact_matrix"][
        "rows"
    ][target][source]
    assert np.isclose(
        updated["contact_matrix"]["rows"][target][source],
        1.3 * before_policy_contact,
    )
    assert np.isclose(
        history["contact_matrix"]["rows"][target][source],
        1.3 * before_history_contact,
    )
    assert updated["vaccine"]["VE_inf"] != before["vaccine"]["VE_inf"]
    assert history["vaccine"]["VE_inf"] != before[PROSPECTIVE_POLICY_KEY]["history_config"][
        "vaccine"
    ]["VE_inf"]


def test_implementation_draws_update_policy_only() -> None:
    config, _ = _make_strategy_config("targeted_pep_high_risk", country="China")
    current, _ = make_intervention_config("current", country_profile="China")
    history_pep_before = deepcopy(config[PROSPECTIVE_POLICY_KEY]["history_config"]["PEP"])
    policy_coverage_before = float(config["PEP"]["coverage_household_contacts"])

    updated = _apply_psa_sample(
        config,
        current,
        strategy="targeted_pep_high_risk",
        sample=_figure2b_sample(),
        parameter_specs=_figure2b_specs(),
    )

    assert np.isclose(
        updated["PEP"]["coverage_household_contacts"],
        min(1.0, 1.4 * policy_coverage_before),
    )
    assert updated[PROSPECTIVE_POLICY_KEY]["history_config"]["PEP"] == history_pep_before

    guided, _ = _make_strategy_config("resistance_guided_treatment", country="China")
    guided_history_before = deepcopy(guided[PROSPECTIVE_POLICY_KEY]["history_config"])
    resistance_specs = {
        "resistance_management_uptake": {
            "distribution": "uniform",
            "min": 0.4,
            "max": 1.0,
            "time_scope": PROSPECTIVE_IMPLEMENTATION,
        },
        "resistance_management_pep_reach_multiplier": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
            "time_scope": PROSPECTIVE_IMPLEMENTATION,
        },
    }
    resistance_updated = apply_resistance_management_sample(
        guided,
        current,
        strategy="resistance_guided_treatment",
        sample={
            "psa_sample_id": 3,
            "resistance_management_uptake": 0.5,
            "resistance_management_pep_reach_multiplier": 0.5,
        },
        pep_restored=True,
        parameter_specs=resistance_specs,
    )
    assert resistance_updated[PROSPECTIVE_POLICY_KEY]["history_config"] == guided_history_before
    assert resistance_updated["treatment"]["resistant"] != guided["treatment"]["resistant"]
    assert resistance_updated["PEP"]["coverage_household_contacts"] == pytest.approx(
        0.5 * current["PEP"]["coverage_household_contacts"]
    )


def test_observation_only_scope_has_no_prospective_model_target() -> None:
    config, _ = _make_strategy_config("current", country="China")

    assert _model_config_targets_for_scope(config, OBSERVATION_ONLY) == ()
    assert len(_model_config_targets_for_scope(config, PROSPECTIVE_IMPLEMENTATION)) == 1
    assert len(_model_config_targets_for_scope(config, STRUCTURAL_ALL_TIME)) == 2


def test_figure2b_contract_rejects_inactive_resistance_uptake_dimension() -> None:
    assert EXPECTED_PARAMETER_NAMES == FIGURE2B_PARAMETER_NAMES
    assert len(FIGURE2B_PARAMETER_NAMES) == 6
    assert "resistance_management_uptake" not in FIGURE2B_PARAMETER_NAMES
    assert RESISTANCE_MANAGEMENT_PARAMETER_NAMES == (
        "resistance_management_uptake",
        "resistance_management_pep_reach_multiplier",
    )

    samples = _sample_table(8, 9, _figure2b_specs())
    assert "resistance_management_uptake" not in samples.columns

    extra_specs = {
        **_figure2b_specs(),
        "resistance_management_uptake": {
            "distribution": "uniform",
            "min": 0.4,
            "max": 1.0,
            "time_scope": PROSPECTIVE_IMPLEMENTATION,
        },
    }
    with pytest.raises(ValueError, match="semantic contract"):
        _sample_table(8, 9, extra_specs)

    config, _ = _make_strategy_config("targeted_pep_high_risk", country="China")
    current, _ = make_intervention_config("current", country_profile="China")
    with pytest.raises(ValueError, match="inactive for all Figure 2b"):
        _apply_psa_sample(
            config,
            current,
            strategy="targeted_pep_high_risk",
            sample={**_figure2b_sample(), "resistance_management_uptake": 0.7},
            parameter_specs=_figure2b_specs(),
        )


def test_all_secondary_strategies_share_the_same_sampled_2027_history() -> None:
    scenarios = _build_scenarios_for_sample(
        load_configs(),
        _figure2b_sample(),
        countries=("China",),
        strategies=SELECTED_STRATEGIES,
        smoke_runtime=False,
        parameter_specs=_figure2b_specs(),
    )

    assert len(scenarios) == len(SELECTED_STRATEGIES) == 12
    histories = {
        item["scenario"]: item["config"][PROSPECTIVE_POLICY_KEY]["history_config"]
        for item in scenarios
    }
    assert set(histories) == set(SELECTED_STRATEGIES)
    history_keys = {_history_config_key(history) for history in histories.values()}
    assert len(history_keys) == 1
    assert "transmission_blocking_vaccine" in histories


def test_registry_time_scope_mismatch_fails_closed() -> None:
    specs = _figure2b_specs()
    specs["fitness_R"]["time_scope"] = PROSPECTIVE_IMPLEMENTATION

    with pytest.raises(ValueError, match="implemented scope"):
        _sample_table(4, 1, specs)


def test_registry_consumer_mismatch_and_missing_registry_fail_closed(
    monkeypatch,
) -> None:
    specs = _figure2b_specs()
    specs["fitness_R"]["consumers"] = [
        "src_python.simulation.run_joint_psa_rank_acceptability",
        "src_python.simulation.run_fitness_grid",
    ]
    with pytest.raises(ValueError, match="implemented consumers"):
        _sample_table(4, 1, specs)

    monkeypatch.setattr(
        joint_runner,
        "load_configs",
        lambda: {"parameter_distributions": {"schema_version": 1}},
    )
    with pytest.raises(ValueError, match="legacy uniform-range fallback is disabled"):
        joint_runner._default_parameter_specs()


@pytest.mark.parametrize("bad_version", [None, True, "1", 1.0, 0, 2])
def test_figure2b_registry_schema_fails_closed(monkeypatch, bad_version) -> None:
    configs = load_configs()
    registry = configs["parameter_distributions"]
    if bad_version is None:
        registry.pop("schema_version")
    else:
        registry["schema_version"] = bad_version
    monkeypatch.setattr(joint_runner, "load_configs", lambda: configs)

    with pytest.raises(ValueError, match="schema_version"):
        joint_runner._default_parameter_specs()


def test_every_figure2b_dimension_changes_its_declared_target() -> None:
    config, _ = _make_strategy_config("targeted_pep_high_risk", country="China")
    current, _ = make_intervention_config("current", country_profile="China")
    labels = [row["label"] for row in config["age_groups"]]
    infant_index = labels.index("infant_0_2m")
    adult_index = labels.index("young_adult_18_39y")

    neutral = {
        "psa_sample_id": 1,
        "infant_contact_multiplier": 1.0,
        "VE_inf_baseline": 0.25,
        "relative_infectiousness_asymptomatic": 0.50,
        "infectious_duration_asymptomatic": 17.3,
        "fitness_R": 1.0,
        "PEP_coverage_multiplier": 1.0,
    }
    contrasts = {
        "infant_contact_multiplier": (0.8, 1.2),
        "VE_inf_baseline": (0.15, 0.45),
        "relative_infectiousness_asymptomatic": (0.20, 0.80),
        "infectious_duration_asymptomatic": (10.0, 25.0),
        "fitness_R": (0.8, 1.2),
        "PEP_coverage_multiplier": (0.7, 1.3),
    }

    def target_value(updated: dict, name: str, *, history: bool) -> float:
        phase = (
            updated[PROSPECTIVE_POLICY_KEY]["history_config"]
            if history
            else updated
        )
        if name == "infant_contact_multiplier":
            return float(phase["contact_matrix"]["rows"][infant_index][adult_index])
        if name == "VE_inf_baseline":
            return float(phase["vaccine"]["VE_inf"])
        if name == "relative_infectiousness_asymptomatic":
            return float(phase["transmission"][name])
        if name == "infectious_duration_asymptomatic":
            return float(phase["natural_history"][name])
        if name == "fitness_R":
            return float(phase["transmission"][name])
        if name == "PEP_coverage_multiplier":
            return float(phase["PEP"]["coverage_household_contacts"])
        raise AssertionError(name)

    for name, (low_value, high_value) in contrasts.items():
        low = _apply_psa_sample(
            config,
            current,
            strategy="targeted_pep_high_risk",
            sample={**neutral, name: low_value},
            parameter_specs=_figure2b_specs(),
        )
        high = _apply_psa_sample(
            config,
            current,
            strategy="targeted_pep_high_risk",
            sample={**neutral, name: high_value},
            parameter_specs=_figure2b_specs(),
        )
        assert target_value(low, name, history=False) != target_value(
            high, name, history=False
        )
        if PARAMETER_TIME_SCOPES[name] == STRUCTURAL_ALL_TIME:
            assert target_value(low, name, history=True) != target_value(
                high, name, history=True
            )
        else:
            assert target_value(low, name, history=True) == target_value(
                high, name, history=True
            )
