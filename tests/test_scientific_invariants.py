"""Scientific invariant and prospective-policy interface tests.

These tests deliberately exercise conservation at the flow boundary instead of
only checking a successful integration.  They also define the minimal public
contract needed to make numerical pathologies and prospective policy timing
auditable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy

import numpy as np
import pytest

from src_python.model import outputs
from src_python.model.compartments import (
    COMPARTMENTS,
    VACCINE_ORIGINS,
    StateIndex,
    exposed_name,
    susceptible_name,
    treated_name,
)
from src_python.model.ode_system import _add_importation
from src_python.model.parameters import PreparedParameters
from src_python.simulation import common as simulation_common
from src_python.simulation.common import make_config
from src_python.simulation.common import load_configs, publication_country_names
from src_python.simulation.run_reporting_scenarios import _apply_reporting_scenario


def _prepared_params(
    *,
    scenario: str,
    importation_rate: float = 0.0,
    resistant_fraction: float = 0.3,
) -> PreparedParameters:
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        load_calibration=False,
    )
    config["simulation"].update(
        {
            "start_time": 0.0,
            "end_time": 4.0,
            "output_time_step": 1.0,
            "burn_in_years": 0.02,
        }
    )
    config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    config["routine_vaccination"]["enabled"] = False
    config["importation"].update(
        {
            "enabled": importation_rate > 0.0,
            "rate_per_100k_per_year": importation_rate,
            "resistant_fraction": resistant_fraction,
            "age_distribution": {
                record["label"]: float(i == 0)
                for i, record in enumerate(config["age_groups"])
            },
        }
    )
    return PreparedParameters.from_config(
        config,
        analysis="scientific_invariant_test",
        scenario=scenario,
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )


def test_policy_horizon_never_exceeds_embedded_demographic_support() -> None:
    configs = load_configs()
    analysis_end_year = int(
        str(configs["baseline"]["calendar"]["analysis_end_date"])[:4]
    )
    history_start_year = int(
        str(configs["baseline"]["simulation"]["history_start_date"])[:4]
    )

    for country in publication_country_names(configs):
        config = make_config(country_profile=country, load_calibration=False)
        trajectory = config["demography"]["wpp_trajectory"]
        assert min(int(year) for year in trajectory["years"]) <= history_start_year, country
        assert analysis_end_year <= int(trajectory["horizon_end_year"]), country
        assert analysis_end_year <= max(int(year) for year in trajectory["years"]), country
        assert analysis_end_year <= max(
            int(year) for year in trajectory["births_by_year"]
        ), country


def _required_output_callable(name: str) -> Callable:
    function = getattr(outputs, name, None)
    assert callable(function), (
        f"src_python.model.outputs.{name} is required by the scientific "
        "invariant/prospective-policy contract"
    )
    return function


@pytest.mark.parametrize("supply_limited", [False, True])
def test_importation_moves_exactly_the_amount_removed_from_susceptible_pools(
    supply_limited: bool,
) -> None:
    """Importation is an S->E transfer, never an external population birth.

    The supply-limited case is the important regression: requested flow can be
    larger than the source pool, so every destination flow must use the actual
    capped removal rather than the uncapped request.
    """

    rate = 1.0e6 if supply_limited else 1.0e-3
    resistant_fraction = 0.37
    params = _prepared_params(
        scenario=f"importation_{'limited' if supply_limited else 'available'}",
        importation_rate=rate,
        resistant_fraction=resistant_fraction,
    )
    index = StateIndex(params.age_groups)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)

    source_amount = 0.125 if supply_limited else 1.0e6
    for origin_idx, origin in enumerate(VACCINE_ORIGINS, start=1):
        state[0, c[susceptible_name(origin)]] = source_amount * origin_idx

    comp = {name: state[:, i] for i, name in enumerate(COMPARTMENTS)}
    dy = np.zeros_like(state)
    _add_importation(dy, comp, params, c)

    removed_by_origin = np.array(
        [-dy[:, c[susceptible_name(origin)]].sum() for origin in VACCINE_ORIGINS]
    )
    sensitive_added_by_origin = np.array(
        [dy[:, c[exposed_name("S", origin)]].sum() for origin in VACCINE_ORIGINS]
    )
    resistant_added_by_origin = np.array(
        [dy[:, c[exposed_name("R", origin)]].sum() for origin in VACCINE_ORIGINS]
    )
    total_added_by_origin = sensitive_added_by_origin + resistant_added_by_origin

    np.testing.assert_allclose(total_added_by_origin, removed_by_origin, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        resistant_added_by_origin,
        resistant_fraction * removed_by_origin,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitive_added_by_origin,
        (1.0 - resistant_fraction) * removed_by_origin,
        rtol=1e-12,
        atol=1e-12,
    )
    assert float(dy.sum()) == pytest.approx(0.0, abs=1e-12)

    if supply_limited:
        available_by_origin = np.array(
            [state[:, c[susceptible_name(origin)]].sum() for origin in VACCINE_ORIGINS]
        )
        np.testing.assert_allclose(removed_by_origin, available_by_origin, rtol=0.0, atol=1e-12)


def test_history_state_api_returns_equal_but_independent_writable_states() -> None:
    """Repeated history preparation must not expose shared mutable state."""

    prepare_history_state = _required_output_callable("prepare_history_state")
    history_params = _prepared_params(scenario="shared_history")
    index = StateIndex(history_params.age_groups)

    first = prepare_history_state(history_params, index)
    second = prepare_history_state(history_params, index)

    assert isinstance(first, np.ndarray)
    assert first.shape == (index.size,)
    assert first.dtype.kind == "f"
    assert first.flags.writeable
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)

    first[0] += 1.0
    assert first[0] != second[0]


def test_prospective_policies_branch_from_the_identical_t0_state_without_mutation() -> None:
    """Policy parameters may differ after t0, never during historical burn-in."""

    prepare_history_state = _required_output_callable("prepare_history_state")
    simulate_policy_from_state = _required_output_callable("simulate_policy_from_state")

    history_params = _prepared_params(scenario="historical_current_practice")
    index = StateIndex(history_params.age_groups)
    shared_t0 = prepare_history_state(history_params, index)
    shared_t0_before = shared_t0.copy()

    policy_a_config = deepcopy(history_params.raw)
    policy_b_config = deepcopy(history_params.raw)
    policy_a_config["simulation"]["burn_in_years"] = 99.0
    policy_b_config["simulation"]["burn_in_years"] = 0.0
    policy_a_config["transmission"]["beta_S"] *= 0.5
    policy_b_config["transmission"]["beta_S"] *= 1.5
    policy_a = PreparedParameters.from_config(
        policy_a_config,
        analysis="scientific_invariant_test",
        scenario="prospective_policy_a",
    )
    policy_b = PreparedParameters.from_config(
        policy_b_config,
        analysis="scientific_invariant_test",
        scenario="prospective_policy_b",
    )

    solution_a = simulate_policy_from_state(policy_a, index, shared_t0)
    solution_b = simulate_policy_from_state(policy_b, index, shared_t0)

    assert solution_a.success and solution_b.success
    np.testing.assert_array_equal(shared_t0, shared_t0_before)
    np.testing.assert_array_equal(solution_a.y[:, 0], shared_t0_before)
    np.testing.assert_array_equal(solution_b.y[:, 0], shared_t0_before)
    np.testing.assert_array_equal(solution_a.y[:, 0], solution_b.y[:, 0])
    assert not np.shares_memory(solution_a.y, shared_t0)
    assert not np.shares_memory(solution_b.y, shared_t0)
    assert not np.shares_memory(solution_a.y, solution_b.y)


def test_state_invariant_diagnostics_expose_negative_mass_and_population_drift() -> None:
    """Numerical clipping must not make raw pathologies invisible to QA."""

    state_invariant_diagnostics = _required_output_callable("state_invariant_diagnostics")
    params = _prepared_params(scenario="diagnostic_fixture")
    index = StateIndex(params.age_groups)
    initial = outputs.initial_state(params, index)
    states = np.repeat(initial[:, None], 3, axis=1)

    empty_compartment = index.index(0, treated_name("R", "unvaccinated"))
    states[empty_compartment, 1] = -0.25
    states[index.index(0, "S"), 2] += 5.0

    diagnostics = state_invariant_diagnostics(states, index)
    assert isinstance(diagnostics, Mapping)
    required = {
        "min_state",
        "negative_state_count",
        "negative_state_mass",
        "initial_population",
        "final_population",
        "max_abs_population_drift",
        "max_rel_population_drift",
    }
    assert required.issubset(diagnostics)
    assert diagnostics["min_state"] == pytest.approx(-0.25)
    assert diagnostics["negative_state_count"] == 1
    assert diagnostics["negative_state_mass"] == pytest.approx(0.25)
    assert diagnostics["initial_population"] == pytest.approx(float(initial.sum()))
    assert diagnostics["final_population"] == pytest.approx(float(initial.sum()) + 5.0)
    assert diagnostics["max_abs_population_drift"] == pytest.approx(5.0)
    assert diagnostics["max_rel_population_drift"] == pytest.approx(5.0 / float(initial.sum()))


def test_solve_model_attaches_recomputable_state_invariant_diagnostics() -> None:
    """Every model result carries inspectable invariants, not only test runs."""

    state_invariant_diagnostics = _required_output_callable("state_invariant_diagnostics")
    params = _prepared_params(scenario="observable_solution")
    index = StateIndex(params.age_groups)

    solution = outputs.solve_model(params, index)

    assert solution.success
    assert isinstance(getattr(solution, "diagnostics", None), Mapping)
    recomputed = state_invariant_diagnostics(solution.y, index)
    for key in (
        "min_state",
        "negative_state_count",
        "negative_state_mass",
        "initial_population",
        "final_population",
        "max_abs_population_drift",
        "max_rel_population_drift",
    ):
        assert solution.diagnostics[key] == pytest.approx(recomputed[key])


def test_prospective_scenario_bundle_prepares_one_shared_history(monkeypatch) -> None:
    """Two policies with the same past incur one burn-in and receive copies."""

    current, current_vaccine = simulation_common.make_intervention_config(
        "current", country_profile="China"
    )
    booster, booster_vaccine = simulation_common.make_intervention_config(
        "adolescent_booster", country_profile="China"
    )
    for config in (current, booster):
        config["simulation"].update(
            {"start_time": 0.0, "end_time": 2.0, "output_time_step": 1.0}
        )
        history = config[simulation_common.PROSPECTIVE_POLICY_KEY]["history_config"]
        history["simulation"].update(
            {
                "start_time": 0.0,
                "end_time": 2.0,
                "output_time_step": 1.0,
                "burn_in_years": 0.01,
            }
        )

    calls = 0
    original = simulation_common.prepare_history_state

    def counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation_common, "prepare_history_state", counted_prepare)
    scenarios = [
        {
            "config": current,
            "analysis": "policy_bundle_test",
            "scenario": "current",
            "vaccine_scenario": current_vaccine,
            "metadata": {"country": "China"},
        },
        {
            "config": booster,
            "analysis": "policy_bundle_test",
            "scenario": "adolescent_booster",
            "vaccine_scenario": booster_vaccine,
            "metadata": {"country": "China"},
        },
    ]

    prepared = simulation_common._prepare_prospective_scenario_items(
        scenarios,
        stem="policy_bundle_test",
        n_jobs=1,
    )

    assert calls == 1
    first = prepared[0]["initial_state_override"]
    second = prepared[1]["initial_state_override"]
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)


def test_age_reporting_scenario_cannot_change_diagnosis_or_treatment_dynamics() -> None:
    """Observation sensitivity remains outside the transmission model."""

    base = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        load_calibration=False,
    )
    changed = _apply_reporting_scenario(
        base,
        {
            "multiplier": 1.0,
            "age_multipliers": {
                record["label"]: 0.25 for record in base["age_groups"]
            },
        },
    )
    before = PreparedParameters.from_config(base, analysis="test", scenario="before")
    after = PreparedParameters.from_config(changed, analysis="test", scenario="after")

    np.testing.assert_array_equal(before.diagnosis_probability, after.diagnosis_probability)
    assert np.any(before.reporting_rate != after.reporting_rate)
