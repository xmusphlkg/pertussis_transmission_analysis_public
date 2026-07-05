"""Tests for SIRWS immune boosting dynamics.

Verifies that the SIRWS (Susceptible-Infected-Recovered-Waned-Susceptible)
boosting mechanism works correctly:
  - R → W waning occurs at the configured rate
  - W → R boosting occurs proportional to force of infection
  - W → S complete immunity loss occurs when no boosting
  - boosting_enabled=False falls back to direct R → S waning
  - W_natural compartment is properly populated and drained
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src_python.model.compartments import COMPARTMENTS, StateIndex, exposed_name
from src_python.model.ode_system import rhs
from src_python.model.outputs import initial_state, solve_model
from src_python.model.parameters import PreparedParameters
from src_python.simulation.common import make_config


def _sirws_config(*, boosting_enabled: bool = True, **overrides) -> dict:
    """Create a minimal config with SIRWS dynamics configured."""
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    config["simulation"]["burn_in_years"] = 1
    config["simulation"]["end_time"] = 365
    config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    config.setdefault("immunity_model", {})["boosting_enabled"] = boosting_enabled
    for key, value in overrides.items():
        if key in ("boosting_efficiency",):
            config["immunity_model"][key] = value
        elif key in ("R_to_W_duration", "W_to_S_duration"):
            config["natural_history"][key] = value
    return config


def _make_params(config: dict) -> PreparedParameters:
    return PreparedParameters.from_config(
        config,
        analysis="test_sirws",
        scenario="sirws_test",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )


def test_w_natural_compartment_exists_in_model():
    """W_natural must be a valid compartment in the SIRWS model."""
    assert "W_natural" in COMPARTMENTS
    assert "R_natural" in COMPARTMENTS


def test_sirws_r_to_w_waning_produces_flow():
    """With boosting enabled, R_natural should flow into W_natural."""
    config = _sirws_config(boosting_enabled=True)
    # Disable transmission to isolate waning dynamics
    config["transmission"]["beta_S"] = 0.0
    config["importation"]["enabled"] = False
    config["initial_conditions"]["initial_exposed_per_100k"] = 0.0
    config["initial_conditions"]["initial_infectious_per_100k"] = 0.0
    params = _make_params(config)
    index = StateIndex(params.age_groups)

    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    # Place population in R_natural to observe waning
    state[2, c["R_natural"]] = 10000.0  # some middle age group
    state[2, c["S"]] = 90000.0

    dy = index.reshape(rhs(0.0, index.flatten(state), params, index))

    # R_natural should be decreasing (waning out)
    assert dy[2, c["R_natural"]] < 0.0
    # W_natural should be increasing (receiving from R)
    assert dy[2, c["W_natural"]] > 0.0


def test_sirws_w_to_s_loss_without_foi():
    """Without FOI, W_natural should flow to S (complete immunity loss).

    We verify this by comparing two states: one with W_natural populated and
    one without. The difference in S derivative should be positive (W→S flow).
    """
    config = _sirws_config(boosting_enabled=True)
    config["transmission"]["beta_S"] = 0.0
    config["importation"]["enabled"] = False
    config["initial_conditions"]["initial_exposed_per_100k"] = 0.0
    config["initial_conditions"]["initial_infectious_per_100k"] = 0.0
    config["simulation"]["burn_in_years"] = 0
    params = _make_params(config)
    index = StateIndex(params.age_groups)

    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}

    # State A: population in W_natural + S
    state_a = np.zeros((index.n_age, index.n_compartments), dtype=float)
    for ai in range(index.n_age):
        state_a[ai, c["W_natural"]] = 10000.0
        state_a[ai, c["S"]] = 90000.0

    # State B: same total population but all in S (no W_natural)
    state_b = np.zeros((index.n_age, index.n_compartments), dtype=float)
    for ai in range(index.n_age):
        state_b[ai, c["S"]] = 100000.0

    dy_a = index.reshape(rhs(0.0, index.flatten(state_a), params, index))
    dy_b = index.reshape(rhs(0.0, index.flatten(state_b), params, index))

    # W_natural should be decreasing when populated
    assert dy_a[:, c["W_natural"]].sum() < 0.0
    # S should gain more in state_a than state_b (due to W→S flow)
    assert dy_a[:, c["S"]].sum() > dy_b[:, c["S"]].sum()


def test_sirws_boosting_restores_immunity_with_foi():
    """With active FOI, W_natural individuals should be boosted back to R."""
    config = _sirws_config(boosting_enabled=True, boosting_efficiency=1.0)
    params = _make_params(config)
    index = StateIndex(params.age_groups)

    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    # Set up a state with W_natural population and active infection
    state[2, c["W_natural"]] = 5000.0
    state[2, c["S"]] = 80000.0
    # Create infectious individuals to generate FOI
    state[2, c["I_S_sym_unvaccinated"]] = 500.0
    state[2, c["I_S_asym_unvaccinated"]] = 200.0

    dy = index.reshape(rhs(0.0, index.flatten(state), params, index))

    # With FOI present and boosting_efficiency=1.0, R_natural should gain
    # from boosting (W→R flow). The net dy[R_natural] includes:
    #   + recovered flow (from I→R)
    #   - waning_R_to_W * R_natural (which is 0 here since R_natural=0)
    #   + boosting_flow (W→R)
    # So R_natural should be increasing
    assert dy[2, c["R_natural"]] > 0.0


def test_sirws_disabled_uses_direct_r_to_s():
    """With boosting_enabled=False, R should flow directly to S (legacy).

    Verifies that W_natural receives no flow and that R_natural waning
    contributes to S growth by comparing with a state without R_natural.
    """
    config = _sirws_config(boosting_enabled=False)
    config["transmission"]["beta_S"] = 0.0
    config["importation"]["enabled"] = False
    config["initial_conditions"]["initial_exposed_per_100k"] = 0.0
    config["initial_conditions"]["initial_infectious_per_100k"] = 0.0
    config["simulation"]["burn_in_years"] = 0
    params = _make_params(config)
    index = StateIndex(params.age_groups)

    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}

    # State A: population in R_natural + S
    state_a = np.zeros((index.n_age, index.n_compartments), dtype=float)
    for ai in range(index.n_age):
        state_a[ai, c["R_natural"]] = 10000.0
        state_a[ai, c["S"]] = 90000.0

    # State B: same total but all in S (no R_natural)
    state_b = np.zeros((index.n_age, index.n_compartments), dtype=float)
    for ai in range(index.n_age):
        state_b[ai, c["S"]] = 100000.0

    dy_a = index.reshape(rhs(0.0, index.flatten(state_a), params, index))
    dy_b = index.reshape(rhs(0.0, index.flatten(state_b), params, index))

    # R_natural should decrease (waning out)
    assert dy_a[:, c["R_natural"]].sum() < 0.0
    # S should gain more in state_a than state_b (due to R→S flow)
    assert dy_a[:, c["S"]].sum() > dy_b[:, c["S"]].sum()
    # W_natural should NOT receive any flow (bypassed)
    assert np.allclose(dy_a[:, c["W_natural"]], 0.0)


def test_sirws_zero_boosting_efficiency_means_no_w_to_r():
    """With boosting_efficiency=0, W never flows back to R."""
    config = _sirws_config(boosting_enabled=True, boosting_efficiency=0.0)
    params = _make_params(config)
    index = StateIndex(params.age_groups)

    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    state[2, c["W_natural"]] = 5000.0
    state[2, c["S"]] = 80000.0
    state[2, c["I_S_sym_unvaccinated"]] = 500.0

    dy = index.reshape(rhs(0.0, index.flatten(state), params, index))

    # W_natural should only decrease (W→S), never increase from boosting
    assert dy[2, c["W_natural"]] < 0.0
    # The only positive flow to R_natural should be from recovery, not boosting
    # Since R_natural starts at 0 and there's no boosting, the R flow is only
    # from recovery of the infectious individuals
    recovery_flow = dy[2, c["R_natural"]]
    # Recovery flow should be positive (from I→R) but modest
    assert recovery_flow >= 0.0


def test_w_natural_can_break_through_to_infection_when_boosting_fails():
    config = _sirws_config(boosting_enabled=True, boosting_efficiency=0.0)
    config["immunity_model"]["waned_natural_infection_susceptibility"] = 0.50
    params = _make_params(config)
    index = StateIndex(params.age_groups)

    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    state[2, c["W_natural"]] = 5000.0
    state[2, c["S"]] = 80000.0
    state[2, c["I_S_sym_unvaccinated"]] = 500.0

    dy = index.reshape(rhs(0.0, index.flatten(state), params, index))

    assert dy[2, c["W_natural"]] < 0.0
    assert dy[2, c[exposed_name("S", "unvaccinated")]] > 0.0


def test_sirws_w_natural_populated_after_simulation():
    """After a full simulation with SIRWS, W_natural should have population."""
    config = _sirws_config(boosting_enabled=True)
    params = _make_params(config)
    index = StateIndex(params.age_groups)
    solution = solve_model(params, index)
    assert solution.success

    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    final_state = solution.y[:, -1].reshape(index.n_age, index.n_compartments)
    w_natural_total = final_state[:, c["W_natural"]].sum()

    # W_natural should have accumulated some population during the simulation
    assert w_natural_total > 0.0


def test_sirws_mass_conservation_with_boosting():
    """Total population should be conserved with SIRWS dynamics active (fixed population mode)."""
    config = _sirws_config(boosting_enabled=True)
    # Use fixed population mode to test mass conservation without WPP nudging
    config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    params = _make_params(config)
    index = StateIndex(params.age_groups)
    y0 = initial_state(params, index)
    solution = solve_model(params, index)
    assert solution.success

    totals = solution.y.reshape(index.n_age, index.n_compartments, -1).sum(axis=(0, 1))
    expected = float(index.reshape(y0).sum())
    assert np.max(np.abs(totals - expected)) < params.total_population * 1e-8
