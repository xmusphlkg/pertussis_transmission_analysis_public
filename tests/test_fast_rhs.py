from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src_python.model.compartments import StateIndex
from src_python.model.fast_rhs import (
    FastRHSUnsupportedError,
    NUMBA_FAST_RHS_AVAILABLE,
    _continuous_npi_multiplier,
    _continuous_routine_delivery_multiplier,
    benchmark_fast_rhs,
    build_fast_rhs,
)
from src_python.model.force_of_infection import _npi_contact_reduction_at
from src_python.model.ode_system import _routine_delivery_multiplier_at, rhs
from src_python.model.outputs import initial_state
from src_python.model.parameters import PreparedParameters
from src_python.simulation.common import make_config


pytestmark = pytest.mark.skipif(
    not NUMBA_FAST_RHS_AVAILABLE,
    reason="the complete fast RHS requires Numba",
)


def _params(*, fixed_demography: bool = False) -> PreparedParameters:
    config = make_config(country_profile="Australia", load_calibration=False)
    config["simulation"]["burn_in_years"] = 2
    config["simulation"]["end_time"] = 730.0
    if fixed_demography:
        config["demography"]["mode"] = "fixed_population_profile"

    # Exercise every time-dependent and optional RHS branch in one numeric
    # equivalence fixture, including overlapping periods (whose precedence is
    # easy to accidentally change during optimization).
    config["transmission"]["multi_year_amplitude"] = 0.13
    config["transmission"]["npi_contact_reduction_periods"] = [
        {
            "start_date": "2025-02-01",
            "end_date": "2025-03-15",
            "reduction": 0.63,
            "ramp_days": 37,
        },
        {
            "start_date": "2025-03-01",
            "end_date": "2025-04-01",
            "reduction": 0.20,
            "ramp_days": 10,
        },
    ]
    config["routine_vaccination"]["delivery_shock_periods"] = [
        {
            "start_date": "2025-01-20",
            "end_date": "2025-02-10",
            "reduction": 0.40,
            "ramp_days": 20,
        },
        {
            "start_date": "2025-02-01",
            "end_date": "2025-02-15",
            "reduction": 0.70,
            "ramp_days": 10,
        },
    ]
    config["resistance"]["anchor_during_dynamics"] = True
    return PreparedParameters.from_config(config, analysis="test", scenario="fast_rhs")


def _dense_random_state(params: PreparedParameters, index: StateIndex, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.gamma(shape=1.4, scale=1.0, size=(index.n_age, index.n_compartments))
    state *= params.population[:, None] / state.sum(axis=1)[:, None]
    return state.reshape(index.size)


def _assert_rhs_equal(
    fast,
    params: PreparedParameters,
    index: StateIndex,
    t: float,
    state: np.ndarray,
) -> None:
    expected = rhs(t, state, params, index)
    actual = fast(t, state)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-9)


def test_fast_rhs_matches_reference_for_wpp_calendar_and_all_optional_flows():
    params = _params(fixed_demography=False)
    index = StateIndex(params.age_groups)
    fast = build_fast_rhs(params, index)
    states = [initial_state(params, index), _dense_random_state(params, index, 20260712)]

    # Negative/fractional dates, leap-year boundary, overlapping shock periods,
    # ramp periods, and WPP interpolation are all represented.
    times = (-500.4, -1.0, -0.2, 0.0, 19.9, 31.0, 45.25, 74.0, 365.0, 729.9)
    for state in states:
        for t in times:
            _assert_rhs_equal(fast, params, index, t, state)


def test_fast_rhs_matches_fixed_profile_demography_and_conserves_population():
    params = _params(fixed_demography=True)
    index = StateIndex(params.age_groups)
    fast = build_fast_rhs(params, index)
    state = _dense_random_state(params, index, 9)

    for t in (-400.5, 0.0, 44.75, 400.25):
        _assert_rhs_equal(fast, params, index, t, state)
        # Infection, treatment, vaccination, importation, anchoring and fixed
        # demographic turnover are all transfers, so total derivative is zero.
        assert abs(float(fast(t, state).sum())) < 1e-6


def test_fast_rhs_matches_noncalendar_legacy_and_disabled_branches():
    config = make_config(load_calibration=False)
    config["calendar"] = {"enabled": False}
    config["simulation"].update({"start_time": 5.0, "end_time": 100.0, "burn_in_years": 0.0})
    config["demography"]["enabled"] = False
    config["routine_vaccination"]["enabled"] = False
    config["importation"]["enabled"] = False
    config["immunity_model"]["boosting_enabled"] = False
    config["transmission"]["multi_year_period_years"] = 0.0
    config["resistance"]["anchor_during_dynamics"] = False
    params = PreparedParameters.from_config(config, analysis="test", scenario="fast_rhs_no_calendar")
    index = StateIndex(params.age_groups)
    fast = build_fast_rhs(params, index)
    state = _dense_random_state(params, index, 14)
    # The reference clamps negative solver-stage values before evaluating RHS.
    state[index.index(0, "V_dose1_recent")] = -10.0

    for t in (-100.2, 0.25, 365.2, 800.0):
        _assert_rhs_equal(fast, params, index, t, state)


def test_fast_rhs_preserves_fractional_time_in_isolated_recovery_ramps():
    config = make_config(country_profile="Australia", load_calibration=False)
    config["calendar"]["analysis_start_date"] = "2025-01-01"
    config["simulation"].update({"start_time": 0.0, "end_time": 40.0, "burn_in_years": 0.0})
    config["transmission"]["npi_contact_reduction_periods"] = [
        {
            "start_date": "2025-01-02",
            "end_date": "2025-01-04",
            "reduction": 0.60,
            "ramp_days": 8.0,
        }
    ]
    config["routine_vaccination"]["delivery_shock_periods"] = [
        {
            "start_date": "2025-01-02",
            "end_date": "2025-01-04",
            "reduction": 0.40,
            "ramp_days": 5.25,
        }
    ]
    params = PreparedParameters.from_config(config, analysis="test", scenario="fractional_ramp")
    index = StateIndex(params.age_groups)
    fast = build_fast_rhs(params, index)
    pack = fast.parameter_pack
    state = initial_state(params, index)

    # Jan-04 is inclusive, so the half-open recovery boundary is Jan-05 at
    # t=4.  Fractional points must follow the exact continuous ramp, not a daily
    # step or interpolation between precomputed multiplier samples.
    for t in (3.999999, 4.0, 4.125, 5.5, 9.249999, 9.25, 11.999999, 12.0):
        calendar_ordinal = params.calendar_ordinal_at(t)
        assert calendar_ordinal is not None
        assert _continuous_npi_multiplier(calendar_ordinal, pack.npi_periods) == pytest.approx(
            _npi_contact_reduction_at(t, params), abs=2e-15
        )
        assert _continuous_routine_delivery_multiplier(
            calendar_ordinal, pack.routine_delivery_periods
        ) == pytest.approx(_routine_delivery_multiplier_at(t, params), abs=2e-15)
        _assert_rhs_equal(fast, params, index, t, state)

    assert _continuous_npi_multiplier(
        float(params.calendar_ordinal_at(4.125)), pack.npi_periods
    ) == pytest.approx(0.40 + 0.60 * 0.125 / 8.0)
    assert _continuous_routine_delivery_multiplier(
        float(params.calendar_ordinal_at(5.5)), pack.routine_delivery_periods
    ) == pytest.approx(0.60 + 0.40 * 1.5 / 5.25)


def test_fast_rhs_uses_minimum_across_overlapping_fractional_ramps():
    config = make_config(country_profile="Australia", load_calibration=False)
    config["calendar"]["analysis_start_date"] = "2025-01-01"
    config["simulation"].update({"start_time": 0.0, "end_time": 40.0, "burn_in_years": 0.0})
    periods = [
        {
            "start_date": "2025-01-02",
            "end_date": "2025-01-04",
            "reduction": 0.50,
            "ramp_days": 10.0,
        },
        {
            "start_date": "2025-01-04",
            "end_date": "2025-01-06",
            "reduction": 0.80,
            "ramp_days": 8.0,
        },
    ]
    config["transmission"]["npi_contact_reduction_periods"] = periods
    config["routine_vaccination"]["delivery_shock_periods"] = periods
    params = PreparedParameters.from_config(config, analysis="test", scenario="overlap_ramps")
    index = StateIndex(params.age_groups)
    fast = build_fast_rhs(params, index)
    pack = fast.parameter_pack
    state = initial_state(params, index)

    # At t=7.25 both recovery tails are active.  The second is more restrictive:
    # min(0.5 + 0.5*3.25/10, 0.2 + 0.8*1.25/8) = 0.325.
    t = 7.25
    calendar_ordinal = float(params.calendar_ordinal_at(t))
    expected = min(0.50 + 0.50 * 3.25 / 10.0, 0.20 + 0.80 * 1.25 / 8.0)
    assert expected == pytest.approx(0.325)
    assert _continuous_npi_multiplier(calendar_ordinal, pack.npi_periods) == pytest.approx(expected)
    assert _continuous_routine_delivery_multiplier(
        calendar_ordinal, pack.routine_delivery_periods
    ) == pytest.approx(expected)
    assert _npi_contact_reduction_at(t, params) == pytest.approx(expected)
    assert _routine_delivery_multiplier_at(t, params) == pytest.approx(expected)
    _assert_rhs_equal(fast, params, index, t, state)

    # During the second active interval, NPI uses the active minimum; routine
    # delivery also evaluates the first tail but retains the same minimum here.
    for active_t in (3.5, 5.875):
        ordinal = float(params.calendar_ordinal_at(active_t))
        assert _continuous_npi_multiplier(ordinal, pack.npi_periods) == pytest.approx(
            _npi_contact_reduction_at(active_t, params), abs=2e-15
        )
        assert _continuous_routine_delivery_multiplier(
            ordinal, pack.routine_delivery_periods
        ) == pytest.approx(_routine_delivery_multiplier_at(active_t, params), abs=2e-15)
        _assert_rhs_equal(fast, params, index, active_t, state)


def test_fast_rhs_is_scipy_compatible_over_short_integration():
    params = _params(fixed_demography=True)
    index = StateIndex(params.age_groups)
    state = initial_state(params, index)
    fast = build_fast_rhs(params, index)

    reference_solution = solve_ivp(
        lambda t, y: rhs(t, y, params, index),
        (0.0, 12.0),
        state,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        t_eval=np.array([0.0, 3.0, 7.0, 12.0]),
    )
    fast_solution = solve_ivp(
        fast,
        (0.0, 12.0),
        state,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        t_eval=np.array([0.0, 3.0, 7.0, 12.0]),
    )

    assert reference_solution.success and fast_solution.success
    np.testing.assert_allclose(fast_solution.y, reference_solution.y, rtol=2e-10, atol=2e-5)


def test_fast_rhs_rejects_out_of_window_state_shape_and_missing_origin_cache():
    params = _params()
    index = StateIndex(params.age_groups)
    fast = build_fast_rhs(params, index)
    state = initial_state(params, index)

    with pytest.raises(FastRHSUnsupportedError, match="outside its configured time span"):
        fast(10_000.0, state)
    with pytest.raises(FastRHSUnsupportedError, match="State has"):
        fast(0.0, state[:-1])

    uncached = replace(params, origin_susceptibility=np.empty(0))
    with pytest.raises(FastRHSUnsupportedError, match="origin_susceptibility has shape"):
        build_fast_rhs(uncached, index)


def test_fast_rhs_rejects_unknown_routine_vaccine_origin():
    config = make_config(load_calibration=False)
    config["routine_vaccination"]["target_origin_distribution_by_age"] = {
        config["age_groups"][1]["label"]: {"dose9_recent": 1.0}
    }
    params = PreparedParameters.from_config(config, analysis="test", scenario="invalid_routine")
    index = StateIndex(params.age_groups)

    with pytest.raises(FastRHSUnsupportedError, match="Unknown vaccine origin"):
        build_fast_rhs(params, index)


def test_microbenchmark_helper_reports_warmed_timings():
    params = _params(fixed_demography=True)
    index = StateIndex(params.age_groups)
    result = benchmark_fast_rhs(params, index, initial_state(params, index), repeats=5)

    assert result["reference_seconds_per_call"] > 0.0
    assert result["fast_seconds_per_call"] > 0.0
    assert result["speedup"] > 0.0
