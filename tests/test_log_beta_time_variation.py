from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src_python.model.compartments import StateIndex
from src_python.model.fast_rhs import (
    FastRHSUnsupportedError,
    NUMBA_FAST_RHS_AVAILABLE,
    _continuous_log_beta_multiplier,
    build_fast_rhs,
)
from src_python.model.force_of_infection import (
    _log_beta_time_variation_multiplier_at,
    compute_force_of_infection,
)
from src_python.model.ode_system import rhs
from src_python.model.outputs import initial_state
from src_python.model.parameters import PreparedParameters
from src_python.model.solver_utils import dynamic_breakpoints
from src_python.simulation.common import make_config


_ABSENT = object()


def _config(periods: object = _ABSENT, *, calendar_enabled: bool = True) -> dict:
    config = make_config(country_profile="Australia", load_calibration=False)
    config["calendar"] = {
        "enabled": calendar_enabled,
        "analysis_start_date": "2020-01-01",
    }
    config["simulation"].update(
        {
            "start_time": 0.0,
            "end_time": 800.0,
            "burn_in_years": 0.0,
            "initial_state_strategy": "fixed_duration",
        }
    )
    config["transmission"]["npi_contact_reduction_periods"] = []
    config["routine_vaccination"]["delivery_shock_periods"] = []
    if periods is _ABSENT:
        config["transmission"].pop("log_beta_time_variation", None)
    else:
        config["transmission"]["log_beta_time_variation"] = {"periods": periods}
    return config


def _params(periods: object = _ABSENT, *, calendar_enabled: bool = True) -> PreparedParameters:
    return PreparedParameters.from_config(
        _config(periods, calendar_enabled=calendar_enabled),
        analysis="test",
        scenario="log_beta_time_variation",
    )


def test_calendar_log_beta_multiplier_preserves_fractional_and_inclusive_boundaries() -> None:
    log_annual = np.log(1.4)
    log_short = np.log(0.65)
    params = _params(
        [
            {
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "log_multiplier": log_annual,
            },
            {
                "start_date": "2021-02-03",
                "end_date": "2021-02-10",
                "log_multiplier": log_short,
            },
        ]
    )

    # 2020 is a leap year: its inclusive Dec-31 interval ends at t=366.
    expected = {
        -1e-9: 1.0,
        0.0: np.exp(log_annual),
        123.456: np.exp(log_annual),
        365.999999: np.exp(log_annual),
        366.0: 1.0,
        398.999999: 1.0,
        399.0: np.exp(log_short),
        406.75: np.exp(log_short),
        407.0: 1.0,
    }
    for t, multiplier in expected.items():
        assert _log_beta_time_variation_multiplier_at(t, params) == pytest.approx(
            multiplier,
            abs=2e-15,
        )


def test_reference_force_of_infection_multiplies_beta_not_the_state_or_reporting() -> None:
    period = [
        {
            "start_date": "2020-01-02",
            "end_date": "2020-01-03",
            "log_multiplier": np.log(1.75),
        }
    ]
    baseline = _params()
    varied = _params(period)
    index = StateIndex(baseline.age_groups)
    state = initial_state(baseline, index)

    baseline_active = compute_force_of_infection(1.25, state, baseline, index)
    varied_active = compute_force_of_infection(1.25, state, varied, index)
    assert np.max(baseline_active["lambda_S_base"]) > 0.0
    np.testing.assert_allclose(
        varied_active["lambda_S_base"],
        baseline_active["lambda_S_base"] * 1.75,
        rtol=5e-14,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        varied_active["lambda_R_base"],
        baseline_active["lambda_R_base"] * 1.75,
        rtol=5e-14,
        atol=1e-16,
    )

    baseline_outside = compute_force_of_infection(3.0, state, baseline, index)
    varied_outside = compute_force_of_infection(3.0, state, varied, index)
    np.testing.assert_array_equal(
        varied_outside["lambda_S_base"], baseline_outside["lambda_S_base"]
    )


def test_solver_breakpoints_include_every_log_beta_start_and_end_plus_one() -> None:
    params = _params(
        [
            {
                "start_date": "2020-01-02",
                "end_date": "2020-01-03",
                "log_multiplier": 0.1,
            },
            {
                "start_date": "2020-01-05",
                "end_date": "2020-01-05",
                "log_multiplier": -0.2,
            },
        ]
    )

    assert dynamic_breakpoints(params, 0.0, 10.0).tolist() == pytest.approx(
        [1.0, 3.0, 4.0, 5.0]
    )


def test_overlapping_log_beta_intervals_are_rejected_everywhere() -> None:
    params = _params(
        [
            {
                "start_date": "2020-01-02",
                "end_date": "2020-01-04",
                "log_multiplier": 0.1,
            },
            {
                # Both intervals include Jan-04, so precedence is ambiguous.
                "start_date": "2020-01-04",
                "end_date": "2020-01-06",
                "log_multiplier": -0.2,
            },
        ]
    )
    index = StateIndex(params.age_groups)

    with pytest.raises(ValueError, match="must not overlap"):
        _log_beta_time_variation_multiplier_at(0.0, params)
    with pytest.raises(ValueError, match="must not overlap"):
        dynamic_breakpoints(params, 0.0, 10.0)
    if NUMBA_FAST_RHS_AVAILABLE:
        with pytest.raises(FastRHSUnsupportedError, match="must not overlap"):
            build_fast_rhs(params, index)


def test_configured_calendar_periods_require_an_enabled_calendar() -> None:
    params = _params(
        [
            {
                "start_date": "2020-01-01",
                "end_date": "2020-01-01",
                "log_multiplier": 0.1,
            }
        ],
        calendar_enabled=False,
    )
    with pytest.raises(ValueError, match="requires an enabled calendar"):
        _log_beta_time_variation_multiplier_at(0.0, params)
    with pytest.raises(ValueError, match="requires an enabled calendar"):
        dynamic_breakpoints(params, 0.0, 10.0)


@pytest.mark.skipif(not NUMBA_FAST_RHS_AVAILABLE, reason="Numba fast RHS unavailable")
def test_fast_rhs_log_beta_period_array_and_reference_parity() -> None:
    params = _params(
        [
            {
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "log_multiplier": 0.17,
            },
            {
                "start_date": "2021-02-03",
                "end_date": "2021-02-10",
                "log_multiplier": -0.31,
            },
        ]
    )
    index = StateIndex(params.age_groups)
    fast = build_fast_rhs(params, index)
    assert fast.parameter_pack.log_beta_periods.shape == (2, 3)
    assert fast.parameter_pack.log_beta_periods.dtype == np.float64

    rng = np.random.default_rng(20260713)
    state = rng.gamma(1.4, 1.0, size=(index.n_age, index.n_compartments))
    state *= params.population[:, None] / state.sum(axis=1)[:, None]
    state = state.reshape(index.size)

    for t in (-0.1, 0.0, 19.875, 365.999999, 366.0, 398.999999, 399.0, 406.9, 407.0):
        ordinal = params.calendar_ordinal_at(t)
        assert ordinal is not None
        assert _continuous_log_beta_multiplier(
            ordinal, fast.parameter_pack.log_beta_periods
        ) == pytest.approx(_log_beta_time_variation_multiplier_at(t, params), abs=2e-15)
        np.testing.assert_allclose(
            fast(t, state),
            rhs(t, state, params, index),
            rtol=5e-13,
            atol=5e-9,
        )


@pytest.mark.skipif(not NUMBA_FAST_RHS_AVAILABLE, reason="Numba fast RHS unavailable")
def test_absent_or_empty_log_beta_configuration_leaves_rhs_unchanged() -> None:
    absent = _params()
    empty_config = _config([])
    empty = PreparedParameters.from_config(
        deepcopy(empty_config),
        analysis="test",
        scenario="empty_log_beta_time_variation",
    )
    index = StateIndex(absent.age_groups)
    state = initial_state(absent, index)
    fast_absent = build_fast_rhs(absent, index)
    fast_empty = build_fast_rhs(empty, index)

    assert fast_absent.parameter_pack.log_beta_periods.shape == (0, 3)
    assert fast_empty.parameter_pack.log_beta_periods.shape == (0, 3)
    for t in (0.0, 17.25, 399.5):
        assert _log_beta_time_variation_multiplier_at(t, absent) == 1.0
        assert _log_beta_time_variation_multiplier_at(t, empty) == 1.0
        np.testing.assert_array_equal(
            rhs(t, state, absent, index), rhs(t, state, empty, index)
        )
        np.testing.assert_array_equal(fast_absent(t, state), fast_empty(t, state))
