from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src_python.model.force_of_infection import _npi_contact_reduction_at
from src_python.model.parameters import PreparedParameters
from src_python.model.compartments import StateIndex
from src_python.model.outputs import history_integration_start_time, prepare_history_state
from src_python.model.solver_utils import dynamic_breakpoints
from src_python.simulation.common import make_config, set_analysis_horizon_years


def _calendar_params(
    *,
    npi_periods: list[dict[str, object]] | None = None,
    diagnostic_periods: list[dict[str, object]] | None = None,
) -> PreparedParameters:
    config = make_config(load_calibration=False)
    config["calendar"]["analysis_start_date"] = "2020-01-01"
    config["simulation"].update(
        {"start_time": 0.0, "end_time": 40.0, "burn_in_years": 0.0}
    )
    config["transmission"]["npi_contact_reduction_periods"] = list(
        npi_periods or []
    )
    config["diagnostic_reporting_time_variation"] = {
        "enabled": bool(diagnostic_periods),
        "periods": list(diagnostic_periods or []),
    }
    return PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="calendar_boundaries",
    )


def test_seasonal_calendar_phase_is_continuous_within_day() -> None:
    params = _calendar_params()

    start = params.calendar_day_of_year_at(0.0)
    quarter_day = params.calendar_day_of_year_at(0.25)
    half_day = params.calendar_day_of_year_at(0.5)

    assert start == pytest.approx(1.0)
    assert start < quarter_day < half_day
    assert half_day - start == pytest.approx(365.0 * 0.5 / 366.0)


def test_five_year_sensitivity_window_is_derived_from_policy_t0() -> None:
    config = make_config(load_calibration=False)
    config["calendar"]["analysis_start_date"] = "2027-01-01"

    end_date = set_analysis_horizon_years(config, 5)

    assert end_date == "2031-12-31"
    # The human-facing end date is inclusive; the continuous solver ends at
    # 2032-01-01 00:00, so the half-open interval contains all five years and
    # the 2028 leap day.
    assert config["simulation"]["end_time"] == pytest.approx(1826.0)


def test_production_horizon_uses_midnight_after_inclusive_end_date() -> None:
    config = make_config(load_calibration=False)

    assert config["calendar"]["analysis_start_date"] == "2027-01-01"
    assert config["calendar"]["analysis_end_date"] == "2050-12-31"
    assert config["simulation"]["end_time"] == pytest.approx(8766.0)
    # A constant one-event-per-day counter over the half-open solver interval
    # must accumulate one event for every included calendar day.
    constant_flow_total = 1.0 * config["simulation"]["end_time"]
    assert constant_flow_total == pytest.approx(8766.0)
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="production_horizon_boundary",
    )
    assert str(params.calendar_date_at(config["simulation"]["end_time"])) == (
        "2051-01-01"
    )


def test_npi_recovery_ramp_is_continuous_after_inclusive_end_date() -> None:
    params = _calendar_params(
        npi_periods=[
            {
                "start_date": "2020-01-02",
                "end_date": "2020-01-03",
                "reduction": 0.5,
                "ramp_days": 10.0,
            }
        ]
    )

    assert _npi_contact_reduction_at(1.0, params) == pytest.approx(0.5)
    assert _npi_contact_reduction_at(3.0, params) == pytest.approx(0.5)
    assert _npi_contact_reduction_at(8.0, params) == pytest.approx(0.75)
    assert _npi_contact_reduction_at(13.0, params) == pytest.approx(1.0)

    assert dynamic_breakpoints(params, 0.0, 20.0).tolist() == pytest.approx(
        [1.0, 3.0, 13.0]
    )


def test_diagnostic_gap_uses_previous_not_future_regime() -> None:
    params = _calendar_params(
        diagnostic_periods=[
            {
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "multiplier": 0.8,
            },
            {
                "start_date": "2020-01-10",
                "end_date": "2020-01-20",
                "multiplier": 1.2,
            },
        ]
    )

    assert params.diagnostic_reporting_multiplier_at(5.0) == pytest.approx(0.8)
    assert params.diagnostic_reporting_multiplier_at(12.0) == pytest.approx(1.2)


def test_fixed_calendar_history_origin_is_independent_of_legacy_burn_length() -> None:
    base = make_config(load_calibration=False)
    base["calendar"]["analysis_start_date"] = "2020-01-01"
    base["simulation"].update(
        {
            "start_time": 0.0,
            "end_time": 5.0,
            "initial_state_strategy": "fixed_calendar_origin",
            "history_start_date": "2018-01-01",
            "rtol": 1e-5,
            "atol": 1e-7,
        }
    )
    three_config = deepcopy(base)
    three_config["simulation"]["burn_in_years"] = 3.0
    params_three = PreparedParameters.from_config(
        three_config,
        analysis="test",
        scenario="calendar_history_three",
    )
    fifty_config = deepcopy(base)
    fifty_config["simulation"]["burn_in_years"] = 50.0
    params_fifty = PreparedParameters.from_config(
        fifty_config,
        analysis="test",
        scenario="calendar_history_fifty",
    )
    index = StateIndex(params_three.age_groups)

    assert history_integration_start_time(params_three) == pytest.approx(-730.0)
    assert history_integration_start_time(params_fifty) == pytest.approx(-730.0)
    assert prepare_history_state(params_three, index) == pytest.approx(
        prepare_history_state(params_fifty, index),
        rel=1e-10,
        abs=1e-7,
    )
