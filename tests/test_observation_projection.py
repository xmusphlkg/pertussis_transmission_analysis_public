from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src_python.model.compartments import StateIndex
from src_python.model.observables import (
    CASE_EVENT_SYMPTOMATIC_ONSET,
    CaseExposure,
    ObservationPlan,
    build_observation_plan,
    project_reported_cases,
    project_reporting_grid,
    solve_case_exposure,
)
from src_python.model.outputs import initial_state
from src_python.model.parameters import PreparedParameters
from src_python.simulation.common import make_config


def _prepared_parameters(
    *,
    diagnostic_periods: list[dict[str, object]] | None = None,
    reporting_time_variation: dict[str, float] | None = None,
    end_time: float = 366.0,
) -> PreparedParameters:
    config = make_config(load_calibration=False)
    config["calendar"]["analysis_start_date"] = "2020-01-01"
    config["calendar"]["analysis_end_date"] = "2021-01-01"
    config["simulation"].update(
        {
            "start_time": 0.0,
            "end_time": float(end_time),
            "burn_in_years": 0,
            "solver_method": "RK45",
            "rtol": 1e-7,
            "atol": 1e-9,
        }
    )
    config["diagnostic_reporting_time_variation"] = {
        "enabled": bool(diagnostic_periods),
        "periods": list(diagnostic_periods or []),
    }
    config["reporting_time_variation"] = dict(reporting_time_variation or {})
    return PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="observation_projection",
    )


def _manual_two_age_exposure() -> CaseExposure:
    plan = ObservationPlan(
        interval_ids=("surveillance-window",),
        interval_start_times=np.array([0.0]),
        interval_end_times=np.array([20.0]),
        segment_start_times=np.array([0.0, 10.0]),
        segment_end_times=np.array([10.0, 20.0]),
        segment_interval_index=np.array([0, 0], dtype=np.int64),
        diagnostic_multipliers=np.array([0.5, 2.0]),
        reporting_trend_multipliers=np.ones(2),
        endpoint_times=np.array([0.0, 10.0, 20.0]),
        age_groups=("younger", "older"),
    )
    return CaseExposure(
        plan=plan,
        case_counts=np.array(
            [
                [100.0, 50.0],
                [40.0, 80.0],
            ]
        ),
        case_event_definition="hand_calculated_fixture",
        numerical_diagnostics={},
    )


def test_projection_matches_two_age_two_diagnostic_period_hand_calculation() -> None:
    exposure = _manual_two_age_exposure()

    predicted = project_reported_cases(
        exposure,
        base_reporting_rates=np.array([0.2, 0.8]),
        reporting_multiplier=2.0,
    )

    # Nested clipping is intentional.  First clip the age-specific calibrated
    # probabilities to [0, 1]: [0.4, 1.0].  Then apply each diagnostic-period
    # multiplier and clip again, yielding probabilities [0.2, 0.5] and
    # [0.8, 1.0].  The two segment contributions are therefore
    # [20, 25] + [32, 80] = [52, 105].
    assert predicted.interval_age_mean.shape == (1, 2)
    assert predicted.interval_age_mean[0] == pytest.approx([52.0, 105.0])
    assert predicted.interval_total_mean == pytest.approx([157.0])

    # A single final clip would incorrectly give 120 older reported cases;
    # retain this assertion so a future vectorization cannot change semantics.
    assert predicted.interval_age_mean[0, 1] != pytest.approx(120.0)


def test_reporting_grid_equals_repeated_scalar_projection() -> None:
    exposure = _manual_two_age_exposure()
    rates = np.array([0.2, 0.8])
    multipliers = np.array([0.0, 0.5, 2.0, 4.0])

    grid = project_reporting_grid(exposure, rates, multipliers)
    scalar = np.stack(
        [
            project_reported_cases(exposure, rates, multiplier).interval_age_mean
            for multiplier in multipliers
        ],
        axis=0,
    )

    assert grid.shape == (len(multipliers), 1, 2)
    assert grid == pytest.approx(scalar)
    assert grid[0] == pytest.approx(np.zeros((1, 2)))


def test_partial_leap_day_interval_is_split_at_diagnostic_change() -> None:
    params = _prepared_parameters(
        diagnostic_periods=[
            {
                "start_date": "2019-01-01",
                "end_date": "2020-02-29",
                "multiplier": 0.5,
            },
            {
                "start_date": "2020-03-01",
                "end_date": "2021-12-31",
                "multiplier": 2.0,
            },
        ]
    )
    observed = pd.DataFrame(
        {
            "observed_interval_id": ["leap-weekend"],
            "period_start": ["2020-02-28"],
            "period_end": ["2020-03-02"],
        }
    )

    plan = build_observation_plan(observed, params)

    assert plan.interval_semantics == "half_open_[start,end)"
    assert plan.interval_ids == ("leap-weekend",)
    assert plan.interval_end_times - plan.interval_start_times == pytest.approx([3.0])
    assert plan.segment_end_times - plan.segment_start_times == pytest.approx([2.0, 1.0])
    assert plan.diagnostic_multipliers == pytest.approx([0.5, 2.0])
    assert plan.segment_interval_index.tolist() == [0, 0]


def test_observed_year_uses_all_366_days_of_leap_year() -> None:
    params = _prepared_parameters()
    observed = pd.DataFrame(
        {
            "observed_interval_id": ["2020"],
            "observed_year": [2020],
        }
    )

    plan = build_observation_plan(observed, params)

    assert plan.interval_start_times == pytest.approx([0.0])
    assert plan.interval_end_times == pytest.approx([366.0])
    assert plan.segment_end_times - plan.segment_start_times == pytest.approx([366.0])
    assert plan.endpoint_times == pytest.approx([0.0, 366.0])


def test_overlapping_observed_intervals_are_rejected() -> None:
    params = _prepared_parameters()
    observed = pd.DataFrame(
        {
            "observed_interval_id": ["first", "overlap"],
            "period_start": ["2020-01-01", "2020-01-09"],
            "period_end": ["2020-01-10", "2020-01-20"],
        }
    )

    with pytest.raises(ValueError, match="Overlapping observed intervals"):
        build_observation_plan(observed, params)


def test_overlapping_inclusive_diagnostic_periods_are_rejected() -> None:
    params = _prepared_parameters(
        diagnostic_periods=[
            {
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "multiplier": 0.8,
            },
            {
                "start_date": "2020-01-10",
                "end_date": "2020-01-20",
                "multiplier": 1.2,
            },
        ]
    )
    observed = pd.DataFrame(
        {
            "period_start": ["2020-01-01"],
            "period_end": ["2020-01-21"],
        }
    )

    with pytest.raises(ValueError, match="Diagnostic-standard periods must not overlap"):
        build_observation_plan(observed, params)


def test_continuous_reporting_trend_is_explicitly_rejected() -> None:
    params = _prepared_parameters(
        reporting_time_variation={
            "start_time": 0.0,
            "end_time": 366.0,
            "start_multiplier": 0.8,
            "end_multiplier": 1.2,
        }
    )
    observed = pd.DataFrame(
        {
            "period_start": ["2020-01-01"],
            "period_end": ["2020-02-01"],
        }
    )

    with pytest.raises(NotImplementedError, match="requires quadrature"):
        build_observation_plan(observed, params)


def test_short_ode_case_counters_are_nonnegative_and_auditable() -> None:
    params = _prepared_parameters(end_time=2.0)
    # Isolate progression of the deliberately seeded exposed population so the
    # counter must be positive without relying on onward transmission/imports.
    params.raw["transmission"]["beta_S"] = 0.0
    params.raw["importation"]["enabled"] = False
    params.raw["initial_conditions"]["initial_exposed_per_100k"] = 10.0
    params.raw["initial_conditions"]["initial_infectious_per_100k"] = 0.0
    params = PreparedParameters.from_config(
        params.raw,
        analysis="test",
        scenario="short_case_counter",
    )
    index = StateIndex(params.age_groups)
    y0 = initial_state(params, index)
    y0_before = y0.copy()
    observed = pd.DataFrame(
        {
            "observed_interval_id": ["day-1", "day-2"],
            "period_start": ["2020-01-01", "2020-01-02"],
            "period_end": ["2020-01-02", "2020-01-03"],
        }
    )
    plan = build_observation_plan(observed, params)

    exposure = solve_case_exposure(
        params,
        index,
        y0,
        plan,
        case_event_definition=CASE_EVENT_SYMPTOMATIC_ONSET,
    )

    assert np.array_equal(y0, y0_before)
    assert exposure.case_counts.shape == (2, index.n_age)
    assert np.isfinite(exposure.case_counts).all()
    assert np.all(exposure.case_counts >= 0.0)
    assert float(exposure.case_counts.sum()) > 0.0
    assert exposure.case_event_definition == CASE_EVENT_SYMPTOMATIC_ONSET
    assert exposure.numerical_diagnostics["nfev"] > 0
    assert exposure.numerical_diagnostics["counter_negative_corrections"] == 0

