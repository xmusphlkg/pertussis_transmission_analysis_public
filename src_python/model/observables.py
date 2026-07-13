"""Lightweight biological-to-observation interface for calibration/inference.

The full publication timeseries is intentionally not part of this module.
Biological dynamics produce true case exposure on atomic calendar segments;
reporting, diagnostic standards, and likelihood parameters are then applied by
small vectorized projection functions without re-solving the ODE.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src_python.model.compartments import (
    VACCINE_ORIGINS,
    StateIndex,
    compartment_index,
    exposed_name,
    susceptible_name,
)
from src_python.model.force_of_infection import compute_force_of_infection
from src_python.model.outputs import model_rhs_callable, state_invariant_diagnostics
from src_python.model.parameters import PreparedParameters
from src_python.model.solver_utils import solve_ivp_piecewise


CASE_EVENT_INFECTION_DESTINED = "infection_destined_symptomatic"
CASE_EVENT_SYMPTOMATIC_ONSET = "symptomatic_onset"
CASE_EVENT_DEFINITIONS = frozenset(
    {CASE_EVENT_INFECTION_DESTINED, CASE_EVENT_SYMPTOMATIC_ONSET}
)


@dataclass(frozen=True)
class ObservationPlan:
    """Atomic, non-overlapping calendar segments for one observation stream."""

    interval_ids: tuple[str, ...]
    interval_start_times: np.ndarray
    interval_end_times: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    segment_interval_index: np.ndarray
    diagnostic_multipliers: np.ndarray
    reporting_trend_multipliers: np.ndarray
    endpoint_times: np.ndarray
    age_groups: tuple[str, ...]
    interval_semantics: str = "half_open_[start,end)"

    @property
    def n_intervals(self) -> int:
        return len(self.interval_ids)

    @property
    def n_segments(self) -> int:
        return len(self.segment_start_times)


@dataclass(frozen=True)
class CaseExposure:
    """True symptomatic-case counts on atomic segments by age."""

    plan: ObservationPlan
    case_counts: np.ndarray
    case_event_definition: str
    numerical_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PredictedCases:
    """Observation-model mean counts by interval and age/total."""

    interval_age_mean: np.ndarray
    interval_total_mean: np.ndarray


def _as_date(value: Any, *, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid {label}: {value!r}")
    return pd.Timestamp(parsed).date()


def _calendar_time(value: date, params: PreparedParameters) -> float:
    if params.calendar_start_date is None:
        raise ValueError("Observation plans require calendar.analysis_start_date")
    start_time = float(params.raw["simulation"].get("start_time", 0.0))
    return start_time + float((value - params.calendar_start_date).days)


def _observed_calendar_intervals(observed: pd.DataFrame) -> list[tuple[str, date, date]]:
    if observed.empty:
        raise ValueError("Observed case frame is empty")
    if {"period_start", "period_end"}.issubset(observed.columns):
        starts = [
            _as_date(value, label="period_start") for value in observed["period_start"]
        ]
        ends = [_as_date(value, label="period_end") for value in observed["period_end"]]
    elif "observed_year" in observed.columns:
        years = pd.to_numeric(observed["observed_year"], errors="raise").astype(int)
        starts = [date(int(year), 1, 1) for year in years]
        ends = [date(int(year) + 1, 1, 1) for year in years]
    else:
        raise ValueError(
            "Observed cases require period_start/period_end or observed_year"
        )

    if "observed_interval_id" in observed.columns:
        ids = tuple(observed["observed_interval_id"].astype(str))
    elif "series_year" in observed.columns:
        ids = tuple(observed["series_year"].astype(str))
    else:
        ids = tuple(str(i) for i in range(len(observed)))
    if len(set(ids)) != len(ids):
        raise ValueError("Observed interval IDs must be unique")

    intervals = list(zip(ids, starts, ends))
    for interval_id, start, end in intervals:
        if end <= start:
            raise ValueError(
                f"Observed interval {interval_id!r} must have period_end after period_start"
            )
    chronological = sorted(intervals, key=lambda item: (item[1], item[2], item[0]))
    for previous, current in zip(chronological, chronological[1:]):
        if current[1] < previous[2]:
            raise ValueError(
                "Overlapping observed intervals cannot be independent likelihood terms: "
                f"{previous[0]!r} and {current[0]!r}"
            )
    return intervals


def _diagnostic_change_dates(params: PreparedParameters) -> tuple[date, ...]:
    variation = params.diagnostic_reporting_time_variation
    if not isinstance(variation, dict) or not bool(variation.get("enabled", False)):
        return ()
    periods = variation.get("periods", [])
    parsed: list[tuple[date, date]] = []
    for period in periods if isinstance(periods, list) else []:
        if not isinstance(period, dict):
            continue
        start = _as_date(period.get("start_date"), label="diagnostic start_date")
        end = _as_date(period.get("end_date"), label="diagnostic end_date")
        if end < start:
            raise ValueError("Diagnostic period end_date precedes start_date")
        parsed.append((start, end))
    parsed.sort()
    for previous, current in zip(parsed, parsed[1:]):
        if current[0] <= previous[1]:
            raise ValueError("Diagnostic-standard periods must not overlap")
    changes = {boundary for start, end in parsed for boundary in (start, end + timedelta(days=1))}
    return tuple(sorted(changes))


def _constant_reporting_trend(params: PreparedParameters) -> float:
    variation = params.reporting_time_variation
    if not isinstance(variation, dict) or not variation:
        return 1.0
    start = float(variation.get("start_multiplier", 1.0))
    end = float(variation.get("end_multiplier", 1.0))
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0.0 or end <= 0.0:
        raise ValueError("Reporting trend multipliers must be finite and > 0")
    if not np.isclose(start, end, rtol=0.0, atol=1e-12):
        raise NotImplementedError(
            "Continuous reporting_time_variation requires quadrature; the "
            "case-exposure fast path refuses to approximate it silently"
        )
    return start


def build_observation_plan(
    observed: pd.DataFrame,
    params: PreparedParameters,
) -> ObservationPlan:
    """Build half-open observed intervals split at diagnostic change dates."""

    intervals = _observed_calendar_intervals(observed)
    diagnostic_changes = _diagnostic_change_dates(params)
    trend = _constant_reporting_trend(params)
    sim = params.raw["simulation"]
    simulation_start = float(sim["start_time"])
    simulation_end = float(sim["end_time"])

    interval_start_times: list[float] = []
    interval_end_times: list[float] = []
    segment_start_times: list[float] = []
    segment_end_times: list[float] = []
    segment_interval_index: list[int] = []
    diagnostic_multipliers: list[float] = []

    for interval_index, (_interval_id, start, end) in enumerate(intervals):
        start_time = _calendar_time(start, params)
        end_time = _calendar_time(end, params)
        if start_time < simulation_start - 1e-9 or end_time > simulation_end + 1e-9:
            raise ValueError(
                f"Observed interval [{start}, {end}) lies outside simulation window "
                f"[{simulation_start}, {simulation_end}]"
            )
        interval_start_times.append(start_time)
        interval_end_times.append(end_time)
        local_dates = [start]
        local_dates.extend(change for change in diagnostic_changes if start < change < end)
        local_dates.append(end)
        for segment_start, segment_end in zip(local_dates, local_dates[1:]):
            segment_start_time = _calendar_time(segment_start, params)
            segment_end_time = _calendar_time(segment_end, params)
            midpoint_time = 0.5 * (segment_start_time + segment_end_time)
            segment_start_times.append(segment_start_time)
            segment_end_times.append(segment_end_time)
            segment_interval_index.append(interval_index)
            diagnostic_multipliers.append(
                float(params.diagnostic_reporting_multiplier_at(midpoint_time))
            )

    segment_starts = np.asarray(segment_start_times, dtype=float)
    segment_ends = np.asarray(segment_end_times, dtype=float)
    endpoints = np.unique(np.concatenate((segment_starts, segment_ends)))
    return ObservationPlan(
        interval_ids=tuple(interval[0] for interval in intervals),
        interval_start_times=np.asarray(interval_start_times, dtype=float),
        interval_end_times=np.asarray(interval_end_times, dtype=float),
        segment_start_times=segment_starts,
        segment_end_times=segment_ends,
        segment_interval_index=np.asarray(segment_interval_index, dtype=np.int64),
        diagnostic_multipliers=np.asarray(diagnostic_multipliers, dtype=float),
        reporting_trend_multipliers=np.full(len(segment_starts), trend, dtype=float),
        endpoint_times=endpoints,
        age_groups=tuple(params.age_groups),
    )


def symptomatic_case_rate_by_age(
    t: float,
    y: np.ndarray,
    params: PreparedParameters,
    index: StateIndex,
    *,
    case_event_definition: str,
) -> np.ndarray:
    """Return the explicitly defined true symptomatic-case event rate by age."""

    definition = str(case_event_definition)
    if definition not in CASE_EVENT_DEFINITIONS:
        raise ValueError(
            f"Unsupported case_event_definition={definition!r}; "
            f"expected one of {sorted(CASE_EVENT_DEFINITIONS)}"
        )
    state = np.maximum(index.reshape(y), 0.0)
    rates = np.zeros(index.n_age, dtype=float)
    if definition == CASE_EVENT_SYMPTOMATIC_ONSET:
        sigma = float(params.rates["latent"])
        for origin_index, origin in enumerate(VACCINE_ORIGINS):
            p_sym = params.origin_symptomatic_prob[origin_index]
            for strain in ("S", "R"):
                exposed = state[:, index.index(0, exposed_name(strain, origin)) % index.n_compartments]
                rates += sigma * exposed * p_sym
        return rates

    foi = compute_force_of_infection(t, y, params, index)
    for origin_index, origin in enumerate(VACCINE_ORIGINS):
        susceptible = state[
            :, index.index(0, susceptible_name(origin)) % index.n_compartments
        ]
        susceptibility = float(params.origin_susceptibility[origin_index])
        p_sym = params.origin_symptomatic_prob[origin_index]
        rates += (
            (np.asarray(foi["lambda_S"]) + np.asarray(foi["lambda_R"]))
            * susceptibility
            * susceptible
            * p_sym
        )
    return rates


def solve_case_exposure(
    params: PreparedParameters,
    index: StateIndex,
    analysis_initial_state: np.ndarray,
    plan: ObservationPlan,
    *,
    case_event_definition: str | None = None,
) -> CaseExposure:
    """Integrate biological state plus non-feedback per-age case counters."""

    if tuple(plan.age_groups) != tuple(index.age_groups):
        raise ValueError("Observation plan and StateIndex age groups do not match")
    definition = str(
        case_event_definition
        or params.observation_model.get(
            "case_event_definition", CASE_EVENT_INFECTION_DESTINED
        )
    )
    if definition not in CASE_EVENT_DEFINITIONS:
        raise ValueError(f"Unsupported case_event_definition={definition!r}")
    sim = params.raw["simulation"]
    solver_method = str(sim.get("solver_method", "RK45"))
    if solver_method.upper() == "RK4":
        raise NotImplementedError("Augmented case counters do not yet support the legacy RK4 wrapper")

    y0 = np.asarray(analysis_initial_state, dtype=float)
    if y0.shape != (index.size,):
        raise ValueError(f"Expected analysis_initial_state shape ({index.size},), got {y0.shape}")
    if not np.isfinite(y0).all():
        raise ValueError("analysis_initial_state must contain only finite values")
    initial_negative_tolerance = max(
        10.0 * float(sim.get("atol", 1e-8)),
        1e-10,
    )
    if float(np.min(y0)) < -initial_negative_tolerance:
        raise ValueError(
            "analysis_initial_state contains a materially negative compartment"
        )
    y0 = np.maximum(y0, 0.0)
    start_time = float(sim["start_time"])
    end_time = float(np.max(plan.endpoint_times))
    if end_time < start_time:
        raise ValueError("Observation plan ends before simulation start")
    endpoints = np.unique(np.concatenate(([start_time], plan.endpoint_times)))
    endpoints = endpoints[(endpoints >= start_time - 1e-12) & (endpoints <= end_time + 1e-12)]
    augmented_initial = np.concatenate((np.array(y0, copy=True), np.zeros(index.n_age)))
    biological_rhs = model_rhs_callable(params, index)
    if definition == CASE_EVENT_SYMPTOMATIC_ONSET:
        exposed_s_indices = np.asarray(
            [compartment_index(exposed_name("S", origin)) for origin in VACCINE_ORIGINS],
            dtype=np.int64,
        )
        exposed_r_indices = np.asarray(
            [compartment_index(exposed_name("R", origin)) for origin in VACCINE_ORIGINS],
            dtype=np.int64,
        )
        symptomatic_probability = np.asarray(
            params.origin_symptomatic_prob,
            dtype=float,
        ).T
        latent_rate = float(params.rates["latent"])

        def case_rate_callable(biological: np.ndarray) -> np.ndarray:
            state = np.maximum(index.reshape(biological), 0.0)
            exposed = state[:, exposed_s_indices] + state[:, exposed_r_indices]
            return latent_rate * np.sum(exposed * symptomatic_probability, axis=1)

    def augmented_rhs(t: float, augmented: np.ndarray) -> np.ndarray:
        biological = augmented[: index.size]
        biological_derivative = biological_rhs(t, biological)
        if definition == CASE_EVENT_SYMPTOMATIC_ONSET:
            case_rate = case_rate_callable(biological)
        else:
            case_rate = symptomatic_case_rate_by_age(
                t,
                biological,
                params,
                index,
                case_event_definition=definition,
            )
        # Keep counters in natural count units.  Scaling them by total
        # population while using one scalar solve_ivp ``atol`` makes the error
        # controller tolerate material count errors in large countries.  The
        # counters do not feed back into biology, and their natural scale gives
        # the relative-tolerance controller the intended accuracy directly.
        return np.concatenate((biological_derivative, case_rate))

    solution = solve_ivp_piecewise(
        fun=augmented_rhs,
        t_span=(start_time, end_time),
        y0=augmented_initial,
        t_eval=endpoints,
        params=params,
        method=solver_method,
        rtol=float(sim.get("rtol", 1e-6)),
        atol=float(sim.get("atol", 1e-8)),
    )
    if not solution.success:
        raise RuntimeError(f"Case-exposure solve failed for {params.scenario}: {solution.message}")

    counters = solution.y[index.size :, :]
    endpoint_index = {round(float(time), 10): i for i, time in enumerate(solution.t)}
    case_counts = np.empty((plan.n_segments, index.n_age), dtype=float)
    for segment_index, (segment_start, segment_end) in enumerate(
        zip(plan.segment_start_times, plan.segment_end_times)
    ):
        start_position = endpoint_index.get(round(float(segment_start), 10))
        end_position = endpoint_index.get(round(float(segment_end), 10))
        if start_position is None or end_position is None:
            raise RuntimeError("ODE output did not contain every atomic observation endpoint")
        case_counts[segment_index] = counters[:, end_position] - counters[:, start_position]
    scale = max(float(np.max(np.abs(case_counts))) if case_counts.size else 0.0, 1.0)
    tolerance = 1e-9 * scale
    minimum = float(np.min(case_counts)) if case_counts.size else 0.0
    if minimum < -tolerance:
        raise RuntimeError(f"Case counters decreased by {minimum}, beyond tolerance {tolerance}")
    corrected_negative = int(np.count_nonzero(case_counts < 0.0))
    case_counts = np.maximum(case_counts, 0.0)
    diagnostics: dict[str, Any] = {
        **state_invariant_diagnostics(solution.y[: index.size], index),
        "nfev": int(solution.nfev),
        "counter_scale": 1.0,
        "counter_negative_corrections": corrected_negative,
        "minimum_raw_segment_case_count": minimum,
    }
    return CaseExposure(
        plan=plan,
        case_counts=case_counts,
        case_event_definition=definition,
        numerical_diagnostics=diagnostics,
    )


def _validate_projection_inputs(
    exposure: CaseExposure,
    base_reporting_rates: np.ndarray,
    diagnostic_multipliers: np.ndarray | None,
    reporting_trend: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_counts = np.asarray(exposure.case_counts, dtype=float)
    expected_case_shape = (
        exposure.plan.n_segments,
        len(exposure.plan.age_groups),
    )
    if case_counts.shape != expected_case_shape:
        raise ValueError(
            f"Expected case_counts shape {expected_case_shape}, got {case_counts.shape}"
        )
    if not np.isfinite(case_counts).all() or np.any(case_counts < 0.0):
        raise ValueError("case_counts must be finite and non-negative")
    interval_index = np.asarray(exposure.plan.segment_interval_index)
    if interval_index.shape != (exposure.plan.n_segments,) or np.any(
        (interval_index < 0) | (interval_index >= exposure.plan.n_intervals)
    ):
        raise ValueError("segment_interval_index contains an invalid interval index")
    rates = np.asarray(base_reporting_rates, dtype=float)
    if rates.shape != (len(exposure.plan.age_groups),):
        raise ValueError(
            f"Expected {len(exposure.plan.age_groups)} base reporting rates, got {rates.shape}"
        )
    if not np.isfinite(rates).all() or np.any((rates < 0.0) | (rates > 1.0)):
        raise ValueError("Base reporting rates must be finite probabilities")
    diagnostic = np.asarray(
        exposure.plan.diagnostic_multipliers
        if diagnostic_multipliers is None
        else diagnostic_multipliers,
        dtype=float,
    )
    trend = np.asarray(
        exposure.plan.reporting_trend_multipliers
        if reporting_trend is None
        else reporting_trend,
        dtype=float,
    )
    expected = (exposure.plan.n_segments,)
    if diagnostic.shape != expected or trend.shape != expected:
        raise ValueError("Diagnostic and reporting-trend multipliers must be segment vectors")
    if not np.isfinite(diagnostic).all() or not np.isfinite(trend).all():
        raise ValueError("Observation multipliers must be finite")
    if np.any(diagnostic <= 0.0) or np.any(trend <= 0.0):
        raise ValueError("Observation multipliers must be > 0")
    return rates, diagnostic, trend


def _aggregate_segment_cases(
    plan: ObservationPlan,
    segment_values: np.ndarray,
) -> np.ndarray:
    """Aggregate segment×age values in O(n_segments) memory and time."""

    values = np.asarray(segment_values, dtype=float)
    if values.shape != (plan.n_segments, len(plan.age_groups)):
        raise ValueError("segment_values shape does not match observation plan")
    interval_values = np.zeros(
        (plan.n_intervals, len(plan.age_groups)),
        dtype=float,
    )
    np.add.at(interval_values, plan.segment_interval_index, values)
    return interval_values


def project_reported_cases(
    exposure: CaseExposure,
    base_reporting_rates: np.ndarray,
    reporting_multiplier: float,
    *,
    diagnostic_multipliers: np.ndarray | None = None,
    reporting_trend: np.ndarray | None = None,
) -> PredictedCases:
    """Project one reporting multiplier using current nested-clipping semantics."""

    rates, diagnostic, trend = _validate_projection_inputs(
        exposure, base_reporting_rates, diagnostic_multipliers, reporting_trend
    )
    multiplier = float(reporting_multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("reporting_multiplier must be finite and >= 0")
    base_probability = np.clip(rates * multiplier, 0.0, 1.0)
    segment_probability = np.clip(
        base_probability[np.newaxis, :] * diagnostic[:, np.newaxis] * trend[:, np.newaxis],
        0.0,
        1.0,
    )
    segment_reported = exposure.case_counts * segment_probability
    interval_age = _aggregate_segment_cases(exposure.plan, segment_reported)
    return PredictedCases(
        interval_age_mean=interval_age,
        interval_total_mean=np.sum(interval_age, axis=1),
    )


def project_reporting_grid(
    exposure: CaseExposure,
    base_reporting_rates: np.ndarray,
    reporting_multipliers: np.ndarray,
    *,
    diagnostic_multipliers: np.ndarray | None = None,
    reporting_trend: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorize an entire reporting grid without additional biological solves."""

    rates, diagnostic, trend = _validate_projection_inputs(
        exposure, base_reporting_rates, diagnostic_multipliers, reporting_trend
    )
    multipliers = np.asarray(reporting_multipliers, dtype=float)
    if multipliers.ndim != 1 or not np.isfinite(multipliers).all() or np.any(multipliers < 0.0):
        raise ValueError("reporting_multipliers must be a finite non-negative vector")
    base_probability = np.clip(multipliers[:, np.newaxis] * rates[np.newaxis, :], 0.0, 1.0)
    segment_probability = np.clip(
        base_probability[:, np.newaxis, :]
        * diagnostic[np.newaxis, :, np.newaxis]
        * trend[np.newaxis, :, np.newaxis],
        0.0,
        1.0,
    )
    segment_reported = segment_probability * exposure.case_counts[np.newaxis, :, :]
    interval_reported = np.zeros(
        (
            len(multipliers),
            exposure.plan.n_intervals,
            len(exposure.plan.age_groups),
        ),
        dtype=float,
    )
    for grid_index in range(len(multipliers)):
        np.add.at(
            interval_reported[grid_index],
            exposure.plan.segment_interval_index,
            segment_reported[grid_index],
        )
    return interval_reported
