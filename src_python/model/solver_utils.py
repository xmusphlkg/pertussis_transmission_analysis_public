"""Numerically reliable integration helpers for calendar-driven dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from src_python.model.force_of_infection import _parse_log_beta_time_variation_periods
from src_python.model.parameters import PreparedParameters


@dataclass
class PiecewiseSolveResult:
    """Small solve_ivp-compatible result assembled across calendar segments."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    nfev: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = datetime.strptime(str(value), "%Y-%m-%d")
    except (TypeError, ValueError):
        return None
    return parsed.date()


def _calendar_time(params: PreparedParameters, value: date) -> float | None:
    if params.calendar_start_date is None:
        return None
    start_time = float(params.raw["simulation"].get("start_time", 0.0))
    return start_time + float((value - params.calendar_start_date).days)


def dynamic_breakpoints(
    params: PreparedParameters,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Return calendar times where the biological RHS jumps or changes slope.

    NPI, vaccination-shock, and log-beta interval starts are true jumps.  Their
    inclusive ends, recovery-ramp ends, calendar-year boundaries, and WPP
    interpolation knots are included as derivative-change points.  Restarting
    the adaptive solver at these known boundaries prevents a large step from
    straddling a process or policy discontinuity and makes tolerance refinement
    meaningful.
    """

    lower = float(min(t_start, t_end))
    upper = float(max(t_start, t_end))
    values: set[float] = set()

    schedule_blocks = (
        (params.transmission.get("npi_contact_reduction_periods", []), "ramp_days"),
        (params.routine_vaccination.get("delivery_shock_periods", []), "ramp_days"),
    )
    for periods, ramp_key in schedule_blocks:
        for period in periods if isinstance(periods, list) else []:
            if not isinstance(period, dict):
                continue
            start = _as_date(period.get("start_date"))
            end = _as_date(period.get("end_date"))
            if start is None or end is None or end < start:
                continue
            end_exclusive = end + timedelta(days=1)
            dates = [start, end_exclusive]
            try:
                ramp_days = float(
                    period.get(
                        ramp_key,
                        params.transmission.get("npi_ramp_days", 0.0),
                    )
                )
            except (TypeError, ValueError):
                ramp_days = 0.0
            if ramp_days > 0.0:
                ramp_end_time = _calendar_time(params, end_exclusive)
                if ramp_end_time is not None:
                    values.add(ramp_end_time + ramp_days)
            for boundary_date in dates:
                boundary_time = _calendar_time(params, boundary_date)
                if boundary_time is not None:
                    values.add(boundary_time)

    log_beta_periods = _parse_log_beta_time_variation_periods(params.transmission)
    if log_beta_periods and params.calendar_start_date is None:
        raise ValueError(
            "transmission.log_beta_time_variation.periods requires an enabled calendar."
        )
    for start_ordinal, end_exclusive_ordinal, _log_multiplier in log_beta_periods:
        for boundary_ordinal in (start_ordinal, end_exclusive_ordinal):
            boundary_time = _calendar_time(params, date.fromordinal(boundary_ordinal))
            if boundary_time is not None:
                values.add(boundary_time)

    if params.calendar_start_date is not None:
        start_time = float(params.raw["simulation"].get("start_time", 0.0))
        start_date = params.calendar_start_date + timedelta(days=np.floor(lower - start_time))
        end_date = params.calendar_start_date + timedelta(days=np.ceil(upper - start_time))
        for year in range(start_date.year, end_date.year + 2):
            boundary_time = _calendar_time(params, date(year, 1, 1))
            if boundary_time is not None:
                values.add(boundary_time)

    return np.asarray(
        sorted(value for value in values if lower < value < upper),
        dtype=float,
    )


def solve_ivp_piecewise(
    *,
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    params: PreparedParameters,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float | np.ndarray = 1e-8,
) -> PiecewiseSolveResult:
    """Integrate while restarting at every known biological time boundary."""

    t_start, t_end = (float(t_span[0]), float(t_span[1]))
    if t_end < t_start:
        raise ValueError("solve_ivp_piecewise requires t_span[1] >= t_span[0]")
    initial = np.asarray(y0, dtype=float)
    if t_end == t_start:
        requested = np.asarray([t_start] if t_eval is None else t_eval, dtype=float)
        return PiecewiseSolveResult(
            t=requested,
            y=np.repeat(initial[:, np.newaxis], len(requested), axis=1),
            success=True,
            message="Zero-length integration interval.",
            nfev=0,
        )

    requested = (
        np.asarray([t_end], dtype=float)
        if t_eval is None
        else np.unique(np.asarray(t_eval, dtype=float))
    )
    if requested.ndim != 1 or np.any(requested < t_start - 1e-10) or np.any(
        requested > t_end + 1e-10
    ):
        raise ValueError("t_eval values must lie within t_span")

    boundaries = np.concatenate(
        (
            np.asarray([t_start]),
            dynamic_breakpoints(params, t_start, t_end),
            np.asarray([t_end]),
        )
    )
    output_times: list[float] = []
    output_states: list[np.ndarray] = []
    current = np.array(initial, copy=True)
    nfev = 0

    if np.any(np.isclose(requested, t_start, rtol=0.0, atol=1e-10)):
        output_times.append(t_start)
        output_states.append(current.copy())

    for left, right in zip(boundaries, boundaries[1:]):
        segment_requested = requested[
            (requested > left + 1e-10) & (requested <= right + 1e-10)
        ]
        internal_eval = np.unique(np.concatenate((segment_requested, np.asarray([right]))))
        solution = solve_ivp(
            fun=fun,
            t_span=(float(left), float(right)),
            y0=current,
            t_eval=internal_eval,
            method=method,
            rtol=float(rtol),
            atol=atol,
        )
        nfev += int(solution.nfev)
        if not solution.success:
            return PiecewiseSolveResult(
                t=np.asarray(output_times, dtype=float),
                y=(
                    np.column_stack(output_states)
                    if output_states
                    else np.empty((len(initial), 0), dtype=float)
                ),
                success=False,
                message=str(solution.message),
                nfev=nfev,
            )
        current = np.array(solution.y[:, -1], copy=True)
        for requested_time in segment_requested:
            position = int(
                np.flatnonzero(
                    np.isclose(solution.t, requested_time, rtol=0.0, atol=1e-9)
                )[0]
            )
            output_times.append(float(requested_time))
            output_states.append(np.array(solution.y[:, position], copy=True))

    order = np.argsort(np.asarray(output_times, dtype=float))
    times = np.asarray(output_times, dtype=float)[order]
    states = np.column_stack(output_states)[:, order]
    return PiecewiseSolveResult(
        t=times,
        y=states,
        success=True,
        message="Piecewise adaptive integration completed.",
        nfev=nfev,
    )
