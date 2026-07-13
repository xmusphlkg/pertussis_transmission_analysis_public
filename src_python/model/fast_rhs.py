"""Complete Numba fast path for the pertussis ODE right-hand side.

The reference :func:`src_python.model.ode_system.rhs` deliberately favours
readability.  That is useful for scientific review, but expensive in an MCMC
loop because every derivative evaluation rebuilds dictionaries and repeatedly
crosses the Python/Numba boundary.  This module keeps the reference RHS as the
scientific specification and provides an independent, fully compiled
implementation of the same equations.

``build_fast_rhs`` performs all configuration parsing once and returns a
SciPy-compatible ``fun(t, y)`` callable.  The compiled kernel receives only
numeric scalars and contiguous NumPy arrays.  Seasonal calendar coefficients
are precomputed over the configured burn-in and analysis window.  NPI and
routine-delivery periods are converted to numeric half-open intervals and
evaluated at continuous solver time inside the kernel.  WPP trajectories retain
the reference model's continuous linear interpolation.

The implementation intentionally uses ``fastmath=False``.  Unsupported shapes,
unknown compartments, malformed WPP trajectories, and evaluations outside the
precomputed calendar window raise :class:`FastRHSUnsupportedError`; the fast
path never silently substitutes a simplified model.
"""

from __future__ import annotations

from datetime import date, datetime
from math import floor
from time import perf_counter
from typing import NamedTuple

import numpy as np

from src_python.model.compartments import (
    COMPARTMENTS,
    VACCINE_ORIGINS,
    StateIndex,
    compartment_index,
    compartment_name,
    exposed_name,
    infectious_name,
    susceptible_name,
    treated_name,
)
from src_python.model.force_of_infection import _parse_log_beta_time_variation_periods
from src_python.model.parameters import PreparedParameters
from src_python.model.treatment import treated_infectiousness_relative, treated_recovery_rate
from src_python.model.vaccination import (
    default_routine_target_origin_distribution,
    origin_is_vaccine_dose,
)

try:
    from numba import njit

    NUMBA_FAST_RHS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal installations
    NUMBA_FAST_RHS_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(function):
            return function

        if args and callable(args[0]):
            return args[0]
        return decorator


class FastRHSUnsupportedError(ValueError):
    """Raised when a configuration cannot be represented without approximation."""


class FastRHSParameters(NamedTuple):
    """Pure-numeric parameter package consumed by the Numba kernel."""

    n_age: int
    n_comp: int
    n_origins: int
    susceptible_indices: np.ndarray
    exposed_s_indices: np.ndarray
    exposed_r_indices: np.ndarray
    infectious_s_sym_indices: np.ndarray
    infectious_s_asym_indices: np.ndarray
    infectious_r_sym_indices: np.ndarray
    infectious_r_asym_indices: np.ndarray
    treated_s_indices: np.ndarray
    treated_r_indices: np.ndarray
    susceptible_index: int
    maternal_index: int
    recovered_index: int
    waned_natural_index: int
    dose1_recent_index: int
    dose1_waned_index: int
    dose2_recent_index: int
    dose2_waned_index: int
    recent_index: int
    waned_index: int
    contact_matrix: np.ndarray
    origin_susceptibility: np.ndarray
    origin_infectiousness: np.ndarray
    origin_symptomatic_probability: np.ndarray
    origin_recovery_multiplier: np.ndarray
    diagnosis_probability: np.ndarray
    pep_detection_rate: np.ndarray
    vaccine_coverage: np.ndarray
    latent_rate: float
    recovery_symptomatic_rate: float
    recovery_asymptomatic_rate: float
    treated_recovery_s_rate: float
    treated_recovery_r_rate: float
    waning_vaccine_rate: float
    waning_vaccine_waned_rate: float
    waning_maternal_rate: float
    waning_natural_rate: float
    waning_r_to_w_rate: float
    waning_w_to_s_rate: float
    treatment_symptomatic_rate: float
    treatment_asymptomatic_rate: float
    relative_infectiousness_asymptomatic: float
    treated_infectiousness_s: float
    treated_infectiousness_r: float
    beta_s: float
    fitness_r: float
    seasonal_amplitude: float
    seasonal_phase: float
    multi_year_amplitude: float
    multi_year_period_days: float
    multi_year_phase: float
    calendar_enabled: bool
    calendar_time_origin: float
    calendar_start_ordinal: float
    calendar_min_day_offset: int
    seasonal_day_table: np.ndarray
    seasonal_day_step_table: np.ndarray
    log_beta_periods: np.ndarray
    npi_periods: np.ndarray
    routine_delivery_periods: np.ndarray
    pep_coverage: float
    pep_effectiveness_s: float
    pep_effectiveness_r: float
    pep_activation_prevalence: float
    boosting_enabled: bool
    boosting_efficiency: float
    waned_natural_susceptibility: float
    routine_enabled: bool
    routine_rate_per_day: float
    routine_max_daily_flow_fraction: float
    routine_target_distribution: np.ndarray
    importation_enabled: bool
    imported_by_age_per_day: np.ndarray
    imported_resistant_fraction: float
    resistance_anchor_enabled: bool
    resistance_anchor_rate_per_day: float
    resistance_anchor_target: float
    demography_mode: int
    aging_rates: np.ndarray
    maternal_exit_age_mask: np.ndarray
    birth_entry_weights: np.ndarray
    wpp_years: np.ndarray
    wpp_population: np.ndarray
    wpp_births: np.ndarray
    wpp_start_year: float
    wpp_nudge_rate_per_day: float
    wpp_year_intercept: float
    wpp_time_origin: float


class FastRHS:
    """SciPy-compatible callable owning a compiled numeric RHS parameter pack."""

    __slots__ = ("parameter_pack", "index", "supported_time_span")

    def __init__(
        self,
        parameter_pack: FastRHSParameters,
        index: StateIndex,
        supported_time_span: tuple[float, float] | None,
    ) -> None:
        self.parameter_pack = parameter_pack
        self.index = index
        self.supported_time_span = supported_time_span

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        time = float(t)
        if not np.isfinite(time):
            raise FastRHSUnsupportedError("Fast RHS requires a finite evaluation time.")

        if self.parameter_pack.calendar_enabled:
            day_offset = floor(time - self.parameter_pack.calendar_time_origin)
            first = self.parameter_pack.calendar_min_day_offset
            last = first + self.parameter_pack.seasonal_day_table.size - 1
            if day_offset < first or day_offset > last:
                assert self.supported_time_span is not None
                raise FastRHSUnsupportedError(
                    "Fast RHS calendar lookup is outside its configured time span "
                    f"[{self.supported_time_span[0]}, {self.supported_time_span[1]}): t={time}."
                )

        state = np.asarray(y, dtype=np.float64)
        if state.size != self.index.size:
            raise FastRHSUnsupportedError(
                f"State has {state.size} entries; expected {self.index.size}."
            )
        state = state.reshape(self.index.size)
        return _rhs_kernel(time, state, self.parameter_pack)


def _f64(value: np.ndarray | list[float], *, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if not np.isfinite(array).all():
        raise FastRHSUnsupportedError(f"{name} must contain only finite values.")
    return array


def _i64(value: np.ndarray | list[int]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=np.int64))


def _boolean(value: np.ndarray | list[bool]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=np.bool_))


def _finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FastRHSUnsupportedError(f"{name} must be numeric.") from exc
    if not np.isfinite(result):
        raise FastRHSUnsupportedError(f"{name} must be finite.")
    return result


def _date_ordinal(value: object) -> int:
    if isinstance(value, datetime):
        return value.date().toordinal()
    if isinstance(value, date):
        return value.toordinal()
    return datetime.strptime(str(value), "%Y-%m-%d").date().toordinal()


def _numeric_periods(
    periods: object,
    *,
    default_ramp_days: float,
) -> np.ndarray:
    """Return ``[start, end_exclusive, reduction, ramp_days]`` numeric rows."""

    parsed: list[tuple[int, int, float, float]] = []
    if not periods:
        return np.empty((0, 4), dtype=np.float64)
    if not isinstance(periods, (list, tuple)):
        raise FastRHSUnsupportedError("Calendar shock periods must be a sequence of mappings.")
    for period in periods:
        if not isinstance(period, dict):
            # Matches the reference path, which ignores non-mapping records.
            continue
        try:
            start = _date_ordinal(period.get("start_date", ""))
            end = _date_ordinal(period.get("end_date", ""))
            reduction = float(period.get("reduction", 0.0))
            ramp_days = float(period.get("ramp_days", default_ramp_days))
        except (TypeError, ValueError):
            # Matches the reference path's explicit invalid-record handling.
            continue
        if (
            end >= start
            and np.isfinite(reduction)
            and 0.0 <= reduction <= 1.0
            and np.isfinite(ramp_days)
            and ramp_days >= 0.0
        ):
            # Configured dates are inclusive calendar days.  Continuous model
            # time therefore enters recovery at midnight after the end date.
            parsed.append((start, end + 1, reduction, ramp_days))
    parsed.sort(key=lambda item: item[0])
    if not parsed:
        return np.empty((0, 4), dtype=np.float64)
    return _f64(parsed, name="calendar shock periods")


def _numeric_log_beta_periods(params: PreparedParameters) -> np.ndarray:
    """Return validated ``[start, end_exclusive, log_multiplier]`` rows."""

    try:
        parsed = _parse_log_beta_time_variation_periods(params.transmission)
    except ValueError as exc:
        raise FastRHSUnsupportedError(str(exc)) from exc
    if parsed and params.calendar_start_date is None:
        raise FastRHSUnsupportedError(
            "transmission.log_beta_time_variation.periods requires an enabled calendar."
        )
    if not parsed:
        return np.empty((0, 3), dtype=np.float64)
    return _f64(parsed, name="log-beta time-variation periods")


def _calendar_tables(
    params: PreparedParameters,
) -> tuple[
    bool,
    float,
    int,
    np.ndarray,
    np.ndarray,
    tuple[float, float] | None,
]:
    if params.calendar_start_date is None:
        ones = np.ones(1, dtype=np.float64)
        return False, 0.0, 0, ones, ones.copy(), None

    simulation = params.raw.get("simulation", {})
    start_time = _finite_float(simulation.get("start_time", 0.0), name="simulation.start_time")
    end_time = _finite_float(simulation.get("end_time", start_time), name="simulation.end_time")
    burn_in_years = _finite_float(simulation.get("burn_in_years", 0.0), name="simulation.burn_in_years")
    if end_time < start_time:
        raise FastRHSUnsupportedError("simulation.end_time must be >= simulation.start_time.")
    if burn_in_years < 0.0:
        raise FastRHSUnsupportedError("simulation.burn_in_years must be >= 0.")

    # Two guard days on either side protect adaptive-solver endpoint probes while
    # keeping out-of-contract evaluations explicit.
    initial_state_strategy = str(
        simulation.get("initial_state_strategy", "fixed_duration")
    ).lower()
    if initial_state_strategy in {"fixed_calendar_origin", "calendar_origin"}:
        try:
            history_ordinal = _date_ordinal(simulation.get("history_start_date", ""))
        except (TypeError, ValueError) as exc:
            raise FastRHSUnsupportedError(
                "fixed_calendar_origin requires a valid simulation.history_start_date"
            ) from exc
        minimum_offset = history_ordinal - params.calendar_start_date.toordinal() - 2
    elif initial_state_strategy in {"fixed_duration", "duration"}:
        minimum_offset = int(np.floor(-365.0 * burn_in_years)) - 2
    else:
        raise FastRHSUnsupportedError(
            f"Unsupported simulation.initial_state_strategy={initial_state_strategy!r}"
        )
    maximum_offset = int(np.floor(end_time - start_time)) + 2
    day_offsets = np.arange(minimum_offset, maximum_offset + 1, dtype=np.int64)

    base_date = np.datetime64(params.calendar_start_date.isoformat(), "D")
    calendar_dates = base_date + day_offsets.astype("timedelta64[D]")
    calendar_year_starts = calendar_dates.astype("datetime64[Y]")
    elapsed_in_year = (calendar_dates - calendar_year_starts).astype(np.int64).astype(np.float64)
    calendar_year_numbers = calendar_year_starts.astype(np.int64) + 1970
    leap_year = (
        (calendar_year_numbers % 4 == 0)
        & ((calendar_year_numbers % 100 != 0) | (calendar_year_numbers % 400 == 0))
    )
    year_lengths = np.where(leap_year, 366.0, 365.0)
    # ``PreparedParameters.calendar_day_of_year_at`` preserves the fractional
    # integrator time and maps both common and leap years continuously onto a
    # 365-day seasonal phase.  Store the affine coefficients for each calendar
    # day instead of approximating that cosine with a step function.
    seasonal_day = 1.0 + 365.0 * elapsed_in_year / year_lengths
    seasonal_day_step = 365.0 / year_lengths

    lower = start_time + minimum_offset
    upper = start_time + maximum_offset + 1.0
    return (
        True,
        start_time,
        minimum_offset,
        _f64(seasonal_day, name="seasonal-day calendar"),
        _f64(seasonal_day_step, name="seasonal-day calendar slope"),
        (lower, upper),
    )


def _routine_target_matrix(params: PreparedParameters) -> np.ndarray:
    configured = params.routine_vaccination.get("target_origin_distribution_by_age", {})
    if configured and not isinstance(configured, dict):
        raise FastRHSUnsupportedError(
            "routine_vaccination.target_origin_distribution_by_age must be a mapping."
        )
    origins = {origin: position for position, origin in enumerate(VACCINE_ORIGINS)}
    result = np.zeros((len(params.age_groups), len(VACCINE_ORIGINS)), dtype=np.float64)
    for age_position, age in enumerate(params.age_groups):
        distribution = configured.get(age) if age in configured else default_routine_target_origin_distribution(age)
        if not isinstance(distribution, dict):
            raise FastRHSUnsupportedError(
                f"Routine target distribution for {age!r} must be a mapping."
            )
        positive_total = 0.0
        for origin, raw_share in distribution.items():
            origin = str(origin)
            if not origin_is_vaccine_dose(origin):
                continue
            if origin not in origins:
                raise FastRHSUnsupportedError(
                    f"Unknown vaccine origin in routine target distribution: {origin!r}."
                )
            share = _finite_float(raw_share, name=f"routine target share {age}.{origin}")
            positive = max(0.0, share)
            result[age_position, origins[origin]] = positive
            positive_total += positive
        if positive_total > 0.0:
            result[age_position, :] /= positive_total
    return result


def _birth_entry_weights(params: PreparedParameters) -> np.ndarray:
    result = np.zeros(len(COMPARTMENTS), dtype=np.float64)
    birth_entry = params.demography.get("birth_entry", {"S": 1.0})
    if not isinstance(birth_entry, dict):
        raise FastRHSUnsupportedError("demography.birth_entry must be a mapping.")
    total_weight = 0.0
    for raw_compartment, raw_weight in birth_entry.items():
        weight = max(0.0, _finite_float(raw_weight, name=f"birth_entry.{raw_compartment}"))
        total_weight += weight
        resolved = compartment_name(str(raw_compartment))
        if resolved not in COMPARTMENTS:
            raise FastRHSUnsupportedError(
                f"Unknown demography birth-entry compartment: {raw_compartment!r}."
            )
        result[compartment_index(resolved)] += weight
    if total_weight <= 0.0:
        result[compartment_index("S")] = 1.0
    else:
        result /= total_weight
    return result


def _wpp_arrays(
    params: PreparedParameters,
    *,
    active: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    if not active:
        return (
            np.array([0.0], dtype=np.float64),
            np.zeros((len(params.age_groups), 1), dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            0.0,
            0.0,
            0.0,
        )

    trajectory = params.demography.get("wpp_trajectory")
    if not isinstance(trajectory, dict):
        raise FastRHSUnsupportedError("Active WPP demography requires a trajectory mapping.")
    try:
        years = sorted(int(item) for item in trajectory["years"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FastRHSUnsupportedError("WPP trajectory years are malformed.") from exc
    if not years or len(set(years)) != len(years):
        raise FastRHSUnsupportedError("WPP trajectory years must be non-empty and unique.")

    population_mapping = trajectory.get("population_by_year")
    births_mapping = trajectory.get("births_by_year")
    if not isinstance(population_mapping, dict) or not isinstance(births_mapping, dict):
        raise FastRHSUnsupportedError("WPP population_by_year and births_by_year must be mappings.")

    population = np.empty((len(params.age_groups), len(years)), dtype=np.float64)
    births = np.empty(len(years), dtype=np.float64)
    try:
        for age_position, age in enumerate(params.age_groups):
            bucket = population_mapping[age]
            if not isinstance(bucket, dict):
                raise TypeError
            population[age_position, :] = [
                float(bucket[year] if year in bucket else bucket[str(year)]) for year in years
            ]
        births[:] = [
            float(births_mapping[year] if year in births_mapping else births_mapping[str(year)])
            for year in years
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise FastRHSUnsupportedError(
            "WPP trajectory must provide every configured year for every age and births."
        ) from exc

    return (
        _f64(years, name="WPP years"),
        _f64(population, name="WPP population"),
        _f64(births, name="WPP births"),
        float(years[0]),
        _finite_float(
            params.demography.get("wpp_nudge_rate_per_year", 1.0),
            name="demography.wpp_nudge_rate_per_year",
        )
        / 365.0,
        float(trajectory.get("analysis_start_year", 2026)),
    )


def build_fast_rhs(params: PreparedParameters, index: StateIndex) -> FastRHS:
    """Build a fully compiled, SciPy-compatible RHS callable.

    Parameters are parsed and validated once.  No model dictionary, date, or
    Python callback is accessed by the derivative kernel.
    """

    if not NUMBA_FAST_RHS_AVAILABLE:
        raise FastRHSUnsupportedError("Numba is required for build_fast_rhs().")
    if not isinstance(params, PreparedParameters):
        raise FastRHSUnsupportedError("params must be a PreparedParameters instance.")
    if tuple(index.age_groups) != tuple(params.age_groups):
        raise FastRHSUnsupportedError("StateIndex age groups must exactly match PreparedParameters.")
    if index.n_compartments != len(COMPARTMENTS):
        raise FastRHSUnsupportedError("Fast RHS requires the canonical compartment layout.")

    n_age = index.n_age
    n_origins = len(VACCINE_ORIGINS)
    expected_arrays = {
        "origin_susceptibility": (params.origin_susceptibility, (n_origins,)),
        "origin_infectiousness": (params.origin_infectiousness, (n_origins,)),
        "origin_symptomatic_prob": (params.origin_symptomatic_prob, (n_origins, n_age)),
        "origin_recovery_mult": (params.origin_recovery_mult, (n_origins,)),
    }
    for name, (array, shape) in expected_arrays.items():
        if np.asarray(array).shape != shape:
            raise FastRHSUnsupportedError(
                f"{name} has shape {np.asarray(array).shape}; expected {shape}. "
                "Construct parameters through PreparedParameters.from_config()."
            )

    susceptible_indices = _i64([compartment_index(susceptible_name(origin)) for origin in VACCINE_ORIGINS])
    exposed_s_indices = _i64([compartment_index(exposed_name("S", origin)) for origin in VACCINE_ORIGINS])
    exposed_r_indices = _i64([compartment_index(exposed_name("R", origin)) for origin in VACCINE_ORIGINS])
    infectious_s_sym_indices = _i64(
        [compartment_index(infectious_name("S", "sym", origin)) for origin in VACCINE_ORIGINS]
    )
    infectious_s_asym_indices = _i64(
        [compartment_index(infectious_name("S", "asym", origin)) for origin in VACCINE_ORIGINS]
    )
    infectious_r_sym_indices = _i64(
        [compartment_index(infectious_name("R", "sym", origin)) for origin in VACCINE_ORIGINS]
    )
    infectious_r_asym_indices = _i64(
        [compartment_index(infectious_name("R", "asym", origin)) for origin in VACCINE_ORIGINS]
    )
    treated_s_indices = _i64([compartment_index(treated_name("S", origin)) for origin in VACCINE_ORIGINS])
    treated_r_indices = _i64([compartment_index(treated_name("R", origin)) for origin in VACCINE_ORIGINS])

    calendar = _calendar_tables(params)
    npi_default_ramp = _finite_float(
        params.transmission.get("npi_ramp_days", 90.0),
        name="transmission.npi_ramp_days",
    )
    npi_periods = _numeric_periods(
        params.transmission.get("npi_contact_reduction_periods"),
        default_ramp_days=npi_default_ramp,
    )
    log_beta_periods = _numeric_log_beta_periods(params)
    routine_delivery_periods = _numeric_periods(
        params.routine_vaccination.get("delivery_shock_periods"),
        default_ramp_days=0.0,
    )

    routine_rate = _finite_float(
        params.routine_vaccination.get("target_relaxation_rate_per_year", 0.0),
        name="routine_vaccination.target_relaxation_rate_per_year",
    ) / 365.0
    routine_max_flow = _finite_float(
        params.routine_vaccination.get("max_daily_flow_fraction", 0.01),
        name="routine_vaccination.max_daily_flow_fraction",
    )
    routine_enabled = bool(params.routine_vaccination.get("enabled", False)) and routine_rate > 0.0

    importation_enabled = bool(params.importation.get("enabled", False))
    importation_rate = _finite_float(
        params.importation.get("rate_per_100k_per_year", 0.0),
        name="importation.rate_per_100k_per_year",
    )
    importation_enabled = importation_enabled and importation_rate > 0.0
    age_distribution = np.array(
        [
            float(
                params.importation.get("age_distribution", {}).get(
                    age, 1.0 / n_age
                )
            )
            for age in params.age_groups
        ],
        dtype=np.float64,
    )
    if not np.isfinite(age_distribution).all() or np.any(age_distribution < 0.0):
        raise FastRHSUnsupportedError("Importation age distribution must be finite and non-negative.")
    distribution_total = float(age_distribution.sum())
    if distribution_total <= 0.0:
        age_distribution[:] = 1.0 / n_age
    else:
        age_distribution /= distribution_total
    imported_by_age = (
        params.total_population * importation_rate / 100_000.0 / 365.0 * age_distribution
    )

    resistance_anchor_rate = _finite_float(
        params.resistance.get("prevalence_anchor_rate_per_year", 0.0),
        name="resistance.prevalence_anchor_rate_per_year",
    ) / 365.0
    resistance_anchor_enabled = (
        bool(params.resistance.get("anchor_during_dynamics", False))
        and resistance_anchor_rate > 0.0
    )
    resistance_anchor_target = float(
        np.clip(
            _finite_float(
                params.resistance.get(
                    "target_prevalence_at_analysis_start",
                    params.initial.get("initial_resistance_prevalence", 0.0),
                ),
                name="resistance target prevalence",
            ),
            0.0,
            1.0,
        )
    )

    demography_enabled = bool(params.demography.get("enabled", False))
    wpp_active = demography_enabled and params.wpp_trajectory_active()
    demography_mode = 2 if wpp_active else (1 if demography_enabled else 0)
    durations = _f64(
        [
            params.demography.get("age_bin_durations_years", {}).get(age, 1.0)
            for age in params.age_groups
        ],
        name="demography age-bin durations",
    )
    if np.any(durations <= 0.0):
        raise FastRHSUnsupportedError("All demography age-bin durations must be > 0.")
    if demography_mode == 1 and bool(params.demography.get("fixed_population_profile", True)):
        reference_age = str(
            params.demography.get("fixed_population_reference_age_group", params.age_groups[0])
        )
        reference_position = params.age_groups.index(reference_age) if reference_age in params.age_groups else 0
        target_population = np.maximum(params.population, 1e-12)
        reference_flow = target_population[reference_position] / (durations[reference_position] * 365.0)
        aging_rates = reference_flow / target_population
    else:
        aging_rates = 1.0 / (durations * 365.0)

    maternal_proxy = params.demography.get("maternal_protection_proxy", {})
    if maternal_proxy and not isinstance(maternal_proxy, dict):
        raise FastRHSUnsupportedError("demography.maternal_protection_proxy must be a mapping.")
    maternal_exit_groups = (
        set(maternal_proxy.get("exit_age_groups", []))
        if maternal_proxy.get("enabled", False)
        else set()
    )
    maternal_exit_mask = _boolean([age in maternal_exit_groups for age in params.age_groups])

    wpp_years, wpp_population, wpp_births, wpp_start_year, wpp_nudge_rate, analysis_start_year = _wpp_arrays(
        params,
        active=wpp_active,
    )
    if params.calendar_start_date is not None:
        jan1 = date(params.calendar_start_date.year, 1, 1)
        wpp_year_intercept = params.calendar_start_date.year + (
            params.calendar_start_date.toordinal() - jan1.toordinal()
        ) / 365.25
        wpp_time_origin = float(params.raw.get("simulation", {}).get("start_time", 0.0))
    else:
        wpp_year_intercept = analysis_start_year
        wpp_time_origin = 0.0

    base_gamma_sym = _finite_float(
        params.rates["recovery_symptomatic"], name="rates.recovery_symptomatic"
    )
    gamma_treated_s = treated_recovery_rate(base_gamma_sym, params.treatment, "S")
    gamma_treated_r = treated_recovery_rate(base_gamma_sym, params.treatment, "R")

    package = FastRHSParameters(
        n_age=n_age,
        n_comp=index.n_compartments,
        n_origins=n_origins,
        susceptible_indices=susceptible_indices,
        exposed_s_indices=exposed_s_indices,
        exposed_r_indices=exposed_r_indices,
        infectious_s_sym_indices=infectious_s_sym_indices,
        infectious_s_asym_indices=infectious_s_asym_indices,
        infectious_r_sym_indices=infectious_r_sym_indices,
        infectious_r_asym_indices=infectious_r_asym_indices,
        treated_s_indices=treated_s_indices,
        treated_r_indices=treated_r_indices,
        susceptible_index=compartment_index("S"),
        maternal_index=compartment_index("M_protected"),
        recovered_index=compartment_index("R_natural"),
        waned_natural_index=compartment_index("W_natural"),
        dose1_recent_index=compartment_index("V_dose1_recent"),
        dose1_waned_index=compartment_index("V_dose1_waned"),
        dose2_recent_index=compartment_index("V_dose2_recent"),
        dose2_waned_index=compartment_index("V_dose2_waned"),
        recent_index=compartment_index("V_recent"),
        waned_index=compartment_index("V_waned"),
        contact_matrix=_f64(params.contact_matrix, name="contact matrix"),
        origin_susceptibility=_f64(params.origin_susceptibility, name="origin susceptibility"),
        origin_infectiousness=_f64(params.origin_infectiousness, name="origin infectiousness"),
        origin_symptomatic_probability=_f64(
            params.origin_symptomatic_prob, name="origin symptomatic probability"
        ),
        origin_recovery_multiplier=_f64(params.origin_recovery_mult, name="origin recovery multiplier"),
        diagnosis_probability=_f64(params.diagnosis_probability, name="diagnosis probability"),
        pep_detection_rate=_f64(params.pep_detection_rate, name="PEP detection rate"),
        vaccine_coverage=_f64(params.vaccine_coverage, name="vaccine coverage"),
        latent_rate=_finite_float(params.rates["latent"], name="rates.latent"),
        recovery_symptomatic_rate=base_gamma_sym,
        recovery_asymptomatic_rate=_finite_float(
            params.rates["recovery_asymptomatic"], name="rates.recovery_asymptomatic"
        ),
        treated_recovery_s_rate=float(gamma_treated_s),
        treated_recovery_r_rate=float(gamma_treated_r),
        waning_vaccine_rate=_finite_float(params.rates["waning_vaccine"], name="rates.waning_vaccine"),
        waning_vaccine_waned_rate=_finite_float(
            params.rates.get("waning_vaccine_waned", params.rates["waning_vaccine"]),
            name="rates.waning_vaccine_waned",
        ),
        waning_maternal_rate=_finite_float(
            params.rates.get("waning_maternal", 0.0), name="rates.waning_maternal"
        ),
        waning_natural_rate=_finite_float(params.rates["waning_natural"], name="rates.waning_natural"),
        waning_r_to_w_rate=_finite_float(
            params.rates.get("waning_R_to_W", params.rates["waning_natural"]),
            name="rates.waning_R_to_W",
        ),
        waning_w_to_s_rate=_finite_float(
            params.rates.get("waning_W_to_S", params.rates["waning_natural"] * 0.5),
            name="rates.waning_W_to_S",
        ),
        treatment_symptomatic_rate=_finite_float(
            params.treatment["treatment_rate_symptomatic"], name="treatment symptomatic rate"
        ),
        treatment_asymptomatic_rate=_finite_float(
            params.treatment["treatment_rate_asymptomatic"], name="treatment asymptomatic rate"
        ),
        relative_infectiousness_asymptomatic=_finite_float(
            params.transmission["relative_infectiousness_asymptomatic"],
            name="relative infectiousness asymptomatic",
        ),
        treated_infectiousness_s=float(treated_infectiousness_relative(params.treatment, "S")),
        treated_infectiousness_r=float(treated_infectiousness_relative(params.treatment, "R")),
        beta_s=_finite_float(params.transmission["beta_S"], name="transmission.beta_S"),
        fitness_r=_finite_float(params.transmission.get("fitness_R", 1.0), name="transmission.fitness_R"),
        seasonal_amplitude=_finite_float(
            params.transmission.get("seasonal_amplitude", 0.0), name="seasonal amplitude"
        ),
        seasonal_phase=_finite_float(params.transmission.get("seasonal_phase", 0.0), name="seasonal phase"),
        multi_year_amplitude=_finite_float(
            params.transmission.get("multi_year_amplitude", 0.0), name="multi-year amplitude"
        ),
        multi_year_period_days=365.0
        * _finite_float(
            params.transmission.get("multi_year_period_years", 4.0), name="multi-year period"
        ),
        multi_year_phase=_finite_float(
            params.transmission.get("multi_year_phase", 0.0), name="multi-year phase"
        ),
        calendar_enabled=calendar[0],
        calendar_time_origin=calendar[1],
        calendar_start_ordinal=(
            float(params.calendar_start_date.toordinal())
            if params.calendar_start_date is not None
            else 0.0
        ),
        calendar_min_day_offset=calendar[2],
        seasonal_day_table=calendar[3],
        seasonal_day_step_table=calendar[4],
        log_beta_periods=log_beta_periods,
        npi_periods=npi_periods,
        routine_delivery_periods=routine_delivery_periods,
        pep_coverage=_finite_float(
            params.pep.get("coverage_household_contacts", 0.0), name="PEP coverage"
        ),
        pep_effectiveness_s=_finite_float(
            params.pep.get("effectiveness_sensitive", 0.0), name="PEP sensitive effectiveness"
        ),
        pep_effectiveness_r=_finite_float(
            params.pep.get("effectiveness_resistant", 0.0), name="PEP resistant effectiveness"
        ),
        pep_activation_prevalence=_finite_float(
            params.pep.get("activation_prevalence", 1e-5), name="PEP activation prevalence"
        ),
        boosting_enabled=bool(params.immunity_model.get("boosting_enabled", True)),
        boosting_efficiency=_finite_float(
            params.immunity_model.get("boosting_efficiency", 0.7), name="boosting efficiency"
        ),
        waned_natural_susceptibility=float(
            np.clip(
                _finite_float(
                    params.immunity_model.get("waned_natural_infection_susceptibility", 0.35),
                    name="waned natural infection susceptibility",
                ),
                0.0,
                1.0,
            )
        ),
        routine_enabled=routine_enabled,
        routine_rate_per_day=routine_rate,
        routine_max_daily_flow_fraction=routine_max_flow,
        routine_target_distribution=_f64(_routine_target_matrix(params), name="routine target matrix"),
        importation_enabled=importation_enabled,
        imported_by_age_per_day=_f64(imported_by_age, name="daily importation by age"),
        imported_resistant_fraction=_finite_float(
            params.importation.get(
                "resistant_fraction", params.initial.get("initial_resistance_prevalence", 0.0)
            ),
            name="importation resistant fraction",
        ),
        resistance_anchor_enabled=resistance_anchor_enabled,
        resistance_anchor_rate_per_day=resistance_anchor_rate,
        resistance_anchor_target=resistance_anchor_target,
        demography_mode=demography_mode,
        aging_rates=_f64(aging_rates, name="demography aging rates"),
        maternal_exit_age_mask=maternal_exit_mask,
        birth_entry_weights=_f64(_birth_entry_weights(params), name="birth-entry weights"),
        wpp_years=wpp_years,
        wpp_population=wpp_population,
        wpp_births=wpp_births,
        wpp_start_year=wpp_start_year,
        wpp_nudge_rate_per_day=wpp_nudge_rate,
        wpp_year_intercept=wpp_year_intercept,
        wpp_time_origin=wpp_time_origin,
    )
    return FastRHS(package, index, calendar[5])


@njit(cache=True, fastmath=False, error_model="numpy")
def _linear_interpolation_position(x: float, points: np.ndarray) -> tuple[int, int, float]:
    size = points.size
    if x <= points[0]:
        return 0, 0, 0.0
    if x >= points[size - 1]:
        return size - 1, size - 1, 0.0
    upper = np.searchsorted(points, x, side="right")
    lower = upper - 1
    fraction = (x - points[lower]) / (points[upper] - points[lower])
    return lower, upper, fraction


@njit(cache=True, fastmath=False, error_model="numpy")
def _continuous_npi_multiplier(calendar_ordinal: float, periods: np.ndarray) -> float:
    """Exact continuous-time equivalent of ``_npi_contact_reduction_at``."""

    multiplier = 1.0
    for position in range(periods.shape[0]):
        start = periods[position, 0]
        end_exclusive = periods[position, 1]
        reduction = periods[position, 2]
        if start <= calendar_ordinal < end_exclusive:
            candidate = 1.0 - reduction
            if candidate < multiplier:
                multiplier = candidate

    # The reference treats an active policy interval as authoritative; recovery
    # tails from earlier intervals are considered only when no interval is active.
    if multiplier < 1.0:
        return multiplier

    for position in range(periods.shape[0]):
        ramp_start = periods[position, 1]
        reduction = periods[position, 2]
        ramp_days = periods[position, 3]
        days_after = calendar_ordinal - ramp_start
        if ramp_days > 0.0 and 0.0 <= days_after < ramp_days:
            candidate = (1.0 - reduction) + reduction * days_after / ramp_days
            if candidate < multiplier:
                multiplier = candidate
    return multiplier


@njit(cache=True, fastmath=False, error_model="numpy")
def _continuous_log_beta_multiplier(calendar_ordinal: float, periods: np.ndarray) -> float:
    """Return the multiplier from validated, non-overlapping log-beta rows."""

    for position in range(periods.shape[0]):
        if periods[position, 0] <= calendar_ordinal < periods[position, 1]:
            return np.exp(periods[position, 2])
    return 1.0


@njit(cache=True, fastmath=False, error_model="numpy")
def _continuous_routine_delivery_multiplier(
    calendar_ordinal: float,
    periods: np.ndarray,
) -> float:
    """Exact continuous-time equivalent of ``_routine_delivery_multiplier_at``."""

    multiplier = 1.0
    for position in range(periods.shape[0]):
        start = periods[position, 0]
        end_exclusive = periods[position, 1]
        reduction = periods[position, 2]
        if start <= calendar_ordinal < end_exclusive:
            candidate = 1.0 - reduction
            if candidate < multiplier:
                multiplier = candidate

    # Routine-delivery shocks combine active periods and all overlapping
    # recovery tails by the most restrictive (minimum) multiplier.
    for position in range(periods.shape[0]):
        ramp_start = periods[position, 1]
        reduction = periods[position, 2]
        ramp_days = periods[position, 3]
        days_after = calendar_ordinal - ramp_start
        if ramp_days > 0.0 and 0.0 <= days_after < ramp_days:
            candidate = (1.0 - reduction) + reduction * days_after / ramp_days
            if candidate < multiplier:
                multiplier = candidate
    if multiplier < 0.0:
        return 0.0
    if multiplier > 1.0:
        return 1.0
    return multiplier


@njit(cache=True, fastmath=False, error_model="numpy")
def _rhs_kernel(t: float, y: np.ndarray, p: FastRHSParameters) -> np.ndarray:
    """Full compiled implementation of ``ode_system.rhs``."""

    state = np.empty((p.n_age, p.n_comp), dtype=np.float64)
    dy = np.zeros((p.n_age, p.n_comp), dtype=np.float64)
    population = np.empty(p.n_age, dtype=np.float64)

    for age in range(p.n_age):
        age_population = 0.0
        offset = age * p.n_comp
        for compartment in range(p.n_comp):
            value = y[offset + compartment]
            # ``np.maximum(value, 0)`` semantics preserve NaN values.
            if value < 0.0:
                value = 0.0
            state[age, compartment] = value
            age_population += value
        if age_population < 1.0:
            age_population = 1.0
        population[age] = age_population

    # Infectious pressure by source age and strain.
    pressure_s = np.zeros(p.n_age, dtype=np.float64)
    pressure_r = np.zeros(p.n_age, dtype=np.float64)
    symptomatic = np.zeros(p.n_age, dtype=np.float64)
    for age in range(p.n_age):
        value_s = 0.0
        value_r = 0.0
        value_symptomatic = 0.0
        for origin in range(p.n_origins):
            infectiousness = p.origin_infectiousness[origin]
            i_s_sym = state[age, p.infectious_s_sym_indices[origin]]
            i_r_sym = state[age, p.infectious_r_sym_indices[origin]]
            value_s += infectiousness * (
                i_s_sym
                + p.relative_infectiousness_asymptomatic
                * state[age, p.infectious_s_asym_indices[origin]]
                + p.treated_infectiousness_s * state[age, p.treated_s_indices[origin]]
            )
            value_r += infectiousness * (
                i_r_sym
                + p.relative_infectiousness_asymptomatic
                * state[age, p.infectious_r_asym_indices[origin]]
                + p.treated_infectiousness_r * state[age, p.treated_r_indices[origin]]
            )
            value_symptomatic += i_s_sym + i_r_sym
        pressure_s[age] = value_s / population[age]
        pressure_r[age] = value_r / population[age]
        symptomatic[age] = value_symptomatic

    if p.calendar_enabled:
        calendar_delta = t - p.calendar_time_origin
        whole_calendar_days = np.floor(calendar_delta)
        calendar_position = int(whole_calendar_days) - p.calendar_min_day_offset
        calendar_fraction = calendar_delta - whole_calendar_days
        seasonal_day = (
            p.seasonal_day_table[calendar_position]
            + p.seasonal_day_step_table[calendar_position] * calendar_fraction
        )
        annual_multiplier = 1.0 + p.seasonal_amplitude * np.cos(
            2.0 * np.pi * (seasonal_day - p.seasonal_phase) / 365.0
        )
        calendar_ordinal = p.calendar_start_ordinal + calendar_delta
        beta_time_multiplier = _continuous_log_beta_multiplier(
            calendar_ordinal,
            p.log_beta_periods,
        )
        npi_multiplier = _continuous_npi_multiplier(calendar_ordinal, p.npi_periods)
        routine_delivery_multiplier = _continuous_routine_delivery_multiplier(
            calendar_ordinal,
            p.routine_delivery_periods,
        )
    else:
        seasonal_day = t % 365.0
        annual_multiplier = 1.0 + p.seasonal_amplitude * np.cos(
            2.0 * np.pi * (seasonal_day - p.seasonal_phase) / 365.0
        )
        npi_multiplier = 1.0
        beta_time_multiplier = 1.0
        routine_delivery_multiplier = 1.0

    if p.multi_year_period_days <= 0.0:
        multi_year_multiplier = 1.0
    else:
        multi_year_multiplier = 1.0 + p.multi_year_amplitude * np.cos(
            2.0 * np.pi * (t - p.multi_year_phase) / p.multi_year_period_days
        )
    seasonal_multiplier = annual_multiplier * multi_year_multiplier * npi_multiplier
    if seasonal_multiplier < 0.0:
        seasonal_multiplier = 0.0
    transmission_multiplier = seasonal_multiplier * beta_time_multiplier

    lambda_s = np.zeros(p.n_age, dtype=np.float64)
    lambda_r = np.zeros(p.n_age, dtype=np.float64)
    detected_prevalence = np.empty(p.n_age, dtype=np.float64)
    for source_age in range(p.n_age):
        detected_prevalence[source_age] = (
            symptomatic[source_age] * p.pep_detection_rate[source_age] / population[source_age]
        )
    for target_age in range(p.n_age):
        mixed_s = 0.0
        mixed_r = 0.0
        detected = 0.0
        for source_age in range(p.n_age):
            contact = p.contact_matrix[target_age, source_age]
            mixed_s += contact * pressure_s[source_age]
            mixed_r += contact * pressure_r[source_age]
            detected += contact * detected_prevalence[source_age]
        base_s = p.beta_s * transmission_multiplier * mixed_s
        base_r = p.beta_s * p.fitness_r * transmission_multiplier * mixed_r
        activation = detected / (detected + p.pep_activation_prevalence)
        coverage = p.pep_coverage * activation
        value_s = base_s * (1.0 - coverage * p.pep_effectiveness_s)
        value_r = base_r * (1.0 - coverage * p.pep_effectiveness_r)
        if value_s < 0.0:
            value_s = 0.0
        if value_r < 0.0:
            value_r = 0.0
        lambda_s[target_age] = value_s
        lambda_r[target_age] = value_r

    # Maternal and vaccine waning.
    for age in range(p.n_age):
        maternal_flow = p.waning_maternal_rate * state[age, p.maternal_index]
        dy[age, p.maternal_index] -= maternal_flow
        dy[age, p.susceptible_index] += maternal_flow

        recent_flow = p.waning_vaccine_rate * state[age, p.dose1_recent_index]
        waned_flow = p.waning_vaccine_waned_rate * state[age, p.dose1_waned_index]
        dy[age, p.dose1_recent_index] -= recent_flow
        dy[age, p.dose1_waned_index] += recent_flow - waned_flow
        dy[age, p.susceptible_index] += waned_flow

        recent_flow = p.waning_vaccine_rate * state[age, p.dose2_recent_index]
        waned_flow = p.waning_vaccine_waned_rate * state[age, p.dose2_waned_index]
        dy[age, p.dose2_recent_index] -= recent_flow
        dy[age, p.dose2_waned_index] += recent_flow - waned_flow
        dy[age, p.susceptible_index] += waned_flow

        recent_flow = p.waning_vaccine_rate * state[age, p.recent_index]
        waned_flow = p.waning_vaccine_waned_rate * state[age, p.waned_index]
        dy[age, p.recent_index] -= recent_flow
        dy[age, p.waned_index] += recent_flow - waned_flow
        dy[age, p.susceptible_index] += waned_flow

    # Infection, progression, treatment, and recovery for both strains and all
    # eight immune/vaccine origins.
    for age in range(p.n_age):
        recovered = 0.0
        treatment_sym = p.treatment_symptomatic_rate * p.diagnosis_probability[age]
        treatment_asym = p.treatment_asymptomatic_rate * p.diagnosis_probability[age]
        for origin in range(p.n_origins):
            susceptible_index = p.susceptible_indices[origin]
            susceptible = state[age, susceptible_index]
            infection_s = lambda_s[age] * p.origin_susceptibility[origin] * susceptible
            infection_r = lambda_r[age] * p.origin_susceptibility[origin] * susceptible
            dy[age, susceptible_index] -= infection_s + infection_r

            probability_symptomatic = p.origin_symptomatic_probability[origin, age]
            recovery_multiplier = p.origin_recovery_multiplier[origin]
            gamma_sym = p.recovery_symptomatic_rate * recovery_multiplier
            gamma_asym = p.recovery_asymptomatic_rate * recovery_multiplier

            exposed_index = p.exposed_s_indices[origin]
            i_sym_index = p.infectious_s_sym_indices[origin]
            i_asym_index = p.infectious_s_asym_indices[origin]
            treated_index = p.treated_s_indices[origin]
            progression = p.latent_rate * state[age, exposed_index]
            dy[age, exposed_index] = infection_s - progression
            dy[age, i_sym_index] = (
                probability_symptomatic * progression
                - treatment_sym * state[age, i_sym_index]
                - gamma_sym * state[age, i_sym_index]
            )
            dy[age, i_asym_index] = (
                (1.0 - probability_symptomatic) * progression
                - treatment_asym * state[age, i_asym_index]
                - gamma_asym * state[age, i_asym_index]
            )
            gamma_treated = p.treated_recovery_s_rate * recovery_multiplier
            dy[age, treated_index] = (
                treatment_sym * state[age, i_sym_index]
                + treatment_asym * state[age, i_asym_index]
                - gamma_treated * state[age, treated_index]
            )
            recovered += (
                gamma_sym * state[age, i_sym_index]
                + gamma_asym * state[age, i_asym_index]
                + gamma_treated * state[age, treated_index]
            )

            exposed_index = p.exposed_r_indices[origin]
            i_sym_index = p.infectious_r_sym_indices[origin]
            i_asym_index = p.infectious_r_asym_indices[origin]
            treated_index = p.treated_r_indices[origin]
            progression = p.latent_rate * state[age, exposed_index]
            dy[age, exposed_index] = infection_r - progression
            dy[age, i_sym_index] = (
                probability_symptomatic * progression
                - treatment_sym * state[age, i_sym_index]
                - gamma_sym * state[age, i_sym_index]
            )
            dy[age, i_asym_index] = (
                (1.0 - probability_symptomatic) * progression
                - treatment_asym * state[age, i_asym_index]
                - gamma_asym * state[age, i_asym_index]
            )
            gamma_treated = p.treated_recovery_r_rate * recovery_multiplier
            dy[age, treated_index] = (
                treatment_sym * state[age, i_sym_index]
                + treatment_asym * state[age, i_asym_index]
                - gamma_treated * state[age, treated_index]
            )
            recovered += (
                gamma_sym * state[age, i_sym_index]
                + gamma_asym * state[age, i_asym_index]
                + gamma_treated * state[age, treated_index]
            )
        dy[age, p.recovered_index] = recovered

    # SIRWS immune boosting and breakthrough infection.
    for age in range(p.n_age):
        if p.boosting_enabled:
            recovered_state = state[age, p.recovered_index]
            waned_state = state[age, p.waned_natural_index]
            r_to_w = p.waning_r_to_w_rate * recovered_state
            boosting = p.boosting_efficiency * (lambda_s[age] + lambda_r[age]) * waned_state
            breakthrough_multiplier = max(0.0, 1.0 - p.boosting_efficiency) * p.waned_natural_susceptibility
            breakthrough_s = breakthrough_multiplier * lambda_s[age] * waned_state
            breakthrough_r = breakthrough_multiplier * lambda_r[age] * waned_state
            w_to_s = p.waning_w_to_s_rate * waned_state

            dy[age, p.recovered_index] += -r_to_w + boosting
            dy[age, p.waned_natural_index] += r_to_w - boosting - w_to_s - breakthrough_s - breakthrough_r
            dy[age, p.susceptible_index] += w_to_s
            dy[age, p.exposed_s_indices[0]] += breakthrough_s
            dy[age, p.exposed_r_indices[0]] += breakthrough_r
        else:
            direct_waning = p.waning_natural_rate * state[age, p.recovered_index]
            dy[age, p.recovered_index] -= direct_waning
            dy[age, p.susceptible_index] += direct_waning

    # Routine vaccination transfers unvaccinated S to configured vaccine pools.
    if p.routine_enabled and routine_delivery_multiplier > 0.0:
        effective_rate = p.routine_rate_per_day * routine_delivery_multiplier
        effective_cap = p.routine_max_daily_flow_fraction * routine_delivery_multiplier
        for age in range(p.n_age):
            current_population = 0.0
            for compartment in range(p.n_comp):
                current_population += state[age, compartment]
            if current_population < 0.0:
                current_population = 0.0
            total_deficit = 0.0
            deficits = np.zeros(p.n_origins, dtype=np.float64)
            for origin in range(p.n_origins):
                share = p.routine_target_distribution[age, origin]
                if share <= 0.0:
                    continue
                desired = current_population * p.vaccine_coverage[age] * share
                deficit = desired - state[age, p.susceptible_indices[origin]]
                if deficit > 0.0:
                    deficits[origin] = deficit
                    total_deficit += deficit
            if total_deficit <= 0.0:
                continue
            available = state[age, p.susceptible_index]
            total_flow = min(effective_rate * total_deficit, effective_cap * available)
            if total_flow <= 0.0:
                continue
            dy[age, p.susceptible_index] -= total_flow
            for origin in range(p.n_origins):
                if deficits[origin] > 0.0:
                    dy[age, p.susceptible_indices[origin]] += total_flow * deficits[origin] / total_deficit

    # Importation is a supply-limited S-origin -> E transfer and therefore
    # population conserving.
    if p.importation_enabled:
        for age in range(p.n_age):
            susceptible_pool = 0.0
            for origin in range(p.n_origins):
                susceptible_pool += state[age, p.susceptible_indices[origin]]
            if susceptible_pool < 1e-12:
                susceptible_pool = 1e-12
            for origin in range(p.n_origins):
                susceptible_index = p.susceptible_indices[origin]
                requested = (
                    p.imported_by_age_per_day[age]
                    * state[age, susceptible_index]
                    / susceptible_pool
                )
                moved = min(requested, state[age, susceptible_index])
                dy[age, susceptible_index] -= moved
                dy[age, p.exposed_r_indices[origin]] += moved * p.imported_resistant_fraction
                dy[age, p.exposed_s_indices[origin]] += moved * (1.0 - p.imported_resistant_fraction)

    # Optional resistance prevalence anchor, applied identically to E, I_sym,
    # I_asym, and T pairs for each origin.
    if p.resistance_anchor_enabled:
        for age in range(p.n_age):
            for origin in range(p.n_origins):
                for pair in range(4):
                    if pair == 0:
                        sensitive_index = p.exposed_s_indices[origin]
                        resistant_index = p.exposed_r_indices[origin]
                    elif pair == 1:
                        sensitive_index = p.infectious_s_sym_indices[origin]
                        resistant_index = p.infectious_r_sym_indices[origin]
                    elif pair == 2:
                        sensitive_index = p.infectious_s_asym_indices[origin]
                        resistant_index = p.infectious_r_asym_indices[origin]
                    else:
                        sensitive_index = p.treated_s_indices[origin]
                        resistant_index = p.treated_r_indices[origin]
                    total = state[age, sensitive_index] + state[age, resistant_index]
                    desired_resistant = p.resistance_anchor_target * total
                    flow = p.resistance_anchor_rate_per_day * (
                        desired_resistant - state[age, resistant_index]
                    )
                    dy[age, sensitive_index] -= flow
                    dy[age, resistant_index] += flow

    # Fixed-profile or WPP demographic turnover.
    if p.demography_mode != 0:
        for age in range(p.n_age - 1):
            rate = p.aging_rates[age]
            for compartment in range(p.n_comp):
                flow = rate * state[age, compartment]
                dy[age, compartment] -= flow
                if p.maternal_exit_age_mask[age] and compartment == p.maternal_index:
                    dy[age + 1, p.susceptible_index] += flow
                else:
                    dy[age + 1, compartment] += flow

        oldest_total_flow = 0.0
        oldest = p.n_age - 1
        for compartment in range(p.n_comp):
            flow = p.aging_rates[oldest] * state[oldest, compartment]
            dy[oldest, compartment] -= flow
            oldest_total_flow += flow

        if p.demography_mode == 1:
            for compartment in range(p.n_comp):
                dy[0, compartment] += oldest_total_flow * p.birth_entry_weights[compartment]
        else:
            current_year = p.wpp_year_intercept + (t - p.wpp_time_origin) / 365.25
            lower, upper, fraction = _linear_interpolation_position(current_year, p.wpp_years)
            annual_births = p.wpp_births[lower]
            if upper != lower:
                annual_births += fraction * (p.wpp_births[upper] - p.wpp_births[lower])
            births_per_day = annual_births / 365.0
            for compartment in range(p.n_comp):
                dy[0, compartment] += births_per_day * p.birth_entry_weights[compartment]

            if p.wpp_nudge_rate_per_day > 0.0 and current_year >= p.wpp_start_year:
                for age in range(p.n_age):
                    target = p.wpp_population[age, lower]
                    if upper != lower:
                        target += fraction * (
                            p.wpp_population[age, upper] - p.wpp_population[age, lower]
                        )
                    if target < 1e-12:
                        target = 1e-12
                    current_population = 0.0
                    for compartment in range(p.n_comp):
                        current_population += state[age, compartment]
                    if current_population < 1e-12:
                        current_population = 1e-12
                    correction = p.wpp_nudge_rate_per_day * (target - current_population)
                    if abs(correction) < 1e-12:
                        continue
                    if correction > 0.0:
                        dy[age, p.susceptible_index] += correction
                    else:
                        for compartment in range(p.n_comp):
                            dy[age, compartment] += correction * state[age, compartment] / current_population

    return dy.reshape(p.n_age * p.n_comp)


def benchmark_fast_rhs(
    params: PreparedParameters,
    index: StateIndex,
    state: np.ndarray,
    *,
    t: float = 0.0,
    repeats: int = 200,
) -> dict[str, float]:
    """Return a small, dependency-free warmed RHS microbenchmark.

    This helper is intentionally not used as a pass/fail performance test;
    absolute timing depends on the host CPU and BLAS implementation.
    """

    if repeats <= 0:
        raise ValueError("repeats must be > 0")
    from src_python.model.ode_system import rhs as reference_rhs

    fast = build_fast_rhs(params, index)
    vector = np.asarray(state, dtype=np.float64).reshape(index.size)
    fast(t, vector)  # compile/warm
    reference_rhs(t, vector, params, index)  # warm partial Numba kernels

    start = perf_counter()
    for _ in range(repeats):
        reference_rhs(t, vector, params, index)
    reference_seconds = (perf_counter() - start) / repeats

    start = perf_counter()
    for _ in range(repeats):
        fast(t, vector)
    fast_seconds = (perf_counter() - start) / repeats
    return {
        "reference_seconds_per_call": reference_seconds,
        "fast_seconds_per_call": fast_seconds,
        "speedup": reference_seconds / fast_seconds,
    }


__all__ = [
    "FastRHS",
    "FastRHSParameters",
    "FastRHSUnsupportedError",
    "NUMBA_FAST_RHS_AVAILABLE",
    "benchmark_fast_rhs",
    "build_fast_rhs",
]
