"""M3 negative-log-posterior inference for the core2026 model.

The posterior combines

* an NB2 likelihood for the native surveillance intervals,
* a Dirichlet-multinomial likelihood for each year's published age groups
  (expected counts from ``expected_source_group_counts``), and
* the priors in ``config/core2026/priors.yaml``.

Optimisation works on unconstrained coordinates: positive parameters on the
log scale, bounded probabilities on the logit scale of their bounds, the
seasonal peak and reporting logits unchanged.  The prior is the density of
those coordinates, so the MAP estimate is the mode in the space the Laplace
and importance-sampling steps will use.

Only the twelve structural parameters require a new simulation; the nine
observation parameters reuse the cached simulation of the current structural
vector.  Central-difference gradients for the structural coordinates are
evaluated in parallel worker processes.  No fit-quality threshold is applied:
every start's objective, convergence state and comparison tables are
written for review.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logit

from src_python.pertussis.config import REPO_ROOT, PriorSpec, load_priors
from src_python.pertussis.grid import AgeGrid
from src_python.pertussis.inputs.bundle import BundleSettings, build_inputs
from src_python.pertussis.model.parameters import ModelParameters
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.model.tables import Calendar
from src_python.pertussis.observation import (
    AGE_BANDS,
    ObservationParameters,
    InvalidReportingProbability,
    validate_effective_reporting_probability,
    dirichlet_multinomial_nll,
    expected_source_group_counts,
    load_case_observations,
    logit_random_walk_nll,
    model_band_matrix,
    negative_binomial_nll,
    reporting_probabilities,
    source_age_years,
)
from src_python.pertussis.states import ACC_MILD, ACC_TYPICAL
from .structure_variants import StructureVariant, aic, get_variant


# Observation streams cover all eight programme countries.  The UK file is an
# England-and-Wales stream; scaling it by the corresponding share of the UK
# population keeps the model denominator and the retained surveillance scope
# aligned (the value is deliberately a named constant so sensitivity runs can
# override it without changing the likelihood).
SUPPORTED_COUNTRIES = (
    "Australia", "Brazil", "China", "Japan", "New_Zealand", "Sweden",
    "Thailand", "United_Kingdom",
)
SUPPORTED_REGIONS = ("England",)
UK_CATCHMENT_FRACTION = 0.89
DIAGNOSTIC_TIMELINE_PATH = REPO_ROOT / "data" / "raw" / "pertussis_diagnostic_standards_timeline.csv"

STRUCTURAL_NAMES = (
    "R0", "seasonal_amplitude", "seasonal_peak", "npi_scale",
    "importation_per_million_per_year", "natural_immunity_years", "boosting_kappa",
    "wP_years", "aP_years", "tdap_years", "sigma_aP", "pi_aP_waned",
)
REPORTING_PARAMETER_NAMES = tuple(f"reporting_logit_{band}" for band in AGE_BANDS)
OBSERVATION_NAMES = REPORTING_PARAMETER_NAMES + (
    "breakthrough_reporting_ratio", "nb_size", "dirichlet_concentration",
)
FREE_PARAMETER_NAMES = STRUCTURAL_NAMES + OBSERVATION_NAMES
# Extra observation coordinate of the S5 variant: shift of the 5-9 and 10-19
# reporting logits from the variant's start year (prior N(0, 1)).
SCHOOL_SHIFT_NAME = "school_detection_logit_shift"
SCHOOL_SHIFT_BOUNDS = (-3.0, 3.0)
SCHOOL_SHIFT_PRIOR_SD = 1.0
SCHOOL_BANDS = ("5-9", "10-19")
REPORTING_LOGIT_BOUNDS = (-8.0, 8.0)
LOGIT_LIMIT = 12.0
# The kernel's seasonal forcing peaks about a month before reported onsets.
SEASONAL_FORCING_LEAD = 0.08
FAILED_OBJECTIVE = 1e10


# --------------------------------------------------------------------------
# Parameter space
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Coordinate:
    """One free parameter: natural bounds, transform and prior on its coordinate."""

    name: str
    transform: str              # log | logit | identity
    lower: float
    upper: float
    prior: tuple[Any, ...]      # ("normal", mean_z, sd_z) | ("beta", a, b) | ("rw",)

    def to_natural(self, z: float) -> float:
        if self.transform == "log":
            return float(np.exp(z))
        if self.transform == "logit":
            return float(self.lower + (self.upper - self.lower) * expit(z))
        return float(z)

    def to_coordinate(self, x: float) -> float:
        if self.transform == "log":
            return float(np.log(x))
        if self.transform == "logit":
            s = (x - self.lower) / (self.upper - self.lower)
            return float(logit(np.clip(s, 1e-12, 1 - 1e-12)))
        return float(x)

    @property
    def coordinate_bounds(self) -> tuple[float, float]:
        if self.transform == "log":
            return float(np.log(self.lower)), float(np.log(self.upper))
        if self.transform == "logit":
            return -LOGIT_LIMIT, LOGIT_LIMIT
        return float(self.lower), float(self.upper)

    def prior_nll(self, z: float) -> float:
        kind = self.prior[0]
        if kind == "normal":
            _, mean, sd = self.prior
            return 0.5 * ((z - mean) / sd) ** 2 + np.log(sd)
        if kind == "beta":
            # Density of u = logit(s) when s ~ Beta(a, b): s^a (1-s)^b / B(a, b).
            _, a, b = self.prior
            return -(a * log_expit(z) + b * log_expit(-z))
        return 0.0


@dataclass
class ParameterSpace:
    coordinates: tuple[Coordinate, ...]
    reporting_prior: dict[str, float]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.coordinates)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [c.coordinate_bounds for c in self.coordinates]

    def natural(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=float)
        if z.shape != (len(self.coordinates),):
            raise ValueError(f"expected {len(self.coordinates)} coordinates")
        return {c.name: c.to_natural(v) for c, v in zip(self.coordinates, z)}

    def coordinates_of(self, values: dict[str, float]) -> np.ndarray:
        return np.asarray([c.to_coordinate(values[c.name]) for c in self.coordinates], dtype=float)

    def prior_nll(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        total = sum(c.prior_nll(v) for c, v in zip(self.coordinates, z))
        index = {c.name: i for i, c in enumerate(self.coordinates)}
        logits = np.asarray([z[index[n]] for n in REPORTING_PARAMETER_NAMES])
        total += logit_random_walk_nll(
            logits,
            first_mean=self.reporting_prior["first_band_mean_logit"],
            first_sd=self.reporting_prior["first_band_sd"],
            step_sd=self.reporting_prior["step_sd"],
        )
        return float(total)

    def prior_mode(self) -> np.ndarray:
        out = []
        for c in self.coordinates:
            if c.prior[0] == "normal":
                out.append(c.prior[1])
            elif c.prior[0] == "beta":
                _, a, b = c.prior
                out.append(float(np.log(a / b)))  # mode of logit(s) for Beta(a, b)
            elif c.prior[0] == "uniform":
                out.append(0.5 * (c.lower + c.upper))
            else:
                out.append(self.reporting_prior["first_band_mean_logit"])
        lo, hi = np.asarray(self.bounds).T
        return np.clip(np.asarray(out, dtype=float), lo, hi)

    def sample_prior(self, rng: np.random.Generator) -> np.ndarray:
        """Sample the joint coordinate prior conditional on hard bounds.

        The reporting random walk must be rejected as a *whole* when any
        reporting coordinate is outside its bounds. Componentwise clipping or
        resampling would change both its marginals and its dependence.
        """
        lo, hi = np.asarray(self.bounds, dtype=float).T
        for _ in range(10000):
            rw_first = rng.normal(self.reporting_prior["first_band_mean_logit"], self.reporting_prior["first_band_sd"])
            rw = rw_first + np.concatenate([[0.0], np.cumsum(rng.normal(0.0, self.reporting_prior["step_sd"], len(AGE_BANDS) - 1))])
            rw_iter = iter(rw)
            out = []
            for c in self.coordinates:
                if c.prior[0] == "normal":
                    out.append(rng.normal(c.prior[1], c.prior[2]))
                elif c.prior[0] == "beta":
                    out.append(float(logit(rng.beta(c.prior[1], c.prior[2]))))
                elif c.prior[0] == "uniform":
                    out.append(rng.uniform(c.lower, c.upper))
                elif c.prior[0] == "rw":
                    out.append(next(rw_iter))
                else:
                    raise ValueError(f"unsupported prior: {c.prior[0]}")
            proposal = np.asarray(out, dtype=float)
            if np.all(np.isfinite(proposal)) and np.all((proposal >= lo) & (proposal <= hi)):
                return proposal
        raise RuntimeError("could not draw a prior within coordinate bounds")


def _coordinate(name: str, spec: PriorSpec, seasonal_peak_mean: float) -> Coordinate:
    lo, hi = spec.bounds if spec.bounds else (None, None)
    if spec.distribution == "lognormal":
        return Coordinate(name, "log", lo, hi, ("normal", float(np.log(spec.settings["median"])), float(spec.settings["log_sd"])))
    if spec.distribution == "beta":
        lo, hi = spec.bounds or (0.0, 1.0)
        return Coordinate(name, "logit", lo, hi, ("beta", float(spec.settings["a"]), float(spec.settings["b"])))
    if spec.distribution == "normal":
        mean = seasonal_peak_mean if spec.settings["mean"] == "country" else float(spec.settings["mean"])
        return Coordinate(name, "identity", lo, hi, ("normal", float(mean), float(spec.settings["sd"])))
    raise ValueError(f"unsupported prior for {name}: {spec.distribution}")


def build_parameter_space(
    specs: dict[str, PriorSpec],
    seasonal_peak_mean: float,
    school_detection_shift: bool = False,
) -> ParameterSpace:
    coords: list[Coordinate] = []
    for name in FREE_PARAMETER_NAMES:
        if name in REPORTING_PARAMETER_NAMES:
            coords.append(Coordinate(name, "identity", *REPORTING_LOGIT_BOUNDS, ("rw",)))
        else:
            spec = specs[name]
            if not spec.estimated:
                raise ValueError(f"{name} is fixed in priors.yaml but listed as free")
            coords.append(_coordinate(name, spec, seasonal_peak_mean))
    if school_detection_shift:
        coords.append(Coordinate(SCHOOL_SHIFT_NAME, "identity", *SCHOOL_SHIFT_BOUNDS, ("normal", 0.0, SCHOOL_SHIFT_PRIOR_SD)))
    rw = specs["reporting_logit"].settings
    return ParameterSpace(
        tuple(coords),
        {k: float(rw[k]) for k in ("first_band_mean_logit", "first_band_sd", "step_sd")},
    )


def observed_seasonal_peak(case: pd.DataFrame, exclude_years: tuple[int, ...] = (2020, 2021, 2022)) -> float:
    """First-harmonic peak (fraction of the year) of within-year relative counts."""

    frame = case.copy()
    # Annual interval midpoints carry no observed seasonal phase. Mixing them
    # with genuine subannual counts would manufacture a midyear peak.
    duration = (pd.to_datetime(frame.period_end) - pd.to_datetime(frame.period_start)).dt.days
    frame = frame.loc[duration < 300]
    start = pd.to_datetime(frame["period_start"])
    frame = frame[~start.dt.year.isin(exclude_years)]
    if frame.empty:
        raise ValueError('No eligible within-year observations for seasonal phase')
    start = pd.to_datetime(frame["period_start"])
    mid = start + (pd.to_datetime(frame["period_end"]) - start) / 2
    frac = (mid.dt.dayofyear.to_numpy() - 0.5) / 365.25
    rel = frame["reported_cases"].to_numpy(float)
    yearly = frame.groupby(start.dt.year)["reported_cases"].transform("mean").to_numpy(float)
    rel = np.divide(rel, yearly, out=np.zeros_like(rel), where=yearly > 0)
    angle = 2 * np.pi * frac
    return float(np.arctan2((rel * np.sin(angle)).sum(), (rel * np.cos(angle)).sum()) / (2 * np.pi) % 1.0)


# --------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceSettings:
    history_start_year: int = 1970
    spinup_years: int = 40
    last_year: int = 2025
    reference_year: int = 2019
    n_starts: int = 4
    max_iterations: int = 300
    gradient_step: float = 1e-3
    n_workers: int | None = None
    seed: int = 20261017
    variant: str = "baseline"
    train_start_year: int | None = None
    train_end_year: int | None = None
    holdout_years: tuple[int, ...] = ()
    # Immutable clinical assumptions. Estimated structural coordinates cannot
    # be overwritten, and each problem owns its simulation cache.
    model_overrides: tuple[tuple[str, float], ...] = ()
    age_observations_path: str | None = None
    case_observations_path: str | None = None
    seasonal_phase_prior: str = 'auto'
    childhood_coverage_variant: str = 'inherited_proxy'
    programme_path: str | None = None
    # Required if any selected country timeline contains a non-national row.
    # national_only uses the existing 1.0 fallback for excluded periods; that
    # fallback is an assumption, not an observed national detection rate.
    diagnostic_geography: str | None = None


@dataclass
class InferenceFit:
    country: str
    parameter_values: dict[str, float]
    coordinates: np.ndarray
    objective: float
    start_index: int
    nfev: int
    nit: int
    success: bool
    message: str
    total_comparison: pd.DataFrame
    age_comparison: pd.DataFrame
    diagnostics: dict[str, Any]
    trace: list[float] = field(default_factory=list)


def _record_days(calendar: Calendar, frames: Iterable[pd.DataFrame]) -> np.ndarray:
    boundaries: list[int] = [0]
    for frame in frames:
        for column in ("period_start", "period_end"):
            if column in frame:
                for value in frame[column]:
                    boundaries.append(calendar.day(pd.Timestamp(value).date().isoformat()))
    for year in range(calendar.first_year, calendar.first_year + calendar.n_years + 1):
        boundaries.append(calendar.year_start_day(year))
    return np.unique(np.asarray(boundaries, dtype=np.int64))


def _split_interval_by_calendar_year(start: Any, end: Any) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Split a half-open observation interval at each calendar-year boundary."""
    left, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(left) or pd.isna(end) or left >= end:
        raise ValueError('observation interval must have increasing valid dates')
    segments = []
    while left < end:
        right = min(end, pd.Timestamp(year=left.year+1, month=1, day=1))
        segments.append((left, right, left.year))
        left = right
    return segments


def _diagnostic_multiplier_frame(country: str, path: Path = DIAGNOSTIC_TIMELINE_PATH,
                                 geographic_policy: str | None = None) -> pd.DataFrame:
    """Return dated detection multipliers, requiring an explicit geographic policy.

    For national_only, omitted intervals use the ordinary 1.0 reference fallback.
    legacy_all explicitly reproduces historical application of subnational rows.
    Neither option establishes an empirically supported national multiplier.
    """
    if geographic_policy not in (None, 'national_only', 'legacy_all'):
        raise ValueError('diagnostic geography must be national_only or legacy_all')
    if not path.exists():
        return pd.DataFrame(columns=("period_start", "period_end", "multiplier"))
    frame = pd.read_csv(path)
    frame = frame[frame["country"].astype(str).eq(country)].copy()
    if frame.empty:
        return pd.DataFrame(columns=("period_start", "period_end", "multiplier"))
    if 'geographic_scope' not in frame or frame['geographic_scope'].isna().any():
        raise ValueError(f'{country}: diagnostic timeline lacks geographic scope')
    local = ~frame['geographic_scope'].astype(str).str.strip().eq('national')
    if local.any() and geographic_policy is None:
        raise ValueError(f'{country}: diagnostic timeline has non-national entries; explicitly choose '
                         'diagnostic_geography=national_only or legacy_all')
    if geographic_policy == 'national_only':
        frame = frame.loc[~local].copy()
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    # Timeline end dates are inclusive.  Treat them as midnight after the
    # date, which makes overlap weighting unambiguous for weekly intervals.
    frame["period_end"] = pd.to_datetime(frame["period_end"]) + pd.Timedelta(days=1)
    frame["multiplier"] = pd.to_numeric(frame["relative_detection_prior_mean"], errors="coerce")
    return frame[["period_start", "period_end", "multiplier"]].dropna()


def _dated_multipliers(starts: Iterable[Any], ends: Iterable[Any], timeline: pd.DataFrame) -> np.ndarray:
    """Overlap-weight a dated detection timeline onto arbitrary intervals."""

    starts = pd.to_datetime(list(starts))
    ends = pd.to_datetime(list(ends))
    out = np.ones(len(starts), dtype=float)
    if timeline.empty:
        return out
    for i, (start, end) in enumerate(zip(starts, ends)):
        duration = max((end - start).total_seconds(), 1.0)
        weighted = 0.0
        covered = 0.0
        for row in timeline.itertuples(index=False):
            left, right = max(start, row.period_start), min(end, row.period_end)
            overlap = max((right - left).total_seconds(), 0.0)
            weighted += overlap * float(row.multiplier)
            covered += overlap
        # Unspecified periods use the modern reference scale of one.
        out[i] = (weighted + max(duration - covered, 0.0)) / duration
    return out


# Profiles without a published annual age composition.  They have no age
# likelihood: their reporting-shape information must come from priors
# informed by the other profiles, never from another country's counts.
NO_AGE_DATA_COUNTRIES = ("Brazil", "Thailand")


class Core2026Inference:
    """Negative-log-posterior problem for one country."""

    failed_objective = FAILED_OBJECTIVE

    def __init__(self, country: str, settings: InferenceSettings = InferenceSettings(), priors_path: Path | None = None, variant: str | None = None):
        if country not in SUPPORTED_COUNTRIES + SUPPORTED_REGIONS:
            raise ValueError(f"unsupported profile {country!r}; choose from {SUPPORTED_COUNTRIES + SUPPORTED_REGIONS}")
        self.country = country
        self.settings = settings
        self.variant: StructureVariant = get_variant(variant or settings.variant)
        self.priors, self.prior_references = load_priors(priors_path) if priors_path else load_priors()
        if country == "England" and (not settings.case_observations_path or not settings.age_observations_path):
            raise ValueError("England requires explicit geographically matched case and age sources")
        self.case = load_case_observations(settings.case_observations_path or country).intervals.copy()
        if country == "England" and ("geographic_scope" not in self.case
                or not self.case.geographic_scope.eq("England").all()):
            raise ValueError("England case observations must identify England geography")
        # The legacy workbook has no Brazil/Thailand age rows. A separately
        # verified explicit source must be usable once it has been acquired.
        no_age_source = country in NO_AGE_DATA_COUNTRIES and not settings.age_observations_path
        self.age_years = [] if no_age_source else source_age_years(
            country, path=Path(settings.age_observations_path) if settings.age_observations_path else None)
        self.model_overrides = dict(settings.model_overrides)
        if len(self.model_overrides) != len(settings.model_overrides):
            raise ValueError("duplicate fixed model override")
        if set(self.model_overrides) & set(STRUCTURAL_NAMES + ("vaccine_waning_shape", "sigma_tdap")):
            raise ValueError("fixed model overrides cannot replace estimated coordinates or waning structure")
        ModelParameters().with_updates(**self.model_overrides).sigma_pi()
        if (settings.train_start_year is not None and settings.train_end_year is not None
                and settings.train_start_year > settings.train_end_year):
            raise ValueError("training year bounds are reversed")
        def train_year(year):
            return ((settings.train_start_year is None or year >= settings.train_start_year)
                    and (settings.train_end_year is None or year <= settings.train_end_year)
                    and year not in settings.holdout_years)
        case_start = pd.to_datetime(self.case.period_start)
        # A native interval crossing an excluded calendar boundary must not
        # leak into training. Classify by both endpoints, with an exclusive end.
        case_end = pd.to_datetime(self.case.period_end) - pd.Timedelta(nanoseconds=1)
        self.case_train_mask = np.asarray([all(train_year(y) for y in range(s.year, e.year+1))
            for s,e in zip(case_start, case_end)], dtype=bool)
        self.age_train_mask = np.asarray([train_year(item.year) for item in self.age_years], dtype=bool)
        if not self.case_train_mask.any():
            raise ValueError("training selection contains no complete case intervals")
        terminal = int(pd.to_datetime(self.case["period_end"]).dt.year.max())
        age_terminal = max((item.year for item in self.age_years), default=terminal)
        last_year = max(int(settings.last_year), terminal, age_terminal)
        bundle = BundleSettings(
            history_start_year=settings.history_start_year,
            spinup_years=settings.spinup_years,
            last_year=last_year,
            reference_year=settings.reference_year,
        )
        if settings.childhood_coverage_variant == 'england_cover_cohort':
            if settings.programme_path:raise ValueError('custom programme cannot be combined with England COVER builder')
            if country != 'England':raise ValueError('England COVER variant requires England')
            from ..inputs.england_cover import build_england_cover_inputs
            self.inputs, self.audit = build_england_cover_inputs(bundle)
        elif settings.childhood_coverage_variant == 'inherited_proxy':
            from ..inputs.programme import load_programme
            programme = load_programme(Path(settings.programme_path)) if settings.programme_path else None
            self.inputs, self.audit = build_inputs(country, bundle, programme=programme)
        else:
            raise ValueError('unknown childhood_coverage_variant')
        self.simulator = Simulator(self.inputs)
        self.calendar = self.inputs.calendar
        self._detection_timeline = _diagnostic_multiplier_frame(
            country, geographic_policy=settings.diagnostic_geography)
        # The probability bound applies over the complete simulated calendar,
        # independently of which outcome observations are held out. Reporting
        # logits can vary by year, so keep each year's highest active multiplier.
        self._reporting_domain_years = np.arange(self.calendar.first_year,
            self.calendar.first_year + self.calendar.n_years, dtype=int)
        timeline_bounds = set(self._detection_timeline.period_start) | set(self._detection_timeline.period_end)
        domain_multipliers = []
        for year in self._reporting_domain_years:
            start, end = pd.Timestamp(year=int(year), month=1, day=1), pd.Timestamp(year=int(year)+1, month=1, day=1)
            boundaries = sorted({start, end} | {d for d in timeline_bounds if start < d < end})
            domain_multipliers.append(float(np.max(_dated_multipliers(boundaries[:-1], boundaries[1:], self._detection_timeline))))
        self._reporting_domain_multipliers = np.asarray(domain_multipliers, dtype=float)
        self.record_days = _record_days(self.calendar, [self.case])
        # Retain exact accumulation at diagnostic-regime changes inside age years.
        extra_days = []
        for item in self.age_years:
            start, end = pd.Timestamp(f"{item.year}-01-01"), pd.Timestamp(f"{item.year + 1}-01-01")
            for boundary in set(self._detection_timeline.period_start) | set(self._detection_timeline.period_end):
                if start < boundary < end:
                    extra_days.append(self.calendar.day(boundary.date().isoformat()))
        # Case intervals that cross a detection change also need a recorded
        # accumulation point, not a time-averaged multiplier over the window.
        detection_boundaries = set(self._detection_timeline.period_start) | set(self._detection_timeline.period_end)
        for start, end in zip(self.case["period_start"], self.case["period_end"]):
            for boundary in detection_boundaries:
                if start < boundary < end:
                    extra_days.append(self.calendar.day(boundary.date().isoformat()))
        self.record_days = np.unique(np.concatenate((self.record_days, np.asarray(extra_days, dtype=np.int64))))
        self.base_y0, self.base_acc0 = self.simulator.initial_state()
        grid = AgeGrid()
        self.bin_to_band = model_band_matrix(grid)
        training_case = self.case.loc[self.case_train_mask]
        annual_only = bool((training_case.interval_days >= 300).all())
        if settings.seasonal_phase_prior not in ('auto', 'uniform'):
            raise ValueError('seasonal_phase_prior must be auto or uniform')
        uniform_phase = annual_only or settings.seasonal_phase_prior == 'uniform'
        self.seasonal_incidence_peak = None if uniform_phase else observed_seasonal_peak(training_case)
        self.space = build_parameter_space(
            self.priors,
            0.5 if uniform_phase else (self.seasonal_incidence_peak - SEASONAL_FORCING_LEAD) % 1.0,
            school_detection_shift=self.variant.school_detection_start_year is not None,
        )
        # Annual observations contain no within-year phase information. Do
        # not mistake the interval midpoint for an observed seasonal peak.
        self.seasonal_prior_basis = "uniform_phase_annual_data" if annual_only else "training_within_year_counts"
        if settings.seasonal_phase_prior == 'uniform':
            self.seasonal_prior_basis = 'explicit_uniform_phase'
        if uniform_phase:
            self.space.coordinates = tuple(
                Coordinate(c.name, c.transform, c.lower, c.upper, ("uniform",))
                if c.name == "seasonal_peak" else c for c in self.space.coordinates)
        # Coordinates a variant switches off (kappa without boosting) stay in
        # the space at their prior but are not counted as fitted parameters.
        self.n_effective_parameters = len(self.space.names) - int(self.variant.disable_natural_boosting)
        self.structural_index = np.arange(len(STRUCTURAL_NAMES))
        self.observation_index = np.arange(len(STRUCTURAL_NAMES), len(self.space.names))
        by_day = {int(day): i for i, day in enumerate(self.record_days)}
        self._interval_rows = np.asarray([
            (by_day[self.calendar.day(pd.Timestamp(s).date().isoformat())],
             by_day[self.calendar.day(pd.Timestamp(e).date().isoformat())])
            for s, e in zip(self.case["period_start"], self.case["period_end"])
        ])
        split = []
        for parent, (start, end) in enumerate(zip(self.case['period_start'], self.case['period_end'])):
            segments = _split_interval_by_calendar_year(start, end)
            if len(segments) > 1:
                split.extend((parent, by_day[self.calendar.day(left.date().isoformat())],
                              by_day[self.calendar.day(right.date().isoformat())], year)
                             for left, right, year in segments)
        self._split_parent = np.asarray([x[0] for x in split], dtype=int)
        self._split_left = np.asarray([x[1] for x in split], dtype=int)
        self._split_right = np.asarray([x[2] for x in split], dtype=int)
        self._split_year = np.asarray([x[3] for x in split], dtype=int)
        # Detection-crossing windows are split at both surveillance changes
        # and calendar years, so each piece receives both applicable weights.
        detection_split = []
        for parent, (start, end) in enumerate(zip(self.case['period_start'], self.case['period_end'])):
            interior = {d for d in detection_boundaries if start < d < end}
            if not interior:
                continue
            years = {pd.Timestamp(year=y, month=1, day=1) for y in range(start.year + 1, end.year + 1)}
            boundaries = sorted({start, end} | interior | {d for d in years if start < d < end})
            multipliers = _dated_multipliers(boundaries[:-1], boundaries[1:], self._detection_timeline)
            detection_split.extend((parent, by_day[self.calendar.day(left.date().isoformat())],
                                    by_day[self.calendar.day(right.date().isoformat())],
                                    left.year, float(multiplier))
                                   for left, right, multiplier in zip(boundaries[:-1], boundaries[1:], multipliers))
        self._detection_split_parent = np.asarray([x[0] for x in detection_split], dtype=int)
        self._detection_split_left = np.asarray([x[1] for x in detection_split], dtype=int)
        self._detection_split_right = np.asarray([x[2] for x in detection_split], dtype=int)
        self._detection_split_year = np.asarray([x[3] for x in detection_split], dtype=int)
        self._detection_split_multiplier = np.asarray([x[4] for x in detection_split], dtype=float)
        self._age_rows = [
            (by_day[self.calendar.year_start_day(item.year)], by_day[self.calendar.year_start_day(item.year + 1)])
            for item in self.age_years
        ]
        self._age_maps = [item.mapping(grid) for item in self.age_years]
        # Only years with within-year detection changes require segmentation.
        self._age_detection_segments = {}
        for i, item in enumerate(self.age_years):
            start, end = pd.Timestamp(f"{item.year}-01-01"), pd.Timestamp(f"{item.year + 1}-01-01")
            boundaries = sorted({start, end} | {d for d in set(self._detection_timeline.period_start) | set(self._detection_timeline.period_end) if start < d < end})
            if len(boundaries) > 2:
                multipliers = _dated_multipliers(boundaries[:-1], boundaries[1:], self._detection_timeline)
                self._age_detection_segments[i] = [(by_day[self.calendar.day(left.date().isoformat())],
                    by_day[self.calendar.day(right.date().isoformat())], float(multiplier))
                    for left, right, multiplier in zip(boundaries[:-1], boundaries[1:], multipliers)]
        self._observed_cases = self.case["reported_cases"].to_numpy(float)
        self._nb_exposure = self.case["interval_days"].to_numpy(float) / 365.2425
        self._interval_years = pd.to_datetime(self.case["period_start"]).dt.year.to_numpy()
        self._interval_detection = _dated_multipliers(
            self.case["period_start"], self.case["period_end"], self._detection_timeline
        )
        self._age_detection = np.asarray([
            _dated_multipliers(
                [pd.Timestamp(f"{item.year}-01-01")],
                [pd.Timestamp(f"{item.year + 1}-01-01")],
                self._detection_timeline,
            )[0]
            for item in self.age_years
        ])
        self.catchment_fraction = UK_CATCHMENT_FRACTION if country == "United_Kingdom" else 1.0
        self._cache: OrderedDict[tuple, dict[str, np.ndarray]] = OrderedDict()
        self.n_simulations = 0
        self.domain_rejections = 0
        self.failures: list[str] = []

    # ---- simulation ------------------------------------------------------

    def _structural_values(self, values: dict[str, float]) -> tuple[float, ...]:
        return tuple(values[name] for name in STRUCTURAL_NAMES)

    def _simulation(self, values: dict[str, float]) -> dict[str, np.ndarray]:
        key = tuple(round(v, 12) for v in self._structural_values(values))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        params = ModelParameters(
            R0=values["R0"], seasonal_amplitude=values["seasonal_amplitude"],
            seasonal_peak=values["seasonal_peak"], npi_scale=values["npi_scale"],
            importation_per_million_per_year=values["importation_per_million_per_year"],
            natural_immunity_years=values["natural_immunity_years"], boosting_kappa=(0.0 if self.variant.disable_natural_boosting else values["boosting_kappa"]),
            vaccine_waning_shape=self.variant.vaccine_waning_shape,
            wP_years=values["wP_years"], aP_years=values["aP_years"], tdap_years=values["tdap_years"],
            sigma_aP=values["sigma_aP"], sigma_tdap=values["sigma_aP"], pi_aP_waned=values["pi_aP_waned"],
        )
        params = params.with_updates(**self.model_overrides)
        result = self.simulator.run(params, self.base_y0, self.base_acc0, 0, self.calendar.n_days, record_days=self.record_days)
        self.n_simulations += 1
        acc = result.acc_records
        left, right = self._interval_rows[:, 0], self._interval_rows[:, 1]
        out = {
            "interval_typical": acc[right, :, ACC_TYPICAL] - acc[left, :, ACC_TYPICAL],
            "interval_mild": acc[right, :, ACC_MILD] - acc[left, :, ACC_MILD],
            "split_typical": acc[self._split_right, :, ACC_TYPICAL] - acc[self._split_left, :, ACC_TYPICAL],
            "split_mild": acc[self._split_right, :, ACC_MILD] - acc[self._split_left, :, ACC_MILD],
            "detection_split_typical": acc[self._detection_split_right, :, ACC_TYPICAL] - acc[self._detection_split_left, :, ACC_TYPICAL],
            "detection_split_mild": acc[self._detection_split_right, :, ACC_MILD] - acc[self._detection_split_left, :, ACC_MILD],
            "year_typical": np.asarray([acc[r, :, ACC_TYPICAL] - acc[l, :, ACC_TYPICAL] for l, r in self._age_rows]),
            "year_mild": np.asarray([acc[r, :, ACC_MILD] - acc[l, :, ACC_MILD] for l, r in self._age_rows]),
            "age_segment_typical": {i: np.asarray([acc[r, :, ACC_TYPICAL] - acc[l, :, ACC_TYPICAL] for l, r, _ in segments]) for i, segments in self._age_detection_segments.items()},
            "age_segment_mild": {i: np.asarray([acc[r, :, ACC_MILD] - acc[l, :, ACC_MILD] for l, r, _ in segments]) for i, segments in self._age_detection_segments.items()},
        }
        # Countries without age-composition data have empty annual arrays.
        # Their interval likelihood is still valid; absent observations are
        # not a failed simulation or a zero observed age distribution.
        arrays = [v for value in out.values() for v in (value.values() if isinstance(value, dict) else [value])]
        if not all(np.isfinite(v).all() and (v.size == 0 or v.min() >= -1e-6) for v in arrays):
            raise FloatingPointError("simulation produced negative or non-finite onsets")
        self._cache[key] = out
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return out

    # ---- likelihood ------------------------------------------------------

    def _reporting_logits(self, values: dict[str, float], year: int) -> np.ndarray:
        """Band reporting logits in force in a calendar year (S5 shift included)."""

        logits = np.asarray([values[n] for n in REPORTING_PARAMETER_NAMES], dtype=float)
        start = self.variant.school_detection_start_year
        if start is not None and year >= start:
            logits = logits.copy()
            for band in SCHOOL_BANDS:
                logits[AGE_BANDS.index(band)] += values[SCHOOL_SHIFT_NAME]
        return logits

    def _validate_reporting_domain(self, values: dict[str, float]) -> None:
        years, multipliers = self._reporting_domain_years, self._reporting_domain_multipliers
        if (getattr(type(self), '_reporting_logits', None) is Core2026Inference._reporting_logits
                and '_reporting_logits' not in vars(self)):
            # The base mapping has at most one school-reporting transition.
            # Subclasses may vary by year, so they retain the general path.
            start = self.variant.school_detection_start_year
            groups = (np.ones(len(years), dtype=bool),) if start is None else (years < start, years >= start)
            for group in groups:
                if group.any():
                    year = int(years[group][0])
                    rates = reporting_probabilities(self._reporting_logits(values, year))
                    validate_effective_reporting_probability(rates, multipliers[group])
        else:
            for year, multiplier in zip(years, multipliers):
                rates = reporting_probabilities(self._reporting_logits(values, int(year)))
                validate_effective_reporting_probability(rates, multiplier)

    def sample_admissible_prior(self, rng: np.random.Generator, max_attempts: int = 10000) -> np.ndarray:
        """Draw from the prior conditional on the valid reporting domain."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        for _ in range(max_attempts):
            proposal = self.space.sample_prior(rng)
            try:
                self._validate_reporting_domain(self.space.natural(proposal))
            except InvalidReportingProbability:
                continue
            return proposal
        raise RuntimeError("could not draw an admissible reporting prior")

    def _components(self, values: dict[str, float], sim: dict[str, np.ndarray]) -> dict[str, Any]:
        Core2026Inference._validate_reporting_domain(self, values)
        ratio = values["breakthrough_reporting_ratio"]
        true_intervals = sim["interval_typical"] + ratio * sim["interval_mild"]
        expected = np.empty(true_intervals.shape[0])
        # Reporting probabilities act on model bins in both likelihood terms,
        # so a period-specific change (S5) is applied identically to totals
        # and to every published age group, whatever its edges.
        for year in np.unique(self._interval_years):
            rows = self._interval_years == year
            weight = self.bin_to_band @ reporting_probabilities(self._reporting_logits(values, int(year)))
            expected[rows] = true_intervals[rows] @ weight
        # A report window may cross Jan 1. Accumulate expected onsets in each
        # calendar segment using the reporting logits actually in force then.
        if self._split_parent.size:
            split_true = sim['split_typical'] + ratio*sim['split_mild']
            split_expected = np.empty(len(self._split_parent))
            for year in np.unique(self._split_year):
                rows = self._split_year == year
                weight = self.bin_to_band @ reporting_probabilities(self._reporting_logits(values, int(year)))
                split_expected[rows] = split_true[rows] @ weight
            split_sum = np.bincount(self._split_parent, weights=split_expected, minlength=len(expected))
            affected = np.unique(self._split_parent)
            expected[affected] = split_sum[affected]
        expected = expected * self._interval_detection
        if self._detection_split_parent.size:
            segment_true = sim['detection_split_typical'] + ratio * sim['detection_split_mild']
            segment_expected = np.empty(len(self._detection_split_parent))
            for year in np.unique(self._detection_split_year):
                rows = self._detection_split_year == year
                weight = self.bin_to_band @ reporting_probabilities(self._reporting_logits(values, int(year)))
                segment_expected[rows] = (segment_true[rows] @ weight) * self._detection_split_multiplier[rows]
            sums = np.bincount(self._detection_split_parent, weights=segment_expected, minlength=len(expected))
            affected = np.unique(self._detection_split_parent)
            expected[affected] = sums[affected]
        expected = expected * self.catchment_fraction
        train = self.case_train_mask
        total_nll = negative_binomial_nll(self._observed_cases[train], expected[train],
            values["nb_size"] * self._nb_exposure[train])
        holdout_total_nll = negative_binomial_nll(self._observed_cases[~train], expected[~train],
            values["nb_size"] * self._nb_exposure[~train])
        observed_vectors, probability_vectors, expected_vectors = [], [], []
        for i, item in enumerate(self.age_years):
            obs_params = ObservationParameters(
                reporting_logits=self._reporting_logits(values, item.year), mild_reporting_ratio=ratio,
            )
            exp_counts = expected_source_group_counts(
                sim["year_typical"][i][None, :], sim["year_mild"][i][None, :],
                model_bands=self.bin_to_band, source_bands=self._age_maps[i], params=obs_params,
                detection_multiplier=self._age_detection[i], catchment_fraction=self.catchment_fraction,
            )[0]
            if i in self._age_detection_segments:
                exp_counts = np.zeros_like(exp_counts)
                for j, (_, _, multiplier) in enumerate(self._age_detection_segments[i]):
                    exp_counts += expected_source_group_counts(
                        sim["age_segment_typical"][i][j][None, :], sim["age_segment_mild"][i][j][None, :],
                        model_bands=self.bin_to_band, source_bands=self._age_maps[i], params=obs_params,
                        detection_multiplier=multiplier, catchment_fraction=self.catchment_fraction,
                    )[0]
            expected_vectors.append(exp_counts)
            observed_vectors.append(np.asarray(item.counts, dtype=float))
            probability_vectors.append(exp_counts / max(float(exp_counts.sum()), 1e-300))
        age_nll = 0.0
        holdout_age_nll = 0.0
        for active, obs, prob in zip(self.age_train_mask, observed_vectors, probability_vectors):
            score = dirichlet_multinomial_nll(obs, prob, values["dirichlet_concentration"])
            if active:
                age_nll += score
            else:
                holdout_age_nll += score
        return {
            "total_nll": float(total_nll), "age_nll": float(age_nll),
            "expected_intervals": expected, "expected_age": expected_vectors,
            "variant": self.variant.name,
            "holdout_total_nll": float(holdout_total_nll),
            "holdout_age_nll": float(holdout_age_nll),
        }

    def objective_z(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        try:
            values = self.space.natural(z)
            # Reject outside the domain before the expensive transmission run.
            self._validate_reporting_domain(values)
            parts = self._components(values, self._simulation(values))
            value = parts["total_nll"] + parts["age_nll"] + self.space.prior_nll(z)
            if not np.isfinite(value):
                raise FloatingPointError("non-finite posterior")
            return float(value)
        except InvalidReportingProbability:
            self.domain_rejections += 1
            return FAILED_OBJECTIVE
        except (FloatingPointError, ValueError, OverflowError) as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            return FAILED_OBJECTIVE

    # Kept for callers that pass natural-scale vectors.
    def objective(self, theta: np.ndarray, *, return_tables: bool = False):
        z = self.space.coordinates_of(dict(zip(self.space.names, np.asarray(theta, dtype=float))))
        if not return_tables:
            return self.objective_z(z)
        return self.evaluate(z)

    def evaluate(self, z: np.ndarray) -> tuple[float, pd.DataFrame, pd.DataFrame, dict[str, float]]:
        values = self.space.natural(z)
        self._validate_reporting_domain(values)
        parts = self._components(values, self._simulation(values))
        prior = self.space.prior_nll(z)
        total = pd.DataFrame({
            "period_start": self.case["period_start"].to_numpy(),
            "period_end": self.case["period_end"].to_numpy(),
            "observed_cases": self._observed_cases,
            "expected_cases": parts["expected_intervals"],
            "in_training": self.case_train_mask,
        })
        rows = []
        for active, item, exp_counts in zip(self.age_train_mask, self.age_years, parts["expected_age"]):
            obs = np.asarray(item.counts, dtype=float)
            for label, o, e in zip(item.labels, obs, exp_counts):
                rows.append({
                    "country": self.country, "year": item.year, "source_age_group": label,
                    "observed_cases": o, "expected_cases": e,
                    "observed_share": o / max(obs.sum(), 1e-300),
                    "expected_share": e / max(exp_counts.sum(), 1e-300),
                    "in_training": bool(active),
                })
        posterior = parts["total_nll"] + parts["age_nll"] + prior
        age_table = pd.DataFrame(rows, columns=['country', 'year', 'source_age_group',
            'observed_cases', 'expected_cases', 'observed_share', 'expected_share', 'in_training'])
        return float(posterior), total, age_table, {
            "total_nll": parts["total_nll"], "age_nll": parts["age_nll"], "prior_nll": prior,
            "holdout_total_nll": parts["holdout_total_nll"],
            "holdout_age_nll": parts["holdout_age_nll"],
        }

    # ---- fitting ---------------------------------------------------------

    def starts(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.settings.seed)
        mode = self.space.prior_mode()
        try:
            self._validate_reporting_domain(self.space.natural(mode))
        except InvalidReportingProbability:
            mode = self.sample_admissible_prior(rng)
        out = [mode]
        for _ in range(max(0, int(self.settings.n_starts) - 1)):
            out.append(self.sample_admissible_prior(rng))
        return out

    def fit(self, gradient: "ParallelGradient | None" = None) -> list[InferenceFit]:
        own = gradient is None
        gradient = gradient or ParallelGradient(self, self.settings.n_workers, self.settings.gradient_step)
        fits: list[InferenceFit] = []
        try:
            for index, start in enumerate(self.starts()):
                fits.append(self._fit_from(index, start, gradient))
        finally:
            if own:
                gradient.close()
        return fits

    def _fit_from(self, index: int, start: np.ndarray, gradient: "ParallelGradient", progress=None) -> InferenceFit:
        trace: list[float] = []
        t0 = time.time()
        failures_before = len(self.failures)
        def record_progress(zk):
            trace.append(self.objective_z(zk))
            if progress is not None:
                progress(len(trace), zk, trace[-1])
        result = minimize(
            gradient.value_and_grad, start, jac=True, method="L-BFGS-B", bounds=self.space.bounds,
            callback=record_progress,
            options={"maxiter": int(self.settings.max_iterations), "maxcor": 20, "gtol": 1e-4, "ftol": 1e-10},
        )
        posterior, total, age, components = self.evaluate(result.x)
        lo, hi = np.asarray(self.space.bounds).T
        projected = np.where(
            (np.isclose(result.x, lo) & (result.jac > 0)) | (np.isclose(result.x, hi) & (result.jac < 0)),
            0.0, result.jac,
        )
        return InferenceFit(
            country=self.country, parameter_values=self.space.natural(result.x), coordinates=result.x,
            objective=posterior, start_index=index, nfev=int(result.nfev), nit=int(result.nit),
            success=bool(result.success), message=str(result.message),
            total_comparison=total, age_comparison=age, trace=trace,
            diagnostics={
                **components,
                "projected_gradient_norm": float(np.linalg.norm(projected)),
                "max_abs_projected_gradient": float(np.max(np.abs(projected))),
                "at_lower_bound": [n for n, v, l in zip(self.space.names, result.x, lo) if np.isclose(v, l)],
                "at_upper_bound": [n for n, v, h in zip(self.space.names, result.x, hi) if np.isclose(v, h)],
                "failed_evaluations": len(self.failures) - failures_before,
                "seconds": time.time() - t0,
                "variant": self.variant.name,
                "n_parameters": self.n_effective_parameters,
                "aic": aic(components["total_nll"] + components["age_nll"], self.n_effective_parameters),
            },
        )


# --------------------------------------------------------------------------
# Parallel central-difference gradient
# --------------------------------------------------------------------------

_WORKER_PROBLEM: Any = None


def _worker_objective(z: np.ndarray) -> float:
    return _WORKER_PROBLEM.objective_z(z)


class ParallelGradient:
    """Objective and central-difference gradient.

    Structural coordinates (which need a new simulation) are differenced in
    worker processes; observation coordinates reuse the parent's cached
    simulation.  ``problem`` needs ``objective_z``, ``structural_index`` and
    ``observation_index``.
    """

    def __init__(self, problem: Any, n_workers: int | None = None, step: float = 1e-3):
        global _WORKER_PROBLEM
        self.problem = problem
        self.step = float(step)
        n_struct = len(problem.structural_index)
        cpu = os.cpu_count() or 2
        self.n_workers = int(n_workers or max(1, min(2 * n_struct, cpu - 2)))
        _WORKER_PROBLEM = problem
        self.pool = (
            ProcessPoolExecutor(self.n_workers, mp_context=mp.get_context("fork"))
            if self.n_workers > 1 else None
        )

    def value_and_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=float)
        f0 = self.problem.objective_z(z)
        grad = np.zeros_like(z)
        h = self.step
        points = []
        for i in self.problem.structural_index:
            for sign in (1.0, -1.0):
                zi = z.copy()
                zi[i] += sign * h
                points.append(zi)
        if self.pool is not None:
            values = list(self.pool.map(_worker_objective, points, chunksize=1))
        else:
            values = [self.problem.objective_z(p) for p in points]
        for k, i in enumerate(self.problem.structural_index):
            grad[i] = (values[2 * k] - values[2 * k + 1]) / (2 * h)
        for i in self.problem.observation_index:
            zp, zm = z.copy(), z.copy()
            zp[i] += h
            zm[i] -= h
            grad[i] = (self.problem.objective_z(zp) - self.problem.objective_z(zm)) / (2 * h)
        if f0 >= FAILED_OBJECTIVE:
            grad[:] = 0.0
        return f0, grad

    def evaluate_many(self, points: np.ndarray) -> list[float]:
        """Objective at many points, in the worker pool when available."""

        points = [np.asarray(p, dtype=float) for p in points]
        if self.pool is None:
            return [self.problem.objective_z(p) for p in points]
        return list(self.pool.map(_worker_objective, points, chunksize=max(1, len(points) // (4 * self.n_workers))))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def run_inference(countries: Iterable[str] = ("Australia", "Sweden"), settings: InferenceSettings = InferenceSettings()) -> list[InferenceFit]:
    output: list[InferenceFit] = []
    for country in countries:
        candidates = Core2026Inference(country, settings).fit()
        output.append(min(candidates, key=lambda fit: fit.objective))
    return output


def compare_variants(country: str, variants: Iterable[str] = ("baseline", "s5_school_detection", "exponential_vaccine_waning", "no_natural_immunity_boosting"), settings: InferenceSettings = InferenceSettings()) -> pd.DataFrame:
    """Fit each structural candidate and return an AIC comparison table."""
    rows = []
    for name in variants:
        fit = min(Core2026Inference(country, settings, variant=name).fit(), key=lambda item: item.objective)
        data_nll = fit.diagnostics["total_nll"] + fit.diagnostics["age_nll"]
        rows.append({"country": country, "variant": name, "objective": fit.objective,
                     "data_nll": data_nll, "n_parameters": fit.diagnostics["n_parameters"],
                     "aic": aic(data_nll, fit.diagnostics["n_parameters"]),
                     "success": fit.success, "nit": fit.nit, "nfev": fit.nfev,
                     "message": fit.message, **fit.parameter_values})
    table = pd.DataFrame(rows)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    return table


def write_fits(fits: list[InferenceFit], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    winner = min(fits, key=lambda f: f.objective)
    rows = []
    for fit in fits:
        suffix = f"start{fit.start_index}"
        fit.total_comparison.to_csv(output_dir / f"{fit.country}_{suffix}_total_comparison.csv", index=False)
        fit.age_comparison.to_csv(output_dir / f"{fit.country}_{suffix}_age_comparison.csv", index=False)
        pd.DataFrame({"iteration": range(1, len(fit.trace) + 1), "objective": fit.trace}).to_csv(
            output_dir / f"{fit.country}_{suffix}_trace.csv", index=False
        )
        diag = {k: (";".join(v) if isinstance(v, list) else v) for k, v in fit.diagnostics.items()}
        rows.append({
            "country": fit.country, "start_index": fit.start_index, "selected": fit is winner,
            "objective": fit.objective, "nit": fit.nit, "nfev": fit.nfev, "success": fit.success,
            "message": fit.message, **diag, **fit.parameter_values,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="core2026 M3 multi-start MAP inference")
    parser.add_argument("--countries", nargs="+", default=["Australia", "Sweden"])
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/core2026/m3_inference"))
    parser.add_argument("--variants", nargs="+", default=["baseline"])
    parser.add_argument("--diagnostic-geography", choices=['national_only', 'legacy_all'],
                        help="Required when a country's diagnostic timeline contains subnational rows")
    args = parser.parse_args()
    settings = InferenceSettings(n_starts=args.n_starts, max_iterations=args.max_iterations,
                                 n_workers=args.n_workers, diagnostic_geography=args.diagnostic_geography)
    summaries = []
    for country in args.countries:
        if len(args.variants) > 1:
            comparison = compare_variants(country, args.variants, settings)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(args.output_dir / f"{country}_aic_comparison.csv", index=False)
            summary = comparison
        else:
            fits = Core2026Inference(country, settings, variant=args.variants[0]).fit()
            summary = write_fits(fits, args.output_dir)
        summaries.append(summary)
        pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / "convergence_summary.csv", index=False)


if __name__ == "__main__":
    main()
