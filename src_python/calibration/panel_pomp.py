"""Arithmetic semi-mechanistic discrepancy POMP for panel count series.

This module deliberately does *not* propagate the compartment model.  It takes
interval-specific expected counts from a deterministic mechanistic run as
offsets, then estimates a country-level static log scale and filters a latent
time-varying discrepancy.  The resulting model is a partially observed Markov
process (POMP) for the discrepancy around a mechanistic forecast; it is not a
full stochastic transmission model and must not be described as one.

The core observation equation is

    Y[c, i] ~ NB2(offset[c, i] * exp(alpha[c] + x[c, i]), k[c, i]),

where ``alpha[c]`` is estimated using training intervals only.  ``x`` follows
an irregular-time OU representation of an AR(1).  An optional damped stochastic
trend gives a coupled two-dimensional OU process.  Process shocks may use a
variance-preserving two-component Gaussian scale mixture.  The NB2 size is
scaled by observed exposure, so partial surveillance windows are not assigned
the same dispersion as complete reference windows.

All filtering and forecast random streams are derived deterministically from
explicit integer seeds.  Predictive summaries use a separate stream and thus
cannot change the likelihood estimate or terminal particle cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.special import gammaln, logsumexp
from joblib import Parallel, delayed


_SECONDS_PER_DAY = 86_400.0
_NANOSECONDS_PER_DAY = _SECONDS_PER_DAY * 1e9
_MIN_SAFE_LOG_MEAN = -700.0
_MAX_SAFE_LOG_MEAN = 700.0


class ParticleDegeneracyError(RuntimeError):
    """Raised when every particle has zero predictive likelihood."""


def _readonly_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _numeric_day_vector(values: Sequence[Any], *, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if np.issubdtype(raw.dtype, np.number):
        result = np.asarray(raw, dtype=float)
    else:
        parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce", utc=True)
        if bool(parsed.isna().any()):
            raise ValueError(f"{label} contains invalid calendar values")
        result = (
            parsed.dt.as_unit("ns").astype("int64").to_numpy(dtype=float)
            / _NANOSECONDS_PER_DAY
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def _one_dimensional_float_vector(
    values: Sequence[Any],
    *,
    label: str,
    length: int,
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,):
        raise ValueError(f"{label} must have shape ({length},)")
    return result


def _boolean_vector(values: Sequence[Any], *, length: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (length,):
        raise ValueError(f"training_mask must have shape ({length},)")
    if raw.dtype != np.bool_:
        try:
            numeric = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("training_mask must contain booleans") from exc
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
            raise ValueError("training_mask must contain booleans or zero/one values")
    return np.asarray(raw, dtype=bool)


@dataclass(frozen=True)
class CountrySeries:
    """One country's ordered count intervals and mechanistic count offsets.

    ``training_mask`` must be a non-empty prefix.  Finite counts after that
    prefix may be used for forecast scoring, but are never assimilated by the
    filter or used to estimate ``static_log_scale``.

    Numeric start/end values are interpreted as days on an arbitrary common
    origin.  Calendar-like values are converted to UTC days since Unix epoch.
    """

    country: str
    interval_ids: Sequence[str]
    interval_start: Sequence[Any]
    interval_end: Sequence[Any]
    observed_counts: Sequence[float]
    mechanistic_offset: Sequence[float]
    exposure_days: Sequence[float]
    training_mask: Sequence[bool]

    def __post_init__(self) -> None:
        country = str(self.country).strip()
        if not country:
            raise ValueError("country must be non-empty")
        ids = tuple(str(value).strip() for value in self.interval_ids)
        n = len(ids)
        if n < 1 or any(not value for value in ids) or len(set(ids)) != n:
            raise ValueError("interval_ids must be non-empty and unique")

        start = _numeric_day_vector(self.interval_start, label="interval_start")
        end = _numeric_day_vector(self.interval_end, label="interval_end")
        if start.shape != (n,) or end.shape != (n,):
            raise ValueError("interval start/end lengths must match interval_ids")
        if np.any(end <= start):
            raise ValueError("every interval_end must be after interval_start")
        if n > 1 and np.any(start[1:] < end[:-1] - 1e-10):
            raise ValueError("country observation intervals must not overlap")

        counts = _one_dimensional_float_vector(
            self.observed_counts,
            label="observed_counts",
            length=n,
        )
        finite_counts = np.isfinite(counts)
        if np.any(counts[finite_counts] < 0.0) or not np.allclose(
            counts[finite_counts],
            np.round(counts[finite_counts]),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("finite observed_counts must be non-negative integers")

        offsets = _one_dimensional_float_vector(
            self.mechanistic_offset,
            label="mechanistic_offset",
            length=n,
        )
        exposure = _one_dimensional_float_vector(
            self.exposure_days,
            label="exposure_days",
            length=n,
        )
        if not np.isfinite(offsets).all() or np.any(offsets <= 0.0):
            raise ValueError("mechanistic_offset must be finite and strictly positive")
        if not np.isfinite(exposure).all() or np.any(exposure <= 0.0):
            raise ValueError("exposure_days must be finite and strictly positive")
        duration = end - start
        if np.any(exposure > duration + 1e-7):
            raise ValueError("exposure_days cannot exceed the enclosing interval duration")

        training = _boolean_vector(self.training_mask, length=n)
        if not bool(training.any()):
            raise ValueError("training_mask must contain at least one training interval")
        first_holdout = np.flatnonzero(~training)
        if len(first_holdout) and bool(training[int(first_holdout[0]) :].any()):
            raise ValueError("training_mask must be a prefix; later intervals are forecasts")
        if not np.isfinite(counts[training]).all():
            raise ValueError("every training interval requires a finite observed count")

        midpoints = 0.5 * (start + end)
        if n > 1 and np.any(np.diff(midpoints) <= 0.0):
            raise ValueError("interval midpoints must be strictly increasing")

        object.__setattr__(self, "country", country)
        object.__setattr__(self, "interval_ids", ids)
        object.__setattr__(self, "interval_start", _readonly_array(start))
        object.__setattr__(self, "interval_end", _readonly_array(end))
        object.__setattr__(self, "observed_counts", _readonly_array(counts))
        object.__setattr__(self, "mechanistic_offset", _readonly_array(offsets))
        object.__setattr__(self, "exposure_days", _readonly_array(exposure))
        object.__setattr__(self, "training_mask", _readonly_array(training, dtype=bool))

    @property
    def midpoint_days(self) -> np.ndarray:
        return 0.5 * (
            np.asarray(self.interval_start, dtype=float)
            + np.asarray(self.interval_end, dtype=float)
        )

    @property
    def n_intervals(self) -> int:
        return len(self.interval_ids)

    @property
    def n_training(self) -> int:
        return int(np.count_nonzero(self.training_mask))


@dataclass(frozen=True)
class DiscrepancyProcessSpec:
    """Irregular-time latent discrepancy process.

    ``rho`` is the correlation across ``reference_time_days`` for the
    discrepancy when the trend is disabled. ``innovation_sd`` is its
    conditional standard deviation across that same reference interval.

    With ``trend_enabled=True``, the continuous state is ``(x, b)`` with

        dx = (-kappa_x * x + trend_coupling * b) dt + sigma_x dW_x
        db = -kappa_b * b dt + sigma_b dW_b.

    The trend is therefore damped rather than extrapolated indefinitely.
    ``trend_innovation_sd`` has the analogous reference-interval meaning for
    the uncoupled trend equation.
    """

    rho: float
    innovation_sd: float
    reference_time_days: float = 365.2425
    robust_mixture_probability: float = 0.0
    robust_mixture_scale: float = 4.0
    jump_rate_per_year: float = 0.0
    jump_sd: float = 0.0
    trend_enabled: bool = False
    trend_damping: float = 0.5
    trend_innovation_sd: float = 0.0
    trend_coupling: float = 1.0

    def __post_init__(self) -> None:
        numeric = {
            "rho": self.rho,
            "innovation_sd": self.innovation_sd,
            "reference_time_days": self.reference_time_days,
            "robust_mixture_probability": self.robust_mixture_probability,
            "robust_mixture_scale": self.robust_mixture_scale,
            "jump_rate_per_year": self.jump_rate_per_year,
            "jump_sd": self.jump_sd,
            "trend_damping": self.trend_damping,
            "trend_innovation_sd": self.trend_innovation_sd,
            "trend_coupling": self.trend_coupling,
        }
        if not all(isfinite(float(value)) for value in numeric.values()):
            raise ValueError("discrepancy-process parameters must be finite")
        if not 0.0 < float(self.rho) < 1.0:
            raise ValueError("rho must lie strictly between zero and one")
        if float(self.innovation_sd) < 0.0:
            raise ValueError("innovation_sd must be non-negative")
        if float(self.reference_time_days) <= 0.0:
            raise ValueError("reference_time_days must be positive")
        if not 0.0 <= float(self.robust_mixture_probability) < 1.0:
            raise ValueError("robust_mixture_probability must be in [0, 1)")
        if float(self.robust_mixture_scale) < 1.0:
            raise ValueError("robust_mixture_scale must be at least one")
        if self.robust_mixture_probability > 0.0 and self.robust_mixture_scale <= 1.0:
            raise ValueError("a non-zero robust mixture requires scale > 1")
        if float(self.jump_rate_per_year) < 0.0:
            raise ValueError("jump_rate_per_year must be non-negative")
        if float(self.jump_sd) < 0.0:
            raise ValueError("jump_sd must be non-negative")
        if self.jump_rate_per_year > 0.0 and self.jump_sd <= 0.0:
            raise ValueError("a positive jump rate requires jump_sd > 0")
        if self.jump_rate_per_year == 0.0 and self.jump_sd != 0.0:
            raise ValueError("jump_sd must be zero when jumps are disabled")
        if self.jump_rate_per_year > 0.0 and self.trend_enabled:
            raise ValueError("jump diffusion and damped trend cannot be combined")
        if self.trend_enabled:
            if not 0.0 < float(self.trend_damping) < 1.0:
                raise ValueError("trend_damping must lie strictly between zero and one")
            if float(self.trend_innovation_sd) < 0.0:
                raise ValueError("trend_innovation_sd must be non-negative")
        elif float(self.trend_innovation_sd) != 0.0:
            raise ValueError("trend_innovation_sd must be zero when trend is disabled")

    @property
    def state_dimension(self) -> int:
        return 2 if self.trend_enabled else 1


@dataclass(frozen=True)
class ObservationSpec:
    """Exposure-scaled NB2 measurement model."""

    dispersion_per_reference_exposure: float
    reference_exposure_days: float = 365.2425

    def __post_init__(self) -> None:
        if not (
            isfinite(float(self.dispersion_per_reference_exposure))
            and float(self.dispersion_per_reference_exposure) > 0.0
        ):
            raise ValueError("dispersion_per_reference_exposure must be positive")
        if not (
            isfinite(float(self.reference_exposure_days))
            and float(self.reference_exposure_days) > 0.0
        ):
            raise ValueError("reference_exposure_days must be positive")

    def interval_size(self, exposure_days: float | np.ndarray) -> np.ndarray:
        exposure = np.asarray(exposure_days, dtype=float)
        if not np.isfinite(exposure).all() or np.any(exposure <= 0.0):
            raise ValueError("exposure_days must be finite and positive")
        return (
            float(self.dispersion_per_reference_exposure)
            * exposure
            / float(self.reference_exposure_days)
        )


@dataclass(frozen=True)
class ParticleFilterConfig:
    n_particles: int = 512
    ess_resample_fraction: float = 0.5
    seed: int = 20260713
    predictive_draws: int = 512
    robust_initialization_steps: int = 32
    assimilate_holdout_observations: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.n_particles, bool) or int(self.n_particles) != self.n_particles:
            raise ValueError("n_particles must be an integer")
        if int(self.n_particles) < 2:
            raise ValueError("n_particles must be at least two")
        if not 0.0 < float(self.ess_resample_fraction) <= 1.0:
            raise ValueError("ess_resample_fraction must be in (0, 1]")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.predictive_draws, bool)
            or int(self.predictive_draws) != self.predictive_draws
            or int(self.predictive_draws) < 0
        ):
            raise ValueError("predictive_draws must be a non-negative integer")
        if (
            isinstance(self.robust_initialization_steps, bool)
            or int(self.robust_initialization_steps) != self.robust_initialization_steps
            or int(self.robust_initialization_steps) < 0
        ):
            raise ValueError("robust_initialization_steps must be a non-negative integer")


@dataclass(frozen=True)
class ScaleSearchConfig:
    method: str = "particle_grid"
    lower: float = -3.0
    upper: float = 3.0
    grid_points: int = 9
    refinement_steps: int = 2
    center_on_training_log_ratio: bool = False
    local_half_width: float = 2.0

    def __post_init__(self) -> None:
        if str(self.method) not in {"particle_grid", "count_ratio"}:
            raise ValueError("scale-search method must be particle_grid or count_ratio")
        if not (isfinite(float(self.lower)) and isfinite(float(self.upper))):
            raise ValueError("scale-search bounds must be finite")
        if float(self.lower) >= float(self.upper):
            raise ValueError("scale-search lower bound must precede upper bound")
        if (
            isinstance(self.grid_points, bool)
            or int(self.grid_points) != self.grid_points
            or int(self.grid_points) < 3
        ):
            raise ValueError("grid_points must be an integer of at least three")
        if (
            isinstance(self.refinement_steps, bool)
            or int(self.refinement_steps) != self.refinement_steps
            or int(self.refinement_steps) < 0
        ):
            raise ValueError("refinement_steps must be a non-negative integer")
        if not isfinite(float(self.local_half_width)) or float(self.local_half_width) <= 0.0:
            raise ValueError("local_half_width must be positive and finite")


@dataclass(frozen=True)
class ForecastSummary:
    country: str
    interval_index: int
    interval_id: str
    training_interval: bool
    observed_count: float
    expected_mean: float
    conditional_mean_q025: float
    conditional_mean_median: float
    conditional_mean_q975: float
    observation_q025: float
    observation_median: float
    observation_q975: float
    predictive_log_score: float
    nb2_size: float
    observation_draws: np.ndarray


@dataclass(frozen=True)
class FilterStepDiagnostic:
    country: str
    interval_index: int
    interval_id: str
    training_interval: bool
    log_likelihood_increment: float
    effective_sample_size: float
    maximum_weight: float
    resampled: bool
    robust_shock_fraction: float


@dataclass(frozen=True)
class ParticleFilterResult:
    country: str
    static_log_scale: float
    training_log_likelihood: float
    forecasts: tuple[ForecastSummary, ...]
    diagnostics: tuple[FilterStepDiagnostic, ...]
    terminal_states: np.ndarray
    terminal_weights: np.ndarray
    terminal_ancestor_ids: np.ndarray


@dataclass(frozen=True)
class StaticScaleFit:
    country: str
    static_log_scale: float
    training_log_likelihood: float
    evaluations: int
    at_search_boundary: bool
    filter_result: ParticleFilterResult


@dataclass(frozen=True)
class PanelLikelihoodResult:
    total_training_log_likelihood: float
    country_results: tuple[ParticleFilterResult, ...]


@dataclass(frozen=True)
class PanelCandidate:
    name: str
    process: DiscrepancyProcessSpec
    observation: ObservationSpec
    scale_search: ScaleSearchConfig = field(default_factory=ScaleSearchConfig)
    estimated_shared_parameter_count: int = 0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("candidate name must be non-empty")
        if (
            isinstance(self.estimated_shared_parameter_count, bool)
            or int(self.estimated_shared_parameter_count)
            != self.estimated_shared_parameter_count
            or int(self.estimated_shared_parameter_count) < 0
        ):
            raise ValueError("estimated_shared_parameter_count must be non-negative")
        object.__setattr__(self, "name", str(self.name).strip())


@dataclass(frozen=True)
class PanelCandidateResult:
    name: str
    country_fits: tuple[StaticScaleFit, ...]
    training_log_likelihood: float
    validation_log_score: float
    country_balanced_validation_log_score: float
    validation_interval_count: int
    aic: float
    selection_score: float
    selection_criterion: str


@dataclass(frozen=True)
class PanelSelectionResult:
    selected_name: str
    selected: PanelCandidateResult
    candidates: tuple[PanelCandidateResult, ...]
    selection_criterion: str
    scope: str = "candidate_selection_validation_not_final_test"


def _stable_seed(base_seed: int, label: str) -> int:
    payload = f"{int(base_seed)}::{label}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def robust_component_scales(probability: float, robust_scale: float) -> tuple[float, float]:
    """Return normal/robust scales normalized to unit mixture variance."""

    probability = float(probability)
    robust_scale = float(robust_scale)
    if not 0.0 <= probability < 1.0 or not np.isfinite(probability):
        raise ValueError("probability must be in [0, 1)")
    if not np.isfinite(robust_scale) or robust_scale < 1.0:
        raise ValueError("robust_scale must be at least one")
    if probability > 0.0 and robust_scale <= 1.0:
        raise ValueError("positive robust probability requires robust_scale > 1")
    normalizer = float(np.sqrt((1.0 - probability) + probability * robust_scale**2))
    return 1.0 / normalizer, robust_scale / normalizer


def nb2_logpmf(
    observed: float | np.ndarray,
    mean: float | np.ndarray,
    size: float | np.ndarray,
) -> np.ndarray:
    """Stable vectorized NB2 log mass with ``Var(Y)=mu+mu^2/size``."""

    y, mu, k = np.broadcast_arrays(
        np.asarray(observed, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(size, dtype=float),
    )
    if not np.isfinite(y).all() or np.any(y < 0.0) or not np.allclose(
        y, np.round(y), rtol=0.0, atol=1e-8
    ):
        raise ValueError("observed counts must be finite non-negative integers")
    if not np.isfinite(mu).all() or np.any(mu <= 0.0):
        raise ValueError("NB2 means must be finite and positive")
    if not np.isfinite(k).all() or np.any(k <= 0.0):
        raise ValueError("NB2 sizes must be finite and positive")
    log_mu = np.log(mu)
    log_k = np.log(k)
    denominator = np.logaddexp(log_k, log_mu)
    log_p = log_k - denominator
    log_one_minus_p = log_mu - denominator
    return (
        gammaln(y + k)
        - gammaln(k)
        - gammaln(y + 1.0)
        + k * log_p
        + y * log_one_minus_p
    )


def _nb2_logpmf_from_log_mean(
    observed: float,
    log_mean: np.ndarray,
    size: float,
) -> np.ndarray:
    y = float(observed)
    k = float(size)
    if not np.isfinite(y) or y < 0.0 or not np.isclose(y, round(y), atol=1e-8):
        raise ValueError("observed count must be a finite non-negative integer")
    if not np.isfinite(log_mean).all() or np.any(log_mean > _MAX_SAFE_LOG_MEAN):
        raise ValueError("particle log means are non-finite or numerically implausible")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("NB2 size must be finite and positive")
    log_k = float(np.log(k))
    denominator = np.logaddexp(log_k, log_mean)
    return (
        gammaln(y + k)
        - gammaln(k)
        - gammaln(y + 1.0)
        + k * (log_k - denominator)
        + y * (log_mean - denominator)
    )


def _project_psd(matrix: np.ndarray, *, label: str) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if not np.isfinite(symmetric).all():
        raise RuntimeError(f"{label} is non-finite")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-10, 1e-13)
    if float(np.min(eigenvalues)) < -tolerance:
        raise RuntimeError(f"{label} is not positive semidefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    return 0.5 * (
        (eigenvectors * clipped) @ eigenvectors.T
        + ((eigenvectors * clipped) @ eigenvectors.T).T
    )


@lru_cache(maxsize=256)
def _continuous_process_matrices(
    process: DiscrepancyProcessSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kappa_x = -float(np.log(process.rho))
    x_diffusion_variance = (
        0.0
        if process.innovation_sd == 0.0
        else float(process.innovation_sd) ** 2
        * (2.0 * kappa_x)
        / (1.0 - np.exp(-2.0 * kappa_x))
    )
    if not process.trend_enabled:
        drift = np.asarray([[-kappa_x]], dtype=float)
        diffusion = np.asarray([[x_diffusion_variance]], dtype=float)
    else:
        kappa_trend = -float(np.log(process.trend_damping))
        trend_diffusion_variance = (
            0.0
            if process.trend_innovation_sd == 0.0
            else float(process.trend_innovation_sd) ** 2
            * (2.0 * kappa_trend)
            / (1.0 - np.exp(-2.0 * kappa_trend))
        )
        drift = np.asarray(
            [
                [-kappa_x, float(process.trend_coupling)],
                [0.0, -kappa_trend],
            ],
            dtype=float,
        )
        diffusion = np.diag([x_diffusion_variance, trend_diffusion_variance])
    stationary = solve_continuous_lyapunov(drift, -diffusion)
    stationary = _project_psd(stationary, label="stationary process covariance")
    return drift, diffusion, stationary


def process_transition_moments(
    process: DiscrepancyProcessSpec,
    delta_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact linear transition, innovation covariance, and stationarity.

    The covariance is calculated as ``P - F P F'`` for the continuous coupled
    OU model.  This preserves irregular calendar spacing without treating every
    surveillance interval as a nominal year.
    """

    delta_days = float(delta_days)
    if not np.isfinite(delta_days) or delta_days <= 0.0:
        raise ValueError("delta_days must be finite and positive")
    drift, _diffusion, stationary = _continuous_process_matrices(process)
    delta_reference = delta_days / float(process.reference_time_days)
    transition = expm(drift * delta_reference)
    innovation = stationary - transition @ stationary @ transition.T
    innovation = _project_psd(innovation, label="process innovation covariance")
    return transition, innovation, stationary.copy()


def _psd_factor(covariance: np.ndarray) -> np.ndarray:
    covariance = _project_psd(covariance, label="particle covariance")
    values, vectors = np.linalg.eigh(covariance)
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))


def systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Systematic resampling indices for a normalized empirical measure."""

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("weights must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    normalized = values / total
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    positions = (float(rng.random()) + np.arange(len(values), dtype=float)) / len(values)
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: Sequence[float],
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    if x.ndim != 1 or weight.shape != x.shape or not np.isfinite(x).all():
        raise ValueError("weighted quantile inputs are invalid")
    if not np.isfinite(weight).all() or np.any(weight < 0.0) or float(weight.sum()) <= 0.0:
        raise ValueError("weighted quantile weights are invalid")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("weighted quantile probabilities must be in [0, 1]")
    order = np.argsort(x)
    ordered = x[order]
    ordered_weight = weight[order]
    cumulative = np.cumsum(ordered_weight)
    cumulative /= float(cumulative[-1])
    positions = np.searchsorted(cumulative, probability, side="left")
    return ordered[np.minimum(positions, len(ordered) - 1)]


def _draw_initial_states(
    process: DiscrepancyProcessSpec,
    config: ParticleFilterConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    _drift, _diffusion, stationary = _continuous_process_matrices(process)
    factor = _psd_factor(stationary)
    states = rng.standard_normal((config.n_particles, process.state_dimension)) @ factor.T
    # A Gaussian draw is exact for a Gaussian OU.  Under the robust mixture it
    # has the correct stationary covariance but not the stationary tails. A few
    # variance-preserving transitions provide a reproducible tail burn-in.
    if (
        process.robust_mixture_probability > 0.0
        or process.jump_rate_per_year > 0.0
    ):
        for _ in range(config.robust_initialization_steps):
            states, _ = _propagate_states(
                states,
                process,
                float(process.reference_time_days),
                rng,
            )
    return states


def _propagate_states(
    states: np.ndarray,
    process: DiscrepancyProcessSpec,
    delta_days: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    transition, covariance, _stationary = process_transition_moments(process, delta_days)
    factor = _psd_factor(covariance)
    standard = rng.standard_normal(np.asarray(states).shape)
    innovation = standard @ factor.T
    probability = float(process.robust_mixture_probability)
    normal_scale, shock_scale = robust_component_scales(
        probability,
        float(process.robust_mixture_scale),
    )
    if probability > 0.0:
        shocks = rng.random(len(states)) < probability
        scales = np.where(shocks, shock_scale, normal_scale)
        innovation *= scales[:, None]
        shock_mask = shocks
    else:
        innovation *= normal_scale
        shock_mask = np.zeros(len(states), dtype=bool)
    propagated = np.asarray(states, dtype=float) @ transition.T + innovation
    jump_rate = float(process.jump_rate_per_year)
    if jump_rate > 0.0:
        elapsed_years = float(delta_days) / float(process.reference_time_days)
        jump_counts = rng.poisson(jump_rate * elapsed_years, size=len(states))
        for jump_index in range(int(np.max(jump_counts, initial=0))):
            active = jump_counts > jump_index
            count = int(np.count_nonzero(active))
            if count < 1:
                continue
            # Conditional on the number of Poisson events, event times are
            # uniform. A jump is attenuated from its event time to the interval
            # end by the OU transition, preserving irregular-time semantics.
            event_age_years = rng.uniform(0.0, elapsed_years, size=count)
            attenuation = np.power(float(process.rho), event_age_years)
            propagated[active, 0] += (
                rng.normal(0.0, float(process.jump_sd), size=count) * attenuation
            )
        shock_mask = shock_mask | (jump_counts > 0)
    shock_fraction = float(np.mean(shock_mask))
    if not np.isfinite(propagated).all():
        raise RuntimeError("latent discrepancy transition produced a non-finite state")
    return propagated, shock_fraction


def _forecast_summary(
    *,
    series: CountrySeries,
    interval_index: int,
    log_means: np.ndarray,
    weights: np.ndarray,
    size: float,
    config: ParticleFilterConfig,
    predictive_particle_indices: np.ndarray | None = None,
) -> ForecastSummary:
    means = np.exp(log_means)
    mean_quantiles = _weighted_quantile(means, weights, (0.025, 0.5, 0.975))
    expected_mean = float(np.sum(weights * means))
    observed = float(series.observed_counts[interval_index])
    mixture_log_weights = np.full(len(weights), -np.inf, dtype=float)
    positive_weight = weights > 0.0
    mixture_log_weights[positive_weight] = np.log(weights[positive_weight])
    predictive_log_score = (
        float("nan")
        if not np.isfinite(observed)
        else float(
            logsumexp(
                mixture_log_weights
                + _nb2_logpmf_from_log_mean(observed, log_means, size)
            )
        )
    )

    draw_count = (
        int(config.predictive_draws)
        if not bool(series.training_mask[interval_index])
        else 0
    )
    if draw_count > 0:
        forecast_seed = _stable_seed(
            config.seed,
            f"forecast::{series.country}::{series.interval_ids[interval_index]}",
        )
        forecast_rng = np.random.default_rng(forecast_seed)
        if predictive_particle_indices is None:
            particle = forecast_rng.choice(
                len(weights),
                size=draw_count,
                replace=True,
                p=weights / float(np.sum(weights)),
            )
        else:
            particle = np.asarray(predictive_particle_indices, dtype=np.int64)
            if particle.shape != (draw_count,) or np.any(
                (particle < 0) | (particle >= len(weights))
            ):
                raise ValueError("predictive_particle_indices are invalid")
        selected_mean = means[particle]
        probability = size / (size + selected_mean)
        draws = forecast_rng.negative_binomial(size, probability)
        observation_quantiles = np.quantile(draws, (0.025, 0.5, 0.975))
    else:
        draws = np.asarray([], dtype=float)
        observation_quantiles = np.full(3, np.nan)

    return ForecastSummary(
        country=series.country,
        interval_index=int(interval_index),
        interval_id=series.interval_ids[interval_index],
        training_interval=bool(series.training_mask[interval_index]),
        observed_count=observed,
        expected_mean=expected_mean,
        conditional_mean_q025=float(mean_quantiles[0]),
        conditional_mean_median=float(mean_quantiles[1]),
        conditional_mean_q975=float(mean_quantiles[2]),
        observation_q025=float(observation_quantiles[0]),
        observation_median=float(observation_quantiles[1]),
        observation_q975=float(observation_quantiles[2]),
        predictive_log_score=predictive_log_score,
        nb2_size=float(size),
        observation_draws=_readonly_array(draws, dtype=float),
    )


def bootstrap_particle_filter(
    series: CountrySeries,
    *,
    static_log_scale: float,
    process: DiscrepancyProcessSpec,
    observation: ObservationSpec,
    config: ParticleFilterConfig = ParticleFilterConfig(),
) -> ParticleFilterResult:
    """Run the bootstrap filter and honest prefix-trained forecasts.

    The mechanistic offset is never recomputed here, and the discrepancy does
    not alter transmission states.  Non-training counts are scored if present
    but never enter the particle weights.
    """

    alpha = float(static_log_scale)
    if not np.isfinite(alpha):
        raise ValueError("static_log_scale must be finite")
    rng = np.random.default_rng(int(config.seed))
    states = _draw_initial_states(process, config, rng)
    n = int(config.n_particles)
    weights = np.full(n, 1.0 / n, dtype=float)
    log_weights = np.full(n, -np.log(n), dtype=float)
    ancestors = np.arange(n, dtype=np.int64)
    forecasts: list[ForecastSummary] = []
    diagnostics: list[FilterStepDiagnostic] = []
    training_log_likelihood = 0.0
    midpoints = series.midpoint_days
    predictive_particle_indices: np.ndarray | None = None

    for interval_index in range(series.n_intervals):
        shock_fraction = float("nan")
        if interval_index > 0:
            delta_days = float(midpoints[interval_index] - midpoints[interval_index - 1])
            states, shock_fraction = _propagate_states(states, process, delta_days, rng)

        log_means = (
            float(np.log(series.mechanistic_offset[interval_index]))
            + alpha
            + states[:, 0]
        )
        if (
            not np.isfinite(log_means).all()
            or np.any(log_means < _MIN_SAFE_LOG_MEAN)
            or np.any(log_means > _MAX_SAFE_LOG_MEAN)
        ):
            raise ValueError("static scale/process produced an invalid predictive count mean")
        interval_size = float(observation.interval_size(series.exposure_days[interval_index]))
        if (
            not bool(series.training_mask[interval_index])
            and int(config.predictive_draws) > 0
            and predictive_particle_indices is None
        ):
            path_rng = np.random.default_rng(
                _stable_seed(config.seed, f"forecast-path::{series.country}")
            )
            predictive_particle_indices = path_rng.choice(
                len(weights),
                size=int(config.predictive_draws),
                replace=True,
                p=weights / float(np.sum(weights)),
            )
        forecast = _forecast_summary(
            series=series,
            interval_index=interval_index,
            log_means=log_means,
            weights=weights,
            size=interval_size,
            config=config,
            predictive_particle_indices=predictive_particle_indices,
        )
        forecasts.append(forecast)

        training_interval = bool(series.training_mask[interval_index])
        assimilated = bool(
            training_interval or config.assimilate_holdout_observations
        )
        log_increment = float("nan")
        resampled = False
        if assimilated:
            observation_log_mass = _nb2_logpmf_from_log_mean(
                float(series.observed_counts[interval_index]),
                log_means,
                interval_size,
            )
            unnormalized = log_weights + observation_log_mass
            normalizer = float(logsumexp(unnormalized))
            if not np.isfinite(normalizer):
                raise ParticleDegeneracyError(
                    f"All particles have zero likelihood for {series.country} "
                    f"interval {series.interval_ids[interval_index]}"
                )
            log_increment = normalizer
            if training_interval:
                training_log_likelihood += normalizer
            log_weights = unnormalized - normalizer
            weights = np.exp(log_weights)
            weights /= float(np.sum(weights))

        effective_sample_size = float(1.0 / np.sum(weights**2))
        maximum_weight = float(np.max(weights))
        if assimilated and effective_sample_size <= config.ess_resample_fraction * n:
            source = systematic_resample(weights, rng)
            states = np.array(states[source], copy=True)
            ancestors = np.array(ancestors[source], copy=True)
            weights = np.full(n, 1.0 / n, dtype=float)
            log_weights = np.full(n, -np.log(n), dtype=float)
            resampled = True

        diagnostics.append(
            FilterStepDiagnostic(
                country=series.country,
                interval_index=int(interval_index),
                interval_id=series.interval_ids[interval_index],
                training_interval=training_interval,
                log_likelihood_increment=log_increment,
                effective_sample_size=effective_sample_size,
                maximum_weight=maximum_weight,
                resampled=resampled,
                robust_shock_fraction=shock_fraction,
            )
        )
        if not training_interval and config.assimilate_holdout_observations:
            # The next prequential forecast conditions on the observation just
            # assimilated, so it must resample predictive particle identities
            # from the updated cloud.
            predictive_particle_indices = None

    return ParticleFilterResult(
        country=series.country,
        static_log_scale=alpha,
        training_log_likelihood=float(training_log_likelihood),
        forecasts=tuple(forecasts),
        diagnostics=tuple(diagnostics),
        terminal_states=_readonly_array(states),
        terminal_weights=_readonly_array(weights),
        terminal_ancestor_ids=_readonly_array(ancestors, dtype=np.int64),
    )


def fit_country_static_log_scale(
    series: CountrySeries,
    *,
    process: DiscrepancyProcessSpec,
    observation: ObservationSpec,
    filter_config: ParticleFilterConfig = ParticleFilterConfig(),
    search: ScaleSearchConfig = ScaleSearchConfig(),
) -> StaticScaleFit:
    """Fit one static arithmetic scale from training intervals only.

    A deterministic refined grid is preferred to a gradient optimizer because
    the particle likelihood is noisy.  Every point uses common random numbers.
    """

    likelihood_config = replace(filter_config, predictive_draws=0)
    lower = float(search.lower)
    upper = float(search.upper)
    if str(search.method) == "count_ratio":
        training = np.asarray(series.training_mask, dtype=bool)
        numerator = float(np.sum(series.observed_counts[training]) + 0.5)
        denominator = float(np.sum(series.mechanistic_offset[training]))
        raw_value = float(np.log(numerator / max(denominator, 1e-300)))
        best_value = float(np.clip(raw_value, lower, upper))
        final_result = bootstrap_particle_filter(
            series,
            static_log_scale=best_value,
            process=process,
            observation=observation,
            config=filter_config,
        )
        return StaticScaleFit(
            country=series.country,
            static_log_scale=best_value,
            training_log_likelihood=float(final_result.training_log_likelihood),
            evaluations=1,
            at_search_boundary=bool(best_value != raw_value),
            filter_result=final_result,
        )
    if bool(search.center_on_training_log_ratio):
        training = np.asarray(series.training_mask, dtype=bool)
        numerator = float(np.sum(series.observed_counts[training]) + 0.5)
        denominator = float(np.sum(series.mechanistic_offset[training]))
        center = float(np.log(numerator / max(denominator, 1e-300)))
        lower = max(lower, center - float(search.local_half_width))
        upper = min(upper, center + float(search.local_half_width))
        if lower >= upper:
            lower, upper = float(search.lower), float(search.upper)
    evaluated: dict[float, float] = {}
    best_value = float("nan")
    best_score = -np.inf

    for refinement in range(int(search.refinement_steps) + 1):
        grid = np.linspace(lower, upper, int(search.grid_points))
        scores: list[float] = []
        for value in grid:
            key = float(value)
            if key not in evaluated:
                result = bootstrap_particle_filter(
                    series,
                    static_log_scale=key,
                    process=process,
                    observation=observation,
                    config=likelihood_config,
                )
                evaluated[key] = float(result.training_log_likelihood)
            scores.append(evaluated[key])
        best_index = int(np.argmax(np.asarray(scores, dtype=float)))
        if scores[best_index] > best_score:
            best_score = float(scores[best_index])
            best_value = float(grid[best_index])
        if refinement < int(search.refinement_steps):
            left = max(best_index - 1, 0)
            right = min(best_index + 1, len(grid) - 1)
            lower = float(grid[left])
            upper = float(grid[right])
            if lower == upper:
                break

    final_result = bootstrap_particle_filter(
        series,
        static_log_scale=best_value,
        process=process,
        observation=observation,
        config=filter_config,
    )
    tolerance = max((float(search.upper) - float(search.lower)) * 1e-10, 1e-12)
    at_boundary = bool(
        best_value <= float(search.lower) + tolerance
        or best_value >= float(search.upper) - tolerance
    )
    return StaticScaleFit(
        country=series.country,
        static_log_scale=best_value,
        training_log_likelihood=float(final_result.training_log_likelihood),
        evaluations=len(evaluated),
        at_search_boundary=at_boundary,
        filter_result=final_result,
    )


def _ordered_panel(series: Sequence[CountrySeries]) -> tuple[CountrySeries, ...]:
    ordered = tuple(sorted(series, key=lambda item: item.country))
    if not ordered:
        raise ValueError("panel must contain at least one country")
    names = [item.country for item in ordered]
    if len(set(names)) != len(names):
        raise ValueError("panel countries must be unique")
    return ordered


def panel_log_likelihood(
    series: Sequence[CountrySeries],
    *,
    country_log_scales: Mapping[str, float],
    process: DiscrepancyProcessSpec,
    observation: ObservationSpec,
    filter_config: ParticleFilterConfig = ParticleFilterConfig(),
) -> PanelLikelihoodResult:
    """Sum independent country particle likelihoods with stable country seeds."""

    ordered = _ordered_panel(series)
    required = {item.country for item in ordered}
    supplied = {str(key) for key in country_log_scales}
    if supplied != required:
        raise ValueError(
            "country_log_scales must match panel countries exactly: "
            f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}"
        )
    results: list[ParticleFilterResult] = []
    for item in ordered:
        country_config = replace(
            filter_config,
            seed=_stable_seed(filter_config.seed, f"country::{item.country}"),
        )
        results.append(
            bootstrap_particle_filter(
                item,
                static_log_scale=float(country_log_scales[item.country]),
                process=process,
                observation=observation,
                config=country_config,
            )
        )
    return PanelLikelihoodResult(
        total_training_log_likelihood=float(
            np.sum([result.training_log_likelihood for result in results])
        ),
        country_results=tuple(results),
    )


def _fit_panel_candidate_item(
    item: CountrySeries,
    candidate: PanelCandidate,
    filter_config: ParticleFilterConfig,
) -> StaticScaleFit:
    country_config = replace(
        filter_config,
        seed=_stable_seed(filter_config.seed, f"country::{item.country}"),
    )
    return fit_country_static_log_scale(
        item,
        process=candidate.process,
        observation=candidate.observation,
        filter_config=country_config,
        search=candidate.scale_search,
    )


def _candidate_result_from_fits(
    ordered: Sequence[CountrySeries],
    candidate: PanelCandidate,
    fits: Sequence[StaticScaleFit],
    selection_criterion: str,
) -> PanelCandidateResult:
    criterion = str(selection_criterion).lower()
    if criterion not in {
        "validation_log_score",
        "country_balanced_validation_log_score",
        "training_aic",
        "training_log_likelihood",
    }:
        raise ValueError(
            "selection_criterion must be validation_log_score, "
            "country_balanced_validation_log_score, training_aic, or "
            "training_log_likelihood"
        )
    if len(fits) != len(ordered):
        raise ValueError("candidate fit count does not match the panel")
    fits = tuple(fits)
    training_log_likelihood = float(
        np.sum([fit.training_log_likelihood for fit in fits])
    )
    validation_scores: list[float] = []
    country_validation_means: list[float] = []
    for fit in fits:
        country_scores: list[float] = []
        for forecast in fit.filter_result.forecasts:
            if (
                not forecast.training_interval
                and np.isfinite(forecast.observed_count)
                and np.isfinite(forecast.predictive_log_score)
            ):
                score = float(forecast.predictive_log_score)
                validation_scores.append(score)
                country_scores.append(score)
        if country_scores:
            country_validation_means.append(float(np.mean(country_scores)))
    validation_count = len(validation_scores)
    validation_log_score = (
        float(np.sum(validation_scores)) if validation_scores else float("nan")
    )
    country_balanced_validation_log_score = (
        float(np.mean(country_validation_means))
        if country_validation_means
        else float("nan")
    )
    parameter_count = len(ordered) + int(candidate.estimated_shared_parameter_count)
    aic = float(2.0 * parameter_count - 2.0 * training_log_likelihood)
    if criterion in {"validation_log_score", "country_balanced_validation_log_score"}:
        if validation_count < 1:
            raise ValueError(
                "validation score selection requires finite non-training observations"
            )
        selection_score = (
            country_balanced_validation_log_score
            if criterion == "country_balanced_validation_log_score"
            else validation_log_score
        )
    elif criterion == "training_aic":
        selection_score = -0.5 * aic
    else:
        selection_score = training_log_likelihood
    return PanelCandidateResult(
        name=candidate.name,
        country_fits=fits,
        training_log_likelihood=training_log_likelihood,
        validation_log_score=validation_log_score,
        country_balanced_validation_log_score=country_balanced_validation_log_score,
        validation_interval_count=validation_count,
        aic=aic,
        selection_score=float(selection_score),
        selection_criterion=criterion,
    )


def evaluate_panel_candidate(
    series: Sequence[CountrySeries],
    candidate: PanelCandidate,
    *,
    filter_config: ParticleFilterConfig = ParticleFilterConfig(),
    selection_criterion: str = "validation_log_score",
    n_jobs: int = 1,
) -> PanelCandidateResult:
    """Fit country scales on training prefixes and evaluate one panel candidate."""

    ordered = _ordered_panel(series)
    if int(n_jobs) == 1:
        fits = [
            _fit_panel_candidate_item(item, candidate, filter_config)
            for item in ordered
        ]
    else:
        fits = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            inner_max_num_threads=1,
        )(
            delayed(_fit_panel_candidate_item)(item, candidate, filter_config)
            for item in ordered
        )
    return _candidate_result_from_fits(
        ordered,
        candidate,
        fits,
        selection_criterion,
    )


def select_panel_candidate(
    series: Sequence[CountrySeries],
    candidates: Sequence[PanelCandidate],
    *,
    filter_config: ParticleFilterConfig = ParticleFilterConfig(),
    selection_criterion: str = "validation_log_score",
    n_jobs: int = 1,
) -> PanelSelectionResult:
    """Select a prespecified candidate; validation scores are not a final test."""

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one panel candidate is required")
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("panel candidate names must be unique")
    if int(n_jobs) == 1:
        evaluated = tuple(
            evaluate_panel_candidate(
                series,
                candidate,
                filter_config=filter_config,
                selection_criterion=selection_criterion,
                n_jobs=1,
            )
            for candidate in candidates
        )
    else:
        ordered = _ordered_panel(series)
        task_count = len(candidates) * len(ordered)
        flat_jobs = min(int(n_jobs), task_count) if int(n_jobs) > 0 else int(n_jobs)
        flat_fits = Parallel(
            n_jobs=flat_jobs,
            backend="loky",
            inner_max_num_threads=1,
        )(
            delayed(_fit_panel_candidate_item)(item, candidate, filter_config)
            for candidate in candidates
            for item in ordered
        )
        evaluated = tuple(
            _candidate_result_from_fits(
                ordered,
                candidate,
                flat_fits[
                    candidate_index * len(ordered) : (candidate_index + 1) * len(ordered)
                ],
                selection_criterion,
            )
            for candidate_index, candidate in enumerate(candidates)
        )
    # Stable name ordering resolves exact numerical ties without depending on
    # caller order.
    selected = sorted(
        evaluated,
        key=lambda result: (-result.selection_score, result.name),
    )[0]
    return PanelSelectionResult(
        selected_name=selected.name,
        selected=selected,
        candidates=evaluated,
        selection_criterion=str(selection_criterion).lower(),
    )


def forecast_country(
    series: CountrySeries,
    *,
    fitted_static_log_scale: float,
    process: DiscrepancyProcessSpec,
    observation: ObservationSpec,
    filter_config: ParticleFilterConfig = ParticleFilterConfig(),
) -> tuple[ForecastSummary, ...]:
    """Return forecasts while assimilating only ``series.training_mask``."""

    return bootstrap_particle_filter(
        series,
        static_log_scale=fitted_static_log_scale,
        process=process,
        observation=observation,
        config=filter_config,
    ).forecasts


__all__ = [
    "CountrySeries",
    "DiscrepancyProcessSpec",
    "FilterStepDiagnostic",
    "ForecastSummary",
    "ObservationSpec",
    "PanelCandidate",
    "PanelCandidateResult",
    "PanelLikelihoodResult",
    "PanelSelectionResult",
    "ParticleDegeneracyError",
    "ParticleFilterConfig",
    "ParticleFilterResult",
    "ScaleSearchConfig",
    "StaticScaleFit",
    "bootstrap_particle_filter",
    "evaluate_panel_candidate",
    "fit_country_static_log_scale",
    "forecast_country",
    "nb2_logpmf",
    "panel_log_likelihood",
    "process_transition_moments",
    "robust_component_scales",
    "select_panel_candidate",
    "systematic_resample",
]
