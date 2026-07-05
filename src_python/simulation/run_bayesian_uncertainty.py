from __future__ import annotations

"""Bayesian uncertainty analysis for the pertussis transmission model.

The production path uses deterministic quadrature over log(beta_S) after
fixing weakly identifiable reporting, vaccine-effect, asymptomatic
infectiousness, duration, and resistant-fitness nuisance parameters at
evidence-based calibrated values. This gives auditable conditional posterior
predictive intervals when the
pre-specified beta-grid tail and quadrature-resolution checks pass.

Adaptive Metropolis remains available for pilot diagnostics and historical
comparison, but it is not the default validated path because beta/reporting/VE
coupling produced unstable high-dimensional MCMC behavior in the country-level
calibrations.

Implemented features:
1. Deterministic beta-grid quadrature with tail, effective-grid-point, and
   maximum-single-weight validity gates.
2. Optional Savitzky-Golay smoothing of noisy log-posterior grids, with raw
   grid values retained for audit.
3. Adaptive Metropolis (Haario et al. 2001) for pilot comparison runs.
4. Rank-normalized split-R-hat and ESS summaries (Vehtari et al. 2021) for
   stochastic MCMC output and descriptive beta-grid quantile samples.
5. Dispersion (k) sensitivity sweep on the stochastic overlay.
6. Full multi-core parallelization across countries, chains, and posterior
   predictive scenarios.

References:
    Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis
    algorithm. Bernoulli, 7(2), 223-242.
    Vehtari, A. et al. (2021). Rank-normalization, folding, and localization:
    An improved R-hat. Bayesian Analysis, 16(2), 667-718.
    Lavine, J.S. et al. (2011). Natural immune boosting in pertussis dynamics
    and the potential for long-term vaccine failure. PNAS, 108(17), 7259-7264.
"""

import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import special as scipy_special

from src_python.calibration.calibrate_baseline import (
    align_annual_case_series,
    calibration_runtime_config,
    observed_annual_case_frame,
    predicted_case_frame,
    retain_recent_observed_window,
    reporting_rate_prior_penalty,
)
from src_python.calibration.likelihood import negative_binomial_nll
from src_python.calibration.mcmc_diagnostics import (
    compute_diagnostics,
    summarize_convergence,
)
from src_python.simulation.common import (
    execute_scenario_list,
    load_configs,
    make_config,
    run_prepared_config,
    write_outputs,
)
from src_python.simulation.stochastic_overlay import (
    StochasticOverlayConfig,
    decompose_variance,
    stochastic_overlay_samples,
    summarize_overlay_intervals,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import available_cpus, configure_worker_thread_limits, parallel_map


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

PARAMETER_NAMES = (
    "log_beta_S",
    "log_reporting_multiplier",
    "logit_VE_sus",
    "logit_VE_inf",
    "logit_relative_infectiousness_asymptomatic",
    "log_infectious_duration_symptomatic",
    "log_infectious_duration_asymptomatic",
    "logit_fitness_R_scaled",
)

N_PARAMS = len(PARAMETER_NAMES)
DEFAULT_OUTPUT_STEM = "bayesian_uncertainty"
DEFAULT_BETA_GRID_FIXED_PARAMETERS = (
    "reporting_multiplier",
    "VE_sus",
    "VE_inf",
    "relative_infectiousness_asymptomatic",
    "fitness_R",
)

PARAMETER_SAMPLE_COLUMNS = (
    "beta_S",
    "reporting_multiplier",
    "VE_sus",
    "VE_inf",
    "relative_infectiousness_asymptomatic",
    "infectious_duration_symptomatic",
    "infectious_duration_asymptomatic",
    "fitness_R",
)

PARAMETER_ALIASES = {
    "log_beta_S": "beta_S",
    "log_reporting_multiplier": "reporting_multiplier",
    "logit_VE_sus": "VE_sus",
    "logit_VE_inf": "VE_inf",
    "logit_relative_infectiousness_asymptomatic": "relative_infectiousness_asymptomatic",
    "log_infectious_duration_symptomatic": "infectious_duration_symptomatic",
    "log_infectious_duration_asymptomatic": "infectious_duration_asymptomatic",
    "logit_fitness_R_scaled": "fitness_R",
}

PARAMETER_INDEX_BY_SAMPLE = {
    name: idx for idx, name in enumerate(PARAMETER_SAMPLE_COLUMNS)
}
PARAMETER_INDEX_BY_NAME = {
    **{name: idx for idx, name in enumerate(PARAMETER_NAMES)},
    **PARAMETER_INDEX_BY_SAMPLE,
    **{alias: PARAMETER_INDEX_BY_SAMPLE[sample] for alias, sample in PARAMETER_ALIASES.items()},
}

# Initial diagonal proposal scales (used before adaptation kicks in).
# These are deliberately conservative because country likelihoods with monthly
# high-count data can be very sharp, especially for China/Japan.
INITIAL_PROPOSAL_SCALES = np.array(
    [0.025, 0.030, 0.035, 0.035, 0.035, 0.025, 0.030, 0.035],
    dtype=float,
)

# Adaptive Metropolis constants (Haario et al. 2001)
AM_EPSILON = 1e-5  # regularization for covariance (slightly larger for stability)
AM_SD = 2.4 ** 2 / N_PARAMS  # optimal scaling factor for Gaussian targets
AM_COMPONENTWISE_STEPS = 600  # componentwise phase length (~75 full parameter cycles)

# Robbins-Monro step-size adaptation (targets 23.4% acceptance for multivariate)
RM_TARGET_ACCEPTANCE = 0.234
RM_INITIAL_SCALE = 1.0
RM_GAMMA = 0.6  # decay exponent for step-size adaptation (0.5 < gamma < 1)
# Scale bounds.  The lower bound must be loose enough for high-information
# countries; if all chains sit at the lower bound with low acceptance, proposals
# are still too large.
RM_LOG_SCALE_MIN = -3.0
RM_LOG_SCALE_MAX = 2.5
LOCAL_PROPOSAL_PROBABILITY = 0.35
BLOCK_PROPOSAL_PROBABILITY = 0.45
PROPOSAL_BLOCKS = (
    (0, 1),  # transmission/reporting scale (strongly correlated)
    (2, 3, 4),  # vaccine acquisition, infectiousness, asymptomatic contribution
    (5, 6),  # infectious durations
    (7,),  # resistance fitness (single parameter)
)


@dataclass(frozen=True)
class ChainTask:
    country: str
    chain: int
    seed: int
    warmup: int
    draws: int
    proposal_scale: float
    enable_time_varying_reporting: bool = True
    thin: int = 2  # thinning interval: store every Nth post-warmup draw
    fix_durations: bool = False  # if True, fix infectious durations (6D sampling)
    fixed_parameters: tuple[str, ...] = ()
    sampler: str = "adaptive_mh"
    likelihood_observation_frequency: str = "monthly"
    dispersion: float = 50.0
    output_stem: str = DEFAULT_OUTPUT_STEM
    initial_samples_path: str | None = None
    initial_strategy: str = "calibrated"
    solver_mode: str = "mcmc_fast"
    prior_sd_scale: float | None = None
    beta_prior_log_sd: float | None = None
    reporting_prior_log_sd: float | None = None
    ve_prior_sd: float | None = None
    rel_asym_prior_sd: float | None = None
    fitness_prior_sd: float | None = None
    parameterization: str = "standard"
    grid_points: int = 161
    grid_log_beta_half_width: float = 0.08
    grid_max_points: int = 641
    grid_max_refinements: int = 10
    grid_tail_drop: float = 20.0
    grid_min_effective_points: float = 10.0
    grid_max_single_weight: float = 0.20
    grid_smoothing: str = "auto"
    grid_savgol_window: int = 21
    grid_n_chains: int = 4
    reuse_valid_beta_grid: bool = False
    grid_eval_jobs: int = 1


@dataclass
class AdaptiveState:
    """Running statistics for Adaptive Metropolis covariance estimation."""
    n: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(N_PARAMS))
    cov: np.ndarray = field(default_factory=lambda: np.eye(N_PARAMS))
    _sum: np.ndarray = field(default_factory=lambda: np.zeros(N_PARAMS))
    _sum_sq: np.ndarray = field(default_factory=lambda: np.zeros((N_PARAMS, N_PARAMS)))

    def update(self, x: np.ndarray) -> None:
        """Welford-style online covariance update."""
        self.n += 1
        self._sum += x
        self._sum_sq += np.outer(x, x)
        self.mean = self._sum / self.n
        if self.n > 1:
            self.cov = (self._sum_sq / self.n - np.outer(self.mean, self.mean)) + AM_EPSILON * np.eye(N_PARAMS)

    def proposal_cov(self) -> np.ndarray:
        """Return the adapted proposal covariance matrix."""
        return AM_SD * self.cov


# ---------------------------------------------------------------------------
# Transform utilities
# ---------------------------------------------------------------------------

def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-9, 1.0 - 1e-9))
    return float(np.log(value / (1.0 - value)))


def _inv_logit(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def _beta_ab(mean: float, sd: float) -> tuple[float, float]:
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    sd = max(float(sd), 1e-6)
    variance = min(sd**2, mean * (1.0 - mean) * 0.95)
    common = mean * (1.0 - mean) / variance - 1.0
    return max(mean * common, 1e-3), max((1.0 - mean) * common, 1e-3)


def _beta_logpdf(value: float, mean: float, sd: float) -> float:
    if not 0.0 < value < 1.0:
        return -np.inf
    alpha, beta = _beta_ab(mean, sd)
    return float(
        (alpha - 1.0) * np.log(value)
        + (beta - 1.0) * np.log1p(-value)
        + scipy_special.gammaln(alpha + beta)
        - scipy_special.gammaln(alpha)
        - scipy_special.gammaln(beta)
    )


def _normal_logpdf(value: float, mean: float, sd: float) -> float:
    sd = max(float(sd), 1e-9)
    z = (float(value) - float(mean)) / sd
    return float(-0.5 * z * z - np.log(sd) - 0.5 * np.log(2.0 * np.pi))


def _scaled_logit_to_value(x: float, lower: float, upper: float) -> float:
    s = _inv_logit(x)
    return float(lower + (upper - lower) * s)


def _value_to_scaled_logit(value: float, lower: float, upper: float) -> float:
    scaled = (float(value) - lower) / max(upper - lower, 1e-12)
    return _logit(float(np.clip(scaled, 1e-9, 1.0 - 1e-9)))


def _log_jacobian_logit(x: float) -> float:
    s = _inv_logit(x)
    return float(np.log(max(s, 1e-12)) + np.log(max(1.0 - s, 1e-12)))


def _log_jacobian_scaled_logit(x: float, lower: float, upper: float) -> float:
    return float(np.log(max(upper - lower, 1e-12)) + _log_jacobian_logit(x))


def _artifact_stem(output_stem: str, canonical_stem: str, suffix: str) -> str:
    """Return canonical artifact names for production and isolated names for pilots."""
    output_stem = str(output_stem or DEFAULT_OUTPUT_STEM)
    if output_stem == DEFAULT_OUTPUT_STEM:
        return canonical_stem
    return f"{output_stem}_{suffix}"


def _mcmc_progress_dir(output_stem: str) -> Any:
    if str(output_stem or DEFAULT_OUTPUT_STEM) == DEFAULT_OUTPUT_STEM:
        return project_path("outputs", "metadata", "mcmc_progress")
    return project_path("outputs", "metadata", f"mcmc_progress_{output_stem}")


def _parameter_index(name: str) -> int:
    key = str(name).strip()
    if key not in PARAMETER_INDEX_BY_NAME:
        valid = sorted(PARAMETER_INDEX_BY_NAME)
        raise ValueError(f"Unknown posterior parameter '{name}'. Valid values include: {valid}")
    return int(PARAMETER_INDEX_BY_NAME[key])


def _fixed_parameter_indices(
    fixed_parameters: tuple[str, ...] | list[str] | None,
    *,
    fix_durations: bool,
) -> set[int]:
    fixed_indices = {_parameter_index(name) for name in (fixed_parameters or ()) if str(name).strip()}
    if fix_durations:
        fixed_indices.update(
            {
                PARAMETER_INDEX_BY_SAMPLE["infectious_duration_symptomatic"],
                PARAMETER_INDEX_BY_SAMPLE["infectious_duration_asymptomatic"],
            }
        )
    return fixed_indices


# ---------------------------------------------------------------------------
# Data loading and parameter vector construction
# ---------------------------------------------------------------------------

def _country_observed(country: str, interval: str = "native") -> pd.DataFrame:
    configs = load_configs()
    observed = observed_annual_case_frame(country)
    recent_years = int(configs["baseline"].get("calibration", {}).get("recent_years", 0))
    observed = retain_recent_observed_window(observed, recent_years)
    return _aggregate_observed_intervals(observed, interval)


def _aggregate_observed_intervals(observed: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate high-frequency surveillance intervals for Bayesian likelihoods.

    Weekly reports are autocorrelated and much noisier than the deterministic
    ODE can represent.  Treating every week as an independent NB observation
    over-weights large high-frequency datasets and creates pathologically
    narrow posteriors.  Monthly aggregation preserves the epidemic-scale signal
    while conserving reported cases.
    """
    interval = str(interval or "native").lower()
    if interval in {"native", "none", "reporting_interval"}:
        return observed
    if interval not in {"monthly", "month"}:
        raise ValueError(f"Unsupported Bayesian likelihood interval: {interval}")
    if not {"period_start", "period_end"}.issubset(observed.columns) or observed.empty:
        return observed

    frame = observed.copy()
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce")
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["reported_cases"] = pd.to_numeric(frame["reported_cases"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["period_start", "period_end"]).copy()
    frame = frame.loc[frame["period_end"].gt(frame["period_start"])].copy()
    if frame.empty:
        return observed.iloc[0:0].copy()

    data_start = frame["period_start"].min()
    data_end = frame["period_end"].max()
    first_month = data_start.to_period("M")
    last_month = (data_end - pd.Timedelta(nanoseconds=1)).to_period("M")
    rows: list[dict[str, Any]] = []

    source_starts = frame["period_start"].to_numpy(dtype="datetime64[ns]")
    source_ends = frame["period_end"].to_numpy(dtype="datetime64[ns]")
    source_days = (source_ends - source_starts).astype("timedelta64[s]").astype(float) / 86400.0
    source_days = np.maximum(source_days, 1e-9)
    source_cases = frame["reported_cases"].to_numpy(dtype=float)

    for period in pd.period_range(first_month, last_month, freq="M"):
        month_start = pd.Timestamp(period.start_time)
        month_end = pd.Timestamp((period + 1).start_time)
        covered_start = max(month_start, data_start)
        covered_end = min(month_end, data_end)
        if covered_end <= covered_start:
            continue

        target_start = np.datetime64(covered_start.to_datetime64())
        target_end = np.datetime64(covered_end.to_datetime64())
        overlap_start = np.maximum(source_starts, target_start)
        overlap_end = np.minimum(source_ends, target_end)
        overlap_days = (overlap_end - overlap_start).astype("timedelta64[s]").astype(float) / 86400.0
        mask = overlap_days > 0.0
        reported = float(np.sum(source_cases[mask] * overlap_days[mask] / source_days[mask]))
        start_date = covered_start.date().isoformat()
        end_date = covered_end.date().isoformat()
        rows.append(
            {
                "series_year": len(rows),
                "observed_year": int(covered_start.year),
                "observed_interval_id": f"{start_date}_{end_date}",
                "period_start": covered_start,
                "period_end": covered_end,
                "interval_days": float((covered_end - covered_start).days),
                "reported_cases": reported,
                "reporting_frequency": "monthly_aggregated",
            }
        )

    return pd.DataFrame(rows)


def _parameterization(priors: dict[str, Any]) -> str:
    return str(priors.get("parameterization", "standard") or "standard").lower()


def _base_log_beta(priors: dict[str, Any], config: dict[str, Any] | None = None) -> float:
    if "base_log_beta_S" in priors:
        return float(priors["base_log_beta_S"])
    if config is not None:
        return float(np.log(float(config["transmission"]["beta_S"])))
    return 0.0


def _encode_reporting_coordinate(log_beta: float, log_reporting: float, priors: dict[str, Any]) -> float:
    """Encode reporting coordinate for the selected transformed parameterization."""
    if _parameterization(priors) == "beta_reporting_product":
        return float(log_reporting + log_beta - _base_log_beta(priors))
    return float(log_reporting)


def _decode_reporting_coordinate(log_beta: float, reporting_coordinate: float, priors: dict[str, Any]) -> float:
    """Decode log reporting multiplier from the selected transformed coordinate."""
    if _parameterization(priors) == "beta_reporting_product":
        return float(reporting_coordinate - log_beta + _base_log_beta(priors))
    return float(reporting_coordinate)


def _initial_vector(
    config: dict[str, Any],
    enable_trend: bool = True,
    priors: dict[str, Any] | None = None,
) -> np.ndarray:
    fitness_bounds = load_configs()["baseline"]["bayesian_uncertainty"]["priors"]["fitness_R"]
    priors = dict(priors or {})
    log_beta = np.log(float(config["transmission"]["beta_S"]))
    log_reporting = np.log(float(config.get("reporting_multiplier", 1.0)))
    vec = [
        log_beta,
        _encode_reporting_coordinate(log_beta, log_reporting, priors),
        _logit(float(config["vaccine"].get("VE_sus", 0.0))),
        _logit(float(config["vaccine"].get("VE_inf", 0.0))),
        _logit(float(config["transmission"]["relative_infectiousness_asymptomatic"])),
        np.log(float(config["natural_history"]["infectious_duration_symptomatic"])),
        np.log(float(config["natural_history"]["infectious_duration_asymptomatic"])),
        _value_to_scaled_logit(
            float(config["transmission"].get("fitness_R", 1.0)),
            float(fitness_bounds.get("min", 0.70)),
            float(fitness_bounds.get("max", 1.25)),
        ),
    ]
    return np.array(vec, dtype=float)


def _sample_from_vector(vector: np.ndarray, priors: dict[str, Any]) -> dict[str, float]:
    fitness = priors["fitness_R"]
    log_beta = float(vector[0])
    log_reporting = _decode_reporting_coordinate(log_beta, float(vector[1]), priors)
    return {
        "beta_S": float(np.exp(log_beta)),
        "reporting_multiplier": float(np.exp(log_reporting)),
        "VE_sus": _inv_logit(vector[2]),
        "VE_inf": _inv_logit(vector[3]),
        "VE_dur": float(
            priors.get(
                "VE_dur_fixed",
                priors.get("VE_dur", {}).get("mean", 0.10),
            )
        ),
        "relative_infectiousness_asymptomatic": _inv_logit(vector[4]),
        "infectious_duration_symptomatic": float(np.exp(vector[5])),
        "infectious_duration_asymptomatic": float(np.exp(vector[6])),
        "fitness_R": _scaled_logit_to_value(
            vector[7],
            float(fitness.get("min", 0.70)),
            float(fitness.get("max", 1.25)),
        ),
        "resistance_prevalence": float(priors.get("resistance_prevalence_fixed", 0.30)),
        "reporting_trend_end_multiplier": float(priors.get("reporting_trend_fixed", 1.0)),
    }


def _vector_from_sample(sample: dict[str, float], priors: dict[str, Any]) -> np.ndarray:
    fitness = priors["fitness_R"]
    log_beta = np.log(float(sample["beta_S"]))
    log_reporting = np.log(float(sample["reporting_multiplier"]))
    return np.array(
        [
            log_beta,
            _encode_reporting_coordinate(log_beta, log_reporting, priors),
            _logit(float(sample["VE_sus"])),
            _logit(float(sample["VE_inf"])),
            _logit(float(sample["relative_infectiousness_asymptomatic"])),
            np.log(float(sample["infectious_duration_symptomatic"])),
            np.log(float(sample["infectious_duration_asymptomatic"])),
            _value_to_scaled_logit(
                float(sample["fitness_R"]),
                float(fitness.get("min", 0.70)),
                float(fitness.get("max", 1.25)),
            ),
        ],
        dtype=float,
    )


def _initial_vector_from_samples(
    path: str,
    country: str,
    chain: int,
    priors: dict[str, Any],
    strategy: str,
) -> np.ndarray:
    samples = pd.read_parquet(project_path(path) if not Path(path).is_absolute() else path)
    samples = samples.loc[samples["country"].astype(str).eq(country)].copy()
    if samples.empty:
        raise ValueError(f"No initial samples for country {country} in {path}")

    strategy = str(strategy).lower()
    if strategy == "chain_best":
        chain_samples = samples.loc[samples["chain"].eq(chain)]
        if not chain_samples.empty:
            samples = chain_samples
        row = samples.loc[samples["posterior_log_prob"].idxmax()]
    elif strategy == "sample_random":
        rng = np.random.default_rng(20260510 + chain * 1009)
        row = samples.iloc[int(rng.integers(0, len(samples)))]
    elif strategy in {"sample_best", "posterior_best"}:
        row = samples.loc[samples["posterior_log_prob"].idxmax()]
    else:
        raise ValueError(f"Unsupported initial sample strategy: {strategy}")

    sample = {
        key: float(row[key])
        for key in PARAMETER_SAMPLE_COLUMNS
        if key in row.index
    }
    return _vector_from_sample(sample, priors)


def _apply_sample(config: dict[str, Any], sample: dict[str, float]) -> dict[str, Any]:
    out = deepcopy(config)
    out["transmission"]["beta_S"] = sample["beta_S"]
    out["reporting_multiplier"] = sample["reporting_multiplier"]
    out["vaccine"]["VE_sus"] = sample["VE_sus"]
    out["vaccine"]["VE_inf"] = sample["VE_inf"]
    out["vaccine"]["VE_dur"] = sample["VE_dur"]
    out["transmission"]["relative_infectiousness_asymptomatic"] = sample[
        "relative_infectiousness_asymptomatic"
    ]
    out["natural_history"]["infectious_duration_symptomatic"] = sample[
        "infectious_duration_symptomatic"
    ]
    out["natural_history"]["infectious_duration_asymptomatic"] = sample[
        "infectious_duration_asymptomatic"
    ]
    out["transmission"]["fitness_R"] = sample["fitness_R"]

    # Resistance prevalence is fixed at the country-calibrated value (not sampled)
    resistance = float(np.clip(sample["resistance_prevalence"], 0.0, 1.0))
    out["initial_conditions"]["initial_resistance_prevalence"] = resistance
    out.setdefault("resistance", {})["target_prevalence_at_analysis_start"] = resistance
    out["resistance"]["importation_fraction"] = resistance
    out.setdefault("importation", {})["resistant_fraction"] = resistance

    # Reporting trend is fixed at 1.0 (no secular change assumed)
    # The reporting_multiplier parameter absorbs the average level.
    trend_end = sample.get("reporting_trend_end_multiplier", 1.0)
    simulation = out.get("simulation", {})
    out["reporting_time_variation"] = {
        "start_time": float(simulation.get("start_time", 0.0)),
        "end_time": float(simulation.get("end_time", 0.0)),
        "start_multiplier": 1.0,
        "end_multiplier": float(trend_end),
    }
    return out


# ---------------------------------------------------------------------------
# Prior and posterior
# ---------------------------------------------------------------------------

def _resistance_prior(country: str, target: float, settings: dict[str, Any]) -> tuple[float, float]:
    path = project_path("data", "raw", "country_resistance_timeline.csv")
    floor_sd = float(settings["priors"].get("resistance_prevalence", {}).get("floor_sd", 0.03))
    if not path.exists():
        return float(np.clip(target, 1e-6, 1.0 - 1e-6)), max(floor_sd, 0.05)
    timeline = pd.read_csv(path)
    rows = timeline.loc[timeline["country"].astype(str).eq(country)].copy()
    if rows.empty:
        return float(np.clip(target, 1e-6, 1.0 - 1e-6)), max(floor_sd, 0.05)
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce")
    rows = rows.sort_values("year").dropna(subset=["year"])
    row = rows.iloc[-1]
    lower = pd.to_numeric(pd.Series([row.get("lower", np.nan)]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([row.get("upper", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        sd = max(float(upper - lower) / 3.92, floor_sd)
    else:
        sample_size = pd.to_numeric(pd.Series([row.get("sample_size", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(sample_size) and sample_size > 0:
            sd = np.sqrt(max(target * (1.0 - target), 1e-6) / float(sample_size))
            sd = max(float(sd), floor_sd)
        else:
            sd = max(floor_sd, 0.05)
    return float(np.clip(target, 1e-6, 1.0 - 1e-6)), float(sd)


def _log_prior(
    vector: np.ndarray,
    base_config: dict[str, Any],
    country: str,
    settings: dict[str, Any],
) -> float:
    priors = settings["priors"]
    sample = _sample_from_vector(vector, priors)

    # Hard bounds
    if not 0.0005 <= sample["beta_S"] <= 0.5:
        return -np.inf
    if not 0.02 <= sample["reporting_multiplier"] <= 20.0:
        return -np.inf
    if not 7.0 <= sample["infectious_duration_symptomatic"] <= 35.0:
        return -np.inf
    if not 5.0 <= sample["infectious_duration_asymptomatic"] <= 28.0:
        return -np.inf

    logp = 0.0

    # Log-normal priors for positive parameters
    logp += _normal_logpdf(
        np.log(sample["beta_S"]),
        np.log(float(base_config["transmission"]["beta_S"])),
        float(priors.get("log_beta_S_sd", 0.40)),
    )
    logp += _normal_logpdf(
        np.log(sample["reporting_multiplier"]),
        np.log(float(base_config.get("reporting_multiplier", 1.0))),
        float(priors.get("log_reporting_multiplier_sd", 0.40)),
    )

    # Beta priors for bounded [0,1] parameters
    for key, vector_idx in (
        ("VE_sus", 2),
        ("VE_inf", 3),
        ("relative_infectiousness_asymptomatic", 4),
    ):
        prior = priors[key]
        logp += _beta_logpdf(sample[key], float(prior["mean"]), float(prior["sd"]))
        logp += _log_jacobian_logit(vector[vector_idx])

    # Log-normal priors for durations
    logp += _normal_logpdf(
        np.log(sample["infectious_duration_symptomatic"]),
        np.log(float(base_config["natural_history"]["infectious_duration_symptomatic"])),
        float(priors["infectious_duration_symptomatic"].get("log_sd", 0.15)),
    )
    logp += _normal_logpdf(
        np.log(sample["infectious_duration_asymptomatic"]),
        np.log(float(base_config["natural_history"]["infectious_duration_asymptomatic"])),
        float(priors["infectious_duration_asymptomatic"].get("log_sd", 0.20)),
    )

    # Normal prior for fitness_R on bounded scale
    fitness_prior = priors["fitness_R"]
    logp += _normal_logpdf(
        sample["fitness_R"],
        float(fitness_prior.get("mean", 1.00)),
        float(fitness_prior.get("sd", 0.12)),
    )
    logp += _log_jacobian_scaled_logit(
        vector[7],
        float(fitness_prior.get("min", 0.70)),
        float(fitness_prior.get("max", 1.25)),
    )

    return float(logp)


def _log_likelihood(
    config: dict[str, Any],
    observed: pd.DataFrame,
    country: str,
    dispersion: float,
    _cache: dict[str, float] | None = None,
) -> float:
    """Compute log-likelihood with optional caching to avoid redundant ODE solves.
    
    For sampler efficiency, we cache results by parameter hash to avoid re-solving
    the ODE for identical parameter sets (which can happen during rejection steps).
    """
    try:
        # Create a hashable key from the parameter values
        if _cache is not None:
            # Hash the key parameters that affect the likelihood
            key_params = (
                config["transmission"]["beta_S"],
                config.get("reporting_multiplier", 1.0),
                config.get("reporting_time_variation", {}).get("end_multiplier", 1.0),
                config["vaccine"]["VE_sus"],
                config["vaccine"]["VE_inf"],
                config["transmission"]["relative_infectiousness_asymptomatic"],
                config["natural_history"]["infectious_duration_symptomatic"],
                config["natural_history"]["infectious_duration_asymptomatic"],
                config["transmission"].get("fitness_R", 1.0),
                config["initial_conditions"]["initial_resistance_prevalence"],
            )
            cache_key = hash(key_params)
            if cache_key in _cache:
                return _cache[cache_key]
        
        timeseries, _ = run_prepared_config(
            config,
            analysis="bayesian_calibration",
            scenario="candidate",
            vaccine_scenario=config["baseline_vaccine_scenario"],
            resistance_scenario=config["baseline_resistance_scenario"],
            metadata={"country": country},
        )
        predicted = predicted_case_frame(timeseries, observed)
        aligned = align_annual_case_series(predicted, observed)
        if len(aligned) < 3:
            result = -np.inf
        else:
            result = float(
                -negative_binomial_nll(
                    aligned["observed_reported_cases"].to_numpy(dtype=float),
                    aligned["predicted_reported_cases"].to_numpy(dtype=float),
                    dispersion=dispersion,
                )
            )
        
        # Cache the result
        if _cache is not None:
            _cache[cache_key] = result
        
        return result
    except Exception:
        return -np.inf


def _log_posterior(
    vector: np.ndarray,
    base_config: dict[str, Any],
    observed: pd.DataFrame,
    country: str,
    settings: dict[str, Any],
    _cache: dict[str, float] | None = None,
) -> tuple[float, dict[str, float] | None]:
    prior = _log_prior(vector, base_config, country, settings)
    if not isfinite(prior):
        return -np.inf, None
    sample = _sample_from_vector(vector, settings["priors"])
    candidate = _apply_sample(base_config, sample)
    likelihood = _log_likelihood(
        candidate, observed, country, float(settings.get("dispersion", 50.0)), _cache=_cache
    )
    if not isfinite(likelihood):
        return -np.inf, None
    posterior = likelihood + prior - reporting_rate_prior_penalty(candidate)
    return float(posterior), sample


def _weighted_grid_quantiles(
    grid_values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Invert a discrete weighted grid CDF with linear interpolation."""
    order = np.argsort(grid_values)
    x = np.asarray(grid_values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    if len(x) == 0 or len(w) == 0 or not np.isfinite(w).any() or float(np.sum(w)) <= 0.0:
        raise ValueError("Cannot sample from an empty or zero-weight posterior grid")
    w = np.clip(w, 0.0, np.inf)
    w = w / float(np.sum(w))
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    cdf_x = np.concatenate(([0.0], cdf))
    value_x = np.concatenate(([x[0]], x))
    return np.interp(np.clip(probabilities, 0.0, 1.0), cdf_x, value_x)


def _sample_beta_grid_draws(
    task: ChainTask,
    settings: dict[str, Any],
    calibrated_start: np.ndarray,
    grid_x: np.ndarray,
    log_post_arr: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Convert a deterministic beta grid into chain-formatted quantile draws."""
    base_probs = (np.arange(task.draws, dtype=float) + 0.5) / float(task.draws)
    rows: list[dict[str, Any]] = []
    for chain in range(1, int(max(task.grid_n_chains, 1)) + 1):
        rng = np.random.default_rng(task.seed + chain * 1009)
        probs = base_probs.copy()
        rng.shuffle(probs)
        sampled_log_beta = _weighted_grid_quantiles(grid_x, weights, probs)
        sampled_logp = np.interp(sampled_log_beta, grid_x, log_post_arr)
        for draw_idx, (log_beta, lp) in enumerate(zip(sampled_log_beta, sampled_logp), start=1):
            sample_vector = calibrated_start.copy()
            sample_vector[PARAMETER_INDEX_BY_SAMPLE["beta_S"]] = float(log_beta)
            sample = _sample_from_vector(sample_vector, settings["priors"])
            row = {
                "country": task.country,
                "chain": chain,
                "draw": draw_idx,
                "step": draw_idx,
                "posterior_log_prob": float(lp),
                "accepted_fraction": 1.0,
                "sampling_method": "beta_grid",
            }
            row.update(sample)
            rows.append(row)
    return pd.DataFrame(rows)


def _normalised_grid_weights(log_posterior: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize grid log posterior values and return quadrature quality metrics."""
    logp = np.asarray(log_posterior, dtype=float)
    finite = np.isfinite(logp)
    weights = np.zeros_like(logp, dtype=float)
    if len(logp) == 0 or not finite.any():
        return weights, {
            "finite": 0.0,
            "max_logp": -np.inf,
            "left_edge_drop": 0.0,
            "right_edge_drop": 0.0,
            "min_edge_drop": 0.0,
            "grid_effective_points": 0.0,
            "grid_max_weight": 1.0,
        }

    safe_logp = logp.copy()
    safe_logp[~finite] = -np.inf
    max_logp = float(np.max(safe_logp))
    weights = np.exp(safe_logp - float(scipy_special.logsumexp(safe_logp)))

    left_edge_drop = (
        float(max_logp - safe_logp[0]) if np.isfinite(safe_logp[0]) else np.inf
    )
    right_edge_drop = (
        float(max_logp - safe_logp[-1]) if np.isfinite(safe_logp[-1]) else np.inf
    )
    min_edge_drop = float(min(left_edge_drop, right_edge_drop))
    grid_effective_points = 1.0 / float(np.sum(weights ** 2))
    grid_max_weight = float(np.max(weights))
    return weights, {
        "finite": 1.0,
        "max_logp": max_logp,
        "left_edge_drop": left_edge_drop,
        "right_edge_drop": right_edge_drop,
        "min_edge_drop": min_edge_drop,
        "grid_effective_points": grid_effective_points,
        "grid_max_weight": grid_max_weight,
    }


def _estimate_local_mode_half_width(
    grid_values: np.ndarray,
    log_posterior: np.ndarray,
    tail_drop_target: float,
) -> float | None:
    """Estimate a useful half-width from curvature near the grid mode."""
    x = np.asarray(grid_values, dtype=float)
    logp = np.asarray(log_posterior, dtype=float)
    finite = np.isfinite(logp)
    if len(x) < 3 or not finite.any():
        return None

    safe_logp = np.where(finite, logp, -np.inf)
    mode_idx = int(np.nanargmax(safe_logp))
    mode_x = float(x[mode_idx])
    mode_logp = float(safe_logp[mode_idx])
    estimates: list[float] = []
    max_offset = min(6, len(x) - 1)
    for offset in range(1, max_offset + 1):
        for idx in (mode_idx - offset, mode_idx + offset):
            if idx < 0 or idx >= len(x) or not np.isfinite(safe_logp[idx]):
                continue
            distance = abs(float(x[idx] - mode_x))
            drop = mode_logp - float(safe_logp[idx])
            if distance <= 0.0 or drop <= 0.05:
                continue
            estimates.append(distance * np.sqrt(tail_drop_target / drop))
        if len(estimates) >= 4:
            break

    if not estimates:
        return None
    return float(np.median(np.asarray(estimates, dtype=float)) * 1.15)


def _smooth_grid_log_posterior_savgol(
    log_posterior: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Return a smooth surrogate for grid integration while retaining scale."""
    raw = np.asarray(log_posterior, dtype=float)
    if len(raw) < 5:
        return raw.copy()
    finite = np.isfinite(raw)
    if finite.sum() < 5:
        return raw.copy()

    x = np.arange(len(raw), dtype=float)
    y = raw.copy()
    if not finite.all():
        y[~finite] = np.interp(x[~finite], x[finite], raw[finite])

    win = int(max(window, 5))
    if win % 2 == 0:
        win += 1
    max_win = len(raw) if len(raw) % 2 == 1 else len(raw) - 1
    win = min(win, max_win)
    if win < 5:
        return raw.copy()
    smoothed = savgol_filter(y, window_length=win, polyorder=2, mode="interp")
    smoothed = np.asarray(smoothed, dtype=float)
    smoothed += float(np.nanmax(raw[finite]) - np.nanmax(smoothed[finite]))
    smoothed[~finite] = -np.inf
    return smoothed


def _integration_grid_log_posterior(
    log_posterior: np.ndarray,
    *,
    smoothing: str,
    savgol_window: int,
    min_effective_points: float = 10.0,
    max_single_weight: float = 0.20,
) -> tuple[np.ndarray, str]:
    """Choose raw or smoothed log posterior values for numerical quadrature."""
    method = str(smoothing or "auto").lower()
    if method not in {"auto", "none", "savgol"}:
        raise ValueError("--grid-smoothing must be one of: auto, none, savgol")
    if method == "none":
        return np.asarray(log_posterior, dtype=float).copy(), "none"

    _, raw_quality = _normalised_grid_weights(log_posterior)
    should_smooth = method == "savgol" or (
        raw_quality["finite"] > 0.0
        and (
            raw_quality["grid_max_weight"] > max_single_weight
            or raw_quality["grid_effective_points"] < min_effective_points
        )
    )
    if not should_smooth:
        return np.asarray(log_posterior, dtype=float).copy(), "none"
    return (
        _smooth_grid_log_posterior_savgol(log_posterior, window=savgol_window),
        "savgol",
    )


def _grid_quality_score(
    quality: dict[str, float],
    *,
    tail_drop_target: float,
    min_effective_points: float,
    max_single_weight: float,
) -> float:
    """Score beta-grid quality for selecting the best refinement attempt."""
    if quality.get("finite", 0.0) <= 0.0:
        return -np.inf
    edge_score = min(float(quality["min_edge_drop"]) / tail_drop_target, 1.0)
    ess_score = min(float(quality["grid_effective_points"]) / min_effective_points, 1.0)
    weight_score = min(max_single_weight / max(float(quality["grid_max_weight"]), 1e-12), 1.0)
    return float(edge_score + ess_score + weight_score)


def _evaluate_grid_values(
    grid: np.ndarray,
    evaluator: Callable[[float], float],
    *,
    n_jobs: int,
    progress_file: Path | None = None,
    half_width_value: float | None = None,
) -> np.ndarray:
    """Evaluate independent one-dimensional grid points, optionally in chunks."""
    grid_values = np.asarray(grid, dtype=float)
    workers = min(max(1, int(n_jobs)), available_cpus(), max(1, len(grid_values)))
    if workers <= 1:
        values: list[float] = []
        for i, x in enumerate(grid_values, start=1):
            values.append(float(evaluator(float(x))))
            if progress_file is not None and (
                i == 1 or i == len(grid_values) or i % max(len(grid_values) // 5, 1) == 0
            ):
                try:
                    progress_file.write_text(
                        f"grid {i}/{len(grid_values)} half_width={half_width_value:.4g}"
                    )
                except OSError:
                    pass
        return np.asarray(values, dtype=float)

    configure_worker_thread_limits()
    chunks = [
        chunk
        for chunk in np.array_split(np.arange(len(grid_values), dtype=int), workers)
        if len(chunk) > 0
    ]

    def evaluate_chunk(indices: np.ndarray) -> list[tuple[int, float]]:
        return [(int(idx), float(evaluator(float(grid_values[int(idx)])))) for idx in indices]

    if progress_file is not None:
        try:
            progress_file.write_text(
                f"grid 1/{len(grid_values)} half_width={half_width_value:.4g}"
            )
        except OSError:
            pass
    chunk_results = Parallel(n_jobs=workers, backend="threading")(
        delayed(evaluate_chunk)(chunk) for chunk in chunks
    )
    values = np.empty(len(grid_values), dtype=float)
    completed = 0
    for chunk_result in chunk_results:
        completed += len(chunk_result)
        for idx, value in chunk_result:
            values[idx] = value
        if progress_file is not None:
            try:
                progress_file.write_text(
                    f"grid {completed}/{len(grid_values)} half_width={half_width_value:.4g}"
                )
            except OSError:
                pass
    return values


def _existing_beta_grid_valid(
    country: str,
    output_stem: str,
    *,
    tail_drop_target: float,
    min_effective_points: float,
    max_single_weight: float,
) -> bool:
    grid_path = project_path("outputs", "metadata", f"beta_grid_{output_stem}", f"{country}_grid.csv")
    if not grid_path.exists():
        return False
    try:
        existing_grid = pd.read_csv(grid_path)
    except (OSError, pd.errors.ParserError):
        return False
    if not {"log_beta_S", "log_posterior"}.issubset(existing_grid.columns) or existing_grid.empty:
        return False
    existing_logp = existing_grid["log_posterior"].to_numpy(dtype=float)
    _, existing_quality = _normalised_grid_weights(existing_logp)
    return bool(
        existing_quality["finite"] > 0.0
        and existing_quality["min_edge_drop"] >= tail_drop_target
        and existing_quality["grid_effective_points"] >= min_effective_points
        and existing_quality["grid_max_weight"] <= max_single_weight
    )


def _evaluate_beta_grid_log_posteriors(
    grid: np.ndarray,
    *,
    calibrated_start: np.ndarray,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    settings: dict[str, Any],
    country: str,
    n_jobs: int,
    progress_file: Path | None = None,
    half_width_value: float | None = None,
) -> np.ndarray:
    grid_values = np.asarray(grid, dtype=float)
    workers = min(max(1, int(n_jobs)), available_cpus(), max(1, len(grid_values)))
    if workers <= 1:
        def evaluator(x: float) -> float:
            vector = calibrated_start.copy()
            vector[PARAMETER_INDEX_BY_SAMPLE["beta_S"]] = float(x)
            lp, _ = _log_posterior(
                vector,
                runtime_base,
                observed,
                country,
                settings,
                _cache=None,
            )
            return float(lp)

        return _evaluate_grid_values(
            grid_values,
            evaluator,
            n_jobs=1,
            progress_file=progress_file,
            half_width_value=half_width_value,
        )

    from src_python.simulation.bayesian_grid_worker import evaluate_beta_grid_chunk

    configure_worker_thread_limits()
    chunks = [
        chunk
        for chunk in np.array_split(np.arange(len(grid_values), dtype=int), workers)
        if len(chunk) > 0
    ]
    payloads = [
        {
            "indices": chunk,
            "grid_values": grid_values[chunk],
            "calibrated_start": calibrated_start,
            "runtime_base": runtime_base,
            "observed": observed,
            "settings": settings,
            "country": country,
        }
        for chunk in chunks
    ]

    if progress_file is not None:
        try:
            progress_file.write_text(
                f"grid 1/{len(grid_values)} half_width={half_width_value:.4g}"
            )
        except OSError:
            pass
    chunk_results = Parallel(n_jobs=workers, backend="loky", inner_max_num_threads=1)(
        delayed(evaluate_beta_grid_chunk)(payload) for payload in payloads
    )
    values = np.empty(len(grid_values), dtype=float)
    completed = 0
    for chunk_result in chunk_results:
        completed += len(chunk_result)
        for idx, value in chunk_result:
            values[idx] = value
        if progress_file is not None:
            try:
                progress_file.write_text(
                    f"grid {completed}/{len(grid_values)} half_width={half_width_value:.4g}"
                )
            except OSError:
                pass
    return values


def _run_beta_grid(
    task: ChainTask,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    settings: dict[str, Any],
    calibrated_start: np.ndarray,
    active_indices: np.ndarray,
) -> pd.DataFrame:
    """Deterministic one-dimensional posterior integration over log(beta_S).

    This is intended for the identified sensitivity-analysis model where all
    weakly identified nuisance parameters are externally fixed and only beta_S
    remains uncertain. It avoids fragile one-dimensional MH convergence while
    preserving the standard posterior-sample artifact used downstream.
    """
    if list(map(int, active_indices)) != [PARAMETER_INDEX_BY_SAMPLE["beta_S"]]:
        active_names = [PARAMETER_SAMPLE_COLUMNS[int(idx)] for idx in active_indices]
        raise ValueError(
            "beta_grid sampler requires exactly beta_S to remain sampled; "
            f"active sampled parameters are {active_names}. Fix reporting, VE/asym, "
            "fitness_R, and durations before using --sampler beta_grid."
        )

    n_chains = int(max(task.grid_n_chains, 1))
    draws = int(max(task.draws, 1))
    grid_points = int(max(task.grid_points, 21))
    if grid_points % 2 == 0:
        grid_points += 1
    grid_max_points = int(max(task.grid_max_points, grid_points))
    if grid_max_points % 2 == 0:
        grid_max_points += 1
    half_width = float(task.grid_log_beta_half_width)
    if half_width <= 0.0:
        raise ValueError("--grid-log-beta-half-width must be positive")
    tail_drop_target = float(max(task.grid_tail_drop, 5.0))
    max_refinements = int(max(task.grid_max_refinements, 0))
    min_effective_points = float(max(task.grid_min_effective_points, 1.0))
    max_single_weight = float(min(max(task.grid_max_single_weight, 0.01), 1.0))
    grid_smoothing = str(task.grid_smoothing or "auto").lower()
    savgol_window = int(max(task.grid_savgol_window, 5))

    progress_dir = _mcmc_progress_dir(task.output_stem)
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"{task.country}_beta_grid.txt"
    grid_dir = project_path("outputs", "metadata", f"beta_grid_{task.output_stem}")
    grid_dir.mkdir(parents=True, exist_ok=True)

    existing_grid_path = grid_dir / f"{task.country}_grid.csv"
    if task.reuse_valid_beta_grid and existing_grid_path.exists():
        existing_grid = pd.read_csv(existing_grid_path)
        if {"log_beta_S", "log_posterior"}.issubset(existing_grid.columns) and not existing_grid.empty:
            existing_logp = existing_grid["log_posterior"].to_numpy(dtype=float)
            existing_weights, existing_quality = _normalised_grid_weights(existing_logp)
            existing_valid = (
                existing_quality["finite"] > 0.0
                and existing_quality["min_edge_drop"] >= tail_drop_target
                and existing_quality["grid_effective_points"] >= min_effective_points
                and existing_quality["grid_max_weight"] <= max_single_weight
            )
            if existing_valid:
                try:
                    progress_file.write_text(
                        "reused valid grid "
                        f"points={len(existing_grid)} "
                        f"edge_drop={existing_quality['min_edge_drop']:.1f} "
                        f"grid_ess={existing_quality['grid_effective_points']:.1f} "
                        f"max_weight={existing_quality['grid_max_weight']:.3f}"
                    )
                except OSError:
                    pass
                return _sample_beta_grid_draws(
                    task,
                    settings,
                    calibrated_start,
                    existing_grid["log_beta_S"].to_numpy(dtype=float),
                    existing_logp,
                    existing_weights,
                )

    center = float(calibrated_start[PARAMETER_INDEX_BY_SAMPLE["beta_S"]])

    def evaluate_grid(center_value: float, half_width_value: float) -> tuple[np.ndarray, np.ndarray]:
        grid = np.linspace(center_value - half_width_value, center_value + half_width_value, grid_points)
        values = _evaluate_beta_grid_log_posteriors(
            grid,
            calibrated_start=calibrated_start,
            runtime_base=runtime_base,
            observed=observed,
            settings=settings,
            country=task.country,
            n_jobs=task.grid_eval_jobs,
            progress_file=progress_file,
            half_width_value=half_width_value,
        )
        return grid, np.asarray(values, dtype=float)

    def evaluate_integration_grid(
        center_value: float,
        half_width_value: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        grid, raw_log_post = evaluate_grid(center_value, half_width_value)
        integration_log_post, smoothing_used = _integration_grid_log_posterior(
            raw_log_post,
            smoothing=grid_smoothing,
            savgol_window=savgol_window,
            min_effective_points=min_effective_points,
            max_single_weight=max_single_weight,
        )
        return grid, raw_log_post, integration_log_post, smoothing_used

    grid_x: np.ndarray | None = None
    raw_log_post_arr: np.ndarray | None = None
    log_post_arr: np.ndarray | None = None
    smoothing_used = "none"

    # Expand the grid if the posterior mass reaches an edge.  A 20 log-unit
    # margin leaves <2e-9 relative edge density, enough for stable quantiles.
    for expansion in range(5):
        grid_x, raw_log_post_arr, log_post_arr, smoothing_used = evaluate_integration_grid(
            center,
            half_width,
        )
        _, quality = _normalised_grid_weights(log_post_arr)
        if quality["finite"] <= 0.0:
            half_width *= 2.0
            continue
        if quality["min_edge_drop"] >= tail_drop_target:
            break
        if expansion < 4:
            growth = np.sqrt(tail_drop_target / max(quality["min_edge_drop"], 1e-6))
            half_width *= float(min(max(growth * 1.2, 1.5), 4.0))

    if grid_x is None or raw_log_post_arr is None or log_post_arr is None:
        raise RuntimeError("beta_grid failed to construct a posterior grid")

    best_score = -np.inf
    best_state: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        int,
        str,
    ] | None = None

    def remember_best_grid() -> None:
        nonlocal best_score, best_state
        if grid_x is None or raw_log_post_arr is None or log_post_arr is None:
            return
        _, current_quality = _normalised_grid_weights(log_post_arr)
        current_score = _grid_quality_score(
            current_quality,
            tail_drop_target=tail_drop_target,
            min_effective_points=min_effective_points,
            max_single_weight=max_single_weight,
        )
        if current_score > best_score:
            best_score = current_score
            best_state = (
                grid_x.copy(),
                raw_log_post_arr.copy(),
                log_post_arr.copy(),
                float(half_width),
                int(grid_points),
                str(smoothing_used),
            )

    remember_best_grid()

    # Refine overly coarse grids by targeting both tail coverage and numerical
    # quadrature resolution.  Sharp country likelihoods can have safe tails but
    # still collapse most posterior mass onto one point if the grid is too wide.
    recent_half_widths: list[float] = [float(half_width)]
    for _ in range(max_refinements):
        weights, quality = _normalised_grid_weights(log_post_arr)
        remember_best_grid()
        if quality["finite"] <= 0.0:
            half_width *= 2.0
            grid_x, raw_log_post_arr, log_post_arr, smoothing_used = evaluate_integration_grid(
                center,
                half_width,
            )
            continue

        finite = np.isfinite(log_post_arr)
        mode_x = float(grid_x[int(np.nanargmax(np.where(finite, log_post_arr, -np.inf)))])
        center = mode_x
        edge_ok = quality["min_edge_drop"] >= tail_drop_target
        resolution_ok = (
            quality["grid_effective_points"] >= min_effective_points
            and quality["grid_max_weight"] <= max_single_weight
        )
        if edge_ok and resolution_ok:
            break

        local_half_width = None
        if (
            not resolution_ok
            and quality["min_edge_drop"] >= tail_drop_target * 0.75
        ):
            local_half_width = _estimate_local_mode_half_width(
                grid_x,
                log_post_arr,
                tail_drop_target,
            )

        previous_half_width = float(half_width)
        if (
            local_half_width is not None
            and local_half_width > 0.0
            and local_half_width < half_width * 0.85
        ):
            half_width = local_half_width
        elif not edge_ok:
            growth = np.sqrt(tail_drop_target / max(quality["min_edge_drop"], 1e-6))
            half_width *= float(min(max(growth * 1.2, 1.5), 4.0))
            if grid_points < grid_max_points and previous_half_width > 0.0:
                previous_step = (2.0 * previous_half_width) / max(grid_points - 1, 1)
                target_points = int(np.ceil((2.0 * half_width) / previous_step)) + 1
                if target_points % 2 == 0:
                    target_points += 1
                grid_points = min(grid_max_points, max(grid_points, target_points))
                if grid_points % 2 == 0:
                    grid_points -= 1
        elif quality["min_edge_drop"] > tail_drop_target * 1.15:
            shrink = np.sqrt((tail_drop_target * 1.15) / quality["min_edge_drop"])
            half_width *= float(min(max(shrink, 0.20), 0.85))
        elif grid_points < grid_max_points:
            grid_points = min(grid_max_points, grid_points * 2 - 1)
            if grid_points % 2 == 0:
                grid_points += 1
        else:
            break

        repeated_width = any(
            np.isclose(half_width, previous, rtol=0.15, atol=0.0)
            for previous in recent_half_widths[-4:]
        )
        if repeated_width:
            if grid_points < grid_max_points:
                grid_points = min(grid_max_points, grid_points * 2 - 1)
                if grid_points % 2 == 0:
                    grid_points += 1
            else:
                break
        recent_half_widths.append(float(half_width))

        grid_x, raw_log_post_arr, log_post_arr, smoothing_used = evaluate_integration_grid(
            center,
            half_width,
        )

    remember_best_grid()
    _, final_quality = _normalised_grid_weights(log_post_arr)
    final_score = _grid_quality_score(
        final_quality,
        tail_drop_target=tail_drop_target,
        min_effective_points=min_effective_points,
        max_single_weight=max_single_weight,
    )
    if best_state is not None and best_score > final_score + 1e-9:
        (
            grid_x,
            raw_log_post_arr,
            log_post_arr,
            half_width,
            grid_points,
            smoothing_used,
        ) = best_state

    finite = np.isfinite(log_post_arr)
    if not finite.any():
        raise RuntimeError(f"beta_grid found no finite posterior support for {task.country}")
    weights, quality = _normalised_grid_weights(log_post_arr)

    grid_rows = pd.DataFrame(
        {
            "country": task.country,
            "log_beta_S": grid_x,
            "beta_S": np.exp(grid_x),
            "log_posterior": log_post_arr,
            "log_posterior_raw": raw_log_post_arr,
            "posterior_weight": weights,
            "grid_half_width": half_width,
            "grid_points": grid_points,
            "grid_tail_drop_target": tail_drop_target,
            "grid_min_effective_points_target": min_effective_points,
            "grid_max_single_weight_target": max_single_weight,
            "grid_left_edge_drop": quality["left_edge_drop"],
            "grid_right_edge_drop": quality["right_edge_drop"],
            "grid_min_edge_drop": quality["min_edge_drop"],
            "grid_effective_points": quality["grid_effective_points"],
            "grid_max_weight": quality["grid_max_weight"],
            "grid_smoothing": smoothing_used,
        }
    )
    write_dataframe(grid_rows, grid_dir / f"{task.country}_grid.csv")

    try:
        ess = 1.0 / float(np.sum(weights ** 2))
        progress_file.write_text(
            f"grid {grid_points}/{grid_points} done half_width={half_width:.4g} "
            f"edge_drop={quality['min_edge_drop']:.1f} grid_ess={ess:.1f} "
            f"max_weight={quality['grid_max_weight']:.3f} smoothing={smoothing_used}"
        )
    except OSError:
        pass

    return _sample_beta_grid_draws(
        task,
        settings,
        calibrated_start,
        grid_x,
        log_post_arr,
        weights,
    )


# ---------------------------------------------------------------------------
# Adaptive Metropolis MCMC chain runner
# ---------------------------------------------------------------------------

def _proposal_covariance(
    adapter: AdaptiveState,
    initial_cov: np.ndarray,
    global_scale: float,
) -> np.ndarray:
    if adapter.n >= AM_COMPONENTWISE_STEPS:
        adapted = adapter.proposal_cov()
        cov = global_scale * (0.85 * adapted + 0.15 * initial_cov)
    else:
        cov = global_scale * initial_cov
    cov = 0.5 * (cov + cov.T)
    cov += AM_EPSILON * np.eye(N_PARAMS)
    return cov


def _draw_mixed_proposal(
    rng: np.random.Generator,
    current: np.ndarray,
    proposal_cov: np.ndarray,
    diagonal_scales: np.ndarray,
    global_scale: float,
    active_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a symmetric proposal from a local/block/full mixture."""
    active = np.asarray(active_indices if active_indices is not None else np.arange(N_PARAMS), dtype=int)
    if len(active) == 0:
        return current.copy()
    active_set = set(int(i) for i in active)
    proposal = current.copy()
    move = rng.random()

    if move < LOCAL_PROPOSAL_PROBABILITY:
        component = int(rng.choice(active))
        std = float(np.sqrt(max(proposal_cov[component, component], 1e-12)))
        proposal[component] += rng.normal(0.0, std)
        return proposal

    if move < LOCAL_PROPOSAL_PROBABILITY + BLOCK_PROPOSAL_PROBABILITY:
        candidate_blocks = [
            np.array([idx for idx in block if idx in active_set], dtype=int)
            for block in PROPOSAL_BLOCKS
        ]
        candidate_blocks = [block for block in candidate_blocks if len(block) > 0]
        block = candidate_blocks[int(rng.integers(0, len(candidate_blocks)))]
        try:
            block_cov = proposal_cov[np.ix_(block, block)]
            L = np.linalg.cholesky(block_cov)
            proposal[block] += L @ rng.standard_normal(len(block))
        except np.linalg.LinAlgError:
            proposal[block] += rng.normal(
                0.0,
                diagonal_scales[block] * max(np.sqrt(global_scale), 1e-6),
            )
        return proposal

    try:
        active_cov = proposal_cov[np.ix_(active, active)]
        L = np.linalg.cholesky(active_cov)
        proposal[active] += L @ rng.standard_normal(len(active))
        return proposal
    except np.linalg.LinAlgError:
        proposal[active] += rng.normal(
            0.0,
            diagonal_scales[active] * max(np.sqrt(global_scale), 1e-6),
        )
        return proposal


def _slice_update_component(
    rng: np.random.Generator,
    current: np.ndarray,
    current_logp: float,
    current_sample: dict[str, float] | None,
    component: int,
    width: float,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    country: str,
    settings: dict[str, Any],
    likelihood_cache: dict[str, float],
    *,
    max_steps_out: int = 6,
    max_shrink_steps: int = 30,
) -> tuple[np.ndarray, float, dict[str, float] | None, bool]:
    """One univariate slice update on the transformed parameter scale."""
    width = max(float(width), 1e-6)
    log_y = float(current_logp - rng.exponential(1.0))
    u = float(rng.random())
    left = float(current[component] - u * width)
    right = float(left + width)

    trial = current.copy()
    for _ in range(max_steps_out):
        trial[component] = left
        left_logp, _ = _log_posterior(
            trial, runtime_base, observed, country, settings, _cache=likelihood_cache
        )
        if not isfinite(left_logp) or left_logp <= log_y:
            break
        left -= width

    trial = current.copy()
    for _ in range(max_steps_out):
        trial[component] = right
        right_logp, _ = _log_posterior(
            trial, runtime_base, observed, country, settings, _cache=likelihood_cache
        )
        if not isfinite(right_logp) or right_logp <= log_y:
            break
        right += width

    for _ in range(max_shrink_steps):
        trial = current.copy()
        proposal_value = float(rng.uniform(left, right))
        trial[component] = proposal_value
        proposed_logp, proposed_sample = _log_posterior(
            trial, runtime_base, observed, country, settings, _cache=likelihood_cache
        )
        if isfinite(proposed_logp) and proposed_logp >= log_y:
            return trial, float(proposed_logp), proposed_sample, True
        if proposal_value < current[component]:
            left = proposal_value
        else:
            right = proposal_value

    return current, current_logp, current_sample, False


def _run_chain(task: ChainTask) -> pd.DataFrame:
    """Run a single MCMC chain with Adaptive Metropolis + Robbins-Monro scaling.

    Three-phase strategy for robust convergence in the transformed parameter space:

    Phase 1 — Componentwise exploration (steps 0 to AM_COMPONENTWISE_STEPS):
        Update one parameter at a time with moderate steps. This guarantees
        ~40-50% acceptance per component and builds a high-quality sample for
        the AM covariance estimate (~45 full parameter cycles).

    Phase 2 — Adaptive warmup (AM_COMPONENTWISE_STEPS to task.warmup):
        Full multivariate AM proposal using the empirical covariance from
        Phase 1. A Robbins-Monro step-size adaptation continuously tunes the
        global scaling factor to target 23.4% acceptance. The covariance
        continues to be updated throughout warmup.

    Phase 3 — Sampling (post-warmup):
        Covariance frozen at the warmup-adapted value. Robbins-Monro scaling
        frozen. Thinned draws are recorded.
    """
    rng = np.random.default_rng(task.seed)
    configs = load_configs()
    settings = deepcopy(configs["baseline"]["bayesian_uncertainty"])
    settings["likelihood_observation_frequency"] = task.likelihood_observation_frequency
    settings["dispersion"] = float(task.dispersion)
    _apply_prior_sd_overrides(
        settings,
        prior_sd_scale=task.prior_sd_scale,
        beta_prior_log_sd=task.beta_prior_log_sd,
        reporting_prior_log_sd=task.reporting_prior_log_sd,
        ve_prior_sd=task.ve_prior_sd,
        rel_asym_prior_sd=task.rel_asym_prior_sd,
        fitness_prior_sd=task.fitness_prior_sd,
    )
    sampler = str(task.sampler).lower()
    if sampler not in {"adaptive_mh", "componentwise_mh", "slice", "beta_grid"}:
        raise ValueError(f"Unsupported Bayesian sampler: {task.sampler}")
    base = make_config(
        vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
        resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
        country_profile=task.country,
    )
    observed = _country_observed(
        task.country,
        interval=str(task.likelihood_observation_frequency),
    )
    runtime_base = calibration_runtime_config(base, observed)
    solver_mode = str(task.solver_mode or "mcmc_fast").lower()
    if solver_mode == "mcmc_fast":
        from src_python.model.rk4_solver import apply_mcmc_solver_overrides
        runtime_base = apply_mcmc_solver_overrides(runtime_base)
    elif solver_mode == "production":
        production_sim = configs["baseline"].get("simulation", {})
        runtime_sim = runtime_base.setdefault("simulation", {})
        for key in ("burn_in_years", "output_time_step", "solver_method", "rtol", "atol"):
            if key in production_sim:
                runtime_sim[key] = production_sim[key]
    elif solver_mode not in {"calibration", "production"}:
        raise ValueError(f"Unsupported Bayesian solver mode: {task.solver_mode}")

    # Inject country-specific fixed values into priors for _sample_from_vector
    # Resistance prevalence is fixed at the country-calibrated value
    country_resistance = float(runtime_base["resistance"]["target_prevalence_at_analysis_start"])
    settings["priors"] = dict(settings["priors"])
    settings["priors"]["resistance_prevalence_fixed"] = country_resistance
    settings["priors"]["reporting_trend_fixed"] = 1.0  # no secular trend
    settings["priors"]["parameterization"] = str(task.parameterization or "standard")
    settings["priors"]["base_log_beta_S"] = float(np.log(float(runtime_base["transmission"]["beta_S"])))
    settings["priors"]["VE_dur_fixed"] = float(
        runtime_base.get("vaccine", {}).get(
            "VE_dur",
            settings["priors"].get("VE_dur", {}).get("mean", 0.10),
        )
    )

    # Determine which parameter indices are fixed (not sampled).  Fixed
    # dimensions are reset to the calibrated country start even when the other
    # dimensions are warm-started from a previous posterior pilot; otherwise a
    # "fixed reporting" pilot could accidentally inherit a chain-specific
    # reporting value from the warm-start file.
    fixed_indices = _fixed_parameter_indices(
        task.fixed_parameters,
        fix_durations=task.fix_durations,
    )

    calibrated_start = _initial_vector(
        runtime_base,
        enable_trend=task.enable_time_varying_reporting,
        priors=settings["priors"],
    )

    if task.initial_samples_path and str(task.initial_strategy).lower() != "calibrated":
        current = _initial_vector_from_samples(
            task.initial_samples_path,
            task.country,
            task.chain,
            settings["priors"],
            task.initial_strategy,
        )
    else:
        current = calibrated_start.copy()

    if fixed_indices:
        for idx in fixed_indices:
            current[idx] = calibrated_start[idx]

    # Indices: 0=log_beta_S, 1=reporting coordinate, 2=logit_VE_sus,
    #          3=logit_VE_inf, 4=logit_rel_inf_asym,
    #          5=log_inf_dur_sym, 6=log_inf_dur_asym, 7=logit_fitness_R
    sampled_mask = np.array([i not in fixed_indices for i in range(N_PARAMS)], dtype=bool)
    active_indices = np.array([i for i in range(N_PARAMS) if i not in fixed_indices], dtype=int)
    if len(active_indices) == 0:
        raise ValueError("At least one Bayesian parameter must remain unfixed.")

    if sampler == "beta_grid":
        return _run_beta_grid(
            task,
            runtime_base,
            observed,
            settings,
            calibrated_start,
            active_indices,
        )

    # Small jitter to disperse chains (only on sampled dimensions)
    jitter_scale = 0.5 * INITIAL_PROPOSAL_SCALES * float(task.proposal_scale)
    jitter = rng.normal(0.0, jitter_scale)
    jitter[~sampled_mask] = 0.0  # no jitter on fixed dimensions
    current = current + jitter

    # Initialize likelihood cache to avoid redundant ODE solves
    likelihood_cache: dict[str, float] = {}

    current_logp, current_sample = _log_posterior(
        current, runtime_base, observed, task.country, settings, _cache=likelihood_cache
    )
    if current_sample is None:
        current_sample = _sample_from_vector(current, settings["priors"])

    # If initial position has -inf posterior, try random starts
    if not isfinite(current_logp):
        for _attempt in range(50):
            trial = _initial_vector(runtime_base, enable_trend=task.enable_time_varying_reporting)
            trial_jitter = rng.normal(0.0, jitter_scale * (1.0 + _attempt * 0.1))
            trial_jitter[~sampled_mask] = 0.0
            trial = trial + trial_jitter
            trial_logp, trial_sample = _log_posterior(
                trial, runtime_base, observed, task.country, settings, _cache=likelihood_cache
            )
            if isfinite(trial_logp):
                current = trial
                current_logp = trial_logp
                current_sample = trial_sample or _sample_from_vector(trial, settings["priors"])
                break

    # Adaptive Metropolis state
    adapter = AdaptiveState()
    adapter.update(current)

    # Initial diagonal proposal
    diagonal_scales = INITIAL_PROPOSAL_SCALES * float(task.proposal_scale)
    initial_cov = np.diag(diagonal_scales ** 2)

    # Robbins-Monro global scaling factor (log-scale for stability)
    log_scale = np.log(RM_INITIAL_SCALE)

    rows: list[dict[str, Any]] = []
    accepted = 0
    total_post_warmup_steps = int(task.draws * task.thin)
    total_steps = int(task.warmup) + total_post_warmup_steps

    # Progress reporting
    _progress_dir = _mcmc_progress_dir(task.output_stem)
    _progress_dir.mkdir(parents=True, exist_ok=True)
    _progress_file = _progress_dir / f"{task.country}_chain{task.chain:02d}.txt"
    _progress_interval = min(200, max(10, total_steps // 5))
    componentwise_steps = min(AM_COMPONENTWISE_STEPS, int(task.warmup))

    for step in range(total_steps):
        if step % _progress_interval == 0:
            try:
                rate = accepted / max(step, 1)
                scale_str = f" scale={np.exp(log_scale):.2f}" if step >= AM_COMPONENTWISE_STEPS else ""
                _progress_file.write_text(
                    f"{step}/{total_steps} accept={rate:.3f}{scale_str}" if step > 0
                    else f"0/{total_steps} starting"
                )
            except OSError:
                pass

        if sampler == "slice":
            component = int(active_indices[step % len(active_indices)])
            current, current_logp, current_sample, accept = _slice_update_component(
                rng,
                current,
                current_logp,
                current_sample,
                component,
                diagonal_scales[component] * max(np.sqrt(np.exp(log_scale)), 1e-6),
                runtime_base,
                observed,
                task.country,
                settings,
                likelihood_cache,
            )
            if accept:
                accepted += 1
            if step < task.warmup:
                adapter.update(current)

        elif sampler == "componentwise_mh":
            component = int(rng.choice(active_indices))
            proposal = current.copy()
            proposal[component] += rng.normal(
                0.0,
                diagonal_scales[component] * max(np.sqrt(np.exp(log_scale)), 1e-6),
            )
            proposed_logp, proposed_sample = _log_posterior(
                proposal, runtime_base, observed, task.country, settings, _cache=likelihood_cache
            )
            accept = isfinite(proposed_logp) and np.log(rng.random()) < proposed_logp - current_logp
            if accept:
                current = proposal
                current_logp = proposed_logp
                current_sample = proposed_sample or _sample_from_vector(current, settings["priors"])
                accepted += 1
            if step < task.warmup:
                adapt_step = step + 1
                gamma_t = 1.0 / (adapt_step ** RM_GAMMA)
                log_scale += gamma_t * (float(accept) - 0.44)
                log_scale = float(np.clip(log_scale, RM_LOG_SCALE_MIN, RM_LOG_SCALE_MAX))
                adapter.update(current)

        elif step < componentwise_steps:
            component = int(active_indices[step % len(active_indices)])
            proposal = current.copy()
            # Componentwise warmup should explore, but over-large single-axis
            # jumps badly overfit the adapted covariance in sharp likelihoods.
            proposal[component] += rng.normal(0.0, diagonal_scales[component] * 2.0)

            proposed_logp, proposed_sample = _log_posterior(
                proposal, runtime_base, observed, task.country, settings, _cache=likelihood_cache
            )
            accept = isfinite(proposed_logp) and np.log(rng.random()) < proposed_logp - current_logp
            if accept:
                current = proposal
                current_logp = proposed_logp
                current_sample = proposed_sample or _sample_from_vector(current, settings["priors"])
                accepted += 1
            adapter.update(current)

        else:
            # Phase 2/3: Full multivariate AM proposal with Robbins-Monro scaling
            global_scale = np.exp(log_scale)

            proposal_cov = _proposal_covariance(adapter, initial_cov, global_scale)
            proposal = _draw_mixed_proposal(
                rng,
                current,
                proposal_cov,
                diagonal_scales,
                global_scale,
                active_indices=active_indices,
            )

            proposed_logp, proposed_sample = _log_posterior(
                proposal, runtime_base, observed, task.country, settings, _cache=likelihood_cache
            )

            # Metropolis acceptance
            accept = isfinite(proposed_logp) and np.log(rng.random()) < proposed_logp - current_logp
            if accept:
                current = proposal
                current_logp = proposed_logp
                current_sample = proposed_sample or _sample_from_vector(current, settings["priors"])
                accepted += 1

            # Robbins-Monro step-size adaptation during warmup
            if step < task.warmup:
                # Adapt log_scale so acceptance rate → RM_TARGET_ACCEPTANCE
                # Update rule: log_scale += gamma_t * (alpha - target)
                # where gamma_t = c / (step - AM_COMPONENTWISE_STEPS + c) decays
                adapt_step = step - AM_COMPONENTWISE_STEPS + 1
                gamma_t = 1.0 / (adapt_step ** RM_GAMMA)
                log_scale += gamma_t * (float(accept) - RM_TARGET_ACCEPTANCE)
                # Clamp to prevent extreme scaling — the lower bound is critical
                # to prevent chains from getting stuck with near-zero step sizes
                log_scale = float(np.clip(log_scale, RM_LOG_SCALE_MIN, RM_LOG_SCALE_MAX))
                # Continue updating covariance during warmup
                adapter.update(current)
            else:
                # Post-warmup draws use the frozen warmup proposal. Continuing
                # step-size adaptation here makes the retained chain non-stationary.
                pass

        # Record post-warmup draws (with thinning)
        if step >= task.warmup:
            post_warmup_step = step - task.warmup
            if post_warmup_step % task.thin == 0:
                draw_idx = post_warmup_step // task.thin + 1
                row = {
                    "country": task.country,
                    "chain": task.chain,
                    "draw": draw_idx,
                    "step": step + 1,
                    "posterior_log_prob": current_logp,
                    "accepted_fraction": accepted / float(step + 1),
                }
                row.update(current_sample)
                rows.append(row)

    # Final progress
    try:
        _progress_file.write_text(
            f"{total_steps}/{total_steps} done accept={accepted/total_steps:.3f} scale={np.exp(log_scale):.2f}"
        )
    except OSError:
        pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Posterior predictive and output generation
# ---------------------------------------------------------------------------

def _sample_columns() -> tuple[str, ...]:
    return (
        "beta_S",
        "reporting_multiplier",
        "VE_sus",
        "VE_inf",
        "VE_dur",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_symptomatic",
        "infectious_duration_asymptomatic",
        "fitness_R",
        "resistance_prevalence",
        "reporting_trend_end_multiplier",
    )


def _diagnostic_sample_columns() -> tuple[str, ...]:
    """Return columns that are genuinely sampled by the current sampler."""
    return (
        "beta_S",
        "reporting_multiplier",
        "VE_sus",
        "VE_inf",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_symptomatic",
        "infectious_duration_asymptomatic",
        "fitness_R",
    )


def _posterior_predictive_scenarios(
    samples: pd.DataFrame, draws_per_country: int
) -> list[dict[str, Any]]:
    configs = load_configs()
    scenarios = []
    rng = np.random.default_rng(
        int(configs["baseline"]["bayesian_uncertainty"].get("random_seed", 20260510)) + 917
    )
    for country, group in samples.groupby("country", sort=False):
        group = group.reset_index(drop=True)
        if len(group) > draws_per_country:
            selected = group.iloc[
                np.sort(rng.choice(len(group), draws_per_country, replace=False))
            ].copy()
        else:
            selected = group.copy()
        for draw_idx, row in enumerate(selected.itertuples(index=False), start=1):
            sample = {}
            for name in _sample_columns():
                val = getattr(row, name, None)
                if val is not None:
                    sample[name] = float(val)
                else:
                    # Fallback for legacy samples without trend
                    sample[name] = 1.0 if name == "reporting_trend_end_multiplier" else 0.0
            config = make_config(
                vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
                resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
                country_profile=country,
            )
            config = _apply_sample(config, sample)
            metadata = {
                "country": country,
                "posterior_draw": draw_idx,
                "posterior_chain": int(getattr(row, "chain")),
                "posterior_log_prob": float(getattr(row, "posterior_log_prob")),
                **{f"posterior_{key}": value for key, value in sample.items()},
            }
            scenarios.append(
                {
                    "config": config,
                    "analysis": "bayesian_uncertainty",
                    "scenario": f"{country}_draw_{draw_idx:03d}",
                    "vaccine_scenario": configs["baseline"]["baseline_vaccine_scenario"],
                    "resistance_scenario": configs["baseline"]["baseline_resistance_scenario"],
                    "metadata": metadata,
                }
            )
    return scenarios


def _write_interval_summaries(
    summary: pd.DataFrame,
    samples: pd.DataFrame,
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> None:
    outcome_cols = [
        "annualized_infant_cases_per_100k",
        "annualized_infections_per_100k",
        "annualized_reported_cases_per_100k",
        "annualized_infant_infections_per_100k",
        "resistant_fraction",
        "resistant_fraction_end",
        "resistant_infections",
    ]
    rows = []
    for country, group in summary.groupby("country", sort=False):
        for outcome in outcome_cols:
            if outcome not in group.columns:
                continue
            values = pd.to_numeric(group[outcome], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
            rows.append(
                {
                    "country": country,
                    "outcome": outcome,
                    "posterior_median": float(median),
                    "credible_interval_low": float(low),
                    "credible_interval_high": float(high),
                    "posterior_draws": int(len(values)),
                }
            )
    write_dataframe(
        pd.DataFrame(rows),
        project_path(
            "outputs/summaries",
            f"{_artifact_stem(output_stem, 'bayesian_uncertainty_intervals_summary', 'intervals_summary')}.csv",
        ),
    )

    parameter_rows = []
    for country, group in samples.groupby("country", sort=False):
        for parameter in _sample_columns():
            if parameter not in group.columns:
                continue
            values = pd.to_numeric(group[parameter], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
            parameter_rows.append(
                {
                    "country": country,
                    "parameter": parameter,
                    "posterior_median": float(median),
                    "credible_interval_low": float(low),
                    "credible_interval_high": float(high),
                    "posterior_draws": int(len(values)),
                }
            )
    write_dataframe(
        pd.DataFrame(parameter_rows),
        project_path(
            "outputs/summaries",
            f"{_artifact_stem(output_stem, 'bayesian_parameter_summary', 'parameter_summary')}.csv",
        ),
    )


# ---------------------------------------------------------------------------
# Stochastic overlay with k-sensitivity sweep
# ---------------------------------------------------------------------------

def _apply_stochastic_overlay(
    summary: pd.DataFrame,
    settings: dict[str, Any],
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> pd.DataFrame:
    """Layer superspreading + household-clustering stochastic replicates on the
    posterior predictive summary, writing combined-uncertainty CrI artifacts.
    """
    overlay = StochasticOverlayConfig.from_dict(settings.get("stochastic_overlay"))
    if not overlay.enabled:
        return pd.DataFrame()

    overlay_samples = stochastic_overlay_samples(summary, overlay=overlay)
    combined_intervals = summarize_overlay_intervals(summary, overlay_samples, overlay=overlay)
    variance_components = decompose_variance(overlay_samples)

    write_dataframe(
        overlay_samples,
        project_path(
            "outputs/simulations",
            f"{_artifact_stem(output_stem, 'bayesian_stochastic_overlay_samples', 'stochastic_overlay_samples')}.csv",
        ),
    )
    write_dataframe(
        combined_intervals,
        project_path(
            "outputs/summaries",
            f"{_artifact_stem(output_stem, 'bayesian_stochastic_overlay_intervals_summary', 'stochastic_overlay_intervals_summary')}.csv",
        ),
    )
    write_dataframe(
        variance_components,
        project_path(
            "outputs/summaries",
            f"{_artifact_stem(output_stem, 'bayesian_stochastic_overlay_variance_components', 'stochastic_overlay_variance_components')}.csv",
        ),
    )
    return combined_intervals


def _run_k_sensitivity_sweep(
    summary: pd.DataFrame,
    settings: dict[str, Any],
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> pd.DataFrame:
    """Run stochastic overlay at multiple k values to show CrI sensitivity.

    Sweeps superspreading_k over [5, 10, 20, 30, 50] to demonstrate how
    the choice of aggregate dispersion affects the combined credible interval
    width. This addresses the concern that k=10 vs k=50 is an assumption.
    """
    k_values = settings.get("stochastic_overlay", {}).get(
        "k_sensitivity_values", [5.0, 10.0, 20.0, 30.0, 50.0]
    )
    base_overlay_dict = settings.get("stochastic_overlay", {})
    all_rows: list[dict[str, Any]] = []

    for k_val in k_values:
        sweep_dict = dict(base_overlay_dict)
        sweep_dict["superspreading_k"] = float(k_val)
        # Use fewer replicates for the sweep to keep runtime manageable
        sweep_dict["replicates_per_draw"] = min(
            int(base_overlay_dict.get("replicates_per_draw", 200)), 100
        )
        overlay = StochasticOverlayConfig.from_dict(sweep_dict)
        overlay_samples = stochastic_overlay_samples(summary, overlay=overlay)
        intervals = summarize_overlay_intervals(summary, overlay_samples, overlay=overlay)
        intervals["superspreading_k_sweep"] = float(k_val)
        all_rows.append(intervals)

    if all_rows:
        k_sensitivity = pd.concat(all_rows, ignore_index=True)
        write_dataframe(
            k_sensitivity,
            project_path(
                "outputs/summaries",
                f"{_artifact_stem(output_stem, 'bayesian_k_sensitivity_sweep', 'k_sensitivity_sweep')}.csv",
            ),
        )
        return k_sensitivity
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def _collect_beta_grid_quality(
    samples: pd.DataFrame,
    output_stem: str,
) -> pd.DataFrame:
    """Collect deterministic grid validity metrics for beta_grid runs."""
    grid_dir = project_path("outputs", "metadata", f"beta_grid_{output_stem}")
    rows: list[dict[str, Any]] = []
    for country in sorted(samples["country"].astype(str).unique()):
        path = grid_dir / f"{country}_grid.csv"
        if not path.exists():
            rows.append(
                {
                    "country": country,
                    "grid_valid": False,
                    "issue": "missing_grid_file",
                    "grid_points": 0,
                    "grid_half_width": np.nan,
                    "grid_min_edge_drop": 0.0,
                    "grid_effective_points": 0.0,
                    "grid_max_weight": 1.0,
                    "grid_smoothing": "missing",
                    "grid_tail_drop_target": 20.0,
                    "grid_min_effective_points_target": 10.0,
                    "grid_max_single_weight_target": 0.20,
                }
            )
            continue

        grid = pd.read_csv(path)
        if grid.empty:
            rows.append(
                {
                    "country": country,
                    "grid_valid": False,
                    "issue": "empty_grid_file",
                    "grid_points": 0,
                    "grid_half_width": np.nan,
                    "grid_min_edge_drop": 0.0,
                    "grid_effective_points": 0.0,
                    "grid_max_weight": 1.0,
                    "grid_smoothing": "missing",
                    "grid_tail_drop_target": 20.0,
                    "grid_min_effective_points_target": 10.0,
                    "grid_max_single_weight_target": 0.20,
                }
            )
            continue

        first = grid.iloc[0]
        tail_drop_target = float(first.get("grid_tail_drop_target", 20.0))
        min_effective_target = float(first.get("grid_min_effective_points_target", 10.0))
        max_weight_target = float(first.get("grid_max_single_weight_target", 0.20))
        if {
            "grid_min_edge_drop",
            "grid_effective_points",
            "grid_max_weight",
        }.issubset(grid.columns):
            min_edge_drop = float(first["grid_min_edge_drop"])
            grid_effective_points = float(first["grid_effective_points"])
            grid_max_weight = float(first["grid_max_weight"])
        else:
            weights, quality = _normalised_grid_weights(
                grid["log_posterior"].to_numpy(dtype=float)
            )
            min_edge_drop = float(quality["min_edge_drop"])
            grid_effective_points = float(quality["grid_effective_points"])
            grid_max_weight = float(quality["grid_max_weight"])
            if "posterior_weight" in grid.columns and np.isfinite(weights).all():
                grid["posterior_weight"] = weights

        issues = []
        if min_edge_drop < tail_drop_target:
            issues.append("edge_tail")
        if grid_effective_points < min_effective_target:
            issues.append("grid_ess")
        if grid_max_weight > max_weight_target:
            issues.append("single_weight")
        rows.append(
            {
                "country": country,
                "grid_valid": len(issues) == 0,
                "issue": ",".join(issues),
                "grid_points": int(len(grid)),
                "grid_half_width": float(first.get("grid_half_width", np.nan)),
                "grid_min_edge_drop": min_edge_drop,
                "grid_effective_points": grid_effective_points,
                "grid_max_weight": grid_max_weight,
                "grid_smoothing": str(first.get("grid_smoothing", "none")),
                "grid_tail_drop_target": tail_drop_target,
                "grid_min_effective_points_target": min_effective_target,
                "grid_max_single_weight_target": max_weight_target,
            }
        )
    return pd.DataFrame(rows)


def _write_convergence_diagnostics(
    samples: pd.DataFrame,
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> dict[str, Any]:
    """Compute and write posterior validity diagnostics.

    Beta-grid runs are gated by deterministic tail/resolution checks. MCMC
    runs are gated by rank-normalized R-hat and ESS. The returned summary is
    included in run metadata.
    """
    param_cols = [c for c in _diagnostic_sample_columns() if c in samples.columns]
    diagnostics = compute_diagnostics(
        samples,
        parameter_columns=tuple(param_cols),
        chain_column="chain",
        country_column="country",
    )
    write_dataframe(
        diagnostics,
        project_path(
            "outputs/summaries",
            f"{_artifact_stem(output_stem, 'bayesian_convergence_diagnostics', 'convergence_diagnostics')}.csv",
        ),
    )

    convergence_summary = summarize_convergence(diagnostics)
    is_beta_grid = (
        "sampling_method" in samples.columns
        and samples["sampling_method"].astype(str).eq("beta_grid").all()
    )
    beta_grid_quality = pd.DataFrame()
    if is_beta_grid:
        beta_grid_quality = _collect_beta_grid_quality(samples, output_stem=output_stem)
        write_dataframe(
            beta_grid_quality,
            project_path(
                "outputs/summaries",
                f"{_artifact_stem(output_stem, 'bayesian_beta_grid_quality', 'beta_grid_quality')}.csv",
            ),
        )
        bad_grid = (
            beta_grid_quality.loc[~beta_grid_quality["grid_valid"], "country"]
            .astype(str)
            .tolist()
        )
        grid_valid = bool(not beta_grid_quality.empty and len(bad_grid) == 0)
        n_total = int(convergence_summary.get("n_parameters_total", 0))
        n_converged = n_total if grid_valid else max(n_total - len(bad_grid), 0)
        convergence_summary = {
            **convergence_summary,
            "all_converged": grid_valid,
            "n_parameters_converged": n_converged,
            "fraction_converged": (n_converged / n_total) if n_total else 0.0,
            "countries_with_issues": bad_grid,
            "beta_grid_quality": beta_grid_quality,
            "method_note": (
                "Deterministic beta-grid quadrature; MCMC R-hat/ESS criteria are "
                "reported for the quantile sample only. Validity is determined "
                "by the beta-grid tail and quadrature-resolution checks below."
            ),
        }

    # Write human-readable summary
    summary_path = project_path(
        "outputs/summaries",
        f"{_artifact_stem(output_stem, 'bayesian_convergence_summary', 'convergence_summary')}.txt",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        if is_beta_grid:
            f.write("Deterministic Beta-grid Quadrature Summary\n")
        else:
            f.write("MCMC Convergence Summary\n")
        f.write("=" * 60 + "\n\n")
        if is_beta_grid:
            f.write(convergence_summary["method_note"] + "\n\n")
        f.write(f"All parameters converged: {convergence_summary['all_converged']}\n")
        f.write(
            f"Parameters converged: {convergence_summary['n_parameters_converged']}"
            f" / {convergence_summary['n_parameters_total']}"
            f" ({convergence_summary.get('fraction_converged', 0):.1%})\n"
        )
        f.write(f"Worst R-hat (rank-normalized): {convergence_summary['worst_rhat']:.4f}\n")
        f.write(f"Minimum bulk ESS: {convergence_summary['min_bulk_ess']:.0f}\n")
        f.write(f"Minimum tail ESS: {convergence_summary['min_tail_ess']:.0f}\n")
        if convergence_summary["countries_with_issues"]:
            f.write(f"\nCountries with convergence issues: "
                    f"{', '.join(convergence_summary['countries_with_issues'])}\n")
        if is_beta_grid:
            f.write("\n\nValidity criteria:\n")
            f.write("  - Posterior evaluated by deterministic log-beta grid/quadrature\n")
            f.write("  - Grid edges must be at least 20 log-posterior units below the mode\n")
            f.write("  - Effective grid points must be at least 10\n")
            f.write("  - No single grid point may carry more than 20% posterior mass\n")
            f.write("  - Reported R-hat/ESS are descriptive for the generated quantile sample only\n")
            if not beta_grid_quality.empty:
                f.write("\nBeta-grid numerical quality:\n")
                f.write(
                    "  - Minimum edge drop: "
                    f"{beta_grid_quality['grid_min_edge_drop'].min():.1f}\n"
                )
                f.write(
                    "  - Minimum effective grid points: "
                    f"{beta_grid_quality['grid_effective_points'].min():.1f}\n"
                )
                f.write(
                    "  - Maximum single grid weight: "
                    f"{beta_grid_quality['grid_max_weight'].max():.3f}\n"
                )
                if "grid_smoothing" in beta_grid_quality.columns:
                    smoothing_counts = (
                        beta_grid_quality["grid_smoothing"].fillna("none").astype(str).value_counts()
                    )
                    smoothing_text = ", ".join(
                        f"{method}={count}" for method, count in smoothing_counts.items()
                    )
                    f.write(f"  - Grid smoothing methods: {smoothing_text}\n")
        else:
            f.write("\n\nConvergence criteria:\n")
            f.write("  - R-hat (rank-normalized) < 1.05\n")
            f.write("  - Bulk ESS > 100\n")
            f.write("\nReference: Vehtari et al. (2021) Bayesian Analysis 16(2):667-718\n")

        # Per-country worst diagnostics
        f.write("\n\nPer-country worst diagnostics:\n")
        f.write("-" * 60 + "\n")
        if diagnostics.empty:
            f.write("  No varying sampled parameters found.\n")
        else:
            for country in sorted(diagnostics["country"].unique()):
                country_diag = diagnostics.loc[diagnostics["country"].eq(country)]
                if country_diag.empty:
                    continue
                if not country_diag["rhat_rank"].notna().any() or not country_diag["bulk_ess"].notna().any():
                    f.write(f"  {country}: diagnostics unavailable for the retained draw count\n")
                    continue
                worst_rhat_row = country_diag.loc[country_diag["rhat_rank"].idxmax()]
                min_ess_row = country_diag.loc[country_diag["bulk_ess"].idxmin()]
                f.write(
                    f"  {country}: worst R-hat={worst_rhat_row['rhat_rank']:.3f} "
                    f"({worst_rhat_row['parameter']}), "
                    f"min ESS={min_ess_row['bulk_ess']:.0f} "
                    f"({min_ess_row['parameter']})\n"
                )

    return convergence_summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _ensure_calibrated_country_starts(countries: list[str]) -> None:
    """Ensure Bayesian chains start from accepted country calibrations."""
    missing: list[str] = []
    configs = load_configs()
    vaccine = configs["baseline"]["baseline_vaccine_scenario"]
    resistance = configs["baseline"]["baseline_resistance_scenario"]
    for country in countries:
        config = make_config(
            vaccine_scenario=vaccine,
            resistance_scenario=resistance,
            country_profile=country,
        )
        if not bool(config.get("metadata", {}).get("calibration_loaded", False)):
            missing.append(country)

    if not missing:
        return

    from src_python.calibration.calibrate_baseline import calibrate_country

    for country in missing:
        calibrate_country(country)

    still_missing: list[str] = []
    for country in missing:
        config = make_config(
            vaccine_scenario=vaccine,
            resistance_scenario=resistance,
            country_profile=country,
        )
        if not bool(config.get("metadata", {}).get("calibration_loaded", False)):
            still_missing.append(country)

    if still_missing:
        raise RuntimeError(
            "Bayesian uncertainty analysis requires accepted country calibrations; missing after "
            f"auto-calibration: {', '.join(still_missing)}"
        )


def _clear_previous_mcmc_progress(
    countries: list[str],
    n_chains: int,
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> None:
    """Remove stale per-chain or beta-grid progress files before a new run starts."""
    progress_dir = _mcmc_progress_dir(output_stem)
    if not progress_dir.exists():
        return
    for country in countries:
        for chain in range(1, n_chains + 1):
            try:
                (progress_dir / f"{country}_chain{chain:02d}.txt").unlink()
            except FileNotFoundError:
                pass
        try:
            (progress_dir / f"{country}_beta_grid.txt").unlink()
        except FileNotFoundError:
            pass


def _apply_prior_sd_overrides(
    settings: dict[str, Any],
    *,
    prior_sd_scale: float | None = None,
    beta_prior_log_sd: float | None = None,
    reporting_prior_log_sd: float | None = None,
    ve_prior_sd: float | None = None,
    rel_asym_prior_sd: float | None = None,
    fitness_prior_sd: float | None = None,
) -> None:
    """Apply pilot prior-width overrides in-place."""
    priors = settings["priors"]

    if prior_sd_scale is not None:
        scale = float(prior_sd_scale)
        if scale <= 0.0:
            raise ValueError("--prior-sd-scale must be positive")
        for key in ("log_beta_S_sd", "log_reporting_multiplier_sd"):
            if key in priors:
                priors[key] = float(priors[key]) * scale
        for key in ("VE_sus", "VE_inf", "relative_infectiousness_asymptomatic", "fitness_R"):
            if key in priors and "sd" in priors[key]:
                priors[key]["sd"] = float(priors[key]["sd"]) * scale
        for key in ("infectious_duration_symptomatic", "infectious_duration_asymptomatic"):
            if key in priors and "log_sd" in priors[key]:
                priors[key]["log_sd"] = float(priors[key]["log_sd"]) * scale

    if beta_prior_log_sd is not None:
        priors["log_beta_S_sd"] = float(beta_prior_log_sd)
    if reporting_prior_log_sd is not None:
        priors["log_reporting_multiplier_sd"] = float(reporting_prior_log_sd)
    if ve_prior_sd is not None:
        priors["VE_sus"]["sd"] = float(ve_prior_sd)
        priors["VE_inf"]["sd"] = float(ve_prior_sd)
    if rel_asym_prior_sd is not None:
        priors["relative_infectiousness_asymptomatic"]["sd"] = float(rel_asym_prior_sd)
    if fitness_prior_sd is not None:
        priors["fitness_R"]["sd"] = float(fitness_prior_sd)


def main(
    n_jobs: int | None = None,
    draws: int | None = None,
    warmup: int | None = None,
    n_chains: int | None = None,
    proposal_scale: float | None = None,
    thin: int | None = None,
    dispersion: float | None = None,
    likelihood_observation_frequency: str | None = None,
    skip_k_sensitivity: bool = False,
    skip_posterior_predictive: bool = False,
    countries_filter: list[str] | None = None,
    fix_durations: bool = False,
    fixed_parameters: list[str] | tuple[str, ...] | None = None,
    sampler: str = "beta_grid",
    output_stem: str = DEFAULT_OUTPUT_STEM,
    initial_samples_path: str | None = None,
    initial_strategy: str = "calibrated",
    solver_mode: str = "calibration",
    prior_sd_scale: float | None = None,
    beta_prior_log_sd: float | None = None,
    reporting_prior_log_sd: float | None = None,
    ve_prior_sd: float | None = None,
    rel_asym_prior_sd: float | None = None,
    fitness_prior_sd: float | None = None,
    parameterization: str = "standard",
    grid_points: int = 161,
    grid_log_beta_half_width: float = 0.08,
    grid_max_points: int = 641,
    grid_max_refinements: int = 10,
    grid_tail_drop: float = 20.0,
    grid_min_effective_points: float = 10.0,
    grid_max_single_weight: float = 0.20,
    grid_smoothing: str = "auto",
    grid_savgol_window: int = 21,
    reuse_valid_beta_grid: bool = False,
    grid_eval_jobs: int | None = None,
):
    """Run the full Bayesian uncertainty analysis pipeline.

    Steps:
    1. Run the configured posterior sampler in parallel
    2. Compute posterior validity diagnostics
    3. Generate posterior predictive simulations in parallel
    4. Compute credible intervals
    5. Apply stochastic overlay (superspreading + household clustering)
    6. Run k-sensitivity sweep
    7. Write all artifacts

    Parameters
    ----------
    countries_filter : optional list of country config_keys to run (for pilot testing)
    fix_durations : if True, fix infectious_duration_symptomatic and
        infectious_duration_asymptomatic at their calibrated values (reduces
        sampled dimensions from 8 to 6)
    """
    configs = load_configs()
    settings = deepcopy(configs["baseline"]["bayesian_uncertainty"])
    sampler = str(sampler).lower()
    if sampler == "beta_grid":
        if fixed_parameters is None:
            fixed_parameters = DEFAULT_BETA_GRID_FIXED_PARAMETERS
        fix_durations = True
    parameterization = str(parameterization or "standard").lower()
    if parameterization not in {"standard", "beta_reporting_product"}:
        raise ValueError(
            "Unsupported Bayesian parameterization: "
            f"{parameterization}. Valid values: standard, beta_reporting_product"
        )
    if dispersion is not None:
        settings["dispersion"] = float(dispersion)
    if likelihood_observation_frequency is not None:
        settings["likelihood_observation_frequency"] = str(likelihood_observation_frequency)
    _apply_prior_sd_overrides(
        settings,
        prior_sd_scale=prior_sd_scale,
        beta_prior_log_sd=beta_prior_log_sd,
        reporting_prior_log_sd=reporting_prior_log_sd,
        ve_prior_sd=ve_prior_sd,
        rel_asym_prior_sd=rel_asym_prior_sd,
        fitness_prior_sd=fitness_prior_sd,
    )

    n_chains = int(n_chains or settings.get("n_chains", 4))
    chain_draws = int(draws or settings.get("draws", 2500))
    chain_warmup = int(warmup or settings.get("warmup", 1500))
    base_seed = int(settings.get("random_seed", 20260510))
    proposal_scale = float(proposal_scale if proposal_scale is not None else settings.get("proposal_scale", 1.0))
    enable_trend = bool(settings.get("enable_time_varying_reporting", True))
    thin = int(thin or settings.get("thin", 1))
    likelihood_interval = str(settings.get("likelihood_observation_frequency", "monthly"))
    chain_dispersion = float(settings.get("dispersion", 50.0))
    countries = list(configs["countries"])

    # Filter countries if specified (for pilot testing)
    if countries_filter:
        available = set(countries)
        countries = [c for c in countries_filter if c in available]
        if not countries:
            raise ValueError(f"No valid countries in filter: {countries_filter}. Available: {sorted(available)}")

    if bool(settings.get("require_calibrated_start", True)):
        _ensure_calibrated_country_starts(countries)

    # Build chain tasks: one per (country, chain) combination
    # All tasks run in parallel across available CPUs
    if sampler == "beta_grid":
        tasks = [
            ChainTask(
                country=country,
                chain=0,
                seed=base_seed + country_idx * 1000 + 137,
                warmup=chain_warmup,
                draws=chain_draws,
                proposal_scale=proposal_scale,
                enable_time_varying_reporting=enable_trend,
                thin=thin,
                fix_durations=fix_durations,
                fixed_parameters=tuple(fixed_parameters or ()),
                sampler=sampler,
                likelihood_observation_frequency=likelihood_interval,
                dispersion=chain_dispersion,
                output_stem=output_stem,
                initial_samples_path=initial_samples_path,
                initial_strategy=initial_strategy,
                solver_mode=solver_mode,
                prior_sd_scale=prior_sd_scale,
                beta_prior_log_sd=beta_prior_log_sd,
                reporting_prior_log_sd=reporting_prior_log_sd,
                ve_prior_sd=ve_prior_sd,
                rel_asym_prior_sd=rel_asym_prior_sd,
                fitness_prior_sd=fitness_prior_sd,
                parameterization=parameterization,
                grid_points=grid_points,
                grid_log_beta_half_width=grid_log_beta_half_width,
                grid_max_points=grid_max_points,
                grid_max_refinements=grid_max_refinements,
                grid_tail_drop=grid_tail_drop,
                grid_min_effective_points=grid_min_effective_points,
                grid_max_single_weight=grid_max_single_weight,
                grid_smoothing=grid_smoothing,
                grid_savgol_window=grid_savgol_window,
                grid_n_chains=n_chains,
                reuse_valid_beta_grid=reuse_valid_beta_grid,
                grid_eval_jobs=1,
            )
            for country_idx, country in enumerate(countries)
        ]
    else:
        tasks = [
            ChainTask(
                country=country,
                chain=chain,
                seed=base_seed + country_idx * 1000 + chain * 137,
                warmup=chain_warmup,
                draws=chain_draws,
                proposal_scale=proposal_scale,
                enable_time_varying_reporting=enable_trend,
                thin=thin,
                fix_durations=fix_durations,
                fixed_parameters=tuple(fixed_parameters or ()),
                sampler=sampler,
                likelihood_observation_frequency=likelihood_interval,
                dispersion=chain_dispersion,
                output_stem=output_stem,
                initial_samples_path=initial_samples_path,
                initial_strategy=initial_strategy,
                solver_mode=solver_mode,
                prior_sd_scale=prior_sd_scale,
                beta_prior_log_sd=beta_prior_log_sd,
                reporting_prior_log_sd=reporting_prior_log_sd,
                ve_prior_sd=ve_prior_sd,
                rel_asym_prior_sd=rel_asym_prior_sd,
                fitness_prior_sd=fitness_prior_sd,
                parameterization=parameterization,
                grid_points=grid_points,
                grid_log_beta_half_width=grid_log_beta_half_width,
                grid_n_chains=n_chains,
            )
            for country_idx, country in enumerate(countries)
            for chain in range(1, n_chains + 1)
        ]

    # Phase 1: posterior sampling or deterministic quadrature.
    # Each country/chain task is independent and can run in parallel.
    effective_n_jobs = n_jobs
    if effective_n_jobs is None or effective_n_jobs < 0:
        cpus = available_cpus()
        # Use up to the number of tasks, leaving 4 CPUs free for system
        effective_n_jobs = min(len(tasks), max(1, cpus - 4))
    outer_n_jobs = int(effective_n_jobs)
    if sampler == "beta_grid":
        if grid_eval_jobs is None or int(grid_eval_jobs) <= 0:
            if reuse_valid_beta_grid:
                compute_task_count = sum(
                    not _existing_beta_grid_valid(
                        task.country,
                        output_stem,
                        tail_drop_target=grid_tail_drop,
                        min_effective_points=grid_min_effective_points,
                        max_single_weight=grid_max_single_weight,
                    )
                    for task in tasks
                )
            else:
                compute_task_count = len(tasks)
            compute_task_count = max(1, min(len(tasks), compute_task_count))
            beta_grid_eval_jobs = max(1, int(effective_n_jobs) // compute_task_count)
        else:
            beta_grid_eval_jobs = int(grid_eval_jobs)
        beta_grid_eval_jobs = min(max(1, beta_grid_eval_jobs), int(effective_n_jobs), available_cpus())
        if beta_grid_eval_jobs > 1:
            outer_n_jobs = max(1, min(len(tasks), int(effective_n_jobs) // beta_grid_eval_jobs))
        tasks = [replace(task, grid_eval_jobs=beta_grid_eval_jobs) for task in tasks]

    _clear_previous_mcmc_progress(countries, n_chains, output_stem=output_stem)
    sampler_desc = "bayesian_beta_grid" if str(sampler).lower() == "beta_grid" else "bayesian_mcmc_chains"
    if outer_n_jobs <= 1:
        sample_frames = [_run_chain(task) for task in tasks]
    else:
        sample_frames = parallel_map(
            _run_chain,
            tasks,
            desc=sampler_desc,
            n_jobs=outer_n_jobs,
        )
    samples = pd.concat(sample_frames, ignore_index=True)
    write_dataframe(
        samples,
        project_path(
            "outputs/simulations",
            f"{_artifact_stem(output_stem, 'bayesian_posterior_samples', 'posterior_samples')}.parquet",
        ),
    )

    # Phase 2: posterior validity diagnostics
    convergence_summary = _write_convergence_diagnostics(samples, output_stem=output_stem)
    if not convergence_summary["all_converged"]:
        import warnings
        sampler_label = "Beta-grid validity" if str(sampler).lower() == "beta_grid" else "MCMC convergence"
        warnings.warn(
            f"{sampler_label} issues detected: "
            f"{convergence_summary['n_parameters_converged']}"
            f"/{convergence_summary['n_parameters_total']} parameters converged. "
            f"Worst descriptive R-hat: {convergence_summary['worst_rhat']:.3f}. "
            f"Consider increasing grid resolution or MCMC warmup/draws as appropriate.",
            stacklevel=2,
        )

    if skip_posterior_predictive:
        return pd.DataFrame(), pd.DataFrame(), samples

    # Phase 3: Posterior predictive simulations (parallelized)
    pp_draws = int(settings.get("posterior_predictive_draws_per_country", 50))
    scenarios = _posterior_predictive_scenarios(samples, pp_draws)
    timeseries, summary = execute_scenario_list(
        scenarios, stem=output_stem, n_jobs=n_jobs
    )
    uncertainty_scope = (
        "conditional_beta_grid"
        if sampler == "beta_grid"
        else "multi_parameter_mcmc"
    )
    write_outputs(
        timeseries,
        summary,
        output_stem,
        extra_metadata={
            "uncertainty_scope": uncertainty_scope,
            "sampler": sampler,
            "fixed_parameters": sorted(str(parameter) for parameter in (fixed_parameters or ())),
            "fix_durations": bool(fix_durations),
            "interpretation_note": (
                "beta_grid outputs integrate over beta_S conditional on fixed nuisance and structural parameters"
                if sampler == "beta_grid"
                else "MCMC outputs sample the configured non-fixed parameter set"
            ),
        },
    )
    _write_interval_summaries(summary, samples, output_stem=output_stem)

    # Phase 4: Stochastic overlay
    _apply_stochastic_overlay(summary, settings, output_stem=output_stem)

    # Phase 5: k-sensitivity sweep (shows CrI sensitivity to dispersion choice)
    if not skip_k_sensitivity:
        _run_k_sensitivity_sweep(summary, settings, output_stem=output_stem)

    return timeseries, summary, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bayesian uncertainty analysis with validated beta-grid or MCMC samplers."
    )
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel workers (-1 for all CPUs)")
    parser.add_argument("--n-chains", type=int, default=None,
                        help="Number of chains to run")
    parser.add_argument("--draws", type=int, default=None,
                        help="Post-warmup draws per chain (default: 2500)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Warmup steps per chain (default: 1500)")
    parser.add_argument("--thin", type=int, default=None,
                        help="Store every Nth post-warmup step")
    parser.add_argument("--proposal-scale", type=float, default=None,
                        help="Multiplier on transformed-scale proposal widths")
    parser.add_argument("--dispersion", type=float, default=None,
                        help="Negative-binomial dispersion for Bayesian likelihood")
    parser.add_argument("--likelihood-observation-frequency", type=str, default=None,
                        help="Observation aggregation interval (e.g. monthly or native)")
    parser.add_argument("--sampler", type=str, default="beta_grid",
                        choices=("adaptive_mh", "componentwise_mh", "slice", "beta_grid"),
                        help="Sampler to use; beta_grid performs deterministic 1D beta posterior integration")
    parser.add_argument("--solver-mode", type=str, default="calibration",
                        choices=("mcmc_fast", "calibration", "production"),
                        help="Solver/runtime fidelity for likelihood evaluations")
    parser.add_argument("--prior-sd-scale", type=float, default=None,
                        help="Scale all Bayesian prior standard deviations for pilot testing")
    parser.add_argument("--beta-prior-log-sd", type=float, default=None,
                        help="Override log_beta_S prior SD")
    parser.add_argument("--reporting-prior-log-sd", type=float, default=None,
                        help="Override log_reporting_multiplier prior SD")
    parser.add_argument("--ve-prior-sd", type=float, default=None,
                        help="Override both VE_sus and VE_inf beta-prior SDs")
    parser.add_argument("--rel-asym-prior-sd", type=float, default=None,
                        help="Override relative_infectiousness_asymptomatic beta-prior SD")
    parser.add_argument("--fitness-prior-sd", type=float, default=None,
                        help="Override fitness_R prior SD")
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM,
                        help="Artifact stem; non-default values isolate pilot outputs")
    parser.add_argument("--initial-samples", type=str, default=None,
                        help="Posterior sample parquet used for warm-start pilots")
    parser.add_argument("--initial-strategy", type=str, default="calibrated",
                        choices=("calibrated", "sample_best", "posterior_best", "chain_best", "sample_random"),
                        help="How to initialize chains when --initial-samples is supplied")
    parser.add_argument("--skip-k-sensitivity", action="store_true",
                        help="Skip the dispersion k sensitivity sweep")
    parser.add_argument("--skip-posterior-predictive", action="store_true",
                        help="Stop after posterior samples and validity diagnostics")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated list of countries to run (for pilot testing)")
    parser.add_argument("--fix-durations", action="store_true",
                        help="Fix infectious durations at calibrated values; beta_grid does this automatically")
    parser.add_argument("--fix-parameters", type=str, default=None,
                        help="Comma-separated sampled parameter names to fix at calibrated values")
    parser.add_argument("--parameterization", type=str, default="standard",
                        choices=("standard", "beta_reporting_product"),
                        help="Transformed-space parameterization for reporting/beta coordinates")
    parser.add_argument("--grid-points", type=int, default=161,
                        help="Number of log-beta grid points for --sampler beta_grid")
    parser.add_argument("--grid-log-beta-half-width", type=float, default=0.08,
                        help="Half-width around calibrated log(beta_S) for --sampler beta_grid")
    parser.add_argument("--grid-max-points", type=int, default=641,
                        help="Maximum log-beta grid points after adaptive beta_grid refinement")
    parser.add_argument("--grid-max-refinements", type=int, default=10,
                        help="Maximum adaptive beta_grid refinement rounds")
    parser.add_argument("--grid-eval-jobs", type=int, default=None,
                        help="Parallel workers for beta_grid point evaluation; defaults to auto allocation")
    parser.add_argument("--grid-tail-drop", type=float, default=20.0,
                        help="Required edge log-posterior drop below the beta_grid mode")
    parser.add_argument("--grid-min-effective-points", type=float, default=10.0,
                        help="Minimum effective grid points required for beta_grid validity")
    parser.add_argument("--grid-max-single-weight", type=float, default=0.20,
                        help="Maximum allowed posterior mass on one beta_grid point")
    parser.add_argument("--grid-smoothing", type=str, default="auto",
                        choices=("auto", "none", "savgol"),
                        help="Optional smoothing for beta_grid log-posterior quadrature")
    parser.add_argument("--grid-savgol-window", type=int, default=21,
                        help="Savitzky-Golay smoothing window for beta_grid when smoothing is active")
    parser.add_argument("--reuse-valid-beta-grid", action="store_true",
                        help="Reuse existing beta_grid country grid files that already pass validity checks")
    args = parser.parse_args()
    main(
        n_jobs=args.n_jobs,
        n_chains=args.n_chains,
        draws=args.draws,
        warmup=args.warmup,
        thin=args.thin,
        proposal_scale=args.proposal_scale,
        dispersion=args.dispersion,
        likelihood_observation_frequency=args.likelihood_observation_frequency,
        skip_k_sensitivity=args.skip_k_sensitivity,
        skip_posterior_predictive=args.skip_posterior_predictive,
        countries_filter=args.countries.split(",") if args.countries else None,
        fix_durations=args.fix_durations,
        fixed_parameters=(
            [name.strip() for name in args.fix_parameters.split(",") if name.strip()]
            if args.fix_parameters else None
        ),
        sampler=args.sampler,
        output_stem=args.output_stem,
        initial_samples_path=args.initial_samples,
        initial_strategy=args.initial_strategy,
        solver_mode=args.solver_mode,
        prior_sd_scale=args.prior_sd_scale,
        beta_prior_log_sd=args.beta_prior_log_sd,
        reporting_prior_log_sd=args.reporting_prior_log_sd,
        ve_prior_sd=args.ve_prior_sd,
        rel_asym_prior_sd=args.rel_asym_prior_sd,
        fitness_prior_sd=args.fitness_prior_sd,
        parameterization=args.parameterization,
        grid_points=args.grid_points,
        grid_log_beta_half_width=args.grid_log_beta_half_width,
        grid_max_points=args.grid_max_points,
        grid_max_refinements=args.grid_max_refinements,
        grid_tail_drop=args.grid_tail_drop,
        grid_min_effective_points=args.grid_min_effective_points,
        grid_max_single_weight=args.grid_max_single_weight,
        grid_smoothing=args.grid_smoothing,
        grid_savgol_window=args.grid_savgol_window,
        reuse_valid_beta_grid=args.reuse_valid_beta_grid,
        grid_eval_jobs=args.grid_eval_jobs,
    )
