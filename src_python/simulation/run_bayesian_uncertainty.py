from __future__ import annotations

"""Optional nonpublication uncertainty research for the transmission model.

The default research route uses each surveillance series once to construct
a regularized annual state-space MAP. A bounded multi-scale Gauss-Newton
mixture proposes beta, reporting, and latent-path states; exact NB2 and AR(1)
target weights correct that proposal before resampling. Those draws are paired
with a shared Latin-hypercube design from external structural priors. This is
deliberately labelled conditional state uncertainty plus structural
sensitivity—not a full joint/hierarchical posterior or MCMC.

This module is not a Figure 2c interval source and is not part of the
publication pipeline. Publication Figure 2c uses the separate full-refit
parametric-bootstrap estimation-CI route.

Legacy beta-grid, importance, SMC, and Metropolis engines remain available for
diagnostics and alternative targets.  They are prevented from reusing the same
likelihood while holding a data-fitted latent path fixed.

Implemented features:
1. Deterministic beta-grid quadrature with tail, effective-grid-point, and
   maximum-single-weight validity gates for conditional diagnostics.
2. Optional Savitzky-Golay smoothing of noisy log-posterior grids, with raw
   grid values retained for audit.
3. Exact-target importance correction of a defensive multi-scale
   Gauss-Newton proposal, with weight and support gates.
4. Adaptive Metropolis (Haario et al. 2001) for explicitly alternative targets.
5. Rank-normalized split-R-hat and ESS summaries (Vehtari et al. 2021) for
   stochastic MCMC output and descriptive beta-grid quantile samples.
6. Dispersion (k) sensitivity sweep on the stochastic overlay.
7. Full multi-core parallelization across countries, chains, and posterior
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
import shutil
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import special as scipy_special
from scipy.stats import genpareto

from src_python.calibration.calibrate_baseline import (
    _annual_ar1_transition_scales,
    _calibration_predicted_means,
    aggregate_observed_case_intervals,
    align_annual_case_series,
    calibration_runtime_config,
    observed_annual_case_frame,
    predicted_case_frame,
    retain_recent_observed_window,
    reporting_rate_prior_penalty,
    grouped_likelihood_observations,
)
from src_python.calibration.likelihood import negative_binomial_nll
from src_python.calibration.state_space_uncertainty import (
    bounded_multivariate_normal_draws as _bounded_multivariate_normal_draws,
)
from src_python.calibration.mcmc_diagnostics import (
    MAX_FATAL_RHAT,
    MAX_RECOMMENDED_RHAT,
    MIN_FATAL_BULK_ESS,
    MIN_FATAL_TAIL_ESS,
    MIN_RECOMMENDED_BULK_ESS,
    MIN_RECOMMENDED_TAIL_ESS,
    compute_diagnostics,
    summarize_convergence,
)
from src_python.model.observables import project_reported_cases, project_reporting_grid
from src_python.simulation.common import (
    PROSPECTIVE_POLICY_KEY,
    current_run_metadata,
    execute_scenario_list,
    load_calibrated_country_artifact,
    load_configs,
    make_config,
    publication_country_names,
    run_prepared_case_exposure,
    run_prepared_config,
    uncertainty_config_fingerprint,
    write_run_metadata,
    write_outputs,
)
from src_python.simulation.bayesian_priors import (
    BAYESIAN_LOCAL_STATE_PARAMETER_NAMES,
    BAYESIAN_PARAMETER_NAMES,
    BAYESIAN_SHARED_PARAMETER_NAMES,
    resolve_bayesian_prior_specs,
)
from src_python.simulation.parameter_distributions import (
    inverse_cdf_from_validated_spec,
    log_pdf_from_validated_spec,
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
    "logit_VE_dur",
    "logit_relative_infectiousness_asymptomatic",
    "log_infectious_duration_symptomatic",
    "log_infectious_duration_asymptomatic",
    "logit_fitness_R_scaled",
)

N_PARAMS = len(PARAMETER_NAMES)
LEGACY_CANONICAL_OUTPUT_STEM = "bayesian_uncertainty"
DEFAULT_OUTPUT_STEM = "bayesian_uncertainty_conditional_research"
RETIRED_MISLABELED_OUTPUT_STEMS = frozenset(
    {
        "bayesian_uncertainty_full_joint",
        "bayesian_uncertainty_figure2c_conditional",
        "bayesian_uncertainty_figure2c_joint",
    }
)
MAX_CASE_EXPOSURE_CACHE_ENTRIES = 32
MAX_SCALAR_LIKELIHOOD_CACHE_ENTRIES = 256
MCMC_SAMPLERS = {"adaptive_mh", "componentwise_mh", "slice"}
JOINT_IMPORTANCE_SAMPLER = "joint_importance"
SMC_SAMPLER = "smc"
STATE_SPACE_EXACT_IMPORTANCE_CUT_SAMPLER = "state_space_exact_importance_cut"
LEGACY_STATE_SPACE_LAPLACE_CUT_SAMPLER = "state_space_laplace_cut"
# Internal compatibility name retained for downstream imports; its value is
# now the scientifically accurate canonical sampler literal.
STATE_SPACE_LAPLACE_CUT_SAMPLER = STATE_SPACE_EXACT_IMPORTANCE_CUT_SAMPLER
SUPPORTED_MULTIPARAMETER_SAMPLERS = MCMC_SAMPLERS | {
    JOINT_IMPORTANCE_SAMPLER,
    SMC_SAMPLER,
    STATE_SPACE_LAPLACE_CUT_SAMPLER,
    LEGACY_STATE_SPACE_LAPLACE_CUT_SAMPLER,
}
DEFAULT_BETA_GRID_FIXED_PARAMETERS = (
    "reporting_multiplier",
    "VE_sus",
    "VE_inf",
    "VE_dur",
    "relative_infectiousness_asymptomatic",
    "fitness_R",
)

PARAMETER_SAMPLE_COLUMNS = (
    "beta_S",
    "reporting_multiplier",
    "VE_sus",
    "VE_inf",
    "VE_dur",
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
    "logit_VE_dur": "VE_dur",
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
# These are deliberately conservative because country likelihoods with
# high-count surveillance data can be very sharp, especially for China/Japan.
INITIAL_PROPOSAL_SCALES = np.array(
    [0.025, 0.030, 0.035, 0.035, 0.045, 0.035, 0.025, 0.030, 0.035],
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
RM_LOG_SCALE_MIN = -7.0
RM_LOG_SCALE_MAX = 4.5
LOCAL_PROPOSAL_PROBABILITY = 0.35
BLOCK_PROPOSAL_PROBABILITY = 0.45
PROPOSAL_BLOCKS = (
    (0, 1),  # transmission/reporting scale (strongly correlated)
    (2, 3, 4, 5),  # vaccine acquisition, infectiousness, duration, asymptomatic contribution
    (6, 7),  # infectious durations
    (8,),  # resistance fitness (single parameter)
)

# Joint-importance defaults for a legacy full-feedback sensitivity path. The sampler draws
# weakly identified nuisance parameters from their configured priors, then
# integrates the sharp beta/reporting posterior on a local grid.  Reporting is an
# observation-layer multiplier, so it is evaluated without rerunning the ODE.
DEFAULT_IMPORTANCE_NUISANCE_DRAWS = 128
DEFAULT_IMPORTANCE_BETA_GRID_POINTS = 13
DEFAULT_IMPORTANCE_REPORTING_GRID_POINTS = 41
DEFAULT_IMPORTANCE_LOG_BETA_HALF_WIDTH = 0.22
DEFAULT_IMPORTANCE_REPORTING_COORDINATE_HALF_WIDTH = 1.60
DEFAULT_IMPORTANCE_DEFENSIVE_PRIOR_FRACTION = 0.20
DEFAULT_IMPORTANCE_PROPOSAL_FLOOR_FRACTION = 0.05
IMPORTANCE_FATAL_MAX_WEIGHT = 0.05
IMPORTANCE_RECOMMENDED_MAX_WEIGHT = 0.02
IMPORTANCE_FATAL_EDGE_WEIGHT = 0.05
IMPORTANCE_RECOMMENDED_EDGE_WEIGHT = 0.01
IMPORTANCE_FATAL_NUISANCE_ESS = 50.0
IMPORTANCE_RECOMMENDED_NUISANCE_ESS = 100.0
# A modular cut does not estimate a nuisance posterior: its structural draws
# are an equal-mass numerical design from the external-evidence prior.  The
# thresholds below therefore assess design size and the *conditional*
# beta/reporting quadrature for every structural draw; they must not be
# described as nuisance convergence ESS thresholds.
MODULAR_CUT_FATAL_MIN_STRUCTURAL_DRAWS = 50
MODULAR_CUT_RECOMMENDED_MIN_STRUCTURAL_DRAWS = 100
MODULAR_CUT_FATAL_MIN_CONDITIONAL_EFFECTIVE_POINTS = 10.0
MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS = 20.0
MODULAR_CUT_FATAL_MIN_AXIS_EFFECTIVE_POINTS = 3.0
MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS = 5.0
MODULAR_CUT_FATAL_MAX_CONDITIONAL_WEIGHT = 0.20
MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT = 0.10
MODULAR_CUT_FATAL_MAX_CONDITIONAL_EDGE_WEIGHT = IMPORTANCE_FATAL_EDGE_WEIGHT
MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT = IMPORTANCE_RECOMMENDED_EDGE_WEIGHT
STATE_LAPLACE_FATAL_MAX_BOUNDARY_REJECTION = 0.75
STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION = 0.50
STATE_IMPORTANCE_FATAL_MIN_ESS = 50.0
STATE_IMPORTANCE_RECOMMENDED_MIN_ESS = 200.0
STATE_IMPORTANCE_FATAL_MAX_WEIGHT = 0.10
STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT = 0.02
STATE_IMPORTANCE_FATAL_MAX_PARETO_K = 1.0
STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K = 0.7
STATE_IMPORTANCE_FATAL_MIN_TAIL_ESS = 5.0
STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS = 20.0
IMPORTANCE_PARAMETERS = PARAMETER_SAMPLE_COLUMNS
IMPORTANCE_NUISANCE_INDICES = np.array([2, 3, 4, 5, 6, 7, 8], dtype=int)

# Tempered sequential Monte Carlo defaults for the legacy full-feedback
# posterior sensitivity route. It is not the publication conditional interval
# engine. SMC retains explicit weight, temperature, and particle-diversity
# diagnostics for sharp multi-modal targets.
DEFAULT_SMC_PARTICLES = 512
DEFAULT_SMC_ESS_FRACTION = 0.65
DEFAULT_SMC_MOVE_STEPS = 3
DEFAULT_SMC_MAX_STAGES = 80
SMC_FATAL_ESS_FRACTION = 0.25
SMC_RECOMMENDED_ESS_FRACTION = 0.50
SMC_FATAL_MAX_WEIGHT = 0.05
SMC_RECOMMENDED_MAX_WEIGHT = 0.02
SMC_FATAL_UNIQUE_PARTICLE_FRACTION = 0.10
SMC_RECOMMENDED_UNIQUE_PARTICLE_FRACTION = 0.25
SMC_FATAL_ISLAND_ESS = 2.0
SMC_RECOMMENDED_ISLAND_ESS_FRACTION = 0.35
SMC_FATAL_MAX_ISLAND_WEIGHT = 0.80
SMC_RECOMMENDED_MAX_ISLAND_WEIGHT = 0.50
SMC_LOCAL_PROPOSAL_PROBABILITY = 0.20
SMC_BLOCK_PROPOSAL_PROBABILITY = 0.45
SMC_LOW_ACCEPTANCE = 0.18
SMC_HIGH_ACCEPTANCE = 0.35


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
    fix_durations: bool = False  # if True, fix infectious durations (7D sampling)
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
    importance_nuisance_draws: int = DEFAULT_IMPORTANCE_NUISANCE_DRAWS
    importance_beta_grid_points: int = DEFAULT_IMPORTANCE_BETA_GRID_POINTS
    importance_reporting_grid_points: int = DEFAULT_IMPORTANCE_REPORTING_GRID_POINTS
    importance_log_beta_half_width: float = DEFAULT_IMPORTANCE_LOG_BETA_HALF_WIDTH
    importance_reporting_coordinate_half_width: float = DEFAULT_IMPORTANCE_REPORTING_COORDINATE_HALF_WIDTH
    importance_defensive_prior_fraction: float = DEFAULT_IMPORTANCE_DEFENSIVE_PRIOR_FRACTION
    smc_particles: int = DEFAULT_SMC_PARTICLES
    smc_ess_fraction: float = DEFAULT_SMC_ESS_FRACTION
    smc_move_steps: int = DEFAULT_SMC_MOVE_STEPS
    smc_max_stages: int = DEFAULT_SMC_MAX_STAGES


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
    """Return historical artifact names only for the explicit legacy stem."""
    output_stem = str(output_stem or DEFAULT_OUTPUT_STEM)
    if output_stem == LEGACY_CANONICAL_OUTPUT_STEM:
        return canonical_stem
    return f"{output_stem}_{suffix}"


def _mcmc_progress_dir(output_stem: str) -> Any:
    if str(output_stem or DEFAULT_OUTPUT_STEM) == LEGACY_CANONICAL_OUTPUT_STEM:
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


def _odd_grid_points(value: int, *, minimum: int = 3) -> int:
    points = int(max(value, minimum))
    if points % 2 == 0:
        points += 1
    return points


def _latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic Latin-hypercube points in (0, 1) for prior proposals."""
    n = int(max(n, 1))
    dim = int(max(dim, 1))
    rng = np.random.default_rng(seed)
    out = np.empty((n, dim), dtype=float)
    for j in range(dim):
        points = (np.arange(n, dtype=float) + rng.random(n)) / float(n)
        rng.shuffle(points)
        out[:, j] = np.clip(points, 1e-9, 1.0 - 1e-9)
    return out


def _configured_prior_spec(priors: dict[str, Any], name: str) -> dict[str, Any] | None:
    specs = priors.get("_distribution_specs", {})
    if isinstance(specs, dict) and name in specs:
        return specs[name]
    return None


def _configured_prior_quantile(
    priors: dict[str, Any],
    name: str,
    unit: float,
) -> float | None:
    spec = _configured_prior_spec(priors, name)
    if spec is None:
        return None
    return float(
        inverse_cdf_from_validated_spec(
            float(unit), spec, context=f"Bayesian prior {name}"
        )
    )


def _configured_prior_bounds(
    priors: dict[str, Any],
    name: str,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    spec = _configured_prior_spec(priors, name)
    if spec is None:
        return fallback
    return float(spec["low"]), float(spec["high"])


def _configured_prior_logpdf(priors: dict[str, Any], name: str, value: float) -> float:
    spec = _configured_prior_spec(priors, name)
    if spec is None:
        raise KeyError(f"No configured distribution specification for Bayesian prior {name!r}")
    return float(
        log_pdf_from_validated_spec(
            float(value), spec, context=f"Bayesian prior {name}"
        )
    )


def _beta_prior_quantile(prior: dict[str, Any], u: float) -> float:
    alpha, beta = _beta_ab(float(prior["mean"]), float(prior["sd"]))
    return float(np.clip(scipy_special.betaincinv(alpha, beta, float(u)), 1e-9, 1.0 - 1e-9))


def _truncated_normal_quantile(mean: float, sd: float, lower: float, upper: float, u: float) -> float:
    sd = max(float(sd), 1e-9)
    lower_z = (float(lower) - float(mean)) / sd
    upper_z = (float(upper) - float(mean)) / sd
    lower_cdf = float(scipy_special.ndtr(lower_z))
    upper_cdf = float(scipy_special.ndtr(upper_z))
    if upper_cdf <= lower_cdf:
        return float(np.clip(mean, lower, upper))
    target = lower_cdf + float(u) * (upper_cdf - lower_cdf)
    return float(mean + sd * scipy_special.ndtri(np.clip(target, 1e-12, 1.0 - 1e-12)))


def _truncated_lognormal_quantile(
    median: float,
    log_sd: float,
    lower: float,
    upper: float,
    u: float,
) -> float:
    value = np.exp(
        _truncated_normal_quantile(
            np.log(float(median)),
            float(log_sd),
            np.log(float(lower)),
            np.log(float(upper)),
            float(u),
        )
    )
    return float(np.clip(value, lower, upper))


def _nuisance_vector_from_unit_cube(
    unit: np.ndarray,
    calibrated_start: np.ndarray,
    base_config: dict[str, Any],
    priors: dict[str, Any],
) -> np.ndarray:
    """Map a 7D unit-cube prior draw onto transformed nuisance coordinates."""
    if len(unit) != len(IMPORTANCE_NUISANCE_INDICES):
        raise ValueError(
            f"Expected {len(IMPORTANCE_NUISANCE_INDICES)} nuisance unit values, got {len(unit)}"
        )
    vector = calibrated_start.copy()
    for vector_idx, name, cube_idx in (
        (2, "VE_sus", 0),
        (3, "VE_inf", 1),
        (4, "VE_dur", 2),
        (5, "relative_infectiousness_asymptomatic", 3),
    ):
        value = _configured_prior_quantile(priors, name, unit[cube_idx])
        if value is None:
            value = _beta_prior_quantile(priors[name], unit[cube_idx])
        vector[vector_idx] = _logit(value)

    symptomatic = _configured_prior_quantile(
        priors, "infectious_duration_symptomatic", unit[4]
    )
    if symptomatic is None:
        symptomatic = _truncated_lognormal_quantile(
            float(base_config["natural_history"]["infectious_duration_symptomatic"]),
            float(priors["infectious_duration_symptomatic"].get("log_sd", 0.15)),
            7.0,
            35.0,
            unit[4],
        )
    vector[6] = np.log(symptomatic)

    asymptomatic = _configured_prior_quantile(
        priors, "infectious_duration_asymptomatic", unit[5]
    )
    if asymptomatic is None:
        asymptomatic = _truncated_lognormal_quantile(
            float(base_config["natural_history"]["infectious_duration_asymptomatic"]),
            float(priors["infectious_duration_asymptomatic"].get("log_sd", 0.20)),
            5.0,
            28.0,
            unit[5],
        )
    vector[7] = np.log(asymptomatic)

    fitness_prior = priors["fitness_R"]
    fitness = _configured_prior_quantile(priors, "fitness_R", unit[6])
    if fitness is None:
        fitness = _truncated_normal_quantile(
            float(fitness_prior.get("mean", 1.0)),
            float(fitness_prior.get("sd", 0.12)),
            float(fitness_prior.get("min", 0.70)),
            float(fitness_prior.get("max", 1.25)),
            unit[6],
        )
    fitness_lower, fitness_upper = _configured_prior_bounds(
        priors,
        "fitness_R",
        (float(fitness_prior.get("min", 0.70)), float(fitness_prior.get("max", 1.25))),
    )
    vector[8] = _value_to_scaled_logit(
        fitness,
        fitness_lower,
        fitness_upper,
    )
    return vector


def _full_vector_from_unit_cube(
    unit: np.ndarray,
    calibrated_start: np.ndarray,
    base_config: dict[str, Any],
    priors: dict[str, Any],
) -> np.ndarray:
    """Map a 9D unit-cube prior draw onto the full transformed parameter vector."""
    unit = np.asarray(unit, dtype=float)
    if len(unit) != N_PARAMS:
        raise ValueError(f"Expected {N_PARAMS} unit values, got {len(unit)}")
    vector = _nuisance_vector_from_unit_cube(
        unit[2:],
        calibrated_start,
        base_config,
        priors,
    )
    beta_value = _configured_prior_quantile(priors, "beta_S", unit[0])
    if beta_value is None:
        log_beta = _truncated_normal_quantile(
            np.log(float(base_config["transmission"]["beta_S"])),
            float(priors.get("log_beta_S_sd", 0.40)),
            np.log(0.0005),
            np.log(0.5),
            unit[0],
        )
    else:
        log_beta = np.log(beta_value)
    reporting_value = _configured_prior_quantile(priors, "reporting_multiplier", unit[1])
    if reporting_value is None:
        log_reporting = _truncated_normal_quantile(
            np.log(float(base_config.get("reporting_multiplier", 1.0))),
            float(priors.get("log_reporting_multiplier_sd", 0.40)),
            np.log(0.02),
            np.log(20.0),
            unit[1],
        )
    else:
        log_reporting = np.log(reporting_value)
    vector[0] = float(log_beta)
    vector[1] = _encode_reporting_coordinate(float(log_beta), float(log_reporting), priors)
    return vector


# ---------------------------------------------------------------------------
# Data loading and parameter vector construction
# ---------------------------------------------------------------------------

def _country_observed(country: str, interval: str = "native") -> pd.DataFrame:
    configs = load_configs()
    observed = observed_annual_case_frame(country)
    recent_years = int(configs["baseline"].get("calibration", {}).get("recent_years", 0))
    observed = retain_recent_observed_window(observed, recent_years)
    _validate_observed_likelihood_frame(observed, country=country)
    aggregated = _aggregate_observed_intervals(observed, interval)
    _validate_observed_likelihood_frame(aggregated, country=country)
    return aggregated


def _validate_observed_likelihood_frame(
    observed: pd.DataFrame,
    *,
    country: str,
) -> None:
    if observed.empty:
        raise ValueError(f"No observed likelihood intervals for {country}")
    cases = pd.to_numeric(observed.get("reported_cases"), errors="coerce")
    if cases.isna().any() or np.any(cases.to_numpy(dtype=float) < 0.0):
        raise ValueError(f"Observed likelihood cases must be finite and non-negative for {country}")
    if "observed_interval_id" in observed.columns and observed[
        "observed_interval_id"
    ].astype(str).duplicated().any():
        duplicates = sorted(
            observed.loc[
                observed["observed_interval_id"].astype(str).duplicated(keep=False),
                "observed_interval_id",
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(
            f"Duplicate observed likelihood intervals for {country}: {duplicates}"
        )
    if {"period_start", "period_end"}.issubset(observed.columns):
        intervals = observed.loc[:, ["period_start", "period_end"]].copy()
        intervals["period_start"] = pd.to_datetime(intervals["period_start"], errors="coerce")
        intervals["period_end"] = pd.to_datetime(intervals["period_end"], errors="coerce")
        if intervals.isna().any().any() or np.any(
            intervals["period_end"] <= intervals["period_start"]
        ):
            raise ValueError(f"Invalid observed likelihood dates for {country}")
        intervals = intervals.sort_values(["period_start", "period_end"])
        if len(intervals) > 1 and np.any(
            intervals["period_start"].iloc[1:].to_numpy()
            < intervals["period_end"].iloc[:-1].to_numpy()
        ):
            raise ValueError(f"Overlapping observed likelihood intervals for {country}")


def _aggregate_observed_intervals(observed: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate high-frequency surveillance intervals for Bayesian likelihoods.

    Weekly reports are autocorrelated and much noisier than the deterministic
    ODE can represent.  Treating every week as an independent NB observation
    over-weights large high-frequency datasets and creates pathologically
    narrow posteriors.  Monthly aggregation preserves the epidemic-scale signal
    while conserving reported cases.
    """
    return aggregate_observed_case_intervals(observed, interval)


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
    start_at_prior_centers: bool = False,
) -> np.ndarray:
    fitness_bounds = load_configs()["baseline"]["bayesian_uncertainty"]["priors"]["fitness_R"]
    priors = dict(priors or {})
    fitness_lower, fitness_upper = _configured_prior_bounds(
        priors,
        "fitness_R",
        (
            float(fitness_bounds.get("min", 0.70)),
            float(fitness_bounds.get("max", 1.25)),
        ),
    )
    log_beta = np.log(float(config["transmission"]["beta_S"]))
    log_reporting = np.log(float(config.get("reporting_multiplier", 1.0)))
    vaccine = config["vaccine"]
    transmission = config["transmission"]
    natural_history = config["natural_history"]
    ve_sus = float(vaccine.get("VE_sus", 0.0))
    ve_inf = float(vaccine.get("VE_inf", 0.0))
    ve_dur = float(vaccine.get("VE_dur", 0.0))
    rel_asym = float(transmission["relative_infectiousness_asymptomatic"])
    fitness_r = float(transmission.get("fitness_R", 1.0))
    duration_sym = float(natural_history["infectious_duration_symptomatic"])
    duration_asym = float(natural_history["infectious_duration_asymptomatic"])
    if start_at_prior_centers:
        # Weakly identified nuisance dimensions should not begin at structural
        # scenario boundaries.  In particular, current aP has VE_dur=0; on the
        # logit scale that is ~18.5 units from the configured prior centre and
        # can make every chain appear stable while trapped near zero.
        ve_sus = float(priors.get("VE_sus", {}).get("mean", ve_sus))
        ve_inf = float(priors.get("VE_inf", {}).get("mean", ve_inf))
        ve_dur = float(priors.get("VE_dur", {}).get("mean", ve_dur))
        rel_asym = float(priors.get("relative_infectiousness_asymptomatic", {}).get("mean", rel_asym))
        fitness_r = float(priors.get("fitness_R", {}).get("mean", fitness_r))
        for name, current_name in (
            ("VE_sus", "ve_sus"),
            ("VE_inf", "ve_inf"),
            ("VE_dur", "ve_dur"),
            ("relative_infectiousness_asymptomatic", "rel_asym"),
            ("fitness_R", "fitness_r"),
        ):
            centre = _configured_prior_quantile(priors, name, 0.5)
            if centre is not None:
                if current_name == "ve_sus":
                    ve_sus = centre
                elif current_name == "ve_inf":
                    ve_inf = centre
                elif current_name == "ve_dur":
                    ve_dur = centre
                elif current_name == "rel_asym":
                    rel_asym = centre
                else:
                    fitness_r = centre
        duration_sym_centre = _configured_prior_quantile(
            priors, "infectious_duration_symptomatic", 0.5
        )
        duration_asym_centre = _configured_prior_quantile(
            priors, "infectious_duration_asymptomatic", 0.5
        )
        if duration_sym_centre is not None:
            duration_sym = duration_sym_centre
        if duration_asym_centre is not None:
            duration_asym = duration_asym_centre
    vec = [
        log_beta,
        _encode_reporting_coordinate(log_beta, log_reporting, priors),
        _logit(ve_sus),
        _logit(ve_inf),
        _logit(ve_dur),
        _logit(rel_asym),
        np.log(duration_sym),
        np.log(duration_asym),
        _value_to_scaled_logit(
            fitness_r,
            fitness_lower,
            fitness_upper,
        ),
    ]
    return np.array(vec, dtype=float)


def _sample_from_vector(vector: np.ndarray, priors: dict[str, Any]) -> dict[str, float]:
    fitness = priors["fitness_R"]
    fitness_lower, fitness_upper = _configured_prior_bounds(
        priors,
        "fitness_R",
        (float(fitness.get("min", 0.70)), float(fitness.get("max", 1.25))),
    )
    log_beta = float(vector[0])
    log_reporting = _decode_reporting_coordinate(log_beta, float(vector[1]), priors)
    return {
        "beta_S": float(np.exp(log_beta)),
        "reporting_multiplier": float(np.exp(log_reporting)),
        "VE_sus": _inv_logit(vector[2]),
        "VE_inf": _inv_logit(vector[3]),
        "VE_dur": _inv_logit(vector[4]),
        "relative_infectiousness_asymptomatic": _inv_logit(vector[5]),
        "infectious_duration_symptomatic": float(np.exp(vector[6])),
        "infectious_duration_asymptomatic": float(np.exp(vector[7])),
        "fitness_R": _scaled_logit_to_value(
            vector[8],
            fitness_lower,
            fitness_upper,
        ),
        "resistance_prevalence": float(priors.get("resistance_prevalence_fixed", 0.30)),
        "reporting_trend_end_multiplier": float(priors.get("reporting_trend_fixed", 1.0)),
    }


def _vector_from_sample(sample: dict[str, float], priors: dict[str, Any]) -> np.ndarray:
    fitness = priors["fitness_R"]
    fitness_lower, fitness_upper = _configured_prior_bounds(
        priors,
        "fitness_R",
        (float(fitness.get("min", 0.70)), float(fitness.get("max", 1.25))),
    )
    log_beta = np.log(float(sample["beta_S"]))
    log_reporting = np.log(float(sample["reporting_multiplier"]))
    return np.array(
        [
            log_beta,
            _encode_reporting_coordinate(log_beta, log_reporting, priors),
            _logit(float(sample["VE_sus"])),
            _logit(float(sample["VE_inf"])),
            _logit(float(sample["VE_dur"])),
            _logit(float(sample["relative_infectiousness_asymptomatic"])),
            np.log(float(sample["infectious_duration_symptomatic"])),
            np.log(float(sample["infectious_duration_asymptomatic"])),
            _value_to_scaled_logit(
                float(sample["fitness_R"]),
                fitness_lower,
                fitness_upper,
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


def _apply_sampled_log_beta_process_path(
    config: dict[str, Any],
    sample: dict[str, float],
) -> None:
    sampled = {
        int(str(key).removeprefix("log_beta_process_")): float(value)
        for key, value in sample.items()
        if str(key).startswith("log_beta_process_")
    }
    if not sampled:
        return
    variation = config.get("transmission", {}).get("log_beta_time_variation")
    if not isinstance(variation, dict) or not isinstance(variation.get("periods"), list):
        raise ValueError("Sampled latent process path requires calibrated calendar periods")
    periods = variation["periods"]
    period_years: list[int] = []
    for period in periods:
        start = pd.Timestamp(period.get("start_date"))
        end = pd.Timestamp(period.get("end_date"))
        if (
            start.month != 1
            or start.day != 1
            or end.month != 12
            or end.day != 31
            or start.year != end.year
        ):
            raise ValueError("Sampled annual latent process path found a non-annual period")
        period_years.append(int(start.year))
    missing_sampled_years = sorted(set(period_years) - set(sampled))
    if missing_sampled_years:
        raise ValueError(
            "Sampled latent process path is missing calibrated years: "
            f"{missing_sampled_years}"
        )
    extra_years = sorted(set(sampled) - set(period_years))
    if extra_years:
        expected = list(range(max(period_years) + 1, max(extra_years) + 1))
        if extra_years != expected:
            raise ValueError(
                "Sampled forecast process years must extend calibrated periods consecutively"
            )
        for year in extra_years:
            periods.append(
                {
                    "start_date": f"{year:04d}-01-01",
                    "end_date": f"{year:04d}-12-31",
                    "log_multiplier": float(sampled[year]),
                    "state_origin": "AR1_prior_predictive_draw",
                }
            )
            period_years.append(year)
    for period, year in zip(periods, period_years):
        value = float(sampled[year])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite sampled latent log-beta value for {year}")
        period["log_multiplier"] = value
    variation["interpretation"] = "latent_AR1_process_exact_importance_draw"


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
    _apply_sampled_log_beta_process_path(out, sample)

    # Resistance prevalence is fixed at the country-calibrated value (not sampled)
    resistance = float(np.clip(sample["resistance_prevalence"], 0.0, 1.0))
    out["initial_conditions"]["initial_resistance_prevalence"] = resistance
    out.setdefault("resistance", {})["target_prevalence_at_analysis_start"] = resistance
    out["resistance"]["importation_fraction"] = resistance
    out.setdefault("importation", {})["resistant_fraction"] = resistance

    # A posterior biological draw defines the historical trajectory as well as
    # the prospective projection.  Reporting remains observation-only and is
    # intentionally not copied into the attached history.  Without this
    # propagation, paired policy runs would burn in at the calibrated centre
    # and branch using a different posterior draw, breaking both pairing and
    # scientific interpretation.
    policy_spec = out.get(PROSPECTIVE_POLICY_KEY)
    if isinstance(policy_spec, dict) and isinstance(policy_spec.get("history_config"), dict):
        history = policy_spec["history_config"]
        history["transmission"]["beta_S"] = sample["beta_S"]
        history["vaccine"]["VE_sus"] = sample["VE_sus"]
        history["vaccine"]["VE_inf"] = sample["VE_inf"]
        history["vaccine"]["VE_dur"] = sample["VE_dur"]
        history["transmission"]["relative_infectiousness_asymptomatic"] = sample[
            "relative_infectiousness_asymptomatic"
        ]
        history["natural_history"]["infectious_duration_symptomatic"] = sample[
            "infectious_duration_symptomatic"
        ]
        history["natural_history"]["infectious_duration_asymptomatic"] = sample[
            "infectious_duration_asymptomatic"
        ]
        history["transmission"]["fitness_R"] = sample["fitness_R"]
        _apply_sampled_log_beta_process_path(history, sample)
        history["initial_conditions"]["initial_resistance_prevalence"] = resistance
        history.setdefault("resistance", {})["target_prevalence_at_analysis_start"] = resistance
        history["resistance"]["importation_fraction"] = resistance
        history.setdefault("importation", {})["resistant_fraction"] = resistance

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


def _registry_log_prior_transformed(
    vector: np.ndarray,
    sample: dict[str, float],
    priors: dict[str, Any],
    names: tuple[str, ...],
) -> float:
    """Evaluate registry priors on the sampler's transformed coordinates."""

    if not all(_configured_prior_spec(priors, name) is not None for name in names):
        raise KeyError("Incomplete Bayesian distribution registry")
    logp = 0.0
    for name in names:
        component = PARAMETER_INDEX_BY_SAMPLE[name]
        value = float(sample[name])
        if not isfinite(value):
            return -np.inf
        component_logp = _configured_prior_logpdf(priors, name, value)
        if not isfinite(component_logp):
            return -np.inf
        logp += component_logp
        if name in {"beta_S", "reporting_multiplier"}:
            # Both parameters are represented by logs.  The optional
            # beta-reporting-product coordinate is a unit-determinant linear
            # map on the two log coordinates, so the same two Jacobian terms
            # apply under either parameterization.
            logp += np.log(value)
        elif name in {
            "VE_sus",
            "VE_inf",
            "VE_dur",
            "relative_infectiousness_asymptomatic",
        }:
            logp += _log_jacobian_logit(float(vector[component]))
        elif name in {
            "infectious_duration_symptomatic",
            "infectious_duration_asymptomatic",
        }:
            logp += float(vector[component])
        elif name == "fitness_R":
            lower, upper = _configured_prior_bounds(priors, name, (0.70, 1.25))
            logp += _log_jacobian_scaled_logit(float(vector[component]), lower, upper)
        else:  # Defensive if the registry is extended without a transform.
            raise KeyError(f"No transformed-coordinate Jacobian is defined for {name!r}")
    return float(logp) if isfinite(logp) else -np.inf


def _log_prior(
    vector: np.ndarray,
    base_config: dict[str, Any],
    country: str,
    settings: dict[str, Any],
) -> float:
    priors = settings["priors"]
    sample = _sample_from_vector(vector, priors)

    registry_names = tuple(PARAMETER_SAMPLE_COLUMNS)
    if all(_configured_prior_spec(priors, name) is not None for name in registry_names):
        return _registry_log_prior_transformed(vector, sample, priors, registry_names)

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
        ("VE_dur", 4),
        ("relative_infectiousness_asymptomatic", 5),
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
        vector[8],
        float(fitness_prior.get("min", 0.70)),
        float(fitness_prior.get("max", 1.25)),
    )

    return float(logp)


def _biological_likelihood_key(config: dict[str, Any]) -> tuple[Any, ...]:
    """Coordinates that can change biological exposure within one run.

    The cache itself is chain/worker-local, so country-level schedules and
    other fixed configuration blocks are already constant.  Observation-only
    reporting parameters are deliberately excluded to make a reporting-axis
    proposal a pure NumPy projection.
    """

    transmission = config["transmission"]
    vaccine = config["vaccine"]
    natural_history = config["natural_history"]
    initial = config["initial_conditions"]
    resistance = config.get("resistance", {})
    importation = config.get("importation", {})
    observation = config.get("observation_model", {})
    log_beta_periods = tuple(
        (
            str(period.get("start_date", "")),
            str(period.get("end_date", "")),
            float(period.get("log_multiplier", 0.0)),
        )
        for period in transmission.get("log_beta_time_variation", {}).get("periods", [])
        if isinstance(period, dict)
    )
    return (
        float(transmission["beta_S"]),
        float(transmission.get("seasonal_amplitude", 0.0)),
        float(transmission.get("seasonal_phase", 0.0)),
        float(transmission.get("multi_year_amplitude", 0.0)),
        float(transmission.get("multi_year_period_years", 4.0)),
        float(transmission.get("multi_year_phase", 0.0)),
        float(transmission.get("relative_infectiousness_asymptomatic", 0.0)),
        float(transmission.get("fitness_R", 1.0)),
        float(vaccine.get("VE_sus", 0.0)),
        float(vaccine.get("VE_sym", 0.0)),
        float(vaccine.get("VE_inf", 0.0)),
        float(vaccine.get("VE_dur", 0.0)),
        float(natural_history["infectious_duration_symptomatic"]),
        float(natural_history["infectious_duration_asymptomatic"]),
        float(initial.get("initial_resistance_prevalence", 0.0)),
        float(resistance.get("target_prevalence_at_analysis_start", 0.0)),
        float(resistance.get("importation_fraction", 0.0)),
        bool(importation.get("enabled", False)),
        float(importation.get("rate_per_100k_per_year", 0.0)),
        float(importation.get("resistant_fraction", 0.0)),
        str(observation.get("case_event_definition", "infection_destined_symptomatic")),
        log_beta_periods,
    )


def _observation_plan_key(config: dict[str, Any]) -> tuple[Any, ...]:
    trend = config.get("reporting_time_variation", {})
    diagnostic = config.get("diagnostic_reporting_time_variation", {})
    diagnostic_periods = tuple(
        (
            str(period.get("start_date", "")),
            str(period.get("end_date", "")),
            float(period.get("multiplier", 1.0)),
        )
        for period in diagnostic.get("periods", [])
        if isinstance(period, dict)
    )
    return (
        float(trend.get("start_multiplier", 1.0)),
        float(trend.get("end_multiplier", 1.0)),
        bool(diagnostic.get("enabled", False)),
        diagnostic_periods,
        str(config.get("calendar", {}).get("analysis_start_date", "")),
        float(config.get("simulation", {}).get("start_time", 0.0)),
        float(config.get("simulation", {}).get("end_time", 0.0)),
    )


def _observed_interval_key(observed: pd.DataFrame) -> tuple[Any, ...]:
    columns = [
        column
        for column in (
            "observed_interval_id",
            "period_start",
            "period_end",
            "observed_year",
        )
        if column in observed.columns
    ]
    return tuple(
        tuple(str(value) for value in observed[column].tolist())
        for column in columns
    )


def _base_reporting_rates(config: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(record.get("reporting_rate", 0.0)) for record in config.get("age_groups", [])],
        dtype=float,
    )


def _bounded_cache_set(
    cache: dict[Any, Any],
    key: tuple[Any, ...],
    value: Any,
    *,
    limit: int,
) -> None:
    cache.pop(key, None)
    cache[key] = value
    namespace = key[0]
    matching = [item for item in cache if isinstance(item, tuple) and item and item[0] == namespace]
    while len(matching) > limit:
        oldest = matching.pop(0)
        if oldest != key:
            cache.pop(oldest, None)


def _log_likelihood(
    config: dict[str, Any],
    observed: pd.DataFrame,
    country: str,
    dispersion: float,
    _cache: dict[Any, Any] | None = None,
) -> float:
    """Compute log-likelihood with optional caching to avoid redundant ODE solves.
    
    For sampler efficiency, we cache results by parameter hash to avoid re-solving
    the ODE for identical parameter sets (which can happen during rejection steps).
    """
    try:
        biological_key = _biological_likelihood_key(config)
        plan_key = (
            _observation_plan_key(config),
            _observed_interval_key(observed),
        )
        base_reporting_rates = _base_reporting_rates(config)
        likelihood_key = (
            "likelihood",
            biological_key,
            plan_key,
            tuple(float(value) for value in base_reporting_rates),
            float(config.get("reporting_multiplier", 1.0)),
            float(dispersion),
        )
        if _cache is not None and likelihood_key in _cache:
            cached_likelihood = _cache.pop(likelihood_key)
            _cache[likelihood_key] = cached_likelihood
            return float(cached_likelihood)

        exposure_key = ("case_exposure", biological_key, plan_key)
        cached_exposure = _cache.get(exposure_key) if _cache is not None else None
        if cached_exposure is not None and _cache is not None:
            _cache.pop(exposure_key)
            _cache[exposure_key] = cached_exposure
        if cached_exposure is None:
            exposure, _prepared_base_reporting_rates = run_prepared_case_exposure(
                config,
                observed,
                analysis="bayesian_calibration",
                scenario="candidate",
                vaccine_scenario=config["baseline_vaccine_scenario"],
                resistance_scenario=config["baseline_resistance_scenario"],
                metadata={"country": country},
            )
            if not np.array_equal(
                np.asarray(_prepared_base_reporting_rates, dtype=float),
                base_reporting_rates,
            ):
                raise RuntimeError(
                    "Prepared and configured base reporting rates disagree"
                )
            if _cache is not None:
                _bounded_cache_set(
                    _cache,
                    exposure_key,
                    exposure,
                    limit=MAX_CASE_EXPOSURE_CACHE_ENTRIES,
                )
        else:
            exposure = cached_exposure
        predicted = project_reported_cases(
            exposure,
            base_reporting_rates,
            float(config.get("reporting_multiplier", 1.0)),
        ).interval_total_mean
        observed_values = pd.to_numeric(
            observed["reported_cases"], errors="coerce"
        ).to_numpy(dtype=float)
        if len(observed_values) < 3 or len(predicted) != len(observed_values):
            result = -np.inf
        else:
            result = float(
                -negative_binomial_nll(
                    observed_values,
                    predicted,
                    dispersion=dispersion,
                )
            )
        
        # Cache the result
        if _cache is not None:
            _bounded_cache_set(
                _cache,
                likelihood_key,
                result,
                limit=MAX_SCALAR_LIKELIHOOD_CACHE_ENTRIES,
            )
        
        return result
    except (FloatingPointError, OverflowError):
        return -np.inf
    except RuntimeError as exc:
        if "solve failed" in str(exc).lower():
            return -np.inf
        if not bool(config.get("simulation", {}).get("strict_likelihood_errors", True)):
            return -np.inf
        raise
    except Exception:
        if not bool(config.get("simulation", {}).get("strict_likelihood_errors", True)):
            return -np.inf
        raise


def _timeseries_with_reporting_multiplier(
    timeseries: pd.DataFrame,
    base_config: dict[str, Any],
    reporting_multiplier: float,
) -> pd.DataFrame:
    """Recompute reported-case counts for an observation-layer multiplier.

    The ODE uses diagnosis_probability for treatment dynamics; reporting_rate is
    only used in output generation.  This lets the joint sampler integrate over
    reporting multipliers without rerunning the transmission model.
    """
    required = {"age_group", "time", "symptomatic_case_rate_per_day"}
    if not required.issubset(timeseries.columns):
        missing = sorted(required.difference(timeseries.columns))
        raise KeyError(f"Cannot recompute reporting likelihood; missing columns: {missing}")

    out = timeseries.copy()
    base_rates = {
        str(record.get("label")): float(record.get("reporting_rate", 0.0))
        for record in base_config.get("age_groups", [])
    }
    reporting_rate = out["age_group"].astype(str).map(base_rates).astype(float)
    if "diagnostic_reporting_multiplier" in out.columns:
        diagnostic_multiplier = pd.to_numeric(
            out["diagnostic_reporting_multiplier"], errors="coerce"
        ).fillna(1.0).to_numpy(dtype=float)
    else:
        diagnostic_multiplier = np.ones(len(out), dtype=float)
    diagnostic_multiplier = np.where(
        np.isfinite(diagnostic_multiplier) & (diagnostic_multiplier > 0.0),
        diagnostic_multiplier,
        1.0,
    )
    reporting_rate = np.clip(
        reporting_rate.to_numpy(dtype=float)
        * float(reporting_multiplier)
        * diagnostic_multiplier,
        0.0,
        1.0,
    )
    out["reported_case_rate_per_day"] = (
        pd.to_numeric(out["symptomatic_case_rate_per_day"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        * reporting_rate
    )

    group_cols = [
        col
        for col in ("analysis", "scenario", "vaccine_scenario", "resistance_scenario", "intervention", "age_group", "strain")
        if col in out.columns
    ]
    out = out.sort_values(group_cols + ["time"]).copy()
    grouped = out.groupby(group_cols, dropna=False) if group_cols else [((), out)]
    reported_counts = pd.Series(0.0, index=out.index, dtype=float)
    for _, group in grouped:
        ordered = group.sort_values("time")
        dt = ordered["time"].diff().fillna(0.0).to_numpy(dtype=float)
        current = ordered["reported_case_rate_per_day"].to_numpy(dtype=float)
        previous = np.concatenate(([current[0] if len(current) else 0.0], current[:-1]))
        reported_counts.loc[ordered.index] = 0.5 * (previous + current) * dt
    out["reported_cases"] = reported_counts.to_numpy(dtype=float)
    return out


def _log_beta_reporting_prior_with_penalty(
    vector: np.ndarray,
    base_config: dict[str, Any],
    priors: dict[str, Any],
) -> float:
    sample = _sample_from_vector(vector, priors)
    registry_names = ("beta_S", "reporting_multiplier")
    if all(_configured_prior_spec(priors, name) is not None for name in registry_names):
        logp = _registry_log_prior_transformed(vector, sample, priors, registry_names)
        candidate = _apply_sample(base_config, sample)
        return float(logp - reporting_rate_prior_penalty(candidate))
    if not 0.0005 <= sample["beta_S"] <= 0.5:
        return -np.inf
    if not 0.02 <= sample["reporting_multiplier"] <= 20.0:
        return -np.inf

    logp = _normal_logpdf(
        np.log(sample["beta_S"]),
        np.log(float(base_config["transmission"]["beta_S"])),
        float(priors.get("log_beta_S_sd", 0.40)),
    )
    logp += _normal_logpdf(
        np.log(sample["reporting_multiplier"]),
        np.log(float(base_config.get("reporting_multiplier", 1.0))),
        float(priors.get("log_reporting_multiplier_sd", 0.40)),
    )
    candidate = _apply_sample(base_config, sample)
    return float(logp - reporting_rate_prior_penalty(candidate))


def _log_nuisance_prior_transformed(
    vector: np.ndarray,
    base_config: dict[str, Any],
    priors: dict[str, Any],
) -> float:
    """Log prior density for nuisance coordinates on the transformed scale."""
    sample = _sample_from_vector(vector, priors)
    registry_names = (
        "VE_sus",
        "VE_inf",
        "VE_dur",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_symptomatic",
        "infectious_duration_asymptomatic",
        "fitness_R",
    )
    if all(_configured_prior_spec(priors, name) is not None for name in registry_names):
        return _registry_log_prior_transformed(vector, sample, priors, registry_names)
    if not 7.0 <= sample["infectious_duration_symptomatic"] <= 35.0:
        return -np.inf
    if not 5.0 <= sample["infectious_duration_asymptomatic"] <= 28.0:
        return -np.inf
    logp = 0.0
    for key, vector_idx in (
        ("VE_sus", 2),
        ("VE_inf", 3),
        ("VE_dur", 4),
        ("relative_infectiousness_asymptomatic", 5),
    ):
        prior = priors[key]
        logp += _beta_logpdf(sample[key], float(prior["mean"]), float(prior["sd"]))
        logp += _log_jacobian_logit(vector[vector_idx])
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
    fitness_prior = priors["fitness_R"]
    logp += _normal_logpdf(
        sample["fitness_R"],
        float(fitness_prior.get("mean", 1.00)),
        float(fitness_prior.get("sd", 0.12)),
    )
    logp += _log_jacobian_scaled_logit(
        vector[8],
        float(fitness_prior.get("min", 0.70)),
        float(fitness_prior.get("max", 1.25)),
    )
    return float(logp)


def _log_mvn_density(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
        delta = x - mean
        solved = np.linalg.solve(cov, delta)
        return float(
            -0.5 * (len(x) * np.log(2.0 * np.pi) + logdet + float(delta @ solved))
        )
    except np.linalg.LinAlgError:
        return -np.inf


def _log_mixture_density(log_a: float, log_b: float) -> float:
    return float(scipy_special.logsumexp([log_a, log_b]))


def _fit_nuisance_gaussian_proposal(
    candidate_frame: pd.DataFrame,
    log_weights: np.ndarray,
    *,
    prior_coordinate_sd: np.ndarray,
    floor_fraction: float = DEFAULT_IMPORTANCE_PROPOSAL_FLOOR_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.exp(np.asarray(log_weights, dtype=float) - float(np.nanmax(log_weights)))
    weights = weights / float(weights.sum())
    nuisance_weight = (
        pd.DataFrame({"nuisance_index": candidate_frame["nuisance_index"].to_numpy(dtype=int), "weight": weights})
        .groupby("nuisance_index", sort=False)["weight"]
        .sum()
    )
    coord_cols = [f"nuisance_z_{idx}" for idx in range(len(IMPORTANCE_NUISANCE_INDICES))]
    coords = (
        candidate_frame.drop_duplicates("nuisance_index")
        .set_index("nuisance_index")
        .loc[nuisance_weight.index, coord_cols]
        .to_numpy(dtype=float)
    )
    w = nuisance_weight.to_numpy(dtype=float)
    w = w / float(w.sum())
    mean = np.average(coords, axis=0, weights=w)
    centered = coords - mean
    cov = (centered * w[:, None]).T @ centered
    floor = float(np.clip(floor_fraction, 0.005, 0.50))
    floor_sd = np.maximum(floor * np.asarray(prior_coordinate_sd, dtype=float), 0.02)
    cov = cov + np.diag(floor_sd**2)
    cov = 0.5 * (cov + cov.T)
    cov += 1e-6 * np.eye(cov.shape[0])
    return mean, cov


def _prior_nuisance_vectors(
    *,
    n: int,
    seed: int,
    calibrated_start: np.ndarray,
    base_config: dict[str, Any],
    priors: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    unit = _latin_hypercube(n, len(IMPORTANCE_NUISANCE_INDICES), seed=seed)
    vectors = np.vstack(
        [
            _nuisance_vector_from_unit_cube(row, calibrated_start, base_config, priors)
            for row in unit
        ]
    )
    log_prior = np.array(
        [_log_nuisance_prior_transformed(vector, base_config, priors) for vector in vectors],
        dtype=float,
    )
    return vectors, log_prior


def _adaptive_nuisance_vectors(
    *,
    n: int,
    seed: int,
    calibrated_start: np.ndarray,
    base_config: dict[str, Any],
    priors: dict[str, Any],
    proposal_mean: np.ndarray,
    proposal_cov: np.ndarray,
    defensive_prior_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    defensive = float(np.clip(defensive_prior_fraction, 0.01, 0.80))
    prior_vectors, prior_log_density = _prior_nuisance_vectors(
        n=n,
        seed=seed + 17,
        calibrated_start=calibrated_start,
        base_config=base_config,
        priors=priors,
    )
    vectors: list[np.ndarray] = []
    log_prior_values: list[float] = []
    log_q_values: list[float] = []
    try:
        chol = np.linalg.cholesky(proposal_cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(proposal_cov + 1e-4 * np.eye(proposal_cov.shape[0]))

    for idx in range(n):
        if rng.random() < defensive:
            vector = prior_vectors[idx].copy()
            log_prior = float(prior_log_density[idx])
        else:
            vector = calibrated_start.copy()
            vector[IMPORTANCE_NUISANCE_INDICES] = proposal_mean + chol @ rng.standard_normal(len(proposal_mean))
            log_prior = _log_nuisance_prior_transformed(vector, base_config, priors)
        z = vector[IMPORTANCE_NUISANCE_INDICES]
        log_normal = _log_mvn_density(z, proposal_mean, proposal_cov)
        log_q = _log_mixture_density(
            np.log(defensive) + log_prior,
            np.log1p(-defensive) + log_normal,
        )
        vectors.append(vector)
        log_prior_values.append(float(log_prior))
        log_q_values.append(float(log_q))
    return np.vstack(vectors), np.asarray(log_prior_values), np.asarray(log_q_values)


def _nuisance_level_importance_weights(
    candidate_frame: pd.DataFrame,
    log_weights: np.ndarray | None = None,
) -> pd.Series:
    if log_weights is None:
        log_weights = candidate_frame["importance_log_weight"].to_numpy(dtype=float)
    working = pd.DataFrame(
        {
            "nuisance_index": candidate_frame["nuisance_index"].to_numpy(dtype=int),
            "log_weight": np.asarray(log_weights, dtype=float),
        }
    )
    working = working.loc[np.isfinite(working["log_weight"].to_numpy(dtype=float))]
    if working.empty:
        raise RuntimeError("No finite nuisance-level importance weights")
    grouped = working.groupby("nuisance_index", sort=False)["log_weight"].agg(
        lambda values: float(scipy_special.logsumexp(values.to_numpy(dtype=float)))
    )
    log_total = float(scipy_special.logsumexp(grouped.to_numpy(dtype=float)))
    return np.exp(grouped - log_total)


def _build_nuisance_mixture_components(
    candidate_frame: pd.DataFrame,
    log_weights: np.ndarray,
    *,
    prior_coordinate_sd: np.ndarray,
    max_centers: int = 8,
) -> list[dict[str, np.ndarray | float]]:
    coord_cols = [f"nuisance_z_{idx}" for idx in range(len(IMPORTANCE_NUISANCE_INDICES))]
    unique_coords = candidate_frame.drop_duplicates("nuisance_index").set_index("nuisance_index")
    nuisance_weights = _nuisance_level_importance_weights(candidate_frame, log_weights)
    nuisance_weights = nuisance_weights.loc[nuisance_weights.index.intersection(unique_coords.index)]
    if nuisance_weights.empty:
        raise RuntimeError("No finite nuisance proposal centers")
    nuisance_weights = nuisance_weights.sort_values(ascending=False)

    prior_sd = np.asarray(prior_coordinate_sd, dtype=float)
    prior_sd = np.where(np.isfinite(prior_sd) & (prior_sd > 0.0), prior_sd, 1.0)
    global_mean, global_cov = _fit_nuisance_gaussian_proposal(
        candidate_frame,
        log_weights,
        prior_coordinate_sd=prior_sd,
        floor_fraction=DEFAULT_IMPORTANCE_PROPOSAL_FLOOR_FRACTION,
    )
    components: list[dict[str, np.ndarray | float]] = [
        {"weight": 0.25, "mean": global_mean, "cov": global_cov}
    ]

    top = nuisance_weights.head(max(1, int(max_centers)))
    center_weights = np.sqrt(top.to_numpy(dtype=float))
    center_weights = center_weights / float(center_weights.sum())
    scale_weights = np.array([0.50, 0.30, 0.20], dtype=float)
    scales = np.array([0.025, 0.075, 0.18], dtype=float)
    local_total = 0.75
    for nuisance_index, center_weight in zip(top.index, center_weights):
        center = unique_coords.loc[nuisance_index, coord_cols].to_numpy(dtype=float)
        for scale, scale_weight in zip(scales, scale_weights):
            sd = np.maximum(scale * prior_sd, 0.015)
            components.append(
                {
                    "weight": float(local_total * center_weight * scale_weight),
                    "mean": center.copy(),
                    "cov": np.diag(sd**2),
                }
            )

    total = float(sum(float(component["weight"]) for component in components))
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("Invalid nuisance proposal mixture weights")
    for component in components:
        component["weight"] = float(component["weight"]) / total
    return components


def _log_nuisance_mixture_density(
    z: np.ndarray,
    log_prior: float,
    gaussian_components: list[dict[str, np.ndarray | float]],
    *,
    defensive_prior_fraction: float,
) -> float:
    defensive = float(np.clip(defensive_prior_fraction, 0.01, 0.80))
    terms: list[float] = []
    if isfinite(log_prior):
        terms.append(float(np.log(defensive) + log_prior))
    gaussian_mass = 1.0 - defensive
    for component in gaussian_components:
        weight = float(component["weight"])
        if weight <= 0.0:
            continue
        log_density = _log_mvn_density(
            z,
            np.asarray(component["mean"], dtype=float),
            np.asarray(component["cov"], dtype=float),
        )
        if isfinite(log_density):
            terms.append(float(np.log(gaussian_mass * weight) + log_density))
    if not terms:
        return -np.inf
    return float(scipy_special.logsumexp(terms))


def _mixture_nuisance_vectors(
    *,
    n: int,
    seed: int,
    calibrated_start: np.ndarray,
    base_config: dict[str, Any],
    priors: dict[str, Any],
    gaussian_components: list[dict[str, np.ndarray | float]],
    defensive_prior_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    defensive = float(np.clip(defensive_prior_fraction, 0.01, 0.80))
    prior_vectors, prior_log_density = _prior_nuisance_vectors(
        n=n,
        seed=seed + 17,
        calibrated_start=calibrated_start,
        base_config=base_config,
        priors=priors,
    )
    component_weights = np.array([float(component["weight"]) for component in gaussian_components], dtype=float)
    component_weights = component_weights / float(component_weights.sum())
    component_cdf = np.cumsum(component_weights)
    component_cdf[-1] = 1.0

    vectors: list[np.ndarray] = []
    log_prior_values: list[float] = []
    log_q_values: list[float] = []
    for idx in range(int(n)):
        if rng.random() < defensive:
            vector = prior_vectors[idx].copy()
            log_prior = float(prior_log_density[idx])
        else:
            component_idx = int(np.searchsorted(component_cdf, rng.random(), side="left"))
            component = gaussian_components[component_idx]
            cov = np.asarray(component["cov"], dtype=float)
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                chol = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
            vector = calibrated_start.copy()
            vector[IMPORTANCE_NUISANCE_INDICES] = (
                np.asarray(component["mean"], dtype=float)
                + chol @ rng.standard_normal(len(IMPORTANCE_NUISANCE_INDICES))
            )
            log_prior = _log_nuisance_prior_transformed(vector, base_config, priors)
        z = vector[IMPORTANCE_NUISANCE_INDICES]
        log_q = _log_nuisance_mixture_density(
            z,
            float(log_prior),
            gaussian_components,
            defensive_prior_fraction=defensive,
        )
        vectors.append(vector)
        log_prior_values.append(float(log_prior))
        log_q_values.append(float(log_q))
    return np.vstack(vectors), np.asarray(log_prior_values), np.asarray(log_q_values)


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
            f"active sampled parameters are {active_names}. Fix reporting, VE, asymptomatic infectiousness, "
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


def _stratified_resample_indices(
    weights: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0 or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError("Cannot resample from empty or invalid importance weights")
    weights = weights / float(weights.sum())
    positions = (np.arange(int(n), dtype=float) + rng.random(int(n))) / float(n)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="left").astype(int)


def _normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(log_weights)
    weights = np.zeros(len(log_weights), dtype=float)
    if not finite.any():
        raise RuntimeError("No finite SMC weights")
    log_total = float(scipy_special.logsumexp(log_weights[finite]))
    weights[finite] = np.exp(log_weights[finite] - log_total)
    weights = np.clip(weights, 0.0, np.inf)
    total = float(weights.sum())
    if not isfinite(total) or total <= 0.0:
        raise RuntimeError("SMC weights normalised to zero")
    return weights / total


def _effective_sample_size_from_weights(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0 or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        return 0.0
    weights = weights / float(weights.sum())
    return float(1.0 / np.sum(weights**2))


def _temperature_weight_ess(
    weights: np.ndarray,
    log_likelihood: np.ndarray,
    current_temperature: float,
    candidate_temperature: float,
) -> tuple[float, np.ndarray]:
    delta = float(candidate_temperature) - float(current_temperature)
    if delta < -1e-12:
        raise ValueError("SMC candidate temperature must be non-decreasing")
    log_weights = np.log(np.clip(weights, 1e-300, 1.0)) + delta * np.asarray(log_likelihood, dtype=float)
    updated = _normalise_log_weights(log_weights)
    return _effective_sample_size_from_weights(updated), updated


def _find_next_smc_temperature(
    weights: np.ndarray,
    log_likelihood: np.ndarray,
    current_temperature: float,
    target_ess: float,
) -> tuple[float, np.ndarray, float]:
    current_temperature = float(np.clip(current_temperature, 0.0, 1.0))
    if current_temperature >= 1.0:
        ess = _effective_sample_size_from_weights(weights)
        return 1.0, np.asarray(weights, dtype=float), ess

    try:
        full_ess, full_weights = _temperature_weight_ess(
            weights,
            log_likelihood,
            current_temperature,
            1.0,
        )
    except RuntimeError:
        full_ess = 0.0
        full_weights = np.asarray(weights, dtype=float)
    if full_ess >= float(target_ess):
        return 1.0, full_weights, float(full_ess)

    low = current_temperature
    high = 1.0
    best_temperature = low
    best_weights = np.asarray(weights, dtype=float)
    best_ess = _effective_sample_size_from_weights(best_weights)
    for _ in range(40):
        mid = 0.5 * (low + high)
        try:
            ess, mid_weights = _temperature_weight_ess(weights, log_likelihood, current_temperature, mid)
        except RuntimeError:
            ess = 0.0
            mid_weights = best_weights
        if ess >= float(target_ess):
            best_temperature = mid
            best_weights = mid_weights
            best_ess = float(ess)
            low = mid
        else:
            high = mid
    if best_temperature <= current_temperature + 1e-10:
        # Keep the run moving, but downstream diagnostics will fail if the
        # resulting weight concentration is not scientifically usable.
        best_temperature = min(1.0, current_temperature + 1e-3)
        best_ess, best_weights = _temperature_weight_ess(
            weights,
            log_likelihood,
            current_temperature,
            best_temperature,
        )
    return float(best_temperature), best_weights, float(best_ess)


def _evaluate_smc_chunk(payload: dict[str, Any]) -> list[tuple[int, float, float, float]]:
    configure_worker_thread_limits()
    vectors = np.asarray(payload["vectors"], dtype=float)
    indices = np.asarray(payload["indices"], dtype=int)
    runtime_base = payload["runtime_base"]
    observed = payload["observed"]
    settings = payload["settings"]
    country = str(payload["country"])
    dispersion = float(settings.get("dispersion", 50.0))
    cache: dict[str, float] = {}
    out: list[tuple[int, float, float, float]] = []
    for idx, vector in zip(indices, vectors):
        log_prior = _log_prior(vector, runtime_base, country, settings)
        if not isfinite(log_prior):
            out.append((int(idx), -np.inf, -np.inf, -np.inf))
            continue
        try:
            sample = _sample_from_vector(vector, settings["priors"])
            candidate = _apply_sample(runtime_base, sample)
            penalty = float(reporting_rate_prior_penalty(candidate))
        except Exception:
            out.append((int(idx), float(log_prior), -np.inf, -np.inf))
            continue
        log_base = float(log_prior - penalty)
        if not isfinite(log_base):
            out.append((int(idx), float(log_prior), -np.inf, -np.inf))
            continue
        log_likelihood = _log_likelihood(
            candidate,
            observed,
            country,
            dispersion,
            _cache=cache,
        )
        out.append((int(idx), float(log_prior), log_base, float(log_likelihood)))
    return out


def _evaluate_smc_vectors(
    vectors: np.ndarray,
    *,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    settings: dict[str, Any],
    country: str,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    n = int(len(vectors))
    log_prior = np.full(n, -np.inf, dtype=float)
    log_base = np.full(n, -np.inf, dtype=float)
    log_likelihood = np.full(n, -np.inf, dtype=float)
    if n == 0:
        return log_prior, log_base, log_likelihood
    workers = min(max(1, int(n_jobs)), available_cpus(), n)
    chunks = [
        chunk
        for chunk in np.array_split(np.arange(n, dtype=int), workers)
        if len(chunk) > 0
    ]
    payloads = [
        {
            "indices": chunk,
            "vectors": vectors[chunk],
            "runtime_base": runtime_base,
            "observed": observed,
            "settings": settings,
            "country": country,
        }
        for chunk in chunks
    ]
    if workers <= 1 or len(payloads) <= 1:
        chunk_results = [_evaluate_smc_chunk(payload) for payload in payloads]
    else:
        chunk_results = Parallel(n_jobs=len(payloads), backend="loky", inner_max_num_threads=1)(
            delayed(_evaluate_smc_chunk)(payload) for payload in payloads
        )
    for chunk_result in chunk_results:
        for idx, prior, base, likelihood in chunk_result:
            log_prior[idx] = float(prior)
            log_base[idx] = float(base)
            log_likelihood[idx] = float(likelihood)
    return log_prior, log_base, log_likelihood


def _smc_target_log_prob(
    log_base: np.ndarray,
    log_likelihood: np.ndarray,
    temperature: float,
) -> np.ndarray:
    return np.asarray(log_base, dtype=float) + float(temperature) * np.asarray(log_likelihood, dtype=float)


def _log_weighted_increment(weights: np.ndarray, log_increment: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    log_increment = np.asarray(log_increment, dtype=float)
    finite = np.isfinite(weights) & (weights > 0.0) & np.isfinite(log_increment)
    if not finite.any():
        return -np.inf
    return float(scipy_special.logsumexp(np.log(weights[finite]) + log_increment[finite]))


def _reflect_vector_into_bounds(vector: np.ndarray, priors: dict[str, Any]) -> np.ndarray:
    out = np.asarray(vector, dtype=float).copy()
    # The beta/reporting transformed bounds depend on each other, so two passes
    # avoid leaving either coordinate outside the coupled admissible region.
    for _ in range(2):
        for component in range(N_PARAMS):
            out = _reflect_component_into_bounds(out, component, priors)
    return out


def _smc_particle_covariance(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    diagonal_floor = np.square(np.maximum(INITIAL_PROPOSAL_SCALES, 1e-3))
    if len(vectors) <= 1:
        return np.diag(diagonal_floor)
    centered = vectors - np.mean(vectors, axis=0)
    cov = centered.T @ centered / float(max(len(vectors) - 1, 1))
    cov = 0.85 * cov + 0.15 * np.diag(diagonal_floor)
    cov = 0.5 * (cov + cov.T)
    cov += np.diag(0.05 * diagonal_floor) + AM_EPSILON * np.eye(N_PARAMS)
    return cov


def _draw_smc_random_walk_proposals(
    current_vectors: np.ndarray,
    *,
    covariance: np.ndarray,
    move_scale: float,
    rng: np.random.Generator,
    priors: dict[str, Any],
) -> np.ndarray:
    current_vectors = np.asarray(current_vectors, dtype=float)
    proposals = current_vectors.copy()
    n = int(len(current_vectors))
    if n == 0:
        return proposals

    covariance = np.asarray(covariance, dtype=float)
    diagonal_scales = np.sqrt(np.maximum(np.diag(covariance), np.square(INITIAL_PROPOSAL_SCALES)))
    scale = max(float(move_scale), 1e-6)
    active_all = np.arange(N_PARAMS, dtype=int)

    for row_idx in range(n):
        move = float(rng.random())
        proposal = current_vectors[row_idx].copy()
        if move < SMC_LOCAL_PROPOSAL_PROBABILITY:
            component = int(rng.integers(0, N_PARAMS))
            proposal[component] += rng.normal(0.0, diagonal_scales[component] * scale)
        elif move < SMC_LOCAL_PROPOSAL_PROBABILITY + SMC_BLOCK_PROPOSAL_PROBABILITY:
            block = np.asarray(PROPOSAL_BLOCKS[int(rng.integers(0, len(PROPOSAL_BLOCKS)))], dtype=int)
            block_cov = covariance[np.ix_(block, block)]
            block_scale = scale * np.sqrt((2.38**2) / max(len(block), 1))
            try:
                chol = np.linalg.cholesky(block_cov)
                proposal[block] += block_scale * (chol @ rng.standard_normal(len(block)))
            except np.linalg.LinAlgError:
                proposal[block] += rng.normal(0.0, diagonal_scales[block] * block_scale)
        else:
            full_scale = scale * np.sqrt(AM_SD)
            try:
                chol = np.linalg.cholesky(covariance)
                proposal[active_all] += full_scale * (chol @ rng.standard_normal(N_PARAMS))
            except np.linalg.LinAlgError:
                proposal[active_all] += rng.normal(0.0, diagonal_scales * full_scale)
        proposals[row_idx] = _reflect_vector_into_bounds(proposal, priors)
    return proposals


def _run_smc_move_steps(
    *,
    vectors: np.ndarray,
    log_prior: np.ndarray,
    log_base: np.ndarray,
    log_likelihood: np.ndarray,
    temperature: float,
    move_steps: int,
    move_scale: float,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    settings: dict[str, Any],
    country: str,
    n_jobs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if move_steps <= 0 or len(vectors) == 0:
        return vectors, log_prior, log_base, log_likelihood, 0.0

    accepted_total = 0
    proposed_total = 0
    current_target = _smc_target_log_prob(log_base, log_likelihood, temperature)
    current_vectors = np.asarray(vectors, dtype=float).copy()
    current_prior = np.asarray(log_prior, dtype=float).copy()
    current_base = np.asarray(log_base, dtype=float).copy()
    current_likelihood = np.asarray(log_likelihood, dtype=float).copy()
    priors = settings["priors"]

    for _ in range(int(move_steps)):
        covariance = _smc_particle_covariance(current_vectors)
        proposals = _draw_smc_random_walk_proposals(
            current_vectors,
            covariance=covariance,
            move_scale=move_scale,
            rng=rng,
            priors=priors,
        )
        proposal_prior, proposal_base, proposal_likelihood = _evaluate_smc_vectors(
            proposals,
            runtime_base=runtime_base,
            observed=observed,
            settings=settings,
            country=country,
            n_jobs=n_jobs,
        )
        proposal_target = _smc_target_log_prob(proposal_base, proposal_likelihood, temperature)
        log_alpha = proposal_target - current_target
        accept = np.isfinite(log_alpha) & (np.log(rng.random(len(log_alpha))) < log_alpha)
        if accept.any():
            current_vectors[accept] = proposals[accept]
            current_prior[accept] = proposal_prior[accept]
            current_base[accept] = proposal_base[accept]
            current_likelihood[accept] = proposal_likelihood[accept]
            current_target[accept] = proposal_target[accept]
        accepted_total += int(accept.sum())
        proposed_total += int(len(accept))

    acceptance_fraction = accepted_total / max(proposed_total, 1)
    return current_vectors, current_prior, current_base, current_likelihood, float(acceptance_fraction)


def _run_smc(
    task: ChainTask,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    settings: dict[str, Any],
    calibrated_start: np.ndarray,
) -> pd.DataFrame:
    n_particles = int(max(task.smc_particles, 8))
    draws = int(max(task.draws, 1))
    target_ess_fraction = float(np.clip(task.smc_ess_fraction, 0.10, 0.95))
    target_ess = target_ess_fraction * n_particles
    move_steps = int(max(task.smc_move_steps, 0))
    max_stages = int(max(task.smc_max_stages, 1))
    workers = int(max(task.grid_eval_jobs, 1))
    rng = np.random.default_rng(task.seed)

    progress_dir = _mcmc_progress_dir(task.output_stem)
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"{task.country}_chain{task.chain:02d}_smc.txt"
    try:
        progress_file.write_text(
            f"starting particles={n_particles} target_ess_fraction={target_ess_fraction:.3f}"
        )
    except OSError:
        pass

    unit = _latin_hypercube(n_particles, N_PARAMS, seed=task.seed + 104729)
    vectors = np.vstack(
        [
            _full_vector_from_unit_cube(row, calibrated_start, runtime_base, settings["priors"])
            for row in unit
        ]
    )
    log_prior, log_base, log_likelihood = _evaluate_smc_vectors(
        vectors,
        runtime_base=runtime_base,
        observed=observed,
        settings=settings,
        country=task.country,
        n_jobs=workers,
    )
    finite = np.isfinite(log_prior) & np.isfinite(log_base) & np.isfinite(log_likelihood)
    if not finite.any():
        raise RuntimeError(f"SMC initial prior draw found no finite posterior support for {task.country}")

    # The initial proposal is the configured prior truncated to hard bounds.
    # Its density differs from _log_prior only by constants, so the initial
    # correction is the reporting-rate prior penalty already included in log_base.
    initial_log_correction = log_base - log_prior
    weights = _normalise_log_weights(initial_log_correction)
    log_evidence = float(scipy_special.logsumexp(initial_log_correction) - np.log(n_particles))
    temperature = 0.0
    ancestor_ids = np.arange(n_particles, dtype=int)
    move_scale = max(float(task.proposal_scale), 0.50)
    stage_records: list[dict[str, float | int | bool]] = []
    acceptance_values: list[float] = []

    for stage in range(1, max_stages + 1):
        previous_temperature = float(temperature)
        previous_weights = weights.copy()
        new_temperature, weights, ess = _find_next_smc_temperature(
            weights,
            log_likelihood,
            temperature,
            target_ess,
        )
        temperature_increment = float(new_temperature - previous_temperature)
        log_evidence += _log_weighted_increment(
            previous_weights,
            temperature_increment * np.asarray(log_likelihood, dtype=float),
        )
        max_weight = float(np.max(weights)) if len(weights) else 1.0
        stage_records.append(
            {
                "stage": int(stage),
                "temperature": float(new_temperature),
                "temperature_increment": temperature_increment,
                "ess": float(ess),
                "ess_fraction": float(ess / n_particles),
                "max_weight": max_weight,
                "move_scale": float(move_scale),
                "log_evidence": float(log_evidence),
            }
        )
        temperature = float(new_temperature)
        try:
            progress_file.write_text(
                f"stage={stage} temperature={temperature:.6f} ess={ess:.1f}/{n_particles} "
                f"max_weight={max_weight:.4f} move_scale={move_scale:.3f} logZ={log_evidence:.2f}"
            )
        except OSError:
            pass

        if ess < 0.98 * n_particles and temperature < 1.0 + 1e-12:
            indices = _stratified_resample_indices(weights, n_particles, rng)
            vectors = vectors[indices].copy()
            log_prior = log_prior[indices].copy()
            log_base = log_base[indices].copy()
            log_likelihood = log_likelihood[indices].copy()
            ancestor_ids = ancestor_ids[indices].copy()
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

        if move_steps > 0:
            vectors, log_prior, log_base, log_likelihood, acceptance = _run_smc_move_steps(
                vectors=vectors,
                log_prior=log_prior,
                log_base=log_base,
                log_likelihood=log_likelihood,
                temperature=temperature,
                move_steps=move_steps,
                move_scale=move_scale,
                runtime_base=runtime_base,
                observed=observed,
                settings=settings,
                country=task.country,
                n_jobs=workers,
                rng=rng,
            )
            acceptance_values.append(float(acceptance))
            if acceptance < SMC_LOW_ACCEPTANCE:
                move_scale = max(0.05, move_scale * 0.70)
            elif acceptance > SMC_HIGH_ACCEPTANCE:
                move_scale = min(8.0, move_scale * 1.25)
            stage_records[-1]["move_acceptance_fraction"] = float(acceptance)
            stage_records[-1]["move_scale_after"] = float(move_scale)

        if temperature >= 1.0 - 1e-10:
            temperature = 1.0
            break

    if temperature < 1.0 - 1e-10:
        try:
            progress_file.write_text(
                f"stopped_before_final_temperature temperature={temperature:.6f} stages={len(stage_records)}"
            )
        except OSError:
            pass

    # Refresh final weights at the achieved temperature after the last move.
    final_log_weight = np.log(np.clip(weights, 1e-300, 1.0)) + (
        1.0 - float(stage_records[-1]["temperature"]) if stage_records else 0.0
    ) * log_likelihood
    if temperature >= 1.0 - 1e-10:
        final_weights = np.asarray(weights, dtype=float)
    else:
        final_weights = _normalise_log_weights(final_log_weight)
    final_ess = _effective_sample_size_from_weights(final_weights)
    final_max_weight = float(np.max(final_weights)) if len(final_weights) else 1.0
    stage_ess_values = [float(record["ess"]) for record in stage_records] or [final_ess]
    stage_weight_values = [float(record["max_weight"]) for record in stage_records] or [final_max_weight]
    min_stage_ess = float(min(stage_ess_values))
    min_stage_ess_fraction = float(min_stage_ess / n_particles)
    max_stage_weight = float(max(stage_weight_values + [final_max_weight]))
    unique_ancestors = int(len(np.unique(ancestor_ids)))
    unique_ancestor_fraction = float(unique_ancestors / n_particles)
    unique_particles = int(len(np.unique(np.round(vectors, decimals=10), axis=0)))
    unique_particle_fraction = float(unique_particles / n_particles)
    reached_final = bool(temperature >= 1.0 - 1e-10)
    mean_acceptance = float(np.mean(acceptance_values)) if acceptance_values else 0.0

    rows: list[dict[str, Any]] = []
    posterior_log_prob = _smc_target_log_prob(log_base, log_likelihood, 1.0)
    for draw_idx, particle_idx in enumerate(range(n_particles), start=1):
        sample = _sample_from_vector(vectors[int(particle_idx)], settings["priors"])
        row: dict[str, Any] = {
            "country": task.country,
            "chain": int(task.chain),
            "draw": int(draw_idx),
            "step": int(draw_idx),
            "posterior_log_prob": float(posterior_log_prob[int(particle_idx)]),
            "log_prior": float(log_prior[int(particle_idx)]),
            "log_base_prior_with_reporting_penalty": float(log_base[int(particle_idx)]),
            "log_likelihood": float(log_likelihood[int(particle_idx)]),
            "accepted_fraction": mean_acceptance,
            "sampling_method": SMC_SAMPLER,
            "source_particle_index": int(particle_idx),
            "source_normalized_weight": float(final_weights[int(particle_idx)]),
            "source_initial_ancestor": int(ancestor_ids[int(particle_idx)]),
            "smc_source_chain": int(task.chain),
            "smc_log_evidence": float(log_evidence),
            "smc_particle_count": int(n_particles),
            "smc_stage_count": int(len(stage_records)),
            "smc_final_temperature": float(temperature),
            "smc_reached_final_temperature": reached_final,
            "smc_target_ess_fraction": float(target_ess_fraction),
            "smc_min_ess": min_stage_ess,
            "smc_min_ess_fraction": min_stage_ess_fraction,
            "smc_final_ess": float(final_ess),
            "smc_final_ess_fraction": float(final_ess / n_particles),
            "smc_final_max_weight": final_max_weight,
            "smc_max_weight": max_stage_weight,
            "smc_unique_initial_ancestors": int(unique_ancestors),
            "smc_unique_ancestor_fraction": unique_ancestor_fraction,
            "smc_unique_particles": int(unique_particles),
            "smc_unique_particle_fraction": unique_particle_fraction,
            "smc_move_steps": int(move_steps),
            "smc_move_acceptance_fraction": mean_acceptance,
            "smc_move_scale_final": float(move_scale),
            "resistance_prevalence": float(settings["priors"].get("resistance_prevalence_fixed", np.nan)),
            "reporting_trend_end_multiplier": float(settings["priors"].get("reporting_trend_fixed", 1.0)),
        }
        row.update(sample)
        rows.append(row)

    stage_path = project_path(
        "outputs/metadata",
        f"{_artifact_stem(task.output_stem, 'bayesian_smc_stage_history', 'smc_stage_history')}_{task.country}_chain{task.chain:02d}.csv",
    )
    try:
        write_dataframe(pd.DataFrame(stage_records), stage_path)
    except Exception:
        pass

    try:
        progress_file.write_text(
            f"done temperature={temperature:.6f} stages={len(stage_records)} final_ess={final_ess:.1f} "
            f"max_weight={max_stage_weight:.4f} unique_ancestors={unique_ancestors}/{n_particles} "
            f"unique_particles={unique_particles}/{n_particles} acceptance={mean_acceptance:.3f} "
            f"logZ={log_evidence:.2f}"
        )
    except OSError:
        pass
    return pd.DataFrame(rows)


def _resample_smc_island_particles(
    samples: pd.DataFrame,
    *,
    chain_count: int,
    draws_per_chain: int,
    base_seed: int,
) -> pd.DataFrame:
    """Combine same-target SMC islands with equal island mass.

    Every island starts from the same configured prior and targets the same
    country posterior.  Its noisy marginal-likelihood estimate is therefore a
    diagnostic, not a finite-sample reason to give one replicate nearly all of
    the posterior mass.  We retain each island as an independent diagnostic
    chain and resample ``draws_per_chain`` rows using only its normalized final
    particle weights.  The resulting pooled empirical measure assigns equal
    mass to every island.
    """
    required = {"country", "chain", "source_normalized_weight", "smc_log_evidence"}
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise KeyError(f"SMC island resampling requires columns: {', '.join(missing)}")
    chain_count = int(max(chain_count, 1))
    draws_per_chain = int(max(draws_per_chain, 1))
    frames: list[pd.DataFrame] = []

    for country_idx, (country, group) in enumerate(samples.groupby("country", sort=False)):
        group = group.reset_index(drop=True).copy()
        source_chain_col = "smc_source_chain" if "smc_source_chain" in group.columns else "chain"
        source_chains = pd.to_numeric(group[source_chain_col], errors="coerce").astype(int)
        group["smc_source_chain"] = source_chains
        chain_logz = (
            group[["smc_source_chain", "smc_log_evidence"]]
            .drop_duplicates("smc_source_chain")
            .sort_values("smc_source_chain")
        )
        island_logz = pd.to_numeric(chain_logz["smc_log_evidence"], errors="coerce").to_numpy(dtype=float)
        finite_islands = np.isfinite(island_logz)
        if not finite_islands.all():
            bad = chain_logz.loc[~finite_islands, "smc_source_chain"].astype(int).tolist()
            raise RuntimeError(f"Non-finite SMC island evidence estimates for {country}: {bad}")
        island_count = int(len(island_logz))
        if island_count != chain_count:
            raise RuntimeError(
                f"Expected {chain_count} SMC islands for {country}, found {island_count}"
            )
        island_weights_arr = np.full(island_count, 1.0 / island_count, dtype=float)
        island_weight_by_chain = {
            int(chain): float(weight)
            for chain, weight in zip(chain_logz["smc_source_chain"].to_numpy(dtype=int), island_weights_arr)
        }
        island_ess = _effective_sample_size_from_weights(island_weights_arr)
        max_island_weight = float(np.max(island_weights_arr)) if len(island_weights_arr) else 1.0
        log_evidence_sd = float(np.std(island_logz, ddof=1)) if island_count > 1 else 0.0
        log_evidence_range = float(np.max(island_logz) - np.min(island_logz)) if island_count else np.nan

        particle_base = pd.to_numeric(group["source_normalized_weight"], errors="coerce").to_numpy(dtype=float)
        combined_weights = np.zeros(len(group), dtype=float)
        selected_frames: list[pd.DataFrame] = []
        selected_source_indices: list[np.ndarray] = []
        rng = np.random.default_rng(int(base_seed) + 17011 + 1009 * int(country_idx))
        for output_chain, source_chain in enumerate(
            chain_logz["smc_source_chain"].to_numpy(dtype=int),
            start=1,
        ):
            source_indices = np.flatnonzero(source_chains.to_numpy(dtype=int) == int(source_chain))
            if len(source_indices) == 0:
                raise RuntimeError(f"SMC island {source_chain} for {country} contains no particles")
            local_weights = np.asarray(particle_base[source_indices], dtype=float)
            if not np.isfinite(local_weights).all() or np.any(local_weights < 0.0):
                raise RuntimeError(f"SMC island {source_chain} for {country} has invalid particle weights")
            local_total = float(local_weights.sum())
            if local_total <= 0.0:
                raise RuntimeError(f"SMC island {source_chain} for {country} has zero particle mass")
            local_weights /= local_total
            combined_weights[source_indices] = local_weights / island_count
            local_selected = _stratified_resample_indices(local_weights, draws_per_chain, rng)
            # Stratified resampling returns monotone particle indices.  Particle
            # row order is an implementation detail, not a time ordering; a
            # random permutation prevents that artificial ordering (and runs
            # of duplicate indices) from contaminating rank/autocorrelation
            # diagnostics that treat each island as an exchangeable chain.
            local_selected = rng.permutation(local_selected)
            global_selected = source_indices[local_selected]
            selected_source_indices.append(global_selected)
            selected_chain = group.iloc[global_selected].reset_index(drop=True).copy()
            selected_chain["chain"] = int(output_chain)
            selected_chain["draw"] = np.arange(1, draws_per_chain + 1, dtype=int)
            selected_chain["step"] = selected_chain["draw"]
            selected_frames.append(selected_chain)
        combined_weights /= float(combined_weights.sum())
        combined_ess = _effective_sample_size_from_weights(combined_weights)
        combined_max_weight = float(np.max(combined_weights)) if len(combined_weights) else 1.0

        selected_indices = np.concatenate(selected_source_indices)
        selected = pd.concat(selected_frames, ignore_index=True)
        selected["smc_island_count"] = int(len(island_weights_arr))
        selected["smc_island_ess"] = float(island_ess)
        selected["smc_island_ess_fraction"] = float(island_ess / max(len(island_weights_arr), 1))
        selected["smc_max_island_weight"] = max_island_weight
        selected["smc_log_evidence_sd"] = log_evidence_sd
        selected["smc_log_evidence_range"] = log_evidence_range
        selected["smc_island_weight"] = [
            island_weight_by_chain.get(int(chain), np.nan)
            for chain in selected["smc_source_chain"].to_numpy(dtype=int)
        ]
        selected["smc_combined_particle_count"] = int(len(group))
        selected["smc_combined_particle_ess"] = float(combined_ess)
        selected["smc_combined_particle_ess_fraction"] = float(combined_ess / max(len(group), 1))
        selected["smc_combined_max_weight"] = combined_max_weight
        selected["smc_combined_source_weight"] = combined_weights[selected_indices]
        aggregate_rules = {
            "smc_particle_count": "max",
            "smc_stage_count": "max",
            "smc_final_temperature": "min",
            "smc_min_ess": "min",
            "smc_min_ess_fraction": "min",
            "smc_final_ess": "min",
            "smc_final_ess_fraction": "min",
            "smc_final_max_weight": "max",
            "smc_max_weight": "max",
            "smc_unique_ancestor_fraction": "min",
            "smc_unique_particle_fraction": "min",
            "smc_move_acceptance_fraction": "mean",
        }
        for column, rule in aggregate_rules.items():
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            if rule == "max":
                value = float(values.max())
            elif rule == "mean":
                value = float(values.mean())
            else:
                value = float(values.min())
            selected[column] = value
        if "smc_reached_final_temperature" in group.columns:
            selected["smc_reached_final_temperature"] = bool(group["smc_reached_final_temperature"].astype(bool).all())
        selected["sampling_method"] = SMC_SAMPLER
        frames.append(selected)

    return pd.concat(frames, ignore_index=True)


def _evaluate_joint_importance_chunk(payload: dict[str, Any]) -> list[dict[str, Any]]:
    configure_worker_thread_limits()
    country = str(payload["country"])
    runtime_base = payload["runtime_base"]
    observed = payload["observed"]
    settings = payload["settings"]
    nuisance_vectors = np.asarray(payload["nuisance_vectors"], dtype=float)
    nuisance_log_prior = np.asarray(payload["nuisance_log_prior"], dtype=float)
    nuisance_log_q = np.asarray(payload["nuisance_log_q"], dtype=float)
    nuisance_indices = np.asarray(payload["nuisance_indices"], dtype=int)
    beta_grid = np.asarray(payload["beta_grid"], dtype=float)
    reporting_grid = np.asarray(payload["reporting_grid"], dtype=float)
    priors = settings["priors"]
    dispersion = float(settings.get("dispersion", 50.0))
    records: list[dict[str, Any]] = []

    for row_pos, (local_idx, nuisance_vector) in enumerate(zip(nuisance_indices, nuisance_vectors)):
        log_nuisance_prior = float(nuisance_log_prior[row_pos])
        log_nuisance_q = float(nuisance_log_q[row_pos])
        if not isfinite(log_nuisance_prior) or not isfinite(log_nuisance_q):
            continue
        for beta_idx, log_beta in enumerate(beta_grid):
            ode_vector = nuisance_vector.copy()
            ode_vector[0] = float(log_beta)
            ode_vector[1] = float(reporting_grid[len(reporting_grid) // 2])
            ode_sample = _sample_from_vector(ode_vector, priors)
            ode_config = _apply_sample(runtime_base, ode_sample)
            try:
                exposure, exposure_reporting_rates = run_prepared_case_exposure(
                    ode_config,
                    observed,
                    analysis="bayesian_joint_importance",
                    scenario=f"{country}_nuisance_{int(local_idx):04d}_beta_{beta_idx:03d}",
                    vaccine_scenario=ode_config["baseline_vaccine_scenario"],
                    resistance_scenario=ode_config["baseline_resistance_scenario"],
                    metadata={"country": country},
                )
                observed_values = pd.to_numeric(
                    observed["reported_cases"], errors="coerce"
                ).to_numpy(dtype=float)
                reporting_samples = [
                    _sample_from_vector(
                        np.concatenate(
                            (
                                ode_vector[:1],
                                np.array([float(reporting_coordinate)]),
                                ode_vector[2:],
                            )
                        ),
                        priors,
                    )["reporting_multiplier"]
                    for reporting_coordinate in reporting_grid
                ]
                projected_grid = project_reporting_grid(
                    exposure,
                    exposure_reporting_rates,
                    np.asarray(reporting_samples, dtype=float),
                )
                predicted_grid = np.sum(projected_grid, axis=2)
            except (FloatingPointError, OverflowError):
                # Explicit numerical failures become missing grid cells and are
                # rejected by the per-nuisance completeness gate below.
                continue
            except RuntimeError as exc:
                if "solve failed" in str(exc).lower():
                    continue
                raise

            for reporting_idx, reporting_coordinate in enumerate(reporting_grid):
                vector = ode_vector.copy()
                vector[1] = float(reporting_coordinate)
                sample = _sample_from_vector(vector, priors)
                prior = _log_beta_reporting_prior_with_penalty(vector, runtime_base, priors)
                if len(observed_values) < 3 or predicted_grid.shape[1] != len(observed_values):
                    likelihood = -np.inf
                else:
                    likelihood = float(
                        -negative_binomial_nll(
                            observed_values,
                            predicted_grid[reporting_idx],
                            dispersion=dispersion,
                        )
                    )
                # Retain zero-mass cells explicitly.  A hard prior boundary is
                # valid zero support, whereas a missing cell indicates a failed
                # ODE/grid evaluation.  Keeping the former lets the modular-cut
                # diagnostics distinguish mathematical truncation from an
                # incomplete numerical grid.
                if isfinite(likelihood) and isfinite(prior):
                    target_log_prob = float(likelihood + prior + log_nuisance_prior)
                    importance_log_weight = float(target_log_prob - log_nuisance_q)
                else:
                    target_log_prob = -np.inf
                    importance_log_weight = -np.inf
                record = {
                    "country": country,
                    "nuisance_index": int(local_idx),
                    "beta_grid_index": int(beta_idx),
                    "reporting_grid_index": int(reporting_idx),
                    "posterior_log_prob": target_log_prob,
                    "importance_log_weight": importance_log_weight,
                    "log_likelihood": float(likelihood),
                    "log_beta_reporting_prior": float(prior),
                    "log_nuisance_prior": log_nuisance_prior,
                    "log_nuisance_proposal": log_nuisance_q,
                    "beta_grid_edge": bool(beta_idx in {0, len(beta_grid) - 1}),
                    "reporting_grid_edge": bool(reporting_idx in {0, len(reporting_grid) - 1}),
                }
                for z_pos, value in enumerate(nuisance_vector[IMPORTANCE_NUISANCE_INDICES]):
                    record[f"nuisance_z_{z_pos}"] = float(value)
                record.update(sample)
                records.append(record)
    return records


def _importance_diagnostics_from_candidates(
    candidates: pd.DataFrame,
    weights: np.ndarray,
    *,
    nuisance_draws: int,
) -> dict[str, float]:
    if len(candidates) != len(weights):
        raise ValueError("Candidate rows and weights have different lengths")
    weights = np.asarray(weights, dtype=float)
    weights = weights / float(weights.sum())
    ess = 1.0 / float(np.sum(weights**2))
    entropy = -float(np.sum(weights * np.log(np.clip(weights, 1e-300, 1.0))))
    edge = candidates["beta_grid_edge"].astype(bool) | candidates["reporting_grid_edge"].astype(bool)
    edge_weight = float(weights[edge.to_numpy(dtype=bool)].sum())
    nuisance_weight = (
        pd.DataFrame({"nuisance_index": candidates["nuisance_index"].to_numpy(dtype=int), "weight": weights})
        .groupby("nuisance_index", sort=False)["weight"]
        .sum()
        .to_numpy(dtype=float)
    )
    nuisance_ess = 1.0 / float(np.sum(nuisance_weight**2)) if len(nuisance_weight) else 0.0
    return {
        "importance_candidate_count": float(len(candidates)),
        "importance_effective_sample_size": float(ess),
        "importance_ess_fraction": float(ess / max(len(candidates), 1)),
        "importance_entropy_effective_sample_size": float(np.exp(entropy)),
        "importance_max_weight": float(weights.max()),
        "importance_edge_weight": edge_weight,
        "importance_nuisance_effective_sample_size": float(nuisance_ess),
        "importance_nuisance_ess_fraction": float(nuisance_ess / max(int(nuisance_draws), 1)),
    }


def _modular_cut_candidate_weights(
    candidates: pd.DataFrame,
    *,
    nuisance_draws: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Normalize beta/reporting weights within each equally weighted nuisance draw."""

    weights = np.zeros(len(candidates), dtype=float)
    positions_by_nuisance: dict[int, np.ndarray] = {}
    probabilities_by_nuisance: dict[int, np.ndarray] = {}
    nuisance_values = candidates["nuisance_index"].to_numpy(dtype=int)
    log_weights = candidates["importance_log_weight"].to_numpy(dtype=float)
    for nuisance_index in range(int(nuisance_draws)):
        positions = np.flatnonzero(nuisance_values == nuisance_index)
        if len(positions) == 0:
            raise ValueError(
                f"No finite conditional support for nuisance draw {nuisance_index}"
            )
        local = log_weights[positions]
        finite = np.isfinite(local)
        if not finite.any():
            raise ValueError(
                f"No finite conditional support for nuisance draw {nuisance_index}"
            )
        probability = np.zeros(len(local), dtype=float)
        probability[finite] = np.exp(local[finite] - float(np.max(local[finite])))
        probability /= float(probability.sum())
        weights[positions] = probability / float(nuisance_draws)
        positions_by_nuisance[nuisance_index] = positions
        probabilities_by_nuisance[nuisance_index] = probability
    return weights, positions_by_nuisance, probabilities_by_nuisance


def _marginal_grid_effective_points(
    grid_index: np.ndarray,
    probability: np.ndarray,
) -> float:
    marginal = (
        pd.DataFrame(
            {
                "grid_index": np.asarray(grid_index, dtype=int),
                "probability": np.asarray(probability, dtype=float),
            }
        )
        .groupby("grid_index", sort=False)["probability"]
        .sum()
        .to_numpy(dtype=float)
    )
    return _effective_sample_size_from_weights(marginal)


def _modular_cut_conditional_grid_diagnostics(
    candidates: pd.DataFrame,
    positions_by_nuisance: dict[int, np.ndarray],
    probabilities_by_nuisance: dict[int, np.ndarray],
    *,
    nuisance_draws: int,
    beta_grid_points: int,
    reporting_grid_points: int,
) -> pd.DataFrame:
    """Audit every modular-cut conditional beta/reporting quadrature.

    Equal structural mass makes an aggregate edge-weight diagnostic an average:
    a badly truncated or under-resolved conditional grid can therefore be hidden
    by many easy nuisance draws.  This table retains the worst-case information
    needed for a scientific gate and is also written as an auditable artifact.
    """

    required = {
        "nuisance_index",
        "beta_grid_index",
        "reporting_grid_index",
        "beta_grid_edge",
        "reporting_grid_edge",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise KeyError(f"Modular-cut candidates missing grid columns: {sorted(missing)}")
    expected_count = int(beta_grid_points) * int(reporting_grid_points)
    if nuisance_draws <= 0 or expected_count <= 0:
        raise ValueError("Modular-cut grid dimensions and nuisance draws must be positive")

    rows: list[dict[str, Any]] = []
    for nuisance_index in range(int(nuisance_draws)):
        if nuisance_index not in positions_by_nuisance or nuisance_index not in probabilities_by_nuisance:
            raise ValueError(f"Missing modular-cut conditional weights for nuisance draw {nuisance_index}")
        positions = np.asarray(positions_by_nuisance[nuisance_index], dtype=int)
        probability = np.asarray(probabilities_by_nuisance[nuisance_index], dtype=float)
        if len(positions) != len(probability) or len(positions) == 0:
            raise ValueError(f"Invalid modular-cut conditional grid for nuisance draw {nuisance_index}")
        if (
            not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or not np.isclose(float(probability.sum()), 1.0, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError(f"Invalid modular-cut conditional probabilities for nuisance draw {nuisance_index}")

        local = candidates.iloc[positions]
        local_nuisance = local["nuisance_index"].to_numpy(dtype=int)
        if np.any(local_nuisance != nuisance_index):
            raise ValueError(f"Conditional positions cross nuisance draws for nuisance {nuisance_index}")
        cell_count = int(
            local.loc[:, ["beta_grid_index", "reporting_grid_index"]]
            .drop_duplicates()
            .shape[0]
        )
        beta_index = local["beta_grid_index"].to_numpy(dtype=int)
        reporting_index = local["reporting_grid_index"].to_numpy(dtype=int)
        indices_in_bounds = bool(
            np.all((beta_index >= 0) & (beta_index < int(beta_grid_points)))
            and np.all(
                (reporting_index >= 0)
                & (reporting_index < int(reporting_grid_points))
            )
        )
        duplicate_cells = int(len(local) - cell_count)
        missing_cells = int(max(expected_count - cell_count, 0))
        grid_complete = bool(
            cell_count == expected_count
            and duplicate_cells == 0
            and indices_in_bounds
        )

        beta_edge = local["beta_grid_edge"].astype(bool).to_numpy(dtype=bool)
        reporting_edge = local["reporting_grid_edge"].astype(bool).to_numpy(dtype=bool)
        combined_edge = beta_edge | reporting_edge
        effective_points = _effective_sample_size_from_weights(probability)
        beta_effective_points = _marginal_grid_effective_points(
            beta_index, probability
        )
        reporting_effective_points = _marginal_grid_effective_points(
            reporting_index, probability
        )
        max_weight = float(np.max(probability))
        beta_lower_edge_weight = float(probability[beta_index == 0].sum())
        beta_upper_edge_weight = float(
            probability[beta_index == int(beta_grid_points) - 1].sum()
        )
        reporting_lower_edge_weight = float(
            probability[reporting_index == 0].sum()
        )
        reporting_upper_edge_weight = float(
            probability[
                reporting_index == int(reporting_grid_points) - 1
            ].sum()
        )
        beta_edge_weight = float(probability[beta_edge].sum())
        reporting_edge_weight = float(probability[reporting_edge].sum())
        combined_edge_weight = float(probability[combined_edge].sum())

        fatal_issues: list[str] = []
        recommended_issues: list[str] = []
        if not grid_complete:
            fatal_issues.append("incomplete_grid")
            recommended_issues.append("incomplete_grid")
        if effective_points < MODULAR_CUT_FATAL_MIN_CONDITIONAL_EFFECTIVE_POINTS:
            fatal_issues.append("conditional_effective_points")
        if effective_points < MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS:
            recommended_issues.append("conditional_effective_points")
        if beta_effective_points < MODULAR_CUT_FATAL_MIN_AXIS_EFFECTIVE_POINTS:
            fatal_issues.append("beta_axis_effective_points")
        if beta_effective_points < MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS:
            recommended_issues.append("beta_axis_effective_points")
        if reporting_effective_points < MODULAR_CUT_FATAL_MIN_AXIS_EFFECTIVE_POINTS:
            fatal_issues.append("reporting_axis_effective_points")
        if reporting_effective_points < MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS:
            recommended_issues.append("reporting_axis_effective_points")
        if max_weight > MODULAR_CUT_FATAL_MAX_CONDITIONAL_WEIGHT:
            fatal_issues.append("conditional_single_weight")
        if max_weight > MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT:
            recommended_issues.append("conditional_single_weight")
        if combined_edge_weight > MODULAR_CUT_FATAL_MAX_CONDITIONAL_EDGE_WEIGHT:
            fatal_issues.append("conditional_edge_weight")
        if combined_edge_weight > MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT:
            recommended_issues.append("conditional_edge_weight")

        rows.append(
            {
                "nuisance_index": int(nuisance_index),
                "expected_candidate_count": expected_count,
                "candidate_count": int(len(local)),
                "unique_grid_cell_count": cell_count,
                "missing_grid_cell_count": missing_cells,
                "duplicate_grid_cell_count": duplicate_cells,
                "grid_indices_in_bounds": indices_in_bounds,
                "grid_complete": grid_complete,
                "conditional_effective_grid_points": float(effective_points),
                "conditional_beta_effective_grid_points": float(beta_effective_points),
                "conditional_reporting_effective_grid_points": float(reporting_effective_points),
                "conditional_max_single_weight": max_weight,
                "conditional_beta_lower_edge_weight": beta_lower_edge_weight,
                "conditional_beta_upper_edge_weight": beta_upper_edge_weight,
                "conditional_reporting_lower_edge_weight": reporting_lower_edge_weight,
                "conditional_reporting_upper_edge_weight": reporting_upper_edge_weight,
                "conditional_beta_edge_weight": beta_edge_weight,
                "conditional_reporting_edge_weight": reporting_edge_weight,
                "conditional_combined_edge_weight": combined_edge_weight,
                "fatal_valid": not fatal_issues,
                "recommended_valid": not recommended_issues,
                "fatal_issues": ",".join(fatal_issues),
                "recommended_issues": ",".join(recommended_issues),
            }
        )
    return pd.DataFrame(rows)


def _summarize_modular_cut_conditional_grid_diagnostics(
    diagnostics: pd.DataFrame,
) -> dict[str, float]:
    if diagnostics.empty:
        raise ValueError("Modular-cut conditional grid diagnostics are empty")
    return {
        "modular_cut_structural_draw_count": float(len(diagnostics)),
        "modular_cut_conditional_incomplete_grid_count": float(
            (~diagnostics["grid_complete"].astype(bool)).sum()
        ),
        "modular_cut_conditional_fatal_failure_count": float(
            (~diagnostics["fatal_valid"].astype(bool)).sum()
        ),
        "modular_cut_conditional_recommended_failure_count": float(
            (~diagnostics["recommended_valid"].astype(bool)).sum()
        ),
        "modular_cut_conditional_min_effective_grid_points": float(
            diagnostics["conditional_effective_grid_points"].min()
        ),
        "modular_cut_conditional_min_beta_effective_grid_points": float(
            diagnostics["conditional_beta_effective_grid_points"].min()
        ),
        "modular_cut_conditional_min_reporting_effective_grid_points": float(
            diagnostics["conditional_reporting_effective_grid_points"].min()
        ),
        "modular_cut_conditional_max_single_weight": float(
            diagnostics["conditional_max_single_weight"].max()
        ),
        "modular_cut_conditional_max_edge_weight": float(
            diagnostics["conditional_combined_edge_weight"].max()
        ),
    }


def _finalize_joint_importance_samples(
    *,
    task: ChainTask,
    candidates: pd.DataFrame,
    weights: np.ndarray,
    quality: dict[str, float],
    progress_file: Path,
    method_metadata: dict[str, Any],
    selected_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    total_draws = int(max(task.draws, 1) * max(task.grid_n_chains, 1))
    if selected_indices is None:
        rng = np.random.default_rng(task.seed + 7919)
        selected_indices = _stratified_resample_indices(weights, total_draws, rng)
    selected_indices = np.asarray(selected_indices, dtype=int)
    if selected_indices.shape != (total_draws,):
        raise ValueError("joint-importance selected index count does not match requested draws")
    selected = candidates.iloc[selected_indices].reset_index(drop=True).copy()
    selected["source_normalized_weight"] = weights[selected_indices]
    for key, value in quality.items():
        selected[key] = float(value)
    for key, value in method_metadata.items():
        selected[key] = value
    selected["sampling_method"] = JOINT_IMPORTANCE_SAMPLER
    selected["accepted_fraction"] = 1.0
    selected["structural_draw_id"] = selected["nuisance_index"].astype(int)

    chain_count = int(max(task.grid_n_chains, 1))
    draws_per_chain = int(max(task.draws, 1))
    selected["chain"] = np.repeat(
        np.arange(1, chain_count + 1, dtype=int),
        draws_per_chain,
    )[: len(selected)]
    selected["draw"] = np.tile(
        np.arange(1, draws_per_chain + 1, dtype=int),
        chain_count,
    )[: len(selected)]
    selected["step"] = selected["draw"]
    selected["source_candidate_index"] = selected_indices

    keep = [
        "country",
        "chain",
        "draw",
        "step",
        "posterior_log_prob",
        "accepted_fraction",
        "sampling_method",
        "inference_structure",
        "structural_draw_id",
        "source_candidate_index",
        "source_normalized_weight",
        "nuisance_index",
        "beta_grid_index",
        "reporting_grid_index",
        "importance_log_weight",
        "modular_conditional_weight",
        "log_likelihood",
        "log_beta_reporting_prior",
        "log_nuisance_prior",
        "log_nuisance_proposal",
        "importance_stage",
        "importance_adaptation_nuisance_draws",
        "importance_refinement_nuisance_draws",
        "importance_gaussian_components",
        "importance_defensive_prior_fraction",
        *PARAMETER_SAMPLE_COLUMNS,
        "resistance_prevalence",
        "reporting_trend_end_multiplier",
        *quality.keys(),
    ]
    keep = [column for column in keep if column in selected.columns]
    try:
        if str(method_metadata.get("inference_structure", "")).lower() == "modular_hierarchical_cut":
            progress_file.write_text(
                f"done candidates={len(candidates)} "
                f"structural_draws={quality['modular_cut_structural_draw_count']:.0f} "
                f"min_conditional_ess={quality['modular_cut_conditional_min_effective_grid_points']:.1f} "
                f"max_conditional_weight={quality['modular_cut_conditional_max_single_weight']:.4f} "
                f"max_conditional_edge_weight={quality['modular_cut_conditional_max_edge_weight']:.4f}"
            )
        else:
            progress_file.write_text(
                f"done candidates={len(candidates)} ess={quality['importance_effective_sample_size']:.1f} "
                f"nuisance_ess={quality['importance_nuisance_effective_sample_size']:.1f} "
                f"max_weight={quality['importance_max_weight']:.4f} "
                f"edge_weight={quality['importance_edge_weight']:.4f}"
            )
    except OSError:
        pass
    return selected.loc[:, keep]


def _run_joint_importance(
    task: ChainTask,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    settings: dict[str, Any],
    calibrated_start: np.ndarray,
) -> pd.DataFrame:
    n_nuisance = int(max(task.importance_nuisance_draws, 1))
    beta_points = _odd_grid_points(task.importance_beta_grid_points, minimum=5)
    reporting_points = _odd_grid_points(task.importance_reporting_grid_points, minimum=5)
    beta_half_width = float(task.importance_log_beta_half_width)
    reporting_half_width = float(task.importance_reporting_coordinate_half_width)
    if beta_half_width <= 0.0 or reporting_half_width <= 0.0:
        raise ValueError("joint_importance grid half-widths must be positive")

    progress_dir = _mcmc_progress_dir(task.output_stem)
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"{task.country}_joint_importance.txt"
    try:
        progress_file.write_text(
            f"starting nuisance={n_nuisance} beta_grid={beta_points} reporting_grid={reporting_points}"
        )
    except OSError:
        pass

    beta_grid = np.linspace(
        float(calibrated_start[0]) - beta_half_width,
        float(calibrated_start[0]) + beta_half_width,
        beta_points,
    )
    reporting_grid = np.linspace(
        float(calibrated_start[1]) - reporting_half_width,
        float(calibrated_start[1]) + reporting_half_width,
        reporting_points,
    )

    workers = int(max(task.grid_eval_jobs, 1))
    priors = settings["priors"]

    def evaluate_nuisance_set(
        nuisance_vectors: np.ndarray,
        nuisance_log_prior: np.ndarray,
        nuisance_log_q: np.ndarray,
        *,
        stage: str,
        nuisance_index_offset: int = 0,
    ) -> pd.DataFrame:
        local_n = int(len(nuisance_vectors))
        if local_n <= 0:
            raise RuntimeError(f"joint_importance received no nuisance proposal points for {task.country}")
        chunks = [
            chunk
            for chunk in np.array_split(np.arange(local_n, dtype=int), min(workers, local_n))
            if len(chunk) > 0
        ]
        payloads = [
            {
                "country": task.country,
                "runtime_base": runtime_base,
                "observed": observed,
                "settings": settings,
                "nuisance_vectors": nuisance_vectors[chunk],
                "nuisance_log_prior": nuisance_log_prior[chunk],
                "nuisance_log_q": nuisance_log_q[chunk],
                "nuisance_indices": chunk + int(nuisance_index_offset),
                "beta_grid": beta_grid,
                "reporting_grid": reporting_grid,
            }
            for chunk in chunks
        ]
        if workers <= 1 or len(payloads) <= 1:
            chunk_records = [_evaluate_joint_importance_chunk(payload) for payload in payloads]
        else:
            chunk_records = Parallel(n_jobs=len(payloads), backend="loky", inner_max_num_threads=1)(
                delayed(_evaluate_joint_importance_chunk)(payload) for payload in payloads
            )
        records = [record for chunk_records_part in chunk_records for record in chunk_records_part]
        if not records:
            raise RuntimeError(f"joint_importance found no finite {stage} posterior support for {task.country}")
        frame = pd.DataFrame(records)
        frame["importance_stage"] = stage
        return frame

    inference_structure = str(
        settings.get("inference_structure", "country_joint_feedback")
    ).lower()
    if inference_structure in {"modular_cut", "modular_hierarchical_cut"}:
        shared_seed = int(settings.get("shared_structural_seed", 20260712))
        nuisance_vectors, nuisance_log_prior = _prior_nuisance_vectors(
            n=n_nuisance,
            seed=shared_seed,
            calibrated_start=calibrated_start,
            base_config=runtime_base,
            priors=priors,
        )
        candidates = evaluate_nuisance_set(
            nuisance_vectors,
            nuisance_log_prior,
            nuisance_log_prior,
            stage="modular_prior_cut",
        )
        available_nuisance = set(
            candidates["nuisance_index"].astype(int).unique().tolist()
        )
        expected_nuisance = set(range(n_nuisance))
        if available_nuisance != expected_nuisance:
            missing = sorted(expected_nuisance - available_nuisance)
            raise RuntimeError(
                "modular cut grid has no finite conditional support for "
                f"{task.country} nuisance draws {missing}; widen the beta/reporting grid"
            )

        (
            weights,
            conditional_positions,
            conditional_probabilities,
        ) = _modular_cut_candidate_weights(
            candidates,
            nuisance_draws=n_nuisance,
        )
        candidates["modular_conditional_weight"] = weights * float(n_nuisance)
        quality = _importance_diagnostics_from_candidates(
            candidates,
            weights,
            nuisance_draws=n_nuisance,
        )
        conditional_diagnostics = _modular_cut_conditional_grid_diagnostics(
            candidates,
            conditional_positions,
            conditional_probabilities,
            nuisance_draws=n_nuisance,
            beta_grid_points=beta_points,
            reporting_grid_points=reporting_points,
        )
        conditional_diagnostics.insert(0, "country", task.country)
        quality.update(
            _summarize_modular_cut_conditional_grid_diagnostics(
                conditional_diagnostics
            )
        )
        # Equal cut-module weights are a design choice, not a nuisance
        # posterior or a convergence estimate.  Preserve NaN in the legacy ESS
        # field and expose the structural draw count under an honest label.
        quality["importance_nuisance_effective_sample_size"] = np.nan
        quality["importance_nuisance_ess_fraction"] = np.nan
        write_dataframe(
            conditional_diagnostics,
            project_path(
                "outputs/metadata",
                f"{_artifact_stem(task.output_stem, 'bayesian_modular_cut_conditional_grid_quality', 'modular_cut_conditional_grid_quality')}_{task.country}.csv",
            ),
        )

        total_draws = int(max(task.draws, 1) * max(task.grid_n_chains, 1))
        structural_rng = np.random.default_rng(shared_seed + 41)
        structural_sequence: list[int] = []
        while len(structural_sequence) < total_draws:
            structural_sequence.extend(
                structural_rng.permutation(n_nuisance).astype(int).tolist()
            )
        conditional_rng = np.random.default_rng(task.seed + 7919)
        selected_indices = np.empty(total_draws, dtype=int)
        for draw_index, nuisance_index in enumerate(structural_sequence[:total_draws]):
            selected_indices[draw_index] = int(
                conditional_rng.choice(
                    conditional_positions[nuisance_index],
                    p=conditional_probabilities[nuisance_index],
                )
            )
        return _finalize_joint_importance_samples(
            task=task,
            candidates=candidates,
            weights=weights,
            quality=quality,
            progress_file=progress_file,
            method_metadata={
                "inference_structure": "modular_hierarchical_cut",
                "importance_adaptation_nuisance_draws": 0,
                "importance_refinement_nuisance_draws": 0,
                "importance_gaussian_components": 0,
                "importance_defensive_prior_fraction": 1.0,
            },
            selected_indices=selected_indices,
        )

    adaptation_draws = int(max(24, min(n_nuisance, n_nuisance // 4 if n_nuisance >= 96 else n_nuisance)))
    stage1_vectors, stage1_log_prior = _prior_nuisance_vectors(
        n=adaptation_draws,
        seed=task.seed + 3109,
        calibrated_start=calibrated_start,
        base_config=runtime_base,
        priors=priors,
    )
    try:
        progress_file.write_text(
            f"adaptation nuisance={adaptation_draws} beta_grid={beta_points} reporting_grid={reporting_points}"
        )
    except OSError:
        pass
    stage1 = evaluate_nuisance_set(
        stage1_vectors,
        stage1_log_prior,
        stage1_log_prior,
        stage="adaptation_prior",
    )
    stage1_log_weight = stage1["importance_log_weight"].to_numpy(dtype=float)
    if not np.isfinite(stage1_log_weight).any():
        raise RuntimeError(f"joint_importance adaptation produced no finite weights for {task.country}")
    prior_coordinate_sd = np.nanstd(
        stage1_vectors[:, IMPORTANCE_NUISANCE_INDICES],
        axis=0,
        ddof=1 if len(stage1_vectors) > 1 else 0,
    )
    prior_coordinate_sd = np.where(np.isfinite(prior_coordinate_sd) & (prior_coordinate_sd > 0.0), prior_coordinate_sd, 1.0)
    stage1_components = _build_nuisance_mixture_components(
        stage1,
        stage1_log_weight,
        prior_coordinate_sd=prior_coordinate_sd,
    )

    refinement_draws = int(max(24, min(n_nuisance, n_nuisance // 2 if n_nuisance >= 96 else n_nuisance)))
    refine_vectors, refine_log_prior, refine_log_q = _mixture_nuisance_vectors(
        n=refinement_draws,
        seed=task.seed + 7193,
        calibrated_start=calibrated_start,
        base_config=runtime_base,
        priors=priors,
        gaussian_components=stage1_components,
        defensive_prior_fraction=task.importance_defensive_prior_fraction,
    )
    try:
        progress_file.write_text(
            f"refinement nuisance={refinement_draws} adaptation={adaptation_draws} "
            f"components={len(stage1_components)} defensive_prior={task.importance_defensive_prior_fraction:.3f}"
        )
    except OSError:
        pass
    refinement = evaluate_nuisance_set(
        refine_vectors,
        refine_log_prior,
        refine_log_q,
        stage="adaptive_refinement",
        nuisance_index_offset=100000,
    )
    adaptation_pool = pd.concat([stage1, refinement], ignore_index=True)
    adaptation_log_weight = adaptation_pool["importance_log_weight"].to_numpy(dtype=float)
    main_components = _build_nuisance_mixture_components(
        adaptation_pool,
        adaptation_log_weight,
        prior_coordinate_sd=prior_coordinate_sd,
    )

    main_vectors, main_log_prior, main_log_q = _mixture_nuisance_vectors(
        n=n_nuisance,
        seed=task.seed + 9151,
        calibrated_start=calibrated_start,
        base_config=runtime_base,
        priors=priors,
        gaussian_components=main_components,
        defensive_prior_fraction=task.importance_defensive_prior_fraction,
    )
    try:
        progress_file.write_text(
            f"main nuisance={n_nuisance} adaptation={adaptation_draws} refinement={refinement_draws} "
            f"components={len(main_components)} defensive_prior={task.importance_defensive_prior_fraction:.3f}"
        )
    except OSError:
        pass

    candidates = evaluate_nuisance_set(
        main_vectors,
        main_log_prior,
        main_log_q,
        stage="adaptive_mixture",
    )
    logp = candidates["importance_log_weight"].to_numpy(dtype=float)
    logp = logp - float(np.nanmax(logp))
    raw_weights = np.exp(logp)
    if not np.isfinite(raw_weights).all() or float(raw_weights.sum()) <= 0.0:
        raise RuntimeError(f"joint_importance produced invalid weights for {task.country}")
    weights = raw_weights / float(raw_weights.sum())
    quality = _importance_diagnostics_from_candidates(
        candidates,
        weights,
        nuisance_draws=n_nuisance,
    )

    return _finalize_joint_importance_samples(
        task=task,
        candidates=candidates,
        weights=weights,
        quality=quality,
        progress_file=progress_file,
        method_metadata={
            "inference_structure": "country_joint_feedback",
            "importance_adaptation_nuisance_draws": int(adaptation_draws),
            "importance_refinement_nuisance_draws": int(refinement_draws),
            "importance_gaussian_components": int(len(main_components)),
            "importance_defensive_prior_fraction": float(
                task.importance_defensive_prior_fraction
            ),
        },
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


def _reflect_value_into_bounds(value: float, lower: float, upper: float) -> float:
    if not (isfinite(value) and isfinite(lower) and isfinite(upper)) or upper <= lower:
        return float(value)
    width = float(upper - lower)
    reflected = (float(value) - lower) % (2.0 * width)
    if reflected > width:
        reflected = 2.0 * width - reflected
    return float(lower + reflected)


def _component_bounds(
    vector: np.ndarray,
    component: int,
    priors: dict[str, Any],
) -> tuple[float, float] | None:
    beta_bounds = _configured_prior_bounds(priors, "beta_S", (0.0005, 0.5))
    reporting_bounds = _configured_prior_bounds(
        priors, "reporting_multiplier", (0.02, 20.0)
    )
    if component == 0:
        lower = float(np.log(beta_bounds[0]))
        upper = float(np.log(beta_bounds[1]))
        if _parameterization(priors) == "beta_reporting_product":
            reporting_lower = float(np.log(reporting_bounds[0]))
            reporting_upper = float(np.log(reporting_bounds[1]))
            reporting_coordinate = float(vector[1])
            base_log_beta = _base_log_beta(priors)
            lower = max(lower, reporting_coordinate + base_log_beta - reporting_upper)
            upper = min(upper, reporting_coordinate + base_log_beta - reporting_lower)
        return lower, upper
    if component == 1:
        reporting_lower = float(np.log(reporting_bounds[0]))
        reporting_upper = float(np.log(reporting_bounds[1]))
        if _parameterization(priors) == "beta_reporting_product":
            log_beta = float(vector[0])
            base_log_beta = _base_log_beta(priors)
            return (
                reporting_lower + log_beta - base_log_beta,
                reporting_upper + log_beta - base_log_beta,
            )
        return reporting_lower, reporting_upper
    if component in {2, 3, 4, 5}:
        name = PARAMETER_SAMPLE_COLUMNS[component]
        if _configured_prior_spec(priors, name) is None:
            return None
        lower, upper = _configured_prior_bounds(priors, name, (0.0, 1.0))
        return _logit(lower), _logit(upper)
    if component == 6:
        lower, upper = _configured_prior_bounds(
            priors, "infectious_duration_symptomatic", (7.0, 35.0)
        )
        return float(np.log(lower)), float(np.log(upper))
    if component == 7:
        lower, upper = _configured_prior_bounds(
            priors, "infectious_duration_asymptomatic", (5.0, 28.0)
        )
        return float(np.log(lower)), float(np.log(upper))
    return None


def _reflect_component_into_bounds(
    vector: np.ndarray,
    component: int,
    priors: dict[str, Any],
) -> np.ndarray:
    bounds = _component_bounds(vector, component, priors)
    if bounds is None:
        return vector
    out = vector.copy()
    out[component] = _reflect_value_into_bounds(float(out[component]), bounds[0], bounds[1])
    return out


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


def _regularized_spd(matrix: np.ndarray) -> np.ndarray:
    """Return a minimally regularized symmetric positive-definite matrix."""

    symmetric = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(max(np.max(eigenvalues), 1.0))
    floor = largest * 1e-12
    regularized = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return 0.5 * (regularized + regularized.T)


def _general_gaussian_mixture_proposal(
    component_means: np.ndarray,
    component_covariances: np.ndarray,
    component_weights: np.ndarray,
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw from and score a fully normalized finite Gaussian mixture."""

    means = np.asarray(component_means, dtype=float)
    covariances = np.asarray(component_covariances, dtype=float)
    weights = np.asarray(component_weights, dtype=float)
    if means.ndim != 2:
        raise ValueError("Gaussian-mixture component means must be two-dimensional")
    component_count, dimension = means.shape
    if covariances.shape != (component_count, dimension, dimension):
        raise ValueError("Gaussian-mixture covariance dimension mismatch")
    if weights.shape != (component_count,) or n < 1:
        raise ValueError("Gaussian-mixture weight/count dimension mismatch")
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("Gaussian-mixture weights must be finite and positive")
    weights = weights / float(np.sum(weights))

    expected_counts = float(n) * weights
    component_draw_counts = np.floor(expected_counts).astype(int)
    remainder = int(n - int(component_draw_counts.sum()))
    if remainder > 0:
        priority = np.argsort(-(expected_counts - component_draw_counts))
        component_draw_counts[priority[:remainder]] += 1
    allocation_weights = component_draw_counts.astype(float) / float(n)
    source_component = np.repeat(
        np.arange(component_count, dtype=int),
        component_draw_counts,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(source_component)
    candidates = np.empty((n, dimension), dtype=float)
    information_matrices: list[np.ndarray] = []
    log_determinants: list[float] = []
    for component in range(component_count):
        covariance = _regularized_spd(covariances[component])
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(log_determinant):
            raise RuntimeError("Gaussian-mixture covariance is not positive definite")
        information_matrices.append(np.linalg.inv(covariance))
        log_determinants.append(float(log_determinant))
        positions = np.flatnonzero(source_component == component)
        if len(positions) > 0:
            candidates[positions] = rng.multivariate_normal(
                means[component],
                covariance,
                size=len(positions),
                check_valid="raise",
            )

    log_component_density = np.empty((n, component_count), dtype=float)
    normalizing_constant = dimension * np.log(2.0 * np.pi)
    for component, (information, log_determinant) in enumerate(
        zip(information_matrices, log_determinants)
    ):
        delta = candidates - means[component]
        quadratic = np.einsum("ij,jk,ik->i", delta, information, delta)
        log_component_density[:, component] = (
            np.log(max(allocation_weights[component], 1e-300))
            - 0.5 * (normalizing_constant + log_determinant + quadratic)
        )
    log_mixture_density = scipy_special.logsumexp(
        log_component_density,
        axis=1,
    )
    return candidates, log_mixture_density, source_component


def _state_gaussian_mixture_proposal(
    mean: np.ndarray,
    covariance: np.ndarray,
    covariance_scales: tuple[float, ...],
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw from, and evaluate, a normalized defensive Gaussian mixture.

    Candidates are deliberately *not* rejection-sampled at parameter bounds.
    Out-of-bounds candidates receive zero target mass later.  This keeps the
    proposal density analytically normalized and avoids using a noisy estimate
    of a truncated-normal normalizing constant in the importance weights.
    """

    centre = np.asarray(mean, dtype=float)
    base_covariance = _regularized_spd(covariance)
    scales = np.asarray(covariance_scales, dtype=float)
    if centre.ndim != 1 or base_covariance.shape != (len(centre), len(centre)):
        raise ValueError("State proposal mean/covariance dimension mismatch")
    if n < 1 or scales.ndim != 1 or len(scales) < 1:
        raise ValueError("State proposal requires candidates and covariance scales")
    if not np.isfinite(scales).all() or bool((scales <= 0.0).any()):
        raise ValueError("State proposal covariance scales must be finite and positive")

    component_covariances: list[np.ndarray] = []
    for scale in scales:
        component_covariance = _regularized_spd(base_covariance * float(scale))
        component_covariances.append(component_covariance)
    component_means = np.repeat(centre[None, :], len(scales), axis=0)
    candidates, log_mixture_density, source_component = (
        _general_gaussian_mixture_proposal(
            component_means,
            np.asarray(component_covariances, dtype=float),
            np.full(len(scales), 1.0 / len(scales), dtype=float),
            n=n,
            seed=seed,
        )
    )
    return candidates, log_mixture_density, source_component, scales


def _tempered_importance_adaptation(
    candidates: np.ndarray,
    log_importance_weight: np.ndarray,
    fallback_covariance: np.ndarray,
    *,
    target_ess: float,
    covariance_regularization: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate proposal moments with weights tempered to a stable ESS.

    The tempered distribution is used only to adapt a new proposal. Final
    inference always uses fresh candidates and untempered exact-target weights,
    so tempering cannot flatten the reported conditional distribution.
    """

    values = np.asarray(candidates, dtype=float)
    log_weight = np.asarray(log_importance_weight, dtype=float)
    finite = np.isfinite(log_weight)
    if values.ndim != 2 or len(values) != len(log_weight):
        raise ValueError("Importance adaptation candidate/weight dimension mismatch")
    finite_count = int(np.sum(finite))
    dimension = values.shape[1]
    required = max(dimension * 4, 32)
    if finite_count < required:
        raise RuntimeError(
            f"Importance adaptation has only {finite_count} finite candidates; "
            f"requires at least {required}"
        )
    finite_log_weight = log_weight[finite]
    finite_values = values[finite]

    def normalized(power: float) -> np.ndarray:
        powered = float(power) * finite_log_weight
        powered -= float(scipy_special.logsumexp(powered))
        return np.exp(powered)

    raw_weight = normalized(1.0)
    raw_ess = float(1.0 / np.sum(raw_weight**2))
    target = float(np.clip(target_ess, dimension * 4.0, finite_count))
    if raw_ess >= target:
        exponent = 1.0
        adaptation_weight = raw_weight
    else:
        # ESS decreases monotonically as the power increases from zero to one.
        lower_power = 0.0
        upper_power = 1.0
        for _ in range(60):
            midpoint = 0.5 * (lower_power + upper_power)
            midpoint_weight = normalized(midpoint)
            midpoint_ess = float(1.0 / np.sum(midpoint_weight**2))
            if midpoint_ess >= target:
                lower_power = midpoint
            else:
                upper_power = midpoint
        exponent = lower_power
        adaptation_weight = normalized(exponent)

    adapted_mean = np.sum(
        finite_values * adaptation_weight[:, None],
        axis=0,
    )
    centered = finite_values - adapted_mean
    covariance_denominator = max(
        1.0 - float(np.sum(adaptation_weight**2)),
        1e-8,
    )
    weighted_covariance = (
        (centered * adaptation_weight[:, None]).T @ centered
    ) / covariance_denominator
    regularization = float(np.clip(covariance_regularization, 0.0, 1.0))
    adapted_covariance = (
        (1.0 - regularization) * weighted_covariance
        + regularization * _regularized_spd(fallback_covariance)
    )
    return (
        adapted_mean,
        _regularized_spd(adapted_covariance),
        float(exponent),
        raw_ess,
    )


def _importance_pareto_shape(log_weight: np.ndarray) -> float:
    """Estimate the generalized-Pareto shape of the largest importance ratios."""

    finite = np.asarray(log_weight, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 50:
        return float("inf")
    ratios = np.exp(finite - float(np.max(finite)))
    tail_count = int(min(len(ratios) // 5, 3.0 * np.sqrt(len(ratios))))
    tail_count = max(tail_count, 20)
    ordered = np.sort(ratios)
    threshold_index = max(len(ordered) - tail_count - 1, 0)
    threshold = float(ordered[threshold_index])
    excess = ordered[-tail_count:] - threshold
    positive = excess[excess > 0.0]
    if len(positive) < 10 or float(np.max(positive)) <= 0.0:
        return 0.0
    try:
        shape, _, _ = genpareto.fit(positive, floc=0.0)
    except (FloatingPointError, RuntimeError, ValueError):
        return float("inf")
    return float(shape) if np.isfinite(shape) else float("inf")


def _weighted_quantile_1d(
    values: np.ndarray,
    weight: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(np.asarray(values, dtype=float))
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weight = np.asarray(weight, dtype=float)[order]
    cumulative = np.cumsum(sorted_weight)
    if len(cumulative) == 0 or cumulative[-1] <= 0.0:
        return float("nan")
    cumulative /= float(cumulative[-1])
    position = int(np.searchsorted(cumulative, probability, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


def _minimum_importance_tail_ess(
    candidates: np.ndarray,
    normalized_weight: np.ndarray,
) -> float:
    """Return the weakest weighted 2.5%/97.5% tail support across coordinates."""

    values = np.asarray(candidates, dtype=float)
    weight = np.asarray(normalized_weight, dtype=float)
    if values.ndim != 2 or weight.ndim != 1 or len(values) != len(weight):
        return 0.0
    finite = np.isfinite(weight) & (weight > 0.0) & np.isfinite(values).all(axis=1)
    if not bool(finite.any()):
        return 0.0
    values = values[finite]
    weight = weight[finite]
    weight /= float(np.sum(weight))
    tail_ess: list[float] = []
    for coordinate in range(values.shape[1]):
        column = values[:, coordinate]
        lower = _weighted_quantile_1d(column, weight, 0.025)
        upper = _weighted_quantile_1d(column, weight, 0.975)
        for mask in (column <= lower, column >= upper):
            tail_weight = weight[mask]
            mass = float(np.sum(tail_weight))
            tail_ess.append(
                0.0
                if mass <= 0.0
                else float(mass**2 / np.sum(tail_weight**2))
            )
    return float(min(tail_ess)) if tail_ess else 0.0


def _state_importance_quality_failures(
    *,
    effective_sample_size: float,
    maximum_weight: float,
    pareto_k: float,
    minimum_tail_ess: float,
    out_of_bounds_fraction: float,
    require_recommended: bool,
) -> list[str]:
    """Return fail-closed exact-importance support/weight diagnostics."""

    max_out_of_bounds = (
        STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION
        if require_recommended
        else STATE_LAPLACE_FATAL_MAX_BOUNDARY_REJECTION
    )
    min_ess = (
        STATE_IMPORTANCE_RECOMMENDED_MIN_ESS
        if require_recommended
        else STATE_IMPORTANCE_FATAL_MIN_ESS
    )
    max_weight = (
        STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT
        if require_recommended
        else STATE_IMPORTANCE_FATAL_MAX_WEIGHT
    )
    max_pareto_k = (
        STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K
        if require_recommended
        else STATE_IMPORTANCE_FATAL_MAX_PARETO_K
    )
    min_tail_ess = (
        STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS
        if require_recommended
        else STATE_IMPORTANCE_FATAL_MIN_TAIL_ESS
    )
    failures: list[str] = []
    if not np.isfinite(effective_sample_size) or effective_sample_size < min_ess:
        failures.append("effective_sample_size")
    if not np.isfinite(maximum_weight) or maximum_weight > max_weight:
        failures.append("maximum_weight")
    if not np.isfinite(pareto_k) or pareto_k > max_pareto_k:
        failures.append("pareto_k")
    if not np.isfinite(minimum_tail_ess) or minimum_tail_ess < min_tail_ess:
        failures.append("minimum_tail_ess")
    if (
        not np.isfinite(out_of_bounds_fraction)
        or out_of_bounds_fraction >= max_out_of_bounds
    ):
        failures.append("out_of_bounds_fraction")
    return failures


def _state_space_exact_negative_log_target(
    vector: np.ndarray,
    *,
    runtime_base: dict[str, Any],
    prior_base: dict[str, Any],
    observed: pd.DataFrame,
    country: str,
    coordinate_names: list[str],
    rho: float,
    innovation_sd: float,
    beta_prior_sd: float,
    reporting_prior_sd: float,
    dispersion: float,
) -> float:
    """Evaluate the exact conditional target represented by the MAP fit."""

    coordinate = np.asarray(vector, dtype=float)
    if len(coordinate) != len(coordinate_names):
        raise ValueError("State target coordinate length mismatch")
    candidate = deepcopy(runtime_base)
    candidate["transmission"]["beta_S"] = float(np.exp(coordinate[0]))
    candidate["reporting_multiplier"] = float(np.exp(coordinate[1]))
    periods = candidate["transmission"]["log_beta_time_variation"]["periods"]
    period_by_year = {
        int(pd.Timestamp(period["start_date"]).year): period for period in periods
    }
    latent_years = [
        int(name.removeprefix("log_beta_process_"))
        for name in coordinate_names[2:]
    ]
    for year, value in zip(latent_years, coordinate[2:]):
        if year not in period_by_year:
            raise ValueError(f"Missing calibrated process period for {year}")
        period_by_year[year]["log_multiplier"] = float(value)

    predicted = _calibration_predicted_means(candidate, observed, country)
    likelihood_observed = grouped_likelihood_observations(observed)
    actual = pd.to_numeric(
        likelihood_observed["reported_cases"], errors="raise"
    ).to_numpy(dtype=float)
    if len(predicted) != len(actual):
        raise ValueError("Exact state target prediction/observation length mismatch")
    latent = coordinate[2:]
    state_starts = pd.Series(
        pd.to_datetime([f"{year:04d}-01-01" for year in latent_years])
    )
    state_ends = pd.Series(
        pd.to_datetime([f"{year + 1:04d}-01-01" for year in latent_years])
    )
    state_midpoints = state_starts + (state_ends - state_starts) / 2
    # Reuse the calibration target's calendar-aware transition contract.  A
    # nominal one-year step is not exactly one tropical year across leap-year
    # boundaries, so integer year differences would define a different target.
    stationary_sd, transition_rho, transition_sd = _annual_ar1_transition_scales(
        state_midpoints,
        rho=rho,
        innovation_sd=innovation_sd,
    )
    process_penalty = 0.5 * float((latent[0] / stationary_sd) ** 2)
    if len(latent) > 1:
        process_penalty += 0.5 * float(
            np.sum(((latent[1:] - transition_rho * latent[:-1]) / transition_sd) ** 2)
        )
    beta_centre = float(np.log(prior_base["transmission"]["beta_S"]))
    reporting_centre = float(np.log(prior_base.get("reporting_multiplier", 1.0)))
    parameter_penalty = 0.5 * float(
        ((coordinate[0] - beta_centre) / beta_prior_sd) ** 2
        + ((coordinate[1] - reporting_centre) / reporting_prior_sd) ** 2
    )
    return float(
        negative_binomial_nll(actual, predicted, dispersion)
        + process_penalty
        + parameter_penalty
        + reporting_rate_prior_penalty(candidate)
    )


def _run_state_space_exact_importance_cut(
    task: ChainTask,
    runtime_base: dict[str, Any],
    observed: pd.DataFrame,
    settings: dict[str, Any],
    calibrated_start: np.ndarray,
) -> pd.DataFrame:
    """Propagate one-use state posterior and shared external structural priors.

    Surveillance data define one conditional state posterior. The stored
    Gauss-Newton approximation is used only as a proposal; candidates are
    corrected against that posterior's exact NB2/AR(1)/prior target before
    resampling. This is numerical evaluation of the same target, not a second
    likelihood factor. Structural quantities form an equal-mass external-prior
    sensitivity block, not a full joint or hierarchical posterior.
    """

    artifact = load_calibrated_country_artifact(task.country)
    if artifact is None:
        raise RuntimeError(
            f"No current accepted calibration artifact for {task.country}"
        )
    metadata = artifact.get("metadata", {})
    approximation = metadata.get("state_posterior_approximation", {})
    if not isinstance(approximation, dict):
        raise RuntimeError(
            f"Calibration artifact for {task.country} has no state posterior approximation"
        )
    method = str(approximation.get("method", ""))
    if method != "gauss_newton_laplace_conditional":
        raise RuntimeError(
            f"Unsupported state posterior approximation for {task.country}: {method!r}"
        )
    if task.beta_prior_log_sd is not None or task.reporting_prior_log_sd is not None:
        raise ValueError(
            "beta/reporting prior-width overrides cannot modify a stored state-space "
            "posterior approximation; recalibrate with explicit process_model priors"
        )
    artifact_dispersion = float(approximation.get("measurement_dispersion", np.nan))
    artifact_frequency = str(
        approximation.get("likelihood_observation_frequency", "")
    ).lower()
    if not np.isclose(artifact_dispersion, float(task.dispersion)):
        raise ValueError(
            f"Requested dispersion={task.dispersion} differs from the calibrated "
            f"state posterior dispersion={artifact_dispersion} for {task.country}"
        )
    if artifact_frequency != str(task.likelihood_observation_frequency).lower():
        raise ValueError(
            "Requested likelihood observation frequency differs from the stored "
            f"state posterior for {task.country}: requested={task.likelihood_observation_frequency}, "
            f"stored={artifact_frequency}"
        )
    coordinate_names = [str(name) for name in approximation.get("coordinate_names", [])]
    if coordinate_names[:2] != ["log_beta_S", "log_reporting_multiplier"]:
        raise RuntimeError(
            f"Invalid state posterior coordinate contract for {task.country}"
        )
    latent_names = coordinate_names[2:]
    if not latent_names or any(
        not name.startswith("log_beta_process_") for name in latent_names
    ):
        raise RuntimeError(
            f"Calibration artifact for {task.country} has no dated latent process coordinates"
        )

    mean = np.asarray(approximation.get("map_vector", []), dtype=float)
    covariance = np.asarray(approximation.get("covariance", []), dtype=float)
    lower = np.asarray(approximation.get("lower_bounds", []), dtype=float)
    upper = np.asarray(approximation.get("upper_bounds", []), dtype=float)
    dimension = len(coordinate_names)
    rank = int(approximation.get("rank", 0))
    condition_number = float(approximation.get("condition_number", np.inf))
    max_condition = float(
        load_configs()["baseline"]
        .get("calibration", {})
        .get("process_model", {})
        .get("max_posterior_condition_number", 1e8)
    )
    if (
        len(mean) != dimension
        or covariance.shape != (dimension, dimension)
        or lower.shape != mean.shape
        or upper.shape != mean.shape
        or rank != dimension
        or not np.isfinite(condition_number)
        or condition_number > max_condition
    ):
        raise RuntimeError(
            f"State posterior approximation failed identifiability gate for {task.country}: "
            f"rank={rank}/{dimension}, condition={condition_number:.6g}, "
            f"limit={max_condition:.6g}"
        )

    latent_years = [int(name.removeprefix("log_beta_process_")) for name in latent_names]
    if any(current != previous + 1 for previous, current in zip(latent_years, latent_years[1:])):
        raise RuntimeError(f"Non-consecutive historical process years for {task.country}")
    production_end_year = int(
        pd.Timestamp(load_configs()["baseline"]["calendar"]["analysis_end_date"]).year
    )
    future_years = list(range(latent_years[-1] + 1, production_end_year + 1))
    variation = (
        artifact.get("config", {})
        .get("transmission", {})
        .get("log_beta_time_variation", {})
    )
    process_rho = float(variation.get("ar1_rho", np.nan))
    process_innovation_sd = float(variation.get("innovation_sd", np.nan))
    if not 0.0 <= process_rho < 1.0 or process_innovation_sd <= 0.0:
        raise RuntimeError(f"Invalid forecast AR(1) process metadata for {task.country}")
    process_settings = load_configs()["baseline"]["calibration"]["process_model"]
    beta_prior_sd = float(
        approximation.get("beta_prior_log_sd", process_settings["log_beta_prior_sd"])
    )
    reporting_prior_sd = float(
        approximation.get(
            "reporting_prior_log_sd",
            process_settings["log_reporting_prior_sd"],
        )
    )

    total_draws = int(max(task.draws, 1) * max(task.grid_n_chains, 1))
    progress_dir = _mcmc_progress_dir(task.output_stem)
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"{task.country}_state_exact_importance.txt"
    progress_file.write_text(
        f"{task.country}: preparing exact-target adaptive importance proposal\n",
        encoding="utf-8",
    )
    importance_settings = settings.get("state_space_importance", {})
    adaptation_candidate_count = int(
        max(64, importance_settings.get("adaptation_candidate_count", 512))
    )
    adaptation_round_candidate_count = int(
        max(64, importance_settings.get("adaptation_round_candidate_count", 512))
    )
    candidate_count = int(max(64, importance_settings.get("candidate_count", 1024)))
    initial_configured_scales = importance_settings.get(
        "initial_proposal_covariance_scales",
        [0.0625, 0.25, 1.0, 4.0],
    )
    initial_covariance_scales = tuple(
        float(value) for value in initial_configured_scales
    )
    configured_scales = importance_settings.get(
        "proposal_covariance_scales",
        [0.25, 0.5, 1.0, 2.0, 4.0],
    )
    covariance_scales = tuple(float(value) for value in configured_scales)
    if any(
        not 0.0 < value <= 16.0
        for value in (*initial_covariance_scales, *covariance_scales)
    ):
        raise ValueError(
            "state-space importance covariance scales must be in (0, 16]"
        )
    adaptation_target_ess = float(
        importance_settings.get(
            "adaptation_target_ess",
            max(64.0, 8.0 * dimension),
        )
    )
    adaptation_regularization = float(
        importance_settings.get("adaptation_covariance_regularization", 0.10)
    )
    prior_base = make_config(country_profile=task.country, load_calibration=False)

    def evaluate_proposal(
        proposal_candidates: np.ndarray,
        proposal_log_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        proposal_in_bounds = np.logical_and(
            np.all(proposal_candidates >= lower, axis=1),
            np.all(proposal_candidates <= upper, axis=1),
        )
        out_of_bounds_fraction = float(1.0 - np.mean(proposal_in_bounds))
        bounded_positions = np.flatnonzero(proposal_in_bounds)
        if len(bounded_positions) < max(32, dimension * 4):
            raise RuntimeError(
                f"State proposal has insufficient bounded support for {task.country}: "
                f"{len(bounded_positions)}/{len(proposal_candidates)} candidates"
            )
        target = np.full(len(proposal_candidates), np.inf, dtype=float)
        def exact_at_position(position: int) -> float:
            return _state_space_exact_negative_log_target(
                proposal_candidates[position],
                runtime_base=runtime_base,
                prior_base=prior_base,
                observed=observed,
                country=task.country,
                coordinate_names=coordinate_names,
                rho=process_rho,
                innovation_sd=process_innovation_sd,
                beta_prior_sd=beta_prior_sd,
                reporting_prior_sd=reporting_prior_sd,
                dispersion=float(task.dispersion),
            )

        positions = [int(position) for position in bounded_positions]
        if int(task.grid_eval_jobs) > 1:
            evaluated = parallel_map(
                exact_at_position,
                positions,
                desc=f"{task.country}_state_exact_target",
                n_jobs=int(task.grid_eval_jobs),
            )
        else:
            evaluated = [exact_at_position(position) for position in positions]
        target[bounded_positions] = np.asarray(evaluated, dtype=float)
        if not np.isfinite(target[proposal_in_bounds]).all():
            raise RuntimeError(f"Non-finite exact state target for {task.country}")
        raw_log_weight = -target - proposal_log_q
        return target, raw_log_weight, proposal_in_bounds, out_of_bounds_fraction

    (
        adaptation_candidates,
        adaptation_log_density,
        _adaptation_source_component,
        _adaptation_scales,
    ) = _state_gaussian_mixture_proposal(
        mean,
        covariance,
        initial_covariance_scales,
        n=adaptation_candidate_count,
        seed=task.seed + 1777,
    )
    (
        _adaptation_target,
        adaptation_raw_log_weight,
        adaptation_in_bounds,
        adaptation_rejection_fraction,
    ) = evaluate_proposal(adaptation_candidates, adaptation_log_density)
    (
        adapted_mean,
        adapted_covariance,
        adaptation_tempering_exponent,
        adaptation_raw_ess,
    ) = _tempered_importance_adaptation(
        adaptation_candidates,
        adaptation_raw_log_weight,
        covariance,
        target_ess=adaptation_target_ess,
        covariance_regularization=adaptation_regularization,
    )
    adaptation_normalized_log_weight = (
        adaptation_raw_log_weight
        - float(scipy_special.logsumexp(adaptation_raw_log_weight))
    )
    adaptation_max_weight = float(
        np.max(np.exp(adaptation_normalized_log_weight))
    )
    progress_file.write_text(
        f"{task.country}: initial proposal evaluated "
        f"ESS={adaptation_raw_ess:.2f}/{adaptation_candidate_count} "
        f"max_weight={adaptation_max_weight:.6f} "
        f"tempering_exponent={adaptation_tempering_exponent:.6f}\n",
        encoding="utf-8",
    )

    max_adaptive_rounds = int(
        max(1, importance_settings.get("max_adaptive_rounds", 3))
    )
    round_ess_history: list[float] = []
    round_max_weight_history: list[float] = []
    round_pareto_k_history: list[float] = []
    round_min_tail_ess_history: list[float] = []
    round_rejection_history: list[float] = []
    round_tempering_history: list[float] = [adaptation_tempering_exponent]
    current_mean = adapted_mean
    current_covariance = adapted_covariance
    for adaptive_round in range(1, max_adaptive_rounds + 1):
        round_candidate_count = (
            candidate_count
            if adaptive_round == max_adaptive_rounds
            else adaptation_round_candidate_count
        )
        (
            candidates,
            proposal_log_density,
            proposal_source_component,
            proposal_scales,
        ) = _state_gaussian_mixture_proposal(
            current_mean,
            current_covariance,
            covariance_scales,
            n=round_candidate_count,
            seed=task.seed + 2777 + (adaptive_round - 1) * 1009,
        )
        (
            exact_negative_log_target,
            raw_log_importance_weight,
            in_bounds,
            rejection_fraction,
        ) = evaluate_proposal(candidates, proposal_log_density)
        log_importance_weight = raw_log_importance_weight.copy()
        log_importance_weight -= float(
            scipy_special.logsumexp(log_importance_weight)
        )
        importance_weight = np.exp(log_importance_weight)
        importance_ess = float(1.0 / np.sum(importance_weight**2))
        importance_max_weight = float(np.max(importance_weight))
        importance_pareto_k = _importance_pareto_shape(
            raw_log_importance_weight
        )
        importance_min_tail_ess = _minimum_importance_tail_ess(
            candidates,
            importance_weight,
        )
        round_ess_history.append(importance_ess)
        round_max_weight_history.append(importance_max_weight)
        round_pareto_k_history.append(importance_pareto_k)
        round_min_tail_ess_history.append(importance_min_tail_ess)
        round_rejection_history.append(rejection_fraction)
        progress_file.write_text(
            f"{task.country}: adaptive round {adaptive_round}/{max_adaptive_rounds} "
            f"ESS={importance_ess:.2f}/{round_candidate_count} "
            f"max_weight={importance_max_weight:.6f} "
            f"pareto_k={importance_pareto_k:.4f} "
            f"min_tail_ess={importance_min_tail_ess:.2f} "
            f"out_of_bounds={rejection_fraction:.4f}\n",
            encoding="utf-8",
        )
        recommended_quality = bool(
            rejection_fraction < STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION
            and importance_ess >= STATE_IMPORTANCE_RECOMMENDED_MIN_ESS
            and importance_max_weight <= STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT
            and importance_pareto_k <= STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K
            and importance_min_tail_ess
            >= STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS
        )
        if recommended_quality or adaptive_round == max_adaptive_rounds:
            break
        (
            current_mean,
            current_covariance,
            next_tempering_exponent,
            _round_raw_ess,
        ) = _tempered_importance_adaptation(
            candidates,
            raw_log_importance_weight,
            current_covariance,
            target_ess=adaptation_target_ess,
            covariance_regularization=adaptation_regularization,
        )
        round_tempering_history.append(next_tempering_exponent)

    localized_mixture_used = False
    localized_tempering_exponent = np.nan
    localized_settings_enabled = bool(
        importance_settings.get("localized_mixture_enabled", True)
    )
    if not recommended_quality and localized_settings_enabled:
        localized_candidate_count = int(
            max(128, importance_settings.get("localized_candidate_count", 2048))
        )
        requested_local_components = int(
            max(2, importance_settings.get("localized_component_count", 8))
        )
        local_covariance_scale = float(
            importance_settings.get("localized_covariance_scale", 0.25)
        )
        if not 0.0 < local_covariance_scale <= 4.0:
            raise ValueError(
                "state_space_importance.localized_covariance_scale must be in (0, 4]"
            )
        (
            localized_global_mean,
            localized_global_covariance,
            localized_tempering_exponent,
            _localized_raw_ess,
        ) = _tempered_importance_adaptation(
            candidates,
            raw_log_importance_weight,
            current_covariance,
            target_ess=max(adaptation_target_ess, 128.0),
            covariance_regularization=adaptation_regularization,
        )
        finite_positions = np.flatnonzero(np.isfinite(raw_log_importance_weight))
        localized_log_weight = (
            localized_tempering_exponent
            * raw_log_importance_weight[finite_positions]
        )
        localized_log_weight -= float(
            scipy_special.logsumexp(localized_log_weight)
        )
        localized_probability = np.exp(localized_log_weight)
        local_component_count = min(
            requested_local_components,
            len(finite_positions),
        )
        localized_rng = np.random.default_rng(task.seed + 7919)
        local_positions = localized_rng.choice(
            finite_positions,
            size=local_component_count,
            replace=False,
            p=localized_probability,
        )
        component_means = np.vstack(
            (
                candidates[local_positions],
                localized_global_mean,
                localized_global_mean,
            )
        )
        local_covariance = _regularized_spd(
            localized_global_covariance * local_covariance_scale
        )
        component_covariances = np.asarray(
            [local_covariance] * local_component_count
            + [
                localized_global_covariance,
                _regularized_spd(localized_global_covariance * 4.0),
            ],
            dtype=float,
        )
        component_weights = np.asarray(
            [0.80 / local_component_count] * local_component_count
            + [0.15, 0.05],
            dtype=float,
        )
        (
            candidates,
            proposal_log_density,
            proposal_source_component,
        ) = _general_gaussian_mixture_proposal(
            component_means,
            component_covariances,
            component_weights,
            n=localized_candidate_count,
            seed=task.seed + 8929,
        )
        proposal_scales = np.asarray(
            [local_covariance_scale] * local_component_count + [1.0, 4.0],
            dtype=float,
        )
        (
            exact_negative_log_target,
            raw_log_importance_weight,
            in_bounds,
            rejection_fraction,
        ) = evaluate_proposal(candidates, proposal_log_density)
        log_importance_weight = raw_log_importance_weight.copy()
        log_importance_weight -= float(
            scipy_special.logsumexp(log_importance_weight)
        )
        importance_weight = np.exp(log_importance_weight)
        importance_ess = float(1.0 / np.sum(importance_weight**2))
        importance_max_weight = float(np.max(importance_weight))
        importance_pareto_k = _importance_pareto_shape(
            raw_log_importance_weight
        )
        importance_min_tail_ess = _minimum_importance_tail_ess(
            candidates,
            importance_weight,
        )
        round_ess_history.append(importance_ess)
        round_max_weight_history.append(importance_max_weight)
        round_pareto_k_history.append(importance_pareto_k)
        round_min_tail_ess_history.append(importance_min_tail_ess)
        round_rejection_history.append(rejection_fraction)
        round_tempering_history.append(localized_tempering_exponent)
        adaptive_round += 1
        localized_mixture_used = True
        progress_file.write_text(
            f"{task.country}: localized mixture round "
            f"ESS={importance_ess:.2f}/{localized_candidate_count} "
            f"max_weight={importance_max_weight:.6f} "
            f"pareto_k={importance_pareto_k:.4f} "
            f"min_tail_ess={importance_min_tail_ess:.2f} "
            f"out_of_bounds={rejection_fraction:.4f}\n",
            encoding="utf-8",
        )

    evaluated_candidate_count = int(len(candidates))

    base_information = np.linalg.pinv(
        _regularized_spd(covariance),
        hermitian=True,
    )
    proposal_delta = candidates - mean
    gauss_newton_quadratic = 0.5 * np.einsum(
        "ij,jk,ik->i",
        proposal_delta,
        base_information,
        proposal_delta,
    )
    importance_entropy_ess = float(
        np.exp(-np.sum(importance_weight * np.log(np.clip(importance_weight, 1e-300, 1.0))))
    )
    require_recommended_importance = bool(
        importance_settings.get("require_recommended_quality", True)
    )
    required_rejection = (
        STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION
        if require_recommended_importance
        else STATE_LAPLACE_FATAL_MAX_BOUNDARY_REJECTION
    )
    required_ess = (
        STATE_IMPORTANCE_RECOMMENDED_MIN_ESS
        if require_recommended_importance
        else STATE_IMPORTANCE_FATAL_MIN_ESS
    )
    required_max_weight = (
        STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT
        if require_recommended_importance
        else STATE_IMPORTANCE_FATAL_MAX_WEIGHT
    )
    required_max_pareto_k = (
        STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K
        if require_recommended_importance
        else STATE_IMPORTANCE_FATAL_MAX_PARETO_K
    )
    required_min_tail_ess = (
        STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS
        if require_recommended_importance
        else STATE_IMPORTANCE_FATAL_MIN_TAIL_ESS
    )
    importance_quality_failures = _state_importance_quality_failures(
        effective_sample_size=importance_ess,
        maximum_weight=importance_max_weight,
        pareto_k=importance_pareto_k,
        minimum_tail_ess=importance_min_tail_ess,
        out_of_bounds_fraction=rejection_fraction,
        require_recommended=require_recommended_importance,
    )

    candidate_audit = pd.DataFrame(
        candidates,
        columns=coordinate_names,
    )
    candidate_audit.insert(0, "country", task.country)
    candidate_audit.insert(1, "candidate_index", np.arange(len(candidates)))
    candidate_audit["exact_negative_log_target"] = exact_negative_log_target
    candidate_audit["proposal_log_density"] = proposal_log_density
    candidate_audit["raw_log_importance_weight"] = raw_log_importance_weight
    candidate_audit["normalized_importance_weight"] = importance_weight
    candidate_audit["in_bounds"] = in_bounds
    candidate_audit["proposal_component"] = proposal_source_component
    candidate_audit["proposal_covariance_scale"] = proposal_scales[
        proposal_source_component
    ]
    candidate_audit["task_seed"] = int(task.seed)
    candidate_audit["quality_standard"] = (
        "recommended" if require_recommended_importance else "fatal_floor"
    )
    candidate_audit["quality_gate_pass"] = not importance_quality_failures
    candidate_audit["quality_gate_failures"] = ";".join(
        importance_quality_failures
    )
    candidate_audit["global_effective_sample_size"] = float(importance_ess)
    candidate_audit["global_maximum_weight"] = float(importance_max_weight)
    candidate_audit["global_pareto_k"] = float(importance_pareto_k)
    candidate_audit["global_minimum_tail_ess"] = float(
        importance_min_tail_ess
    )
    candidate_audit["global_out_of_bounds_fraction"] = float(
        rejection_fraction
    )
    write_dataframe(
        candidate_audit,
        project_path(
            "outputs",
            "diagnostics",
            f"{task.output_stem}_{task.country}_state_importance_candidates.csv",
        ),
    )
    if importance_quality_failures:
        raise RuntimeError(
            f"Exact-target state importance gate failed for {task.country}: "
            f"ESS={importance_ess:.2f} (minimum {required_ess:.0f}), "
            f"max_weight={importance_max_weight:.6f} "
            f"(maximum {required_max_weight:.3f}), "
            f"pareto_k={importance_pareto_k:.4f} "
            f"(maximum {required_max_pareto_k:.2f}), "
            f"min_tail_ess={importance_min_tail_ess:.2f} "
            f"(minimum {required_min_tail_ess:.0f}), "
            f"out_of_bounds={rejection_fraction:.3f} "
            f"(maximum {required_rejection:.3f}); "
            f"failed_metrics={importance_quality_failures}; "
            f"quality_standard={'recommended' if require_recommended_importance else 'fatal_floor'}"
        )
    state_rng = np.random.default_rng(task.seed + 1999)
    selected_candidate_indices = _stratified_resample_indices(
        importance_weight,
        total_draws,
        state_rng,
    )
    state_draws = candidates[selected_candidate_indices]
    selected_exact_target = exact_negative_log_target[selected_candidate_indices]
    selected_proposal_log_density = proposal_log_density[selected_candidate_indices]
    selected_source_weight = importance_weight[selected_candidate_indices]
    selected_source_scale = proposal_scales[
        proposal_source_component[selected_candidate_indices]
    ]
    exact_delta = exact_negative_log_target[in_bounds]
    exact_delta = exact_delta - float(np.min(exact_delta))
    bounded_quadratic = gauss_newton_quadratic[in_bounds]
    exact_quadratic_correlation = (
        float(np.corrcoef(exact_delta, bounded_quadratic)[0, 1])
        if len(exact_delta) > 2
        and float(np.std(exact_delta)) > 0.0
        and float(np.std(bounded_quadratic)) > 0.0
        else np.nan
    )

    process_draw_columns: dict[str, np.ndarray] = {
        name: state_draws[:, 2 + index]
        for index, name in enumerate(latent_names)
    }
    process_rng = np.random.default_rng(task.seed + 2389)
    previous_state = state_draws[:, -1].copy()
    for year in future_years:
        previous_state = (
            process_rho * previous_state
            + process_rng.normal(0.0, process_innovation_sd, size=total_draws)
        )
        process_draw_columns[f"log_beta_process_{year}"] = previous_state.copy()

    priors = settings["priors"]
    structural_draws = int(max(task.importance_nuisance_draws, 1))
    shared_seed = int(settings.get("shared_structural_seed", 20260712))
    unit_design = _latin_hypercube(
        structural_draws,
        len(IMPORTANCE_NUISANCE_INDICES),
        shared_seed,
    )
    nuisance_vectors = np.vstack(
        [
            _nuisance_vector_from_unit_cube(
                row,
                calibrated_start,
                runtime_base,
                priors,
            )
            for row in unit_design
        ]
    )
    structural_rng = np.random.default_rng(shared_seed + 41)
    structural_sequence: list[int] = []
    while len(structural_sequence) < total_draws:
        structural_sequence.extend(
            structural_rng.permutation(structural_draws).astype(int).tolist()
        )

    rows: list[dict[str, Any]] = []
    chain_count = int(max(task.grid_n_chains, 1))
    draws_per_chain = int(max(task.draws, 1))
    for position, structural_draw_id in enumerate(structural_sequence[:total_draws]):
        coordinate = state_draws[position]
        vector = nuisance_vectors[structural_draw_id].copy()
        vector[0] = float(coordinate[0])
        vector[1] = _encode_reporting_coordinate(
            float(coordinate[0]),
            float(coordinate[1]),
            priors,
        )
        sample = _sample_from_vector(vector, priors)
        row: dict[str, Any] = {
            "country": task.country,
            "chain": int(position // draws_per_chain + 1),
            "draw": int(position % draws_per_chain + 1),
            "step": int(position % draws_per_chain + 1),
            "sampling_method": STATE_SPACE_LAPLACE_CUT_SAMPLER,
            "inference_structure": "reference_structure_state_space_exact_importance_cut",
            "uncertainty_target": (
                "conditional_state_posterior_plus_external_structural_prior_sensitivity"
            ),
            "structural_draw_id": int(structural_draw_id),
            "nuisance_index": int(structural_draw_id),
            "posterior_log_prob": float(-selected_exact_target[position]),
            "exact_state_log_target": float(-selected_exact_target[position]),
            "state_proposal_log_density": float(
                selected_proposal_log_density[position]
            ),
            # Retained as a compatibility alias for downstream readers. It is
            # now the normalized mixture log density, not a Laplace target.
            "laplace_log_kernel": float(selected_proposal_log_density[position]),
            "posterior_log_prob_interpretation": "exact_conditional_state_log_target_up_to_constant",
            "source_normalized_weight": float(selected_source_weight[position]),
            "accepted_fraction": 1.0,
            "state_posterior_rank": int(rank),
            "state_posterior_dimension": int(dimension),
            "state_posterior_condition_number": float(condition_number),
            "state_laplace_boundary_rejection_fraction": float(rejection_fraction),
            "state_proposal_out_of_bounds_fraction": float(rejection_fraction),
            "state_importance_candidate_count": int(evaluated_candidate_count),
            "state_importance_bounded_candidate_count": int(np.sum(in_bounds)),
            "state_importance_adaptation_candidate_count": int(
                adaptation_candidate_count
            ),
            "state_importance_adaptation_round_candidate_count": int(
                adaptation_round_candidate_count
            ),
            "state_importance_adaptation_bounded_candidate_count": int(
                np.sum(adaptation_in_bounds)
            ),
            "state_importance_adaptation_raw_ess": float(adaptation_raw_ess),
            "state_importance_adaptation_max_weight": float(
                adaptation_max_weight
            ),
            "state_importance_adaptation_tempering_exponent": float(
                adaptation_tempering_exponent
            ),
            "state_importance_adaptation_out_of_bounds_fraction": float(
                adaptation_rejection_fraction
            ),
            "state_importance_adaptation_covariance_regularization": float(
                adaptation_regularization
            ),
            "state_importance_adaptive_round_count": int(adaptive_round),
            "state_importance_localized_mixture_used": bool(
                localized_mixture_used
            ),
            "state_importance_localized_tempering_exponent": float(
                localized_tempering_exponent
            ),
            "state_importance_round_ess_history": ",".join(
                f"{value:.8g}" for value in round_ess_history
            ),
            "state_importance_round_max_weight_history": ",".join(
                f"{value:.8g}" for value in round_max_weight_history
            ),
            "state_importance_round_pareto_k_history": ",".join(
                f"{value:.8g}" for value in round_pareto_k_history
            ),
            "state_importance_round_min_tail_ess_history": ",".join(
                f"{value:.8g}" for value in round_min_tail_ess_history
            ),
            "state_importance_round_out_of_bounds_history": ",".join(
                f"{value:.8g}" for value in round_rejection_history
            ),
            "state_importance_round_tempering_exponent_history": ",".join(
                f"{value:.8g}" for value in round_tempering_history
            ),
            "state_importance_effective_sample_size": float(importance_ess),
            "state_importance_ess_fraction": float(
                importance_ess / evaluated_candidate_count
            ),
            "state_importance_entropy_effective_sample_size": float(
                importance_entropy_ess
            ),
            "state_importance_max_weight": float(importance_max_weight),
            "state_importance_pareto_k": float(importance_pareto_k),
            "state_importance_min_tail_ess": float(
                importance_min_tail_ess
            ),
            "state_importance_proposal_covariance_scales": ",".join(
                f"{value:.8g}" for value in proposal_scales
            ),
            "state_importance_source_covariance_scale": float(
                selected_source_scale[position]
            ),
            "state_importance_exact_quadratic_correlation": float(
                exact_quadratic_correlation
            ),
            "state_importance_source_candidate_index": int(
                selected_candidate_indices[position]
            ),
            "state_importance_task_seed": int(task.seed),
            "state_importance_inner_evaluation_jobs": int(task.grid_eval_jobs),
            "structural_prior_design_size": int(structural_draws),
            "forecast_process_rho": float(process_rho),
            "forecast_process_innovation_sd": float(process_innovation_sd),
            "forecast_process_years": int(len(future_years)),
            "historical_process_end_year": int(latent_years[-1]),
        }
        row.update(sample)
        row.update(
            {
                name: float(values[position])
                for name, values in process_draw_columns.items()
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame["chain"].max() != chain_count:
        raise RuntimeError("Internal state-space draw chain layout error")
    progress_file.write_text(
        f"{task.country}: accepted exact-target importance distribution "
        f"ESS={importance_ess:.2f}/{evaluated_candidate_count} "
        f"max_weight={importance_max_weight:.6f} "
        f"pareto_k={importance_pareto_k:.4f} "
        f"min_tail_ess={importance_min_tail_ess:.2f}\n",
        encoding="utf-8",
    )
    return frame


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
    sampler = str(task.sampler).lower()
    if sampler not in SUPPORTED_MULTIPARAMETER_SAMPLERS | {"beta_grid"}:
        raise ValueError(f"Unsupported Bayesian sampler: {task.sampler}")
    base = make_config(
        vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
        resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
        country_profile=task.country,
        # State-space propagation consumes the one-use calibration artifact.
        # Alternative likelihood samplers instead target the deterministic
        # no-process model from pre-surveillance priors and must not load a
        # data-fitted latent path.
        load_calibration=(sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER),
    )
    observed = _country_observed(
        task.country,
        interval=str(task.likelihood_observation_frequency),
    )
    runtime_base = calibration_runtime_config(base, observed)
    solver_mode = str(task.solver_mode or "mcmc_fast").lower()
    if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER and solver_mode != "calibration":
        raise ValueError(
            "The exact-target state-space route must use solver_mode='calibration' "
            "because its stored MAP/covariance and importance target must share the "
            "same numerical likelihood. Use the numerical-fidelity gate to benchmark "
            "faster solvers, not to change the inference target during propagation."
        )
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

    process_interpretation = str(
        runtime_base.get("transmission", {})
        .get("log_beta_time_variation", {})
        .get("interpretation", "")
    )
    if (
        sampler != STATE_SPACE_LAPLACE_CUT_SAMPLER
        and process_interpretation.startswith("latent_AR1_process_")
    ):
        raise RuntimeError(
            "Refusing to evaluate the surveillance likelihood a second time while a "
            "latent transmission path fitted to those observations is fixed in the "
            "runtime. Use state_space_exact_importance_cut, or construct a genuinely joint "
            "state/parameter likelihood without loading the fitted path."
        )

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
    if (
        sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
        and (
            task.beta_prior_log_sd is not None
            or task.reporting_prior_log_sd is not None
        )
    ):
        raise ValueError(
            "beta/reporting prior-width overrides cannot modify a stored state-space "
            "posterior approximation; recalibrate with explicit process_model priors"
        )
    registry_parameter_names = (
        BAYESIAN_SHARED_PARAMETER_NAMES
        if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
        else BAYESIAN_PARAMETER_NAMES
    )
    settings["priors"]["_distribution_specs"] = resolve_bayesian_prior_specs(
        configs.get("parameter_distributions", {}),
        runtime_base,
        parameter_names=registry_parameter_names,
        prior_sd_scale=task.prior_sd_scale,
        beta_prior_log_sd=task.beta_prior_log_sd,
        reporting_prior_log_sd=task.reporting_prior_log_sd,
        ve_prior_sd=task.ve_prior_sd,
        rel_asym_prior_sd=task.rel_asym_prior_sd,
        fitness_prior_sd=task.fitness_prior_sd,
    )
    # The fitness transform needs the same support as the normalized registry
    # density.  Retain these aliases for legacy helper call sites.
    fitness_spec = settings["priors"]["_distribution_specs"]["fitness_R"]
    settings["priors"]["fitness_R"]["min"] = float(fitness_spec["low"])
    settings["priors"]["fitness_R"]["max"] = float(fitness_spec["high"])

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
        start_at_prior_centers=False,
    )
    if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER:
        if fixed_indices:
            raise ValueError(
                "state_space_exact_importance_cut propagates all registered structural priors "
                "and cannot fix posterior dimensions"
            )
        return _run_state_space_exact_importance_cut(
            task,
            runtime_base,
            observed,
            settings,
            calibrated_start,
        )
    prior_centered_start = _initial_vector(
        runtime_base,
        enable_trend=task.enable_time_varying_reporting,
        priors=settings["priors"],
        start_at_prior_centers=(sampler != "beta_grid"),
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
        current = prior_centered_start.copy()

    if fixed_indices:
        for idx in fixed_indices:
            current[idx] = calibrated_start[idx]

    # Indices: 0=log_beta_S, 1=reporting coordinate, 2=logit_VE_sus,
    #          3=logit_VE_inf, 4=logit_VE_dur, 5=logit_rel_inf_asym,
    #          6=log_inf_dur_sym, 7=log_inf_dur_asym, 8=logit_fitness_R
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
    if sampler == JOINT_IMPORTANCE_SAMPLER:
        if fixed_indices:
            fixed_names = [PARAMETER_SAMPLE_COLUMNS[idx] for idx in sorted(fixed_indices)]
            raise ValueError(
                "joint_importance integrates all configured sampled dimensions and does not support fixed sampled parameters: "
                f"{fixed_names}"
            )
        return _run_joint_importance(
            task,
            runtime_base,
            observed,
            settings,
            calibrated_start,
        )
    if sampler == SMC_SAMPLER:
        if fixed_indices:
            fixed_names = [PARAMETER_SAMPLE_COLUMNS[idx] for idx in sorted(fixed_indices)]
            raise ValueError(
                "smc is a legacy full-feedback posterior sensitivity sampler and does "
                "not support fixed sampled parameters: "
                f"{fixed_names}"
            )
        return _run_smc(
            task,
            runtime_base,
            observed,
            settings,
            calibrated_start,
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
            trial = prior_centered_start.copy()
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
            proposal = _reflect_component_into_bounds(proposal, component, settings["priors"])
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
            proposal = _reflect_component_into_bounds(proposal, component, settings["priors"])

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
        "VE_dur",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_symptomatic",
        "infectious_duration_asymptomatic",
        "fitness_R",
    )


def _posterior_predictive_scenarios(
    samples: pd.DataFrame,
    draws_per_country: int,
    *,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    configs = load_configs()
    scenarios = []
    rng = np.random.default_rng(
        int(
            configs["baseline"]["bayesian_uncertainty"].get("random_seed", 20260510)
            if random_seed is None
            else random_seed
        )
        + 917
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
            row_values = row._asdict()
            sample = {}
            for name in _sample_columns():
                val = getattr(row, name, None)
                if val is not None:
                    sample[name] = float(val)
                else:
                    # Fallback for legacy samples without trend
                    sample[name] = 1.0 if name == "reporting_trend_end_multiplier" else 0.0
            sample.update(
                {
                    str(name): float(value)
                    for name, value in row_values.items()
                    if str(name).startswith("log_beta_process_")
                    and value is not None
                    and pd.notna(value)
                }
            )
            config = make_config(
                vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
                resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
                country_profile=country,
            )
            state_exact_route = str(
                getattr(row, "sampling_method", "")
            ).lower() == STATE_SPACE_LAPLACE_CUT_SAMPLER
            if state_exact_route:
                # Structural rows are an external-prior sensitivity design,
                # not likelihood-updated country states. Calibration predictive
                # checks therefore hold the reference structure fixed and vary
                # only beta, reporting, dated states, and future innovations.
                sample.update(
                    {
                        "VE_sus": float(config["vaccine"]["VE_sus"]),
                        "VE_inf": float(config["vaccine"]["VE_inf"]),
                        "VE_dur": float(config["vaccine"].get("VE_dur", 0.0)),
                        "relative_infectiousness_asymptomatic": float(
                            config["transmission"][
                                "relative_infectiousness_asymptomatic"
                            ]
                        ),
                        "infectious_duration_symptomatic": float(
                            config["natural_history"][
                                "infectious_duration_symptomatic"
                            ]
                        ),
                        "infectious_duration_asymptomatic": float(
                            config["natural_history"][
                                "infectious_duration_asymptomatic"
                            ]
                        ),
                        "fitness_R": float(config["transmission"]["fitness_R"]),
                    }
                )
            config = _apply_sample(config, sample)
            metadata = {
                "country": country,
                "posterior_draw": draw_idx,
                "posterior_chain": int(getattr(row, "chain")),
                "posterior_log_prob": float(getattr(row, "posterior_log_prob")),
                "predictive_uncertainty_scope": (
                    "conditional_state_process_reference_structure"
                    if state_exact_route
                    else "legacy_sampler_parameter_distribution"
                ),
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
    inference_structures = set(
        samples.get("inference_structure", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.lower()
    )
    conditional_state_route = inference_structures == {
        "reference_structure_state_space_exact_importance_cut"
    }
    interval_type = (
        "95% conditional uncertainty interval"
        if conditional_state_route
        else "95% alternative-target posterior interval (diagnostic)"
    )
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
                    "uncertainty_median": float(median),
                    "uncertainty_interval_low": float(low),
                    "uncertainty_interval_high": float(high),
                    "uncertainty_draws": int(len(values)),
                    "interval_type": interval_type,
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
                    "uncertainty_median": float(median),
                    "uncertainty_interval_low": float(low),
                    "uncertainty_interval_high": float(high),
                    "uncertainty_draws": int(len(values)),
                    "parameter_source_role": (
                        "conditional_state_target"
                        if conditional_state_route
                        and parameter in {"beta_S", "reporting_multiplier"}
                        else "shared_external_prior_design"
                        if conditional_state_route
                        else "alternative_target_parameter"
                    ),
                    "interval_type": interval_type,
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
    posterior predictive summary, writing combined-uncertainty interval artifacts.
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
    """Run stochastic overlay at multiple k values to show interval sensitivity.

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


def _compute_importance_diagnostics(
    samples: pd.DataFrame,
    parameter_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for country, group in samples.groupby("country", sort=False):
        quality = group.iloc[0]
        inference_structure = str(
            quality.get("inference_structure", "country_joint_feedback")
        ).lower()
        is_modular_cut = inference_structure in {
            "modular_cut",
            "modular_hierarchical_cut",
        }
        importance_ess = float(quality.get("importance_effective_sample_size", np.nan))
        nuisance_ess = float(quality.get("importance_nuisance_effective_sample_size", np.nan))
        max_weight = float(quality.get("importance_max_weight", np.nan))
        edge_weight = float(quality.get("importance_edge_weight", np.nan))
        candidate_count = int(float(quality.get("importance_candidate_count", 0.0) or 0.0))
        n_chains = int(group["chain"].nunique()) if "chain" in group.columns else 1
        total_draws = int(len(group))
        modular_metrics = {
            name: float(quality.get(name, np.nan))
            for name in (
                "modular_cut_structural_draw_count",
                "modular_cut_conditional_incomplete_grid_count",
                "modular_cut_conditional_fatal_failure_count",
                "modular_cut_conditional_recommended_failure_count",
                "modular_cut_conditional_min_effective_grid_points",
                "modular_cut_conditional_min_beta_effective_grid_points",
                "modular_cut_conditional_min_reporting_effective_grid_points",
                "modular_cut_conditional_max_single_weight",
                "modular_cut_conditional_max_edge_weight",
            )
        }
        if is_modular_cut:
            structural_draws = modular_metrics["modular_cut_structural_draw_count"]
            incomplete_grids = modular_metrics[
                "modular_cut_conditional_incomplete_grid_count"
            ]
            fatal_failures = modular_metrics[
                "modular_cut_conditional_fatal_failure_count"
            ]
            recommended_failures = modular_metrics[
                "modular_cut_conditional_recommended_failure_count"
            ]
            conditional_ess = modular_metrics[
                "modular_cut_conditional_min_effective_grid_points"
            ]
            beta_axis_ess = modular_metrics[
                "modular_cut_conditional_min_beta_effective_grid_points"
            ]
            reporting_axis_ess = modular_metrics[
                "modular_cut_conditional_min_reporting_effective_grid_points"
            ]
            conditional_max_weight = modular_metrics[
                "modular_cut_conditional_max_single_weight"
            ]
            conditional_edge_weight = modular_metrics[
                "modular_cut_conditional_max_edge_weight"
            ]
            converged = (
                np.isfinite(structural_draws)
                and structural_draws >= MODULAR_CUT_FATAL_MIN_STRUCTURAL_DRAWS
                and np.isfinite(incomplete_grids)
                and incomplete_grids == 0.0
                and np.isfinite(fatal_failures)
                and fatal_failures == 0.0
                and np.isfinite(conditional_ess)
                and conditional_ess >= MODULAR_CUT_FATAL_MIN_CONDITIONAL_EFFECTIVE_POINTS
                and np.isfinite(beta_axis_ess)
                and beta_axis_ess >= MODULAR_CUT_FATAL_MIN_AXIS_EFFECTIVE_POINTS
                and np.isfinite(reporting_axis_ess)
                and reporting_axis_ess >= MODULAR_CUT_FATAL_MIN_AXIS_EFFECTIVE_POINTS
                and np.isfinite(conditional_max_weight)
                and conditional_max_weight <= MODULAR_CUT_FATAL_MAX_CONDITIONAL_WEIGHT
                and np.isfinite(conditional_edge_weight)
                and conditional_edge_weight <= MODULAR_CUT_FATAL_MAX_CONDITIONAL_EDGE_WEIGHT
            )
            recommended_converged = (
                np.isfinite(structural_draws)
                and structural_draws >= MODULAR_CUT_RECOMMENDED_MIN_STRUCTURAL_DRAWS
                and np.isfinite(incomplete_grids)
                and incomplete_grids == 0.0
                and np.isfinite(recommended_failures)
                and recommended_failures == 0.0
                and np.isfinite(conditional_ess)
                and conditional_ess >= MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS
                and np.isfinite(beta_axis_ess)
                and beta_axis_ess >= MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS
                and np.isfinite(reporting_axis_ess)
                and reporting_axis_ess >= MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS
                and np.isfinite(conditional_max_weight)
                and conditional_max_weight <= MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT
                and np.isfinite(conditional_edge_weight)
                and conditional_edge_weight <= MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT
            )
            diagnostic_method = "modular_cut_conditional_grid_quality"
            diagnostic_bulk_ess = conditional_ess
        else:
            converged = (
                np.isfinite(importance_ess)
                and importance_ess >= MIN_FATAL_BULK_ESS
                and np.isfinite(nuisance_ess)
                and nuisance_ess >= IMPORTANCE_FATAL_NUISANCE_ESS
                and np.isfinite(max_weight)
                and max_weight <= IMPORTANCE_FATAL_MAX_WEIGHT
                and np.isfinite(edge_weight)
                and edge_weight <= IMPORTANCE_FATAL_EDGE_WEIGHT
            )
            recommended_converged = (
                np.isfinite(importance_ess)
                and importance_ess >= MIN_RECOMMENDED_BULK_ESS
                and np.isfinite(nuisance_ess)
                and nuisance_ess >= IMPORTANCE_RECOMMENDED_NUISANCE_ESS
                and np.isfinite(max_weight)
                and max_weight <= IMPORTANCE_RECOMMENDED_MAX_WEIGHT
                and np.isfinite(edge_weight)
                and edge_weight <= IMPORTANCE_RECOMMENDED_EDGE_WEIGHT
            )
            diagnostic_method = "joint_importance_weight_quality"
            diagnostic_bulk_ess = importance_ess
        for parameter in parameter_columns:
            values = pd.to_numeric(group[parameter], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            rows.append(
                {
                    "country": country,
                    "parameter": parameter,
                    "diagnostic_method": diagnostic_method,
                    "inference_structure": inference_structure,
                    "rhat": 1.0,
                    "rhat_rank": 1.0,
                    "bulk_ess": diagnostic_bulk_ess,
                    "tail_ess": diagnostic_bulk_ess,
                    "importance_effective_sample_size": importance_ess,
                    "importance_nuisance_effective_sample_size": nuisance_ess,
                    "importance_max_weight": max_weight,
                    "importance_edge_weight": edge_weight,
                    "importance_candidate_count": candidate_count,
                    **modular_metrics,
                    "n_chains": n_chains,
                    "total_draws": total_draws,
                    "converged": bool(converged),
                    "recommended_converged": bool(recommended_converged),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def _compute_smc_diagnostics(
    samples: pd.DataFrame,
    parameter_columns: tuple[str, ...],
) -> pd.DataFrame:
    try:
        rank_frames: list[pd.DataFrame] = []
        for _, country_samples in samples.groupby("country", sort=False):
            country_parameters = tuple(
                parameter
                for parameter in parameter_columns
                if parameter in country_samples.columns
                and pd.to_numeric(
                    country_samples[parameter], errors="coerce"
                ).notna().any()
            )
            if country_parameters:
                rank_frames.append(
                    compute_diagnostics(
                        country_samples,
                        parameter_columns=country_parameters,
                        chain_column="chain",
                        country_column="country",
                    )
                )
        rank_diagnostics = (
            pd.concat(rank_frames, ignore_index=True)
            if rank_frames
            else pd.DataFrame()
        )
    except Exception:
        rank_diagnostics = pd.DataFrame()
    rank_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not rank_diagnostics.empty:
        for row in rank_diagnostics.to_dict("records"):
            rank_lookup[(str(row["country"]), str(row["parameter"]))] = row

    rows: list[dict[str, Any]] = []
    for country, group in samples.groupby("country", sort=False):
        country_key = str(country)
        observed_chain_count = int(group["chain"].nunique()) if "chain" in group.columns else 1
        per_chain = (
            group.sort_values(["chain", "draw"])
            .groupby("chain", sort=False)
            .first()
            .reset_index()
        )
        particle_count = int(pd.to_numeric(per_chain.get("smc_particle_count", pd.Series([0])), errors="coerce").max())
        particle_count = max(particle_count, 1)
        min_stage_ess = float(pd.to_numeric(per_chain.get("smc_min_ess"), errors="coerce").min())
        final_ess = float(pd.to_numeric(per_chain.get("smc_final_ess"), errors="coerce").min())
        min_ess_fraction = float(pd.to_numeric(per_chain.get("smc_min_ess_fraction"), errors="coerce").min())
        final_ess_fraction = float(pd.to_numeric(per_chain.get("smc_final_ess_fraction"), errors="coerce").min())
        max_weight = float(pd.to_numeric(per_chain.get("smc_max_weight"), errors="coerce").max())
        final_max_weight = float(pd.to_numeric(per_chain.get("smc_final_max_weight"), errors="coerce").max())
        combined_particle_count = int(
            pd.to_numeric(per_chain.get("smc_combined_particle_count", pd.Series([particle_count])), errors="coerce").max()
        )
        combined_particle_count = max(combined_particle_count, particle_count)
        combined_particle_ess = float(
            pd.to_numeric(per_chain.get("smc_combined_particle_ess", pd.Series([final_ess])), errors="coerce").min()
        )
        combined_particle_ess_fraction = float(
            pd.to_numeric(
                per_chain.get("smc_combined_particle_ess_fraction", pd.Series([final_ess_fraction])),
                errors="coerce",
            ).min()
        )
        combined_max_weight = float(
            pd.to_numeric(per_chain.get("smc_combined_max_weight", pd.Series([final_max_weight])), errors="coerce").max()
        )
        island_count = int(
            pd.to_numeric(per_chain.get("smc_island_count", pd.Series([observed_chain_count])), errors="coerce").max()
        )
        island_count = max(island_count, 1)
        island_ess = float(
            pd.to_numeric(per_chain.get("smc_island_ess", pd.Series([float(island_count)])), errors="coerce").min()
        )
        island_ess_fraction = float(
            pd.to_numeric(per_chain.get("smc_island_ess_fraction", pd.Series([1.0])), errors="coerce").min()
        )
        max_island_weight = float(
            pd.to_numeric(per_chain.get("smc_max_island_weight", pd.Series([1.0 / island_count])), errors="coerce").max()
        )
        log_evidence_sd = float(
            pd.to_numeric(per_chain.get("smc_log_evidence_sd", pd.Series([np.nan])), errors="coerce").max()
        )
        log_evidence_range = float(
            pd.to_numeric(per_chain.get("smc_log_evidence_range", pd.Series([np.nan])), errors="coerce").max()
        )
        unique_ancestor_fraction = float(
            pd.to_numeric(per_chain.get("smc_unique_ancestor_fraction"), errors="coerce").min()
        )
        unique_particle_fraction = float(
            pd.to_numeric(per_chain.get("smc_unique_particle_fraction"), errors="coerce").min()
        )
        stage_count = int(pd.to_numeric(per_chain.get("smc_stage_count"), errors="coerce").max())
        final_temperature = float(pd.to_numeric(per_chain.get("smc_final_temperature"), errors="coerce").min())
        move_acceptance = float(
            pd.to_numeric(per_chain.get("smc_move_acceptance_fraction"), errors="coerce").mean()
        )
        reached_final = bool(
            "smc_reached_final_temperature" in per_chain.columns
            and per_chain["smc_reached_final_temperature"].astype(bool).all()
        )
        fatal_ess_floor = max(MIN_FATAL_BULK_ESS, SMC_FATAL_ESS_FRACTION * particle_count)
        recommended_ess_floor = max(MIN_RECOMMENDED_BULK_ESS, SMC_RECOMMENDED_ESS_FRACTION * particle_count)
        recommended_island_ess_floor = max(
            SMC_FATAL_ISLAND_ESS,
            SMC_RECOMMENDED_ISLAND_ESS_FRACTION * island_count,
        )
        smc_floor_ok = (
            reached_final
            and np.isfinite(min_stage_ess)
            and min_stage_ess >= SMC_FATAL_ESS_FRACTION * particle_count
            and np.isfinite(final_ess)
            and final_ess >= fatal_ess_floor
            and np.isfinite(combined_particle_ess)
            and combined_particle_ess >= fatal_ess_floor
            and np.isfinite(max_weight)
            and max_weight <= SMC_FATAL_MAX_WEIGHT
            and np.isfinite(unique_particle_fraction)
            and unique_particle_fraction >= SMC_FATAL_UNIQUE_PARTICLE_FRACTION
            and np.isfinite(island_ess)
            and island_ess >= SMC_FATAL_ISLAND_ESS
            and np.isfinite(max_island_weight)
            and max_island_weight <= SMC_FATAL_MAX_ISLAND_WEIGHT
        )
        smc_recommended_ok = (
            reached_final
            and np.isfinite(min_stage_ess)
            and min_stage_ess >= SMC_RECOMMENDED_ESS_FRACTION * particle_count
            and np.isfinite(final_ess)
            and final_ess >= recommended_ess_floor
            and np.isfinite(combined_particle_ess)
            and combined_particle_ess >= MIN_RECOMMENDED_BULK_ESS
            and np.isfinite(max_weight)
            and max_weight <= SMC_RECOMMENDED_MAX_WEIGHT
            and np.isfinite(unique_particle_fraction)
            and unique_particle_fraction >= SMC_RECOMMENDED_UNIQUE_PARTICLE_FRACTION
            and np.isfinite(island_ess)
            and island_ess >= recommended_island_ess_floor
            and np.isfinite(max_island_weight)
            and max_island_weight <= SMC_RECOMMENDED_MAX_ISLAND_WEIGHT
        )
        n_chains = observed_chain_count
        total_draws = int(len(group))
        for parameter in parameter_columns:
            values = pd.to_numeric(group[parameter], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            rank = rank_lookup.get((country_key, str(parameter)), {})
            rank_rhat = float(rank.get("rhat", np.inf))
            rank_rhat_rank = float(rank.get("rhat_rank", np.inf))
            rank_bulk = float(rank.get("bulk_ess", 0.0))
            rank_tail = float(rank.get("tail_ess", 0.0))
            if not np.isfinite(rank_rhat_rank):
                rank_rhat = np.inf
                rank_rhat_rank = np.inf
            if not np.isfinite(rank_bulk):
                rank_bulk = 0.0
            if not np.isfinite(rank_tail):
                rank_tail = 0.0
            conservative_bulk = float(min(rank_bulk, final_ess)) if np.isfinite(final_ess) else 0.0
            conservative_tail = float(min(rank_tail, final_ess)) if np.isfinite(final_ess) else 0.0
            rank_floor_ok = (
                np.isfinite(rank_rhat_rank)
                and rank_rhat_rank <= MAX_FATAL_RHAT
                and conservative_bulk >= MIN_FATAL_BULK_ESS
                and conservative_tail >= MIN_FATAL_TAIL_ESS
            )
            rank_recommended_ok = (
                np.isfinite(rank_rhat_rank)
                and rank_rhat_rank <= MAX_RECOMMENDED_RHAT
                and conservative_bulk >= MIN_RECOMMENDED_BULK_ESS
                and conservative_tail >= MIN_RECOMMENDED_TAIL_ESS
            )
            rows.append(
                {
                    "country": country,
                    "parameter": parameter,
                    "diagnostic_method": "tempered_smc_rank_and_particle_quality",
                    "rhat": rank_rhat,
                    "rhat_rank": rank_rhat_rank,
                    "bulk_ess": conservative_bulk,
                    "tail_ess": conservative_tail,
                    "smc_particle_count": particle_count,
                    "smc_stage_count": stage_count,
                    "smc_final_temperature": final_temperature,
                    "smc_min_ess": min_stage_ess,
                    "smc_final_ess": final_ess,
                    "smc_min_ess_fraction": min_ess_fraction,
                    "smc_final_ess_fraction": final_ess_fraction,
                    "smc_max_weight": max_weight,
                    "smc_final_max_weight": final_max_weight,
                    "smc_combined_particle_count": combined_particle_count,
                    "smc_combined_particle_ess": combined_particle_ess,
                    "smc_combined_particle_ess_fraction": combined_particle_ess_fraction,
                    "smc_combined_max_weight": combined_max_weight,
                    "smc_island_count": island_count,
                    "smc_island_ess": island_ess,
                    "smc_island_ess_fraction": island_ess_fraction,
                    "smc_max_island_weight": max_island_weight,
                    "smc_log_evidence_sd": log_evidence_sd,
                    "smc_log_evidence_range": log_evidence_range,
                    "smc_recommended_island_ess_floor": recommended_island_ess_floor,
                    "smc_unique_ancestor_fraction": unique_ancestor_fraction,
                    "smc_unique_particle_fraction": unique_particle_fraction,
                    "smc_move_acceptance_fraction": move_acceptance,
                    "smc_reached_final_temperature": reached_final,
                    "smc_fatal_ess_floor": fatal_ess_floor,
                    "smc_recommended_ess_floor": recommended_ess_floor,
                    "n_chains": n_chains,
                    "total_draws": total_draws,
                    "converged": bool(smc_floor_ok and rank_floor_ok),
                    "recommended_converged": bool(smc_recommended_ok and rank_recommended_ok),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _compute_state_space_exact_importance_diagnostics(
    samples: pd.DataFrame,
    parameter_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Audit exact-target importance/design quality without pretending MCMC."""

    rows: list[dict[str, Any]] = []
    structural_names = set(PARAMETER_SAMPLE_COLUMNS[2:])
    for country, group in samples.groupby("country", sort=False):
        first = group.iloc[0]
        rank = int(first["state_posterior_rank"])
        dimension = int(first["state_posterior_dimension"])
        condition = float(first["state_posterior_condition_number"])
        rejection = float(first["state_laplace_boundary_rejection_fraction"])
        importance_candidates = int(first["state_importance_candidate_count"])
        importance_ess = float(first["state_importance_effective_sample_size"])
        importance_max_weight = float(first["state_importance_max_weight"])
        importance_pareto_k = float(first["state_importance_pareto_k"])
        importance_min_tail_ess = float(
            first["state_importance_min_tail_ess"]
        )
        structural_design = int(first["structural_prior_design_size"])
        total_draws = int(len(group))
        state_valid = bool(
            rank == dimension
            and np.isfinite(condition)
            and condition <= 1e8
            and np.isfinite(rejection)
            and rejection < STATE_LAPLACE_FATAL_MAX_BOUNDARY_REJECTION
            and importance_ess >= STATE_IMPORTANCE_FATAL_MIN_ESS
            and importance_max_weight <= STATE_IMPORTANCE_FATAL_MAX_WEIGHT
            and importance_pareto_k <= STATE_IMPORTANCE_FATAL_MAX_PARETO_K
            and importance_min_tail_ess >= STATE_IMPORTANCE_FATAL_MIN_TAIL_ESS
        )
        state_recommended = bool(
            state_valid
            and rejection < STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION
            and importance_ess >= STATE_IMPORTANCE_RECOMMENDED_MIN_ESS
            and importance_max_weight <= STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT
            and importance_pareto_k
            <= STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K
            and importance_min_tail_ess
            >= STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS
        )
        for parameter in parameter_columns:
            values = pd.to_numeric(group[parameter], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            design_ess = (
                min(total_draws, structural_design)
                if parameter in structural_names
                else min(total_draws, importance_ess)
            )
            converged = bool(state_valid and structural_design >= 50 and total_draws >= 100)
            recommended = bool(
                state_recommended and structural_design >= 100 and total_draws >= 400
            )
            rows.append(
                {
                    "country": country,
                    "parameter": parameter,
                    "diagnostic_method": "state_space_exact_importance_and_prior_design_quality",
                    # These are resampled importance/design draws, not
                    # independent Markov chains. R-hat is not applicable.
                    "rhat": np.nan,
                    "rhat_rank": np.nan,
                    "bulk_ess": np.nan,
                    "tail_ess": np.nan,
                    "parameter_effective_support_size": float(design_ess),
                    "parameter_effective_support_interpretation": (
                        "importance_ESS_for_state_coordinates"
                        if parameter not in structural_names
                        else "equal_mass_external_prior_design_size"
                    ),
                    "state_posterior_rank": rank,
                    "state_posterior_dimension": dimension,
                    "state_posterior_condition_number": condition,
                    "state_laplace_boundary_rejection_fraction": rejection,
                    "state_importance_candidate_count": importance_candidates,
                    "state_importance_effective_sample_size": importance_ess,
                    "state_importance_ess_fraction": float(
                        importance_ess / max(importance_candidates, 1)
                    ),
                    "state_importance_max_weight": importance_max_weight,
                    "state_importance_pareto_k": importance_pareto_k,
                    "state_importance_min_tail_ess": importance_min_tail_ess,
                    "structural_prior_design_size": structural_design,
                    "n_chains": int(group["chain"].nunique()),
                    "total_draws": total_draws,
                    "converged": converged,
                    "recommended_converged": recommended,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
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
    is_joint_importance = (
        "sampling_method" in samples.columns
        and samples["sampling_method"].astype(str).str.lower().eq(JOINT_IMPORTANCE_SAMPLER).all()
    )
    is_modular_cut = bool(
        is_joint_importance
        and "inference_structure" in samples.columns
        and samples["inference_structure"]
        .astype(str)
        .str.lower()
        .isin({"modular_cut", "modular_hierarchical_cut"})
        .all()
    )
    is_smc = (
        "sampling_method" in samples.columns
        and samples["sampling_method"].astype(str).str.lower().eq(SMC_SAMPLER).all()
    )
    is_state_space_exact_importance = (
        "sampling_method" in samples.columns
        and samples["sampling_method"]
        .astype(str)
        .str.lower()
        .eq(STATE_SPACE_LAPLACE_CUT_SAMPLER)
        .all()
    )
    if is_state_space_exact_importance:
        diagnostics = _compute_state_space_exact_importance_diagnostics(
            samples,
            tuple(param_cols),
        )
    elif is_joint_importance:
        diagnostics = _compute_importance_diagnostics(samples, tuple(param_cols))
    elif is_smc:
        diagnostics = _compute_smc_diagnostics(samples, tuple(param_cols))
    else:
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
            "all_recommended_converged": grid_valid,
            "n_parameters_converged": n_converged,
            "n_parameters_recommended_converged": n_converged,
            "fraction_converged": (n_converged / n_total) if n_total else 0.0,
            "fraction_recommended_converged": (n_converged / n_total) if n_total else 0.0,
            "countries_with_issues": bad_grid,
            "countries_with_recommended_issues": bad_grid,
            "beta_grid_quality": beta_grid_quality,
            "method_note": (
                "Deterministic beta-grid quadrature; MCMC R-hat/ESS criteria are "
                "reported for the quantile sample only. Validity is determined "
                "by the beta-grid tail and quadrature-resolution checks below."
            ),
            }

    if is_joint_importance:
        if is_modular_cut:
            convergence_summary = {
                **convergence_summary,
                "method_note": (
                    "Modular hierarchical cut: structural nuisance draws form an equal-mass "
                    "external-prior design, while each country's beta/reporting distribution is "
                    "normalized conditionally within every structural draw. R-hat and nuisance "
                    "posterior ESS are not applicable. Validity is gated by structural design "
                    "size and worst-case, per-nuisance conditional-grid completeness, truncation, "
                    "and two-axis resolution."
                ),
                "modular_cut_thresholds": {
                    "min_structural_prior_draws": MODULAR_CUT_RECOMMENDED_MIN_STRUCTURAL_DRAWS,
                    "min_conditional_effective_grid_points": MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS,
                    "min_axis_effective_grid_points": MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS,
                    "max_conditional_single_weight": MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT,
                    "max_conditional_edge_weight": MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT,
                    "require_complete_conditional_grids": True,
                },
            }

        else:
            convergence_summary = {
                **convergence_summary,
                "method_note": (
                    "Full joint posterior sampled by self-normalised importance resampling. "
                    "Weakly identified nuisance parameters are drawn from their configured priors, "
                    "while beta/reporting are integrated on a local grid. R-hat is not applicable; "
                    "the displayed ESS fields are importance-weight ESS repeated by parameter for audit compatibility."
                ),
                "importance_thresholds": {
                    "min_effective_sample_size": MIN_RECOMMENDED_BULK_ESS,
                    "min_nuisance_effective_sample_size": IMPORTANCE_RECOMMENDED_NUISANCE_ESS,
                    "max_single_weight": IMPORTANCE_RECOMMENDED_MAX_WEIGHT,
                    "max_edge_weight": IMPORTANCE_RECOMMENDED_EDGE_WEIGHT,
                },
            }
    if is_state_space_exact_importance:
        convergence_summary = {
            **convergence_summary,
            "method_note": (
                "A defensive multi-scale Gauss-Newton proposal for beta, reporting, "
                "and the historical latent log-beta path is corrected with the exact "
                "NB2/AR(1)/prior target, then combined with an equal-mass shared "
                "external-prior Latin-hypercube design for structural parameters. "
                "Synthetic chain labels are batching only; MCMC R-hat is not applicable. "
                "Gates assess local curvature rank/conditioning, proposal support, exact-"
                "target importance ESS/maximum weight, and structural design size."
            ),
            "state_importance_thresholds": {
                "min_effective_sample_size": STATE_IMPORTANCE_RECOMMENDED_MIN_ESS,
                "max_single_weight": STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT,
                "max_pareto_k": STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K,
                "min_tail_effective_support": STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS,
                "max_out_of_bounds_fraction": STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION,
            },
        }
    if is_smc:
        convergence_summary = {
            **convergence_summary,
            "method_note": (
                "Full joint posterior sampled by tempered sequential Monte Carlo. "
                "Independent same-target SMC islands retain equal replicate mass and are "
                "resampled within island using final particle weights; marginal-likelihood "
                "variation is retained as a numerical diagnostic rather than a pooling weight. "
                "validity is gated by adaptive-temperature particle ESS, maximum normalized weight, "
                "final-particle diversity, replicate coverage, and rank-normalized diagnostics "
                "on the resampled posterior draws."
            ),
            "smc_thresholds": {
                "min_stage_ess_fraction": SMC_RECOMMENDED_ESS_FRACTION,
                "min_final_effective_sample_size": MIN_RECOMMENDED_BULK_ESS,
                "max_single_weight": SMC_RECOMMENDED_MAX_WEIGHT,
                "min_unique_particle_fraction": SMC_RECOMMENDED_UNIQUE_PARTICLE_FRACTION,
                "min_island_effective_sample_size_fraction": SMC_RECOMMENDED_ISLAND_ESS_FRACTION,
                "max_island_weight": SMC_RECOMMENDED_MAX_ISLAND_WEIGHT,
                "max_rhat": MAX_RECOMMENDED_RHAT,
                "min_rank_bulk_ess": MIN_RECOMMENDED_BULK_ESS,
                "min_rank_tail_ess": MIN_RECOMMENDED_TAIL_ESS,
            },
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
        elif is_state_space_exact_importance:
            f.write("Conditional State-space Exact-importance + Structural Prior Design Summary\n")
            f.write("\n\nConditional exact-target importance quality criteria:\n")
            f.write("  - Minimum validity floor:\n")
            f.write("    * State-posterior curvature rank must equal its dimension\n")
            f.write("    * State-posterior condition number <= 1e8\n")
            f.write(
                "    * Bounded-proposal rejection fraction < "
                f"{STATE_LAPLACE_FATAL_MAX_BOUNDARY_REJECTION:.2f}\n"
            )
            f.write(
                "    * Exact-target importance ESS >= "
                f"{STATE_IMPORTANCE_FATAL_MIN_ESS:.0f}\n"
            )
            f.write(
                "    * Maximum normalized importance weight <= "
                f"{STATE_IMPORTANCE_FATAL_MAX_WEIGHT:.2f}\n"
            )
            f.write(
                "    * Generalized-Pareto tail shape k <= "
                f"{STATE_IMPORTANCE_FATAL_MAX_PARETO_K:.2f}\n"
            )
            f.write(
                "    * Minimum coordinate-tail effective support >= "
                f"{STATE_IMPORTANCE_FATAL_MIN_TAIL_ESS:.0f}\n"
            )
            f.write("    * Structural design draws >= 50 and output draws >= 100\n")
            f.write("  - Recommended optional nonpublication research standard:\n")
            f.write(
                "    * Bounded-proposal rejection fraction < "
                f"{STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION:.2f}\n"
            )
            f.write(
                "    * Exact-target importance ESS >= "
                f"{STATE_IMPORTANCE_RECOMMENDED_MIN_ESS:.0f}\n"
            )
            f.write(
                "    * Maximum normalized importance weight <= "
                f"{STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT:.2f}\n"
            )
            f.write(
                "    * Generalized-Pareto tail shape k <= "
                f"{STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K:.2f}\n"
            )
            f.write(
                "    * Minimum coordinate-tail effective support >= "
                f"{STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS:.0f}\n"
            )
            f.write("    * Structural design draws >= 100 and output draws >= 400\n")
            f.write(
                "  - Synthetic chain identifiers are output batches, not Markov chains; "
                "R-hat is not interpreted for this route.\n"
            )
        elif is_modular_cut:
            f.write("Modular Cut Conditional Grid Quality Summary\n")
        elif is_joint_importance:
            f.write("Joint Importance Posterior Quality Summary\n")
        elif is_smc:
            f.write("Tempered SMC Posterior Quality Summary\n")
        else:
            f.write("MCMC Convergence Summary\n")
        f.write("=" * 60 + "\n\n")
        if is_beta_grid:
            f.write(convergence_summary["method_note"] + "\n\n")
        if is_joint_importance:
            f.write(convergence_summary["method_note"] + "\n\n")
        if is_smc:
            f.write(convergence_summary["method_note"] + "\n\n")
        if is_state_space_exact_importance:
            f.write(convergence_summary["method_note"] + "\n\n")
        quality_verb = "passed quality checks" if is_state_space_exact_importance else "converged"
        f.write(
            f"All parameters {quality_verb}: "
            f"{convergence_summary['all_converged']}\n"
        )
        f.write(
            f"Parameters that {quality_verb}: {convergence_summary['n_parameters_converged']}"
            f" / {convergence_summary['n_parameters_total']}"
            f" ({convergence_summary.get('fraction_converged', 0):.1%})\n"
        )
        if not is_beta_grid:
            recommended_label = (
                "All parameters met the recommended conditional quality standard"
                if is_state_space_exact_importance
                else "All parameters met recommended convergence"
            )
            f.write(
                f"{recommended_label}: "
                f"{convergence_summary.get('all_recommended_converged', False)}\n"
            )
            f.write(
                "Parameters meeting recommended convergence: "
                f"{convergence_summary.get('n_parameters_recommended_converged', 0)}"
                f" / {convergence_summary['n_parameters_total']}"
                f" ({convergence_summary.get('fraction_recommended_converged', 0):.1%})\n"
            )
        if is_joint_importance:
            f.write("R-hat: not applicable to deterministic importance resampling\n")
            if not diagnostics.empty:
                if is_modular_cut:
                    f.write("Nuisance posterior ESS: not applicable (equal-mass prior design)\n")
                    f.write(
                        "Structural prior design draws: "
                        f"{pd.to_numeric(diagnostics['modular_cut_structural_draw_count'], errors='coerce').min():.0f}\n"
                    )
                    f.write(
                        "Worst conditional effective grid points: "
                        f"{pd.to_numeric(diagnostics['modular_cut_conditional_min_effective_grid_points'], errors='coerce').min():.1f}\n"
                    )
                    f.write(
                        "Worst conditional beta/reporting-axis effective points: "
                        f"{pd.to_numeric(diagnostics['modular_cut_conditional_min_beta_effective_grid_points'], errors='coerce').min():.1f} / "
                        f"{pd.to_numeric(diagnostics['modular_cut_conditional_min_reporting_effective_grid_points'], errors='coerce').min():.1f}\n"
                    )
                    f.write(
                        "Worst conditional single-cell weight: "
                        f"{pd.to_numeric(diagnostics['modular_cut_conditional_max_single_weight'], errors='coerce').max():.4f}\n"
                    )
                    f.write(
                        "Worst conditional beta/reporting edge weight: "
                        f"{pd.to_numeric(diagnostics['modular_cut_conditional_max_edge_weight'], errors='coerce').max():.4f}\n"
                    )
                else:
                    f.write(
                        "Minimum importance ESS: "
                        f"{pd.to_numeric(diagnostics['importance_effective_sample_size'], errors='coerce').min():.0f}\n"
                    )
                    f.write(
                        "Minimum nuisance ESS: "
                        f"{pd.to_numeric(diagnostics['importance_nuisance_effective_sample_size'], errors='coerce').min():.0f}\n"
                    )
                    f.write(
                        "Maximum single normalized weight: "
                        f"{pd.to_numeric(diagnostics['importance_max_weight'], errors='coerce').max():.4f}\n"
                    )
                    f.write(
                        "Maximum beta/reporting edge weight: "
                        f"{pd.to_numeric(diagnostics['importance_edge_weight'], errors='coerce').max():.4f}\n"
                    )
        elif is_smc:
            f.write(f"Worst R-hat (rank-normalized): {convergence_summary['worst_rhat']:.4f}\n")
            f.write(f"Minimum conservative bulk ESS: {convergence_summary['min_bulk_ess']:.0f}\n")
            f.write(f"Minimum conservative tail ESS: {convergence_summary['min_tail_ess']:.0f}\n")
            if not diagnostics.empty:
                f.write(
                    "Minimum SMC stage ESS fraction: "
                    f"{pd.to_numeric(diagnostics['smc_min_ess_fraction'], errors='coerce').min():.3f}\n"
                )
                f.write(
                    "Minimum SMC final ESS: "
                    f"{pd.to_numeric(diagnostics['smc_final_ess'], errors='coerce').min():.0f}\n"
                )
                if "smc_combined_particle_ess" in diagnostics.columns:
                    f.write(
                        "Minimum equal-island pooled particle ESS: "
                        f"{pd.to_numeric(diagnostics['smc_combined_particle_ess'], errors='coerce').min():.0f}\n"
                    )
                if "smc_island_ess" in diagnostics.columns:
                    f.write(
                        "Minimum equal-mass island ESS: "
                        f"{pd.to_numeric(diagnostics['smc_island_ess'], errors='coerce').min():.2f}\n"
                    )
                    f.write(
                        "Maximum equal-mass island weight: "
                        f"{pd.to_numeric(diagnostics['smc_max_island_weight'], errors='coerce').max():.3f}\n"
                    )
                if "smc_log_evidence_range" in diagnostics.columns:
                    f.write(
                        "Maximum cross-island log-evidence range (diagnostic only): "
                        f"{pd.to_numeric(diagnostics['smc_log_evidence_range'], errors='coerce').max():.2f}\n"
                    )
                f.write(
                    "Maximum SMC single normalized weight: "
                    f"{pd.to_numeric(diagnostics['smc_max_weight'], errors='coerce').max():.4f}\n"
                )
                f.write(
                    "Minimum SMC unique-particle fraction: "
                    f"{pd.to_numeric(diagnostics['smc_unique_particle_fraction'], errors='coerce').min():.3f}\n"
                )
        else:
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
        elif is_modular_cut:
            f.write("\n\nModular-cut conditional-grid quality criteria:\n")
            f.write("  - Structural nuisance draws are equal-mass prior design points, not posterior ESS.\n")
            f.write("  - Every nuisance draw must have a complete beta/reporting grid.\n")
            f.write("  - Minimum validity floor:\n")
            f.write(
                f"    * Structural prior design draws >= {MODULAR_CUT_FATAL_MIN_STRUCTURAL_DRAWS}\n"
            )
            f.write(
                f"    * Worst conditional effective grid points >= {MODULAR_CUT_FATAL_MIN_CONDITIONAL_EFFECTIVE_POINTS:.0f}\n"
            )
            f.write(
                f"    * Worst beta and reporting marginal effective points >= {MODULAR_CUT_FATAL_MIN_AXIS_EFFECTIVE_POINTS:.0f}\n"
            )
            f.write(
                f"    * Worst conditional single-cell weight <= {MODULAR_CUT_FATAL_MAX_CONDITIONAL_WEIGHT:.2f}\n"
            )
            f.write(
                f"    * Worst conditional combined edge weight <= {MODULAR_CUT_FATAL_MAX_CONDITIONAL_EDGE_WEIGHT:.2f}\n"
            )
            f.write("  - Recommended strict numerical standard:\n")
            f.write(
                f"    * Structural prior design draws >= {MODULAR_CUT_RECOMMENDED_MIN_STRUCTURAL_DRAWS}\n"
            )
            f.write(
                f"    * Worst conditional effective grid points >= {MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS:.0f}\n"
            )
            f.write(
                f"    * Worst beta and reporting marginal effective points >= {MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS:.0f}\n"
            )
            f.write(
                f"    * Worst conditional single-cell weight <= {MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT:.2f}\n"
            )
            f.write(
                f"    * Worst conditional combined edge weight <= {MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT:.2f}\n"
            )
            f.write(
                "  - A strict audited sensitivity run uses --fail-on-warnings, so every "
                "conditional grid must meet the recommended worst-case standard.\n"
            )
        elif is_joint_importance:
            f.write("\n\nImportance-sampling quality criteria:\n")
            f.write("  - Minimum validity floor:\n")
            f.write(f"    * Importance ESS >= {MIN_FATAL_BULK_ESS:.0f}\n")
            f.write(f"    * Nuisance marginal ESS >= {IMPORTANCE_FATAL_NUISANCE_ESS:.0f}\n")
            f.write(f"    * Maximum single normalized weight <= {IMPORTANCE_FATAL_MAX_WEIGHT:.3f}\n")
            f.write(f"    * Combined beta/reporting edge weight <= {IMPORTANCE_FATAL_EDGE_WEIGHT:.3f}\n")
            f.write("  - Recommended strict numerical standard:\n")
            f.write(f"    * Importance ESS >= {MIN_RECOMMENDED_BULK_ESS:.0f}\n")
            f.write(f"    * Nuisance marginal ESS >= {IMPORTANCE_RECOMMENDED_NUISANCE_ESS:.0f}\n")
            f.write(f"    * Maximum single normalized weight <= {IMPORTANCE_RECOMMENDED_MAX_WEIGHT:.3f}\n")
            f.write(f"    * Combined beta/reporting edge weight <= {IMPORTANCE_RECOMMENDED_EDGE_WEIGHT:.3f}\n")
            f.write(
                "  - A strict audited sensitivity run uses --fail-on-warnings, "
                "so the recommended weight-quality standard is enforced for final outputs.\n"
            )
        elif is_smc:
            f.write("\n\nSMC quality criteria:\n")
            f.write("  - Minimum validity floor:\n")
            f.write("    * Adaptive tempering must reach final temperature 1.0\n")
            f.write(f"    * Stage ESS fraction >= {SMC_FATAL_ESS_FRACTION:.2f}\n")
            f.write(f"    * Final particle ESS >= max({MIN_FATAL_BULK_ESS:.0f}, {SMC_FATAL_ESS_FRACTION:.2f} x particles)\n")
            f.write(f"    * Maximum single normalized weight <= {SMC_FATAL_MAX_WEIGHT:.3f}\n")
            f.write(f"    * Unique final particle fraction >= {SMC_FATAL_UNIQUE_PARTICLE_FRACTION:.2f}\n")
            f.write(f"    * Equal-mass island ESS >= {SMC_FATAL_ISLAND_ESS:.1f}\n")
            f.write(f"    * Maximum equal-mass island weight <= {SMC_FATAL_MAX_ISLAND_WEIGHT:.2f}\n")
            f.write(f"    * Rank-normalized R-hat <= {MAX_FATAL_RHAT:.2f}\n")
            f.write(f"    * Conservative bulk ESS >= {MIN_FATAL_BULK_ESS:.0f}\n")
            f.write(f"    * Conservative tail ESS >= {MIN_FATAL_TAIL_ESS:.0f}\n")
            f.write("  - Recommended strict numerical standard:\n")
            f.write(f"    * Stage ESS fraction >= {SMC_RECOMMENDED_ESS_FRACTION:.2f}\n")
            f.write(f"    * Final particle ESS >= max({MIN_RECOMMENDED_BULK_ESS:.0f}, {SMC_RECOMMENDED_ESS_FRACTION:.2f} x particles)\n")
            f.write(f"    * Maximum single normalized weight <= {SMC_RECOMMENDED_MAX_WEIGHT:.3f}\n")
            f.write(f"    * Unique final particle fraction >= {SMC_RECOMMENDED_UNIQUE_PARTICLE_FRACTION:.2f}\n")
            f.write(f"    * Equal-mass island ESS >= {SMC_RECOMMENDED_ISLAND_ESS_FRACTION:.2f} x islands\n")
            f.write(f"    * Maximum equal-mass island weight <= {SMC_RECOMMENDED_MAX_ISLAND_WEIGHT:.2f}\n")
            f.write(f"    * Rank-normalized R-hat <= {MAX_RECOMMENDED_RHAT:.2f}\n")
            f.write(f"    * Conservative bulk/tail ESS >= {MIN_RECOMMENDED_BULK_ESS:.0f}\n")
            f.write(
                "  - A strict audited sensitivity run uses --fail-on-warnings, "
                "so the recommended particle and replicate diagnostics are enforced for final outputs.\n"
            )
        else:
            f.write("\n\nConvergence criteria:\n")
            f.write("  - Minimum validity floor:\n")
            f.write(f"    * R-hat (rank-normalized) <= {MAX_FATAL_RHAT:.2f}\n")
            f.write(f"    * Bulk ESS >= {MIN_FATAL_BULK_ESS:.0f}\n")
            f.write(f"    * Tail ESS >= {MIN_FATAL_TAIL_ESS:.0f}\n")
            f.write("  - Recommended strict numerical standard:\n")
            f.write(f"    * R-hat (rank-normalized) <= {MAX_RECOMMENDED_RHAT:.2f}\n")
            f.write(f"    * Bulk ESS >= {MIN_RECOMMENDED_BULK_ESS:.0f}\n")
            f.write(f"    * Tail ESS >= {MIN_RECOMMENDED_TAIL_ESS:.0f}\n")
            f.write(
                "  - A strict audited sensitivity run uses --fail-on-warnings, "
                "so the recommended standard is enforced for final outputs.\n"
            )
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
                if is_joint_importance:
                    row = country_diag.iloc[0]
                    if is_modular_cut:
                        f.write(
                            f"  {country}: structural draws={row['modular_cut_structural_draw_count']:.0f}, "
                            f"min conditional ESS={row['modular_cut_conditional_min_effective_grid_points']:.1f}, "
                            f"min axis ESS={min(row['modular_cut_conditional_min_beta_effective_grid_points'], row['modular_cut_conditional_min_reporting_effective_grid_points']):.1f}, "
                            f"max conditional weight={row['modular_cut_conditional_max_single_weight']:.4f}, "
                            f"max conditional edge weight={row['modular_cut_conditional_max_edge_weight']:.4f}\n"
                        )
                    else:
                        f.write(
                            f"  {country}: importance ESS={row['importance_effective_sample_size']:.0f}, "
                            f"nuisance ESS={row['importance_nuisance_effective_sample_size']:.0f}, "
                            f"max weight={row['importance_max_weight']:.4f}, "
                            f"edge weight={row['importance_edge_weight']:.4f}\n"
                        )
                elif is_smc:
                    row = country_diag.iloc[0]
                    worst_rhat_row = country_diag.loc[country_diag["rhat_rank"].idxmax()]
                    f.write(
                        f"  {country}: SMC min ESS={row['smc_min_ess']:.0f}, "
                        f"final ESS={row['smc_final_ess']:.0f}, "
                        f"combined ESS={row.get('smc_combined_particle_ess', np.nan):.0f}, "
                        f"island ESS={row.get('smc_island_ess', np.nan):.2f}, "
                        f"max island weight={row.get('smc_max_island_weight', np.nan):.3f}, "
                        f"max weight={row['smc_max_weight']:.4f}, "
                        f"unique particles={row['smc_unique_particle_fraction']:.3f}, "
                        f"worst R-hat={worst_rhat_row['rhat_rank']:.3f} "
                        f"({worst_rhat_row['parameter']})\n"
                    )
                else:
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


def _remove_path(path: Path) -> None:
    """Remove one known output path without following unrelated paths."""

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _clear_previous_sampler_artifacts(
    countries: list[str],
    *,
    sampler: str,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    reuse_valid_beta_grid: bool = False,
) -> None:
    """Clear stale uncertainty artifacts belonging to exactly one output stem.

    A failed rerun must not leave an earlier sampler's diagnostics or countries
    looking current.  All paths below are derived from the same artifact naming
    helper used for writes.  Other stems are never globbed or removed.  Explicit
    beta-grid reuse preserves only grids for the countries in the new run.
    """

    stem = str(output_stem or DEFAULT_OUTPUT_STEM).strip()
    if stem in RETIRED_MISLABELED_OUTPUT_STEMS:
        raise ValueError(
            f"Output stem {stem!r} is retired because it mislabeled a legacy "
            "research route as Figure 2c or as a full joint posterior; use "
            f"{DEFAULT_OUTPUT_STEM!r}."
        )
    if not stem or Path(stem).name != stem or stem in {".", ".."}:
        raise ValueError(f"Output stem must be a single safe path component: {output_stem!r}")

    _remove_path(Path(_mcmc_progress_dir(stem)))

    grid_dir = Path(project_path("outputs", "metadata", f"beta_grid_{stem}"))
    preserve_selected_grids = bool(
        str(sampler).lower() == "beta_grid" and reuse_valid_beta_grid
    )
    if grid_dir.exists() and preserve_selected_grids:
        retained_names = {
            f"{country}_grid{extension}"
            for country in countries
            for extension in (".csv", ".parquet")
        }
        for path in grid_dir.iterdir():
            if path.name not in retained_names:
                _remove_path(path)
    else:
        _remove_path(grid_dir)

    exact_paths = [
        project_path("outputs", "simulations", f"{stem}.csv"),
        project_path("outputs", "simulations", f"{stem}.parquet"),
        project_path("outputs", "summaries", f"{stem}_summary.csv"),
        project_path("outputs", "summaries", f"{stem}_summary.parquet"),
        project_path("outputs", "metadata", f"{stem}_run_metadata.json"),
    ]
    artifact_specs = (
        ("outputs/simulations", "bayesian_posterior_samples", "posterior_samples", ".parquet"),
        ("outputs/simulations", "bayesian_stochastic_overlay_samples", "stochastic_overlay_samples", ".csv"),
        ("outputs/summaries", "bayesian_uncertainty_intervals_summary", "intervals_summary", ".csv"),
        ("outputs/summaries", "bayesian_parameter_summary", "parameter_summary", ".csv"),
        ("outputs/summaries", "bayesian_stochastic_overlay_intervals_summary", "stochastic_overlay_intervals_summary", ".csv"),
        ("outputs/summaries", "bayesian_stochastic_overlay_variance_components", "stochastic_overlay_variance_components", ".csv"),
        ("outputs/summaries", "bayesian_k_sensitivity_sweep", "k_sensitivity_sweep", ".csv"),
        ("outputs/summaries", "bayesian_convergence_diagnostics", "convergence_diagnostics", ".csv"),
        ("outputs/summaries", "bayesian_beta_grid_quality", "beta_grid_quality", ".csv"),
        ("outputs/summaries", "bayesian_convergence_summary", "convergence_summary", ".txt"),
    )
    for directory, canonical, suffix, extension in artifact_specs:
        artifact = _artifact_stem(stem, canonical, suffix)
        exact_paths.append(project_path(directory, f"{artifact}{extension}"))
        if extension == ".csv":
            exact_paths.append(project_path(directory, f"{artifact}.parquet"))
        elif extension == ".parquet":
            exact_paths.append(project_path(directory, f"{artifact}.csv"))
    for path in exact_paths:
        _remove_path(Path(path))

    metadata_dir = Path(project_path("outputs", "metadata"))
    sampler_prefixes = (
        f"{_artifact_stem(stem, 'bayesian_smc_stage_history', 'smc_stage_history')}_",
        f"{_artifact_stem(stem, 'bayesian_modular_cut_conditional_grid_quality', 'modular_cut_conditional_grid_quality')}_",
    )
    if metadata_dir.exists():
        for path in metadata_dir.iterdir():
            if any(path.name.startswith(prefix) for prefix in sampler_prefixes):
                _remove_path(path)
    diagnostics_dir = Path(project_path("outputs", "diagnostics"))
    if diagnostics_dir.exists():
        prefix = f"{stem}_"
        suffixes = (
            "_state_importance_candidates.csv",
            "_state_importance_candidates.parquet",
        )
        for path in diagnostics_dir.iterdir():
            if path.name.startswith(prefix) and path.name.endswith(suffixes):
                _remove_path(path)


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
    random_seed: int | None = None,
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
    sampler: str = STATE_SPACE_LAPLACE_CUT_SAMPLER,
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
    importance_nuisance_draws: int = DEFAULT_IMPORTANCE_NUISANCE_DRAWS,
    importance_beta_grid_points: int = DEFAULT_IMPORTANCE_BETA_GRID_POINTS,
    importance_reporting_grid_points: int = DEFAULT_IMPORTANCE_REPORTING_GRID_POINTS,
    importance_log_beta_half_width: float = DEFAULT_IMPORTANCE_LOG_BETA_HALF_WIDTH,
    importance_reporting_coordinate_half_width: float = DEFAULT_IMPORTANCE_REPORTING_COORDINATE_HALF_WIDTH,
    importance_defensive_prior_fraction: float = DEFAULT_IMPORTANCE_DEFENSIVE_PRIOR_FRACTION,
    smc_particles: int | None = None,
    smc_ess_fraction: float | None = None,
    smc_move_steps: int | None = None,
    smc_max_stages: int | None = None,
):
    """Run the selected uncertainty-analysis pipeline.

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
    if str(output_stem) in RETIRED_MISLABELED_OUTPUT_STEMS:
        raise ValueError(
            f"Output stem {output_stem!r} is retired because it mislabeled a legacy "
            "research route as Figure 2c or as a full joint posterior; use "
            f"{DEFAULT_OUTPUT_STEM!r}."
        )
    configs = load_configs()
    settings = deepcopy(configs["baseline"]["bayesian_uncertainty"])
    sampler = str(sampler).lower()
    if sampler == LEGACY_STATE_SPACE_LAPLACE_CUT_SAMPLER:
        sampler = STATE_SPACE_EXACT_IMPORTANCE_CUT_SAMPLER
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
    n_chains = int(n_chains or settings.get("n_chains", 4))
    chain_draws = int(draws or settings.get("draws", 2500))
    chain_warmup = int(warmup or settings.get("warmup", 1500))
    base_seed = int(
        settings.get("random_seed", 20260510)
        if random_seed is None
        else random_seed
    )
    proposal_scale = float(proposal_scale if proposal_scale is not None else settings.get("proposal_scale", 1.0))
    enable_trend = bool(settings.get("enable_time_varying_reporting", True))
    thin = int(thin or settings.get("thin", 1))
    likelihood_interval = str(settings.get("likelihood_observation_frequency", "monthly"))
    chain_dispersion = float(settings.get("dispersion", 50.0))
    smc_particles = int(smc_particles if smc_particles is not None else settings.get("smc_particles", DEFAULT_SMC_PARTICLES))
    smc_ess_fraction = float(
        smc_ess_fraction if smc_ess_fraction is not None else settings.get("smc_ess_fraction", DEFAULT_SMC_ESS_FRACTION)
    )
    smc_move_steps = int(smc_move_steps if smc_move_steps is not None else settings.get("smc_move_steps", DEFAULT_SMC_MOVE_STEPS))
    smc_max_stages = int(smc_max_stages if smc_max_stages is not None else settings.get("smc_max_stages", DEFAULT_SMC_MAX_STAGES))
    countries = publication_country_names(configs)
    publication_exclusions = {
        str(country): str(reason)
        for country, reason in settings.get("publication_country_exclusions", {}).items()
    }

    # Filter countries if specified (for pilot testing)
    if countries_filter:
        available = set(countries)
        explicitly_excluded = sorted(set(countries_filter).intersection(publication_exclusions))
        if explicitly_excluded:
            reasons = "; ".join(
                f"{country}: {publication_exclusions[country]}"
                for country in explicitly_excluded
            )
            raise ValueError(
                "Requested countries are outside the prespecified calibrated-publication set: "
                + reasons
            )
        countries = [c for c in countries_filter if c in available]
        if not countries:
            raise ValueError(f"No valid countries in filter: {countries_filter}. Available: {sorted(available)}")

    if (
        sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
        and bool(settings.get("require_calibrated_start", True))
    ):
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
    elif sampler in {JOINT_IMPORTANCE_SAMPLER, STATE_SPACE_LAPLACE_CUT_SAMPLER}:
        if fixed_parameters or fix_durations:
            raise ValueError(
                f"{sampler} propagates all configured uncertainty dimensions and cannot fix them. "
                f"fixed_parameters={fixed_parameters}, fix_durations={fix_durations}"
            )
        tasks = [
            ChainTask(
                country=country,
                chain=0,
                # A stride larger than every stage-specific offset prevents
                # deterministic RNG-stream collisions between countries.
                seed=base_seed + country_idx * 100_000 + 811,
                warmup=0,
                draws=chain_draws,
                proposal_scale=proposal_scale,
                enable_time_varying_reporting=enable_trend,
                thin=thin,
                fix_durations=False,
                fixed_parameters=(),
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
                grid_n_chains=n_chains,
                grid_eval_jobs=1,
                importance_nuisance_draws=importance_nuisance_draws,
                importance_beta_grid_points=importance_beta_grid_points,
                importance_reporting_grid_points=importance_reporting_grid_points,
                importance_log_beta_half_width=importance_log_beta_half_width,
                importance_reporting_coordinate_half_width=importance_reporting_coordinate_half_width,
                importance_defensive_prior_fraction=importance_defensive_prior_fraction,
            )
            for country_idx, country in enumerate(countries)
        ]
    elif sampler == SMC_SAMPLER:
        if fixed_parameters or fix_durations:
            raise ValueError(
                "smc is a legacy full-feedback sensitivity sampler and cannot fix "
                "posterior dimensions. "
                f"fixed_parameters={fixed_parameters}, fix_durations={fix_durations}"
            )
        tasks = [
            ChainTask(
                country=country,
                chain=chain,
                seed=base_seed + country_idx * 1000 + chain * 997 + 421,
                warmup=0,
                draws=chain_draws,
                proposal_scale=proposal_scale,
                enable_time_varying_reporting=enable_trend,
                thin=thin,
                fix_durations=False,
                fixed_parameters=(),
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
                grid_n_chains=n_chains,
                grid_eval_jobs=1,
                smc_particles=smc_particles,
                smc_ess_fraction=smc_ess_fraction,
                smc_move_steps=smc_move_steps,
                smc_max_stages=smc_max_stages,
            )
            for country_idx, country in enumerate(countries)
            for chain in range(1, n_chains + 1)
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
    elif sampler == JOINT_IMPORTANCE_SAMPLER:
        if grid_eval_jobs is None or int(grid_eval_jobs) <= 0:
            compute_task_count = max(1, min(len(tasks), len(countries)))
            joint_eval_jobs = max(1, int(effective_n_jobs) // compute_task_count)
        else:
            joint_eval_jobs = int(grid_eval_jobs)
        joint_eval_jobs = min(max(1, joint_eval_jobs), int(effective_n_jobs), available_cpus())
        if joint_eval_jobs > 1:
            outer_n_jobs = max(1, min(len(tasks), int(effective_n_jobs) // joint_eval_jobs))
        tasks = [replace(task, grid_eval_jobs=joint_eval_jobs) for task in tasks]
    elif sampler == STATE_SPACE_EXACT_IMPORTANCE_CUT_SAMPLER:
        if grid_eval_jobs is None or int(grid_eval_jobs) <= 0:
            state_eval_jobs = max(
                1,
                int(effective_n_jobs) // max(1, min(len(tasks), len(countries))),
            )
        else:
            state_eval_jobs = int(grid_eval_jobs)
        state_eval_jobs = min(
            max(1, state_eval_jobs),
            int(effective_n_jobs),
            available_cpus(),
        )
        if state_eval_jobs > 1:
            outer_n_jobs = max(
                1,
                min(len(tasks), int(effective_n_jobs) // state_eval_jobs),
            )
        tasks = [replace(task, grid_eval_jobs=state_eval_jobs) for task in tasks]
    elif sampler == SMC_SAMPLER:
        if grid_eval_jobs is None or int(grid_eval_jobs) <= 0:
            compute_task_count = max(1, min(len(tasks), int(effective_n_jobs)))
            smc_eval_jobs = max(1, int(effective_n_jobs) // compute_task_count)
        else:
            smc_eval_jobs = int(grid_eval_jobs)
        smc_eval_jobs = min(max(1, smc_eval_jobs), int(effective_n_jobs), available_cpus())
        if smc_eval_jobs > 1:
            outer_n_jobs = max(1, min(len(tasks), int(effective_n_jobs) // smc_eval_jobs))
        tasks = [replace(task, grid_eval_jobs=smc_eval_jobs) for task in tasks]

    _clear_previous_sampler_artifacts(
        countries,
        sampler=sampler,
        output_stem=output_stem,
        reuse_valid_beta_grid=reuse_valid_beta_grid,
    )
    if str(sampler).lower() == "beta_grid":
        sampler_desc = "bayesian_beta_grid"
    elif str(sampler).lower() == JOINT_IMPORTANCE_SAMPLER:
        sampler_desc = "bayesian_joint_importance"
    elif str(sampler).lower() == STATE_SPACE_LAPLACE_CUT_SAMPLER:
            sampler_desc = "state_space_exact_importance_cut"
    elif str(sampler).lower() == SMC_SAMPLER:
        sampler_desc = "bayesian_smc"
    else:
        sampler_desc = "bayesian_mcmc_chains"
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
    if sampler == SMC_SAMPLER:
        samples = _resample_smc_island_particles(
            samples,
            chain_count=n_chains,
            draws_per_chain=chain_draws,
            base_seed=base_seed,
        )
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
        if str(sampler).lower() == "beta_grid":
            sampler_label = "Beta-grid validity"
        elif str(sampler).lower() == STATE_SPACE_LAPLACE_CUT_SAMPLER:
            sampler_label = "State-space exact-importance/design quality"
        elif str(sampler).lower() == JOINT_IMPORTANCE_SAMPLER:
            sampler_label = "Joint-importance posterior quality"
        elif str(sampler).lower() == SMC_SAMPLER:
            sampler_label = "Tempered-SMC posterior quality"
        else:
            sampler_label = "MCMC convergence"
        warnings.warn(
            f"{sampler_label} issues detected: "
            f"{convergence_summary['n_parameters_converged']}"
            f"/{convergence_summary['n_parameters_total']} parameters converged. "
            f"Worst descriptive R-hat or quality surrogate: {convergence_summary['worst_rhat']:.3f}. "
            f"Consider increasing grid/importance resolution or MCMC warmup/draws as appropriate.",
            stacklevel=2,
        )
    elif not convergence_summary.get("all_recommended_converged", True):
        import warnings
        warnings.warn(
            "The uncertainty run passed its minimum numerical validity floor but did not "
            "meet the recommended design/approximation standard. The strict audit is "
            "expected to fail unless the run is extended or retuned.",
            stacklevel=2,
        )

    if sampler == "beta_grid":
        uncertainty_scope = "conditional_beta_grid"
        interpretation_note = "beta_grid outputs integrate over beta_S conditional on fixed nuisance and structural parameters"
    elif sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER:
        uncertainty_scope = "conditional_state_space_exact_importance_modular_sensitivity"
        interpretation_note = (
            "surveillance enters once through a conditional multi-scale Gauss-Newton "
            "proposal for beta, reporting, and the annual latent log-beta path, with "
            "exact-target importance correction before resampling; "
            "future annual process innovations are propagated from the fitted fixed-"
            "hyperparameter AR(1), and shared structural parameters come from an "
            "external-prior design. "
            "Intervals are conditional on fixed process/measurement hyperparameters and "
            "the reference structural calibration, not full Bayesian credible intervals"
        )
    elif sampler == JOINT_IMPORTANCE_SAMPLER:
        inference_structure = str(
            settings.get("inference_structure", "country_joint_feedback")
        ).lower()
        if inference_structure in {"modular_cut", "modular_hierarchical_cut"}:
            uncertainty_scope = "conditional_modular_cut_uncertainty"
            interpretation_note = (
                "joint_importance propagates an equal-mass external structural-prior design "
                "through country-specific conditional beta/reporting grids; this is a modular-cut "
                "uncertainty distribution, not a full joint or hierarchical posterior"
            )
        else:
            uncertainty_scope = "multi_parameter_joint_posterior"
            interpretation_note = (
                "joint_importance outputs sample the configured country-level posterior by "
                "importance-proposal nuisance draws and beta/reporting grid integration"
            )
    elif sampler == SMC_SAMPLER:
        uncertainty_scope = "multi_parameter_tempered_smc_posterior"
        interpretation_note = (
            "smc outputs sample the configured full country-level posterior by adaptive likelihood tempering, "
            "particle resampling, and Metropolis rejuvenation with particle-quality diagnostics"
        )
    else:
        uncertainty_scope = "multi_parameter_mcmc"
        interpretation_note = "MCMC outputs sample the configured non-fixed parameter set"
    state_target_contract_by_country: dict[str, dict[str, Any]] = {}
    if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER:
        for country, group in samples.groupby("country", sort=False):
            artifact = load_calibrated_country_artifact(str(country))
            approximation = (
                artifact.get("metadata", {}).get("state_posterior_approximation", {})
                if artifact is not None
                else {}
            )
            state_target_contract_by_country[str(country)] = {
                "beta_prior_log_sd": float(
                    approximation.get("beta_prior_log_sd", np.nan)
                ),
                "reporting_prior_log_sd": float(
                    approximation.get("reporting_prior_log_sd", np.nan)
                ),
                "ar1_rho": float(approximation.get("process_ar1_rho", np.nan)),
                "innovation_sd": float(
                    approximation.get("process_innovation_sd", np.nan)
                ),
                "measurement_dispersion": float(
                    approximation.get("measurement_dispersion", np.nan)
                ),
                "task_seed": int(group["state_importance_task_seed"].iloc[0]),
            }
    posterior_metadata = {
        "uncertainty_scope": uncertainty_scope,
        "sampler": sampler,
        "random_seed": int(base_seed),
        "fixed_parameters": sorted(str(parameter) for parameter in (fixed_parameters or ())),
        "fix_durations": bool(fix_durations),
        "n_chains": int(n_chains),
        "warmup": int(chain_warmup),
        "draws_per_chain": int(chain_draws),
        "thin": int(thin),
        "proposal_scale": float(proposal_scale),
        "importance_nuisance_draws": int(importance_nuisance_draws),
        "importance_beta_grid_points": int(importance_beta_grid_points),
        "importance_reporting_grid_points": int(importance_reporting_grid_points),
        "importance_log_beta_half_width": float(importance_log_beta_half_width),
        "importance_reporting_coordinate_half_width": float(importance_reporting_coordinate_half_width),
        "importance_defensive_prior_fraction": float(importance_defensive_prior_fraction),
        "smc_particles": int(smc_particles),
        "smc_ess_fraction": float(smc_ess_fraction),
        "smc_move_steps": int(smc_move_steps),
        "smc_max_stages": int(smc_max_stages),
        "parameterization": str(parameterization),
        "prior_registry": (
            "config/parameter_distributions.yaml::bayesian_joint (structural external-prior block only)"
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else "config/parameter_distributions.yaml::bayesian_joint"
        ),
        "registry_prior_parameter_names": list(
            BAYESIAN_SHARED_PARAMETER_NAMES
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else BAYESIAN_PARAMETER_NAMES
        ),
        "calibration_state_prior_parameter_names": (
            [
                *BAYESIAN_LOCAL_STATE_PARAMETER_NAMES,
                "annual_log_beta_process",
            ]
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else []
        ),
        "uncertainty_config_hash": uncertainty_config_fingerprint(configs),
        "prior_width_overrides": {
            "prior_sd_scale": prior_sd_scale,
            "beta_prior_log_sd": beta_prior_log_sd,
            "reporting_prior_log_sd": reporting_prior_log_sd,
            "ve_prior_sd": ve_prior_sd,
            "rel_asym_prior_sd": rel_asym_prior_sd,
            "fitness_prior_log_sd": fitness_prior_sd,
        },
        "prior_sd_scale_scope": (
            "seven shared structural registry priors only"
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else "all nine registry priors"
        ),
        "prior_density_contract": (
            "state block uses the calibration.process_model Gaussian priors stored in each "
            "artifact; the seven structural quantities use normalized external-prior "
            "inverse-CDF Latin-hypercube draws"
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else "inverse-CDF initialization and transformed log-prior evaluation use the same normalized truncated distribution specifications"
        ),
        "inference_structure": str(
            settings.get("inference_structure", "country_joint_feedback")
        ),
        "chain_semantics": (
            "synthetic_output_batches_not_markov_chains"
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else "sampler_specific"
        ),
        "parameter_source_roles": (
            {
                "beta_S": "conditional_state_posterior",
                "reporting_multiplier": "conditional_state_posterior",
                "historical_log_beta_path": "conditional_state_posterior",
                "future_log_beta_path": "AR1_prior_predictive_process",
                "VE_sus_VE_inf_VE_dur_asymptomatic_durations_fitness": "shared_external_prior_design",
                "intervention_implementation": "intervention_prior_design",
            }
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else {}
        ),
        "state_calibration_prior_contract": (
            {
                "source": "accepted_country_calibration_artifacts",
                "by_country": state_target_contract_by_country,
                "importance_proposal_settings": deepcopy(
                    settings.get("state_space_importance", {})
                ),
                "shared_structural_seed": int(
                    settings.get("shared_structural_seed", 20260712)
                ),
            }
            if sampler == STATE_SPACE_LAPLACE_CUT_SAMPLER
            else {}
        ),
        "likelihood_observation_frequency": str(likelihood_interval),
        "dispersion": float(chain_dispersion),
        "posterior_sample_rows": int(len(samples)),
        "posterior_samples_path": str(
            project_path(
                "outputs/simulations",
                f"{_artifact_stem(output_stem, 'bayesian_posterior_samples', 'posterior_samples')}.parquet",
            )
        ),
        "publication_country_exclusions": publication_exclusions,
        "countries_included": list(countries),
        "convergence_summary": convergence_summary,
        "interpretation_note": interpretation_note,
        "analysis_role": "optional_nonpublication_legacy_research",
        "publication_path": False,
        "figure2c_interval_source": False,
    }

    if skip_posterior_predictive:
        write_run_metadata(
            output_stem,
            current_run_metadata(
                output_stem,
                row_counts={
                    "posterior_samples": int(len(samples)),
                    "posterior_predictive_summary": 0,
                    "posterior_predictive_timeseries": 0,
                },
            )
            | posterior_metadata
            | {
                "posterior_predictive_skipped": True,
                "posterior_samples_path": str(
                    project_path(
                        "outputs/simulations",
                        f"{_artifact_stem(output_stem, 'bayesian_posterior_samples', 'posterior_samples')}.parquet",
                    )
                ),
            },
        )
        return pd.DataFrame(), pd.DataFrame(), samples

    # Phase 3: Posterior predictive simulations (parallelized)
    pp_draws = int(settings.get("posterior_predictive_draws_per_country", 50))
    scenarios = _posterior_predictive_scenarios(
        samples,
        pp_draws,
        random_seed=base_seed,
    )
    timeseries, summary = execute_scenario_list(
        scenarios, stem=output_stem, n_jobs=n_jobs
    )
    write_outputs(
        timeseries,
        summary,
        output_stem,
        extra_metadata=posterior_metadata | {"posterior_predictive_skipped": False},
        require_calibrated=False,
    )
    _write_interval_summaries(summary, samples, output_stem=output_stem)

    # Phase 4: Stochastic overlay
    _apply_stochastic_overlay(summary, settings, output_stem=output_stem)

    # Phase 5: k-sensitivity sweep (shows interval sensitivity to dispersion choice)
    if not skip_k_sensitivity:
        _run_k_sensitivity_sweep(summary, settings, output_stem=output_stem)

    return timeseries, summary, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run optional nonpublication conditional state-space uncertainty research, "
            "or an explicitly selected legacy beta-grid/MCMC/SMC sensitivity engine. "
            "This entry point is not a Figure 2c interval source."
        )
    )
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel workers (-1 for all CPUs)")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Override the configured Monte Carlo seed for reproducibility checks")
    parser.add_argument("--n-chains", type=int, default=None,
                        help="Number of chains to run")
    parser.add_argument("--draws", type=int, default=None,
                        help="Output draws per synthetic batch/chain (configured research default: 128)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Warmup steps for legacy MCMC engines (state exact-importance route: 0)")
    parser.add_argument("--thin", type=int, default=None,
                        help="Store every Nth post-warmup step")
    parser.add_argument("--proposal-scale", type=float, default=None,
                        help="Multiplier on transformed-scale proposal widths")
    parser.add_argument("--dispersion", type=float, default=None,
                        help="Negative-binomial dispersion for Bayesian likelihood")
    parser.add_argument("--likelihood-observation-frequency", type=str, default=None,
                        help="Observation aggregation interval (annual, monthly, or native)")
    parser.add_argument("--sampler", type=str, default=STATE_SPACE_LAPLACE_CUT_SAMPLER,
                        choices=("adaptive_mh", "componentwise_mh", "slice", "beta_grid", JOINT_IMPORTANCE_SAMPLER, SMC_SAMPLER, STATE_SPACE_LAPLACE_CUT_SAMPLER, LEGACY_STATE_SPACE_LAPLACE_CUT_SAMPLER),
                        help=(
                            "Uncertainty engine; state_space_exact_importance_cut is the default research route "
                            "(state_space_laplace_cut is accepted only as a legacy input alias): "
                            "one-use conditional state inference, exact-target importance correction, "
                            "and shared external structural priors"
                        ))
    parser.add_argument("--solver-mode", type=str, default="calibration",
                        choices=("mcmc_fast", "calibration", "production"),
                        help="Solver/runtime fidelity for likelihood evaluations; the state-space exact-target research route requires calibration")
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
                        help="Override fitness_R log-scale prior SD in the central Bayesian registry")
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM,
                        help=(
                            f"Artifact stem for optional nonpublication research "
                            f"(default: {DEFAULT_OUTPUT_STEM}); never a Figure 2c source"
                        ))
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
                        help="Inner workers for beta-grid or exact-state target evaluation; defaults to auto allocation")
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
    parser.add_argument("--importance-nuisance-draws", type=int, default=DEFAULT_IMPORTANCE_NUISANCE_DRAWS,
                        help="Latin-hypercube nuisance prior draws per country for --sampler joint_importance")
    parser.add_argument("--importance-beta-grid-points", type=int, default=DEFAULT_IMPORTANCE_BETA_GRID_POINTS,
                        help="Log-beta grid points per nuisance draw for --sampler joint_importance")
    parser.add_argument("--importance-reporting-grid-points", type=int, default=DEFAULT_IMPORTANCE_REPORTING_GRID_POINTS,
                        help="Reporting-coordinate grid points per beta/nuisance draw for --sampler joint_importance")
    parser.add_argument("--importance-log-beta-half-width", type=float, default=DEFAULT_IMPORTANCE_LOG_BETA_HALF_WIDTH,
                        help="Log-beta grid half-width around the calibrated value for --sampler joint_importance")
    parser.add_argument("--importance-reporting-coordinate-half-width", type=float,
                        default=DEFAULT_IMPORTANCE_REPORTING_COORDINATE_HALF_WIDTH,
                        help="Reporting-coordinate grid half-width around the calibrated value for --sampler joint_importance")
    parser.add_argument("--importance-defensive-prior-fraction", type=float,
                        default=DEFAULT_IMPORTANCE_DEFENSIVE_PRIOR_FRACTION,
                        help="Mixture weight assigned to direct prior nuisance proposals for --sampler joint_importance")
    parser.add_argument("--smc-particles", type=int, default=None,
                        help="Particles per independent replicate for --sampler smc")
    parser.add_argument("--smc-ess-fraction", type=float, default=None,
                        help="Adaptive tempering ESS fraction target for --sampler smc")
    parser.add_argument("--smc-move-steps", type=int, default=None,
                        help="Random-walk Metropolis rejuvenation moves per SMC temperature stage")
    parser.add_argument("--smc-max-stages", type=int, default=None,
                        help="Maximum adaptive temperature stages for --sampler smc")
    args = parser.parse_args()
    main(
        n_jobs=args.n_jobs,
        random_seed=args.random_seed,
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
        importance_nuisance_draws=args.importance_nuisance_draws,
        importance_beta_grid_points=args.importance_beta_grid_points,
        importance_reporting_grid_points=args.importance_reporting_grid_points,
        importance_log_beta_half_width=args.importance_log_beta_half_width,
        importance_reporting_coordinate_half_width=args.importance_reporting_coordinate_half_width,
        importance_defensive_prior_fraction=args.importance_defensive_prior_fraction,
        smc_particles=args.smc_particles,
        smc_ess_fraction=args.smc_ess_fraction,
        smc_move_steps=args.smc_move_steps,
        smc_max_stages=args.smc_max_stages,
    )
