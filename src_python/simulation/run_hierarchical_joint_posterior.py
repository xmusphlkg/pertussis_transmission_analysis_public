"""Optional legacy/research multi-country full-feedback Bayesian posterior.

This runner is not a Figure 2c interval source and is not part of the
publication pipeline. Figure 2c uses the separate estimation-CI bootstrap.

This module targets one joint probability model rather than the former
conditional/cut construction.  Seven biological parameters are shared across
countries.  Each country retains its own baseline transmission, reporting
multiplier, and annual latent log-transmission AR(1) path.  Surveillance data
from every country therefore update the shared parameters through the product
of country marginal likelihoods.

The nonlinear age-structured ODE cannot be evaluated inside Stan without a
second, independently maintained model implementation.  We instead use nested
self-normalised importance sampling against the exact existing NB2/AR(1)/ODE
target.  Stored Gauss-Newton state approximations are proposals only; every
candidate is re-evaluated under the complete target and under each proposed
shared structural draw. Global and local importance-support diagnostics remain
available for optional methodological research.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from joblib import Parallel, delayed, dump as joblib_dump, load as joblib_load
import numpy as np
import pandas as pd
from scipy import special as scipy_special
from scipy.optimize import least_squares

from src_python.calibration.calibrate_baseline import (
    _annual_ar1_transition_scales,
    _calibration_predicted_means,
    _gauss_newton_covariance,
    _negative_binomial_deviance_residuals,
    calibration_runtime_config,
    grouped_likelihood_observations,
    reporting_rate_prior_penalty,
)
from src_python.calibration.likelihood import negative_binomial_nll
from src_python.simulation.bayesian_priors import (
    BAYESIAN_LOCAL_STATE_PARAMETER_NAMES,
    BAYESIAN_SHARED_PARAMETER_NAMES,
    resolve_bayesian_prior_specs,
)
from src_python.simulation.common import (
    PROSPECTIVE_POLICY_KEY,
    config_fingerprint,
    current_run_metadata,
    file_sha256,
    load_calibrated_country_artifact,
    load_configs,
    make_config,
    publication_country_names,
    source_code_fingerprint,
    write_run_metadata,
)
from src_python.simulation.run_bayesian_uncertainty import (
    IMPORTANCE_NUISANCE_INDICES,
    PARAMETER_SAMPLE_COLUMNS,
    _build_nuisance_mixture_components,
    _country_observed,
    _importance_pareto_shape,
    _initial_vector,
    _minimum_importance_tail_ess,
    _mixture_nuisance_vectors,
    _normalise_log_weights,
    _prior_nuisance_vectors,
    _regularized_spd,
    _sample_from_vector,
    _state_space_exact_negative_log_target,
    _stratified_resample_indices,
    _tempered_importance_adaptation,
    _vector_from_sample,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import available_cpus, configure_worker_thread_limits


DEFAULT_STEM = "bayesian_uncertainty_joint_research"
ANALYSIS_ROLE = "optional_nonpublication_legacy_research"
PUBLICATION_PATH = False
FIGURE2C_INTERVAL_SOURCE = False
SAMPLING_METHOD = "multi_country_joint_state_importance"
INFERENCE_STRUCTURE = "cross_country_joint_state_space_exact_importance"
UNCERTAINTY_SCOPE = "multi_country_full_feedback_joint_state_space_posterior"
CHECKPOINT_SCHEMA_VERSION = "joint_state_exact_importance_v2"

SHARED_PARAMETER_NAMES = tuple(
    PARAMETER_SAMPLE_COLUMNS[index] for index in IMPORTANCE_NUISANCE_INDICES
)
LOCAL_PARAMETER_NAMES = (
    "beta_S",
    "reporting_multiplier",
    "annual_log_beta_process",
)
REGISTRY_PRIOR_PARAMETER_NAMES = BAYESIAN_SHARED_PARAMETER_NAMES
CALIBRATION_STATE_PRIOR_PARAMETER_NAMES = LOCAL_PARAMETER_NAMES
if SHARED_PARAMETER_NAMES != REGISTRY_PRIOR_PARAMETER_NAMES:
    raise RuntimeError(
        "Hierarchical shared parameters drifted from the Bayesian prior registry contract"
    )
if LOCAL_PARAMETER_NAMES[:2] != BAYESIAN_LOCAL_STATE_PARAMETER_NAMES:
    raise RuntimeError(
        "Hierarchical local-state parameters drifted from the calibration-prior contract"
    )

# These are numerical integration standards, not claims about biological fit.
# Predictive calibration and endpoint Monte Carlo error are separate gates.
RECOMMENDED_MIN_GLOBAL_ESS = 400.0
RECOMMENDED_MAX_GLOBAL_WEIGHT = 0.02
RECOMMENDED_MAX_GLOBAL_PARETO_K = 0.70
RECOMMENDED_MIN_GLOBAL_TAIL_ESS = 50.0
RECOMMENDED_MIN_LOCAL_ESS = 50.0
RECOMMENDED_MAX_LOCAL_WEIGHT = 0.10
RECOMMENDED_MAX_LOCAL_PARETO_K = 0.70
RECOMMENDED_MIN_LOCAL_TAIL_ESS = 10.0
RECOMMENDED_MAX_FAILED_LOCAL_POSTERIOR_MASS = 0.005
RECOMMENDED_MIN_STATE_IN_BOUNDS_FRACTION = 0.50
POSTERIOR_RELEVANT_MASS = 0.995
STATE_PROPOSAL_DEGREES_OF_FREEDOM = 5.0


@dataclass(frozen=True)
class CountryStateContext:
    country: str
    runtime_reference: dict[str, Any]
    prior_base: dict[str, Any]
    observed: pd.DataFrame
    priors: dict[str, Any]
    structural_start: np.ndarray
    coordinate_names: tuple[str, ...]
    state_mean: np.ndarray
    state_covariance: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    process_rho: float
    process_innovation_sd: float
    beta_prior_sd: float
    reporting_prior_sd: float
    dispersion: float
    historical_end_year: int
    forecast_end_year: int


@dataclass(frozen=True)
class StateProposal:
    candidates: np.ndarray
    log_density: np.ndarray
    in_bounds: np.ndarray
    source_component: np.ndarray
    covariance_scales: np.ndarray


@dataclass(frozen=True)
class StageEvaluation:
    log_marginal_by_country: dict[str, np.ndarray]
    local_log_weight_by_country: dict[str, np.ndarray]
    local_state_candidate_by_country: dict[str, np.ndarray] | None
    local_state_log_density_by_country: dict[str, np.ndarray] | None
    local_diagnostics: pd.DataFrame
    log_global_weight: np.ndarray
    global_weight: np.ndarray


def _general_student_t_mixture_proposal(
    component_means: np.ndarray,
    component_covariances: np.ndarray,
    component_weights: np.ndarray,
    *,
    n: int,
    seed: int,
    degrees_of_freedom: float = STATE_PROPOSAL_DEGREES_OF_FREEDOM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw and score a normalized heavy-tailed multivariate-t mixture.

    ``component_covariances`` are the desired component covariances.  The
    multivariate-t scale matrices are adjusted by ``(df - 2) / df`` so proposal
    covariance remains comparable to the adapted Gaussian geometry while the
    polynomial tails dominate the target's approximately Gaussian AR(1) tails.
    """

    means = np.asarray(component_means, dtype=float)
    covariances = np.asarray(component_covariances, dtype=float)
    weights = np.asarray(component_weights, dtype=float)
    df = float(degrees_of_freedom)
    if means.ndim != 2:
        raise ValueError("Student-t mixture means must be two-dimensional")
    component_count, dimension = means.shape
    if covariances.shape != (component_count, dimension, dimension):
        raise ValueError("Student-t mixture covariance dimension mismatch")
    if weights.shape != (component_count,) or n < 1:
        raise ValueError("Student-t mixture weight/count dimension mismatch")
    if df <= 2.0 or not np.isfinite(df):
        raise ValueError("Student-t proposal degrees of freedom must exceed two")
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("Student-t mixture weights must be finite and positive")
    weights = weights / float(weights.sum())

    expected_counts = float(n) * weights
    component_draw_counts = np.floor(expected_counts).astype(int)
    remainder = int(n - int(component_draw_counts.sum()))
    if remainder > 0:
        priority = np.argsort(-(expected_counts - component_draw_counts))
        component_draw_counts[priority[:remainder]] += 1
    allocation_weights = component_draw_counts.astype(float) / float(n)
    source_component = np.repeat(
        np.arange(component_count, dtype=int), component_draw_counts
    )
    rng = np.random.default_rng(int(seed))
    rng.shuffle(source_component)
    candidates = np.empty((int(n), dimension), dtype=float)
    scale_matrices: list[np.ndarray] = []
    information_matrices: list[np.ndarray] = []
    log_determinants: list[float] = []
    scale_factor = (df - 2.0) / df
    for component in range(component_count):
        covariance = _regularized_spd(covariances[component])
        scale = _regularized_spd(covariance * scale_factor)
        sign, log_determinant = np.linalg.slogdet(scale)
        if sign <= 0.0 or not np.isfinite(log_determinant):
            raise RuntimeError("Student-t mixture scale is not positive definite")
        scale_matrices.append(scale)
        information_matrices.append(np.linalg.inv(scale))
        log_determinants.append(float(log_determinant))
        positions = np.flatnonzero(source_component == component)
        if len(positions) > 0:
            normal = rng.multivariate_normal(
                np.zeros(dimension), scale, size=len(positions), check_valid="raise"
            )
            chi_square = rng.chisquare(df, size=len(positions))
            candidates[positions] = (
                means[component]
                + normal / np.sqrt(chi_square[:, None] / df)
            )

    log_component_density = np.empty((int(n), component_count), dtype=float)
    constant = float(
        scipy_special.gammaln((df + dimension) / 2.0)
        - scipy_special.gammaln(df / 2.0)
        - 0.5 * dimension * np.log(df * np.pi)
    )
    for component, (information, log_determinant) in enumerate(
        zip(information_matrices, log_determinants)
    ):
        delta = candidates - means[component]
        quadratic = np.einsum("ij,jk,ik->i", delta, information, delta)
        log_component_density[:, component] = (
            np.log(max(allocation_weights[component], 1e-300))
            + constant
            - 0.5 * log_determinant
            - 0.5 * (df + dimension) * np.log1p(quadratic / df)
        )
    return (
        candidates,
        scipy_special.logsumexp(log_component_density, axis=1),
        source_component,
    )


def _context(country: str, *, configs: dict[str, Any]) -> CountryStateContext:
    artifact = load_calibrated_country_artifact(country)
    if artifact is None:
        raise RuntimeError(f"No accepted state-space calibration artifact for {country}")
    approximation = artifact.get("metadata", {}).get("state_posterior_approximation", {})
    if approximation.get("method") != "gauss_newton_laplace_conditional":
        raise RuntimeError(f"Unsupported or missing state proposal for {country}")

    coordinate_names = tuple(str(value) for value in approximation.get("coordinate_names", []))
    if coordinate_names[:2] != ("log_beta_S", "log_reporting_multiplier"):
        raise RuntimeError(f"Invalid state coordinate contract for {country}")
    latent_names = coordinate_names[2:]
    if not latent_names or any(not value.startswith("log_beta_process_") for value in latent_names):
        raise RuntimeError(f"Missing annual latent state coordinates for {country}")
    latent_years = [int(value.removeprefix("log_beta_process_")) for value in latent_names]
    if any(current != previous + 1 for previous, current in zip(latent_years, latent_years[1:])):
        raise RuntimeError(f"Latent state years are not consecutive for {country}")

    state_mean = np.asarray(approximation.get("map_vector", []), dtype=float)
    state_covariance = np.asarray(approximation.get("covariance", []), dtype=float)
    lower = np.asarray(approximation.get("lower_bounds", []), dtype=float)
    upper = np.asarray(approximation.get("upper_bounds", []), dtype=float)
    dimension = len(coordinate_names)
    rank = int(approximation.get("rank", 0))
    condition = float(approximation.get("condition_number", np.inf))
    max_condition = float(
        configs["baseline"].get("calibration", {}).get("process_model", {}).get(
            "max_posterior_condition_number", 1e8
        )
    )
    if (
        len(state_mean) != dimension
        or state_covariance.shape != (dimension, dimension)
        or lower.shape != (dimension,)
        or upper.shape != (dimension,)
        or rank != dimension
        or not np.isfinite(condition)
        or condition > max_condition
    ):
        raise RuntimeError(
            f"State proposal failed the rank/conditioning gate for {country}: "
            f"rank={rank}/{dimension}, condition={condition:.6g}"
        )

    observed = _country_observed(
        country,
        interval=str(approximation.get("likelihood_observation_frequency", "annual")),
    )
    reference = make_config(country_profile=country, load_calibration=True)
    runtime_reference = calibration_runtime_config(reference, observed)
    prior_base = make_config(country_profile=country, load_calibration=False)

    settings = deepcopy(configs["baseline"]["bayesian_uncertainty"])
    settings["priors"] = dict(settings["priors"])
    settings["priors"]["resistance_prevalence_fixed"] = float(
        runtime_reference["resistance"]["target_prevalence_at_analysis_start"]
    )
    settings["priors"]["reporting_trend_fixed"] = 1.0
    settings["priors"]["parameterization"] = "beta_reporting_product"
    settings["priors"]["base_log_beta_S"] = float(
        np.log(float(runtime_reference["transmission"]["beta_S"]))
    )
    settings["priors"]["VE_dur_fixed"] = float(
        runtime_reference.get("vaccine", {}).get(
            "VE_dur", settings["priors"].get("VE_dur", {}).get("mean", 0.10)
        )
    )
    settings["priors"]["_distribution_specs"] = resolve_bayesian_prior_specs(
        configs.get("parameter_distributions", {}),
        prior_base,
        parameter_names=REGISTRY_PRIOR_PARAMETER_NAMES,
    )
    fitness_spec = settings["priors"]["_distribution_specs"]["fitness_R"]
    settings["priors"]["fitness_R"]["min"] = float(fitness_spec["low"])
    settings["priors"]["fitness_R"]["max"] = float(fitness_spec["high"])
    structural_start = _initial_vector(
        runtime_reference,
        enable_trend=False,
        priors=settings["priors"],
        start_at_prior_centers=False,
    )

    process_rho = float(approximation.get("process_ar1_rho", np.nan))
    process_innovation_sd = float(approximation.get("process_innovation_sd", np.nan))
    dispersion = float(approximation.get("measurement_dispersion", np.nan))
    if not 0.0 <= process_rho < 1.0 or process_innovation_sd <= 0.0 or dispersion <= 0.0:
        raise RuntimeError(f"Invalid state-space hyperparameters for {country}")

    return CountryStateContext(
        country=country,
        runtime_reference=runtime_reference,
        prior_base=prior_base,
        observed=observed,
        priors=settings["priors"],
        structural_start=structural_start,
        coordinate_names=coordinate_names,
        state_mean=state_mean,
        state_covariance=_regularized_spd(state_covariance),
        lower=lower,
        upper=upper,
        process_rho=process_rho,
        process_innovation_sd=process_innovation_sd,
        beta_prior_sd=float(approximation["beta_prior_log_sd"]),
        reporting_prior_sd=float(approximation["reporting_prior_log_sd"]),
        dispersion=dispersion,
        historical_end_year=max(latent_years),
        forecast_end_year=int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_end_date"]).year
        ),
    )


def _assert_shared_prior_contract(contexts: Iterable[CountryStateContext]) -> None:
    contexts = list(contexts)
    if not contexts:
        raise ValueError("At least one country is required")
    reference = contexts[0].priors["_distribution_specs"]
    for context in contexts[1:]:
        current = context.priors["_distribution_specs"]
        mismatch = [
            name for name in SHARED_PARAMETER_NAMES if current.get(name) != reference.get(name)
        ]
        if mismatch:
            raise ValueError(
                f"Shared structural priors differ for {context.country}: {mismatch}"
            )


def _state_proposal(
    context: CountryStateContext,
    *,
    count: int,
    covariance_scales: tuple[float, ...],
    seed: int,
) -> StateProposal:
    scales = np.asarray(covariance_scales, dtype=float)
    component_covariances = np.asarray(
        [
            _regularized_spd(context.state_covariance * float(scale))
            for scale in scales
        ],
        dtype=float,
    )
    candidates, log_density, source_component = _general_student_t_mixture_proposal(
        np.repeat(context.state_mean[None, :], len(scales), axis=0),
        component_covariances,
        np.full(len(scales), 1.0 / len(scales), dtype=float),
        n=int(count),
        seed=int(seed),
    )
    in_bounds = np.logical_and(
        np.all(candidates >= context.lower, axis=1),
        np.all(candidates <= context.upper, axis=1),
    )
    return StateProposal(
        candidates=candidates,
        log_density=log_density,
        in_bounds=in_bounds,
        source_component=source_component,
        covariance_scales=scales,
    )


def _state_proposal_from_moments(
    context: CountryStateContext,
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    count: int,
    covariance_scales: tuple[float, ...],
    seed: int,
) -> StateProposal:
    scales = np.asarray(covariance_scales, dtype=float)
    centre = np.asarray(mean, dtype=float)
    base_covariance = _regularized_spd(covariance)
    component_covariances = np.asarray(
        [
            _regularized_spd(base_covariance * float(scale))
            for scale in scales
        ],
        dtype=float,
    )
    candidates, log_density, source_component = _general_student_t_mixture_proposal(
        np.repeat(centre[None, :], len(scales), axis=0),
        component_covariances,
        np.full(len(scales), 1.0 / len(scales), dtype=float),
        n=int(count),
        seed=int(seed),
    )
    in_bounds = np.logical_and(
        np.all(candidates >= context.lower, axis=1),
        np.all(candidates <= context.upper, axis=1),
    )
    return StateProposal(
        candidates=candidates,
        log_density=log_density,
        in_bounds=in_bounds,
        source_component=source_component,
        covariance_scales=scales,
    )


def _state_proposal_from_posterior_draws(
    context: CountryStateContext,
    posterior_draws: np.ndarray,
    *,
    count: int,
    seed: int,
    max_local_components: int = 32,
) -> StateProposal:
    """Build a defensive KDE-like t mixture from exact-target adaptation draws."""

    draws = np.asarray(posterior_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(context.coordinate_names):
        raise ValueError("Posterior adaptation draw dimension mismatch")
    component_count = min(int(max_local_components), len(draws))
    positions = np.linspace(
        0, len(draws) - 1, component_count, dtype=int
    )
    local_means = draws[positions]
    centre = np.mean(draws, axis=0)
    covariance = _regularized_spd(np.cov(draws, rowvar=False, ddof=1))
    dimension = draws.shape[1]
    bandwidth_variance = float(
        max(component_count, 2) ** (-2.0 / (dimension + 4.0))
    )
    local_covariance = _regularized_spd(covariance * bandwidth_variance)
    component_means = np.vstack((local_means, centre, centre))
    component_covariances = np.asarray(
        [local_covariance] * component_count
        + [covariance, _regularized_spd(covariance * 9.0)],
        dtype=float,
    )
    component_weights = np.asarray(
        [0.80 / component_count] * component_count + [0.15, 0.05],
        dtype=float,
    )
    candidates, log_density, source_component = _general_student_t_mixture_proposal(
        component_means,
        component_covariances,
        component_weights,
        n=int(count),
        seed=int(seed),
    )
    in_bounds = np.logical_and(
        np.all(candidates >= context.lower, axis=1),
        np.all(candidates <= context.upper, axis=1),
    )
    return StateProposal(
        candidates=candidates,
        log_density=log_density,
        in_bounds=in_bounds,
        source_component=source_component,
        covariance_scales=np.asarray(
            [bandwidth_variance] * component_count + [1.0, 9.0], dtype=float
        ),
    )


def _reference_state_log_weight(
    context: CountryStateContext,
    proposal: StateProposal,
) -> np.ndarray:
    return _state_log_weight(
        context,
        proposal,
        runtime_base=context.runtime_reference,
        prior_base=context.prior_base,
    )


def _state_log_weight(
    context: CountryStateContext,
    proposal: StateProposal,
    *,
    runtime_base: dict[str, Any],
    prior_base: dict[str, Any],
) -> np.ndarray:
    raw = np.full(len(proposal.candidates), -np.inf, dtype=float)
    for position in np.flatnonzero(proposal.in_bounds):
        negative_log_target = _state_space_exact_negative_log_target(
            proposal.candidates[position],
            runtime_base=runtime_base,
            prior_base=prior_base,
            observed=context.observed,
            country=context.country,
            coordinate_names=list(context.coordinate_names),
            rho=context.process_rho,
            innovation_sd=context.process_innovation_sd,
            beta_prior_sd=context.beta_prior_sd,
            reporting_prior_sd=context.reporting_prior_sd,
            dispersion=context.dispersion,
        )
        if np.isfinite(negative_log_target):
            raw[position] = -float(negative_log_target) - float(
                proposal.log_density[position]
            )
    return raw


def _state_gaussian_prior_geometry(
    context: CountryStateContext,
    prior_base: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, covariance, and precision for the local Gaussian prior."""

    dimension = len(context.coordinate_names)
    mean = np.zeros(dimension, dtype=float)
    mean[0] = float(np.log(prior_base["transmission"]["beta_S"]))
    mean[1] = float(np.log(prior_base.get("reporting_multiplier", 1.0)))
    covariance = np.zeros((dimension, dimension), dtype=float)
    covariance[0, 0] = context.beta_prior_sd**2
    covariance[1, 1] = context.reporting_prior_sd**2

    latent_years = [
        int(name.removeprefix("log_beta_process_"))
        for name in context.coordinate_names[2:]
    ]
    state_starts = pd.Series(
        pd.to_datetime([f"{year:04d}-01-01" for year in latent_years])
    )
    state_ends = pd.Series(
        pd.to_datetime([f"{year + 1:04d}-01-01" for year in latent_years])
    )
    midpoints = state_starts + (state_ends - state_starts) / 2
    stationary_sd, transition_rho, transition_sd = _annual_ar1_transition_scales(
        midpoints,
        rho=context.process_rho,
        innovation_sd=context.process_innovation_sd,
    )
    latent_dimension = len(latent_years)
    innovation_map = np.zeros((latent_dimension, latent_dimension), dtype=float)
    innovation_map[0, 0] = float(stationary_sd)
    for position in range(1, latent_dimension):
        innovation_map[position] = (
            float(transition_rho[position - 1])
            * innovation_map[position - 1]
        )
        innovation_map[position, position] = float(transition_sd[position - 1])
    covariance[2:, 2:] = innovation_map @ innovation_map.T
    covariance = _regularized_spd(covariance)
    return mean, covariance, np.linalg.inv(covariance)


def _elliptical_slice_state_adaptation(
    context: CountryStateContext,
    *,
    start: np.ndarray,
    runtime_base: dict[str, Any],
    prior_base: dict[str, Any],
    count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Learn local posterior geometry with exact-target elliptical slice draws.

    The Gaussian beta/reporting/AR(1) prior defines each ellipse. Reporting-rate
    penalties and the complete NB2/ODE likelihood remain in the slice factor.
    These correlated draws adapt a later proposal only; fresh importance draws
    and exact weights determine the reported posterior and marginal likelihood.
    """

    prior_mean, prior_covariance, prior_precision = _state_gaussian_prior_geometry(
        context, prior_base
    )
    rng = np.random.default_rng(int(seed))
    current = np.asarray(start, dtype=float).copy()

    def log_slice_factor(coordinate: np.ndarray) -> float:
        if bool((coordinate < context.lower).any()) or bool(
            (coordinate > context.upper).any()
        ):
            return -np.inf
        negative_log_target = _state_space_exact_negative_log_target(
            coordinate,
            runtime_base=runtime_base,
            prior_base=prior_base,
            observed=context.observed,
            country=context.country,
            coordinate_names=list(context.coordinate_names),
            rho=context.process_rho,
            innovation_sd=context.process_innovation_sd,
            beta_prior_sd=context.beta_prior_sd,
            reporting_prior_sd=context.reporting_prior_sd,
            dispersion=context.dispersion,
        )
        if not np.isfinite(negative_log_target):
            return -np.inf
        delta = coordinate - prior_mean
        gaussian_penalty = 0.5 * float(delta @ prior_precision @ delta)
        return -float(negative_log_target) + gaussian_penalty

    current_factor = float(log_slice_factor(current))
    if not np.isfinite(current_factor):
        raise RuntimeError(
            f"Elliptical-slice adaptation has a non-finite start for {context.country}"
        )
    burnin = max(32, int(count) // 2)
    retained = np.empty((int(count), len(current)), dtype=float)
    total_evaluations = 1
    bracket_evaluations: list[int] = []
    for iteration in range(burnin + int(count)):
        direction = rng.multivariate_normal(
            np.zeros(len(current)), prior_covariance, check_valid="raise"
        )
        log_threshold = current_factor + float(np.log(rng.random()))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        angle_min = angle - 2.0 * np.pi
        angle_max = angle
        centred = current - prior_mean
        accepted = False
        evaluations = 0
        for _attempt in range(200):
            proposed = (
                prior_mean
                + centred * np.cos(angle)
                + direction * np.sin(angle)
            )
            proposed_factor = float(log_slice_factor(proposed))
            total_evaluations += 1
            evaluations += 1
            if proposed_factor >= log_threshold:
                current = proposed
                current_factor = proposed_factor
                accepted = True
                break
            if angle < 0.0:
                angle_min = angle
            else:
                angle_max = angle
            angle = float(rng.uniform(angle_min, angle_max))
        if not accepted:
            raise RuntimeError(
                f"Elliptical-slice bracket failed for {context.country}"
            )
        bracket_evaluations.append(evaluations)
        if iteration >= burnin:
            retained[iteration - burnin] = current
    return retained, {
        "state_ess_adaptation_draws": float(len(retained)),
        "state_ess_adaptation_burnin": float(burnin),
        "state_ess_target_evaluations": float(total_evaluations),
        "state_ess_mean_bracket_evaluations": float(
            np.mean(bracket_evaluations)
        ),
        "state_ess_max_bracket_evaluations": float(
            np.max(bracket_evaluations)
        ),
    }


def _localized_state_proposal(
    context: CountryStateContext,
    current: StateProposal,
    raw_log_weight: np.ndarray,
    *,
    count: int,
    seed: int,
    local_components: int = 8,
    fallback_covariance: np.ndarray | None = None,
) -> tuple[StateProposal, float]:
    """Build a defensive local mixture around exact-target high-weight states."""

    adapted_mean, adapted_covariance, exponent, _ = _tempered_importance_adaptation(
        current.candidates,
        raw_log_weight,
        (
            context.state_covariance
            if fallback_covariance is None
            else np.asarray(fallback_covariance, dtype=float)
        ),
        target_ess=max(32.0, 4.0 * len(context.coordinate_names)),
        covariance_regularization=0.05,
    )
    finite_positions = np.flatnonzero(np.isfinite(raw_log_weight))
    component_count = min(int(local_components), len(finite_positions))
    powered = exponent * raw_log_weight[finite_positions]
    powered -= float(scipy_special.logsumexp(powered))
    probability = np.exp(powered)
    rng = np.random.default_rng(int(seed))
    mode_position = int(finite_positions[np.argmax(raw_log_weight[finite_positions])])
    remaining = finite_positions[finite_positions != mode_position]
    selected = [mode_position]
    if component_count > 1:
        remaining_probability = probability[finite_positions != mode_position]
        remaining_probability /= float(remaining_probability.sum())
        selected.extend(
            rng.choice(
                remaining,
                size=component_count - 1,
                replace=False,
                p=remaining_probability,
            ).astype(int).tolist()
        )

    local_covariance = _regularized_spd(adapted_covariance * 0.25)
    component_means = np.vstack(
        (
            current.candidates[np.asarray(selected, dtype=int)],
            adapted_mean,
            adapted_mean,
            adapted_mean,
        )
    )
    component_covariances = np.asarray(
        [local_covariance] * component_count
        + [
            _regularized_spd(adapted_covariance),
            _regularized_spd(adapted_covariance * 4.00),
            _regularized_spd(adapted_covariance * 16.00),
        ],
        dtype=float,
    )
    component_weights = np.asarray(
        [0.70 / component_count] * component_count + [0.15, 0.10, 0.05],
        dtype=float,
    )
    candidates, log_density, source_component = _general_student_t_mixture_proposal(
        component_means,
        component_covariances,
        component_weights,
        n=int(count),
        seed=int(seed) + 17,
    )
    scales = np.asarray(
        [0.25] * component_count + [1.00, 4.00, 16.00], dtype=float
    )
    in_bounds = np.logical_and(
        np.all(candidates >= context.lower, axis=1),
        np.all(candidates <= context.upper, axis=1),
    )
    return (
        StateProposal(
            candidates=candidates,
            log_density=log_density,
            in_bounds=in_bounds,
            source_component=source_component,
            covariance_scales=scales,
        ),
        float(exponent),
    )


def _adapt_state_proposal(
    context: CountryStateContext,
    *,
    count: int,
    final_covariance_scales: tuple[float, ...],
    seed: int,
    rounds: int = 2,
    localized_rounds: int = 3,
) -> tuple[StateProposal, list[dict[str, Any]]]:
    """Adapt a fresh state proposal using exact reference-structure weights."""

    proposal = _state_proposal(
        context,
        count=count,
        covariance_scales=(0.0625, 0.25, 1.0, 4.0),
        seed=seed,
    )
    fallback_covariance = context.state_covariance
    rows: list[dict[str, Any]] = []
    for round_index in range(1, int(rounds) + 1):
        raw = _reference_state_log_weight(context, proposal)
        finite = np.isfinite(raw)
        if int(finite.sum()) < max(32, len(context.coordinate_names) * 4):
            raise RuntimeError(
                f"Reference state adaptation has insufficient finite support for "
                f"{context.country}: {int(finite.sum())}/{len(raw)}"
            )
        ess, maximum, pareto_k, tail_ess, _ = _local_weight_summary(
            proposal.candidates, raw
        )
        rows.append(
            {
                "country": context.country,
                "adaptation_round": round_index,
                "candidate_count": int(len(raw)),
                "finite_candidate_count": int(finite.sum()),
                "in_bounds_fraction": float(np.mean(proposal.in_bounds)),
                "raw_effective_sample_size": ess,
                "raw_maximum_weight": maximum,
                "raw_pareto_k": pareto_k,
                "raw_minimum_tail_ess": tail_ess,
            }
        )
        adapted_mean, adapted_covariance, exponent, _raw_ess = (
            _tempered_importance_adaptation(
                proposal.candidates,
                raw,
                fallback_covariance,
                target_ess=max(32.0, 4.0 * len(context.coordinate_names)),
                covariance_regularization=0.10,
            )
        )
        rows[-1]["tempering_exponent"] = float(exponent)
        fallback_covariance = adapted_covariance
        proposal = _state_proposal_from_moments(
            context,
            mean=adapted_mean,
            covariance=adapted_covariance,
            count=count,
            covariance_scales=final_covariance_scales,
            seed=seed + round_index * 1009,
        )

    for localized_round in range(1, int(localized_rounds) + 1):
        raw = _reference_state_log_weight(context, proposal)
        ess, maximum, pareto_k, tail_ess, _ = _local_weight_summary(
            proposal.candidates, raw
        )
        proposal, exponent = _localized_state_proposal(
            context,
            proposal,
            raw,
            count=count,
            seed=seed + 10007 + localized_round * 1009,
        )
        rows.append(
            {
                "country": context.country,
                "adaptation_round": int(rounds) + localized_round,
                "adaptation_type": "localized_mixture_input",
                "candidate_count": int(len(raw)),
                "finite_candidate_count": int(np.isfinite(raw).sum()),
                "in_bounds_fraction": float(np.mean(proposal.in_bounds)),
                "raw_effective_sample_size": ess,
                "raw_maximum_weight": maximum,
                "raw_pareto_k": pareto_k,
                "raw_minimum_tail_ess": tail_ess,
                "tempering_exponent": exponent,
            }
        )

    final_raw = _reference_state_log_weight(context, proposal)
    finite = np.isfinite(final_raw)
    if not finite.any():
        raise RuntimeError(f"Adapted state proposal has no finite support for {context.country}")
    ess, maximum, pareto_k, tail_ess, _ = _local_weight_summary(
        proposal.candidates, final_raw
    )
    rows.append(
        {
            "country": context.country,
            "adaptation_round": int(rounds) + int(localized_rounds) + 1,
            "adaptation_type": "final_localized_mixture",
            "candidate_count": int(len(final_raw)),
            "finite_candidate_count": int(finite.sum()),
            "in_bounds_fraction": float(np.mean(proposal.in_bounds)),
            "raw_effective_sample_size": ess,
            "raw_maximum_weight": maximum,
            "raw_pareto_k": pareto_k,
            "raw_minimum_tail_ess": tail_ess,
            "tempering_exponent": np.nan,
        }
    )
    return proposal, rows


def _warm_started_state_map(
    context: CountryStateContext,
    structural_runtime_base: dict[str, Any],
    structural_prior_base: dict[str, Any],
    *,
    optimizer_maxiter: int,
) -> tuple[dict[str, Any], Any]:
    """Refit the annual state target from the accepted country MAP warm start."""

    latent_years = [
        int(name.removeprefix("log_beta_process_"))
        for name in context.coordinate_names[2:]
    ]
    state_starts = pd.Series(
        pd.to_datetime([f"{year:04d}-01-01" for year in latent_years])
    )
    state_ends = pd.Series(
        pd.to_datetime([f"{year + 1:04d}-01-01" for year in latent_years])
    )
    state_midpoints = state_starts + (state_ends - state_starts) / 2
    stationary_sd, transition_rho, transition_sd = _annual_ar1_transition_scales(
        state_midpoints,
        rho=context.process_rho,
        innovation_sd=context.process_innovation_sd,
    )
    likelihood_observed = grouped_likelihood_observations(context.observed)
    observed_values = pd.to_numeric(
        likelihood_observed["reported_cases"], errors="raise"
    ).to_numpy(dtype=float)
    exposure_cache: dict[tuple[Any, ...], tuple[Any, np.ndarray]] = {}

    def configured(vector: np.ndarray) -> dict[str, Any]:
        candidate = deepcopy(structural_runtime_base)
        candidate["transmission"]["beta_S"] = float(np.exp(vector[0]))
        candidate["reporting_multiplier"] = float(np.exp(vector[1]))
        candidate["transmission"]["log_beta_time_variation"] = {
            "enabled": True,
            "interpretation": "latent_AR1_process_joint_posterior_map",
            "ar1_rho": context.process_rho,
            "innovation_sd": context.process_innovation_sd,
            "periods": [
                {
                    "start_date": pd.Timestamp(start).date().isoformat(),
                    "end_date": (
                        pd.Timestamp(end) - pd.Timedelta(days=1)
                    ).date().isoformat(),
                    "log_multiplier": float(value),
                }
                for start, end, value in zip(
                    state_starts, state_ends, vector[2:]
                )
            ],
        }
        return candidate

    beta_center = float(np.log(structural_prior_base["transmission"]["beta_S"]))
    reporting_center = float(
        np.log(structural_prior_base.get("reporting_multiplier", 1.0))
    )

    def residuals(vector: np.ndarray) -> np.ndarray:
        candidate = configured(vector)
        prediction = _calibration_predicted_means(
            candidate,
            context.observed,
            context.country,
            exposure_cache=exposure_cache,
        )
        data_residual = _negative_binomial_deviance_residuals(
            observed_values,
            prediction,
            context.dispersion,
        )
        latent = np.asarray(vector[2:], dtype=float)
        process_residual = np.empty(len(latent), dtype=float)
        process_residual[0] = latent[0] / stationary_sd
        if len(latent) > 1:
            process_residual[1:] = (
                latent[1:] - transition_rho * latent[:-1]
            ) / transition_sd
        reporting_penalty = reporting_rate_prior_penalty(candidate)
        return np.concatenate(
            (
                data_residual,
                process_residual,
                np.asarray(
                    [
                        (vector[0] - beta_center) / context.beta_prior_sd,
                        (vector[1] - reporting_center)
                        / context.reporting_prior_sd,
                        np.sqrt(max(2.0 * reporting_penalty, 0.0)),
                    ]
                ),
            )
        )

    start = np.clip(
        np.asarray(context.state_mean, dtype=float),
        context.lower + 1e-10,
        context.upper - 1e-10,
    )
    max_nfev = max(40, int(optimizer_maxiter) * 8)
    fit = least_squares(
        residuals,
        start,
        bounds=(context.lower, context.upper),
        max_nfev=max_nfev,
        xtol=1e-4,
        ftol=1e-4,
        gtol=1e-4,
    )
    retry_used = False
    if not fit.success:
        retry_used = True
        fit = least_squares(
            residuals,
            np.clip(fit.x, context.lower + 1e-10, context.upper - 1e-10),
            bounds=(context.lower, context.upper),
            max_nfev=max_nfev * 2,
            xtol=5e-5,
            ftol=5e-5,
            gtol=5e-5,
        )
    calibrated = configured(fit.x)
    covariance, rank, condition = _gauss_newton_covariance(fit.jac)
    prediction = _calibration_predicted_means(
        calibrated,
        context.observed,
        context.country,
        exposure_cache=exposure_cache,
    )
    latent = np.asarray(fit.x[2:], dtype=float)
    process_penalty = 0.5 * float((latent[0] / stationary_sd) ** 2)
    if len(latent) > 1:
        process_penalty += 0.5 * float(
            np.sum(
                ((latent[1:] - transition_rho * latent[:-1]) / transition_sd)
                ** 2
            )
        )
    process_penalty += 0.5 * float(
        ((fit.x[0] - beta_center) / context.beta_prior_sd) ** 2
        + ((fit.x[1] - reporting_center) / context.reporting_prior_sd) ** 2
    )
    score = float(
        negative_binomial_nll(observed_values, prediction, context.dispersion)
        + process_penalty
        + reporting_rate_prior_penalty(calibrated)
    )
    result = type("WarmStateMapResult", (), {})()
    result.success = bool(fit.success and np.isfinite(score))
    result.message = str(fit.message)
    result.posterior_map_vector = np.asarray(fit.x, dtype=float)
    result.posterior_covariance = covariance
    result.posterior_rank = int(rank)
    result.posterior_condition_number = float(condition)
    result.process_cost = float(fit.cost)
    result.process_optimality = float(fit.optimality)
    result.process_nfev = int(fit.nfev)
    result.retry_used = bool(retry_used)
    result.score = score
    return calibrated, result


def _conditional_state_importance(
    context: CountryStateContext,
    structural_vector: np.ndarray,
    *,
    state_adaptation_candidates: int,
    state_candidates: int,
    covariance_scales: tuple[float, ...],
    optimizer_maxiter: int,
    seed: int,
    adaptive_rounds: int,
    localized_rounds: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate one country's state posterior for one shared structural draw."""

    structural_sample = _sample_from_vector(structural_vector, context.priors)
    structural_prior_base = _apply_shared_structure(
        context.prior_base, structural_sample
    )
    structural_runtime_base = calibration_runtime_config(
        structural_prior_base, context.observed
    )
    calibrated, fit = _warm_started_state_map(
        context,
        structural_runtime_base,
        structural_prior_base,
        optimizer_maxiter=int(optimizer_maxiter),
    )
    mean = np.asarray(fit.posterior_map_vector, dtype=float)
    covariance = _regularized_spd(
        np.asarray(fit.posterior_covariance, dtype=float)
    )
    if (
        int(fit.posterior_rank) != len(context.coordinate_names)
        or not np.isfinite(float(fit.posterior_condition_number))
        or not np.isfinite(np.asarray(fit.posterior_map_vector, dtype=float)).all()
        or not np.isfinite(np.asarray(fit.posterior_covariance, dtype=float)).all()
    ):
        raise RuntimeError(
            f"Conditional state optimization failed for {context.country}: "
            f"optimizer_success={fit.success}, rank={fit.posterior_rank}/"
            f"{len(context.coordinate_names)}, condition={fit.posterior_condition_number}"
        )
    runtime = calibration_runtime_config(calibrated, context.observed)
    elliptical_draws, elliptical_diagnostics = (
        _elliptical_slice_state_adaptation(
            context,
            start=mean,
            runtime_base=runtime,
            prior_base=structural_prior_base,
            count=int(state_adaptation_candidates),
            seed=int(seed) + 503,
        )
    )
    adapted_covariance = _regularized_spd(
        np.cov(elliptical_draws, rowvar=False, ddof=1)
    )
    proposal = _state_proposal_from_posterior_draws(
        context,
        elliptical_draws,
        count=(
            int(state_candidates)
            if int(adaptive_rounds) == 0 and int(localized_rounds) == 0
            else int(state_adaptation_candidates)
        ),
        seed=int(seed),
    )
    raw = _state_log_weight(
        context,
        proposal,
        runtime_base=runtime,
        prior_base=structural_prior_base,
    )
    fallback_covariance = adapted_covariance
    for adaptive_round in range(1, int(adaptive_rounds) + 1):
        adapted_mean, adapted_covariance, _exponent, _ = (
            _tempered_importance_adaptation(
                proposal.candidates,
                raw,
                fallback_covariance,
                target_ess=max(64.0, 8.0 * len(context.coordinate_names)),
                covariance_regularization=0.10,
            )
        )
        fallback_covariance = adapted_covariance
        proposal = _state_proposal_from_moments(
            context,
            mean=adapted_mean,
            covariance=adapted_covariance,
            count=int(state_adaptation_candidates),
            covariance_scales=covariance_scales,
            seed=int(seed) + adaptive_round * 2003,
        )
        raw = _state_log_weight(
            context,
            proposal,
            runtime_base=runtime,
            prior_base=structural_prior_base,
        )
    for localized_round in range(1, int(localized_rounds) + 1):
        proposal, _ = _localized_state_proposal(
            context,
            proposal,
            raw,
            count=(
                int(state_candidates)
                if localized_round == int(localized_rounds)
                else int(state_adaptation_candidates)
            ),
            seed=(
                int(seed)
                + int(adaptive_rounds) * 2003
                + localized_round * 1009
            ),
            fallback_covariance=fallback_covariance,
        )
        raw = _state_log_weight(
            context,
            proposal,
            runtime_base=runtime,
            prior_base=structural_prior_base,
        )
    finite = np.isfinite(raw)
    if not finite.any():
        raise RuntimeError(
            f"Conditional state importance has no finite support for {context.country}"
        )
    log_marginal = float(
        scipy_special.logsumexp(raw[finite]) - np.log(len(raw))
    )
    ess, maximum, pareto_k, tail_ess, _ = _local_weight_summary(
        proposal.candidates, raw
    )
    diagnostic = {
        "country": context.country,
        "log_marginal_likelihood": log_marginal,
        "state_map_cost": float(fit.process_cost),
        "state_map_optimizer_success": bool(fit.success),
        "state_map_message": str(fit.message),
        "state_map_optimality": float(fit.process_optimality),
        "state_map_nfev": int(fit.process_nfev),
        "state_map_retry_used": bool(fit.retry_used),
        "state_posterior_rank": int(fit.posterior_rank),
        "state_posterior_dimension": int(len(context.coordinate_names)),
        "state_posterior_condition_number": float(
            fit.posterior_condition_number
        ),
        "state_effective_sample_size": ess,
        "state_maximum_weight": maximum,
        "state_pareto_k": pareto_k,
        "state_minimum_tail_ess": tail_ess,
        "state_in_bounds_fraction": float(np.mean(proposal.in_bounds)),
        **elliptical_diagnostics,
    }
    return (
        log_marginal,
        proposal.candidates,
        proposal.log_density,
        raw,
        diagnostic,
    )


def _apply_shared_structure(
    runtime_reference: dict[str, Any],
    structural_sample: dict[str, float],
) -> dict[str, Any]:
    out = deepcopy(runtime_reference)
    out["vaccine"]["VE_sus"] = float(structural_sample["VE_sus"])
    out["vaccine"]["VE_inf"] = float(structural_sample["VE_inf"])
    out["vaccine"]["VE_dur"] = float(structural_sample["VE_dur"])
    out["transmission"]["relative_infectiousness_asymptomatic"] = float(
        structural_sample["relative_infectiousness_asymptomatic"]
    )
    out["natural_history"]["infectious_duration_symptomatic"] = float(
        structural_sample["infectious_duration_symptomatic"]
    )
    out["natural_history"]["infectious_duration_asymptomatic"] = float(
        structural_sample["infectious_duration_asymptomatic"]
    )
    out["transmission"]["fitness_R"] = float(structural_sample["fitness_R"])
    policy = out.get(PROSPECTIVE_POLICY_KEY)
    if isinstance(policy, dict) and isinstance(policy.get("history_config"), dict):
        history = policy["history_config"]
        history["vaccine"]["VE_sus"] = out["vaccine"]["VE_sus"]
        history["vaccine"]["VE_inf"] = out["vaccine"]["VE_inf"]
        history["vaccine"]["VE_dur"] = out["vaccine"]["VE_dur"]
        history["transmission"]["relative_infectiousness_asymptomatic"] = out[
            "transmission"
        ]["relative_infectiousness_asymptomatic"]
        history["natural_history"]["infectious_duration_symptomatic"] = out[
            "natural_history"
        ]["infectious_duration_symptomatic"]
        history["natural_history"]["infectious_duration_asymptomatic"] = out[
            "natural_history"
        ]["infectious_duration_asymptomatic"]
        history["transmission"]["fitness_R"] = out["transmission"]["fitness_R"]
    return out


def _local_weight_summary(
    candidates: np.ndarray,
    raw_log_weight: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    probability = _normalise_log_weights(raw_log_weight)
    ess = float(1.0 / np.sum(probability**2))
    maximum = float(np.max(probability))
    pareto_k = float(_importance_pareto_shape(raw_log_weight))
    tail_ess = float(_minimum_importance_tail_ess(candidates, probability))
    return ess, maximum, pareto_k, tail_ess, probability


def _evaluate_country_structural_chunk(
    context: CountryStateContext,
    structural_vectors: np.ndarray,
    structural_indices: np.ndarray,
    *,
    state_adaptation_candidates: int,
    state_candidates: int,
    covariance_scales: tuple[float, ...],
    optimizer_maxiter: int,
    adaptive_rounds: int,
    localized_rounds: int,
    seed: int,
    retain_state_candidates: bool,
    checkpoint_path: Path | None = None,
    checkpoint_fingerprint: str | None = None,
) -> dict[str, Any]:
    configure_worker_thread_limits()
    if checkpoint_path is not None and checkpoint_path.exists():
        cached = joblib_load(checkpoint_path)
        if (
            isinstance(cached, dict)
            and cached.get("checkpoint_fingerprint") == checkpoint_fingerprint
        ):
            return cached["result"]
    indices = np.asarray(structural_indices, dtype=int)
    state_count = int(state_candidates)
    log_weight_matrix = np.full((len(indices), state_count), -np.inf, dtype=float)
    state_candidate_cube = (
        np.full(
            (len(indices), state_count, len(context.coordinate_names)),
            np.nan,
            dtype=float,
        )
        if retain_state_candidates
        else None
    )
    state_log_density_matrix = (
        np.full((len(indices), state_count), np.nan, dtype=float)
        if retain_state_candidates
        else None
    )
    log_marginal = np.full(len(indices), -np.inf, dtype=float)
    rows: list[dict[str, Any]] = []
    for local_position, structural_index in enumerate(indices):
        (
            local_log_marginal,
            local_candidates,
            local_log_density,
            local_log_weight,
            diagnostic,
        ) = _conditional_state_importance(
            context,
            structural_vectors[int(structural_index)],
            state_adaptation_candidates=state_adaptation_candidates,
            state_candidates=state_count,
            covariance_scales=covariance_scales,
            optimizer_maxiter=optimizer_maxiter,
            adaptive_rounds=adaptive_rounds,
            seed=(
                int(seed)
                + int(structural_index) * 104729
                + sum(ord(value) for value in context.country) * 1009
            ),
            localized_rounds=localized_rounds,
        )
        log_marginal[local_position] = local_log_marginal
        log_weight_matrix[local_position] = local_log_weight
        if retain_state_candidates:
            assert state_candidate_cube is not None
            assert state_log_density_matrix is not None
            state_candidate_cube[local_position] = local_candidates
            state_log_density_matrix[local_position] = local_log_density
        diagnostic["structural_index"] = int(structural_index)
        rows.append(diagnostic)
    result = {
        "country": context.country,
        "indices": indices,
        "log_marginal": log_marginal,
        "log_weight_matrix": log_weight_matrix,
        "state_candidate_cube": state_candidate_cube,
        "state_log_density_matrix": state_log_density_matrix,
        "diagnostics": rows,
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(
            checkpoint_path.suffix + f".{os.getpid()}.tmp"
        )
        joblib_dump(
            {
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "result": result,
            },
            temporary,
            compress=1,
        )
        temporary.replace(checkpoint_path)
    return result


def _evaluate_stage(
    contexts: list[CountryStateContext],
    structural_vectors: np.ndarray,
    structural_log_prior: np.ndarray,
    structural_log_proposal: np.ndarray,
    *,
    state_adaptation_candidates: int,
    state_candidates: int,
    state_covariance_scales: tuple[float, ...],
    optimizer_maxiter: int,
    adaptive_rounds: int,
    localized_rounds: int,
    n_jobs: int,
    seed: int,
    retain_state_candidates: bool,
    checkpoint_dir: Path | None = None,
) -> StageEvaluation:
    structural_count = len(structural_vectors)
    workers = min(max(1, int(n_jobs)), available_cpus())
    run_fingerprint = hashlib.sha256()
    run_fingerprint.update(CHECKPOINT_SCHEMA_VERSION.encode("utf-8"))
    run_fingerprint.update(config_fingerprint().encode("utf-8"))
    run_fingerprint.update(source_code_fingerprint().encode("utf-8"))
    # A single structural draw is already a long-running unit.  One-draw
    # chunks provide useful load balancing and make checkpoints granular enough
    # that a late failure never discards hours of completed inner integrations.
    chunks = [np.asarray([index], dtype=int) for index in range(structural_count)]
    tasks: list[tuple[CountryStateContext, np.ndarray, Path | None, str]] = []
    for context in contexts:
        context_fingerprint = run_fingerprint.copy()
        context_fingerprint.update(context.country.encode("utf-8"))
        context_fingerprint.update(repr(context.coordinate_names).encode("utf-8"))
        for array in (
            context.state_mean,
            context.state_covariance,
            context.lower,
            context.upper,
        ):
            context_fingerprint.update(np.asarray(array, dtype=np.float64).tobytes())
        context_fingerprint.update(
            pd.util.hash_pandas_object(
                context.observed, index=True
            ).to_numpy(dtype=np.uint64).tobytes()
        )
        context_fingerprint.update(repr(context.prior_base).encode("utf-8"))
        context_fingerprint.update(repr(context.runtime_reference).encode("utf-8"))
        context_fingerprint.update(
            repr(
                (
                    context.process_rho,
                    context.process_innovation_sd,
                    context.beta_prior_sd,
                    context.reporting_prior_sd,
                    context.dispersion,
                    context.historical_end_year,
                    context.forecast_end_year,
                )
            ).encode("utf-8")
        )
        for chunk in chunks:
            digest = context_fingerprint.copy()
            digest.update(np.asarray(chunk, dtype=np.int64).tobytes())
            digest.update(
                np.asarray(structural_vectors[chunk], dtype=np.float64).tobytes()
            )
            digest.update(
                repr(
                    (
                        state_adaptation_candidates,
                        state_candidates,
                        state_covariance_scales,
                        optimizer_maxiter,
                        adaptive_rounds,
                        localized_rounds,
                        seed,
                        retain_state_candidates,
                    )
                ).encode("utf-8")
            )
            fingerprint = digest.hexdigest()
            checkpoint_path = None
            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / (
                    f"{context.country}_{int(chunk[0]):06d}_{int(chunk[-1]):06d}.joblib"
                )
            tasks.append((context, chunk, checkpoint_path, fingerprint))
    worker_count = min(workers, len(tasks))
    if worker_count == 1:
        parts = [
            _evaluate_country_structural_chunk(
                context,
                structural_vectors,
                chunk,
                state_adaptation_candidates=state_adaptation_candidates,
                state_candidates=state_candidates,
                covariance_scales=state_covariance_scales,
                optimizer_maxiter=optimizer_maxiter,
                adaptive_rounds=adaptive_rounds,
                localized_rounds=localized_rounds,
                seed=seed,
                retain_state_candidates=retain_state_candidates,
                checkpoint_path=checkpoint_path,
                checkpoint_fingerprint=fingerprint,
            )
            for context, chunk, checkpoint_path, fingerprint in tasks
        ]
    else:
        parts = Parallel(n_jobs=worker_count, backend="loky", inner_max_num_threads=1)(
            delayed(_evaluate_country_structural_chunk)(
                context,
                structural_vectors,
                chunk,
                state_adaptation_candidates=state_adaptation_candidates,
                state_candidates=state_candidates,
                covariance_scales=state_covariance_scales,
                optimizer_maxiter=optimizer_maxiter,
                adaptive_rounds=adaptive_rounds,
                localized_rounds=localized_rounds,
                seed=seed,
                retain_state_candidates=retain_state_candidates,
                checkpoint_path=checkpoint_path,
                checkpoint_fingerprint=fingerprint,
            )
            for context, chunk, checkpoint_path, fingerprint in tasks
        )

    log_marginal_by_country: dict[str, np.ndarray] = {}
    local_log_weight_by_country: dict[str, np.ndarray] = {}
    local_state_candidate_by_country: dict[str, np.ndarray] | None = (
        {} if retain_state_candidates else None
    )
    local_state_log_density_by_country: dict[str, np.ndarray] | None = (
        {} if retain_state_candidates else None
    )
    diagnostic_rows: list[dict[str, Any]] = []
    for context in contexts:
        state_count = int(state_candidates)
        marginal = np.full(structural_count, -np.inf, dtype=float)
        local_weight = np.full((structural_count, state_count), -np.inf, dtype=float)
        candidate_cube = (
            np.full(
                (structural_count, state_count, len(context.coordinate_names)),
                np.nan,
                dtype=float,
            )
            if retain_state_candidates
            else None
        )
        log_density_matrix = (
            np.full((structural_count, state_count), np.nan, dtype=float)
            if retain_state_candidates
            else None
        )
        for part in parts:
            if part["country"] != context.country:
                continue
            indices = part["indices"]
            marginal[indices] = part["log_marginal"]
            local_weight[indices, :] = part["log_weight_matrix"]
            if retain_state_candidates:
                assert candidate_cube is not None and log_density_matrix is not None
                candidate_cube[indices] = part["state_candidate_cube"]
                log_density_matrix[indices] = part["state_log_density_matrix"]
            diagnostic_rows.extend(part["diagnostics"])
        log_marginal_by_country[context.country] = marginal
        local_log_weight_by_country[context.country] = local_weight
        if retain_state_candidates:
            assert local_state_candidate_by_country is not None
            assert local_state_log_density_by_country is not None
            local_state_candidate_by_country[context.country] = candidate_cube
            local_state_log_density_by_country[context.country] = log_density_matrix

    marginal_matrix = np.column_stack(
        [log_marginal_by_country[context.country] for context in contexts]
    )
    log_global_weight = (
        np.asarray(structural_log_prior, dtype=float)
        - np.asarray(structural_log_proposal, dtype=float)
        + np.sum(marginal_matrix, axis=1)
    )
    global_weight = _normalise_log_weights(log_global_weight)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["country", "structural_index"]
    ).reset_index(drop=True)
    return StageEvaluation(
        log_marginal_by_country=log_marginal_by_country,
        local_log_weight_by_country=local_log_weight_by_country,
        local_state_candidate_by_country=local_state_candidate_by_country,
        local_state_log_density_by_country=local_state_log_density_by_country,
        local_diagnostics=diagnostics,
        log_global_weight=log_global_weight,
        global_weight=global_weight,
    )


def _posterior_relevant_indices(weight: np.ndarray, mass: float = POSTERIOR_RELEVANT_MASS) -> np.ndarray:
    probability = np.asarray(weight, dtype=float)
    order = np.argsort(-probability)
    cumulative = np.cumsum(probability[order])
    count = int(np.searchsorted(cumulative, float(mass), side="left") + 1)
    return np.sort(order[:count])


def _quality(
    structural_vectors: np.ndarray,
    stage: StageEvaluation,
) -> dict[str, Any]:
    weight = stage.global_weight
    coordinates = structural_vectors[:, IMPORTANCE_NUISANCE_INDICES]
    global_ess = float(1.0 / np.sum(weight**2))
    global_maximum = float(np.max(weight))
    global_pareto = float(_importance_pareto_shape(stage.log_global_weight))
    global_tail = float(_minimum_importance_tail_ess(coordinates, weight))

    diagnostics = stage.local_diagnostics.copy()
    diagnostics["structural_posterior_weight"] = diagnostics["structural_index"].map(
        dict(enumerate(weight))
    )
    diagnostics["local_recommended"] = (
        diagnostics["state_effective_sample_size"].ge(RECOMMENDED_MIN_LOCAL_ESS)
        & diagnostics["state_maximum_weight"].le(RECOMMENDED_MAX_LOCAL_WEIGHT)
        & diagnostics["state_pareto_k"].le(RECOMMENDED_MAX_LOCAL_PARETO_K)
        & diagnostics["state_minimum_tail_ess"].ge(RECOMMENDED_MIN_LOCAL_TAIL_ESS)
        & diagnostics["state_in_bounds_fraction"].ge(
            RECOMMENDED_MIN_STATE_IN_BOUNDS_FRACTION
        )
    )
    failure_by_structural = (
        diagnostics.groupby("structural_index", sort=False)["local_recommended"]
        .all()
        .reindex(range(len(weight)), fill_value=False)
    )
    failed_mass = float(weight[~failure_by_structural.to_numpy(dtype=bool)].sum())
    relevant = _posterior_relevant_indices(weight)
    relevant_diagnostics = diagnostics.loc[
        diagnostics["structural_index"].isin(relevant)
    ]
    minimum_local_ess = float(relevant_diagnostics["state_effective_sample_size"].min())
    maximum_local_weight = float(relevant_diagnostics["state_maximum_weight"].max())
    maximum_local_pareto = float(relevant_diagnostics["state_pareto_k"].max())
    minimum_local_tail = float(relevant_diagnostics["state_minimum_tail_ess"].min())
    minimum_in_bounds = float(relevant_diagnostics["state_in_bounds_fraction"].min())

    checks = {
        "global_effective_sample_size": global_ess >= RECOMMENDED_MIN_GLOBAL_ESS,
        "global_maximum_weight": global_maximum <= RECOMMENDED_MAX_GLOBAL_WEIGHT,
        "global_pareto_k": global_pareto <= RECOMMENDED_MAX_GLOBAL_PARETO_K,
        "global_minimum_tail_ess": global_tail >= RECOMMENDED_MIN_GLOBAL_TAIL_ESS,
        "local_failed_posterior_mass": (
            failed_mass <= RECOMMENDED_MAX_FAILED_LOCAL_POSTERIOR_MASS
        ),
        "local_relevant_minimum_ess": minimum_local_ess >= RECOMMENDED_MIN_LOCAL_ESS,
        "local_relevant_maximum_weight": maximum_local_weight <= RECOMMENDED_MAX_LOCAL_WEIGHT,
        "local_relevant_maximum_pareto_k": maximum_local_pareto <= RECOMMENDED_MAX_LOCAL_PARETO_K,
        "local_relevant_minimum_tail_ess": minimum_local_tail >= RECOMMENDED_MIN_LOCAL_TAIL_ESS,
        "state_proposal_in_bounds_fraction": (
            minimum_in_bounds >= RECOMMENDED_MIN_STATE_IN_BOUNDS_FRACTION
        ),
    }
    return {
        "global_importance_effective_sample_size": global_ess,
        "global_importance_ess_fraction": global_ess / max(len(weight), 1),
        "global_importance_maximum_weight": global_maximum,
        "global_importance_pareto_k": global_pareto,
        "global_importance_minimum_tail_ess": global_tail,
        "posterior_relevant_structural_candidates": int(len(relevant)),
        "local_failed_posterior_mass": failed_mass,
        "local_relevant_minimum_ess": minimum_local_ess,
        "local_relevant_maximum_weight": maximum_local_weight,
        "local_relevant_maximum_pareto_k": maximum_local_pareto,
        "local_relevant_minimum_tail_ess": minimum_local_tail,
        "state_proposal_minimum_in_bounds_fraction": minimum_in_bounds,
        "checks": checks,
        "all_recommended_converged": bool(all(checks.values())),
    }


def _adapted_structural_proposal(
    vectors: np.ndarray,
    log_weight: np.ndarray,
    fallback_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    coordinates = vectors[:, IMPORTANCE_NUISANCE_INDICES]
    return _tempered_importance_adaptation(
        coordinates,
        log_weight,
        fallback_covariance,
        target_ess=max(32.0, 8.0 * coordinates.shape[1]),
        covariance_regularization=0.10,
    )


def _structural_candidate_frame(
    vectors: np.ndarray,
    *,
    index_offset: int = 0,
) -> pd.DataFrame:
    coordinates = np.asarray(vectors[:, IMPORTANCE_NUISANCE_INDICES], dtype=float)
    return pd.DataFrame(
        {
            "nuisance_index": np.arange(len(vectors), dtype=int) + int(index_offset),
            **{
                f"nuisance_z_{position}": coordinates[:, position]
                for position in range(coordinates.shape[1])
            },
        }
    )


def _posterior_rows(
    contexts: list[CountryStateContext],
    structural_vectors: np.ndarray,
    structural_log_prior: np.ndarray,
    stage: StageEvaluation,
    quality: dict[str, Any],
    *,
    draws_per_batch: int,
    batches: int,
    seed: int,
) -> pd.DataFrame:
    if (
        stage.local_state_candidate_by_country is None
        or stage.local_state_log_density_by_country is None
    ):
        raise ValueError("Posterior sampling requires retained conditional state proposals")
    total_draws = int(draws_per_batch) * int(batches)
    rng = np.random.default_rng(int(seed))
    selected_structural = _stratified_resample_indices(
        stage.global_weight, total_draws, rng
    )
    rows: list[dict[str, Any]] = []
    for posterior_draw, structural_index in enumerate(selected_structural, start=1):
        batch = int((posterior_draw - 1) // draws_per_batch + 1)
        draw = int((posterior_draw - 1) % draws_per_batch + 1)
        shared_sample = _sample_from_vector(
            structural_vectors[int(structural_index)], contexts[0].priors
        )
        selected_state: dict[str, tuple[int, np.ndarray, float]] = {}
        joint_log_target = float(structural_log_prior[int(structural_index)])
        for context in contexts:
            raw = stage.local_log_weight_by_country[context.country][int(structural_index)]
            probability = _normalise_log_weights(raw)
            state_index = int(rng.choice(len(probability), p=probability))
            coordinate = stage.local_state_candidate_by_country[context.country][
                int(structural_index), state_index
            ]
            local_log_density = stage.local_state_log_density_by_country[
                context.country
            ][int(structural_index), state_index]
            local_log_target = float(raw[state_index] + local_log_density)
            joint_log_target += local_log_target
            selected_state[context.country] = (
                state_index,
                coordinate,
                float(probability[state_index]),
            )

        for country_position, context in enumerate(contexts):
            state_index, coordinate, local_probability = selected_state[context.country]
            sample = {
                "beta_S": float(np.exp(coordinate[0])),
                "reporting_multiplier": float(np.exp(coordinate[1])),
                **{name: float(shared_sample[name]) for name in SHARED_PARAMETER_NAMES},
                "resistance_prevalence": float(
                    context.runtime_reference["resistance"][
                        "target_prevalence_at_analysis_start"
                    ]
                ),
                "reporting_trend_end_multiplier": 1.0,
            }
            process_values = {
                name: float(value)
                for name, value in zip(context.coordinate_names[2:], coordinate[2:])
            }
            process_rng = np.random.default_rng(
                int(seed)
                + posterior_draw * 1009
                + country_position * 104729
            )
            previous = float(coordinate[-1])
            for year in range(context.historical_end_year + 1, context.forecast_end_year + 1):
                previous = float(
                    context.process_rho * previous
                    + process_rng.normal(0.0, context.process_innovation_sd)
                )
                process_values[f"log_beta_process_{year}"] = previous

            row: dict[str, Any] = {
                "country": context.country,
                "chain": batch,
                "draw": draw,
                "posterior_draw": posterior_draw,
                "structural_draw_id": posterior_draw,
                "source_structural_index": int(structural_index),
                "source_structural_weight": float(
                    stage.global_weight[int(structural_index)]
                ),
                "source_state_index": state_index,
                "source_state_conditional_weight": local_probability,
                "posterior_log_prob": joint_log_target,
                "sampling_method": SAMPLING_METHOD,
                "inference_structure": INFERENCE_STRUCTURE,
                "uncertainty_target": UNCERTAINTY_SCOPE,
                "accepted_fraction": 1.0,
                "forecast_process_rho": context.process_rho,
                "forecast_process_innovation_sd": context.process_innovation_sd,
                "forecast_process_years": max(
                    context.forecast_end_year - context.historical_end_year, 0
                ),
                "historical_process_end_year": context.historical_end_year,
                **sample,
                **process_values,
            }
            for key, value in quality.items():
                if key != "checks" and isinstance(
                    value, (bool, int, float, np.bool_, np.integer, np.floating)
                ):
                    row[key] = value.item() if isinstance(value, np.generic) else value
            rows.append(row)
    return pd.DataFrame(rows)


def _state_proposal_audit(
    contexts: list[CountryStateContext],
    proposals: dict[str, StateProposal],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for context in contexts:
        proposal = proposals[context.country]
        frame = pd.DataFrame(proposal.candidates, columns=context.coordinate_names)
        frame.insert(0, "country", context.country)
        frame.insert(1, "state_candidate_index", np.arange(len(frame), dtype=int))
        frame["proposal_log_density"] = proposal.log_density
        frame["proposal_component"] = proposal.source_component
        frame["proposal_covariance_scale"] = proposal.covariance_scales[
            proposal.source_component
        ]
        frame["in_bounds"] = proposal.in_bounds
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def run_hierarchical_joint_posterior(
    *,
    countries: list[str] | None = None,
    adaptation_draws: int = 256,
    refinement_draws: int = 256,
    final_draws: int = 4096,
    state_adaptation_candidates: int = 256,
    outer_adaptation_state_candidates: int = 512,
    state_candidates: int = 2048,
    state_covariance_scales: tuple[float, ...] = (0.25, 1.0, 4.0, 9.0),
    state_optimizer_maxiter: int = 10,
    state_adaptive_rounds: int = 2,
    state_localized_rounds: int = 2,
    draws_per_batch: int = 512,
    batches: int = 4,
    defensive_prior_fraction: float = 0.20,
    n_jobs: int = 64,
    seed: int = 20260715,
    output_stem: str = DEFAULT_STEM,
    require_recommended: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if (
        adaptation_draws < 32
        or refinement_draws < 32
        or final_draws < 16
        or state_adaptation_candidates < 64
        or outer_adaptation_state_candidates < 64
        or state_candidates < 64
    ):
        raise ValueError(
            "Importance stages require at least 32/32/16/64/64/64 candidates"
        )
    if draws_per_batch < 1 or batches < 1:
        raise ValueError("Posterior output batches and draws must be positive")
    minimum_global_tail_candidates = int(
        np.ceil(RECOMMENDED_MIN_GLOBAL_TAIL_ESS / 0.025)
    )
    if require_recommended and final_draws < minimum_global_tail_candidates:
        raise ValueError(
            "Recommended global tail-ESS support is mathematically impossible: "
            f"final_draws={final_draws}, required>={minimum_global_tail_candidates} "
            f"for {RECOMMENDED_MIN_GLOBAL_TAIL_ESS:g} effective samples in each "
            "2.5% tail."
        )
    configs = load_configs()
    calibrated_countries = publication_country_names(configs)
    resolved_countries = countries or calibrated_countries
    outside = sorted(set(resolved_countries).difference(calibrated_countries))
    if outside:
        raise ValueError(f"Countries outside the prespecified calibrated set: {outside}")
    contexts = [_context(country, configs=configs) for country in resolved_countries]
    _assert_shared_prior_contract(contexts)
    reference = contexts[0]
    checkpoint_root = project_path(
        "outputs", "metadata", f"{output_stem}_checkpoints"
    )

    adaptation_vectors, adaptation_log_prior = _prior_nuisance_vectors(
        n=int(adaptation_draws),
        seed=int(seed) + 2003,
        calibrated_start=reference.structural_start,
        base_config=reference.prior_base,
        priors=reference.priors,
    )
    adaptation_stage = _evaluate_stage(
        contexts,
        adaptation_vectors,
        adaptation_log_prior,
        adaptation_log_prior,
        state_adaptation_candidates=int(state_adaptation_candidates),
        state_candidates=int(
            min(state_candidates, outer_adaptation_state_candidates)
        ),
        state_covariance_scales=tuple(
            float(value) for value in state_covariance_scales
        ),
        optimizer_maxiter=int(state_optimizer_maxiter),
        adaptive_rounds=int(state_adaptive_rounds),
        localized_rounds=int(state_localized_rounds),
        n_jobs=n_jobs,
        seed=int(seed) + 3001,
        retain_state_candidates=False,
        checkpoint_dir=checkpoint_root / "adaptation",
    )
    prior_coordinate_sd = np.std(
        adaptation_vectors[:, IMPORTANCE_NUISANCE_INDICES], axis=0, ddof=1
    )
    fallback_covariance = np.diag(np.maximum(prior_coordinate_sd, 0.02) ** 2)
    (
        _proposal_mean,
        _proposal_covariance,
        adaptation_tempering_exponent,
        adaptation_raw_ess,
    ) = _adapted_structural_proposal(
        adaptation_vectors,
        adaptation_stage.log_global_weight,
        fallback_covariance,
    )

    adaptation_frame = _structural_candidate_frame(adaptation_vectors)
    adaptation_components = _build_nuisance_mixture_components(
        adaptation_frame,
        adaptation_stage.log_global_weight,
        prior_coordinate_sd=prior_coordinate_sd,
    )
    refinement_vectors, refinement_log_prior, refinement_log_proposal = (
        _mixture_nuisance_vectors(
            n=int(refinement_draws),
            seed=int(seed) + 3503,
            calibrated_start=reference.structural_start,
            base_config=reference.prior_base,
            priors=reference.priors,
            gaussian_components=adaptation_components,
            defensive_prior_fraction=float(defensive_prior_fraction),
        )
    )
    refinement_stage = _evaluate_stage(
        contexts,
        refinement_vectors,
        refinement_log_prior,
        refinement_log_proposal,
        state_adaptation_candidates=int(state_adaptation_candidates),
        state_candidates=int(
            min(state_candidates, outer_adaptation_state_candidates)
        ),
        state_covariance_scales=tuple(
            float(value) for value in state_covariance_scales
        ),
        optimizer_maxiter=int(state_optimizer_maxiter),
        adaptive_rounds=int(state_adaptive_rounds),
        localized_rounds=int(state_localized_rounds),
        n_jobs=n_jobs,
        seed=int(seed) + 3757,
        retain_state_candidates=False,
        checkpoint_dir=checkpoint_root / "refinement",
    )
    refinement_frame = _structural_candidate_frame(
        refinement_vectors,
        index_offset=len(adaptation_vectors),
    )
    pooled_frame = pd.concat(
        (adaptation_frame, refinement_frame), ignore_index=True
    )
    pooled_log_weight = np.concatenate(
        (adaptation_stage.log_global_weight, refinement_stage.log_global_weight)
    )
    final_components = _build_nuisance_mixture_components(
        pooled_frame,
        pooled_log_weight,
        prior_coordinate_sd=prior_coordinate_sd,
    )

    final_vectors, final_log_prior, final_log_proposal = _mixture_nuisance_vectors(
        n=int(final_draws),
        seed=int(seed) + 4001,
        calibrated_start=reference.structural_start,
        base_config=reference.prior_base,
        priors=reference.priors,
        gaussian_components=final_components,
        defensive_prior_fraction=float(defensive_prior_fraction),
    )
    final_stage = _evaluate_stage(
        contexts,
        final_vectors,
        final_log_prior,
        final_log_proposal,
        state_adaptation_candidates=int(state_adaptation_candidates),
        state_candidates=int(state_candidates),
        state_covariance_scales=tuple(
            float(value) for value in state_covariance_scales
        ),
        optimizer_maxiter=int(state_optimizer_maxiter),
        adaptive_rounds=int(state_adaptive_rounds),
        localized_rounds=int(state_localized_rounds),
        n_jobs=n_jobs,
        seed=int(seed) + 5003,
        retain_state_candidates=True,
        checkpoint_dir=checkpoint_root / "final",
    )
    quality = _quality(final_vectors, final_stage)
    posterior = _posterior_rows(
        contexts,
        final_vectors,
        final_log_prior,
        final_stage,
        quality,
        draws_per_batch=int(draws_per_batch),
        batches=int(batches),
        seed=int(seed) + 6007,
    )

    structural_samples = [
        _sample_from_vector(vector, reference.priors) for vector in final_vectors
    ]
    structural_audit = pd.DataFrame(
        {
            "structural_index": np.arange(len(final_vectors), dtype=int),
            "joint_log_importance_weight": final_stage.log_global_weight,
            "normalized_weight": final_stage.global_weight,
            **{
                name: [sample[name] for sample in structural_samples]
                for name in SHARED_PARAMETER_NAMES
            },
        }
    )
    local_diagnostics = final_stage.local_diagnostics.copy()
    local_diagnostics["structural_posterior_weight"] = local_diagnostics[
        "structural_index"
    ].map(dict(enumerate(final_stage.global_weight)))

    posterior_path = project_path(
        "outputs", "simulations", f"{output_stem}_posterior_samples.parquet"
    )
    structural_path = project_path(
        "outputs", "diagnostics", f"{output_stem}_structural_importance_audit.csv"
    )
    local_path = project_path(
        "outputs", "diagnostics", f"{output_stem}_local_state_importance_audit.csv"
    )
    write_dataframe(posterior, posterior_path)
    write_dataframe(structural_audit, structural_path)
    write_dataframe(local_diagnostics, local_path)

    total_parameters = len(SHARED_PARAMETER_NAMES) + sum(
        len(context.coordinate_names) for context in contexts
    )
    all_recommended = bool(quality["all_recommended_converged"])
    convergence_summary = {
        "diagnostic_method": "nested_exact_target_importance_support",
        "n_parameters_total": int(total_parameters),
        "n_parameters_converged": int(total_parameters if all_recommended else 0),
        "n_parameters_recommended_converged": int(
            total_parameters if all_recommended else 0
        ),
        "all_converged": all_recommended,
        "all_recommended_converged": all_recommended,
        "worst_rhat": None,
        "rhat_applicable": False,
        "min_bulk_ess": float(
            min(
                quality["global_importance_effective_sample_size"],
                quality["local_relevant_minimum_ess"],
            )
        ),
        "min_tail_ess": float(
            min(
                quality["global_importance_minimum_tail_ess"],
                quality["local_relevant_minimum_tail_ess"],
            )
        ),
        "importance_quality": quality,
        "method_note": (
            "R-hat is not applicable to independent nested importance samples. "
            "Validity is determined by global shared-parameter and posterior-relevant "
            "country-state ESS, maximum weight, Pareto-k, tail support, and proposal "
            "boundary coverage."
        ),
    }
    metadata = current_run_metadata(
        output_stem,
        row_counts={
            "posterior_samples": int(len(posterior)),
            "structural_candidates": int(len(structural_audit)),
            "local_state_diagnostics": int(len(local_diagnostics)),
        },
    ) | {
        "sampler": SAMPLING_METHOD,
        "inference_structure": INFERENCE_STRUCTURE,
        "uncertainty_scope": UNCERTAINTY_SCOPE,
        "analysis_role": ANALYSIS_ROLE,
        "publication_path": PUBLICATION_PATH,
        "figure2c_interval_source": FIGURE2C_INTERVAL_SOURCE,
        "countries": resolved_countries,
        "shared_parameters": list(SHARED_PARAMETER_NAMES),
        "country_specific_parameters": list(LOCAL_PARAMETER_NAMES),
        "prior_registry": (
            "config/parameter_distributions.yaml::bayesian_joint "
            "(shared structural parameters only)"
        ),
        "registry_prior_parameter_names": list(REGISTRY_PRIOR_PARAMETER_NAMES),
        "calibration_state_prior_parameter_names": list(
            CALIBRATION_STATE_PRIOR_PARAMETER_NAMES
        ),
        "calibration_state_prior_source": "accepted_country_calibration_artifacts",
        "adaptation_draws": int(adaptation_draws),
        "refinement_draws": int(refinement_draws),
        "final_importance_draws": int(final_draws),
        "state_candidates_per_country": int(state_candidates),
        "outer_adaptation_state_candidates_per_country": int(
            outer_adaptation_state_candidates
        ),
        "state_adaptation_candidates_per_country": int(
            state_adaptation_candidates
        ),
        "state_covariance_scales": [float(value) for value in state_covariance_scales],
        "state_optimizer_maxiter": int(state_optimizer_maxiter),
        "state_adaptive_rounds": int(state_adaptive_rounds),
        "state_localized_rounds": int(state_localized_rounds),
        "defensive_prior_fraction": float(defensive_prior_fraction),
        "n_chains": int(batches),
        "draws_per_chain": int(draws_per_batch),
        "chain_labels_are_batches": True,
        "seed": int(seed),
        "n_jobs": int(n_jobs),
        "checkpoint_root": str(checkpoint_root),
        "posterior_samples_sha256": file_sha256(posterior_path),
        "structural_importance_audit_sha256": file_sha256(structural_path),
        "local_state_importance_audit_sha256": file_sha256(local_path),
        "adaptation_raw_ess": float(adaptation_raw_ess),
        "adaptation_tempering_exponent": float(adaptation_tempering_exponent),
        "quality": quality,
        "convergence_summary": convergence_summary,
        "posterior_interpretation": (
            "Joint posterior under the prespecified multi-country mechanistic "
            "NB2/AR(1) model. Shared biological parameters, country beta/reporting, "
            "and annual latent transmission states are updated together by all "
            "included surveillance likelihoods."
        ),
        "parameter_source_roles": {
            **{name: "shared_joint_posterior" for name in SHARED_PARAMETER_NAMES},
            "beta_S": "country_joint_posterior",
            "reporting_multiplier": "country_joint_posterior",
            "annual_log_beta_process": "country_joint_posterior",
        },
    }
    write_run_metadata(output_stem, metadata)
    if require_recommended and not all_recommended:
        failed = [name for name, passed in quality["checks"].items() if not passed]
        raise RuntimeError(
            "Joint state-space posterior failed recommended importance checks: "
            + ", ".join(failed)
        )
    return posterior, structural_audit, local_diagnostics, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", type=str, default=None)
    parser.add_argument("--adaptation-draws", type=int, default=256)
    parser.add_argument("--refinement-draws", type=int, default=256)
    parser.add_argument("--final-draws", type=int, default=4096)
    parser.add_argument("--state-adaptation-candidates", type=int, default=256)
    parser.add_argument("--outer-adaptation-state-candidates", type=int, default=512)
    parser.add_argument("--state-candidates", type=int, default=2048)
    parser.add_argument(
        "--state-covariance-scales", type=str, default="0.25,1,4,9"
    )
    parser.add_argument("--state-optimizer-maxiter", type=int, default=10)
    parser.add_argument("--state-adaptive-rounds", type=int, default=2)
    parser.add_argument("--state-localized-rounds", type=int, default=2)
    parser.add_argument("--draws-per-batch", type=int, default=512)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--defensive-prior-fraction", type=float, default=0.20)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-stem", type=str, default=DEFAULT_STEM)
    parser.add_argument("--require-recommended", action="store_true")
    args = parser.parse_args()
    run_hierarchical_joint_posterior(
        countries=args.countries.split(",") if args.countries else None,
        adaptation_draws=args.adaptation_draws,
        refinement_draws=args.refinement_draws,
        final_draws=args.final_draws,
        state_adaptation_candidates=args.state_adaptation_candidates,
        outer_adaptation_state_candidates=args.outer_adaptation_state_candidates,
        state_candidates=args.state_candidates,
        state_covariance_scales=tuple(
            float(value) for value in args.state_covariance_scales.split(",")
        ),
        state_optimizer_maxiter=args.state_optimizer_maxiter,
        state_adaptive_rounds=args.state_adaptive_rounds,
        state_localized_rounds=args.state_localized_rounds,
        draws_per_batch=args.draws_per_batch,
        batches=args.batches,
        defensive_prior_fraction=args.defensive_prior_fraction,
        n_jobs=args.n_jobs,
        seed=args.seed,
        output_stem=args.output_stem,
        require_recommended=args.require_recommended,
    )


if __name__ == "__main__":
    main()
