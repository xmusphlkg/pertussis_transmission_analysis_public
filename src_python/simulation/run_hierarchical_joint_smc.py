"""Optional legacy/research exact-target multi-country tempered SMC.

This runner is not a Figure 2c interval source and is not part of the
publication pipeline. Figure 2c uses the separate estimation-CI bootstrap.

The particle state is the complete joint parameter vector: seven biological
parameters shared by all countries plus each country's beta, reporting rate,
and annual AR(1) log-transmission path.  Adaptive likelihood tempering moves
particles from the normalized joint prior to the full NB2/ODE posterior.
Country-state and shared-parameter Metropolis kernels leave every intermediate
target invariant.  No Laplace or importance approximation enters the final
particle weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed, dump as joblib_dump, load as joblib_load
import numpy as np
import pandas as pd
from scipy import special as scipy_special

from src_python.calibration.mcmc_diagnostics import summarize_convergence
from src_python.simulation.common import (
    current_run_metadata,
    config_fingerprint,
    file_sha256,
    load_configs,
    publication_country_names,
    source_code_fingerprint,
    write_run_metadata,
)
from src_python.simulation.run_bayesian_uncertainty import (
    IMPORTANCE_NUISANCE_INDICES,
    _compute_smc_diagnostics,
    _latin_hypercube,
    _log_nuisance_prior_transformed,
    _nuisance_vector_from_unit_cube,
    _normalise_log_weights,
    _sample_from_vector,
    _state_space_exact_negative_log_target,
    _stratified_resample_indices,
    _vector_from_sample,
)
from src_python.simulation.run_hierarchical_joint_posterior import (
    ANALYSIS_ROLE,
    CALIBRATION_STATE_PRIOR_PARAMETER_NAMES,
    DEFAULT_STEM,
    FIGURE2C_INTERVAL_SOURCE,
    LOCAL_PARAMETER_NAMES,
    PUBLICATION_PATH,
    REGISTRY_PRIOR_PARAMETER_NAMES,
    SHARED_PARAMETER_NAMES,
    CountryStateContext,
    _apply_shared_structure,
    _assert_shared_prior_contract,
    _context,
    _state_gaussian_prior_geometry,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import available_cpus, configure_worker_thread_limits


SAMPLING_METHOD = "multi_country_joint_tempered_smc"
INFERENCE_STRUCTURE = "cross_country_joint_state_space_tempered_smc"
UNCERTAINTY_SCOPE = "multi_country_full_feedback_joint_state_space_posterior"
DIAGNOSTIC_METHOD = "tempered_smc_rank_and_particle_quality"

RECOMMENDED_MIN_ISLAND_PARTICLES = 512
RECOMMENDED_MIN_ISLANDS = 4
RECOMMENDED_MIN_STAGE_ESS_FRACTION = 0.65
RECOMMENDED_MAX_FINAL_WEIGHT = 0.02
RECOMMENDED_MIN_UNIQUE_PARTICLE_FRACTION = 0.25
RECOMMENDED_MIN_MOVE_ACCEPTANCE = 0.10
RECOMMENDED_MAX_MOVE_ACCEPTANCE = 0.60
RECOMMENDED_MAX_BOUNDARY_MASS = 0.01
GUIDED_STATE_COVARIANCE_SCALE = 4.0
GUIDED_BRIDGE_TEMPERATURE_SCALE = 1e-4
SHARED_POPULATION_GUIDE_SWITCH_TEMPERATURE = 1e-3
DIFFERENTIAL_EVOLUTION_GLOBAL_JUMP_PROBABILITY = 0.10
DIFFERENTIAL_EVOLUTION_JITTER_FRACTION = 1e-3
MODE_BANK_DEGREES_OF_FREEDOM = 50.0
MODE_BANK_COVARIANCE_INFLATION = 1.00
MODE_BANK_COVARIANCE_FLOOR_FRACTION = 0.01
MODE_BANK_ACTIVATION_TEMPERATURE = 0.80
MODE_BANK_CROSSFIT_COVARIANCE_INFLATION = 1.00
MODE_BANK_CROSSFIT_COVARIANCE_FLOOR_FRACTION = 0.05


def _sample_standard_deviation(values: np.ndarray) -> float:
    """Return a finite between-replicate SD, including the one-replicate case."""

    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _stage_move_cycle(stage: int, *, mode_bank_active: bool) -> tuple[str, ...]:
    secondary = (
        ("local", "guided", "joint")
        if mode_bank_active
        else ("joint", "local", "guided")
    )
    offset = (int(stage) - 1) % len(secondary)
    rotated = tuple(
        secondary[(offset + index) % len(secondary)]
        for index in range(len(secondary))
    )
    return ("mode_bank",) + rotated if mode_bank_active else rotated


def _mean_move_fractions(
    rounds: int,
    *,
    mode_bank_active: bool,
) -> dict[str, float]:
    stages = range(1, 4)
    counts = {kind: 0 for kind in ("mode_bank", "joint", "local", "guided")}
    total = 0
    for stage in stages:
        cycle = _stage_move_cycle(stage, mode_bank_active=mode_bank_active)
        for index in range(int(rounds)):
            counts[cycle[index % len(cycle)]] += 1
            total += 1
    return {kind: count / total for kind, count in counts.items()}


@dataclass(frozen=True)
class PriorGeometry:
    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray


@dataclass
class ParticleCloud:
    structural: np.ndarray
    state_by_country: dict[str, np.ndarray]
    structural_log_prior: np.ndarray
    state_log_prior_by_country: dict[str, np.ndarray]
    potential_by_country: dict[str, np.ndarray]
    weights: np.ndarray
    island: np.ndarray
    ancestor: np.ndarray


@dataclass(frozen=True)
class JointMoveAcceptance:
    overall: float
    local: float
    global_jump: float


@dataclass(frozen=True)
class ModeBank:
    """Fixed multimodal proposal learned from non-publication pilot particles."""

    means: tuple[np.ndarray, ...]
    covariances: tuple[np.ndarray, ...]
    precisions: tuple[np.ndarray, ...]
    log_determinants: tuple[float, ...]
    cholesky_factors: tuple[np.ndarray, ...]
    degrees_of_freedom: float
    source_stems: tuple[str, ...]
    source_hashes: tuple[str, ...]
    dimension: int


def _cloud_payload(cloud: ParticleCloud) -> dict[str, Any]:
    """Return a class-independent checkpoint representation."""

    return {
        "structural": cloud.structural,
        "state_by_country": cloud.state_by_country,
        "structural_log_prior": cloud.structural_log_prior,
        "state_log_prior_by_country": cloud.state_log_prior_by_country,
        "potential_by_country": cloud.potential_by_country,
        "weights": cloud.weights,
        "island": cloud.island,
        "ancestor": cloud.ancestor,
    }


def _cloud_from_payload(payload: dict[str, Any]) -> ParticleCloud:
    return ParticleCloud(**payload)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    joblib_dump(payload, temporary, compress=1)
    temporary.replace(path)


def _log_gaussian_kernel(
    values: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(values, dtype=float) - np.asarray(mean, dtype=float)
    return -0.5 * np.einsum("ij,jk,ik->i", delta, precision, delta)


def _reflect_into_box(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Reflect a symmetric random walk into a finite rectangular support."""

    x = np.asarray(values, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    if bool((~np.isfinite(width)).any()) or bool((width <= 0.0).any()):
        raise ValueError("SMC state bounds must be finite and ordered")
    folded = np.mod(x - lo, 2.0 * width)
    return lo + np.where(folded <= width, folded, 2.0 * width - folded)


def _island_indices(island: np.ndarray) -> list[np.ndarray]:
    return [
        np.flatnonzero(island == value)
        for value in np.unique(np.asarray(island, dtype=int))
    ]


def _island_weight_diagnostics(
    weights: np.ndarray,
    island: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ess: list[float] = []
    maximum: list[float] = []
    for positions in _island_indices(island):
        local = np.asarray(weights[positions], dtype=float)
        local /= float(local.sum())
        ess.append(float(1.0 / np.sum(local**2)))
        maximum.append(float(np.max(local)))
    return np.asarray(ess), np.asarray(maximum)


def _incremented_weights(
    weights: np.ndarray,
    log_potential: np.ndarray,
    island: np.ndarray,
    temperature_increment: float,
) -> np.ndarray:
    updated = np.zeros(len(weights), dtype=float)
    for positions in _island_indices(island):
        log_weight = (
            np.log(np.clip(weights[positions], 1e-300, 1.0))
            + float(temperature_increment) * log_potential[positions]
        )
        updated[positions] = _normalise_log_weights(log_weight)
    return updated


def _find_next_temperature(
    weights: np.ndarray,
    log_potential: np.ndarray,
    island: np.ndarray,
    current: float,
    target_ess_fraction: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    sizes = np.asarray([len(value) for value in _island_indices(island)], dtype=float)

    def evaluate(candidate: float) -> tuple[np.ndarray, np.ndarray]:
        candidate_weights = _incremented_weights(
            weights, log_potential, island, float(candidate) - float(current)
        )
        ess, _ = _island_weight_diagnostics(candidate_weights, island)
        return candidate_weights, ess

    full_weights, full_ess = evaluate(1.0)
    if bool((full_ess >= target_ess_fraction * sizes).all()):
        return 1.0, full_weights, full_ess

    low = float(current)
    high = 1.0
    best_weights = np.asarray(weights, dtype=float).copy()
    best_ess, _ = _island_weight_diagnostics(best_weights, island)
    for _ in range(48):
        midpoint = 0.5 * (low + high)
        candidate_weights, candidate_ess = evaluate(midpoint)
        if bool((candidate_ess >= target_ess_fraction * sizes).all()):
            low = midpoint
            best_weights = candidate_weights
            best_ess = candidate_ess
        else:
            high = midpoint
    if low <= float(current) + 1e-10:
        low = min(1.0, float(current) + 1e-4)
        best_weights, best_ess = evaluate(low)
    return float(low), best_weights, best_ess


def _state_prior_draws(
    geometry: PriorGeometry,
    context: CountryStateContext,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(count), len(geometry.mean)), dtype=float)
    filled = 0
    attempts = 0
    while filled < int(count):
        batch = max(64, 2 * (int(count) - filled))
        candidate = rng.multivariate_normal(
            geometry.mean,
            geometry.covariance,
            size=batch,
            check_valid="raise",
        )
        valid = np.logical_and(
            np.all(candidate >= context.lower, axis=1),
            np.all(candidate <= context.upper, axis=1),
        )
        accepted = candidate[valid]
        take = min(len(accepted), int(count) - filled)
        if take:
            out[filled : filled + take] = accepted[:take]
            filled += take
        attempts += batch
        if attempts > max(100000, 1000 * int(count)) and filled < int(count):
            raise RuntimeError(
                f"Joint SMC could not sample bounded state prior for {context.country}"
            )
    return out


def _evaluate_country_chunk(
    context: CountryStateContext,
    structural: np.ndarray,
    states: np.ndarray,
    indices: np.ndarray,
    state_prior_mean: np.ndarray,
    state_prior_precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    configure_worker_thread_limits()
    local_indices = np.asarray(indices, dtype=int)
    potential = np.full(len(local_indices), -np.inf, dtype=float)
    state_log_prior = _log_gaussian_kernel(
        states[local_indices], state_prior_mean, state_prior_precision
    )
    for output_position, particle_index in enumerate(local_indices):
        try:
            shared_sample = _sample_from_vector(
                structural[int(particle_index)], context.priors
            )
            prior_base = _apply_shared_structure(context.prior_base, shared_sample)
            # The accepted runtime carries the complete dated process-period
            # skeleton.  The exact target overwrites beta, reporting, and every
            # historical process coordinate below, while the shared draw
            # updates the biological mechanism in both history and projection.
            runtime_base = _apply_shared_structure(
                context.runtime_reference, shared_sample
            )
            negative_log_target = _state_space_exact_negative_log_target(
                states[int(particle_index)],
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
                potential[output_position] = (
                    -float(negative_log_target) - state_log_prior[output_position]
                )
        except Exception:
            potential[output_position] = -np.inf
    return local_indices, potential


def _evaluate_country_potential(
    context: CountryStateContext,
    structural: np.ndarray,
    states: np.ndarray,
    geometry: PriorGeometry,
    *,
    n_jobs: int,
    active: np.ndarray | None = None,
) -> np.ndarray:
    count = len(structural)
    selected = (
        np.arange(count, dtype=int)
        if active is None
        else np.flatnonzero(np.asarray(active, dtype=bool))
    )
    output = np.full(count, -np.inf, dtype=float)
    if len(selected) == 0:
        return output
    workers = min(max(1, int(n_jobs)), available_cpus(), len(selected))
    chunks = [value for value in np.array_split(selected, workers) if len(value)]
    if len(chunks) == 1:
        results = [
            _evaluate_country_chunk(
                context,
                structural,
                states,
                chunks[0],
                geometry.mean,
                geometry.precision,
            )
        ]
    else:
        results = Parallel(
            n_jobs=len(chunks), backend="loky", inner_max_num_threads=1
        )(
            delayed(_evaluate_country_chunk)(
                context,
                structural,
                states,
                chunk,
                geometry.mean,
                geometry.precision,
            )
            for chunk in chunks
        )
    for indices, values in results:
        output[indices] = values
    return output


def _regularized_covariance(
    values: np.ndarray,
    floor: np.ndarray,
    *,
    floor_fraction: float = 0.10,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    floor = np.asarray(floor, dtype=float)
    fraction = float(floor_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Covariance floor fraction must be in [0, 1]")
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    covariance = (1.0 - fraction) * covariance + fraction * floor
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minimum = max(float(np.max(eigenvalues)) * 1e-10, 1e-10)
    covariance = (
        eigenvectors * np.maximum(eigenvalues, minimum)
    ) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def _with_process_support(
    context: CountryStateContext,
    absolute_bound: float | None,
) -> CountryStateContext:
    """Widen latent-process numerical support without changing its Gaussian prior."""

    if absolute_bound is None:
        return context
    bound = float(absolute_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("Process support bound must be positive and finite")
    lower = np.asarray(context.lower, dtype=float).copy()
    upper = np.asarray(context.upper, dtype=float).copy()
    if len(lower) > 2:
        lower[2:] = np.minimum(lower[2:], -bound)
        upper[2:] = np.maximum(upper[2:], bound)
    return replace(context, lower=lower, upper=upper)


def _state_to_unbounded(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map bounded state coordinates to R and return log |dx/dz|."""

    x = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    if bool((width <= 0.0).any()):
        raise ValueError("State support must have strictly positive width")
    probability = (x - lower) / width
    if bool(((probability < -1e-10) | (probability > 1.0 + 1e-10)).any()):
        raise ValueError("State coordinate lies outside the configured support")
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    unbounded = np.log(probability) - np.log1p(-probability)
    log_jacobian = np.sum(
        np.log(width) + np.log(probability) + np.log1p(-probability),
        axis=1,
    )
    return unbounded, log_jacobian


def _state_from_unbounded(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map R coordinates into a bounded state box and return log |dx/dz|."""

    z = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    probability = scipy_special.expit(z)
    state = lower + width * probability
    log_jacobian = np.sum(
        np.log(width)
        - np.logaddexp(0.0, -z)
        - np.logaddexp(0.0, z),
        axis=1,
    )
    return state, log_jacobian


def _cloud_mode_coordinates(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete joint coordinates in the mode-bank parameterization."""

    blocks = [cloud.structural[:, IMPORTANCE_NUISANCE_INDICES]]
    log_jacobian = np.zeros(len(cloud.structural), dtype=float)
    for context in contexts:
        transformed, contribution = _state_to_unbounded(
            cloud.state_by_country[context.country],
            context.lower,
            context.upper,
        )
        blocks.append(transformed)
        log_jacobian += contribution
    return np.column_stack(blocks), log_jacobian


def _pilot_mode_coordinates(
    posterior: pd.DataFrame,
    contexts: list[CountryStateContext],
) -> np.ndarray:
    """Reconstruct paired internal joint coordinates from a pilot posterior."""

    required = {
        "country",
        "structural_draw_id",
        "beta_S",
        "reporting_multiplier",
        *SHARED_PARAMETER_NAMES,
    }
    missing = required.difference(posterior.columns)
    if missing:
        raise ValueError(
            "Mode-bank pilot posterior is missing columns: "
            + ", ".join(sorted(missing))
        )
    draw_ids = np.sort(posterior["structural_draw_id"].dropna().unique())
    if len(draw_ids) < 16:
        raise ValueError("Each mode-bank pilot requires at least 16 paired draws")
    reference = contexts[0]
    shared = (
        posterior.sort_values(["structural_draw_id", "country"])
        .groupby("structural_draw_id", sort=True)
        .first()
        .reindex(draw_ids)
    )
    structural = np.vstack(
        [
            _vector_from_sample(record, reference.priors)[
                IMPORTANCE_NUISANCE_INDICES
            ]
            for record in shared.to_dict("records")
        ]
    )
    blocks = [structural]
    for context in contexts:
        local = posterior.loc[posterior["country"].eq(context.country)].set_index(
            "structural_draw_id"
        )
        if local.index.has_duplicates:
            raise ValueError(
                f"Mode-bank pilot contains duplicate paired rows for {context.country}"
            )
        local = local.reindex(draw_ids)
        if local.isna().all(axis=1).any():
            raise ValueError(
                f"Mode-bank pilot does not contain one paired row per draw for {context.country}"
            )
        process_columns = list(context.coordinate_names[2:])
        missing_process = set(process_columns).difference(local.columns)
        if missing_process:
            raise ValueError(
                f"Mode-bank pilot is missing {context.country} process columns: "
                + ", ".join(sorted(missing_process))
            )
        state = np.column_stack(
            (
                np.log(pd.to_numeric(local["beta_S"]).to_numpy(dtype=float)),
                np.log(
                    pd.to_numeric(local["reporting_multiplier"]).to_numpy(
                        dtype=float
                    )
                ),
                *[
                    pd.to_numeric(local[column]).to_numpy(dtype=float)
                    for column in process_columns
                ],
            )
        )
        transformed, _ = _state_to_unbounded(
            state,
            context.lower,
            context.upper,
        )
        blocks.append(transformed)
    return np.column_stack(blocks)


def _cloud_from_completed_posterior(
    posterior: pd.DataFrame,
    contexts: list[CountryStateContext],
    *,
    maximum_particles: int | None = None,
) -> ParticleCloud:
    """Reconstruct an equal-mass cloud for exact proposal-kernel audits.

    Completed posterior files store the full unnormalised joint log target in
    ``posterior_log_prob``.  Keeping that value in the structural cache and
    zeroing the country caches reproduces the same current target without
    rerunning the ODE.  Any accepted proposal replaces all caches with their
    ordinary decomposed values, so the resulting cloud remains valid.
    """

    required = {
        "country",
        "structural_draw_id",
        "posterior_log_prob",
        "beta_S",
        "reporting_multiplier",
        *SHARED_PARAMETER_NAMES,
    }
    missing = required.difference(posterior.columns)
    if missing:
        raise ValueError(
            "Completed posterior is missing columns: "
            + ", ".join(sorted(missing))
        )
    country_ids = [
        set(
            posterior.loc[
                posterior["country"].eq(context.country), "structural_draw_id"
            ].dropna()
        )
        for context in contexts
    ]
    draw_ids = np.asarray(sorted(set.intersection(*country_ids)))
    if maximum_particles is not None and len(draw_ids) > int(maximum_particles):
        positions = np.linspace(
            0,
            len(draw_ids) - 1,
            int(maximum_particles),
            dtype=int,
        )
        draw_ids = draw_ids[positions]
    if len(draw_ids) < 4:
        raise ValueError("A reconstructed proposal audit requires at least 4 draws")

    reference = contexts[0]
    shared = (
        posterior.loc[posterior["structural_draw_id"].isin(draw_ids)]
        .sort_values(["structural_draw_id", "country"])
        .groupby("structural_draw_id", sort=True)
        .first()
        .reindex(draw_ids)
    )
    structural = np.vstack(
        [
            _vector_from_sample(record, reference.priors)
            for record in shared.to_dict("records")
        ]
    )
    structural_log_prior = pd.to_numeric(
        shared["posterior_log_prob"]
    ).to_numpy(dtype=float).copy()
    if not np.isfinite(structural_log_prior).all():
        raise ValueError("Completed posterior contains non-finite joint targets")

    state_by_country: dict[str, np.ndarray] = {}
    state_log_prior_by_country: dict[str, np.ndarray] = {}
    potential_by_country: dict[str, np.ndarray] = {}
    for context in contexts:
        local = posterior.loc[
            posterior["country"].eq(context.country)
            & posterior["structural_draw_id"].isin(draw_ids)
        ].set_index("structural_draw_id")
        if local.index.has_duplicates:
            raise ValueError(
                f"Completed posterior has duplicate paired rows for {context.country}"
            )
        local = local.reindex(draw_ids)
        process_columns = list(context.coordinate_names[2:])
        missing_process = set(process_columns).difference(local.columns)
        if missing_process or local.isna().all(axis=1).any():
            raise ValueError(
                f"Completed posterior has incomplete paired states for {context.country}"
            )
        local_target = pd.to_numeric(local["posterior_log_prob"]).to_numpy(
            dtype=float
        )
        if not np.allclose(
            local_target,
            structural_log_prior,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError(
                f"Joint target is inconsistent across rows for {context.country}"
            )
        state_by_country[context.country] = np.column_stack(
            (
                np.log(pd.to_numeric(local["beta_S"]).to_numpy(dtype=float)),
                np.log(
                    pd.to_numeric(local["reporting_multiplier"]).to_numpy(
                        dtype=float
                    )
                ),
                *[
                    pd.to_numeric(local[column]).to_numpy(dtype=float)
                    for column in process_columns
                ],
            )
        )
        state_log_prior_by_country[context.country] = np.zeros(len(draw_ids))
        potential_by_country[context.country] = np.zeros(len(draw_ids))
    count = len(draw_ids)
    return ParticleCloud(
        structural=structural,
        state_by_country=state_by_country,
        structural_log_prior=structural_log_prior,
        state_log_prior_by_country=state_log_prior_by_country,
        potential_by_country=potential_by_country,
        weights=np.full(count, 1.0 / count),
        island=np.ones(count, dtype=int),
        ancestor=np.arange(count, dtype=int),
    )


def _load_mode_bank(
    stems: list[str],
    contexts: list[CountryStateContext],
    *,
    covariance_inflation: float = MODE_BANK_COVARIANCE_INFLATION,
    covariance_floor_fraction: float = MODE_BANK_COVARIANCE_FLOOR_FRACTION,
    degrees_of_freedom: float = MODE_BANK_DEGREES_OF_FREEDOM,
) -> ModeBank:
    """Fit a fixed heavy-tailed mixture proposal from pilot SMC modes."""

    if not stems:
        raise ValueError("Mode-bank construction requires at least one pilot stem")
    inflation = float(covariance_inflation)
    floor_fraction = float(covariance_floor_fraction)
    df = float(degrees_of_freedom)
    if not np.isfinite(inflation) or inflation <= 0.0:
        raise ValueError("Mode-bank covariance inflation must be positive")
    if not 0.0 <= floor_fraction <= 1.0:
        raise ValueError("Mode-bank covariance floor fraction must be in [0, 1]")
    if not np.isfinite(df) or df <= 2.0:
        raise ValueError("Mode-bank Student-t degrees of freedom must exceed 2")
    values_by_mode: list[np.ndarray] = []
    source_hashes: list[str] = []
    for stem in stems:
        posterior_path = project_path(
            "outputs", "simulations", f"{stem}_posterior_samples.parquet"
        )
        metadata_path = project_path(
            "outputs", "metadata", f"{stem}_run_metadata.json"
        )
        if not posterior_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Incomplete mode-bank pilot stem {stem!r}: "
                f"{posterior_path}, {metadata_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("sampler") != SAMPLING_METHOD
            or metadata.get("inference_structure") != INFERENCE_STRUCTURE
            or metadata.get("uncertainty_scope") != UNCERTAINTY_SCOPE
        ):
            raise ValueError(
                f"Mode-bank pilot {stem!r} does not target the joint SMC posterior"
            )
        posterior = pd.read_parquet(posterior_path)
        values_by_mode.append(_pilot_mode_coordinates(posterior, contexts))
        source_hashes.append(file_sha256(posterior_path))
    dimension = int(values_by_mode[0].shape[1])
    if any(values.shape[1] != dimension for values in values_by_mode):
        raise ValueError("Mode-bank pilots have inconsistent joint dimensions")
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    precisions: list[np.ndarray] = []
    log_determinants: list[float] = []
    cholesky_factors: list[np.ndarray] = []
    for values in values_by_mode:
        empirical = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        diagonal = np.diag(np.maximum(np.diag(empirical), 1e-4))
        covariance = inflation * _regularized_covariance(
            values,
            diagonal,
            floor_fraction=floor_fraction,
        )
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(log_determinant):
            raise RuntimeError("Mode-bank covariance is not positive definite")
        means.append(np.mean(values, axis=0))
        covariances.append(covariance)
        precisions.append(np.linalg.inv(covariance))
        log_determinants.append(float(log_determinant))
        cholesky_factors.append(np.linalg.cholesky(covariance))
    return ModeBank(
        means=tuple(means),
        covariances=tuple(covariances),
        precisions=tuple(precisions),
        log_determinants=tuple(log_determinants),
        cholesky_factors=tuple(cholesky_factors),
        degrees_of_freedom=df,
        source_stems=tuple(stems),
        source_hashes=tuple(source_hashes),
        dimension=dimension,
    )


def _mode_bank_log_density(values: np.ndarray, mode_bank: ModeBank) -> np.ndarray:
    """Score an equal-weight multivariate Student-t mixture."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != mode_bank.dimension:
        raise ValueError("Mode-bank coordinate dimension mismatch")
    dimension = mode_bank.dimension
    df = mode_bank.degrees_of_freedom
    constant = float(
        scipy_special.gammaln((df + dimension) / 2.0)
        - scipy_special.gammaln(df / 2.0)
        - 0.5 * dimension * np.log(df * np.pi)
    )
    components: list[np.ndarray] = []
    for mean, precision, log_determinant in zip(
        mode_bank.means,
        mode_bank.precisions,
        mode_bank.log_determinants,
    ):
        delta = x - mean
        quadratic = np.einsum("ij,jk,ik->i", delta, precision, delta)
        components.append(
            constant
            - 0.5 * log_determinant
            - 0.5 * (df + dimension) * np.log1p(quadratic / df)
        )
    return scipy_special.logsumexp(
        np.column_stack(components), axis=1
    ) - np.log(len(components))


def _draw_mode_bank(
    mode_bank: ModeBank,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw complete joint proposals from the fixed Student-t mixture."""

    source = rng.integers(0, len(mode_bank.means), size=int(count))
    output = np.empty((int(count), mode_bank.dimension), dtype=float)
    df = mode_bank.degrees_of_freedom
    for component, (mean, cholesky) in enumerate(
        zip(mode_bank.means, mode_bank.cholesky_factors)
    ):
        positions = np.flatnonzero(source == component)
        if not len(positions):
            continue
        normal = rng.standard_normal((len(positions), mode_bank.dimension)) @ cholesky.T
        radial = np.sqrt(rng.chisquare(df, size=len(positions)) / df)
        output[positions] = mean + normal / radial[:, None]
    return output


def _joint_mode_bank_independence_move(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    mode_bank: ModeBank,
    *,
    temperature: float,
    n_jobs: int,
    rng: np.random.Generator,
    active: np.ndarray | None = None,
) -> float:
    """Make an exact complete-state independence-MH jump across pilot modes."""

    total = len(cloud.structural)
    target_mask = (
        np.ones(total, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool).copy()
    )
    if target_mask.shape != (total,) or not target_mask.any():
        raise ValueError("Mode-bank active mask must select at least one particle")
    targets = np.flatnonzero(target_mask)
    current_coordinates, current_log_jacobian = _cloud_mode_coordinates(
        cloud, contexts
    )
    proposal_coordinates = current_coordinates.copy()
    proposal_coordinates[targets] = _draw_mode_bank(
        mode_bank, len(targets), rng
    )
    structural_proposal = cloud.structural.copy()
    structural_width = len(IMPORTANCE_NUISANCE_INDICES)
    structural_proposal[
        targets[:, None], IMPORTANCE_NUISANCE_INDICES
    ] = proposal_coordinates[
        targets, :structural_width
    ]
    proposed_state: dict[str, np.ndarray] = {}
    proposed_log_jacobian = current_log_jacobian.copy()
    target_log_jacobian = np.zeros(len(targets), dtype=float)
    offset = structural_width
    for context in contexts:
        width = len(context.coordinate_names)
        state, contribution = _state_from_unbounded(
            proposal_coordinates[targets, offset : offset + width],
            context.lower,
            context.upper,
        )
        proposed_state[context.country] = cloud.state_by_country[
            context.country
        ].copy()
        proposed_state[context.country][targets] = state
        target_log_jacobian += contribution
        offset += width
    proposed_log_jacobian[targets] = target_log_jacobian
    if offset != mode_bank.dimension:
        raise RuntimeError("Mode-bank proposal did not consume every coordinate")

    reference = contexts[0]
    proposed_structural_prior = np.asarray(
        [
            _log_nuisance_prior_transformed(
                vector, reference.prior_base, reference.priors
            )
            for vector in structural_proposal
        ],
        dtype=float,
    )
    valid = target_mask & np.isfinite(proposed_structural_prior)
    proposed_state_prior: dict[str, np.ndarray] = {}
    proposed_potential: dict[str, np.ndarray] = {}
    for context in contexts:
        country = context.country
        geometry = geometries[country]
        proposed_state_prior[country] = _log_gaussian_kernel(
            proposed_state[country], geometry.mean, geometry.precision
        )
        valid &= np.isfinite(proposed_state_prior[country])
        proposed_potential[country] = _evaluate_country_potential(
            context,
            structural_proposal,
            proposed_state[country],
            geometry,
            n_jobs=n_jobs,
            active=valid,
        )
        valid &= np.isfinite(proposed_potential[country])

    current_target = cloud.structural_log_prior.copy()
    proposed_target = proposed_structural_prior.copy()
    for context in contexts:
        country = context.country
        current_target += (
            cloud.state_log_prior_by_country[country]
            + float(temperature) * cloud.potential_by_country[country]
        )
        proposed_target += (
            proposed_state_prior[country]
            + float(temperature) * proposed_potential[country]
        )
    current_log_q = _mode_bank_log_density(
        current_coordinates[targets], mode_bank
    )
    proposed_log_q = _mode_bank_log_density(
        proposal_coordinates[targets], mode_bank
    )
    log_alpha = np.full(total, -np.inf, dtype=float)
    log_alpha[targets] = (
        proposed_target[targets]
        + proposed_log_jacobian[targets]
        - current_target[targets]
        - current_log_jacobian[targets]
        + current_log_q
        - proposed_log_q
    )
    accept = valid & (np.log(rng.random(total)) < log_alpha)
    if accept.any():
        cloud.structural[accept] = structural_proposal[accept]
        cloud.structural_log_prior[accept] = proposed_structural_prior[accept]
        for context in contexts:
            country = context.country
            cloud.state_by_country[country][accept] = proposed_state[country][accept]
            cloud.state_log_prior_by_country[country][accept] = proposed_state_prior[
                country
            ][accept]
            cloud.potential_by_country[country][accept] = proposed_potential[country][
                accept
            ]
    return float(np.mean(accept[targets]))


def _mode_bank_with_empirical_component(
    mode_bank: ModeBank,
    values: np.ndarray,
    *,
    covariance_inflation: float = MODE_BANK_CROSSFIT_COVARIANCE_INFLATION,
    covariance_floor_fraction: float = (
        MODE_BANK_CROSSFIT_COVARIANCE_FLOOR_FRACTION
    ),
) -> ModeBank:
    """Append one current-population component to a fixed historical bank."""

    coordinates = np.asarray(values, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != mode_bank.dimension
        or len(coordinates) < 4
    ):
        raise ValueError(
            "Adaptive mode component requires at least four joint coordinates"
        )
    empirical = np.atleast_2d(np.cov(coordinates, rowvar=False, ddof=1))
    diagonal = np.diag(np.maximum(np.diag(empirical), 1e-4))
    covariance = float(covariance_inflation) * _regularized_covariance(
        coordinates,
        diagonal,
        floor_fraction=float(covariance_floor_fraction),
    )
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise RuntimeError("Adaptive mode covariance is not positive definite")
    return ModeBank(
        means=mode_bank.means + (np.mean(coordinates, axis=0),),
        covariances=mode_bank.covariances + (covariance,),
        precisions=mode_bank.precisions + (np.linalg.inv(covariance),),
        log_determinants=mode_bank.log_determinants
        + (float(log_determinant),),
        cholesky_factors=mode_bank.cholesky_factors
        + (np.linalg.cholesky(covariance),),
        degrees_of_freedom=mode_bank.degrees_of_freedom,
        source_stems=mode_bank.source_stems
        + ("adaptive_crossfit_population",),
        source_hashes=mode_bank.source_hashes + ("conditional_current_cloud",),
        dimension=mode_bank.dimension,
    )


def _joint_crossfit_mode_bank_move(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    mode_bank: ModeBank,
    *,
    temperature: float,
    n_jobs: int,
    rng: np.random.Generator,
) -> float:
    """Use complementary particles to make an exact adaptive mode-bank move.

    Each target half is updated with a proposal fitted only to the fixed donor
    half plus the immutable historical bank.  Conditional on the donor half,
    the proposal density is fixed and the ordinary independence-MH ratio is
    exact.  Reversing the roles composes two invariant block kernels while
    ensuring the proposal covers both the current and previously found modes.
    """

    total = len(cloud.structural)
    if total < 8:
        raise ValueError("Cross-fitted mode-bank moves require at least 8 particles")
    shuffled = rng.permutation(total)
    split = total // 2
    halves = (shuffled[:split], shuffled[split:])
    accepted_weighted = 0.0
    proposed = 0
    for half_index in range(2):
        targets = halves[half_index]
        donors = halves[1 - half_index]
        coordinates, _ = _cloud_mode_coordinates(cloud, contexts)
        adaptive_bank = _mode_bank_with_empirical_component(
            mode_bank,
            coordinates[donors],
        )
        active = np.zeros(total, dtype=bool)
        active[targets] = True
        acceptance = _joint_mode_bank_independence_move(
            cloud,
            contexts,
            geometries,
            adaptive_bank,
            temperature=temperature,
            n_jobs=n_jobs,
            rng=rng,
            active=active,
        )
        accepted_weighted += acceptance * len(targets)
        proposed += len(targets)
    return float(accepted_weighted / proposed)


def _joint_mode_bank_move(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    mode_bank: ModeBank,
    *,
    temperature: float,
    n_jobs: int,
    rng: np.random.Generator,
    crossfit_current_population: bool,
) -> float:
    """Dispatch an exact fixed or complementary-half mode-bank kernel.

    A mode bank learned from an independent pilot generation can be kept fully
    fixed.  This avoids diluting a deliberately weighted historical mixture
    with a current-population component while retaining the ordinary exact
    independence-MH correction.  Cross-fitting remains available for runs
    that benefit from an additional adaptive local component.
    """

    if crossfit_current_population:
        return _joint_crossfit_mode_bank_move(
            cloud,
            contexts,
            geometries,
            mode_bank,
            temperature=temperature,
            n_jobs=n_jobs,
            rng=rng,
        )
    return _joint_mode_bank_independence_move(
        cloud,
        contexts,
        geometries,
        mode_bank,
        temperature=temperature,
        n_jobs=n_jobs,
        rng=rng,
    )


def _resample_islands(
    cloud: ParticleCloud,
    *,
    rng: np.random.Generator,
    force: bool,
    threshold_fraction: float = 0.95,
) -> np.ndarray:
    resampled = np.zeros(len(cloud.weights), dtype=bool)
    for positions in _island_indices(cloud.island):
        local_weight = cloud.weights[positions]
        local_weight /= float(local_weight.sum())
        ess = 1.0 / float(np.sum(local_weight**2))
        if not force and ess >= float(threshold_fraction) * len(positions):
            cloud.weights[positions] = local_weight
            continue
        selected_local = _stratified_resample_indices(local_weight, len(positions), rng)
        selected = positions[selected_local]
        cloud.structural[positions] = cloud.structural[selected]
        cloud.structural_log_prior[positions] = cloud.structural_log_prior[selected]
        cloud.ancestor[positions] = cloud.ancestor[selected]
        for country in cloud.state_by_country:
            cloud.state_by_country[country][positions] = cloud.state_by_country[country][selected]
            cloud.state_log_prior_by_country[country][positions] = (
                cloud.state_log_prior_by_country[country][selected]
            )
            cloud.potential_by_country[country][positions] = (
                cloud.potential_by_country[country][selected]
            )
        cloud.weights[positions] = 1.0 / len(positions)
        resampled[positions] = True
    return resampled


def _shared_move(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    *,
    temperature: float,
    scale: float,
    global_covariance_fraction: float,
    n_jobs: int,
    rng: np.random.Generator,
) -> float:
    proposals = cloud.structural.copy()
    all_active = cloud.structural[:, IMPORTANCE_NUISANCE_INDICES]
    global_floor_sd = np.maximum(np.std(all_active, axis=0, ddof=1), 0.05)
    global_covariance = _regularized_covariance(
        all_active,
        np.diag(np.square(0.10 * global_floor_sd)),
    )
    for positions in _island_indices(cloud.island):
        active_coordinates = cloud.structural[
            positions[:, None], IMPORTANCE_NUISANCE_INDICES
        ]
        covariance = _regularized_covariance(
            active_coordinates,
            global_covariance,
            floor_fraction=float(global_covariance_fraction),
        )
        covariance *= float(scale) ** 2 * (
            2.38**2 / len(IMPORTANCE_NUISANCE_INDICES)
        )
        proposals[
            positions[:, None], IMPORTANCE_NUISANCE_INDICES
        ] += rng.multivariate_normal(
            np.zeros(len(IMPORTANCE_NUISANCE_INDICES)),
            covariance,
            size=len(positions),
            check_valid="raise",
        )
    reference = contexts[0]
    proposed_prior = np.asarray(
        [
            _log_nuisance_prior_transformed(
                vector, reference.prior_base, reference.priors
            )
            for vector in proposals
        ],
        dtype=float,
    )
    valid = np.isfinite(proposed_prior)
    proposed_potential: dict[str, np.ndarray] = {}
    for context in contexts:
        proposed_potential[context.country] = _evaluate_country_potential(
            context,
            proposals,
            cloud.state_by_country[context.country],
            geometries[context.country],
            n_jobs=n_jobs,
            active=valid,
        )
        valid &= np.isfinite(proposed_potential[context.country])
    current_total = np.sum(
        np.column_stack(list(cloud.potential_by_country.values())), axis=1
    )
    proposed_total = np.sum(
        np.column_stack(list(proposed_potential.values())), axis=1
    )
    log_alpha = (
        proposed_prior
        - cloud.structural_log_prior
        + float(temperature) * (proposed_total - current_total)
    )
    accept = valid & (np.log(rng.random(len(log_alpha))) < log_alpha)
    if accept.any():
        cloud.structural[accept] = proposals[accept]
        cloud.structural_log_prior[accept] = proposed_prior[accept]
        for country in proposed_potential:
            cloud.potential_by_country[country][accept] = proposed_potential[country][accept]
    return float(np.mean(accept))


def _country_state_move(
    cloud: ParticleCloud,
    context: CountryStateContext,
    geometry: PriorGeometry,
    *,
    temperature: float,
    scale: float,
    global_covariance_fraction: float,
    n_jobs: int,
    rng: np.random.Generator,
) -> float:
    country = context.country
    current = cloud.state_by_country[country]
    proposal = current.copy()
    global_covariance = _regularized_covariance(
        current,
        geometry.covariance,
    )
    for positions in _island_indices(cloud.island):
        covariance = _regularized_covariance(
            current[positions],
            global_covariance,
            floor_fraction=float(global_covariance_fraction),
        )
        covariance *= float(scale) ** 2 * (2.38**2 / current.shape[1])
        proposal[positions] += rng.multivariate_normal(
            np.zeros(current.shape[1]),
            covariance,
            size=len(positions),
            check_valid="raise",
        )
    proposal = _reflect_into_box(proposal, context.lower, context.upper)
    proposed_log_prior = _log_gaussian_kernel(
        proposal, geometry.mean, geometry.precision
    )
    proposed_potential = _evaluate_country_potential(
        context,
        cloud.structural,
        proposal,
        geometry,
        n_jobs=n_jobs,
    )
    valid = np.isfinite(proposed_potential)
    log_alpha = (
        proposed_log_prior
        - cloud.state_log_prior_by_country[country]
        + float(temperature)
        * (proposed_potential - cloud.potential_by_country[country])
    )
    accept = valid & (np.log(rng.random(len(log_alpha))) < log_alpha)
    if accept.any():
        cloud.state_by_country[country][accept] = proposal[accept]
        cloud.state_log_prior_by_country[country][accept] = proposed_log_prior[accept]
        cloud.potential_by_country[country][accept] = proposed_potential[accept]
    return float(np.mean(accept))


def _shared_prior_independence_move(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    *,
    temperature: float,
    n_jobs: int,
    rng: np.random.Generator,
) -> float:
    """Propose a full shared block from its exact prior and MH-correct it."""

    reference = contexts[0]
    proposals = np.vstack(
        [
            _nuisance_vector_from_unit_cube(
                rng.random(len(IMPORTANCE_NUISANCE_INDICES)),
                reference.structural_start,
                reference.prior_base,
                reference.priors,
            )
            for _ in range(len(cloud.structural))
        ]
    )
    proposed_prior = np.asarray(
        [
            _log_nuisance_prior_transformed(
                vector, reference.prior_base, reference.priors
            )
            for vector in proposals
        ],
        dtype=float,
    )
    valid = np.isfinite(proposed_prior)
    proposed_potential: dict[str, np.ndarray] = {}
    for context in contexts:
        proposed_potential[context.country] = _evaluate_country_potential(
            context,
            proposals,
            cloud.state_by_country[context.country],
            geometries[context.country],
            n_jobs=n_jobs,
            active=valid,
        )
        valid &= np.isfinite(proposed_potential[context.country])
    current_total = np.sum(
        np.column_stack(list(cloud.potential_by_country.values())), axis=1
    )
    proposed_total = np.sum(
        np.column_stack(list(proposed_potential.values())), axis=1
    )
    # q is the normalized shared prior, so both the target-prior and proposal
    # density ratios cancel exactly.
    log_alpha = float(temperature) * (proposed_total - current_total)
    accept = valid & (np.log(rng.random(len(log_alpha))) < log_alpha)
    if accept.any():
        cloud.structural[accept] = proposals[accept]
        cloud.structural_log_prior[accept] = proposed_prior[accept]
        for country in proposed_potential:
            cloud.potential_by_country[country][accept] = proposed_potential[country][
                accept
            ]
    return float(np.mean(accept))


def _log_gaussian_mixture_density(
    values: np.ndarray,
    means: list[np.ndarray],
    covariances: list[np.ndarray],
) -> np.ndarray:
    """Score an equal-weight normalized Gaussian mixture."""

    x = np.asarray(values, dtype=float)
    dimension = x.shape[1]
    components: list[np.ndarray] = []
    for mean, covariance in zip(means, covariances):
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(log_determinant):
            raise RuntimeError("Population guide covariance is not positive definite")
        precision = np.linalg.inv(covariance)
        components.append(
            _log_gaussian_kernel(x, mean, precision)
            - 0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant)
        )
    return scipy_special.logsumexp(
        np.column_stack(components), axis=1
    ) - np.log(len(components))


def _shared_population_independence_move(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    *,
    temperature: float,
    n_jobs: int,
    rng: np.random.Generator,
) -> float:
    """Independence-MH shared move from a cross-island Gaussian mixture."""

    active = cloud.structural[:, IMPORTANCE_NUISANCE_INDICES]
    global_sd = np.maximum(np.std(active, axis=0, ddof=1), 0.05)
    global_covariance = _regularized_covariance(
        active,
        np.diag(np.square(0.10 * global_sd)),
    )
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for positions in _island_indices(cloud.island):
        local = active[positions]
        means.append(np.mean(local, axis=0))
        covariances.append(
            2.0
            * _regularized_covariance(
                local,
                global_covariance,
                floor_fraction=0.25,
            )
        )
    source = rng.integers(0, len(means), size=len(active))
    proposed_active = np.empty_like(active)
    for component in range(len(means)):
        positions = np.flatnonzero(source == component)
        if len(positions):
            proposed_active[positions] = rng.multivariate_normal(
                means[component],
                covariances[component],
                size=len(positions),
                check_valid="raise",
            )
    proposals = cloud.structural.copy()
    proposals[:, IMPORTANCE_NUISANCE_INDICES] = proposed_active
    reference = contexts[0]
    proposed_prior = np.asarray(
        [
            _log_nuisance_prior_transformed(
                vector, reference.prior_base, reference.priors
            )
            for vector in proposals
        ],
        dtype=float,
    )
    valid = np.isfinite(proposed_prior)
    proposed_potential: dict[str, np.ndarray] = {}
    for context in contexts:
        proposed_potential[context.country] = _evaluate_country_potential(
            context,
            proposals,
            cloud.state_by_country[context.country],
            geometries[context.country],
            n_jobs=n_jobs,
            active=valid,
        )
        valid &= np.isfinite(proposed_potential[context.country])
    current_total = np.sum(
        np.column_stack(list(cloud.potential_by_country.values())), axis=1
    )
    proposed_total = np.sum(
        np.column_stack(list(proposed_potential.values())), axis=1
    )
    current_log_q = _log_gaussian_mixture_density(active, means, covariances)
    proposed_log_q = _log_gaussian_mixture_density(
        proposed_active, means, covariances
    )
    log_alpha = (
        proposed_prior
        - cloud.structural_log_prior
        + float(temperature) * (proposed_total - current_total)
        + current_log_q
        - proposed_log_q
    )
    accept = valid & (np.log(rng.random(len(log_alpha))) < log_alpha)
    if accept.any():
        cloud.structural[accept] = proposals[accept]
        cloud.structural_log_prior[accept] = proposed_prior[accept]
        for country in proposed_potential:
            cloud.potential_by_country[country][accept] = proposed_potential[country][
                accept
            ]
    return float(np.mean(accept))


def _joint_differential_evolution_move(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    *,
    temperature: float,
    scale: float,
    n_jobs: int,
    rng: np.random.Generator,
) -> JointMoveAcceptance:
    """Move shared and all country-state coordinates in one exact MH block.

    The complete cross-island population is randomly split in two.  One half is
    updated using ordered donor pairs from the fixed complementary half, then
    the roles are reversed.  Conditional on the donor half, the
    differential-evolution increment and Gaussian jitter are symmetric, so the
    ordinary tempered-target MH ratio is exact.  Cross-island donors let an
    island recover a posterior mode retained by another island, while each
    island still resamples and estimates evidence separately.  Updating all
    country states with the shared block lets the kernel travel along the
    strong cross-country posterior ridge that separate shared and state moves
    cannot traverse.
    """

    total = len(cloud.structural)
    active_indices = np.asarray(IMPORTANCE_NUISANCE_INDICES, dtype=int)
    joint_dimension = len(active_indices) + sum(
        cloud.state_by_country[context.country].shape[1] for context in contexts
    )
    base_gamma = float(scale) * float(2.38 / np.sqrt(2.0 * joint_dimension))
    accepted = 0
    proposed = 0
    accepted_local = 0
    proposed_local = 0
    accepted_global = 0
    proposed_global = 0
    reference = contexts[0]

    population_positions = np.arange(total, dtype=int)
    if len(population_positions) < 4:
        raise ValueError(
            "Joint differential-evolution moves require at least 4 particles"
        )
    shuffled = rng.permutation(population_positions)
    split = len(shuffled) // 2
    halves = (shuffled[:split], shuffled[split:])
    for half_index in range(2):
            targets = halves[half_index]
            donors = halves[1 - half_index]
            if len(targets) == 0 or len(donors) < 2:
                continue

            structural_proposal = cloud.structural.copy()
            state_proposal = {
                context.country: cloud.state_by_country[context.country].copy()
                for context in contexts
            }
            structural_donor_values = cloud.structural[
                donors[:, None], active_indices
            ]
            structural_jitter_sd = DIFFERENTIAL_EVOLUTION_JITTER_FRACTION * np.maximum(
                np.std(structural_donor_values, axis=0, ddof=1), 0.05
            )
            state_jitter_sd = {
                context.country: DIFFERENTIAL_EVOLUTION_JITTER_FRACTION
                * np.maximum(
                    np.std(
                        cloud.state_by_country[context.country][donors],
                        axis=0,
                        ddof=1,
                    ),
                    0.05
                    * np.sqrt(np.diag(geometries[context.country].covariance)),
                )
                for context in contexts
            }
            global_jump = np.zeros(total, dtype=bool)

            for target in targets:
                donor_a, donor_b = rng.choice(donors, size=2, replace=False)
                global_jump[target] = bool(
                    rng.random()
                    < DIFFERENTIAL_EVOLUTION_GLOBAL_JUMP_PROBABILITY
                )
                gamma = 1.0 if global_jump[target] else base_gamma
                structural_proposal[target, active_indices] += (
                    gamma
                    * (
                        cloud.structural[donor_a, active_indices]
                        - cloud.structural[donor_b, active_indices]
                    )
                    + rng.normal(0.0, structural_jitter_sd)
                )
                for context in contexts:
                    country = context.country
                    state_proposal[country][target] += (
                        gamma
                        * (
                            cloud.state_by_country[country][donor_a]
                            - cloud.state_by_country[country][donor_b]
                        )
                        + rng.normal(0.0, state_jitter_sd[country])
                    )

            for context in contexts:
                country = context.country
                state_proposal[country][targets] = _reflect_into_box(
                    state_proposal[country][targets],
                    context.lower,
                    context.upper,
                )

            proposed_structural_prior = np.asarray(
                [
                    _log_nuisance_prior_transformed(
                        vector, reference.prior_base, reference.priors
                    )
                    for vector in structural_proposal
                ],
                dtype=float,
            )
            target_mask = np.zeros(total, dtype=bool)
            target_mask[targets] = True
            valid = target_mask & np.isfinite(proposed_structural_prior)
            proposed_state_prior: dict[str, np.ndarray] = {}
            proposed_potential: dict[str, np.ndarray] = {}
            for context in contexts:
                country = context.country
                geometry = geometries[country]
                proposed_state_prior[country] = _log_gaussian_kernel(
                    state_proposal[country], geometry.mean, geometry.precision
                )
                valid &= np.isfinite(proposed_state_prior[country])
                proposed_potential[country] = _evaluate_country_potential(
                    context,
                    structural_proposal,
                    state_proposal[country],
                    geometry,
                    n_jobs=n_jobs,
                    active=valid,
                )
                valid &= np.isfinite(proposed_potential[country])

            log_alpha = proposed_structural_prior - cloud.structural_log_prior
            for context in contexts:
                country = context.country
                log_alpha += (
                    proposed_state_prior[country]
                    - cloud.state_log_prior_by_country[country]
                    + float(temperature)
                    * (
                        proposed_potential[country]
                        - cloud.potential_by_country[country]
                    )
                )
            accept = valid & (np.log(rng.random(total)) < log_alpha)
            if accept.any():
                cloud.structural[accept] = structural_proposal[accept]
                cloud.structural_log_prior[accept] = proposed_structural_prior[accept]
                for context in contexts:
                    country = context.country
                    cloud.state_by_country[country][accept] = state_proposal[country][
                        accept
                    ]
                    cloud.state_log_prior_by_country[country][accept] = (
                        proposed_state_prior[country][accept]
                    )
                    cloud.potential_by_country[country][accept] = proposed_potential[
                        country
                    ][accept]
            accepted += int(np.sum(accept[targets]))
            proposed += len(targets)
            accepted_global += int(np.sum(accept[targets] & global_jump[targets]))
            proposed_global += int(np.sum(global_jump[targets]))
            accepted_local += int(np.sum(accept[targets] & ~global_jump[targets]))
            proposed_local += int(np.sum(~global_jump[targets]))

    return JointMoveAcceptance(
        overall=float(accepted / proposed) if proposed else 0.0,
        local=float(accepted_local / proposed_local) if proposed_local else 0.0,
        global_jump=(
            float(accepted_global / proposed_global) if proposed_global else 0.0
        ),
    )


def _country_guided_independence_move(
    cloud: ParticleCloud,
    context: CountryStateContext,
    prior_geometry: PriorGeometry,
    proposal_geometry: PriorGeometry,
    *,
    temperature: float,
    n_jobs: int,
    rng: np.random.Generator,
) -> float:
    """Independence-MH move from a bounded Laplace state proposal."""

    country = context.country
    current = cloud.state_by_country[country]
    proposal = _state_prior_draws(
        proposal_geometry,
        context,
        count=len(current),
        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
    )
    proposed_log_prior = _log_gaussian_kernel(
        proposal, prior_geometry.mean, prior_geometry.precision
    )
    proposed_log_q = _log_gaussian_kernel(
        proposal, proposal_geometry.mean, proposal_geometry.precision
    )
    current_log_q = _log_gaussian_kernel(
        current, proposal_geometry.mean, proposal_geometry.precision
    )
    proposed_potential = _evaluate_country_potential(
        context,
        cloud.structural,
        proposal,
        prior_geometry,
        n_jobs=n_jobs,
    )
    valid = np.isfinite(proposed_potential)
    log_alpha = (
        proposed_log_prior
        - cloud.state_log_prior_by_country[country]
        + float(temperature)
        * (proposed_potential - cloud.potential_by_country[country])
        + current_log_q
        - proposed_log_q
    )
    accept = valid & (np.log(rng.random(len(log_alpha))) < log_alpha)
    if accept.any():
        cloud.state_by_country[country][accept] = proposal[accept]
        cloud.state_log_prior_by_country[country][accept] = proposed_log_prior[accept]
        cloud.potential_by_country[country][accept] = proposed_potential[accept]
    return float(np.mean(accept))


def _tempered_guided_geometry(
    prior: PriorGeometry,
    laplace: PriorGeometry,
    temperature: float,
) -> PriorGeometry:
    """Deterministically bridge the independence proposal from prior to Laplace."""

    weight = _guided_geometry_weight(temperature)
    mean = (1.0 - weight) * prior.mean + weight * laplace.mean
    covariance = (1.0 - weight) * prior.covariance + weight * laplace.covariance
    covariance = 0.5 * (covariance + covariance.T)
    return PriorGeometry(mean, covariance, np.linalg.inv(covariance))


def _guided_geometry_weight(temperature: float) -> float:
    clipped = float(np.clip(float(temperature), 0.0, 1.0))
    return float(
        np.log1p(clipped / GUIDED_BRIDGE_TEMPERATURE_SCALE)
        / np.log1p(1.0 / GUIDED_BRIDGE_TEMPERATURE_SCALE)
    )


def _initial_cloud(
    contexts: list[CountryStateContext],
    geometries: dict[str, PriorGeometry],
    *,
    islands: int,
    particles_per_island: int,
    seed: int,
    n_jobs: int,
) -> ParticleCloud:
    total = int(islands) * int(particles_per_island)
    reference = contexts[0]
    structural_parts: list[np.ndarray] = []
    for island in range(int(islands)):
        unit = _latin_hypercube(
            int(particles_per_island),
            len(IMPORTANCE_NUISANCE_INDICES),
            seed=int(seed) + island * 104729,
        )
        structural_parts.append(
            np.vstack(
                [
                    _nuisance_vector_from_unit_cube(
                        row,
                        reference.structural_start,
                        reference.prior_base,
                        reference.priors,
                    )
                    for row in unit
                ]
            )
        )
    structural = np.vstack(structural_parts)
    structural_log_prior = np.asarray(
        [
            _log_nuisance_prior_transformed(
                vector, reference.prior_base, reference.priors
            )
            for vector in structural
        ],
        dtype=float,
    )
    if not np.isfinite(structural_log_prior).all():
        raise RuntimeError("Joint SMC prior generator produced invalid shared draws")
    state_by_country: dict[str, np.ndarray] = {}
    state_log_prior: dict[str, np.ndarray] = {}
    potential: dict[str, np.ndarray] = {}
    for country_position, context in enumerate(contexts):
        geometry = geometries[context.country]
        state = _state_prior_draws(
            geometry,
            context,
            count=total,
            seed=int(seed) + (country_position + 1) * 1009,
        )
        state_by_country[context.country] = state
        state_log_prior[context.country] = _log_gaussian_kernel(
            state, geometry.mean, geometry.precision
        )
        potential[context.country] = _evaluate_country_potential(
            context,
            structural,
            state,
            geometry,
            n_jobs=n_jobs,
        )
        finite = np.isfinite(potential[context.country])
        finite_by_island = finite.reshape(int(islands), int(particles_per_island)).sum(axis=1)
        if bool((finite_by_island < max(8, int(0.25 * particles_per_island))).any()):
            raise RuntimeError(
                "Joint SMC initial prior has insufficient finite exact-target "
                f"support for {context.country}: {finite_by_island.tolist()}"
            )
    island = np.repeat(np.arange(1, int(islands) + 1), int(particles_per_island))
    weights = np.tile(
        np.full(int(particles_per_island), 1.0 / int(particles_per_island)),
        int(islands),
    )
    joint_finite = np.isfinite(
        np.sum(np.column_stack(list(potential.values())), axis=1)
    ).reshape(int(islands), int(particles_per_island)).sum(axis=1)
    if bool((joint_finite < max(8, int(0.25 * particles_per_island))).any()):
        raise RuntimeError(
            "Joint SMC initial prior has insufficient joint finite support by "
            f"island: {joint_finite.tolist()}"
        )
    return ParticleCloud(
        structural=structural,
        state_by_country=state_by_country,
        structural_log_prior=structural_log_prior,
        state_log_prior_by_country=state_log_prior,
        potential_by_country=potential,
        weights=weights,
        island=island,
        ancestor=np.arange(total, dtype=int),
    )


def _posterior_rows(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    *,
    seed: int,
    stage_count: int,
    minimum_stage_ess_fraction: float,
    maximum_stage_weight: float,
    move_acceptance: float,
    island_log_evidence: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    total = len(cloud.structural)
    island_positions = _island_indices(cloud.island)
    reordered = np.concatenate(
        [rng.permutation(positions) for positions in island_positions]
    )
    unique_particles = len(
        np.unique(
            np.round(
                np.column_stack(
                    (
                        cloud.structural[:, IMPORTANCE_NUISANCE_INDICES],
                        *[cloud.state_by_country[c.country] for c in contexts],
                    )
                ),
                decimals=10,
            ),
            axis=0,
        )
    )
    unique_fraction = float(unique_particles / total)
    unique_ancestor_fraction = float(len(np.unique(cloud.ancestor)) / total)
    log_evidence = np.asarray(island_log_evidence, dtype=float)
    island_weight = _normalise_log_weights(log_evidence)
    island_ess = float(1.0 / np.sum(island_weight**2))
    maximum_island_weight = float(np.max(island_weight))
    combined_ess = float(len(island_positions[0]) * island_ess)
    for particle_index in reordered:
        island = int(cloud.island[particle_index])
        within_draw = int(np.flatnonzero(reordered[cloud.island[reordered] == island] == particle_index)[0] + 1)
        shared = _sample_from_vector(cloud.structural[particle_index], contexts[0].priors)
        joint_log_prob = float(cloud.structural_log_prior[particle_index])
        joint_log_prob += sum(
            float(cloud.state_log_prior_by_country[c.country][particle_index])
            + float(cloud.potential_by_country[c.country][particle_index])
            for c in contexts
        )
        for country_position, context in enumerate(contexts):
            coordinate = cloud.state_by_country[context.country][particle_index]
            process_values = {
                name: float(value)
                for name, value in zip(context.coordinate_names[2:], coordinate[2:])
            }
            process_rng = np.random.default_rng(
                int(seed) + int(particle_index) * 1009 + country_position * 104729
            )
            previous = float(coordinate[-1])
            for year in range(
                context.historical_end_year + 1, context.forecast_end_year + 1
            ):
                previous = float(
                    context.process_rho * previous
                    + process_rng.normal(0.0, context.process_innovation_sd)
                )
                process_values[f"log_beta_process_{year}"] = previous
            rows.append(
                {
                    "country": context.country,
                    "chain": island,
                    "draw": within_draw,
                    "posterior_draw": int(particle_index + 1),
                    "structural_draw_id": int(particle_index + 1),
                    "source_particle_index": int(particle_index),
                    "source_initial_ancestor": int(cloud.ancestor[particle_index]),
                    "source_normalized_weight": 1.0 / total,
                    "posterior_log_prob": joint_log_prob,
                    "sampling_method": SAMPLING_METHOD,
                    "inference_structure": INFERENCE_STRUCTURE,
                    "uncertainty_target": UNCERTAINTY_SCOPE,
                    "accepted_fraction": move_acceptance,
                    "forecast_process_rho": context.process_rho,
                    "forecast_process_innovation_sd": context.process_innovation_sd,
                    "forecast_process_years": max(
                        context.forecast_end_year - context.historical_end_year, 0
                    ),
                    "historical_process_end_year": context.historical_end_year,
                    "smc_particle_count": len(island_positions[0]),
                    "smc_stage_count": int(stage_count),
                    "smc_final_temperature": 1.0,
                    "smc_min_ess": minimum_stage_ess_fraction * len(island_positions[0]),
                    "smc_final_ess": float(len(island_positions[0])),
                    "smc_min_ess_fraction": minimum_stage_ess_fraction,
                    "smc_final_ess_fraction": 1.0,
                    "smc_max_weight": maximum_stage_weight,
                    "smc_final_max_weight": 1.0 / len(island_positions[0]),
                    "smc_combined_particle_count": total,
                    "smc_combined_particle_ess": combined_ess,
                    "smc_combined_particle_ess_fraction": combined_ess / total,
                    "smc_combined_max_weight": maximum_island_weight
                    / len(island_positions[0]),
                    "smc_island_count": len(island_positions),
                    "smc_island_ess": island_ess,
                    "smc_island_ess_fraction": island_ess / len(island_positions),
                    "smc_max_island_weight": maximum_island_weight,
                    "smc_log_evidence_sd": _sample_standard_deviation(log_evidence),
                    "smc_log_evidence_range": float(
                        np.max(log_evidence) - np.min(log_evidence)
                    ),
                    "smc_unique_ancestor_fraction": unique_ancestor_fraction,
                    "smc_unique_particle_fraction": unique_fraction,
                    "smc_move_acceptance_fraction": move_acceptance,
                    "smc_reached_final_temperature": True,
                    "beta_S": float(np.exp(coordinate[0])),
                    "reporting_multiplier": float(np.exp(coordinate[1])),
                    **{name: float(shared[name]) for name in SHARED_PARAMETER_NAMES},
                    "resistance_prevalence": float(
                        context.runtime_reference["resistance"][
                            "target_prevalence_at_analysis_start"
                        ]
                    ),
                    "reporting_trend_end_multiplier": 1.0,
                    **process_values,
                }
            )
    return pd.DataFrame(rows)


def _boundary_audit(
    cloud: ParticleCloud,
    contexts: list[CountryStateContext],
    *,
    edge_fraction: float = 0.01,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        values = cloud.state_by_country[context.country]
        width = context.upper - context.lower
        lower_edge = context.lower + float(edge_fraction) * width
        upper_edge = context.upper - float(edge_fraction) * width
        for position, coordinate in enumerate(context.coordinate_names):
            column = values[:, position]
            lower_mass = float(np.mean(column <= lower_edge[position]))
            upper_mass = float(np.mean(column >= upper_edge[position]))
            rows.append(
                {
                    "country": context.country,
                    "coordinate": coordinate,
                    "lower_bound": float(context.lower[position]),
                    "upper_bound": float(context.upper[position]),
                    "edge_fraction_of_span": float(edge_fraction),
                    "lower_edge_mass": lower_mass,
                    "upper_edge_mass": upper_mass,
                    "maximum_edge_mass": max(lower_mass, upper_mass),
                    "q005": float(np.quantile(column, 0.005)),
                    "q500": float(np.quantile(column, 0.500)),
                    "q995": float(np.quantile(column, 0.995)),
                }
            )
    return pd.DataFrame(rows)


def run_hierarchical_joint_smc(
    *,
    countries: list[str] | None = None,
    islands: int = 4,
    particles_per_island: int = 1024,
    target_ess_fraction: float = 0.70,
    max_stages: int = 100,
    move_rounds_per_stage: int = 4,
    final_move_rounds: int = 4,
    final_mode_bank_rounds: int = 0,
    global_covariance_fraction: float = 0.25,
    n_jobs: int = 96,
    seed: int = 20260715,
    output_stem: str = DEFAULT_STEM,
    require_recommended: bool = False,
    mode_bank_stems: list[str] | None = None,
    mode_bank_covariance_inflation: float = MODE_BANK_COVARIANCE_INFLATION,
    mode_bank_covariance_floor_fraction: float = (
        MODE_BANK_COVARIANCE_FLOOR_FRACTION
    ),
    mode_bank_degrees_of_freedom: float = MODE_BANK_DEGREES_OF_FREEDOM,
    mode_bank_activation_temperature: float = MODE_BANK_ACTIVATION_TEMPERATURE,
    mode_bank_crossfit_current_population: bool = True,
    process_support_bound: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if int(islands) < 1 or int(particles_per_island) < 16:
        raise ValueError("Joint SMC requires at least 1 island and 16 particles/island")
    if not 0.25 <= float(target_ess_fraction) <= 0.95:
        raise ValueError("Joint SMC target ESS fraction must be in [0.25, 0.95]")
    if int(move_rounds_per_stage) < 1 or int(final_move_rounds) < 1:
        raise ValueError("Joint SMC move-round counts must be positive")
    if int(final_mode_bank_rounds) < 0:
        raise ValueError("Final mode-bank round count cannot be negative")
    if int(final_mode_bank_rounds) and not mode_bank_stems:
        raise ValueError("Final mode-bank rounds require a mode-bank proposal")
    if not 0.0 <= float(global_covariance_fraction) <= 0.75:
        raise ValueError("Global covariance fraction must be in [0, 0.75]")
    if not 0.0 <= float(mode_bank_activation_temperature) <= 1.0:
        raise ValueError("Mode-bank activation temperature must be in [0, 1]")
    if require_recommended and (
        int(islands) < RECOMMENDED_MIN_ISLANDS
        or int(particles_per_island) < RECOMMENDED_MIN_ISLAND_PARTICLES
    ):
        raise ValueError(
            "Recommended joint SMC requires at least 4 islands and 512 particles/island"
        )
    configs = load_configs()
    calibrated_countries = publication_country_names(configs)
    resolved = countries or calibrated_countries
    outside = sorted(set(resolved).difference(calibrated_countries))
    if outside:
        raise ValueError(f"Countries outside the prespecified calibrated set: {outside}")
    contexts = [
        _with_process_support(
            _context(country, configs=configs),
            process_support_bound,
        )
        for country in resolved
    ]
    _assert_shared_prior_contract(contexts)
    mode_bank = (
        _load_mode_bank(
            list(mode_bank_stems),
            contexts,
            covariance_inflation=mode_bank_covariance_inflation,
            covariance_floor_fraction=mode_bank_covariance_floor_fraction,
            degrees_of_freedom=mode_bank_degrees_of_freedom,
        )
        if mode_bank_stems
        else None
    )
    expected_mode_dimension = len(IMPORTANCE_NUISANCE_INDICES) + sum(
        len(context.coordinate_names) for context in contexts
    )
    if mode_bank is not None and mode_bank.dimension != expected_mode_dimension:
        raise ValueError(
            "Mode-bank joint dimension does not match the requested countries: "
            f"{mode_bank.dimension} != {expected_mode_dimension}"
        )
    geometries: dict[str, PriorGeometry] = {}
    guided_geometries: dict[str, PriorGeometry] = {}
    for context in contexts:
        mean, covariance, precision = _state_gaussian_prior_geometry(
            context, context.prior_base
        )
        geometries[context.country] = PriorGeometry(mean, covariance, precision)
        guided_covariance = (
            float(GUIDED_STATE_COVARIANCE_SCALE)
            * np.asarray(context.state_covariance, dtype=float)
        )
        guided_covariance = 0.5 * (guided_covariance + guided_covariance.T)
        guided_geometries[context.country] = PriorGeometry(
            np.asarray(context.state_mean, dtype=float),
            guided_covariance,
            np.linalg.inv(guided_covariance),
        )

    workers = min(max(1, int(n_jobs)), available_cpus())
    fingerprint = hashlib.sha256()
    fingerprint.update(config_fingerprint().encode("utf-8"))
    fingerprint.update(source_code_fingerprint().encode("utf-8"))
    fingerprint.update(
        repr(
            (
                resolved,
                int(islands),
                int(particles_per_island),
                float(target_ess_fraction),
                int(max_stages),
                int(move_rounds_per_stage),
                int(final_move_rounds),
                int(final_mode_bank_rounds),
                float(global_covariance_fraction),
                int(seed),
                float(process_support_bound)
                if process_support_bound is not None
                else None,
                tuple(mode_bank.source_stems) if mode_bank is not None else (),
                tuple(mode_bank.source_hashes) if mode_bank is not None else (),
                float(mode_bank_covariance_inflation),
                float(mode_bank_covariance_floor_fraction),
                float(mode_bank_degrees_of_freedom),
                float(mode_bank_activation_temperature),
                bool(mode_bank_crossfit_current_population),
            )
        ).encode("utf-8")
    )
    run_fingerprint = fingerprint.hexdigest()
    checkpoint_path = project_path(
        "outputs", "metadata", f"{output_stem}_smc_checkpoint.joblib"
    )
    rng = np.random.default_rng(int(seed))
    if checkpoint_path.exists():
        checkpoint = joblib_load(checkpoint_path)
        if checkpoint.get("fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Stale/incompatible SMC checkpoint exists: {checkpoint_path}"
            )
        cloud = _cloud_from_payload(checkpoint["cloud"])
        temperature = float(checkpoint["temperature"])
        shared_scale = float(checkpoint["shared_scale"])
        joint_scale = float(checkpoint.get("joint_scale", 1.0))
        state_scale = dict(checkpoint["state_scale"])
        stage_rows = list(checkpoint["stage_rows"])
        all_acceptance = list(checkpoint["all_acceptance"])
        all_mode_bank_acceptance = list(
            checkpoint.get("all_mode_bank_acceptance", [])
        )
        evidence = np.asarray(checkpoint["evidence"], dtype=float)
        minimum_stage_fraction = float(checkpoint["minimum_stage_fraction"])
        maximum_stage_weight = float(checkpoint["maximum_stage_weight"])
        rng.bit_generator.state = checkpoint["rng_state"]
    else:
        cloud = _initial_cloud(
            contexts,
            geometries,
            islands=int(islands),
            particles_per_island=int(particles_per_island),
            seed=int(seed) + 1009,
            n_jobs=workers,
        )
        temperature = 0.0
        shared_scale = 1.0
        joint_scale = 1.0
        state_scale = {context.country: 1.0 for context in contexts}
        stage_rows: list[dict[str, Any]] = []
        all_acceptance: list[float] = []
        all_mode_bank_acceptance: list[float] = []
        evidence = np.zeros(int(islands), dtype=float)
        minimum_stage_fraction = 1.0
        maximum_stage_weight = 0.0

    for stage in range(len(stage_rows) + 1, int(max_stages) + 1):
        total_potential = np.sum(
            np.column_stack(list(cloud.potential_by_country.values())), axis=1
        )
        previous = float(temperature)
        previous_weights = cloud.weights.copy()
        temperature, cloud.weights, island_ess = _find_next_temperature(
            cloud.weights,
            total_potential,
            cloud.island,
            temperature,
            float(target_ess_fraction),
        )
        increment = float(temperature - previous)
        island_max: list[float] = []
        for island_position, positions in enumerate(_island_indices(cloud.island)):
            log_increment = increment * total_potential[positions]
            evidence[island_position] += float(
                scipy_special.logsumexp(
                    np.log(np.clip(previous_weights[positions], 1e-300, 1.0))
                    + log_increment
                )
            )
            local = cloud.weights[positions]
            local /= float(local.sum())
            island_max.append(float(np.max(local)))
        fraction = island_ess / float(particles_per_island)
        minimum_stage_fraction = min(minimum_stage_fraction, float(np.min(fraction)))
        maximum_stage_weight = max(maximum_stage_weight, float(np.max(island_max)))
        resampled = _resample_islands(cloud, rng=rng, force=False)
        stage_guided_geometries = {
            context.country: _tempered_guided_geometry(
                geometries[context.country],
                guided_geometries[context.country],
                temperature,
            )
            for context in contexts
        }

        shared_acceptance_values: list[float] = []
        joint_block_acceptance_values: list[float] = []
        joint_local_acceptance_values: list[float] = []
        joint_global_acceptance_values: list[float] = []
        mode_bank_acceptance_values: list[float] = []
        shared_local_acceptance_values: list[float] = []
        shared_guided_acceptance_values: list[float] = []
        country_acceptance_values: dict[str, list[float]] = {
            context.country: [] for context in contexts
        }
        country_guided_acceptance_values: dict[str, list[float]] = {
            context.country: [] for context in contexts
        }
        shared_guided_proposal = (
            "prior"
            if temperature < SHARED_POPULATION_GUIDE_SWITCH_TEMPERATURE
            else "population_gaussian_mixture"
        )
        mode_bank_active = bool(
            mode_bank is not None
            and temperature >= float(mode_bank_activation_temperature)
        )
        move_cycle = _stage_move_cycle(
            stage,
            mode_bank_active=mode_bank_active,
        )
        for _move_round in range(int(move_rounds_per_stage)):
            move_kind = move_cycle[_move_round % len(move_cycle)]
            if move_kind == "mode_bank":
                if mode_bank is None:
                    raise RuntimeError("Mode-bank move scheduled without a proposal")
                mode_bank_acceptance = _joint_mode_bank_move(
                    cloud,
                    contexts,
                    geometries,
                    mode_bank,
                    temperature=temperature,
                    n_jobs=workers,
                    rng=rng,
                    crossfit_current_population=(
                        mode_bank_crossfit_current_population
                    ),
                )
                mode_bank_acceptance_values.append(mode_bank_acceptance)
                all_mode_bank_acceptance.append(mode_bank_acceptance)
                shared_acceptance_values.append(mode_bank_acceptance)
                for context in contexts:
                    country_acceptance_values[context.country].append(
                        mode_bank_acceptance
                    )
                all_acceptance.append(mode_bank_acceptance)
                continue
            if move_kind == "joint":
                joint_result = _joint_differential_evolution_move(
                    cloud,
                    contexts,
                    geometries,
                    temperature=temperature,
                    scale=joint_scale,
                    n_jobs=workers,
                    rng=rng,
                )
                joint_acceptance = joint_result.overall
                joint_block_acceptance_values.append(joint_acceptance)
                joint_local_acceptance_values.append(joint_result.local)
                joint_global_acceptance_values.append(joint_result.global_jump)
                shared_acceptance_values.append(joint_acceptance)
                for context in contexts:
                    country_acceptance_values[context.country].append(
                        joint_acceptance
                    )
                all_acceptance.append(joint_acceptance)
                if joint_result.local < 0.15:
                    joint_scale = max(0.10, joint_scale * 0.80)
                elif joint_result.local > 0.35:
                    joint_scale = min(4.0, joint_scale * 1.20)
                continue

            guided_round = move_kind == "guided"
            if guided_round:
                if shared_guided_proposal == "prior":
                    shared_acceptance = _shared_prior_independence_move(
                        cloud,
                        contexts,
                        geometries,
                        temperature=temperature,
                        n_jobs=workers,
                        rng=rng,
                    )
                else:
                    shared_acceptance = _shared_population_independence_move(
                        cloud,
                        contexts,
                        geometries,
                        temperature=temperature,
                        n_jobs=workers,
                        rng=rng,
                    )
                shared_guided_acceptance_values.append(shared_acceptance)
            else:
                shared_acceptance = _shared_move(
                    cloud,
                    contexts,
                    geometries,
                    temperature=temperature,
                    scale=shared_scale,
                    global_covariance_fraction=global_covariance_fraction,
                    n_jobs=workers,
                    rng=rng,
                )
                shared_local_acceptance_values.append(shared_acceptance)
            shared_acceptance_values.append(shared_acceptance)
            all_acceptance.append(shared_acceptance)
            if not guided_round and shared_acceptance < 0.15:
                shared_scale = max(0.05, shared_scale * 0.70)
            elif not guided_round and shared_acceptance > 0.40:
                shared_scale = min(4.0, shared_scale * 1.20)
            for context in contexts:
                if guided_round:
                    acceptance = _country_guided_independence_move(
                        cloud,
                        context,
                        geometries[context.country],
                        stage_guided_geometries[context.country],
                        temperature=temperature,
                        n_jobs=workers,
                        rng=rng,
                    )
                    country_guided_acceptance_values[context.country].append(
                        acceptance
                    )
                else:
                    acceptance = _country_state_move(
                        cloud,
                        context,
                        geometries[context.country],
                        temperature=temperature,
                        scale=state_scale[context.country],
                        global_covariance_fraction=global_covariance_fraction,
                        n_jobs=workers,
                        rng=rng,
                    )
                country_acceptance_values[context.country].append(acceptance)
                all_acceptance.append(acceptance)
                if not guided_round and acceptance < 0.15:
                    state_scale[context.country] = max(
                        0.05, state_scale[context.country] * 0.70
                    )
                elif not guided_round and acceptance > 0.40:
                    state_scale[context.country] = min(
                        4.0, state_scale[context.country] * 1.20
                    )
        shared_acceptance = float(np.mean(shared_acceptance_values))
        country_acceptance = {
            country: float(np.mean(values))
            for country, values in country_acceptance_values.items()
        }
        stage_rows.append(
            {
                "stage": stage,
                "temperature": temperature,
                "temperature_increment": increment,
                "minimum_island_ess": float(np.min(island_ess)),
                "minimum_island_ess_fraction": float(np.min(fraction)),
                "maximum_island_particle_weight": float(np.max(island_max)),
                "resampled_particle_fraction": float(np.mean(resampled)),
                "move_rounds_per_stage": int(move_rounds_per_stage),
                "global_covariance_fraction": float(global_covariance_fraction),
                "joint_block_move_fraction": float(
                    len(joint_block_acceptance_values)
                    / int(move_rounds_per_stage)
                ),
                "joint_block_move_acceptance": _mean_or_zero(
                    joint_block_acceptance_values
                ),
                "joint_local_jump_acceptance": _mean_or_zero(
                    joint_local_acceptance_values
                ),
                "joint_global_jump_acceptance": _mean_or_zero(
                    joint_global_acceptance_values
                ),
                "joint_differential_scale": joint_scale,
                "mode_bank_active": mode_bank_active,
                "mode_bank_move_fraction": float(
                    len(mode_bank_acceptance_values)
                    / int(move_rounds_per_stage)
                ),
                "mode_bank_move_acceptance": _mean_or_zero(
                    mode_bank_acceptance_values
                ),
                "guided_geometry_weight": _guided_geometry_weight(temperature),
                "shared_guided_proposal": shared_guided_proposal,
                "shared_move_acceptance": shared_acceptance,
                "shared_local_move_acceptance": _mean_or_zero(
                    shared_local_acceptance_values
                ),
                "shared_guided_independence_acceptance": _mean_or_zero(
                    shared_guided_acceptance_values
                ),
                "mean_country_state_move_acceptance": float(
                    np.mean(list(country_acceptance.values()))
                ),
                "mean_country_guided_independence_acceptance": _mean_or_zero(
                    [
                        value
                        for values in country_guided_acceptance_values.values()
                        for value in values
                    ]
                ),
                "minimum_country_state_move_acceptance": float(
                    np.min(list(country_acceptance.values()))
                ),
                "shared_move_scale": shared_scale,
                **{
                    f"state_acceptance_{country}": value
                    for country, value in country_acceptance.items()
                },
            }
        )
        _atomic_checkpoint(
            checkpoint_path,
            {
                "fingerprint": run_fingerprint,
                "cloud": _cloud_payload(cloud),
                "temperature": temperature,
                "shared_scale": shared_scale,
                "joint_scale": joint_scale,
                "state_scale": state_scale,
                "stage_rows": stage_rows,
                "all_acceptance": all_acceptance,
                "all_mode_bank_acceptance": all_mode_bank_acceptance,
                "evidence": evidence,
                "minimum_stage_fraction": minimum_stage_fraction,
                "maximum_stage_weight": maximum_stage_weight,
                "rng_state": rng.bit_generator.state,
            },
        )
        # Human-readable progress survives terminal/session loss; absence of
        # final metadata still prevents a partial run from passing any gate.
        write_dataframe(
            pd.DataFrame(stage_rows),
            project_path(
                "outputs", "diagnostics", f"{output_stem}_smc_stage_audit.csv"
            ),
        )
        if temperature >= 1.0 - 1e-12:
            temperature = 1.0
            break

    reached_final = bool(temperature >= 1.0 - 1e-12)
    if not reached_final:
        raise RuntimeError(
            f"Joint SMC stopped at temperature={temperature:.8f} after {max_stages} stages"
        )
    _resample_islands(cloud, rng=rng, force=True)
    final_move_cycle = (
        ("mode_bank", "local", "guided", "joint")
        if mode_bank is not None
        else ("joint", "local", "guided")
    )
    for final_round in range(int(final_move_rounds)):
        move_kind = final_move_cycle[final_round % len(final_move_cycle)]
        if move_kind == "mode_bank":
            if mode_bank is None:
                raise RuntimeError("Mode-bank move scheduled without a proposal")
            mode_bank_acceptance = _joint_mode_bank_move(
                cloud,
                contexts,
                geometries,
                mode_bank,
                temperature=1.0,
                n_jobs=workers,
                rng=rng,
                crossfit_current_population=(
                    mode_bank_crossfit_current_population
                ),
            )
            all_mode_bank_acceptance.append(mode_bank_acceptance)
            all_acceptance.append(mode_bank_acceptance)
            continue
        if move_kind == "joint":
            joint_result = _joint_differential_evolution_move(
                cloud,
                contexts,
                geometries,
                temperature=1.0,
                scale=joint_scale,
                n_jobs=workers,
                rng=rng,
            )
            all_acceptance.append(joint_result.overall)
            continue

        guided_round = move_kind == "guided"
        if guided_round:
            shared_acceptance = _shared_population_independence_move(
                cloud,
                contexts,
                geometries,
                temperature=1.0,
                n_jobs=workers,
                rng=rng,
            )
        else:
            shared_acceptance = _shared_move(
                cloud,
                contexts,
                geometries,
                temperature=1.0,
                scale=shared_scale,
                global_covariance_fraction=global_covariance_fraction,
                n_jobs=workers,
                rng=rng,
            )
        all_acceptance.append(shared_acceptance)
        for context in contexts:
            if guided_round:
                acceptance = _country_guided_independence_move(
                    cloud,
                    context,
                    geometries[context.country],
                    _tempered_guided_geometry(
                        geometries[context.country],
                        guided_geometries[context.country],
                        1.0,
                    ),
                    temperature=1.0,
                    n_jobs=workers,
                    rng=rng,
                )
            else:
                acceptance = _country_state_move(
                    cloud,
                    context,
                    geometries[context.country],
                    temperature=1.0,
                    scale=state_scale[context.country],
                    global_covariance_fraction=global_covariance_fraction,
                    n_jobs=workers,
                    rng=rng,
                )
            all_acceptance.append(acceptance)

    if mode_bank is not None:
        for _ in range(int(final_mode_bank_rounds)):
            mode_bank_acceptance = _joint_mode_bank_move(
                cloud,
                contexts,
                geometries,
                mode_bank,
                temperature=1.0,
                n_jobs=workers,
                rng=rng,
                crossfit_current_population=(
                    mode_bank_crossfit_current_population
                ),
            )
            all_mode_bank_acceptance.append(mode_bank_acceptance)
            all_acceptance.append(mode_bank_acceptance)

    mean_acceptance = float(np.mean(all_acceptance))
    nominal_move_fractions = _mean_move_fractions(
        int(move_rounds_per_stage),
        mode_bank_active=mode_bank is not None,
    )
    posterior = _posterior_rows(
        cloud,
        contexts,
        seed=int(seed) + 9001,
        stage_count=len(stage_rows),
        minimum_stage_ess_fraction=minimum_stage_fraction,
        maximum_stage_weight=maximum_stage_weight,
        move_acceptance=mean_acceptance,
        island_log_evidence=evidence,
    )
    parameter_columns = tuple(SHARED_PARAMETER_NAMES) + (
        "beta_S",
        "reporting_multiplier",
    ) + tuple(
        sorted(
            column
            for column in posterior.columns
            if str(column).startswith("log_beta_process_")
        )
    )
    diagnostics = _compute_smc_diagnostics(posterior, parameter_columns)
    convergence = summarize_convergence(diagnostics)
    stage_audit = pd.DataFrame(stage_rows)
    boundary_audit = _boundary_audit(cloud, contexts)
    maximum_boundary_mass = float(boundary_audit["maximum_edge_mass"].max())
    quality_checks = {
        "reached_final_temperature": reached_final,
        "minimum_stage_ess_fraction": minimum_stage_fraction
        >= RECOMMENDED_MIN_STAGE_ESS_FRACTION,
        "maximum_final_particle_weight": (1.0 / int(particles_per_island))
        <= RECOMMENDED_MAX_FINAL_WEIGHT,
        "move_acceptance": RECOMMENDED_MIN_MOVE_ACCEPTANCE
        <= mean_acceptance
        <= RECOMMENDED_MAX_MOVE_ACCEPTANCE,
        "state_boundary_mass": maximum_boundary_mass
        <= RECOMMENDED_MAX_BOUNDARY_MASS,
        "rank_and_tail_diagnostics": bool(
            not diagnostics.empty
            and diagnostics["recommended_converged"].astype(bool).all()
        ),
    }
    quality = {
        "checks": quality_checks,
        "all_recommended_converged": bool(all(quality_checks.values())),
        "minimum_stage_ess_fraction": minimum_stage_fraction,
        "maximum_stage_weight": maximum_stage_weight,
        "final_particle_weight": 1.0 / int(particles_per_island),
        "mean_move_acceptance": mean_acceptance,
        "maximum_state_boundary_mass": maximum_boundary_mass,
        "island_log_evidence": evidence.tolist(),
        "log_evidence_sd": _sample_standard_deviation(evidence),
        "log_evidence_range": float(np.max(evidence) - np.min(evidence)),
    }

    posterior_path = project_path(
        "outputs", "simulations", f"{output_stem}_posterior_samples.parquet"
    )
    diagnostics_path = project_path(
        "outputs", "summaries", f"{output_stem}_convergence_diagnostics.csv"
    )
    stage_path = project_path(
        "outputs", "diagnostics", f"{output_stem}_smc_stage_audit.csv"
    )
    boundary_path = project_path(
        "outputs", "diagnostics", f"{output_stem}_smc_boundary_audit.csv"
    )
    write_dataframe(posterior, posterior_path)
    write_dataframe(diagnostics, diagnostics_path)
    write_dataframe(stage_audit, stage_path)
    write_dataframe(boundary_audit, boundary_path)
    metadata = current_run_metadata(
        output_stem,
        row_counts={
            "posterior_samples": len(posterior),
            "convergence_diagnostics": len(diagnostics),
            "smc_stages": len(stage_audit),
            "smc_boundary_coordinates": len(boundary_audit),
        },
    ) | {
        "sampler": SAMPLING_METHOD,
        "inference_structure": INFERENCE_STRUCTURE,
        "uncertainty_scope": UNCERTAINTY_SCOPE,
        "analysis_role": ANALYSIS_ROLE,
        "publication_path": PUBLICATION_PATH,
        "figure2c_interval_source": FIGURE2C_INTERVAL_SOURCE,
        "countries": resolved,
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
        "islands": int(islands),
        "n_chains": int(islands),
        "particles_per_island": int(particles_per_island),
        "draws_per_chain": int(particles_per_island),
        "warmup": 0,
        "thin": 1,
        "parameterization": "beta_reporting_product",
        "chain_labels_are_smc_islands": True,
        "smc_islands_interact_through_exact_mh_moves": True,
        "combined_particles": int(islands) * int(particles_per_island),
        "target_ess_fraction": float(target_ess_fraction),
        "max_stages": int(max_stages),
        "move_rounds_per_stage": int(move_rounds_per_stage),
        "final_move_rounds": int(final_move_rounds),
        "final_mode_bank_rounds": int(final_mode_bank_rounds),
        "global_covariance_fraction": float(global_covariance_fraction),
        "process_support_bound": (
            float(process_support_bound)
            if process_support_bound is not None
            else None
        ),
        "mode_bank_enabled": mode_bank is not None,
        "mode_bank_source_stems": (
            list(mode_bank.source_stems) if mode_bank is not None else []
        ),
        "mode_bank_source_hashes": (
            list(mode_bank.source_hashes) if mode_bank is not None else []
        ),
        "mode_bank_dimension": (
            int(mode_bank.dimension) if mode_bank is not None else None
        ),
        "mode_bank_covariance_inflation": float(
            mode_bank_covariance_inflation
        ),
        "mode_bank_covariance_floor_fraction": float(
            mode_bank_covariance_floor_fraction
        ),
        "mode_bank_degrees_of_freedom": float(
            mode_bank_degrees_of_freedom
        ),
        "mode_bank_activation_temperature": float(
            mode_bank_activation_temperature
        ),
        "mode_bank_crossfit_current_population": bool(
            mode_bank is not None and mode_bank_crossfit_current_population
        ),
        "mode_bank_kernel": (
            "complementary_half_crossfit_independence_mh"
            if mode_bank is not None and mode_bank_crossfit_current_population
            else "fixed_historical_independence_mh"
            if mode_bank is not None
            else None
        ),
        "mode_bank_crossfit_covariance_inflation": float(
            MODE_BANK_CROSSFIT_COVARIANCE_INFLATION
        ),
        "mode_bank_crossfit_covariance_floor_fraction": float(
            MODE_BANK_CROSSFIT_COVARIANCE_FLOOR_FRACTION
        ),
        "mean_mode_bank_acceptance": _mean_or_zero(
            all_mode_bank_acceptance
        ),
        "joint_block_move_fraction": float(
            nominal_move_fractions["joint"]
        ),
        "mode_bank_move_fraction": float(
            nominal_move_fractions["mode_bank"]
        ),
        "guided_independence_move_fraction": float(
            nominal_move_fractions["guided"]
        ),
        "post_activation_move_schedule": (
            "mode_bank_plus_rotating_local_guided_joint"
            if mode_bank is not None
            else "joint_local_guided"
        ),
        "differential_evolution_global_jump_probability": float(
            DIFFERENTIAL_EVOLUTION_GLOBAL_JUMP_PROBABILITY
        ),
        "differential_evolution_jitter_fraction": float(
            DIFFERENTIAL_EVOLUTION_JITTER_FRACTION
        ),
        "differential_evolution_scale_adaptation": (
            "multiply_by_0.80_below_0.15_local_acceptance_"
            "and_1.20_above_0.35"
        ),
        "final_differential_evolution_scale": float(joint_scale),
        "differential_evolution_donor_scope": "cross_island_complete_population",
        "guided_state_covariance_scale": float(GUIDED_STATE_COVARIANCE_SCALE),
        "guided_geometry_bridge": "log_temperature_prior_to_inflated_laplace",
        "guided_bridge_temperature_scale": float(
            GUIDED_BRIDGE_TEMPERATURE_SCALE
        ),
        "shared_population_guide_switch_temperature": float(
            SHARED_POPULATION_GUIDE_SWITCH_TEMPERATURE
        ),
        "seed": int(seed),
        "n_jobs": workers,
        "checkpoint_path": str(checkpoint_path),
        "posterior_samples_sha256": file_sha256(posterior_path),
        "convergence_diagnostics_sha256": file_sha256(diagnostics_path),
        "smc_stage_audit_sha256": file_sha256(stage_path),
        "smc_boundary_audit_sha256": file_sha256(boundary_path),
        "quality": quality,
        "convergence_summary": convergence,
        "posterior_interpretation": (
            "Full-feedback joint posterior sampled by adaptive tempered SMC. "
            "All final particles have equal mass after island-wise resampling and "
            "MH-corrected joint differential-evolution, multimodal independence, "
            "local, and guided "
            "rejuvenation. Pilot particles define only fixed historical proposal "
            "components and do not enter the final posterior sample."
        ),
        "parameter_source_roles": {
            **{name: "shared_joint_posterior" for name in SHARED_PARAMETER_NAMES},
            "beta_S": "country_joint_posterior",
            "reporting_multiplier": "country_joint_posterior",
            "annual_log_beta_process": "country_joint_posterior",
        },
    }
    write_run_metadata(output_stem, metadata)
    checkpoint_path.unlink(missing_ok=True)
    if require_recommended and not quality["all_recommended_converged"]:
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError(
            "Joint tempered SMC failed recommended research diagnostic checks: "
            + ", ".join(failed)
        )
    return posterior, diagnostics, stage_audit, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default=None)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--particles-per-island", type=int, default=1024)
    parser.add_argument("--target-ess-fraction", type=float, default=0.70)
    parser.add_argument("--max-stages", type=int, default=100)
    parser.add_argument("--move-rounds-per-stage", type=int, default=4)
    parser.add_argument("--final-move-rounds", type=int, default=4)
    parser.add_argument("--final-mode-bank-rounds", type=int, default=0)
    parser.add_argument("--global-covariance-fraction", type=float, default=0.25)
    parser.add_argument("--n-jobs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-stem", default=DEFAULT_STEM)
    parser.add_argument("--require-recommended", action="store_true")
    parser.add_argument(
        "--mode-bank-stems",
        default=None,
        help="Comma-separated completed pilot stems defining a fixed proposal",
    )
    parser.add_argument(
        "--mode-bank-covariance-inflation",
        type=float,
        default=MODE_BANK_COVARIANCE_INFLATION,
    )
    parser.add_argument(
        "--mode-bank-covariance-floor-fraction",
        type=float,
        default=MODE_BANK_COVARIANCE_FLOOR_FRACTION,
    )
    parser.add_argument(
        "--mode-bank-degrees-of-freedom",
        type=float,
        default=MODE_BANK_DEGREES_OF_FREEDOM,
    )
    parser.add_argument(
        "--mode-bank-activation-temperature",
        type=float,
        default=MODE_BANK_ACTIVATION_TEMPERATURE,
    )
    parser.add_argument(
        "--mode-bank-crossfit-current-population",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append a complementary-half empirical component to the fixed mode "
            "bank; disable for a purely historical independence-MH proposal"
        ),
    )
    parser.add_argument(
        "--process-support-bound",
        type=float,
        default=None,
        help="Symmetric numerical support for annual log-beta process states",
    )
    args = parser.parse_args()
    run_hierarchical_joint_smc(
        countries=args.countries.split(",") if args.countries else None,
        islands=args.islands,
        particles_per_island=args.particles_per_island,
        target_ess_fraction=args.target_ess_fraction,
        max_stages=args.max_stages,
        move_rounds_per_stage=args.move_rounds_per_stage,
        final_move_rounds=args.final_move_rounds,
        final_mode_bank_rounds=args.final_mode_bank_rounds,
        global_covariance_fraction=args.global_covariance_fraction,
        n_jobs=args.n_jobs,
        seed=args.seed,
        output_stem=args.output_stem,
        require_recommended=args.require_recommended,
        mode_bank_stems=(
            [stem for stem in args.mode_bank_stems.split(",") if stem]
            if args.mode_bank_stems
            else None
        ),
        mode_bank_covariance_inflation=args.mode_bank_covariance_inflation,
        mode_bank_covariance_floor_fraction=(
            args.mode_bank_covariance_floor_fraction
        ),
        mode_bank_degrees_of_freedom=args.mode_bank_degrees_of_freedom,
        mode_bank_activation_temperature=args.mode_bank_activation_temperature,
        mode_bank_crossfit_current_population=(
            args.mode_bank_crossfit_current_population
        ),
        process_support_bound=args.process_support_bound,
    )


if __name__ == "__main__":
    main()
