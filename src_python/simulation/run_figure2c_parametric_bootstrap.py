from __future__ import annotations

"""Paired parametric-bootstrap confidence intervals for Figure 2c.

Each bootstrap replicate regenerates a complete annual AR(1) transmission path
and NB2 surveillance observations, refits the country state-space model, and
then propagates that same refit through current practice and all programme
strategies.  The resulting percentile intervals are frequentist confidence
intervals for the fitted model estimand.  They are deliberately not labelled
as posterior credible intervals or future-observation prediction intervals.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from src_python.calibration.calibrate_baseline import (
    _annual_ar1_transition_scales,
    _calibration_predicted_means,
    calibration_observed_case_frame,
    calibration_runtime_config,
    extend_annual_log_beta_conditional_mean,
    grouped_likelihood_observations,
    state_space_map_calibration,
)
from src_python.simulation.common import (
    current_run_metadata,
    load_calibrated_country_artifact,
    load_configs,
    make_config,
    publication_country_names,
    validate_calibration_artifacts,
    write_run_metadata,
)
from src_python.simulation.programme_uncertainty_helpers import (
    FIXED_PROGRAMME_REFERENCE_SOURCE,
    INTERVENTION_UNCERTAINTY_PREFIX,
    PRIMARY_RATE,
    PRIMARY_REDUCTION,
    PRIMARY_TOTAL,
    PROGRAMME_STRATEGIES as FIGURE2C_STRATEGIES,
    build_programme_scenarios as _build_scenarios,
    fixed_programme_reference_inputs,
    pair_programme_draws as _paired_draws,
)
from src_python.simulation.common import execute_scenario_summary_list
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import (
    available_cpus,
    configure_worker_thread_limits,
)


STEM = "figure2c_parametric_bootstrap"
ANALYSIS_ROLE = "publication_estimation_confidence_interval"
PUBLICATION_PATH = True
FIGURE2C_INTERVAL_SOURCE = True
DRAW_PATH = project_path(
    "outputs", "tables", "figure2c_programme_paired_bootstrap_draws.csv"
)
INTERVAL_PATH = project_path(
    "outputs", "summaries", "figure2c_programme_paired_confidence_intervals.csv"
)
FIT_DIAGNOSTIC_PATH = project_path(
    "outputs", "diagnostics", f"{STEM}_fit_diagnostics.csv"
)
STABILITY_PATH = project_path(
    "outputs", "diagnostics", f"{STEM}_interval_stability.csv"
)
CHECKPOINT_DRAW_PATH = project_path(
    "outputs", "simulations", f"{STEM}_checkpoint_draws.parquet"
)
CHECKPOINT_FIT_PATH = project_path(
    "outputs", "simulations", f"{STEM}_checkpoint_fits.parquet"
)
CHECKPOINT_METADATA_PATH = project_path(
    "outputs", "metadata", f"{STEM}_checkpoint.json"
)
CHECKPOINT_ARCHIVE_ROOT = project_path("outputs", "archive", STEM)
FRONTIER_PATH = project_path(
    "outputs", "tables", "lancet_child_adolescent_decision_frontier.csv"
)

INTERVENTION_STRATEGIES = tuple(
    strategy for strategy in FIGURE2C_STRATEGIES if strategy != "current"
)
INTERVAL_TYPE = "95% parametric-bootstrap confidence interval"
INTERVAL_BASIS = (
    "Paired marginal parametric bootstrap. Annual latent transmission paths "
    "are regenerated from the prespecified AR(1) process, surveillance counts "
    "are regenerated from the fitted NB2 observation model, and the country "
    "state-space model is fully refitted before the same fitted replicate is "
    "propagated through current practice and intervention. Biological and "
    "intervention-definition inputs remain fixed at prespecified reference values."
)
STATISTICAL_TARGET = "frequentist_confidence_interval"
CONFIDENCE_INTERVAL_METHOD = "percentile_parametric_bootstrap"
BOOTSTRAP_DATA_GENERATION = "marginal_AR1_process_plus_NB2_measurement"
BOOTSTRAP_REFIT = "country_state_space_MAP_full_refit_per_replicate"
VARIED_ESTIMATION_COMPONENTS = (
    "annual_AR1_latent_transmission_path",
    "NB2_surveillance_observations",
    "state_space_MAP_refit",
)
FIXED_REFERENCE_INPUTS = (
    "biological_parameters",
    "intervention_definitions",
    "AR1_hyperparameters",
    "NB2_dispersion",
)
# ``maximum_endpoint_mcse`` was an earlier outcome-scale diagnostic.  It is
# intentionally not an alias for the current scale-invariant tail-probability
# gate: accepting it silently would make the configured threshold look active
# when the runner was actually using its default.
REJECTED_SETTING_KEYS = ("maximum_endpoint_mcse",)
DEFAULT_REPLICATES = 1024
DEFAULT_MINIMUM_SUCCESSFUL_REPLICATES = 1000
DEFAULT_MINIMUM_SUCCESS_FRACTION = 0.95
DEFAULT_MAX_TAIL_PROBABILITY_MCSE = 0.005
DEFAULT_STABILITY_BLOCKS = 8
DEFAULT_CHUNK_SIZE = 100


@dataclass(frozen=True)
class BootstrapTask:
    country: str
    country_index: int
    replicate: int
    seed: int
    maxiter: int
    attempt: int = 1


@dataclass
class BootstrapResult:
    diagnostic: dict[str, Any]
    draws: list[dict[str, Any]]


def _bootstrap_settings(configs: dict[str, Any]) -> dict[str, Any]:
    settings = (
        configs["baseline"]
        .get("bayesian_uncertainty", {})
        .get("figure2c_parametric_bootstrap_confidence_interval", {})
    )
    rejected = sorted(set(settings).intersection(REJECTED_SETTING_KEYS))
    if rejected:
        raise ValueError(
            "Figure 2c bootstrap contains unconsumed legacy setting(s): "
            f"{rejected}. Use maximum_tail_probability_mcse for the active "
            "scale-invariant publication gate."
        )
    return settings


def _validate_figure2c_inputs(countries: list[str]) -> None:
    """Require the accepted calibrations and decision-frontier table."""

    validate_calibration_artifacts(
        countries,
        context="Figure 2c parametric bootstrap",
    )
    if not FRONTIER_PATH.is_file():
        raise FileNotFoundError(FRONTIER_PATH)


def _fixed_intervention_reference_values(
    configs: dict[str, Any],
) -> dict[str, float]:
    """Resolve inputs from formal programme definitions without uncertainty."""

    return fixed_programme_reference_inputs(configs)


def _assert_estimation_ci_refit_contract(
    result: Any,
    *,
    expected_process_years: np.ndarray,
    dispersion: float,
    rho: float,
    innovation_sd: float,
) -> None:
    """Fail if a refit expands beyond the prespecified estimation-CI scope."""

    process_years = tuple(int(year) for year in expected_process_years)
    expected_coordinate_names = (
        "log_beta_S",
        "log_reporting_multiplier",
        *(f"log_beta_process_{year}" for year in process_years),
    )
    coordinate_names = tuple(
        str(name) for name in getattr(result, "posterior_coordinate_names", ())
    )
    expected_dimension = len(expected_coordinate_names)
    coordinate_scope_ok = bool(process_years) and process_years == tuple(
        sorted(set(process_years))
    ) and coordinate_names == expected_coordinate_names
    posterior_dimension_ok = (
        np.asarray(getattr(result, "posterior_map_vector", None)).shape
        == (expected_dimension,)
        and np.asarray(getattr(result, "posterior_lower_bounds", None)).shape
        == (expected_dimension,)
        and np.asarray(getattr(result, "posterior_upper_bounds", None)).shape
        == (expected_dimension,)
        and np.asarray(getattr(result, "posterior_covariance", None)).shape
        == (expected_dimension, expected_dimension)
    )
    fixed_hyperparameters_ok = (
        np.isclose(
            float(getattr(result, "measurement_dispersion", np.nan)),
            float(dispersion),
        )
        and np.isclose(
            float(getattr(result, "process_ar1_rho", np.nan)), float(rho)
        )
        and np.isclose(
            float(getattr(result, "process_innovation_sd", np.nan)),
            float(innovation_sd),
        )
    )
    if (
        not coordinate_scope_ok
        or not posterior_dimension_ok
        or not fixed_hyperparameters_ok
    ):
        raise RuntimeError(
            "Figure 2c refit left the estimation-CI contract: it must contain "
            "exactly beta, reporting, and one latent AR(1) state for every "
            "expected surveillance year with matching posterior dimensions, "
            "while AR(1) hyperparameters and NB2 dispersion remain fixed."
        )


def _state_years(observed: pd.DataFrame) -> np.ndarray:
    starts = pd.to_datetime(observed["period_start"], errors="raise")
    ends = pd.to_datetime(observed["period_end"], errors="raise")
    first = int(starts.dt.year.min())
    last = int((ends - pd.Timedelta(nanoseconds=1)).dt.year.max())
    return np.arange(first, last + 1, dtype=int)


def _bounded_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    sd: float,
    bound: float,
) -> float:
    for _ in range(10_000):
        value = float(rng.normal(mean, sd))
        if -bound <= value <= bound:
            return value
    raise RuntimeError("Failed to draw a bounded AR(1) state")


def _draw_bounded_ar1_path(
    rng: np.random.Generator,
    years: np.ndarray,
    *,
    rho: float,
    innovation_sd: float,
    bound: float,
) -> np.ndarray:
    years = np.asarray(years, dtype=int)
    if years.ndim != 1 or len(years) < 1 or np.any(np.diff(years) <= 0):
        raise ValueError("AR(1) state years must be a non-empty increasing vector")
    starts = pd.Series(pd.to_datetime([f"{year:04d}-01-01" for year in years]))
    ends = pd.Series(pd.to_datetime([f"{year + 1:04d}-01-01" for year in years]))
    midpoints = starts + (ends - starts) / 2
    stationary_sd, transition_rho, transition_sd = _annual_ar1_transition_scales(
        midpoints,
        rho=float(rho),
        innovation_sd=float(innovation_sd),
    )
    states = np.empty(len(years), dtype=float)
    states[0] = _bounded_normal(
        rng, mean=0.0, sd=float(stationary_sd), bound=float(bound)
    )
    for index in range(1, len(years)):
        states[index] = _bounded_normal(
            rng,
            mean=float(transition_rho[index - 1] * states[index - 1]),
            sd=float(transition_sd[index - 1]),
            bound=float(bound),
        )
    return states


def _with_annual_process_path(
    config: dict[str, Any],
    years: np.ndarray,
    states: np.ndarray,
    *,
    rho: float,
    innovation_sd: float,
) -> dict[str, Any]:
    years = np.asarray(years, dtype=int)
    states = np.asarray(states, dtype=float)
    if years.shape != states.shape:
        raise ValueError("Annual process years and states must align")
    out = deepcopy(config)
    out["transmission"]["log_beta_time_variation"] = {
        "enabled": True,
        "interpretation": "parametric_bootstrap_AR1_process_draw",
        "ar1_rho": float(rho),
        "innovation_sd": float(innovation_sd),
        "periods": [
            {
                "start_date": f"{int(year):04d}-01-01",
                "end_date": f"{int(year):04d}-12-31",
                "log_multiplier": float(value),
                "state_origin": "AR1_parametric_bootstrap_draw",
            }
            for year, value in zip(years, states)
        ],
    }
    return out


def _allocate_group_count(
    frame: pd.DataFrame,
    positions: np.ndarray,
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(positions) == 1:
        return np.asarray([int(total)], dtype=int)
    if "interval_days" in frame.columns:
        weights = pd.to_numeric(
            frame.iloc[positions]["interval_days"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
    else:
        weights = np.ones(len(positions), dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(positions), dtype=float)
    weights = weights / float(weights.sum())
    return rng.multinomial(int(total), weights).astype(int)


def _synthetic_observed_frame(
    observed: pd.DataFrame,
    group_means: np.ndarray,
    *,
    dispersion: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    grouped = grouped_likelihood_observations(observed)
    means = np.asarray(group_means, dtype=float)
    if len(grouped) != len(means) or not np.isfinite(means).all():
        raise ValueError("Bootstrap likelihood means do not align with observations")
    r = max(float(dispersion), 1e-9)
    p = r / (r + np.clip(means, 1e-9, None))
    counts = rng.negative_binomial(r, p).astype(int)
    synthetic = observed.copy()
    synthetic["reported_cases"] = 0
    if "likelihood_group_id" not in observed.columns:
        synthetic["reported_cases"] = counts
        return synthetic, counts

    ordered_group_ids = grouped["likelihood_group_id"].astype(str).tolist()
    observed_group_ids = observed["likelihood_group_id"].astype(str).to_numpy()
    for group_id, count in zip(ordered_group_ids, counts):
        positions = np.flatnonzero(observed_group_ids == group_id)
        if not len(positions):
            raise RuntimeError(f"Missing likelihood group {group_id!r}")
        synthetic.iloc[
            positions, synthetic.columns.get_loc("reported_cases")
        ] = _allocate_group_count(synthetic, positions, int(count), rng)
    regenerated = grouped_likelihood_observations(synthetic)
    regenerated_counts = pd.to_numeric(
        regenerated["reported_cases"], errors="raise"
    ).to_numpy(dtype=int)
    if not np.array_equal(regenerated_counts, counts):
        raise RuntimeError("Synthetic reporting-interval allocation changed group totals")
    return synthetic, counts


def _sample_row_from_refit(
    country: str,
    replicate: int,
    calibrated: dict[str, Any],
    configs: dict[str, Any],
) -> pd.DataFrame:
    vaccine = calibrated["vaccine"]
    transmission = calibrated["transmission"]
    natural_history = calibrated["natural_history"]
    initial = calibrated["initial_conditions"]
    sample: dict[str, Any] = {
        "country": country,
        "chain": 1,
        "draw": int(replicate),
        "posterior_draw": int(replicate),
        "posterior_log_prob": np.nan,
        "beta_S": float(transmission["beta_S"]),
        "reporting_multiplier": float(calibrated.get("reporting_multiplier", 1.0)),
        "VE_sus": float(vaccine["VE_sus"]),
        "VE_inf": float(vaccine["VE_inf"]),
        "VE_dur": float(vaccine["VE_dur"]),
        "relative_infectiousness_asymptomatic": float(
            transmission["relative_infectiousness_asymptomatic"]
        ),
        "infectious_duration_symptomatic": float(
            natural_history["infectious_duration_symptomatic"]
        ),
        "infectious_duration_asymptomatic": float(
            natural_history["infectious_duration_asymptomatic"]
        ),
        "fitness_R": float(transmission["fitness_R"]),
        "resistance_prevalence": float(
            initial.get("initial_resistance_prevalence", 0.0)
        ),
        "reporting_trend_end_multiplier": 1.0,
    }
    periods = transmission.get("log_beta_time_variation", {}).get("periods", [])
    for period in periods:
        year = int(pd.Timestamp(period["start_date"]).year)
        sample[f"log_beta_process_{year}"] = float(period["log_multiplier"])
    fixed_intervention = _fixed_intervention_reference_values(configs)
    for key, value in fixed_intervention.items():
        sample[f"{INTERVENTION_UNCERTAINTY_PREFIX}{key}"] = float(value)
    return pd.DataFrame([sample])


def _task_seed(seed: int, country_index: int, replicate: int) -> int:
    state = np.random.SeedSequence(
        [int(seed), int(country_index), int(replicate)]
    ).generate_state(1)
    return int(state[0])


def _bootstrap_optimizer_max_nfev(
    process: dict[str, Any], maxiter: int
) -> int:
    """Scale only a bootstrap retry above the production evaluation floor."""

    return max(
        int(process.get("max_nfev", 40)),
        max(40, int(maxiter) * 8),
    )


def _run_bootstrap_task(task: BootstrapTask) -> BootstrapResult:
    diagnostic: dict[str, Any] = {
        "country": task.country,
        "bootstrap_replicate": int(task.replicate),
        "seed": int(task.seed),
        "attempt": int(task.attempt),
        "optimizer_maxiter": int(task.maxiter),
        "success": False,
        "error_type": "",
        "error_message": "",
    }
    try:
        configs = load_configs()
        calibration = configs["baseline"].get("calibration", {})
        process = calibration.get("process_model", {})
        dispersion = float(calibration.get("dispersion", 50.0))
        rho = float(process.get("ar1_rho", 0.5))
        innovation_sd = float(process.get("log_beta_innovation_sd", 0.6))
        process_bound = float(process.get("max_abs_log_beta_deviation", 2.5))
        artifact = load_calibrated_country_artifact(task.country)
        if artifact is None:
            raise RuntimeError(f"Missing accepted calibration for {task.country}")
        observed = calibration_observed_case_frame(task.country)
        years = _state_years(observed)
        rng = np.random.default_rng(int(task.seed))

        reference = calibration_runtime_config(
            deepcopy(artifact["config"]), observed
        )
        process_states = _draw_bounded_ar1_path(
            rng,
            years,
            rho=rho,
            innovation_sd=innovation_sd,
            bound=process_bound,
        )
        generating_config = _with_annual_process_path(
            reference,
            years,
            process_states,
            rho=rho,
            innovation_sd=innovation_sd,
        )
        generating_means = _calibration_predicted_means(
            generating_config, observed, task.country
        )
        synthetic_observed, generated_counts = _synthetic_observed_frame(
            observed,
            generating_means,
            dispersion=dispersion,
            rng=rng,
        )

        base = make_config(
            vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
            resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
            country_profile=task.country,
            load_calibration=False,
        )
        base = calibration_runtime_config(base, synthetic_observed)
        calibrated, result = state_space_map_calibration(
            base,
            task.country,
            synthetic_observed,
            dispersion=dispersion,
            maxiter=int(task.maxiter),
            # The production configuration deliberately fixes a 256-evaluation
            # floor. Override only the per-bootstrap runtime copy so adaptive
            # retry budgets are real without invalidating accepted calibration
            # artifacts or changing the production calibration definition.
            process_overrides={
                "max_nfev": _bootstrap_optimizer_max_nfev(
                    process, task.maxiter
                )
            },
        )
        _assert_estimation_ci_refit_contract(
            result,
            expected_process_years=years,
            dispersion=dispersion,
            rho=rho,
            innovation_sd=innovation_sd,
        )
        dimension = int(len(result.posterior_map_vector))
        if not bool(result.success) or not np.isfinite(float(result.fun)):
            raise RuntimeError(f"State-space refit failed: {result.message}")
        if int(result.posterior_rank) != dimension:
            raise RuntimeError(
                f"Rank-deficient state-space refit: {result.posterior_rank}/{dimension}"
            )
        maximum_condition = float(
            process.get("max_posterior_condition_number", 1e8)
        )
        if (
            not np.isfinite(float(result.posterior_condition_number))
            or float(result.posterior_condition_number) > maximum_condition
        ):
            raise RuntimeError(
                "Ill-conditioned state-space refit: "
                f"{result.posterior_condition_number} > {maximum_condition}"
            )
        production_end = pd.Timestamp(
            configs["baseline"]["calendar"]["analysis_end_date"]
        )
        calibrated = extend_annual_log_beta_conditional_mean(
            calibrated, through_year=int(production_end.year)
        )
        selected = _sample_row_from_refit(
            task.country, task.replicate, calibrated, configs
        )
        scenarios = _build_scenarios(
            configs,
            selected,
            strategies=FIGURE2C_STRATEGIES,
            analysis=STEM,
        )
        # Nested progress bars from the seven serial scenarios would otherwise
        # flood the production log from every outer worker.
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            summary = execute_scenario_summary_list(
                scenarios,
                stem=f"{STEM}_{task.country}_{task.replicate}",
                n_jobs=1,
            )
        paired = _paired_draws(summary).rename(
            columns={"posterior_draw": "bootstrap_replicate"}
        )
        if len(paired) != len(INTERVENTION_STRATEGIES):
            raise RuntimeError(
                f"Expected {len(INTERVENTION_STRATEGIES)} paired strategies, "
                f"received {len(paired)}"
            )
        keep = [
            "country",
            "bootstrap_replicate",
            "strategy",
            "strategy_label",
            "current_rate",
            "intervention_rate",
            "current_total_cases",
            "intervention_total_cases",
            PRIMARY_REDUCTION,
        ]
        draws = paired.loc[:, keep].to_dict(orient="records")
        diagnostic.update(
            {
                "success": True,
                "fit_score": float(result.fun),
                "optimizer_nfev": int(result.process_nfev),
                "posterior_dimension": dimension,
                "posterior_rank": int(result.posterior_rank),
                "posterior_condition_number": float(
                    result.posterior_condition_number
                ),
                "generated_total_cases": int(np.sum(generated_counts)),
                "generated_min_group_cases": int(np.min(generated_counts)),
                "generated_max_group_cases": int(np.max(generated_counts)),
                "maximum_abs_generated_process_state": float(
                    np.max(np.abs(process_states))
                ),
            }
        )
        return BootstrapResult(diagnostic=diagnostic, draws=draws)
    except Exception as exc:  # A failed replicate is retained and audited.
        diagnostic["error_type"] = type(exc).__name__
        diagnostic["error_message"] = str(exc)[:1000]
        return BootstrapResult(diagnostic=diagnostic, draws=[])


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_checkpoint() -> tuple[pd.DataFrame, pd.DataFrame]:
    draws = (
        pd.read_parquet(CHECKPOINT_DRAW_PATH)
        if CHECKPOINT_DRAW_PATH.exists()
        else pd.DataFrame()
    )
    fits = (
        pd.read_parquet(CHECKPOINT_FIT_PATH)
        if CHECKPOINT_FIT_PATH.exists()
        else pd.DataFrame()
    )
    return draws, fits


def _checkpoint_bundle_paths() -> tuple[Path, ...]:
    """Return only the active checkpoint bundle and its exact atomic temporaries."""

    return (
        CHECKPOINT_METADATA_PATH,
        CHECKPOINT_DRAW_PATH,
        CHECKPOINT_FIT_PATH,
        CHECKPOINT_METADATA_PATH.with_suffix(".json.tmp"),
        CHECKPOINT_DRAW_PATH.with_name(CHECKPOINT_DRAW_PATH.name + ".tmp.parquet"),
        CHECKPOINT_FIT_PATH.with_name(CHECKPOINT_FIT_PATH.name + ".tmp.parquet"),
    )


def _checkpoint_archive_directory(
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = CHECKPOINT_ARCHIVE_ROOT / f"checkpoint_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix:02d}")
        suffix += 1
    return candidate


def _archive_incompatible_checkpoint_bundle(
    *,
    reason: str,
) -> Path:
    """Move an incompatible active checkpoint aside without deleting any bytes.

    Only the exact active bundle is considered. Historical pilots, prior manual
    snapshots, and final Figure 2c outputs are deliberately outside this list.
    Every move is rolled back if the bundle cannot be archived completely.
    """

    sources = [path for path in _checkpoint_bundle_paths() if path.exists()]
    if not sources:
        raise RuntimeError("Checkpoint archival was requested but no active files exist")
    archive_dir = _checkpoint_archive_directory()
    records = [
        {
            "original_path": str(path),
            "archive_name": path.name,
            "size_bytes": int(path.stat().st_size),
        }
        for path in sources
    ]
    archive_dir.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    try:
        for source in sources:
            destination = archive_dir / source.name
            if destination.exists():
                raise FileExistsError(f"Checkpoint archive target exists: {destination}")
            source.replace(destination)
            moved.append((source, destination))

        manifest = {
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "files": records,
        }
        temporary_manifest = archive_dir / "manifest.json.tmp"
        final_manifest = archive_dir / "manifest.json"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_manifest.replace(final_manifest)
    except Exception as exc:
        rollback_errors: list[str] = []
        for source, destination in reversed(moved):
            try:
                if destination.exists():
                    destination.replace(source)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O
                rollback_errors.append(f"{destination} -> {source}: {rollback_exc}")
        for residue in (archive_dir / "manifest.json.tmp", archive_dir / "manifest.json"):
            if residue.exists():
                residue.unlink()
        try:
            archive_dir.rmdir()
        except OSError:
            pass
        detail = f"; rollback failures={rollback_errors}" if rollback_errors else ""
        raise RuntimeError(f"Could not archive incompatible Figure 2c checkpoint{detail}") from exc
    return archive_dir


def _validate_or_create_checkpoint_metadata(
    *,
    countries: list[str],
    replicates: int,
    seed: int,
    maxiter: int,
) -> Path | None:
    active_paths = _checkpoint_bundle_paths()
    temporary_paths = active_paths[3:]
    archive_reason = ""
    metadata: dict[str, Any] | None = None
    if CHECKPOINT_METADATA_PATH.exists():
        try:
            loaded = json.loads(CHECKPOINT_METADATA_PATH.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("checkpoint metadata must be a JSON object")
            metadata = loaded
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            archive_reason = f"checkpoint metadata are unreadable: {type(exc).__name__}: {exc}"
        else:
            expected = {
                "countries": countries,
                "replicates": int(replicates),
                "seed": int(seed),
                "maxiter": int(maxiter),
            }
            mismatches = [
                key for key, value in expected.items() if metadata.get(key) != value
            ]
            if mismatches:
                archive_reason = (
                    "checkpoint design fields do not match the requested run: "
                    + ", ".join(mismatches)
                )
            elif any(path.exists() for path in temporary_paths):
                archive_reason = "atomic checkpoint temporary files remain from an interrupted write"
            elif CHECKPOINT_DRAW_PATH.exists() != CHECKPOINT_FIT_PATH.exists():
                archive_reason = "checkpoint draw and fit files are orphaned"
            else:
                return None
    elif any(path.exists() for path in active_paths[1:]):
        archive_reason = "checkpoint data exist without compatible metadata"

    archive_dir: Path | None = None
    if archive_reason:
        archive_dir = _archive_incompatible_checkpoint_bundle(
            reason=archive_reason,
        )
    CHECKPOINT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_METADATA_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "countries": countries,
                "replicates": int(replicates),
                "seed": int(seed),
                "maxiter": int(maxiter),
            },
            indent=2,
            sort_keys=True,
        )
    )
    temporary.replace(CHECKPOINT_METADATA_PATH)
    return archive_dir


def _merge_results(
    checkpoint_draws: pd.DataFrame,
    checkpoint_fits: pd.DataFrame,
    results: list[BootstrapResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = pd.DataFrame([result.diagnostic for result in results])
    draw_rows = pd.DataFrame(
        [row for result in results for row in result.draws]
    )
    keys = ["country", "bootstrap_replicate"]
    if not checkpoint_fits.empty and not fit_rows.empty:
        incoming = set(map(tuple, fit_rows[keys].to_numpy()))
        keep = ~checkpoint_fits[keys].apply(tuple, axis=1).isin(incoming)
        checkpoint_fits = checkpoint_fits.loc[keep]
    fits = pd.concat([checkpoint_fits, fit_rows], ignore_index=True)
    if not checkpoint_draws.empty and not draw_rows.empty:
        incoming = set(map(tuple, draw_rows[keys].drop_duplicates().to_numpy()))
        keep = ~checkpoint_draws[keys].apply(tuple, axis=1).isin(incoming)
        checkpoint_draws = checkpoint_draws.loc[keep]
    draws = pd.concat([checkpoint_draws, draw_rows], ignore_index=True)
    return draws, fits


def _run_tasks_with_streaming_checkpoints(
    tasks: list[BootstrapTask],
    *,
    workers: int,
    checkpoint_size: int,
    total: int,
    checkpoint_draws: pd.DataFrame,
    checkpoint_fits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep workers busy while atomically saving every completed result block."""

    if not tasks:
        return checkpoint_draws, checkpoint_fits
    result_stream = Parallel(
        n_jobs=workers,
        backend="loky",
        inner_max_num_threads=1,
        return_as="generator_unordered",
        batch_size=1,
    )(delayed(_run_bootstrap_task)(task) for task in tasks)
    buffered: list[BootstrapResult] = []
    for result in result_stream:
        buffered.append(result)
        if len(buffered) < checkpoint_size:
            continue
        checkpoint_draws, checkpoint_fits = _merge_results(
            checkpoint_draws, checkpoint_fits, buffered
        )
        _atomic_write_parquet(checkpoint_draws, CHECKPOINT_DRAW_PATH)
        _atomic_write_parquet(checkpoint_fits, CHECKPOINT_FIT_PATH)
        success = int(checkpoint_fits["success"].astype(bool).sum())
        print(
            f"checkpoint completed={len(checkpoint_fits)}/{total}, "
            f"successful={success}, failed={len(checkpoint_fits) - success}",
            flush=True,
        )
        buffered = []
    if buffered:
        checkpoint_draws, checkpoint_fits = _merge_results(
            checkpoint_draws, checkpoint_fits, buffered
        )
        _atomic_write_parquet(checkpoint_draws, CHECKPOINT_DRAW_PATH)
        _atomic_write_parquet(checkpoint_fits, CHECKPOINT_FIT_PATH)
        success = int(checkpoint_fits["success"].astype(bool).sum())
        print(
            f"checkpoint completed={len(checkpoint_fits)}/{total}, "
            f"successful={success}, failed={len(checkpoint_fits) - success}",
            flush=True,
        )
    return checkpoint_draws, checkpoint_fits


def _delete_block_quantile_mcse(
    values: np.ndarray,
    replicate_ids: np.ndarray,
    quantile: float,
    *,
    blocks: int,
) -> float:
    values = np.asarray(values, dtype=float)
    replicate_ids = np.asarray(replicate_ids, dtype=int)
    estimates: list[float] = []
    for block in range(int(blocks)):
        retained = replicate_ids % int(blocks) != block
        if retained.sum() < 20:
            continue
        estimates.append(float(np.quantile(values[retained], quantile)))
    if len(estimates) < 2:
        return float("nan")
    estimates_array = np.asarray(estimates, dtype=float)
    return float(
        np.sqrt(
            (len(estimates_array) - 1.0)
            / len(estimates_array)
            * np.sum((estimates_array - estimates_array.mean()) ** 2)
        )
    )


def _tail_probability_mcse(replicates: int, quantile: float = 0.025) -> float:
    """Binomial Monte Carlo error of the empirical CDF at a tail quantile.

    Unlike outcome-scale quantile error, this diagnostic is invariant to the
    units and shape of the estimand.  It therefore remains meaningful for the
    multimodal bootstrap distributions that can arise from epidemic dynamics.
    """

    if int(replicates) < 1 or not 0.0 < float(quantile) < 1.0:
        return float("nan")
    return float(
        np.sqrt(float(quantile) * (1.0 - float(quantile)) / int(replicates))
    )


def _frontier_point_estimates(countries: list[str]) -> pd.DataFrame:
    if not FRONTIER_PATH.exists():
        raise FileNotFoundError(FRONTIER_PATH)
    frontier = pd.read_csv(FRONTIER_PATH)
    selected = frontier.loc[
        frontier["optimization_constraint"].astype(str).eq("program_only")
        & frontier["country"]
        .astype(str)
        .str.replace(" ", "_", regex=False)
        .isin(countries)
        & frontier["strategy"].astype(str).isin(INTERVENTION_STRATEGIES)
    ].copy()
    selected["country"] = (
        selected["country"].astype(str).str.replace(" ", "_", regex=False)
    )
    selected["strategy"] = selected["strategy"].astype(str)
    if (
        len(selected) != len(countries) * len(INTERVENTION_STRATEGIES)
        or selected[["country", "strategy"]].duplicated().any()
    ):
        raise ValueError("Figure 2 frontier does not contain one programme cell")
    return selected.loc[
        :, ["country", "strategy", "primary_case_reduction"]
    ].rename(columns={"primary_case_reduction": "reduction_point_estimate"})


def _summarise_intervals(
    draws: pd.DataFrame,
    *,
    countries: list[str],
    stability_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = _frontier_point_estimates(countries)
    rows: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for (country, strategy), group in draws.groupby(
        ["country", "strategy"], sort=True
    ):
        values = pd.to_numeric(
            group[PRIMARY_REDUCTION], errors="raise"
        ).to_numpy(dtype=float)
        replicate_ids = pd.to_numeric(
            group["bootstrap_replicate"], errors="raise"
        ).to_numpy(dtype=int)
        q025 = float(np.quantile(values, 0.025))
        q975 = float(np.quantile(values, 0.975))
        q025_mcse = _delete_block_quantile_mcse(
            values, replicate_ids, 0.025, blocks=stability_blocks
        )
        q975_mcse = _delete_block_quantile_mcse(
            values, replicate_ids, 0.975, blocks=stability_blocks
        )
        tail_probability_mcse = _tail_probability_mcse(len(values))
        first = group.iloc[0]
        rows.append(
            {
                "country": country,
                "strategy": strategy,
                "scenario_key": strategy,
                "scenario_label": first["strategy_label"],
                "outcome": "child_adolescent_cases",
                "reduction_median": float(np.quantile(values, 0.50)),
                "reduction_q025": q025,
                "reduction_q25": float(np.quantile(values, 0.25)),
                "reduction_q75": float(np.quantile(values, 0.75)),
                "reduction_q975": q975,
                "current_rate_median": float(group["current_rate"].median()),
                "current_rate_q025": float(group["current_rate"].quantile(0.025)),
                "current_rate_q975": float(group["current_rate"].quantile(0.975)),
                "intervention_rate_median": float(
                    group["intervention_rate"].median()
                ),
                "intervention_rate_q025": float(
                    group["intervention_rate"].quantile(0.025)
                ),
                "intervention_rate_q975": float(
                    group["intervention_rate"].quantile(0.975)
                ),
                "bootstrap_replicates": int(group["bootstrap_replicate"].nunique()),
                "uncertainty_draws": int(group["bootstrap_replicate"].nunique()),
                "interval_type": INTERVAL_TYPE,
                "interval_basis": INTERVAL_BASIS,
                "confidence_interval_method": CONFIDENCE_INTERVAL_METHOD,
                "reduction_q025_mcse": q025_mcse,
                "reduction_q975_mcse": q975_mcse,
                "tail_probability_mcse": tail_probability_mcse,
            }
        )
        stability.append(
            {
                "country": country,
                "strategy": strategy,
                "bootstrap_replicates": int(group["bootstrap_replicate"].nunique()),
                "reduction_q025": q025,
                "reduction_q975": q975,
                "reduction_q95_width": q975 - q025,
                "reduction_q025_mcse": q025_mcse,
                "reduction_q975_mcse": q975_mcse,
                "maximum_endpoint_mcse": max(q025_mcse, q975_mcse),
                "tail_probability_mcse": tail_probability_mcse,
                "expected_replicates_per_2.5pct_tail": 0.025 * len(values),
            }
        )
    intervals = pd.DataFrame(rows).merge(
        points, on=["country", "strategy"], how="left", validate="one_to_one"
    )
    intervals["bootstrap_bias"] = (
        intervals["reduction_median"] - intervals["reduction_point_estimate"]
    )
    return (
        intervals.sort_values(["country", "strategy"]).reset_index(drop=True),
        pd.DataFrame(stability)
        .sort_values(["country", "strategy"])
        .reset_index(drop=True),
    )


def run_parametric_bootstrap(
    *,
    countries: list[str] | None = None,
    replicates: int | None = None,
    minimum_successful_replicates: int | None = None,
    minimum_success_fraction: float | None = None,
    maximum_tail_probability_mcse: float | None = None,
    stability_blocks: int | None = None,
    n_jobs: int | None = None,
    chunk_size: int | None = None,
    seed: int | None = None,
    maxiter: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    fixed_programme_reference = _fixed_intervention_reference_values(configs)
    settings = _bootstrap_settings(configs)
    replicates = int(settings.get("replicates_per_country", DEFAULT_REPLICATES)) if replicates is None else int(replicates)
    minimum_successful_replicates = (
        int(settings.get("minimum_successful_replicates", DEFAULT_MINIMUM_SUCCESSFUL_REPLICATES))
        if minimum_successful_replicates is None
        else int(minimum_successful_replicates)
    )
    minimum_success_fraction = (
        float(settings.get("minimum_success_fraction", DEFAULT_MINIMUM_SUCCESS_FRACTION))
        if minimum_success_fraction is None
        else float(minimum_success_fraction)
    )
    maximum_tail_probability_mcse = (
        float(
            settings.get(
                "maximum_tail_probability_mcse",
                DEFAULT_MAX_TAIL_PROBABILITY_MCSE,
            )
        )
        if maximum_tail_probability_mcse is None
        else float(maximum_tail_probability_mcse)
    )
    stability_blocks = (
        int(settings.get("stability_blocks", DEFAULT_STABILITY_BLOCKS))
        if stability_blocks is None
        else int(stability_blocks)
    )
    n_jobs = int(settings.get("parallel_workers", 100)) if n_jobs is None else int(n_jobs)
    chunk_size = (
        int(settings.get("checkpoint_chunk_size", DEFAULT_CHUNK_SIZE))
        if chunk_size is None
        else int(chunk_size)
    )
    seed = int(settings.get("seed", 20260716)) if seed is None else int(seed)
    publication = publication_country_names(configs)
    resolved = publication if countries is None else list(countries)
    outside = sorted(set(resolved).difference(publication))
    if outside or len(set(resolved)) != len(resolved):
        raise ValueError(f"Invalid bootstrap countries: {outside or resolved}")
    _validate_figure2c_inputs(resolved)
    if int(replicates) < 100:
        raise ValueError("Parametric bootstrap requires at least 100 replicates")
    if int(minimum_successful_replicates) > int(replicates):
        raise ValueError("Minimum successful replicates cannot exceed requested")
    if not 0.0 < float(minimum_success_fraction) <= 1.0:
        raise ValueError("Minimum success fraction must be in (0, 1]")
    calibration = configs["baseline"].get("calibration", {})
    resolved_maxiter = int(maxiter or calibration.get("maxiter", 30))
    retry_maxiters = sorted(
        {
            int(value)
            for value in settings.get("retry_maxiter_schedule", [64, 128])
            if int(value) > resolved_maxiter
        }
    )
    if not retry_maxiters:
        raise ValueError(
            "Figure 2 bootstrap retry_maxiter_schedule must contain a value "
            "above the first-pass maxiter"
        )
    workers = min(max(1, int(n_jobs)), available_cpus())
    chunk_size = max(workers, int(chunk_size))
    archived_checkpoint = _validate_or_create_checkpoint_metadata(
        countries=resolved,
        replicates=int(replicates),
        seed=int(seed),
        maxiter=resolved_maxiter,
    )
    if archived_checkpoint is not None:
        print(
            "Archived an incompatible Figure 2c checkpoint bundle at "
            f"{archived_checkpoint}",
            flush=True,
        )
    checkpoint_draws, checkpoint_fits = _load_checkpoint()
    checkpoint_keys = (
        set(
            map(
                tuple,
                checkpoint_fits[["country", "bootstrap_replicate"]].to_numpy(),
            )
        )
        if not checkpoint_fits.empty
        else set()
    )
    tasks = [
        BootstrapTask(
            country=country,
            country_index=country_index,
            replicate=replicate,
            seed=_task_seed(seed, country_index, replicate),
            maxiter=resolved_maxiter,
            attempt=1,
        )
        for country_index, country in enumerate(resolved, start=1)
        for replicate in range(1, int(replicates) + 1)
        if (country, replicate) not in checkpoint_keys
    ]
    configure_worker_thread_limits()
    total = len(resolved) * int(replicates)
    print(
        f"Figure 2 parametric bootstrap: completed={len(checkpoint_keys)}/{total}, "
        f"pending={len(tasks)}, workers={workers}",
        flush=True,
    )
    checkpoint_draws, checkpoint_fits = _run_tasks_with_streaming_checkpoints(
        tasks,
        workers=workers,
        checkpoint_size=chunk_size,
        total=total,
        checkpoint_draws=checkpoint_draws,
        checkpoint_fits=checkpoint_fits,
    )

    # A numerical optimizer failure should not silently reduce bootstrap size.
    # Retry only failures, using the identical synthetic-data seed and genuinely
    # increasing evaluation budgets. The replacement merge keeps one row per
    # country-replicate and never discards a successful draw.
    for retry_maxiter in retry_maxiters:
        failed = checkpoint_fits.loc[
            ~checkpoint_fits["success"].astype(bool)
        ].copy()
        if failed.empty:
            break
        previous_budget = pd.to_numeric(
            failed.get("optimizer_maxiter", pd.Series(index=failed.index)),
            errors="coerce",
        ).fillna(0)
        failed = failed.loc[previous_budget.lt(int(retry_maxiter))]
        if failed.empty:
            continue
        country_indices = {
            country: index for index, country in enumerate(resolved, start=1)
        }
        retry_tasks = [
            BootstrapTask(
                country=str(row.country),
                country_index=country_indices[str(row.country)],
                replicate=int(row.bootstrap_replicate),
                seed=int(row.seed),
                maxiter=int(retry_maxiter),
                attempt=int(row.attempt) + 1,
            )
            for row in failed.itertuples(index=False)
        ]
        print(
            f"Retrying {len(retry_tasks)} failed bootstrap refits with "
            f"maxiter={retry_maxiter}",
            flush=True,
        )
        checkpoint_draws, checkpoint_fits = _run_tasks_with_streaming_checkpoints(
            retry_tasks,
            workers=workers,
            checkpoint_size=chunk_size,
            total=total,
            checkpoint_draws=checkpoint_draws,
            checkpoint_fits=checkpoint_fits,
        )

    fits = checkpoint_fits.sort_values(
        ["country", "bootstrap_replicate"]
    ).reset_index(drop=True)
    draws = checkpoint_draws.sort_values(
        ["country", "strategy", "bootstrap_replicate"]
    ).reset_index(drop=True)
    success_by_country = (
        fits.assign(success=fits["success"].astype(bool))
        .groupby("country")["success"]
        .agg(["sum", "count"])
    )
    insufficient = success_by_country.loc[
        success_by_country["sum"].lt(int(minimum_successful_replicates))
        | (success_by_country["sum"] / success_by_country["count"]).lt(
            float(minimum_success_fraction)
        )
    ]
    if not insufficient.empty:
        raise RuntimeError(
            "Parametric bootstrap did not meet the country success gate: "
            + insufficient.to_dict(orient="index").__repr__()
        )
    intervals, stability = _summarise_intervals(
        draws, countries=resolved, stability_blocks=int(stability_blocks)
    )
    if not np.isfinite(stability["tail_probability_mcse"]).all():
        raise RuntimeError("Bootstrap tail-probability MCSE could not be estimated")
    if float(stability["tail_probability_mcse"].max()) > float(
        maximum_tail_probability_mcse
    ):
        raise RuntimeError(
            "Bootstrap tail-probability MCSE exceeds the publication gate: "
            f"{stability['tail_probability_mcse'].max():.6f} > "
            f"{maximum_tail_probability_mcse:.6f}"
        )
    write_dataframe(draws, DRAW_PATH)
    write_dataframe(intervals, INTERVAL_PATH)
    write_dataframe(fits, FIT_DIAGNOSTIC_PATH)
    write_dataframe(stability, STABILITY_PATH)
    metadata = current_run_metadata(
        STEM,
        row_counts={
            "paired_bootstrap_draws": int(len(draws)),
            "confidence_interval_rows": int(len(intervals)),
            "fit_diagnostics": int(len(fits)),
            "interval_stability_rows": int(len(stability)),
        },
    ) | {
        "analysis_role": ANALYSIS_ROLE,
        "publication_path": PUBLICATION_PATH,
        "figure2c_interval_source": FIGURE2C_INTERVAL_SOURCE,
        "statistical_target": STATISTICAL_TARGET,
        "interval_type": INTERVAL_TYPE,
        "interval_basis": INTERVAL_BASIS,
        "confidence_interval_method": CONFIDENCE_INTERVAL_METHOD,
        "bootstrap_data_generation": BOOTSTRAP_DATA_GENERATION,
        "bootstrap_refit": BOOTSTRAP_REFIT,
        "varied_estimation_components": list(VARIED_ESTIMATION_COMPONENTS),
        "paired_scenario_contrast": True,
        "future_observation_prediction_interval": False,
        "posterior_credible_interval": False,
        "fixed_reference_inputs": list(FIXED_REFERENCE_INPUTS),
        "fixed_programme_reference_source": FIXED_PROGRAMME_REFERENCE_SOURCE,
        "fixed_programme_reference_inputs": fixed_programme_reference,
        "countries": resolved,
        "strategies": list(INTERVENTION_STRATEGIES),
        "replicates_requested_per_country": int(replicates),
        "successful_replicates_by_country": {
            country: int(row["sum"])
            for country, row in success_by_country.iterrows()
        },
        "minimum_successful_replicates": int(minimum_successful_replicates),
        "minimum_success_fraction": float(minimum_success_fraction),
        "stability_blocks": int(stability_blocks),
        "maximum_tail_probability_mcse_threshold": float(
            maximum_tail_probability_mcse
        ),
        "observed_maximum_tail_probability_mcse": float(
            stability["tail_probability_mcse"].max()
        ),
        # Retained as a transparent outcome-scale sensitivity diagnostic. It
        # is not a publication gate because it is not scale invariant and can
        # be large at a genuine mixture boundary.
        "observed_maximum_endpoint_mcse": float(
            stability["maximum_endpoint_mcse"].max()
        ),
        "seed": int(seed),
        "n_jobs": int(workers),
        "maxiter": int(resolved_maxiter),
        "retry_maxiter_schedule": retry_maxiters,
        "checkpoint_archive_path": (
            str(archived_checkpoint) if archived_checkpoint is not None else None
        ),
    }
    write_run_metadata(STEM, metadata)
    return draws, intervals, fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument(
        "--minimum-successful-replicates",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--minimum-success-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--maximum-tail-probability-mcse",
        type=float,
        default=None,
    )
    parser.add_argument("--stability-blocks", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    args = parser.parse_args()
    run_parametric_bootstrap(
        countries=args.countries.split(",") if args.countries else None,
        replicates=args.replicates,
        minimum_successful_replicates=args.minimum_successful_replicates,
        minimum_success_fraction=args.minimum_success_fraction,
        maximum_tail_probability_mcse=args.maximum_tail_probability_mcse,
        stability_blocks=args.stability_blocks,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        seed=args.seed,
        maxiter=args.maxiter,
    )


if __name__ == "__main__":
    main()
