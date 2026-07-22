from __future__ import annotations

"""Conditional current-practice full-refit bootstrap intervals for Figure 1b.

Each replicate holds the accepted fitted annual latent transmission path as the
generating trajectory, regenerates grouped NB2 surveillance observations, fully
refits the country state-space model, and runs current practice once. The same
fitted replicate supplies reported-case, symptomatic-case, and infection
indices. These are conditional frequentist confidence intervals, not posterior
credible or future-observation intervals.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from src_python.calibration.calibrate_baseline import (
    _calibration_predicted_means,
    calibration_observed_case_frame,
    calibration_runtime_config,
    extend_annual_log_beta_conditional_mean,
    state_space_map_calibration,
)
from src_python.simulation.common import (
    config_fingerprint,
    current_run_metadata,
    execute_scenario_summary_list,
    file_sha256,
    load_calibrated_country_artifact,
    load_configs,
    make_config,
    publication_country_names,
    source_code_fingerprint,
    validated_calibration_artifact_path_hashes,
    write_run_metadata,
)
from src_python.simulation.programme_uncertainty_helpers import (
    build_programme_scenarios as _build_scenarios,
)
from src_python.simulation.run_figure2c_parametric_bootstrap import (
    BootstrapResult,
    BootstrapTask,
    _atomic_write_parquet,
    _bootstrap_optimizer_max_nfev,
    _delete_block_quantile_mcse,
    _merge_results,
    _sample_row_from_refit,
    _state_years,
    _synthetic_observed_frame,
    _tail_probability_mcse,
    _task_seed,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import available_cpus, configure_worker_thread_limits


STEM = "figure1b_current_practice_conditional_parametric_bootstrap"
DRAW_PATH = project_path(
    "outputs", "tables", "figure1b_current_practice_conditional_bootstrap_draws.csv"
)
INTERVAL_PATH = project_path(
    "outputs", "summaries", "figure1b_current_practice_conditional_confidence_intervals.csv"
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

OUTCOME_COLUMNS = {
    "Reports": "annualized_child_adolescent_reported_cases_per_100k",
    "Symptomatic": "annualized_child_adolescent_cases_per_100k",
    "Infections": "annualized_child_adolescent_infections_per_100k",
}
DRAW_COLUMNS = {
    "Reports": "reports_rate_per_100k",
    "Symptomatic": "symptomatic_rate_per_100k",
    "Infections": "infections_rate_per_100k",
}
INTERVAL_TYPE = "95% parametric-bootstrap confidence interval"
INTERVAL_METHOD = "percentile_parametric_bootstrap"
INTERVAL_BASIS = (
    "Current-practice-only conditional parametric bootstrap. The accepted "
    "fitted annual latent transmission path is held as the generating "
    "trajectory, surveillance counts are regenerated from the fitted NB2 "
    "observation model, and the country state-space model is fully refitted "
    "before the same fitted replicate is propagated once through current "
    "practice. Biological inputs, AR(1) hyperparameters, and NB2 dispersion "
    "remain fixed at prespecified reference values."
)
DEFAULT_REPLICATES = 1024
DEFAULT_MINIMUM_SUCCESSFUL_REPLICATES = 1000
DEFAULT_MINIMUM_SUCCESS_FRACTION = 0.95
DEFAULT_MAX_TAIL_PROBABILITY_MCSE = 0.005
DEFAULT_STABILITY_BLOCKS = 8
DEFAULT_CHUNK_SIZE = 50


def _fitted_process_state_values(
    config: dict[str, Any], observed: pd.DataFrame
) -> np.ndarray:
    """Validate and return the fitted latent states covering the data window."""

    variation = config.get("transmission", {}).get("log_beta_time_variation", {})
    periods = variation.get("periods", []) if isinstance(variation, dict) else []
    if not bool(variation.get("enabled", False)) or not periods:
        raise RuntimeError(
            "Conditional Figure 1b bootstrap requires an accepted fitted latent path"
        )
    fitted_by_year: dict[int, float] = {}
    for period in periods:
        year = int(pd.Timestamp(period["start_date"]).year)
        value = float(period["log_multiplier"])
        if year in fitted_by_year or not np.isfinite(value):
            raise RuntimeError("Accepted fitted latent path is invalid or duplicated")
        fitted_by_year[year] = value
    required_years = _state_years(observed)
    missing_years = sorted(set(map(int, required_years)).difference(fitted_by_year))
    if missing_years:
        raise RuntimeError(
            "Accepted fitted latent path does not cover observation years: "
            + repr(missing_years)
        )
    return np.asarray([fitted_by_year[int(year)] for year in required_years], dtype=float)


def _run_current_practice_task(task: BootstrapTask) -> BootstrapResult:
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
        artifact = load_calibrated_country_artifact(task.country)
        if artifact is None:
            raise RuntimeError(f"Missing accepted calibration for {task.country}")
        observed = calibration_observed_case_frame(task.country)
        rng = np.random.default_rng(int(task.seed))

        reference = calibration_runtime_config(deepcopy(artifact["config"]), observed)
        process_states = _fitted_process_state_values(reference, observed)
        generating_means = _calibration_predicted_means(
            reference, observed, task.country
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
            process_overrides={
                "max_nfev": _bootstrap_optimizer_max_nfev(process, task.maxiter)
            },
        )
        dimension = int(len(result.posterior_map_vector))
        if not bool(result.success) or not np.isfinite(float(result.fun)):
            raise RuntimeError(f"State-space refit failed: {result.message}")
        if int(result.posterior_rank) != dimension:
            raise RuntimeError(
                f"Rank-deficient state-space refit: {result.posterior_rank}/{dimension}"
            )
        maximum_condition = float(process.get("max_posterior_condition_number", 1e8))
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
            strategies=("current",),
            analysis=STEM,
        )
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            summary = execute_scenario_summary_list(
                scenarios,
                stem=f"{STEM}_{task.country}_{task.replicate}",
                n_jobs=1,
            )
        current = summary.loc[summary["strategy"].astype(str).eq("current")].copy()
        if len(current) != 1:
            raise RuntimeError(
                f"Expected one current-practice row, received {len(current)}"
            )
        row = current.iloc[0]
        draw: dict[str, Any] = {
            "country": task.country,
            "bootstrap_replicate": int(task.replicate),
        }
        for outcome, source_column in OUTCOME_COLUMNS.items():
            value = float(row[source_column])
            if not np.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    f"Invalid {outcome} current-practice index: {value}"
                )
            draw[DRAW_COLUMNS[outcome]] = value

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
                "generating_process_path": "accepted_fitted_latent_AR1_path",
            }
        )
        return BootstrapResult(diagnostic=diagnostic, draws=[draw])
    except Exception as exc:
        diagnostic["error_type"] = type(exc).__name__
        diagnostic["error_message"] = str(exc)[:1000]
        return BootstrapResult(diagnostic=diagnostic, draws=[])


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


def _checkpoint_fingerprint(
    *, countries: list[str], replicates: int, seed: int, maxiter: int
) -> str:
    payload = {
        "schema": 1,
        "config_hash": config_fingerprint(),
        "source_code_hash": source_code_fingerprint(),
        "countries": countries,
        "replicates": int(replicates),
        "seed": int(seed),
        "maxiter": int(maxiter),
        "calibration_artifact_hashes": {
            country: file_sha256(
                project_path(
                    "outputs",
                    "calibrations",
                    f"{country.replace(' ', '_')}_calibrated_config.yaml",
                )
            )
            for country in countries
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_or_create_checkpoint_metadata(
    *, fingerprint: str, countries: list[str], replicates: int, seed: int, maxiter: int
) -> None:
    if CHECKPOINT_METADATA_PATH.exists():
        metadata = json.loads(CHECKPOINT_METADATA_PATH.read_text())
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Figure 1b checkpoint fingerprint does not match the current run"
            )
        return
    CHECKPOINT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_METADATA_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
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


def _run_tasks_with_checkpoints(
    tasks: list[BootstrapTask],
    *,
    workers: int,
    checkpoint_size: int,
    total: int,
    checkpoint_draws: pd.DataFrame,
    checkpoint_fits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tasks:
        return checkpoint_draws, checkpoint_fits
    result_stream = Parallel(
        n_jobs=workers,
        backend="loky",
        inner_max_num_threads=1,
        return_as="generator_unordered",
        batch_size=1,
    )(delayed(_run_current_practice_task)(task) for task in tasks)
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
    return checkpoint_draws, checkpoint_fits


def _summarise_intervals(
    draws: pd.DataFrame, *, countries: list[str], stability_blocks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for country in countries:
        group = draws.loc[draws["country"].astype(str).eq(country)].copy()
        replicate_ids = pd.to_numeric(
            group["bootstrap_replicate"], errors="raise"
        ).to_numpy(dtype=int)
        for outcome, draw_column in DRAW_COLUMNS.items():
            values = pd.to_numeric(group[draw_column], errors="raise").to_numpy(
                dtype=float
            )
            q025 = float(np.quantile(values, 0.025))
            q975 = float(np.quantile(values, 0.975))
            q025_mcse = _delete_block_quantile_mcse(
                values, replicate_ids, 0.025, blocks=stability_blocks
            )
            q975_mcse = _delete_block_quantile_mcse(
                values, replicate_ids, 0.975, blocks=stability_blocks
            )
            tail_probability_mcse = _tail_probability_mcse(len(values))
            common = {
                "country": country,
                "outcome": outcome,
                "bootstrap_replicates": int(len(values)),
                "interval_type": INTERVAL_TYPE,
                "interval_basis": INTERVAL_BASIS,
                "confidence_interval_method": INTERVAL_METHOD,
            }
            interval_rows.append(
                common
                | {
                    "rate_median": float(np.quantile(values, 0.50)),
                    "rate_q025": q025,
                    "rate_q975": q975,
                    "rate_q025_mcse": q025_mcse,
                    "rate_q975_mcse": q975_mcse,
                    "tail_probability_mcse": tail_probability_mcse,
                }
            )
            stability_rows.append(
                {
                    "country": country,
                    "outcome": outcome,
                    "bootstrap_replicates": int(len(values)),
                    "rate_q025": q025,
                    "rate_q975": q975,
                    "rate_q95_width": q975 - q025,
                    "rate_q025_mcse": q025_mcse,
                    "rate_q975_mcse": q975_mcse,
                    "maximum_endpoint_mcse": max(q025_mcse, q975_mcse),
                    "tail_probability_mcse": tail_probability_mcse,
                }
            )
    return pd.DataFrame(interval_rows), pd.DataFrame(stability_rows)


def run_current_practice_bootstrap(
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
    settings = (
        configs["baseline"]
        .get("bayesian_uncertainty", {})
        .get(
            "figure1b_current_practice_conditional_parametric_bootstrap_confidence_interval",
            {},
        )
    )
    replicates = int(
        settings.get("replicates_per_country", DEFAULT_REPLICATES)
        if replicates is None
        else replicates
    )
    minimum_successful_replicates = int(
        settings.get(
            "minimum_successful_replicates", DEFAULT_MINIMUM_SUCCESSFUL_REPLICATES
        )
        if minimum_successful_replicates is None
        else minimum_successful_replicates
    )
    minimum_success_fraction = float(
        settings.get("minimum_success_fraction", DEFAULT_MINIMUM_SUCCESS_FRACTION)
        if minimum_success_fraction is None
        else minimum_success_fraction
    )
    maximum_tail_probability_mcse = float(
        settings.get(
            "maximum_tail_probability_mcse", DEFAULT_MAX_TAIL_PROBABILITY_MCSE
        )
        if maximum_tail_probability_mcse is None
        else maximum_tail_probability_mcse
    )
    stability_blocks = int(
        settings.get("stability_blocks", DEFAULT_STABILITY_BLOCKS)
        if stability_blocks is None
        else stability_blocks
    )
    n_jobs = int(
        settings.get("parallel_workers", 48) if n_jobs is None else n_jobs
    )
    chunk_size = int(
        settings.get("checkpoint_chunk_size", DEFAULT_CHUNK_SIZE)
        if chunk_size is None
        else chunk_size
    )
    seed = int(settings.get("seed", 20260718) if seed is None else seed)
    publication = publication_country_names(configs)
    resolved = publication if countries is None else list(countries)
    outside = sorted(set(resolved).difference(publication))
    if outside or len(set(resolved)) != len(resolved):
        raise ValueError(f"Invalid bootstrap countries: {outside or resolved}")
    if replicates < 100:
        raise ValueError("Figure 1b bootstrap requires at least 100 replicates")
    if minimum_successful_replicates > replicates:
        raise ValueError("Minimum successful replicates cannot exceed requested")
    if not 0.0 < minimum_success_fraction <= 1.0:
        raise ValueError("Minimum success fraction must be in (0, 1]")
    calibration_input_hashes = validated_calibration_artifact_path_hashes(
        resolved,
        context="Figure 1b parametric bootstrap",
    )

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
        raise ValueError("Figure 1b retry schedule must exceed first-pass maxiter")
    workers = min(max(1, n_jobs), available_cpus())
    chunk_size = max(workers, chunk_size)
    fingerprint = _checkpoint_fingerprint(
        countries=resolved,
        replicates=replicates,
        seed=seed,
        maxiter=resolved_maxiter,
    )
    _validate_or_create_checkpoint_metadata(
        fingerprint=fingerprint,
        countries=resolved,
        replicates=replicates,
        seed=seed,
        maxiter=resolved_maxiter,
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
        for replicate in range(1, replicates + 1)
        if (country, replicate) not in checkpoint_keys
    ]
    configure_worker_thread_limits()
    total = len(resolved) * replicates
    print(
        f"Figure 1b current-practice bootstrap: completed={len(checkpoint_keys)}/"
        f"{total}, pending={len(tasks)}, workers={workers}",
        flush=True,
    )
    checkpoint_draws, checkpoint_fits = _run_tasks_with_checkpoints(
        tasks,
        workers=workers,
        checkpoint_size=chunk_size,
        total=total,
        checkpoint_draws=checkpoint_draws,
        checkpoint_fits=checkpoint_fits,
    )

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
        failed = failed.loc[previous_budget.lt(retry_maxiter)]
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
                maxiter=retry_maxiter,
                attempt=int(row.attempt) + 1,
            )
            for row in failed.itertuples(index=False)
        ]
        print(
            f"Retrying {len(retry_tasks)} failed Figure 1b refits with "
            f"maxiter={retry_maxiter}",
            flush=True,
        )
        checkpoint_draws, checkpoint_fits = _run_tasks_with_checkpoints(
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
        ["country", "bootstrap_replicate"]
    ).reset_index(drop=True)
    success_by_country = (
        fits.assign(success=fits["success"].astype(bool))
        .groupby("country")["success"]
        .agg(["sum", "count"])
    )
    insufficient = success_by_country.loc[
        success_by_country["sum"].lt(minimum_successful_replicates)
        | (success_by_country["sum"] / success_by_country["count"]).lt(
            minimum_success_fraction
        )
    ]
    if not insufficient.empty:
        raise RuntimeError(
            "Figure 1b bootstrap did not meet the country success gate: "
            + repr(insufficient.to_dict(orient="index"))
        )
    successful_counts = fits.loc[fits["success"].astype(bool)].groupby(
        "country"
    )["bootstrap_replicate"].nunique()
    observed_counts = draws.groupby("country")["bootstrap_replicate"].nunique()
    if (
        draws.duplicated(["country", "bootstrap_replicate"]).any()
        or successful_counts.to_dict() != observed_counts.to_dict()
        or not np.isfinite(
            draws[list(DRAW_COLUMNS.values())].to_numpy(dtype=float)
        ).all()
    ):
        raise RuntimeError("Figure 1b successful refits and retained draws do not match")

    intervals, stability = _summarise_intervals(
        draws, countries=resolved, stability_blocks=stability_blocks
    )
    if (
        not np.isfinite(stability["tail_probability_mcse"]).all()
        or float(stability["tail_probability_mcse"].max())
        > maximum_tail_probability_mcse
    ):
        raise RuntimeError("Figure 1b tail-probability MCSE exceeds its publication gate")

    write_dataframe(draws, DRAW_PATH)
    write_dataframe(intervals, INTERVAL_PATH)
    write_dataframe(fits, FIT_DIAGNOSTIC_PATH)
    write_dataframe(stability, STABILITY_PATH)
    metadata = current_run_metadata(
        STEM,
        row_counts={
            "current_practice_bootstrap_draws": int(len(draws)),
            "confidence_interval_rows": int(len(intervals)),
            "fit_diagnostics": int(len(fits)),
            "interval_stability_rows": int(len(stability)),
        },
    ) | {
        "statistical_target": "frequentist_confidence_interval",
        "interval_type": INTERVAL_TYPE,
        "interval_basis": INTERVAL_BASIS,
        "confidence_interval_method": INTERVAL_METHOD,
        "bootstrap_data_generation": "conditional_fitted_AR1_path_plus_NB2_measurement",
        "bootstrap_refit": "country_state_space_MAP_full_refit_per_replicate",
        "conditional_on_fitted_latent_process_path": True,
        "latent_process_path_regenerated": False,
        "scenario_scope": "current_practice_only",
        "outcomes": list(OUTCOME_COLUMNS),
        "future_observation_prediction_interval": False,
        "posterior_credible_interval": False,
        "countries": resolved,
        "replicates_requested_per_country": replicates,
        "successful_replicates_by_country": {
            country: int(row["sum"])
            for country, row in success_by_country.iterrows()
        },
        "minimum_successful_replicates": minimum_successful_replicates,
        "minimum_success_fraction": minimum_success_fraction,
        "maximum_tail_probability_mcse_threshold": maximum_tail_probability_mcse,
        "observed_maximum_tail_probability_mcse": float(
            stability["tail_probability_mcse"].max()
        ),
        "seed": seed,
        "n_jobs": workers,
        "maxiter": resolved_maxiter,
        "retry_maxiter_schedule": retry_maxiters,
        "input_artifact_path_sha256": calibration_input_hashes,
        "output_artifact_sha256": {
            "current_practice_bootstrap_draws": file_sha256(DRAW_PATH),
            "confidence_intervals": file_sha256(INTERVAL_PATH),
            "fit_diagnostics": file_sha256(FIT_DIAGNOSTIC_PATH),
            "interval_stability": file_sha256(STABILITY_PATH),
        },
    }
    write_run_metadata(STEM, metadata)
    return draws, intervals, fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--minimum-successful-replicates", type=int, default=None)
    parser.add_argument("--minimum-success-fraction", type=float, default=None)
    parser.add_argument("--maximum-tail-probability-mcse", type=float, default=None)
    parser.add_argument("--stability-blocks", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    args = parser.parse_args()
    run_current_practice_bootstrap(
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
