from __future__ import annotations

"""Run Bayesian sampler pilot grids and summarize each candidate automatically.

Auxiliary simulation script: not part of the current publication pipeline.

The suite can compare historical adaptive-MH MCMC settings with the validated
beta-grid posterior path used for production uncertainty intervals.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src_python.simulation.summarize_mcmc_pilots import main as summarize_pilots
from src_python.utils.io import project_path


@dataclass(frozen=True)
class PilotSpec:
    stem: str
    countries: str
    sampler: str
    solver_mode: str = "calibration"
    warmup: int = 120
    draws: int = 120
    thin: int = 1
    n_chains: int = 4
    n_jobs: int = 4
    proposal_scale: float = 0.25
    fix_durations: bool = True
    fix_parameters: tuple[str, ...] = ()
    initial_samples: str | None = None
    initial_strategy: str = "calibrated"
    dispersion: float | None = None
    likelihood_observation_frequency: str | None = None
    parameterization: str = "standard"
    grid_points: int | None = None
    grid_log_beta_half_width: float | None = None
    grid_max_points: int | None = None
    grid_max_refinements: int | None = None
    grid_tail_drop: float | None = None
    grid_min_effective_points: float | None = None
    grid_max_single_weight: float | None = None
    grid_smoothing: str | None = None
    grid_savgol_window: int | None = None
    extra: dict[str, str] = field(default_factory=dict)
    timeout_minutes: float | None = None


def suite_specs(name: str) -> list[PilotSpec]:
    if name == "sa_repair_stage1":
        return [
            PilotSpec(
                stem="repair_sa_calib_adapt_s02_w20d20",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.20,
                warmup=20,
                draws=20,
                timeout_minutes=20,
            ),
            PilotSpec(
                stem="repair_sa_calib_adapt_s04_w20d20",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=20,
                draws=20,
                timeout_minutes=20,
            ),
            PilotSpec(
                stem="repair_sa_calib_comp_s02_w20d20",
                countries="South_Africa",
                sampler="componentwise_mh",
                proposal_scale=0.20,
                warmup=20,
                draws=20,
                timeout_minutes=20,
            ),
            PilotSpec(
                stem="repair_sa_calib_slice_s02_w12d12",
                countries="South_Africa",
                sampler="slice",
                proposal_scale=0.20,
                warmup=12,
                draws=12,
                timeout_minutes=20,
            ),
            PilotSpec(
                stem="repair_sa_fast_comp_s02_w80d80",
                countries="South_Africa",
                sampler="componentwise_mh",
                solver_mode="mcmc_fast",
                proposal_scale=0.20,
                warmup=80,
                draws=80,
                timeout_minutes=20,
            ),
            PilotSpec(
                stem="repair_sa_fast_adapt_s04_w120d80",
                countries="South_Africa",
                sampler="adaptive_mh",
                solver_mode="mcmc_fast",
                proposal_scale=0.40,
                warmup=120,
                draws=80,
                timeout_minutes=20,
            ),
        ]
    if name == "sa_repair_stage2":
        return [
            PilotSpec(
                stem="repair_sa_calib_adapt_s04_best_w120d120",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=120,
                draws=120,
                initial_samples="outputs/simulations/repair_sa_calib_adapt_s04_w20d20_posterior_samples.parquet",
                initial_strategy="posterior_best",
                timeout_minutes=75,
            ),
            PilotSpec(
                stem="repair_sa_calib_adapt_s06_best_w120d120",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.60,
                warmup=120,
                draws=120,
                initial_samples="outputs/simulations/repair_sa_calib_adapt_s04_w20d20_posterior_samples.parquet",
                initial_strategy="posterior_best",
                timeout_minutes=75,
            ),
            PilotSpec(
                stem="repair_sa_calib_adapt_s04_random_w120d120",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=120,
                draws=120,
                initial_samples="outputs/simulations/repair_sa_calib_adapt_s04_w20d20_posterior_samples.parquet",
                initial_strategy="sample_random",
                timeout_minutes=75,
            ),
            PilotSpec(
                stem="repair_sa_calib_comp_s04_best_w80d80",
                countries="South_Africa",
                sampler="componentwise_mh",
                proposal_scale=0.40,
                warmup=80,
                draws=80,
                initial_samples="outputs/simulations/repair_sa_calib_adapt_s04_w20d20_posterior_samples.parquet",
                initial_strategy="posterior_best",
                timeout_minutes=60,
            ),
        ]
    if name == "sa_parameterization_stage1":
        warm_start = "outputs/simulations/repair_sa_calib_adapt_s04_w20d20_posterior_samples.parquet"
        return [
            PilotSpec(
                stem="repair_sa_calib_fixreport_adapt_s04_w40d40",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=40,
                draws=40,
                fix_parameters=("reporting_multiplier",),
                initial_samples=warm_start,
                initial_strategy="posterior_best",
                timeout_minutes=30,
            ),
            PilotSpec(
                stem="repair_sa_calib_fixveasym_adapt_s04_w40d40",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=40,
                draws=40,
                fix_parameters=("VE_sus", "VE_inf", "relative_infectiousness_asymptomatic"),
                initial_samples=warm_start,
                initial_strategy="posterior_best",
                timeout_minutes=30,
            ),
            PilotSpec(
                stem="repair_sa_calib_fixfit_adapt_s04_w40d40",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=40,
                draws=40,
                fix_parameters=("fitness_R",),
                initial_samples=warm_start,
                initial_strategy="posterior_best",
                timeout_minutes=30,
            ),
        ]
    if name == "sa_prior_stage1":
        warm_start = "outputs/simulations/repair_sa_calib_adapt_s04_w20d20_posterior_samples.parquet"
        return [
            PilotSpec(
                stem="repair_sa_calib_tightve_adapt_s04_w40d40",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=40,
                draws=40,
                initial_samples=warm_start,
                initial_strategy="posterior_best",
                extra={
                    "--ve-prior-sd": "0.025",
                    "--rel-asym-prior-sd": "0.05",
                },
                timeout_minutes=30,
            ),
            PilotSpec(
                stem="repair_sa_calib_tightreport_adapt_s04_w40d40",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=40,
                draws=40,
                initial_samples=warm_start,
                initial_strategy="posterior_best",
                extra={
                    "--beta-prior-log-sd": "0.40",
                    "--reporting-prior-log-sd": "0.30",
                },
                timeout_minutes=30,
            ),
            PilotSpec(
                stem="repair_sa_calib_tightall_adapt_s04_w40d40",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=40,
                draws=40,
                initial_samples=warm_start,
                initial_strategy="posterior_best",
                extra={
                    "--prior-sd-scale": "0.50",
                    "--ve-prior-sd": "0.025",
                    "--reporting-prior-log-sd": "0.30",
                },
                timeout_minutes=30,
            ),
        ]
    if name == "core_repair_stage1":
        return [
            PilotSpec(
                stem="repair_core_calib_adapt_s04_w20d20",
                countries="South_Africa,Thailand,Japan,China",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=20,
                draws=20,
                n_jobs=8,
                timeout_minutes=45,
            ),
        ]
    if name == "sa_repair":
        return [
            PilotSpec(
                stem="repair_sa_calib_comp_s02_w120d120",
                countries="South_Africa",
                sampler="componentwise_mh",
                proposal_scale=0.20,
            ),
            PilotSpec(
                stem="repair_sa_calib_comp_s04_w120d120",
                countries="South_Africa",
                sampler="componentwise_mh",
                proposal_scale=0.40,
            ),
            PilotSpec(
                stem="repair_sa_calib_adapt_s04_w220d120",
                countries="South_Africa",
                sampler="adaptive_mh",
                proposal_scale=0.40,
                warmup=220,
                draws=120,
            ),
            PilotSpec(
                stem="repair_sa_calib_slice_s02_w60d60",
                countries="South_Africa",
                sampler="slice",
                proposal_scale=0.20,
                warmup=60,
                draws=60,
            ),
        ]
    if name == "sa_parameterization":
        return [
            PilotSpec(
                stem="repair_sa_calib_fixreport_comp_s025_w120d120",
                countries="South_Africa",
                sampler="componentwise_mh",
                proposal_scale=0.25,
                fix_parameters=("reporting_multiplier",),
            ),
            PilotSpec(
                stem="repair_sa_calib_fixveasym_comp_s025_w120d120",
                countries="South_Africa",
                sampler="componentwise_mh",
                proposal_scale=0.25,
                fix_parameters=("VE_sus", "VE_inf", "relative_infectiousness_asymptomatic"),
            ),
            PilotSpec(
                stem="repair_sa_calib_fixfit_comp_s025_w120d120",
                countries="South_Africa",
                sampler="componentwise_mh",
                proposal_scale=0.25,
                fix_parameters=("fitness_R",),
            ),
        ]
    if name == "core_repair_smoke":
        return [
            PilotSpec(
                stem="repair_core_calib_comp_s025_w80d80",
                countries="South_Africa,Thailand,Japan,China",
                sampler="componentwise_mh",
                proposal_scale=0.25,
                warmup=80,
                draws=80,
                n_jobs=8,
            ),
        ]
    if name == "identifiability_repair_stage1":
        countries = "South_Africa,Thailand,Japan,China"
        common = {
            "countries": countries,
            "sampler": "adaptive_mh",
            "solver_mode": "calibration",
            "proposal_scale": 0.40,
            "warmup": 60,
            "draws": 60,
            "n_jobs": 8,
            "timeout_minutes": 90,
        }
        return [
            PilotSpec(
                stem="ident_core_standard_adapt_s04_w60d60",
                **common,
            ),
            PilotSpec(
                stem="ident_core_product_adapt_s04_w60d60",
                parameterization="beta_reporting_product",
                **common,
            ),
            PilotSpec(
                stem="ident_core_tight_report_beta_adapt_s04_w60d60",
                extra={
                    "--beta-prior-log-sd": "0.25",
                    "--reporting-prior-log-sd": "0.20",
                },
                **common,
            ),
            PilotSpec(
                stem="ident_core_fixreport_adapt_s04_w60d60",
                fix_parameters=("reporting_multiplier",),
                **common,
            ),
            PilotSpec(
                stem="ident_core_tight_ve_asym_adapt_s04_w60d60",
                extra={
                    "--ve-prior-sd": "0.02",
                    "--rel-asym-prior-sd": "0.05",
                },
                **common,
            ),
            PilotSpec(
                stem="ident_core_fixveasym_adapt_s04_w60d60",
                fix_parameters=("VE_sus", "VE_inf", "relative_infectiousness_asymptomatic"),
                **common,
            ),
            PilotSpec(
                stem="ident_core_fixreport_fixveasym_adapt_s04_w60d60",
                fix_parameters=(
                    "reporting_multiplier",
                    "VE_sus",
                    "VE_inf",
                    "relative_infectiousness_asymptomatic",
                ),
                **common,
            ),
            PilotSpec(
                stem="ident_core_fixreport_fixveasym_fixfit_betagrid_g161_d60",
                sampler="beta_grid",
                proposal_scale=1.0,
                warmup=0,
                draws=60,
                fix_parameters=(
                    "reporting_multiplier",
                    "VE_sus",
                    "VE_inf",
                    "relative_infectiousness_asymptomatic",
                    "fitness_R",
                ),
                grid_points=161,
                grid_log_beta_half_width=0.08,
                grid_max_points=321,
                grid_max_refinements=5,
                grid_smoothing="auto",
                **{k: v for k, v in common.items() if k not in {"sampler", "proposal_scale", "warmup", "draws"}},
            ),
            PilotSpec(
                stem="ident_core_product_tightall_adapt_s04_w60d60",
                parameterization="beta_reporting_product",
                extra={
                    "--beta-prior-log-sd": "0.30",
                    "--reporting-prior-log-sd": "0.20",
                    "--ve-prior-sd": "0.03",
                    "--rel-asym-prior-sd": "0.05",
                },
                **common,
            ),
        ]
    if name == "identifiability_repair_smoke":
        countries = "South_Africa,Thailand,Japan,China"
        common = {
            "countries": countries,
            "sampler": "adaptive_mh",
            "solver_mode": "calibration",
            "proposal_scale": 0.40,
            "warmup": 20,
            "draws": 20,
            "n_jobs": 8,
            "timeout_minutes": 45,
        }
        return [
            PilotSpec(
                stem="ident_smoke_standard_adapt_s04_w20d20",
                **common,
            ),
            PilotSpec(
                stem="ident_smoke_product_adapt_s04_w20d20",
                parameterization="beta_reporting_product",
                **common,
            ),
            PilotSpec(
                stem="ident_smoke_tight_report_beta_adapt_s04_w20d20",
                extra={
                    "--beta-prior-log-sd": "0.25",
                    "--reporting-prior-log-sd": "0.20",
                },
                **common,
            ),
            PilotSpec(
                stem="ident_smoke_fixreport_adapt_s04_w20d20",
                fix_parameters=("reporting_multiplier",),
                **common,
            ),
            PilotSpec(
                stem="ident_smoke_tight_ve_asym_adapt_s04_w20d20",
                extra={
                    "--ve-prior-sd": "0.02",
                    "--rel-asym-prior-sd": "0.05",
                },
                **common,
            ),
            PilotSpec(
                stem="ident_smoke_fixveasym_adapt_s04_w20d20",
                fix_parameters=("VE_sus", "VE_inf", "relative_infectiousness_asymptomatic"),
                **common,
            ),
            PilotSpec(
                stem="ident_smoke_fixreport_fixveasym_adapt_s04_w20d20",
                fix_parameters=(
                    "reporting_multiplier",
                    "VE_sus",
                    "VE_inf",
                    "relative_infectiousness_asymptomatic",
                ),
                **common,
            ),
            PilotSpec(
                stem="ident_smoke_fixreport_fixveasym_fixfit_betagrid_g81_d20",
                sampler="beta_grid",
                proposal_scale=1.0,
                warmup=0,
                draws=20,
                fix_parameters=(
                    "reporting_multiplier",
                    "VE_sus",
                    "VE_inf",
                    "relative_infectiousness_asymptomatic",
                    "fitness_R",
                ),
                grid_points=81,
                grid_log_beta_half_width=0.08,
                grid_max_points=321,
                grid_max_refinements=5,
                grid_smoothing="auto",
                **{k: v for k, v in common.items() if k not in {"sampler", "proposal_scale", "warmup", "draws"}},
            ),
            PilotSpec(
                stem="ident_smoke_product_tightall_adapt_s04_w20d20",
                parameterization="beta_reporting_product",
                extra={
                    "--beta-prior-log-sd": "0.30",
                    "--reporting-prior-log-sd": "0.20",
                    "--ve-prior-sd": "0.03",
                    "--rel-asym-prior-sd": "0.05",
                },
                **common,
            ),
        ]
    raise ValueError(f"Unknown pilot suite: {name}")


def _samples_path(stem: str) -> Path:
    return project_path("outputs", "simulations", f"{stem}_posterior_samples.parquet")


def command_for(spec: PilotSpec) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src_python.simulation.run_bayesian_uncertainty",
        "--countries",
        spec.countries,
        "--n-chains",
        str(spec.n_chains),
        "--warmup",
        str(spec.warmup),
        "--draws",
        str(spec.draws),
        "--thin",
        str(spec.thin),
        "--n-jobs",
        str(spec.n_jobs),
        "--sampler",
        spec.sampler,
        "--solver-mode",
        spec.solver_mode,
        "--proposal-scale",
        str(spec.proposal_scale),
        "--skip-posterior-predictive",
        "--skip-k-sensitivity",
        "--output-stem",
        spec.stem,
        "--parameterization",
        spec.parameterization,
    ]
    if spec.fix_durations:
        cmd.append("--fix-durations")
    if spec.fix_parameters:
        cmd.extend(["--fix-parameters", ",".join(spec.fix_parameters)])
    if spec.initial_samples:
        cmd.extend(["--initial-samples", spec.initial_samples])
        cmd.extend(["--initial-strategy", spec.initial_strategy])
    if spec.dispersion is not None:
        cmd.extend(["--dispersion", str(spec.dispersion)])
    if spec.likelihood_observation_frequency:
        cmd.extend(["--likelihood-observation-frequency", spec.likelihood_observation_frequency])
    if spec.grid_points is not None:
        cmd.extend(["--grid-points", str(spec.grid_points)])
    if spec.grid_log_beta_half_width is not None:
        cmd.extend(["--grid-log-beta-half-width", str(spec.grid_log_beta_half_width)])
    if spec.grid_max_points is not None:
        cmd.extend(["--grid-max-points", str(spec.grid_max_points)])
    if spec.grid_max_refinements is not None:
        cmd.extend(["--grid-max-refinements", str(spec.grid_max_refinements)])
    if spec.grid_tail_drop is not None:
        cmd.extend(["--grid-tail-drop", str(spec.grid_tail_drop)])
    if spec.grid_min_effective_points is not None:
        cmd.extend(["--grid-min-effective-points", str(spec.grid_min_effective_points)])
    if spec.grid_max_single_weight is not None:
        cmd.extend(["--grid-max-single-weight", str(spec.grid_max_single_weight)])
    if spec.grid_smoothing is not None:
        cmd.extend(["--grid-smoothing", str(spec.grid_smoothing)])
    if spec.grid_savgol_window is not None:
        cmd.extend(["--grid-savgol-window", str(spec.grid_savgol_window)])
    for key, value in spec.extra.items():
        cmd.extend([key, value])
    return cmd


def run_specs(
    specs: list[PilotSpec],
    *,
    force: bool = False,
    max_minutes: float | None = None,
    stop_on_failure: bool = False,
    parallel_pilots: int = 1,
) -> None:
    log_dir = project_path("outputs", "metadata", "mcmc_pilot_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "pilot_grid_status.jsonl"
    status_lock = threading.Lock()

    def append_status(record: dict[str, object]) -> None:
        with status_lock:
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def run_one(spec: PilotSpec) -> tuple[PilotSpec, int, bool]:
        log_path = log_dir / f"{spec.stem}.log"
        if _samples_path(spec.stem).exists() and not force:
            return spec, 0, False

        cmd = command_for(spec)
        append_status({
            "event": "start",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "spec": asdict(spec),
            "command": cmd,
        })

        timeout = None
        if max_minutes is not None:
            timeout = max_minutes * 60.0
        elif spec.timeout_minutes is not None:
            timeout = spec.timeout_minutes * 60.0

        timed_out = False
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=project_path(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nTIMEOUT after {timeout:.0f} seconds\n")
                if exc.output:
                    log.write(str(exc.output))
                if exc.stderr:
                    log.write(str(exc.stderr))

        append_status({
            "event": "finish",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "stem": spec.stem,
            "returncode": returncode,
            "timed_out": timed_out,
            "log": str(log_path),
        })

        return spec, returncode, timed_out

    completed: list[str] = []
    runnable = list(specs)
    parallel_pilots = max(1, int(parallel_pilots))

    if parallel_pilots == 1:
        results = [run_one(spec) for spec in runnable]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=parallel_pilots) as executor:
            futures = [executor.submit(run_one, spec) for spec in runnable]
            for future in as_completed(futures):
                results.append(future.result())

    for spec, returncode, _timed_out in results:
        if returncode != 0:
            if stop_on_failure:
                raise RuntimeError(
                    f"Pilot {spec.stem} failed with code {returncode}; "
                    f"see {log_dir / f'{spec.stem}.log'}"
                )
            continue
        if _samples_path(spec.stem).exists():
            completed.append(spec.stem)

    if completed:
        summarize_pilots(completed, output="outputs/summaries/mcmc_pilot_grid_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated Bayesian sampler pilot suites.")
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Pilot suite to run; may be supplied multiple times",
    )
    parser.add_argument("--force", action="store_true", help="Rerun pilots even if sample output exists")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Override per-pilot timeout in minutes",
    )
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop at the first failed pilot")
    parser.add_argument(
        "--parallel-pilots",
        type=int,
        default=1,
        help="Number of pilot subprocesses to run concurrently",
    )
    args = parser.parse_args()

    suites = args.suite or ["sa_repair"]
    specs: list[PilotSpec] = []
    for suite in suites:
        specs.extend(suite_specs(suite))
    run_specs(
        specs,
        force=args.force,
        max_minutes=args.max_minutes,
        stop_on_failure=args.stop_on_failure,
        parallel_pilots=args.parallel_pilots,
    )


if __name__ == "__main__":
    main()
