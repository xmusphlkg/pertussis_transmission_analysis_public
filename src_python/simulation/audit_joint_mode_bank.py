from __future__ import annotations

"""Audit exact complete-state mode-bank MH proposals at the full posterior."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.calibration.mcmc_diagnostics import summarize_convergence
from src_python.simulation.common import (
    current_run_metadata,
    load_configs,
    publication_country_names,
    write_run_metadata,
)
from src_python.simulation.run_hierarchical_joint_posterior import (
    SHARED_PARAMETER_NAMES,
    _context,
    _state_gaussian_prior_geometry,
)
from src_python.simulation.run_hierarchical_joint_smc import (
    ModeBank,
    PriorGeometry,
    _cloud_from_completed_posterior,
    _cloud_mode_coordinates,
    _joint_crossfit_mode_bank_move,
    _joint_mode_bank_independence_move,
    _load_mode_bank,
    _mode_bank_log_density,
    _posterior_rows,
    _with_process_support,
)
from src_python.simulation.run_bayesian_uncertainty import (
    _compute_smc_diagnostics,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import available_cpus, configure_worker_thread_limits


def _parse_origin_spec(origin_spec: str) -> tuple[str, int | None]:
    """Return an output stem and optional chain selector.

    A completed multi-chain audit can be continued without materialising four
    duplicate parquet files by spelling an origin as ``stem::chain=N``.
    Ordinary stems remain backward compatible.
    """

    marker = "::chain="
    if marker not in origin_spec:
        return origin_spec, None
    stem, chain_text = origin_spec.rsplit(marker, 1)
    if not stem or not chain_text:
        raise ValueError(
            "Origin chain selectors must use 'stem::chain=N'"
        )
    try:
        chain = int(chain_text)
    except ValueError as exc:
        raise ValueError(
            "Origin chain selectors must use an integer chain number"
        ) from exc
    if chain < 1:
        raise ValueError("Origin chain numbers must be positive")
    return stem, chain


def _single_component_bank(mode_bank: ModeBank, index: int) -> ModeBank:
    return ModeBank(
        means=(mode_bank.means[index],),
        covariances=(mode_bank.covariances[index],),
        precisions=(mode_bank.precisions[index],),
        log_determinants=(mode_bank.log_determinants[index],),
        cholesky_factors=(mode_bank.cholesky_factors[index],),
        degrees_of_freedom=mode_bank.degrees_of_freedom,
        source_stems=(mode_bank.source_stems[index],),
        dimension=mode_bank.dimension,
    )


def _nearest_mode(values: np.ndarray, mode_bank: ModeBank) -> np.ndarray:
    scores = np.column_stack(
        [
            _mode_bank_log_density(values, _single_component_bank(mode_bank, index))
            for index in range(len(mode_bank.means))
        ]
    )
    return np.argmax(scores, axis=1)


def audit_joint_mode_bank(
    *,
    mode_bank_stems: list[str],
    origin_stems: list[str],
    covariance_inflation: float,
    covariance_floor_fraction: float,
    degrees_of_freedom: float,
    process_support_bound: float,
    particles_per_origin: int,
    rounds: int,
    crossfit: bool,
    write_final_posterior: bool,
    n_jobs: int,
    seed: int,
    output_stem: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if int(rounds) < 1:
        raise ValueError("Mode-bank audit rounds must be positive")
    configure_worker_thread_limits()
    configs = load_configs()
    contexts = [
        _with_process_support(
            _context(country, configs=configs), process_support_bound
        )
        for country in publication_country_names(configs)
    ]
    geometries: dict[str, PriorGeometry] = {}
    for context in contexts:
        mean, covariance, precision = _state_gaussian_prior_geometry(
            context, context.prior_base
        )
        geometries[context.country] = PriorGeometry(mean, covariance, precision)
    mode_bank = _load_mode_bank(
        mode_bank_stems,
        contexts,
        covariance_inflation=covariance_inflation,
        covariance_floor_fraction=covariance_floor_fraction,
        degrees_of_freedom=degrees_of_freedom,
    )
    workers = min(max(1, int(n_jobs)), available_cpus())
    rows: list[dict[str, Any]] = []
    final_posterior_parts: list[pd.DataFrame] = []
    for origin_index, origin_spec in enumerate(origin_stems):
        origin_stem, origin_chain = _parse_origin_spec(origin_spec)
        posterior_path = project_path(
            "outputs", "simulations", f"{origin_stem}_posterior_samples.parquet"
        )
        posterior = pd.read_parquet(posterior_path)
        if origin_chain is not None:
            if "chain" not in posterior.columns:
                raise ValueError(
                    f"Origin {origin_stem!r} has no chain column"
                )
            posterior = posterior.loc[
                pd.to_numeric(posterior["chain"], errors="coerce")
                == origin_chain
            ].copy()
            if posterior.empty:
                raise ValueError(
                    f"Origin {origin_stem!r} has no chain {origin_chain}"
                )
        cloud = _cloud_from_completed_posterior(
            posterior,
            contexts,
            maximum_particles=int(particles_per_origin),
        )
        before_coordinates, _ = _cloud_mode_coordinates(cloud, contexts)
        before_mode = _nearest_mode(before_coordinates, mode_bank)
        rng = np.random.default_rng(int(seed) + origin_index * 104729)
        origin_acceptance: list[float] = []
        for round_index in range(1, int(rounds) + 1):
            move = (
                _joint_crossfit_mode_bank_move
                if crossfit
                else _joint_mode_bank_independence_move
            )
            acceptance = move(
                cloud,
                contexts,
                geometries,
                mode_bank,
                temperature=1.0,
                n_jobs=workers,
                rng=rng,
            )
            origin_acceptance.append(acceptance)
            after_coordinates, _ = _cloud_mode_coordinates(cloud, contexts)
            after_mode = _nearest_mode(after_coordinates, mode_bank)
            changed = before_mode != after_mode
            rows.append(
                {
                    "origin_stem": origin_stem,
                    "origin_chain": origin_chain,
                    "origin_mode_index": origin_index + 1,
                    "round": round_index,
                    "particle_count": len(before_mode),
                    "mh_acceptance": acceptance,
                    "accepted_particle_count": int(
                        round(acceptance * len(before_mode))
                    ),
                    "accepted_cross_mode_count": int(np.sum(changed)),
                    "accepted_cross_mode_fraction": float(np.mean(changed)),
                    "distinct_modes_before": int(len(np.unique(before_mode))),
                    "distinct_modes_after": int(len(np.unique(after_mode))),
                    **{
                        f"after_mode_{index + 1}_count": int(
                            np.sum(after_mode == index)
                        )
                        for index in range(len(mode_bank.means))
                    },
                }
            )
            before_mode = after_mode
        if write_final_posterior:
            final = _posterior_rows(
                cloud,
                contexts,
                seed=int(seed) + origin_index * 104729 + 9001,
                stage_count=int(rounds),
                minimum_stage_ess_fraction=1.0,
                maximum_stage_weight=1.0 / len(cloud.structural),
                move_acceptance=float(np.mean(origin_acceptance)),
                island_log_evidence=np.asarray([0.0]),
            )
            final["chain"] = origin_index + 1
            final["draw"] = final.groupby("country").cumcount() + 1
            offset = origin_index * len(cloud.structural)
            final["posterior_draw"] = final["posterior_draw"] + offset
            final["structural_draw_id"] = final["structural_draw_id"] + offset
            final["sampling_method"] = (
                "completed_smc_plus_exact_crossfit_mode_bank_audit"
            )
            final["audit_origin_stem"] = origin_stem
            final["audit_origin_chain"] = origin_chain
            final["audit_rejuvenation_rounds"] = int(rounds)
            final_posterior_parts.append(final)
    audit = pd.DataFrame(rows)
    output_path = project_path(
        "outputs", "diagnostics", f"{output_stem}_mode_bank_audit.csv"
    )
    write_dataframe(audit, output_path)
    row_counts = {"mode_bank_origin_audits": len(audit)}
    final_posterior_path = None
    diagnostics_path = None
    convergence = None
    if write_final_posterior:
        final_posterior = pd.concat(final_posterior_parts, ignore_index=True)
        parameter_columns = tuple(SHARED_PARAMETER_NAMES) + (
            "beta_S",
            "reporting_multiplier",
        ) + tuple(
            sorted(
                column
                for column in final_posterior.columns
                if str(column).startswith("log_beta_process_")
            )
        )
        diagnostics = _compute_smc_diagnostics(
            final_posterior, parameter_columns
        )
        convergence = summarize_convergence(diagnostics)
        final_posterior_path = project_path(
            "outputs",
            "simulations",
            f"{output_stem}_posterior_samples.parquet",
        )
        diagnostics_path = project_path(
            "outputs",
            "summaries",
            f"{output_stem}_convergence_diagnostics.csv",
        )
        write_dataframe(final_posterior, final_posterior_path)
        write_dataframe(diagnostics, diagnostics_path)
        row_counts["final_posterior_samples"] = len(final_posterior)
        row_counts["convergence_diagnostics"] = len(diagnostics)
    metadata = current_run_metadata(
        output_stem,
        row_counts=row_counts,
    ) | {
        "audit": "exact_complete_state_mode_bank_independence_mh",
        "mode_bank_source_stems": mode_bank_stems,
        "origin_stems": origin_stems,
        "mode_bank_dimension": mode_bank.dimension,
        "covariance_inflation": float(covariance_inflation),
        "covariance_floor_fraction": float(covariance_floor_fraction),
        "degrees_of_freedom": float(degrees_of_freedom),
        "process_support_bound": float(process_support_bound),
        "particles_per_origin": int(particles_per_origin),
        "rounds": int(rounds),
        "crossfit_current_population": bool(crossfit),
        "final_posterior_written": bool(write_final_posterior),
        "n_jobs": workers,
        "seed": int(seed),
        "mean_mh_acceptance": float(audit["mh_acceptance"].mean()),
        "total_accepted_cross_mode_count": int(
            audit["accepted_cross_mode_count"].sum()
        ),
        "convergence_summary": convergence,
    }
    write_run_metadata(output_stem, metadata)
    return audit, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-bank-stems", required=True)
    parser.add_argument("--origin-stems", required=True)
    parser.add_argument("--covariance-inflation", type=float, required=True)
    parser.add_argument(
        "--covariance-floor-fraction", type=float, default=0.01
    )
    parser.add_argument("--degrees-of-freedom", type=float, required=True)
    parser.add_argument("--process-support-bound", type=float, default=5.0)
    parser.add_argument("--particles-per-origin", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--crossfit", action="store_true")
    parser.add_argument("--write-final-posterior", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    audit_joint_mode_bank(
        mode_bank_stems=args.mode_bank_stems.split(","),
        origin_stems=args.origin_stems.split(","),
        covariance_inflation=args.covariance_inflation,
        covariance_floor_fraction=args.covariance_floor_fraction,
        degrees_of_freedom=args.degrees_of_freedom,
        process_support_bound=args.process_support_bound,
        particles_per_origin=args.particles_per_origin,
        rounds=args.rounds,
        crossfit=args.crossfit,
        write_final_posterior=args.write_final_posterior,
        n_jobs=args.n_jobs,
        seed=args.seed,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
