"""Run the publication joint PSA with an interruption-safe provenance checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import (
    current_run_metadata,
    load_configs,
    publication_country_names,
    source_code_fingerprint,
    uncertainty_config_fingerprint,
    write_run_metadata,
)
from src_python.simulation.run_joint_psa_rank_acceptability import (
    ACCEPTABILITY_PATH,
    RANK_SAMPLE_PATH,
    RUN_SUMMARY_PATH,
    SAMPLE_DESIGN,
    SAMPLE_PATH,
    SELECTED_STRATEGIES,
    SIMULATION_SUMMARY_PATH,
    SIMULATION_TS_PATH,
    STEM,
    UNCERTAINTY_SCHEMA_VERSION,
    _completed_rank_samples,
    _default_parameter_specs,
    _retain_matching_completed_draws,
    _resume_is_current,
    _sample_table,
    run_joint_psa,
)


def _remove_stale_outputs() -> None:
    for path in (
        SAMPLE_PATH,
        RANK_SAMPLE_PATH,
        ACCEPTABILITY_PATH,
        RUN_SUMMARY_PATH,
        SIMULATION_SUMMARY_PATH,
        SIMULATION_TS_PATH,
    ):
        for artifact in (Path(path), Path(path).with_suffix(".parquet")):
            artifact.unlink(missing_ok=True)


def _initialise_checkpoint(
    *,
    sample_size: int,
    seed: int,
    sample_batch_size: int,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
) -> None:
    configs = load_configs()
    metadata = current_run_metadata(
        STEM,
        row_counts={
            "parameter_samples": int(sample_size),
            "rank_samples": 0,
            "checkpoint_completed_samples": 0,
        },
    )
    metadata.update(
        {
            "sample_size_requested": int(sample_size),
            "sample_seed": int(seed),
            "countries": list(countries),
            "strategies": list(strategies),
            "resume": True,
            "sample_batch_size": int(sample_batch_size),
            "smoke_runtime": False,
            "keep_timeseries": False,
            "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
            "uncertainty_config_hash": uncertainty_config_fingerprint(configs),
            "sample_design": SAMPLE_DESIGN,
            "run_status": "checkpoint_initialised_before_first_batch",
        }
    )
    write_run_metadata(STEM, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    configs = load_configs()
    countries = tuple(publication_country_names(configs))
    strategies = tuple(SELECTED_STRATEGIES)
    current, reason = _resume_is_current(configs, source_code_fingerprint())
    if not current:
        print(f"Starting a fresh joint PSA because {reason}")
        _remove_stale_outputs()
        _initialise_checkpoint(
            sample_size=int(args.samples),
            seed=int(args.seed),
            sample_batch_size=int(args.sample_batch_size),
            countries=countries,
            strategies=strategies,
        )
    else:
        registry = configs.get("parameter_distributions", {})
        settings = registry.get("joint_rank_psa", {}) if isinstance(registry, dict) else {}
        specs = settings.get("parameters", {}) if isinstance(settings, dict) else {}
        specs = specs or _default_parameter_specs()
        samples = _sample_table(
            int(args.samples),
            int(args.seed),
            specs,
            schema_version=UNCERTAINTY_SCHEMA_VERSION,
        )
        completed, existing = _completed_rank_samples(
            RANK_SAMPLE_PATH,
            countries=countries,
            strategies=strategies,
        )
        matching, _ = _retain_matching_completed_draws(completed, existing, samples)
        if len(matching) == int(args.samples):
            print(
                f"Joint PSA already complete: {len(matching)}/{int(args.samples)} "
                "fixed-seed samples match current provenance."
            )
            return

    run_joint_psa(
        sample_size=int(args.samples),
        seed=int(args.seed),
        countries=countries,
        strategies=strategies,
        n_jobs=args.n_jobs,
        sample_batch_size=int(args.sample_batch_size),
        resume=True,
        smoke_runtime=False,
        keep_timeseries=False,
    )


if __name__ == "__main__":
    main()
