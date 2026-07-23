"""Run the publication joint PSA with an interruption-safe provenance checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import (
    current_run_metadata,
    load_configs,
    output_metadata_path,
    publication_country_names,
    read_run_metadata,
    validate_run_metadata,
    write_run_metadata,
)
from src_python.simulation.run_joint_psa_rank_acceptability import (
    ACCEPTABILITY_PATH,
    EXPECTED_PARAMETER_NAMES,
    RANK_SAMPLE_PATH,
    RUN_SUMMARY_PATH,
    SAMPLE_DESIGN,
    SAMPLE_PATH,
    PROGRAMME_ONLY_STRATEGIES,
    SELECTED_STRATEGIES,
    SIMULATION_SUMMARY_PATH,
    SIMULATION_TS_PATH,
    STEM,
    UNDER18_PROGRAMME_ACCEPTABILITY_PATH,
    UNDER18_PROGRAMME_RANK_SAMPLE_PATH,
    UNDER18_PROGRAMME_RUN_SUMMARY_PATH,
    UNCERTAINTY_SCHEMA_VERSION,
    _validate_calibration_inputs,
    _completed_rank_samples,
    _default_parameter_specs,
    _retain_matching_completed_draws,
    _resume_metadata_available,
    _sample_table,
    run_joint_psa,
)
from src_python.utils.io import write_dataframe


JOINT_OUTPUT_PATHS = (
    SAMPLE_PATH,
    RANK_SAMPLE_PATH,
    ACCEPTABILITY_PATH,
    RUN_SUMMARY_PATH,
    SIMULATION_SUMMARY_PATH,
    SIMULATION_TS_PATH,
    UNDER18_PROGRAMME_RANK_SAMPLE_PATH,
    UNDER18_PROGRAMME_ACCEPTABILITY_PATH,
    UNDER18_PROGRAMME_RUN_SUMMARY_PATH,
)
JOINT_METADATA_PATH = output_metadata_path(STEM)
JOINT_ARCHIVE_ROOT = ROOT / "outputs" / "archive" / STEM
FIGURE2B_INTERPRETATION = (
    "Selected-input deterministic design-frequency summary across prespecified "
    "settings."
)


def _normalize_figure2b_interpretation(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an order-preserving copy with the locked non-inferential label."""

    if "interpretation" not in frame.columns:
        raise ValueError(
            "Figure 2b interpretation normalization requires an interpretation column"
        )
    normalized = frame.copy(deep=True)
    normalized.loc[:, "interpretation"] = FIGURE2B_INTERPRETATION
    return normalized


def _assert_table_variants_equivalent(
    csv_frame: pd.DataFrame,
    parquet_frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Fail closed when paired table variants differ in values or row order."""

    if list(csv_frame.columns) != list(parquet_frame.columns):
        raise ValueError(f"{label} CSV/Parquet columns or column order disagree")
    try:
        pd.testing.assert_frame_equal(
            csv_frame.reset_index(drop=True),
            parquet_frame.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-15,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise ValueError(f"{label} CSV/Parquet values or row order disagree") from exc


def _atomic_write_figure2b_table_variants(
    normalized_tables: dict[Path, pd.DataFrame],
) -> None:
    """Transactionally replace both CSV/Parquet variants for all input tables."""

    transaction = uuid4().hex
    staged: list[tuple[Path, Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    try:
        for configured_csv, frame in normalized_tables.items():
            csv_path = Path(configured_csv)
            parquet_path = csv_path.with_suffix(".parquet")
            if not csv_path.is_file() or not parquet_path.is_file():
                raise FileNotFoundError(
                    "Figure 2b interpretation normalization requires both table "
                    f"variants: {csv_path}, {parquet_path}"
                )
            staged_csv = csv_path.with_name(
                f".{csv_path.stem}.{transaction}.tmp.csv"
            )
            staged_parquet = csv_path.with_name(
                f".{csv_path.stem}.{transaction}.tmp.parquet"
            )
            write_dataframe(frame, staged_csv)
            if not staged_csv.is_file() or not staged_parquet.is_file():
                raise OSError(
                    f"Could not stage both Figure 2b table variants for {csv_path}"
                )
            _assert_table_variants_equivalent(
                pd.read_csv(staged_csv),
                pd.read_parquet(staged_parquet),
                label=f"staged {csv_path.name}",
            )
            staged.extend(
                (
                    (
                        csv_path,
                        staged_csv,
                        csv_path.with_name(f".{csv_path.name}.{transaction}.bak"),
                    ),
                    (
                        parquet_path,
                        staged_parquet,
                        parquet_path.with_name(
                            f".{parquet_path.name}.{transaction}.bak"
                        ),
                    ),
                )
            )

        for target, temporary, backup in staged:
            if backup.exists():
                raise FileExistsError(
                    f"Figure 2b interpretation backup already exists: {backup}"
                )
            target.replace(backup)
            backups.append((target, backup))
            temporary.replace(target)

        for csv_path, expected in normalized_tables.items():
            csv_path = Path(csv_path)
            installed_csv = pd.read_csv(csv_path)
            installed_parquet = pd.read_parquet(csv_path.with_suffix(".parquet"))
            _assert_table_variants_equivalent(
                installed_csv,
                installed_parquet,
                label=f"installed {csv_path.name}",
            )
            _assert_table_variants_equivalent(
                installed_csv,
                expected,
                label=f"installed {csv_path.name} versus normalized source",
            )
    except BaseException as exc:
        rollback_errors: list[str] = []
        for target, backup in reversed(backups):
            try:
                if backup.exists():
                    if target.exists():
                        target.unlink()
                    backup.replace(target)
            except BaseException as rollback_exc:  # pragma: no cover - catastrophic I/O
                rollback_errors.append(f"{backup} -> {target}: {rollback_exc}")
        for _target, temporary, backup in staged:
            for residue in (temporary, backup):
                try:
                    if residue.exists():
                        residue.unlink()
                except OSError:
                    pass
        detail = f"; rollback failures={rollback_errors}" if rollback_errors else ""
        if isinstance(exc, (KeyboardInterrupt, SystemExit)) and not rollback_errors:
            raise
        raise RuntimeError(
            f"Could not atomically normalize Figure 2b interpretation{detail}"
        ) from exc
    else:
        for _target, backup in backups:
            backup.unlink()


def normalize_figure2b_interpretation_outputs() -> None:
    """Normalize completed Figure 2b summaries without changing other values."""

    normalized_tables: dict[Path, pd.DataFrame] = {}
    for configured_csv in (
        UNDER18_PROGRAMME_ACCEPTABILITY_PATH,
        UNDER18_PROGRAMME_RUN_SUMMARY_PATH,
    ):
        csv_path = Path(configured_csv)
        parquet_path = csv_path.with_suffix(".parquet")
        if not csv_path.is_file() or not parquet_path.is_file():
            raise FileNotFoundError(
                "Figure 2b interpretation normalization requires both table "
                f"variants: {csv_path}, {parquet_path}"
            )
        csv_frame = pd.read_csv(csv_path)
        parquet_frame = pd.read_parquet(parquet_path)
        _assert_table_variants_equivalent(
            csv_frame,
            parquet_frame,
            label=csv_path.name,
        )
        normalized_tables[csv_path] = _normalize_figure2b_interpretation(
            parquet_frame
        )
    _atomic_write_figure2b_table_variants(normalized_tables)


def _joint_output_bundle_paths() -> tuple[Path, ...]:
    """Return only exact active joint-PSA artifacts and their table variants."""

    paths: list[Path] = [Path(JOINT_METADATA_PATH)]
    for configured in JOINT_OUTPUT_PATHS:
        path = Path(configured)
        paths.append(path)
        if path.suffix == ".csv":
            paths.append(path.with_suffix(".parquet"))
        elif path.suffix == ".parquet":
            paths.append(path.with_suffix(".csv"))
    return tuple(dict.fromkeys(paths))


def _joint_archive_directory() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = JOINT_ARCHIVE_ROOT / f"run_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix:02d}")
        suffix += 1
    return candidate


def _remove_empty_archive_directories(
    archive_dir: Path,
    destinations: list[Path],
) -> None:
    directories = {destination.parent for destination in destinations}
    directories.update(
        parent
        for path in tuple(directories)
        for parent in path.parents
        if parent != archive_dir and archive_dir in parent.parents
    )
    for directory in sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        archive_dir.rmdir()
    except OSError:
        pass


def _archive_stale_outputs(
    *,
    reason: str,
) -> Path | None:
    """Move an exact stale output bundle aside, rolling back on any failure."""

    sources = [path for path in _joint_output_bundle_paths() if path.exists()]
    if not sources:
        return None
    archive_dir = _joint_archive_directory()
    records = [
        {
            "original_path": str(path),
            "archive_path": str(path.relative_to(ROOT)),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sources
    ]
    archive_dir.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    destinations: list[Path] = []
    try:
        for source in sources:
            destination = archive_dir / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destinations.append(destination)
            if destination.exists():
                raise FileExistsError(f"Joint PSA archive target exists: {destination}")
            source.replace(destination)
            moved.append((source, destination))

        for record in records:
            archived = archive_dir / record["archive_path"]
            if int(archived.stat().st_size) != int(record["size_bytes"]):
                raise OSError(f"Joint PSA archive size verification failed: {archived}")

        manifest = {
            "schema_version": 1,
            "stem": STEM,
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
    except BaseException as exc:
        rollback_errors: list[str] = []
        for source, destination in reversed(moved):
            try:
                if destination.exists():
                    if source.exists():
                        raise FileExistsError(
                            "Refusing to overwrite an active path during rollback: "
                            f"{source}"
                        )
                    destination.replace(source)
            except BaseException as rollback_exc:  # pragma: no cover - catastrophic I/O
                rollback_errors.append(f"{destination} -> {source}: {rollback_exc}")
        for residue in (archive_dir / "manifest.json.tmp", archive_dir / "manifest.json"):
            if residue.exists():
                residue.unlink()
        _remove_empty_archive_directories(archive_dir, destinations)
        detail = f"; rollback failures={rollback_errors}" if rollback_errors else ""
        if isinstance(exc, (KeyboardInterrupt, SystemExit)) and not rollback_errors:
            raise
        raise RuntimeError(f"Could not archive stale joint PSA outputs{detail}") from exc
    return archive_dir


def _resume_design_compatibility(
    metadata: dict[str, object],
    *,
    sample_size: int,
    seed: int,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
) -> tuple[bool, str]:
    """Require the cached run to match every statistical-design input."""

    expected: dict[str, object] = {
        "sample_size_requested": int(sample_size),
        "sample_seed": int(seed),
        "countries": list(countries),
        "strategies": list(strategies),
        "sample_design": SAMPLE_DESIGN,
        "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "smoke_runtime": False,
        "keep_timeseries": False,
    }
    mismatches = [
        f"{key}={metadata.get(key, 'missing')} (expected {value})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "joint PSA statistical-design inputs match"


def _initialise_checkpoint(
    *,
    sample_size: int,
    seed: int,
    sample_batch_size: int,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
    archived_previous_run: Path | None = None,
) -> None:
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
            "sample_design": SAMPLE_DESIGN,
            "run_status": "checkpoint_initialised_before_first_batch",
            "archived_previous_run": (
                str(archived_previous_run)
                if archived_previous_run is not None
                else None
            ),
        }
    )
    write_run_metadata(STEM, metadata)


def _finalise_completed_metadata(
    *,
    sample_size: int,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
) -> None:
    """Mark the resumable parent complete only after exact artifact checks."""

    import numpy as np
    import pandas as pd

    metadata = validate_run_metadata(STEM)
    rank = pd.read_csv(UNDER18_PROGRAMME_RANK_SAMPLE_PATH)
    parameter_samples = pd.read_csv(SAMPLE_PATH)
    programme_strategies = tuple(PROGRAMME_ONLY_STRATEGIES)
    expected_programme_strategies = len(programme_strategies)
    expected_rows = len(countries) * sample_size * expected_programme_strategies
    required_rank_columns = {
        "country",
        "psa_sample_id",
        "strategy",
        "rank",
        "total_child_adolescent_cases",
        "sample_design",
        "uncertainty_schema_version",
        *EXPECTED_PARAMETER_NAMES,
    }
    if not required_rank_columns.issubset(rank.columns):
        raise ValueError(
            "Joint PSA completion audit failed; rank artifacts lack required columns"
        )
    if "resistance_management_uptake" in rank.columns:
        raise ValueError(
            "Joint PSA completion audit failed; legacy management-uptake dimension remains"
        )
    required_parameter_columns = {
        "psa_sample_id",
        "sample_design",
        "uncertainty_schema_version",
        *EXPECTED_PARAMETER_NAMES,
    }
    if set(parameter_samples.columns) != required_parameter_columns:
        raise ValueError(
            "Joint PSA completion audit failed; parameter samples are not the exact "
            "six-input Figure 2b design"
        )
    rank["psa_sample_id"] = pd.to_numeric(
        rank["psa_sample_id"], errors="coerce"
    )
    completed_ids = rank.groupby("country")["psa_sample_id"].nunique()
    rank["rank"] = pd.to_numeric(rank["rank"], errors="coerce")
    rank["total_child_adolescent_cases"] = pd.to_numeric(
        rank["total_child_adolescent_cases"], errors="coerce"
    )
    parameter_samples["psa_sample_id"] = pd.to_numeric(
        parameter_samples["psa_sample_id"], errors="coerce"
    )
    integer_sample_ids = bool(
        rank["psa_sample_id"].notna().all()
        and (rank["psa_sample_id"] % 1.0).eq(0.0).all()
        and parameter_samples["psa_sample_id"].notna().all()
        and (parameter_samples["psa_sample_id"] % 1.0).eq(0.0).all()
    )
    sample_ids_by_country = {
        str(country): frozenset(group["psa_sample_id"].astype(int).tolist())
        for country, group in rank.groupby("country")
    } if integer_sample_ids else {}
    shared_rank_sample_ids = bool(
        len(sample_ids_by_country) == len(countries)
        and len(set(sample_ids_by_country.values())) == 1
    )
    parameter_sample_ids = (
        frozenset(parameter_samples["psa_sample_id"].astype(int).tolist())
        if integer_sample_ids
        else frozenset()
    )
    rank_sample_ids = (
        next(iter(sample_ids_by_country.values()))
        if shared_rank_sample_ids
        else frozenset()
    )
    parameter_samples_match = bool(
        len(parameter_samples) == sample_size
        and parameter_samples["psa_sample_id"].nunique() == sample_size
        and parameter_sample_ids == rank_sample_ids
    )
    parameter_value_columns = [
        "sample_design",
        "uncertainty_schema_version",
        *EXPECTED_PARAMETER_NAMES,
    ]
    missing_parameter_values = sorted(
        set(parameter_value_columns).difference(rank.columns)
    )
    parameter_sample_values_match = False
    if parameter_samples_match and not missing_parameter_values:
        parameter_reference = parameter_samples.loc[
            :, ["psa_sample_id", *parameter_value_columns]
        ].copy()
        merged_parameters = rank.loc[
            :, ["psa_sample_id", *parameter_value_columns]
        ].merge(
            parameter_reference,
            on="psa_sample_id",
            how="left",
            suffixes=("_rank", "_sample"),
            validate="many_to_one",
        )
        value_checks: list[bool] = []
        numeric_parameter_columns = {
            "uncertainty_schema_version",
            *EXPECTED_PARAMETER_NAMES,
        }
        for column in parameter_value_columns:
            observed = merged_parameters[f"{column}_rank"]
            expected = merged_parameters[f"{column}_sample"]
            if column in numeric_parameter_columns:
                observed_numeric = pd.to_numeric(observed, errors="coerce")
                expected_numeric = pd.to_numeric(expected, errors="coerce")
                value_checks.append(
                    bool(
                        observed_numeric.notna().all()
                        and expected_numeric.notna().all()
                        and np.isfinite(observed_numeric.to_numpy(dtype=float)).all()
                        and np.isfinite(expected_numeric.to_numpy(dtype=float)).all()
                        and np.allclose(
                            observed_numeric.to_numpy(dtype=float),
                            expected_numeric.to_numpy(dtype=float),
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    )
                )
            else:
                value_checks.append(
                    bool(
                        observed.notna().all()
                        and expected.notna().all()
                        and observed.astype(str).eq(expected.astype(str)).all()
                    )
                )
        parameter_sample_values_match = bool(
            len(value_checks) == len(parameter_value_columns)
            and all(value_checks)
        )
    recomputed_rank = rank.groupby(
        ["country", "psa_sample_id"], dropna=False
    )["total_child_adolescent_cases"].rank(method="min", ascending=True)
    unrounded_ranks_match = bool(
        rank["total_child_adolescent_cases"].notna().all()
        and np.isfinite(rank["total_child_adolescent_cases"].to_numpy()).all()
        and rank["total_child_adolescent_cases"].ge(0.0).all()
        and np.allclose(
            rank["rank"].to_numpy(dtype=float),
            recomputed_rank.to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        )
    )
    rank_one = rank.loc[rank["rank"].eq(1)]
    blocks = rank.groupby(["country", "psa_sample_id"], dropna=False).agg(
        rows=("strategy", "size"),
        strategies=("strategy", "nunique"),
        ranks=("rank", "nunique"),
        rank_min=("rank", "min"),
        rank_max=("rank", "max"),
        rank_one=("rank", lambda values: int(values.eq(1).sum())),
    )
    if (
        len(rank) != expected_rows
        or set(rank["country"].astype(str)) != set(countries)
        or set(rank["strategy"].astype(str)) != set(programme_strategies)
        or rank.duplicated(["country", "psa_sample_id", "strategy"]).any()
        or len(completed_ids) != len(countries)
        or not completed_ids.eq(sample_size).all()
        or not integer_sample_ids
        or not shared_rank_sample_ids
        or not parameter_samples_match
        or not parameter_sample_values_match
        or not unrounded_ranks_match
        or len(rank_one) != len(countries) * sample_size
        or len(blocks) != len(countries) * sample_size
        or not blocks["rows"].eq(expected_programme_strategies).all()
        or not blocks["strategies"].eq(expected_programme_strategies).all()
        or not blocks["ranks"].eq(expected_programme_strategies).all()
        or not blocks["rank_min"].eq(1).all()
        or not blocks["rank_max"].eq(expected_programme_strategies).all()
        or not blocks["rank_one"].eq(1).all()
    ):
        raise ValueError(
            "Joint PSA completion audit failed; refusing to mark rank artifacts complete"
        )

    normalize_figure2b_interpretation_outputs()
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    metadata["generated_at_utc"] = completed_at_utc
    metadata["completed_at_utc"] = completed_at_utc
    metadata["run_status"] = "complete"
    metadata["completed_sample_count"] = int(sample_size)
    metadata["programme_only_strategies"] = list(programme_strategies)
    metadata["expected_under18_programme_rank_rows"] = int(expected_rows)
    metadata["row_counts"]["under18_programme_rank_samples"] = int(len(rank))
    metadata["completion_audit"] = {
        "exact_requested_samples_per_profile": True,
        "exact_one_rank1_strategy_per_profile_sample": True,
        "shared_sample_ids_across_profiles": True,
        "parameter_sample_ids_match_rank_parent": True,
        "parameter_sample_values_match_rank_parent": True,
        "ranks_recomputed_from_unrounded_burdens": True,
        "programme_strategy_count": expected_programme_strategies,
        "all_profiles_complete": True,
    }
    archived_runs = sorted(JOINT_ARCHIVE_ROOT.glob("run_*/manifest.json"))
    if archived_runs:
        metadata["archived_previous_run"] = str(archived_runs[-1].parent)
        metadata["archived_previous_runs"] = [
            str(manifest.parent) for manifest in archived_runs
        ]
    write_run_metadata(STEM, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--normalize-figure2b-interpretation-only",
        action="store_true",
        help=(
            "Normalize completed Figure 2b interpretation fields, rerun the "
            "completion audit, and refresh metadata without simulations."
        ),
    )
    args = parser.parse_args()

    configs = load_configs()
    countries = tuple(publication_country_names(configs))
    strategies = tuple(SELECTED_STRATEGIES)
    if args.normalize_figure2b_interpretation_only:
        _finalise_completed_metadata(
            sample_size=int(args.samples),
            countries=countries,
            strategies=strategies,
        )
        print("Normalized Figure 2b interpretation outputs and refreshed metadata.")
        return
    _validate_calibration_inputs(countries)
    current, reason = _resume_metadata_available()
    if current:
        try:
            resume_metadata = read_run_metadata(STEM)
        except (FileNotFoundError, OSError, UnicodeError, ValueError, TypeError) as exc:
            current = False
            reason = f"run metadata unavailable during design audit: {exc}"
        else:
            current, reason = _resume_design_compatibility(
                resume_metadata,
                sample_size=int(args.samples),
                seed=int(args.seed),
                countries=countries,
                strategies=strategies,
            )
    if not current:
        print(f"Starting a fresh joint PSA because {reason}")
        archived_previous_run = _archive_stale_outputs(
            reason=reason,
        )
        if archived_previous_run is not None:
            print(f"Archived the exact stale joint-PSA bundle at {archived_previous_run}")
        _initialise_checkpoint(
            sample_size=int(args.samples),
            seed=int(args.seed),
            sample_batch_size=int(args.sample_batch_size),
            countries=countries,
            strategies=strategies,
            archived_previous_run=archived_previous_run,
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
        primary_endpoint_outputs = (
            UNDER18_PROGRAMME_RANK_SAMPLE_PATH,
            UNDER18_PROGRAMME_ACCEPTABILITY_PATH,
            UNDER18_PROGRAMME_RUN_SUMMARY_PATH,
        )
        if len(matching) == int(args.samples) and all(
            Path(path).exists() or Path(path).with_suffix(".parquet").exists()
            for path in primary_endpoint_outputs
        ):
            print(
                f"Joint PSA already complete: {len(matching)}/{int(args.samples)} "
                "fixed-seed samples and primary-endpoint summaries match the requested design."
            )
            try:
                _finalise_completed_metadata(
                    sample_size=int(args.samples),
                    countries=countries,
                    strategies=strategies,
                )
            except ValueError as exc:
                print(
                    "Existing primary-endpoint derivatives failed the exact completion "
                    f"audit ({exc}); rebuilding them from the matching full-rank parent."
                )
            else:
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
    _finalise_completed_metadata(
        sample_size=int(args.samples),
        countries=countries,
        strategies=strategies,
    )


if __name__ == "__main__":
    main()
