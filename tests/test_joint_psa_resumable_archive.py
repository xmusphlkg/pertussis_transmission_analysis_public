from __future__ import annotations

import json
from pathlib import Path

import pytest

from manuscript_notes import run_joint_psa_resumable as wrapper
from manuscript_notes import render_supplementary_tables as supplement_tables
from src_python.simulation.common import file_sha256


def _configure_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "project"
    metadata = root / "outputs" / "metadata" / "joint.json"
    table = root / "outputs" / "tables" / "joint.csv"
    simulation = root / "outputs" / "simulations" / "joint.parquet"
    archive_root = root / "outputs" / "archive" / "joint"
    monkeypatch.setattr(wrapper, "ROOT", root)
    monkeypatch.setattr(wrapper, "JOINT_METADATA_PATH", metadata)
    monkeypatch.setattr(wrapper, "JOINT_OUTPUT_PATHS", (table, simulation))
    monkeypatch.setattr(wrapper, "JOINT_ARCHIVE_ROOT", archive_root)
    return metadata, table, simulation, archive_root


def test_stale_joint_psa_bundle_is_archived_exactly_with_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, table, simulation, archive_root = _configure_bundle(
        tmp_path,
        monkeypatch,
    )
    table_parquet = table.with_suffix(".parquet")
    simulation_csv = simulation.with_suffix(".csv")
    unrelated = table.parent / "joint_pilot.csv"
    files = {
        metadata: b'{"config_hash":"old"}',
        table: b"sample,value\n1,2\n",
        table_parquet: b"table-parquet-placeholder",
        simulation: b"simulation-parquet-placeholder",
        simulation_csv: b"simulation-csv-placeholder",
    }
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    unrelated.write_text("must remain active", encoding="utf-8")
    expected_hashes = {str(path): file_sha256(path) for path in files}

    archive_dir = wrapper._archive_stale_outputs(
        reason="uncertainty_config_hash mismatch",
        old_fingerprint="old-fingerprint",
        new_fingerprint="new-fingerprint",
    )

    assert archive_dir is not None
    assert archive_dir.parent == archive_root
    assert unrelated.read_text(encoding="utf-8") == "must remain active"
    assert all(not path.exists() for path in files)
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    assert manifest["reason"] == "uncertainty_config_hash mismatch"
    assert manifest["schema_version"] == 1
    assert manifest["stem"] == wrapper.STEM
    assert manifest["old_fingerprint"] == "old-fingerprint"
    assert manifest["new_fingerprint"] == "new-fingerprint"
    assert len(manifest["files"]) == len(files)
    for record in manifest["files"]:
        original = Path(record["original_path"])
        archived = archive_dir / record["archive_path"]
        assert archived.read_bytes() == files[original]
        assert record["sha256"] == expected_hashes[str(original)]
        assert record["size_bytes"] == len(files[original])


def test_stale_joint_psa_archive_rolls_back_every_move_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, table, simulation, archive_root = _configure_bundle(
        tmp_path,
        monkeypatch,
    )
    files = {
        metadata: b"metadata",
        table: b"table",
        simulation: b"simulation",
    }
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    original_replace = Path.replace
    failed = False

    def fail_once(source: Path, target: Path) -> Path:
        nonlocal failed
        if source == simulation and not failed:
            failed = True
            raise OSError("injected move failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(RuntimeError, match="Could not archive stale joint PSA outputs"):
        wrapper._archive_stale_outputs(
            reason="test rollback",
            old_fingerprint="old",
            new_fingerprint="new",
        )

    assert failed
    assert all(path.read_bytes() == payload for path, payload in files.items())
    assert not any(archive_root.glob("run_*"))


def test_stale_joint_psa_archive_is_noop_when_no_active_bundle_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata, _table, _simulation, archive_root = _configure_bundle(
        tmp_path,
        monkeypatch,
    )
    assert (
        wrapper._archive_stale_outputs(
            reason="metadata unavailable",
            old_fingerprint="",
            new_fingerprint="new",
        )
        is None
    )
    assert not archive_root.exists()


def test_stale_joint_psa_archive_rolls_back_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, table, simulation, archive_root = _configure_bundle(
        tmp_path,
        monkeypatch,
    )
    files = {metadata: b"metadata", table: b"table", simulation: b"simulation"}
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    original_replace = Path.replace

    def interrupt(source: Path, target: Path) -> Path:
        if source == simulation:
            raise KeyboardInterrupt
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        wrapper._archive_stale_outputs(
            reason="test interrupt",
            old_fingerprint="old",
            new_fingerprint="new",
        )

    assert all(path.read_bytes() == payload for path, payload in files.items())
    assert not any(archive_root.glob("run_*"))


def test_stale_joint_psa_archive_rolls_back_failed_hash_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, table, simulation, archive_root = _configure_bundle(
        tmp_path,
        monkeypatch,
    )
    files = {metadata: b"metadata", table: b"table", simulation: b"simulation"}
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    original_hash = file_sha256

    def corrupt_archive_hash(path: Path) -> str:
        digest = original_hash(path)
        return "0" * 64 if "archive" in path.parts else digest

    monkeypatch.setattr(wrapper, "file_sha256", corrupt_archive_hash)
    with pytest.raises(RuntimeError, match="Could not archive stale joint PSA outputs"):
        wrapper._archive_stale_outputs(
            reason="test verification",
            old_fingerprint="old",
            new_fingerprint="new",
        )

    assert all(path.read_bytes() == payload for path, payload in files.items())
    assert not any(archive_root.glob("run_*"))


def test_joint_psa_resume_requires_exact_statistical_design() -> None:
    countries = ("Country A", "Country B")
    strategies = tuple(wrapper.SELECTED_STRATEGIES)
    metadata: dict[str, object] = {
        "sample_size_requested": 128,
        "sample_seed": 20260521,
        "countries": list(countries),
        "strategies": list(strategies),
        "sample_design": wrapper.SAMPLE_DESIGN,
        "uncertainty_schema_version": wrapper.UNCERTAINTY_SCHEMA_VERSION,
        "smoke_runtime": False,
        "keep_timeseries": False,
        "sample_batch_size": 99,
        "input_artifact_path_sha256": {"outputs/calibrations/A.yaml": "a" * 64},
    }
    compatible, _ = wrapper._resume_design_compatibility(
        metadata,
        sample_size=128,
        seed=20260521,
        countries=countries,
        strategies=strategies,
        calibration_input_hashes={"outputs/calibrations/A.yaml": "a" * 64},
    )
    assert compatible

    mutations: dict[str, object] = {
        "sample_size_requested": 64,
        "sample_seed": 1,
        "countries": ["Country B", "Country A"],
        "strategies": list(reversed(strategies)),
        "sample_design": "other",
        "uncertainty_schema_version": 999,
        "smoke_runtime": True,
        "keep_timeseries": True,
    }
    for key, value in mutations.items():
        stale = dict(metadata)
        stale[key] = value
        compatible, reason = wrapper._resume_design_compatibility(
            stale,
            sample_size=128,
            seed=20260521,
            countries=countries,
            strategies=strategies,
            calibration_input_hashes={"outputs/calibrations/A.yaml": "a" * 64},
        )
        assert not compatible
        assert key in reason

    compatible, reason = wrapper._resume_design_compatibility(
        metadata,
        sample_size=128,
        seed=20260521,
        countries=countries,
        strategies=strategies,
        calibration_input_hashes={"outputs/calibrations/A.yaml": "b" * 64},
    )
    assert not compatible
    assert "input_artifact_path_sha256" in reason


def test_joint_contract_fingerprint_changes_with_calibration_artifact() -> None:
    common = {
        "config_hash": "config",
        "uncertainty_hash": "uncertainty",
        "source_hash": "source",
    }
    first = wrapper._joint_contract_fingerprint(
        **common,
        calibration_input_hashes={"outputs/calibrations/A.yaml": "a" * 64},
    )
    second = wrapper._joint_contract_fingerprint(
        **common,
        calibration_input_hashes={"outputs/calibrations/A.yaml": "b" * 64},
    )
    assert first != second


def test_supplement_table_reader_rejects_legacy_seventh_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        "run_status": "complete",
        "figure2b_parameter_names": list(wrapper.EXPECTED_PARAMETER_NAMES),
        "excluded_dead_dimensions": ["resistance_management_uptake"],
    }
    monkeypatch.setattr(supplement_tables, "ROOT", tmp_path)
    monkeypatch.setattr(
        supplement_tables,
        "validate_run_metadata",
        lambda _stem: metadata,
    )
    path = tmp_path / "outputs" / "tables" / "joint_psa_parameter_samples.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "psa_sample_id",
        "sample_design",
        "uncertainty_schema_version",
        *wrapper.EXPECTED_PARAMETER_NAMES,
        "resistance_management_uptake",
    ]
    path.write_text(",".join(columns) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seven-dimensional"):
        supplement_tables.joint_psa_parameter_rows()


def test_supplement_table_reader_accepts_exact_current_six_input_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        "run_status": "complete",
        "figure2b_parameter_names": list(wrapper.EXPECTED_PARAMETER_NAMES),
        "excluded_dead_dimensions": ["resistance_management_uptake"],
    }
    monkeypatch.setattr(supplement_tables, "ROOT", tmp_path)
    monkeypatch.setattr(
        supplement_tables,
        "validate_run_metadata",
        lambda _stem: metadata,
    )
    path = tmp_path / "outputs" / "tables" / "joint_psa_parameter_samples.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "psa_sample_id",
        "sample_design",
        "uncertainty_schema_version",
        *wrapper.EXPECTED_PARAMETER_NAMES,
    ]
    values = ["1", wrapper.SAMPLE_DESIGN, "1", *("0.5" for _ in wrapper.EXPECTED_PARAMETER_NAMES)]
    path.write_text(
        ",".join(columns) + "\n" + ",".join(values) + "\n",
        encoding="utf-8",
    )

    rows = supplement_tables.joint_psa_parameter_rows()
    assert len(rows) == 1
    assert set(rows[0]) == set(columns)
