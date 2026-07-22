from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from manuscript_notes import validate_publication_outputs as validator


def _write_parent_bound_figure2_bundle(tmp_path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    parent_digests: dict[str, str] = {}
    for index, (parent, (stem, relative_path, digest_key)) in enumerate(
        validator.FIGURE2_PARENT_ARTIFACTS.items(), start=1
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"canonical parent {parent}\n", encoding="utf-8")
        digest = validator.file_sha256(path)
        parent_digests[parent] = digest
        record = {
            "config_hash": "config-current",
            "source_code_hash": "source-current",
            "git": {"commit": "commit-current"},
            "generated_at_utc": f"2026-07-{index:02d}T00:00:00+00:00",
        }
        if digest_key is not None:
            record["output_artifact_sha256"] = {digest_key: digest}
        metadata[stem] = record

    bundle = validator._figure2_parent_bundle_sha256(parent_digests)
    provenance_rows = []
    for parent, (stem, _, _) in validator.FIGURE2_PARENT_ARTIFACTS.items():
        record = metadata[stem]
        provenance_rows.append(
            {
                "parent": parent,
                "config_hash": record["config_hash"],
                "source_code_hash": record["source_code_hash"],
                "git_commit": record["git"]["commit"],
                "artifact_sha256": parent_digests[parent],
                "generated_at_utc": record["generated_at_utc"],
                "estimand_or_role": f"role for {parent}",
                validator.FIGURE2_PARENT_BUNDLE_COLUMN: bundle,
            }
        )
    provenance_path = tmp_path / validator.FIGURE2_PROVENANCE_PATH
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(provenance_rows).to_csv(provenance_path, index=False)
    for relative_path in validator.FIGURE2_DERIVED_SOURCE_TABLES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "value": [1],
                validator.FIGURE2_PARENT_BUNDLE_COLUMN: [bundle],
            }
        ).to_csv(path, index=False)
    return metadata


def test_figure2_derived_tables_are_bound_to_all_four_current_parents(
    tmp_path, monkeypatch
) -> None:
    metadata = _write_parent_bound_figure2_bundle(tmp_path)
    monkeypatch.setattr(
        validator, "project_path", lambda *parts: tmp_path.joinpath(*parts)
    )

    observed = validator.validate_figure2_derived_provenance(metadata)

    assert len(observed) == 64


def test_figure2_derived_gate_rejects_cli_refreshed_parent_with_old_tables(
    tmp_path, monkeypatch
) -> None:
    metadata = _write_parent_bound_figure2_bundle(tmp_path)
    monkeypatch.setattr(
        validator, "project_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    refreshed = deepcopy(metadata)
    _, relative_path, digest_key = validator.FIGURE2_PARENT_ARTIFACTS["reference"]
    parent_path = tmp_path / relative_path
    parent_path.write_text("refreshed reference parent\n", encoding="utf-8")
    refreshed["figure2_programme_reference"]["output_artifact_sha256"][
        digest_key
    ] = validator.file_sha256(parent_path)
    refreshed["figure2_programme_reference"][
        "generated_at_utc"
    ] = "2026-07-20T00:00:00+00:00"

    with pytest.raises(AssertionError, match="stale (artifact_sha256|generated_at_utc)"):
        validator.validate_figure2_derived_provenance(refreshed)


def test_figure2_derived_gate_rejects_one_table_with_an_old_bundle(
    tmp_path, monkeypatch
) -> None:
    metadata = _write_parent_bound_figure2_bundle(tmp_path)
    monkeypatch.setattr(
        validator, "project_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    stale_path = tmp_path / validator.FIGURE2_DERIVED_SOURCE_TABLES[2]
    stale = pd.read_csv(stale_path)
    stale[validator.FIGURE2_PARENT_BUNDLE_COLUMN] = "0" * 64
    stale.to_csv(stale_path, index=False)

    with pytest.raises(AssertionError, match="stale parent bundle"):
        validator.validate_figure2_derived_provenance(metadata)


def test_figure2_derived_gate_rejects_old_provenance_after_source_reattestation(
    tmp_path, monkeypatch
) -> None:
    metadata = _write_parent_bound_figure2_bundle(tmp_path)
    monkeypatch.setattr(
        validator, "project_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    refreshed = deepcopy(metadata)
    for record in refreshed.values():
        record["source_code_hash"] = "source-reattested"

    with pytest.raises(AssertionError, match="stale source_code_hash"):
        validator.validate_figure2_derived_provenance(refreshed)
