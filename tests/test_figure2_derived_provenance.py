from __future__ import annotations

import pandas as pd
import pytest

from manuscript_notes import validate_publication_outputs as validator


def _write_figure2_bundle(tmp_path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for parent, (stem, relative_path) in validator.FIGURE2_PARENT_ARTIFACTS.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"canonical parent {parent}\n", encoding="utf-8")
        metadata[stem] = {}

    provenance_rows = []
    for parent, (stem, _) in validator.FIGURE2_PARENT_ARTIFACTS.items():
        provenance_rows.append(
            {
                "parent": parent,
                "estimand_or_role": f"role for {parent}",
            }
        )
    provenance_path = tmp_path / validator.FIGURE2_PROVENANCE_PATH
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(provenance_rows).to_csv(provenance_path, index=False)
    for relative_path in validator.FIGURE2_DERIVED_SOURCE_TABLES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"value": [1]}).to_csv(path, index=False)
    return metadata


def test_figure2_derived_tables_require_all_four_parents_and_nonempty_tables(
    tmp_path, monkeypatch
) -> None:
    metadata = _write_figure2_bundle(tmp_path)
    monkeypatch.setattr(
        validator, "project_path", lambda *parts: tmp_path.joinpath(*parts)
    )

    assert validator.validate_figure2_derived_provenance(metadata) is None


def test_figure2_derived_gate_rejects_an_empty_table(
    tmp_path, monkeypatch
) -> None:
    metadata = _write_figure2_bundle(tmp_path)
    monkeypatch.setattr(
        validator, "project_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    empty_path = tmp_path / validator.FIGURE2_DERIVED_SOURCE_TABLES[2]
    pd.DataFrame(columns=["value"]).to_csv(empty_path, index=False)

    with pytest.raises(AssertionError, match="source table is empty"):
        validator.validate_figure2_derived_provenance(metadata)


def test_figure2_derived_gate_rejects_missing_estimand_role(
    tmp_path, monkeypatch
) -> None:
    metadata = _write_figure2_bundle(tmp_path)
    monkeypatch.setattr(
        validator, "project_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    provenance_path = tmp_path / validator.FIGURE2_PROVENANCE_PATH
    provenance = pd.read_csv(provenance_path)
    provenance.loc[0, "estimand_or_role"] = ""
    provenance.to_csv(provenance_path, index=False)

    with pytest.raises(AssertionError, match="no estimand or role"):
        validator.validate_figure2_derived_provenance(metadata)
