from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from manuscript_notes import generate_lancet_child_adolescent_tables as tables


def test_frontier_only_never_reads_bootstrap_or_writes_table1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[str] = []
    writes: list[str] = []
    metadata_writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(tables, "load_configs", lambda: {})
    monkeypatch.setattr(
        tables,
        "publication_country_names",
        lambda _configs: ["Test_Profile"],
    )
    monkeypatch.setattr(
        tables,
        "_validate_parent_artifacts",
        lambda *_args, **_kwargs: None,
    )

    def fake_read(relative_path: str) -> pd.DataFrame:
        assert relative_path != tables.BOOTSTRAP_RELATIVE_PATH
        reads.append(relative_path)
        return pd.DataFrame({"source": [relative_path]})

    monkeypatch.setattr(tables, "_read", fake_read)
    monkeypatch.setattr(
        tables,
        "_augment_with_pediatric_metrics",
        lambda frame, **_kwargs: frame,
    )
    monkeypatch.setattr(
        tables,
        "_load_intervention_rows",
        lambda *_args: pd.DataFrame({"combined": [1]}),
    )
    burden = pd.DataFrame({"primary_case_metric": ["child_adolescent_0_17y"]})
    frontier = pd.DataFrame({"country": ["Test_Profile"], "strategy": ["current"]})
    preferred = pd.DataFrame({"country": ["Test_Profile"]})
    summary = pd.DataFrame({"strategy": ["current"]})
    monkeypatch.setattr(tables, "_strategy_burden_frame", lambda _frame: burden)
    monkeypatch.setattr(
        tables,
        "_frontier_and_preferred",
        lambda _frame: (frontier, preferred),
    )
    monkeypatch.setattr(tables, "_strategy_summary", lambda _frame: summary)
    monkeypatch.setattr(
        tables,
        "_write",
        lambda _frame, relative_path: writes.append(relative_path),
    )
    monkeypatch.setattr(
        tables,
        "current_run_metadata",
        lambda stem, **kwargs: {"stem": stem, **kwargs},
    )
    monkeypatch.setattr(
        tables,
        "write_run_metadata",
        lambda stem, metadata: metadata_writes.append((stem, metadata)),
    )

    observed = tables.generate_frontier_only()

    assert observed is frontier
    assert reads == [
        "outputs/summaries/intervention_scenarios_summary.csv",
        "outputs/summaries/vaccine_scenarios_summary.csv",
        "outputs/summaries/figure2_programme_reference_summary.csv",
    ]
    assert writes == [tables.FRONTIER_RELATIVE_PATH]
    assert tables.TABLE1_RELATIVE_PATH not in writes
    assert metadata_writes[0][0] == tables.FRONTIER_STEM
    assert metadata_writes[0][1]["bootstrap_independent"] is True
    assert metadata_writes[0][1]["reads_bootstrap_artifact"] is False


def test_frontier_parent_validation_requires_calibrations_and_parent_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    countries = tuple(f"Profile_{index}" for index in range(9))
    parent = tmp_path / "parent.csv"
    parent.write_text("country,value\nTest,1\n", encoding="utf-8")
    calibration_calls: list[tuple[str, ...]] = []
    metadata_calls: list[str] = []
    monkeypatch.setattr(
        tables,
        "validate_calibration_artifacts",
        lambda observed, **_kwargs: calibration_calls.append(tuple(observed)),
    )
    monkeypatch.setattr(tables, "_parent_artifact_paths", lambda _stem: (parent,))
    monkeypatch.setattr(
        tables,
        "validate_run_metadata",
        lambda stem: metadata_calls.append(stem) or {},
    )

    tables._validate_parent_artifacts(
        ("figure2_programme_reference",),
        countries=countries,
    )

    assert calibration_calls == [countries]
    assert metadata_calls == ["figure2_programme_reference"]

    parent.unlink()
    with pytest.raises(FileNotFoundError, match="parent.csv"):
        tables._validate_parent_artifacts(
            ("figure2_programme_reference",),
            countries=countries,
        )


def test_bootstrap_preflight_recomputes_and_matches_current_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame({"country": ["Test_Profile"], "strategy": ["current"]})
    observed = expected.copy()
    calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    monkeypatch.setattr(
        tables,
        "_build_frontier_products",
        lambda: SimpleNamespace(
            frontier=expected,
        ),
    )
    monkeypatch.setattr(
        tables,
        "_validated_frontier_artifact",
        lambda: observed,
    )
    monkeypatch.setattr(
        tables,
        "_assert_same_frontier",
        lambda expected_frame, observed_frame: calls.append(
            (expected_frame, observed_frame)
        ),
    )
    monkeypatch.setattr(
        tables,
        "_write",
        lambda *_args, **_kwargs: pytest.fail("frontier validation must be read-only"),
    )

    result = tables.validate_frontier_for_bootstrap()

    assert result is observed
    assert calls == [(expected, observed)]
