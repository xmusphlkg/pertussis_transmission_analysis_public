from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from manuscript_notes import run_joint_psa_resumable as wrapper


def _fixture_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    acceptability = pd.DataFrame(
        {
            "country": ["Country_B", "Country_A", "Country_A"],
            "strategy": ["strategy_2", "strategy_1", "strategy_2"],
            "rank": [1, 1, 2],
            "frequency_rank_1": [0.25, 0.75, 0.25],
            "n_psa_samples": [128, 128, 128],
            "interpretation": [
                "Selected-parameter deterministic frequency; not a posterior probability."
            ]
            * 3,
        }
    )
    summary = acceptability.loc[acceptability["rank"].eq(1)].copy()
    return acceptability, summary


def _write_variants(frame: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(csv_path.with_suffix(".parquet"), index=False)


def _configure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    acceptability_path = tmp_path / "tables" / "acceptability.csv"
    summary_path = tmp_path / "summaries" / "summary.csv"
    acceptability, summary = _fixture_tables()
    _write_variants(acceptability, acceptability_path)
    _write_variants(summary, summary_path)
    monkeypatch.setattr(
        wrapper,
        "UNDER18_PROGRAMME_ACCEPTABILITY_PATH",
        acceptability_path,
    )
    monkeypatch.setattr(
        wrapper,
        "UNDER18_PROGRAMME_RUN_SUMMARY_PATH",
        summary_path,
    )
    return acceptability_path, summary_path, acceptability, summary


def test_normalize_figure2b_interpretation_is_pure_and_order_preserving() -> None:
    acceptability, _summary = _fixture_tables()
    original = acceptability.copy(deep=True)

    normalized = wrapper._normalize_figure2b_interpretation(acceptability)

    pd.testing.assert_frame_equal(acceptability, original)
    pd.testing.assert_frame_equal(
        normalized.drop(columns="interpretation"),
        original.drop(columns="interpretation"),
    )
    assert normalized.index.equals(original.index)
    assert normalized["interpretation"].eq(wrapper.FIGURE2B_INTERPRETATION).all()


def test_normalize_figure2b_outputs_changes_only_interpretation_and_matches_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptability_path, summary_path, acceptability, summary = _configure_paths(
        tmp_path,
        monkeypatch,
    )

    wrapper.normalize_figure2b_interpretation_outputs()

    for csv_path, original in (
        (acceptability_path, acceptability),
        (summary_path, summary),
    ):
        csv_frame = pd.read_csv(csv_path)
        parquet_frame = pd.read_parquet(csv_path.with_suffix(".parquet"))
        pd.testing.assert_frame_equal(csv_frame, parquet_frame, check_dtype=False)
        pd.testing.assert_frame_equal(
            csv_frame.drop(columns="interpretation"),
            original.drop(columns="interpretation"),
            check_dtype=False,
        )
        assert csv_frame["interpretation"].eq(
            wrapper.FIGURE2B_INTERPRETATION
        ).all()


def test_normalize_figure2b_outputs_rolls_back_all_variants_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptability_path, summary_path, _acceptability, _summary = _configure_paths(
        tmp_path,
        monkeypatch,
    )
    targets = (
        acceptability_path,
        acceptability_path.with_suffix(".parquet"),
        summary_path,
        summary_path.with_suffix(".parquet"),
    )
    original_bytes = {path: path.read_bytes() for path in targets}
    original_replace = Path.replace
    failed = False

    def fail_once(source: Path, target: Path) -> Path:
        nonlocal failed
        if (
            not failed
            and source.name.endswith(".tmp.csv")
            and target == summary_path
        ):
            failed = True
            raise OSError("injected interpretation replacement failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(
        RuntimeError,
        match="Could not atomically normalize Figure 2b interpretation",
    ):
        wrapper.normalize_figure2b_interpretation_outputs()

    assert failed
    assert all(path.read_bytes() == original_bytes[path] for path in targets)
    assert not list(tmp_path.rglob("*.tmp.csv"))
    assert not list(tmp_path.rglob("*.tmp.parquet"))
    assert not list(tmp_path.rglob("*.bak"))


def test_extended_data_figure9_uses_selected_input_design_frequency_axis() -> None:
    panels = Path("scripts_R/extended_data/figure_9/panels.R").read_text(
        encoding="utf-8"
    )

    assert 'x = "Selected-input design frequency"' in panels
    assert "Selected-parameter frequency" not in panels
