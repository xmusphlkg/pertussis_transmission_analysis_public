from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from manuscript_notes import generate_figure2_programme_reference as reference_generator
from manuscript_notes.generate_figure2_programme_reference import (
    PRIMARY_RATE,
    PRIMARY_REDUCTION,
    PRIMARY_TOTAL,
    STRATEGIES,
    _canonicalise_and_validate,
)


def _reference_rows() -> pd.DataFrame:
    burdens = {
        "current": 200.0,
        "timeliness_only": 170.0,
        "maternal_immunization": 160.0,
        "pregnancy_tdap_scaleup": 190.0,
        "adolescent_booster": 195.0,
        "cocooning_adjunct": 165.0,
        "targeted_pep_high_risk": 180.0,
    }
    return pd.DataFrame(
        {
            "country": "Test_Profile",
            "scenario": list(STRATEGIES),
            "strategy": list(STRATEGIES),
            PRIMARY_RATE: [burdens[strategy] for strategy in STRATEGIES],
            PRIMARY_TOTAL: [1000.0 * burdens[strategy] for strategy in STRATEGIES],
            # Deliberately contaminated source values must never survive.
            PRIMARY_REDUCTION: np.repeat(0.999, len(STRATEGIES)),
        }
    )


def test_reference_parent_recomputes_one_common_denominator_and_preserves_ranking() -> None:
    result = _canonicalise_and_validate(
        _reference_rows(), countries=("Test_Profile",)
    )
    assert result["current_primary_cases_per_100k"].nunique() == 1
    assert result["current_primary_cases_per_100k"].iat[0] == pytest.approx(200.0)
    observed = result.set_index("strategy")[PRIMARY_REDUCTION]
    assert observed["current"] == pytest.approx(0.0)
    assert observed["maternal_immunization"] == pytest.approx(0.2)
    programme = result.loc[result["strategy"].ne("current")]
    assert programme.loc[programme[PRIMARY_RATE].idxmin(), "strategy"] == (
        programme.loc[programme[PRIMARY_REDUCTION].idxmax(), "strategy"]
    )


def test_reference_parent_rejects_missing_strategy_row() -> None:
    incomplete = _reference_rows().iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly one row"):
        _canonicalise_and_validate(incomplete, countries=("Test_Profile",))


def _nine_country_reference_rows(countries: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat(
        [_reference_rows().assign(country=country) for country in countries],
        ignore_index=True,
    )


def _patch_reference_generation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calibration_results: list[dict[str, str]],
) -> tuple[list[object], list[tuple[str, dict]]]:
    countries = tuple(f"Profile_{index}" for index in range(9))
    summary = _nine_country_reference_rows(countries)
    timeseries = pd.DataFrame({"row": [1, 2, 3]})
    writes: list[object] = []
    metadata_writes: list[tuple[str, dict]] = []
    calibration_calls = iter(calibration_results)

    monkeypatch.setattr(reference_generator, "load_configs", lambda: {})
    monkeypatch.setattr(
        reference_generator,
        "publication_country_names",
        lambda _configs: list(countries),
    )
    monkeypatch.setattr(reference_generator, "_build_scenarios", lambda _countries: [{}])
    monkeypatch.setattr(
        reference_generator,
        "execute_scenario_list",
        lambda *_args, **_kwargs: (timeseries.copy(), summary.copy()),
    )
    monkeypatch.setattr(
        reference_generator,
        "add_relative_reductions",
        lambda frame, **_kwargs: frame,
    )
    monkeypatch.setattr(
        reference_generator,
        "enforce_calibration_status",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        reference_generator,
        "validated_calibration_artifact_path_hashes",
        lambda *_args, **_kwargs: next(calibration_calls),
    )
    monkeypatch.setattr(
        reference_generator,
        "write_dataframe",
        lambda _frame, path: writes.append(path),
    )
    monkeypatch.setattr(
        reference_generator,
        "current_run_metadata",
        lambda stem, **kwargs: {"stem": stem, **kwargs},
    )
    monkeypatch.setattr(
        reference_generator,
        "file_sha256",
        lambda path: f"sha256:{path.name}",
    )
    monkeypatch.setattr(
        reference_generator,
        "write_run_metadata",
        lambda stem, metadata: metadata_writes.append((stem, metadata)),
    )
    return writes, metadata_writes


def test_reference_generate_records_exact_nine_calibration_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_hashes = {
        f"outputs/calibrations/Profile_{index}_calibrated_config.yaml": f"hash-{index}"
        for index in range(9)
    }
    writes, metadata_writes = _patch_reference_generation(
        monkeypatch,
        calibration_results=[calibration_hashes, calibration_hashes.copy()],
    )

    result = reference_generator.generate(n_jobs=3)

    assert len(result) == 9 * len(STRATEGIES)
    assert writes == [reference_generator.TIMESERIES_PATH, reference_generator.OUTPUT_PATH]
    assert len(metadata_writes) == 1
    stem, metadata = metadata_writes[0]
    assert stem == reference_generator.STEM
    assert metadata["input_artifact_path_sha256"] == calibration_hashes
    assert len(metadata["input_artifact_path_sha256"]) == 9
    assert metadata["row_counts"] == {
        "timeseries": 3,
        "summary": 9 * len(STRATEGIES),
    }
    assert metadata["output_artifact_sha256"] == {
        "timeseries": f"sha256:{reference_generator.TIMESERIES_PATH.name}",
        "summary": f"sha256:{reference_generator.OUTPUT_PATH.name}",
    }


def test_reference_generate_rejects_calibration_change_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {
        f"outputs/calibrations/Profile_{index}_calibrated_config.yaml": f"hash-{index}"
        for index in range(9)
    }
    after = {**before, next(iter(before)): "changed"}
    writes, metadata_writes = _patch_reference_generation(
        monkeypatch,
        calibration_results=[before, after],
    )

    with pytest.raises(RuntimeError, match="changed while"):
        reference_generator.generate(n_jobs=2)

    assert writes == []
    assert metadata_writes == []
