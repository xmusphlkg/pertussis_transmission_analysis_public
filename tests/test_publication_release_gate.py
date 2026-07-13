from __future__ import annotations

import pandas as pd
import pytest

from src_python.validation import publication_gate
from src_python.validation.publication_gate import (
    FIGURE2_REQUIRED_AUDIT_CHECKS,
    predictive_publication_gate_failures,
)
from src_python.simulation.common import file_sha256


def _passing_gate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "execution_complete": True,
                "data_integrity_gate_pass": True,
                "countries_completed": 3,
                "minimum_folds_per_country": 3,
                "model_beats_all_baselines_log_score": True,
                "primary_predictive_95_coverage": 0.90,
                "primary_predictive_mean_log1p_width": 2.8,
                "primary_predictive_scope": (
                    "country_balanced_one_interval_ahead_prequential"
                ),
                "predictive_width_gate_pass": True,
                "mc_replicates": 3,
                "monte_carlo_stability_pass": True,
                "predictive_gate_pass": True,
                "publication_gate_pass": True,
                "claim_scope": (
                    "notification-index forecasting and conditional scenario projections"
                ),
                "full_compartment_pomp_claimed": False,
                "posterior_parameter_uncertainty_claimed": False,
            }
        ]
    )


def _passing_metadata() -> dict[str, object]:
    return {
        "countries_included": ["A", "B", "C"],
        "data_integrity_gate_pass": True,
        "predictive_gate_pass": True,
        "publication_gate_pass": True,
        "forecast_mode": "prequential",
        "scope": "semi_mechanistic_discrepancy_pomp_not_full_compartment_pomp",
        "full_compartment_pomp_claimed": False,
        "posterior_parameter_uncertainty_claimed": False,
        "particles": 512,
        "predictive_draws": 4096,
        "mc_replicates": 3,
        "test_years": [2023, 2024, 2025, 2026],
        "fold_counts": {"A": 3, "B": 3, "C": 3},
        "fold_test_years": {
            "A": [2023, 2024, 2025],
            "B": [2023, 2024, 2025],
            "C": [2023, 2024, 2025],
        },
        "row_counts": {
            "panel_pomp_rolling_hindcast_folds": 9,
            "panel_pomp_rolling_hindcast_intervals": 100,
        },
        "input_artifact_sha256": {
            "pertussis_incidence_timeseries": "a" * 64,
        },
        "output_artifact_sha256": {
            "panel_pomp_rolling_hindcast_gate": "b" * 64,
            "panel_pomp_rolling_hindcast_folds": "c" * 64,
            "panel_pomp_rolling_hindcast_intervals": "d" * 64,
        },
    }


def test_predictive_publication_parent_requires_complete_scope_and_counts() -> None:
    failures = predictive_publication_gate_failures(
        _passing_gate(),
        _passing_metadata(),
        expected_countries=["A", "B", "C"],
    )

    assert failures == []


def test_predictive_publication_parent_fails_closed_on_scope_or_country_drift() -> None:
    gate = _passing_gate()
    gate.loc[0, "posterior_parameter_uncertainty_claimed"] = True
    metadata = _passing_metadata()
    metadata["countries_included"] = ["A", "B"]

    failures = predictive_publication_gate_failures(
        gate,
        metadata,
        expected_countries=["A", "B", "C"],
    )

    assert any("posterior_parameter_uncertainty_claimed=true" in failure for failure in failures)
    assert any("do not match publication countries" in failure for failure in failures)


def test_predictive_publication_parent_rejects_string_false_and_fractional_counts() -> None:
    gate = _passing_gate()
    gate = gate.astype(
        {"countries_completed": float, "minimum_folds_per_country": float}
    )
    gate.loc[0, "countries_completed"] = 1.5
    gate.loc[0, "minimum_folds_per_country"] = 1.5
    metadata = _passing_metadata()
    metadata["publication_gate_pass"] = "false"

    failures = predictive_publication_gate_failures(
        gate,
        metadata,
        expected_countries=["A", "B", "C"],
    )

    assert any("positive integer" in failure for failure in failures)
    assert any("publication_gate_pass=false" in failure for failure in failures)


def test_predictive_publication_parent_cross_checks_metadata_fold_rows() -> None:
    metadata = _passing_metadata()
    metadata["row_counts"] = {
        "panel_pomp_rolling_hindcast_folds": 9,
        "panel_pomp_rolling_hindcast_intervals": 0,
    }
    metadata["fold_test_years"] = {
        "A": [2023, 2024, 2025],
        "B": [2023, 2024, 2025],
        "C": [2023, 2024],
    }

    failures = predictive_publication_gate_failures(
        _passing_gate(),
        metadata,
        expected_countries=["A", "B", "C"],
    )

    assert any("fold years are incomplete for C" in failure for failure in failures)
    assert any("row_counts.panel_pomp_rolling_hindcast_intervals" in failure for failure in failures)


def test_figure2_parent_requires_fresh_digest_and_complete_recommended_checks(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "figure2_audit.csv"
    audit = pd.DataFrame(
        {
            "check": sorted(FIGURE2_REQUIRED_AUDIT_CHECKS),
            "status": "pass",
            "severity": "fatal",
        }
    )
    audit.to_csv(path, index=False)
    artifact_paths = {
        name: tmp_path / f"{name}.artifact"
        for name in publication_gate.FIGURE2_AUDITED_ARTIFACT_PATHS
    }
    for name, artifact_path in artifact_paths.items():
        artifact_path.write_text(name, encoding="utf-8")
    metadata = {
        "passed": True,
        "warnings_are_fatal": True,
        "posterior_stem": publication_gate.FIGURE2_POSTERIOR_STEM,
        "figure2c_stem": publication_gate.FIGURE2_SOURCE_STEM,
        "audit_table_sha256": file_sha256(path),
        "audited_artifact_sha256": {
            name: file_sha256(artifact_path)
            for name, artifact_path in artifact_paths.items()
        },
    }
    monkeypatch.setattr(publication_gate, "FIGURE2_AUDIT_PATH", path)
    monkeypatch.setattr(
        publication_gate,
        "FIGURE2_AUDITED_ARTIFACT_PATHS",
        artifact_paths,
    )
    monkeypatch.setattr(
        publication_gate,
        "require_predictive_publication_gate",
        lambda **_kwargs: {"publication_gate_pass": True},
    )
    monkeypatch.setattr(
        publication_gate,
        "validate_run_metadata",
        lambda _stem: metadata,
    )

    result = publication_gate.require_figure2_publication_gate(
        expected_countries=["A"]
    )
    assert result["figure2_audit_metadata"] == metadata

    audit.loc[0, "status"] = None
    audit.to_csv(path, index=False)
    metadata["audit_table_sha256"] = file_sha256(path)
    with pytest.raises(RuntimeError, match="invalid status"):
        publication_gate.require_figure2_publication_gate(
            expected_countries=["A"]
        )
