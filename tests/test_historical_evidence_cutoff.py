from __future__ import annotations

import pandas as pd
import pytest

from src_python.simulation.common import make_config


def _evidence_years(config: dict) -> list[int]:
    raw = str(config.get("metadata", {}).get("resistance_timeline_evidence_years", ""))
    return [int(value) for value in raw.split(";") if value.strip()]


def test_historical_cutoff_excludes_same_year_resistance_and_diagnostic_regimes() -> None:
    config = make_config(
        country_profile="China",
        load_calibration=False,
        evidence_cutoff_date="2024-01-01",
    )

    metadata = config["metadata"]
    assert metadata["evidence_cutoff_date"] == "2024-01-01"
    assert metadata["resistance_timeline_anchor_year"] <= 2023
    assert all(year <= 2023 for year in _evidence_years(config))

    diagnostic_starts = {
        pd.Timestamp(period["start_date"])
        for period in config["diagnostic_reporting_time_variation"]["periods"]
    }
    assert diagnostic_starts
    assert max(diagnostic_starts) < pd.Timestamp("2024-01-01")
    assert pd.Timestamp("2024-04-01") not in diagnostic_starts


def test_historical_cutoff_keeps_generic_resistance_when_no_past_evidence_exists() -> None:
    baseline = make_config(country_profile="China", load_calibration=False)
    historical = make_config(
        country_profile="China",
        load_calibration=False,
        evidence_cutoff_date="2010-01-01",
    )

    assert historical["metadata"]["resistance_timeline_applied"] is False
    assert (
        historical["resistance"]["country_timeline"]["method"]
        == "no_evidence_at_or_before_cutoff"
    )
    # The fallback is the prespecified generic resistance scenario, not a
    # back-cast of China's first later resistance measurement.
    assert historical["resistance"]["target_prevalence_at_analysis_start"] != (
        baseline["resistance"]["target_prevalence_at_analysis_start"]
    )


def test_historical_cutoff_excludes_npi_that_had_not_started() -> None:
    config = make_config(
        country_profile="Australia",
        load_calibration=False,
        evidence_cutoff_date="2020-01-01",
    )

    assert config["transmission"]["npi_contact_reduction_periods"] == []
    assert (
        config["metadata"]["npi_contact_reduction_evidence_cutoff_date"]
        == "2020-01-01"
    )


def test_historical_cutoff_refuses_future_fitted_calibration_artifact() -> None:
    with pytest.raises(ValueError, match="cannot load a calibration artifact"):
        make_config(
            country_profile="Australia",
            evidence_cutoff_date="2024-01-01",
        )


def test_default_current_evidence_contract_is_unchanged() -> None:
    config = make_config(country_profile="China", load_calibration=False)

    assert config["metadata"]["resistance_timeline_anchor_year"] == 2025
    diagnostic_starts = {
        period["start_date"]
        for period in config["diagnostic_reporting_time_variation"]["periods"]
    }
    assert "2024-04-01" in diagnostic_starts


def test_historical_cutoff_removes_outcome_profile_terms_and_lags_dtp_coverage() -> None:
    historical = make_config(
        country_profile="China",
        load_calibration=False,
        evidence_cutoff_date="2024-01-01",
    )

    assert historical["transmission"]["seasonal_amplitude"] == 0.0
    assert historical["transmission"]["seasonal_phase"] == 0.0
    assert historical["transmission"]["multi_year_amplitude"] == 0.0
    assert historical["metadata"]["outcome_derived_profile_terms_disabled"] is True
    assert historical["metadata"]["historical_dtp_coverage_year"] <= 2022
    assert historical["metadata"]["historical_maternal_coverage_excluded"] is True
    assert historical["metadata"]["npi_contact_reduction_reduction_column"] == (
        "contact_reduction_mean"
    )
