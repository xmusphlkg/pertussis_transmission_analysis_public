from __future__ import annotations

from pathlib import Path

import pytest

from src_python.utils import validation


EXPECTED_SIMULATE_SUMMARY_STEMS = (
    "baseline_timeseries",
    "vaccine_scenarios",
    "resistance_scenarios",
    "reporting_scenarios",
    "country_scenarios",
    "veinf_resistance_grid",
    "fitness_resistance_grid",
    "intervention_scenarios",
    "routine_timeliness_sensitivity",
    "sensitivity_runs",
    "immunity_sensitivity",
    "resistance_fitness_sensitivity",
)
EXPECTED_SIMULATE_RUNNER_FRAGMENTS = {
    "baseline_timeseries": "run_baseline",
    "country_scenarios": "run_country_scenarios",
    "vaccine_scenarios": "run_vaccine_scenarios",
    "resistance_scenarios": "run_resistance_scenarios",
    "reporting_scenarios": "run_reporting_scenarios",
    "veinf_resistance_grid": "run_heatmap_grid",
    "fitness_resistance_grid": "run_fitness_grid",
    "intervention_scenarios": "run_intervention_scenarios",
    "routine_timeliness_sensitivity": "run_routine_timeliness_sensitivity",
    "sensitivity_runs": "run_sensitivity",
    "immunity_sensitivity": "run_immunity_sensitivity",
    "resistance_fitness_sensitivity": "run_resistance_fitness_sensitivity",
}
EXPECTED_PUBLICATION_METADATA_STEMS = (
    "resistance_hindcast",
    "calibration_diagnostics",
    "age_pattern_sensitivity",
    "resistance_mechanism_decomposition",
    "program_portfolio_factorial",
    "infant_contact_sensitivity",
    "maternal_duration_sensitivity",
    "shock_recovery_sensitivity",
    "temporal_assumption_sensitivity",
    "treatment_implementation_sensitivity",
    "individual_stochastic_toy",
    "joint_psa_rank_acceptability",
    "fitness_resistance_grid_posterior_sample_diagnostics",
    "fitness_resistance_grid_posterior_benefit",
    "fitness_resistance_grid_psa_benefit",
    "health_utility_analysis",
    "vaccine_pipeline_mapping",
    "high_risk_review_tables",
    "lancet_child_adolescent_tables",
)


def test_core_validation_stems_match_makefile_simulate_outputs() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert validation.CORE_OUTPUT_STEMS == EXPECTED_SIMULATE_SUMMARY_STEMS
    assert "bayesian_uncertainty" not in validation.CORE_OUTPUT_STEMS
    assert validation.OPTIONAL_EXISTING_OUTPUT_STEMS == ("bayesian_uncertainty",)
    assert validation.PUBLICATION_METADATA_STEMS == EXPECTED_PUBLICATION_METADATA_STEMS
    for stem in EXPECTED_SIMULATE_SUMMARY_STEMS:
        assert EXPECTED_SIMULATE_RUNNER_FRAGMENTS[stem] in makefile
    assert "data/raw/covid_npi_contact_reduction_timeline.csv" in makefile


def test_publication_validation_stems_have_required_table_entries() -> None:
    assert set(validation.PUBLICATION_REQUIRED_TABLES) == set(validation.PUBLICATION_METADATA_STEMS)
    for stem, tables in validation.PUBLICATION_REQUIRED_TABLES.items():
        assert tables, stem
        assert all(path.startswith("outputs/") for path in tables)


def test_validate_main_output_windows_requires_core_summaries(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(validation, "CORE_OUTPUT_STEMS", ("missing_core",))
    monkeypatch.setattr(validation, "OPTIONAL_EXISTING_OUTPUT_STEMS", ())
    monkeypatch.setattr(validation, "project_path", lambda *parts: tmp_path.joinpath(*parts))

    with pytest.raises(AssertionError, match="Missing required core summary output"):
        validation.validate_main_output_windows()


def test_validate_main_output_windows_skips_missing_optional_summaries(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(validation, "CORE_OUTPUT_STEMS", ())
    monkeypatch.setattr(validation, "OPTIONAL_EXISTING_OUTPUT_STEMS", ("bayesian_uncertainty",))
    monkeypatch.setattr(validation, "project_path", lambda *parts: tmp_path.joinpath(*parts))

    validation.validate_main_output_windows()


def test_validate_publication_outputs_rejects_missing_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(validation, "PUBLICATION_METADATA_STEMS", ("missing_publication",))
    monkeypatch.setattr(validation, "PUBLICATION_REQUIRED_TABLES", {"missing_publication": ()})
    monkeypatch.setattr(validation, "project_path", lambda *parts: tmp_path.joinpath(*parts))

    with pytest.raises(FileNotFoundError):
        validation.validate_publication_outputs()
