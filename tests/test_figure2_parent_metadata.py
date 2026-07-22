from __future__ import annotations

from copy import deepcopy

import pytest

from manuscript_notes import validate_figure2_parent_metadata as validator
from src_python.simulation.run_joint_psa_rank_acceptability import (
    FIGURE2B_PARAMETER_NAMES,
)


def _metadata(stem: str) -> dict:
    common = {"stem": stem}
    if stem == "joint_psa_rank_acceptability":
        return {
            **common,
            "figure2b_parameter_names": list(FIGURE2B_PARAMETER_NAMES),
            "excluded_dead_dimensions": ["resistance_management_uptake"],
        }
    if stem == "figure2c_parametric_bootstrap":
        countries = [f"Country_{index}" for index in range(9)]
        strategies = [f"strategy_{index}" for index in range(6)]
        successful = {country: 1024 for country in countries}
        digests = {
            "paired_bootstrap_draws": "draws",
            "confidence_intervals": "intervals",
            "fit_diagnostics": "fits",
            "interval_stability": "stability",
        }
        return {
            **common,
            "statistical_target": "frequentist_confidence_interval",
            "interval_type": validator.EXPECTED_FIGURE2C_INTERVAL_TYPE,
            "interval_basis": validator.EXPECTED_FIGURE2C_INTERVAL_BASIS,
            "confidence_interval_method": "percentile_parametric_bootstrap",
            "bootstrap_data_generation": "marginal_AR1_process_plus_NB2_measurement",
            "bootstrap_refit": "country_state_space_MAP_full_refit_per_replicate",
            "analysis_role": "publication_estimation_confidence_interval",
            "varied_estimation_components": list(
                validator.EXPECTED_FIGURE2C_VARIED_COMPONENTS
            ),
            "fixed_reference_inputs": list(validator.EXPECTED_FIGURE2C_FIXED_INPUTS),
            "paired_scenario_contrast": True,
            "publication_path": True,
            "figure2c_interval_source": True,
            "posterior_credible_interval": False,
            "future_observation_prediction_interval": False,
            "countries": countries,
            "strategies": strategies,
            "replicates_requested_per_country": 1024,
            "successful_replicates_by_country": successful,
            "minimum_successful_replicates": 1000,
            "minimum_success_fraction": 0.95,
            "maximum_tail_probability_mcse_threshold": 0.005,
            "observed_maximum_tail_probability_mcse": 0.004,
            "row_counts": {
                "fit_diagnostics": 9 * 1024,
                "confidence_interval_rows": 9 * 6,
                "interval_stability_rows": 9 * 6,
                "paired_bootstrap_draws": sum(successful.values()) * 6,
            },
            "output_artifact_sha256": digests,
        }
    if stem == "figure2c_parametric_bootstrap_quality_audit":
        return {
            **common,
            "analysis_role": "publication_estimation_confidence_interval_quality_audit",
            "publication_path": True,
            "figure2c_interval_source": False,
            "audits_figure2c_interval_source": True,
            "passed": True,
            "warnings_are_fatal": True,
            "figure2c_source_stem": "figure2c_parametric_bootstrap",
            "audited_artifact_sha256": {
                "paired_bootstrap_draws_sha256": "draws",
                "confidence_intervals_sha256": "intervals",
                "fit_diagnostics_sha256": "fits",
                "interval_stability_sha256": "stability",
            },
        }
    return common


def test_figure2_parent_validator_checks_every_current_stem(monkeypatch) -> None:
    calls: list[str] = []

    def fake_validate(stem: str) -> dict:
        calls.append(stem)
        return _metadata(stem)

    monkeypatch.setattr(validator, "validate_run_metadata", fake_validate)
    observed = validator.validate_figure2_parent_metadata()

    assert tuple(calls) == validator.FIGURE2_PARENT_STEMS
    assert set(observed) == set(validator.FIGURE2_PARENT_STEMS)


@pytest.mark.parametrize(
    ("stem", "field", "value", "message"),
    [
        (
            "joint_psa_rank_acceptability",
            "figure2b_parameter_names",
            [*FIGURE2B_PARAMETER_NAMES, "resistance_management_uptake"],
            "six-input",
        ),
        (
            "figure2c_parametric_bootstrap",
            "statistical_target",
            "posterior_credible_interval",
            "wrong statistical_target",
        ),
        (
            "figure2c_parametric_bootstrap",
            "confidence_interval_method",
            "posterior_quantiles",
            "wrong confidence_interval_method",
        ),
        (
            "figure2c_parametric_bootstrap",
            "posterior_credible_interval",
            True,
            "posterior credible interval",
        ),
        (
            "figure2c_parametric_bootstrap",
            "future_observation_prediction_interval",
            True,
            "future-observation interval",
        ),
        (
            "figure2c_parametric_bootstrap",
            "publication_path",
            False,
            "publication path",
        ),
        (
            "figure2c_parametric_bootstrap",
            "fixed_reference_inputs",
            ["biological_parameters"],
            "locked reference inputs",
        ),
        (
            "figure2c_parametric_bootstrap_quality_audit",
            "passed",
            False,
            "did not pass",
        ),
    ],
)
def test_figure2_parent_validator_rejects_wrong_semantics(
    monkeypatch,
    stem: str,
    field: str,
    value: object,
    message: str,
) -> None:
    records = {name: _metadata(name) for name in validator.FIGURE2_PARENT_STEMS}
    records[stem] = deepcopy(records[stem])
    records[stem][field] = value
    monkeypatch.setattr(
        validator,
        "validate_run_metadata",
        lambda name: deepcopy(records[name]),
    )

    with pytest.raises(ValueError, match=message):
        validator.validate_figure2_parent_metadata()


def test_figure2_parent_validator_rejects_incomplete_bootstrap(monkeypatch) -> None:
    records = {name: _metadata(name) for name in validator.FIGURE2_PARENT_STEMS}
    records["figure2c_parametric_bootstrap"]["row_counts"]["fit_diagnostics"] -= 1
    monkeypatch.setattr(
        validator, "validate_run_metadata", lambda name: deepcopy(records[name])
    )
    with pytest.raises(ValueError, match="complete bootstrap run"):
        validator.validate_figure2_parent_metadata()


def test_figure2_parent_validator_rejects_mismatched_audit_digests(monkeypatch) -> None:
    records = {name: _metadata(name) for name in validator.FIGURE2_PARENT_STEMS}
    records["figure2c_parametric_bootstrap_quality_audit"][
        "audited_artifact_sha256"
    ]["confidence_intervals_sha256"] = "wrong"
    monkeypatch.setattr(
        validator, "validate_run_metadata", lambda name: deepcopy(records[name])
    )
    with pytest.raises(ValueError, match="does not match"):
        validator.validate_figure2_parent_metadata()
