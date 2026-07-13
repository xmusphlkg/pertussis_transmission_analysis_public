from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from manuscript_notes import generate_high_risk_review_tables as review_tables
from src_python.model.parameters import PreparedParameters
from src_python.simulation import run_sensitivity as sensitivity_runner
from src_python.simulation.common import (
    config_fingerprint,
    load_configs,
    make_config,
    uncertainty_config_fingerprint,
)
from src_python.simulation.parameter_distributions import inverse_cdf, validate_distribution_spec
from src_python.simulation.run_all import BAYESIAN_FIXED_PARAMETERS
from src_python.simulation.run_joint_psa_rank_acceptability import (
    SAMPLE_DESIGN as JOINT_SAMPLE_DESIGN,
    _retain_matching_completed_draws,
    _sample_table as joint_sample_table,
)
from src_python.simulation.run_sensitivity import (
    SAMPLE_DESIGN,
    _add_tornado_metrics,
    _apply_sample,
    _baseline_sensitivity_config,
    _sample_table,
    _settings,
)


def _global_settings():
    configs = load_configs()
    settings, schema_version = _settings(configs)
    return configs, settings, schema_version


def test_literature_distribution_registry_is_loaded_and_valid() -> None:
    configs, settings, schema_version = _global_settings()
    registry = configs["parameter_distributions"]
    source_ids = set(registry["literature_sources"])

    assert schema_version == 1
    assert registry["sampling_contract"]["uncertainty_kind"] == "epistemic"
    assert "waning_rate_natural" not in settings["parameters"]
    assert "natural_immunity_duration" in settings["parameters"]
    assert "vaccine_disease_efficacy" in settings["parameters"]
    for name, spec in settings["parameters"].items():
        normalized = validate_distribution_spec(spec, context=f"parameter {name!r}")
        assert normalized["distribution"] != "uniform"
        assert spec.get("evidence_class")
        assert spec.get("source_ids")
        assert set(spec["source_ids"]).issubset(source_ids)
    for spec in registry["structural_or_conditioned_parameters"].values():
        assert spec.get("treatment")
        assert spec.get("reason")


def test_global_inverse_cdf_lhs_is_reproducible_and_nonuniform() -> None:
    _, settings, schema_version = _global_settings()
    first = _sample_table(settings, sample_size=128, seed=2026, schema_version=schema_version)
    second = _sample_table(settings, sample_size=128, seed=2026, schema_version=schema_version)

    pd.testing.assert_frame_equal(first, second)
    assert first["sample_design"].eq(SAMPLE_DESIGN).all()
    assert first["uncertainty_schema_version"].eq(1).all()
    assert first["sample_id"].tolist() == list(range(1, 129))

    # A beta prior is deliberately concentrated toward high overall disease VE;
    # it must not look like the old uniform range screen.
    efficacy = first["vaccine_disease_efficacy"]
    assert efficacy.mean() > 0.86
    assert efficacy.between(0.70, 0.97).all()


def test_uncertainty_registry_has_a_separate_provenance_fingerprint() -> None:
    configs = load_configs()
    changed = load_configs()
    changed["parameter_distributions"]["global_sensitivity"]["sample_size"] += 1

    assert config_fingerprint(changed) == config_fingerprint(configs)
    assert uncertainty_config_fingerprint(changed) != uncertainty_config_fingerprint(configs)


def test_sensitivity_uses_the_same_china_country_profile_as_the_baseline_runner(
    monkeypatch,
) -> None:
    configs = load_configs()
    calls: dict[str, str] = {}

    def fake_make_config(*, vaccine_scenario, resistance_scenario, country_profile):
        calls.update(
            vaccine_scenario=vaccine_scenario,
            resistance_scenario=resistance_scenario,
            country_profile=country_profile,
        )
        return {
            "country": country_profile,
            "metadata": {
                "calibration_country": country_profile,
                "calibration_loaded": True,
            },
        }

    monkeypatch.setattr(sensitivity_runner, "make_config", fake_make_config)

    country, config = _baseline_sensitivity_config(configs)

    assert country == configs["baseline"].get("baseline_country_profile", "China")
    assert country == "China"
    assert config["country"] == country
    assert config["metadata"]["calibration_country"] == country
    assert config["metadata"]["calibration_loaded"] is True
    assert calls["country_profile"] == country


def test_semantic_parameter_updates_are_coherent_and_active() -> None:
    configs, settings, _ = _global_settings()
    specs = settings["parameters"]
    base = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
        load_calibration=False,
    )
    sample = {name: float(inverse_cdf(0.5, spec, context=name)) for name, spec in specs.items()}
    updated = _apply_sample(base, sample, specs)
    reversed_updated = _apply_sample(base, sample, dict(reversed(list(specs.items()))))

    ve_sus = float(updated["vaccine"]["VE_sus"])
    ve_sym = float(updated["vaccine"]["VE_sym"])
    overall = 1.0 - (1.0 - ve_sus) * (1.0 - ve_sym)
    assert np.isclose(overall, sample["vaccine_disease_efficacy"])
    assert np.isclose(
        updated["natural_history"]["infectious_duration_asymptomatic"],
        sample["infectious_duration_symptomatic"] * sample["asymptomatic_duration_ratio"],
    )
    assert np.isclose(
        reversed_updated["natural_history"]["infectious_duration_asymptomatic"],
        updated["natural_history"]["infectious_duration_asymptomatic"],
    )
    assert np.isclose(
        updated["natural_history"]["R_to_W_duration"],
        sample["natural_immunity_duration"] / 2.0,
    )
    assert np.isclose(
        updated["natural_history"]["W_to_S_duration"],
        sample["natural_immunity_duration"] / 2.0,
    )
    assert np.isclose(
        updated["natural_history"]["recovered_immunity_duration"],
        sample["natural_immunity_duration"],
    )
    assert np.isclose(
        updated["natural_history"]["vaccine_protection_duration"],
        sample["vaccine_protection_duration"] / 2.0,
    )
    assert np.isclose(
        updated["immunity_model"]["waned_vaccine_duration"],
        sample["vaccine_protection_duration"] / 2.0,
    )

    params = PreparedParameters.from_config(updated, analysis="test", scenario="distribution_draw")
    assert np.isclose(params.rates["waning_R_to_W"], 2.0 / sample["natural_immunity_duration"])
    assert np.isclose(params.rates["waning_W_to_S"], 2.0 / sample["natural_immunity_duration"])
    assert np.isclose(
        params.rates["waning_vaccine"],
        2.0 / sample["vaccine_protection_duration"],
    )
    assert np.isclose(
        params.rates["waning_vaccine_waned"],
        2.0 / sample["vaccine_protection_duration"],
    )
    assert params.rates["waning_R_to_W"] != 1.0 / float(base["natural_history"]["R_to_W_duration"])


def test_reporting_factor_uses_reported_case_not_true_case_tornado_endpoint() -> None:
    frame = pd.DataFrame(
        {
            "reporting_multiplier_factor": [0.5, 1.0, 1.5],
            "biological_parameter": [0.1, 0.2, 0.3],
            "total_reported_cases": [10.0, 20.0, 30.0],
            "total_infant_cases": [30.0, 20.0, 10.0],
            "total_child_adolescent_cases": [60.0, 40.0, 20.0],
        }
    )
    specs = {
        "reporting_multiplier_factor": {"outcome": "total_reported_cases"},
        "biological_parameter": {},
    }

    _add_tornado_metrics(frame, specs)

    assert frame["corr_reporting_multiplier_factor_reported_cases"].iloc[0] == 1.0
    assert "corr_reporting_multiplier_factor_infant_cases" not in frame
    assert frame["corr_biological_parameter_child_adolescent_cases"].iloc[0] == -1.0


def test_joint_rank_psa_uses_registry_distributions_and_rejects_stale_resume() -> None:
    configs = load_configs()
    specs = configs["parameter_distributions"]["joint_rank_psa"]["parameters"]
    samples = joint_sample_table(32, 77, specs)
    repeated = joint_sample_table(32, 77, specs)

    pd.testing.assert_frame_equal(samples, repeated)
    assert samples["sample_design"].eq(JOINT_SAMPLE_DESIGN).all()
    assert samples["VE_inf_baseline"].between(0.05, 0.60).all()
    assert samples["VE_inf_baseline"].mean() < 0.35

    existing = pd.DataFrame(
        {
            "psa_sample_id": [1, 1],
            "country": ["A", "A"],
            "strategy": ["current", "other"],
            **{column: [999.0, 999.0] for column in specs},
        }
    )
    completed, retained = _retain_matching_completed_draws({1}, existing, samples)
    assert completed == set()
    assert retained.empty

    expected_row = samples.iloc[0].to_dict()
    matching = pd.DataFrame(
        [
            {**expected_row, "country": "A", "strategy": "current"},
            {**expected_row, "country": "A", "strategy": "other"},
        ]
    )
    completed, retained = _retain_matching_completed_draws({1}, matching, samples)
    assert completed == {1}
    assert len(retained) == 2

    matching["uncertainty_schema_version"] = 999
    completed, retained = _retain_matching_completed_draws({1}, matching, samples)
    assert completed == set()
    assert retained.empty

    with pytest.raises(ValueError, match="semantic contract"):
        joint_sample_table(4, 1, {**specs, "unimplemented_parameter": {"min": 0, "max": 1}})


def test_beta_grid_orchestration_leaves_exactly_one_parameter_active() -> None:
    # Durations are fixed separately; every other Bayesian parameter must be in
    # this tuple or the beta-grid sampler sees more than its single beta target.
    assert "VE_dur" in BAYESIAN_FIXED_PARAMETERS


def test_review_screening_separates_reporting_from_true_disease_endpoint(monkeypatch) -> None:
    configs, settings, schema_version = _global_settings()
    draws = _sample_table(settings, sample_size=64, seed=41, schema_version=schema_version)
    biological = [
        name for name, spec in settings["parameters"].items() if spec.get("outcome") is None
    ]
    draws["annualized_child_adolescent_cases_per_100k"] = sum(
        (idx + 1) * draws[name] / max(float(draws[name].mean()), 1e-12)
        for idx, name in enumerate(biological)
    )
    draws["annualized_reported_cases_per_100k"] = 100.0 * draws[
        "reporting_multiplier_factor"
    ]

    captured: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(review_tables, "_read_csv", lambda _path: draws)
    monkeypatch.setattr(review_tables, "load_configs", lambda: configs)
    monkeypatch.setattr(
        review_tables,
        "_write",
        lambda frame, path: captured.setdefault(path, frame.copy()),
    )

    review_tables.sensitivity_correlations()

    result = captured["outputs/tables/sensitivity_correlation_screening.csv"]
    reporting = result.loc[result["parameter"].eq("reporting_multiplier_factor")].iloc[0]
    assert reporting["outcome"] == "annualized_reported_cases_per_100k"
    assert np.isclose(reporting["partial_rank_correlation"], 1.0)
    assert result.loc[~result["parameter"].eq("reporting_multiplier_factor"), "outcome"].eq(
        "annualized_child_adolescent_cases_per_100k"
    ).all()
    assert result["screening_note"].str.contains("64-sample inverse-CDF").all()


def test_review_screening_recognizes_legacy_output_despite_shared_column_names(monkeypatch) -> None:
    configs, _, _ = _global_settings()
    legacy_specs = configs["sensitivity"]["parameters"]
    rng = np.random.default_rng(3)
    legacy = pd.DataFrame(
        {name: rng.normal(size=64) for name in legacy_specs}
    )
    legacy["annualized_infant_cases_per_100k"] = rng.normal(size=64)
    captured: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(review_tables, "_read_csv", lambda _path: legacy)
    monkeypatch.setattr(review_tables, "load_configs", lambda: configs)
    monkeypatch.setattr(
        review_tables,
        "_write",
        lambda frame, path: captured.setdefault(path, frame.copy()),
    )

    review_tables.sensitivity_correlations()

    result = captured["outputs/tables/sensitivity_correlation_screening.csv"]
    assert set(result["parameter"]) == set(legacy_specs)
    assert result["evidence_class"].eq("legacy_range").all()
    assert result["screening_note"].str.contains("legacy uniform").all()
