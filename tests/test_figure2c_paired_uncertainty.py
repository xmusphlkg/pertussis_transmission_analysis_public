from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src_python.simulation.audit_figure2c_full_cri as audit_module
import src_python.simulation.run_figure2c_paired_uncertainty as figure2c_module
from src_python.simulation.audit_figure2c_full_cri import (
    REQUIRED_INTERVENTION_UNCERTAINTY_COLUMNS,
    _convergence_checks,
    _draw_checks,
    _figure_metadata_checks,
    _interval_checks,
    _interval_width_audit,
    _interval_matches_draw_quantiles,
    _posterior_parameter_width_checks,
    _posterior_runtime_checks,
)
from src_python.simulation.run_figure2c_paired_uncertainty import (
    FIGURE2C_STRATEGIES,
    INTERVENTION_UNCERTAINTY_PREFIX,
    JOINT_POSTERIOR_PARAMETERS,
    PRIMARY_REDUCTION,
    _apply_intervention_uncertainty,
    _build_scenarios,
    _interval_summary,
    _paired_draws,
    _select_posterior_samples,
    _validate_joint_posterior_samples,
)


def test_paired_draws_compute_reduction_with_matched_current_draws() -> None:
    summary = pd.DataFrame(
        [
            {
                "country": "A",
                "posterior_draw": 1,
                "structural_draw_id": 101,
                "strategy": "current",
                "strategy_label": "Current",
                "posterior_chain": 0,
                "posterior_source_draw": 10,
                "posterior_log_prob": -1.0,
                "annualized_child_adolescent_cases_per_100k": 100.0,
                "total_child_adolescent_cases": 1000.0,
            },
            {
                "country": "A",
                "posterior_draw": 1,
                "structural_draw_id": 101,
                "strategy": "timeliness_only",
                "strategy_label": "Routine timeliness",
                "posterior_chain": 0,
                "posterior_source_draw": 10,
                "posterior_log_prob": -1.0,
                "annualized_child_adolescent_cases_per_100k": 80.0,
                "total_child_adolescent_cases": 800.0,
            },
            {
                "country": "A",
                "posterior_draw": 2,
                "structural_draw_id": 202,
                "strategy": "current",
                "strategy_label": "Current",
                "posterior_chain": 0,
                "posterior_source_draw": 11,
                "posterior_log_prob": -2.0,
                "annualized_child_adolescent_cases_per_100k": 200.0,
                "total_child_adolescent_cases": 2000.0,
            },
            {
                "country": "A",
                "posterior_draw": 2,
                "structural_draw_id": 202,
                "strategy": "timeliness_only",
                "strategy_label": "Routine timeliness",
                "posterior_chain": 0,
                "posterior_source_draw": 11,
                "posterior_log_prob": -2.0,
                "annualized_child_adolescent_cases_per_100k": 150.0,
                "total_child_adolescent_cases": 1500.0,
            },
        ]
    )

    draws = _paired_draws(summary)

    assert np.allclose(draws[PRIMARY_REDUCTION], [0.2, 0.25])
    assert list(draws["current_rate"]) == [100.0, 200.0]
    assert list(draws["intervention_rate"]) == [80.0, 150.0]
    assert list(draws["structural_draw_id"]) == [101, 202]


def test_retired_full_joint_stem_is_rejected_before_reading_outputs() -> None:
    retired_path = (
        "outputs/simulations/"
        "bayesian_uncertainty_full_joint_posterior_samples.parquet"
    )
    with pytest.raises(ValueError, match="incorrectly labels conditional uncertainty"):
        figure2c_module._posterior_metadata(retired_path)
    with pytest.raises(ValueError, match="mislabeled conditional uncertainty"):
        audit_module.main(posterior_stem="bayesian_uncertainty_full_joint")


def test_figure2c_rejects_exploratory_country_before_reading_samples() -> None:
    with pytest.raises(ValueError, match="outside the prespecified publication set"):
        figure2c_module.main(
            countries_filter=["Australia", "South_Africa"],
            posterior_samples_path="does-not-exist.parquet",
        )


def test_figure2c_quality_bypass_cannot_write_canonical_outputs() -> None:
    with pytest.raises(ValueError, match="retired for the canonical Figure 2c"):
        figure2c_module.main(allow_conditional_posterior=True)


def test_interval_summary_uses_draw_level_quantiles() -> None:
    draws = pd.DataFrame(
        {
            "country": ["A"] * 5,
            "posterior_draw": [1, 2, 3, 4, 5],
            "strategy": ["timeliness_only"] * 5,
            "strategy_label": ["Routine timeliness"] * 5,
            "current_rate": [100, 100, 100, 100, 100],
            "intervention_rate": [90, 80, 70, 60, 50],
            PRIMARY_REDUCTION: [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    interval = _interval_summary(draws).iloc[0]

    assert interval["reduction_median"] == 0.3
    assert np.isclose(interval["reduction_q025"], np.quantile(draws[PRIMARY_REDUCTION], 0.025))
    assert np.isclose(interval["reduction_q975"], np.quantile(draws[PRIMARY_REDUCTION], 0.975))
    assert interval["posterior_draws"] == 5
    assert interval["uncertainty_draws"] == 5
    assert interval["interval_type"] == (
        "95% alternative-target posterior interval (diagnostic)"
    )


def test_interval_summary_labels_exact_state_route_as_conditional_uncertainty() -> None:
    draws = pd.DataFrame(
        {
            "country": ["A"] * 4,
            "posterior_draw": [1, 2, 3, 4],
            "strategy": ["timeliness_only"] * 4,
            "strategy_label": ["Routine timeliness"] * 4,
            "current_rate": [100.0] * 4,
            "intervention_rate": [90.0, 85.0, 80.0, 75.0],
            PRIMARY_REDUCTION: [0.10, 0.15, 0.20, 0.25],
            "inference_structure": [
                "reference_structure_state_space_exact_importance_cut"
            ]
            * 4,
        }
    )

    interval = _interval_summary(draws).iloc[0]

    assert interval["interval_type"] == "95% conditional uncertainty interval"
    assert "exact-target importance-corrected" in interval["interval_basis"]
    assert interval["uncertainty_draws"] == 4


def test_build_scenarios_retains_structural_draw_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        figure2c_module,
        "make_intervention_config",
        lambda *args, **kwargs: ({"simulation": {}}, "current_vaccine"),
    )
    monkeypatch.setattr(
        figure2c_module,
        "_apply_sample",
        lambda config, sample: config,
    )
    sample = {
        "country": "A",
        "posterior_draw": 1,
        "chain": 2,
        "draw": 17,
        "structural_draw_id": 7,
        "inference_structure": "modular_hierarchical_cut",
        "beta_S": 0.02,
        "reporting_multiplier": 1.0,
        "VE_sus": 0.5,
        "VE_inf": 0.2,
        "VE_dur": 0.3,
        "relative_infectiousness_asymptomatic": 0.5,
        "infectious_duration_symptomatic": 24.0,
        "infectious_duration_asymptomatic": 17.3,
        "fitness_R": 1.0,
        "resistance_prevalence": 0.3,
        "reporting_trend_end_multiplier": 1.0,
    }

    scenarios = _build_scenarios(
        {"baseline": {}, "interventions": {}},
        pd.DataFrame([sample]),
        strategies=("current",),
    )

    assert len(scenarios) == 1
    assert scenarios[0]["metadata"]["structural_draw_id"] == 7
    assert scenarios[0]["metadata"]["inference_structure"] == "modular_hierarchical_cut"


def test_full_cri_rejects_beta_grid_posterior_samples() -> None:
    rows = []
    for draw in range(1, 4):
        row = {"country": "A", "sampling_method": "beta_grid"}
        for idx, parameter in enumerate(JOINT_POSTERIOR_PARAMETERS, start=1):
            row[parameter] = float(idx + draw / 100.0)
        rows.append(row)
    samples = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="conditional beta-grid"):
        _validate_joint_posterior_samples(
            samples,
            countries=["A"],
            posterior_metadata={
                "sampler": "beta_grid",
                "uncertainty_scope": "conditional_beta_grid",
                "fixed_parameters": ["reporting_multiplier"],
                "fix_durations": True,
            },
            posterior_stem="bayesian_uncertainty",
            allow_conditional_posterior=False,
            min_variable_parameters=len(JOINT_POSTERIOR_PARAMETERS),
        )


def test_full_cri_rejects_nonconverged_joint_posterior() -> None:
    rows = []
    for draw in range(1, 4):
        row = {"country": "A", "sampling_method": "smc"}
        for idx, parameter in enumerate(JOINT_POSTERIOR_PARAMETERS, start=1):
            row[parameter] = float(idx + draw / 100.0)
        rows.append(row)

    with pytest.raises(ValueError, match="recommended quality gate"):
        _validate_joint_posterior_samples(
            pd.DataFrame(rows),
            countries=["A"],
            posterior_metadata={
                "sampler": "smc",
                "uncertainty_scope": "multi_parameter_tempered_smc_posterior",
                "fixed_parameters": [],
                "fix_durations": False,
                "convergence_summary": {
                    "n_parameters_total": len(JOINT_POSTERIOR_PARAMETERS),
                    "n_parameters_converged": 0,
                    "n_parameters_recommended_converged": 0,
                    "all_converged": False,
                    "all_recommended_converged": False,
                },
            },
            posterior_stem="pilot",
            allow_conditional_posterior=False,
            min_variable_parameters=len(JOINT_POSTERIOR_PARAMETERS),
        )


def test_intervention_uncertainty_updates_strategy_definition() -> None:
    intervention = {
        "coverage_min_updates": {
            "infant_0_2m": 0.75,
            "adolescent_10_17y": 0.90,
            "young_adult_18_39y": 0.55,
        },
        "vaccine_overrides": {"maternal_VE_sus": 0.55, "maternal_VE_sym": 0.92},
        "natural_history_overrides": {"maternal_protection_duration": 180.0},
        "pep_updates": {"coverage_household_contacts": 0.45},
        "contact_matrix_reduction": {"reduction_fraction": 0.15},
    }
    row = {
        f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_coverage_floor": 0.61,
        f"{INTERVENTION_UNCERTAINTY_PREFIX}adolescent_coverage_floor": 0.87,
        f"{INTERVENTION_UNCERTAINTY_PREFIX}young_adult_coverage_floor": 0.48,
        f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sus": 0.43,
        f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sym": 0.89,
        f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_protection_duration_days": 150.0,
        f"{INTERVENTION_UNCERTAINTY_PREFIX}targeted_pep_coverage": 0.33,
        f"{INTERVENTION_UNCERTAINTY_PREFIX}contact_reduction_fraction": 0.11,
    }

    updated = _apply_intervention_uncertainty(intervention, row)

    assert updated["coverage_min_updates"]["infant_0_2m"] == 0.61
    assert updated["coverage_min_updates"]["adolescent_10_17y"] == 0.87
    assert updated["coverage_min_updates"]["young_adult_18_39y"] == 0.48
    assert updated["vaccine_overrides"]["maternal_VE_sus"] == 0.43
    assert updated["vaccine_overrides"]["maternal_VE_sym"] == 0.89
    assert updated["natural_history_overrides"]["maternal_protection_duration"] == 150.0
    assert updated["pep_updates"]["coverage_household_contacts"] == 0.33
    assert updated["contact_matrix_reduction"]["reduction_fraction"] == 0.11


def test_full_cri_audit_recomputes_interval_quantiles_from_draws() -> None:
    draws = pd.DataFrame(
        {
            "country": ["A"] * 4,
            "strategy": ["timeliness_only"] * 4,
            "posterior_draw": [1, 2, 3, 4],
            PRIMARY_REDUCTION: [0.10, 0.20, 0.30, 0.40],
        }
    )
    good_interval = pd.DataFrame(
        {
            "country": ["A"],
            "strategy": ["timeliness_only"],
            "reduction_q025": [draws[PRIMARY_REDUCTION].quantile(0.025)],
            "reduction_median": [draws[PRIMARY_REDUCTION].quantile(0.500)],
            "reduction_q975": [draws[PRIMARY_REDUCTION].quantile(0.975)],
            "uncertainty_draws": [4],
        }
    )
    bad_interval = good_interval.copy()
    bad_interval["reduction_q975"] = 0.99

    rows: list[dict[str, object]] = []
    _interval_matches_draw_quantiles(rows, good_interval, draws)
    assert rows[-1]["check"] == "interval_quantiles_recomputed_from_draws"
    assert rows[-1]["status"] == "pass"

    rows = []
    _interval_matches_draw_quantiles(rows, bad_interval, draws)
    assert rows[-1]["check"] == "interval_quantiles_recomputed_from_draws"
    assert rows[-1]["status"] == "fail"


def test_full_cri_interval_width_audit_reports_percentage_points() -> None:
    intervals = pd.DataFrame(
        {
            "country": ["A"],
            "strategy": ["timeliness_only"],
            "reduction_q025": [0.10],
            "reduction_median": [0.20],
            "reduction_q975": [0.35],
            "uncertainty_draws": [200],
        }
    )

    width_audit = _interval_width_audit(intervals).iloc[0]

    assert np.isclose(width_audit["reduction_q95_width"], 0.25)
    assert np.isclose(width_audit["reduction_q95_width_percentage_points"], 25.0)
    assert np.isclose(width_audit["relative_width_vs_abs_median"], 1.25)


def test_full_cri_audit_flags_degenerate_posterior_parameter_widths() -> None:
    parameter_audit = pd.DataFrame(
        {
            "country": ["A", "A"],
            "parameter": ["VE_dur", "beta_S"],
            "q95_width": [1e-8, 1e-8],
            "relative_q95_width": [1e-7, 1e-4],
        }
    )

    rows: list[dict[str, object]] = []
    _posterior_parameter_width_checks(rows, parameter_audit)

    width_rows = [
        row for row in rows
        if row["check"] == "uncertainty_parameter_widths_not_degenerate"
    ]
    assert len(width_rows) == 1
    assert width_rows[0]["status"] == "fail"
    assert width_rows[0]["severity"] == "warning"
    assert "VE_dur" in width_rows[0]["details"]
    assert "beta_S" in width_rows[0]["details"]


def test_full_cri_audit_accepts_non_degenerate_posterior_parameter_widths() -> None:
    parameter_audit = pd.DataFrame(
        {
            "country": ["A", "A"],
            "parameter": ["VE_dur", "beta_S"],
            "q95_width": [0.08, 0.0002],
            "relative_q95_width": [0.8, 0.01],
        }
    )

    rows: list[dict[str, object]] = []
    _posterior_parameter_width_checks(rows, parameter_audit)

    failures = [row for row in rows if row["status"] == "fail"]
    assert failures == []


def test_full_cri_interval_audit_records_quantile_resolution() -> None:
    intervals = pd.DataFrame(
        {
            "country": ["A"],
            "strategy": ["timeliness_only"],
            "reduction_q025": [0.10],
            "reduction_median": [0.20],
            "reduction_q975": [0.35],
            "uncertainty_draws": [200],
            "interval_type": ["95% conditional uncertainty interval"],
            "interval_basis": [
                "Paired exact-target importance-corrected state draws plus an "
                "external structural-prior design; fixed hyperparameters."
            ],
        }
    )

    rows: list[dict[str, object]] = []
    _interval_checks(
        rows,
        intervals,
        expected_countries={"A"},
        expected_draws=200,
        expected_inference_structure=(
            "reference_structure_state_space_exact_importance_cut"
        ),
    )

    resolution_rows = [
        row for row in rows
        if row["check"] == "uncertainty_interval_quantile_monte_carlo_resolution"
    ]
    assert len(resolution_rows) == 1
    assert resolution_rows[0]["severity"] == "warning"
    assert "expected_draws_in_each_2.5pct_tail=5.0" in resolution_rows[0]["details"]


def test_full_cri_audit_accepts_complete_paired_draw_grid() -> None:
    intervention_strategies = [strategy for strategy in FIGURE2C_STRATEGIES if strategy != "current"]
    records = []
    for strategy in intervention_strategies:
        for posterior_draw in (1, 2):
            row = {
                "country": "A",
                "strategy": strategy,
                "posterior_draw": posterior_draw,
                "posterior_chain": posterior_draw,
                "current_rate": 100.0 + posterior_draw,
                "intervention_rate": 80.0 + posterior_draw,
                PRIMARY_REDUCTION: 1.0 - (80.0 + posterior_draw) / (100.0 + posterior_draw),
            }
            for idx, parameter in enumerate(JOINT_POSTERIOR_PARAMETERS, start=1):
                row[f"posterior_{parameter}"] = float(idx + posterior_draw / 10.0)
            for idx, parameter in enumerate(REQUIRED_INTERVENTION_UNCERTAINTY_COLUMNS, start=1):
                row[f"{INTERVENTION_UNCERTAINTY_PREFIX}{parameter}"] = 0.10 + idx / 100.0 + posterior_draw / 1000.0
            records.append(row)
    draws = pd.DataFrame(records)
    bounds = {
        f"{INTERVENTION_UNCERTAINTY_PREFIX}{parameter}": (0.0, 1.0)
        for parameter in REQUIRED_INTERVENTION_UNCERTAINTY_COLUMNS
    }

    rows: list[dict[str, object]] = []
    _draw_checks(
        rows,
        draws,
        expected_countries={"A"},
        expected_draws=2,
        expected_chains=2,
        intervention_bounds=bounds,
    )

    fatal_failures = [
        row for row in rows
        if row["severity"] == "fatal" and row["status"] == "fail"
    ]
    assert fatal_failures == []


def test_full_cri_audit_records_missing_required_outputs(tmp_path, monkeypatch) -> None:
    original_project_path = audit_module.project_path

    def _isolated_project_path(*parts):
        name = str(parts[-1]) if parts else ""
        if name.endswith("_posterior_samples.parquet"):
            return tmp_path / "missing_posterior.parquet"
        if name.endswith("_convergence_diagnostics.csv"):
            return tmp_path / "missing_convergence.csv"
        return original_project_path(*parts)

    monkeypatch.setattr(audit_module, "project_path", _isolated_project_path)
    monkeypatch.setattr(audit_module, "INTERVAL_PATH", tmp_path / "missing_intervals.csv")
    monkeypatch.setattr(audit_module, "DRAW_PATH", tmp_path / "missing_draws.csv")
    monkeypatch.setattr(audit_module, "OUTPUT_PATH", tmp_path / "audit.csv")

    def _missing_metadata(stem: str) -> dict[str, object]:
        raise FileNotFoundError(f"missing metadata for {stem}")

    monkeypatch.setattr(audit_module, "validate_run_metadata", _missing_metadata)
    monkeypatch.setattr(audit_module, "write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        audit_module,
        "require_predictive_publication_gate",
        lambda **_kwargs: {},
    )

    with pytest.raises(SystemExit) as exc:
        audit_module.main(expected_draws=200, expected_chains=10, fail_on_warnings=True)

    assert exc.value.code == 1
    audit = pd.read_csv(tmp_path / "audit.csv")
    missing_checks = audit.loc[audit["status"].eq("fail"), "check"].tolist()
    assert "run_metadata_file_exists" in missing_checks
    assert "posterior_sample_file_exists" in missing_checks
    assert "diagnostics_file_exists" in missing_checks
    assert "interval_file_exists" in missing_checks
    assert "draw_file_exists" in missing_checks


def test_full_cri_audit_rejects_stale_figure2c_source_metadata() -> None:
    rows: list[dict[str, object]] = []
    _figure_metadata_checks(
        rows,
        {
            "posterior_sample_stem": "figure2c_full_cri_smoke",
            "posterior_samples_path": "outputs/simulations/figure2c_full_cri_smoke_posterior_samples.parquet",
            "countries": ["Australia"],
            "strategies": list(FIGURE2C_STRATEGIES),
            "uncertainty_draws_per_country": 2,
        },
        configured_countries={"Australia", "Brazil"},
        expected_draws=200,
    )

    failures = {
        row["check"]
        for row in rows
        if row["severity"] == "fatal" and row["status"] == "fail"
    }
    assert "posterior_source_stem_matches_conditional_release" in failures
    assert "posterior_source_path_matches_conditional_release" in failures
    assert "all_configured_countries_included" in failures
    assert "uncertainty_draws_per_country_matches_expected" in failures


def test_full_cri_audit_checks_posterior_runtime_settings(monkeypatch) -> None:
    monkeypatch.setattr(audit_module, "config_fingerprint", lambda: "current-hash")
    monkeypatch.setattr(audit_module, "source_code_fingerprint", lambda: "current-source")
    rows: list[dict[str, object]] = []
    _posterior_runtime_checks(
        rows,
        {
            "sampler": "adaptive_mh",
            "parameterization": "beta_reporting_product",
            "n_chains": 10,
            "draws_per_chain": 1000,
            "warmup": 2000,
            "thin": 2,
            "config_hash": "current-hash",
            "source_code_hash": "current-source",
        },
        expected_chains=10,
    )

    failures = [row for row in rows if row["severity"] == "fatal" and row["status"] == "fail"]
    assert failures == []

    rows = []
    _posterior_runtime_checks(
        rows,
        {
            "sampler": "beta_grid",
            "parameterization": "standard",
            "n_chains": 4,
            "draws_per_chain": 1000,
            "warmup": 2000,
            "thin": 2,
            "config_hash": "posterior-hash",
        },
        expected_chains=10,
    )

    failed_checks = {
        row["check"]
        for row in rows
        if row["severity"] == "fatal" and row["status"] == "fail"
    }
    assert "sampler_matches_recognized_uncertainty_target" in failed_checks
    assert "parameterization_beta_reporting_product" in failed_checks
    assert "expected_chain_count_in_metadata" in failed_checks


def test_full_cri_audit_flags_marginal_convergence_as_warning() -> None:
    diagnostics = pd.DataFrame(
        {
            "country": ["A", "A", "B"],
            "parameter": ["beta_S", "VE_sus", "fitness_R"],
            "rhat_rank": [1.008, 1.020, 1.006],
            "bulk_ess": [500.0, 250.0, 430.0],
            "tail_ess": [460.0, 300.0, 420.0],
            "converged": [True, True, True],
        }
    )

    rows: list[dict[str, object]] = []
    _convergence_checks(rows, diagnostics)

    fatal_failures = [
        row for row in rows
        if row["severity"] == "fatal" and row["status"] == "fail"
    ]
    warning_failures = {
        row["check"]
        for row in rows
        if row["severity"] == "warning" and row["status"] == "fail"
    }
    assert fatal_failures == []
    assert "recommended_rhat_under_1_01" in warning_failures
    assert "recommended_bulk_ess_over_400" in warning_failures
    assert "recommended_tail_ess_over_400" in warning_failures


def test_full_cri_audit_accepts_recommended_convergence() -> None:
    diagnostics = pd.DataFrame(
        {
            "country": ["A", "A", "B"],
            "parameter": ["beta_S", "VE_sus", "fitness_R"],
            "rhat_rank": [1.004, 1.009, 1.006],
            "bulk_ess": [500.0, 420.0, 430.0],
            "tail_ess": [460.0, 410.0, 420.0],
            "converged": [True, True, True],
        }
    )

    rows: list[dict[str, object]] = []
    _convergence_checks(rows, diagnostics)

    failures = [row for row in rows if row["status"] == "fail"]
    assert failures == []


def test_full_cri_audit_uses_importance_weight_quality_for_joint_importance() -> None:
    diagnostics = pd.DataFrame(
        {
            "country": ["A", "A"],
            "parameter": ["beta_S", "VE_sus"],
            "diagnostic_method": ["joint_importance_weight_quality"] * 2,
            "rhat_rank": [1.0, 1.0],
            "bulk_ess": [800.0, 800.0],
            "tail_ess": [800.0, 800.0],
            "importance_effective_sample_size": [800.0, 800.0],
            "importance_nuisance_effective_sample_size": [140.0, 140.0],
            "importance_max_weight": [0.01, 0.01],
            "importance_edge_weight": [0.003, 0.003],
            "converged": [True, True],
            "recommended_converged": [True, True],
        }
    )

    rows: list[dict[str, object]] = []
    _convergence_checks(rows, diagnostics)

    failures = [row for row in rows if row["status"] == "fail"]
    assert failures == []

    bad = diagnostics.copy()
    bad["importance_max_weight"] = 0.08
    bad["recommended_converged"] = False
    rows = []
    _convergence_checks(rows, bad)
    warning_failures = {
        row["check"]
        for row in rows
        if row["severity"] == "warning" and row["status"] == "fail"
    }
    assert "importance_max_weight_under_0_02" in warning_failures
    assert "all_country_parameters_meet_recommended_importance_quality" in warning_failures


def test_full_cri_audit_uses_worst_per_nuisance_grid_quality_for_modular_cut() -> None:
    diagnostics = pd.DataFrame(
        {
            "country": ["A", "A"],
            "parameter": ["beta_S", "VE_sus"],
            "diagnostic_method": ["modular_cut_conditional_grid_quality"] * 2,
            "rhat_rank": [1.0, 1.0],
            "bulk_ess": [30.0, 30.0],
            "tail_ess": [30.0, 30.0],
            "converged": [True, True],
            "recommended_converged": [True, True],
            "modular_cut_structural_draw_count": [128.0, 128.0],
            "modular_cut_conditional_incomplete_grid_count": [0.0, 0.0],
            "modular_cut_conditional_recommended_failure_count": [0.0, 0.0],
            "modular_cut_conditional_min_effective_grid_points": [30.0, 30.0],
            "modular_cut_conditional_min_beta_effective_grid_points": [6.0, 6.0],
            "modular_cut_conditional_min_reporting_effective_grid_points": [8.0, 8.0],
            "modular_cut_conditional_max_single_weight": [0.08, 0.08],
            "modular_cut_conditional_max_edge_weight": [0.005, 0.005],
        }
    )

    rows: list[dict[str, object]] = []
    _convergence_checks(rows, diagnostics)

    assert [row for row in rows if row["status"] == "fail"] == []
    assert not any(
        row["check"] == "importance_nuisance_ess_over_100" for row in rows
    )

    bad = diagnostics.copy()
    bad["recommended_converged"] = False
    bad["modular_cut_conditional_recommended_failure_count"] = 1.0
    bad["modular_cut_conditional_max_edge_weight"] = 0.08
    rows = []
    _convergence_checks(rows, bad)
    warning_failures = {
        row["check"]
        for row in rows
        if row["severity"] == "warning" and row["status"] == "fail"
    }

    assert "modular_cut_all_conditional_grids_meet_recommended_quality" in warning_failures
    assert "modular_cut_conditional_edge_weight_under_0_01" in warning_failures


def test_full_cri_audit_uses_particle_quality_for_smc() -> None:
    diagnostics = pd.DataFrame(
        {
            "country": ["A", "A"],
            "parameter": ["beta_S", "VE_sus"],
            "diagnostic_method": ["tempered_smc_rank_and_particle_quality"] * 2,
            "rhat_rank": [1.004, 1.006],
            "bulk_ess": [500.0, 480.0],
            "tail_ess": [460.0, 440.0],
            "smc_reached_final_temperature": [True, True],
            "smc_min_ess_fraction": [0.62, 0.62],
            "smc_final_ess": [520.0, 520.0],
            "smc_combined_particle_ess": [1800.0, 1800.0],
            "smc_max_weight": [0.012, 0.012],
            "smc_unique_particle_fraction": [0.80, 0.80],
            "smc_island_count": [6, 6],
            "smc_island_ess": [4.2, 4.2],
            "smc_max_island_weight": [0.32, 0.32],
            "converged": [True, True],
            "recommended_converged": [True, True],
        }
    )

    rows: list[dict[str, object]] = []
    _convergence_checks(rows, diagnostics)

    failures = [row for row in rows if row["status"] == "fail"]
    assert failures == []

    bad = diagnostics.copy()
    bad["smc_max_weight"] = 0.08
    bad["recommended_converged"] = False
    rows = []
    _convergence_checks(rows, bad)
    warning_failures = {
        row["check"]
        for row in rows
        if row["severity"] == "warning" and row["status"] == "fail"
    }
    assert "smc_max_weight_under_0_02" in warning_failures
    assert "all_country_parameters_meet_recommended_smc_quality" in warning_failures

    bad = diagnostics.copy()
    bad["smc_island_ess"] = 1.5
    bad["smc_max_island_weight"] = 0.85
    bad["recommended_converged"] = False
    rows = []
    _convergence_checks(rows, bad)
    warning_failures = {
        row["check"]
        for row in rows
        if row["severity"] == "warning" and row["status"] == "fail"
    }
    assert "smc_island_ess_over_0_35_islands" in warning_failures
    assert "smc_max_island_weight_under_0_50" in warning_failures


def test_select_posterior_samples_balances_draws_across_chains() -> None:
    samples = pd.DataFrame(
        {
            "country": ["A"] * 30,
            "chain": np.repeat([1, 2, 3], 10),
            "draw": list(range(1, 11)) * 3,
            "beta_S": np.linspace(0.01, 0.03, 30),
        }
    )

    selected = _select_posterior_samples(
        samples,
        countries=["A"],
        draws_per_country=6,
        seed=123,
    )

    assert len(selected) == 6
    assert selected.groupby("chain").size().to_dict() == {1: 2, 2: 2, 3: 2}
    assert list(selected["posterior_draw"]) == [1, 2, 3, 4, 5, 6]


def test_select_posterior_samples_preserves_cross_country_structural_pairing() -> None:
    records: list[dict[str, object]] = []
    for country_index, country in enumerate(("A", "B")):
        for chain in (1, 2):
            for draw in range(1, 7):
                records.append(
                    {
                        "country": country,
                        "chain": chain,
                        "draw": draw,
                        "structural_draw_id": ((chain - 1) * 6 + draw - 1) % 4,
                        "inference_structure": "modular_hierarchical_cut",
                        "beta_S": 0.01 + 0.001 * country_index + 0.0001 * draw,
                    }
                )
    samples = pd.DataFrame(records)
    # Row order is deliberately unrelated across countries; source keys, not
    # input order, define the shared modular draw.
    samples = pd.concat(
        [
            samples.loc[samples["country"].eq("B")].sample(frac=1.0, random_state=8),
            samples.loc[samples["country"].eq("A")].sample(frac=1.0, random_state=3),
        ],
        ignore_index=True,
    )

    selected = _select_posterior_samples(
        samples,
        countries=["A", "B"],
        draws_per_country=6,
        seed=123,
    )

    assert len(selected) == 12
    assert selected.groupby("posterior_draw")["country"].nunique().eq(2).all()
    assert selected.groupby("posterior_draw")["structural_draw_id"].nunique().eq(1).all()
    assert selected.groupby("posterior_draw")["chain"].nunique().eq(1).all()
    assert selected.groupby("posterior_draw")["draw"].nunique().eq(1).all()
    structural_counts = (
        selected.loc[selected["country"].eq("A"), "structural_draw_id"].value_counts()
    )
    assert set(structural_counts.index) == {0, 1, 2, 3}
    assert int(structural_counts.max() - structural_counts.min()) <= 1
    assert selected.groupby(["country", "chain"]).size().to_dict() == {
        ("A", 1): 3,
        ("A", 2): 3,
        ("B", 1): 3,
        ("B", 2): 3,
    }


def test_select_posterior_samples_rejects_misaligned_structural_source_draw() -> None:
    samples = pd.DataFrame(
        [
            {
                "country": country,
                "chain": 1,
                "draw": draw,
                "structural_draw_id": draw - 1,
                "inference_structure": "modular_hierarchical_cut",
            }
            for country in ("A", "B")
            for draw in (1, 2, 3)
        ]
    )
    samples.loc[samples["country"].eq("B") & samples["draw"].eq(2), "structural_draw_id"] = 99

    with pytest.raises(ValueError, match="not aligned across countries"):
        _select_posterior_samples(
            samples,
            countries=["A", "B"],
            draws_per_country=2,
            seed=123,
        )


def test_full_cri_audit_rejects_cross_country_structural_unpairing() -> None:
    draws = pd.DataFrame(
        {
            "country": ["A", "B"],
            "strategy": ["timeliness_only", "timeliness_only"],
            "posterior_draw": [1, 1],
            "posterior_chain": [1, 1],
            "structural_draw_id": [7, 8],
        }
    )
    rows: list[dict[str, object]] = []
    _draw_checks(
        rows,
        draws,
        expected_countries={"A", "B"},
        expected_draws=1,
        expected_chains=1,
        intervention_bounds={},
        require_structural_pairing=True,
    )

    pairing_check = next(
        row for row in rows if row["check"] == "structural_draw_id_paired_across_countries"
    )
    assert pairing_check["status"] == "fail"
    assert pairing_check["severity"] == "fatal"
