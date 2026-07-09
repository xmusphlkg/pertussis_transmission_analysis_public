from __future__ import annotations

from pathlib import Path

import pandas as pd

from publication_inputs.generate_rank_horizon_age_validation_notes import build_outputs


def _write_csv(root: Path, relative_path: str, rows: list[dict[str, object]]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_rank_horizon_age_validation_notes_build_from_existing_tables(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        "outputs/tables/intervention_rank_robustness.csv",
        [
            {
                "scenario": "combined_strategy",
                "median_rank": 1.0,
                "countries_ranked_first": 8,
                "countries_within_10_percent_of_best": 8,
                "median_infant_cases_per_100k": 11.2,
            },
            {
                "scenario": "next_generation_vaccine",
                "median_rank": 2.0,
                "countries_ranked_first": 2,
                "countries_within_10_percent_of_best": 2,
                "median_infant_cases_per_100k": 31.6,
            },
        ],
    )
    _write_csv(
        tmp_path,
        "outputs/tables/intervention_rank_stability_diagnostics.csv",
        [
            {
                "scenario": "combined_strategy",
                "full_horizon_countries_ranked_top_two": 10,
                "analysis_window_cells": 50,
                "analysis_window_cells_ranked_first": 36,
                "analysis_window_cells_ranked_top_two": 47,
                "analysis_window_cells_positive_reduction": 48,
                "infant_age_window_cells": 100,
                "infant_age_window_cells_ranked_first": 73,
                "infant_age_window_cells_ranked_top_two": 94,
                "infant_age_window_cells_positive_reduction": 96,
                "median_infant_age_window_reduction": 0.98,
                "rank_stability_interpretation": "Most stable lowest-burden scenario.",
            },
            {
                "scenario": "next_generation_vaccine",
                "full_horizon_countries_ranked_top_two": 10,
                "analysis_window_cells": 50,
                "analysis_window_cells_ranked_first": 11,
                "analysis_window_cells_ranked_top_two": 48,
                "analysis_window_cells_positive_reduction": 48,
                "infant_age_window_cells": 100,
                "infant_age_window_cells_ranked_first": 21,
                "infant_age_window_cells_ranked_top_two": 89,
                "infant_age_window_cells_positive_reduction": 96,
                "median_infant_age_window_reduction": 0.97,
                "rank_stability_interpretation": "Often near the lowest modeled burden.",
            },
        ],
    )
    _write_csv(
        tmp_path,
        "outputs/summaries/joint_psa_rank_acceptability_summary.csv",
        [
            {
                "country": "All_countries_pooled",
                "strategy": "combined_strategy",
                "probability_rank_1": 0.77,
                "probability_top_2": 1.00,
                "probability_within_10_percent_of_best": 0.86,
                "median_rank": 1.0,
                "median_relative_reduction_vs_current": 0.68,
                "n_psa_samples": 128,
            },
            {
                "country": "All_countries_pooled",
                "strategy": "next_generation_vaccine",
                "probability_rank_1": 0.23,
                "probability_top_2": 0.97,
                "probability_within_10_percent_of_best": 0.35,
                "median_rank": 2.0,
                "median_relative_reduction_vs_current": 0.59,
                "n_psa_samples": 128,
            },
        ],
    )
    _write_csv(
        tmp_path,
        "outputs/tables/lancet_age_pattern_weighted_strategy_summary.csv",
        [
            {
                "ordering_basis": "age_pattern_weighted",
                "strategy": "combined_strategy",
                "strategy_rank_within_basis": 1.0,
                "weighted_median_primary_case_reduction": 0.66,
            },
            {
                "ordering_basis": "age_pattern_weighted",
                "strategy": "next_generation_vaccine",
                "strategy_rank_within_basis": 2.0,
                "weighted_median_primary_case_reduction": 0.52,
            },
        ],
    )
    _write_csv(
        tmp_path,
        "outputs/tables/age_pattern_country_weights.csv",
        [
            {
                "country": "United_States",
                "external_label": "Infants <1 y share",
                "model_age_groups": "infant_0_2m;infant_3_11m",
                "model_window": "2025-2050 scenario horizon",
                "external_value": 0.12,
                "modeled_value": 0.15,
                "absolute_difference": 0.03,
                "tolerance_abs": 0.08,
                "standardized_difference": 0.375,
                "age_pattern_weight": 0.93,
                "passes_weight_threshold": True,
                "source_note": "External check; not calibration.",
            },
        ],
    )
    _write_csv(
        tmp_path,
        "outputs/tables/age_pattern_scenario_ordering_sensitivity.csv",
        [
            {
                "ordering_basis": "all_profiles_unweighted",
                "scenario_class": "combined_stress_test_package",
                "class_rank": 1.0,
                "class_order_sequence": "combined_stress_test_package > high_transmission_blocking_vaccine_target",
                "qualitative_tier_order_sequence": "transmission_blocking_or_combined_top_tier",
                "rank_shift_vs_all_profiles": 0.0,
                "strict_class_order_changed_vs_all_profiles": False,
                "qualitative_ordering_changed_vs_all_profiles": False,
                "country_count": 10,
                "effective_country_weight_sum": 10.0,
            },
            {
                "ordering_basis": "external_age_pattern_pass_filter",
                "scenario_class": "combined_stress_test_package",
                "class_rank": 2.0,
                "class_order_sequence": "high_transmission_blocking_vaccine_target > combined_stress_test_package",
                "qualitative_tier_order_sequence": "transmission_blocking_or_combined_top_tier",
                "rank_shift_vs_all_profiles": 1.0,
                "strict_class_order_changed_vs_all_profiles": True,
                "qualitative_ordering_changed_vs_all_profiles": False,
                "country_count": 2,
                "effective_country_weight_sum": 2.0,
            },
        ],
    )

    paths = build_outputs(root=tmp_path)

    strategy = pd.read_csv(paths["strategy"])
    assert list(strategy["strategy"]) == ["combined_strategy", "next_generation_vaccine"]
    assert strategy.loc[0, "horizon_top_two_share"] == 47 / 50
    assert strategy.loc[0, "joint_psa_probability_rank_1_pooled"] == 0.77

    age = pd.read_csv(paths["age"])
    assert age.loc[0, "validation_status"] == "passes_weight_threshold"
    assert age.loc[0, "model_minus_external_percentage_points"] == 3.0

    ordering = pd.read_csv(paths["ordering"])
    assert "external_age_pattern_pass_filter" in set(ordering["ordering_basis"])
    assert paths["markdown"].read_text(encoding="utf-8").startswith(
        "# Rank, Horizon, and Age-Pattern Validation Summary"
    )
    assert paths["metadata"].exists()
