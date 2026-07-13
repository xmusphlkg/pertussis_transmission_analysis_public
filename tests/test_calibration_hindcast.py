from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src_python.validation.run_calibration_hindcast import (
    extend_ar1_conditional_mean_path,
    hindcast_execution_failures,
    last_year_naive_means,
    rolling_year_splits,
    summarize_hindcasts,
)


def test_rolling_year_splits_never_leak_test_year_into_training() -> None:
    observed = pd.DataFrame(
        {
            "observed_year": [2020, 2021, 2022, 2023, 2024, 2025],
            "reported_cases": [1, 2, 3, 4, 5, 6],
        }
    )
    splits = rolling_year_splits(observed, minimum_training_years=4)

    assert len(splits) == 2
    assert splits[0][0]["observed_year"].tolist() == [2020, 2021, 2022, 2023]
    assert splits[0][1]["observed_year"].tolist() == [2024]
    assert splits[1][0]["observed_year"].max() == 2024
    assert splits[1][1]["observed_year"].tolist() == [2025]


def test_ar1_hindcast_extension_decays_last_filtered_state() -> None:
    calibrated = {
        "transmission": {
            "log_beta_time_variation": {
                "ar1_rho": 0.5,
                "periods": [
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2024-12-31",
                        "log_multiplier": 0.8,
                    }
                ],
            }
        }
    }
    extended = extend_ar1_conditional_mean_path(calibrated, through_year=2026)
    periods = extended["transmission"]["log_beta_time_variation"]["periods"]

    assert [row["log_multiplier"] for row in periods] == pytest.approx([0.8, 0.4, 0.2])
    assert periods[-1]["end_date"] == "2026-12-31"
    assert calibrated["transmission"]["log_beta_time_variation"]["periods"][-1][
        "start_date"
    ] == "2024-01-01"


def test_last_year_naive_scales_for_partial_surveillance_exposure() -> None:
    training = pd.DataFrame(
        {
            "observed_year": [2023, 2024],
            "reported_cases": [80.0, 100.0],
            "interval_days": [80.0, 100.0],
        }
    )
    test = pd.DataFrame(
        {
            "observed_year": [2025],
            "reported_cases": [75.0],
            "interval_days": [50.0],
        }
    )

    assert last_year_naive_means(training, test).tolist() == pytest.approx([50.0])


def _hindcast_rows(*, wide: bool = False) -> pd.DataFrame:
    q025, q975 = (0.0, 1_000_000.0) if wide else (8.0, 12.0)
    width = q975 - q025
    return pd.DataFrame(
        [
            {
                "country": "Example",
                "fold": fold,
                "test_year": 2023 + fold,
                "training_end_year": 2022 + fold,
                "evidence_cutoff_date": f"{2023 + fold}-01-01",
                "resistance_evidence_anchor_year": 2022 + fold,
                "resistance_timeline_applied": True,
                "resistance_evidence_years": str(2022 + fold),
                "diagnostic_periods_known_at_origin": 0,
                "future_evidence_used": False,
                "optimizer_success": True,
                "observed_cases": 10.0,
                "predicted_cases": 10.0,
                "naive_predicted_cases": 13.0,
                "absolute_log1p_error": 0.05,
                "naive_absolute_log1p_error": 0.20,
                "negative_binomial_nll": 5.0,
                "naive_negative_binomial_nll": 7.0,
                "process_mixture_negative_binomial_nll": 5.0,
                "absolute_log1p_error_improvement_over_naive": 0.15,
                "negative_binomial_nll_improvement_over_naive": 2.0,
                "process_mixture_negative_binomial_nll_improvement_over_naive": 2.0,
                "smape": 0.0,
                "nb95_interval_coverage": 1.0,
                "naive_nb95_interval_coverage": 1.0,
                "nb95_interval_mean_width": 4.0,
                "naive_nb95_interval_mean_width": 5.0,
                "nb95_interval_mean_relative_width": 0.4,
                "naive_nb95_interval_mean_relative_width": 0.5,
                "process_predictive_mean_q025": q025,
                "process_predictive_mean_median": 10.0,
                "process_predictive_mean_q975": q975,
                "observation_predictive_q025": q025,
                "observation_predictive_median": 10.0,
                "observation_predictive_q975": q975,
                "observation_predictive_95_coverage": True,
                "observation_predictive_95_width": width,
                "observation_predictive_95_relative_width": width / 10.0,
                "observation_predictive_95_log1p_width": float(
                    np.log1p(q975) - np.log1p(q025)
                ),
                "laplace_boundary_rejection_fraction": 0.0,
                "predictive_process_draws": 8,
                "predictive_measurement_draws": 32,
                "posterior_rank": 6,
                "posterior_dimension": 6,
                "posterior_condition_number": 10.0,
                "process_nfev": 12,
                "state_coordinate_uncertainty_mode": "map_conditioned",
                "predictive_uncertainty_scope": (
                    "conditional_on_MAP_state_with_future_AR1_process_and_NB2_measurement"
                ),
                "posterior_coverage_claimed": False,
            }
            for fold in (1, 2, 3)
        ]
    )


def test_predictive_adequacy_requires_scores_coverage_and_sharpness() -> None:
    summary = summarize_hindcasts(_hindcast_rows(), {"Example": 3})
    country = summary.loc[summary["assessment_scope"].eq("country")].iloc[0]
    overall = summary.loc[summary["assessment_scope"].eq("overall")].iloc[0]

    assert bool(country["predictive_adequacy_pass"])
    assert country["predictive_adequacy_status"] == "pass_limited_fold_evidence"
    assert bool(overall["predictive_adequacy_pass"])


def test_predictive_adequacy_rejects_covered_but_non_discriminating_interval() -> None:
    summary = summarize_hindcasts(_hindcast_rows(wide=True), {"Example": 3})
    country = summary.loc[summary["assessment_scope"].eq("country")].iloc[0]

    assert not bool(country["predictive_adequacy_pass"])
    assert (
        "observation_predictive_interval_non_discriminating"
        in country["predictive_adequacy_reasons"]
    )


def test_uncorrected_gaussian_state_approximation_cannot_pass_publication_gate() -> None:
    rows = _hindcast_rows()
    rows["state_coordinate_uncertainty_mode"] = "gaussian_approximation"

    summary = summarize_hindcasts(rows, {"Example": 3})
    country = summary.loc[summary["assessment_scope"].eq("country")].iloc[0]

    assert not bool(country["predictive_adequacy_pass"])
    assert (
        "uncorrected_gaussian_state_approximation_not_publication_valid"
        in country["predictive_adequacy_reasons"]
    )


def test_hindcast_execution_gate_detects_missing_and_nonfinite_folds() -> None:
    result = _hindcast_rows().iloc[:1].copy()
    result.loc[:, "negative_binomial_nll"] = float("nan")

    failures = hindcast_execution_failures(result, {"Example": 3})

    assert any("expected 3 folds but obtained 1" in failure for failure in failures)
    assert any("negative_binomial_nll" in failure for failure in failures)


def test_hindcast_execution_gate_requires_uncertainty_scope_contract() -> None:
    missing = _hindcast_rows().drop(columns=["predictive_uncertainty_scope"])
    missing_failures = hindcast_execution_failures(missing, {"Example": 3})
    mislabelled = _hindcast_rows()
    mislabelled["posterior_coverage_claimed"] = True
    mislabelled_failures = hindcast_execution_failures(
        mislabelled,
        {"Example": 3},
    )

    assert any("missing required columns" in failure for failure in missing_failures)
    assert any(
        "must not claim posterior interval coverage" in failure
        for failure in mislabelled_failures
    )


def test_single_fold_is_complete_but_not_claimed_predictively_adequate() -> None:
    result = _hindcast_rows().iloc[:1].copy()
    summary = summarize_hindcasts(result, {"Example": 1})
    country = summary.loc[summary["assessment_scope"].eq("country")].iloc[0]

    assert bool(country["execution_complete"])
    assert not bool(country["predictive_adequacy_pass"])
    assert (
        country["predictive_adequacy_reasons"]
        == "insufficient_folds_for_predictive_assessment"
    )
