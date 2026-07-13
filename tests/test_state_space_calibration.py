from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src_python.calibration.calibrate_baseline import (
    _annual_ar1_transition_scales,
    _gauss_newton_covariance,
    _negative_binomial_deviance_residuals,
    extend_annual_log_beta_conditional_mean,
)
from src_python.calibration.likelihood import negative_binomial_nll


def test_nb_deviance_residuals_equal_likelihood_ratio_to_saturated_model() -> None:
    observed = np.array([0.0, 3.0, 20.0, 500.0])
    mean = np.array([0.5, 5.0, 12.0, 700.0])
    dispersion = 2.5

    residual = _negative_binomial_deviance_residuals(
        observed,
        mean,
        dispersion,
    )
    likelihood_ratio = negative_binomial_nll(
        observed,
        mean,
        dispersion,
    ) - negative_binomial_nll(
        observed,
        np.maximum(observed, 1e-12),
        dispersion,
    )

    assert 0.5 * float(np.sum(residual**2)) == pytest.approx(
        likelihood_ratio,
        rel=1e-9,
        abs=2e-9,
    )
    assert np.sign(residual).tolist() == [-1.0, -1.0, 1.0, -1.0]


def test_annual_ar1_scales_partial_year_by_actual_midpoint_distance() -> None:
    midpoints = pd.Series(pd.to_datetime(["2024-01-01", "2024-07-01", "2025-07-01"]))
    stationary_sd, transition_rho, transition_sd = _annual_ar1_transition_scales(
        midpoints,
        rho=0.5,
        innovation_sd=0.6,
    )

    assert stationary_sd == pytest.approx(0.6 / np.sqrt(1.0 - 0.5**2))
    assert transition_rho[0] > transition_rho[1]
    assert transition_sd[0] < transition_sd[1]
    expected_delta = 182.0 / 365.2425
    assert transition_rho[0] == pytest.approx(0.5**expected_delta)
    assert transition_sd[0] == pytest.approx(
        stationary_sd * np.sqrt(1.0 - transition_rho[0] ** 2)
    )


def test_annual_ar1_rejects_non_increasing_midpoints() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _annual_ar1_transition_scales(
            pd.Series(pd.to_datetime(["2025-01-01", "2025-01-01"])),
            rho=0.5,
            innovation_sd=0.6,
        )


def test_gauss_newton_covariance_matches_linear_gaussian_information() -> None:
    jacobian = np.array([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    covariance, rank, condition = _gauss_newton_covariance(jacobian)

    assert covariance == pytest.approx(np.linalg.inv(jacobian.T @ jacobian))
    assert rank == 2
    assert condition >= 1.0


def test_gauss_newton_covariance_reports_rank_deficiency() -> None:
    covariance, rank, condition = _gauss_newton_covariance(
        np.array([[1.0, 1.0], [2.0, 2.0]])
    )

    assert rank == 1
    assert np.isfinite(covariance).all()
    assert condition > 1e12


def test_conditional_mean_process_extension_is_annual_and_nonmutating() -> None:
    config = {
        "transmission": {
            "log_beta_time_variation": {
                "ar1_rho": 0.25,
                "innovation_sd": 0.6,
                "periods": [
                    {
                        "start_date": "2026-01-01",
                        "end_date": "2026-12-31",
                        "log_multiplier": 0.8,
                    }
                ],
            }
        }
    }
    extended = extend_annual_log_beta_conditional_mean(config, through_year=2028)
    periods = extended["transmission"]["log_beta_time_variation"]["periods"]

    assert [period["log_multiplier"] for period in periods] == pytest.approx([0.8, 0.2, 0.05])
    assert periods[-1]["end_date"] == "2028-12-31"
    assert len(config["transmission"]["log_beta_time_variation"]["periods"]) == 1
