from __future__ import annotations

import numpy as np
import pandas as pd

from src_python.calibration.mcmc_diagnostics import compute_diagnostics
from src_python.model.parameters import PreparedParameters
from src_python.simulation import common as simulation_common
from src_python.simulation.common import load_calibrated_country_artifact, load_configs, make_config
from src_python.simulation.run_bayesian_uncertainty import (
    _aggregate_observed_intervals,
    _apply_prior_sd_overrides,
    _apply_sample,
    _artifact_stem,
    _country_observed,
    _diagnostic_sample_columns,
    _evaluate_grid_values,
    _initial_vector,
    _estimate_local_mode_half_width,
    _integration_grid_log_posterior,
    _normalised_grid_weights,
    _sample_from_vector,
    _sample_columns,
    _vector_from_sample,
    _weighted_grid_quantiles,
)
from src_python.utils.io import read_table, write_dataframe, write_yaml


def test_monthly_bayesian_observation_aggregation_conserves_cases():
    native = _country_observed("United_States", interval="native")
    monthly = _aggregate_observed_intervals(native, "monthly")

    assert len(monthly) < len(native)
    assert np.isclose(monthly["reported_cases"].sum(), native["reported_cases"].sum())
    assert monthly["period_end"].gt(monthly["period_start"]).all()
    assert monthly["reporting_frequency"].eq("monthly_aggregated").all()


def test_time_varying_reporting_sample_is_wired_into_model_parameters():
    config = make_config(country_profile="Australia")
    config["simulation"]["start_time"] = 0.0
    config["simulation"]["end_time"] = 365.0
    sample = {
        "beta_S": float(config["transmission"]["beta_S"]),
        "reporting_multiplier": 0.5,
        "VE_sus": 0.25,
        "VE_inf": 0.25,
        "VE_dur": 0.10,
        "relative_infectiousness_asymptomatic": 0.45,
        "infectious_duration_symptomatic": 21.0,
        "infectious_duration_asymptomatic": 14.0,
        "fitness_R": 1.0,
        "resistance_prevalence": 0.05,
        "reporting_trend_end_multiplier": 1.2,
    }

    updated = _apply_sample(config, sample)
    params = PreparedParameters.from_config(
        updated,
        analysis="test",
        scenario="reporting_trend",
    )

    assert updated["reporting_time_variation"]["end_multiplier"] == 1.2
    assert np.allclose(params.reporting_rate_at(0.0), params.reporting_rate)
    assert np.allclose(params.reporting_rate_at(365.0), params.reporting_rate * 1.2)


def test_stale_calibration_artifact_is_not_loaded_by_default(monkeypatch, tmp_path):
    baseline_beta = float(load_configs()["baseline"]["transmission"]["beta_S"])
    stale_artifact_path = tmp_path / "Australia_calibrated_config.yaml"
    write_yaml(
        {
            "config": {"transmission": {"beta_S": baseline_beta * 2.0}},
            "metadata": {
                "accepted": True,
                "config_hash": "stale-config-hash",
                "calibration_config_hash": "stale-calibration-hash",
                "calibration_status": "calibrated_to_reported_cases",
            },
        },
        stale_artifact_path,
    )

    simulation_common._load_calibrated_country_artifact_cached.cache_clear()
    monkeypatch.setattr(
        simulation_common,
        "calibrated_country_artifact_path",
        lambda country: stale_artifact_path,
    )
    try:
        config = make_config(country_profile="Australia")

        assert not bool(config["metadata"]["calibration_loaded"])
        assert float(config["transmission"]["beta_S"]) == baseline_beta

        artifact = load_calibrated_country_artifact("Australia", allow_stale=True)
        assert artifact is not None
        assert artifact["metadata"]["calibration_hash_status"] == "stale_parameter_overlay"
    finally:
        simulation_common._load_calibrated_country_artifact_cached.cache_clear()


def test_convergence_diagnostics_exclude_fixed_mcmc_columns():
    diagnostic_columns = set(_diagnostic_sample_columns())

    assert diagnostic_columns.issubset(set(_sample_columns()))
    assert "VE_dur" not in diagnostic_columns
    assert "resistance_prevalence" not in diagnostic_columns
    assert "reporting_trend_end_multiplier" not in diagnostic_columns


def test_mcmc_fixed_ve_duration_uses_calibrated_value():
    config = make_config(country_profile="South_Africa")
    priors = dict(load_configs()["baseline"]["bayesian_uncertainty"]["priors"])
    priors["resistance_prevalence_fixed"] = 0.02
    priors["reporting_trend_fixed"] = 1.0
    priors["VE_dur_fixed"] = float(config["vaccine"]["VE_dur"])

    sample = _sample_from_vector(_initial_vector(config, enable_trend=False), priors)

    assert sample["VE_dur"] == config["vaccine"]["VE_dur"]


def test_compute_diagnostics_drops_constant_parameters():
    samples = pd.DataFrame(
        {
            "country": ["X"] * 8,
            "chain": [1, 1, 1, 1, 2, 2, 2, 2],
            "varying": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41],
            "fixed": [14.0] * 8,
        }
    )

    diagnostics = compute_diagnostics(
        samples,
        parameter_columns=("varying", "fixed"),
        chain_column="chain",
        country_column="country",
    )

    assert diagnostics["parameter"].tolist() == ["varying"]


def test_prior_sd_overrides_tighten_requested_priors():
    settings = {
        "priors": {
            "log_beta_S_sd": 0.8,
            "log_reporting_multiplier_sd": 0.8,
            "VE_sus": {"sd": 0.05},
            "VE_inf": {"sd": 0.05},
            "relative_infectiousness_asymptomatic": {"sd": 0.10},
            "infectious_duration_symptomatic": {"log_sd": 0.15},
            "infectious_duration_asymptomatic": {"log_sd": 0.20},
            "fitness_R": {"sd": 0.12},
        }
    }

    _apply_prior_sd_overrides(
        settings,
        prior_sd_scale=0.5,
        ve_prior_sd=0.02,
        reporting_prior_log_sd=0.3,
    )

    priors = settings["priors"]
    assert priors["log_beta_S_sd"] == 0.4
    assert priors["log_reporting_multiplier_sd"] == 0.3
    assert priors["VE_sus"]["sd"] == 0.02
    assert priors["VE_inf"]["sd"] == 0.02
    assert priors["relative_infectiousness_asymptomatic"]["sd"] == 0.05
    assert priors["fitness_R"]["sd"] == 0.06


def test_pilot_artifact_stems_do_not_reuse_canonical_names():
    assert _artifact_stem("bayesian_uncertainty", "bayesian_posterior_samples", "posterior_samples") == "bayesian_posterior_samples"
    assert _artifact_stem("pilot_slice", "bayesian_posterior_samples", "posterior_samples") == "pilot_slice_posterior_samples"


def test_beta_reporting_product_parameterization_round_trips():
    config = make_config(country_profile="South_Africa")
    priors = dict(load_configs()["baseline"]["bayesian_uncertainty"]["priors"])
    priors["parameterization"] = "beta_reporting_product"
    priors["base_log_beta_S"] = np.log(float(config["transmission"]["beta_S"]))
    priors["resistance_prevalence_fixed"] = float(config["resistance"]["target_prevalence_at_analysis_start"])
    priors["reporting_trend_fixed"] = 1.0
    priors["VE_dur_fixed"] = float(config["vaccine"]["VE_dur"])

    vector = _initial_vector(config, enable_trend=False, priors=priors)
    sample = _sample_from_vector(vector, priors)
    round_trip = _vector_from_sample(sample, priors)

    assert np.allclose(vector, round_trip)
    assert np.isclose(sample["beta_S"], config["transmission"]["beta_S"])
    assert np.isclose(sample["reporting_multiplier"], config.get("reporting_multiplier", 1.0))


def test_beta_reporting_product_coordinate_preserves_case_scale_when_beta_moves():
    config = make_config(country_profile="South_Africa")
    priors = dict(load_configs()["baseline"]["bayesian_uncertainty"]["priors"])
    priors["parameterization"] = "beta_reporting_product"
    priors["base_log_beta_S"] = np.log(float(config["transmission"]["beta_S"]))
    priors["resistance_prevalence_fixed"] = float(config["resistance"]["target_prevalence_at_analysis_start"])
    priors["reporting_trend_fixed"] = 1.0
    priors["VE_dur_fixed"] = float(config["vaccine"]["VE_dur"])

    vector = _initial_vector(config, enable_trend=False, priors=priors)
    moved = vector.copy()
    moved[0] += np.log(1.25)
    sample = _sample_from_vector(moved, priors)

    baseline_product = float(config["transmission"]["beta_S"]) * float(config.get("reporting_multiplier", 1.0))
    moved_product = sample["beta_S"] * sample["reporting_multiplier"]
    assert np.isclose(moved_product, baseline_product)


def test_weighted_grid_quantiles_follow_grid_cdf():
    grid = np.array([0.0, 1.0, 2.0])
    weights = np.array([0.25, 0.50, 0.25])
    probs = np.array([0.125, 0.50, 0.875])

    quantiles = _weighted_grid_quantiles(grid, weights, probs)

    assert np.allclose(quantiles, [0.0, 0.5, 1.5])


def test_parallel_grid_value_evaluation_preserves_grid_order(tmp_path):
    grid = np.linspace(-2.0, 2.0, 17)

    def evaluator(x: float) -> float:
        return x * x - 0.5 * x

    serial = _evaluate_grid_values(grid, evaluator, n_jobs=1)
    parallel = _evaluate_grid_values(
        grid,
        evaluator,
        n_jobs=2,
        progress_file=tmp_path / "grid_progress.txt",
        half_width_value=0.1,
    )

    assert np.allclose(parallel, serial)
    assert "grid 17/17" in (tmp_path / "grid_progress.txt").read_text()


def test_normalised_grid_weights_report_tail_and_resolution_quality():
    logp = np.array([-25.0, -1.0, 0.0, -1.0, -30.0])

    weights, quality = _normalised_grid_weights(logp)

    assert np.isclose(weights.sum(), 1.0)
    assert quality["min_edge_drop"] == 25.0
    assert quality["grid_effective_points"] > 1.0
    assert quality["grid_max_weight"] < 1.0


def test_local_mode_half_width_shrinks_around_sharp_peak():
    grid = np.linspace(-1.0, 1.0, 101)
    logp = -0.5 * (grid / 0.02) ** 2

    half_width = _estimate_local_mode_half_width(grid, logp, tail_drop_target=20.0)

    assert half_width is not None
    assert 0.07 < half_width < 0.16


def test_auto_grid_smoothing_only_activates_for_collapsed_weights():
    smooth_logp = -0.5 * (np.linspace(-2.0, 2.0, 21) ** 2)
    used_logp, method = _integration_grid_log_posterior(
        smooth_logp,
        smoothing="auto",
        savgol_window=9,
    )
    assert method == "none"
    assert np.allclose(used_logp, smooth_logp)

    spiky = np.full(21, -20.0)
    spiky[10] = 0.0
    used_logp, method = _integration_grid_log_posterior(
        spiky,
        smoothing="auto",
        savgol_window=9,
    )
    _, raw_quality = _normalised_grid_weights(spiky)
    _, smooth_quality = _normalised_grid_weights(used_logp)
    assert method == "savgol"
    assert smooth_quality["grid_max_weight"] < raw_quality["grid_max_weight"]


def test_large_simulation_write_removes_stale_csv(tmp_path):
    path = tmp_path / "outputs" / "simulations" / "bayesian_posterior_samples.csv"
    path.parent.mkdir(parents=True)
    path.write_text("x\n999\n", encoding="utf-8")

    write_dataframe(pd.DataFrame({"x": [1, 2]}), path)

    assert not path.exists()
    loaded = read_table(path)
    assert loaded["x"].tolist() == [1, 2]
