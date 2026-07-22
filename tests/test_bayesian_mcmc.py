from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src_python.simulation import run_bayesian_uncertainty as bayesian_uncertainty
from src_python.calibration.mcmc_diagnostics import compute_diagnostics, summarize_convergence
from src_python.calibration.calibrate_baseline import (
    _annual_ar1_transition_scales,
    aggregate_projected_likelihood_means,
    grouped_likelihood_observations,
)
from src_python.model.parameters import PreparedParameters
from src_python.simulation import common as simulation_common
from src_python.simulation.bayesian_priors import (
    BAYESIAN_LOCAL_STATE_PARAMETER_NAMES,
    BAYESIAN_PARAMETER_CONSUMERS,
    BAYESIAN_PARAMETER_NAMES,
    BAYESIAN_SHARED_PARAMETER_NAMES,
    resolve_bayesian_prior_specs,
)
from src_python.simulation.common import load_calibrated_country_artifact, load_configs, make_config
from src_python.simulation.parameter_distributions import inverse_cdf, log_pdf
from src_python.simulation.run_bayesian_uncertainty import (
    JOINT_IMPORTANCE_SAMPLER,
    SMC_SAMPLER,
    _compute_importance_diagnostics,
    _compute_state_space_exact_importance_diagnostics,
    _aggregate_observed_intervals,
    _apply_prior_sd_overrides,
    _apply_sample,
    _artifact_stem,
    _bounded_multivariate_normal_draws,
    _country_observed,
    _diagnostic_sample_columns,
    _evaluate_grid_values,
    _full_vector_from_unit_cube,
    _general_gaussian_mixture_proposal,
    _initial_vector,
    _latin_hypercube,
    _nuisance_vector_from_unit_cube,
    _estimate_local_mode_half_width,
    _integration_grid_log_posterior,
    _importance_pareto_shape,
    _log_prior,
    _modular_cut_candidate_weights,
    _modular_cut_conditional_grid_diagnostics,
    _minimum_importance_tail_ess,
    _normalised_grid_weights,
    _posterior_predictive_scenarios,
    _reflect_component_into_bounds,
    _resample_smc_island_particles,
    _sample_from_vector,
    _sample_columns,
    _state_gaussian_mixture_proposal,
    _state_importance_quality_failures,
    _state_space_exact_negative_log_target,
    _tempered_importance_adaptation,
    _timeseries_with_reporting_multiplier,
    _vector_from_sample,
    _weighted_grid_quantiles,
    _write_interval_summaries,
)
from src_python.utils.io import read_table, write_dataframe, write_yaml


def test_monthly_bayesian_observation_aggregation_conserves_cases():
    native = _country_observed("United_States", interval="native")
    monthly = _aggregate_observed_intervals(native, "monthly")

    assert len(monthly) < len(native)
    assert np.isclose(monthly["reported_cases"].sum(), native["reported_cases"].sum())
    assert monthly["period_end"].gt(monthly["period_start"]).all()
    assert monthly["reporting_frequency"].eq("monthly_aggregated").all()


def test_conditional_interval_outputs_do_not_emit_credible_interval_aliases(
    monkeypatch,
) -> None:
    written: list[pd.DataFrame] = []
    monkeypatch.setattr(
        bayesian_uncertainty,
        "write_dataframe",
        lambda frame, _path: written.append(frame.copy()),
    )
    summary = pd.DataFrame(
        {
            "country": ["A", "A", "A"],
            "annualized_reported_cases_per_100k": [9.0, 10.0, 12.0],
        }
    )
    samples = pd.DataFrame(
        {
            "country": ["A", "A", "A"],
            "beta_S": [0.03, 0.031, 0.032],
            "inference_structure": [
                "reference_structure_state_space_exact_importance_cut"
            ]
            * 3,
        }
    )

    _write_interval_summaries(summary, samples, output_stem="test")

    assert len(written) == 2
    for frame in written:
        assert not any("credible_interval" in column for column in frame.columns)
        assert not any(column.startswith("posterior_") for column in frame.columns)
        assert frame["interval_type"].eq(
            "95% conditional uncertainty interval"
        ).all()


def test_reporting_fast_path_retains_diagnostic_standard_timeline_multiplier():
    timeseries = pd.DataFrame(
        {
            "age_group": ["child", "child"],
            "time": [0.0, 1.0],
            "symptomatic_case_rate_per_day": [10.0, 10.0],
            "diagnostic_reporting_multiplier": [0.5, 1.5],
        }
    )
    base = {"age_groups": [{"label": "child", "reporting_rate": 0.2}]}

    reported = _timeseries_with_reporting_multiplier(timeseries, base, 2.0)

    assert reported["reported_case_rate_per_day"].tolist() == pytest.approx([2.0, 6.0])


def test_modular_cut_keeps_each_structural_draw_equally_weighted() -> None:
    candidates = pd.DataFrame(
        {
            "nuisance_index": [0, 0, 1, 1],
            "importance_log_weight": [0.0, np.log(3.0), np.log(9.0), np.log(1.0)],
        }
    )

    weights, positions, probabilities = _modular_cut_candidate_weights(
        candidates,
        nuisance_draws=2,
    )

    assert weights == pytest.approx([0.125, 0.375, 0.45, 0.05])
    assert weights[positions[0]].sum() == pytest.approx(0.5)
    assert weights[positions[1]].sum() == pytest.approx(0.5)
    assert probabilities[0] == pytest.approx([0.25, 0.75])
    assert probabilities[1] == pytest.approx([0.9, 0.1])


def test_state_gaussian_mixture_proposal_uses_normalized_density() -> None:
    candidates, log_density, source, scales = _state_gaussian_mixture_proposal(
        np.asarray([0.0]),
        np.asarray([[1.0]]),
        (0.5, 2.0),
        n=200,
        seed=19,
    )

    x = candidates[:, 0]
    component_1 = np.exp(-0.5 * x**2 / 0.5) / np.sqrt(2.0 * np.pi * 0.5)
    component_2 = np.exp(-0.5 * x**2 / 2.0) / np.sqrt(2.0 * np.pi * 2.0)
    expected = np.log(0.5 * component_1 + 0.5 * component_2)

    assert np.allclose(log_density, expected)
    assert np.bincount(source, minlength=2).tolist() == [100, 100]
    assert scales.tolist() == pytest.approx([0.5, 2.0])


def test_importance_tail_diagnostics_distinguish_uniform_and_concentrated_support() -> None:
    candidates = np.column_stack(
        (np.linspace(-2.0, 2.0, 1000), np.linspace(3.0, -3.0, 1000))
    )
    uniform = np.full(1000, 1.0 / 1000.0)
    concentrated = np.full(1000, 0.1 / 999.0)
    concentrated[0] = 0.9

    assert _importance_pareto_shape(np.zeros(1000)) == pytest.approx(0.0)
    assert _minimum_importance_tail_ess(candidates, uniform) >= 20.0
    assert _minimum_importance_tail_ess(candidates, concentrated) < 5.0


def test_state_importance_publication_gate_rejects_each_unstable_metric() -> None:
    passing = {
        "effective_sample_size": 250.0,
        "maximum_weight": 0.01,
        "pareto_k": 0.5,
        "minimum_tail_ess": 25.0,
        "out_of_bounds_fraction": 0.1,
        "require_recommended": True,
    }
    assert _state_importance_quality_failures(**passing) == []

    for key, value in {
        "effective_sample_size": 199.0,
        "maximum_weight": 0.021,
        "pareto_k": 0.71,
        "minimum_tail_ess": 19.0,
        "out_of_bounds_fraction": 0.5,
    }.items():
        candidate = passing | {key: value}
        assert key in _state_importance_quality_failures(**candidate)


def test_state_importance_diagnostics_do_not_mislabel_support_as_mcmc_ess() -> None:
    samples = pd.DataFrame(
        {
            "country": ["Example"] * 100,
            "chain": np.repeat([1, 2], 50),
            "beta_S": np.linspace(0.03, 0.05, 100),
            "state_posterior_rank": 3,
            "state_posterior_dimension": 3,
            "state_posterior_condition_number": 10.0,
            "state_laplace_boundary_rejection_fraction": 0.1,
            "state_importance_candidate_count": 1000,
            "state_importance_effective_sample_size": 250.0,
            "state_importance_max_weight": 0.01,
            "state_importance_pareto_k": 0.4,
            "state_importance_min_tail_ess": 25.0,
            "structural_prior_design_size": 128,
        }
    )

    diagnostics = _compute_state_space_exact_importance_diagnostics(
        samples,
        ("beta_S",),
    )

    assert diagnostics["rhat"].isna().all()
    assert diagnostics["bulk_ess"].isna().all()
    assert diagnostics["tail_ess"].isna().all()
    assert diagnostics["parameter_effective_support_size"].iloc[0] == pytest.approx(100.0)


def test_exact_state_target_reuses_calendar_aware_ar1_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    years = [2024, 2025, 2026]
    coordinate = np.asarray([np.log(0.04), np.log(1.2), 1.2, -0.8, 0.7])
    runtime = {
        "transmission": {
            "beta_S": 0.04,
            "log_beta_time_variation": {
                "periods": [
                    {
                        "start_date": f"{year}-01-01",
                        "end_date": f"{year}-12-31",
                        "log_multiplier": 0.0,
                    }
                    for year in years
                ]
            },
        },
        "reporting_multiplier": 1.2,
    }
    observed = pd.DataFrame({"reported_cases": [1.0]})
    monkeypatch.setattr(
        bayesian_uncertainty,
        "_calibration_predicted_means",
        lambda *_args, **_kwargs: np.asarray([1.0]),
    )
    monkeypatch.setattr(
        bayesian_uncertainty,
        "grouped_likelihood_observations",
        lambda _observed: observed,
    )
    monkeypatch.setattr(
        bayesian_uncertainty,
        "negative_binomial_nll",
        lambda *_args, **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        bayesian_uncertainty,
        "reporting_rate_prior_penalty",
        lambda _config: 0.0,
    )

    actual = _state_space_exact_negative_log_target(
        coordinate,
        runtime_base=runtime,
        prior_base=runtime,
        observed=observed,
        country="Australia",
        coordinate_names=[
            "log_beta_S",
            "log_reporting_multiplier",
            *[f"log_beta_process_{year}" for year in years],
        ],
        rho=0.5,
        innovation_sd=0.6,
        beta_prior_sd=1.0,
        reporting_prior_sd=1.0,
        dispersion=50.0,
    )
    starts = pd.Series(pd.to_datetime([f"{year}-01-01" for year in years]))
    ends = pd.Series(pd.to_datetime([f"{year + 1}-01-01" for year in years]))
    stationary_sd, transition_rho, transition_sd = _annual_ar1_transition_scales(
        starts + (ends - starts) / 2,
        rho=0.5,
        innovation_sd=0.6,
    )
    latent = coordinate[2:]
    expected = 0.5 * (
        (latent[0] / stationary_sd) ** 2
        + np.sum(
            ((latent[1:] - transition_rho * latent[:-1]) / transition_sd) ** 2
        )
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_general_gaussian_mixture_scores_distinct_component_means() -> None:
    candidates, log_density, source = _general_gaussian_mixture_proposal(
        np.asarray([[-1.0], [2.0]]),
        np.asarray([[[0.5]], [[1.5]]]),
        np.asarray([0.25, 0.75]),
        n=400,
        seed=17,
    )

    x = candidates[:, 0]
    first = np.exp(-0.5 * (x + 1.0) ** 2 / 0.5) / np.sqrt(2.0 * np.pi * 0.5)
    second = np.exp(-0.5 * (x - 2.0) ** 2 / 1.5) / np.sqrt(2.0 * np.pi * 1.5)
    assert np.allclose(log_density, np.log(0.25 * first + 0.75 * second))
    assert np.bincount(source, minlength=2).tolist() == [100, 300]


def test_state_gaussian_mixture_importance_recovers_target_moments() -> None:
    candidates, log_proposal, _source, _scales = _state_gaussian_mixture_proposal(
        np.asarray([0.0]),
        np.asarray([[1.0]]),
        (0.125, 0.5, 2.0),
        n=30000,
        seed=23,
    )
    x = candidates[:, 0]
    target_log_density = -0.5 * (np.log(2.0 * np.pi) + x**2)
    log_weight = target_log_density - log_proposal
    weight = np.exp(log_weight - np.max(log_weight))
    weight /= weight.sum()

    assert float(np.sum(weight * x)) == pytest.approx(0.0, abs=0.025)
    assert float(np.sum(weight * x**2)) == pytest.approx(1.0, abs=0.04)


def test_tempered_importance_adaptation_stabilizes_concentrated_weights() -> None:
    rng = np.random.default_rng(29)
    candidates = rng.normal(size=(400, 2))
    log_weight = -12.0 * (
        (candidates[:, 0] - 0.8) ** 2
        + (candidates[:, 1] + 0.4) ** 2
    )

    mean, covariance, exponent, raw_ess = _tempered_importance_adaptation(
        candidates,
        log_weight,
        np.eye(2),
        target_ess=80.0,
        covariance_regularization=0.1,
    )

    tempered = np.exp(exponent * log_weight - np.max(exponent * log_weight))
    tempered /= tempered.sum()
    tempered_ess = 1.0 / np.sum(tempered**2)
    assert raw_ess < 80.0
    assert 0.0 < exponent < 1.0
    assert tempered_ess == pytest.approx(80.0, rel=1e-6)
    assert mean[0] > 0.45
    assert mean[1] < -0.15
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def _modular_grid_candidates(
    local_probabilities: list[np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for nuisance_index, probability in enumerate(local_probabilities):
        beta_points, reporting_points = probability.shape
        for beta_index in range(beta_points):
            for reporting_index in range(reporting_points):
                mass = float(probability[beta_index, reporting_index])
                rows.append(
                    {
                        "nuisance_index": nuisance_index,
                        "beta_grid_index": beta_index,
                        "reporting_grid_index": reporting_index,
                        "beta_grid_edge": beta_index in {0, beta_points - 1},
                        "reporting_grid_edge": reporting_index
                        in {0, reporting_points - 1},
                        "importance_log_weight": np.log(mass)
                        if mass > 0.0
                        else -np.inf,
                    }
                )
    return pd.DataFrame(rows)


def test_modular_cut_conditional_gate_detects_local_edge_mass_hidden_by_global_average() -> None:
    edge = np.zeros((7, 7), dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    bad = np.zeros((7, 7), dtype=float)
    bad[edge] = 0.08 / float(edge.sum())
    bad[~edge] = 0.92 / float((~edge).sum())
    good = np.zeros((7, 7), dtype=float)
    good[~edge] = 1.0 / float((~edge).sum())
    candidates = _modular_grid_candidates([bad, good])
    weights, positions, probabilities = _modular_cut_candidate_weights(
        candidates,
        nuisance_draws=2,
    )

    global_edge = (
        candidates["beta_grid_edge"].astype(bool)
        | candidates["reporting_grid_edge"].astype(bool)
    ).to_numpy(dtype=bool)
    assert weights[global_edge].sum() == pytest.approx(0.04)

    diagnostics = _modular_cut_conditional_grid_diagnostics(
        candidates,
        positions,
        probabilities,
        nuisance_draws=2,
        beta_grid_points=7,
        reporting_grid_points=7,
    )

    assert diagnostics.loc[0, "conditional_combined_edge_weight"] == pytest.approx(0.08)
    assert not bool(diagnostics.loc[0, "fatal_valid"])
    assert "conditional_edge_weight" in diagnostics.loc[0, "fatal_issues"]
    assert bool(diagnostics.loc[1, "fatal_valid"])


def test_modular_cut_conditional_gate_checks_each_grid_axis_resolution() -> None:
    probability = np.zeros((5, 41), dtype=float)
    probability[2, 1:-1] = 1.0 / 39.0
    candidates = _modular_grid_candidates([probability])
    _, positions, probabilities = _modular_cut_candidate_weights(
        candidates,
        nuisance_draws=1,
    )

    diagnostics = _modular_cut_conditional_grid_diagnostics(
        candidates,
        positions,
        probabilities,
        nuisance_draws=1,
        beta_grid_points=5,
        reporting_grid_points=41,
    )
    row = diagnostics.iloc[0]

    assert row["conditional_effective_grid_points"] == pytest.approx(39.0)
    assert row["conditional_beta_effective_grid_points"] == pytest.approx(1.0)
    assert row["conditional_reporting_effective_grid_points"] == pytest.approx(39.0)
    assert "beta_axis_effective_points" in row["fatal_issues"]
    assert not bool(row["fatal_valid"])


def test_annual_bayesian_observation_aggregation_conserves_cases():
    native = _country_observed("Australia", interval="native")
    annual = _aggregate_observed_intervals(native, "annual")

    assert len(annual) < len(native)
    assert np.isclose(annual["reported_cases"].sum(), native["reported_cases"].sum())
    assert annual["period_end"].gt(annual["period_start"]).all()
    assert annual["reporting_frequency"].eq("annual_aggregated").all()


def test_annual_aggregation_preserves_integer_nb_count_support() -> None:
    native = _country_observed("United_Kingdom", interval="native")
    annual = _aggregate_observed_intervals(native, "annual")

    assert np.allclose(
        annual["reported_cases"].to_numpy(dtype=float),
        np.round(annual["reported_cases"].to_numpy(dtype=float)),
    )
    assert annual["aggregation_assignment"].eq("whole_interval_midpoint").all()


def test_aggregation_does_not_integrate_across_unobserved_gaps() -> None:
    native = pd.DataFrame(
        {
            "observed_interval_id": ["a", "b"],
            "period_start": pd.to_datetime(["2025-01-01", "2025-01-10"]),
            "period_end": pd.to_datetime(["2025-01-08", "2025-01-17"]),
            "reported_cases": [3.0, 4.0],
        }
    )
    annual = _aggregate_observed_intervals(native, "annual")

    assert len(annual) == 2
    assert annual["reported_cases"].tolist() == [3.0, 4.0]
    assert annual["interval_days"].tolist() == [7.0, 7.0]
    grouped = grouped_likelihood_observations(annual)
    predicted = aggregate_projected_likelihood_means(annual, np.array([2.5, 5.5]))
    assert grouped["reported_cases"].tolist() == [7.0]
    assert grouped["interval_days"].tolist() == [14.0]
    assert predicted.tolist() == pytest.approx([8.0])


def test_beta_reporting_product_reflection_respects_joint_bounds():
    config = make_config(country_profile="Australia")
    priors = load_configs()["baseline"]["bayesian_uncertainty"]["priors"].copy()
    priors["parameterization"] = "beta_reporting_product"
    priors["base_log_beta_S"] = float(np.log(config["transmission"]["beta_S"]))
    vector = _initial_vector(config, priors=priors)

    vector[0] = np.log(1e-8)
    reflected = _reflect_component_into_bounds(vector, 0, priors)
    sample = _sample_from_vector(reflected, priors)

    assert 0.0005 <= sample["beta_S"] <= 0.5
    assert 0.02 <= sample["reporting_multiplier"] <= 20.0


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


def test_bounded_laplace_draws_are_reproducible_and_inside_support() -> None:
    args = (
        np.array([0.0, 0.0]),
        np.array([[0.25, 0.08], [0.08, 0.16]]),
        np.array([-0.75, -0.75]),
        np.array([0.75, 0.75]),
    )
    first, rejection = _bounded_multivariate_normal_draws(*args, n=200, seed=17)
    second, _ = _bounded_multivariate_normal_draws(*args, n=200, seed=17)

    assert first == pytest.approx(second)
    assert np.logical_and(first >= -0.75, first <= 0.75).all()
    assert 0.0 <= rejection < 1.0


def test_apply_sample_propagates_latent_state_draw_into_history() -> None:
    config = make_config(country_profile="Australia", load_calibration=False)
    config["transmission"]["log_beta_time_variation"] = {
        "interpretation": "latent_AR1_process_MAP_point_estimate",
        "periods": [
            {"start_date": "2025-01-01", "end_date": "2025-12-31", "log_multiplier": 0.0},
            {"start_date": "2026-01-01", "end_date": "2026-12-31", "log_multiplier": 0.0},
        ],
    }
    history = deepcopy(config)
    config[simulation_common.PROSPECTIVE_POLICY_KEY] = {"history_config": history}
    sample = {
        "beta_S": float(config["transmission"]["beta_S"]),
        "reporting_multiplier": 1.0,
        "VE_sus": 0.2,
        "VE_inf": 0.2,
        "VE_dur": 0.1,
        "relative_infectiousness_asymptomatic": 0.5,
        "infectious_duration_symptomatic": 21.0,
        "infectious_duration_asymptomatic": 14.0,
        "fitness_R": 1.0,
        "resistance_prevalence": 0.05,
        "reporting_trend_end_multiplier": 1.0,
        "log_beta_process_2025": -0.25,
        "log_beta_process_2026": 0.4,
        "log_beta_process_2027": 0.1,
    }

    updated = _apply_sample(config, sample)
    current_values = [
        row["log_multiplier"]
        for row in updated["transmission"]["log_beta_time_variation"]["periods"]
    ]
    history_values = [
        row["log_multiplier"]
        for row in updated[simulation_common.PROSPECTIVE_POLICY_KEY]["history_config"]
        ["transmission"]["log_beta_time_variation"]["periods"]
    ]
    assert current_values == pytest.approx([-0.25, 0.4, 0.1])
    assert history_values == pytest.approx(current_values)


def test_state_route_predictive_scenarios_hold_external_structure_at_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = {
        "vaccine": {"VE_sus": 0.15, "VE_inf": 0.25, "VE_dur": 0.0},
        "transmission": {
            "relative_infectiousness_asymptomatic": 0.45,
            "fitness_R": 1.0,
        },
        "natural_history": {
            "infectious_duration_symptomatic": 21.0,
            "infectious_duration_asymptomatic": 14.0,
        },
    }
    configs = {
        "baseline": {
            "bayesian_uncertainty": {"random_seed": 7},
            "baseline_vaccine_scenario": "symptom_protective",
            "baseline_resistance_scenario": "country_timeline",
        }
    }
    monkeypatch.setattr(bayesian_uncertainty, "load_configs", lambda: configs)
    monkeypatch.setattr(
        bayesian_uncertainty,
        "make_config",
        lambda **_kwargs: deepcopy(reference),
    )
    monkeypatch.setattr(
        bayesian_uncertainty,
        "_apply_sample",
        lambda config, sample: config | {"applied_sample": deepcopy(sample)},
    )
    sample = {
        "country": "Australia",
        "sampling_method": "state_space_exact_importance_cut",
        "chain": 1,
        "posterior_log_prob": -2.0,
        "beta_S": 0.04,
        "reporting_multiplier": 1.1,
        "VE_sus": 0.9,
        "VE_inf": 0.9,
        "VE_dur": 0.9,
        "relative_infectiousness_asymptomatic": 0.9,
        "infectious_duration_symptomatic": 7.0,
        "infectious_duration_asymptomatic": 6.0,
        "fitness_R": 1.2,
        "resistance_prevalence": 0.05,
        "reporting_trend_end_multiplier": 1.0,
        "log_beta_process_2027": 0.2,
    }

    scenario = _posterior_predictive_scenarios(
        pd.DataFrame([sample]),
        1,
        random_seed=11,
    )[0]
    applied = scenario["config"]["applied_sample"]

    assert applied["beta_S"] == pytest.approx(0.04)
    assert applied["reporting_multiplier"] == pytest.approx(1.1)
    assert applied["log_beta_process_2027"] == pytest.approx(0.2)
    assert applied["VE_sus"] == pytest.approx(reference["vaccine"]["VE_sus"])
    assert applied["VE_inf"] == pytest.approx(reference["vaccine"]["VE_inf"])
    assert applied["VE_dur"] == pytest.approx(reference["vaccine"]["VE_dur"])
    assert applied["relative_infectiousness_asymptomatic"] == pytest.approx(0.45)
    assert applied["infectious_duration_symptomatic"] == pytest.approx(21.0)
    assert applied["infectious_duration_asymptomatic"] == pytest.approx(14.0)
    assert applied["fitness_R"] == pytest.approx(1.0)
    assert scenario["metadata"]["predictive_uncertainty_scope"] == (
        "conditional_state_process_reference_structure"
    )


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
                "source_code_hash": simulation_common.source_code_fingerprint(),
                "calibration_source_code_hash": simulation_common.calibration_source_code_fingerprint(),
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


def test_current_calibration_artifact_only_overlays_fitted_parameters_and_state(
    monkeypatch,
    tmp_path,
):
    configs = load_configs()
    production_start = configs["baseline"]["calendar"]["analysis_start_date"]
    artifact_path = tmp_path / "Australia_calibrated_config.yaml"
    latent_periods = [
        {
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "log_multiplier": 0.2,
        }
    ]
    write_yaml(
        {
            "config": {
                "transmission": {
                    "beta_S": 0.031,
                    "log_beta_time_variation": {
                        "interpretation": "latent_AR1_process_MAP_point_estimate",
                        "periods": latent_periods,
                    },
                },
                "reporting_multiplier": 1.2,
                "simulation": {"rtol": 0.5},
                "calendar": {"analysis_start_date": "1900-01-01"},
                "observation_model": {"case_event_definition": "wrong_event"},
            },
            "metadata": {
                "accepted": True,
                "config_hash": simulation_common.config_fingerprint(),
                "calibration_config_hash": simulation_common.calibration_config_fingerprint(),
                "source_code_hash": simulation_common.source_code_fingerprint(),
                "calibration_source_code_hash": simulation_common.calibration_source_code_fingerprint(),
            },
        },
        artifact_path,
    )
    simulation_common._load_calibrated_country_artifact_cached.cache_clear()
    monkeypatch.setattr(
        simulation_common,
        "calibrated_country_artifact_path",
        lambda country: artifact_path,
    )
    try:
        config = make_config(country_profile="Australia")
        assert config["transmission"]["beta_S"] == pytest.approx(0.031)
        assert config["reporting_multiplier"] == pytest.approx(1.2)
        assert config["transmission"]["log_beta_time_variation"]["periods"] == latent_periods
        assert config["calendar"]["analysis_start_date"] == production_start
        assert config["simulation"]["rtol"] != pytest.approx(0.5)
        assert config["observation_model"]["case_event_definition"] != "wrong_event"
        assert config["metadata"]["calibration_overlay_mode"] == (
            "fitted_parameters_and_state_only"
        )
    finally:
        simulation_common._load_calibrated_country_artifact_cached.cache_clear()


def test_calibration_artifact_from_different_source_is_never_loaded(monkeypatch, tmp_path):
    artifact_path = tmp_path / "Australia_calibrated_config.yaml"
    write_yaml(
        {
            "config": {"transmission": {"beta_S": 0.05}},
            "metadata": {
                "accepted": True,
                "config_hash": simulation_common.config_fingerprint(),
                "calibration_config_hash": simulation_common.calibration_config_fingerprint(),
                "source_code_hash": "stale-source-code",
                "calibration_source_code_hash": "stale-calibration-source-code",
            },
        },
        artifact_path,
    )
    simulation_common._load_calibrated_country_artifact_cached.cache_clear()
    monkeypatch.setattr(
        simulation_common,
        "calibrated_country_artifact_path",
        lambda country: artifact_path,
    )
    try:
        assert load_calibrated_country_artifact("Australia") is None
        assert load_calibrated_country_artifact("Australia", allow_stale=True) is None
    finally:
        simulation_common._load_calibrated_country_artifact_cached.cache_clear()


def test_convergence_diagnostics_include_sampled_mcmc_columns():
    diagnostic_columns = set(_diagnostic_sample_columns())

    assert diagnostic_columns.issubset(set(_sample_columns()))
    assert "VE_dur" in diagnostic_columns
    assert "resistance_prevalence" not in diagnostic_columns
    assert "reporting_trend_end_multiplier" not in diagnostic_columns


def test_mcmc_ve_duration_is_sampled_from_vector():
    config = make_config(country_profile="South_Africa")
    config["vaccine"]["VE_dur"] = 0.20
    priors = dict(load_configs()["baseline"]["bayesian_uncertainty"]["priors"])
    priors["resistance_prevalence_fixed"] = 0.02
    priors["reporting_trend_fixed"] = 1.0

    sample = _sample_from_vector(_initial_vector(config, enable_trend=False), priors)

    assert np.isclose(sample["VE_dur"], config["vaccine"]["VE_dur"])


def test_full_mcmc_prior_centered_start_avoids_ve_duration_boundary():
    config = make_config(country_profile="Australia")
    config["vaccine"]["VE_dur"] = 0.0
    priors = dict(load_configs()["baseline"]["bayesian_uncertainty"]["priors"])
    priors["resistance_prevalence_fixed"] = float(config["resistance"]["target_prevalence_at_analysis_start"])
    priors["reporting_trend_fixed"] = 1.0

    calibrated_sample = _sample_from_vector(
        _initial_vector(config, enable_trend=False, priors=priors),
        priors,
    )
    prior_centered_sample = _sample_from_vector(
        _initial_vector(
            config,
            enable_trend=False,
            priors=priors,
            start_at_prior_centers=True,
        ),
        priors,
    )

    assert calibrated_sample["VE_dur"] < 1e-6
    assert np.isclose(prior_centered_sample["VE_dur"], priors["VE_dur"]["mean"])
    assert np.isclose(prior_centered_sample["VE_sus"], priors["VE_sus"]["mean"])
    assert np.isclose(prior_centered_sample["relative_infectiousness_asymptomatic"], priors["relative_infectiousness_asymptomatic"]["mean"])
    assert np.isclose(prior_centered_sample["beta_S"], config["transmission"]["beta_S"])
    assert np.isclose(prior_centered_sample["reporting_multiplier"], config.get("reporting_multiplier", 1.0))


def test_joint_importance_nuisance_prior_mapping_varies_full_parameter_set():
    config = make_config(country_profile="Australia")
    priors = dict(load_configs()["baseline"]["bayesian_uncertainty"]["priors"])
    priors["resistance_prevalence_fixed"] = float(config["resistance"]["target_prevalence_at_analysis_start"])
    priors["reporting_trend_fixed"] = 1.0
    priors["parameterization"] = "beta_reporting_product"
    priors["base_log_beta_S"] = np.log(float(config["transmission"]["beta_S"]))

    start = _initial_vector(config, enable_trend=False, priors=priors)
    cube = _latin_hypercube(4, 7, seed=123)
    samples = [
        _sample_from_vector(
            _nuisance_vector_from_unit_cube(unit, start, config, priors),
            priors,
        )
        for unit in cube
    ]

    assert len({round(sample["VE_sus"], 6) for sample in samples}) > 1
    assert len({round(sample["VE_dur"], 6) for sample in samples}) > 1
    assert len({round(sample["infectious_duration_asymptomatic"], 6) for sample in samples}) > 1
    assert all(7.0 <= sample["infectious_duration_symptomatic"] <= 35.0 for sample in samples)
    assert all(0.70 <= sample["fitness_R"] <= 1.25 for sample in samples)
    assert all(np.isclose(sample["beta_S"], config["transmission"]["beta_S"]) for sample in samples)


def _registry_backed_priors(config):
    configs = load_configs()
    priors = deepcopy(configs["baseline"]["bayesian_uncertainty"]["priors"])
    priors["resistance_prevalence_fixed"] = float(
        config["resistance"]["target_prevalence_at_analysis_start"]
    )
    priors["reporting_trend_fixed"] = 1.0
    priors["parameterization"] = "beta_reporting_product"
    priors["base_log_beta_S"] = np.log(float(config["transmission"]["beta_S"]))
    priors["_distribution_specs"] = resolve_bayesian_prior_specs(
        configs["parameter_distributions"], config
    )
    return priors


def test_bayesian_registry_keeps_pre_surveillance_centres_and_evidence_labels():
    config = make_config(country_profile="Australia")
    configs = load_configs()
    specs = resolve_bayesian_prior_specs(configs["parameter_distributions"], config)

    assert tuple(specs) == BAYESIAN_PARAMETER_NAMES
    configured = configs["parameter_distributions"]["bayesian_joint"]["parameters"]
    assert inverse_cdf(0.5, specs["beta_S"]) == pytest.approx(
        configured["beta_S"]["median"]
    )
    assert inverse_cdf(0.5, specs["reporting_multiplier"]) == pytest.approx(
        configured["reporting_multiplier"]["median"]
    )
    assert not bool(configured["beta_S"].get("center_on_calibrated", False))
    assert not bool(configured["reporting_multiplier"].get("center_on_calibrated", False))
    assert specs["infectious_duration_symptomatic"]["median"] == 24.0
    assert specs["fitness_R"]["distribution"] == "truncated_lognormal"
    assert specs["VE_sus"]["evidence_class"] == "vaccine_mechanism_scenario_prior"
    assert specs["VE_dur"]["evidence_class"] == "weak_model_prior"


def test_bayesian_registry_declares_consumers_per_actual_prior_target():
    configured = load_configs()["parameter_distributions"]["bayesian_joint"][
        "parameters"
    ]

    for name in BAYESIAN_PARAMETER_NAMES:
        assert tuple(configured[name]["consumers"]) == BAYESIAN_PARAMETER_CONSUMERS[
            name
        ]
    for name in BAYESIAN_LOCAL_STATE_PARAMETER_NAMES:
        assert BAYESIAN_PARAMETER_CONSUMERS[name] == (
            "src_python.simulation.run_bayesian_uncertainty",
        )
    for name in BAYESIAN_SHARED_PARAMETER_NAMES:
        assert set(BAYESIAN_PARAMETER_CONSUMERS[name]) == {
            "src_python.simulation.run_bayesian_uncertainty",
            "src_python.simulation.run_hierarchical_joint_posterior",
            "src_python.simulation.run_hierarchical_joint_smc",
        }


def test_shared_registry_resolution_does_not_consume_local_state_prior_specs():
    config = make_config(country_profile="Australia")
    registry = deepcopy(load_configs()["parameter_distributions"])
    parameters = registry["bayesian_joint"]["parameters"]
    for name in BAYESIAN_LOCAL_STATE_PARAMETER_NAMES:
        parameters.pop(name)

    shared = resolve_bayesian_prior_specs(
        registry,
        config,
        parameter_names=BAYESIAN_SHARED_PARAMETER_NAMES,
    )

    assert tuple(shared) == BAYESIAN_SHARED_PARAMETER_NAMES
    assert not set(shared) & set(BAYESIAN_LOCAL_STATE_PARAMETER_NAMES)
    with pytest.raises(ValueError, match="exactly match"):
        resolve_bayesian_prior_specs(registry, config)


@pytest.mark.parametrize(
    ("parameter_names", "message"),
    [
        (("VE_sus", "VE_sus"), "duplicate"),
        (("VE_sus", "not_a_parameter"), "unknown"),
        ((), "at least one"),
    ],
)
def test_bayesian_registry_parameter_subset_fails_closed(parameter_names, message):
    config = make_config(country_profile="Australia")
    registry = load_configs()["parameter_distributions"]

    with pytest.raises(ValueError, match=message):
        resolve_bayesian_prior_specs(
            registry,
            config,
            parameter_names=parameter_names,
        )


def test_bayesian_registry_rejects_override_outside_selected_contract():
    config = make_config(country_profile="Australia")
    registry = load_configs()["parameter_distributions"]

    with pytest.raises(ValueError, match="outside this runner"):
        resolve_bayesian_prior_specs(
            registry,
            config,
            parameter_names=BAYESIAN_SHARED_PARAMETER_NAMES,
            beta_prior_log_sd=1.0,
        )


@pytest.mark.parametrize("bad_version", [None, True, "1", 1.0, 0, 2])
def test_bayesian_registry_schema_fails_closed(bad_version):
    config = make_config(country_profile="Australia")
    registry = deepcopy(load_configs()["parameter_distributions"])
    if bad_version is None:
        registry.pop("schema_version")
    else:
        registry["schema_version"] = bad_version

    with pytest.raises(ValueError, match="schema_version"):
        resolve_bayesian_prior_specs(registry, config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda parameters: parameters.__setitem__(
                "unimplemented_extra", deepcopy(parameters["fitness_R"])
            ),
            "exactly match",
        ),
        (
            lambda parameters: parameters["fitness_R"].__setitem__(
                "path", "transmission.beta_S"
            ),
            "implemented path",
        ),
        (
            lambda parameters: parameters["reporting_multiplier"].__setitem__(
                "time_scope", "structural_all_time"
            ),
            "implemented scope",
        ),
        (
            lambda parameters: parameters["VE_inf"].__setitem__("consumers", []),
            "implemented consumers",
        ),
    ],
)
def test_bayesian_registry_semantic_contract_fails_closed(mutation, message):
    config = make_config(country_profile="Australia")
    registry = deepcopy(load_configs()["parameter_distributions"])
    mutation(registry["bayesian_joint"]["parameters"])

    with pytest.raises(ValueError, match=message):
        resolve_bayesian_prior_specs(registry, config)


def test_full_nine_dimensional_prior_draw_and_log_density_use_same_registry_specs():
    config = make_config(country_profile="Australia")
    priors = _registry_backed_priors(config)
    settings = {"priors": priors}
    calibrated = _initial_vector(config, enable_trend=False, priors=priors)
    unit = np.array([0.17, 0.83, 0.21, 0.72, 0.33, 0.64, 0.28, 0.77, 0.41])

    vector = _full_vector_from_unit_cube(unit, calibrated, config, priors)
    sample = _sample_from_vector(vector, priors)
    actual = _log_prior(vector, config, "Australia", settings)

    expected = 0.0
    for name in BAYESIAN_PARAMETER_NAMES:
        value = sample[name]
        spec = priors["_distribution_specs"][name]
        assert spec["low"] <= value <= spec["high"]
        expected += log_pdf(value, spec)
    expected += np.log(sample["beta_S"]) + np.log(sample["reporting_multiplier"])
    for name in (
        "VE_sus",
        "VE_inf",
        "VE_dur",
        "relative_infectiousness_asymptomatic",
    ):
        value = sample[name]
        expected += np.log(value) + np.log1p(-value)
    expected += np.log(sample["infectious_duration_symptomatic"])
    expected += np.log(sample["infectious_duration_asymptomatic"])
    fitness_spec = priors["_distribution_specs"]["fitness_R"]
    scaled_fitness = (sample["fitness_R"] - fitness_spec["low"]) / (
        fitness_spec["high"] - fitness_spec["low"]
    )
    expected += (
        np.log(fitness_spec["high"] - fitness_spec["low"])
        + np.log(scaled_fitness)
        + np.log1p(-scaled_fitness)
    )

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_bayesian_registry_bounds_are_density_support_not_only_proposal_clipping():
    config = make_config(country_profile="Australia")
    priors = _registry_backed_priors(config)
    settings = {"priors": priors}
    calibrated = _initial_vector(config, enable_trend=False, priors=priors)
    vector = _full_vector_from_unit_cube(np.full(9, 0.5), calibrated, config, priors)

    outside = vector.copy()
    outside[0] = np.log(priors["_distribution_specs"]["beta_S"]["high"] * 1.01)

    assert np.isfinite(_log_prior(vector, config, "Australia", settings))
    assert _log_prior(outside, config, "Australia", settings) == -np.inf


def test_bayesian_registry_width_override_is_applied_exactly_once():
    config = make_config(country_profile="Australia")
    registry = load_configs()["parameter_distributions"]

    specs = resolve_bayesian_prior_specs(registry, config, prior_sd_scale=0.5)

    configured_beta_width = registry["bayesian_joint"]["parameters"]["beta_S"]["log_sd"]
    assert specs["beta_S"]["log_sd"] == pytest.approx(0.5 * configured_beta_width)
    configured_reporting_width = registry["bayesian_joint"]["parameters"][
        "reporting_multiplier"
    ]["log_sd"]
    assert specs["reporting_multiplier"]["log_sd"] == pytest.approx(
        0.5 * configured_reporting_width
    )
    assert specs["VE_sus"]["sd"] == pytest.approx(0.05)
    assert specs["fitness_R"]["log_sd"] == pytest.approx(0.075)


def test_canonical_bayesian_outputs_require_uncertainty_registry_metadata():
    assert "bayesian_uncertainty" in simulation_common.UNCERTAINTY_REGISTRY_METADATA_STEMS
    assert "bayesian_uncertainty_conditional_research" in simulation_common.UNCERTAINTY_REGISTRY_METADATA_STEMS
    assert "bayesian_uncertainty_joint_research" in simulation_common.UNCERTAINTY_REGISTRY_METADATA_STEMS
    assert "bayesian_uncertainty_figure2c_joint" not in simulation_common.UNCERTAINTY_REGISTRY_METADATA_STEMS
    assert (
        "bayesian_uncertainty_figure2c_joint"
        in simulation_common.RETIRED_UNCERTAINTY_REGISTRY_METADATA_STEMS
    )
    assert "bayesian_uncertainty_full_joint" not in simulation_common.UNCERTAINTY_REGISTRY_METADATA_STEMS
    assert (
        "bayesian_uncertainty_full_joint"
        in simulation_common.RETIRED_UNCERTAINTY_REGISTRY_METADATA_STEMS
    )


def test_bayesian_entrypoint_defaults_are_explicitly_nonpublication_research():
    assert bayesian_uncertainty.DEFAULT_OUTPUT_STEM == (
        "bayesian_uncertainty_conditional_research"
    )
    assert "bayesian_uncertainty_figure2c_conditional" in (
        bayesian_uncertainty.RETIRED_MISLABELED_OUTPUT_STEMS
    )
    assert "bayesian_uncertainty_figure2c_joint" in (
        bayesian_uncertainty.RETIRED_MISLABELED_OUTPUT_STEMS
    )


def test_joint_importance_diagnostics_use_weight_quality_columns():
    samples = pd.DataFrame(
        {
            "country": ["A"] * 6,
            "chain": [1, 1, 1, 2, 2, 2],
            "sampling_method": [JOINT_IMPORTANCE_SAMPLER] * 6,
            "beta_S": np.linspace(0.01, 0.02, 6),
            "VE_sus": np.linspace(0.3, 0.5, 6),
            "importance_effective_sample_size": [900.0] * 6,
            "importance_nuisance_effective_sample_size": [120.0] * 6,
            "importance_max_weight": [0.01] * 6,
            "importance_edge_weight": [0.002] * 6,
            "importance_candidate_count": [1000] * 6,
        }
    )

    diagnostics = _compute_importance_diagnostics(samples, ("beta_S", "VE_sus"))

    assert diagnostics["diagnostic_method"].eq("joint_importance_weight_quality").all()
    assert diagnostics["converged"].all()
    assert diagnostics["recommended_converged"].all()
    assert diagnostics["bulk_ess"].tolist() == [900.0, 900.0]


def test_modular_cut_diagnostics_gate_worst_conditional_grid_not_equal_design_ess():
    samples = pd.DataFrame(
        {
            "country": ["A"] * 6,
            "chain": [1, 1, 1, 2, 2, 2],
            "sampling_method": [JOINT_IMPORTANCE_SAMPLER] * 6,
            "inference_structure": ["modular_hierarchical_cut"] * 6,
            "beta_S": np.linspace(0.01, 0.02, 6),
            "importance_effective_sample_size": [2500.0] * 6,
            # Equal structural weights are deliberately not reported as a
            # nuisance posterior/convergence ESS.
            "importance_nuisance_effective_sample_size": [np.nan] * 6,
            "importance_max_weight": [0.001] * 6,
            "importance_edge_weight": [0.001] * 6,
            "importance_candidate_count": [5000] * 6,
            "modular_cut_structural_draw_count": [128.0] * 6,
            "modular_cut_conditional_incomplete_grid_count": [0.0] * 6,
            "modular_cut_conditional_fatal_failure_count": [0.0] * 6,
            "modular_cut_conditional_recommended_failure_count": [0.0] * 6,
            "modular_cut_conditional_min_effective_grid_points": [30.0] * 6,
            "modular_cut_conditional_min_beta_effective_grid_points": [6.0] * 6,
            "modular_cut_conditional_min_reporting_effective_grid_points": [8.0] * 6,
            "modular_cut_conditional_max_single_weight": [0.08] * 6,
            "modular_cut_conditional_max_edge_weight": [0.005] * 6,
        }
    )

    diagnostics = _compute_importance_diagnostics(samples, ("beta_S",))

    assert diagnostics["diagnostic_method"].eq(
        "modular_cut_conditional_grid_quality"
    ).all()
    assert diagnostics["converged"].all()
    assert diagnostics["recommended_converged"].all()
    assert diagnostics["importance_nuisance_effective_sample_size"].isna().all()
    assert diagnostics["bulk_ess"].tolist() == [30.0]

    bad = samples.copy()
    bad["modular_cut_conditional_fatal_failure_count"] = 1.0
    bad["modular_cut_conditional_recommended_failure_count"] = 1.0
    bad["modular_cut_conditional_max_edge_weight"] = 0.08
    diagnostics = _compute_importance_diagnostics(bad, ("beta_S",))

    assert not diagnostics["converged"].any()
    assert not diagnostics["recommended_converged"].any()


def test_smc_island_resampling_assigns_equal_mass_and_preserves_islands():
    rows = []
    for source_chain, logz in [(1, 0.0), (2, -4.0)]:
        for particle in range(3):
            rows.append(
                {
                    "country": "A",
                    "chain": source_chain,
                    "draw": particle + 1,
                    "step": particle + 1,
                    "sampling_method": SMC_SAMPLER,
                    "posterior_log_prob": -10.0 - source_chain,
                    "source_particle_index": particle,
                    "source_normalized_weight": 1.0 / 3.0,
                    "smc_source_chain": source_chain,
                    "smc_log_evidence": logz,
                    "smc_particle_count": 3,
                    "smc_stage_count": 2,
                    "smc_final_temperature": 1.0,
                    "smc_reached_final_temperature": True,
                    "smc_min_ess": 2.8,
                    "smc_min_ess_fraction": 2.8 / 3.0,
                    "smc_final_ess": 3.0,
                    "smc_final_ess_fraction": 1.0,
                    "smc_final_max_weight": 1.0 / 3.0,
                    "smc_max_weight": 0.40,
                    "smc_unique_ancestor_fraction": 1.0,
                    "smc_unique_particle_fraction": 1.0,
                    "smc_move_acceptance_fraction": 0.30,
                    "beta_S": 0.01 * source_chain + particle * 1e-4,
                    "reporting_multiplier": 1.0,
                    "VE_sus": 0.4,
                    "VE_inf": 0.2,
                    "VE_dur": 0.1,
                    "relative_infectiousness_asymptomatic": 0.35,
                    "infectious_duration_symptomatic": 21.0,
                    "infectious_duration_asymptomatic": 14.0,
                    "fitness_R": 1.0,
                }
            )
    resampled = _resample_smc_island_particles(
        pd.DataFrame(rows),
        chain_count=2,
        draws_per_chain=20,
        base_seed=123,
    )

    assert len(resampled) == 40
    assert resampled.groupby("chain").size().tolist() == [20, 20]
    assert resampled.groupby("chain")["smc_source_chain"].first().to_dict() == {1: 1, 2: 2}
    assert resampled["smc_source_chain"].value_counts().sort_index().tolist() == [20, 20]
    assert resampled["smc_island_weight"].unique().tolist() == [0.5]
    assert resampled["smc_island_ess"].iloc[0] == pytest.approx(2.0)
    assert resampled["smc_max_island_weight"].iloc[0] == pytest.approx(0.5)
    assert resampled["smc_log_evidence_range"].iloc[0] == pytest.approx(4.0)
    assert resampled["smc_combined_particle_ess"].iloc[0] == pytest.approx(6.0)
    assert not resampled.loc[resampled["chain"].eq(1), "source_particle_index"].is_monotonic_increasing
    assert "smc_combined_particle_ess" in resampled.columns


def test_smc_equal_island_pooling_recovers_symmetric_synthetic_target_despite_noisy_logz():
    """A noisy evidence estimate must not move a same-target posterior mean."""

    rows = []
    target_values = {1: np.linspace(-2.0, 0.0, 101), 2: np.linspace(0.0, 2.0, 101)}
    for source_chain, values in target_values.items():
        for particle, beta_s in enumerate(values):
            rows.append(
                {
                    "country": "synthetic",
                    "chain": source_chain,
                    "draw": particle + 1,
                    "source_normalized_weight": 1.0 / len(values),
                    "smc_source_chain": source_chain,
                    # Deliberately inconsistent estimates emulate the failure
                    # mode observed in the production pilot.
                    "smc_log_evidence": 50.0 if source_chain == 1 else -50.0,
                    "beta_S": beta_s,
                }
            )

    pooled = _resample_smc_island_particles(
        pd.DataFrame(rows),
        chain_count=2,
        draws_per_chain=101,
        base_seed=20260712,
    )

    assert pooled.groupby("chain").size().to_dict() == {1: 101, 2: 101}
    assert pooled["beta_S"].mean() == pytest.approx(0.0, abs=0.03)
    assert pooled["smc_log_evidence_range"].iloc[0] == pytest.approx(100.0)


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


def test_summarize_convergence_reports_recommended_thresholds():
    diagnostics = pd.DataFrame(
        {
            "country": ["A", "A", "B"],
            "parameter": ["beta_S", "VE_sus", "fitness_R"],
            "rhat_rank": [1.006, 1.020, 1.008],
            "bulk_ess": [500.0, 250.0, 430.0],
            "tail_ess": [460.0, 300.0, 420.0],
            "converged": [True, True, True],
            "recommended_converged": [True, False, True],
        }
    )

    summary = summarize_convergence(diagnostics)

    assert summary["all_converged"] is True
    assert summary["all_recommended_converged"] is False
    assert summary["n_parameters_recommended_converged"] == 2
    assert summary["countries_with_recommended_issues"] == ["A"]
    assert summary["recommended_thresholds"]["max_rhat"] == 1.01
    assert summary["recommended_thresholds"]["min_bulk_ess"] == 400.0
    assert summary["recommended_thresholds"]["min_tail_ess"] == 400.0


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
    assert _artifact_stem(
        bayesian_uncertainty.DEFAULT_OUTPUT_STEM,
        "bayesian_posterior_samples",
        "posterior_samples",
    ) == "bayesian_uncertainty_conditional_research_posterior_samples"
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
