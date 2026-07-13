from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import nbinom

from src_python.calibration.panel_pomp import (
    CountrySeries,
    DiscrepancyProcessSpec,
    ObservationSpec,
    PanelCandidate,
    ParticleFilterConfig,
    ScaleSearchConfig,
    bootstrap_particle_filter,
    _propagate_states,
    fit_country_static_log_scale,
    nb2_logpmf,
    panel_log_likelihood,
    process_transition_moments,
    robust_component_scales,
    select_panel_candidate,
    systematic_resample,
)


def _series(
    country: str = "Example",
    *,
    counts: tuple[float, ...] = (100.0, 120.0, 90.0, 110.0),
    offsets: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0),
    training: tuple[bool, ...] = (True, True, True, False),
) -> CountrySeries:
    n = len(counts)
    starts = np.asarray([0.0, 100.0, 230.0, 365.0][:n])
    ends = np.asarray([90.0, 220.0, 320.0, 455.0][:n])
    return CountrySeries(
        country=country,
        interval_ids=tuple(f"q{i + 1}" for i in range(n)),
        interval_start=starts,
        interval_end=ends,
        observed_counts=counts,
        mechanistic_offset=offsets,
        exposure_days=ends - starts,
        training_mask=training,
    )


def _deterministic_process() -> DiscrepancyProcessSpec:
    return DiscrepancyProcessSpec(rho=0.6, innovation_sd=0.0)


def _filter_config(seed: int = 42, predictive_draws: int = 128) -> ParticleFilterConfig:
    return ParticleFilterConfig(
        n_particles=96,
        ess_resample_fraction=0.5,
        seed=seed,
        predictive_draws=predictive_draws,
        robust_initialization_steps=8,
    )


def test_country_series_validates_calendar_intervals_counts_and_training_prefix() -> None:
    calendar = CountrySeries(
        country="Calendar",
        interval_ids=("2020", "2021"),
        interval_start=("2020-01-01", "2021-01-01"),
        interval_end=("2021-01-01", "2022-01-01"),
        observed_counts=(10, np.nan),
        mechanistic_offset=(9.0, 11.0),
        exposure_days=(366.0, 365.0),
        training_mask=(True, False),
    )
    assert calendar.interval_end[0] - calendar.interval_start[0] == pytest.approx(366.0)
    assert calendar.midpoint_days[1] > calendar.midpoint_days[0]

    with pytest.raises(ValueError, match="non-negative integers"):
        _series(counts=(1.5, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="prefix"):
        _series(training=(True, False, True, False))
    with pytest.raises(ValueError, match="exceed"):
        CountrySeries(
            country="Bad",
            interval_ids=("a",),
            interval_start=(0.0,),
            interval_end=(5.0,),
            observed_counts=(1,),
            mechanistic_offset=(1.0,),
            exposure_days=(6.0,),
            training_mask=(True,),
        )


def test_nb2_logpmf_matches_scipy_parameterization() -> None:
    observed = np.asarray([0.0, 1.0, 10.0, 1000.0])
    mean = np.asarray([0.1, 3.0, 12.0, 850.0])
    size = np.asarray([0.7, 2.5, 40.0, 125.0])
    probability = size / (size + mean)

    actual = nb2_logpmf(observed, mean, size)
    expected = nbinom.logpmf(observed, size, probability)

    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


def test_observation_size_scales_linearly_with_exposure() -> None:
    observation = ObservationSpec(
        dispersion_per_reference_exposure=80.0,
        reference_exposure_days=400.0,
    )
    assert observation.interval_size(np.asarray([100.0, 200.0, 400.0])) == pytest.approx(
        [20.0, 40.0, 80.0]
    )


def test_irregular_ar1_transition_matches_scalar_analytic_solution() -> None:
    process = DiscrepancyProcessSpec(
        rho=0.6,
        innovation_sd=0.4,
        reference_time_days=365.2425,
    )
    delta_days = 137.0
    transition, innovation, stationary = process_transition_moments(process, delta_days)
    elapsed = delta_days / process.reference_time_days
    expected_phi = process.rho**elapsed
    expected_stationary = process.innovation_sd**2 / (1.0 - process.rho**2)

    assert transition == pytest.approx(np.asarray([[expected_phi]]), rel=2e-12)
    assert stationary == pytest.approx(np.asarray([[expected_stationary]]), rel=2e-12)
    assert innovation == pytest.approx(
        np.asarray([[expected_stationary * (1.0 - expected_phi**2)]]),
        rel=2e-12,
    )


def test_damped_trend_transition_is_stationary_and_positive_semidefinite() -> None:
    process = DiscrepancyProcessSpec(
        rho=0.7,
        innovation_sd=0.3,
        trend_enabled=True,
        trend_damping=0.45,
        trend_innovation_sd=0.08,
        trend_coupling=0.6,
    )
    transition, innovation, stationary = process_transition_moments(process, 73.0)

    assert transition.shape == innovation.shape == stationary.shape == (2, 2)
    assert np.linalg.eigvalsh(innovation).min() >= -1e-12
    assert np.linalg.eigvalsh(stationary).min() >= -1e-12
    assert stationary == pytest.approx(
        transition @ stationary @ transition.T + innovation,
        rel=2e-10,
        abs=2e-12,
    )

    # Independent SciPy Lyapunov parity for the declared continuous process.
    kappa_x = -np.log(process.rho)
    kappa_b = -np.log(process.trend_damping)
    drift = np.asarray([[-kappa_x, process.trend_coupling], [0.0, -kappa_b]])
    q_x = process.innovation_sd**2 * 2.0 * kappa_x / (1.0 - process.rho**2)
    q_b = (
        process.trend_innovation_sd**2
        * 2.0
        * kappa_b
        / (1.0 - process.trend_damping**2)
    )
    expected_stationary = solve_continuous_lyapunov(drift, -np.diag([q_x, q_b]))
    assert stationary == pytest.approx(expected_stationary, rel=2e-10, abs=2e-12)


def test_robust_process_mixture_preserves_nominal_second_moment() -> None:
    probability = 0.08
    normal_scale, shock_scale = robust_component_scales(probability, 5.0)
    mixture_variance = (
        (1.0 - probability) * normal_scale**2
        + probability * shock_scale**2
    )
    assert mixture_variance == pytest.approx(1.0)
    assert shock_scale > normal_scale


def test_compound_poisson_jump_rate_scales_with_elapsed_time() -> None:
    process = DiscrepancyProcessSpec(
        rho=0.7,
        innovation_sd=0.0,
        jump_rate_per_year=1.0,
        jump_sd=1.0,
    )
    states = np.zeros((20_000, 1), dtype=float)
    propagated, shock_fraction = _propagate_states(
        states,
        process,
        process.reference_time_days / 2.0,
        np.random.default_rng(44),
    )

    expected = 1.0 - np.exp(-0.5)
    assert shock_fraction == pytest.approx(expected, abs=0.015)
    assert np.mean(propagated[:, 0] != 0.0) == pytest.approx(expected, abs=0.015)


def test_systematic_resampling_is_deterministic_and_respects_zero_mass() -> None:
    weights = np.asarray([0.0, 0.2, 0.8])
    first = systematic_resample(weights, np.random.default_rng(123))
    second = systematic_resample(weights, np.random.default_rng(123))

    assert np.array_equal(first, second)
    assert len(first) == len(weights)
    assert not np.any(first == 0)
    assert np.all((first >= 1) & (first <= 2))


def test_deterministic_particle_filter_matches_direct_scipy_nb_likelihood() -> None:
    series = _series(training=(True, True, True, True))
    observation = ObservationSpec(
        dispersion_per_reference_exposure=60.0,
        reference_exposure_days=90.0,
    )
    alpha = np.log(1.1)
    result = bootstrap_particle_filter(
        series,
        static_log_scale=alpha,
        process=_deterministic_process(),
        observation=observation,
        config=_filter_config(predictive_draws=0),
    )
    mean = series.mechanistic_offset * np.exp(alpha)
    size = observation.interval_size(series.exposure_days)
    expected = float(np.sum(nbinom.logpmf(series.observed_counts, size, size / (size + mean))))

    assert result.training_log_likelihood == pytest.approx(expected, rel=2e-12)
    assert all(
        diagnostic.effective_sample_size == pytest.approx(96.0)
        for diagnostic in result.diagnostics
    )
    assert not any(diagnostic.resampled for diagnostic in result.diagnostics)


def test_predictive_draw_count_cannot_change_filter_likelihood_or_particles() -> None:
    process = DiscrepancyProcessSpec(
        rho=0.55,
        innovation_sd=0.35,
        robust_mixture_probability=0.05,
        robust_mixture_scale=4.0,
    )
    common = dict(
        series=_series(),
        static_log_scale=0.1,
        process=process,
        observation=ObservationSpec(45.0, reference_exposure_days=90.0),
    )
    no_draws = bootstrap_particle_filter(
        **common,
        config=_filter_config(seed=90210, predictive_draws=0),
    )
    many_draws = bootstrap_particle_filter(
        **common,
        config=_filter_config(seed=90210, predictive_draws=300),
    )

    assert no_draws.training_log_likelihood == many_draws.training_log_likelihood
    assert np.array_equal(no_draws.terminal_states, many_draws.terminal_states)
    assert np.array_equal(no_draws.terminal_weights, many_draws.terminal_weights)
    assert np.isnan(no_draws.forecasts[-1].observation_median)
    assert np.isfinite(many_draws.forecasts[-1].observation_median)


def test_nontraining_counts_never_change_scale_fit_or_terminal_cloud() -> None:
    low_holdout = _series(counts=(100.0, 100.0, 100.0, 1.0))
    high_holdout = _series(counts=(100.0, 100.0, 100.0, 10_000.0))
    process = _deterministic_process()
    observation = ObservationSpec(80.0, reference_exposure_days=90.0)
    config = _filter_config(seed=17, predictive_draws=0)
    search = ScaleSearchConfig(lower=-0.5, upper=0.5, grid_points=7, refinement_steps=2)

    low_fit = fit_country_static_log_scale(
        low_holdout,
        process=process,
        observation=observation,
        filter_config=config,
        search=search,
    )
    high_fit = fit_country_static_log_scale(
        high_holdout,
        process=process,
        observation=observation,
        filter_config=config,
        search=search,
    )

    assert low_fit.static_log_scale == high_fit.static_log_scale
    assert low_fit.training_log_likelihood == high_fit.training_log_likelihood
    assert np.array_equal(
        low_fit.filter_result.terminal_states,
        high_fit.filter_result.terminal_states,
    )
    assert low_fit.filter_result.forecasts[-1].predictive_log_score != (
        high_fit.filter_result.forecasts[-1].predictive_log_score
    )


def test_forecasts_are_ordered_and_holdouts_are_not_assimilated() -> None:
    result = bootstrap_particle_filter(
        _series(),
        static_log_scale=0.0,
        process=DiscrepancyProcessSpec(
            rho=0.65,
            innovation_sd=0.25,
            trend_enabled=True,
            trend_damping=0.4,
            trend_innovation_sd=0.04,
        ),
        observation=ObservationSpec(50.0, reference_exposure_days=90.0),
        config=_filter_config(seed=81, predictive_draws=400),
    )
    heldout = result.forecasts[-1]

    assert not heldout.training_interval
    assert heldout.conditional_mean_q025 <= heldout.conditional_mean_median
    assert heldout.conditional_mean_median <= heldout.conditional_mean_q975
    assert heldout.observation_q025 <= heldout.observation_median <= heldout.observation_q975
    assert np.isnan(result.diagnostics[-1].log_likelihood_increment)


def test_prequential_mode_updates_only_after_scoring_each_holdout() -> None:
    low_first_holdout = _series(
        counts=(100.0, 100.0, 10.0, 100.0),
        training=(True, True, False, False),
    )
    high_first_holdout = _series(
        counts=(100.0, 100.0, 500.0, 100.0),
        training=(True, True, False, False),
    )
    common = dict(
        static_log_scale=0.0,
        process=DiscrepancyProcessSpec(rho=0.75, innovation_sd=0.6),
        observation=ObservationSpec(25.0, reference_exposure_days=90.0),
        config=ParticleFilterConfig(
            n_particles=1024,
            seed=731,
            predictive_draws=256,
            assimilate_holdout_observations=True,
        ),
    )

    low = bootstrap_particle_filter(low_first_holdout, **common)
    high = bootstrap_particle_filter(high_first_holdout, **common)
    low_holdouts = [forecast for forecast in low.forecasts if not forecast.training_interval]
    high_holdouts = [forecast for forecast in high.forecasts if not forecast.training_interval]

    assert low.training_log_likelihood == high.training_log_likelihood
    assert low_holdouts[0].expected_mean == high_holdouts[0].expected_mean
    assert low_holdouts[1].expected_mean != high_holdouts[1].expected_mean
    assert np.isfinite(low.diagnostics[-2].log_likelihood_increment)
    assert np.isfinite(high.diagnostics[-2].log_likelihood_increment)


def test_panel_likelihood_equals_country_sum_and_is_order_invariant() -> None:
    a = _series("A")
    b = _series("B", counts=(80.0, 85.0, 90.0, 95.0), offsets=(85.0,) * 4)
    kwargs = dict(
        country_log_scales={"A": 0.05, "B": -0.05},
        process=DiscrepancyProcessSpec(rho=0.6, innovation_sd=0.15),
        observation=ObservationSpec(55.0, reference_exposure_days=90.0),
        filter_config=_filter_config(seed=55, predictive_draws=0),
    )
    forward = panel_log_likelihood([a, b], **kwargs)
    reverse = panel_log_likelihood([b, a], **kwargs)

    assert forward.total_training_log_likelihood == reverse.total_training_log_likelihood
    assert forward.total_training_log_likelihood == pytest.approx(
        sum(item.training_log_likelihood for item in forward.country_results)
    )
    assert [item.country for item in forward.country_results] == ["A", "B"]
    with pytest.raises(ValueError, match="match panel countries exactly"):
        panel_log_likelihood([a, b], **{**kwargs, "country_log_scales": {"A": 0.0}})


def test_panel_candidate_selection_uses_unassimilated_validation_log_score() -> None:
    a = _series("A", counts=(100.0, 100.0, 100.0, 100.0))
    b = _series("B", counts=(80.0, 80.0, 80.0, 80.0), offsets=(80.0,) * 4)
    process = _deterministic_process()
    search = ScaleSearchConfig(lower=-0.3, upper=0.3, grid_points=5, refinement_steps=1)
    sharp = PanelCandidate(
        name="sharp",
        process=process,
        observation=ObservationSpec(2_000.0, reference_exposure_days=90.0),
        scale_search=search,
    )
    diffuse = PanelCandidate(
        name="diffuse",
        process=process,
        observation=ObservationSpec(0.2, reference_exposure_days=90.0),
        scale_search=search,
    )
    selection = select_panel_candidate(
        [b, a],
        [diffuse, sharp],
        filter_config=replace(_filter_config(seed=8), n_particles=48),
        selection_criterion="validation_log_score",
    )

    assert selection.selected_name == "sharp"
    assert selection.selected.validation_interval_count == 2
    assert selection.scope == "candidate_selection_validation_not_final_test"
    assert all(fit.filter_result.forecasts[-1].training_interval is False for fit in selection.selected.country_fits)

    balanced = select_panel_candidate(
        [b, a],
        [diffuse, sharp],
        filter_config=replace(_filter_config(seed=8), n_particles=48),
        selection_criterion="country_balanced_validation_log_score",
    )
    assert balanced.selected_name == "sharp"
    assert np.isfinite(balanced.selected.country_balanced_validation_log_score)


def test_configuration_validation_rejects_scientifically_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        DiscrepancyProcessSpec(rho=1.0, innovation_sd=0.2)
    with pytest.raises(ValueError, match="trend_innovation_sd must be zero"):
        DiscrepancyProcessSpec(
            rho=0.5,
            innovation_sd=0.2,
            trend_enabled=False,
            trend_innovation_sd=0.1,
        )
    with pytest.raises(ValueError, match="positive jump rate"):
        DiscrepancyProcessSpec(
            rho=0.5,
            innovation_sd=0.2,
            jump_rate_per_year=0.5,
        )
    with pytest.raises(ValueError, match="positive robust probability"):
        robust_component_scales(0.1, 1.0)
    with pytest.raises(ValueError, match="at least two"):
        ParticleFilterConfig(n_particles=1)
    with pytest.raises(ValueError, match="particle_grid or count_ratio"):
        ScaleSearchConfig(method="unknown")


def test_count_ratio_scale_is_training_only_and_avoids_particle_grid_search() -> None:
    series = _series(
        "ratio",
        counts=(100.0, 100.0, 100.0, 9999.0),
        offsets=(50.0, 50.0, 50.0, 50.0),
    )
    fit = fit_country_static_log_scale(
        series,
        process=_deterministic_process(),
        observation=ObservationSpec(20.0, reference_exposure_days=90.0),
        filter_config=replace(_filter_config(seed=12), n_particles=16),
        search=ScaleSearchConfig(method="count_ratio", lower=-5.0, upper=5.0),
    )

    assert fit.static_log_scale == pytest.approx(np.log(300.5 / 150.0))
    assert fit.evaluations == 1
