from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src_python.simulation.stochastic_overlay import (
    COUNT_OUTCOME_DENOMINATORS,
    StochasticOverlayConfig,
    decompose_variance,
    design_effect,
    effective_dispersion,
    sample_negative_binomial,
    stochastic_overlay_samples,
    summarize_overlay_intervals,
)


def test_design_effect_matches_ball_formula():
    # Ball, Mollison & Scalia-Tomba (1997): DEFF = 1 + (m_bar - 1) * SAR
    assert design_effect(1.0, 0.8) == pytest.approx(1.0)  # singletons -> no clustering
    assert design_effect(3.5, 0.0) == pytest.approx(1.0)  # SAR 0 -> no clustering
    assert design_effect(3.5, 0.8) == pytest.approx(1.0 + 2.5 * 0.8)


def test_effective_dispersion_combines_superspreading_and_clustering():
    k_ss = 10.0
    deff = 3.0
    # At small mu, clustering penalty dominates -> tighter (smaller) k_eff than k_ss.
    k_small = effective_dispersion(5.0, superspreading_k=k_ss, design_effect_value=deff)
    # At large mu, clustering contribution vanishes -> k_eff -> k_ss.
    k_large = effective_dispersion(1_000_000.0, superspreading_k=k_ss, design_effect_value=deff)
    assert k_small < k_ss
    assert k_large == pytest.approx(k_ss, rel=1e-3)
    # DEFF = 1 means no household penalty -> k_eff == k_ss regardless of mu.
    assert effective_dispersion(100.0, superspreading_k=k_ss, design_effect_value=1.0) == pytest.approx(k_ss)


def test_effective_dispersion_scales_superspreading_over_analysis_years():
    k_one_year = effective_dispersion(
        1_000_000.0,
        superspreading_k=10.0,
        design_effect_value=1.0,
        aggregation_units=1.0,
    )
    k_multi_year = effective_dispersion(
        1_000_000.0,
        superspreading_k=10.0,
        design_effect_value=1.0,
        aggregation_units=26.0,
    )

    assert k_one_year == pytest.approx(10.0)
    assert k_multi_year == pytest.approx(260.0)


def test_negative_binomial_has_correct_mean_and_overdispersion():
    rng = np.random.default_rng(0)
    mu = 400.0
    k = 10.0
    samples = sample_negative_binomial(mu, dispersion_k=k, size=200_000, rng=rng)
    # Sampling mean should be close to mu.
    assert np.isclose(samples.mean(), mu, rtol=0.02)
    expected_var = mu + mu**2 / k
    # Variance within a few percent of analytical NB2 variance.
    assert np.isclose(samples.var(), expected_var, rtol=0.08)


def test_overlay_rate_outcomes_round_trip_counts_and_rates():
    summary = pd.DataFrame(
        [
            {
                "country": "TestLand",
                "posterior_draw": 1,
                "total_population": 10_000_000.0,
                "infant_population": 200_000.0,
                "child_adolescent_population": 2_500_000.0,
                "analysis_years": 30.0,
                "annualized_reported_cases_per_100k": 20.0,
                "annualized_infant_cases_per_100k": 150.0,
                "annualized_child_adolescent_cases_per_100k": 90.0,
                "total_reported_cases": 20.0 * 10_000_000.0 * 30.0 / 100_000.0,
                "total_child_adolescent_cases": 90.0 * 2_500_000.0 * 30.0 / 100_000.0,
                "resistant_infections": 1_500.0,
            }
        ]
    )
    overlay = StochasticOverlayConfig(replicates_per_draw=500, random_seed=7)
    samples = stochastic_overlay_samples(summary, overlay=overlay)
    assert not samples.empty
    assert samples["aggregation_units"].min() == pytest.approx(30.0)
    assert samples["superspreading_k_effective"].min() == pytest.approx(
        overlay.superspreading_k * 30.0
    )
    assert set(samples["outcome"].unique()) >= {
        "annualized_reported_cases_per_100k",
        "annualized_infant_cases_per_100k",
        "annualized_child_adolescent_cases_per_100k",
        "total_reported_cases",
        "total_child_adolescent_cases",
        "resistant_infections",
    }
    # Sampled replicate mean should be close to the ODE expectation for each outcome.
    for outcome, expected in [
        ("annualized_reported_cases_per_100k", 20.0),
        ("annualized_infant_cases_per_100k", 150.0),
        ("annualized_child_adolescent_cases_per_100k", 90.0),
        ("resistant_infections", 1_500.0),
    ]:
        draws = samples.loc[samples["outcome"].eq(outcome), "value"].to_numpy(dtype=float)
        assert np.isclose(draws.mean(), expected, rtol=0.15)


def test_combined_interval_is_at_least_as_wide_as_parameter_interval():
    rng = np.random.default_rng(42)
    n_draws = 40
    population = 5_000_000.0
    years = 30.0
    rate_draws = rng.normal(25.0, 2.5, size=n_draws)
    summary = pd.DataFrame(
        {
            "country": "TestLand",
            "posterior_draw": np.arange(1, n_draws + 1),
            "total_population": population,
            "infant_population": 100_000.0,
            "analysis_years": years,
            "annualized_reported_cases_per_100k": rate_draws,
        }
    )
    overlay = StochasticOverlayConfig(
        replicates_per_draw=300,
        random_seed=3,
        superspreading_k=10.0,
        household_mean_size=3.5,
        household_secondary_attack_rate=0.8,
    )
    samples = stochastic_overlay_samples(summary, overlay=overlay)
    intervals = summarize_overlay_intervals(summary, samples, overlay=overlay)
    row = intervals.loc[intervals["outcome"].eq("annualized_reported_cases_per_100k")].iloc[0]
    param_width = row["parameter_credible_interval_high"] - row["parameter_credible_interval_low"]
    combined_width = row["combined_credible_interval_high"] - row["combined_credible_interval_low"]
    assert combined_width >= param_width
    assert bool(row["stochastic_overlay_applied"])


def test_variance_decomposition_allocates_to_both_components():
    rng = np.random.default_rng(2)
    n_draws = 30
    summary = pd.DataFrame(
        {
            "country": "Grid",
            "posterior_draw": np.arange(1, n_draws + 1),
            "total_population": 1_000_000.0,
            "infant_population": 20_000.0,
            "analysis_years": 30.0,
            "annualized_reported_cases_per_100k": rng.normal(30.0, 4.0, size=n_draws),
        }
    )
    overlay = StochasticOverlayConfig(replicates_per_draw=250, random_seed=11)
    samples = stochastic_overlay_samples(summary, overlay=overlay)
    components = decompose_variance(samples)
    row = components.loc[components["outcome"].eq("annualized_reported_cases_per_100k")].iloc[0]
    assert row["parameter_variance"] > 0.0
    assert row["stochastic_variance"] > 0.0
    assert 0.0 < row["parameter_variance_share"] < 1.0


def test_count_outcome_denominator_map_is_nonempty():
    # Guards against accidental constant removal.
    assert "annualized_reported_cases_per_100k" in COUNT_OUTCOME_DENOMINATORS
    assert "annualized_child_adolescent_cases_per_100k" in COUNT_OUTCOME_DENOMINATORS
