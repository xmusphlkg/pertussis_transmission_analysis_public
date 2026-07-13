from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src_python.simulation.parameter_distributions import (
    inverse_cdf,
    latin_hypercube_draw_table,
    log_pdf,
    validate_distribution_spec,
)


@pytest.mark.parametrize(
    ("spec", "low", "high"),
    [
        ({"distribution": "uniform", "low": -2.0, "high": 3.0}, -2.0, 3.0),
        ({"distribution": "beta", "mean": 0.4, "sd": 0.1}, 0.0, 1.0),
        (
            {
                "distribution": "beta",
                "mean": 0.4,
                "sd": 0.1,
                "low": 0.2,
                "high": 0.7,
            },
            0.2,
            0.7,
        ),
        (
            {
                "distribution": "beta_pert",
                "low": 2.0,
                "mode": 4.0,
                "high": 9.0,
                "shape": 4.0,
            },
            2.0,
            9.0,
        ),
        (
            {
                "distribution": "truncated_normal",
                "mean": 5.0,
                "sd": 2.0,
                "low": 1.0,
                "high": 7.0,
            },
            1.0,
            7.0,
        ),
        (
            {
                "distribution": "truncated_lognormal",
                "median": 3.0,
                "log_sd": 0.4,
                "low": 0.0,
                "high": 12.0,
            },
            0.0,
            12.0,
        ),
        (
            {"distribution": "triangular", "low": -3.0, "mode": 1.0, "high": 2.0},
            -3.0,
            2.0,
        ),
        ({"distribution": "loguniform", "low": 0.1, "high": 100.0}, 0.1, 100.0),
    ],
)
def test_inverse_cdf_is_vectorized_finite_and_respects_bounds(spec, low, high):
    unit = np.linspace(0.0, 1.0, 1001).reshape(7, 11, 13)
    values = inverse_cdf(unit, spec)

    assert isinstance(values, np.ndarray)
    assert values.shape == unit.shape
    assert np.isfinite(values).all()
    assert values.min() >= low
    assert values.max() <= high
    assert values.flat[0] == pytest.approx(low)
    assert values.flat[-1] == pytest.approx(high)


def test_scalar_inverse_cdf_returns_float():
    value = inverse_cdf(0.5, {"distribution": "loguniform", "low": 1.0, "high": 100.0})

    assert isinstance(value, float)
    assert value == pytest.approx(10.0)


@pytest.mark.parametrize(
    "spec",
    [
        {"distribution": "beta", "mean": 0.35, "sd": 0.12, "low": 0.10, "high": 0.80},
        {
            "distribution": "truncated_normal",
            "mean": 1.0,
            "sd": 0.3,
            "low": 0.7,
            "high": 1.25,
        },
        {
            "distribution": "truncated_lognormal",
            "median": 17.3,
            "log_sd": 0.30,
            "low": 7.0,
            "high": 28.0,
        },
        {
            "distribution": "complement_truncated_lognormal",
            "median": 0.823,
            "log_sd": 0.633,
            "low": 0.05,
            "high": 0.99,
        },
    ],
)
def test_log_pdf_is_normalized_over_truncated_support(spec):
    normalized = validate_distribution_spec(spec)
    low, high = normalized["low"], normalized["high"]
    grid = np.linspace(low, high, 100_001)
    density = np.exp(log_pdf(grid, spec))

    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=2e-4)
    assert log_pdf(low - max(abs(low), 1.0), spec) == -np.inf
    assert log_pdf(high + max(abs(high), 1.0), spec) == -np.inf


def test_log_pdf_matches_inverse_cdf_local_probability_mass():
    spec = {
        "distribution": "truncated_lognormal",
        "median": 24.0,
        "log_sd": 0.103,
        "low": 14.0,
        "high": 35.0,
    }
    probability = 0.37
    epsilon = 1e-5
    lower = inverse_cdf(probability - epsilon, spec)
    upper = inverse_cdf(probability + epsilon, spec)
    midpoint = inverse_cdf(probability, spec)
    numerical_density = (2.0 * epsilon) / (upper - lower)

    assert np.exp(log_pdf(midpoint, spec)) == pytest.approx(numerical_density, rel=2e-5)


def test_legacy_min_max_spec_is_normalized_to_uniform_without_mutation():
    legacy = {"min": 2, "max": 6, "path": "transmission.beta_S"}
    normalized = validate_distribution_spec(legacy)

    assert legacy == {"min": 2, "max": 6, "path": "transmission.beta_S"}
    assert normalized["distribution"] == "uniform"
    assert normalized["low"] == 2.0
    assert normalized["high"] == 6.0
    assert inverse_cdf(np.array([0.0, 0.25, 1.0]), legacy).tolist() == [2.0, 3.0, 6.0]


def test_stratified_beta_quantiles_recover_underlying_mean_and_sd():
    n = 20_000
    unit_midpoints = (np.arange(n, dtype=float) + 0.5) / n
    values = inverse_cdf(
        unit_midpoints,
        {"distribution": "beta", "mean": 0.35, "sd": 0.10},
    )

    assert np.mean(values) == pytest.approx(0.35, abs=2e-4)
    assert np.std(values) == pytest.approx(0.10, abs=4e-4)


def test_beta_pert_and_triangular_stratified_moments_match_theory():
    n = 20_000
    unit_midpoints = (np.arange(n, dtype=float) + 0.5) / n
    pert_spec = {
        "distribution": "beta_pert",
        "low": 1.0,
        "mode": 4.0,
        "high": 10.0,
        "shape": 4.0,
    }
    triangular_spec = {
        "distribution": "triangular",
        "low": 1.0,
        "mode": 4.0,
        "high": 10.0,
    }

    pert_values = inverse_cdf(unit_midpoints, pert_spec)
    triangular_values = inverse_cdf(unit_midpoints, triangular_spec)
    expected_pert_mean = (1.0 + 4.0 * 4.0 + 10.0) / (4.0 + 2.0)

    assert np.mean(pert_values) == pytest.approx(expected_pert_mean, abs=3e-4)
    assert np.mean(triangular_values) == pytest.approx((1.0 + 4.0 + 10.0) / 3.0, abs=3e-4)


def test_symmetric_log_truncation_preserves_lognormal_median():
    spec = {
        "distribution": "truncated_lognormal",
        "median": 2.0,
        "log_sd": 0.5,
        "low": 0.2,
        "high": 20.0,
    }

    assert inverse_cdf(0.5, spec) == pytest.approx(2.0)


def test_asymmetric_log_truncation_preserves_requested_truncated_median():
    spec = {
        "distribution": "truncated_lognormal",
        "median": 1.0,
        "log_sd": 0.35,
        "low": 0.5,
        "high": 1.5,
    }

    assert inverse_cdf(0.5, spec) == pytest.approx(1.0, abs=1e-12)


def test_complement_lognormal_models_asymmetric_effectiveness_interval():
    spec = {
        "distribution": "complement_truncated_lognormal",
        "median": 0.823,
        "log_sd": 0.633,
        "low": 0.05,
        "high": 0.99,
    }
    quantiles = inverse_cdf(np.array([0.025, 0.5, 0.975]), spec)

    assert quantiles[0] == pytest.approx(0.391, abs=0.025)
    assert quantiles[1] == pytest.approx(0.823, abs=1e-12)
    assert quantiles[2] == pytest.approx(0.949, abs=0.01)


def test_latin_hypercube_preserves_names_reproducibility_and_marginal_strata():
    specs = [
        ("zeta", {"min": 2.0, "max": 6.0}),
        (
            "alpha",
            {"distribution": "triangular", "low": 0.0, "mode": 0.3, "high": 1.0},
        ),
    ]
    first = latin_hypercube_draw_table(specs, 128, seed=2718)
    second = latin_hypercube_draw_table(specs, 128, seed=2718)
    different = latin_hypercube_draw_table(specs, 128, seed=2719)

    assert first.columns.tolist() == ["zeta", "alpha"]
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(different)

    sorted_unit = np.sort((first["zeta"].to_numpy() - 2.0) / 4.0)
    occupied_strata = np.floor(sorted_unit * len(first)).astype(int)
    np.testing.assert_array_equal(occupied_strata, np.arange(len(first)))


def test_spike_and_slab_has_requested_stratified_spike_proportion():
    specs = {
        "implementation_failure": {
            "distribution": "spike_and_slab",
            "spike_probability": 0.2,
            "spike": {"value": 0.0},
            "slab": {
                "distribution": "beta_pert",
                "low": 0.1,
                "mode": 0.6,
                "high": 0.9,
                "shape": 4.0,
            },
        }
    }
    draws = latin_hypercube_draw_table(specs, 1000, seed=42)
    values = draws["implementation_failure"].to_numpy()

    assert np.count_nonzero(values == 0.0) == 200
    assert np.all((values[values != 0.0] >= 0.1) & (values[values != 0.0] <= 0.9))


def test_spike_value_alias_normalizes_idempotently():
    spec = {
        "distribution": "spike_and_slab",
        "spike_probability": 0.25,
        "spike_value": 0.0,
        "slab": {"min": 1.0, "max": 2.0},
    }

    normalized = validate_distribution_spec(spec)
    renormalized = validate_distribution_spec(normalized)

    assert "spike_value" not in normalized
    assert renormalized == normalized
    assert inverse_cdf(np.array([0.1, 0.5]), normalized).tolist() == [0.0, 4.0 / 3.0]


def test_right_hand_spike_and_slab_inverse_cdf_is_monotone():
    spec = {
        "distribution": "spike_and_slab",
        "spike_probability": 0.2,
        "spike": 10.0,
        "slab": {"min": 0.0, "max": 1.0},
    }
    values = inverse_cdf(np.linspace(0.0, 1.0, 101), spec)

    assert np.all(np.diff(values) >= 0.0)
    assert np.count_nonzero(values == 10.0) == 21


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({}, "missing 'distribution'"),
        ({"distribution": None, "min": 0, "max": 1}, "non-empty string"),
        ({"distribution": "mystery", "low": 0, "high": 1}, "unsupported distribution"),
        ({"min": 0}, "missing 'max'"),
        ({"min": 1, "max": 1}, "lower bound must be less"),
        (
            {"distribution": "uniform", "low": 0, "high": 1, "min": 0, "max": 2},
            "specify different bounds",
        ),
        ({"distribution": "beta", "mean": 0.0, "sd": 0.1}, "mean.*strictly"),
        ({"distribution": "beta", "mean": 0.5, "sd": 1e-300}, "sd.*too small"),
        ({"distribution": "beta", "mean": 0.5, "sd": 0.5}, "sd.*too large"),
        (
            {"distribution": "beta", "mean": 0.5, "sd": 0.1, "low": -0.1, "high": 0.8},
            "within \\[0, 1\\]",
        ),
        (
            {"distribution": "beta_pert", "low": 0, "mode": 2, "high": 1},
            "mode.*within",
        ),
        (
            {"distribution": "truncated_normal", "mean": 0, "sd": 0, "low": -1, "high": 1},
            "sd.*greater than zero",
        ),
        (
            {
                "distribution": "truncated_lognormal",
                "median": -1,
                "log_sd": 0.2,
                "low": 0,
                "high": 2,
            },
            "median.*greater than zero",
        ),
        ({"distribution": "loguniform", "low": 0, "high": 2}, "strictly positive"),
        (
            {
                "distribution": "spike_and_slab",
                "spike_probability": 1.2,
                "spike": 0,
                "slab": {"min": 1, "max": 2},
            },
            "spike_probability.*within",
        ),
        (
            {"distribution": "spike_and_slab", "spike_probability": 0.2, "spike": 0},
            "missing required nested component 'slab'",
        ),
        (
            {
                "distribution": "spike_and_slab",
                "spike_probability": 0.2,
                "spike": 0.5,
                "slab": {"min": 0.0, "max": 1.0},
            },
            "component supports overlap",
        ),
    ],
)
def test_invalid_distribution_specs_raise_clear_value_errors(spec, message):
    with pytest.raises(ValueError, match=message):
        validate_distribution_spec(spec, context="parameter 'test_parameter'")


@pytest.mark.parametrize("unit", [[-0.01], [1.01], [np.nan], [np.inf]])
def test_invalid_unit_quantiles_raise_value_error(unit):
    with pytest.raises(ValueError, match="unit quantiles"):
        inverse_cdf(unit, {"min": 0.0, "max": 1.0})


@pytest.mark.parametrize("n_draws", [0, -1, 1.5, True])
def test_latin_hypercube_rejects_invalid_draw_counts(n_draws):
    with pytest.raises(ValueError, match="positive integer"):
        latin_hypercube_draw_table({"x": {"min": 0, "max": 1}}, n_draws, seed=1)
