from __future__ import annotations

import numpy as np
import pandas as pd

from src_python.simulation import run_fitness_grid as fitness_grid
from src_python.simulation.run_bayesian_uncertainty import _apply_sample, _sample_columns


def _posterior_samples(countries: tuple[str, ...] = ("A", "B"), draws: int = 5) -> pd.DataFrame:
    rows = []
    for country in countries:
        for draw in range(1, draws + 1):
            row = {
                "country": country,
                "chain": 1,
                "draw": draw,
                "posterior_log_prob": -float(draw),
                "beta_S": 0.02 + 0.001 * draw,
                "reporting_multiplier": 1.0,
                "VE_sus": 0.15,
                "VE_inf": 0.25,
                "VE_dur": 0.0,
                "relative_infectiousness_asymptomatic": 0.35,
                "infectious_duration_symptomatic": 21.0,
                "infectious_duration_asymptomatic": 14.0,
                "fitness_R": 1.2,
                "resistance_prevalence": 0.1,
                "reporting_trend_end_multiplier": 1.0,
            }
            assert set(_sample_columns()).issubset(row)
            rows.append(row)
    return pd.DataFrame(rows)


def test_select_posterior_draws_is_seeded_and_balanced_by_country() -> None:
    samples = _posterior_samples(draws=8)

    first = fitness_grid._select_posterior_draws(
        samples,
        countries=["A", "B"],
        draws_per_country=3,
        seed=123,
    )
    second = fitness_grid._select_posterior_draws(
        samples,
        countries=["A", "B"],
        draws_per_country=3,
        seed=123,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("country").size().to_dict() == {"A": 3, "B": 3}
    assert first.groupby("country")["posterior_draw"].apply(list).to_dict() == {
        "A": [1, 2, 3],
        "B": [1, 2, 3],
    }


def test_grid_overrides_are_applied_after_posterior_sample_parameters() -> None:
    config = {
        "transmission": {
            "beta_S": 0.01,
            "relative_infectiousness_asymptomatic": 0.1,
            "fitness_R": 0.9,
        },
        "reporting_multiplier": 1.0,
        "vaccine": {"VE_sus": 0.0, "VE_inf": 0.0, "VE_dur": 0.0},
        "natural_history": {
            "infectious_duration_symptomatic": 10.0,
            "infectious_duration_asymptomatic": 10.0,
        },
        "initial_conditions": {},
        "resistance": {},
        "importation": {},
        "simulation": {"start_time": 0.0, "end_time": 365.0},
    }
    sample = _posterior_samples(countries=("A",), draws=1).iloc[0]

    sampled = _apply_sample(config, fitness_grid._posterior_sample_from_row(sample))
    overridden = fitness_grid._apply_grid_overrides(sampled, fitness_r=0.85, ve_inf=0.55)

    assert np.isclose(overridden["transmission"]["beta_S"], sample["beta_S"])
    assert np.isclose(overridden["reporting_multiplier"], sample["reporting_multiplier"])
    assert np.isclose(overridden["transmission"]["fitness_R"], 0.85)
    assert np.isclose(overridden["vaccine"]["VE_inf"], 0.55)


def test_selected_posterior_sample_diagnostics_describe_varying_parameters() -> None:
    samples = _posterior_samples(countries=("A",), draws=4)

    diagnostics = fitness_grid._summarise_selected_posterior_samples(samples)
    by_parameter = diagnostics.set_index("parameter")

    assert len(diagnostics) == len(tuple(_sample_columns()))
    assert by_parameter.loc["beta_S", "posterior_draws"] == 4
    assert by_parameter.loc["beta_S", "unique_values"] == 4
    assert bool(by_parameter.loc["beta_S", "varies_within_country"])
    assert not bool(by_parameter.loc["VE_sus", "varies_within_country"])
    assert bool(by_parameter.loc["VE_inf", "grid_override_in_fig3d"])
    assert bool(by_parameter.loc["fitness_R", "grid_override_in_fig3d"])
    assert by_parameter.loc["beta_S", "uncertainty_source"] == fitness_grid.UNCERTAINTY_SOURCE


def test_summarise_posterior_benefits_uses_paired_low_high_draws() -> None:
    high_values = [80.0, 70.0, 60.0, 50.0]
    rows = []
    for draw, high in enumerate(high_values, start=1):
        rows.extend(
            [
                {
                    "country": "A",
                    "posterior_draw": draw,
                    "fitness_group": "Neutral (1.00)",
                    "grid_fitness_R": 1.0,
                    "low_grid_VE_inf": 0.05,
                    "high_grid_VE_inf": 0.55,
                    "ve_endpoint": "low",
                    "annualized_infant_cases_per_100k": 100.0,
                },
                {
                    "country": "A",
                    "posterior_draw": draw,
                    "fitness_group": "Neutral (1.00)",
                    "grid_fitness_R": 1.0,
                    "low_grid_VE_inf": 0.05,
                    "high_grid_VE_inf": 0.55,
                    "ve_endpoint": "high",
                    "annualized_infant_cases_per_100k": high,
                },
            ]
        )
    summary = fitness_grid._summarise_posterior_benefits(pd.DataFrame(rows))
    benefits = 1.0 - np.array(high_values) / 100.0

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["posterior_draws"] == 4
    assert row["uncertainty_source"] == fitness_grid.UNCERTAINTY_SOURCE
    assert np.isclose(row["median_relative_benefit"], np.percentile(benefits, 50.0))
    assert np.isclose(row["q025_relative_benefit"], np.percentile(benefits, 2.5))
    assert np.isclose(row["q975_relative_benefit"], np.percentile(benefits, 97.5))


def test_fig3d_psa_nuisance_sample_keeps_grid_overrides_fixed() -> None:
    config = {
        "age_groups": [
            {"label": "infant_0_2m"},
            {"label": "child_1_4y"},
            {"label": "young_adult_18_39y"},
        ],
        "contact_matrix": {"rows": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]},
        "transmission": {
            "relative_infectiousness_asymptomatic": 0.25,
            "fitness_R": 1.20,
        },
        "natural_history": {"infectious_duration_asymptomatic": 10.0},
        "vaccine": {"VE_inf": 0.25},
        "PEP": {"coverage_household_contacts": 0.40},
        "reporting_multiplier": 1.0,
    }
    sample = {
        "psa_sample_id": 7,
        "infant_contact_multiplier": 1.50,
        "VE_inf_baseline": 0.60,
        "relative_infectiousness_asymptomatic": 0.70,
        "infectious_duration_asymptomatic": 18.0,
        "fitness_R": 1.25,
        "PEP_coverage_multiplier": 1.50,
    }

    sampled = fitness_grid._apply_fig3d_psa_nuisance_sample(config, sample)
    overridden = fitness_grid._apply_grid_overrides(sampled, fitness_r=0.85, ve_inf=0.55)

    assert np.isclose(overridden["reporting_multiplier"], 1.0)
    assert np.isclose(overridden["contact_matrix"]["rows"][0][1], 3.0)
    assert np.isclose(overridden["contact_matrix"]["rows"][0][2], 4.5)
    assert np.isclose(overridden["transmission"]["relative_infectiousness_asymptomatic"], 0.70)
    assert np.isclose(overridden["natural_history"]["infectious_duration_asymptomatic"], 18.0)
    assert np.isclose(overridden["PEP"]["coverage_household_contacts"], 0.60)
    assert np.isclose(overridden["transmission"]["fitness_R"], 0.85)
    assert np.isclose(overridden["vaccine"]["VE_inf"], 0.55)


def test_fig3d_psa_loader_accepts_current_true_case_design_without_reporting(tmp_path) -> None:
    row = {
        "psa_sample_id": 1,
        "sample_design": "latin_hypercube_inverse_cdf",
        "uncertainty_schema_version": 1,
        "infant_contact_multiplier": 1.0,
        "VE_inf_baseline": 0.25,
        "relative_infectiousness_asymptomatic": 0.5,
        "infectious_duration_asymptomatic": 17.0,
        "fitness_R": 1.0,
        "resistance_management_uptake": 0.7,
        "PEP_coverage_multiplier": 1.0,
    }
    path = tmp_path / "psa.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    loaded = fitness_grid._load_psa_samples(path)

    assert loaded["psa_sample_id"].tolist() == [1]
    assert "reporting_multiplier" not in loaded.columns


def test_summarise_psa_benefits_uses_paired_low_high_samples() -> None:
    high_values = [90.0, 65.0, 40.0, 30.0]
    rows = []
    for sample_id, high in enumerate(high_values, start=1):
        rows.extend(
            [
                {
                    "country": "A",
                    "psa_sample_id": sample_id,
                    "fitness_group": "Advantage (1.10)",
                    "grid_fitness_R": 1.10,
                    "low_grid_VE_inf": 0.05,
                    "high_grid_VE_inf": 0.55,
                    "ve_endpoint": "low",
                    "annualized_infant_cases_per_100k": 100.0,
                },
                {
                    "country": "A",
                    "psa_sample_id": sample_id,
                    "fitness_group": "Advantage (1.10)",
                    "grid_fitness_R": 1.10,
                    "low_grid_VE_inf": 0.05,
                    "high_grid_VE_inf": 0.55,
                    "ve_endpoint": "high",
                    "annualized_infant_cases_per_100k": high,
                },
            ]
        )

    draws, summary = fitness_grid._summarise_psa_benefits(pd.DataFrame(rows))
    benefits = 1.0 - np.array(high_values) / 100.0

    assert len(draws) == 4
    assert set(draws["psa_sample_id"]) == {1, 2, 3, 4}
    assert np.allclose(draws.sort_values("psa_sample_id")["relative_benefit"], benefits)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["psa_samples"] == 4
    assert row["uncertainty_source"] == fitness_grid.PSA_UNCERTAINTY_SOURCE
    assert np.isclose(row["median_relative_benefit"], np.percentile(benefits, 50.0))
    assert np.isclose(row["q025_relative_benefit"], np.percentile(benefits, 2.5))
    assert np.isclose(row["q975_relative_benefit"], np.percentile(benefits, 97.5))
