from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src_python.model.compartments import (
    COMPARTMENTS,
    VACCINE_ORIGINS,
    StateIndex,
    exposed_name,
    infectious_name,
    susceptible_name,
)
from src_python.model.contact_matrix import balance_reciprocity, reciprocity_error
from src_python.model.force_of_infection import _npi_contact_reduction_at
from src_python.model.ode_system import _routine_delivery_multiplier_at, rhs
from src_python.model.outputs import (
    GREGORIAN_YEAR_DAYS,
    _compute_severity_outcomes,
    _daily_metrics,
    active_resistant_fraction,
    initial_state,
    solve_model,
    summarize_timeseries,
)
from src_python.model.parameters import PreparedParameters
from src_python.calibration.calibrate_baseline import (
    annual_reported_cases,
    apply_calibration_vector,
    align_annual_case_series,
    initial_calibration_vector,
    observed_annual_cases,
    observed_annual_case_frame,
    reported_cases_for_observed_intervals,
    reporting_rate_prior_penalty,
)
from src_python.simulation import common as simulation_common
from src_python.simulation.common import add_relative_reductions, load_configs, make_config, run_prepared_config, validate_run_metadata
from src_python.utils.io import load_yaml, project_path
from src_python.utils.validation import validate_timeseries


def _baseline_params(*, country_profile: str | None = None, fixed_population: bool = True) -> PreparedParameters:
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile=country_profile,
    )
    config["simulation"]["burn_in_years"] = 2
    config["simulation"]["end_time"] = 365
    if fixed_population:
        # Keep legacy conservation-based tests by running in the fixed-profile mode.
        config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    return PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="baseline",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )


def test_initial_population_matches_config():
    params = _baseline_params()
    index = StateIndex(params.age_groups)
    y0 = initial_state(params, index)
    age_totals = index.reshape(y0).sum(axis=1)
    assert np.allclose(age_totals, params.population)


def test_solution_population_conserved_and_nonnegative():
    params = _baseline_params()
    index = StateIndex(params.age_groups)
    y0 = initial_state(params, index)
    solution = solve_model(params, index)
    assert solution.success
    totals = solution.y.reshape(index.n_age, index.n_compartments, -1).sum(axis=(0, 1))
    expected = float(index.reshape(y0).sum())
    assert np.max(np.abs(totals - expected)) < params.total_population * 1e-8
    assert solution.y.min() > -1e-5


def test_wpp_trajectory_drives_population_toward_target():
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile="Australia",
    )
    # Short window so we can verify WPP nudging moves the total population from
    # the 2024 snapshot toward the 2026 trajectory target.
    config["calendar"]["analysis_start_date"] = "2025-01-01"
    config["simulation"]["initial_state_strategy"] = "fixed_duration"
    config["simulation"]["burn_in_years"] = 1
    config["simulation"]["end_time"] = 365
    config["transmission"]["beta_S"] = 0.0
    config["importation"]["enabled"] = False
    config["initial_conditions"]["initial_exposed_per_100k"] = 0.0
    config["initial_conditions"]["initial_infectious_per_100k"] = 0.0
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="wpp_trajectory",
    )
    assert params.wpp_trajectory_active()
    index = StateIndex(params.age_groups)
    solution = solve_model(params, index)
    assert solution.success
    final_state = solution.y[:, -1].reshape(index.n_age, index.n_compartments).sum(axis=1)
    snapshot_total = float(params.total_population)
    wpp_target_start = float(params.wpp_population_at(2025.0).sum())
    wpp_target_end = float(params.wpp_population_at(2026.0).sum())
    final_total = float(final_state.sum())
    # The modelled total population should sit between the snapshot and the
    # corresponding WPP target (monotone nudging) rather than staying pinned to
    # the snapshot.
    lower = min(snapshot_total, wpp_target_end)
    upper = max(snapshot_total, wpp_target_end)
    assert lower - 1.0 <= final_total <= upper + 1.0
    # And materially move away from the snapshot toward the WPP target.
    snapshot_error = abs(final_total - snapshot_total)
    target_error = abs(final_total - wpp_target_end)
    assert target_error < snapshot_error


def test_wpp_runtime_cache_does_not_mutate_config():
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile="Brazil",
    )
    config["simulation"]["burn_in_years"] = 0
    config["simulation"]["end_time"] = 30
    trajectory = config["demography"]["wpp_trajectory"]

    run_prepared_config(
        config,
        analysis="test",
        scenario="wpp_cache_isolated",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )

    assert not any(str(key).startswith("_wpp_") for key in trajectory)


def test_demography_keeps_config_age_population_as_fixed_point():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate", country_profile="China")
    config["simulation"]["burn_in_years"] = 3
    config["simulation"]["end_time"] = 28
    # Exercise the fixed-population branch explicitly. The WPP-driven branch
    # is covered by test_wpp_trajectory_drives_population_toward_target.
    config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    config["transmission"]["beta_S"] = 0.0
    config["importation"]["enabled"] = False
    config["initial_conditions"]["initial_exposed_per_100k"] = 0.0
    config["initial_conditions"]["initial_infectious_per_100k"] = 0.0
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="age_fixed_point",
    )
    index = StateIndex(params.age_groups)
    solution = solve_model(params, index)
    age_totals = solution.y[:, 0].reshape(index.n_age, index.n_compartments).sum(axis=1)
    assert np.allclose(age_totals, params.population, rtol=1e-4, atol=1e-2)


def test_timeseries_has_valid_epidemiological_columns():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["simulation"]["burn_in_years"] = 1
    config["simulation"]["end_time"] = 56
    timeseries, summary = run_prepared_config(
        config,
        analysis="test",
        scenario="baseline",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    validate_timeseries(timeseries)
    first = timeseries.loc[np.isclose(timeseries["time"], timeseries["time"].min())]
    assert first["cumulative_cases"].max() == 0.0
    assert first["total_infections"].max() == 0.0
    for col in [
        "calendar_date",
        "calendar_year",
        "maternal_origin_infection_share",
        "dose1_origin_infection_share",
        "dose2_origin_infection_share",
        "dose3plus_origin_infection_share",
    ]:
        assert col in timeseries
    for col in [
        "calendar_start_date",
        "calendar_end_date",
        "calendar_start_year",
        "calendar_end_year",
        "maternal_origin_infection_share",
        "dose1_origin_infection_share",
        "dose2_origin_infection_share",
        "dose3plus_origin_infection_share",
    ]:
        assert col in summary


def test_severity_cascade_uses_configured_probabilities():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="configured_severity",
    )
    severity = params.severity_model

    assert severity["age_death_given_hospitalization"]["infant_0_2m"] == pytest.approx(0.020)
    assert severity["age_death_given_hospitalization"]["infant_3_11m"] == pytest.approx(0.005)
    assert severity["origin_conditional_ve_hospitalization"]["maternal"] == pytest.approx(0.50)

    hospitalization, deaths, symptomatic = _compute_severity_outcomes(
        {"maternal": 100.0},
        "infant_0_2m",
        severity_model=severity,
    )
    assert symptomatic == pytest.approx(100.0)
    assert hospitalization == pytest.approx(100.0 * 0.60 * (1.0 - 0.50))
    assert deaths == pytest.approx(hospitalization * 0.020 * (1.0 - 0.80))

    overall_maternal_ve_hosp = 1.0 - (
        (1.0 - 0.75 * 0.55)
        * (1.0 - 0.75 * 0.92)
        * (1.0 - severity["origin_conditional_ve_hospitalization"]["maternal"])
    )
    assert overall_maternal_ve_hosp == pytest.approx(0.91, abs=0.002)
    assert "structural cascade mapping" in severity["note"]


def test_legacy_severity_fallback_uses_documented_infant_cfr():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config.pop("severity_model")
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="legacy_severity_fallback",
    )
    assert params.severity_model == {}

    hospitalization, deaths, _ = _compute_severity_outcomes(
        {"unvaccinated": 100.0},
        "infant_0_2m",
    )

    assert hospitalization == pytest.approx(60.0)
    assert deaths == pytest.approx(60.0 * 0.020)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda severity: severity["age_hospitalization_probability"].pop("infant_0_2m"),
            "invalid keys",
        ),
        (
            lambda severity: severity["origin_conditional_ve_death"].__setitem__("maternal", 1.01),
            r"within \[0, 1\]",
        ),
        (
            lambda severity: severity.__setitem__("age_hospitalisation_probability", {}),
            "unknown keys",
        ),
        (
            lambda severity: severity.pop("origin_conditional_ve_hospitalization"),
            "all severity probability maps",
        ),
    ],
)
def test_severity_configuration_rejects_invalid_keys_and_probabilities(mutate, message):
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    mutate(config["severity_model"])

    with pytest.raises(ValueError, match=message):
        PreparedParameters.from_config(
            config,
            analysis="test",
            scenario="invalid_severity",
        )


def test_daily_metrics_uses_origin_cached_maternal_susceptibility():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["vaccine"]["VE_sus"] = 0.0
    config["vaccine"]["VE_sym"] = 0.0
    config["immunity_model"]["maternal_relative_effect"] = 1.0
    config["immunity_model"]["maternal_VE_sus"] = 1.0
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="daily_metrics_maternal_cache",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    index = StateIndex(params.age_groups)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    age_idx = 0
    state[age_idx, c[susceptible_name("unvaccinated")]] = 1000.0
    state[age_idx, c[susceptible_name("maternal")]] = 1000.0
    state[age_idx, c[infectious_name("S", "sym", "unvaccinated")]] = 10.0

    rows = _daily_metrics(0.0, index.flatten(state), params, index)
    row = next(r for r in rows if r["age_group"] == params.age_groups[age_idx] and r["strain"] == "sensitive")

    assert row["total_infection_rate_per_day"] > 0.0
    assert row["maternal_origin_infection_share"] == pytest.approx(0.0)


def test_daily_metrics_treated_cases_use_absolute_diagnosis_probability():
    params = _baseline_params()
    index = StateIndex(params.age_groups)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    young_age_idx = 0
    older_age_idx = index.age_groups.index("elderly_65plus")
    i_sym = c[infectious_name("S", "sym", "unvaccinated")]
    state[young_age_idx, i_sym] = 100.0
    state[older_age_idx, i_sym] = 100.0

    rows = _daily_metrics(0.0, index.flatten(state), params, index)
    young = next(r for r in rows if r["age_group"] == params.age_groups[young_age_idx] and r["strain"] == "sensitive")
    older = next(r for r in rows if r["age_group"] == params.age_groups[older_age_idx] and r["strain"] == "sensitive")
    tr_sym = float(params.treatment["treatment_rate_symptomatic"])

    assert young["treated_case_rate_per_day"] == pytest.approx(tr_sym * params.diagnosis_probability[young_age_idx] * 100.0)
    assert older["treated_case_rate_per_day"] == pytest.approx(tr_sym * params.diagnosis_probability[older_age_idx] * 100.0)
    assert older["treated_case_rate_per_day"] < young["treated_case_rate_per_day"]


def test_summary_uses_infant_population_for_infant_incidence():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["simulation"]["burn_in_years"] = 1
    config["simulation"]["end_time"] = 56
    _, summary = run_prepared_config(
        config,
        analysis="test",
        scenario="infant_denominator",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    row = summary.iloc[0]
    assert row["infant_population"] < row["total_population"]
    expected = (
        row["total_infant_cases"]
        / max(row["analysis_years"] * row["infant_population"], 1e-9)
        * 100_000.0
    )
    assert np.isclose(row["annualized_infant_cases_per_100k"], expected)


def test_summary_uses_child_adolescent_population_for_child_adolescent_incidence():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["simulation"]["burn_in_years"] = 1
    config["simulation"]["end_time"] = 56
    _, summary = run_prepared_config(
        config,
        analysis="test",
        scenario="child_adolescent_denominator",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    row = summary.iloc[0]
    assert row["child_adolescent_population"] > row["infant_population"]
    assert row["total_child_adolescent_cases"] >= row["total_infant_cases"]
    expected = (
        row["total_child_adolescent_cases"]
        / max(row["analysis_years"] * row["child_adolescent_population"], 1e-9)
        * 100_000.0
    )
    assert np.isclose(row["annualized_child_adolescent_cases_per_100k"], expected)


def test_reporting_multiplier_does_not_change_true_transmission_dynamics():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["simulation"]["burn_in_years"] = 1
    config["simulation"]["end_time"] = 56
    low_reporting = deepcopy(config)
    high_reporting = deepcopy(config)
    low_reporting["reporting_multiplier"] = 0.5
    high_reporting["reporting_multiplier"] = 1.5

    _, low_summary = run_prepared_config(
        low_reporting,
        analysis="test",
        scenario="low_reporting",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    _, high_summary = run_prepared_config(
        high_reporting,
        analysis="test",
        scenario="high_reporting",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )

    assert np.isclose(low_summary["total_infections"].iloc[0], high_summary["total_infections"].iloc[0])
    assert high_summary["total_reported_cases"].iloc[0] > low_summary["total_reported_cases"].iloc[0]


def test_routine_delivery_shock_reduces_vaccination_flow():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["transmission"]["beta_S"] = 0.0
    config["importation"]["enabled"] = False
    config["initial_conditions"]["initial_exposed_per_100k"] = 0.0
    config["initial_conditions"]["initial_infectious_per_100k"] = 0.0
    config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    config["calendar"]["analysis_start_date"] = "2020-01-01"
    config["routine_vaccination"]["delivery_shock_periods"] = [
        {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "reduction": 1.0,
            "ramp_days": 0,
        }
    ]
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="routine_delivery_shock",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    assert _routine_delivery_multiplier_at(0.0, params) == pytest.approx(0.0)

    no_shock_config = deepcopy(config)
    no_shock_config["routine_vaccination"].pop("delivery_shock_periods")
    no_shock_params = PreparedParameters.from_config(
        no_shock_config,
        analysis="test",
        scenario="routine_delivery_no_shock",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    index = StateIndex(params.age_groups)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    age_idx = index.age_groups.index("child_1_4y")
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    state[age_idx, c["S"]] = 1000.0

    dy_shock = index.reshape(rhs(0.0, index.flatten(state), params, index))
    dy_no_shock = index.reshape(rhs(0.0, index.flatten(state), no_shock_params, index))

    assert dy_no_shock[age_idx, c[susceptible_name("recent")]] > 0.0
    assert dy_shock[age_idx, c[susceptible_name("recent")]] == pytest.approx(0.0)


def test_maternal_proxy_does_not_age_into_long_lived_vaccine_state():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["transmission"]["beta_S"] = 0.0
    config["importation"]["enabled"] = False
    config["routine_vaccination"]["enabled"] = False
    # Use fixed-population mode to avoid WPP nudging interfering with the test
    config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="maternal_proxy",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    index = StateIndex(params.age_groups)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    state[0, c["S"]] = 900.0
    state[0, c["M_protected"]] = 100.0

    dy = index.reshape(rhs(0.0, index.flatten(state), params, index))
    aging_rate = 1.0 / (0.1667 * 365.0)

    assert np.isclose(dy[1, c["M_protected"]], 0.0)
    assert np.isclose(dy[1, c["S"]], aging_rate * 1000.0)


def test_invalid_probability_inputs_are_rejected_instead_of_silently_clipped():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate")
    config["age_groups"][0]["vaccine_coverage"] = 1.2
    with pytest.raises(ValueError, match="vaccine coverage"):
        PreparedParameters.from_config(
            config,
            analysis="test",
            scenario="invalid_coverage",
            vaccine_scenario="symptom_protective",
            resistance_scenario="moderate",
        )


def test_vaccine_source_compartments_track_recent_and_waned_breakthroughs():
    params = _baseline_params()
    index = StateIndex(params.age_groups)
    y0 = initial_state(params, index)
    state = index.reshape(y0)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    source_seeded = [
        float(state[:, c[exposed_name("S", origin)]].sum() + state[:, c[infectious_name("S", "sym", origin)]].sum())
        for origin in VACCINE_ORIGINS
    ]
    assert source_seeded[1] > 0.0
    assert source_seeded[2] > 0.0


def test_initial_state_tracks_maternal_and_partial_dose_history_explicitly():
    params = _baseline_params()
    index = StateIndex(params.age_groups)
    state = index.reshape(initial_state(params, index))
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}

    assert state[index.age_groups.index("infant_0_2m"), c["M_protected"]] > 0.0
    assert state[index.age_groups.index("infant_3_11m"), c["V_dose1_recent"]] > 0.0
    assert state[index.age_groups.index("infant_3_11m"), c["V_dose2_recent"]] > 0.0
    assert state[index.age_groups.index("child_1_4y"), c["V_recent"]] > 0.0


def test_burn_in_rebalances_resistance_to_analysis_start_target():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="low", country_profile="Australia")
    config["simulation"]["burn_in_years"] = 2
    config["simulation"]["end_time"] = 28
    params = PreparedParameters.from_config(
        config,
        analysis="test",
        scenario="resistance_rebalance",
        vaccine_scenario="symptom_protective",
        resistance_scenario="low",
    )
    index = StateIndex(params.age_groups)
    solution = solve_model(params, index)
    assert solution.success
    assert np.isclose(active_resistant_fraction(solution.y[:, 0], index), 0.05, atol=1e-8)


def test_country_resistance_timeline_sets_country_specific_anchor():
    china = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="country_timeline",
        country_profile="China",
    )
    united_states = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="country_timeline",
        country_profile="United_States",
    )

    assert china["resistance"]["target_prevalence_at_analysis_start"] > 0.90
    assert 0.0 < united_states["resistance"]["target_prevalence_at_analysis_start"] <= 0.05
    assert united_states["resistance"]["country_timeline"]["evidence_years"] == "2024"
    assert united_states["resistance"]["country_timeline"]["evidence_type"] == "low_detected_model_anchor"
    assert "No population-based national resistant fraction" in united_states["resistance"]["country_timeline"]["notes"]
    assert china["importation"]["resistant_fraction"] == china["resistance"]["importation_fraction"]
    assert united_states["importation"]["resistant_fraction"] == united_states["resistance"]["importation_fraction"]
    assert china["resistance"]["prevalence_anchor_rate_per_year"] == 2.0
    assert not china["resistance"].get("anchor_during_dynamics", False)
    assert china["metadata"]["resistance_timeline_anchor_year"] == 2025
    assert not china["metadata"]["resistance_timeline_allows_future_evidence"]


def test_generic_resistance_scenario_is_not_replaced_by_country_timeline():
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile="China",
    )
    assert np.isclose(config["resistance"]["target_prevalence_at_analysis_start"], 0.30)


def test_explicit_resistance_override_takes_precedence_over_country_timeline():
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="country_timeline",
        country_profile="China",
        resistance_overrides={
            "target_prevalence_at_analysis_start": 0.40,
            "initial_resistance_prevalence": 0.40,
            "importation_fraction": 0.40,
        },
    )
    assert np.isclose(config["resistance"]["target_prevalence_at_analysis_start"], 0.40)
    assert np.isclose(config["importation"]["resistant_fraction"], 0.40)


def test_country_npi_timeline_uses_external_mean_reduction_by_default():
    baseline = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile="China",
        load_calibration=False,
    )
    assert baseline["transmission"]["npi_contact_reduction_periods"]
    assert baseline["transmission"]["npi_contact_reduction_periods"][0]["reduction"] > 0.0
    assert (
        baseline["metadata"]["npi_contact_reduction_reduction_column"]
        == "contact_reduction_mean"
    )

    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile="China",
        load_calibration=False,
        config_overrides={
            "transmission": {
                "npi_contact_reduction": {
                    "enabled": True,
                    "reduction_column": "contact_reduction_mean",
                }
            }
        },
    )
    config["calendar"]["analysis_start_date"] = "2020-01-01"
    config["simulation"]["start_time"] = 0.0
    config["simulation"]["end_time"] = 365.0
    params = PreparedParameters.from_config(config, analysis="test", scenario="npi_timeline")

    assert config["transmission"]["npi_contact_reduction_periods"]
    assert config["transmission"]["npi_contact_reduction_periods"][0]["reduction"] > 0.0
    assert _npi_contact_reduction_at(40.0, params) < 1.0
    assert config["metadata"]["npi_contact_reduction_periods_loaded"] >= 1


def test_directional_contact_matrix_reduction_skips_reciprocity_rebalancing():
    base_config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile="Australia",
        load_calibration=False,
    )
    intervention = load_configs()["interventions"]["combined_strategy"]
    reduced = simulation_common.apply_intervention_definition(base_config, intervention)
    age_labels = [record["label"] for record in reduced["age_groups"]]
    target_idx = age_labels.index("infant_0_2m")
    source_idx = age_labels.index("young_adult_18_39y")

    assert reduced["contact_matrix"]["reciprocity_correction"]["enabled"] is False
    assert reduced["metadata"]["contact_matrix_directional_intervention"] is True
    assert reduced["contact_matrix"]["rows"][target_idx][source_idx] == pytest.approx(
        base_config["contact_matrix"]["rows"][target_idx][source_idx] * 0.85
    )


def test_equal_strain_effects_do_not_drive_resistance_fixation():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate", country_profile="Australia")
    config["simulation"]["burn_in_years"] = 1
    config["simulation"]["end_time"] = 365
    config["treatment"]["resistant"] = dict(config["treatment"]["sensitive"])
    config["PEP"]["effectiveness_resistant"] = config["PEP"]["effectiveness_sensitive"]
    config["transmission"]["fitness_R"] = 1.0
    config["importation"]["resistant_fraction"] = 0.30
    config["resistance"]["importation_fraction"] = 0.30
    _, summary = run_prepared_config(
        config,
        analysis="test",
        scenario="neutral_resistance",
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
    )
    assert float(summary["resistant_fraction_end"].iloc[0]) < 0.90
    assert np.isclose(float(summary["resistant_fraction_start"].iloc[0]), 0.30, atol=0.05)


def test_contact_matrix_reciprocity_correction_balances_population_flows():
    matrix = np.array([[4.0, 8.0], [1.0, 3.0]])
    population = np.array([100.0, 1000.0])
    corrected = balance_reciprocity(matrix, population)
    assert reciprocity_error(matrix, population) > 0.0
    assert reciprocity_error(corrected, population) < 1e-12


def test_calibration_vector_updates_declared_parameters():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate", country_profile="Australia")
    vector = initial_calibration_vector(config)
    vector[0] = np.log(0.041)
    vector[1] = np.log(1.25)
    updated = apply_calibration_vector(config, vector)
    assert np.isclose(updated["transmission"]["beta_S"], 0.041)
    assert np.isclose(updated["reporting_multiplier"], 1.25)
    assert updated["importation"]["resistant_fraction"] == updated["resistance"]["importation_fraction"]


def test_reporting_rate_prior_penalty_is_zero_at_baseline_and_positive_when_shifted():
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate", country_profile="Australia")
    assert np.isclose(reporting_rate_prior_penalty(config), 0.0)

    config["reporting_multiplier"] = 1.5
    assert reporting_rate_prior_penalty(config) > 0.0


def test_calibration_observed_cases_use_all_harmonized_intervals():
    observed = observed_annual_case_frame("Australia")
    assert "observed_interval_id" in observed.columns
    assert observed["observed_year"].max() >= 2025
    assert observed_annual_cases("Australia").shape[0] == observed.shape[0]


def test_calibration_alignment_uses_shared_years_only():
    predicted = pd.DataFrame(
        {
            "series_year": [0, 1, 2, 3],
            "reported_cases": [10.0, 11.0, 12.0, 13.0],
        }
    )
    observed = pd.DataFrame(
        {
            "series_year": [0, 1],
            "observed_year": [2019, 2020],
            "reported_cases": [3.0, 4.0],
        }
    )

    aligned = align_annual_case_series(predicted, observed)
    assert list(aligned["series_year"]) == [0, 1]
    assert list(aligned["predicted_reported_cases"]) == [10.0, 11.0]
    assert list(aligned["observed_reported_cases"]) == [3.0, 4.0]


def test_annual_reported_cases_can_align_by_calendar_year():
    timeseries = pd.DataFrame(
        {
            "calendar_year": [2020, 2020, 2021],
            "time": [0.0, 7.0, 365.0],
            "reported_cases": [1.0, 2.0, 4.0],
        }
    )

    annual = annual_reported_cases(timeseries)

    assert list(annual["observed_year"]) == [2020, 2021]
    assert list(annual["reported_cases"]) == [3.0, 4.0]


def test_annual_reported_cases_splits_calendar_intervals_across_years():
    timeseries = pd.DataFrame(
        {
            "analysis": ["test", "test"],
            "scenario": ["candidate", "candidate"],
            "age_group": ["all", "all"],
            "strain": ["sensitive", "sensitive"],
            "time": [0.0, 20.0],
            "calendar_date": ["2020-12-22", "2021-01-11"],
            "reported_cases": [0.0, 20.0],
        }
    )

    annual = annual_reported_cases(timeseries)

    assert list(annual["observed_year"]) == [2020, 2021]
    assert np.allclose(annual["reported_cases"], [10.0, 10.0])


def test_reported_cases_can_align_to_observed_partial_intervals():
    timeseries = pd.DataFrame(
        {
            "analysis": ["test", "test", "test"],
            "scenario": ["candidate", "candidate", "candidate"],
            "age_group": ["all", "all", "all"],
            "strain": ["sensitive", "sensitive", "sensitive"],
            "time": [0.0, 10.0, 20.0],
            "calendar_date": ["2020-01-01", "2020-01-11", "2020-01-21"],
            "reported_cases": [0.0, 10.0, 20.0],
        }
    )
    observed = pd.DataFrame(
        {
            "series_year": [0, 1],
            "observed_year": [2020, 2020],
            "observed_interval_id": ["a", "b"],
            "period_start": pd.to_datetime(["2020-01-01", "2020-01-06"]),
            "period_end": pd.to_datetime(["2020-01-06", "2020-01-21"]),
            "reported_cases": [5.0, 25.0],
        }
    )

    predicted = reported_cases_for_observed_intervals(timeseries, observed)
    aligned = align_annual_case_series(predicted, observed)

    assert list(predicted["observed_interval_id"]) == ["a", "b"]
    assert np.allclose(predicted["reported_cases"], [5.0, 25.0])
    assert np.allclose(aligned["predicted_reported_cases"], [5.0, 25.0])


def test_summarize_timeseries_detects_recurring_peak_intervals():
    times = np.arange(0.0, 9.0 * 365.0, 30.0)
    peak_centers = np.array([365.0, 4.0 * 365.0, 7.0 * 365.0])
    baseline = np.full_like(times, 15.0, dtype=float)
    for center in peak_centers:
        baseline += 120.0 * np.exp(-0.5 * ((times - center) / 60.0) ** 2)
    timeseries = pd.DataFrame(
        {
            "analysis": "test",
            "scenario": "demo",
            "vaccine_scenario": "symptom_protective",
            "resistance_scenario": "country_timeline",
            "intervention": "",
            "age_group": "adult_18plus",
            "strain": "sensitive",
            "time": times,
            "total_infections": baseline,
            "total_infection_rate_per_day": baseline,
            "symptomatic_cases": baseline * 0.5,
            "symptomatic_case_rate_per_day": baseline * 0.5,
            "reported_cases": baseline * 0.2,
            "reported_case_rate_per_day": baseline * 0.2,
            "asymptomatic_infections": baseline * 0.5,
            "infant_cases": np.zeros_like(times),
            "infant_infections": np.zeros_like(times),
            "treated_cases": np.zeros_like(times),
            "PEP_averted_cases": np.zeros_like(times),
            "active_resistant_fraction": np.zeros_like(times),
            "population": np.full_like(times, 1_000_000.0),
            "total_population": np.full_like(times, 1_000_000.0),
        }
    )

    summary = summarize_timeseries(timeseries, dt=30.0)
    row = summary.iloc[0]

    assert row["n_epidemic_peaks"] == 3
    assert row["analysis_years"] == pytest.approx(
        (times.max() - times.min()) / GREGORIAN_YEAR_DAYS
    )
    assert np.isfinite(row["mean_peak_interval_years"])
    assert 2.5 < row["mean_peak_interval_years"] < 3.5


def test_country_profiles_carry_reporting_prior_bands_into_runtime_config():
    configs = load_configs()["countries"]
    australia_prior = configs["Australia"]["reporting_rate_prior"]
    sweden_prior = configs["Sweden"]["reporting_rate_prior"]

    assert australia_prior["method"] == "literature_range"
    assert sweden_prior["evidence_class"] == "direct_preschool_anchor"
    assert australia_prior["age_groups"]["infant_0_2m"]["lower"] < 0.60 < australia_prior["age_groups"]["infant_0_2m"]["upper"]
    assert sweden_prior["age_groups"]["child_1_4y"]["lower"] > australia_prior["age_groups"]["child_1_4y"]["lower"]

    runtime = make_config(vaccine_scenario="symptom_protective", resistance_scenario="moderate", country_profile="Sweden")
    assert runtime["reporting_rate_prior"]["method"] == "literature_range"
    assert runtime["reporting_rate_prior"]["note"]


def test_runtime_config_exposes_default_reporting_multiplier():
    runtime = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario="moderate",
        country_profile="Japan",
        load_calibration=False,
    )

    assert runtime["reporting_multiplier"] == 1.0


def test_missing_run_metadata_is_rejected():
    with pytest.raises(FileNotFoundError):
        validate_run_metadata("__missing_test_output__")


def _run_metadata_fixture() -> dict[str, object]:
    return {
        "schema_version": simulation_common.METADATA_SCHEMA_VERSION,
        "stem": "test_output",
        "config_hash": "current-config",
        "source_code_hash": "current-source",
        "dependency_versions": {
            "python": "3.12.3",
            **{name: "1.0" for name in simulation_common.DEPENDENCY_VERSION_PACKAGES},
        },
        "row_counts": {},
    }


def test_run_metadata_missing_dependency_version_is_rejected(monkeypatch):
    metadata = _run_metadata_fixture()
    metadata["dependency_versions"].pop("numba")
    monkeypatch.setattr(simulation_common, "read_run_metadata", lambda stem: metadata)
    monkeypatch.setattr(simulation_common, "config_fingerprint", lambda: "current-config")
    monkeypatch.setattr(simulation_common, "source_code_fingerprint", lambda: "current-source")

    with pytest.raises(ValueError, match="dependency_versions missing numba"):
        simulation_common.validate_run_metadata("test_output")


def test_run_metadata_dependency_check_can_be_deferred_for_qc(monkeypatch):
    metadata = _run_metadata_fixture()
    metadata["dependency_versions"].pop("numba")
    monkeypatch.setattr(simulation_common, "read_run_metadata", lambda stem: metadata)
    monkeypatch.setattr(simulation_common, "config_fingerprint", lambda: "current-config")
    monkeypatch.setattr(simulation_common, "source_code_fingerprint", lambda: "current-source")

    assert simulation_common.validate_run_metadata(
        "test_output",
        require_dependency_versions=False,
    ) is metadata


@pytest.mark.parametrize("recorded_hash", [None, "stale-source"])
def test_run_metadata_requires_current_source_code_hash(monkeypatch, recorded_hash):
    metadata = _run_metadata_fixture()
    if recorded_hash is None:
        metadata.pop("source_code_hash")
    else:
        metadata["source_code_hash"] = recorded_hash
    monkeypatch.setattr(simulation_common, "read_run_metadata", lambda stem: metadata)
    monkeypatch.setattr(simulation_common, "config_fingerprint", lambda: "current-config")
    monkeypatch.setattr(simulation_common, "source_code_fingerprint", lambda: "current-source")

    with pytest.raises(ValueError, match="source-code fingerprint|stale source code"):
        simulation_common.validate_run_metadata("test_output")


def test_source_code_fingerprint_is_stable_and_excludes_bytecode_cache(monkeypatch, tmp_path):
    model_dir = tmp_path / "src_python" / "model"
    cache_dir = model_dir / "__pycache__"
    calibration_dir = tmp_path / "src_python" / "calibration"
    simulation_dir = tmp_path / "src_python" / "simulation"
    model_dir.mkdir(parents=True)
    cache_dir.mkdir()
    calibration_dir.mkdir(parents=True)
    simulation_dir.mkdir(parents=True)
    source_path = model_dir / "example.py"
    calibration_path = calibration_dir / "fit.py"
    common_path = simulation_dir / "common.py"
    analysis_path = simulation_dir / "analysis.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    calibration_path.write_text("FIT = 1\n", encoding="utf-8")
    common_path.write_text("COMMON = 1\n", encoding="utf-8")
    analysis_path.write_text("ANALYSIS = 1\n", encoding="utf-8")
    bytecode_path = cache_dir / "example.py"
    bytecode_path.write_text("ignored = 1\n", encoding="utf-8")
    monkeypatch.setattr(simulation_common, "project_path", lambda *parts: tmp_path.joinpath(*parts))

    simulation_common.source_code_fingerprint.cache_clear()
    first = simulation_common.source_code_fingerprint()
    simulation_common.calibration_source_code_fingerprint.cache_clear()
    first_calibration = simulation_common.calibration_source_code_fingerprint()
    simulation_common.source_code_fingerprint.cache_clear()
    assert simulation_common.source_code_fingerprint() == first

    bytecode_path.write_text("ignored = 2\n", encoding="utf-8")
    simulation_common.source_code_fingerprint.cache_clear()
    assert simulation_common.source_code_fingerprint() == first

    calibration_path.write_text("FIT = 2\n", encoding="utf-8")
    simulation_common.source_code_fingerprint.cache_clear()
    assert simulation_common.source_code_fingerprint() != first
    simulation_common.calibration_source_code_fingerprint.cache_clear()
    assert simulation_common.calibration_source_code_fingerprint() != first_calibration

    calibration_path.write_text("FIT = 1\n", encoding="utf-8")
    analysis_path.write_text("ANALYSIS = 2\n", encoding="utf-8")
    simulation_common.calibration_source_code_fingerprint.cache_clear()
    assert simulation_common.calibration_source_code_fingerprint() == first_calibration

    common_path.write_text("COMMON = 2\n", encoding="utf-8")
    simulation_common.calibration_source_code_fingerprint.cache_clear()
    assert simulation_common.calibration_source_code_fingerprint() != first_calibration
    simulation_common.source_code_fingerprint.cache_clear()
    simulation_common.calibration_source_code_fingerprint.cache_clear()


def test_runtime_data_hash_ignores_non_model_provenance_text(monkeypatch, tmp_path):
    path = tmp_path / "country_resistance_timeline.csv"

    def write_timeline(*, source: str, notes: str, resistant_fraction: str = "0.02") -> None:
        path.write_text(
            "\n".join(
                [
                    "country,iso3,year,sample_size,resistant_fraction,lower,upper,evidence_type,source,notes",
                    f"Example,EXA,2025,,{resistant_fraction},0.00,0.05,anchor,{source},{notes}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(simulation_common, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    data_sources = {"country_resistance_timeline_csv": "country_resistance_timeline.csv"}

    write_timeline(source="https://example.org/source-a", notes="first note")
    first_hash = simulation_common._runtime_data_file_hashes(data_sources)["country_resistance_timeline_csv"]

    write_timeline(source="https://example.org/source-b", notes="reworded note")
    provenance_only_hash = simulation_common._runtime_data_file_hashes(data_sources)[
        "country_resistance_timeline_csv"
    ]
    assert provenance_only_hash == first_hash

    write_timeline(source="https://example.org/source-b", notes="reworded note", resistant_fraction="0.03")
    material_hash = simulation_common._runtime_data_file_hashes(data_sources)["country_resistance_timeline_csv"]
    assert material_hash != first_hash


def test_relative_reductions_fall_back_to_global_reference_when_needed():
    summary = pd.DataFrame(
        {
            "country": ["Australia", "China"],
            "scenario": ["Australia", "China"],
            "total_infant_cases": [10.0, 20.0],
            "total_infections": [100.0, 200.0],
            "total_reported_cases": [5.0, 10.0],
            "resistant_infections": [0.0, 0.0],
        }
    )

    reduced = add_relative_reductions(summary, reference_scenario="China")

    australia = reduced.loc[reduced["country"].eq("Australia")].iloc[0]
    china = reduced.loc[reduced["country"].eq("China")].iloc[0]
    assert np.isclose(australia["relative_reduction_total_infections"], 0.5)
    assert np.isclose(china["relative_reduction_total_infections"], 0.0)
    assert np.isnan(australia["relative_reduction_resistant_infections"])


def test_relative_reductions_group_by_country_when_reference_exists_per_country():
    summary = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "scenario": ["current", "improved", "current", "improved"],
            "total_infant_cases": [10.0, 5.0, 20.0, 5.0],
            "total_infections": [100.0, 50.0, 200.0, 50.0],
            "total_reported_cases": [8.0, 4.0, 16.0, 4.0],
            "resistant_infections": [2.0, 1.0, 4.0, 1.0],
        }
    )

    reduced = add_relative_reductions(summary, reference_scenario="current")

    improved_a = reduced.loc[reduced["country"].eq("A") & reduced["scenario"].eq("improved")].iloc[0]
    improved_b = reduced.loc[reduced["country"].eq("B") & reduced["scenario"].eq("improved")].iloc[0]
    assert np.isclose(improved_a["relative_reduction_total_infections"], 0.5)
    assert np.isclose(improved_b["relative_reduction_total_infections"], 0.75)


def test_uncertainty_and_fitness_runtime_blocks_are_available():
    configs = load_configs()
    bayesian = configs["baseline"]["bayesian_uncertainty"]
    fitness_grid = configs["baseline"]["fitness_grid"]

    assert bayesian["n_chains"] >= 4
    assert "VE_sus" in bayesian["priors"]
    assert "VE_inf" in bayesian["priors"]
    assert min(fitness_grid["fitness_R_values"]) <= 0.70
    assert max(fitness_grid["fitness_R_values"]) >= 1.25


def test_vaccine_uncertainty_config_distinguishes_infection_and_infectiousness_effects():
    bayesian = load_configs()["baseline"]["bayesian_uncertainty"]
    ve_sus_note = bayesian["priors"]["VE_sus"]["note"]
    ve_inf_note = bayesian["priors"]["VE_inf"]["note"]

    assert "susceptibility" in ve_sus_note
    assert "infectiousness" in ve_inf_note
    assert "infection-acquisition" not in ve_inf_note


def test_active_waning_durations_are_reported_with_evidence_distributions():
    table = pd.read_csv(project_path("manuscript_notes/parameter_table.csv")).set_index("parameter")
    assert not bool(table.loc["Post-infection protection duration", "used_in_sensitivity_analysis"])
    assert "inactive" in str(table.loc["Post-infection protection duration", "range"])
    assert bool(table.loc["Vaccine-derived protection duration", "used_in_sensitivity_analysis"])
    assert "half" in str(table.loc["Vaccine-derived protection duration", "range"])
    for parameter in [
        "SIRWS recovered-to-waned duration",
        "SIRWS waned-to-susceptible duration",
    ]:
        assert bool(table.loc[parameter, "used_in_sensitivity_analysis"])
        assert "half" in str(table.loc[parameter, "range"])


def test_manuscript_parameter_table_covers_core_values_and_ranges():
    table = pd.read_csv(project_path("manuscript_notes/parameter_table.csv"))
    assert len(table) >= 50
    assert not table["range"].astype(str).str.strip().eq("").any()
    assert not table["source_or_assumption"].astype(str).str.strip().eq("").any()

    required_parameters = {
        "Seasonal forcing amplitude",
        "SIRWS boosting efficiency",
        "Age-specific reporting probabilities",
        "Reporting multiplier",
        "Negative-binomial calibration dispersion",
        "Routine vaccination maximum daily flow",
        "Sensitive-strain treatment duration reduction",
        "Resistant-strain treatment infectiousness reduction",
        "PEP activation prevalence proxy",
    }
    assert required_parameters.issubset(set(table["parameter"]))


def test_supplementary_parameter_renderer_preserves_small_nonzero_values():
    from manuscript_notes.render_supplementary_tables import format_value

    assert format_value("1e-5", "baseline_value") == "1.00e-05"
    assert format_value("2e-5", "baseline_value") == "2.00e-05"
