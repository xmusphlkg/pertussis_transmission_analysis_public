from __future__ import annotations

import numpy as np
import pandas as pd

from src_python.model.compartments import StateIndex
from src_python.model.outputs import initial_state, solve_model
from src_python.model.parameters import PreparedParameters
from src_python.simulation.common import load_configs, make_config, run_prepared_config, validate_run_metadata
from src_python.utils.io import project_path, read_table


EXPECTED_TIMESERIES_COLUMNS = {
    "time",
    "age_group",
    "strain",
    "scenario",
    "symptomatic_cases",
    "symptomatic_case_rate_per_day",
    "asymptomatic_infections",
    "asymptomatic_infection_rate_per_day",
    "total_infections",
    "total_infection_rate_per_day",
    "reported_cases",
    "reported_case_rate_per_day",
    "infant_cases",
    "infant_infections",
    "child_1_9_cases",
    "adolescent_cases",
    "child_adolescent_cases",
    "child_adolescent_infections",
    "child_adolescent_reported_cases",
    "resistant_fraction",
    "treated_cases",
    "PEP_FOI_reduction_case_proxy",
    "PEP_averted_cases",
    "infection_to_recovery_rate_ratio",
    "cumulative_cases",
    "cumulative_reported_cases",
    "cumulative_infections",
}

CORE_OUTPUT_STEMS = (
    "baseline_timeseries",
    "vaccine_scenarios",
    "resistance_scenarios",
    "reporting_scenarios",
    "country_scenarios",
    "veinf_resistance_grid",
    "fitness_resistance_grid",
    "intervention_scenarios",
    "routine_timeliness_sensitivity",
    "sensitivity_runs",
    "immunity_sensitivity",
    "resistance_fitness_sensitivity",
)
OPTIONAL_EXISTING_OUTPUT_STEMS = (
    "bayesian_uncertainty",
)
PUBLICATION_METADATA_STEMS = (
    "resistance_hindcast",
    "calibration_diagnostics",
    "age_pattern_sensitivity",
    "resistance_mechanism_decomposition",
    "program_portfolio_factorial",
    "infant_contact_sensitivity",
    "maternal_duration_sensitivity",
    "shock_recovery_sensitivity",
    "temporal_assumption_sensitivity",
    "treatment_implementation_sensitivity",
    "individual_stochastic_toy",
    "joint_psa_rank_acceptability",
    "fitness_resistance_grid_posterior_sample_diagnostics",
    "fitness_resistance_grid_posterior_benefit",
    "fitness_resistance_grid_psa_benefit",
    "health_utility_analysis",
    "vaccine_pipeline_mapping",
    "high_risk_review_tables",
    "lancet_child_adolescent_tables",
)
PUBLICATION_REQUIRED_TABLES = {
    "resistance_hindcast": ("outputs/tables/resistance_hindcast_results.csv",),
    "calibration_diagnostics": ("outputs/tables/calibration_interval_diagnostics.csv",),
    "age_pattern_sensitivity": ("outputs/tables/age_pattern_country_weights.csv",),
    "resistance_mechanism_decomposition": ("outputs/summaries/resistance_mechanism_decomposition_summary.csv",),
    "program_portfolio_factorial": ("outputs/summaries/program_portfolio_factorial_summary.csv",),
    "infant_contact_sensitivity": ("outputs/summaries/infant_contact_sensitivity_summary.csv",),
    "maternal_duration_sensitivity": ("outputs/summaries/maternal_duration_sensitivity_summary.csv",),
    "shock_recovery_sensitivity": ("outputs/tables/shock_recovery_sensitivity.csv",),
    "temporal_assumption_sensitivity": ("outputs/summaries/temporal_assumption_sensitivity_summary.csv",),
    "treatment_implementation_sensitivity": ("outputs/summaries/treatment_implementation_sensitivity_summary.csv",),
    "individual_stochastic_toy": ("outputs/tables/individual_stochastic_toy_summary.csv",),
    "joint_psa_rank_acceptability": ("outputs/summaries/joint_psa_rank_acceptability_summary.csv",),
    "fitness_resistance_grid_posterior_sample_diagnostics": (
        "outputs/summaries/fitness_resistance_grid_posterior_sample_diagnostics.csv",
    ),
    "fitness_resistance_grid_posterior_benefit": (
        "outputs/summaries/fitness_resistance_grid_posterior_benefit_summary.csv",
    ),
    "fitness_resistance_grid_psa_benefit": (
        "outputs/tables/fitness_resistance_grid_psa_benefit_summary.csv",
    ),
    "health_utility_analysis": ("outputs/tables/intervention_health_utility_loss_summary.csv",),
    "vaccine_pipeline_mapping": ("outputs/tables/vaccine_pipeline_mechanism_mapping.csv",),
    "high_risk_review_tables": ("outputs/tables/limitation_diagnostic_map.csv",),
    "lancet_child_adolescent_tables": ("outputs/tables/lancet_child_adolescent_strategy_summary.csv",),
}

def validate_timeseries(df: pd.DataFrame) -> None:
    missing = EXPECTED_TIMESERIES_COLUMNS.difference(df.columns)
    if missing:
        raise AssertionError(f"Missing expected output columns: {sorted(missing)}")
    numeric = df.select_dtypes(include=["number"])
    if not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError("Timeseries contains NaN or infinite values.")

    nonnegative_cols = [
        "symptomatic_cases",
        "symptomatic_case_rate_per_day",
        "asymptomatic_infections",
        "asymptomatic_infection_rate_per_day",
        "total_infections",
        "total_infection_rate_per_day",
        "reported_cases",
        "reported_case_rate_per_day",
        "infant_cases",
        "infant_infections",
        "child_1_9_cases",
        "adolescent_cases",
        "child_adolescent_cases",
        "child_adolescent_infections",
        "child_adolescent_reported_cases",
        "treated_cases",
        "PEP_FOI_reduction_case_proxy",
        "PEP_averted_cases",
        "cumulative_cases",
        "cumulative_infections",
    ]
    min_value = float(df[nonnegative_cols].min().min())
    if min_value < -1e-8:
        raise AssertionError(f"Negative model output detected: {min_value}")
    if (df["reported_cases"] - df["symptomatic_cases"] > 1e-8).any():
        raise AssertionError("Reported cases exceed symptomatic infections.")
    if (df["reported_case_rate_per_day"] - df["symptomatic_case_rate_per_day"] > 1e-8).any():
        raise AssertionError("Reported case rates exceed symptomatic infection rates.")
    first_time = float(df["time"].min())
    initial_rows = df.loc[np.isclose(df["time"].to_numpy(dtype=float), first_time)]
    for column in ["cumulative_cases", "cumulative_reported_cases", "cumulative_infections"]:
        if initial_rows[column].abs().max() > 1e-8:
            raise AssertionError(f"{column} must start at zero.")
    if not df["resistant_fraction"].between(-1e-8, 1.0 + 1e-8).all():
        raise AssertionError("Resistant fraction is outside [0, 1].")


def validate_population_conservation() -> None:
    resistance_name = load_configs()["baseline"].get("baseline_resistance_scenario", "country_timeline")
    config = make_config(vaccine_scenario="symptom_protective", resistance_scenario=resistance_name)
    # Population conservation is only meaningful when the WPP trajectory is
    # disabled. The dynamic WPP branch has its own regression test in the
    # model test suite.
    config.setdefault("demography", {})["mode"] = "fixed_population_profile"
    params = PreparedParameters.from_config(
        config,
        analysis="validation",
        scenario="population_conservation",
        vaccine_scenario="symptom_protective",
        resistance_scenario=resistance_name,
    )
    index = StateIndex(params.age_groups)
    y0 = initial_state(params, index)
    solution = solve_model(params, index)
    if not solution.success:
        raise AssertionError(solution.message)
    totals = solution.y.reshape(index.n_age, index.n_compartments, -1).sum(axis=(0, 1))
    expected = float(index.reshape(y0).sum())
    max_abs_error = float(np.max(np.abs(totals - expected)))
    tolerance = max(1e-4, expected * 1e-8)
    if max_abs_error > tolerance:
        raise AssertionError(f"Population is not conserved; max error={max_abs_error}")
    if solution.y.min() < -1e-5:
        raise AssertionError(f"Negative compartment value detected: {solution.y.min()}")


def validate_baseline_outputs() -> None:
    resistance_name = load_configs()["baseline"].get("baseline_resistance_scenario", "country_timeline")
    path = project_path("outputs/simulations/baseline_timeseries.parquet")
    if path.exists() or path.with_suffix(".csv").exists():
        validate_run_metadata("baseline_timeseries")
        df = read_table(path)
    else:
        df, _ = run_prepared_config(
            make_config(vaccine_scenario="symptom_protective", resistance_scenario=resistance_name),
            analysis="baseline",
            scenario="baseline",
            vaccine_scenario="symptom_protective",
            resistance_scenario=resistance_name,
        )
    validate_timeseries(df)


def _validate_summary_window(stem: str) -> None:
    configs = load_configs()
    baseline = configs["baseline"]
    expected_start = str(baseline.get("calendar", {}).get("analysis_start_date", ""))
    expected_years = float(baseline["simulation"]["end_time"] - baseline["simulation"]["start_time"]) / 365.0
    summary_path = project_path("outputs", "summaries", f"{stem}_summary.csv")
    validate_run_metadata(stem)
    summary = read_table(summary_path)
    if summary.empty:
        raise AssertionError(f"{stem} summary is empty.")
    if "calendar_start_date" not in summary.columns:
        raise AssertionError(f"{stem} summary is missing calendar_start_date.")
    starts = set(summary["calendar_start_date"].astype(str))
    if starts != {expected_start}:
        raise AssertionError(
            f"{stem} uses calendar_start_date values {sorted(starts)}, expected {expected_start}."
        )
    years = pd.to_numeric(summary["analysis_years"], errors="coerce")
    if years.isna().any() or not np.allclose(years.to_numpy(dtype=float), expected_years, rtol=0.0, atol=1e-6):
        observed = sorted(set(float(value) for value in years.dropna().round(6)))
        raise AssertionError(f"{stem} analysis_years values {observed}, expected {expected_years:.6f}.")


def validate_main_output_windows() -> None:
    for stem in CORE_OUTPUT_STEMS:
        summary_path = project_path("outputs", "summaries", f"{stem}_summary.csv")
        if not summary_path.exists():
            raise AssertionError(f"Missing required core summary output: {summary_path}")
        _validate_summary_window(stem)
    for stem in OPTIONAL_EXISTING_OUTPUT_STEMS:
        summary_path = project_path("outputs", "summaries", f"{stem}_summary.csv")
        if summary_path.exists():
            _validate_summary_window(stem)


def validate_publication_outputs() -> None:
    for stem in PUBLICATION_METADATA_STEMS:
        validate_run_metadata(stem)
        required_tables = PUBLICATION_REQUIRED_TABLES.get(stem, ())
        for relative_path in required_tables:
            path = project_path(relative_path)
            if not path.exists() and not path.with_suffix(".parquet").exists():
                raise AssertionError(f"Missing required publication table for {stem}: {path}")
            table = read_table(path)
            if table.empty:
                raise AssertionError(f"Required publication table is empty for {stem}: {path}")


def main() -> None:
    validate_population_conservation()
    validate_baseline_outputs()
    validate_main_output_windows()
    validate_publication_outputs()
    print("Validation checks passed.")


if __name__ == "__main__":
    main()
