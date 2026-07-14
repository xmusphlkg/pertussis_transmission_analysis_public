"""Validate the active point-estimate/POMP publication output contract."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuscript_notes.release_gates import require_current_block_stress_result
from src_python.model.outputs import GREGORIAN_YEAR_DAYS
from src_python.simulation.common import (
    load_configs,
    publication_country_names,
    validate_run_metadata,
)
from src_python.simulation.run_joint_psa_rank_acceptability import PROGRAMME_ONLY_STRATEGIES
from src_python.utils.io import project_path, read_table
from src_python.utils.validation import (
    CORE_OUTPUT_STEMS,
    PUBLICATION_REQUIRED_TABLES,
    validate_baseline_outputs,
    validate_population_conservation,
)
from src_python.validation.publication_gate import require_predictive_publication_gate


ACTIVE_PUBLICATION_METADATA_STEMS = (
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
    "fitness_resistance_grid_psa_benefit",
    "health_utility_analysis",
    "vaccine_pipeline_mapping",
    "high_risk_review_tables",
    "lancet_child_adolescent_tables",
)


def _validate_release_summary_window(stem: str) -> None:
    baseline = load_configs()["baseline"]
    expected_start = str(baseline.get("calendar", {}).get("analysis_start_date", ""))
    expected_years = (
        float(baseline["simulation"]["end_time"] - baseline["simulation"]["start_time"])
        / GREGORIAN_YEAR_DAYS
    )
    validate_run_metadata(stem)
    summary = read_table(project_path("outputs", "summaries", f"{stem}_summary.csv"))
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
    if years.isna().any() or not np.allclose(
        years.to_numpy(dtype=float), expected_years, rtol=0.0, atol=1e-6
    ):
        observed = sorted(set(float(value) for value in years.dropna().round(6)))
        raise AssertionError(
            f"{stem} analysis_years values {observed}, expected {expected_years:.6f}."
        )


def validate_release_output_windows() -> None:
    for stem in CORE_OUTPUT_STEMS:
        summary_path = project_path("outputs", "summaries", f"{stem}_summary.csv")
        if not summary_path.exists():
            raise AssertionError(f"Missing required core summary output: {summary_path}")
        _validate_release_summary_window(stem)


def validate_active_publication_outputs() -> None:
    require_predictive_publication_gate()
    require_current_block_stress_result()
    for stem in ACTIVE_PUBLICATION_METADATA_STEMS:
        validate_run_metadata(stem)
        for relative_path in PUBLICATION_REQUIRED_TABLES.get(stem, ()):
            path = project_path(relative_path)
            if not path.exists() and not path.with_suffix(".parquet").exists():
                raise AssertionError(f"Missing required publication table for {stem}: {path}")
            if read_table(path).empty:
                raise AssertionError(f"Required publication table is empty for {stem}: {path}")


def validate_under18_programme_psa() -> None:
    """Require a complete selected-parameter sensitivity for the primary endpoint."""

    configs = load_configs()
    countries = tuple(publication_country_names(configs))
    strategies = tuple(PROGRAMME_ONLY_STRATEGIES)
    samples = read_table(
        project_path("outputs", "tables", "joint_psa_under18_programme_rank_samples.csv")
    )
    expected_rows = 128 * len(countries) * len(strategies)
    if len(samples) != expected_rows:
        raise AssertionError(
            f"Primary-endpoint joint PSA has {len(samples)} rows, expected {expected_rows}."
        )
    if set(samples["country"].astype(str)) != set(countries):
        raise AssertionError("Primary-endpoint joint PSA country set does not match publication profiles.")
    if set(samples["strategy"].astype(str)) != set(strategies):
        raise AssertionError("Primary-endpoint joint PSA programme strategy set is incomplete.")
    if samples["psa_sample_id"].nunique() != 128:
        raise AssertionError("Primary-endpoint joint PSA does not contain 128 unique paired samples.")
    if samples.duplicated(["psa_sample_id", "country", "strategy"]).any():
        raise AssertionError("Primary-endpoint joint PSA contains duplicate sample-country-strategy cells.")
    group_sizes = samples.groupby(["psa_sample_id", "country"], dropna=False).size()
    if not group_sizes.eq(len(strategies)).all():
        raise AssertionError("Primary-endpoint joint PSA has incomplete within-sample strategy comparisons.")
    ranks = pd.to_numeric(samples["rank"], errors="coerce")
    if ranks.isna().any() or not ranks.between(1, len(strategies)).all():
        raise AssertionError("Primary-endpoint joint PSA ranks are missing or outside the programme set.")

    summary = read_table(
        project_path(
            "outputs",
            "summaries",
            "joint_psa_under18_programme_rank_acceptability_summary.csv",
        )
    )
    expected_summary_countries = set(countries) | {"All_countries_pooled"}
    if set(summary["country"].astype(str)) != expected_summary_countries:
        raise AssertionError("Primary-endpoint rank-frequency summary has the wrong country set.")
    if set(summary["strategy"].astype(str)) != set(strategies):
        raise AssertionError("Primary-endpoint rank-frequency summary has the wrong strategy set.")
    if not pd.to_numeric(summary["n_psa_samples"], errors="coerce").eq(128).all():
        raise AssertionError("Primary-endpoint rank-frequency summary is not based on 128 samples.")
    frequency = pd.to_numeric(summary["frequency_rank_1"], errors="coerce")
    if frequency.isna().any() or not frequency.between(0.0, 1.0).all():
        raise AssertionError("Primary-endpoint rank-one frequencies are invalid.")
    sums = summary.assign(_frequency=frequency).groupby("country")["_frequency"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-10):
        raise AssertionError("Primary-endpoint rank-one frequencies do not sum to one.")
    if not summary["interpretation"].astype(str).str.contains("not a posterior", case=False).all():
        raise AssertionError("Primary-endpoint sensitivity is missing its non-posterior interpretation guardrail.")


def main() -> None:
    validate_population_conservation()
    validate_baseline_outputs()
    validate_release_output_windows()
    validate_active_publication_outputs()
    validate_under18_programme_psa()
    print("Active publication validation checks passed.")


if __name__ == "__main__":
    main()
