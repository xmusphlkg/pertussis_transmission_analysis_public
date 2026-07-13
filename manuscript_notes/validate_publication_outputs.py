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
from src_python.simulation.common import load_configs, validate_run_metadata
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


def main() -> None:
    validate_population_conservation()
    validate_baseline_outputs()
    validate_release_output_windows()
    validate_active_publication_outputs()
    print("Active publication validation checks passed.")


if __name__ == "__main__":
    main()
