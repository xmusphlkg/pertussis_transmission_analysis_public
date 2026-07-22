from __future__ import annotations

"""Audit the optional legacy paired-CrI research route.

The historical module and artifact filenames are retained for compatibility.
Neither this audit nor the artifacts it checks are a publication path or a
Figure 2c interval source. Canonical Figure 2c uncertainty is the separate
full-refit parametric-bootstrap estimation CI.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.calibration.mcmc_diagnostics import (
    MAX_FATAL_RHAT,
    MAX_RECOMMENDED_RHAT,
    MIN_FATAL_BULK_ESS,
    MIN_FATAL_TAIL_ESS,
    MIN_RECOMMENDED_BULK_ESS,
    MIN_RECOMMENDED_TAIL_ESS,
)
from src_python.simulation.common import (
    config_fingerprint,
    current_run_metadata,
    file_sha256,
    load_configs,
    publication_country_names,
    source_code_fingerprint,
    validate_run_metadata,
    write_run_metadata,
)
from src_python.simulation.run_bayesian_uncertainty import (
    MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT,
    MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT,
    MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS,
    MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS,
    MODULAR_CUT_RECOMMENDED_MIN_STRUCTURAL_DRAWS,
    STATE_LAPLACE_FATAL_MAX_BOUNDARY_REJECTION,
    STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION,
    STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT,
    STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K,
    STATE_IMPORTANCE_RECOMMENDED_MIN_ESS,
    STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS,
    _artifact_stem,
    _importance_pareto_shape,
    _minimum_importance_tail_ess,
)
from src_python.simulation.run_hierarchical_joint_posterior import (
    POSTERIOR_RELEVANT_MASS,
    RECOMMENDED_MAX_FAILED_LOCAL_POSTERIOR_MASS,
    RECOMMENDED_MAX_GLOBAL_PARETO_K,
    RECOMMENDED_MAX_GLOBAL_WEIGHT,
    RECOMMENDED_MAX_LOCAL_PARETO_K,
    RECOMMENDED_MAX_LOCAL_WEIGHT,
    RECOMMENDED_MIN_GLOBAL_ESS,
    RECOMMENDED_MIN_GLOBAL_TAIL_ESS,
    RECOMMENDED_MIN_LOCAL_ESS,
    RECOMMENDED_MIN_LOCAL_TAIL_ESS,
    RECOMMENDED_MIN_STATE_IN_BOUNDS_FRACTION,
    SHARED_PARAMETER_NAMES,
)
from src_python.simulation.run_hierarchical_joint_smc import (
    INFERENCE_STRUCTURE as JOINT_INFERENCE_STRUCTURE,
    SAMPLING_METHOD as JOINT_SAMPLING_METHOD,
    UNCERTAINTY_SCOPE as JOINT_UNCERTAINTY_SCOPE,
)
from src_python.simulation.run_figure2c_paired_uncertainty import (
    FIGURE2C_STRATEGIES,
    INTERVENTION_UNCERTAINTY_DEFAULTS,
    INTERVENTION_UNCERTAINTY_PREFIX,
    JOINT_POSTERIOR_PARAMETERS,
    MODULAR_INFERENCE_STRUCTURES,
    PRIMARY_REDUCTION,
    RETIRED_POSTERIOR_STEMS,
    EXTERNAL_STRUCTURAL_PRIOR_PARAMETERS,
)
from src_python.utils.io import project_path, write_dataframe


# The research posterior stem is explicit; the two following stems retain
# historical filenames only so old local automation can still find its files.
POSTERIOR_STEM = "bayesian_uncertainty_joint_research"
FIGURE2C_STEM = "figure2c_paired_programme_uncertainty"
AUDIT_STEM = "figure2c_joint_credible_interval_quality_audit"
ANALYSIS_ROLE = "optional_nonpublication_legacy_research_audit"
PUBLICATION_PATH = False
FIGURE2C_INTERVAL_SOURCE = False

POSTERIOR_SAMPLE_PATH = project_path("outputs", "simulations", f"{POSTERIOR_STEM}_posterior_samples.parquet")
STRUCTURAL_IMPORTANCE_PATH = project_path(
    "outputs", "diagnostics", f"{POSTERIOR_STEM}_structural_importance_audit.csv"
)
LOCAL_IMPORTANCE_PATH = project_path(
    "outputs", "diagnostics", f"{POSTERIOR_STEM}_local_state_importance_audit.csv"
)
SMC_STAGE_PATH = project_path(
    "outputs", "diagnostics", f"{POSTERIOR_STEM}_smc_stage_audit.csv"
)
SMC_DIAGNOSTICS_PATH = project_path(
    "outputs", "summaries", f"{POSTERIOR_STEM}_convergence_diagnostics.csv"
)
SMC_BOUNDARY_PATH = project_path(
    "outputs", "diagnostics", f"{POSTERIOR_STEM}_smc_boundary_audit.csv"
)
INTERVAL_PATH = project_path("outputs", "summaries", "figure2c_programme_paired_credible_intervals.csv")
DRAW_PATH = project_path("outputs", "tables", "figure2c_programme_paired_credible_interval_draws.csv")
OUTPUT_PATH = project_path("outputs", "tables", f"{AUDIT_STEM}.csv")

REQUIRED_INTERVENTION_UNCERTAINTY_COLUMNS = (
    "adolescent_coverage_floor",
    "maternal_coverage_floor",
    "young_adult_coverage_floor",
    "contact_reduction_fraction",
    "targeted_pep_coverage",
    "maternal_protection_duration_days",
    "maternal_VE_sus",
    "maternal_VE_sym",
)

POSTERIOR_ABSOLUTE_WIDTH_FLOORS = {
    "VE_sus": 0.005,
    "VE_inf": 0.005,
    "VE_dur": 0.005,
    "relative_infectiousness_asymptomatic": 0.005,
    "infectious_duration_symptomatic": 0.25,
    "infectious_duration_asymptomatic": 0.25,
    "fitness_R": 0.005,
}

POSTERIOR_RELATIVE_WIDTH_FLOORS = {
    "beta_S": 0.002,
    "reporting_multiplier": 0.002,
}

FULL_JOINT_SCOPES = {
    "multi_parameter_mcmc",
    "multi_parameter_joint_posterior",
    "multi_parameter_tempered_smc_posterior",
    JOINT_UNCERTAINTY_SCOPE,
}
FULL_JOINT_SAMPLERS = {
    "adaptive_mh", "componentwise_mh", "slice", "joint_importance", "smc",
    JOINT_SAMPLING_METHOD,
}
CONDITIONAL_STATE_SPACE_SCOPE = "conditional_state_space_exact_importance_modular_sensitivity"
CONDITIONAL_STATE_SPACE_SAMPLER = "state_space_exact_importance_cut"
SMC_RECOMMENDED_ESS_FRACTION = 0.50
SMC_RECOMMENDED_MAX_WEIGHT = 0.02
SMC_RECOMMENDED_UNIQUE_PARTICLE_FRACTION = 0.25
SMC_RECOMMENDED_ISLAND_ESS_FRACTION = 0.35
SMC_RECOMMENDED_MAX_ISLAND_WEIGHT = 0.50

# The fatal thresholds are a last-resort validity floor. This optional legacy
# research route is audited with --fail-on-warnings, so the recommended
# thresholds imported above are required before interpreting its research
# outputs. Passing does not make the route publication eligible.


def _status(condition: bool) -> str:
    return "pass" if condition else "fail"


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _add(
    rows: list[dict[str, Any]],
    *,
    category: str,
    check: str,
    passed: bool,
    severity: str,
    details: str,
) -> None:
    rows.append(
        {
            "category": category,
            "check": check,
            "status": _status(passed),
            "severity": severity,
            "details": details,
        }
    )


def _posterior_metadata_checks(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    sampler = str(metadata.get("sampler", "")).lower()
    scope = str(metadata.get("uncertainty_scope", "")).lower()
    fixed_parameters = list(metadata.get("fixed_parameters") or [])
    fix_durations = bool(metadata.get("fix_durations", False))
    convergence_summary = metadata.get("convergence_summary") or {}
    recognized_target = bool(
        (sampler == JOINT_SAMPLING_METHOD and scope == JOINT_UNCERTAINTY_SCOPE)
        or (sampler == CONDITIONAL_STATE_SPACE_SAMPLER and scope == CONDITIONAL_STATE_SPACE_SCOPE)
        or (sampler != "beta_grid" and scope in FULL_JOINT_SCOPES)
    )
    _add(
        rows,
        category="posterior_metadata",
        check="recognized_multi_parameter_uncertainty_scope",
        passed=recognized_target,
        severity="fatal",
        details=f"sampler={sampler or 'missing'}, uncertainty_scope={scope or 'missing'}",
    )
    _add(
        rows,
        category="posterior_metadata",
        check="no_fixed_joint_parameters",
        passed=not fixed_parameters and not fix_durations,
        severity="fatal",
        details=f"fixed_parameters={fixed_parameters}, fix_durations={fix_durations}",
    )
    _add(
        rows,
        category="posterior_metadata",
        check="convergence_summary_all_converged",
        passed=bool(convergence_summary.get("all_converged", False)),
        severity="fatal",
        details=(
            f"all_converged={convergence_summary.get('all_converged')}, "
            f"n={convergence_summary.get('n_parameters_converged')}/"
            f"{convergence_summary.get('n_parameters_total')}, "
            f"worst_rhat={convergence_summary.get('worst_rhat')}, "
            f"min_bulk_ess={convergence_summary.get('min_bulk_ess')}, "
            f"min_tail_ess={convergence_summary.get('min_tail_ess')}"
        ),
    )


def _posterior_runtime_checks(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    expected_chains: int,
) -> None:
    sampler = str(metadata.get("sampler", "")).lower()
    if sampler == JOINT_SAMPLING_METHOD:
        observed_batches = int(metadata.get("n_chains") or -1)
        observed_draws = int(metadata.get("draws_per_chain") or -1)
        _add(
            rows,
            category="posterior_runtime",
            check="sampler_matches_recognized_uncertainty_target",
            passed=True,
            severity="fatal",
            details=f"sampler={sampler}",
        )
        _add(
            rows,
            category="posterior_runtime",
            check="expected_chain_count_in_metadata",
            passed=observed_batches == int(expected_chains),
            severity="fatal",
            details=f"smc_islands={observed_batches}, expected={expected_chains}",
        )
        _add(
            rows,
            category="posterior_runtime",
            check="positive_draw_warmup_thin_metadata",
            passed=observed_draws > 0
            and metadata.get("chain_labels_are_smc_islands") is True,
            severity="fatal",
            details=(
                f"particles_per_island={observed_draws}, "
                "chain_labels_are_smc_islands="
                f"{metadata.get('chain_labels_are_smc_islands')}"
            ),
        )
        _add(
            rows,
            category="posterior_runtime",
            check="joint_inference_structure_recorded",
            passed=str(metadata.get("inference_structure", "")).lower()
            == JOINT_INFERENCE_STRUCTURE,
            severity="fatal",
            details=f"inference_structure={metadata.get('inference_structure')}",
        )
        posterior_hash = str(metadata.get("config_hash", "") or "missing")
        current_hash = str(config_fingerprint())
        _add(
            rows,
            category="posterior_runtime",
            check="config_hash_recorded_for_traceability",
            passed=posterior_hash != "missing",
            severity="fatal",
            details=f"posterior_config_hash={posterior_hash}, current_config_hash={current_hash}",
        )
        _add(
            rows,
            category="posterior_runtime",
            check="current_config_hash_comparison",
            passed=posterior_hash == current_hash,
            severity="fatal",
            details=f"posterior_config_hash={posterior_hash}, current_config_hash={current_hash}",
        )
        posterior_source_hash = str(metadata.get("source_code_hash", "") or "missing")
        current_source_hash = str(source_code_fingerprint())
        _add(
            rows,
            category="posterior_runtime",
            check="current_source_code_hash_comparison",
            passed=posterior_source_hash == current_source_hash,
            severity="fatal",
            details=(
                f"posterior_source_code_hash={posterior_source_hash}, "
                f"current_source_code_hash={current_source_hash}"
            ),
        )
        return
    parameterization = str(metadata.get("parameterization", "")).lower()
    observed_chains = int(metadata.get("n_chains") or -1)
    observed_draws = int(metadata.get("draws_per_chain") or -1)
    observed_thin = int(metadata.get("thin") or -1)
    observed_warmup = int(metadata.get("warmup") or -1)
    _add(
        rows,
        category="posterior_runtime",
        check="sampler_matches_recognized_uncertainty_target",
        passed=(
            sampler in FULL_JOINT_SAMPLERS
            or sampler == CONDITIONAL_STATE_SPACE_SAMPLER
        ),
        severity="fatal",
        details=f"sampler={sampler or 'missing'}",
    )
    _add(
        rows,
        category="posterior_runtime",
        check="parameterization_beta_reporting_product",
        passed=parameterization == "beta_reporting_product",
        severity="fatal",
        details=f"parameterization={parameterization or 'missing'}",
    )
    _add(
        rows,
        category="posterior_runtime",
        check="expected_chain_count_in_metadata",
        passed=observed_chains == int(expected_chains),
        severity="fatal",
        details=f"n_chains={observed_chains}, expected={expected_chains}",
    )
    _add(
        rows,
        category="posterior_runtime",
        check="positive_draw_warmup_thin_metadata",
        passed=observed_draws > 0 and observed_warmup >= 0 and observed_thin > 0,
        severity="fatal",
        details=f"draws_per_chain={observed_draws}, warmup={observed_warmup}, thin={observed_thin}",
    )
    posterior_hash = str(metadata.get("config_hash", "") or "missing")
    current_hash = str(config_fingerprint())
    _add(
        rows,
        category="posterior_runtime",
        check="config_hash_recorded_for_traceability",
        passed=bool(posterior_hash and posterior_hash != "missing"),
        severity="fatal",
        details=f"posterior_config_hash={posterior_hash}, current_config_hash={current_hash}",
    )
    _add(
        rows,
        category="posterior_runtime",
        check="current_config_hash_comparison",
        passed=posterior_hash == current_hash,
        severity="fatal",
        details=f"posterior_config_hash={posterior_hash}, current_config_hash={current_hash}",
    )
    posterior_source_hash = str(metadata.get("source_code_hash", "") or "missing")
    current_source_hash = str(source_code_fingerprint())
    _add(
        rows,
        category="posterior_runtime",
        check="current_source_code_hash_comparison",
        passed=posterior_source_hash == current_source_hash,
        severity="fatal",
        details=(
            f"posterior_source_code_hash={posterior_source_hash}, "
            f"current_source_code_hash={current_source_hash}"
        ),
    )


def _figure_metadata_checks(
    rows: list[dict[str, Any]],
    figure_metadata: dict[str, Any],
    *,
    configured_countries: set[str],
    expected_draws: int,
    expected_inference_structure: str | None = None,
    expected_posterior_stem: str = POSTERIOR_STEM,
    expected_posterior_path: Path = POSTERIOR_SAMPLE_PATH,
) -> None:
    countries = set(str(country) for country in figure_metadata.get("countries", []))
    strategies = set(str(strategy) for strategy in figure_metadata.get("strategies", []))
    expected_strategies = set(FIGURE2C_STRATEGIES)
    observed_stem = str(figure_metadata.get("posterior_sample_stem", ""))
    observed_path = str(figure_metadata.get("posterior_samples_path", ""))
    expected_path = Path(expected_posterior_path)
    if observed_path:
        observed_path_obj = Path(observed_path)
        if not observed_path_obj.is_absolute():
            observed_path_obj = project_path(observed_path)
        paths_match = observed_path_obj.resolve(strict=False) == expected_path.resolve(strict=False)
    else:
        paths_match = False
    _add(
        rows,
        category="figure2c_metadata",
        check="posterior_source_stem_matches_joint_release",
        passed=observed_stem == expected_posterior_stem,
        severity="fatal",
        details=(
            f"posterior_sample_stem={observed_stem or 'missing'}, "
            f"expected={expected_posterior_stem}"
        ),
    )
    _add(
        rows,
        category="figure2c_metadata",
        check="posterior_source_path_matches_joint_release",
        passed=paths_match,
        severity="fatal",
        details=f"posterior_samples_path={observed_path or 'missing'}, expected={expected_path}",
    )
    _add(
        rows,
        category="figure2c_metadata",
        check="all_configured_countries_included",
        passed=countries == configured_countries,
        severity="fatal",
        details=f"countries={sorted(countries)}, expected={sorted(configured_countries)}",
    )
    _add(
        rows,
        category="figure2c_metadata",
        check="strategy_set_matches_figure2c",
        passed=strategies == expected_strategies,
        severity="fatal",
        details=f"strategies={sorted(strategies)}, expected={sorted(expected_strategies)}",
    )
    observed_draws = figure_metadata.get("uncertainty_draws_per_country")
    _add(
        rows,
        category="figure2c_metadata",
        check="uncertainty_draws_per_country_matches_expected",
        passed=int(observed_draws or -1) == int(expected_draws),
        severity="fatal",
        details=f"uncertainty_draws_per_country={observed_draws}, expected={expected_draws}",
    )
    expected_structure = str(expected_inference_structure or "").strip().lower()
    if expected_structure == JOINT_INFERENCE_STRUCTURE:
        observed_structure = str(
            figure_metadata.get("inference_structure", "")
        ).strip().lower()
        paired = figure_metadata.get("structural_draw_pairing") is True
        basis = str(figure_metadata.get("interval_basis", "")).lower()
        _add(
            rows,
            category="figure2c_metadata",
            check="joint_inference_structure_retained",
            passed=observed_structure == JOINT_INFERENCE_STRUCTURE,
            severity="fatal",
            details=f"observed={observed_structure or 'missing'}, expected={JOINT_INFERENCE_STRUCTURE}",
        )
        _add(
            rows,
            category="figure2c_metadata",
            check="structural_draw_pairing_declared",
            passed=paired and "joint_posterior" in basis,
            severity="fatal",
            details=(
                f"structural_draw_pairing={figure_metadata.get('structural_draw_pairing')}, "
                f"interval_basis={basis or 'missing'}"
            ),
        )
    elif expected_structure in MODULAR_INFERENCE_STRUCTURES:
        observed_structure = str(figure_metadata.get("inference_structure", "")).strip().lower()
        paired = figure_metadata.get("structural_draw_pairing") is True
        selection = str(
            figure_metadata.get("uncertainty_sample_selection", "")
        ).lower()
        _add(
            rows,
            category="figure2c_metadata",
            check="modular_inference_structure_retained",
            passed=observed_structure == expected_structure,
            severity="fatal",
            details=f"observed={observed_structure or 'missing'}, expected={expected_structure}",
        )
        _add(
            rows,
            category="figure2c_metadata",
            check="structural_draw_pairing_declared",
            passed=(
                paired
                and "structural_draw" in selection
                and "state_importance_resample" in selection
            ),
            severity="fatal",
            details=(
                f"structural_draw_pairing={figure_metadata.get('structural_draw_pairing')}, "
                f"uncertainty_sample_selection={selection or 'missing'}"
            ),
        )


def _posterior_structural_pairing_checks(
    rows: list[dict[str, Any]],
    samples: pd.DataFrame,
    *,
    expected_countries: set[str],
    required: bool,
) -> None:
    present = "structural_draw_id" in samples.columns
    _add(
        rows,
        category="posterior_samples",
        check="structural_draw_id_present",
        passed=present or not required,
        severity="fatal",
        details=f"required={required}, present={present}",
    )
    if not present:
        return

    required_keys = {"country", "chain", "draw"}
    missing_keys = sorted(required_keys.difference(samples.columns))
    identifiers = pd.to_numeric(samples["structural_draw_id"], errors="coerce")
    finite_integer = bool(
        identifiers.notna().all()
        and np.isfinite(identifiers.to_numpy(dtype=float)).all()
        and np.equal(identifiers, np.floor(identifiers)).all()
    )
    _add(
        rows,
        category="posterior_samples",
        check="structural_draw_id_finite_integer",
        passed=finite_integer,
        severity="fatal",
        details=f"invalid_rows={int((~identifiers.notna()).sum())}",
    )
    if missing_keys or not finite_integer:
        _add(
            rows,
            category="posterior_samples",
            check="structural_source_positions_paired_across_countries",
            passed=False,
            severity="fatal",
            details=f"missing_source_keys={missing_keys}, valid_structural_ids={finite_integer}",
        )
        return

    data = samples.copy()
    data["country"] = data["country"].astype(str)
    data["structural_draw_id"] = identifiers.astype(np.int64)
    duplicated = data.duplicated(["country", "chain", "draw"], keep=False)
    alignment = data.groupby(["chain", "draw"], dropna=False).agg(
        rows=("country", "size"),
        countries=("country", "nunique"),
        structural_ids=("structural_draw_id", "nunique"),
    )
    aligned = (
        not duplicated.any()
        and alignment["rows"].eq(len(expected_countries)).all()
        and alignment["countries"].eq(len(expected_countries)).all()
        and alignment["structural_ids"].eq(1).all()
    )
    _add(
        rows,
        category="posterior_samples",
        check="structural_source_positions_paired_across_countries",
        passed=bool(aligned),
        severity="fatal",
        details=(
            f"duplicate_country_source_rows={int(duplicated.sum())}, "
            f"misaligned_source_positions={int((~(alignment['rows'].eq(len(expected_countries)) & alignment['countries'].eq(len(expected_countries)) & alignment['structural_ids'].eq(1))).sum())}"
        ),
    )

    inconsistent: list[str] = []
    for parameter in EXTERNAL_STRUCTURAL_PRIOR_PARAMETERS:
        if parameter not in data.columns:
            inconsistent.append(f"{parameter}:missing")
            continue
        values = pd.to_numeric(data[parameter], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            inconsistent.append(f"{parameter}:nonfinite")
            continue
        grouped = pd.DataFrame(
            {"structural_draw_id": data["structural_draw_id"], "value": values}
        ).groupby("structural_draw_id")["value"]
        minima = grouped.min()
        maxima = grouped.max()
        tolerance = 1e-10 + 1e-10 * np.maximum(minima.abs(), maxima.abs())
        if ((maxima - minima) > tolerance).any():
            inconsistent.append(f"{parameter}:inconsistent")
    _add(
        rows,
        category="posterior_samples",
        check="structural_parameters_shared_by_draw_id",
        passed=not inconsistent,
        severity="fatal",
        details="all shared" if not inconsistent else ", ".join(inconsistent),
    )


def _posterior_sample_checks(
    rows: list[dict[str, Any]],
    samples: pd.DataFrame,
    *,
    expected_countries: set[str],
    expected_chains: int,
    expected_draws_per_chain: int,
    require_structural_pairing: bool = False,
    require_annual_process_draws: bool = False,
) -> pd.DataFrame:
    countries = set(samples.get("country", pd.Series(dtype=str)).astype(str))
    _add(
        rows,
        category="posterior_samples",
        check="country_coverage",
        passed=countries == expected_countries,
        severity="fatal",
        details=f"observed={sorted(countries)}, expected={sorted(expected_countries)}",
    )
    _posterior_structural_pairing_checks(
        rows,
        samples,
        expected_countries=expected_countries,
        required=require_structural_pairing,
    )
    if require_annual_process_draws:
        process_columns = sorted(
            column
            for column in samples.columns
            if str(column).startswith("log_beta_process_")
        )
        parsed_years: list[int] = []
        invalid_names: list[str] = []
        for column in process_columns:
            try:
                parsed_years.append(int(str(column).removeprefix("log_beta_process_")))
            except ValueError:
                invalid_names.append(str(column))
        configs = load_configs()
        forecast_start = int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_start_date"]).year
        )
        forecast_end = int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_end_date"]).year
        )
        required_years = set(range(forecast_start, forecast_end + 1))
        union_complete = bool(
            process_columns
            and not invalid_names
            and required_years.issubset(parsed_years)
            and parsed_years == list(range(min(parsed_years), max(parsed_years) + 1))
        )
        country_path_issues: list[str] = []
        for country, group in samples.groupby("country", sort=False):
            present_columns = [
                column for column in process_columns if group[column].notna().any()
            ]
            present_years = [
                int(str(column).removeprefix("log_beta_process_"))
                for column in present_columns
            ]
            complete_country = bool(
                present_years
                and present_years
                == list(range(min(present_years), max(present_years) + 1))
                and max(present_years) == forecast_end
                and required_years.issubset(present_years)
            )
            finite_country = bool(
                complete_country
                and np.isfinite(
                    group.loc[:, present_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=float)
                ).all()
            )
            if not complete_country or not finite_country:
                country_path_issues.append(
                    f"{country}:years={present_years},finite={finite_country}"
                )
        complete = bool(union_complete and not country_path_issues)
        _add(
            rows,
            category="posterior_samples",
            check="annual_process_draw_years_complete",
            passed=complete,
            severity="fatal",
            details=(
                f"years={parsed_years[:3]}...{parsed_years[-3:] if parsed_years else []}, "
                f"required_forecast={forecast_start}-{forecast_end}, invalid={invalid_names}"
            ),
        )
        finite_process = bool(complete and not country_path_issues)
        _add(
            rows,
            category="posterior_samples",
            check="annual_process_draws_finite",
            passed=finite_process,
            severity="fatal",
            details=(
                f"columns={len(process_columns)}, rows={len(samples)}, "
                f"issues={country_path_issues[:5]}"
            ),
        )
        process_metadata_columns = {
            "forecast_process_rho",
            "forecast_process_innovation_sd",
            "forecast_process_years",
            "historical_process_end_year",
        }
        metadata_missing = process_metadata_columns.difference(samples.columns)
        _add(
            rows,
            category="posterior_samples",
            check="forecast_process_hyperparameter_columns_present",
            passed=not metadata_missing,
            severity="fatal",
            details=f"missing={sorted(metadata_missing)}",
        )
    missing_parameters = [name for name in JOINT_POSTERIOR_PARAMETERS if name not in samples.columns]
    _add(
        rows,
        category="posterior_samples",
        check="joint_parameter_columns_present",
        passed=not missing_parameters,
        severity="fatal",
        details=f"missing={missing_parameters}",
    )

    audit_rows: list[dict[str, Any]] = []
    if missing_parameters:
        return pd.DataFrame(audit_rows)

    for country, group in samples.groupby("country", sort=True):
        chain_count = int(group["chain"].nunique()) if "chain" in group.columns else 0
        _add(
            rows,
            category="posterior_samples",
            check=f"{country}_chain_count",
            passed=chain_count >= expected_chains,
            severity="fatal",
            details=f"chains={chain_count}, expected>={expected_chains}",
        )
        if "chain" not in group.columns:
            _add(
                rows,
                category="posterior_samples",
                check=f"{country}_draws_per_chain",
                passed=False,
                severity="fatal",
                details="missing chain column",
            )
        elif expected_draws_per_chain <= 0:
            _add(
                rows,
                category="posterior_samples",
                check=f"{country}_draws_per_chain",
                passed=False,
                severity="fatal",
                details=f"missing or invalid expected_draws_per_chain={expected_draws_per_chain}",
            )
        else:
            draws_per_chain = group.groupby("chain", dropna=False).size()
            complete_chains = draws_per_chain.eq(expected_draws_per_chain)
            _add(
                rows,
                category="posterior_samples",
                check=f"{country}_draws_per_chain",
                passed=bool(
                    chain_count == expected_chains
                    and len(draws_per_chain) == expected_chains
                    and complete_chains.all()
                ),
                severity="fatal",
                details=(
                    f"min={int(draws_per_chain.min())}, max={int(draws_per_chain.max())}, "
                    f"expected={expected_draws_per_chain}, chains={chain_count}/{expected_chains}"
                ),
            )
        for parameter in JOINT_POSTERIOR_PARAMETERS:
            values = _finite_series(group, parameter).dropna()
            q025 = float(values.quantile(0.025)) if not values.empty else np.nan
            q500 = float(values.quantile(0.500)) if not values.empty else np.nan
            q975 = float(values.quantile(0.975)) if not values.empty else np.nan
            width = q975 - q025 if np.isfinite(q025) and np.isfinite(q975) else np.nan
            rel_width = (
                float(width / abs(q500))
                if np.isfinite(width) and np.isfinite(q500) and abs(q500) > 0
                else np.nan
            )
            variable = bool(values.nunique(dropna=True) > 1 and np.isfinite(width) and width > max(abs(q500) * 1e-8, 1e-10))
            audit_rows.append(
                {
                    "country": country,
                    "parameter": parameter,
                    "n": int(len(values)),
                    "n_unique": int(values.nunique(dropna=True)),
                    "q025": q025,
                    "median": q500,
                    "q975": q975,
                    "q95_width": float(width),
                    "relative_q95_width": rel_width,
                    "variable": variable,
                }
            )

    parameter_audit = pd.DataFrame(audit_rows)
    nonvariable = parameter_audit.loc[~parameter_audit["variable"]]
    _add(
        rows,
        category="posterior_samples",
        check="all_joint_parameters_vary_by_country",
        passed=nonvariable.empty,
        severity="fatal",
        details=(
            "all country-parameter combinations vary"
            if nonvariable.empty
            else nonvariable.loc[:, ["country", "parameter", "n_unique", "q95_width"]].to_dict("records").__repr__()
        ),
    )
    return parameter_audit


def _posterior_parameter_width_checks(rows: list[dict[str, Any]], parameter_audit: pd.DataFrame) -> None:
    required = {"country", "parameter", "q95_width", "relative_q95_width"}
    missing = required.difference(parameter_audit.columns)
    _add(
        rows,
        category="uncertainty_parameter_widths",
        check="posterior_width_audit_columns_present",
        passed=not missing,
        severity="fatal",
        details=f"missing={sorted(missing)}",
    )
    if missing:
        return

    narrow_rows: list[dict[str, Any]] = []
    for row in parameter_audit.to_dict(orient="records"):
        parameter = str(row["parameter"])
        q95_width = float(row["q95_width"]) if pd.notna(row["q95_width"]) else np.nan
        relative_q95_width = (
            float(row["relative_q95_width"]) if pd.notna(row["relative_q95_width"]) else np.nan
        )
        absolute_floor = POSTERIOR_ABSOLUTE_WIDTH_FLOORS.get(parameter)
        relative_floor = POSTERIOR_RELATIVE_WIDTH_FLOORS.get(parameter)
        if absolute_floor is not None and (not np.isfinite(q95_width) or q95_width < absolute_floor):
            narrow_rows.append(
                {
                    "country": row["country"],
                    "parameter": parameter,
                    "q95_width": q95_width,
                    "floor": absolute_floor,
                    "floor_type": "absolute",
                }
            )
        if relative_floor is not None and (
            not np.isfinite(relative_q95_width) or relative_q95_width < relative_floor
        ):
            narrow_rows.append(
                {
                    "country": row["country"],
                    "parameter": parameter,
                    "q95_width": relative_q95_width,
                    "floor": relative_floor,
                    "floor_type": "relative",
                }
            )

    _add(
        rows,
        category="uncertainty_parameter_widths",
        check="uncertainty_parameter_widths_not_degenerate",
        passed=not narrow_rows,
        severity="warning",
        details=(
            "all posterior parameter widths above degeneracy floors"
            if not narrow_rows
            else narrow_rows[:30].__repr__()
        ),
    )


def _intervention_uncertainty_bounds() -> dict[str, tuple[float, float]]:
    configs = load_configs()
    baseline = configs["baseline"]
    figure_settings = baseline.get("bayesian_uncertainty", {}).get("figure2c_joint_credible_interval", {})
    configured_priors = figure_settings.get("intervention_priors", {})

    bounds: dict[str, tuple[float, float]] = {}
    for name, defaults in INTERVENTION_UNCERTAINTY_DEFAULTS.items():
        values = {**defaults, **configured_priors.get(name, {})}
        bounds[f"{INTERVENTION_UNCERTAINTY_PREFIX}{name}"] = (float(values["low"]), float(values["high"]))
    bounds[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sus"] = (0.0, 1.0)
    bounds[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sym"] = (0.0, 1.0)
    return bounds


def _joint_importance_checks(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    structural: pd.DataFrame,
    local: pd.DataFrame,
    *,
    expected_countries: set[str],
) -> None:
    """Independently audit the numerical support behind the joint posterior."""

    required_structural = {
        "structural_index",
        "joint_log_importance_weight",
        "normalized_weight",
        *SHARED_PARAMETER_NAMES,
    }
    required_local = {
        "country",
        "structural_index",
        "state_effective_sample_size",
        "state_maximum_weight",
        "state_pareto_k",
        "state_minimum_tail_ess",
        "state_in_bounds_fraction",
    }
    missing_structural = sorted(required_structural.difference(structural.columns))
    missing_local = sorted(required_local.difference(local.columns))
    _add(
        rows,
        category="joint_importance",
        check="joint_importance_quality_columns_present",
        passed=not missing_structural and not missing_local,
        severity="fatal",
        details=f"missing_structural={missing_structural}, missing_local={missing_local}",
    )
    if missing_structural or missing_local:
        return

    quality = metadata.get("quality")
    quality_checks = quality.get("checks") if isinstance(quality, dict) else None
    metadata_complete = bool(
        isinstance(quality, dict)
        and isinstance(quality_checks, dict)
        and quality_checks
        and all(isinstance(value, (bool, np.bool_)) for value in quality_checks.values())
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_importance_quality_metadata_present",
        passed=metadata_complete,
        severity="fatal",
        details=(
            f"quality_keys={sorted(quality.keys()) if isinstance(quality, dict) else []}, "
            f"check_keys={sorted(quality_checks) if isinstance(quality_checks, dict) else []}"
        ),
    )

    structural_index = pd.to_numeric(
        structural["structural_index"], errors="coerce"
    )
    weights = pd.to_numeric(structural["normalized_weight"], errors="coerce")
    log_weights = pd.to_numeric(
        structural["joint_log_importance_weight"], errors="coerce"
    )
    valid_weight = bool(
        structural_index.notna().all()
        and not structural_index.duplicated().any()
        and weights.notna().all()
        and log_weights.notna().all()
        and np.isfinite(weights.to_numpy(dtype=float)).all()
        and np.isfinite(log_weights.to_numpy(dtype=float)).all()
        and weights.ge(0).all()
        and np.isclose(float(weights.sum()), 1.0, rtol=1e-10, atol=1e-12)
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_structural_weights_valid",
        passed=valid_weight,
        severity="fatal",
        details=(
            f"rows={len(structural)}, unique_indices={structural_index.nunique()}, "
            f"weight_sum={weights.sum(skipna=True)}"
        ),
    )
    if not valid_weight:
        return

    coordinate_frame = structural.loc[:, list(SHARED_PARAMETER_NAMES)].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_coordinates = bool(
        np.isfinite(coordinate_frame.to_numpy(dtype=float)).all()
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_structural_coordinates_finite",
        passed=finite_coordinates,
        severity="fatal",
        details=f"rows={len(coordinate_frame)}, dimensions={len(SHARED_PARAMETER_NAMES)}",
    )
    if not finite_coordinates:
        return

    weight_array = weights.to_numpy(dtype=float)
    global_ess = float(1.0 / np.sum(weight_array**2))
    global_max_weight = float(weight_array.max())
    global_pareto_k = float(
        _importance_pareto_shape(log_weights.to_numpy(dtype=float))
    )
    global_tail_ess = float(
        _minimum_importance_tail_ess(
            coordinate_frame.to_numpy(dtype=float), weight_array
        )
    )
    global_pass = bool(
        global_ess >= RECOMMENDED_MIN_GLOBAL_ESS
        and global_max_weight <= RECOMMENDED_MAX_GLOBAL_WEIGHT
        and global_pareto_k <= RECOMMENDED_MAX_GLOBAL_PARETO_K
        and global_tail_ess >= RECOMMENDED_MIN_GLOBAL_TAIL_ESS
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_global_importance_all_recommended",
        passed=global_pass,
        severity="fatal",
        details=(
            f"ESS={global_ess:.6g}>={RECOMMENDED_MIN_GLOBAL_ESS}, "
            f"max_weight={global_max_weight:.6g}<={RECOMMENDED_MAX_GLOBAL_WEIGHT}, "
            f"Pareto_k={global_pareto_k:.6g}<={RECOMMENDED_MAX_GLOBAL_PARETO_K}, "
            f"min_tail_ESS={global_tail_ess:.6g}>={RECOMMENDED_MIN_GLOBAL_TAIL_ESS}"
        ),
    )

    local_data = local.copy()
    local_data["country"] = local_data["country"].astype(str)
    local_data["structural_index"] = pd.to_numeric(
        local_data["structural_index"], errors="coerce"
    )
    for column in required_local.difference({"country", "structural_index"}):
        local_data[column] = pd.to_numeric(local_data[column], errors="coerce")
    expected_rows = len(structural) * len(expected_countries)
    local_grid_valid = bool(
        len(local_data) == expected_rows
        and set(local_data["country"]) == expected_countries
        and not local_data.duplicated(["country", "structural_index"]).any()
        and local_data.loc[:, list(required_local - {"country"})].notna().all().all()
        and np.isfinite(
            local_data.loc[:, list(required_local - {"country"})].to_numpy(
                dtype=float
            )
        ).all()
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_local_importance_grid_complete",
        passed=local_grid_valid,
        severity="fatal",
        details=(
            f"rows={len(local_data)}/{expected_rows}, "
            f"countries={sorted(set(local_data['country']))}"
        ),
    )
    if not local_grid_valid:
        return

    order = np.argsort(-weight_array)
    count = int(
        np.searchsorted(
            np.cumsum(weight_array[order]), POSTERIOR_RELEVANT_MASS, side="left"
        )
        + 1
    )
    relevant_ids = set(
        structural_index.iloc[order[:count]].astype(int).tolist()
    )
    relevant = local_data.loc[
        local_data["structural_index"].astype(int).isin(relevant_ids)
    ]
    local_recommended = (
        local_data["state_effective_sample_size"].ge(RECOMMENDED_MIN_LOCAL_ESS)
        & local_data["state_maximum_weight"].le(RECOMMENDED_MAX_LOCAL_WEIGHT)
        & local_data["state_pareto_k"].le(RECOMMENDED_MAX_LOCAL_PARETO_K)
        & local_data["state_minimum_tail_ess"].ge(RECOMMENDED_MIN_LOCAL_TAIL_ESS)
        & local_data["state_in_bounds_fraction"].ge(
            RECOMMENDED_MIN_STATE_IN_BOUNDS_FRACTION
        )
    )
    structural_ok = (
        pd.DataFrame(
            {
                "structural_index": local_data["structural_index"].astype(int),
                "recommended": local_recommended,
            }
        )
        .groupby("structural_index", sort=False)["recommended"]
        .all()
    )
    failed_ids = set(structural_ok.index[~structural_ok].astype(int))
    failed_mass = float(
        structural.loc[
            structural_index.astype(int).isin(failed_ids), "normalized_weight"
        ].sum()
    )
    local_pass = bool(
        relevant["state_effective_sample_size"].min() >= RECOMMENDED_MIN_LOCAL_ESS
        and relevant["state_maximum_weight"].max() <= RECOMMENDED_MAX_LOCAL_WEIGHT
        and relevant["state_pareto_k"].max() <= RECOMMENDED_MAX_LOCAL_PARETO_K
        and relevant["state_minimum_tail_ess"].min()
        >= RECOMMENDED_MIN_LOCAL_TAIL_ESS
        and relevant["state_in_bounds_fraction"].min()
        >= RECOMMENDED_MIN_STATE_IN_BOUNDS_FRACTION
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_local_importance_all_recommended",
        passed=local_pass,
        severity="fatal",
        details=(
            f"posterior_relevant_structural_draws={len(relevant_ids)}, "
            f"min_ESS={relevant['state_effective_sample_size'].min():.6g}, "
            f"max_weight={relevant['state_maximum_weight'].max():.6g}, "
            f"max_Pareto_k={relevant['state_pareto_k'].max():.6g}, "
            f"min_tail_ESS={relevant['state_minimum_tail_ess'].min():.6g}, "
            f"min_in_bounds={relevant['state_in_bounds_fraction'].min():.6g}"
        ),
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_local_failed_posterior_mass_below_limit",
        passed=failed_mass <= RECOMMENDED_MAX_FAILED_LOCAL_POSTERIOR_MASS,
        severity="fatal",
        details=(
            f"failed_posterior_mass={failed_mass:.6g}, "
            f"maximum={RECOMMENDED_MAX_FAILED_LOCAL_POSTERIOR_MASS}"
        ),
    )
    _add(
        rows,
        category="joint_importance",
        check="joint_importance_metadata_all_recommended",
        passed=bool(
            metadata_complete
            and quality.get("all_recommended_converged") is True
            and all(bool(value) for value in quality_checks.values())
        ),
        severity="fatal",
        details=(
            f"all_recommended_converged={quality.get('all_recommended_converged') if isinstance(quality, dict) else None}, "
            f"checks={quality_checks}"
        ),
    )


def _joint_smc_checks(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    stages: pd.DataFrame,
    diagnostics: pd.DataFrame,
    boundary: pd.DataFrame,
) -> None:
    required = {
        "stage",
        "temperature",
        "temperature_increment",
        "minimum_island_ess_fraction",
        "maximum_island_particle_weight",
        "move_rounds_per_stage",
        "global_covariance_fraction",
        "guided_geometry_weight",
        "shared_guided_proposal",
        "shared_move_acceptance",
        "shared_guided_independence_acceptance",
        "mean_country_state_move_acceptance",
        "mean_country_guided_independence_acceptance",
    }
    missing = required.difference(stages.columns)
    _add(
        rows,
        category="joint_smc",
        check="smc_stage_columns_present",
        passed=not missing,
        severity="fatal",
        details=f"missing={sorted(missing)}",
    )
    if missing or stages.empty:
        return
    temperature = _finite_series(stages, "temperature")
    increment = _finite_series(stages, "temperature_increment")
    ess_fraction = _finite_series(stages, "minimum_island_ess_fraction")
    shared_acceptance = _finite_series(stages, "shared_move_acceptance")
    state_acceptance = _finite_series(
        stages, "mean_country_state_move_acceptance"
    )
    move_rounds = _finite_series(stages, "move_rounds_per_stage")
    global_covariance = _finite_series(stages, "global_covariance_fraction")
    guided_geometry_weight = _finite_series(stages, "guided_geometry_weight")
    shared_guided = _finite_series(
        stages, "shared_guided_independence_acceptance"
    )
    state_guided = _finite_series(
        stages, "mean_country_guided_independence_acceptance"
    )
    temperature_valid = bool(
        temperature.notna().all()
        and temperature.between(0.0, 1.0).all()
        and temperature.is_monotonic_increasing
        and np.isclose(float(temperature.iloc[-1]), 1.0)
        and increment.gt(0.0).all()
    )
    _add(
        rows,
        category="joint_smc",
        check="smc_temperature_path_reaches_full_posterior",
        passed=temperature_valid,
        severity="fatal",
        details=(
            f"stages={len(stages)}, final_temperature={temperature.iloc[-1]}, "
            f"minimum_increment={increment.min()}"
        ),
    )
    _add(
        rows,
        category="joint_smc",
        check="smc_each_stage_preserves_recommended_island_ess",
        passed=bool(
            ess_fraction.notna().all()
            and ess_fraction.ge(SMC_RECOMMENDED_ESS_FRACTION - 1e-8).all()
        ),
        severity="fatal",
        details=(
            f"minimum_fraction={ess_fraction.min()}, "
            f"recommended>={SMC_RECOMMENDED_ESS_FRACTION}"
        ),
    )
    acceptance_valid = bool(
        shared_acceptance.notna().all()
        and state_acceptance.notna().all()
        and shared_acceptance.between(0.0, 1.0).all()
        and state_acceptance.between(0.0, 1.0).all()
        and float(pd.concat((shared_acceptance, state_acceptance)).mean()) >= 0.10
    )
    _add(
        rows,
        category="joint_smc",
        check="smc_population_geometry_rejuvenation_recorded",
        passed=bool(
            move_rounds.notna().all()
            and move_rounds.ge(4).all()
            and global_covariance.notna().all()
            and global_covariance.between(0.10, 0.75).all()
            and guided_geometry_weight.notna().all()
            and guided_geometry_weight.between(0.0, 1.0).all()
            and np.isclose(float(guided_geometry_weight.iloc[-1]), 1.0)
            and int(metadata.get("move_rounds_per_stage", 0)) >= 4
            and 0.10
            <= float(metadata.get("global_covariance_fraction", -1.0))
            <= 0.75
        ),
        severity="fatal",
        details=(
            f"move_rounds={sorted(move_rounds.unique().tolist())}, "
            f"global_covariance_fraction={sorted(global_covariance.unique().tolist())}, "
            f"guided_geometry_final_weight={guided_geometry_weight.iloc[-1]}"
        ),
    )
    _add(
        rows,
        category="joint_smc",
        check="smc_guided_independence_moves_are_active",
        passed=bool(
            shared_guided.notna().all()
            and state_guided.notna().all()
            and shared_guided.between(0.0, 1.0).all()
            and state_guided.between(0.0, 1.0).all()
            and float(pd.concat((shared_guided, state_guided)).mean()) >= 0.01
            and float(metadata.get("guided_independence_move_fraction", 0.0))
            >= 0.5
            and float(metadata.get("guided_state_covariance_scale", 0.0)) >= 1.0
            and str(stages["shared_guided_proposal"].iloc[-1])
            == "population_gaussian_mixture"
        ),
        severity="fatal",
        details=(
            f"mean_shared_guided={shared_guided.mean()}, "
            f"mean_state_guided={state_guided.mean()}, "
            f"final_shared_guide={stages['shared_guided_proposal'].iloc[-1]}"
        ),
    )
    _add(
        rows,
        category="joint_smc",
        check="smc_rejuvenation_moves_are_active",
        passed=acceptance_valid,
        severity="fatal",
        details=(
            f"mean_shared={shared_acceptance.mean()}, "
            f"mean_state={state_acceptance.mean()}"
        ),
    )
    boundary_required = {
        "country",
        "coordinate",
        "lower_edge_mass",
        "upper_edge_mass",
        "maximum_edge_mass",
    }
    boundary_missing = boundary_required.difference(boundary.columns)
    boundary_mass = _finite_series(boundary, "maximum_edge_mass")
    _add(
        rows,
        category="joint_smc",
        check="smc_state_posterior_not_truncated_by_numerical_bounds",
        passed=bool(
            not boundary_missing
            and not boundary.empty
            and boundary_mass.notna().all()
            and boundary_mass.le(0.01).all()
        ),
        severity="fatal",
        details=(
            f"missing={sorted(boundary_missing)}, "
            f"maximum_edge_mass={boundary_mass.max(skipna=True)}, threshold<=0.01"
        ),
    )
    quality = metadata.get("quality") or {}
    checks = quality.get("checks") if isinstance(quality, dict) else {}
    _add(
        rows,
        category="joint_smc",
        check="joint_smc_metadata_all_recommended",
        passed=bool(
            quality.get("all_recommended_converged") is True
            and isinstance(checks, dict)
            and checks
            and all(bool(value) for value in checks.values())
        ),
        severity="fatal",
        details=(
            f"all_recommended={quality.get('all_recommended_converged')}, "
            f"checks={checks}"
        ),
    )
    _convergence_checks(rows, diagnostics)


def _convergence_checks(rows: list[dict[str, Any]], diagnostics: pd.DataFrame) -> None:
    required = {"country", "parameter", "rhat_rank", "bulk_ess", "tail_ess", "converged"}
    missing = required.difference(diagnostics.columns)
    _add(
        rows,
        category="convergence",
        check="diagnostic_columns_present",
        passed=not missing,
        severity="fatal",
        details=f"missing={sorted(missing)}",
    )
    if missing:
        return

    method = (
        diagnostics.get("diagnostic_method", pd.Series("mcmc", index=diagnostics.index))
        .fillna("mcmc")
        .astype(str)
        .str.lower()
    )
    is_modular_cut = bool(method.eq("modular_cut_conditional_grid_quality").all())
    is_joint_importance = bool(
        method.isin(
            {
                "joint_importance_weight_quality",
                "modular_cut_conditional_grid_quality",
            }
        ).all()
    )
    is_smc = bool(method.eq("tempered_smc_rank_and_particle_quality").all())
    is_state_space_exact_importance = bool(
        method.eq("state_space_exact_importance_and_prior_design_quality").all()
    )
    converged = diagnostics["converged"].astype(bool)
    _add(
        rows,
        category="convergence",
        check="all_country_parameters_converged",
        passed=bool(converged.all()),
        severity="fatal",
        details=f"converged={int(converged.sum())}/{len(converged)}",
    )
    if is_state_space_exact_importance:
        required_state = {
            "state_posterior_rank",
            "state_posterior_dimension",
            "state_posterior_condition_number",
            "state_laplace_boundary_rejection_fraction",
            "structural_prior_design_size",
            "state_importance_candidate_count",
            "state_importance_effective_sample_size",
            "state_importance_max_weight",
            "state_importance_pareto_k",
            "state_importance_min_tail_ess",
            "recommended_converged",
        }
        state_missing = required_state.difference(diagnostics.columns)
        _add(
            rows,
            category="convergence",
            check="state_space_exact_importance_quality_columns_present",
            passed=not state_missing,
            severity="fatal",
            details=f"missing={sorted(state_missing)}",
        )
        if state_missing:
            return
        rank = _finite_series(diagnostics, "state_posterior_rank")
        dimension = _finite_series(diagnostics, "state_posterior_dimension")
        condition = _finite_series(diagnostics, "state_posterior_condition_number")
        rejection = _finite_series(
            diagnostics, "state_laplace_boundary_rejection_fraction"
        )
        structural = _finite_series(diagnostics, "structural_prior_design_size")
        state_importance_ess = _finite_series(
            diagnostics, "state_importance_effective_sample_size"
        )
        state_importance_max_weight = _finite_series(
            diagnostics, "state_importance_max_weight"
        )
        state_importance_pareto_k = _finite_series(
            diagnostics, "state_importance_pareto_k"
        )
        state_importance_min_tail_ess = _finite_series(
            diagnostics, "state_importance_min_tail_ess"
        )
        recommended = diagnostics["recommended_converged"].astype(bool)
        _add(
            rows,
            category="convergence",
            check="state_space_hessian_full_rank",
            passed=bool(rank.eq(dimension).all()),
            severity="fatal",
            details=f"min_rank_margin={(rank - dimension).min(skipna=True)}",
        )
        _add(
            rows,
            category="convergence",
            check="state_space_hessian_condition_under_1e8",
            passed=bool(condition.le(1e8).all()),
            severity="fatal",
            details=f"max_condition={condition.max(skipna=True)}",
        )
        _add(
            rows,
            category="convergence",
            check="laplace_boundary_rejection_under_recommended_limit",
            passed=bool(
                rejection.lt(STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION).all()
            ),
            severity="warning",
            details=(
                f"max_rejection_fraction={rejection.max(skipna=True)}, "
                f"recommended<{STATE_LAPLACE_RECOMMENDED_MAX_BOUNDARY_REJECTION}, "
                f"fatal<{STATE_LAPLACE_FATAL_MAX_BOUNDARY_REJECTION}"
            ),
        )
        _add(
            rows,
            category="convergence",
            check="structural_prior_design_has_at_least_100_points",
            passed=bool(structural.ge(100).all()),
            severity="warning",
            details=f"min_design_size={structural.min(skipna=True)}",
        )
        _add(
            rows,
            category="convergence",
            check="state_space_exact_importance_all_parameters_recommended",
            passed=bool(recommended.all()),
            severity="warning",
            details=f"recommended={int(recommended.sum())}/{len(recommended)}",
        )
        _add(
            rows,
            category="convergence",
            check="exact_state_importance_ess_meets_recommended",
            passed=bool(
                state_importance_ess.ge(STATE_IMPORTANCE_RECOMMENDED_MIN_ESS).all()
            ),
            severity="warning",
            details=(
                f"min_ess={state_importance_ess.min(skipna=True)}, "
                f"recommended>={STATE_IMPORTANCE_RECOMMENDED_MIN_ESS}"
            ),
        )
        _add(
            rows,
            category="convergence",
            check="exact_state_importance_max_weight_meets_recommended",
            passed=bool(
                state_importance_max_weight.le(
                    STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT
                ).all()
            ),
            severity="warning",
            details=(
                f"max_weight={state_importance_max_weight.max(skipna=True)}, "
                f"recommended<={STATE_IMPORTANCE_RECOMMENDED_MAX_WEIGHT}"
            ),
        )
        _add(
            rows,
            category="convergence",
            check="exact_state_importance_pareto_k_meets_recommended",
            passed=bool(
                state_importance_pareto_k.le(
                    STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K
                ).all()
            ),
            severity="warning",
            details=(
                f"max_pareto_k={state_importance_pareto_k.max(skipna=True)}, "
                f"recommended<={STATE_IMPORTANCE_RECOMMENDED_MAX_PARETO_K}"
            ),
        )
        _add(
            rows,
            category="convergence",
            check="exact_state_importance_tail_support_meets_recommended",
            passed=bool(
                state_importance_min_tail_ess.ge(
                    STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS
                ).all()
            ),
            severity="warning",
            details=(
                f"min_tail_ess={state_importance_min_tail_ess.min(skipna=True)}, "
                f"recommended>={STATE_IMPORTANCE_RECOMMENDED_MIN_TAIL_ESS}"
            ),
        )
        return
    if is_joint_importance:
        recommended = (
            diagnostics.get("recommended_converged", pd.Series(False, index=diagnostics.index))
            .astype(bool)
        )
        _add(
            rows,
            category="convergence",
            check="all_country_parameters_meet_recommended_importance_quality",
            passed=bool(recommended.all()),
            severity="warning",
            details=f"recommended={int(recommended.sum())}/{len(recommended)}",
        )
        if is_modular_cut:
            modular_required = {
                "modular_cut_structural_draw_count",
                "modular_cut_conditional_incomplete_grid_count",
                "modular_cut_conditional_recommended_failure_count",
                "modular_cut_conditional_min_effective_grid_points",
                "modular_cut_conditional_min_beta_effective_grid_points",
                "modular_cut_conditional_min_reporting_effective_grid_points",
                "modular_cut_conditional_max_single_weight",
                "modular_cut_conditional_max_edge_weight",
            }
            modular_missing = modular_required.difference(diagnostics.columns)
            _add(
                rows,
                category="convergence",
                check="modular_cut_conditional_quality_columns_present",
                passed=not modular_missing,
                severity="fatal",
                details=f"missing={sorted(modular_missing)}",
            )
            if modular_missing:
                return
            structural_draws = _finite_series(
                diagnostics, "modular_cut_structural_draw_count"
            )
            incomplete = _finite_series(
                diagnostics, "modular_cut_conditional_incomplete_grid_count"
            )
            recommended_failures = _finite_series(
                diagnostics,
                "modular_cut_conditional_recommended_failure_count",
            )
            conditional_ess = _finite_series(
                diagnostics,
                "modular_cut_conditional_min_effective_grid_points",
            )
            beta_axis_ess = _finite_series(
                diagnostics,
                "modular_cut_conditional_min_beta_effective_grid_points",
            )
            reporting_axis_ess = _finite_series(
                diagnostics,
                "modular_cut_conditional_min_reporting_effective_grid_points",
            )
            conditional_max_weight = _finite_series(
                diagnostics,
                "modular_cut_conditional_max_single_weight",
            )
            conditional_edge_weight = _finite_series(
                diagnostics,
                "modular_cut_conditional_max_edge_weight",
            )
            _add(
                rows,
                category="convergence",
                check="modular_cut_complete_conditional_grids",
                passed=bool(incomplete.dropna().eq(0.0).all()),
                severity="fatal",
                details=f"max_incomplete_grids={incomplete.max(skipna=True)}",
            )
            _add(
                rows,
                category="convergence",
                check="modular_cut_structural_prior_draws_over_100",
                passed=bool(
                    structural_draws.dropna()
                    .ge(MODULAR_CUT_RECOMMENDED_MIN_STRUCTURAL_DRAWS)
                    .all()
                ),
                severity="warning",
                details=(
                    f"min_structural_draws={structural_draws.min(skipna=True)}, "
                    f"recommended>={MODULAR_CUT_RECOMMENDED_MIN_STRUCTURAL_DRAWS}"
                ),
            )
            _add(
                rows,
                category="convergence",
                check="modular_cut_all_conditional_grids_meet_recommended_quality",
                passed=bool(recommended_failures.dropna().eq(0.0).all()),
                severity="warning",
                details=(
                    "max_recommended_failures="
                    f"{recommended_failures.max(skipna=True)}"
                ),
            )
            _add(
                rows,
                category="convergence",
                check="modular_cut_conditional_effective_points_over_20",
                passed=bool(
                    conditional_ess.dropna()
                    .ge(MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS)
                    .all()
                ),
                severity="warning",
                details=(
                    f"min_conditional_ess={conditional_ess.min(skipna=True)}, "
                    f"recommended>={MODULAR_CUT_RECOMMENDED_MIN_CONDITIONAL_EFFECTIVE_POINTS}"
                ),
            )
            axis_ess = pd.concat([beta_axis_ess, reporting_axis_ess], ignore_index=True)
            _add(
                rows,
                category="convergence",
                check="modular_cut_each_axis_effective_points_over_5",
                passed=bool(
                    axis_ess.dropna()
                    .ge(MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS)
                    .all()
                ),
                severity="warning",
                details=(
                    f"min_axis_ess={axis_ess.min(skipna=True)}, "
                    f"recommended>={MODULAR_CUT_RECOMMENDED_MIN_AXIS_EFFECTIVE_POINTS}"
                ),
            )
            _add(
                rows,
                category="convergence",
                check="modular_cut_conditional_max_weight_under_0_10",
                passed=bool(
                    conditional_max_weight.dropna()
                    .le(MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT)
                    .all()
                ),
                severity="warning",
                details=(
                    f"max_conditional_weight={conditional_max_weight.max(skipna=True)}, "
                    f"recommended<={MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_WEIGHT}"
                ),
            )
            _add(
                rows,
                category="convergence",
                check="modular_cut_conditional_edge_weight_under_0_01",
                passed=bool(
                    conditional_edge_weight.dropna()
                    .le(MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT)
                    .all()
                ),
                severity="warning",
                details=(
                    f"max_conditional_edge_weight={conditional_edge_weight.max(skipna=True)}, "
                    f"recommended<={MODULAR_CUT_RECOMMENDED_MAX_CONDITIONAL_EDGE_WEIGHT}"
                ),
            )
            return
        importance_required = {
            "importance_effective_sample_size",
            "importance_nuisance_effective_sample_size",
            "importance_max_weight",
            "importance_edge_weight",
        }
        importance_missing = importance_required.difference(diagnostics.columns)
        _add(
            rows,
            category="convergence",
            check="importance_quality_columns_present",
            passed=not importance_missing,
            severity="fatal",
            details=f"missing={sorted(importance_missing)}",
        )
        if importance_missing:
            return
        ess = _finite_series(diagnostics, "importance_effective_sample_size")
        nuisance_ess = _finite_series(diagnostics, "importance_nuisance_effective_sample_size")
        max_weight = _finite_series(diagnostics, "importance_max_weight")
        edge_weight = _finite_series(diagnostics, "importance_edge_weight")
        _add(
            rows,
            category="convergence",
            check="importance_ess_over_400",
            passed=bool(ess.dropna().ge(MIN_RECOMMENDED_BULK_ESS).all()),
            severity="warning",
            details=f"min_importance_ess={ess.min(skipna=True)}, recommended>={MIN_RECOMMENDED_BULK_ESS}",
        )
        _add(
            rows,
            category="convergence",
            check="importance_nuisance_ess_over_100",
            passed=bool(nuisance_ess.dropna().ge(100.0).all()),
            severity="warning",
            details=f"min_nuisance_ess={nuisance_ess.min(skipna=True)}, recommended>=100",
        )
        _add(
            rows,
            category="convergence",
            check="importance_max_weight_under_0_02",
            passed=bool(max_weight.dropna().le(0.02).all()),
            severity="warning",
            details=f"max_weight={max_weight.max(skipna=True)}, recommended<=0.02",
        )
        _add(
            rows,
            category="convergence",
            check="importance_edge_weight_under_0_01",
            passed=bool(edge_weight.dropna().le(0.01).all()),
            severity="warning",
            details=f"max_edge_weight={edge_weight.max(skipna=True)}, recommended<=0.01",
        )
        return

    if is_smc:
        recommended = (
            diagnostics.get("recommended_converged", pd.Series(False, index=diagnostics.index))
            .astype(bool)
        )
        _add(
            rows,
            category="convergence",
            check="all_country_parameters_meet_recommended_smc_quality",
            passed=bool(recommended.all()),
            severity="warning",
            details=f"recommended={int(recommended.sum())}/{len(recommended)}",
        )
        smc_required = {
            "smc_reached_final_temperature",
            "smc_min_ess_fraction",
            "smc_final_ess",
            "smc_combined_particle_ess",
            "smc_max_weight",
            "smc_unique_particle_fraction",
            "smc_island_count",
            "smc_island_ess",
            "smc_max_island_weight",
        }
        smc_missing = smc_required.difference(diagnostics.columns)
        _add(
            rows,
            category="convergence",
            check="smc_quality_columns_present",
            passed=not smc_missing,
            severity="fatal",
            details=f"missing={sorted(smc_missing)}",
        )
        if smc_missing:
            return
        reached_final = diagnostics["smc_reached_final_temperature"].astype(bool)
        min_stage_fraction = _finite_series(diagnostics, "smc_min_ess_fraction")
        final_ess = _finite_series(diagnostics, "smc_final_ess")
        combined_ess = _finite_series(diagnostics, "smc_combined_particle_ess")
        max_weight = _finite_series(diagnostics, "smc_max_weight")
        unique_fraction = _finite_series(diagnostics, "smc_unique_particle_fraction")
        island_count = _finite_series(diagnostics, "smc_island_count")
        island_ess = _finite_series(diagnostics, "smc_island_ess")
        max_island_weight = _finite_series(diagnostics, "smc_max_island_weight")
        _add(
            rows,
            category="convergence",
            check="smc_reaches_final_temperature",
            passed=bool(reached_final.all()),
            severity="fatal",
            details=f"reached={int(reached_final.sum())}/{len(reached_final)}",
        )
        _add(
            rows,
            category="convergence",
            check="smc_stage_ess_fraction_over_0_50",
            passed=bool(min_stage_fraction.dropna().ge(SMC_RECOMMENDED_ESS_FRACTION).all()),
            severity="warning",
            details=(
                f"min_stage_ess_fraction={min_stage_fraction.min(skipna=True)}, "
                f"recommended>={SMC_RECOMMENDED_ESS_FRACTION}"
            ),
        )
        _add(
            rows,
            category="convergence",
            check="smc_final_ess_over_400",
            passed=bool(final_ess.dropna().ge(MIN_RECOMMENDED_BULK_ESS).all()),
            severity="warning",
            details=f"min_final_ess={final_ess.min(skipna=True)}, recommended>={MIN_RECOMMENDED_BULK_ESS}",
        )
        _add(
            rows,
            category="convergence",
            check="smc_combined_particle_ess_over_400",
            passed=bool(combined_ess.dropna().ge(MIN_RECOMMENDED_BULK_ESS).all()),
            severity="warning",
            details=(
                f"min_combined_particle_ess={combined_ess.min(skipna=True)}, "
                f"recommended>={MIN_RECOMMENDED_BULK_ESS}"
            ),
        )
        island_floor = island_count * SMC_RECOMMENDED_ISLAND_ESS_FRACTION
        _add(
            rows,
            category="convergence",
            check="smc_island_ess_over_0_35_islands",
            passed=bool(island_ess.dropna().ge(island_floor.dropna()).all()),
            severity="warning",
            details=(
                f"min_island_ess={island_ess.min(skipna=True)}, "
                f"min_recommended={island_floor.min(skipna=True)}"
            ),
        )
        _add(
            rows,
            category="convergence",
            check="smc_max_island_weight_under_0_50",
            passed=bool(max_island_weight.dropna().le(SMC_RECOMMENDED_MAX_ISLAND_WEIGHT).all()),
            severity="warning",
            details=(
                f"max_island_weight={max_island_weight.max(skipna=True)}, "
                f"recommended<={SMC_RECOMMENDED_MAX_ISLAND_WEIGHT}"
            ),
        )
        _add(
            rows,
            category="convergence",
            check="smc_max_weight_under_0_02",
            passed=bool(max_weight.dropna().le(SMC_RECOMMENDED_MAX_WEIGHT).all()),
            severity="warning",
            details=f"max_weight={max_weight.max(skipna=True)}, recommended<={SMC_RECOMMENDED_MAX_WEIGHT}",
        )
        _add(
            rows,
            category="convergence",
            check="smc_unique_particle_fraction_over_0_25",
            passed=bool(unique_fraction.dropna().ge(SMC_RECOMMENDED_UNIQUE_PARTICLE_FRACTION).all()),
            severity="warning",
            details=(
                f"min_unique_particle_fraction={unique_fraction.min(skipna=True)}, "
                f"recommended>={SMC_RECOMMENDED_UNIQUE_PARTICLE_FRACTION}"
            ),
        )

    rhat = _finite_series(diagnostics, "rhat_rank")
    bulk = _finite_series(diagnostics, "bulk_ess")
    tail = _finite_series(diagnostics, "tail_ess")
    _add(
        rows,
        category="convergence",
        check="worst_rhat_under_1_05",
        passed=bool(rhat.dropna().le(MAX_FATAL_RHAT).all()),
        severity="fatal",
        details=f"worst_rhat={rhat.max(skipna=True)}, threshold<={MAX_FATAL_RHAT}",
    )
    _add(
        rows,
        category="convergence",
        check="bulk_ess_over_100",
        passed=bool(bulk.dropna().ge(MIN_FATAL_BULK_ESS).all()),
        severity="fatal",
        details=f"min_bulk_ess={bulk.min(skipna=True)}, threshold>={MIN_FATAL_BULK_ESS}",
    )
    _add(
        rows,
        category="convergence",
        check="tail_ess_over_50",
        passed=bool(tail.dropna().ge(MIN_FATAL_TAIL_ESS).all()),
        severity="fatal",
        details=f"min_tail_ess={tail.min(skipna=True)}, threshold>={MIN_FATAL_TAIL_ESS}",
    )
    _add(
        rows,
        category="convergence",
        check="recommended_rhat_under_1_01",
        passed=bool(rhat.dropna().le(MAX_RECOMMENDED_RHAT).all()),
        severity="warning",
        details=f"worst_rhat={rhat.max(skipna=True)}, recommended<={MAX_RECOMMENDED_RHAT}",
    )
    _add(
        rows,
        category="convergence",
        check="recommended_bulk_ess_over_400",
        passed=bool(bulk.dropna().ge(MIN_RECOMMENDED_BULK_ESS).all()),
        severity="warning",
        details=f"min_bulk_ess={bulk.min(skipna=True)}, recommended>={MIN_RECOMMENDED_BULK_ESS}",
    )
    _add(
        rows,
        category="convergence",
        check="recommended_tail_ess_over_400",
        passed=bool(tail.dropna().ge(MIN_RECOMMENDED_TAIL_ESS).all()),
        severity="warning",
        details=f"min_tail_ess={tail.min(skipna=True)}, recommended>={MIN_RECOMMENDED_TAIL_ESS}",
    )


def _interval_checks(
    rows: list[dict[str, Any]],
    intervals: pd.DataFrame,
    *,
    expected_countries: set[str],
    expected_draws: int,
    expected_inference_structure: str = "",
) -> None:
    intervention_strategies = set(FIGURE2C_STRATEGIES) - {"current"}
    expected_rows = len(expected_countries) * len(intervention_strategies)
    _add(
        rows,
        category="intervals",
        check="uncertainty_interval_quantile_monte_carlo_resolution",
        passed=expected_draws >= 2000,
        severity="warning",
        details=(
            f"uncertainty_draws_per_country_strategy_expected={expected_draws}, "
            f"expected_draws_in_each_2.5pct_tail={0.025 * expected_draws:.1f}, "
            f"quantile_grid_spacing={1.0 / (expected_draws + 1):.4f}"
        ),
    )
    _add(
        rows,
        category="intervals",
        check="expected_country_strategy_rows",
        passed=len(intervals) == expected_rows,
        severity="fatal",
        details=f"rows={len(intervals)}, expected={expected_rows}",
    )
    countries = set(intervals.get("country", pd.Series(dtype=str)).astype(str))
    strategies = set(intervals.get("strategy", pd.Series(dtype=str)).astype(str))
    _add(
        rows,
        category="intervals",
        check="country_and_strategy_coverage",
        passed=countries == expected_countries and strategies == intervention_strategies,
        severity="fatal",
        details=f"countries={sorted(countries)}, strategies={sorted(strategies)}",
    )
    for column in ("reduction_q025", "reduction_median", "reduction_q975", "uncertainty_draws"):
        _add(
            rows,
            category="intervals",
            check=f"{column}_finite",
            passed=column in intervals.columns and _finite_series(intervals, column).notna().all(),
            severity="fatal",
            details=f"column={column}, missing_or_nan={0 if column in intervals.columns else len(intervals)}",
        )
    if {"reduction_q025", "reduction_median", "reduction_q975"}.issubset(intervals.columns):
        q025 = _finite_series(intervals, "reduction_q025")
        q500 = _finite_series(intervals, "reduction_median")
        q975 = _finite_series(intervals, "reduction_q975")
        ordered = q025.le(q500) & q500.le(q975)
        width = q975 - q025
        _add(
            rows,
            category="intervals",
            check="uncertainty_interval_quantiles_ordered",
            passed=bool(ordered.all()),
            severity="fatal",
            details=f"bad_rows={int((~ordered).sum())}",
        )
        _add(
            rows,
            category="intervals",
            check="uncertainty_interval_width_positive",
            passed=bool(width.gt(0).all()),
            severity="fatal",
            details=f"min_width={width.min(skipna=True)}, median_width={width.median(skipna=True)}",
        )
        very_narrow = width.lt(0.002)
        _add(
            rows,
            category="intervals",
            check="very_narrow_width_diagnostic",
            passed=not bool(very_narrow.all()),
            severity="warning",
            details=(
                f"rows_width_below_0.2_percentage_points={int(very_narrow.sum())}/{len(width)}, "
                f"median_width_percentage_points={100 * width.median(skipna=True):.3f}"
            ),
        )
        _add(
            rows,
            category="intervals",
            check="uncertainty_interval_width_distribution",
            passed=True,
            severity="informational",
            details=(
                f"min_width_pp={100 * width.min(skipna=True):.3f}, "
                f"q25_width_pp={100 * width.quantile(0.25):.3f}, "
                f"median_width_pp={100 * width.median(skipna=True):.3f}, "
                f"q75_width_pp={100 * width.quantile(0.75):.3f}, "
                f"max_width_pp={100 * width.max(skipna=True):.3f}"
            ),
        )
    if "uncertainty_draws" in intervals.columns:
        draws = _finite_series(intervals, "uncertainty_draws")
        _add(
            rows,
            category="intervals",
            check="posterior_draw_count",
            passed=bool(draws.ge(expected_draws).all()),
            severity="fatal",
            details=f"min_draws={draws.min(skipna=True)}, expected>={expected_draws}",
        )
    if "interval_type" in intervals.columns:
        types = intervals["interval_type"].astype(str)
        joint_expected = (
            str(expected_inference_structure).lower() == JOINT_INFERENCE_STRUCTURE
        )
        expected_phrase = (
            "95% joint posterior credible interval"
            if joint_expected
            else "conditional"
        )
        _add(
            rows,
            category="intervals",
            check="interval_type_matches_uncertainty_target",
            passed=bool(types.str.contains(expected_phrase, case=False, regex=False).all()),
            severity="fatal",
            details=f"types={sorted(types.unique())}, expected_phrase={expected_phrase}",
        )
    else:
        _add(
            rows,
            category="intervals",
            check="interval_type_matches_uncertainty_target",
            passed=False,
            severity="fatal",
            details="interval_type column is missing",
        )
    if "interval_basis" in intervals.columns:
        basis = intervals["interval_basis"].astype(str).str.lower()
        joint_expected = (
            str(expected_inference_structure).lower() == JOINT_INFERENCE_STRUCTURE
        )
        exact_state_expected = (
            str(expected_inference_structure).lower()
            == "reference_structure_state_space_exact_importance_cut"
        )
        if joint_expected:
            basis_matches = bool(
                basis.str.contains("full-feedback", regex=False).all()
                and basis.str.contains("posterior draw", regex=False).all()
                and basis.str.contains("updated jointly", regex=False).all()
                and not basis.str.contains(
                    r"conditional uncertainty|modular[ _-]?cut|external structural-prior",
                    regex=True,
                ).any()
            )
        elif exact_state_expected:
            basis_matches = bool(
                basis.str.contains("exact-target importance-corrected", regex=False).all()
                and basis.str.contains("external structural-prior", regex=False).all()
                and basis.str.contains("fixed", regex=False).all()
            )
        else:
            basis_matches = bool(basis.str.contains("conditional", regex=False).all())
        _add(
            rows,
            category="intervals",
            check="interval_basis_is_full_feedback_joint_posterior",
            passed=bool(basis_matches),
            severity="fatal",
            details=f"unique_basis={sorted(intervals['interval_basis'].astype(str).unique())}",
        )
    else:
        _add(
            rows,
            category="intervals",
            check="interval_basis_is_full_feedback_joint_posterior",
            passed=False,
            severity="fatal",
            details="interval_basis column is missing",
        )


def _interval_width_audit(intervals: pd.DataFrame) -> pd.DataFrame:
    required = {"country", "strategy", "reduction_q025", "reduction_median", "reduction_q975", "uncertainty_draws"}
    if not required.issubset(intervals.columns):
        return pd.DataFrame()
    out = intervals.loc[:, sorted(required)].copy()
    q025 = pd.to_numeric(out["reduction_q025"], errors="coerce")
    median = pd.to_numeric(out["reduction_median"], errors="coerce")
    q975 = pd.to_numeric(out["reduction_q975"], errors="coerce")
    width = q975 - q025
    out["reduction_q95_width"] = width
    out["reduction_q95_width_percentage_points"] = 100.0 * width
    out["relative_width_vs_abs_median"] = np.where(
        median.abs().gt(0),
        width / median.abs(),
        np.nan,
    )
    return out.sort_values(["country", "strategy"]).reset_index(drop=True)


def _draw_checks(
    rows: list[dict[str, Any]],
    draws: pd.DataFrame,
    *,
    expected_countries: set[str],
    expected_draws: int,
    expected_chains: int,
    intervention_bounds: dict[str, tuple[float, float]],
    require_structural_pairing: bool = False,
    expected_structural_draw_ids: int | None = None,
    require_annual_process_draws: bool = False,
) -> None:
    intervention_strategies = set(FIGURE2C_STRATEGIES) - {"current"}
    if require_annual_process_draws:
        configs = load_configs()
        start_year = int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_start_date"]).year
        )
        end_year = int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_end_date"]).year
        )
        expected_process_columns = {
            f"posterior_log_beta_process_{year}"
            for year in range(start_year, end_year + 1)
        }
        missing_process = expected_process_columns.difference(draws.columns)
        finite_process = bool(
            not missing_process
            and np.isfinite(
                draws.loc[:, sorted(expected_process_columns)]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            ).all()
        )
        _add(
            rows,
            category="paired_draws",
            check="forecast_process_draws_retained_and_finite",
            passed=finite_process,
            severity="fatal",
            details=f"missing={sorted(missing_process)}, expected_years={start_year}-{end_year}",
        )
    expected_rows = len(expected_countries) * len(intervention_strategies) * expected_draws
    _add(
        rows,
        category="paired_draws",
        check="expected_draw_rows",
        passed=len(draws) == expected_rows,
        severity="fatal",
        details=f"rows={len(draws)}, expected={expected_rows}",
    )
    countries = set(draws.get("country", pd.Series(dtype=str)).astype(str))
    strategies = set(draws.get("strategy", pd.Series(dtype=str)).astype(str))
    _add(
        rows,
        category="paired_draws",
        check="country_and_strategy_coverage",
        passed=countries == expected_countries and strategies == intervention_strategies,
        severity="fatal",
        details=f"countries={sorted(countries)}, strategies={sorted(strategies)}",
    )
    required_keys = {"country", "strategy", "posterior_draw"}
    if required_keys.issubset(draws.columns):
        duplicate_keys = draws.duplicated(["country", "strategy", "posterior_draw"])
        _add(
            rows,
            category="paired_draws",
            check="country_strategy_draw_keys_unique",
            passed=not bool(duplicate_keys.any()),
            severity="fatal",
            details=f"duplicate_rows={int(duplicate_keys.sum())}",
        )
        per_combo = (
            draws.groupby(["country", "strategy"], dropna=False)["posterior_draw"]
            .agg(rows="size", unique_draws="nunique")
            .reset_index()
        )
        complete = per_combo["rows"].eq(expected_draws) & per_combo["unique_draws"].eq(expected_draws)
        _add(
            rows,
            category="paired_draws",
            check="draw_count_per_country_strategy",
            passed=bool(len(per_combo) == len(expected_countries) * len(intervention_strategies) and complete.all()),
            severity="fatal",
            details=(
                "all country-strategy combinations have expected draws"
                if bool(len(per_combo) == len(expected_countries) * len(intervention_strategies) and complete.all())
                else per_combo.loc[~complete].head(20).to_dict("records").__repr__()
            ),
        )
    structural_present = "structural_draw_id" in draws.columns
    _add(
        rows,
        category="paired_draws",
        check="structural_draw_id_retained",
        passed=structural_present or not require_structural_pairing,
        severity="fatal",
        details=f"required={require_structural_pairing}, present={structural_present}",
    )
    if structural_present:
        structural_ids = pd.to_numeric(draws["structural_draw_id"], errors="coerce")
        valid_ids = bool(
            structural_ids.notna().all()
            and np.isfinite(structural_ids.to_numpy(dtype=float)).all()
            and np.equal(structural_ids, np.floor(structural_ids)).all()
        )
        _add(
            rows,
            category="paired_draws",
            check="structural_draw_id_finite_integer",
            passed=valid_ids,
            severity="fatal",
            details=f"invalid_rows={int(structural_ids.isna().sum())}",
        )
        if valid_ids and "posterior_draw" in draws.columns:
            structural_ids = structural_ids.astype(np.int64)
            draw_numbers = pd.to_numeric(draws["posterior_draw"], errors="coerce")
            alignment = pd.DataFrame(
                {"posterior_draw": draw_numbers, "structural_draw_id": structural_ids}
            ).groupby("posterior_draw", dropna=False)["structural_draw_id"].nunique()
            paired = bool(alignment.eq(1).all())
            _add(
                rows,
                category="paired_draws",
                check="structural_draw_id_paired_across_countries",
                passed=paired,
                severity="fatal",
                details=f"mispaired_posterior_draws={int((~alignment.eq(1)).sum())}",
            )
            draw_map = pd.DataFrame(
                {"posterior_draw": draw_numbers, "structural_draw_id": structural_ids}
            ).drop_duplicates()
            structural_counts = draw_map["structural_draw_id"].value_counts()
            balanced = bool(
                not structural_counts.empty
                and int(structural_counts.max() - structural_counts.min()) <= 1
            )
            _add(
                rows,
                category="paired_draws",
                check="structural_draw_selection_balanced",
                passed=balanced,
                severity="fatal",
                details=(
                    f"selected_ids={len(structural_counts)}, "
                    f"min_repeats={structural_counts.min() if not structural_counts.empty else 'missing'}, "
                    f"max_repeats={structural_counts.max() if not structural_counts.empty else 'missing'}"
                ),
            )
            if expected_structural_draw_ids is not None:
                _add(
                    rows,
                    category="paired_draws",
                    check="structural_draw_selection_covers_expected_ids",
                    passed=len(structural_counts) == int(expected_structural_draw_ids),
                    severity="fatal",
                    details=(
                        f"selected_ids={len(structural_counts)}, "
                        f"expected={int(expected_structural_draw_ids)}"
                    ),
                )

            inconsistent: list[str] = []
            for parameter in EXTERNAL_STRUCTURAL_PRIOR_PARAMETERS:
                column = f"posterior_{parameter}"
                if column not in draws.columns:
                    inconsistent.append(f"{column}:missing")
                    continue
                values = pd.to_numeric(draws[column], errors="coerce")
                if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                    inconsistent.append(f"{column}:nonfinite")
                    continue
                grouped = pd.DataFrame(
                    {"structural_draw_id": structural_ids, "value": values}
                ).groupby("structural_draw_id")["value"]
                minima = grouped.min()
                maxima = grouped.max()
                tolerance = 1e-10 + 1e-10 * np.maximum(minima.abs(), maxima.abs())
                if ((maxima - minima) > tolerance).any():
                    inconsistent.append(f"{column}:inconsistent")
            _add(
                rows,
                category="paired_draws",
                check="structural_posterior_values_shared_across_countries",
                passed=not inconsistent,
                severity="fatal",
                details="all shared" if not inconsistent else ", ".join(inconsistent),
            )
        else:
            _add(
                rows,
                category="paired_draws",
                check="structural_draw_id_paired_across_countries",
                passed=False,
                severity="fatal",
                details="invalid structural_draw_id or missing posterior_draw",
            )
    if "posterior_chain" in draws.columns and {"country", "strategy"}.issubset(draws.columns):
        chain_coverage = (
            draws.groupby(["country", "strategy"], dropna=False)["posterior_chain"]
            .nunique(dropna=True)
            .reset_index(name="n_chains")
        )
        complete = chain_coverage["n_chains"].ge(expected_chains)
        _add(
            rows,
            category="paired_draws",
            check="selected_draws_cover_all_posterior_chains",
            passed=bool(len(chain_coverage) == len(expected_countries) * len(intervention_strategies) and complete.all()),
            severity="fatal",
            details=(
                "all country-strategy combinations cover all posterior chains"
                if bool(len(chain_coverage) == len(expected_countries) * len(intervention_strategies) and complete.all())
                else chain_coverage.loc[~complete].head(20).to_dict("records").__repr__()
            ),
        )
    else:
        _add(
            rows,
            category="paired_draws",
            check="selected_draws_cover_all_posterior_chains",
            passed=False,
            severity="fatal",
            details="posterior_chain, country, or strategy column missing",
        )
    for column in ("current_rate", "intervention_rate"):
        if column not in draws.columns:
            _add(
                rows,
                category="paired_draws",
                check=f"{column}_present",
                passed=False,
                severity="fatal",
                details="missing",
            )
            continue
        values = _finite_series(draws, column)
        nonnegative = values.ge(0).all()
        positive_if_current = values.gt(0).all() if column == "current_rate" else nonnegative
        _add(
            rows,
            category="paired_draws",
            check=f"{column}_finite_nonnegative",
            passed=bool(values.notna().all() and positive_if_current),
            severity="fatal",
            details=f"nan={int(values.isna().sum())}, min={values.min(skipna=True)}, max={values.max(skipna=True)}",
        )
    if PRIMARY_REDUCTION in draws.columns:
        values = _finite_series(draws, PRIMARY_REDUCTION)
        _add(
            rows,
            category="paired_draws",
            check="relative_reduction_finite",
            passed=bool(values.notna().all()),
            severity="fatal",
            details=f"nan={int(values.isna().sum())}, min={values.min(skipna=True)}, max={values.max(skipna=True)}",
        )
        implausible = values.lt(-10.0) | values.gt(1.0)
        _add(
            rows,
            category="paired_draws",
            check="relative_reduction_plausible_range",
            passed=not bool(implausible.any()),
            severity="warning",
            details=f"outside_-10_to_1={int(implausible.sum())}/{len(values)}",
        )
    missing_posterior = [f"posterior_{name}" for name in JOINT_POSTERIOR_PARAMETERS if f"posterior_{name}" not in draws.columns]
    _add(
        rows,
        category="paired_draws",
        check="posterior_parameter_columns_retained",
        passed=not missing_posterior,
        severity="fatal",
        details=f"missing={missing_posterior}",
    )
    if not missing_posterior and "country" in draws.columns:
        selected_audit_rows: list[dict[str, Any]] = []
        for country, group in draws.groupby("country", sort=True):
            for name in JOINT_POSTERIOR_PARAMETERS:
                column = f"posterior_{name}"
                values = _finite_series(group, column).dropna()
                q025 = float(values.quantile(0.025)) if not values.empty else np.nan
                q975 = float(values.quantile(0.975)) if not values.empty else np.nan
                width = q975 - q025 if np.isfinite(q025) and np.isfinite(q975) else np.nan
                selected_audit_rows.append(
                    {
                        "country": country,
                        "parameter": name,
                        "n_unique": int(values.nunique(dropna=True)),
                        "q95_width": float(width),
                    }
                )
        selected_audit = pd.DataFrame(selected_audit_rows)
        nonvariable = selected_audit.loc[selected_audit["n_unique"].le(1)]
        _add(
            rows,
            category="paired_draws",
            check="selected_posterior_parameters_vary_by_country",
            passed=nonvariable.empty,
            severity="fatal",
            details=(
                "all selected country-parameter combinations vary"
                if nonvariable.empty
                else nonvariable.head(20).to_dict("records").__repr__()
            ),
        )
    missing_intervention = [
        f"{INTERVENTION_UNCERTAINTY_PREFIX}{name}"
        for name in REQUIRED_INTERVENTION_UNCERTAINTY_COLUMNS
        if f"{INTERVENTION_UNCERTAINTY_PREFIX}{name}" not in draws.columns
    ]
    _add(
        rows,
        category="paired_draws",
        check="intervention_uncertainty_columns_retained",
        passed=not missing_intervention,
        severity="fatal",
        details=f"missing={missing_intervention}",
    )
    for name in REQUIRED_INTERVENTION_UNCERTAINTY_COLUMNS:
        column = f"{INTERVENTION_UNCERTAINTY_PREFIX}{name}"
        if column not in draws.columns:
            continue
        values = _finite_series(draws, column).dropna()
        _add(
            rows,
            category="paired_draws",
            check=f"{column}_varies",
            passed=values.nunique(dropna=True) > 1,
            severity="fatal",
            details=f"n_unique={values.nunique(dropna=True)}, q025={values.quantile(0.025)}, q975={values.quantile(0.975)}",
        )
        low, high = intervention_bounds.get(column, (-np.inf, np.inf))
        in_bounds = values.ge(low - 1e-12) & values.le(high + 1e-12)
        _add(
            rows,
            category="paired_draws",
            check=f"{column}_within_configured_bounds",
            passed=bool(in_bounds.all()),
            severity="fatal",
            details=f"bounds=[{low}, {high}], min={values.min(skipna=True)}, max={values.max(skipna=True)}",
        )


def _interval_matches_draw_quantiles(
    rows: list[dict[str, Any]],
    intervals: pd.DataFrame,
    draws: pd.DataFrame,
) -> None:
    required_interval = {"country", "strategy", "reduction_q025", "reduction_median", "reduction_q975", "uncertainty_draws"}
    required_draw = {"country", "strategy", PRIMARY_REDUCTION, "posterior_draw"}
    missing_interval = required_interval.difference(intervals.columns)
    missing_draw = required_draw.difference(draws.columns)
    _add(
        rows,
        category="cross_file_consistency",
        check="interval_and_draw_columns_present",
        passed=not missing_interval and not missing_draw,
        severity="fatal",
        details=f"missing_interval={sorted(missing_interval)}, missing_draw={sorted(missing_draw)}",
    )
    if missing_interval or missing_draw:
        return
    recomputed = (
        draws.groupby(["country", "strategy"], sort=True)
        .agg(
            reduction_q025=(PRIMARY_REDUCTION, lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.025))),
            reduction_median=(PRIMARY_REDUCTION, lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.500))),
            reduction_q975=(PRIMARY_REDUCTION, lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.975))),
            uncertainty_draws=("posterior_draw", "nunique"),
        )
        .reset_index()
    )
    merged = intervals.merge(recomputed, on=["country", "strategy"], suffixes=("_file", "_draws"), how="outer", indicator=True)
    key_match = merged["_merge"].eq("both").all()
    numeric_match = key_match
    max_abs_diff = 0.0
    if key_match:
        for column in ("reduction_q025", "reduction_median", "reduction_q975"):
            diff = (
                pd.to_numeric(merged[f"{column}_file"], errors="coerce")
                - pd.to_numeric(merged[f"{column}_draws"], errors="coerce")
            ).abs()
            max_abs_diff = max(max_abs_diff, float(diff.max(skipna=True)))
            numeric_match = numeric_match and bool(diff.le(1e-10).all())
        draw_diff = (
            pd.to_numeric(merged["uncertainty_draws_file"], errors="coerce")
            - pd.to_numeric(merged["uncertainty_draws_draws"], errors="coerce")
        ).abs()
        numeric_match = numeric_match and bool(draw_diff.eq(0).all())
    _add(
        rows,
        category="cross_file_consistency",
        check="interval_quantiles_recomputed_from_draws",
        passed=bool(key_match and numeric_match),
        severity="fatal",
        details=(
            f"rows={len(merged)}, keys_match={key_match}, "
            f"max_abs_quantile_diff={max_abs_diff:.3g}"
        ),
    )


def _read_metadata_if_available(
    rows: list[dict[str, Any]],
    *,
    stem: str,
    category: str,
) -> dict[str, Any] | None:
    try:
        metadata = validate_run_metadata(stem)
    except (FileNotFoundError, ValueError) as exc:
        _add(
            rows,
            category=category,
            check="run_metadata_file_exists",
            passed=False,
            severity="fatal",
            details=str(exc),
        )
        return None
    _add(
        rows,
        category=category,
        check="run_metadata_file_exists",
        passed=True,
        severity="fatal",
        details=f"stem={stem}",
    )
    return metadata


def _file_exists_check(
    rows: list[dict[str, Any]],
    *,
    category: str,
    check: str,
    path: Path,
) -> bool:
    exists = path.exists()
    _add(
        rows,
        category=category,
        check=check,
        passed=exists,
        severity="fatal",
        details=str(path),
    )
    return exists


def main(
    *,
    expected_draws: int = 200,
    expected_chains: int = 10,
    fail_on_warnings: bool = True,
    posterior_stem: str = POSTERIOR_STEM,
) -> pd.DataFrame:
    # Warning failures cannot be downgraded by a direct Python call. The
    # argument is retained only for compatibility with older research
    # orchestration code; passing this audit never grants publication status.
    warnings_are_fatal = True
    print(
        "LEGACY NONPUBLICATION RESEARCH AUDIT: this route is not a Figure 2c "
        "interval source."
    )
    if posterior_stem in RETIRED_POSTERIOR_STEMS:
        raise ValueError(
            f"Posterior stem {posterior_stem!r} is retired because it mislabeled "
            f"conditional uncertainty; use {POSTERIOR_STEM!r}."
        )
    configured_country_list = publication_country_names(load_configs())
    rows: list[dict[str, Any]] = []
    posterior_sample_path = project_path(
        "outputs",
        "simulations",
        f"{_artifact_stem(posterior_stem, 'bayesian_posterior_samples', 'posterior_samples')}.parquet",
    )
    structural_importance_path = project_path(
        "outputs", "diagnostics", f"{posterior_stem}_structural_importance_audit.csv"
    )
    local_importance_path = project_path(
        "outputs", "diagnostics", f"{posterior_stem}_local_state_importance_audit.csv"
    )
    metadata = _read_metadata_if_available(
        rows,
        stem=posterior_stem,
        category="posterior_metadata",
    )
    is_joint_smc = bool(
        metadata
        and str(metadata.get("sampler", "")).lower() == JOINT_SAMPLING_METHOD
    )
    figure_metadata = _read_metadata_if_available(
        rows,
        stem=FIGURE2C_STEM,
        category="figure2c_metadata",
    ) or {}
    _add(
        rows,
        category="figure2c_metadata",
        check="quality_bypass_disabled",
        passed=bool(figure_metadata)
        and figure_metadata.get("allow_conditional_posterior") is False,
        severity="fatal",
        details=(
            "allow_conditional_posterior="
            f"{figure_metadata.get('allow_conditional_posterior', 'missing')}"
        ),
    )
    configs = load_configs()
    configured_countries = set(configured_country_list)
    expected_inference_structure = JOINT_INFERENCE_STRUCTURE
    require_structural_pairing = True
    intervention_bounds = _intervention_uncertainty_bounds()
    expected_countries = configured_countries

    posterior_samples_exist = _file_exists_check(
        rows,
        category="posterior_samples",
        check="posterior_sample_file_exists",
        path=posterior_sample_path,
    )
    if is_joint_smc:
        structural_importance_exists = False
        local_importance_exists = False
        smc_stage_exists = _file_exists_check(
            rows,
            category="joint_smc",
            check="smc_stage_audit_file_exists",
            path=project_path(
                "outputs", "diagnostics", f"{posterior_stem}_smc_stage_audit.csv"
            ),
        )
        smc_diagnostics_exists = _file_exists_check(
            rows,
            category="joint_smc",
            check="smc_convergence_diagnostics_file_exists",
            path=project_path(
                "outputs", "summaries", f"{posterior_stem}_convergence_diagnostics.csv"
            ),
        )
        smc_boundary_exists = _file_exists_check(
            rows,
            category="joint_smc",
            check="smc_boundary_audit_file_exists",
            path=project_path(
                "outputs", "diagnostics", f"{posterior_stem}_smc_boundary_audit.csv"
            ),
        )
    else:
        structural_importance_exists = _file_exists_check(
            rows,
            category="joint_importance",
            check="structural_importance_file_exists",
            path=structural_importance_path,
        )
        local_importance_exists = _file_exists_check(
            rows,
            category="joint_importance",
            check="local_state_importance_file_exists",
            path=local_importance_path,
        )
        smc_stage_exists = False
        smc_diagnostics_exists = False
        smc_boundary_exists = False
    interval_exists = _file_exists_check(
        rows,
        category="intervals",
        check="interval_file_exists",
        path=INTERVAL_PATH,
    )
    draw_exists = _file_exists_check(
        rows,
        category="paired_draws",
        check="draw_file_exists",
        path=DRAW_PATH,
    )

    for digest_field, path, exists in (
        ("posterior_samples_sha256", posterior_sample_path, posterior_samples_exist),
        ("paired_draws_sha256", DRAW_PATH, draw_exists),
        ("paired_intervals_sha256", INTERVAL_PATH, interval_exists),
    ):
        recorded_digest = str(figure_metadata.get(digest_field, ""))
        observed_digest = file_sha256(path) if exists else ""
        _add(
            rows,
            category="artifact_provenance",
            check=f"{digest_field}_matches_current_artifact",
            passed=bool(recorded_digest)
            and bool(observed_digest)
            and recorded_digest == observed_digest,
            severity="fatal",
            details=(
                f"recorded={recorded_digest or 'missing'}, "
                f"observed={observed_digest or 'missing'}, path={path}"
            ),
        )

    posterior_diagnostic_artifacts = (
        (
            "smc_stage_audit_sha256",
            project_path(
                "outputs", "diagnostics", f"{posterior_stem}_smc_stage_audit.csv"
            ),
            smc_stage_exists,
        ),
        (
            "convergence_diagnostics_sha256",
            project_path(
                "outputs", "summaries", f"{posterior_stem}_convergence_diagnostics.csv"
            ),
            smc_diagnostics_exists,
        ),
        (
            "smc_boundary_audit_sha256",
            project_path(
                "outputs", "diagnostics", f"{posterior_stem}_smc_boundary_audit.csv"
            ),
            smc_boundary_exists,
        ),
    ) if is_joint_smc else (
        (
            "structural_importance_audit_sha256",
            structural_importance_path,
            structural_importance_exists,
        ),
        (
            "local_state_importance_audit_sha256",
            local_importance_path,
            local_importance_exists,
        ),
    )
    for digest_field, path, exists in posterior_diagnostic_artifacts:
        recorded_digest = str((metadata or {}).get(digest_field, ""))
        observed_digest = file_sha256(path) if exists else ""
        _add(
            rows,
            category="artifact_provenance",
            check=f"{digest_field}_matches_current_artifact",
            passed=bool(recorded_digest)
            and bool(observed_digest)
            and recorded_digest == observed_digest,
            severity="fatal",
            details=(
                f"recorded={recorded_digest or 'missing'}, "
                f"observed={observed_digest or 'missing'}, path={path}"
            ),
        )

    if metadata is not None:
        _posterior_metadata_checks(rows, metadata)
        _posterior_runtime_checks(rows, metadata, expected_chains=expected_chains)
    if figure_metadata is not None:
        _figure_metadata_checks(
            rows,
            figure_metadata,
            configured_countries=configured_countries,
            expected_draws=expected_draws,
            expected_inference_structure=expected_inference_structure,
            expected_posterior_stem=posterior_stem,
            expected_posterior_path=posterior_sample_path,
        )

    parameter_audit = pd.DataFrame()
    expected_structural_draw_ids: int | None = None
    if posterior_samples_exist:
        samples = pd.read_parquet(posterior_sample_path)
        if require_structural_pairing and "structural_draw_id" in samples.columns:
            source_ids = pd.to_numeric(samples["structural_draw_id"], errors="coerce").dropna()
            expected_structural_draw_ids = min(int(source_ids.nunique()), int(expected_draws))
        expected_draws_per_chain = int((metadata or {}).get("draws_per_chain") or 0)
        require_annual_process_draws = bool(
            str((metadata or {}).get("sampler", "")).lower()
            in {CONDITIONAL_STATE_SPACE_SAMPLER, JOINT_SAMPLING_METHOD}
        )
        parameter_audit = _posterior_sample_checks(
            rows,
            samples,
            expected_countries=expected_countries,
            expected_chains=expected_chains,
            expected_draws_per_chain=expected_draws_per_chain,
            require_structural_pairing=require_structural_pairing,
            require_annual_process_draws=require_annual_process_draws,
        )
        if not parameter_audit.empty:
            _posterior_parameter_width_checks(rows, parameter_audit)
    if metadata is not None and structural_importance_exists and local_importance_exists:
        _joint_importance_checks(
            rows,
            metadata,
            pd.read_csv(structural_importance_path),
            pd.read_csv(local_importance_path),
            expected_countries=expected_countries,
        )
    if (
        metadata is not None
        and smc_stage_exists
        and smc_diagnostics_exists
        and smc_boundary_exists
    ):
        _joint_smc_checks(
            rows,
            metadata,
            pd.read_csv(
                project_path(
                    "outputs", "diagnostics", f"{posterior_stem}_smc_stage_audit.csv"
                )
            ),
            pd.read_csv(
                project_path(
                    "outputs", "summaries", f"{posterior_stem}_convergence_diagnostics.csv"
                )
            ),
            pd.read_csv(
                project_path(
                    "outputs", "diagnostics", f"{posterior_stem}_smc_boundary_audit.csv"
                )
            ),
        )
    intervals: pd.DataFrame | None = None
    draws: pd.DataFrame | None = None
    if interval_exists:
        intervals = pd.read_csv(INTERVAL_PATH)
        _interval_checks(
            rows,
            intervals,
            expected_countries=expected_countries,
            expected_draws=expected_draws,
            expected_inference_structure=expected_inference_structure,
        )
    if draw_exists:
        draws = pd.read_csv(DRAW_PATH)
        _draw_checks(
            rows,
            draws,
            expected_countries=expected_countries,
            expected_draws=expected_draws,
            expected_chains=expected_chains,
            intervention_bounds=intervention_bounds,
            require_structural_pairing=require_structural_pairing,
            expected_structural_draw_ids=expected_structural_draw_ids,
            require_annual_process_draws=bool(
                str((metadata or {}).get("sampler", "")).lower()
                in {CONDITIONAL_STATE_SPACE_SAMPLER, JOINT_SAMPLING_METHOD}
            ),
        )
    if intervals is not None and draws is not None:
        _interval_matches_draw_quantiles(rows, intervals, draws)

    audit = pd.DataFrame(rows)
    write_dataframe(audit, OUTPUT_PATH)
    if not parameter_audit.empty:
        for suffix in (".csv", ".parquet"):
            project_path(
                "outputs",
                "tables",
                f"{AUDIT_STEM}_posterior_parameter_widths{suffix}",
            ).unlink(missing_ok=True)
        write_dataframe(parameter_audit, project_path("outputs", "tables", f"{AUDIT_STEM}_uncertainty_parameter_widths.csv"))
    if intervals is not None:
        interval_width_audit = _interval_width_audit(intervals)
        if not interval_width_audit.empty:
            write_dataframe(interval_width_audit, project_path("outputs", "tables", f"{AUDIT_STEM}_interval_widths.csv"))
    fatal_failures = audit.loc[audit["severity"].eq("fatal") & audit["status"].eq("fail")]
    warning_failures = audit.loc[audit["severity"].eq("warning") & audit["status"].eq("fail")]
    write_run_metadata(
        AUDIT_STEM,
        current_run_metadata(
            AUDIT_STEM,
            row_counts={
                "audit_checks": int(len(audit)),
                "fatal_failures": int(len(fatal_failures)),
                "warning_failures": int(len(warning_failures)),
            },
        )
        | {
            "analysis_role": ANALYSIS_ROLE,
            "publication_path": PUBLICATION_PATH,
            "figure2c_interval_source": FIGURE2C_INTERVAL_SOURCE,
            "posterior_stem": posterior_stem,
            "legacy_compatibility_artifact_names": True,
            "figure2c_stem": FIGURE2C_STEM,
            "expected_draws_per_country_strategy": int(expected_draws),
            "expected_chains_per_country": int(expected_chains),
            "passed": bool(fatal_failures.empty and warning_failures.empty),
            "warnings_are_fatal": warnings_are_fatal,
            "audit_table_sha256": file_sha256(OUTPUT_PATH),
            "audited_artifact_sha256": {
                "posterior_samples_sha256": figure_metadata.get(
                    "posterior_samples_sha256"
                ),
                "paired_draws_sha256": figure_metadata.get("paired_draws_sha256"),
                "paired_intervals_sha256": figure_metadata.get(
                    "paired_intervals_sha256"
                ),
                "smc_stage_audit_sha256": (metadata or {}).get(
                    "smc_stage_audit_sha256"
                ),
                "convergence_diagnostics_sha256": (metadata or {}).get(
                    "convergence_diagnostics_sha256"
                ),
                "smc_boundary_audit_sha256": (metadata or {}).get(
                    "smc_boundary_audit_sha256"
                ),
                "structural_importance_audit_sha256": (metadata or {}).get(
                    "structural_importance_audit_sha256"
                ),
                "local_state_importance_audit_sha256": (metadata or {}).get(
                    "local_state_importance_audit_sha256"
                ),
            },
        },
    )
    if not fatal_failures.empty or not warning_failures.empty:
        print(audit.to_string(index=False))
        raise SystemExit(1)
    print(audit.to_string(index=False))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Audit optional legacy nonpublication paired-CrI research outputs. "
            "Historical filenames are compatibility-only; this is not a Figure 2c source."
        )
    )
    parser.add_argument("--expected-draws", type=int, default=200)
    parser.add_argument("--expected-chains", type=int, default=10)
    parser.add_argument("--posterior-stem", default=POSTERIOR_STEM)
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()
    main(
        expected_draws=args.expected_draws,
        expected_chains=args.expected_chains,
        fail_on_warnings=args.fail_on_warnings,
        posterior_stem=args.posterior_stem,
    )
