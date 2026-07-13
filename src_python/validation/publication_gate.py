"""Reusable fail-closed checks for predictive and Figure 2 release gates."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    file_sha256,
    load_configs,
    publication_country_names,
    validate_run_metadata,
)
from src_python.utils.io import project_path
from src_python.validation.run_panel_pomp_hindcast import (
    PREQUENTIAL_RUN_STEM,
)


PREDICTIVE_GATE_PATH = project_path(
    "outputs",
    "tables",
    "panel_pomp_rolling_hindcast_gate.csv",
)
PREDICTIVE_FOLD_PATH = project_path(
    "outputs", "tables", "panel_pomp_rolling_hindcast_folds.csv"
)
PREDICTIVE_INTERVAL_PATH = project_path(
    "outputs", "tables", "panel_pomp_rolling_hindcast_intervals.csv"
)
PREDICTIVE_INPUT_PATH = project_path(
    "data", "processed", "pertussis_incidence_timeseries.csv"
)
FIGURE2_AUDIT_STEM = "figure2c_conditional_quality_audit"
FIGURE2_AUDIT_PATH = project_path(
    "outputs", "tables", f"{FIGURE2_AUDIT_STEM}.csv"
)
FIGURE2_POSTERIOR_STEM = "bayesian_uncertainty_figure2c_conditional"
FIGURE2_SOURCE_STEM = "figure2c_paired_programme_uncertainty"
FIGURE2_AUDITED_ARTIFACT_PATHS = {
    "posterior_samples_sha256": project_path(
        "outputs",
        "simulations",
        f"{FIGURE2_POSTERIOR_STEM}_posterior_samples.parquet",
    ),
    "paired_draws_sha256": project_path(
        "outputs", "tables", "figure2c_programme_paired_conditional_interval_draws.csv"
    ),
    "paired_intervals_sha256": project_path(
        "outputs", "summaries", "figure2c_programme_paired_conditional_intervals.csv"
    ),
}
FIGURE2_REQUIRED_AUDIT_CHECKS = frozenset(
    {
        "quality_bypass_disabled",
        "state_space_exact_importance_quality_columns_present",
        "state_space_exact_importance_all_parameters_recommended",
        "exact_state_importance_ess_meets_recommended",
        "exact_state_importance_max_weight_meets_recommended",
        "exact_state_importance_pareto_k_meets_recommended",
        "exact_state_importance_tail_support_meets_recommended",
        "interval_type_matches_uncertainty_target",
        "interval_basis_is_conditional_not_joint",
        "expected_country_strategy_rows",
        "country_and_strategy_coverage",
    }
)


def _strict_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Expected a complete Boolean value, received {value!r}")


def _strict_positive_integer(value: Any, *, name: str) -> int:
    """Parse an artifact count without silently truncating fractional values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, received {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer, received {value!r}"
        ) from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer, received {value!r}")
    return int(numeric)


def _legacy_predictive_publication_gate_failures(
    gate: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    expected_countries: Sequence[str],
) -> list[str]:
    """Return reasons why a hindcast artifact is not a publication parent."""

    failures: list[str] = []
    required_gate_columns = {
        "execution_gate_pass",
        "predictive_adequacy_pass",
        "publication_gate_pass",
        "expected_fold_count",
        "completed_fold_count",
        "publication_requires_predictive_adequacy",
        "state_coordinate_uncertainty_mode",
        "predictive_uncertainty_scope",
        "posterior_coverage_claimed",
    }
    missing = sorted(required_gate_columns.difference(gate.columns))
    if missing:
        return [f"predictive gate is missing columns: {missing}"]
    if len(gate) != 1:
        return [f"predictive gate must contain exactly one row, found {len(gate)}"]
    row = gate.iloc[0]
    for column in (
        "execution_gate_pass",
        "predictive_adequacy_pass",
        "publication_gate_pass",
    ):
        try:
            passed = _strict_boolean(row[column])
        except ValueError as exc:
            failures.append(f"{column}: {exc}")
        else:
            if not passed:
                failures.append(f"{column}=false")
    try:
        posterior_claimed = _strict_boolean(row["posterior_coverage_claimed"])
    except ValueError as exc:
        failures.append(f"posterior_coverage_claimed: {exc}")
    else:
        if posterior_claimed:
            failures.append("conditional hindcast is mislabelled as posterior coverage")
    try:
        publication_requires_adequacy = _strict_boolean(
            row["publication_requires_predictive_adequacy"]
        )
    except ValueError as exc:
        failures.append(f"publication_requires_predictive_adequacy: {exc}")
    else:
        if not publication_requires_adequacy:
            failures.append("publication gate does not require predictive adequacy")
    if str(row["state_coordinate_uncertainty_mode"]) != PUBLICATION_STATE_UNCERTAINTY_MODE:
        failures.append("predictive gate did not use map_conditioned state coordinates")
    if str(row["predictive_uncertainty_scope"]) != (
        PUBLICATION_PREDICTIVE_UNCERTAINTY_SCOPE
    ):
        failures.append("predictive gate has the wrong uncertainty scope")
    try:
        expected_folds = _strict_positive_integer(
            row["expected_fold_count"], name="expected_fold_count"
        )
        completed_folds = _strict_positive_integer(
            row["completed_fold_count"], name="completed_fold_count"
        )
    except ValueError as exc:
        failures.append(str(exc))
        expected_folds = completed_folds = None
    if (
        expected_folds is not None
        and completed_folds is not None
        and expected_folds != completed_folds
    ):
        failures.append(
            f"predictive folds incomplete: completed={completed_folds}, expected={expected_folds}"
        )

    expected = list(expected_countries)
    observed = [str(value) for value in metadata.get("countries_included", [])]
    if observed != expected:
        failures.append(
            f"predictive gate countries={observed} do not match publication countries={expected}"
        )
    for key in (
        "execution_gate_pass",
        "predictive_adequacy_pass",
        "publication_gate_pass",
    ):
        try:
            metadata_passed = _strict_boolean(metadata.get(key))
        except ValueError as exc:
            failures.append(f"metadata {key}: {exc}")
        else:
            if not metadata_passed:
                failures.append(f"predictive run metadata has {key}=false")
    if str(metadata.get("state_coordinate_uncertainty_mode", "")) != (
        PUBLICATION_STATE_UNCERTAINTY_MODE
    ):
        failures.append("predictive run metadata has the wrong state uncertainty mode")
    if str(metadata.get("predictive_uncertainty_scope", "")) != (
        PUBLICATION_PREDICTIVE_UNCERTAINTY_SCOPE
    ):
        failures.append("predictive run metadata has the wrong uncertainty scope")
    try:
        metadata_posterior_claim = _strict_boolean(
            metadata.get("posterior_coverage_claimed")
        )
    except ValueError as exc:
        failures.append(f"metadata posterior_coverage_claimed: {exc}")
    else:
        if metadata_posterior_claim:
            failures.append("predictive metadata claims posterior coverage")

    metadata_expected_raw = metadata.get("expected_fold_counts")
    if not isinstance(metadata_expected_raw, dict):
        failures.append("predictive metadata is missing expected_fold_counts")
    else:
        metadata_expected: dict[str, int] = {}
        for country, value in metadata_expected_raw.items():
            try:
                metadata_expected[str(country)] = _strict_positive_integer(
                    value,
                    name=f"expected_fold_counts[{country}]",
                )
            except ValueError as exc:
                failures.append(str(exc))
        if set(metadata_expected) != set(expected):
            failures.append(
                "predictive metadata fold-count countries do not match publication countries"
            )
        elif expected_folds is not None and sum(metadata_expected.values()) != expected_folds:
            failures.append(
                "predictive gate expected_fold_count does not match metadata country folds"
            )

    fold_years = metadata.get("fold_test_years")
    if not isinstance(fold_years, dict):
        failures.append("predictive metadata is missing fold_test_years")
    elif isinstance(metadata_expected_raw, dict):
        for country in expected:
            years = fold_years.get(country)
            expected_count = metadata_expected_raw.get(country)
            try:
                expected_count_int = _strict_positive_integer(
                    expected_count,
                    name=f"expected_fold_counts[{country}]",
                )
            except ValueError:
                continue
            if (
                not isinstance(years, list)
                or len(years) != expected_count_int
                or len(set(years)) != len(years)
            ):
                failures.append(
                    f"predictive metadata fold_test_years are incomplete for {country}"
                )

    row_counts = metadata.get("row_counts")
    if not isinstance(row_counts, dict):
        failures.append("predictive metadata is missing row_counts")
    else:
        try:
            metadata_fold_rows = _strict_positive_integer(
                row_counts.get("fold_scores"), name="row_counts.fold_scores"
            )
        except ValueError as exc:
            failures.append(str(exc))
        else:
            if completed_folds is not None and metadata_fold_rows != completed_folds:
                failures.append(
                    "predictive gate completed_fold_count does not match metadata row count"
                )
    return failures


def predictive_publication_gate_failures(
    gate: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    expected_countries: Sequence[str],
) -> list[str]:
    """Validate the canonical prequential panel-POMP publication parent."""

    failures: list[str] = []
    required_gate_columns = {
        "execution_complete",
        "data_integrity_gate_pass",
        "countries_completed",
        "minimum_folds_per_country",
        "model_beats_all_baselines_log_score",
        "primary_predictive_95_coverage",
        "primary_predictive_mean_log1p_width",
        "primary_predictive_scope",
        "predictive_width_gate_pass",
        "mc_replicates",
        "monte_carlo_stability_pass",
        "predictive_gate_pass",
        "publication_gate_pass",
        "claim_scope",
        "full_compartment_pomp_claimed",
        "posterior_parameter_uncertainty_claimed",
    }
    missing = sorted(required_gate_columns.difference(gate.columns))
    if missing:
        return [f"predictive gate is missing columns: {missing}"]
    if len(gate) != 1:
        return [f"predictive gate must contain exactly one row, found {len(gate)}"]
    row = gate.iloc[0]
    for column in (
        "execution_complete",
        "data_integrity_gate_pass",
        "model_beats_all_baselines_log_score",
        "predictive_width_gate_pass",
        "monte_carlo_stability_pass",
        "predictive_gate_pass",
        "publication_gate_pass",
    ):
        try:
            passed = _strict_boolean(row[column])
        except ValueError as exc:
            failures.append(f"{column}: {exc}")
        else:
            if not passed:
                failures.append(f"{column}=false")
    for column in (
        "full_compartment_pomp_claimed",
        "posterior_parameter_uncertainty_claimed",
    ):
        try:
            claimed = _strict_boolean(row[column])
        except ValueError as exc:
            failures.append(f"{column}: {exc}")
        else:
            if claimed:
                failures.append(f"{column}=true")

    expected = list(expected_countries)
    try:
        completed_countries = _strict_positive_integer(
            row["countries_completed"], name="countries_completed"
        )
        minimum_folds = _strict_positive_integer(
            row["minimum_folds_per_country"], name="minimum_folds_per_country"
        )
        mc_replicates = _strict_positive_integer(
            row["mc_replicates"], name="mc_replicates"
        )
    except ValueError as exc:
        failures.append(str(exc))
        completed_countries = minimum_folds = mc_replicates = None
    if completed_countries is not None and completed_countries != len(expected):
        failures.append(
            f"countries_completed={completed_countries}, expected={len(expected)}"
        )
    if minimum_folds is not None and minimum_folds < 3:
        failures.append("fewer than three outer folds for at least one country")
    if mc_replicates is not None and mc_replicates < 3:
        failures.append("fewer than three independent Monte Carlo replicates")
    if str(row["primary_predictive_scope"]) != (
        "country_balanced_one_interval_ahead_prequential"
    ):
        failures.append("predictive gate has the wrong primary scope")
    if str(row["claim_scope"]) != (
        "notification-index forecasting and conditional scenario projections"
    ):
        failures.append("predictive gate has the wrong scientific claim scope")

    observed_countries = [str(value) for value in metadata.get("countries_included", [])]
    if observed_countries != expected:
        failures.append(
            f"predictive gate countries={observed_countries} do not match "
            f"publication countries={expected}"
        )
    if str(metadata.get("forecast_mode", "")) != "prequential":
        failures.append("predictive metadata is not prequential")
    if str(metadata.get("scope", "")) != (
        "semi_mechanistic_discrepancy_pomp_not_full_compartment_pomp"
    ):
        failures.append("predictive metadata has the wrong model scope")
    for key in (
        "data_integrity_gate_pass",
        "predictive_gate_pass",
        "publication_gate_pass",
    ):
        try:
            passed = _strict_boolean(metadata.get(key))
        except ValueError as exc:
            failures.append(f"metadata {key}: {exc}")
        else:
            if not passed:
                failures.append(f"predictive metadata has {key}=false")
    for key in (
        "full_compartment_pomp_claimed",
        "posterior_parameter_uncertainty_claimed",
    ):
        try:
            claimed = _strict_boolean(metadata.get(key))
        except ValueError as exc:
            failures.append(f"metadata {key}: {exc}")
        else:
            if claimed:
                failures.append(f"predictive metadata has {key}=true")

    for key, minimum in (
        ("particles", 512),
        ("predictive_draws", 4096),
        ("mc_replicates", 3),
    ):
        try:
            value = _strict_positive_integer(metadata.get(key), name=key)
        except ValueError as exc:
            failures.append(str(exc))
        else:
            if value < minimum:
                failures.append(f"predictive metadata {key}={value}, minimum={minimum}")
    if metadata.get("test_years") != [2023, 2024, 2025, 2026]:
        failures.append("predictive metadata does not contain the canonical 2023-2026 folds")

    fold_counts_raw = metadata.get("fold_counts")
    if not isinstance(fold_counts_raw, dict) or set(fold_counts_raw) != set(expected):
        failures.append("predictive metadata fold-count countries are incomplete")
    else:
        parsed_fold_counts: dict[str, int] = {}
        for country, value in fold_counts_raw.items():
            try:
                parsed_fold_counts[str(country)] = _strict_positive_integer(
                    value, name=f"fold_counts[{country}]"
                )
            except ValueError as exc:
                failures.append(str(exc))
        if parsed_fold_counts and min(parsed_fold_counts.values()) < 3:
            failures.append("predictive metadata has fewer than three country folds")
    fold_years = metadata.get("fold_test_years")
    if not isinstance(fold_years, dict) or set(fold_years) != set(expected):
        failures.append("predictive metadata fold years are incomplete")
    elif isinstance(fold_counts_raw, dict):
        for country in expected:
            years = fold_years.get(country)
            if not isinstance(years, list) or len(years) != int(fold_counts_raw[country]):
                failures.append(f"predictive metadata fold years are incomplete for {country}")

    row_counts = metadata.get("row_counts")
    fold_key = "panel_pomp_rolling_hindcast_folds"
    interval_key = "panel_pomp_rolling_hindcast_intervals"
    if not isinstance(row_counts, dict):
        failures.append("predictive metadata is missing row counts")
    else:
        for key in (fold_key, interval_key):
            try:
                _strict_positive_integer(row_counts.get(key), name=f"row_counts.{key}")
            except ValueError as exc:
                failures.append(str(exc))

    input_hashes = metadata.get("input_artifact_sha256")
    if not isinstance(input_hashes, dict) or input_hashes.get(
        "pertussis_incidence_timeseries"
    ) is None:
        failures.append("predictive input data hash is missing")
    elif len(str(input_hashes["pertussis_incidence_timeseries"])) != 64:
        failures.append("predictive input data hash is invalid")
    output_hashes = metadata.get("output_artifact_sha256")
    required_outputs = {
        "panel_pomp_rolling_hindcast_gate": PREDICTIVE_GATE_PATH,
        fold_key: PREDICTIVE_FOLD_PATH,
        interval_key: PREDICTIVE_INTERVAL_PATH,
    }
    if not isinstance(output_hashes, dict):
        failures.append("predictive output hashes are missing")
    else:
        for stem in required_outputs:
            if len(str(output_hashes.get(stem, ""))) != 64:
                failures.append(f"predictive output hash is missing or invalid: {stem}")
    return failures


def predictive_artifact_freshness_failures(metadata: dict[str, Any]) -> list[str]:
    """Cross-check recorded POMP input/output digests against the filesystem."""

    failures: list[str] = []
    input_hashes = metadata.get("input_artifact_sha256")
    if not isinstance(input_hashes, dict) or not PREDICTIVE_INPUT_PATH.exists():
        failures.append("predictive input data artifact is missing")
    elif input_hashes.get("pertussis_incidence_timeseries") != file_sha256(
        PREDICTIVE_INPUT_PATH
    ):
        failures.append("predictive input data hash is stale")
    output_hashes = metadata.get("output_artifact_sha256")
    required_outputs = {
        "panel_pomp_rolling_hindcast_gate": PREDICTIVE_GATE_PATH,
        "panel_pomp_rolling_hindcast_folds": PREDICTIVE_FOLD_PATH,
        "panel_pomp_rolling_hindcast_intervals": PREDICTIVE_INTERVAL_PATH,
    }
    if not isinstance(output_hashes, dict):
        failures.append("predictive output hashes are missing")
    else:
        for stem, path in required_outputs.items():
            if not path.exists() or output_hashes.get(stem) != file_sha256(path):
                failures.append(f"predictive output hash is stale or missing: {stem}")
    return failures


def require_predictive_publication_gate(
    *,
    expected_countries: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a fresh hindcast artifact and raise unless every gate is satisfied."""

    if not PREDICTIVE_GATE_PATH.exists():
        raise FileNotFoundError(
            f"Predictive publication gate is missing: {PREDICTIVE_GATE_PATH}"
        )
    metadata = validate_run_metadata(PREQUENTIAL_RUN_STEM)
    countries = list(
        expected_countries
        if expected_countries is not None
        else publication_country_names(load_configs())
    )
    failures = predictive_publication_gate_failures(
        pd.read_csv(PREDICTIVE_GATE_PATH),
        metadata,
        expected_countries=countries,
    )
    failures.extend(predictive_artifact_freshness_failures(metadata))
    if failures:
        raise RuntimeError(
            "Predictive publication gate failed:\n  - " + "\n  - ".join(failures)
        )
    return metadata


def require_figure2_publication_gate(
    *,
    expected_countries: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Require fresh predictive, exact-state, paired-draw, and audit parents."""

    predictive_metadata = require_predictive_publication_gate(
        expected_countries=expected_countries
    )
    if not FIGURE2_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Figure 2 conditional quality audit is missing: {FIGURE2_AUDIT_PATH}"
        )
    metadata = validate_run_metadata(FIGURE2_AUDIT_STEM)
    failures: list[str] = []
    for key in ("passed", "warnings_are_fatal"):
        try:
            value = _strict_boolean(metadata.get(key))
        except ValueError as exc:
            failures.append(f"Figure 2 audit metadata {key}: {exc}")
        else:
            if not value:
                failures.append(f"Figure 2 audit metadata has {key}=false")
    if metadata.get("posterior_stem") != FIGURE2_POSTERIOR_STEM:
        failures.append("Figure 2 audit metadata references a noncanonical posterior stem")
    if metadata.get("figure2c_stem") != FIGURE2_SOURCE_STEM:
        failures.append("Figure 2 audit metadata references a noncanonical source-data stem")
    recorded_audit_digest = str(metadata.get("audit_table_sha256", ""))
    observed_audit_digest = file_sha256(FIGURE2_AUDIT_PATH)
    if not recorded_audit_digest or recorded_audit_digest != observed_audit_digest:
        failures.append("Figure 2 audit table digest does not match its run metadata")
    audited_digests = metadata.get("audited_artifact_sha256")
    if not isinstance(audited_digests, dict):
        failures.append("Figure 2 audit metadata is missing audited artifact digests")
    else:
        for digest_name, artifact_path in FIGURE2_AUDITED_ARTIFACT_PATHS.items():
            if not artifact_path.exists():
                failures.append(f"Figure 2 audited artifact is missing: {artifact_path}")
                continue
            recorded = str(audited_digests.get(digest_name, ""))
            observed_digest = file_sha256(artifact_path)
            if not recorded or recorded != observed_digest:
                failures.append(
                    f"Figure 2 audited artifact digest changed: {artifact_path}"
                )

    audit = pd.read_csv(FIGURE2_AUDIT_PATH)
    required_columns = {"check", "status", "severity"}
    missing_columns = sorted(required_columns.difference(audit.columns))
    if missing_columns:
        failures.append(f"Figure 2 quality audit is missing columns: {missing_columns}")
    else:
        checks = set(audit["check"].dropna().astype(str))
        missing_checks = sorted(FIGURE2_REQUIRED_AUDIT_CHECKS.difference(checks))
        if missing_checks:
            failures.append(
                f"Figure 2 quality audit is missing required checks: {missing_checks}"
            )
        invalid_status = audit["status"].isna() | ~audit["status"].astype(str).isin(
            ["pass", "fail"]
        )
        invalid_severity = audit["severity"].isna() | ~audit[
            "severity"
        ].astype(str).isin(["fatal", "warning", "informational"])
        if bool(invalid_status.any()):
            failures.append("Figure 2 quality audit has missing or invalid status values")
        if bool(invalid_severity.any()):
            failures.append("Figure 2 quality audit has missing or invalid severity values")
        failed_release_checks = audit.loc[
            audit["severity"].astype(str).isin(["fatal", "warning"])
            & ~audit["status"].astype(str).eq("pass")
        ]
        if not failed_release_checks.empty:
            failures.append(
                "Figure 2 quality audit has failed fatal/warning checks: "
                + ", ".join(failed_release_checks["check"].astype(str).tolist())
            )
    if failures:
        raise RuntimeError("Figure 2 publication gate failed:\n  - " + "\n  - ".join(failures))
    return {
        "predictive_metadata": predictive_metadata,
        "figure2_audit_metadata": metadata,
    }


def main() -> None:
    """CLI used by non-Python publication entry points for freshness checks."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-figure2",
        action="store_true",
        help="Require the complete fresh predictive and conditional Figure 2 gate.",
    )
    args = parser.parse_args()
    if args.require_figure2:
        require_figure2_publication_gate()
        print("Figure 2 publication gate passed")
    else:
        require_predictive_publication_gate()
        print("Predictive publication gate passed")


if __name__ == "__main__":
    main()
