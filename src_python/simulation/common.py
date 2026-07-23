from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src_python.model.compartments import StateIndex
from src_python.model.observables import (
    CaseExposure,
    build_observation_plan,
    solve_case_exposure,
)
from src_python.model.outputs import (
    compute_timeseries,
    infer_output_dt,
    prepare_history_state,
    solve_model,
    summarize_timeseries,
)
from src_python.model.parameters import PreparedParameters
from src_python.simulation.parameter_distributions import (
    validate_uncertainty_registry_schema,
)
from src_python.utils.io import (
    deep_update,
    ensure_output_dirs,
    load_yaml,
    project_path,
    read_table,
    set_by_dotted_path,
    write_dataframe,
)
from src_python.utils.parallel import parallel_map


COUNTRY_RESISTANCE_TIMELINE_COLUMNS = {
    "country",
    "iso3",
    "year",
    "resistant_fraction",
    "lower",
    "upper",
    "evidence_type",
    "source",
    "notes",
}

DIAGNOSTIC_STANDARD_TIMELINE_COLUMNS = {
    "country",
    "iso3",
    "period_start",
    "period_end",
    "geographic_scope",
    "surveillance_regime",
    "primary_diagnostic_methods",
    "case_definition_or_reporting_change",
    "relative_detection_prior_mean",
    "relative_detection_prior_lower",
    "relative_detection_prior_upper",
    "effect_direction",
    "evidence_strength",
    "source_ids",
    "notes",
}

NPI_CONTACT_REDUCTION_TIMELINE_COLUMNS = {
    "country",
    "iso3",
    "period_start",
    "period_end",
    "baseline_contact_reduction",
    "contact_reduction_mean",
    "contact_reduction_lower",
    "contact_reduction_upper",
    "ramp_days",
    "evidence_strength",
    "notes",
}

DTP_COVERAGE_COLUMNS = {
    "CODE",
    "YEAR",
    "ANTIGEN",
    "COVERAGE_CATEGORY",
    "COVERAGE",
}

REQUIRED_RUNTIME_BLOCKS = (
    "baseline_parameters",
    "vaccine_scenarios",
    "resistance_scenarios",
    "intervention_scenarios",
    "sensitivity_parameters",
    "data_sources",
)

METADATA_SCHEMA_VERSION = 1
DEPENDENCY_VERSION_PACKAGES = ("numpy", "pandas", "scipy", "pyyaml", "joblib", "numba", "pyarrow")
PROSPECTIVE_POLICY_KEY = "_prospective_policy"
PROSPECTIVE_POLICY_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def _load_configs_cached() -> dict[str, dict[str, Any]]:
    settings_path = project_path("config/model_settings.yaml")
    if not settings_path.exists():
        raise FileNotFoundError("config/model_settings.yaml is the runtime configuration source and was not found.")
    settings = load_yaml(settings_path)
    runtime = settings.get("runtime", {})
    missing = [block for block in REQUIRED_RUNTIME_BLOCKS if block not in runtime]
    if missing:
        raise ValueError(f"config/model_settings.yaml is missing runtime blocks: {missing}")
    baseline = deepcopy(runtime["baseline_parameters"])
    _resolve_calendar_horizon(baseline)
    for optional_block in ("fitness_grid", "bayesian_uncertainty"):
        if optional_block in runtime:
            baseline[optional_block] = deepcopy(runtime[optional_block])
    distribution_path = project_path("config/parameter_distributions.yaml")
    parameter_distributions = load_yaml(distribution_path) if distribution_path.exists() else {}
    validate_uncertainty_registry_schema(
        parameter_distributions,
        context="config/parameter_distributions.yaml",
    )
    return {
        "settings": settings,
        "baseline": baseline,
        "vaccines": runtime["vaccine_scenarios"],
        "resistance": runtime["resistance_scenarios"],
        "interventions": runtime["intervention_scenarios"],
        "sensitivity": runtime["sensitivity_parameters"],
        "parameter_distributions": parameter_distributions,
        "data_sources": runtime["data_sources"],
        "countries": load_yaml(project_path("config/country_profiles.yaml")),
    }


def load_configs() -> dict[str, dict[str, Any]]:
    """Load runtime configuration with process-local YAML caching.

    Scenario-grid builders call ``make_config`` hundreds of times. Parsing the
    large YAML files on every call dominated runtime, so the immutable source
    parse is cached and callers receive a deep copy they may safely mutate.
    """
    return deepcopy(_load_configs_cached())


def publication_country_names(
    configs: dict[str, Any] | None = None,
) -> list[str]:
    """Return the prespecified calibrated/publication country set."""

    resolved = configs or load_configs()
    profiles = resolved.get("countries", {})
    exclusions = (
        resolved.get("baseline", {})
        .get("bayesian_uncertainty", {})
        .get("publication_country_exclusions", {})
    )
    unknown = sorted(set(exclusions).difference(profiles))
    if unknown:
        raise ValueError(f"Publication country exclusions reference unknown profiles: {unknown}")
    selected = [str(country) for country in profiles if country not in exclusions]
    if not selected:
        raise ValueError("The calibrated publication country set is empty")
    return selected


def _resolve_calendar_horizon(baseline: dict[str, Any]) -> None:
    """Derive simulation.end_time from calendar dates when available.

    The yaml keeps `calendar.analysis_start_date` / `calendar.analysis_end_date`
    as the single source of truth for the production analysis window. Both
    dates are user-facing *inclusive* dates; the continuous solver therefore
    integrates to midnight after ``analysis_end_date`` so that the internal
    interval is half-open and contains every requested calendar day. Tests and
    the calibration runtime still override `simulation.end_time` directly and
    are respected here.
    """
    from datetime import date, datetime

    simulation = baseline.setdefault("simulation", {})
    calendar = baseline.setdefault("calendar", {})
    start_date = calendar.get("analysis_start_date")
    end_date = calendar.get("analysis_end_date")

    def _parse(value: Any) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()

    start = _parse(start_date)
    end = _parse(end_date)
    explicit_end_time = simulation.get("end_time")

    if start is not None and end is not None:
        if end <= start:
            raise ValueError(
                f"calendar.analysis_end_date ({end_date}) must be after analysis_start_date ({start_date})."
            )
        derived_end_time = float((end - start).days + 1)
        if explicit_end_time is None:
            simulation["end_time"] = derived_end_time
        else:
            # Keep explicit overrides but flag meaningful mismatches.
            if abs(float(explicit_end_time) - derived_end_time) > 1.0:
                # Prefer the explicit value (tests, calibration windows) but
                # record the calendar-derived duration for traceability.
                simulation["end_time_calendar_derived"] = derived_end_time
    elif explicit_end_time is None:
        raise ValueError(
            "simulation.end_time cannot be derived: set either calendar.analysis_end_date "
            "(with analysis_start_date) or simulation.end_time in the runtime baseline."
        )

    simulation.setdefault("start_time", 0)


def set_analysis_horizon_years(
    config: dict[str, Any],
    years: int,
) -> str:
    """Set a calendar-consistent N-year window from the configured policy t0."""

    if int(years) < 1:
        raise ValueError("Analysis horizon years must be positive")
    calendar = config.setdefault("calendar", {})
    if not calendar.get("analysis_start_date"):
        raise ValueError("A calendar analysis_start_date is required")
    start = pd.Timestamp(calendar["analysis_start_date"])
    end = start + pd.DateOffset(years=int(years)) - pd.Timedelta(days=1)
    calendar["analysis_end_date"] = end.date().isoformat()
    # ``end`` is the last included calendar date; the ODE endpoint is midnight
    # immediately after it. This makes an N-year window exactly
    # [start, start + N years) even when it contains leap days.
    config.setdefault("simulation", {})["end_time"] = float(
        ((end + pd.Timedelta(days=1)) - start).days
    )
    return calendar["analysis_end_date"]


def _git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=project_path(), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(run_git(["status", "--short"])),
    }


def _dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in DEPENDENCY_VERSION_PACKAGES:
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def output_metadata_path(stem: str) -> Path:
    return project_path("outputs", "metadata", f"{stem}_run_metadata.json")


def current_run_metadata(stem: str, *, row_counts: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "stem": stem,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "dependency_versions": _dependency_versions(),
        "row_counts": row_counts or {},
    }


def write_run_metadata(stem: str, metadata: dict[str, Any]) -> None:
    path = output_metadata_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def read_run_metadata(stem: str) -> dict[str, Any]:
    path = output_metadata_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"Missing run metadata for {stem}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dependency_version_audit_for_metadata(stem: str, metadata: dict[str, Any]) -> dict[str, Any]:
    dependency_versions = metadata.get("dependency_versions") or {}
    if not isinstance(dependency_versions, dict):
        return {
            "stem": stem,
            "dependency_versions_is_object": False,
            "missing_keys": [],
            "not_installed_keys": [],
            "failures": [f"{stem}: dependency_versions is not an object"],
        }

    required_keys = {"python", *DEPENDENCY_VERSION_PACKAGES}
    missing = sorted(key for key in required_keys if key not in dependency_versions)
    not_installed = sorted(
        key
        for key in required_keys
        if dependency_versions.get(key) == "not-installed"
    )
    failures: list[str] = []
    if missing:
        failures.append(f"{stem}: dependency_versions missing {', '.join(missing)}")
    if not_installed:
        failures.append(f"{stem}: dependency_versions recorded not-installed for {', '.join(not_installed)}")
    return {
        "stem": stem,
        "dependency_versions_is_object": True,
        "missing_keys": missing,
        "not_installed_keys": not_installed,
        "failures": failures,
    }


def dependency_version_failures_for_metadata(stem: str, metadata: dict[str, Any]) -> list[str]:
    audit = dependency_version_audit_for_metadata(stem, metadata)
    return list(audit["failures"])


def validate_run_metadata(stem: str, *, require_dependency_versions: bool = True) -> dict[str, Any]:
    """Validate schema and dependency fields in run metadata."""

    metadata = read_run_metadata(stem)
    if int(metadata.get("schema_version", -1)) != METADATA_SCHEMA_VERSION:
        raise ValueError(f"Run metadata schema mismatch for {stem}.")
    if require_dependency_versions:
        dependency_failures = dependency_version_failures_for_metadata(stem, metadata)
        if dependency_failures:
            raise ValueError("; ".join(dependency_failures))
    return metadata


def calibrated_country_artifact_path(country: str) -> Path:
    safe_country = str(country).strip().replace(" ", "_")
    return project_path("outputs", "calibrations", f"{safe_country}_calibrated_config.yaml")


@lru_cache(maxsize=None)
def _load_calibrated_country_artifact_cached(
    country: str,
) -> dict[str, Any] | None:
    path = calibrated_country_artifact_path(country)
    if not path.exists():
        return None

    artifact = load_yaml(path)
    if not isinstance(artifact, dict):
        return None

    metadata = artifact.get("metadata", {}) if isinstance(artifact.get("metadata", {}), dict) else {}
    if not bool(metadata.get("accepted", False)):
        return None

    artifact = deepcopy(artifact)
    return artifact


def load_calibrated_country_artifact(
    country: str,
) -> dict[str, Any] | None:
    artifact = _load_calibrated_country_artifact_cached(country)
    return deepcopy(artifact) if artifact is not None else None


def validate_calibration_artifacts(
    countries: Iterable[str],
    *,
    context: str,
) -> tuple[Path, ...]:
    """Require an accepted calibration artifact for each requested country."""

    paths: list[Path] = []
    unavailable: list[str] = []
    for raw_country in countries:
        country = str(raw_country)
        artifact = load_calibrated_country_artifact(country)
        path = calibrated_country_artifact_path(country)
        if artifact is None or not path.is_file():
            unavailable.append(country)
            continue
        paths.append(path)
    if unavailable:
        raise RuntimeError(
            f"{context} requires accepted calibration artifacts for every publication "
            f"country. Missing or unaccepted: {', '.join(sorted(unavailable))}"
        )
    return tuple(sorted(paths))


def _value_at_dotted_path(config: dict[str, Any], path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _calibration_parameter_overlay(calibrated_config: dict[str, Any]) -> dict[str, Any]:
    """Extract fitted calibration parameters without copying stale inputs."""
    overlay: dict[str, Any] = {}
    for path in (
        "transmission.beta_S",
        "transmission.seasonal_amplitude",
        "transmission.log_beta_time_variation",
        "reporting_multiplier",
        "importation.rate_per_100k_per_year",
        "importation.resistant_fraction",
        "resistance.importation_fraction",
    ):
        value = _value_at_dotted_path(calibrated_config, path)
        if value is not None:
            set_by_dotted_path(overlay, path, value)
    return overlay


def _clear_stem_outputs(stem: str) -> None:
    candidates = [
        project_path("outputs", "simulations", f"{stem}.csv"),
        project_path("outputs", "simulations", f"{stem}.parquet"),
        project_path("outputs", "summaries", f"{stem}_summary.csv"),
        project_path("outputs", "summaries", f"{stem}_summary.parquet"),
        output_metadata_path(stem),
    ]
    for path in candidates:
        if path.exists():
            path.unlink()


def _apply_vaccine(config: dict[str, Any], vaccine: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    out["vaccine"] = {k: v for k, v in vaccine.items() if k.startswith("VE_")}
    return out


def _apply_resistance(config: dict[str, Any], resistance: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    target = float(
        resistance.get(
            "target_prevalence_at_analysis_start",
            resistance.get("initial_resistance_prevalence", 0.0),
        )
    )
    out["initial_conditions"]["initial_resistance_prevalence"] = float(
        resistance.get("initial_resistance_prevalence", target)
    )
    out.setdefault("resistance", {})
    out["resistance"]["target_prevalence_at_analysis_start"] = target
    out["resistance"]["importation_fraction"] = float(resistance.get("importation_fraction", target))
    out["resistance"]["rebalance_after_burn_in"] = bool(resistance.get("rebalance_after_burn_in", True))
    out["resistance"]["prevalence_anchor_rate_per_year"] = float(
        resistance.get(
            "prevalence_anchor_rate_per_year",
            out.get("resistance", {}).get("prevalence_anchor_rate_per_year", 0.0),
        )
    )
    out["resistance"]["anchor_during_dynamics"] = bool(resistance.get("anchor_during_dynamics", False))
    out["resistance"]["use_country_resistance_timeline"] = bool(
        resistance.get("use_country_resistance_timeline", False)
    )
    out.setdefault("importation", {})["resistant_fraction"] = out["resistance"]["importation_fraction"]
    out["transmission"]["fitness_R"] = float(resistance.get("fitness_R", out["transmission"].get("fitness_R", 1.0)))
    return out


def _load_country_resistance_timeline(data_sources: dict[str, Any]) -> pd.DataFrame:
    relative_path = data_sources.get("country_resistance_timeline_csv", "data/raw/country_resistance_timeline.csv")
    path = project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Country resistance timeline not found: {path}")
    timeline = pd.read_csv(path)
    missing = COUNTRY_RESISTANCE_TIMELINE_COLUMNS.difference(timeline.columns)
    if missing:
        raise ValueError(f"Country resistance timeline is missing columns: {sorted(missing)}")

    timeline = timeline.copy()
    timeline["country"] = timeline["country"].astype(str)
    timeline["iso3"] = timeline["iso3"].astype(str).str.upper()
    timeline["evidence_type"] = timeline["evidence_type"].astype(str).str.strip()
    timeline["source"] = timeline["source"].astype(str).str.strip()
    timeline["year"] = pd.to_numeric(timeline["year"], errors="coerce")
    for column in ["resistant_fraction", "lower", "upper"]:
        timeline[column] = pd.to_numeric(timeline[column], errors="coerce")
    timeline = timeline.dropna(subset=["country", "iso3", "year", "resistant_fraction"])
    timeline["year"] = timeline["year"].astype(int)

    for column in ["resistant_fraction", "lower", "upper"]:
        values = timeline[column].dropna()
        if not values.between(0.0, 1.0).all():
            raise ValueError(f"Country resistance timeline column {column} must be within [0, 1].")
    for column in ["evidence_type", "source"]:
        if timeline[column].eq("").any() or timeline[column].str.lower().eq("nan").any():
            raise ValueError(f"Country resistance timeline column {column} must be populated.")
    return timeline


def _load_diagnostic_standard_timeline(data_sources: dict[str, Any]) -> pd.DataFrame:
    relative_path = data_sources.get(
        "diagnostic_standard_timeline_csv",
        "data/raw/pertussis_diagnostic_standards_timeline.csv",
    )
    path = project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Pertussis diagnostic standard timeline not found: {path}")

    timeline = pd.read_csv(path)
    missing = DIAGNOSTIC_STANDARD_TIMELINE_COLUMNS.difference(timeline.columns)
    if missing:
        raise ValueError(f"Pertussis diagnostic standard timeline is missing columns: {sorted(missing)}")

    timeline = timeline.copy()
    timeline["country"] = timeline["country"].astype(str)
    timeline["iso3"] = timeline["iso3"].astype(str).str.upper()
    for column in ("period_start", "period_end"):
        timeline[column] = pd.to_datetime(timeline[column], errors="coerce")
    timeline = timeline.dropna(subset=["country", "iso3", "period_start", "period_end"]).copy()
    if timeline.empty:
        raise ValueError("Pertussis diagnostic standard timeline has no valid rows.")
    if not timeline["period_end"].ge(timeline["period_start"]).all():
        raise ValueError("Diagnostic standard timeline period_end must be on or after period_start.")

    for column in (
        "relative_detection_prior_mean",
        "relative_detection_prior_lower",
        "relative_detection_prior_upper",
    ):
        timeline[column] = pd.to_numeric(timeline[column], errors="coerce")
        if timeline[column].isna().any() or not timeline[column].gt(0.0).all():
            raise ValueError(f"Diagnostic standard timeline column {column} must be positive and finite.")
    if not (
        timeline["relative_detection_prior_lower"].le(timeline["relative_detection_prior_mean"]).all()
        and timeline["relative_detection_prior_mean"].le(timeline["relative_detection_prior_upper"]).all()
    ):
        raise ValueError("Diagnostic standard timeline prior lower/mean/upper columns are inconsistent.")

    for column in (
        "geographic_scope",
        "surveillance_regime",
        "primary_diagnostic_methods",
        "effect_direction",
        "evidence_strength",
        "source_ids",
    ):
        timeline[column] = timeline[column].astype(str).str.strip()
        if timeline[column].eq("").any() or timeline[column].str.lower().eq("nan").any():
            raise ValueError(f"Diagnostic standard timeline column {column} must be populated.")
    forbidden_dynamic_terms = ("immunity debt", "reduced circulation", "true infections")
    combined_text = (
        timeline["case_definition_or_reporting_change"].fillna("").astype(str)
        + " "
        + timeline["notes"].fillna("").astype(str)
    ).str.lower()
    offending = timeline.loc[
        combined_text.apply(lambda text: any(term in text for term in forbidden_dynamic_terms)),
        ["country", "period_start", "period_end"],
    ]
    if not offending.empty:
        records = "; ".join(
            f"{row.country} {row.period_start.date()}..{row.period_end.date()}"
            for row in offending.itertuples(index=False)
        )
        raise ValueError(
            "Diagnostic standard timeline must describe only detection, care-seeking, testing, "
            f"or reporting changes; move transmission-dynamic explanations to the NPI timeline. Offending rows: {records}"
        )
    return timeline.sort_values(["country", "period_start", "period_end"]).reset_index(drop=True)


def _load_npi_contact_reduction_timeline(data_sources: dict[str, Any]) -> pd.DataFrame:
    relative_path = data_sources.get(
        "npi_contact_reduction_timeline_csv",
        "data/raw/covid_npi_contact_reduction_timeline.csv",
    )
    path = project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"COVID/NPI contact reduction timeline not found: {path}")

    timeline = pd.read_csv(path)
    missing = NPI_CONTACT_REDUCTION_TIMELINE_COLUMNS.difference(timeline.columns)
    if missing:
        raise ValueError(f"COVID/NPI contact reduction timeline is missing columns: {sorted(missing)}")

    timeline = timeline.copy()
    timeline["country"] = timeline["country"].astype(str)
    timeline["iso3"] = timeline["iso3"].astype(str).str.upper()
    for column in ("period_start", "period_end"):
        timeline[column] = pd.to_datetime(timeline[column], errors="coerce")
    timeline = timeline.dropna(subset=["country", "iso3", "period_start", "period_end"]).copy()
    if timeline.empty:
        raise ValueError("COVID/NPI contact reduction timeline has no valid rows.")
    if not timeline["period_end"].ge(timeline["period_start"]).all():
        raise ValueError("COVID/NPI contact reduction period_end must be on or after period_start.")

    for column in (
        "baseline_contact_reduction",
        "contact_reduction_mean",
        "contact_reduction_lower",
        "contact_reduction_upper",
    ):
        timeline[column] = pd.to_numeric(timeline[column], errors="coerce")
        if timeline[column].isna().any() or not timeline[column].between(0.0, 1.0).all():
            raise ValueError(f"COVID/NPI contact reduction timeline column {column} must be within [0, 1].")
    if not (
        timeline["contact_reduction_lower"].le(timeline["contact_reduction_mean"]).all()
        and timeline["contact_reduction_mean"].le(timeline["contact_reduction_upper"]).all()
    ):
        raise ValueError("COVID/NPI contact reduction lower/mean/upper columns are inconsistent.")

    timeline["ramp_days"] = pd.to_numeric(timeline["ramp_days"], errors="coerce")
    if timeline["ramp_days"].isna().any() or timeline["ramp_days"].lt(0.0).any():
        raise ValueError("COVID/NPI contact reduction ramp_days must be finite and non-negative.")
    for column in ("evidence_strength", "notes"):
        timeline[column] = timeline[column].astype(str).str.strip()
        if timeline[column].eq("").any() or timeline[column].str.lower().eq("nan").any():
            raise ValueError(f"COVID/NPI contact reduction timeline column {column} must be populated.")
    return timeline.sort_values(["country", "period_start", "period_end"]).reset_index(drop=True)


def _timeline_rows_for_country(timeline: pd.DataFrame, country: str, iso3: str | None) -> pd.DataFrame:
    country_key = str(country)
    iso3_key = str(iso3 or "").upper()
    rows = timeline.loc[timeline["country"].eq(country_key)]
    if rows.empty and iso3_key:
        rows = timeline.loc[timeline["iso3"].eq(iso3_key)]
    if rows.empty:
        raise KeyError(f"No country resistance timeline rows found for {country_key} ({iso3_key}).")
    return rows.sort_values("year")


def _diagnostic_rows_for_country(timeline: pd.DataFrame, country: str, iso3: str | None) -> pd.DataFrame:
    country_key = str(country)
    iso3_key = str(iso3 or "").upper()
    rows = timeline.loc[timeline["country"].eq(country_key)]
    if rows.empty and iso3_key:
        rows = timeline.loc[timeline["iso3"].eq(iso3_key)]
    if rows.empty:
        raise KeyError(f"No diagnostic standard timeline rows found for {country_key} ({iso3_key}).")
    return rows.sort_values(["period_start", "period_end"])


def _npi_rows_for_country(timeline: pd.DataFrame, country: str, iso3: str | None) -> pd.DataFrame:
    country_key = str(country)
    iso3_key = str(iso3 or "").upper()
    rows = timeline.loc[timeline["country"].eq(country_key)]
    if rows.empty and iso3_key:
        rows = timeline.loc[timeline["iso3"].eq(iso3_key)]
    if rows.empty:
        raise KeyError(f"No COVID/NPI contact reduction timeline rows found for {country_key} ({iso3_key}).")
    return rows.sort_values(["period_start", "period_end"])


def _interpolate_optional(years: np.ndarray, values: np.ndarray, analysis_year: int) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    finite_years = years[finite]
    finite_values = values[finite]
    if analysis_year <= finite_years.min():
        return float(finite_values[np.argmin(finite_years)])
    if analysis_year >= finite_years.max():
        return float(finite_values[np.argmax(finite_years)])
    return float(np.interp(analysis_year, finite_years, finite_values))


def _country_resistance_estimate(rows: pd.DataFrame, anchor_year: int, *, allow_future: bool = False) -> dict[str, Any]:
    evidence_pool = rows.copy()
    if not allow_future:
        evidence_pool = evidence_pool.loc[evidence_pool["year"].astype(int) <= anchor_year]
        if evidence_pool.empty:
            raise ValueError(
                f"No resistance evidence is available at or before {anchor_year}; "
                "set allow_future_resistance_evidence=true only for explicitly labelled current-evidence scenarios."
            )

    years = evidence_pool["year"].to_numpy(dtype=float)
    values = evidence_pool["resistant_fraction"].to_numpy(dtype=float)
    target = _interpolate_optional(years, values, anchor_year)
    lower = _interpolate_optional(years, evidence_pool["lower"].to_numpy(dtype=float), anchor_year)
    upper = _interpolate_optional(years, evidence_pool["upper"].to_numpy(dtype=float), anchor_year)

    if anchor_year in set(evidence_pool["year"].astype(int)):
        method = "exact_year"
        evidence_rows = evidence_pool.loc[evidence_pool["year"].eq(anchor_year)]
    elif anchor_year < int(evidence_pool["year"].min()) or anchor_year > int(evidence_pool["year"].max()):
        method = "nearest_year" if allow_future else "nearest_past_year"
        nearest_idx = (evidence_pool["year"].astype(int) - anchor_year).abs().idxmin()
        evidence_rows = evidence_pool.loc[[nearest_idx]]
    else:
        method = "linear_interpolation"
        before = evidence_pool.loc[evidence_pool["year"] < anchor_year].tail(1)
        after = evidence_pool.loc[evidence_pool["year"] > anchor_year].head(1)
        evidence_rows = pd.concat([before, after], ignore_index=True)

    def join_unique(column: str) -> str:
        values = [str(value) for value in evidence_rows[column].dropna().unique() if str(value)]
        return "; ".join(values)

    return {
        "resistance_anchor_year": int(anchor_year),
        "resistant_fraction": float(np.clip(target, 0.0, 1.0)),
        "lower": float(np.clip(lower, 0.0, 1.0)) if np.isfinite(lower) else np.nan,
        "upper": float(np.clip(upper, 0.0, 1.0)) if np.isfinite(upper) else np.nan,
        "method": method,
        "evidence_years": ";".join(str(int(year)) for year in evidence_rows["year"].tolist()),
        "evidence_type": join_unique("evidence_type"),
        "source": join_unique("source"),
        "notes": join_unique("notes"),
    }


def _apply_country_resistance_timeline(
    config: dict[str, Any],
    *,
    country: str,
    country_profile: dict[str, Any],
    data_sources: dict[str, Any],
    evidence_cutoff_year: int | None = None,
) -> dict[str, Any]:
    out = deepcopy(config)
    configured_anchor_year = int(
        data_sources.get(
            "resistance_anchor_year",
            data_sources.get("analysis_year", 2023),
        )
    )
    anchor_year = configured_anchor_year
    if evidence_cutoff_year is not None:
        anchor_year = min(anchor_year, int(evidence_cutoff_year))
    allow_future = bool(data_sources.get("allow_future_resistance_evidence", False))
    iso3 = country_profile.get("iso3")
    timeline = _load_country_resistance_timeline(data_sources)
    rows = _timeline_rows_for_country(timeline, country, iso3)
    try:
        estimate = _country_resistance_estimate(
            rows,
            anchor_year,
            allow_future=allow_future,
        )
    except ValueError:
        if evidence_cutoff_year is None:
            raise
        # Historical validation must never borrow the first resistance datum
        # from the future.  If no country evidence existed at the forecast
        # origin, retain the prespecified generic resistance scenario instead
        # of silently back-casting a later measured prevalence.
        resistance = out.setdefault("resistance", {})
        resistance["country_timeline"] = {
            "country": country,
            "iso3": iso3,
            "resistance_anchor_year": int(anchor_year),
            "applied": False,
            "method": "no_evidence_at_or_before_cutoff",
            "evidence_years": "",
        }
        metadata = out.setdefault("metadata", {})
        metadata["resistance_timeline_country"] = country
        metadata["resistance_timeline_iso3"] = iso3 or ""
        metadata["resistance_timeline_anchor_year"] = int(anchor_year)
        metadata["resistance_timeline_configured_anchor_year"] = int(
            configured_anchor_year
        )
        metadata["resistance_timeline_evidence_cutoff_year"] = int(
            evidence_cutoff_year
        )
        metadata["resistance_timeline_applied"] = False
        metadata["resistance_timeline_method"] = (
            "no_evidence_at_or_before_cutoff"
        )
        return out
    target = estimate["resistant_fraction"]

    out["initial_conditions"]["initial_resistance_prevalence"] = target
    out.setdefault("resistance", {})
    out["resistance"]["target_prevalence_at_analysis_start"] = target
    out["resistance"]["importation_fraction"] = target
    out["resistance"]["prevalence_anchor_rate_per_year"] = float(
        out["resistance"].get("prevalence_anchor_rate_per_year", 2.0)
    )
    out["resistance"]["country_timeline"] = {
        "country": country,
        "iso3": iso3,
        **estimate,
    }
    out.setdefault("importation", {})["resistant_fraction"] = target
    metadata = out.setdefault("metadata", {})
    metadata["resistance_timeline_country"] = country
    metadata["resistance_timeline_iso3"] = iso3 or ""
    metadata["resistance_timeline_anchor_year"] = anchor_year
    metadata["resistance_timeline_configured_anchor_year"] = int(
        configured_anchor_year
    )
    metadata["resistance_timeline_evidence_cutoff_year"] = (
        int(evidence_cutoff_year) if evidence_cutoff_year is not None else None
    )
    metadata["resistance_timeline_applied"] = True
    metadata["resistance_timeline_allows_future_evidence"] = allow_future
    metadata["resistance_timeline_method"] = estimate["method"]
    metadata["resistance_timeline_evidence_years"] = estimate["evidence_years"]
    metadata["resistance_timeline_evidence_type"] = estimate["evidence_type"]
    metadata["resistance_timeline_source"] = estimate["source"]
    metadata["resistance_timeline_notes"] = estimate["notes"]
    return out


def _apply_diagnostic_standard_timeline(
    config: dict[str, Any],
    *,
    country: str,
    country_profile: dict[str, Any],
    data_sources: dict[str, Any],
    evidence_cutoff_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    out = deepcopy(config)
    observation_model = out.setdefault("observation_model", {})
    settings = observation_model.get("diagnostic_standards", {})
    if not isinstance(settings, dict):
        settings = {}
    if not bool(settings.get("enabled", True)):
        out.pop("diagnostic_reporting_time_variation", None)
        observation_model["diagnostic_standards"] = {**settings, "enabled": False}
        return out

    multiplier_column = str(settings.get("multiplier_column", "relative_detection_prior_mean"))
    allowed_columns = {
        "relative_detection_prior_mean",
        "relative_detection_prior_lower",
        "relative_detection_prior_upper",
    }
    if multiplier_column not in allowed_columns:
        raise ValueError(
            "Diagnostic standards multiplier_column must be one of "
            f"{sorted(allowed_columns)}; got {multiplier_column!r}."
        )

    iso3 = str(country_profile.get("iso3", "")).upper()
    timeline = _load_diagnostic_standard_timeline(data_sources)
    rows = _diagnostic_rows_for_country(timeline, country, iso3)
    if evidence_cutoff_date is not None:
        # A reporting regime that begins on or after the forecast origin is
        # unknown to a genuine prospective hindcast.  Periods already active
        # at the origin may persist according to their prespecified interval.
        rows = rows.loc[
            pd.to_datetime(rows["period_start"], errors="raise")
            < evidence_cutoff_date
        ].copy()
    if rows.empty:
        out.pop("diagnostic_reporting_time_variation", None)
        observation_model["diagnostic_standards"] = {
            **settings,
            "enabled": False,
            "periods_loaded": 0,
            "disabled_reason": "no_regime_known_before_evidence_cutoff",
        }
        metadata = out.setdefault("metadata", {})
        metadata["diagnostic_standard_country"] = country
        metadata["diagnostic_standard_iso3"] = iso3
        metadata["diagnostic_standard_periods_loaded"] = 0
        metadata["diagnostic_standard_evidence_cutoff_date"] = (
            evidence_cutoff_date.date().isoformat()
            if evidence_cutoff_date is not None
            else None
        )
        return out

    periods: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        periods.append(
            {
                "start_date": pd.Timestamp(getattr(row, "period_start")).date().isoformat(),
                "end_date": pd.Timestamp(getattr(row, "period_end")).date().isoformat(),
                "multiplier": float(getattr(row, multiplier_column)),
                "prior_mean": float(getattr(row, "relative_detection_prior_mean")),
                "prior_lower": float(getattr(row, "relative_detection_prior_lower")),
                "prior_upper": float(getattr(row, "relative_detection_prior_upper")),
                "geographic_scope": str(getattr(row, "geographic_scope")),
                "surveillance_regime": str(getattr(row, "surveillance_regime")),
                "primary_diagnostic_methods": str(getattr(row, "primary_diagnostic_methods")),
                "effect_direction": str(getattr(row, "effect_direction")),
                "evidence_strength": str(getattr(row, "evidence_strength")),
                "source_ids": str(getattr(row, "source_ids")),
                "notes": str(getattr(row, "notes")),
            }
        )

    variation = {
        "enabled": True,
        "country": country,
        "iso3": iso3,
        "multiplier_column": multiplier_column,
        "periods": periods,
    }
    out["diagnostic_reporting_time_variation"] = variation
    observation_model["diagnostic_standards"] = {
        **settings,
        "enabled": True,
        "multiplier_column": multiplier_column,
        "periods_loaded": len(periods),
    }
    metadata = out.setdefault("metadata", {})
    metadata["diagnostic_standard_country"] = country
    metadata["diagnostic_standard_iso3"] = iso3
    metadata["diagnostic_standard_periods_loaded"] = len(periods)
    metadata["diagnostic_standard_multiplier_column"] = multiplier_column
    metadata["diagnostic_standard_evidence_cutoff_date"] = (
        evidence_cutoff_date.date().isoformat()
        if evidence_cutoff_date is not None
        else None
    )
    return out


def _apply_npi_contact_reduction_timeline(
    config: dict[str, Any],
    *,
    country: str,
    country_profile: dict[str, Any],
    data_sources: dict[str, Any],
    evidence_cutoff_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    out = deepcopy(config)
    transmission = out.setdefault("transmission", {})
    settings = transmission.get("npi_contact_reduction", {})
    if not isinstance(settings, dict):
        settings = {}
    if not bool(settings.get("enabled", True)):
        transmission.pop("npi_contact_reduction_periods", None)
        transmission["npi_contact_reduction"] = {**settings, "enabled": False}
        return out

    multiplier_column = str(settings.get("reduction_column", "contact_reduction_mean"))
    allowed_columns = {
        "contact_reduction_mean",
        "contact_reduction_lower",
        "contact_reduction_upper",
        "baseline_contact_reduction",
    }
    if multiplier_column not in allowed_columns:
        raise ValueError(
            "NPI contact reduction reduction_column must be one of "
            f"{sorted(allowed_columns)}; got {multiplier_column!r}."
        )

    iso3 = str(country_profile.get("iso3", "")).upper()
    timeline = _load_npi_contact_reduction_timeline(data_sources)
    rows = _npi_rows_for_country(timeline, country, iso3)
    if evidence_cutoff_date is not None:
        rows = rows.loc[
            pd.to_datetime(rows["period_start"], errors="raise")
            < evidence_cutoff_date
        ].copy()

    periods: list[dict[str, Any]] = []
    ramp_days_values: list[float] = []
    for row in rows.itertuples(index=False):
        ramp_days = float(getattr(row, "ramp_days"))
        ramp_days_values.append(ramp_days)
        periods.append(
            {
                "start_date": pd.Timestamp(getattr(row, "period_start")).date().isoformat(),
                "end_date": pd.Timestamp(getattr(row, "period_end")).date().isoformat(),
                "reduction": float(getattr(row, multiplier_column)),
                "baseline_reduction": float(getattr(row, "baseline_contact_reduction")),
                "reduction_mean": float(getattr(row, "contact_reduction_mean")),
                "reduction_lower": float(getattr(row, "contact_reduction_lower")),
                "reduction_upper": float(getattr(row, "contact_reduction_upper")),
                "ramp_days": ramp_days,
                "evidence_strength": str(getattr(row, "evidence_strength")),
                "notes": str(getattr(row, "notes")),
            }
        )

    transmission["npi_contact_reduction_periods"] = periods
    if ramp_days_values:
        transmission["npi_ramp_days"] = float(max(ramp_days_values))
    transmission["npi_contact_reduction"] = {
        **settings,
        "enabled": True,
        "reduction_column": multiplier_column,
        "periods_loaded": len(periods),
    }
    metadata = out.setdefault("metadata", {})
    metadata["npi_contact_reduction_country"] = country
    metadata["npi_contact_reduction_iso3"] = iso3
    metadata["npi_contact_reduction_periods_loaded"] = len(periods)
    metadata["npi_contact_reduction_reduction_column"] = multiplier_column
    metadata["npi_contact_reduction_evidence_cutoff_date"] = (
        evidence_cutoff_date.date().isoformat()
        if evidence_cutoff_date is not None
        else None
    )
    return out


@lru_cache(maxsize=4)
def _load_dtp_coverage_panel(relative_path: str) -> pd.DataFrame:
    """Load annual DTP evidence used by forecast-origin-safe profiles."""

    path = project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"DTP coverage workbook not found: {path}")
    panel = pd.read_excel(path, sheet_name="Sheet1")
    missing = DTP_COVERAGE_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"DTP coverage workbook is missing columns: {sorted(missing)}")
    panel = panel.loc[:, sorted(DTP_COVERAGE_COLUMNS)].copy()
    panel["CODE"] = panel["CODE"].astype(str).str.upper().str.strip()
    panel["ANTIGEN"] = panel["ANTIGEN"].astype(str).str.upper().str.strip()
    panel["COVERAGE_CATEGORY"] = (
        panel["COVERAGE_CATEGORY"].astype(str).str.upper().str.strip()
    )
    panel["YEAR"] = pd.to_numeric(panel["YEAR"], errors="coerce")
    panel["COVERAGE"] = pd.to_numeric(panel["COVERAGE"], errors="coerce")
    panel = panel.dropna(subset=["YEAR", "COVERAGE"]).copy()
    panel["YEAR"] = panel["YEAR"].astype(int)
    return panel.loc[
        panel["ANTIGEN"].isin({"DTPCV1", "DTPCV3"})
        & panel["COVERAGE"].between(0.0, 100.0)
    ].copy()


def _historical_dtp_pair(
    data_sources: dict[str, Any],
    *,
    iso3: str,
    evidence_cutoff_date: pd.Timestamp,
) -> tuple[int, float, float, str, str]:
    """Return the latest DTP1/DTP3 pair available at a forecast origin."""

    relative_path = str(data_sources.get("dtp_coverage_xlsx", "")).strip()
    if not relative_path:
        raise ValueError("runtime.data_sources.dtp_coverage_xlsx is required")
    # At 1 January the immediately preceding year's final coverage is not yet
    # generally available. Use only a fully elapsed reporting year.
    maximum_year = int(evidence_cutoff_date.year) - 2
    panel = _load_dtp_coverage_panel(relative_path)
    rows = panel.loc[
        panel["CODE"].eq(str(iso3).upper()) & panel["YEAR"].le(maximum_year)
    ].copy()
    category_rank = {"WUENIC": 0, "OFFICIAL": 1, "ADMIN": 2}
    rows["_rank"] = rows["COVERAGE_CATEGORY"].map(category_rank).fillna(99)
    rows = rows.sort_values(
        ["YEAR", "ANTIGEN", "_rank"], ascending=[False, True, True]
    )
    for year in sorted(rows["YEAR"].unique(), reverse=True):
        year_rows = rows.loc[rows["YEAR"].eq(int(year))]
        selected: dict[str, pd.Series] = {}
        for antigen in ("DTPCV1", "DTPCV3"):
            antigen_rows = year_rows.loc[year_rows["ANTIGEN"].eq(antigen)]
            if not antigen_rows.empty:
                selected[antigen] = antigen_rows.iloc[0]
        if len(selected) == 2:
            return (
                int(year),
                float(selected["DTPCV1"]["COVERAGE"]) / 100.0,
                float(selected["DTPCV3"]["COVERAGE"]) / 100.0,
                str(selected["DTPCV1"]["COVERAGE_CATEGORY"]),
                str(selected["DTPCV3"]["COVERAGE_CATEGORY"]),
            )
    raise ValueError(
        f"No complete DTP1/DTP3 evidence pair for {iso3} at or before {maximum_year}"
    )


def _apply_historical_profile_cutoff(
    config: dict[str, Any],
    *,
    country: str,
    country_profile: dict[str, Any],
    data_sources: dict[str, Any],
    evidence_cutoff_date: pd.Timestamp,
) -> dict[str, Any]:
    """Remove outcome-derived profile terms and backdate vaccine evidence."""

    out = deepcopy(config)
    transmission = out.setdefault("transmission", {})
    for key in (
        "seasonal_amplitude",
        "seasonal_phase",
        "multi_year_amplitude",
        "multi_year_phase",
    ):
        transmission[key] = 0.0

    iso3 = str(country_profile.get("iso3", "")).upper()
    coverage_year, dtp1, dtp3, dtp1_category, dtp3_category = _historical_dtp_pair(
        data_sources,
        iso3=iso3,
        evidence_cutoff_date=evidence_cutoff_date,
    )
    country_source = next(
        (
            value
            for value in data_sources.get("countries", {}).values()
            if str(value.get("config_key", "")) == country
        ),
        {},
    )
    coverage_meta = dict(country_source)
    # Maternal estimates currently lack source-availability dates. Excluding
    # them is conservative and avoids retrospective backcasting.
    coverage_meta.update(
        {
            "config_key": country,
            "maternal_coverage": 0.0,
            "maternal_program": False,
        }
    )
    from src_python.data.build_country_inputs import coverage_by_age

    coverage = coverage_by_age(dtp1, dtp3, coverage_meta)
    for record in out.get("age_groups", []):
        label = str(record.get("label", ""))
        if label in coverage:
            record["vaccine_coverage"] = float(coverage[label])
    infant_coverage = float(coverage.get("infant_0_2m", 0.0))
    out.setdefault("demography", {})["birth_entry"] = {
        "S": 1.0 - infant_coverage,
        "V": infant_coverage,
    }
    out.setdefault("metadata", {}).update(
        {
            "outcome_derived_profile_terms_disabled": True,
            "historical_dtp_coverage_year": int(coverage_year),
            "historical_dtp1_coverage": float(dtp1),
            "historical_dtp3_coverage": float(dtp3),
            "historical_dtp1_category": dtp1_category,
            "historical_dtp3_category": dtp3_category,
            "historical_dtp_publication_lag_years": 1,
            "historical_maternal_coverage_excluded": True,
        }
    )
    return out


def _apply_coverage_updates(config: dict[str, Any], updates: dict[str, float] | None) -> dict[str, Any]:
    out = deepcopy(config)
    if not updates:
        return out
    for record in out["age_groups"]:
        if record["label"] in updates:
            record["vaccine_coverage"] = float(updates[record["label"]])
    return out


def _safe_probability(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    if not np.isfinite(numeric):
        numeric = float(default)
    return float(np.clip(numeric, 0.0, 1.0))


def _apply_coverage_min_updates(config: dict[str, Any], updates: dict[str, float] | None) -> dict[str, Any]:
    """Raise age-specific coverage to at least the scenario target.

    Policy scenarios such as "scale up pregnancy Tdap to 75%" should not lower
    countries that already exceed that target.  This helper preserves the
    country-profile value when it is higher than the scenario floor.
    """
    out = deepcopy(config)
    if not updates:
        return out
    for record in out["age_groups"]:
        label = record["label"]
        if label in updates:
            current = _safe_probability(record.get("vaccine_coverage", 0.0))
            target = _safe_probability(updates[label])
            record["vaccine_coverage"] = max(current, target)
    return out


def make_config(
    *,
    vaccine_scenario: str | None = None,
    resistance_scenario: str | None = None,
    country_profile: str | None = None,
    vaccine_overrides: dict[str, Any] | None = None,
    resistance_overrides: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
    load_calibration: bool = True,
    evidence_cutoff_date: str | datetime | pd.Timestamp | None = None,
) -> dict[str, Any]:
    configs = load_configs()
    base = deepcopy(configs["baseline"])
    country_name = country_profile or base.get("baseline_country_profile")

    resolved_evidence_cutoff: pd.Timestamp | None = None
    if evidence_cutoff_date is not None:
        resolved_evidence_cutoff = pd.Timestamp(evidence_cutoff_date)
        if pd.isna(resolved_evidence_cutoff):
            raise ValueError("evidence_cutoff_date must be a valid date")
        if resolved_evidence_cutoff.tzinfo is not None:
            resolved_evidence_cutoff = resolved_evidence_cutoff.tz_localize(None)
        resolved_evidence_cutoff = resolved_evidence_cutoff.normalize()
        if load_calibration:
            raise ValueError(
                "Historical evidence cutoffs cannot load a calibration artifact "
                "that may have used later observations; set load_calibration=False."
            )

    out = deepcopy(base)
    if country_name and country_name in configs["countries"]:
        out = _apply_country_profile_from_profile(out, country_name, configs["countries"][country_name])
        if resolved_evidence_cutoff is not None:
            out = _apply_historical_profile_cutoff(
                out,
                country=country_name,
                country_profile=configs["countries"][country_name],
                data_sources=configs["data_sources"],
                evidence_cutoff_date=resolved_evidence_cutoff,
            )

    calibrated_artifact = (
        load_calibrated_country_artifact(country_name)
        if country_name and load_calibration
        else None
    )
    calibration_overlay_only = False
    calibration_parameter_overlay: dict[str, Any] = {}
    if calibrated_artifact:
        calibrated_config = calibrated_artifact.get("config", calibrated_artifact)
        if isinstance(calibrated_config, dict):
            production_simulation = deepcopy(out.get("simulation", {}))
            production_calendar = deepcopy(out.get("calendar", {}))
            artifact_metadata = calibrated_artifact.get("metadata", {})
            # A calibration artifact is not a second production configuration.
            # Copy only explicitly fitted quantities. This prevents a short calibration calendar, solver
            # tolerances, observation settings, or stale scenario inputs from
            # leaking into forecast runs.  The latent process path is itself a
            # fitted state estimate and is therefore part of this narrow
            # overlay.
            calibration_overlay_only = True
            calibration_parameter_overlay = _calibration_parameter_overlay(calibrated_config)
            out = deep_update(out, calibration_parameter_overlay)
            # Calibration artifacts are produced with a shortened, calendar-aligned
            # runtime. Reuse fitted country parameters, but keep production scenario
            # runs on the configured analysis horizon.
            out["simulation"] = production_simulation
            out["calendar"] = production_calendar
            metadata = out.setdefault("metadata", {})
            metadata["calibration_loaded"] = True
            metadata["calibration_country"] = country_name
            metadata["calibration_artifact_path"] = str(calibrated_country_artifact_path(country_name))
            metadata["calibration_overlay_mode"] = "fitted_parameters_and_state_only"
            for key in (
                "accepted",
                "calibration_status",
                "fit_score",
                "data_fit_score",
                "optimizer_success",
            ):
                if key in artifact_metadata:
                    metadata[f"calibration_{key}"] = artifact_metadata[key]
    else:
        out.setdefault("metadata", {})["calibration_loaded"] = False

    vaccine_name = vaccine_scenario or base["baseline_vaccine_scenario"]
    resistance_name = resistance_scenario or base["baseline_resistance_scenario"]

    vaccine = deepcopy(configs["vaccines"][vaccine_name])
    if vaccine_overrides:
        vaccine = deep_update(vaccine, vaccine_overrides)
    if (
        not calibrated_artifact
        or calibration_overlay_only
        or vaccine_name != base["baseline_vaccine_scenario"]
        or vaccine_overrides
    ):
        out = _apply_vaccine(out, vaccine)

    resistance = deepcopy(configs["resistance"][resistance_name])
    resistance_prevalence_overridden = bool(
        resistance_overrides
        and {
            "target_prevalence_at_analysis_start",
            "initial_resistance_prevalence",
            "importation_fraction",
            "prevalence_anchor_rate_per_year",
        }.intersection(resistance_overrides)
    )
    if resistance_overrides:
        resistance = deep_update(resistance, resistance_overrides)
    if (
        not calibrated_artifact
        or calibration_overlay_only
        or resistance_name != base["baseline_resistance_scenario"]
        or resistance_overrides
    ):
        out = _apply_resistance(out, resistance)

    if config_overrides:
        out = deep_update(out, config_overrides)

    if country_name and country_name in configs["countries"]:
        out = _apply_npi_contact_reduction_timeline(
            out,
            country=country_name,
            country_profile=configs["countries"][country_name],
            data_sources=configs["data_sources"],
            evidence_cutoff_date=resolved_evidence_cutoff,
        )

    if country_name and country_name in configs["countries"]:
        uses_country_timeline = out.get("resistance", {}).get("use_country_resistance_timeline", False)
        if (
            uses_country_timeline
            and not resistance_prevalence_overridden
            and (not calibrated_artifact or calibration_overlay_only)
        ):
            out = _apply_country_resistance_timeline(
                out,
                country=country_name,
                country_profile=configs["countries"][country_name],
                data_sources=configs["data_sources"],
                evidence_cutoff_year=(
                    int(
                        (
                            resolved_evidence_cutoff
                            - pd.Timedelta(nanoseconds=1)
                        ).year
                    )
                    if resolved_evidence_cutoff is not None
                    else None
                ),
            )
    if calibration_overlay_only and calibration_parameter_overlay:
        post_overlay = deepcopy(calibration_parameter_overlay)
        if resistance_overrides:
            post_overlay.pop("importation", None)
            if isinstance(post_overlay.get("resistance"), dict):
                post_overlay["resistance"].pop("importation_fraction", None)
                if not post_overlay["resistance"]:
                    post_overlay.pop("resistance", None)
        out = deep_update(out, post_overlay)

    if country_name and country_name in configs["countries"]:
        out = _apply_diagnostic_standard_timeline(
            out,
            country=country_name,
            country_profile=configs["countries"][country_name],
            data_sources=configs["data_sources"],
            evidence_cutoff_date=resolved_evidence_cutoff,
        )

    if resolved_evidence_cutoff is not None:
        out.setdefault("metadata", {})["evidence_cutoff_date"] = (
            resolved_evidence_cutoff.date().isoformat()
        )

    if sum(float(value) for key, value in out.get("vaccine", {}).items() if key.startswith("VE_")) == 0.0:
        for record in out["age_groups"]:
            record["vaccine_coverage"] = 0.0
        out.setdefault("demography", {})["birth_entry"] = {"S": 1.0, "V": 0.0}
    # Materialize the transmission-layer diagnosis probabilities before any
    # downstream reporting-only scenario mutates age-specific reporting rates.
    # PreparedParameters historically defaulted diagnosis to the *current*
    # reporting_rate values, which made observation-layer sensitivity analyses
    # silently alter treatment and resistance dynamics.
    if "diagnosis_probability" not in out:
        out["diagnosis_probability"] = {
            str(record["label"]): float(record.get("reporting_rate", 0.0))
            for record in out["age_groups"]
        }
    out.setdefault("reporting_multiplier", 1.0)
    return out


def _without_prospective_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow model-config view without orchestration metadata."""

    return {key: value for key, value in config.items() if key != PROSPECTIVE_POLICY_KEY}


def prospective_policy_history(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return a defensive copy of the historical config attached to a policy."""

    spec = config.get(PROSPECTIVE_POLICY_KEY)
    if not isinstance(spec, dict):
        return None
    history = spec.get("history_config")
    if not isinstance(history, dict):
        raise ValueError("Prospective-policy metadata is missing a historical config")
    return deepcopy(_without_prospective_policy(history))


def attach_prospective_policy_history(
    policy_config: dict[str, Any],
    history_config: dict[str, Any],
    *,
    history_vaccine_scenario: str = "",
    history_resistance_scenario: str = "",
) -> dict[str, Any]:
    """Attach an explicit pre-policy configuration to a prospective scenario.

    This private orchestration block is removed before constructing
    ``PreparedParameters``.  Keeping it beside the config makes direct callers
    scientifically safe, while scenario executors can extract and deduplicate
    it to share one t0 state across many policies.
    """

    out = deepcopy(_without_prospective_policy(policy_config))
    history = deepcopy(_without_prospective_policy(history_config))
    policy_start = float(out.get("simulation", {}).get("start_time", 0.0))
    out[PROSPECTIVE_POLICY_KEY] = {
        "schema_version": PROSPECTIVE_POLICY_SCHEMA_VERSION,
        "start_time": policy_start,
        "history_config": history,
        "history_vaccine_scenario": str(history_vaccine_scenario),
        "history_resistance_scenario": str(history_resistance_scenario),
    }
    metadata = out.setdefault("metadata", {})
    metadata["prospective_policy"] = True
    metadata["policy_start_time"] = policy_start
    metadata["simulation_phase"] = "prospective_policy"
    return out


def apply_intervention_definition(
    config: dict[str, Any],
    intervention: dict[str, Any],
    *,
    history_config: dict[str, Any] | None = None,
    history_vaccine_scenario: str = "",
    history_resistance_scenario: str = "",
) -> dict[str, Any]:
    """Apply one intervention definition to an already prepared config.

    This is used for both single predefined profiles and supplementary
    portfolio analyses where several program levers are layered in one run.
    """
    existing_history = prospective_policy_history(config)
    historical = deepcopy(
        history_config
        if history_config is not None
        else existing_history
        if existing_history is not None
        else _without_prospective_policy(config)
    )
    config = deepcopy(_without_prospective_policy(config))
    coverage_updates = intervention.get("coverage_updates", {})
    coverage_min_updates = intervention.get("coverage_min_updates", {})
    config = _apply_coverage_updates(config, coverage_updates)
    config = _apply_coverage_min_updates(config, coverage_min_updates)
    if "vaccine_overrides" in intervention:
        # If the intervention specifies maternal_VE_* keys, apply them to the
        # immunity_model section instead of overriding global VE parameters.
        # This ensures maternal immunization only affects the M_protected
        # compartment's protection level, not all vaccine-origin states.
        overrides = dict(intervention["vaccine_overrides"])
        maternal_ve_keys = {k: v for k, v in overrides.items() if k.startswith("maternal_")}
        global_ve_keys = {k: v for k, v in overrides.items() if not k.startswith("maternal_")}
        if global_ve_keys:
            config["vaccine"] = deep_update(config["vaccine"], global_ve_keys)
        if maternal_ve_keys:
            immunity = config.setdefault("immunity_model", {})
            for key, value in maternal_ve_keys.items():
                # maternal_VE_sus -> stored as maternal_VE_sus in immunity_model
                immunity[key] = float(value)
    if "treatment_updates" in intervention:
        config["treatment"] = deep_update(config["treatment"], intervention["treatment_updates"])
    if "pep_updates" in intervention:
        config["PEP"] = deep_update(config["PEP"], intervention["pep_updates"])
    if "natural_history_overrides" in intervention:
        config["natural_history"] = deep_update(
            config["natural_history"], intervention["natural_history_overrides"]
        )

    # If the intervention updates infant_0_2m coverage, propagate to birth_entry
    # so that newborns actually enter M_protected at the specified maternal
    # immunization coverage rate. Without this, the coverage_updates value for
    # infant_0_2m only affects routine vaccination (which returns {} for that
    # age group), leaving birth_entry unchanged and maternal protection ineffective.
    if "infant_0_2m" in coverage_updates:
        maternal_cov = _safe_probability(coverage_updates["infant_0_2m"])
    elif "infant_0_2m" in coverage_min_updates:
        existing_birth = _safe_probability(
            config.get("demography", {}).get("birth_entry", {}).get("V", 0.0)
        )
        maternal_cov = max(existing_birth, _safe_probability(coverage_min_updates["infant_0_2m"]))
    else:
        maternal_cov = None
    if maternal_cov is not None:
        demography = config.setdefault("demography", {})
        demography["birth_entry"] = {"S": float(1.0 - maternal_cov), "V": maternal_cov}

    # Apply close-contact exposure reduction: reduce effective contacts.
    # from a source age group to target age groups, representing the reduced
    # transmission from vaccinated household contacts (primarily mothers) to infants.
    cmr = intervention.get("contact_matrix_reduction")
    if cmr:
        age_labels = [record["label"] for record in config["age_groups"]]
        source_age = cmr["source_age"]
        target_ages = cmr["target_ages"]
        reduction = float(cmr["reduction_fraction"])
        if source_age in age_labels:
            source_idx = age_labels.index(source_age)
            rows = config["contact_matrix"]["rows"]
            for target_age in target_ages:
                if target_age in age_labels:
                    target_idx = age_labels.index(target_age)
                    # Reduce contacts FROM source TO target (row=target, col=source
                    # in the "who-acquires-infection-from-whom" convention)
                    rows[target_idx][source_idx] *= (1.0 - reduction)
        correction = config.setdefault("contact_matrix", {}).setdefault("reciprocity_correction", {})
        correction["enabled"] = False
        metadata = config.setdefault("metadata", {})
        metadata["contact_matrix_directional_intervention"] = True
        metadata["contact_matrix_reciprocity_correction_after_intervention"] = False

    effect_scope = str(intervention.get("effect_scope", "prospective")).lower()
    if effect_scope in {"structural", "historical", "all_time"}:
        config.setdefault("metadata", {})["prospective_policy"] = False
        config["metadata"]["simulation_phase"] = "structural_history"
        return config
    if effect_scope != "prospective":
        raise ValueError(
            f"Unsupported intervention effect_scope={effect_scope!r}; "
            "use 'prospective' or 'structural'"
        )
    return attach_prospective_policy_history(
        config,
        historical,
        history_vaccine_scenario=history_vaccine_scenario,
        history_resistance_scenario=history_resistance_scenario,
    )


def make_intervention_config(
    name: str,
    *,
    country_profile: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    configs = load_configs()
    intervention = configs["interventions"][name]
    baseline_vaccine_name = str(configs["baseline"].get("baseline_vaccine_scenario"))
    baseline_resistance_name = str(configs["baseline"].get("baseline_resistance_scenario"))
    vaccine_name = intervention.get("vaccine_scenario", baseline_vaccine_name)
    history_config = make_config(
        vaccine_scenario=baseline_vaccine_name,
        resistance_scenario=baseline_resistance_name,
        country_profile=country_profile,
        config_overrides=config_overrides,
    )
    config = make_config(
        vaccine_scenario=vaccine_name,
        resistance_scenario=baseline_resistance_name,
        country_profile=country_profile,
        config_overrides=config_overrides,
    )
    config = apply_intervention_definition(
        config,
        intervention,
        history_config=history_config,
        history_vaccine_scenario=baseline_vaccine_name,
        history_resistance_scenario=baseline_resistance_name,
    )
    return config, vaccine_name


def _apply_country_profile_from_profile(config: dict[str, Any], country: str, profile: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    for record in out["age_groups"]:
        label = record["label"]
        if label in profile.get("population", {}):
            record["population"] = float(profile["population"][label])
        if label in profile.get("reporting_rate", {}):
            record["reporting_rate"] = float(profile["reporting_rate"][label])
            pep_detection_profile = profile.get("pep_detection_rate", profile.get("reporting_rate", {}))
            record["pep_detection_rate"] = float(
                pep_detection_profile.get(label, record["reporting_rate"])
            )
        if label in profile.get("vaccine_coverage", {}):
            record["vaccine_coverage"] = float(profile["vaccine_coverage"][label])
    if "reporting_rate_prior" in profile:
        out["reporting_rate_prior"] = profile["reporting_rate_prior"]
    if "contact_matrix" in profile:
        out["contact_matrix"]["rows"] = profile["contact_matrix"]
    if "contact_reciprocity" in profile:
        out.setdefault("metadata", {})["contact_reciprocity"] = profile["contact_reciprocity"]
    if "vaccine_schedule" in profile:
        schedule = profile.get("vaccine_schedule", {})
        metadata = out.setdefault("metadata", {})
        for key in (
            "routine_age_pattern",
            "routine_first_shot_months",
            "routine_last_shot_months",
            "routine_dose_count",
            "routine_scheduler_code",
        ):
            if key in schedule:
                metadata[key] = schedule[key]
    if "birth_entry" in profile:
        out.setdefault("demography", {})["birth_entry"] = profile["birth_entry"]
    if "demography_trajectory" in profile:
        demography = out.setdefault("demography", {})
        demography["wpp_trajectory"] = deepcopy(profile["demography_trajectory"])
        # Signal to the ODE that population is driven by WPP rather than held
        # fixed at a single snapshot. The caller can still disable this in tests
        # by resetting demography["mode"] back to "fixed_population_profile".
        demography.setdefault("mode", "wpp_trajectory")
    if "transmission_overrides" in profile:
        out["transmission"] = deep_update(out["transmission"], profile["transmission_overrides"])
    out["country"] = country
    return out


def _prepare_run_context(
    config: dict[str, Any],
    *,
    analysis: str,
    scenario: str,
    vaccine_scenario: str = "",
    resistance_scenario: str = "",
    intervention: str = "",
    metadata: dict[str, Any] | None = None,
    history_config: dict[str, Any] | None = None,
    initial_state_override: np.ndarray | None = None,
) -> tuple[
    PreparedParameters,
    StateIndex,
    PreparedParameters | None,
    dict[str, Any],
]:
    """Resolve policy/history parameters once for every simulation surface.

    Keeping this logic shared is scientifically important: full publication
    output and the lightweight likelihood path must start from exactly the
    same historical state and must apply the same prospective-policy boundary.
    """

    embedded_spec = config.get(PROSPECTIVE_POLICY_KEY)
    embedded_history = prospective_policy_history(config)
    if history_config is not None and embedded_history is not None:
        # Explicit orchestration wins, but reject a mismatched policy origin.
        explicit_payload = json.dumps(
            _without_prospective_policy(history_config),
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        embedded_payload = json.dumps(
            embedded_history,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        if explicit_payload != embedded_payload:
            raise ValueError("Explicit history_config disagrees with embedded policy history")
    resolved_history = history_config or embedded_history
    model_config = _without_prospective_policy(config)
    params = PreparedParameters.from_config(
        model_config,
        analysis=analysis,
        scenario=scenario,
        vaccine_scenario=vaccine_scenario,
        resistance_scenario=resistance_scenario,
        intervention=intervention,
        metadata=metadata,
    )
    index = StateIndex(params.age_groups)
    history_params: PreparedParameters | None = None
    history_model_config: dict[str, Any] | None = None
    if resolved_history is not None:
        history_model_config = _without_prospective_policy(resolved_history)
        if float(history_model_config["simulation"]["start_time"]) != float(
            model_config["simulation"]["start_time"]
        ):
            raise ValueError("Historical and policy simulation.start_time must match")
        history_calendar_start = str(
            history_model_config.get("calendar", {}).get("analysis_start_date", "")
        )
        policy_calendar_start = str(model_config.get("calendar", {}).get("analysis_start_date", ""))
        if history_calendar_start != policy_calendar_start:
            raise ValueError("Historical and policy calendar.analysis_start_date must match")
        history_age_groups = tuple(
            str(record["label"]) for record in history_model_config.get("age_groups", [])
        )
        if history_age_groups != tuple(params.age_groups):
            raise ValueError("Historical and policy age-group order must match")
    if history_model_config is not None and initial_state_override is None:
        history_vaccine = ""
        history_resistance = ""
        if isinstance(embedded_spec, dict):
            history_vaccine = str(embedded_spec.get("history_vaccine_scenario", ""))
            history_resistance = str(embedded_spec.get("history_resistance_scenario", ""))
        history_params = PreparedParameters.from_config(
            history_model_config,
            analysis=analysis,
            scenario=f"{scenario}__history",
            vaccine_scenario=history_vaccine,
            resistance_scenario=history_resistance,
            intervention="historical_current_practice",
            metadata=metadata,
        )
    return params, index, history_params, model_config


def run_prepared_case_exposure(
    config: dict[str, Any],
    observed: pd.DataFrame,
    *,
    analysis: str,
    scenario: str,
    vaccine_scenario: str = "",
    resistance_scenario: str = "",
    intervention: str = "",
    metadata: dict[str, Any] | None = None,
    history_config: dict[str, Any] | None = None,
    initial_state_override: np.ndarray | None = None,
    case_event_definition: str | None = None,
) -> tuple[CaseExposure, np.ndarray]:
    """Solve only biology plus exact observed-interval case counters.

    The returned age-specific reporting rates are the unmultiplied base
    probabilities.  Reporting multipliers and diagnostic standards can
    therefore be projected repeatedly without another ODE solve.
    """

    params, index, history_params, model_config = _prepare_run_context(
        config,
        analysis=analysis,
        scenario=scenario,
        vaccine_scenario=vaccine_scenario,
        resistance_scenario=resistance_scenario,
        intervention=intervention,
        metadata=metadata,
        history_config=history_config,
        initial_state_override=initial_state_override,
    )
    if initial_state_override is None:
        t0_state = prepare_history_state(history_params or params, index)
    else:
        t0_state = np.array(initial_state_override, dtype=float, copy=True)
    plan = build_observation_plan(observed, params)
    exposure = solve_case_exposure(
        params,
        index,
        t0_state,
        plan,
        case_event_definition=case_event_definition,
    )
    base_reporting_rates = np.asarray(
        [float(record.get("reporting_rate", 0.0)) for record in model_config["age_groups"]],
        dtype=float,
    )
    return exposure, base_reporting_rates


def run_prepared_config(
    config: dict[str, Any],
    *,
    analysis: str,
    scenario: str,
    vaccine_scenario: str = "",
    resistance_scenario: str = "",
    intervention: str = "",
    metadata: dict[str, Any] | None = None,
    history_config: dict[str, Any] | None = None,
    initial_state_override: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params, index, history_params, _model_config = _prepare_run_context(
        config,
        analysis=analysis,
        scenario=scenario,
        vaccine_scenario=vaccine_scenario,
        resistance_scenario=resistance_scenario,
        intervention=intervention,
        metadata=metadata,
        history_config=history_config,
        initial_state_override=initial_state_override,
    )
    solution = solve_model(
        params,
        index,
        history_params=history_params,
        initial_state_override=initial_state_override,
    )
    timeseries = compute_timeseries(solution, params, index)
    summary = summarize_timeseries(
        timeseries,
        dt=infer_output_dt(timeseries),
    )
    for key, value in params.metadata.items():
        if isinstance(value, (str, int, float, bool, np.number)):
            timeseries[key] = value
            summary[key] = value
    for key, value in (metadata or {}).items():
        timeseries[key] = value
        summary[key] = value
    _add_absolute_fit_context(summary, params.raw, metadata or {})
    return timeseries, summary


def _add_absolute_fit_context(summary: pd.DataFrame, config: dict[str, Any], metadata: dict[str, Any]) -> None:
    country = metadata.get("country") or config.get("country")
    if not country:
        summary["absolute_fit_status"] = "not_country_specific"
        return
    configs = load_configs()
    countries = configs["countries"]
    observed = countries.get(str(country), {}).get("observed_incidence", {})
    observed_mean = metadata.get(
        "observed_mean_annual_reported_incidence_per_100k",
        observed.get("observed_mean_annual_reported_incidence_per_100k", np.nan),
    )
    summary["observed_mean_annual_reported_incidence_per_100k"] = observed_mean
    modeled = summary["annualized_reported_cases_per_100k"].astype(float)
    if np.isfinite(float(observed_mean)) and float(observed_mean) > 0:
        summary["model_to_observed_reported_incidence_ratio"] = modeled / float(observed_mean)
    else:
        summary["model_to_observed_reported_incidence_ratio"] = np.nan
    tolerance = float(configs["baseline"].get("calibration", {}).get("relative_incidence_tolerance", 0.25))
    ratio = summary["model_to_observed_reported_incidence_ratio"].astype(float)
    summary["absolute_fit_relative_error"] = (ratio - 1.0).abs()
    summary["absolute_fit_relative_tolerance"] = tolerance
    calibration_loaded = bool(config.get("metadata", {}).get("calibration_loaded", False))
    summary["calibration_loaded"] = calibration_loaded
    is_calibration = summary["analysis"].eq("calibration")
    is_within_tolerance = summary["absolute_fit_relative_error"].le(tolerance) & np.isfinite(
        summary["absolute_fit_relative_error"].astype(float)
    )
    default_status = "calibrated_country_scenario_analysis" if calibration_loaded else "uncalibrated_scenario_analysis"
    summary["absolute_fit_status"] = np.select(
        [is_calibration & is_within_tolerance, is_calibration & ~is_within_tolerance],
        ["calibrated_to_reported_cases", "calibration_failed_absolute_fit"],
        default=default_status,
    )


def _history_config_key(history_config: dict[str, Any]) -> str:
    """Canonical within-run key for sharing an identical historical trajectory."""

    payload = deepcopy(_without_prospective_policy(history_config))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("prospective_policy", None)
        metadata.pop("policy_start_time", None)
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _prepare_history_config_item(item: dict[str, Any]) -> np.ndarray:
    history_config = _without_prospective_policy(item["history_config"])
    params = PreparedParameters.from_config(
        history_config,
        analysis=str(item.get("analysis", "prospective_history")),
        scenario=f"{item.get('scenario', 'policy')}__history",
        vaccine_scenario=str(item.get("history_vaccine_scenario", "")),
        resistance_scenario=str(item.get("history_resistance_scenario", "")),
        intervention="historical_current_practice",
        metadata=item.get("metadata"),
    )
    index = StateIndex(params.age_groups)
    return prepare_history_state(params, index)


def _prepare_prospective_scenario_items(
    scenarios: list[dict[str, Any]],
    *,
    stem: str,
    n_jobs: int | None,
) -> list[dict[str, Any]]:
    """Compute each unique history once, then attach copied t0 states.

    The grouping occurs before loky dispatch.  A process-local dictionary cache
    would not guarantee that related policies reach the same worker and would
    therefore fail to remove duplicate burn-ins.
    """

    prepared = [dict(item) for item in scenarios]
    history_tasks: dict[str, dict[str, Any]] = {}
    item_history_keys: dict[int, str] = {}
    for position, item in enumerate(prepared):
        if item.get("initial_state_override") is not None:
            continue
        explicit_history = item.get("history_config")
        embedded_history = prospective_policy_history(item["config"])
        history = explicit_history if isinstance(explicit_history, dict) else embedded_history
        if history is None:
            continue
        key = _history_config_key(history)
        item_history_keys[position] = key
        if key in history_tasks:
            continue
        spec = item["config"].get(PROSPECTIVE_POLICY_KEY, {})
        history_tasks[key] = {
            "history_config": history,
            "analysis": item.get("analysis", "prospective_history"),
            "scenario": item.get("scenario", "policy"),
            "history_vaccine_scenario": (
                spec.get("history_vaccine_scenario", "") if isinstance(spec, dict) else ""
            ),
            "history_resistance_scenario": (
                spec.get("history_resistance_scenario", "") if isinstance(spec, dict) else ""
            ),
            "metadata": item.get("metadata"),
        }
    if not history_tasks:
        return prepared

    keys = list(history_tasks)
    states = parallel_map(
        _prepare_history_config_item,
        [history_tasks[key] for key in keys],
        desc=f"{stem}_history",
        n_jobs=n_jobs,
    )
    state_by_key = {key: state for key, state in zip(keys, states)}
    for position, key in item_history_keys.items():
        # Every policy receives an independent writable copy.  This is cheap
        # (~4.7 KiB for the current 592-state model) and prevents cross-policy
        # aliasing even if a future integrator mutates its input.
        prepared[position]["initial_state_override"] = np.array(
            state_by_key[key], dtype=float, copy=True
        )
    return prepared


def _run_scenario_item(item: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_prepared_config(**item)


def _run_scenario_summary_item(item: dict[str, Any]) -> pd.DataFrame:
    _, summary = run_prepared_config(**item)
    return summary


def _scenario_execution_n_jobs(
    scenarios: list[dict[str, Any]],
    requested: int | None,
) -> int | None:
    """Resolve a top-level scenario budget without defeating an env cap.

    A non-``None`` argument is an explicit allocation, including the smaller
    budgets passed by nested schedulers.  Otherwise ``PERTUSSIS_N_JOBS`` is the
    top-level run budget and must take precedence over a configuration default
    such as ``simulation.n_jobs: -1``.  Returning ``None`` in that case lets
    :func:`parallel_map` resolve and cap the environment value in one place.
    """

    if requested is not None:
        return requested
    if os.environ.get("PERTUSSIS_N_JOBS"):
        return None
    return scenarios[0]["config"].get("simulation", {}).get("n_jobs")


def add_relative_reductions(
    summary: pd.DataFrame,
    *,
    reference_scenario: str,
) -> pd.DataFrame:
    out = summary.copy()
    mapping = {
        "relative_reduction_infant_cases": "total_infant_cases",
        "relative_reduction_child_1_9_cases": "total_child_1_9_cases",
        "relative_reduction_adolescent_cases": "total_adolescent_cases",
        "relative_reduction_child_adolescent_cases": "total_child_adolescent_cases",
        "relative_reduction_child_adolescent_infections": "total_child_adolescent_infections",
        "relative_reduction_child_adolescent_reported_cases": "total_child_adolescent_reported_cases",
        "relative_reduction_total_infections": "total_infections",
        "relative_reduction_reported_cases": "total_reported_cases",
        "relative_reduction_resistant_infections": "resistant_infections",
        "relative_reduction_deaths": "total_deaths",
        "relative_reduction_infant_deaths": "total_infant_deaths",
        "relative_reduction_child_adolescent_deaths": "total_child_adolescent_deaths",
    }
    for new_col in mapping:
        out[new_col] = np.nan

    if "country" not in out.columns:
        grouped = [(None, out.index)]
    else:
        country_groups = [(key, group.index) for key, group in out.groupby(["country"], dropna=False)]
        every_country_has_reference = all(
            out.loc[idx, "scenario"].eq(reference_scenario).any()
            for _, idx in country_groups
        )
        grouped = country_groups if every_country_has_reference else [(None, out.index)]

    for _, idx in grouped:
        group = out.loc[idx]
        reference = group.loc[group["scenario"].eq(reference_scenario)]
        if reference.empty:
            continue
        base = reference.iloc[0]
        for new_col, source_col in mapping.items():
            if source_col not in out.columns:
                continue
            denom = float(base[source_col])
            out.loc[idx, new_col] = 1.0 - out.loc[idx, source_col] / denom if denom > 0 else np.nan
    out["relative_reduction_vs_baseline"] = out["relative_reduction_total_infections"]
    return out


def write_outputs(
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    stem: str,
    *,
    extra_metadata: dict[str, Any] | None = None,
    require_calibrated: bool = True,
) -> None:
    if require_calibrated:
        enforce_calibration_status(summary, stem=stem)
        if {"country", "calibration_loaded"}.issubset(summary.columns):
            calibrated_countries = tuple(
                sorted(
                    summary.loc[
                        summary["calibration_loaded"].eq(True).fillna(False), "country"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )
            if calibrated_countries:
                validate_calibration_artifacts(
                    calibrated_countries,
                    context=f"Output {stem}",
                )
    resolved_extra_metadata = dict(extra_metadata or {})
    ensure_output_dirs()
    project_path("outputs", "metadata").mkdir(parents=True, exist_ok=True)
    _clear_stem_outputs(stem)
    write_dataframe(timeseries, project_path(f"outputs/simulations/{stem}.parquet"))
    write_dataframe(summary, project_path(f"outputs/summaries/{stem}_summary.csv"))
    metadata = current_run_metadata(
        stem,
        row_counts={
            "timeseries": int(len(timeseries)),
            "summary": int(len(summary)),
        },
    )
    if resolved_extra_metadata:
        metadata.update(resolved_extra_metadata)
    write_run_metadata(stem, metadata)


def execute_scenario_list(
    scenarios: list[dict[str, Any]],
    *,
    stem: str,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not scenarios:
        raise ValueError(f"No scenarios were provided for {stem}.")
    n_jobs = _scenario_execution_n_jobs(scenarios, n_jobs)
    prepared_scenarios = _prepare_prospective_scenario_items(
        scenarios,
        stem=stem,
        n_jobs=n_jobs,
    )
    results = parallel_map(_run_scenario_item, prepared_scenarios, desc=stem, n_jobs=n_jobs)
    frames = [ts for ts, _ in results]
    summaries = [sm for _, sm in results]
    return pd.concat(frames, ignore_index=True), pd.concat(summaries, ignore_index=True)


def execute_scenario_summary_list(
    scenarios: list[dict[str, Any]],
    *,
    stem: str,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    if not scenarios:
        raise ValueError(f"No scenarios were provided for {stem}.")
    n_jobs = _scenario_execution_n_jobs(scenarios, n_jobs)
    prepared_scenarios = _prepare_prospective_scenario_items(
        scenarios,
        stem=stem,
        n_jobs=n_jobs,
    )
    summaries = parallel_map(
        _run_scenario_summary_item,
        prepared_scenarios,
        desc=stem,
        n_jobs=n_jobs,
    )
    return pd.concat(summaries, ignore_index=True)


def run_scenario_list(
    scenarios: list[dict[str, Any]],
    *,
    stem: str,
    reference_scenario: str,
    n_jobs: int | None = None,
    require_calibrated: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeseries, summary = execute_scenario_list(scenarios, stem=stem, n_jobs=n_jobs)
    summary = add_relative_reductions(summary, reference_scenario=reference_scenario)
    if require_calibrated:
        enforce_calibration_status(summary, stem=stem)
    write_outputs(timeseries, summary, stem)
    return timeseries, summary


def enforce_calibration_status(summary: pd.DataFrame, *, stem: str) -> None:
    """Raise if any scenario in a production summary is uncalibrated.

    This hard check prevents the pipeline from producing outputs labelled as
    calibrated country analyses when accepted calibration artifacts are missing.
    Scenarios explicitly marked as exploratory are exempt.
    """
    if "calibration_loaded" not in summary.columns:
        return
    uncalibrated = summary.loc[
        summary["calibration_loaded"].eq(False)
        & ~summary.get("analysis", pd.Series(dtype=str)).str.contains("exploratory|test|validation", na=True)
    ]
    if not uncalibrated.empty:
        countries = sorted(uncalibrated.get("country", pd.Series(["unknown"])).unique())
        raise RuntimeError(
            f"[{stem}] Calibration enforcement failed: {len(uncalibrated)} scenario(s) "
            f"for countries {countries} have calibration_loaded=False. "
            f"Run calibration first (python -m src_python.calibration.run_all) or "
            f"set require_calibrated=False for exploratory analyses."
        )
def write_manuscript_tables() -> None:
    configs = load_configs()
    baseline = configs["baseline"]
    vaccines = configs["vaccines"]
    resistance = configs["resistance"]
    interventions = configs["interventions"]
    countries = configs["countries"]
    settings = configs.get("settings", {})
    parameter_sources = settings.get("parameter_sources", {})

    distribution_registry = configs.get("parameter_distributions", {})
    evidence_psa = (
        distribution_registry.get("global_sensitivity", {})
        if isinstance(distribution_registry, dict)
        else {}
    )
    evidence_specs = evidence_psa.get("parameters", {})
    if not isinstance(evidence_specs, dict) or not evidence_specs:
        raise ValueError(
            "Manuscript parameter tables require the canonical "
            "parameter_distributions.global_sensitivity.parameters registry; "
            "legacy sensitivity-range fallback is disabled."
        )
    selected_sensitivity_specs = evidence_specs
    sensitivity_paths = {
        spec.get("path")
        for spec in selected_sensitivity_specs.values()
        if isinstance(spec, dict) and spec.get("path")
    }
    sensitivity_by_path = {
        str(spec.get("path")): spec
        for spec in selected_sensitivity_specs.values()
        if isinstance(spec, dict) and spec.get("path")
    }
    if evidence_specs:
        sensitivity_path_aliases: dict[str, str] = {}
        semantic_sensitivity_by_path = {
            "vaccine.VE_sym": evidence_specs.get("vaccine_disease_efficacy"),
            "natural_history.infectious_duration_asymptomatic": evidence_specs.get(
                "asymptomatic_duration_ratio"
            ),
            "natural_history.R_to_W_duration": evidence_specs.get("natural_immunity_duration"),
            "natural_history.W_to_S_duration": evidence_specs.get("natural_immunity_duration"),
            "natural_history.vaccine_protection_duration": evidence_specs.get(
                "vaccine_protection_duration"
            ),
            "immunity_model.waned_vaccine_duration": evidence_specs.get(
                "vaccine_protection_duration"
            ),
        }
        semantic_sensitivity_by_path = {
            path: spec for path, spec in semantic_sensitivity_by_path.items() if isinstance(spec, dict)
        }
    else:
        sensitivity_path_aliases = {
            "natural_history.recovered_immunity_duration": "rates.waning_natural",
            "natural_history.vaccine_protection_duration": "rates.waning_vaccine",
        }
        semantic_sensitivity_by_path = {}
    parameter_source_aliases = {
        "vaccine.VE_sus": "vaccine_scenarios",
        "vaccine.VE_sym": "vaccine_scenarios",
        "vaccine.VE_inf": "vaccine_scenarios",
        "vaccine.VE_dur": "vaccine_scenarios",
        "observation.reporting_probability_baseline": "observation_model",
    }
    parameter_source_overrides = {
        "simulation.output_time_step": {
            "source": "Analysis design",
            "note": "Weekly saved output supports annualised summaries while calibration can use coarser monthly saved output for speed.",
        },
        "simulation.rtol": {
            "source": "Numerical implementation setting",
            "note": "Production tolerance used by the RK45 solver; calibration uses a slightly looser tolerance for repeated likelihood evaluations.",
        },
        "simulation.atol": {
            "source": "Numerical implementation setting",
            "note": "Production absolute tolerance used by the RK45 solver; calibration uses a slightly looser tolerance for repeated likelihood evaluations.",
        },
        "transmission.seasonal_phase": {
            "source": "Calibrated/processed pertussis incidence",
            "note": "Country profiles override the shared default when seasonal timing can be inferred from reported-case timing.",
        },
        "transmission.relative_infectiousness_asymptomatic": {
            "source": "Literature-informed mechanism assumption",
            "note": "Human household evidence does not identify this ratio; it is varied only in dedicated structural diagnostics.",
        },
        "transmission.fitness_R": {
            "source": "Weak surveillance-informed model prior",
            "note": "Baseline is fitness-neutral; surveillance does not separately identify biological transmission fitness.",
        },
        "transmission.multi_year_amplitude": {
            "source": "Model-structure assumption",
            "note": "Weak phase-locking is a zero-versus-enabled structural choice rather than a literature-estimated continuous parameter.",
        },
        "transmission.multi_year_phase": {
            "source": "Model-structure assumption",
            "note": "Optional multi-year recurrence phase is fixed in the submitted primary analysis.",
        },
        "importation.rate_per_100k_per_year": {
            "source": "Importation implementation assumption",
            "note": "Low-level imported infection seeding prevents deterministic extinction and is bounded in calibration support.",
        },
        "natural_history.maternal_protection_duration": {
            "source": "Pregnancy-vaccination effectiveness evidence and scenario assumption",
            "note": "Baseline passive maternal-protection duration is short-lived; pregnancy Tdap and the optional legacy nonpublication research intervention prior vary the duration.",
        },
        "natural_history.latent_duration": {
            "source": "Interval-censored pertussis incubation evidence",
            "note": "Two outbreak analyses give materially different incubation estimates; their broad synthesis is an explicit proxy for the unobserved latent-period mean.",
        },
        "natural_history.infectious_duration_symptomatic": {
            "source": "SIRWS pertussis model fit",
            "note": "The evidence-prior PSA encodes uncertainty around a fitted 3.7-week mean.",
        },
        "natural_history.infectious_duration_asymptomatic": {
            "source": "Weak model prior",
            "note": "Human studies establish asymptomatic infection but do not identify its duration relative to symptomatic infection.",
        },
        "natural_history.recovered_immunity_duration": {
            "source": "Legacy no-boosting fallback",
            "note": "This field is inactive when SIRWS boosting is enabled; active natural-immunity uncertainty is applied to the R-to-W and W-to-S stages.",
        },
        "natural_history.vaccine_protection_duration": {
            "source": "aP waning literature-to-model mapping",
            "note": "This is an explicit duration mapping rather than a directly observed duration distribution.",
        },
        "natural_history.R_to_W_duration": {
            "source": "Pertussis waning and immune-boosting model evidence",
            "note": "Equal R and W stages approximate an Erlang shape-2 total natural-immunity residence time when boosting is absent.",
        },
        "natural_history.W_to_S_duration": {
            "source": "Pertussis waning and immune-boosting model evidence",
            "note": "Equal R and W stages approximate an Erlang shape-2 total natural-immunity residence time when boosting is absent.",
        },
        "immunity_model.boosting_efficiency": {
            "source": "SIRWS immune-boosting model assumption",
            "note": "Controls what fraction of re-exposure events boost waned-natural immunity back toward the recovered state.",
        },
        "immunity_model.waned_natural_infection_susceptibility": {
            "source": "SIRWS immune-boosting model assumption",
            "note": "Susceptibility scalar for waned-natural-immunity breakthrough infection after failed boosting.",
        },
        "immunity_model.waned_relative_effect": {
            "source": "Vaccine-origin mapping assumption",
            "note": "Residual vaccine-effect weight assigned to waned vaccine-origin states.",
        },
        "immunity_model.maternal_relative_effect": {
            "source": "Maternal-origin mapping assumption",
            "note": "Relative vaccine-effect weight assigned to the maternal-protection origin.",
        },
        "immunity_model.dose1_relative_effect": {
            "source": "Dose-history mapping assumption",
            "note": "Relative vaccine-effect weight assigned after one primary-series dose.",
        },
        "immunity_model.dose2_relative_effect": {
            "source": "Dose-history mapping assumption",
            "note": "Relative vaccine-effect weight assigned after two primary-series doses.",
        },
        "immunity_model.waned_vaccine_duration": {
            "source": "aP waning literature-to-model mapping",
            "note": "The sampled total duration is split equally between recent and waned vaccine-origin stages.",
        },
        "immunity_model.initial_recent_fraction": {
            "source": "Initial immunity-history mapping assumption",
            "note": "Shared fallback for the fraction of vaccinated-origin individuals initialized as recently protected; age-specific values override it.",
        },
        "vaccine.VE_sus": {
            "source": "Literature-informed vaccine-mechanism scenario",
            "note": "Human effectiveness studies do not separately identify acquisition and symptom effects; VE_sus is fixed by the selected mechanism scenario in the evidence-prior PSA.",
        },
        "vaccine.VE_sym": {
            "source": "Household-exposure overall vaccine-effect evidence",
            "note": "VE_sym is derived conditional on scenario VE_sus so acquisition and symptom effects are not sampled independently.",
        },
        "vaccine.VE_inf": {
            "source": "Literature-informed vaccine-mechanism scenario",
            "note": "Human evidence supports residual transmission but does not identify an independent VE_inf distribution.",
        },
        "vaccine.VE_dur": {
            "source": "Vaccine-mechanism scenario assumption",
            "note": "No human study identifies an independent infectious-duration vaccine effect.",
        },
        "observation.reporting_probability_baseline": {
            "source": "Age-specific reporting priors and country calibration",
            "note": "Baseline age-specific reporting probabilities are country-calibrated and reported in the fitted reporting-probability table.",
        },
        "reporting_multiplier": {
            "source": "Calibrated observation layer plus transfer prior",
            "note": "The evidence-prior factor is applied around the calibrated country value and assessed only against reported cases.",
        },
        "calibration.dispersion": {
            "source": "Negative-binomial observation-model assumption",
            "note": "Dispersion parameter used in the reported-case calibration likelihood.",
        },
        "calibration.recent_years": {
            "source": "Calibration design",
            "note": "Number of recent surveillance years used by the staged calibration objective when available.",
        },
        "calibration.relative_incidence_tolerance": {
            "source": "Calibration acceptance criterion",
            "note": "Maximum tolerated relative deviation of modeled from observed mean annualised reported incidence during retained calibration.",
        },
        "resistance.target_prevalence_at_analysis_start": {
            "source": "Country resistance evidence and fixed stress-test scenarios",
            "note": "Country-timeline analyses use the latest admissible country-specific resistance anchor; fixed scenarios retain low-to-very-high contrasts.",
        },
        "resistance.importation_fraction": {
            "source": "Country resistance evidence and fixed stress-test scenarios",
            "note": "The resistant fraction among imports follows the country-timeline or fixed resistance-scenario anchor.",
        },
        "resistance.prevalence_anchor_rate_per_year": {
            "source": "Resistance anchoring implementation assumption",
            "note": "Burn-in rebalance rate used to move strain composition toward the evidence-based analysis-start anchor.",
        },
        "resistance.rebalance_after_burn_in": {
            "source": "Resistance anchoring implementation assumption",
            "note": "Submitted country-timeline analyses rebalance resistant prevalence after burn-in to match the evidence anchor.",
        },
        "routine_vaccination.target_relaxation_rate_per_year": {
            "source": "Routine-immunisation implementation assumption",
            "note": "Rate at which routine vaccination relaxes toward age-specific target dose-history origins.",
        },
        "routine_vaccination.max_daily_flow_fraction": {
            "source": "Routine-immunisation numerical safeguard",
            "note": "Caps daily flow from unvaccinated susceptible states to avoid numerical overshoot in the ODE implementation.",
        },
        "initial_conditions.initial_exposed_per_100k": {
            "source": "Initial-condition implementation assumption",
            "note": "Low-level exposed seeding at model start; long burn-in reduces dependence on this value.",
        },
        "initial_conditions.initial_infectious_per_100k": {
            "source": "Initial-condition implementation assumption",
            "note": "Low-level infectious seeding at model start; long burn-in reduces dependence on this value.",
        },
        "initial_conditions.initial_resistance_prevalence": {
            "source": "Initial-condition implementation assumption",
            "note": "Starting resistant fraction before burn-in; submitted country-timeline analyses subsequently rebalance to the evidence anchor.",
        },
        "PEP.coverage_household_contacts": {
            "source": "CDC guidance plus elicited implementation prior",
            "note": "Guidance identifies priority contacts but does not estimate coverage; the distribution is explicitly an implementation prior.",
        },
        "PEP.effectiveness_sensitive": {
            "source": "Timely PEP effectiveness study",
            "note": "The prior is centred on 82.3% effectiveness for PEP delivered within seven days.",
        },
        "PEP.effectiveness_resistant": {
            "source": "CDC resistance and PEP guidance",
            "note": "Standard macrolide PEP for confirmed high-level resistance is not assigned a pseudo-empirical continuous prior.",
        },
        "PEP.activation_prevalence": {
            "source": "PEP implementation threshold assumption",
            "note": "Low detection/prevalence proxy that activates PEP reach in the reduced-form contact-management pathway.",
        },
        "treatment.sensitive.infectious_duration_reduction": {
            "source": "CDC guidance plus treatment-effect implementation assumption",
            "note": "Standard macrolide treatment is assumed to shorten infectious duration for sensitive infections when started early.",
        },
        "treatment.sensitive.infectiousness_reduction": {
            "source": "CDC guidance plus treatment-effect implementation assumption",
            "note": "Standard macrolide treatment is assumed to reduce infectiousness for sensitive infections when started early.",
        },
        "treatment.resistant.infectious_duration_reduction": {
            "source": "Macrolide-resistance treatment-effect assumption",
            "note": "Standard macrolide management is assumed to shorten resistant infections less than sensitive infections; resistance-guided management varies this.",
        },
        "treatment.resistant.infectiousness_reduction": {
            "source": "Macrolide-resistance treatment-effect assumption",
            "note": "Standard macrolide management is assumed to reduce resistant infectiousness less than sensitive infectiousness; resistance-guided management varies this.",
        },
    }
    baseline_vaccine_name = str(
        baseline.get("calibration", {}).get("baseline_vaccine_scenario", "symptom_protective")
    )
    baseline_vaccine = vaccines.get(baseline_vaccine_name, vaccines.get("symptom_protective", {}))

    def _fmt_parameter_value(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if bool(value) else "No"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return ""
        if abs(numeric) < 0.001 and numeric != 0:
            return f"{numeric:.2e}"
        if abs(numeric) < 0.1:
            return f"{numeric:.4g}"
        return f"{numeric:.4g}"

    def _range_from_distribution(spec: dict[str, Any], *, unit: str = "") -> str:
        low = _fmt_parameter_value(spec.get("low", spec.get("min")))
        high = _fmt_parameter_value(spec.get("high", spec.get("max")))
        suffix = f" {unit}" if unit and unit not in {"ratio", "proportion"} else ""
        if "distribution" not in spec:
            raise ValueError(
                "Manuscript parameter-table uncertainty specifications must "
                "declare an explicit distribution."
            )
        distribution = str(spec["distribution"]).replace("_", "-")
        return f"{low} to {high}{suffix} ({distribution} support)".strip()

    def _parameter_range_text(path: str, unit: str) -> str:
        if path in sensitivity_by_path:
            return _range_from_distribution(sensitivity_by_path[path], unit=unit)
        if path in semantic_sensitivity_by_path:
            spec = semantic_sensitivity_by_path[path]
            if path == "vaccine.VE_sym":
                return (
                    f"Derived from overall vaccine disease efficacy; "
                    f"{_range_from_distribution(spec, unit='proportion')}"
                )
            if path == "natural_history.infectious_duration_asymptomatic":
                return (
                    f"Derived as {_fmt_parameter_value(spec.get('low'))} to "
                    f"{_fmt_parameter_value(spec.get('high'))} times the sampled symptomatic duration "
                    "(beta ratio support)"
                )
            if path in {"natural_history.R_to_W_duration", "natural_history.W_to_S_duration"}:
                low = 0.5 * float(spec["low"])
                high = 0.5 * float(spec["high"])
                return (
                    f"{_fmt_parameter_value(low)} to {_fmt_parameter_value(high)} {unit} per stage "
                    "(half of sampled total natural-immunity duration)"
                )
            if path in {
                "natural_history.vaccine_protection_duration",
                "immunity_model.waned_vaccine_duration",
            }:
                low = 0.5 * float(spec["low"])
                high = 0.5 * float(spec["high"])
                return (
                    f"{_fmt_parameter_value(low)} to {_fmt_parameter_value(high)} {unit} per stage "
                    "(half of sampled total vaccine-origin protection duration)"
                )
            return _range_from_distribution(spec, unit=unit)
        alias = sensitivity_path_aliases.get(path)
        if alias and alias in sensitivity_by_path:
            spec = sensitivity_by_path[alias]
            lower_duration = 1.0 / float(spec["max"])
            upper_duration = 1.0 / float(spec["min"])
            return (
                f"{_fmt_parameter_value(lower_duration)} to {_fmt_parameter_value(upper_duration)} {unit} "
                f"(reciprocal of {alias}={_fmt_parameter_value(spec['min'])} to "
                f"{_fmt_parameter_value(spec['max'])} per day)"
            )
        explicit_ranges = {
            "simulation.end_time": (
                f"{baseline['calendar']['analysis_start_date']} to "
                f"{baseline['calendar']['analysis_end_date']} in the current analysis"
            ),
            "simulation.history_start_date": (
                f"{baseline['simulation']['history_start_date']} fixed calendar origin; "
                f"prospective policy t0={baseline['calendar']['analysis_start_date']}"
            ),
            "simulation.output_time_step": "7 days in production output; calibration saves every 30 days",
            "simulation.rtol": "1e-5 production; 1e-4 calibration",
            "simulation.atol": "1e-7 production; 1e-6 calibration",
            "transmission.beta_S": "Country-calibrated; bounded calibration support 0.002 to 0.20",
            "transmission.seasonal_phase": "Country-specific processed value when inferred; shared default day 30",
            "transmission.multi_year_period_years": "3 to 5 years when country surveillance supports recurrence; otherwise fixed at 4 years",
            "transmission.multi_year_phase": "Fixed at 0 in the submitted primary analysis",
            "natural_history.latent_duration": "7 to 10 days clinical range",
            "natural_history.maternal_protection_duration": "90 days baseline; 180 days in pregnancy Tdap scale-up; 90 to 270 days in the optional legacy nonpublication research intervention prior",
            "natural_history.recovered_immunity_duration": "Legacy fallback only; inactive under the submitted SIRWS boosting model",
            "natural_history.R_to_W_duration": "Fixed at 1825 days in the submitted primary analysis",
            "natural_history.W_to_S_duration": "Fixed at 3650 days in the submitted primary analysis",
            "immunity_model.boosting_efficiency": "0 to 1 admissible; fixed at 0.70 in the submitted primary analysis",
            "immunity_model.waned_natural_infection_susceptibility": "0 to 1 admissible; fixed at 0.35 in the submitted primary analysis",
            "immunity_model.waned_relative_effect": "0 to 1 admissible; fixed at 0.35 in the submitted primary analysis",
            "immunity_model.maternal_relative_effect": "0 to 1 admissible; fixed at 0.75 in the submitted primary analysis",
            "immunity_model.dose1_relative_effect": "0 to 1 admissible; fixed at 0.45 in the submitted primary analysis",
            "immunity_model.dose2_relative_effect": "0 to 1 admissible; fixed at 0.75 in the submitted primary analysis",
            "immunity_model.waned_vaccine_duration": "Fixed at 3650 days in the submitted primary analysis",
            "immunity_model.initial_recent_fraction": "Age-specific initial fractions 0.06 to 1.00; shared fallback 0.35",
            "observation.reporting_probability_baseline": "Country- and age-specific calibrated probabilities; prior bounds reported in the fitted reporting-probability table",
            "reporting_multiplier": "Calibration and sensitivity support 0.50 to 1.50; optional legacy nonpublication research prior SD 0.80 on log scale",
            "calibration.dispersion": "Fixed at 50 in the negative-binomial likelihood; stochastic overlay explores k=5 to 50 separately",
            "calibration.recent_years": "Up to 6 recent observed years when available",
            "calibration.relative_incidence_tolerance": "Fixed at 0.25 for retained mean-incidence calibration",
            "treatment.treatment_rate_asymptomatic": "Fixed low-rate pathway in submitted analyses",
            "treatment.sensitive.infectious_duration_reduction": "Fixed at 0.20 for standard sensitive-strain treatment",
            "treatment.sensitive.infectiousness_reduction": "Fixed at 0.15 for standard sensitive-strain treatment",
            "treatment.resistant.infectious_duration_reduction": "0.10 under standard macrolide practice; 0.45 under resistance-guided management",
            "treatment.resistant.infectiousness_reduction": "0.05 under standard macrolide practice; 0.35 under resistance-guided management",
            "PEP.effectiveness_sensitive": "Fixed at 0.70 for standard macrolide-sensitive PEP",
            "PEP.activation_prevalence": "Fixed at 0.00002 in the submitted primary analysis",
            "resistance.target_prevalence_at_analysis_start": "Country-timeline anchor or fixed scenarios 0.05, 0.30, 0.70, and 0.95",
            "resistance.importation_fraction": "Country-timeline anchor or fixed scenarios 0.05, 0.30, 0.70, and 0.95",
            "resistance.prevalence_anchor_rate_per_year": "Fixed at 2.0 per year for country-timeline anchoring",
            "resistance.rebalance_after_burn_in": "Enabled in country-timeline analyses",
            "importation.rate_per_100k_per_year": "Bounded calibration support 0.01 to 2.00",
            "routine_vaccination.target_relaxation_rate_per_year": "2.0 per year baseline; 6.0 per year in routine-timeliness sensitivity",
            "routine_vaccination.max_daily_flow_fraction": "0.01 per day baseline; 0.03 per day in routine-timeliness sensitivity",
            "initial_conditions.initial_exposed_per_100k": "Fixed at 1.5 per 100,000 before burn-in",
            "initial_conditions.initial_infectious_per_100k": "Fixed at 0.6 per 100,000 before burn-in",
            "initial_conditions.initial_resistance_prevalence": "Fixed at 0.30 before burn-in; country-timeline analyses rebalance after burn-in",
        }
        return explicit_ranges.get(path, "Fixed in the submitted primary analysis")

    parameter_specs = [
        ("simulation.end_time", "Analysis horizon", "Simulation analysis horizon", baseline["simulation"]["end_time"], "days"),
        (
            "simulation.history_start_date",
            "Pre-analysis history origin",
            "Fixed calendar origin for historical state reconstruction",
            baseline["simulation"]["history_start_date"],
            "calendar date",
        ),
        (
            "simulation.output_time_step",
            "Saved output interval",
            "Time interval between saved model outputs",
            baseline["simulation"]["output_time_step"],
            "days",
        ),
        (
            "simulation.rtol",
            "Production solver relative tolerance",
            "Relative tolerance for the RK45 ODE solver",
            baseline["simulation"]["rtol"],
            "unitless",
        ),
        (
            "simulation.atol",
            "Production solver absolute tolerance",
            "Absolute tolerance for the RK45 ODE solver",
            baseline["simulation"]["atol"],
            "unitless",
        ),
        (
            "transmission.beta_S",
            "Sensitive-strain transmission coefficient ($\\beta_{\\mathrm{sens}}$)",
            "Transmission rate for macrolide-sensitive pertussis",
            baseline["transmission"]["beta_S"],
            "per contact day",
        ),
        (
            "transmission.relative_infectiousness_asymptomatic",
            "Relative infectiousness of asymptomatic infection",
            "Relative infectiousness of asymptomatic infection",
            baseline["transmission"]["relative_infectiousness_asymptomatic"],
            "ratio",
        ),
        (
            "transmission.fitness_R",
            "Resistant-strain relative fitness ($f_R$)",
            "Relative transmission fitness of macrolide-resistant infections",
            baseline["transmission"]["fitness_R"],
            "ratio",
        ),
        (
            "transmission.seasonal_amplitude",
            "Seasonal forcing amplitude",
            "Annual cosine forcing amplitude",
            baseline["transmission"]["seasonal_amplitude"],
            "ratio",
        ),
        (
            "transmission.seasonal_phase",
            "Seasonal forcing phase",
            "Day-of-year phase for annual cosine forcing",
            baseline["transmission"]["seasonal_phase"],
            "day of year",
        ),
        (
            "transmission.multi_year_period_years",
            "Inter-epidemic recurrence period",
            "Target/diagnostic inter-epidemic period",
            baseline["transmission"]["multi_year_period_years"],
            "years",
        ),
        (
            "transmission.multi_year_amplitude",
            "Multi-year recurrence amplitude",
            "Weak multi-year phase-locking amplitude",
            baseline["transmission"]["multi_year_amplitude"],
            "ratio",
        ),
        (
            "transmission.multi_year_phase",
            "Multi-year recurrence phase",
            "Phase of optional multi-year recurrence forcing",
            baseline["transmission"]["multi_year_phase"],
            "days",
        ),
        ("natural_history.latent_duration", "Latent period", "Latent period duration", baseline["natural_history"]["latent_duration"], "days"),
        (
            "natural_history.infectious_duration_symptomatic",
            "Symptomatic infectious period",
            "Symptomatic infectious duration",
            baseline["natural_history"]["infectious_duration_symptomatic"],
            "days",
        ),
        (
            "natural_history.infectious_duration_asymptomatic",
            "Asymptomatic infectious period",
            "Asymptomatic infectious duration",
            baseline["natural_history"]["infectious_duration_asymptomatic"],
            "days",
        ),
        (
            "natural_history.maternal_protection_duration",
            "Passive maternal-protection duration",
            "Duration of short-lived maternally derived infant protection",
            baseline["natural_history"]["maternal_protection_duration"],
            "days",
        ),
        (
            "natural_history.recovered_immunity_duration",
            "Post-infection protection duration",
            "Duration of post-infection protection",
            baseline["natural_history"]["recovered_immunity_duration"],
            "days",
        ),
        (
            "natural_history.vaccine_protection_duration",
            "Vaccine-derived protection duration",
            "Duration of vaccine-derived protection proxy",
            baseline["natural_history"]["vaccine_protection_duration"],
            "days",
        ),
        (
            "natural_history.R_to_W_duration",
            "SIRWS recovered-to-waned duration",
            "Duration before fully immune recovered state moves to waned-but-boostable state",
            baseline["natural_history"]["R_to_W_duration"],
            "days",
        ),
        (
            "natural_history.W_to_S_duration",
            "SIRWS waned-to-susceptible duration",
            "Duration before waned-but-boostable state loses residual natural immunity",
            baseline["natural_history"]["W_to_S_duration"],
            "days",
        ),
        (
            "immunity_model.boosting_efficiency",
            "SIRWS boosting efficiency",
            "Fraction of waned-natural exposure events that restore immunity",
            baseline["immunity_model"]["boosting_efficiency"],
            "proportion",
        ),
        (
            "immunity_model.waned_natural_infection_susceptibility",
            "Waned-natural infection susceptibility",
            "Susceptibility scalar for the waned-natural-immunity state",
            baseline["immunity_model"]["waned_natural_infection_susceptibility"],
            "proportion",
        ),
        (
            "immunity_model.waned_relative_effect",
            "Waned vaccine-origin effect weight",
            "Relative vaccine-effect weight assigned to waned vaccine-origin histories",
            baseline["immunity_model"]["waned_relative_effect"],
            "proportion",
        ),
        (
            "immunity_model.maternal_relative_effect",
            "Maternal-origin effect weight",
            "Relative vaccine-effect weight assigned to maternal-protection origin histories",
            baseline["immunity_model"]["maternal_relative_effect"],
            "proportion",
        ),
        (
            "immunity_model.dose1_relative_effect",
            "Dose-1 origin effect weight",
            "Relative vaccine-effect weight assigned after one primary-series dose",
            baseline["immunity_model"]["dose1_relative_effect"],
            "proportion",
        ),
        (
            "immunity_model.dose2_relative_effect",
            "Dose-2 origin effect weight",
            "Relative vaccine-effect weight assigned after two primary-series doses",
            baseline["immunity_model"]["dose2_relative_effect"],
            "proportion",
        ),
        (
            "immunity_model.waned_vaccine_duration",
            "Waned vaccine-origin loss duration",
            "Duration of residual protection for waned vaccine-origin histories",
            baseline["immunity_model"]["waned_vaccine_duration"],
            "days",
        ),
        (
            "immunity_model.initial_recent_fraction",
            "Initial recent-vaccine-origin fraction",
            "Fallback fraction of vaccinated-origin people initialized as recently protected",
            baseline["immunity_model"]["initial_recent_fraction"],
            "proportion",
        ),
        (
            "vaccine.VE_sus",
            "Vaccine susceptibility effect ($VE_{\\mathrm{sus}}$)",
            f"Baseline {baseline_vaccine_name} reduction in susceptibility to infection",
            baseline_vaccine.get("VE_sus", np.nan),
            "proportion",
        ),
        (
            "vaccine.VE_sym",
            "Vaccine symptom effect ($VE_{\\mathrm{sym}}$)",
            f"Baseline {baseline_vaccine_name} reduction in symptomatic disease given infection",
            baseline_vaccine.get("VE_sym", np.nan),
            "proportion",
        ),
        (
            "vaccine.VE_inf",
            "Vaccine infectiousness effect ($VE_{\\mathrm{inf}}$)",
            f"Baseline {baseline_vaccine_name} reduction in onward infectiousness",
            baseline_vaccine.get("VE_inf", np.nan),
            "proportion",
        ),
        (
            "vaccine.VE_dur",
            "Vaccine infectious-duration effect ($VE_{\\mathrm{dur}}$)",
            f"Baseline {baseline_vaccine_name} reduction in infectious duration",
            baseline_vaccine.get("VE_dur", np.nan),
            "proportion",
        ),
        (
            "importation.rate_per_100k_per_year",
            "Imported infection seeding rate",
            "Low-level imported infections per 100,000 persons per year",
            baseline["importation"]["rate_per_100k_per_year"],
            "per 100,000 per year",
        ),
        (
            "resistance.target_prevalence_at_analysis_start",
            "Target resistant fraction at analysis start",
            "Macrolide-resistant fraction after burn-in rebalance",
            baseline["resistance"]["target_prevalence_at_analysis_start"],
            "proportion",
        ),
        (
            "resistance.importation_fraction",
            "Resistant fraction among imports",
            "Fraction of imported infections assigned to the resistant strain",
            baseline["resistance"]["importation_fraction"],
            "proportion",
        ),
        (
            "resistance.prevalence_anchor_rate_per_year",
            "Resistance prevalence anchoring rate",
            "Annual rate used during burn-in rebalance toward evidence-based resistant fraction",
            baseline["resistance"]["prevalence_anchor_rate_per_year"],
            "per year",
        ),
        (
            "resistance.rebalance_after_burn_in",
            "Resistance rebalance after burn-in",
            "Whether burn-in output is rebalanced to the evidence-based resistant fraction",
            baseline["resistance"]["rebalance_after_burn_in"],
            "boolean",
        ),
        (
            "observation.reporting_probability_baseline",
            "Age-specific reporting probabilities",
            "Observation-layer probabilities converting symptomatic cases to reported cases",
            "Country- and age-specific",
            "proportion",
        ),
        (
            "reporting_multiplier",
            "Reporting multiplier",
            "Country-calibrated multiplier applied to age-specific reporting probabilities",
            1.0,
            "ratio",
        ),
        (
            "calibration.dispersion",
            "Negative-binomial calibration dispersion",
            "Dispersion parameter for reported-case likelihood",
            baseline["calibration"]["dispersion"],
            "count-scale dispersion",
        ),
        (
            "calibration.recent_years",
            "Calibration observation window",
            "Number of recent surveillance years used for calibration when available",
            baseline["calibration"]["recent_years"],
            "years",
        ),
        (
            "calibration.relative_incidence_tolerance",
            "Calibration mean-incidence tolerance",
            "Retained-fit tolerance for model-to-observed mean reported incidence",
            baseline["calibration"]["relative_incidence_tolerance"],
            "relative difference",
        ),
        (
            "routine_vaccination.target_relaxation_rate_per_year",
            "Routine vaccination target-relaxation rate",
            "Annual relaxation rate toward age-specific vaccine-origin targets",
            baseline["routine_vaccination"]["target_relaxation_rate_per_year"],
            "per year",
        ),
        (
            "routine_vaccination.max_daily_flow_fraction",
            "Routine vaccination maximum daily flow",
            "Maximum daily fraction moved from unvaccinated susceptible state into vaccine-origin states",
            baseline["routine_vaccination"]["max_daily_flow_fraction"],
            "per day",
        ),
        (
            "initial_conditions.initial_exposed_per_100k",
            "Initial exposed seeding",
            "Initial exposed infections per 100,000 persons before burn-in",
            baseline["initial_conditions"]["initial_exposed_per_100k"],
            "per 100,000",
        ),
        (
            "initial_conditions.initial_infectious_per_100k",
            "Initial infectious seeding",
            "Initial infectious infections per 100,000 persons before burn-in",
            baseline["initial_conditions"]["initial_infectious_per_100k"],
            "per 100,000",
        ),
        (
            "initial_conditions.initial_resistance_prevalence",
            "Initial resistant fraction before burn-in",
            "Initial resistant fraction used before burn-in rebalance",
            baseline["initial_conditions"]["initial_resistance_prevalence"],
            "proportion",
        ),
        (
            "treatment.treatment_rate_symptomatic",
            "Treatment initiation rate for symptomatic infection",
            "Daily transition from symptomatic infection to treatment",
            baseline["treatment"]["treatment_rate_symptomatic"],
            "per day",
        ),
        (
            "treatment.treatment_rate_asymptomatic",
            "Treatment initiation rate for asymptomatic infection",
            "Daily transition from asymptomatic infection to treatment",
            baseline["treatment"]["treatment_rate_asymptomatic"],
            "per day",
        ),
        (
            "treatment.sensitive.infectious_duration_reduction",
            "Sensitive-strain treatment duration reduction",
            "Reduction in infectious duration under standard treatment for sensitive infections",
            baseline["treatment"]["sensitive"]["infectious_duration_reduction"],
            "proportion",
        ),
        (
            "treatment.sensitive.infectiousness_reduction",
            "Sensitive-strain treatment infectiousness reduction",
            "Reduction in infectiousness under standard treatment for sensitive infections",
            baseline["treatment"]["sensitive"]["infectiousness_reduction"],
            "proportion",
        ),
        (
            "treatment.resistant.infectious_duration_reduction",
            "Resistant-strain treatment duration reduction",
            "Reduction in infectious duration under standard macrolide treatment for resistant infections",
            baseline["treatment"]["resistant"]["infectious_duration_reduction"],
            "proportion",
        ),
        (
            "treatment.resistant.infectiousness_reduction",
            "Resistant-strain treatment infectiousness reduction",
            "Reduction in infectiousness under standard macrolide treatment for resistant infections",
            baseline["treatment"]["resistant"]["infectiousness_reduction"],
            "proportion",
        ),
        (
            "PEP.coverage_household_contacts",
            "PEP reach among close contacts",
            "Dynamic PEP coverage ceiling among close contacts",
            baseline["PEP"]["coverage_household_contacts"],
            "proportion",
        ),
        (
            "PEP.effectiveness_sensitive",
            "PEP effectiveness for macrolide-sensitive infection",
            "Reduction in susceptible exposure under standard PEP for sensitive infection",
            baseline["PEP"]["effectiveness_sensitive"],
            "proportion",
        ),
        (
            "PEP.effectiveness_resistant",
            "PEP effectiveness for macrolide-resistant infection",
            "Reduction in susceptible exposure under standard macrolide PEP for resistant infection",
            baseline["PEP"]["effectiveness_resistant"],
            "proportion",
        ),
        (
            "PEP.activation_prevalence",
            "PEP activation prevalence proxy",
            "Detection/prevalence proxy threshold for activating PEP reach",
            baseline["PEP"]["activation_prevalence"],
            "prevalence",
        ),
    ]
    parameter_rows = []
    for path, display_name, description, value, unit in parameter_specs:
        source_key = parameter_source_aliases.get(path, path)
        source_note = parameter_source_overrides.get(
            path,
            parameter_sources.get(source_key, parameter_sources.get(source_key.split(".")[0], {})),
        )
        sensitivity_path = sensitivity_path_aliases.get(path, path)
        used_in_sensitivity = (
            path in sensitivity_paths
            or sensitivity_path in sensitivity_paths
            or path in semantic_sensitivity_by_path
        )
        parameter_rows.append(
            {
                "parameter": display_name,
                "description": description,
                "baseline_value": value,
                "range": _parameter_range_text(path, unit),
                "unit": unit,
                "source_or_assumption": source_note.get("source", ""),
                "source_note": source_note.get("note", ""),
                "used_in_sensitivity_analysis": used_in_sensitivity,
            }
        )
    write_dataframe(pd.DataFrame(parameter_rows), project_path("manuscript_notes/parameter_table.csv"))

    vaccine_rows = []
    for name, values in vaccines.items():
        row = {"scenario": name}
        row.update({k: values[k] for k in ["VE_sus", "VE_sym", "VE_inf", "VE_dur"]})
        row["description"] = values.get("description", "")
        vaccine_rows.append(row)
    write_dataframe(pd.DataFrame(vaccine_rows), project_path("manuscript_notes/scenario_table.csv"))

    resistance_rows = []
    for name, values in resistance.items():
        resistance_rows.append(
            {
                "scenario": name,
                "target_prevalence_at_analysis_start": values.get(
                    "target_prevalence_at_analysis_start",
                    values.get("initial_resistance_prevalence"),
                ),
                "initial_resistance_prevalence_deprecated": values.get("initial_resistance_prevalence", np.nan),
                "importation_fraction": values.get("importation_fraction", np.nan),
                "prevalence_anchor_rate_per_year": values.get("prevalence_anchor_rate_per_year", np.nan),
                "anchor_during_dynamics": bool(values.get("anchor_during_dynamics", False)),
                "uses_country_resistance_timeline": bool(values.get("use_country_resistance_timeline", False)),
                "fitness_R": values.get("fitness_R", 1.0),
                "treatment_effect_resistant": baseline["treatment"]["resistant"]["infectious_duration_reduction"],
                "PEP_effectiveness_resistant": baseline["PEP"]["effectiveness_resistant"],
                "description": values.get("description", ""),
            }
        )
    write_dataframe(pd.DataFrame(resistance_rows), project_path("manuscript_notes/resistance_scenario_table.csv"))

    intervention_metadata = {
        "current": {
            "scenario_category": "Comparator",
            "interpretive_status": "Current practice",
            "modified_control_levers": "Country-specific schedule, coverage, treatment, and PEP assumptions retained.",
            "interpretation_note": "Baseline comparator; implementation intensity is zero by construction.",
        },
        "higher_child_coverage": {
            "scenario_category": "Program-only option",
            "interpretive_status": "Implementable routine-program lever",
            "modified_control_levers": "Nominal infant and childhood coverage floors raised; dose timing, vaccine mechanism, treatment, and PEP assumptions unchanged.",
            "interpretation_note": "Tests marginal nominal coverage gains, not the value of routine vaccination itself.",
        },
        "timeliness_only": {
            "scenario_category": "Program-only option",
            "interpretive_status": "Implementable routine-program lever",
            "modified_control_levers": "Routine-delivery rate and age-bin vaccine-origin targets shifted toward the dose histories implied by each profile's recommended schedule (for example W6/W10/W14, M2/M4/M6, or M3/M5/M12); nominal coverage and vaccine mechanism unchanged.",
            "interpretation_note": "Tests reducing delays relative to the local schedule rather than imposing a common absolute start age.",
        },
        "adolescent_booster": {
            "scenario_category": "Program-only option",
            "interpretive_status": "Program-dependent lever",
            "modified_control_levers": "Older-child or adolescent booster coverage raised where schedule structure permits; infant DTaP and pregnancy Tdap unchanged.",
            "interpretation_note": "Indirect infant benefit depends on older-age transmission and was selected only in the shortest-overlap calibration profile.",
        },
        "pregnancy_tdap_scaleup": {
            "scenario_category": "Program-only option",
            "interpretive_status": "Implementable maternal-program lever",
            "modified_control_levers": "Newborn passive protection increased; routine infant DTaP coverage, adult contact structure, and treatment assumptions unchanged.",
            "interpretation_note": "Direct early-infant protection component; not an adult-source reduction package.",
        },
        "cocooning_adjunct": {
            "scenario_category": "Program-only option",
            "interpretive_status": "Implementation-dependent adjunct",
            "modified_control_levers": "Close-contact adult protection and adult-to-infant exposure reduction strengthened at a smaller magnitude than the full infant-exposure package.",
            "interpretation_note": "Should be read as adjunctive household/contact management, not a replacement for pregnancy Tdap.",
        },
        "maternal_immunization": {
            "scenario_category": "Program-only composite",
            "interpretive_status": "Mechanistic package",
            "modified_control_levers": "Pregnancy Tdap, close-contact adult protection, reproductive-age adult boosting, and reduced adult-to-infant contact combined.",
            "interpretation_note": "Primary infant-exposure package; not a maternal-vaccination-only effect estimate and decomposed in supplementary analyses.",
        },
        "targeted_pep_high_risk": {
            "scenario_category": "Program-only option",
            "interpretive_status": "Implementation-dependent contact-management lever",
            "modified_control_levers": "PEP reach increased among guideline-prioritized contacts and high-risk infant settings; routine vaccination unchanged.",
            "interpretation_note": "Represents outbreak/contact-management support, not broad community PEP.",
        },
        "maternal_direct_antibody_only": {
            "scenario_category": "Component diagnostic",
            "interpretive_status": "Not a standalone optimization policy",
            "modified_control_levers": "Newborn passive protection increased; adult boosting and contact-reduction components disabled.",
            "interpretation_note": "Isolates the direct maternal-antibody pathway within the infant-exposure package.",
        },
        "maternal_adult_boosting_only": {
            "scenario_category": "Component diagnostic",
            "interpretive_status": "Mechanistic decomposition",
            "modified_control_levers": "Reproductive-age adult boosting enabled; direct newborn antibody and contact-reduction components disabled.",
            "interpretation_note": "Isolates reduced adult-source infection risk from direct infant antibody protection.",
        },
        "maternal_cocooning_only": {
            "scenario_category": "Component diagnostic",
            "interpretive_status": "Mechanistic decomposition",
            "modified_control_levers": "Adult-to-infant contact reduction enabled; direct newborn antibody and adult boosting components disabled.",
            "interpretation_note": "Isolates reduced infant exposure from household/contact assumptions.",
        },
        "resistance_guided_treatment": {
            "scenario_category": "Resistance-management option",
            "interpretive_status": "Reduced-form proxy",
            "modified_control_levers": "Clinic-laboratory-public-health pathway: earlier MRBP suspicion and testing, shorter treated resistant infectious duration, lower resistant infectiousness, and improved resistant-strain PEP effectiveness assumed.",
            "interpretation_note": "Represents resistance-aware clinical recognition, laboratory confirmation, eligible alternative treatment, high-risk-contact PEP, adherence, and surveillance feedback rather than a drug-specific protocol.",
        },
        "transmission_blocking_vaccine": {
            "scenario_category": "Future product target",
            "interpretive_status": "Hypothetical unavailable product",
            "modified_control_levers": "Onward infectiousness reduction strengthened under future vaccine assumptions; program delivery otherwise follows country profile.",
            "interpretation_note": "Product-target scenario for mechanism comparison, not an available policy.",
        },
        "next_generation_vaccine": {
            "scenario_category": "Future product target",
            "interpretive_status": "Hypothetical unavailable product",
            "modified_control_levers": "Susceptibility, symptom, onward infectiousness, and duration effects strengthened in a high-blocking target profile.",
            "interpretation_note": "Upper-bound vaccine target used to compare product mechanisms.",
        },
        "combined_strategy": {
            "scenario_category": "Future stress-test package",
            "interpretive_status": "Non-implementable stress test",
            "modified_control_levers": "Future vaccine target, infant-exposure package, adolescent booster, targeted PEP, and resistance-guided management combined.",
            "interpretation_note": "Tests maximum layering logic and should not be interpreted as a single implementable policy package.",
        },
    }
    intervention_rows = []
    empty_metadata = {
        "scenario_category": "",
        "interpretive_status": "",
        "modified_control_levers": "",
        "interpretation_note": "",
    }
    intervention_descriptions = {
        name: " ".join(str(values.get("description", "")).split())
        for name, values in interventions.items()
    }
    intervention_descriptions.setdefault(
        "timeliness_only",
        "Schedule-relative routine timeliness improvement: delivery shifted toward age-appropriate dose histories implied by each country profile's routine schedule, with nominal final coverage unchanged.",
    )
    intervention_descriptions["resistance_guided_treatment"] = (
        "Clinic-laboratory-public-health pathway for suspected or confirmed "
        "macrolide-resistant pertussis: earlier resistant-strain recognition, "
        "improved treatment effect for eligible resistant infections, and more "
        "effective resistant-strain PEP. This is a reduced-form scenario, not a "
        "drug-specific clinical protocol."
    )
    preferred_intervention_order = (
        "current",
        "higher_child_coverage",
        "timeliness_only",
        "adolescent_booster",
        "pregnancy_tdap_scaleup",
        "cocooning_adjunct",
        "maternal_immunization",
        "targeted_pep_high_risk",
        "maternal_direct_antibody_only",
        "maternal_adult_boosting_only",
        "maternal_cocooning_only",
        "resistance_guided_treatment",
        "transmission_blocking_vaccine",
        "next_generation_vaccine",
        "combined_strategy",
    )
    ordered_intervention_names = [
        name for name in preferred_intervention_order if name in intervention_descriptions
    ]
    ordered_intervention_names.extend(
        name for name in intervention_descriptions if name not in ordered_intervention_names
    )
    for name in ordered_intervention_names:
        description = intervention_descriptions[name]
        row = {"strategy": name, "description": description}
        row.update(intervention_metadata.get(name, empty_metadata))
        intervention_rows.append(row)
    write_dataframe(pd.DataFrame(intervention_rows), project_path("manuscript_notes/intervention_scenario_table.csv"))

    reporting_rows = []
    for name, values in baseline["reporting_rate_sensitivity"].items():
        reporting_rows.append(
            {
                "scenario": name,
                "multiplier": values.get("multiplier", np.nan),
                "uses_age_multipliers": bool(values.get("age_multipliers")),
                "uses_time_variation": bool(values.get("time_variation")),
                "description": "Reporting-rate sensitivity assumption.",
            }
        )
    write_dataframe(pd.DataFrame(reporting_rows), project_path("manuscript_notes/reporting_scenario_table.csv"))

    fitness_grid = baseline.get("fitness_grid", {})
    fitness_rows = [
        {
            "fitness_R": float(fitness),
            "VE_inf": float(ve_inf),
            "description": fitness_grid.get("description", ""),
        }
        for fitness in fitness_grid.get("fitness_R_values", [])
        for ve_inf in fitness_grid.get("VE_inf_values", [])
    ]
    if fitness_rows:
        write_dataframe(pd.DataFrame(fitness_rows), project_path("manuscript_notes/fitness_grid_table.csv"))

    bayesian = baseline.get("bayesian_uncertainty", {})
    bayesian_interpretation_overrides = {
        "VE_sus": (
            "Literature-informed decomposition parameter for reduced susceptibility "
            "after aP vaccination. Aggregate studies support high disease protection "
            "and waning protection, but do not directly identify this component from "
            "surveillance data; [0.30, 0.60] is retained as a modeling range for "
            "sensitivity and pilot uncertainty runs."
        ),
        "VE_dur": (
            "Mechanistic duration-shortening proxy for reduced infectious duration "
            "among vaccinated infections. It is propagated as an external-prior "
            "sensitivity dimension in the optional legacy nonpublication research analysis and remains a vaccine-mechanism "
            "sensitivity parameter in other diagnostics."
        ),
        "resistance_prevalence": (
            "Resistance prevalence is fixed at the country-timeline value during the "
            "optional legacy nonpublication research analysis and beta-grid diagnostics. The country_resistance_timeline.csv "
            "provides well-constrained estimates for most countries; fixed, low-to-very-high "
            "resistance scenarios and the fitness grid evaluate the corresponding "
            "structural uncertainty."
        ),
    }
    prior_rows = []
    if bayesian:
        prior_range_overrides = {
            "VE_sus": "Normalized beta support 0.01 to 0.60",
            "VE_inf": "Normalized beta support 0.05 to 0.60",
            "VE_dur": "Normalized beta support 0.001 to 0.50",
            "relative_infectiousness_asymptomatic": "Normalized beta support 0.05 to 0.95",
            "infectious_duration_symptomatic": "Truncated lognormal support 14 to 35 days",
            "infectious_duration_asymptomatic": "Truncated lognormal support 7 to 28 days",
            "fitness_R": "0.70 to 1.25",
            "resistance_prevalence": "Country-timeline anchor; fixed scenarios 0.05, 0.30, 0.70, and 0.95",
            "maternal_VE_sus": "0.30 to 0.80 interpretation range",
            "maternal_VE_sym": "0.00 to 1.00 beta support; high-confidence mass near 0.92",
        }

        def _prior_row(
            parameter: str,
            group: str,
            value_or_center: Any,
            uncertainty_range: str,
            unit: str,
            distribution: str,
            figure2c_role: str,
            interpretation: str,
        ) -> dict[str, Any]:
            return {
                "parameter_group": group,
                "parameter": parameter,
                "value_or_center": value_or_center,
                "uncertainty_range": uncertainty_range,
                "unit": unit,
                "distribution": distribution,
                "figure2c_role": figure2c_role,
                "interpretation": interpretation,
            }

        process_model = baseline.get("calibration", {}).get("process_model", {})
        beta_state_prior_sd = float(process_model.get("log_beta_prior_sd", 1.0))
        reporting_state_prior_sd = float(
            process_model.get("log_reporting_prior_sd", 0.5)
        )
        beta_state_bounds = process_model.get("beta_S_bounds", [0.0005, 0.5])
        reporting_state_bounds = process_model.get(
            "reporting_multiplier_bounds", [0.5, 1.5]
        )
        prior_rows.extend(
            [
                _prior_row(
                    "log_beta_S",
                    "Optional legacy nonpublication research conditional state target",
                    "Pre-surveillance country configuration",
                    f"Bounded beta_S support {beta_state_bounds[0]} to {beta_state_bounds[1]}",
                    "log scale",
                    f"Normal(log pre-surveillance sensitive-strain transmission coefficient, {beta_state_prior_sd})",
                    "Updated once by annual surveillance and propagated by exact-target importance resampling",
                    "Conditional transmission-rate uncertainty; fixed AR(1) and measurement hyperparameters.",
                ),
                _prior_row(
                    "log_reporting_multiplier",
                    "Optional legacy nonpublication research conditional state target",
                    "Pre-surveillance multiplier 1.0",
                    f"Bounded multiplier support {reporting_state_bounds[0]} to {reporting_state_bounds[1]}",
                    "log scale",
                    f"Normal(log pre-surveillance reporting multiplier, {reporting_state_prior_sd})",
                    "Updated once by annual surveillance and propagated by exact-target importance resampling",
                    "Conditional surveillance/reporting uncertainty.",
                ),
            ]
        )
        for parameter, values in bayesian.get("priors", {}).items():
            if isinstance(values, dict):
                if {"mean", "sd", "min", "max"}.issubset(values):
                    prior = f"Truncated normal(mean={values['mean']}, sd={values['sd']}, range={values['min']}-{values['max']})"
                elif {"mean", "sd"}.issubset(values):
                    prior = f"Beta(mean={values['mean']}, sd={values['sd']})"
                elif "log_sd" in values:
                    prior = f"Log-normal around baseline, log_sd={values['log_sd']}"
                elif "floor_sd" in values:
                    prior = f"Fixed country timeline, floor_sd={values['floor_sd']}"
                else:
                    prior = str(values)
                center = values.get("mean", baseline["natural_history"].get(parameter, "Country-specific"))
                if parameter == "fitness_R":
                    center = values.get("mean", 1.0)
                elif parameter == "resistance_prevalence":
                    center = "Country-timeline anchor"
                elif parameter == "infectious_duration_symptomatic":
                    center = baseline["natural_history"]["infectious_duration_symptomatic"]
                elif parameter == "infectious_duration_asymptomatic":
                    center = baseline["natural_history"]["infectious_duration_asymptomatic"]
                role = "Propagated as an equal-mass external-prior dimension in optional legacy nonpublication research"
                group = "Optional legacy nonpublication research external structural prior"
                if parameter == "resistance_prevalence":
                    role = "Fixed in the optional legacy nonpublication research interval; structural uncertainty evaluated by resistance scenarios"
                    group = "Fixed structural input"
                elif parameter.startswith("maternal_VE_"):
                    role = "Sampled as draw-level intervention-effect uncertainty in the optional legacy nonpublication research interval"
                    group = "Optional legacy nonpublication research intervention effect"
                prior_rows.append(
                    _prior_row(
                        parameter,
                        group,
                        center,
                        prior_range_overrides.get(parameter, "0.00 to 1.00 beta support"),
                        "days" if parameter.startswith("infectious_duration") else "proportion" if "VE" in parameter or parameter == "relative_infectiousness_asymptomatic" else "ratio" if parameter == "fitness_R" else "proportion",
                        prior,
                        role,
                        bayesian_interpretation_overrides.get(parameter, values.get("note", "")),
                    )
                )
        for parameter, values in (
            bayesian.get("figure2c_joint_credible_interval", {}).get("intervention_priors", {}).items()
        ):
            if isinstance(values, dict) and {"low", "mode", "high"}.issubset(values):
                unit = "days" if parameter.endswith("_days") else "proportion"
                prior_rows.append(
                    _prior_row(
                        f"figure2c_{parameter}",
                        "Optional legacy nonpublication research intervention implementation" if "coverage" in parameter else "Optional legacy nonpublication research intervention effect",
                        values["mode"],
                        f"{values['low']} to {values['high']}",
                        unit,
                        f"Triangular(low={values['low']}, mode={values['mode']}, high={values['high']})",
                        "Sampled at draw level and paired across current-practice and intervention simulations only in optional legacy nonpublication research",
                        values.get("note", ""),
                    )
                )
        write_dataframe(pd.DataFrame(prior_rows), project_path("manuscript_notes/bayesian_prior_table.csv"))

    country_rows = [
        {
            "country": name,
            "description": values.get("description", ""),
            "total_population": sum(float(v) for v in values.get("population", {}).values()),
            "seasonal_phase": values.get("transmission_overrides", {}).get("seasonal_phase", np.nan),
            "seasonal_amplitude": values.get("transmission_overrides", {}).get("seasonal_amplitude", np.nan),
            "multi_year_period_years": values.get("transmission_overrides", {}).get("multi_year_period_years", np.nan),
            "multi_year_amplitude": values.get("transmission_overrides", {}).get("multi_year_amplitude", np.nan),
            "observed_mean_annual_reported_incidence_per_100k": values.get("observed_incidence", {}).get("observed_mean_annual_reported_incidence_per_100k", np.nan),
            "observed_peak_annual_reported_incidence_per_100k": values.get("observed_incidence", {}).get("observed_peak_annual_reported_incidence_per_100k", np.nan),
            "vaccine_product": values.get("vaccine_schedule", {}).get("vaccine_product", ""),
            "adolescent_booster": values.get("vaccine_schedule", {}).get("adolescent_booster", np.nan),
            "maternal_program": values.get("vaccine_schedule", {}).get("maternal_program", np.nan),
            "contact_source": values.get("contact_source", ""),
            "source_type_note": "; ".join(
                f"{key}={value}" for key, value in values.get("source_types", {}).items()
            ),
            "source_or_assumption": "WPP population, PertussisIncidence seasonality/cycles, WUENIC/JRF schedule metadata, Prem/contactdata contacts",
        }
        for name, values in countries.items()
    ]
    write_dataframe(pd.DataFrame(country_rows), project_path("manuscript_notes/country_profile_table.csv"))

    intervention_summary_path = project_path("outputs/summaries/intervention_scenarios_summary.csv")
    if intervention_summary_path.exists():
        try:
            validate_run_metadata("intervention_scenarios")
        except Exception:
            return
        else:
            intervention_summary = read_table(intervention_summary_path)
            cols = [
                "country",
                "scenario",
                "total_infections",
                "total_reported_cases",
                "total_infant_cases",
                "resistant_infections",
                "relative_reduction_infant_cases",
                "relative_reduction_total_infections",
            ]
            table = intervention_summary.loc[:, cols].rename(
                columns={
                    "scenario": "strategy",
                    "total_reported_cases": "reported_cases",
                }
            )
            write_dataframe(table, project_path("outputs/tables/table_4_intervention_comparison.csv"))
