from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src_python.model.compartments import StateIndex
from src_python.model.outputs import compute_timeseries, infer_output_dt, solve_model, summarize_timeseries
from src_python.model.parameters import PreparedParameters
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

RUNTIME_DATA_HASH_COLUMNS = {
    "country_resistance_timeline_csv": (
        "country",
        "iso3",
        "year",
        "resistant_fraction",
        "lower",
        "upper",
        "evidence_type",
    ),
    "diagnostic_standard_timeline_csv": (
        "country",
        "iso3",
        "period_start",
        "period_end",
        "relative_detection_prior_mean",
        "relative_detection_prior_lower",
        "relative_detection_prior_upper",
        "effect_direction",
        "evidence_strength",
    ),
    "npi_contact_reduction_timeline_csv": (
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
    ),
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
    return {
        "settings": settings,
        "baseline": baseline,
        "vaccines": runtime["vaccine_scenarios"],
        "resistance": runtime["resistance_scenarios"],
        "interventions": runtime["intervention_scenarios"],
        "sensitivity": runtime["sensitivity_parameters"],
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


def _resolve_calendar_horizon(baseline: dict[str, Any]) -> None:
    """Derive simulation.end_time from calendar dates when available.

    The yaml keeps `calendar.analysis_start_date` / `calendar.analysis_end_date`
    as the single source of truth for the production analysis window. Tests and
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
        derived_end_time = float((end - start).days)
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


def _config_fingerprint_from_configs(configs: dict[str, Any]) -> str:
    payload = {
        "settings_runtime": configs["settings"].get("runtime", {}),
        "countries": configs["countries"],
        "data_file_hashes": _runtime_data_file_hashes(configs["settings"].get("runtime", {}).get("data_sources", {})),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_data_file_hashes(data_sources: dict[str, Any]) -> dict[str, str]:
    """Hash runtime CSV fields that materially change model dynamics."""
    keys = (
        "country_resistance_timeline_csv",
        "diagnostic_standard_timeline_csv",
        "npi_contact_reduction_timeline_csv",
    )
    hashes: dict[str, str] = {}
    for key in keys:
        relative_path = data_sources.get(key)
        if not relative_path:
            continue
        path = project_path(relative_path)
        if not path.exists() or not path.is_file():
            hashes[key] = "missing"
            continue
        digest = hashlib.sha256()
        digest.update(_canonical_runtime_data_bytes(key, path))
        hashes[key] = digest.hexdigest()
    return hashes


def _canonical_runtime_data_bytes(key: str, path: Path) -> bytes:
    columns = RUNTIME_DATA_HASH_COLUMNS.get(key)
    if columns is None:
        with path.open("rb") as handle:
            return handle.read()

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return json.dumps({"missing_columns": missing}, sort_keys=True).encode("utf-8")

    material = frame.loc[:, list(columns)].copy()
    for column in material.columns:
        material[column] = material[column].astype(str).str.strip()
    material = material.sort_values(list(material.columns), kind="mergesort").reset_index(drop=True)
    return material.to_csv(index=False, lineterminator="\n").encode("utf-8")


@lru_cache(maxsize=1)
def _config_fingerprint_cached() -> str:
    return _config_fingerprint_from_configs(_load_configs_cached())


def config_fingerprint(configs: dict[str, Any] | None = None) -> str:
    if configs is None:
        return _config_fingerprint_cached()
    return _config_fingerprint_from_configs(configs)


def _calibration_config_fingerprint_from_configs(configs: dict[str, Any]) -> str:
    runtime = configs["settings"].get("runtime", {})
    payload = {
        "baseline_parameters": runtime.get("baseline_parameters", {}),
        "vaccine_scenarios": runtime.get("vaccine_scenarios", {}),
        "resistance_scenarios": runtime.get("resistance_scenarios", {}),
        "data_sources": runtime.get("data_sources", {}),
        "countries": configs["countries"],
        "data_file_hashes": _runtime_data_file_hashes(runtime.get("data_sources", {})),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _calibration_config_fingerprint_cached() -> str:
    return _calibration_config_fingerprint_from_configs(_load_configs_cached())


def calibration_config_fingerprint(configs: dict[str, Any] | None = None) -> str:
    """Hash only the runtime inputs that affect country calibration.

    Bayesian settings, sensitivity grids, interventions, and plotting-only
    configuration should not invalidate accepted calibration artifacts.  When
    they do, MCMC silently falls back to uncalibrated defaults, which is both
    slow and statistically brittle.
    """
    if configs is None:
        return _calibration_config_fingerprint_cached()
    return _calibration_config_fingerprint_from_configs(configs)


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
    configs = load_configs()
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "stem": stem,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_fingerprint(configs),
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
    metadata = read_run_metadata(stem)
    if int(metadata.get("schema_version", -1)) != METADATA_SCHEMA_VERSION:
        raise ValueError(f"Run metadata schema mismatch for {stem}.")
    expected_hash = config_fingerprint()
    if metadata.get("config_hash") != expected_hash:
        raise ValueError(
            f"Output {stem} was generated from a stale configuration "
            f"({metadata.get('config_hash')}); current config hash is {expected_hash}."
        )
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
    allow_stale: bool = False,
    current_hash: str = "",
    current_calibration_hash: str = "",
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

    source_hash = str(metadata.get("config_hash", ""))
    source_calibration_hash = str(metadata.get("calibration_config_hash", ""))
    hash_is_current = bool(source_hash and source_hash == current_hash)
    calibration_hash_is_current = bool(
        source_calibration_hash and source_calibration_hash == current_calibration_hash
    )
    if not hash_is_current and not calibration_hash_is_current and not allow_stale:
        return None

    status = "current" if hash_is_current or calibration_hash_is_current else "stale_parameter_overlay"
    artifact = deepcopy(artifact)
    artifact.setdefault("metadata", {})["calibration_hash_status"] = status
    return artifact


def load_calibrated_country_artifact(
    country: str,
    *,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
    artifact = _load_calibrated_country_artifact_cached(
        country,
        bool(allow_stale),
        config_fingerprint(),
        calibration_config_fingerprint(),
    )
    return deepcopy(artifact) if artifact is not None else None


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
) -> dict[str, Any]:
    out = deepcopy(config)
    anchor_year = int(data_sources.get("resistance_anchor_year", data_sources.get("analysis_year", 2023)))
    allow_future = bool(data_sources.get("allow_future_resistance_evidence", False))
    iso3 = country_profile.get("iso3")
    timeline = _load_country_resistance_timeline(data_sources)
    rows = _timeline_rows_for_country(timeline, country, iso3)
    estimate = _country_resistance_estimate(rows, anchor_year, allow_future=allow_future)
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
    return out


def _apply_npi_contact_reduction_timeline(
    config: dict[str, Any],
    *,
    country: str,
    country_profile: dict[str, Any],
    data_sources: dict[str, Any],
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
) -> dict[str, Any]:
    configs = load_configs()
    base = deepcopy(configs["baseline"])
    country_name = country_profile or base.get("baseline_country_profile")

    out = deepcopy(base)
    if country_name and country_name in configs["countries"]:
        out = _apply_country_profile_from_profile(out, country_name, configs["countries"][country_name])

    calibration_settings = base.get("calibration", {})
    allow_stale_calibration = bool(calibration_settings.get("allow_stale_parameter_overlay", False))
    calibrated_artifact = (
        load_calibrated_country_artifact(
            country_name,
            allow_stale=allow_stale_calibration,
        )
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
            hash_status = str(artifact_metadata.get("calibration_hash_status", "current"))
            if hash_status == "stale_parameter_overlay":
                calibration_overlay_only = True
                calibration_parameter_overlay = _calibration_parameter_overlay(calibrated_config)
                out = deep_update(out, calibration_parameter_overlay)
            else:
                out = deep_update(out, calibrated_config)
            # Calibration artifacts are produced with a shortened, calendar-aligned
            # runtime. Reuse fitted country parameters, but keep production scenario
            # runs on the configured analysis horizon.
            out["simulation"] = production_simulation
            out["calendar"] = production_calendar
            metadata = out.setdefault("metadata", {})
            metadata["calibration_loaded"] = True
            metadata["calibration_country"] = country_name
            metadata["calibration_artifact_path"] = str(calibrated_country_artifact_path(country_name))
            metadata["calibration_hash_status"] = hash_status
            for key in (
                "config_hash",
                "calibration_config_hash",
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
        )

    if sum(float(value) for key, value in out.get("vaccine", {}).items() if key.startswith("VE_")) == 0.0:
        for record in out["age_groups"]:
            record["vaccine_coverage"] = 0.0
        out.setdefault("demography", {})["birth_entry"] = {"S": 1.0, "V": 0.0}
    return out


def apply_intervention_definition(config: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
    """Apply one intervention definition to an already prepared config.

    This is used for both single predefined profiles and supplementary
    portfolio analyses where several program levers are layered in one run.
    """
    config = deepcopy(config)
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

    return config


def make_intervention_config(
    name: str,
    *,
    country_profile: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    configs = load_configs()
    intervention = configs["interventions"][name]
    vaccine_name = intervention.get("vaccine_scenario", configs["baseline"].get("baseline_vaccine_scenario"))
    config = make_config(
        vaccine_scenario=vaccine_name,
        resistance_scenario=configs["baseline"].get("baseline_resistance_scenario"),
        country_profile=country_profile,
        config_overrides=config_overrides,
    )
    config = apply_intervention_definition(config, intervention)
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


def run_prepared_config(
    config: dict[str, Any],
    *,
    analysis: str,
    scenario: str,
    vaccine_scenario: str = "",
    resistance_scenario: str = "",
    intervention: str = "",
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = PreparedParameters.from_config(
        config,
        analysis=analysis,
        scenario=scenario,
        vaccine_scenario=vaccine_scenario,
        resistance_scenario=resistance_scenario,
        intervention=intervention,
        metadata=metadata,
    )
    index = StateIndex(params.age_groups)
    solution = solve_model(params, index)
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


def _run_scenario_item(item: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_prepared_config(**item)


def _run_scenario_summary_item(item: dict[str, Any]) -> pd.DataFrame:
    _, summary = run_prepared_config(**item)
    return summary


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
) -> None:
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
    if extra_metadata:
        metadata.update(extra_metadata)
    write_run_metadata(stem, metadata)


def execute_scenario_list(
    scenarios: list[dict[str, Any]],
    *,
    stem: str,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not scenarios:
        raise ValueError(f"No scenarios were provided for {stem}.")
    if n_jobs is None:
        n_jobs = scenarios[0]["config"].get("simulation", {}).get("n_jobs")
    results = parallel_map(_run_scenario_item, scenarios, desc=stem, n_jobs=n_jobs)
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
    if n_jobs is None:
        n_jobs = scenarios[0]["config"].get("simulation", {}).get("n_jobs")
    summaries = parallel_map(_run_scenario_summary_item, scenarios, desc=stem, n_jobs=n_jobs)
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
    calibrated country analyses when calibration artifacts are missing or stale.
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
    if "calibration_hash_status" in summary.columns:
        stale = summary.loc[summary["calibration_hash_status"].eq("stale_parameter_overlay")]
        if not stale.empty:
            countries = sorted(stale.get("country", pd.Series(["unknown"])).astype(str).unique())
            raise RuntimeError(
                f"[{stem}] Calibration enforcement failed: {len(stale)} scenario(s) "
                f"for countries {countries} use stale calibration parameter overlays. "
                f"Re-run calibration under the current configuration before production simulations."
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

    sensitivity_paths = {
        spec.get("path")
        for spec in configs["sensitivity"].get("parameters", {}).values()
        if isinstance(spec, dict)
    }
    sensitivity_path_aliases = {
        "natural_history.recovered_immunity_duration": "rates.waning_natural",
        "natural_history.vaccine_protection_duration": "rates.waning_vaccine",
    }
    parameter_specs = [
        ("simulation.end_time", "Simulation analysis horizon", baseline["simulation"]["end_time"], "days"),
        ("simulation.burn_in_years", "Pre-analysis burn-in horizon", baseline["simulation"]["burn_in_years"], "years"),
        ("transmission.beta_S", "Transmission rate for macrolide-sensitive pertussis", baseline["transmission"]["beta_S"], "per contact day"),
        ("transmission.relative_infectiousness_asymptomatic", "Relative infectiousness of asymptomatic infection", baseline["transmission"]["relative_infectiousness_asymptomatic"], "ratio"),
        ("transmission.multi_year_period_years", "Target/diagnostic inter-epidemic period", baseline["transmission"]["multi_year_period_years"], "years"),
        ("transmission.multi_year_amplitude", "Weak multi-year phase-locking amplitude", baseline["transmission"]["multi_year_amplitude"], "ratio"),
        ("natural_history.latent_duration", "Latent period duration", baseline["natural_history"]["latent_duration"], "days"),
        ("natural_history.infectious_duration_symptomatic", "Symptomatic infectious duration", baseline["natural_history"]["infectious_duration_symptomatic"], "days"),
        ("natural_history.infectious_duration_asymptomatic", "Asymptomatic infectious duration", baseline["natural_history"]["infectious_duration_asymptomatic"], "days"),
        ("natural_history.recovered_immunity_duration", "Duration of post-infection protection", baseline["natural_history"]["recovered_immunity_duration"], "days"),
        ("natural_history.vaccine_protection_duration", "Duration of vaccine-derived protection proxy", baseline["natural_history"]["vaccine_protection_duration"], "days"),
        ("treatment.treatment_rate_symptomatic", "Daily transition from symptomatic infection to treatment", baseline["treatment"]["treatment_rate_symptomatic"], "per day"),
        ("treatment.treatment_rate_asymptomatic", "Daily transition from asymptomatic infection to treatment", baseline["treatment"]["treatment_rate_asymptomatic"], "per day"),
        ("PEP.coverage_household_contacts", "Dynamic PEP coverage ceiling among close contacts", baseline["PEP"]["coverage_household_contacts"], "proportion"),
    ]
    parameter_rows = []
    for path, description, value, unit in parameter_specs:
        source_note = parameter_sources.get(path, parameter_sources.get(path.split(".")[0], {}))
        sensitivity_path = sensitivity_path_aliases.get(path, path)
        used_in_sensitivity = path in sensitivity_paths or sensitivity_path in sensitivity_paths
        range_note = "see config/model_settings.yaml sensitivity_parameters"
        if path in sensitivity_path_aliases and sensitivity_path in sensitivity_paths:
            range_note = f"see config/model_settings.yaml sensitivity_parameters (reciprocal of {sensitivity_path})"
        parameter_rows.append(
            {
                "parameter": path,
                "description": description,
                "baseline_value": value,
                "range": range_note,
                "unit": unit,
                "source_or_assumption": source_note.get("source", ""),
                "source_note": source_note.get("note", ""),
                "used_in_sensitivity_analysis": used_in_sensitivity,
            }
        )
    write_dataframe(pd.DataFrame(parameter_rows), project_path("publication_inputs/parameter_table.csv"))

    vaccine_rows = []
    for name, values in vaccines.items():
        row = {"scenario": name}
        row.update({k: values[k] for k in ["VE_sus", "VE_sym", "VE_inf", "VE_dur"]})
        row["description"] = values.get("description", "")
        vaccine_rows.append(row)
    write_dataframe(pd.DataFrame(vaccine_rows), project_path("publication_inputs/scenario_table.csv"))

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
    write_dataframe(pd.DataFrame(resistance_rows), project_path("publication_inputs/resistance_scenario_table.csv"))

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
    write_dataframe(pd.DataFrame(intervention_rows), project_path("publication_inputs/intervention_scenario_table.csv"))

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
    write_dataframe(pd.DataFrame(reporting_rows), project_path("publication_inputs/reporting_scenario_table.csv"))

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
        write_dataframe(pd.DataFrame(fitness_rows), project_path("publication_inputs/fitness_grid_table.csv"))

    bayesian = baseline.get("bayesian_uncertainty", {})
    bayesian_interpretation_overrides = {
        "VE_sus": (
            "Literature-informed decomposition parameter for reduced susceptibility "
            "after aP vaccination. Aggregate studies support high disease protection "
            "and waning protection, but do not directly identify this component from "
            "surveillance data; [0.30, 0.60] is retained as a modeling range for "
            "sensitivity and pilot uncertainty runs."
        ),
        "resistance_prevalence": (
            "Resistance prevalence is fixed at the country-timeline value during the "
            "primary conditional beta-grid analysis. The country_resistance_timeline.csv "
            "provides well-constrained estimates for most countries; fixed, low-to-very-high "
            "resistance scenarios and the fitness grid evaluate the corresponding "
            "structural uncertainty."
        ),
    }
    prior_rows = []
    if bayesian:
        prior_rows.extend(
            [
                {
                    "parameter": "log_beta_S",
                    "prior": f"Normal(log calibrated beta_S, {bayesian.get('priors', {}).get('log_beta_S_sd', '')})",
                    "interpretation": "Transmission-rate uncertainty",
                },
                {
                    "parameter": "log_reporting_multiplier",
                    "prior": f"Normal(log calibrated reporting multiplier, {bayesian.get('priors', {}).get('log_reporting_multiplier_sd', '')})",
                    "interpretation": "Surveillance/reporting uncertainty",
                },
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
                prior_rows.append(
                    {
                        "parameter": parameter,
                        "prior": prior,
                        "interpretation": bayesian_interpretation_overrides.get(
                            parameter,
                            values.get("note", ""),
                        ),
                    }
                )
        write_dataframe(pd.DataFrame(prior_rows), project_path("publication_inputs/bayesian_prior_table.csv"))

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
    write_dataframe(pd.DataFrame(country_rows), project_path("publication_inputs/country_profile_table.csv"))

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
