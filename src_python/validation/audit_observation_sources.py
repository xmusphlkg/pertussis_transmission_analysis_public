"""Reconcile surveillance streams before making absolute-burden claims."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src_python.simulation.common import current_run_metadata, write_run_metadata
from src_python.utils.io import ensure_output_dirs, project_path, write_dataframe


AUDIT_STEM = "observation_source_reconciliation"


def _native_annual(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"config_key", "Cases", "period_start", "period_end"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Native incidence table is missing columns: {sorted(missing)}")
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce")
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["Cases"] = pd.to_numeric(frame["Cases"], errors="coerce")
    frame = frame.dropna(subset=["period_start", "period_end", "Cases"]).copy()
    frame["year"] = (
        frame["period_start"] + (frame["period_end"] - frame["period_start"]) / 2
    ).dt.year
    return (
        frame.groupby(["config_key", "year"], as_index=False)
        .agg(native_cases=("Cases", "sum"), native_intervals=("Cases", "size"))
    )


def _age_annual(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"country", "year", "reported_cases", "is_unknown_age"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Age-case table is missing columns: {sorted(missing)}")
    unknown = frame["is_unknown_age"].astype(str).str.lower().isin({"true", "1"})
    frame["reported_cases"] = pd.to_numeric(frame["reported_cases"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.loc[~unknown].dropna(subset=["year", "reported_cases"]).copy()
    frame["year"] = frame["year"].astype(int)
    return (
        frame.groupby(["country", "year"], as_index=False)["reported_cases"]
        .sum()
        .rename(columns={"country": "config_key", "reported_cases": "age_known_cases"})
    )


def _who_annual(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"config_key", "year", "reported_cases"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"WHO case table is missing columns: {sorted(missing)}")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["reported_cases"] = pd.to_numeric(frame["reported_cases"], errors="coerce")
    frame = frame.dropna(subset=["year", "reported_cases"]).copy()
    frame["year"] = frame["year"].astype(int)
    return (
        frame.groupby(["config_key", "year"], as_index=False)["reported_cases"]
        .sum()
        .rename(columns={"reported_cases": "who_cases"})
    )


def classify_source_reconciliation(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ratios and a predeclared interpretation contract."""

    out = frame.copy()
    out["national_reference_cases"] = out["who_cases"].where(
        out["who_cases"].notna(), out["age_known_cases"]
    )
    out["national_reference_source"] = np.select(
        [out["who_cases"].notna(), out["age_known_cases"].notna()],
        ["WHO_JRF", "age_known_total"],
        default="none",
    )
    denominator = pd.to_numeric(out["national_reference_cases"], errors="coerce")
    native = pd.to_numeric(out["native_cases"], errors="coerce")
    out["native_to_national_ratio"] = native / denominator.where(denominator.gt(0.0))
    ratio = out["native_to_national_ratio"]
    out["reconciliation_status"] = np.select(
        [
            ratio.isna(),
            ratio.lt(2.0 / 3.0) | ratio.gt(1.5),
            ratio.lt(0.8) | ratio.gt(1.25),
        ],
        ["no_national_comparator", "material_scale_mismatch", "moderate_scale_mismatch"],
        default="compatible_scale",
    )
    out["model_use_contract"] = np.where(
        out["reconciliation_status"].eq("compatible_scale"),
        "notification_index_with_static_source_scale",
        "shape_only_with_static_source_scale_no_absolute_burden_claim",
    )
    return out


def build_observation_source_audit() -> pd.DataFrame:
    native = _native_annual(project_path("data/processed/pertussis_incidence_timeseries.csv"))
    age = _age_annual(project_path("data/processed/pertussis_age_case_observations.csv"))
    who = _who_annual(project_path("data/processed/who_pertussis_reported_cases.csv"))
    merged = native.merge(age, on=["config_key", "year"], how="outer").merge(
        who, on=["config_key", "year"], how="outer"
    )
    return classify_source_reconciliation(
        merged.sort_values(["config_key", "year"]).reset_index(drop=True)
    )


def country_source_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country, frame in audit.groupby("config_key", sort=True):
        comparable = frame.loc[frame["native_to_national_ratio"].notna()].copy()
        statuses = set(comparable["reconciliation_status"].astype(str))
        material = "material_scale_mismatch" in statuses
        rows.append(
            {
                "country": country,
                "comparable_years": int(len(comparable)),
                "median_native_to_national_ratio": (
                    float(comparable["native_to_national_ratio"].median())
                    if not comparable.empty
                    else np.nan
                ),
                "minimum_native_to_national_ratio": (
                    float(comparable["native_to_national_ratio"].min())
                    if not comparable.empty
                    else np.nan
                ),
                "maximum_native_to_national_ratio": (
                    float(comparable["native_to_national_ratio"].max())
                    if not comparable.empty
                    else np.nan
                ),
                "material_scale_mismatch": bool(material),
                "absolute_burden_claim_allowed": bool(
                    len(comparable) >= 3 and not material
                ),
                "model_use_contract": (
                    "shape_only_with_static_source_scale_no_absolute_burden_claim"
                    if material or len(comparable) < 3
                    else "notification_index_with_static_source_scale"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ensure_output_dirs()
    audit = build_observation_source_audit()
    summary = country_source_summary(audit)
    write_dataframe(audit, project_path("outputs/tables/observation_source_reconciliation.csv"))
    write_dataframe(summary, project_path("outputs/tables/observation_source_reconciliation_summary.csv"))
    write_run_metadata(
        AUDIT_STEM,
        current_run_metadata(
            AUDIT_STEM,
            row_counts={"annual_comparisons": len(audit), "countries": len(summary)},
        )
        | {
            "absolute_burden_claims_allowed_for_all_countries": bool(
                summary["absolute_burden_claim_allowed"].all()
            ),
            "scope": "source_reconciliation_not_model_validation",
        },
    )


if __name__ == "__main__":
    main()
