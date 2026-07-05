from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.data.xlsx import read_xlsx
from src_python.utils.io import load_yaml, project_path


OUTPUT_CSV = project_path("data/processed/country_profile_inputs.csv")


CATEGORY_RANK = {
    "official coverage": 3,
    "administrative coverage": 2,
    "article": 1,
}

DTP_CATEGORY_RANK = {
    "official coverage": 3,
    "administrative coverage": 2,
    "who/unicef estimates of national immunization coverage": 1,
}


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    repo_path = project_path(path)
    if repo_path.exists():
        return repo_path
    return path


def _clean_number(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else np.nan


def _parse_year(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"\d{4}", text)
    return float(match.group(0)) if match else np.nan


def _age_to_months(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.match(r"(?i)^(?P<unit>[mwy])\s*(?P<value>\d+(?:\.\d+)?)", text)
    if not match:
        match = re.search(r"(?i)(?P<unit>[mwy])\s*(?P<value>\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    unit = match.group("unit").lower()
    amount = float(match.group("value"))
    if unit == "w":
        return amount / 4.345238095238095
    if unit == "y":
        return amount * 12.0
    return amount


def load_data_sources() -> dict[str, Any]:
    settings_path = project_path("config/model_settings.yaml")
    if not settings_path.exists():
        raise FileNotFoundError("config/model_settings.yaml is required.")
    settings = load_yaml(settings_path)
    sources = settings.get("runtime", {}).get("data_sources")
    if not sources:
        raise ValueError("config/model_settings.yaml is missing runtime.data_sources block.")
    return sources


def _country_rows() -> pd.DataFrame:
    sources = load_data_sources()
    rows = []
    for _, meta in sources["countries"].items():
        rows.append(
            {
                "config_key": meta["config_key"],
                "iso3": meta["iso3"],
                "country_name": meta.get("contactdata_country", meta["config_key"]),
            }
        )
    return pd.DataFrame(rows)


def _select_maternal_row(df: pd.DataFrame, *, iso3: str) -> pd.Series | None:
    rows = df.loc[df["CODE"].eq(iso3)].copy()
    if rows.empty:
        return None

    rows["coverage_numeric"] = rows["Coverage"].map(_clean_number)
    rows = rows.loc[rows["coverage_numeric"].notna()].copy()
    if rows.empty:
        return None

    rows["category_rank"] = (
        rows["COVERAGE_CATEGORY_DESCRIPTION"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(CATEGORY_RANK)
        .fillna(0)
    )
    rows["year_numeric"] = rows["YEAR"].map(_parse_year)
    rows = rows.sort_values(
        ["year_numeric", "category_rank", "coverage_numeric"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return rows.iloc[0]


def _dtp_source_type(category: Any) -> str:
    text = str(category).strip().lower()
    if "official coverage" in text:
        return "official"
    if "administrative coverage" in text:
        return "administrative"
    if "who/unicef estimates of national immunization coverage" in text:
        return "wuenic"
    return "measured"


def _select_dtp_row(df: pd.DataFrame, *, iso3: str, antigen: str) -> pd.Series | None:
    rows = df.loc[
        df["GROUP"].astype(str).eq("COUNTRIES")
        & df["CODE"].astype(str).eq(iso3)
        & df["ANTIGEN"].astype(str).eq(antigen)
    ].copy()
    if rows.empty:
        return None

    rows["coverage_numeric"] = rows["COVERAGE"].map(_clean_number)
    rows = rows.loc[rows["coverage_numeric"].notna()].copy()
    if rows.empty:
        return None

    rows["category_rank"] = (
        rows["COVERAGE_CATEGORY_DESCRIPTION"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(DTP_CATEGORY_RANK)
        .fillna(0)
    )
    rows["year_numeric"] = rows["YEAR"].map(_parse_year)
    rows = rows.sort_values(
        ["year_numeric", "category_rank", "coverage_numeric"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return rows.iloc[0]


def _routine_age_pattern(routine_rows: pd.DataFrame) -> str:
    ages = []
    for value in routine_rows["AGEADMINISTERED"].fillna(""):
        text = str(value).strip()
        if not text:
            continue
        if text not in ages:
            ages.append(text)
    ages.sort(key=lambda item: (np.isnan(_age_to_months(item)), _age_to_months(item), item))
    return ";".join(ages)


def _first_nonempty(values: pd.Series | list[Any] | tuple[Any, ...]) -> str:
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _schedule_measurements(
    detailed_schedule: pd.DataFrame,
    summary_row: pd.Series | None,
) -> dict[str, Any]:
    routine_rows = detailed_schedule.loc[detailed_schedule["TARGETPOP_DESCRIPTION"].eq("General/routine")].copy()
    pregnant_rows = detailed_schedule.loc[detailed_schedule["TARGETPOP_DESCRIPTION"].eq("Pregnant women")].copy()
    adult_rows = detailed_schedule.loc[detailed_schedule["TARGETPOP_DESCRIPTION"].eq("Adults")].copy()
    risk_rows = detailed_schedule.loc[detailed_schedule["TARGETPOP_DESCRIPTION"].eq("Risk group(s)")].copy()
    summary_maternal_program = _summary_flag(summary_row, "VaccinePregnant")
    summary_adult_program = _summary_flag(summary_row, "VaccineAdult")
    summary_risk_program = _summary_flag(summary_row, "VaccineRisk")

    routine_age_pattern = _routine_age_pattern(routine_rows)
    routine_ages = [age for age in routine_age_pattern.split(";") if age]
    adolescent_booster = any(_age_to_months(age) >= 120.0 for age in routine_ages)
    routine_dose_count = float(len(routine_ages))

    routine_first_shot = np.nan
    routine_last_shot = np.nan
    schedule_year = np.nan
    routine_scheduler_codes = sorted(
        {
            str(value).strip()
            for value in routine_rows["SCHEDULERCODE"].fillna("")
            if str(value).strip()
        }
    )
    routine_scheduler_code = ";".join(routine_scheduler_codes)
    if summary_row is not None:
        routine_first_shot = _clean_number(summary_row.get("TimeFirstShot", np.nan))
        routine_last_shot = _clean_number(summary_row.get("TimeLastShot", np.nan))

    if "YEAR" in detailed_schedule.columns:
        years = pd.to_numeric(detailed_schedule["YEAR"], errors="coerce").dropna()
        if not years.empty:
            schedule_year = float(years.max())

    pregnant_time_window = ""
    if summary_row is not None:
        pregnant_time_window = _first_nonempty([summary_row.get("VaccinePregnantTime", "")])
    if not pregnant_time_window and not pregnant_rows.empty:
        pregnant_time_window = _first_nonempty(pregnant_rows["AGEADMINISTERED"].tolist())
    if not pregnant_time_window and not (len(pregnant_rows) > 0 or summary_maternal_program):
        pregnant_time_window = "No routine maternal programme recorded"

    return {
        "maternal_program": bool(len(pregnant_rows) > 0 or summary_maternal_program),
        "maternal_program_note": pregnant_time_window,
        "adult_program": bool(len(adult_rows) > 0 or summary_adult_program),
        "risk_program": bool(len(risk_rows) > 0 or summary_risk_program),
        "adolescent_booster": bool(adolescent_booster),
        "routine_dose_count": int(round(float(routine_dose_count))),
        "routine_age_pattern": routine_age_pattern,
        "routine_first_shot_months": float(routine_first_shot) if not np.isnan(routine_first_shot) else np.nan,
        "routine_last_shot_months": float(routine_last_shot) if not np.isnan(routine_last_shot) else np.nan,
        "routine_scheduler_code": routine_scheduler_code,
        "schedule_year": schedule_year,
    }


def _summary_flag(summary_row: pd.Series | None, column: str) -> bool:
    if summary_row is None or column not in summary_row:
        return False
    value = summary_row.get(column, "")
    text = str(value).strip().lower()
    if text in {"", "nan", "na", "none"}:
        return False
    if text in {"1", "1.0", "true", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "no", "n"}:
        return False
    numeric = _clean_number(value)
    return bool(not np.isnan(numeric) and numeric > 0.0)


def _maternal_measurement(maternal_row: pd.Series | None, *, maternal_program: bool) -> dict[str, Any]:
    if maternal_row is None:
        if maternal_program:
            return {
                "maternal_coverage": np.nan,
                "maternal_coverage_raw": "",
                "maternal_coverage_year": np.nan,
                "maternal_coverage_category": "",
                "maternal_coverage_source_type": "missing",
                "maternal_coverage_source": "",
            }
        return {
            "maternal_coverage": 0.0,
            "maternal_coverage_raw": "0",
            "maternal_coverage_year": np.nan,
            "maternal_coverage_category": "No routine maternal programme",
            "maternal_coverage_source_type": "derived_no_program",
            "maternal_coverage_source": "No routine maternal programme recorded in schedule sources.",
        }

    coverage_percent = float(maternal_row["coverage_numeric"])
    coverage_proportion = coverage_percent / 100.0 if coverage_percent > 1.0 else coverage_percent
    source_type = str(maternal_row.get("COVERAGE_CATEGORY_DESCRIPTION", "")).strip().lower()
    if "official" in source_type:
        source_type = "official"
    elif "administrative" in source_type:
        source_type = "administrative"
    elif "article" in source_type:
        source_type = "article"
    else:
        source_type = "measured"

    return {
        "maternal_coverage": float(coverage_proportion),
        "maternal_coverage_raw": str(maternal_row.get("Coverage", "")).strip(),
        "maternal_coverage_year": maternal_row.get("YEAR", np.nan),
        "maternal_coverage_category": str(maternal_row.get("COVERAGE_CATEGORY_DESCRIPTION", "")).strip(),
        "maternal_coverage_source_type": source_type,
        "maternal_coverage_source": str(maternal_row.get("source", "")).strip(),
    }


def _dtp_measurement(dtp_df: pd.DataFrame, *, iso3: str, antigen: str, source: str) -> dict[str, Any]:
    row = _select_dtp_row(dtp_df, iso3=iso3, antigen=antigen)
    if row is None:
        raise ValueError(f"Missing DTP coverage for {iso3} / {antigen} in {source}.")
    category = str(row.get("COVERAGE_CATEGORY_DESCRIPTION", "")).strip()
    coverage_percent = float(row["coverage_numeric"])
    coverage_proportion = coverage_percent / 100.0 if coverage_percent > 1.0 else coverage_percent
    return {
        "coverage": float(coverage_proportion),
        "year": row.get("YEAR", np.nan),
        "category": category,
        "source_type": _dtp_source_type(category),
        "source": source,
    }


def _schedule_note(pattern: str, maternal_program_note: str, adult_program: bool, risk_program: bool) -> str:
    if pattern:
        note = f"Routine schedule: {pattern}."
    else:
        note = "Routine schedule: unavailable."
    if maternal_program_note:
        note += f" Maternal programme: {maternal_program_note}."
    else:
        note += " Maternal programme: not reported."
    if adult_program or risk_program:
        extras = []
        if adult_program:
            extras.append("adult doses")
        if risk_program:
            extras.append("risk-group doses")
        note += f" Adult/risk doses: {', '.join(extras)}."
    return note


def build_country_profile_inputs() -> pd.DataFrame:
    sources = load_data_sources()
    if "dtp_coverage_xlsx" not in sources:
        raise KeyError("Missing runtime.data_sources.dtp_coverage_xlsx in config/model_settings.yaml.")
    dtp_source_path = _resolve_path(sources["dtp_coverage_xlsx"])
    dtp_source_name = Path(dtp_source_path).name
    dtp_df = read_xlsx(dtp_source_path)
    schedule_source_path = _resolve_path(sources["who_schedule_xlsx"])
    summary_source_path = _resolve_path(sources["vaccine_schedule_summary_xlsx"])
    schedule_source = f"{schedule_source_path.name}; {summary_source_path.name}"
    maternal_df = read_xlsx(_resolve_path(sources["maternal_immunization_coverage_xlsx"]))
    schedule_df = read_xlsx(schedule_source_path)
    summary_df = read_xlsx(summary_source_path, sheet_name="VaccineSchedule")

    rows = []
    for country_code, meta in sources["countries"].items():
        config_key = meta["config_key"]
        iso3 = meta["iso3"]
        country_name = meta.get("contactdata_country", config_key)

        summary_matches = summary_df.loc[summary_df["CODE"].eq(iso3)].copy()
        summary_row = summary_matches.iloc[0] if not summary_matches.empty else None

        detailed = schedule_df.loc[
            schedule_df["ISO_3_CODE"].eq(iso3) & schedule_df["DISEASEDESCRIPTION"].eq("Pertussis")
        ].copy()
        schedule = _schedule_measurements(detailed, summary_row)
        maternal_row = _select_maternal_row(maternal_df, iso3=iso3)
        maternal = _maternal_measurement(maternal_row, maternal_program=schedule["maternal_program"])
        if not schedule["maternal_program"] and np.isnan(maternal["maternal_coverage_year"]):
            maternal["maternal_coverage_year"] = schedule.get("schedule_year", np.nan)
        dtp1 = _dtp_measurement(dtp_df, iso3=iso3, antigen="DTPCV1", source=dtp_source_name)
        dtp3 = _dtp_measurement(dtp_df, iso3=iso3, antigen="DTPCV3", source=dtp_source_name)

        rows.append(
            {
                "config_key": config_key,
                "iso3": iso3,
                "country_name": country_name,
                "dtp1_coverage": dtp1["coverage"],
                "dtp1_coverage_year": dtp1["year"],
                "dtp1_coverage_category": dtp1["category"],
                "dtp1_coverage_source_type": dtp1["source_type"],
                "dtp1_coverage_source": dtp1["source"],
                "dtp3_coverage": dtp3["coverage"],
                "dtp3_coverage_year": dtp3["year"],
                "dtp3_coverage_category": dtp3["category"],
                "dtp3_coverage_source_type": dtp3["source_type"],
                "dtp3_coverage_source": dtp3["source"],
                "dtp_source_type": "mixed"
                if dtp1["source_type"] != dtp3["source_type"]
                else dtp1["source_type"],
                "dtp_source": dtp1["source"],
                **maternal,
                **schedule,
                "schedule_note": _schedule_note(
                    schedule["routine_age_pattern"],
                    schedule["maternal_program_note"],
                    schedule["adult_program"],
                    schedule["risk_program"],
                ),
                "schedule_source_type": "measured",
                "schedule_source": schedule_source,
            }
        )

    out = pd.DataFrame(rows)
    ordered_columns = [
        "config_key",
        "iso3",
        "country_name",
        "dtp1_coverage",
        "dtp1_coverage_year",
        "dtp1_coverage_category",
        "dtp1_coverage_source_type",
        "dtp1_coverage_source",
        "dtp3_coverage",
        "dtp3_coverage_year",
        "dtp3_coverage_category",
        "dtp3_coverage_source_type",
        "dtp3_coverage_source",
        "dtp_source_type",
        "dtp_source",
        "maternal_program",
        "maternal_program_note",
        "maternal_coverage",
        "maternal_coverage_raw",
        "maternal_coverage_year",
        "maternal_coverage_category",
        "maternal_coverage_source_type",
        "maternal_coverage_source",
        "adult_program",
        "risk_program",
        "adolescent_booster",
        "routine_dose_count",
        "routine_age_pattern",
        "routine_first_shot_months",
        "routine_last_shot_months",
        "routine_scheduler_code",
        "schedule_note",
        "schedule_source_type",
        "schedule_source",
    ]
    return out.loc[:, ordered_columns]


def load_country_profile_inputs() -> pd.DataFrame:
    df = build_country_profile_inputs()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return df


def main() -> None:
    df = build_country_profile_inputs()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
