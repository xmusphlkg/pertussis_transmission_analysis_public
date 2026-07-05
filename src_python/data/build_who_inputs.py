from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from src_python.data.xlsx import read_xlsx
from src_python.utils.io import load_yaml, project_path


def _data_sources() -> dict[str, Any]:
    settings_path = project_path("config/model_settings.yaml")
    if not settings_path.exists():
        raise FileNotFoundError("config/model_settings.yaml is required.")
    settings = load_yaml(settings_path)
    sources = settings.get("runtime", {}).get("data_sources")
    if not sources:
        raise ValueError("config/model_settings.yaml is missing runtime.data_sources block.")
    return sources


def _source_path(source_key: str) -> Path:
    sources = _data_sources()
    return project_path(sources[source_key])


def _clean_number(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else np.nan


def _year_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if str(column).isdigit()]


def _country_reference() -> pd.DataFrame:
    rows = []
    for _, meta in _data_sources().get("countries", {}).items():
        rows.append(
            {
                "config_key": meta["config_key"],
                "iso3": meta["iso3"],
                "config_country_name": meta.get("contactdata_country", meta["config_key"]),
            }
        )
    reference = pd.DataFrame(rows)
    introduction = read_xlsx(_source_path("who_ap_introduction_xlsx"))
    name_by_iso = (
        introduction.loc[:, ["ISO_3_CODE", "COUNTRYNAME"]]
        .dropna()
        .drop_duplicates("ISO_3_CODE")
        .rename(columns={"ISO_3_CODE": "iso3", "COUNTRYNAME": "who_country_name"})
    )
    return reference.merge(name_by_iso, on="iso3", how="left")


def _selected_iso3() -> set[str]:
    return set(_country_reference()["iso3"].dropna().astype(str))


def build_reported_cases() -> pd.DataFrame:
    df = read_xlsx(_source_path("who_reported_cases_xlsx"))
    ref = _country_reference()
    rows = []
    for _, record in df.iterrows():
        name = str(record["Country / Region"]).strip()
        match = ref.loc[ref["who_country_name"].eq(name)]
        if match.empty:
            continue
        config_key = match["config_key"].iloc[0] if not match.empty else ""
        iso3 = match["iso3"].iloc[0] if not match.empty else ""
        for year_col in _year_columns(df):
            cases = _clean_number(record[year_col])
            if np.isnan(cases):
                continue
            rows.append(
                {
                    "country_or_region": name,
                    "config_key": config_key,
                    "iso3": iso3,
                    "year": int(year_col),
                    "reported_cases": cases,
                    "source": "WHO/UNICEF Joint Reporting Form via WHO Immunization Data Portal",
                    "source_type": "measured",
                }
            )
    return pd.DataFrame(rows)


def build_region_incidence() -> pd.DataFrame:
    df = read_xlsx(_source_path("who_region_incidence_xlsx"))
    rows = []
    for _, record in df.iterrows():
        name = str(record["Country / Region"]).strip()
        for year_col in _year_columns(df):
            incidence = _clean_number(record[year_col])
            if np.isnan(incidence):
                continue
            rows.append(
                {
                    "region": name,
                    "year": int(year_col),
                    "reported_incidence_per_million": incidence,
                    "source": "WHO/UNICEF Joint Reporting Form via WHO Immunization Data Portal",
                    "source_type": "measured",
                }
            )
    return pd.DataFrame(rows)


def build_ap_introduction() -> pd.DataFrame:
    df = read_xlsx(_source_path("who_ap_introduction_xlsx"))
    out = df.rename(
        columns={
            "ISO_3_CODE": "iso3",
            "COUNTRYNAME": "country_name",
            "WHO_REGION": "who_region",
            "YEAR": "year",
            "INTRO": "introduced",
        }
    ).loc[:, ["iso3", "country_name", "who_region", "year", "introduced"]]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["introduced"] = out["introduced"].astype(str).str.lower().eq("yes")
    out = out.loc[out["iso3"].isin(_selected_iso3())].copy()
    out["source_type"] = "measured"
    return out


def build_schedule() -> pd.DataFrame:
    df = read_xlsx(_source_path("who_schedule_xlsx"))
    out = df.rename(
        columns={
            "ISO_3_CODE": "iso3",
            "COUNTRYNAME": "country_name",
            "WHO_REGION": "who_region",
            "YEAR": "year",
            "VACCINECODE": "vaccine_code",
            "TARGETPOP": "target_population_code",
            "TARGETPOP_DESCRIPTION": "target_population",
            "AGEADMINISTERED": "age_administered",
        }
    )
    out = out.loc[out["iso3"].isin(_selected_iso3())].copy()
    out["source_type"] = "measured"
    return out


def build_maternal_coverage() -> pd.DataFrame:
    df = read_xlsx(_source_path("maternal_immunization_coverage_xlsx"))
    out = df.rename(
        columns={
            "CODE": "iso3",
            "NAME": "country_name",
            "YEAR": "year",
            "COVERAGE_CATEGORY_DESCRIPTION": "coverage_category",
            "Coverage": "coverage_raw",
        }
    )
    out["coverage_percent"] = out["coverage_raw"].apply(_clean_number)
    out["coverage_proportion"] = np.where(out["coverage_percent"] > 1.0, out["coverage_percent"] / 100.0, out["coverage_percent"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.loc[out["iso3"].isin(_selected_iso3())].copy()
    out["source_type"] = "measured"
    return out


def build_vaccine_schedule_summary() -> pd.DataFrame:
    df = read_xlsx(_source_path("vaccine_schedule_summary_xlsx"), sheet_name="VaccineSchedule")
    out = df.rename(columns={"CODE": "iso3", "NAME": "country_name"})
    out = out.loc[out["iso3"].isin(_selected_iso3())].copy()
    out["source_type"] = "measured"
    return out


def main() -> None:
    out_dir = project_path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "who_pertussis_reported_cases.csv": build_reported_cases(),
        "who_pertussis_region_incidence.csv": build_region_incidence(),
        "who_ap_introduction.csv": build_ap_introduction(),
        "who_vaccination_schedule_pertussis.csv": build_schedule(),
        "maternal_immunization_coverage_processed.csv": build_maternal_coverage(),
        "vaccine_schedule_summary_processed.csv": build_vaccine_schedule_summary(),
    }
    for filename, df in outputs.items():
        df.to_csv(out_dir / filename, index=False)


if __name__ == "__main__":
    main()
