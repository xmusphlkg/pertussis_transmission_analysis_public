from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

from src_python.data.build_country_profile_inputs import load_country_profile_inputs
from src_python.data.xlsx import read_xlsx_sheets
from src_python.model.contact_matrix import balance_reciprocity, reciprocity_error
from src_python.utils.io import load_yaml, project_path


AGE_GROUPS = (
    "infant_0_2m",
    "infant_3_11m",
    "child_1_4y",
    "child_5_9y",
    "adolescent_10_17y",
    "young_adult_18_39y",
    "middle_adult_40_64y",
    "elderly_65plus",
)
CONTACT_LOCATIONS = ("all", "home", "school", "work", "other")


BASE_CONTACT_MATRIX = [
    [5.5, 2.2, 1.2, 0.5, 1.0],
    [1.8, 6.2, 2.6, 0.8, 1.2],
    [0.7, 2.1, 9.0, 2.8, 1.4],
    [0.3, 0.8, 3.0, 10.5, 2.6],
    [0.4, 0.7, 1.3, 2.5, 6.0],
]

PREM_CONTACT_BINS = tuple(range(0, 80, 5))
EXCEL_DATE_ORIGIN = pd.Timestamp("1899-12-30")
INFANT_0_2M_FRACTION = 2.0 / 12.0
INFANT_3_11M_FRACTION = 1.0 - INFANT_0_2M_FRACTION


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


REPORTING_DEFAULTS = {
    "infant_0_2m": 0.60,
    "infant_3_11m": 0.50,
    "child_1_4y": 0.25,
    "child_5_9y": 0.18,
    "adolescent_10_17y": 0.08,
    "young_adult_18_39y": 0.05,
    "middle_adult_40_64y": 0.03,
    "elderly_65plus": 0.04,
}

REPORTING_PRIOR_SPECS = {
    "Australia": {
        "evidence_class": "serology_proxy",
        "note": "Australian serology and notification studies suggest under-notification in older children and adults.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.30, "upper": 0.75},
            "infant_3_11m": {"lower": 0.25, "upper": 0.70},
            "child_1_4y": {"lower": 0.10, "upper": 0.50},
            "child_5_9y": {"lower": 0.08, "upper": 0.40},
            "adolescent_10_17y": {"lower": 0.04, "upper": 0.20},
            "young_adult_18_39y": {"lower": 0.01, "upper": 0.12},
            "middle_adult_40_64y": {"lower": 0.005, "upper": 0.10},
            "elderly_65plus": {"lower": 0.01, "upper": 0.12},
        },
    },
    "China": {
        "evidence_class": "active_surveillance_proxy",
        "note": "Active surveillance shows the passive system is likely to miss a substantial share of cases across age groups.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.20, "upper": 0.70},
            "infant_3_11m": {"lower": 0.20, "upper": 0.65},
            "child_1_4y": {"lower": 0.05, "upper": 0.40},
            "child_5_9y": {"lower": 0.04, "upper": 0.30},
            "adolescent_10_17y": {"lower": 0.03, "upper": 0.15},
            "young_adult_18_39y": {"lower": 0.003, "upper": 0.08},
            "middle_adult_40_64y": {"lower": 0.003, "upper": 0.06},
            "elderly_65plus": {"lower": 0.003, "upper": 0.08},
        },
    },
    "United_Kingdom": {
        "evidence_class": "notification_efficiency_low",
        "note": "England and Wales notification-efficiency studies imply low completeness overall, with underdiagnosis persisting in older ages.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.30, "upper": 0.75},
            "infant_3_11m": {"lower": 0.25, "upper": 0.70},
            "child_1_4y": {"lower": 0.10, "upper": 0.45},
            "child_5_9y": {"lower": 0.08, "upper": 0.35},
            "adolescent_10_17y": {"lower": 0.04, "upper": 0.20},
            "young_adult_18_39y": {"lower": 0.005, "upper": 0.10},
            "middle_adult_40_64y": {"lower": 0.005, "upper": 0.08},
            "elderly_65plus": {"lower": 0.005, "upper": 0.10},
        },
    },
    "Japan": {
        "evidence_class": "laboratory_surveillance_proxy",
        "note": "Japanese laboratory-based surveillance improves detection, but school-age and adult under-ascertainment remains likely.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.25, "upper": 0.70},
            "infant_3_11m": {"lower": 0.20, "upper": 0.65},
            "child_1_4y": {"lower": 0.08, "upper": 0.45},
            "child_5_9y": {"lower": 0.06, "upper": 0.35},
            "adolescent_10_17y": {"lower": 0.03, "upper": 0.15},
            "young_adult_18_39y": {"lower": 0.005, "upper": 0.08},
            "middle_adult_40_64y": {"lower": 0.005, "upper": 0.06},
            "elderly_65plus": {"lower": 0.005, "upper": 0.08},
        },
    },
    "New_Zealand": {
        "evidence_class": "high_income_underreporting_proxy",
        "note": "New Zealand surveillance is likely better than passive-only settings, but under-reporting is still acknowledged in the literature.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.30, "upper": 0.75},
            "infant_3_11m": {"lower": 0.25, "upper": 0.70},
            "child_1_4y": {"lower": 0.10, "upper": 0.50},
            "child_5_9y": {"lower": 0.08, "upper": 0.40},
            "adolescent_10_17y": {"lower": 0.04, "upper": 0.18},
            "young_adult_18_39y": {"lower": 0.01, "upper": 0.10},
            "middle_adult_40_64y": {"lower": 0.008, "upper": 0.08},
            "elderly_65plus": {"lower": 0.01, "upper": 0.10},
        },
    },
    "Sweden": {
        "evidence_class": "direct_preschool_anchor",
        "note": "Swedish Child Health Centre validation provides the strongest direct preschool completeness anchor in this set.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.40, "upper": 0.80},
            "infant_3_11m": {"lower": 0.35, "upper": 0.75},
            "child_1_4y": {"lower": 0.25, "upper": 0.60},
            "child_5_9y": {"lower": 0.20, "upper": 0.50},
            "adolescent_10_17y": {"lower": 0.08, "upper": 0.25},
            "young_adult_18_39y": {"lower": 0.02, "upper": 0.12},
            "middle_adult_40_64y": {"lower": 0.015, "upper": 0.10},
            "elderly_65plus": {"lower": 0.02, "upper": 0.12},
        },
    },
    "Brazil": {
        "evidence_class": "passive_surveillance_proxy",
        "note": "Brazilian national notification data support calibration to reported cases, but under-ascertainment outside infants remains likely.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.20, "upper": 0.70},
            "infant_3_11m": {"lower": 0.18, "upper": 0.65},
            "child_1_4y": {"lower": 0.05, "upper": 0.40},
            "child_5_9y": {"lower": 0.04, "upper": 0.30},
            "adolescent_10_17y": {"lower": 0.03, "upper": 0.15},
            "young_adult_18_39y": {"lower": 0.003, "upper": 0.08},
            "middle_adult_40_64y": {"lower": 0.003, "upper": 0.06},
            "elderly_65plus": {"lower": 0.003, "upper": 0.08},
        },
    },
    "Thailand": {
        "evidence_class": "passive_surveillance_proxy",
        "note": "Thailand surveillance provides a national reported-case series, with broad reporting-prior bounds retained for likely passive-system under-ascertainment.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.20, "upper": 0.70},
            "infant_3_11m": {"lower": 0.18, "upper": 0.65},
            "child_1_4y": {"lower": 0.05, "upper": 0.40},
            "child_5_9y": {"lower": 0.04, "upper": 0.30},
            "adolescent_10_17y": {"lower": 0.03, "upper": 0.15},
            "young_adult_18_39y": {"lower": 0.003, "upper": 0.08},
            "middle_adult_40_64y": {"lower": 0.003, "upper": 0.06},
            "elderly_65plus": {"lower": 0.003, "upper": 0.08},
        },
    },
    "United_States": {
        "evidence_class": "capture_recapture_proxy",
        "note": "US capture-recapture and hospitalization studies imply substantial underreporting, especially outside the infant age range.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.30, "upper": 0.75},
            "infant_3_11m": {"lower": 0.25, "upper": 0.70},
            "child_1_4y": {"lower": 0.10, "upper": 0.50},
            "child_5_9y": {"lower": 0.08, "upper": 0.40},
            "adolescent_10_17y": {"lower": 0.04, "upper": 0.18},
            "young_adult_18_39y": {"lower": 0.005, "upper": 0.10},
            "middle_adult_40_64y": {"lower": 0.005, "upper": 0.08},
            "elderly_65plus": {"lower": 0.005, "upper": 0.10},
        },
    },
    "South_Africa": {
        "evidence_class": "passive_notification_proxy",
        "note": "South Africa NMCSS passive notification system; NICD hospital-based surveillance (2013-2018) suggests substantial under-ascertainment outside infants, consistent with middle-income passive systems.",
        "age_groups": {
            "infant_0_2m": {"lower": 0.20, "upper": 0.70},
            "infant_3_11m": {"lower": 0.18, "upper": 0.65},
            "child_1_4y": {"lower": 0.05, "upper": 0.40},
            "child_5_9y": {"lower": 0.04, "upper": 0.30},
            "adolescent_10_17y": {"lower": 0.03, "upper": 0.15},
            "young_adult_18_39y": {"lower": 0.003, "upper": 0.08},
            "middle_adult_40_64y": {"lower": 0.003, "upper": 0.06},
            "elderly_65plus": {"lower": 0.003, "upper": 0.08},
        },
    },
}


def _text_or_default(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return default if text.lower() == "nan" else text


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    repo_path = project_path(path)
    if repo_path.exists():
        return repo_path
    return path


def load_data_sources() -> dict[str, Any]:
    settings_path = project_path("config/model_settings.yaml")
    if not settings_path.exists():
        raise FileNotFoundError("config/model_settings.yaml is required.")
    settings = load_yaml(settings_path)
    sources = settings.get("runtime", {}).get("data_sources")
    if not sources:
        raise ValueError("config/model_settings.yaml is missing runtime.data_sources block.")
    return sources


def _reporting_rate_prior(country: str) -> dict[str, Any]:
    spec = REPORTING_PRIOR_SPECS[country]
    prior = {
        "method": "literature_range",
        "evidence_class": spec["evidence_class"],
        "note": spec["note"],
        "age_groups": {},
    }
    for age_group in AGE_GROUPS:
        band = spec["age_groups"][age_group]
        lower = float(band["lower"])
        upper = float(band["upper"])
        point = float(REPORTING_DEFAULTS[age_group])
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"Invalid reporting prior band for {country} / {age_group}: {lower} to {upper}.")
        # Widen the band if the point estimate falls outside (country-specific
        # priors may be tighter than the global default for some age groups)
        if point < lower:
            lower = max(0.0, point * 0.8)
        if point > upper:
            upper = min(1.0, point * 1.2)
        prior["age_groups"][age_group] = {"lower": lower, "upper": upper}
    return prior


def aggregate_wpp_population(wpp_csv: Path, *, iso3: str, year: int) -> dict[str, float]:
    usecols = ["Iso3", "Time", "Sex", "AgeStart", "Value"]
    df = pd.read_csv(wpp_csv, usecols=usecols)
    df = df.loc[
        df["Iso3"].eq(iso3)
        & df["Time"].eq(year)
        & df["Sex"].eq("Both sexes")
    ].copy()
    if df.empty:
        raise ValueError(f"No WPP population rows found for {iso3} in {year}.")

    age = df["AgeStart"].astype(float)
    value = df["Value"].astype(float)
    age0 = float(value.loc[age.eq(0)].sum())
    return {
        "infant_0_2m": age0 * INFANT_0_2M_FRACTION,
        "infant_3_11m": age0 * INFANT_3_11M_FRACTION,
        "child_1_4y": float(value.loc[age.between(1, 4)].sum()),
        "child_5_9y": float(value.loc[age.between(5, 9)].sum()),
        "adolescent_10_17y": float(value.loc[age.between(10, 17)].sum()),
        "young_adult_18_39y": float(value.loc[age.between(18, 39)].sum()),
        "middle_adult_40_64y": float(value.loc[age.between(40, 64)].sum()),
        "elderly_65plus": float(value.loc[age.ge(65)].sum()),
    }


def _aggregate_ages_to_groups(age: np.ndarray, value: np.ndarray) -> dict[str, float]:
    age0 = float(value[age == 0].sum())
    return {
        "infant_0_2m": age0 * INFANT_0_2M_FRACTION,
        "infant_3_11m": age0 * INFANT_3_11M_FRACTION,
        "child_1_4y": float(value[(age >= 1) & (age <= 4)].sum()),
        "child_5_9y": float(value[(age >= 5) & (age <= 9)].sum()),
        "adolescent_10_17y": float(value[(age >= 10) & (age <= 17)].sum()),
        "young_adult_18_39y": float(value[(age >= 18) & (age <= 39)].sum()),
        "middle_adult_40_64y": float(value[(age >= 40) & (age <= 64)].sum()),
        "elderly_65plus": float(value[age >= 65].sum()),
    }


def load_wpp_annual_by_age_group(
    wpp_csv: Path,
    *,
    iso3: str,
    years: list[int] | tuple[int, ...] | None = None,
    births_csv: Path | None = None,
) -> pd.DataFrame:
    """Return a year x age-group population table for ``iso3``.

    Columns: ``year``, ``infant_0_2m``, ``infant_3_11m``, ``child_1_6y``,
    ``school_7_17y``, ``adult_18plus``, ``births`` (age-0 count, used to drive
    demographic entries), ``total_population``.

    If ``births_csv`` is provided and contains years not in the main WPP file,
    those years are added using the earliest available age structure scaled by
    the ratio of births (as a proxy for total population scaling). This allows
    extending the trajectory back to 1950 for burn-in purposes.
    """
    usecols = ["Iso3", "Time", "Sex", "AgeStart", "Value"]
    df = pd.read_csv(wpp_csv, usecols=usecols)
    df = df.loc[df["Iso3"].eq(iso3) & df["Sex"].eq("Both sexes")].copy()
    if df.empty:
        raise ValueError(f"No WPP population rows found for {iso3}.")

    # Build rows from full age-structured data
    available_years = sorted(df["Time"].unique())
    rows = []
    for year_val, group in df.groupby("Time", sort=True):
        age = group["AgeStart"].to_numpy(dtype=int)
        value = group["Value"].to_numpy(dtype=float)
        bucket = _aggregate_ages_to_groups(age, value)
        rows.append(
            {
                "year": int(year_val),
                **{k: float(v) for k, v in bucket.items()},
                "births": float(value[age == 0].sum()),
                "total_population": float(value.sum()),
            }
        )

    # Extend backward using births-only data if available
    if births_csv is not None and births_csv.exists():
        births_df = pd.read_csv(births_csv)
        # The WPP data file may contain multiple indicators (e.g. population
        # by age AND crude birth rate).  We only want the age-0 population
        # count (IndicatorId == 47) which serves as the births proxy.
        # If crude birth rate (IndicatorId == 55) is also available, use it
        # to derive a better total_population estimate for backward years.
        cbr_lookup: dict[int, float] = {}
        if "IndicatorId" in births_df.columns:
            cbr_rows = births_df.loc[
                births_df["IndicatorId"].eq(55) & births_df["Iso3"].eq(iso3)
            ]
            for _, crow in cbr_rows.iterrows():
                cbr_lookup[int(crow["Time"])] = float(crow["Value"])
            births_df = births_df.loc[births_df["IndicatorId"].eq(47)]
        births_country = births_df.loc[births_df["Iso3"].eq(iso3)].copy()
        if not births_country.empty:
            earliest_year = min(available_years)
            # Get the earliest full age-structure as reference
            ref_group = df.loc[df["Time"].eq(earliest_year)]
            ref_age = ref_group["AgeStart"].to_numpy(dtype=int)
            ref_value = ref_group["Value"].to_numpy(dtype=float)
            ref_bucket = _aggregate_ages_to_groups(ref_age, ref_value)
            ref_births = float(ref_value[ref_age == 0].sum())
            ref_total = float(ref_value.sum())

            for _, brow in births_country.iterrows():
                byear = int(brow["Time"])
                if byear >= earliest_year:
                    continue
                bvalue = float(brow["Value"])
                # Scale the reference age structure by the birth ratio
                scale = bvalue / max(ref_births, 1.0)
                scaled_bucket = {k: float(v) * scale for k, v in ref_bucket.items()}
                # Use crude birth rate to derive a more accurate total population
                # when available: total_pop = births / (CBR / 1000)
                if byear in cbr_lookup and cbr_lookup[byear] > 0:
                    total_pop = bvalue / (cbr_lookup[byear] / 1000.0)
                else:
                    total_pop = ref_total * scale
                rows.append(
                    {
                        "year": byear,
                        **scaled_bucket,
                        "births": bvalue,
                        "total_population": total_pop,
                    }
                )

    result = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if years is not None:
        result = result.loc[result["year"].isin(list(years))].reset_index(drop=True)
        if result.empty:
            raise ValueError(f"No WPP population rows for {iso3} in requested years {list(years)}.")
    return result


def load_wpp_one_year(wpp_csv: Path, *, iso3: str, year: int) -> pd.DataFrame:
    usecols = ["Iso3", "Time", "Sex", "AgeStart", "Value"]
    df = pd.read_csv(wpp_csv, usecols=usecols)
    df = df.loc[
        df["Iso3"].eq(iso3)
        & df["Time"].eq(year)
        & df["Sex"].eq("Both sexes")
    ].copy()
    if df.empty:
        raise ValueError(f"No WPP population rows found for {iso3} in {year}.")
    df["AgeStart"] = df["AgeStart"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    return df


def load_incidence(
    incidence_path: Path,
    country_code: str,
    *,
    surveillance_year: int | None = None,
) -> pd.DataFrame:
    df = _read_incidence_source(incidence_path)
    country_code = _normalize_country_code(country_code)
    df = df.loc[df["Country"].eq(country_code)].copy()
    if df.empty:
        raise ValueError(f"No PertussisIncidence rows found for {country_code}.")
    if surveillance_year is not None:
        df = df.loc[df["Year"].le(int(surveillance_year))].copy()
        if df.empty:
            raise ValueError(
                f"No PertussisIncidence rows found for {country_code} at or before surveillance year {surveillance_year}."
            )
    df = _drop_blank_incidence_columns(df)
    return df.sort_values(["period_start", "period_end", "Year"]).reset_index(drop=True)


def _normalize_country_code(value: Any) -> str:
    return str(value).strip().upper()


def _read_incidence_source(path: Path) -> pd.DataFrame:
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pertussis incidence source not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        frames = []
        for sheet_name, sheet_df in read_xlsx_sheets(path).items():
            if sheet_df.empty:
                continue
            frame = sheet_df.copy()
            frame.columns = [str(column).strip() for column in frame.columns]
            frame["source_sheet"] = str(sheet_name).strip()
            if "Country" not in frame.columns:
                frame["Country"] = frame["source_sheet"]
            frames.append(frame)
        if not frames:
            raise ValueError(f"No incidence worksheets found in {path}.")
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = pd.read_csv(path)
        raw["source_sheet"] = ""
    return _normalize_incidence_frame(raw)


def _normalize_incidence_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    required = {"Date", "Year", "Disease", "Cases", "Country"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Pertussis incidence source is missing columns: {sorted(missing)}")
    for column in ["Month", "Week", "URL", "source_sheet"]:
        if column not in df.columns:
            df[column] = ""

    df["Country"] = df["Country"].map(_normalize_country_code)
    missing_country = df["Country"].eq("") | df["Country"].str.lower().eq("nan")
    df.loc[missing_country, "Country"] = df.loc[missing_country, "source_sheet"].map(_normalize_country_code)
    df["source_sheet"] = df["source_sheet"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce").fillna(0.0)
    df["Date"] = _parse_incidence_dates(df["Date"])
    df = df.dropna(subset=["Country", "Year", "Date"]).copy()
    df["Year"] = df["Year"].astype(int)

    has_week = df["Week"].notna()
    # A valid month must be in range 1-12; values outside this (e.g. Excel
    # serial dates stored in the Month column for weekly-reporting sheets)
    # should not be treated as monthly data.
    has_month = df["Month"].notna() & df["Month"].between(1, 12) & ~has_week
    df["reporting_frequency"] = np.select(
        [has_week, has_month],
        ["weekly", "monthly"],
        default="annual",
    )
    period_start = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if has_month.any():
        monthly_start = pd.to_datetime(
            {
                "year": df.loc[has_month, "Year"].astype(int),
                "month": df.loc[has_month, "Month"].astype(int),
                "day": 1,
            },
            errors="coerce",
        )
        period_start.loc[has_month] = monthly_start
    period_start.loc[has_week] = df.loc[has_week, "Date"]
    annual_mask = ~(has_week | has_month)
    period_start.loc[annual_mask] = pd.to_datetime(
        {"year": df.loc[annual_mask, "Year"].astype(int), "month": 1, "day": 1},
        errors="coerce",
    )
    period_start = period_start.fillna(df["Date"])
    df["period_start"] = period_start

    period_end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    period_end.loc[has_week] = df.loc[has_week, "period_start"] + pd.to_timedelta(7, unit="D")
    if has_month.any():
        period_end.loc[has_month] = df.loc[has_month, "period_start"] + pd.DateOffset(months=1)
    period_end.loc[annual_mask] = df.loc[annual_mask, "period_start"] + pd.DateOffset(years=1)
    df["period_end"] = period_end
    df["interval_days"] = (df["period_end"] - df["period_start"]).dt.days.astype(float)
    df["period_midpoint"] = df["period_start"] + (df["period_end"] - df["period_start"]) / 2
    df = df.loc[df["interval_days"].gt(0)].copy()
    return df


def _parse_incidence_dates(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    dates = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    numeric_mask = numeric.notna()
    dates.loc[numeric_mask] = EXCEL_DATE_ORIGIN + pd.to_timedelta(numeric.loc[numeric_mask], unit="D")
    dates.loc[~numeric_mask] = pd.to_datetime(values.loc[~numeric_mask], errors="coerce")
    return dates


def _drop_blank_incidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = []
    for column in df.columns:
        if str(column).startswith("column_"):
            continue
        values = df[column]
        if values.isna().all():
            continue
        if values.astype(str).str.strip().replace({"nan": "", "NaT": ""}).eq("").all():
            continue
        keep_columns.append(column)
    return df.loc[:, keep_columns].copy()


def infer_seasonality(incidence: pd.DataFrame) -> dict[str, float]:
    df = incidence.loc[
        incidence["Cases"].gt(0)
        & incidence.get("reporting_frequency", pd.Series("", index=incidence.index)).isin(["weekly", "monthly"])
    ].copy()
    if df.empty:
        return {"seasonal_amplitude": 0.12, "seasonal_phase": 30.0}

    date_column = "period_midpoint" if "period_midpoint" in df.columns else "Date"
    day_of_year = df[date_column].dt.dayofyear.clip(upper=365).to_numpy(dtype=float)
    theta = 2.0 * np.pi * (day_of_year - 1.0) / 365.0
    weights = df["Cases"].to_numpy(dtype=float)
    sin_sum = float(np.sum(weights * np.sin(theta)))
    cos_sum = float(np.sum(weights * np.cos(theta)))
    total = max(float(np.sum(weights)), 1e-9)
    concentration = min(1.0, np.sqrt(sin_sum**2 + cos_sum**2) / total)
    phase = (np.arctan2(sin_sum, cos_sum) * 365.0 / (2.0 * np.pi) + 1.0) % 365.0
    amplitude = float(np.clip(0.08 + 0.55 * concentration, 0.08, 0.35))
    return {
        "seasonal_amplitude": amplitude,
        "seasonal_phase": float(phase),
        "seasonal_concentration": float(concentration),
    }


def infer_multiyear_cycle(incidence: pd.DataFrame) -> dict[str, float]:
    annual = _annualized_incidence_table(incidence)
    annual = annual.loc[annual["coverage_days"].ge(300.0)].sort_values("Year")
    if len(annual) < 6:
        return {"multi_year_period_years": 4.0, "multi_year_amplitude": 0.0, "multi_year_supported": False}

    values = annual["annualized_cases"].to_numpy(dtype=float)
    years = annual["Year"].to_numpy(dtype=float)
    prominence = max(float(np.max(values)) * 0.10, 1.0)
    peak_idx, _ = find_peaks(values, distance=2, prominence=prominence)
    intervals = np.diff(years[peak_idx])
    plausible = intervals[(intervals >= 3.0) & (intervals <= 5.0)]
    if len(plausible) == 0:
        return {
            "multi_year_period_years": 4.0,
            "multi_year_amplitude": 0.0,
            "multi_year_supported": False,
            "observed_peak_years": ";".join(str(int(years[i])) for i in peak_idx),
        }
    period = float(np.median(plausible)) if len(plausible) else 4.0
    cv = float(np.std(values) / max(np.mean(values), 1e-9))
    amplitude = float(np.clip(0.06 + 0.08 * cv, 0.06, 0.16))
    return {
        "multi_year_period_years": period,
        "multi_year_amplitude": amplitude,
        "multi_year_supported": True,
        "observed_peak_years": ";".join(str(int(years[i])) for i in peak_idx),
    }


def observed_annual_incidence(incidence: pd.DataFrame, total_population: float) -> dict[str, float]:
    annual = _annualized_incidence_table(incidence)
    annual_incidence = annual["annualized_cases"].to_numpy(dtype=float) / max(total_population, 1e-9) * 100_000.0
    return {
        "observed_mean_annual_reported_incidence_per_100k": float(np.mean(annual_incidence)),
        "observed_peak_annual_reported_incidence_per_100k": float(np.max(annual_incidence)),
        "observed_record_count": int(len(incidence)),
        "observed_first_year": int(annual["Year"].min()),
        "observed_last_year": int(annual["Year"].max()),
    }


def _annualized_incidence_table(incidence: pd.DataFrame) -> pd.DataFrame:
    df = incidence.copy()
    if "interval_days" not in df.columns:
        df["interval_days"] = 365.0
    annual = (
        df.groupby("Year", as_index=False)
        .agg(Cases=("Cases", "sum"), coverage_days=("interval_days", "sum"))
        .sort_values("Year")
        .reset_index(drop=True)
    )
    annual["annualized_cases"] = annual["Cases"] / annual["coverage_days"].clip(lower=1.0) * 365.0
    return annual


def coverage_by_age(dtp1_coverage: float, dtp3_coverage: float, meta: dict[str, Any]) -> dict[str, float]:
    if pd.isna(dtp1_coverage) or pd.isna(dtp3_coverage):
        raise ValueError(f"Missing measured DTP coverage for {meta.get('config_key', 'unknown')}.")
    dtp1 = float(dtp1_coverage)
    dtp3 = float(dtp3_coverage)
    child_boosters = float(meta.get("child_booster_doses", 1))
    maternal_coverage = float(meta.get("maternal_coverage", 0.0))
    if pd.isna(maternal_coverage):
        maternal_coverage = 0.0
    adolescent_booster = bool(meta.get("adolescent_booster", False))

    infant_birth_protection = 0.02 + 0.55 * maternal_coverage
    partial_infant_series = 0.75 * dtp1 + 0.12 * dtp3 + 0.12 * maternal_coverage
    preschool_boost = 0.88 + 0.04 * min(child_boosters, 2.0)
    school_age_boost = 0.80 + 0.06 * min(child_boosters, 2.0)
    adolescent_boost = 0.55 + 0.12 * min(child_boosters, 2.0) + (0.15 if adolescent_booster else 0.0)
    young_adult_proxy = 0.20 + (0.15 if adolescent_booster else 0.0) + (0.05 * maternal_coverage)
    middle_adult_proxy = 0.12 + (0.05 if adolescent_booster else 0.0) + (0.02 * maternal_coverage)
    elderly_proxy = 0.08 + (0.02 if adolescent_booster else 0.0)
    return {
        "infant_0_2m": float(np.clip(infant_birth_protection, 0.02, 0.75)),
        "infant_3_11m": float(np.clip(partial_infant_series, 0.0, 0.95)),
        "child_1_4y": float(np.clip(dtp3 * preschool_boost, 0.0, 0.98)),
        "child_5_9y": float(np.clip(dtp3 * school_age_boost, 0.0, 0.96)),
        "adolescent_10_17y": float(np.clip(dtp3 * adolescent_boost, 0.0, 0.95)),
        "young_adult_18_39y": float(np.clip(young_adult_proxy, 0.10, 0.75)),
        "middle_adult_40_64y": float(np.clip(middle_adult_proxy, 0.08, 0.50)),
        "elderly_65plus": float(np.clip(elderly_proxy, 0.05, 0.30)),
    }


def _model_age_population_by_prem_bin(one_year_population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in one_year_population.iterrows():
        age = int(record["AgeStart"])
        value = float(record["Value"])
        prem_bin = min((age // 5) * 5, 75)
        if age == 0:
            rows.extend(
                [
                    {
                        "age_group": "infant_0_2m",
                        "prem_bin": prem_bin,
                        "population": value * INFANT_0_2M_FRACTION,
                    },
                    {
                        "age_group": "infant_3_11m",
                        "prem_bin": prem_bin,
                        "population": value * INFANT_3_11M_FRACTION,
                    },
                ]
            )
        elif 1 <= age <= 4:
            rows.append({"age_group": "child_1_4y", "prem_bin": prem_bin, "population": value})
        elif 5 <= age <= 9:
            rows.append({"age_group": "child_5_9y", "prem_bin": prem_bin, "population": value})
        elif 10 <= age <= 17:
            rows.append({"age_group": "adolescent_10_17y", "prem_bin": prem_bin, "population": value})
        elif 18 <= age <= 39:
            rows.append({"age_group": "young_adult_18_39y", "prem_bin": prem_bin, "population": value})
        elif 40 <= age <= 64:
            rows.append({"age_group": "middle_adult_40_64y", "prem_bin": prem_bin, "population": value})
        else:
            rows.append({"age_group": "elderly_65plus", "prem_bin": prem_bin, "population": value})
    return pd.DataFrame(rows)


def _prem_bin_label(lower: int) -> str:
    return f"[{lower:02d},{lower + 5:02d})"


def aggregate_contact_matrix(
    contact_df: pd.DataFrame,
    one_year_population: pd.DataFrame,
    *,
    country_key: str,
    location: str | None = None,
) -> tuple[list[list[float]], dict[str, float | bool]]:
    country_contacts = contact_df.loc[contact_df["country"].eq(country_key)].copy()
    if "location" in country_contacts.columns:
        contact_location = "all" if location is None else location
        country_contacts = country_contacts.loc[country_contacts["location"].eq(contact_location)].copy()
    elif location not in (None, "all"):
        raise ValueError(f"No location-specific contact matrix rows available for {country_key}, location={location}.")
    if country_contacts.empty:
        suffix = "" if location in (None, "all") else f", location={location}"
        raise ValueError(f"No Prem/contactdata matrix rows found for {country_key}{suffix}.")

    labels = [_prem_bin_label(lower) for lower in PREM_CONTACT_BINS]
    fine = (
        country_contacts.pivot_table(
            index="source_age_bin",
            columns="target_age_bin",
            values="contacts_per_day",
            aggfunc="mean",
        )
        .reindex(index=labels, columns=labels)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    overlap = _model_age_population_by_prem_bin(one_year_population)
    bin_pop = (
        overlap.pivot_table(index="age_group", columns="prem_bin", values="population", aggfunc="sum")
        .reindex(index=AGE_GROUPS, columns=PREM_CONTACT_BINS)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    model_pop = np.maximum(bin_pop.sum(axis=1), 1e-12)
    fine_pop = np.maximum(bin_pop.sum(axis=0), 1e-12)
    row_weights = bin_pop / model_pop[:, None]
    column_fractions = bin_pop / fine_pop[None, :]
    aggregated = row_weights @ fine @ column_fractions.T
    model_population = np.maximum(bin_pop.sum(axis=1), 1e-12)
    error_before = reciprocity_error(aggregated, model_population)
    balanced = balance_reciprocity(aggregated, model_population)
    error_after = reciprocity_error(balanced, model_population)
    return balanced.round(6).tolist(), {
        "reciprocity_correction_applied": True,
        "reciprocity_error_before": float(error_before),
        "reciprocity_error_after": float(error_after),
    }


def build_profiles() -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    sources = load_data_sources()
    wpp_csv = _resolve_path(sources["wpp_population_csv"])
    incidence_path = _resolve_path(
        sources.get("pertussis_incidence_path")
        or sources.get("pertussis_incidence_xlsx")
        or sources.get("pertussis_incidence_csv")
    )
    contact_csv = _resolve_path(sources["contact_matrix_csv"])
    year = int(sources.get("population_year", sources.get("analysis_year", 2023)))
    cutoff_policy = str(sources.get("incidence_cutoff_policy", "surveillance_year"))
    surveillance_year = None if cutoff_policy == "all_records" else int(
        sources.get("surveillance_year", sources.get("analysis_year", 2023))
    )
    contact_df = pd.read_csv(contact_csv)
    setting_contact_csv = sources.get("contact_matrix_by_location_csv")
    setting_contact_df = (
        pd.read_csv(_resolve_path(setting_contact_csv))
        if setting_contact_csv and _resolve_path(setting_contact_csv).exists()
        else pd.DataFrame()
    )
    profile_inputs = load_country_profile_inputs().set_index("config_key", drop=False)

    # Analysis horizon drives how many WPP years we embed into the YAML profile.
    # We always emit a 2 year buffer on either side of the analysis window so the
    # ODE demographic trajectory can interpolate without extrapolating.
    settings_path = project_path("config/model_settings.yaml")
    horizon_start_year = None
    horizon_end_year = None
    if settings_path.exists():
        settings = load_yaml(settings_path)
        calendar = settings.get("runtime", {}).get("baseline_parameters", {}).get("calendar", {})
        if calendar.get("analysis_start_date"):
            horizon_start_year = int(str(calendar["analysis_start_date"])[:4])
        if calendar.get("analysis_end_date"):
            horizon_end_year = int(str(calendar["analysis_end_date"])[:4])
    if horizon_start_year is None:
        horizon_start_year = year
    if horizon_end_year is None:
        horizon_end_year = year + 25
    # Extend the trajectory window back to cover the burn-in period.
    # The full age-structured WPP data starts at 1990; for years before that
    # we use the 1990 age structure with births from the extended WPP births file.
    burn_in_years = int(
        settings.get("runtime", {}).get("baseline_parameters", {}).get("simulation", {}).get("burn_in_years", 60)
    ) if settings_path.exists() else 60
    embed_start_year = max(1950, horizon_start_year - burn_in_years - 2)
    embed_end_year = min(2050, horizon_end_year + 2)

    profiles: dict[str, Any] = {}
    population_rows = []
    annual_population_rows = []
    seasonality_rows = []
    contact_rows = []
    contact_diagnostic_rows = []
    setting_contact_rows = []
    setting_contact_diagnostic_rows = []
    incidence_frames = []

    for country_code, meta in sources["countries"].items():
        key = meta["config_key"]
        iso3 = meta["iso3"]
        if key not in profile_inputs.index:
            raise KeyError(f"Missing processed profile inputs for {key}.")
        measurement = profile_inputs.loc[key].to_dict()
        measured_maternal_coverage = float(measurement["maternal_coverage"])
        measured_adolescent_booster = bool(measurement["adolescent_booster"])
        measured_maternal_program = bool(measurement["maternal_program"])
        measured_schedule_note = _text_or_default(measurement["schedule_note"])
        measured_routine_dose_count = int(measurement["routine_dose_count"])
        measured_routine_age_pattern = _text_or_default(measurement["routine_age_pattern"])
        measured_routine_first_shot = float(measurement["routine_first_shot_months"])
        measured_routine_last_shot = float(measurement["routine_last_shot_months"])
        measured_routine_scheduler_code = _text_or_default(measurement["routine_scheduler_code"])
        measured_schedule_source = _text_or_default(measurement["schedule_source"])
        incidence = load_incidence(incidence_path, country_code, surveillance_year=surveillance_year)
        incidence = incidence.assign(config_key=key, iso3=iso3)
        incidence_frames.append(incidence)

        one_year_population = load_wpp_one_year(wpp_csv, iso3=iso3, year=year)
        population = aggregate_wpp_population(wpp_csv, iso3=iso3, year=year)
        # Pass the extended births CSV to allow trajectory extension back to 1950
        births_csv_path = _resolve_path(sources["wpp_births_csv"]) if "wpp_births_csv" in sources else None
        annual_trajectory = load_wpp_annual_by_age_group(wpp_csv, iso3=iso3, births_csv=births_csv_path)
        seasonality = infer_seasonality(incidence)
        cycle = infer_multiyear_cycle(incidence)
        meta_for_vaccination = dict(meta)
        meta_for_vaccination["maternal_coverage"] = measured_maternal_coverage
        meta_for_vaccination["adolescent_booster"] = bool(measured_adolescent_booster)
        meta_for_vaccination["maternal_program"] = bool(measured_maternal_program)
        dtp1 = float(measurement["dtp1_coverage"])
        dtp3 = float(measurement["dtp3_coverage"])
        vaccination = coverage_by_age(dtp1, dtp3, meta_for_vaccination)
        contact_matrix, contact_diagnostics = aggregate_contact_matrix(contact_df, one_year_population, country_key=key)
        setting_contact_matrices: dict[str, list[list[float]]] = {}
        if not setting_contact_df.empty:
            for location in CONTACT_LOCATIONS:
                setting_matrix, setting_diagnostics = aggregate_contact_matrix(
                    setting_contact_df,
                    one_year_population,
                    country_key=key,
                    location=location,
                )
                setting_contact_matrices[location] = setting_matrix
                for source_idx, source_age in enumerate(AGE_GROUPS):
                    for target_idx, target_age in enumerate(AGE_GROUPS):
                        setting_contact_rows.append(
                            {
                                "country": key,
                                "iso3": iso3,
                                "location": location,
                                "source_age_group": source_age,
                                "target_age_group": target_age,
                                "contacts_per_day": setting_matrix[source_idx][target_idx],
                                "source": (
                                    f"Prem/contactdata {meta.get('contactdata_source', 'unknown')} {location} "
                                    "matrix aggregated with WPP age weights and reciprocity balanced"
                                ),
                                "source_type": "derived",
                            }
                        )
                setting_contact_diagnostic_rows.append(
                    {
                        "country": key,
                        "iso3": iso3,
                        "location": location,
                        **setting_diagnostics,
                        "source_type": "derived",
                    }
                )
        total_population = sum(population.values())
        observed_incidence = observed_annual_incidence(incidence, total_population)
        maternal_coverage = float(measured_maternal_coverage)
        birth_vaccinated = float(np.clip(0.02 + 0.55 * maternal_coverage, 0.02, 0.75))

        population_rows.extend(
            {
                "country": key,
                "iso3": iso3,
                "year": year,
                "age_group": age_group,
                "population": value,
                "source": "UN WPP 2024 one-year age population",
                "source_type": "derived",
            }
            for age_group, value in population.items()
        )
        for _, row in annual_trajectory.iterrows():
            annual_population_rows.append(
                {
                    "country": key,
                    "iso3": iso3,
                    "year": int(row["year"]),
                    **{age: float(row[age]) for age in AGE_GROUPS},
                    "births": float(row["births"]),
                    "total_population": float(row["total_population"]),
                    "source": "UN WPP 2024 one-year age population (annual 1990-2050)",
                    "source_type": "derived",
                }
            )
        for source_idx, source_age in enumerate(AGE_GROUPS):
            for target_idx, target_age in enumerate(AGE_GROUPS):
                contact_rows.append(
                    {
                        "country": key,
                        "iso3": iso3,
                        "source_age_group": source_age,
                        "target_age_group": target_age,
                        "contacts_per_day": contact_matrix[source_idx][target_idx],
                        "source": (
                            f"Prem/contactdata {meta.get('contactdata_source', 'unknown')} "
                            "matrix aggregated with WPP age weights and reciprocity balanced"
                        ),
                        "source_type": "derived",
                    }
                )
        contact_diagnostic_rows.append(
            {
                "country": key,
                "iso3": iso3,
                **contact_diagnostics,
                "source_type": "derived",
            }
        )
        seasonality_rows.append(
            {
                "country": key,
                "iso3": iso3,
                **seasonality,
                **cycle,
                **observed_incidence,
                "source": "PertussisIncidence reported case time series",
                "source_type": "derived",
            }
        )

        trajectory_window = annual_trajectory.loc[
            annual_trajectory["year"].between(embed_start_year, embed_end_year)
        ].copy()
        demography_trajectory = {
            "source": "UN WPP 2024 one-year age-sex population, 1990-2050, aggregated to model age groups",
            "horizon_start_year": int(embed_start_year),
            "horizon_end_year": int(embed_end_year),
            "analysis_start_year": int(horizon_start_year),
            "analysis_end_year": int(horizon_end_year),
            "years": [int(y) for y in trajectory_window["year"].tolist()],
            "population_by_year": {
                age: {
                    int(row["year"]): float(row[age])
                    for _, row in trajectory_window.iterrows()
                }
                for age in AGE_GROUPS
            },
            "births_by_year": {
                int(row["year"]): float(row["births"])
                for _, row in trajectory_window.iterrows()
            },
            "total_population_by_year": {
                int(row["year"]): float(row["total_population"])
                for _, row in trajectory_window.iterrows()
            },
        }

        profiles[key] = {
            "description": (
                f"Data-derived profile for {key}; WPP {year} population, "
                f"harmonized surveillance-derived seasonality "
                f"{'using all available records' if surveillance_year is None else f'through {surveillance_year}'}, "
                "Prem/contactdata-derived contact matrix; "
                f"annual demographic trajectory embedded for {embed_start_year}-{embed_end_year}."
            ),
            "iso3": iso3,
            "population": {k: float(v) for k, v in population.items()},
            "total_population": float(total_population),
            "observed_incidence": observed_incidence,
            "vaccine_coverage": {k: float(v) for k, v in vaccination.items()},
            "demography_trajectory": demography_trajectory,
            "source_types": {
                "population": "derived",
                "observed_incidence": "derived",
                "vaccine_coverage": "measured",
                "birth_entry": "derived",
                "reporting_rate": "assumption",
                "reporting_rate_prior": "literature_prior",
                "seasonality": "derived",
                "multi_year_cycle": "derived",
                "contact_matrix": "derived",
                "vaccine_schedule": "measured",
                "demography_trajectory": "derived",
            },
            "vaccine_schedule": {
                "vaccine_product": meta.get("vaccine_product", "unknown"),
                "primary_doses": int(meta.get("primary_doses", 3)),
                "child_booster_doses": int(meta.get("child_booster_doses", 0)),
                "adolescent_booster": bool(measured_adolescent_booster),
                "maternal_program": bool(measured_maternal_program),
                "maternal_coverage": float(maternal_coverage),
                "maternal_program_note": _text_or_default(measurement["maternal_program_note"]),
                "routine_dose_count": int(measured_routine_dose_count),
                "routine_age_pattern": measured_routine_age_pattern,
                "routine_first_shot_months": float(measured_routine_first_shot),
                "routine_last_shot_months": float(measured_routine_last_shot),
                "routine_scheduler_code": measured_routine_scheduler_code,
                "schedule_note": measured_schedule_note,
                "source": _text_or_default(measured_schedule_source),
            },
            "birth_entry": {"S": float(1.0 - birth_vaccinated), "V": birth_vaccinated},
            "reporting_rate": REPORTING_DEFAULTS,
            "reporting_rate_prior": _reporting_rate_prior(key),
            "transmission_overrides": {
                "seasonal_amplitude": float(seasonality["seasonal_amplitude"]),
                "seasonal_phase": float(seasonality["seasonal_phase"]),
                "multi_year_amplitude": float(cycle["multi_year_amplitude"]),
                "multi_year_period_years": float(cycle["multi_year_period_years"]),
            },
            "contact_reciprocity": contact_diagnostics,
            "contact_source": (
                f"Prem/contactdata {meta.get('contactdata_source', 'unknown')} "
                "country matrix aggregated to project age groups with population-weighted reciprocity correction"
            ),
            "contact_matrix": contact_matrix,
        }

    incidence_out = pd.concat(incidence_frames, ignore_index=True)
    return (
        profiles,
        pd.DataFrame(population_rows),
        pd.DataFrame(annual_population_rows),
        pd.DataFrame(seasonality_rows),
        pd.DataFrame(contact_rows),
        pd.DataFrame(contact_diagnostic_rows),
        pd.DataFrame(setting_contact_rows),
        pd.DataFrame(setting_contact_diagnostic_rows),
        incidence_out,
    )


def main() -> None:
    (
        profiles,
        population,
        annual_population,
        seasonality,
        contacts,
        contact_diagnostics,
        setting_contacts,
        setting_contact_diagnostics,
        incidence,
    ) = build_profiles()
    project_path("data/processed").mkdir(parents=True, exist_ok=True)
    population.to_csv(project_path("data/processed/wpp_country_age_groups.csv"), index=False)
    annual_population.to_csv(
        project_path("data/processed/wpp_country_age_groups_annual.csv"), index=False
    )
    seasonality.to_csv(project_path("data/processed/pertussis_incidence_seasonality.csv"), index=False)
    contacts.to_csv(project_path("data/processed/country_contact_matrices_8groups.csv"), index=False)
    contact_diagnostics.to_csv(project_path("data/processed/contact_matrix_reciprocity_diagnostics.csv"), index=False)
    if not setting_contacts.empty:
        setting_contacts.to_csv(
            project_path("data/processed/country_contact_matrices_by_location_8groups.csv"),
            index=False,
        )
        setting_contact_diagnostics.to_csv(
            project_path("data/processed/contact_matrix_by_location_reciprocity_diagnostics.csv"),
            index=False,
        )
    incidence.to_csv(project_path("data/processed/pertussis_incidence_timeseries.csv"), index=False)

    with project_path("config/country_profiles.yaml").open("w", encoding="utf-8") as handle:
        yaml.dump(profiles, handle, Dumper=NoAliasDumper, sort_keys=False)


if __name__ == "__main__":
    main()
