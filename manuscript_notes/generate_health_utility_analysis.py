from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import current_run_metadata, write_run_metadata

SOURCE = ROOT / "outputs" / "summaries" / "intervention_scenarios_summary.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
WRITTEN_ROW_COUNTS: dict[str, int] = {}

MORTALITY_COLUMNS = (
    "total_deaths",
    "total_infant_deaths",
    "annualized_deaths_per_million",
    "annualized_infant_deaths_per_100k",
    "relative_reduction_deaths",
    "relative_reduction_infant_deaths",
)
HOSPITALIZATION_PATTERNS = ("hosp", "admit")

UTILITY_SCENARIOS = (
    {
        "utility_scenario": "lower_burden",
        "infant_life_years_lost_per_death": 60.0,
        "acute_illness_disutility": 0.03,
        "acute_illness_duration_days": 14.0,
        "infant_hospitalization_probability_per_case": 0.20,
        "noninfant_hospitalization_probability_per_case": 0.005,
        "hospitalization_excess_disutility": 0.10,
        "hospitalization_duration_days": 5.0,
    },
    {
        "utility_scenario": "central_exploratory",
        "infant_life_years_lost_per_death": 70.0,
        "acute_illness_disutility": 0.05,
        "acute_illness_duration_days": 21.0,
        "infant_hospitalization_probability_per_case": 0.40,
        "noninfant_hospitalization_probability_per_case": 0.010,
        "hospitalization_excess_disutility": 0.20,
        "hospitalization_duration_days": 7.0,
    },
    {
        "utility_scenario": "higher_burden",
        "infant_life_years_lost_per_death": 80.0,
        "acute_illness_disutility": 0.10,
        "acute_illness_duration_days": 28.0,
        "infant_hospitalization_probability_per_case": 0.60,
        "noninfant_hospitalization_probability_per_case": 0.030,
        "hospitalization_excess_disutility": 0.30,
        "hospitalization_duration_days": 10.0,
    },
)

METHOD_NOTE = (
    "Exploratory health-utility burden only: infant mortality is counted as undiscounted "
    "life-years lost, acute illness uses symptomatic cases, and hospitalizations are "
    "imputed from cases because the model summary has no hospitalization/admission columns. "
    "No costs, discounting, age weighting, or formal cost-effectiveness decision rule is applied."
)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    WRITTEN_ROW_COUNTS[path.relative_to(ROOT).as_posix()] = int(len(df))
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def _detect_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    lower = {column: column.lower() for column in columns}
    mortality = [column for column in MORTALITY_COLUMNS if column in columns]
    hospitalization = [
        column
        for column, lowered in lower.items()
        if any(pattern in lowered for pattern in HOSPITALIZATION_PATTERNS)
    ]
    return mortality, hospitalization


def _required_summary_columns() -> list[str]:
    return [
        "country",
        "scenario",
        "intervention",
        "analysis_years",
        "total_population",
        "total_symptomatic_cases",
        "total_infant_cases",
        "total_deaths",
        "total_infant_deaths",
    ]


def load_summary() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing intervention summary: {SOURCE}")
    summary = pd.read_csv(SOURCE)
    missing = [column for column in _required_summary_columns() if column not in summary.columns]
    if missing:
        raise ValueError(f"Missing required intervention summary columns: {missing}")
    return summary


def assumption_table(summary_columns: list[str]) -> pd.DataFrame:
    mortality_columns, hospitalization_columns = _detect_columns(summary_columns)
    assumptions = pd.DataFrame(UTILITY_SCENARIOS)
    assumptions["model_mortality_columns_found"] = ", ".join(mortality_columns)
    assumptions["model_hospitalization_columns_found"] = ", ".join(hospitalization_columns) or "none"
    assumptions["method_note"] = METHOD_NOTE
    return assumptions


def detailed_loss_table(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[:, _required_summary_columns()].copy()
    numeric_cols = [
        "analysis_years",
        "total_population",
        "total_symptomatic_cases",
        "total_infant_cases",
        "total_deaths",
        "total_infant_deaths",
    ]
    for column in numeric_cols:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base["total_noninfant_cases"] = (base["total_symptomatic_cases"] - base["total_infant_cases"]).clip(lower=0)

    rows: list[pd.DataFrame] = []
    for assumptions in UTILITY_SCENARIOS:
        frame = base.copy()
        for key, value in assumptions.items():
            frame[key] = value

        frame["infant_life_years_lost"] = (
            frame["total_infant_deaths"] * assumptions["infant_life_years_lost_per_death"]
        )
        frame["acute_illness_qaly_loss"] = (
            frame["total_symptomatic_cases"]
            * assumptions["acute_illness_disutility"]
            * assumptions["acute_illness_duration_days"]
            / 365.25
        )
        frame["estimated_infant_hospitalizations"] = (
            frame["total_infant_cases"] * assumptions["infant_hospitalization_probability_per_case"]
        )
        frame["estimated_noninfant_hospitalizations"] = (
            frame["total_noninfant_cases"] * assumptions["noninfant_hospitalization_probability_per_case"]
        )
        frame["estimated_total_hospitalizations"] = (
            frame["estimated_infant_hospitalizations"] + frame["estimated_noninfant_hospitalizations"]
        )
        frame["hospitalization_qaly_loss"] = (
            frame["estimated_total_hospitalizations"]
            * assumptions["hospitalization_excess_disutility"]
            * assumptions["hospitalization_duration_days"]
            / 365.25
        )
        frame["total_qaly_like_loss"] = (
            frame["infant_life_years_lost"]
            + frame["acute_illness_qaly_loss"]
            + frame["hospitalization_qaly_loss"]
        )
        denominator = frame["analysis_years"] * frame["total_population"].replace(0, np.nan)
        frame["total_qaly_like_loss_per_100k_year"] = frame["total_qaly_like_loss"] / denominator * 100_000.0
        for component in ("infant_life_years_lost", "acute_illness_qaly_loss", "hospitalization_qaly_loss"):
            frame[f"{component}_share"] = component_divide(frame[component], frame["total_qaly_like_loss"])
        rows.append(frame)

    detailed = pd.concat(rows, ignore_index=True)
    current = detailed.loc[
        detailed["scenario"].eq("current"),
        ["country", "utility_scenario", "total_qaly_like_loss", "total_qaly_like_loss_per_100k_year"],
    ].rename(
        columns={
            "total_qaly_like_loss": "current_total_qaly_like_loss",
            "total_qaly_like_loss_per_100k_year": "current_total_qaly_like_loss_per_100k_year",
        }
    )
    detailed = detailed.merge(current, on=["country", "utility_scenario"], how="left")
    detailed["qaly_like_loss_averted_vs_current"] = (
        detailed["current_total_qaly_like_loss"] - detailed["total_qaly_like_loss"]
    )
    detailed["qaly_like_loss_averted_vs_current_per_100k_year"] = (
        detailed["current_total_qaly_like_loss_per_100k_year"]
        - detailed["total_qaly_like_loss_per_100k_year"]
    )
    detailed["relative_qaly_like_loss_reduction_vs_current"] = 1.0 - (
        detailed["total_qaly_like_loss"] / detailed["current_total_qaly_like_loss"].replace(0, np.nan)
    )
    detailed["method_note"] = METHOD_NOTE

    ordered = [
        "country",
        "scenario",
        "intervention",
        "utility_scenario",
        "total_qaly_like_loss",
        "total_qaly_like_loss_per_100k_year",
        "qaly_like_loss_averted_vs_current",
        "qaly_like_loss_averted_vs_current_per_100k_year",
        "relative_qaly_like_loss_reduction_vs_current",
        "infant_life_years_lost",
        "acute_illness_qaly_loss",
        "hospitalization_qaly_loss",
        "infant_life_years_lost_share",
        "acute_illness_qaly_loss_share",
        "hospitalization_qaly_loss_share",
        "estimated_total_hospitalizations",
        "estimated_infant_hospitalizations",
        "estimated_noninfant_hospitalizations",
        "total_symptomatic_cases",
        "total_infant_cases",
        "total_deaths",
        "total_infant_deaths",
        "infant_life_years_lost_per_death",
        "acute_illness_disutility",
        "acute_illness_duration_days",
        "infant_hospitalization_probability_per_case",
        "noninfant_hospitalization_probability_per_case",
        "hospitalization_excess_disutility",
        "hospitalization_duration_days",
        "method_note",
    ]
    return detailed.loc[:, ordered].sort_values(["utility_scenario", "country", "scenario"])


def component_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _q25(values: pd.Series) -> float:
    return float(np.nanquantile(values.to_numpy(dtype=float), 0.25))


def _q75(values: pd.Series) -> float:
    return float(np.nanquantile(values.to_numpy(dtype=float), 0.75))


def summary_loss_table(detailed: pd.DataFrame) -> pd.DataFrame:
    grouped = detailed.groupby(["utility_scenario", "scenario"], as_index=False)
    summary = grouped.agg(
        country_count=("country", "nunique"),
        median_total_qaly_like_loss_per_100k_year=("total_qaly_like_loss_per_100k_year", "median"),
        q25_total_qaly_like_loss_per_100k_year=("total_qaly_like_loss_per_100k_year", _q25),
        q75_total_qaly_like_loss_per_100k_year=("total_qaly_like_loss_per_100k_year", _q75),
        median_qaly_like_loss_averted_vs_current_per_100k_year=(
            "qaly_like_loss_averted_vs_current_per_100k_year",
            "median",
        ),
        q25_qaly_like_loss_averted_vs_current_per_100k_year=(
            "qaly_like_loss_averted_vs_current_per_100k_year",
            _q25,
        ),
        q75_qaly_like_loss_averted_vs_current_per_100k_year=(
            "qaly_like_loss_averted_vs_current_per_100k_year",
            _q75,
        ),
        median_relative_qaly_like_loss_reduction_vs_current=(
            "relative_qaly_like_loss_reduction_vs_current",
            "median",
        ),
        median_infant_mortality_life_year_share=("infant_life_years_lost_share", "median"),
        median_acute_illness_share=("acute_illness_qaly_loss_share", "median"),
        median_hospitalization_share=("hospitalization_qaly_loss_share", "median"),
    )
    assumptions = pd.DataFrame(UTILITY_SCENARIOS)
    summary = summary.merge(assumptions, on="utility_scenario", how="left")
    summary["interpretation_note"] = METHOD_NOTE
    return summary.sort_values(["utility_scenario", "median_qaly_like_loss_averted_vs_current_per_100k_year"])


def main() -> None:
    summary = load_summary()
    assumptions = assumption_table(list(summary.columns))
    detailed = detailed_loss_table(summary)
    compact = summary_loss_table(detailed)

    _write(assumptions, TABLE_DIR / "intervention_health_utility_assumptions.csv")
    _write(detailed, TABLE_DIR / "intervention_health_utility_loss.csv")
    _write(compact, TABLE_DIR / "intervention_health_utility_loss_summary.csv")
    metadata = current_run_metadata("health_utility_analysis", row_counts=WRITTEN_ROW_COUNTS)
    metadata.update(
        {
            "source": SOURCE.relative_to(ROOT).as_posix(),
            "utility_scenarios": [scenario["utility_scenario"] for scenario in UTILITY_SCENARIOS],
        }
    )
    write_run_metadata("health_utility_analysis", metadata)

    _, hospitalization_columns = _detect_columns(list(summary.columns))
    print(
        "Wrote exploratory health-utility outputs; "
        f"hospitalization columns found: {', '.join(hospitalization_columns) or 'none'}."
    )


if __name__ == "__main__":
    main()
