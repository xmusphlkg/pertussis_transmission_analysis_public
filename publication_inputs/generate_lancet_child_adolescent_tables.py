from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import current_run_metadata, write_run_metadata
from src_python.utils.io import project_path, read_table, write_dataframe


INFANT_AGE_GROUPS = ("infant_0_2m", "infant_3_11m")
CHILD_1_9_AGE_GROUPS = ("child_1_4y", "child_5_9y")
ADOLESCENT_AGE_GROUPS = ("adolescent_10_17y",)
CHILD_ADOLESCENT_AGE_GROUPS = (*INFANT_AGE_GROUPS, *CHILD_1_9_AGE_GROUPS, *ADOLESCENT_AGE_GROUPS)

SELECTED_STRATEGIES = (
    "current",
    "higher_child_coverage",
    "timeliness_only",
    "adolescent_booster",
    "pregnancy_tdap_scaleup",
    "cocooning_adjunct",
    "maternal_immunization",
    "targeted_pep_high_risk",
    "resistance_guided_treatment",
    "transmission_blocking_vaccine",
    "next_generation_vaccine",
    "combined_strategy",
)

PROGRAM_ONLY_STRATEGIES = (
    "current",
    "higher_child_coverage",
    "timeliness_only",
    "adolescent_booster",
    "pregnancy_tdap_scaleup",
    "cocooning_adjunct",
    "maternal_immunization",
    "targeted_pep_high_risk",
)

PROGRAM_PLUS_RESISTANCE_STRATEGIES = (*PROGRAM_ONLY_STRATEGIES, "resistance_guided_treatment")
FUTURE_PRODUCT_TARGET_STRATEGIES = (
    *PROGRAM_PLUS_RESISTANCE_STRATEGIES,
    "transmission_blocking_vaccine",
    "next_generation_vaccine",
    "combined_strategy",
)

CONSTRAINT_STRATEGIES = {
    "program_only": PROGRAM_ONLY_STRATEGIES,
    "program_plus_resistance": PROGRAM_PLUS_RESISTANCE_STRATEGIES,
    "future_product_target": FUTURE_PRODUCT_TARGET_STRATEGIES,
}

STRATEGY_LABELS = {
    "current": "Current practice",
    "higher_child_coverage": "Nominal coverage floor",
    "timeliness_only": "Routine timeliness",
    "adolescent_booster": "Adolescent booster",
    "pregnancy_tdap_scaleup": "Pregnancy Tdap scale-up",
    "cocooning_adjunct": "Close-contact adult adjunct",
    "maternal_immunization": "Infant-exposure reduction composite",
    "targeted_pep_high_risk": "Targeted high-risk PEP",
    "resistance_guided_treatment": "Resistance-guided management",
    "transmission_blocking_vaccine": "Transmission-blocking vaccine target",
    "next_generation_vaccine": "High-transmission-blocking vaccine target",
    "combined_strategy": "Combined future stress-test profile",
}

LANCET_STRATEGY_ROLE = {
    "current": "Comparator",
    "higher_child_coverage": "Routine child-immunisation delivery",
    "timeliness_only": "Routine child-immunisation delivery",
    "adolescent_booster": "Adolescent transmission lever",
    "pregnancy_tdap_scaleup": "Direct infant protection within child-health frame",
    "cocooning_adjunct": "Household and close-contact exposure lever",
    "maternal_immunization": "Infant exposure and maternal-child protection composite",
    "targeted_pep_high_risk": "Clinical and outbreak management lever",
    "resistance_guided_treatment": "Resistance-aware clinical management lever",
    "transmission_blocking_vaccine": "Future child/adolescent transmission-blocking target",
    "next_generation_vaccine": "Future child/adolescent high-blocking target",
    "combined_strategy": "Exploratory combined stress test",
}

IMPLEMENTATION_INTENSITY = {
    "current": 0,
    "higher_child_coverage": 1,
    "timeliness_only": 1,
    "adolescent_booster": 1,
    "pregnancy_tdap_scaleup": 1,
    "targeted_pep_high_risk": 1,
    "cocooning_adjunct": 2,
    "resistance_guided_treatment": 2,
    "maternal_immunization": 3,
    "transmission_blocking_vaccine": 4,
    "next_generation_vaccine": 4,
    "combined_strategy": 5,
}

PRIMARY_RATE = "annualized_child_adolescent_cases_per_100k"
PRIMARY_TOTAL = "total_child_adolescent_cases"
PRIMARY_REDUCTION = "relative_reduction_child_adolescent_cases"
FALLBACK_RATE = "annualized_infant_cases_per_100k"
FALLBACK_TOTAL = "total_infant_cases"
FALLBACK_REDUCTION = "relative_reduction_infant_cases"

TIMESERIES_DERIVED_COLUMNS = (
    "country",
    "scenario",
    "age_group",
    "time",
    "population",
    "symptomatic_cases",
    "total_infections",
    "reported_cases",
    "deaths",
    "hospitalizations",
)


WRITTEN_ROW_COUNTS: dict[str, int] = {}


def _write(df: pd.DataFrame, relative_path: str) -> None:
    path = project_path(relative_path)
    write_dataframe(df, path)
    WRITTEN_ROW_COUNTS[relative_path] = int(len(df))


def _read(relative_path: str) -> pd.DataFrame:
    return read_table(project_path(relative_path))


def _available(data: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(column in data.columns for column in columns)


def _safe_numeric(series: pd.Series | float | int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _weighted_quantile(values: pd.Series, weights: pd.Series, probability: float) -> float:
    data = pd.DataFrame({"value": _safe_numeric(values), "weight": _safe_numeric(weights)})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data["weight"].gt(0.0)]
    if data.empty:
        return float("nan")
    data = data.sort_values("value")
    cumulative = data["weight"].cumsum().to_numpy(dtype=float)
    total = float(cumulative[-1])
    if total <= 0.0:
        return float("nan")
    return float(np.interp(probability * total, cumulative, data["value"].to_numpy(dtype=float)))


def _iqr_text(values: pd.Series, weights: pd.Series | None = None) -> str:
    if weights is None:
        values = _safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return ""
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
    else:
        q25 = _weighted_quantile(values, weights, 0.25)
        q75 = _weighted_quantile(values, weights, 0.75)
    if not np.isfinite(q25) or not np.isfinite(q75):
        return ""
    return f"{q25:.4g} to {q75:.4g}"


def _merge_metric(out: pd.DataFrame, metrics: pd.DataFrame, column: str) -> pd.DataFrame:
    derived_col = f"{column}_derived"
    if derived_col not in out.columns:
        return out
    if column not in out.columns:
        out[column] = np.nan
    current = _safe_numeric(out[column])
    derived = _safe_numeric(out[derived_col])
    fill_mask = current.isna() & derived.notna()
    out.loc[fill_mask, column] = derived.loc[fill_mask]
    return out.drop(columns=[derived_col])


def _sum_for_age_groups(data: pd.DataFrame, age_groups: tuple[str, ...], value_col: str, out_col: str) -> pd.DataFrame:
    keys = ["country", "scenario"]
    if value_col not in data.columns:
        return pd.DataFrame(columns=keys + [out_col])
    return (
        data.loc[data["age_group"].isin(age_groups)]
        .groupby(keys, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: out_col})
    )


def _population_for_age_groups(data: pd.DataFrame, age_groups: tuple[str, ...], out_col: str) -> pd.DataFrame:
    keys = ["country", "scenario"]
    return (
        data.loc[data["age_group"].isin(age_groups)]
        .groupby(keys + ["age_group"], as_index=False)["population"]
        .mean()
        .groupby(keys, as_index=False)["population"]
        .sum()
        .rename(columns={"population": out_col})
    )


def _derive_pediatric_metrics_from_timeseries(stem: str) -> pd.DataFrame:
    path = project_path("outputs", "simulations", f"{stem}.parquet")
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_parquet(path, columns=list(TIMESERIES_DERIVED_COLUMNS))
    data = data.loc[data["country"].notna() & data["scenario"].notna()].copy()
    if data.empty:
        return pd.DataFrame()

    keys = ["country", "scenario"]
    metrics = (
        data.groupby(keys, as_index=False)["time"]
        .agg(time_min="min", time_max="max")
        .assign(analysis_years_from_timeseries=lambda x: (x["time_max"] - x["time_min"]) / 365.0)
        .drop(columns=["time_min", "time_max"])
    )
    metrics["analysis_years_from_timeseries"] = metrics["analysis_years_from_timeseries"].clip(lower=1.0 / 365.0)

    parts = [
        _sum_for_age_groups(data, CHILD_1_9_AGE_GROUPS, "symptomatic_cases", "total_child_1_9_cases"),
        _sum_for_age_groups(data, ADOLESCENT_AGE_GROUPS, "symptomatic_cases", "total_adolescent_cases"),
        _sum_for_age_groups(data, CHILD_ADOLESCENT_AGE_GROUPS, "symptomatic_cases", "total_child_adolescent_cases"),
        _sum_for_age_groups(
            data,
            CHILD_ADOLESCENT_AGE_GROUPS,
            "total_infections",
            "total_child_adolescent_infections",
        ),
        _sum_for_age_groups(
            data,
            CHILD_ADOLESCENT_AGE_GROUPS,
            "reported_cases",
            "total_child_adolescent_reported_cases",
        ),
        _sum_for_age_groups(data, INFANT_AGE_GROUPS, "deaths", "total_infant_deaths"),
        _sum_for_age_groups(data, INFANT_AGE_GROUPS, "hospitalizations", "total_infant_hospitalizations"),
        _sum_for_age_groups(data, CHILD_ADOLESCENT_AGE_GROUPS, "deaths", "total_child_adolescent_deaths"),
        _sum_for_age_groups(
            data,
            CHILD_ADOLESCENT_AGE_GROUPS,
            "hospitalizations",
            "total_child_adolescent_hospitalizations",
        ),
        _population_for_age_groups(data, CHILD_1_9_AGE_GROUPS, "child_1_9_population"),
        _population_for_age_groups(data, ADOLESCENT_AGE_GROUPS, "adolescent_population"),
        _population_for_age_groups(data, CHILD_ADOLESCENT_AGE_GROUPS, "child_adolescent_population"),
    ]
    for part in parts:
        metrics = metrics.merge(part, on=keys, how="left")
    return metrics


def _fill_rate(
    out: pd.DataFrame,
    *,
    total_col: str,
    population_col: str,
    rate_col: str,
    scale: float = 100_000.0,
) -> None:
    if total_col not in out.columns or population_col not in out.columns:
        return
    years = _safe_numeric(out.get("analysis_years", out.get("analysis_years_from_timeseries")))
    total = _safe_numeric(out[total_col])
    population = _safe_numeric(out[population_col]).replace(0, np.nan)
    derived = total / (years * population).replace(0, np.nan) * scale
    if rate_col not in out.columns:
        out[rate_col] = np.nan
    current = _safe_numeric(out[rate_col])
    mask = current.isna() & derived.notna()
    out.loc[mask, rate_col] = derived.loc[mask]


def _add_relative_reduction(
    out: pd.DataFrame,
    *,
    reference_scenario: str,
    source_col: str,
    reduction_col: str,
) -> None:
    if source_col not in out.columns:
        return
    reference = (
        out.loc[out["scenario"].eq(reference_scenario), ["country", source_col]]
        .drop_duplicates("country")
        .rename(columns={source_col: f"{source_col}_reference"})
    )
    if reference.empty:
        return
    merged = out[["country"]].merge(reference, on="country", how="left")
    denom = _safe_numeric(merged[f"{source_col}_reference"]).replace(0, np.nan)
    values = _safe_numeric(out[source_col])
    derived = 1.0 - values / denom
    if reduction_col not in out.columns:
        out[reduction_col] = np.nan
    current = _safe_numeric(out[reduction_col])
    mask = current.isna() & derived.notna()
    out.loc[mask, reduction_col] = derived.loc[mask]


def _augment_with_pediatric_metrics(summary: pd.DataFrame, *, stem: str, reference_scenario: str) -> pd.DataFrame:
    metrics = _derive_pediatric_metrics_from_timeseries(stem)
    if metrics.empty:
        return summary
    out = summary.merge(metrics, on=["country", "scenario"], how="left", suffixes=("", "_derived"))
    for column in metrics.columns:
        if column in {"country", "scenario"}:
            continue
        out = _merge_metric(out, metrics, column)
    if "analysis_years" not in out.columns:
        out["analysis_years"] = out.get("analysis_years_from_timeseries", np.nan)
    _fill_rate(
        out,
        total_col="total_child_1_9_cases",
        population_col="child_1_9_population",
        rate_col="annualized_child_1_9_cases_per_100k",
    )
    _fill_rate(
        out,
        total_col="total_adolescent_cases",
        population_col="adolescent_population",
        rate_col="annualized_adolescent_cases_per_100k",
    )
    _fill_rate(
        out,
        total_col="total_child_adolescent_cases",
        population_col="child_adolescent_population",
        rate_col=PRIMARY_RATE,
    )
    _fill_rate(
        out,
        total_col="total_child_adolescent_infections",
        population_col="child_adolescent_population",
        rate_col="annualized_child_adolescent_infections_per_100k",
    )
    _fill_rate(
        out,
        total_col="total_child_adolescent_reported_cases",
        population_col="child_adolescent_population",
        rate_col="annualized_child_adolescent_reported_cases_per_100k",
    )
    _fill_rate(
        out,
        total_col="total_infant_deaths",
        population_col="infant_population",
        rate_col="annualized_infant_deaths_per_100k",
    )
    _fill_rate(
        out,
        total_col="total_infant_hospitalizations",
        population_col="infant_population",
        rate_col="annualized_infant_hospitalizations_per_100k",
    )
    _fill_rate(
        out,
        total_col="total_child_adolescent_deaths",
        population_col="child_adolescent_population",
        rate_col="annualized_child_adolescent_deaths_per_million",
        scale=1_000_000.0,
    )
    _fill_rate(
        out,
        total_col="total_child_adolescent_hospitalizations",
        population_col="child_adolescent_population",
        rate_col="annualized_child_adolescent_hospitalizations_per_100k",
    )
    for reduction_col, source_col in {
        "relative_reduction_child_1_9_cases": "total_child_1_9_cases",
        "relative_reduction_adolescent_cases": "total_adolescent_cases",
        PRIMARY_REDUCTION: PRIMARY_TOTAL,
        "relative_reduction_child_adolescent_infections": "total_child_adolescent_infections",
        "relative_reduction_child_adolescent_reported_cases": "total_child_adolescent_reported_cases",
        "relative_reduction_infant_deaths": "total_infant_deaths",
        "relative_reduction_infant_hospitalizations": "total_infant_hospitalizations",
        "relative_reduction_child_adolescent_deaths": "total_child_adolescent_deaths",
    }.items():
        _add_relative_reduction(
            out,
            reference_scenario=reference_scenario,
            source_col=source_col,
            reduction_col=reduction_col,
        )
    return out


def _metric_availability(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required = [
        PRIMARY_RATE,
        PRIMARY_TOTAL,
        PRIMARY_REDUCTION,
        "child_adolescent_population",
        "annualized_child_1_9_cases_per_100k",
        "annualized_adolescent_cases_per_100k",
        FALLBACK_RATE,
        FALLBACK_TOTAL,
        FALLBACK_REDUCTION,
    ]
    rows = []
    for source, frame in frames.items():
        for column in required:
            rows.append(
                {
                    "source_table": source,
                    "column": column,
                    "available": column in frame.columns,
                    "non_missing_values": int(frame[column].notna().sum()) if column in frame.columns else 0,
                }
            )
    return pd.DataFrame(rows)


def _resistant_infections_per_100k(df: pd.DataFrame) -> pd.Series:
    denominator = pd.to_numeric(df.get("analysis_years"), errors="coerce").replace(0, np.nan) * pd.to_numeric(
        df.get("total_population"), errors="coerce"
    ).replace(0, np.nan)
    return pd.to_numeric(df.get("resistant_infections"), errors="coerce") / denominator * 100_000.0


def _load_intervention_rows(
    intervention: pd.DataFrame,
    timeliness: pd.DataFrame,
    vaccine: pd.DataFrame,
) -> pd.DataFrame:
    intervention = intervention.loc[
        intervention["scenario"].isin(
            [
                "current",
                "higher_child_coverage",
                "adolescent_booster",
                "pregnancy_tdap_scaleup",
                "cocooning_adjunct",
                "maternal_immunization",
                "targeted_pep_high_risk",
                "resistance_guided_treatment",
                "next_generation_vaccine",
                "combined_strategy",
            ]
        )
    ].copy()

    timeliness = timeliness.loc[timeliness["strategy"].eq("timeliness_only")].copy()
    timeliness["scenario"] = "timeliness_only"
    timeliness["intervention"] = "timeliness_only"

    current = vaccine.loc[vaccine["scenario"].eq("symptom_protective")].copy()
    current = current.rename(
        columns={
            FALLBACK_RATE: "current_infant_cases_per_100k",
            "annualized_infections_per_100k": "current_infections_per_100k",
            "annualized_reported_cases_per_100k": "current_reported_cases_per_100k",
            "resistant_infections": "current_resistant_infections",
        }
    )
    current_cols = [
        "country",
        "current_infant_cases_per_100k",
        "current_infections_per_100k",
        "current_reported_cases_per_100k",
        "current_resistant_infections",
    ]
    if PRIMARY_RATE in vaccine.columns:
        current = current.rename(columns={PRIMARY_RATE: "current_child_adolescent_cases_per_100k"})
        current_cols.append("current_child_adolescent_cases_per_100k")
    transmission = vaccine.loc[vaccine["scenario"].eq("transmission_blocking")].copy()
    transmission = transmission.merge(current[current_cols], on="country", how="left")
    transmission["scenario"] = "transmission_blocking_vaccine"
    transmission["strategy"] = "transmission_blocking_vaccine"
    transmission["intervention"] = "transmission_blocking_vaccine"
    if PRIMARY_RATE in transmission.columns and "current_child_adolescent_cases_per_100k" in transmission.columns:
        transmission[PRIMARY_REDUCTION] = 1.0 - transmission[PRIMARY_RATE] / transmission[
            "current_child_adolescent_cases_per_100k"
        ].replace(0, np.nan)
    if FALLBACK_RATE in transmission.columns:
        transmission[FALLBACK_REDUCTION] = 1.0 - transmission[FALLBACK_RATE] / transmission[
            "current_infant_cases_per_100k"
        ].replace(0, np.nan)
    transmission["relative_reduction_total_infections"] = 1.0 - transmission["annualized_infections_per_100k"] / transmission[
        "current_infections_per_100k"
    ].replace(0, np.nan)
    transmission["relative_reduction_reported_cases"] = 1.0 - transmission[
        "annualized_reported_cases_per_100k"
    ] / transmission["current_reported_cases_per_100k"].replace(0, np.nan)
    transmission["relative_reduction_resistant_infections"] = 1.0 - transmission["resistant_infections"] / transmission[
        "current_resistant_infections"
    ].replace(0, np.nan)
    transmission = transmission.drop(columns=[col for col in transmission.columns if col.startswith("current_")])

    combined = pd.concat([intervention, timeliness, transmission], ignore_index=True, sort=False)
    combined["strategy"] = combined["scenario"]
    return combined.loc[combined["strategy"].isin(SELECTED_STRATEGIES)].copy()


def _select_primary_metric(data: pd.DataFrame) -> tuple[str, str, str, str]:
    has_child_adolescent = _available(data, [PRIMARY_RATE, PRIMARY_TOTAL]) and data[PRIMARY_RATE].notna().any()
    if has_child_adolescent:
        return PRIMARY_RATE, PRIMARY_TOTAL, PRIMARY_REDUCTION, "child_adolescent_0_17y"
    return FALLBACK_RATE, FALLBACK_TOTAL, FALLBACK_REDUCTION, "infant_fallback_requires_simulation_rerun"


def _strategy_burden_frame(data: pd.DataFrame) -> pd.DataFrame:
    rate_col, total_col, reduction_col, metric_basis = _select_primary_metric(data)
    out = data.copy()
    out["strategy_label"] = out["strategy"].map(STRATEGY_LABELS).fillna(out["strategy"])
    out["lancet_strategy_role"] = out["strategy"].map(LANCET_STRATEGY_ROLE).fillna("")
    out["implementation_intensity"] = out["strategy"].map(IMPLEMENTATION_INTENSITY).astype(float)
    out["primary_case_metric"] = metric_basis
    out["primary_cases_per_100k"] = pd.to_numeric(out[rate_col], errors="coerce")
    out["primary_total_cases"] = pd.to_numeric(out[total_col], errors="coerce")
    out["primary_case_reduction"] = pd.to_numeric(out.get(reduction_col), errors="coerce")
    current = out.loc[out["strategy"].eq("current"), ["country", "primary_cases_per_100k", "primary_total_cases"]].rename(
        columns={
            "primary_cases_per_100k": "current_primary_cases_per_100k",
            "primary_total_cases": "current_primary_total_cases",
        }
    )
    out = out.merge(current, on="country", how="left")
    missing_reduction = out["primary_case_reduction"].isna()
    out.loc[missing_reduction, "primary_case_reduction"] = 1.0 - out.loc[
        missing_reduction, "primary_cases_per_100k"
    ] / out.loc[missing_reduction, "current_primary_cases_per_100k"].replace(0, np.nan)
    out["annualized_resistant_infections_per_100k"] = _resistant_infections_per_100k(out)
    out["metric_availability_note"] = np.where(
        metric_basis == "child_adolescent_0_17y",
        "Primary Lancet metric generated from model age groups <18 years.",
        "Legacy fallback: child/adolescent <18-year metrics are unavailable; rerun current simulations.",
    )
    if "relative_reduction_resistant_infections" not in out:
        out["relative_reduction_resistant_infections"] = np.nan
    for source_col, out_col in [
        (FALLBACK_RATE, "infant_cases_per_100k"),
        ("annualized_infant_hospitalizations_per_100k", "infant_hospitalizations_per_100k"),
        ("annualized_infant_deaths_per_100k", "infant_deaths_per_100k"),
        ("annualized_child_1_9_cases_per_100k", "child_1_9_cases_per_100k"),
        ("annualized_adolescent_cases_per_100k", "adolescent_cases_per_100k"),
        ("annualized_child_adolescent_infections_per_100k", "child_adolescent_infections_per_100k"),
        ("annualized_child_adolescent_reported_cases_per_100k", "child_adolescent_reported_cases_per_100k"),
        ("annualized_child_adolescent_deaths_per_million", "child_adolescent_deaths_per_million"),
        (
            "annualized_child_adolescent_hospitalizations_per_100k",
            "child_adolescent_hospitalizations_per_100k",
        ),
    ]:
        out[out_col] = _safe_numeric(out[source_col]) if source_col in out.columns else np.nan
    keep = [
        "country",
        "strategy",
        "strategy_label",
        "lancet_strategy_role",
        "implementation_intensity",
        "primary_case_metric",
        "primary_cases_per_100k",
        "primary_total_cases",
        "current_primary_cases_per_100k",
        "primary_case_reduction",
        "infant_cases_per_100k",
        "infant_hospitalizations_per_100k",
        "infant_deaths_per_100k",
        "child_1_9_cases_per_100k",
        "adolescent_cases_per_100k",
        "child_adolescent_infections_per_100k",
        "child_adolescent_reported_cases_per_100k",
        "child_adolescent_deaths_per_million",
        "child_adolescent_hospitalizations_per_100k",
        "relative_reduction_child_1_9_cases",
        "relative_reduction_adolescent_cases",
        "relative_reduction_infant_deaths",
        "relative_reduction_infant_hospitalizations",
        "relative_reduction_child_adolescent_infections",
        "relative_reduction_child_adolescent_reported_cases",
        "annualized_resistant_infections_per_100k",
        "relative_reduction_resistant_infections",
        "relative_reduction_total_infections",
        "relative_reduction_reported_cases",
        "metric_availability_note",
    ]
    return out.loc[:, [column for column in keep if column in out.columns]].sort_values(
        ["country", "implementation_intensity", "strategy"]
    )


def _is_non_dominated(frame: pd.DataFrame) -> pd.Series:
    benefits = frame[["primary_case_reduction", "relative_reduction_resistant_infections"]].fillna(-np.inf)
    intensities = pd.to_numeric(frame["implementation_intensity"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    values = benefits.to_numpy(dtype=float)
    flags = []
    for i, row in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            at_least_as_good = np.all(other >= row - 1e-12) and intensities[j] <= intensities[i] + 1e-12
            strictly_better = np.any(other > row + 1e-12) or intensities[j] < intensities[i] - 1e-12
            if at_least_as_good and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=frame.index)


def _frontier_and_preferred(burden: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    preferred_rows = []
    for constraint, strategies in CONSTRAINT_STRATEGIES.items():
        frame = burden.loc[burden["strategy"].isin(strategies)].copy()
        frame["optimization_constraint"] = constraint
        for country, idx in frame.groupby("country").groups.items():
            country_frame = frame.loc[idx].copy()
            frame.loc[idx, "non_dominated_lancet_outcome"] = _is_non_dominated(country_frame)
            frame.loc[idx, "primary_case_rank_within_constraint"] = country_frame["primary_cases_per_100k"].rank(
                method="min",
                ascending=True,
            )
            best = country_frame.sort_values(["primary_cases_per_100k", "implementation_intensity"]).iloc[0]
            preferred_rows.append(
                {
                    "country": country,
                    "optimization_constraint": constraint,
                    "preferred_strategy": best["strategy"],
                    "preferred_strategy_label": best["strategy_label"],
                    "preferred_primary_cases_per_100k": best["primary_cases_per_100k"],
                    "preferred_primary_case_reduction": best["primary_case_reduction"],
                    "primary_case_metric": best["primary_case_metric"],
                    "metric_availability_note": best["metric_availability_note"],
                }
            )
        frames.append(frame)
    frontier = pd.concat(frames, ignore_index=True)
    preferred = pd.DataFrame(preferred_rows).sort_values(["country", "optimization_constraint"])
    return frontier.sort_values(["optimization_constraint", "country", "implementation_intensity", "strategy"]), preferred


def _strategy_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (constraint, strategy), group in frontier.groupby(["optimization_constraint", "strategy"], sort=False):
        rows.append(
            {
                "optimization_constraint": constraint,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "lancet_strategy_role": LANCET_STRATEGY_ROLE.get(strategy, ""),
                "primary_case_metric": group["primary_case_metric"].iloc[0],
                "countries_ranked_first_for_primary_cases": int(
                    np.sum(group["primary_case_rank_within_constraint"].to_numpy(dtype=float) == 1)
                ),
                "countries_non_dominated": int(group["non_dominated_lancet_outcome"].sum()),
                "median_primary_case_reduction": float(pd.to_numeric(group["primary_case_reduction"], errors="coerce").median()),
                "iqr_primary_case_reduction": (
                    f"{pd.to_numeric(group['primary_case_reduction'], errors='coerce').quantile(0.25):.4g} to "
                    f"{pd.to_numeric(group['primary_case_reduction'], errors='coerce').quantile(0.75):.4g}"
                ),
                "median_primary_cases_per_100k": float(pd.to_numeric(group["primary_cases_per_100k"], errors="coerce").median()),
                "metric_availability_note": group["metric_availability_note"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["optimization_constraint", "countries_ranked_first_for_primary_cases", "median_primary_case_reduction"],
        ascending=[True, False, False],
    )


def _logic_blueprint(metric_basis: str) -> pd.DataFrame:
    metric_text = (
        "children and adolescents younger than 18 years"
        if metric_basis == "child_adolescent_0_17y"
        else "infants under a legacy compatibility fallback"
    )
    rows = [
        {
            "manuscript_section": "Title",
            "recommended_logic": "Name the population, not only the intervention: pertussis control for children and adolescents after resurgence.",
            "implementation_status": "ready_for_draft_revision",
        },
        {
            "manuscript_section": "Objective",
            "recommended_logic": f"Compare profile-dependent strategies for reducing modeled pertussis cases in {metric_text}.",
            "implementation_status": "requires_simulation_rerun" if metric_basis != "child_adolescent_0_17y" else "ready_for_results",
        },
        {
            "manuscript_section": "Primary outcome",
            "recommended_logic": "Use annualized modeled symptomatic cases in model age groups <18 years; report infant, children, and adolescent endpoints as prespecified secondary strata.",
            "implementation_status": "core_code_updated",
        },
        {
            "manuscript_section": "Calibration statement",
            "recommended_logic": "Move infant-only calibration caveat out of the headline limitation; state that age-stratified outputs are conditional and age-specific public targets are used for validation/triangulation where available.",
            "implementation_status": "draft_revision_needed",
        },
        {
            "manuscript_section": "Results hierarchy",
            "recommended_logic": "Lead with pediatric strategy ranking and age-stratum tradeoffs; treat resistance management and future vaccines as secondary/exploratory decision domains.",
            "implementation_status": "draft_revision_needed",
        },
        {
            "manuscript_section": "Related manuscript disclosure",
            "recommended_logic": "Retain transparent npj Vaccines disclosure as related surveillance-methods work; emphasize this submission's distinct age-structured decision model and child/adolescent endpoint.",
            "implementation_status": "ready_for_cover_letter",
        },
    ]
    return pd.DataFrame(rows)


def _age_case_inventory() -> pd.DataFrame:
    path = project_path("data", "processed", "pertussis_age_case_inventory.csv")
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "country",
                "age_case_data_availability",
                "lancet_endpoint_validation_tier",
                "interpretation_note",
            ]
        )
    inventory = pd.read_csv(path)
    tier_map = {
        "direct_infant_under1_available": "age_corrected_direct_infant",
        "under5_only_no_infant_split": "age_triangulated_under5",
        "not_available_in_local_age_xlsx": "age_unavailable_aggregate_calibrated",
    }
    out = inventory.copy()
    out["lancet_endpoint_validation_tier"] = out["age_case_data_availability"].map(tier_map).fillna(
        "age_triangulated_partial"
    )
    out["interpretation_note"] = np.select(
        [
            out["lancet_endpoint_validation_tier"].eq("age_corrected_direct_infant"),
            out["lancet_endpoint_validation_tier"].eq("age_triangulated_under5"),
            out["lancet_endpoint_validation_tier"].eq("age_unavailable_aggregate_calibrated"),
        ],
        [
            "Direct under-1 age data support stronger infant-stratum interpretation; child/adolescent endpoints remain conditional model contrasts.",
            "Under-5 data support triangulation but do not split the two infant model strata.",
            "No local age-case workbook data were available; age-stratified outputs are aggregate-calibrated model contrasts.",
        ],
        default="Age-bin alignment is partial; use as descriptive validation rather than a hard calibration target.",
    )
    keep = [
        "country",
        "iso3",
        "first_year",
        "latest_year",
        "source_age_group_count",
        "exact_infant_year_count",
        "under5_year_count",
        "exact_model_aligned_row_count",
        "latest_target_year",
        "latest_preferred_age_group",
        "latest_preferred_target_role",
        "latest_preferred_reported_cases",
        "latest_preferred_case_share",
        "latest_preferred_incidence_per_100k",
        "age_case_data_availability",
        "lancet_endpoint_validation_tier",
        "interpretation_note",
    ]
    return out.loc[:, [column for column in keep if column in out.columns]]


def _baseline_pediatric_burden(burden: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    current = burden.loc[burden["strategy"].eq("current")].copy()
    if current.empty:
        return pd.DataFrame()
    keep = [
        "country",
        "primary_case_metric",
        "primary_cases_per_100k",
        "infant_cases_per_100k",
        "infant_hospitalizations_per_100k",
        "infant_deaths_per_100k",
        "child_1_9_cases_per_100k",
        "adolescent_cases_per_100k",
        "child_adolescent_infections_per_100k",
        "child_adolescent_reported_cases_per_100k",
        "child_adolescent_deaths_per_million",
        "child_adolescent_hospitalizations_per_100k",
        "metric_availability_note",
    ]
    out = current.loc[:, [column for column in keep if column in current.columns]].copy()
    if not inventory.empty:
        out = out.merge(
            inventory[
                [
                    "country",
                    "age_case_data_availability",
                    "lancet_endpoint_validation_tier",
                    "latest_preferred_target_role",
                ]
            ],
            on="country",
            how="left",
        )
    return out.sort_values("country")


def _endpoint_shift_summary(burden: pd.DataFrame) -> pd.DataFrame:
    endpoints = {
        "infant_0_11m": "infant_cases_per_100k",
        "child_1_9y": "child_1_9_cases_per_100k",
        "adolescent_10_17y": "adolescent_cases_per_100k",
        "child_adolescent_0_17y": "primary_cases_per_100k",
    }
    rows = []
    for country, group in burden.groupby("country", sort=False):
        ranks = {}
        for endpoint, column in endpoints.items():
            if column not in group.columns:
                continue
            ranks[endpoint] = _safe_numeric(group[column]).rank(method="min", ascending=True)
        for idx, record in group.iterrows():
            row = {
                "country": country,
                "strategy": record["strategy"],
                "strategy_label": record["strategy_label"],
                "primary_case_metric": record["primary_case_metric"],
            }
            for endpoint, column in endpoints.items():
                row[f"{endpoint}_cases_per_100k"] = record.get(column, np.nan)
                if endpoint in ranks:
                    row[f"{endpoint}_rank"] = float(ranks[endpoint].loc[idx])
            if "infant_0_11m" in ranks and "child_adolescent_0_17y" in ranks:
                row["rank_shift_child_adolescent_vs_infant"] = (
                    row.get("child_adolescent_0_17y_rank", np.nan) - row.get("infant_0_11m_rank", np.nan)
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["country", "child_adolescent_0_17y_rank", "strategy"])


def _age_pattern_fit_current() -> pd.DataFrame:
    path = project_path("outputs", "tables", "age_pattern_country_weights.csv")
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path)
    out["lancet_use"] = "Age-pattern triangulation and post-calibration weighting; not a universal calibration target."
    return out


def _age_pattern_weighted_strategy_summary(burden: pd.DataFrame) -> pd.DataFrame:
    weights_path = project_path("outputs", "tables", "age_pattern_country_weights.csv")
    bases: dict[str, pd.DataFrame] = {
        "all_profiles_unweighted": pd.DataFrame(
            {"country": sorted(burden["country"].dropna().unique()), "analysis_weight": 1.0}
        )
    }
    if weights_path.exists():
        weights = pd.read_csv(weights_path)
        country_weights = (
            weights.groupby("country", as_index=False)
            .agg(
                country_age_pattern_weight=("age_pattern_weight", "min"),
                all_checks_pass_weight_threshold=("passes_weight_threshold", "all"),
            )
            .copy()
        )
        bases["age_data_profiles_unweighted"] = country_weights[["country"]].assign(analysis_weight=1.0)
        bases["age_pattern_weighted"] = country_weights.rename(
            columns={"country_age_pattern_weight": "analysis_weight"}
        )[["country", "analysis_weight"]]
        bases["age_pattern_pass_filter"] = country_weights.loc[
            country_weights["all_checks_pass_weight_threshold"].eq(True), ["country"]
        ].assign(analysis_weight=1.0)
    rows = []
    for basis, basis_weights in bases.items():
        data = burden.merge(basis_weights, on="country", how="inner")
        data = data.loc[data["analysis_weight"].gt(0.0)].copy()
        if data.empty:
            continue
        for strategy, group in data.groupby("strategy", sort=False):
            weights = _safe_numeric(group["analysis_weight"]).fillna(0.0)
            reduction = _safe_numeric(group["primary_case_reduction"])
            primary_rate = _safe_numeric(group["primary_cases_per_100k"])
            rows.append(
                {
                    "ordering_basis": basis,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                    "primary_case_metric": group["primary_case_metric"].iloc[0],
                    "country_count": int(group["country"].nunique()),
                    "effective_country_weight_sum": float(weights.sum()),
                    "weighted_median_primary_case_reduction": _weighted_quantile(reduction, weights, 0.5),
                    "weighted_iqr_primary_case_reduction": _iqr_text(reduction, weights),
                    "weighted_median_primary_cases_per_100k": _weighted_quantile(primary_rate, weights, 0.5),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["strategy_rank_within_basis"] = out.groupby("ordering_basis")[
        "weighted_median_primary_case_reduction"
    ].rank(method="min", ascending=False)
    return out.sort_values(["ordering_basis", "strategy_rank_within_basis", "strategy"])


def _rank_margin_label(excess: float) -> str:
    if not np.isfinite(excess):
        return ""
    if excess <= 5.0:
        return "Near-tie"
    if excess < 25.0:
        return "Modest margin"
    return "Clear margin"


def _country_label(country: str) -> str:
    return str(country).replace("_", " ")


def _table1_programme_priorities(frontier: pd.DataFrame) -> pd.DataFrame:
    program = frontier.loc[
        frontier["optimization_constraint"].eq("program_only")
        & frontier["strategy"].isin(PROGRAM_ONLY_STRATEGIES)
    ].copy()
    rows = []
    for country, group in program.groupby("country", sort=False):
        current_rows = group.loc[group["strategy"].eq("current")]
        if current_rows.empty:
            continue
        ranked = (
            group.loc[~group["strategy"].eq("current")]
            .sort_values(["primary_cases_per_100k", "implementation_intensity", "strategy"])
            .reset_index(drop=True)
        )
        if len(ranked) < 2:
            continue
        current = current_rows.iloc[0]
        best = ranked.iloc[0]
        runner_up = ranked.iloc[1]
        excess = float(runner_up["primary_cases_per_100k"] - best["primary_cases_per_100k"])
        rows.append(
            {
                "programme_profile": _country_label(country),
                "current_practice_cases_per_100k_under18": current["primary_cases_per_100k"],
                "lowest_burden_programme_only_strategy": best["strategy_label"],
                "reduction_percent": 100.0 * best["primary_case_reduction"],
                "cases_averted_per_100k_under18": current["primary_cases_per_100k"]
                - best["primary_cases_per_100k"],
                "infant_hospitalisations_averted_per_100k_infants": current[
                    "infant_hospitalizations_per_100k"
                ]
                - best["infant_hospitalizations_per_100k"],
                "runner_up_excess_cases_per_100k_under18": excess,
                "rank_margin": _rank_margin_label(excess),
            }
        )
    return pd.DataFrame(rows).sort_values("programme_profile")


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervention = _augment_with_pediatric_metrics(
        _read("outputs/summaries/intervention_scenarios_summary.csv"),
        stem="intervention_scenarios",
        reference_scenario="current",
    )
    timeliness = _augment_with_pediatric_metrics(
        _read("outputs/summaries/routine_timeliness_sensitivity_summary.csv"),
        stem="routine_timeliness_sensitivity",
        reference_scenario="current",
    )
    vaccine = _augment_with_pediatric_metrics(
        _read("outputs/summaries/vaccine_scenarios_summary.csv"),
        stem="vaccine_scenarios",
        reference_scenario="no_vaccine",
    )
    _write(
        _metric_availability(
            {
                "intervention_scenarios_summary": intervention,
                "routine_timeliness_sensitivity_summary": timeliness,
                "vaccine_scenarios_summary": vaccine,
            }
        ),
        "outputs/tables/lancet_child_adolescent_metric_availability.csv",
    )

    burden = _strategy_burden_frame(_load_intervention_rows(intervention, timeliness, vaccine))
    frontier, preferred = _frontier_and_preferred(burden)
    summary = _strategy_summary(frontier)
    metric_basis = str(burden["primary_case_metric"].iloc[0]) if not burden.empty else "unknown"
    inventory = _age_case_inventory()

    _write(burden, "outputs/tables/lancet_child_adolescent_strategy_burden.csv")
    _write(frontier, "outputs/tables/lancet_child_adolescent_decision_frontier.csv")
    _write(preferred, "outputs/tables/lancet_child_adolescent_preferred_strategies.csv")
    _write(summary, "outputs/tables/lancet_child_adolescent_strategy_summary.csv")
    _write(_table1_programme_priorities(frontier), "outputs/tables/table1_profile_programme_priorities.csv")
    _write(_logic_blueprint(metric_basis), "outputs/tables/lancet_child_adolescent_logic_blueprint.csv")
    _write(inventory, "outputs/tables/lancet_age_case_data_inventory.csv")
    _write(_baseline_pediatric_burden(burden, inventory), "outputs/tables/lancet_baseline_pediatric_burden.csv")
    _write(_endpoint_shift_summary(burden), "outputs/tables/lancet_endpoint_shift_summary.csv")
    _write(_age_pattern_fit_current(), "outputs/tables/lancet_age_pattern_fit_current.csv")
    _write(
        _age_pattern_weighted_strategy_summary(burden),
        "outputs/tables/lancet_age_pattern_weighted_strategy_summary.csv",
    )
    write_run_metadata(
        "lancet_child_adolescent_tables",
        current_run_metadata("lancet_child_adolescent_tables", row_counts=WRITTEN_ROW_COUNTS),
    )
    return burden, preferred, summary


if __name__ == "__main__":
    main()
