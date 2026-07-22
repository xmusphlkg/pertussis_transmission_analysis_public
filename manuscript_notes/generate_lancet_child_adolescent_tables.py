from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import (
    current_run_metadata,
    file_sha256,
    load_configs,
    publication_country_names,
    validate_run_metadata,
    validated_calibration_artifact_path_hashes,
    write_run_metadata,
)
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

FIGURE2_PROGRAMME_STRATEGIES = (
    "timeliness_only",
    "maternal_immunization",
    "pregnancy_tdap_scaleup",
    "adolescent_booster",
    "cocooning_adjunct",
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
    "maternal_immunization": "Infant-exposure package",
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
    "maternal_immunization": "High-intensity infant-exposure pathway package",
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

TABLE_STEM = "lancet_child_adolescent_tables"
FRONTIER_STEM = "lancet_child_adolescent_decision_frontier"
BOOTSTRAP_STEM = "figure2c_parametric_bootstrap"
BOOTSTRAP_AUDIT_STEM = "figure2c_parametric_bootstrap_quality_audit"
FRONTIER_RELATIVE_PATH = (
    "outputs/tables/lancet_child_adolescent_decision_frontier.csv"
)
BOOTSTRAP_RELATIVE_PATH = (
    "outputs/tables/figure2c_programme_paired_bootstrap_draws.csv"
)
TABLE1_RELATIVE_PATH = "outputs/tables/table1_profile_programme_priorities.csv"
FRONTIER_PARENT_STEMS = (
    "intervention_scenarios",
    "vaccine_scenarios",
    "figure2_programme_reference",
)

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


def _metadata_artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_path().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _parent_artifact_paths(stem: str) -> tuple[Path, ...]:
    return (
        project_path("outputs", "summaries", f"{stem}_summary.csv"),
        project_path("outputs", "summaries", f"{stem}_summary.parquet"),
        project_path("outputs", "simulations", f"{stem}.parquet"),
    )


def _validated_parent_artifact_hashes(
    stems: Iterable[str],
    *,
    countries: tuple[str, ...],
) -> dict[str, str]:
    """Bind deterministic parents to the same nine accepted calibrations."""

    expected_calibrations = validated_calibration_artifact_path_hashes(
        countries,
        context=f"{FRONTIER_STEM} deterministic parents",
    )
    artifact_hashes = dict(expected_calibrations)
    for stem in stems:
        metadata = validate_run_metadata(stem)
        recorded = metadata.get("input_artifact_path_sha256")
        if not isinstance(recorded, dict) or dict(sorted(recorded.items())) != dict(
            sorted(expected_calibrations.items())
        ):
            raise ValueError(
                f"Deterministic parent {stem} is not bound to exactly the current "
                f"{len(expected_calibrations)} accepted calibration artifacts."
            )
        for path in _parent_artifact_paths(stem):
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact_hashes[_metadata_artifact_path(path)] = file_sha256(path)
    return dict(sorted(artifact_hashes.items()))


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
    programme_reference: pd.DataFrame,
    vaccine: pd.DataFrame,
) -> pd.DataFrame:
    # Figure 2 programme rows must share one production-runtime current-practice
    # trajectory.  Do not splice the coarser routine-timeliness diagnostic into
    # this decision frontier.
    locked_programme_strategies = {
        "current",
        "timeliness_only",
        "adolescent_booster",
        "pregnancy_tdap_scaleup",
        "cocooning_adjunct",
        "maternal_immunization",
        "targeted_pep_high_risk",
    }
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
    intervention = intervention.loc[
        ~intervention["scenario"].isin(locked_programme_strategies)
    ].copy()

    programme_reference = programme_reference.loc[
        programme_reference["scenario"].isin(locked_programme_strategies)
    ].copy()
    programme_reference["strategy"] = programme_reference["scenario"]
    programme_reference["intervention"] = programme_reference["scenario"]
    expected_countries = set(intervention["country"].astype(str))
    observed_countries = set(programme_reference["country"].astype(str))
    if expected_countries != observed_countries:
        raise ValueError(
            "Locked Figure 2 programme parent has a country mismatch: "
            f"missing={sorted(expected_countries - observed_countries)}, "
            f"unexpected={sorted(observed_countries - expected_countries)}"
        )
    expected_rows = len(expected_countries) * len(locked_programme_strategies)
    if (
        len(programme_reference) != expected_rows
        or programme_reference[["country", "strategy"]].duplicated().any()
    ):
        raise ValueError(
            "Locked Figure 2 programme parent requires one current and six "
            "programme rows per profile"
        )

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

    combined = pd.concat(
        [intervention, programme_reference, transmission],
        ignore_index=True,
        sort=False,
    )
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
    out["source_primary_case_reduction"] = pd.to_numeric(
        out.get(reduction_col), errors="coerce"
    )
    current = out.loc[out["strategy"].eq("current"), ["country", "primary_cases_per_100k", "primary_total_cases"]].rename(
        columns={
            "primary_cases_per_100k": "current_primary_cases_per_100k",
            "primary_total_cases": "current_primary_total_cases",
        }
    )
    out = out.drop(
        columns=[
            column
            for column in (
                "current_primary_cases_per_100k",
                "current_primary_total_cases",
            )
            if column in out.columns
        ]
    )
    out = out.merge(current, on="country", how="left")
    out["primary_case_reduction"] = 1.0 - out["primary_cases_per_100k"] / out[
        "current_primary_cases_per_100k"
    ].replace(0, np.nan)
    out["source_minus_common_denominator_reduction"] = (
        out["source_primary_case_reduction"] - out["primary_case_reduction"]
    )
    finite_required = out[[
        "primary_cases_per_100k",
        "current_primary_cases_per_100k",
        "primary_case_reduction",
    ]].to_numpy(dtype=float)
    if not np.isfinite(finite_required).all():
        raise ValueError("Programme burdens and common current denominators must be finite")
    algebraic = 1.0 - out["primary_cases_per_100k"] / out[
        "current_primary_cases_per_100k"
    ]
    if not np.allclose(
        out["primary_case_reduction"].to_numpy(dtype=float),
        algebraic.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("Programme reductions failed the common-denominator audit")
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
        "source_primary_case_reduction",
        "source_minus_common_denominator_reduction",
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


def _country_label(country: str) -> str:
    return str(country).replace("_", " ")


def _table1_programme_priorities(
    frontier: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
) -> pd.DataFrame:
    program = frontier.loc[
        frontier["optimization_constraint"].eq("program_only")
        & frontier["strategy"].isin(("current", *FIGURE2_PROGRAMME_STRATEGIES))
    ].copy()
    required_bootstrap = {
        "country",
        "bootstrap_replicate",
        "strategy",
        "current_rate",
        "intervention_rate",
    }
    missing_bootstrap = sorted(required_bootstrap.difference(bootstrap_draws.columns))
    if missing_bootstrap:
        raise KeyError(
            f"Figure 2 paired bootstrap draws are missing columns: {missing_bootstrap}"
        )
    bootstrap = bootstrap_draws.loc[
        bootstrap_draws["strategy"].isin(FIGURE2_PROGRAMME_STRATEGIES)
    ].copy()
    bootstrap["country"] = bootstrap["country"].astype(str).str.replace(
        " ", "_", regex=False
    )
    bootstrap["bootstrap_replicate"] = pd.to_numeric(
        bootstrap["bootstrap_replicate"], errors="raise"
    ).astype(int)
    for column in ("current_rate", "intervention_rate"):
        bootstrap[column] = pd.to_numeric(bootstrap[column], errors="raise")
    paired_blocks = bootstrap.groupby(["country", "bootstrap_replicate"])
    block_sizes = paired_blocks.size()
    strategy_counts = paired_blocks["strategy"].nunique()
    current_ranges = paired_blocks["current_rate"].agg(lambda values: values.max() - values.min())
    if (
        (block_sizes != len(FIGURE2_PROGRAMME_STRATEGIES)).any()
        or (strategy_counts != len(FIGURE2_PROGRAMME_STRATEGIES)).any()
        or (current_ranges > 1e-10).any()
    ):
        raise ValueError(
            "Figure 2 bootstrap rows must contain all six strategies paired to one "
            "current-practice rate in every replicate"
        )

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
        country_draws = bootstrap.loc[bootstrap["country"].eq(str(country))]
        leader_draws = country_draws.loc[
            country_draws["strategy"].eq(str(best["strategy"])),
            ["bootstrap_replicate", "current_rate", "intervention_rate"],
        ].rename(columns={"intervention_rate": "leader_rate"})
        runner_draws = country_draws.loc[
            country_draws["strategy"].eq(str(runner_up["strategy"])),
            ["bootstrap_replicate", "current_rate", "intervention_rate"],
        ].rename(
            columns={
                "current_rate": "runner_current_rate",
                "intervention_rate": "runner_rate",
            }
        )
        paired = leader_draws.merge(
            runner_draws,
            on="bootstrap_replicate",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(leader_draws) or len(paired) != len(runner_draws):
            raise ValueError(f"Incomplete leader/runner bootstrap pairing for {country}")
        if not np.allclose(
            paired["current_rate"],
            paired["runner_current_rate"],
            rtol=1e-12,
            atol=1e-10,
        ):
            raise AssertionError(f"Paired current-practice rates disagree for {country}")
        absolute_reduction = paired["current_rate"] - paired["leader_rate"]
        paired_margin = paired["runner_rate"] - paired["leader_rate"]
        effect_q025, effect_q975 = np.quantile(absolute_reduction, [0.025, 0.975])
        margin_q025, margin_q975 = np.quantile(paired_margin, [0.025, 0.975])
        rows.append(
            {
                "programme_profile": _country_label(country),
                "current_practice_cases_per_100k_under18": current["primary_cases_per_100k"],
                "lowest_burden_programme_only_strategy": best["strategy_label"],
                "reduction_percent": 100.0 * best["primary_case_reduction"],
                "cases_averted_per_100k_under18": current["primary_cases_per_100k"]
                - best["primary_cases_per_100k"],
                "cases_averted_per_100k_under18_q025": float(effect_q025),
                "cases_averted_per_100k_under18_q975": float(effect_q975),
                "infant_hospitalisations_averted_per_100k_infants": current[
                    "infant_hospitalizations_per_100k"
                ]
                - best["infant_hospitalizations_per_100k"],
                "second_ranked_programme_only_strategy": runner_up["strategy_label"],
                "runner_up_excess_cases_per_100k_under18": excess,
                "runner_up_excess_cases_per_100k_under18_q025": float(margin_q025),
                "runner_up_excess_cases_per_100k_under18_q975": float(margin_q975),
                "paired_interval_status": (
                    "above_zero" if float(margin_q025) > 0.0 else "includes_zero"
                ),
                "successful_paired_bootstrap_replicates": int(len(paired)),
            }
        )
    return pd.DataFrame(rows).sort_values("programme_profile")


@dataclass(frozen=True)
class FrontierProducts:
    countries: tuple[str, ...]
    input_artifact_path_sha256: dict[str, str]
    intervention: pd.DataFrame
    vaccine: pd.DataFrame
    programme_reference: pd.DataFrame
    burden: pd.DataFrame
    frontier: pd.DataFrame
    preferred: pd.DataFrame
    summary: pd.DataFrame


def _read_augmented_parent(stem: str, *, reference_scenario: str) -> pd.DataFrame:
    return _augment_with_pediatric_metrics(
        _read(f"outputs/summaries/{stem}_summary.csv"),
        stem=stem,
        reference_scenario=reference_scenario,
    )


def _build_frontier_products() -> FrontierProducts:
    countries = tuple(publication_country_names(load_configs()))
    input_hashes = _validated_parent_artifact_hashes(
        FRONTIER_PARENT_STEMS,
        countries=countries,
    )
    intervention = _read_augmented_parent(
        "intervention_scenarios",
        reference_scenario="current",
    )
    vaccine = _read_augmented_parent(
        "vaccine_scenarios",
        reference_scenario="symptom_protective",
    )
    programme_reference = _read_augmented_parent(
        "figure2_programme_reference",
        reference_scenario="current",
    )
    burden = _strategy_burden_frame(
        _load_intervention_rows(intervention, programme_reference, vaccine)
    )
    frontier, preferred = _frontier_and_preferred(burden)
    return FrontierProducts(
        countries=countries,
        input_artifact_path_sha256=input_hashes,
        intervention=intervention,
        vaccine=vaccine,
        programme_reference=programme_reference,
        burden=burden,
        frontier=frontier,
        preferred=preferred,
        summary=_strategy_summary(frontier),
    )


def generate_frontier_only() -> pd.DataFrame:
    """Write the deterministic decision frontier without reading bootstrap data."""

    WRITTEN_ROW_COUNTS.clear()
    products = _build_frontier_products()
    _write(products.frontier, FRONTIER_RELATIVE_PATH)
    frontier_path = project_path(FRONTIER_RELATIVE_PATH)
    metadata = current_run_metadata(
        FRONTIER_STEM,
        row_counts={"decision_frontier": int(len(products.frontier))},
    ) | {
        "analysis_role": "prebootstrap_deterministic_decision_frontier",
        "publication_path": True,
        "bootstrap_independent": True,
        "reads_bootstrap_artifact": False,
        "countries": list(products.countries),
        "programme_strategies": list(FIGURE2_PROGRAMME_STRATEGIES),
        "deterministic_parent_stems": list(FRONTIER_PARENT_STEMS),
        "input_artifact_path_sha256": products.input_artifact_path_sha256,
        "output_artifact_sha256": {
            "decision_frontier": file_sha256(frontier_path),
        },
    }
    write_run_metadata(FRONTIER_STEM, metadata)
    return products.frontier


def _validated_frontier_artifact(
    *,
    expected_input_hashes: dict[str, str],
) -> pd.DataFrame:
    metadata = validate_run_metadata(FRONTIER_STEM)
    if (
        metadata.get("analysis_role")
        != "prebootstrap_deterministic_decision_frontier"
        or metadata.get("bootstrap_independent") is not True
        or metadata.get("reads_bootstrap_artifact") is not False
    ):
        raise ValueError("Decision-frontier metadata have the wrong pre-bootstrap role.")
    if metadata.get("input_artifact_path_sha256") != expected_input_hashes:
        raise ValueError("Decision frontier is stale relative to deterministic parents.")
    frontier_path = project_path(FRONTIER_RELATIVE_PATH)
    if not frontier_path.is_file():
        raise FileNotFoundError(frontier_path)
    observed_digest = file_sha256(frontier_path)
    recorded_outputs = metadata.get("output_artifact_sha256")
    if (
        not isinstance(recorded_outputs, dict)
        or recorded_outputs.get("decision_frontier") != observed_digest
    ):
        raise ValueError("Decision frontier does not match its producer metadata.")
    return pd.read_csv(frontier_path)


def _assert_same_frontier(expected: pd.DataFrame, observed: pd.DataFrame) -> None:
    sort_columns = [
        "optimization_constraint",
        "country",
        "implementation_intensity",
        "strategy",
    ]
    if set(expected.columns) != set(observed.columns):
        raise ValueError("Decision frontier columns changed after bootstrap.")
    expected_sorted = expected.sort_values(sort_columns).reset_index(drop=True)
    observed_sorted = observed.loc[:, expected.columns].sort_values(sort_columns).reset_index(
        drop=True
    )
    try:
        pd.testing.assert_frame_equal(
            expected_sorted,
            observed_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "Decision frontier changed between the pre-bootstrap and final table stages."
        ) from exc


def _validated_bootstrap_draws() -> tuple[pd.DataFrame, dict[str, str]]:
    bootstrap_metadata = validate_run_metadata(BOOTSTRAP_STEM)
    audit_metadata = validate_run_metadata(BOOTSTRAP_AUDIT_STEM)
    if (
        audit_metadata.get("passed") is not True
        or audit_metadata.get("warnings_are_fatal") is not True
        or audit_metadata.get("figure2c_source_stem") != BOOTSTRAP_STEM
    ):
        raise ValueError("Figure 2c bootstrap quality audit is missing or did not pass.")
    bootstrap_path = project_path(BOOTSTRAP_RELATIVE_PATH)
    if not bootstrap_path.is_file():
        raise FileNotFoundError(bootstrap_path)
    observed_digest = file_sha256(bootstrap_path)
    recorded_outputs = bootstrap_metadata.get("output_artifact_sha256")
    audited_outputs = audit_metadata.get("audited_artifact_sha256")
    if (
        not isinstance(recorded_outputs, dict)
        or recorded_outputs.get("paired_bootstrap_draws") != observed_digest
        or not isinstance(audited_outputs, dict)
        or audited_outputs.get("paired_bootstrap_draws_sha256") != observed_digest
    ):
        raise ValueError("Figure 2c bootstrap draws do not match source and audit metadata.")
    return pd.read_csv(bootstrap_path), {
        _metadata_artifact_path(bootstrap_path): observed_digest
    }


def validate_frontier_for_bootstrap() -> pd.DataFrame:
    """Fail closed unless the saved frontier matches every current parent."""

    products = _build_frontier_products()
    frontier = _validated_frontier_artifact(
        expected_input_hashes=products.input_artifact_path_sha256
    )
    _assert_same_frontier(products.frontier, frontier)
    return frontier


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate complete post-bootstrap Lancet tables from locked parents."""

    WRITTEN_ROW_COUNTS.clear()
    products = _build_frontier_products()
    frontier = _validated_frontier_artifact(
        expected_input_hashes=products.input_artifact_path_sha256
    )
    _assert_same_frontier(products.frontier, frontier)
    bootstrap_draws, bootstrap_input_hash = _validated_bootstrap_draws()
    timeliness_input_hashes = _validated_parent_artifact_hashes(
        ("routine_timeliness_sensitivity",),
        countries=products.countries,
    )
    timeliness = _read_augmented_parent(
        "routine_timeliness_sensitivity",
        reference_scenario="current",
    )
    _write(
        _metric_availability(
            {
                "intervention_scenarios_summary": products.intervention,
                "routine_timeliness_sensitivity_summary": timeliness,
                "figure2_programme_reference_summary": products.programme_reference,
                "vaccine_scenarios_summary": products.vaccine,
            }
        ),
        "outputs/tables/lancet_child_adolescent_metric_availability.csv",
    )

    burden = products.burden
    preferred = products.preferred
    summary = products.summary
    metric_basis = str(burden["primary_case_metric"].iloc[0]) if not burden.empty else "unknown"
    inventory = _age_case_inventory()

    _write(burden, "outputs/tables/lancet_child_adolescent_strategy_burden.csv")
    _write(preferred, "outputs/tables/lancet_child_adolescent_preferred_strategies.csv")
    _write(summary, "outputs/tables/lancet_child_adolescent_strategy_summary.csv")
    _write(
        _table1_programme_priorities(frontier, bootstrap_draws),
        TABLE1_RELATIVE_PATH,
    )
    _write(_logic_blueprint(metric_basis), "outputs/tables/lancet_child_adolescent_logic_blueprint.csv")
    _write(inventory, "outputs/tables/lancet_age_case_data_inventory.csv")
    _write(_baseline_pediatric_burden(burden, inventory), "outputs/tables/lancet_baseline_pediatric_burden.csv")
    _write(_endpoint_shift_summary(burden), "outputs/tables/lancet_endpoint_shift_summary.csv")
    _write(_age_pattern_fit_current(), "outputs/tables/lancet_age_pattern_fit_current.csv")
    _write(
        _age_pattern_weighted_strategy_summary(burden),
        "outputs/tables/lancet_age_pattern_weighted_strategy_summary.csv",
    )
    input_hashes = {
        **products.input_artifact_path_sha256,
        **timeliness_input_hashes,
        _metadata_artifact_path(project_path(FRONTIER_RELATIVE_PATH)): file_sha256(
            project_path(FRONTIER_RELATIVE_PATH)
        ),
        **bootstrap_input_hash,
    }
    output_hashes = {
        relative_path: file_sha256(project_path(relative_path))
        for relative_path in WRITTEN_ROW_COUNTS
    }
    metadata = current_run_metadata(
        TABLE_STEM,
        row_counts=WRITTEN_ROW_COUNTS,
    ) | {
        "frontier_source_stem": FRONTIER_STEM,
        "bootstrap_source_stem": BOOTSTRAP_STEM,
        "bootstrap_audit_stem": BOOTSTRAP_AUDIT_STEM,
        "input_artifact_path_sha256": dict(sorted(input_hashes.items())),
        "output_artifact_sha256": output_hashes,
    }
    write_run_metadata(TABLE_STEM, metadata)
    return burden, preferred, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontier-only",
        action="store_true",
        help="Generate only the bootstrap-independent deterministic decision frontier.",
    )
    parser.add_argument(
        "--validate-frontier",
        action="store_true",
        help="Validate the saved frontier and all deterministic parents without writing.",
    )
    arguments = parser.parse_args()
    if arguments.frontier_only and arguments.validate_frontier:
        parser.error("--frontier-only and --validate-frontier are mutually exclusive")
    if arguments.frontier_only:
        generate_frontier_only()
    elif arguments.validate_frontier:
        validate_frontier_for_bootstrap()
    else:
        main()
