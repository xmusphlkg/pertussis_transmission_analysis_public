from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import current_run_metadata, write_run_metadata
from src_python.utils.io import project_path, write_dataframe


SELECTED_INTERVENTIONS = (
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
)

SCENARIO_CLASS = {
    "current": "current_practice",
    "higher_child_coverage": "routine_program_marginal_levers",
    "adolescent_booster": "routine_program_marginal_levers",
    "pregnancy_tdap_scaleup": "infant_protection_and_exposure_reduction",
    "cocooning_adjunct": "infant_protection_and_exposure_reduction",
    "maternal_immunization": "infant_protection_and_exposure_reduction",
    "targeted_pep_high_risk": "management_modifiers",
    "resistance_guided_treatment": "management_modifiers",
    "next_generation_vaccine": "high_transmission_blocking_vaccine_target",
    "combined_strategy": "combined_stress_test_package",
}

SCENARIO_LABEL = {
    "current": "Current practice",
    "higher_child_coverage": "Nominal coverage floor without timeliness improvement",
    "adolescent_booster": "Adolescent booster",
    "pregnancy_tdap_scaleup": "Pregnancy Tdap scale-up",
    "cocooning_adjunct": "Close-contact adult adjunct",
    "maternal_immunization": "Infant-exposure reduction strategy",
    "targeted_pep_high_risk": "Targeted high-risk PEP",
    "resistance_guided_treatment": "Resistance-guided management",
    "next_generation_vaccine": "High-transmission-blocking vaccine target",
    "combined_strategy": "Combined strategy",
}

SCENARIO_CLASS_ORDER = (
    "current_practice",
    "routine_program_marginal_levers",
    "infant_protection_and_exposure_reduction",
    "management_modifiers",
    "high_transmission_blocking_vaccine_target",
    "combined_stress_test_package",
)

SCENARIO_QUALITATIVE_TIER = {
    "combined_stress_test_package": "transmission_blocking_or_combined_top_tier",
    "high_transmission_blocking_vaccine_target": "transmission_blocking_or_combined_top_tier",
    "infant_protection_and_exposure_reduction": "infant_protection_or_management_middle_tier",
    "management_modifiers": "infant_protection_or_management_middle_tier",
    "current_practice": "current_practice_reference_tier",
    "routine_program_marginal_levers": "routine_marginal_low_tier",
}


@dataclass(frozen=True)
class AgePatternCheck:
    country: str
    metric: str
    target_value: float
    tolerance_abs: float
    age_groups: tuple[str, ...]
    external_label: str
    source_note: str
    model_window: str = "configured prospective scenario horizon"


AGE_PATTERN_CHECKS = (
    AgePatternCheck(
        country="United_States",
        metric="reported_case_share",
        target_value=0.124,
        tolerance_abs=0.080,
        age_groups=("infant_0_2m", "infant_3_11m"),
        external_label="Infants <1 y share of 2025 provisional reported cases",
        source_note="US provisional 2025 surveillance age distribution used as an external check; not a calibration target.",
    ),
    AgePatternCheck(
        country="United_Kingdom",
        metric="reported_case_share",
        target_value=0.060,
        tolerance_abs=0.100,
        age_groups=("infant_0_2m", "infant_3_11m"),
        external_label="Infants <1 y share of 2024 laboratory-confirmed cases in England",
        source_note="England 2024 laboratory-confirmed age distribution used as an external check; not a calibration target.",
    ),
    AgePatternCheck(
        country="Sweden",
        metric="reported_case_share",
        target_value=0.048,
        tolerance_abs=0.100,
        age_groups=("infant_0_2m", "infant_3_11m"),
        external_label="EU/EEA 2024 infant case share proxy for Sweden external triangulation",
        source_note="EU/EEA 2024 infant case share used as a broad external age-pattern check for Sweden.",
    ),
    AgePatternCheck(
        country="Australia",
        metric="reported_case_share",
        target_value=0.570,
        tolerance_abs=0.200,
        age_groups=("child_5_9y", "adolescent_10_17y"),
        external_label="Children 5-14 y share of 2024 reported cases, compared with the model 5-17 y proxy",
        source_note=(
            "Australia 2024 school-age-dominated distribution used as a coarse check; "
            "model age bins require a 5-17 y proxy."
        ),
    ),
)


def _weighted_quantile(values: pd.Series, weights: pd.Series, probability: float) -> float:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "weight": weights})
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


def _iqr_text(values: pd.Series, weights: pd.Series) -> str:
    q25 = _weighted_quantile(values, weights, 0.25)
    q75 = _weighted_quantile(values, weights, 0.75)
    if not np.isfinite(q25) or not np.isfinite(q75):
        return ""
    return f"{q25:.4g} to {q75:.4g}"


def _read_intervention_summary() -> pd.DataFrame:
    path = project_path("outputs", "summaries", "intervention_scenarios_summary.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing intervention scenario summary: {path}")
    data = pd.read_csv(path)
    required = {
        "country",
        "scenario",
        "annualized_infant_cases_per_100k",
        "relative_reduction_infant_cases",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Intervention scenario summary is missing columns: {sorted(missing)}")
    data = data.loc[data["scenario"].isin(SELECTED_INTERVENTIONS)].copy()
    data["scenario_class"] = data["scenario"].map(SCENARIO_CLASS)
    data = data.loc[data["scenario_class"].notna()].copy()
    return data


def _current_age_pattern_frame() -> pd.DataFrame:
    path = project_path("outputs", "simulations", "intervention_scenarios.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing intervention scenario timeseries: {path}")
    columns = ["country", "scenario", "age_group", "reported_cases"]
    data = pd.read_parquet(path, columns=columns)
    data = data.loc[data["scenario"].eq("current")].copy()
    age = (
        data.groupby(["country", "age_group"], as_index=False)
        .agg(reported_cases=("reported_cases", "sum"))
        .copy()
    )
    totals = age.groupby("country", as_index=False)["reported_cases"].sum().rename(
        columns={"reported_cases": "total_reported_cases"}
    )
    return age.merge(totals, on="country", how="left")


def _age_pattern_weights(min_pass_weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    age = _current_age_pattern_frame()
    rows: list[dict[str, Any]] = []
    for check in AGE_PATTERN_CHECKS:
        country_age = age.loc[age["country"].eq(check.country)]
        if country_age.empty:
            modeled_value = np.nan
            denominator = np.nan
        else:
            numerator = float(country_age.loc[country_age["age_group"].isin(check.age_groups), "reported_cases"].sum())
            denominator = float(country_age["total_reported_cases"].iloc[0])
            modeled_value = numerator / denominator if denominator > 0.0 else np.nan
        absolute_difference = modeled_value - check.target_value if np.isfinite(modeled_value) else np.nan
        standardized_difference = (
            absolute_difference / check.tolerance_abs
            if np.isfinite(absolute_difference) and check.tolerance_abs > 0
            else np.nan
        )
        weight = float(np.exp(-0.5 * standardized_difference**2)) if np.isfinite(standardized_difference) else 0.0
        rows.append(
            {
                "country": check.country,
                "metric": check.metric,
                "external_label": check.external_label,
                "model_age_groups": ";".join(check.age_groups),
                "model_window": check.model_window,
                "external_value": check.target_value,
                "modeled_value": modeled_value,
                "absolute_difference": absolute_difference,
                "tolerance_abs": check.tolerance_abs,
                "standardized_difference": standardized_difference,
                "age_pattern_weight": weight,
                "passes_weight_threshold": weight >= min_pass_weight,
                "source_note": check.source_note,
            }
        )

    detail = pd.DataFrame(rows)
    country = (
        detail.groupby("country", as_index=False)
        .agg(
            n_age_pattern_checks=("metric", "count"),
            minimum_age_pattern_weight=("age_pattern_weight", "min"),
            country_age_pattern_weight=(
                "age_pattern_weight",
                lambda x: float(np.exp(np.log(np.clip(x, 1e-12, 1.0)).mean())),
            ),
            all_checks_pass_weight_threshold=("passes_weight_threshold", "all"),
        )
        .sort_values("country")
    )
    return detail, country


def _basis_country_weights(
    summary: pd.DataFrame,
    country_weights: pd.DataFrame,
    *,
    min_pass_weight: float,
) -> dict[str, pd.DataFrame]:
    all_countries = pd.DataFrame({"country": sorted(summary["country"].dropna().unique()), "analysis_weight": 1.0})
    external = country_weights.loc[:, ["country", "country_age_pattern_weight", "all_checks_pass_weight_threshold"]].copy()
    external_unweighted = external.loc[:, ["country"]].assign(analysis_weight=1.0)
    external_weighted = external.rename(columns={"country_age_pattern_weight": "analysis_weight"})[
        ["country", "analysis_weight"]
    ]
    pass_filter = external.loc[
        external["country_age_pattern_weight"].ge(min_pass_weight)
        & external["all_checks_pass_weight_threshold"].eq(True),
        ["country"],
    ].assign(analysis_weight=1.0)
    if pass_filter.empty:
        pass_filter = external.loc[external["country_age_pattern_weight"].ge(min_pass_weight), ["country"]].assign(
            analysis_weight=1.0
        )

    return {
        "all_profiles_unweighted": all_countries,
        "external_profiles_unweighted": external_unweighted,
        "external_age_pattern_weighted": external_weighted,
        "external_age_pattern_pass_filter": pass_filter,
    }


def _summarize_basis(
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    basis_name: str,
) -> pd.DataFrame:
    data = summary.merge(weights, on="country", how="inner")
    data = data.loc[data["analysis_weight"].gt(0.0)].copy()
    if data.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for scenario_class in SCENARIO_CLASS_ORDER:
        group = data.loc[data["scenario_class"].eq(scenario_class)].copy()
        if group.empty:
            continue
        scenarios = sorted(group["scenario"].unique())
        countries = sorted(group["country"].unique())
        weights_for_rows = pd.to_numeric(group["analysis_weight"], errors="coerce").fillna(0.0)
        infant = pd.to_numeric(group["annualized_infant_cases_per_100k"], errors="coerce")
        reduction = pd.to_numeric(group["relative_reduction_infant_cases"], errors="coerce")
        rows.append(
            {
                "ordering_basis": basis_name,
                "scenario_class": scenario_class,
                "scenarios_in_class": ";".join(SCENARIO_LABEL.get(scenario, scenario) for scenario in scenarios),
                "countries_included": ";".join(countries),
                "country_count": len(countries),
                "scenario_country_cells": len(group),
                "effective_country_weight_sum": float(weights["analysis_weight"].sum()),
                "weighted_median_infant_cases_per_100k": _weighted_quantile(infant, weights_for_rows, 0.5),
                "weighted_iqr_infant_cases_per_100k": _iqr_text(infant, weights_for_rows),
                "weighted_median_relative_reduction_infant_cases": _weighted_quantile(reduction, weights_for_rows, 0.5),
                "weighted_iqr_relative_reduction_infant_cases": _iqr_text(reduction, weights_for_rows),
            }
        )

    out = pd.DataFrame(rows)
    out["class_rank_basis"] = "weighted median relative reduction in infant cases vs current practice"
    out["class_rank"] = out["weighted_median_relative_reduction_infant_cases"].rank(method="min", ascending=False)
    out = out.sort_values(["class_rank", "scenario_class"]).reset_index(drop=True)
    sequence = " > ".join(out.sort_values(["class_rank", "scenario_class"])["scenario_class"].tolist())
    out["class_order_sequence"] = sequence
    tier_sequence_values = []
    for scenario_class in out.sort_values(["class_rank", "scenario_class"])["scenario_class"].tolist():
        tier = SCENARIO_QUALITATIVE_TIER.get(scenario_class, scenario_class)
        if not tier_sequence_values or tier_sequence_values[-1] != tier:
            tier_sequence_values.append(tier)
    out["qualitative_tier_order_sequence"] = " > ".join(tier_sequence_values)
    return out


def _add_order_change_flags(ordering: pd.DataFrame) -> pd.DataFrame:
    out = ordering.copy()
    reference = out.loc[out["ordering_basis"].eq("all_profiles_unweighted"), ["scenario_class", "class_rank"]].rename(
        columns={"class_rank": "reference_class_rank_all_profiles"}
    )
    out = out.merge(reference, on="scenario_class", how="left")
    out["rank_shift_vs_all_profiles"] = out["class_rank"] - out["reference_class_rank_all_profiles"]

    reference_sequence = (
        out.loc[out["ordering_basis"].eq("all_profiles_unweighted"), "class_order_sequence"].dropna().iloc[0]
        if out["ordering_basis"].eq("all_profiles_unweighted").any()
        else ""
    )
    reference_tier_sequence = (
        out.loc[out["ordering_basis"].eq("all_profiles_unweighted"), "qualitative_tier_order_sequence"]
        .dropna()
        .iloc[0]
        if out["ordering_basis"].eq("all_profiles_unweighted").any()
        else ""
    )
    out["strict_class_order_changed_vs_all_profiles"] = out["class_order_sequence"].ne(reference_sequence)
    out["qualitative_ordering_changed_vs_all_profiles"] = out["qualitative_tier_order_sequence"].ne(
        reference_tier_sequence
    )
    out["interpretation_note"] = np.select(
        [
            out["qualitative_ordering_changed_vs_all_profiles"],
            out["strict_class_order_changed_vs_all_profiles"],
        ],
        [
            "Qualitative tier order differs from the unweighted 10-profile comparison.",
            "Exact class order differs only within a qualitative tier; tier-level interpretation matches.",
        ],
        default="Qualitative tier order matches the unweighted 10-profile comparison.",
    )
    return out


def main(min_pass_weight: float = 0.50) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _read_intervention_summary()
    age_detail, country_weights = _age_pattern_weights(min_pass_weight=min_pass_weight)
    detail = age_detail.merge(country_weights, on="country", how="left")

    basis_weights = _basis_country_weights(summary, country_weights, min_pass_weight=min_pass_weight)
    ordering_frames = [
        _summarize_basis(summary, weights, basis_name)
        for basis_name, weights in basis_weights.items()
        if not weights.empty
    ]
    ordering = pd.concat(ordering_frames, ignore_index=True) if ordering_frames else pd.DataFrame()
    ordering = _add_order_change_flags(ordering) if not ordering.empty else ordering

    write_dataframe(detail, project_path("outputs", "tables", "age_pattern_country_weights.csv"))
    write_dataframe(ordering, project_path("outputs", "tables", "age_pattern_scenario_ordering_sensitivity.csv"))
    metadata = current_run_metadata(
        "age_pattern_sensitivity",
        row_counts={
            "country_weights": int(len(detail)),
            "scenario_ordering_sensitivity": int(len(ordering)),
        },
    )
    metadata.update({"min_pass_weight": float(min_pass_weight)})
    write_run_metadata("age_pattern_sensitivity", metadata)
    return detail, ordering


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External age-pattern weighted scenario-class ordering sensitivity diagnostic."
    )
    parser.add_argument(
        "--min-pass-weight",
        type=float,
        default=0.50,
        help="Minimum country age-pattern weight used by the pass-filter diagnostic.",
    )
    args = parser.parse_args()
    main(min_pass_weight=args.min_pass_weight)
