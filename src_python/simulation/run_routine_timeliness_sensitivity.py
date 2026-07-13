from __future__ import annotations

import argparse
from copy import deepcopy
import re
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    load_configs,
    make_intervention_config,
    publication_country_names,
    run_scenario_list,
)
from src_python.utils.io import project_path, write_dataframe


TIMELINESS_RATE_PER_YEAR = 6.0
TIMELINESS_MAX_DAILY_FLOW_FRACTION = 0.03
TIMELINESS_GRACE_MONTHS = 1.0
TIMELINESS_RECENT_WINDOW_MONTHS = 48.0
WEEKS_PER_MONTH = 365.25 / 12.0 / 7.0

FALLBACK_TIMELINESS_TARGET_DISTRIBUTION = {
    "infant_3_11m": {"dose1_recent": 0.10, "dose2_recent": 0.30, "recent": 0.60},
    "child_1_4y": {"recent": 0.80, "waned": 0.20},
    "child_5_9y": {"recent": 0.65, "waned": 0.35},
}

SCHEDULE_AWARE_AGE_BINS_MONTHS = {
    "infant_3_11m": (3.0, 12.0),
    "child_1_4y": (12.0, 60.0),
    "child_5_9y": (60.0, 120.0),
}

SCENARIO_DEFINITIONS = (
    (
        "current",
        "current",
        False,
        "Current practice with country-profile routine coverage and default age-bin vaccine-origin timing.",
    ),
    (
        "coverage_floor_only",
        "higher_child_coverage",
        False,
        "Nominal coverage floor without timeliness improvement used in the main intervention comparison.",
    ),
    (
        "timeliness_only",
        "current",
        True,
        "Current coverage with faster routine-vaccination relaxation and schedule-relative age-appropriate vaccine-origin targets.",
    ),
    (
        "coverage_floor_plus_timeliness",
        "higher_child_coverage",
        True,
        "Coverage floor plus faster routine-vaccination relaxation and schedule-relative age-appropriate vaccine-origin targets.",
    ),
)


def _set_summary_runtime(config: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    out["simulation"]["output_time_step"] = 30.0
    out["simulation"]["rtol"] = max(float(out["simulation"].get("rtol", 1e-5)), 1e-4)
    out["simulation"]["atol"] = max(float(out["simulation"].get("atol", 1e-7)), 1e-6)
    return out


def _metadata_for_timeliness_output(config: dict[str, Any], *, apply_timeliness: bool) -> dict[str, Any]:
    metadata = config.get("metadata", {})
    infant_target = (
        config.get("routine_vaccination", {})
        .get("target_origin_distribution_by_age", {})
        .get("infant_3_11m", {})
        if apply_timeliness
        else {}
    )
    return {
        "routine_age_pattern": metadata.get("routine_age_pattern", ""),
        "routine_first_shot_months": metadata.get("routine_first_shot_months", np.nan),
        "routine_dose_count": metadata.get("routine_dose_count", np.nan),
        "timeliness_definition": metadata.get("timeliness_definition", "") if apply_timeliness else "",
        "timeliness_grace_months": metadata.get("timeliness_grace_months", np.nan) if apply_timeliness else np.nan,
        "timeliness_recent_window_months": (
            metadata.get("timeliness_recent_window_months", np.nan) if apply_timeliness else np.nan
        ),
        "timeliness_infant_3_11m_dose1_recent_target": infant_target.get("dose1_recent", np.nan),
        "timeliness_infant_3_11m_dose2_recent_target": infant_target.get("dose2_recent", np.nan),
        "timeliness_infant_3_11m_dose3plus_recent_target": infant_target.get("recent", np.nan),
        "timeliness_infant_3_11m_dose3plus_waned_target": infant_target.get("waned", np.nan),
    }


def _parse_schedule_age_months(age_pattern: str | None) -> list[float]:
    if not age_pattern:
        return []
    ages: list[float] = []
    for raw_token in str(age_pattern).split(";"):
        token = raw_token.strip().upper()
        if not token:
            continue
        parts = token.split("-")
        parsed_parts: list[float] = []
        last_unit = ""
        for part in parts:
            match = re.fullmatch(r"([WMY]?)([0-9]+(?:\.[0-9]+)?)", part.strip())
            if not match:
                continue
            unit = match.group(1) or last_unit
            last_unit = unit
            value = float(match.group(2))
            if unit == "W":
                parsed_parts.append(value / WEEKS_PER_MONTH)
            elif unit == "M":
                parsed_parts.append(value)
            elif unit == "Y":
                parsed_parts.append(value * 12.0)
        if parsed_parts:
            ages.append(min(parsed_parts))
    return sorted(ages)


def _dose_origin_for_count(dose_count: int, *, age_month: float, due_age_month: float) -> str:
    if dose_count <= 1:
        return "dose1_recent"
    if dose_count == 2:
        return "dose2_recent"
    if age_month - due_age_month <= TIMELINESS_RECENT_WINDOW_MONTHS:
        return "recent"
    return "waned"


def _target_distribution_for_age_bin(
    schedule_months: list[float],
    *,
    age_start_month: float,
    age_end_month: float,
    steps: int = 480,
) -> dict[str, float]:
    if not schedule_months:
        return {}
    due_months = [age + TIMELINESS_GRACE_MONTHS for age in schedule_months]
    counts: dict[str, float] = {}
    width = age_end_month - age_start_month
    if width <= 0.0:
        return {}
    for step in range(steps):
        age_month = age_start_month + (step + 0.5) * width / steps
        due = [due_age for due_age in due_months if due_age <= age_month]
        if not due:
            continue
        origin = _dose_origin_for_count(len(due), age_month=age_month, due_age_month=due[-1])
        counts[origin] = counts.get(origin, 0.0) + 1.0
    total = sum(counts.values())
    if total <= 0.0:
        return {}
    return {origin: value / total for origin, value in counts.items()}


def timeliness_target_distribution_from_schedule(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    metadata = config.get("metadata", {})
    schedule_months = _parse_schedule_age_months(metadata.get("routine_age_pattern"))
    if not schedule_months:
        return deepcopy(FALLBACK_TIMELINESS_TARGET_DISTRIBUTION)
    targets: dict[str, dict[str, float]] = {}
    for age_group, (age_start, age_end) in SCHEDULE_AWARE_AGE_BINS_MONTHS.items():
        distribution = _target_distribution_for_age_bin(
            schedule_months,
            age_start_month=age_start,
            age_end_month=age_end,
        )
        targets[age_group] = distribution or deepcopy(FALLBACK_TIMELINESS_TARGET_DISTRIBUTION[age_group])
    return targets


def _apply_timeliness(config: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    routine = out.setdefault("routine_vaccination", {})
    routine["target_relaxation_rate_per_year"] = max(
        float(routine.get("target_relaxation_rate_per_year", 0.0)),
        TIMELINESS_RATE_PER_YEAR,
    )
    routine["max_daily_flow_fraction"] = max(
        float(routine.get("max_daily_flow_fraction", 0.0)),
        TIMELINESS_MAX_DAILY_FLOW_FRACTION,
    )
    target_by_age = deepcopy(routine.get("target_origin_distribution_by_age", {}))
    schedule_targets = timeliness_target_distribution_from_schedule(out)
    for age_group, distribution in schedule_targets.items():
        target_by_age[age_group] = deepcopy(distribution)
    routine["target_origin_distribution_by_age"] = target_by_age
    metadata = out.setdefault("metadata", {})
    metadata["timeliness_definition"] = "schedule_relative"
    metadata["timeliness_grace_months"] = TIMELINESS_GRACE_MONTHS
    metadata["timeliness_recent_window_months"] = TIMELINESS_RECENT_WINDOW_MONTHS
    metadata["timeliness_target_distribution_by_age"] = deepcopy(schedule_targets)
    return out


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    for country in publication_country_names(configs):
        for scenario_name, intervention_name, apply_timeliness, note in SCENARIO_DEFINITIONS:
            config, vaccine_name = make_intervention_config(intervention_name, country_profile=country)
            if apply_timeliness:
                config = _apply_timeliness(config)
            config = _set_summary_runtime(config)
            timeliness_metadata = _metadata_for_timeliness_output(
                config,
                apply_timeliness=bool(apply_timeliness),
            )
            scenarios.append(
                {
                    "config": config,
                    "analysis": "routine_timeliness_sensitivity",
                    "scenario": scenario_name,
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": intervention_name,
                    "metadata": {
                        "country": country,
                        "strategy": scenario_name,
                        "coverage_floor_applied": intervention_name == "higher_child_coverage",
                        "timeliness_applied": bool(apply_timeliness),
                        "timeliness_rate_per_year": TIMELINESS_RATE_PER_YEAR if apply_timeliness else np.nan,
                        "timeliness_max_daily_flow_fraction": (
                            TIMELINESS_MAX_DAILY_FLOW_FRACTION if apply_timeliness else np.nan
                        ),
                        "implementation_note": note,
                        **timeliness_metadata,
                    },
                }
            )
    return scenarios


def _summarize(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = summary.copy()
    data["strategy"] = data.get("strategy", data["scenario"])
    data["annualized_infant_cases_per_100k"] = pd.to_numeric(
        data["annualized_infant_cases_per_100k"],
        errors="coerce",
    )
    data["relative_reduction_infant_cases"] = pd.to_numeric(
        data["relative_reduction_infant_cases"],
        errors="coerce",
    )
    data["relative_reduction_total_infections"] = pd.to_numeric(
        data["relative_reduction_total_infections"],
        errors="coerce",
    )
    current_by_country = (
        data.loc[data["strategy"].eq("current"), ["country", "annualized_infant_cases_per_100k"]]
        .rename(columns={"annualized_infant_cases_per_100k": "current_infant_cases_per_100k"})
    )
    country = data.merge(current_by_country, on="country", how="left")
    country = country[
        [
            "country",
            "strategy",
            "current_infant_cases_per_100k",
            "annualized_infant_cases_per_100k",
            "relative_reduction_infant_cases",
            "relative_reduction_total_infections",
            "coverage_floor_applied",
            "timeliness_applied",
            "routine_age_pattern",
            "routine_first_shot_months",
            "routine_dose_count",
            "timeliness_definition",
            "timeliness_grace_months",
            "timeliness_recent_window_months",
            "timeliness_infant_3_11m_dose1_recent_target",
            "timeliness_infant_3_11m_dose2_recent_target",
            "timeliness_infant_3_11m_dose3plus_recent_target",
            "timeliness_infant_3_11m_dose3plus_waned_target",
            "implementation_note",
        ]
    ].rename(columns={"annualized_infant_cases_per_100k": "scenario_infant_cases_per_100k"})

    rows: list[dict[str, Any]] = []
    for strategy, group in country.groupby("strategy", sort=False):
        reductions = pd.to_numeric(group["relative_reduction_infant_cases"], errors="coerce")
        infections = pd.to_numeric(group["relative_reduction_total_infections"], errors="coerce")
        scenario_cases = pd.to_numeric(group["scenario_infant_cases_per_100k"], errors="coerce")
        current_cases = pd.to_numeric(group["current_infant_cases_per_100k"], errors="coerce")
        first = group.iloc[0]
        rows.append(
            {
                "strategy": strategy,
                "coverage_floor_applied": bool(first.get("coverage_floor_applied", False)),
                "timeliness_applied": bool(first.get("timeliness_applied", False)),
                "median_current_infant_cases_per_100k": float(current_cases.median(skipna=True)),
                "median_scenario_infant_cases_per_100k": float(scenario_cases.median(skipna=True)),
                "median_relative_reduction_infant_cases": float(reductions.median(skipna=True)),
                "iqr_relative_reduction_infant_cases": (
                    f"{reductions.quantile(0.25):.4g} to {reductions.quantile(0.75):.4g}"
                ),
                "countries_with_positive_reduction": int((reductions > 0.0).sum()),
                "median_relative_reduction_total_infections": float(infections.median(skipna=True)),
                "countries": int(group["country"].nunique()),
                "implementation_note": first.get("implementation_note", ""),
            }
        )
    table = pd.DataFrame(rows)
    return table, country


def main(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    timeseries, summary = run_scenario_list(
        _build_scenarios(configs),
        stem="routine_timeliness_sensitivity",
        reference_scenario="current",
        n_jobs=n_jobs,
    )
    table, country = _summarize(summary)
    write_dataframe(table, project_path("outputs", "tables", "routine_timeliness_sensitivity.csv"))
    write_dataframe(country, project_path("outputs", "tables", "routine_timeliness_sensitivity_country.csv"))
    return timeseries, summary, table, country


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run routine coverage-floor and timeliness sensitivity scenarios.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
