from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import (
    current_run_metadata,
    load_configs,
    write_run_metadata,
)
from src_python.validation.publication_gate import require_predictive_publication_gate

WRITTEN_ROW_COUNTS: dict[str, int] = {}


def _require_figure2_release_inputs() -> None:
    require_predictive_publication_gate()


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

INFANT_AGE_GROUPS = ("infant_0_2m", "infant_3_11m")

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

INTERVENTION_INTERVAL_AUDIT_STRATEGIES = (
    "higher_child_coverage",
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

CONSTRAINT_STRATEGIES = {
    "program_only": PROGRAM_ONLY_STRATEGIES,
    "program_plus_resistance": PROGRAM_PLUS_RESISTANCE_STRATEGIES,
    "future_product_target": FUTURE_PRODUCT_TARGET_STRATEGIES,
}

PSA_CONSTRAINT_STRATEGIES = {
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
    "maternal_immunization": "Infant-exposure reduction",
    "targeted_pep_high_risk": "Targeted high-risk PEP",
    "resistance_guided_treatment": "Resistance-guided management",
    "transmission_blocking_vaccine": "Transmission-blocking vaccine target",
    "next_generation_vaccine": "High-transmission-blocking vaccine target",
    "combined_strategy": "Combined future stress-test profile",
}

INTERVENTION_INTERVAL_LABELS = {
    "higher_child_coverage": "Nominal coverage",
    "adolescent_booster": "Adolescent booster",
    "pregnancy_tdap_scaleup": "Pregnancy Tdap",
    "cocooning_adjunct": "Close-contact adjunct",
    "maternal_immunization": "Infant exposure",
    "targeted_pep_high_risk": "Targeted PEP",
    "resistance_guided_treatment": "Resistance mgmt",
    "transmission_blocking_vaccine": "Transmission blocking",
    "next_generation_vaccine": "High-blocking target",
    "combined_strategy": "Combined",
}

STRATEGY_DOMAIN = {
    "current": "Comparator",
    "higher_child_coverage": "Program-only",
    "timeliness_only": "Program-only",
    "adolescent_booster": "Program-only",
    "pregnancy_tdap_scaleup": "Program-only",
    "cocooning_adjunct": "Program-only",
    "maternal_immunization": "Program-only composite",
    "targeted_pep_high_risk": "Program-only",
    "resistance_guided_treatment": "Resistance management",
    "transmission_blocking_vaccine": "Future product target",
    "next_generation_vaccine": "Future product target",
    "combined_strategy": "Future stress test",
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

_CALENDAR = load_configs()["baseline"]["calendar"]
ANALYSIS_START = pd.Timestamp(_CALENDAR["analysis_start_date"])
ANALYSIS_END = pd.Timestamp(_CALENDAR["analysis_end_date"])


def _inclusive_year_end(start: pd.Timestamp, years: int) -> pd.Timestamp:
    return min(start + pd.DateOffset(years=years) - pd.Timedelta(days=1), ANALYSIS_END)


HORIZON_WINDOWS = (
    ("first_5_years", ANALYSIS_START, _inclusive_year_end(ANALYSIS_START, 5)),
    ("first_10_years", ANALYSIS_START, _inclusive_year_end(ANALYSIS_START, 10)),
    ("first_15_years", ANALYSIS_START, _inclusive_year_end(ANALYSIS_START, 15)),
    (
        "excluding_initial_5_years",
        min(ANALYSIS_START + pd.DateOffset(years=5), ANALYSIS_END),
        ANALYSIS_END,
    ),
    ("full_horizon", ANALYSIS_START, ANALYSIS_END),
)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def _write(df: pd.DataFrame, path: str) -> None:
    full = ROOT / path
    full.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full, index=False)
    WRITTEN_ROW_COUNTS[path] = int(len(df))
    try:
        df.to_parquet(full.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def _read_intervention_timeseries(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "outputs/simulations/intervention_scenarios.parquet", columns=columns)


def higher_child_coverage_mechanism() -> None:
    summary = _read_csv("outputs/summaries/intervention_scenarios_summary.csv")
    summary = summary.loc[summary["scenario"].isin(["current", "higher_child_coverage"])].copy()
    wide = summary.pivot(index="country", columns="scenario", values="annualized_infant_cases_per_100k")

    ts = _read_intervention_timeseries(
        [
            "country",
            "scenario",
            "age_group",
            "strain",
            "total_infections",
            "symptomatic_cases",
            "infant_cases",
            "total_population",
            "population",
            "vaccinated_origin_infection_share",
            "waned_origin_infection_share",
            "maternal_origin_infection_share",
            "dose1_origin_infection_share",
            "dose2_origin_infection_share",
            "dose3plus_origin_infection_share",
        ]
    )
    ts = ts.loc[ts["scenario"].isin(["current", "higher_child_coverage"])].copy()

    age = (
        ts.groupby(["country", "scenario", "age_group"], as_index=False)
        .agg(total_infections=("total_infections", "sum"), symptomatic_cases=("symptomatic_cases", "sum"))
        .pivot(index=["country", "age_group"], columns="scenario")
    )
    age.columns = [f"{metric}_{scenario}" for metric, scenario in age.columns]
    age = age.reset_index()
    age["absolute_change_infections"] = age["total_infections_higher_child_coverage"] - age["total_infections_current"]
    age["relative_change_infections"] = age["absolute_change_infections"] / age["total_infections_current"].replace(0, np.nan)
    age["relative_change_symptomatic_cases"] = (
        age["symptomatic_cases_higher_child_coverage"] - age["symptomatic_cases_current"]
    ) / age["symptomatic_cases_current"].replace(0, np.nan)
    age_summary = (
        age.groupby("age_group", as_index=False)
        .agg(
            median_relative_change_infections=("relative_change_infections", "median"),
            q25_relative_change_infections=("relative_change_infections", lambda x: float(np.nanquantile(x, 0.25))),
            q75_relative_change_infections=("relative_change_infections", lambda x: float(np.nanquantile(x, 0.75))),
            median_relative_change_symptomatic_cases=("relative_change_symptomatic_cases", "median"),
            countries_with_infection_increase=("relative_change_infections", lambda x: int(np.sum(np.asarray(x) > 0))),
        )
        .sort_values("age_group")
    )
    _write(age_summary, "outputs/tables/higher_child_coverage_age_shift.csv")

    drivers = (
        age.sort_values(["country", "absolute_change_infections"], ascending=[True, False])
        .groupby("country", as_index=False)
        .first()[["country", "age_group", "absolute_change_infections", "relative_change_infections"]]
        .rename(
            columns={
                "age_group": "largest_absolute_infection_increase_age_group",
                "absolute_change_infections": "largest_absolute_infection_increase",
                "relative_change_infections": "relative_change_in_that_age_group",
            }
        )
    )
    country = wide.reset_index()
    country["relative_reduction_infant_cases"] = 1.0 - (
        country["higher_child_coverage"] / country["current"].replace(0, np.nan)
    )
    country = country.merge(drivers, on="country", how="left")
    country = country.rename(
        columns={
            "current": "current_infant_cases_per_100k",
            "higher_child_coverage": "higher_child_coverage_infant_cases_per_100k",
        }
    )
    _write(country, "outputs/tables/higher_child_coverage_country_mechanism.csv")

    origin_cols = [
        "vaccinated_origin_infection_share",
        "waned_origin_infection_share",
        "maternal_origin_infection_share",
        "dose1_origin_infection_share",
        "dose2_origin_infection_share",
        "dose3plus_origin_infection_share",
    ]
    infant = ts.loc[ts["age_group"].isin(INFANT_AGE_GROUPS)].copy()
    rows: list[dict[str, object]] = []
    for (country_name, scenario), group in infant.groupby(["country", "scenario"]):
        weights = group["total_infections"].to_numpy(dtype=float)
        denominator = float(weights.sum())
        row: dict[str, object] = {
            "country": country_name,
            "scenario": scenario,
            "infant_infections": denominator,
        }
        for col in origin_cols:
            values = group[col].fillna(0.0).to_numpy(dtype=float)
            row[col] = float(np.average(values, weights=weights)) if denominator > 0 else np.nan
        rows.append(row)
    _write(pd.DataFrame(rows), "outputs/tables/infant_vaccine_history_origin_shares.csv")


def intervention_rank_robustness() -> None:
    summary = _read_csv("outputs/summaries/intervention_scenarios_summary.csv")
    summary = summary.loc[summary["scenario"].isin(SELECTED_INTERVENTIONS)].copy()
    summary["rank"] = summary.groupby("country")["annualized_infant_cases_per_100k"].rank(method="min", ascending=True)
    best = summary.groupby("country", as_index=False)["annualized_infant_cases_per_100k"].min().rename(
        columns={"annualized_infant_cases_per_100k": "best_infant_cases_per_100k"}
    )
    summary = summary.merge(best, on="country", how="left")
    summary["within_10_percent_of_best"] = summary["annualized_infant_cases_per_100k"] <= 1.10 * summary[
        "best_infant_cases_per_100k"
    ]
    rank_summary = (
        summary.groupby("scenario", as_index=False)
        .agg(
            median_rank=("rank", "median"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            countries_ranked_first=("rank", lambda x: int(np.sum(np.asarray(x) == 1))),
            countries_within_10_percent_of_best=("within_10_percent_of_best", "sum"),
            median_infant_cases_per_100k=("annualized_infant_cases_per_100k", "median"),
        )
        .sort_values(["median_rank", "median_infant_cases_per_100k"])
    )
    rank_summary["rank_basis"] = (
        "Empirical scenario-order distribution across 10 purposively selected country profiles; not a decision-ready policy comparison."
    )
    _write(rank_summary, "outputs/tables/intervention_rank_robustness.csv")

    country_rank = summary.loc[
        :,
        [
            "country",
            "scenario",
            "rank",
            "annualized_infant_cases_per_100k",
            "relative_reduction_infant_cases",
            "within_10_percent_of_best",
        ],
    ].sort_values(["country", "rank", "scenario"])
    _write(country_rank, "outputs/tables/intervention_country_rank_heatmap.csv")


def horizon_rank_sensitivity() -> None:
    ts = _read_intervention_timeseries(
        ["country", "scenario", "calendar_date", "time", "age_group", "strain", "infant_cases", "population"]
    )
    ts = ts.loc[ts["scenario"].isin(SELECTED_INTERVENTIONS)].copy()
    ts["calendar_date"] = pd.to_datetime(ts["calendar_date"], errors="coerce")
    rows: list[pd.DataFrame] = []
    for label, start, end in HORIZON_WINDOWS:
        frame = ts.loc[ts["calendar_date"].between(start, end)].copy()
        infant_cases = frame.groupby(["country", "scenario"], as_index=False)["infant_cases"].sum()
        infant_population = (
            frame.loc[frame["age_group"].isin(INFANT_AGE_GROUPS)]
            .drop_duplicates(["country", "scenario", "time", "age_group"])
            .groupby(["country", "scenario", "time"], as_index=False)["population"]
            .sum()
            .groupby(["country", "scenario"], as_index=False)["population"]
            .mean()
            .rename(columns={"population": "mean_infant_population"})
        )
        out = infant_cases.merge(infant_population, on=["country", "scenario"], how="left")
        years = max((end - start).days / 365.0, 1e-9)
        out["analysis_window"] = label
        out["analysis_years"] = years
        out["annualized_infant_cases_per_100k"] = out["infant_cases"] / (
            years * out["mean_infant_population"].replace(0, np.nan)
        ) * 100_000.0
        out["rank"] = out.groupby("country")["annualized_infant_cases_per_100k"].rank(method="min", ascending=True)
        current = out.loc[out["scenario"].eq("current"), ["country", "annualized_infant_cases_per_100k"]].rename(
            columns={"annualized_infant_cases_per_100k": "current_infant_cases_per_100k"}
        )
        out = out.merge(current, on="country", how="left")
        out["relative_reduction_infant_cases"] = 1.0 - out["annualized_infant_cases_per_100k"] / out[
            "current_infant_cases_per_100k"
        ].replace(0, np.nan)
        rows.append(out)
    combined = pd.concat(rows, ignore_index=True)
    _write(combined, "outputs/tables/intervention_horizon_rank_sensitivity.csv")

    rank_summary = (
        combined.groupby(["analysis_window", "scenario"], as_index=False)
        .agg(
            median_rank=("rank", "median"),
            countries_ranked_first=("rank", lambda x: int(np.sum(np.asarray(x) == 1))),
            median_relative_reduction_infant_cases=("relative_reduction_infant_cases", "median"),
        )
        .sort_values(["analysis_window", "median_rank"])
    )
    _write(rank_summary, "outputs/tables/intervention_horizon_rank_summary.csv")


def _age_stratified_infant_frame() -> pd.DataFrame:
    ts = _read_intervention_timeseries(
        [
            "country",
            "scenario",
            "calendar_date",
            "time",
            "age_group",
            "strain",
            "infant_cases",
            "infant_infections",
            "population",
        ]
    )
    ts = ts.loc[ts["scenario"].isin(SELECTED_INTERVENTIONS) & ts["age_group"].isin(INFANT_AGE_GROUPS)].copy()
    ts["calendar_date"] = pd.to_datetime(ts["calendar_date"], errors="coerce")
    return ts


def _summarise_age_stratified_window(frame: pd.DataFrame, years: float) -> pd.DataFrame:
    burden = (
        frame.groupby(["country", "scenario", "age_group"], as_index=False)
        .agg(
            infant_cases=("infant_cases", "sum"),
            infant_infections=("infant_infections", "sum"),
        )
    )
    population = (
        frame.drop_duplicates(["country", "scenario", "time", "age_group"])
        .groupby(["country", "scenario", "age_group"], as_index=False)["population"]
        .mean()
        .rename(columns={"population": "mean_infant_age_population"})
    )
    out = burden.merge(population, on=["country", "scenario", "age_group"], how="left")
    denominator = years * out["mean_infant_age_population"].replace(0, np.nan)
    out["analysis_years"] = years
    out["annualized_infant_cases_per_100k"] = out["infant_cases"] / denominator * 100_000.0
    out["annualized_infant_infections_per_100k"] = out["infant_infections"] / denominator * 100_000.0
    current = out.loc[
        out["scenario"].eq("current"),
        ["country", "age_group", "annualized_infant_cases_per_100k"],
    ].rename(columns={"annualized_infant_cases_per_100k": "current_infant_cases_per_100k"})
    out = out.merge(current, on=["country", "age_group"], how="left")
    out["relative_reduction_infant_cases"] = 1.0 - out["annualized_infant_cases_per_100k"] / out[
        "current_infant_cases_per_100k"
    ].replace(0, np.nan)
    return out


def infant_age_split_intervention() -> None:
    ts = _age_stratified_infant_frame()
    start = ANALYSIS_START
    end = ANALYSIS_END
    frame = ts.loc[ts["calendar_date"].between(start, end)].copy()
    years = max((end - start).days / 365.0, 1e-9)
    out = _summarise_age_stratified_window(frame, years)
    out = out.sort_values(["country", "age_group", "scenario"])
    _write(out, "outputs/tables/infant_age_split_intervention.csv")


def infant_age_split_horizon_sensitivity() -> None:
    ts = _age_stratified_infant_frame()
    rows: list[pd.DataFrame] = []
    for label, start, end in HORIZON_WINDOWS:
        frame = ts.loc[ts["calendar_date"].between(start, end)].copy()
        years = max((end - start).days / 365.0, 1e-9)
        out = _summarise_age_stratified_window(frame, years)
        out["analysis_window"] = label
        out["rank"] = out.groupby(["country", "age_group"])["annualized_infant_cases_per_100k"].rank(
            method="min", ascending=True
        )
        rows.append(out)
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["analysis_window", "country", "age_group", "rank", "scenario"])
    _write(combined, "outputs/tables/infant_age_split_horizon_sensitivity.csv")


def intervention_rank_stability_diagnostics() -> None:
    full = _read_csv("outputs/tables/intervention_country_rank_heatmap.csv")
    horizon = _read_csv("outputs/tables/intervention_horizon_rank_sensitivity.csv")
    infant_age = _read_csv("outputs/tables/infant_age_split_horizon_sensitivity.csv")
    rows: list[dict[str, object]] = []
    for scenario in SELECTED_INTERVENTIONS:
        full_s = full.loc[full["scenario"].eq(scenario)]
        horizon_s = horizon.loc[horizon["scenario"].eq(scenario)]
        age_s = infant_age.loc[infant_age["scenario"].eq(scenario)]
        top2_age = int(np.sum(age_s["rank"].to_numpy(dtype=float) <= 2))
        age_cells = int(len(age_s))
        first_age = int(np.sum(age_s["rank"].to_numpy(dtype=float) == 1))
        positive_age = int(np.sum(age_s["relative_reduction_infant_cases"].to_numpy(dtype=float) > 0))
        if age_cells and first_age / age_cells >= 0.50:
            interpretation = "Most stable lowest-burden scenario across country, horizon, and infant-age diagnostics."
        elif age_cells and top2_age / age_cells >= 0.50:
            interpretation = "Often near the lowest modeled burden, but not consistently ordered first."
        elif age_cells and positive_age / age_cells >= 0.50:
            interpretation = "Usually lower burden than current practice, but ordering is horizon- and age-stratum-dependent."
        else:
            interpretation = "Low-benefit or unstable scenario in these deterministic diagnostics."
        rows.append(
            {
                "scenario": scenario,
                "full_horizon_median_rank": float(full_s["rank"].median()),
                "full_horizon_countries_ranked_first": int(np.sum(full_s["rank"].to_numpy(dtype=float) == 1)),
                "full_horizon_countries_ranked_top_two": int(np.sum(full_s["rank"].to_numpy(dtype=float) <= 2)),
                "analysis_window_cells": int(len(horizon_s)),
                "analysis_window_cells_ranked_first": int(np.sum(horizon_s["rank"].to_numpy(dtype=float) == 1)),
                "analysis_window_cells_ranked_top_two": int(np.sum(horizon_s["rank"].to_numpy(dtype=float) <= 2)),
                "analysis_window_cells_positive_reduction": int(
                    np.sum(horizon_s["relative_reduction_infant_cases"].to_numpy(dtype=float) > 0)
                ),
                "infant_age_window_cells": age_cells,
                "infant_age_window_cells_ranked_first": first_age,
                "infant_age_window_cells_ranked_top_two": top2_age,
                "infant_age_window_cells_positive_reduction": positive_age,
                "median_infant_age_window_reduction": float(age_s["relative_reduction_infant_cases"].median()),
                "minimum_infant_age_window_reduction": float(age_s["relative_reduction_infant_cases"].min()),
                "maximum_infant_age_window_reduction": float(age_s["relative_reduction_infant_cases"].max()),
                "rank_stability_interpretation": interpretation,
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["infant_age_window_cells_ranked_first", "infant_age_window_cells_ranked_top_two"], ascending=False
    )
    _write(out, "outputs/tables/intervention_rank_stability_diagnostics.csv")


def deterministic_event_scale_diagnostics() -> None:
    summary = _read_csv("outputs/summaries/intervention_scenarios_summary.csv")
    summary = summary.loc[summary["scenario"].isin(SELECTED_INTERVENTIONS)].copy()
    summary["annual_total_infections_count"] = summary["total_infections"] / summary["analysis_years"].replace(0, np.nan)
    summary["annual_reported_cases_count"] = summary["total_reported_cases"] / summary["analysis_years"].replace(0, np.nan)
    summary["annual_infant_cases_count"] = summary["total_infant_cases"] / summary["analysis_years"].replace(0, np.nan)

    def flag(row: pd.Series) -> str:
        if row["annual_total_infections_count"] < 1000:
            return "Low aggregate event count; deterministic persistence is a strong assumption."
        if row["annual_infant_cases_count"] < 25:
            return "Low infant-event count; modeled infant cases are especially sensitive to stochastic variation."
        if row["annual_reported_cases_count"] < 50:
            return "Low reported-event count; surveillance stochasticity is likely material."
        return "Aggregate event counts are not near extinction, but stochastic clustering remains unmodeled."

    out = summary.loc[
        :,
        [
            "country",
            "scenario",
            "analysis_years",
            "total_infections",
            "total_reported_cases",
            "total_infant_cases",
            "annualized_infant_cases_per_100k",
            "n_epidemic_peaks",
            "resistant_fraction_end",
            "annual_total_infections_count",
            "annual_reported_cases_count",
            "annual_infant_cases_count",
        ],
    ].copy()
    out["event_scale_flag"] = out.apply(flag, axis=1)
    out = out.sort_values(["scenario", "annual_total_infections_count", "country"])
    _write(out, "outputs/tables/deterministic_event_scale_diagnostics.csv")


def resistance_parameter_justification() -> None:
    rows = [
        {
            "parameter_group": "Country-specific starting resistant fraction",
            "baseline_value": "Latest admissible country timeline anchor; fixed scenarios also used 5%, 30%, 70%, and 95%",
            "explored_range_or_scenarios": "Country timeline; fixed low, moderate, high, and very-high resistance scenarios",
            "source_or_anchor": "Country resistance timeline assembled from China, Japan, Australia, Americas, Europe, and low-anchor surveillance reports through the evidence lock",
            "rationale": "Separates observed or conservative starting strain composition from subsequent modeled selection dynamics.",
            "expected_direction_of_bias": "Higher starting resistant fraction increases resistant burden and the apparent value of resistance-guided management; lower anchors delay resistant dominance.",
            "residual_caveat": "Resistance sampling is heterogeneous across countries and years; anchors are not a globally representative surveillance system.",
        },
        {
            "parameter_group": "Resistant-strain relative fitness (fitness_R)",
            "baseline_value": "1.00 (fitness neutral)",
            "explored_range_or_scenarios": "0.70-1.25 grid and selected-parameter sensitivity range; selected narrative contrasts at 0.85, 1.00, and 1.15",
            "source_or_anchor": "Rapid MRBP expansion and international spread without a demonstrated transmission penalty; rationale summarized in the resistance-parameter justification table.",
            "rationale": "Avoids assuming a persistent fitness cost when epidemiologic trajectories in China, Japan, and Australia do not rule out neutral or above-neutral fitness.",
            "expected_direction_of_bias": "Lower fitness reduces projected resistant fraction and resistance-guided management benefit; higher fitness accelerates replacement and increases resistant burden.",
            "residual_caveat": "Fitness is represented as one transmission scalar and may vary with vaccine history, treatment pressure, strain background, and host immunity.",
        },
        {
            "parameter_group": "Sensitive-strain treatment effect",
            "baseline_value": "Infectious-duration reduction 0.20; infectiousness reduction 0.15",
            "explored_range_or_scenarios": "Treatment implementation and resistance-mechanism decomposition scenarios",
            "source_or_anchor": "CDC pertussis treatment/PEP guidance and model scenario assumptions",
            "rationale": "Represents early macrolide benefit for susceptible infections without assuming treatment fully blocks transmission.",
            "expected_direction_of_bias": "Stronger sensitive-strain treatment benefit increases selection pressure favoring resistant strains; weaker benefit reduces modeled treatment-mediated selection.",
            "residual_caveat": "Real-world treatment effect depends on timing, diagnosis, adherence, and clinical practice, none of which are explicitly modeled as individual pathways.",
        },
        {
            "parameter_group": "Resistant-strain treatment effect under standard macrolide practice",
            "baseline_value": "Infectious-duration reduction 0.10; infectiousness reduction 0.05",
            "explored_range_or_scenarios": "Equalized treatment counterfactual; resistance-guided management-pathway alternative",
            "source_or_anchor": "Resistance-guided management scenario assumption informed by macrolide resistance biology and treatment guidance",
            "rationale": "Allows resistant infections to receive less benefit from standard macrolide management while testing whether that differential drives replacement.",
            "expected_direction_of_bias": "Lower resistant treatment benefit increases resistant burden and infant cases; equalizing treatment effects lowers selection for resistance.",
            "residual_caveat": "The model does not identify strain-specific treatment effect from patient-level outcome data.",
        },
        {
            "parameter_group": "Postexposure prophylaxis (PEP) coverage",
            "baseline_value": "Household-contact coverage 0.30",
            "explored_range_or_scenarios": "0.05-0.60 in sensitivity analysis and selected-parameter sensitivity multiplier; implementation scenarios vary PEP reach",
            "source_or_anchor": "CDC/PAHO-style public health PEP guidance translated into scenario coverage assumptions",
            "rationale": "Represents partial household/contact implementation for prioritized contacts.",
            "expected_direction_of_bias": "Higher PEP reach amplifies any strain-specific PEP effectiveness differential; lower PEP reach weakens PEP-mediated selection and management benefit.",
            "residual_caveat": "PEP targeting, timing, adherence, and contact tracing are not explicit household processes in the deterministic model.",
        },
        {
            "parameter_group": "PEP effectiveness by strain",
            "baseline_value": "Sensitive 0.70; resistant 0.10 under standard macrolide PEP",
            "explored_range_or_scenarios": "Resistant PEP effectiveness 0.00-0.50; equalized PEP and treatment+PEP decomposition scenarios",
            "source_or_anchor": "Macrolide-resistance mechanism, clinical guidance, and resistance-management scenario assumptions",
            "rationale": "Tests whether strain-specific prophylaxis failure can plausibly create selection pressure under standard macrolide PEP.",
            "expected_direction_of_bias": "A larger sensitive-resistant PEP gap favors resistant strains; equalized PEP effectiveness markedly lowers projected end resistant fraction.",
            "residual_caveat": "PEP effectiveness is not estimated from strain-specific household trial data; results are stress tests conditional on PEP reach and timing.",
        },
        {
            "parameter_group": "Resistance-guided management scenario",
            "baseline_value": "Symptomatic treatment rate 0.065; resistant infectious-duration reduction 0.45; resistant infectiousness reduction 0.35; resistant PEP effectiveness 0.45",
            "explored_range_or_scenarios": "Treatment/PEP implementation scenarios and selected-parameter sensitivity uptake multiplier",
            "source_or_anchor": "CDC treatment and antibiotic-resistance guidance translated into a resistance-guided testing-and-alternative-treatment scenario",
            "rationale": "Represents improved recognition of resistance and use of effective alternatives plus an assumed improvement in prophylaxis effectiveness.",
            "expected_direction_of_bias": "Higher uptake or assumed PEP improvement increases projected benefit; low testing reach and uptake reduce or delay benefit.",
            "residual_caveat": "Testing availability, turnaround time, clinician suspicion, drug tolerability, and adherence are not modeled explicitly.",
        },
        {
            "parameter_group": "Resistant importation",
            "baseline_value": "Low-level importation enabled; default rate 0.20 per 100 000 persons/year with country/scenario resistant fraction",
            "explored_range_or_scenarios": "Resistance mechanism decomposition separates ongoing importation from fitness and treatment/PEP differentials",
            "source_or_anchor": "Persistence/reintroduction assumption anchored to observed international spread",
            "rationale": "Prevents deterministic extinction of rare resistant strains while allowing decomposition of whether importation alone drives high end fractions.",
            "expected_direction_of_bias": "Higher importation affects persistence and timing; mechanism decomposition suggests it is not the main driver of near-complete replacement in the main runs.",
            "residual_caveat": "Importation is modelled as smooth low-level reintroduction; stochastic travel and outbreak-linked introductions are outside this term.",
        },
    ]
    _write(pd.DataFrame(rows), "outputs/tables/resistance_parameter_justification.csv")


def limitation_diagnostic_map() -> None:
    rows = [
        {
            "limitation_domain": "Infant outcomes without direct age-specific calibration",
            "added_or_existing_diagnostic": "Overall calibration fit, fitted reporting gradients, infant contact sensitivity, event-scale diagnostics, and routine-timeliness, infant-age/window, and external age-pattern weighted ordering diagnostics.",
            "supplement_location": "supplementary methods, tables S12 and S13, and figures S6 and S7",
            "residual_interpretation": "Infant estimates are conditional model outputs; age-pattern weighting is a partial external consistency diagnostic, not full recalibration.",
        },
        {
            "limitation_domain": "Strategy-profile ordering under selected-parameter sensitivity",
            "added_or_existing_diagnostic": "Country-level order positions, analysis-window order positions, infant-age/window order positions, strategy-ordering summary, Figure 2A-C decision-framework source data, retained regret source data, and selected-parameter deterministic strategy-ordering diagnostics.",
            "supplement_location": "Figure 2A-C and figures S5 and S6",
            "residual_interpretation": "Order-position frequencies are conditional on the selected deterministic sensitivity ranges and do not include costs, feasibility, or equity weights.",
        },
        {
            "limitation_domain": "Deterministic dynamics without stochastic extinction or superspreading",
            "added_or_existing_diagnostic": "Event-scale diagnostics identify low-event cells where deterministic persistence assumptions matter most; a small individual stochastic toy model illustrates contact-clustering sensitivity.",
            "supplement_location": "figure S7",
            "residual_interpretation": "Near-zero burdens and low-event cells should be read as deterministic thresholds, not stochastic elimination probabilities.",
        },
        {
            "limitation_domain": "No explicit household clustering, contact tracing, or adherence model",
            "added_or_existing_diagnostic": "Resistance-guided management implementation sensitivity, infant contact-matrix sensitivity, maternal package component decomposition, and individual stochastic contact-clustering illustration.",
            "supplement_location": "figures S5, S7, and S10",
            "residual_interpretation": "Age-structured proxy diagnostics do not replace household or contact-tracing simulations.",
        },
        {
            "limitation_domain": "Macrolide-resistant strain dynamics depend on fitness and management assumptions",
            "added_or_existing_diagnostic": "Resistance mechanism decomposition, fitness grids, hindcast plausibility checks, treatment/PEP implementation sensitivity, vaccine-infectiousness thresholds, and resistance-parameter justification.",
            "supplement_location": "figures S9, S10, and S11, and table S16",
            "residual_interpretation": "Resistance trajectories are stress tests of selection mechanisms under specified assumptions.",
        },
        {
            "limitation_domain": "No costs, utility weights, feasibility, or equity weights",
            "added_or_existing_diagnostic": "Exploratory burden translation from model deaths and symptomatic cases, with hospitalization imputed from transparent scenario assumptions.",
            "supplement_location": "Repository health-utility output tables",
            "residual_interpretation": "Costs, decision thresholds, discounting, feasibility constraints, and equity weights remain outside this exploratory burden translation.",
        },
        {
            "limitation_domain": "In-development vaccine products cannot be treated as available policies",
            "added_or_existing_diagnostic": "Pipeline-to-mechanism mapping for intranasal BPZE1, OMV-based platforms, genetically detoxified recombinant aP vaccines, and new multicomponent aP candidates.",
            "supplement_location": "table S17",
            "residual_interpretation": "Candidate products were represented through mechanism profiles and sensitivity ranges for mechanism comparison.",
        },
    ]
    _write(pd.DataFrame(rows), "outputs/tables/limitation_diagnostic_map.csv")


def sensitivity_correlations() -> None:
    df = _read_csv("outputs/summaries/sensitivity_runs_summary.csv")
    legacy_params = [
        "VE_sus",
        "VE_sym",
        "VE_inf",
        "VE_dur",
        "infectious_duration_symptomatic",
        "infectious_duration_asymptomatic",
        "waning_rate_vaccine",
        "waning_rate_natural",
        "relative_infectiousness_asymptomatic",
        "seasonal_amplitude",
        "multi_year_amplitude",
        "treatment_rate_symptomatic",
        "PEP_coverage",
        "PEP_effectiveness_resistant",
        "fitness_R",
        "reporting_rate_multiplier",
    ]
    configs = load_configs()
    registry = configs.get("parameter_distributions", {})
    global_settings = registry.get("global_sensitivity", {}) if isinstance(registry, dict) else {}
    configured_specs = global_settings.get("parameters", {})
    configured_params = [name for name in configured_specs if name in df.columns]
    schema = pd.to_numeric(
        df.get("uncertainty_schema_version", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    uses_evidence_registry = bool(schema.ge(1).any())

    # Existing checked-in summaries predate the evidence-prior registry.  Keep
    # the table generator usable with them, while selecting the new named
    # columns automatically after the sensitivity analysis is rerun.
    if uses_evidence_registry:
        missing = sorted(set(configured_specs).difference(configured_params))
        if missing:
            raise ValueError(
                "Evidence-prior sensitivity summary is missing configured parameter columns: "
                f"{missing}"
            )
        specs = {name: configured_specs[name] for name in configured_params}
        sample_design = "inverse-CDF Latin-hypercube"
    else:
        params = [name for name in legacy_params if name in df.columns]
        legacy_specs = configs.get("sensitivity", {}).get("parameters", {})
        specs = {
            name: {
                **legacy_specs.get(name, {}),
                "distribution": "uniform",
                "evidence_class": "legacy_range",
            }
            for name in params
        }
        sample_design = "legacy uniform Latin-hypercube"

    outcome_columns = {
        "total_child_adolescent_cases": "annualized_child_adolescent_cases_per_100k",
        "total_infant_cases": "annualized_infant_cases_per_100k",
        "total_reported_cases": "annualized_reported_cases_per_100k",
        "total_infant_deaths": "annualized_infant_deaths_per_100k",
        "total_child_adolescent_hospitalizations": (
            "annualized_child_adolescent_hospitalizations_per_100k"
        ),
    }
    grouped: dict[str, list[str]] = {}
    default_summary_outcome = (
        "total_child_adolescent_cases" if uses_evidence_registry else "total_infant_cases"
    )
    for name, spec in specs.items():
        summary_outcome = str(spec.get("outcome", default_summary_outcome))
        outcome = outcome_columns.get(summary_outcome, summary_outcome)
        if outcome not in df.columns:
            raise KeyError(f"Sensitivity outcome {outcome!r} for {name!r} is not in summary")
        grouped.setdefault(outcome, []).append(name)

    rows = []
    for outcome, params in grouped.items():
        frame = df.loc[:, params + [outcome]].dropna().copy()
        ranks = frame.apply(stats.rankdata)
        y = ranks[outcome].to_numpy(dtype=float)
        xmat = ranks[params].to_numpy(dtype=float)
        for idx, param in enumerate(params):
            x = xmat[:, idx]
            other = np.delete(xmat, idx, axis=1)
            if other.shape[1]:
                design = np.column_stack([np.ones(len(other)), other])
                x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
                y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
                prcc = float(stats.pearsonr(x_resid, y_resid).statistic)
            else:
                # With a single observation-layer factor there are no other
                # covariates to partial out, so PRCC reduces to rank correlation.
                prcc = float(stats.pearsonr(x, y).statistic)
            pearson = float(stats.pearsonr(frame[param], frame[outcome]).statistic)
            spearman = float(stats.spearmanr(frame[param], frame[outcome]).statistic)
            spec = specs[param]
            rows.append(
                {
                    "parameter": param,
                    "outcome": outcome,
                    "distribution": spec.get("distribution", "uniform"),
                    "evidence_class": spec.get("evidence_class", "not_specified"),
                    "pearson_r": pearson,
                    "spearman_r": spearman,
                    "partial_rank_correlation": prcc,
                    "absolute_prcc": abs(prcc),
                    "screening_note": (
                        f"Exploratory {len(frame)}-sample {sample_design} screening; "
                        "not variance decomposition."
                    ),
                }
            )
    out = pd.DataFrame(rows).sort_values("absolute_prcc", ascending=False)
    _write(out, "outputs/tables/sensitivity_correlation_screening.csv")


def veinf_thresholds_against_comparators() -> None:
    grid = _read_csv("outputs/summaries/veinf_resistance_grid_summary.csv")
    intervention = _read_csv("outputs/summaries/intervention_scenarios_summary.csv")
    comparator_map = {
        "infant_exposure_reduction_strategy": "maternal_immunization",
        "resistance_guided_treatment": "resistance_guided_treatment",
    }
    selected_prevalence = [0.0, 0.5, 1.0]
    rows: list[dict[str, object]] = []
    for prevalence in selected_prevalence:
        grid_p = grid.loc[np.isclose(grid["grid_resistance_prevalence"], prevalence)].copy()
        for comparator_label, scenario in comparator_map.items():
            comp = intervention.loc[intervention["scenario"].eq(scenario), ["country", "annualized_infant_cases_per_100k"]]
            comp = comp.rename(columns={"annualized_infant_cases_per_100k": "comparator_infant_cases_per_100k"})
            merged = grid_p.merge(comp, on="country", how="inner")
            thresholds = []
            for country, group in merged.groupby("country"):
                eligible = group.loc[
                    group["annualized_infant_cases_per_100k"].le(group["comparator_infant_cases_per_100k"])
                ]
                thresholds.append(float(eligible["grid_VE_inf"].min()) if not eligible.empty else np.nan)
            finite = [value for value in thresholds if np.isfinite(value)]
            rows.append(
                {
                    "comparator": comparator_label,
                    "resistance_prevalence": prevalence,
                    "median_minimum_VE_inf": float(np.nanmedian(thresholds)) if finite else np.nan,
                    "countries_reaching_comparator": int(len(finite)),
                    "countries_evaluated": int(len(thresholds)),
                    "threshold_basis": "VE_inf-only grid; VE_sus and VE_dur held at the grid baseline.",
                }
            )

    base_prevalence = 0.5
    grid_p = grid.loc[np.isclose(grid["grid_resistance_prevalence"], base_prevalence)].copy()
    reductions_by_ve: dict[float, list[float]] = {}
    for country, group in grid_p.groupby("country"):
        group = group.sort_values("grid_VE_inf")
        baseline = group.loc[np.isclose(group["grid_VE_inf"], 0.2), "annualized_infant_cases_per_100k"]
        if baseline.empty:
            continue
        baseline_value = float(baseline.iloc[0])
        for _, row in group.iterrows():
            ve_inf = float(row["grid_VE_inf"])
            reduction = 1.0 - float(row["annualized_infant_cases_per_100k"]) / max(baseline_value, 1e-9)
            reductions_by_ve.setdefault(ve_inf, []).append(reduction)
    for target in (0.25, 0.50, 0.75):
        eligible = [
            (ve_inf, reductions)
            for ve_inf, reductions in sorted(reductions_by_ve.items())
            if float(np.nanmedian(reductions)) >= target
        ]
        if eligible:
            selected_ve, selected_reductions = eligible[0]
            countries_reaching = int(np.sum(np.asarray(selected_reductions) >= target))
            countries_evaluated = int(len(selected_reductions))
            median_minimum = float(selected_ve)
        else:
            countries_reaching = 0
            countries_evaluated = int(max((len(v) for v in reductions_by_ve.values()), default=0))
            median_minimum = np.nan
        rows.append(
            {
                "comparator": f"{int(target * 100)}% reduction vs VE_inf_0.20",
                "resistance_prevalence": base_prevalence,
                "median_minimum_VE_inf": median_minimum,
                "countries_reaching_comparator": countries_reaching,
                "countries_evaluated": countries_evaluated,
                "threshold_basis": "Median across evaluated profiles on the VE_inf-only grid at 50% starting resistance prevalence.",
            }
        )
    _write(pd.DataFrame(rows), "outputs/tables/veinf_comparator_thresholds.csv")


def _resistant_infections_per_100k(df: pd.DataFrame) -> pd.Series:
    denominator = df["analysis_years"].replace(0, np.nan) * df["total_population"].replace(0, np.nan)
    return df["resistant_infections"] / denominator * 100_000.0


def _relative_reduction(values: pd.Series, baseline: pd.Series) -> pd.Series:
    out = 1.0 - values / baseline.replace(0, np.nan)
    zero_baseline = baseline.abs() <= 1e-12
    out.loc[zero_baseline & (values.abs() <= 1e-12)] = 0.0
    return out


def _transmission_blocking_profile() -> pd.DataFrame:
    vaccine = _read_csv("outputs/summaries/vaccine_scenarios_summary.csv")
    current = vaccine.loc[vaccine["scenario"].eq("symptom_protective")].copy()
    current = current.rename(
        columns={
            "annualized_infant_cases_per_100k": "current_infant_cases_per_100k",
            "annualized_infections_per_100k": "current_infections_per_100k",
            "annualized_reported_cases_per_100k": "current_reported_cases_per_100k",
            "resistant_infections": "current_resistant_infections",
        }
    )[
        [
            "country",
            "current_infant_cases_per_100k",
            "current_infections_per_100k",
            "current_reported_cases_per_100k",
            "current_resistant_infections",
        ]
    ]
    out = vaccine.loc[vaccine["scenario"].eq("transmission_blocking")].copy()
    out = out.merge(current, on="country", how="left")
    out["scenario"] = "transmission_blocking_vaccine"
    out["strategy"] = "transmission_blocking_vaccine"
    out["intervention"] = "transmission_blocking_vaccine"
    out["relative_reduction_infant_cases"] = 1.0 - out["annualized_infant_cases_per_100k"] / out[
        "current_infant_cases_per_100k"
    ].replace(0, np.nan)
    out["relative_reduction_total_infections"] = 1.0 - out["annualized_infections_per_100k"] / out[
        "current_infections_per_100k"
    ].replace(0, np.nan)
    out["relative_reduction_reported_cases"] = 1.0 - out["annualized_reported_cases_per_100k"] / out[
        "current_reported_cases_per_100k"
    ].replace(0, np.nan)
    out["relative_reduction_resistant_infections"] = _relative_reduction(
        out["resistant_infections"], out["current_resistant_infections"]
    )
    return out.drop(columns=[col for col in out.columns if col.startswith("current_")])


def _optimization_burden_frame() -> pd.DataFrame:
    intervention = _read_csv("outputs/summaries/intervention_scenarios_summary.csv")
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

    timeliness = _read_csv("outputs/summaries/routine_timeliness_sensitivity_summary.csv")
    timeliness = timeliness.loc[timeliness["strategy"].eq("timeliness_only")].copy()
    timeliness["scenario"] = "timeliness_only"
    timeliness["intervention"] = "timeliness_only"

    combined = pd.concat([intervention, timeliness, _transmission_blocking_profile()], ignore_index=True, sort=False)
    combined["strategy"] = combined["scenario"]
    combined["strategy_label"] = combined["strategy"].map(STRATEGY_LABELS)
    combined["decision_domain"] = combined["strategy"].map(STRATEGY_DOMAIN)
    combined["implementation_intensity"] = combined["strategy"].map(IMPLEMENTATION_INTENSITY).astype(float)
    combined["annualized_resistant_infections_per_100k"] = _resistant_infections_per_100k(combined)

    current = combined.loc[combined["strategy"].eq("current")].copy()
    current = current.rename(
        columns={
            "annualized_resistant_infections_per_100k": "current_resistant_infections_per_100k",
            "annualized_infant_cases_per_100k": "current_infant_cases_per_100k",
            "annualized_infections_per_100k": "current_infections_per_100k",
            "annualized_reported_cases_per_100k": "current_reported_cases_per_100k",
        }
    )[
        [
            "country",
            "current_resistant_infections_per_100k",
            "current_infant_cases_per_100k",
            "current_infections_per_100k",
            "current_reported_cases_per_100k",
        ]
    ]
    combined = combined.merge(current, on="country", how="left")
    combined["relative_reduction_resistant_infections"] = _relative_reduction(
        combined["annualized_resistant_infections_per_100k"],
        combined["current_resistant_infections_per_100k"],
    )
    combined["relative_reduction_infant_cases"] = 1.0 - combined["annualized_infant_cases_per_100k"] / combined[
        "current_infant_cases_per_100k"
    ].replace(0, np.nan)
    combined["relative_reduction_total_infections"] = 1.0 - combined["annualized_infections_per_100k"] / combined[
        "current_infections_per_100k"
    ].replace(0, np.nan)
    combined["relative_reduction_reported_cases"] = 1.0 - combined[
        "annualized_reported_cases_per_100k"
    ] / combined["current_reported_cases_per_100k"].replace(0, np.nan)
    return combined


def _is_non_dominated(frame: pd.DataFrame) -> pd.Series:
    benefits = frame[["relative_reduction_infant_cases", "relative_reduction_resistant_infections"]].fillna(-np.inf)
    intensities = pd.to_numeric(frame["implementation_intensity"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    non_dominated = []
    values = benefits.to_numpy(dtype=float)
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
        non_dominated.append(not dominated)
    return pd.Series(non_dominated, index=frame.index)


def _constraint_frontier_points(burden: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for constraint, strategies in CONSTRAINT_STRATEGIES.items():
        frame = burden.loc[burden["strategy"].isin(strategies)].copy()
        frame["optimization_constraint"] = constraint
        for _country, idx in frame.groupby("country").groups.items():
            frame.loc[idx, "non_dominated_outcome"] = _is_non_dominated(frame.loc[idx])
            frame.loc[idx, "infant_case_rank_within_constraint"] = frame.loc[idx, "annualized_infant_cases_per_100k"].rank(
                method="min", ascending=True
            )
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    keep = [
        "optimization_constraint",
        "country",
        "strategy",
        "strategy_label",
        "decision_domain",
        "implementation_intensity",
        "annualized_infant_cases_per_100k",
        "annualized_resistant_infections_per_100k",
        "relative_reduction_infant_cases",
        "relative_reduction_resistant_infections",
        "relative_reduction_total_infections",
        "relative_reduction_reported_cases",
        "non_dominated_outcome",
        "infant_case_rank_within_constraint",
    ]
    return out.loc[:, keep].sort_values(["optimization_constraint", "country", "implementation_intensity", "strategy"])


def _current_rate_interval_frame(outcome: str, interval_source: str = "combined") -> pd.DataFrame:
    overlay_path = ROOT / "outputs/summaries/bayesian_stochastic_overlay_intervals_summary.csv"
    if overlay_path.exists():
        intervals = pd.read_csv(overlay_path)
        if interval_source == "parameter":
            center_col = "propagated_uncertainty_median"
            low_col = "propagated_uncertainty_interval_low"
            high_col = "propagated_uncertainty_interval_high"
        else:
            center_col = "combined_prediction_interval_median"
            low_col = "combined_prediction_interval_low"
            high_col = "combined_prediction_interval_high"
    else:
        intervals = _read_csv("outputs/summaries/bayesian_uncertainty_intervals_summary.csv")
        center_col = "uncertainty_median"
        low_col = "uncertainty_interval_low"
        high_col = "uncertainty_interval_high"

    missing = {low_col, high_col}.difference(intervals.columns)
    if missing:
        raise ValueError(f"Uncertainty interval source {interval_source!r} lacks columns: {sorted(missing)}")

    keep_cols = ["country", low_col, high_col]
    if center_col is not None:
        keep_cols.append(center_col)
    intervals = intervals.loc[intervals["outcome"].eq(outcome), keep_cols].copy()
    if intervals.empty:
        raise ValueError(f"No current-practice uncertainty interval found for outcome: {outcome}")
    intervals = intervals.rename(
        columns={
            low_col: "current_rate_q025",
            high_col: "current_rate_q975",
            center_col: "current_rate_interval_center",
        }
    )
    if "current_rate_interval_center" not in intervals.columns:
        intervals["current_rate_interval_center"] = (
            intervals["current_rate_q025"] + intervals["current_rate_q975"]
        ) / 2.0
    return intervals


def _paired_programme_primary_interval_audit(
    burden: pd.DataFrame,
    spec: dict[str, object],
) -> pd.DataFrame:
    _require_figure2_release_inputs()
    interval_path = ROOT / str(spec["paired_interval_path"])
    if not interval_path.exists():
        raise FileNotFoundError(
            f"Figure 2c paired conditional uncertainty intervals are required but missing: {interval_path}. "
            "Run python -m src_python.simulation.run_figure2c_paired_uncertainty first."
        )
    paired = pd.read_csv(interval_path)
    required = {
        "country",
        "strategy",
        "scenario_key",
        "scenario_label",
        "outcome",
        "reduction_median",
        "reduction_q025",
        "reduction_q975",
        "current_rate_q025",
        "current_rate_q975",
        "intervention_rate_q025",
        "intervention_rate_q975",
        "uncertainty_draws",
        "interval_type",
        "interval_basis",
    }
    missing = required.difference(paired.columns)
    if missing:
        raise ValueError(f"Figure 2c paired interval table lacks columns: {sorted(missing)}")
    interval_types = set(paired["interval_type"].dropna().astype(str))
    if interval_types != {"95% conditional uncertainty interval"}:
        raise ValueError(
            "Figure 2c requires the exact conditional interval type; "
            f"observed={sorted(interval_types)}"
        )
    basis = paired["interval_basis"].fillna("").astype(str).str.lower()
    if not (
        basis.str.contains("exact-target importance-corrected", regex=False).all()
        and basis.str.contains("external structural-prior", regex=False).all()
        and basis.str.contains("fixed", regex=False).all()
    ):
        raise ValueError(
            "Figure 2c interval basis does not match the exact state-importance "
            "conditional uncertainty contract"
        )

    rate_col = str(spec["rate_col"])
    reduction_col = str(spec["reduction_col"])
    strategies = tuple(str(strategy) for strategy in spec["strategies"])
    deterministic = burden.loc[
        burden["strategy"].isin(strategies),
        ["country", "strategy", reduction_col, rate_col],
    ].rename(
        columns={
            reduction_col: "deterministic_reduction",
            rate_col: "deterministic_rate",
        }
    )
    audit = paired.loc[paired["strategy"].isin(strategies)].copy()
    audit = audit.merge(deterministic, on=["country", "strategy"], how="left", validate="one_to_one")
    if audit["deterministic_reduction"].isna().any():
        missing_keys = audit.loc[
            audit["deterministic_reduction"].isna(), ["country", "strategy"]
        ].drop_duplicates()
        raise ValueError(f"Figure 2c paired interval rows lack deterministic matches: {missing_keys.to_dict('records')}")

    audit[reduction_col] = pd.to_numeric(audit["deterministic_reduction"], errors="coerce")
    audit["scenario_label"] = audit["strategy"].map(INTERVENTION_INTERVAL_LABELS).fillna(audit["scenario_label"])
    for optional_col in ("reduction_q25", "reduction_q75", "current_rate_median", "intervention_rate_median"):
        if optional_col not in audit.columns:
            audit[optional_col] = np.nan

    return audit[
        [
            "country",
            "scenario_key",
            "scenario_label",
            "outcome",
            reduction_col,
            "reduction_median",
            "reduction_q025",
            "reduction_q25",
            "reduction_q75",
            "reduction_q975",
            "current_rate_median",
            "current_rate_q025",
            "current_rate_q975",
            "intervention_rate_median",
            "intervention_rate_q025",
            "intervention_rate_q975",
            "uncertainty_draws",
            "interval_type",
            "interval_basis",
        ]
    ].sort_values(["country", "scenario_key"])


def intervention_scenario_point_audit() -> None:
    burden = _optimization_burden_frame()
    point_specs = [
        {
            "rate_col": "annualized_infant_cases_per_100k",
            "reduction_col": "relative_reduction_infant_cases",
            "output_path": "outputs/tables/figure2b_intervention_scenario_point_audit.csv",
            "strategies": INTERVENTION_INTERVAL_AUDIT_STRATEGIES,
            "outcome_label": "infant_cases",
        },
        {
            "rate_col": "annualized_child_adolescent_cases_per_100k",
            "reduction_col": "relative_reduction_child_adolescent_cases",
            "output_path": "outputs/tables/figure2c_programme_primary_scenario_point_audit.csv",
            "strategies": (
                "timeliness_only",
                "adolescent_booster",
                "pregnancy_tdap_scaleup",
                "cocooning_adjunct",
                "maternal_immunization",
                "targeted_pep_high_risk",
            ),
            "outcome_label": "child_adolescent_cases",
        },
    ]

    for spec in point_specs:
        rate_col = str(spec["rate_col"])
        reduction_col = str(spec["reduction_col"])
        audit = burden.loc[burden["strategy"].isin(spec["strategies"])].copy()
        audit["scenario_key"] = audit["strategy"]
        audit["scenario_label"] = audit["strategy"].map(INTERVENTION_INTERVAL_LABELS).fillna(audit["strategy_label"])
        audit["outcome"] = spec["outcome_label"]
        audit["estimate_type"] = "deterministic conditional scenario point estimate"
        audit["uncertainty_interval_reported"] = False
        audit["claim_scope"] = (
            "within-profile contrast under the calibrated deterministic reference structure; "
            "not a posterior estimate or validated annual forecast"
        )

        _write(
            audit[
                [
                    "country",
                    "scenario_key",
                    "scenario_label",
                    "outcome",
                    reduction_col,
                    rate_col,
                    "estimate_type",
                    "uncertainty_interval_reported",
                    "claim_scope",
                ]
            ].sort_values(["country", "scenario_key"]),
            str(spec["output_path"]),
        )


def _non_dominated_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (constraint, strategy), group in frontier.groupby(["optimization_constraint", "strategy"], sort=False):
        rows.append(
            {
                "optimization_constraint": constraint,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "decision_domain": STRATEGY_DOMAIN.get(strategy, ""),
                "implementation_intensity": IMPLEMENTATION_INTENSITY.get(strategy, np.nan),
                "countries_non_dominated": int(group["non_dominated_outcome"].sum()),
                "countries_ranked_first_for_infant_cases": int(
                    np.sum(group["infant_case_rank_within_constraint"].to_numpy(dtype=float) == 1)
                ),
                "median_infant_case_reduction": float(group["relative_reduction_infant_cases"].median()),
                "median_resistant_infection_reduction": float(group["relative_reduction_resistant_infections"].median()),
                "median_infant_cases_per_100k": float(group["annualized_infant_cases_per_100k"].median()),
                "median_resistant_infections_per_100k": float(
                    group["annualized_resistant_infections_per_100k"].median()
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "optimization_constraint",
            "countries_ranked_first_for_infant_cases",
            "countries_non_dominated",
            "median_infant_case_reduction",
        ],
        ascending=[True, False, False, False],
    )


def _constrained_preferred_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (constraint, country), group in frontier.groupby(["optimization_constraint", "country"], sort=True):
        infant_best = group.sort_values(["annualized_infant_cases_per_100k", "implementation_intensity"]).iloc[0]
        resistant_best = group.sort_values(["annualized_resistant_infections_per_100k", "implementation_intensity"]).iloc[0]
        nondom = group.loc[group["non_dominated_outcome"], "strategy_label"].tolist()
        rows.append(
            {
                "country": country,
                "optimization_constraint": constraint,
                "preferred_strategy_primary_infant_cases": infant_best["strategy"],
                "preferred_strategy_label": infant_best["strategy_label"],
                "preferred_infant_cases_per_100k": infant_best["annualized_infant_cases_per_100k"],
                "preferred_infant_case_reduction": infant_best["relative_reduction_infant_cases"],
                "preferred_resistant_infection_reduction": infant_best["relative_reduction_resistant_infections"],
                "secondary_resistance_preferred_strategy": resistant_best["strategy"],
                "secondary_resistance_preferred_label": resistant_best["strategy_label"],
                "non_dominated_strategy_labels": "; ".join(nondom),
                "non_dominated_strategy_count": len(nondom),
            }
        )
    return pd.DataFrame(rows).sort_values(["country", "optimization_constraint"])


def _psa_regret_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = _read_csv("outputs/tables/joint_psa_infant_rank_samples.csv")
    n_samples = int(pd.to_numeric(samples["psa_sample_id"], errors="coerce").nunique())
    rows: list[pd.DataFrame] = []
    for constraint, strategies in PSA_CONSTRAINT_STRATEGIES.items():
        frame = samples.loc[samples["strategy"].isin(strategies)].copy()
        frame["optimization_constraint"] = constraint
        best = frame.groupby(["country", "psa_sample_id"], as_index=False)["annualized_infant_cases_per_100k"].min()
        best = best.rename(columns={"annualized_infant_cases_per_100k": "best_infant_cases_per_100k"})
        frame = frame.merge(best, on=["country", "psa_sample_id"], how="left")
        current = frame.loc[
            frame["strategy"].eq("current"),
            ["country", "psa_sample_id", "annualized_infant_cases_per_100k"],
        ].rename(columns={"annualized_infant_cases_per_100k": "current_infant_cases_per_100k"})
        frame = frame.merge(current, on=["country", "psa_sample_id"], how="left")
        frame["absolute_regret_infant_cases_per_100k"] = (
            frame["annualized_infant_cases_per_100k"] - frame["best_infant_cases_per_100k"]
        )
        frame["relative_regret_vs_best"] = frame["absolute_regret_infant_cases_per_100k"] / frame[
            "best_infant_cases_per_100k"
        ].replace(0, np.nan)
        frame["standardized_regret_vs_current"] = frame["absolute_regret_infant_cases_per_100k"] / frame[
            "current_infant_cases_per_100k"
        ].replace(0, np.nan)
        frame["is_best_in_draw"] = np.isclose(
            frame["annualized_infant_cases_per_100k"], frame["best_infant_cases_per_100k"]
        )
        frame["within_10_percent_of_constraint_best"] = frame["annualized_infant_cases_per_100k"] <= (
            1.10 * frame["best_infant_cases_per_100k"]
        )
        rows.append(frame)
    regret = pd.concat(rows, ignore_index=True)

    summary = (
        regret.groupby(["optimization_constraint", "strategy"], as_index=False)
        .agg(
            mean_absolute_regret_infant_cases_per_100k=("absolute_regret_infant_cases_per_100k", "mean"),
            median_absolute_regret_infant_cases_per_100k=("absolute_regret_infant_cases_per_100k", "median"),
            maximum_absolute_regret_infant_cases_per_100k=("absolute_regret_infant_cases_per_100k", "max"),
            mean_relative_regret_vs_best=("relative_regret_vs_best", "mean"),
            mean_standardized_regret_vs_current=("standardized_regret_vs_current", "mean"),
            median_standardized_regret_vs_current=("standardized_regret_vs_current", "median"),
            maximum_standardized_regret_vs_current=("standardized_regret_vs_current", "max"),
            probability_best_in_draw=("is_best_in_draw", "mean"),
            probability_within_10_percent_of_best=("within_10_percent_of_constraint_best", "mean"),
            median_infant_cases_per_100k=("annualized_infant_cases_per_100k", "median"),
            n_decision_cells=("absolute_regret_infant_cases_per_100k", "size"),
        )
        .sort_values(["optimization_constraint", "mean_absolute_regret_infant_cases_per_100k"])
    )
    summary["strategy_label"] = summary["strategy"].map(STRATEGY_LABELS)
    summary["sensitivity_source"] = (
        f"{n_samples}-sample selected-parameter Latin-hypercube rank analysis including program, "
        "resistance-management, and future product-target profiles."
    )

    country = (
        regret.groupby(["optimization_constraint", "country", "strategy"], as_index=False)
        .agg(
            mean_absolute_regret_infant_cases_per_100k=("absolute_regret_infant_cases_per_100k", "mean"),
            maximum_absolute_regret_infant_cases_per_100k=("absolute_regret_infant_cases_per_100k", "max"),
            mean_standardized_regret_vs_current=("standardized_regret_vs_current", "mean"),
            maximum_standardized_regret_vs_current=("standardized_regret_vs_current", "max"),
            probability_best_in_draw=("is_best_in_draw", "mean"),
            probability_within_10_percent_of_best=("within_10_percent_of_constraint_best", "mean"),
            median_infant_cases_per_100k=("annualized_infant_cases_per_100k", "median"),
            n_psa_samples=("psa_sample_id", "nunique"),
        )
        .sort_values(["optimization_constraint", "country", "mean_absolute_regret_infant_cases_per_100k"])
    )
    country["strategy_label"] = country["strategy"].map(STRATEGY_LABELS)
    return summary, country


def _resistance_preference_tradeoff_tables(burden: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = burden.loc[burden["strategy"].isin(PROGRAM_PLUS_RESISTANCE_STRATEGIES)].copy()
    frame["infant_score"] = pd.to_numeric(frame["relative_reduction_infant_cases"], errors="coerce").fillna(-np.inf)
    frame["resistance_score"] = (
        pd.to_numeric(frame["relative_reduction_resistant_infections"], errors="coerce").fillna(0.0)
    )
    frame["implementation_intensity"] = pd.to_numeric(frame["implementation_intensity"], errors="coerce").fillna(np.inf)
    lambda_rows: list[dict[str, object]] = []
    for resistance_weight in np.round(np.linspace(0.0, 1.0, 101), 2):
        weighted = frame.copy()
        weighted["preference_score"] = (
            (1.0 - resistance_weight) * weighted["infant_score"]
            + resistance_weight * weighted["resistance_score"]
        )
        preferred = (
            weighted.sort_values(
                [
                    "country",
                    "preference_score",
                    "annualized_infant_cases_per_100k",
                    "implementation_intensity",
                ],
                ascending=[True, False, True, True],
            )
            .groupby("country", as_index=False)
            .first()
        )
        for strategy, group in preferred.groupby("strategy", sort=False):
            lambda_rows.append(
                {
                    "resistance_weight_lambda": resistance_weight,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                    "countries_preferred": int(group["country"].nunique()),
                    "median_preference_score": float(group["preference_score"].median(skipna=True)),
                    "median_infant_case_reduction": float(group["infant_score"].median(skipna=True)),
                    "median_resistant_infection_reduction": float(group["resistance_score"].median(skipna=True)),
                }
            )
    lambda_summary = pd.DataFrame(lambda_rows).sort_values(["resistance_weight_lambda", "strategy"])

    country_rows: list[dict[str, object]] = []
    for country, group in frame.groupby("country", sort=True):
        current_cases = float(
            group.loc[group["strategy"].eq("current"), "annualized_infant_cases_per_100k"].iloc[0]
        )
        min_lambda = np.nan
        for resistance_weight in np.round(np.linspace(0.0, 1.0, 101), 2):
            weighted = group.copy()
            weighted["preference_score"] = (
                (1.0 - resistance_weight) * weighted["infant_score"]
                + resistance_weight * weighted["resistance_score"]
            )
            winner = weighted.sort_values(
                ["preference_score", "annualized_infant_cases_per_100k", "implementation_intensity"],
                ascending=[False, True, True],
            ).iloc[0]
            if winner["strategy"] == "resistance_guided_treatment":
                min_lambda = float(resistance_weight)
                break
        rg = group.loc[group["strategy"].eq("resistance_guided_treatment")].iloc[0]
        infant_best = group.sort_values(["annualized_infant_cases_per_100k", "implementation_intensity"]).iloc[0]
        country_rows.append(
            {
                "country": country,
                "minimum_lambda_for_resistance_guided_preferred": min_lambda,
                "resistance_guided_infant_case_regret_per_100k": float(
                    rg["annualized_infant_cases_per_100k"] - infant_best["annualized_infant_cases_per_100k"]
                ),
                "resistance_guided_standardized_infant_regret_vs_current": float(
                    (rg["annualized_infant_cases_per_100k"] - infant_best["annualized_infant_cases_per_100k"])
                    / current_cases
                )
                if current_cases > 0
                else np.nan,
                "resistance_guided_resistant_infection_reduction": float(rg["resistance_score"]),
            }
        )
    country_threshold = pd.DataFrame(country_rows)

    epsilon_rows: list[dict[str, object]] = []
    for threshold in (0.0, 0.5, 0.8, 0.9, 0.95):
        for country, group in frame.groupby("country", sort=True):
            current_cases = float(
                group.loc[group["strategy"].eq("current"), "annualized_infant_cases_per_100k"].iloc[0]
            )
            infant_best = group.sort_values(["annualized_infant_cases_per_100k", "implementation_intensity"]).iloc[0]
            feasible = group.loc[group["resistance_score"].ge(threshold)].copy()
            if feasible.empty:
                epsilon_rows.append(
                    {
                        "resistance_reduction_constraint": threshold,
                        "country": country,
                        "preferred_strategy": "",
                        "preferred_strategy_label": "",
                        "feasible_strategy_count": 0,
                        "infant_case_regret_per_100k": np.nan,
                        "standardized_infant_regret_vs_current": np.nan,
                        "resistant_infection_reduction": np.nan,
                    }
                )
                continue
            winner = feasible.sort_values(["annualized_infant_cases_per_100k", "implementation_intensity"]).iloc[0]
            regret = float(winner["annualized_infant_cases_per_100k"] - infant_best["annualized_infant_cases_per_100k"])
            epsilon_rows.append(
                {
                    "resistance_reduction_constraint": threshold,
                    "country": country,
                    "preferred_strategy": winner["strategy"],
                    "preferred_strategy_label": winner["strategy_label"],
                    "feasible_strategy_count": int(len(feasible)),
                    "infant_case_regret_per_100k": regret,
                    "standardized_infant_regret_vs_current": regret / current_cases if current_cases > 0 else np.nan,
                    "resistant_infection_reduction": float(winner["resistance_score"]),
                }
            )
    epsilon_country = pd.DataFrame(epsilon_rows)
    epsilon_summary = (
        epsilon_country.groupby(["resistance_reduction_constraint", "preferred_strategy"], as_index=False)
        .agg(
            preferred_strategy_label=("preferred_strategy_label", "first"),
            countries_preferred=("country", "nunique"),
            median_infant_case_regret_per_100k=("infant_case_regret_per_100k", "median"),
            median_standardized_infant_regret_vs_current=("standardized_infant_regret_vs_current", "median"),
            median_resistant_infection_reduction=("resistant_infection_reduction", "median"),
        )
        .sort_values(["resistance_reduction_constraint", "countries_preferred"], ascending=[True, False])
    )
    return lambda_summary, country_threshold, epsilon_summary


def _burden_category(value: float, low: float, high: float) -> str:
    if value <= low:
        return "Lower modeled infant burden"
    if value >= high:
        return "Higher modeled infant burden"
    return "Intermediate modeled infant burden"


def _country_portfolio_table(burden: pd.DataFrame, preferred: pd.DataFrame, regret_country: pd.DataFrame) -> pd.DataFrame:
    current = burden.loc[burden["strategy"].eq("current")].copy()
    low_q, high_q = current["annualized_infant_cases_per_100k"].quantile([0.33, 0.67])
    age_weights = _read_csv("outputs/tables/age_pattern_country_weights.csv")
    age_map = {
        row["country"]: row
        for _, row in age_weights.iterrows()
    }
    robust = (
        regret_country.loc[regret_country["optimization_constraint"].eq("program_plus_resistance")]
        .sort_values(["country", "mean_absolute_regret_infant_cases_per_100k", "maximum_absolute_regret_infant_cases_per_100k"])
        .groupby("country", as_index=False)
        .first()
    )
    robust_map = {row["country"]: row for _, row in robust.iterrows()}
    pref_wide = preferred.pivot(
        index="country",
        columns="optimization_constraint",
        values="preferred_strategy_label",
    )
    rows: list[dict[str, object]] = []
    for _, row in current.sort_values("country").iterrows():
        country = row["country"]
        age_row = age_map.get(country)
        if age_row is None:
            age_status = "Not externally assessed; no comparable public age-pattern check"
            strength = "Not externally assessed"
        elif bool(age_row.get("all_checks_pass_weight_threshold", False)):
            age_status = f"Acceptable external age-pattern agreement (weight {float(age_row['country_age_pattern_weight']):.2f})"
            strength = "Externally supported"
        else:
            age_status = f"Weak external age-pattern agreement (weight {float(age_row['country_age_pattern_weight']):.2f})"
            strength = "Externally discordant"
        robust_row = robust_map.get(country)
        rows.append(
            {
                "country": country,
                "current_resistance_fraction_start": row["resistant_fraction_start"],
                "baseline_modeled_infant_cases_per_100k": row["annualized_infant_cases_per_100k"],
                "baseline_modeled_infant_burden_category": _burden_category(
                    float(row["annualized_infant_cases_per_100k"]), float(low_q), float(high_q)
                ),
                "best_program_only_strategy": pref_wide.loc[country, "program_only"],
                "best_program_plus_resistance_strategy": pref_wide.loc[country, "program_plus_resistance"],
                "best_future_product_target_strategy": pref_wide.loc[country, "future_product_target"],
                "minimum_regret_program_plus_resistance_strategy": robust_row["strategy_label"] if robust_row is not None else "",
                "age_pattern_agreement": age_status,
                "interpretation_strength": strength,
            }
        )
    return pd.DataFrame(rows)


def _implementation_intensity_score_definitions() -> pd.DataFrame:
    rows = [
        {
            "implementation_intensity": 0,
            "category": "Current practice",
            "definition": "Current routine vaccination and standard treatment/PEP assumptions.",
            "assigned_strategy_profiles": "Current practice",
            "decision_use": "Comparator and lowest-intensity reference.",
        },
        {
            "implementation_intensity": 1,
            "category": "Single routine program modification",
            "definition": "One added or scaled routine vaccination or targeted prophylaxis component without a new diagnostic-management pathway.",
            "assigned_strategy_profiles": "Nominal coverage floor; Routine timeliness; Adolescent booster; Pregnancy Tdap scale-up; Targeted high-risk PEP",
            "decision_use": "Program-only strategy profile.",
        },
        {
            "implementation_intensity": 2,
            "category": "Clinical or contact-management addition",
            "definition": "Added close-contact adult operations or resistance-guided testing/treatment and assumed resistant-strain PEP improvement.",
            "assigned_strategy_profiles": "Close-contact adult adjunct; Resistance-guided management",
            "decision_use": "Program or resistance-management strategy profile.",
        },
        {
            "implementation_intensity": 3,
            "category": "Multi-component exposure-reduction package",
            "definition": "Layered infant-protection package combining pregnancy Tdap scale-up, close-contact adult protection, reproductive-age boosting, and infant-contact reduction.",
            "assigned_strategy_profiles": "Infant-exposure reduction",
            "decision_use": "Composite program profile with greater operational dependence.",
        },
        {
            "implementation_intensity": 4,
            "category": "Future product target",
            "definition": "Hypothetical vaccine product target with stronger transmission-blocking assumptions; not currently implementable as a policy lever.",
            "assigned_strategy_profiles": "Transmission-blocking vaccine target; High-transmission-blocking vaccine target",
            "decision_use": "Future product-target domain only.",
        },
        {
            "implementation_intensity": 5,
            "category": "Future stress-test package",
            "definition": "Mechanistic upper-bound profile combining future vaccine assumptions with multiple program and management components.",
            "assigned_strategy_profiles": "Combined future stress-test profile",
            "decision_use": "Stress test, not a deployable policy recommendation.",
        },
    ]
    return pd.DataFrame(rows)


def resistance_management_policy_decomposition() -> None:
    mechanism = _read_csv("outputs/tables/resistance_mechanism_decomposition.csv")
    implementation = _read_csv("outputs/tables/treatment_implementation_sensitivity.csv")

    mechanism_label = {
        "baseline_full_mechanism": "Baseline resistant-strain mechanism",
        "equal_treatment_effect": "Treatment differential removed",
        "equal_pep_effect": "PEP differential removed",
        "no_treatment_or_pep_differential": "Treatment and PEP differentials removed",
        "fitness_cost": "Fitness-cost stress test",
        "no_resistant_importation": "No ongoing resistant importation",
    }
    treatment_label = {"yes": "Baseline strain-specific differential", "no": "Equalized with sensitive strain"}
    pep_label = {"yes": "Baseline strain-specific differential", "no": "Equalized with sensitive strain"}
    mechanism_rows: list[dict[str, object]] = []
    for _, row in mechanism.iterrows():
        scenario = str(row["scenario"])
        mechanism_rows.append(
            {
                "analysis_layer": "Long-horizon mechanism decomposition",
                "scenario": scenario,
                "policy_read": mechanism_label.get(scenario, scenario),
                "analysis_window": f"{ANALYSIS_START.year}-{ANALYSIS_END.year}",
                "treatment_component": treatment_label.get(str(row["treatment_differential"]), ""),
                "pep_component": pep_label.get(str(row["pep_differential"]), ""),
                "testing_or_uptake_component": "Not varied",
                "median_infant_case_reduction_vs_current": np.nan,
                "iqr_infant_case_reduction_vs_current": "",
                "countries_with_positive_reduction": np.nan,
                "median_infant_cases_per_100k": row["median_infant_cases_per_100k"],
                "resistance_metric": "Median resistant infections per 100k",
                "median_resistance_metric": row["median_resistant_infections_per_100k"],
                "end_resistant_fraction": row["median_end_resistant_fraction"],
                "interpretation": row["interpretation"],
                "source_table": "resistance_mechanism_decomposition.csv",
            }
        )

    implementation_label = {
        "current_near_term": "Current near-term management",
        "guided_uptake_25_pep_restored": "25% testing/treatment uptake plus assumed PEP improvement",
        "guided_uptake_50_pep_restored": "50% testing/treatment uptake plus assumed PEP improvement",
        "guided_uptake_75_pep_restored": "75% testing/treatment uptake plus assumed PEP improvement",
        "guided_uptake_100_pep_restored": "100% testing/treatment uptake plus assumed PEP improvement",
        "guided_uptake_50_no_pep_restoration": "50% testing/treatment uptake only",
        "guided_uptake_100_no_pep_restoration": "100% testing/treatment uptake only",
        "guided_uptake_50_low_pep_reach": "50% uptake plus assumed PEP improvement with lower PEP reach",
    }
    implementation_rows: list[dict[str, object]] = []
    for _, row in implementation.iterrows():
        pep_restored = str(row["pep_restored"])
        implementation_rows.append(
            {
                "analysis_layer": "Near-term implementation sensitivity",
                "scenario": row["scenario"],
                "policy_read": implementation_label.get(str(row["scenario"]), row["scenario"]),
                "analysis_window": "2027-2031",
                "treatment_component": f"Guided-treatment uptake {float(row['implementation_uptake']):.2g}",
                "pep_component": "Assumed improved resistant-strain PEP" if pep_restored == "yes" else "Baseline resistant-strain PEP",
                "testing_or_uptake_component": f"PEP reach multiplier {float(row['pep_coverage_multiplier']):.2g}",
                "median_infant_case_reduction_vs_current": row["median_infant_case_reduction_vs_current_5y"],
                "iqr_infant_case_reduction_vs_current": row["iqr_infant_case_reduction_vs_current_5y"],
                "countries_with_positive_reduction": row["countries_with_positive_reduction"],
                "median_infant_cases_per_100k": row["median_infant_cases_per_100k"],
                "resistance_metric": "",
                "median_resistance_metric": np.nan,
                "end_resistant_fraction": np.nan,
                "interpretation": row["implementation_note"],
                "source_table": "treatment_implementation_sensitivity.csv",
            }
        )

    out = pd.DataFrame([*mechanism_rows, *implementation_rows])
    order = [
        "Baseline resistant-strain mechanism",
        "Treatment differential removed",
        "PEP differential removed",
        "Treatment and PEP differentials removed",
        "Fitness-cost stress test",
        "No ongoing resistant importation",
        "Current near-term management",
        "25% testing/treatment uptake plus assumed PEP improvement",
        "50% testing/treatment uptake plus assumed PEP improvement",
        "75% testing/treatment uptake plus assumed PEP improvement",
        "100% testing/treatment uptake plus assumed PEP improvement",
        "50% testing/treatment uptake only",
        "100% testing/treatment uptake only",
        "50% uptake plus assumed PEP improvement with lower PEP reach",
    ]
    out["display_order"] = out["policy_read"].map({label: idx for idx, label in enumerate(order, start=1)})
    _write(
        out.sort_values(["analysis_layer", "display_order"]).drop(columns="display_order"),
        "outputs/tables/resistance_management_policy_decomposition.csv",
    )


def constrained_optimization_tables() -> None:
    burden = _optimization_burden_frame()
    frontier = _constraint_frontier_points(burden)
    _write(frontier, "outputs/tables/optimization_frontier_points.csv")
    non_dominated = _non_dominated_summary(frontier)
    _write(non_dominated, "outputs/tables/optimization_non_dominated_strategies.csv")
    _write(_implementation_intensity_score_definitions(), "outputs/tables/implementation_intensity_score_definitions.csv")
    preferred = _constrained_preferred_summary(frontier)
    _write(preferred, "outputs/tables/constrained_optimization_summary.csv")
    regret_summary, regret_country = _psa_regret_outputs()
    _write(regret_summary, "outputs/tables/optimization_regret_summary.csv")
    _write(regret_country, "outputs/tables/optimization_regret_country_summary.csv")
    lambda_summary, lambda_country, epsilon_summary = _resistance_preference_tradeoff_tables(burden)
    _write(lambda_summary, "outputs/tables/resistance_preference_weight_summary.csv")
    _write(lambda_country, "outputs/tables/resistance_preference_country_thresholds.csv")
    _write(epsilon_summary, "outputs/tables/resistance_epsilon_constraint_summary.csv")
    country_portfolios = _country_portfolio_table(burden, preferred, regret_country)
    _write(country_portfolios, "outputs/tables/country_profile_preferred_portfolios.csv")


def main() -> None:
    # Fail before unlinking or writing any canonical review/source table.  A
    # late check would leave a partially refreshed publication layer when the
    # predictive gate is closed.
    _require_figure2_release_inputs()
    for suffix in (".csv", ".parquet"):
        for stale_stem in (
            "figure2b_intervention_predictive_interval_audit",
            "figure2c_programme_primary_conditional_uncertainty_audit",
            "figure2c_programme_primary_joint_cri_audit",
        ):
            (ROOT / f"outputs/tables/{stale_stem}{suffix}").unlink(missing_ok=True)
    higher_child_coverage_mechanism()
    intervention_rank_robustness()
    horizon_rank_sensitivity()
    infant_age_split_intervention()
    infant_age_split_horizon_sensitivity()
    intervention_rank_stability_diagnostics()
    deterministic_event_scale_diagnostics()
    resistance_parameter_justification()
    limitation_diagnostic_map()
    sensitivity_correlations()
    veinf_thresholds_against_comparators()
    resistance_management_policy_decomposition()
    intervention_scenario_point_audit()
    constrained_optimization_tables()
    write_run_metadata(
        "high_risk_review_tables",
        current_run_metadata("high_risk_review_tables", row_counts=WRITTEN_ROW_COUNTS),
    )


if __name__ == "__main__":
    main()
