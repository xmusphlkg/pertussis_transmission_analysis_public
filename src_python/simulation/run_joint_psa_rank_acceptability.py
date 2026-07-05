from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src_python.simulation.common import (
    current_run_metadata,
    execute_scenario_list,
    load_configs,
    make_config,
    make_intervention_config,
    write_run_metadata,
)
from src_python.simulation.run_routine_timeliness_sensitivity import _apply_timeliness
from src_python.utils.io import project_path, write_dataframe


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
INFANT_TARGETS = ("infant_0_2m", "infant_3_11m")
HOUSEHOLD_LIKE_SOURCES = (
    "child_1_4y",
    "child_5_9y",
    "adolescent_10_17y",
    "young_adult_18_39y",
    "middle_adult_40_64y",
)
GUIDED_MANAGEMENT_STRATEGIES = {"resistance_guided_treatment", "combined_strategy"}

STEM = "joint_psa_rank_acceptability"
SAMPLE_PATH = project_path("outputs", "tables", "joint_psa_parameter_samples.csv")
RANK_SAMPLE_PATH = project_path("outputs", "tables", "joint_psa_infant_rank_samples.csv")
ACCEPTABILITY_PATH = project_path("outputs", "tables", "joint_psa_rank_acceptability.csv")
RUN_SUMMARY_PATH = project_path("outputs", "summaries", "joint_psa_rank_acceptability_summary.csv")
SIMULATION_SUMMARY_PATH = project_path("outputs", "summaries", "joint_psa_scenario_summary.csv")
SIMULATION_TS_PATH = project_path("outputs", "simulations", "joint_psa_rank_acceptability.parquet")


def _write_incremental(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_index(axis=1)
    df.to_csv(path, index=False)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def _read_existing(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _clip_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _scale_ve_inf(config: dict[str, Any], sample: dict[str, float]) -> None:
    baseline_reference = 0.25
    multiplier = float(sample["VE_inf_baseline"]) / baseline_reference
    vaccine = config.setdefault("vaccine", {})
    vaccine["VE_inf"] = _clip_probability(float(vaccine.get("VE_inf", baseline_reference)) * multiplier)


def _apply_infant_contact_multiplier(config: dict[str, Any], multiplier: float) -> None:
    labels = [record["label"] for record in config["age_groups"]]
    rows = config["contact_matrix"]["rows"]
    for target in INFANT_TARGETS:
        if target not in labels:
            continue
        target_idx = labels.index(target)
        for source in HOUSEHOLD_LIKE_SOURCES:
            if source not in labels:
                continue
            source_idx = labels.index(source)
            rows[target_idx][source_idx] = float(rows[target_idx][source_idx]) * float(multiplier)


def _interpolate(base: float, target: float, fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(base + fraction * (target - base))


def _apply_guided_management_uptake(
    config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    strategy: str,
    uptake: float,
) -> None:
    if strategy not in GUIDED_MANAGEMENT_STRATEGIES:
        return
    for key in ("infectious_duration_reduction", "infectiousness_reduction"):
        config["treatment"]["resistant"][key] = _clip_probability(
            _interpolate(
                float(current_config["treatment"]["resistant"][key]),
                float(config["treatment"]["resistant"][key]),
                uptake,
            )
        )
    config["treatment"]["treatment_rate_symptomatic"] = max(
        0.0,
        _interpolate(
            float(current_config["treatment"]["treatment_rate_symptomatic"]),
            float(config["treatment"]["treatment_rate_symptomatic"]),
            uptake,
        ),
    )
    config["PEP"]["effectiveness_resistant"] = _clip_probability(
        _interpolate(
            float(current_config["PEP"]["effectiveness_resistant"]),
            float(config["PEP"]["effectiveness_resistant"]),
            uptake,
        )
    )


def _make_strategy_config(
    strategy: str,
    *,
    country: str,
    resistance_name: str,
) -> tuple[dict[str, Any], str]:
    if strategy == "timeliness_only":
        config, vaccine_name = make_intervention_config("current", country_profile=country)
        return _apply_timeliness(config), vaccine_name
    if strategy == "transmission_blocking_vaccine":
        vaccine_name = "transmission_blocking"
        config = make_config(
            vaccine_scenario=vaccine_name,
            resistance_scenario=resistance_name,
            country_profile=country,
        )
        return config, vaccine_name
    return make_intervention_config(strategy, country_profile=country)


def _apply_psa_sample(
    config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    strategy: str,
    sample: dict[str, float],
) -> dict[str, Any]:
    out = deepcopy(config)
    out["reporting_multiplier"] = float(sample["reporting_multiplier"])
    _scale_ve_inf(out, sample)
    _apply_infant_contact_multiplier(out, float(sample["infant_contact_multiplier"]))
    out["transmission"]["relative_infectiousness_asymptomatic"] = float(
        sample["relative_infectiousness_asymptomatic"]
    )
    out["natural_history"]["infectious_duration_asymptomatic"] = float(
        sample["infectious_duration_asymptomatic"]
    )
    out["transmission"]["fitness_R"] = float(sample["fitness_R"])
    _apply_guided_management_uptake(
        out,
        current_config,
        strategy=strategy,
        uptake=float(sample["resistance_management_uptake"]),
    )
    out["PEP"]["coverage_household_contacts"] = _clip_probability(
        float(out["PEP"].get("coverage_household_contacts", 0.0)) * float(sample["PEP_coverage_multiplier"])
    )
    out.setdefault("metadata", {})["joint_psa_sample_id"] = int(sample["psa_sample_id"])
    return out


def _sample_table(sample_size: int, seed: int) -> pd.DataFrame:
    names = [
        "reporting_multiplier_unit",
        "infant_contact_multiplier",
        "VE_inf_baseline",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_asymptomatic",
        "fitness_R",
        "resistance_management_uptake",
        "PEP_coverage_multiplier",
    ]
    bounds = np.array(
        [
            [0.0, 1.0],
            [0.75, 1.50],
            [0.05, 0.60],
            [0.25, 0.85],
            [7.0, 21.0],
            [0.70, 1.25],
            [0.40, 1.00],
            [0.50, 1.50],
        ],
        dtype=float,
    )
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    matrix = qmc.scale(sampler.random(sample_size), bounds[:, 0], bounds[:, 1])
    df = pd.DataFrame(matrix, columns=names)
    log_low, log_high = np.log(0.50), np.log(1.50)
    df["reporting_multiplier"] = np.exp(log_low + df.pop("reporting_multiplier_unit") * (log_high - log_low))
    df["psa_sample_id"] = np.arange(1, sample_size + 1, dtype=int)
    df["sample_design"] = "latin_hypercube_joint"
    return df[
        [
            "psa_sample_id",
            "sample_design",
            "reporting_multiplier",
            "infant_contact_multiplier",
            "VE_inf_baseline",
            "relative_infectiousness_asymptomatic",
            "infectious_duration_asymptomatic",
            "fitness_R",
            "resistance_management_uptake",
            "PEP_coverage_multiplier",
        ]
    ]


def _set_smoke_runtime(config: dict[str, Any], *, output_time_step: float = 90.0) -> None:
    config.setdefault("calendar", {})["analysis_end_date"] = "2025-12-31"
    config.setdefault("simulation", {})["end_time"] = 365.0
    config["simulation"]["output_time_step"] = float(output_time_step)
    config["simulation"]["rtol"] = max(float(config["simulation"].get("rtol", 1e-5)), 1e-4)
    config["simulation"]["atol"] = max(float(config["simulation"].get("atol", 1e-7)), 1e-6)


def _build_scenarios_for_sample(
    configs: dict[str, Any],
    sample: dict[str, float],
    *,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
    smoke_runtime: bool,
    existing_cells: set[tuple[int, str, str]] | None = None,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    sample_id = int(sample["psa_sample_id"])
    existing_cells = existing_cells or set()
    for country in countries:
        current_config, _ = make_intervention_config("current", country_profile=country)
        for strategy in strategies:
            if (sample_id, country, strategy) in existing_cells:
                continue
            config, vaccine_name = _make_strategy_config(
                strategy,
                country=country,
                resistance_name=resistance_name,
            )
            config = _apply_psa_sample(config, current_config, strategy=strategy, sample=sample)
            if smoke_runtime:
                _set_smoke_runtime(config)
            scenarios.append(
                {
                    "config": config,
                    "analysis": "joint_psa_rank_acceptability",
                    "scenario": strategy,
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": strategy,
                    "metadata": {
                        "country": country,
                        "strategy": strategy,
                        **{key: value for key, value in sample.items() if key != "sample_design"},
                    },
                }
            )
    return scenarios


def _rank_sample_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if "strategy" not in out.columns:
        if "scenario" not in out.columns:
            raise KeyError("Expected either 'strategy' or 'scenario' in PSA summary output")
        out["strategy"] = out["scenario"]
    out["psa_sample_id"] = pd.to_numeric(out["psa_sample_id"], errors="raise").astype(int)
    out["rank"] = out.groupby(["country", "psa_sample_id"])["total_infant_cases"].rank(
        method="min",
        ascending=True,
    )
    best = out.groupby(["country", "psa_sample_id"], as_index=False)["total_infant_cases"].min().rename(
        columns={"total_infant_cases": "best_total_infant_cases"}
    )
    current = out.loc[
        out["strategy"].eq("current"),
        ["country", "psa_sample_id", "total_infant_cases", "annualized_infant_cases_per_100k"],
    ].rename(
        columns={
            "total_infant_cases": "current_total_infant_cases",
            "annualized_infant_cases_per_100k": "current_annualized_infant_cases_per_100k",
        }
    )
    out = out.merge(best, on=["country", "psa_sample_id"], how="left")
    out = out.merge(current, on=["country", "psa_sample_id"], how="left")
    out["relative_reduction_infant_cases_vs_current"] = 1.0 - out["total_infant_cases"] / out[
        "current_total_infant_cases"
    ].replace(0, np.nan)
    out["within_10_percent_of_best"] = out["total_infant_cases"] <= 1.10 * out["best_total_infant_cases"]
    keep = [
        "psa_sample_id",
        "country",
        "strategy",
        "rank",
        "total_infant_cases",
        "annualized_infant_cases_per_100k",
        "relative_reduction_infant_cases_vs_current",
        "within_10_percent_of_best",
        "reporting_multiplier",
        "infant_contact_multiplier",
        "VE_inf_baseline",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_asymptomatic",
        "fitness_R",
        "resistance_management_uptake",
        "PEP_coverage_multiplier",
    ]
    return out.loc[:, keep].sort_values(["psa_sample_id", "country", "rank", "strategy"]).reset_index(drop=True)


def _acceptability_from_rank_samples(rank_samples: pd.DataFrame, strategies: tuple[str, ...]) -> pd.DataFrame:
    rank_samples = rank_samples.copy()
    rank_samples["rank"] = pd.to_numeric(rank_samples["rank"], errors="coerce")
    ranks = list(range(1, len(strategies) + 1))
    rows: list[dict[str, Any]] = []

    def append_rows(country_label: str, group: pd.DataFrame) -> None:
        grouped = group.groupby("strategy", dropna=False)
        for strategy in strategies:
            strategy_group = grouped.get_group(strategy) if strategy in grouped.groups else pd.DataFrame()
            n = int(strategy_group["psa_sample_id"].nunique()) if not strategy_group.empty else 0
            rank_values = strategy_group["rank"].to_numpy(dtype=float) if n else np.array([], dtype=float)
            cases = (
                pd.to_numeric(strategy_group["annualized_infant_cases_per_100k"], errors="coerce")
                if n
                else pd.Series(dtype=float)
            )
            reductions = (
                pd.to_numeric(strategy_group["relative_reduction_infant_cases_vs_current"], errors="coerce")
                if n
                else pd.Series(dtype=float)
            )
            for rank in ranks:
                rows.append(
                    {
                        "country": country_label,
                        "strategy": strategy,
                        "rank": rank,
                        "rank_acceptability_probability": float(np.mean(rank_values == rank)) if n else np.nan,
                        "probability_rank_1": float(np.mean(rank_values == 1)) if n else np.nan,
                        "probability_top_2": float(np.mean(rank_values <= 2)) if n else np.nan,
                        "probability_top_3": float(np.mean(rank_values <= 3)) if n else np.nan,
                        "probability_within_10_percent_of_best": float(
                            strategy_group["within_10_percent_of_best"].mean()
                        )
                        if n
                        else np.nan,
                        "mean_rank": float(np.nanmean(rank_values)) if n else np.nan,
                        "median_rank": float(np.nanmedian(rank_values)) if n else np.nan,
                        "median_infant_cases_per_100k": float(cases.median(skipna=True)) if n else np.nan,
                        "q025_infant_cases_per_100k": float(cases.quantile(0.025)) if n else np.nan,
                        "q975_infant_cases_per_100k": float(cases.quantile(0.975)) if n else np.nan,
                        "median_relative_reduction_vs_current": float(reductions.median(skipna=True)) if n else np.nan,
                        "n_psa_samples": n,
                        "n_rank_observations": int(len(strategy_group)) if n else 0,
                    }
                )

    for country, group in rank_samples.groupby("country", sort=True):
        append_rows(str(country), group)
    append_rows("All_countries_pooled", rank_samples)
    acceptability = pd.DataFrame(rows)
    return acceptability.sort_values(["country", "rank", "strategy"]).reset_index(drop=True)


def _run_summary(acceptability: pd.DataFrame) -> pd.DataFrame:
    rank1 = acceptability.loc[acceptability["rank"].eq(1)].copy()
    return rank1.sort_values(["country", "probability_rank_1", "probability_top_2"], ascending=[True, False, False])


def _completed_rank_samples(
    path: Path,
    *,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
) -> tuple[set[int], pd.DataFrame]:
    existing = _read_existing(path)
    if existing.empty or "psa_sample_id" not in existing:
        return set(), pd.DataFrame()
    existing = existing.loc[
        existing["country"].astype(str).isin(set(countries))
        & existing["strategy"].astype(str).isin(set(strategies))
    ].copy()
    if existing.empty:
        return set(), pd.DataFrame()
    existing["psa_sample_id"] = pd.to_numeric(existing["psa_sample_id"], errors="raise").astype(int)
    existing = existing.drop_duplicates(["psa_sample_id", "country", "strategy"], keep="last")
    expected_per_sample = max(1, len(countries) * len(strategies))
    requested_countries = set(countries)
    requested_strategies = set(strategies)
    completed: set[int] = set()
    for sample_id, group in existing.groupby("psa_sample_id"):
        if int(len(group)) < expected_per_sample:
            continue
        if set(group["country"].astype(str)) != requested_countries:
            continue
        if set(group["strategy"].astype(str)) != requested_strategies:
            continue
        completed.add(int(sample_id))
    return completed, existing


def _complete_outcome_rows(
    outcomes: pd.DataFrame,
    *,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    expected_per_sample = max(1, len(countries) * len(strategies))
    requested_countries = set(countries)
    requested_strategies = set(strategies)
    data = outcomes.loc[
        outcomes["country"].astype(str).isin(requested_countries)
        & outcomes["strategy"].astype(str).isin(requested_strategies)
    ].copy()
    if data.empty:
        return pd.DataFrame()
    data["psa_sample_id"] = pd.to_numeric(data["psa_sample_id"], errors="raise").astype(int)
    data = data.drop_duplicates(["psa_sample_id", "country", "strategy"], keep="last")
    complete_ids: list[int] = []
    for sample_id, group in data.groupby("psa_sample_id"):
        if int(len(group)) != expected_per_sample:
            continue
        if set(group["country"].astype(str)) != requested_countries:
            continue
        if set(group["strategy"].astype(str)) != requested_strategies:
            continue
        complete_ids.append(int(sample_id))
    if not complete_ids:
        return pd.DataFrame()
    return data.loc[data["psa_sample_id"].isin(complete_ids)].copy()


def run_joint_psa(
    *,
    sample_size: int,
    seed: int,
    countries: tuple[str, ...],
    strategies: tuple[str, ...],
    n_jobs: int | None,
    sample_batch_size: int,
    resume: bool,
    smoke_runtime: bool,
    keep_timeseries: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    samples = _sample_table(sample_size, seed)
    write_dataframe(samples, SAMPLE_PATH)

    if resume:
        completed, completed_rank = _completed_rank_samples(RANK_SAMPLE_PATH, countries=countries, strategies=strategies)
    else:
        completed, completed_rank = set(), pd.DataFrame()
    outcome_frames = [completed_rank] if not completed_rank.empty else []
    summary_frames = []
    timeseries_frames = []
    existing_cells = (
        {
            (int(row.psa_sample_id), str(row.country), str(row.strategy))
            for row in completed_rank[["psa_sample_id", "country", "strategy"]].itertuples(index=False)
        }
        if not completed_rank.empty
        else set()
    )

    pending_samples = [
        sample for sample in samples.to_dict(orient="records") if int(sample["psa_sample_id"]) not in completed
    ]
    batch_size = max(1, int(sample_batch_size))

    for batch_start in range(0, len(pending_samples), batch_size):
        batch = pending_samples[batch_start : batch_start + batch_size]
        scenarios: list[dict[str, Any]] = []
        for sample in batch:
            scenarios.extend(
                _build_scenarios_for_sample(
                    configs,
                    sample,
                    countries=countries,
                    strategies=strategies,
                    smoke_runtime=smoke_runtime,
                    existing_cells=existing_cells,
                )
            )
        if not scenarios:
            continue
        first_sample = int(batch[0]["psa_sample_id"])
        last_sample = int(batch[-1]["psa_sample_id"])
        batch_stem = (
            f"{STEM}_sample_{first_sample:04d}"
            if first_sample == last_sample
            else f"{STEM}_samples_{first_sample:04d}_{last_sample:04d}"
        )
        timeseries, summary = execute_scenario_list(
            scenarios,
            stem=batch_stem,
            n_jobs=n_jobs,
        )
        outcome_frames.append(summary)
        summary_frames.append(summary)
        if keep_timeseries:
            timeseries_frames.append(timeseries)
        combined_outcomes = pd.concat(outcome_frames, ignore_index=True)
        complete_outcomes = _complete_outcome_rows(combined_outcomes, countries=countries, strategies=strategies)
        combined_rank = _rank_sample_summary(complete_outcomes) if not complete_outcomes.empty else pd.DataFrame()
        _write_incremental(combined_rank, RANK_SAMPLE_PATH)
        partial_acceptability = _acceptability_from_rank_samples(combined_rank, strategies)
        write_dataframe(partial_acceptability, ACCEPTABILITY_PATH)
        write_dataframe(_run_summary(partial_acceptability), RUN_SUMMARY_PATH)

    combined_outcomes = pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame()
    complete_outcomes = _complete_outcome_rows(combined_outcomes, countries=countries, strategies=strategies)
    rank_samples = _rank_sample_summary(complete_outcomes) if not complete_outcomes.empty else _read_existing(RANK_SAMPLE_PATH)
    acceptability = _acceptability_from_rank_samples(rank_samples, strategies)
    run_summary = _run_summary(acceptability)
    write_dataframe(acceptability, ACCEPTABILITY_PATH)
    write_dataframe(run_summary, RUN_SUMMARY_PATH)
    if not complete_outcomes.empty:
        write_dataframe(complete_outcomes, SIMULATION_SUMMARY_PATH)
    if keep_timeseries and timeseries_frames:
        write_dataframe(pd.concat(timeseries_frames, ignore_index=True), SIMULATION_TS_PATH)

    metadata = current_run_metadata(
        STEM,
        row_counts={
            "parameter_samples": int(len(samples)),
            "rank_samples": int(len(rank_samples)),
            "rank_acceptability": int(len(acceptability)),
            "run_summary": int(len(run_summary)),
        },
    )
    metadata.update(
        {
            "sample_size_requested": int(sample_size),
            "sample_seed": int(seed),
            "countries": list(countries),
            "strategies": list(strategies),
            "resume": bool(resume),
            "sample_batch_size": int(batch_size),
            "smoke_runtime": bool(smoke_runtime),
            "keep_timeseries": bool(keep_timeseries),
            "parameter_ranges": {
                "reporting_multiplier": [0.50, 1.50],
                "infant_contact_multiplier": [0.75, 1.50],
                "VE_inf_baseline": [0.05, 0.60],
                "relative_infectiousness_asymptomatic": [0.25, 0.85],
                "infectious_duration_asymptomatic": [7.0, 21.0],
                "fitness_R": [0.70, 1.25],
                "resistance_management_uptake": [0.40, 1.00],
                "PEP_coverage_multiplier": [0.50, 1.50],
            },
        }
    )
    write_run_metadata(STEM, metadata)
    return rank_samples, acceptability


def _parse_csv_tuple(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    parser = argparse.ArgumentParser(
        description="Run selected-parameter joint PSA rank-stability diagnostics for infant-case interventions."
    )
    parser.add_argument("--samples", type=int, default=int(configs["sensitivity"].get("sample_size", 48)))
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--countries", type=str, default="")
    parser.add_argument("--strategies", type=str, default=",".join(SELECTED_STRATEGIES))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="Number of PSA parameter draws to execute in one parallel scenario batch.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing rank-sample output and rerun samples.")
    parser.add_argument("--smoke-runtime", action="store_true", help="Use a one-year coarse-output runtime for smoke tests.")
    parser.add_argument("--keep-timeseries", action="store_true", help="Persist PSA time series; off by default to avoid huge files.")
    args = parser.parse_args()

    countries = _parse_csv_tuple(args.countries, tuple(configs["countries"].keys()))
    strategies = _parse_csv_tuple(args.strategies, SELECTED_STRATEGIES)
    return run_joint_psa(
        sample_size=int(args.samples),
        seed=int(args.seed),
        countries=countries,
        strategies=strategies,
        n_jobs=args.n_jobs,
        sample_batch_size=int(args.sample_batch_size),
        resume=not args.no_resume,
        smoke_runtime=bool(args.smoke_runtime),
        keep_timeseries=bool(args.keep_timeseries),
    )


if __name__ == "__main__":
    main()
