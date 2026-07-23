from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    PROSPECTIVE_POLICY_KEY,
    current_run_metadata,
    enforce_calibration_status,
    execute_scenario_list,
    load_configs,
    make_intervention_config,
    publication_country_names,
    read_run_metadata,
    set_analysis_horizon_years,
    validate_calibration_artifacts,
    write_run_metadata,
)
from src_python.simulation.parameter_distributions import (
    UNCERTAINTY_REGISTRY_SCHEMA_VERSION,
    latin_hypercube_draw_table,
    validate_distribution_spec,
    validate_uncertainty_registry_schema,
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
PROGRAMME_ONLY_STRATEGIES = (
    "timeliness_only",
    "adolescent_booster",
    "pregnancy_tdap_scaleup",
    "cocooning_adjunct",
    "maternal_immunization",
    "targeted_pep_high_risk",
)
INFANT_TARGETS = ("infant_0_2m", "infant_3_11m")
HOUSEHOLD_LIKE_SOURCES = (
    "child_1_4y",
    "child_5_9y",
    "adolescent_10_17y",
    "young_adult_18_39y",
    "middle_adult_40_64y",
)
STEM = "joint_psa_rank_acceptability"
UNCERTAINTY_SCHEMA_VERSION = UNCERTAINTY_REGISTRY_SCHEMA_VERSION
SAMPLE_DESIGN = "latin_hypercube_inverse_cdf"
RUNNER_CONSUMER = "src_python.simulation.run_joint_psa_rank_acceptability"
FITNESS_GRID_CONSUMER = "src_python.simulation.run_fitness_grid"
STRUCTURAL_ALL_TIME = "structural_all_time"
PROSPECTIVE_IMPLEMENTATION = "prospective_implementation"
OBSERVATION_ONLY = "observation_only"
VALID_PARAMETER_TIME_SCOPES = frozenset(
    {STRUCTURAL_ALL_TIME, PROSPECTIVE_IMPLEMENTATION, OBSERVATION_ONLY}
)
FIGURE2B_PARAMETER_NAMES = (
    "infant_contact_multiplier",
    "VE_inf_baseline",
    "relative_infectiousness_asymptomatic",
    "infectious_duration_asymptomatic",
    "fitness_R",
    "PEP_coverage_multiplier",
)
# Backward-compatible name used by the rank-table and resumability code.  Its
# contract is deliberately Figure-2b-specific: a parameter that is inactive for
# all six programme strategies must not be admitted to this tuple.
EXPECTED_PARAMETER_NAMES = FIGURE2B_PARAMETER_NAMES
PARAMETER_TIME_SCOPES = {
    "infant_contact_multiplier": STRUCTURAL_ALL_TIME,
    "VE_inf_baseline": STRUCTURAL_ALL_TIME,
    "relative_infectiousness_asymptomatic": STRUCTURAL_ALL_TIME,
    "infectious_duration_asymptomatic": STRUCTURAL_ALL_TIME,
    "fitness_R": STRUCTURAL_ALL_TIME,
    "PEP_coverage_multiplier": PROSPECTIVE_IMPLEMENTATION,
}
PARAMETER_CONSUMERS = {
    name: (
        (RUNNER_CONSUMER, FITNESS_GRID_CONSUMER)
        if name
        in {
            "infant_contact_multiplier",
            "relative_infectiousness_asymptomatic",
            "infectious_duration_asymptomatic",
            "PEP_coverage_multiplier",
        }
        else (RUNNER_CONSUMER,)
    )
    for name in FIGURE2B_PARAMETER_NAMES
}
SAMPLE_PATH = project_path("outputs", "tables", "joint_psa_parameter_samples.csv")
RANK_SAMPLE_PATH = project_path("outputs", "tables", "joint_psa_infant_rank_samples.csv")
ACCEPTABILITY_PATH = project_path("outputs", "tables", "joint_psa_rank_acceptability.csv")
RUN_SUMMARY_PATH = project_path("outputs", "summaries", "joint_psa_rank_acceptability_summary.csv")
UNDER18_PROGRAMME_RANK_SAMPLE_PATH = project_path(
    "outputs", "tables", "joint_psa_under18_programme_rank_samples.csv"
)
UNDER18_PROGRAMME_ACCEPTABILITY_PATH = project_path(
    "outputs", "tables", "joint_psa_under18_programme_rank_acceptability.csv"
)
UNDER18_PROGRAMME_RUN_SUMMARY_PATH = project_path(
    "outputs", "summaries", "joint_psa_under18_programme_rank_acceptability_summary.csv"
)
SIMULATION_SUMMARY_PATH = project_path("outputs", "summaries", "joint_psa_scenario_summary.csv")
SIMULATION_TS_PATH = project_path("outputs", "simulations", "joint_psa_rank_acceptability.parquet")


def _validate_calibration_inputs(
    countries: tuple[str, ...],
) -> None:
    """Require accepted country calibrations before running the PSA.

    The custom batched runner calls ``execute_scenario_list`` directly, so it
    cannot rely only on the generic ``run_scenario_list`` publication guard.
    Resolve every calibration before any PSA output is written.
    """

    validate_calibration_artifacts(
        countries,
        context="Joint PSA",
    )


def _resume_metadata_available() -> tuple[bool, str]:
    try:
        read_run_metadata(STEM)
    except (FileNotFoundError, ValueError) as exc:
        return False, f"run metadata unavailable: {exc}"
    return True, "run metadata available"


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


def _validated_parameter_time_scopes(
    parameter_specs: dict[str, dict[str, Any]],
    *,
    expected_names: tuple[str, ...],
) -> dict[str, str]:
    """Validate registry scopes against the implemented temporal semantics.

    Scope metadata is not descriptive decoration: it controls whether a draw
    changes the shared pre-2027 history, the prospective policy only, or no
    transmission-model input at all.  The code-side mapping is a fail-closed
    whitelist so a registry typo cannot silently change a parameter's meaning.
    """

    if set(parameter_specs) != set(expected_names):
        raise ValueError(
            "Parameter registry must match the implemented semantic contract; "
            f"expected={list(expected_names)}, actual={list(parameter_specs)}"
        )
    scopes: dict[str, str] = {}
    for name in expected_names:
        spec = parameter_specs[name]
        declared = spec.get("time_scope")
        if declared is None:
            raise ValueError(f"Parameter {name!r} is missing required time_scope metadata")
        scope = str(declared).strip().lower()
        if scope not in VALID_PARAMETER_TIME_SCOPES:
            raise ValueError(
                f"Parameter {name!r} has unsupported time_scope={declared!r}; "
                f"expected one of {sorted(VALID_PARAMETER_TIME_SCOPES)}"
            )
        implemented = PARAMETER_TIME_SCOPES.get(name)
        if implemented is None:
            raise ValueError(f"Parameter {name!r} has no implemented time-scope contract")
        if scope != implemented:
            raise ValueError(
                f"Parameter {name!r} declares time_scope={scope!r}, but its "
                f"implemented scope is {implemented!r}"
            )
        declared_consumers = spec.get("consumers")
        expected_consumers = PARAMETER_CONSUMERS[name]
        if not isinstance(declared_consumers, list) or tuple(declared_consumers) != (
            expected_consumers
        ):
            raise ValueError(
                f"Parameter {name!r} declares consumers={declared_consumers!r}, but "
                f"its implemented consumers are {list(expected_consumers)!r}"
            )
        scopes[name] = scope
    return scopes


def _attached_history_config(config: dict[str, Any]) -> dict[str, Any] | None:
    spec = config.get(PROSPECTIVE_POLICY_KEY)
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("Prospective-policy metadata must be a mapping")
    history = spec.get("history_config")
    if not isinstance(history, dict):
        raise ValueError("Prospective-policy metadata is missing a historical config")
    return history


def _model_config_targets_for_scope(
    config: dict[str, Any],
    scope: str,
) -> tuple[dict[str, Any], ...]:
    """Resolve which transmission-model phases a scoped draw may mutate."""

    if scope == OBSERVATION_ONLY:
        return ()
    if scope == PROSPECTIVE_IMPLEMENTATION:
        return (config,)
    if scope == STRUCTURAL_ALL_TIME:
        history = _attached_history_config(config)
        return (config,) if history is None else (config, history)
    raise ValueError(f"Unsupported parameter time scope {scope!r}")


def _scale_ve_inf(config: dict[str, Any], value: float) -> None:
    baseline_reference = 0.25
    multiplier = float(value) / baseline_reference
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


def _apply_parameter_value(
    config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    strategy: str,
    name: str,
    value: float,
) -> None:
    """Apply one implemented draw to one model-config phase."""

    if name == "infant_contact_multiplier":
        _apply_infant_contact_multiplier(config, value)
    elif name == "VE_inf_baseline":
        _scale_ve_inf(config, value)
    elif name == "relative_infectiousness_asymptomatic":
        config["transmission"]["relative_infectiousness_asymptomatic"] = value
    elif name == "infectious_duration_asymptomatic":
        config["natural_history"]["infectious_duration_asymptomatic"] = value
    elif name == "fitness_R":
        config["transmission"]["fitness_R"] = value
    elif name == "PEP_coverage_multiplier":
        config["PEP"]["coverage_household_contacts"] = _clip_probability(
            float(config["PEP"].get("coverage_household_contacts", 0.0)) * value
        )
    else:
        raise ValueError(f"Parameter {name!r} has no implemented config consumer")


def _make_strategy_config(
    strategy: str,
    *,
    country: str,
) -> tuple[dict[str, Any], str]:
    if strategy == "timeliness_only":
        config, vaccine_name = make_intervention_config("current", country_profile=country)
        return _apply_timeliness(config), vaccine_name
    # This scenario used to bypass the intervention constructor and therefore
    # lacked the baseline history shared by every other 2027 policy.  It is an
    # ordinary registered prospective intervention now, so use the same path.
    return make_intervention_config(strategy, country_profile=country)


def _apply_psa_sample(
    config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    strategy: str,
    sample: dict[str, float],
    parameter_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the six Figure-2b draws with explicit temporal scope."""

    missing = [name for name in FIGURE2B_PARAMETER_NAMES if name not in sample]
    if missing:
        raise ValueError(f"Figure 2b PSA sample is missing parameters: {missing}")
    if "resistance_management_uptake" in sample:
        raise ValueError(
            "resistance_management_uptake is inactive for all Figure 2b programme-only "
            "strategies; consume it through the independent resistance-management analysis"
        )
    if parameter_specs is None:
        scopes = {name: PARAMETER_TIME_SCOPES[name] for name in FIGURE2B_PARAMETER_NAMES}
    else:
        scopes = _validated_parameter_time_scopes(
            parameter_specs,
            expected_names=FIGURE2B_PARAMETER_NAMES,
        )

    out = deepcopy(config)
    for name in FIGURE2B_PARAMETER_NAMES:
        scope = scopes[name]
        for target in _model_config_targets_for_scope(out, scope):
            _apply_parameter_value(
                target,
                current_config,
                strategy=strategy,
                name=name,
                value=float(sample[name]),
            )
    out.setdefault("metadata", {})["joint_psa_sample_id"] = int(sample["psa_sample_id"])
    out["metadata"]["joint_psa_parameter_time_scopes"] = dict(scopes)
    return out


def _default_parameter_specs() -> dict[str, dict[str, Any]]:
    """Load the canonical Figure 2b registry or fail closed.

    The historical uniform-range fallback was unsafe because a missing YAML
    block silently changed the statistical design while retaining the same
    output names.  Keep this helper name for existing callers, but require the
    registered inverse-CDF specifications.
    """

    registry = load_configs().get("parameter_distributions", {})
    validate_uncertainty_registry_schema(registry)
    settings = registry.get("joint_rank_psa", {}) if isinstance(registry, dict) else {}
    specs = settings.get("parameters", {}) if isinstance(settings, dict) else {}
    if not isinstance(specs, dict) or not specs:
        raise ValueError(
            "Current Figure 2b PSA requires parameter_distributions."
            "joint_rank_psa.parameters; legacy uniform-range fallback is disabled"
        )
    _validated_parameter_time_scopes(
        specs,
        expected_names=FIGURE2B_PARAMETER_NAMES,
    )
    return specs


def _sample_table(
    sample_size: int,
    seed: int,
    parameter_specs: dict[str, dict[str, Any]] | None = None,
    *,
    schema_version: int = UNCERTAINTY_SCHEMA_VERSION,
) -> pd.DataFrame:
    schema_version = validate_uncertainty_registry_schema(
        {"schema_version": schema_version},
        context="Figure 2b PSA sample contract",
    )
    specs = parameter_specs or _default_parameter_specs()
    _validated_parameter_time_scopes(specs, expected_names=FIGURE2B_PARAMETER_NAMES)
    df = latin_hypercube_draw_table(specs, sample_size, seed=seed)
    df.insert(0, "uncertainty_schema_version", schema_version)
    df["psa_sample_id"] = np.arange(1, sample_size + 1, dtype=int)
    df["sample_design"] = SAMPLE_DESIGN
    return df[
        [
            "psa_sample_id",
            "sample_design",
            "uncertainty_schema_version",
            *EXPECTED_PARAMETER_NAMES,
        ]
    ]


def _set_smoke_runtime(config: dict[str, Any], *, output_time_step: float = 90.0) -> None:
    set_analysis_horizon_years(config, 1)
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
    parameter_specs: dict[str, dict[str, Any]] | None = None,
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
            )
            config = _apply_psa_sample(
                config,
                current_config,
                strategy=strategy,
                sample=sample,
                parameter_specs=parameter_specs,
            )
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
                        **sample,
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
    required_outcomes = {
        "total_infant_cases",
        "annualized_infant_cases_per_100k",
        "total_child_adolescent_cases",
        "annualized_child_adolescent_cases_per_100k",
    }
    missing_outcomes = sorted(required_outcomes.difference(out.columns))
    if missing_outcomes:
        raise KeyError(
            "Joint PSA outcomes are missing endpoint columns required for infant and "
            f"primary <18 ranking: {missing_outcomes}"
        )
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
        "total_child_adolescent_cases",
        "annualized_child_adolescent_cases_per_100k",
        "relative_reduction_infant_cases_vs_current",
        "within_10_percent_of_best",
        *EXPECTED_PARAMETER_NAMES,
    ]
    for optional in ("sample_design", "uncertainty_schema_version"):
        if optional in out.columns:
            keep.append(optional)
    return out.loc[:, keep].sort_values(["psa_sample_id", "country", "rank", "strategy"]).reset_index(drop=True)


def _rank_programme_under18_summary(
    rank_samples: pd.DataFrame,
    strategies: tuple[str, ...] = PROGRAMME_ONLY_STRATEGIES,
) -> pd.DataFrame:
    """Rank implementable programme scenarios on the manuscript primary endpoint.

    The selected-parameter PSA originally retained only infant ranks even though
    every simulation also produced the child/adolescent (<18 years) endpoint.
    This endpoint-specific table prevents an infant ranking from being used as
    evidence for the manuscript's primary programme-only ranking.
    """

    required = {
        "psa_sample_id",
        "country",
        "strategy",
        "total_child_adolescent_cases",
        "annualized_child_adolescent_cases_per_100k",
    }
    missing = sorted(required.difference(rank_samples.columns))
    if missing:
        raise KeyError(f"Joint PSA rank samples are missing primary-endpoint columns: {missing}")

    data = rank_samples.copy()
    data["psa_sample_id"] = pd.to_numeric(data["psa_sample_id"], errors="raise").astype(int)
    current = data.loc[
        data["strategy"].eq("current"),
        [
            "country",
            "psa_sample_id",
            "total_child_adolescent_cases",
            "annualized_child_adolescent_cases_per_100k",
        ],
    ].rename(
        columns={
            "total_child_adolescent_cases": "current_total_child_adolescent_cases",
            "annualized_child_adolescent_cases_per_100k": (
                "current_annualized_child_adolescent_cases_per_100k"
            ),
        }
    )
    out = data.loc[data["strategy"].isin(strategies)].copy()
    observed = set(out["strategy"].astype(str))
    missing_strategies = sorted(set(strategies).difference(observed))
    if missing_strategies:
        raise ValueError(f"Primary-endpoint joint PSA is missing programme strategies: {missing_strategies}")

    group_keys = ["country", "psa_sample_id"]
    out["rank"] = out.groupby(group_keys)["total_child_adolescent_cases"].rank(
        method="min",
        ascending=True,
    )
    best = out.groupby(group_keys, as_index=False)["total_child_adolescent_cases"].min().rename(
        columns={"total_child_adolescent_cases": "best_total_child_adolescent_cases"}
    )
    out = out.merge(best, on=group_keys, how="left")
    out = out.merge(current, on=group_keys, how="left", validate="many_to_one")
    if out["current_total_child_adolescent_cases"].isna().any():
        raise ValueError("Primary-endpoint joint PSA is missing current-practice comparators")
    out["relative_reduction_under18_cases_vs_current"] = 1.0 - (
        out["total_child_adolescent_cases"]
        / out["current_total_child_adolescent_cases"].replace(0, np.nan)
    )
    out["within_10_percent_of_best"] = out["total_child_adolescent_cases"] <= (
        1.10 * out["best_total_child_adolescent_cases"]
    )
    keep = [
        "psa_sample_id",
        "country",
        "strategy",
        "rank",
        "total_child_adolescent_cases",
        "annualized_child_adolescent_cases_per_100k",
        "relative_reduction_under18_cases_vs_current",
        "within_10_percent_of_best",
    ]
    keep.extend(column for column in EXPECTED_PARAMETER_NAMES if column in out.columns)
    for optional in ("sample_design", "uncertainty_schema_version"):
        if optional in out.columns:
            keep.append(optional)
    return out.loc[:, keep].sort_values(
        ["psa_sample_id", "country", "rank", "strategy"]
    ).reset_index(drop=True)


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


def _acceptability_from_programme_under18_rank_samples(
    rank_samples: pd.DataFrame,
    strategies: tuple[str, ...] = PROGRAMME_ONLY_STRATEGIES,
) -> pd.DataFrame:
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
                pd.to_numeric(
                    strategy_group["annualized_child_adolescent_cases_per_100k"], errors="coerce"
                )
                if n
                else pd.Series(dtype=float)
            )
            reductions = (
                pd.to_numeric(
                    strategy_group["relative_reduction_under18_cases_vs_current"], errors="coerce"
                )
                if n
                else pd.Series(dtype=float)
            )
            for rank in ranks:
                rows.append(
                    {
                        "country": country_label,
                        "strategy": strategy,
                        "rank": rank,
                        "rank_acceptability_frequency": float(np.mean(rank_values == rank)) if n else np.nan,
                        "frequency_rank_1": float(np.mean(rank_values == 1)) if n else np.nan,
                        "frequency_top_2": float(np.mean(rank_values <= 2)) if n else np.nan,
                        "frequency_top_3": float(np.mean(rank_values <= 3)) if n else np.nan,
                        "frequency_within_10_percent_of_best": float(
                            strategy_group["within_10_percent_of_best"].mean()
                        )
                        if n
                        else np.nan,
                        "mean_rank": float(np.nanmean(rank_values)) if n else np.nan,
                        "median_rank": float(np.nanmedian(rank_values)) if n else np.nan,
                        "median_under18_cases_per_100k": float(cases.median(skipna=True)) if n else np.nan,
                        "q025_under18_cases_per_100k": float(cases.quantile(0.025)) if n else np.nan,
                        "q975_under18_cases_per_100k": float(cases.quantile(0.975)) if n else np.nan,
                        "median_relative_reduction_vs_current": (
                            float(reductions.median(skipna=True)) if n else np.nan
                        ),
                        "n_psa_samples": n,
                        "n_rank_observations": int(len(strategy_group)) if n else 0,
                        "interpretation": (
                            "Selected-parameter deterministic frequency; not a posterior probability."
                        ),
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


def _under18_programme_run_summary(acceptability: pd.DataFrame) -> pd.DataFrame:
    rank1 = acceptability.loc[acceptability["rank"].eq(1)].copy()
    return rank1.sort_values(
        ["country", "frequency_rank_1", "frequency_top_2"],
        ascending=[True, False, False],
    )


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


def _retain_matching_completed_draws(
    completed: set[int],
    existing: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[set[int], pd.DataFrame]:
    """Reject resumed cells generated under a different parameter design."""

    if not completed or existing.empty:
        return set(), pd.DataFrame()
    parameter_columns = [
        column
        for column in samples.columns
        if column not in {"psa_sample_id", "sample_design", "uncertainty_schema_version"}
    ]
    expected = samples.set_index("psa_sample_id")
    matching: set[int] = set()
    for sample_id in sorted(completed):
        rows = existing.loc[existing["psa_sample_id"].eq(sample_id)]
        if rows.empty or sample_id not in expected.index:
            continue
        first = rows.iloc[0]
        if any(column not in first.index for column in parameter_columns):
            continue
        if "sample_design" not in first.index or str(first["sample_design"]) != str(
            expected.loc[sample_id, "sample_design"]
        ):
            continue
        if "uncertainty_schema_version" not in first.index:
            continue
        try:
            actual_schema = int(first["uncertainty_schema_version"])
            target_schema = int(expected.loc[sample_id, "uncertainty_schema_version"])
        except (TypeError, ValueError):
            continue
        if actual_schema != target_schema:
            continue
        actual = pd.to_numeric(first[parameter_columns], errors="coerce").to_numpy(dtype=float)
        target = expected.loc[sample_id, parameter_columns].to_numpy(dtype=float)
        if np.isfinite(actual).all() and np.allclose(actual, target, rtol=1e-10, atol=1e-12):
            matching.add(int(sample_id))
    retained = existing.loc[existing["psa_sample_id"].isin(matching)].copy()
    return matching, retained


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
    _validate_calibration_inputs(countries)
    registry = configs.get("parameter_distributions", {})
    schema_version = validate_uncertainty_registry_schema(registry)
    joint_settings = registry.get("joint_rank_psa", {}) if isinstance(registry, dict) else {}
    parameter_specs = joint_settings.get("parameters", {}) if isinstance(joint_settings, dict) else {}
    parameter_specs = parameter_specs or _default_parameter_specs()
    samples = _sample_table(
        sample_size,
        seed,
        parameter_specs,
        schema_version=schema_version,
    )
    write_dataframe(samples, SAMPLE_PATH)

    if resume:
        resume_current, resume_reason = _resume_metadata_available()
        if resume_current:
            completed, completed_rank = _completed_rank_samples(
                RANK_SAMPLE_PATH,
                countries=countries,
                strategies=strategies,
            )
            completed, completed_rank = _retain_matching_completed_draws(
                completed,
                completed_rank,
                samples,
            )
        else:
            print(f"Ignoring cached joint-PSA outcomes: {resume_reason}")
            completed, completed_rank = set(), pd.DataFrame()
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
                    parameter_specs=parameter_specs,
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
        enforce_calibration_status(summary, stem=batch_stem)
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
        programme_under18_rank = _rank_programme_under18_summary(combined_rank)
        _write_incremental(programme_under18_rank, UNDER18_PROGRAMME_RANK_SAMPLE_PATH)
        programme_under18_acceptability = _acceptability_from_programme_under18_rank_samples(
            programme_under18_rank
        )
        write_dataframe(
            programme_under18_acceptability,
            UNDER18_PROGRAMME_ACCEPTABILITY_PATH,
        )
        write_dataframe(
            _under18_programme_run_summary(programme_under18_acceptability),
            UNDER18_PROGRAMME_RUN_SUMMARY_PATH,
        )

    combined_outcomes = pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame()
    complete_outcomes = _complete_outcome_rows(combined_outcomes, countries=countries, strategies=strategies)
    rank_samples = _rank_sample_summary(complete_outcomes) if not complete_outcomes.empty else _read_existing(RANK_SAMPLE_PATH)
    acceptability = _acceptability_from_rank_samples(rank_samples, strategies)
    run_summary = _run_summary(acceptability)
    programme_under18_rank = _rank_programme_under18_summary(rank_samples)
    programme_under18_acceptability = _acceptability_from_programme_under18_rank_samples(
        programme_under18_rank
    )
    programme_under18_run_summary = _under18_programme_run_summary(
        programme_under18_acceptability
    )
    write_dataframe(acceptability, ACCEPTABILITY_PATH)
    write_dataframe(run_summary, RUN_SUMMARY_PATH)
    _write_incremental(programme_under18_rank, UNDER18_PROGRAMME_RANK_SAMPLE_PATH)
    write_dataframe(programme_under18_acceptability, UNDER18_PROGRAMME_ACCEPTABILITY_PATH)
    write_dataframe(programme_under18_run_summary, UNDER18_PROGRAMME_RUN_SUMMARY_PATH)
    if not rank_samples.empty:
        # Persist a compact, complete checkpoint rather than a sparse union of
        # prior compact ranks and only the most recent raw scenario batch.
        write_dataframe(rank_samples, SIMULATION_SUMMARY_PATH)
    if keep_timeseries and timeseries_frames:
        write_dataframe(pd.concat(timeseries_frames, ignore_index=True), SIMULATION_TS_PATH)

    metadata = current_run_metadata(
        STEM,
        row_counts={
            "parameter_samples": int(len(samples)),
            "rank_samples": int(len(rank_samples)),
            "rank_acceptability": int(len(acceptability)),
            "run_summary": int(len(run_summary)),
            "under18_programme_rank_samples": int(len(programme_under18_rank)),
            "under18_programme_rank_acceptability": int(len(programme_under18_acceptability)),
            "under18_programme_run_summary": int(len(programme_under18_run_summary)),
        },
    )
    metadata.update(
        {
            "sample_size_requested": int(sample_size),
            "sample_seed": int(seed),
            "countries": list(countries),
            "strategies": list(strategies),
            "programme_only_strategies": list(PROGRAMME_ONLY_STRATEGIES),
            "primary_policy_endpoint": "annualized_child_adolescent_cases_per_100k",
            "resume": bool(resume),
            "sample_batch_size": int(batch_size),
            "smoke_runtime": bool(smoke_runtime),
            "keep_timeseries": bool(keep_timeseries),
            "uncertainty_schema_version": schema_version,
            "sample_design": SAMPLE_DESIGN,
            "figure2b_parameter_names": list(FIGURE2B_PARAMETER_NAMES),
            "parameter_time_scopes": _validated_parameter_time_scopes(
                parameter_specs,
                expected_names=FIGURE2B_PARAMETER_NAMES,
            ),
            "excluded_dead_dimensions": ["resistance_management_uptake"],
            "parameter_distributions": {
                name: validate_distribution_spec(spec, context=f"joint PSA parameter {name!r}")
                for name, spec in parameter_specs.items()
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
    registry = configs.get("parameter_distributions", {})
    validate_uncertainty_registry_schema(registry)
    joint_settings = registry.get("joint_rank_psa", {}) if isinstance(registry, dict) else {}
    parser = argparse.ArgumentParser(
        description=(
            "Run the six-parameter Figure 2b programme rank-stability diagnostic "
            "and its separate infant-endpoint secondary ranking."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=int(joint_settings.get("sample_size", 128)),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(joint_settings.get("random_seed", 20260521)),
    )
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

    publication_countries = tuple(publication_country_names(configs))
    countries = _parse_csv_tuple(args.countries, publication_countries)
    outside_publication_scope = sorted(set(countries) - set(publication_countries))
    if outside_publication_scope:
        raise ValueError(
            "Joint PSA publication outputs exclude countries outside the prespecified "
            "publication set: " + ", ".join(outside_publication_scope)
        )
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
