"""Optional non-publication legacy paired-posterior research runner.

This module is not a Figure 2c interval source and is not part of the
publication pipeline.  Publication Figure 2c uses the separate parametric
bootstrap estimation-confidence-interval runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    current_run_metadata,
    execute_scenario_summary_list,
    load_configs,
    publication_country_names,
    read_run_metadata,
    write_run_metadata,
)
from src_python.simulation.check_bayesian_quality import (
    check_bayesian_quality,
    posterior_quality_failures,
)
from src_python.simulation.run_bayesian_uncertainty import (
    RETIRED_MISLABELED_OUTPUT_STEMS,
)
from src_python.simulation.programme_uncertainty_helpers import (
    INTERVENTION_UNCERTAINTY_DEFAULTS,
    INTERVENTION_UNCERTAINTY_PREFIX,
    PRIMARY_RATE,
    PRIMARY_REDUCTION,
    PRIMARY_TOTAL,
    PROGRAMME_STRATEGIES as FIGURE2C_STRATEGIES,
    PROGRAMME_STRATEGY_LABELS as STRATEGY_LABELS,
    PROPAGATED_UNCERTAINTY_PARAMETERS,
    _beta_ab,
    _draw_beta_prior,
    _sample_from_row as _posterior_sample_from_row,
    _strategy_config_from_sampled_current,
    _triangular_parameters,
    apply_intervention_inputs as _apply_intervention_uncertainty,
    attach_intervention_inputs as _attach_intervention_uncertainty,
    build_programme_scenarios as _build_programme_scenarios,
    draw_intervention_inputs as _intervention_uncertainty_draw,
    pair_programme_draws as _paired_draws,
    posterior_stem_from_sample_path as _posterior_stem_from_sample_path,
)
from src_python.utils.io import project_path, write_dataframe


STEM = "figure2c_paired_programme_uncertainty"
ANALYSIS_ROLE = "optional_nonpublication_legacy_research"
PUBLICATION_PATH = False
FIGURE2C_INTERVAL_SOURCE = False
FULL_CRI_POSTERIOR_STEM = "bayesian_uncertainty_figure2c_joint"
RETIRED_POSTERIOR_STEMS = RETIRED_MISLABELED_OUTPUT_STEMS
DEFAULT_POSTERIOR_SAMPLE_PATH = project_path(
    "outputs", "simulations", f"{FULL_CRI_POSTERIOR_STEM}_posterior_samples.parquet"
)
SCENARIO_SUMMARY_PATH = project_path("outputs", "summaries", f"{STEM}_summary.csv")
DRAW_PATH = project_path("outputs", "tables", "figure2c_programme_paired_credible_interval_draws.csv")
INTERVAL_PATH = project_path("outputs", "summaries", "figure2c_programme_paired_credible_intervals.csv")
POSTERIOR_AUDIT_PATH = project_path("outputs", "metadata", f"{STEM}_uncertainty_parameter_audit.csv")
RETIRED_INTERVAL_PATHS = (
    project_path("outputs", "tables", "figure2c_programme_paired_conditional_interval_draws.csv"),
    project_path("outputs", "summaries", "figure2c_programme_paired_conditional_intervals.csv"),
    project_path("outputs", "metadata", f"{STEM}_posterior_parameter_audit.csv"),
)

# Backward-compatible import alias. Beta/reporting are country-specific posterior
# coordinates; the remaining dimensions are shared across countries and updated
# by the product of country likelihoods in this legacy joint research route.
JOINT_POSTERIOR_PARAMETERS = PROPAGATED_UNCERTAINTY_PARAMETERS
EXTERNAL_STRUCTURAL_PRIOR_PARAMETERS = tuple(
    parameter
    for parameter in JOINT_POSTERIOR_PARAMETERS
    if parameter not in {"beta_S", "reporting_multiplier"}
)
STRUCTURAL_POSTERIOR_PARAMETERS = EXTERNAL_STRUCTURAL_PRIOR_PARAMETERS
MODULAR_INFERENCE_STRUCTURES = {
    "modular_cut",
    "modular_hierarchical_cut",
    "reference_structure_state_space_exact_importance_cut",
}
FULL_JOINT_INFERENCE_STRUCTURES = {
    "cross_country_joint_state_space_exact_importance",  # retired numerical route
    "cross_country_joint_state_space_tempered_smc",
}

MINIMUM_CREDIBLE_INTERVAL_DRAWS = 2000


def _posterior_metadata(path: str | Path) -> tuple[str | None, dict[str, Any]]:
    stem = _posterior_stem_from_sample_path(path)
    if stem is None:
        return None, {}
    if stem in RETIRED_POSTERIOR_STEMS:
        raise ValueError(
            f"Posterior stem {stem!r} is retired because it incorrectly labels "
            "conditional uncertainty as a full joint posterior. Rerun with "
            f"{FULL_CRI_POSTERIOR_STEM!r}."
        )
    try:
        return stem, read_run_metadata(stem)
    except FileNotFoundError:
        return stem, {}


def _posterior_parameter_audit(samples: pd.DataFrame, *, countries: list[str]) -> pd.DataFrame:
    missing = [parameter for parameter in JOINT_POSTERIOR_PARAMETERS if parameter not in samples.columns]
    if missing:
        raise KeyError(f"Posterior samples are missing required joint parameters: {', '.join(missing)}")
    rows: list[dict[str, Any]] = []
    for country in countries:
        group = samples.loc[samples["country"].astype(str).eq(country)]
        if group.empty:
            raise ValueError(f"No posterior samples available for country: {country}")
        for parameter in JOINT_POSTERIOR_PARAMETERS:
            values = pd.to_numeric(group[parameter], errors="coerce").dropna()
            if values.empty:
                raise ValueError(f"No finite posterior values for {country}/{parameter}")
            q025 = float(values.quantile(0.025))
            q500 = float(values.quantile(0.500))
            q975 = float(values.quantile(0.975))
            width = q975 - q025
            variable = bool(values.nunique(dropna=True) > 1 and width > max(abs(q500) * 1e-8, 1e-10))
            rows.append(
                {
                    "country": country,
                    "parameter": parameter,
                    "n_samples": int(len(values)),
                    "n_unique": int(values.nunique(dropna=True)),
                    "median": q500,
                    "q025": q025,
                    "q975": q975,
                    "q95_width": float(width),
                    "relative_q95_width": float(width / abs(q500)) if abs(q500) > 0 else np.nan,
                    "variable": variable,
                }
            )
    return pd.DataFrame(rows)


def _structural_pairing_requested(samples: pd.DataFrame) -> bool:
    structures: set[str] = set()
    if "inference_structure" in samples.columns:
        structures = {
            str(value).strip().lower()
            for value in samples["inference_structure"].dropna().unique()
            if str(value).strip()
        }
        if len(structures) > 1:
            raise ValueError(
                "Posterior samples mix incompatible inference structures: "
                f"{sorted(structures)}"
            )
    return bool(structures.intersection(MODULAR_INFERENCE_STRUCTURES)) or (
        "structural_draw_id" in samples.columns
    )


def _validate_structural_source_pairing(
    samples: pd.DataFrame,
    *,
    countries: list[str],
) -> bool:
    """Validate the shared structural-draw contract of a modular posterior.

    The modular sampler emits the same ``(chain, draw, structural_draw_id)``
    design for every country. Country surveillance changes only the
    conditional beta/reporting draw. This legacy analysis must therefore subsample those
    source positions jointly rather than construct unrelated country draws.
    """

    requested = _structural_pairing_requested(samples)
    if not requested:
        return False
    if "structural_draw_id" not in samples.columns:
        raise KeyError(
            "Modular posterior samples must retain structural_draw_id for "
            "cross-country pairing"
        )
    missing_keys = [column for column in ("country", "chain", "draw") if column not in samples.columns]
    if missing_keys:
        raise KeyError(
            "Structurally paired posterior samples require source keys: "
            + ", ".join(missing_keys)
        )

    requested_countries = [str(country) for country in countries]
    data = samples.loc[
        samples["country"].astype(str).isin(requested_countries)
    ].copy()
    observed_countries = set(data["country"].astype(str))
    if observed_countries != set(requested_countries):
        missing = sorted(set(requested_countries) - observed_countries)
        raise ValueError(f"No structurally paired posterior samples for countries: {missing}")

    structural_ids = pd.to_numeric(data["structural_draw_id"], errors="coerce")
    if structural_ids.isna().any() or not np.isfinite(structural_ids.to_numpy(dtype=float)).all():
        raise ValueError("structural_draw_id must be finite for every modular posterior row")
    if not np.equal(structural_ids, np.floor(structural_ids)).all():
        raise ValueError("structural_draw_id must contain integer identifiers")
    data["structural_draw_id"] = structural_ids.astype(np.int64)
    if data[["country", "chain", "draw"]].isna().any().any():
        raise ValueError("country, chain, and draw source keys cannot be missing")
    duplicated = data.duplicated(["country", "chain", "draw"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Modular posterior source keys must be unique within country; "
            f"found {int(duplicated.sum())} duplicate rows"
        )

    source_alignment = data.groupby(["chain", "draw"], dropna=False).agg(
        rows=("country", "size"),
        countries=("country", lambda values: values.astype(str).nunique()),
        structural_ids=("structural_draw_id", "nunique"),
    )
    aligned = (
        source_alignment["rows"].eq(len(requested_countries))
        & source_alignment["countries"].eq(len(requested_countries))
        & source_alignment["structural_ids"].eq(1)
    )
    if not aligned.all():
        examples = source_alignment.loc[~aligned].head(10).reset_index().to_dict("records")
        raise ValueError(
            "structural_draw_id is not aligned across countries for common "
            f"(chain, draw) source positions: {examples}"
        )

    # A structural identifier must refer to the same external-evidence draw in
    # every country. Use a tight numerical tolerance to allow serialization
    # round-off while rejecting country-specific structural re-randomization.
    for parameter in EXTERNAL_STRUCTURAL_PRIOR_PARAMETERS:
        if parameter not in data.columns:
            continue
        values = pd.to_numeric(data[parameter], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Structural posterior parameter {parameter} must be finite for every row"
            )
        grouped = pd.DataFrame(
            {"structural_draw_id": data["structural_draw_id"], "value": values}
        ).groupby("structural_draw_id")["value"]
        minima = grouped.min()
        maxima = grouped.max()
        tolerance = 1e-10 + 1e-10 * np.maximum(minima.abs(), maxima.abs())
        inconsistent = (maxima - minima) > tolerance
        if inconsistent.any():
            bad_ids = inconsistent.index[inconsistent].astype(int).tolist()[:10]
            raise ValueError(
                f"structural_draw_id does not map to a shared {parameter} value; "
                f"inconsistent IDs: {bad_ids}"
            )
    return True


def _validate_joint_posterior_samples(
    samples: pd.DataFrame,
    *,
    countries: list[str],
    posterior_metadata: dict[str, Any],
    posterior_stem: str | None,
    allow_conditional_posterior: bool,
    min_variable_parameters: int,
) -> pd.DataFrame:
    if "country" not in samples.columns:
        raise KeyError("Posterior samples must include a country column")
    audit = _posterior_parameter_audit(samples, countries=countries)
    _validate_structural_source_pairing(samples, countries=countries)
    if allow_conditional_posterior:
        return audit

    if not posterior_metadata:
        raise ValueError(
            "Audited legacy paired-posterior research requires run metadata. "
            f"No metadata found for posterior sample stem {posterior_stem!r}."
        )
    sampler = str(posterior_metadata.get("sampler", "")).lower()
    scope = str(posterior_metadata.get("uncertainty_scope", "")).lower()
    if sampler in {
        "state_space_exact_importance_cut",
        "multi_country_joint_state_importance",
        "multi_country_joint_tempered_smc",
    }:
        process_columns = sorted(
            column
            for column in samples.columns
            if str(column).startswith("log_beta_process_")
        )
        process_years: list[int] = []
        for column in process_columns:
            try:
                process_years.append(int(str(column).removeprefix("log_beta_process_")))
            except ValueError as exc:
                raise ValueError(f"Invalid annual process draw column: {column}") from exc
        configs = load_configs()
        forecast_start = int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_start_date"]).year
        )
        forecast_end = int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_end_date"]).year
        )
        expected_forecast_years = set(range(forecast_start, forecast_end + 1))
        if (
            not process_years
            or process_years != list(range(min(process_years), max(process_years) + 1))
            or not expected_forecast_years.issubset(process_years)
        ):
            raise ValueError(
                "State-space posterior samples require a complete consecutive annual "
                f"process path through {forecast_end}; observed years={process_years}"
            )
        required_process_metadata = {
            "forecast_process_rho",
            "forecast_process_innovation_sd",
            "forecast_process_years",
            "historical_process_end_year",
        }
        missing_process_metadata = required_process_metadata.difference(samples.columns)
        if missing_process_metadata:
            raise ValueError(
                "State-space posterior samples are missing process metadata: "
                + ", ".join(sorted(missing_process_metadata))
            )
        process_settings = configs["baseline"]["calibration"]["process_model"]
        expected_rho = float(process_settings["ar1_rho"])
        expected_sd = float(process_settings["log_beta_innovation_sd"])
        for country, group in samples.groupby("country", sort=False):
            present_columns = [
                column for column in process_columns if group[column].notna().any()
            ]
            present_years = [
                int(str(column).removeprefix("log_beta_process_"))
                for column in present_columns
            ]
            if (
                not present_years
                or present_years
                != list(range(min(present_years), max(present_years) + 1))
                or max(present_years) != forecast_end
                or not expected_forecast_years.issubset(present_years)
            ):
                raise ValueError(
                    f"Incomplete annual process path for {country}: {present_years}"
                )
            values = (
                group.loc[:, present_columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"State-space annual process draws contain missing/non-finite values for {country}"
                )
            rho_values = pd.to_numeric(group["forecast_process_rho"], errors="coerce")
            sd_values = pd.to_numeric(
                group["forecast_process_innovation_sd"], errors="coerce"
            )
            historical_end = pd.to_numeric(
                group["historical_process_end_year"], errors="coerce"
            )
            future_counts = pd.to_numeric(
                group["forecast_process_years"], errors="coerce"
            )
            if (
                not np.allclose(rho_values, expected_rho)
                or not np.allclose(sd_values, expected_sd)
            ):
                raise ValueError(
                    f"State-space process hyperparameters differ from config for {country}"
                )
            if historical_end.nunique(dropna=False) != 1:
                raise ValueError(f"Historical process end year varies within {country}")
            expected_future_count = forecast_end - int(historical_end.iloc[0])
            if not future_counts.eq(expected_future_count).all():
                raise ValueError(
                    f"forecast_process_years does not match the path for {country}"
                )
    if sampler == "beta_grid" or scope == "conditional_beta_grid":
        raise ValueError(
            "Refusing to compute legacy paired uncertainty from one-parameter "
            "conditional beta-grid samples. "
            f"sampler={posterior_metadata.get('sampler')!r}, "
            f"uncertainty_scope={posterior_metadata.get('uncertainty_scope')!r}."
        )
    if (
        "sampling_method" in samples.columns
        and not samples["sampling_method"].empty
        and samples["sampling_method"].astype(str).str.lower().eq("beta_grid").all()
    ):
        raise ValueError(
            "Refusing to compute legacy paired uncertainty from rows labelled "
            "sampling_method=beta_grid."
        )
    fixed_parameters = sorted(str(parameter) for parameter in posterior_metadata.get("fixed_parameters", []) or [])
    if fixed_parameters or bool(posterior_metadata.get("fix_durations", False)):
        raise ValueError(
            "Legacy paired uncertainty requires all registered dimensions to vary. "
            f"fixed_parameters={fixed_parameters}, fix_durations={posterior_metadata.get('fix_durations')}."
        )

    quality_failures = posterior_quality_failures(
        posterior_metadata,
        require_recommended=True,
    )
    if quality_failures:
        raise ValueError(
            "Legacy paired uncertainty requires samples that pass the recommended "
            "quality gate: " + "; ".join(quality_failures)
        )

    variable_counts = audit.groupby("country")["variable"].sum().astype(int)
    bad = variable_counts.loc[variable_counts.lt(int(min_variable_parameters))]
    if not bad.empty:
        details = ", ".join(f"{country}: {count}/{len(JOINT_POSTERIOR_PARAMETERS)}" for country, count in bad.items())
        raise ValueError(
            "Posterior samples do not show real multi-parameter variation for all countries. "
            f"Variable parameter counts: {details}."
        )
    return audit


def _balanced_group_quotas(
    values: pd.Series,
    *,
    total: int,
    rng: np.random.Generator,
) -> dict[Any, int]:
    capacities = values.value_counts(dropna=False).to_dict()
    groups = list(capacities)
    if not groups or total > int(sum(capacities.values())):
        raise ValueError("Requested posterior draws exceed available joint source positions")
    rng.shuffle(groups)
    base = total // len(groups)
    remainder = total % len(groups)
    quotas = {
        group: min(base + (1 if index < remainder else 0), int(capacities[group]))
        for index, group in enumerate(groups)
    }
    remaining = int(total - sum(quotas.values()))
    while remaining > 0:
        updated = False
        for group in groups:
            if quotas[group] < int(capacities[group]):
                quotas[group] += 1
                remaining -= 1
                updated = True
                if remaining == 0:
                    break
        if not updated:
            raise ValueError("Could not allocate balanced posterior draw quotas")
        rng.shuffle(groups)
    return quotas


def _jointly_stratified_source_indices(
    source_positions: pd.DataFrame,
    *,
    chain_quotas: dict[Any, int],
    structural_quotas: dict[Any, int],
    rng: np.random.Generator,
) -> list[int]:
    """Meet chain and structural-draw quotas using shared source positions."""

    pools: dict[tuple[Any, int], list[int]] = {}
    for (chain, structural_id), positions in source_positions.groupby(
        ["chain", "structural_draw_id"], sort=False
    ).groups.items():
        shuffled = np.asarray(list(positions), dtype=int)
        rng.shuffle(shuffled)
        pools[(chain, int(structural_id))] = shuffled.tolist()

    remaining_chain = {chain: int(count) for chain, count in chain_quotas.items()}
    remaining_structural = {
        int(structural_id): int(count)
        for structural_id, count in structural_quotas.items()
    }
    selected: list[int] = []
    while sum(remaining_structural.values()) > 0:
        feasible: dict[int, list[Any]] = {}
        for structural_id, count in remaining_structural.items():
            if count <= 0:
                continue
            feasible[structural_id] = [
                chain
                for chain, chain_count in remaining_chain.items()
                if chain_count > 0 and pools.get((chain, structural_id))
            ]
        impossible = [
            structural_id
            for structural_id, count in remaining_structural.items()
            if count > 0 and len(feasible.get(structural_id, ())) < 1
        ]
        if impossible:
            raise ValueError(
                "Could not jointly satisfy chain and structural-draw selection quotas; "
                f"blocked structural IDs: {impossible[:10]}"
            )

        # Allocate the most constrained structural draw first. Remaining-chain
        # capacity breaks ties, with RNG used only among equivalent choices.
        min_options = min(len(chains) for chains in feasible.values())
        structural_candidates = [
            structural_id
            for structural_id, chains in feasible.items()
            if len(chains) == min_options
        ]
        max_need = max(remaining_structural[structural_id] for structural_id in structural_candidates)
        structural_candidates = [
            structural_id
            for structural_id in structural_candidates
            if remaining_structural[structural_id] == max_need
        ]
        structural_id = int(rng.choice(np.asarray(structural_candidates, dtype=int)))
        chain_candidates = feasible[structural_id]
        max_chain_need = max(remaining_chain[chain] for chain in chain_candidates)
        chain_candidates = [
            chain for chain in chain_candidates if remaining_chain[chain] == max_chain_need
        ]
        chain = chain_candidates[int(rng.integers(0, len(chain_candidates)))]
        selected.append(pools[(chain, structural_id)].pop())
        remaining_chain[chain] -= 1
        remaining_structural[structural_id] -= 1

    if any(count != 0 for count in remaining_chain.values()):
        raise ValueError("Joint structural selection did not meet balanced chain quotas")
    return selected


def _select_structurally_paired_samples(
    samples: pd.DataFrame,
    *,
    countries: list[str],
    draws_per_country: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    _validate_structural_source_pairing(samples, countries=countries)
    data = samples.loc[samples["country"].astype(str).isin(countries)].copy()
    data["country"] = data["country"].astype(str)
    data["structural_draw_id"] = pd.to_numeric(
        data["structural_draw_id"], errors="raise"
    ).astype(np.int64)
    data = data.drop(columns=["posterior_draw"], errors="ignore")

    source_positions = (
        data.loc[:, ["chain", "draw", "structural_draw_id"]]
        .drop_duplicates(["chain", "draw"])
        .reset_index(drop=True)
    )
    if len(source_positions) < draws_per_country:
        raise ValueError(
            "Not enough aligned conditional-state/external-design positions for the legacy analysis: "
            f"available={len(source_positions)}, requested={draws_per_country}"
        )

    chain_quotas = _balanced_group_quotas(
        source_positions["chain"],
        total=draws_per_country,
        rng=rng,
    )
    structural_quotas = _balanced_group_quotas(
        source_positions["structural_draw_id"],
        total=draws_per_country,
        rng=rng,
    )
    selected_indices = _jointly_stratified_source_indices(
        source_positions,
        chain_quotas=chain_quotas,
        structural_quotas=structural_quotas,
        rng=rng,
    )
    selection = (
        source_positions.loc[selected_indices]
        .sort_values(["chain", "draw"], kind="stable")
        .reset_index(drop=True)
    )
    selection["posterior_draw"] = np.arange(1, len(selection) + 1, dtype=int)

    selected = data.merge(
        selection,
        on=["chain", "draw", "structural_draw_id"],
        how="inner",
        validate="many_to_one",
    )
    expected_rows = len(countries) * draws_per_country
    if len(selected) != expected_rows:
        raise ValueError(
            "Joint structural posterior selection did not return one country row "
            f"per selected source position: rows={len(selected)}, expected={expected_rows}"
        )
    paired_ids = selected.groupby("posterior_draw")["structural_draw_id"].nunique()
    if not paired_ids.eq(1).all():
        raise RuntimeError("Internal error: selected posterior draws lost structural pairing")

    country_order = {str(country): index for index, country in enumerate(countries)}
    selected["_country_order"] = selected["country"].map(country_order)
    return (
        selected.sort_values(["_country_order", "posterior_draw"], kind="stable")
        .drop(columns="_country_order")
        .reset_index(drop=True)
    )


def _select_posterior_samples(
    samples: pd.DataFrame,
    *,
    countries: list[str],
    draws_per_country: int,
    seed: int,
) -> pd.DataFrame:
    if draws_per_country < 1:
        raise ValueError("draws_per_country must be positive")
    rng = np.random.default_rng(seed)
    if _structural_pairing_requested(samples):
        return _select_structurally_paired_samples(
            samples,
            countries=countries,
            draws_per_country=draws_per_country,
            rng=rng,
        )
    selected_frames: list[pd.DataFrame] = []
    for country in countries:
        group = samples.loc[samples["country"].astype(str).eq(country)].reset_index(drop=True)
        if group.empty:
            raise ValueError(f"No posterior samples available for country: {country}")
        if len(group) > draws_per_country:
            if "chain" in group.columns and group["chain"].nunique(dropna=True) > 1:
                sampled_indices: list[int] = []
                chain_values = list(group["chain"].dropna().sort_values().unique())
                rng.shuffle(chain_values)
                base = draws_per_country // len(chain_values)
                remainder = draws_per_country % len(chain_values)
                quotas = {
                    chain: base + (1 if index < remainder else 0)
                    for index, chain in enumerate(chain_values)
                }
                capacities = group.groupby("chain").size().to_dict()
                remaining = 0
                for chain, quota in list(quotas.items()):
                    capacity = int(capacities.get(chain, 0))
                    if quota > capacity:
                        remaining += quota - capacity
                        quotas[chain] = capacity
                while remaining > 0:
                    updated = False
                    for chain in chain_values:
                        capacity = int(capacities.get(chain, 0))
                        if quotas[chain] < capacity:
                            quotas[chain] += 1
                            remaining -= 1
                            updated = True
                            if remaining == 0:
                                break
                    if not updated:
                        break
                for chain in chain_values:
                    chain_index = group.index[group["chain"].eq(chain)].to_numpy(dtype=int)
                    quota = int(quotas[chain])
                    if quota > 0:
                        sampled_indices.extend(rng.choice(chain_index, quota, replace=False).tolist())
                if len(sampled_indices) < draws_per_country:
                    available = np.setdiff1d(group.index.to_numpy(dtype=int), np.asarray(sampled_indices, dtype=int))
                    top_up = int(min(draws_per_country - len(sampled_indices), len(available)))
                    if top_up > 0:
                        sampled_indices.extend(rng.choice(available, top_up, replace=False).tolist())
                group = group.iloc[np.sort(np.asarray(sampled_indices, dtype=int))].copy()
            else:
                group = group.iloc[np.sort(rng.choice(len(group), draws_per_country, replace=False))].copy()
        else:
            group = group.copy()
        group["posterior_draw"] = np.arange(1, len(group) + 1, dtype=int)
        selected_frames.append(group)
    return pd.concat(selected_frames, ignore_index=True)


def _build_scenarios(
    configs: dict[str, Any],
    selected_samples: pd.DataFrame,
    *,
    strategies: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Backward-compatible legacy wrapper around the shared scenario builder."""

    return _build_programme_scenarios(
        configs,
        selected_samples,
        strategies=strategies,
        analysis=STEM,
    )


def _q(values: pd.Series, quantile: float) -> float:
    return float(pd.to_numeric(values, errors="coerce").quantile(quantile))


def _interval_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (country, strategy), group in draws.groupby(["country", "strategy"], sort=True):
        first = group.iloc[0]
        inference_structure = str(first.get("inference_structure", ""))
        if inference_structure in FULL_JOINT_INFERENCE_STRUCTURES:
            interval_type = "95% joint posterior credible interval"
            interval_basis = (
                "Paired current/intervention outcomes propagated from the same "
                "multi-country full-feedback posterior draw. Shared biological "
                "parameters, country beta/reporting, and annual latent transmission "
                "states are updated jointly by all included surveillance likelihoods."
            )
        elif inference_structure == "reference_structure_state_space_exact_importance_cut":
            interval_type = "95% conditional uncertainty interval"
            interval_basis = (
                "Paired exact-target importance-corrected state draws plus a shared "
                "external structural-prior design and draw-level intervention priors. "
                "Process/measurement hyperparameters and the reference structural "
                "calibration are fixed."
            )
        elif inference_structure in {"modular_cut", "modular_hierarchical_cut"}:
            interval_type = "95% conditional modular-cut uncertainty interval"
            interval_basis = (
                "Paired country-conditional beta/reporting grid draws with a shared "
                "external structural-prior design and draw-level intervention priors; "
                "this is not a full joint or hierarchical posterior interval."
            )
        else:
            interval_type = "95% alternative-target posterior interval (diagnostic)"
            interval_basis = (
                "Paired draws from a legacy no-process alternative target with "
                "draw-level intervention implementation/effect priors. This route "
                "is diagnostic and is not eligible for the publication figure."
            )
        rows.append(
            {
                "country": country,
                "strategy": strategy,
                "scenario_key": strategy,
                "scenario_label": first["strategy_label"],
                "outcome": "child_adolescent_cases",
                "reduction_median": _q(group[PRIMARY_REDUCTION], 0.50),
                "reduction_q025": _q(group[PRIMARY_REDUCTION], 0.025),
                "reduction_q25": _q(group[PRIMARY_REDUCTION], 0.25),
                "reduction_q75": _q(group[PRIMARY_REDUCTION], 0.75),
                "reduction_q975": _q(group[PRIMARY_REDUCTION], 0.975),
                "current_rate_median": _q(group["current_rate"], 0.50),
                "current_rate_q025": _q(group["current_rate"], 0.025),
                "current_rate_q975": _q(group["current_rate"], 0.975),
                "intervention_rate_median": _q(group["intervention_rate"], 0.50),
                "intervention_rate_q025": _q(group["intervention_rate"], 0.025),
                "intervention_rate_q975": _q(group["intervention_rate"], 0.975),
                "posterior_draws": int(group["posterior_draw"].nunique()),
                "uncertainty_draws": int(group["posterior_draw"].nunique()),
                "interval_type": interval_type,
                "interval_basis": interval_basis,
            }
        )
    return pd.DataFrame(rows).sort_values(["country", "strategy"]).reset_index(drop=True)


def main(
    *,
    n_jobs: int | None = None,
    draws_per_country: int | None = None,
    posterior_samples_path: str | Path = DEFAULT_POSTERIOR_SAMPLE_PATH,
    countries_filter: list[str] | None = None,
    allow_conditional_posterior: bool = False,
    intervention_uncertainty: bool = True,
    min_variable_parameters: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if allow_conditional_posterior:
        raise ValueError(
            "--allow-conditional-posterior is retired because it bypassed the legacy "
            "research quality checks. Use a separate diagnostic runner/output stem "
            "for exploratory inputs."
        )
    configs = load_configs()
    settings = configs["baseline"].get("bayesian_uncertainty", {})
    countries = publication_country_names(configs)
    if countries_filter:
        if len(set(countries_filter)) != len(countries_filter):
            raise ValueError("Country filter contains duplicate entries")
        requested = set(countries_filter)
        outside_publication_scope = sorted(requested.difference(countries))
        if outside_publication_scope:
            raise ValueError(
                "Legacy paired-posterior research is restricted to the configured "
                "country set: " + ", ".join(outside_publication_scope)
            )
        countries = [country for country in countries if country in requested]

    figure_settings = settings.get("figure2c_joint_credible_interval", {})
    draws = int(
        draws_per_country
        or figure_settings.get("posterior_draws_per_country")
        or settings.get("posterior_predictive_draws_per_country", 100)
    )
    if draws < MINIMUM_CREDIBLE_INTERVAL_DRAWS:
        raise ValueError(
            "Legacy paired-posterior research requires at least "
            f"{MINIMUM_CREDIBLE_INTERVAL_DRAWS} paired posterior draws per country "
            "so each 2.5% credible-interval tail has at least 50 empirical draws; "
            f"received {draws}."
        )
    seed = int(settings.get("random_seed", 20260510)) + 917
    posterior_stem, metadata = _posterior_metadata(posterior_samples_path)
    samples = pd.read_parquet(Path(posterior_samples_path))
    if not allow_conditional_posterior:
        if posterior_stem is None:
            raise ValueError(
                "Cannot verify posterior quality because its metadata stem could not be inferred"
            )
        try:
            metadata = check_bayesian_quality(
                posterior_stem,
                require_recommended=True,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Posterior quality validation failed: {exc}") from exc
    required_variable = int(min_variable_parameters or len(JOINT_POSTERIOR_PARAMETERS))
    audit = _validate_joint_posterior_samples(
        samples,
        countries=countries,
        posterior_metadata=metadata,
        posterior_stem=posterior_stem,
        allow_conditional_posterior=allow_conditional_posterior,
        min_variable_parameters=required_variable,
    )
    write_dataframe(audit, POSTERIOR_AUDIT_PATH)
    selected = _select_posterior_samples(samples, countries=countries, draws_per_country=draws, seed=seed)
    structural_pairing = "structural_draw_id" in selected.columns
    inference_values = (
        selected["inference_structure"].dropna().astype(str).unique().tolist()
        if "inference_structure" in selected.columns
        else []
    )
    if len(inference_values) > 1:
        raise ValueError(f"Selected samples mix inference structures: {sorted(inference_values)}")
    inference_structure = (
        inference_values[0]
        if inference_values
        else str(metadata.get("inference_structure", "country_joint_feedback"))
    )
    selected = _attach_intervention_uncertainty(
        selected,
        configs,
        seed=seed + 101,
        enabled=intervention_uncertainty,
    )
    scenarios = _build_scenarios(configs, selected, strategies=FIGURE2C_STRATEGIES)
    summary = execute_scenario_summary_list(scenarios, stem=STEM, n_jobs=n_jobs)
    paired = _paired_draws(summary)
    intervals = _interval_summary(paired)
    process_columns = sorted(
        column
        for column in selected.columns
        if str(column).startswith("log_beta_process_")
    )
    interval_basis = (
        "paired_multi_country_joint_posterior_credible_interval"
        if inference_structure in FULL_JOINT_INFERENCE_STRUCTURES
        else "paired_conditional_state_space_exact_importance_process_and_structural_uncertainty"
        if inference_structure == "reference_structure_state_space_exact_importance_cut"
        else "paired_modular_or_joint_parameter_and_intervention_uncertainty"
    )

    for retired_path in RETIRED_INTERVAL_PATHS:
        for candidate in (retired_path, retired_path.with_suffix(".parquet")):
            candidate.unlink(missing_ok=True)
    write_dataframe(summary, SCENARIO_SUMMARY_PATH)
    write_dataframe(paired, DRAW_PATH)
    write_dataframe(intervals, INTERVAL_PATH)
    write_run_metadata(
        STEM,
        current_run_metadata(
            STEM,
            row_counts={
                "scenario_summary": int(len(summary)),
                "paired_draws": int(len(paired)),
                "interval_summary": int(len(intervals)),
                "uncertainty_parameter_audit": int(len(audit)),
            },
        )
        | {
            "analysis_role": ANALYSIS_ROLE,
            "publication_path": PUBLICATION_PATH,
            "figure2c_interval_source": FIGURE2C_INTERVAL_SOURCE,
            "posterior_samples_path": str(posterior_samples_path),
            "posterior_sample_stem": posterior_stem,
            "posterior_sample_metadata": {
                key: metadata.get(key)
                for key in (
                    "uncertainty_scope",
                    "sampler",
                    "fixed_parameters",
                    "fix_durations",
                    "n_chains",
                    "warmup",
                    "draws_per_chain",
                    "thin",
                    "parameterization",
                    "inference_structure",
                    "convergence_summary",
                )
                if key in metadata
            },
            "posterior_draws_per_country": draws,
            "uncertainty_draws_per_country": draws,
            "countries": countries,
            "strategies": list(FIGURE2C_STRATEGIES),
            "propagated_uncertainty_parameters": list(
                PROPAGATED_UNCERTAINTY_PARAMETERS
            ),
            "parameter_source_roles": metadata.get("parameter_source_roles", {}),
            "annual_process_draw_columns": process_columns,
            "annual_process_draw_years": [
                int(column.removeprefix("log_beta_process_"))
                for column in process_columns
            ],
            "inference_structure": inference_structure,
            "structural_draw_pairing": bool(structural_pairing),
            "selected_structural_draw_ids": (
                int(selected["structural_draw_id"].nunique()) if structural_pairing else 0
            ),
            "uncertainty_sample_selection": (
                "equal_mass_joint_smc_particles_balanced_by_island_without_replacement"
                if inference_structure
                == "cross_country_joint_state_space_tempered_smc"
                else "state_importance_resample_balanced_by_batch_and_structural_draw_without_replacement"
                if structural_pairing
                else "balanced_by_synthetic_batch_without_replacement"
            ),
            "min_variable_parameters_required": required_variable,
            "intervention_uncertainty": bool(intervention_uncertainty),
            "intervention_uncertainty_priors": INTERVENTION_UNCERTAINTY_DEFAULTS,
            "allow_conditional_posterior": bool(allow_conditional_posterior),
            "interval_basis": interval_basis,
        },
    )
    return summary, paired, intervals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run optional_nonpublication_legacy_research paired-posterior "
            "programme uncertainty; this is not a Figure 2c interval source."
        )
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--draws-per-country", type=int, default=None)
    parser.add_argument("--posterior-samples", type=str, default=str(DEFAULT_POSTERIOR_SAMPLE_PATH))
    parser.add_argument("--countries", type=str, default=None)
    parser.add_argument(
        "--disable-intervention-uncertainty",
        action="store_true",
        help="Use modal intervention assumptions instead of sampling intervention-effect priors.",
    )
    parser.add_argument(
        "--min-variable-parameters",
        type=int,
        default=None,
        help="Minimum number of propagated uncertainty parameters that must vary per country.",
    )
    args = parser.parse_args()
    main(
        n_jobs=args.n_jobs,
        draws_per_country=args.draws_per_country,
        posterior_samples_path=args.posterior_samples,
        countries_filter=args.countries.split(",") if args.countries else None,
        allow_conditional_posterior=False,
        intervention_uncertainty=not args.disable_intervention_uncertainty,
        min_variable_parameters=args.min_variable_parameters,
    )
