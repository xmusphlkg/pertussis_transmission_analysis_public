from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    apply_intervention_definition,
    current_run_metadata,
    execute_scenario_summary_list,
    file_sha256,
    load_configs,
    make_intervention_config,
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
    _apply_sample,
    _sample_columns,
)
from src_python.simulation.run_routine_timeliness_sensitivity import _apply_timeliness
from src_python.utils.io import project_path, write_dataframe
from src_python.validation.publication_gate import (
    require_predictive_publication_gate,
)


STEM = "figure2c_paired_programme_uncertainty"
FULL_CRI_POSTERIOR_STEM = "bayesian_uncertainty_figure2c_conditional"
RETIRED_POSTERIOR_STEMS = RETIRED_MISLABELED_OUTPUT_STEMS
DEFAULT_POSTERIOR_SAMPLE_PATH = project_path(
    "outputs", "simulations", f"{FULL_CRI_POSTERIOR_STEM}_posterior_samples.parquet"
)
SCENARIO_SUMMARY_PATH = project_path("outputs", "summaries", f"{STEM}_summary.csv")
DRAW_PATH = project_path("outputs", "tables", "figure2c_programme_paired_conditional_interval_draws.csv")
INTERVAL_PATH = project_path("outputs", "summaries", "figure2c_programme_paired_conditional_intervals.csv")
POSTERIOR_AUDIT_PATH = project_path("outputs", "metadata", f"{STEM}_uncertainty_parameter_audit.csv")
RETIRED_INTERVAL_PATHS = (
    project_path("outputs", "tables", "figure2c_programme_paired_credible_interval_draws.csv"),
    project_path("outputs", "summaries", "figure2c_programme_paired_credible_intervals.csv"),
    project_path("outputs", "metadata", f"{STEM}_posterior_parameter_audit.csv"),
)

PRIMARY_RATE = "annualized_child_adolescent_cases_per_100k"
PRIMARY_TOTAL = "total_child_adolescent_cases"
PRIMARY_REDUCTION = "relative_reduction_child_adolescent_cases"
PROPAGATED_UNCERTAINTY_PARAMETERS = (
    "beta_S",
    "reporting_multiplier",
    "VE_sus",
    "VE_inf",
    "VE_dur",
    "relative_infectiousness_asymptomatic",
    "infectious_duration_symptomatic",
    "infectious_duration_asymptomatic",
    "fitness_R",
)
# Backward-compatible import alias. These parameters no longer all represent a
# country posterior: only beta/reporting are state-target coordinates; the
# remaining dimensions are a shared external-prior sensitivity design.
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

INTERVENTION_UNCERTAINTY_DEFAULTS = {
    "adolescent_coverage_floor": {"low": 0.75, "mode": 0.90, "high": 0.98},
    "maternal_coverage_floor": {"low": 0.55, "mode": 0.75, "high": 0.90},
    "young_adult_coverage_floor": {"low": 0.35, "mode": 0.55, "high": 0.75},
    "contact_reduction_fraction": {"low": 0.05, "mode": 0.15, "high": 0.30},
    "targeted_pep_coverage": {"low": 0.05, "mode": 0.45, "high": 0.60},
    "maternal_protection_duration_days": {"low": 90.0, "mode": 180.0, "high": 270.0},
}
INTERVENTION_UNCERTAINTY_PREFIX = "intervention_uncertainty_"

FIGURE2C_STRATEGIES = (
    "current",
    "timeliness_only",
    "adolescent_booster",
    "pregnancy_tdap_scaleup",
    "cocooning_adjunct",
    "maternal_immunization",
    "targeted_pep_high_risk",
)

STRATEGY_LABELS = {
    "current": "Current practice",
    "timeliness_only": "Routine timeliness",
    "adolescent_booster": "Adolescent booster",
    "pregnancy_tdap_scaleup": "Pregnancy Tdap",
    "cocooning_adjunct": "Close-contact adjunct",
    "maternal_immunization": "Infant exposure",
    "targeted_pep_high_risk": "Targeted PEP",
}


def _posterior_stem_from_sample_path(path: str | Path) -> str | None:
    name = Path(path).name
    if name == "bayesian_posterior_samples.parquet":
        return "bayesian_uncertainty"
    suffix = "_posterior_samples.parquet"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return None


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
    conditional beta/reporting draw. Figure 2c must therefore subsample those
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
            "Audited Figure 2c uncertainty requires posterior run metadata. "
            f"No metadata found for posterior sample stem {posterior_stem!r}."
        )
    sampler = str(posterior_metadata.get("sampler", "")).lower()
    scope = str(posterior_metadata.get("uncertainty_scope", "")).lower()
    if sampler == "state_space_exact_importance_cut":
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
            "Refusing to compute Figure 2c paired uncertainty from one-parameter "
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
            "Refusing to compute Figure 2c paired uncertainty from rows labelled "
            "sampling_method=beta_grid."
        )
    fixed_parameters = sorted(str(parameter) for parameter in posterior_metadata.get("fixed_parameters", []) or [])
    if fixed_parameters or bool(posterior_metadata.get("fix_durations", False)):
        raise ValueError(
            "Figure 2c paired uncertainty requires all registered dimensions to vary. "
            f"fixed_parameters={fixed_parameters}, fix_durations={posterior_metadata.get('fix_durations')}."
        )

    quality_failures = posterior_quality_failures(
        posterior_metadata,
        require_recommended=True,
    )
    if quality_failures:
        raise ValueError(
            "Figure 2c paired uncertainty requires samples that pass the recommended "
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


def _triangular_parameters(settings: dict[str, Any], name: str) -> dict[str, float]:
    configured = settings.get("intervention_priors", {}).get(name, {})
    values = {**INTERVENTION_UNCERTAINTY_DEFAULTS[name], **configured}
    low = float(values["low"])
    mode = float(values["mode"])
    high = float(values["high"])
    if not low <= mode <= high:
        raise ValueError(f"Invalid triangular prior for {name}: low <= mode <= high is required")
    return {"low": low, "mode": mode, "high": high}


def _beta_ab(mean: float, sd: float) -> tuple[float, float]:
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    sd = max(float(sd), 1e-6)
    variance = min(sd**2, mean * (1.0 - mean) * 0.95)
    common = mean * (1.0 - mean) / variance - 1.0
    return max(mean * common, 1e-3), max((1.0 - mean) * common, 1e-3)


def _draw_beta_prior(rng: np.random.Generator, prior: dict[str, Any]) -> float:
    alpha, beta = _beta_ab(float(prior["mean"]), float(prior["sd"]))
    return float(rng.beta(alpha, beta))


def _intervention_uncertainty_draw(
    configs: dict[str, Any],
    rng: np.random.Generator,
    *,
    stochastic: bool,
) -> dict[str, float]:
    settings = configs["baseline"].get("bayesian_uncertainty", {}).get("figure2c_conditional_uncertainty", {})
    priors = configs["baseline"].get("bayesian_uncertainty", {}).get("priors", {})

    def triangular(name: str) -> float:
        params = _triangular_parameters(settings, name)
        if not stochastic:
            return float(params["mode"])
        return float(rng.triangular(params["low"], params["mode"], params["high"]))

    maternal_sus_prior = priors.get("maternal_VE_sus", {"mean": 0.55, "sd": 0.12})
    maternal_sym_prior = priors.get("maternal_VE_sym", {"mean": 0.92, "sd": 0.05})
    return {
        "adolescent_coverage_floor": triangular("adolescent_coverage_floor"),
        "maternal_coverage_floor": triangular("maternal_coverage_floor"),
        "young_adult_coverage_floor": triangular("young_adult_coverage_floor"),
        "contact_reduction_fraction": triangular("contact_reduction_fraction"),
        "targeted_pep_coverage": triangular("targeted_pep_coverage"),
        "maternal_protection_duration_days": triangular("maternal_protection_duration_days"),
        "maternal_VE_sus": (
            _draw_beta_prior(rng, maternal_sus_prior)
            if stochastic
            else float(maternal_sus_prior.get("mean", 0.55))
        ),
        "maternal_VE_sym": (
            _draw_beta_prior(rng, maternal_sym_prior)
            if stochastic
            else float(maternal_sym_prior.get("mean", 0.92))
        ),
    }


def _attach_intervention_uncertainty(
    selected_samples: pd.DataFrame,
    configs: dict[str, Any],
    *,
    seed: int,
    enabled: bool,
) -> pd.DataFrame:
    out = selected_samples.copy()
    rng = np.random.default_rng(seed)
    draws = [
        _intervention_uncertainty_draw(configs, rng, stochastic=enabled)
        for _ in range(len(out))
    ]
    for key in draws[0].keys() if draws else ():
        out[f"{INTERVENTION_UNCERTAINTY_PREFIX}{key}"] = [float(draw[key]) for draw in draws]
    return out


def _apply_intervention_uncertainty(intervention: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(intervention)
    coverage = out.setdefault("coverage_min_updates", {}) if "coverage_min_updates" in out else None
    if coverage is not None:
        if "adolescent_10_17y" in coverage:
            coverage["adolescent_10_17y"] = float(
                row[f"{INTERVENTION_UNCERTAINTY_PREFIX}adolescent_coverage_floor"]
            )
        if "infant_0_2m" in coverage:
            coverage["infant_0_2m"] = float(
                row[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_coverage_floor"]
            )
        if "young_adult_18_39y" in coverage:
            coverage["young_adult_18_39y"] = float(
                row[f"{INTERVENTION_UNCERTAINTY_PREFIX}young_adult_coverage_floor"]
            )

    vaccine = out.setdefault("vaccine_overrides", {}) if "vaccine_overrides" in out else None
    if vaccine is not None:
        if "maternal_VE_sus" in vaccine:
            vaccine["maternal_VE_sus"] = float(row[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sus"])
        if "maternal_VE_sym" in vaccine:
            vaccine["maternal_VE_sym"] = float(row[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sym"])

    natural_history = out.setdefault("natural_history_overrides", {}) if "natural_history_overrides" in out else None
    if natural_history is not None and "maternal_protection_duration" in natural_history:
        natural_history["maternal_protection_duration"] = float(
            row[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_protection_duration_days"]
        )

    pep = out.setdefault("pep_updates", {}) if "pep_updates" in out else None
    if pep is not None and "coverage_household_contacts" in pep:
        pep["coverage_household_contacts"] = float(row[f"{INTERVENTION_UNCERTAINTY_PREFIX}targeted_pep_coverage"])

    contact = out.setdefault("contact_matrix_reduction", {}) if "contact_matrix_reduction" in out else None
    if contact is not None and "reduction_fraction" in contact:
        contact["reduction_fraction"] = float(row[f"{INTERVENTION_UNCERTAINTY_PREFIX}contact_reduction_fraction"])
    return out


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
            "Not enough aligned conditional-state/external-design positions for Figure 2c: "
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


def _posterior_sample_from_row(row: pd.Series) -> dict[str, float]:
    missing = [name for name in _sample_columns() if name != "reporting_trend_end_multiplier" and name not in row.index]
    if missing:
        raise KeyError(f"Posterior sample row is missing required columns: {', '.join(missing)}")
    sample: dict[str, float] = {}
    for name in _sample_columns():
        if name in row.index and pd.notna(row[name]):
            sample[name] = float(row[name])
        elif name == "reporting_trend_end_multiplier":
            sample[name] = 1.0
        else:
            raise ValueError(f"Posterior sample value is missing for required column: {name}")
    sample.update(
        {
            str(name): float(value)
            for name, value in row.items()
            if str(name).startswith("log_beta_process_") and pd.notna(value)
        }
    )
    return sample


def _strategy_config_from_sampled_current(
    configs: dict[str, Any],
    *,
    sampled_current: dict[str, Any],
    vaccine_name: str,
    strategy: str,
    sample_row: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if strategy == "current":
        return deepcopy(sampled_current), vaccine_name
    if strategy == "timeliness_only":
        return _apply_timeliness(sampled_current), vaccine_name

    intervention = configs["interventions"][strategy]
    if intervention.get("vaccine_scenario") not in {None, vaccine_name}:
        raise ValueError(
            "Figure 2c paired uncertainty currently expects program strategies "
            f"without vaccine_scenario swaps; got {strategy!r}."
        )
    intervention = _apply_intervention_uncertainty(intervention, sample_row)
    return apply_intervention_definition(sampled_current, intervention), vaccine_name


def _build_scenarios(
    configs: dict[str, Any],
    selected_samples: pd.DataFrame,
    *,
    strategies: tuple[str, ...],
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    base_configs: dict[str, tuple[dict[str, Any], str]] = {
        country: make_intervention_config("current", country_profile=country)
        for country in sorted(selected_samples["country"].astype(str).unique())
    }
    for row in selected_samples.to_dict(orient="records"):
        country = str(row["country"])
        posterior_draw = int(row["posterior_draw"])
        sample = _posterior_sample_from_row(pd.Series(row))
        base_config, current_vaccine_name = base_configs[country]
        sampled_current = _apply_sample(base_config, sample)
        posterior_metadata = {
            f"posterior_{parameter}": float(sample[parameter])
            for parameter in JOINT_POSTERIOR_PARAMETERS
            if parameter in sample
        }
        posterior_metadata.update(
            {
                f"posterior_{parameter}": float(value)
                for parameter, value in sample.items()
                if str(parameter).startswith("log_beta_process_")
            }
        )
        intervention_metadata = {
            key: float(value)
            for key, value in row.items()
            if str(key).startswith(INTERVENTION_UNCERTAINTY_PREFIX)
            and pd.notna(value)
        }
        for strategy in strategies:
            config, vaccine_name = _strategy_config_from_sampled_current(
                configs,
                sampled_current=sampled_current,
                vaccine_name=current_vaccine_name,
                strategy=strategy,
                sample_row=row,
            )
            metadata = {
                "country": country,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "posterior_draw": posterior_draw,
                "posterior_chain": int(row["chain"]) if "chain" in row and pd.notna(row["chain"]) else -1,
                "posterior_source_draw": int(row["draw"]) if "draw" in row and pd.notna(row["draw"]) else -1,
                "posterior_log_prob": (
                    float(row["posterior_log_prob"])
                    if "posterior_log_prob" in row and pd.notna(row["posterior_log_prob"])
                    else np.nan
                ),
            } | posterior_metadata | intervention_metadata
            if "structural_draw_id" in row and pd.notna(row["structural_draw_id"]):
                metadata["structural_draw_id"] = int(row["structural_draw_id"])
            if "inference_structure" in row and pd.notna(row["inference_structure"]):
                metadata["inference_structure"] = str(row["inference_structure"])
            scenarios.append(
                {
                    "config": config,
                    "analysis": STEM,
                    "scenario": f"{country}_draw_{posterior_draw:03d}_{strategy}",
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": strategy,
                    "metadata": metadata,
                }
            )
    return scenarios


def _paired_draws(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    data["posterior_draw"] = pd.to_numeric(data["posterior_draw"], errors="raise").astype(int)
    data["strategy"] = data["strategy"].astype(str)

    pair_keys = ["country", "posterior_draw"]
    if "structural_draw_id" in data.columns:
        structural_ids = pd.to_numeric(data["structural_draw_id"], errors="coerce")
        if structural_ids.isna().any() or not np.equal(structural_ids, np.floor(structural_ids)).all():
            raise ValueError("Scenario summaries contain invalid structural_draw_id values")
        data["structural_draw_id"] = structural_ids.astype(np.int64)
        draw_alignment = data.groupby("posterior_draw")["structural_draw_id"].nunique()
        if not draw_alignment.eq(1).all():
            bad_draws = draw_alignment.index[~draw_alignment.eq(1)].astype(int).tolist()[:10]
            raise ValueError(
                "Scenario summaries lost cross-country structural pairing for "
                f"posterior draws: {bad_draws}"
            )
        pair_keys.append("structural_draw_id")

    current = data.loc[
        data["strategy"].eq("current"),
        [*pair_keys, PRIMARY_RATE, PRIMARY_TOTAL],
    ].rename(
        columns={
            PRIMARY_RATE: "current_rate",
            PRIMARY_TOTAL: "current_total_cases",
        }
    )
    intervention = data.loc[~data["strategy"].eq("current")].copy()
    paired = intervention.merge(current, on=pair_keys, how="left", validate="many_to_one")
    if paired["current_rate"].isna().any():
        missing = paired.loc[paired["current_rate"].isna(), ["country", "posterior_draw"]].drop_duplicates()
        raise ValueError(f"Missing paired current rows for {len(missing)} country-draw combination(s)")

    paired["intervention_rate"] = pd.to_numeric(paired[PRIMARY_RATE], errors="coerce")
    paired["intervention_total_cases"] = pd.to_numeric(paired[PRIMARY_TOTAL], errors="coerce")
    paired[PRIMARY_REDUCTION] = 1.0 - paired["intervention_rate"] / paired["current_rate"].replace(0, np.nan)
    keep = [
        "country",
        "posterior_draw",
        *(["structural_draw_id"] if "structural_draw_id" in paired.columns else []),
        "posterior_chain",
        "posterior_source_draw",
        "posterior_log_prob",
        "strategy",
        "strategy_label",
        "current_rate",
        "intervention_rate",
        "current_total_cases",
        "intervention_total_cases",
        PRIMARY_REDUCTION,
    ]
    if "inference_structure" in paired.columns:
        keep.append("inference_structure")
    keep.extend(
        sorted(
            column
            for column in paired.columns
            if column.startswith("posterior_")
            or column.startswith(INTERVENTION_UNCERTAINTY_PREFIX)
        )
    )
    keep = list(dict.fromkeys(keep))
    return paired.loc[:, keep].sort_values(["country", "strategy", "posterior_draw"]).reset_index(drop=True)


def _q(values: pd.Series, quantile: float) -> float:
    return float(pd.to_numeric(values, errors="coerce").quantile(quantile))


def _interval_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (country, strategy), group in draws.groupby(["country", "strategy"], sort=True):
        first = group.iloc[0]
        inference_structure = str(first.get("inference_structure", ""))
        if inference_structure == "reference_structure_state_space_exact_importance_cut":
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
            "--allow-conditional-posterior is retired for the canonical Figure 2c "
            "paths because it bypassed quality checks. Use a separate diagnostic "
            "runner/output stem for exploratory inputs."
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
                "Figure 2c excludes countries outside the prespecified publication "
                "set: " + ", ".join(outside_publication_scope)
            )
        countries = [country for country in countries if country in requested]

    require_predictive_publication_gate(expected_countries=countries)

    figure_settings = settings.get("figure2c_conditional_uncertainty", {})
    draws = int(
        draws_per_country
        or figure_settings.get("posterior_draws_per_country")
        or settings.get("posterior_predictive_draws_per_country", 100)
    )
    seed = int(settings.get("random_seed", 20260510)) + 917
    posterior_stem, metadata = _posterior_metadata(posterior_samples_path)
    samples = pd.read_parquet(Path(posterior_samples_path))
    if not allow_conditional_posterior:
        if posterior_stem is None:
            raise ValueError(
                "Cannot verify posterior freshness because its metadata stem could not be inferred"
            )
        try:
            metadata = check_bayesian_quality(
                posterior_stem,
                require_recommended=True,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Posterior freshness/quality validation failed: {exc}") from exc
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
        "paired_conditional_state_space_exact_importance_process_and_structural_uncertainty"
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
            "posterior_samples_path": str(posterior_samples_path),
            "posterior_samples_sha256": file_sha256(posterior_samples_path),
            "paired_draws_sha256": file_sha256(DRAW_PATH),
            "paired_intervals_sha256": file_sha256(INTERVAL_PATH),
            "scenario_summary_sha256": file_sha256(SCENARIO_SUMMARY_PATH),
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
                "state_importance_resample_balanced_by_batch_and_structural_draw_without_replacement"
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
        description="Run paired conditional uncertainty intervals for Figure 2c programme reductions."
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
