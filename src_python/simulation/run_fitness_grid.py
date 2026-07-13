from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    current_run_metadata,
    execute_scenario_summary_list,
    load_configs,
    make_config,
    publication_country_names,
    run_scenario_list,
    validate_run_metadata,
    write_run_metadata,
)
from src_python.simulation.check_bayesian_quality import check_bayesian_quality
from src_python.simulation.run_bayesian_uncertainty import _apply_sample, _sample_columns
from src_python.simulation.run_figure2c_paired_uncertainty import (
    _posterior_stem_from_sample_path,
)
from src_python.simulation.run_joint_psa_rank_acceptability import (
    EXPECTED_PARAMETER_NAMES as JOINT_PSA_PARAMETER_NAMES,
    STEM as JOINT_PSA_STEM,
    _apply_infant_contact_multiplier,
    _clip_probability,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.validation.publication_gate import require_predictive_publication_gate


POSTERIOR_BENEFIT_STEM = "fitness_resistance_grid_posterior_benefit"
POSTERIOR_DIAGNOSTIC_STEM = "fitness_resistance_grid_posterior_sample_diagnostics"
PSA_BENEFIT_STEM = "fitness_resistance_grid_psa_benefit"
DEFAULT_PSA_SAMPLE_PATH = project_path("outputs", "tables", "joint_psa_parameter_samples.csv")
DEFAULT_POSTERIOR_SAMPLE_PATH = project_path(
    "outputs",
    "simulations",
    "bayesian_uncertainty_figure2c_conditional_posterior_samples.parquet",
)
FITNESS_TARGETS = (
    ("Fitness cost (0.85)", 0.85),
    ("Neutral (1.00)", 1.00),
    ("Advantage (1.10)", 1.10),
)
UNCERTAINTY_SOURCE = "conditional_state_process_plus_external_prior_design"
UNCERTAINTY_SCOPE = (
    "Paired low/high VE_inf comparison propagating the selected conditional state/process "
    "and external-prior design file; "
    "grid_fitness_R and grid_VE_inf are fixed overrides for Figure 3D."
)
GRID_OVERRIDE_PARAMETERS = frozenset({"VE_inf", "fitness_R"})
PSA_APPLIED_COLUMNS = frozenset(
    {
        "infant_contact_multiplier",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_asymptomatic",
        "PEP_coverage_multiplier",
    }
)
PSA_REQUIRED_COLUMNS = frozenset(
    {"psa_sample_id", "sample_design", "uncertainty_schema_version", *JOINT_PSA_PARAMETER_NAMES}
)
PSA_IGNORED_GRID_COLUMNS = frozenset(
    set(JOINT_PSA_PARAMETER_NAMES).difference(PSA_APPLIED_COLUMNS)
)
PSA_UNCERTAINTY_SOURCE = "targeted_latin_hypercube_psa"
PSA_UNCERTAINTY_SCOPE = (
    "Targeted Figure 3D probabilistic sensitivity analysis varying infant-contact, "
    "asymptomatic infectiousness/duration, and PEP nuisance parameters while holding "
    "grid_fitness_R and grid_VE_inf fixed for paired low/high VE_inf comparisons; "
    "observation-only reporting uncertainty is excluded from the true-case endpoint."
)


def _grid_values(settings: dict[str, Any], key: str, default: list[float]) -> list[float]:
    values = settings.get(key, default)
    return [float(value) for value in values]


def _nearest_grid_value(values: list[float] | tuple[float, ...], target: float) -> float:
    if not values:
        raise ValueError("Cannot choose a nearest grid value from an empty list.")
    return float(min(values, key=lambda value: abs(float(value) - float(target))))


def _fitness_targets(fitness_values: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "fitness_group": label,
            "target_fitness_R": float(target),
            "grid_fitness_R": _nearest_grid_value(fitness_values, target),
        }
        for label, target in FITNESS_TARGETS
    ]


def _validate_posterior_samples(samples: pd.DataFrame, countries: list[str]) -> None:
    required = {"country", "chain", "draw", "posterior_log_prob", *_sample_columns()}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Posterior sample file is missing required column(s): {', '.join(missing)}")
    missing_countries = sorted(set(countries) - set(samples["country"].astype(str)))
    if missing_countries:
        raise ValueError(
            "Posterior sample file is missing country draw(s) for: "
            + ", ".join(missing_countries)
        )


def _select_posterior_draws(
    samples: pd.DataFrame,
    *,
    countries: list[str],
    draws_per_country: int,
    seed: int,
) -> pd.DataFrame:
    if draws_per_country < 1:
        return pd.DataFrame(columns=list(samples.columns) + ["posterior_draw"])

    _validate_posterior_samples(samples, countries)
    rng = np.random.default_rng(int(seed))
    selected_frames: list[pd.DataFrame] = []
    for country in countries:
        group = samples.loc[samples["country"].astype(str).eq(str(country))].reset_index(drop=True)
        if len(group) > draws_per_country:
            positions = np.sort(rng.choice(len(group), size=draws_per_country, replace=False))
            selected = group.iloc[positions].copy()
        else:
            selected = group.copy()
        selected["posterior_draw"] = np.arange(1, len(selected) + 1, dtype=int)
        selected_frames.append(selected)
    return pd.concat(selected_frames, ignore_index=True)


def _posterior_sample_from_row(row: pd.Series) -> dict[str, float]:
    return {name: float(row[name]) for name in _sample_columns()}


def _summarise_selected_posterior_samples(selected_draws: pd.DataFrame) -> pd.DataFrame:
    """Describe which selected posterior parameters vary within each country."""
    if selected_draws.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for country, country_draws in selected_draws.groupby("country", sort=True):
        for parameter in _sample_columns():
            values = pd.to_numeric(country_draws[parameter], errors="coerce").dropna()
            n_values = int(values.size)
            if n_values == 0:
                rows.append(
                    {
                        "country": country,
                        "parameter": parameter,
                        "posterior_draws": 0,
                        "unique_values": 0,
                        "mean": np.nan,
                        "sd": np.nan,
                        "coefficient_of_variation": np.nan,
                        "q025": np.nan,
                        "median": np.nan,
                        "q975": np.nan,
                        "varies_within_country": False,
                        "grid_override_in_fig3d": parameter in GRID_OVERRIDE_PARAMETERS,
                        "uncertainty_source": UNCERTAINTY_SOURCE,
                    }
                )
                continue

            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n_values > 1 else 0.0
            cv = float(sd / abs(mean)) if mean != 0 else np.nan
            q025, median, q975 = np.percentile(values.to_numpy(dtype=float), [2.5, 50.0, 97.5])
            unique_values = int(values.nunique(dropna=True))
            rows.append(
                {
                    "country": country,
                    "parameter": parameter,
                    "posterior_draws": n_values,
                    "unique_values": unique_values,
                    "mean": mean,
                    "sd": sd,
                    "coefficient_of_variation": cv,
                    "q025": float(q025),
                    "median": float(median),
                    "q975": float(q975),
                    "varies_within_country": bool(unique_values > 1 and sd > 0.0),
                    "grid_override_in_fig3d": parameter in GRID_OVERRIDE_PARAMETERS,
                    "uncertainty_source": UNCERTAINTY_SOURCE,
                }
            )
    return pd.DataFrame(rows)


def _apply_grid_overrides(config: dict[str, Any], *, fitness_r: float, ve_inf: float) -> dict[str, Any]:
    config["transmission"]["fitness_R"] = float(fitness_r)
    config["vaccine"]["VE_inf"] = float(ve_inf)
    return config


def _paired_endpoint_benefits(endpoint_summary: pd.DataFrame, *, draw_column: str) -> pd.DataFrame:
    required = {
        "country",
        draw_column,
        "fitness_group",
        "grid_fitness_R",
        "low_grid_VE_inf",
        "high_grid_VE_inf",
        "ve_endpoint",
        "annualized_infant_cases_per_100k",
    }
    missing = sorted(required - set(endpoint_summary.columns))
    if missing:
        raise ValueError(f"Endpoint summary is missing column(s): {', '.join(missing)}")

    endpoint_summary = endpoint_summary.copy()
    endpoint_summary["annualized_infant_cases_per_100k"] = pd.to_numeric(
        endpoint_summary["annualized_infant_cases_per_100k"],
        errors="coerce",
    )
    paired = endpoint_summary.pivot_table(
        index=[
            "country",
            draw_column,
            "fitness_group",
            "grid_fitness_R",
            "low_grid_VE_inf",
            "high_grid_VE_inf",
        ],
        columns="ve_endpoint",
        values="annualized_infant_cases_per_100k",
        aggfunc="first",
    ).reset_index()
    paired.columns.name = None
    if {"low", "high"} - set(paired.columns):
        raise ValueError("Endpoint summary must contain both low and high VE_inf rows.")

    paired = paired.dropna(subset=["low", "high"]).copy()
    paired = paired.loc[pd.to_numeric(paired["low"], errors="coerce").gt(0)].copy()
    paired = paired.rename(
        columns={
            "low": "low_infant_cases_per_100k",
            "high": "high_infant_cases_per_100k",
        }
    )
    paired["relative_benefit"] = (
        1.0
        - paired["high_infant_cases_per_100k"].astype(float)
        / paired["low_infant_cases_per_100k"].astype(float)
    )
    return paired.reset_index(drop=True)


def _summarise_paired_benefits(
    paired: pd.DataFrame,
    *,
    draw_column: str,
    draw_count_column: str,
    uncertainty_source: str,
    uncertainty_scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "country",
        "fitness_group",
        "grid_fitness_R",
        "low_grid_VE_inf",
        "high_grid_VE_inf",
    ]
    for key, group in paired.groupby(group_cols, sort=True, dropna=False):
        values = group["relative_benefit"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        q025, median, q975 = np.percentile(values, [2.5, 50.0, 97.5])
        country, fitness_group, grid_fitness, low_ve, high_ve = key
        rows.append(
            {
                "country": country,
                "fitness_group": fitness_group,
                "grid_fitness_R": float(grid_fitness),
                "low_grid_VE_inf": float(low_ve),
                "high_grid_VE_inf": float(high_ve),
                "median_relative_benefit": float(median),
                "q025_relative_benefit": float(q025),
                "q975_relative_benefit": float(q975),
                draw_count_column: int(group[draw_column].nunique()),
                "uncertainty_source": uncertainty_source,
                "uncertainty_scope": uncertainty_scope,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {label: idx for idx, (label, _) in enumerate(FITNESS_TARGETS)}
    out["_fitness_order"] = out["fitness_group"].map(order)
    out = out.sort_values(["country", "_fitness_order"]).drop(columns="_fitness_order")
    return out.reset_index(drop=True)


def _build_posterior_benefit_scenarios(
    selected_draws: pd.DataFrame,
    *,
    resistance_name: str,
    fitness_targets: list[dict[str, Any]],
    low_ve_inf: float,
    high_ve_inf: float,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for row in selected_draws.to_dict(orient="records"):
        country = str(row["country"])
        posterior_draw = int(row["posterior_draw"])
        sample = _posterior_sample_from_row(pd.Series(row))
        for target in fitness_targets:
            grid_fitness = float(target["grid_fitness_R"])
            for endpoint, ve_inf in (("low", low_ve_inf), ("high", high_ve_inf)):
                config = make_config(
                    vaccine_scenario="symptom_protective",
                    resistance_scenario=resistance_name,
                    country_profile=country,
                )
                config = _apply_sample(config, sample)
                config = _apply_grid_overrides(
                    config,
                    fitness_r=grid_fitness,
                    ve_inf=float(ve_inf),
                )
                scenarios.append(
                    {
                        "config": config,
                        "analysis": "fitness_resistance_grid_posterior_benefit",
                        "scenario": (
                            f"{country}_draw_{posterior_draw:03d}_"
                            f"fitness_{grid_fitness:.2f}_VEinf_{float(ve_inf):.2f}"
                        ),
                        "vaccine_scenario": "symptom_protective",
                        "resistance_scenario": resistance_name,
                        "metadata": {
                            "country": country,
                            "posterior_draw": posterior_draw,
                            "posterior_chain": int(row["chain"]),
                            "posterior_source_draw": int(row["draw"]),
                            "posterior_log_prob": float(row["posterior_log_prob"]),
                            "fitness_group": str(target["fitness_group"]),
                            "target_fitness_R": float(target["target_fitness_R"]),
                            "grid_fitness_R": grid_fitness,
                            "grid_VE_inf": float(ve_inf),
                            "ve_endpoint": endpoint,
                            "low_grid_VE_inf": float(low_ve_inf),
                            "high_grid_VE_inf": float(high_ve_inf),
                            "uncertainty_source": UNCERTAINTY_SOURCE,
                            "uncertainty_scope": UNCERTAINTY_SCOPE,
                        },
                    }
                )
    return scenarios


def _summarise_posterior_benefits(endpoint_summary: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_endpoint_benefits(endpoint_summary, draw_column="posterior_draw")
    return _summarise_paired_benefits(
        paired,
        draw_column="posterior_draw",
        draw_count_column="posterior_draws",
        uncertainty_source=UNCERTAINTY_SOURCE,
        uncertainty_scope=UNCERTAINTY_SCOPE,
    )


def _load_psa_samples(path: str | Path, *, sample_limit: int | None = None) -> pd.DataFrame:
    sample_path = Path(path)
    if not sample_path.is_absolute():
        sample_path = project_path(sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"PSA sample file not found: {sample_path}")
    if sample_path.resolve() == Path(DEFAULT_PSA_SAMPLE_PATH).resolve():
        validate_run_metadata(JOINT_PSA_STEM)

    samples = pd.read_parquet(sample_path) if sample_path.suffix == ".parquet" else pd.read_csv(sample_path)
    missing = sorted(PSA_REQUIRED_COLUMNS - set(samples.columns))
    if missing:
        raise ValueError(f"PSA sample file is missing required column(s): {', '.join(missing)}")
    samples = samples.copy()
    samples["psa_sample_id"] = pd.to_numeric(samples["psa_sample_id"], errors="raise").astype(int)
    samples = samples.sort_values("psa_sample_id").reset_index(drop=True)
    if sample_limit is not None:
        if sample_limit < 1:
            raise ValueError("sample_limit must be >= 1 when provided.")
        samples = samples.head(int(sample_limit)).copy()
    return samples


def _apply_fig3d_psa_nuisance_sample(
    config: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(config)
    _apply_infant_contact_multiplier(out, float(sample["infant_contact_multiplier"]))
    out["transmission"]["relative_infectiousness_asymptomatic"] = float(
        sample["relative_infectiousness_asymptomatic"]
    )
    out["natural_history"]["infectious_duration_asymptomatic"] = float(
        sample["infectious_duration_asymptomatic"]
    )
    pep = out.setdefault("PEP", {})
    pep["coverage_household_contacts"] = _clip_probability(
        float(pep.get("coverage_household_contacts", 0.0)) * float(sample["PEP_coverage_multiplier"])
    )
    out.setdefault("metadata", {})["fig3d_psa_sample_id"] = int(sample["psa_sample_id"])
    return out


def _optional_float(sample: dict[str, Any], key: str) -> float | None:
    if key not in sample or pd.isna(sample[key]):
        return None
    return float(sample[key])


def _build_psa_benefit_scenarios(
    samples: pd.DataFrame,
    *,
    countries: list[str],
    resistance_name: str,
    fitness_targets: list[dict[str, Any]],
    low_ve_inf: float,
    high_ve_inf: float,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for row in samples.to_dict(orient="records"):
        psa_sample_id = int(row["psa_sample_id"])
        for country in countries:
            for target in fitness_targets:
                grid_fitness = float(target["grid_fitness_R"])
                for endpoint, ve_inf in (("low", low_ve_inf), ("high", high_ve_inf)):
                    config = make_config(
                        vaccine_scenario="symptom_protective",
                        resistance_scenario=resistance_name,
                        country_profile=country,
                    )
                    config = _apply_fig3d_psa_nuisance_sample(config, row)
                    config = _apply_grid_overrides(
                        config,
                        fitness_r=grid_fitness,
                        ve_inf=float(ve_inf),
                    )
                    scenarios.append(
                        {
                            "config": config,
                            "analysis": "fitness_resistance_grid_psa_benefit",
                            "scenario": (
                                f"sample_{psa_sample_id:03d}_{country}_"
                                f"fitness_{grid_fitness:.2f}_VEinf_{float(ve_inf):.2f}"
                            ),
                            "vaccine_scenario": "symptom_protective",
                            "resistance_scenario": resistance_name,
                            "metadata": {
                                "country": country,
                                "psa_sample_id": psa_sample_id,
                                "fitness_group": str(target["fitness_group"]),
                                "target_fitness_R": float(target["target_fitness_R"]),
                                "grid_fitness_R": grid_fitness,
                                "grid_VE_inf": float(ve_inf),
                                "ve_endpoint": endpoint,
                                "low_grid_VE_inf": float(low_ve_inf),
                                "high_grid_VE_inf": float(high_ve_inf),
                                "infant_contact_multiplier": float(row["infant_contact_multiplier"]),
                                "relative_infectiousness_asymptomatic": float(
                                    row["relative_infectiousness_asymptomatic"]
                                ),
                                "infectious_duration_asymptomatic": float(
                                    row["infectious_duration_asymptomatic"]
                                ),
                                "PEP_coverage_multiplier": float(row["PEP_coverage_multiplier"]),
                                "psa_source_VE_inf_baseline": _optional_float(row, "VE_inf_baseline"),
                                "psa_source_fitness_R": _optional_float(row, "fitness_R"),
                                "grid_override_VE_inf": True,
                                "grid_override_fitness_R": True,
                                "uncertainty_source": PSA_UNCERTAINTY_SOURCE,
                                "uncertainty_scope": PSA_UNCERTAINTY_SCOPE,
                            },
                        }
                    )
    return scenarios


def _summarise_psa_benefits(endpoint_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = _paired_endpoint_benefits(endpoint_summary, draw_column="psa_sample_id")
    paired["uncertainty_source"] = PSA_UNCERTAINTY_SOURCE
    paired["uncertainty_scope"] = PSA_UNCERTAINTY_SCOPE
    summary = _summarise_paired_benefits(
        paired,
        draw_column="psa_sample_id",
        draw_count_column="psa_samples",
        uncertainty_source=PSA_UNCERTAINTY_SOURCE,
        uncertainty_scope=PSA_UNCERTAINTY_SCOPE,
    )
    return paired, summary


def _run_posterior_benefit_intervals(
    *,
    configs: dict[str, Any],
    resistance_name: str,
    fitness_values: list[float],
    ve_inf_values: list[float],
    posterior_draws: int,
    posterior_sample_path: str | Path,
    posterior_seed: int,
    posterior_batch_size: int,
    n_jobs: int | None,
) -> pd.DataFrame:
    if posterior_draws < 1:
        return pd.DataFrame()

    sample_path = Path(posterior_sample_path)
    if not sample_path.is_absolute():
        sample_path = project_path(sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Posterior sample file not found: {sample_path}")
    posterior_stem = _posterior_stem_from_sample_path(sample_path)
    if posterior_stem is None:
        raise ValueError(
            "Posterior benefit input must use the canonical "
            "<stem>_posterior_samples file contract"
        )
    posterior_metadata = check_bayesian_quality(
        posterior_stem,
        require_recommended=True,
    )
    expected_sample_path = project_path(
        "outputs",
        "simulations",
        f"{posterior_stem}_posterior_samples{sample_path.suffix}",
    )
    if sample_path.resolve() != expected_sample_path.resolve():
        raise ValueError(
            "Posterior metadata stem does not resolve to the supplied sample path: "
            f"stem={posterior_stem}, path={sample_path}"
        )

    samples = pd.read_parquet(sample_path) if sample_path.suffix == ".parquet" else pd.read_csv(sample_path)
    countries = publication_country_names(configs)
    require_predictive_publication_gate(expected_countries=countries)
    selected = _select_posterior_draws(
        samples,
        countries=countries,
        draws_per_country=int(posterior_draws),
        seed=int(posterior_seed),
    )
    sample_diagnostics = _summarise_selected_posterior_samples(selected)
    write_dataframe(
        sample_diagnostics,
        project_path("outputs", "summaries", f"{POSTERIOR_DIAGNOSTIC_STEM}.csv"),
    )
    diagnostic_metadata = current_run_metadata(
        POSTERIOR_DIAGNOSTIC_STEM,
        row_counts={
            "posterior_sample_diagnostics": int(len(sample_diagnostics)),
        },
    )
    diagnostic_metadata.update(
        {
            "posterior_sample_path": str(sample_path),
            "posterior_sample_stem": posterior_stem,
            "posterior_sampler": str(posterior_metadata.get("sampler", "")),
            "posterior_draws_per_country_requested": int(posterior_draws),
            "posterior_draws_selected": int(len(selected)),
            "posterior_seed": int(posterior_seed),
            "uncertainty_source": UNCERTAINTY_SOURCE,
            "uncertainty_scope": (
                "Parameter variation audit for the posterior samples selected for "
                "Figure 3D paired low/high VE_inf benefit intervals."
            ),
        }
    )
    write_run_metadata(POSTERIOR_DIAGNOSTIC_STEM, diagnostic_metadata)
    targets = _fitness_targets(fitness_values)
    low_ve_inf = float(min(ve_inf_values))
    high_ve_inf = float(max(ve_inf_values))

    summary_frames: list[pd.DataFrame] = []
    batch_size = max(1, int(posterior_batch_size))
    for batch_start in range(0, len(selected), batch_size):
        batch = selected.iloc[batch_start : batch_start + batch_size].copy()
        scenarios = _build_posterior_benefit_scenarios(
            batch,
            resistance_name=resistance_name,
            fitness_targets=targets,
            low_ve_inf=low_ve_inf,
            high_ve_inf=high_ve_inf,
        )
        summary = execute_scenario_summary_list(
            scenarios,
            stem=f"{POSTERIOR_BENEFIT_STEM}_batch_{batch_start // batch_size + 1:04d}",
            n_jobs=n_jobs,
        )
        summary_frames.append(summary)

    endpoint_summary = pd.concat(summary_frames, ignore_index=True)
    benefit_summary = _summarise_posterior_benefits(endpoint_summary)
    write_dataframe(
        benefit_summary,
        project_path("outputs", "summaries", f"{POSTERIOR_BENEFIT_STEM}_summary.csv"),
    )
    metadata = current_run_metadata(
        POSTERIOR_BENEFIT_STEM,
        row_counts={
            "posterior_endpoint_summary": int(len(endpoint_summary)),
            "posterior_benefit_summary": int(len(benefit_summary)),
        },
    )
    metadata.update(
        {
            "posterior_sample_path": str(sample_path),
            "posterior_sample_stem": posterior_stem,
            "posterior_sampler": str(posterior_metadata.get("sampler", "")),
            "posterior_draws_per_country_requested": int(posterior_draws),
            "posterior_draws_selected": int(len(selected)),
            "posterior_seed": int(posterior_seed),
            "posterior_batch_size": int(batch_size),
            "fitness_targets": targets,
            "low_grid_VE_inf": low_ve_inf,
            "high_grid_VE_inf": high_ve_inf,
            "uncertainty_source": UNCERTAINTY_SOURCE,
            "uncertainty_scope": UNCERTAINTY_SCOPE,
            "posterior_sample_diagnostics": f"outputs/summaries/{POSTERIOR_DIAGNOSTIC_STEM}.csv",
        }
    )
    write_run_metadata(POSTERIOR_BENEFIT_STEM, metadata)
    return benefit_summary


def _run_psa_benefit_intervals(
    *,
    configs: dict[str, Any],
    resistance_name: str,
    fitness_values: list[float],
    ve_inf_values: list[float],
    psa_sample_path: str | Path,
    psa_sample_limit: int | None,
    psa_batch_size: int,
    n_jobs: int | None,
) -> pd.DataFrame:
    samples = _load_psa_samples(psa_sample_path, sample_limit=psa_sample_limit)
    targets = _fitness_targets(fitness_values)
    countries = publication_country_names(configs)
    low_ve_inf = float(min(ve_inf_values))
    high_ve_inf = float(max(ve_inf_values))

    sample_path = Path(psa_sample_path)
    if not sample_path.is_absolute():
        sample_path = project_path(sample_path)

    write_dataframe(
        samples,
        project_path("outputs", "tables", f"{PSA_BENEFIT_STEM}_parameter_samples_used.csv"),
    )

    summary_frames: list[pd.DataFrame] = []
    batch_size = max(1, int(psa_batch_size))
    for batch_start in range(0, len(samples), batch_size):
        batch = samples.iloc[batch_start : batch_start + batch_size].copy()
        scenarios = _build_psa_benefit_scenarios(
            batch,
            countries=countries,
            resistance_name=resistance_name,
            fitness_targets=targets,
            low_ve_inf=low_ve_inf,
            high_ve_inf=high_ve_inf,
        )
        summary = execute_scenario_summary_list(
            scenarios,
            stem=f"{PSA_BENEFIT_STEM}_batch_{batch_start // batch_size + 1:04d}",
            n_jobs=n_jobs,
        )
        summary_frames.append(summary)

    endpoint_summary = pd.concat(summary_frames, ignore_index=True)
    benefit_draws, benefit_summary = _summarise_psa_benefits(endpoint_summary)
    write_dataframe(
        benefit_draws,
        project_path("outputs", "tables", f"{PSA_BENEFIT_STEM}_draws.csv"),
    )
    write_dataframe(
        benefit_summary,
        project_path("outputs", "tables", f"{PSA_BENEFIT_STEM}_summary.csv"),
    )
    metadata = current_run_metadata(
        PSA_BENEFIT_STEM,
        row_counts={
            "psa_endpoint_summary": int(len(endpoint_summary)),
            "psa_benefit_draws": int(len(benefit_draws)),
            "psa_benefit_summary": int(len(benefit_summary)),
            "psa_parameter_samples_used": int(len(samples)),
        },
    )
    metadata.update(
        {
            "psa_sample_path": str(sample_path),
            "psa_sample_limit": None if psa_sample_limit is None else int(psa_sample_limit),
            "psa_samples_used": int(len(samples)),
            "psa_batch_size": int(batch_size),
            "fitness_targets": targets,
            "low_grid_VE_inf": low_ve_inf,
            "high_grid_VE_inf": high_ve_inf,
            "varied_psa_parameters": sorted(PSA_APPLIED_COLUMNS),
            "ignored_grid_parameter_columns": sorted(PSA_IGNORED_GRID_COLUMNS),
            "paired_comparison": True,
            "uncertainty_source": PSA_UNCERTAINTY_SOURCE,
            "uncertainty_scope": PSA_UNCERTAINTY_SCOPE,
            "psa_draws": f"outputs/tables/{PSA_BENEFIT_STEM}_draws.csv",
            "psa_parameter_samples_used": (
                f"outputs/tables/{PSA_BENEFIT_STEM}_parameter_samples_used.csv"
            ),
        }
    )
    write_run_metadata(PSA_BENEFIT_STEM, metadata)
    return benefit_summary


def main(
    n_jobs: int | None = None,
    *,
    posterior_draws: int = 100,
    posterior_sample_path: str | Path = DEFAULT_POSTERIOR_SAMPLE_PATH,
    posterior_seed: int = 20260521,
    posterior_batch_size: int = 10,
    psa_benefit_samples: int = 0,
    psa_sample_path: str | Path = DEFAULT_PSA_SAMPLE_PATH,
    psa_batch_size: int = 4,
    run_deterministic_grid: bool = True,
):
    configs = load_configs()
    grid_settings = configs["baseline"].get("fitness_grid", {})
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    fitness_values = _grid_values(
        grid_settings,
        "fitness_R_values",
        [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25],
    )
    ve_inf_values = _grid_values(grid_settings, "VE_inf_values", [0.08, 0.25, 0.40, 0.60, 0.75])

    result = None
    if run_deterministic_grid:
        scenarios = []
        for country in publication_country_names(configs):
            for fitness_r in fitness_values:
                for ve_inf in ve_inf_values:
                    scenario = f"fitness_{fitness_r:.2f}_VEinf_{ve_inf:.2f}"
                    config = make_config(
                        vaccine_scenario="symptom_protective",
                        resistance_scenario=resistance_name,
                        country_profile=country,
                        vaccine_overrides={"VE_inf": float(ve_inf)},
                        resistance_overrides={"fitness_R": float(fitness_r)},
                    )
                    scenarios.append(
                        {
                            "config": config,
                            "analysis": "fitness_resistance_grid",
                            "scenario": scenario,
                            "vaccine_scenario": "symptom_protective",
                            "resistance_scenario": resistance_name,
                            "metadata": {
                                "country": country,
                                "grid_fitness_R": float(fitness_r),
                                "grid_VE_inf": float(ve_inf),
                            },
                        }
                    )

        reference_fitness = min(fitness_values, key=lambda value: abs(value - 0.70))
        reference_ve_inf = min(ve_inf_values, key=lambda value: abs(value - 0.08))
        reference = f"fitness_{reference_fitness:.2f}_VEinf_{reference_ve_inf:.2f}"
        result = run_scenario_list(
            scenarios,
            stem="fitness_resistance_grid",
            reference_scenario=reference,
            n_jobs=n_jobs,
        )
    _run_posterior_benefit_intervals(
        configs=configs,
        resistance_name=resistance_name,
        fitness_values=fitness_values,
        ve_inf_values=ve_inf_values,
        posterior_draws=int(posterior_draws),
        posterior_sample_path=posterior_sample_path,
        posterior_seed=int(posterior_seed),
        posterior_batch_size=int(posterior_batch_size),
        n_jobs=n_jobs,
    )
    if int(psa_benefit_samples) != 0:
        psa_sample_limit = None if int(psa_benefit_samples) < 0 else int(psa_benefit_samples)
        _run_psa_benefit_intervals(
            configs=configs,
            resistance_name=resistance_name,
            fitness_values=fitness_values,
            ve_inf_values=ve_inf_values,
            psa_sample_path=psa_sample_path,
            psa_sample_limit=psa_sample_limit,
            psa_batch_size=int(psa_batch_size),
            n_jobs=n_jobs,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continuous resistant-fitness and VE_inf stress grid.")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=100,
        help="Conditional uncertainty draws per country for Fig 3D benefit intervals. Use 0 to skip.",
    )
    parser.add_argument(
        "--posterior-sample-path",
        type=str,
        default=str(DEFAULT_POSTERIOR_SAMPLE_PATH),
        help="Path to quality-gated conditional state/external-prior samples used for Fig 3D uncertainty intervals.",
    )
    parser.add_argument("--posterior-seed", type=int, default=20260521)
    parser.add_argument(
        "--posterior-batch-size",
        type=int,
        default=10,
        help="Number of country-draw posterior rows to execute per endpoint batch.",
    )
    parser.add_argument(
        "--skip-deterministic-grid",
        action="store_true",
        help="Refresh posterior Fig 3D benefit intervals without rerunning the deterministic full grid.",
    )
    parser.add_argument(
        "--psa-benefit-samples",
        type=int,
        default=0,
        help=(
            "Run targeted Fig 3D LHS PSA with this many samples. "
            "Use -1 for all samples in --psa-sample-path; default 0 skips."
        ),
    )
    parser.add_argument(
        "--psa-sample-path",
        type=str,
        default=str(DEFAULT_PSA_SAMPLE_PATH),
        help="Path to joint PSA parameter samples used for targeted Fig 3D robustness intervals.",
    )
    parser.add_argument(
        "--psa-batch-size",
        type=int,
        default=4,
        help="Number of PSA rows to execute per endpoint batch.",
    )
    args = parser.parse_args()
    main(
        n_jobs=args.n_jobs,
        posterior_draws=args.posterior_draws,
        posterior_sample_path=args.posterior_sample_path,
        posterior_seed=args.posterior_seed,
        posterior_batch_size=args.posterior_batch_size,
        psa_benefit_samples=args.psa_benefit_samples,
        psa_sample_path=args.psa_sample_path,
        psa_batch_size=args.psa_batch_size,
        run_deterministic_grid=not args.skip_deterministic_grid,
    )
