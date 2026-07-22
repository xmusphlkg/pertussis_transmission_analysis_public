"""Independent PSA for resistance-management implementation uncertainty.

This analysis deliberately sits outside the Figure 2b programme-ranking PSA.
It varies prospective uptake and guided-pathway PEP reach, crosses the same
draws with restored and not-restored resistant-strain PEP structural strata,
keeps the 2027 historical state common, and compares the resulting burdens
with routine timeliness over the existing five-year implementation window.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    current_run_metadata,
    enforce_calibration_status,
    execute_scenario_summary_list,
    load_configs,
    make_intervention_config,
    publication_country_names,
    set_analysis_horizon_years,
    uncertainty_config_fingerprint,
    validated_calibration_artifact_path_hashes,
    write_run_metadata,
)
from src_python.simulation.parameter_distributions import (
    UNCERTAINTY_REGISTRY_SCHEMA_VERSION,
    latin_hypercube_draw_table,
    validate_distribution_spec,
    validate_uncertainty_registry_schema,
)
from src_python.simulation.resistance_management_uptake import (
    apply_resistance_management_sample,
)
from src_python.simulation.run_routine_timeliness_sensitivity import _apply_timeliness
from src_python.utils.io import project_path, write_dataframe


STEM = "resistance_management_psa"
SAMPLE_DESIGN = "latin_hypercube_inverse_cdf_resistance_management"
UNCERTAINTY_SCHEMA_VERSION = UNCERTAINTY_REGISTRY_SCHEMA_VERSION
EXPECTED_PARAMETER_NAMES = (
    "resistance_management_uptake",
    "resistance_management_pep_reach_multiplier",
)
PEP_RESTORATION_STRATA = (
    ("restored", True),
    ("not_restored", False),
)

SAMPLE_PATH = project_path(
    "outputs", "tables", "resistance_management_psa_parameter_samples.csv"
)
DRAW_PATH = project_path(
    "outputs", "tables", "resistance_management_psa_effect_samples.csv"
)
SUMMARY_PATH = project_path(
    "outputs", "summaries", "resistance_management_psa_summary.csv"
)


def _settings(configs: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    registry = configs.get("parameter_distributions", {})
    schema_version = validate_uncertainty_registry_schema(registry)
    settings = registry.get("resistance_management_psa")
    if not isinstance(settings, Mapping):
        raise ValueError("Missing parameter_distributions.resistance_management_psa")
    return dict(settings), schema_version


def _parameter_specs(settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = settings.get("parameters")
    if not isinstance(raw, Mapping):
        raise ValueError("resistance_management_psa.parameters must be a mapping")
    specs = {str(name): dict(spec) for name, spec in raw.items()}
    if tuple(specs) != EXPECTED_PARAMETER_NAMES:
        raise ValueError(
            "Resistance-management PSA parameter registry must exactly match its "
            f"implemented semantic contract; expected={list(EXPECTED_PARAMETER_NAMES)}, "
            f"actual={list(specs)}"
        )
    for name, spec in specs.items():
        validate_distribution_spec(spec, context=f"resistance-management PSA parameter {name!r}")
        if spec.get("time_scope") != "prospective_implementation":
            raise ValueError(
                f"Resistance-management parameter {name!r} must declare "
                "time_scope: prospective_implementation"
            )
        consumers = spec.get("consumers", [])
        if (
            not isinstance(consumers, list)
            or "src_python.simulation.run_resistance_management_psa" not in consumers
        ):
            raise ValueError(
                f"Resistance-management parameter {name!r} must declare this runner "
                "in its consumers list"
            )
    return specs


def _sample_table(
    sample_size: int,
    seed: int,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    *,
    schema_version: int = UNCERTAINTY_SCHEMA_VERSION,
) -> pd.DataFrame:
    schema_version = validate_uncertainty_registry_schema(
        {"schema_version": schema_version},
        context="resistance-management PSA sample contract",
    )
    if int(sample_size) < 2:
        raise ValueError("Resistance-management PSA requires at least two samples")
    specs = {str(name): dict(spec) for name, spec in parameter_specs.items()}
    if tuple(specs) != EXPECTED_PARAMETER_NAMES:
        raise ValueError(
            "Resistance-management PSA sample columns do not match the implemented "
            f"contract: {list(specs)}"
        )
    draws = latin_hypercube_draw_table(specs, int(sample_size), seed=int(seed))
    draws.insert(0, "uncertainty_schema_version", schema_version)
    draws.insert(0, "sample_design", SAMPLE_DESIGN)
    draws.insert(0, "psa_sample_id", np.arange(1, int(sample_size) + 1, dtype=int))
    return draws.loc[
        :,
        [
            "psa_sample_id",
            "sample_design",
            "uncertainty_schema_version",
            *EXPECTED_PARAMETER_NAMES,
        ],
    ]


def _set_runtime(config: dict[str, Any], *, years: int) -> None:
    set_analysis_horizon_years(config, int(years))
    config["simulation"]["output_time_step"] = 30.0
    config["simulation"]["rtol"] = max(
        float(config["simulation"].get("rtol", 1e-5)), 1e-4
    )
    config["simulation"]["atol"] = max(
        float(config["simulation"].get("atol", 1e-7)), 1e-6
    )


def _comparator_scenarios(
    countries: list[str],
    *,
    resistance_name: str,
    horizon_years: int,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for country in countries:
        config, vaccine_name = make_intervention_config("current", country_profile=country)
        config = _apply_timeliness(config)
        _set_runtime(config, years=horizon_years)
        scenarios.append(
            {
                "config": config,
                "analysis": STEM,
                "scenario": f"{country}_routine_timeliness",
                "vaccine_scenario": vaccine_name,
                "resistance_scenario": resistance_name,
                "intervention": "timeliness_only",
                "metadata": {
                    "country": country,
                    "resistance_management_role": "comparator",
                    "programme_base": "routine_timeliness",
                },
            }
        )
    return scenarios


def _guided_scenarios(
    samples: pd.DataFrame,
    countries: list[str],
    *,
    resistance_name: str,
    horizon_years: int,
    parameter_specs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    if parameter_specs is None:
        settings, _ = _settings(load_configs())
        parameter_specs = _parameter_specs(settings)
    for row in samples.to_dict(orient="records"):
        sample_id = int(row["psa_sample_id"])
        uptake = float(row["resistance_management_uptake"])
        pep_reach_multiplier = float(
            row["resistance_management_pep_reach_multiplier"]
        )
        for country in countries:
            current_config, _ = make_intervention_config(
                "current", country_profile=country
            )
            full_config, vaccine_name = make_intervention_config(
                "resistance_guided_treatment", country_profile=country
            )
            # Hold the programme lever fixed on both sides of the comparison.
            # Otherwise the uptake effect would be confounded with the absence
            # of routine-timeliness improvement in the guided arm.
            current_config = _apply_timeliness(current_config)
            full_config = _apply_timeliness(full_config)
            for pep_restoration, pep_restored in PEP_RESTORATION_STRATA:
                config = apply_resistance_management_sample(
                    full_config,
                    current_config,
                    strategy="resistance_guided_treatment",
                    sample=row,
                    pep_restored=pep_restored,
                    parameter_specs=parameter_specs,
                )
                _set_runtime(config, years=horizon_years)
                config.setdefault("metadata", {})[
                    "resistance_management_psa_sample_id"
                ] = sample_id
                scenarios.append(
                    {
                        "config": config,
                        "analysis": STEM,
                        "scenario": (
                            f"{country}_guided_{pep_restoration}_sample_{sample_id:04d}"
                        ),
                        "vaccine_scenario": vaccine_name,
                        "resistance_scenario": resistance_name,
                        "intervention": "resistance_guided_treatment",
                        "metadata": {
                            "country": country,
                            "psa_sample_id": sample_id,
                            "sample_design": str(row["sample_design"]),
                            "uncertainty_schema_version": int(
                                row["uncertainty_schema_version"]
                            ),
                            "resistance_management_uptake": uptake,
                            "resistance_management_pep_reach_multiplier": (
                                pep_reach_multiplier
                            ),
                            "resistance_management_role": "guided_pathway",
                            "programme_base": "routine_timeliness",
                            "pep_restoration": pep_restoration,
                            "pep_restored": pep_restored,
                            "baseline_treatment_rate_symptomatic": float(
                                current_config["treatment"][
                                    "treatment_rate_symptomatic"
                                ]
                            ),
                            "full_guided_treatment_rate_symptomatic": float(
                                full_config["treatment"][
                                    "treatment_rate_symptomatic"
                                ]
                            ),
                            "applied_treatment_rate_symptomatic": float(
                                config["treatment"]["treatment_rate_symptomatic"]
                            ),
                            "baseline_pep_coverage": float(
                                current_config["PEP"][
                                    "coverage_household_contacts"
                                ]
                            ),
                            "applied_pep_coverage": float(
                                config["PEP"]["coverage_household_contacts"]
                            ),
                            "baseline_pep_effectiveness_resistant": float(
                                current_config["PEP"]["effectiveness_resistant"]
                            ),
                            "full_guided_pep_effectiveness_resistant": float(
                                full_config["PEP"]["effectiveness_resistant"]
                            ),
                            "applied_pep_effectiveness_resistant": float(
                                config["PEP"]["effectiveness_resistant"]
                            ),
                        },
                    }
                )
    return scenarios


_BURDEN_COLUMNS = (
    "total_child_adolescent_cases",
    "annualized_child_adolescent_cases_per_100k",
    "total_infant_cases",
    "annualized_infant_cases_per_100k",
    "resistant_infections",
)

_APPLICATION_AUDIT_COLUMNS = (
    "baseline_treatment_rate_symptomatic",
    "full_guided_treatment_rate_symptomatic",
    "applied_treatment_rate_symptomatic",
    "baseline_pep_coverage",
    "applied_pep_coverage",
    "baseline_pep_effectiveness_resistant",
    "full_guided_pep_effectiveness_resistant",
    "applied_pep_effectiveness_resistant",
)


def _effect_draws(
    comparator: pd.DataFrame,
    guided: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    required_comparator = {"country", *_BURDEN_COLUMNS}
    required_guided = {
        "country",
        "psa_sample_id",
        "pep_restoration",
        "pep_restored",
        *EXPECTED_PARAMETER_NAMES,
        *_APPLICATION_AUDIT_COLUMNS,
        *_BURDEN_COLUMNS,
    }
    missing_comparator = sorted(required_comparator.difference(comparator.columns))
    missing_guided = sorted(required_guided.difference(guided.columns))
    if missing_comparator or missing_guided:
        raise KeyError(
            "Resistance-management PSA endpoint output is incomplete; "
            f"comparator_missing={missing_comparator}, guided_missing={missing_guided}"
        )
    if comparator.duplicated("country").any():
        raise ValueError("Resistance-management comparator must contain one row per country")
    if guided.duplicated(["country", "psa_sample_id", "pep_restoration"]).any():
        raise ValueError(
            "Resistance-management guided output contains duplicate "
            "country-sample-stratum rows"
        )

    comparator_columns = {column: f"timeliness_{column}" for column in _BURDEN_COLUMNS}
    guided_columns = {column: f"guided_{column}" for column in _BURDEN_COLUMNS}
    out = guided.loc[
        :,
        [
            "country",
            "psa_sample_id",
            "pep_restoration",
            "pep_restored",
            *EXPECTED_PARAMETER_NAMES,
            *_APPLICATION_AUDIT_COLUMNS,
            *_BURDEN_COLUMNS,
        ],
    ].rename(columns=guided_columns)
    out = out.merge(
        comparator.loc[:, ["country", *_BURDEN_COLUMNS]].rename(
            columns=comparator_columns
        ),
        on="country",
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        samples.loc[
            :,
            [
                "psa_sample_id",
                "sample_design",
                "uncertainty_schema_version",
                *EXPECTED_PARAMETER_NAMES,
            ],
        ],
        on="psa_sample_id",
        how="left",
        suffixes=("", "_sample"),
        validate="many_to_one",
    )
    for parameter in EXPECTED_PARAMETER_NAMES:
        sample_column = f"{parameter}_sample"
        if not np.allclose(
            pd.to_numeric(out[parameter], errors="raise"),
            pd.to_numeric(out[sample_column], errors="raise"),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"Resistance-management {parameter} metadata do not match the draw table"
            )
        out = out.drop(columns=sample_column)

    endpoint_map = {
        "under18_cases": "total_child_adolescent_cases",
        "infant_cases": "total_infant_cases",
        "resistant_infections": "resistant_infections",
    }
    for label, column in endpoint_map.items():
        denominator = pd.to_numeric(
            out[f"timeliness_{column}"], errors="raise"
        ).replace(0.0, np.nan)
        numerator = pd.to_numeric(out[f"guided_{column}"], errors="raise")
        out[f"relative_reduction_{label}_vs_timeliness"] = 1.0 - numerator / denominator
        out[f"guided_lower_{label}"] = numerator < denominator
    return out.sort_values(
        ["country", "pep_restoration", "psa_sample_id"]
    ).reset_index(drop=True)


def _activity_audit(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoints = (
        "relative_reduction_under18_cases_vs_timeliness",
        "relative_reduction_resistant_infections_vs_timeliness",
    )
    for (country, pep_restoration), group in draws.groupby(
        ["country", "pep_restoration"], sort=True
    ):
        uptake = pd.to_numeric(
            group["resistance_management_uptake"], errors="raise"
        )
        reach = pd.to_numeric(
            group["resistance_management_pep_reach_multiplier"], errors="raise"
        )
        expected_treatment = (
            pd.to_numeric(
                group["baseline_treatment_rate_symptomatic"], errors="raise"
            )
            + uptake
            * (
                pd.to_numeric(
                    group["full_guided_treatment_rate_symptomatic"],
                    errors="raise",
                )
                - pd.to_numeric(
                    group["baseline_treatment_rate_symptomatic"], errors="raise"
                )
            )
        )
        expected_coverage = np.minimum(
            1.0,
            pd.to_numeric(group["baseline_pep_coverage"], errors="raise")
            * reach,
        )
        if pep_restoration == "restored":
            expected_pep_effectiveness = (
                pd.to_numeric(
                    group["baseline_pep_effectiveness_resistant"], errors="raise"
                )
                + uptake
                * (
                    pd.to_numeric(
                        group["full_guided_pep_effectiveness_resistant"],
                        errors="raise",
                    )
                    - pd.to_numeric(
                        group["baseline_pep_effectiveness_resistant"],
                        errors="raise",
                    )
                )
            )
        elif pep_restoration == "not_restored":
            expected_pep_effectiveness = pd.to_numeric(
                group["baseline_pep_effectiveness_resistant"], errors="raise"
            )
        else:
            raise RuntimeError(
                f"Unknown resistance-management PEP restoration stratum {pep_restoration!r}"
            )

        application_checks = {
            "resistance_management_uptake": np.allclose(
                pd.to_numeric(
                    group["applied_treatment_rate_symptomatic"], errors="raise"
                ),
                expected_treatment,
                rtol=1e-12,
                atol=1e-12,
            )
            and np.allclose(
                pd.to_numeric(
                    group["applied_pep_effectiveness_resistant"], errors="raise"
                ),
                expected_pep_effectiveness,
                rtol=1e-12,
                atol=1e-12,
            ),
            "resistance_management_pep_reach_multiplier": np.allclose(
                pd.to_numeric(group["applied_pep_coverage"], errors="raise"),
                expected_coverage,
                rtol=1e-12,
                atol=1e-12,
            ),
        }
        configured_targets = {
            "resistance_management_uptake": pd.to_numeric(
                group["applied_treatment_rate_symptomatic"], errors="raise"
            ),
            "resistance_management_pep_reach_multiplier": pd.to_numeric(
                group["applied_pep_coverage"], errors="raise"
            ),
        }
        for parameter, values in (
            ("resistance_management_uptake", uptake),
            ("resistance_management_pep_reach_multiplier", reach),
        ):
            target_values = configured_targets[parameter]
            row: dict[str, Any] = {
                "country": country,
                "pep_restoration": pep_restoration,
                "parameter": parameter,
                "unique_values": int(values.nunique()),
                "sampled_range": float(values.max() - values.min()),
                "configured_target_range": float(
                    target_values.max() - target_values.min()
                ),
                "application_verified": bool(application_checks[parameter]),
            }
            for endpoint in endpoints:
                outcome = pd.to_numeric(group[endpoint], errors="raise")
                row[f"{endpoint}_range"] = float(outcome.max() - outcome.min())
                row[f"{endpoint}_spearman"] = (
                    float(values.corr(outcome, method="spearman"))
                    if values.nunique() > 1 and outcome.nunique() > 1
                    else float("nan")
                )
            row["parameter_active"] = bool(
                application_checks[parameter]
                and row["sampled_range"] > 0.0
                and row["configured_target_range"] > 1e-12
            )
            rows.append(row)
    audit = pd.DataFrame(rows)
    if audit.empty or not audit["parameter_active"].all():
        failed = audit.loc[~audit.get("parameter_active", pd.Series(dtype=bool))]
        raise RuntimeError(
            "A resistance-management PSA parameter is inactive or misapplied: "
            + ", ".join(
                (
                    failed.get("country", pd.Series(dtype=str)).astype(str)
                    + "/"
                    + failed.get("pep_restoration", pd.Series(dtype=str)).astype(str)
                    + "/"
                    + failed.get("parameter", pd.Series(dtype=str)).astype(str)
                )
            )
        )
    return audit


def _summarise(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoints = (
        "under18_cases",
        "infant_cases",
        "resistant_infections",
    )

    def append(label: str, pep_restoration: str, group: pd.DataFrame) -> None:
        row: dict[str, Any] = {
            "country": label,
            "pep_restoration": pep_restoration,
            "psa_samples": int(group["psa_sample_id"].nunique()),
            "interpretation": (
                "Elicited-input design summary within one deterministic PEP-restoration "
                "stratum; not a posterior probability or Figure 2b input."
            ),
        }
        for endpoint in endpoints:
            values = pd.to_numeric(
                group[f"relative_reduction_{endpoint}_vs_timeliness"],
                errors="raise",
            )
            row[f"median_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.median()
            )
            row[f"q025_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.quantile(0.025)
            )
            row[f"q975_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.quantile(0.975)
            )
            row[f"fraction_design_draws_guided_lower_{endpoint}"] = float(
                group[f"guided_lower_{endpoint}"].mean()
            )
        rows.append(row)

    for pep_restoration, _ in PEP_RESTORATION_STRATA:
        stratum = draws.loc[draws["pep_restoration"].eq(pep_restoration)]
        for country, group in stratum.groupby("country", sort=True):
            append(str(country), pep_restoration, group)
        append("All_countries_pooled", pep_restoration, stratum)
    return pd.DataFrame(rows)


def run_resistance_management_psa(
    *,
    sample_size: int | None = None,
    seed: int | None = None,
    countries: list[str] | None = None,
    n_jobs: int | None = None,
    sample_batch_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    settings, schema_version = _settings(configs)
    specs = _parameter_specs(settings)
    resolved_sample_size = int(
        settings.get("sample_size", 128) if sample_size is None else sample_size
    )
    resolved_seed = int(settings.get("random_seed", 20260522) if seed is None else seed)
    horizon_years = int(settings.get("analysis_horizon_years", 5))
    if horizon_years < 1:
        raise ValueError("resistance_management_psa.analysis_horizon_years must be positive")
    all_countries = publication_country_names(configs)
    resolved_countries = all_countries if countries is None else [str(item) for item in countries]
    unknown = sorted(set(resolved_countries).difference(all_countries))
    if unknown:
        raise ValueError(
            "Resistance-management PSA excludes countries outside the publication set: "
            + ", ".join(unknown)
        )
    calibration_input_hashes = validated_calibration_artifact_path_hashes(
        resolved_countries,
        context="Resistance-management PSA",
    )
    samples = _sample_table(
        resolved_sample_size,
        resolved_seed,
        specs,
        schema_version=schema_version,
    )
    write_dataframe(samples, SAMPLE_PATH)

    resistance_name = configs["baseline"].get(
        "baseline_resistance_scenario", "country_timeline"
    )
    comparator = execute_scenario_summary_list(
        _comparator_scenarios(
            resolved_countries,
            resistance_name=resistance_name,
            horizon_years=horizon_years,
        ),
        stem=f"{STEM}_routine_timeliness_comparator",
        n_jobs=n_jobs,
    )
    enforce_calibration_status(
        comparator,
        stem=f"{STEM}_routine_timeliness_comparator",
    )
    guided_frames: list[pd.DataFrame] = []
    batch_size = max(1, int(sample_batch_size))
    for start in range(0, len(samples), batch_size):
        batch = samples.iloc[start : start + batch_size].copy()
        first_id = int(batch["psa_sample_id"].iloc[0])
        last_id = int(batch["psa_sample_id"].iloc[-1])
        guided_batch = execute_scenario_summary_list(
            _guided_scenarios(
                batch,
                resolved_countries,
                resistance_name=resistance_name,
                horizon_years=horizon_years,
                parameter_specs=specs,
            ),
            stem=f"{STEM}_samples_{first_id:04d}_{last_id:04d}",
            n_jobs=n_jobs,
        )
        enforce_calibration_status(
            guided_batch,
            stem=f"{STEM}_samples_{first_id:04d}_{last_id:04d}",
        )
        guided_frames.append(guided_batch)
    guided = pd.concat(guided_frames, ignore_index=True)
    draws = _effect_draws(comparator, guided, samples)
    activity = _activity_audit(draws)
    summary = _summarise(draws)
    write_dataframe(draws, DRAW_PATH)
    write_dataframe(summary, SUMMARY_PATH)

    metadata = current_run_metadata(
        STEM,
        row_counts={
            "parameter_samples": int(len(samples)),
            "comparator_rows": int(len(comparator)),
            "guided_rows": int(len(guided)),
            "effect_samples": int(len(draws)),
            "summary_rows": int(len(summary)),
            "activity_audit_rows": int(len(activity)),
        },
    )
    metadata.update(
        {
            "sample_size_requested": resolved_sample_size,
            "sample_seed": resolved_seed,
            "sample_design": SAMPLE_DESIGN,
            "uncertainty_schema_version": schema_version,
            "uncertainty_config_hash": uncertainty_config_fingerprint(configs),
            "countries": resolved_countries,
            "analysis_horizon_years": horizon_years,
            "comparator": "routine_timeliness",
            "programme_base_in_both_arms": "routine_timeliness",
            "contrast": "routine_timeliness_plus_guided_management_vs_routine_timeliness_alone",
            "sampled_parameters": list(EXPECTED_PARAMETER_NAMES),
            "parameter_time_scopes": {
                name: str(spec["time_scope"]) for name, spec in specs.items()
            },
            "parameter_distributions": {
                name: validate_distribution_spec(
                    spec, context=f"resistance-management PSA parameter {name!r}"
                )
                for name, spec in specs.items()
            },
            "deterministic_structural_strata": [
                label for label, _ in PEP_RESTORATION_STRATA
            ],
            "paired_across_pep_restoration_strata": True,
            "pep_restoration_probability_assigned": False,
            "structural_strata": settings.get("structural_strata", {}),
            "activity_audit": activity.to_dict(orient="records"),
            "figure2b_excluded": True,
            "input_artifact_path_sha256": calibration_input_hashes,
            "interpretation": (
                "Independent resistance-management implementation PSA; design summaries "
                "are not posterior probabilities."
            ),
        }
    )
    write_run_metadata(STEM, metadata)
    return draws, summary, activity


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--countries", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    args = parser.parse_args()
    countries = (
        None
        if args.countries is None
        else [item.strip() for item in args.countries.split(",") if item.strip()]
    )
    return run_resistance_management_psa(
        sample_size=args.samples,
        seed=args.seed,
        countries=countries,
        n_jobs=args.n_jobs,
        sample_batch_size=args.sample_batch_size,
    )


if __name__ == "__main__":
    main()
