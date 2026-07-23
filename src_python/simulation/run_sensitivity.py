from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    PROSPECTIVE_POLICY_KEY,
    enforce_calibration_status,
    execute_scenario_list,
    load_configs,
    make_config,
    validate_calibration_artifacts,
    write_outputs,
)
from src_python.simulation.parameter_distributions import (
    UNCERTAINTY_REGISTRY_SCHEMA_VERSION,
    latin_hypercube_draw_table,
    validate_distribution_spec,
    validate_uncertainty_registry_schema,
)
from src_python.utils.io import project_path, set_by_dotted_path, write_dataframe


SAMPLE_PATH = project_path("outputs", "tables", "sensitivity_parameter_samples.csv")
SAMPLE_DESIGN = "latin_hypercube_inverse_cdf"
UNCERTAINTY_SCHEMA_VERSION = UNCERTAINTY_REGISTRY_SCHEMA_VERSION
RUNNER_CONSUMER = "src_python.simulation.run_sensitivity"
STRUCTURAL_ALL_TIME = "structural_all_time"
OBSERVATION_ONLY = "observation_only"
GLOBAL_SENSITIVITY_PARAMETER_NAMES = (
    "latent_duration",
    "vaccine_disease_efficacy",
    "infectious_duration_symptomatic",
    "asymptomatic_duration_ratio",
    "natural_immunity_duration",
    "vaccine_protection_duration",
    "PEP_effectiveness_sensitive",
    "PEP_coverage",
    "fitness_R",
    "reporting_multiplier_factor",
)
GLOBAL_SENSITIVITY_PARAMETER_TIME_SCOPES = {
    **{
        name: STRUCTURAL_ALL_TIME
        for name in GLOBAL_SENSITIVITY_PARAMETER_NAMES
        if name != "reporting_multiplier_factor"
    },
    "reporting_multiplier_factor": OBSERVATION_ONLY,
}


def _validated_parameter_time_scopes(
    parameter_specs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Fail closed unless the registry matches the implemented 10-D contract."""

    actual_names = set(parameter_specs)
    expected_names = set(GLOBAL_SENSITIVITY_PARAMETER_NAMES)
    if actual_names != expected_names:
        raise ValueError(
            "Global sensitivity parameter registry must match the implemented "
            "10-parameter semantic contract; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    scopes: dict[str, str] = {}
    for name in GLOBAL_SENSITIVITY_PARAMETER_NAMES:
        spec = parameter_specs[name]
        declared = spec.get("time_scope")
        implemented = GLOBAL_SENSITIVITY_PARAMETER_TIME_SCOPES[name]
        if declared != implemented:
            raise ValueError(
                f"Global sensitivity parameter {name!r} declares "
                f"time_scope={declared!r}, but its implemented scope is "
                f"{implemented!r}"
            )
        consumers = spec.get("consumers")
        if not isinstance(consumers, list) or RUNNER_CONSUMER not in consumers:
            raise ValueError(
                f"Global sensitivity parameter {name!r} must declare "
                f"{RUNNER_CONSUMER!r} as a consumer"
            )
        scopes[name] = implemented

    reporting_spec = parameter_specs["reporting_multiplier_factor"]
    reporting_contract = {
        "apply": "multiply",
        "path": "reporting_multiplier",
        "outcome": "total_reported_cases",
    }
    for field, expected in reporting_contract.items():
        if reporting_spec.get(field) != expected:
            raise ValueError(
                "reporting_multiplier_factor must remain observation-only: "
                f"expected {field}={expected!r}, got "
                f"{reporting_spec.get(field)!r}"
            )
    return scopes


def _attached_history_config(config: dict[str, Any]) -> dict[str, Any] | None:
    policy = config.get(PROSPECTIVE_POLICY_KEY)
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise ValueError("Prospective-policy metadata must be a mapping")
    history = policy.get("history_config")
    if not isinstance(history, dict):
        raise ValueError("Prospective-policy metadata is missing a historical config")
    return history


def _set_parameter_value(config: dict[str, Any], path: str, value: float) -> None:
    """Apply a resolved scalar while retaining legacy rate aliases."""

    if path == "reporting_multiplier":
        config["reporting_multiplier"] = value
    elif path == "rates.waning_vaccine":
        config["natural_history"]["vaccine_protection_duration"] = 1.0 / max(value, 1e-12)
    elif path == "rates.waning_natural":
        config["natural_history"]["recovered_immunity_duration"] = 1.0 / max(value, 1e-12)
    else:
        set_by_dotted_path(config, path, value)


def _apply_parameter(
    config: dict[str, Any],
    *,
    name: str,
    spec: dict[str, Any],
    value: float,
) -> None:
    """Apply one epistemic draw using its semantic update contract."""

    apply = str(spec.get("apply", "replace"))
    if apply == "replace":
        path = str(spec.get("path", ""))
        if not path:
            raise ValueError(f"Parameter {name!r} requires a dotted 'path' for replace updates")
        _set_parameter_value(config, path, value)
        return

    if apply == "multiply":
        path = str(spec.get("path", ""))
        if path != "reporting_multiplier":
            raise ValueError(f"Parameter {name!r}: multiply currently supports reporting_multiplier only")
        config["reporting_multiplier"] = float(config.get("reporting_multiplier", 1.0)) * value
        return

    if apply == "derive_VE_sym_from_overall_disease_efficacy":
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Parameter {name!r}: overall disease efficacy must lie in [0, 1]")
        vaccine = config.setdefault("vaccine", {})
        ve_sus = float(vaccine.get("VE_sus", 0.0))
        if not 0.0 <= ve_sus < 1.0 or value < ve_sus:
            raise ValueError(
                f"Parameter {name!r}: overall efficacy {value:.6g} must be >= VE_sus "
                f"{ve_sus:.6g} and VE_sus must be < 1"
            )
        vaccine["VE_sym"] = float(np.clip(1.0 - (1.0 - value) / (1.0 - ve_sus), 0.0, 1.0))
        return

    if apply == "scale_symptomatic_duration":
        target = str(spec.get("target", "natural_history.infectious_duration_asymptomatic"))
        symptomatic = float(config["natural_history"]["infectious_duration_symptomatic"])
        _set_parameter_value(config, target, symptomatic * value)
        return

    if apply == "split_sirws_immunity_duration":
        if value <= 0.0:
            raise ValueError(f"Parameter {name!r}: immunity duration must be > 0")
        # Two equal stages give an Erlang/Gamma shape-2 residence time in the
        # absence of boosting. Keep the legacy no-boosting duration coherent.
        half = 0.5 * value
        natural_history = config.setdefault("natural_history", {})
        natural_history["R_to_W_duration"] = half
        natural_history["W_to_S_duration"] = half
        natural_history["recovered_immunity_duration"] = value
        return

    if apply == "split_vaccine_protection_duration":
        if value <= 0.0:
            raise ValueError(f"Parameter {name!r}: vaccine protection duration must be > 0")
        half = 0.5 * value
        natural_history = config.setdefault("natural_history", {})
        immunity_model = config.setdefault("immunity_model", {})
        natural_history["vaccine_protection_duration"] = half
        immunity_model["waned_vaccine_duration"] = half
        return

    raise ValueError(f"Parameter {name!r} has unsupported apply mode {apply!r}")


def _apply_sample(
    config: dict[str, Any],
    sample: dict[str, float],
    specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply a coherent parameter draw to a copied model configuration.

    ``specs=None`` preserves the historical dotted-path API for callers and
    tests.  Passing named specifications enables semantic updates for derived
    vaccine effects and staged SIRWS immunity.
    """

    out = deepcopy(config)
    if specs is None:
        for path, value in sample.items():
            _set_parameter_value(out, path, float(value))
        return out

    scopes = _validated_parameter_time_scopes(specs)
    missing = [name for name in specs if name not in sample]
    extra = [name for name in sample if name not in specs]
    if missing or extra:
        raise ValueError(f"Parameter draw/spec mismatch: missing={missing}, extra={extra}")
    # Apply resolved scalar values before quantities derived from them.  This
    # makes the result independent of YAML insertion order (for example, the
    # asymptomatic-duration ratio must see the sampled symptomatic duration).
    direct_modes = {"replace", "multiply"}
    ordered_items = [
        (name, spec)
        for name, spec in specs.items()
        if str(spec.get("apply", "replace")) in direct_modes
    ] + [
        (name, spec)
        for name, spec in specs.items()
        if str(spec.get("apply", "replace")) not in direct_modes
    ]
    for name, spec in ordered_items:
        targets = [out]
        if scopes[name] == STRUCTURAL_ALL_TIME:
            history = _attached_history_config(out)
            if history is not None:
                targets.append(history)
        elif scopes[name] != OBSERVATION_ONLY:
            raise ValueError(
                f"Global sensitivity parameter {name!r} has unsupported "
                f"implemented scope {scopes[name]!r}"
            )
        for target in targets:
            _apply_parameter(
                target,
                name=name,
                spec=spec,
                value=float(sample[name]),
            )
    return out


def _settings(configs: dict[str, Any]) -> tuple[dict[str, Any], int]:
    registry = configs.get("parameter_distributions", {})
    schema_version = validate_uncertainty_registry_schema(registry)
    configured = registry.get("global_sensitivity")
    if not isinstance(configured, dict) or not configured.get("parameters"):
        raise ValueError(
            "Current global sensitivity requires parameter_distributions."
            "global_sensitivity.parameters; legacy range fallback is disabled"
        )
    _validated_parameter_time_scopes(configured["parameters"])
    return configured, schema_version


def _sample_table(
    settings: dict[str, Any],
    *,
    sample_size: int | None = None,
    seed: int | None = None,
    schema_version: int = UNCERTAINTY_SCHEMA_VERSION,
) -> pd.DataFrame:
    specs = settings.get("parameters", {})
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Sensitivity settings require a non-empty 'parameters' mapping")
    schema_version = validate_uncertainty_registry_schema(
        {"schema_version": schema_version},
        context="global sensitivity sample contract",
    )
    _validated_parameter_time_scopes(specs)
    n = int(settings.get("sample_size", 48) if sample_size is None else sample_size)
    random_seed = int(settings.get("random_seed", 20260430) if seed is None else seed)
    draws = latin_hypercube_draw_table(specs, n, seed=random_seed)
    draws.insert(0, "sample_design", SAMPLE_DESIGN)
    draws.insert(0, "uncertainty_schema_version", schema_version)
    draws.insert(0, "sample_id", np.arange(1, len(draws) + 1, dtype=int))
    return draws


def _derived_metadata(config: dict[str, Any]) -> dict[str, float]:
    natural_history = config["natural_history"]
    return {
        "resolved_VE_sym": float(config["vaccine"]["VE_sym"]),
        "resolved_infectious_duration_asymptomatic": float(
            natural_history["infectious_duration_asymptomatic"]
        ),
        "resolved_R_to_W_duration": float(natural_history["R_to_W_duration"]),
        "resolved_W_to_S_duration": float(natural_history["W_to_S_duration"]),
        "resolved_vaccine_recent_duration": float(
            natural_history["vaccine_protection_duration"]
        ),
        "resolved_vaccine_waned_duration": float(
            config["immunity_model"]["waned_vaccine_duration"]
        ),
        "resolved_reporting_multiplier": float(config.get("reporting_multiplier", 1.0)),
    }


def _baseline_sensitivity_config(configs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the calibrated country profile used by the reference PSA.

    The core baseline runner uses China when no explicit baseline country is
    configured.  Sensitivity draws must use that same country profile rather
    than the generic placeholder population/contact matrix in
    ``baseline_parameters``.
    """

    country = str(configs["baseline"].get("baseline_country_profile", "China"))
    if country not in configs.get("countries", {}):
        raise KeyError(f"Configured baseline country profile {country!r} is not available")
    resistance_name = configs["baseline"].get(
        "baseline_resistance_scenario",
        "country_timeline",
    )
    config = make_config(
        vaccine_scenario="symptom_protective",
        resistance_scenario=resistance_name,
        country_profile=country,
    )
    return country, config


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    sensitivity, schema_version = _settings(configs)
    specs = sensitivity["parameters"]
    parameter_time_scopes = _validated_parameter_time_scopes(specs)
    names = list(specs)
    baseline_country = str(
        configs["baseline"].get("baseline_country_profile", "China")
    )
    validate_calibration_artifacts(
        (baseline_country,),
        context="Global sensitivity analysis",
    )
    samples = _sample_table(sensitivity, schema_version=schema_version)
    write_dataframe(samples, SAMPLE_PATH)

    scenarios = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    resolved_baseline_country, base_config = _baseline_sensitivity_config(configs)
    if resolved_baseline_country != baseline_country:
        raise RuntimeError("Sensitivity baseline-country resolution changed unexpectedly")
    for row in samples.to_dict(orient="records"):
        run_idx = int(row["sample_id"])
        sampled_values = {name: float(row[name]) for name in names}
        config = _apply_sample(base_config, sampled_values, specs)
        metadata = {
            **sampled_values,
            **_derived_metadata(config),
            "sample_id": run_idx,
            "sample_design": str(row["sample_design"]),
            "uncertainty_schema_version": int(row["uncertainty_schema_version"]),
            "country": baseline_country,
        }
        scenario = f"lhs_{run_idx:03d}"
        scenarios.append(
            {
                "config": config,
                "analysis": "sensitivity",
                "scenario": scenario,
                "vaccine_scenario": "sampled",
                "resistance_scenario": resistance_name,
                "metadata": metadata,
            }
        )

    timeseries, summary = execute_scenario_list(scenarios, stem="sensitivity_lhs")
    enforce_calibration_status(summary, stem="sensitivity_runs")
    _add_tornado_metrics(summary, specs)
    write_outputs(
        timeseries,
        summary,
        "sensitivity_runs",
        extra_metadata={
            "uncertainty_schema_version": int(schema_version),
            "sample_design": samples["sample_design"].iloc[0],
            "sample_seed": int(sensitivity.get("random_seed", 20260430)),
            "sample_size": int(len(samples)),
            "parameter_names": names,
            "parameter_time_scopes": parameter_time_scopes,
            "parameter_distributions": {
                name: validate_distribution_spec(spec, context=f"sensitivity parameter {name!r}")
                for name, spec in specs.items()
            },
            "parameter_sample_path": str(SAMPLE_PATH),
            "baseline_country_profile": baseline_country,
            "uncertainty_scope": (
                "epistemic_parameter_means_single_calibrated_baseline_profile_conditional_on_"
                "fixed_calibrated_transmission_reporting_and_contact_matrix"
            ),
        },
    )
    return timeseries, summary


def _add_tornado_metrics(
    summary: pd.DataFrame,
    parameter_specs: dict[str, dict[str, Any]],
) -> None:
    suffixes = {
        "total_child_adolescent_cases": "child_adolescent_cases",
        "total_infant_cases": "infant_cases",
        "total_reported_cases": "reported_cases",
        "total_infant_deaths": "infant_deaths",
        "total_child_adolescent_hospitalizations": "child_adolescent_hospitalizations",
    }
    for name, spec in parameter_specs.items():
        # The submitted analysis uses symptomatic cases in people younger than
        # 18 years as its primary burden endpoint. Observation-only parameters
        # may explicitly select a reported-case outcome in the registry.
        outcome_name = str(spec.get("outcome", "total_child_adolescent_cases"))
        if outcome_name not in summary:
            raise KeyError(f"Sensitivity outcome {outcome_name!r} for {name!r} is not in summary")
        outcome = summary[outcome_name].to_numpy(dtype=float)
        x = summary[name].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(outcome) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(x, outcome)[0, 1])
        suffix = suffixes.get(outcome_name, outcome_name)
        summary[f"corr_{name}_{suffix}"] = correlation


if __name__ == "__main__":
    main()
