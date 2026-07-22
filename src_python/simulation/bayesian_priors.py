"""Resolve the central distribution registry for country-level Bayesian runs."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from src_python.simulation.parameter_distributions import (
    validate_distribution_spec,
    validate_uncertainty_registry_schema,
)


BAYESIAN_PARAMETER_NAMES = (
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
BAYESIAN_PARAMETER_PATHS = {
    "beta_S": "transmission.beta_S",
    "reporting_multiplier": "reporting_multiplier",
    "VE_sus": "vaccine.VE_sus",
    "VE_inf": "vaccine.VE_inf",
    "VE_dur": "vaccine.VE_dur",
    "relative_infectiousness_asymptomatic": (
        "transmission.relative_infectiousness_asymptomatic"
    ),
    "infectious_duration_symptomatic": (
        "natural_history.infectious_duration_symptomatic"
    ),
    "infectious_duration_asymptomatic": (
        "natural_history.infectious_duration_asymptomatic"
    ),
    "fitness_R": "transmission.fitness_R",
}
BAYESIAN_PARAMETER_TIME_SCOPES = {
    name: (
        "observation_only"
        if name == "reporting_multiplier"
        else "structural_all_time"
    )
    for name in BAYESIAN_PARAMETER_NAMES
}
BAYESIAN_LOCAL_STATE_PARAMETER_NAMES = (
    "beta_S",
    "reporting_multiplier",
)
BAYESIAN_SHARED_PARAMETER_NAMES = tuple(
    name
    for name in BAYESIAN_PARAMETER_NAMES
    if name not in BAYESIAN_LOCAL_STATE_PARAMETER_NAMES
)
BAYESIAN_STANDALONE_CONSUMER = (
    "src_python.simulation.run_bayesian_uncertainty"
)
BAYESIAN_HIERARCHICAL_CONSUMERS = (
    "src_python.simulation.run_hierarchical_joint_posterior",
    "src_python.simulation.run_hierarchical_joint_smc",
)
BAYESIAN_PARAMETER_CONSUMERS = {
    name: (
        (BAYESIAN_STANDALONE_CONSUMER,)
        if name in BAYESIAN_LOCAL_STATE_PARAMETER_NAMES
        else (BAYESIAN_STANDALONE_CONSUMER, *BAYESIAN_HIERARCHICAL_CONSUMERS)
    )
    for name in BAYESIAN_PARAMETER_NAMES
}


def _path_value(config: dict[str, Any], path: str) -> float:
    current: Any = config
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Bayesian prior centre path {path!r} is missing from runtime config")
        current = current[part]
    try:
        return float(current)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bayesian prior centre path {path!r} is not numeric") from exc


def _scale_width(spec: dict[str, Any], scale: float) -> None:
    if "sd" in spec:
        spec["sd"] = float(spec["sd"]) * scale
    elif "log_sd" in spec:
        spec["log_sd"] = float(spec["log_sd"]) * scale


def resolve_bayesian_prior_specs(
    registry: dict[str, Any],
    runtime_config: dict[str, Any],
    *,
    parameter_names: Iterable[str] | None = None,
    prior_sd_scale: float | None = None,
    beta_prior_log_sd: float | None = None,
    reporting_prior_log_sd: float | None = None,
    ve_prior_sd: float | None = None,
    rel_asym_prior_sd: float | None = None,
    fitness_prior_sd: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return selected validated priors from ``bayesian_joint``.

    Country-centred specifications replace their configured placeholder centre
    with the accepted calibrated runtime value before distribution validation.
    Width overrides are retained for short sampler pilots, but they modify the
    same specification used by both inverse-CDF proposal draws and log-density
    evaluation. ``parameter_names`` makes each runner's consumption boundary
    explicit: unselected specifications are neither resolved nor validated as
    distributions by that runner.
    """

    validate_uncertainty_registry_schema(registry)
    expected_names = set(BAYESIAN_PARAMETER_NAMES)
    if parameter_names is None:
        selected_names = BAYESIAN_PARAMETER_NAMES
    else:
        if isinstance(parameter_names, (str, bytes)):
            raise ValueError("parameter_names must be an iterable of parameter names")
        selected_names = tuple(str(name) for name in parameter_names)
        if not selected_names:
            raise ValueError("parameter_names must select at least one Bayesian prior")
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("parameter_names contains duplicate Bayesian prior names")
        unknown = sorted(set(selected_names) - expected_names)
        if unknown:
            raise ValueError(f"parameter_names contains unknown Bayesian priors: {unknown}")

    selected_set = set(selected_names)
    require_exact_block = selected_set == expected_names
    block = registry.get("bayesian_joint", {})
    configured = block.get("parameters", {}) if isinstance(block, dict) else {}
    actual_names = set(configured) if isinstance(configured, dict) else set()
    missing = sorted(selected_set - actual_names)
    extra = sorted(actual_names - expected_names) if require_exact_block else []
    if missing or extra:
        contract = (
            "the implemented nine-parameter contract"
            if require_exact_block
            else "this runner's selected prior contract"
        )
        match_phrase = "must exactly match" if require_exact_block else "must match"
        raise ValueError(
            f"parameter_distributions.bayesian_joint.parameters {match_phrase} "
            f"{contract}; missing={missing}, extra={extra}"
        )

    specs = {name: deepcopy(dict(configured[name])) for name in selected_names}
    for name, spec in specs.items():
        expected_path = BAYESIAN_PARAMETER_PATHS[name]
        if spec.get("path") != expected_path:
            raise ValueError(
                f"Bayesian prior {name!r} declares path={spec.get('path')!r}, but "
                f"its implemented path is {expected_path!r}"
            )
        expected_scope = BAYESIAN_PARAMETER_TIME_SCOPES[name]
        if spec.get("time_scope") != expected_scope:
            raise ValueError(
                f"Bayesian prior {name!r} declares time_scope="
                f"{spec.get('time_scope')!r}, but its implemented scope is "
                f"{expected_scope!r}"
            )
        declared_consumers = spec.get("consumers")
        expected_consumers = BAYESIAN_PARAMETER_CONSUMERS[name]
        if not isinstance(declared_consumers, list) or tuple(
            declared_consumers
        ) != expected_consumers:
            raise ValueError(
                f"Bayesian prior {name!r} declares consumers="
                f"{declared_consumers!r}, but its implemented consumers are "
                f"{list(expected_consumers)!r}"
            )
        if bool(spec.get("center_on_calibrated", False)):
            center = _path_value(runtime_config, str(spec.get("path", "")))
            distribution = str(spec.get("distribution", "")).lower()
            if distribution == "truncated_lognormal":
                spec["median"] = center
            elif distribution == "truncated_normal":
                spec["mean"] = center
            else:
                raise ValueError(
                    f"Bayesian prior {name!r} uses center_on_calibrated with unsupported "
                    f"distribution {distribution!r}"
                )

    if prior_sd_scale is not None:
        scale = float(prior_sd_scale)
        if scale <= 0.0:
            raise ValueError("--prior-sd-scale must be positive")
        for spec in specs.values():
            _scale_width(spec, scale)

    explicit_widths = {
        "beta_S": beta_prior_log_sd,
        "reporting_multiplier": reporting_prior_log_sd,
        "relative_infectiousness_asymptomatic": rel_asym_prior_sd,
        "fitness_R": fitness_prior_sd,
    }
    for name, width in explicit_widths.items():
        if width is None:
            continue
        if name not in selected_set:
            raise ValueError(
                f"Prior width override for {name} cannot be applied because that "
                "parameter is outside this runner's registry-consumption contract"
            )
        width = float(width)
        if width <= 0.0:
            raise ValueError(f"Prior width override for {name} must be positive")
        field = "log_sd" if "log_sd" in specs[name] else "sd"
        specs[name][field] = width
    if ve_prior_sd is not None:
        missing_ve_names = sorted({"VE_sus", "VE_inf"} - selected_set)
        if missing_ve_names:
            raise ValueError(
                "--ve-prior-sd cannot be applied because these parameters are outside "
                f"this runner's registry-consumption contract: {missing_ve_names}"
            )
        width = float(ve_prior_sd)
        if width <= 0.0:
            raise ValueError("--ve-prior-sd must be positive")
        specs["VE_sus"]["sd"] = width
        specs["VE_inf"]["sd"] = width

    return {
        name: validate_distribution_spec(spec, context=f"bayesian_joint.parameters.{name}")
        for name, spec in specs.items()
    }
