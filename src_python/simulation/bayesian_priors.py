"""Resolve the central distribution registry for country-level Bayesian runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src_python.simulation.parameter_distributions import validate_distribution_spec


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
    prior_sd_scale: float | None = None,
    beta_prior_log_sd: float | None = None,
    reporting_prior_log_sd: float | None = None,
    ve_prior_sd: float | None = None,
    rel_asym_prior_sd: float | None = None,
    fitness_prior_sd: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return validated country-resolved priors from ``bayesian_joint``.

    Country-centred specifications replace their configured placeholder centre
    with the accepted calibrated runtime value before distribution validation.
    Width overrides are retained for short sampler pilots, but they modify the
    same specification used by both inverse-CDF proposal draws and log-density
    evaluation.
    """

    block = registry.get("bayesian_joint", {})
    configured = block.get("parameters", {}) if isinstance(block, dict) else {}
    missing = [name for name in BAYESIAN_PARAMETER_NAMES if name not in configured]
    if missing:
        raise ValueError(
            "parameter_distributions.bayesian_joint.parameters is missing: "
            + ", ".join(missing)
        )

    specs = {name: deepcopy(dict(configured[name])) for name in BAYESIAN_PARAMETER_NAMES}
    for name, spec in specs.items():
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
        width = float(width)
        if width <= 0.0:
            raise ValueError(f"Prior width override for {name} must be positive")
        field = "log_sd" if "log_sd" in specs[name] else "sd"
        specs[name][field] = width
    if ve_prior_sd is not None:
        width = float(ve_prior_sd)
        if width <= 0.0:
            raise ValueError("--ve-prior-sd must be positive")
        specs["VE_sus"]["sd"] = width
        specs["VE_inf"]["sd"] = width

    return {
        name: validate_distribution_spec(spec, context=f"bayesian_joint.parameters.{name}")
        for name, spec in specs.items()
    }

