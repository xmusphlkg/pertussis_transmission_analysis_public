"""Prospective resistance-management implementation transform.

This module is intentionally independent of the Figure 2b programme-ranking
runner. It interpolates the configured resistance-guided pathway from the
matching current-practice policy, applies the sampled PEP-reach multiplier to
the guided policy only, and leaves the attached pre-policy history unchanged.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


RESISTANCE_MANAGEMENT_PARAMETER_NAMES = (
    "resistance_management_uptake",
    "resistance_management_pep_reach_multiplier",
)
PROSPECTIVE_IMPLEMENTATION = "prospective_implementation"
RESISTANCE_MANAGEMENT_STRATEGIES = ("resistance_guided_treatment",)


def _clip_probability(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _interpolate(base: float, target: float, fraction: float) -> float:
    fraction = _clip_probability(fraction)
    return float(base + fraction * (target - base))


def _finite_nonnegative(value: Any, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}")
    return numeric


def _probability(value: Any, *, name: str) -> float:
    numeric = _finite_nonnegative(value, name=name)
    if numeric > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {value!r}")
    return numeric


def _apply_guided_management_uptake(
    config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    uptake: float,
    pep_restored: bool,
) -> None:
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
    if pep_restored:
        config["PEP"]["effectiveness_resistant"] = _clip_probability(
            _interpolate(
                float(current_config["PEP"]["effectiveness_resistant"]),
                float(config["PEP"]["effectiveness_resistant"]),
                uptake,
            )
        )
    else:
        config["PEP"]["effectiveness_resistant"] = _clip_probability(
            float(current_config["PEP"]["effectiveness_resistant"])
        )


def apply_resistance_management_sample(
    config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    strategy: str,
    sample: dict[str, Any],
    pep_restored: bool,
    parameter_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the two implementation draws to the prospective policy only.

    ``pep_restored`` is deliberately a deterministic structural stratum rather
    than a sampled Bernoulli parameter. The same continuous draw can therefore
    be paired across restored and not-restored scenarios without implying a
    probability for either structural assumption.
    """

    if strategy not in RESISTANCE_MANAGEMENT_STRATEGIES:
        raise ValueError(
            f"Independent resistance-management PSA does not admit strategy {strategy!r}; "
            f"expected one of {list(RESISTANCE_MANAGEMENT_STRATEGIES)}"
        )
    missing = [
        name for name in RESISTANCE_MANAGEMENT_PARAMETER_NAMES if name not in sample
    ]
    if missing:
        raise ValueError(
            f"Resistance-management PSA sample is missing parameters: {missing}"
        )

    if parameter_specs is None:
        scopes = {
            name: PROSPECTIVE_IMPLEMENTATION
            for name in RESISTANCE_MANAGEMENT_PARAMETER_NAMES
        }
    else:
        if tuple(parameter_specs) != RESISTANCE_MANAGEMENT_PARAMETER_NAMES:
            raise ValueError(
                "Resistance-management parameter registry must exactly match its "
                f"implemented contract; expected={list(RESISTANCE_MANAGEMENT_PARAMETER_NAMES)}, "
                f"actual={list(parameter_specs)}"
            )
        scopes = {}
        for name in RESISTANCE_MANAGEMENT_PARAMETER_NAMES:
            scope = str(parameter_specs[name].get("time_scope", "")).strip().lower()
            if scope != PROSPECTIVE_IMPLEMENTATION:
                raise ValueError(
                    f"Resistance-management parameter {name!r} must declare "
                    f"time_scope={PROSPECTIVE_IMPLEMENTATION!r}"
                )
            scopes[name] = scope

    if not isinstance(pep_restored, bool):
        raise TypeError("pep_restored must be an explicit bool structural stratum")

    uptake = _probability(
        sample["resistance_management_uptake"],
        name="resistance_management_uptake",
    )
    reach_multiplier = _finite_nonnegative(
        sample["resistance_management_pep_reach_multiplier"],
        name="resistance_management_pep_reach_multiplier",
    )

    out = deepcopy(config)
    _apply_guided_management_uptake(
        out,
        current_config,
        uptake=uptake,
        pep_restored=pep_restored,
    )
    out["PEP"]["coverage_household_contacts"] = _clip_probability(
        float(current_config["PEP"]["coverage_household_contacts"])
        * reach_multiplier
    )
    metadata = out.setdefault("metadata", {})
    metadata["resistance_management_parameter_time_scopes"] = dict(scopes)
    metadata["resistance_management_pep_restoration"] = (
        "restored" if pep_restored else "not_restored"
    )
    if "psa_sample_id" in sample:
        metadata["resistance_management_psa_sample_id"] = int(sample["psa_sample_id"])
    return out
