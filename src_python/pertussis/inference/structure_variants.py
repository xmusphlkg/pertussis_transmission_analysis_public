"""Structural alternatives compared by information criteria in M3.

Each variant changes one assumption of the baseline model:

* ``s5_school_detection``: from ``school_detection_start_year`` the reporting
  logits of the 5-9 and 10-19 bands are shifted by an *estimated* amount
  (one extra observation parameter), representing increased testing of
  school-age children;
* ``exponential_vaccine_waning``: wP, aP and Tdap protection wane
  exponentially instead of through two Erlang stages, with the same mean
  duration -- the memoryless alternative to cohort-specific waning;
* ``no_natural_immunity_boosting``: re-exposure does not boost waned natural
  immunity (``boosting_kappa`` fixed at zero).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureVariant:
    name: str
    description: str
    vaccine_waning_shape: str = "erlang2"
    disable_natural_boosting: bool = False
    school_detection_start_year: int | None = None


VARIANTS = {
    "baseline": StructureVariant("baseline", "Erlang-2 vaccine and natural immunity; stationary age-specific detection"),
    "s5_school_detection": StructureVariant(
        "s5_school_detection", "Estimated shift in school-age (5-19) reporting logits from 2023",
        school_detection_start_year=2023,
    ),
    "exponential_vaccine_waning": StructureVariant(
        "exponential_vaccine_waning", "Exponential waning of wP, aP and Tdap protection",
        vaccine_waning_shape="exponential",
    ),
    "no_natural_immunity_boosting": StructureVariant(
        "no_natural_immunity_boosting", "Natural immunity is not boosted by re-exposure",
        disable_natural_boosting=True,
    ),
}

ALIASES = {
    "s5": "s5_school_detection",
    "school_detection": "s5_school_detection",
    "no_boosting": "no_natural_immunity_boosting",
    "exponential": "exponential_vaccine_waning",
    "exponential_waning": "exponential_vaccine_waning",
}


def get_variant(name: str | None) -> StructureVariant:
    key = ALIASES.get((name or "baseline").lower(), (name or "baseline").lower())
    if key not in VARIANTS:
        raise ValueError(f"unknown structural variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[key]


def aic(data_nll: float, n_parameters: int) -> float:
    """AIC from the data negative log likelihood at the fitted point."""

    return float(2.0 * data_nll + 2.0 * n_parameters)
