"""Shared programme-scenario and paired-draw helpers.

This module contains mechanism-level utilities used by multiple simulation
runners.  It deliberately does not assign a publication role or an interval
interpretation; each caller supplies its own analysis stem and statistical
contract.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    apply_intervention_definition,
    make_intervention_config,
)
from src_python.simulation.run_bayesian_uncertainty import (
    _apply_sample,
    _sample_columns,
)
from src_python.simulation.run_routine_timeliness_sensitivity import (
    _apply_timeliness,
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

INTERVENTION_UNCERTAINTY_DEFAULTS = {
    "adolescent_coverage_floor": {"low": 0.75, "mode": 0.90, "high": 0.98},
    "maternal_coverage_floor": {"low": 0.55, "mode": 0.75, "high": 0.90},
    "young_adult_coverage_floor": {"low": 0.35, "mode": 0.55, "high": 0.75},
    "contact_reduction_fraction": {"low": 0.05, "mode": 0.15, "high": 0.30},
    "targeted_pep_coverage": {"low": 0.05, "mode": 0.45, "high": 0.60},
    "maternal_protection_duration_days": {
        "low": 90.0,
        "mode": 180.0,
        "high": 270.0,
    },
}
INTERVENTION_UNCERTAINTY_PREFIX = "intervention_uncertainty_"
FIXED_PROGRAMME_REFERENCE_SOURCE = (
    "configs.interventions_formal_programme_definitions"
)

PROGRAMME_STRATEGIES = (
    "current",
    "timeliness_only",
    "adolescent_booster",
    "pregnancy_tdap_scaleup",
    "cocooning_adjunct",
    "maternal_immunization",
    "targeted_pep_high_risk",
)

PROGRAMME_STRATEGY_LABELS = {
    "current": "Current practice",
    "timeliness_only": "Routine timeliness",
    "adolescent_booster": "Adolescent booster",
    "pregnancy_tdap_scaleup": "Pregnancy Tdap",
    "cocooning_adjunct": "Close-contact adjunct",
    "maternal_immunization": "Infant exposure",
    "targeted_pep_high_risk": "Targeted PEP",
}


def posterior_stem_from_sample_path(path: str | Path) -> str | None:
    """Infer a run-metadata stem from a conventional sample-file name."""

    name = Path(path).name
    if name == "bayesian_posterior_samples.parquet":
        return "bayesian_uncertainty"
    suffix = "_posterior_samples.parquet"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return None


def _required_programme_value(
    interventions: dict[str, Any],
    strategy: str,
    *path: str,
) -> float:
    try:
        value: Any = interventions[strategy]
        for key in path:
            value = value[key]
    except (KeyError, TypeError) as exc:
        location = ".".join(("interventions", strategy, *path))
        raise KeyError(
            f"Formal programme definition is missing required value: {location}"
        ) from exc
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        location = ".".join(("interventions", strategy, *path))
        raise ValueError(
            f"Formal programme definition must be numeric: {location}={value!r}"
        ) from exc
    if not np.isfinite(resolved):
        location = ".".join(("interventions", strategy, *path))
        raise ValueError(
            f"Formal programme definition must be finite: {location}={value!r}"
        )
    return resolved


def _require_matching_programme_values(
    *,
    name: str,
    first: float,
    second: float,
    first_source: str,
    second_source: str,
) -> None:
    if not np.isclose(first, second, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Formal programme definitions disagree for {name}: "
            f"{first_source}={first} versus {second_source}={second}"
        )


def fixed_programme_reference_inputs(
    configs: dict[str, Any],
) -> dict[str, float]:
    """Read fixed programme inputs from the formal intervention definitions.

    Duplicated pregnancy/maternal and cocooning/maternal values are checked for
    equality so the fixed estimation-CI reference cannot silently combine
    inconsistent programme definitions.  Legacy stochastic-prior modes are not
    consulted by this function.
    """

    interventions = configs.get("interventions")
    if not isinstance(interventions, dict):
        raise KeyError("configs.interventions formal programme definitions are required")

    adolescent_coverage = _required_programme_value(
        interventions,
        "adolescent_booster",
        "coverage_min_updates",
        "adolescent_10_17y",
    )
    pregnancy_coverage = _required_programme_value(
        interventions,
        "pregnancy_tdap_scaleup",
        "coverage_min_updates",
        "infant_0_2m",
    )
    pregnancy_ve_sus = _required_programme_value(
        interventions,
        "pregnancy_tdap_scaleup",
        "vaccine_overrides",
        "maternal_VE_sus",
    )
    pregnancy_ve_sym = _required_programme_value(
        interventions,
        "pregnancy_tdap_scaleup",
        "vaccine_overrides",
        "maternal_VE_sym",
    )
    pregnancy_duration = _required_programme_value(
        interventions,
        "pregnancy_tdap_scaleup",
        "natural_history_overrides",
        "maternal_protection_duration",
    )
    maternal_coverage = _required_programme_value(
        interventions,
        "maternal_immunization",
        "coverage_min_updates",
        "infant_0_2m",
    )
    maternal_ve_sus = _required_programme_value(
        interventions,
        "maternal_immunization",
        "vaccine_overrides",
        "maternal_VE_sus",
    )
    maternal_ve_sym = _required_programme_value(
        interventions,
        "maternal_immunization",
        "vaccine_overrides",
        "maternal_VE_sym",
    )
    maternal_duration = _required_programme_value(
        interventions,
        "maternal_immunization",
        "natural_history_overrides",
        "maternal_protection_duration",
    )
    cocooning_coverage = _required_programme_value(
        interventions,
        "cocooning_adjunct",
        "coverage_min_updates",
        "young_adult_18_39y",
    )
    cocooning_contact = _required_programme_value(
        interventions,
        "cocooning_adjunct",
        "contact_matrix_reduction",
        "reduction_fraction",
    )
    maternal_cocooning_coverage = _required_programme_value(
        interventions,
        "maternal_immunization",
        "coverage_min_updates",
        "young_adult_18_39y",
    )
    maternal_contact = _required_programme_value(
        interventions,
        "maternal_immunization",
        "contact_matrix_reduction",
        "reduction_fraction",
    )
    targeted_pep_coverage = _required_programme_value(
        interventions,
        "targeted_pep_high_risk",
        "pep_updates",
        "coverage_household_contacts",
    )

    matched = (
        (
            "maternal coverage",
            pregnancy_coverage,
            maternal_coverage,
            "pregnancy_tdap_scaleup.coverage_min_updates.infant_0_2m",
            "maternal_immunization.coverage_min_updates.infant_0_2m",
        ),
        (
            "maternal VE_sus",
            pregnancy_ve_sus,
            maternal_ve_sus,
            "pregnancy_tdap_scaleup.vaccine_overrides.maternal_VE_sus",
            "maternal_immunization.vaccine_overrides.maternal_VE_sus",
        ),
        (
            "maternal VE_sym",
            pregnancy_ve_sym,
            maternal_ve_sym,
            "pregnancy_tdap_scaleup.vaccine_overrides.maternal_VE_sym",
            "maternal_immunization.vaccine_overrides.maternal_VE_sym",
        ),
        (
            "maternal protection duration",
            pregnancy_duration,
            maternal_duration,
            "pregnancy_tdap_scaleup.natural_history_overrides.maternal_protection_duration",
            "maternal_immunization.natural_history_overrides.maternal_protection_duration",
        ),
        (
            "young-adult coverage",
            cocooning_coverage,
            maternal_cocooning_coverage,
            "cocooning_adjunct.coverage_min_updates.young_adult_18_39y",
            "maternal_immunization.coverage_min_updates.young_adult_18_39y",
        ),
        (
            "contact reduction",
            cocooning_contact,
            maternal_contact,
            "cocooning_adjunct.contact_matrix_reduction.reduction_fraction",
            "maternal_immunization.contact_matrix_reduction.reduction_fraction",
        ),
    )
    for name, first, second, first_source, second_source in matched:
        _require_matching_programme_values(
            name=name,
            first=first,
            second=second,
            first_source=first_source,
            second_source=second_source,
        )

    proportion_values = {
        "adolescent_coverage_floor": adolescent_coverage,
        "maternal_coverage_floor": pregnancy_coverage,
        "young_adult_coverage_floor": cocooning_coverage,
        "contact_reduction_fraction": cocooning_contact,
        "targeted_pep_coverage": targeted_pep_coverage,
        "maternal_VE_sus": pregnancy_ve_sus,
        "maternal_VE_sym": pregnancy_ve_sym,
    }
    invalid = {
        name: value
        for name, value in proportion_values.items()
        if not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(
            "Formal programme probabilities/fractions must lie in [0, 1]: "
            f"{invalid}"
        )
    if pregnancy_duration <= 0.0:
        raise ValueError(
            "Formal maternal_protection_duration must be strictly positive"
        )
    return {
        "adolescent_coverage_floor": adolescent_coverage,
        "maternal_coverage_floor": pregnancy_coverage,
        "young_adult_coverage_floor": cocooning_coverage,
        "contact_reduction_fraction": cocooning_contact,
        "targeted_pep_coverage": targeted_pep_coverage,
        "maternal_protection_duration_days": pregnancy_duration,
        "maternal_VE_sus": pregnancy_ve_sus,
        "maternal_VE_sym": pregnancy_ve_sym,
    }


def _triangular_parameters(
    settings: dict[str, Any], name: str
) -> dict[str, float]:
    configured = settings.get("intervention_priors", {}).get(name, {})
    values = {**INTERVENTION_UNCERTAINTY_DEFAULTS[name], **configured}
    low = float(values["low"])
    mode = float(values["mode"])
    high = float(values["high"])
    if not low <= mode <= high:
        raise ValueError(
            f"Invalid triangular prior for {name}: low <= mode <= high is required"
        )
    return {"low": low, "mode": mode, "high": high}


def _beta_ab(mean: float, sd: float) -> tuple[float, float]:
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    sd = max(float(sd), 1e-6)
    variance = min(sd**2, mean * (1.0 - mean) * 0.95)
    common = mean * (1.0 - mean) / variance - 1.0
    return max(mean * common, 1e-3), max((1.0 - mean) * common, 1e-3)


def _draw_beta_prior(
    rng: np.random.Generator, prior: dict[str, Any]
) -> float:
    alpha, beta = _beta_ab(float(prior["mean"]), float(prior["sd"]))
    return float(rng.beta(alpha, beta))


def draw_intervention_inputs(
    configs: dict[str, Any],
    rng: np.random.Generator,
    *,
    stochastic: bool,
) -> dict[str, float]:
    """Resolve programme-definition inputs at their modes or draw them.

    The nested configuration location is retained for compatibility with the
    existing input schema.  Calling runners determine whether these values are
    fixed references or stochastic research inputs.
    """

    uncertainty = configs["baseline"].get("bayesian_uncertainty", {})
    settings = uncertainty.get("figure2c_joint_credible_interval", {})
    priors = uncertainty.get("priors", {})

    def triangular(name: str) -> float:
        parameters = _triangular_parameters(settings, name)
        if not stochastic:
            return float(parameters["mode"])
        return float(
            rng.triangular(
                parameters["low"], parameters["mode"], parameters["high"]
            )
        )

    maternal_sus_prior = priors.get(
        "maternal_VE_sus", {"mean": 0.55, "sd": 0.12}
    )
    maternal_sym_prior = priors.get(
        "maternal_VE_sym", {"mean": 0.92, "sd": 0.05}
    )
    return {
        "adolescent_coverage_floor": triangular("adolescent_coverage_floor"),
        "maternal_coverage_floor": triangular("maternal_coverage_floor"),
        "young_adult_coverage_floor": triangular("young_adult_coverage_floor"),
        "contact_reduction_fraction": triangular("contact_reduction_fraction"),
        "targeted_pep_coverage": triangular("targeted_pep_coverage"),
        "maternal_protection_duration_days": triangular(
            "maternal_protection_duration_days"
        ),
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


def attach_intervention_inputs(
    selected_samples: pd.DataFrame,
    configs: dict[str, Any],
    *,
    seed: int,
    enabled: bool,
) -> pd.DataFrame:
    out = selected_samples.copy()
    rng = np.random.default_rng(seed)
    draws = [
        draw_intervention_inputs(configs, rng, stochastic=enabled)
        for _ in range(len(out))
    ]
    for key in draws[0].keys() if draws else ():
        out[f"{INTERVENTION_UNCERTAINTY_PREFIX}{key}"] = [
            float(draw[key]) for draw in draws
        ]
    return out


def apply_intervention_inputs(
    intervention: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    out = deepcopy(intervention)
    coverage = (
        out.setdefault("coverage_min_updates", {})
        if "coverage_min_updates" in out
        else None
    )
    if coverage is not None:
        if "adolescent_10_17y" in coverage:
            coverage["adolescent_10_17y"] = float(
                row[
                    f"{INTERVENTION_UNCERTAINTY_PREFIX}adolescent_coverage_floor"
                ]
            )
        if "infant_0_2m" in coverage:
            coverage["infant_0_2m"] = float(
                row[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_coverage_floor"]
            )
        if "young_adult_18_39y" in coverage:
            coverage["young_adult_18_39y"] = float(
                row[
                    f"{INTERVENTION_UNCERTAINTY_PREFIX}young_adult_coverage_floor"
                ]
            )

    vaccine = (
        out.setdefault("vaccine_overrides", {})
        if "vaccine_overrides" in out
        else None
    )
    if vaccine is not None:
        if "maternal_VE_sus" in vaccine:
            vaccine["maternal_VE_sus"] = float(
                row[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sus"]
            )
        if "maternal_VE_sym" in vaccine:
            vaccine["maternal_VE_sym"] = float(
                row[f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_VE_sym"]
            )

    natural_history = (
        out.setdefault("natural_history_overrides", {})
        if "natural_history_overrides" in out
        else None
    )
    if natural_history is not None and "maternal_protection_duration" in natural_history:
        natural_history["maternal_protection_duration"] = float(
            row[
                f"{INTERVENTION_UNCERTAINTY_PREFIX}maternal_protection_duration_days"
            ]
        )

    pep = out.setdefault("pep_updates", {}) if "pep_updates" in out else None
    if pep is not None and "coverage_household_contacts" in pep:
        pep["coverage_household_contacts"] = float(
            row[f"{INTERVENTION_UNCERTAINTY_PREFIX}targeted_pep_coverage"]
        )

    contact = (
        out.setdefault("contact_matrix_reduction", {})
        if "contact_matrix_reduction" in out
        else None
    )
    if contact is not None and "reduction_fraction" in contact:
        contact["reduction_fraction"] = float(
            row[f"{INTERVENTION_UNCERTAINTY_PREFIX}contact_reduction_fraction"]
        )
    return out


def _sample_from_row(row: pd.Series) -> dict[str, float]:
    required = _sample_columns()
    missing = [
        name
        for name in required
        if name != "reporting_trend_end_multiplier" and name not in row.index
    ]
    if missing:
        raise KeyError(
            "Programme sample row is missing required columns: "
            + ", ".join(missing)
        )
    sample: dict[str, float] = {}
    for name in required:
        if name in row.index and pd.notna(row[name]):
            sample[name] = float(row[name])
        elif name == "reporting_trend_end_multiplier":
            sample[name] = 1.0
        else:
            raise ValueError(
                f"Programme sample value is missing for required column: {name}"
            )
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
            "Paired programme scenarios require strategies without "
            f"vaccine_scenario swaps; got {strategy!r}."
        )
    intervention = apply_intervention_inputs(intervention, sample_row)
    return apply_intervention_definition(sampled_current, intervention), vaccine_name


def build_programme_scenarios(
    configs: dict[str, Any],
    selected_samples: pd.DataFrame,
    *,
    strategies: tuple[str, ...],
    analysis: str = "programme_uncertainty",
) -> list[dict[str, Any]]:
    """Build matched programme scenarios without assigning interval semantics."""

    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get(
        "baseline_resistance_scenario", "country_timeline"
    )
    base_configs: dict[str, tuple[dict[str, Any], str]] = {
        country: make_intervention_config("current", country_profile=country)
        for country in sorted(selected_samples["country"].astype(str).unique())
    }
    for row in selected_samples.to_dict(orient="records"):
        country = str(row["country"])
        draw_id = int(row["posterior_draw"])
        sample = _sample_from_row(pd.Series(row))
        base_config, current_vaccine_name = base_configs[country]
        sampled_current = _apply_sample(base_config, sample)
        sample_metadata = {
            f"posterior_{parameter}": float(sample[parameter])
            for parameter in PROPAGATED_UNCERTAINTY_PARAMETERS
            if parameter in sample
        }
        sample_metadata.update(
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
                "strategy_label": PROGRAMME_STRATEGY_LABELS.get(strategy, strategy),
                "posterior_draw": draw_id,
                "posterior_chain": (
                    int(row["chain"])
                    if "chain" in row and pd.notna(row["chain"])
                    else -1
                ),
                "posterior_source_draw": (
                    int(row["draw"])
                    if "draw" in row and pd.notna(row["draw"])
                    else -1
                ),
                "posterior_log_prob": (
                    float(row["posterior_log_prob"])
                    if "posterior_log_prob" in row
                    and pd.notna(row["posterior_log_prob"])
                    else np.nan
                ),
            } | sample_metadata | intervention_metadata
            if "structural_draw_id" in row and pd.notna(row["structural_draw_id"]):
                metadata["structural_draw_id"] = int(row["structural_draw_id"])
            if "inference_structure" in row and pd.notna(row["inference_structure"]):
                metadata["inference_structure"] = str(row["inference_structure"])
            scenarios.append(
                {
                    "config": config,
                    "analysis": str(analysis),
                    "scenario": f"{country}_draw_{draw_id:03d}_{strategy}",
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": strategy,
                    "metadata": metadata,
                }
            )
    return scenarios


def pair_programme_draws(summary: pd.DataFrame) -> pd.DataFrame:
    """Pair each programme outcome with the matched current-practice draw."""

    data = summary.copy()
    data["posterior_draw"] = pd.to_numeric(
        data["posterior_draw"], errors="raise"
    ).astype(int)
    data["strategy"] = data["strategy"].astype(str)

    pair_keys = ["country", "posterior_draw"]
    if "structural_draw_id" in data.columns:
        structural_ids = pd.to_numeric(data["structural_draw_id"], errors="coerce")
        if structural_ids.isna().any() or not np.equal(
            structural_ids, np.floor(structural_ids)
        ).all():
            raise ValueError(
                "Scenario summaries contain invalid structural_draw_id values"
            )
        data["structural_draw_id"] = structural_ids.astype(np.int64)
        draw_alignment = data.groupby("posterior_draw")[
            "structural_draw_id"
        ].nunique()
        if not draw_alignment.eq(1).all():
            bad_draws = (
                draw_alignment.index[~draw_alignment.eq(1)]
                .astype(int)
                .tolist()[:10]
            )
            raise ValueError(
                "Scenario summaries lost cross-country structural pairing for "
                f"draws: {bad_draws}"
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
    paired = intervention.merge(
        current, on=pair_keys, how="left", validate="many_to_one"
    )
    if paired["current_rate"].isna().any():
        missing = paired.loc[
            paired["current_rate"].isna(), ["country", "posterior_draw"]
        ].drop_duplicates()
        raise ValueError(
            f"Missing paired current rows for {len(missing)} country-draw combination(s)"
        )

    paired["intervention_rate"] = pd.to_numeric(
        paired[PRIMARY_RATE], errors="coerce"
    )
    paired["intervention_total_cases"] = pd.to_numeric(
        paired[PRIMARY_TOTAL], errors="coerce"
    )
    paired[PRIMARY_REDUCTION] = 1.0 - paired["intervention_rate"] / paired[
        "current_rate"
    ].replace(0, np.nan)
    keep = [
        "country",
        "posterior_draw",
        *(
            ["structural_draw_id"]
            if "structural_draw_id" in paired.columns
            else []
        ),
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
    return (
        paired.loc[:, keep]
        .sort_values(["country", "strategy", "posterior_draw"])
        .reset_index(drop=True)
    )
