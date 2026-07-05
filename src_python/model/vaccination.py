from __future__ import annotations

import numpy as np


def vaccine_susceptibility(ve_sus: float, *, relative_effect: float = 1.0) -> float:
    return max(0.0, 1.0 - max(0.0, relative_effect) * ve_sus)


def origin_relative_effect(
    origin: str,
    *,
    waned_relative_effect: float = 0.35,
    maternal_relative_effect: float = 0.75,
    dose1_relative_effect: float = 0.45,
    dose2_relative_effect: float = 0.75,
) -> float:
    waned = float(np.clip(waned_relative_effect, 0.0, 1.0))
    if origin == "maternal":
        return float(np.clip(maternal_relative_effect, 0.0, 1.0))
    if origin.startswith("dose1_"):
        dose_effect = float(np.clip(dose1_relative_effect, 0.0, 1.0))
        return dose_effect * (waned if origin.endswith("_waned") else 1.0)
    if origin.startswith("dose2_"):
        dose_effect = float(np.clip(dose2_relative_effect, 0.0, 1.0))
        return dose_effect * (waned if origin.endswith("_waned") else 1.0)
    if origin == "recent":
        return 1.0
    if origin == "waned":
        return waned
    return 0.0


def origin_symptomatic_probability(
    base_probability: np.ndarray,
    ve_sym: float,
    origin: str,
    *,
    waned_relative_effect: float = 0.35,
    maternal_relative_effect: float = 0.75,
    dose1_relative_effect: float = 0.45,
    dose2_relative_effect: float = 0.75,
) -> np.ndarray:
    effect = origin_relative_effect(
        origin,
        waned_relative_effect=waned_relative_effect,
        maternal_relative_effect=maternal_relative_effect,
        dose1_relative_effect=dose1_relative_effect,
        dose2_relative_effect=dose2_relative_effect,
    )
    return np.clip(np.asarray(base_probability, dtype=float) * (1.0 - ve_sym * effect), 0.0, 1.0)


def origin_infectiousness_multiplier(
    ve_inf: float,
    origin: str,
    *,
    waned_relative_effect: float = 0.35,
    maternal_relative_effect: float = 0.75,
    dose1_relative_effect: float = 0.45,
    dose2_relative_effect: float = 0.75,
) -> float:
    effect = origin_relative_effect(
        origin,
        waned_relative_effect=waned_relative_effect,
        maternal_relative_effect=maternal_relative_effect,
        dose1_relative_effect=dose1_relative_effect,
        dose2_relative_effect=dose2_relative_effect,
    )
    return float(np.clip(1.0 - ve_inf * effect, 0.0, 1.0))


def origin_recovery_rate_multiplier(
    ve_dur: float,
    origin: str,
    *,
    waned_relative_effect: float = 0.35,
    maternal_relative_effect: float = 0.75,
    dose1_relative_effect: float = 0.45,
    dose2_relative_effect: float = 0.75,
) -> float:
    effect = origin_relative_effect(
        origin,
        waned_relative_effect=waned_relative_effect,
        maternal_relative_effect=maternal_relative_effect,
        dose1_relative_effect=dose1_relative_effect,
        dose2_relative_effect=dose2_relative_effect,
    )
    duration_multiplier = max(0.05, 1.0 - ve_dur * effect)
    return 1.0 / duration_multiplier


def origin_is_vaccine_dose(origin: str) -> bool:
    return origin.startswith("dose") or origin in {"recent", "waned"}


def origin_is_waned(origin: str) -> bool:
    return origin.endswith("_waned") or origin == "waned"


def origin_dose_category(origin: str) -> str:
    if origin.startswith("dose1_"):
        return "dose1"
    if origin.startswith("dose2_"):
        return "dose2"
    if origin in {"recent", "waned"}:
        return "dose3plus"
    return origin


def default_initial_origin_distribution(age: str) -> dict[str, float]:
    if age == "infant_0_2m":
        return {"maternal": 1.0}
    if age == "infant_3_11m":
        return {"dose1_recent": 0.25, "dose2_recent": 0.35, "recent": 0.40}
    if age == "child_1_4y":
        return {"dose2_waned": 0.10, "recent": 0.55, "waned": 0.35}
    if age == "child_5_9y":
        return {"dose2_waned": 0.10, "recent": 0.40, "waned": 0.50}
    if age == "adolescent_10_17y":
        return {"dose2_waned": 0.10, "recent": 0.25, "waned": 0.65}
    if age == "young_adult_18_39y":
        return {"dose2_waned": 0.10, "recent": 0.10, "waned": 0.80}
    if age == "middle_adult_40_64y":
        return {"dose2_waned": 0.05, "recent": 0.05, "waned": 0.90}
    if age == "elderly_65plus":
        return {"dose2_waned": 0.05, "recent": 0.03, "waned": 0.92}
    return {"dose2_waned": 0.10, "recent": 0.10, "waned": 0.80}


def default_routine_target_origin_distribution(age: str) -> dict[str, float]:
    if age == "infant_0_2m":
        return {}
    if age == "infant_3_11m":
        return {"dose1_recent": 0.25, "dose2_recent": 0.35, "recent": 0.40}
    if age == "child_1_4y":
        return {"recent": 0.65, "waned": 0.35}
    if age == "child_5_9y":
        return {"recent": 0.50, "waned": 0.50}
    if age == "adolescent_10_17y":
        return {"recent": 0.30, "waned": 0.70}
    if age == "young_adult_18_39y":
        return {"recent": 0.10, "waned": 0.90}
    if age == "middle_adult_40_64y":
        return {"recent": 0.05, "waned": 0.95}
    if age == "elderly_65plus":
        return {"recent": 0.03, "waned": 0.97}
    return {"recent": 0.10, "waned": 0.90}


def vaccinated_infection_share(
    susceptible: np.ndarray,
    vaccinated_recent: np.ndarray,
    ve_sus: float,
    vaccinated_waned: np.ndarray | None = None,
    *,
    waned_relative_effect: float = 0.35,
) -> np.ndarray:
    recent = np.maximum(vaccinated_recent, 0.0) * vaccine_susceptibility(ve_sus)
    if vaccinated_waned is None:
        effective_waned = np.zeros_like(recent, dtype=float)
        protected_waned = effective_waned
    else:
        effective_waned = np.maximum(vaccinated_waned, 0.0) * vaccine_susceptibility(
            ve_sus,
            relative_effect=waned_relative_effect,
        )
        protected_waned = effective_waned * np.clip(waned_relative_effect, 0.0, 1.0)
    effective_total = np.maximum(susceptible, 0.0) + recent + effective_waned
    protected_equivalent = recent + protected_waned
    return np.divide(
        protected_equivalent,
        effective_total,
        out=np.zeros_like(effective_total, dtype=float),
        where=effective_total > 0,
    )


def symptomatic_probability(
    base_probability: np.ndarray,
    infection_from_susceptible: np.ndarray,
    infection_from_vaccinated: np.ndarray,
    ve_sym: float,
    infection_from_waned: np.ndarray | None = None,
    *,
    waned_relative_effect: float = 0.35,
) -> np.ndarray:
    unvaccinated_sym = np.clip(base_probability, 0.0, 1.0)
    vaccinated_sym = np.clip(base_probability * (1.0 - ve_sym), 0.0, 1.0)
    if infection_from_waned is None:
        total = infection_from_susceptible + infection_from_vaccinated
        weighted = infection_from_susceptible * unvaccinated_sym + infection_from_vaccinated * vaccinated_sym
    else:
        waned_effect = np.clip(waned_relative_effect, 0.0, 1.0)
        waned_sym = np.clip(base_probability * (1.0 - ve_sym * waned_effect), 0.0, 1.0)
        total = infection_from_susceptible + infection_from_vaccinated + infection_from_waned
        weighted = (
            infection_from_susceptible * unvaccinated_sym
            + infection_from_vaccinated * vaccinated_sym
            + infection_from_waned * waned_sym
        )
    fallback = 0.5 * (unvaccinated_sym + vaccinated_sym)
    return np.divide(weighted, total, out=fallback.copy(), where=total > 0)


def infectiousness_multiplier(vaccine_origin_share: np.ndarray, ve_inf: float) -> np.ndarray:
    return np.clip(1.0 - vaccine_origin_share * ve_inf, 0.0, 1.0)


def recovery_rate_multiplier(vaccine_origin_share: np.ndarray, ve_dur: float) -> np.ndarray:
    duration_multiplier = np.clip(1.0 - vaccine_origin_share * ve_dur, 0.05, None)
    return 1.0 / duration_multiplier
