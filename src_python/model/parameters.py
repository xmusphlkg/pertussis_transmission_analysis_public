from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from src_python.model.contact_matrix import balance_reciprocity, reciprocity_error


@dataclass(frozen=True)
class PreparedParameters:
    raw: dict[str, Any]
    analysis: str
    scenario: str
    vaccine_scenario: str
    resistance_scenario: str
    intervention: str
    age_groups: tuple[str, ...]
    population: np.ndarray
    vaccine_coverage: np.ndarray
    symptom_probability: np.ndarray
    reporting_rate: np.ndarray
    diagnosis_probability: np.ndarray
    pep_detection_rate: np.ndarray
    contact_matrix: np.ndarray
    vaccine: dict[str, float]
    rates: dict[str, float]
    transmission: dict[str, float]
    treatment: dict[str, Any]
    pep: dict[str, float]
    initial: dict[str, Any]
    immunity_model: dict[str, Any]
    observation_model: dict[str, Any]
    resistance: dict[str, Any]
    demography: dict[str, Any]
    routine_vaccination: dict[str, Any]
    importation: dict[str, Any]
    calendar: dict[str, Any]
    calendar_start_date: date | None
    reporting_time_variation: dict[str, float] = field(default_factory=dict)
    diagnostic_reporting_time_variation: dict[str, Any] = field(default_factory=dict)
    reporting_multiplier: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------------
    # Pre-computed per-origin constant arrays (built once in from_config).
    # These are used by rhs / force_of_infection on every ODE step, so caching
    # them here eliminates ~500k redundant Python function calls per solve.
    # Shape: (n_origins,) for scalar-per-origin; (n_origins, n_age) for age-varying.
    # ---------------------------------------------------------------------------
    # origin_relative_effects[i]  = origin_relative_effect(VACCINE_ORIGINS[i], ...)
    origin_relative_effects: np.ndarray = field(default_factory=lambda: np.empty(0))
    # origin_susceptibility[i]    = vaccine_susceptibility(VE_sus, relative_effect=...)
    origin_susceptibility: np.ndarray = field(default_factory=lambda: np.empty(0))
    # origin_infectiousness[i]    = 1 - VE_inf * relative_effect  (clipped [0,1])
    origin_infectiousness: np.ndarray = field(default_factory=lambda: np.empty(0))
    # origin_symptomatic_prob[i, age] = base_prob[age] * (1 - VE_sym * effect)
    origin_symptomatic_prob: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    # origin_recovery_mult[i]     = 1 / max(0.05, 1 - VE_dur * effect)
    origin_recovery_mult: np.ndarray = field(default_factory=lambda: np.empty(0))

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        analysis: str,
        scenario: str,
        vaccine_scenario: str = "",
        resistance_scenario: str = "",
        intervention: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "PreparedParameters":
        age_records = config["age_groups"]
        age_groups = tuple(record["label"] for record in age_records)
        population = np.array([record["population"] for record in age_records], dtype=float)
        if not np.isfinite(population).all() or np.any(population <= 0.0):
            raise ValueError("Age-group populations must be finite and > 0.")

        vaccine = {key: float(value) for key, value in config.get("vaccine", {}).items() if key.startswith("VE_")}
        if not vaccine:
            vaccine = {"VE_sus": 0.0, "VE_sym": 0.0, "VE_inf": 0.0, "VE_dur": 0.0}
        _validate_probability_values(vaccine.values(), "Vaccine efficacy values")

        coverage = _probability_array(
            [record.get("vaccine_coverage", 0.0) for record in age_records],
            "Age-group vaccine coverage",
        )
        if sum(vaccine.values()) == 0:
            coverage = np.zeros_like(coverage)

        reporting_multiplier = float(config.get("reporting_multiplier", 1.0))
        reporting_base = _probability_array(
            [record.get("reporting_rate", 0.0) for record in age_records],
            "Age-group reporting rates",
        )
        pep_detection = _probability_array(
            [record.get("pep_detection_rate", record.get("reporting_rate", 0.0)) for record in age_records],
            "Age-group PEP detection rates",
        )
        # diagnosis_probability is the fraction of infections that are
        # diagnosed and potentially treated. It defaults to the baseline
        # reporting_rate (before reporting_multiplier) but can be overridden
        # independently via config["diagnosis_probability"]. This decouples
        # the observation model (reporting_rate × reporting_multiplier) from
        # the treatment/resistance selection dynamics in the ODE.
        diagnosis_prob_config = config.get("diagnosis_probability")
        if diagnosis_prob_config is not None:
            diagnosis_probability = _probability_array(
                [diagnosis_prob_config.get(record["label"], record.get("reporting_rate", 0.0))
                 for record in age_records],
                "Age-group diagnosis probabilities",
            )
        else:
            diagnosis_probability = reporting_base.copy()
        reporting = reporting_base
        reporting = np.clip(reporting * reporting_multiplier, 0.0, 1.0)

        rows = config["contact_matrix"]["rows"]
        contact_matrix = np.array(rows, dtype=float)
        if contact_matrix.shape != (len(age_groups), len(age_groups)):
            raise ValueError("Contact matrix dimensions must match age groups.")
        if not np.isfinite(contact_matrix).all() or np.any(contact_matrix < 0.0):
            raise ValueError("Contact matrix entries must be finite and non-negative.")
        contact_metadata = {}
        correction = config.get("contact_matrix", {}).get("reciprocity_correction", {})
        if correction.get("enabled", False):
            before = reciprocity_error(contact_matrix, population)
            contact_matrix = balance_reciprocity(contact_matrix, population)
            after = reciprocity_error(contact_matrix, population)
            contact_metadata = {
                "contact_reciprocity_error_before": before,
                "contact_reciprocity_error_after": after,
            }

        natural_history = config["natural_history"]
        immunity_model = config.get("immunity_model", {})
        waned_vaccine_duration = float(
            immunity_model.get(
                "waned_vaccine_duration",
                natural_history.get("waned_vaccine_duration", natural_history["vaccine_protection_duration"]),
            )
        )
        rates = {
            "latent": 1.0 / float(natural_history["latent_duration"]),
            "recovery_symptomatic": 1.0 / float(natural_history["infectious_duration_symptomatic"]),
            "recovery_asymptomatic": 1.0 / float(natural_history["infectious_duration_asymptomatic"]),
            "waning_natural": 1.0 / float(natural_history["recovered_immunity_duration"]),
            "waning_vaccine": 1.0 / float(natural_history["vaccine_protection_duration"]),
            "waning_vaccine_waned": 1.0 / waned_vaccine_duration if waned_vaccine_duration > 0 else 0.0,
            "waning_maternal": 1.0 / float(natural_history.get("maternal_protection_duration", 90.0)),
            # SIRWS boosting model rates
            "waning_R_to_W": 1.0 / float(natural_history.get("R_to_W_duration", natural_history["recovered_immunity_duration"])),
            "waning_W_to_S": 1.0 / float(natural_history.get("W_to_S_duration", float(natural_history["recovered_immunity_duration"]) * 2.0)),
        }
        rates.update(config.get("rates", {}))

        symptom_probability = _probability_array(
            [record.get("symptom_probability", 0.4) for record in age_records],
            "Age-group symptom probabilities",
        )

        config_metadata = {
            key: value
            for key, value in config.get("metadata", {}).items()
            if isinstance(value, (str, int, float, bool, np.number))
        }
        calendar = config.get("calendar", {})
        calendar_start_date = _parse_calendar_start_date(calendar)

        # Pre-compute per-origin constant arrays (avoids repeated calls inside rhs)
        from src_python.model.compartments import VACCINE_ORIGINS
        from src_python.model.vaccination import (
            origin_relative_effect as _ore,
            vaccine_susceptibility as _vs,
        )
        _waned = float(immunity_model.get("waned_relative_effect", 0.35))
        _maternal = float(immunity_model.get("maternal_relative_effect", 0.75))
        _dose1 = float(immunity_model.get("dose1_relative_effect", 0.45))
        _dose2 = float(immunity_model.get("dose2_relative_effect", 0.75))
        _ve_sus = float(vaccine.get("VE_sus", 0.0))
        _ve_inf = float(vaccine.get("VE_inf", 0.0))
        _ve_sym = float(vaccine.get("VE_sym", 0.0))
        _ve_dur = float(vaccine.get("VE_dur", 0.0))
        n_age = len(age_groups)

        # Check for maternal-specific VE overrides (from maternal immunization intervention)
        _maternal_ve_sus = float(immunity_model.get("maternal_VE_sus", _ve_sus))
        _maternal_ve_sym = float(immunity_model.get("maternal_VE_sym", _ve_sym))
        _maternal_ve_inf = float(immunity_model.get("maternal_VE_inf", _ve_inf))
        _maternal_ve_dur = float(immunity_model.get("maternal_VE_dur", _ve_dur))

        _rel_effects = np.array([
            _ore(o, waned_relative_effect=_waned, maternal_relative_effect=_maternal,
                 dose1_relative_effect=_dose1, dose2_relative_effect=_dose2)
            for o in VACCINE_ORIGINS
        ], dtype=np.float64)

        # Build susceptibility array with maternal-specific VE_sus for the maternal origin
        _susceptibility = np.empty(len(VACCINE_ORIGINS), dtype=np.float64)
        for i, (o, r) in enumerate(zip(VACCINE_ORIGINS, _rel_effects)):
            ve_sus_for_origin = _maternal_ve_sus if o == "maternal" else _ve_sus
            _susceptibility[i] = _vs(ve_sus_for_origin, relative_effect=float(r))

        # Build infectiousness array with maternal-specific VE_inf
        _infectiousness = np.empty(len(VACCINE_ORIGINS), dtype=np.float64)
        for i, (o, r) in enumerate(zip(VACCINE_ORIGINS, _rel_effects)):
            ve_inf_for_origin = _maternal_ve_inf if o == "maternal" else _ve_inf
            _infectiousness[i] = float(np.clip(1.0 - ve_inf_for_origin * float(r), 0.0, 1.0))

        # Build symptomatic probability array with maternal-specific VE_sym
        _sym_prob = np.empty((len(VACCINE_ORIGINS), n_age), dtype=np.float64)
        for i, (o, r) in enumerate(zip(VACCINE_ORIGINS, _rel_effects)):
            ve_sym_for_origin = _maternal_ve_sym if o == "maternal" else _ve_sym
            _sym_prob[i] = np.clip(symptom_probability * (1.0 - ve_sym_for_origin * float(r)), 0.0, 1.0)

        # Build recovery multiplier array with maternal-specific VE_dur
        _recovery_mult = np.empty(len(VACCINE_ORIGINS), dtype=np.float64)
        for i, (o, r) in enumerate(zip(VACCINE_ORIGINS, _rel_effects)):
            ve_dur_for_origin = _maternal_ve_dur if o == "maternal" else _ve_dur
            _recovery_mult[i] = 1.0 / max(0.05, 1.0 - ve_dur_for_origin * float(r))

        return cls(
            raw=config,
            analysis=analysis,
            scenario=scenario,
            vaccine_scenario=vaccine_scenario,
            resistance_scenario=resistance_scenario,
            intervention=intervention,
            age_groups=age_groups,
            population=population,
            vaccine_coverage=coverage,
            symptom_probability=symptom_probability,
            reporting_rate=reporting,
            diagnosis_probability=diagnosis_probability,
            pep_detection_rate=pep_detection,
            contact_matrix=contact_matrix,
            vaccine=vaccine,
            rates=rates,
            transmission=config["transmission"],
            treatment=config["treatment"],
            pep=config["PEP"],
            initial=config["initial_conditions"],
            immunity_model=immunity_model,
            observation_model=config.get("observation_model", {}),
            resistance=config.get("resistance", {}),
            demography=deepcopy(config.get("demography", {"enabled": False})),
            routine_vaccination=deepcopy(config.get("routine_vaccination", {"enabled": False})),
            importation=deepcopy(config.get("importation", {"enabled": False})),
            calendar=calendar,
            calendar_start_date=calendar_start_date,
            reporting_time_variation=config.get("reporting_time_variation", {}),
            diagnostic_reporting_time_variation=deepcopy(config.get("diagnostic_reporting_time_variation", {})),
            reporting_multiplier=reporting_multiplier,
            metadata={**config_metadata, **contact_metadata, **(metadata or {})},
            origin_relative_effects=_rel_effects,
            origin_susceptibility=_susceptibility,
            origin_infectiousness=_infectiousness,
            origin_symptomatic_prob=_sym_prob,
            origin_recovery_mult=_recovery_mult,
        )

    @property
    def total_population(self) -> float:
        return float(self.population.sum())

    @property
    def infant_age_groups(self) -> set[str]:
        return {"infant_0_2m", "infant_3_11m"}

    def reporting_rate_at(self, t: float) -> np.ndarray:
        rate = self.reporting_rate

        if self.reporting_time_variation:
            start_time = float(self.reporting_time_variation.get("start_time", self.raw["simulation"]["start_time"]))
            end_time = float(self.reporting_time_variation.get("end_time", self.raw["simulation"]["end_time"]))
            start_multiplier = float(self.reporting_time_variation.get("start_multiplier", 1.0))
            end_multiplier = float(self.reporting_time_variation.get("end_multiplier", 1.0))
            if end_time <= start_time:
                reporting_multiplier = end_multiplier
            else:
                progress = np.clip((float(t) - start_time) / (end_time - start_time), 0.0, 1.0)
                reporting_multiplier = start_multiplier + progress * (end_multiplier - start_multiplier)
            rate = rate * reporting_multiplier

        diagnostic_multiplier = self.diagnostic_reporting_multiplier_at(t)
        if diagnostic_multiplier != 1.0:
            rate = rate * diagnostic_multiplier
        return np.clip(rate, 0.0, 1.0)

    def diagnostic_reporting_multiplier_at(self, t: float) -> float:
        variation = self.diagnostic_reporting_time_variation
        if not isinstance(variation, dict) or not bool(variation.get("enabled", False)):
            return 1.0
        periods = variation.get("periods", [])
        if not isinstance(periods, list) or not periods:
            return 1.0

        calendar_date = self.calendar_date_at(t)
        if calendar_date is None:
            return 1.0

        parsed: list[tuple[date, date, float]] = []
        for period in periods:
            if not isinstance(period, dict):
                continue
            try:
                start = _parse_iso_date(period.get("start_date"))
                end = _parse_iso_date(period.get("end_date"))
                multiplier = float(period.get("multiplier", 1.0))
            except (TypeError, ValueError):
                continue
            if end < start or not np.isfinite(multiplier) or multiplier <= 0.0:
                continue
            parsed.append((start, end, multiplier))
        if not parsed:
            return 1.0

        parsed.sort(key=lambda item: (item[0], item[1]))
        for start, end, multiplier in parsed:
            if start <= calendar_date <= end:
                return multiplier
        if calendar_date < parsed[0][0]:
            return parsed[0][2]
        return parsed[-1][2]

    def pep_detection_rate_at(self, t: float) -> np.ndarray:
        return self.pep_detection_rate

    def calendar_date_at(self, t: float) -> date | None:
        if self.calendar_start_date is None:
            return None
        start_time = float(self.raw["simulation"].get("start_time", 0.0))
        return self.calendar_start_date + timedelta(days=float(t) - start_time)

    def calendar_year_at(self, t: float) -> int | None:
        calendar_date = self.calendar_date_at(t)
        return calendar_date.year if calendar_date else None

    def calendar_day_of_year_at(self, t: float) -> float:
        calendar_date = self.calendar_date_at(t)
        if calendar_date is None:
            return float(t % 365.0)
        return float(min(calendar_date.timetuple().tm_yday, 365))

    def wpp_trajectory_active(self) -> bool:
        """Return True when country-specific WPP annual trajectory is wired in."""
        demography = self.demography or {}
        mode = str(demography.get("mode", "")).lower()
        if mode == "fixed_population_profile":
            return False
        trajectory = demography.get("wpp_trajectory")
        if not isinstance(trajectory, dict):
            return False
        if not trajectory.get("population_by_year") or not trajectory.get("births_by_year"):
            return False
        return True

    def wpp_population_at(self, year: float) -> np.ndarray:
        """Interpolate WPP target population by age group for the given year.

        Builds a (n_age × n_years) matrix cache on first call so subsequent
        calls (7000+ per ODE solve) avoid repeated dict lookups and per-age loops.
        """
        traj = self.demography["wpp_trajectory"]
        if "_wpp_years_arr" not in traj:
            years_sorted = sorted(int(y) for y in traj["years"])
            traj["_wpp_years_arr"] = np.array(years_sorted, dtype=float)
            pop_matrix = np.empty((len(self.age_groups), len(years_sorted)), dtype=float)
            for i, age in enumerate(self.age_groups):
                bucket = traj["population_by_year"].get(age, {})
                pop_matrix[i] = [float(bucket[y]) for y in years_sorted]
            traj["_wpp_pop_matrix"] = pop_matrix
        years_arr: np.ndarray = traj["_wpp_years_arr"]
        pop_matrix: np.ndarray = traj["_wpp_pop_matrix"]
        yr = float(year)
        idx = np.searchsorted(years_arr, yr, side="right")
        if idx == 0:
            return pop_matrix[:, 0].copy()
        if idx >= len(years_arr):
            return pop_matrix[:, -1].copy()
        t0, t1 = years_arr[idx - 1], years_arr[idx]
        frac = (yr - t0) / (t1 - t0)
        return pop_matrix[:, idx - 1] + frac * (pop_matrix[:, idx] - pop_matrix[:, idx - 1])

    def wpp_daily_birth_rate_at(self, year: float) -> float:
        """Interpolate WPP annual births and convert to a per-day flow."""
        traj = self.demography["wpp_trajectory"]
        if "_wpp_years_arr" not in traj:
            self.wpp_population_at(year)  # trigger cache build
        if "_wpp_births_arr" not in traj:
            years_sorted = [int(y) for y in sorted(traj["years"])]
            traj["_wpp_births_arr"] = np.array(
                [float(traj["births_by_year"][y]) for y in years_sorted], dtype=float
            )
        years_arr: np.ndarray = traj["_wpp_years_arr"]
        births_arr: np.ndarray = traj["_wpp_births_arr"]
        annual_births = float(np.interp(float(year), years_arr, births_arr))
        return annual_births / 365.0


def _validate_probability_values(values: Any, label: str) -> None:
    arr = np.asarray(list(values), dtype=float)
    if not np.isfinite(arr).all() or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{label} must be finite probabilities within [0, 1].")


def _probability_array(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if not np.isfinite(arr).all() or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{label} must be finite probabilities within [0, 1].")
    return arr


def _parse_calendar_start_date(calendar: dict[str, Any]) -> date | None:
    if not calendar or not bool(calendar.get("enabled", False)):
        return None
    value = calendar.get("analysis_start_date", calendar.get("start_date"))
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _parse_iso_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), "%Y-%m-%d").date()
