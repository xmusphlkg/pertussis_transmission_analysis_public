from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from src_python.model.compartments import (
    COMPARTMENTS,
    STRAINS,
    VACCINE_ORIGINS,
    StateIndex,
    exposed_name,
    infectious_name,
    susceptible_name,
    treated_name,
)
from src_python.model.force_of_infection import compute_force_of_infection
from src_python.model.ode_system import rhs
from src_python.model.parameters import PreparedParameters
from src_python.model.treatment import treated_recovery_rate
from src_python.model.vaccination import (
    default_initial_origin_distribution,
    origin_dose_category,
    origin_is_waned,
    origin_relative_effect,
    origin_symptomatic_probability,
    vaccine_susceptibility,
)


RATE_TO_COUNT_COLUMNS = {
    "symptomatic_case_rate_per_day": "symptomatic_cases",
    "asymptomatic_infection_rate_per_day": "asymptomatic_infections",
    "total_infection_rate_per_day": "total_infections",
    "reported_case_rate_per_day": "reported_cases",
    "infant_case_rate_per_day": "infant_cases",
    "infant_infection_rate_per_day": "infant_infections",
    "child_1_9_case_rate_per_day": "child_1_9_cases",
    "adolescent_case_rate_per_day": "adolescent_cases",
    "child_adolescent_case_rate_per_day": "child_adolescent_cases",
    "child_adolescent_infection_rate_per_day": "child_adolescent_infections",
    "child_adolescent_reported_case_rate_per_day": "child_adolescent_reported_cases",
    "hospitalization_rate_per_day": "hospitalizations",
    "infant_hospitalization_rate_per_day": "infant_hospitalizations",
    "child_adolescent_hospitalization_rate_per_day": "child_adolescent_hospitalizations",
    "mortality_rate_per_day": "deaths",
    "infant_mortality_rate_per_day": "infant_deaths",
    "child_adolescent_mortality_rate_per_day": "child_adolescent_deaths",
    "treated_case_rate_per_day": "treated_cases",
    "PEP_FOI_reduction_case_proxy_rate_per_day": "PEP_FOI_reduction_case_proxy",
    "PEP_averted_case_rate_per_day": "PEP_averted_cases",
}

INFANT_AGE_GROUPS = ("infant_0_2m", "infant_3_11m")
CHILD_1_9_AGE_GROUPS = ("child_1_4y", "child_5_9y")
ADOLESCENT_AGE_GROUPS = ("adolescent_10_17y",)
CHILD_ADOLESCENT_AGE_GROUPS = (*INFANT_AGE_GROUPS, *CHILD_1_9_AGE_GROUPS, *ADOLESCENT_AGE_GROUPS)

ORIGIN_SHARE_COLUMNS = (
    "vaccinated_origin_infection_share",
    "waned_origin_infection_share",
    "maternal_origin_infection_share",
    "dose1_origin_infection_share",
    "dose2_origin_infection_share",
    "dose3plus_origin_infection_share",
)


# ---------------------------------------------------------------------------
# Pertussis severity and mortality model
# ---------------------------------------------------------------------------
# Vaccine protection operates as a cascade of conditional probabilities:
#
#   Infection → Symptomatic → Hospitalized → Death
#
# At each stage, vaccination (including maternal antibodies) provides
# ADDITIONAL protection beyond the previous stage. This means:
#   - VE_sus: reduces probability of infection
#   - VE_sym: reduces probability of symptoms GIVEN infection
#   - VE_hosp: reduces probability of hospitalization GIVEN symptomatic disease
#   - VE_death: reduces probability of death GIVEN hospitalization
#
# The overall VE against death is:
#   VE_death_overall = 1 - (1-VE_sus)*(1-VE_sym)*(1-VE_hosp)*(1-VE_death)
#
# For maternal Tdap in <2mo infants:
#   VE_sus ≈ 0.55, VE_sym ≈ 0.68 (via maternal_relative_effect*0.92)
#   VE_hosp|symptomatic ≈ 0.72 (Amirthalingam 2014; Baxter 2017)
#   VE_death|hospitalized ≈ 0.80 (inferred from near-zero deaths in
#     vaccinated mothers' infants across multiple surveillance systems)
#   Combined VE_death_overall ≈ 1 - 0.45*0.32*0.28*0.20 = 99.2%
#
# Sources:
#   - Amirthalingam et al. (2014) Lancet: VE against hospitalization 91%
#     (this is the OVERALL VE, not conditional; conditional VE_hosp|sym ≈ 72%)
#   - Skoff et al. (2017) Clin Infect Dis: VE against confirmed disease 78%
#   - Baxter et al. (2017) Pediatrics: VE against hospitalization 91.4%
#   - CIDRAP 2025: CDC reported 41% of infant patients <6mo hospitalized, 10 died
#   - Denmark 2024 epidemic: zero deaths among infants of vaccinated mothers
#   - Netherlands 2024: VE against hospitalization 93% (Eurosurveillance 2026)
#
# For unvaccinated infants, the cascade probabilities are:
#   P(hospitalization | symptomatic, <2mo) ≈ 0.60 (CDC: 41-67% hospitalized)
#   P(death | hospitalized, <2mo) ≈ 0.015-0.025 (UK 2024: 11/~700 hospitalized)
#   P(hospitalization | symptomatic, 3-11mo) ≈ 0.25
#   P(death | hospitalized, 3-11mo) ≈ 0.005
# ---------------------------------------------------------------------------

# Age-specific hospitalization probability GIVEN symptomatic disease (unvaccinated baseline)
_AGE_HOSPITALIZATION_PROBABILITY: dict[str, float] = {
    "infant_0_2m": 0.60,        # CDC: 41-67% of <6mo cases hospitalized
    "infant_3_11m": 0.25,       # Lower but still substantial
    "child_1_4y": 0.02,         # Rare hospitalization
    "child_5_9y": 0.003,        # Very rare
    "adolescent_10_17y": 0.001, # Extremely rare
    "young_adult_18_39y": 0.0008,
    "middle_adult_40_64y": 0.002,  # Slightly elevated with comorbidities
    "elderly_65plus": 0.008,    # Elevated due to frailty
}

# Age-specific death probability GIVEN hospitalization (unvaccinated baseline)
_AGE_DEATH_GIVEN_HOSPITALIZATION: dict[str, float] = {
    "infant_0_2m": 0.010,       # US PICU data: ~10 deaths / ~1000 hospitalized <2mo
    "infant_3_11m": 0.002,      # Lower, partially vaccinated
    "child_1_4y": 0.0003,       # Very rare
    "child_5_9y": 0.0001,       # Extremely rare
    "adolescent_10_17y": 0.00005,
    "young_adult_18_39y": 0.00005,
    "middle_adult_40_64y": 0.0003,  # Comorbidity-related
    "elderly_65plus": 0.002,    # Pneumonia complications
}

# Vaccine effectiveness against hospitalization GIVEN symptomatic disease.
# This is the CONDITIONAL VE (not overall VE which combines all stages).
# Maternal antibodies provide strong protection against severe disease
# even when they fail to prevent infection or mild symptoms.
_ORIGIN_VE_HOSPITALIZATION: dict[str, float] = {
    "unvaccinated": 0.0,
    "maternal": 0.72,           # Derived: overall VE_hosp 91% / VE_sym contribution
    "dose1_recent": 0.50,
    "dose1_waned": 0.20,
    "dose2_recent": 0.65,
    "dose2_waned": 0.30,
    "recent": 0.75,             # Full primary series, recently vaccinated
    "waned": 0.35,              # Waned but residual protection against severity
}

# Vaccine effectiveness against death GIVEN hospitalization.
# Even among hospitalized cases, vaccinated infants have better outcomes
# (less likely to develop malignant pertussis with pulmonary hypertension).
_ORIGIN_VE_DEATH: dict[str, float] = {
    "unvaccinated": 0.0,
    "maternal": 0.80,           # Near-complete protection against fatal outcomes
    "dose1_recent": 0.60,
    "dose1_waned": 0.25,
    "dose2_recent": 0.70,
    "dose2_waned": 0.35,
    "recent": 0.80,
    "waned": 0.40,
}


def _compute_severity_outcomes(
    symptomatic_rate_by_origin: dict[str, float],
    age_group: str,
    *,
    maternal_coverage: float = 0.0,
    residual_maternal_protection_for_unvaccinated: bool = False,
) -> tuple[float, float, float]:
    """Compute hospitalization and death rates from origin-specific symptomatic rates.

    Returns (total_hospitalization_rate, total_death_rate, total_symptomatic_rate).
    Each origin's contribution is modified by its VE against hospitalization and death.

    By default, maternal severity protection is applied only to the explicit
    maternal origin. An optional sensitivity flag can apply residual maternal
    protection to unvaccinated-origin infants, but this is disabled in the
    baseline because population-level maternal coverage cannot identify which
    unvaccinated-origin infants were born to vaccinated mothers.

    References:
        - Healy et al. (2013): anti-PT antibodies decline with half-life ~6 weeks
          but remain above protective levels for severe disease for 4-6 months
        - Amirthalingam et al. (2014): VE against hospitalization (91%) exceeds
          VE against confirmed disease (78%), consistent with residual protection
        - Warfel et al. (2014): baboon model shows partial immunity reduces
          disease severity even when colonization is not prevented
    """
    base_p_hosp = _AGE_HOSPITALIZATION_PROBABILITY.get(age_group, 0.001)
    base_p_death = _AGE_DEATH_GIVEN_HOSPITALIZATION.get(age_group, 0.0005)

    # Optional residual severity protection for unvaccinated-origin infants in
    # maternal-program sensitivity analyses. Disabled by default to avoid
    # applying a population-average maternal coverage value to every
    # unvaccinated-origin infant.
    is_infant = age_group in ("infant_0_2m", "infant_3_11m")
    residual_ve_hosp = (
        0.40 * maternal_coverage
        if is_infant and residual_maternal_protection_for_unvaccinated
        else 0.0
    )
    residual_ve_death = (
        0.50 * maternal_coverage
        if is_infant and residual_maternal_protection_for_unvaccinated
        else 0.0
    )

    total_hosp = 0.0
    total_death = 0.0
    total_sym = 0.0

    for origin, sym_rate in symptomatic_rate_by_origin.items():
        if sym_rate <= 0:
            continue
        ve_hosp = _ORIGIN_VE_HOSPITALIZATION.get(origin, 0.0)
        ve_death = _ORIGIN_VE_DEATH.get(origin, 0.0)

        # Apply residual protection to unvaccinated infants
        if origin == "unvaccinated" and is_infant:
            ve_hosp = max(ve_hosp, residual_ve_hosp)
            ve_death = max(ve_death, residual_ve_death)

        # Conditional probability of hospitalization for this origin
        p_hosp = base_p_hosp * (1.0 - ve_hosp)
        # Conditional probability of death given hospitalization for this origin
        p_death_given_hosp = base_p_death * (1.0 - ve_death)

        hosp_rate = sym_rate * p_hosp
        death_rate = hosp_rate * p_death_given_hosp

        total_hosp += hosp_rate
        total_death += death_rate
        total_sym += sym_rate

    return total_hosp, total_death, total_sym


def strain_state_names(strain: str) -> tuple[str, ...]:
    return (
        *(exposed_name(strain, origin) for origin in VACCINE_ORIGINS),
        *(infectious_name(strain, symptom, origin) for symptom in ("sym", "asym") for origin in VACCINE_ORIGINS),
        *(treated_name(strain, origin) for origin in VACCINE_ORIGINS),
    )


def initial_state(params: PreparedParameters, index: StateIndex) -> np.ndarray:
    state = np.zeros((index.n_age, index.n_compartments), dtype=float)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    total_seed_exposed = params.total_population * float(params.initial["initial_exposed_per_100k"]) / 100_000.0
    total_seed_infectious = params.total_population * float(params.initial["initial_infectious_per_100k"]) / 100_000.0
    resistant_fraction = float(params.initial["initial_resistance_prevalence"])
    seed_distribution = params.initial.get("seed_age_distribution", {})

    for age_idx, age in enumerate(index.age_groups):
        coverage = params.vaccine_coverage[age_idx]
        origin_distribution = _initial_origin_distribution(params, age)
        origin_total = max(float(sum(max(0.0, value) for value in origin_distribution.values())), 1e-12)
        for origin, share in origin_distribution.items():
            if origin not in VACCINE_ORIGINS or origin == "unvaccinated":
                continue
            state[age_idx, c[susceptible_name(origin)]] += (
                params.population[age_idx] * coverage * max(0.0, float(share)) / origin_total
            )
        state[age_idx, c["S"]] = params.population[age_idx] * (1.0 - coverage)

        share = float(seed_distribution.get(age, 1.0 / index.n_age))
        exposed = total_seed_exposed * share
        infectious = total_seed_infectious * share
        seeded = exposed + infectious
        source_pools = {
            origin: state[age_idx, c[susceptible_name(origin)]]
            for origin in VACCINE_ORIGINS
        }
        source_total = max(float(sum(source_pools.values())), 1e-12)
        p_sym_by_origin = {
            origin: float(
                origin_symptomatic_probability(
                    np.array([params.symptom_probability[age_idx]], dtype=float),
                    float(params.vaccine.get("VE_sym", 0.0)),
                    origin,
                    waned_relative_effect=float(params.immunity_model.get("waned_relative_effect", 0.35)),
                    maternal_relative_effect=float(params.immunity_model.get("maternal_relative_effect", 0.75)),
                    dose1_relative_effect=float(params.immunity_model.get("dose1_relative_effect", 0.45)),
                    dose2_relative_effect=float(params.immunity_model.get("dose2_relative_effect", 0.75)),
                )[0]
            )
            for origin in VACCINE_ORIGINS
        }

        for origin, pool in source_pools.items():
            origin_share = float(pool) / source_total
            origin_exposed = exposed * origin_share
            origin_infectious = infectious * origin_share
            p_sym = p_sym_by_origin[origin]
            state[age_idx, c[exposed_name("R", origin)]] = origin_exposed * resistant_fraction
            state[age_idx, c[exposed_name("S", origin)]] = origin_exposed * (1.0 - resistant_fraction)
            state[age_idx, c[infectious_name("R", "sym", origin)]] = origin_infectious * resistant_fraction * p_sym
            state[age_idx, c[infectious_name("R", "asym", origin)]] = origin_infectious * resistant_fraction * (1.0 - p_sym)
            state[age_idx, c[infectious_name("S", "sym", origin)]] = origin_infectious * (1.0 - resistant_fraction) * p_sym
            state[age_idx, c[infectious_name("S", "asym", origin)]] = origin_infectious * (1.0 - resistant_fraction) * (1.0 - p_sym)

        for origin, pool in source_pools.items():
            compartment = susceptible_name(origin)
            state[age_idx, c[compartment]] = max(
                0.0,
                state[age_idx, c[compartment]] - seeded * pool / source_total,
            )

    return index.flatten(state)


def _initial_origin_distribution(params: PreparedParameters, age: str) -> dict[str, float]:
    configured = params.immunity_model.get("initial_origin_distribution_by_age", {})
    if age in configured:
        return {str(origin): float(share) for origin, share in configured[age].items()}
    if params.immunity_model.get("mode") == "recent_waned_proxy":
        recent_fraction = _initial_recent_vaccine_fraction(params, age)
        if age == "infant_0_2m":
            return {"maternal": 1.0}
        if age == "infant_3_11m":
            return {"dose1_recent": 0.25, "dose2_recent": 0.35, "recent": 0.40}
        return {"recent": recent_fraction, "waned": 1.0 - recent_fraction}
    return default_initial_origin_distribution(age)


def _initial_recent_vaccine_fraction(params: PreparedParameters, age: str) -> float:
    fractions = params.immunity_model.get("initial_recent_fraction_by_age", {})
    if age in fractions:
        return float(np.clip(float(fractions[age]), 0.0, 1.0))
    return float(np.clip(float(params.immunity_model.get("initial_recent_fraction", 0.5)), 0.0, 1.0))


def target_resistance_prevalence(params: PreparedParameters) -> float:
    resistance = params.resistance or {}
    target = resistance.get(
        "target_prevalence_at_analysis_start",
        params.initial.get("initial_resistance_prevalence", 0.0),
    )
    return float(np.clip(float(target), 0.0, 1.0))


def rebalance_resistant_prevalence(
    y: np.ndarray,
    params: PreparedParameters,
    index: StateIndex,
    *,
    target: float | None = None,
) -> np.ndarray:
    target_fraction = target_resistance_prevalence(params) if target is None else float(np.clip(target, 0.0, 1.0))
    state = np.maximum(index.reshape(y), 0.0).copy()
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    for origin in VACCINE_ORIGINS:
        for sensitive, resistant in (
            (exposed_name("S", origin), exposed_name("R", origin)),
            (infectious_name("S", "sym", origin), infectious_name("R", "sym", origin)),
            (infectious_name("S", "asym", origin), infectious_name("R", "asym", origin)),
            (treated_name("S", origin), treated_name("R", origin)),
        ):
            total = state[:, c[sensitive]] + state[:, c[resistant]]
            state[:, c[resistant]] = total * target_fraction
            state[:, c[sensitive]] = total * (1.0 - target_fraction)
    return index.flatten(state)


def active_resistant_fraction(y: np.ndarray, index: StateIndex) -> float:
    state = np.maximum(index.reshape(y), 0.0)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}
    resistant = sum(float(state[:, c[name]].sum()) for name in strain_state_names("R"))
    sensitive = sum(float(state[:, c[name]].sum()) for name in strain_state_names("S"))
    total = resistant + sensitive
    return resistant / total if total > 0 else 0.0


def solve_model(params: PreparedParameters, index: StateIndex):
    sim = params.raw["simulation"]
    output_time_step = float(sim.get("output_time_step", sim.get("time_step", 1.0)))
    start_time = float(sim["start_time"])
    end_time = float(sim["end_time"])
    t_eval = np.arange(
        start_time,
        end_time + output_time_step,
        output_time_step,
    )
    t_eval = t_eval[t_eval <= end_time]
    if t_eval[-1] < end_time:
        t_eval = np.append(t_eval, end_time)
    y0 = initial_state(params, index)

    solver_method = str(sim.get("solver_method", "LSODA"))
    use_rk4 = solver_method.upper() == "RK4"

    burn_in_years = float(sim.get("burn_in_years", 0.0))
    if burn_in_years > 0:
        burn_t_start = start_time - burn_in_years * 365.0
        if use_rk4:
            from src_python.model.rk4_solver import solve_rk4
            dt_burnin = float(sim.get("rk4_dt_burnin", 2.0))
            burn_solution = solve_rk4(
                params, index, y0,
                t_span=(burn_t_start, start_time),
                dt=dt_burnin,
            )
        else:
            burn_solution = solve_ivp(
                fun=lambda t, y: rhs(t, y, params, index),
                t_span=(burn_t_start, start_time),
                y0=y0,
                t_eval=[start_time],
                method=solver_method,
                rtol=float(sim.get("rtol", 1e-6)),
                atol=float(sim.get("atol", 1e-8)),
            )
        if not burn_solution.success:
            raise RuntimeError(f"Burn-in failed for {params.scenario}: {burn_solution.message}")
        y0 = np.maximum(burn_solution.y[:, -1], 0.0)
        if params.resistance.get("rebalance_after_burn_in", True):
            y0 = rebalance_resistant_prevalence(y0, params, index)

    if use_rk4:
        from src_python.model.rk4_solver import solve_rk4
        dt_analysis = float(sim.get("rk4_dt_analysis", 1.0))
        return solve_rk4(
            params, index, y0,
            t_span=(start_time, end_time),
            t_eval=t_eval,
            dt=dt_analysis,
        )

    return solve_ivp(
        fun=lambda t, y: rhs(t, y, params, index),
        t_span=(start_time, end_time),
        y0=y0,
        t_eval=t_eval,
        method=solver_method,
        rtol=float(sim.get("rtol", 1e-6)),
        atol=float(sim.get("atol", 1e-8)),
    )


def _daily_metrics(t: float, y: np.ndarray, params: PreparedParameters, index: StateIndex) -> list[dict]:
    state = np.maximum(index.reshape(y), 0.0)
    comp = {name: state[:, i] for i, name in enumerate(COMPARTMENTS)}
    foi = compute_force_of_infection(t, y, params, index)
    active_resistance = active_resistant_fraction(y, index)

    use_origin_cache = (
        params.origin_susceptibility.shape == (len(VACCINE_ORIGINS),)
        and params.origin_symptomatic_prob.shape == (len(VACCINE_ORIGINS), index.n_age)
        and params.origin_recovery_mult.shape == (len(VACCINE_ORIGINS),)
    )
    if use_origin_cache:
        susceptibility_arr = params.origin_susceptibility
        sym_prob_arr = params.origin_symptomatic_prob
        recovery_mult_arr = params.origin_recovery_mult
    else:
        ve_sus = float(params.vaccine.get("VE_sus", 0.0))
        ve_sym = float(params.vaccine.get("VE_sym", 0.0))
        ve_dur = float(params.vaccine.get("VE_dur", 0.0))
        maternal_ve_sus = float(params.immunity_model.get("maternal_VE_sus", ve_sus))
        maternal_ve_sym = float(params.immunity_model.get("maternal_VE_sym", ve_sym))
        maternal_ve_dur = float(params.immunity_model.get("maternal_VE_dur", ve_dur))
        waned_relative_effect = float(params.immunity_model.get("waned_relative_effect", 0.35))
        maternal_relative_effect = float(params.immunity_model.get("maternal_relative_effect", 0.75))
        dose1_relative_effect = float(params.immunity_model.get("dose1_relative_effect", 0.45))
        dose2_relative_effect = float(params.immunity_model.get("dose2_relative_effect", 0.75))

        relative_effect_arr = np.array(
            [
                origin_relative_effect(
                    origin,
                    waned_relative_effect=waned_relative_effect,
                    maternal_relative_effect=maternal_relative_effect,
                    dose1_relative_effect=dose1_relative_effect,
                    dose2_relative_effect=dose2_relative_effect,
                )
                for origin in VACCINE_ORIGINS
            ],
            dtype=float,
        )
        susceptibility_arr = np.array(
            [
                vaccine_susceptibility(
                    maternal_ve_sus if origin == "maternal" else ve_sus,
                    relative_effect=float(relative_effect),
                )
                for origin, relative_effect in zip(VACCINE_ORIGINS, relative_effect_arr)
            ],
            dtype=float,
        )
        sym_prob_arr = np.vstack(
            [
                np.clip(
                    params.symptom_probability
                    * (1.0 - (maternal_ve_sym if origin == "maternal" else ve_sym) * float(relative_effect)),
                    0.0,
                    1.0,
                )
                for origin, relative_effect in zip(VACCINE_ORIGINS, relative_effect_arr)
            ]
        )
        recovery_mult_arr = np.array(
            [
                1.0 / max(
                    0.05,
                    1.0 - (maternal_ve_dur if origin == "maternal" else ve_dur) * float(relative_effect),
                )
                for origin, relative_effect in zip(VACCINE_ORIGINS, relative_effect_arr)
            ],
            dtype=float,
        )

    susceptibility_by_origin = {}
    source_pool_by_origin = {}
    for oi, origin in enumerate(VACCINE_ORIGINS):
        susceptibility_by_origin[origin] = float(susceptibility_arr[oi])
        source_pool_by_origin[origin] = comp[susceptible_name(origin)]

    infection_flows = {strain: {} for strain in STRAINS}
    p_sym_by_origin = {
        origin: sym_prob_arr[oi]
        for oi, origin in enumerate(VACCINE_ORIGINS)
    }
    pep_averted = {strain: {} for strain in STRAINS}
    for strain in STRAINS:
        lam = foi[f"lambda_{strain}"]
        base_lam = foi[f"lambda_{strain}_base"]
        for origin in VACCINE_ORIGINS:
            infection_flows[strain][origin] = (
                lam * susceptibility_by_origin[origin] * source_pool_by_origin[origin]
            )
            base_flow = base_lam * susceptibility_by_origin[origin] * source_pool_by_origin[origin]
            pep_averted[strain][origin] = np.maximum(base_flow - infection_flows[strain][origin], 0.0)

    base_gamma_sym = float(params.rates["recovery_symptomatic"])
    base_gamma_asym = float(params.rates["recovery_asymptomatic"])
    gamma_t_s = treated_recovery_rate(base_gamma_sym, params.treatment, "S")
    gamma_t_r = treated_recovery_rate(base_gamma_sym, params.treatment, "R")
    tr_sym_base = float(params.treatment["treatment_rate_symptomatic"])
    tr_asym_base = float(params.treatment["treatment_rate_asymptomatic"])
    diagnosis_prob = params.diagnosis_probability
    tr_sym_by_age = tr_sym_base * diagnosis_prob
    tr_asym_by_age = tr_asym_base * diagnosis_prob
    reporting_rate = params.reporting_rate_at(t)
    diagnostic_reporting_multiplier = params.diagnostic_reporting_multiplier_at(t)
    severity_model = params.raw.get("severity_model", {})
    residual_maternal_protection = bool(
        severity_model.get("residual_maternal_protection_for_unvaccinated_origin", False)
    )

    rows = []
    current_age_population = state.sum(axis=1)
    current_total_population = float(current_age_population.sum())
    for age_idx, age in enumerate(index.age_groups):
        age_population = float(current_age_population[age_idx])
        for strain, strain_label in (("S", "sensitive"), ("R", "resistant")):
            sym_cases = 0.0
            asym_infections = 0.0
            total_infections = 0.0
            treated_cases = 0.0
            recoveries = 0.0
            pep_averted_cases = 0.0
            vaccinated_origin_infections = 0.0
            waned_origin_infections = 0.0
            maternal_origin_infections = 0.0
            dose1_origin_infections = 0.0
            dose2_origin_infections = 0.0
            dose3plus_origin_infections = 0.0
            sym_cases_by_origin: dict[str, float] = {}
            for oi, origin in enumerate(VACCINE_ORIGINS):
                flow = float(infection_flows[strain][origin][age_idx])
                p_sym_origin = float(p_sym_by_origin[origin][age_idx])
                origin_sym = flow * p_sym_origin
                sym_cases += origin_sym
                sym_cases_by_origin[origin] = sym_cases_by_origin.get(origin, 0.0) + origin_sym
                asym_infections += flow * (1.0 - p_sym_origin)
                total_infections += flow
                pep_averted_cases += float(pep_averted[strain][origin][age_idx]) * p_sym_origin
                if origin != "unvaccinated":
                    vaccinated_origin_infections += flow
                if origin_is_waned(origin):
                    waned_origin_infections += flow
                dose_category = origin_dose_category(origin)
                if dose_category == "maternal":
                    maternal_origin_infections += flow
                elif dose_category == "dose1":
                    dose1_origin_infections += flow
                elif dose_category == "dose2":
                    dose2_origin_infections += flow
                elif dose_category == "dose3plus":
                    dose3plus_origin_infections += flow

                recovery_multiplier = float(recovery_mult_arr[oi])
                gamma_sym = base_gamma_sym * recovery_multiplier
                gamma_asym = base_gamma_asym * recovery_multiplier
                gamma_t = (gamma_t_s if strain == "S" else gamma_t_r) * recovery_multiplier
                i_sym = infectious_name(strain, "sym", origin)
                i_asym = infectious_name(strain, "asym", origin)
                treated = treated_name(strain, origin)
                treated_cases += (
                    tr_sym_by_age[age_idx] * comp[i_sym][age_idx]
                    + tr_asym_by_age[age_idx] * comp[i_asym][age_idx]
                )
                recoveries += (
                    gamma_sym * comp[i_sym][age_idx]
                    + gamma_asym * comp[i_asym][age_idx]
                    + gamma_t * comp[treated][age_idx]
                )
            reported_cases = sym_cases * reporting_rate[age_idx]
            is_infant = age in params.infant_age_groups
            is_child_1_9 = age in CHILD_1_9_AGE_GROUPS
            is_adolescent = age in ADOLESCENT_AGE_GROUPS
            is_child_adolescent = age in CHILD_ADOLESCENT_AGE_GROUPS

            # Severity cascade: compute hospitalization and death rates using
            # origin-specific vaccine protection at each severity stage.
            # This correctly captures that maternal antibodies provide STRONGER
            # protection against severe outcomes than against mild disease.
            hospitalization_rate_per_day, mortality_rate_per_day, _ = _compute_severity_outcomes(
                sym_cases_by_origin, age,
                maternal_coverage=float(params.demography.get("birth_entry", {}).get("V", 0.0)),
                residual_maternal_protection_for_unvaccinated=residual_maternal_protection,
            )

            vaccinated_origin_share = vaccinated_origin_infections / total_infections if total_infections > 0 else 0.0
            waned_origin_share = waned_origin_infections / total_infections if total_infections > 0 else 0.0
            maternal_origin_share = maternal_origin_infections / total_infections if total_infections > 0 else 0.0
            dose1_origin_share = dose1_origin_infections / total_infections if total_infections > 0 else 0.0
            dose2_origin_share = dose2_origin_infections / total_infections if total_infections > 0 else 0.0
            dose3plus_origin_share = dose3plus_origin_infections / total_infections if total_infections > 0 else 0.0
            calendar_date = params.calendar_date_at(t)

            rows.append(
                {
                    "time": t,
                    "calendar_date": calendar_date.isoformat() if calendar_date else "",
                    "calendar_year": calendar_date.year if calendar_date else np.nan,
                    "calendar_day_of_year": calendar_date.timetuple().tm_yday if calendar_date else np.nan,
                    "diagnostic_reporting_multiplier": diagnostic_reporting_multiplier,
                    "age_group": age,
                    "strain": strain_label,
                    "analysis": params.analysis,
                    "scenario": params.scenario,
                    "vaccine_scenario": params.vaccine_scenario,
                    "resistance_scenario": params.resistance_scenario,
                    "intervention": params.intervention,
                    "VE_sus": float(params.vaccine.get("VE_sus", 0.0)),
                    "VE_sym": float(params.vaccine.get("VE_sym", 0.0)),
                    "VE_inf": float(params.vaccine.get("VE_inf", 0.0)),
                    "VE_dur": float(params.vaccine.get("VE_dur", 0.0)),
                    "initial_resistance_prevalence": float(params.initial["initial_resistance_prevalence"]),
                    "target_resistance_prevalence_at_analysis_start": target_resistance_prevalence(params),
                    "active_resistant_fraction": active_resistance,
                    "fitness_R": float(params.transmission.get("fitness_R", 1.0)),
                    "pep_coverage_dynamic": float(foi["pep_coverage"]),
                    "population": age_population,
                    "reference_population": float(params.population[age_idx]),
                    "total_population": current_total_population,
                    "reference_total_population": float(params.total_population),
                    "symptomatic_case_rate_per_day": float(sym_cases),
                    "asymptomatic_infection_rate_per_day": float(asym_infections),
                    "total_infection_rate_per_day": float(total_infections),
                    "reported_case_rate_per_day": float(reported_cases),
                    "symptomatic_incidence_per_100k_year": float(sym_cases * 365.0 * 100_000.0 / max(age_population, 1e-9)),
                    "reported_incidence_per_100k_year": float(reported_cases * 365.0 * 100_000.0 / max(age_population, 1e-9)),
                    "infection_incidence_per_100k_year": float(total_infections * 365.0 * 100_000.0 / max(age_population, 1e-9)),
                    "infant_case_rate_per_day": float(sym_cases if is_infant else 0.0),
                    "infant_infection_rate_per_day": float(total_infections if is_infant else 0.0),
                    "child_1_9_case_rate_per_day": float(sym_cases if is_child_1_9 else 0.0),
                    "adolescent_case_rate_per_day": float(sym_cases if is_adolescent else 0.0),
                    "child_adolescent_case_rate_per_day": float(sym_cases if is_child_adolescent else 0.0),
                    "child_adolescent_infection_rate_per_day": float(total_infections if is_child_adolescent else 0.0),
                    "child_adolescent_reported_case_rate_per_day": float(reported_cases if is_child_adolescent else 0.0),
                    "mortality_rate_per_day": float(mortality_rate_per_day),
                    "infant_mortality_rate_per_day": float(mortality_rate_per_day if is_infant else 0.0),
                    "child_adolescent_mortality_rate_per_day": float(
                        mortality_rate_per_day if is_child_adolescent else 0.0
                    ),
                    "hospitalization_rate_per_day": float(hospitalization_rate_per_day),
                    "infant_hospitalization_rate_per_day": float(hospitalization_rate_per_day if is_infant else 0.0),
                    "child_adolescent_hospitalization_rate_per_day": float(
                        hospitalization_rate_per_day if is_child_adolescent else 0.0
                    ),
                    "treated_case_rate_per_day": float(treated_cases),
                    "PEP_FOI_reduction_case_proxy_rate_per_day": float(pep_averted_cases),
                    "PEP_averted_case_rate_per_day": float(pep_averted_cases),
                    "infection_to_recovery_rate_ratio": float(total_infections / max(recoveries, 1e-9)),
                    "vaccinated_origin_infection_share": float(vaccinated_origin_share),
                    "waned_origin_infection_share": float(waned_origin_share),
                    "maternal_origin_infection_share": float(maternal_origin_share),
                    "dose1_origin_infection_share": float(dose1_origin_share),
                    "dose2_origin_infection_share": float(dose2_origin_share),
                    "dose3plus_origin_infection_share": float(dose3plus_origin_share),
                }
            )
    return rows


def compute_timeseries(solution, params: PreparedParameters, index: StateIndex) -> pd.DataFrame:
    if not solution.success:
        raise RuntimeError(f"ODE solver failed for {params.scenario}: {solution.message}")

    rows: list[dict] = []
    for col, t in enumerate(solution.t):
        rows.extend(_daily_metrics(float(t), solution.y[:, col], params, index))

    df = pd.DataFrame(rows)
    df = _add_interval_counts(df)
    key = ["analysis", "scenario", "time", "age_group"]
    resistant = (
        df.assign(resistant_infections=np.where(df["strain"].eq("resistant"), df["total_infections"], 0.0))
        .groupby(key, as_index=False)["resistant_infections"]
        .sum()
    )
    total = df.groupby(key, as_index=False)["total_infections"].sum().rename(columns={"total_infections": "all_infections"})
    df = df.merge(total, on=key, how="left").merge(resistant, on=key, how="left")
    df["resistant_fraction"] = np.divide(
        df["resistant_infections"],
        df["all_infections"],
        out=np.zeros(len(df), dtype=float),
        where=df["all_infections"].to_numpy() > 0,
    )
    df = df.drop(columns=["all_infections", "resistant_infections"])

    group_cols = ["analysis", "scenario", "age_group", "strain"]
    df = df.sort_values(group_cols + ["time"])
    df["cumulative_cases"] = df.groupby(group_cols)["symptomatic_cases"].cumsum()
    df["cumulative_reported_cases"] = df.groupby(group_cols)["reported_cases"].cumsum()
    df["cumulative_infections"] = df.groupby(group_cols)["total_infections"].cumsum()
    return df


def _add_interval_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["analysis", "scenario", "age_group", "strain", "time"]).copy()
    group_cols = ["analysis", "scenario", "age_group", "strain"]
    dt = out.groupby(group_cols)["time"].diff().fillna(0.0).to_numpy(dtype=float)
    for rate_col, count_col in RATE_TO_COUNT_COLUMNS.items():
        previous = out.groupby(group_cols)[rate_col].shift(1).fillna(out[rate_col]).to_numpy(dtype=float)
        current = out[rate_col].to_numpy(dtype=float)
        out[count_col] = 0.5 * (previous + current) * dt
    return out


def infer_output_dt(df: pd.DataFrame) -> float:
    times = np.sort(df["time"].drop_duplicates().to_numpy(dtype=float))
    if len(times) < 2:
        return 1.0
    return float(np.median(np.diff(times)))


def _infection_weighted_share(group: pd.DataFrame, col: str) -> float:
    if col not in group:
        return np.nan
    weights = group["total_infections"].to_numpy(dtype=float)
    denominator = float(weights.sum())
    if denominator <= 0:
        return 0.0
    values = group[col].fillna(0.0).to_numpy(dtype=float)
    return float(np.average(values, weights=weights))


def _sum_column(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return 0.0
    return float(group[column].sum())


def _population_for_age_groups(age_population: pd.Series, age_groups: tuple[str, ...]) -> float:
    return float(age_population.loc[age_population.index.isin(age_groups)].sum())


def summarize_timeseries(
    df: pd.DataFrame,
    *,
    baseline: pd.DataFrame | None = None,
    dt: float = 1.0,
) -> pd.DataFrame:
    scenario_cols = ["analysis", "scenario", "vaccine_scenario", "resistance_scenario", "intervention"]
    summaries = []
    for keys, group in df.groupby(scenario_cols, dropna=False):
        daily = group.groupby("time", as_index=False).agg(
            total_infections=("total_infections", "sum"),
            total_infection_rate_per_day=("total_infection_rate_per_day", "sum"),
            symptomatic_cases=("symptomatic_cases", "sum"),
            symptomatic_case_rate_per_day=("symptomatic_case_rate_per_day", "sum"),
            reported_cases=("reported_cases", "sum"),
            reported_case_rate_per_day=("reported_case_rate_per_day", "sum"),
        )
        peak_idx = int(daily["total_infection_rate_per_day"].idxmax())
        peak_incidence_rate = float(daily.loc[peak_idx, "total_infection_rate_per_day"])
        time_to_peak = float(daily.loc[peak_idx, "time"])
        active = daily.loc[daily["total_infection_rate_per_day"] >= max(1e-9, peak_incidence_rate * 0.01), "time"]
        outbreak_duration = float(active.max() - active.min()) if len(active) else 0.0
        peak_times = _epidemic_peak_times(daily, value_col="reported_cases")
        peak_intervals = np.diff(peak_times) / 365.0 if len(peak_times) >= 2 else np.array([])

        total_infections = float(group["total_infections"].sum())
        total_symptomatic = float(group["symptomatic_cases"].sum())
        total_asymptomatic = float(group["asymptomatic_infections"].sum())
        total_reported = float(group["reported_cases"].sum())
        total_infant_cases = float(group["infant_cases"].sum())
        total_infant_infections = float(group["infant_infections"].sum())
        total_child_1_9_cases = _sum_column(group, "child_1_9_cases")
        total_adolescent_cases = _sum_column(group, "adolescent_cases")
        total_child_adolescent_cases = _sum_column(group, "child_adolescent_cases")
        total_child_adolescent_infections = _sum_column(group, "child_adolescent_infections")
        total_child_adolescent_reported_cases = _sum_column(group, "child_adolescent_reported_cases")
        total_deaths = float(group["deaths"].sum()) if "deaths" in group.columns else 0.0
        total_infant_deaths = float(group["infant_deaths"].sum()) if "infant_deaths" in group.columns else 0.0
        total_child_adolescent_deaths = _sum_column(group, "child_adolescent_deaths")
        total_child_adolescent_hospitalizations = _sum_column(group, "child_adolescent_hospitalizations")
        total_pep_proxy = (
            _sum_column(group, "PEP_FOI_reduction_case_proxy")
            if "PEP_FOI_reduction_case_proxy" in group.columns
            else _sum_column(group, "PEP_averted_cases")
        )
        total_pep_legacy = (
            _sum_column(group, "PEP_averted_cases")
            if "PEP_averted_cases" in group.columns
            else total_pep_proxy
        )
        resistant_infections = float(group.loc[group["strain"].eq("resistant"), "total_infections"].sum())
        resistant_fraction_start = _resistant_fraction_at_time(group, float(group["time"].min()))
        resistant_fraction_end = _resistant_fraction_at_time(group, float(group["time"].max()))
        total_population = float(group["total_population"].mean()) if "total_population" in group else np.nan
        age_population = group.groupby("age_group")["population"].mean()
        infant_population = _population_for_age_groups(age_population, INFANT_AGE_GROUPS)
        child_1_9_population = _population_for_age_groups(age_population, CHILD_1_9_AGE_GROUPS)
        adolescent_population = _population_for_age_groups(age_population, ADOLESCENT_AGE_GROUPS)
        child_adolescent_population = _population_for_age_groups(age_population, CHILD_ADOLESCENT_AGE_GROUPS)
        duration_days = float(group["time"].max()) - float(group["time"].min())
        duration_years = max(duration_days / 365.0, dt / 365.0)

        summary = {
            "analysis": keys[0],
            "scenario": keys[1],
            "vaccine_scenario": keys[2],
            "resistance_scenario": keys[3],
            "intervention": keys[4],
            "calendar_start_date": _calendar_value(group, "calendar_date", "min"),
            "calendar_end_date": _calendar_value(group, "calendar_date", "max"),
            "calendar_start_year": _calendar_value(group, "calendar_year", "min"),
            "calendar_end_year": _calendar_value(group, "calendar_year", "max"),
            "total_symptomatic_cases": total_symptomatic,
            "total_infections": total_infections,
            "total_reported_cases": total_reported,
            "total_infant_cases": total_infant_cases,
            "total_infant_infections": total_infant_infections,
            "total_child_1_9_cases": total_child_1_9_cases,
            "total_adolescent_cases": total_adolescent_cases,
            "total_child_adolescent_cases": total_child_adolescent_cases,
            "total_child_adolescent_infections": total_child_adolescent_infections,
            "total_child_adolescent_reported_cases": total_child_adolescent_reported_cases,
            "total_deaths": total_deaths,
            "total_infant_deaths": total_infant_deaths,
            "total_child_adolescent_deaths": total_child_adolescent_deaths,
            "total_child_adolescent_hospitalizations": total_child_adolescent_hospitalizations,
            "resistant_infections": resistant_infections,
            "resistant_fraction": resistant_infections / total_infections if total_infections > 0 else 0.0,
            "resistant_fraction_start": resistant_fraction_start,
            "resistant_fraction_end": resistant_fraction_end,
            "total_population": total_population,
            "infant_population": infant_population,
            "child_1_9_population": child_1_9_population,
            "adolescent_population": adolescent_population,
            "child_adolescent_population": child_adolescent_population,
            "analysis_years": duration_years,
            "annualized_infections_per_100k": total_infections
            / max(duration_years * total_population, 1e-9)
            * 100_000.0,
            "annualized_reported_cases_per_100k": total_reported
            / max(duration_years * total_population, 1e-9)
            * 100_000.0,
            "annualized_infant_cases_per_100k": total_infant_cases
            / max(duration_years * infant_population, 1e-9)
            * 100_000.0,
            "annualized_infant_infections_per_100k": total_infant_infections
            / max(duration_years * infant_population, 1e-9)
            * 100_000.0,
            "annualized_child_1_9_cases_per_100k": total_child_1_9_cases
            / max(duration_years * child_1_9_population, 1e-9)
            * 100_000.0,
            "annualized_adolescent_cases_per_100k": total_adolescent_cases
            / max(duration_years * adolescent_population, 1e-9)
            * 100_000.0,
            "annualized_child_adolescent_cases_per_100k": total_child_adolescent_cases
            / max(duration_years * child_adolescent_population, 1e-9)
            * 100_000.0,
            "annualized_child_adolescent_infections_per_100k": total_child_adolescent_infections
            / max(duration_years * child_adolescent_population, 1e-9)
            * 100_000.0,
            "annualized_child_adolescent_reported_cases_per_100k": total_child_adolescent_reported_cases
            / max(duration_years * child_adolescent_population, 1e-9)
            * 100_000.0,
            "annualized_deaths_per_million": total_deaths
            / max(duration_years * total_population, 1e-9)
            * 1_000_000.0,
            "annualized_infant_deaths_per_100k": total_infant_deaths
            / max(duration_years * infant_population, 1e-9)
            * 100_000.0,
            "annualized_child_adolescent_deaths_per_million": total_child_adolescent_deaths
            / max(duration_years * child_adolescent_population, 1e-9)
            * 1_000_000.0,
            "annualized_child_adolescent_hospitalizations_per_100k": total_child_adolescent_hospitalizations
            / max(duration_years * child_adolescent_population, 1e-9)
            * 100_000.0,
            "peak_incidence_per_100k_year": peak_incidence_rate * 365.0 * 100_000.0 / max(total_population, 1e-9),
            "treated_cases": float(group["treated_cases"].sum()),
            "PEP_FOI_reduction_case_proxy": total_pep_proxy,
            "PEP_averted_cases": total_pep_legacy,
            "peak_incidence": peak_incidence_rate,
            "time_to_peak": time_to_peak,
            "outbreak_duration": outbreak_duration,
            "n_epidemic_peaks": int(len(peak_times)),
            "mean_peak_interval_years": float(np.mean(peak_intervals)) if len(peak_intervals) else np.nan,
            "proportion_asymptomatic": total_asymptomatic / total_infections if total_infections > 0 else 0.0,
            "case_to_infection_ratio": total_symptomatic / total_infections if total_infections > 0 else 0.0,
            "mean_infection_to_case_ratio": total_infections / total_symptomatic if total_symptomatic > 0 else np.nan,
        }
        for col in ORIGIN_SHARE_COLUMNS:
            summary[col] = _infection_weighted_share(group, col)
        summaries.append(summary)

    out = pd.DataFrame(summaries)
    if baseline is not None and not baseline.empty:
        base = baseline.iloc[0]
        for col in [
            "total_infections",
            "total_reported_cases",
            "total_infant_cases",
            "total_child_1_9_cases",
            "total_adolescent_cases",
            "total_child_adolescent_cases",
            "total_child_adolescent_infections",
            "total_child_adolescent_reported_cases",
            "resistant_infections",
        ]:
            denom = float(base[col])
            out[f"relative_reduction_{col}"] = 1.0 - out[col] / denom if denom > 0 else np.nan
        out["relative_reduction_vs_baseline"] = out["relative_reduction_total_infections"]
    else:
        out["relative_reduction_vs_baseline"] = np.nan
    return out


def _calendar_value(group: pd.DataFrame, column: str, reducer: str):
    if column not in group or group[column].isna().all():
        return ""
    values = group[column]
    if column == "calendar_date":
        values = values.loc[values.astype(str).str.len() > 0]
        if values.empty:
            return ""
        return str(values.min() if reducer == "min" else values.max())
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    value = numeric.min() if reducer == "min" else numeric.max()
    return int(value)


def _resistant_fraction_at_time(group: pd.DataFrame, time: float) -> float:
    at_time = group.loc[np.isclose(group["time"].to_numpy(dtype=float), time)]
    if at_time.empty:
        return 0.0
    if "active_resistant_fraction" in at_time:
        return float(at_time["active_resistant_fraction"].iloc[0])
    resistant = float(at_time.loc[at_time["strain"].eq("resistant"), "total_infections"].sum())
    total = float(at_time["total_infections"].sum())
    return resistant / total if total > 0 else 0.0


def _epidemic_peak_times(daily: pd.DataFrame, *, value_col: str = "total_infections") -> np.ndarray:
    series = daily.sort_values("time").copy()
    if value_col not in series:
        return np.array([], dtype=float)
    values = series[value_col].to_numpy(dtype=float)
    times = series["time"].to_numpy(dtype=float)
    if len(values) < 5 or float(np.nanmax(values)) <= 0.0:
        return np.array([], dtype=float)

    diffs = np.diff(times)
    dt = float(np.median(diffs)) if len(diffs) else 1.0
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 1.0
    # Use a multi-year spacing threshold so annual seasonality does not get
    # mistaken for distinct epidemic waves.
    min_distance_years = 2.5
    min_distance = max(3, int(round(min_distance_years * 365.0 / dt)))
    prominence = max(float(np.nanmax(values)) * 0.01, 1e-9)
    peak_indices, _ = find_peaks(values, distance=min_distance, prominence=prominence)
    return times[peak_indices]
