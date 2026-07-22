from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import load_configs, validate_run_metadata
from src_python.simulation.run_joint_psa_rank_acceptability import (
    FIGURE2B_PARAMETER_NAMES,
    STEM as JOINT_PSA_STEM,
)


TARGET = ROOT / "manuscript" / "appendix_templates" / "supplementary_tables.md"
TABLES_HEADING = "## Supplementary tables"
SECTION_HEADING_RE = re.compile(r"(?m)^## ")
PAGE_BREAK = '<div style="page-break-after: always;"></div>'
_MODEL_CALENDAR = load_configs()["baseline"]["calendar"]
_ANALYSIS_START = str(_MODEL_CALENDAR["analysis_start_date"])
_ANALYSIS_END = str(_MODEL_CALENDAR["analysis_end_date"])
_ANALYSIS_YEARS = f"{_ANALYSIS_START[:4]}-{_ANALYSIS_END[:4]}"

AGE_LABELS = {
    "infant_0_2m": "0-2 mo",
    "infant_3_11m": "3-11 mo",
    "child_1_4y": "1-4 y",
    "child_5_9y": "5-9 y",
    "adolescent_10_17y": "10-17 y",
    "young_adult_18_39y": "18-39 y",
    "middle_adult_40_64y": "40-64 y",
    "elderly_65plus": "65+ y",
}

DISPLAY_LABELS = {
    "adolescent_booster": "Adolescent booster",
    "age_biased": "Age-biased",
    "all_profiles_unweighted": "All profiles, unweighted",
    "adult_focused_improvement": "Adult-focused improvement",
    "baseline": "Baseline",
    "baseline_full_mechanism": "Full baseline mechanism",
    "burn_in": "Burn-in",
    "calibrated_to_reported_cases": "Calibrated to reported cases",
    "child_1_4y": "1-4 y",
    "child_5_9y": "5-9 y",
    "china_passive_system": "China passive system",
    "combined_strategy": "Combined strategy",
    "combined_stress_test_package": "Combined stress-test package",
    "coverage_floor_only": "Nominal coverage floor without timeliness improvement",
    "coverage_floor_plus_timeliness": "Coverage floor plus timeliness improvement",
    "country_timeline": "Country timeline",
    "country_timeline_fitness_advantage": "Country timeline with fitness advantage",
    "country_timeline_fitness_cost": "Country timeline with fitness cost",
    "current": "Current practice",
    "current_near_term": "Current near-term practice",
    "current_practice": "Current practice",
    "cocooning_adjunct": "Close-contact adult adjunct",
    "elderly_65plus": "65+ y",
    "enhanced_surveillance": "Enhanced surveillance",
    "equal_pep_effect": "Equal PEP effect",
    "equal_treatment_effect": "Equal treatment effect",
    "external_age_pattern_pass_filter": "External age-pattern pass filter",
    "external_age_pattern_weighted": "External age-pattern weighted",
    "external_profiles_unweighted": "External profiles, unweighted",
    "fitness_cost": "Fitness-cost stress test",
    "fitness_grid": "fitness grid",
    "fitness_sensitivity": "fitness sensitivity",
    "guided_uptake_100_no_pep_restoration": "100% guided management; baseline PEP",
    "guided_uptake_100_pep_restored": "100% guided management; assumed PEP improvement",
    "guided_uptake_25_pep_restored": "25% guided management; assumed PEP improvement",
    "guided_uptake_50_low_pep_reach": "50% guided management; low PEP reach",
    "guided_uptake_50_no_pep_restoration": "50% guided management; baseline PEP",
    "guided_uptake_50_pep_restored": "50% guided management; assumed PEP improvement",
    "guided_uptake_75_pep_restored": "75% guided management; assumed PEP improvement",
    "high": "High",
    "high_fitness_advantage": "High resistance with fitness advantage",
    "high_transmission_blocking_vaccine_target": "High-transmission-blocking vaccine target",
    "higher_child_coverage": "Nominal coverage floor without timeliness improvement",
    "infant_0_2m": "0-2 mo",
    "infant_3_11m": "3-11 mo",
    "infant_high_adult_very_low": "Infant high, adult very low",
    "infant_moderate_adult_minimal": "Infant moderate, adult minimal",
    "infant_protection_and_exposure_reduction": "Infant protection and exposure reduction",
    "infection_blocking": "Infection-blocking",
    "low": "Low",
    "management_modifiers": "Management modifiers",
    "maternal_adult_boosting_only": "Reproductive-age adult boosting only",
    "maternal_cocooning_only": "Contact reduction only",
    "maternal_direct_antibody_only": "Direct maternal antibody only",
    "maternal_immunization": "High-intensity infant-exposure package",
    "medium": "Medium",
    "middle_adult_40_64y": "40-64 y",
    "moderate": "Moderate",
    "next_generation": "Upper-bound transmission-blocking",
    "next_generation_vaccine": "High-transmission-blocking vaccine target",
    "no_pep": "No PEP",
    "no_resistant_importation": "No resistant importation",
    "no_treatment_or_pep_differential": "No treatment or PEP differential",
    "no_vaccine": "No vaccine",
    "new_zealand": "New Zealand",
    "npi_contact_shock": "NPI contact shock",
    "npi_country_profile": "NPI country profile",
    "npi_none": "No NPI contact shock",
    "npi_reduction_half": "Half NPI contact shock",
    "overall": "Overall",
    "pandemic_npi": "Pandemic/NPI",
    "post_pandemic": "Postpandemic",
    "pregnancy_tdap_plus_adult_household_package": "Pregnancy Tdap plus adult-household package",
    "pregnancy_tdap_scaleup": "Pregnancy Tdap scale-up",
    "resistance_guided_treatment": "Resistance-guided management",
    "routine_program_marginal_levers": "Routine-program marginal levers",
    "south_africa": "South Africa",
    "symptom_protective": "Current aP profile",
    "time_varying": "Time-varying",
    "timeliness_only": "Timeliness only",
    "targeted_pep_high_risk": "Targeted high-risk PEP",
    "transmission_blocking": "Transmission-blocking",
    "united_kingdom": "United Kingdom",
    "united_states": "United States",
    "very_high": "Very high",
    "young_adult_18_39y": "18-39 y",
    "first_5_years": "First 5 years",
    "first_10_years": "First 10 years",
    "first_15_years": "First 15 years",
    "excluding_initial_5_years": "After first 5 years",
    "full_horizon": "Full analysis horizon",
}

DISPLAY_LABEL_COLUMNS = {
    "age_group",
    "analysis_window",
    "ordering_basis",
    "calibrated_value_source",
    "design_level",
    "fitness_or_comparator",
    "intervention",
    "absolute_fit_status",
    "largest_absolute_infection_increase_age_group",
    "largest_increase_age_group",
    "low_event_countries",
    "pep_restored",
    "period",
    "resistance_scenario",
    "scenario",
    "scenario_class",
    "setting",
    "stratum",
    "strategy",
    "target_or_comparator",
    "temporal_dimension",
    "vaccine_scenario",
}

TEXT_FRAGMENT_COLUMNS = {
    "countries",
    "description",
    "design_level",
    "dimension",
    "distribution",
    "domain",
    "evidence_or_diagnostic",
    "evidence_or_rationale",
    "evidence_source",
    "expected_direction_of_bias",
    "explored_range_or_scenarios",
    "analysis_role",
    "fixed_or_conditioned",
    "fitness_or_comparator",
    "grid_values",
    "implementation_note",
    "interpretation",
    "interpretation_note",
    "location_or_role",
    "low_event_countries",
    "mechanistic_relevance",
    "model_representation",
    "modified_control_levers",
    "notes_display",
    "parameter_group",
    "parameter_settings",
    "rationale",
    "primary_role",
    "reason_not_modeled_as_available_policy",
    "residual_caveat",
    "residual_interpretation",
    "selected_contrasts",
    "scenarios_in_class",
    "setting",
    "source_or_anchor",
    "source_provenance",
    "target_or_comparator",
    "uncertainty_range",
    "value",
    "value_or_center",
}

RANGE_FORMAT_EXEMPT_COLUMNS = {
    "range",
    "time_range",
    "uncertainty_range",
}

FORMULA_LABELS = {
    "beta_S": "$\\beta_{\\mathrm{sens}}$",
    "fitness_R": "$f_R$",
    "VE_dur": "$VE_{\\mathrm{dur}}$",
    "VE_inf": "$VE_{\\mathrm{inf}}$",
    "VE_sus": "$VE_{\\mathrm{sus}}$",
    "VE_sym": "$VE_{\\mathrm{sym}}$",
}


def display_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text == "":
        return ""
    key = text.lower()
    if key in DISPLAY_LABELS:
        return DISPLAY_LABELS[key]
    if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", text):
        return text.replace("_", " ").capitalize()
    return text


def display_text_fragments(value: str) -> str:
    text = value
    text = text.replace("post-COVID-19", "post-pandemic")
    text = text.replace("Post-COVID-19", "Post-pandemic")
    text = text.replace("postexposure", "post-exposure")
    text = text.replace("Postexposure", "Post-exposure")
    text = text.replace("CDC/PAHO-style", "CDC and PAHO public health")
    text = text.replace("external-vs-modelled", "external compared with modelled")
    text = text.replace("high-vs-low", "high compared with low")
    text = text.replace(
        "Analysis-design choice for post-pandemic projection horizon",
        "Prespecified post-pandemic analysis horizon",
    )
    text = text.replace("Analysis-design choice for ", "Prespecified ")
    text = re.sub(r"\bvs\b", "versus", text)
    fragment_labels = {
        **AGE_LABELS,
        **{key: label for key, label in DISPLAY_LABELS.items() if "_" in key or any(char.isdigit() for char in key)},
    }
    for key, label in sorted(fragment_labels.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])",
            lambda _match, label=label: label,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(
        r"VE_inf_([0-9.]+)",
        lambda match: f"$VE_{{\\mathrm{{inf}}}}$ {match.group(1)}",
        text,
        flags=re.IGNORECASE,
    )
    for key, label in sorted(FORMULA_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(
            rf"(?<![A-Za-z0-9_\\$]){re.escape(key)}(?![A-Za-z0-9_])",
            lambda _match, label=label: label,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"=(True|true)\b", "=Yes", text)
    text = re.sub(r"=(False|false)\b", "=No", text)
    return text


@dataclass(frozen=True)
class TableSpec:
    number: str
    title: str
    source: Path | str
    columns: tuple[str, ...]
    labels: tuple[str, ...]
    rows: Callable[[], list[dict[str, str]]] | None = None
    sort_by: tuple[str, ...] = ()
    column_styles: tuple[str, ...] = ()


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    full_path = ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(f"Supplementary table input is missing: {path}")
    with full_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_joint_psa_parent() -> None:
    metadata = validate_run_metadata(JOINT_PSA_STEM)
    if metadata.get("run_status") != "complete":
        raise ValueError("Joint PSA parent is not marked complete")
    if tuple(metadata.get("figure2b_parameter_names", ())) != FIGURE2B_PARAMETER_NAMES:
        raise ValueError("Joint PSA parent is not the exact six-input Figure 2b design")
    if metadata.get("excluded_dead_dimensions") != ["resistance_management_uptake"]:
        raise ValueError("Joint PSA parent does not explicitly exclude management uptake")


def _validated_joint_psa_rows(
    path: Path | str,
    *,
    table_kind: str,
) -> list[dict[str, str]]:
    _validate_joint_psa_parent()
    full_path = ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(f"Supplementary table input is missing: {path}")
    with full_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if "resistance_management_uptake" in columns:
        raise ValueError(f"Legacy seven-dimensional joint PSA table rejected: {path}")
    if table_kind == "parameter_samples":
        expected = {
            "psa_sample_id",
            "sample_design",
            "uncertainty_schema_version",
            *FIGURE2B_PARAMETER_NAMES,
        }
        if columns != expected:
            raise ValueError(
                "Joint PSA parameter table must have the exact six-input schema; "
                f"missing={sorted(expected - columns)}, extra={sorted(columns - expected)}"
            )
    elif table_kind == "rank_samples":
        missing = set(FIGURE2B_PARAMETER_NAMES) - columns
        if missing:
            raise ValueError(f"Joint PSA rank table lacks sampled inputs: {sorted(missing)}")
    elif table_kind != "acceptability":
        raise ValueError(f"Unknown joint PSA supplementary-table kind: {table_kind}")
    return rows


def joint_psa_parameter_rows() -> list[dict[str, str]]:
    return _validated_joint_psa_rows(
        "outputs/tables/joint_psa_parameter_samples.csv",
        table_kind="parameter_samples",
    )


def joint_psa_acceptability_rows() -> list[dict[str, str]]:
    return _validated_joint_psa_rows(
        "outputs/tables/joint_psa_rank_acceptability.csv",
        table_kind="acceptability",
    )


def separate_bare_url_list(text: str) -> str:
    return re.sub(r"(https?://[^\s;|]+);\s+(?=https?://)", r"\1<br>", text)


def source_link_display(text: str) -> str:
    parts = [
        part.strip()
        for part in re.split(r";\s*|\s+and\s+(?=https?://)", str(text).strip())
        if part.strip()
    ]
    if not parts:
        return ""
    if not all(part.startswith(("http://", "https://")) for part in parts):
        return text
    if len(parts) == 1:
        return f"[source]({parts[0]})"
    return "<br>".join(f"[source {index}]({part})" for index, part in enumerate(parts, start=1))


def outcome_definition_rows() -> list[dict[str, str]]:
    return [
        {
            "quantity": "Total infections",
            "definition": "Symptomatic plus asymptomatic incident infections integrated over the analysis interval; treated infections are counted at infection onset, not as a separate new infection.",
            "denominator": "Mean total population over the interval.",
            "primary_use": "Overall transmission burden.",
        },
        {
            "quantity": "Reported cases",
            "definition": "Symptomatic incident infections multiplied by age-specific reporting probabilities and integrated over the analysis interval.",
            "denominator": "Mean total population over the interval.",
            "primary_use": "Calibration target and surveillance-comparable burden.",
        },
        {
            "quantity": "Child/adolescent cases (<18 years)",
            "definition": "Symptomatic incident infections in the 0-2 month, 3-11 month, 1-4 year, 5-9 year, and 10-17 year model age groups.",
            "denominator": "Mean population in the five model age groups from birth through age 17 years.",
            "primary_use": "Primary Lancet Child & Adolescent Health endpoint.",
        },
        {
            "quantity": "Infant cases",
            "definition": "Symptomatic incident infections in the 0-2 month and 3-11 month age groups.",
            "denominator": "Mean population in the two infant age groups.",
            "primary_use": "Secondary severe-risk age-stratum outcome.",
        },
        {
            "quantity": "Resistant infections",
            "definition": "Total incident infections attributed to the macrolide-resistant strain.",
            "denominator": "Mean total population over the interval.",
            "primary_use": "Resistance burden and treatment relevance.",
        },
        {
            "quantity": "Resistant fraction",
            "definition": "For interval summaries, resistant incident infections divided by all incident infections; for start/end strain dynamics, resistant active exposed, infectious, and treated compartments divided by all active strain-specific compartments.",
            "denominator": "Total infections or active infected compartments, depending on summary.",
            "primary_use": "Strain-composition diagnostic.",
        },
        {
            "quantity": "PEP-averted cases",
            "definition": "Difference between pre-PEP and post-PEP symptomatic infection flows under the same state trajectory.",
            "denominator": "Not a population-normalized compartment count unless explicitly annualized.",
            "primary_use": "Diagnostic estimate of prophylaxis effect.",
        },
        {
            "quantity": "Relative reduction",
            "definition": "$1 - Z/Z_0$, where $Z$ is the scenario outcome and $Z_0$ is the comparator outcome.",
            "denominator": "Scenario-specific comparator.",
            "primary_use": "Cross-scenario intervention comparison.",
        },
    ]


def fixed_model_setting_rows() -> list[dict[str, str]]:
    return [
        {
            "aspect": "Model class",
            "setting": "Deterministic age-structured compartmental ODE",
            "value": "Two strain classes, country-specific demographics, vaccination histories, and treatment states are explicit; PEP modifies the force of infection and is retained as an averted-case diagnostic.",
        },
        {
            "aspect": "Age structure",
            "setting": "Eight model age groups",
            "value": "0-2 months, 3-11 months, 1-4 years, 5-9 years, 10-17 years, 18-39 years, 40-64 years, and 65 years or older.",
        },
        {
            "aspect": "Strain structure",
            "setting": "Two strain classes",
            "value": "Macrolide-sensitive and macrolide-resistant strains are simulated separately.",
        },
        {
            "aspect": "Vaccine-history structure",
            "setting": "Explicit origin states",
            "value": "Unvaccinated, maternally protected, dose-1 recent/waned, dose-2 recent/waned, and dose-3-plus recent/waned states retain distinct effects.",
        },
        {
            "aspect": "Burn-in and horizon",
            "setting": "Long burn-in plus analysis window",
            "value": f"Fifteen-year pre-analysis history followed by the saved analysis from {_ANALYSIS_START} through {_ANALYSIS_END}; calendar dates, including leap years, define the exact exposure time.",
        },
        {
            "aspect": "Time scale",
            "setting": "Daily rates with weekly saved output",
            "value": "All state equations are evaluated in days, and output is stored every 7 days for downstream summaries.",
        },
        {
            "aspect": "Numerical solver",
            "setting": "Adaptive Runge-Kutta integration",
            "value": "RK45 with relative tolerance 1e-5 and absolute tolerance 1e-7.",
        },
        {
            "aspect": "Seasonality",
            "setting": "Annual cosine forcing",
            "value": "Annual cosine forcing is used by default; an optional 4-year diagnostic term is available when surveillance peaks support multi-year recurrence.",
        },
        {
            "aspect": "Demography",
            "setting": "WPP trajectory-driven age turnover",
            "value": "Births and ageing are driven by UN World Population Prospects 2024 annual trajectories with gentle nudging toward target age profiles; a fixed-profile fallback is retained for tests.",
        },
        {
            "aspect": "Observation model",
            "setting": "Age-specific reporting probabilities",
            "value": "Reported cases are symptomatic infections multiplied by age-specific reporting probabilities; diagnosis rates and PEP-detection proxies are separate parameters.",
        },
        {
            "aspect": "Calibration target",
            "setting": "Reported surveillance intervals",
            "value": "The country-specific sensitive-strain transmission coefficient, $\\beta_{\\mathrm{sens}}$, is selected using a negative-binomial reported-case likelihood, and retained fits must match observed annualised reported incidence within the configured tolerance.",
        },
        {
            "aspect": "Resistance anchoring",
            "setting": "Evidence-based initialisation",
            "value": f"Country-specific anchors use the latest admissible evidence before the {_ANALYSIS_START[:4]} prospective analysis; low-level resistant importation prevents deterministic extinction.",
        },
        {
            "aspect": "Sensitivity screening",
            "setting": "Evidence-prior inverse-CDF Latin-hypercube screening",
            "value": "One hundred twenty-eight stratified unit-cube draws were transformed through parameter-specific marginal distributions for Pearson, Spearman, and PRCC screening, separate from posterior inference.",
        },
        {
            "aspect": "Predictive validation",
            "setting": "Leakage-safe semi-mechanistic discrepancy POMP",
            "value": "Mechanistic expectations are used as offsets in an irregular-time latent discrepancy process with exposure-scaled NB2 observations. One-interval-ahead prequential validation is the primary prediction task; an unassimilated one-year block forecast is a separate stress test. This is not a full stochastic compartmental POMP and does not provide Figure 2 posterior intervals.",
        },
        {
            "aspect": "Resistance fitness stress test",
            "setting": "Continuous $f_R$ grid",
            "value": "Macrolide-resistant strain fitness is varied from 0.70 to 1.25 and crossed with vaccine infectiousness-effect assumptions.",
        },
    ]


def pomp_validation_parameter_rows() -> list[dict[str, str]]:
    """Return the formal predictive-validation specification for reporting."""

    return [
        {
            "parameter_group": "Model scope",
            "parameter": "mechanistic_offset",
            "value_or_center": "Deterministic country-specific expected interval counts",
            "uncertainty_range": "Not sampled as a posterior parameter",
            "unit": "reported cases",
            "distribution": "Fixed offset within each fold",
            "analysis_role": "Predictive validation only",
            "interpretation": "The stochastic layer is a semi-mechanistic discrepancy POMP, not a full stochastic compartmental POMP.",
        },
        {
            "parameter_group": "Latent discrepancy",
            "parameter": "process_candidates",
            "value_or_center": "Persistent, robust, and compound-Poisson-jump damped processes",
            "uncertainty_range": "Selected using three preceding validation years",
            "unit": "log notification rate",
            "distribution": "Irregular-time transition laws",
            "analysis_role": "Inner-fold model selection",
            "interpretation": "Country candidate scores are shrunk towards panel scores to reduce unstable local selection.",
        },
        {
            "parameter_group": "Jump process",
            "parameter": "jump_rate_and_size",
            "value_or_center": "0.35 events per year; log-scale SD 1.25",
            "uncertainty_range": "Poisson event count scales with elapsed calendar time",
            "unit": "events per year; log scale",
            "distribution": "Compound Poisson with within-interval OU attenuation",
            "analysis_role": "Abrupt surveillance/transmission discrepancy candidate",
            "interpretation": "Jump probability is scientifically scaled to irregular interval duration rather than applied once per row.",
        },
        {
            "parameter_group": "Observation model",
            "parameter": "NB2_dispersion_grid",
            "value_or_center": "10, 30, 100, 300, 1000",
            "uncertainty_range": "Selected inside each outer fold",
            "unit": "size parameter",
            "distribution": "Exposure-scaled negative binomial",
            "analysis_role": "Predictive measurement dispersion",
            "interpretation": "The interval size parameter scales with exposure duration so irregular reporting windows remain comparable.",
        },
        {
            "parameter_group": "Formal computation",
            "parameter": "particles_predictive_draws_replicates",
            "value_or_center": "512 particles; 4096 predictive draws; 3 replicates",
            "uncertainty_range": "Independent deterministic seeds recorded by fold",
            "unit": "counts",
            "distribution": "Bootstrap particle filter and empirical predictive law",
            "analysis_role": "Monte Carlo stability gate",
            "interpretation": "The three repeats quantify numerical Monte Carlo variability separately from epidemiological variation.",
        },
        {
            "parameter_group": "Forecast ensemble",
            "parameter": "stacking_components",
            "value_or_center": "POMP, seasonal naive, exponentially weighted recent rate, damped log trend",
            "uncertainty_range": "Weights learned from past validation observations only",
            "unit": "mixture weight",
            "distribution": "Probability and point-prediction stacking",
            "analysis_role": "Prespecified comparator and ensemble layer",
            "interpretation": "The ensemble must outperform every component baseline on country-balanced log score.",
        },
        {
            "parameter_group": "Primary validation",
            "parameter": "prequential_outer_folds",
            "value_or_center": "2023-26; 34 country-year folds; 693 intervals",
            "uncertainty_range": "Observation assimilated only after its forecast is scored",
            "unit": "folds and reporting intervals",
            "distribution": "One-reporting-interval-ahead",
            "analysis_role": "Canonical predictive publication gate",
            "interpretation": "Primary metrics are country-balanced log score, coverage, sharpness, and point error.",
        },
        {
            "parameter_group": "Stress test and claim scope",
            "parameter": "one_year_block",
            "value_or_center": "All within-year observations withheld",
            "uncertainty_range": "Failed annual joint-coverage gate",
            "unit": "one year",
            "distribution": "Unassimilated block forecast",
            "analysis_role": "Long-horizon interpretation boundary",
            "interpretation": "2027-50 outputs are conditional scenarios, not validated annual forecasts or absolute-burden predictions.",
        },
    ]


def prior_model_comparison_rows() -> list[dict[str, str]]:
    return [
        {
            "prior_model": "Wearing and Rohani 2009",
            "age_structure": "Population-level transmission signatures",
            "waning": "Yes",
            "asymptomatic_transmission": "Implicit or limited",
            "vaccine_infection_blocking": "Composite immunity",
            "vaccine_infectiousness_reduction": "Not separated",
            "resistance": "No",
            "treatment_pep": "No",
            "infant_specific_outcome": "No",
        },
        {
            "prior_model": "Lavine et al 2011",
            "age_structure": "Age-structured pertussis dynamics",
            "waning": "Yes, with immune boosting",
            "asymptomatic_transmission": "Limited",
            "vaccine_infection_blocking": "Composite protection",
            "vaccine_infectiousness_reduction": "Not separated",
            "resistance": "No",
            "treatment_pep": "No",
            "infant_specific_outcome": "Limited",
        },
        {
            "prior_model": "Althouse and Scarpino 2015",
            "age_structure": "Age-structured transmission model",
            "waning": "Yes",
            "asymptomatic_transmission": "Yes",
            "vaccine_infection_blocking": "Composite or scenario-level",
            "vaccine_infectiousness_reduction": "Not decomposed into $VE_{\\mathrm{inf}}$ and $VE_{\\mathrm{dur}}$",
            "resistance": "No",
            "treatment_pep": "No",
            "infant_specific_outcome": "Limited",
        },
        {
            "prior_model": "Chit et al 2018",
            "age_structure": "Meta-analysis plus modeling",
            "waning": "Yes",
            "asymptomatic_transmission": "Not primary focus",
            "vaccine_infection_blocking": "Vaccine-effectiveness endpoint",
            "vaccine_infectiousness_reduction": "No",
            "resistance": "No",
            "treatment_pep": "No",
            "infant_specific_outcome": "Limited",
        },
        {
            "prior_model": "Domenech de Celles et al 2018",
            "age_structure": "Age-structured transmission model",
            "waning": "Yes",
            "asymptomatic_transmission": "Implicit in transmission structure",
            "vaccine_infection_blocking": "Composite vaccine-history protection",
            "vaccine_infectiousness_reduction": "Not separated",
            "resistance": "No",
            "treatment_pep": "No",
            "infant_specific_outcome": "Yes, but not resistance-guided",
        },
        {
            "prior_model": "Drivers of resurgence pilot report 2025",
            "age_structure": "Multi-country age-structured model",
            "waning": "Yes",
            "asymptomatic_transmission": "Yes or implicit",
            "vaccine_infection_blocking": "Included",
            "vaccine_infectiousness_reduction": "Incomplete separation",
            "resistance": "No",
            "treatment_pep": "No resistance-guided PEP",
            "infant_specific_outcome": "Yes",
        },
        {
            "prior_model": "Current model",
            "age_structure": "Eight age groups and country-specific contact matrices",
            "waning": "SIRWS waning and boosting",
            "asymptomatic_transmission": "Explicit",
            "vaccine_infection_blocking": "$VE_{\\mathrm{sus}}$",
            "vaccine_infectiousness_reduction": "$VE_{\\mathrm{inf}}$ and $VE_{\\mathrm{dur}}$ separated",
            "resistance": "Two strain classes with fitness and importation",
            "treatment_pep": "Strain-specific treatment and PEP assumptions",
            "infant_specific_outcome": "Infant secondary endpoint; <18-year child/adolescent cases are primary",
        },
    ]


def _latest_resistance_anchor_by_country() -> dict[str, str]:
    rows = read_csv_rows("data/raw/country_resistance_timeline.csv")
    anchors: dict[str, dict[str, str]] = {}
    for row in rows:
        country = row.get("country", "")
        try:
            year = int(float(row.get("year", "")))
            fraction = float(row.get("resistant_fraction", ""))
        except ValueError:
            continue
        current = anchors.get(country)
        if current is None or year >= int(float(current.get("year", "0"))):
            anchors[country] = {
                "year": str(year),
                "anchor": f"{fraction * 100:.2f}% ({year})",
            }
    return {country: value["anchor"] for country, value in anchors.items()}


EVIDENCE_TYPE_LABELS = {
    "global_surveillance_extrapolation": "Global-surveillance extrapolation",
    "low_detected_model_anchor": "Low detected model anchor",
    "low_imported_model_anchor": "Low imported model anchor",
    "measured_historical_national_isolate_fraction": "Measured historical national isolate fraction",
    "measured_multicenter_isolate_fraction": "Measured multicenter isolate fraction",
    "measured_multistate_surveillance_fraction": "Measured multistate surveillance fraction",
    "measured_national_genomic_surveillance_fraction": "Measured national genomic surveillance fraction",
    "measured_national_surveillance_fraction": "Measured national surveillance fraction",
    "measured_regional_case_series_fraction": "Measured regional case series fraction",
    "measured_regional_isolate_fraction": "Measured regional isolate fraction",
}


def _percent_text(value: str) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _resistance_interval_text(row: dict[str, str]) -> str:
    point = _percent_text(row.get("resistant_fraction", ""))
    lower = _percent_text(row.get("lower", ""))
    upper = _percent_text(row.get("upper", ""))
    if point and lower and upper:
        return f"{point} ({lower}-{upper})"
    return point


def _sample_size_text(row: dict[str, str]) -> str:
    sample_size = str(row.get("sample_size", "")).strip()
    if sample_size:
        try:
            return f"{int(round(float(sample_size))):,}"
        except ValueError:
            return sample_size
    evidence_type = str(row.get("evidence_type", "")).lower()
    if "model_anchor" in evidence_type or "extrapolation" in evidence_type:
        return "Not publicly reported; model anchor"
    return "Not publicly reported"


def _resistance_note(row: dict[str, str]) -> str:
    if str(row.get("sample_size", "")).strip():
        return row.get("notes", "")
    evidence_type = str(row.get("evidence_type", "")).lower()
    if evidence_type == "global_surveillance_extrapolation":
        return row.get("notes", "") or (
            "No country-specific denominator located; conservative low extrapolated anchor retained pending surveillance."
        )
    return row.get("notes", "") or "No public denominator located; conservative low anchor retained pending surveillance."


def resistance_evidence_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows("data/raw/country_resistance_timeline.csv"):
        evidence_type = row.get("evidence_type", "")
        rows.append(
            {
                **row,
                "sample_size_display": _sample_size_text(row),
                "resistant_fraction_interval": _resistance_interval_text(row),
                "evidence_label": EVIDENCE_TYPE_LABELS.get(evidence_type, display_label(evidence_type)),
                "source_display": source_link_display(row.get("source", "")),
                "notes_display": _resistance_note(row),
            }
        )
    return rows


def country_selection_rows() -> list[dict[str, str]]:
    inputs = {row.get("config_key", ""): row for row in read_csv_rows("data/processed/country_profile_inputs.csv")}
    profile = {row.get("country", ""): row for row in read_csv_rows("manuscript_notes/country_profile_table.csv")}
    anchors = _latest_resistance_anchor_by_country()
    who_region = {
        "Australia": "Western Pacific",
        "Brazil": "Americas",
        "China": "Western Pacific",
        "Japan": "Western Pacific",
        "New_Zealand": "Western Pacific",
        "South_Africa": "African",
        "Sweden": "European",
        "Thailand": "South-East Asia",
        "United_Kingdom": "European",
        "United_States": "Americas",
    }
    inclusion_reason = {
        "Australia": "High recent reported incidence; measured low but detectable resistance; mature maternal and booster program.",
        "Brazil": "Large Americas profile with whole-cell pertussis schedule, maternal program, and detected resistant cases without national fraction.",
        "China": "Large population, marked postpandemic resurgence, and near-complete measured macrolide resistance anchor.",
        "Japan": "Western Pacific resurgence and high measured resistance in 2024-2025 reports.",
        "New_Zealand": "Small high-income profile with maternal and adolescent programs and emerging resistance concern.",
        "South_Africa": "African-region profile with shorter overlapping calibration window and contrasting demography.",
        "Sweden": "European profile with high-quality surveillance and booster program contrast.",
        "Thailand": "South-East Asian low reported-incidence profile with whole-cell pertussis schedule and low maternal coverage.",
        "United_Kingdom": "European maternal-program profile with established pregnancy vaccination and surveillance data.",
        "United_States": "Large Americas profile with adolescent and maternal Tdap program and low reported resistance.",
    }
    data_quality = {
        "Australia": "High",
        "Brazil": "Moderate",
        "China": "High",
        "Japan": "High",
        "New_Zealand": "Moderate",
        "South_Africa": "Moderate",
        "Sweden": "High",
        "Thailand": "Moderate",
        "United_Kingdom": "High",
        "United_States": "High",
    }

    rows: list[dict[str, str]] = []
    for key in who_region:
        source = inputs.get(key, {})
        summary = profile.get(key, {})
        booster_parts = []
        dose_count = source.get("routine_dose_count", "")
        if dose_count:
            booster_parts.append(f"{dose_count} routine doses")
        if str(source.get("adolescent_booster", "")).lower() == "true":
            booster_parts.append("adolescent booster")
        elif source.get("adolescent_booster", "") != "":
            booster_parts.append("no adolescent booster")
        rows.append(
            {
                "country": key.replace("_", " "),
                "who_region": who_region[key],
                "population": summary.get("total_population", ""),
                "dtp3_coverage": source.get("dtp3_coverage", ""),
                "booster_schedule": "; ".join(booster_parts),
                "maternal_vaccination_policy": source.get("maternal_program_note", ""),
                "recent_reported_incidence": summary.get("observed_mean_annual_reported_incidence_per_100k", ""),
                "resistance_anchor": anchors.get(key.replace("_", " "), anchors.get(key, "")),
                "reason_for_inclusion": inclusion_reason[key],
                "data_quality_rating": data_quality[key],
            }
        )
    return rows


def _parse_semicolon_key_values(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in str(text).split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _short_decimal_text(value: str) -> str:
    try:
        number = float(value)
        if abs(number) < 0.005:
            number = 0.0
        return f"{number:.2f}"
    except (TypeError, ValueError):
        return value


def _format_reporting_prior_bounds(text: str) -> str:
    parsed = _parse_semicolon_key_values(text)
    if not parsed:
        return ""
    lines = []
    for age_key in AGE_LABELS:
        value = parsed.get(age_key, "")
        match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\[([+-]?(?:\d+(?:\.\d*)?|\.\d+)),([+-]?(?:\d+(?:\.\d*)?|\.\d+))\]",
            value,
        )
        if match:
            mean, lower, upper = match.groups()
            lines.append(
                f"{AGE_LABELS[age_key]}: $p_{{rep}}={_short_decimal_text(mean)}$ "
                f"$[{_short_decimal_text(lower)}, {_short_decimal_text(upper)}]$"
            )
        elif value:
            lines.append(f"{AGE_LABELS[age_key]}: {value}")
    return "\n".join(lines)


def _format_decimal_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            number = float(token)
            if token in {"0.065", "+0.065", "-0.065", ".065", "+.065", "-.065"}:
                return f"{number:.3f}"
            if abs(number) < 0.005:
                number = 0.0
            return f"{number:.2f}"
        except ValueError:
            return token

    return re.sub(r"(?<![A-Za-z0-9.])[-+]?(?:\d+\.\d+|\.\d+)(?!(?:[A-Za-z0-9]|\.\d))", replace, text)


def _format_range_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            number = float(token)
            if abs(number) < 0.005:
                number = 0.0
            return f"{number:.2f}"
        except ValueError:
            return token

    return re.sub(
        r"(?<![A-Za-z0-9.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9.])",
        replace,
        text,
    )


def reporting_probability_rows() -> list[dict[str, str]]:
    rows = []
    calibration_rows = read_csv_rows("outputs/tables/calibration_all_countries.csv")
    for row in calibration_rows:
        country = row.get("country", "")
        by_age = _parse_semicolon_key_values(row.get("reporting_multiplier_by_age", ""))
        priors = row.get("reporting_rate_prior_by_age", "")
        try:
            child_1_4_vals = [
                float(by_age.get("child_1_4y", "nan")),
            ]
            school_adolescent_5_17_vals = [
                float(by_age.get("child_5_9y", "nan")),
                float(by_age.get("adolescent_10_17y", "nan")),
            ]
            adult_vals = [
                float(by_age.get("young_adult_18_39y", "nan")),
                float(by_age.get("middle_adult_40_64y", "nan")),
                float(by_age.get("elderly_65plus", "nan")),
            ]
        except ValueError:
            child_1_4_vals = school_adolescent_5_17_vals = adult_vals = []

        def mean_text(values: list[float]) -> str:
            finite = [value for value in values if value == value]
            if not finite:
                return ""
            return f"{sum(finite) / len(finite):.2f}"

        rows.append(
            {
                "country": country,
                "infant_0_2m_reporting_probability": by_age.get("infant_0_2m", ""),
                "infant_3_11m_reporting_probability": by_age.get("infant_3_11m", ""),
                "child_1_4y_reporting_probability": mean_text(child_1_4_vals),
                "school_adolescent_5_17y_reporting_probability": mean_text(school_adolescent_5_17_vals),
                "adult_18plus_reporting_probability": mean_text(adult_vals),
                "prior_bounds": _format_reporting_prior_bounds(priors),
                "calibrated_value_source": row.get("reporting_rate_prior_evidence_class", ""),
            }
        )
    return rows


PARAMETER_GROUP_BY_NAME = {
    "Analysis horizon": "Analysis design and numerics",
    "Pre-analysis history origin": "Analysis design and numerics",
    "Saved output interval": "Analysis design and numerics",
    "Production solver relative tolerance": "Analysis design and numerics",
    "Production solver absolute tolerance": "Analysis design and numerics",
    "Sensitive-strain transmission coefficient ($\\beta_{\\mathrm{sens}}$)": "Transmission and seasonality",
    "Relative infectiousness of asymptomatic infection": "Transmission and seasonality",
    "Resistant-strain relative fitness ($f_R$)": "Transmission and seasonality",
    "Seasonal forcing amplitude": "Transmission and seasonality",
    "Seasonal forcing phase": "Transmission and seasonality",
    "Inter-epidemic recurrence period": "Transmission and seasonality",
    "Multi-year recurrence amplitude": "Transmission and seasonality",
    "Multi-year recurrence phase": "Transmission and seasonality",
    "Latent period": "Natural history and immunity",
    "Symptomatic infectious period": "Natural history and immunity",
    "Asymptomatic infectious period": "Natural history and immunity",
    "Passive maternal-protection duration": "Natural history and immunity",
    "Post-infection protection duration": "Natural history and immunity",
    "Vaccine-derived protection duration": "Natural history and immunity",
    "SIRWS recovered-to-waned duration": "Natural history and immunity",
    "SIRWS waned-to-susceptible duration": "Natural history and immunity",
    "SIRWS boosting efficiency": "Natural history and immunity",
    "Waned-natural infection susceptibility": "Natural history and immunity",
    "Waned vaccine-origin effect weight": "Natural history and immunity",
    "Maternal-origin effect weight": "Natural history and immunity",
    "Dose-1 origin effect weight": "Natural history and immunity",
    "Dose-2 origin effect weight": "Natural history and immunity",
    "Waned vaccine-origin loss duration": "Natural history and immunity",
    "Initial recent-vaccine-origin fraction": "Natural history and immunity",
    "Vaccine susceptibility effect ($VE_{\\mathrm{sus}}$)": "Vaccine mechanism",
    "Vaccine symptom effect ($VE_{\\mathrm{sym}}$)": "Vaccine mechanism",
    "Vaccine infectiousness effect ($VE_{\\mathrm{inf}}$)": "Vaccine mechanism",
    "Vaccine infectious-duration effect ($VE_{\\mathrm{dur}}$)": "Vaccine mechanism",
    "Imported infection seeding rate": "Resistance and importation",
    "Target resistant fraction at analysis start": "Resistance and importation",
    "Resistant fraction among imports": "Resistance and importation",
    "Resistance prevalence anchoring rate": "Resistance and importation",
    "Resistance rebalance after burn-in": "Resistance and importation",
    "Initial resistant fraction before burn-in": "Resistance and importation",
    "Age-specific reporting probabilities": "Observation and calibration",
    "Reporting multiplier": "Observation and calibration",
    "Negative-binomial calibration dispersion": "Observation and calibration",
    "Calibration observation window": "Observation and calibration",
    "Calibration mean-incidence tolerance": "Observation and calibration",
    "Routine vaccination target-relaxation rate": "Routine vaccination and initialisation",
    "Routine vaccination maximum daily flow": "Routine vaccination and initialisation",
    "Initial exposed seeding": "Routine vaccination and initialisation",
    "Initial infectious seeding": "Routine vaccination and initialisation",
    "Treatment initiation rate for symptomatic infection": "Treatment and post-exposure prophylaxis",
    "Treatment initiation rate for asymptomatic infection": "Treatment and post-exposure prophylaxis",
    "Sensitive-strain treatment duration reduction": "Treatment and post-exposure prophylaxis",
    "Sensitive-strain treatment infectiousness reduction": "Treatment and post-exposure prophylaxis",
    "Resistant-strain treatment duration reduction": "Treatment and post-exposure prophylaxis",
    "Resistant-strain treatment infectiousness reduction": "Treatment and post-exposure prophylaxis",
    "PEP reach among close contacts": "Treatment and post-exposure prophylaxis",
    "PEP effectiveness for macrolide-sensitive infection": "Treatment and post-exposure prophylaxis",
    "PEP effectiveness for macrolide-resistant infection": "Treatment and post-exposure prophylaxis",
    "PEP activation prevalence proxy": "Treatment and post-exposure prophylaxis",
}


def parameter_table_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv_rows("manuscript_notes/parameter_table.csv"):
        _reject_legacy_figure2c_uncertainty_language(
            row,
            source="manuscript_notes/parameter_table.csv",
        )
        parameter = row.get("parameter", "")
        rows.append(
            {
                **row,
                "parameter_group": PARAMETER_GROUP_BY_NAME.get(parameter, "Other model parameter"),
            }
        )
    return rows


_LEGACY_FIGURE2C_UNCERTAINTY = re.compile(
    r"figure\s*2c|joint\s+(?:credible\s+interval|cri)|credible\s+interval",
    flags=re.IGNORECASE,
)


def _reject_legacy_figure2c_uncertainty_language(
    row: dict[str, str],
    *,
    source: str,
) -> None:
    text = " ".join(str(value) for value in row.values())
    match = _LEGACY_FIGURE2C_UNCERTAINTY.search(text)
    if match:
        raise ValueError(
            f"{source} retains prohibited Figure 2c uncertainty language: "
            f"{match.group(0)!r}"
        )


_BAYESIAN_PARAMETER_ALIASES = {
    "figure2c_adolescent_coverage_floor": "optional_research_adolescent_coverage_floor",
    "figure2c_maternal_coverage_floor": "optional_research_maternal_coverage_floor",
    "figure2c_young_adult_coverage_floor": "optional_research_young_adult_coverage_floor",
    "figure2c_contact_reduction_fraction": "optional_research_contact_reduction_fraction",
    "figure2c_targeted_pep_coverage": "optional_research_targeted_pep_coverage",
    "figure2c_maternal_protection_duration_days": (
        "optional_research_maternal_protection_duration_days"
    ),
}


BAYESIAN_PARAMETER_LABELS = {
    "log_beta_S": "$\\log(\\beta_{\\mathrm{sens}})$",
    "log_reporting_multiplier": "$\\log(m_{\\mathrm{rep}})$",
    "VE_sus": "$VE_{\\mathrm{sus}}$",
    "VE_inf": "$VE_{\\mathrm{inf}}$",
    "VE_dur": "$VE_{\\mathrm{dur}}$",
    "relative_infectiousness_asymptomatic": "$\\rho_{\\mathrm{asym}}$",
    "infectious_duration_symptomatic": "$D_{\\mathrm{sym}}$",
    "infectious_duration_asymptomatic": "$D_{\\mathrm{asym}}$",
    "fitness_R": "$f_R$",
    "resistance_prevalence": "$p_R$",
    "maternal_VE_sus": "$VE^{\\mathrm{mat}}_{\\mathrm{sus}}$",
    "maternal_VE_sym": "$VE^{\\mathrm{mat}}_{\\mathrm{sym}}$",
    "optional_research_adolescent_coverage_floor": "Adolescent booster coverage floor",
    "optional_research_maternal_coverage_floor": "Pregnancy Tdap coverage floor",
    "optional_research_young_adult_coverage_floor": "Reproductive-age adult coverage floor",
    "optional_research_contact_reduction_fraction": "Adult-to-infant contact-reduction fraction",
    "optional_research_targeted_pep_coverage": "Targeted household-contact PEP coverage",
    "optional_research_maternal_protection_duration_days": "Passive maternal-protection duration",
}


def bayesian_prior_rows() -> list[dict[str, str]]:
    rows = []
    for raw_row in read_csv_rows("manuscript_notes/bayesian_prior_table.csv"):
        row = dict(raw_row)
        if "analysis_role" not in row and "figure2c_role" in row:
            legacy_role = row.pop("figure2c_role", "")
            if "optional legacy nonpublication" not in legacy_role.lower():
                raise ValueError(
                    "Legacy bayesian_prior_table figure2c_role can be normalized only "
                    "when it is explicitly optional nonpublication research."
                )
            row["analysis_role"] = "optional_nonpublication_legacy_research"
        parameter = _BAYESIAN_PARAMETER_ALIASES.get(
            row.get("parameter", ""),
            row.get("parameter", ""),
        )
        row["parameter"] = parameter
        _reject_legacy_figure2c_uncertainty_language(
            row,
            source="manuscript_notes/bayesian_prior_table.csv",
        )
        if row.get("analysis_role") != "optional_nonpublication_legacy_research":
            raise ValueError(
                "Every bayesian_prior_table row must be labelled "
                "optional_nonpublication_legacy_research."
            )
        rows.append(
            {
                **row,
                "parameter": BAYESIAN_PARAMETER_LABELS.get(parameter, display_label(parameter)),
                "interpretation": re.sub(r"\s+", " ", row.get("interpretation", "")).strip(),
            }
        )
    return rows


def resistance_scenario_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv_rows("manuscript_notes/resistance_scenario_table.csv"):
        out = dict(row)
        if out.get("uses_country_resistance_timeline", "").strip().lower() in {"true", "yes", "1"}:
            out["target_prevalence_at_analysis_start"] = "Country-specific timeline estimate"
            out["importation_fraction"] = "Country-specific timeline estimate"
        rows.append(out)
    return rows


def veinf_threshold_rows() -> list[dict[str, str]]:
    source_rows = read_csv_rows("outputs/summaries/fitness_resistance_grid_summary.csv")
    values: dict[tuple[str, float, float], float] = {}
    for row in source_rows:
        country = row.get("country", "")
        try:
            fitness = round(float(row.get("grid_fitness_R", "")), 3)
            ve_inf = round(float(row.get("grid_VE_inf", "")), 3)
            infant = float(row.get("annualized_infant_cases_per_100k", ""))
        except ValueError:
            continue
        values[(country, fitness, ve_inf)] = infant

    selected_fitness = (0.85, 1.00, 1.10, 1.15)
    thresholds = (0.25, 0.50, 0.75)
    ve_grid = sorted({ve for _, _, ve in values})
    rows: list[dict[str, str]] = []
    for fitness in selected_fitness:
        countries = sorted({country for country, fit, ve in values if fit == round(fitness, 3) and ve == 0.25})
        reductions_by_ve: dict[float, list[float]] = {}
        for ve_inf in ve_grid:
            if ve_inf < 0.25:
                continue
            reductions = []
            for country in countries:
                baseline = values.get((country, round(fitness, 3), 0.25))
                outcome = values.get((country, round(fitness, 3), ve_inf))
                if baseline and outcome is not None:
                    reductions.append(1.0 - outcome / baseline)
            if reductions:
                reductions_by_ve[ve_inf] = reductions

        for threshold in thresholds:
            reached = [
                (ve_inf, reductions)
                for ve_inf, reductions in reductions_by_ve.items()
                if median(reductions) >= threshold
            ]
            if reached:
                ve_inf, reductions = reached[0]
                rows.append(
                    {
                        "fitness_R": f"{fitness:.2f}",
                        "target_reduction": f"{threshold * 100:.2f}%",
                        "minimum_VE_inf": f"{ve_inf:.2f}",
                        "median_reduction_at_minimum": f"{median(reductions) * 100:.2f}%",
                        "countries_meeting_threshold": f"{sum(value >= threshold for value in reductions)}/{len(reductions)}",
                        "interpretation": "Threshold reached on the simulated $VE_{\\mathrm{inf}}$ grid.",
                    }
                )
            else:
                final_ve = max(reductions_by_ve)
                final_reductions = reductions_by_ve[final_ve]
                rows.append(
                    {
                        "fitness_R": f"{fitness:.2f}",
                        "target_reduction": f"{threshold * 100:.2f}%",
                        "minimum_VE_inf": f"Not reached through {final_ve:.2f}",
                        "median_reduction_at_minimum": f"{median(final_reductions) * 100:.2f}% at {final_ve:.2f}",
                        "countries_meeting_threshold": f"{sum(value >= threshold for value in final_reductions)}/{len(final_reductions)}",
                        "interpretation": "Threshold not reached on the simulated grid.",
                    }
                )
    return rows


def _safe_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _median_text(values: list[float]) -> str:
    finite = [value for value in values if value == value]
    if not finite:
        return ""
    return f"{median(finite):.2f}"


def _quantile(values: list[float], probability: float) -> float | None:
    finite = sorted(value for value in values if value == value)
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(finite) - 1)
    weight = position - lower
    return finite[lower] * (1 - weight) + finite[upper] * weight


def _iqr_text(values: list[float]) -> str:
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    if q1 is None or q3 is None:
        return ""
    return f"{q1:.2f}-{q3:.2f}"


def _unique_numeric_text(values: list[float], digits: int = 2) -> str:
    unique = sorted({round(value, digits + 2) for value in values if value == value})
    return ", ".join(f"{value:.{digits}f}" for value in unique)


def country_inputs_selection_rows() -> list[dict[str, str]]:
    profile = {
        row.get("country", ""): row
        for row in read_csv_rows("manuscript_notes/country_profile_table.csv")
    }
    rows = []
    for row in country_selection_rows():
        key = row.get("country", "").replace(" ", "_")
        summary = profile.get(key, {})
        vaccine_product = summary.get("vaccine_product", "")
        vaccine_product_labels = {
            "aP": "acellular pertussis",
            "wP": "whole-cell pertussis",
        }
        vaccine_product = vaccine_product_labels.get(vaccine_product, vaccine_product)
        coverage = row.get("dtp3_coverage", "")
        try:
            coverage_text = f"DTP3 coverage {float(coverage) * 100:.2f}%"
        except (TypeError, ValueError):
            coverage_text = ""
        maternal_policy = row.get("maternal_vaccination_policy", "").strip()
        if maternal_policy.lower() in {"", "na", "nan", "none"}:
            maternal_policy = ""
        elif maternal_policy.lower().startswith("no routine"):
            maternal_policy = "no routine maternal program"
        else:
            maternal_policy = re.sub(r"(\d)w\b", r"\1 weeks", maternal_policy)
            maternal_policy = f"maternal {maternal_policy}"
        program_profile = "; ".join(
            value
            for value in (
                f"{vaccine_product} vaccine" if vaccine_product else "",
                coverage_text,
                row.get("booster_schedule", ""),
                maternal_policy,
            )
            if value
        )
        rows.append(
            {
                **row,
                "program_profile": program_profile,
                "seasonal_phase": summary.get("seasonal_phase", ""),
                "seasonal_amplitude": summary.get("seasonal_amplitude", ""),
                "vaccine_product": vaccine_product,
                "contact_source": summary.get("contact_source", ""),
            }
        )
    return rows


def calibration_diagnostic_rows() -> list[dict[str, str]]:
    calibration = {
        row.get("country", ""): row
        for row in read_csv_rows("outputs/tables/calibration_all_countries.csv")
    }
    rows = []
    def na_if_blank(value: object) -> str:
        text = "" if value is None else str(value).strip()
        return "NA" if text == "" or text.lower() == "nan" else text

    for row in read_csv_rows("outputs/tables/calibration_fit_diagnostics_summary.csv"):
        country = row.get("country", "")
        fit = calibration.get(country, {})
        rows.append(
            {
                "country": country,
                "period": row.get("period", ""),
                "calibration_accepted": fit.get("calibration_accepted", ""),
                "absolute_fit_status": fit.get("absolute_fit_status", ""),
                "calibrated_beta": fit.get("calibrated_beta", ""),
                "observed_mean_annual_reported_incidence_per_100k": fit.get(
                    "observed_mean_annual_reported_incidence_per_100k", ""
                ),
                "annualized_reported_cases_per_100k": fit.get("annualized_reported_cases_per_100k", ""),
                "model_to_observed_reported_incidence_ratio": fit.get(
                    "model_to_observed_reported_incidence_ratio", ""
                ),
                "n_intervals": row.get("n_intervals", ""),
                "observed_total_reported_cases": row.get("observed_total_reported_cases", ""),
                "predicted_total_reported_cases": row.get("predicted_total_reported_cases", ""),
                "mean_absolute_percentage_error": row.get("mean_absolute_percentage_error", ""),
                "peak_observed_year": na_if_blank(row.get("peak_observed_year", "")),
                "peak_predicted_year": na_if_blank(row.get("peak_predicted_year", "")),
                "peak_timing_error_years": na_if_blank(row.get("peak_timing_error_years", "")),
                "peak_magnitude_ratio": na_if_blank(row.get("peak_magnitude_ratio", "")),
            }
        )
    return rows


def fitness_grid_summary_rows() -> list[dict[str, str]]:
    rows = read_csv_rows("manuscript_notes/fitness_grid_table.csv")
    fitness_values = [_safe_float(row.get("fitness_R")) for row in rows]
    ve_values = [_safe_float(row.get("VE_inf")) for row in rows]
    fitness = [value for value in fitness_values if value is not None]
    ve_inf = [value for value in ve_values if value is not None]
    description = next((row.get("description", "").strip() for row in rows if row.get("description", "").strip()), "")
    return [
        {
            "dimension": "Resistant-strain relative fitness",
            "grid_values": _unique_numeric_text(fitness, digits=2),
            "selected_contrasts": "0.85, 1.00, and 1.15 emphasized in the main text.",
            "interpretation": "Values below 1.00 impose a resistant-strain transmission penalty; values above 1.00 impose a transmission advantage.",
        },
        {
            "dimension": "Vaccine infectiousness effect, $VE_{\\mathrm{inf}}$",
            "grid_values": _unique_numeric_text(ve_inf, digits=2),
            "selected_contrasts": "0.05 to 0.55 range used for figure S11 surfaces.",
            "interpretation": "$VE_{\\mathrm{inf}}$ reduces onward infectiousness among infected vaccine-history origins; it is not an infection-acquisition endpoint.",
        },
        {
            "dimension": "Crossed grid",
            "grid_values": f"{len(set(fitness))} $f_R$ values $\\times$ {len(set(ve_inf))} $VE_{{\\mathrm{{inf}}}}$ values",
            "selected_contrasts": f"{len(rows)} simulated combinations retained in repository source table.",
            "interpretation": description,
        },
    ]


def vaccine_threshold_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in veinf_threshold_rows():
        rows.append(
            {
                "threshold_type": "Reduction target",
                "fitness_or_comparator": f"$f_R$={row.get('fitness_R', '')}",
                "resistance_prevalence": "",
                "target_or_comparator": row.get("target_reduction", ""),
                "minimum_VE_inf": row.get("minimum_VE_inf", ""),
                "countries": row.get("countries_meeting_threshold", ""),
                "interpretation": row.get("interpretation", ""),
            }
        )
    for row in read_csv_rows("outputs/tables/veinf_comparator_thresholds.csv"):
        minimum_ve_inf = row.get("median_minimum_VE_inf", "")
        rows.append(
            {
                "threshold_type": "Comparator threshold",
                "fitness_or_comparator": row.get("comparator", ""),
                "resistance_prevalence": row.get("resistance_prevalence", ""),
                "target_or_comparator": row.get("threshold_basis", ""),
                "minimum_VE_inf": minimum_ve_inf if minimum_ve_inf else "Not reached",
                "countries": f"{row.get('countries_reaching_comparator', '')}/{row.get('countries_evaluated', '')}",
                "interpretation": "Median minimum $VE_{\\mathrm{inf}}$ needed to meet or exceed the comparator across evaluated countries.",
            }
        )
    return rows


def infant_contact_maternal_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows("outputs/tables/infant_contact_sensitivity.csv"):
        rows.append(
            {
                "sensitivity_dimension": "Infant contact multiplier",
                "strategy": row.get("strategy", ""),
                "setting": row.get("infant_contact_multiplier", ""),
                "median_infant_cases_per_100k_5y": row.get("median_infant_cases_per_100k", ""),
                "iqr_infant_cases_per_100k_5y": row.get("iqr_infant_cases_per_100k", ""),
                "median_infant_case_reduction_vs_current_5y": "",
                "iqr_reduction": "",
                "countries_with_positive_reduction": "",
                "countries": row.get("countries", ""),
                "interpretation": row.get("interpretation", ""),
            }
        )
    for row in read_csv_rows("outputs/tables/maternal_duration_sensitivity.csv"):
        rows.append(
            {
                "sensitivity_dimension": "Maternal passive-protection duration",
                "strategy": row.get("strategy", ""),
                "setting": row.get("maternal_protection_duration_days", ""),
                "median_infant_cases_per_100k_5y": row.get("median_infant_cases_per_100k_5y", ""),
                "iqr_infant_cases_per_100k_5y": row.get("iqr_infant_cases_per_100k_5y", ""),
                "median_infant_case_reduction_vs_current_5y": row.get(
                    "median_infant_case_reduction_vs_current_5y", ""
                ),
                "iqr_reduction": row.get("iqr_infant_case_reduction_vs_current_5y", ""),
                "countries_with_positive_reduction": row.get("countries_with_positive_reduction", ""),
                "countries": row.get("countries", ""),
                "interpretation": row.get("interpretation", ""),
            }
        )
    return rows


def higher_child_coverage_diagnostic_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows("outputs/tables/higher_child_coverage_country_mechanism.csv"):
        rows.append(
            {
                "diagnostic": "Country infant-case change",
                "stratum": row.get("country", ""),
                "current_infant_cases_per_100k": row.get("current_infant_cases_per_100k", ""),
                "higher_child_coverage_infant_cases_per_100k": row.get(
                    "higher_child_coverage_infant_cases_per_100k", ""
                ),
                "relative_change_or_share": row.get("relative_reduction_infant_cases", ""),
                "largest_increase_age_group": row.get("largest_absolute_infection_increase_age_group", ""),
                "age_shift_iqr": "",
                "countries_with_increase": "",
                "interpretation": "Country-level modeled infant cases and the age group with the largest absolute infection increase.",
            }
        )
    for row in read_csv_rows("outputs/tables/higher_child_coverage_age_shift.csv"):
        iqr = f"{row.get('q25_relative_change_infections', '')} to {row.get('q75_relative_change_infections', '')}"
        rows.append(
            {
                "diagnostic": "Age-shift summary",
                "stratum": row.get("age_group", ""),
                "current_infant_cases_per_100k": "",
                "higher_child_coverage_infant_cases_per_100k": "",
                "relative_change_or_share": row.get("median_relative_change_infections", ""),
                "largest_increase_age_group": "",
                "age_shift_iqr": iqr,
                "countries_with_increase": row.get("countries_with_infection_increase", ""),
                "interpretation": "Median age-specific infection change under the nominal coverage-floor scenario without timeliness improvement.",
            }
        )

    origin_rows = read_csv_rows("outputs/tables/infant_vaccine_history_origin_shares.csv")
    for scenario in ("current", "higher_child_coverage"):
        selected = [row for row in origin_rows if row.get("scenario") == scenario]
        rows.append(
            {
                "diagnostic": "Vaccine-history origin share",
                "stratum": scenario,
                "current_infant_cases_per_100k": "",
                "higher_child_coverage_infant_cases_per_100k": "",
                "relative_change_or_share": _median_text(
                    [
                        value
                        for value in (_safe_float(row.get("vaccinated_origin_infection_share")) for row in selected)
                        if value is not None
                    ]
                ),
                "largest_increase_age_group": "",
                "age_shift_iqr": "",
                "countries_with_increase": str(len(selected)),
                "interpretation": "Median vaccinated-origin infant infection share; source CSV retains dose-specific shares.",
            }
        )
    for row in read_csv_rows("outputs/tables/routine_timeliness_sensitivity.csv"):
        if row.get("strategy") == "current":
            continue
        rows.append(
            {
                "diagnostic": "Routine timeliness sensitivity",
                "stratum": row.get("strategy", ""),
                "current_infant_cases_per_100k": row.get("median_current_infant_cases_per_100k", ""),
                "higher_child_coverage_infant_cases_per_100k": row.get(
                    "median_scenario_infant_cases_per_100k", ""
                ),
                "relative_change_or_share": row.get("median_relative_reduction_infant_cases", ""),
                "largest_increase_age_group": "",
                "age_shift_iqr": row.get("iqr_relative_reduction_infant_cases", ""),
                "countries_with_increase": row.get("countries_with_positive_reduction", ""),
                "interpretation": row.get("implementation_note", ""),
            }
        )
    return rows


def infant_age_summary_rows() -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in read_csv_rows("outputs/tables/infant_age_split_horizon_sensitivity.csv"):
        key = (row.get("analysis_window", ""), row.get("age_group", ""), row.get("scenario", ""))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, str]] = []
    for (window, age_group, scenario), group in sorted(grouped.items()):
        cases = [
            value
            for value in (_safe_float(row.get("annualized_infant_cases_per_100k")) for row in group)
            if value is not None
        ]
        reductions = [
            value
            for value in (_safe_float(row.get("relative_reduction_infant_cases")) for row in group)
            if value is not None
        ]
        ranks = [value for value in (_safe_float(row.get("rank")) for row in group) if value is not None]
        rows.append(
            {
                "analysis_window": window,
                "age_group": age_group,
                "scenario": scenario,
                "median_infant_cases_per_100k": _median_text(cases),
                "iqr_infant_cases_per_100k": _iqr_text(cases),
                "median_relative_reduction": _median_text(reductions),
                "median_rank": _median_text(ranks),
                "countries_with_positive_reduction": str(sum(value > 0 for value in reductions)),
                "countries": str(len(group)),
            }
        )
    return rows


def event_scale_summary_rows() -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv_rows("outputs/tables/deterministic_event_scale_diagnostics.csv"):
        grouped.setdefault(row.get("scenario", ""), []).append(row)
    rows: list[dict[str, str]] = []
    for scenario, group in sorted(grouped.items()):
        annual_infant_counts = [
            value
            for value in (_safe_float(row.get("annual_infant_cases_count")) for row in group)
            if value is not None
        ]
        infant_rates = [
            value
            for value in (_safe_float(row.get("annualized_infant_cases_per_100k")) for row in group)
            if value is not None
        ]
        low_flags = [
            row.get("country", "")
            for row in group
            if "Low" in row.get("event_scale_flag", "")
        ]
        rows.append(
            {
                "scenario": scenario,
                "countries": str(len(group)),
                "median_annual_infant_cases_count": _median_text(annual_infant_counts),
                "minimum_annual_infant_cases_count": f"{min(annual_infant_counts):.2f}" if annual_infant_counts else "",
                "median_infant_cases_per_100k": _median_text(infant_rates),
                "low_event_countries": "; ".join(low_flags) if low_flags else "None",
                "interpretation": "Low-event countries are most sensitive to stochastic extinction or clustering assumptions.",
            }
        )
    return rows


def joint_psa_summary_rows() -> list[dict[str, str]]:
    reduction_ranges: dict[str, str] = {}
    rank_samples = _validated_joint_psa_rows(
        "outputs/tables/joint_psa_infant_rank_samples.csv",
        table_kind="rank_samples",
    )
    reductions_by_strategy: dict[str, list[float]] = {}
    for rank_row in rank_samples:
        strategy = rank_row.get("strategy", "")
        reduction = _safe_float(rank_row.get("relative_reduction_infant_cases_vs_current"))
        if strategy and reduction is not None:
            reductions_by_strategy.setdefault(strategy, []).append(reduction)
    for strategy, values in reductions_by_strategy.items():
        q025 = _quantile(values, 0.025)
        q975 = _quantile(values, 0.975)
        if q025 is not None and q975 is not None:
            reduction_ranges[strategy] = f"{q025:.4g} to {q975:.4g}"

    rows = []
    for row in joint_psa_acceptability_rows():
        if row.get("country") != "All_countries_pooled" or row.get("rank") != "1":
            continue
        row = dict(row)
        row["relative_reduction_range"] = reduction_ranges.get(row.get("strategy", ""), "")
        rows.append(row)
    return rows


def _compact_numeric(value: object) -> str:
    return _short_decimal_text(str(value).strip())


def stochastic_toy_key_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv_rows("outputs/tables/individual_stochastic_toy_summary.csv"):
        total_interval = (
            f"{_compact_numeric(row.get('median_total_infections', ''))} "
            f"({_compact_numeric(row.get('q025_total_infections', ''))}-"
            f"{_compact_numeric(row.get('q975_total_infections', ''))})"
        )
        infant_summary = (
            f"mean {_compact_numeric(row.get('mean_infant_infections', ''))}; "
            f"Pr(any) {_compact_numeric(row.get('probability_any_infant_infection', ''))}; "
            f"Q95 {_compact_numeric(row.get('q95_infant_infections', ''))}"
        )
        rows.append(
            {
                "country": row.get("country", ""),
                "contact_structure": display_label(row.get("scenario", "")),
                "extinction_probability_3_or_fewer": row.get("extinction_probability_3_or_fewer", ""),
                "outbreak_probability_20plus": row.get("outbreak_probability_20plus", ""),
                "total_infections_interval": total_interval,
                "infant_infection_summary": infant_summary,
                "mean_household_clusters_touched": row.get("mean_household_clusters_touched", ""),
            }
        )
    return rows


PIPELINE_SUMMARIES = {
    "BPZE1 live-attenuated intranasal vaccine": {
        "candidate_or_platform": "BPZE1 intranasal live attenuated",
        "development_status": "Phase 2b adult/challenge evidence; school-age trial registered.",
        "transmission_signal": "Mucosal immunity and colonization reduction; closest to the high-transmission-blocking target.",
        "model_use": "Upper-bound transmission-blocking profile plus $VE_{\\mathrm{inf}}$ sensitivity; no product-specific efficacy assigned.",
    },
    "Outer-membrane-vesicle or OMV-adjuvanted pertussis platforms": {
        "candidate_or_platform": "OMV or OMV-adjuvanted platforms",
        "development_status": "Preclinical/translational evidence; no late-stage pertussis efficacy trial identified.",
        "transmission_signal": "Broader antigenic and Th1/Th17 responses; possible effects on susceptibility, infectiousness, or duration.",
        "model_use": "Covered by infection-/transmission-blocking profiles and $VE_{\\mathrm{inf}}$/$VE_{\\mathrm{dur}}$ ranges.",
    },
    "Genetically detoxified recombinant pertussis-toxin acellular vaccines": {
        "candidate_or_platform": "Recombinant PT acellular boosters",
        "development_status": "Licensed recombinant boosters reported in Asia; Pertagen2x phase II/III registered.",
        "transmission_signal": "Potentially stronger or more durable antibody response; not primarily mucosal transmission blocking.",
        "model_use": "Mapped to adolescent booster, current aP, infection-blocking, or waning-duration sensitivity.",
    },
    "New multi-component acellular pertussis combination vaccines": {
        "candidate_or_platform": "New multi-component acellular combinations",
        "development_status": "CanSino DTcP phase 3 active-not-recruiting; other products remain platform-specific.",
        "transmission_signal": "Relevant to clinical protection and possibly infection blocking; limited direct carriage evidence.",
        "model_use": "Covered by current aP and infection-blocking profiles; no separate product scenario.",
    },
}


def vaccine_pipeline_summary_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv_rows("outputs/tables/vaccine_pipeline_mechanism_mapping.csv"):
        summary = PIPELINE_SUMMARIES.get(row.get("candidate_or_platform", ""), {})
        rows.append(
            {
                "candidate_or_platform": summary.get("candidate_or_platform", row.get("candidate_or_platform", "")),
                "development_status": summary.get(
                    "development_status",
                    row.get("development_status_as_of_2026_05_21", ""),
                ),
                "transmission_signal": summary.get("transmission_signal", row.get("mechanistic_relevance", "")),
                "model_use": summary.get("model_use", row.get("model_representation", "")),
                "evidence_source": row.get("evidence_source", ""),
            }
        )
    return rows


def interpretation_guardrail_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows("outputs/tables/limitation_diagnostic_map.csv"):
        rows.append(
            {
                "guardrail_type": "Limitation diagnostic",
                "domain": row.get("limitation_domain", ""),
                "evidence_or_diagnostic": row.get("added_or_existing_diagnostic", ""),
                "location_or_role": row.get("supplement_location", ""),
                "interpretation": row.get("residual_interpretation", ""),
            }
        )
    for row in read_csv_rows("outputs/tables/strategy_domain_interpretation.csv"):
        rows.append(
            {
                "guardrail_type": "Strategy-domain assumption",
                "domain": row.get("strategy_domain_or_assumption", ""),
                "evidence_or_diagnostic": row.get("evidence_strength", ""),
                "location_or_role": row.get("decision_role", ""),
                "interpretation": row.get("scenario_comparison_interpretation", ""),
            }
        )
    return rows


def study_parameter_design_rows() -> list[dict[str, str]]:
    rows = [
        {
            "analysis_component": "Country profiles and calibration",
            "design_level": "Nine accepted publication calibrations plus one exploratory profile",
            "parameter_settings": "Australia, Brazil, China, Japan, New Zealand, Sweden, Thailand, United Kingdom, and United States form the calibrated publication set; South Africa is retained only for exploratory deterministic analysis because it has fewer than five recent non-overlapping surveillance intervals. Profiles include country-specific demography, contacts, vaccination, seasonality, surveillance intervals, resistance anchors, $\\beta_{\\mathrm{sens}}$, and reporting multipliers.",
            "source_provenance": "Population denominators [13], schedule and coverage records [14], contact matrices [15-17], reported-case intervals [18], treatment and PEP assumptions [19,20], resistance guidance [21,22], and country resistance reports [23,24], [25], [26], [27], [28,29]; calibrated $\\beta_{\\mathrm{sens}}$ and reporting multipliers are model-estimated from reported-case intervals.",
            "fixed_or_conditioned": f"Common deterministic ODE structure, age partition, natural-history defaults, 15-year pre-analysis history, and {_ANALYSIS_YEARS} saved horizon.",
            "primary_role": "Defines calibrated current-practice comparators and cross-country heterogeneity.",
            "detail_location": "tables S1, S2, S5, S6, S12, and S13.",
        },
    ]

    vaccine_source = {
        "no_vaccine": "Null counterfactual with all vaccine-effect parameters set to zero; no external efficacy claim.",
        "symptom_protective": "Acellular-pertussis-like disease protection, asymptomatic-transmission structure, incomplete infection blocking, and waning informed by the WHO vaccine framework [1], transmission evidence [5,6], and duration-of-protection studies [7-9].",
        "infection_blocking": "Mechanistic scenario above the population-average aP profile, bounded by vaccine-framework assumptions [1], transmission evidence [5,6], and waning studies [7-9], then checked against vaccine-pipeline interpretation in table S17.",
        "transmission_blocking": "Improved-transmission-blocking scenario informed by the WHO vaccine framework [1], aP/wP transmission evidence [5,6], waning studies [7-9], and product-target reasoning in table S17; not a licensed product estimate.",
        "next_generation": "Upper-bound high-transmission-blocking product-target profile; represented as a hypothetical mechanism profile using vaccine-framework assumptions [1], transmission evidence [5,6], waning studies [7-9], and pipeline mapping in table S17.",
    }
    for row in read_csv_rows("manuscript_notes/scenario_table.csv"):
        scenario = row.get("scenario", "")
        rows.append(
            {
                "analysis_component": "Vaccine-mechanism profile",
                "design_level": scenario,
                "parameter_settings": "; ".join(
                    (
                        f"$VE_{{\\mathrm{{sus}}}}$={row.get('VE_sus', '')}",
                        f"$VE_{{\\mathrm{{sym}}}}$={row.get('VE_sym', '')}",
                        f"$VE_{{\\mathrm{{inf}}}}$={row.get('VE_inf', '')}",
                        f"$VE_{{\\mathrm{{dur}}}}$={row.get('VE_dur', '')}",
                    )
                ),
                "source_provenance": vaccine_source.get(
                    scenario,
                    "Vaccine-mechanism scenario derived from manuscript_notes/scenario_table.csv and interpreted through figure S10 and table S17.",
                ),
                "fixed_or_conditioned": (
                    "All five vaccine settings branch from the same current-aP state at the "
                    "2027 policy origin; post-start mechanism parameters apply immediately "
                    "to existing and subsequent vaccine-origin compartments. Other natural-history, "
                    "contact, reporting, and resistance settings are held to the country baseline."
                ),
                "primary_role": row.get("description", "").strip(),
                "detail_location": "Figure S10 and table S17.",
            }
        )

    resistance_source = {
        "country_timeline": "Country-specific resistance anchors combined clinical guidance [21,22] with country reports from China [23,24], Australia [25], Japan [26], the Americas [27], and regional MRBP evidence [28,29]; raw evidence is tabulated in table S11 and parameter rationale in table S16.",
        "low": "Fixed prevalence stress-test anchored to observed low-prevalence settings and conservative imported-risk assumptions [21,27-29]; see tables S10, S11, and S16.",
        "moderate": "Fixed prevalence stress-test spanning plausible intermediate resistance pressure, using clinical guidance [21,22], China and Australia reports [23-25], Japan and Americas reports [26,27], and regional MRBP evidence [28,29]; see tables S10, S11, and S16.",
        "high": "Fixed prevalence stress-test motivated by high-prevalence MRBP reports in China [23,24], Japan [26], and regional evidence [28,29]; see tables S10, S11, and S16.",
        "very_high": "Upper prevalence stress-test motivated by near-fixation observations in China and high-prevalence Japanese clusters [23,24,26]; see tables S10, S11, and S16.",
        "country_timeline_fitness_cost": "Counterfactual fitness-cost sensitivity retained to bound traditional resistance-cost assumptions against observed MRBP expansion in China [23,24], Australia [25], Japan and the Americas [26,27], and regional reports [28,29].",
        "country_timeline_fitness_advantage": "Fitness-advantage sensitivity motivated by rapid MRBP expansion and international spread in China [23,24], Australia [25], Japan and the Americas [26,27], and regional reports [28,29], without a demonstrated transmission penalty.",
        "high_fitness_advantage": "Worst-case stress test combining high starting resistance with a fitness-advantaged strain; rationale summarized in table S16 and resistance evidence from China [23,24], Japan [26], and regional MRBP reports [28,29].",
    }
    for row in read_csv_rows("manuscript_notes/resistance_scenario_table.csv"):
        scenario = row.get("scenario", "")
        target_value = row.get("target_prevalence_at_analysis_start", "")
        importation_value = row.get("importation_fraction", "")
        if row.get("uses_country_resistance_timeline", "").strip().lower() in {"true", "yes", "1"}:
            target_value = "country-specific timeline estimate"
            importation_value = "country-specific timeline estimate"
        rows.append(
            {
                "analysis_component": "Macrolide-resistance scenario",
                "design_level": scenario,
                "parameter_settings": (
                    "target resistant fraction={target}; importation resistant fraction={importation}; "
                    "anchoring rate={anchor} per year; country timeline={timeline}; $f_R$={fitness}; "
                    "resistant treatment effect={treatment}; resistant PEP effectiveness={pep}"
                ).format(
                    target=target_value,
                    importation=importation_value,
                    anchor=row.get("prevalence_anchor_rate_per_year", ""),
                    timeline=row.get("uses_country_resistance_timeline", ""),
                    fitness=row.get("fitness_R", ""),
                    treatment=row.get("treatment_effect_resistant", ""),
                    pep=row.get("PEP_effectiveness_resistant", ""),
                ),
                "source_provenance": resistance_source.get(
                    scenario,
                    "Resistance scenario derived from manuscript_notes/resistance_scenario_table.csv; evidence and rationale in tables S11 and S16.",
                ),
                "fixed_or_conditioned": "Country-timeline anchors use latest admissible evidence through the 2025 analysis-year anchor; fixed scenarios provide low-to-very-high contrasts.",
                "primary_role": row.get("description", "").strip(),
                "detail_location": "tables S10, S11, and S16, and figures S9, S10, and S11.",
            }
        )

    intervention_source = {
        "current": "Country-specific schedule and coverage inputs from WHO/UNICEF and national records [1,14], with standard treatment/PEP assumptions from CDC guidance [20].",
        "higher_child_coverage": "Scenario modification of country routine childhood coverage using floor targets and country schedule inputs [1,14].",
        "adolescent_booster": "Scenario modification of booster timing/coverage using country schedule inputs and pertussis vaccine guidance [1,14].",
        "pregnancy_tdap_scaleup": "Pregnancy Tdap scale-up scenario informed by maternal-program evidence [10-12], WHO vaccine position-paper guidance [1], and infant-specific effectiveness estimates [40,41].",
        "cocooning_adjunct": "Close-contact adult immunization/contact-reduction adjunct interpreted as an implementation-dependent infant-exposure reduction proxy alongside pregnancy Tdap.",
        "maternal_immunization": "High-intensity infant-exposure pathway package combining pregnancy Tdap scale-up and a close-contact adult adjunct, with maternal and adult-source components decomposed in figure S5.",
        "targeted_pep_high_risk": "Targeted PEP scenario translated from CDC guidance prioritizing household contacts, infants, and high-risk infant settings [20].",
        "maternal_direct_antibody_only": "Component diagnostic based on maternal-program evidence [10-12] and infant-specific effectiveness estimates [40,41].",
        "maternal_adult_boosting_only": "Component diagnostic separating adult boosting from direct infant antibody and contact-reduction effects; informed by maternal-program interpretation [10-12] and infant-specific estimates [40,41].",
        "maternal_cocooning_only": "Component diagnostic for household/contact reduction, interpreted with maternal-program evidence [10-12] and infant-protection estimates [40,41].",
        "resistance_guided_treatment": "Resistance-guided management pathway translated from CDC treatment/PEP and antibiotic-resistance guidance [20,21].",
        "next_generation_vaccine": "Hypothetical product-target scenario interpreted through the WHO vaccine framework [1], transmission evidence [5,6], waning studies [7-9], and vaccine-pipeline mapping in table S17.",
        "combined_strategy": "Composite stress test combining pregnancy Tdap-based infant protection, close-contact adult adjuncts, adolescent boosting, targeted PEP, resistance-guided management, and transmission-blocking assumptions.",
    }
    for row in read_csv_rows("manuscript_notes/intervention_scenario_table.csv"):
        strategy = row.get("strategy", "")
        rows.append(
            {
                "analysis_component": "Intervention strategy scenario",
                "design_level": strategy,
                "parameter_settings": "{category}; {status}".format(
                    category=row.get("scenario_category", ""),
                    status=row.get("interpretive_status", ""),
                ),
                "source_provenance": intervention_source.get(
                    strategy,
                    "Intervention scenario derived from manuscript_notes/intervention_scenario_table.csv and detailed in table S9.",
                ),
                "fixed_or_conditioned": "Strategies are grouped by decision domain; costs, feasibility, equity weights, and implementation constraints are interpretive considerations.",
                "primary_role": row.get("description", "").strip(),
                "detail_location": "table S9 and figures S5-S7 and S10.",
            }
        )

    for row in read_csv_rows("manuscript_notes/reporting_scenario_table.csv"):
        settings = []
        multiplier = row.get("multiplier", "").strip()
        if multiplier:
            settings.append(f"overall multiplier={multiplier}")
        for label, key in (
            ("age multipliers", "uses_age_multipliers"),
            ("time variation", "uses_time_variation"),
        ):
            value = row.get(key, "").strip()
            if value:
                settings.append(f"{label}={value}")
        rows.append(
            {
                "analysis_component": "Observation and reporting sensitivity",
                "design_level": row.get("scenario", ""),
                "parameter_settings": "; ".join(settings),
                "source_provenance": "Reporting sensitivities are scenario perturbations around literature-informed reporting priors from notification-efficiency and serology studies [30,31], capture-recapture and cough-serology evidence [32,33], and active surveillance [34]; fitted probabilities are shown in table S13.",
                "fixed_or_conditioned": "Reporting scenarios perturb the observation process only; PEP activation uses a separate detection proxy.",
                "primary_role": "Separates surveillance completeness from true transmission and resistant-strain dynamics.",
                "detail_location": "supplementary methods and tables S7 and S13.",
            }
        )

    rows.extend(
        [
        {
            "analysis_component": "Vaccine-resistance interaction grids",
            "design_level": "$VE_{\\mathrm{inf}}$-only grid and continuous $f_R \\times VE_{\\mathrm{inf}}$ grid",
            "parameter_settings": "$f_R$ values 0.70-1.25; $VE_{\\mathrm{inf}}$ values 0.05-0.55; $VE_{\\mathrm{inf}}$-only thresholds also vary resistance prevalence anchors and resistant importation fraction.",
            "source_provenance": "Grid bounds combine vaccine-framework and transmission uncertainty [1], [5,6], waning uncertainty [7-9], resistance guidance [21,22], and country resistance evidence [23,24], [25], [26], [27], [28,29]; exact grid values are retained in manuscript_notes/fitness_grid_table.csv and visualized in figure S11.",
            "fixed_or_conditioned": "$VE_{\\mathrm{sus}}$ and $VE_{\\mathrm{dur}}$ held at grid-baseline values for $VE_{\\mathrm{inf}}$-only thresholds; country profiles remain calibrated.",
            "primary_role": "Identifies transmission-blocking thresholds and shows how resistant fitness modifies vaccine benefit.",
            "detail_location": "figures S10 and S11, and manuscript_notes/fitness_grid_table.csv.",
        },
        {
            "analysis_component": "Exploratory uncertainty and robustness diagnostics",
            "design_level": "Sensitivity screens and robustness diagnostics",
            "parameter_settings": "128-draw evidence-prior inverse-CDF Latin-hypercube screening; 128 selected-input joint strategy-ordering samples; routine timeliness, temporal, infant-contact, maternal-duration, treatment/PEP, event-scale, and stochastic toy diagnostics.",
            "source_provenance": "Designed as robustness diagnostics following immunisation-model reporting guidance [35], using evidence distributions and explicitly labelled weak or implementation priors documented in the retained uncertainty registry and summarised graphically in figure S6.",
            "fixed_or_conditioned": "Diagnostics are not full posterior or decision analyses; they support strategy-ordering and structural-robustness interpretation.",
            "primary_role": "Quantifies which assumptions threaten interpretation of age-stratified burden and strategy-ordering conclusions.",
            "detail_location": "figures S6, S7, and S10.",
        },
        {
            "analysis_component": "Semi-mechanistic discrepancy POMP validation",
            "design_level": "One-reporting-interval-ahead prequential prediction plus one-year block stress test",
            "parameter_settings": "Deterministic mechanistic offsets; irregular-time persistent, robust, and compound-Poisson-jump discrepancy candidates; exposure-scaled NB2 observations; 512 particles, 4096 predictive draws, and three Monte Carlo replicates for the formal 2023-26 folds.",
            "source_provenance": "Processed integer surveillance intervals, mechanistic scenario code, and prespecified baseline forecasts; input and output SHA-256 digests are retained in run metadata.",
            "fixed_or_conditioned": "Candidates are selected inside each outer fold from three preceding validation years, with country scores shrunk towards the panel score; no future evidence is used.",
            "primary_role": "Tests panel-level notification-index prediction and defines the boundary between validated short-horizon prediction and conditional long-horizon scenario projection.",
            "detail_location": "predictive POMP specification table and outputs/tables/panel_pomp_* artifacts.",
        },
        ]
    )
    return rows


FULL_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        number="S1",
        title="Country-specific population, surveillance, vaccination, and seasonal-forcing inputs.",
        source="manuscript_notes/country_profile_table.csv",
        columns=(
            "country",
            "total_population",
            "seasonal_phase",
            "seasonal_amplitude",
            "observed_mean_annual_reported_incidence_per_100k",
            "vaccine_product",
            "adolescent_booster",
            "maternal_program",
        ),
        labels=(
            "Country",
            "Population",
            "Seasonal phase",
            "Seasonal amplitude",
            "Mean reported incidence per 100k",
            "Vaccine product",
            "Adolescent booster",
            "Maternal program",
        ),
    ),
    TableSpec(
        number="S2",
        title="Study design matrix for scenario, sensitivity, and predictive-validation analyses.",
        source="analysis design summary derived from source scenario tables",
        rows=study_parameter_design_rows,
        columns=(
            "analysis_component",
            "design_level",
            "parameter_settings",
            "source_provenance",
            "fixed_or_conditioned",
            "primary_role",
            "detail_location",
        ),
        labels=(
            "Analysis component",
            "Design level",
            "Parameter settings",
            "Source/provenance",
            "Fixed or conditioned assumptions",
            "Primary role",
            "Detailed location",
        ),
    ),
    TableSpec(
        number="S3",
        title="Macrolide-resistance initialization, importation, and fitness assumptions.",
        source="manuscript_notes/resistance_scenario_table.csv",
        rows=resistance_scenario_rows,
        columns=(
            "scenario",
            "target_prevalence_at_analysis_start",
            "importation_fraction",
            "prevalence_anchor_rate_per_year",
            "uses_country_resistance_timeline",
            "fitness_R",
            "description",
        ),
        labels=(
            "Scenario",
            "Target resistant fraction",
            "Importation resistant fraction",
            "Anchor rate per year",
            "Country timeline",
            "$f_R$",
            "Description",
        ),
    ),
    TableSpec(
        number="S4",
        title="Intervention strategy definitions, modified control levers, and decision domain.",
        source="manuscript_notes/intervention_scenario_table.csv",
        columns=(
            "strategy",
            "scenario_category",
            "interpretive_status",
            "description",
            "modified_control_levers",
            "interpretation_note",
        ),
        labels=(
            "Strategy",
            "Scenario category",
            "Interpretive status",
            "Scenario definition",
            "Modified control levers",
            "Interpretation note",
        ),
    ),
    TableSpec(
        number="S5",
        title="Baseline model parameter values, admissible ranges, and evidence provenance.",
        source="manuscript_notes/parameter_table.csv",
        rows=parameter_table_rows,
        columns=(
            "parameter_group",
            "parameter",
            "description",
            "baseline_value",
            "range",
            "unit",
            "source_or_assumption",
            "used_in_sensitivity_analysis",
        ),
        labels=(
            "Parameter group",
            "Parameter",
            "Description",
            "Baseline value",
            "Range",
            "Unit",
            "Source or assumption",
            "Sensitivity",
        ),
    ),
    TableSpec(
        number="S6",
        title="Reporting-rate sensitivity scenarios used to probe surveillance uncertainty.",
        source="manuscript_notes/reporting_scenario_table.csv",
        columns=("scenario", "multiplier", "uses_age_multipliers", "uses_time_variation", "description"),
        labels=("Scenario", "Multiplier", "Age multipliers", "Time variation", "Description"),
    ),
    TableSpec(
        number="S7",
        title="Country-specific macrolide-resistance evidence used for resistance anchoring.",
        source="data/raw/country_resistance_timeline.csv",
        columns=("country", "iso3", "year", "sample_size", "resistant_fraction", "lower", "upper", "evidence_type", "source"),
        labels=("Country", "ISO3", "Year", "Sample size", "Resistant fraction", "Lower", "Upper", "Evidence type", "Source"),
        sort_by=("country", "year"),
    ),
    TableSpec(
        number="S8",
        title="Calibration acceptance, absolute-fit diagnostics, and fitted transmission parameters.",
        source="outputs/tables/calibration_all_countries.csv",
        columns=(
            "country",
            "calibration_accepted",
            "optimizer_success",
            "absolute_fit_status",
            "observed_mean_annual_reported_incidence_per_100k",
            "annualized_reported_cases_per_100k",
            "model_to_observed_reported_incidence_ratio",
            "data_fit_score",
            "fit_score",
            "calibrated_beta",
        ),
        labels=(
            "Country",
            "Accepted",
            "Optimizer success",
            "Fit status",
            "Observed reported incidence per 100k",
            "Model reported incidence per 100k",
            "Model/observed ratio",
            "Data fit score",
            "Fit score",
            "Calibrated beta",
        ),
        sort_by=("country",),
    ),
    TableSpec(
        number="S9",
        title="Country-differentiated intervention outcome summaries by country and strategy.",
        source="outputs/summaries/intervention_scenarios_summary.csv",
        columns=(
            "country",
            "scenario",
            "total_infections",
            "total_reported_cases",
            "total_infant_cases",
            "resistant_infections",
            "relative_reduction_infant_cases",
            "relative_reduction_total_infections",
        ),
        labels=(
            "Country",
            "Strategy",
            "Total infections",
            "Reported cases",
            "Infant cases",
            "Resistant infections",
            "Infant-case reduction",
            "Infection reduction",
        ),
        sort_by=("country", "scenario"),
    ),
    TableSpec(
        number="S10",
        title="Model-derived outcomes, denominators, and comparative summary definitions.",
        source="static outcome definitions",
        rows=outcome_definition_rows,
        columns=("quantity", "definition", "denominator", "primary_use"),
        labels=("Quantity", "Definition", "Denominator or reference population", "Primary use"),
    ),
    TableSpec(
        number="S11",
        title="Core model structure, observation model, and implementation settings.",
        source="configuration summary derived from the analysis pipeline",
        rows=fixed_model_setting_rows,
        columns=("aspect", "setting", "value"),
        labels=("Aspect", "Setting", "Value"),
    ),
    TableSpec(
        number="S12",
        title="Semi-mechanistic POMP predictive-validation specification and claim controls.",
        source="formal panel-POMP runner specification and run metadata",
        rows=pomp_validation_parameter_rows,
        columns=(
            "parameter_group",
            "parameter",
            "value_or_center",
            "uncertainty_range",
            "unit",
            "distribution",
            "analysis_role",
            "interpretation",
        ),
        labels=(
            "Parameter group",
            "Parameter",
            "Value or centre",
            "Range",
            "Unit",
            "Prior or sampling distribution",
            "Validation role",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S13",
        title="Continuous macrolide-resistant fitness and vaccine infectiousness grid.",
        source="manuscript_notes/fitness_grid_table.csv",
        columns=("fitness_R", "VE_inf", "description"),
        labels=("$f_R$", "$VE_{\\mathrm{inf}}$", "Description"),
    ),
    TableSpec(
        number="S14",
        title="Selected prior pertussis models and mechanistic features compared with the current model.",
        source="static prior model comparison",
        rows=prior_model_comparison_rows,
        columns=(
            "prior_model",
            "age_structure",
            "waning",
            "asymptomatic_transmission",
            "vaccine_infection_blocking",
            "vaccine_infectiousness_reduction",
            "resistance",
            "treatment_pep",
            "infant_specific_outcome",
        ),
        labels=(
            "Model",
            "Age structure",
            "Waning",
            "Asymptomatic transmission",
            "Vaccine infection blocking",
            "Vaccine infectiousness reduction",
            "Resistance",
            "Treatment/PEP",
            "Infant-specific outcome",
        ),
    ),
    TableSpec(
        number="S15",
        title="Policy-diverse country selection rationale, programmatic dimensions, and data-quality rating.",
        source="data/processed/country_profile_inputs.csv plus manuscript_notes/country_profile_table.csv",
        rows=country_selection_rows,
        columns=(
            "country",
            "who_region",
            "population",
            "dtp3_coverage",
            "booster_schedule",
            "maternal_vaccination_policy",
            "recent_reported_incidence",
            "resistance_anchor",
            "reason_for_inclusion",
            "data_quality_rating",
        ),
        labels=(
            "Country",
            "WHO region",
            "Population",
            "DTP3 coverage",
            "Booster schedule",
            "Maternal vaccination policy",
            "Recent reported incidence per 100k",
            "Resistance anchor",
            "Reason for inclusion",
            "Data quality",
        ),
    ),
    TableSpec(
        number="S16",
        title="Fitted age-specific reporting probabilities and country-specific prior bounds.",
        source="outputs/tables/calibration_all_countries.csv",
        rows=reporting_probability_rows,
        columns=(
            "country",
            "infant_0_2m_reporting_probability",
            "infant_3_11m_reporting_probability",
            "child_1_4y_reporting_probability",
            "school_adolescent_5_17y_reporting_probability",
            "adult_18plus_reporting_probability",
            "prior_bounds",
            "calibrated_value_source",
        ),
        labels=(
            "Country",
            "Infant 0-2 mo",
            "Infant 3-11 mo",
            "Child 1-4 y",
            "School/adolescent 5-17 y",
            "Adult 18+ y",
            "Prior bounds by age",
            "Evidence",
        ),
        sort_by=("country",),
        column_styles=(
            "",
            "",
            "",
            "",
            "",
            "",
            "display:inline-block; min-width:26em; white-space:normal;",
            "display:inline-block; max-width:8em; white-space:normal;",
        ),
    ),
    TableSpec(
        number="S17",
        title="Macrolide-resistance mechanism decomposition across importation, treatment, PEP, and fitness assumptions.",
        source="outputs/tables/resistance_mechanism_decomposition.csv",
        columns=(
            "scenario",
            "importation",
            "treatment_differential",
            "pep_differential",
            "fitness_R",
            "median_end_resistant_fraction",
            "iqr_end_resistant_fraction",
            "median_infant_cases_per_100k",
            "median_resistant_infections_per_100k",
            "interpretation",
        ),
        labels=(
            "Scenario",
            "Importation",
            "Treatment differential",
            "PEP differential",
            "$f_R$",
            "Median end resistant fraction",
            "Across-profile IQR end resistant fraction",
            "Median infant cases per 100k",
            "Median resistant infections per 100k",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S18",
        title="Threshold analysis for vaccine infectiousness reduction under selected resistant-fitness assumptions.",
        source="outputs/summaries/fitness_resistance_grid_summary.csv",
        rows=veinf_threshold_rows,
        columns=(
            "fitness_R",
            "target_reduction",
            "minimum_VE_inf",
            "median_reduction_at_minimum",
            "countries_meeting_threshold",
            "interpretation",
        ),
        labels=(
            "$f_R$",
            "Target infant-case reduction vs $VE_{\\mathrm{inf}}$=0.25",
            "Minimum $VE_{\\mathrm{inf}}$",
            "Median reduction at threshold",
            "Countries meeting threshold",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S19",
        title="Calibration diagnostics by country and calibration period.",
        source="outputs/tables/calibration_fit_diagnostics_summary.csv",
        columns=(
            "country",
            "period",
            "n_intervals",
            "observed_total_reported_cases",
            "predicted_total_reported_cases",
            "mean_absolute_percentage_error",
            "peak_observed_year",
            "peak_predicted_year",
            "peak_timing_error_years",
            "peak_magnitude_ratio",
        ),
        labels=(
            "Country",
            "Period",
            "Intervals",
            "Observed reported cases",
            "Modeled reported cases",
            "Conventional interval MAPE",
            "Observed peak year",
            "Modeled peak year",
            "Peak timing error, y",
            "Peak magnitude ratio",
        ),
        sort_by=("country", "period"),
    ),
    TableSpec(
        number="S20",
        title="Resistance-management mechanism decomposition and near-term implementation sensitivity.",
        source="outputs/tables/resistance_management_policy_decomposition.csv",
        columns=(
            "analysis_layer",
            "policy_read",
            "analysis_window",
            "treatment_component",
            "pep_component",
            "testing_or_uptake_component",
            "median_infant_case_reduction_vs_current",
            "iqr_infant_case_reduction_vs_current",
            "countries_with_positive_reduction",
            "median_infant_cases_per_100k",
            "median_resistance_metric",
            "end_resistant_fraction",
            "interpretation",
        ),
        labels=(
            "Analysis layer",
            "Policy read",
            "Analysis window",
            "Treatment component",
            "PEP component",
            "Testing/uptake component",
            "Median infant-case reduction vs current",
            "Across-profile IQR reduction",
            "Countries with positive reduction",
            "Median infant cases per 100k",
            "Median resistant infections per 100k",
            "End resistant fraction",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S21",
        title="Near-term infant contact-matrix sensitivity for current practice and pregnancy vaccination plus adult/household transmission-reduction proxies.",
        source="outputs/tables/infant_contact_sensitivity.csv",
        columns=(
            "strategy",
            "infant_contact_multiplier",
            "median_infant_cases_per_100k",
            "iqr_infant_cases_per_100k",
            "median_all_infections_per_100k",
            "countries",
            "interpretation",
        ),
        labels=(
            "Strategy",
            "Infant-contact multiplier",
            "Median infant cases per 100k",
            "Across-profile IQR infant cases per 100k",
            "Median all infections per 100k",
            "Countries",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S22",
        title="Country-specific mechanism diagnostic for the higher child-coverage scenario.",
        source="outputs/tables/higher_child_coverage_country_mechanism.csv",
        columns=(
            "country",
            "current_infant_cases_per_100k",
            "higher_child_coverage_infant_cases_per_100k",
            "relative_reduction_infant_cases",
            "largest_absolute_infection_increase_age_group",
            "largest_absolute_infection_increase",
            "relative_change_in_that_age_group",
        ),
        labels=(
            "Country",
            "Current infant cases per 100k",
            "Nominal floor, no timeliness infant cases per 100k",
            "Relative infant-case reduction",
            "Largest infection-increase age group",
            "Largest absolute infection increase",
            "Relative change in that age group",
        ),
        sort_by=("country",),
    ),
    TableSpec(
        number="S23",
        title="Age-specific infection shifts under the higher child-coverage scenario.",
        source="outputs/tables/higher_child_coverage_age_shift.csv",
        columns=(
            "age_group",
            "median_relative_change_infections",
            "q25_relative_change_infections",
            "q75_relative_change_infections",
            "median_relative_change_symptomatic_cases",
            "countries_with_infection_increase",
        ),
        labels=(
            "Age group",
            "Median infection change",
            "Q25 infection change",
            "Q75 infection change",
            "Median symptomatic-case change",
            "Countries with infection increase",
        ),
    ),
    TableSpec(
        number="S24",
        title=(
            "Empirical intervention scenario-ordering robustness across the nine "
            "calibrated publication profiles; South Africa is exploratory only."
        ),
        source="outputs/tables/intervention_rank_robustness.csv",
        columns=(
            "scenario",
            "median_rank",
            "min_rank",
            "max_rank",
            "countries_ranked_first",
            "countries_within_10_percent_of_best",
            "median_infant_cases_per_100k",
            "rank_basis",
        ),
        labels=(
            "Scenario",
            "Median order position",
            "Minimum order position",
            "Maximum order position",
            "Countries ordered first",
            "Countries within 10% of best",
            "Median infant cases per 100k",
            "Ordering basis",
        ),
    ),
    TableSpec(
        number="S25",
        title="Intervention scenario-ordering sensitivity to analysis-window choice.",
        source="outputs/tables/intervention_horizon_rank_summary.csv",
        columns=(
            "analysis_window",
            "scenario",
            "median_rank",
            "countries_ranked_first",
            "median_relative_reduction_infant_cases",
        ),
        labels=(
            "Analysis window",
            "Scenario",
            "Median order position",
            "Countries ordered first",
            "Median infant-case reduction",
        ),
    ),
    TableSpec(
        number="S26",
        title="Pearson, Spearman, and PRCC parameter-sensitivity screening correlations; marginal design and sample count are reported in each row.",
        source="outputs/tables/sensitivity_correlation_screening.csv",
        columns=(
            "parameter",
            "outcome",
            "distribution",
            "evidence_class",
            "pearson_r",
            "spearman_r",
            "partial_rank_correlation",
            "screening_note",
        ),
        labels=(
            "Parameter",
            "Outcome",
            "Distribution",
            "Evidence class",
            "Pearson r",
            "Spearman r",
            "PRCC",
            "Screening note",
        ),
    ),
    TableSpec(
        number="S27",
        title="$VE_{\\mathrm{inf}}$ threshold diagnostics against intervention comparators and target reductions.",
        source="outputs/tables/veinf_comparator_thresholds.csv",
        columns=(
            "comparator",
            "resistance_prevalence",
            "median_minimum_VE_inf",
            "countries_reaching_comparator",
            "countries_evaluated",
            "threshold_basis",
        ),
        labels=(
            "Comparator",
            "Starting resistance prevalence",
            "Median minimum $VE_{\\mathrm{inf}}$",
            "Countries reaching comparator",
            "Countries evaluated",
            "Threshold basis",
        ),
    ),
    TableSpec(
        number="S28",
        title="Infant infection vaccine-history origin shares under current and higher child-coverage scenarios.",
        source="outputs/tables/infant_vaccine_history_origin_shares.csv",
        columns=(
            "country",
            "scenario",
            "infant_infections",
            "vaccinated_origin_infection_share",
            "waned_origin_infection_share",
            "maternal_origin_infection_share",
            "dose1_origin_infection_share",
            "dose2_origin_infection_share",
            "dose3plus_origin_infection_share",
        ),
        labels=(
            "Country",
            "Scenario",
            "Infant infections",
            "Vaccinated-origin share",
            "Waned-origin share",
            "Maternal-origin share",
            "Dose-1 share",
            "Dose-2 share",
            "Dose-3-plus share",
        ),
        sort_by=("country", "scenario"),
    ),
    TableSpec(
        number="S29",
        title="Near-term temporal assumption sensitivity for burn-in duration and COVID-19 NPI contact-shock assumptions.",
        source="outputs/tables/temporal_assumption_sensitivity.csv",
        columns=(
            "temporal_dimension",
            "scenario",
            "countries",
            "burn_in_years",
            "npi_reduction_scale",
            "median_infant_cases_per_100k_5y",
            "iqr_infant_cases_per_100k_5y",
            "median_all_infections_per_100k_5y",
            "median_end_resistant_fraction_5y",
            "implementation_note",
        ),
        labels=(
            "Temporal dimension",
            "Scenario",
            "Countries",
            "Burn-in years",
            "NPI reduction scale",
            "Median infant cases per 100k, 5 y",
            "Across-profile IQR infant cases per 100k, 5 y",
            "Median all infections per 100k, 5 y",
            "Median end resistant fraction, 5 y",
            "Implementation note",
        ),
    ),
    TableSpec(
        number="S30",
        title="Infant age-stratified intervention outcomes across the 0-2 month and 3-11 month strata.",
        source="outputs/tables/infant_age_split_intervention.csv",
        columns=(
            "country",
            "age_group",
            "scenario",
            "annualized_infant_cases_per_100k",
            "annualized_infant_infections_per_100k",
            "relative_reduction_infant_cases",
        ),
        labels=(
            "Country",
            "Infant age stratum",
            "Scenario",
            "Infant cases per 100k/y",
            "Infant infections per 100k/y",
            "Infant-case reduction",
        ),
        sort_by=("country", "age_group", "scenario"),
    ),
    TableSpec(
        number="S31",
        title="Infant age-stratified intervention scenario-ordering sensitivity by analysis window.",
        source="outputs/tables/infant_age_split_horizon_sensitivity.csv",
        columns=(
            "analysis_window",
            "country",
            "age_group",
            "scenario",
            "rank",
            "annualized_infant_cases_per_100k",
            "relative_reduction_infant_cases",
        ),
        labels=(
            "Analysis window",
            "Country",
            "Infant age stratum",
            "Scenario",
            "Order position",
            "Infant cases per 100k/y",
            "Infant-case reduction",
        ),
        sort_by=("analysis_window", "country", "age_group", "rank", "scenario"),
    ),
    TableSpec(
        number="S32",
        title="Infant intervention conditional-scenario point-estimate audit.",
        source="outputs/tables/figure2b_intervention_scenario_point_audit.csv",
        columns=(
            "country",
            "scenario_key",
            "scenario_label",
            "relative_reduction_infant_cases",
            "annualized_infant_cases_per_100k",
            "estimate_type",
            "uncertainty_interval_reported",
            "claim_scope",
        ),
        labels=(
            "Country",
            "Scenario key",
            "Scenario label",
            "Point reduction",
            "Infant cases per 100k/y",
            "Estimate type",
            "Uncertainty interval reported",
            "Claim scope",
        ),
        sort_by=("country", "scenario_key"),
    ),
    TableSpec(
        number="S33",
        title="Cross-diagnostic intervention scenario-ordering stability across countries, analysis windows, and infant age strata.",
        source="outputs/tables/intervention_rank_stability_diagnostics.csv",
        columns=(
            "scenario",
            "full_horizon_median_rank",
            "full_horizon_countries_ranked_first",
            "full_horizon_countries_ranked_top_two",
            "analysis_window_cells_ranked_first",
            "analysis_window_cells_ranked_top_two",
            "infant_age_window_cells_ranked_first",
            "infant_age_window_cells_ranked_top_two",
            "infant_age_window_cells_positive_reduction",
            "median_infant_age_window_reduction",
            "rank_stability_interpretation",
        ),
        labels=(
            "Scenario",
            "Full-horizon median order position",
            "Countries ordered first",
            "Countries ordered top 2",
            "Window cells ordered first",
            "Window cells ordered top 2",
            "Age-window cells ordered first",
            "Age-window cells ordered top 2",
            "Age-window cells with reduction",
            "Median age-window reduction",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S34",
        title="Deterministic event-scale diagnostics for stochastic-interpretation sensitivity.",
        source="outputs/tables/deterministic_event_scale_diagnostics.csv",
        columns=(
            "country",
            "scenario",
            "annual_total_infections_count",
            "annual_reported_cases_count",
            "annual_infant_cases_count",
            "annualized_infant_cases_per_100k",
            "n_epidemic_peaks",
            "resistant_fraction_end",
            "event_scale_flag",
        ),
        labels=(
            "Country",
            "Scenario",
            "Annual total infections",
            "Annual reported cases",
            "Annual infant cases",
            "Infant cases per 100k/y",
            "Epidemic peaks",
            "End resistant fraction",
            "Event-scale flag",
        ),
    ),
    TableSpec(
        number="S35",
        title="Limitation-to-diagnostic map and residual interpretation.",
        source="outputs/tables/limitation_diagnostic_map.csv",
        columns=(
            "limitation_domain",
            "added_or_existing_diagnostic",
            "supplement_location",
            "residual_interpretation",
        ),
        labels=(
            "Limitation domain",
            "Added or existing diagnostic",
            "Supplement location",
            "Residual interpretation",
        ),
    ),
    TableSpec(
        number="S36",
        title="Exploratory health-utility burden scenarios derived from model deaths and symptomatic cases.",
        source="outputs/tables/intervention_health_utility_loss_summary.csv",
        columns=(
            "utility_scenario",
            "scenario",
            "country_count",
            "infant_life_years_lost_per_death",
            "acute_illness_disutility",
            "acute_illness_duration_days",
            "infant_hospitalization_probability_per_case",
            "noninfant_hospitalization_probability_per_case",
            "hospitalization_excess_disutility",
            "hospitalization_duration_days",
            "median_total_qaly_like_loss_per_100k_year",
            "q25_total_qaly_like_loss_per_100k_year",
            "q75_total_qaly_like_loss_per_100k_year",
            "median_qaly_like_loss_averted_vs_current_per_100k_year",
            "median_relative_qaly_like_loss_reduction_vs_current",
            "median_infant_mortality_life_year_share",
            "median_acute_illness_share",
            "median_hospitalization_share",
            "interpretation_note",
        ),
        labels=(
            "Utility scenario",
            "Intervention scenario",
            "Countries",
            "Infant LY lost/death",
            "Acute disutility",
            "Acute days",
            "Infant hospitalization probability",
            "Noninfant hospitalization probability",
            "Hospitalization excess disutility",
            "Hospitalization days",
            "Median modeled utility loss per 100k/y",
            "Q25 modeled utility loss per 100k/y",
            "Q75 modeled utility loss per 100k/y",
            "Median modeled utility loss averted per 100k/y",
            "Median relative loss reduction",
            "Median infant mortality share",
            "Median acute illness share",
            "Median hospitalization share",
            "Interpretation note",
        ),
        sort_by=("utility_scenario", "scenario"),
    ),
    TableSpec(
        number="S37",
        title="Secondary infant-case ordering frequencies across the selected deterministic input design.",
        source="outputs/tables/joint_psa_rank_acceptability.csv",
        rows=joint_psa_acceptability_rows,
        columns=(
            "country",
            "strategy",
            "rank",
            "rank_acceptability_probability",
            "probability_rank_1",
            "probability_top_2",
            "probability_within_10_percent_of_best",
            "mean_rank",
            "median_rank",
            "median_infant_cases_per_100k",
            "q025_infant_cases_per_100k",
            "q975_infant_cases_per_100k",
            "median_relative_reduction_vs_current",
            "n_psa_samples",
            "n_rank_observations",
        ),
        labels=(
            "Country",
            "Strategy",
            "Order position",
            "Design frequency at order position",
            "Frequency lowest burden",
            "Frequency among two lowest burdens",
            "Frequency within 10% of best",
            "Mean order position",
            "Median order position",
            "Median infant cases per 100k/y",
            "Q2.5 infant cases per 100k/y",
            "Q97.5 infant cases per 100k/y",
            "Median reduction vs current",
            "Sensitivity samples",
            "Order observations",
        ),
        sort_by=("country", "rank", "strategy"),
    ),
    TableSpec(
        number="S38",
        title="Selected-input deterministic sensitivity sampled parameter sets.",
        source="outputs/tables/joint_psa_parameter_samples.csv",
        rows=joint_psa_parameter_rows,
        columns=(
            "psa_sample_id",
            "sample_design",
            "uncertainty_schema_version",
            "infant_contact_multiplier",
            "VE_inf_baseline",
            "relative_infectiousness_asymptomatic",
            "infectious_duration_asymptomatic",
            "fitness_R",
            "PEP_coverage_multiplier",
        ),
        labels=(
            "Sample",
            "Design",
            "Uncertainty schema",
            "Infant contact multiplier",
            "Baseline $VE_{\\mathrm{inf}}$",
            "Asymptomatic infectiousness",
            "Asymptomatic duration",
            "$f_R$",
            "PEP coverage multiplier",
        ),
        sort_by=("psa_sample_id",),
    ),
    TableSpec(
        number="S39",
        title="Individual stochastic contact-clustering toy model summary.",
        source="outputs/tables/individual_stochastic_toy_summary.csv",
        columns=(
            "country",
            "scenario",
            "n_replicates",
            "population_size",
            "target_reproduction_number",
            "setting_matrix_available",
            "extinction_probability_3_or_fewer",
            "outbreak_probability_20plus",
            "median_total_infections",
            "q025_total_infections",
            "q975_total_infections",
            "mean_infant_infections",
            "probability_any_infant_infection",
            "q95_infant_infections",
            "structural_sensitivity_caveat",
        ),
        labels=(
            "Country",
            "Scenario",
            "Replicates",
            "Synthetic population size",
            "Target R",
            "Setting matrix available",
            "Pr(extinction <=3 infections)",
            "Pr(outbreak >=20 infections)",
            "Median total infections",
            "Q2.5 total infections",
            "Q97.5 total infections",
            "Mean infant infections",
            "Pr(any infant infection)",
            "Q95 infant infections",
            "Caveat",
        ),
        sort_by=("country", "scenario"),
    ),
    TableSpec(
        number="S40",
        title="Contact-data audit for the individual stochastic toy model.",
        source="outputs/tables/individual_stochastic_toy_contact_audit.csv",
        columns=(
            "country",
            "raw_rows_for_country",
            "raw_setting_matrix_available",
            "processed_setting_matrix_available",
            "country_profile_contact_groups",
            "contact_source",
            "toy_model_setting_use",
            "structural_sensitivity_caveat",
        ),
        labels=(
            "Country",
            "Raw rows",
            "Raw setting matrix available",
            "Processed setting matrix available",
            "Profile contact groups",
            "Contact source",
            "Toy-model setting use",
            "Caveat",
        ),
        sort_by=("country",),
    ),
    TableSpec(
        number="S41",
        title="Vaccine-pipeline mechanism mapping to modeled scenario profiles.",
        source="outputs/tables/vaccine_pipeline_mechanism_mapping.csv",
        columns=(
            "candidate_or_platform",
            "route_or_platform",
            "development_status_as_of_2026_05_21",
            "mechanistic_relevance",
            "model_representation",
            "reason_not_modeled_as_available_policy",
            "evidence_source",
        ),
        labels=(
            "Candidate or platform",
            "Route/platform",
            "Development status",
            "Mechanistic relevance",
            "Model representation",
            "Reason not modeled as available policy",
            "Evidence source",
        ),
    ),
    TableSpec(
        number="S42",
        title="Macrolide-resistance parameter justification and expected direction of bias.",
        source="outputs/tables/resistance_parameter_justification.csv",
        columns=(
            "parameter_group",
            "baseline_value",
            "explored_range_or_scenarios",
            "source_or_anchor",
            "rationale",
            "expected_direction_of_bias",
            "residual_caveat",
        ),
        labels=(
            "Parameter group",
            "Baseline value",
            "Explored range or scenarios",
            "Source or anchor",
            "Rationale",
            "Expected direction of bias",
            "Residual caveat",
        ),
    ),
    TableSpec(
        number="S43",
        title="Near-term maternal passive-protection duration sensitivity.",
        source="outputs/tables/maternal_duration_sensitivity.csv",
        columns=(
            "strategy",
            "maternal_protection_duration_days",
            "median_infant_cases_per_100k_5y",
            "iqr_infant_cases_per_100k_5y",
            "median_infant_case_reduction_vs_current_5y",
            "iqr_infant_case_reduction_vs_current_5y",
            "countries_with_positive_reduction",
            "countries",
            "interpretation",
        ),
        labels=(
            "Strategy",
            "Maternal protection duration, d",
            "Median infant cases per 100k, 5 y",
            "Across-profile IQR infant cases per 100k, 5 y",
            "Median infant-case reduction vs current, 5 y",
            "Across-profile IQR reduction",
            "Countries with positive reduction",
            "Countries",
            "Interpretation",
        ),
    ),
)


# Submitted-appendix table set. FULL_TABLES is retained for traceability, but
# the active appendix uses this compressed set to
# avoid listing long audit tables that are better kept as supplementary source-data files.
TABLES = (
    TableSpec(
        number="S1",
        title="Country-profile selection rationale, program features, and resistance anchors.",
        source="data/processed/country_profile_inputs.csv plus manuscript_notes/country_profile_table.csv",
        rows=country_inputs_selection_rows,
        columns=(
            "country",
            "who_region",
            "program_profile",
            "resistance_anchor",
            "data_quality_rating",
            "reason_for_inclusion",
        ),
        labels=(
            "Country",
            "WHO region",
            "Program profile",
            "Resistance anchor",
            "Data quality",
            "Reason for inclusion",
        ),
    ),
    TableSpec(
        number="S2",
        title="Study design matrix for scenario, sensitivity, and predictive-validation analyses.",
        source="analysis design summary derived from source scenario tables",
        rows=study_parameter_design_rows,
        columns=(
            "analysis_component",
            "design_level",
            "parameter_settings",
            "source_provenance",
            "fixed_or_conditioned",
            "primary_role",
            "detail_location",
        ),
        labels=(
            "Analysis component",
            "Design level",
            "Parameter settings",
            "Source/provenance",
            "Fixed or conditioned assumptions",
            "Primary role",
            "Detailed location",
        ),
    ),
    TableSpec(
        number="S3",
        title="Macrolide-resistance scenario assumptions for starting fraction, importation, and strain fitness.",
        source="manuscript_notes/resistance_scenario_table.csv",
        rows=resistance_scenario_rows,
        columns=(
            "scenario",
            "target_prevalence_at_analysis_start",
            "importation_fraction",
            "prevalence_anchor_rate_per_year",
            "uses_country_resistance_timeline",
            "fitness_R",
            "description",
        ),
        labels=(
            "Scenario",
            "Target resistant fraction",
            "Importation resistant fraction",
            "Anchor rate per year",
            "Country timeline",
            "$f_R$",
            "Description",
        ),
    ),
    TableSpec(
        number="S4",
        title="Intervention strategy definitions, decision domains, and modified control levers.",
        source="manuscript_notes/intervention_scenario_table.csv",
        columns=(
            "strategy",
            "scenario_category",
            "interpretive_status",
            "description",
            "modified_control_levers",
            "interpretation_note",
        ),
        labels=(
            "Strategy",
            "Scenario category",
            "Interpretive status",
            "Scenario definition",
            "Modified control levers",
            "Interpretation note",
        ),
    ),
    TableSpec(
        number="S5",
        title="Baseline model parameter values, admissible ranges, and evidence provenance.",
        source="manuscript_notes/parameter_table.csv",
        rows=parameter_table_rows,
        columns=(
            "parameter_group",
            "parameter",
            "description",
            "baseline_value",
            "range",
            "unit",
            "source_or_assumption",
            "used_in_sensitivity_analysis",
        ),
        labels=(
            "Parameter group",
            "Parameter",
            "Description",
            "Baseline value",
            "Range",
            "Unit",
            "Source or assumption",
            "Sensitivity",
        ),
    ),
    TableSpec(
        number="S6",
        title="Country-specific macrolide-resistance evidence used to assign starting anchors.",
        source="data/raw/country_resistance_timeline.csv",
        rows=resistance_evidence_rows,
        columns=(
            "country",
            "year",
            "sample_size_display",
            "resistant_fraction_interval",
            "evidence_label",
            "source_display",
            "notes_display",
        ),
        labels=("Country", "Year", "Sample size", "Resistance estimate", "Evidence class", "Source", "Note"),
        sort_by=("country", "year"),
    ),
    TableSpec(
        number="S7",
        title="Calibration acceptance, fitted transmission parameters, and reported-case fit diagnostics.",
        source="outputs/tables/calibration_all_countries.csv plus calibration_fit_diagnostics_summary.csv",
        rows=calibration_diagnostic_rows,
        columns=(
            "country",
            "period",
            "calibration_accepted",
            "absolute_fit_status",
            "calibrated_beta",
            "observed_mean_annual_reported_incidence_per_100k",
            "annualized_reported_cases_per_100k",
            "model_to_observed_reported_incidence_ratio",
            "n_intervals",
            "observed_total_reported_cases",
            "predicted_total_reported_cases",
            "mean_absolute_percentage_error",
            "peak_observed_year",
            "peak_predicted_year",
            "peak_timing_error_years",
            "peak_magnitude_ratio",
        ),
        labels=(
            "Country",
            "Period",
            "Accepted",
            "Fit status",
            "Calibrated beta",
            "Observed incidence per 100k",
            "Modeled incidence per 100k",
            "Model/observed ratio",
            "Intervals",
            "Observed reports",
            "Modeled reports",
            "Conventional interval MAPE",
            "Observed peak year",
            "Modeled peak year",
            "Peak timing error, y",
            "Peak magnitude ratio",
        ),
        sort_by=("country", "period"),
    ),
    TableSpec(
        number="S8",
        title="Model-derived outcomes, denominators, and comparative summary definitions.",
        source="static outcome definitions",
        rows=outcome_definition_rows,
        columns=("quantity", "definition", "denominator", "primary_use"),
        labels=("Quantity", "Definition", "Denominator or reference population", "Primary use"),
    ),
    TableSpec(
        number="S9",
        title="Core model structure, observation model, and implementation settings.",
        source="configuration summary derived from the analysis pipeline",
        rows=fixed_model_setting_rows,
        columns=("aspect", "setting", "value"),
        labels=("Aspect", "Setting", "Value"),
    ),
    TableSpec(
        number="S10",
        title="Semi-mechanistic POMP predictive-validation specification and claim controls.",
        source="formal panel-POMP runner specification and run metadata",
        rows=pomp_validation_parameter_rows,
        columns=(
            "parameter_group",
            "parameter",
            "value_or_center",
            "uncertainty_range",
            "unit",
            "distribution",
            "analysis_role",
            "interpretation",
        ),
        labels=(
            "Parameter group",
            "Parameter",
            "Value or centre",
            "Range",
            "Unit",
            "Prior or sampling distribution",
            "Validation role",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S11",
        title="Condensed macrolide-resistant fitness and vaccine infectiousness grid definition.",
        source="manuscript_notes/fitness_grid_table.csv",
        rows=fitness_grid_summary_rows,
        columns=("dimension", "grid_values", "selected_contrasts", "interpretation"),
        labels=("Dimension", "Grid values", "Selected contrasts", "Interpretation"),
    ),
    TableSpec(
        number="S12",
        title="Fitted age-specific reporting probabilities and country-specific prior bounds.",
        source="outputs/tables/calibration_all_countries.csv",
        rows=reporting_probability_rows,
        columns=(
            "country",
            "infant_0_2m_reporting_probability",
            "infant_3_11m_reporting_probability",
            "child_1_4y_reporting_probability",
            "school_adolescent_5_17y_reporting_probability",
            "adult_18plus_reporting_probability",
            "prior_bounds",
            "calibrated_value_source",
        ),
        labels=(
            "Country",
            "0-2 mo",
            "3-11 mo",
            "1-4 y",
            "5-17 y",
            "18+ y",
            "Prior bounds",
            "Evidence",
        ),
        sort_by=("country",),
        column_styles=(
            "",
            "",
            "",
            "",
            "",
            "",
            "display:inline-block; min-width:26em; white-space:normal;",
            "display:inline-block; max-width:8em; white-space:normal;",
        ),
    ),
    TableSpec(
        number="S13",
        title="Macrolide-resistance mechanism decomposition across importation, treatment, PEP, and fitness assumptions.",
        source="outputs/tables/resistance_mechanism_decomposition.csv",
        columns=(
            "scenario",
            "importation",
            "treatment_differential",
            "pep_differential",
            "fitness_R",
            "median_end_resistant_fraction",
            "iqr_end_resistant_fraction",
            "median_infant_cases_per_100k",
            "median_resistant_infections_per_100k",
            "interpretation",
        ),
        labels=(
            "Scenario",
            "Importation",
            "Treatment differential",
            "PEP differential",
            "$f_R$",
            "Median end resistant fraction",
            "Across-profile IQR end resistant fraction",
            "Median infant cases per 100k",
            "Median resistant infections per 100k",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S14",
        title="Vaccine infectiousness-effect threshold diagnostics.",
        source="outputs/summaries/fitness_resistance_grid_summary.csv plus outputs/tables/veinf_comparator_thresholds.csv",
        rows=vaccine_threshold_rows,
        columns=(
            "threshold_type",
            "fitness_or_comparator",
            "resistance_prevalence",
            "target_or_comparator",
            "minimum_VE_inf",
            "countries",
            "interpretation",
        ),
        labels=(
            "Threshold type",
            "Fitness or comparator",
            "Resistance prevalence",
            "Target or comparator basis",
            "Minimum $VE_{\\mathrm{inf}}$",
            "Countries",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S15",
        title="Country-differentiated intervention outcome summaries by country and strategy.",
        source="outputs/summaries/intervention_scenarios_summary.csv",
        columns=(
            "country",
            "scenario",
            "total_infections",
            "total_reported_cases",
            "total_infant_cases",
            "resistant_infections",
            "relative_reduction_infant_cases",
            "relative_reduction_total_infections",
        ),
        labels=(
            "Country",
            "Strategy",
            "Total infections",
            "Reported cases",
            "Infant cases",
            "Resistant infections",
            "Infant-case reduction",
            "Infection reduction",
        ),
        sort_by=("country", "scenario"),
    ),
    TableSpec(
        number="S16",
        title="Resistance-management mechanism decomposition and near-term implementation sensitivity.",
        source="outputs/tables/resistance_management_policy_decomposition.csv",
        columns=(
            "analysis_layer",
            "policy_read",
            "analysis_window",
            "treatment_component",
            "pep_component",
            "testing_or_uptake_component",
            "median_infant_case_reduction_vs_current",
            "iqr_infant_case_reduction_vs_current",
            "countries_with_positive_reduction",
            "median_infant_cases_per_100k",
            "median_resistance_metric",
            "end_resistant_fraction",
            "interpretation",
        ),
        labels=(
            "Analysis layer",
            "Policy read",
            "Analysis window",
            "Treatment component",
            "PEP component",
            "Testing/uptake component",
            "Median infant-case reduction vs current",
            "Across-profile IQR reduction",
            "Countries with positive reduction",
            "Median infant cases per 100k",
            "Median resistant infections per 100k",
            "End resistant fraction",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S17",
        title="Infant-contact and maternal passive-protection sensitivity diagnostics.",
        source="outputs/tables/infant_contact_sensitivity.csv plus maternal_duration_sensitivity.csv",
        rows=infant_contact_maternal_rows,
        columns=(
            "sensitivity_dimension",
            "strategy",
            "setting",
            "median_infant_cases_per_100k_5y",
            "iqr_infant_cases_per_100k_5y",
            "median_infant_case_reduction_vs_current_5y",
            "iqr_reduction",
            "countries_with_positive_reduction",
            "countries",
            "interpretation",
        ),
        labels=(
            "Sensitivity dimension",
            "Strategy",
            "Setting",
            "Median infant cases per 100k, 5 y",
            "Across-profile IQR infant cases per 100k, 5 y",
            "Median infant-case reduction vs current, 5 y",
            "Across-profile IQR reduction",
            "Countries with positive reduction",
            "Countries",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S18",
        title="Routine nominal coverage-floor and timeliness mechanism diagnostics.",
        source="higher_child_coverage country, age-shift, origin-share, and routine timeliness diagnostic CSVs",
        rows=higher_child_coverage_diagnostic_rows,
        columns=(
            "diagnostic",
            "stratum",
            "current_infant_cases_per_100k",
            "higher_child_coverage_infant_cases_per_100k",
            "relative_change_or_share",
            "largest_increase_age_group",
            "age_shift_iqr",
            "countries_with_increase",
            "interpretation",
        ),
        labels=(
            "Diagnostic",
            "Country, age group, or scenario",
            "Reference infant cases per 100k",
            "Scenario infant cases per 100k",
            "Relative change or share",
            "Largest increase age group",
            "Across-profile IQR",
            "Countries with increase or positive reduction",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S19",
        title="Intervention scenario-ordering sensitivity to analysis-window choice.",
        source="outputs/tables/intervention_horizon_rank_summary.csv",
        columns=(
            "analysis_window",
            "scenario",
            "median_rank",
            "countries_ranked_first",
            "median_relative_reduction_infant_cases",
        ),
        labels=(
            "Analysis window",
            "Scenario",
            "Median order position",
            "Countries ordered first",
            "Median infant-case reduction",
        ),
    ),
    TableSpec(
        number="S20",
        title="Cross-diagnostic intervention scenario-ordering stability across countries, analysis windows, and infant age strata.",
        source="outputs/tables/intervention_rank_stability_diagnostics.csv",
        columns=(
            "scenario",
            "full_horizon_median_rank",
            "full_horizon_countries_ranked_first",
            "full_horizon_countries_ranked_top_two",
            "analysis_window_cells_ranked_first",
            "analysis_window_cells_ranked_top_two",
            "infant_age_window_cells_ranked_first",
            "infant_age_window_cells_ranked_top_two",
            "infant_age_window_cells_positive_reduction",
            "median_infant_age_window_reduction",
            "rank_stability_interpretation",
        ),
        labels=(
            "Scenario",
            "Full-horizon median order position",
            "Countries ordered first",
            "Countries ordered top 2",
            "Window cells ordered first",
            "Window cells ordered top 2",
            "Age-window cells ordered first",
            "Age-window cells ordered top 2",
            "Age-window cells with reduction",
            "Median age-window reduction",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S21",
        title="Near-term temporal assumption sensitivity for burn-in duration and COVID-19 NPI contact-shock assumptions.",
        source="outputs/tables/temporal_assumption_sensitivity.csv",
        columns=(
            "temporal_dimension",
            "scenario",
            "countries",
            "burn_in_years",
            "npi_reduction_scale",
            "median_infant_cases_per_100k_5y",
            "iqr_infant_cases_per_100k_5y",
            "median_all_infections_per_100k_5y",
            "median_end_resistant_fraction_5y",
            "implementation_note",
        ),
        labels=(
            "Temporal dimension",
            "Scenario",
            "Countries",
            "Burn-in years",
            "NPI reduction scale",
            "Median infant cases per 100k, 5 y",
            "Across-profile IQR infant cases per 100k, 5 y",
            "Median all infections per 100k, 5 y",
            "Median end resistant fraction, 5 y",
            "Implementation note",
        ),
    ),
    TableSpec(
        number="S22",
        title="Infant age-stratified intervention outcomes summarized by analysis window.",
        source="outputs/tables/infant_age_split_horizon_sensitivity.csv",
        rows=infant_age_summary_rows,
        columns=(
            "analysis_window",
            "age_group",
            "scenario",
            "median_infant_cases_per_100k",
            "iqr_infant_cases_per_100k",
            "median_relative_reduction",
            "median_rank",
            "countries_with_positive_reduction",
            "countries",
        ),
        labels=(
            "Analysis window",
            "Infant age stratum",
            "Scenario",
            "Median infant cases per 100k/y",
            "Across-profile IQR infant cases per 100k/y",
            "Median infant-case reduction",
            "Median order position",
            "Countries with positive reduction",
            "Countries",
        ),
    ),
    TableSpec(
        number="S23",
        title="Deterministic event-scale diagnostics for stochastic-interpretation sensitivity.",
        source="outputs/tables/deterministic_event_scale_diagnostics.csv",
        rows=event_scale_summary_rows,
        columns=(
            "scenario",
            "countries",
            "median_annual_infant_cases_count",
            "minimum_annual_infant_cases_count",
            "median_infant_cases_per_100k",
            "low_event_countries",
            "interpretation",
        ),
        labels=(
            "Scenario",
            "Countries",
            "Median annual infant cases",
            "Minimum annual infant cases",
            "Median infant cases per 100k/y",
            "Low-event countries",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S24",
        title="Interpretation guardrails for limitations and strategy-domain assumptions.",
        source="outputs/tables/limitation_diagnostic_map.csv plus strategy_domain_interpretation.csv",
        rows=interpretation_guardrail_rows,
        columns=(
            "guardrail_type",
            "domain",
            "evidence_or_diagnostic",
            "location_or_role",
            "interpretation",
        ),
        labels=(
            "Type",
            "Domain",
            "Evidence or diagnostic",
            "Location or decision role",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S25",
        title="Selected-input deterministic sensitivity diagnostics for country-differentiated infant-case strategy ordering.",
        source="outputs/tables/joint_psa_rank_acceptability.csv",
        rows=joint_psa_summary_rows,
        columns=(
            "strategy",
            "probability_rank_1",
            "probability_top_2",
            "probability_top_3",
            "probability_within_10_percent_of_best",
            "mean_rank",
            "median_rank",
            "median_infant_cases_per_100k",
            "q025_infant_cases_per_100k",
            "q975_infant_cases_per_100k",
            "median_relative_reduction_vs_current",
            "relative_reduction_range",
            "n_psa_samples",
        ),
        labels=(
            "Strategy",
            "Design frequency ordered first",
            "Design frequency among top 2",
            "Design frequency among top 3",
            "Design frequency within 10% of best",
            "Mean order position",
            "Median order position",
            "Median infant cases per 100k/y",
            "Q2.5 infant cases per 100k/y",
            "Q97.5 infant cases per 100k/y",
            "Median reduction vs current",
            "Reduction envelope (Q2.5-Q97.5)",
            "Sensitivity samples",
        ),
    ),
    TableSpec(
        number="S26",
        title="Individual stochastic contact-clustering toy model key diagnostics (100 replicates, synthetic population 1,500, target R=1.08; structural sensitivity only).",
        source="outputs/tables/individual_stochastic_toy_summary.csv",
        rows=stochastic_toy_key_rows,
        columns=(
            "country",
            "contact_structure",
            "extinction_probability_3_or_fewer",
            "outbreak_probability_20plus",
            "total_infections_interval",
            "infant_infection_summary",
            "mean_household_clusters_touched",
        ),
        labels=(
            "Country",
            "Contact structure",
            "Pr(extinct <=3)",
            "Pr(outbreak >=20 infections)",
            "Total infections, median (2.5%-97.5% replicate range)",
            "Infant infections (mean; Pr any; Q95)",
            "Mean household clusters",
        ),
        sort_by=("country", "contact_structure"),
    ),
    TableSpec(
        number="S27",
        title="Vaccine-pipeline mechanisms mapped to modeled product-target profiles.",
        source="outputs/tables/vaccine_pipeline_mechanism_mapping.csv",
        rows=vaccine_pipeline_summary_rows,
        columns=(
            "candidate_or_platform",
            "development_status",
            "transmission_signal",
            "model_use",
            "evidence_source",
        ),
        labels=(
            "Candidate/platform",
            "Development status",
            "Transmission-relevant signal",
            "Model use",
            "Evidence source",
        ),
    ),
    TableSpec(
        number="S28",
        title="Macrolide-resistance parameter rationale, expected bias direction, and residual caveats.",
        source="outputs/tables/resistance_parameter_justification.csv",
        columns=(
            "parameter_group",
            "baseline_value",
            "explored_range_or_scenarios",
            "source_or_anchor",
            "rationale",
            "expected_direction_of_bias",
            "residual_caveat",
        ),
        labels=(
            "Parameter group",
            "Baseline value",
            "Explored range or scenarios",
            "Source or anchor",
            "Rationale",
            "Expected direction of bias",
            "Residual caveat",
        ),
    ),
    TableSpec(
        number="S29",
        title="Strategy domains, country-differentiated decision role, and interpretation of model assumptions.",
        source="outputs/tables/strategy_domain_interpretation.csv",
        columns=(
            "strategy_domain_or_assumption",
            "evidence_strength",
            "decision_role",
            "scenario_comparison_interpretation",
        ),
        labels=(
            "Strategy domain or assumption",
            "Evidence strength",
            "Decision role",
            "How this should affect scenario-comparison interpretation",
        ),
    ),
    TableSpec(
        number="S30",
        title="Implementation-intensity scores used in dominance and robustness diagnostics.",
        source="outputs/tables/implementation_intensity_score_definitions.csv",
        columns=(
            "implementation_intensity",
            "category",
            "definition",
            "assigned_strategy_profiles",
            "decision_use",
        ),
        labels=(
            "Score",
            "Category",
            "Definition",
            "Assigned strategy profiles",
            "Decision use",
        ),
    ),
    TableSpec(
        number="S31",
        title="External age-pattern weighted scenario-class ordering sensitivity.",
        source="outputs/tables/age_pattern_scenario_ordering_sensitivity.csv",
        columns=(
            "ordering_basis",
            "scenario_class",
            "scenarios_in_class",
            "country_count",
            "weighted_median_infant_cases_per_100k",
            "weighted_median_relative_reduction_infant_cases",
            "class_rank_basis",
            "class_rank",
            "rank_shift_vs_all_profiles",
            "strict_class_order_changed_vs_all_profiles",
            "qualitative_ordering_changed_vs_all_profiles",
            "interpretation_note",
        ),
        labels=(
            "Ordering basis",
            "Scenario class",
            "Scenario profiles in class",
            "Countries",
            "Weighted median infant cases per 100k/y",
            "Weighted median infant-case reduction",
            "Order basis",
            "Class order position",
            "Order shift vs all profiles",
            "Exact class order changed",
            "Qualitative tier order changed",
            "Interpretation",
        ),
    ),
    TableSpec(
        number="S32",
        title="Formula-to-parameter evidence map for model equations, values, rationale, and sensitivity anchors.",
        source="outputs/tables/formula_parameter_evidence_map.csv",
        columns=(
            "formula_or_model_block",
            "parameter_or_setting",
            "value_used",
            "evidence_or_rationale",
            "sensitivity_or_supplement_link",
        ),
        labels=(
            "Formula or model block",
            "Parameter or setting",
            "Value used",
            "Evidence or rationale",
            "Sensitivity or appendix link",
        ),
    ),
    TableSpec(
        number="S33",
        title="Data sources, preprocessing rules, missing-data handling, and configuration outputs.",
        source="outputs/tables/data_provenance_preprocessing_map.csv",
        columns=(
            "data_domain",
            "original_source",
            "time_range",
            "spatial_scope",
            "age_granularity",
            "analysis_use",
            "preprocessing",
            "missing_data_handling",
            "version_or_access",
            "configuration_output",
        ),
        labels=(
            "Data domain",
            "Original source",
            "Time range",
            "Spatial scope",
            "Age granularity",
            "Analysis use",
            "Preprocessing",
            "Missing-data handling",
            "Version or access",
            "Configuration output",
        ),
    ),
)

FOLDED_OR_FIGURE_CONVERTED_TABLE_TITLES = {
    "Condensed macrolide-resistant fitness and vaccine infectiousness grid definition.",
    "Routine nominal coverage-floor and timeliness mechanism diagnostics.",
    "Intervention scenario-ordering sensitivity to analysis-window choice.",
    "Cross-diagnostic intervention scenario-ordering stability across countries, analysis windows, and infant age strata.",
    "Infant age-stratified intervention outcomes summarized by analysis window.",
    "Selected-input deterministic sensitivity diagnostics for country-differentiated infant-case strategy ordering.",
    "External age-pattern weighted scenario-class ordering sensitivity.",
    "Macrolide-resistance mechanism decomposition across importation, treatment, PEP, and fitness assumptions.",
    "Vaccine infectiousness-effect threshold diagnostics.",
    "Country-differentiated intervention outcome summaries by country and strategy.",
    "Resistance-management mechanism decomposition and near-term implementation sensitivity.",
    "Infant-contact and maternal passive-protection sensitivity diagnostics.",
    "Near-term temporal assumption sensitivity for burn-in duration and COVID-19 NPI contact-shock assumptions.",
    "Deterministic event-scale diagnostics for stochastic-interpretation sensitivity.",
    "Individual stochastic contact-clustering toy model key diagnostics (100 replicates, synthetic population 1,500, target R=1.08; structural sensitivity only).",
    "Strategy domains, country-differentiated decision role, and interpretation of model assumptions.",
}

APPENDIX_TABLE_TITLE_ORDER = (
    "Country-profile selection rationale, program features, and resistance anchors.",
    "Data sources, preprocessing rules, missing-data handling, and configuration outputs.",
    "Study design matrix for scenario, sensitivity, and predictive-validation analyses.",
    "Model-derived outcomes, denominators, and comparative summary definitions.",
    "Core model structure, observation model, and implementation settings.",
    "Baseline model parameter values, admissible ranges, and evidence provenance.",
    "Semi-mechanistic POMP predictive-validation specification and claim controls.",
    "Formula-to-parameter evidence map for model equations, values, rationale, and sensitivity anchors.",
    "Intervention strategy definitions, decision domains, and modified control levers.",
    "Macrolide-resistance scenario assumptions for starting fraction, importation, and strain fitness.",
    "Country-specific macrolide-resistance evidence used to assign starting anchors.",
    "Calibration acceptance, fitted transmission parameters, and reported-case fit diagnostics.",
    "Fitted age-specific reporting probabilities and country-specific prior bounds.",
    "Interpretation guardrails for limitations and strategy-domain assumptions.",
    "Implementation-intensity scores used in dominance and robustness diagnostics.",
    "Macrolide-resistance parameter rationale, expected bias direction, and residual caveats.",
    "Vaccine-pipeline mechanisms mapped to modeled product-target profiles.",
)


def order_appendix_tables(tables: tuple[TableSpec, ...]) -> tuple[TableSpec, ...]:
    by_title = {spec.title: spec for spec in tables}
    expected = set(APPENDIX_TABLE_TITLE_ORDER)
    actual = set(by_title)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing submitted table titles: {sorted(missing)}")
        if extra:
            details.append(f"unexpected submitted table titles: {sorted(extra)}")
        raise ValueError("; ".join(details))
    return tuple(by_title[title] for title in APPENDIX_TABLE_TITLE_ORDER)


TABLES = tuple(spec for spec in TABLES if spec.title not in FOLDED_OR_FIGURE_CONVERTED_TABLE_TITLES)
TABLES = order_appendix_tables(TABLES)
TABLES = tuple(replace(spec, number=f"S{index}") for index, spec in enumerate(TABLES, start=1))


def sort_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, str]]:
    if not keys:
        return rows
    return sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in keys))


def format_number(value: float, *, integer_like: bool = False, year_like: bool = False) -> str:
    if year_like:
        return str(int(round(value)))
    if integer_like and float(value).is_integer():
        return f"{value:,.0f}"
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:,.2f}"


def format_small_parameter_value(value: float) -> str:
    if value == 0:
        return "0.00"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    if abs(value) < 0.01:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if abs(value) < 0.1:
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return format_number(value)


def format_value(value: object, column: str = "") -> str:
    text = "" if value is None else str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    if column == "country":
        text = text.replace("_", " ")
    column_lower = column.lower()
    if column_lower in DISPLAY_LABEL_COLUMNS or column_lower.endswith("_scenario") or column_lower.endswith("_strategy"):
        text = display_label(text)
    lowered = text.lower()
    if lowered in {"true", "false", "yes", "no"}:
        return "Yes" if lowered in {"true", "yes"} else "No"
    if column_lower in TEXT_FRAGMENT_COLUMNS:
        text = display_text_fragments(text)
        text = _format_decimal_tokens(text)
    if (
        column_lower != "prior_bounds"
        and column_lower not in RANGE_FORMAT_EXEMPT_COLUMNS
        and any(token in column_lower for token in ("iqr", "interval", "range", "bounds"))
    ):
        text = _format_range_tokens(text)
    text = separate_bare_url_list(text)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return text
    try:
        year_like = (
            (column_lower == "year" or column_lower.endswith("_year"))
            and "per_" not in column_lower
            and "life_year" not in column_lower
        )
        integer_like = year_like or column.lower() in {
            "sample_size",
            "total_infections",
            "reported_cases",
            "total_infant_cases",
            "resistant_infections",
            "timeseries_rows",
            "summary_rows",
            "countries",
            "countries_with_increase",
            "countries_ranked_first",
            "countries_within_10_percent_of_best",
            "countries_with_positive_reduction",
            "countries_with_infection_increase",
            "countries_reaching_comparator",
            "countries_evaluated",
            "country_count",
            "class_rank",
            "reference_class_rank_all_profiles",
            "rank_shift_vs_all_profiles",
            "rank",
            "implementation_intensity",
            "full_horizon_countries_ranked_first",
            "full_horizon_countries_ranked_top_two",
            "analysis_window_cells",
            "analysis_window_cells_ranked_first",
            "analysis_window_cells_ranked_top_two",
            "analysis_window_cells_positive_reduction",
            "infant_age_window_cells",
            "infant_age_window_cells_ranked_first",
            "infant_age_window_cells_ranked_top_two",
            "infant_age_window_cells_positive_reduction",
            "n_epidemic_peaks",
            "n_intervals",
            "psa_sample_id",
            "n_psa_samples",
            "n_replicates",
            "population_size",
            "sample_size_display",
            "raw_rows_for_country",
            "processed_rows_for_country",
            "country_profile_contact_groups",
        }
        number = float(text)
        if column_lower == "baseline_value" and abs(number) < 0.1:
            return format_small_parameter_value(number)
        return format_number(number, integer_like=integer_like, year_like=year_like)
    except ValueError:
        return text


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def styled_table_cell(text: str, style: str) -> str:
    if not style:
        return text
    return f'<span style="{style}">{text}</span>'


def markdown_table(
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    column_styles: tuple[str, ...] = (),
) -> str:
    if not rows:
        return "_No source rows were found when the table was generated._"
    if column_styles and len(column_styles) != len(columns):
        raise ValueError("column_styles must match the number of rendered columns")
    styles = column_styles or tuple("" for _ in columns)
    header_values = [
        styled_table_cell(markdown_escape(label), style)
        for label, style in zip(labels, styles, strict=True)
    ]
    header = "| " + " | ".join(header_values) + " |"
    divider = "| " + " | ".join("---" for _ in labels) + " |"
    body = []
    for row in rows:
        values = [
            styled_table_cell(markdown_escape(format_value(row.get(column, ""), column)), style)
            for column, style in zip(columns, styles, strict=True)
        ]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def render_table(spec: TableSpec) -> str:
    rows = spec.rows() if spec.rows else read_csv_rows(spec.source)
    rows = sort_rows(rows, spec.sort_by)
    table = markdown_table(rows, spec.columns, spec.labels, spec.column_styles)
    display_number = spec.number[1:] if spec.number.startswith("S") else spec.number
    return f"### Table S{display_number}. {spec.title}\n\n{table}"


def render_all_tables() -> str:
    separator = f"\n\n{PAGE_BREAK}\n\n"
    return separator.join(render_table(spec) for spec in TABLES) + f"\n\n{PAGE_BREAK}"


def replace_table_section(document: str) -> str:
    headings = (TABLES_HEADING,)
    heading = next((candidate for candidate in headings if candidate in document), "")
    if not heading:
        raise ValueError(f"Missing table section for generated tables in {TARGET}")
    start = document.index(heading)
    prefix = document[:start]
    body_start = start + len(heading)
    next_section = SECTION_HEADING_RE.search(document, body_start)
    suffix = document[next_section.start() :] if next_section else ""
    table_section = TABLES_HEADING + "\n\n" + render_all_tables() + "\n"
    parts = []
    if prefix.strip():
        parts.append(prefix.rstrip())
    parts.append(table_section.rstrip())
    if suffix.strip():
        parts.append(suffix.strip())
    return "\n\n".join(parts) + "\n"


def replace_block(document: str, spec: TableSpec) -> str:
    pattern = re.compile(
        rf"<!-- BEGIN TABLE {re.escape(spec.number)} -->.*?"
        rf"<!-- END TABLE {re.escape(spec.number)} -->",
        flags=re.DOTALL,
    )
    rendered = render_table(spec)
    if not pattern.search(document):
        legacy_pattern = re.compile(
            rf"(?ms)^(?:###\s+Table {re.escape(spec.number)}\.|\*\*Table {re.escape(spec.number)}\.).*?"
            rf"(?=^(?:###\s+Table S\d+\.|\*\*Table S\d+\.)|\Z)"
        )
        if legacy_pattern.search(document):
            return legacy_pattern.sub(rendered, document, count=1)
        if TABLES_HEADING in document:
            return document.rstrip() + "\n\n" + rendered + "\n"
        raise ValueError(f"Missing table section for {spec.number} in {TARGET}")
    return pattern.sub(rendered, document, count=1)


def main() -> None:
    document = TARGET.read_text(encoding="utf-8")
    document = replace_table_section(document)
    TARGET.write_text(document, encoding="utf-8")
    print(f"Updated {TARGET.relative_to(ROOT)} with {len(TABLES)} generated tables.")


if __name__ == "__main__":
    main()
