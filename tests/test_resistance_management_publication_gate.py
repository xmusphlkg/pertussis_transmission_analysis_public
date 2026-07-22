from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manuscript_notes import validate_publication_outputs as validator


def _complete_management_fixture() -> tuple[
    list[str], dict, dict[str, pd.DataFrame]
]:
    countries = [f"Country_{index}" for index in range(9)]
    sample_ids = np.arange(1, 129, dtype=int)
    uptake = np.linspace(0.40, 1.00, 128)
    reach = np.linspace(0.50, 1.00, 128)
    samples = pd.DataFrame(
        {
            "psa_sample_id": sample_ids,
            "sample_design": validator.RESISTANCE_MANAGEMENT_SAMPLE_DESIGN,
            "uncertainty_schema_version": (
                validator.RESISTANCE_MANAGEMENT_SCHEMA_VERSION
            ),
            "resistance_management_uptake": uptake,
            "resistance_management_pep_reach_multiplier": reach,
        }
    )

    draw_rows: list[dict] = []
    for country_index, country in enumerate(countries):
        for pep_restoration, pep_restored in validator.PEP_RESTORATION_STRATA:
            for sample_id, sampled_uptake, sampled_reach in zip(
                sample_ids, uptake, reach, strict=True
            ):
                baseline_treatment = 0.02
                full_treatment = 0.06
                baseline_coverage = 0.30
                baseline_effectiveness = 0.10
                full_effectiveness = 0.45
                sample_fraction = (float(sample_id) - 1.0) / 127.0
                restoration_shift = 0.03 if pep_restored else -0.01
                under18_reduction = (
                    -0.08
                    + 0.16 * sample_fraction
                    + 0.005 * country_index
                    + restoration_shift
                )
                infant_reduction = (
                    -0.10
                    + 0.14 * sample_fraction
                    + 0.003 * country_index
                    + restoration_shift
                )
                resistant_reduction = (
                    0.05
                    + 0.50 * sample_fraction
                    + 0.01 * country_index
                    + (0.10 if pep_restored else 0.0)
                )
                timeliness_under18_total = 1000.0 + country_index
                timeliness_under18_rate = 100.0 + country_index
                timeliness_infant_total = 100.0 + country_index
                timeliness_infant_rate = 10.0 + country_index
                timeliness_resistant = 50.0 + country_index
                guided_under18_total = timeliness_under18_total * (
                    1.0 - under18_reduction
                )
                guided_under18_rate = timeliness_under18_rate * (
                    1.0 - under18_reduction
                )
                guided_infant_total = timeliness_infant_total * (
                    1.0 - infant_reduction
                )
                guided_infant_rate = timeliness_infant_rate * (
                    1.0 - infant_reduction
                )
                guided_resistant = timeliness_resistant * (
                    1.0 - resistant_reduction
                )
                observed_under18_reduction = (
                    1.0 - guided_under18_total / timeliness_under18_total
                )
                observed_infant_reduction = (
                    1.0 - guided_infant_total / timeliness_infant_total
                )
                observed_resistant_reduction = (
                    1.0 - guided_resistant / timeliness_resistant
                )
                draw_rows.append(
                    {
                        "country": country,
                        "psa_sample_id": sample_id,
                        "pep_restoration": pep_restoration,
                        "pep_restored": pep_restored,
                        "sample_design": (
                            validator.RESISTANCE_MANAGEMENT_SAMPLE_DESIGN
                        ),
                        "uncertainty_schema_version": (
                            validator.RESISTANCE_MANAGEMENT_SCHEMA_VERSION
                        ),
                        "resistance_management_uptake": sampled_uptake,
                        "resistance_management_pep_reach_multiplier": sampled_reach,
                        "baseline_treatment_rate_symptomatic": baseline_treatment,
                        "full_guided_treatment_rate_symptomatic": full_treatment,
                        "applied_treatment_rate_symptomatic": (
                            baseline_treatment
                            + sampled_uptake
                            * (full_treatment - baseline_treatment)
                        ),
                        "baseline_pep_coverage": baseline_coverage,
                        "applied_pep_coverage": baseline_coverage * sampled_reach,
                        "baseline_pep_effectiveness_resistant": baseline_effectiveness,
                        "full_guided_pep_effectiveness_resistant": full_effectiveness,
                        "applied_pep_effectiveness_resistant": (
                            baseline_effectiveness
                            + sampled_uptake
                            * (full_effectiveness - baseline_effectiveness)
                            if pep_restored
                            else baseline_effectiveness
                        ),
                        "guided_total_child_adolescent_cases": guided_under18_total,
                        "guided_annualized_child_adolescent_cases_per_100k": (
                            guided_under18_rate
                        ),
                        "guided_total_infant_cases": guided_infant_total,
                        "guided_annualized_infant_cases_per_100k": (
                            guided_infant_rate
                        ),
                        "guided_resistant_infections": guided_resistant,
                        "timeliness_total_child_adolescent_cases": (
                            timeliness_under18_total
                        ),
                        "timeliness_annualized_child_adolescent_cases_per_100k": (
                            timeliness_under18_rate
                        ),
                        "timeliness_total_infant_cases": timeliness_infant_total,
                        "timeliness_annualized_infant_cases_per_100k": (
                            timeliness_infant_rate
                        ),
                        "timeliness_resistant_infections": timeliness_resistant,
                        "relative_reduction_under18_cases_vs_timeliness": (
                            observed_under18_reduction
                        ),
                        "guided_lower_under18_cases": (
                            guided_under18_total < timeliness_under18_total
                        ),
                        "relative_reduction_infant_cases_vs_timeliness": (
                            observed_infant_reduction
                        ),
                        "guided_lower_infant_cases": (
                            guided_infant_total < timeliness_infant_total
                        ),
                        "relative_reduction_resistant_infections_vs_timeliness": (
                            observed_resistant_reduction
                        ),
                        "guided_lower_resistant_infections": (
                            guided_resistant < timeliness_resistant
                        ),
                    }
                )
    draws = pd.DataFrame(draw_rows)

    summary_rows: list[dict] = []

    def append_summary(country: str, stratum: str, group: pd.DataFrame) -> None:
        row = {
            "country": country,
            "pep_restoration": stratum,
            "psa_samples": int(group["psa_sample_id"].nunique()),
            "interpretation": (
                "Elicited-input design summary; not a posterior probability "
                "or Figure 2b input."
            ),
        }
        for endpoint in ("under18_cases", "infant_cases", "resistant_infections"):
            values = group[f"relative_reduction_{endpoint}_vs_timeliness"]
            row[f"median_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.median()
            )
            row[f"q025_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.quantile(0.025)
            )
            row[f"q975_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.quantile(0.975)
            )
            row[f"fraction_design_draws_guided_lower_{endpoint}"] = float(
                group[f"guided_lower_{endpoint}"].mean()
            )
        summary_rows.append(row)

    for stratum, _ in validator.PEP_RESTORATION_STRATA:
        stratum_draws = draws.loc[draws["pep_restoration"].eq(stratum)]
        for country, group in stratum_draws.groupby("country", sort=True):
            append_summary(str(country), stratum, group)
        append_summary("All_countries_pooled", stratum, stratum_draws)
    summary = pd.DataFrame(summary_rows)
    activity = [
        {
            "country": country,
            "pep_restoration": stratum,
            "parameter": parameter,
            "unique_values": 128,
            "sampled_range": 0.5,
            "configured_target_range": 0.2,
            "application_verified": True,
            "parameter_active": True,
        }
        for country in countries
        for stratum, _ in validator.PEP_RESTORATION_STRATA
        for parameter in validator.RESISTANCE_MANAGEMENT_PARAMETER_NAMES
    ]
    metadata = {
        "sample_size_requested": 128,
        "sample_seed": 20260522,
        "sample_design": validator.RESISTANCE_MANAGEMENT_SAMPLE_DESIGN,
        "uncertainty_schema_version": validator.RESISTANCE_MANAGEMENT_SCHEMA_VERSION,
        "countries": countries,
        "analysis_horizon_years": 5,
        "comparator": "routine_timeliness",
        "programme_base_in_both_arms": "routine_timeliness",
        "contrast": (
            "routine_timeliness_plus_guided_management_vs_"
            "routine_timeliness_alone"
        ),
        "sampled_parameters": list(
            validator.RESISTANCE_MANAGEMENT_PARAMETER_NAMES
        ),
        "parameter_time_scopes": {
            parameter: "prospective_implementation"
            for parameter in validator.RESISTANCE_MANAGEMENT_PARAMETER_NAMES
        },
        "parameter_distributions": {
            parameter: {}
            for parameter in validator.RESISTANCE_MANAGEMENT_PARAMETER_NAMES
        },
        "deterministic_structural_strata": ["restored", "not_restored"],
        "paired_across_pep_restoration_strata": True,
        "pep_restoration_probability_assigned": False,
        "figure2b_excluded": True,
        "row_counts": {
            "parameter_samples": 128,
            "comparator_rows": 9,
            "guided_rows": 2304,
            "effect_samples": 2304,
            "summary_rows": 20,
            "activity_audit_rows": 36,
        },
        "activity_audit": activity,
    }
    return countries, metadata, {
        "resistance_management_psa_parameter_samples.csv": samples,
        "resistance_management_psa_effect_samples.csv": draws,
        "resistance_management_psa_summary.csv": summary,
    }


def _install_fixture(monkeypatch, countries, tables) -> None:
    monkeypatch.setattr(
        validator,
        "load_configs",
        lambda: {
            "parameter_distributions": {
                "resistance_management_psa": {
                    "parameters": {
                        "resistance_management_uptake": {
                            "low": 0.40,
                            "high": 1.00,
                        },
                        "resistance_management_pep_reach_multiplier": {
                            "low": 0.50,
                            "high": 1.00,
                        },
                    }
                }
            }
        },
    )
    monkeypatch.setattr(
        validator, "publication_country_names", lambda _configs: list(countries)
    )
    monkeypatch.setattr(
        validator,
        "read_table",
        lambda path: tables[Path(path).name].copy(deep=True),
    )


def test_resistance_management_publication_gate_accepts_complete_2d_design(
    monkeypatch,
) -> None:
    countries, metadata, tables = _complete_management_fixture()
    _install_fixture(monkeypatch, countries, tables)

    validator.validate_resistance_management_psa(metadata)


def test_resistance_management_publication_gate_accepts_ode_scale_roundoff(
    monkeypatch,
) -> None:
    countries, metadata, tables = _complete_management_fixture()
    rounded = {name: frame.copy(deep=True) for name, frame in tables.items()}
    effects = rounded["resistance_management_psa_effect_samples.csv"]
    comparator_rate = effects.loc[
        0, "timeliness_annualized_infant_cases_per_100k"
    ]
    effects.loc[0, "guided_annualized_infant_cases_per_100k"] += (
        3e-9 * comparator_rate
    )
    _install_fixture(monkeypatch, countries, rounded)

    validator.validate_resistance_management_psa(deepcopy(metadata))


def test_resistance_management_publication_gate_rejects_unconsumed_reach(
    monkeypatch,
) -> None:
    countries, metadata, tables = _complete_management_fixture()
    broken = {name: frame.copy(deep=True) for name, frame in tables.items()}
    broken["resistance_management_psa_effect_samples.csv"].loc[
        0, "applied_pep_coverage"
    ] += 0.05
    _install_fixture(monkeypatch, countries, broken)

    with pytest.raises(AssertionError, match="PEP reach application"):
        validator.validate_resistance_management_psa(deepcopy(metadata))


def test_resistance_management_publication_gate_rejects_corrupt_raw_effect(
    monkeypatch,
) -> None:
    countries, metadata, tables = _complete_management_fixture()
    broken = {name: frame.copy(deep=True) for name, frame in tables.items()}
    broken["resistance_management_psa_effect_samples.csv"].loc[
        0, "relative_reduction_under18_cases_vs_timeliness"
    ] += 0.01
    _install_fixture(monkeypatch, countries, broken)

    with pytest.raises(AssertionError, match="endpoint arithmetic.*under18_cases"):
        validator.validate_resistance_management_psa(deepcopy(metadata))


def test_resistance_management_publication_gate_rejects_corrupt_annualized_effect(
    monkeypatch,
) -> None:
    countries, metadata, tables = _complete_management_fixture()
    broken = {name: frame.copy(deep=True) for name, frame in tables.items()}
    broken["resistance_management_psa_effect_samples.csv"].loc[
        0, "guided_annualized_infant_cases_per_100k"
    ] += 0.01
    _install_fixture(monkeypatch, countries, broken)

    with pytest.raises(
        AssertionError,
        match="annualized and total burden contrasts disagree.*infant_cases",
    ):
        validator.validate_resistance_management_psa(deepcopy(metadata))


def test_resistance_management_publication_gate_rejects_corrupt_summary(
    monkeypatch,
) -> None:
    countries, metadata, tables = _complete_management_fixture()
    broken = {name: frame.copy(deep=True) for name, frame in tables.items()}
    summary = broken["resistance_management_psa_summary.csv"]
    pooled_index = summary.index[
        summary["country"].eq("All_countries_pooled")
    ][0]
    summary.loc[
        pooled_index,
        "median_relative_reduction_resistant_infections_vs_timeliness",
    ] += 0.01
    _install_fixture(monkeypatch, countries, broken)

    with pytest.raises(
        AssertionError,
        match=(
            "summary disagrees.*"
            "median_relative_reduction_resistant_infections_vs_timeliness"
        ),
    ):
        validator.validate_resistance_management_psa(deepcopy(metadata))


def test_resistance_management_publication_gate_rejects_figure2b_reentry(
    monkeypatch,
) -> None:
    countries, metadata, tables = _complete_management_fixture()
    metadata["figure2b_excluded"] = False
    _install_fixture(monkeypatch, countries, tables)

    with pytest.raises(AssertionError, match="figure2b_excluded"):
        validator.validate_resistance_management_psa(metadata)
