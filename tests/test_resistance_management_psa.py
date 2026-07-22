from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src_python.simulation import run_joint_psa_rank_acceptability as figure2b
from src_python.simulation import run_resistance_management_psa as resistance_psa
from src_python.simulation.common import PROSPECTIVE_POLICY_KEY, load_configs


def _registry_contract() -> tuple[dict, dict, int]:
    configs = load_configs()
    settings, schema_version = resistance_psa._settings(configs)
    specs = resistance_psa._parameter_specs(settings)
    return settings, specs, schema_version


def _sample_frame(
    uptakes: tuple[float, ...] = (0.4, 0.8),
    pep_reach_multipliers: tuple[float, ...] = (0.5, 1.0),
) -> pd.DataFrame:
    assert len(uptakes) == len(pep_reach_multipliers)
    return pd.DataFrame(
        {
            "psa_sample_id": np.arange(1, len(uptakes) + 1, dtype=int),
            "sample_design": resistance_psa.SAMPLE_DESIGN,
            "uncertainty_schema_version": resistance_psa.UNCERTAINTY_SCHEMA_VERSION,
            "resistance_management_uptake": uptakes,
            "resistance_management_pep_reach_multiplier": pep_reach_multipliers,
        }
    )


def _endpoint_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = _sample_frame()
    comparator = pd.DataFrame(
        {
            "country": ["A"],
            "total_child_adolescent_cases": [100.0],
            "annualized_child_adolescent_cases_per_100k": [10.0],
            "total_infant_cases": [20.0],
            "annualized_infant_cases_per_100k": [2.0],
            "resistant_infections": [10.0],
        }
    )
    rows = []
    burdens = {
        ("restored", 1): (110.0, 11.0, 22.0, 2.2, 9.0),
        ("restored", 2): (70.0, 7.0, 14.0, 1.4, 5.0),
        ("not_restored", 1): (115.0, 11.5, 23.0, 2.3, 9.5),
        ("not_restored", 2): (80.0, 8.0, 16.0, 1.6, 6.0),
    }
    for pep_restoration, pep_restored in resistance_psa.PEP_RESTORATION_STRATA:
        for sample_id, uptake, reach in ((1, 0.4, 0.5), (2, 0.8, 1.0)):
            child, child_rate, infant, infant_rate, resistant = burdens[
                (pep_restoration, sample_id)
            ]
            rows.append(
                {
                    "country": "A",
                    "psa_sample_id": sample_id,
                    "pep_restoration": pep_restoration,
                    "pep_restored": pep_restored,
                    "resistance_management_uptake": uptake,
                    "resistance_management_pep_reach_multiplier": reach,
                    "baseline_treatment_rate_symptomatic": 0.02,
                    "full_guided_treatment_rate_symptomatic": 0.06,
                    "applied_treatment_rate_symptomatic": 0.02
                    + uptake * (0.06 - 0.02),
                    "baseline_pep_coverage": 0.30,
                    "applied_pep_coverage": 0.30 * reach,
                    "baseline_pep_effectiveness_resistant": 0.10,
                    "full_guided_pep_effectiveness_resistant": 0.45,
                    "applied_pep_effectiveness_resistant": (
                        0.10 + uptake * (0.45 - 0.10)
                        if pep_restored
                        else 0.10
                    ),
                    "total_child_adolescent_cases": child,
                    "annualized_child_adolescent_cases_per_100k": child_rate,
                    "total_infant_cases": infant,
                    "annualized_infant_cases_per_100k": infant_rate,
                    "resistant_infections": resistant,
                }
            )
    guided = pd.DataFrame(rows)
    return comparator, guided, samples


def test_registry_is_exactly_two_dimensional_and_fail_closed() -> None:
    settings, specs, schema_version = _registry_contract()

    assert schema_version == resistance_psa.UNCERTAINTY_SCHEMA_VERSION == 1
    assert tuple(specs) == resistance_psa.EXPECTED_PARAMETER_NAMES == (
        "resistance_management_uptake",
        "resistance_management_pep_reach_multiplier",
    )
    for spec in specs.values():
        assert spec["time_scope"] == "prospective_implementation"
        assert spec["consumers"] == [
            "src_python.simulation.run_resistance_management_psa"
        ]

    extra_dimension = deepcopy(settings)
    extra_dimension["parameters"]["inactive_parameter"] = deepcopy(
        specs["resistance_management_uptake"]
    )
    with pytest.raises(ValueError, match="exactly match"):
        resistance_psa._parameter_specs(extra_dimension)

    missing_dimension = deepcopy(settings)
    del missing_dimension["parameters"]["resistance_management_pep_reach_multiplier"]
    with pytest.raises(ValueError, match="exactly match"):
        resistance_psa._parameter_specs(missing_dimension)


@pytest.mark.parametrize("bad_version", [None, True, "1", 1.0, 0, 2])
def test_resistance_management_registry_schema_fails_closed(bad_version) -> None:
    configs = load_configs()
    registry = configs["parameter_distributions"]
    if bad_version is None:
        registry.pop("schema_version")
    else:
        registry["schema_version"] = bad_version

    with pytest.raises(ValueError, match="schema_version"):
        resistance_psa._settings(configs)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("time_scope", "structural_all_time", "prospective_implementation"),
        ("consumers", [], "consumers list"),
        (
            "consumers",
            ["untrusted.run_resistance_management_psa"],
            "consumers list",
        ),
    ],
)
def test_registry_time_scope_and_consumer_are_fail_closed(
    field: str,
    bad_value: object,
    message: str,
) -> None:
    settings, _, _ = _registry_contract()
    broken = deepcopy(settings)
    broken["parameters"]["resistance_management_pep_reach_multiplier"][field] = bad_value

    with pytest.raises(ValueError, match=message):
        resistance_psa._parameter_specs(broken)


def test_lhs_is_reproducible_bounded_and_nonconstant() -> None:
    _, specs, schema_version = _registry_contract()
    first = resistance_psa._sample_table(
        64,
        2026,
        specs,
        schema_version=schema_version,
    )
    repeated = resistance_psa._sample_table(
        64,
        2026,
        specs,
        schema_version=schema_version,
    )
    other_seed = resistance_psa._sample_table(
        64,
        2027,
        specs,
        schema_version=schema_version,
    )

    pd.testing.assert_frame_equal(first, repeated)
    assert not first.equals(other_seed)
    assert first.columns.tolist() == [
        "psa_sample_id",
        "sample_design",
        "uncertainty_schema_version",
        "resistance_management_uptake",
        "resistance_management_pep_reach_multiplier",
    ]
    assert first["psa_sample_id"].tolist() == list(range(1, 65))
    assert first["sample_design"].eq(resistance_psa.SAMPLE_DESIGN).all()
    assert first["uncertainty_schema_version"].eq(schema_version).all()
    uptake = first["resistance_management_uptake"]
    uptake_spec = specs["resistance_management_uptake"]
    assert uptake.between(float(uptake_spec["low"]), float(uptake_spec["high"])).all()
    assert uptake.nunique() == len(first)
    assert float(uptake.max() - uptake.min()) > 0.25
    reach = first["resistance_management_pep_reach_multiplier"]
    reach_spec = specs["resistance_management_pep_reach_multiplier"]
    assert reach.between(float(reach_spec["low"]), float(reach_spec["high"])).all()
    assert reach.nunique() == len(first)
    assert float(reach.max() - reach.min()) > 0.40

    with pytest.raises(ValueError, match="at least two samples"):
        resistance_psa._sample_table(1, 2026, specs)


def test_guided_uptake_changes_only_prospective_policy_with_common_history() -> None:
    samples = _sample_frame((0.4, 1.0), (0.5, 1.0))
    comparator = resistance_psa._comparator_scenarios(
        ["Australia"],
        resistance_name="country_timeline",
        horizon_years=5,
    )[0]
    guided = resistance_psa._guided_scenarios(
        samples,
        ["Australia"],
        resistance_name="country_timeline",
        horizon_years=5,
    )

    assert len(guided) == 4
    comparator_history = comparator["config"][PROSPECTIVE_POLICY_KEY]["history_config"]
    histories = [
        scenario["config"][PROSPECTIVE_POLICY_KEY]["history_config"]
        for scenario in guided
    ]
    assert all(history == comparator_history for history in histories)
    assert all(
        scenario["config"]["routine_vaccination"]
        == comparator["config"]["routine_vaccination"]
        for scenario in guided
    )
    assert comparator["metadata"]["programme_base"] == "routine_timeliness"
    assert all(
        scenario["metadata"]["programme_base"] == "routine_timeliness"
        for scenario in guided
    )

    restored = [
        scenario for scenario in guided
        if scenario["metadata"]["pep_restoration"] == "restored"
    ]
    not_restored = [
        scenario for scenario in guided
        if scenario["metadata"]["pep_restoration"] == "not_restored"
    ]
    low = restored[0]["config"]
    high = restored[1]["config"]
    prospective_paths = (
        ("treatment", "treatment_rate_symptomatic"),
        ("treatment", "resistant", "infectious_duration_reduction"),
        ("treatment", "resistant", "infectiousness_reduction"),
        ("PEP", "effectiveness_resistant"),
    )

    def nested(config: dict, path: tuple[str, ...]) -> float:
        value = config
        for key in path:
            value = value[key]
        return float(value)

    assert all(nested(low, path) != nested(high, path) for path in prospective_paths)
    assert all(
        nested(histories[0], path) == nested(histories[1], path)
        for path in prospective_paths
    )
    assert low["PEP"]["coverage_household_contacts"] == pytest.approx(0.15)
    assert high["PEP"]["coverage_household_contacts"] == pytest.approx(0.30)
    assert all(
        scenario["config"]["PEP"]["effectiveness_resistant"]
        == comparator["config"]["PEP"]["effectiveness_resistant"]
        for scenario in not_restored
    )
    assert low["metadata"]["resistance_management_psa_sample_id"] == 1
    assert high["metadata"]["resistance_management_psa_sample_id"] == 2


def test_effect_draws_and_design_summary_use_paired_timeliness_comparator() -> None:
    comparator, guided, samples = _endpoint_frames()

    draws = resistance_psa._effect_draws(comparator, guided, samples)

    assert len(draws) == 4
    restored = draws.loc[draws["pep_restoration"].eq("restored")].sort_values(
        "psa_sample_id"
    )
    assert restored["relative_reduction_under18_cases_vs_timeliness"].tolist() == pytest.approx(
        [-0.1, 0.3]
    )
    assert restored["relative_reduction_infant_cases_vs_timeliness"].tolist() == pytest.approx(
        [-0.1, 0.3]
    )
    assert restored[
        "relative_reduction_resistant_infections_vs_timeliness"
    ].tolist() == pytest.approx([0.1, 0.5])
    assert not restored["guided_lower_under18_cases"].iloc[0]
    assert restored["guided_lower_under18_cases"].iloc[1]

    summary = resistance_psa._summarise(draws)
    assert summary["country"].tolist() == [
        "A",
        "All_countries_pooled",
        "A",
        "All_countries_pooled",
    ]
    assert summary["pep_restoration"].tolist() == [
        "restored",
        "restored",
        "not_restored",
        "not_restored",
    ]
    assert summary["psa_samples"].eq(2).all()
    assert summary.loc[
        summary["pep_restoration"].eq("restored"),
        "median_relative_reduction_under18_cases_vs_timeliness",
    ].tolist() == pytest.approx([0.1, 0.1])
    assert summary.loc[
        summary["pep_restoration"].eq("restored"),
        "fraction_design_draws_guided_lower_under18_cases",
    ].eq(0.5).all()
    assert summary["interpretation"].str.contains("not a posterior").all()
    assert summary["interpretation"].str.contains(
        "not a posterior probability or Figure 2b input"
    ).all()


def test_effect_draws_reject_duplicate_or_mismatched_uptake_metadata() -> None:
    comparator, guided, samples = _endpoint_frames()
    duplicate = pd.concat([guided, guided.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate country-sample-stratum"):
        resistance_psa._effect_draws(comparator, duplicate, samples)

    mismatched = guided.copy()
    mismatched.loc[
        mismatched["psa_sample_id"].eq(2),
        "resistance_management_pep_reach_multiplier",
    ] = 0.7
    with pytest.raises(ValueError, match="do not match"):
        resistance_psa._effect_draws(comparator, mismatched, samples)


def test_activity_audit_verifies_both_parameters_in_both_structural_strata() -> None:
    comparator, guided, samples = _endpoint_frames()
    draws = resistance_psa._effect_draws(comparator, guided, samples)

    audit = resistance_psa._activity_audit(draws)

    assert len(audit) == 4
    assert set(audit["pep_restoration"]) == {"restored", "not_restored"}
    assert set(audit["parameter"]) == set(resistance_psa.EXPECTED_PARAMETER_NAMES)
    assert audit["parameter_active"].all()
    assert audit["application_verified"].all()
    assert audit["unique_values"].eq(2).all()

    inactive = draws.copy()
    inactive["applied_pep_coverage"] = 0.30
    with pytest.raises(RuntimeError, match="inactive or misapplied"):
        resistance_psa._activity_audit(inactive)


def test_figure2b_registry_and_consumer_exclude_management_uptake() -> None:
    configs = load_configs()
    figure2b_specs = configs["parameter_distributions"]["joint_rank_psa"][
        "parameters"
    ]
    management_spec = configs["parameter_distributions"]["resistance_management_psa"][
        "parameters"
    ]["resistance_management_uptake"]

    assert "resistance_management_uptake" not in figure2b.FIGURE2B_PARAMETER_NAMES
    assert "resistance_management_uptake" not in figure2b.EXPECTED_PARAMETER_NAMES
    assert "resistance_management_uptake" not in figure2b_specs
    with pytest.raises(ValueError, match="semantic contract"):
        figure2b._sample_table(
            4,
            11,
            {**figure2b_specs, "resistance_management_uptake": management_spec},
        )

    base, _ = figure2b._make_strategy_config(
        "timeliness_only",
        country="Australia",
    )
    current, _ = figure2b._make_strategy_config("current", country="Australia")
    figure2b_sample = figure2b._sample_table(2, 12, figure2b_specs).iloc[0].to_dict()
    figure2b_sample["resistance_management_uptake"] = 0.75
    with pytest.raises(ValueError, match="inactive for all Figure 2b"):
        figure2b._apply_psa_sample(
            base,
            current,
            strategy="timeliness_only",
            sample=figure2b_sample,
            parameter_specs=figure2b_specs,
        )
