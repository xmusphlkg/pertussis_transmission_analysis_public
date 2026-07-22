from __future__ import annotations

import importlib.util

from src_python.simulation.common import load_configs
from src_python.simulation.parameter_distributions import validate_distribution_spec


ALLOWED_TIME_SCOPES = {
    "structural_all_time",
    "prospective_implementation",
    "observation_only",
}


def _registry() -> dict:
    return load_configs()["parameter_distributions"]


def test_sampled_parameters_declare_time_scope_and_consumers() -> None:
    registry = _registry()
    for block_name in (
        "global_sensitivity",
        "bayesian_joint",
        "joint_rank_psa",
        "resistance_management_psa",
    ):
        for parameter_name, spec in registry[block_name]["parameters"].items():
            context = f"{block_name}.parameters.{parameter_name}"
            normalized = validate_distribution_spec(spec, context=context)
            assert normalized["time_scope"] in ALLOWED_TIME_SCOPES
            assert normalized["consumers"]
            assert all(isinstance(consumer, str) and consumer for consumer in normalized["consumers"])
            assert isinstance(normalized.get("source_ids"), list)
            assert all(
                source_id in registry["literature_sources"]
                for source_id in normalized["source_ids"]
            )


def test_declared_distribution_consumers_are_importable_modules() -> None:
    registry = _registry()
    for block_name in (
        "global_sensitivity",
        "bayesian_joint",
        "joint_rank_psa",
        "resistance_management_psa",
    ):
        for parameter_name, spec in registry[block_name]["parameters"].items():
            for consumer in spec["consumers"]:
                assert importlib.util.find_spec(consumer) is not None, (
                    f"{block_name}.parameters.{parameter_name} declares missing "
                    f"consumer module {consumer!r}"
                )
            for reader in spec.get("contract_readers", []):
                assert importlib.util.find_spec(reader) is not None, (
                    f"{block_name}.parameters.{parameter_name} declares missing "
                    f"contract-reader module {reader!r}"
                )


def test_global_screen_uses_under18_endpoint_except_for_reporting() -> None:
    block = _registry()["global_sensitivity"]
    assert block["primary_outcome"] == "total_child_adolescent_cases"
    reporting = block["parameters"]["reporting_multiplier_factor"]
    assert reporting["outcome"] == "total_reported_cases"
    assert reporting["time_scope"] == "observation_only"
    assert all(
        spec["time_scope"] == "structural_all_time"
        for name, spec in block["parameters"].items()
        if name != "reporting_multiplier_factor"
    )


def test_figure2b_and_resistance_management_have_disjoint_parameter_contracts() -> None:
    registry = _registry()
    figure2b = registry["joint_rank_psa"]
    expected_figure2b = {
        "infant_contact_multiplier",
        "VE_inf_baseline",
        "relative_infectiousness_asymptomatic",
        "infectious_duration_asymptomatic",
        "fitness_R",
        "PEP_coverage_multiplier",
    }
    assert figure2b["primary_outcome"] == "total_child_adolescent_cases"
    assert set(figure2b["parameters"]) == expected_figure2b
    grid_overrides = {"VE_inf_baseline", "fitness_R"}
    for name, spec in figure2b["parameters"].items():
        assert spec["consumers"][0] == (
            "src_python.simulation.run_joint_psa_rank_acceptability"
        )
        if name in grid_overrides:
            assert spec["consumers"] == [
                "src_python.simulation.run_joint_psa_rank_acceptability"
            ]
            assert spec["contract_readers"] == [
                "src_python.simulation.run_fitness_grid"
            ]
            assert "replaces its sampled value" in spec["consumer_note"]
        else:
            assert spec["consumers"] == [
                "src_python.simulation.run_joint_psa_rank_acceptability",
                "src_python.simulation.run_fitness_grid",
            ]
    assert figure2b["parameters"]["PEP_coverage_multiplier"]["time_scope"] == (
        "prospective_implementation"
    )
    assert all(
        spec["time_scope"] == "structural_all_time"
        for name, spec in figure2b["parameters"].items()
        if name != "PEP_coverage_multiplier"
    )

    resistance = registry["resistance_management_psa"]
    assert resistance["analysis_horizon_years"] == 5
    assert resistance["comparator"] == "routine_timeliness"
    assert resistance["primary_outcome"] == "total_child_adolescent_cases"
    assert resistance["secondary_outcome"] == "resistant_infections"
    restoration = resistance["structural_strata"]["pep_restoration"]
    assert restoration["levels"] == ["restored", "not_restored"]
    assert restoration["probability_assigned"] is False
    assert set(resistance["parameters"]) == {
        "resistance_management_uptake",
        "resistance_management_pep_reach_multiplier",
    }
    for spec in resistance["parameters"].values():
        assert spec["time_scope"] == "prospective_implementation"
        assert spec["consumers"] == [
            "src_python.simulation.run_resistance_management_psa"
        ]
    reach = resistance["parameters"][
        "resistance_management_pep_reach_multiplier"
    ]
    assert reach["distribution"] == "uniform"
    assert (reach["low"], reach["high"]) == (0.5, 1.0)


def test_bayesian_joint_registry_is_not_a_publication_interval_source() -> None:
    block = _registry()["bayesian_joint"]
    assert block["publication_path"] is False
    assert "not used for Figure 2c" in block["description"]
