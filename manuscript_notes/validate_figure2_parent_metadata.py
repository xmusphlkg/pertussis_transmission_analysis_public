"""Fail closed when Figure 2 parent artifacts are missing or semantically wrong."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import validate_run_metadata
from src_python.simulation.run_joint_psa_rank_acceptability import (
    FIGURE2B_PARAMETER_NAMES,
)


FIGURE2_PARENT_STEMS = (
    "figure2_programme_reference",
    "figure2c_parametric_bootstrap",
    "figure2c_parametric_bootstrap_quality_audit",
    "joint_psa_rank_acceptability",
    "intervention_scenarios",
)

EXPECTED_FIGURE2C_INTERVAL_TYPE = "95% parametric-bootstrap confidence interval"
EXPECTED_FIGURE2C_INTERVAL_BASIS = (
    "Paired marginal parametric bootstrap. Annual latent transmission paths "
    "are regenerated from the prespecified AR(1) process, surveillance counts "
    "are regenerated from the fitted NB2 observation model, and the country "
    "state-space model is fully refitted before the same fitted replicate is "
    "propagated through current practice and intervention. Biological and "
    "intervention-definition inputs remain fixed at prespecified reference values."
)
EXPECTED_FIGURE2C_VARIED_COMPONENTS = (
    "annual_AR1_latent_transmission_path",
    "NB2_surveillance_observations",
    "state_space_MAP_refit",
)
EXPECTED_FIGURE2C_FIXED_INPUTS = (
    "biological_parameters",
    "intervention_definitions",
    "AR1_hyperparameters",
    "NB2_dispersion",
)
EXPECTED_FIGURE2C_REPLICATES_PER_COUNTRY = 1024
EXPECTED_FIGURE2C_COUNTRIES = 9
EXPECTED_FIGURE2C_STRATEGIES = 6


def _validate_figure2c_estimation_ci_metadata(
    bootstrap: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """Require one complete, audited full-refit estimation-CI parent."""

    exact_fields = {
        "statistical_target": "frequentist_confidence_interval",
        "interval_type": EXPECTED_FIGURE2C_INTERVAL_TYPE,
        "interval_basis": EXPECTED_FIGURE2C_INTERVAL_BASIS,
        "confidence_interval_method": "percentile_parametric_bootstrap",
        "bootstrap_data_generation": "marginal_AR1_process_plus_NB2_measurement",
        "bootstrap_refit": "country_state_space_MAP_full_refit_per_replicate",
    }
    for field, expected in exact_fields.items():
        if bootstrap.get(field) != expected:
            raise ValueError(
                f"Figure 2c parent has the wrong {field}; expected {expected!r}"
            )
    if bootstrap.get("analysis_role") != "publication_estimation_confidence_interval":
        raise ValueError("Figure 2c parent has the wrong publication estimation-CI role")
    if tuple(bootstrap.get("varied_estimation_components", ())) != (
        EXPECTED_FIGURE2C_VARIED_COMPONENTS
    ):
        raise ValueError("Figure 2c parent does not vary the locked estimation components")
    if tuple(bootstrap.get("fixed_reference_inputs", ())) != (
        EXPECTED_FIGURE2C_FIXED_INPUTS
    ):
        raise ValueError("Figure 2c parent does not fix the locked reference inputs")
    if bootstrap.get("paired_scenario_contrast") is not True:
        raise ValueError("Figure 2c parent is not a paired scenario contrast")
    if bootstrap.get("publication_path") is not True:
        raise ValueError("Figure 2c parent is not marked as the publication path")
    if bootstrap.get("figure2c_interval_source") is not True:
        raise ValueError("Figure 2c parent is not marked as the Figure 2c interval source")
    if bootstrap.get("posterior_credible_interval") is not False:
        raise ValueError("Figure 2c parent must not be a posterior credible interval")
    if bootstrap.get("future_observation_prediction_interval") is not False:
        raise ValueError("Figure 2c parent must not be a future-observation interval")

    countries = tuple(str(value) for value in bootstrap.get("countries", ()))
    strategies = tuple(str(value) for value in bootstrap.get("strategies", ()))
    if len(countries) != EXPECTED_FIGURE2C_COUNTRIES or len(set(countries)) != len(countries):
        raise ValueError("Figure 2c parent does not contain nine unique country profiles")
    if len(strategies) != EXPECTED_FIGURE2C_STRATEGIES or len(set(strategies)) != len(strategies):
        raise ValueError("Figure 2c parent does not contain six unique programme strategies")
    requested = int(bootstrap.get("replicates_requested_per_country", -1))
    if requested != EXPECTED_FIGURE2C_REPLICATES_PER_COUNTRY:
        raise ValueError("Figure 2c parent does not contain 1024 requested refits per country")
    success = bootstrap.get("successful_replicates_by_country", {})
    if not isinstance(success, dict) or set(success) != set(countries):
        raise ValueError("Figure 2c parent has incomplete country refit counts")
    successful_counts = {country: int(success[country]) for country in countries}
    minimum_count = int(bootstrap.get("minimum_successful_replicates", -1))
    minimum_fraction = float(bootstrap.get("minimum_success_fraction", -1.0))
    if minimum_count != 1000 or minimum_fraction != 0.95:
        raise ValueError("Figure 2c parent has the wrong successful-refit gate")
    if any(
        count < minimum_count
        or count > requested
        or count / requested < minimum_fraction
        for count in successful_counts.values()
    ):
        raise ValueError("Figure 2c parent did not meet the country successful-refit gate")

    row_counts = bootstrap.get("row_counts", {})
    expected_fits = requested * len(countries)
    expected_interval_rows = len(countries) * len(strategies)
    expected_draw_rows = sum(successful_counts.values()) * len(strategies)
    expected_rows = {
        "fit_diagnostics": expected_fits,
        "confidence_interval_rows": expected_interval_rows,
        "interval_stability_rows": expected_interval_rows,
        "paired_bootstrap_draws": expected_draw_rows,
    }
    if not isinstance(row_counts, dict) or any(
        int(row_counts.get(key, -1)) != value for key, value in expected_rows.items()
    ):
        raise ValueError("Figure 2c parent row counts do not prove a complete bootstrap run")
    threshold = float(bootstrap.get("maximum_tail_probability_mcse_threshold", -1.0))
    observed = float(bootstrap.get("observed_maximum_tail_probability_mcse", float("inf")))
    if threshold != 0.005 or observed < 0.0 or observed > threshold:
        raise ValueError("Figure 2c parent did not meet the tail-probability MCSE gate")

    if (
        audit.get("analysis_role")
        != "publication_estimation_confidence_interval_quality_audit"
        or audit.get("publication_path") is not True
        or audit.get("figure2c_interval_source") is not False
        or audit.get("audits_figure2c_interval_source") is not True
    ):
        raise ValueError("Figure 2c quality audit has the wrong publication role")
    if audit.get("passed") is not True or audit.get("warnings_are_fatal") is not True:
        raise ValueError("Figure 2c parametric-bootstrap quality audit did not pass")
    if audit.get("figure2c_source_stem") != "figure2c_parametric_bootstrap":
        raise ValueError("Figure 2c quality audit points to the wrong source stem")
def validate_figure2_parent_metadata() -> dict[str, dict[str, Any]]:
    metadata = {stem: validate_run_metadata(stem) for stem in FIGURE2_PARENT_STEMS}

    rank = metadata["joint_psa_rank_acceptability"]
    if tuple(rank.get("figure2b_parameter_names", ())) != FIGURE2B_PARAMETER_NAMES:
        raise ValueError("Figure 2b parent does not implement the locked six-input contract")
    if rank.get("excluded_dead_dimensions") != ["resistance_management_uptake"]:
        raise ValueError("Figure 2b parent does not explicitly exclude management uptake")

    bootstrap = metadata["figure2c_parametric_bootstrap"]
    audit = metadata["figure2c_parametric_bootstrap_quality_audit"]
    _validate_figure2c_estimation_ci_metadata(bootstrap, audit)
    return metadata


def main() -> None:
    validate_figure2_parent_metadata()
    print("Figure 2 parent metadata are current and satisfy the publication contract.")


if __name__ == "__main__":
    main()
