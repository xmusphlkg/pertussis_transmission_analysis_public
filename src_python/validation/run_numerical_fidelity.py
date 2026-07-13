"""Country-level solver-refinement gates for the likelihood path."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src_python.calibration.calibrate_baseline import (
    calibration_observed_case_frame,
    calibration_runtime_config,
)
from src_python.calibration.likelihood import negative_binomial_nll
from src_python.model.observables import project_reported_cases
from src_python.simulation.common import (
    load_configs,
    make_config,
    run_prepared_case_exposure,
)
from src_python.utils.io import project_path, write_dataframe


DEFAULT_L1_RELATIVE_TOLERANCE = 5e-3
DEFAULT_MAX_INTERVAL_RELATIVE_TOLERANCE = 2e-2
DEFAULT_NLL_PER_INTERVAL_TOLERANCE = 0.10


def _prediction(
    config: dict[str, Any],
    observed: pd.DataFrame,
    country: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    exposure, base_reporting_rates = run_prepared_case_exposure(
        config,
        observed,
        analysis="numerical_fidelity",
        scenario=f"{country}_solver_refinement",
        vaccine_scenario=config["baseline_vaccine_scenario"],
        resistance_scenario=config["baseline_resistance_scenario"],
        metadata={"country": country},
    )
    prediction = project_reported_cases(
        exposure,
        base_reporting_rates,
        float(config.get("reporting_multiplier", 1.0)),
    ).interval_total_mean
    return prediction, exposure.numerical_diagnostics


def evaluate_country_numerical_fidelity(
    country: str,
    *,
    refinement_factor: float = 10.0,
) -> dict[str, Any]:
    if refinement_factor <= 1.0:
        raise ValueError("refinement_factor must be > 1")
    configs = load_configs()
    observed = calibration_observed_case_frame(country)
    base = make_config(
        vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
        resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
        country_profile=country,
        load_calibration=False,
    )
    coarse = calibration_runtime_config(base, observed)
    refined = deepcopy(coarse)
    refined["simulation"]["rtol"] = float(coarse["simulation"]["rtol"]) / refinement_factor
    refined["simulation"]["atol"] = float(coarse["simulation"]["atol"]) / refinement_factor

    coarse_prediction, coarse_diagnostics = _prediction(coarse, observed, country)
    refined_prediction, refined_diagnostics = _prediction(refined, observed, country)
    difference = coarse_prediction - refined_prediction
    denominator = np.maximum(np.abs(refined_prediction), 1.0)
    l1_relative = float(
        np.sum(np.abs(difference)) / max(float(np.sum(np.abs(refined_prediction))), 1.0)
    )
    max_interval_relative = float(np.max(np.abs(difference) / denominator))
    observed_values = observed["reported_cases"].to_numpy(dtype=float)
    dispersion = float(configs["baseline"]["calibration"].get("dispersion", 50.0))
    coarse_nll = negative_binomial_nll(observed_values, coarse_prediction, dispersion)
    refined_nll = negative_binomial_nll(observed_values, refined_prediction, dispersion)
    nll_per_interval_difference = float(
        abs(coarse_nll - refined_nll) / max(len(observed_values), 1)
    )
    passed = bool(
        l1_relative <= DEFAULT_L1_RELATIVE_TOLERANCE
        and max_interval_relative <= DEFAULT_MAX_INTERVAL_RELATIVE_TOLERANCE
        and nll_per_interval_difference <= DEFAULT_NLL_PER_INTERVAL_TOLERANCE
        and int(coarse_diagnostics.get("counter_negative_corrections", 0)) == 0
        and int(refined_diagnostics.get("counter_negative_corrections", 0)) == 0
    )
    return {
        "country": country,
        "case_event_definition": str(
            coarse.get("observation_model", {}).get("case_event_definition", "")
        ),
        "coarse_rtol": float(coarse["simulation"]["rtol"]),
        "coarse_atol": float(coarse["simulation"]["atol"]),
        "refined_rtol": float(refined["simulation"]["rtol"]),
        "refined_atol": float(refined["simulation"]["atol"]),
        "prediction_l1_relative_difference": l1_relative,
        "prediction_max_interval_relative_difference": max_interval_relative,
        "nll_per_interval_absolute_difference": nll_per_interval_difference,
        "coarse_nfev": int(coarse_diagnostics.get("nfev", 0)),
        "refined_nfev": int(refined_diagnostics.get("nfev", 0)),
        "coarse_min_state": float(coarse_diagnostics.get("min_state", np.nan)),
        "refined_min_state": float(refined_diagnostics.get("min_state", np.nan)),
        "numerical_fidelity_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", action="append", dest="countries")
    parser.add_argument("--refinement-factor", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default="outputs/tables/numerical_fidelity_gate.csv",
    )
    args = parser.parse_args()
    countries = args.countries or list(load_configs()["countries"])
    results = pd.DataFrame(
        [
            evaluate_country_numerical_fidelity(
                country,
                refinement_factor=args.refinement_factor,
            )
            for country in countries
        ]
    )
    write_dataframe(results, project_path(args.output))
    print(results.to_string(index=False))
    if not bool(results["numerical_fidelity_passed"].all()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
