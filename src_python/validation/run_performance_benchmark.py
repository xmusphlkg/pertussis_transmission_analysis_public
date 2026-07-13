"""Reproducible end-to-end benchmark for full output and likelihood paths."""

from __future__ import annotations

import argparse
from copy import deepcopy
from time import perf_counter

import pandas as pd

from src_python.calibration.calibrate_baseline import (
    calibration_observed_case_frame,
    calibration_runtime_config,
)
from src_python.model.observables import project_reported_cases
from src_python.simulation.common import (
    load_configs,
    make_config,
    run_prepared_case_exposure,
    run_prepared_config,
)
from src_python.utils.io import project_path, write_dataframe


def _timed(callable_):
    start = perf_counter()
    value = callable_()
    return value, perf_counter() - start


def benchmark_country(country: str) -> pd.DataFrame:
    configs = load_configs()
    observed = calibration_observed_case_frame(country)
    base = make_config(
        vaccine_scenario=configs["baseline"]["baseline_vaccine_scenario"],
        resistance_scenario=configs["baseline"]["baseline_resistance_scenario"],
        country_profile=country,
        load_calibration=False,
    )
    runtime = calibration_runtime_config(base, observed)

    reference = deepcopy(runtime)
    reference["simulation"]["rhs_backend"] = "python_reference"
    _, reference_seconds = _timed(
        lambda: run_prepared_config(
            reference,
            analysis="performance_benchmark",
            scenario="python_reference_full_output",
            vaccine_scenario=reference["baseline_vaccine_scenario"],
            resistance_scenario=reference["baseline_resistance_scenario"],
            metadata={"country": country},
        )
    )

    _, compiled_seconds = _timed(
        lambda: run_prepared_config(
            runtime,
            analysis="performance_benchmark",
            scenario="numba_full_output",
            vaccine_scenario=runtime["baseline_vaccine_scenario"],
            resistance_scenario=runtime["baseline_resistance_scenario"],
            metadata={"country": country},
        )
    )

    (exposure, base_rates), likelihood_seconds = _timed(
        lambda: run_prepared_case_exposure(
            runtime,
            observed,
            analysis="performance_benchmark",
            scenario="numba_case_exposure",
            vaccine_scenario=runtime["baseline_vaccine_scenario"],
            resistance_scenario=runtime["baseline_resistance_scenario"],
            metadata={"country": country},
        )
    )
    _, reporting_seconds = _timed(
        lambda: project_reported_cases(
            exposure,
            base_rates,
            float(runtime.get("reporting_multiplier", 1.0)) * 1.01,
        )
    )
    return pd.DataFrame(
        [
            {
                "country": country,
                "reference_full_output_seconds": reference_seconds,
                "compiled_full_output_seconds": compiled_seconds,
                "compiled_case_exposure_seconds": likelihood_seconds,
                "reporting_reprojection_seconds": reporting_seconds,
                "full_output_speedup": reference_seconds / compiled_seconds,
                "likelihood_vs_reference_speedup": reference_seconds / likelihood_seconds,
                "reporting_reprojection_speedup": reference_seconds / max(reporting_seconds, 1e-12),
                "case_event_definition": exposure.case_event_definition,
                "n_observation_intervals": len(observed),
                "case_exposure_nfev": int(exposure.numerical_diagnostics.get("nfev", 0)),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="Australia")
    parser.add_argument(
        "--output",
        default="outputs/tables/performance_benchmark.csv",
    )
    args = parser.parse_args()
    result = benchmark_country(args.country)
    write_dataframe(result, project_path(args.output))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
