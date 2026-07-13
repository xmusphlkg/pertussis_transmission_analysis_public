"""Diagnostic-only one-at-a-time sensitivity.

This legacy runner is not a hyperparameter-selection procedure because it
reuses in-sample fits and does not explore interactions.  Use
``run_state_space_hyperparameter_selection`` for the joint grid with
leave-one-country-out predictive evaluation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src_python.calibration.calibrate_baseline import (
    _calibration_predicted_means,
    calibration_observed_case_frame,
    calibration_runtime_config,
    extend_annual_log_beta_conditional_mean,
    grouped_likelihood_observations,
    state_space_map_calibration,
)
from src_python.calibration.likelihood import negative_binomial_nll
from src_python.simulation.common import load_configs, make_config, run_prepared_config
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import parallel_map


DEFAULT_VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "baseline", "rho": 0.50, "innovation_sd": 0.60, "dispersion": 50.0},
    {"variant": "rho_low", "rho": 0.20, "innovation_sd": 0.60, "dispersion": 50.0},
    {"variant": "rho_high", "rho": 0.80, "innovation_sd": 0.60, "dispersion": 50.0},
    {"variant": "innovation_low", "rho": 0.50, "innovation_sd": 0.30, "dispersion": 50.0},
    {"variant": "innovation_high", "rho": 0.50, "innovation_sd": 0.90, "dispersion": 50.0},
    {"variant": "dispersion_low", "rho": 0.50, "innovation_sd": 0.60, "dispersion": 10.0},
    {"variant": "dispersion_high", "rho": 0.50, "innovation_sd": 0.60, "dispersion": 100.0},
)


def _task(payload: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    country, variant = payload
    configs = load_configs()
    observed = calibration_observed_case_frame(country)
    likelihood_observed = grouped_likelihood_observations(observed)
    base = calibration_runtime_config(
        make_config(country_profile=country, load_calibration=False),
        observed,
    )
    fitted, result = state_space_map_calibration(
        base,
        country,
        observed,
        dispersion=float(variant["dispersion"]),
        maxiter=int(configs["baseline"].get("calibration", {}).get("maxiter", 12)),
        process_overrides={
            "ar1_rho": float(variant["rho"]),
            "log_beta_innovation_sd": float(variant["innovation_sd"]),
        },
    )
    predicted = _calibration_predicted_means(fitted, observed, country)
    actual = pd.to_numeric(
        likelihood_observed["reported_cases"], errors="raise"
    ).to_numpy(dtype=float)

    production = make_config(country_profile=country, load_calibration=False)
    production["transmission"]["beta_S"] = float(fitted["transmission"]["beta_S"])
    production["reporting_multiplier"] = float(fitted.get("reporting_multiplier", 1.0))
    extended = extend_annual_log_beta_conditional_mean(
        fitted,
        through_year=int(
            pd.Timestamp(configs["baseline"]["calendar"]["analysis_end_date"]).year
        ),
    )
    production["transmission"]["log_beta_time_variation"] = deepcopy(
        extended["transmission"]["log_beta_time_variation"]
    )
    _timeseries, summary = run_prepared_config(
        production,
        analysis="state_space_hyperparameter_sensitivity",
        scenario=f"{country}_{variant['variant']}",
        metadata={"country": country},
    )
    summary_row = summary.iloc[0]
    latent = np.asarray(result.process_vector, dtype=float)
    return {
        "country": country,
        **variant,
        "beta_S": float(fitted["transmission"]["beta_S"]),
        "reporting_multiplier": float(fitted.get("reporting_multiplier", 1.0)),
        "latent_process_sd": float(np.std(latent, ddof=0)),
        "latent_process_max_abs": float(np.max(np.abs(latent))),
        "fit_nb_nll": float(
            negative_binomial_nll(actual, predicted, float(variant["dispersion"]))
        ),
        "fit_smape": float(
            np.mean(2.0 * np.abs(predicted - actual) / np.maximum(predicted + actual, 1e-12))
        ),
        "fit_total_ratio": float(predicted.sum() / max(actual.sum(), 1e-12)),
        "posterior_rank": int(result.posterior_rank),
        "posterior_dimension": int(len(result.posterior_map_vector)),
        "posterior_condition_number": float(result.posterior_condition_number),
        "process_nfev": int(result.process_nfev),
        "forecast_annualized_child_adolescent_cases_per_100k": float(
            summary_row.get("annualized_child_adolescent_cases_per_100k", np.nan)
        ),
        "forecast_total_child_adolescent_cases": float(
            summary_row.get("total_child_adolescent_cases", np.nan)
        ),
        "interpretation": (
            "one_at_a_time_fixed_hyperparameter_sensitivity; not posterior weights"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only OAT state-space sensitivity; use "
            "run_state_space_hyperparameter_selection for configuration choice"
        )
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=["Australia", "China", "United_Kingdom"],
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    tasks = [
        (country, dict(variant))
        for country in args.countries
        for variant in DEFAULT_VARIANTS
    ]
    rows = parallel_map(
        _task,
        tasks,
        desc="state_space_hyperparameter_sensitivity",
        n_jobs=args.n_jobs,
    )
    write_dataframe(
        pd.DataFrame(rows),
        project_path("outputs", "tables", "state_space_hyperparameter_sensitivity.csv"),
    )


if __name__ == "__main__":
    main()
