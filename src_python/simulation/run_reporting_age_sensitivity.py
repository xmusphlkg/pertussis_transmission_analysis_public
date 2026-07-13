"""Age-specific reporting rate sensitivity analysis.

Exploratory simulation script: not part of the current publication pipeline.

This module systematically explores how uncertainty in age-specific reporting
rates affects model outputs, particularly:
  - Total estimated infection burden (true vs reported)
  - Age distribution of true infections vs reported cases
  - Infant-specific burden estimates
  - Resistance burden (which depends on total infections, not just reported)

Literature basis for reporting rate uncertainty:
  - Infants (0-11m): 30-70% reported (Ontario: 37% hospital cases reported;
    enhanced surveillance settings may reach 60-70%)
  - Children (1-6y): 5-50% (Sweden CHC: 52%; passive systems much lower)
  - School-age (7-17y): 3-20% (Ontario age 1+: 11%; England/Wales: 5-25%)
  - Adults (18+): 0.3-12% (US adults 50+: 1-2% diagnosed vs modelled;
    Birmingham GP study: <1.2%; Ontario extreme: 0.003%)

The analysis produces:
  1. Scenario-based sensitivity (configured scenarios from model_settings.yaml)
  2. Latin Hypercube sampling of age-specific reporting rates within literature bounds
  3. Partial rank correlation coefficients (PRCC) for each age group's reporting
     rate against key outcomes

References:
  - Clarkson & Fine 1985 (England/Wales notification efficiency)
  - Crowcroft et al. 2018 (Ontario capture-recapture)
  - Mark & Granstrom 1991 (Sweden CHC sensitivity)
  - Masseria et al. 2015 (US adults 50+)
  - Pereira et al. 2000 (Birmingham GP cough study)
  - Chen et al. 2016 (Yiwu active surveillance, China)
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from src_python.simulation.common import (
    execute_scenario_list,
    load_configs,
    make_config,
    publication_country_names,
    write_outputs,
)
from src_python.utils.io import project_path, write_dataframe


# Literature-derived bounds for age-specific reporting rates
# These represent the plausible range from the evidence summary
AGE_REPORTING_BOUNDS = {
    "infant_0_2m": {"min": 0.30, "max": 0.75, "baseline": 0.60},
    "infant_3_11m": {"min": 0.25, "max": 0.70, "baseline": 0.50},
    "child_1_4y": {"min": 0.05, "max": 0.50, "baseline": 0.25},
    "child_5_9y": {"min": 0.04, "max": 0.40, "baseline": 0.18},
    "adolescent_10_17y": {"min": 0.03, "max": 0.20, "baseline": 0.08},
    "young_adult_18_39y": {"min": 0.003, "max": 0.12, "baseline": 0.05},
    "middle_adult_40_64y": {"min": 0.003, "max": 0.10, "baseline": 0.03},
    "elderly_65plus": {"min": 0.003, "max": 0.12, "baseline": 0.04},
}

AGE_GROUPS = list(AGE_REPORTING_BOUNDS.keys())


def _apply_age_reporting_rates(
    config: dict[str, Any], rates: dict[str, float]
) -> dict[str, Any]:
    """Apply specific reporting rates to each age group."""
    out = deepcopy(config)
    for record in out["age_groups"]:
        label = record["label"]
        if label in rates:
            record["reporting_rate"] = float(np.clip(rates[label], 0.0, 1.0))
    return out


def _build_lhs_scenarios(
    configs: dict[str, Any],
    sample_size: int = 64,
    seed: int = 20260512,
) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    """Build Latin Hypercube sampled scenarios for age-specific reporting rates.

    Returns (scenarios, sample_matrix, parameter_names) where sample_matrix
    has shape (sample_size, n_age_groups) with values in the literature bounds.
    """
    n_params = len(AGE_GROUPS)
    lower = np.array([AGE_REPORTING_BOUNDS[ag]["min"] for ag in AGE_GROUPS])
    upper = np.array([AGE_REPORTING_BOUNDS[ag]["max"] for ag in AGE_GROUPS])

    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(sample_size)
    sample_matrix = qmc.scale(unit_samples, lower, upper)

    resistance_name = configs["baseline"].get(
        "baseline_resistance_scenario", "country_timeline"
    )
    scenarios = []

    for country in publication_country_names(configs):
        for run_idx, row in enumerate(sample_matrix, start=1):
            rates = {AGE_GROUPS[i]: float(row[i]) for i in range(n_params)}
            config = make_config(
                vaccine_scenario="symptom_protective",
                resistance_scenario=resistance_name,
                country_profile=country,
            )
            config = _apply_age_reporting_rates(config, rates)

            metadata = {
                "country": country,
                "lhs_run": run_idx,
                **{f"reporting_rate_{ag}": float(row[i]) for i, ag in enumerate(AGE_GROUPS)},
            }
            scenarios.append(
                {
                    "config": config,
                    "analysis": "reporting_age_sensitivity_lhs",
                    "scenario": f"lhs_{run_idx:03d}",
                    "vaccine_scenario": "symptom_protective",
                    "resistance_scenario": resistance_name,
                    "metadata": metadata,
                }
            )

    return scenarios, sample_matrix, [f"reporting_rate_{ag}" for ag in AGE_GROUPS]


def _compute_prcc(
    summary: pd.DataFrame,
    parameter_names: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Compute Partial Rank Correlation Coefficients (PRCC).

    PRCC measures the monotonic relationship between each input parameter
    and each outcome after removing the linear effects of all other parameters.
    Values near +/-1 indicate strong sensitivity; near 0 indicates insensitivity.
    """
    rows: list[dict[str, Any]] = []

    for country, group in summary.groupby("country", sort=False):
        # Extract parameter values
        param_cols = [c for c in parameter_names if c in group.columns]
        if not param_cols:
            continue

        X = group[param_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(X) < 10:
            continue

        for outcome in outcome_columns:
            if outcome not in group.columns:
                continue
            y = pd.to_numeric(group.loc[X.index, outcome], errors="coerce")
            valid = y.notna() & np.isfinite(y)
            X_valid = X.loc[valid]
            y_valid = y.loc[valid]

            if len(y_valid) < 10:
                continue

            # Rank transform
            X_ranked = X_valid.rank()
            y_ranked = y_valid.rank()

            for param in param_cols:
                # Partial correlation: regress out other parameters
                other_params = [p for p in param_cols if p != param]
                if other_params:
                    # Residualize param and outcome against other params
                    X_others = X_ranked[other_params].to_numpy(dtype=float)
                    x_param = X_ranked[param].to_numpy(dtype=float)
                    y_out = y_ranked.to_numpy(dtype=float)

                    # OLS residuals for param
                    X_aug = np.column_stack([np.ones(len(X_others)), X_others])
                    try:
                        beta_x = np.linalg.lstsq(X_aug, x_param, rcond=None)[0]
                        resid_x = x_param - X_aug @ beta_x

                        beta_y = np.linalg.lstsq(X_aug, y_out, rcond=None)[0]
                        resid_y = y_out - X_aug @ beta_y

                        if np.std(resid_x) > 0 and np.std(resid_y) > 0:
                            prcc_val = float(np.corrcoef(resid_x, resid_y)[0, 1])
                        else:
                            prcc_val = 0.0
                    except (np.linalg.LinAlgError, ValueError):
                        prcc_val = float("nan")
                else:
                    # Only one parameter: use Spearman directly
                    corr, _ = spearmanr(
                        X_ranked[param].to_numpy(), y_ranked.to_numpy()
                    )
                    prcc_val = float(corr)

                # Extract age group name from parameter name
                age_group = param.replace("reporting_rate_", "")

                rows.append(
                    {
                        "country": country,
                        "parameter": param,
                        "age_group": age_group,
                        "outcome": outcome,
                        "prcc": prcc_val,
                        "n_samples": int(len(y_valid)),
                        "abs_prcc": abs(prcc_val),
                    }
                )

    return pd.DataFrame(rows)


def _compute_outcome_elasticity(
    summary: pd.DataFrame,
    parameter_names: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Compute outcome elasticity: % change in outcome per % change in reporting rate.

    This gives an intuitive measure of how sensitive each outcome is to
    reporting rate changes in each age group.
    """
    rows: list[dict[str, Any]] = []

    for country, group in summary.groupby("country", sort=False):
        param_cols = [c for c in parameter_names if c in group.columns]
        if not param_cols:
            continue

        for outcome in outcome_columns:
            if outcome not in group.columns:
                continue
            y = pd.to_numeric(group[outcome], errors="coerce")
            valid = y.notna() & np.isfinite(y) & (y > 0)

            for param in param_cols:
                x = pd.to_numeric(group[param], errors="coerce")
                both_valid = valid & x.notna() & (x > 0)
                if both_valid.sum() < 10:
                    continue

                # Log-log regression for elasticity
                log_x = np.log(x[both_valid].to_numpy(dtype=float))
                log_y = np.log(y[both_valid].to_numpy(dtype=float))

                if np.std(log_x) > 0:
                    elasticity = float(
                        np.polyfit(log_x, log_y, 1)[0]
                    )
                else:
                    elasticity = 0.0

                age_group = param.replace("reporting_rate_", "")
                rows.append(
                    {
                        "country": country,
                        "parameter": param,
                        "age_group": age_group,
                        "outcome": outcome,
                        "elasticity": elasticity,
                        "interpretation": (
                            f"{elasticity:.2f}% change in {outcome} per 1% change "
                            f"in {age_group} reporting rate"
                        ),
                    }
                )

    return pd.DataFrame(rows)


def main(n_jobs: int | None = None, sample_size: int | None = None):
    """Run age-specific reporting rate sensitivity analysis.

    Produces:
      - LHS-sampled scenarios across literature reporting rate bounds
      - PRCC (Partial Rank Correlation Coefficients) for each age group
      - Outcome elasticity estimates
      - Summary of which age groups' reporting rates matter most
    """
    configs = load_configs()
    n_samples = sample_size or 64

    # Phase 1: Build and run LHS scenarios
    scenarios, sample_matrix, param_names = _build_lhs_scenarios(
        configs, sample_size=n_samples
    )

    timeseries, summary = execute_scenario_list(
        scenarios, stem="reporting_age_sensitivity", n_jobs=n_jobs
    )
    write_outputs(timeseries, summary, "reporting_age_sensitivity")

    # Phase 2: Compute sensitivity metrics
    outcome_cols = [
        "annualized_reported_cases_per_100k",
        "annualized_infections_per_100k",
        "annualized_infant_cases_per_100k",
        "annualized_infant_infections_per_100k",
        "total_reported_cases",
        "total_infections",
        "resistant_infections",
    ]

    prcc = _compute_prcc(summary, param_names, outcome_cols)
    write_dataframe(
        prcc,
        project_path("outputs/summaries/reporting_age_sensitivity_prcc.csv"),
    )

    elasticity = _compute_outcome_elasticity(summary, param_names, outcome_cols)
    write_dataframe(
        elasticity,
        project_path("outputs/summaries/reporting_age_sensitivity_elasticity.csv"),
    )

    # Phase 3: Summary table of most influential age groups per outcome
    if not prcc.empty:
        influence_summary = (
            prcc.groupby(["outcome", "age_group"])
            .agg(
                mean_abs_prcc=("abs_prcc", "mean"),
                mean_prcc=("prcc", "mean"),
                n_countries=("country", "nunique"),
            )
            .reset_index()
            .sort_values(["outcome", "mean_abs_prcc"], ascending=[True, False])
        )
        write_dataframe(
            influence_summary,
            project_path("outputs/summaries/reporting_age_influence_ranking.csv"),
        )

    return timeseries, summary, prcc, elasticity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run age-specific reporting rate sensitivity analysis."
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of LHS samples (default: 64)")
    args = parser.parse_args()
    main(n_jobs=args.n_jobs, sample_size=args.sample_size)
