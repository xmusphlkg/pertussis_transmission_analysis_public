from __future__ import annotations

"""Summarize isolated Bayesian MCMC pilot runs.

Pilot stems are the values passed to ``run_bayesian_uncertainty --output-stem``.
The script reads ``<stem>_posterior_samples`` and ``<stem>_convergence_diagnostics``
artifacts, then writes a compact comparison table.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_python.utils.io import project_path, read_table, write_dataframe


SAMPLED_PARAMETERS = (
    "beta_S",
    "reporting_multiplier",
    "VE_sus",
    "VE_inf",
    "relative_infectiousness_asymptomatic",
    "infectious_duration_symptomatic",
    "infectious_duration_asymptomatic",
    "fitness_R",
)


def _read_optional(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except FileNotFoundError:
        return pd.DataFrame()


def summarize_stem(stem: str) -> pd.DataFrame:
    samples_path = project_path("outputs", "simulations", f"{stem}_posterior_samples.parquet")
    diagnostics_path = project_path("outputs", "summaries", f"{stem}_convergence_diagnostics.csv")
    samples = _read_optional(samples_path)
    diagnostics = _read_optional(diagnostics_path)

    if samples.empty:
        return pd.DataFrame(
            [
                {
                    "stem": stem,
                    "country": "",
                    "status": "missing_samples",
                }
            ]
        )

    parameter_cols = [col for col in SAMPLED_PARAMETERS if col in samples.columns]
    rows: list[dict[str, Any]] = []
    for country, group in samples.groupby("country", sort=False):
        chain_sizes = group.groupby("chain").size()
        transition_rates = []
        unique_states = []
        final_acceptance = []
        for _, chain_group in group.sort_values("draw").groupby("chain", sort=True):
            if parameter_cols:
                changed = chain_group[parameter_cols].diff().abs().sum(axis=1) > 1e-12
                transition_rates.append(float(changed.iloc[1:].mean()) if len(changed) > 1 else np.nan)
                unique_states.append(int(len(chain_group[parameter_cols].drop_duplicates())))
            final_acceptance.append(float(chain_group["accepted_fraction"].iloc[-1]))

        country_diag = diagnostics.loc[diagnostics["country"].eq(country)] if not diagnostics.empty else pd.DataFrame()
        row = {
            "stem": stem,
            "country": country,
            "status": "ok",
            "n_chains": int(chain_sizes.size),
            "draws_per_chain_min": int(chain_sizes.min()),
            "draws_per_chain_max": int(chain_sizes.max()),
            "parameters_diagnosed": int(len(country_diag)) if not country_diag.empty else 0,
            "parameters_converged": int(country_diag["converged"].sum()) if not country_diag.empty else 0,
            "worst_rhat_rank": (
                float(country_diag["rhat_rank"].max())
                if not country_diag.empty and country_diag["rhat_rank"].notna().any()
                else np.nan
            ),
            "min_bulk_ess": (
                float(country_diag["bulk_ess"].min())
                if not country_diag.empty and country_diag["bulk_ess"].notna().any()
                else np.nan
            ),
            "min_tail_ess": (
                float(country_diag["tail_ess"].min())
                if not country_diag.empty and country_diag["tail_ess"].notna().any()
                else np.nan
            ),
            "transition_rate_min": float(np.nanmin(transition_rates)) if transition_rates else np.nan,
            "transition_rate_mean": float(np.nanmean(transition_rates)) if transition_rates else np.nan,
            "unique_states_min": int(np.nanmin(unique_states)) if unique_states else 0,
            "unique_states_mean": float(np.nanmean(unique_states)) if unique_states else np.nan,
            "accepted_fraction_min": float(np.nanmin(final_acceptance)) if final_acceptance else np.nan,
            "accepted_fraction_mean": float(np.nanmean(final_acceptance)) if final_acceptance else np.nan,
            "accepted_fraction_max": float(np.nanmax(final_acceptance)) if final_acceptance else np.nan,
            "posterior_log_prob_mean": float(group["posterior_log_prob"].mean()),
            "posterior_log_prob_max": float(group["posterior_log_prob"].max()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main(stems: list[str], output: str | None = None) -> pd.DataFrame:
    if not stems:
        pattern = project_path("outputs", "simulations").glob("*_posterior_samples.parquet")
        stems = sorted(
            path.name.removesuffix("_posterior_samples.parquet")
            for path in pattern
            if path.name.startswith("pilot_")
        )
    frames = [summarize_stem(stem) for stem in stems]
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path = output or "outputs/summaries/mcmc_pilot_comparison.csv"
    write_dataframe(summary, project_path(output_path))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize Bayesian MCMC pilot stems.")
    parser.add_argument("stems", nargs="*", help="Pilot output stems to summarize")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()
    result = main(args.stems, output=args.output)
    if result.empty:
        print("No pilot outputs found.")
    else:
        print(result.to_string(index=False))
