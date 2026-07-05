"""MCMC convergence diagnostics: R-hat, effective sample size, and chain summaries.

Implements split-R̂ (Vehtari et al. 2021, "Rank-normalized split-R̂") and bulk/tail
effective sample size for multi-chain posterior samples. These diagnostics are
essential for validating that the Bayesian uncertainty analysis has converged
before interpreting credible intervals.

References:
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
    Rank-normalization, folding, and localization: An improved R̂ for assessing
    convergence of MCMC. Bayesian Analysis, 16(2), 667-718.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def split_chains(chains: list[np.ndarray]) -> list[np.ndarray]:
    """Split each chain in half to detect within-chain non-stationarity."""
    split = []
    for chain in chains:
        n = len(chain)
        if n < 4:
            split.append(chain)
            continue
        mid = n // 2
        split.append(chain[:mid])
        split.append(chain[mid:])
    return split


def _chain_mean_var(chains: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute per-chain means and variances."""
    m = len(chains)
    means = np.array([np.mean(c) for c in chains])
    variances = np.array([np.var(c, ddof=1) if len(c) > 1 else 0.0 for c in chains])
    n = int(np.mean([len(c) for c in chains]))
    return means, variances, n


def rhat(chains: list[np.ndarray]) -> float:
    """Compute split-R̂ for a single parameter across multiple chains.

    Returns R̂ ≈ 1.0 when chains have converged. Values > 1.01 suggest
    insufficient convergence.
    """
    split = split_chains(chains)
    if len(split) < 2:
        return float("nan")
    means, variances, n = _chain_mean_var(split)
    if n < 2:
        return float("nan")
    m = len(split)
    B = n * float(np.var(means, ddof=1))  # between-chain variance
    W = float(np.mean(variances))  # within-chain variance
    if W <= 0.0:
        return float("nan")
    var_hat = (1.0 - 1.0 / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def rhat_rank_normalized(chains: list[np.ndarray]) -> float:
    """Rank-normalized split-R̂ (Vehtari et al. 2021).

    More robust than basic R̂ for heavy-tailed or multimodal posteriors.
    """
    all_values = np.concatenate(chains)
    if len(all_values) < 4:
        return float("nan")
    # Rank-normalize: replace values with normal quantiles of their ranks
    ranks = _fractional_ranks(all_values)
    z_scores = _normal_quantile(ranks)

    # Redistribute ranked values back into chain structure
    ranked_chains = []
    offset = 0
    for chain in chains:
        n = len(chain)
        ranked_chains.append(z_scores[offset:offset + n])
        offset += n

    return rhat(ranked_chains)


def _fractional_ranks(values: np.ndarray) -> np.ndarray:
    """Compute fractional ranks (average rank for ties), scaled to (0, 1)."""
    n = len(values)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Handle ties by averaging
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=float))
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j
    # Scale to (0, 1)
    return (ranks - 0.5) / n


def _normal_quantile(p: np.ndarray) -> np.ndarray:
    """Inverse normal CDF (probit) for rank normalization."""
    from scipy.special import ndtri
    return ndtri(np.clip(p, 1e-10, 1.0 - 1e-10))


def effective_sample_size(chains: list[np.ndarray]) -> float:
    """Bulk effective sample size using autocorrelation-based estimator.

    Uses the initial positive sequence estimator (Geyer 1992) for stable
    ESS estimation.
    """
    split = split_chains(chains)
    if len(split) < 2:
        return float("nan")
    means, variances, n = _chain_mean_var(split)
    m = len(split)
    B = n * float(np.var(means, ddof=1))
    W = float(np.mean(variances))
    if W <= 0.0:
        return float("nan")
    var_hat = (1.0 - 1.0 / n) * W + B / n

    # Compute autocorrelation-based correction
    # Use variogram-based estimator for robustness
    max_lag = n - 1
    rho_hat = np.zeros(max_lag)
    for lag in range(1, max_lag):
        variogram = 0.0
        count = 0
        for chain in split:
            if len(chain) <= lag:
                continue
            diff = chain[lag:] - chain[:-lag]
            variogram += float(np.sum(diff ** 2))
            count += len(chain) - lag
        if count > 0:
            V_t = variogram / count
            rho_hat[lag] = 1.0 - V_t / (2.0 * var_hat)
        else:
            break

    # Initial positive sequence: sum consecutive pairs until negative
    tau = 1.0
    for t in range(1, max_lag - 1, 2):
        pair_sum = rho_hat[t] + rho_hat[t + 1] if t + 1 < max_lag else rho_hat[t]
        if pair_sum < 0:
            break
        tau += 2.0 * pair_sum

    tau = max(tau, 1.0)
    total_draws = sum(len(c) for c in split)
    return float(total_draws / tau)


def tail_effective_sample_size(chains: list[np.ndarray], quantile: float = 0.05) -> float:
    """Tail ESS: effective sample size for extreme quantiles.

    Computes ESS for the indicator I(x <= q) where q is the specified quantile,
    which measures how well the tails are explored.
    """
    all_values = np.concatenate(chains)
    threshold = float(np.quantile(all_values, quantile))
    indicator_chains = [np.asarray(c <= threshold, dtype=float) for c in chains]
    ess_low = effective_sample_size(indicator_chains)

    threshold_high = float(np.quantile(all_values, 1.0 - quantile))
    indicator_chains_high = [np.asarray(c >= threshold_high, dtype=float) for c in chains]
    ess_high = effective_sample_size(indicator_chains_high)

    # Return minimum of lower and upper tail ESS
    if np.isfinite(ess_low) and np.isfinite(ess_high):
        return min(ess_low, ess_high)
    if np.isfinite(ess_low):
        return ess_low
    return ess_high


def compute_diagnostics(
    samples: pd.DataFrame,
    parameter_columns: tuple[str, ...],
    chain_column: str = "chain",
    country_column: str = "country",
    drop_constant: bool = True,
    constant_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Compute R̂, ESS, and tail-ESS for all parameters across all countries.

    Parameters
    ----------
    samples : DataFrame with posterior draws including chain and country identifiers.
    parameter_columns : Names of parameter columns to diagnose.
    chain_column : Column identifying the chain index.
    country_column : Column identifying the country.

    Returns
    -------
    DataFrame with one row per non-constant (country, parameter) and columns:
        rhat, rhat_rank, bulk_ess, tail_ess, n_chains, total_draws,
        converged (True if rhat < 1.05 and bulk_ess > 100)

    Fixed parameters are excluded by default. Including them makes R-hat and ESS
    either undefined or spuriously tiny, which obscures the diagnostics for
    genuinely sampled dimensions.
    """
    rows: list[dict[str, Any]] = []
    for country, group in samples.groupby(country_column, sort=False):
        chain_ids = sorted(group[chain_column].unique())
        for param in parameter_columns:
            values_by_chain = [
                group.loc[group[chain_column].eq(cid), param].to_numpy(dtype=float)
                for cid in chain_ids
            ]
            # Filter out empty chains
            values_by_chain = [c for c in values_by_chain if len(c) > 0]
            if values_by_chain and drop_constant:
                all_values = np.concatenate(values_by_chain)
                if (
                    len(all_values) > 0
                    and np.nanmax(all_values) - np.nanmin(all_values)
                    <= float(constant_tolerance)
                ):
                    continue
            if len(values_by_chain) < 2:
                rows.append({
                    "country": country,
                    "parameter": param,
                    "rhat": float("nan"),
                    "rhat_rank": float("nan"),
                    "bulk_ess": float("nan"),
                    "tail_ess": float("nan"),
                    "n_chains": len(values_by_chain),
                    "total_draws": sum(len(c) for c in values_by_chain),
                    "converged": False,
                    "mean": float("nan"),
                    "sd": float("nan"),
                })
                continue

            r = rhat(values_by_chain)
            r_rank = rhat_rank_normalized(values_by_chain)
            ess_bulk = effective_sample_size(values_by_chain)
            ess_tail = tail_effective_sample_size(values_by_chain)
            total_draws = sum(len(c) for c in values_by_chain)
            all_values = np.concatenate(values_by_chain)

            converged = (
                np.isfinite(r_rank)
                and r_rank < 1.05
                and np.isfinite(ess_bulk)
                and ess_bulk > 100
            )

            rows.append({
                "country": country,
                "parameter": param,
                "rhat": float(r),
                "rhat_rank": float(r_rank),
                "bulk_ess": float(ess_bulk),
                "tail_ess": float(ess_tail) if np.isfinite(ess_tail) else float("nan"),
                "n_chains": len(values_by_chain),
                "total_draws": total_draws,
                "converged": bool(converged),
                "mean": float(np.mean(all_values)),
                "sd": float(np.std(all_values, ddof=1)),
            })

    return pd.DataFrame(rows)


def summarize_convergence(diagnostics: pd.DataFrame) -> dict[str, Any]:
    """Produce a high-level convergence summary from per-parameter diagnostics."""
    if diagnostics.empty:
        return {
            "all_converged": False,
            "n_parameters_total": 0,
            "n_parameters_converged": 0,
            "worst_rhat": float("nan"),
            "min_bulk_ess": float("nan"),
            "min_tail_ess": float("nan"),
            "countries_with_issues": [],
        }

    n_total = len(diagnostics)
    n_converged = int(diagnostics["converged"].sum())
    worst_rhat = float(diagnostics["rhat_rank"].max()) if diagnostics["rhat_rank"].notna().any() else float("nan")
    min_bulk = float(diagnostics["bulk_ess"].min()) if diagnostics["bulk_ess"].notna().any() else float("nan")
    min_tail = float(diagnostics["tail_ess"].min()) if diagnostics["tail_ess"].notna().any() else float("nan")

    issues = diagnostics.loc[~diagnostics["converged"]]
    countries_with_issues = sorted(issues["country"].unique().tolist()) if not issues.empty else []

    return {
        "all_converged": bool(n_converged == n_total),
        "n_parameters_total": n_total,
        "n_parameters_converged": n_converged,
        "fraction_converged": n_converged / max(n_total, 1),
        "worst_rhat": worst_rhat,
        "min_bulk_ess": min_bulk,
        "min_tail_ess": min_tail,
        "countries_with_issues": countries_with_issues,
    }
