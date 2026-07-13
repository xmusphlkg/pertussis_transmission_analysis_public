"""Stochastic observation overlay for the deterministic ODE posterior predictive.

The ODE delivers the expected trajectory for each Bayesian posterior draw. This
module layers two well-established sources of stochastic heterogeneity on top,
at the country-year aggregate scale:

1. Superspreading. Individual-level offspring distributions for respiratory
   pathogens are over-dispersed (Lloyd-Smith et al., Nature 2005, estimate
   individual dispersion k_individual ≈ 0.1-0.6). That individual-level
   heterogeneity propagates, *attenuated* by aggregation, into year-to-year
   surveillance counts. Empirical surveillance fits put the aggregate-scale
   negative-binomial dispersion in the k ≈ 5-50 range (Bretó et al. 2009;
   Endo et al. 2020 for the reconciliation between individual and
   aggregate-level overdispersion), which is the scale we parameterise here.
   The same dispersion is already used by the calibration likelihood
   (``dispersion: 50`` in ``config/model_settings.yaml``); the stochastic
   overlay uses a slightly more dispersed default (k ≈ 10) so the combined
   predictive interval reflects the transmission-level heterogeneity that does not collapse
   fully under aggregation, without re-inflating to individual-level
   branching-process variance.

2. Household / close-contact clustering. Case counts within households are
   correlated, which inflates the sampling variance of aggregate counts by a
   design effect ``DEFF = 1 + (m̄ - 1) * SAR`` (Ball, Mollison &
   Scalia-Tomba 1997), where ``m̄`` is the mean household size and SAR is the
   secondary attack rate inside households. For pertussis, household SAR is
   reported at 0.7-0.9 (CDC household transmission studies).

For an expected aggregate count ``μ`` we combine the two into an effective
negative-binomial dispersion via

    Var(Y) = μ * DEFF + μ² / k_ss
           ≜ μ + μ² / k_eff
    ⇒ 1 / k_eff = 1 / k_ss + (DEFF - 1) / μ

Country-level totals have large ``μ``, so household clustering's contribution
shrinks asymptotically as expected; at small ``μ`` (rare outcomes such as
resistant paediatric infections) household clustering becomes the dominant
variance source. Because the model reports annualized averages over a multi-year
analysis horizon, the aggregate superspreading dispersion is scaled by the
number of analysis years before drawing horizon-level counts. This avoids
treating a 26-year cumulative burden as a single annual observation and prevents
identical multiplicative intervals for all large-count outcomes. Draws from the
combined distribution are propagated alongside the parameter posterior so the
reported 95% predictive interval reflects *both* parameter uncertainty (Bayesian) and stochastic
observation variance (superspreading + household clustering), addressing the
Lavine et al. (2011) concern that long-term pertussis projections are sensitive
to parameter uncertainty by making the uncertainty explicit, decomposable, and
calibration-consistent.

This overlay is *pragmatic*: it does not replace an individual-based or
branching-process simulation, but it captures the two dispersion sources that
a deterministic compartmental model structurally cannot, without requiring a
full model rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StochasticOverlayConfig:
    enabled: bool = True
    random_seed: int = 20260510
    replicates_per_draw: int = 100
    superspreading_k: float = 10.0
    superspreading_k_min: float = 1.0
    household_mean_size: float = 3.5
    household_secondary_attack_rate: float = 0.80
    min_expected_count: float = 1.0
    credible_interval: tuple[float, float] = (2.5, 97.5)
    aggregate_over_analysis_years: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StochasticOverlayConfig":
        if not data:
            return cls()
        ci = data.get("credible_interval", (2.5, 97.5))
        if isinstance(ci, dict):
            ci = (float(ci.get("low", 2.5)), float(ci.get("high", 97.5)))
        else:
            ci = (float(ci[0]), float(ci[1]))
        return cls(
            enabled=bool(data.get("enabled", True)),
            random_seed=int(data.get("random_seed", 20260510)),
            replicates_per_draw=max(1, int(data.get("replicates_per_draw", 100))),
            superspreading_k=max(1e-3, float(data.get("superspreading_k", 10.0))),
            superspreading_k_min=max(1e-3, float(data.get("superspreading_k_min", 1.0))),
            household_mean_size=max(1.0, float(data.get("household_mean_size", 3.5))),
            household_secondary_attack_rate=float(
                np.clip(data.get("household_secondary_attack_rate", 0.80), 0.0, 1.0)
            ),
            min_expected_count=max(0.0, float(data.get("min_expected_count", 1.0))),
            credible_interval=ci,
            aggregate_over_analysis_years=bool(data.get("aggregate_over_analysis_years", True)),
        )


COUNT_OUTCOME_DENOMINATORS: dict[str, tuple[str, ...]] = {
    # annualized rate column -> (population column, analysis_years column)
    "annualized_infections_per_100k": ("total_population", "analysis_years"),
    "annualized_reported_cases_per_100k": ("total_population", "analysis_years"),
    "annualized_infant_cases_per_100k": ("infant_population", "analysis_years"),
    "annualized_infant_infections_per_100k": ("infant_population", "analysis_years"),
    "annualized_child_1_9_cases_per_100k": ("child_1_9_population", "analysis_years"),
    "annualized_adolescent_cases_per_100k": ("adolescent_population", "analysis_years"),
    "annualized_child_adolescent_cases_per_100k": ("child_adolescent_population", "analysis_years"),
    "annualized_child_adolescent_infections_per_100k": ("child_adolescent_population", "analysis_years"),
    "annualized_child_adolescent_reported_cases_per_100k": ("child_adolescent_population", "analysis_years"),
}

COUNT_TOTAL_OUTCOMES: tuple[str, ...] = (
    "total_infections",
    "total_reported_cases",
    "total_infant_cases",
    "total_infant_infections",
    "total_child_1_9_cases",
    "total_adolescent_cases",
    "total_child_adolescent_cases",
    "total_child_adolescent_infections",
    "total_child_adolescent_reported_cases",
    "resistant_infections",
)


def design_effect(mean_size: float, secondary_attack_rate: float) -> float:
    """Household/close-contact design effect (Ball, Mollison & Scalia-Tomba 1997)."""
    size = max(float(mean_size), 1.0)
    sar = float(np.clip(secondary_attack_rate, 0.0, 1.0))
    return float(1.0 + (size - 1.0) * sar)


def effective_dispersion(
    expected_count: float,
    *,
    superspreading_k: float,
    design_effect_value: float,
    superspreading_k_min: float = 0.05,
    aggregation_units: float = 1.0,
) -> float:
    """Combined negative-binomial dispersion ``k_eff`` for superspreading + clustering."""
    mu = max(float(expected_count), 0.0)
    units = max(float(aggregation_units), 1.0) if np.isfinite(aggregation_units) else 1.0
    k_ss = max(float(superspreading_k) * units, float(superspreading_k_min))
    deff = max(float(design_effect_value), 1.0)
    if mu <= 0.0:
        return k_ss
    inv_k_eff = 1.0 / k_ss + max(deff - 1.0, 0.0) / mu
    if inv_k_eff <= 0.0:
        return float("inf")
    return float(1.0 / inv_k_eff)


def sample_negative_binomial(
    expected_count: float,
    *,
    dispersion_k: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``size`` NegBin samples with mean ``expected_count`` and dispersion ``k``.

    Uses the mean/dispersion (NB2) parameterisation with Var = μ + μ^2 / k.
    """
    mu = max(float(expected_count), 0.0)
    if mu <= 0.0 or not np.isfinite(mu):
        return np.zeros(size, dtype=float)
    k = float(dispersion_k)
    if not np.isfinite(k) or k <= 0.0:
        return rng.poisson(lam=mu, size=size).astype(float)
    p = k / (k + mu)
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return rng.negative_binomial(n=k, p=p, size=size).astype(float)


def _resolve_expected_count(
    row: pd.Series,
    outcome: str,
) -> tuple[float, float, float]:
    """Return expected count, rate denominator, and aggregation units.

    ``rate_denominator`` converts a sampled count back into the outcome's natural
    unit (``annualized_*_per_100k`` rate or raw count).
    """
    if outcome in COUNT_OUTCOME_DENOMINATORS:
        pop_col, years_col = COUNT_OUTCOME_DENOMINATORS[outcome]
        population = float(pd.to_numeric(pd.Series([row.get(pop_col, np.nan)]), errors="coerce").iloc[0])
        years = float(pd.to_numeric(pd.Series([row.get(years_col, np.nan)]), errors="coerce").iloc[0])
        rate = float(pd.to_numeric(pd.Series([row.get(outcome, np.nan)]), errors="coerce").iloc[0])
        if not (np.isfinite(population) and np.isfinite(years) and np.isfinite(rate)):
            return (np.nan, np.nan, np.nan)
        if population <= 0.0 or years <= 0.0:
            return (np.nan, np.nan, np.nan)
        expected = rate * population * years / 100_000.0
        # Rate = count / (population * years) * 1e5
        denominator = max(population * years, 1e-9) / 100_000.0
        return (float(expected), float(denominator), float(years))
    if outcome in COUNT_TOTAL_OUTCOMES:
        expected = float(pd.to_numeric(pd.Series([row.get(outcome, np.nan)]), errors="coerce").iloc[0])
        if not np.isfinite(expected):
            return (np.nan, np.nan, np.nan)
        years = float(pd.to_numeric(pd.Series([row.get("analysis_years", 1.0)]), errors="coerce").iloc[0])
        if not np.isfinite(years) or years <= 0.0:
            years = 1.0
        return (float(expected), 1.0, float(years))
    return (np.nan, np.nan, np.nan)


def stochastic_overlay_samples(
    summary: pd.DataFrame,
    *,
    overlay: StochasticOverlayConfig,
    outcomes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Generate stochastic replicates layered on top of each posterior draw.

    Returns a long-format DataFrame with columns ``country``, ``outcome``,
    ``posterior_draw``, ``replicate``, ``value`` (outcome value after the
    stochastic overlay in its native unit), plus diagnostic columns.
    """
    outcomes = tuple(outcomes) if outcomes is not None else (
        *COUNT_OUTCOME_DENOMINATORS.keys(),
        *COUNT_TOTAL_OUTCOMES,
    )
    rng = np.random.default_rng(overlay.random_seed)
    rows: list[dict[str, Any]] = []
    deff_value = design_effect(overlay.household_mean_size, overlay.household_secondary_attack_rate)

    for _, row in summary.iterrows():
        country = row.get("country", "")
        draw = row.get("posterior_draw", np.nan)
        for outcome in outcomes:
            if outcome not in row.index:
                continue
            expected_count, rate_denominator, aggregation_units = _resolve_expected_count(row, outcome)
            if not np.isfinite(expected_count) or expected_count < overlay.min_expected_count:
                continue
            units = aggregation_units if overlay.aggregate_over_analysis_years else 1.0
            k_eff = effective_dispersion(
                expected_count,
                superspreading_k=overlay.superspreading_k,
                design_effect_value=deff_value,
                superspreading_k_min=overlay.superspreading_k_min,
                aggregation_units=units,
            )
            counts = sample_negative_binomial(
                expected_count,
                dispersion_k=k_eff,
                size=overlay.replicates_per_draw,
                rng=rng,
            )
            values = counts / rate_denominator if rate_denominator > 0 else counts
            for replicate, value in enumerate(values, start=1):
                rows.append(
                    {
                        "country": country,
                        "outcome": outcome,
                        "posterior_draw": draw,
                        "replicate": replicate,
                        "value": float(value),
                        "expected_count": float(expected_count),
                        "aggregation_units": float(units),
                        "dispersion_k_effective": float(k_eff),
                        "design_effect": float(deff_value),
                        "superspreading_k": float(overlay.superspreading_k),
                        "superspreading_k_effective": float(overlay.superspreading_k * units),
                    }
                )
    return pd.DataFrame(rows)


def summarize_overlay_intervals(
    summary: pd.DataFrame,
    overlay_samples: pd.DataFrame,
    *,
    overlay: StochasticOverlayConfig,
    outcomes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pool propagated-uncertainty draws with stochastic replicates into a 95% PI.

    Also emits a propagated-uncertainty-only 95% interval for comparison so the
    stochastic overlay's contribution is explicit. These names deliberately do
    not imply that every input draw comes from a joint posterior.
    """
    outcomes = tuple(outcomes) if outcomes is not None else (
        *COUNT_OUTCOME_DENOMINATORS.keys(),
        *COUNT_TOTAL_OUTCOMES,
    )
    low_q, high_q = overlay.credible_interval
    deff_value = design_effect(overlay.household_mean_size, overlay.household_secondary_attack_rate)

    rows: list[dict[str, Any]] = []
    for country, group in summary.groupby("country", sort=False):
        overlay_country = overlay_samples.loc[overlay_samples["country"].eq(country)] if not overlay_samples.empty else overlay_samples
        for outcome in outcomes:
            if outcome not in group.columns:
                continue
            parameter_only = pd.to_numeric(group[outcome], errors="coerce").dropna().to_numpy(dtype=float)
            if parameter_only.size == 0:
                continue
            param_low, param_median, param_high = np.percentile(parameter_only, [low_q, 50.0, high_q])

            overlay_for_outcome = (
                overlay_country.loc[overlay_country["outcome"].eq(outcome), "value"].to_numpy(dtype=float)
                if not overlay_country.empty
                else np.array([], dtype=float)
            )
            if overlay_for_outcome.size > 0:
                combined_low, combined_median, combined_high = np.percentile(
                    overlay_for_outcome, [low_q, 50.0, high_q]
                )
                has_overlay = True
            else:
                combined_low, combined_median, combined_high = param_low, param_median, param_high
                has_overlay = False

            rows.append(
                {
                    "country": country,
                    "outcome": outcome,
                    "propagated_uncertainty_median": float(param_median),
                    "propagated_uncertainty_interval_low": float(param_low),
                    "propagated_uncertainty_interval_high": float(param_high),
                    "combined_prediction_interval_median": float(combined_median),
                    "combined_prediction_interval_low": float(combined_low),
                    "combined_prediction_interval_high": float(combined_high),
                    "uncertainty_draws": int(parameter_only.size),
                    "interval_type": (
                        "conditional parameter/process plus stochastic prediction interval"
                    ),
                    "stochastic_replicates_per_draw": int(overlay.replicates_per_draw) if has_overlay else 0,
                    "stochastic_overlay_applied": bool(has_overlay),
                    "superspreading_k": float(overlay.superspreading_k),
                    "aggregate_over_analysis_years": bool(overlay.aggregate_over_analysis_years),
                    "household_design_effect": float(deff_value),
                    "interval_low_percentile": float(low_q),
                    "interval_high_percentile": float(high_q),
                }
            )
    return pd.DataFrame(rows)


def decompose_variance(
    overlay_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Partition total variance into parameter vs stochastic components per outcome.

    Uses the law of total variance over (posterior_draw, replicate):
    ``Var_total = Var_between_draws(mean) + E_draws(Var_within_draw)``.
    """
    if overlay_samples.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "outcome",
                "parameter_variance",
                "stochastic_variance",
                "total_variance",
                "parameter_variance_share",
            ]
        )
    grouped = overlay_samples.groupby(["country", "outcome", "posterior_draw"], sort=False)["value"]
    draw_mean = grouped.mean().rename("draw_mean")
    draw_var = grouped.var(ddof=1).fillna(0.0).rename("draw_var")
    diagnostics = pd.concat([draw_mean, draw_var], axis=1).reset_index()

    rows: list[dict[str, Any]] = []
    for (country, outcome), block in diagnostics.groupby(["country", "outcome"], sort=False):
        parameter_variance = float(np.var(block["draw_mean"].to_numpy(dtype=float), ddof=1)) if len(block) > 1 else 0.0
        stochastic_variance = float(np.mean(block["draw_var"].to_numpy(dtype=float)))
        total = parameter_variance + stochastic_variance
        share = parameter_variance / total if total > 0 else np.nan
        rows.append(
            {
                "country": country,
                "outcome": outcome,
                "parameter_variance": parameter_variance,
                "stochastic_variance": stochastic_variance,
                "total_variance": total,
                "parameter_variance_share": share,
            }
        )
    return pd.DataFrame(rows)
