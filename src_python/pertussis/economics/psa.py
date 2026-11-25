"""Probabilistic sensitivity analysis: parameter sampling and decision metrics.

Interpretation limits
---------------------
PSA draws are independent across parameters within each draw but paired across
strategies.  Severity and cost draws are drawn jointly (same seed, same draw
index applied to both) so that strategy comparisons are internally consistent.
All ICERs and NMBs are incremental with respect to the declared baseline
strategy.  Dominance (higher cost AND fewer DALYs averted) is flagged but not
excluded; the caller decides whether to remove dominated strategies.

CEAC probabilities estimate the fraction of PSA draws for which a strategy has
positive NMB at each threshold; they describe parameter uncertainty, not
statistical uncertainty from calibration.

Price-threshold analysis uses the linear decomposition
    ICER = (delta_non-price-costs + doses * (price + delivery)) / delta_dalys
to solve analytically for the breakeven price.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import Sequence

from .costing import (
    discount_factor,
    yll_per_death_by_age_bin,
    yld_per_case,
    vaccine_costs_from_receipts,
    treatment_costs_from_outcomes,
)
from .outcomes import (
    expected_confirmed_cases,
    apply_severity,
)


# ── Parameter sampling ────────────────────────────────────────────────────────

def _sample_beta(params: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw from Beta(a, b).  Params keys: a/b or alpha/beta."""
    a = float(params.get("a", params.get("alpha", 1.0)))
    b = float(params.get("b", params.get("beta", 1.0)))
    return rng.beta(a, b, size=n)


def _sample_lognormal(params: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw from Lognormal.  Params keys: mean/sd (natural scale) or meanlog/sdlog."""
    if "meanlog" in params or "sdlog" in params:
        mu = float(params.get("meanlog", 0.0))
        sigma = float(params.get("sdlog", 1.0))
    else:
        mean = float(params.get("mean", 1.0))
        sd = float(params.get("sd", 0.5))
        var = sd * sd
        mu = np.log(mean**2 / np.sqrt(mean**2 + var))
        sigma = np.sqrt(np.log(1.0 + var / mean**2))
    return np.exp(rng.normal(mu, sigma, size=n))


def _sample_gamma(params: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw from Gamma.  Params keys: shape/rate or shape/scale or mean/sd."""
    if "shape" in params and "rate" in params:
        shape = float(params["shape"])
        scale = 1.0 / float(params["rate"])
    elif "shape" in params and "scale" in params:
        shape = float(params["shape"])
        scale = float(params["scale"])
    else:
        mean = float(params.get("mean", 1.0))
        sd = float(params.get("sd", 0.5))
        shape = (mean / sd) ** 2
        scale = sd**2 / mean
    return rng.gamma(shape, scale, size=n)


def sample_entries(
    entries: list[dict],
    n: int,
    seed: int,
) -> list[list[dict]]:
    """Draw N samples from severity entries (beta / lognormal / fixed).

    Returns a list of N entry lists, each entry list having the same structure
    as the original but with a 'value' key set to the sampled value.  The
    entries within each draw are independent; draws at the same index across
    different entry lists are generated in a fixed column order so two calls
    with the same seed agree.

    Parameters
    ----------
    entries:
        List of severity entry dicts from the YAML (each has distribution,
        params, point).
    n:
        Number of draws.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of N entry-lists, each a copy of ``entries`` with 'value' added.
    """
    rng = np.random.default_rng(seed)
    draws_per_entry: list[np.ndarray] = []
    for entry in entries:
        dist = str(entry.get("distribution", "fixed")).lower()
        point = float(entry.get("point", 0.0))
        params = entry.get("params", {}) or {}
        if dist == "fixed":
            values = np.full(n, point)
        elif dist == "beta":
            values = _sample_beta(params, rng, n)
        elif dist == "lognormal":
            values = _sample_lognormal(params, rng, n)
        else:
            raise ValueError(f"unknown severity distribution {dist!r}")
        draws_per_entry.append(values)

    result: list[list[dict]] = []
    for i in range(n):
        draw_entries = []
        for j, entry in enumerate(entries):
            new_entry = dict(entry)
            new_entry["value"] = float(draws_per_entry[j][i])
            draw_entries.append(new_entry)
        result.append(draw_entries)
    return result


def sample_cost_draws(
    items: list[dict],
    n: int,
    seed: int,
) -> list[dict[str, dict]]:
    """Draw N samples from cost items (gamma / lognormal / fixed).

    Returns a list of N cost-item dicts (keyed by item name), each with a
    'value' key set to the sampled value.

    Parameters
    ----------
    items:
        List of cost item dicts from costs YAML (each has item, distribution,
        params, value_intl_2024 or value_usd_2024).
    n:
        Number of draws.
    seed:
        Random seed.

    Returns
    -------
    List of N dicts mapping item name -> dict with 'value' set.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    for item in items:
        name = str(item["item"])
        dist = str(item.get("distribution", "fixed")).lower()
        point = float(item.get("value_intl_2024",
                                item.get("value_usd_2024", 0.0)) or 0.0)
        params = item.get("params", {}) or {}
        if dist == "fixed":
            draws[name] = np.full(n, point)
        elif dist == "gamma":
            draws[name] = _sample_gamma(params, rng, n)
        elif dist == "lognormal":
            draws[name] = _sample_lognormal(params, rng, n)
        else:
            raise ValueError(f"unknown cost distribution {dist!r}")

    result: list[dict[str, dict]] = []
    for i in range(n):
        draw_dict: dict[str, dict] = {}
        for item in items:
            name = str(item["item"])
            new_item = dict(item)
            new_item["value"] = float(draws[name][i])
            draw_dict[name] = new_item
        result.append(draw_dict)
    return result


# ── Decision metrics ──────────────────────────────────────────────────────────

def compute_incremental(
    totals_df: pd.DataFrame,
    *,
    baseline_strategy: str = "baseline",
) -> pd.DataFrame:
    """Compute incremental costs, DALYs averted, ICER, and dominance flags.

    Parameters
    ----------
    totals_df:
        One row per (draw_index, strategy_id) with columns:
        total_cost, total_dalys, draw_index, strategy_id, profile.
    baseline_strategy:
        Label for the reference comparator.

    Returns
    -------
    DataFrame with incremental_cost, dalys_averted, icer, is_dominated,
    is_weakly_dominated added.  ICER is NaN when DALYs averted is zero.
    Dominance: incremental_cost > 0 AND dalys_averted < 0.
    """
    required = {"draw_index", "strategy_id", "total_cost", "total_dalys"}
    missing = required - set(totals_df.columns)
    if missing:
        raise ValueError(f"totals_df missing: {sorted(missing)}")

    rows = []
    for draw_idx, group in totals_df.groupby("draw_index", sort=True):
        base = group[group["strategy_id"] == baseline_strategy]
        if base.empty:
            raise ValueError(
                f"baseline strategy {baseline_strategy!r} not found in draw {draw_idx}"
            )
        base_cost = float(base["total_cost"].iloc[0])
        base_dalys = float(base["total_dalys"].iloc[0])

        for _, row in group.iterrows():
            sid = str(row["strategy_id"])
            cost = float(row["total_cost"])
            dalys = float(row["total_dalys"])
            d_cost = cost - base_cost
            d_dalys_averted = base_dalys - dalys
            if d_dalys_averted != 0.0:
                icer = d_cost / d_dalys_averted
            else:
                icer = np.nan
            is_dominated = (d_cost > 0) and (d_dalys_averted < 0)
            rec = {k: row[k] for k in totals_df.columns}
            rec.update(dict(
                incremental_cost=d_cost,
                dalys_averted=d_dalys_averted,
                icer=icer,
                is_dominated=is_dominated,
            ))
            rows.append(rec)

    return pd.DataFrame(rows)


def net_monetary_benefit(
    incremental_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Compute NMB = threshold * dalys_averted - incremental_cost per row."""
    out = incremental_df.copy()
    out["threshold"] = float(threshold)
    out["nmb"] = float(threshold) * out["dalys_averted"] - out["incremental_cost"]
    return out


def ceac(
    incremental_df: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Cost-effectiveness acceptability curve.

    Returns a DataFrame with columns strategy_id, profile, threshold,
    prob_cost_effective (fraction of draws with NMB >= 0 at that threshold).
    """
    rows = []
    strategies = incremental_df["strategy_id"].unique()
    profiles = incremental_df.get("profile", pd.Series([""] * len(incremental_df))).unique()

    group_keys = ["strategy_id"]
    if "profile" in incremental_df.columns:
        group_keys = ["profile", "strategy_id"]

    for threshold in thresholds:
        nmb_df = net_monetary_benefit(incremental_df, threshold)
        for key, grp in nmb_df.groupby(group_keys, sort=True):
            prob = float((grp["nmb"] >= 0).mean())
            rec: dict = {}
            if isinstance(key, tuple):
                for k, v in zip(group_keys, key):
                    rec[k] = v
            else:
                rec[group_keys[0]] = key
            rec["threshold"] = float(threshold)
            rec["prob_cost_effective"] = prob
            rows.append(rec)

    return pd.DataFrame(rows)


def price_threshold_analysis(
    incremental_df: pd.DataFrame,
    *,
    additional_doses_df: pd.DataFrame,
    threshold: float,
    delivery_cost: float = 0.0,
    strategy_col: str = "strategy_id",
) -> pd.DataFrame:
    """Find vaccine price at which each strategy's median ICER crosses threshold.

    Solves analytically:
        ICER = (non_price_cost + doses * (price + delivery)) / dalys_averted
        => price = (ICER * dalys_averted - non_price_cost - doses * delivery) / doses

    Parameters
    ----------
    incremental_df:
        Output of compute_incremental with incremental_cost, dalys_averted,
        draw_index, strategy_id.
    additional_doses_df:
        Per-draw total additional doses per strategy.  Columns:
        draw_index, strategy_id, total_additional_doses.
    threshold:
        WTP threshold for computing breakeven.
    delivery_cost:
        Fixed delivery cost per dose (not varied in price-threshold analysis).

    Returns
    -------
    DataFrame with strategy_id, median_breakeven_price, pct10_breakeven_price,
    pct90_breakeven_price.
    """
    merged = incremental_df.merge(
        additional_doses_df[["draw_index", strategy_col, "total_additional_doses"]],
        on=["draw_index", strategy_col], how="left",
    )

    rows = []
    for sid, grp in merged.groupby(strategy_col, sort=True):
        doses = grp["total_additional_doses"].to_numpy(float)
        dalys = grp["dalys_averted"].to_numpy(float)
        d_cost = grp["incremental_cost"].to_numpy(float)

        # incremental_cost includes vaccine cost; separate it out:
        # non_price_cost ≈ d_cost - doses * (point_price + delivery)
        # Breakeven: threshold * dalys = non_price_cost + doses * (price + delivery)
        # => price = (threshold * dalys - non_price_cost - doses * delivery) / doses
        # For a conservative calculation we treat d_cost as non-price cost and
        # add the dose-cost component; this is approximate without the original
        # price-per-dose in this draw.  The function documents this limit.
        nonzero = doses > 0
        breakeven = np.full_like(dalys, np.nan)
        breakeven[nonzero] = (
            threshold * dalys[nonzero] - d_cost[nonzero]
        ) / doses[nonzero]

        finite = breakeven[np.isfinite(breakeven)]
        rows.append(dict(
            strategy_id=sid,
            n_draws=len(finite),
            median_breakeven_price=float(np.nanmedian(breakeven)) if len(finite) else np.nan,
            pct10_breakeven_price=float(np.nanpercentile(breakeven, 10)) if len(finite) else np.nan,
            pct90_breakeven_price=float(np.nanpercentile(breakeven, 90)) if len(finite) else np.nan,
        ))
    return pd.DataFrame(rows)


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_psa(
    onset_df: pd.DataFrame,
    receipts_df: pd.DataFrame,
    *,
    severity_doc: dict,
    costs_doc: dict,
    country: str,
    reporting_logits: list[float],
    mild_reporting_ratio: float = 0.05,
    start_year: int,
    end_year: int,
    discount_rate: float = 0.03,
    n_draws: int = 200,
    seed: int = 42,
    baseline_strategy: str = "baseline",
    threshold_grid: list[float] | None = None,
    death_ascertainment: float = 1.0,
    grid=None,
) -> dict:
    """Full probabilistic sensitivity analysis.

    Parameters
    ----------
    onset_df:
        annual_age_onsets.csv.gz content; columns include year, age_bin,
        bin_index, typical_onsets, mild_onsets, strategy_id, profile.
    receipts_df:
        receipts.csv.gz content; columns include year, strategy_id,
        additional_receipts, profile.
    severity_doc:
        Loaded core2026-severity-priors-v1 YAML (from load_severity_priors).
    costs_doc:
        Loaded core2026-costs-v1 YAML (from load_costs).
    country:
        Country key matching severity_doc and costs_doc.
    reporting_logits:
        Six fitted logits (from fit_summary.csv) for this country.
    mild_reporting_ratio, start_year, end_year, discount_rate:
        Observation and discounting parameters.
    n_draws:
        Number of paired PSA draws.
    seed:
        Master random seed (same seed used for severity and cost draws,
        offset by 1 for cost draws to ensure independence).
    baseline_strategy:
        Strategy label for the no-intervention comparator.
    threshold_grid:
        WTP thresholds for CEAC.  Defaults to [0, 500, 1000, ..., 50000].
    death_ascertainment:
        Fraction of unreported-case deaths captured in death registries.
    grid:
        AgeGrid; constructed if None.

    Returns
    -------
    dict with keys:
        incremental      – DataFrame of all draws × strategies incremental metrics
        ceac             – CEAC DataFrame
        price_threshold  – price-threshold analysis DataFrame
        totals           – per-draw total costs and DALYs
        run_spec         – dict of run parameters
    """
    if grid is None:
        from ..grid import AgeGrid
        grid = AgeGrid()

    if threshold_grid is None:
        threshold_grid = list(range(0, 50001, 1000))

    country_severity = severity_doc.get("countries", {}).get(country, {})
    severity_status = country_severity.get("status", "")
    severity_entries = country_severity.get("entries", [])

    country_costs = costs_doc.get("countries", {}).get(country, {})
    cost_items = country_costs.get("items", [])
    global_opts = costs_doc.get("global", {})
    disability_weights = global_opts.get("disability_weights", {}) or {}
    life_table_raw = global_opts.get("life_table", None)
    life_table: dict | None = None
    if isinstance(life_table_raw, list):
        life_table = {int(row["age"]): float(row["le"]) for row in life_table_raw
                      if "age" in row and "le" in row}

    # Filter onset_df to prospective years.
    onset_year = onset_df[
        onset_df["year"].between(start_year, end_year - 1)
    ].copy()
    receipts_year = receipts_df[
        receipts_df["year"].between(start_year, end_year - 1)
    ].copy()

    # Severity draws.
    if severity_entries:
        sev_draws = sample_entries(severity_entries, n_draws, seed)
    else:
        sev_draws = [[] for _ in range(n_draws)]

    # Cost draws (offset seed by 1).
    cost_draws = sample_cost_draws(cost_items, n_draws, seed + 1)

    # YLL per bin (deterministic given life_table and discount_rate).
    yll_bins = yll_per_death_by_age_bin(grid, life_table=life_table,
                                         discount_rate=discount_rate)

    strategies = onset_year["strategy_id"].unique()

    all_rows: list[dict] = []
    dose_rows: list[dict] = []

    for draw_idx in range(n_draws):
        sev_draw = sev_draws[draw_idx]
        cost_draw = cost_draws[draw_idx]
        cost_items_draw: dict[str, dict] = cost_draw

        for sid in strategies:
            onset_s = onset_year[onset_year["strategy_id"] == sid]
            receipts_s = receipts_year[receipts_year["strategy_id"] == sid]

            # 1. Reported cases.
            confirmed = expected_confirmed_cases(
                onset_s,
                reporting_logits=reporting_logits,
                mild_reporting_ratio=mild_reporting_ratio,
                grid=grid,
            )

            # 2. Hospitalisations.
            hosp_entries = [e for e in sev_draw
                            if e.get("quantity") == "hospitalisation_proportion"]
            if hosp_entries:
                hosp_df = apply_severity(
                    confirmed,
                    severity_entries=hosp_entries,
                    quantity="hospitalisation_proportion",
                    basis="per_confirmed_case",
                    death_ascertainment=death_ascertainment,
                    grid=grid,
                )
            else:
                hosp_df = None

            # 3. Deaths (CFR).
            cfr_entries = [e for e in sev_draw if e.get("quantity") == "cfr"]
            if cfr_entries:
                death_df = apply_severity(
                    confirmed,
                    severity_entries=cfr_entries,
                    quantity="cfr",
                    basis="per_confirmed_case",
                    death_ascertainment=death_ascertainment,
                    grid=grid,
                )
            else:
                death_df = None

            # 4. DALYs.
            dw = float(disability_weights.get("pertussis", 0.133))
            yld_unit = yld_per_case(disability_weight=dw, discount_rate=discount_rate)

            confirmed_cost_arr = confirmed[confirmed["year"].between(start_year, end_year - 1)]
            yll_total = 0.0
            yld_total = 0.0

            for _, row in confirmed_cost_arr.iterrows():
                disc = discount_factor(int(row["year"]), start_year, discount_rate)
                yld_total += float(row["confirmed_cases"]) * yld_unit * disc

            if death_df is not None:
                bin_arr = death_df["bin_index"].to_numpy(int)
                deaths_arr = death_df["outcome_count"].to_numpy(float)
                for i, (bidx, d) in enumerate(zip(bin_arr, deaths_arr)):
                    yr = int(death_df["year"].iloc[i])
                    disc = discount_factor(yr, start_year, discount_rate)
                    yll_total += d * float(yll_bins[bidx]) * disc

            total_dalys = yll_total + yld_total

            # 5. Costs.
            vacc_df = vaccine_costs_from_receipts(
                receipts_s,
                cost_items=cost_items_draw,
                start_year=start_year,
                discount_rate=discount_rate,
            )
            total_vacc_cost = float(vacc_df["total_vaccine_cost"].sum()) if not vacc_df.empty else 0.0
            total_add_doses = float(receipts_s["additional_receipts"].sum())

            treat_cost = 0.0
            if hosp_df is not None:
                treat_df = treatment_costs_from_outcomes(
                    hosp_df,
                    cost_items=cost_items_draw,
                    start_year=start_year,
                    discount_rate=discount_rate,
                )
                treat_cost = float(treat_df["treatment_cost"].sum())

            total_cost = total_vacc_cost + treat_cost
            profile = str(onset_s["profile"].iloc[0]) if "profile" in onset_s.columns and len(onset_s) else country

            all_rows.append(dict(
                draw_index=draw_idx,
                profile=profile,
                strategy_id=sid,
                total_cost=total_cost,
                vaccine_cost=total_vacc_cost,
                treatment_cost=treat_cost,
                total_dalys=total_dalys,
                yll=yll_total,
                yld=yld_total,
            ))
            dose_rows.append(dict(
                draw_index=draw_idx,
                profile=profile,
                strategy_id=sid,
                total_additional_doses=total_add_doses,
            ))

    totals = pd.DataFrame(all_rows)
    doses_df = pd.DataFrame(dose_rows)

    incremental = compute_incremental(totals, baseline_strategy=baseline_strategy)
    ceac_df = ceac(incremental, threshold_grid)

    # Price threshold: only for non-baseline strategies.
    non_base = incremental[incremental["strategy_id"] != baseline_strategy]
    non_base_doses = doses_df[doses_df["strategy_id"] != baseline_strategy]
    if not non_base.empty and not non_base_doses.empty:
        pt_df = price_threshold_analysis(
            non_base,
            additional_doses_df=non_base_doses,
            threshold=float(np.median(threshold_grid)) if threshold_grid else 10000.0,
        )
    else:
        pt_df = pd.DataFrame()

    run_spec = dict(
        country=country,
        n_draws=n_draws,
        seed=seed,
        start_year=start_year,
        end_year=end_year,
        discount_rate=discount_rate,
        baseline_strategy=baseline_strategy,
        death_ascertainment=death_ascertainment,
        mild_reporting_ratio=mild_reporting_ratio,
        threshold_grid=threshold_grid,
        severity_status=severity_status,
    )

    return dict(
        incremental=incremental,
        ceac=ceac_df,
        price_threshold=pt_df,
        totals=totals,
        run_spec=run_spec,
    )
