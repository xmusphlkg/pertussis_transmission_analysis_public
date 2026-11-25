"""Vaccine and treatment cost calculation, discounting, and DALY computation.

Interpretation limits
---------------------
Costs are in 2024 international dollars (value_intl_2024) or 2024 USD
(value_usd_2024) as declared in the YAML.  Both units are accepted; callers
must track which is active.  Results are not converted between units.

DALYs use onset-year discounting at 3 % per annum from start_year.  YLL uses
a period life table (global.life_table in costs YAML); if absent a simple
80-year life expectancy is used.  YLD uses GBD disability weights
(global.disability_weights in costs YAML) and a 21-day episode duration.

All outputs are per-draw totals for one paired strategy set.  Callers must
aggregate across draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Discounting ───────────────────────────────────────────────────────────────

def discount_factor(year: int, start_year: int, rate: float = 0.03) -> float:
    """Annual discount factor: (1 + rate)^{-(year - start_year)}.

    Returns 1.0 when year <= start_year.
    """
    return (1.0 + float(rate)) ** (-max(0, int(year) - int(start_year)))


# ── YLL ───────────────────────────────────────────────────────────────────────

def yll_per_death(
    age_mid_years: float,
    *,
    life_table: dict | None = None,
    discount_rate: float = 0.03,
) -> float:
    """Discounted YLL per death at ``age_mid_years``.

    Uses annuity discounting: integral_0^le e^{-r*t} dt = (1 - e^{-r*le})/r
    for r > 0.  Falls back to le for r = 0.  Life expectancy is looked up by
    integer age from ``life_table`` (dict mapping int -> float) or computed as
    max(0, 80 - age) when the table is absent.
    """
    age = float(age_mid_years)
    if life_table is not None:
        le = float(life_table.get(int(np.floor(age)), max(0.0, 80.0 - age)))
    else:
        le = max(0.0, 80.0 - age)
    r = float(discount_rate)
    if r > 0 and le > 0:
        return (1.0 - np.exp(-r * le)) / r
    return le


def yll_per_death_by_age_bin(
    grid,
    *,
    life_table: dict | None = None,
    discount_rate: float = 0.03,
) -> np.ndarray:
    """(n_bins,) discounted YLL per death for each model age bin.

    Uses the mid-point of each bin's lower/upper year bounds (capped at 100
    for the open bin).
    """
    out = np.zeros(grid.n_bins, dtype=float)
    for b in range(grid.n_bins):
        lo = float(grid.lower_years[b])
        hi = float(grid.upper_years[b])
        if np.isinf(hi):
            hi = 100.0
        out[b] = yll_per_death((lo + hi) / 2.0, life_table=life_table,
                                discount_rate=discount_rate)
    return out


# ── YLD ───────────────────────────────────────────────────────────────────────

def yld_per_case(
    *,
    disability_weight: float = 0.133,
    episode_days: float = 21.0,
    discount_rate: float = 0.03,
) -> float:
    """Discounted YLD per confirmed case (acute symptomatic episode).

    GBD 2019 default: disability weight 0.133, episode 21 days.  The
    discount correction over a 21-day episode is negligible but is included
    for consistency with the YLL formula.
    """
    ep_years = float(episode_days) / 365.25
    r = float(discount_rate)
    dw = float(disability_weight)
    if r > 0 and ep_years > 0:
        return dw * (1.0 - np.exp(-r * ep_years)) / r
    return dw * ep_years


# ── Combined DALYs ────────────────────────────────────────────────────────────

def compute_dalys(
    confirmed_df: pd.DataFrame,
    death_df: pd.DataFrame | None,
    *,
    life_table: dict | None = None,
    disability_weights: dict | None = None,
    start_year: int,
    discount_rate: float = 0.03,
    episode_days: float = 21.0,
    grid=None,
) -> pd.DataFrame:
    """Compute total DALYs (YLL + YLD) per row, discounted from start_year.

    Parameters
    ----------
    confirmed_df:
        Has confirmed_cases, year, age_bin, bin_index, profile, strategy_id.
    death_df:
        Has outcome_count (deaths), year, age_bin, bin_index, profile,
        strategy_id.  If None, YLL is zero.
    life_table:
        dict mapping integer age to remaining life expectancy (years).
    disability_weights:
        dict; uses entry ``pertussis`` for YLD.  Falls back to GBD 0.133.
    start_year, discount_rate:
        Discounting parameters.
    episode_days:
        Acute episode duration for YLD.
    grid:
        AgeGrid; constructed from defaults if None.

    Returns
    -------
    DataFrame grouped by (profile, strategy_id, year, age_bin, bin_index)
    with columns yll, yld, dalys, confirmed_cases, deaths.
    """
    if grid is None:
        from ..grid import AgeGrid
        grid = AgeGrid()

    dw = float((disability_weights or {}).get("pertussis", 0.133))
    _yld_per_case = yld_per_case(disability_weight=dw,
                                  episode_days=episode_days,
                                  discount_rate=discount_rate)

    rows: list[dict] = []

    for _, row in confirmed_df.iterrows():
        year = int(row["year"])
        disc = discount_factor(year, start_year, discount_rate)
        cases = float(row.get("confirmed_cases", 0.0))
        bidx = int(row.get("bin_index", 0))
        yld_val = cases * _yld_per_case * disc
        rows.append(dict(
            profile=str(row.get("profile", "")),
            strategy_id=str(row.get("strategy_id", "")),
            year=year,
            age_bin=str(row.get("age_bin", "")),
            bin_index=bidx,
            confirmed_cases=cases,
            deaths=0.0,
            yll=0.0,
            yld=yld_val,
            dalys=yld_val,
        ))

    if death_df is not None:
        yll_by_bin = yll_per_death_by_age_bin(grid, life_table=life_table,
                                               discount_rate=discount_rate)
        for _, row in death_df.iterrows():
            year = int(row["year"])
            disc = discount_factor(year, start_year, discount_rate)
            deaths = float(row.get("outcome_count", 0.0))
            bidx = int(row.get("bin_index", 0))
            yll_val = deaths * float(yll_by_bin[bidx]) * disc
            rows.append(dict(
                profile=str(row.get("profile", "")),
                strategy_id=str(row.get("strategy_id", "")),
                year=year,
                age_bin=str(row.get("age_bin", "")),
                bin_index=bidx,
                confirmed_cases=0.0,
                deaths=deaths,
                yll=yll_val,
                yld=0.0,
                dalys=yll_val,
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["profile", "strategy_id", "year", "age_bin", "bin_index"],
                        sort=False)
              .sum(numeric_only=True)
              .reset_index())


# ── Vaccine costs ─────────────────────────────────────────────────────────────

def vaccine_costs_from_receipts(
    receipts_df: pd.DataFrame,
    *,
    cost_items: dict[str, dict],
    start_year: int,
    discount_rate: float = 0.03,
) -> pd.DataFrame:
    """Vaccine + delivery costs from receipts additional_receipts column.

    Parameters
    ----------
    receipts_df:
        Columns: year, strategy_id, additional_receipts, profile (optional).
        ``additional_receipts`` is the incremental dose count above baseline
        for that year.
    cost_items:
        Dict keyed by item name (vaccine_price_maternal, delivery_maternal,
        vaccine_price_adolescent, delivery_adolescent, …) from costs YAML.
        Each entry may have a sampled 'value' (PSA draw) or fall back to
        value_intl_2024 / value_usd_2024.
    start_year, discount_rate:
        Discounting parameters.

    Returns
    -------
    DataFrame with columns profile, strategy_id, year, additional_doses,
    dose_cost, delivery_cost, total_vaccine_cost (all discounted).
    """
    required = {"year", "strategy_id", "additional_receipts"}
    missing = required - set(receipts_df.columns)
    if missing:
        raise ValueError(f"receipts_df missing: {sorted(missing)}")

    def _val(name: str) -> float:
        item = cost_items.get(name, {})
        if "value" in item:
            return float(item["value"])
        v = item.get("value_intl_2024", item.get("value_usd_2024", 0.0))
        return float(v) if v is not None else 0.0

    rows = []
    for _, row in receipts_df.iterrows():
        year = int(row["year"])
        sid = str(row["strategy_id"])
        profile = str(row.get("profile", ""))
        doses = float(row["additional_receipts"])
        disc = discount_factor(year, start_year, discount_rate)

        if "maternal" in sid:
            price = _val("vaccine_price_maternal")
            delivery = _val("delivery_maternal")
        elif "adolescent" in sid:
            price = _val("vaccine_price_adolescent")
            delivery = _val("delivery_adolescent")
        else:
            # Routine or timing: use maternal price by convention.
            price = _val("vaccine_price_maternal")
            delivery = _val("delivery_maternal")

        rows.append(dict(
            profile=profile,
            strategy_id=sid,
            year=year,
            additional_doses=doses,
            dose_cost=doses * price * disc,
            delivery_cost=doses * delivery * disc,
            total_vaccine_cost=doses * (price + delivery) * disc,
        ))
    return pd.DataFrame(rows)


# ── Treatment costs ───────────────────────────────────────────────────────────

def treatment_costs_from_outcomes(
    outcome_df: pd.DataFrame,
    confirmed_df: pd.DataFrame | None = None,
    *,
    cost_items: dict[str, dict],
    start_year: int,
    discount_rate: float = 0.03,
) -> pd.DataFrame:
    """Treatment costs from hospitalisation, ICU, LOS, and outpatient rows.

    Parameters
    ----------
    outcome_df:
        Has quantity, outcome_count, year, age_bin, profile, strategy_id.
        quantity is one of hospitalisation_proportion, icu_given_hospital,
        los_days, cfr (not directly costed).
    confirmed_df:
        If provided, outpatient costs are added for non-hospitalised confirmed
        cases.  Requires confirmed_cases column.
    cost_items, start_year, discount_rate:
        As for vaccine_costs_from_receipts.

    Returns
    -------
    DataFrame with columns profile, strategy_id, year, age_bin, quantity,
    outcome_count, treatment_cost.
    """
    def _val(name: str) -> float:
        item = cost_items.get(name, {})
        if "value" in item:
            return float(item["value"])
        v = item.get("value_intl_2024", item.get("value_usd_2024", 0.0))
        return float(v) if v is not None else 0.0

    cost_per_inpatient = _val("inpatient_episode")
    cost_per_icu_day = _val("icu_day")
    cost_per_inpatient_day = _val("inpatient_day")
    cost_per_op = _val("outpatient_episode")

    rows = []
    for _, row in outcome_df.iterrows():
        qty = str(row.get("quantity", ""))
        count = float(row.get("outcome_count", 0.0))
        year = int(row["year"])
        disc = discount_factor(year, start_year, discount_rate)

        if qty == "hospitalisation_proportion":
            cost = count * cost_per_inpatient * disc
        elif qty == "icu_given_hospital":
            cost = count * cost_per_icu_day * disc
        elif qty == "los_days":
            cost = count * cost_per_inpatient_day * disc
        else:
            cost = 0.0

        rows.append(dict(
            profile=str(row.get("profile", "")),
            strategy_id=str(row.get("strategy_id", "")),
            year=year,
            age_bin=str(row.get("age_bin", "")),
            quantity=qty,
            outcome_count=count,
            treatment_cost=cost,
        ))

    if confirmed_df is not None:
        for _, row in confirmed_df.iterrows():
            year = int(row["year"])
            disc = discount_factor(year, start_year, discount_rate)
            confirmed = float(row.get("confirmed_cases", 0.0))
            hosp = float(row.get("hosp_count", 0.0))
            outpatients = max(0.0, confirmed - hosp)
            rows.append(dict(
                profile=str(row.get("profile", "")),
                strategy_id=str(row.get("strategy_id", "")),
                year=year,
                age_bin=str(row.get("age_bin", "")),
                quantity="outpatient",
                outcome_count=outpatients,
                treatment_cost=outpatients * cost_per_op * disc,
            ))

    return pd.DataFrame(rows)
