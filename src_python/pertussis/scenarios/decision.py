"""Paired decision metrics on both remaining-burden and incremental-benefit scales.

The no-change baseline is an available option. Benefit loss is undefined when
no intervention improves it. Thresholds describe sensitivity, not equivalence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["profile", "structure_id", "draw_id", "budget_id"]
ENDPOINTS = ("infant_0_2m_cases", "age1_17_cases")
ADDITIONAL_DOSE_STRATEGIES = (
    "maternal_scaleup", "routine_scaleup", "adolescent_one_time_booster",
)


def benefit_decisions(results: pd.DataFrame, *, include_timing: bool = False) -> pd.DataFrame:
    """Return metrics per paired draw, with explicit baseline and tie handling."""
    required = KEYS + ["strategy_id", *ENDPOINTS]
    missing = set(required) - set(results)
    if missing:
        raise ValueError(f"Missing decision columns: {sorted(missing)}")
    rows = []
    expected = set(ADDITIONAL_DOSE_STRATEGIES) | {"baseline"}
    if include_timing:
        expected.add("routine_timing_shift_minus_1_month")
    for key, group in results.groupby(KEYS, sort=True, dropna=False):
        group = group[group.strategy_id.isin(expected)].copy()
        if group.strategy_id.duplicated().any() or set(group.strategy_id) != expected:
            raise ValueError(f"Incomplete or duplicated paired decision set: {key}")
        for endpoint in ENDPOINTS:
            values = group[endpoint].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Nonfinite or negative counts: {key}, {endpoint}")
            baseline = float(group.loc[group.strategy_id.eq("baseline"), endpoint].iloc[0])
            best = float(values.min())
            best_benefit = baseline - best
            # Count-scale tolerance is numerical, not a scientific margin.
            tol = 1e-10 * max(1.0, baseline)
            winners = np.isclose(values, best, rtol=0.0, atol=tol)
            for i, (_, row) in enumerate(group.iterrows()):
                count = float(row[endpoint])
                gap = max(0.0, count - best)
                loss = gap / best_benefit if best_benefit > tol else np.nan
                record = dict(zip(KEYS, key))
                record.update(
                    decision_set="all_four" if include_timing else "additional_doses",
                    endpoint=endpoint, strategy_id=row.strategy_id,
                    baseline_cases=baseline, cases=count,
                    cases_averted=baseline-count, best_cases_averted=best_benefit,
                    remaining_burden_regret=gap / best if best > 0 else np.nan,
                    benefit_loss_fraction=loss,
                    positive_best_benefit=best_benefit > tol,
                    improves_baseline=count < baseline-tol,
                    rank=1 + int(np.sum(values < count-tol)),
                    rank1_credit=float(winners[i]) / int(winners.sum()),
                    calibration_weight=row.get("calibration_weight", np.nan),
                )
                for threshold in (5, 10, 20):
                    record[f"within_{threshold}pct_best_benefit"] = (
                        float(loss <= threshold/100) if np.isfinite(loss) else np.nan
                    )
                rows.append(record)
    return pd.DataFrame(rows)


def summarize_benefit_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarise draws; missing benefit comparisons are excluded and counted."""
    keys = ["decision_set", "profile", "budget_id", "endpoint", "strategy_id"]
    rows = []
    for key, group in decisions.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record.update(
            replicates=len(group),
            positive_benefit_replicates=int(group.positive_best_benefit.sum()),
            rank1_frequency=float(group.rank1_credit.mean()),
            median_benefit_loss_fraction=float(group.benefit_loss_fraction.median())
            if group.benefit_loss_fraction.notna().any() else np.nan,
            improves_baseline_frequency=float(group.improves_baseline.mean()),
        )
        for threshold in (5, 10, 20):
            column = f"within_{threshold}pct_best_benefit"
            valid = group[column].notna()
            record[f"{column}_frequency"] = float(group.loc[valid, column].mean()) if valid.any() else np.nan
        weights = group.calibration_weight.to_numpy(float)
        if np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0:
            record["weighted_rank1_frequency"] = float(np.average(group.rank1_credit, weights=weights))
            valid = group.benefit_loss_fraction.notna().to_numpy()
            record["positive_benefit_weight_mass"] = float(weights[valid].sum()/weights.sum())
            for threshold in (5, 10, 20):
                column = f"within_{threshold}pct_best_benefit"
                record[f"weighted_{column}_frequency"] = (
                    float(np.average(group.loc[valid, column], weights=weights[valid]))
                    if weights[valid].sum() > 0 else np.nan
                )
        rows.append(record)
    return pd.DataFrame(rows)
