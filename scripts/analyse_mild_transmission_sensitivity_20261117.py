"""Prospective sensitivity to the relative infectiousness of mild infection.

The retained fitted histories are held fixed while ``rho_mild`` is changed to
0.25, 0.50, 0.75 or 1.00.  For each setting and value we replay the current
adolescent programme, acquisition-only, equal-strength and onward-only nasal
arms.  This is a structural diagnostic of the mechanism ranking; it is not a
new fit and it does not represent an independent asymptomatic compartment.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import analyse_review_revision_20261110 as rev
from scripts import analyse_review_round4_20261112 as r4
from scripts import build_bpze1_policy_design_20261107 as design

ROOT, BASE = rev.ROOT, rev.BASE
OUT = BASE / "02_sensitivity/review_round6"
RHOS = (0.25, 0.50, 0.75, 1.00)
ALLOCATIONS = ("acquisition", "equal", "onward")


def run_arm(ctx, params, efficacy, onward, coverage):
    info, inputs, _params, hist, scenarios, _catalog, dates, days = ctx
    events = np.array([
        (*rev._target_event(inputs, float(info["adolescent_age"]), year), coverage)
        for year in range(dates[0].year, dates[-1].year)
    ])
    sim = rev.BPZE1Simulator(scenarios["current"])
    result = sim.run(
        params, hist.y, hist.acc, days[0], days[-1], record_days=days,
        efficacy=float(efficacy), infectiousness_multiplier=float(onward),
        mean_years=3.0, start_day=days[0], nasal_events=events,
    )
    raw = np.diff(result.acc_records, axis=0)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("invalid onset trajectory")
    return raw


def one_country(country: str) -> pd.DataFrame:
    ctx = rev.setup(country)
    info, _inputs, base_params, _hist, _scenarios, catalog, dates, _days = ctx
    current = catalog[catalog.strategy_id == "current"].iloc[0]
    coverage = max(0.7, float(current.adolescent_coverage))
    ratio = json.loads(rev.bp.CONFIG.read_text())["clinical_input"]["point"]
    profiles = {
        "acquisition": design.profile(ratio, 0.5, 1.0, 3.0),
        "equal": design.profile(ratio, 0.5, 0.5, 3.0),
        "onward": design.profile(ratio, 0.5, 0.0, 3.0),
    }
    rows = []
    for rho in RHOS:
        params = replace(base_params, rho_mild=float(rho))
        raw = [run_arm(ctx, params, 0.0, 1.0, coverage)]
        for label in ALLOCATIONS:
            p = profiles[label]
            raw.append(run_arm(ctx, params, p["efficacy"], p["infectiousness_multiplier"], coverage))
        raw = np.stack(raw)
        data = rev.clinical(country, raw, dates[:-1], dates[1:], info)
        for j, label in enumerate(ALLOCATIONS, start=1):
            for row in r4.endpoints(data, j, horizons=(20, 30)):
                rows.append({
                    "country": country,
                    "rho_mild": float(rho),
                    "allocation": label,
                    "nasal_coverage": coverage,
                    **row,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"mild_rho_{country}.csv", index=False)
    return frame


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, rho, horizon), g in detail.groupby(["country", "rho_mild", "horizon"], sort=True):
        by = g.set_index("allocation")
        d = by.loc["onward", "dalys"] - by.loc["acquisition", "dalys"]
        c = by.loc["onward", "infant_cases"] - by.loc["acquisition", "infant_cases"]
        rows.append({
            "country": country,
            "rho_mild": rho,
            "horizon": int(horizon),
            "onward_minus_acquisition_dalys": float(d),
            "onward_minus_acquisition_dalys_q025": float(by.loc["onward", "dalys_q025"] - by.loc["acquisition", "dalys_q975"]),
            "onward_minus_acquisition_dalys_q975": float(by.loc["onward", "dalys_q975"] - by.loc["acquisition", "dalys_q025"]),
            "onward_minus_acquisition_infant_cases": float(c),
            "onward_dalys_ranked_higher": bool(d > 0),
            "onward_infant_cases_ranked_higher": bool(c > 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=list(rev.COUNTRIES))
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.concat([one_country(c) for c in args.countries], ignore_index=True)
    summary = summarize(detail)
    summary.to_csv(OUT / "mild_rho_ranking_summary.csv", index=False)
    counts = (summary.groupby(["rho_mild", "horizon"], as_index=False)
              .agg(settings_onward_higher_dalys=("onward_dalys_ranked_higher", "sum"),
                   settings_onward_higher_infant_cases=("onward_infant_cases_ranked_higher", "sum")))
    counts.to_csv(OUT / "mild_rho_ranking_counts.csv", index=False)
    metadata = {
        "rho_mild_values": list(RHOS),
        "countries": list(args.countries),
        "horizons": [20, 30],
        "arms": list(ALLOCATIONS),
        "history": "adopted fitted history held fixed",
        "refit": False,
        "independent_asymptomatic_compartment": False,
        "interpretation": "prospective structural sensitivity; no probability weights assigned",
        "sources": [
            "scripts/analyse_mild_transmission_sensitivity_20261117.py",
            "scripts/analyse_review_revision_20261110.py",
            "scripts/build_bpze1_policy_design_20261107.py",
            "src_python/pertussis/extensions/bpze1.py",
            "src_python/pertussis/model/parameters.py",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {OUT} ({len(detail)} arm rows, {len(summary)} ranking rows)")


if __name__ == "__main__":
    main()
