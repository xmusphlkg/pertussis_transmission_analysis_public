"""Targeted sensitivity analysis for maternal-protection assumptions.

The analysis is prospective and keeps the adopted fitted history fixed.  It
separates three questions that cannot be identified from the English infant
endpoint alone: passive-protection duration, the share of births entering
protection in time, and the allocation of protection between acquisition and
clinical course.  Duration scenarios are recalibrated to the same 60-day
86.2% endpoint when feasible; timely-entry scenarios retain the reference
biological parameters and therefore change the effective programme VE.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scripts import analyse_review_revision_20261110 as rev
from scripts import build_bpze1_policy_design_20261107 as design
from src_python.pertussis.extensions.bpze1 import BPZE1Simulator
from src_python.pertussis.model.clinical_maternal import (
    clinical_window_ve,
    fit_clinical_target,
)
from src_python.pertussis.scenarios.programme_scenarios import _target_event

ROOT = rev.ROOT
OUT = ROOT / "outputs/core2026/02_sensitivity/mucosal_vaccines/maternal_assumption_sensitivity_20261117"
COUNTRY = "China"
HORIZONS = (20, 30)
WINDOWS = (60.0, 90.0, 120.0, 180.0)


def parameter_scenarios(base):
    """Return labelled prospective parameter/entry-share scenarios."""
    scenarios = []
    # Keep the clinical target fixed while changing the duration.  The
    # calibration is conditional on sigma=1, matching the adopted pathway.
    for days in (120.0, 180.0, 240.0):
        fitted = fit_clinical_target(
            replace(base, maternal_days=days, maternal_waning_stages=4),
            target=0.862, sigma=1.0, window=60.0,
            effective_share=0.9, infant_coverage=0.95,
        )
        if fitted is not None:
            scenarios.append((f"duration_{int(days)}d_recalibrated", fitted, 0.9,
                              "duration_recalibrated"))
    # Delivery/timing scenarios retain the reference biological effect.  The
    # resulting 60-day VE is intentionally allowed to move with entry share.
    for share in (0.75, 0.9, 1.0):
        scenarios.append((f"timely_entry_{share:g}", base, share,
                          "timely_entry_fixed_biology"))
    # Keep the same 60-day endpoint while shifting protection towards reduced
    # acquisition.  This is a structural pathway comparison, not an estimate.
    reduced_acq = fit_clinical_target(
        replace(base, maternal_days=120.0, maternal_waning_stages=4),
        target=0.862, sigma=0.3, window=60.0,
        effective_share=0.9, infant_coverage=0.95,
    )
    if reduced_acq is not None:
        scenarios.append(("reduced_acquisition_recalibrated", reduced_acq, 0.9,
                          "acquisition_plus_course_recalibrated"))
    return scenarios


def run_arm(ctx, scenario_id, params, entry_share, nasal):
    info, inputs, _, history, scenarios, _, dates, days = ctx
    ratio = json.loads(rev.bp.CONFIG.read_text())["clinical_input"]["point"]
    profile = design.profile(ratio, 0.5, 0.5, 3.0)
    events = np.array([
        (*_target_event(inputs, float(info["adolescent_age"]), year), 0.7)
        for year in range(2027, 2057)
    ])
    simulator = BPZE1Simulator(scenarios[scenario_id])
    kwargs = dict(
        efficacy=profile["efficacy"] if nasal else 0.0,
        infectiousness_multiplier=profile["infectiousness_multiplier"] if nasal else 1.0,
        mean_years=3.0,
        start_day=days[0],
        nasal_events=events if nasal else None,
    )
    # Scenarios were built at the adopted share.  Replace the birth-entry
    # vector prospectively so the timing/transfer assumption is isolated.
    scenario_inputs = scenarios[scenario_id]
    first = scenario_inputs.calendar.year_start_day(2027)
    last = scenario_inputs.calendar.year_start_day(2057)
    mat = scenario_inputs.maternal_effective.copy()
    raw_coverage = mat[first:last] / 0.9
    mat[first:last] = raw_coverage * entry_share
    simulator = BPZE1Simulator(replace(scenario_inputs, maternal_effective=mat))
    result = simulator.run(params, history.y, history.acc, days[0], days[-1],
                           record_days=days, **kwargs)
    return np.diff(result.acc_records, axis=0)


def clinical_window_rows(base, scenarios):
    rows = []
    for label, params, share, role in scenarios:
        ve = clinical_window_ve(params, windows=WINDOWS, effective_share=share,
                                infant_coverage=0.95)
        for window in WINDOWS:
            rows.append(dict(
                scenario=label, scenario_role=role,
                maternal_days=params.maternal_days,
                timely_entry_share=share,
                sigma_maternal=params.sigma_maternal,
                pi_maternal=params.pi_maternal,
                window_days=window,
                modelled_window_ve=ve[window],
            ))
    return pd.DataFrame(rows)


def policy_rows(ctx, scenarios):
    rows = []
    arm_ids = [
        ("M0_no_nasal", "current", False),
        # Standalone nasal vaccination uses the same baseline-Tdap background
        # as the no-nasal comparator.  The 70% target is an independent nasal
        # event, not a simultaneous adolescent Tdap expansion.
        ("M0_nasal", "current", True),
        ("M90_no_nasal", "current_M0.9_Akeep", False),
        ("M90_nasal", "current_M0.9_Akeep", True),
    ]
    info = ctx[0]
    for label, params, share, role in scenarios:
        raw = np.stack([run_arm(ctx, sid, params, share, nasal)
                        for _, sid, nasal in arm_ids])
        data = rev.clinical(COUNTRY, raw, ctx[6][:-1], ctx[6][1:], info)
        for horizon in HORIZONS:
            health = np.asarray([
                data["yll"][:, j, :horizon, -1].sum(axis=1).mean()
                + data["yld_reported_uniform"][j, :horizon, -1].sum()
                for j in range(4)
            ])
            yll = np.asarray([
                data["yll"][:, j, :horizon, -1].sum(axis=1).mean()
                for j in range(4)
            ])
            yld = health - yll
            m0 = health[1] - health[0]
            m90 = health[3] - health[2]
            rows.append(dict(
                country=COUNTRY, scenario=label, scenario_role=role,
                maternal_days=params.maternal_days,
                timely_entry_share=share,
                sigma_maternal=params.sigma_maternal,
                pi_maternal=params.pi_maternal,
                horizon=horizon,
                nasal_benefit_m0=m0,
                nasal_benefit_m90=m90,
                maternal_interaction=m90-m0,
                percent_decline_in_nasal_benefit=(1-m90/m0) if m0 > 0 else np.nan,
                m0_yll=(yll[1]-yll[0]), m90_yll=(yll[3]-yll[2]),
                m0_yld=(yld[1]-yld[0]), m90_yld=(yld[3]-yld[2]),
            ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default=COUNTRY)
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    if args.country != COUNTRY:
        raise SystemExit("This targeted diagnostic is defined for China, the setting carrying the 58.3% headline interaction.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = rev.setup(COUNTRY)
    base = ctx[2]
    scenarios = parameter_scenarios(base)
    clinical_window_rows(base, scenarios).to_csv(out / "clinical_window_sensitivity.csv", index=False)
    policy_rows(ctx, scenarios).to_csv(out / "policy_interaction_sensitivity.csv", index=False)
    metadata = {
        "country": COUNTRY,
        "history": "adopted",
        "prospective_only": True,
        "historical_state_unchanged": True,
        "scenarios": [
            {"label": x[0], "maternal_days": x[1].maternal_days,
             "maternal_waning_stages": x[1].maternal_waning_stages,
             "sigma_maternal": x[1].sigma_maternal,
             "pi_maternal": x[1].pi_maternal,
             "timely_entry_share": x[2], "role": x[3]}
            for x in scenarios
        ],
        "clinical_target": {"window_days": 60, "maternal_ve": 0.862,
                            "effective_share_for_recalibration": 0.9},
        "nasal_profile": {"coverage": 0.7, "translation_exponent": 0.5,
                          "allocation": 0.5, "mean_years": 3},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
