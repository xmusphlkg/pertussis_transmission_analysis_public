"""Build clinical-data-anchored BPZE1 policy profiles on the adopted calendar.

The observed input is the randomized-arm arithmetic culture AUC ratio. The
bridge acts on its log relative effect. A reference day and an independent
waning clock translate this reference profile into protected-state parameters;
the model does not fit the human challenge study's culture observations.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/extensions/bpze1_policy_20261107.json"
EVIDENCE = Path("outputs/core2026/05_evidence/mucosal_vaccines/trial_reanalysis")
LEDGER = Path("analysis/05_evidence/mucosal_vaccines/clinical_endpoints_20261107.json")
REANALYSIS = Path("scripts/analyse_bpze1_clinical_evidence_20261107.py")
BUILDER = Path("scripts/build_bpze1_policy_design_20261107.py")
METRIC = "whole_group_arithmetic_auc_ratio"
REFERENCE_DAY = 90.0


def survival(day: float, mean_years: float) -> float:
    """Erlang-2 protection survival since vaccination."""
    if not math.isfinite(day) or day < 0:
        raise ValueError("Reference day must be finite and nonnegative")
    if not math.isfinite(mean_years) or mean_years <= 0:
        raise ValueError("Protection duration must be finite and positive")
    z = 2.0 * day / (mean_years * 365.25)
    return math.exp(-z) * (1.0 + z)


def profile(ratio: float, bridge: float, allocation: float, duration: float,
            label: str = "point", reference_day: float = REFERENCE_DAY) -> dict:
    """Calibrate a low-exposure reference product, retaining incompatible cases."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("AUC ratio must be finite and strictly positive")
    if not math.isfinite(bridge) or not 0 <= bridge <= 1:
        raise ValueError("Bridge strength must be between zero and one")
    if not math.isfinite(allocation) or not 0 <= allocation <= 1:
        raise ValueError("Allocation must be between zero and one")
    s = survival(reference_day, duration)
    residual = ratio ** bridge
    state = (residual - (1.0 - s)) / s
    identity = f"b{round(100*bridge):03d}_a{round(100*allocation):03d}_d{duration:02g}_{label}"
    common = dict(candidate_id=identity, mean_years=duration,
                  bridge_strength=bridge, allocation=allocation,
                  clinical_anchor_label=label, observed_auc_ratio=ratio,
                  reference_day=reference_day,
                  reference_residual_potential=residual,
                  protected_state_residual=state, anchor_survival=s)
    if not 0 <= state <= 1:
        reason = ("reference_residual_below_unprotected_fraction" if state < 0
                  else "reference_residual_above_one")
        return dict(**common, compatible=False, rejection_reason=reason,
                    minimum_reference_residual=1.0 - s,
                    explanation="The declared waning clock and a protected-state multiplier between zero and one cannot attain this reference residual; no parameter clipping is applied.")
    # A zero product is split by its stated allocation, with x**0 = 1.
    susceptibility = state ** allocation
    infectiousness = state ** (1.0 - allocation)
    common.update(compatible=True, efficacy=1.0 - susceptibility,
                  infectiousness_multiplier=infectiousness,
                  acquisition_multiplier=susceptibility,
                  role="zero_effect_control" if bridge == 0 else "clinical_anchored_policy_scenario")
    for day in (60, 90, 120):
        s_day = survival(day, duration)
        common[f"predicted_residual_day{day}"] = 1.0 - s_day + s_day * state
    return common


def clinical_inputs(root: Path = ROOT) -> tuple[dict, dict]:
    """Read original interval anchors and retain the complete bootstrap record."""
    meta_path = root / EVIDENCE / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    for filename in ("effect_decomposition.csv", "bootstrap_joint_effects.csv.gz"):
        path = root / EVIDENCE / filename
        if not path.is_file():
            raise ValueError(f"Clinical source missing: {filename}")
    workbook = root / metadata["source_path"]
    if not workbook.is_file():
        raise ValueError("Participant workbook is missing")
    with (root / EVIDENCE / "effect_decomposition.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["metric"] == METRIC]
    if len(rows) != 1:
        raise ValueError("Expected one whole-group arithmetic AUC row")
    row = rows[0]
    values = dict(point=float(row["estimate"]),
                  ci95_lower=float(row["bootstrap_percentile_95_lower"]),
                  ci95_upper=float(row["bootstrap_percentile_95_upper"]))
    if not 0 < values["ci95_lower"] <= values["point"] <= values["ci95_upper"]:
        raise ValueError("Clinical AUC interval anchors are invalid")
    with gzip.open(root / EVIDENCE / "bootstrap_joint_effects.csv.gz", "rt", newline="") as handle:
        ratios = [float(record[METRIC]) for record in csv.DictReader(handle)]
    expected = metadata["bootstrap"]["replicates"]
    if len(ratios) != expected or any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("Bootstrap AUC ratios must retain all finite positive records")
    record = dict(metric=METRIC, **values, bootstrap_replicates=len(ratios),
                  bootstrap_above_one=sum(x > 1 for x in ratios),
                  bootstrap_minimum=min(ratios), bootstrap_maximum=max(ratios),
                  participant_counts={"BPZE1": 24, "placebo": 21},
                  bootstrap_handling="All resamples remain in the source without clipping. Point and 2.5th/97.5th percentile anchors define sensitivity profiles; they do not constitute a joint probabilistic policy analysis.")
    sources = [BUILDER, REANALYSIS, LEDGER, EVIDENCE / "effect_decomposition.csv",
               EVIDENCE / "bootstrap_joint_effects.csv.gz", EVIDENCE / "metadata.json",
               Path(metadata["source_path"])]
    return record, [str(path) for path in sources]


def build_design(root: Path = ROOT) -> dict:
    clinical, source_files = clinical_inputs(root)
    planned = [(b, a, 3, "point") for b in (.25, .5, 1.) for a in (0., .5, 1.)]
    planned += [(b, .5, d, "point") for d in (1, 6) for b in (.25, .5, 1.)]
    planned += [(.5, .5, 10, "point"), (.5, .5, 3, "ci95_lower"),
                (.5, .5, 3, "ci95_upper"), (0., .5, 3, "point")]
    all_profiles = [profile(clinical[label], b, a, d, label) for b, a, d, label in planned]
    accepted = [p for p in all_profiles if p["compatible"]]
    rejected = [p for p in all_profiles if not p["compatible"]]
    return dict(
        analysis="bpze1_clinical_anchored_policy_extension", date="2026-11-07",
        source_files=source_files,
        time=dict(start_year=2027, end_year_exclusive=2057, horizons=[20, 30], discount_rate=.03),
        economics=dict(n_draws=1000, clinical_cost_seed=20261101,
                       primary_burden="reported_uniform", primary_mortality="included_base",
                       currency="2024_international_dollars",
                       price_estimand="Additional procurement and administration cost per actual nasal receipt, with standard programme costs retained by route"),
        routes=["standalone", "coadmin"],
        countries=["Australia", "England", "Sweden", "Japan", "New_Zealand", "China", "Brazil", "Thailand"],
        clinical_input=clinical,
        bridge=dict(
            reference_day=REFERENCE_DAY,
            reference_day_basis="90 days is the declared midpoint of the 60-120-day vaccination-to-challenge window, not an observed participant-specific challenge time.",
            formula="reference_residual_potential = observed_arithmetic_auc_ratio ** bridge_strength",
            scale="Attenuation of the observed log relative AUC effect; bridge strength 0.5 halves the log effect.",
            calibration_target="Expected multiplicative effect of susceptibility and infectiousness in a vaccinated cohort at one low-intensity exposure on reference day 90.",
            protected_state_formula="(reference_residual_potential - (1 - Erlang2_survival_at_reference_day)) / Erlang2_survival_at_reference_day",
            allocation_formula="acquisition_multiplier = protected_state_residual ** allocation; infectiousness_multiplier = protected_state_residual ** (1-allocation)",
            allocation_interpretation="Structural distribution of a fixed protected-state product between acquisition and onward contribution; the allocation is not identified by the challenge AUC.",
            transport_dimensions=["Culture AUC to effective transmission contribution", "wP-primed adults to adolescents with their setting-specific priming history", "Trial background immunity to the modelled immunisation states and coadministration"],
            observation_scope="The data anchor is randomized-arm culture AUC on challenge days 9, 11 and 14; clinical culture trajectories and natural human-to-human transmission are distinct outcomes.",
            waning=dict(distribution="erlang", shape=2, start="vaccination", stage_rate_formula="2/(mean_years*365.25)", interpretation="Protection duration is a separate policy condition; clinical observations establish the short-term AUC anchor."),
            inferential_scope="Profiles are clinical-data-anchored transmission scenarios. Their reference-product calibration does not fit the human challenge observation process."),
        primary_reference_profile="b050_a050_d03_point",
        primary_reference_role="Declared illustrative policy reference within structural and duration comparisons; not an estimated clinical transmission efficacy or an empirically selected mechanism.",
        profiles=[p for p in accepted if p["bridge_strength"] > 0],
        zero_control=next(p for p in accepted if p["bridge_strength"] == 0),
        rejected_profiles=rejected,
        counts=dict(positive_profiles=sum(p["bridge_strength"] > 0 for p in accepted),
                    zero_effect_profiles=sum(p["bridge_strength"] == 0 for p in accepted),
                    rejected_profiles=len(rejected)),
        comparisons=dict(coverage_grid="All original maternal coverage levels crossed with positive adolescent nasal coverage levels", standard_choice_set="All 73 original standard strategies across the eight settings", pairing="Same history, valuation reference, horizon and clinical/cost draw position", uncertainty="Clinical/cost intervals are conditional on the specified clinical anchor, bridge, allocation and duration; clinical anchor endpoints are evaluated separately."),
        output_directory="outputs/core2026/02_sensitivity/mucosal_vaccines/bpze1_policy_20261107")


def encoded(design: dict) -> bytes:
    return (json.dumps(design, indent=2, allow_nan=False) + "\n").encode()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CONFIG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = encoded(build_design())
    if args.check:
        if not args.out.exists() or args.out.read_bytes() != data:
            raise SystemExit(f"BPZE1 design differs from current clinical inputs: {args.out}")
        print("BPZE1 clinical policy design verified")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
