"""Recover paired infant-only outcomes at the evaluated maternal coverage points."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyse_review_revision_20261110 import clinical

CORE = ROOT / "outputs/core2026"
POLICY = CORE / "02_sensitivity/mucosal_vaccines/bpze1_policy_20261107"
STANDARD = CORE / "current_preserving_strategies_20261103"
RESULTS = CORE / "07_manuscript/model_study_revision/results_revision"
PROFILE = "b050_a050_d03_point"


def setting_rows(country):
    standard = STANDARD / country
    folder = POLICY / country / "profiles" / PROFILE
    programmes = pd.read_csv(standard / "strategy_catalog.csv").set_index("strategy_id")
    baseline = programmes.loc["current"]
    products = pd.read_csv(folder / "product_catalog.csv").query("nasal_coverage == 0.9")
    assert not products.duplicated(["route", "maternal_coverage"]).any()
    ages = pd.read_csv(folder / "age_outcomes.csv.gz").query(
        "age_group == 'infant_0' and burden == 'reported_uniform' and mortality == 'included_base'"
    ).set_index(["strategy_id", "horizon"])
    ids = ["current", *sorted(set(products.matched_standard_strategy) - {"current"})]
    with np.load(standard / "projection_arrays.npz") as arrays:
        standard_ids = arrays["strategy_ids"].tolist()
        raw = list(arrays["onsets"][[standard_ids.index(s) for s in ids]])
        starts, ends = arrays["month_starts"], arrays["month_ends"]
    for arm in products.itertuples():
        with np.load(folder / arm.trajectory_path) as arrays:
            raw.append(arrays["onsets"])
        ids.append(arm.strategy_id)
    paired = clinical(country, np.stack(raw), starts, ends, json.loads((standard / "complete.json").read_text()))
    rows = []
    for arm in products.itertuples():
        comparator = arm.matched_standard_strategy
        control = programmes.loc[comparator]
        np.testing.assert_allclose([arm.maternal_coverage, arm.tdap_coverage],
                                   [control.maternal_coverage, control.adolescent_coverage])
        if arm.route == "standalone":
            assert arm.tdap_coverage == baseline.adolescent_coverage
        for horizon in (20, 30):
            draws = paired["yll"][:, :, :horizon, 0].sum(axis=2)
            gain = draws[:, ids.index(arm.strategy_id)] - draws[:, ids.index(comparator)]
            expected = ages.at[(arm.strategy_id, horizon), "yll_averted"] - ages.at[(comparator, horizon), "yll_averted"]
            assert gain.shape == (1000,) and np.isfinite(gain).all()
            np.testing.assert_allclose(gain.mean(), expected, atol=1e-5)
            low, high = np.quantile(gain, [.025, .975])
            row = dict(country=country, horizon=horizon, route=arm.route, profile_id=PROFILE,
                       maternal_baseline=baseline.maternal_coverage, maternal_coverage=arm.maternal_coverage,
                       nasal_coverage=arm.nasal_coverage, tdap_coverage=arm.tdap_coverage,
                       strategy_id=arm.strategy_id, comparator=comparator, infant_yll_averted=gain.mean(),
                       yll_q025=low, yll_q975=high)
            # At unchanged adolescent Tdap coverage this same no-nasal arm isolates maternal expansion.
            if arm.route == "standalone":
                deaths = paired["deaths"][:, ids.index(comparator), :horizon, 0].sum(axis=1)
                np.testing.assert_allclose(deaths.mean(), ages.at[(comparator, horizon), "deaths_averted"], atol=1e-5)
                dlo, dhi = np.quantile(deaths, [.025, .975])
                row.update(maternal_only_deaths_averted=deaths.mean(), deaths_q025=dlo, deaths_q975=dhi)
            rows.append(row)
    return rows


if __name__ == "__main__":
    countries = json.loads((POLICY / "metadata.json").read_text())["countries"]
    data = pd.DataFrame(row for country in countries for row in setting_rows(country))
    baseline = data.query("maternal_coverage == maternal_baseline")
    ref = baseline.merge(pd.read_csv(RESULTS / "infant_outcomes_reference.csv"),
                         on=["country", "strategy_id", "comparator", "horizon"], validate="one_to_one")
    assert len(countries) == 8 and len(ref) == 32
    np.testing.assert_allclose(ref[["infant_yll_averted", "yll_q025", "yll_q975"]],
                               ref[["yll_averted", "yll_averted_q025", "yll_averted_q975"]], atol=1e-6)
    data.to_csv(RESULTS / "maternal_coverage_infant_outcomes.csv", index=False)
