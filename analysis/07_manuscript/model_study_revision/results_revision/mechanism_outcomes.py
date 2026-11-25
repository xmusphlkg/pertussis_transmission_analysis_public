"""Recover paired clinical outcomes for the adopted equal-strength and matched contrasts."""

import json
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import analyse_mechanism_matching_geometry_20261113 as geometry
from scripts import analyse_review_revision_20261110 as review
from scripts import analyse_review_round4_20261112 as grid_model

AUDIT = ROOT / "outputs/core2026/07_manuscript/model_study_revision/final_methodological_audit/matching"
OUT = AUDIT.parent.parent / "results_revision"


def summaries(data, indices, horizon):
    arms = {arm: geometry.clinical_components(data, index, horizon)
            for arm, index in indices.items()}
    for values in arms.values():
        values["harmonised_years"] = values["infant_yll"] + values["yld"]
    result = {}
    for metric in ("infant_cases", "infant_deaths", "infant_yll", "harmonised_years", "dalys"):
        first, second = arms["onward"][metric], arms["acquisition"][metric]
        for arm, value in (("onward", first), ("acquisition", second),
                           ("difference", first - second)):
            if metric == "infant_cases":
                result[arm + "_" + metric] = value[0]
            else:
                result.update(review.describe(value, arm + "_" + metric))
    return result


def main():
    matched = pd.read_csv(review.OUT / "matched_summary.csv").query("perturbation == 'reference'")
    endpoints = pd.read_csv(geometry.OUT / "reference_grid_overlap.csv").set_index(["country", "horizon"])
    responses = pd.read_csv(review.BASE / "02_sensitivity/mucosal_vaccines/bpze1_policy_20261107/followup/mechanism_response.csv")
    responses = responses.query("route == 'standalone' and bridge_strength == 0.5 and mean_years == 3 and clinical_anchor_label == 'point'")
    grid = pd.read_csv(grid_model.OUT / "mechanism_grid.csv")
    labels = list(product(grid_model.STRENGTHS, repeat=2))
    equal_rows, matched_rows = [], []
    for country in review.COUNTRIES:
        info = json.loads((review.bp.adopted.OUT / country / "complete.json").read_text())
        with np.load(grid_model.OUT / f"grid_{country}.npz") as saved:
            dates = pd.DatetimeIndex(saved["dates"])
            raw = saved["onsets"][[labels.index(point) for point in [(0, 0), (0, 1), (1, 0)]]]
        data = review.clinical(country, raw, dates[:-1], dates[1:], info)
        for horizon in (20, 30):
            row = summaries(data, {"onward": 1, "acquisition": 2}, horizon)
            points = grid.query("country == @country and horizon == @horizon and (acquisition_log_strength > 0 or onward_log_strength > 0)")
            assert len(points) == 24
            row.update(country=country, horizon=horizon, nasal_coverage=points.nasal_coverage.iloc[0],
                       combinations_with_fewer_mean_infant_deaths=(points.infant_deaths > 0).sum())
            for arm, allocation in (("onward", 0), ("acquisition", 1)):
                check = responses.query("country == @country and horizon == @horizon and allocation == @allocation")
                assert len(check) == 1
                for metric, source in (("infant_yll", "infant_0_yll_averted"), ("infant_deaths", "infant_0_deaths_averted"),
                                       ("infant_cases", "infant_0_reported_cases_averted"), ("dalys", "dalys_averted_mean")):
                    np.testing.assert_allclose(row[f"{arm}_{metric}"], check[source].iloc[0], atol=.001, rtol=1e-5)
            equal_rows.append(row)

        for condition in matched[matched.country == country].itertuples():
            horizon = condition.horizon
            endpoint = endpoints.loc[(country, horizon)]
            record = dict(country=country, horizon=horizon, nasal_coverage=condition.nasal_coverage,
                          status=condition.status, target_adolescent_cases=condition.target_adolescent_cases_averted,
                          onward_endpoint=endpoint.onward_endpoint, acquisition_endpoint=endpoint.acquisition_endpoint)
            assert (min(record["onward_endpoint"], record["acquisition_endpoint"]) > 0) == (condition.status == "matched")
            if condition.status == "matched":
                folder = AUDIT / country
                roots = pd.read_csv(folder / "roots.csv").query("horizon == @horizon and target_label == 'primary_half_endpoint'")
                with np.load(folder / "root_trajectories.npz") as saved:
                    np.testing.assert_array_equal(dates, saved["dates"])
                    trajectories = [saved["current"]]
                    for arm in ("onward", "acquisition"):
                        strength = getattr(condition, arm + "_strength")
                        root = roots[(roots.mechanism == arm) & np.isclose(roots.root_strength, strength, atol=1e-5)]
                        assert len(root) == 1
                        record[arm + "_strength"] = root.root_strength.iloc[0]
                        branch = int(root.branch.iloc[0])
                        trajectories.append(saved[f"{horizon}_primary_half_endpoint_{arm}_{branch}"])
                data = review.clinical(country, np.stack(trajectories), dates[:-1], dates[1:], info)
                record.update(summaries(data, {"onward": 1, "acquisition": 2}, horizon))
                for old, new in (("onward_minus_acquisition_dalys", "difference_dalys"),
                                 ("onward_minus_acquisition_dalys_q025", "difference_dalys_q025"),
                                 ("onward_minus_acquisition_dalys_q975", "difference_dalys_q975"),
                                 ("onward_infant_cases_averted", "onward_infant_cases"),
                                 ("acquisition_infant_cases_averted", "acquisition_infant_cases")):
                    np.testing.assert_allclose(record[new], getattr(condition, old), atol=.02, rtol=1e-4)
            matched_rows.append(record)
    assert len(matched_rows) == len(equal_rows) == 16
    assert sum(row["status"] == "matched" for row in matched_rows) == 12
    OUT.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(equal_rows).to_csv(OUT / "equal_strength_infant_outcomes.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(OUT / "matched_infant_outcomes.csv", index=False)


if __name__ == "__main__":
    main()
