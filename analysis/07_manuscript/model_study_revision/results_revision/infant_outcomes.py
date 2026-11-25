"""Summarise matched infant outcomes from the adopted fixed-transmission projections.

Run from the project root with PYTHONPATH=. and the project Python environment.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyse_review_revision_20261110 import clinical


ROOT = Path(__file__).resolve().parents[4]
CORE = ROOT / 'outputs/core2026'
POLICY = CORE / '02_sensitivity/mucosal_vaccines/bpze1_policy_20261107'
STANDARD = CORE / 'current_preserving_strategies_20261103'
OUTPUT = CORE / '07_manuscript/model_study_revision/results_revision'
REFERENCE = 'b050_a050_d03_point'
OUTCOMES = ['reported_cases_averted', 'hospitalisations_averted', 'deaths_averted', 'yll_averted']


def main():
    design = json.loads((POLICY / 'metadata.json').read_text())
    profiles = design['profile_ids']
    assert len(set(profiles)) == 17
    windows = pd.read_csv(POLICY / 'followup/case_time_windows.csv.gz')
    windows = windows[windows.followup_start.eq(1)].set_index(['country', 'strategy_id', 'followup_end'])
    rows, baseline_rows = [], []
    for country in design['countries']:
        standard = STANDARD / country
        info = json.loads((standard / 'complete.json').read_text())
        catalog = pd.read_csv(standard / 'strategy_catalog.csv').set_index('strategy_id')
        maternal = catalog.loc['current', 'maternal_coverage']
        with np.load(standard / 'projection_arrays.npz') as data:
            comparator_onsets = dict(zip(data['strategy_ids'].astype(str), data['onsets']))
            starts, ends = data['month_starts'], data['month_ends']
        for profile in profiles:
            local = POLICY / country / 'profiles' / profile
            products = pd.read_csv(local / 'product_catalog.csv')
            selected = products[np.isclose(products.maternal_coverage, maternal) &
                                np.isclose(products.nasal_coverage, .9)]
            assert len(selected) == 2 and set(selected.route) == {'standalone', 'coadmin'}
            age = pd.read_csv(local / 'age_outcomes.csv.gz')
            age = age.query("burden == 'reported_uniform' and mortality == 'included_base' and age_group == 'infant_0'").set_index(['strategy_id', 'horizon'])
            if profile == REFERENCE:
                selected_ids = list(selected.matched_standard_strategy.unique())
                raw = [comparator_onsets[name] for name in selected_ids]
                for arm in selected.itertuples():
                    with np.load(local / arm.trajectory_path) as arr:
                        raw.append(arr['onsets'])
                    selected_ids.append(arm.strategy_id)
                # Zero events supply an accounting origin for absolute baseline deaths.
                raw.append(np.zeros_like(raw[0]))
                paired = clinical(country, np.stack(raw), starts, ends, info)
                current = selected_ids.index('current')
                death_delta = paired['deaths'][:, :, :20, 0].sum(axis=2)
                baseline = death_delta[:, -1] - death_delta[:, current]
                assert np.all(baseline >= 0)
                standalone = selected.loc[selected.route.eq('standalone')].iloc[0]
                assert standalone.matched_standard_strategy == 'current'
                nasal = selected_ids.index(standalone.strategy_id)
                reported = paired['reported'][:, :20, 0].sum(axis=1)
                cases = reported[-1] - reported[current]
                baseline_rows.append(dict(country=country, horizon=20, infant_reported_cases=cases,
                    infant_deaths=baseline.mean(), infant_deaths_q025=np.quantile(baseline, .025),
                    infant_deaths_q975=np.quantile(baseline, .975),
                    standalone_percent_baseline_cases=100*(reported[nasal]-reported[current])/cases,
                    standalone_percent_baseline_deaths=100*(death_delta[:,nasal]-death_delta[:,current]).mean()/baseline.mean(),
                    standalone_deaths_averted_first_10_years=paired['deaths'][:,nasal,:10,0].sum(axis=1).mean(),
                    standalone_deaths_averted_second_10_years=paired['deaths'][:,nasal,10:20,0].sum(axis=1).mean()))
            for arm in selected.itertuples():
                matched = arm.matched_standard_strategy
                assert np.isclose(catalog.loc[matched, 'adolescent_coverage'], arm.tdap_coverage)
                assert np.isclose(catalog.loc[matched, 'maternal_coverage'], maternal)
                for horizon in (20, 30):
                    contrast = age.loc[(arm.strategy_id, horizon), OUTCOMES] - age.loc[(matched, horizon), OUTCOMES]
                    cases = windows.loc[(country, arm.strategy_id, horizon), 'infant_0_reported_cases_averted']
                    np.testing.assert_allclose(contrast.reported_cases_averted, cases, atol=1e-7)
                    row = dict(country=country, profile_id=profile, duration_years=arm.mean_years,
                               translation_exponent=arm.bridge_strength, acquisition_allocation=arm.allocation,
                               clinical_anchor=arm.clinical_anchor_label, route=arm.route, horizon=horizon,
                               maternal_coverage=maternal, nasal_coverage=arm.nasal_coverage,
                               tdap_coverage=arm.tdap_coverage, comparator=matched, strategy_id=arm.strategy_id,
                               **contrast.to_dict())
                    if profile == REFERENCE:
                        for label, field in [('deaths_averted','deaths'), ('yll_averted','yll')]:
                            totals = paired[field][:, :, :horizon, 0].sum(axis=2)
                            draw = totals[:, selected_ids.index(arm.strategy_id)] - totals[:, selected_ids.index(matched)]
                            assert len(draw) == 1000
                            np.testing.assert_allclose(draw.mean(), row[label], atol=1e-7)
                            row[label+'_q025'], row[label+'_q975'] = np.quantile(draw, [.025, .975])
                    rows.append(row)
    results = pd.DataFrame(rows)
    assert len(results) == 8*17*2*2 and not results.duplicated(['country','profile_id','route','horizon']).any()
    reference = results.query('profile_id == @REFERENCE')
    report = []
    for (country, horizon), group in results[results.route.eq('standalone')].groupby(['country','horizon'], sort=False):
        assert len(group) == 17 and group.profile_id.nunique() == 17
        row = dict(country=country, horizon=horizon)
        for metric in OUTCOMES:
            v = group[metric]
            row.update({metric+'_positive_profiles': int((v>0).sum()), metric+'_min': v.min(),
                        metric+'_median': v.median(), metric+'_max': v.max()})
        report.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.drop(columns=[c for c in results if c.endswith(('_q025','_q975'))]).to_csv(OUTPUT/'infant_outcomes_by_profile.csv', index=False)
    reference.to_csv(OUTPUT/'infant_outcomes_reference.csv', index=False)
    pd.DataFrame(baseline_rows).to_csv(OUTPUT/'baseline_infant_deaths.csv', index=False)
    pd.DataFrame(report).to_csv(OUTPUT/'infant_profile_ranges.csv', index=False)


if __name__ == '__main__':
    main()
