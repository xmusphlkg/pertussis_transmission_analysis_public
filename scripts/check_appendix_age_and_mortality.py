"""Replay monthly age accounting and value course-specific infant mortality."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.analyse_core2026_england_mortality import exposure_by_year
from scripts.run_bpze1_policy_extension_20261107 import arm_id, policy_catalog
from src_python.pertussis.config import REPO_ROOT
from src_python.pertussis.economics.evaluate import DEATH_RISK_SOURCE
from src_python.pertussis.model.kernel import _annual_shift, _monthly_shift
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.observation.mortality import coarsened_two_age_risk_draws, joint_risk_posterior_draws
from src_python.pertussis.scenarios.current_preserving import coverage_grid
from src_python.pertussis.scenarios.fitted import build_fitted_inputs, fit_parameters, load_fit

BASE = REPO_ROOT / 'outputs/core2026'
STANDARD = BASE / 'current_preserving_strategies_20261103'
NASAL = BASE / '02_sensitivity/mucosal_vaccines/bpze1_policy_20261107'
OUT = BASE / '02_sensitivity/appendix_age_mortality'
DATA = REPO_ROOT / 'data/processed/health_outcomes_20261026'
RISK_RATIOS = (0., .1, .25, .5)


def age_check(item):
    country, directory = item
    spec, fit = load_fit(REPO_ROOT / directory)
    inputs = build_fitted_inputs(spec, 2057)
    params = fit_parameters(spec, fit.to_dict())
    meta = json.loads((STANDARD / country / 'complete.json').read_text())
    policies, _, _ = coverage_grid(inputs, start_year=2027, end_year=2057,
                                   adolescent_age=meta['adolescent_age'],
                                   maternal_effective_share=meta.get('maternal_effective_share', .9))
    sim = Simulator(policies['current'])
    dates = pd.DatetimeIndex(inputs.calendar.dates)
    days = np.flatnonzero((dates.day == 15) & (dates.year <= 2056))
    assert len(days) == 157 * 12
    result = sim.run(params, *sim.initial_state(), 0, days[-1]+1, snap_days=days)
    populations = result.snapshots.sum(axis=2)
    assert np.isfinite(populations).all() and populations.min() >= 0
    errors = []
    for day, population in zip(days, populations):
        date = dates[day]
        first = pd.Timestamp(date.year, date.month, 1)
        mid = first + (first + pd.offsets.MonthBegin(1) - first) / 2
        start = pd.Timestamp(date.year, 1, 1)
        fraction = (mid-start)/(pd.Timestamp(date.year+1, 1, 1)-start)
        W = inputs.grid.completed_age_weights(fraction)
        bands = np.column_stack([W[:, :1].sum(1), W[:, 1:10].sum(1), W[:, 10:20].sum(1), W[:, 20:].sum(1)])
        assert np.all(bands >= 0)
        np.testing.assert_allclose(bands.sum(1), 1, atol=1e-14, rtol=0)
        errors.append(abs((population @ bands).sum()-population.sum())/population.sum())
    # Distinct integer counts expose duplication or loss for each compartment.
    tracked = np.arange(result.y.size, dtype=float).reshape(result.y.shape)
    before = tracked.copy()
    _annual_shift(tracked, 24, 42)
    np.testing.assert_array_equal(tracked[43], before[43]+before[42])
    _monthly_shift(tracked, 24, 24)
    np.testing.assert_array_equal(tracked.sum(0), before.sum(0))
    assert max(errors) < 1e-12
    return dict(country=country, months=len(days), max_relative_population_error=max(errors))


def mortality_risks():
    eng = BASE/'programme_scenarios_20261029/England_risk_2001_2023'
    aus = BASE/'v2_australia_mortality_20261102'
    hist = pd.read_csv(BASE/'health_outcomes_20261026/fitted_cycle2025_england_main/historical_age_onsets.csv.gz')
    deaths = pd.read_csv(DATA/'england_linked_infant_deaths.csv').query('2001 <= year <= 2023')
    au = pd.read_csv(aus/'risk_draws.csv.gz')
    en = pd.read_csv(eng/'risk_draws.csv.gz')
    def strata(scenario):
        return au[au.risk_scenario.eq(scenario)].sort_values('risk_age').drop_duplicates('risk_age')
    ref = strata('ACR_reference')
    typical = ref.historical_effective_onsets.to_numpy()
    mild = strata('ACR_equal_course_risk').historical_effective_onsets.to_numpy()-typical
    totals = pd.read_csv(DATA/'australia_infant_death_total_constraints.csv').query("stream == 'ACR' and start_year == 2012").iloc[0]
    risks = {}
    for ratio in RISK_RATIOS:
        aq = joint_risk_posterior_draws(ref.count_lower.to_numpy(),ref.count_upper.to_numpy(),
                typical + ratio*mild,total_lower=int(totals.infant_total_lower),total_upper=int(totals.infant_total_upper),
                ascertainment=1.,prior_shape=.5,size=4096,seed=20261102)
        eq = coarsened_two_age_risk_draws(deaths.deaths_lower.to_numpy(),exposure_by_year(hist,deaths.year,ratio),
                ascertainment=1.,prior_shape=.5,size=4096,seed=20261026)
        for label, q, frame, scenario in [('Australia', aq, au, 'ACR_reference'), ('England', eq, en, 'reference')]:
            if ratio == 0:
                saved = frame[frame.risk_scenario.eq(scenario)].pivot(index='risk_draw_id',columns='risk_age',values='p_death_given_typical')
                np.testing.assert_allclose(q, saved, rtol=1e-7, atol=1e-14)
            risks[label, ratio] = q
    return risks


def mortality_check():
    risks = mortality_risks()
    rows = []
    for country, donor in DEATH_RISK_SOURCE.items():
        catalog = pd.read_csv(STANDARD/country/'strategy_catalog.csv')
        with np.load(STANDARD / country / 'projection_arrays.npz') as standard:
            assert str(standard['month_starts'][0]).startswith('2027-01')
            arms = dict(zip(standard['strategy_ids'].astype(str), standard['onsets']))
        policies = policy_catalog(catalog)
        current = catalog.loc[catalog.strategy_id.eq('current')].squeeze()
        coverage = max(.7, current.adolescent_coverage)
        subset = policies[policies.route.eq('standalone') & np.isclose(policies.nasal_coverage, coverage)]
        split = 2 if donor == 'Australia' else 3
        def counts(arr):
            # Channels record typical onsets, mild onsets and entry into latency.
            infant = arr[:240, :12, :3].sum(0)
            return np.stack([infant[:split].sum(0), infant[split:].sum(0)])
        high = catalog.loc[np.isclose(catalog.maternal_coverage, .9) &
                           np.isclose(catalog.adolescent_coverage, current.adolescent_coverage),
                           'strategy_id'].item()
        contrasts = {'maternal_to_90': counts(arms['current'])-counts(arms[high])}
        for background, label in [('current', 'nasal_at_current'), (high, 'nasal_at_90')]:
            policy = subset[subset.matched_standard_strategy.eq(background)].iloc[0]
            folder = NASAL/country/'profiles/b050_a050_d03_point/raw'
            with np.load(folder/(arm_id('b050_a050_d03_point',policy)+'.npz')) as z:
                contrasts[label] = counts(arms[background]) - counts(z['onsets'])
        for contrast, diff in contrasts.items():
            for ratio in RISK_RATIOS:
                q = risks[donor, ratio]
                typical_deaths = q @ diff[:, 0]
                mild_deaths = (ratio * q) @ diff[:, 1]
                expected = typical_deaths + mild_deaths
                rows.append(dict(country=country, contrast=contrast,
                    baseline_maternal_coverage=current.maternal_coverage,
                    nasal_coverage=0. if contrast == 'maternal_to_90' else coverage,
                    mild_to_typical_risk=ratio, typical_onsets_averted=diff[:,0].sum(),
                    mild_onsets_averted=diff[:,1].sum(), total_onsets_averted=diff[:, :2].sum(),
                    acquisitions_averted=diff[:, 2].sum(),
                    typical_deaths_averted=typical_deaths.mean(), mild_deaths_averted=mild_deaths.mean(),
                    mean_deaths_averted=expected.mean(), q025=np.quantile(expected,.025),q975=np.quantile(expected,.975)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['age','mortality'])
    part = parser.parse_args().part
    OUT.mkdir(parents=True,exist_ok=True)
    if part == 'age':
        fits = json.loads((BASE/'current_preserving_fit_map_20261103.json').read_text())
        with ProcessPoolExecutor(max_workers=2) as pool:
            table = pd.DataFrame(pool.map(age_check,fits.items()))
    else:
        table = mortality_check()
    table.to_csv(OUT/(part+'.csv'),index=False)
