"""Summarise existing programme trajectories for Table 1 and the age comparison."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'outputs/core2026'
OUT = DATA / '07_manuscript/model_study_revision/results_revision'
PROGRAMMES = DATA / 'current_preserving_strategies_20261103'
SETTINGS = ('Australia', 'England', 'Sweden', 'Japan', 'New_Zealand', 'China', 'Brazil', 'Thailand')
AGES = ['infant_0', 'child_1_9', 'adolescent_10_19', 'adult_20plus']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = []
    effects = []
    clinical = pd.read_csv(DATA / 'current_preserving_economics_20261103/age_effects.csv')
    clinical = clinical.query("horizon == 20 and burden == 'reported_uniform' and mortality == 'included_base' and age_group == 'infant_0'")
    for country in SETTINGS:
        folder = PROGRAMMES / country
        catalog = pd.read_csv(folder / 'strategy_catalog.csv')
        current = catalog.set_index('strategy_id').loc['current']
        births = pd.read_csv(folder / 'population_births.csv').query("strategy_id == 'current' and 2027 <= year <= 2046")
        cases = pd.read_csv(folder / 'annual_cases.csv').query('2027 <= year <= 2046')
        assert births.year.nunique() == len(births) == 20
        assert cases.groupby(['strategy_id', 'age_group']).size().eq(20).all()
        cumulative = cases.groupby(['strategy_id', 'age_group']).reported_cases.sum().unstack()
        np.testing.assert_allclose(cumulative[AGES].sum(axis=1), cumulative.all_ages)
        infant_cases = cumulative.at['current', 'infant_0']
        baseline.append(dict(country=country, maternal_coverage=current.maternal_coverage,
                             adolescent_coverage=current.adolescent_coverage, live_births=births.births.sum(),
                             infant_reported_cases=infant_cases,
                             infant_reported_cases_per_100000_live_births=infant_cases * 100000 / births.births.sum()))
        for programme, arm in [('Maternal', 'current_M0.7_Akeep'), ('Adolescent Tdap', 'current_Mkeep_A0.7')]:
            if arm not in cumulative.index:
                continue
            difference = cumulative.loc['current'] - cumulative.loc[arm]
            row = clinical.loc[(clinical.country == country) & (clinical.strategy_id == arm)].squeeze()
            np.testing.assert_allclose(difference.infant_0, row.reported_cases_averted)
            effects.append(dict(country=country, programme=programme, **difference[AGES + ['all_ages']].to_dict(),
                                infant_deaths_averted=row.deaths_averted, infant_yll_averted=row.yll_averted))
    assert len(effects) == 10
    pd.DataFrame(baseline).to_csv(OUT / 'baseline_programmes.csv', index=False)
    pd.DataFrame(effects).to_csv(OUT / 'standard_expansions.csv', index=False)


if __name__ == '__main__':
    main()
