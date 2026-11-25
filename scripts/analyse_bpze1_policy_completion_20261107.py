"""Complete the BPZE1 policy narrative from verified, finished model outputs.

This module reads existing simulations only. It resolves selected coverage,
decomposes price and health outcomes, and preserves annual age-specific levels.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/core2026/02_sensitivity/mucosal_vaccines/bpze1_policy_20261107'
STANDARD = ROOT / 'outputs/core2026/current_preserving_strategies_20261103'
REFERENCE = 'b050_a050_d03_point'
COUNTRIES = ['Australia', 'England', 'Sweden', 'Japan', 'New_Zealand', 'China', 'Brazil', 'Thailand']
BURDENS = ['reported_uniform', 'symptomatic_differentiated']
KEYS = ['country', 'profile_id', 'horizon', 'burden', 'mortality', 'threshold_id', 'threshold_intl2024']


class Inputs:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.sources = []
        self.metadata = {}

    def meta(self, relative):
        if relative not in self.metadata:
            path = self.folder / relative
            data = json.loads(path.read_text())
            self.sources.append(str(path.relative_to(ROOT)))
            self.metadata[relative] = data
        return self.metadata[relative]

    def csv(self, relative, manifest=None):
        self.meta(manifest) if manifest else None
        path = self.folder / relative
        self.sources.append(str(path.relative_to(ROOT)))
        return pd.read_csv(path)

    def standard(self, country):
        path = STANDARD / country / 'strategy_catalog.csv'
        relative = f'{country}/profiles/{REFERENCE}/projection_metadata.json'
        projection_path = self.folder / relative
        self.meta('metadata.json')
        projection = json.loads(projection_path.read_text())
        self.sources.extend([str(projection_path.relative_to(ROOT)), str(path.relative_to(ROOT))])
        return pd.read_csv(path)


def highest_reference(frame):
    maximum = frame.groupby('country').threshold_intl2024.transform('max')
    result = frame.loc[frame.threshold_intl2024.eq(maximum)].copy()
    # Equal-valued source endpoints remain in all-reference tables. For figure
    # tables choose one source record deterministically, without averaging values.
    first = result.groupby('country').threshold_id.agg(lambda x: sorted(set(x))[0])
    return result.loc[result.threshold_id.eq(result.country.map(first))].copy()


def resolve_frontier(frontier, catalog, standards):
    lookup = catalog.set_index(['country', 'strategy_id'])
    assert not lookup.index.duplicated().any()
    rows = []
    for item in frontier.to_dict('records'):
        assert item['price_lower'] >= 0 and item['price_upper'] > item['price_lower']
        for sid in item['optimal_strategies'].split('|'):
            record = {k: item[k] for k in KEYS + ['price_lower', 'price_upper', 'ties_at_lower', 'ties_at_upper']}
            country = item['country']
            if (country, sid) in lookup.index:
                p = lookup.loc[(country, sid)]
                record.update(strategy_id=sid, route=p.route, maternal_coverage=p.maternal_coverage,
                              nasal_coverage=p.nasal_coverage, tdap_coverage=p.tdap_coverage)
            else:
                p = standards[country].set_index('strategy_id').loc[sid]
                record.update(strategy_id=sid, route='standard', maternal_coverage=p.maternal_coverage,
                              nasal_coverage=0., tdap_coverage=p.adolescent_coverage)
            record['baseline_maternal_coverage'] = float(standards[country].loc[
                standards[country].strategy_id.eq('current'), 'maternal_coverage'].iloc[0])
            record['maternal_expanded'] = bool(record['maternal_coverage'] > record['baseline_maternal_coverage'] + 1e-12)
            rows.append(record)
    return pd.DataFrame(rows)


def collect(folder):
    data = Inputs(folder)
    meta = data.meta('metadata.json')
    assert meta['countries'] == COUNTRIES and len(meta['profile_ids']) == 17
    catalog = data.csv('product_catalog.csv', 'metadata.json')
    standards = {country: data.standard(country) for country in COUNTRIES}
    fronts = data.csv('price_frontiers.csv.gz', 'metadata.json')
    boundaries = data.csv('route_price_summary.csv', 'metadata.json')
    primary_fronts = fronts.loc[fronts.burden.eq('reported_uniform') & fronts.mortality.eq('included_base')]
    all_primary = resolve_frontier(primary_fronts, catalog, standards)
    reference = all_primary.loc[all_primary.profile_id.eq(REFERENCE)].copy()
    adoption_rows = []
    for key, group in reference.groupby(KEYS, sort=False):
        selected = group.loc[group.route.ne('standard')]
        ceiling = boundaries.loc[(boundaries[list(KEYS[:-1])].eq(pd.Series(dict(zip(KEYS[:-1], key[:-1]))))).all(axis=1)]
        max_ceiling = float(ceiling.maximum_nasal_price_vs_best_standard.max())
        has_nasal = not selected.empty
        assert has_nasal == (max_ceiling > 0)
        if has_nasal:
            np.testing.assert_allclose(selected.price_upper.max(), max_ceiling, rtol=1e-10, atol=1e-10)
        record = dict(zip(KEYS, key))
        record.update(nasal_adoption_interval=has_nasal, maximum_nasal_cost=max_ceiling,
                      selected_nasal_routes='|'.join(sorted(set(selected.route))),
                      selected_maternal_coverages='|'.join(f'{v:g}' for v in sorted(set(selected.maternal_coverage))),
                      selected_nasal_coverages='|'.join(f'{v:g}' for v in sorted(set(selected.nasal_coverage))),
                      maternal_expanded_in_nasal_interval=bool(selected.maternal_expanded.any()))
        adoption_rows.append(record)
    adoption = pd.DataFrame(adoption_rows)
    assert len(adoption) == 104 and not adoption.duplicated(KEYS).any()
    selected_primary = all_primary.loc[all_primary.route.ne('standard')].copy()
    maternal_by_profile = selected_primary.groupby(['country', 'profile_id', 'horizon']).agg(
        selected_interval_rows=('strategy_id', 'size'),
        selected_maternal_min=('maternal_coverage', 'min'),
        selected_maternal_max=('maternal_coverage', 'max'),
        baseline_maternal_coverage=('baseline_maternal_coverage', 'first'),
        any_maternal_expansion=('maternal_expanded', 'any')).reset_index()
    reference_raw = fronts.loc[fronts.profile_id.eq(REFERENCE) & fronts.burden.isin(BURDENS) & fronts.mortality.eq('included_base')]
    reference_highest = resolve_frontier(highest_reference(reference_raw), catalog, standards)
    route_highest = highest_reference(boundaries.loc[boundaries.profile_id.eq(REFERENCE) & boundaries.burden.isin(BURDENS) & boundaries.mortality.eq('included_base')])
    route_highest = route_highest[KEYS + ['route', 'maximum_nasal_price_vs_best_standard',
        'strategies_at_maximum_nasal_price', 'zero_price_optimal_strategies_within_route']].copy()
    assert len(route_highest) == 64

    costs = data.csv('followup/reference_profile_policy_summary.csv', 'followup/metadata.json').copy()
    assert len(costs) == 32 and costs.best_standard_strategy.eq('current').all()
    assert (costs.discounted_nasal_doses > 0).all()
    cost_columns = {
        'treatment_saving': 'incremental_treatment_cost_mean',
        'standard_vaccine_saving': 'standard_vaccine_incremental_cost_mean',
        'standard_delivery_saving': 'standard_delivery_incremental_cost_mean'}
    costs['health_value'] = costs.threshold_intl2024 * costs.dalys_averted_mean / costs.discounted_nasal_doses
    for target, source in cost_columns.items():
        costs[target] = -costs[source] / costs.discounted_nasal_doses
    costs['total'] = costs[['health_value', *cost_columns]].sum(axis=1)
    np.testing.assert_allclose(costs.total, costs.maximum_nasal_price_vs_best_standard, rtol=1e-9, atol=1e-8)
    costs['cost_savings_fraction'] = costs[list(cost_columns)].sum(axis=1) / costs.total
    costs['health_value_fraction'] = costs.health_value / costs.total
    costs = costs[['country', 'profile_id', 'horizon', 'route', 'threshold_intl2024', 'strategy_id',
        'maternal_coverage', 'nasal_coverage', 'tdap_coverage', 'discounted_nasal_doses',
        'health_value', *cost_columns, 'total', 'cost_savings_fraction', 'health_value_fraction',
        'dalys_averted_mean', 'dalys_averted_q025', 'dalys_averted_q975',
        'infant_0_hospitalisations_averted', 'infant_0_deaths_averted']]

    matched = data.csv('followup/matched_background_contrasts.csv.gz', 'followup/metadata.json')
    china = matched.loc[matched.country.eq('China') & matched.profile_id.eq(REFERENCE) &
        matched.horizon.eq(20) & matched.route.eq('standalone') & matched.nasal_coverage.eq(.7)].sort_values('maternal_coverage').copy()
    assert set(china.maternal_coverage) == {0., .5, .7, .9}
    component_cols = [f'{metric}_averted_{stat}' for metric in ('dalys', 'yll', 'yld') for stat in ('mean', 'q025', 'q975')]
    china = china[['country', 'profile_id', 'horizon', 'route', 'maternal_coverage', 'nasal_coverage', *component_cols]]
    for metric in ('dalys', 'yll', 'yld'):
        china[f'{metric}_reduction_from_m0'] = china[f'{metric}_averted_mean'].iloc[0] - china[f'{metric}_averted_mean']
    china['yll_share_of_daly_reduction'] = np.where(china.dalys_reduction_from_m0.ne(0), china.yll_reduction_from_m0 / china.dalys_reduction_from_m0, np.nan)
    interactions = data.csv('followup/maternal_interactions.csv.gz', 'followup/metadata.json')
    np.testing.assert_allclose(interactions.dalys_averted_mean, interactions.yll_averted_mean + interactions.yld_averted_mean, rtol=1e-9, atol=1e-8)
    interactions = interactions[['country', 'profile_id', 'bridge_strength', 'allocation', 'mean_years',
        'clinical_anchor_label', 'horizon', 'route', 'nasal_coverage', 'maternal_lower', 'maternal_higher', *component_cols]].copy()
    interactions['yll_fraction_of_interaction'] = np.where(interactions.dalys_averted_mean.ne(0),
        interactions.yll_averted_mean / interactions.dalys_averted_mean, np.nan)

    mechanism = data.csv('followup/mechanism_paired_contrasts.csv', 'followup/metadata.json')
    assert len(mechanism) == 96 and not mechanism.duplicated(['country', 'bridge_strength', 'horizon', 'route']).any()
    rank_rows = []
    for key, group in mechanism.groupby(['route', 'bridge_strength', 'horizon'], sort=False):
        assert len(group) == 8
        rank_rows.append(dict(zip(['route', 'bridge_strength', 'horizon'], key)) | dict(
            settings=len(group), acquisition_larger_adolescent_gain=int((group.adolescent_10_19_reported_cases_averted < 0).sum()),
            onward_larger_daly_gain=int((group.dalys_averted_mean > 0).sum()),
            onward_daly_interval_above_zero=int((group.dalys_averted_q025 > 0).sum()),
            acquisition_larger_daly_settings='|'.join(group.loc[group.dalys_averted_mean.lt(0), 'country'])))
    ranks = pd.DataFrame(rank_rows)

    annual_rows, lobe_rows, window_rows = [], [], []
    for country in ('Brazil', 'Japan'):
        local = f'{country}/profiles/{REFERENCE}'
        annual = data.csv(local + '/annual_cases.csv.gz', local + '/economic_metadata.json')
        receipts = data.csv(local + '/annual_receipts_and_infections.csv', local + '/economic_metadata.json')
        current_m = float(standards[country].loc[standards[country].strategy_id.eq('current'), 'maternal_coverage'].iloc[0])
        candidate = catalog.loc[catalog.country.eq(country) & catalog.profile_id.eq(REFERENCE) &
            catalog.route.eq('standalone') & catalog.maternal_coverage.eq(current_m) & catalog.nasal_coverage.eq(.9)]
        assert len(candidate) == 1
        sid = candidate.strategy_id.iloc[0]
        dose = receipts.loc[receipts.strategy_id.eq(sid)].set_index('year')
        current_dose = receipts.loc[receipts.strategy_id.eq('current')].set_index('year')
        for age in ('infant_0', 'adolescent_10_19', 'all_ages'):
            group = annual.loc[annual.age_group.eq(age)]
            current = group.loc[group.strategy_id.eq('current')].set_index('year').sort_index()
            nasal = group.loc[group.strategy_id.eq(sid)].set_index('year').sort_index()
            assert list(current.index) == list(nasal.index) == list(range(2027, 2057))
            record = pd.DataFrame(dict(country=country, profile_id=REFERENCE, strategy_id=sid,
                route='standalone', maternal_coverage=current_m, nasal_coverage=.9, age_group=age,
                year=current.index.to_numpy()))
            for metric in ('reported_cases', 'symptomatic_cases'):
                record['current_' + metric] = current[metric].to_numpy()
                record['nasal_' + metric] = nasal[metric].to_numpy()
                record[metric + '_averted'] = record['current_' + metric] - record['nasal_' + metric]
                record['cumulative_' + metric + '_averted'] = record[metric + '_averted'].cumsum()
                np.testing.assert_allclose(record[metric + '_averted'], nasal[metric + '_averted'], rtol=1e-9, atol=1e-7)
            record['current_infections_all_ages'] = current_dose.loc[current.index, 'infections'].to_numpy()
            record['nasal_infections_all_ages'] = dose.loc[current.index, 'infections'].to_numpy()
            annual_rows.append(record)
            signs = np.sign(record.reported_cases_averted.to_numpy())
            starts = np.r_[0, np.flatnonzero(signs[1:] != signs[:-1]) + 1]
            ends = np.r_[starts[1:], len(record)]
            for start, end in zip(starts, ends):
                block = record.iloc[start:end]
                lobe_rows.append(dict(country=country, age_group=age, first_year=int(block.year.iloc[0]),
                    last_year=int(block.year.iloc[-1]), sign=int(signs[start]),
                    reported_cases_averted=float(block.reported_cases_averted.sum()),
                    symptomatic_cases_averted=float(block.symptomatic_cases_averted.sum())))
            for begin, end in [(2027, 2046), (2047, 2056), (2027, 2056)]:
                block = record.loc[record.year.between(begin, end)]
                window_rows.append(dict(country=country, profile_id=REFERENCE, strategy_id=sid,
                    age_group=age, first_year=begin, last_year=end,
                    reported_cases_averted=float(block.reported_cases_averted.sum()),
                    symptomatic_cases_averted=float(block.symptomatic_cases_averted.sum()),
                    positive_years=int(block.reported_cases_averted.gt(0).sum()),
                    negative_years=int(block.reported_cases_averted.lt(0).sum())))
    annual = pd.concat(annual_rows, ignore_index=True)
    windows = pd.DataFrame(window_rows)
    follow_windows = data.csv('followup/case_time_windows.csv.gz', 'followup/metadata.json')
    for item in windows.to_dict('records'):
        check = follow_windows.loc[follow_windows.country.eq(item['country']) & follow_windows.strategy_id.eq(item['strategy_id']) &
            follow_windows.followup_start.eq(item['first_year'] - 2026) & follow_windows.followup_end.eq(item['last_year'] - 2026)]
        assert len(check) == 1
        for metric in ('reported_cases', 'symptomatic_cases'):
            np.testing.assert_allclose(item[metric + '_averted'], check[f'{item["age_group"]}_{metric}_averted'].iloc[0], rtol=1e-9, atol=1e-7)

    outputs = {
        'reference_frontiers_all_values.csv': reference,
        'reference_adoption_summary.csv': adoption,
        'primary_selected_nasal_intervals.csv.gz': selected_primary,
        'primary_selected_maternal_summary.csv': maternal_by_profile,
        'reference_frontiers_highest.csv': reference_highest,
        'reference_route_boundaries_highest.csv': route_highest,
        'reference_cost_components.csv': costs,
        'china_maternal_components.csv': china,
        'maternal_interaction_components.csv.gz': interactions,
        'mechanism_differences.csv': mechanism,
        'mechanism_rank_summary.csv': ranks,
        'annual_case_dynamics.csv': annual,
        'annual_effect_sign_blocks.csv': pd.DataFrame(lobe_rows),
        'dynamic_window_summary.csv': windows,
    }
    checks = dict(reference_comparisons=len(adoption), reference_positive_adoption_comparisons=int(adoption.nasal_adoption_interval.sum()),
        primary_nasal_interval_rows=len(selected_primary), primary_nasal_intervals_with_maternal_expansion=int(selected_primary.maternal_expanded.sum()),
        cost_boundary_reconstructions=len(costs), maternal_component_identities=len(interactions),
        mechanism_contrasts=len(mechanism), annual_case_rows=len(annual), dynamic_window_crosschecks=len(windows) * 2)
    return outputs, data.sources, checks


def run(folder=DEFAULT, *, check=False):
    folder = Path(folder).resolve()
    outputs, sources, checks = collect(folder)
    destination = folder / 'completion'
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        payload = frame.to_csv(index=False, float_format='%.15g').encode()
        if name.endswith('.gz'):
            payload = gzip.compress(payload, mtime=0)
        path = destination / name
        if check:
            assert path.read_bytes() == payload, 'Recomputed table differs: ' + name
        else:
            path.write_bytes(payload)
    metadata = dict(status='completed_policy_narrative_analysis', countries=COUNTRIES, reference_profile=REFERENCE,
        table_rows={k: len(v) for k, v in outputs.items()}, checks=checks,
        definitions=dict(frontiers='Intervals jointly compare standard, standalone nasal and coadministration strategies at a common nasal cost per dose. Source valuation records remain separate.',
            costs='Mean price boundary equals health value plus treatment, standard vaccine and standard delivery savings per discounted nasal dose; the selected standard comparator is current provision in these 32 primary rows.',
            maternal='Positive reductions in the China table equal marginal nasal benefit at maternal coverage 0 minus that at the row coverage. General interactions equal higher-coverage marginal benefit minus lower-coverage marginal benefit.',
            mechanism='Allocation 0 minus allocation 1, holding the clinical point anchor, 3-year mean duration and policy coverage fixed. Counts enumerate model conditions.',
            annual='Undiscounted annual cases for fixed baseline maternal coverage and 90% standalone nasal coverage. Consecutive sign blocks describe temporal effects, without attributing them to a specific biological process.'))
    payload = json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    if check:
        assert (destination / 'metadata.json').read_text() == payload, 'Metadata differs from verified inputs'
    else:
        (destination / 'metadata.json').write_text(payload)
    print(json.dumps(dict(status='passed' if check else 'completed', **checks), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    run(args.input_dir, check=args.check)


if __name__ == '__main__':
    main()
