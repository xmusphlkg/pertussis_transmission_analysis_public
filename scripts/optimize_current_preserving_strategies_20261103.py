"""Expected NHB, complete threshold frontiers and modelled-price decisions.

All alternatives preserve encoded current programmes. Price scenarios scale
the model's vaccine-price item. Japan's maternal item bundles vaccination and
administration. Separate delivery items and treatment unit costs stay fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_eight_country_candidate_frontiers_20261103 import frontier, ROOT, BASE
from scripts.summarize_eight_country_nhb_20261103 import summarize_nhb
from scripts.evaluate_current_preserving_strategies_20261103 import BURDENS


def threshold_points():
    oldpath = BASE/'eight_country_nhb_results_20261103/threshold_anchors.csv'
    newpath = BASE/'evidence_threshold_ranges_20261103/points.csv'
    old = pd.read_csv(oldpath)
    old['threshold_id'] = 'legacy_'+old.threshold_type
    old['threshold_intl2024'] = old.threshold
    old['threshold_family'] = 'legacy_research_anchor'
    old['role'] = old.threshold_role
    old['health_unit'] = 'mixed_QALY_proxy_or_GDP_or_assumed_DALY'
    old['source_id'] = 'legacy_configuration'
    old['method'] = old.threshold_type
    old['range_type'] = 'not_probability_distribution'
    new = pd.read_csv(newpath)
    new['threshold_family'] = 'sourced_DALY_estimate'
    fields = ['country', 'threshold_id', 'threshold_intl2024', 'threshold_family', 'role',
              'health_unit', 'source_id', 'method', 'range_type']
    result = pd.concat([old[fields], new[fields]], ignore_index=True)
    if result.duplicated(['country', 'threshold_id']).any() or not (result.threshold_intl2024 > 0).all():
        raise ValueError('Invalid threshold points')
    return result, [oldpath, newpath]


def effect_for(z, horizon, burden, mortality):
    result = z[f'yll_{horizon}'] + z[f'yld_{horizon}_{burden}'][None, :]
    if mortality == 'borrowed_noninfant':
        result = result+z[f'extra_yll_{horizon}']
    return result


def price_frontier(arms, effects, costs, vaccine_m, vaccine_a, lam):
    """Exact mean-NMB envelope in the shared vaccine-price-item multiplier q."""
    v = np.asarray(vaccine_m)+np.asarray(vaccine_a)
    other = np.asarray(costs)-v
    # NMB(q)=lambda*effect-other-q*v. Re-use the tested affine envelope solver.
    _, intervals = frontier(pd.DataFrame(dict(strategy_id=arms, dalys_averted=-v,
                                              incremental_cost=other-lam*np.asarray(effects))))
    intervals = intervals[['strategies', 'lambda_lower', 'lambda_upper', 'all_ties_at_lower_boundary']].rename(
        columns={'lambda_lower': 'price_multiplier_lower', 'lambda_upper': 'price_multiplier_upper'})
    for row in intervals.itertuples():
        q = row.price_multiplier_lower+1 if np.isinf(row.price_multiplier_upper) else (row.price_multiplier_lower+row.price_multiplier_upper)/2
        values = lam*np.asarray(effects)-other-q*v
        for arm in row.strategies.split('|'):
            np.testing.assert_allclose(values[arms.index(arm)], values.max(), rtol=1e-9, atol=1e-6)
    return intervals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--economic-dir', type=Path, default=BASE/'current_preserving_economics_20261103')
    p.add_argument('--output-dir', type=Path, default=BASE/'current_preserving_optimization_20261103')
    p.add_argument('--skip-price-grid', action='store_true')
    args = p.parse_args()
    args.economic_dir, args.output_dir = args.economic_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    thresholds, paths = threshold_points()
    paths += [Path(__file__).resolve(), args.economic_dir/'metadata.json', args.economic_dir/'summary.csv',
              ROOT/'scripts/summarize_eight_country_nhb_20261103.py', ROOT/'scripts/build_eight_country_candidate_frontiers_20261103.py']
    source_summary = pd.read_csv(args.economic_dir/'summary.csv', dtype={'strategy_id': str})
    tables = {k: [] for k in ('decisions', 'strategy_nhb', 'threshold_intervals', 'price_intervals',
                               'independent_price_grid', 'delivery_sensitivity', 'factorial_contrasts')}
    for country in source_summary.country.unique():
        path = args.economic_dir/f'{country}_components.npz'
        paths.append(path)
        z = np.load(path, allow_pickle=False)
        arms = z['strategy_ids'].astype(str).tolist()
        refs = thresholds[thresholds.country.eq(country)]
        for horizon in (20, 30):
            cost = z[f'cost_{horizon}']
            vaccine = z[f'vaccine_{horizon}']
            delivery = z[f'delivery_{horizon}']
            for mortality in ('included_base', 'borrowed_noninfant'):
                factors = {b: effect_for(z, horizon, b, mortality) for b in BURDENS}
                # Additive factorial effects (weight effect defined at reported scope).
                for i, arm in enumerate(arms):
                    r0, r1, s0, s1 = [factors[b][:, i].mean() for b in BURDENS]
                    tables['factorial_contrasts'].append(pd.DataFrame([dict(country=country, horizon=horizon,
                        mortality=mortality, strategy_id=arm, baseline_dalys_averted=r0,
                        scope_only_delta=s0-r0, weights_only_delta=r1-r0,
                        interaction=s1-s0-r1+r0, joint_delta=s1-r0)]))
                    np.testing.assert_allclose((s0-r0)+(r1-r0)+(s1-s0-r1+r0), s1-r0, atol=1e-8)
                for burden, effects in factors.items():
                    tags = dict(country=country, horizon=horizon, burden=burden, mortality=mortality,
                        price_scope=('maternal_vaccine_and_administration_bundle_adolescent_vaccine_item'
                                     if country == 'Japan' else 'modelled_vaccine_price_items'))
                    means = pd.DataFrame(dict(strategy_id=arms, dalys_averted=effects.mean(axis=0), incremental_cost=cost.mean(axis=0)))
                    existing = source_summary[(source_summary.country==country)&(source_summary.horizon==horizon)&
                        (source_summary.burden==burden)&(source_summary.mortality==mortality)].set_index('strategy_id').loc[arms]
                    np.testing.assert_allclose(means.dalys_averted, existing.dalys_averted, rtol=1e-9, atol=1e-7)
                    np.testing.assert_allclose(means.incremental_cost, existing.incremental_cost, rtol=1e-10, atol=1e-6)
                    _, intervals = frontier(means)
                    tables['threshold_intervals'].append(intervals.assign(**tags))
                    for ref in refs.to_dict('records'):
                        lam = ref['threshold_intl2024']
                        nhb, decision = summarize_nhb(effects, cost, lam, arms=arms)
                        labels = {**tags, **{k: v for k, v in ref.items() if k != 'country'}}
                        tables['decisions'].append(pd.DataFrame([{**decision, **labels}]))
                        tables['strategy_nhb'].append(nhb.assign(**labels))
                        if burden in ('reported_uniform', 'symptomatic_differentiated'):
                            pi = price_frontier(arms, effects.mean(axis=0), cost.mean(axis=0),
                                vaccine[..., 0].mean(axis=0), vaccine[..., 1].mean(axis=0), lam)
                            tables['price_intervals'].append(pi.assign(**labels))
                        if burden != 'reported_uniform' or mortality != 'included_base':
                            continue
                        # Separately recorded delivery items vary independently of price items.
                        for programme, j in (('maternal', 0), ('adolescent', 1)):
                            for mult in (.5, 1., 2.):
                                altered = cost+(mult-1)*delivery[..., j]
                                _, dec = summarize_nhb(effects, altered, lam, arms=arms)
                                tables['delivery_sensitivity'].append(pd.DataFrame([{**dec, **labels,
                                    'delivery_programme': programme, 'delivery_multiplier': mult}]))
                        if not args.skip_price_grid:
                            em = effects.mean(axis=0)
                            vm, va = vaccine[..., 0].mean(axis=0), vaccine[..., 1].mean(axis=0)
                            other = cost.mean(axis=0)-vm-va
                            records = []
                            for pm in np.linspace(0, 2, 21):
                                for pa in np.linspace(0, 2, 21):
                                    values = lam*em-other-pm*vm-pa*va
                                    tolerance = 64*np.finfo(float).eps*max(1., np.abs(values).max())
                                    winners = '|'.join(arms[i] for i in np.flatnonzero(values.max()-values <= tolerance))
                                    records.append(dict(**labels, maternal_price_multiplier=pm,
                                        adolescent_price_multiplier=pa, preferred_strategies=winners, mean_nhb=values.max()/lam))
                            tables['independent_price_grid'].append(pd.DataFrame(records))
        print(country, 'threshold and price decisions complete', flush=True)
    for name, frames in tables.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(args.output_dir/(name+'.csv.gz' if name=='independent_price_grid' else name+'.csv'), index=False)
    thresholds.to_csv(args.output_dir/'threshold_points.csv', index=False)
    (args.output_dir/'metadata.json').write_text(json.dumps(dict(
        status='conditional_expected_NHB_complete_finite_grid', reference='encoded_current',
        source_economics=str(args.economic_dir), price_change='vaccine_price_item_in_all_future_alternatives',
        japan_maternal_price='includes_administration_in_bundled_item; cannot_isolate_procurement_with_current_inputs',
        delivery_multiplier_scope='analyst_scenario_not_probability_interval',
        thresholds='each_sourced_method_separate_not_equal_probability_draws',
        source_files=[str(q.relative_to(ROOT)) for q in paths],
        output_files=[name + ('.csv.gz' if name == 'independent_price_grid' else '.csv')
                      for name in tables],
        limits='fixed transmission candidate; included burden and separately labelled borrowed mortality; finite coverage grid'), indent=2)+'\n')


if __name__ == '__main__':
    main()
