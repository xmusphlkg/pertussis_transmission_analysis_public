"""Revalue existing standard and reference nasal trajectories at 0%, 3% and 5%.

No transmission simulation or producer mutation is performed. Event-year costs,
health and actual administrations use annual discounting; future life-years and
within-episode disability use the corresponding continuous discount rate, as in
the adopted 3% valuation. Every 3% paired component is checked against its
original cache before the alternative-rate results are accepted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import evaluate_current_preserving_strategies_20261103 as ev
from scripts import run_bpze1_policy_extension_20261107 as bp
from scripts.optimize_current_preserving_strategies_20261103 import effect_for, threshold_points
from src_python.pertussis.extensions.mucosal_economics import (
    nasal_comparator_ceilings, nasal_price_frontier, nhb_at_nasal_price,
)

ROOT = ev.ROOT
OUT = ev.BASE / '02_sensitivity/mucosal_vaccines/discount_sensitivity'
PROFILE = 'b050_a050_d03_point'
COUNTRIES = ('Australia', 'England', 'Sweden', 'Japan', 'New_Zealand', 'China', 'Brazil', 'Thailand')
RATES = (0., .03, .05)
HORIZONS = (20, 30)
BURDEN = 'reported_uniform'
MORTALITY = 'included_base'


def check_rate(rate):
    if np.ndim(rate) != 0 or not np.isfinite(rate) or rate < 0:
        raise ValueError('A finite nonnegative scalar discount rate is required')
    return float(rate)


def annual_weights(years, rate):
    rate = check_rate(rate)
    if not isinstance(years, (int, np.integer)) or years < 1:
        raise ValueError('A positive integer number of years is required')
    return (1 + rate) ** -np.arange(years)


def discounted_duration(years, rate):
    """Present years of a constant flow over a duration; exact zero-rate limit."""
    rate = check_rate(rate)
    duration = np.asarray(years, dtype=float)
    if not np.isfinite(duration).all() or (duration < 0).any():
        raise ValueError('Finite nonnegative durations are required')
    return duration if rate == 0 else -np.expm1(-rate * duration) / rate


def episode_unit(entry, rate):
    return float(entry['value']) * float(discounted_duration(float(entry['duration_days']) / 365.25, rate))


def discounted_receipts(doses, horizon, rate):
    doses = np.asarray(doses, dtype=float)
    if doses.ndim < 1 or doses.shape[-1] < horizon or not np.isfinite(doses).all() or (doses < 0).any():
        raise ValueError('Finite nonnegative actual annual administrations are required')
    return doses[..., :horizon] @ annual_weights(horizon, rate)


def value_components(country, arms, raw, doses, starts, ends, info, severity, costs, life, *, rate, n=1000):
    """Independent rate-explicit implementation of the adopted paired valuation."""
    rate = check_rate(rate)
    grid = ev.AgeGrid()
    if raw.shape != (len(arms), len(starts), grid.n_bins, 3) or len(starts) != 360:
        raise ValueError('Expected the complete 30-year monthly outcome grid')
    if (not np.isfinite(raw).all() or (raw < 0).any() or doses.shape != (len(arms), 2, 30)
            or not np.isfinite(doses).all() or (doses < 0).any()):
        raise ValueError('Invalid outcomes or administration ledger')
    if len(set(arms)) != len(arms) or 'current' not in arms:
        raise ValueError('Unique strategies including current are required')
    current = arms.index('current')
    logits, ratio = ev._reporting(info, ROOT)
    fractions, groups = ev.month_geometry(starts, ends, grid)
    reporting = ev.monthly_reporting_weights(grid, logits, starts, ends)
    typical, mild = raw[..., 0], raw[..., 1]
    reported_t = np.einsum('smb,mbg->smg', typical, reporting)
    reported_m = np.einsum('smb,mbg->smg', mild * ratio, reporting)
    all_t = np.einsum('smb,mbg->smg', typical, groups)
    all_m = np.einsum('smb,mbg->smg', mild, groups)
    cases, base = reported_t + reported_m, typical + ratio * mild
    kernel = ev.sample_kernel(country, severity, costs, life, grid, logits, n=n, rate=rate)
    disability = costs['global']['disability_weights']
    ylds = ev.morbidity_factorial(reported_t, reported_m, all_t, all_m,
        episode_unit(disability['pertussis_acute_moderate'], rate),
        episode_unit(disability['pertussis_acute_mild'], rate))
    cache, components = {}, {}
    for horizon in HORIZONS:
        mask = np.arange(360) < horizon * 12
        discount = mask * annual_weights(30, rate)[np.arange(360) // 12]
        hosp = ev.integrate_endpoints(base, fractions, discount, kernel['hospital'])
        hosp_cost = ev.integrate_endpoints(base, fractions, discount, kernel['hospital_cost'])
        reported = np.einsum('smg,m->sg', cases, discount)
        outpatient = (reported[None] - hosp) * kernel['outpatient'][:, None, None]
        if np.min(outpatient) < -1e-5:
            raise ValueError('More hospitalisations than reported cases')
        if kernel['infant_risks'] is not None:
            pt, pm = kernel['infant_risks']
            counts_t = np.einsum('smb,m->sb', typical, discount)
            counts_m = np.einsum('smb,m->sb', mild, discount)
            total_yll = (np.einsum('sb,nb,b->ns', counts_t, pt, kernel['yll_bin'])
                         + np.einsum('sb,nb,b->ns', counts_m, pm, kernel['yll_bin']))
        else:
            total_yll = ev.integrate_endpoints(base, fractions, discount, kernel['yll'])[..., -1]
        overlay = kernel['overlay']
        extra_yll = ev.integrate_endpoints(base, fractions, discount,
            (overlay['yll_early'], overlay['yll_late']))[..., -1]
        receipts = discounted_receipts(doses, horizon, rate)
        vaccine = kernel['price'][:, None, :] * receipts[None]
        delivery = kernel['delivery'][:, None, :] * receipts[None]
        total_cost = vaccine.sum(axis=2) + delivery.sum(axis=2) + hosp_cost[..., -1] + outpatient[..., -1]
        cache[f'cost_{horizon}'] = total_cost - total_cost[:, [current]]
        cache[f'vaccine_{horizon}'] = vaccine - vaccine[:, [current], :]
        cache[f'delivery_{horizon}'] = delivery - delivery[:, [current], :]
        cache[f'extra_yll_{horizon}'] = extra_yll[:, [current]] - extra_yll
        cache[f'yll_{horizon}'] = total_yll[:, [current]] - total_yll
        for burden, monthly_yld in ylds.items():
            yld = np.einsum('smg,m->sg', monthly_yld, discount)
            cache[f'yld_{horizon}_{burden}'] = (yld[[current]] - yld)[:, -1]
        components[f'hospital_cost_{horizon}'] = hosp_cost[..., -1] - hosp_cost[:, [current], -1]
        components[f'outpatient_cost_{horizon}'] = outpatient[..., -1] - outpatient[:, [current], -1]
        components[f'maternal_doses_{horizon}'] = receipts[:, 0]
        components[f'tdap_doses_{horizon}'] = receipts[:, 1]
    return cache, components


def load_manifest(path):
    """Load a producer record; numerical checks are performed separately."""
    return json.loads(Path(path).read_text())


def load_country(country):
    folder = bp.OUT / country / 'profiles' / PROFILE
    projection = json.loads((folder / 'projection_metadata.json').read_text())
    original_meta = json.loads((folder / 'economic_metadata.json').read_text())
    original_path = folder / 'raw/paired_components.npz'
    source = bp.adopted.OUT / country
    info = json.loads((source / 'complete.json').read_text())
    products = pd.read_csv(folder / 'product_catalog.csv')
    with np.load(source / 'projection_arrays.npz', allow_pickle=False) as data:
        arms = data['strategy_ids'].astype(str).tolist()
        starts, ends = data['month_starts'], data['month_ends']
        raw, doses = list(data['onsets']), list(data['annual_doses'])
    standard_ids = list(arms)
    nasal = [np.zeros(30) for _ in arms]
    paths = [source / 'projection_arrays.npz', source / 'complete.json', source / 'strategy_catalog.csv',
             folder / 'projection_metadata.json', folder / 'economic_metadata.json',
             folder / 'product_catalog.csv', original_path]
    for product in products.itertuples():
        path = folder / product.trajectory_path
        with np.load(path, allow_pickle=False) as data:
            raw.append(data['onsets']); doses.append(data['annual_doses']); nasal.append(data['nasal_doses'])
        arms.append(product.strategy_id)
        paths.append(path)
    with np.load(original_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['strategy_ids'], arms)
        original = {key: data[key] for key in data.files if key != 'strategy_ids'}
    return (arms, np.stack(raw), np.stack(doses), starts, ends, info,
            products, standard_ids, np.stack(nasal), original, paths)


def highest_reference(country):
    references, paths = threshold_points()
    relevant = references.loc[references.country.eq(country)
        & references.threshold_family.eq('sourced_DALY_estimate')].copy()
    highest = relevant.loc[relevant.threshold_intl2024.eq(relevant.threshold_intl2024.max())]
    row = highest.sort_values('threshold_id').iloc[0]
    return row.to_dict(), paths


def result_tables(country, rate, arms, cache, components, nasal, products, reference):
    matching = dict(zip(products.strategy_id, products.matched_standard_strategy))
    routes = dict(zip(products.strategy_id, products.route))
    product_lookup = products.set_index('strategy_id')
    lam = float(reference['threshold_intl2024'])
    all_rows, route_rows, joint_rows = [], [], []
    for horizon in HORIZONS:
        health = effect_for(cache, horizon, BURDEN, MORTALITY)
        costs = cache[f'cost_{horizon}']
        dd = discounted_receipts(nasal, horizon, rate)
        ceilings = nasal_comparator_ceilings(arms, health, costs, dd, lam, matching)
        ceiling_by_arm = {row['strategy_id']: row for row in ceilings}
        fronts = nasal_price_frontier(arms, health, costs, dd, lam)
        nhb = nhb_at_nasal_price(health, costs, dd, lam, 0.)
        mean_nhb = nhb.mean(axis=0)
        tags = dict(country=country, profile_id=PROFILE, discount_rate=rate, horizon=horizon,
                    burden=BURDEN, mortality=MORTALITY, threshold_id=reference['threshold_id'],
                    threshold_intl2024=lam)
        joint_rows.append(dict(**tags, zero_price_optimal_strategies=fronts[0]['ties_at_lower'],
            maximum_expected_nhb=float(mean_nhb.max()),
            best_standard_strategies=ceilings[0]['best_standard_strategies'],
            maximum_nasal_ceiling=max(row['ceiling_vs_best_standard'] for row in ceilings)))
        for index, arm in enumerate(arms):
            row = dict(**tags, strategy_id=arm, route=routes.get(arm, 'standard'),
                mean_dalys_averted=float(health[:, index].mean()),
                dalys_q025=float(np.quantile(health[:, index], .025)),
                dalys_q975=float(np.quantile(health[:, index], .975)),
                mean_yll_averted=float(cache[f'yll_{horizon}'][:, index].mean()),
                yld_averted=float(cache[f'yld_{horizon}_{BURDEN}'][index]),
                mean_incremental_cost=float(costs[:, index].mean()),
                mean_zero_price_nhb=float(mean_nhb[index]), discounted_nasal_doses=float(dd[index]),
                discounted_maternal_doses=float(components[f'maternal_doses_{horizon}'][index]),
                discounted_tdap_doses=float(components[f'tdap_doses_{horizon}'][index]))
            if arm in matching:
                row.update(ceiling_by_arm[arm])
                for field in ('maternal_coverage', 'nasal_coverage', 'tdap_coverage'):
                    row[field] = float(product_lookup.loc[arm, field])
            all_rows.append(row)
        for route in bp.ROUTES:
            candidates = [row for row in ceilings if routes[row['strategy_id']] == route]
            best = max(row['ceiling_vs_best_standard'] for row in candidates)
            winners = [row for row in candidates if np.isclose(row['ceiling_vs_best_standard'], best,
                                                                rtol=1e-10, atol=1e-9)]
            subset = [i for i, arm in enumerate(arms) if arm not in matching or routes[arm] == route]
            best_nhb = mean_nhb[subset].max()
            zero = [arms[i] for i in subset if np.isclose(mean_nhb[i], best_nhb, rtol=1e-10, atol=1e-9)]
            route_rows.append(dict(**tags, route=route, maximum_nasal_ceiling=float(best),
                ceiling_strategies='|'.join(row['strategy_id'] for row in winners),
                best_standard_strategies=winners[0]['best_standard_strategies'],
                zero_price_optimal_strategies='|'.join(zero),
                maximum_expected_nhb=float(best_nhb)))
    return pd.DataFrame(all_rows), pd.DataFrame(route_rows), pd.DataFrame(joint_rows)


def run_country(country, output_dir):
    dest = Path(output_dir) / country
    manifest = dest / 'manifest.json'
    if manifest.exists():
        load_manifest(manifest)
        return {'country': country, 'status': 'verified_existing'}
    (arms, raw, doses, starts, ends, info, products, standard_ids, nasal,
     original, paths) = load_country(country)
    costs, severity = ev.load_costs(ev.COST), ev.load_severity_priors(ev.SEV)
    life_path = ROOT / costs['global']['life_table']['csv']
    life = pd.read_csv(life_path)
    reference, ref_paths = highest_reference(country)
    paths += [Path(__file__).resolve(), Path(ev.__file__).resolve(), ev.COST, ev.SEV, life_path, *ref_paths,
              ROOT / 'src_python/pertussis/extensions/mucosal_economics.py']
    dest.mkdir(parents=True, exist_ok=True)
    collected = [[], [], []]
    comparisons = []
    outputs = []
    for rate in RATES:
        cache, components = value_components(country, arms, raw, doses, starts, ends, info,
                                             severity, costs, life, rate=rate)
        if rate == .03:
            if set(cache) != set(original):
                raise ValueError('Original paired component names differ')
            for key, expected in original.items():
                np.testing.assert_allclose(cache[key], expected, rtol=1e-9, atol=1e-5,
                                           err_msg=f'{country}: 3% paired reconstruction: {key}')
                comparisons.append(dict(country=country, component=key,
                    values=int(expected.size), maximum_absolute_error=float(np.max(np.abs(cache[key]-expected)))))
        target = dest / f'paired_components_rate_{int(rate*100):02d}.npz'
        np.savez_compressed(target, strategy_ids=np.asarray(arms), **cache, **components,
            **{f'nasal_doses_{horizon}': discounted_receipts(nasal, horizon, rate) for horizon in HORIZONS})
        outputs.append(target)
        for store, frame in zip(collected, result_tables(country, rate, arms, cache, components,
                                                         nasal, products, reference)):
            store.append(frame)
    for name, frames in zip(('all_candidates', 'route_summary', 'joint_summary'), collected):
        target = dest / f'{name}.csv'
        pd.concat(frames, ignore_index=True).to_csv(target, index=False)
        outputs.append(target)
    target = dest / 'three_percent_reproduction.csv'
    pd.DataFrame(comparisons).to_csv(target, index=False)
    outputs.append(target)
    manifest.write_text(json.dumps(dict(status='completed_discount_sensitivity', country=country,
        profile_id=PROFILE, rates=RATES, horizons=HORIZONS, n_draws=1000,
        standard_strategies=len(standard_ids), nasal_strategies=len(products),
        original_component_comparisons=len(comparisons),
        input_files=[str(path.relative_to(ROOT)) for path in paths if path.exists()],
        output_files=[str(path.relative_to(ROOT)) for path in outputs]), indent=2)+'\n')
    print(f'{country}: all {len(arms)} candidates revalued; 3% paired components reproduced', flush=True)
    return {'country': country, 'status': 'completed'}


def summarise(output_dir):
    output_dir = Path(output_dir)
    source_paths, frames = [], {name: [] for name in ('all_candidates', 'route_summary', 'joint_summary',
                                                    'three_percent_reproduction')}
    for country in COUNTRIES:
        folder = output_dir / country
        record = load_manifest(folder / 'manifest.json')
        source_paths.append(folder / 'manifest.json')
        for name in frames:
            path = folder / f'{name}.csv'
            frames[name].append(pd.read_csv(path))
            source_paths.append(path)
    outputs = []
    tables = {}
    for name, parts in frames.items():
        tables[name] = pd.concat(parts, ignore_index=True)
        path = output_dir / f'{name}.csv'
        tables[name].to_csv(path, index=False)
        outputs.append(path)
    route, joint = tables['route_summary'], tables['joint_summary']
    assert len(route) == 96 and len(joint) == 48
    contrasts = []
    for (country, horizon, route_name), group in route.groupby(['country', 'horizon', 'route']):
        reference = group.loc[group.discount_rate.eq(.03)].iloc[0]
        for row in group.to_dict('records'):
            row['ceiling_difference_from_3pct'] = row['maximum_nasal_ceiling']-reference.maximum_nasal_ceiling
            row['standard_choice_changed_from_3pct'] = row['best_standard_strategies'] != reference.best_standard_strategies
            row['zero_price_choice_changed_from_3pct'] = row['zero_price_optimal_strategies'] != reference.zero_price_optimal_strategies
            contrasts.append(row)
    comparison = pd.DataFrame(contrasts)
    path = output_dir / 'rate_comparisons.csv'; comparison.to_csv(path, index=False); outputs.append(path)
    reader = ['| Setting | Years | Standalone, 0% | Standalone, 3% | Standalone, 5% | Coadministration, 0% | Coadministration, 3% | Coadministration, 5% |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for country in COUNTRIES:
        for horizon in HORIZONS:
            group = route.loc[route.country.eq(country) & route.horizon.eq(horizon)]
            values = [float(group.loc[group.route.eq(r) & group.discount_rate.eq(d), 'maximum_nasal_ceiling'].iloc[0])
                      for r in bp.ROUTES for d in RATES]
            reader.append('| '+country.replace('_', ' ')+f' | {horizon} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    path = output_dir / 'reader_table.md'; path.write_text('\n'.join(reader)+'\n'); outputs.append(path)
    for country in COUNTRIES:
        source_paths.extend(ROOT / path for path in load_manifest(output_dir/country/'manifest.json').get('output_files', []))
    path = output_dir / 'summary_manifest.json'
    path.write_text(json.dumps(dict(status='completed_discount_sensitivity_summary', countries=COUNTRIES,
        rates=RATES, horizons=HORIZONS, n_draws=1000, profile_id=PROFILE,
        scope='All existing standard and reference-profile nasal candidates; reported uniform morbidity and included mortality; highest sourced DALY valuation per setting, held fixed across rates. No new ODE integration.',
        discounting='Annual event-year costs, DALYs and actual administrations; same rate for continuous within-episode YLD and remaining-life YLL.',
        reference_component_values_checked=int(tables['three_percent_reproduction']['values'].sum()),
        maximum_3pct_absolute_error=float(tables['three_percent_reproduction'].maximum_absolute_error.max()),
        input_files=[str(p.relative_to(ROOT)) for p in sorted(set(source_paths+[Path(__file__).resolve()])) if p.exists()],
        output_files=[str(p.relative_to(ROOT)) for p in outputs]), indent=2)+'\n')
    load_manifest(path)
    return comparison


def set_policy_dir(path):
    bp.OUT = Path(path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--countries', nargs='+', choices=COUNTRIES, default=list(COUNTRIES))
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--policy-dir', type=Path, default=bp.OUT)
    args = parser.parse_args()
    set_policy_dir(args.policy_dir)
    args.output_dir = args.output_dir.resolve()
    if args.check:
        load_manifest(args.output_dir / 'summary_manifest.json')
        for country in COUNTRIES:
            load_manifest(args.output_dir / country / 'manifest.json')
        print('Discount sensitivity numerical checks passed; 3% paired reconstruction recorded.')
        return
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.jobs == 1:
        for country in args.countries:
            run_country(country, args.output_dir)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=set_policy_dir, initargs=(bp.OUT,)) as pool:
            list(pool.map(run_country, args.countries, [args.output_dir]*len(args.countries)))
    if all((args.output_dir / country / 'manifest.json').exists() for country in COUNTRIES):
        summarise(args.output_dir)


if __name__ == '__main__':
    main()
