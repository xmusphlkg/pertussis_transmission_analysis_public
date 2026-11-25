"""Compare trial-informed BPZE1 routes in the adopted eight-setting policy grid.

Each clinical/bridging profile is evaluated separately. Within that comparison,
all standard strategies, independent nasal vaccination and Tdap coadministration
compete using the same paired clinical/cost draws and actual dose ledgers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from scripts import run_mucosal_vaccine_extension_20261107 as prior
from scripts import run_current_preserving_strategies_20261103 as adopted
from scripts import evaluate_current_preserving_strategies_20261103 as economic
from scripts.optimize_current_preserving_strategies_20261103 import threshold_points, effect_for
from src_python.pertussis.config import REPO_ROOT as ROOT
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.scenarios.current_preserving import coverage_grid
from src_python.pertussis.scenarios.fitted import build_fitted_inputs, fit_parameters, history_to, load_fit
from src_python.pertussis.scenarios.programme_scenarios import _target_event

CONFIG = ROOT / 'config/extensions/bpze1_policy_20261107.json'
BASE = ROOT / 'outputs/core2026'
OUT = BASE / '02_sensitivity/mucosal_vaccines/bpze1_policy_20261107'
MODULES = [Path(__file__).resolve(), ROOT / 'src_python/pertussis/extensions/bpze1.py',
           ROOT / 'src_python/pertussis/extensions/mucosal.py',
           ROOT / 'src_python/pertussis/extensions/mucosal_economics.py', Path(prior.__file__).resolve()]
ROUTES = ('standalone', 'coadmin')
dump_json, save_arrays = prior.dump_json, prior.save_arrays

def load_design(path):
    config = json.loads(Path(path).read_text())
    timing = config['time']
    if (timing['start_year'], timing['end_year_exclusive'], timing['horizons'], timing['discount_rate']) != (2027, 2057, [20, 30], .03):
        raise ValueError('Use the adopted 2027-2056 calendar, horizons and discount rate')
    if (config['economics']['n_draws'], config['economics']['clinical_cost_seed']) != (1000, 20261101):
        raise ValueError('Use the adopted 1000 paired clinical/cost draws and seed')
    profiles = config['profiles']
    ids = [p['candidate_id'] for p in profiles]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_]+', p) for p in ids):
        raise ValueError('Unique path-safe candidate IDs required')
    for p in profiles:
        e, r, d = (p[key] for key in ('efficacy', 'infectiousness_multiplier', 'mean_years'))
        if not all(np.isfinite(x) for x in (e, r, d)) or not 0 <= e <= 1 or not 0 <= r <= 1 or d*365.25 < 2:
            raise ValueError('Invalid clinical bridge profile')
        if e == 0 and r == 1:
            raise ValueError('Zero-effect controls are generated separately')
    return config


def selected_profiles(config, ids=None):
    wanted = [p['candidate_id'] for p in config['profiles']] if ids is None else list(ids)
    if len(wanted) != len(set(wanted)) or not set(wanted) <= {p['candidate_id'] for p in config['profiles']}:
        raise ValueError('Unknown or duplicate selected profile')
    return [p for p in config['profiles'] if p['candidate_id'] in wanted]


def policy_catalog(catalog):
    """Cross both routes with every feasible maternal/nasal coverage pair.

Standalone keeps Tdap at its original coverage. Its background comparator is
therefore the same maternal strategy with baseline Tdap; the separate coverage
comparator is the conventional Tdap strategy at the nasal target coverage.
"""
    if catalog.strategy_id.duplicated().any() or catalog.empty:
        raise ValueError('Unique nonempty standard catalog required')
    current = catalog.loc[catalog.strategy_id.eq('current')].iloc[0]
    rows = []
    for row in catalog.itertuples():
        if row.adolescent_coverage <= 0:
            continue
        background = catalog.loc[np.isclose(catalog.maternal_coverage, row.maternal_coverage)
                                 & np.isclose(catalog.adolescent_coverage, current.adolescent_coverage)]
        if len(background) != 1:
            raise ValueError('Expected one baseline-Tdap comparator for each maternal coverage')
        for route in ROUTES:
            sid = row.strategy_id if route == 'coadmin' else background.iloc[0].strategy_id
            rows.append({'route': route, 'coverage_strategy': row.strategy_id,
                'matched_standard_strategy': sid, 'target_coverage_standard_strategy': row.strategy_id,
                'maternal_coverage': float(row.maternal_coverage),
                'adolescent_coverage': float(row.adolescent_coverage),
                'nasal_coverage': float(row.adolescent_coverage),
                'tdap_coverage': float(row.adolescent_coverage if route == 'coadmin' else current.adolescent_coverage),
                'adolescent_age': float(row.adolescent_target_age)})
    if not rows:
        raise ValueError('Expected positive nasal coverage choices')
    return pd.DataFrame(rows)


def arm_id(profile_id, row):
    return f'bpze1_{profile_id}_{row["route"]}__{row["coverage_strategy"]}'


def profile_tags(profile):
    return {('profile_id' if key == 'candidate_id' else key): value
            for key, value in profile.items() if value is None or isinstance(value, (str, int, float, bool))}


def projection_sources(country, config_path, old_folder, old_info):
    sources = {name: True for name in prior.projection_sources(country, config_path, old_folder, old_info)}
    sources.update({str(p.relative_to(ROOT)): True for p in MODULES})
    sources.update({str(p): True for p in load_design(config_path).get('source_files', [])})
    return sources


def valid_metadata(path, sources, root):
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    names = record.get('output_files') or record.get('output_sha256', {})
    return bool(names) and all((root / name).is_file() for name in names)


def run_country(country, config_path, output_dir, profile_ids=None):
    from src_python.pertussis.extensions.bpze1 import BPZE1Simulator

    config = load_design(config_path)
    profiles = selected_profiles(config, profile_ids)
    dest = Path(output_dir) / country
    prior.checked.configure()
    old_folder, old_info, _ = adopted.source_for(country)
    sources = projection_sources(country, config_path, old_folder, old_info)
    selected = ['zero_control', *[p['candidate_id'] for p in profiles]]
    if all(valid_metadata(dest / 'profiles' / pid / 'projection_metadata.json', sources,
                          dest / 'profiles' / pid) for pid in selected):
        return {'country': country, 'status': 'reused_projections', 'profiles': len(profiles)}
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spec, fit = load_fit(ROOT / old_info['fit_directory'])
    start, end = config['time']['start_year'], config['time']['end_year_exclusive']
    inputs = build_fitted_inputs(spec, end)
    params = fit_parameters(spec, fit.to_dict())
    info = json.loads((adopted.OUT / country / 'complete.json').read_text())
    age, effective = float(info['adolescent_age']), float(info.get('maternal_effective_share', .9))
    scenarios, catalog, _ = coverage_grid(inputs, start_year=start, end_year=end,
        adolescent_age=age, maternal_effective_share=effective)
    policies = policy_catalog(catalog)
    with np.load(adopted.OUT / country / 'projection_arrays.npz', allow_pickle=False) as z:
        standard = {key: z[key] for key in z.files}
    standard_ids = standard['strategy_ids'].astype(str).tolist()
    if standard_ids != list(scenarios):
        raise ValueError('Rebuilt catalog differs from the adopted catalog')
    first, last = inputs.calendar.year_start_day(start), inputs.calendar.year_start_day(end)
    dates = pd.DatetimeIndex(inputs.calendar.dates)
    rec = np.flatnonzero((dates >= f'{start}-01-01') & (dates <= f'{end}-01-01') & (dates.day == 1))
    if len(rec) != 361:
        raise ValueError('Expected 361 monthly endpoints')
    history = history_to(inputs, params, start)
    replay = Simulator(scenarios['current']).run(params, history.y, history.acc, first, last, record_days=rec)
    np.testing.assert_allclose(np.diff(replay.acc_records, axis=0), standard['onsets'][0], rtol=1e-9, atol=1e-6)
    control = dict(candidate_id='zero_control', efficacy=0., infectiousness_multiplier=1., mean_years=3.)
    total_arms = 0
    for profile in [control, *profiles]:
        pid = profile['candidate_id']
        folder = dest / 'profiles' / pid
        completed = folder / 'projection_metadata.json'
        if valid_metadata(completed, sources, folder):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'raw').mkdir(exist_ok=True)
        identity = folder / 'projection_sources.json'
        if identity.exists() and json.loads(identity.read_text()) != sources:
            raise ValueError('Partial projection sources differ; choose a new output directory')
        dump_json(identity, sources)
        arms, validation, paths = [], [], []
        for policy in policies.to_dict('records'):
            arm = arm_id(pid, policy)
            path, record_path = folder / 'raw' / (arm + '.npz'), folder / 'raw' / (arm + '.json')
            scenario = scenarios[policy['matched_standard_strategy']]
            if path.exists() and record_path.exists():
                with np.load(path, allow_pickle=False) as z:
                    raw, doses, nasal = z['onsets'], z['annual_doses'], z['nasal_doses']
            else:
                simulator = BPZE1Simulator(scenario)
                if policy['route'] == 'coadmin':
                    mask = prior.target_mask(simulator, inputs, age, start, end)
                    nasal_events = None
                else:
                    mask = np.zeros(simulator.tables['ev_day'].size, dtype=bool)
                    nasal_events = np.asarray([(*_target_event(inputs, age, y), policy['nasal_coverage'])
                                               for y in range(start, end)], dtype=float)
                result = simulator.run(params, history.y, history.acc, first, last,
                    record_days=rec, efficacy=float(profile['efficacy']),
                    infectiousness_multiplier=float(profile['infectiousness_multiplier']),
                    mean_years=float(profile['mean_years']), start_day=first,
                    event_mask=mask, nasal_events=nasal_events)
                raw = np.diff(result.acc_records, axis=0)
                doses = prior.annual_doses(inputs, scenario, result, effective, start, end)
                nasal = result.nasal_receipt_ledger[start-inputs.calendar.first_year:end-inputs.calendar.first_year]
                if any(not np.isfinite(x).all() or (x < 0).any() for x in (raw, doses, nasal)):
                    raise ValueError('Invalid trajectory or dose ledger: ' + arm)
                if policy['route'] == 'coadmin':
                    np.testing.assert_allclose(nasal, doses[1], rtol=1e-10, atol=1e-6)
                save_arrays(path, onsets=raw, annual_doses=doses, nasal_doses=nasal)
                dump_json(record_path, {'strategy_id': arm})
            index = standard_ids.index(policy['matched_standard_strategy'])
            expected_raw = standard['onsets'][index]
            expected_doses = standard['annual_doses'][index]
            if (raw.shape != expected_raw.shape or doses.shape != expected_doses.shape
                    or nasal.shape != (30,) or not np.isfinite(raw).all()
                    or not np.isfinite(doses).all() or not np.isfinite(nasal).all()
                    or (raw < 0).any() or (doses < 0).any() or (nasal < 0).any()):
                raise ValueError('Invalid trajectory shape or values: ' + arm)
            index = standard_ids.index(policy['matched_standard_strategy'])
            if pid == 'zero_control':
                np.testing.assert_allclose(raw, standard['onsets'][index], rtol=1e-8, atol=1e-5)
                np.testing.assert_allclose(doses, standard['annual_doses'][index], rtol=1e-9, atol=1e-6)
                validation.append({'country': country, **policy,
                    'monthly_onset_max_absolute_difference': float(np.abs(raw-standard['onsets'][index]).max()),
                    'annual_dose_max_absolute_difference': float(np.abs(doses-standard['annual_doses'][index]).max())})
            arms.append({'country': country, 'strategy_id': arm, **profile_tags(profile), **policy,
                         'trajectory_path': str(path.relative_to(folder))})
            paths.extend([path, record_path])
        pd.DataFrame(arms).to_csv(folder / 'product_catalog.csv', index=False)
        paths.append(folder / 'product_catalog.csv')
        if validation:
            pd.DataFrame(validation).to_csv(folder / 'zero_effect_validation.csv', index=False)
            paths.append(folder / 'zero_effect_validation.csv')
        dump_json(completed, {'status': 'completed_projections', 'country': country, 'profile_id': pid,
            'new_product_strategies': len(arms), 'zero_effect_replays': len(validation),
        'source_files': list(sources), 'output_files': [str(p.relative_to(folder)) for p in paths]})
        total_arms += len(arms)
        print(f'{country}: {pid}, {len(arms)} route/coverage trajectories complete', flush=True)
    return {'country': country, 'status': 'completed_projections', 'new_product_strategies': total_arms,
            'seconds': time.perf_counter()-started}


def evaluate_profile(country, config_path, output_dir, profile):
    from src_python.pertussis.extensions.mucosal_economics import (
        nasal_price_frontier, nasal_price_limits, nasal_comparator_ceilings, nhb_at_nasal_price)

    config = load_design(config_path)
    pid = profile['candidate_id']
    dest = Path(output_dir) / country / 'profiles' / pid
    projection_path = dest / 'projection_metadata.json'
    projection = json.loads(projection_path.read_text())
    prior_metadata = BASE / 'current_preserving_economics_20261103/metadata.json'
    source = adopted.OUT / country
    standard_cache = BASE / 'current_preserving_economics_20261103' / f'{country}_components.npz'
    costs, severity = economic.load_costs(economic.COST), economic.load_severity_priors(economic.SEV)
    life_path = ROOT / costs['global']['life_table']['csv']
    references, reference_paths = threshold_points()
    references = references.loc[references.country.eq(country) & references.threshold_family.eq('sourced_DALY_estimate')]
    paths = [*MODULES, Path(config_path), projection_path, source / 'complete.json', prior_metadata,
             standard_cache, economic.COST, economic.SEV, life_path, *reference_paths,
             ROOT / 'scripts/optimize_current_preserving_strategies_20261103.py']
    sources = {str(p.relative_to(ROOT)): True for p in sorted(set(paths))}
    if valid_metadata(dest / 'economic_metadata.json', sources, dest):
        return
    info = json.loads((source / 'complete.json').read_text())
    products = pd.read_csv(dest / 'product_catalog.csv')
    with np.load(source / 'projection_arrays.npz', allow_pickle=False) as z:
        standard_ids = z['strategy_ids'].astype(str).tolist()
        starts, ends, raw, doses = z['month_starts'], z['month_ends'], list(z['onsets']), list(z['annual_doses'])
    arms, nasal_doses = list(standard_ids), [np.zeros(30) for _ in standard_ids]
    for row in products.itertuples():
        with np.load(dest / row.trajectory_path, allow_pickle=False) as z:
            raw.append(z['onsets']); doses.append(z['annual_doses']); nasal_doses.append(z['nasal_doses'])
        arms.append(row.strategy_id)
    try:
        raw, doses, nasal_doses = np.stack(raw), np.stack(doses), np.stack(nasal_doses)
    except ValueError as exc:
        raise ValueError('trajectory shape mismatch before economic evaluation') from exc
    summary, age, annual, _, cache, clinical_meta = economic.evaluate_arrays(
        country, arms, raw, doses, starts, ends, info, severity, costs, pd.read_csv(life_path), n=1000)
    with np.load(standard_cache, allow_pickle=False) as original:
        np.testing.assert_array_equal(original['strategy_ids'], np.asarray(standard_ids))
        for key, value in cache.items():
            standard_part = value[:len(standard_ids)] if value.ndim == 1 else value[:, :len(standard_ids)]
            np.testing.assert_allclose(standard_part, original[key], rtol=1e-9, atol=1e-5,
                                       err_msg='Standard paired reconstruction: ' + key)
    save_arrays(dest / 'raw/paired_components.npz', strategy_ids=np.asarray(arms), **cache)
    labels = products.drop(columns=['country', 'trajectory_path'])
    summary.rename(columns={'incremental_cost': 'incremental_cost_before_nasal_price'}).merge(
        labels, on='strategy_id', how='left').to_csv(dest / 'health_and_base_costs.csv', index=False)
    age.to_csv(dest / 'age_outcomes.csv.gz', index=False)
    annual.to_csv(dest / 'annual_cases.csv.gz', index=False)
    receipts = []
    for i, arm in enumerate(arms):
        for y in range(30):
            receipts.append({'country': country, 'profile_id': pid, 'strategy_id': arm, 'year': 2027+y,
                'maternal_doses': doses[i, 0, y], 'tdap_doses': doses[i, 1, y], 'nasal_doses': nasal_doses[i, y],
                'infections': raw[i, y*12:(y+1)*12, :, 2].sum()})
    pd.DataFrame(receipts).to_csv(dest / 'annual_receipts_and_infections.csv', index=False)
    tables = {key: [] for key in ('price_frontiers', 'strategy_price_intervals', 'comparator_ceilings',
                                  'zero_price_decisions', 'zero_price_strategy_uncertainty',
                                  'highest_reference_summary', 'route_price_summary')}
    matching = dict(zip(products.strategy_id, products.matched_standard_strategy))
    product_routes = dict(zip(products.strategy_id, products.route))
    indices = {a: i for i, a in enumerate(arms)}
    for horizon in (20, 30):
        dd = nasal_doses[:, :horizon] @ (1.03**-np.arange(horizon))
        base_cost = cache[f'cost_{horizon}']
        for mortality in ('included_base', 'borrowed_noninfant'):
            for burden in economic.BURDENS:
                health = effect_for(cache, horizon, burden, mortality)
                for ref in references.to_dict('records'):
                    lam = float(ref['threshold_intl2024'])
                    tags = {'country': country, **profile_tags(profile), 'horizon': horizon, 'burden': burden,
                        'mortality': mortality, 'threshold_id': ref['threshold_id'], 'threshold_intl2024': lam}
                    fronts = nasal_price_frontier(arms, health, base_cost, dd, lam)
                    caps = nasal_comparator_ceilings(arms, health, base_cost, dd, lam, matching)
                    limits = nasal_price_limits(arms, health, base_cost, dd, lam)
                    for name, records in [('price_frontiers', fronts), ('strategy_price_intervals', limits),
                                          ('comparator_ceilings', caps)]:
                        tables[name].extend({**tags, **row, 'route': product_routes.get(row.get('strategy_id'), '')}
                                            for row in records)
                    nhb = nhb_at_nasal_price(health, base_cost, dd, lam, 0.)
                    means = nhb.mean(axis=0)
                    tolerance = 128*np.finfo(float).eps*np.maximum(1., np.abs(nhb).max(axis=1))
                    optimal = nhb.max(axis=1)[:, None]-nhb <= tolerance[:, None]
                    frequency = (optimal/optimal.sum(axis=1)[:, None]).mean(axis=0)
                    np.testing.assert_allclose(frequency.sum(), 1., atol=1e-12)
                    for i, arm in enumerate(arms):
                        lo, hi = np.quantile(nhb[:, i], [.025, .975])
                        tables['zero_price_strategy_uncertainty'].append({**tags, 'strategy_id': arm,
                            'route': product_routes.get(arm, 'standard'), 'nasal_price': 0., 'mean_nhb': float(means[i]),
                            'nhb_q025': float(lo), 'nhb_q975': float(hi),
                            'conditional_optimality_frequency': float(frequency[i])})
                    greatest = max(row['ceiling_vs_best_standard'] for row in caps)
                    winning_caps = [row for row in caps if np.isclose(row['ceiling_vs_best_standard'], greatest,
                                                                     rtol=1e-10, atol=1e-9)]
                    winners = fronts[0]['ties_at_lower'].split('|')
                    decision = {**tags, 'nasal_price': 0., 'optimal_strategies': '|'.join(winners),
                        'maximum_expected_nhb': float(means.max()),
                        'nasal_in_expected_optimal_set': any(a in matching for a in winners),
                        'maximum_nasal_price_vs_best_standard': greatest,
                        'strategies_at_maximum_nasal_price': '|'.join(row['strategy_id'] for row in winning_caps)}
                    tables['zero_price_decisions'].append(decision)
                    if lam == references.threshold_intl2024.max():
                        tables['highest_reference_summary'].append(decision)
                    for route in ROUTES:
                        route_caps = [row for row in caps if product_routes[row['strategy_id']] == route]
                        best = max(row['ceiling_vs_best_standard'] for row in route_caps)
                        best_rows = [row for row in route_caps if np.isclose(row['ceiling_vs_best_standard'], best,
                                                                            rtol=1e-10, atol=1e-9)]
                        subset = standard_ids + products.loc[products.route.eq(route), 'strategy_id'].tolist()
                        j = [indices[a] for a in subset]
                        max_nhb = means[j].max()
                        route_winners = [a for a in subset if np.isclose(means[indices[a]], max_nhb, rtol=1e-10, atol=1e-9)]
                        tables['route_price_summary'].append({**tags, 'route': route,
                            'maximum_nasal_price_vs_best_standard': best,
                            'strategies_at_maximum_nasal_price': '|'.join(row['strategy_id'] for row in best_rows),
                            'zero_price_optimal_strategies_within_route': '|'.join(route_winners),
                            'maximum_expected_nhb_within_route': float(max_nhb)})
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(dest / (name + '.csv'), index=False)
    names = ['raw/paired_components.npz', 'health_and_base_costs.csv', 'age_outcomes.csv.gz',
             'annual_cases.csv.gz', 'annual_receipts_and_infections.csv', *[name + '.csv' for name in tables]]
    dump_json(dest / 'economic_metadata.json', {'status': 'completed_paired_policy_evaluation',
        'country': country, 'profile_id': pid, 'standard_strategies': len(standard_ids),
        'nasal_strategies': len(products), 'clinical_cost_draws': 1000, 'routes_compete_jointly': list(ROUTES),
        'profile_uncertainty': 'separate clinical/bridge/duration profiles; draws within profile are clinical/cost uncertainty',
        'clinical_calculation': clinical_meta, 'standard_components_reconstructed': len(cache),
        'source_files': list(sources), 'output_files': names})
    print(f'{country}: {pid}, paired policy evaluation complete', flush=True)


def evaluate_country(country, config_path, output_dir, profile_ids=None):
    profiles = selected_profiles(load_design(config_path), profile_ids)
    for profile in profiles:
        evaluate_profile(country, config_path, output_dir, profile)
    return {'country': country, 'status': 'completed_economics', 'profiles': len(profiles)}


def collect_outputs(output_dir, countries, profile_ids, config_path, stage):
    root = Path(output_dir)
    paths, block_paths = [], []
    names = ['product_catalog']
    if stage in ('all', 'economics'):
        names += ['health_and_base_costs', 'comparator_ceilings', 'price_frontiers', 'strategy_price_intervals',
                  'zero_price_decisions', 'highest_reference_summary', 'zero_price_strategy_uncertainty', 'route_price_summary']
    folders = [root / c / 'profiles' / p for c in countries for p in profile_ids]
    for folder in folders:
        block_paths.append(folder / 'projection_metadata.json')
        if stage in ('all', 'economics'):
            block_paths.append(folder / 'economic_metadata.json')
    for name in names:
        frame = pd.concat([pd.read_csv(folder / (name + '.csv')).assign(
            profile_id=folder.name) for folder in folders], ignore_index=True)
        suffix = '.csv.gz' if name in ('comparator_ceilings', 'price_frontiers', 'strategy_price_intervals',
                                      'zero_price_strategy_uncertainty') else '.csv'
        path = root / (name + suffix)
        frame.to_csv(path, index=False)
        paths.append(path)
    controls = [root / c / 'profiles/zero_control' for c in countries]
    pd.concat([pd.read_csv(folder / 'zero_effect_validation.csv') for folder in controls], ignore_index=True).to_csv(
        root / 'zero_effect_validation.csv', index=False)
    paths.append(root / 'zero_effect_validation.csv')
    block_paths.extend(folder / 'projection_metadata.json' for folder in controls)
    paths.extend(block_paths)
    dump_json(root / 'metadata.json', {'status': 'completed_' + stage, 'countries': list(countries),
        'profile_ids': list(profile_ids), 'blocks': [str(p.relative_to(root)) for p in block_paths],
        'source_files': [str(p.relative_to(ROOT)) for p in [Path(config_path), *MODULES]],
        'output_files': [str(p.relative_to(root)) for p in paths]})


def check_run(output_dir):
    root = Path(output_dir)
    record = json.loads((root / 'metadata.json').read_text())
    for name in record['blocks']:
        path = root / name
        if not path.is_file():
            raise ValueError('Missing block metadata: ' + str(path))
    print(json.dumps({'status': 'passed', 'countries': record['countries'], 'profiles': record['profile_ids']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', '--out', dest='output_dir', type=Path, default=OUT)
    parser.add_argument('--countries', nargs='+', choices=adopted.COUNTRIES, default=list(adopted.COUNTRIES))
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--workers', '--jobs', dest='workers', type=int, default=2)
    parser.add_argument('--stage', choices=('all', 'projections', 'economics', 'check'), default='all')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    args.config, args.output_dir = args.config.resolve(), args.output_dir.resolve()
    if not 1 <= args.workers <= 4 or len(set(args.countries)) != len(args.countries):
        raise ValueError('Use 1-4 workers and unique countries')
    config = load_design(args.config)
    profiles = selected_profiles(config, args.profiles)
    ids = [p['candidate_id'] for p in profiles]
    if args.check or args.stage == 'check':
        check_run(args.output_dir)
        return
    if args.plan:
        plan = []
        for country in args.countries:
            catalog = pd.read_csv(adopted.OUT / country / 'strategy_catalog.csv')
            count = len(policy_catalog(catalog))
            plan.append({'country': country, 'standard_strategies': len(catalog), 'routes': list(ROUTES),
                         'route_coverage_pairs': count, 'profiles': len(ids),
                         'positive_trajectories': count*len(ids), 'zero_controls': count})
        print(json.dumps(plan, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from scripts.run_current_paper_workflow_20261105 import workspace_lock
    with workspace_lock():
        for stage, worker in [('projections', run_country), ('economics', evaluate_country)]:
            if args.stage in ('all', stage):
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(worker, c, str(args.config), str(args.output_dir), ids) for c in args.countries]
                    for future in as_completed(futures):
                        print(json.dumps(future.result()), flush=True)
        collect_outputs(args.output_dir, args.countries, ids, args.config, args.stage)
    check_run(args.output_dir)


if __name__ == '__main__':
    main()
