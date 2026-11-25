"""Project and value an intranasal add-on to the adopted Tdap programmes.

Each efficacy/duration profile is compared with the complete standard-product
strategy grid. Nasal doses have their own receipt ledger and delivered price.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts import project_current_preserving_checked_20261105 as checked
from scripts import run_current_preserving_strategies_20261103 as adopted
from scripts import evaluate_current_preserving_strategies_20261103 as economic
from scripts.optimize_current_preserving_strategies_20261103 import threshold_points, effect_for
from src_python.pertussis.config import REPO_ROOT as ROOT
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.scenarios.current_preserving import coverage_grid
from src_python.pertussis.scenarios.fitted import build_fitted_inputs, fit_parameters, history_to, load_fit
from src_python.pertussis.scenarios.programme_scenarios import _target_event
from src_python.pertussis.states import TDAP_BOOSTER

CONFIG = ROOT / 'config/extensions/mucosal_vaccine_20261107.json'
BASE = ROOT / 'outputs/core2026'
OUT = BASE / '02_sensitivity/mucosal_vaccines/extension_20261107'
MODULES = [Path(__file__).resolve(), ROOT / 'src_python/pertussis/extensions/mucosal.py',
           ROOT / 'src_python/pertussis/extensions/mucosal_economics.py']


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_design(path):
    config = json.loads(Path(path).read_text())
    timing, product = config['time'], config['mucosal']
    if (timing['start_year'], timing['end_year_exclusive'], timing['horizons'], timing['discount_rate']) != (2027, 2057, [20, 30], .03):
        raise ValueError('This extension uses the adopted 2027-2056 and 3% evaluation calendar')
    efficacy = product['extra_infection_hazard_reduction']
    duration = product['waning']['mean_years']
    if (len(set(efficacy)) != len(efficacy) or 0 not in efficacy
            or any(not np.isfinite(e) or not 0 <= e <= 1 for e in efficacy)
            or len(set(duration)) != len(duration)
            or any(not np.isfinite(d) or d <= 0 for d in duration)):
        raise ValueError('Invalid product profiles')
    if config['economics']['n_draws'] != 1000:
        raise ValueError('Use the adopted 1000 paired clinical/cost draws')
    administration = product['administration']
    settings = (product['mode'], product['waning']['distribution'], product['waning']['shape'],
        product['conditional_symptom_modifier'], product['breakthrough_infectiousness_modifier'],
        administration['coverage_fraction_of_actual_tdap_recipients'], administration['nasal_doses_per_receipt'],
        config['economics']['clinical_cost_seed'])
    if settings != ('add_on_to_tdap', 'erlang', 2, 1., 1., 1., 1, 20261101):
        raise ValueError('Configuration differs from the implemented coadministration mechanism or paired sampler')
    countries = [c['country'] for c in config['countries']]
    if len(set(countries)) != len(countries) or set(countries) != set(adopted.COUNTRIES):
        raise ValueError('Expected the eight adopted settings')
    return config


def profile_id(efficacy, duration):
    return f'e{100*efficacy:g}_d{duration:02g}'


def profiles(config):
    return [(float(e), float(d)) for e in config['mucosal']['extra_infection_hazard_reduction'] if e > 0
            for d in config['mucosal']['waning']['mean_years']]


def projection_sources(country, config_path, old_folder, old_info):
    paths = [*MODULES, Path(config_path), adopted.OUT / country / 'complete.json',
             adopted.OUT / country / 'projection_arrays.npz', adopted.OUT / country / 'strategy_catalog.csv',
             old_folder / 'complete.json']
    paths.extend(ROOT / name for name in old_info.get('source_files', []))
    fit = ROOT / old_info['fit_directory']
    paths.extend(fit / name for name in ('run_spec.json', 'fit_summary.csv', 'complete.json'))
    return sorted({str(p.relative_to(ROOT)) for p in paths})


def target_mask(simulator, inputs, age, first_year, last_year):
    table = simulator.tables
    selected = {_target_event(inputs, age, year) for year in range(first_year, last_year)}
    mask = np.array([kind == TDAP_BOOSTER and (int(day), int(bin_index)) in selected
                     for day, bin_index, kind in zip(table['ev_day'], table['ev_bin'], table['ev_kind'])])
    return mask


def annual_doses(inputs, scenario, result, effective, start, end):
    doses = np.empty((2, end-start))
    for i, year in enumerate(range(start, end)):
        a, b = inputs.calendar.year_start_day(year), inputs.calendar.year_start_day(year+1)
        doses[0, i] = np.sum(scenario.maternal_effective[a:b] / effective * inputs.births_per_day[a:b])
        doses[1, i] = result.receipt_ledger[year-inputs.calendar.first_year, TDAP_BOOSTER, 0]
    return doses


def run_country(country, config_path, output_dir):
    from src_python.pertussis.extensions.mucosal import MucosalSimulator

    config = load_design(config_path)
    country_design = next(row for row in config['countries'] if row['country'] == country)
    dest = Path(output_dir) / country
    checked.configure()
    old_folder, old_info, _ = adopted.source_for(country)
    sources = projection_sources(country, config_path, old_folder, old_info)
    completed = dest / 'projection_metadata.json'
    if completed.exists():
        meta = json.loads(completed.read_text())
        return {'country': country, 'status': 'reused', 'profiles': len(profiles(config)), 'arms': meta['new_product_strategies']}
    dest.mkdir(parents=True, exist_ok=True)
    raw_dir = dest / 'raw'
    raw_dir.mkdir(exist_ok=True)
    identity = dest / 'projection_sources.json'
    if identity.exists() and not json.loads(identity.read_text()):
        raise ValueError('Partial projection source record is empty')
    dump_json(identity, sources)
    started = time.perf_counter()
    fit_dir = ROOT / old_info['fit_directory']
    spec, fit = load_fit(fit_dir)
    start, end = config['time']['start_year'], config['time']['end_year_exclusive']
    inputs = build_fitted_inputs(spec, end)
    params = fit_parameters(spec, fit.to_dict())
    age = float(country_design['adolescent_age'])
    effective = float(old_info.get('maternal_effective_share', .9))
    scenarios, catalog, _ = coverage_grid(inputs, start_year=start, end_year=end,
        adolescent_age=age, maternal_effective_share=effective)
    selected = country_design['source_strategy_ids']
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(scenarios):
        raise ValueError('Invalid matched standard strategy selection')
    if not catalog.set_index('strategy_id').loc[selected].adolescent_coverage.gt(0).all():
        raise ValueError('Nasal coadministration requires a positive Tdap coverage')
    with np.load(adopted.OUT / country / 'projection_arrays.npz', allow_pickle=False) as archive:
        standard = {key: archive[key] for key in archive.files}
    standard_ids = standard['strategy_ids'].astype(str).tolist()
    if standard_ids != list(scenarios):
        raise ValueError('Rebuilt standard strategy catalog differs from the adopted catalog')
    first, last = inputs.calendar.year_start_day(start), inputs.calendar.year_start_day(end)
    dates = pd.DatetimeIndex(inputs.calendar.dates)
    rec = np.flatnonzero((dates >= f'{start}-01-01') & (dates <= f'{end}-01-01') & (dates.day == 1))
    if len(rec) != 361:
        raise ValueError('Expected 361 monthly endpoints')
    history = history_to(inputs, params, start)
    replay = Simulator(scenarios['current']).run(params, history.y, history.acc, first, last, record_days=rec)
    np.testing.assert_allclose(np.diff(replay.acc_records, axis=0), standard['onsets'][0], rtol=1e-9, atol=1e-6)
    validations, output_paths, arms = [], [], []
    for sid in selected:
        scenario = scenarios[sid]
        simulator = MucosalSimulator(scenario)
        mask = target_mask(simulator, inputs, age, start, end)
        for e, duration in [(0., 3.), *profiles(config)]:
            pid = profile_id(e, duration)
            arm = f'nasal_{pid}__{sid}'
            path = raw_dir / (arm + '.npz')
            record_path = raw_dir / (arm + '.json')
            if path.exists() and record_path.exists():
                record = json.loads(record_path.read_text())
                with np.load(path, allow_pickle=False) as z:
                    raw, doses, nasal = z['onsets'], z['annual_doses'], z['nasal_doses']
            else:
                result = simulator.run(params, history.y, history.acc, first, last,
                    record_days=rec, efficacy=e, mean_years=duration, start_day=first, event_mask=mask)
                raw = np.diff(result.acc_records, axis=0)
                doses = annual_doses(inputs, scenario, result, effective, start, end)
                nasal = result.nasal_receipt_ledger[start-inputs.calendar.first_year:end-inputs.calendar.first_year]
                if (not np.isfinite(raw).all() or (raw < 0).any()
                        or not np.isfinite(doses).all() or (doses < 0).any()
                        or not np.isfinite(nasal).all() or (nasal < 0).any()):
                    raise ValueError('Invalid product trajectory or dose ledger: ' + arm)
                np.testing.assert_allclose(nasal, doses[1], rtol=1e-10, atol=1e-6)
                save_arrays(path, onsets=raw, annual_doses=doses, nasal_doses=nasal)
                dump_json(record_path, {'strategy_id': arm, 'trajectory_path': str(path.relative_to(dest))})
            index = standard_ids.index(sid)
            if e == 0:
                np.testing.assert_allclose(raw, standard['onsets'][index], rtol=1e-8, atol=1e-5)
                np.testing.assert_allclose(doses, standard['annual_doses'][index], rtol=1e-9, atol=1e-6)
                validations.append({'country': country, 'matched_strategy': sid,
                    'monthly_onset_max_absolute_difference': float(np.abs(raw-standard['onsets'][index]).max()),
                    'annual_dose_max_absolute_difference': float(np.abs(doses-standard['annual_doses'][index]).max()),
                    'nasal_equals_actual_tdap_doses': True})
            else:
                coverage = catalog.set_index('strategy_id').loc[sid]
                arms.append({'country': country, 'strategy_id': arm, 'profile_id': pid,
                    'extra_infection_hazard_reduction': e, 'mean_protection_years': duration,
                    'matched_standard_strategy': sid, 'maternal_coverage': float(coverage.maternal_coverage),
                    'adolescent_coverage': float(coverage.adolescent_coverage), 'adolescent_age': age,
                    'trajectory_path': str(path.relative_to(dest))})
            output_paths.extend([path, record_path])
        print(f'{country}: completed nasal profiles for {sid}', flush=True)
    pd.DataFrame(arms).to_csv(dest / 'product_catalog.csv', index=False)
    pd.DataFrame(validations).to_csv(dest / 'zero_effect_validation.csv', index=False)
    output_paths.extend([dest / 'product_catalog.csv', dest / 'zero_effect_validation.csv'])
    meta = {'status': 'completed_product_projections', 'country': country, 'start_year': start,
        'end_year_exclusive': end, 'new_product_strategies': len(arms), 'zero_effect_replays': len(validations),
        'profiles': len(profiles(config)), 'seconds': time.perf_counter()-started,
        'source_files': sources, 'output_files': [str(p.relative_to(dest)) for p in output_paths]}
    dump_json(completed, meta)
    return {key: meta[key] for key in ('country', 'status', 'new_product_strategies', 'seconds')}


def evaluate_country(country, config_path, output_dir):
    from src_python.pertussis.extensions.mucosal_economics import (
        nasal_price_frontier, nasal_price_limits, nasal_comparator_ceilings, nhb_at_nasal_price)

    config = load_design(config_path)
    dest = Path(output_dir) / country
    projection_path = dest / 'projection_metadata.json'
    projection = json.loads(projection_path.read_text())
    prior_metadata = BASE / 'current_preserving_economics_20261103/metadata.json'
    if not prior_metadata.is_file():
        raise ValueError('Adopted economic metadata is missing')
    source = adopted.OUT / country
    info = json.loads((source / 'complete.json').read_text())
    products = pd.read_csv(dest / 'product_catalog.csv')
    with np.load(source / 'projection_arrays.npz', allow_pickle=False) as z:
        standard_ids = z['strategy_ids'].astype(str).tolist()
        starts, ends = z['month_starts'], z['month_ends']
        raw = list(z['onsets'])
        doses = list(z['annual_doses'])
    arms = list(standard_ids)
    nasal_doses = [np.zeros(30) for _ in standard_ids]
    for row in products.itertuples():
        with np.load(dest / row.trajectory_path, allow_pickle=False) as z:
            raw.append(z['onsets'])
            doses.append(z['annual_doses'])
            nasal_doses.append(z['nasal_doses'])
        arms.append(row.strategy_id)
    raw, doses, nasal_doses = np.stack(raw), np.stack(doses), np.stack(nasal_doses)
    costs = economic.load_costs(economic.COST)
    severity = economic.load_severity_priors(economic.SEV)
    life_path = ROOT / costs['global']['life_table']['csv']
    life = pd.read_csv(life_path)
    summary, age, annual, _, cache, clinical_meta = economic.evaluate_arrays(
        country, arms, raw, doses, starts, ends, info, severity, costs, life,
        n=config['economics']['n_draws'])
    standard_cache = BASE / 'current_preserving_economics_20261103' / f'{country}_components.npz'
    with np.load(standard_cache, allow_pickle=False) as original:
        np.testing.assert_array_equal(original['strategy_ids'], np.asarray(standard_ids))
        for key, value in cache.items():
            standard_part = value[:len(standard_ids)] if value.ndim == 1 else value[:, :len(standard_ids)]
            np.testing.assert_allclose(standard_part, original[key], rtol=1e-9, atol=1e-5,
                                       err_msg='Standard economic reconstruction: ' + key)
    save_arrays(dest / 'raw/paired_components.npz', strategy_ids=np.asarray(arms), **cache)
    labels = products.drop(columns=['country', 'trajectory_path'])
    summary = summary.rename(columns={'incremental_cost': 'incremental_cost_before_nasal_price'})
    summary.merge(labels, on='strategy_id', how='left').to_csv(dest / 'health_and_base_costs.csv', index=False)
    age.to_csv(dest / 'age_outcomes.csv.gz', index=False)
    annual.to_csv(dest / 'annual_cases.csv.gz', index=False)
    receipt_rows = []
    for i, arm in enumerate(arms):
        for y in range(30):
            receipt_rows.append({'country': country, 'strategy_id': arm, 'year': 2027+y,
                'maternal_doses': doses[i, 0, y], 'tdap_doses': doses[i, 1, y],
                'nasal_doses': nasal_doses[i, y], 'infections': raw[i, y*12:(y+1)*12, :, 2].sum()})
    pd.DataFrame(receipt_rows).to_csv(dest / 'annual_receipts_and_infections.csv', index=False)
    references, reference_paths = threshold_points()
    references = references.loc[references.country.eq(country) & references.threshold_family.eq('sourced_DALY_estimate')]
    tables = {key: [] for key in ('price_frontiers', 'strategy_price_intervals', 'comparator_ceilings',
                                  'zero_price_decisions', 'zero_price_strategy_uncertainty',
                                  'highest_reference_summary')}
    for pid, product_group in products.groupby('profile_id', sort=False):
        subset = standard_ids + product_group.strategy_id.tolist()
        indices = [arms.index(arm) for arm in subset]
        matching = dict(zip(product_group.strategy_id, product_group.matched_standard_strategy))
        profile = product_group.iloc[0]
        for horizon in (20, 30):
            dd = nasal_doses[indices, :horizon] @ ((1.03)**-np.arange(horizon))
            base_cost = cache[f'cost_{horizon}'][:, indices]
            for mortality, burden in [(m, b) for m in ('included_base', 'borrowed_noninfant') for b in economic.BURDENS]:
                health = effect_for(cache, horizon, burden, mortality)[:, indices]
                for ref in references.to_dict('records'):
                    lam = float(ref['threshold_intl2024'])
                    tags = {'country': country, 'profile_id': pid, 'horizon': horizon, 'burden': burden,
                        'mortality': mortality, 'threshold_id': ref['threshold_id'],
                        'threshold_intl2024': lam,
                        'extra_infection_hazard_reduction': float(profile.extra_infection_hazard_reduction),
                        'mean_protection_years': float(profile.mean_protection_years)}
                    fronts = nasal_price_frontier(subset, health, base_cost, dd, lam)
                    limits = nasal_price_limits(subset, health, base_cost, dd, lam)
                    caps = nasal_comparator_ceilings(subset, health, base_cost, dd, lam, matching)
                    for name, records in [('price_frontiers', fronts), ('strategy_price_intervals', limits),
                                          ('comparator_ceilings', caps)]:
                        tables[name].extend({**tags, **row} for row in records)
                    nhb = nhb_at_nasal_price(health, base_cost, dd, lam, 0.)
                    means = nhb.mean(axis=0)
                    winners = fronts[0]['ties_at_lower'].split('|')
                    draw_tolerance = 128*np.finfo(float).eps*np.maximum(1., np.abs(nhb).max(axis=1))
                    draw_optimal = nhb.max(axis=1)[:, None]-nhb <= draw_tolerance[:, None]
                    frequency = (draw_optimal/draw_optimal.sum(axis=1)[:, None]).mean(axis=0)
                    np.testing.assert_allclose(frequency.sum(), 1., atol=1e-12)
                    for i, arm in enumerate(subset):
                        lo, hi = np.quantile(nhb[:, i], [.025, .975])
                        tables['zero_price_strategy_uncertainty'].append({**tags, 'strategy_id': arm,
                            'nasal_price': 0., 'mean_nhb': float(means[i]), 'nhb_q025': float(lo),
                            'nhb_q975': float(hi), 'conditional_optimality_frequency': float(frequency[i])})
                    greatest = max(row['ceiling_vs_best_standard'] for row in caps)
                    winning_caps = [row for row in caps if np.isclose(row['ceiling_vs_best_standard'], greatest, rtol=1e-10, atol=1e-9)]
                    decision = {**tags, 'nasal_price': 0., 'optimal_strategies': '|'.join(winners),
                        'maximum_expected_nhb': float(means.max()),
                        'nasal_in_expected_optimal_set': any(arm in matching for arm in winners),
                        'maximum_nasal_price_vs_best_standard': greatest,
                        'strategies_at_maximum_nasal_price': '|'.join(row['strategy_id'] for row in winning_caps)}
                    tables['zero_price_decisions'].append(decision)
                    if lam == references.threshold_intl2024.max():
                        tables['highest_reference_summary'].append(decision)
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(dest / (name + '.csv'), index=False)
    filenames = ['raw/paired_components.npz', 'health_and_base_costs.csv', 'age_outcomes.csv.gz',
                 'annual_cases.csv.gz', 'annual_receipts_and_infections.csv',
                 *[name + '.csv' for name in tables]]
    dump_json(dest / 'economic_metadata.json', {'status': 'completed_paired_nasal_price_frontiers',
        'country': country, 'standard_strategies': len(standard_ids), 'nasal_strategies': len(products),
        'clinical_cost_draws': 1000, 'profiles_evaluated_separately': len(products.profile_id.unique()),
        'burdens': list(economic.BURDENS), 'mortality_scenarios': ['included_base', 'borrowed_noninfant'],
        'optimality_frequency': 'paired clinical/cost draws; equal weight among tied winners; zero additional nasal price',
        'clinical_calculation': clinical_meta, 'standard_components_reconstructed': len(cache),
        'output_files': filenames})
    print(f'{country}: paired health outcomes and product price frontiers complete', flush=True)


def check_run(output_dir):
    root = Path(output_dir)
    metadata = json.loads((root / 'metadata.json').read_text())
    for country in metadata['countries']:
        folder = root / country
        record = json.loads((folder / 'projection_metadata.json').read_text())
        for name in record.get('output_files', []):
            if not (folder / name).is_file():
                raise ValueError('Missing projection output: ' + name)
        economic_path = folder / 'economic_metadata.json'
        if metadata['status'] in ('completed_all', 'completed_economics'):
            record = json.loads(economic_path.read_text())
            for name in record.get('output_files', []):
                if not (folder / name).is_file():
                    raise ValueError('Missing economic output: ' + name)
    print(json.dumps({'status': 'passed', 'countries': metadata['countries']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--countries', nargs='+', choices=adopted.COUNTRIES, default=list(adopted.COUNTRIES))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--stage', choices=('all', 'projections', 'economics', 'check'), default='all')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    args.config, args.output_dir = args.config.resolve(), args.output_dir.resolve()
    if not 1 <= args.workers <= 4 or len(set(args.countries)) != len(args.countries):
        raise ValueError('Use 1-4 workers and unique countries')
    config = load_design(args.config)
    if args.plan:
        plan = [{'country': c['country'], 'matched_standard_strategies': len(c['source_strategy_ids']),
                 'nasal_trajectories': len(c['source_strategy_ids'])*len(profiles(config)),
                 'zero_effect_replays': len(c['source_strategy_ids'])}
                for c in config['countries'] if c['country'] in args.countries]
        print(json.dumps(plan, indent=2))
        return
    if args.stage == 'check':
        check_run(args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from scripts.run_current_paper_workflow_20261105 import workspace_lock
    with workspace_lock():
        if args.stage in ('all', 'projections'):
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_country, c, str(args.config), str(args.output_dir)): c for c in args.countries}
                for future in as_completed(futures):
                    print(json.dumps(future.result()), flush=True)
        if args.stage in ('all', 'economics'):
            for country in args.countries:
                evaluate_country(country, args.config, args.output_dir)
        paths = [args.output_dir / c / 'projection_metadata.json' for c in args.countries]
        combined_tables = ['product_catalog', 'zero_effect_validation']
        if args.stage in ('all', 'economics'):
            paths.extend(args.output_dir / c / 'economic_metadata.json' for c in args.countries)
            combined_tables.extend(['health_and_base_costs', 'comparator_ceilings', 'price_frontiers',
                                    'strategy_price_intervals', 'zero_price_decisions', 'highest_reference_summary',
                                    'zero_price_strategy_uncertainty'])
        for name in combined_tables:
            combined = pd.concat([pd.read_csv(args.output_dir / c / (name + '.csv')) for c in args.countries], ignore_index=True)
            suffix = '.csv.gz' if name in ('comparator_ceilings', 'price_frontiers', 'strategy_price_intervals',
                                         'zero_price_strategy_uncertainty') else '.csv'
            path = args.output_dir / (name + suffix)
            combined.to_csv(path, index=False)
            paths.append(path)
        meta = {'status': 'completed_' + args.stage, 'countries': args.countries,
                'source_files': [str(p.relative_to(ROOT)) for p in [args.config, *MODULES]],
                'output_files': [str(p.relative_to(args.output_dir)) for p in paths]}
        dump_json(args.output_dir / 'metadata.json', meta)
    check_run(args.output_dir)


if __name__ == '__main__':
    main()
