"""Project selected BPZE1 mechanisms across adopted transmission fits.

This adds nasal trajectories to saved standard strategies, retaining the original
paired clinical/cost draws and England's particle-specific mortality calibration.
It also records immune-state snapshots under the adopted and full-refit histories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts import evaluate_current_preserving_strategies_20261103 as ev
from scripts import run_bpze1_policy_extension_20261107 as bp
from scripts.run_covid_refit_20261104 import build_scenario_inputs
from scripts.optimize_current_preserving_strategies_20261103 import effect_for, threshold_points
from src_python.pertussis.economics.projections import conditional_risk_arrays
from src_python.pertussis.extensions.bpze1 import BPZE1Simulator, _force_of_infection
from src_python.pertussis.model import kernel as K
from src_python.pertussis.model.parameters import ModelParameters
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.scenarios.aggregation import AGE_GROUPS
from src_python.pertussis.scenarios.current_preserving import coverage_grid
from src_python.pertussis.scenarios.fitted import build_fitted_inputs, fit_parameters, history_to, load_fit
from src_python.pertussis.scenarios.programme_scenarios import _target_event
from src_python.pertussis.states import S0, SW, SA, R, W

ROOT, BASE = bp.ROOT, bp.BASE
OUT = bp.OUT / 'transmission_followup'
LOCAL = BASE / 'current_preserving_local_uncertainty_20261103'
LOCAL_ECON = BASE / 'current_preserving_local_uncertainty_economics_20261103'
REFIT = BASE / '04_covid/full_refit_20261105T070745Z'
PROFILE_IDS = ('b050_a000_d03_point', 'b050_a050_d03_point', 'b050_a100_d03_point')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def jobs():
    selected = pd.read_csv(LOCAL / 'selected_rows.csv')
    assert len(selected) == 60
    assert set(selected.country) == {'England', 'Brazil'}
    assert not selected.duplicated(['country', 'draw_id']).any()
    np.testing.assert_allclose(selected.selected_particle_weight, 1 / 30, rtol=1e-12, atol=0)
    assert selected.groupby(['country', 'source_replicate']).size().tolist() == [15, 15, 15, 15]
    result = []
    for row in selected.to_dict('records'):
        result.append({**row, 'kind': 'local', 'history': 'adopted_local',
                       'label': f"local_{row['country']}_{row['draw_id']:03d}",
                       'nasal_coverage': .7, 'n_draws': 50})
    for country in ('Japan', 'Brazil'):
        for history in ('adopted', 'baseline', 'recovery_365'):
            result.append(dict(kind='history', country=country, history=history,
                label=f'history_{country}_{history}', nasal_coverage=.9, n_draws=1000,
                source_replicate=-1, draw_id=-1))
    return result


def inputs_for(job):
    country = job['country']
    if job['kind'] == 'local':
        source = ROOT / job['projection_root'] / country
        original_cache = LOCAL_ECON / f"draw_{job['draw_id']:03d}" / country / 'components.npz'
    elif job['history'] == 'adopted':
        source = bp.adopted.OUT / country
        original_cache = BASE / 'current_preserving_economics_20261103' / f'{country}_components.npz'
    else:
        source = REFIT / 'policy/projections' / job['history'] / country
        original_cache = None
    info = json.loads((source / 'complete.json').read_text())
    if job['kind'] == 'local':
        for key in ('source_parameter_row', 'source_replicate', 'draw_id', 'saved_local_index'):
            if int(info[key]) != int(job[key]):
                raise ValueError('Saved particle identity mismatch: ' + key)
        np.testing.assert_allclose(info['selected_particle_weight'], job['selected_particle_weight'], rtol=1e-12, atol=0)
    fit_dir = ROOT / info['fit_directory']
    spec, fit = load_fit(fit_dir)
    if job['kind'] == 'local':
        params = ModelParameters(**info['parameters'])
        inputs = build_fitted_inputs(spec, 2057)
    elif job['history'] == 'adopted':
        params = fit_parameters(spec, fit.to_dict())
        inputs = build_fitted_inputs(spec, 2057)
    else:
        params = fit_parameters(spec, fit.to_dict())
        inputs = build_scenario_inputs(spec, 2057, job['history'])
    paths = [source / 'complete.json', source / 'projection_arrays.npz',
             source / 'strategy_catalog.csv', *[fit_dir / n for n in
             ('run_spec.json', 'complete.json', 'fit_summary.csv')]]
    if original_cache:
        paths.append(original_cache)
    if job['kind'] == 'local' and country == 'England':
        paths.append(ROOT / info['conditional_risk_path'])
    return source, info, params, inputs, paths, original_cache


def state_records(simulator, params, result, days, dates, arm, profile):
    """Describe pre-event monthly states and instantaneous infection pressure."""
    sig, _ = params.sigma_pi()
    P = params.kernel_vector(simulator.transmission_scale(params))
    t = simulator.tables
    grid = simulator.grid
    ages = np.arange(101)
    membership = np.column_stack([(ages >= lo) & (ages < hi) for lo, hi in AGE_GROUPS.values()])
    rows = []
    for i, day in enumerate(days):
        layer = result.mucosal_snapshots[i]
        y = layer.sum(axis=0)
        start = pd.Timestamp(dates[i].year, 1, 1)
        end = pd.Timestamp(dates[i].year + 1, 1, 1)
        fractions = (dates[i] - start).days / (end - start).days
        groups = grid.completed_age_weights(fractions) @ membership
        protection = np.array([1., 1. - profile['efficacy'], 1. - profile['efficacy']])
        effective = np.einsum('lbc,c,l->b', layer, sig, protection)
        lam = np.zeros(grid.n_bins)
        ng = t['c_home'].shape[1]
        _force_of_infection(layer, 0., int(day), P, t['npi_r'], t['imp_mult'], t['season_on'],
            t['year_frac'], t['year_idx'], t['c_home'], t['c_non'], t['group'], t['contact_mult'],
            lam, np.zeros(ng), np.zeros(ng), np.zeros(ng), profile['infectiousness_multiplier'])
        quantities = {'population': y.sum(axis=1), 'never_infected_unvaccinated': y[:, S0],
            'waned_primed': y[:, SW] + y[:, SA], 'natural_immune': y[:, R],
            'waned_natural': y[:, W], 'effective_susceptible': effective,
            'instantaneous_infections_per_day': lam * effective,
            'instantaneous_natural_boosts_per_day': P[K.P_KAPPA] * lam * y[:, W]}
        aggregate = {name: values @ groups for name, values in quantities.items()}
        pressure = (lam * y.sum(axis=1)) @ groups
        for g, group in enumerate(AGE_GROUPS):
            row = {'strategy_id': arm, 'date': str(dates[i].date()), 'age_group': group,
                   **{name: value[g] for name, value in aggregate.items()}}
            row['population_weighted_force_per_day'] = pressure[g] / row['population'] if row['population'] else 0.
            rows.append(row)
    return rows


def run_job(job, output_dir):
    started = time.perf_counter()
    target = Path(output_dir) / 'blocks' / job['label']
    source, info, params, inputs, paths, original_cache = inputs_for(job)
    config = bp.load_design(bp.CONFIG)
    profiles = bp.selected_profiles(config, PROFILE_IDS)
    costs, severity = ev.load_costs(ev.COST), ev.load_severity_priors(ev.SEV)
    life_path = ROOT / costs['global']['life_table']['csv']
    paths += [Path(__file__), bp.CONFIG, ev.COST, ev.SEV, life_path,
              LOCAL / 'selected_rows.csv', REFIT / 'validated_fit_map.json',
              Path(bp.__file__), Path(ev.__file__),
              ROOT / 'scripts/run_covid_refit_20261104.py',
              ROOT / 'scripts/optimize_current_preserving_strategies_20261103.py',
              ROOT / 'scripts/run_mucosal_vaccine_extension_20261107.py']
    # All infant-risk records referenced by the paired economic kernel are bound
    # before execution, including the donor calibration used in Japan.
    if not any(e['quantity'] == 'cfr' for e in severity['countries'][job['country']].get('entries', [])):
        reference_risk, _ = ev.reference_risk_for(job['country'], '2001_2023', ev.AU)
        if reference_risk is None:
            raise ValueError('Expected an explicit adopted infant-risk source')
        paths.append(Path(reference_risk))
    for folder in ('model', 'extensions', 'scenarios', 'economics', 'inputs'):
        paths += list((ROOT / 'src_python/pertussis' / folder).glob('*.py'))
    paths += [ROOT / 'src_python/pertussis/grid.py', ROOT / 'src_python/pertussis/states.py']
    complete = target / 'metadata.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        if record['job'] != job:
            raise ValueError('Completed follow-up design changed: ' + job['label'])
        return record
    target.mkdir(parents=True, exist_ok=True)
    original = json.loads((bp.adopted.OUT / job['country'] / 'complete.json').read_text())
    effective, age = original['maternal_effective_share'], original['adolescent_age']
    scenarios, catalog, _ = coverage_grid(inputs, start_year=2027, end_year=2057,
                                        adolescent_age=age, maternal_effective_share=effective)
    current = catalog.set_index('strategy_id').loc['current']
    coverage = max(job['nasal_coverage'], current.adolescent_coverage)
    dates = pd.DatetimeIndex(inputs.calendar.dates)
    rec = np.flatnonzero((dates >= '2027-01-01') & (dates <= '2057-01-01') & (dates.day == 1))
    first, last = int(rec[0]), int(rec[-1])
    history = history_to(inputs, params, 2027)
    standard_path = source / 'projection_arrays.npz'
    if job['kind'] == 'local' and not standard_path.exists():
        standard_ids = list(scenarios)
        starts, ends = dates[rec[:-1]].to_numpy(), dates[rec[1:]].to_numpy()
        raw, doses = [], []
        for sid in standard_ids:
            scenario = scenarios[sid]
            result = Simulator(scenario).run(params, history.y, history.acc, first, last, record_days=rec)
            raw.append(np.diff(result.acc_records, axis=0))
            doses.append(bp.prior.annual_doses(inputs, scenario, result, effective, 2027, 2057))
    else:
        with np.load(standard_path, allow_pickle=False) as z:
            standard_ids = z['strategy_ids'].astype(str).tolist()
            starts, ends = z['month_starts'], z['month_ends']
            raw, doses = list(z['onsets']), list(z['annual_doses'])
    assert standard_ids == list(scenarios)
    snap = (np.flatnonzero((dates >= '2027-01-01') & (dates < '2057-01-01') & (dates.day == 15))
            if job['kind'] == 'history' else np.array([], dtype=int))
    simulator = BPZE1Simulator(scenarios['current'])
    no_effect = dict(efficacy=0., infectiousness_multiplier=1., mean_years=3.)
    replay = simulator.run(params, history.y, history.acc, first, last, record_days=rec,
                           snap_days=snap, event_mask=np.zeros(simulator.tables['ev_day'].size, dtype=bool), **no_effect)
    np.testing.assert_allclose(np.diff(replay.acc_records, axis=0), raw[0], rtol=1e-9, atol=1e-6)
    baseline_doses = bp.prior.annual_doses(inputs, scenarios['current'], replay, effective, 2027, 2057)
    np.testing.assert_allclose(baseline_doses, doses[0], rtol=1e-10, atol=1e-6)
    states = state_records(simulator, params, replay, snap, dates[snap], 'current', no_effect) if len(snap) else []
    nasal = [np.zeros(30) for _ in standard_ids]
    arms = list(standard_ids)
    events = np.asarray([(*_target_event(inputs, age, year), coverage) for year in range(2027, 2057)])
    # The first particle and each historical fit also replay active zero-effect nasal events.
    zero_error = None
    if job['kind'] == 'history' or job['draw_id'] == 0:
        zero = simulator.run(params, history.y, history.acc, first, last, record_days=rec,
                             nasal_events=events, **no_effect)
        np.testing.assert_array_equal(zero.acc_records, replay.acc_records)
        zero_error = float(np.max(np.abs(zero.acc_records - replay.acc_records)))
    for profile in profiles:
        arm = profile['candidate_id']
        result = simulator.run(params, history.y, history.acc, first, last, record_days=rec,
            snap_days=snap, nasal_events=events, efficacy=profile['efficacy'],
            infectiousness_multiplier=profile['infectiousness_multiplier'], mean_years=profile['mean_years'])
        arms.append(arm)
        raw.append(np.diff(result.acc_records, axis=0))
        doses.append(bp.prior.annual_doses(inputs, scenarios['current'], result, effective, 2027, 2057))
        nasal.append(result.nasal_receipt_ledger[2027-inputs.calendar.first_year:2057-inputs.calendar.first_year])
        if len(snap):
            states.extend(state_records(simulator, params, result, snap, dates[snap], arm, profile))
    raw, doses, nasal = np.stack(raw), np.stack(doses), np.stack(nasal)
    original_kernel = ev.sample_kernel
    if job['kind'] == 'local' and job['country'] == 'England':
        risk_path = ROOT / info['conditional_risk_path']
        if not risk_path.is_file():
            raise ValueError('Conditional mortality calibration is missing')
        risks = pd.read_csv(risk_path)
        if (not risks.source_parameter_row.eq(job['source_parameter_row']).all()
                or risks.draw_id.nunique() != 1 or risks.risk_draw_id.nunique() != 512):
            raise ValueError('Mortality calibration is not paired to this transmission particle')
        def matched_kernel(*args, **kwargs):
            result = original_kernel(*args, **kwargs)
            n = kwargs.get('n', 1000)
            pt, pm, _ = conditional_risk_arrays(risks, np.repeat(risks.draw_id.iloc[0], n),
                np.random.default_rng(20261103), job['country'], ev.AgeGrid().n_bins)
            result['infant_risks'], result['risk_path'] = (pt, pm), risk_path
            return result
        ev.sample_kernel = matched_kernel
    try:
        summary, ages, annual, _, cache, clinical = ev.evaluate_arrays(job['country'], arms, raw, doses,
            starts, ends, info, severity, costs, pd.read_csv(life_path), n=job['n_draws'])
    finally:
        ev.sample_kernel = original_kernel
    if original_cache:
        with np.load(original_cache, allow_pickle=False) as old:
            np.testing.assert_array_equal(old['strategy_ids'], np.asarray(standard_ids))
            for key, value in cache.items():
                section = value[:len(standard_ids)] if value.ndim == 1 else value[:, :len(standard_ids)]
                np.testing.assert_allclose(section, old[key], rtol=1e-9, atol=1e-5, err_msg=key)
    np.savez_compressed(target / 'components.npz', strategy_ids=np.asarray(arms), nasal_doses=nasal, **cache)
    summary.to_csv(target / 'summary.csv', index=False)
    ages.to_csv(target / 'age_effects.csv.gz', index=False)
    annual.to_csv(target / 'annual_cases.csv.gz', index=False)
    outputs = ['components.npz', 'summary.csv', 'age_effects.csv.gz', 'annual_cases.csv.gz']
    if states:
        pd.DataFrame(states).to_csv(target / 'monthly_states.csv.gz', index=False)
        outputs.append('monthly_states.csv.gz')
    result = dict(status='completed', job=job, nasal_coverage=float(coverage),
        maternal_coverage=float(current.maternal_coverage), standard_strategies=len(standard_ids),
        nasal_profiles=list(PROFILE_IDS), zero_effect_max_difference=zero_error,
        standard_current_replayed=True, paired_standard_components_reconstructed=original_cache is not None,
        clinical_calculation=clinical, elapsed_seconds=time.perf_counter()-started,
        output_files=outputs)
    write_json(complete, result)
    print(job['label'], f"completed in {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def quantiles(values, prefix):
    values = np.asarray(values)
    return {prefix + '_mean': float(values.mean()), prefix + '_q025': float(np.quantile(values, .025)),
            prefix + '_q975': float(np.quantile(values, .975))}


def summarize(output_dir):
    dest = Path(output_dir)
    references, reference_paths = threshold_points()
    records = []
    for job in jobs():
        folder = dest / 'blocks' / job['label']
        meta = json.loads((folder / 'metadata.json').read_text())
        assert meta['job'] == job
        with np.load(folder / 'components.npz', allow_pickle=False) as z:
            components = {k: z[k] for k in z.files}
        records.append((job, meta, folder, components))
    summaries, contrasts, prices, particle_means, ages, years, states = [], [], [], [], [], [], []
    groups = []
    for country in ('England', 'Brazil'):
        subset = [r for r in records if r[0]['kind'] == 'local' and r[0]['country'] == country]
        assert len(subset) == 30
        assert pd.Series([r[0]['source_replicate'] for r in subset]).value_counts().tolist() == [15, 15]
        groups.append((dict(country=country, fit_group='local_pooled', transmission_particles=30), subset))
        for replicate in (0, 1):
            groups.append((dict(country=country, fit_group=f'local_replicate_{replicate}', transmission_particles=15),
                           [r for r in subset if r[0]['source_replicate'] == replicate]))
    for record in records:
        if record[0]['kind'] == 'history':
            groups.append((dict(country=record[0]['country'], fit_group=record[0]['history'], transmission_particles=1), [record]))
        tags = {k: record[0][k] for k in ('kind', 'country', 'history', 'draw_id', 'source_replicate')}
        for file, collection in [('age_effects.csv.gz', ages), ('annual_cases.csv.gz', years)]:
            frame = pd.read_csv(record[2] / file)
            frame = frame.loc[frame.strategy_id.isin(['current', *PROFILE_IDS])]
            collection.append(frame.assign(**tags))
        if (record[2] / 'monthly_states.csv.gz').exists():
            states.append(pd.read_csv(record[2] / 'monthly_states.csv.gz').assign(**tags))
    for tags, subset in groups:
        arms = subset[0][3]['strategy_ids'].astype(str).tolist()
        assert all(r[3]['strategy_ids'].astype(str).tolist() == arms for r in subset)
        indices = [arms.index(p) for p in PROFILE_IDS]
        standard = [i for i, arm in enumerate(arms) if arm not in PROFILE_IDS]
        for horizon in (20, 30):
            for burden in ('reported_uniform', 'symptomatic_differentiated'):
                effects = np.concatenate([effect_for(r[3], horizon, burden, 'included_base') for r in subset])
                cost = np.concatenate([r[3][f'cost_{horizon}'] for r in subset])
                dd_parts = [r[3]['nasal_doses'][:, :horizon] @ (1.03**-np.arange(horizon)) for r in subset]
                dd = np.concatenate([np.repeat(d[None], r[0]['n_draws'], axis=0) for d, r in zip(dd_parts, subset)])
                label = {**tags, 'horizon': horizon, 'burden': burden, 'mortality': 'included_base',
                         'clinical_cost_draws_per_particle': subset[0][0]['n_draws']}
                for profile, i in zip(PROFILE_IDS, indices):
                    summaries.append({**label, 'profile_id': profile, **quantiles(effects[:, i], 'dalys_averted'),
                                      **quantiles(cost[:, i], 'non_nasal_incremental_cost'),
                                      'mean_discounted_nasal_doses': float(dd[:, i].mean())})
                    if tags['fit_group'] == 'local_pooled':
                        for record in subset:
                            e = effect_for(record[3], horizon, burden, 'included_base')[:, i]
                            particle_means.append({**label, 'profile_id': profile,
                                'draw_id': record[0]['draw_id'], 'source_replicate': record[0]['source_replicate'],
                                'source_parameter_row': record[0]['source_parameter_row'], 'dalys_averted_mean': float(e.mean())})
                difference = effects[:, indices[0]] - effects[:, indices[2]]
                contrasts.append({**label, 'contrast': 'onward_minus_acquisition',
                                  **quantiles(difference, 'dalys_difference'),
                                  'paired_fraction_above_zero': float(np.mean(difference > 0))})
                refs = references.loc[references.country.eq(tags['country']) & references.threshold_family.eq('sourced_DALY_estimate')]
                for ref in refs.to_dict('records'):
                    lam = ref['threshold_intl2024']
                    nmb = lam * effects - cost
                    comparator = standard[int(np.argmax(nmb[:, standard].mean(axis=0)))]
                    for profile, i in zip(PROFILE_IDS, indices):
                        numerator = nmb[:, i] - nmb[:, comparator]
                        # One common delivered price is applied across the transmission ensemble.
                        ceiling = numerator.mean() / dd[:, i].mean()
                        per_draw = numerator / dd[:, i]
                        residual = numerator - ceiling * dd[:, i]
                        assert abs(residual.mean()) < 1e-7 * max(1., np.abs(numerator).max())
                        prices.append({**label, 'profile_id': profile, 'threshold_id': ref['threshold_id'],
                            'threshold_intl2024': lam, 'standard_comparator': arms[comparator],
                            'common_price_ceiling': float(ceiling),
                            **quantiles(per_draw, 'draw_specific_price_ceiling'),
                            'expected_nmb_at_ceiling': float(residual.mean())})
    tables = {'health_summary.csv': pd.DataFrame(summaries), 'mechanism_contrasts.csv': pd.DataFrame(contrasts),
              'price_ceilings.csv': pd.DataFrame(prices), 'particle_means.csv': pd.DataFrame(particle_means),
              'age_effects.csv.gz': pd.concat(ages), 'annual_cases.csv.gz': pd.concat(years),
              'monthly_states.csv.gz': pd.concat(states)}
    for name, table in tables.items():
        table.to_csv(dest / name, index=False)
    write_json(dest / 'metadata.json', dict(status='completed_transmission_followup',
        local_countries=['England', 'Brazil'], local_particles_per_country=30,
        local_clinical_cost_draws_per_particle=50, historical_countries=['Japan', 'Brazil'],
        histories=['adopted', 'baseline', 'recovery_365'], historical_clinical_cost_draws=1000,
        profiles=list(PROFILE_IDS), nasal_trajectories=sum(len(r[1]['nasal_profiles']) for r in records),
        replayed_standard_trajectories=sum(r[1]['standard_current_replayed'] for r in records),
        active_zero_effect_replays=sum(r[1]['zero_effect_max_difference'] is not None for r in records),
        output_files=list(tables),
        price_definition='Ratio of mean incremental NMB before nasal cost to mean discounted nasal doses; comparator fixed by expected NMB.',
        snapshots='States on the 15th of each month, before any events that day; completed ages use the actual calendar-year fraction.'))
    print('Summaries completed', flush=True)


def check(output_dir):
    root = Path(output_dir)
    meta = json.loads((root / 'metadata.json').read_text())
    counts = dict(blocks=0, nasal_trajectories=0, standard_replays=0, zero_effect_replays=0)
    for job in jobs():
        folder = root / 'blocks' / job['label']
        block = json.loads((folder / 'metadata.json').read_text())
        assert block['job'] == job and block['status'] == 'completed' and block['standard_current_replayed']
        assert block['nasal_profiles'] == list(PROFILE_IDS)
        if job['kind'] == 'local' or job['history'] == 'adopted':
            assert block['paired_standard_components_reconstructed']
        expected_zero = job['kind'] == 'history' or job['draw_id'] == 0
        assert (block['zero_effect_max_difference'] is not None) == expected_zero
        if expected_zero:
            assert block['zero_effect_max_difference'] == 0
        counts['blocks'] += 1
        counts['nasal_trajectories'] += len(block['nasal_profiles'])
        counts['standard_replays'] += int(block['standard_current_replayed'])
        counts['zero_effect_replays'] += int(expected_zero)
    assert counts['nasal_trajectories'] == meta['nasal_trajectories']
    assert counts['standard_replays'] == meta['replayed_standard_trajectories']
    assert counts['zero_effect_replays'] == meta['active_zero_effect_replays']
    print(json.dumps({'status': 'passed', **counts}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--stage', choices=('run', 'summarize', 'check'), default='run')
    parser.add_argument('--only', nargs='+', help='Run selected job labels for timing or resumption')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1–4 workers')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 'check':
        check(args.output_dir)
    elif args.stage == 'summarize':
        summarize(args.output_dir)
    else:
        planned = jobs()
        if args.only:
            assert set(args.only) <= {j['label'] for j in planned}
            planned = [j for j in planned if j['label'] in args.only]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_job, j, args.output_dir) for j in planned]):
                future.result()
        if not args.only:
            summarize(args.output_dir)
            check(args.output_dir)


if __name__ == '__main__':
    main()
