"""Match adolescent outcomes under the two retained full-coordinate refits.

The original endpoint rule, coverage, three-year protection, root tolerances and
paired clinical evaluation are retained. Each fitted history has its own inputs,
comparator and matching target. Existing producers and outputs are read only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from scripts import analyse_review_revision_20261110 as rev
from scripts import analyse_review_round4_20261112 as r4
from scripts import build_bpze1_policy_design_20261107 as design

ROOT = rev.ROOT
OUT = rev.BASE / '02_sensitivity/mucosal_vaccines/matched_refit_followup'
HISTORIES = ('baseline', 'recovery_365')
HORIZONS = (20, 30)


def history_complete_path(country, history):
    if history not in HISTORIES:
        raise ValueError(f'Unsupported fitted history: {history}')
    return rev.REFIT / 'policy/projections' / history / country / 'complete.json'


def load_context(country, history):
    expected = json.loads(history_complete_path(country, history).read_text())
    ctx = rev.setup(country, history=history)
    if ctx[0]['fit_directory'] != expected['fit_directory']:
        raise ValueError(f'Fitted history mismatch for {country}/{history}')
    return ctx


def solve_matches(evaluate, horizons=HORIZONS):
    """Solve each horizon independently; evaluate returns monthly age-group gains."""
    rows = []
    for horizon in horizons:
        gains = {a: float(evaluate(a, 1.)[:horizon * 12, 2].sum()) for a in (0, 100)}
        target = .5 * min(gains.values())
        row = dict(horizon=horizon, target_adolescent_cases_averted=target,
                   onward_endpoint_adolescent_gain=gains[0],
                   acquisition_endpoint_adolescent_gain=gains[100])
        if target <= 0:
            failed = [name for a, name in ((0, 'onward'), (100, 'acquisition')) if gains[a] <= 0]
            rows.append(dict(row, status='no_positive_endpoint_overlap',
                             unmatched_reason='+'.join(failed) + '_endpoint_nonpositive'))
            continue
        roots = {}
        for a, name in ((0, 'onward'), (100, 'acquisition')):
            f = lambda strength: float(evaluate(a, strength)[:horizon * 12, 2].sum()) - target
            root, result = brentq(f, 0., 1., xtol=1e-7, rtol=1e-8, full_output=True)
            if not result.converged:
                raise RuntimeError(f'{name} root did not converge at {horizon} years')
            roots[a] = root
            row[name + '_strength'] = root
            row[name + '_root_iterations'] = result.iterations
            row[name + '_root_function_calls'] = result.function_calls
            row[name + '_adolescent_cases_averted'] = float(evaluate(a, root)[:horizon * 12, 2].sum())
        error = max(abs(float(evaluate(a, roots[a])[:horizon * 12, 2].sum()) - target)
                    for a in (0, 100))
        tolerance = max(1e-3, abs(target) * 1e-6)
        if error >= tolerance:
            raise ValueError(f'Matching error {error} exceeds {tolerance}')
        rows.append(dict(row, status='matched', unmatched_reason='',
                         max_matching_error=error, matching_tolerance=tolerance))
    return rows


def load_manifest(path):
    """Load a producer record; numerical checks are performed by the caller."""
    return json.loads(path.read_text())


def job_folder(country, history):
    return OUT / 'jobs' / f'{country}__{history}'


def compute_job(country, history):
    folder = job_folder(country, history)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        saved = load_manifest(manifest)
        return dict(country=country, history=history, status='reused',
                    new_projection_integrations=saved['new_projection_integrations'])
    folder.mkdir(parents=True, exist_ok=True)
    opened = set()

    def capture(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(args[0]).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(OUT) and path.suffix in ('.csv', '.gz', '.json', '.yaml', '.npz'):
                opened.add(path)

    # Each task runs in a fresh process, so this input audit is task-local.
    sys.addaudithook(capture)
    started = time.monotonic()
    prior_manifest = r4.OUT / f'refits_{country}_manifest.json'
    load_manifest(prior_manifest)
    ctx = load_context(country, history)
    info, inputs, params, hist, scenarios, catalog, dates, days = ctx
    coverage = max(.7, float(catalog.loc[catalog.strategy_id == 'current', 'adolescent_coverage'].iloc[0]))
    source_raw = r4.OUT / f'refits_{country}_{history}.npz'
    with np.load(source_raw) as saved:
        raw = saved['onsets'].copy()
        np.testing.assert_array_equal(saved['dates'], dates.to_numpy())
    if raw.shape != (4, 360, rev.AgeGrid().n_bins, 3):
        raise ValueError(f'Unexpected saved trajectory shape: {raw.shape}')
    current = raw[0]
    endpoints = {0: raw[1], 100: raw[3]}
    profiles = {a: next(p for p in rev.bp.load_design(rev.bp.CONFIG)['profiles']
                        if p['candidate_id'] == f'b050_a{a:03d}_d03_point') for a in (0, 100)}
    logits, ratio = rev.ev._reporting(info, ROOT)
    reporting = rev.ev.monthly_reporting_weights(rev.AgeGrid(), logits, dates[:-1], dates[1:])

    def case_gains(onsets):
        diff = current - onsets
        return np.einsum('mb,mbg->mg', diff[..., 0] + ratio * diff[..., 1], reporting)

    cache = {(a, 0.): (current, case_gains(current)) for a in (0, 100)}
    cache.update({(a, 1.): (onsets, case_gains(onsets)) for a, onsets in endpoints.items()})
    integrations = 0

    def evaluate(a, strength):
        nonlocal integrations
        key = (a, float(strength))
        if key not in cache:
            p = profiles[a]
            efficacy = 1 - (1 - p['efficacy']) ** strength
            onward = p['infectiousness_multiplier'] ** strength
            onsets, _ = rev.simulate(ctx, efficacy, onward, coverage)
            integrations += 1
            cache[key] = (onsets, case_gains(onsets))
        return cache[key][1]

    # Check the reused arms against the prior reader results before root finding.
    previous = pd.read_csv(r4.OUT / f'refits_{country}.csv')
    previous = previous[(previous.history == history) & previous.allocation.isin([0., 1.])]
    for row in previous.itertuples():
        a = int(row.allocation * 100)
        np.testing.assert_allclose(evaluate(a, 1.)[:row.horizon * 12, 2].sum(),
                                   row.adolescent_cases, rtol=1e-8, atol=1e-5)
    rows = solve_matches(evaluate)
    trajectories = {'current': current, 'onward_endpoint': endpoints[0],
                    'acquisition_endpoint': endpoints[100], 'dates': dates.to_numpy()}
    draws = {}
    for row in rows:
        row.update(country=country, history=history, nasal_coverage=coverage,
                   mean_protection_years=3., fit_directory=info['fit_directory'])
        if row['status'] != 'matched':
            continue
        horizon = row['horizon']
        onraw = cache[(0, row['onward_strength'])][0]
        acraw = cache[(100, row['acquisition_strength'])][0]
        trajectories[f'onward_{horizon}'] = onraw
        trajectories[f'acquisition_{horizon}'] = acraw
        data = rev.clinical(country, np.stack([current, onraw, acraw]), dates[:-1], dates[1:], info)
        on = data['yll'][:, 1, :horizon, -1].sum(axis=1) + data['yld_reported_uniform'][1, :horizon, -1].sum()
        ac = data['yll'][:, 2, :horizon, -1].sum(axis=1) + data['yld_reported_uniform'][2, :horizon, -1].sum()
        deaths = data['deaths'][:, 1, :horizon, 0].sum(axis=1) - data['deaths'][:, 2, :horizon, 0].sum(axis=1)
        row.update(onward_infant_cases_averted=float(data['reported'][1, :horizon, 0].sum()),
                   acquisition_infant_cases_averted=float(data['reported'][2, :horizon, 0].sum()),
                   **rev.describe(on, 'onward_dalys'), **rev.describe(ac, 'acquisition_dalys'),
                   **rev.describe(on - ac, 'onward_minus_acquisition_dalys'),
                   **rev.describe(deaths, 'onward_minus_acquisition_infant_deaths'))
        row['onward_minus_acquisition_infant_cases'] = row['onward_infant_cases_averted'] - row['acquisition_infant_cases_averted']
        for j, name in ((1, 'onward'), (2, 'acquisition')):
            np.testing.assert_allclose(data['reported'][j, :horizon, 2].sum(),
                                       row[name + '_adolescent_cases_averted'], rtol=1e-8, atol=1e-5)
        draws[f'onward_dalys_{horizon}'] = on
        draws[f'acquisition_dalys_{horizon}'] = ac
        draws[f'difference_{horizon}'] = on - ac
        draws[f'infant_deaths_difference_{horizon}'] = deaths
    table_path = folder / 'summary.csv'
    trajectory_path = folder / 'trajectories.npz'
    draws_path = folder / 'paired_draws.npz'
    pd.DataFrame(rows).to_csv(table_path, index=False)
    np.savez_compressed(trajectory_path, **trajectories)
    np.savez_compressed(draws_path, **draws)
    input_files = sorted({str(path.relative_to(ROOT)) for path in opened
                          if path.is_relative_to(ROOT)} | {
                              str(path.relative_to(ROOT)) for path in
                              (Path(__file__).resolve(), Path(rev.__file__).resolve(),
                               Path(r4.__file__).resolve(), Path(design.__file__).resolve(),
                               Path(rev.ev.__file__).resolve(), prior_manifest, source_raw)
                              if path.is_relative_to(ROOT)})
    record = dict(country=country, history=history, fit_directory=info['fit_directory'],
                  seconds=time.monotonic() - started, reused_projection_trajectories=3,
                  history_replays=1, new_projection_integrations=integrations,
                  matched_conditions=sum(row['status'] == 'matched' for row in rows),
                  input_files=input_files,
                  output_files=[str(path.relative_to(ROOT)) for path in
                                (table_path, trajectory_path, draws_path)])
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return {k: record[k] for k in ('country', 'history', 'seconds', 'new_projection_integrations', 'matched_conditions')}


def validate_rows(frame):
    if frame.duplicated(['country', 'history', 'horizon']).any():
        raise ValueError('Duplicate setting/history/horizon')
    for row in frame.itertuples():
        target = .5 * min(row.onward_endpoint_adolescent_gain, row.acquisition_endpoint_adolescent_gain)
        np.testing.assert_allclose(row.target_adolescent_cases_averted, target, rtol=1e-12, atol=1e-9)
        if row.status == 'matched':
            if target <= 0 or not (0 < row.onward_strength < 1 and 0 < row.acquisition_strength < 1):
                raise ValueError('Invalid matched root or target')
            error = max(abs(row.onward_adolescent_cases_averted - target),
                        abs(row.acquisition_adolescent_cases_averted - target))
            if error >= max(1e-3, abs(target) * 1e-6):
                raise ValueError('Saved result does not match its own horizon target')
        elif row.status != 'no_positive_endpoint_overlap' or target > 0:
            raise ValueError('Invalid unmatched status')


def condition_summary(frame):
    good = frame[frame.status == 'matched']
    difference = good.onward_minus_acquisition_dalys
    infant = good.onward_infant_cases_averted - good.acquisition_infant_cases_averted
    return dict(conditions=len(frame), matched=len(good), unmatched=len(frame) - len(good),
                positive_mean_daly_difference=int((difference > 0).sum()),
                negative_mean_daly_difference=int((difference < 0).sum()),
                paired_interval_above_zero=int((good.onward_minus_acquisition_dalys_q025 > 0).sum()),
                positive_infant_case_difference=int((infant > 0).sum()),
                daly_difference_min=float(difference.min()), daly_difference_max=float(difference.max()),
                infant_case_difference_min=float(infant.min()), infant_case_difference_max=float(infant.max()),
                maximum_matching_error=float(good.max_matching_error.max()))


def summarize(countries):
    manifests = [job_folder(c, h) / 'manifest.json' for c in countries for h in HISTORIES]
    records = [load_manifest(p) for p in manifests]
    inputs = [p.parent / 'summary.csv' for p in manifests]
    frame = pd.concat([pd.read_csv(p) for p in inputs], ignore_index=True)
    frame = frame.sort_values(['country', 'history', 'horizon']).reset_index(drop=True)
    validate_rows(frame)
    source = rev.OUT / 'matched_summary.csv'
    reference = pd.read_csv(source).query("perturbation == 'reference'")
    summary = dict(countries=list(countries), histories=list(HISTORIES), horizons=list(HORIZONS),
                   new_refit_conditions=condition_summary(frame),
                   adopted_reference_conditions=condition_summary(reference),
                   by_history={h: condition_summary(frame[frame.history == h]) for h in HISTORIES},
                   new_projection_integrations=sum(r['new_projection_integrations'] for r in records),
                   reused_projection_trajectories=sum(r['reused_projection_trajectories'] for r in records),
                   history_replays=sum(r['history_replays'] for r in records),
                   cumulative_job_seconds=sum(r['seconds'] for r in records),
                   scope='Unweighted fitted-history conditions; paired clinical intervals, not transmission posterior intervals.')
    full = OUT / 'matched_refit_summary.csv'
    reader = OUT / 'matched_refit_reader.csv'
    markdown = OUT / 'matched_refit_reader.md'
    summary_path = OUT / 'summary.json'
    frame.to_csv(full, index=False)
    columns = ['country', 'history', 'horizon', 'status', 'target_adolescent_cases_averted',
               'onward_strength', 'acquisition_strength', 'onward_infant_cases_averted',
               'acquisition_infant_cases_averted', 'onward_minus_acquisition_dalys',
               'onward_minus_acquisition_dalys_q025', 'onward_minus_acquisition_dalys_q975']
    display = frame[columns].copy()
    display.loc[display.status != 'matched', 'target_adolescent_cases_averted'] = np.nan
    display.to_csv(reader, index=False)
    names = ['Setting', 'Fitted history', 'Years', 'Status', 'Matched adolescent cases',
             'Onward strength', 'Acquisition strength', 'Onward infant cases averted',
             'Acquisition infant cases averted', 'DALY difference', 'Lower 95% UI', 'Upper 95% UI']
    lines = ['| ' + ' | '.join(names) + ' |', '| ' + ' | '.join(['---'] * len(names)) + ' |']
    for row in display.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append('—')
            elif isinstance(value, (float, np.floating)):
                cells.append(f'{value:.6g}')
            else:
                cells.append(str(value).replace('_', ' '))
        lines.append('| ' + ' | '.join(cells) + ' |')
    note = ('Each history supplies its own fitted parameters, ascertainment, initial state and comparator. '
            'Standalone nasal coverage is 70% or the higher baseline adolescent coverage; mean protection is three years. '
            'Each horizon independently matches half the smaller positive endpoint reduction in adolescent reported cases. '
            'DALY differences are onward-contribution minus acquisition protection, with paired 95% clinical intervals. '
            'No positive endpoint overlap refers to this endpoint-based rule and does not exclude an interior positive effect. '
            'The fitted histories have no probability weights.\n\n')
    markdown.write_text(note + '\n'.join(lines) + '\n')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    metadata = dict(input_files=[str(path.relative_to(ROOT)) for path in
                                 (inputs + manifests + [source, Path(__file__).resolve()])],
                    output_files=[str(path.relative_to(ROOT)) for path in
                                  (full, reader, markdown, summary_path)])
    (OUT / 'summary_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return summary


def check(countries):
    total_integrations = 0
    for country in countries:
        for history in HISTORIES:
            folder = job_folder(country, history)
            record = load_manifest(folder / 'manifest.json')
            expected = json.loads(history_complete_path(country, history).read_text())
            if record['fit_directory'] != expected['fit_directory']:
                raise ValueError('Saved result uses the wrong fitted history')
            frame = pd.read_csv(folder / 'summary.csv')
            validate_rows(frame)
            if set(frame.horizon) != set(HORIZONS) or len(frame) != len(HORIZONS):
                raise ValueError('Incomplete horizon results')
            with np.load(folder / 'paired_draws.npz') as draws:
                for row in frame[frame.status == 'matched'].itertuples():
                    h = row.horizon
                    diff = draws[f'onward_dalys_{h}'] - draws[f'acquisition_dalys_{h}']
                    np.testing.assert_array_equal(diff, draws[f'difference_{h}'])
                    for name, value in rev.describe(diff, 'onward_minus_acquisition_dalys').items():
                        np.testing.assert_allclose(value, getattr(row, name), rtol=1e-12, atol=1e-9)
            total_integrations += record['new_projection_integrations']
    load_manifest(OUT / 'summary_manifest.json')
    frame = pd.read_csv(OUT / 'matched_refit_summary.csv')
    validate_rows(frame)
    if len(frame) != len(countries) * len(HISTORIES) * len(HORIZONS):
        raise ValueError('Summary does not cover every requested condition')
    print(f'Verified {len(countries) * len(HISTORIES)} fitted-history jobs, '
          f'{len(frame)} conditions and {total_integrations} new prospective integrations', flush=True)


def set_output_dir(path):
    global OUT
    OUT = Path(path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'summary', 'check'), default='all')
    parser.add_argument('--countries', nargs='+', choices=rev.COUNTRIES, default=list(rev.COUNTRIES))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    set_output_dir(args.output_dir)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage == 'check':
        check(args.countries)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'all':
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1,
                                 initializer=set_output_dir, initargs=(OUT,)) as pool:
            futures = [pool.submit(compute_job, country, history)
                       for country in args.countries for history in HISTORIES]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    print(json.dumps(summarize(args.countries), indent=2), flush=True)
    check(args.countries)


if __name__ == '__main__':
    main()
