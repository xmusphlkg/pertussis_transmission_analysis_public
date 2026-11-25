"""Inspect non-monotonic matching and clinical magnitude using retained models.

Every sign-changing bracket on the existing mechanism grid is retained. The
additional matching uses adopted fits; full-refit magnitude summaries revalue
saved trajectories without new transmission integrations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from scripts import analyse_review_revision_20261110 as rev
from scripts import analyse_review_round4_20261112 as r4
from scripts import analyse_matched_refit_followup_20261113 as matched

ROOT = rev.ROOT
OUT = rev.BASE / '02_sensitivity/mucosal_vaccines/matching_geometry'
METRICS = ('dalys', 'yll', 'infant_yll', 'yld', 'infant_cases', 'infant_deaths')
MECHANISMS = ('onward', 'acquisition')


def grid_brackets(strengths, gains, target):
    """Return exact sampled roots and all sampled sign-changing intervals."""
    strengths, gains = np.asarray(strengths, float), np.asarray(gains, float)
    if (len(strengths) != len(gains) or len(strengths) < 2
            or not np.isfinite(gains).all() or not np.all(np.diff(strengths) > 0)):
        raise ValueError('Expected ordered strengths and finite gains')
    residuals = gains - target
    exact = [float(x) for x, y in zip(strengths, residuals) if abs(y) <= 1e-10]
    intervals = [(float(a), float(b)) for a, b, fa, fb in
                 zip(strengths[:-1], strengths[1:], residuals[:-1], residuals[1:])
                 if abs(fa) > 1e-10 and abs(fb) > 1e-10 and fa * fb < 0]
    return exact, intervals


def solve_branches(evaluate, strengths, sampled_gains, target):
    exact, brackets = grid_brackets(strengths, sampled_gains, target)
    roots = [dict(strength=x, bracket_lower=x, bracket_upper=x, source='exact_grid_point')
             for x in exact]
    for lower, upper in brackets:
        root, result = brentq(lambda x: evaluate(x) - target, lower, upper,
                              xtol=1e-7, rtol=1e-8, full_output=True)
        if not result.converged:
            raise RuntimeError('Branch root failed to converge')
        roots.append(dict(strength=float(root), bracket_lower=lower, bracket_upper=upper,
                          source='grid_sign_change'))
    for root in roots:
        root['achieved_cases'] = float(evaluate(root['strength']))
        root['matching_error'] = abs(root['achieved_cases'] - target)
        if root['matching_error'] >= max(1e-3, abs(target) * 1e-6):
            raise ValueError('Branch matching tolerance exceeded')
    return sorted(roots, key=lambda row: row['strength'])


def axes(frame):
    return {
        'onward': frame[frame.acquisition_log_strength == 0].sort_values('onward_log_strength'),
        'acquisition': frame[frame.onward_log_strength == 0].sort_values('acquisition_log_strength'),
    }


def scan_grid():
    frame = pd.read_csv(r4.OUT / 'mechanism_grid.csv')
    reference = pd.read_csv(rev.OUT / 'matched_summary.csv').query("perturbation == 'reference'")
    rows = []
    for (country, horizon), group in frame.groupby(['country', 'horizon']):
        axis = axes(group)
        row = dict(country=country, horizon=int(horizon))
        for name, subset in axis.items():
            row[name + '_endpoint'] = float(subset.adolescent_cases.iloc[-1])
            row[name + '_sampled_peak'] = float(subset.adolescent_cases.max())
        minimum = min(row[name + '_sampled_peak'] for name in MECHANISMS)
        row['sampled_positive_overlap'] = minimum > 0
        row['half_sampled_common_peak_target'] = .5 * minimum if minimum > 0 else np.nan
        old = reference[(reference.country == country) & (reference.horizon == horizon)]
        if len(old) != 1:
            raise ValueError('Reference condition not uniquely identified')
        row['endpoint_rule_status'] = old.status.iloc[0]
        row['additional_interior_overlap'] = (row['endpoint_rule_status'] != 'matched' and minimum > 0)
        for name, subset in axis.items():
            if minimum > 0:
                exact, intervals = grid_brackets(subset[name + '_log_strength'],
                                                 subset.adolescent_cases, .5 * minimum)
                row[name + '_sampled_branches'] = len(exact) + len(intervals)
            else:
                row[name + '_sampled_branches'] = 0
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'reference_grid_overlap.csv', index=False)
    return result


def clinical_components(data, index, horizon):
    yll = data['yll'][:, index, :horizon, -1].sum(axis=1)
    yld = np.full(len(yll), data['yld_reported_uniform'][index, :horizon, -1].sum())
    return dict(dalys=yll + yld, yll=yll,
                infant_yll=data['yll'][:, index, :horizon, 0].sum(axis=1), yld=yld,
                infant_cases=np.full(len(yll), data['reported'][index, :horizon, 0].sum()),
                infant_deaths=data['deaths'][:, index, :horizon, 0].sum(axis=1))


def comparison(data, onward_index, acquisition_index, horizon):
    on = clinical_components(data, onward_index, horizon)
    ac = clinical_components(data, acquisition_index, horizon)
    # The last zero-onset bookkeeping array lets the same evaluator return the
    # current arm's absolute burden. It is never a simulated intervention.
    baseline = clinical_components(data, -1, horizon)
    row, draws = {}, {}
    for metric in METRICS:
        for label, value in (('onward', on[metric]), ('acquisition', ac[metric]),
                             ('difference', on[metric] - ac[metric])):
            row.update(rev.describe(value, label + '_' + metric))
            draws[label + '_' + metric] = value
        row['baseline_' + metric] = float(baseline[metric].mean())
    row['difference_percent_baseline_dalys'] = 100 * row['difference_dalys'] / row['baseline_dalys']
    row['difference_percent_baseline_infant_cases'] = 100 * row['difference_infant_cases'] / row['baseline_infant_cases']
    return row, draws


def compute_geometry(country, horizon, targets):
    folder = OUT / 'jobs' / country
    manifest = folder / 'manifest.json'
    if manifest.exists():
        record = matched.load_manifest(manifest)
        return dict(country=country, status='reused', new_projection_integrations=record['new_projection_integrations'])
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_manifest = r4.OUT / f'grid_{country}_manifest.json'
    matched.load_manifest(source_manifest)
    ctx = rev.setup(country)
    info, inputs, params, hist, scenarios, catalog, dates, days = ctx
    table = pd.read_csv(r4.OUT / f'grid_{country}.csv')
    table = table[table.horizon == horizon]
    coverage = float(table.nasal_coverage.iloc[0])
    axis = axes(table)
    source_raw = r4.OUT / f'grid_{country}.npz'
    with np.load(source_raw) as saved:
        raw = saved['onsets'].copy()
        np.testing.assert_array_equal(saved['dates'], dates.to_numpy())
    labels = list(product(r4.STRENGTHS, repeat=2))
    current = raw[labels.index((0., 0.))]
    logits, ratio = rev.ev._reporting(info, ROOT)
    reporting = rev.ev.monthly_reporting_weights(rev.AgeGrid(), logits, dates[:-1], dates[1:])

    def adolescent(onsets):
        change = current - onsets
        return float(np.einsum('mb,mbg->mg', change[..., 0] + ratio * change[..., 1],
                              reporting)[:horizon * 12, 2].sum())

    cache = {}
    for name in MECHANISMS:
        for strength in r4.STRENGTHS:
            index = labels.index((0., strength) if name == 'onward' else (strength, 0.))
            cache[(name, strength)] = (raw[index], adolescent(raw[index]))
        np.testing.assert_allclose([cache[(name, s)][1] for s in r4.STRENGTHS],
                                   axis[name].adolescent_cases, rtol=1e-8, atol=1e-5)
    profiles = {name: next(p for p in rev.bp.load_design(rev.bp.CONFIG)['profiles']
                          if p['candidate_id'] == f'b050_a{allocation:03d}_d03_point')
                for name, allocation in (('onward', 0), ('acquisition', 100))}
    integrations = 0

    def evaluate(name, strength):
        nonlocal integrations
        key = (name, float(strength))
        if key not in cache:
            profile = profiles[name]
            onsets, _ = rev.simulate(ctx, 1 - (1 - profile['efficacy'])**strength,
                                     profile['infectiousness_multiplier']**strength, coverage)
            cache[key] = (onsets, adolescent(onsets))
            integrations += 1
        return cache[key][1]

    rows, root_rows, samples, saved_trajectories = [], [], {}, {'dates': dates.to_numpy(), 'current': current}
    for target_label, fraction, rule in targets:
        peak = min(float(axis[name].adolescent_cases.max() if rule == 'sampled_peak'
                         else axis[name].adolescent_cases.iloc[-1]) for name in MECHANISMS)
        target = fraction * peak
        if target <= 0:
            raise ValueError('Target must be positive')
        roots = {}
        for name in MECHANISMS:
            roots[name] = solve_branches(lambda x: evaluate(name, x), r4.STRENGTHS,
                                         axis[name].adolescent_cases, target)
            for j, root in enumerate(roots[name]):
                root.update(country=country, horizon=horizon, target_label=target_label,
                            target_fraction=fraction, target_rule=rule, target_cases=target,
                            mechanism=name, branch=j, nasal_coverage=coverage)
                root_rows.append(dict(root))
        trajectories = [current]
        indices = {}
        for name in MECHANISMS:
            for j, root in enumerate(roots[name]):
                indices[(name, j)] = len(trajectories)
                onsets = cache[(name, root['strength'])][0]
                trajectories.append(onsets)
                saved_trajectories[f'{target_label}_{name}_{j}'] = onsets
        trajectories.append(np.zeros_like(current))
        data = rev.clinical(country, np.stack(trajectories), dates[:-1], dates[1:], info)
        for on_index, ac_index in product(range(len(roots['onward'])), range(len(roots['acquisition']))):
            row, draws = comparison(data, indices[('onward', on_index)], indices[('acquisition', ac_index)], horizon)
            row.update(country=country, horizon=horizon, target_label=target_label, target_rule=rule,
                       target_fraction=fraction, target_cases=target, nasal_coverage=coverage,
                       onward_branch=on_index, acquisition_branch=ac_index,
                       onward_strength=roots['onward'][on_index]['strength'],
                       acquisition_strength=roots['acquisition'][ac_index]['strength'])
            rows.append(row)
            prefix = f'{target_label}_{on_index}_{ac_index}_'
            samples.update({prefix + key: value for key, value in draws.items()})
    paths = [folder / 'comparisons.csv', folder / 'roots.csv', folder / 'paired_draws.npz', folder / 'trajectories.npz']
    pd.DataFrame(rows).to_csv(paths[0], index=False)
    pd.DataFrame(root_rows).to_csv(paths[1], index=False)
    np.savez_compressed(paths[2], **samples)
    np.savez_compressed(paths[3], **saved_trajectories)
    input_files = [str(path.relative_to(ROOT)) for path in
                   (Path(__file__).resolve(), Path(matched.__file__).resolve(),
                    source_manifest, source_raw, r4.OUT / f'grid_{country}.csv')]
    record = dict(country=country, horizon=horizon, targets=targets, fit_directory=info['fit_directory'],
                  new_projection_integrations=integrations, history_replays=1,
                  seconds=time.monotonic() - started, input_files=input_files,
                  output_files=[str(path.relative_to(ROOT)) for path in paths])
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return {key: record[key] for key in ('country', 'new_projection_integrations', 'seconds')}


def magnitude_summary():
    """Revalue every retained matched full-refit trajectory, with no ODE runs."""
    frame = pd.read_csv(matched.OUT / 'matched_refit_summary.csv')
    prediction_path = rev.BASE / '04_covid/refit_review_20261105/prediction_comparison.csv'
    predictions = pd.read_csv(prediction_path)
    rows, samples = [], {}
    for (country, history), group in frame.groupby(['country', 'history']):
        folder = matched.job_folder(country, history)
        manifest = folder / 'manifest.json'
        matched.load_manifest(manifest)
        info = json.loads(matched.history_complete_path(country, history).read_text())
        with np.load(folder / 'trajectories.npz') as saved:
            current, dates = saved['current'], saved['dates']
            for old in group[group.status == 'matched'].itertuples():
                h = old.horizon
                raw = np.stack([current, saved[f'onward_{h}'], saved[f'acquisition_{h}'], np.zeros_like(current)])
                data = rev.clinical(country, raw, dates[:-1], dates[1:], info)
                row, draws = comparison(data, 1, 2, h)
                row.update(country=country, history=history, horizon=h,
                           target_cases=old.target_adolescent_cases_averted,
                           onward_strength=old.onward_strength, acquisition_strength=old.acquisition_strength)
                for metric, previous_name in (('dalys', 'onward_minus_acquisition_dalys'),
                                             ('infant_cases', 'onward_minus_acquisition_infant_cases'),
                                             ('infant_deaths', 'onward_minus_acquisition_infant_deaths')):
                    np.testing.assert_allclose(row['difference_' + metric], getattr(old, previous_name),
                                               rtol=1e-11, atol=1e-8)
                diagnostics = predictions[(predictions.country == country) & (predictions.scenario == history)]
                for window in ('training', 'posttraining'):
                    diagnostic = diagnostics[diagnostics.window == window]
                    if len(diagnostic) > 1:
                        raise ValueError('Ambiguous diagnostic window')
                    for field in ('wape', 'expected_observed_ratio', 'observed_total', 'expected_total'):
                        row[window + '_' + field] = float(diagnostic.iloc[0][field]) if len(diagnostic) else np.nan
                    row[window + '_years'] = str(diagnostic.iloc[0].available_years) if len(diagnostic) else 'not available'
                rows.append(row)
                samples.update({f'{country}_{history}_{h}_{key}': value for key, value in draws.items()})
    output = OUT / 'matched_refit_clinical_magnitude.csv'
    draw_path = OUT / 'matched_refit_clinical_draws.npz'
    pd.DataFrame(rows).to_csv(output, index=False)
    np.savez_compressed(draw_path, **samples)
    record = dict(new_projection_integrations=0, matched_conditions=len(rows),
                  input_files=[str(path.relative_to(ROOT)) for path in
                               (prediction_path, matched.OUT / 'matched_refit_summary.csv',
                                Path(__file__).resolve())],
                  output_files=[str(path.relative_to(ROOT)) for path in (output, draw_path)])
    (OUT / 'magnitude_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return pd.DataFrame(rows)


def summary():
    scan = scan_grid()
    rows, roots = [], []
    manifests = list((OUT / 'jobs').glob('*/manifest.json'))
    records = [matched.load_manifest(path) for path in manifests]
    for path in manifests:
        rows.append(pd.read_csv(path.parent / 'comparisons.csv'))
        roots.append(pd.read_csv(path.parent / 'roots.csv'))
    comparisons, branch_roots = pd.concat(rows, ignore_index=True), pd.concat(roots, ignore_index=True)
    comparisons.to_csv(OUT / 'branch_comparisons.csv', index=False)
    branch_roots.to_csv(OUT / 'branch_roots.csv', index=False)
    magnitude = magnitude_summary()
    report = dict(reference_conditions=len(scan), endpoint_unmatched=int((scan.endpoint_rule_status != 'matched').sum()),
                  additional_sampled_interior_overlap=int(scan.additional_interior_overlap.sum()),
                  total_roots=len(branch_roots), branch_comparisons=len(comparisons),
                  positive_mean_daly_contrasts=int((comparisons.difference_dalys > 0).sum()),
                  positive_mean_infant_case_contrasts=int((comparisons.difference_infant_cases > 0).sum()),
                  interval_above_zero=int((comparisons.difference_dalys_q025 > 0).sum()),
                  interval_crossing_zero=int(((comparisons.difference_dalys_q025 <= 0)
                                             & (comparisons.difference_dalys_q975 >= 0)).sum()),
                  new_projection_integrations=sum(record['new_projection_integrations'] for record in records),
                  history_replays=len(records), cumulative_job_seconds=sum(record['seconds'] for record in records),
                  revalued_matched_refit_conditions=len(magnitude))
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    input_files = manifests + [path.parent / filename for path in manifests for filename in ('comparisons.csv', 'roots.csv')]
    input_files += [r4.OUT / 'mechanism_grid.csv', rev.OUT / 'matched_summary.csv', OUT / 'magnitude_manifest.json', Path(__file__).resolve()]
    outputs = [OUT / name for name in ('reference_grid_overlap.csv', 'branch_comparisons.csv', 'branch_roots.csv', 'summary.json')]
    (OUT / 'summary_manifest.json').write_text(json.dumps(dict(
        input_files=[str(path.relative_to(ROOT)) for path in input_files],
        output_files=[str(path.relative_to(ROOT)) for path in outputs]), indent=2) + '\n')
    return report


def verify_draw_summary(row, arrays, prefix):
    for metric in METRICS:
        on, ac = arrays[prefix + 'onward_' + metric], arrays[prefix + 'acquisition_' + metric]
        difference = arrays[prefix + 'difference_' + metric]
        np.testing.assert_array_equal(on - ac, difference)
        for label, values in (('onward', on), ('acquisition', ac), ('difference', difference)):
            for name, value in rev.describe(values, label + '_' + metric).items():
                np.testing.assert_allclose(value, getattr(row, name), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(arrays[prefix + 'difference_dalys'],
                               arrays[prefix + 'difference_yll'] + arrays[prefix + 'difference_yld'],
                               rtol=1e-12, atol=1e-8)


def check():
    matched.load_manifest(OUT / 'summary_manifest.json')
    matched.load_manifest(OUT / 'magnitude_manifest.json')
    for manifest in (OUT / 'jobs').glob('*/manifest.json'):
        record = matched.load_manifest(manifest)
        roots = pd.read_csv(manifest.parent / 'roots.csv')
        comparisons = pd.read_csv(manifest.parent / 'comparisons.csv')
        table = pd.read_csv(r4.OUT / f"grid_{record['country']}.csv")
        axis = axes(table[table.horizon == record['horizon']])
        for target, part in roots.groupby('target_label'):
            for name in MECHANISMS:
                exact, brackets = grid_brackets(r4.STRENGTHS, axis[name].adolescent_cases, float(part.target_cases.iloc[0]))
                expected = set((value, value) for value in exact) | set(brackets)
                selected = part[part.mechanism == name]
                actual = set(zip(selected.bracket_lower, selected.bracket_upper))
                if len(selected) != len(expected) or actual != expected:
                    raise ValueError('Missing or duplicate sampled branch')
                if (selected.matching_error >= np.maximum(1e-3, abs(selected.target_cases) * 1e-6)).any():
                    raise ValueError('Matching error exceeds tolerance')
            expected_pairs = set(product(part[part.mechanism == 'onward'].branch,
                                         part[part.mechanism == 'acquisition'].branch))
            subset = comparisons[comparisons.target_label == target]
            if len(subset) != len(expected_pairs) or set(zip(subset.onward_branch, subset.acquisition_branch)) != expected_pairs:
                raise ValueError('Missing or duplicate root combination')
        with np.load(manifest.parent / 'paired_draws.npz') as arrays:
            for row in comparisons.itertuples():
                verify_draw_summary(row, arrays, f'{row.target_label}_{row.onward_branch}_{row.acquisition_branch}_')
    magnitude = pd.read_csv(OUT / 'matched_refit_clinical_magnitude.csv')
    with np.load(OUT / 'matched_refit_clinical_draws.npz') as arrays:
        for row in magnitude.itertuples():
            verify_draw_summary(row, arrays, f'{row.country}_{row.history}_{row.horizon}_')
    print('Checked all sampled branches, root combinations, matched errors and paired clinical components.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'summary', 'check'), default='all')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage == 'check':
        check()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'all':
        scan = scan_grid()
        jobs = [(r.country, int(r.horizon), [('half_sampled_peak', .5, 'sampled_peak')])
                for r in scan[scan.additional_interior_overlap].itertuples()]
        jobs.append(('Sweden', 30, [('quarter_endpoint', .25, 'endpoint'), ('three_quarter_endpoint', .75, 'endpoint')]))
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            futures = [pool.submit(compute_geometry, *job) for job in jobs]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    print(json.dumps(summary(), indent=2), flush=True)
    check()


if __name__ == '__main__':
    main()
