"""Paired maternal–nasal interaction under existing full-coordinate refits.

Only prospective maternal symptom protection varies. Each history supplies its
own initial state, fitted parameters and ascertainment. Original producers and
outputs remain read only; completed jobs are reused from their numerical records.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from scripts import analyse_review_revision_20261110 as rev
from scripts import analyse_review_round4_20261112 as r4
from scripts import build_bpze1_policy_design_20261107 as design

ROOT = rev.ROOT
OUT = rev.BASE / '02_sensitivity/mucosal_vaccines/maternal_refit_interaction'
HISTORIES = ('baseline', 'recovery_365')
SCALES = (1., .5)
HORIZONS = (20, 30)
ARMS = ('low_maternal_no_nasal', 'low_maternal_nasal',
        'high_maternal_no_nasal', 'high_maternal_nasal')
METRICS = ('dalys', 'yll', 'infant_yll', 'yld', 'infant_cases', 'infant_deaths')


def load_manifest(path):
    """Load a producer record; numerical checks are performed by the caller."""
    return json.loads(path.read_text())


def history_path(country, history):
    if history not in HISTORIES:
        raise ValueError(f'Unsupported fitted history: {history}')
    return rev.REFIT / 'policy/projections' / history / country / 'complete.json'


def load_context(country, history):
    expected = json.loads(history_path(country, history).read_text())
    ctx = rev.setup(country, history=history)
    if ctx[0]['fit_directory'] != expected['fit_directory']:
        raise ValueError('Wrong fitted history')
    return ctx


def maternal_parameters(params, scale):
    if scale not in SCALES:
        raise ValueError('Unsupported maternal protection scale')
    return replace(params, pi_maternal=1 - scale * (1 - params.pi_maternal))


def select_programmes(catalog):
    baseline = catalog[catalog.strategy_id == 'current']
    if len(baseline) != 1:
        raise ValueError('Expected one current programme')
    current = baseline.iloc[0]
    high = catalog[np.isclose(catalog.maternal_coverage, .9)
                   & np.isclose(catalog.adolescent_coverage, current.adolescent_coverage)]
    if len(high) != 1 or current.maternal_coverage >= .9:
        raise ValueError('Expected a distinct 90% maternal programme retaining baseline Tdap')
    return current, high.iloc[0], max(.7, float(current.adolescent_coverage))


def endpoint_draws(data, horizon):
    """Four-arm gains versus arm zero; mortality remains paired across arms."""
    yll = data['yll'][:, :, :horizon, -1].sum(axis=2)
    yld = data['yld_reported_uniform'][:, :horizon, -1].sum(axis=1)
    yld = np.broadcast_to(yld, yll.shape)
    cases = data['reported'][:, :horizon, 0].sum(axis=1)
    return dict(dalys=yll + yld, yll=yll,
                infant_yll=data['yll'][:, :, :horizon, 0].sum(axis=2), yld=yld,
                infant_cases=np.broadcast_to(cases, yll.shape),
                infant_deaths=data['deaths'][:, :, :horizon, 0].sum(axis=2))


def paired_interaction(values):
    """Positive interaction means higher maternal coverage increases nasal gain."""
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError('Expected paired draw × four-arm array')
    low = values[:, 1] - values[:, 0]
    high = values[:, 3] - values[:, 2]
    return low, high, high - low


def percent_reduction(low, high):
    """Ratio of conditional means; nonpositive low-arm benefit has no percentage."""
    denominator = float(np.mean(low))
    return 100 * (denominator - float(np.mean(high))) / denominator if denominator > 0 else np.nan


def result_rows(data, country, history, scale, low_coverage, nasal_coverage):
    rows, arms, arrays = [], [], {}
    for horizon in HORIZONS:
        endpoint = endpoint_draws(data, horizon)
        info = dict(country=country, history=history, protection_scale=scale,
                    horizon=horizon, low_maternal_coverage=low_coverage,
                    high_maternal_coverage=.9, nasal_coverage=nasal_coverage)
        row = dict(info)
        for metric, values in endpoint.items():
            low, high, contrast = paired_interaction(values)
            for label, samples in (('low', low), ('high', high), ('interaction', contrast)):
                row.update(rev.describe(samples, f'{label}_{metric}'))
                arrays[f'{metric}_{label}_{horizon}'] = samples
            arrays[f'{metric}_arms_{horizon}'] = values
        row['percent_nasal_daly_reduction'] = percent_reduction(
            arrays[f'dalys_low_{horizon}'], arrays[f'dalys_high_{horizon}'])
        row['percentage_status'] = ('defined_positive_low_gain' if row['low_dalys'] > 0
                                    else 'undefined_nonpositive_low_gain')
        rows.append(row)
        for j, name in enumerate(ARMS):
            arm = dict(info, arm=name,
                       maternal_coverage=low_coverage if j < 2 else .9,
                       arm_nasal_coverage=nasal_coverage if j % 2 else 0.,
                       reference='low_maternal_no_nasal_with_same_protection_scale')
            for metric, values in endpoint.items():
                arm.update(rev.describe(values[:, j], metric + '_averted'))
            arms.append(arm)
    return rows, arms, arrays


def job_folder(country, history):
    return OUT / 'jobs' / f'{country}__{history}'


def compute_job(country, history):
    folder = job_folder(country, history)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        record = load_manifest(manifest)
        return dict(country=country, history=history, status='reused',
                    new_projection_integrations=record['new_projection_integrations'])
    folder.mkdir(parents=True, exist_ok=True)
    opened = set()

    def capture(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(args[0]).resolve()
            if (path.is_relative_to(ROOT) and not path.is_relative_to(OUT)
                    and path.suffix in ('.csv', '.gz', '.json', '.yaml', '.npz')):
                opened.add(path)

    sys.addaudithook(capture)
    started = time.monotonic()
    prior_manifest = r4.OUT / f'refits_{country}_manifest.json'
    load_manifest(prior_manifest)
    ctx = load_context(country, history)
    info, inputs, params, hist, scenarios, catalog, dates, days = ctx
    low, high, coverage = select_programmes(catalog)
    profile = next(p for p in rev.bp.load_design(rev.bp.CONFIG)['profiles']
                   if p['candidate_id'] == rev.PID)
    previous_path = r4.OUT / f'refits_{country}_{history}.npz'
    with np.load(previous_path) as saved:
        old = saved['onsets'].copy()
        np.testing.assert_array_equal(saved['dates'], dates.to_numpy())
    previous = pd.read_csv(r4.OUT / f'refits_{country}.csv')
    previous = previous[(previous.history == history) & np.isclose(previous.allocation, .5)]
    rows, arm_rows, draws, trajectories = [], [], {}, {'dates': dates.to_numpy()}
    checks = []
    for scale in SCALES:
        changed = maternal_parameters(params, scale)
        raw = []
        for programme in (low.strategy_id, high.strategy_id):
            for nasal in (False, True):
                onsets, _ = rev.simulate(
                    ctx, profile['efficacy'] if nasal else 0.,
                    profile['infectiousness_multiplier'] if nasal else 1.,
                    coverage if nasal else 0., scenario=programme, params=changed)
                raw.append(onsets)
        raw = np.stack(raw)
        if scale == 1.:
            for j, prior_j in ((0, 0), (1, 2)):
                np.testing.assert_allclose(raw[j], old[prior_j], rtol=2e-8, atol=2e-4)
                checks.append(dict(arm=ARMS[j], maximum_absolute_onset_difference=
                                   float(np.max(np.abs(raw[j] - old[prior_j])))))
        data = rev.clinical(country, raw, dates[:-1], dates[1:], info)
        part, arms, samples = result_rows(data, country, history, scale,
                                         float(low.maternal_coverage), coverage)
        if scale == 1.:
            for row in part:
                expected = previous[previous.horizon == row['horizon']].iloc[0]
                for metric in ('dalys', 'infant_cases', 'infant_deaths'):
                    np.testing.assert_allclose(row['low_' + metric], expected[metric],
                                               rtol=2e-8, atol=2e-4)
        rows.extend(part)
        arm_rows.extend(arms)
        prefix = f'scale_{scale:g}_'
        draws.update({prefix + k: v for k, v in samples.items()})
        trajectories[prefix + 'onsets'] = raw
    paths = [folder / 'summary.csv', folder / 'arms.csv', folder / 'paired_draws.npz',
             folder / 'trajectories.npz']
    pd.DataFrame(rows).to_csv(paths[0], index=False)
    pd.DataFrame(arm_rows).to_csv(paths[1], index=False)
    np.savez_compressed(paths[2], **draws)
    np.savez_compressed(paths[3], **trajectories)
    input_files = sorted({str(path.relative_to(ROOT)) for path in opened
                          if path.is_relative_to(ROOT)} | {
                              str(path.relative_to(ROOT)) for path in
                              (Path(__file__).resolve(), Path(rev.__file__).resolve(),
                               Path(r4.__file__).resolve(), Path(design.__file__).resolve(),
                               Path(rev.ev.__file__).resolve(), prior_manifest, previous_path)
                              if path.is_relative_to(ROOT)})
    record = dict(country=country, history=history, fit_directory=info['fit_directory'],
                  seconds=time.monotonic() - started, new_projection_integrations=8,
                  history_replays=1, reproduction_checks=checks,
                  low_maternal_coverage=float(low.maternal_coverage), high_maternal_coverage=.9,
                  nasal_coverage=coverage, mean_nasal_protection_years=3.,
                  input_files=input_files,
                  output_files=[str(path.relative_to(ROOT)) for path in paths])
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return {k: record[k] for k in ('country', 'history', 'seconds', 'new_projection_integrations')}


def condition_summary(frame):
    return dict(conditions=len(frame), negative_mean_interaction=int((frame.interaction_dalys < 0).sum()),
                positive_mean_interaction=int((frame.interaction_dalys > 0).sum()),
                interval_below_zero=int((frame.interaction_dalys_q975 < 0).sum()),
                interval_above_zero=int((frame.interaction_dalys_q025 > 0).sum()),
                interval_includes_zero=int(((frame.interaction_dalys_q025 <= 0)
                                           & (frame.interaction_dalys_q975 >= 0)).sum()),
                nonpositive_low_nasal_gain=int((frame.low_dalys <= 0).sum()),
                nonpositive_high_nasal_gain=int((frame.high_dalys <= 0).sum()))


def summarize(countries):
    manifests = [job_folder(c, h) / 'manifest.json' for c in countries for h in HISTORIES]
    records = [load_manifest(p) for p in manifests]
    inputs = [p.parent / name for p in manifests for name in ('summary.csv', 'arms.csv')]
    frame = pd.concat([pd.read_csv(p.parent / 'summary.csv') for p in manifests], ignore_index=True)
    frame = frame.sort_values(['country', 'history', 'protection_scale', 'horizon']).reset_index(drop=True)
    arms = pd.concat([pd.read_csv(p.parent / 'arms.csv') for p in manifests], ignore_index=True)
    complete = OUT / 'maternal_refit_interaction.csv'
    armfile = OUT / 'maternal_refit_arms.csv'
    reader = OUT / 'maternal_refit_reader.csv'
    markdown = OUT / 'maternal_refit_reader.md'
    summaryfile = OUT / 'summary.json'
    frame.to_csv(complete, index=False)
    arms.to_csv(armfile, index=False)
    columns = ['country', 'history', 'protection_scale', 'horizon', 'low_maternal_coverage',
               'nasal_coverage', 'low_dalys', 'high_dalys', 'interaction_dalys',
               'interaction_dalys_q025', 'interaction_dalys_q975',
               'interaction_infant_yll', 'interaction_yld',
               'interaction_infant_cases', 'interaction_infant_deaths', 'percent_nasal_daly_reduction']
    display = frame[columns]
    display.to_csv(reader, index=False)
    labels = ['Setting', 'History', 'Maternal protection scale', 'Years', 'Baseline maternal coverage',
              'Nasal coverage', 'Nasal DALYs at baseline maternal coverage', 'Nasal DALYs at 90% maternal coverage',
              'Interaction DALYs', 'Lower 95% UI', 'Upper 95% UI', 'Interaction infant YLL',
              'Interaction all-age YLD', 'Interaction infant cases', 'Interaction infant deaths', 'Nasal DALY reduction, %']
    lines = ['| ' + ' | '.join(labels) + ' |', '| ' + ' | '.join(['---'] * len(labels)) + ' |']
    for row in display.itertuples(index=False, name=None):
        cells = ['—' if pd.isna(v) else f'{v:.6g}' if isinstance(v, (float, np.floating))
                 else str(v).replace('_', ' ') for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    note = ('Each row contrasts the nasal benefit at 90% maternal coverage with its benefit at baseline maternal coverage. '
            'Negative interaction means that higher maternal coverage reduces additional nasal benefit; all signs are retained. '
            'Standalone nasal coverage is 70% or the higher baseline adolescent coverage, with the reference equal-log-allocation '
            'three-year profile. Maternal symptom protection is retained or halved prospectively; historical states and timely '
            'protective entry are unchanged. Each fitted history has its own parameters and ascertainment. '
            'DALYs and YLL are discounted, infant clinical events are undiscounted, and intervals use paired clinical draws. '
            'Percentages are ratios of conditional means and are omitted when baseline-maternal nasal benefit is nonpositive. '
            'The fitted histories and protection scales are unweighted conditions.\n\n')
    markdown.write_text(note + '\n'.join(lines) + '\n')
    summary = dict(countries=list(countries), histories=list(HISTORIES), scales=list(SCALES),
                   overall=condition_summary(frame),
                   by_scale={str(s): condition_summary(frame[np.isclose(frame.protection_scale, s)]) for s in SCALES},
                   by_horizon={str(h): condition_summary(frame[frame.horizon == h]) for h in HORIZONS},
                   new_projection_integrations=sum(r['new_projection_integrations'] for r in records),
                   history_replays=len(records), cumulative_job_seconds=sum(r['seconds'] for r in records),
                   scope='Unweighted fitted-history and maternal-protection conditions; paired clinical intervals.')
    summaryfile.write_text(json.dumps(summary, indent=2) + '\n')
    record = dict(input_files=[str(path.relative_to(ROOT)) for path in
                               (inputs + manifests + [Path(__file__).resolve()])],
                  output_files=[str(path.relative_to(ROOT)) for path in
                                (complete, armfile, reader, markdown, summaryfile)])
    (OUT / 'summary_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return summary


def check(countries):
    for country in countries:
        for history in HISTORIES:
            folder = job_folder(country, history)
            record = load_manifest(folder / 'manifest.json')
            expected = json.loads(history_path(country, history).read_text())
            if record['fit_directory'] != expected['fit_directory']:
                raise ValueError('Saved fitted history mismatch')
            rows = pd.read_csv(folder / 'summary.csv')
            if set(zip(rows.protection_scale, rows.horizon)) != {(s, h) for s in SCALES for h in HORIZONS} or len(rows) != 4:
                raise ValueError('Incomplete protection/horizon results')
            with np.load(folder / 'paired_draws.npz') as arrays:
                for row in rows.itertuples():
                    prefix = f'scale_{row.protection_scale:g}_'
                    for metric in METRICS:
                        values = arrays[f'{prefix}{metric}_arms_{row.horizon}']
                        for label, actual in zip(('low', 'high', 'interaction'), paired_interaction(values)):
                            saved = arrays[f'{prefix}{metric}_{label}_{row.horizon}']
                            np.testing.assert_array_equal(actual, saved)
                            for name, value in rev.describe(actual, f'{label}_{metric}').items():
                                np.testing.assert_allclose(value, getattr(row, name), rtol=1e-12, atol=1e-9)
                    low = arrays[f'{prefix}dalys_low_{row.horizon}']
                    high = arrays[f'{prefix}dalys_high_{row.horizon}']
                    np.testing.assert_allclose(percent_reduction(low, high), row.percent_nasal_daly_reduction,
                                               rtol=1e-12, atol=1e-9, equal_nan=True)
                    np.testing.assert_allclose(row.interaction_dalys, row.interaction_yll + row.interaction_yld,
                                               rtol=1e-12, atol=1e-8)
    load_manifest(OUT / 'summary_manifest.json')
    frame = pd.read_csv(OUT / 'maternal_refit_interaction.csv')
    if len(frame) != len(countries) * 8 or frame.duplicated(['country', 'history', 'protection_scale', 'horizon']).any():
        raise ValueError('Incomplete or duplicated summary')
    print(f'Verified {len(countries) * 2} fitted-history jobs, {len(frame)} conditions and '
          f'{len(countries) * 16} prospective integrations', flush=True)


def set_output_dir(path):
    global OUT
    OUT = Path(path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'summary', 'check'), default='all')
    parser.add_argument('--countries', nargs='+', choices=rev.COUNTRIES, default=list(rev.COUNTRIES))
    parser.add_argument('--workers', type=int, default=4)
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
            futures = [pool.submit(compute_job, c, h) for c in args.countries for h in HISTORIES]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    print(json.dumps(summarize(args.countries), indent=2), flush=True)
    check(args.countries)


if __name__ == '__main__':
    main()
