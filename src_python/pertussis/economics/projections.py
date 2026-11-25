"""Validated paired trajectory inputs for economic PSA.

Posterior tasks are already resampled/equally weighted by the projection runner.
Reporting parameters and modular mortality risks follow the same transmission
row. Repeated economic draws are not counted as new transmission particles.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..grid import AgeGrid

STRATEGIES = ('baseline', 'maternal_scaleup', 'adolescent_one_time_booster')
LOGIT_COLUMNS = ['reporting_logit_<1', 'reporting_logit_1-4', 'reporting_logit_5-9',
                 'reporting_logit_10-19', 'reporting_logit_20-59', 'reporting_logit_60+']


def fit_is_converged(completion, row):
    # Optimizer success can be true after a function-tolerance stop even when
    # the stationarity check fails. Preserve the projection runner's label.
    gradient = float(row.get('max_abs_projected_gradient', np.inf))
    return bool(completion.get('fit_success', False) and
                'diagnostic' not in completion.get('inference_status', '') and
                np.isfinite(gradient) and gradient < .05)


def projection_fit_is_converged(completion, row):
    """Separate historical fit convergence from prospective structural status."""
    if (completion.get('inference_status') == 'structural_diagnostic_point' and
            completion.get('history_invariant') is True):
        return fit_is_converged({'fit_success': completion.get('fit_success', False)}, row)
    return fit_is_converged(completion, row)


def read_trajectory(folder, years):
    folder = Path(folder)
    onsets = pd.read_csv(folder / 'annual_age_onsets.csv.gz')
    onsets = onsets[onsets.year.isin(years) & onsets.strategy_id.isin(STRATEGIES)]
    if onsets.draw_id.nunique() != 1:
        raise ValueError('trajectory must contain exactly one transmission draw')
    n_bins = AgeGrid().n_bins
    index = pd.MultiIndex.from_product([STRATEGIES, years, range(n_bins)],
                                      names=['strategy_id', 'year', 'bin_index'])
    keys = index.names
    if onsets.duplicated(keys).any():
        raise ValueError('duplicate onset cells')
    table = onsets.set_index(keys).reindex(index)
    values = table[['typical_onsets', 'mild_onsets']].to_numpy(float)
    if len(onsets) != len(index) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('incomplete or invalid strategy/year/age onset grid')
    receipts = pd.read_csv(folder / 'receipts.csv.gz')
    receipts = receipts[receipts.year.isin(years) & receipts.strategy_id.isin(STRATEGIES)]
    if receipts.duplicated(['strategy_id', 'year']).any():
        raise ValueError('duplicate receipt cells')
    doses = np.zeros((len(STRATEGIES), len(years)))
    for s, sid in enumerate(STRATEGIES):
        subset = receipts[receipts.strategy_id.eq(sid)].set_index('year')
        if sid == 'baseline' and subset.empty:
            continue
        v = subset.reindex(years).additional_receipts.to_numpy(float)
        if not np.isfinite(v).all() or (v < 0).any():
            raise ValueError('incomplete or invalid receipt years')
        if sid == 'baseline' and np.any(v != 0):
            raise ValueError('baseline additional receipts must be zero')
        doses[s] = v
    shape = (len(STRATEGIES), len(years), n_bins)
    return dict(typical=values[:, 0].reshape(shape), mild=values[:, 1].reshape(shape), doses=doses)


def check_hashes(hashes, repo_root):
    for name, digest in hashes.items():
        if name.startswith(('src_python/pertussis/model/', 'src_python/pertussis/inputs/',
                            'data/processed/', 'config/core2026/programmes/')) or name in (
                                'src_python/pertussis/grid.py', 'src_python/pertussis/states.py'):
            if hashlib.sha256((Path(repo_root) / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'stale transmission source: {name}')


def conditional_risk_arrays(risks, draw_ids, rng, country, n_bins):
    scenario = {'Australia': 'ACR_reference', 'England': 'reference'}[country]
    risks = risks[risks.risk_scenario.eq(scenario)]
    if 'profile' in risks and not risks.profile.eq(country).all():
        raise ValueError('mortality risk country mismatch')
    if country == 'Australia' and 'age_start_months' not in risks:
        # Published ACR split, also used by load_reference_risks.
        risks = risks.assign(age_start_months=risks.risk_age.map({0: 0, 1: 2}),
                             age_end_months_exclusive=risks.risk_age.map({0: 2, 1: 12}))
    out = [np.zeros((len(draw_ids), n_bins)) for _ in range(2)]
    selected_risks = np.zeros(len(draw_ids), dtype=int)
    groups = {int(k): g for k, g in risks.groupby('draw_id')}
    for i, draw in enumerate(draw_ids):
        if int(draw) not in groups:
            raise ValueError(f'missing conditional mortality risks for transmission draw {draw}')
        group = groups[int(draw)]
        rid = rng.choice(group.risk_draw_id.unique())
        selected_risks[i] = rid
        rows = group[group.risk_draw_id.eq(rid)].sort_values('age_start_months')
        covered = []
        for row in rows.itertuples():
            a, b = int(row.age_start_months), int(row.age_end_months_exclusive)
            if not 0 <= a < b <= 12:
                raise ValueError('invalid conditional mortality age bounds')
            covered.extend(range(a, b))
            for arr, value in zip(out, (row.p_death_given_typical, row.p_death_given_mild)):
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('invalid mortality risk')
                arr[i, a:b] = value
        if covered != list(range(12)):
            raise ValueError('mortality ages must partition infancy without overlaps')
    return *out, selected_risks


def load_posterior(country, config, years, n, rng, repo_root):
    root = Path(repo_root)
    folder = root / config['projection_dir']
    spec = json.loads((folder / 'run_spec.json').read_text())
    if not spec.get('quality', {}).get('usable_local_ensemble', False):
        raise ValueError('transmission ensemble quality did not pass')
    fit_spec = spec['parent']['parent']
    if fit_spec['country'] != country:
        raise ValueError('transmission country mismatch')
    check_hashes(fit_spec['source_hashes'], root)
    check_hashes(spec.get('source_hashes', {}), root)
    tasks = json.loads((folder / 'complete.json').read_text())['completed_tasks']
    if len(tasks) < 2 or len(set(tasks)) != len(tasks) or len(tasks) != spec['draws']:
        raise ValueError('incomplete posterior projection tasks')
    parameters = pd.read_csv(root / config['parameter_file'])
    selected = rng.integers(0, len(tasks), size=n)
    loaded = {}
    for idx in np.unique(selected):
        task = folder / tasks[idx]
        meta = json.loads((task / 'complete.json').read_text())
        if meta['profile'] != country or meta['start_year'] != years[0]:
            raise ValueError('posterior task country/start mismatch')
        if not meta['inference_status'].startswith('training_local_posterior'):
            raise ValueError('posterior task does not declare posterior provenance')
        source_row = int(meta['source_parameter_row'])
        if not 0 <= source_row < len(parameters):
            raise ValueError('posterior parameter row outside table')
        row = parameters.iloc[source_row]
        # Confirm that the supplied table actually generated this task.
        for name in ('R0', 'aP_years', 'tdap_years', 'natural_immunity_years'):
            if not np.isclose(row[name], meta['parameters'][name], rtol=1e-10):
                raise ValueError(f'posterior parameter table mismatch: {name}')
        trajectory = read_trajectory(task, years)
        loaded[idx] = (trajectory, meta, row)
    out = {key: np.stack([loaded[i][0][key] for i in selected]) for key in ('typical', 'mild', 'doses')}
    out.update(years=years, logits=np.stack([loaded[i][2][LOGIT_COLUMNS].to_numpy(float) for i in selected]),
        mild_ratio=np.array([loaded[i][2].breakthrough_reporting_ratio for i in selected]),
        transmission_draw_id=np.array([loaded[i][1]['draw_id'] for i in selected]),
        source_parameter_row=np.array([loaded[i][1]['source_parameter_row'] for i in selected]),
        projection_dir=str(folder), fit_dir=str(spec['parent']['parent_fit']), fit_converged=True,
        transmission_uncertainty='local_parameter_posterior', available_transmission_draws=len(tasks))
    if not np.isfinite(out['logits']).all() or not np.isfinite(out['mild_ratio']).all():
        raise ValueError('nonfinite reporting parameters')
    if country in ('Australia', 'England'):
        risks = pd.read_csv(root / config['risk_file'])
        identities = risks[['draw_id', 'source_parameter_row']].drop_duplicates()
        expected = pd.DataFrame(dict(draw_id=out['transmission_draw_id'],
                                     expected_row=out['source_parameter_row'])).drop_duplicates()
        matched = expected.merge(identities, on='draw_id', how='left', validate='one_to_one')
        if not matched.expected_row.eq(matched.source_parameter_row).all():
            raise ValueError('mortality risks were calibrated to different transmission parameters')
        out['risk_typical'], out['risk_mild'], out['risk_draw_id'] = conditional_risk_arrays(
            risks, out['transmission_draw_id'], rng, country, out['typical'].shape[-1])
    return out
