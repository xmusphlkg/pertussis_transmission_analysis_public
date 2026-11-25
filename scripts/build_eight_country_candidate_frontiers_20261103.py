"""Finite-set mean cost/health frontiers, with no chosen willingness-to-pay.

These exploratory choices include policy withdrawal. They are not validated
incremental additions to the encoded national programme or recommendations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/core2026'
COUNTRIES = {'Australia', 'Brazil', 'China', 'England', 'Japan', 'New_Zealand', 'Sweden', 'Thailand'}
ARMS = ('M0', '0A', 'MA', 'current')


def frontier(frame):
    """Solve every line's upper-envelope interval for lambda >= 0 exactly.

Equivalent cost/health alternatives are retained. A collinear strategy optimal
only at a switch is retained as a boundary tie, not called extended dominated.
    """
    if frame.empty or frame.strategy_id.duplicated().any():
        raise ValueError('Unique nonempty strategy set required')
    effects = frame.dalys_averted.to_numpy(float)
    costs = frame.incremental_cost.to_numpy(float)
    if not np.isfinite(effects).all() or not np.isfinite(costs).all():
        raise ValueError('Finite costs and health effects required')
    records = []
    for i, sid in enumerate(frame.strategy_id):
        lower, upper = 0., np.inf
        strict, weak, identical = [], [], []
        feasible = True
        for j, other in enumerate(frame.strategy_id):
            if i == j:
                continue
            if costs[j] < costs[i] and effects[j] > effects[i]:
                strict.append(other)
            elif costs[j] <= costs[i] and effects[j] >= effects[i] and (costs[j] < costs[i] or effects[j] > effects[i]):
                weak.append(other)
            elif costs[j] == costs[i] and effects[j] == effects[i]:
                identical.append(other)
            de, dc = effects[i] - effects[j], costs[i] - costs[j]
            if de > 0:
                lower = max(lower, dc / de)
            elif de < 0:
                upper = min(upper, dc / de)
            elif dc > 0:
                feasible = False
        feasible = feasible and upper >= lower
        dominated = bool(strict or weak)
        status = ('strictly_dominated' if strict else 'weakly_dominated' if weak else
                  'extended_dominated' if not feasible else 'boundary_only' if lower == upper else 'frontier')
        records.append(dict(strategy_id=sid, dalys_averted=effects[i], incremental_cost=costs[i],
            status=status, never_optimal=not feasible,
            strict_dominators='|'.join(strict), weak_dominators='|'.join(weak), equivalent_strategies='|'.join(identical),
            pareto_nondominated=not dominated, extended_dominated=(not dominated and not feasible),
            supported_for_nonnegative_lambda=feasible,
            supported_only_at_boundary=bool(feasible and lower == upper),
            lambda_lower=lower if feasible else np.nan, lambda_upper=upper if feasible else np.nan))
    result = pd.DataFrame(records)
    # Independently sample every region partitioned by all pairwise crossings.
    crossings = {0.}
    for i in range(len(costs)):
        for j in range(i):
            if effects[i] != effects[j]:
                point = (costs[i]-costs[j])/(effects[i]-effects[j])
                if point >= 0:
                    crossings.add(float(point))
    knots = sorted(crossings)
    probes = knots + [(a+b)/2 for a,b in zip(knots[:-1],knots[1:])] + [knots[-1] + max(1.,abs(knots[-1]))]
    for lam in probes:
        nmb = lam * effects - costs
        best = nmb.max()
        scale = max(1., np.abs(costs).max(), np.abs(lam * effects).max())
        numerical_nmb_tolerance = 32 * np.finfo(float).eps * scale
        represented = []
        for row in result.itertuples():
            at_lower = np.isclose(lam, row.lambda_lower, rtol=64*np.finfo(float).eps, atol=64*np.finfo(float).eps)
            at_upper = np.isclose(lam, row.lambda_upper, rtol=64*np.finfo(float).eps, atol=64*np.finfo(float).eps)
            if row.supported_for_nonnegative_lambda and (row.lambda_lower <= lam or at_lower) and (lam <= row.lambda_upper or at_upper):
                index = frame.strategy_id.tolist().index(row.strategy_id)
                represented.append(index)
                if abs(nmb[index]-best) > 1e-10 * scale:
                    raise AssertionError('Envelope interval fails direct NMB comparison')
        if not represented:
            raise AssertionError('Envelope has an uncovered nonnegative threshold')
        actual_winners = set(np.flatnonzero(np.abs(nmb-best) <= numerical_nmb_tolerance))
        if not actual_winners.issubset(represented):
            raise AssertionError('Envelope omitted a direct NMB maximizer')
    # Nonzero-width supported intervals form the efficient sequence. Economic
    # duplicates share a segment, and all boundary winners are listed separately.
    supported = result[result.supported_for_nonnegative_lambda & ~result.supported_only_at_boundary]
    segments = []
    for (_, _, lo, hi), group in supported.groupby(['dalys_averted','incremental_cost','lambda_lower','lambda_upper'], dropna=False):
        segments.append(dict(strategies='|'.join(group.strategy_id), dalys_averted=float(group.dalys_averted.iloc[0]),
            incremental_cost=float(group.incremental_cost.iloc[0]), lambda_lower=float(lo), lambda_upper=float(hi)))
    segments.sort(key=lambda r:r['lambda_lower'])
    for i, row in enumerate(segments):
        row['sequence'] = i
        row['previous_strategies'] = segments[i-1]['strategies'] if i else ''
        row['incremental_cost_per_daly_from_previous'] = ((row['incremental_cost'] - segments[i-1]['incremental_cost']) /
            (row['dalys_averted'] - segments[i-1]['dalys_averted'])) if i else np.nan
        if i:
            np.testing.assert_allclose(row['incremental_cost_per_daly_from_previous'],row['lambda_lower'],rtol=1e-10,atol=1e-8)
        boundary = row['lambda_lower']
        winners = result[result.supported_for_nonnegative_lambda & (result.lambda_lower <= boundary) & (result.lambda_upper >= boundary)]
        row['all_ties_at_lower_boundary'] = '|'.join(winners.strategy_id)
    if not segments:
        raise AssertionError('Finite alternatives must have a nonempty envelope')
    return result, pd.DataFrame(segments)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE/'eight_country_candidate_frontiers_20261103')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = BASE/'v2_eight_country_corrected_japan_monthly_economics_horizons_20261102'
    audit_path = BASE/'research_summary_audit_20261103/audit.json'
    audited = json.loads(audit_path.read_text())
    if audited['status'] != 'passed_saved_data_arithmetic_not_model_validation':
        raise ValueError('Expected saved-data audit missing')
    meta = json.loads((source/'metadata.json').read_text())
    if meta['status'] != 'exploratory_20_30_year_economics_not_final':
        raise ValueError('Unexpected input version')
    for name, digest in meta['sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source changed: '+name)
    comparison = pd.read_csv(source/'comparison.csv',dtype={'strategy_id':str})
    selected = comparison[comparison.scope.eq('all_ages')]
    expected = {(c,h,s) for c in COUNTRIES for h in (20,30) for s in ('M0','0A','MA')}
    keys = ['country','horizon','strategy_id']
    if len(selected) != 48 or selected.duplicated(keys).any() or set(selected[keys].itertuples(index=False,name=None)) != expected:
        raise ValueError('Incomplete 8 x 2 x 3 mean grid')
    if not selected.method.eq('monthly').all():
        raise ValueError('Mixed age allocation')
    paths = [Path(__file__).resolve(), ROOT/'tests/core2026/test_candidate_frontiers_20261103.py',
             source/'comparison.csv', source/'metadata.json', audit_path]
    all_frontiers, all_segments, input_means = [], [], []
    for (country,horizon), group in selected.groupby(['country','horizon']):
        suffix = '_30y' if horizon == 30 else ''
        prefix = 'v2_japan_official_age_monthly_economics' if country == 'Japan' else 'v2_monthly_candidate_economics'
        path = BASE/f'{prefix}{suffix}_20261102/{country}_paired.csv.gz'
        paths.append(path)
        if audited['sha256'][str(path.relative_to(ROOT))] != sha(path):
            raise ValueError('Paired source changed: '+str(path))
        draw = pd.read_csv(path,dtype={'strategy_id':str})
        draw = draw[draw.scope.eq('all_ages')]
        if len(draw) != 4000 or draw.duplicated(['strategy_id','draw']).any() or set(draw.strategy_id) != set(ARMS) or not draw.country.eq(country).all():
            raise ValueError('Invalid paired grid')
        if draw[['draw','monthly_dalys_averted','monthly_incremental_cost']].isna().any().any() or not np.isfinite(draw[['monthly_dalys_averted','monthly_incremental_cost']]).all().all():
            raise ValueError('Missing or nonfinite paired values')
        identities = [set(g.draw) for _,g in draw.groupby('strategy_id')]
        if any(ids != identities[0] for ids in identities) or len(identities[0]) != 1000:
            raise ValueError('Strategies have different draws')
        means = draw.groupby('strategy_id')[['monthly_dalys_averted','monthly_incremental_cost']].mean().rename(columns={
            'monthly_dalys_averted':'dalys_averted','monthly_incremental_cost':'incremental_cost'}).reset_index()
        ref = group.set_index('strategy_id').loc[['M0','0A','MA']]
        np.testing.assert_allclose(means.set_index('strategy_id').loc[ref.index,['dalys_averted','incremental_cost']],
            ref[['dalys_averted','incremental_cost']],rtol=1e-10,atol=1e-6)
        means = pd.concat([pd.DataFrame([dict(strategy_id='00',dalys_averted=0.,incremental_cost=0.)]),means],ignore_index=True)
        if group.mortality_scope.nunique() != 1 or group.fit_scope.nunique() != 1:
            raise ValueError('Mixed model or mortality scope')
        tags = dict(country=country,horizon=horizon,mortality_scope=group.mortality_scope.iloc[0],fit_scope=group.fit_scope.iloc[0])
        input_means.append(means.assign(**tags))
        for decision_set in ('standardized_four','standardized_plus_encoded_current'):
            subset = means if decision_set.endswith('current') else means[means.strategy_id.ne('current')]
            f,s = frontier(subset)
            all_frontiers.append(f.assign(decision_set=decision_set,**tags))
            all_segments.append(s.assign(decision_set=decision_set,**tags))
    if audited['sha256'][str((source/'comparison.csv').relative_to(ROOT))] != sha(source/'comparison.csv'):
        raise ValueError('Summary source changed')
    frontiers,segments = pd.concat(all_frontiers,ignore_index=True),pd.concat(all_segments,ignore_index=True)
    args.output_dir.mkdir(parents=True)
    pd.concat(input_means,ignore_index=True).to_csv(args.output_dir/'mean_outcomes.csv',index=False)
    frontiers.to_csv(args.output_dir/'strategy_frontiers.csv',index=False)
    segments.to_csv(args.output_dir/'threshold_intervals.csv',index=False)
    metadata = dict(status='passed_finite_candidate_mean_frontier_checks_only',countries=sorted(COUNTRIES),horizons=[20,30],
        decision_sets=['standardized_four','standardized_plus_encoded_current'],mean_rows=80,frontier_rows=len(frontiers),
        interval_rows=len(segments),lambda_unit='2024 international dollars per included DALY averted',
        positive_cost='additional cost vs 00',positive_effect='DALYs averted vs 00',
        objective='Maximum expected net health benefit for lambda > 0; equivalent to maximum expected net monetary benefit. At lambda=0 only monetary benefit is defined.',
        scope='Fixed point transmission; means of paired clinical/cost draws. No threshold selected, no uncertainty in optimality quantified. Six countries omit noninfant mortality. Extended dominance is nonoptimality under a linear net-benefit objective, not permission to combine nonlinear transmission strategies as mixtures. All options may withdraw or replace existing programmes; current means saved encoded programme, not verified current policy. Not an incremental add-on optimisation or policy recommendation.',
        sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('sha256','scope')},indent=2))
    print(segments[['country','horizon','decision_set','strategies','lambda_lower','lambda_upper']].to_string(index=False))


if __name__ == '__main__':
    main()
