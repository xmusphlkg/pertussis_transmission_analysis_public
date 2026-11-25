"""Conditional net-health-benefit summaries of the saved five-strategy set.

Run from the repository root with python3 -m scripts.summarize_eight_country_nhb_20261103.
Thresholds are existing research anchors, not asserted national DALY thresholds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_eight_country_candidate_frontiers_20261103 import ROOT, BASE, COUNTRIES, sha
from src_python.pertussis.economics.loader import load_costs
from src_python.pertussis.economics.evaluate import thresholds, imputed_threshold_types

ARMS = ('00', 'M0', '0A', 'MA', 'current')
COSTS = ROOT/'config/core2026/economics/costs_unified_provenance_20261102.yaml'
FRONT = BASE/'eight_country_candidate_frontiers_20261103'
COEFF = BASE/'v2_eight_country_corrected_japan_price_burden_20261102/coefficients.csv'


def paired_arrays(frame, expected_draws=1000):
    """Require identical draw identifiers; never pair by incidental row order."""
    f = frame[frame.scope.eq('all_ages')].copy()
    if set(f.strategy_id) != set(ARMS[1:]) or f.duplicated(['strategy_id', 'draw']).any():
        raise ValueError('Expected four unique strategy-by-draw grids')
    cols = ['draw', 'monthly_dalys_averted', 'monthly_incremental_cost']
    if f[cols].isna().any().any() or not np.isfinite(f[cols]).all().all():
        raise ValueError('Nonfinite draw, effect or cost')
    ids = sorted(f.draw.unique())
    if len(ids) != expected_draws or any(set(g.draw) != set(ids) for _, g in f.groupby('strategy_id')):
        raise ValueError('Unmatched draw identifiers')
    arrays = []
    for col in cols[1:]:
        values = f.pivot(index='draw', columns='strategy_id', values=col).loc[ids, list(ARMS[1:])].to_numpy()
        arrays.append(np.column_stack([np.zeros(len(ids)), values]))
    return ids, arrays[0], arrays[1]


def summarize_nhb(effects, costs, lam, arms=ARMS):
    """Expected-NHB decision, draw frequencies and expected opportunity losses.

Quantiles describe the supplied draws, not confidence intervals for the mean.
Tied draw winners receive fractional credit; inclusive frequencies are also kept.
"""
    effects, costs = np.asarray(effects, float), np.asarray(costs, float)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError('NHB requires a finite positive threshold')
    if effects.ndim != 2 or effects.shape != costs.shape or effects.shape[1] != len(arms) or not len(effects):
        raise ValueError('Expected matched nonempty draw-by-strategy arrays')
    if not np.isfinite(effects).all() or not np.isfinite(costs).all():
        raise ValueError('Finite health and cost arrays required')
    nhb = effects - costs/lam
    means = nhb.mean(axis=0)
    # Scale-aware roundoff tolerance, small enough not to erase meaningful gaps.
    tolerance = 64*np.finfo(float).eps*np.maximum(1., np.abs(nhb).max(axis=1))
    draw_best = nhb.max(axis=1)
    tied = draw_best[:, None] - nhb <= tolerance[:, None]
    fractions = tied/tied.sum(axis=1)[:, None]
    mean_tolerance = 64*np.finfo(float).eps*max(1., np.abs(nhb).max())
    winning = means.max()-means <= mean_tolerance
    losses = draw_best[:, None]-nhb
    order = np.sort(means)[::-1]
    other = means[~winning]
    best_indices = np.flatnonzero(winning)
    summary = dict(expected_nhb_optimal_strategies='|'.join(arms[i] for i in best_indices),
        maximum_expected_nhb=float(means.max()), second_rank_gap=float(order[0]-order[1]) if len(order)>1 else np.nan,
        gap_to_best_non_tied_strategy=float(means.max()-other.max()) if len(other) else np.nan,
        best_non_tied_strategies='|'.join(arms[i] for i in range(len(arms)) if not winning[i] and other.max()-means[i] <= mean_tolerance) if len(other) else '',
        conditional_expected_loss_of_mean_optimum=float(losses[:, best_indices[0]].mean()),
        mean_of_drawwise_best_nhb=float(draw_best.mean()), draw_count=len(effects))
    rows = []
    for i, sid in enumerate(arms):
        lo, hi = np.quantile(nhb[:, i], [.025, .975], method='linear')
        rows.append(dict(strategy_id=sid, expected_nhb=float(means[i]), nhb_q025=float(lo), nhb_q975=float(hi),
            conditional_optimal_frequency_including_ties=float(tied[:, i].mean()),
            conditional_optimal_frequency_fractional_ties=float(fractions[:, i].mean()),
            expected_nhb_optimal=bool(winning[i]), expected_nhb_gap_to_best=float(means.max()-means[i]),
            conditional_expected_opportunity_loss=float(losses[:, i].mean()),
            mean_incremental_cost=float(costs[:, i].mean()), mean_dalys_averted=float(effects[:, i].mean())))
    np.testing.assert_allclose(fractions.mean(axis=0).sum(), 1., atol=1e-12)
    np.testing.assert_allclose(losses.mean(axis=0), draw_best.mean()-means, rtol=1e-10, atol=1e-9)
    return pd.DataFrame(rows), summary


def anchor_status(kind, imputed):
    if imputed:
        return 'income_group_ratio_imputation_not_country_estimate'
    return {'gdp_multiple': 'GDP_reference_not_recommended_decision_threshold',
            'national': 'existing_QALY_reference_used_as_DALY_sensitivity_proxy',
            'opportunity_cost': 'existing_research_anchor_with_country_specific_evidence_status'}[kind]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE/'eight_country_nhb_results_20261103')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_audit = BASE/'research_summary_audit_20261103/audit.json'
    audit = json.loads(source_audit.read_text())
    if audit['status'] != 'passed_saved_data_arithmetic_not_model_validation':
        raise ValueError('Unexpected arithmetic audit')
    front_meta = json.loads((FRONT/'metadata.json').read_text())
    for name, digest in front_meta['sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frontier input changed: '+name)
    doc = load_costs(COSTS)
    if doc['reference_currency'] != 'intl_2024':
        raise ValueError('Expected 2024 international dollars')
    coeff = pd.read_csv(COEFF, dtype={'strategy_id': str})
    expected_keys = set(coeff[['country', 'threshold_type']].itertuples(index=False, name=None))
    anchors = []
    for country in sorted(COUNTRIES):
        configured = {t['type']: t for t in doc['countries'][country]['macro']['thresholds']}
        for kind, lam in thresholds(doc, country).items():
            ref = coeff[(coeff.country == country) & (coeff.threshold_type == kind)].threshold
            if len(ref) != 4 or ref.nunique() != 1:
                raise ValueError('Missing/ambiguous saved threshold coefficient')
            np.testing.assert_allclose(lam, ref.iloc[0], rtol=1e-12, atol=1e-8)
            imputed = kind in imputed_threshold_types(doc, country)
            entry = configured.get(kind, {})
            anchors.append(dict(country=country, threshold_type=kind, threshold=lam,
                threshold_status=anchor_status(kind, imputed), threshold_imputed=imputed,
                original_source_outcome_unit=entry.get('outcome_unit', 'not_explicit_in_config'),
                original_source_description=entry.get('source', {}).get('citation', 'Income-group mean opportunity-cost/GDP ratio applied by evaluate.thresholds'),
                original_source_url=entry.get('source', {}).get('url', ''), threshold_role='existing_research_anchor_not_official_DALY_threshold'))
    anchors = pd.DataFrame(anchors)
    if set(anchors[['country', 'threshold_type']].itertuples(index=False, name=None)) != expected_keys:
        raise ValueError('Threshold grid mismatch')
    means = pd.read_csv(FRONT/'mean_outcomes.csv', dtype={'strategy_id': str})
    intervals = pd.read_csv(FRONT/'threshold_intervals.csv')
    intervals = intervals[intervals.decision_set.eq('standardized_plus_encoded_current')]
    paths = [Path(__file__).resolve(), ROOT/'tests/core2026/test_nhb_results_20261103.py', COSTS, COEFF,
             source_audit, FRONT/'metadata.json', FRONT/'mean_outcomes.csv', FRONT/'threshold_intervals.csv',
             ROOT/'src_python/pertussis/economics/loader.py', ROOT/'src_python/pertussis/economics/evaluate.py']
    rows, decisions, probe_rows, probe_decisions = [], [], [], []
    for country in sorted(COUNTRIES):
        for horizon in (20, 30):
            suffix = '_30y' if horizon == 30 else ''
            prefix = 'v2_japan_official_age_monthly_economics' if country == 'Japan' else 'v2_monthly_candidate_economics'
            path = BASE/f'{prefix}{suffix}_20261102/{country}_paired.csv.gz'
            if audit['sha256'][str(path.relative_to(ROOT))] != sha(path):
                raise ValueError('Paired source hash mismatch')
            paths.append(path)
            frame = pd.read_csv(path, dtype={'strategy_id': str})
            if not frame.country.eq(country).all():
                raise ValueError('Country mismatch')
            ids, effects, costs = paired_arrays(frame)
            reference = means[(means.country == country) & (means.horizon == horizon)].set_index('strategy_id').loc[list(ARMS)]
            np.testing.assert_allclose(effects.mean(axis=0), reference.dalys_averted, rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(costs.mean(axis=0), reference.incremental_cost, rtol=1e-10, atol=1e-6)
            tags = dict(country=country, horizon=horizon, mortality_scope=reference.mortality_scope.iloc[0],
                fit_scope=reference.fit_scope.iloc[0], noninfant_mortality_estimated=country in ('Brazil', 'Thailand'),
                health_denominator='included_DALY_not_complete_all_age_DALY' if country not in ('Brazil', 'Thailand') else 'all_age_DALY',
                decision_set='standardized_plus_encoded_current', reference_strategy='00',
                uncertainty_scope='1000_clinical_cost_draws_conditional_on_fixed_transmission',
                threshold_unit='2024_international_dollars_per_included_DALY', draw_count=len(ids))
            def evaluate(lam, extra, table_list, decision_list):
                detail, decision = summarize_nhb(effects, costs, lam)
                # A common paired rebase must preserve every decision/loss/frequency.
                rebased, shifted = summarize_nhb(effects-effects[:, [-1]], costs-costs[:, [-1]], lam)
                assert decision['expected_nhb_optimal_strategies'] == shifted['expected_nhb_optimal_strategies']
                for col in ('conditional_expected_opportunity_loss', 'conditional_optimal_frequency_fractional_ties', 'expected_nhb_gap_to_best'):
                    np.testing.assert_allclose(detail[col], rebased[col], rtol=1e-10, atol=1e-8)
                detail['expected_nhb_vs_encoded_current'] = rebased.expected_nhb
                detail['nhb_vs_encoded_current_q025'] = rebased.nhb_q025
                detail['nhb_vs_encoded_current_q975'] = rebased.nhb_q975
                detail['exact_zero_effect_and_cost_all_draws'] = np.all((effects == 0) & (costs == 0), axis=0)
                table_list.append(detail.assign(threshold=lam, **tags, **extra))
                decision_list.append({**decision, 'threshold':lam, **tags, **extra})
            for anchor in anchors[anchors.country.eq(country)].to_dict('records'):
                extra = {k: v for k, v in anchor.items() if k not in ('country', 'threshold')}
                evaluate(anchor['threshold'], extra, rows, decisions)
            boundaries = intervals[(intervals.country == country) & (intervals.horizon == horizon) & intervals.lambda_lower.gt(0)]
            for boundary in boundaries.itertuples():
                for multiplier in (.5, 1., 2.):
                    extra = dict(threshold_type='frontier_boundary_probe', threshold_status='diagnostic_not_country_threshold',
                        switch_threshold=boundary.lambda_lower, boundary_multiplier=multiplier,
                        switch_from=boundary.previous_strategies, switch_to=boundary.strategies)
                    evaluate(boundary.lambda_lower*multiplier, extra, probe_rows, probe_decisions)
    first_switch = intervals[intervals.lambda_lower.gt(0)].sort_values('lambda_lower').groupby(['country','horizon']).first().reset_index()
    distance = first_switch[['country','horizon','lambda_lower','strategies']].rename(columns={
        'lambda_lower':'first_switch_threshold','strategies':'first_nonzero_frontier_strategies'}).merge(
        anchors[['country','threshold_type','threshold','threshold_status']], on='country', validate='many_to_many')
    distance['first_switch_to_anchor_ratio'] = distance.first_switch_threshold/distance.threshold
    args.output_dir.mkdir(parents=True)
    tables = {'threshold_anchors.csv': anchors, 'strategy_nhb.csv': pd.concat(rows, ignore_index=True),
              'decisions.csv': pd.DataFrame(decisions), 'boundary_probe_nhb.csv': pd.concat(probe_rows, ignore_index=True),
              'boundary_probe_decisions.csv': pd.DataFrame(probe_decisions), 'threshold_distance.csv':distance}
    for name, table in tables.items():
        table.to_csv(args.output_dir/name, index=False)
    meta = dict(status='passed_conditional_saved_draw_NHB_checks_not_model_validation', countries=sorted(COUNTRIES),
        horizons=[20, 30], strategies=list(ARMS), threshold_anchors=len(anchors), table_rows={k:len(v) for k,v in tables.items()},
        objective='maximize mean(DALYs_averted - incremental_cost/threshold)', quantiles='2.5th and 97.5th empirical linear quantiles of 1000 paired clinical/cost draws; not CIs on means',
        threshold_handling='load_costs and evaluate.thresholds use stored value_intl_2024, with income-group imputation where absent; exact match to saved coefficients verified; no USD value substituted',
        tie_handling='all mean ties preserved; inclusive draw frequency and fractional tie frequency both reported; second_rank_gap includes ties, gap_to_best_non_tied_strategy excludes mean ties',
        opportunity_loss='mean(drawwise maximum NHB minus strategy NHB); conditional on supplied clinical/cost uncertainty and finite five-strategy set',
        boundary_probes='0.5x, 1x and 2x each positive efficient-frontier switch threshold; diagnostics, not probabilities or country threshold recommendations',
        zero_reference='00 removes prospective maternal and target-age adolescent programmes, keeps other routine vaccination; current is saved encoded schedule, not verified contemporary policy',
        scope='70% standardized coverage for M0/0A/MA. Fixed transmission point, no transmission posterior. No independent maternal/adolescent coverage grid or current-plus add-on arms. Six countries omit noninfant YLL. Research threshold anchors include QALY-to-DALY proxies and imputation. No national policy recommendation.',
        sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (args.output_dir/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k in ('status','table_rows','threshold_anchors')}, indent=2))
    print(tables['decisions.csv'][['country','horizon','threshold_type','threshold','expected_nhb_optimal_strategies','gap_to_best_non_tied_strategy','conditional_expected_loss_of_mean_optimum']].to_string(index=False))


if __name__ == '__main__':
    main()
