"""Age/time health and economic results for the integrated eight-country paper.

All projections have a common historical boundary and directly simulated MA.
This runner preserves all signed effects, exports both cost scopes, and keeps
conditional durability and unobserved-incidence scenarios explicitly labelled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src_python.pertussis.config import REPO_ROOT
from src_python.pertussis.economics.loader import load_costs, load_severity_priors
from src_python.pertussis.economics.unified_eval import (
    evaluate_common, summarise_common, annual_case_changes, price_burden_surface, health_cost_status)
from src_python.pertussis.economics.evaluate import DEATH_RISK_SOURCE
from src_python.pertussis.scenarios.mortality_extrapolation import REFERENCE_SOURCES
from src_python.pertussis.states import MODEL_VERSION
from scripts.run_core2026_common_strategies import FIT_DIRS

ROOT = REPO_ROOT
COSTS = ROOT / 'config/core2026/economics/costs_unified_20261101.yaml'
SEVERITY = ROOT / 'config/core2026/economics/severity_priors.yaml'
ENGLAND_RISK = ROOT / 'outputs/core2026/programme_scenarios_20261029/England_risk_2001_2023/risk_draws.csv.gz'


def validate_australia_risk(path):
    """Reject diagnostic, modified, or no-longer-matched point-risk draws."""
    path = Path(path)
    spec = json.loads((path.parent/'run_spec.json').read_text())
    complete = json.loads((path.parent/'complete.json').read_text())
    if (spec.get('model_version') != MODEL_VERSION or not spec.get('fit_stationary')
        or spec.get('scope') != 'fitted_point' or complete.get('status') != 'complete'):
        raise ValueError('Australian mortality risks require a stationary current-model point fit')
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec.get('risk_draws_sha256'):
        raise ValueError('Australian mortality risk draws changed since calibration')
    fit = Path(spec['fit_directory'])
    for name, key in [('run_spec.json','fit_run_spec_sha256'), ('fit_summary.csv','fit_summary_sha256')]:
        if hashlib.sha256((fit/name).read_bytes()).hexdigest() != spec.get(key):
            raise ValueError('Australian mortality fit changed since calibration')
    return spec


def reference_risk_for(country, england_period, australia_risk):
    source = DEATH_RISK_SOURCE.get(country)
    if source == 'England' and england_period == '2001_2023':
        return ENGLAND_RISK, 'reference'
    if source == 'Australia' and australia_risk is not None:
        return australia_risk, 'ACR_reference'
    return None, 'reference'


def table1_from_breakdown(breakdown):
    """One row per country/strategy/age/window, with signed averted effects.

    Infant economics attributes all programme costs and infant treatment costs.
    Other age-specific health rows do not allocate shared programme costs.
    """
    b = breakdown[(breakdown.analysis == 'main') & (breakdown.coverage == .7)
                  & (breakdown.horizon == 20) & (breakdown.projection_scenario == 'fitted')
                  & breakdown.strategy_id.isin(['M0', '0A', 'MA'])]
    keys = ['country', 'strategy_id', 'window', 'age_group']
    rows = []
    for key, part in b.groupby(keys, sort=False):
        r = dict(zip(keys, key))
        change = part[part.kind == 'change_vs_00'].set_index('measure')
        for name in ('confirmed_cases', 'all_symptomatic_onsets', 'hospitalisations', 'deaths', 'yll', 'yld', 'dalys'):
            v = change.loc[name]
            r[name + '_averted'] = -v['mean']
            r[name + '_averted_lo'], r[name + '_averted_hi'] = -v['hi'], -v['lo']
        if 'programme_plus_scope_care_cost' in change.index:
            v = change.loc['programme_plus_scope_care_cost']
            r.update(incremental_cost=v['mean'], incremental_cost_lo=v['lo'], incremental_cost_hi=v['hi'])
            r['icer'] = v['mean'] / r['dalys_averted'] if r['dalys_averted'] > 0 else np.nan
            r['health_cost_status'] = health_cost_status(v['mean'], r['dalys_averted'])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projection-root', type=Path, default=ROOT / 'outputs/core2026/common_strategies_20261101')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--n-draws', type=int, default=1000)
    p.add_argument('--countries', nargs='+', default=list(FIT_DIRS))
    p.add_argument('--include-overlap', action='store_true')
    p.add_argument('--australia-risk', type=Path, help='Point-conditional v2 Australian risk_draws.csv.gz')
    p.add_argument('--england-risk-period', choices=['2001_2023','2016_2023'], default='2001_2023',
                   help='One England risk calibration period shared by England and donor countries')
    p.add_argument('--require-matched-australia-risk', action='store_true',
                   help='Require recalibrated Australian risk rather than the historical default')
    p.add_argument('--inpatient-fallback', choices=['error', 'documented_extrapolation'], default='error',
                   help='Explicitly allow age ranges without matching episode costs to use labelled fallback assumptions')
    p.add_argument('--costs', type=Path, default=COSTS, help='Cost configuration used for this run')
    args = p.parse_args()
    if not args.costs.is_absolute():
        args.costs = ROOT / args.costs
    if not args.costs.is_relative_to(ROOT):
        p.error('--costs must be inside the repository for source-hash provenance')
    if args.require_matched_australia_risk and args.australia_risk is None:
        p.error('--require-matched-australia-risk requires --australia-risk')
    australian_risk_spec = validate_australia_risk(args.australia_risk) if args.australia_risk else None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    costs, severity = load_costs(args.costs), load_severity_priors(SEVERITY)
    life_path = ROOT / costs['global']['life_table']['csv']
    life = pd.read_csv(life_path)
    tables = {k: [] for k in ('summary', 'breakdown', 'frontier', 'decision', 'marginal',
                              'annual_cases', 'price_burden_grid', 'price_burden_boundary', 'price_burden_coefficients')}
    metadata = []
    begun = time.monotonic()
    for country in args.countries:
        scenarios = [('fitted', args.projection_root / 'horizon30', [20, 30], [.5, .7, .9])]
        scenarios += [(f'tdap_{n}y', args.projection_root / f'tdap_{n}y', [20, 30], [.7]) for n in (1, 3, 6, 10)]
        if args.include_overlap and country in ('China', 'Japan', 'Thailand'):
            scenarios += [('overlap_fitted', args.projection_root / 'overlap_fitted', [20, 30], [.7])]
            scenarios += [(f'overlap_tdap_{n}y', args.projection_root / f'overlap_tdap_{n}y', [20], [.7]) for n in (1, 6, 10)]
        for label, parent, horizons, coverages in scenarios:
            for cov in coverages:
                folder = parent / country / f'coverage_{round(cov*100):02d}'
                if not (folder / 'complete.json').exists():
                    raise ValueError(f'missing completed projection: {folder}')
                if country == 'Australia' and australian_risk_spec is not None:
                    projection_fit = json.loads((folder/'complete.json').read_text())['fit_directory']
                    if Path(projection_fit).resolve() != Path(australian_risk_spec['fit_directory']).resolve():
                        raise ValueError('Australian mortality risk and projection use different transmission fits')
                for horizon in horizons:
                    analyses = [('main', {})]
                    if cov == .7:
                        analyses += [('all_symptomatic_yld', dict(yld_scope='all_symptomatic'))]
                    if label == 'fitted' and cov == .7 and horizon == 20:
                        analyses += [('all_typical_yld', dict(yld_scope='all_typical')),
                                     ('severe_weight_0133', dict(acute_weight_override=.133))]
                    for analysis, extra in analyses:
                        breakdown_needed = label == 'fitted' and analysis == 'main'
                        risk_source = DEATH_RISK_SOURCE.get(country)
                        risk_file,risk_scenario = reference_risk_for(country,args.england_risk_period,args.australia_risk)
                        draws, bd, meta = evaluate_common(
                            country, folder, severity, costs, life, n_draws=args.n_draws,
                            seed=20261101, horizon=horizon, make_breakdown=breakdown_needed,
                            risk_file=risk_file, risk_scenario=risk_scenario,
                            inpatient_fallback=args.inpatient_fallback, **extra)
                        summary, frontier, decision, marginal = summarise_common(draws, meta)
                        tags = dict(analysis=analysis, coverage=cov, horizon=horizon, projection_scenario=label,
                                    booster_structure=meta['booster_structure'], tdap_years=meta['tdap_years'],
                                    england_risk_period=args.england_risk_period if risk_source == 'England' else None)
                        for name, frame in [('summary', summary), ('frontier', frontier), ('decision', decision),
                                            ('marginal', marginal), ('breakdown', bd)]:
                            if not frame.empty:
                                tables[name].append(frame.assign(**tags))
                        metadata.append({**meta, **tags})
                        if label == 'fitted' and analysis == 'main' and cov == .7 and horizon == 20:
                            grid, bounds, coeff = price_burden_surface(draws, meta)
                            for name, frame in [('price_burden_grid', grid), ('price_burden_boundary', bounds),
                                                ('price_burden_coefficients', coeff)]:
                                tables[name].append(frame)
                            draws.to_csv(args.output_dir / f'paired_economic_draws_{country}.csv.gz', index=False)
                        del draws
                if cov == .7 and label in ('fitted', 'overlap_fitted'):
                    tables['annual_cases'].append(annual_case_changes(country, folder).assign(projection_scenario=label))
            print(f'{country}: {label} done; elapsed {time.monotonic()-begun:.0f}s', flush=True)
        (args.output_dir / 'progress.json').write_text(json.dumps(dict(last_country=country, evaluations=len(metadata))))
    for name, frames in tables.items():
        if frames:
            tables[name] = pd.concat(frames, ignore_index=True)
            tables[name].to_csv(args.output_dir / f'{name}.csv', index=False)
    table1 = table1_from_breakdown(tables['breakdown'])
    table1.to_csv(args.output_dir / 'table1_age_window.csv', index=False)
    table1[(table1.window == 'all_2027_2046') & table1.age_group.isin(['all_ages', 'infant_0'])].to_csv(
        args.output_dir / 'table1_economics.csv', index=False)
    paths = [args.costs, SEVERITY, life_path, ENGLAND_RISK, ENGLAND_RISK.parent/'run_spec.json', Path(__file__),
             ROOT / 'src_python/pertussis/economics/unified_eval.py',
             ROOT / 'src_python/pertussis/economics/age_scoped_hospital_cost.py']
    if args.england_risk_period == '2016_2023':
        paths += [ROOT/REFERENCE_SOURCES['England']['risk_draws'], ROOT/REFERENCE_SOURCES['England']['run_spec']]
    if args.australia_risk:
        paths += [args.australia_risk.resolve(), (args.australia_risk.parent/'run_spec.json').resolve(),
                  (args.australia_risk.parent/'complete.json').resolve()]
    (args.output_dir / 'run_spec.json').write_text(json.dumps(dict(
        schema='unified-analysis-v2', n_draws=args.n_draws, seed=20261101, evaluations=metadata,
        transmission_uncertainty='fitted_point; local posterior evaluated separately',
        australian_risk_calibration=australian_risk_spec,
        england_risk_calibration_period=args.england_risk_period,
        source_hashes={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        price_burden_definition='vaccine prices scaled together; delivery/medical costs fixed; deaths/YLL scaled after projection',
        duration_definition='Tdap waning changed from 2027 for all Tdap protection; fitted history held fixed; no refit',
        full_incidence_definition='all typical and mild symptomatic onsets; weights 0.051 and 0.006; both 21 days',
        old_weight_sensitivity='0.133 is severe acute infectious disease, previously mislabeled as moderate',
        elapsed_seconds=time.monotonic()-begun), indent=2, default=str) + '\n')
    (args.output_dir / 'complete.json').write_text(json.dumps(dict(
        evaluations=len(metadata), table1_rows=len(table1), countries=args.countries, status='complete')) + '\n')


if __name__ == '__main__':
    main()
