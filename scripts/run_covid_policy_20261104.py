"""Evaluate COVID recovery fits and terminal ascertainment on the full coverage grid.

Uses current monthly clinical valuation, matched draws and unchanged programmes.
Historical scenarios are built by run_covid_refit_20261104; future observation
scenarios change reporting probabilities, never biological onset trajectories.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_current_preserving_strategies_20261103 import (
    BASE, ROOT, COST, SEV, evaluate_arrays,
)
from scripts.optimize_current_preserving_strategies_20261103 import (
    threshold_points, effect_for, price_frontier,
)
from scripts.build_eight_country_candidate_frontiers_20261103 import frontier
from scripts.summarize_eight_country_nhb_20261103 import summarize_nhb
from scripts.sensitivity_future_detection_20261102 import scaled_logits
from src_python.pertussis.economics.loader import load_costs, load_severity_priors
from src_python.pertussis.economics.evaluate import LOGIT_COLUMNS
from src_python.pertussis.economics.unified_eval import _reporting
from src_python.pertussis.inference.core2026 import _diagnostic_multiplier_frame, _dated_multipliers
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.scenarios.current_preserving import coverage_grid
from src_python.pertussis.scenarios.fitted import load_fit, fit_parameters, history_to
from src_python.pertussis.states import TDAP_BOOSTER

MAIN = BASE/'current_preserving_strategies_20261103'
OUT = BASE/'covid_policy_20261104'
COUNTRIES = ('Australia','England','Sweden','Japan','New_Zealand','China','Brazil','Thailand')


def observation_info(info, multiplier):
    logits, ratio = _reporting(info, ROOT)
    adjusted = scaled_logits(logits, multiplier)
    return {**info, 'reporting': {**dict(zip(LOGIT_COLUMNS, adjusted)),
                                'breakthrough_reporting_ratio': ratio}}


def terminal_multiplier(country, spec=None):
    # Match the fit loader's exact country key; missing national rows retain 1.
    timeline = _diagnostic_multiplier_frame(country, geographic_policy='national_only')
    if spec is not None and spec.get('covid_scenario', '').startswith('detection_'):
        raise ValueError('Historical detection variants require their own terminal timeline')
    return float(_dated_multipliers(['2025-12-31'], ['2026-01-01'], timeline)[0])


def project_refit(country, fit_dir, scenario_id, dest):
    from scripts.run_covid_refit_20261104 import build_scenario_inputs
    spec, row = load_fit(fit_dir)
    inputs = build_scenario_inputs(spec, 2057, scenario_id)
    params = fit_parameters(spec, row.to_dict())
    original = json.loads((MAIN/country/'complete.json').read_text())
    age, effective = original['adolescent_age'], original['maternal_effective_share']
    scenarios, catalog, _ = coverage_grid(inputs, start_year=2027, end_year=2057,
        adolescent_age=age, maternal_effective_share=effective)
    reference = pd.read_csv(MAIN/country/'strategy_catalog.csv').drop(columns='country')
    pd.testing.assert_frame_equal(catalog.replace({None: np.nan}), reference, check_dtype=False)
    dates = pd.DatetimeIndex(inputs.calendar.dates)
    rec = np.flatnonzero((dates >= '2027-01-01') & (dates <= '2057-01-01') & (dates.day == 1))
    if len(rec) != 361:
        raise ValueError('Expected complete monthly projection')
    history = history_to(inputs, params, 2027)
    arrays, doses = [], []
    for sid, inp in scenarios.items():
        result = Simulator(inp).run(params, history.y, history.acc, int(rec[0]), int(rec[-1]), record_days=rec)
        raw = np.diff(result.acc_records, axis=0)
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError('Invalid monthly projection')
        receipt = np.zeros((2,30))
        for j, year in enumerate(range(2027,2057)):
            a,b = inputs.calendar.year_start_day(year),inputs.calendar.year_start_day(year+1)
            receipt[0,j] = np.sum(inp.maternal_effective[a:b]/effective*inputs.births_per_day[a:b])
            receipt[1,j] = result.receipt_ledger[year-inputs.calendar.first_year,TDAP_BOOSTER,0]
        arrays.append(raw); doses.append(receipt)
    info = dict(country=country, fit_directory=str(Path(fit_dir).resolve().relative_to(ROOT)),
                draw_id=int(row.start_index), scenario_id=scenario_id,
                reporting={c: float(row[c]) for c in LOGIT_COLUMNS+['breakthrough_reporting_ratio']})
    dest.mkdir(parents=True, exist_ok=False)
    catalog.to_csv(dest/'strategy_catalog.csv',index=False)
    np.savez_compressed(dest/'projection_arrays.npz',strategy_ids=np.array(list(scenarios)),
        month_starts=dates[rec[:-1]].strftime('%Y-%m-%d').to_numpy(dtype=str),
        month_ends=dates[rec[1:]].strftime('%Y-%m-%d').to_numpy(dtype=str),
        onsets=np.stack(arrays), annual_doses=np.stack(doses))
    (dest/'complete.json').write_text(json.dumps(info,indent=2)+'\n')
    return dest


def decisions(cache, arms, country, refs):
    choices, prices, bounds = [], [], []
    for horizon in (20,30):
        costs = cache[f'cost_{horizon}']
        vaccine = cache[f'vaccine_{horizon}']
        for burden in ('reported_uniform','symptomatic_differentiated'):
            effects = effect_for(cache,horizon,burden,'included_base')
            tags = dict(country=country,horizon=horizon,burden=burden,mortality='included_base')
            means = pd.DataFrame(dict(strategy_id=arms,dalys_averted=effects.mean(0),incremental_cost=costs.mean(0)))
            _, intervals = frontier(means)
            bounds.append(intervals.assign(**tags))
            for ref in refs[refs.country.eq(country)].to_dict('records'):
                lam = ref['threshold_intl2024']
                _, choice = summarize_nhb(effects,costs,lam,arms=arms)
                choices.append({**tags,**ref,**choice})
            max_ref = refs[refs.country.eq(country)].threshold_intl2024.max()
            prices.append(price_frontier(arms,effects.mean(0),costs.mean(0),
                vaccine[...,0].mean(0),vaccine[...,1].mean(0),max_ref).assign(**tags,threshold_intl2024=max_ref))
    return pd.DataFrame(choices),pd.concat(prices,ignore_index=True),pd.concat(bounds,ignore_index=True)


def run(country, scenario, folder, output, n_draws, observation_modes):
    dest=output/scenario/country
    if (dest/'complete.json').exists():
        completed=json.loads((dest/'complete.json').read_text())
        if completed['n_draws'] != n_draws or set(completed['multipliers']) != set(observation_modes):
            raise ValueError('Existing policy result has a different requested design')
        return completed
    dest.mkdir(parents=True,exist_ok=True)
    info=json.loads((folder/'complete.json').read_text())
    z=np.load(folder/'projection_arrays.npz',allow_pickle=False)
    arms=z['strategy_ids'].astype(str).tolist()
    costs,severity=load_costs(COST),load_severity_priors(SEV)
    life_path=ROOT/costs['global']['life_table']['csv']; life=pd.read_csv(life_path)
    refs,ref_paths=threshold_points();refs=refs[refs.threshold_family.eq('sourced_DALY_estimate')]
    records={k:[] for k in ('summary','age_effects','annual_cases','decisions','price_intervals','threshold_intervals')}
    multipliers={}; baseline_annual=None
    for mode in observation_modes:
        multiplier=1. if mode=='reference_1' else terminal_multiplier(country)
        multipliers[mode]=multiplier
        adjusted=observation_info(info,multiplier)
        summary,ages,annual,draws,cache,meta=evaluate_arrays(country,arms,z['onsets'],z['annual_doses'],
            z['month_starts'],z['month_ends'],adjusted,severity,costs,life,n=n_draws)
        if mode=='reference_1':
            baseline_annual=annual
            if scenario=='saved_current' and n_draws==1000:
                saved=pd.read_csv(BASE/'current_preserving_economics_20261103/summary.csv')
                keys=['country','horizon','strategy_id','burden','mortality']
                a=summary.set_index(keys).sort_index()
                b=saved[saved.country.eq(country)].set_index(keys).sort_index()
                np.testing.assert_allclose(a[['dalys_averted','incremental_cost']],
                    b[['dalys_averted','incremental_cost']],rtol=1e-9,atol=1e-6)
        elif baseline_annual is not None:
            np.testing.assert_allclose(annual.symptomatic_cases,baseline_annual.symptomatic_cases,rtol=1e-12)
            np.testing.assert_allclose(annual.reported_cases,baseline_annual.reported_cases*multiplier,rtol=1e-10,atol=1e-7)
        dc,pc,tc=decisions(cache,arms,country,refs)
        for key,table in zip(records,(summary,ages,annual,dc,pc,tc)):
            records[key].append(table.assign(historical_scenario=scenario,observation_scenario=mode,
                                             future_detection_multiplier=multiplier))
        np.savez_compressed(dest/f'{mode}_components.npz',strategy_ids=np.array(arms),**cache)
    for key,frames in records.items():
        pd.concat(frames,ignore_index=True).to_csv(dest/f'{key}.csv',index=False)
    sources=[Path(__file__).resolve(),folder/'complete.json',folder/'projection_arrays.npz',
        COST,SEV,life_path,*ref_paths,ROOT/'data/raw/pertussis_diagnostic_standards_timeline.csv',
        ROOT/'scripts/evaluate_current_preserving_strategies_20261103.py']
    result=dict(country=country,historical_scenario=scenario,n_draws=n_draws,strategies=len(arms),
        multipliers=multipliers,status='completed_conditional_policy_sensitivity',
        observation_definition='Hold national diagnostic multiplier on 2025-12-31 throughout 2027-2056; compare reference 1.',
        clinical_definition='Per-reported-case severity and costs held fixed; infant per-onset mortality unchanged in six settings. Observation sensitivity is not an estimate of biological disease change.',
        projection_source=str(folder.relative_to(ROOT)),
        source_files=[str(p.relative_to(ROOT)) for p in sources if p.is_file()])
    (dest/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(country,scenario,multipliers,'valued',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--countries',nargs='+',default=list(COUNTRIES),choices=COUNTRIES)
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--n-draws',type=int,default=1000)
    p.add_argument('--fit-map',type=Path,help='List of country, scenario_id, fit_directory records')
    p.add_argument('--observation-modes',nargs='+',default=['reference_1','terminal_2025'],choices=['reference_1','terminal_2025'])
    args=p.parse_args(); args.output_dir=args.output_dir.resolve()
    jobs=([dict(country=c,scenario_id='saved_current') for c in args.countries] if args.fit_map is None
          else json.loads(args.fit_map.read_text()))
    results=[]
    for job in jobs:
        country,scenario=job['country'],job['scenario_id']
        if country not in args.countries: continue
        folder=MAIN/country
        if scenario!='saved_current':
            folder=args.output_dir/'projections'/scenario/country
            if not (folder/'complete.json').exists():
                folder=project_refit(country,ROOT/job['fit_directory'],scenario,folder)
        results.append(run(country,scenario,folder,args.output_dir,args.n_draws,args.observation_modes))
    for name in ('summary','decisions','price_intervals','threshold_intervals','age_effects'):
        paths=[args.output_dir/r['historical_scenario']/r['country']/f'{name}.csv' for r in results]
        pd.concat([pd.read_csv(x) for x in paths],ignore_index=True).to_csv(args.output_dir/f'{name}.csv',index=False)
    (args.output_dir/'metadata.json').write_text(json.dumps(dict(status='completed_requested_jobs',jobs=results),indent=2)+'\n')


if __name__=='__main__':
    main()
