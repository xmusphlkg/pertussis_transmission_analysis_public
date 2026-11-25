"""Project eight current-preserving, independently varied coverage grids.

python -m scripts.run_current_preserving_strategies_20261103 --workers 3
Each country is checkpointed with complete.json, written only after validation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src_python.pertussis.config import REPO_ROOT as ROOT
from src_python.pertussis.economics.unified_eval import _reporting
from src_python.pertussis.economics.evaluate import LOGIT_COLUMNS
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.scenarios.aggregation import AGE_GROUPS
from src_python.pertussis.scenarios.current_preserving import coverage_grid
from src_python.pertussis.scenarios.fitted import build_fitted_inputs, fit_parameters, history_to, load_fit
from src_python.pertussis.scenarios.health_outcomes import annual_age_onsets
from src_python.pertussis.scenarios.monthly_reporting import monthly_reported_cases, monthly_reporting_weights
from src_python.pertussis.scenarios.programme_scenarios import _target_event
from src_python.pertussis.states import TDAP_BOOSTER, MODEL_VERSION

COUNTRIES = ('Australia','England','Sweden','Japan','New_Zealand','China','Brazil','Thailand')
START, END = 2027, 2057
BASE = ROOT/'outputs/core2026'
OUT = BASE/'current_preserving_strategies_20261103'
def source_for(country):
    """Locate the historical current-programme comparator for the adopted fit."""
    reference = json.loads((OUT / country / 'complete.json').read_text())
    folder = ROOT / reference['original_projection_directory']
    info = json.loads((folder/'complete.json').read_text())
    if country == 'Japan' and 'official_age' not in info['fit_directory']:
        raise ValueError('Japan requires official-age corrected fit')
    return folder, info, {}


def run_country(country, output_dir):
    started=time.perf_counter();dest=Path(output_dir)/country
    if dest.exists():
        raise FileExistsError(dest)
    old_folder, old_info, source_revisions=source_for(country)
    fit_dir=ROOT/old_info['fit_directory'];spec,fit=load_fit(fit_dir)
    inputs=build_fitted_inputs(spec,END);params=fit_parameters(spec,fit.to_dict())
    # Japan's corrected projection manifest uses country rather than profile and
    # its early producer omitted these two optional fields.
    age=float(old_info.get('adolescent_age', 12.));effective=float(old_info.get('maternal_effective_share', .9))
    scenarios,catalog,aliases=coverage_grid(inputs,start_year=START,end_year=END,
        adolescent_age=age,maternal_effective_share=effective)
    first,last=inputs.calendar.year_start_day(START),inputs.calendar.year_start_day(END)
    expected_targets={_target_event(inputs,age,y) for y in range(START,END)}
    # This project's saved eight schedules have no additional Tdap event. Thus
    # the original kernel's Tdap receipts exactly count the adolescent programme.
    # Do not silently include a different-age programme if input schedules change.
    ev=inputs.events;selected=(ev.kind==TDAP_BOOSTER)&(ev.day>=first)&(ev.day<last)
    if not set(zip(ev.day[selected],ev.bin[selected])).issubset(expected_targets):
        raise ValueError('Additional existing Tdap events require separate receipt attribution')
    logits,ratio=_reporting(old_info,ROOT)
    dates=pd.DatetimeIndex(inputs.calendar.dates)
    rec=np.flatnonzero((dates>=f'{START}-01-01')&(dates<=f'{END}-01-01')&(dates.day==1))
    if len(rec)!=361 or rec[0]!=first or rec[-1]!=last:
        raise ValueError('Incomplete 30-year monthly calendar')
    starts,ends=dates[rec[:-1]],dates[rec[1:]]
    annual_idx=np.arange(0,len(rec),12)
    weights=monthly_reporting_weights(inputs.grid,logits,starts,ends)
    history=history_to(inputs,params,START)
    history_elapsed=time.perf_counter()-started
    old_onsets=pd.read_csv(old_folder/'annual_age_onsets.csv.gz',dtype={'strategy_id':str})
    old_current=old_onsets[(old_onsets.strategy_id=='current')&old_onsets.year.between(START,END-1)].sort_values(['year','bin_index'])
    old_receipts=pd.read_csv(old_folder/'receipts.csv.gz',dtype={'strategy_id':str})
    dest.mkdir(parents=True)
    monthly_rows=[];annual_rows=[];receipt_rows=[];case_rows=[];dose_arrays=[];onset_arrays=[];pop_rows=[]
    baseline_raw=None;baseline_cases=None;raw_ledger=[]
    for sid,scenario in scenarios.items():
        tic=time.perf_counter()
        result=Simulator(scenario).run(params,history.y,history.acc,first,last,record_days=rec,snap_days=rec[annual_idx])
        raw=np.diff(result.acc_records,axis=0)
        if not np.isfinite(raw).all() or (raw<0).any():
            raise ValueError('Invalid onset projection: '+sid)
        annual_result=replace(result,record_days=rec[annual_idx],acc_records=result.acc_records[annual_idx])
        annual=annual_age_onsets(annual_result,inputs.grid,inputs.calendar).assign(country=country,profile=country,strategy_id=sid)
        reports=monthly_reported_cases(raw[:,:,0],raw[:,:,1],ratio,weights)
        doses=np.empty((2,END-START));coverage=catalog.set_index('strategy_id').loc[sid]
        for yi,year in enumerate(range(START,END)):
            a,b=inputs.calendar.year_start_day(year),inputs.calendar.year_start_day(year+1)
            doses[0,yi]=np.sum(scenario.maternal_effective[a:b]/effective*inputs.births_per_day[a:b])
            doses[1,yi]=result.receipt_ledger[year-inputs.calendar.first_year,TDAP_BOOSTER,0]
            if not np.isfinite(doses[:,yi]).all() or (doses[:,yi]<0).any():
                raise ValueError('Invalid receipts')
            for j,programme in enumerate(('maternal','adolescent')):
                receipt_rows.append(dict(country=country,profile=country,strategy_id=sid,year=year,
                    programme=programme,doses=doses[j,yi],coverage=coverage[programme+'_coverage']))
            pop_rows.append(dict(country=country,strategy_id=sid,year=year,
                births=float(inputs.births_per_day[a:b].sum()),
                population_start_of_year_before_events=float(result.snapshots[yi].sum())))
        if sid=='current':
            if len(old_current)!=len(annual):
                raise ValueError('Saved current onset grid incomplete')
            np.testing.assert_allclose(annual[['typical_onsets','mild_onsets','infections']],
                old_current[['typical_onsets','mild_onsets','infections']],rtol=1e-9,atol=1e-6)
            ref=old_receipts[(old_receipts.strategy_id=='current')&old_receipts.year.between(START,END-1)]
            for j,programme in enumerate(('maternal','adolescent')):
                np.testing.assert_allclose(doses[j],ref[ref.programme==programme].sort_values('year').doses,rtol=1e-9,atol=1e-6)
            baseline_raw,baseline_cases=raw.copy(),reports.copy()
            # Independently reconstruct the no-expansion scenario, then replay
            # its complete trajectory; equality is stronger than a renamed arm.
            from src_python.pertussis.scenarios.current_preserving import expand_programmes
            unchanged=expand_programmes(inputs,start_year=START,end_year=END,adolescent_age=age,
                maternal_effective_share=effective,maternal_target=0.,adolescent_target=0.)
            replay=Simulator(unchanged).run(params,history.y,history.acc,first,last,record_days=rec)
            np.testing.assert_array_equal(replay.acc_records,result.acc_records)
            np.testing.assert_array_equal(replay.receipt_ledger,result.receipt_ledger)
        elif baseline_raw is None:
            raise AssertionError('Current must be first')
        nb=inputs.grid.n_bins
        month=pd.DataFrame(dict(country=country,strategy_id=sid,year=np.repeat(starts.year,nb),
            month=np.repeat(starts.month,nb),month_start=np.repeat(starts.strftime('%Y-%m-%d'),nb),
            bin_index=np.tile(np.arange(nb),360),typical_onsets=raw[:,:,0].ravel(),
            mild_onsets=raw[:,:,1].ravel(),infections=raw[:,:,2].ravel()))
        monthly_rows.append(month);annual_rows.append(annual);dose_arrays.append(doses);onset_arrays.append(raw)
        for mi,date in enumerate(starts):
            for gi,group in enumerate(AGE_GROUPS):
                case_rows.append(dict(country=country,strategy_id=sid,year=date.year,month=date.month,
                    age_group=group,reported_cases=reports[mi,gi],reported_cases_averted_vs_current=baseline_cases[mi,gi]-reports[mi,gi]))
        raw_ledger.append(result.receipt_ledger[START-inputs.calendar.first_year:END-inputs.calendar.first_year])
        print(f'{country} {sid}: 30-year projection {time.perf_counter()-tic:.2f}s',flush=True)
    catalog.insert(0,'country',country);aliases.insert(0,'country',country)
    catalog.to_csv(dest/'strategy_catalog.csv',index=False);aliases.to_csv(dest/'aliases.csv',index=False)
    pd.concat(monthly_rows,ignore_index=True).to_csv(dest/'monthly_age_onsets.csv.gz',index=False)
    pd.concat(annual_rows,ignore_index=True).to_csv(dest/'annual_age_onsets.csv.gz',index=False)
    receipts=pd.DataFrame(receipt_rows);receipts.to_csv(dest/'receipts.csv.gz',index=False)
    cases=pd.DataFrame(case_rows);cases.to_csv(dest/'monthly_cases.csv.gz',index=False)
    cases.groupby(['country','strategy_id','year','age_group'],as_index=False)[['reported_cases','reported_cases_averted_vs_current']].sum().to_csv(dest/'annual_cases.csv',index=False)
    pd.DataFrame(pop_rows).to_csv(dest/'population_births.csv',index=False)
    np.savez_compressed(dest/'projection_arrays.npz',strategy_ids=np.array(list(scenarios)),
        month_starts=starts.strftime('%Y-%m-%d').to_numpy(dtype=str),month_ends=ends.strftime('%Y-%m-%d').to_numpy(dtype=str),
        onsets=np.stack(onset_arrays),annual_doses=np.stack(dose_arrays),raw_receipt_ledger=np.stack(raw_ledger),
        bin_labels=np.array(inputs.grid.labels()),bin_kind=inputs.grid.kind,
        reporting_logits=logits,mild_reporting_ratio=ratio)
    metadata=dict(status='completed_current_preserving_fixed_candidate_projection',country=country,profile=country,
        design='current_preserving_independent_coverage_grid',model_version=MODEL_VERSION,
        start_year=START,end_year_exclusive=END,horizon=30,main_horizon=20,strategies=list(scenarios),
        coverage_grid=[.5,.7,.9],alias_target=.7,adolescent_age=age,maternal_effective_share=effective,
        fit_directory=str(fit_dir.relative_to(ROOT)),draw_id=int(fit.start_index),parameters=params.as_dict(),
        reporting={**dict(zip(LOGIT_COLUMNS,logits.tolist())),'breakthrough_reporting_ratio':ratio},
        original_projection_directory=str(old_folder.relative_to(ROOT)),
        recorded_source_revisions_requiring_current_replay=source_revisions,
        history_and_input_seconds=history_elapsed,total_seconds=time.perf_counter()-started,
        current_replay='30 yearly onset grids and maternal/adolescent receipts match saved compatible current; zero-target replay is bitwise identical',
        receipt_definition='Total vaccinated pregnancies and kernel scheduled target-age Tdap receipts for each scenario. Incremental doses require subtraction of current. Existing infant/child events preserved; raw receipt ledger retained for all dose kinds.',
        projection_definition='Joint nonlinear transmission simulation for every unique grid point, shared pre-2027 history; no event deletion and no existing coverage reduction.',
        npz_axes={'onsets':['strategy','month','model_bin','typical_mild_infections'],
                  'annual_doses':['strategy','maternal_adolescent','year_2027_2056']},
        scope='Encoded schedules retained, not independently verified contemporary national policy. Fixed candidate fits, Japan official-age corrected; six-country noninfant mortality gap unchanged. No health/economic valuation performed here.',
        output_files=[p.name for p in dest.iterdir() if p.is_file()])
    (dest/'complete.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return dict(country=country,strategies=len(scenarios),seconds=metadata['total_seconds'],fit_directory=metadata['fit_directory'],status=metadata['status'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--countries',nargs='+',default=list(COUNTRIES),choices=COUNTRIES)
    args=parser.parse_args()
    if not 1<=args.workers<=4:
        raise ValueError('Use 1–4 workers')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if any((args.output_dir/c).exists() for c in args.countries):
        raise FileExistsError('Country output exists; use a new output directory')
    done=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_country,c,str(args.output_dir)):c for c in args.countries}
        for future in as_completed(futures):
            done.append(future.result())
            (args.output_dir/'progress.json').write_text(json.dumps(done,indent=2)+'\n')
    catalogs=pd.concat([pd.read_csv(args.output_dir/c/'strategy_catalog.csv') for c in args.countries],ignore_index=True)
    catalogs.to_csv(args.output_dir/'strategy_catalog.csv',index=False)
    pd.concat([pd.read_csv(args.output_dir/c/'aliases.csv') for c in args.countries],ignore_index=True).to_csv(args.output_dir/'aliases.csv',index=False)
    (args.output_dir/'metadata.json').write_text(json.dumps(dict(status='completed_current_preserving_projections',
        countries=args.countries,strategies=len(catalogs),country_runs=done,main_horizon=20,sensitivity_horizon=30,
        output_files=['strategy_catalog.csv', 'aliases.csv', 'progress.json']),indent=2)+'\n')
    print(json.dumps(done,indent=2),flush=True)


if __name__=='__main__':
    main()
