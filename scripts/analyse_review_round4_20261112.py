"""Independent mucosal strengths, full-refit propagation and Brazil horizons.

All comparisons are unweighted structural diagnostics. Existing scientific
producers and parameter fits are read, never refreshed or replaced.
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
from scripts import build_bpze1_policy_design_20261107 as design

ROOT, BASE = rev.ROOT, rev.BASE
OUT = BASE / '02_sensitivity/review_round4'
STRENGTHS = (0., .25, .5, .75, 1.)


def run(ctx, acquisition=0., onward=1., duration=3., coverage=.9, dt=1.):
    info, inputs, params, hist, scenarios, catalog, dates, days = ctx
    events = np.array([(*rev._target_event(inputs, float(info['adolescent_age']), y), coverage)
                       for y in range(dates[0].year, dates[-1].year)])
    sim = rev.BPZE1Simulator(scenarios['current'])
    result = sim.run(params, hist.y, hist.acc, days[0], days[-1], record_days=days,
                    efficacy=acquisition, infectiousness_multiplier=onward,
                    mean_years=duration, start_day=days[0], nasal_events=events, dt=dt)
    raw = np.diff(result.acc_records, axis=0)
    assert np.isfinite(raw).all() and raw.min() >= 0
    return raw


def endpoints(data, j, horizons=(20, 30)):
    rows = []
    for horizon in horizons:
        health = data['yll'][:, j, :horizon, -1].sum(axis=1) + data['yld_reported_uniform'][j, :horizon, -1].sum()
        deaths = data['deaths'][:, j, :horizon, 0].sum(axis=1)
        rows.append(dict(horizon=horizon, **rev.describe(health, 'dalys'),
                         **rev.describe(deaths, 'infant_deaths'),
                         infant_cases=data['reported'][j, :horizon, 0].sum(),
                         adolescent_cases=data['reported'][j, :horizon, 2].sum(),
                         all_age_cases=data['reported'][j, :horizon, -1].sum()))
    return rows


def grid(country):
    ctx = rev.setup(country)
    info, inputs, params, hist, scenarios, catalog, dates, days = ctx
    current = catalog[catalog.strategy_id == 'current'].iloc[0]
    coverage = max(.7, float(current.adolescent_coverage))
    ratio = json.loads(rev.bp.CONFIG.read_text())['clinical_input']['point']
    residual = design.profile(ratio, .5, 1., 3.)['protected_state_residual']
    raw = []
    labels = []
    for a in STRENGTHS:
        for o in STRENGTHS:
            raw.append(run(ctx, 1 - residual**a, residual**o, coverage=coverage))
            labels.append(dict(acquisition_log_strength=a, onward_log_strength=o,
                               acquisition_reduction=1-residual**a,
                               onward_reduction=1-residual**o,
                               day90_residual=1-design.survival(90,3)+design.survival(90,3)*residual**(a+o)))
    raw = np.stack(raw)
    with np.load(rev.bp.adopted.OUT/country/'projection_arrays.npz') as z:
        ids = z['strategy_ids'].astype(str).tolist()
        np.testing.assert_allclose(raw[0], z['onsets'][ids.index('current')], rtol=2e-8, atol=2e-4)
    # The three original points must reproduce the adopted trajectories.
    for a, o in ((0., 1.), (.5, .5), (1., 0.)):
        profile = design.profile(ratio, .5, a, 3.)
        folder = rev.bp.OUT/country/'profiles'/profile['candidate_id']
        cat = pd.read_csv(folder/'product_catalog.csv')
        selected = cat[(cat.route=='standalone') & np.isclose(cat.maternal_coverage,current.maternal_coverage)
                       & np.isclose(cat.nasal_coverage,coverage)]
        assert len(selected)==1
        with np.load(folder/selected.iloc[0].trajectory_path) as z:
            np.testing.assert_allclose(raw[STRENGTHS.index(a)*5+STRENGTHS.index(o)],z['onsets'],rtol=2e-8,atol=2e-4)
    data = rev.clinical(country, raw, dates[:-1], dates[1:], info)
    rows = [{**dict(country=country,nasal_coverage=coverage), **label, **row}
            for j,label in enumerate(labels) for row in endpoints(data,j)]
    pd.DataFrame(rows).to_csv(OUT/f'grid_{country}.csv',index=False)
    np.savez_compressed(OUT/f'grid_{country}.npz',onsets=raw,dates=dates.to_numpy())


def refits(country):
    ratio = json.loads(rev.bp.CONFIG.read_text())['clinical_input']['point']
    rows=[]
    for history in ('adopted','baseline','recovery_365'):
        ctx=rev.setup(country,history=history)
        info,inputs,params,hist,sc,catalog,dates,days=ctx
        coverage=max(.7,float(catalog[catalog.strategy_id=='current'].adolescent_coverage.iloc[0]))
        raw=[run(ctx,coverage=coverage)]
        for a in (0.,.5,1.):
            p=design.profile(ratio,.5,a,3.)
            raw.append(run(ctx,p['efficacy'],p['infectiousness_multiplier'],coverage=coverage))
        data=rev.clinical(country,np.stack(raw),dates[:-1],dates[1:],info)
        for j,a in enumerate((0.,.5,1.),1):
            rows += [dict(country=country,history=history,allocation=a,nasal_coverage=coverage,**r)
                     for r in endpoints(data,j)]
        np.savez_compressed(OUT/f'refits_{country}_{history}.npz',onsets=np.stack(raw),dates=dates.to_numpy())
    pd.DataFrame(rows).to_csv(OUT/f'refits_{country}.csv',index=False)


def brazil_context(start, multiplier):
    info=json.loads((rev.bp.adopted.OUT/'Brazil/complete.json').read_text())
    spec,fit=rev.load_fit(ROOT/info['fit_directory'])
    p=rev.fit_parameters(spec,fit.to_dict())
    inputs=rev.build_fitted_inputs(spec,start+50)
    hist=rev.history_to(inputs,p,start)
    sc,cat,_=rev.coverage_grid(inputs,start_year=start,end_year=start+50,
        adolescent_age=info['adolescent_age'],maternal_effective_share=info.get('maternal_effective_share',.9))
    dates=pd.date_range(f'{start}-01-01',f'{start+50}-01-01',freq='MS')
    days=np.array([inputs.calendar.day(d.to_datetime64()) for d in dates],dtype=np.int64)
    return info,inputs,replace(p,R0=p.R0*multiplier),hist,sc,cat,dates,days


def brazil(start, multiplier, dt=1.):
    start=int(start);multiplier=float(multiplier);dt=float(dt)
    ctx=brazil_context(start,multiplier)
    info,inputs,params,hist,sc,cat,dates,days=ctx
    ratio=json.loads(rev.bp.CONFIG.read_text())['clinical_input']['point']
    raw=[run(ctx,dt=dt)]
    durations=(1.,3.,6.,10.) if dt==1 else (3.,)
    for duration in durations:
        p=design.profile(ratio,.5,.5,duration)
        assert p['compatible']
        raw.append(run(ctx,p['efficacy'],p['infectiousness_multiplier'],duration=duration,dt=dt))
    raw=np.stack(raw)
    if start==2027 and multiplier==1 and dt==1:
        with np.load(rev.OUT/'long_raw_Brazil_adopted_unchanged.npz') as z:
            np.testing.assert_allclose(raw[0],z['current'],rtol=2e-8,atol=2e-4)
            np.testing.assert_allclose(raw[2],z['nasal'],rtol=2e-8,atol=2e-4)
    data=rev.clinical('Brazil',raw,dates[:-1],dates[1:],info)
    rows=[];annual=[]
    for j,duration in enumerate(durations,1):
        label=dict(country='Brazil',start_year=start,r0_multiplier=multiplier,duration=duration,dt=dt,nasal_coverage=.9)
        rows += [{**label,**r} for r in endpoints(data,j,(20,30,40,50))]
        for y in range(50):
            annual.append(dict(**label,year=start+y,
                infant_deaths=float(data['deaths'][:,j,y,0].mean()),
                dalys=float(data['yll'][:,j,y,-1].mean()+data['yld_reported_uniform'][j,y,-1])))
    key=f'brazil_{start}_{multiplier:g}_{dt:g}'
    pd.DataFrame(rows).to_csv(OUT/f'{key}.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/f'{key}_annual.csv',index=False)
    np.savez_compressed(OUT/f'{key}.npz',onsets=raw,dates=dates.to_numpy(),durations=durations)


def check_manifest(path):
    m=json.loads(path.read_text())
    for p in m.get('output_files',[]):
        if not (ROOT/p).is_file(): raise ValueError('Missing output: '+p)
    return m


def task(job):
    OUT.mkdir(parents=True,exist_ok=True)
    name='_'.join(map(str,job));meta=OUT/f'{name}_manifest.json'
    if meta.exists():check_manifest(meta);return name+' reused'
    opened=set();written=set()
    def capture(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            p=Path(args[0]).resolve()
            if p.is_relative_to(ROOT) and p.suffix in ('.csv','.gz','.json','.yaml','.npz'):
                mode=args[1]
                (written if isinstance(mode,str) and any(c in mode for c in ('w','a','+')) else opened).add(p)
    sys.addaudithook(capture);start=time.time()
    {'grid':grid,'refits':refits,'brazil':brazil}[job[0]](*job[1:])
    sources=(opened-written)|{Path(__file__).resolve(),Path(rev.__file__).resolve(),Path(design.__file__).resolve(),
        Path(rev.ev.__file__).resolve(),ROOT/'src_python/pertussis/extensions/bpze1.py',
        ROOT/'src_python/pertussis/model/kernel.py',ROOT/'src_python/pertussis/scenarios/fitted.py'}
    m=dict(job=job,seconds=time.time()-start,
        input_files=[str(p.relative_to(ROOT)) for p in sorted(sources) if p.is_file()],
        output_files=[str(p.relative_to(ROOT)) for p in sorted(written) if p.is_relative_to(OUT) and p.is_file()])
    meta.write_text(json.dumps(m,indent=2)+'\n')
    return name+' completed'


def set_policy_dir(path):
    rev.bp.OUT = Path(path).resolve()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','grid','refits','brazil','check'),default='all')
    p.add_argument('--countries',nargs='+',default=list(rev.COUNTRIES));p.add_argument('--workers',type=int,default=2)
    p.add_argument('--policy-dir', type=Path, default=rev.bp.OUT)
    args=p.parse_args()
    set_policy_dir(args.policy_dir)
    if args.stage=='check':
        files=list(OUT.glob('*_manifest.json'));assert len(files)==26,len(files)
        for f in files:check_manifest(f)
        print('Verified 26 fourth-review numerical jobs');return
    jobs=[(s,c) for s in ('grid','refits') if args.stage in ('all',s) for c in args.countries]
    if args.stage in ('all','brazil'):
        jobs += [('brazil',str(y),str(m)) for y in (2027,2029,2032) for m in (.8,1.,1.2)]
        jobs += [('brazil','2027','1.0','0.5')]
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1,
                             initializer=set_policy_dir, initargs=(rev.bp.OUT,)) as pool:
        for f in as_completed([pool.submit(task,j) for j in jobs]):print(f.result(),flush=True)


if __name__=='__main__':main()
