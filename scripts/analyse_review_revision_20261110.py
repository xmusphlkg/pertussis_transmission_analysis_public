"""Review-directed diagnostics; adopted scientific producers remain unchanged.

Stages are independently resumable. No new probability distribution is assigned
to structural or clinical-bridge scenarios.
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
from scipy.optimize import brentq
from scripts import run_bpze1_policy_extension_20261107 as bp
from scripts import evaluate_current_preserving_strategies_20261103 as ev
from scripts.analyse_bpze1_transmission_followup_20261107 import REFIT
from scripts.run_covid_refit_20261104 import build_scenario_inputs
from scripts.optimize_current_preserving_strategies_20261103 import effect_for, threshold_points
from src_python.pertussis.extensions.bpze1 import BPZE1Simulator
from src_python.pertussis.extensions.mucosal_economics import nasal_comparator_ceilings, nasal_price_frontier
from src_python.pertussis.grid import AgeGrid
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.scenarios.fitted import load_fit, fit_parameters, build_fitted_inputs, history_to
from src_python.pertussis.scenarios.current_preserving import coverage_grid
from src_python.pertussis.scenarios.programme_scenarios import _target_event
from src_python.pertussis.states import R, W
ROOT, BASE = bp.ROOT, bp.BASE
OUT = BASE/'02_sensitivity/review_revision'
COUNTRIES = tuple(json.loads(bp.CONFIG.read_text())['countries'])
PID = 'b050_a050_d03_point'

def describe(a, name):
    return {name: float(np.mean(a)), name+'_q025': float(np.quantile(a,.025)), name+'_q975': float(np.quantile(a,.975))}

def clinical(country, raw, starts, ends, info):
    """Annual reported/symptomatic cases and paired health, arbitrary full years.

    DALYs discount both future life-years and event years exactly as adopted.
    Annual deaths are undiscounted. Returns differences against first arm.
    """
    grid = AgeGrid()
    nyr = len(starts)//12
    assert len(starts)==nyr*12 and raw.shape[1:]==(nyr*12,grid.n_bins,3)
    assert np.isfinite(raw).all() and raw.min()>=0
    logits, ratio = ev._reporting(info, ROOT)
    fractions, groups = ev.month_geometry(starts, ends, grid)
    reporting = ev.monthly_reporting_weights(grid, logits, starts, ends)
    t,m = raw[...,0],raw[...,1]
    rt = np.einsum('smb,mbg->smg',t,reporting)
    rm = np.einsum('smb,mbg->smg',m*ratio,reporting)
    tc = np.einsum('smb,mbg->smg',t,groups)
    mc = np.einsum('smb,mbg->smg',m,groups)
    costs,severity = ev.load_costs(ev.COST),ev.load_severity_priors(ev.SEV)
    life = pd.read_csv(ROOT/costs['global']['life_table']['csv'])
    k = ev.sample_kernel(country,severity,costs,life,grid,logits,n=1000)
    def unit(name):
        d=costs['global']['disability_weights'][name]
        return float(d['value'])*-np.expm1(-.03*float(d['duration_days'])/365.25)/.03
    ylds=ev.morbidity_factorial(rt,rm,tc,mc,unit('pertussis_acute_moderate'),unit('pertussis_acute_mild'))
    data={'reported':(rt+rm).reshape(len(raw),nyr,12,5).sum(axis=2),
          'symptomatic':(tc+mc).reshape(len(raw),nyr,12,5).sum(axis=2)}
    yd,dd=[],[]
    for y in range(nyr):
        sl=slice(y*12,(y+1)*12)
        if k['infant_risks'] is None:
            b=(t+ratio*m)[:,sl]
            deaths=ev.integrate_endpoints(b,fractions[sl],np.ones(12),k['death'])
            yll=ev.integrate_endpoints(b,fractions[sl],np.ones(12),k['yll'])/1.03**y
        else:
            pt,pm=k['infant_risks']
            tb,mb=t[:,sl].sum(axis=1),m[:,sl].sum(axis=1)
            d=np.einsum('sb,nb->ns',tb,pt)+np.einsum('sb,nb->ns',mb,pm)
            v=np.einsum('sb,nb,b->ns',tb,pt,k['yll_bin'])+np.einsum('sb,nb,b->ns',mb,pm,k['yll_bin'])
            deaths=np.zeros((1000,len(raw),5)); yll=np.zeros_like(deaths)
            deaths[...,0]=deaths[...,-1]=d
            yll[...,0]=yll[...,-1]=v/1.03**y
        dd.append(deaths[:,[0]]-deaths);yd.append(yll[:,[0]]-yll)
    data['deaths']=np.stack(dd,axis=2);data['yll']=np.stack(yd,axis=2)
    for b,v in ylds.items():
        v=v.reshape(len(raw),nyr,12,5).sum(axis=2)/1.03**np.arange(nyr)[None,:,None]
        data['yld_'+b]=v[[0]]-v
    data['reported']=data['reported'][[0]]-data['reported']
    data['symptomatic']=data['symptomatic'][[0]]-data['symptomatic']
    return data

def setup(country,end=2057,history='adopted'):
    folder = bp.adopted.OUT/country if history=='adopted' else REFIT/'policy/projections'/history/country
    info=json.loads((folder/'complete.json').read_text())
    if 'adolescent_age' not in info:
        adopted_info=json.loads((bp.adopted.OUT/country/'complete.json').read_text())
        info['adolescent_age']=adopted_info['adolescent_age']
        info['maternal_effective_share']=adopted_info.get('maternal_effective_share',.9)
    spec,fit=load_fit(ROOT/info['fit_directory'])
    params=fit_parameters(spec,fit.to_dict())
    inputs=build_fitted_inputs(spec,end) if history=='adopted' else build_scenario_inputs(spec,end,history)
    prior=history_to(inputs,params,2027)
    scenarios,cat,_=coverage_grid(inputs,start_year=2027,end_year=end,
        adolescent_age=info['adolescent_age'],maternal_effective_share=info.get('maternal_effective_share',.9))
    dates=pd.date_range('2027-01-01',f'{end}-01-01',freq='MS')
    days=np.array([inputs.calendar.day(d.to_datetime64()) for d in dates],dtype=np.int64)
    return info,inputs,params,prior,scenarios,cat,dates,days

def simulate(ctx,efficacy=0.,onward=1.,coverage=.9,scenario='current',initial=None,params=None):
    info,inputs,p,h,sc,cat,dates,days=ctx
    events=np.array([(*_target_event(inputs,float(info['adolescent_age']),y),coverage) for y in range(2027,dates[-1].year)])
    sim=BPZE1Simulator(sc[scenario])
    res=sim.run(p if params is None else params,h.y if initial is None else initial,h.acc,days[0],days[-1],
        record_days=days,efficacy=efficacy,infectiousness_multiplier=onward,mean_years=3.,
        start_day=days[0],nasal_events=events)
    raw=np.diff(res.acc_records,axis=0)
    assert np.isfinite(raw).all() and raw.min()>=0
    return raw,res

def dose_stage():
    src=BASE/'02_sensitivity/cross_setting/cross_setting_drivers_20261107/strategy_efficiency_and_accounting.csv'
    table=pd.read_csv(src);rows=[];rank=[]
    for country in COUNTRIES:
        with np.load(BASE/'current_preserving_economics_20261103'/f'{country}_components.npz') as z:
            arms=z['strategy_ids'].astype(str).tolist()
            for h in (20,30):
                subset=table[(table.country==country)&(table.horizon==h)&table.programme.isin(['M','A'])]
                idx=[arms.index(a) for a in subset.strategy_id]
                dose=subset.discounted_added_doses.to_numpy()
                for burden in ev.BURDENS:
                    for scale in (0.,.1,.25,.5,1.,2.):
                        health=scale*z[f'yll_{h}'][:,idx]+z[f'yld_{h}_{burden}'][idx][None]
                        values=health/dose[None]*1000
                        for j,r in enumerate(subset.itertuples()):
                            programme=float(r.vaccine_cost+r.delivery_cost)
                            rows.append(dict(country=country,horizon=h,strategy_id=r.strategy_id,programme=r.programme,
                                burden=burden,yll_scale=scale,added_receipts=r.added_doses,discounted_added_receipts=dose[j],
                                **describe(values[:,j],'dalys_per_1000_discounted_receipts'),
                                dalys_per_1000_undiscounted_receipts=health[:,j].mean()/r.added_doses*1000,
                                dalys_per_million_programme_expenditure=health[:,j].mean()/programme*1e6,
                                dalys_per_million_net_expenditure=health[:,j].mean()/r.net_cost*1e6 if r.net_cost>0 else np.nan))
                        mi=np.where(subset.programme.to_numpy()=='M')[0];ai=np.where(subset.programme.to_numpy()=='A')[0]
                        if len(mi) and len(ai):
                            difference=values[:,mi].min(axis=1)-values[:,ai].max(axis=1)
                            rank.append(dict(country=country,horizon=h,burden=burden,yll_scale=scale,
                                min_maternal_mean=values[:,mi].mean(axis=0).min(),max_adolescent_mean=values[:,ai].mean(axis=0).max(),
                                all_maternal_means_greater=bool(values[:,mi].mean(axis=0).min()>values[:,ai].mean(axis=0).max()),
                                conditional_frequency_all_maternal_greater=float((difference>0).mean()),
                                **describe(difference,'paired_minimum_margin')))
    pd.DataFrame(rows).to_csv(OUT/'dose_productivity.csv',index=False)
    pd.DataFrame(rank).to_csv(OUT/'dose_rankings.csv',index=False)

def hospital_stage(country):
    """Reference product; all standard alternatives and both nasal routes reselected."""
    folder=bp.OUT/country/'profiles'/PID
    source=bp.adopted.OUT/country
    info=json.loads((source/'complete.json').read_text())
    products=pd.read_csv(folder/'product_catalog.csv')
    with np.load(source/'projection_arrays.npz') as z:
        raw=list(z['onsets']);starts=z['month_starts'];ends=z['month_ends']
        standard=z['strategy_ids'].astype(str).tolist()
    for r in products.itertuples():
        with np.load(folder/r.trajectory_path) as z:raw.append(z['onsets'])
    raw=np.stack(raw)
    with np.load(folder/'raw/paired_components.npz') as z:cache={k:z[k] for k in z.files}
    arms=cache['strategy_ids'].astype(str).tolist();ci=arms.index('current')
    assert arms==standard+products.strategy_id.tolist()
    receipts=pd.read_csv(folder/'annual_receipts_and_infections.csv')
    grid=AgeGrid();logits,ratio=ev._reporting(info,ROOT)
    fractions,_=ev.month_geometry(starts,ends,grid)
    reporting=ev.monthly_reporting_weights(grid,logits,starts,ends)
    base=raw[...,0]+ratio*raw[...,1];reported=np.einsum('smb,mbg->smg',base,reporting)
    costs,severity=ev.load_costs(ev.COST),ev.load_severity_priors(ev.SEV)
    k=ev.sample_kernel(country,severity,costs,pd.read_csv(ROOT/costs['global']['life_table']['csv']),grid,logits)
    refs,_=threshold_points();refs=refs[(refs.country==country)&(refs.threshold_family=='sourced_DALY_estimate')]
    matching=dict(zip(products.strategy_id,products.matched_standard_strategy));routes=dict(zip(products.strategy_id,products.route))
    rows=[];components=[];fronts=[];uncertainty=[]
    for h in (20,30):
        disc=(np.arange(360)<h*12)*1.03**-(np.arange(360)//12)
        hd=ev.integrate_endpoints(base,fractions,disc,k['hospital'])[...,-1]
        hc=ev.integrate_endpoints(base,fractions,disc,k['hospital_cost'])[...,-1]
        rd=np.einsum('sm,m->s',reported[...,-1],disc)
        allop=rd[None]*k['outpatient'][:,None];displaced=hd*k['outpatient'][:,None]
        delta=lambda a:a-a[:,[ci]]
        program=(cache[f'vaccine_{h}']+cache[f'delivery_{h}']).sum(axis=2)
        np.testing.assert_allclose(program+delta(allop+hc-displaced),cache[f'cost_{h}'],rtol=2e-9,atol=3e-5)
        nd=np.array([sum(r.nasal_doses/1.03**(r.year-2027) for r in receipts[(receipts.strategy_id==a)&(receipts.year<2027+h)].itertuples()) for a in arms])
        for i,a in enumerate(arms):
            components.append(dict(country=country,horizon=h,strategy_id=a,
                hospital_episode_savings=-delta(hc)[:,i].mean(),outpatient_savings=-delta(allop-displaced)[:,i].mean(),
                all_outpatient_savings=-delta(allop)[:,i].mean(),hospital_premium_savings=-delta(hc-displaced)[:,i].mean(),
                standard_programme_savings=-program[:,i].mean(),discounted_nasal_receipts=nd[i]))
        # Risk reduction reallocates episodes to outpatient care. Episode cost varies separately.
        settings=[(s,1.,f'risk_{s:g}') for s in (0.,.25,.5,1.)]+[(1.,u,f'unit_cost_{u:g}') for u in (.5,2.)]
        # Removing gross hospital savings retains the adopted outpatient spending.
        settings.append((None,None,'no_gross_hospital_savings'))
        for s,u,label in settings:
            cost=(cache[f'cost_{h}']-delta(hc)) if s is None else program+delta(allop+s*(u*hc-displaced))
            for burden in ('reported_uniform','symptomatic_differentiated'):
                health=effect_for(cache,h,burden,'included_base')
                for ref in refs.to_dict('records'):
                    lam=float(ref['threshold_intl2024']);tags=dict(country=country,horizon=h,burden=burden,
                        hospital_scenario=label,threshold_id=ref['threshold_id'],threshold=lam)
                    caps=nasal_comparator_ceilings(arms,health,cost,nd,lam,matching)
                    env=nasal_price_frontier(arms,health,cost,nd,lam)
                    for row in env:fronts.append({**tags,**row})
                    for route in bp.ROUTES:
                        best=max((r for r in caps if routes[r['strategy_id']]==route),key=lambda r:r['ceiling_vs_best_standard'])
                        rows.append({**tags,'route':route,**best})
                    nmb=lam*health-cost;means=nmb.mean(axis=0);best=int(means.argmax())
                    tol=128*np.finfo(float).eps*np.maximum(1.,np.abs(nmb).max(axis=1))
                    ties=nmb.max(axis=1)[:,None]-nmb<=tol[:,None]
                    weights=ties/ties.sum(axis=1)[:,None]
                    uncertainty.append({**tags,'nasal_price':0.,'selected_strategy':arms[best],
                        'expected_nmb':float(means[best]),'conditional_optimality_frequency':float(weights[:,best].mean()),
                        'conditional_evpi':float(nmb.max(axis=1).mean()-means[best])})
    for name,data in [('hospital_boundaries',rows),('hospital_components',components),('hospital_frontiers',fronts),('conditional_decisions',uncertainty)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}_{country}.csv',index=False)


def save_annual(data,country,label):
    rows=[];totals=[]
    for y in range(data['reported'].shape[1]):
        for g,age in enumerate(ev.AGE_GROUPS):
            for metric in ('deaths','dalys'):
                values=data['deaths'][:,1,y,g] if metric=='deaths' else data['yll'][:,1,y,g]+data['yld_reported_uniform'][1,y,g]
                rows.append(dict(country=country,scenario=label,year=2027+y,age_group=age,metric=metric,**describe(values,'averted')))
    for h in (10,20,30,40,50):
        if h>data['reported'].shape[1]:continue
        for g,age in enumerate(ev.AGE_GROUPS):
            d=data['deaths'][:,1,:h,g].sum(axis=1)
            v=data['yll'][:,1,:h,g].sum(axis=1)+data['yld_reported_uniform'][1,:h,g].sum()
            totals.append(dict(country=country,scenario=label,horizon=h,age_group=age,
                **describe(d,'deaths_averted'),**describe(v,'dalys_averted'),
                reported_cases_averted=data['reported'][1,:h,g].sum(),symptomatic_cases_averted=data['symptomatic'][1,:h,g].sum()))
    pd.DataFrame(rows).to_csv(OUT/f'annual_{country}_{label}.csv',index=False)
    pd.DataFrame(totals).to_csv(OUT/f'long_totals_{country}_{label}.csv',index=False)

def long_stage(country,history='adopted',state='unchanged'):
    ctx=setup(country,2077,history);info,inputs,p,h,sc,cat,dates,days=ctx
    y=h.y.copy()
    if state!='unchanged':
        src,dst=(R,W) if state=='more_waned' else (W,R)
        moved=.1*y[:,src].copy();y[:,src]-=moved;y[:,dst]+=moved
        np.testing.assert_allclose(y.sum(axis=1),h.y.sum(axis=1),rtol=1e-14,atol=1e-8)
    profile=next(x for x in bp.load_design(bp.CONFIG)['profiles'] if x['candidate_id']==PID)
    raw0,_=simulate(ctx,initial=y)
    raw1,_=simulate(ctx,profile['efficacy'],profile['infectiousness_multiplier'],initial=y)
    label=history+'_'+state
    if history=='adopted' and state=='unchanged':
        with np.load(bp.adopted.OUT/country/'projection_arrays.npz') as z:
            j=z['strategy_ids'].astype(str).tolist().index('current')
            np.testing.assert_allclose(raw0[:360],z['onsets'][j],rtol=2e-8,atol=2e-4)
        cat=pd.read_csv(bp.OUT/country/'profiles'/PID/'product_catalog.csv')
        product=cat[(cat.route=='standalone')&(cat.matched_standard_strategy=='current')&np.isclose(cat.nasal_coverage,.9)].iloc[0]
        with np.load(bp.OUT/country/'profiles'/PID/product.trajectory_path) as z:
            np.testing.assert_allclose(raw1[:360],z['onsets'],rtol=2e-8,atol=2e-4)
    data=clinical(country,np.stack([raw0,raw1]),dates[:-1],dates[1:],info)
    if history=='adopted' and state=='unchanged':
        with np.load(bp.OUT/country/'profiles'/PID/'raw/paired_components.npz') as z:
            j=z['strategy_ids'].astype(str).tolist().index(product.strategy_id)
            for horizon in (20,30):
                np.testing.assert_allclose(data['yll'][:,1,:horizon,-1].sum(axis=1),z[f'yll_{horizon}'][:,j],rtol=1e-8,atol=1e-5)
                np.testing.assert_allclose(data['yld_reported_uniform'][1,:horizon,-1].sum(),z[f'yld_{horizon}_reported_uniform'][j],rtol=1e-8,atol=1e-6)
    np.savez_compressed(OUT/f'long_raw_{country}_{label}.npz',current=raw0,nasal=raw1,dates=dates.to_numpy())
    save_annual(data,country,label)


def matched_stage(country,perturb='reference'):
    ctx=list(setup(country));info,inputs,p,h,sc,cat,dates,days=ctx
    if perturb.startswith('r0_'):
        ctx[2]=replace(p,R0=p.R0*float(perturb.split('_')[1]))
    elif perturb.startswith('contact_'):
        scale=float(perturb.split('_')[1])
        # Perturb adolescent mixing prospectively, both directions. Historical
        # contacts and the reference-year transmission scale remain unchanged.
        age=inputs.grid.contact_group
        # Contact groups are five-year bands: 10-19 are groups 2 and 3.
        for scenario in sc.values():
            for field in ('contacts_home','contacts_nonhome'):
                a=getattr(scenario,field).copy();start=2027-inputs.calendar.first_year
                factors=np.ones(a.shape[1]);factors[2:4]=np.sqrt(scale)
                a[start:]*=factors[None,:,None]*factors[None,None,:]
                setattr(scenario,field,a)
    current,_=simulate(ctx)
    cov=max(.7,float(cat[cat.strategy_id=='current'].adolescent_coverage.iloc[0]))
    # Match within the attainable effects from zero to the reference b=.5 endpoint.
    endpoints={a:next(x for x in bp.load_design(bp.CONFIG)['profiles'] if x['candidate_id']==f'b050_a{a:03d}_d03_point') for a in (0,100)}
    cache={}
    def evaluate(a,strength):
        key=(a,float(strength))
        if key not in cache:
            pr=endpoints[a];eff=1-(1-pr['efficacy'])**strength;on=pr['infectiousness_multiplier']**strength
            raw,_=simulate(ctx,eff,on,cov)
            # Case counts only during root search, full clinical evaluation once matched.
            logits,ratio=ev._reporting(info,ROOT)
            reporting=ev.monthly_reporting_weights(AgeGrid(),logits,dates[:-1],dates[1:])
            cases=np.einsum('mb,mbg->mg',(current-raw)[...,0]+ratio*(current-raw)[...,1],reporting)
            cache[key]=(raw,cases)
        return cache[key]
    rows=[]
    for horizon in (20,30):
        gains={a:evaluate(a,1.)[1][:horizon*12,2].sum() for a in (0,100)}
        # Half the smaller endpoint gives an interior common target. Bracket is
        # certified explicitly; this estimates one contrast, not a unique bridge.
        target=.5*min(gains.values())
        if target<=0:
            rows.append(dict(country=country,perturbation=perturb,horizon=horizon,status='no_positive_endpoint_overlap'))
            continue
        found={}
        for a in (0,100):
            f=lambda x:evaluate(a,x)[1][:horizon*12,2].sum()-target
            root=brentq(f,0.,1.,xtol=1e-7,rtol=1e-8)
            found[a]=(root,*evaluate(a,root))
        data=clinical(country,np.stack([current,found[0][1],found[100][1]]),dates[:-1],dates[1:],info)
        on=data['yll'][:,1,:horizon,-1].sum(axis=1)+data['yld_reported_uniform'][1,:horizon,-1].sum()
        ac=data['yll'][:,2,:horizon,-1].sum(axis=1)+data['yld_reported_uniform'][2,:horizon,-1].sum()
        err=max(abs(found[a][2][:horizon*12,2].sum()-target) for a in (0,100))
        assert err<max(1e-3,abs(target)*1e-6)
        rows.append(dict(country=country,perturbation=perturb,horizon=horizon,status='matched',nasal_coverage=cov,
            target_adolescent_cases_averted=target,max_matching_error=err,onward_strength=found[0][0],acquisition_strength=found[100][0],
            onward_endpoint_adolescent_gain=gains[0],acquisition_endpoint_adolescent_gain=gains[100],
            onward_infant_cases_averted=data['reported'][1,:horizon,0].sum(),acquisition_infant_cases_averted=data['reported'][2,:horizon,0].sum(),
            **describe(on,'onward_dalys'),**describe(ac,'acquisition_dalys'),**describe(on-ac,'onward_minus_acquisition_dalys')))
    pd.DataFrame(rows).to_csv(OUT/f'matched_{country}_{perturb}.csv',index=False)

def maternal_stage(country):
    ctx=setup(country);info,inputs,p,h,sc,cat,dates,days=ctx
    current=cat[cat.strategy_id=='current'].iloc[0]
    subset=cat[(np.isclose(cat.adolescent_coverage,current.adolescent_coverage)|np.isclose(cat.maternal_coverage,current.maternal_coverage))]
    rows=[]
    # Halve the symptom-protection contrast with unprotected typical infection;
    # historical state, coverage and protection duration remain fixed.
    for scale in (.5,1.):
        params=replace(p,pi_maternal=1-scale*(1-p.pi_maternal))
        arms=['current']+[a for a in subset.strategy_id if a!='current'];raw=[];doses=[]
        for a in arms:
            sim=Simulator(sc[a]);res=sim.run(params,h.y,h.acc,days[0],days[-1],record_days=days)
            raw.append(np.diff(res.acc_records,axis=0))
            doses.append(bp.prior.annual_doses(inputs,sc[a],res,info.get('maternal_effective_share',.9),2027,2057))
        data=clinical(country,np.stack(raw),dates[:-1],dates[1:],info);doses=np.stack(doses)
        for horizon in (20,30):
            dd=(doses[:,:,:horizon]@1.03**-np.arange(horizon)).sum(axis=1);dd-=dd[0]
            for j,a in enumerate(arms[1:],1):
                r=cat[cat.strategy_id==a].iloc[0]
                gain=data['yll'][:,j,:horizon,-1].sum(axis=1)+data['yld_reported_uniform'][j,:horizon,-1].sum()
                rows.append(dict(country=country,horizon=horizon,protection_contrast_scale=scale,strategy_id=a,
                    programme='M' if r.maternal_coverage>current.maternal_coverage else 'A',
                    **describe(gain/dd[j]*1000,'dalys_per_1000_discounted_receipts')))
    pd.DataFrame(rows).to_csv(OUT/f'maternal_protection_{country}.csv',index=False)


def task(job):
    stage,*args=job;OUT.mkdir(parents=True,exist_ok=True)
    label='_'.join(job);meta=OUT/(label+'_metadata.json')
    if meta.exists():
        return label+' reused'
    opened=set();written=set()
    def capture(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            p=Path(args[0]).resolve()
            if p.is_relative_to(ROOT) and p.suffix in ('.csv','.gz','.json','.yaml','.npz'):
                if isinstance(args[1],str) and any(c in args[1] for c in ('w','a','+')):written.add(p)
                else:opened.add(p)
    sys.addaudithook(capture)
    started=time.time()
    {'dose':dose_stage,'hospital':hospital_stage,'long':long_stage,'matched':matched_stage,'maternal':maternal_stage}[stage](*args)
    sources=(opened-written)|{Path(__file__).resolve(),ev.COST,ev.SEV,bp.CONFIG,Path(ev.__file__).resolve(),ROOT/'src_python/pertussis/extensions/bpze1.py'}
    record=dict(job=job,seconds=time.time()-started,
        input_files=[str(p.relative_to(ROOT)) for p in sorted(sources) if p.exists()],
        output_files=[str(p.relative_to(OUT)) for p in sorted(written) if p.exists() and p.is_relative_to(OUT)])
    meta.write_text(json.dumps(record,indent=2)+'\n')
    return label+' completed '+str(round(record['seconds'],1))+'s'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['dose','hospital','long','matched','maternal','all','check'],default='all')
    parser.add_argument('--countries',nargs='+',default=list(COUNTRIES));parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.stage=='check':
        files=list(OUT.glob('*_metadata.json'));assert files
        for file in files:
            m=json.loads(file.read_text())
            for name in m.get('output_files',[]):assert (OUT/name).is_file(),name
        print('Verified',len(files),'analysis jobs');return
    jobs=[]
    for stage in (['dose','hospital','long','matched','maternal'] if args.stage=='all' else [args.stage]):
        if stage=='dose':jobs.append(('dose',));continue
        for country in args.countries:
            if stage=='long':
                if country not in ('Japan','Brazil'):continue
                jobs += [(stage,country,h,'unchanged') for h in ('adopted','baseline','recovery_365')]
                jobs += [(stage,country,'adopted',s) for s in ('more_waned','more_recovered')]
            elif stage=='matched':
                jobs.append((stage,country,'reference'))
                if country in ('Sweden','China'):
                    jobs += [(stage,country,s) for s in ('r0_0.8','r0_1.2','contact_0.8','contact_1.2')]
            else:jobs.append((stage,country))
    # Fresh worker per task keeps the read/write provenance hook task-local.
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
        futures=[pool.submit(task,j) for j in jobs]
        for f in as_completed(futures):print(f.result(),flush=True)

if __name__=='__main__':main()
