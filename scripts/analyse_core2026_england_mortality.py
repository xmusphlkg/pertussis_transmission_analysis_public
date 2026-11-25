"""England death risks conditional on one transmission trajectory.

The official linked series observes total infant deaths, not exact age at death.
A two-age Poisson mixture integrates this missing age rather than assigning all
infant deaths to under3m. Annual 2024/2025 deaths are reserved for prediction.
No posterior transmission uncertainty or national policy probability is claimed.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import poisson
from src_python.pertussis.observation.mortality import matched_infant_exposure,coarsened_two_age_risk_draws
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/health_outcomes_20261026/england_linked_infant_deaths.csv'
SCENARIOS={'reference':(1.,0.,.5),'half_ascertainment':(.5,0.,.5),
    'mild_risk_ratio_0.1':(1.,.1,.5),'equal_course_risk':(1.,1.,.5),'uniform_risk_prior':(1.,0.,1.)}


def exposure_by_year(history,years,mild_ratio):
    out=[]
    for year in years:
        row=[]
        for lo,hi in [(0,3),(3,12)]:
            obs=dict(profile='England',geographic_scope='England',start_year=int(year),end_year_exclusive=int(year)+1,
                age_start_months=lo,age_end_months_exclusive=hi,year_mapping='zero_delay_confirmation_to_onset_assumption')
            row.append(matched_infant_exposure(history,obs,mild_risk_ratio=mild_ratio))
        out.append(row)
    return np.asarray(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--risk-draws',type=int,default=4096);p.add_argument('--train-start',type=int,default=2016)
    p.add_argument('--train-end',type=int,default=2023)
    args=p.parse_args()
    analyse(args)


def analyse(args):
    """Return saved conditional frames; callable by the paired ensemble analysis."""
    seed=getattr(args,'seed',20261026)
    if args.train_end>=2024:raise ValueError('this protocol reserves 2024 and2025 death counts')
    meta=json.loads((args.input_dir/'complete.json').read_text())
    if meta['profile']!='England':raise ValueError('England regional trajectory required')
    history=pd.read_csv(args.input_dir/'historical_age_onsets.csv.gz')
    future=pd.read_csv(args.input_dir/'annual_age_onsets.csv.gz');receipts=pd.read_csv(args.input_dir/'receipts.csv.gz')
    obs=pd.read_csv(DATA);train=obs[obs.year.between(args.train_start,args.train_end)]
    if len(train)!=args.train_end-args.train_start+1:raise ValueError('missing training death years')
    if not (train.deaths_lower==train.deaths_upper).all():raise ValueError('exact annual total required')
    checks=[];risks=[];outcomes=[];predictive_draws=[]
    future=future[future.bin_index.lt(12)].assign(risk_age=lambda x:np.where(x.bin_index.lt(3),0,1))
    totals=future.groupby(['strategy_id','year','risk_age'])[['typical_onsets','mild_onsets']].sum()
    for scenario,(ascertainment,ratio,shape) in SCENARIOS.items():
        exposure=exposure_by_year(history,train.year,ratio)
        q=coarsened_two_age_risk_draws(train.deaths_lower.to_numpy(),exposure,ascertainment=ascertainment,
            prior_shape=shape,size=args.risk_draws,seed=seed)
        for j in range(2):
            risks.append(pd.DataFrame(dict(risk_scenario=scenario,risk_draw_id=np.arange(args.risk_draws),risk_age=j,
                p_death_given_typical=q[:,j],p_death_given_mild=q[:,j]*ratio,
                death_ascertainment=ascertainment,prior_shape=shape,
                age_start_months=0 if j==0 else 3,age_end_months_exclusive=3 if j==0 else 12)))
        for r in obs[obs.year.between(args.train_start,2025)].itertuples():
            e=exposure_by_year(history,[r.year],ratio)[0]
            means=ascertainment*(q@e)
            predictive=np.random.default_rng(seed+20000+r.year).poisson(means)
            predictive_draws.append(pd.DataFrame(dict(risk_scenario=scenario,year=r.year,
                risk_draw_id=np.arange(args.risk_draws),observed=r.deaths_lower,
                role='train' if r.year<=args.train_end else 'mortality_holdout',
                expected=means,predictive=predictive)))
            lo,med,hi=np.quantile(predictive,[.025,.5,.975])
            checks.append(dict(risk_scenario=scenario,year=r.year,observed=r.deaths_lower,
                role='train' if r.year<=args.train_end else 'mortality_holdout',
                predictive_lower=lo,predictive_median=med,predictive_upper=hi,
                expected_mean=means.mean(),observed_in_interval=lo<=r.deaths_lower<=hi,
                log_predictive_probability=float(logsumexp(poisson.logpmf(r.deaths_lower,means))-np.log(len(means)))))
        for horizon in [1,3,5,10,20]:
            end=meta['start_year']+horizon;sub=totals[totals.index.get_level_values('year')<end]
            summed=sub.groupby(['strategy_id','risk_age']).sum();burdens={}
            for strategy in future.strategy_id.unique():
                g=summed.loc[strategy].reindex([0,1])
                burdens[strategy]=(g.typical_onsets.to_numpy()+ratio*g.mild_onsets.to_numpy())@q.T
            for strategy,comp in [('baseline','baseline'),('maternal_scaleup','baseline'),
                    ('adolescent_one_time_booster','baseline'),('maternal_scaleup','adolescent_one_time_booster')]:
                averted=burdens[comp]-burdens[strategy]
                dose=receipts[(receipts.strategy_id==strategy)&(receipts.year<end)].additional_receipts.sum() if comp=='baseline' else np.nan
                outcomes.append(pd.DataFrame(dict(profile='England',risk_scenario=scenario,risk_draw_id=np.arange(args.risk_draws),
                    horizon_years=horizon,strategy_id=strategy,comparator=comp,expected_deaths=burdens[strategy],
                    averted=averted,relative_reduction=averted/burdens[comp],additional_receipts=dose,
                    averted_per_1000_receipts=averted*1000/dose if dose>0 else np.nan)))
    out=pd.concat(outcomes,ignore_index=True);summary=[]
    for keys,g in out.groupby(['risk_scenario','horizon_years','strategy_id','comparator']):
        row=dict(zip(['risk_scenario','horizon_years','strategy_id','comparator'],keys))
        for field in ['expected_deaths','averted','relative_reduction','averted_per_1000_receipts']:
            v=g[field].dropna();quantiles=np.quantile(v,[.025,.5,.975]) if len(v) else [np.nan]*3
            row.update({field+'_'+n:x for n,x in zip(['lower','median','upper'],quantiles)})
        row['fraction_positive']=float((g.averted>0).mean());summary.append(row)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    pd.concat(risks).to_csv(args.output_dir/'risk_draws.csv.gz',index=False)
    out.to_csv(args.output_dir/'death_contrasts.csv.gz',index=False)
    pd.DataFrame(summary).to_csv(args.output_dir/'death_summary.csv',index=False)
    pd.DataFrame(checks).to_csv(args.output_dir/'death_predictive_checks.csv',index=False)
    pd.concat(predictive_draws,ignore_index=True).to_csv(args.output_dir/'death_predictive_draws.csv.gz',index=False)
    (args.output_dir/'run_spec.json').write_text(json.dumps(dict(trajectory=meta,death_source_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        train_years=train.year.tolist(),holdout_years=[2024,2025],unknown_age_model='two_age_exact_poisson_mixture',
        risk_age_groups_months=[[0,3],[3,12]],year_mapping='zero_delay_confirmation_to_onset_assumption',
        inference='mortality risk conditional on one trajectory; excludes transmission parameter uncertainty',
        risk_draws=args.risk_draws,seed=seed,scenarios=SCENARIOS),indent=2)+'\n')
    print(f'{len(out)} conditional death comparisons; {len(checks)} annual predictive checks')
    return pd.concat(risks),out,pd.DataFrame(checks)

if __name__=='__main__':main()
