"""Explore a held-2026 detection assumption in future reported-scale valuation.

Post-processing only: transmission, intervention receipts and infant per-onset
fatality risks stay fixed; published-source multipliers are uncertain proxies,
not independent evidence about future testing or care seeking. CFR/hospital
proportions per report are held fixed and may not be transportable when
ascertainment changes. This is not an alternative validated main analysis.
"""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from src_python.pertussis.config import REPO_ROOT
from src_python.pertussis.inference.core2026 import _diagnostic_multiplier_frame,_dated_multipliers
from src_python.pertussis.economics.loader import load_costs,load_severity_priors
from src_python.pertussis.economics import unified_eval as ue
from scripts.complete_core2026_unified_analysis import reference_risk_for,validate_australia_risk
ROOT=REPO_ROOT;DATA=ROOT/'outputs/core2026/v2_unified_national_geography_candidate_20261102'
OUT=ROOT/'outputs/core2026/v2_future_detection_sensitivity_20261102'
COSTS=ROOT/'config/core2026/economics/costs_unified_provenance_20261102.yaml'
SEVERITY=ROOT/'config/core2026/economics/severity_priors.yaml'
AURISK=ROOT/'outputs/core2026/v2_australia_mortality_20261102/risk_draws.csv.gz'
COUNTRIES=('Australia','England','Sweden','Japan','New_Zealand','China','Brazil','Thailand')
KEYS=['country','strategy_id']
METRICS=['confirmed_cases_averted','all_symptomatic_onsets_averted','hospitalisations_averted','deaths_averted','yll_averted','yld_averted','dalys_averted','incremental_cost','incremental_cost_infant','icer_ratio_of_means']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scaled_logits(logits,multiplier):
    multiplier=float(multiplier)
    if not np.isfinite(multiplier) or multiplier<=0:raise ValueError('multiplier must be finite and positive')
    p=expit(np.asarray(logits,dtype=float))*multiplier
    if not np.isfinite(p).all() or (p<=0).any() or (p>=1).any():raise ValueError('scaled reporting probabilities must lie strictly between 0 and 1')
    return logit(p)
@contextmanager
def detection_multiplier(multiplier):
    original=ue._reporting
    def adjusted(completion,repo_root):
        logits,ratio=original(completion,repo_root)
        return scaled_logits(logits,multiplier),ratio
    ue._reporting=adjusted
    try:yield
    finally:ue._reporting=original

def main():
    if OUT.exists():raise FileExistsError(OUT)
    audit=json.loads((DATA/'numerical_audit_v2.json').read_text());spec=json.loads((DATA/'run_spec.json').read_text())
    if audit['status']!='passed' or spec['n_draws']!=1000 or spec['seed']!=20261101:raise ValueError('candidate results or sampling settings differ')
    for name,digest in spec['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'economic input changed: {name}')
    validate_australia_risk(AURISK)
    costs,severity=load_costs(COSTS),load_severity_priors(SEVERITY)
    life_path=ROOT/costs['global']['life_table']['csv'];life=pd.read_csv(life_path)
    existing=pd.read_csv(DATA/'summary.csv');existing=existing[(existing.analysis=='main')&(existing.coverage==.7)&(existing.horizon==20)&(existing.projection_scenario=='fitted')]
    if len(existing)!=len(COUNTRIES)*4:raise ValueError('incomplete baseline summary')
    results=[];meta_records=[];source_paths=[Path(__file__).resolve(),DATA/'run_spec.json',DATA/'summary.csv',DATA/'numerical_audit_v2.json',COSTS,SEVERITY,life_path,AURISK,AURISK.parent/'run_spec.json',ROOT/'src_python/pertussis/economics/unified_eval.py',ROOT/'src_python/pertussis/inference/core2026.py',ROOT/'data/raw/pertussis_diagnostic_standards_timeline.csv']
    for country in COUNTRIES:
        records=[x for x in spec['evaluations'] if x['country']==country and x['coverage']==.7 and x['horizon']==20 and x['analysis']=='main' and x['projection_scenario']=='fitted']
        if len(records)!=1:raise ValueError(f'{country}: scenario metadata mismatch')
        folder=ROOT/records[0]['scenario_dir'];completion=json.loads((folder/'complete.json').read_text());fit=ROOT/completion['fit_directory']
        timeline=_diagnostic_multiplier_frame(country,geographic_policy='national_only')
        recent=float(_dated_multipliers(['2026-12-31'],['2027-01-01'],timeline)[0])
        risk,scenario=reference_risk_for(country,'2001_2023',AURISK)
        results_by_label={}
        for label,multiplier in [('reference_1',1.),('held_2026',recent)]:
            with detection_multiplier(multiplier):
                draws,_,meta=ue.evaluate_common(country,folder,severity,costs,life,n_draws=1000,seed=20261101,horizon=20,
                    make_breakdown=False,risk_file=risk,risk_scenario=scenario,inpatient_fallback='documented_extrapolation')
                summary,_,_,_=ue.summarise_common(draws,meta)
            results_by_label[label]=summary.set_index('strategy_id')
            results.append(summary.assign(sensitivity=label,future_detection_multiplier=multiplier))
            del draws
        a=results_by_label['reference_1'];b=results_by_label['held_2026'];saved=existing[existing.country==country].set_index('strategy_id')
        for metric in METRICS:
            np.testing.assert_allclose(a[metric].sort_index(),saved[metric].sort_index(),rtol=1e-9,atol=1e-7,equal_nan=True,err_msg=f'{country}: {metric} baseline mismatch')
        np.testing.assert_allclose(b.confirmed_cases_averted,a.confirmed_cases_averted*recent,rtol=1e-9,atol=1e-6)
        np.testing.assert_allclose(b.yld_averted,a.yld_averted*recent,rtol=1e-9,atol=1e-7)
        np.testing.assert_allclose(b.all_symptomatic_onsets_averted,a.all_symptomatic_onsets_averted,rtol=1e-10,atol=1e-6)
        death_scale=recent if country in ('Brazil','Thailand') else 1.
        np.testing.assert_allclose(b.yll_averted,a.yll_averted*death_scale,rtol=1e-9,atol=1e-7)
        meta_records.append({'country':country,'held_multiplier':recent,'timeline_has_rows':bool(len(timeline)),'death_pathway':meta['death_pathway'],'scenario_dir':str(folder.relative_to(ROOT)),'fit_dir':str(fit.relative_to(ROOT))})
        source_paths.extend([folder/'complete.json',folder/'annual_age_onsets.csv.gz',folder/'receipts.csv.gz',fit/'fit_summary.csv'])
        if risk:source_paths.extend([risk,risk.parent/'run_spec.json'])
        print(country,'done',recent,flush=True)
    result=pd.concat(results,ignore_index=True)
    a=result[result.sensitivity=='reference_1'];b=result[result.sensitivity=='held_2026']
    compare=a.merge(b,on=KEYS,suffixes=('_reference','_held'),validate='one_to_one')
    OUT.mkdir(parents=True);result.to_csv(OUT/'all_strategies.csv',index=False);compare.to_csv(OUT/'paired_detection_sensitivity.csv',index=False)
    info={'status':'completed_conditional_sensitivity','countries':meta_records,'sample_draws':1000,'seed':20261101,'horizon_years':[2027,2046],
      'definition':'Hold the detection multiplier in the input timeline on 2026-12-31 throughout 2027–2046; if no national row exists use 1.0. Existing main valuation uses 1.0.',
      'checks':['baseline reproduces saved point-based economics','reported cases and YLD scale with multiplier','biological symptomatic incidence unchanged','YLL follows recorded course-risk pathway','scaled reporting probabilities strictly below 1'],
      'limitations':['The historical detection timeline is an assumed proxy, not measured age-specific 2027 reporting.','China 2.25 draws on screening evidence in medically attended children; applying it equally at all ages is an explicitly unvalidated sensitivity.','Brazil and Thailand recent multipliers are labelled uncertain in the original timeline; this is not validated national evidence.','Per-reported-case hospitalization and CFR risks and unit costs are held fixed although changes in case ascertainment can change severity mix.','Only point transmission and conditional cost/severity draws; no transmission posterior or independent model validation.','England has no same-name timeline and 1.0 is a default, not evidence of stable detection.'],
      'sha256':{str(p.resolve().relative_to(ROOT)):sha(p) for p in sorted(set(source_paths))}}
    (OUT/'metadata.json').write_text(json.dumps(info,indent=2)+'\n');print('Conditional future detection sensitivity completed.',flush=True)
if __name__=='__main__':main()
