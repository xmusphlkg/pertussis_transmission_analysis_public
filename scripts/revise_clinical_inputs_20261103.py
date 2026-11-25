"""Age-matched Swedish hospital input and explicit borrowed noninfant sensitivity.

The severity document is compatible with existing economics loaders. Mortality
overlays are intentionally opt-in and NEVER inserted into countries.entries:
doing so would replace, rather than supplement, the calibrated infant pathway.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src_python.pertussis.config import REPO_ROOT as ROOT
from src_python.pertussis.economics.evaluate import (
    sample_bin_values, sample_entry, sample_cost, life_expectancy, severity_doc_iso3,
)
from src_python.pertussis.economics.loader import load_costs, load_severity_priors
from src_python.pertussis.economics.monthly_age_severity import clinical_risk_endpoints
from src_python.pertussis.economics.joint_age_severity import joint_risk_weights
from src_python.pertussis.economics.unified_eval import read_common_projection, _reporting
from src_python.pertussis.economics.programme_eval import risk_file_infant_risks
from src_python.pertussis.grid import AgeGrid
from src_python.pertussis.scenarios.aggregation import AGE_GROUPS

BASE_SEVERITY = ROOT/'config/core2026/economics/severity_priors_provenance_20261102.yaml'
REVISED_SEVERITY = ROOT/'config/core2026/economics/severity_revised_sensitivity_20261103.yaml'
COSTS = ROOT/'config/core2026/economics/costs_unified_provenance_20261102.yaml'
OUT = ROOT/'outputs/core2026/clinical_input_revision_20261103'
MISSING_NONINFANT = ('Australia', 'England', 'Sweden', 'Japan', 'New_Zealand', 'China')
SWEDEN_URL = 'https://onlinelibrary.wiley.com/doi/10.1111/apa.70300'


def build_revised_severity(base_doc):
    doc = deepcopy(base_doc)
    country = doc['countries']['Sweden']
    country['entries'] = [e for e in country.get('entries', [])
                          if e['quantity'] != 'hospitalisation_proportion']
    country['entries'].append(dict(
        quantity='hospitalisation_proportion', basis='per_confirmed_case',
        age_start_years=0.0, age_end_years=0.5, distribution='beta',
        params=dict(alpha=218.5, beta=127.5), point=218/345,
        source=dict(citation='Aronsson B et al. Acta Paediatrica. DOI 10.1111/apa.70300; online 2025-09-15; print 2026;115:2566-2571.',
                    url=SWEDEN_URL, locator='Methods 2.1; Results 3.1 and 3.2; Table 1',
                    period='2009-2015', verification='fulltext', accessed='2026-11-03'),
        notes='Historical sensitivity among 345 included surviving laboratory-confirmed infants aged <6 months; 218 hospitalised for at least one night. Deaths excluded; two further cases excluded for missing source-of-infection information. Beta(218.5,127.5) uses Jeffreys half-count prior and represents sampling uncertainty within this cohort, not temporal/selection transportability. Constant within <6 months is an assumption. Ages 6-11 months use explicit Brazil hospital donor; no extrapolation of this Swedish fraction above 6 months.'))
    country['gaps'] = [g for g in country.get('gaps', []) if g['quantity'] != 'hospitalisation_proportion']
    country['gaps'].append(dict(quantity='hospitalisation_proportion', age_groups='6 months and older',
        status='no_age_matched_national_source_in_this_revision',
        reason='Aronsson study covers only surviving infants younger than 6 months.',
        borrowing_rule='Retain explicitly labelled Brazil national age-specific hospital donor.'))
    doc['revision_20261103'] = dict(
        purpose='Correct source subgroup and age boundary; historical survivor-cohort sensitivity, not validated current national severity.',
        original_source=str(BASE_SEVERITY.relative_to(ROOT)),
        noninfant_mortality_main='Unestimated in six countries; calibrated infant mortality unchanged.',
        optional_noninfant_overlay=dict(name='Brazil_recorded_outcome_CFR_transfer',
            countries=list(MISSING_NONINFANT), age_start_years=1.0,
            implementation='scripts.revise_clinical_inputs_20261103.sample_noninfant_risk_endpoints',
            status='structural_cross_country_transfer_sensitivity_not_national_estimate',
            multiplier_scenarios=[0.5, 1.0, 2.0],
            multiplier_interpretation='Analyst-selected stress tests, not confidence limits or probabilities.',
            limitation='Brazil denominator is confirmed cases with recorded outcomes; transport to other surveillance endpoints is an explicit unvalidated assumption.'))
    return doc


def noninfant_cfr_entries(severity_doc, donor='Brazil'):
    """Return donor entries clipped to >=1y, with unambiguous transfer labels."""
    result=[]
    for entry in severity_doc['countries'][donor].get('entries', []):
        end=entry.get('age_end_years')
        if entry['quantity'] != 'cfr' or (end is not None and float(end)<=1):
            continue
        e=deepcopy(entry)
        e['age_start_years']=max(1.,float(e['age_start_years']))
        e['borrowed_from']=donor
        e['transfer_status']='unvalidated_noninfant_surveillance_CFR_transfer_sensitivity'
        result.append(e)
    if not result:
        raise ValueError('No noninfant donor CFR entries')
    return result


def sample_noninfant_risk_endpoints(country, severity_doc, costs_doc, life_table,
        grid, groups, logits, *, n_draws=1000, seed=20261103, discount_rate=.03,
        donor='Brazil', multiplier=1.0):
    """Return extra death/YLL weights (draw,bin,group) at year endpoints.

    Weights already contain reporting probability: multiply by typical +
    mild_reporting_ratio*mild, NOT by already reported cases. Use
    monthly_clinical_outcomes with month midyear fractions, annual-discount
    YLL at the caller, and add to existing infant YLL/deaths. A common seed
    gives shared donor risk draws across countries; paired arms reuse weights.
    Brazil/Thailand return zero overlays to avoid counting mortality twice.
    """
    if n_draws<1 or not np.isfinite(multiplier) or multiplier<0 or discount_rate<0:
        raise ValueError('Invalid draw count, multiplier or discount rate')
    entries=noninfant_cfr_entries(severity_doc,donor)
    zero=np.zeros((n_draws,grid.n_bins,len(groups)))
    if country not in MISSING_NONINFANT:
        return dict(death_early=zero.copy(),death_late=zero.copy(),
                    yll_early=zero.copy(),yll_late=zero.copy(),
                    death_midyear=zero.copy(),yll_midyear=zero.copy(),
                    applied=False,status='existing_national_CFR_no_overlay')
    cfr,use_own,samples=sample_bin_values(entries,grid,np.random.default_rng(seed),n_draws,
                                        return_entry_samples=True)
    if np.any(cfr[:,:12]):
        raise AssertionError('Noninfant overlay entered infant bins')
    # Preserve a valid probability even for optional structural multipliers.
    scaled_samples=np.minimum(samples[0]*multiplier,1.)
    scaled_bins=np.minimum(cfr*multiplier,1.)
    args=(grid,groups,logits,entries,[],scaled_samples,samples[1],use_own,scaled_bins)
    death_early,death_late=clinical_risk_endpoints(*args)
    ages,ex=life_expectancy(life_table,severity_doc_iso3(costs_doc,country))
    def yll_at_age(b,age):
        remaining=np.interp(age,ages,ex)
        return remaining if discount_rate==0 else -np.expm1(-discount_rate*remaining)/discount_rate
    yll_early,yll_late=clinical_risk_endpoints(*args,segment_multiplier=yll_at_age)
    death_mid=joint_risk_weights(*args)
    yll_mid=joint_risk_weights(*args,segment_multiplier=yll_at_age)
    np.testing.assert_allclose((death_early+death_late)/2,death_mid,atol=1e-14)
    np.testing.assert_allclose((yll_early+yll_late)/2,yll_mid,atol=1e-13)
    return dict(death_early=death_early,death_late=death_late,
                yll_early=yll_early,yll_late=yll_late,
                death_midyear=death_mid,yll_midyear=yll_mid,
                applied=True,status='borrowed_noninfant_CFR_sensitivity_not_national_estimate',
                donor=donor,multiplier=multiplier,seed=seed,
                sampling='donor_beta_sampling_only_no_transfer_or_transmission_uncertainty')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_config():
    doc=build_revised_severity(load_severity_priors(BASE_SEVERITY))
    text=yaml.safe_dump(doc,sort_keys=False,allow_unicode=True,width=100)
    if REVISED_SEVERITY.exists() and REVISED_SEVERITY.read_text()!=text:
        raise FileExistsError('Refusing to overwrite a different revision configuration')
    REVISED_SEVERITY.write_text(text)
    load_severity_priors(REVISED_SEVERITY)
    return doc


def sweden_paired_hospital_revision(severity, revised, costs, grid, *, n=1000):
    """Preserve original donor/cost draws; alter only Sweden's infant risks."""
    from scripts.complete_core2026_unified_analysis import reference_risk_for
    rng=np.random.default_rng(20261101)
    own=[e for e in severity['countries']['Sweden']['entries'] if e['quantity']=='hospitalisation_proportion']
    donor=[e for e in severity['countries']['Brazil']['entries'] if e['quantity']=='hospitalisation_proportion']
    old,_,samples=sample_bin_values(own,grid,rng,n,donor_entries=donor,return_entry_samples=True)
    path,scenario=reference_risk_for('Sweden','2001_2023',ROOT/'outputs/core2026/v2_australia_mortality_20261102/risk_draws.csv.gz')
    risk_file_infant_risks(path,scenario,rng,n,grid)
    items={x['item']:x for x in costs['countries']['Sweden']['items']}
    prices={k:sample_cost(items[k],rng,n) for k in ('vaccine_price_maternal','vaccine_price_adolescent','delivery_maternal','delivery_adolescent')}
    outpatient=sample_cost(items['outpatient_episode'],rng,n)
    los=[e for e in severity['countries']['China']['entries'] if e['quantity']=='los_days']
    young=sample_entry(min(los,key=lambda e:float(e['age_start_years'])),rng,n)
    older=sample_entry(max(los,key=lambda e:float(e['age_start_years'])),rng,n)
    day=sample_cost(items['inpatient_day'],rng,n)
    inpatient=day[:,None]*np.where(grid.upper_years[None,:]<=.5,young[:,None],older[:,None])
    new=old.copy()
    entry=[e for e in revised['countries']['Sweden']['entries'] if e['quantity']=='hospitalisation_proportion'][0]
    new[:,:6]=sample_entry(entry,np.random.default_rng(20261103),n)[:,None]
    # Explicitly recover the original Brazil 2-11-month sampled donor values.
    assert float(donor[1]['age_start_years'])<.5 and float(donor[1]['age_end_years'])==1
    new[:,6:12]=samples[1][:,1,None]
    np.testing.assert_array_equal(new[:,12:],old[:,12:])
    return old,new,inpatient,outpatient,prices,path


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    revised=write_config()
    severity=load_severity_priors(BASE_SEVERITY)
    costs=load_costs(COSTS)
    life_path=ROOT/costs['global']['life_table']['csv'];life=pd.read_csv(life_path)
    grid=AgeGrid();n=1000
    donor_path=ROOT/'data/processed/brazil_severity_20261028/brazil_severity_pooled_2016_2025.csv'
    donor_table=pd.read_csv(donor_path).set_index('age_group')
    donor_rows=[]
    for label,e in zip(('1-4','5-9','10-19','20-59','60+'),noninfant_cfr_entries(severity)):
        r=donor_table.loc[label];events=int(r.pertussis_deaths);denom=int(r.recorded_outcomes)
        assert e['params']==dict(alpha=events+.5,beta=denom-events+.5)
        donor_rows.append(dict(age_group=label,pertussis_deaths=events,recorded_outcomes=denom,
            confirmed_cases=int(r.confirmed_cases),alpha=events+.5,beta=denom-events+.5,
            observed_cfr=events/denom,beta_mean=(events+.5)/(denom+1),
            transfer_status='sensitivity_not_recipient_country_estimate'))
    swe_old,swe_new,inpatient,outpatient,prices,risk_path=sweden_paired_hospital_revision(severity,revised,costs,grid)
    spec_path=ROOT/'outputs/core2026/v2_joint_cost_yll_full_candidate_20261102/run_spec.json'
    spec=json.loads(spec_path.read_text())
    anchors_path=ROOT/'outputs/core2026/eight_country_nhb_results_20261103/threshold_anchors.csv'
    anchors=pd.read_csv(anchors_path)
    old_swe_path=spec_path.parent/'paired_economic_draws_Sweden.csv.gz'
    old_swe=pd.read_csv(old_swe_path,dtype={'strategy_id':str})
    paths=[Path(__file__).resolve(),BASE_SEVERITY,REVISED_SEVERITY,COSTS,life_path,donor_path,
           spec_path,anchors_path,old_swe_path,risk_path]
    records=[];overlay_rows=[];hospital_rows=[];decision_rows=[];draw_parts=[]
    countries=sorted(costs['countries'])
    for country in countries:
        candidates=[e for e in spec['evaluations'] if e['country']==country and e.get('coverage')==.7
                    and e.get('analysis')=='main' and e.get('years')==[2027,2046] and e.get('duration_scope')=='fitted']
        assert len(candidates)==1
        folder=Path(candidates[0]['scenario_dir'])
        if country=='Japan':
            folder=ROOT/'outputs/core2026/v2_japan_official_age_structural_policy_sensitivity_20261102/baseline/projection'
        paths.extend([folder/'complete.json',folder/'annual_age_onsets.csv.gz',folder/'receipts.csv.gz'])
        for horizon in (20,30):
            years=np.arange(2027,2027+horizon);disc=1.03**-np.arange(horizon)
            arms,typical,mild,doses,meta=read_common_projection(folder,years)
            logits,ratio=_reporting(meta,ROOT)
            base_onsets=typical+ratio*mild
            suffix='_30y' if horizon==30 else ''
            name='v2_japan_official_age_monthly_economics' if country=='Japan' else 'v2_monthly_candidate_economics'
            paired_path=ROOT/f'outputs/core2026/{name}{suffix}_20261102/{country}_paired.csv.gz'
            paths.append(paired_path)
            paired=pd.read_csv(paired_path,dtype={'strategy_id':str})
            paired=paired[paired.scope.eq('all_ages')]
            effects=np.zeros((n,len(arms)));csts=np.zeros_like(effects)
            for j,arm in enumerate(arms):
                if arm=='00':continue
                x=paired[paired.strategy_id.eq(arm)].sort_values('draw')
                assert x.draw.tolist()==list(range(n))
                effects[:,j]=x.monthly_dalys_averted;csts[:,j]=x.monthly_incremental_cost
            extra_cost=np.zeros_like(csts)
            if country=='Sweden':
                report=1/(1+np.exp(-logits[0]))
                reported=base_onsets*report
                for j,arm in enumerate(arms):
                    difference=reported[arms.index('00')]-reported[j]
                    diff_d=np.einsum('yb,y->b',difference,disc)
                    diff_u=difference.sum(axis=0)
                    extra_cost[:,j]=-np.sum(diff_d[None,:]*(swe_new-swe_old)*(inpatient-outpatient[:,None]),axis=1)
                    old_h=(diff_u[None,:]*swe_old)[:,:12].sum(axis=1)
                    new_h=(diff_u[None,:]*swe_new)[:,:12].sum(axis=1)
                    hospital_rows.append(dict(country=country,horizon=horizon,strategy_id=arm,
                        original_infant_hospitalisations_averted=old_h.mean(),
                        revised_infant_hospitalisations_averted=new_h.mean(),
                        change_in_incremental_treatment_cost=extra_cost[:,j].mean(),
                        infant_death_yll_and_reported_yld_change=0.,mapping='infant_month_bins_exact'))
                    if horizon==20:
                        r=old_swe[old_swe.strategy_id.eq(arm)].sort_values('draw')
                        for item,col in [('vaccine_price_maternal','unit_price_maternal'),('vaccine_price_adolescent','unit_price_adolescent'),('delivery_maternal','unit_delivery_maternal'),('delivery_adolescent','unit_delivery_adolescent')]:
                            np.testing.assert_allclose(prices[item],r[col],rtol=1e-10,atol=1e-8)
                        conf_d=np.einsum('yb,y->b',reported[j],disc)
                        treatment=(conf_d[None,:]*(swe_old*inpatient+(1-swe_old)*outpatient[:,None]))[:,:12].sum(axis=1)
                        np.testing.assert_allclose(treatment,r.infant_treatment_cost,rtol=1e-9,atol=1e-5)
            scenarios=[('original_included_burden',effects,csts),
                       ('age_matched_sweden_hospital',effects,csts+extra_cost)]
            for multiplier in (.5,1.,2.):
                weights=sample_noninfant_risk_endpoints(country,revised,costs,life,grid,AGE_GROUPS,logits,
                                                       multiplier=multiplier)
                deaths=np.einsum('syb,dbg->sdg',base_onsets,weights['death_midyear'],optimize=True)
                yll=np.einsum('syb,dbg,y->sdg',base_onsets,weights['yll_midyear'],disc,optimize=True)
                delta_yll=yll[arms.index('00')]-yll
                delta_deaths=deaths[arms.index('00')]-deaths
                np.testing.assert_array_equal(delta_yll[:,:,0],0.)
                np.testing.assert_allclose(delta_yll[:,:,:-1].sum(axis=2),delta_yll[:,:,-1],rtol=1e-12,atol=1e-10)
                label=f'age_matched_hospital_noninfant_donor_x{multiplier:g}'
                scenarios.append((label,effects+delta_yll[:,:,-1].T,csts+extra_cost))
                for j,arm in enumerate(arms):
                    for g,age in enumerate(AGE_GROUPS):
                        values=delta_yll[j,:,g]
                        overlay_rows.append(dict(country=country,horizon=horizon,strategy_id=arm,
                            multiplier=multiplier,age_group=age,extra_deaths_averted=delta_deaths[j,:,g].mean(),
                            extra_yll_averted=values.mean(),extra_yll_q025=np.quantile(values,.025),
                            extra_yll_q975=np.quantile(values,.975),applied=weights['applied'],
                            source='Brazil_recorded_outcome_CFR' if weights['applied'] else 'none_existing_national_CFR',
                            mapping='annual_midyear_overlay_on_existing_monthly_base',
                            scope='borrowed_sensitivity_no_national_mortality_completion'))
            for label,eff,cost in scenarios:
                for j,arm in enumerate(arms):
                    records.append(dict(country=country,horizon=horizon,strategy_id=arm,scenario=label,
                        dalys_averted=eff[:,j].mean(),incremental_cost=cost[:,j].mean(),
                        dalys_q025=np.quantile(eff[:,j],.025),dalys_q975=np.quantile(eff[:,j],.975),
                        change_dalys_from_original=(eff-effects)[:,j].mean(),
                        change_cost_from_original=(cost-csts)[:,j].mean(),
                        health_scope='included_burden' if 'donor_' not in label else 'included_plus_borrowed_noninfant_sensitivity',
                        age_mapping='base_monthly_infant_revision_exact_noninfant_overlay_annual_midyear'))
                    draw_parts.append(pd.DataFrame(dict(country=country,horizon=horizon,strategy_id=arm,
                        scenario=label,draw=np.arange(n),dalys_averted=eff[:,j],incremental_cost=cost[:,j])))
                for anchor in anchors[anchors.country.eq(country)].itertuples():
                    nhb=eff-cost/anchor.threshold;mean=nhb.mean(axis=0)
                    win=[arms[i] for i in range(len(arms)) if abs(mean[i]-mean.max())<1e-8]
                    decision_rows.append(dict(country=country,horizon=horizon,scenario=label,
                        threshold_type=anchor.threshold_type,threshold=anchor.threshold,
                        optimal_strategies='|'.join(win),maximum_expected_nhb=mean.max(),
                        threshold_status=anchor.threshold_status))
            print(f'{country} {horizon}y: sensitivity complete',flush=True)
    OUT.mkdir(parents=True)
    pd.DataFrame(donor_rows).to_csv(OUT/'noninfant_donor_parameters.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'health_economic_impact.csv',index=False)
    pd.DataFrame(overlay_rows).to_csv(OUT/'noninfant_mortality_overlay.csv',index=False)
    pd.DataFrame(hospital_rows).to_csv(OUT/'sweden_hospital_revision.csv',index=False)
    pd.DataFrame(decision_rows).to_csv(OUT/'conditional_decisions.csv',index=False)
    pd.concat(draw_parts,ignore_index=True).to_csv(OUT/'paired_sensitivity_draws.csv.gz',index=False)
    metadata=dict(status='completed_age_matched_hospital_and_borrowed_noninfant_sensitivity',
        n_draws=n,projection_scope='fixed_transmission_saved_70percent_five_strategy_set',
        hospital_revision='Sweden <6m cohort fraction; >=6m Brazil donor; no mortality/YLD change',
        noninfant_revision='opt_in_donor_sensitivity_not_national_estimation; retained infant pathways',
        overlay_mapping='annual_midyear on archived annual fine-age onsets; not a fully monthly revaluation',
        donor_multipliers='0.5/1/2 are analyst-selected stress tests, not uncertainty bounds',
        hospital_randomness='original donor, infant mortality, and all cost draws replayed; only Swedish <6m risks independently sampled',
        sources_verified_online_20261103=[SWEDEN_URL,
          'https://www.gov.uk/government/publications/pertussis-laboratory-confirmed-cases-reported-in-england-2025/laboratory-confirmed-cases-of-pertussis-in-england-annual-report-for-2025',
          'https://id-info.jihs.go.jp/surveillance/iasr/pathogens/vol47/551/551r03.html'],
        table_rows=dict(health_economic_impact=len(records),noninfant_overlay=len(overlay_rows),
                        sweden_hospital=len(hospital_rows),decisions=len(decision_rows)),
        sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))})
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
