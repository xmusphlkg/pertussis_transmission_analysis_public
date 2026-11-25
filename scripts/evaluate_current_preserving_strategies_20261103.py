"""Monthly clinical/economic valuation for arbitrary current-preserving strategies.

Case ascertainment and disability weights form a 2x2 factorial. A borrowed
noninfant mortality scenario is explicit and does not replace infant risks.
All strategy comparisons share the same clinical and cost draws.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src_python.pertussis.config import REPO_ROOT as ROOT
from src_python.pertussis.grid import AgeGrid
from src_python.pertussis.economics.age_scoped_hospital_cost import MissingAgeCostError, allocate_hospital_cost
from src_python.pertussis.economics.evaluate import (
    DEATH_RISK_SOURCE, HOSPITAL_DONOR, LOS_DONOR, bin_bounds, life_expectancy,
    reference_infant_risks, sample_bin_values, sample_cost, sample_entry,
    severity_doc_iso3, yll_by_bin,
)
from src_python.pertussis.economics.loader import load_costs, load_severity_priors
from src_python.pertussis.economics.monthly_age_severity import clinical_risk_endpoints
from src_python.pertussis.economics.programme_eval import risk_file_infant_risks
from src_python.pertussis.economics.unified_eval import _reporting
from src_python.pertussis.scenarios.aggregation import AGE_GROUPS
from src_python.pertussis.scenarios.monthly_reporting import monthly_reporting_weights
from scripts.complete_core2026_unified_analysis import reference_risk_for
from scripts.revise_clinical_inputs_20261103 import sample_noninfant_risk_endpoints, MISSING_NONINFANT

BASE = ROOT/'outputs/core2026'
COST = ROOT/'config/core2026/economics/costs_unified_provenance_20261102.yaml'
SEV = ROOT/'config/core2026/economics/severity_revised_sensitivity_20261103.yaml'
AU = BASE/'v2_australia_mortality_20261102/risk_draws.csv.gz'
BURDENS = ('reported_uniform', 'reported_differentiated', 'symptomatic_uniform', 'symptomatic_differentiated')


def month_geometry(starts, ends, grid):
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    if (len(starts) == 0 or (starts.day != 1).any() or starts.isna().any() or ends.isna().any()
            or not (starts + pd.offsets.MonthBegin(1)).equals(ends) or not starts[1:].equals(ends[:-1])):
        raise ValueError('Complete calendar months required')
    fractions = []
    for a, b in zip(starts, ends):
        y0, y1 = pd.Timestamp(a.year, 1, 1), pd.Timestamp(a.year+1, 1, 1)
        fractions.append(((a-y0).days+(b-a).days/2)/(y1-y0).days)
    fractions = np.asarray(fractions)
    ages = np.arange(101)
    membership = np.column_stack([(ages >= lo) & (ages < hi) for lo, hi in AGE_GROUPS.values()])
    groups = np.stack([grid.completed_age_weights(f) @ membership for f in fractions])
    np.testing.assert_allclose(groups[:, :, :-1].sum(axis=2), groups[:, :, -1], atol=1e-13)
    return fractions, groups


def integrate_endpoints(base, fractions, time_weights, endpoints):
    """Strategy x month x bin -> draw x strategy x group, without 4D expansion."""
    early = np.einsum('smb,m->sb', base, time_weights*(1-fractions))
    late = np.einsum('smb,m->sb', base, time_weights*fractions)
    return (np.einsum('sb,nbg->nsg', early, endpoints[0], optimize=True)
            + np.einsum('sb,nbg->nsg', late, endpoints[1], optimize=True))


def morbidity_factorial(reported_typical, reported_mild, typical, mild, moderate_unit, mild_unit):
    """Four matched burdens: ascertainment range x episode disability weights."""
    arrays = [np.asarray(x, float) for x in (reported_typical, reported_mild, typical, mild)]
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() or (a < 0).any() for a in arrays):
        raise ValueError('Matched finite nonnegative episode arrays required')
    rt, rm, t, m = arrays
    return dict(zip(BURDENS, ((rt+rm)*moderate_unit, rt*moderate_unit+rm*mild_unit,
                              (t+m)*moderate_unit, t*moderate_unit+m*mild_unit)))


def sample_kernel(country, severity, costs, life, grid, logits, n=1000, seed=20261101, rate=.03):
    rng = np.random.default_rng(seed)
    own = severity['countries'][country].get('entries') or []
    ho = [e for e in own if e['quantity'] == 'hospitalisation_proportion']
    hd = [e for e in severity['countries'][HOSPITAL_DONOR]['entries'] if e['quantity'] == 'hospitalisation_proportion']
    hosp, use, samples = sample_bin_values(ho, grid, rng, n, donor_entries=hd, return_entry_samples=True)
    hargs = (grid, AGE_GROUPS, logits, ho, hd, *samples, use, hosp)
    hospital = clinical_risk_endpoints(*hargs)
    cfr_entries = [e for e in own if e['quantity'] == 'cfr']
    risk_path = None
    ages, ex = life_expectancy(life, severity_doc_iso3(costs, country))
    if cfr_entries:
        cfr, usec, cs = sample_bin_values(cfr_entries, grid, rng, n, return_entry_samples=True)
        cargs = (grid, AGE_GROUPS, logits, cfr_entries, [], *cs, usec, cfr)
        death = clinical_risk_endpoints(*cargs)
        def life_at_age(b, age):
            le = np.interp(age, ages, ex)
            return le if rate == 0 else -np.expm1(-rate*le)/rate
        yll = clinical_risk_endpoints(*cargs, segment_multiplier=life_at_age)
        infant_risks = None
    else:
        risk_path, risk_scenario = reference_risk_for(country, '2001_2023', AU)
        infant_risks = (risk_file_infant_risks(risk_path, risk_scenario, rng, n, grid)
                        if risk_path is not None else reference_infant_risks(DEATH_RISK_SOURCE[country], rng, n, grid, ROOT))
        death = yll = None
    item_list = costs['countries'][country]['items']
    items = {i['item']: i for i in item_list}
    pm = sample_cost(items.get('vaccine_price_maternal'), rng, n)
    pa = sample_cost(items['vaccine_price_adolescent'], rng, n) if 'vaccine_price_adolescent' in items else pm
    dm, da = [sample_cost(items.get('delivery_'+p), rng, n) for p in ('maternal', 'adolescent')]
    outpatient = sample_cost(items.get('outpatient_episode'), rng, n)
    scoped, fallback = [], None
    if 'inpatient_episode' in items:
        scoped = [dict(age_start_years=i.get('age_start_years', 0), age_end_years=i.get('age_end_years'),
                       draws=sample_cost(i, rng, n)) for i in item_list if i['item'] == 'inpatient_episode']
        low, high = bin_bounds(grid)
        try:
            inpatient, _ = allocate_hospital_cost(low, high, scoped)
            inpatient_basis = 'age_matched_episode_cost'
        except MissingAgeCostError:
            if 'inpatient_day' in items:
                los = [e for e in severity['countries'][LOS_DONOR]['entries'] if e['quantity'] == 'los_days']
                old = sample_entry(max(los, key=lambda e: float(e['age_start_years'])), rng, n)
                fallback = np.repeat((sample_cost(items['inpatient_day'], rng, n)*old)[:, None], grid.n_bins, axis=1)
                inpatient_basis = 'age_matched_episode_plus_borrowed_child_LOS_day_cost'
            elif len(scoped) == 1:
                fallback = np.repeat(scoped[0]['draws'][:, None], grid.n_bins, axis=1)
                inpatient_basis = 'explicit_episode_cost_age_extrapolation'
            else:
                raise ValueError('No declared inpatient fallback')
            inpatient, _ = allocate_hospital_cost(low, high, scoped, fallback=fallback)
    else:
        los = [e for e in severity['countries'][LOS_DONOR]['entries'] if e['quantity'] == 'los_days']
        young = sample_entry(min(los, key=lambda e: float(e['age_start_years'])), rng, n)
        old = sample_entry(max(los, key=lambda e: float(e['age_start_years'])), rng, n)
        _, high = bin_bounds(grid)
        inpatient = sample_cost(items['inpatient_day'], rng, n)[:, None]*np.where((high <= .5)[None, :], young[:, None], old[:, None])
        inpatient_basis = 'day_cost_times_borrowed_child_LOS'
    def inpatient_at_age(b, age):
        matches = [e for e in scoped if e['age_start_years'] <= age < (np.inf if e['age_end_years'] is None else e['age_end_years'])]
        if len(matches) > 1:
            raise ValueError('Overlapping inpatient ages')
        if matches:
            return matches[0]['draws']
        if scoped and fallback is None:
            raise MissingAgeCostError('Uncovered hospital cost age')
        return fallback[:, b] if fallback is not None else inpatient[:, b]
    bounds = [v for e in scoped for v in (e['age_start_years'], e['age_end_years']) if v is not None]
    hcost = clinical_risk_endpoints(*hargs, segment_multiplier=inpatient_at_age, extra_boundaries=bounds)
    overlay = sample_noninfant_risk_endpoints(country, severity, costs, life, grid, AGE_GROUPS, logits,
                                               n_draws=n, seed=20261103, discount_rate=rate)
    return dict(hospital=hospital, hospital_cost=hcost, death=death, yll=yll,
                infant_risks=infant_risks, yll_bin=yll_by_bin(grid, ages, ex, rate),
                price=np.stack([pm, pa], axis=1), delivery=np.stack([dm, da], axis=1),
                outpatient=outpatient, overlay=overlay, risk_path=risk_path, inpatient_basis=inpatient_basis)


def evaluate_arrays(country, arms, raw, doses, starts, ends, info, severity, costs, life, *, n=1000):
    grid, rate = AgeGrid(), .03
    start_year = int(pd.DatetimeIndex(starts)[0].year)
    if raw.shape != (len(arms), len(starts), grid.n_bins, 3) or len(starts) != 360:
        raise ValueError('Expected complete 30-year monthly typical/mild/infection grid')
    if (not np.isfinite(raw).all() or (raw < 0).any() or doses.shape != (len(arms), 2, 30)
            or not np.isfinite(doses).all() or (doses < 0).any()):
        raise ValueError('Invalid outcomes or dose ledger')
    if 'current' not in arms or len(set(arms)) != len(arms):
        raise ValueError('Unique strategies including current required')
    ci = arms.index('current')
    logits, ratio = _reporting(info, ROOT)
    fractions, group_weights = month_geometry(starts, ends, grid)
    reporting = monthly_reporting_weights(grid, logits, starts, ends)
    t, m = raw[..., 0], raw[..., 1]
    reported_t = np.einsum('smb,mbg->smg', t, reporting)
    reported_m = np.einsum('smb,mbg->smg', m*ratio, reporting)
    typical = np.einsum('smb,mbg->smg', t, group_weights)
    mild = np.einsum('smb,mbg->smg', m, group_weights)
    case = reported_t+reported_m
    base = t+ratio*m
    np.testing.assert_allclose(case[..., :-1].sum(axis=2), case[..., -1], rtol=1e-12, atol=1e-7)
    k = sample_kernel(country, severity, costs, life, grid, logits, n=n)
    def episode_unit(name):
        dw = costs['global']['disability_weights'][name]
        return float(dw['value'])*(-np.expm1(-rate*float(dw['duration_days'])/365.25))/rate
    ylds = morbidity_factorial(reported_t, reported_m, typical, mild,
                              episode_unit('pertussis_acute_moderate'), episode_unit('pertussis_acute_mild'))
    summary, age_rows, annual_rows, draws, cache = [], [], [], [], {}
    for horizon in (20, 30):
        mask = np.arange(360) < horizon*12
        disc = mask*(1+rate)**-(np.arange(360)//12)
        h = integrate_endpoints(base, fractions, mask, k['hospital'])
        hdisc = integrate_endpoints(base, fractions, disc, k['hospital'])
        hcost = integrate_endpoints(base, fractions, disc, k['hospital_cost'])
        reported_disc = np.einsum('smg,m->sg', case, disc)
        opcost = (reported_disc[None]-hdisc)*k['outpatient'][:, None, None]
        if np.min(opcost) < -1e-5:
            raise ValueError('More hospitalisations than reported cases')
        if k['infant_risks'] is not None:
            pt, pm = k['infant_risks']
            def infant_outcome(tw, yll=False):
                counts_t = np.einsum('smb,m->sb', t, tw)
                counts_m = np.einsum('smb,m->sb', m, tw)
                factor = k['yll_bin'] if yll else np.ones(grid.n_bins)
                a = np.einsum('sb,nb,b->ns', counts_t, pt, factor)+np.einsum('sb,nb,b->ns', counts_m, pm, factor)
                result = np.zeros((n, len(arms), len(AGE_GROUPS)))
                result[..., 0] = result[..., -1] = a
                return result
            deaths, yll = infant_outcome(mask), infant_outcome(disc, True)
        else:
            deaths = integrate_endpoints(base, fractions, mask, k['death'])
            yll = integrate_endpoints(base, fractions, disc, k['yll'])
        ov = k['overlay']
        extra_yll = integrate_endpoints(base, fractions, disc, (ov['yll_early'], ov['yll_late']))
        extra_deaths = integrate_endpoints(base, fractions, mask, (ov['death_early'], ov['death_late']))
        dd = np.einsum('spy,y->sp', doses[..., :horizon], (1+rate)**-np.arange(horizon))
        vaccine_by_programme = k['price'][:, None, :]*dd[None]
        delivery_by_programme = k['delivery'][:, None, :]*dd[None]
        vaccine = vaccine_by_programme.sum(axis=2)
        delivery = delivery_by_programme.sum(axis=2)
        total_cost = vaccine+delivery+hcost[..., -1]+opcost[..., -1]
        dc = total_cost-total_cost[:, [ci]]
        vaccine_delta = vaccine_by_programme-vaccine_by_programme[:, [ci], :]
        delivery_delta = delivery_by_programme-delivery_by_programme[:, [ci], :]
        cache[f'cost_{horizon}'] = dc
        cache[f'vaccine_{horizon}'] = vaccine_delta
        cache[f'delivery_{horizon}'] = delivery_delta
        cache[f'extra_yll_{horizon}'] = extra_yll[:, [ci], -1]-extra_yll[..., -1]
        cache[f'yll_{horizon}'] = yll[:, [ci], -1]-yll[..., -1]
        for burden, yld_monthly in ylds.items():
            yld = np.einsum('smg,m->sg', yld_monthly, disc)
            d_yld = yld[[ci]]-yld
            cache[f'yld_{horizon}_{burden}'] = d_yld[:, -1]
            for mortality in ('included_base', 'borrowed_noninfant'):
                use_overlay = mortality == 'borrowed_noninfant'
                selected_yll = yll + (extra_yll if use_overlay else 0)
                selected_deaths = deaths + (extra_deaths if use_overlay else 0)
                dalys = selected_yll+yld[None]
                effect = dalys[:, [ci], -1]-dalys[..., -1]
                for s, arm in enumerate(arms):
                    lo, hi = np.quantile(effect[:, s], [.025, .975])
                    tags = dict(country=country, start_year=start_year, horizon=horizon, strategy_id=arm, burden=burden, mortality=mortality)
                    summary.append(dict(**tags, dalys_averted=effect[:, s].mean(), dalys_q025=lo, dalys_q975=hi,
                        incremental_cost=dc[:, s].mean(), vaccine_increment=vaccine_delta[:, s].sum(axis=1).mean(),
                        delivery_increment=delivery_delta[:, s].sum(axis=1).mean(),
                        reference_strategy='current', n_draws=n, uncertainty='clinical_cost_conditional_fixed_transmission',
                        noninfant_mortality_estimated=(country not in MISSING_NONINFANT),
                        noninfant_mortality_borrowed=(country in MISSING_NONINFANT and use_overlay)))
                    if burden == 'reported_uniform':
                        draws.append(pd.DataFrame(dict(**tags, draw=np.arange(n), dalys_averted=effect[:, s], incremental_cost=dc[:, s])))
                    for g, group in enumerate(AGE_GROUPS):
                        unknown = country in MISSING_NONINFANT and not use_overlay and group not in ('infant_0', 'all_ages')
                        delta_yll = selected_yll[:, ci, g]-selected_yll[:, s, g]
                        delta_death = selected_deaths[:, ci, g]-selected_deaths[:, s, g]
                        age_rows.append(dict(**tags, age_group=group,
                            reported_cases_averted=(case[ci, mask, g]-case[s, mask, g]).sum(),
                            hospitalisations_averted=(h[:, ci, g]-h[:, s, g]).mean(),
                            deaths_averted=np.nan if unknown else delta_death.mean(),
                            yll_averted=np.nan if unknown else delta_yll.mean(), yld_averted=d_yld[s, g],
                            dalys_averted=np.nan if unknown else delta_yll.mean()+d_yld[s, g]))
    for s, arm in enumerate(arms):
        for y in range(30):
            for g, group in enumerate(AGE_GROUPS):
                sl = slice(y*12, (y+1)*12)
                annual_rows.append(dict(country=country, strategy_id=arm, year=start_year+y, age_group=group,
                    reported_cases=case[s, sl, g].sum(), reported_cases_averted=(case[ci, sl, g]-case[s, sl, g]).sum(),
                    symptomatic_cases=(typical[s, sl, g]+mild[s, sl, g]).sum(),
                    symptomatic_cases_averted=(typical[ci, sl, g]+mild[ci, sl, g]-typical[s, sl, g]-mild[s, sl, g]).sum()))
    meta = dict(country=country, start_year=start_year, risk_path=str(k['risk_path']), inpatient_basis=k['inpatient_basis'],
                hospital_scope='Sweden_historical_survivor_under_6m_else_explicit_country_or_Brazil_donor',
                fixed_transmission=True, reference='current', morbidity_factorial=BURDENS,
                mortality_overlay=k['overlay']['status'], rate=rate)
    return pd.DataFrame(summary), pd.DataFrame(age_rows), pd.DataFrame(annual_rows), pd.concat(draws, ignore_index=True), cache, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projection-root', type=Path, default=BASE/'current_preserving_strategies_20261103')
    p.add_argument('--output-dir', type=Path, default=BASE/'current_preserving_economics_20261103')
    p.add_argument('--countries', nargs='+', default=['Australia', 'England', 'Sweden', 'Japan', 'New_Zealand', 'China', 'Brazil', 'Thailand'])
    p.add_argument('--n-draws', type=int, default=1000)
    p.add_argument('--severity', type=Path, default=SEV)
    p.add_argument('--costs', type=Path, default=COST)
    args = p.parse_args()
    args.projection_root = args.projection_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.costs, args.severity = args.costs.resolve(), args.severity.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    costs, severity = load_costs(args.costs), load_severity_priors(args.severity)
    life_path = ROOT/costs['global']['life_table']['csv']
    life = pd.read_csv(life_path)
    sources = [Path(__file__).resolve(), args.costs.resolve(), args.severity.resolve(), life_path,
               ROOT/'scripts/revise_clinical_inputs_20261103.py',
               ROOT/'src_python/pertussis/economics/monthly_age_severity.py',
               ROOT/'src_python/pertussis/scenarios/monthly_reporting.py']
    tables = [[], [], []]
    metas = []
    for country in args.countries:
        folder = args.projection_root/country
        paths = list(folder.glob('*.npz'))
        if len(paths) != 1:
            raise ValueError(f'Expected one monthly NPZ in {folder}')
        info = json.loads((folder/'complete.json').read_text())
        z = np.load(paths[0], allow_pickle=False)
        arms = z['strategy_ids'].astype(str).tolist()
        raw, doses = z['onsets'], z['annual_doses']
        result = evaluate_arrays(country, arms, raw, doses, z['month_starts'], z['month_ends'], info, severity, costs, life, n=args.n_draws)
        summary, ages, annual, draws, cache, meta = result
        for collect, table in zip(tables, (summary, ages, annual)):
            collect.append(table)
        draws.to_csv(args.output_dir/f'{country}_paired.csv.gz', index=False)
        np.savez_compressed(args.output_dir/f'{country}_components.npz', strategy_ids=np.asarray(arms), **cache)
        metas.append(meta)
        sources.extend([paths[0], folder/'complete.json'])
        risk_path = meta['risk_path']
        if risk_path != 'None':
            sources.append(Path(risk_path))
        print(country, len(arms), 'strategies valued', flush=True)
    for name, frames in zip(('summary', 'age_effects', 'annual_cases'), tables):
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir/f'{name}.csv', index=False)
    (args.output_dir/'metadata.json').write_text(json.dumps(dict(
        status='conditional_current_preserving_monthly_valuation', n_draws=args.n_draws,
        decision_set='independent_coverage_grid_no_reduction_of_encoded_programmes',
        countries=metas, currency='2024_international_dollars',
        source_files=[str(q.relative_to(ROOT)) for q in set(sources)],
        output_files=['summary.csv', 'age_effects.csv', 'annual_cases.csv',
                      *[f"{row['country']}_paired.csv.gz" for row in metas]],
        limitations=['fixed candidate transmission parameters', 'historical Swedish survivor-cohort hospital risk',
                     'unvalidated donor transfer in noninfant mortality sensitivity', 'cost assumptions and age extrapolations']), indent=2)+'\n')


if __name__ == '__main__':
    main()
