"""Eight-country cost-effectiveness evaluation on fitted projections.

One paired probabilistic sensitivity analysis per country, comparing
``baseline``, ``maternal_scaleup`` and ``adolescent_one_time_booster`` under the
same annual additional-dose budget. All money is 2024 international dollars;
costs and health effects are discounted at the same annual rate from the first
prospective year.

Clinical cascade
----------------
* Confirmed cases = (typical + r * mild) * reporting probability, using each
  country's own fitted reporting logits and breakthrough ratio r.
* Hospitalisations = confirmed cases * per-confirmed-case hospitalisation
  proportion. A country's own entries are used where they cover an age bin;
  other bins use the donor (Brazil SINAN, the only age-complete national
  series). The donor is declared in every output.
* Deaths use one of two declared pathways. Countries with national
  per-confirmed-case fatality data (Brazil, Thailand) apply it to confirmed
  cases at all ages. The other countries apply calibrated per-onset infant risks
  (Australia or England, at that source's own infant boundary) to modelled
  typical and mild infant onsets; no deaths are attributed above age 1 in that
  pathway because the reference calibration covers infants only.
* Treatment costs are counted for confirmed cases only (health-system
  perspective, conservative): hospitalised cases at the inpatient episode cost
  (or bed-day cost times length of stay), the rest at the outpatient cost.
* DALYs = discounted YLL (WPP 2024 country life table, continuous annuity at
  the discount rate from the age of death) + YLD for typical onsets.

Transmission inputs may be fitted points or validated local posterior tasks.
The uncertainty scope and fit status are retained in every output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import REPO_ROOT
from ..grid import AgeGrid, band_matrix
from ..scenarios.mortality_extrapolation import load_reference_risks
from .psa import _sample_gamma, _sample_lognormal
from .projections import read_trajectory, load_posterior, check_hashes, projection_fit_is_converged

BAND_BOUNDS = [(0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 60.0), (60.0, np.inf)]
LOGIT_COLUMNS = ['reporting_logit_<1', 'reporting_logit_1-4', 'reporting_logit_5-9',
                 'reporting_logit_10-19', 'reporting_logit_20-59', 'reporting_logit_60+']
STRATEGIES = ('baseline', 'maternal_scaleup', 'adolescent_one_time_booster')
COUNTRIES = ('Australia', 'England', 'Sweden', 'Japan', 'New_Zealand', 'China', 'Brazil', 'Thailand')

_HO = 'outputs/core2026/health_outcomes_20261026'
_CH = 'outputs/core2026/country_health_calibration_20261027'
PROJECTIONS = {
    'Australia': (f'{_HO}/fitted_cycle2025_australia_main', f'{_HO}/polish_cycle2025_australia_main'),
    'England': (f'{_HO}/fitted_cycle2025_england_main', f'{_HO}/polish_cycle2025_england_main'),
    **{c: (f'{_CH}/{c}/projection', f'{_CH}/{c}/fit')
       for c in ('Sweden', 'Japan', 'New_Zealand', 'China', 'Brazil', 'Thailand')},
    # Refinements retain the original model/training split and pass independent
    # finite-difference stationarity checks. Older diagnostic runs stay on disk.
    **{c: (f'outputs/core2026/model_resolution_20261029/{c}/projection',
           f'outputs/core2026/model_resolution_20261029/{c}/preconditioned')
       for c in ('Japan', 'New_Zealand')},
}
# Per-onset infant death risk source for countries without national
# per-confirmed-case fatality data (borrowing rules of severity_priors.yaml).
DEATH_RISK_SOURCE = {'Australia': 'Australia', 'England': 'England', 'China': 'Australia',
                     'Japan': 'England', 'New_Zealand': 'England', 'Sweden': 'England'}
MAIN_RISK_SCENARIO = {'Australia': 'ACR_reference', 'England': 'reference'}
HOSPITAL_DONOR = 'Brazil'
LOS_DONOR = 'China'


def _upper(value):
    return np.inf if value is None or str(value).strip().lower() in {'inf', 'null', 'none', ''} else float(value)


def bin_bounds(grid):
    """Model bin age bounds in years; the open bin is capped at 100."""
    lo = np.asarray(grid.lower_years, float)
    hi = np.where(np.isinf(grid.upper_years), 100.0, np.asarray(grid.upper_years, float))
    return lo, hi


def entry_overlap(entries, grid):
    """(n_bins, n_entries) share of each bin's age span covered by each entry."""
    lo, hi = bin_bounds(grid)
    out = np.zeros((grid.n_bins, len(entries)))
    for j, e in enumerate(entries):
        a, b = float(e['age_start_years']), min(_upper(e['age_end_years']), 100.0)
        out[:, j] = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None) / (hi - lo)
    return out


def sample_entry(entry, rng, n):
    dist, params = entry['distribution'], entry.get('params') or {}
    if dist == 'beta':
        return rng.beta(float(params.get('alpha', params.get('a'))),
                        float(params.get('beta', params.get('b'))), size=n)
    if dist == 'lognormal':
        return _sample_lognormal(params, rng, n)
    return np.full(n, float(entry['point']))


def sample_bin_values(entries, grid, rng, n, donor_entries=None, *, return_entry_samples=False):
    """(n, n_bins) draws; bins not mostly covered by own entries use the donor."""
    def draws(es):
        w = entry_overlap(es, grid)
        cover = w.sum(axis=1)
        vals = np.column_stack([sample_entry(e, rng, n) for e in es]) if es else np.zeros((n, 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            out = (vals @ w.T) / np.where(cover > 0, cover, 1.0)
        return out, cover, vals
    own, cover, own_samples = draws(entries) if entries else (np.zeros((n, grid.n_bins)), np.zeros(grid.n_bins), np.zeros((n, 0)))
    use_own = cover >= 0.5
    if donor_entries is None:
        result = (own, use_own)
        details = (own_samples, np.zeros((n, 0)))
    else:
        donor, _, donor_samples = draws(donor_entries)
        result = (np.where(use_own[None, :], own, donor), use_own)
        details = (own_samples, donor_samples)
    return (*result, details) if return_entry_samples else result


def sample_cost(item, rng, n):
    if item is None:
        return np.zeros(n)
    dist, params = item['distribution'], dict(item.get('params') or {})
    params.pop('unit', None)
    if dist == 'gamma' and params.get('sd', 0) > 0:
        return _sample_gamma(params, rng, n)
    if dist == 'lognormal' and params.get('sd', 0) > 0:
        return _sample_lognormal(params, rng, n)
    return np.full(n, float(item['value_intl_2024']))


def life_expectancy(table, iso3):
    t = table[table.ISO3_code.eq(iso3)].sort_values('AgeGrpStart')
    if t.empty:
        raise ValueError(f'no life table for {iso3}')
    return t.AgeGrpStart.to_numpy(float), t.ex.to_numpy(float)


def yll_by_bin(grid, ages, ex, rate):
    """Discounted YLL per death at each bin midpoint (continuous annuity)."""
    lo, hi = bin_bounds(grid)
    le = np.interp((lo + hi) / 2, ages, ex)
    return le if rate == 0 else (1 - np.exp(-rate * le)) / rate


def load_projection(country, repo_root=REPO_ROOT, start_year=2027, horizon=20, point_inputs=None):
    paths = (point_inputs['projection_dir'], point_inputs['fit_dir']) if point_inputs else PROJECTIONS[country]
    proj, fit = (Path(repo_root) / p for p in paths)
    years = np.arange(start_year, start_year + horizon)
    trajectory = read_trajectory(proj, years)
    completion = json.loads((proj / 'complete.json').read_text())
    fit_spec = json.loads((fit / 'run_spec.json').read_text())
    check_hashes(fit_spec['source_hashes'], repo_root)
    table = pd.read_csv(fit / 'fit_summary.csv')
    # Reporting must follow the trajectory's selected fit, not another row.
    matches = table[table.start_index.eq(completion['draw_id'])]
    if len(matches) != 1:
        raise ValueError('projection fit row is missing or ambiguous')
    summary = matches.iloc[0]
    return dict(**trajectory, years=years, logits=summary[LOGIT_COLUMNS].to_numpy(float),
                mild_ratio=float(summary['breakthrough_reporting_ratio']),
                projection_dir=str(proj), fit_dir=str(fit),
                fit_converged=projection_fit_is_converged(completion, summary),
                analysis_status=completion['inference_status'],
                model_structure=completion.get('structure', 'legacy_29_states'),
                source_fit_converged=bool(completion['fit_success']),
                transmission_draw_id=int(completion['draw_id']), source_parameter_row=int(completion['draw_id']),
                transmission_uncertainty='none_fitted_point', available_transmission_draws=1)


def reference_infant_risks(source, rng, n, grid, repo_root=REPO_ROOT):
    """(n, n_bins) per-onset death risks for typical and mild onsets, infants only."""
    risks, _ = load_reference_risks(source, repo_root=repo_root)
    risks = risks[risks.risk_scenario.eq(MAIN_RISK_SCENARIO[source])]
    ids = rng.choice(np.sort(risks.risk_draw_id.unique()), size=n, replace=True)
    out = {}
    for col in ('p_death_given_typical', 'p_death_given_mild'):
        table = risks.pivot(index='risk_draw_id', columns='risk_age', values=col)
        arr = np.zeros((n, grid.n_bins))
        bounds = risks[['risk_age', 'age_start_months', 'age_end_months_exclusive']].drop_duplicates()
        for _, row in bounds.iterrows():
            months = np.arange(int(row.age_start_months), int(row.age_end_months_exclusive))
            arr[:, months] = table.loc[ids, row.risk_age].to_numpy()[:, None]
        out[col] = arr
    return out['p_death_given_typical'], out['p_death_given_mild']


def evaluate_country(country, severity_doc, costs_doc, life_table, *, n_draws=1000, seed=20261029,
                     discount_rate=0.03, start_year=2027, horizon=20, repo_root=REPO_ROOT,
                     posterior_inputs=None, point_inputs=None):
    if n_draws < 1 or horizon < 1 or not np.isfinite(discount_rate) or discount_rate < 0:
        raise ValueError('positive draws/horizon and a finite nonnegative discount rate required')
    grid = AgeGrid()
    rng = np.random.default_rng(seed)
    if posterior_inputs is not None and point_inputs is not None:
        raise ValueError('choose point or posterior transmission inputs, not both')
    if posterior_inputs is None:
        p = (load_projection(country, repo_root, start_year, horizon) if point_inputs is None else
             load_projection(country, repo_root, start_year, horizon, point_inputs=point_inputs))
        for key in ('typical', 'mild', 'doses', 'logits', 'mild_ratio',
                    'transmission_draw_id', 'source_parameter_row'):
            value = np.asarray(p[key])
            p[key] = np.broadcast_to(value, (n_draws,) + value.shape)
    else:
        p = load_posterior(country, posterior_inputs, np.arange(start_year, start_year+horizon),
                           n_draws, rng, repo_root)
    disc = (1 + discount_rate) ** -(p['years'] - start_year)
    report = expit(p['logits']) @ band_matrix(grid, BAND_BOUNDS, 0.5).T
    confirmed = (p['typical'] + p['mild_ratio'][:, None, None, None] * p['mild']) * report[:, None, None, :]
    conf_d = np.einsum('nsyb,y->nsb', confirmed, disc)
    typ_d = np.einsum('nsyb,y->nsb', p['typical'], disc)
    mild_d = np.einsum('nsyb,y->nsb', p['mild'], disc)
    doses_d = p['doses'] @ disc

    sev = severity_doc['countries'][country]
    donor = severity_doc['countries'][HOSPITAL_DONOR]['entries']
    own = sev.get('entries') or []
    hosp_own = [e for e in own if e['quantity'] == 'hospitalisation_proportion']
    hosp, hosp_own_bins = sample_bin_values(
        hosp_own, grid, rng, n_draws,
        donor_entries=[e for e in donor if e['quantity'] == 'hospitalisation_proportion'])

    # Deaths, discounted (for YLL) and undiscounted (for counts).
    cfr_own = [e for e in own if e['quantity'] == 'cfr']
    if cfr_own:
        cfr, _ = sample_bin_values(cfr_own, grid, rng, n_draws)
        deaths_d = conf_d * cfr[:, None, :]
        deaths_u = confirmed.sum(axis=2) * cfr[:, None, :]
        death_pathway = 'national_per_confirmed_case_cfr'
    else:
        source = DEATH_RISK_SOURCE[country]
        pt, pm = ((p['risk_typical'], p['risk_mild']) if 'risk_typical' in p else
                  reference_infant_risks(source, rng, n_draws, grid, repo_root))
        deaths_d = typ_d * pt[:, None, :] + mild_d * pm[:, None, :]
        deaths_u = (p['typical'].sum(axis=2) * pt[:, None, :]
                    + p['mild'].sum(axis=2) * pm[:, None, :])
        death_pathway = f'{source}_calibrated_per_onset_infant_risk'

    items = {i['item']: i for i in costs_doc['countries'][country]['items']}
    price = {'maternal_scaleup': sample_cost(items.get('vaccine_price_maternal'), rng, n_draws)}
    price['adolescent_one_time_booster'] = (
        sample_cost(items['vaccine_price_adolescent'], rng, n_draws)
        if 'vaccine_price_adolescent' in items else price['maternal_scaleup'])
    delivery = {'maternal_scaleup': sample_cost(items.get('delivery_maternal'), rng, n_draws),
                'adolescent_one_time_booster': sample_cost(items.get('delivery_adolescent'), rng, n_draws)}
    outpatient = sample_cost(items.get('outpatient_episode'), rng, n_draws)
    if 'inpatient_episode' in items:
        inpatient = np.repeat(sample_cost(items['inpatient_episode'], rng, n_draws)[:, None], grid.n_bins, 1)
        inpatient_basis = 'episode_cost'
    else:
        los_entries = [e for e in severity_doc['countries'][LOS_DONOR]['entries'] if e['quantity'] == 'los_days']
        young = sample_entry(min(los_entries, key=lambda e: float(e['age_start_years'])), rng, n_draws)
        old = sample_entry(max(los_entries, key=lambda e: float(e['age_start_years'])), rng, n_draws)
        lo, hi = bin_bounds(grid)
        los = np.where((hi <= 0.5)[None, :], young[:, None], old[:, None])
        inpatient = sample_cost(items['inpatient_day'], rng, n_draws)[:, None] * los
        inpatient_basis = f'bed_day_cost_x_{LOS_DONOR}_length_of_stay'

    ages, ex = life_expectancy(life_table, severity_doc_iso3(costs_doc, country))
    yll = yll_by_bin(grid, ages, ex, discount_rate)
    dw = costs_doc['global']['disability_weights']['pertussis_acute_moderate']
    episode = float(dw['duration_days']) / 365.25
    yld_unit = float(dw['value']) * ((1 - np.exp(-discount_rate * episode)) / discount_rate
                                     if discount_rate else episode)

    rows = []
    for s, sid in enumerate(STRATEGIES):
        hosp_cost = (conf_d[:, s] * hosp * inpatient).sum(axis=1)
        op_cost = (conf_d[:, s] * (1 - hosp) * outpatient[:, None]).sum(axis=1)
        if sid == 'baseline':
            vacc, deliv_cost = np.zeros(n_draws), np.zeros(n_draws)
        else:
            vacc, deliv_cost = doses_d[:, s] * price[sid], doses_d[:, s] * delivery[sid]
        yll_total = (deaths_d[:, s] * yll).sum(axis=1)
        yld_total = typ_d[:, s].sum(axis=1) * yld_unit
        infant = slice(0, 12)
        rows.append(pd.DataFrame(dict(
            country=country, draw=np.arange(n_draws), strategy_id=sid,
            vaccine_cost=vacc, delivery_cost=deliv_cost, hospital_cost=hosp_cost,
            outpatient_cost=op_cost, total_cost=vacc + deliv_cost + hosp_cost + op_cost,
            yll=yll_total, yld=yld_total, dalys=yll_total + yld_total,
            typical_onsets=p['typical'][:, s].sum(axis=(1, 2)),
            infant_typical_onsets=p['typical'][:, s, :, infant].sum(axis=(1, 2)),
            confirmed_cases=confirmed[:, s].sum(axis=(1, 2)),
            hospitalisations=(confirmed[:, s].sum(axis=1) * hosp).sum(axis=1),
            deaths=deaths_u[:, s].sum(axis=1), infant_deaths=deaths_u[:, s, infant].sum(axis=1),
            discounted_doses=doses_d[:, s], doses=p['doses'][:, s].sum(axis=1),
            transmission_draw_id=p['transmission_draw_id'], source_parameter_row=p['source_parameter_row'],
            conditional_risk_draw_id=p.get('risk_draw_id', np.full(n_draws, -1)),
            fit_converged=p['fit_converged'], transmission_uncertainty=p['transmission_uncertainty'],
            model_structure=p.get('model_structure', 'legacy_29_states'),
            analysis_status=p.get('analysis_status', p['transmission_uncertainty']),
            unit_price=price.get(sid, np.zeros(n_draws)),
            unit_delivery=delivery.get(sid, np.zeros(n_draws)))))
    draws = pd.concat(rows, ignore_index=True)
    meta = dict(country=country, death_pathway=death_pathway, inpatient_basis=inpatient_basis,
                hospital_donor=HOSPITAL_DONOR,
                icu_costing='no_separate_surcharge; ICU may be embedded in sourced inpatient episode costs',
                hospital_own_bins=int(hosp_own_bins.sum()), n_bins=grid.n_bins,
                reporting_logits_mean=p['logits'].mean(axis=0).tolist(), mild_ratio_mean=float(p['mild_ratio'].mean()),
                fit_converged=p['fit_converged'], transmission_uncertainty=p['transmission_uncertainty'],
                model_structure=p.get('model_structure', 'legacy_29_states'),
                analysis_status=p.get('analysis_status', p['transmission_uncertainty']),
                available_transmission_draws=p['available_transmission_draws'],
                sampled_transmission_draws=int(np.unique(p['transmission_draw_id']).size),
                mortality_uncertainty=('conditional_on_matched_transmission_draw' if 'risk_typical' in p
                    else 'national_cfr' if cfr_own else 'reference_risks_conditional_on_donor_point'),
                projection_dir=p['projection_dir'], fit_dir=p['fit_dir'],
                n_draws=n_draws, seed=seed, discount_rate=discount_rate,
                years=[int(p['years'][0]), int(p['years'][-1])],
                thresholds=thresholds(costs_doc, country))
    return draws, meta


def severity_doc_iso3(costs_doc, country):
    return costs_doc['global']['life_table']['country_iso3'][country]


HIGH_INCOME = ('Australia', 'England', 'Sweden', 'Japan', 'New_Zealand')


def imputed_opportunity_cost_ratio(costs_doc, country):
    """Mean opportunity-cost/GDP-per-capita ratio of sourced countries in the
    same income group, used where a country has no sourced opportunity-cost
    threshold. A declared imputation rule, not a national estimate."""
    group = HIGH_INCOME if country in HIGH_INCOME else tuple(
        c for c in costs_doc['countries'] if c not in HIGH_INCOME)
    ratios = []
    for c in group:
        macro = costs_doc['countries'][c]['macro']
        for t in macro['thresholds']:
            if t['type'] == 'opportunity_cost':
                ratios.append(float(t['value_intl_2024']) / float(macro['gdp_per_capita_intl_2024']))
    if not ratios:
        raise ValueError(f'no sourced opportunity-cost threshold in the income group of {country}')
    return float(np.mean(ratios))


def thresholds(costs_doc, country, impute_opportunity_cost=True):
    out = {t['type']: float(t['value_intl_2024'])
           for t in costs_doc['countries'][country]['macro']['thresholds']}
    if impute_opportunity_cost and 'opportunity_cost' not in out:
        gdp = float(costs_doc['countries'][country]['macro']['gdp_per_capita_intl_2024'])
        out['opportunity_cost'] = imputed_opportunity_cost_ratio(costs_doc, country) * gdp
    return out


def imputed_threshold_types(costs_doc, country):
    sourced = {t['type'] for t in costs_doc['countries'][country]['macro']['thresholds']}
    return [] if 'opportunity_cost' in sourced else ['opportunity_cost']


def incremental(draws, baseline='baseline'):
    """Each strategy against the baseline within draw; paired by construction."""
    base = draws[draws.strategy_id.eq(baseline)].set_index(['country', 'draw'])
    out = draws[~draws.strategy_id.eq(baseline)].copy()
    key = pd.MultiIndex.from_frame(out[['country', 'draw']])
    for col in ('total_cost', 'dalys', 'hospital_cost', 'outpatient_cost', 'typical_onsets',
                'infant_typical_onsets', 'hospitalisations', 'deaths', 'infant_deaths'):
        out[f'delta_{col}'] = out[col].to_numpy() - base.loc[key, col].to_numpy()
    out['dalys_averted'] = -out.delta_dalys
    out['incremental_cost'] = out.delta_total_cost
    out['treatment_change'] = out.delta_hospital_cost + out.delta_outpatient_cost
    return out


def summarise(draws, meta):
    inc = incremental(draws)
    q = lambda x: (float(np.mean(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    rows = []
    for sid, g in inc.groupby('strategy_id', sort=False):
        row = dict(country=meta['country'], strategy_id=sid, doses=float(g.doses.mean()),
                   fit_converged=meta.get('fit_converged'),
                   model_structure=meta.get('model_structure', 'legacy_29_states'),
                   analysis_status=meta.get('analysis_status'),
                   transmission_uncertainty=meta.get('transmission_uncertainty'),
                   sampled_transmission_draws=meta.get('sampled_transmission_draws'))
        for col, name in [('incremental_cost', 'incremental_cost'), ('dalys_averted', 'dalys_averted'),
                          ('delta_typical_onsets', 'typical_onsets_change'),
                          ('delta_infant_typical_onsets', 'infant_typical_onsets_change'),
                          ('delta_hospitalisations', 'hospitalisations_change'),
                          ('delta_infant_deaths', 'infant_deaths_change'),
                          ('delta_deaths', 'deaths_change'), ('treatment_change', 'treatment_cost_change')]:
            row[name], row[f'{name}_lo'], row[f'{name}_hi'] = q(g[col])
        row['icer_ratio_of_means'] = (row['incremental_cost'] / row['dalys_averted']
                                      if row['dalys_averted'] > 0 else np.nan)
        row['dominance'] = ('dominated' if row['dalys_averted'] < 0 and row['incremental_cost'] > 0
                            else 'dominant' if row['dalys_averted'] > 0 and row['incremental_cost'] < 0
                            else 'tradeoff_or_zero_effect')
        for kind, lam in meta['thresholds'].items():
            nmb = lam * g.dalys_averted - g.incremental_cost
            row[f'p_cost_effective_{kind}'] = float((nmb > 0).mean())
            breakeven = (lam * g.dalys_averted - g.treatment_change
                         - g.discounted_doses * g.unit_delivery) / g.discounted_doses.replace(0, np.nan)
            row[f'breakeven_price_{kind}'], row[f'breakeven_price_{kind}_lo'], \
                row[f'breakeven_price_{kind}_hi'] = (float(breakeven.median()),
                                                     float(breakeven.quantile(.025)),
                                                     float(breakeven.quantile(.975)))
        row['mean_unit_price'] = float(g.unit_price.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def acceptability(draws, meta, multiples=np.linspace(0, 3, 61)):
    """Probability each strategy has the highest NMB, over thresholds expressed
    as multiples of GDP per capita (Int$) and at each declared threshold."""
    gdp = meta['thresholds']['gdp_multiple']
    lams = [(f'{m:.2f}xGDP', m * gdp) for m in multiples] + list(meta['thresholds'].items())
    wide_cost = draws.pivot(index='draw', columns='strategy_id', values='total_cost')
    wide_daly = draws.pivot(index='draw', columns='strategy_id', values='dalys')
    rows = []
    for label, lam in lams:
        nmb = -lam * wide_daly - wide_cost
        best = nmb.idxmax(axis=1)
        for sid in STRATEGIES:
            rows.append(dict(country=meta['country'], threshold_label=label, threshold=lam,
                             strategy_id=sid, p_optimal=float((best == sid).mean())))
    return pd.DataFrame(rows)
