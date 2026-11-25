"""Cost-effectiveness of programme-versus-no-programme scenarios.

Reuses the clinical cascade, costing and DALY helpers of ``evaluate.py`` but
reads the arm labels written by ``scenarios/programme_scenarios.py``
(``baseline`` plus ``<programme>_cNN``).  ``baseline`` is the no-programme
comparator for that programme; every arm is compared with it within draw.
Doses are the vaccinated persons implied by each arm (``additional_receipts``).
Transmission is a fitted point; PSA covers cost, severity and borrowed-risk
uncertainty only, and outputs say so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import REPO_ROOT
from ..grid import AgeGrid, band_matrix
from .evaluate import (BAND_BOUNDS, DEATH_RISK_SOURCE, HOSPITAL_DONOR, LOGIT_COLUMNS, LOS_DONOR,
                       bin_bounds, life_expectancy, reference_infant_risks, sample_bin_values,
                       sample_cost, sample_entry, severity_doc_iso3, thresholds, yll_by_bin)


def risk_file_infant_risks(path, scenario, rng, n, grid):
    """(n, n_bins) per-onset infant risks from a recalibrated risk_draws file.

    Draw identifiers are subsampled jointly so the two infant age groups of one
    draw stay together; bounds come from the file's own monthly columns.
    """
    risks = pd.read_csv(path)
    risks = risks[risks.risk_scenario.eq(scenario)]
    if risks.empty:
        raise ValueError(f'risk scenario {scenario!r} missing from {path}')
    ids = rng.choice(np.sort(risks.risk_draw_id.unique()), size=n, replace=True)
    bounds = risks[['risk_age', 'age_start_months', 'age_end_months_exclusive']].drop_duplicates()
    out = []
    for col in ('p_death_given_typical', 'p_death_given_mild'):
        table = risks.pivot(index='risk_draw_id', columns='risk_age', values=col)
        arr = np.zeros((n, grid.n_bins))
        for _, row in bounds.iterrows():
            months = np.arange(int(row.age_start_months), int(row.age_end_months_exclusive))
            arr[:, months] = table.loc[ids, row.risk_age].to_numpy()[:, None]
        out.append(arr)
    return out[0], out[1]


def read_programme_projection(folder, years):
    """Arms, onsets (arm, year, bin) and doses (arm, year) from one scenario run."""
    folder = Path(folder)
    onsets = pd.read_csv(folder / 'annual_age_onsets.csv.gz')
    onsets = onsets[onsets.year.isin(years)]
    arms = ['baseline'] + sorted(a for a in onsets.strategy_id.unique() if a != 'baseline')
    n_bins = AgeGrid().n_bins
    index = pd.MultiIndex.from_product([arms, years, range(n_bins)],
                                      names=['strategy_id', 'year', 'bin_index'])
    if onsets.duplicated(index.names).any():
        raise ValueError('duplicate onset cells')
    table = onsets.set_index(index.names).reindex(index)
    values = table[['typical_onsets', 'mild_onsets']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('incomplete or invalid arm/year/age onset grid')
    receipts = pd.read_csv(folder / 'receipts.csv.gz')
    doses = np.zeros((len(arms), len(years)))
    for s, arm in enumerate(arms[1:], start=1):
        v = receipts[receipts.strategy_id.eq(arm)].set_index('year').reindex(years).additional_receipts
        if v.isna().any() or (v < 0).any():
            raise ValueError(f'incomplete receipts for {arm}')
        doses[s] = v.to_numpy(float)
    shape = (len(arms), len(years), n_bins)
    completion = json.loads((folder / 'complete.json').read_text())
    return arms, values[:, 0].reshape(shape), values[:, 1].reshape(shape), doses, completion


def _programme_of(arm):
    return arm.split('_c')[0]


def evaluate_programme(country, scenario_dir, severity_doc, costs_doc, life_table, *,
                       n_draws=1000, seed=20261029, discount_rate=0.03, start_year=2027,
                       horizon=20, repo_root=REPO_ROOT, risk_file=None, risk_scenario='reference',
                       yld_scope='confirmed', infant_survivor_qaly=0.097, bereavement_qaly=3.70,
                       counterfactual_infant_deaths_per_year=None):
    """``yld_scope``: 'confirmed' (main; acute YLD on surveillance-confirmed
    cases only, whose count is anchored to data) or 'all_typical' (sensitivity;
    every modelled typical onset, whose adult volume is not identified).
    Hospitalised surviving infants add ``infant_survivor_qaly`` and each death
    adds ``bereavement_qaly`` (Sandmann et al. 2021); set either to 0 to drop."""
    if yld_scope not in ('confirmed', 'all_typical'):
        raise ValueError('yld_scope must be confirmed or all_typical')
    grid = AgeGrid()
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + horizon)
    arms, typical, mild, doses, completion = read_programme_projection(scenario_dir, years)
    if 'reporting' in completion:
        # Posterior draws carry the reporting row that produced their onsets.
        rep = completion['reporting']
        logits = np.array([rep[c] for c in LOGIT_COLUMNS], float)
        mild_ratio = float(rep['breakthrough_reporting_ratio'])
    else:
        fit = Path(repo_root) / completion['fit_directory']
        summary = pd.read_csv(fit / 'fit_summary.csv')
        row = summary[summary.start_index.eq(completion['draw_id'])]
        if len(row) != 1:
            raise ValueError('scenario fit row is missing or ambiguous')
        logits = row.iloc[0][LOGIT_COLUMNS].to_numpy(float)
        mild_ratio = float(row.iloc[0]['breakthrough_reporting_ratio'])

    disc = (1 + discount_rate) ** -(years - start_year)
    report = band_matrix(grid, BAND_BOUNDS, 0.5) @ expit(logits)              # (bins,)
    confirmed = (typical + mild_ratio * mild) * report[None, None, :]          # (arm, yr, bin)
    conf_d = np.einsum('syb,y->sb', confirmed, disc)
    typ_d = np.einsum('syb,y->sb', typical, disc)
    mild_d = np.einsum('syb,y->sb', mild, disc)
    doses_d = doses @ disc

    sev = severity_doc['countries'][country]
    own = sev.get('entries') or []
    donor = severity_doc['countries'][HOSPITAL_DONOR]['entries']
    hosp, _ = sample_bin_values(
        [e for e in own if e['quantity'] == 'hospitalisation_proportion'], grid, rng, n_draws,
        donor_entries=[e for e in donor if e['quantity'] == 'hospitalisation_proportion'])
    cfr_own = [e for e in own if e['quantity'] == 'cfr']
    if cfr_own:
        cfr, _ = sample_bin_values(cfr_own, grid, rng, n_draws)
        deaths_d = conf_d[None] * cfr[:, None, :]                              # (n, arm, bin)
        deaths_u = confirmed.sum(axis=1)[None] * cfr[:, None, :]
        death_pathway = 'national_per_confirmed_case_cfr'
    else:
        source = DEATH_RISK_SOURCE[country]
        pt, pm = (risk_file_infant_risks(risk_file, risk_scenario, rng, n_draws, grid)
                  if risk_file is not None else
                  reference_infant_risks(source, rng, n_draws, grid, repo_root))
        deaths_d = typ_d[None] * pt[:, None, :] + mild_d[None] * pm[:, None, :]
        deaths_u = typical.sum(axis=1)[None] * pt[:, None, :] + mild.sum(axis=1)[None] * pm[:, None, :]
        death_pathway = f'{source}_calibrated_per_onset_infant_risk'

    # Counterfactual intensity: scale every arm's deaths by one common factor so
    # that the no-programme comparator has the declared mean annual infant
    # deaths (e.g. an epidemic-year level).  Scaling is shared across arms, so
    # relative effects are unchanged; only the absolute burden moves.
    death_scale = 1.0
    if counterfactual_infant_deaths_per_year is not None:
        base_infant = deaths_u[:, 0, :12].sum(axis=1).mean() / len(years)
        if base_infant <= 0:
            raise ValueError('comparator has no infant deaths to rescale')
        death_scale = float(counterfactual_infant_deaths_per_year) / base_infant
        deaths_d = deaths_d * death_scale
        deaths_u = deaths_u * death_scale

    items = {i['item']: i for i in costs_doc['countries'][country]['items']}
    price = {'maternal': sample_cost(items.get('vaccine_price_maternal'), rng, n_draws)}
    price['adolescent'] = (sample_cost(items['vaccine_price_adolescent'], rng, n_draws)
                           if 'vaccine_price_adolescent' in items else price['maternal'])
    delivery = {'maternal': sample_cost(items.get('delivery_maternal'), rng, n_draws),
                'adolescent': sample_cost(items.get('delivery_adolescent'), rng, n_draws)}
    outpatient = sample_cost(items.get('outpatient_episode'), rng, n_draws)
    if 'inpatient_episode' in items:
        inpatient = np.repeat(sample_cost(items['inpatient_episode'], rng, n_draws)[:, None], grid.n_bins, 1)
        inpatient_basis = 'episode_cost'
    else:
        los = [e for e in severity_doc['countries'][LOS_DONOR]['entries'] if e['quantity'] == 'los_days']
        young = sample_entry(min(los, key=lambda e: float(e['age_start_years'])), rng, n_draws)
        old = sample_entry(max(los, key=lambda e: float(e['age_start_years'])), rng, n_draws)
        _, hi = bin_bounds(grid)
        inpatient = sample_cost(items['inpatient_day'], rng, n_draws)[:, None] * np.where(
            (hi <= 0.5)[None, :], young[:, None], old[:, None])
        inpatient_basis = f'bed_day_cost_x_{LOS_DONOR}_length_of_stay'

    ages, ex = life_expectancy(life_table, severity_doc_iso3(costs_doc, country))
    yll = yll_by_bin(grid, ages, ex, discount_rate)
    dw = costs_doc['global']['disability_weights']['pertussis_acute_moderate']
    episode = float(dw['duration_days']) / 365.25
    yld_unit = float(dw['value']) * ((1 - np.exp(-discount_rate * episode)) / discount_rate
                                     if discount_rate else episode)

    rows = []
    for s, arm in enumerate(arms):
        prog = None if arm == 'baseline' else _programme_of(arm)
        vacc = np.zeros(n_draws) if prog is None else doses_d[s] * price[prog]
        deliv = np.zeros(n_draws) if prog is None else doses_d[s] * delivery[prog]
        hosp_cost = (conf_d[s][None] * hosp * inpatient).sum(axis=1)
        op_cost = (conf_d[s][None] * (1 - hosp) * outpatient[:, None]).sum(axis=1)
        yll_t = (deaths_d[:, s] * yll[None]).sum(axis=1)
        yld_base = conf_d[s].sum() if yld_scope == 'confirmed' else typ_d[s].sum()
        hosp_inf_d = (conf_d[s][None, :12] * hosp[:, :12]).sum(axis=1)
        deaths_s = deaths_d[:, s].sum(axis=1)
        yld_t = (yld_base * yld_unit
                 + np.clip(hosp_inf_d - deaths_d[:, s, :12].sum(axis=1), 0, None) * infant_survivor_qaly
                 + deaths_s * bereavement_qaly)
        rows.append(pd.DataFrame(dict(
            country=country, draw=np.arange(n_draws), strategy_id=arm,
            programme=prog or 'none', vaccine_cost=vacc, delivery_cost=deliv,
            hospital_cost=hosp_cost, outpatient_cost=op_cost,
            total_cost=vacc + deliv + hosp_cost + op_cost,
            yll=yll_t, yld=yld_t, dalys=yll_t + yld_t,
            typical_onsets=typical[s].sum(), infant_typical_onsets=typical[s, :, :12].sum(),
            hospitalisations=(confirmed[s].sum(axis=0)[None] * hosp).sum(axis=1),
            deaths=deaths_u[:, s].sum(axis=1), infant_deaths=deaths_u[:, s, :12].sum(axis=1),
            doses=doses[s].sum(), discounted_doses=doses_d[s],
            unit_price=np.zeros(n_draws) if prog is None else price[prog],
            unit_delivery=np.zeros(n_draws) if prog is None else delivery[prog])))
    draws = pd.concat(rows, ignore_index=True)
    meta = dict(country=country, design='programme_versus_no_programme',
                yld_scope=yld_scope, infant_survivor_qaly=infant_survivor_qaly,
                bereavement_qaly=bereavement_qaly,
                counterfactual_infant_deaths_per_year=counterfactual_infant_deaths_per_year,
                death_scale=death_scale,
                programme=sorted({_programme_of(a) for a in arms[1:]}),
                death_pathway=death_pathway, inpatient_basis=inpatient_basis,
                transmission_uncertainty='none_fitted_point', fit_directory=completion['fit_directory'],
                scenario_dir=str(scenario_dir), n_draws=n_draws, seed=seed,
                discount_rate=discount_rate, years=[int(years[0]), int(years[-1])],
                thresholds=thresholds(costs_doc, country))
    return draws, meta


def summarise_programme(draws, meta):
    base = draws[draws.strategy_id.eq('baseline')].set_index('draw')
    q = lambda x: (float(np.mean(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    rows = []
    for arm, g in draws[~draws.strategy_id.eq('baseline')].groupby('strategy_id', sort=True):
        g = g.set_index('draw')
        d = {c: g[c] - base[c] for c in ('total_cost', 'dalys', 'hospital_cost', 'outpatient_cost',
                                          'typical_onsets', 'infant_typical_onsets',
                                          'hospitalisations', 'deaths', 'infant_deaths')}
        averted, inc = -d['dalys'], d['total_cost']
        treat = d['hospital_cost'] + d['outpatient_cost']
        row = dict(country=meta['country'], strategy_id=arm, doses=float(g.doses.mean()),
                   transmission_uncertainty=meta['transmission_uncertainty'])
        for name, x in [('incremental_cost', inc), ('dalys_averted', averted),
                        ('typical_onsets_averted', -d['typical_onsets']),
                        ('infant_typical_onsets_averted', -d['infant_typical_onsets']),
                        ('hospitalisations_averted', -d['hospitalisations']),
                        ('infant_deaths_averted', -d['infant_deaths']),
                        ('treatment_cost_change', treat)]:
            row[name], row[f'{name}_lo'], row[f'{name}_hi'] = q(x)
        m_c, m_e = row['incremental_cost'], row['dalys_averted']
        row['icer_ratio_of_means'] = m_c / m_e if m_e > 0 else np.nan
        row['dominance'] = ('dominant' if m_e > 0 and m_c < 0 else
                            'dominated' if m_e < 0 and m_c > 0 else 'tradeoff')
        for kind, lam in meta['thresholds'].items():
            row[f'p_cost_effective_{kind}'] = float(((lam * averted - inc) > 0).mean())
            be = (lam * averted - treat - g.discounted_doses * g.unit_delivery) / g.discounted_doses
            row[f'breakeven_price_{kind}'] = float(be.median())
        row['mean_unit_price'] = float(g.unit_price.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_posterior_programme(country, posterior_dir, programme, severity_doc, costs_doc,
                                 life_table, *, economic_draws_per_transmission=50,
                                 seed=20261029, **kwargs):
    """Pool paired PSA over transmission posterior draws in ``draw_NNN/<programme>/``.

    Each transmission draw gets its own block of economic draws, so cost and
    severity uncertainty are nested inside transmission uncertainty. Draw
    indices are renumbered globally and the transmission row is kept.
    """
    folders = sorted(Path(posterior_dir).glob(f'draw_*/{programme}/complete.json'))
    if not folders:
        raise ValueError(f'no completed posterior draws under {posterior_dir}')
    frames, meta = [], None
    for i, done in enumerate(folders):
        draws, meta = evaluate_programme(country, done.parent, severity_doc, costs_doc, life_table,
                                         n_draws=economic_draws_per_transmission,
                                         seed=seed + i, **kwargs)
        draws['transmission_draw'] = i
        draws['draw'] = draws['draw'] + i * economic_draws_per_transmission
        frames.append(draws)
    meta = dict(meta, transmission_uncertainty='posterior_draws',
                n_transmission_draws=len(folders), scenario_dir=str(posterior_dir),
                n_draws=len(folders) * economic_draws_per_transmission)
    return pd.concat(frames, ignore_index=True), meta
