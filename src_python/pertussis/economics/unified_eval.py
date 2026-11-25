"""Unified evaluation of the common strategy set (00, M0, 0A, MA, current).

One health-burden definition serves every age group and time window:

* confirmed cases = (typical + r * mild) * age-specific reporting probability;
* acute YLD = disability weight x episode duration x confirmed cases (or all
  modelled typical onsets as a labelled sensitivity);
* YLL = deaths x discounted remaining life expectancy at the bin midpoint;
* ``dalys = yll + yld``.  Survivor-hospitalisation and bereavement QALY losses
  are reported in the separate ``qaly_extension`` column and never enter DALYs.

Deaths use the per-confirmed-case fatality where a country has one (all ages)
and otherwise calibrated per-onset infant risks (infants only); the pathway and
its age scope are recorded in the metadata.  Costs are attributed to the
programme that incurs them: each programme's doses carry its own vaccine price
and delivery cost.  Every strategy is compared with ``00`` within draw, so cost,
severity and borrowed-risk draws are paired across strategies.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import REPO_ROOT
from ..grid import AgeGrid, band_matrix
from ..scenarios.aggregation import (AGE_GROUPS, WINDOWS_20Y, WINDOWS_30Y_EXTRA, age_group_matrix)
from ..scenarios.common_strategies import STRATEGY_IDS
from .evaluate import (BAND_BOUNDS, DEATH_RISK_SOURCE, HOSPITAL_DONOR, LOGIT_COLUMNS, LOS_DONOR,
                       bin_bounds, life_expectancy, reference_infant_risks, sample_bin_values,
                       sample_cost, sample_entry, severity_doc_iso3, thresholds, yll_by_bin)
from .programme_eval import risk_file_infant_risks
from .age_scoped_hospital_cost import MissingAgeCostError, allocate_hospital_cost
from .joint_age_severity import joint_risk_weights

PROGRAMMES = ('maternal', 'adolescent')
DESIGNED_SET = ('00', 'M0', '0A', 'MA')   # strategies compared on one frontier
HEALTH_MEASURES = ('typical_onsets', 'confirmed_cases', 'hospitalisations', 'deaths',
                   'mild_onsets', 'all_symptomatic_onsets', 'yll', 'yld', 'dalys')


def windows_for_horizon(horizon):
    if horizon == 20:
        return dict(WINDOWS_20Y)
    if horizon == 30:
        return {**WINDOWS_20Y, **WINDOWS_30Y_EXTRA}
    raise ValueError('windows are defined for 20- and 30-year horizons')


def read_common_projection(folder, years):
    """Onsets (strategy, year, bin) and doses (strategy, programme, year)."""
    folder = Path(folder)
    years = list(years)
    n_bins = AgeGrid().n_bins
    onsets = pd.read_csv(folder / 'annual_age_onsets.csv.gz')
    onsets = onsets[onsets.year.isin(years)]
    arms = [s for s in STRATEGY_IDS if s in set(onsets.strategy_id)]
    if arms != list(STRATEGY_IDS):
        raise ValueError(f'projection must contain strategies {STRATEGY_IDS}, found {arms}')
    index = pd.MultiIndex.from_product([arms, years, range(n_bins)],
                                      names=['strategy_id', 'year', 'bin_index'])
    if onsets.duplicated(index.names).any():
        raise ValueError('duplicate onset cells')
    table = onsets.set_index(index.names).reindex(index)
    values = table[['typical_onsets', 'mild_onsets']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('incomplete or invalid strategy/year/age onset grid')
    shape = (len(arms), len(years), n_bins)
    receipts = pd.read_csv(folder / 'receipts.csv.gz')
    ridx = pd.MultiIndex.from_product([arms, PROGRAMMES, years], names=['strategy_id', 'programme', 'year'])
    if receipts.duplicated(ridx.names).any():
        raise ValueError('duplicate receipt cells')
    doses = receipts.set_index(ridx.names).reindex(ridx).doses.to_numpy(float)
    if not np.isfinite(doses).all() or (doses < 0).any():
        raise ValueError('incomplete or invalid dose ledger')
    completion = json.loads((folder / 'complete.json').read_text())
    return (arms, values[:, 0].reshape(shape), values[:, 1].reshape(shape),
            doses.reshape(len(arms), len(PROGRAMMES), len(years)), completion)


def _reporting(completion, repo_root):
    if 'reporting' in completion:
        rep = completion['reporting']
        return (np.array([rep[c] for c in LOGIT_COLUMNS], float),
                float(rep['breakthrough_reporting_ratio']))
    fit = Path(repo_root) / completion['fit_directory']
    row = pd.read_csv(fit / 'fit_summary.csv')
    row = row[row.start_index.eq(completion['draw_id'])]
    if len(row) != 1:
        raise ValueError('scenario fit row is missing or ambiguous')
    return (row.iloc[0][LOGIT_COLUMNS].to_numpy(float),
            float(row.iloc[0]['breakthrough_reporting_ratio']))


def _window_mask(years, windows):
    names, cols = list(windows), []
    for name in names:
        y0, y1 = windows[name]
        if y0 < years.min() or y1 > years.max():
            raise ValueError(f'window {name} lies outside the projection years')
        cols.append((years >= y0) & (years <= y1))
    return names, np.stack(cols, axis=1).astype(float)


def _agg(x, wmask, gmat):
    """(..., S, Y, B) -> (..., S, W, G) with window masks and age-group weights."""
    return np.einsum('...syb,yw,bg->...swg', x, wmask, gmat)


def _joint_reporting_group_weights(grid, groups, logits):
    """Mid-year joint reporting and completed-age group weights for each bin.

    Do not multiply marginal reporting and group proportions: a cohort bin
    can contain two completed ages with different reporting probabilities.
    This remains a mid-year approximation to event timing within a year.
    """
    ages = np.arange(101)
    weights = grid.completed_age_weights(.5, max_age=100)
    report_bands = np.column_stack([(ages >= lo) & (ages < hi) for lo, hi in BAND_BOUNDS])
    group_bands = np.column_stack([(ages >= lo) & (ages < hi) for lo, hi in groups.values()])
    probabilities = expit(np.asarray(logits, dtype=float))
    if probabilities.shape != (len(BAND_BOUNDS),) or not np.isfinite(probabilities).all():
        raise ValueError('six finite reporting logits required')
    per_age = report_bands @ probabilities
    marginal = weights @ per_age
    if np.any(marginal <= 0):
        raise ValueError('reporting probability must be positive')
    joint = weights @ (per_age[:, None] * group_bands)
    return marginal, joint / marginal[:, None]


def evaluate_common(country, scenario_dir, severity_doc, costs_doc, life_table, *, n_draws=1000,
                    seed=20261031, discount_rate=0.03, start_year=2027, horizon=20,
                    repo_root=REPO_ROOT, risk_file=None, risk_scenario='reference',
                    yld_scope='confirmed', infant_survivor_qaly=0.097, bereavement_qaly=3.70,
                    death_multiplier=1.0, groups=AGE_GROUPS, windows=None,
                    acute_weight_override=None, conditional_risks=None, make_breakdown=True,
                    retain_window_draws=False, inpatient_fallback='error'):
    """Return (draws, breakdown, meta) for one country's common-strategy projection.

    ``draws`` has one row per (draw, strategy) over the whole horizon.
    ``breakdown`` has draw-summarised levels and changes versus ``00`` for every
    age group and window.  ``death_multiplier`` scales all deaths after the
    fact; it is a post-processing burden scenario, not a refitted counterfactual.
    """
    if yld_scope not in ('confirmed', 'all_typical', 'all_symptomatic'):
        raise ValueError('yld_scope must be confirmed, all_typical or all_symptomatic')
    if n_draws < 1 or horizon < 1 or discount_rate < 0 or death_multiplier < 0:
        raise ValueError('positive draws/horizon and nonnegative rate/multiplier required')
    windows = windows_for_horizon(horizon) if windows is None else windows
    grid = AgeGrid()
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + horizon)
    arms, typical, mild, doses, completion = read_common_projection(scenario_dir, years)
    logits, mild_ratio = _reporting(completion, repo_root)
    names, wmask = _window_mask(years, windows)
    full = np.ones((len(years), 1))
    _, gnames, gmat = age_group_matrix(grid, groups)
    disc = (1 + discount_rate) ** -(years - start_year)
    report, reported_gmat = _joint_reporting_group_weights(grid, groups, logits)  # (B,), (B,G)
    confirmed = (typical + mild_ratio * mild) * report[None, None, :]               # (S, Y, B)
    acute_base = confirmed if yld_scope == 'confirmed' else typical

    sev = severity_doc['countries'][country]
    own = sev.get('entries') or []
    donor = severity_doc['countries'][HOSPITAL_DONOR]['entries']
    hosp_own = [e for e in own if e['quantity'] == 'hospitalisation_proportion']
    hosp_donor = [e for e in donor if e['quantity'] == 'hospitalisation_proportion']
    hosp, hosp_use_own, hosp_entries = sample_bin_values(
        hosp_own, grid, rng, n_draws, donor_entries=hosp_donor, return_entry_samples=True)
    hosp_joint = joint_risk_weights(grid, groups, logits, hosp_own, hosp_donor,
                                    *hosp_entries, hosp_use_own, hosp)
    cfr_own = [e for e in own if e['quantity'] == 'cfr']
    if cfr_own:
        cfr, cfr_use_own, cfr_entries = sample_bin_values(cfr_own, grid, rng, n_draws, return_entry_samples=True)
        cfr_joint = joint_risk_weights(grid, groups, logits, cfr_own, [], *cfr_entries, cfr_use_own, cfr) * death_multiplier
        deaths = confirmed[None] * cfr[:, None, None, :]                             # (N, S, Y, B)
        death_pathway, death_scope = 'national_per_confirmed_case_cfr', 'all_ages'
    else:
        source = DEATH_RISK_SOURCE[country]
        if conditional_risks is not None:
            from .projections import conditional_risk_arrays
            risks = conditional_risks
            expected = int(completion['source_parameter_row'])
            if not risks.source_parameter_row.eq(expected).all():
                raise ValueError('conditional mortality risks do not match the transmission parameter row')
            risk_ids = risks.draw_id.unique()
            if len(risk_ids) != 1:
                raise ValueError('one matched transmission draw is required in conditional risks')
            pt, pm, _ = conditional_risk_arrays(risks, np.repeat(risk_ids[0], n_draws), rng, country, grid.n_bins)
        else:
            if completion.get('inference_status') == 'posterior_draw' and country in ('England', 'Australia'):
                raise ValueError('posterior evaluation requires matched conditional mortality risks')
            pt, pm = (risk_file_infant_risks(risk_file, risk_scenario, rng, n_draws, grid)
                  if risk_file is not None else
                  reference_infant_risks(source, rng, n_draws, grid, repo_root))
        deaths = typical[None] * pt[:, None, None, :] + mild[None] * pm[:, None, None, :]
        death_pathway, death_scope = f'{source}_calibrated_per_onset_infant_risk', 'infants_only'
    deaths = deaths * death_multiplier

    if inpatient_fallback not in ('error', 'documented_extrapolation'):
        raise ValueError('unknown inpatient cost fallback')
    item_list = costs_doc['countries'][country]['items']
    items = {i['item']: i for i in item_list}
    price = {'maternal': sample_cost(items.get('vaccine_price_maternal'), rng, n_draws)}
    price['adolescent'] = (sample_cost(items['vaccine_price_adolescent'], rng, n_draws)
                           if 'vaccine_price_adolescent' in items else price['maternal'])
    delivery = {'maternal': sample_cost(items.get('delivery_maternal'), rng, n_draws),
                'adolescent': sample_cost(items.get('delivery_adolescent'), rng, n_draws)}
    outpatient = sample_cost(items.get('outpatient_episode'), rng, n_draws)
    scoped = []
    fallback = None
    if 'inpatient_episode' in items:
        scoped = [dict(age_start_years=i.get('age_start_years', 0),
                       age_end_years=i.get('age_end_years', None),
                       draws=sample_cost(i, rng, n_draws))
                  for i in item_list if i['item'] == 'inpatient_episode']
        lo, hi = bin_bounds(grid)
        try:
            inpatient, cost_coverage = allocate_hospital_cost(lo, hi, scoped)
            inpatient_basis = 'age_scoped_episode_cost'
        except MissingAgeCostError as exc:
            if inpatient_fallback == 'error':
                raise
            if 'inpatient_day' in items:
                los = [e for e in severity_doc['countries'][LOS_DONOR]['entries'] if e['quantity'] == 'los_days']
                old = sample_entry(max(los, key=lambda e: float(e['age_start_years'])), rng, n_draws)
                fallback = np.repeat((sample_cost(items['inpatient_day'], rng, n_draws) * old)[:, None], grid.n_bins, axis=1)
                inpatient_basis = 'age_scoped_episode_with_day_cost_borrowed_child_los_fallback'
            elif len(scoped) == 1:
                fallback = np.repeat(scoped[0]['draws'][:, None], grid.n_bins, axis=1)
                inpatient_basis = 'age_scoped_episode_with_explicit_episode_cost_extrapolation'
            else:
                raise ValueError('multiple episode costs need an explicit fallback rule') from exc
            inpatient, cost_coverage = allocate_hospital_cost(lo, hi, scoped, fallback=fallback)
    else:
        los = [e for e in severity_doc['countries'][LOS_DONOR]['entries'] if e['quantity'] == 'los_days']
        young = sample_entry(min(los, key=lambda e: float(e['age_start_years'])), rng, n_draws)
        old = sample_entry(max(los, key=lambda e: float(e['age_start_years'])), rng, n_draws)
        _, hi = bin_bounds(grid)
        inpatient = sample_cost(items['inpatient_day'], rng, n_draws)[:, None] * np.where(
            (hi <= 0.5)[None, :], young[:, None], old[:, None])
        inpatient_basis = f'bed_day_cost_x_{LOS_DONOR}_length_of_stay'

    ages, ex = life_expectancy(life_table, severity_doc_iso3(costs_doc, country))
    yll_bin = yll_by_bin(grid, ages, ex, discount_rate)

    # Retain the exact same sampled unit-cost entries/fallbacks; apply each
    # at the segment where the associated hospital risk is incurred.
    def inpatient_at_age(b, age):
        matches = [e for e in scoped if float(e['age_start_years']) <= age <
                   (np.inf if e['age_end_years'] is None else float(e['age_end_years']))]
        if len(matches) > 1:
            raise ValueError('overlapping age-specific inpatient costs')
        if matches:
            return matches[0]['draws']
        if scoped and fallback is None:
            raise MissingAgeCostError('inpatient segment lacks a cost')
        return fallback[:, b] if fallback is not None else inpatient[:, b]

    cost_boundaries = [v for e in scoped for v in (e['age_start_years'], e['age_end_years']) if v is not None]
    hosp_cost_joint = joint_risk_weights(grid, groups, logits, hosp_own, hosp_donor,
        *hosp_entries, hosp_use_own, hosp, segment_multiplier=inpatient_at_age,
        extra_boundaries=cost_boundaries)
    if death_pathway == 'national_per_confirmed_case_cfr':
        def yll_at_age(b, age):
            remaining = np.interp(age, ages, ex)
            return remaining if discount_rate == 0 else -np.expm1(-discount_rate * remaining) / discount_rate
        cfr_yll_joint = joint_risk_weights(grid, groups, logits, cfr_own, [], *cfr_entries,
            cfr_use_own, cfr, segment_multiplier=yll_at_age) * death_multiplier
    dw = costs_doc['global']['disability_weights']['pertussis_acute_moderate']
    episode = float(dw['duration_days']) / 365.25
    acute_weight = float(dw['value'] if acute_weight_override is None else acute_weight_override)
    yld_unit = acute_weight * ((1 - np.exp(-discount_rate * episode)) / discount_rate
                                     if discount_rate else episode)
    acute_yld = acute_base * yld_unit
    mild_weight, mild_duration = None, None
    if yld_scope == 'all_symptomatic':
        mild_dw = costs_doc['global']['disability_weights']['pertussis_acute_mild']
        mild_weight, mild_duration = float(mild_dw['value']), float(mild_dw['duration_days'])
        mild_episode = mild_duration / 365.25
        mild_unit = mild_weight * ((1 - np.exp(-discount_rate * mild_episode)) / discount_rate
                                  if discount_rate else mild_episode)
        acute_yld = typical * yld_unit + mild * mild_unit

    d4 = disc[None, None, :, None]
    hosp_cases = confirmed[None] * hosp[:, None, None, :]                             # (N, S, Y, B)
    tensors = {                                                                        # per (N or 1, S, Y, B)
        'typical_onsets': typical[None],
        'mild_onsets': mild[None],
        'all_symptomatic_onsets': (typical + mild)[None],
        'confirmed_cases': confirmed[None],
        'deaths': deaths,
        'yll': deaths * (yll_bin[None, None, None, :] * d4),
        'yld': acute_yld[None] * d4,
    }
    infant = (np.arange(grid.n_bins) < 12)[None, None, None, :]


    doses_d = np.einsum('spy,y,yw->spw', doses, disc, np.concatenate([wmask, full], axis=1))
    dose_total = doses.sum(axis=2)                                                     # (S, P)
    disc_doses = doses_d[:, :, -1]                                                     # (S, P) full horizon
    vacc = {p: disc_doses[:, k][None, :] * price[p][:, None] for k, p in enumerate(PROGRAMMES)}
    deliv = {p: disc_doses[:, k][None, :] * delivery[p][:, None] for k, p in enumerate(PROGRAMMES)}

    # Severity is applied jointly with reporting and group membership, using
    # the same sampled entry risks as the historical bin-level calculation.
    reported_measures = {'confirmed_cases'}
    if yld_scope == 'confirmed':
        reported_measures.add('yld')
    base_onsets = typical + mild_ratio * mild

    def joint_agg(masks, joint, discount=False):
        time = masks * disc[:, None] if discount else masks
        bins = np.einsum('syb,yw->swb', base_onsets, time, optimize=True)
        return np.einsum('swb,nbg->nswg', bins, joint, optimize=True)

    def aggregate_all(masks):
        result = {k: _agg(v, masks, reported_gmat if k in reported_measures else gmat)
                  for k, v in tensors.items()
                  if k not in ('dalys', 'hospitalisations', 'hospital_cost', 'outpatient_cost', 'qaly_extension')}
        result['hospitalisations'] = joint_agg(masks, hosp_joint)
        result['hospital_cost'] = joint_agg(masks, hosp_cost_joint, discount=True)
        reported_discounted = _agg(confirmed[None] * d4, masks, reported_gmat)
        result['outpatient_cost'] = (reported_discounted - joint_agg(masks, hosp_joint, discount=True)) * outpatient[:, None, None, None]
        if death_pathway == 'national_per_confirmed_case_cfr':
            result['deaths'] = joint_agg(masks, cfr_joint)
            result['yll'] = joint_agg(masks, cfr_yll_joint, discount=True)
            death_discounted = joint_agg(masks, cfr_joint, discount=True)
        else:
            death_discounted = _agg(deaths * d4, masks, gmat)
        # Survivors are infants only; in each infant month all groups and
        # reporting rates coincide, so the existing within-bin clip is retained.
        survivors = np.clip(hosp_cases - deaths, 0, None) * infant * infant_survivor_qaly * d4
        result['qaly_extension'] = _agg(survivors, masks, gmat) + death_discounted * bereavement_qaly
        result['dalys'] = result['yll'] + result['yld']
        result['treatment_cost'] = result['hospital_cost'] + result['outpatient_cost']
        return result

    # Whole-horizon table: one row per (draw, strategy).
    allw = {k: np.broadcast_to(a, (n_draws,) + a.shape[1:])
            for k, a in aggregate_all(full).items()}
    rows = []
    for s, arm in enumerate(arms):
        row = dict(country=country, draw=np.arange(n_draws), strategy_id=arm)
        for p in PROGRAMMES:
            row[f'vaccine_cost_{p}'] = vacc[p][:, s]
            row[f'delivery_cost_{p}'] = deliv[p][:, s]
            row[f'doses_{p}'] = np.full(n_draws, dose_total[s, PROGRAMMES.index(p)])
            row[f'discounted_doses_{p}'] = np.full(n_draws, disc_doses[s, PROGRAMMES.index(p)])
            row[f'unit_price_{p}'] = price[p]
            row[f'unit_delivery_{p}'] = delivery[p]
        gi = gnames.index('all_ages')
        row['hospital_cost'] = allw['hospital_cost'][:, s, 0, gi]
        row['outpatient_cost'] = allw['outpatient_cost'][:, s, 0, gi]
        row['treatment_cost'] = allw['treatment_cost'][:, s, 0, gi]
        row['vaccine_cost'] = sum(vacc[p][:, s] for p in PROGRAMMES)
        row['delivery_cost'] = sum(deliv[p][:, s] for p in PROGRAMMES)
        row['total_cost'] = row['vaccine_cost'] + row['delivery_cost'] + row['treatment_cost']
        infant_g = gnames.index('infant_0')
        row['infant_treatment_cost'] = allw['treatment_cost'][:, s, 0, infant_g]
        row['total_cost_infant_scope'] = row['vaccine_cost'] + row['delivery_cost'] + row['infant_treatment_cost']
        for m in (*HEALTH_MEASURES, 'qaly_extension'):
            for g, gname in enumerate(gnames):
                row[m if gname == 'all_ages' else f'{m}__{gname}'] = allw[m][:, s, 0, g]
        rows.append(pd.DataFrame(row))
    draws = pd.concat(rows, ignore_index=True)

    # Age group x window breakdown, levels and change versus 00 (draw-paired).
    q = lambda x: (float(np.mean(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    base = arms.index('00')
    breakdown = []
    wide = aggregate_all(wmask) if make_breakdown or retain_window_draws else {}
    for measure, arr in (wide.items() if make_breakdown else []):
        if measure in ('hospital_cost', 'outpatient_cost'):
            continue
        arr = np.broadcast_to(arr, (n_draws,) + arr.shape[1:])
        for s, arm in enumerate(arms):
            for w, wname in enumerate(names):
                for g, gname in enumerate(gnames):
                    lvl = arr[:, s, w, g]
                    chg = lvl - arr[:, base, w, g]
                    for kind, x in (('level', lvl), ('change_vs_00', chg)):
                        mean, lo, hi = q(x)
                        breakdown.append((country, arm, wname, gname, measure, kind, mean, lo, hi))
    # Programme costs by window (all ages): attributed to the programme that incurs them.
    for s, arm in enumerate(arms if make_breakdown else []):
        for w, wname in enumerate(names):
            for p_i, p in enumerate(PROGRAMMES):
                for label, unit in (('vaccine_cost', price[p]), ('delivery_cost', delivery[p])):
                    lvl = doses_d[s, p_i, w] * unit
                    chg = lvl - doses_d[base, p_i, w] * unit
                    for kind, x in (('level', lvl), ('change_vs_00', chg)):
                        mean, lo, hi = q(x)
                        breakdown.append((country, arm, wname, 'all_ages', f'{label}_{p}', kind, mean, lo, hi))
            programme_cost = sum(doses_d[s, j, w] * (price[p] + delivery[p])
                                 for j, p in enumerate(PROGRAMMES))
            base_programme = sum(doses_d[base, j, w] * (price[p] + delivery[p])
                                for j, p in enumerate(PROGRAMMES))
            for scope in ('all_ages', 'infant_0'):
                g = gnames.index(scope)
                lvl = programme_cost + wide['treatment_cost'][:, s, w, g]
                chg = lvl - base_programme - wide['treatment_cost'][:, base, w, g]
                for kind, x in (('level', lvl), ('change_vs_00', chg)):
                    mean, lo, hi = q(x)
                    breakdown.append((country, arm, wname, scope, 'programme_plus_scope_care_cost',
                                      kind, mean, lo, hi))
    breakdown = pd.DataFrame(breakdown, columns=['country', 'strategy_id', 'window', 'age_group',
                                                 'measure', 'kind', 'mean', 'lo', 'hi'])
    if retain_window_draws:
        window_rows = []
        for s, arm in enumerate(arms):
            for w, wname in enumerate(names):
                programme_cost = sum(doses_d[s, j, w] * (price[p] + delivery[p])
                                     for j, p in enumerate(PROGRAMMES))
                for g, gname in enumerate(gnames):
                    rec = dict(country=country, draw=np.arange(n_draws), strategy_id=arm,
                               window=wname, age_group=gname)
                    for k in (*HEALTH_MEASURES, 'treatment_cost'):
                        rec[k] = np.broadcast_to(wide[k][:, s, w, g], (n_draws,))
                    if gname in ('infant_0', 'all_ages'):
                        rec['programme_plus_scope_care_cost'] = programme_cost + rec['treatment_cost']
                    window_rows.append(pd.DataFrame(rec))
        draws.attrs['window_draws'] = pd.concat(window_rows, ignore_index=True)
    meta = dict(country=country, design='common_strategy_set', strategies=list(arms),
                yld_scope=yld_scope, dalys='yll_plus_acute_yld',
                age_reporting_allocation='joint_midyear_reporting_severity_cost_and_cfr_yll; per-onset infant YLL at monthly midpoint',
                qaly_extension_in_dalys=False, infant_survivor_qaly=infant_survivor_qaly,
                acute_disability_weight=acute_weight, acute_duration_days=float(dw['duration_days']),
                mild_disability_weight=mild_weight, mild_duration_days=mild_duration,
                infant_cost_scope='all programme costs plus infant treatment costs only',
                bereavement_qaly=bereavement_qaly, death_multiplier=death_multiplier,
                death_pathway=death_pathway, death_age_scope=death_scope,
                inpatient_basis=inpatient_basis, inpatient_fallback=inpatient_fallback,
                inpatient_episode_age_ranges=[dict(age_start_years=i.get('age_start_years', 0),
                                                   age_end_years=i.get('age_end_years', None))
                                              for i in item_list if i['item'] == 'inpatient_episode'],
                hospital_donor=HOSPITAL_DONOR,
                transmission_uncertainty=('posterior_draws' if completion.get('inference_status') == 'posterior_draw'
                                          else 'none_fitted_point'),
                fit_directory=completion['fit_directory'], scenario_dir=str(scenario_dir),
                coverage=completion.get('coverage'), n_draws=n_draws, seed=seed,
                discount_rate=discount_rate, years=[int(years[0]), int(years[-1])],
                booster_structure=completion.get('booster_structure', 'replacement'),
                tdap_years=completion.get('tdap_years'),
                duration_scope=completion.get('duration_scope', 'fitted'),
                age_groups={g: list(map(float, groups[g])) for g in gnames}, windows=windows,
                thresholds=thresholds(costs_doc, country))
    return draws, breakdown, meta


def efficiency_frontier(names, cost, effect):
    """Simple and extended dominance on mean cost and mean effect.

    Returns a frame with status ``frontier``, ``dominated`` or
    ``extended_dominated`` and, for frontier options after the cheapest, the
    incremental cost per unit of effect against the previous frontier option.
    """
    cost, effect = np.asarray(cost, float), np.asarray(effect, float)
    order = list(np.argsort(cost, kind='stable'))
    dominated = set()
    for i in order:
        for j in order:
            if j == i:
                continue
            better = cost[j] <= cost[i] and effect[j] >= effect[i]
            strict = cost[j] < cost[i] or effect[j] > effect[i] or j < i
            if better and strict:
                dominated.add(i)
                break
    extended = set()
    while True:
        live = [i for i in order if i not in dominated and i not in extended]
        for k in range(1, len(live) - 1):
            a, b, c = live[k - 1], live[k], live[k + 1]
            if (cost[b] - cost[a]) / (effect[b] - effect[a]) > (cost[c] - cost[b]) / (effect[c] - effect[b]):
                extended.add(b)
                break
        else:
            break
    live = [i for i in order if i not in dominated and i not in extended]
    icer = {i: (cost[i] - cost[p]) / (effect[i] - effect[p]) for p, i in zip(live[:-1], live[1:])}
    return pd.DataFrame(dict(
        strategy_id=list(names), mean_cost=cost, mean_effect=effect,
        status=['dominated' if i in dominated else 'extended_dominated' if i in extended else 'frontier'
                for i in range(len(names))],
        incremental_icer=[icer.get(i, np.nan) for i in range(len(names))]))


def summarise_common(draws, meta):
    """(summary, frontier, decision, marginal) tables for one country."""
    arms = list(meta['strategies'])
    by = {a: draws[draws.strategy_id.eq(a)].set_index('draw') for a in arms}
    base = by['00']
    q = lambda x: (float(np.mean(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    change = lambda a, col, ref='00': by[a][col] - by[ref][col]
    cols = {'incremental_cost': ('total_cost', 1), 'treatment_cost_change': ('treatment_cost', 1),
            'incremental_cost_infant': ('total_cost_infant_scope', 1),
            'dalys_averted': ('dalys', -1), 'yll_averted': ('yll', -1), 'yld_averted': ('yld', -1),
            'all_symptomatic_onsets_averted': ('all_symptomatic_onsets', -1),
            'typical_onsets_averted': ('typical_onsets', -1), 'mild_onsets_averted': ('mild_onsets', -1),
            'confirmed_cases_averted': ('confirmed_cases', -1), 'hospitalisations_averted': ('hospitalisations', -1),
            'deaths_averted': ('deaths', -1), 'qaly_extension_averted': ('qaly_extension', -1)}
    for g in meta['age_groups']:
        if g == 'all_ages':
            continue
        for m in ('confirmed_cases', 'all_symptomatic_onsets', 'hospitalisations', 'deaths', 'dalys'):
            cols[f'{m}_averted__{g}'] = (f'{m}__{g}', -1)
    rows = []
    for a in arms[1:]:
        row = dict(country=meta['country'], strategy_id=a,
                   doses_maternal=float(by[a].doses_maternal.mean()),
                   doses_adolescent=float(by[a].doses_adolescent.mean()),
                   death_age_scope=meta['death_age_scope'],
                   transmission_uncertainty=meta['transmission_uncertainty'])
        for name, (col, sign) in cols.items():
            row[name], row[f'{name}_lo'], row[f'{name}_hi'] = q(sign * change(a, col))
        m_c, m_e = row['incremental_cost'], row['dalys_averted']
        row['icer_ratio_of_means'] = m_c / m_e if m_e > 0 else np.nan
        row['health_cost_status'] = health_cost_status(m_c, m_e)
        infant_e = row['dalys_averted__infant_0']
        row['icer_infant_ratio_of_means'] = row['incremental_cost_infant'] / infant_e if infant_e > 0 else np.nan
        row['infant_health_cost_status'] = health_cost_status(row['incremental_cost_infant'], infant_e)
        single = {'M0': 'maternal', '0A': 'adolescent'}.get(a)
        for kind, lam in meta['thresholds'].items():
            nmb = lam * (-change(a, 'dalys')) - change(a, 'total_cost')
            row[f'p_cost_effective_{kind}'] = float((nmb > 0).mean())
            if single:
                dd = by[a][f'discounted_doses_{single}']
                be = (lam * (-change(a, 'dalys')) - change(a, 'treatment_cost')
                      - dd * by[a][f'unit_delivery_{single}']) / dd
                row[f'breakeven_price_{kind}'] = float(be.median())
            else:
                row[f'breakeven_price_{kind}'] = np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)

    designed = [a for a in DESIGNED_SET if a in arms]
    mean_cost = [by[a].total_cost.mean() for a in designed]
    mean_eff = [-(by[a].dalys - base.dalys).mean() for a in designed]
    frontier = efficiency_frontier(designed, mean_cost, mean_eff).assign(country=meta['country'])

    dec = []
    for kind, lam in meta['thresholds'].items():
        nmb = pd.DataFrame({a: lam * (-(by[a].dalys - base.dalys)) - (by[a].total_cost - base.total_cost)
                            for a in designed})
        best = nmb.idxmax(axis=1)
        for a in designed:
            dec.append(dict(country=meta['country'], strategy_id=a, threshold_type=kind, threshold=lam,
                            mean_nmb=float(nmb[a].mean()), p_nmb_positive=float((nmb[a] > 0).mean()),
                            p_optimal=float((best == a).mean())))
    decision = pd.DataFrame(dec)

    # Marginal effect of one programme given the other, and the interaction term.
    marg = []
    for name, (col, sign) in cols.items():
        if name.endswith(('_lo', '_hi')):
            continue
        if not {'M0', '0A', 'MA'} <= set(arms):
            break
        m0, a0, ma = (sign * change(s, col) for s in ('M0', '0A', 'MA'))
        parts = {'maternal_alone': m0, 'adolescent_alone': a0, 'both': ma,
                 'maternal_added_to_adolescent': ma - a0, 'adolescent_added_to_maternal': ma - m0,
                 'interaction': ma - m0 - a0}
        for label, x in parts.items():
            mean, lo, hi = q(x)
            marg.append(dict(country=meta['country'], measure=name, effect=label, mean=mean, lo=lo, hi=hi))
    return summary, frontier, decision, pd.DataFrame(marg)


def health_cost_status(cost, effect):
    if effect <= 0:
        return 'more_cost_no_health_gain' if cost >= 0 else 'less_cost_less_health'
    return 'cost_saving_health_gain' if cost < 0 else 'more_cost_health_gain'


def annual_case_changes(country, scenario_dir, *, start_year=2027, horizon=30, repo_root=REPO_ROOT):
    """Undiscounted annual cases; observed-scale cases and unobserved symptomatic courses stay distinct."""
    years = np.arange(start_year, start_year + horizon)
    arms, typical, mild, _, completion = read_common_projection(scenario_dir, years)
    logits, ratio = _reporting(completion, repo_root)
    grid, names, mat = age_group_matrix()
    reporting, reported_mat = _joint_reporting_group_weights(grid, AGE_GROUPS, logits)
    values = {'confirmed_cases': (typical + ratio * mild) * reporting[None, None, :],
              'typical_onsets': typical, 'mild_onsets': mild, 'all_symptomatic_onsets': typical + mild}
    rows = []
    for measure, arr in values.items():
        a = arr @ (reported_mat if measure == 'confirmed_cases' else mat)
        for s, arm in enumerate(arms):
            for yi, year in enumerate(years):
                for g, name in enumerate(names):
                    rows.append(dict(country=country, strategy_id=arm, year=int(year), age_group=name,
                                     measure=measure, level=a[s, yi, g], baseline=a[0, yi, g],
                                     averted=a[0, yi, g] - a[s, yi, g]))
    return pd.DataFrame(rows)


def price_burden_surface(draws, meta, *, price_multipliers=None, burden_multipliers=None):
    """Expected-NMB regions with programme vaccine prices scaled together; delivery and case costs fixed.

    Death burden multiplies YLL after projection. This is a conditional mortality
    scenario, not increased incidence or a refitted transmission counterfactual.
    Negative effects and negative break-even prices are retained.
    """
    prices = np.linspace(0, 2, 81) if price_multipliers is None else np.asarray(price_multipliers)
    burdens = np.linspace(.5, 8, 76) if burden_multipliers is None else np.asarray(burden_multipliers)
    means = draws.groupby('strategy_id').mean(numeric_only=True)
    base = means.loc['00']
    coeffs, grids, bounds = [], [], []
    for kind, lam in meta['thresholds'].items():
        matrices = []
        for arm in DESIGNED_SET:
            d = means.loc[arm] - base
            vacc, other = d.vaccine_cost, d.delivery_cost + d.treatment_cost
            yll, yld = -d.yll, -d.yld
            nmb = lam * (yll * burdens[:, None] + yld) - other - vacc * prices[None, :]
            matrices.append(np.broadcast_to(nmb, (len(burdens), len(prices))))
            coeffs.append(dict(country=meta['country'], strategy_id=arm, threshold_type=kind,
                               threshold=lam, vaccine_cost_increment=vacc, other_cost_increment=other,
                               yll_averted=yll, yld_averted=yld))
            if vacc > 0:
                for b in burdens:
                    bounds.append(dict(country=meta['country'], strategy_id=arm, threshold_type=kind,
                                       burden_multiplier=b, breakeven_price_multiplier=(lam*(yll*b+yld)-other)/vacc))
        stack = np.stack(matrices)
        best = stack.argmax(axis=0)
        for bi, b in enumerate(burdens):
            for pi, price in enumerate(prices):
                grids.append(dict(country=meta['country'], threshold_type=kind, threshold=lam,
                                  burden_multiplier=b, price_multiplier=price,
                                  preferred_strategy=DESIGNED_SET[best[bi, pi]],
                                  mean_nmb=stack[best[bi, pi], bi, pi]))
    return pd.DataFrame(grids), pd.DataFrame(bounds), pd.DataFrame(coeffs)


def evaluate_common_posterior(country, posterior_dir, severity_doc, costs_doc, life_table, *,
                              economic_draws_per_transmission=50, seed=20261031, **kwargs):
    """Pool paired PSA over transmission posterior draws in ``draw_NNN/``.

    Cost, severity and borrowed-risk draws are nested inside each transmission
    draw; the breakdown is recomputed from pooled draws only for the whole
    horizon, so a pooled breakdown table is not returned.
    """
    folders = sorted(Path(posterior_dir).glob('draw_*/complete.json'))
    if not folders:
        raise ValueError(f'no completed posterior draws under {posterior_dir}')
    frames, meta = [], None
    for i, done in enumerate(folders):
        draws, _, meta = evaluate_common(country, done.parent, severity_doc, costs_doc, life_table,
                                         n_draws=economic_draws_per_transmission, seed=seed + i, **kwargs)
        draws['transmission_draw'] = i
        draws['draw'] = draws['draw'] + i * economic_draws_per_transmission
        frames.append(draws)
    meta = dict(meta, transmission_uncertainty='posterior_draws', n_transmission_draws=len(folders),
                scenario_dir=str(posterior_dir), n_draws=len(folders) * economic_draws_per_transmission)
    return pd.concat(frames, ignore_index=True), meta
