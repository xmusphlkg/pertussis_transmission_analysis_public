"""Programme-versus-no-programme scenarios at realistic coverage.

Each evaluation keeps the rest of the national schedule as currently encoded
and compares one programme against its absence from the first prospective
year:

* ``maternal``: comparator has no maternal vaccination from ``start_year``;
  arms vaccinate a share ``c`` of pregnancies (births) with the declared
  effective share of protected newborns.  For countries with an existing
  programme this is the "continue versus stop" question; elsewhere it is an
  introduction.
* ``adolescent``: comparator has no Tdap event at the target cohort from
  ``start_year``; arms dose a share ``c`` of eligible children in each
  cohort reaching the target age.

Doses are counted as the vaccinated persons implied by each arm relative to
its comparator: ``c`` times births for maternal arms, and the kernel's
scheduled Tdap receipts at the target cohort for adolescent arms.  The
marginal fixed-budget design (``receipts.py``) is unchanged and remains
available as a budget-impact sensitivity.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from ..model.simulate import Simulator
from ..model.tables import DoseEvents
from ..states import TDAP_BOOSTER
from .health_outcomes import annual_age_onsets

COVERAGES = (0.5, 0.7, 0.9)


def arm_id(programme, coverage):
    return f'{programme}_c{int(round(coverage * 100)):02d}'


def _target_event(inputs, age, year):
    """(day, bin) of the adolescent cohort event in a calendar year."""
    k = int(np.floor(age + .5))
    day = inputs.calendar.year_start_day(year) + int(round((age + .5 - k) * 365.25))
    return day, inputs.grid.cohort_bin(k)


def _without_adolescent(inputs, age, start_day):
    """Drop prospective Tdap events at the target cohort; history is kept."""
    ev = inputs.events
    k_bin = inputs.grid.cohort_bin(int(np.floor(age + .5)))
    drop = (np.asarray(ev.kind) == TDAP_BOOSTER) & (np.asarray(ev.bin) == k_bin) & (np.asarray(ev.day) >= start_day)
    keep = ~drop
    return DoseEvents(np.asarray(ev.day)[keep], np.asarray(ev.bin)[keep],
                      np.asarray(ev.kind)[keep], np.asarray(ev.fraction, float)[keep])


def _with_adolescent(events, inputs, age, coverage, start_year, end_year):
    days, bins = list(events.day), list(events.bin)
    kinds, fracs = list(events.kind), list(np.asarray(events.fraction, float))
    for year in range(start_year, end_year):
        day, b = _target_event(inputs, age, year)
        if day < inputs.calendar.n_days:
            days.append(day); bins.append(b); kinds.append(TDAP_BOOSTER); fracs.append(float(coverage))
    return DoseEvents(np.asarray(days, np.int64), np.asarray(bins, np.int64),
                      np.asarray(kinds, np.int64), np.asarray(fracs, float))


def run_programme_scenarios(inputs, params, country, *, programme, history_state, history_acc,
                            start_year, end_year, adolescent_age=12., maternal_effective_share=.9,
                            coverages=COVERAGES):
    """Return (annual_age_onsets, receipts) for comparator and coverage arms."""
    if programme not in ('maternal', 'adolescent'):
        raise ValueError('programme must be maternal or adolescent')
    if not 0 <= maternal_effective_share <= 1 or any(not 0 < c <= 1 for c in coverages):
        raise ValueError('coverage and effective share must lie in (0, 1]')
    cal = inputs.calendar
    first, last = cal.year_start_day(start_year), cal.year_start_day(end_year)
    years = range(start_year, end_year)
    record = np.array([cal.year_start_day(y) for y in range(start_year, end_year + 1)])

    if programme == 'maternal':
        base_mat = inputs.maternal_effective.copy()
        base_mat[first:] = 0.
        comparator = replace(inputs, maternal_effective=base_mat)
    else:
        comparator = replace(inputs, events=_without_adolescent(inputs, adolescent_age, first))

    def simulate(scenario):
        return Simulator(scenario).run(params, history_state, history_acc, first, last, record_days=record)

    results = {'baseline': (simulate(comparator), comparator)}
    for c in coverages:
        if programme == 'maternal':
            mat = comparator.maternal_effective.copy()
            mat[first:last] = c * maternal_effective_share
            scenario = replace(comparator, maternal_effective=mat)
        else:
            scenario = replace(comparator, events=_with_adolescent(
                comparator.events, inputs, adolescent_age, c, start_year, end_year))
        results[arm_id(programme, c)] = (simulate(scenario), scenario)

    onsets, receipts = [], []
    base_ledger = results['baseline'][0].receipt_ledger
    for sid, (result, scenario) in results.items():
        onsets.append(annual_age_onsets(result, inputs.grid, cal).assign(profile=country, strategy_id=sid))
        if sid == 'baseline':
            continue
        c = float(sid.rsplit('_c', 1)[1]) / 100
        for year in years:
            yi = year - cal.first_year
            if programme == 'maternal':
                births = inputs.births_per_day[cal.year_start_day(year):cal.year_start_day(year + 1)].sum()
                doses = c * births
            else:
                doses = (result.receipt_ledger[yi, TDAP_BOOSTER, 0] - base_ledger[yi, TDAP_BOOSTER, 0])
            receipts.append(dict(year=year, profile=country, strategy_id=sid, programme=programme,
                                 coverage=c, additional_receipts=float(max(doses, 0.)),
                                 adolescent_target_age=adolescent_age if programme == 'adolescent' else np.nan))
    return pd.concat(onsets, ignore_index=True), pd.DataFrame(receipts)
