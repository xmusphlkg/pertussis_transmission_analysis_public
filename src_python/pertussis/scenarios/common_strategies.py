"""Common strategy set for the unified paper: 00, M0, 0A, MA and the current programme.

Every strategy starts from the same saved history (state and accumulators at
``start_year``) and differs only from that date on:

* ``00``      no maternal vaccination and no target-cohort Tdap;
* ``M0``      maternal vaccination at ``coverage``, no Tdap;
* ``0A``      Tdap at ``coverage`` in the target cohort, no maternal vaccination;
* ``MA``      both, simulated jointly so the transmission interaction is kept;
* ``current`` the encoded national schedule continued unchanged.

The rest of the national schedule (infant series, child boosters) is untouched.
Doses are reported per programme so that each programme is costed with its own
price and delivery cost.  Maternal doses are the vaccinated pregnancies, i.e.
``coverage`` times births (for ``current``, the encoded effective share divided
by the declared effective share, times births).  Adolescent doses are the
kernel's scheduled Tdap receipts at the target cohort minus those of ``00``.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from ..model.simulate import Simulator
from ..states import TDAP_BOOSTER
from .health_outcomes import annual_age_onsets
from .programme_scenarios import _with_adolescent, _without_adolescent

STRATEGY_IDS = ('00', 'M0', '0A', 'MA', 'current')
USES = {'00': (False, False), 'M0': (True, False), '0A': (False, True), 'MA': (True, True)}


def common_scenarios(inputs, *, start_year, end_year, adolescent_age, maternal_effective_share,
                     coverage):
    """Scenario ``ModelInputs`` for every strategy in the common set."""
    if not 0 < coverage <= 1 or not 0 < maternal_effective_share <= 1:
        raise ValueError('coverage and effective share must lie in (0, 1]')
    cal = inputs.calendar
    first, last = cal.year_start_day(start_year), cal.year_start_day(end_year)
    no_mat = inputs.maternal_effective.copy()
    no_mat[first:] = 0.
    no_tdap = _without_adolescent(inputs, adolescent_age, first)
    base = replace(inputs, maternal_effective=no_mat, events=no_tdap)
    out = {'current': inputs, '00': base}
    mat = no_mat.copy()
    mat[first:last] = coverage * maternal_effective_share
    out['M0'] = replace(base, maternal_effective=mat)
    adol = _with_adolescent(no_tdap, inputs, adolescent_age, coverage, start_year, end_year)
    out['0A'] = replace(base, events=adol)
    out['MA'] = replace(base, maternal_effective=mat, events=adol)
    return {k: out[k] for k in STRATEGY_IDS}


def _maternal_doses(strategy, scenario, inputs, year, coverage, maternal_effective_share):
    cal = inputs.calendar
    a, b = cal.year_start_day(year), cal.year_start_day(year + 1)
    births = inputs.births_per_day[a:b]
    if strategy == 'current':
        return float((inputs.maternal_effective[a:b] / maternal_effective_share * births).sum())
    return float(coverage * births.sum()) if USES[strategy][0] else 0.


def run_common_strategies(inputs, params, country, *, history_state, history_acc, start_year,
                          end_year, coverage=0.7, adolescent_age=12., maternal_effective_share=.9,
                          simulator_class=Simulator):
    """Return (annual age onsets, per-programme receipts) for the five strategies."""
    scenarios = common_scenarios(inputs, start_year=start_year, end_year=end_year,
                                 adolescent_age=adolescent_age,
                                 maternal_effective_share=maternal_effective_share,
                                 coverage=coverage)
    cal = inputs.calendar
    first, last = cal.year_start_day(start_year), cal.year_start_day(end_year)
    record = np.array([cal.year_start_day(y) for y in range(start_year, end_year + 1)])
    def simulate(sc):
        return simulator_class(sc).run(params, history_state, history_acc, first, last, record_days=record)

    results = {sid: simulate(sc) for sid, sc in scenarios.items()}
    # Adolescent doses are the receipts the target-cohort Tdap adds, measured against
    # the same maternal state without that programme.  The kernel defers dosing of
    # actively infected people, so receipts depend slightly on transmission and an
    # unpaired comparator would book that difference as doses.
    no_adolescent = {'00': results['00'], 'M0': results['M0'],
                     '0A': results['00'], 'MA': results['M0'],
                     'current': simulate(replace(
                         inputs, events=_without_adolescent(inputs, adolescent_age, first)))}
    onsets, receipts = [], []
    for sid, result in results.items():
        onsets.append(annual_age_onsets(result, inputs.grid, cal).assign(profile=country, strategy_id=sid))
        for year in range(start_year, end_year):
            yi = year - cal.first_year
            mat = _maternal_doses(sid, scenarios[sid], inputs, year, coverage, maternal_effective_share)
            adol = float(result.receipt_ledger[yi, TDAP_BOOSTER, 0]
                         - no_adolescent[sid].receipt_ledger[yi, TDAP_BOOSTER, 0])
            if adol < -1e-6:
                raise RuntimeError(f'negative adolescent doses for {sid} in {year}: {adol}')
            adol = max(adol, 0.)
            for programme, doses in (('maternal', mat), ('adolescent', adol)):
                receipts.append(dict(year=year, profile=country, strategy_id=sid, programme=programme,
                                     coverage=coverage, doses=doses))
    return pd.concat(onsets, ignore_index=True), pd.DataFrame(receipts)
