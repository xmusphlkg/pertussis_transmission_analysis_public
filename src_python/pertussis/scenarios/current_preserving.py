"""Independent coverage expansions that retain every existing programme.

Coverage targets are absolute fractions of eligible recipients, not additional
percentage points. None continues the original time-varying programme. An
expansion takes the pointwise maximum with existing coverage. Existing events
at other ages or dates remain untouched, including other Tdap events.
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from ..model.tables import DoseEvents
from ..states import TDAP_BOOSTER
from .programme_scenarios import _target_event


def target_coverage(inputs, start_year, end_year, adolescent_age):
    """Existing fraction at the exact target event; preserve all other events."""
    ev = inputs.events
    rows = []
    for year in range(start_year, end_year):
        day, b = _target_event(inputs, adolescent_age, year)
        selected = (ev.day == day) & (ev.bin == b) & (ev.kind == TDAP_BOOSTER)
        if selected.sum() > 1:
            raise ValueError('Ambiguous duplicate target Tdap events')
        rows.append(float(ev.fraction[selected][0]) if selected.any() else 0.)
    return np.asarray(rows)


def expand_programmes(inputs, *, start_year, end_year, adolescent_age,
                      maternal_effective_share=.9, maternal_target=None, adolescent_target=None):
    if not 0 < maternal_effective_share <= 1:
        raise ValueError('Invalid maternal effective share')
    for target in (maternal_target, adolescent_target):
        if target is not None and (not np.isfinite(target) or not 0 <= target <= 1):
            raise ValueError('Coverage targets must be fractions in [0,1] or None')
    first, last = inputs.calendar.year_start_day(start_year), inputs.calendar.year_start_day(end_year)
    if first >= last or first < 0 or last > inputs.calendar.n_days:
        raise ValueError('Invalid projection interval')
    maternal = inputs.maternal_effective.copy()
    if (maternal[first:last] > maternal_effective_share + 1e-12).any():
        raise ValueError('Encoded maternal coverage exceeds one under the specified effective share')
    if maternal_target is not None:
        maternal[first:last] = np.maximum(maternal[first:last], maternal_target*maternal_effective_share)
    ev = inputs.events
    days, bins, kinds, fractions = [list(np.asarray(x)) for x in (ev.day, ev.bin, ev.kind, ev.fraction)]
    target_coverage(inputs, start_year, end_year, adolescent_age)
    if adolescent_target is not None:
        for year in range(start_year, end_year):
            day, b = _target_event(inputs, adolescent_age, year)
            matches = np.flatnonzero((ev.day == day) & (ev.bin == b) & (ev.kind == TDAP_BOOSTER))
            if len(matches):
                j = int(matches[0]); fractions[j] = max(fractions[j], adolescent_target)
            elif adolescent_target > 0:
                days.append(day); bins.append(b); kinds.append(TDAP_BOOSTER); fractions.append(adolescent_target)
    events = DoseEvents(np.asarray(days, np.int64), np.asarray(bins, np.int64),
                        np.asarray(kinds, np.int64), np.asarray(fractions, float))
    return replace(inputs, maternal_effective=maternal, events=events)


def coverage_grid(inputs, *, start_year, end_year, adolescent_age,
                  maternal_effective_share=.9, targets=(.5, .7, .9), alias_target=.7):
    """Unique feasible grid plus four named aliases at a stated expansion target.

The grid includes unchanged coverage and targets above its maximum. The saved
eight countries have constant prospective coverage; varying schedules require
additional partially expanding targets rather than this constant-floor grid.
"""
    first, last = inputs.calendar.year_start_day(start_year), inputs.calendar.year_start_day(end_year)
    m = inputs.maternal_effective[first:last]/maternal_effective_share
    a = target_coverage(inputs, start_year, end_year, adolescent_age)
    if not np.allclose(m, m[0], rtol=0, atol=1e-12) or not np.allclose(a, a[0], rtol=0, atol=1e-12):
        raise ValueError('This grid requires constant prospective baseline coverages')
    if not np.isfinite(alias_target) or not 0 <= alias_target <= 1:
        raise ValueError('Invalid alias target')
    targets = sorted(set(float(x) for x in (*targets, alias_target)))
    if any(not np.isfinite(x) or not 0 <= x <= 1 for x in targets):
        raise ValueError('Invalid coverage grid')
    ms = [None]+[x for x in targets if x > m[0]+1e-12]
    aa = [None]+[x for x in targets if x > a[0]+1e-12]
    scenarios, rows, ids = {}, [], {}
    for mt, at in product(ms, aa):
        sid = 'current' if mt is None and at is None else 'current_M'+('keep' if mt is None else f'{mt:.6g}')+'_A'+('keep' if at is None else f'{at:.6g}')
        scenarios[sid] = inputs if sid == 'current' else expand_programmes(inputs,
            start_year=start_year, end_year=end_year, adolescent_age=adolescent_age,
            maternal_effective_share=maternal_effective_share, maternal_target=mt, adolescent_target=at)
        ids[(mt, at)] = sid
        rows.append(dict(strategy_id=sid, maternal_target=mt, adolescent_target=at,
            maternal_coverage=float(m[0]) if mt is None else mt,
            adolescent_coverage=float(a[0]) if at is None else at,
            original_maternal_coverage=float(m[0]), original_adolescent_coverage=float(a[0]),
            adolescent_target_age=adolescent_age))
    aliases = []
    for label, has_m, has_a in [('current',False,False),('current+M',True,False),('current+A',False,True),('current+MA',True,True)]:
        mt = alias_target if has_m and alias_target > m[0]+1e-12 else None
        at = alias_target if has_a and alias_target > a[0]+1e-12 else None
        aliases.append(dict(alias=label, requested_expansion_target=alias_target, strategy_id=ids[(mt, at)],
                            actual_maternal_coverage=float(m[0]) if mt is None else mt,
                            actual_adolescent_coverage=float(a[0]) if at is None else at))
    return scenarios, pd.DataFrame(rows), pd.DataFrame(aliases)
