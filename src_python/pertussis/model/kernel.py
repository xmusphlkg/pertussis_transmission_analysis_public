"""The single numba integration kernel.

Continuous dynamics (infection, progression, waning, births, adult ageing and
demographic nudging) are integrated with a fixed one-day RK4 step.  Discrete
calendar events are applied at the start of a day, in this order:

1. 1 January: cohort bins shift by one year (last cohort bin joins the first
   adult bin); the first cohort bin is emptied.
2. Month start: monthly bins shift by one month; the 23-month bin joins the
   first cohort bin; infant-series dose fractions for that month are applied at
   each monthly bin's entry.
3. Cohort dose events scheduled for that day (school-age and adolescent doses,
   applied once to the whole birth cohort in one bin).

All inputs are tables indexed by day, month or year since the calendar start
(1 January of the first simulated year).  Nothing in this module reads files or
configuration.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..states import (
    CHILD_BOOSTER,
    D1,
    D2,
    DOSE1,
    DOSE2,
    DOSE3,
    EM,
    ET,
    IM,
    IT,
    LATE_PRIMARY,
    MV1,
    MV2,
    MATERNAL_STAGES,
    NA,
    NC, TW1, TW2, JOINT_PROTECTION, PRIMARY_PROTECTION, TDAP_PROTECTION,
    PA1,
    PA2,
    PRODUCT_WP,
    PW1,
    PW2,
    SA,
    SUSCEPTIBLE_STATES,
    SW,
    T1,
    T2,
    TDAP_BOOSTER,
    M,
    R,
    S0,
    W,
)

# Kernel parameter vector layout (all rates per day).
P_Q = 0            # transmission probability scale per contact
P_SEAS_AMP = 1     # cosine seasonal amplitude
P_SEAS_PEAK = 2    # seasonal peak as a fraction of the calendar year
P_NPI_EPS = 3      # scale on the non-home contact reduction timeline
P_IOTA = 4         # importation hazard per capita per day
P_LATENT = 5       # 1 / latent period
P_REC_T = 6        # 1 / infectious period, typical
P_REC_M = 7        # 1 / infectious period, mild
P_RHO_M = 8        # relative infectiousness of mild infection
P_OMEGA_M = 9      # 1 / mean maternal protection
P_WP_STAGE = 10    # 2 / mean wP protection (Erlang-2 stage rate)
P_AP_STAGE = 11    # 2 / mean aP protection
P_T_STAGE = 12     # stages / mean Tdap protection in either priming group
P_NAT_STAGE = 13   # 2 / mean natural immunity (R -> W -> Sw)
P_KAPPA = 14       # boosting of W per unit force of infection
P_D1_WANE = 15     # loss of partial one-dose protection
P_D2_WANE = 16     # loss of partial two-dose protection
P_NUDGE = 17       # demographic relaxation rate towards target populations
P_VAC_SHAPE = 18   # 2 for Erlang-2, 1 for exponential vaccine protection
P_MAT_STAGES = 19
N_PARAMS = 20

FLAG_MONTH_START = 1
FLAG_YEAR_START = 2

_SUSC = np.array(SUSCEPTIBLE_STATES, dtype=np.int64)
_MAT = np.array(MATERNAL_STAGES, dtype=np.int64)
_JOINT = np.array(JOINT_PROTECTION, dtype=np.int64)
_PRIMARY = np.array(PRIMARY_PROTECTION, dtype=np.int64)
_TDAP = np.array(TDAP_PROTECTION, dtype=np.int64)
_TWO_PI = 2.0 * np.pi
_DAYS_PER_YEAR = 365.25


@njit(cache=True)
def dose_eligible(y, b, kind):
    """Recipients, including protected recipients whose state does not change.

    Active infection is deferred. Booster eligibility follows apply_dose's
    primed/recovered population; receipts are doses, not unique people.
    """
    if kind <= DOSE3:
        total = y[b, S0 if kind == DOSE1 else (D1 if kind == DOSE2 else D2)]
        for stage in range(4):
            total += y[b, _MAT[stage, kind]]
        return total
    if kind == CHILD_BOOSTER or kind == TDAP_BOOSTER:
        total = 0.0
        for s in (D1, D2, PW1, PW2, PA1, PA2, T1, T2, SW, SA, R, W):
            total += y[b, s]
        for state in range(TW1, NC):
            total += y[b, state]
        if kind == CHILD_BOOSTER:
            for stage in range(4):
                total += y[b, _MAT[stage, 1]] + y[b, _MAT[stage, 2]]
        return total
    total = y[b, S0] + y[b, D1] + y[b, D2]
    for stage in range(4):
        for tag in range(3):
            total += y[b, _MAT[stage, tag]]
    return total


@njit(cache=True)
def receipt_fraction(y, b, kind, h, requested):
    eligible = dose_eligible(y, b, kind)
    extra = min(requested, max(0.0, (1.0-h)*eligible))
    return h + (extra/eligible if eligible > 0.0 else 0.0), h*eligible, extra


@njit(cache=True)
def _move(y, b, src, dst, h):
    if src == dst:
        return
    x = h * y[b, src]
    y[b, src] -= x
    y[b, dst] += x


@njit(cache=True)
def apply_dose(y, b, kind, h, product):
    """Move fraction ``h`` of eligible states in bin ``b`` for one dose kind.

    Advancing moves are ordered from the most to the least advanced state so a
    single dose never advances anyone twice.  Boosters reach only people who
    have had at least one dose (and the naturally immune); never-vaccinated
    children (S0, M) are not boosted, so booster coverage never raises
    first-dose coverage.
    """

    p1 = PW1 if product == PRODUCT_WP else PA1
    if kind == DOSE1:
        _move(y, b, S0, D1, h)
        _move(y, b, M, MV1, h)
    elif kind == DOSE2:
        _move(y, b, D1, D2, h)
        _move(y, b, MV1, MV2, h)
    elif kind == DOSE3:
        _move(y, b, D2, p1, h)
        _move(y, b, MV2, p1, h)
    elif kind == CHILD_BOOSTER:
        for row in range(2):
            _move(y, b, _JOINT[row, 2], _JOINT[row, 0], h)
            _move(y, b, _JOINT[row, 3], _JOINT[row, 1], h)
        _move(y, b, TW2, TW1, h)
        _move(y, b, PW2, PW1, h)
        _move(y, b, SW, PW1, h)
        _move(y, b, PA2, PA1, h)
        _move(y, b, SA, PA1, h)
        _move(y, b, T2, T1, h)
        _move(y, b, D2, p1, h)
        _move(y, b, MV2, p1, h)
        _move(y, b, D1, D2, h)
        _move(y, b, MV1, MV2, h)
        _move(y, b, W, R, h)
    elif kind == TDAP_BOOSTER:
        # Renew only the Tdap clock; preserve primary stage and priming type.
        # Existing joint states must move before newly vaccinated people enter.
        for row in range(2):
            _move(y, b, _JOINT[row, 1], _JOINT[row, 0], h)
            _move(y, b, _JOINT[row, 3], _JOINT[row, 2], h)
        _move(y, b, T2, T1, h)
        _move(y, b, TW2, TW1, h)
        _move(y, b, PA1, _JOINT[0, 0], h)
        _move(y, b, PA2, _JOINT[0, 2], h)
        _move(y, b, PW1, _JOINT[1, 0], h)
        _move(y, b, PW2, _JOINT[1, 2], h)
        _move(y, b, SA, T1, h)
        _move(y, b, SW, TW1, h)
        _move(y, b, D2, p1, h)
        _move(y, b, D1, D2, h)
        _move(y, b, W, R, h)
    elif kind == LATE_PRIMARY:
        # A complete primary series given after infancy, applied as one
        # cohort event: everyone without a completed series is primed.
        _move(y, b, S0, p1, h)
        _move(y, b, D1, p1, h)
        _move(y, b, D2, p1, h)
        _move(y, b, M, p1, h)
        _move(y, b, MV1, p1, h)
        _move(y, b, MV2, p1, h)
    # Receipt advances infant-dose history without restarting passive stages.
    for stage in range(1, 4):
        m, mv1, mv2 = _MAT[stage, 0], _MAT[stage, 1], _MAT[stage, 2]
        if kind == DOSE1:
            _move(y, b, m, mv1, h)
        elif kind == DOSE2:
            _move(y, b, mv1, mv2, h)
        elif kind == DOSE3:
            _move(y, b, mv2, p1, h)
        elif kind == CHILD_BOOSTER:
            _move(y, b, mv2, p1, h)
            _move(y, b, mv1, mv2, h)
        elif kind == LATE_PRIMARY:
            _move(y, b, m, p1, h)
            _move(y, b, mv1, p1, h)
            _move(y, b, mv2, p1, h)


@njit(cache=True)
def _monthly_shift(y, n_monthly, first_cohort):
    for c in range(NC):
        y[first_cohort, c] += y[n_monthly - 1, c]
    for m in range(n_monthly - 1, 0, -1):
        for c in range(NC):
            y[m, c] = y[m - 1, c]
    for c in range(NC):
        y[0, c] = 0.0


@njit(cache=True)
def _annual_shift(y, first_cohort, last_cohort):
    first_adult = last_cohort + 1
    for c in range(NC):
        y[first_adult, c] += y[last_cohort, c]
    for k in range(last_cohort, first_cohort, -1):
        for c in range(NC):
            y[k, c] = y[k - 1, c]
    for c in range(NC):
        y[first_cohort, c] = 0.0


@njit(cache=True)
def force_of_infection(
    y, t_offset, d, P, npi_r, imp_mult, season_on, year_frac, year_idx,
    c_home, c_non, group, contact_mult, lam, infg, popg, lamg,
):
    """Fill ``lam`` (per bin, per day) for state ``y`` at day ``d + t_offset``."""

    nb = y.shape[0]
    ng = infg.shape[0]
    for g in range(ng):
        infg[g] = 0.0
        popg[g] = 0.0
    rho_m = P[P_RHO_M]
    for b in range(nb):
        n = 0.0
        for c in range(NC):
            n += y[b, c]
        g = group[b]
        popg[g] += n
        infg[g] += y[b, IT] + rho_m * y[b, IM]
    for g in range(ng):
        if popg[g] > 0.0:
            infg[g] = infg[g] / popg[g]
        else:
            infg[g] = 0.0
    season = 1.0 + P[P_SEAS_AMP] * season_on[d] * np.cos(
        _TWO_PI * (year_frac[d] + t_offset / _DAYS_PER_YEAR - P[P_SEAS_PEAK])
    )
    mult = 1.0 - P[P_NPI_EPS] * npi_r[d]
    if mult < 0.0:
        mult = 0.0
    yi = year_idx[d]
    scale = P[P_Q] * season
    for g in range(ng):
        acc = 0.0
        for h in range(ng):
            acc += (c_home[yi, g, h] + mult * c_non[yi, g, h]) * infg[h]
        lamg[g] = scale * acc
    iota = P[P_IOTA] * imp_mult[d]
    for b in range(nb):
        lam[b] = lamg[group[b]] * contact_mult[b] + iota


@njit(cache=True)
def _overlap_waning(y, dy, P):
    """Additional generator terms only; all flows conserve population."""
    exponential = P[P_VAC_SHAPE] < 1.5
    td = P[P_T_STAGE]
    for b in range(y.shape[0]):
        x = td*y[b, TW1]
        dy[b, TW1] -= x; dy[b, SW if exponential else TW2] += x
        x = td*y[b, TW2]
        dy[b, TW2] -= x; dy[b, SW] += x
        for row in range(2):
            primary = P[P_AP_STAGE] if row == 0 else P[P_WP_STAGE]
            for ps in range(2):
                for ts in range(2):
                    source = _JOINT[row, 2*ps+ts]
                    # Loss/progression of primary protection keeps Tdap stage.
                    dest = _TDAP[row, ts] if ps == 1 or exponential else _JOINT[row, 2+ts]
                    x = primary*y[b, source]
                    dy[b, source] -= x; dy[b, dest] += x
                    # Loss/progression of Tdap keeps the original primary stage.
                    dest = _PRIMARY[row, ps] if ts == 1 or exponential else _JOINT[row, 2*ps+1]
                    x = td*y[b, source]
                    dy[b, source] -= x; dy[b, dest] += x


@njit(cache=True)
def rhs(
    y, dy, dacc, t_offset, d, P, sig, pityp,
    births, mat_eff, npi_r, imp_mult, season_on, year_frac, month_idx, year_idx,
    nudge_target, primary_product, c_home, c_non, group, age_rate, contact_mult,
    lam, infg, popg, lamg,
):
    nb = y.shape[0]
    for b in range(nb):
        for c in range(NC):
            dy[b, c] = 0.0
        for a in range(NA):
            dacc[b, a] = 0.0
    force_of_infection(
        y, t_offset, d, P, npi_r, imp_mult, season_on, year_frac, year_idx,
        c_home, c_non, group, contact_mult, lam, infg, popg, lamg,
    )
    lat = P[P_LATENT]
    rec_t = P[P_REC_T]
    rec_m = P[P_REC_M]
    om = P[P_OMEGA_M]
    wps = P[P_WP_STAGE]
    aps = P[P_AP_STAGE]
    ts = P[P_T_STAGE]
    nat = P[P_NAT_STAGE]
    kappa = P[P_KAPPA]
    d1w = P[P_D1_WANE]
    d2w = P[P_D2_WANE]
    nudge = P[P_NUDGE]
    mi = month_idx[d]
    d2_target = SW if primary_product[mi] == PRODUCT_WP else SA
    for b in range(nb):
        lb = lam[b]
        in_t = 0.0
        in_m = 0.0
        for i in range(_SUSC.size):
            s = _SUSC[i]
            x = lb * sig[s] * y[b, s]
            dy[b, s] -= x
            in_t += x * pityp[s]
            in_m += x * (1.0 - pityp[s])
        boost = kappa * lb * y[b, W]
        dy[b, W] -= boost
        dy[b, R] += boost
        on_t = lat * y[b, ET]
        on_m = lat * y[b, EM]
        dy[b, ET] += in_t - on_t
        dy[b, EM] += in_m - on_m
        dy[b, IT] += on_t - rec_t * y[b, IT]
        dy[b, IM] += on_m - rec_m * y[b, IM]
        dy[b, R] += rec_t * y[b, IT] + rec_m * y[b, IM]
        dacc[b, 0] = on_t
        dacc[b, 1] = on_m
        dacc[b, 2] = in_t + in_m
        # Waning.
        mat_stages = int(P[P_MAT_STAGES])
        for stage in range(mat_stages):
            for tag in range(3):
                src = _MAT[stage, tag]
                dst = (_MAT[stage+1, tag] if stage+1 < mat_stages
                       else (S0 if tag == 0 else (D1 if tag == 1 else D2)))
                x = om * mat_stages * y[b, src]
                dy[b, src] -= x
                dy[b, dst] += x
        # Vaccine protection is Erlang-2 by default; the exponential
        # alternative (structural comparison) leaves stage 1 directly, with
        # the stage rate set to 1/mean so the mean duration is unchanged.
        exponential = P[P_VAC_SHAPE] < 1.5
        x = wps * y[b, PW1]
        dy[b, PW1] -= x
        dy[b, SW if exponential else PW2] += x
        x = wps * y[b, PW2]
        dy[b, PW2] -= x
        dy[b, SW] += x
        x = aps * y[b, PA1]
        dy[b, PA1] -= x
        dy[b, SA if exponential else PA2] += x
        x = aps * y[b, PA2]
        dy[b, PA2] -= x
        dy[b, SA] += x
        x = ts * y[b, T1]
        dy[b, T1] -= x
        dy[b, SA if exponential else T2] += x
        x = ts * y[b, T2]
        dy[b, T2] -= x
        dy[b, SA] += x
        x = nat * y[b, R]
        dy[b, R] -= x
        dy[b, W] += x
        x = nat * y[b, W]
        dy[b, W] -= x
        dy[b, SW] += x
        x = d1w * y[b, D1]
        dy[b, D1] -= x
        dy[b, S0] += x
        x = d2w * y[b, D2]
        dy[b, D2] -= x
        dy[b, d2_target] += x
        # Demographic relaxation towards the target population, applied in
        # proportion to every compartment so it never changes immune shares.
        target = nudge_target[mi, b]
        if target > 0.0:
            n = 0.0
            for c in range(NC):
                n += y[b, c]
            if n > 0.0:
                r = nudge * np.log(target / n)
                for c in range(NC):
                    dy[b, c] += r * y[b, c]
        a = age_rate[b]
        if a > 0.0:
            for c in range(NC):
                x = a * y[b, c]
                dy[b, c] -= x
                dy[b + 1, c] += x
    _overlap_waning(y, dy, P)
    bth = births[d]
    dy[0, M] += bth * mat_eff[d]
    dy[0, S0] += bth * (1.0 - mat_eff[d])


@njit(cache=True)
def integrate(
    y, acc, day0, day1, P, sig, pityp,
    births, mat_eff, npi_r, imp_mult, season_on, year_frac, month_idx, year_idx, flags,
    nudge_target, primary_product, infant_dose, c_home, c_non,
    group, age_rate, contact_mult, n_monthly, first_cohort, last_cohort,
    ev_day, ev_bin, ev_kind, ev_frac,
    record_days, acc_out, snap_days, snap_out,
    infant_extra, event_extra, receipt_ledger,
):
    """Integrate from the start of ``day0`` to the start of ``day1`` in place.

    ``acc_out[i]`` receives the accumulators at the start of ``record_days[i]``
    and ``snap_out[i]`` the state at the start of ``snap_days[i]`` (before that
    day's events).  Days outside ``[day0, day1]`` are ignored.
    """

    nb = y.shape[0]
    ng = c_home.shape[1]
    lam = np.zeros(nb)
    infg = np.zeros(ng)
    popg = np.zeros(ng)
    lamg = np.zeros(ng)
    k1 = np.zeros((nb, NC))
    k2 = np.zeros((nb, NC))
    k3 = np.zeros((nb, NC))
    k4 = np.zeros((nb, NC))
    a1 = np.zeros((nb, NA))
    a2 = np.zeros((nb, NA))
    a3 = np.zeros((nb, NA))
    a4 = np.zeros((nb, NA))
    tmp = np.zeros((nb, NC))
    rp = 0
    while rp < record_days.size and record_days[rp] < day0:
        rp += 1
    sp = 0
    while sp < snap_days.size and snap_days[sp] < day0:
        sp += 1
    ep = 0
    while ep < ev_day.size and ev_day[ep] < day0:
        ep += 1
    n_inf_kinds = infant_dose.shape[2]
    for d in range(day0, day1):
        while rp < record_days.size and record_days[rp] == d:
            for b in range(nb):
                for a in range(NA):
                    acc_out[rp, b, a] = acc[b, a]
            rp += 1
        while sp < snap_days.size and snap_days[sp] == d:
            for b in range(nb):
                for c in range(NC):
                    snap_out[sp, b, c] = y[b, c]
            sp += 1
        fl = flags[d]
        if fl & FLAG_YEAR_START:
            _annual_shift(y, first_cohort, last_cohort)
        if fl & FLAG_MONTH_START:
            _monthly_shift(y, n_monthly, first_cohort)
            mi = month_idx[d]
            prod = primary_product[mi]
            for m in range(n_monthly):
                for kk in range(n_inf_kinds - 1, -1, -1):
                    h = infant_dose[mi, m, kk]
                    requested = infant_extra[mi, m, kk]
                    if h > 0.0 or requested > 0.0:
                        kind = kk if kk < 3 else CHILD_BOOSTER
                        h, scheduled, extra = receipt_fraction(y, m, kind, h, requested)
                        receipt_ledger[year_idx[d], kind, 0] += scheduled
                        receipt_ledger[year_idx[d], kind, 1] += extra
                        receipt_ledger[year_idx[d], kind, 2] += requested-extra
                        apply_dose(y, m, kind, h, prod)
        while ep < ev_day.size and ev_day[ep] == d:
            kind = ev_kind[ep]
            h, scheduled, extra = receipt_fraction(y, ev_bin[ep], kind, ev_frac[ep], event_extra[ep])
            receipt_ledger[year_idx[d], kind, 0] += scheduled
            receipt_ledger[year_idx[d], kind, 1] += extra
            receipt_ledger[year_idx[d], kind, 2] += event_extra[ep]-extra
            apply_dose(y, ev_bin[ep], kind, h, primary_product[month_idx[d]])
            ep += 1
        # RK4 over one day.
        rhs(y, k1, a1, 0.0, d, P, sig, pityp, births, mat_eff, npi_r, imp_mult, season_on,
            year_frac, month_idx, year_idx, nudge_target, primary_product, c_home, c_non,
            group, age_rate, contact_mult, lam, infg, popg, lamg)
        for b in range(nb):
            for c in range(NC):
                tmp[b, c] = y[b, c] + 0.5 * k1[b, c]
        rhs(tmp, k2, a2, 0.5, d, P, sig, pityp, births, mat_eff, npi_r, imp_mult, season_on,
            year_frac, month_idx, year_idx, nudge_target, primary_product, c_home, c_non,
            group, age_rate, contact_mult, lam, infg, popg, lamg)
        for b in range(nb):
            for c in range(NC):
                tmp[b, c] = y[b, c] + 0.5 * k2[b, c]
        rhs(tmp, k3, a3, 0.5, d, P, sig, pityp, births, mat_eff, npi_r, imp_mult, season_on,
            year_frac, month_idx, year_idx, nudge_target, primary_product, c_home, c_non,
            group, age_rate, contact_mult, lam, infg, popg, lamg)
        for b in range(nb):
            for c in range(NC):
                tmp[b, c] = y[b, c] + k3[b, c]
        rhs(tmp, k4, a4, 1.0, d, P, sig, pityp, births, mat_eff, npi_r, imp_mult, season_on,
            year_frac, month_idx, year_idx, nudge_target, primary_product, c_home, c_non,
            group, age_rate, contact_mult, lam, infg, popg, lamg)
        for b in range(nb):
            for c in range(NC):
                v = y[b, c] + (k1[b, c] + 2.0 * k2[b, c] + 2.0 * k3[b, c] + k4[b, c]) / 6.0
                y[b, c] = v if v > 0.0 else 0.0
            for a in range(NA):
                acc[b, a] += (a1[b, a] + 2.0 * a2[b, a] + 2.0 * a3[b, a] + a4[b, a]) / 6.0
    while rp < record_days.size and record_days[rp] == day1:
        for b in range(nb):
            for a in range(NA):
                acc_out[rp, b, a] = acc[b, a]
        rp += 1
    while sp < snap_days.size and snap_days[sp] == day1:
        for b in range(nb):
            for c in range(NC):
                snap_out[sp, b, c] = y[b, c]
        sp += 1
