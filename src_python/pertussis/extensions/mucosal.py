"""Independent mucosal acquisition protection coadministered with scheduled Tdap.

The original 39 compartments are replicated across three mucosal layers: no
protection, Erlang stage 1, and Erlang stage 2. Each layer retains every original
immune state and clock. The original continuous kernel supplies progression,
recovery, vaccine/natural waning and adult ageing; shared infection pressure,
population relaxation and births are added once over the combined population.
Mucosal protection persists on its own clock through breakthrough infection.

Only selected Tdap recipients after ``start_day`` enter mucosal stage 1. All
eligible recipients, including naturally immune recipients, receive both doses.
No change to diphtheria/tetanus provision is represented. Efficacy multiplies
acquisition hazards by (1-efficacy), with the original typical/mild split,
infectiousness and durations. It is a scenario parameter, not clinical VE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from ..model import kernel as K
from ..model.simulate import SimulationResult, Simulator
from ..states import (NC, NA, N_DOSE_KINDS, TDAP_BOOSTER, CHILD_BOOSTER,
                      D1, D2, PW1, PW2, PA1, PA2, T1, T2, SW, SA, R, W,
                      TW1, M, S0, ET, EM, SUSCEPTIBLE_STATES)

_SUSCEPTIBLE = np.asarray(SUSCEPTIBLE_STATES, dtype=np.int64)
_TDAP_ELIGIBLE = np.asarray((D1, D2, PW1, PW2, PA1, PA2, T1, T2, SW, SA, R, W)
                            + tuple(range(TW1, NC)), dtype=np.int64)


@dataclass
class MucosalResult(SimulationResult):
    nasal_receipt_ledger: np.ndarray | None = None  # year, actual nasal receipts
    nasal_event_receipts: np.ndarray | None = None  # sorted ModelInputs events
    mucosal_y: np.ndarray | None = None  # layer, bin, original compartment
    mucosal_snapshots: np.ndarray | None = None  # snapshot, layer, bin, compartment


@njit(cache=True)
def _joint_dose(y, b, h, product):
    """Partition actual Tdap recipients, update Tdap, then renew their nasal clock."""
    vaccinated = np.zeros_like(y)
    receipts = 0.
    for layer in range(3):
        for j in range(_TDAP_ELIGIBLE.size):
            c = _TDAP_ELIGIBLE[j]
            amount = h*y[layer, b, c]
            vaccinated[layer, b, c] = amount
            y[layer, b, c] -= amount
            receipts += amount
        K.apply_dose(vaccinated[layer], b, TDAP_BOOSTER, 1., product)
    for c in range(NC):
        for layer in range(3):
            y[1, b, c] += vaccinated[layer, b, c]
    return receipts


@njit(cache=True)
def _rhs(y, dy, dacc, offset, day, P, P_no_force, sig, pityp, t,
         zero_births, zero_nudge, zero_contacts, efficacy, nasal_rate):
    (births, mat_eff, npi_r, imp_mult, season_on, year_frac, month_idx, year_idx,
     flags, nudge_target, primary_product, infant_dose, c_home, c_non, group,
     age_rate, contact_mult, ev_day, ev_bin, ev_kind, ev_frac) = t
    nb = y.shape[1]
    ng = c_home.shape[1]
    total = y[0]+y[1]+y[2]
    common_lambda = np.zeros(nb)
    infg, popg, lamg = np.zeros(ng), np.zeros(ng), np.zeros(ng)
    K.force_of_infection(total, offset, day, P, npi_r, imp_mult, season_on,
                         year_frac, year_idx, c_home, c_non, group, contact_mult,
                         common_lambda, infg, popg, lamg)
    unused_lambda = np.zeros(nb)
    layer_acc = np.zeros((nb, NA))
    dacc[:] = 0.
    for layer in range(3):
        # Zero force also disables natural boosting in the original rhs;
        # both are restored below using the common population's force.
        K.rhs(y[layer], dy[layer], layer_acc, offset, day, P_no_force, sig, pityp,
              zero_births, mat_eff, npi_r, imp_mult, season_on, year_frac,
              month_idx, year_idx, zero_nudge, primary_product,
              zero_contacts, zero_contacts, group, age_rate, contact_mult,
              unused_lambda, infg, popg, lamg)
        dacc += layer_acc
        protection = 1. if layer == 0 else 1.-efficacy
        for b in range(nb):
            typical, mild = 0., 0.
            for j in range(_SUSCEPTIBLE.size):
                c = _SUSCEPTIBLE[j]
                flow = common_lambda[b]*sig[c]*protection*y[layer, b, c]
                dy[layer, b, c] -= flow
                typical += flow*pityp[c]
                mild += flow*(1.-pityp[c])
            dy[layer, b, ET] += typical
            dy[layer, b, EM] += mild
            dacc[b, 2] += typical+mild
            boosted = P[K.P_KAPPA]*common_lambda[b]*y[layer, b, W]
            dy[layer, b, W] -= boosted
            dy[layer, b, R] += boosted
            target = nudge_target[month_idx[day], b]
            number = total[b].sum()
            if target > 0. and number > 0.:
                rate = P[K.P_NUDGE]*np.log(target/number)
                for c in range(NC):
                    dy[layer, b, c] += rate*y[layer, b, c]
    for b in range(nb):
        for c in range(NC):
            first = nasal_rate*y[1, b, c]
            second = nasal_rate*y[2, b, c]
            dy[1, b, c] -= first
            dy[2, b, c] += first-second
            dy[0, b, c] += second
    dy[0, 0, M] += births[day]*mat_eff[day]
    dy[0, 0, S0] += births[day]*(1.-mat_eff[day])


@njit(cache=True)
def _integrate(y, acc, day0, day1, P, sig, pityp, t,
               n_monthly, first_cohort, last_cohort, rec, acc_out, snp, snapshots,
               receipts, nasal_receipts, nasal_events, efficacy, nasal_rate,
               start_day, event_mask):
    (births, mat_eff, npi_r, imp_mult, season_on, year_frac, month_idx, year_idx,
     flags, nudge_target, primary_product, infant_dose, c_home, c_non, group,
     age_rate, contact_mult, ev_day, ev_bin, ev_kind, ev_frac) = t
    P_no_force = P.copy()
    P_no_force[K.P_IOTA] = 0.
    zero_births = np.zeros_like(births)
    zero_nudge = np.zeros_like(nudge_target)
    zero_contacts = np.zeros_like(c_home)
    nb = y.shape[1]
    ks = np.zeros((4, 3, nb, NC))
    aa = np.zeros((4, nb, NA))
    tmp = np.zeros_like(y)
    rp, sp, ep = 0, 0, 0
    while ep < ev_day.size and ev_day[ep] < day0:
        ep += 1
    for day in range(day0, day1):
        while rp < rec.size and rec[rp] == day:
            acc_out[rp] = acc
            rp += 1
        while sp < snp.size and snp[sp] == day:
            snapshots[sp] = y
            sp += 1
        if flags[day] & K.FLAG_YEAR_START:
            for layer in range(3):
                K._annual_shift(y[layer], first_cohort, last_cohort)
        if flags[day] & K.FLAG_MONTH_START:
            for layer in range(3):
                K._monthly_shift(y[layer], n_monthly, first_cohort)
            mi = month_idx[day]
            for b in range(n_monthly):
                for kk in range(infant_dose.shape[2]-1, -1, -1):
                    h = infant_dose[mi, b, kk]
                    if h > 0.:
                        kind = kk if kk < 3 else CHILD_BOOSTER
                        for layer in range(3):
                            receipts[year_idx[day], kind, 0] += h*K.dose_eligible(y[layer], b, kind)
                            K.apply_dose(y[layer], b, kind, h, primary_product[mi])
        while ep < ev_day.size and ev_day[ep] == day:
            b, kind, h = ev_bin[ep], ev_kind[ep], ev_frac[ep]
            amount = 0.
            for layer in range(3):
                amount += h*K.dose_eligible(y[layer], b, kind)
            receipts[year_idx[day], kind, 0] += amount
            product = primary_product[month_idx[day]]
            if event_mask[ep] and day >= start_day:
                actual = _joint_dose(y, b, h, product)
                nasal_events[ep] = actual
                nasal_receipts[year_idx[day]] += actual
            else:
                for layer in range(3):
                    K.apply_dose(y[layer], b, kind, h, product)
            ep += 1
        _rhs(y, ks[0], aa[0], 0., day, P, P_no_force, sig, pityp, t,
             zero_births, zero_nudge, zero_contacts, efficacy, nasal_rate)
        for stage in range(1, 4):
            step = .5 if stage < 3 else 1.
            tmp[:] = y+step*ks[stage-1]
            _rhs(tmp, ks[stage], aa[stage], step, day, P, P_no_force, sig, pityp, t,
                 zero_births, zero_nudge, zero_contacts, efficacy, nasal_rate)
        y[:] = y+(ks[0]+2.*ks[1]+2.*ks[2]+ks[3])/6.
        for layer in range(3):
            for b in range(nb):
                for c in range(NC):
                    if y[layer, b, c] < 0.:
                        y[layer, b, c] = 0.
        acc[:] = acc+(aa[0]+2.*aa[1]+2.*aa[2]+aa[3])/6.
    while rp < rec.size and rec[rp] == day1:
        acc_out[rp] = acc
        rp += 1
    while sp < snp.size and snp[sp] == day1:
        snapshots[sp] = y
        sp += 1


_TABLE_KEYS = ('births', 'mat_eff', 'npi_r', 'imp_mult', 'season_on', 'year_frac',
               'month_idx', 'year_idx', 'flags', 'nudge_target', 'primary_product',
               'infant_dose', 'c_home', 'c_non', 'group', 'age_rate', 'contact_mult',
               'ev_day', 'ev_bin', 'ev_kind', 'ev_frac')


class MucosalSimulator(Simulator):
    """Add-on nasal/Tdap coadministration with the adopted simulator interface.

    ``event_mask`` aligns with sorted events in ``self.tables`` and may select
    Tdap events only. ``start_day`` is an absolute model-calendar day index.
    ``mucosal_y0`` supports continuation; by default all historical people enter
    layer 0. Snapshots follow the original before-events calendar convention.
    """
    def run(self, params, y0, acc0, day0, day1, record_days=None, snap_days=None,
            *, efficacy, mean_years, start_day=None, event_mask=None, mucosal_y0=None):
        if not np.isfinite(efficacy) or not 0 <= efficacy <= 1:
            raise ValueError('efficacy must be between zero and one')
        if not np.isfinite(mean_years) or mean_years*365.25 < 2:
            raise ValueError('mean protection must be at least two days for daily RK4')
        if not 0 <= day0 <= day1 <= self.calendar.n_days:
            raise ValueError('invalid integration days')
        if int(day0) != day0 or int(day1) != day1:
            raise ValueError('integration days must be integers')
        start_day = day0 if start_day is None else start_day
        if int(start_day) != start_day or not 0 <= start_day <= self.calendar.n_days:
            raise ValueError('start_day must be a model-calendar day index')
        nb = self.grid.n_bins
        if np.shape(y0) != (nb, NC) or np.shape(acc0) != (nb, NA):
            raise ValueError('initial state does not match the adopted model')
        if not np.isfinite(y0).all() or np.min(y0) < 0 or not np.isfinite(acc0).all():
            raise ValueError('initial values must be finite and population nonnegative')
        def days(value):
            raw = np.asarray([] if value is None else value)
            if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != raw.astype(np.int64)):
                raise ValueError('record days must be a one-dimensional integer array')
            out = raw.astype(np.int64)
            if np.any(np.diff(out) < 0) or np.any((out < day0) | (out > day1)):
                raise ValueError('record days must be sorted within the run interval')
            return out
        rec, snp = days(record_days), days(snap_days)
        is_tdap = self.tables['ev_kind'] == TDAP_BOOSTER
        mask = is_tdap.copy() if event_mask is None else np.asarray(event_mask)
        if mask.dtype != np.bool_ or mask.shape != is_tdap.shape or np.any(mask & ~is_tdap):
            raise ValueError('event_mask must select sorted Tdap events only')
        layers = np.zeros((3, nb, NC))
        if mucosal_y0 is None:
            layers[0] = y0
        else:
            initial = np.asarray(mucosal_y0, dtype=float)
            if initial.shape != layers.shape or not np.isfinite(initial).all() or np.min(initial) < 0:
                raise ValueError('invalid initial mucosal layers')
            np.testing.assert_allclose(initial.sum(axis=0), y0, rtol=1e-11, atol=1e-8)
            layers[:] = initial
        active = mask & (self.tables['ev_day'] >= max(day0, start_day)) & (self.tables['ev_day'] < day1) & (self.tables['ev_frac'] > 0)
        nasal_receipts = np.zeros(self.calendar.n_years)
        nasal_events = np.zeros(mask.size)
        if not active.any() and not layers[1:].any():
            base = super().run(params, y0, acc0, day0, day1, rec, snp)
            layers[0] = base.y
            snapshots = np.zeros((snp.size, 3, nb, NC))
            snapshots[:, 0] = base.snapshots
            return MucosalResult(**base.__dict__, nasal_receipt_ledger=nasal_receipts,
                nasal_event_receipts=nasal_events, mucosal_y=layers, mucosal_snapshots=snapshots)
        acc = np.array(acc0, dtype=float, copy=True)
        records = np.zeros((rec.size, nb, NA))
        snapshots = np.zeros((snp.size, 3, nb, NC))
        receipts = np.zeros((self.calendar.n_years, N_DOSE_KINDS, 3))
        P = params.kernel_vector(self.transmission_scale(params))
        sig, pityp = params.sigma_pi()
        _integrate(layers, acc, int(day0), int(day1), P, sig, pityp,
                   tuple(self.tables[key] for key in _TABLE_KEYS), self.grid.n_monthly,
                   self.grid.first_cohort_bin, self.grid.last_cohort_bin, rec, records,
                   snp, snapshots, receipts, nasal_receipts, nasal_events, float(efficacy),
                   2./(mean_years*365.25), int(start_day), mask)
        total, total_snap = layers.sum(axis=0), snapshots.sum(axis=1)
        # Canonical aggregate output for the zero-efficacy control; the extended
        # integration remains necessary for exact selected-event dose counting
        # and the independent product clocks, and must reproduce that control.
        if efficacy == 0:
            base = super().run(params, y0, acc0, day0, day1, rec, snp)
            for rebuilt, adopted in ((total, base.y), (acc, base.acc),
                                     (records, base.acc_records), (total_snap, base.snapshots),
                                     (receipts, base.receipt_ledger)):
                np.testing.assert_allclose(rebuilt, adopted, rtol=2e-9, atol=1e-6)
            # Align the latent product labels to the canonical totals at
            # floating-point precision so zero-effect output can be resumed.
            factors = np.divide(base.y, total, out=np.ones_like(total), where=total > 0)
            layers *= factors[None]
            layers[0] += np.where(total == 0, base.y, 0.)
            factors_snap = np.divide(base.snapshots, total_snap,
                                     out=np.ones_like(total_snap), where=total_snap > 0)
            snapshots *= factors_snap[:, None]
            snapshots[:, 0] += np.where(total_snap == 0, base.snapshots, 0.)
            total, acc, records, total_snap, receipts = (base.y, base.acc, base.acc_records,
                                                        base.snapshots, base.receipt_ledger)
        return MucosalResult(total, acc, rec, records, snp, total_snap, receipts,
                             nasal_receipts, nasal_events, layers, snapshots)
