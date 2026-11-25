"""Run the kernel on a ``ModelInputs`` calendar.

The calendar's leading years are the spin-up (constant inputs, seasonality
off; the builder decides how many).  ``Simulator.run`` integrates any day
range from a given state, so the same object serves spin-up + history, and
scenario projections from a saved state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import KIND_COHORT, KIND_MONTHLY
from ..states import NA, NC, N_DOSE_KINDS, IT, R, S0, SW, W
from .kernel import integrate
from .ngm import q_from_r0
from .parameters import ModelParameters
from .tables import ModelInputs


@dataclass
class SimulationResult:
    y: np.ndarray            # state at the end day, (n_bins, NC)
    acc: np.ndarray          # cumulative accumulators at the end day, (n_bins, NA)
    record_days: np.ndarray
    acc_records: np.ndarray  # (n_records, n_bins, NA)
    snap_days: np.ndarray
    snapshots: np.ndarray    # (n_snapshots, n_bins, NC)
    receipt_ledger: np.ndarray | None = None  # year, dose kind, scheduled/extra/unspent

    def interval_totals(self, accumulator: int) -> np.ndarray:
        """Per-bin increments between consecutive record days, (n_records-1, n_bins)."""

        return np.diff(self.acc_records[:, :, accumulator], axis=0)


class Simulator:
    def __init__(self, inputs: ModelInputs):
        self.inputs = inputs
        self.grid = inputs.grid
        self.calendar = inputs.calendar
        self.tables = inputs.kernel_tables()
        ref = inputs.reference_year - inputs.calendar.first_year
        self.reference_contacts = inputs.contacts_home[ref] + inputs.contacts_nonhome[ref]

    def transmission_scale(self, params: ModelParameters) -> float:
        return q_from_r0(params.R0, self.reference_contacts, params.infectious_days_typical)

    def initial_state(self, seed_fraction: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        """A crude day-0 state; the spin-up removes its influence.

        Monthly bins hold one month of births; cohort bin 2 is empty on
        1 January; other bins take their day-0 target population (or one year
        of births when no target is set).  Non-infant bins start with mixed
        immunity so the spin-up begins near an endemic configuration.
        """

        grid, t = self.grid, self.tables
        births_year = float(t["births"][0]) * 365.25
        y = np.zeros((grid.n_bins, NC))
        acc = np.zeros((grid.n_bins, NA))
        for b in range(grid.n_bins):
            target = float(self.inputs.nudge_target[0, b])
            if grid.kind[b] == KIND_MONTHLY:
                pop = births_year / 12.0
            elif grid.kind[b] == KIND_COHORT and grid.cohort_age[b] == grid.cohort_first:
                pop = 0.0
            else:
                pop = target if target > 0 else births_year
            if grid.kind[b] == KIND_MONTHLY and b < 12:
                y[b, S0] = pop
            else:
                y[b, S0] = 0.1 * pop
                y[b, SW] = 0.3 * pop
                y[b, R] = 0.3 * pop
                y[b, W] = 0.3 * pop
            seed = seed_fraction * pop
            y[b, IT] += seed
            y[b, S0] -= min(seed, y[b, S0])
        return y, acc

    def run(
        self,
        params: ModelParameters,
        y0: np.ndarray,
        acc0: np.ndarray,
        day0: int,
        day1: int,
        record_days: np.ndarray | None = None,
        snap_days: np.ndarray | None = None,
        infant_extra: np.ndarray | None = None,
        event_extra: np.ndarray | None = None,
    ) -> SimulationResult:
        if not 0 <= day0 <= day1 <= self.calendar.n_days:
            raise ValueError("day range outside the calendar")
        if np.shape(y0) != (self.grid.n_bins, NC) or np.shape(acc0) != (self.grid.n_bins, NA):
            raise ValueError("initial state shape differs from the current model; rebuild cached histories")
        t = self.tables
        q = self.transmission_scale(params)
        P = params.kernel_vector(q)
        sig, pityp = params.sigma_pi()
        rec = np.asarray([] if record_days is None else record_days, dtype=np.int64)
        snp = np.asarray([] if snap_days is None else snap_days, dtype=np.int64)
        if np.any(np.diff(rec) < 0) or np.any(np.diff(snp) < 0):
            raise ValueError("record and snapshot days must be sorted")
        y = np.array(y0, dtype=float, copy=True)
        acc = np.array(acc0, dtype=float, copy=True)
        acc_out = np.full((rec.size, self.grid.n_bins, NA), np.nan)
        snap_out = np.full((snp.size, self.grid.n_bins, NC), np.nan)
        infant_extra = np.zeros_like(t["infant_dose"]) if infant_extra is None else np.asarray(infant_extra, dtype=float)
        event_extra = np.zeros_like(t["ev_frac"]) if event_extra is None else np.asarray(event_extra, dtype=float)
        for value, expected in ((infant_extra, t["infant_dose"]), (event_extra, t["ev_frac"])):
            if value.shape != expected.shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("extra dose requests must be finite, nonnegative and match event tables")
        receipts = np.zeros((self.calendar.n_years, N_DOSE_KINDS, 3))
        grid = self.grid
        integrate(
            y, acc, int(day0), int(day1), P, sig, pityp,
            t["births"], t["mat_eff"], t["npi_r"], t["imp_mult"], t["season_on"],
            t["year_frac"], t["month_idx"], t["year_idx"], t["flags"],
            t["nudge_target"], t["primary_product"], t["infant_dose"], t["c_home"], t["c_non"],
            t["group"], t["age_rate"], t["contact_mult"],
            grid.n_monthly, grid.first_cohort_bin, grid.last_cohort_bin,
            t["ev_day"], t["ev_bin"], t["ev_kind"], t["ev_frac"],
            rec, acc_out, snp, snap_out,
            infant_extra, event_extra, receipts,
        )
        return SimulationResult(y, acc, rec, acc_out, snp, snap_out, receipts)
