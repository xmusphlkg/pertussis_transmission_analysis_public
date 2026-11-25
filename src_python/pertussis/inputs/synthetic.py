"""Synthetic stationary inputs for tests and benchmarks.

A closed, stationary population (constant births, one target cohort per year,
no NPI) with a simple assortative contact matrix and a configurable 3+1
infant schedule, one preschool booster and one adolescent Tdap dose.
"""

from __future__ import annotations

import numpy as np

from ..grid import N_CONTACT_GROUPS, AgeGrid, KIND_ADULT, KIND_COHORT, KIND_OPEN
from ..model.tables import Calendar, DoseEvents, ModelInputs
from ..states import CHILD_BOOSTER, N_INFANT_DOSE_KINDS, PRODUCT_AP, PRODUCT_WP, TDAP_BOOSTER


def assortative_contacts(total_per_day: float = 15.0, within_share: float = 0.4) -> np.ndarray:
    g = N_CONTACT_GROUPS
    idx = np.arange(g)
    distance = np.abs(idx[:, None] - idx[None, :])
    base = np.exp(-distance / 2.0)
    base = base / base.sum(axis=1, keepdims=True) * (1.0 - within_share)
    base[idx, idx] += within_share
    return total_per_day * base


def stationary_targets(grid: AgeGrid, births_per_year: float) -> np.ndarray:
    target = np.zeros(grid.n_bins)
    for b in range(grid.n_bins):
        if grid.kind[b] == KIND_COHORT and grid.cohort_age[b] > grid.cohort_first:
            target[b] = births_per_year
        elif grid.kind[b] == KIND_ADULT:
            target[b] = births_per_year * (grid.upper_years[b] - grid.lower_years[b])
        elif grid.kind[b] == KIND_OPEN:
            target[b] = births_per_year * 12.0
    return target


def synthetic_inputs(
    first_year: int = 1950,
    n_years: int = 40,
    spinup_years: int = 20,
    births_per_year: float = 100_000.0,
    infant_schedule_months: tuple[int, ...] = (2, 4, 6),
    primary_coverage: tuple[float, ...] = (0.95, 0.93, 0.92),
    toddler_booster_month: int | None = 18,
    toddler_booster_coverage: float = 0.85,
    preschool_booster_age: float | None = 4.0,
    preschool_booster_coverage: float = 0.85,
    adolescent_age: float | None = 12.0,
    adolescent_coverage: float = 0.75,
    ap_switch_year: int | None = None,
    maternal_coverage: float = 0.0,
    nudging: bool = True,
    vaccination: bool = True,
) -> ModelInputs:
    grid = AgeGrid()
    cal = Calendar(first_year, n_years)
    nd, nm, ny = cal.n_days, cal.n_months, cal.n_years
    births = np.full(nd, births_per_year / 365.25)
    target = stationary_targets(grid, births_per_year) if nudging else np.zeros(grid.n_bins)
    nudge_target = np.tile(target, (nm, 1))
    season_on = (cal.year_idx >= spinup_years).astype(float)
    product = np.full(nm, PRODUCT_WP, dtype=np.int64)
    if ap_switch_year is not None:
        product[(np.arange(nm) // 12 + first_year) >= ap_switch_year] = PRODUCT_AP
    infant_dose = np.zeros((nm, grid.n_monthly, N_INFANT_DOSE_KINDS))
    days: list[int] = []
    bins: list[int] = []
    kinds: list[int] = []
    fracs: list[float] = []
    if vaccination:
        previous = 1.0
        for k, (month, cov) in enumerate(zip(infant_schedule_months, primary_coverage)):
            infant_dose[:, month, k] = cov / previous
            previous = cov
        if toddler_booster_month is not None:
            infant_dose[:, toddler_booster_month, 3] = toddler_booster_coverage
        for age, cov, kind in (
            (preschool_booster_age, preschool_booster_coverage, CHILD_BOOSTER),
            (adolescent_age, adolescent_coverage, TDAP_BOOSTER),
        ):
            if age is None:
                continue
            k = int(np.floor(age + 0.5))
            f = age + 0.5 - k
            for year in range(first_year, first_year + n_years):
                day = cal.year_start_day(year) + int(round(f * 365.25))
                days.append(min(day, nd - 1))
                bins.append(grid.cohort_bin(k))
                kinds.append(kind)
                fracs.append(cov)
    events = DoseEvents(
        np.asarray(days, dtype=np.int64),
        np.asarray(bins, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64),
        np.asarray(fracs, dtype=float),
    )
    contacts = assortative_contacts()
    home = np.tile(0.3 * contacts, (ny, 1, 1))
    nonhome = np.tile(0.7 * contacts, (ny, 1, 1))
    return ModelInputs(
        grid=grid,
        calendar=cal,
        births_per_day=births,
        maternal_effective=np.full(nd, maternal_coverage),
        npi_reduction=np.zeros(nd),
        import_multiplier=np.ones(nd),
        season_on=season_on,
        nudge_target=nudge_target,
        primary_product=product,
        infant_dose=infant_dose,
        contacts_home=home,
        contacts_nonhome=nonhome,
        contact_multiplier=np.ones(grid.n_bins),
        events=events,
        reference_year=first_year,
        metadata={"source": "synthetic", "spinup_years": spinup_years},
    )
