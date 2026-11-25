"""Births and target populations from UN WPP 2024.

Sources (``data/raw/external``):

* ``wpp2024_population_by_age_sex_1990_2050.csv``: mid-year population by
  single year of age 0-100 (100 = 100+), both sexes, medium variant.
* ``WPP data.csv``: crude birth rate per 1,000, 1950-2050.

Annual births are ``CBR * total mid-year population / 1000``.  Single-year
populations exist only from 1990, so all demographic inputs are held at their
1990 values before 1990 (spin-up and the 1970-1989 history) and at 2050 values
after 2050.

Targets for the kernel's demographic relaxation:

* cohort bin ``k >= 3`` in calendar year ``Y`` holds birth year ``B = Y - k``.
  On 1 July of year ``X`` that birth year has completed ages ``X-B-1`` (born
  July-December) and ``X-B`` (born January-June), so its size is
  ``0.5 * (N[X-B-1](X) + N[X-B](X))``.  Monthly targets interpolate this
  cohort trajectory linearly between mid-years, so a target never jumps when
  the cohort changes bin on 1 January;
* adult bins and the open bin use WPP summed over their half-year-offset age
  range (20.5-25.5, ..., 70.5+; see ``grid``), counting the boundary single
  years with weight one half;
* monthly bins and the filling cohort bin 2 have no target (births only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..grid import KIND_ADULT, KIND_COHORT, KIND_OPEN, AgeGrid, uniform_age_overlap
from ..model.tables import Calendar

RAW = REPO_ROOT / "data" / "raw" / "external"
WPP_SINGLE_YEAR = RAW / "wpp2024_population_by_age_sex_1990_2050.csv"
WPP_LONG = RAW / "WPP data.csv"
FIRST_DATA_YEAR = 1990
LAST_DATA_YEAR = 2050
MAX_AGE = 100

COUNTRY_ISO3 = {
    "Australia": "AUS",
    "Brazil": "BRA",
    "China": "CHN",
    "Japan": "JPN",
    "New_Zealand": "NZL",
    "Sweden": "SWE",
    "Thailand": "THA",
    "United_Kingdom": "GBR",
}


@dataclass(frozen=True)
class Demography:
    iso3: str
    years: np.ndarray        # consecutive available calendar years
    population: np.ndarray   # (n_years, 101) mid-year population by single year of age
    births: np.ndarray       # (n_years,) births during the calendar year

    def __post_init__(self):
        if (len(self.years) == 0 or np.any(np.diff(self.years) != 1)
                or self.population.shape != (len(self.years), MAX_AGE + 1)
                or self.births.shape != (len(self.years),)):
            raise ValueError("demography requires consecutive years and aligned population/birth arrays")
        if (not np.isfinite(self.population).all() or not np.isfinite(self.births).all()
                or np.any(self.population < 0) or np.any(self.births <= 0)):
            raise ValueError("demography must contain finite nonnegative populations and positive births")

    def year_row(self, year: int) -> int:
        return int(np.clip(year, self.years[0], self.years[-1]) - self.years[0])

    def single_age(self, age: int, year: int) -> float:
        if age < 0 or age > MAX_AGE:
            return 0.0
        return float(self.population[self.year_row(year), age])

    def age_sum(self, lo: int, hi: int, year: int) -> float:
        hi = min(hi, MAX_AGE + 1)
        return float(self.population[self.year_row(year), lo:hi].sum())

    def continuous_age_sum(self, lower: float, upper: float, year: int) -> float:
        """Population aged ``[lower, upper)``, uniform within each single year."""

        weights = uniform_age_overlap(lower, min(upper, MAX_AGE + 1.0), MAX_AGE) * (
            min(upper, MAX_AGE + 1.0) - lower
        )
        return float(self.population[self.year_row(year)] @ weights)

    def cohort_mid_year(self, birth_year: int, year: int) -> float:
        """Size on 1 July of ``year`` of everyone born in calendar ``birth_year``."""

        x = year - birth_year
        return 0.5 * (self.single_age(x - 1, year) + self.single_age(x, year))

    def births_in(self, year: int) -> float:
        return float(self.births[self.year_row(year)])


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def load_demography(iso3: str) -> Demography:
    single = _read_csv(WPP_SINGLE_YEAR)
    single = single[(single["Iso3"] == iso3) & (single["Sex"] == "Both sexes")]
    if single.empty:
        raise KeyError(f"no WPP single-year data for {iso3}")
    years = np.arange(FIRST_DATA_YEAR, LAST_DATA_YEAR + 1)
    table = (
        single.pivot_table(index="Time", columns="AgeStart", values="Value", aggfunc="sum")
        .reindex(index=years, columns=range(MAX_AGE + 1))
    )
    if table.isna().any().any():
        raise ValueError(f"WPP single-year table for {iso3} has gaps")
    population = table.to_numpy(dtype=float)
    longrun = _read_csv(WPP_LONG, usecols=["IndicatorName", "Iso3", "Time", "Value"])
    cbr = (
        longrun[(longrun["Iso3"] == iso3) & (longrun["IndicatorName"] == "Crude birth rate")]
        .set_index("Time")["Value"]
        .reindex(years)
    )
    if cbr.isna().any():
        raise ValueError(f"crude birth rate missing for {iso3}")
    births = cbr.to_numpy(dtype=float) * population.sum(axis=1) / 1000.0
    return Demography(iso3=iso3, years=years, population=population, births=births)


def births_per_day(demo: Demography, calendar: Calendar) -> np.ndarray:
    years = calendar.year_idx + calendar.first_year
    out = np.empty(calendar.n_days)
    for year in np.unique(years):
        mask = years == year
        out[mask] = demo.births_in(int(year)) / mask.sum()
    return out


def _interp_mid_year(values_at_mid_year, tau: float) -> float:
    """Linear interpolation of a function given on mid-years ``X + 0.5``."""

    lower = int(np.floor(tau - 0.5))
    w = tau - 0.5 - lower
    return (1.0 - w) * values_at_mid_year(lower) + w * values_at_mid_year(lower + 1)


def nudge_targets(demo: Demography, grid: AgeGrid, calendar: Calendar) -> np.ndarray:
    targets = np.zeros((calendar.n_months, grid.n_bins))
    for mi in range(calendar.n_months):
        year = calendar.first_year + mi // 12
        tau = year + ((mi % 12) + 0.5) / 12.0
        for b in range(grid.n_bins):
            kind = grid.kind[b]
            if kind == KIND_COHORT and grid.cohort_age[b] > grid.cohort_first:
                birth_year = year - int(grid.cohort_age[b])
                targets[mi, b] = _interp_mid_year(lambda x: demo.cohort_mid_year(birth_year, x), tau)
            elif kind in (KIND_ADULT, KIND_OPEN):
                lo = float(grid.lower_years[b])
                hi = MAX_AGE + 1.0 if kind == KIND_OPEN else float(grid.upper_years[b])
                targets[mi, b] = _interp_mid_year(lambda x: demo.continuous_age_sum(lo, hi, x), tau)
    return targets


def contact_group_population(demo: Demography, year: int, n_groups: int = 15) -> np.ndarray:
    """Mid-year population in the model's contact groups (0-4 ... 65-69, 70+)."""

    out = np.empty(n_groups)
    for g in range(n_groups - 1):
        out[g] = demo.age_sum(5 * g, 5 * g + 5, year)
    out[n_groups - 1] = demo.age_sum(5 * (n_groups - 1), MAX_AGE + 1, year)
    return out
