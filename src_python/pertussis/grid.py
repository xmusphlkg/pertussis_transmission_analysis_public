"""Age grid with monthly infant bins, birth-year cohort bins and adult bins.

Bin layout (54 bins by default):

* ``monthly``: ages 0-23 months, one bin per month of age.  Every bin shifts by
  one at each calendar month start, so a bin always holds one birth month.
* ``cohort``: completed ages 2-20 years, one bin per birth year.  Bin ``k``
  fills month by month while children leave the 23-month bin (k = 2) or holds
  the birth year ``Y - k`` during calendar year ``Y`` (k >= 3).  All cohort bins
  shift rigidly on 1 January, so birth cohorts are never smeared.
* ``adult``: five-year bins labelled 20-24 ... 65-69 with continuous ageing at
  rate 1/5 per year, and an open bin labelled 70+.  A cohort enters the first
  adult bin on 1 January aged 20.0-21.0, so with continuous ageing the adult
  bins hold ages offset by half a year (20.5-25.5, ..., 70.5+); their
  ``lower_years``/``upper_years`` carry this offset and the labels the nominal
  ages.

Cohort bin ``k >= 3`` contains completed ages ``k - 1`` and ``k``; at year
fraction ``f`` a share ``f`` of the cohort has already had its birthday,
assuming births are uniform within the year.  ``completed_age_weights`` uses
this Lexis relation to map model bins to reported age bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KIND_MONTHLY = 0
KIND_COHORT = 1
KIND_ADULT = 2
KIND_OPEN = 3

# Prem et al. synthetic matrices use 16 five-year groups, 0-4 ... 75-79 (the
# last group is treated as 75+).  The model merges 70-74 and 75+ into one 70+
# contact group, giving 15 contact groups.
N_CONTACT_GROUPS = 15

ADULT_OFFSET_YEARS = 0.5
# Completed-age detail inside the open bin is not modelled; for mapping it is
# spread uniformly up to this age.
OPEN_BIN_SPREAD_TO = 85.0


def uniform_age_overlap(lower: float, upper: float, max_age: int = 100) -> np.ndarray:
    """Shares of a uniform density on ``[lower, upper)`` by completed age."""

    ages = np.arange(max_age + 1, dtype=float)
    overlap = np.clip(np.minimum(ages + 1.0, upper) - np.maximum(ages, lower), 0.0, None)
    return overlap / (upper - lower)


@dataclass(frozen=True)
class AgeGrid:
    n_monthly: int = 24
    cohort_first: int = 2
    cohort_last: int = 20
    adult_edges: tuple[int, ...] = (20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70)
    kind: np.ndarray = field(init=False, repr=False)
    lower_years: np.ndarray = field(init=False, repr=False)
    upper_years: np.ndarray = field(init=False, repr=False)
    contact_group: np.ndarray = field(init=False, repr=False)
    adult_ageing_rate_per_day: np.ndarray = field(init=False, repr=False)
    cohort_age: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cohort_first != 2 or self.n_monthly != 24:
            raise ValueError("cohort bins must start at age 2 after 24 monthly bins")
        if self.adult_edges[0] != self.cohort_last:
            raise ValueError("adult bins must start where cohort bins end")
        kinds: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        group: list[int] = []
        cohort_age: list[int] = []
        rate: list[float] = []
        for month in range(self.n_monthly):
            kinds.append(KIND_MONTHLY)
            lower.append(month / 12.0)
            upper.append((month + 1) / 12.0)
            group.append(0)
            cohort_age.append(-1)
            rate.append(0.0)
        for age in range(self.cohort_first, self.cohort_last + 1):
            kinds.append(KIND_COHORT)
            # Mid-year content: ages age-0.5 .. age+0.5 (age 2: 2..3).
            lower.append(float(age - 1) if age > self.cohort_first else float(age))
            upper.append(float(age + 1))
            group.append(min(age // 5, N_CONTACT_GROUPS - 1))
            cohort_age.append(age)
            rate.append(0.0)
        edges = self.adult_edges
        for left, right in zip(edges[:-1], edges[1:]):
            kinds.append(KIND_ADULT)
            lower.append(left + ADULT_OFFSET_YEARS)
            upper.append(right + ADULT_OFFSET_YEARS)
            group.append(min(left // 5, N_CONTACT_GROUPS - 1))
            cohort_age.append(-1)
            rate.append(1.0 / ((right - left) * 365.25))
        kinds.append(KIND_OPEN)
        lower.append(edges[-1] + ADULT_OFFSET_YEARS)
        upper.append(np.inf)
        group.append(N_CONTACT_GROUPS - 1)
        cohort_age.append(-1)
        rate.append(0.0)
        object.__setattr__(self, "kind", np.asarray(kinds, dtype=np.int64))
        object.__setattr__(self, "lower_years", np.asarray(lower, dtype=float))
        object.__setattr__(self, "upper_years", np.asarray(upper, dtype=float))
        object.__setattr__(self, "contact_group", np.asarray(group, dtype=np.int64))
        object.__setattr__(self, "cohort_age", np.asarray(cohort_age, dtype=np.int64))
        object.__setattr__(self, "adult_ageing_rate_per_day", np.asarray(rate, dtype=float))

    @property
    def n_bins(self) -> int:
        return int(self.kind.size)

    @property
    def first_cohort_bin(self) -> int:
        return self.n_monthly

    @property
    def last_cohort_bin(self) -> int:
        return self.n_monthly + (self.cohort_last - self.cohort_first)

    @property
    def first_adult_bin(self) -> int:
        return self.last_cohort_bin + 1

    def cohort_bin(self, age: int) -> int:
        if not self.cohort_first <= age <= self.cohort_last:
            raise ValueError(f"no cohort bin for age {age}")
        return self.n_monthly + age - self.cohort_first

    def labels(self) -> list[str]:
        out: list[str] = []
        for b in range(self.n_bins):
            k = self.kind[b]
            if k == KIND_MONTHLY:
                out.append(f"m{b:02d}")
            elif k == KIND_COHORT:
                out.append(f"y{self.cohort_age[b]:02d}")
            elif k == KIND_ADULT:
                lo = int(self.lower_years[b] - ADULT_OFFSET_YEARS)
                hi = int(self.upper_years[b] - ADULT_OFFSET_YEARS)
                out.append(f"{lo}-{hi - 1}")
            else:
                out.append(f"{int(self.lower_years[b] - ADULT_OFFSET_YEARS)}+")
        return out

    def completed_age_weights(self, year_fraction: float, max_age: int = 100) -> np.ndarray:
        """Return a ``(n_bins, max_age + 1)`` matrix of completed-age shares.

        Row ``b`` gives the share of bin ``b``'s content (or events) at each
        completed age, at the given fraction of the calendar year.  Adult bins
        are spread uniformly over their half-year-offset age range; the open
        bin uniformly from 70.5 to ``OPEN_BIN_SPREAD_TO`` (only meaningful for
        reported bands whose edges are at or below 70, which is how
        observation bands are configured).
        """

        f = float(np.clip(year_fraction, 0.0, 1.0))
        weights = np.zeros((self.n_bins, max_age + 1), dtype=float)
        for b in range(self.n_bins):
            k = self.kind[b]
            if k == KIND_MONTHLY:
                weights[b, 0 if b < 12 else 1] = 1.0
            elif k == KIND_COHORT:
                age = int(self.cohort_age[b])
                if age == self.cohort_first:
                    weights[b, age] = 1.0
                else:
                    weights[b, age] += f
                    weights[b, age - 1] += 1.0 - f
            elif k == KIND_ADULT:
                weights[b] = uniform_age_overlap(self.lower_years[b], self.upper_years[b], max_age)
            else:
                weights[b] = uniform_age_overlap(self.lower_years[b], OPEN_BIN_SPREAD_TO, max_age)
        return weights


def band_matrix(
    grid: AgeGrid,
    bands: list[tuple[float, float]],
    year_fraction: float,
) -> np.ndarray:
    """Return ``(n_bins, n_bands)`` shares mapping bin events to age bands.

    ``bands`` are ``[lower, upper)`` completed-age intervals in years (upper may
    be ``inf``); monthly bins are mapped by their exact month range so bands such
    as ``<1`` and ``1-4`` are resolved.  Every band edge at or above 70 other
    than the open end must be avoided, because the open bin carries no age
    detail beyond 70.
    """

    for lo, hi in bands:
        if lo > 70 or (70 < hi < np.inf):
            raise ValueError(f"band [{lo}, {hi}) needs age detail above 70")
    completed = grid.completed_age_weights(year_fraction)
    ages = np.arange(completed.shape[1])
    out = np.zeros((grid.n_bins, len(bands)), dtype=float)
    for j, (lo, hi) in enumerate(bands):
        inside = (ages >= lo) & (ages < hi)
        out[:, j] = completed[:, inside].sum(axis=1)
        # Completed-year weights collapse all twelve infant bins to age zero.
        # Preserve the grid's monthly resolution for sub-year source bands.
        monthly = grid.kind == KIND_MONTHLY
        left = grid.lower_years[monthly]
        right = grid.upper_years[monthly]
        out[monthly, j] = np.maximum(0.0, np.minimum(right, hi) - np.maximum(left, lo)) / (right - left)
    return out
