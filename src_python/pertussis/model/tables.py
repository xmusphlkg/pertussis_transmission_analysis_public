"""Calendar and input tables consumed by the kernel.

``ModelInputs`` is the only object the simulator needs.  It is built from raw
country data by ``pertussis.inputs.bundle`` (or synthetically in tests) and holds
every time-varying input as a dense table on a calendar that starts on
1 January of ``first_year``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grid import N_CONTACT_GROUPS, AgeGrid
from ..states import N_INFANT_DOSE_KINDS
from .kernel import FLAG_MONTH_START, FLAG_YEAR_START


@dataclass(frozen=True)
class Calendar:
    first_year: int
    n_years: int
    dates: np.ndarray = field(init=False, repr=False)
    year_frac: np.ndarray = field(init=False, repr=False)
    month_idx: np.ndarray = field(init=False, repr=False)
    year_idx: np.ndarray = field(init=False, repr=False)
    flags: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        start = np.datetime64(f"{self.first_year:04d}-01-01")
        end = np.datetime64(f"{self.first_year + self.n_years:04d}-01-01")
        dates = np.arange(start, end, dtype="datetime64[D]")
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        months = dates.astype("datetime64[M]").astype(int) % 12
        month_start = dates.astype("datetime64[M]").astype("datetime64[D]") == dates
        year_start = dates.astype("datetime64[Y]").astype("datetime64[D]") == dates
        doy = (dates - dates.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)
        flags = month_start.astype(np.int64) * FLAG_MONTH_START + year_start.astype(np.int64) * FLAG_YEAR_START
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "year_frac", doy.astype(float) / 365.25)
        object.__setattr__(self, "month_idx", ((years - self.first_year) * 12 + months).astype(np.int64))
        object.__setattr__(self, "year_idx", (years - self.first_year).astype(np.int64))
        object.__setattr__(self, "flags", flags.astype(np.int64))

    @property
    def n_days(self) -> int:
        return int(self.dates.size)

    @property
    def n_months(self) -> int:
        return 12 * self.n_years

    def day(self, date: str | np.datetime64) -> int:
        """Day index of a date (may equal ``n_days`` for the end boundary)."""

        value = int((np.datetime64(date, "D") - self.dates[0]).astype(int))
        if not 0 <= value <= self.n_days:
            raise ValueError(f"{date} is outside the calendar")
        return value

    def year_start_day(self, year: int) -> int:
        return self.day(f"{year:04d}-01-01")

    def month_start_days(self, first: str, last_exclusive: str) -> np.ndarray:
        months = np.arange(np.datetime64(first, "M"), np.datetime64(last_exclusive, "M"))
        return np.asarray([self.day(m.astype("datetime64[D]")) for m in months], dtype=np.int64)


@dataclass
class DoseEvents:
    """Cohort dose events: one row per (day, bin, kind) with fraction dosed."""

    day: np.ndarray
    bin: np.ndarray
    kind: np.ndarray
    fraction: np.ndarray

    @classmethod
    def empty(cls) -> "DoseEvents":
        return cls(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=float),
        )

    def sorted(self) -> "DoseEvents":
        order = np.lexsort((self.kind, self.bin, self.day))
        return DoseEvents(
            np.asarray(self.day, dtype=np.int64)[order],
            np.asarray(self.bin, dtype=np.int64)[order],
            np.asarray(self.kind, dtype=np.int64)[order],
            np.asarray(self.fraction, dtype=float)[order],
        )


@dataclass
class ModelInputs:
    grid: AgeGrid
    calendar: Calendar
    births_per_day: np.ndarray        # (n_days,)
    maternal_effective: np.ndarray    # (n_days,) share of births entering M
    npi_reduction: np.ndarray         # (n_days,) non-home contact reduction timeline in [0, 1]
    import_multiplier: np.ndarray     # (n_days,)
    season_on: np.ndarray             # (n_days,) 0 during spin-up
    nudge_target: np.ndarray          # (n_months, n_bins); <= 0 means no relaxation
    primary_product: np.ndarray       # (n_months,) PRODUCT_WP / PRODUCT_AP
    infant_dose: np.ndarray           # (n_months, n_monthly, 4) conditional fractions at entry
    contacts_home: np.ndarray         # (n_years, G, G) contacts per day, participant rows
    contacts_nonhome: np.ndarray      # (n_years, G, G)
    contact_multiplier: np.ndarray    # (n_bins,)
    events: DoseEvents
    reference_year: int               # contact year used to convert R0 to q
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        cal, grid = self.calendar, self.grid
        nd, nm, ny, nb = cal.n_days, cal.n_months, cal.n_years, grid.n_bins
        g = N_CONTACT_GROUPS
        shapes = {
            "births_per_day": (nd,),
            "maternal_effective": (nd,),
            "npi_reduction": (nd,),
            "import_multiplier": (nd,),
            "season_on": (nd,),
            "nudge_target": (nm, nb),
            "primary_product": (nm,),
            "infant_dose": (nm, grid.n_monthly, N_INFANT_DOSE_KINDS),
            "contacts_home": (ny, g, g),
            "contacts_nonhome": (ny, g, g),
            "contact_multiplier": (nb,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        for name in ("births_per_day", "npi_reduction", "import_multiplier", "contact_multiplier"):
            if np.any(np.asarray(getattr(self, name)) < 0):
                raise ValueError(f"{name} must be non-negative")
        for name in ("maternal_effective", "infant_dose"):
            value = np.asarray(getattr(self, name))
            if np.any(value < 0) or np.any(value > 1):
                raise ValueError(f"{name} must lie in [0, 1]")
        ev = self.events
        if np.any(ev.fraction < 0) or np.any(ev.fraction > 1):
            raise ValueError("dose event fractions must lie in [0, 1]")
        if ev.day.size and (ev.day.min() < 0 or ev.day.max() >= nd):
            raise ValueError("dose event outside the calendar")
        if ev.bin.size and np.any(grid.kind[ev.bin] == 0):
            raise ValueError("cohort dose events cannot target monthly bins")
        if not cal.first_year <= self.reference_year < cal.first_year + ny:
            raise ValueError("reference_year outside the calendar")

    def kernel_tables(self) -> dict[str, np.ndarray]:
        cal, grid = self.calendar, self.grid
        ev = self.events.sorted()
        return {
            "births": np.ascontiguousarray(self.births_per_day, dtype=float),
            "mat_eff": np.ascontiguousarray(self.maternal_effective, dtype=float),
            "npi_r": np.ascontiguousarray(self.npi_reduction, dtype=float),
            "imp_mult": np.ascontiguousarray(self.import_multiplier, dtype=float),
            "season_on": np.ascontiguousarray(self.season_on, dtype=float),
            "year_frac": cal.year_frac,
            "month_idx": cal.month_idx,
            "year_idx": cal.year_idx,
            "flags": cal.flags,
            "nudge_target": np.ascontiguousarray(self.nudge_target, dtype=float),
            "primary_product": np.ascontiguousarray(self.primary_product, dtype=np.int64),
            "infant_dose": np.ascontiguousarray(self.infant_dose, dtype=float),
            "c_home": np.ascontiguousarray(self.contacts_home, dtype=float),
            "c_non": np.ascontiguousarray(self.contacts_nonhome, dtype=float),
            "group": grid.contact_group,
            "age_rate": grid.adult_ageing_rate_per_day,
            "contact_mult": np.ascontiguousarray(self.contact_multiplier, dtype=float),
            "ev_day": ev.day,
            "ev_bin": ev.bin,
            "ev_kind": ev.kind,
            "ev_frac": ev.fraction,
        }
