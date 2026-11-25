"""Encode a country's pertussis vaccination programme into kernel tables.

The model-ready programme file (``config/core2026/programmes/<Country>.yaml``;
format documented in ``config/core2026/programmes/README.md`` and
``TEMPLATE.yaml``) lists dated schedule versions, the primary-series product
over time, named annual coverage series and maternal programme periods.  This
module validates it and produces:

* ``infant_dose`` (n_months, 24, 4): conditional fractions applied at monthly
  bin entry for dose 1, dose 2, dose 3 and the toddler booster, spread over
  later months by the delay distribution;
* ``primary_product`` (n_months,): wP or aP for completion of the series;
* cohort ``DoseEvents`` for doses at ages of 2.5 years and above;
* ``maternal_effective`` (n_days,): share of births entering the maternally
  protected state.

Coverage of a calendar year applies to doses delivered in that year.  Dates
before ``freeze_before`` (the start of the historical run) use the programme
as of that date, so the spin-up runs under constant early conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..grid import AgeGrid
from ..model.tables import Calendar, DoseEvents
from ..states import CHILD_BOOSTER, LATE_PRIMARY, N_INFANT_DOSE_KINDS, PRODUCT_AP, PRODUCT_WP, TDAP_BOOSTER
from .demography import RAW

SCHEMA = "core2026-programme-v1"
WUENIC_PATH = RAW.parents[1] / "processed" / "wuenic_dtp_coverage.csv"
INFANT_KINDS = {"dose1": 0, "dose2": 1, "dose3": 2, "toddler_booster": 3}
COHORT_KINDS = {"child_booster": CHILD_BOOSTER, "tdap_booster": TDAP_BOOSTER, "late_primary": LATE_PRIMARY}
PRODUCTS = {"wP": PRODUCT_WP, "aP": PRODUCT_AP}
EVIDENCE_GRADES = {"A", "B", "C", "D"}
MIN_COHORT_DOSE_AGE = 2.5


def _date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


@lru_cache(maxsize=1)
def _wuenic_table() -> pd.DataFrame:
    return pd.read_csv(WUENIC_PATH)


def wuenic_series(iso3: str, antigen: str) -> dict[int, float]:
    t = _wuenic_table()
    rows = t[(t["CODE"] == iso3) & (t["ANTIGEN"] == antigen)]
    return {int(y): float(c) / 100.0 for y, c in zip(rows["YEAR"], rows["COVERAGE"])}


@dataclass(frozen=True)
class SeriesSegment:
    first: int
    last: int
    kind: str                  # value | wuenic | interpolate
    payload: Any
    sources: tuple[str, ...]
    evidence_grade: str


@dataclass
class CoverageSeries:
    name: str
    segments: list[SeriesSegment]
    iso3: str
    derived: tuple[str, ...] = ()
    _cache: dict[int, float] = field(default_factory=dict, repr=False)

    def value(self, year: int, programme: "Programme") -> float:
        if year in self._cache:
            return self._cache[year]
        if self.derived:
            out = float(np.mean([programme.coverage(n, year) for n in self.derived]))
        else:
            seg = next((s for s in self.segments if s.first <= year <= s.last), None)
            if seg is None:
                raise ValueError(f"coverage series {self.name} has no value for {year}")
            if seg.kind == "value":
                out = float(seg.payload)
            elif seg.kind == "wuenic":
                table = wuenic_series(self.iso3, seg.payload)
                if year not in table:
                    raise ValueError(f"WUENIC {seg.payload} missing {self.iso3} {year}")
                out = table[year]
            else:
                anchors = sorted((int(k), float(v)) for k, v in seg.payload.items())
                xs, ys = zip(*anchors)
                out = float(np.interp(year, xs, ys))
        if not 0.0 <= out <= 1.0:
            raise ValueError(f"coverage {self.name} {year} = {out} outside [0, 1]")
        self._cache[year] = out
        return out


@dataclass(frozen=True)
class DoseSpec:
    name: str
    kind: str
    age: float                 # months for infant doses, years for cohort doses
    start: date
    end: date | None
    coverage: str
    sources: tuple[str, ...]
    birth_year_start: int | None = None
    birth_year_end: int | None = None

    def active(self, when: date) -> bool:
        return self.start <= when and (self.end is None or when < self.end)

    def applies_to_birth_year(self, year: int) -> bool:
        return (self.birth_year_start is None or year >= self.birth_year_start) and (
            self.birth_year_end is None or year < self.birth_year_end
        )


@dataclass(frozen=True)
class MaternalPeriod:
    start: date
    end: date | None
    coverage: str
    effective_share: float

    def active(self, when: date) -> bool:
        return self.start <= when and (self.end is None or when < self.end)


@dataclass
class Programme:
    country: str
    iso3: str
    status: str
    products: list[tuple[date, int]]
    coverage_series: dict[str, CoverageSeries]
    infant_doses: list[DoseSpec]
    cohort_doses: list[DoseSpec]
    maternal: list[MaternalPeriod]
    delay_pmf: np.ndarray
    sources: dict[str, dict]
    path: Path | None = None

    def coverage(self, name: str, year: int) -> float:
        if name not in self.coverage_series:
            raise KeyError(f"unknown coverage series {name}")
        return self.coverage_series[name].value(year, self)

    def product_at(self, when: date) -> int:
        current = None
        for start, product in self.products:
            if start <= when:
                current = product
        if current is None:
            raise ValueError(f"no primary-series product defined at {when}")
        return current


def _parse_segments(name: str, entries: list[dict], sources: dict) -> list[SeriesSegment]:
    segments = []
    for e in entries:
        kinds = [k for k in ("value", "wuenic", "interpolate") if k in e]
        if len(kinds) != 1:
            raise ValueError(f"{name}: each segment needs exactly one of value/wuenic/interpolate")
        grade = str(e.get("evidence_grade", ""))
        if grade not in EVIDENCE_GRADES:
            raise ValueError(f"{name}: evidence_grade must be one of {sorted(EVIDENCE_GRADES)}")
        refs = tuple(e.get("sources", []))
        missing = [r for r in refs if r not in sources]
        if missing or not refs:
            raise ValueError(f"{name}: segment needs known sources, got {refs}")
        segments.append(SeriesSegment(int(e["from"]), int(e["to"]), kinds[0], e[kinds[0]], refs, grade))
    segments.sort(key=lambda s: s.first)
    for a, b in zip(segments[:-1], segments[1:]):
        if b.first <= a.last:
            raise ValueError(f"{name}: overlapping segments {a.first}-{a.last} and {b.first}-{b.last}")
    return segments


def _evidence_grade(entry: dict, label: str) -> str:
    grade = str(entry.get("evidence_grade", ""))
    if grade not in EVIDENCE_GRADES:
        raise ValueError(f"{label}: evidence_grade must be one of {sorted(EVIDENCE_GRADES)}")
    return grade


def _source_refs(entry: dict, sources: dict, label: str) -> tuple[str, ...]:
    refs = tuple(entry.get("sources", []))
    missing = [ref for ref in refs if ref not in sources]
    if missing or not refs:
        raise ValueError(f"{label}: needs known sources")
    _evidence_grade(entry, label)
    return refs


def _parse_doses(entries: list[dict], kinds: dict, age_key: str, sources: dict) -> list[DoseSpec]:
    out = []
    for e in entries:
        if e["kind"] not in kinds:
            raise ValueError(f"dose {e.get('name')}: kind must be one of {sorted(kinds)}")
        refs = _source_refs(e, sources, f"dose {e.get('name')}")
        out.append(DoseSpec(
            name=str(e["name"]), kind=str(e["kind"]), age=float(e[age_key]),
            start=_date(e["start"]), end=_date(e.get("end")), coverage=str(e["coverage"]), sources=refs,
            birth_year_start=(None if e.get("birth_year_start") is None else int(e["birth_year_start"])),
            birth_year_end=(None if e.get("birth_year_end") is None else int(e["birth_year_end"])),
        ))
    return out


def _apply_variant(raw: dict, variant: str | None) -> dict:
    if variant is None:
        return raw
    variants = raw.get("sensitivity_variants") or {}
    if variant not in variants:
        raise KeyError(f"unknown programme sensitivity variant {variant!r}")
    out = deepcopy(raw)
    for operation in variants[variant].get("operations", []):
        kind = operation.get("kind")
        if kind == "set_product_start":
            product = operation["product"]
            entry = next(e for e in out["primary_product"] if e["product"] == product)
            entry["start"] = operation["start"]
        elif kind == "set_coverage_segment":
            series = out["coverage_series"][operation["series"]]
            match = next(e for e in series if int(e["from"]) == int(operation["from"]) and int(e["to"]) == int(operation["to"]))
            match.pop("value", None)
            match.pop("wuenic", None)
            match.pop("interpolate", None)
            match["value"] = float(operation["value"])
        elif kind == "set_dose_field":
            if operation["block"] == "maternal":
                blocks = out["maternal"]
                entry = blocks[0] if operation["name"] == "__period_start__" else next(
                    e for e in blocks if e.get("name") == operation["name"]
                )
            else:
                blocks = out["infant_doses"] if operation["block"] == "infant_doses" else out["cohort_doses"]
                entry = next(e for e in blocks if e["name"] == operation["name"])
            entry[operation["field"]] = operation["value"]
        elif kind == "set_birth_cohort":
            blocks = out["infant_doses"] if operation["block"] == "infant_doses" else out["cohort_doses"]
            entry = next(e for e in blocks if e["name"] == operation["name"])
            entry["birth_year_start"] = operation.get("birth_year_start")
            entry["birth_year_end"] = operation.get("birth_year_end")
        elif kind == "set_coverage_reference":
            blocks = out["infant_doses"] if operation["block"] == "infant_doses" else out["cohort_doses"]
            entry = next(e for e in blocks if e["name"] == operation["name"])
            entry["coverage"] = operation["coverage"]
        else:
            raise ValueError(f"unsupported programme sensitivity operation {kind!r}")
    return out


def load_programme(path: Path, variant: str | None = None) -> Programme:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = _apply_variant(raw, variant)
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA}")
    sources = raw.get("sources") or {}
    for key, ref in sources.items():
        if not ref.get("citation"):
            raise ValueError(f"source {key} needs a citation")
    iso3 = raw["iso3"]
    products = []
    for e in raw["primary_product"]:
        if e["product"] not in PRODUCTS:
            raise ValueError(f"product must be one of {sorted(PRODUCTS)}")
        _source_refs(e, sources, f"primary_product {e.get('start')}")
        products.append((_date(e["start"]), PRODUCTS[e["product"]]))
    products.sort()
    series: dict[str, CoverageSeries] = {}
    for name, spec in (raw.get("coverage_series") or {}).items():
        if isinstance(spec, dict) and "derived_mean" in spec:
            series[name] = CoverageSeries(name, [], iso3, derived=tuple(spec["derived_mean"]))
        else:
            series[name] = CoverageSeries(name, _parse_segments(name, spec, sources), iso3)
    infant = _parse_doses(raw.get("infant_doses") or [], INFANT_KINDS, "age_months", sources)
    cohort = _parse_doses(raw.get("cohort_doses") or [], COHORT_KINDS, "age_years", sources)
    for d in infant:
        if not 0 <= d.age < 24:
            raise ValueError(f"infant dose {d.name}: age_months must be in [0, 24)")
    for d in cohort:
        if d.age < MIN_COHORT_DOSE_AGE:
            raise ValueError(f"cohort dose {d.name}: ages below {MIN_COHORT_DOSE_AGE} y belong in infant_doses")
    for d in (*infant, *cohort):
        if d.coverage not in series:
            raise ValueError(f"dose {d.name} references unknown coverage series {d.coverage}")
    maternal = []
    for e in raw.get("maternal") or []:
        _source_refs(e, sources, f"maternal {e.get('start')}")
        share = float(e.get("effective_share", 0.9))
        if not 0 <= share <= 1:
            raise ValueError("maternal effective_share must be in [0, 1]")
        if e["coverage"] not in series:
            raise ValueError(f"maternal period references unknown series {e['coverage']}")
        maternal.append(MaternalPeriod(_date(e["start"]), _date(e.get("end")), e["coverage"], share))
    timeliness = raw.get("timeliness") or {}
    _source_refs(timeliness, sources, "timeliness")
    pmf = np.asarray(timeliness.get("delay_pmf_months", [1.0]), dtype=float)
    if pmf.ndim != 1 or np.any(pmf < 0) or not np.isclose(pmf.sum(), 1.0):
        raise ValueError("timeliness.delay_pmf_months must be non-negative and sum to 1")
    return Programme(
        country=raw["country"], iso3=iso3, status=str(raw.get("status", "provisional")),
        products=products, coverage_series=series, infant_doses=infant, cohort_doses=cohort,
        maternal=maternal, delay_pmf=pmf, sources=sources, path=Path(path),
    )


def delay_fractions(coverage: float, pmf: np.ndarray) -> np.ndarray:
    """Conditional fractions ``h_d`` so that a share ``coverage * pmf[d]`` of the
    cohort is dosed ``d`` months late, applied to those not yet dosed."""

    out = np.zeros(pmf.size)
    done = 0.0
    for d, p in enumerate(pmf):
        remaining = 1.0 - done
        out[d] = coverage * p / remaining if remaining > 0 else 0.0
        done += coverage * p
    return np.clip(out, 0.0, 1.0)


@dataclass
class ProgrammeTables:
    infant_dose: np.ndarray
    primary_product: np.ndarray
    events: DoseEvents
    maternal_effective: np.ndarray
    audit: pd.DataFrame


def encode(
    programme: Programme,
    grid: AgeGrid,
    calendar: Calendar,
    freeze_before: date,
    delay_pmf: np.ndarray | None = None,
) -> ProgrammeTables:
    pmf = programme.delay_pmf if delay_pmf is None else np.asarray(delay_pmf, dtype=float)
    nm = calendar.n_months
    product = np.zeros(nm, dtype=np.int64)
    audit_rows: list[dict] = []
    month_plans: list[dict[str, tuple[int, float]]] = []
    for mi in range(nm):
        year = calendar.first_year + mi // 12
        when = max(date(year, mi % 12 + 1, 1), freeze_before)
        product[mi] = programme.product_at(when)
        month_plans.append(_infant_plan(programme, when))
    infant, delivered = _track_infant_targets(month_plans, pmf, grid.n_monthly)
    days, bins, kinds, fracs = [], [], [], []
    for year in range(calendar.first_year, calendar.first_year + calendar.n_years):
        for dose in programme.cohort_doses:
            k = int(np.floor(dose.age + 0.5))
            f = dose.age + 0.5 - k
            day = calendar.year_start_day(year) + int(round(f * 365.25))
            day = min(day, calendar.year_start_day(year) + 364)
            when = max(calendar.dates[day].astype(object), freeze_before)
            if not dose.active(when) or k > grid.cohort_last:
                continue
            # Cohort bin k holds the birth year Y - k throughout calendar year Y.
            birth_year = year - k
            if not dose.applies_to_birth_year(birth_year):
                continue
            cov = programme.coverage(dose.coverage, when.year)
            if cov <= 0:
                continue
            days.append(day)
            bins.append(grid.cohort_bin(k))
            kinds.append(COHORT_KINDS[dose.kind])
            fracs.append(cov)
            if year >= freeze_before.year:
                audit_rows.append({"year": year, "dose": dose.name, "kind": dose.kind, "age_years": dose.age, "coverage": cov})
    events = DoseEvents(
        np.asarray(days, dtype=np.int64), np.asarray(bins, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64), np.asarray(fracs, dtype=float),
    )
    maternal = np.zeros(calendar.n_days)
    effective_dates = np.maximum(calendar.dates, np.datetime64(freeze_before, "D"))
    effective_years = effective_dates.astype("datetime64[Y]").astype(int) + 1970
    for period in programme.maternal:
        on = effective_dates >= np.datetime64(period.start, "D")
        if period.end is not None:
            on &= effective_dates < np.datetime64(period.end, "D")
        for year in np.unique(effective_years[on]):
            value = programme.coverage(period.coverage, int(year)) * period.effective_share
            maternal[on & (effective_years == year)] = value
    for year in range(max(freeze_before.year, calendar.first_year), calendar.first_year + calendar.n_years):
        mi = (year - calendar.first_year) * 12
        for name, slot in INFANT_KINDS.items():
            dose = next((d for d in programme.infant_doses if d.kind == name and d.active(date(year, 1, 1))), None)
            if dose:
                audit_rows.append({
                    "year": year, "dose": dose.name, "kind": name, "age_months": dose.age,
                    "coverage": programme.coverage(dose.coverage, year),
                    "product": "aP" if product[mi] == PRODUCT_AP else "wP",
                    # Share of the cohort born in January of this year that has
                    # received the dose by 24 months (absolute, all doses).
                    "delivered_by_24m": float(delivered[mi, slot]),
                })
    return ProgrammeTables(infant, product, events, maternal, pd.DataFrame(audit_rows))


def _infant_plan(programme: Programme, when: date) -> dict[str, tuple[int, float]]:
    """Active infant doses at a date: kind -> (scheduled month bin, absolute coverage)."""

    plan: dict[str, tuple[int, float]] = {}
    for dose in programme.infant_doses:
        if not dose.active(when):
            continue
        if dose.kind in plan:
            raise ValueError(f"two active {dose.kind} doses on {when}")
        plan[dose.kind] = (int(round(dose.age)), programme.coverage(dose.coverage, when.year))
    for later, earlier in (("dose2", "dose1"), ("dose3", "dose2")):
        if later in plan:
            prior = plan.get(earlier, (0, 0.0))[1]
            if plan[later][1] > prior + 1e-9:
                raise ValueError(f"{later} coverage exceeds {earlier} on {when}")
    return plan


def _track_infant_targets(
    month_plans: list[dict[str, tuple[int, float]]],
    pmf: np.ndarray,
    n_monthly: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional dose fractions that make every birth cohort reach its targets.

    Each birth cohort is followed through the monthly bins with the kernel's
    dose moves (ignoring infection and waning, which do not change dose
    counts).  The target for dose ``k`` at ``d`` months after its scheduled
    month is ``coverage_k * F(d)`` (``F`` the cumulative delay distribution);
    each month delivers the gap between the target and what the cohort has
    already received, among those eligible.  Late first doses therefore still
    receive later doses instead of missing a window, and doses scheduled in the
    same month are delivered in the next month.  Kinds are applied in the
    kernel's order (toddler booster, dose 3, dose 2, dose 1).

    Returns ``(fractions, delivered)``: fractions ``(n_months, n_monthly, 4)``
    and, per calendar month of birth, the absolute share of that cohort that
    has received each kind by the last monthly bin.
    """

    nm = len(month_plans)
    cdf = np.cumsum(pmf)
    out = np.zeros((nm, n_monthly, N_INFANT_DOSE_KINDS))
    delivered = np.zeros((nm, N_INFANT_DOSE_KINDS))
    toddler = INFANT_KINDS["toddler_booster"]
    for birth in range(-n_monthly, nm):
        # n[j]: share with exactly j primary doses (3 = primed); booster: share boosted.
        n = np.array([1.0, 0.0, 0.0, 0.0])
        boosted = 0.0
        for m in range(n_monthly):
            mi = birth + m
            if mi >= nm:
                break
            # Cohorts born before the calendar start follow the first month's
            # programme, as the spin-up does.
            plan = month_plans[max(mi, 0)]
            for kind in ("toddler_booster", "dose3", "dose2", "dose1"):
                if kind not in plan:
                    continue
                age, coverage = plan[kind]
                d = m - age
                if d < 0:
                    continue
                # A cohort is eligible only if the dose was in the programme in
                # the month it was due, and its target is that month's
                # coverage: introducing a dose or raising coverage does not
                # trigger catch-up for cohorts already past the scheduled age.
                scheduled = max(birth + age, 0)
                if scheduled < nm:
                    due_plan = month_plans[scheduled]
                    if kind not in due_plan:
                        continue
                    coverage = due_plan[kind][1]
                target = coverage * cdf[min(d, cdf.size - 1)]
                if kind == "toddler_booster":
                    # Boosters reach only children with at least one dose; the
                    # kernel cannot tell boosted from not-yet-boosted children,
                    # so the delay distribution is applied to that whole pool.
                    eligible = n[1:].sum()
                    if eligible <= 1e-15:
                        continue
                    share = min(coverage / eligible, 1.0)
                    done = share * (cdf[d - 1] if 0 < d <= cdf.size else (1.0 if d > cdf.size else 0.0))
                    step = share * (pmf[d] if d < pmf.size else 0.0)
                    h = step / (1.0 - done) if done < 1.0 else 0.0
                    h = min(max(h, 0.0), 1.0)
                    if h > 0:
                        boosted += h * (eligible - boosted) if eligible > boosted else 0.0
                        # The booster advances incomplete series by one dose.
                        n = np.array([n[0], n[1] * (1 - h), n[2] * (1 - h) + n[1] * h, n[3] + n[2] * h])
                        if mi >= 0:
                            out[mi, m, toddler] = h
                    continue
                k = INFANT_KINDS[kind] + 1          # number of doses after this dose
                achieved = n[k:].sum()
                gap = target - achieved
                eligible = n[k - 1]
                h = gap / eligible if gap > 0 and eligible > 1e-15 else 0.0
                h = min(max(h, 0.0), 1.0)
                if h > 0:
                    moved = h * n[k - 1]
                    n[k - 1] -= moved
                    n[k] += moved
                    if mi >= 0:
                        out[mi, m, INFANT_KINDS[kind]] = h
        if 0 <= birth < nm:
            delivered[birth] = [n[1:].sum(), n[2:].sum(), n[3], boosted]
    return out, delivered
