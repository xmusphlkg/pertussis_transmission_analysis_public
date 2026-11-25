"""Core2026 surveillance observation model.

The simulator accumulates typical and mild symptomatic onsets in its model age
bins.  This module supplies the other half of the M3 contract:

* validated native case intervals and annual age-composition observations;
* fixed-delay projection into six reporting age bands;
* NB2 likelihood for interval totals;
* Dirichlet-multinomial likelihood for annual known-age counts; and
* a logit random-walk penalty for the time-invariant age reporting profile.

All functions are deterministic and side-effect free.  They accept arrays so
the same implementation can be used by optimisation, Laplace approximations,
and posterior diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import gammaln, expit

from ..config import REPO_ROOT
from ..grid import AgeGrid, band_matrix


AGE_BANDS: tuple[str, ...] = ("<1", "1-4", "5-9", "10-19", "20-59", "60+")
DEFAULT_CASE_DIRECTORY = REPO_ROOT / "data" / "processed" / "analysis_observations"
DEFAULT_AGE_PATH = REPO_ROOT / "data" / "processed" / "pertussis_age_case_observations_source_audited.csv"


def _as_float_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _check_counts(value: Any, *, name: str) -> np.ndarray:
    array = _as_float_array(value, name=name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _resolve_path(path_or_name: str | Path, directory: Path | None = None) -> Path:
    path = Path(path_or_name)
    if path.exists():
        return path
    if directory is not None:
        candidate = Path(directory) / path
        if candidate.exists():
            return candidate
        if path.suffix == "":
            candidate = Path(directory) / f"{path.name}.csv"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def _validate_intervals(frame: pd.DataFrame, *, count_column: str) -> pd.DataFrame:
    required = {"period_start", "period_end", count_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"case observations missing columns: {missing}")
    out = frame.copy()
    out["period_start"] = pd.to_datetime(out["period_start"], errors="coerce")
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    if out[["period_start", "period_end"]].isna().any().any():
        raise ValueError("case observation intervals contain invalid dates")
    out[count_column] = pd.to_numeric(out[count_column], errors="coerce")
    if out[count_column].isna().any() or (out[count_column] < 0).any():
        raise ValueError(f"{count_column} must contain finite non-negative counts")
    if (out["period_end"] <= out["period_start"]).any():
        raise ValueError("case observation intervals must have positive duration")
    ordered = out.sort_values(["period_start", "period_end"], kind="stable")
    if (ordered["period_start"].iloc[1:].to_numpy() < ordered["period_end"].iloc[:-1].to_numpy()).any():
        raise ValueError("case observation intervals overlap")
    out["interval_days"] = (out["period_end"] - out["period_start"]).dt.total_seconds() / 86400.0
    out["observed_year"] = out["period_start"].dt.year.astype(int)
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class CaseObservations:
    """Native surveillance intervals for one country."""

    country: str
    intervals: pd.DataFrame
    count_column: str = "reported_cases"

    def __post_init__(self) -> None:
        if not str(self.country).strip():
            raise ValueError("country must be non-empty")
        validated = _validate_intervals(self.intervals, count_column=self.count_column)
        object.__setattr__(self, "intervals", validated)

    @property
    def counts(self) -> np.ndarray:
        return self.intervals[self.count_column].to_numpy(dtype=float)

    @property
    def starts(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.intervals["period_start"])

    @property
    def ends(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.intervals["period_end"])


@dataclass(frozen=True)
class AgeCompositionObservations:
    """Annual known-age counts in the model reporting bands."""

    country: str
    counts: pd.DataFrame
    year_column: str = "year"
    band_column: str = "reporting_band"
    count_column: str = "reported_cases"
    band_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        frame = self.counts.copy()
        required = {self.year_column, self.band_column, self.count_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"age observations missing columns: {missing}")
        frame[self.year_column] = pd.to_numeric(frame[self.year_column], errors="coerce")
        frame[self.count_column] = pd.to_numeric(frame[self.count_column], errors="coerce")
        if frame[[self.year_column, self.count_column]].isna().any().any():
            raise ValueError("age observations contain invalid year or count")
        if (frame[self.count_column] < 0).any():
            raise ValueError("age counts must be non-negative")
        duplicate = frame.duplicated([self.year_column, self.band_column])
        if duplicate.any():
            raise ValueError("age observations have duplicate year/band rows")
        object.__setattr__(self, "counts", frame.reset_index(drop=True))
        labels = tuple(self.band_labels or frame[self.band_column].astype(str).drop_duplicates().tolist())
        if not labels:
            raise ValueError("age observations require at least one age band")
        missing_labels = set(frame[self.band_column].astype(str)) - set(labels)
        if missing_labels:
            raise ValueError(f"band_labels do not cover observations: {sorted(missing_labels)}")
        object.__setattr__(self, "band_labels", labels)

    def as_matrix(self, years: Iterable[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(sorted(set(int(y) for y in years)), dtype=int) if years is not None else np.asarray(sorted(self.counts[self.year_column].unique()), dtype=int)
        matrix = np.zeros((requested.size, len(self.band_labels)), dtype=float)
        label_index = {label: i for i, label in enumerate(self.band_labels)}
        year_index = {int(year): i for i, year in enumerate(requested)}
        for row in self.counts.itertuples(index=False):
            year = int(getattr(row, self.year_column))
            if year not in year_index:
                continue
            matrix[year_index[year], label_index[str(getattr(row, self.band_column))]] = float(getattr(row, self.count_column))
        return requested, matrix


def load_case_observations(country_or_path: str | Path, *, directory: Path | None = None) -> CaseObservations:
    """Load a validated native case stream by country name or CSV path."""

    path = _resolve_path(country_or_path, directory or DEFAULT_CASE_DIRECTORY)
    frame = pd.read_csv(path)
    country = path.stem if path.stem else str(country_or_path)
    requested = str(country_or_path)
    if Path(requested).suffix == "" and requested == path.stem:
        country = requested
    elif "Country" in frame.columns and frame["Country"].notna().any():
        country = str(frame["Country"].dropna().iloc[0])
    if "reported_cases" not in frame.columns and "Cases" in frame.columns:
        frame = frame.rename(columns={"Cases": "reported_cases"})
    return CaseObservations(country=country, intervals=frame)


def _normalize_age_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert audited source age rows to the six M3 bands.

    Rows can already contain ``reporting_band`` or can identify a source age
    interval via ``age_start_years`` and ``age_end_years_exclusive``.  Partial
    overlaps are rejected instead of silently assigning an observation to a
    wrong band.
    """

    out = frame.copy()
    if "reporting_band" in out.columns:
        return out
    if "source_age_group" in out.columns:
        out["reporting_band"] = out["source_age_group"].astype(str)
        return out
    if "normalized_age_group" in out.columns:
        out["reporting_band"] = out["normalized_age_group"].astype(str)
        return out
    if not {"age_start_years", "age_end_years_exclusive"}.issubset(out.columns):
        raise ValueError("age observations require reporting_band or age bounds")
    bounds = ((0.0, 1.0, "<1"), (1.0, 5.0, "1-4"), (5.0, 10.0, "5-9"), (10.0, 20.0, "10-19"), (20.0, 60.0, "20-59"), (60.0, np.inf, "60+"))
    labels: list[str] = []
    for lo, hi in zip(out["age_start_years"], out["age_end_years_exclusive"]):
        matches = [name for left, right, name in bounds if float(lo) == left and (np.isinf(right) and np.isinf(float(hi)) or float(hi) == right)]
        labels.append(matches[0] if len(matches) == 1 else f"{float(lo):g}-{float(hi):g}")
    out["reporting_band"] = labels
    return out


def load_age_compositions(country_or_path: str | Path | None = None, *, path: Path | None = None) -> AgeCompositionObservations | dict[str, AgeCompositionObservations]:
    """Load annual age observations.

    With a country or CSV path, return one object.  With no argument, load the
    audited repository-wide age table and return one object per country.
    """

    chosen = path if path is not None else (country_or_path if country_or_path is not None and Path(country_or_path).suffix else DEFAULT_AGE_PATH)
    frame = pd.read_csv(_resolve_path(chosen))
    frame = _normalize_age_rows(frame)
    if "reported_cases" not in frame.columns:
        raise ValueError("age observations require reported_cases")
    if country_or_path is not None and Path(str(country_or_path)).suffix == "":
        key = str(country_or_path)
        if "country" in frame.columns:
            mask = frame["country"].astype(str).isin({key}) | frame.get("iso3", pd.Series(index=frame.index, dtype=str)).astype(str).eq(key)
            frame = frame.loc[mask].copy()
        if frame.empty:
            raise KeyError(f"no age observations for {key}")
        return AgeCompositionObservations(key, frame)
    if "country" not in frame.columns:
        return AgeCompositionObservations("unknown", frame)
    return {str(country): AgeCompositionObservations(str(country), group) for country, group in frame.groupby("country", sort=True)}


# Age data excluded from the likelihood.  Japan before 2018 comes from
# paediatric sentinel surveillance rather than all-facility notification.
# New Zealand 2021 has no 15-19 group in the source series, so its known-age
# rows do not cover all ages; the gap is a source gap, not a model choice.
EXCLUDED_AGE_YEARS: dict[str, tuple[int, ...]] = {
    "Japan": (2015, 2016, 2017),
    "New_Zealand": (2021,),
}
# Source groups are merged above this age because the model's open bin has no
# age detail beyond it.
OPEN_AGE_FLOOR = 70.0


@dataclass(frozen=True)
class SourceAgeYear:
    """Known-age counts for one country-year in that year's own source groups.

    ``edges`` are the ``[lower, upper)`` completed-age intervals in years after
    merging every group that starts at or above ``OPEN_AGE_FLOOR`` (or straddles
    it) into one open group.  Unknown-age rows are dropped.
    """

    country: str
    year: int
    labels: tuple[str, ...]
    edges: tuple[tuple[float, float], ...]
    counts: np.ndarray

    def mapping(self, grid: AgeGrid, year_fraction: float = 0.5) -> np.ndarray:
        """``(n_bins, n_groups)`` shares of each model bin's events per group."""

        return band_matrix(grid, list(self.edges), year_fraction)


def source_age_years(
    country: str,
    *,
    path: Path | None = None,
    excluded: dict[str, tuple[int, ...]] | None = None,
) -> list[SourceAgeYear]:
    """Load one country's annual known-age counts on each year's native groups.

    Every year's known-age rows must partition ``[0, inf)`` exactly; groups with
    a lower bound at or above ``OPEN_AGE_FLOOR`` (or crossing it) are merged into
    a single ``70+``-type group so the model can map onto it.
    """

    frame = pd.read_csv(path or DEFAULT_AGE_PATH)
    frame = frame[frame["country"].astype(str) == country]
    if frame.empty:
        raise KeyError(f"no age observations for {country}")
    skip = set((excluded if excluded is not None else EXCLUDED_AGE_YEARS).get(country, ()))
    out: list[SourceAgeYear] = []
    for year, rows in frame.groupby("year", sort=True):
        if int(year) in skip:
            continue
        rows = rows[~rows["is_unknown_age"].astype(bool)]
        lo = rows["age_start_years"].astype(float).to_numpy()
        hi = rows["age_end_years_exclusive"].astype(float).to_numpy()
        order = np.argsort(lo)
        lo, hi = lo[order], hi[order]
        labels = rows["source_age_group"].astype(str).to_numpy()[order]
        counts = rows["reported_cases"].astype(float).to_numpy()[order]
        if lo[0] != 0.0 or not np.isinf(hi[-1]) or np.any(lo[1:] != hi[:-1]):
            raise ValueError(f"{country} {year}: known-age groups do not partition all ages")
        keep = lo < OPEN_AGE_FLOOR
        tail = ~keep | (hi > OPEN_AGE_FLOOR)
        tail_start = float(lo[tail].min())
        edges = [(float(a), float(b)) for a, b in zip(lo[~tail], hi[~tail])] + [(tail_start, np.inf)]
        merged_labels = [str(x) for x in labels[~tail]] + [f"{tail_start:g}+"]
        merged_counts = np.concatenate([counts[~tail], [counts[tail].sum()]])
        out.append(SourceAgeYear(country, int(year), tuple(merged_labels), tuple(edges), merged_counts))
    return out


def model_band_matrix(
    grid: AgeGrid,
    year_fraction: float = 0.5,
    bands: tuple[tuple[float, float], ...] = ((0, 1), (1, 5), (5, 10), (10, 20), (20, 60), (60, np.inf)),
) -> np.ndarray:
    """``(n_bins, 6)`` shares of each model bin's events in the reporting bands.

    These are the bands on which the age-specific reporting probabilities are
    defined, not the bands the source publishes.
    """

    return band_matrix(grid, list(bands), year_fraction)


def expected_source_group_counts(
    onset_typical: Any,
    onset_mild: Any,
    *,
    model_bands: np.ndarray,
    source_bands: np.ndarray,
    params: "ObservationParameters",
    detection_multiplier: Any = 1.0,
    catchment_fraction: float | None = None,
) -> np.ndarray:
    """Expected reported counts in a country-year's own source age groups.

    ``onset_typical`` and ``onset_mild`` are ``(n_rows, n_bins)`` biological
    onsets; ``model_bands`` maps bins to the six reporting bands and
    ``source_bands`` maps bins to that year's published groups.  The reporting
    probability of each model band is spread over the source groups in
    proportion to the bin shares they contain, so a band that cuts across a
    model reporting band (for example 15-24 against 10-19 and 20-59) receives
    the corresponding mix.
    """

    typical = _check_counts(onset_typical, name="onset_typical")
    mild = _check_counts(onset_mild, name="onset_mild")
    if typical.shape != mild.shape or typical.ndim != 2:
        raise ValueError("onset arrays must have the same shape (rows, bins)")
    model = _as_float_array(model_bands, name="model_bands", ndim=2)
    source = _as_float_array(source_bands, name="source_bands", ndim=2)
    if model.shape[0] != typical.shape[1] or source.shape[0] != typical.shape[1]:
        raise ValueError("band matrices must have one row per model bin")
    if not np.allclose(model.sum(axis=1), 1.0) or not np.allclose(source.sum(axis=1), 1.0):
        raise ValueError("band matrices must give each bin unit total weight")
    rates = reporting_probabilities(params.reporting_logits)
    true_bins = typical + float(params.mild_reporting_ratio) * mild
    # bin -> source group, weighted by the reporting probability of the model
    # band the bin belongs to.
    bin_weight = model @ rates
    weighted = true_bins * bin_weight[None, :]
    detection = _as_float_array(detection_multiplier, name="detection_multiplier")
    if detection.size not in (1, typical.shape[0]):
        raise ValueError("detection_multiplier must be scalar or one per row")
    validate_effective_reporting_probability(rates, detection)
    scale = float(detection.item()) if detection.size == 1 else detection[:, None]
    fraction = float(params.catchment_fraction if catchment_fraction is None else catchment_fraction)
    return (weighted * scale * fraction) @ source


def negative_binomial_nll(observed: Any, mean: Any, size: Any) -> float:
    """NB2 negative log likelihood, with variance ``mean + mean² / size``."""

    y, mu, r = np.broadcast_arrays(_check_counts(observed, name="observed"), _as_float_array(mean, name="mean"), _as_float_array(size, name="size"))
    if np.any(mu < 0.0) or np.any(r <= 0.0):
        raise ValueError("mean must be non-negative and size must be positive")
    mu_safe = np.maximum(mu, 1e-12)
    p = r / (r + mu_safe)
    logp = gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) + r * np.log(p) + y * np.log1p(-p)
    logp = np.where((y == 0.0) & (mu == 0.0), 0.0, logp)
    return float(-np.sum(logp))


def dirichlet_multinomial_nll(observed: Any, probabilities: Any, concentration: Any) -> float:
    """Dirichlet-multinomial NLL for one or more count vectors."""

    counts = _check_counts(observed, name="observed")
    probs = _as_float_array(probabilities, name="probabilities")
    if counts.ndim == 1:
        counts = counts[None, :]
    if probs.ndim == 1:
        probs = probs[None, :]
    if counts.shape != probs.shape:
        raise ValueError("observed and probabilities must have the same shape")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")
    row_sums = probs.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("each probability row must have positive mass")
    probs = probs / row_sums[:, None]
    kappa = np.asarray(concentration, dtype=float)
    if not np.isfinite(kappa).all() or np.any(kappa <= 0.0):
        raise ValueError("concentration must be finite and positive")
    kappa = np.broadcast_to(kappa, (counts.shape[0],))
    total = counts.sum(axis=1)
    alpha = probs * kappa[:, None]
    # A group with zero expected probability contributes nothing when it is
    # also unobserved, and makes the composition impossible when it is not.
    positive = alpha > 0.0
    if np.any(~positive & (counts > 0.0)):
        return float("inf")
    safe = np.where(positive, alpha, 1.0)
    group = np.where(positive, gammaln(safe + counts) - gammaln(safe), 0.0) - gammaln(counts + 1.0)
    logp = gammaln(total + 1.0) + gammaln(kappa) - gammaln(kappa + total) + np.sum(group, axis=1)
    return float(-np.sum(logp))


def reporting_probabilities(logits: Any) -> np.ndarray:
    """Map reporting logits to probabilities in (0, 1)."""

    values = _as_float_array(logits, name="reporting logits")
    return expit(values)


class InvalidReportingProbability(ValueError):
    """A reporting probability times a detection multiplier exceeds one."""


def validate_effective_reporting_probability(rates: Any, detection: Any) -> None:
    """Reject, rather than clip, an impossible effective reporting probability."""
    probabilities = _as_float_array(rates, name="reporting probabilities")
    multipliers = _as_float_array(detection, name="detection multiplier")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)) or np.any(multipliers <= 0.0):
        raise InvalidReportingProbability("reporting probabilities and detection multipliers are outside their domains")
    if probabilities.size and multipliers.size and float(probabilities.max()) * float(multipliers.max()) > 1.0 + 1e-12:
        raise InvalidReportingProbability("effective reporting probability exceeds one")


def logit_random_walk_nll(logits: Any, *, first_mean: float = -1.4, first_sd: float = 1.5, step_sd: float = 0.7) -> float:
    """Gaussian prior penalty for the time-invariant age-reporting profile."""

    values = _as_float_array(logits, name="reporting logits")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("reporting logits must be a non-empty vector")
    if first_sd <= 0.0 or step_sd <= 0.0:
        raise ValueError("random-walk standard deviations must be positive")
    first = ((values[0] - first_mean) / first_sd) ** 2 + 2.0 * np.log(first_sd)
    if values.size == 1:
        return float(0.5 * first)
    steps = np.diff(values)
    return float(0.5 * (first + np.sum((steps / step_sd) ** 2 + 2.0 * np.log(step_sd))))


@dataclass(frozen=True)
class ObservationParameters:
    """Observation-only parameters used by :func:`project_reported_cases`."""

    reporting_logits: np.ndarray
    mild_reporting_ratio: float = 0.05
    detection_multiplier: float | np.ndarray = 1.0
    onset_to_notification_days: float = 0.0
    catchment_fraction: float = 1.0

    def __post_init__(self) -> None:
        logits = _as_float_array(self.reporting_logits, name="reporting_logits", ndim=1)
        if logits.size != len(AGE_BANDS):
            raise ValueError(f"reporting_logits must have {len(AGE_BANDS)} entries")
        if not 0.0 <= self.mild_reporting_ratio <= 1.0:
            raise ValueError("mild_reporting_ratio must lie in [0, 1]")
        if self.onset_to_notification_days < 0.0 or self.catchment_fraction <= 0.0:
            raise ValueError("delay must be non-negative and catchment_fraction positive")
        detection = _as_float_array(self.detection_multiplier, name="detection_multiplier")
        if np.any(detection <= 0.0):
            raise ValueError("detection_multiplier must be positive")
        object.__setattr__(self, "reporting_logits", logits)
        object.__setattr__(self, "detection_multiplier", detection)


@dataclass(frozen=True)
class ObservationProjection:
    """Projected reported means, separated by age band and total."""

    interval_age_mean: np.ndarray
    interval_total_mean: np.ndarray
    reporting_probabilities: np.ndarray


def _numeric_days(values: Iterable[Any], *, origin: pd.Timestamp | None = None) -> np.ndarray:
    values_array = np.asarray(values)
    is_datetime = np.issubdtype(values_array.dtype, np.datetime64)
    if values_array.dtype.kind in {"O", "U", "S"}:
        parsed = pd.to_datetime(values_array, errors="coerce")
        is_datetime = not parsed.isna().any()
    if is_datetime:
        dates = pd.to_datetime(values)
        base = dates.min() if origin is None else pd.Timestamp(origin)
        return (dates - base).total_seconds().to_numpy(dtype=float) / 86400.0
    array = np.asarray(values, dtype=float)
    return _as_float_array(array, name="interval times")


def _shift_to_target(source_counts: np.ndarray, source_starts: np.ndarray, source_ends: np.ndarray, target_starts: np.ndarray, target_ends: np.ndarray, delay: float) -> np.ndarray:
    out = np.zeros((target_starts.size, source_counts.shape[1]), dtype=float)
    for i, (start, end) in enumerate(zip(source_starts + delay, source_ends + delay)):
        duration = end - start
        if duration <= 0.0:
            raise ValueError("source intervals must have positive duration")
        for j, (left, right) in enumerate(zip(target_starts, target_ends)):
            overlap = max(0.0, min(end, right) - max(start, left))
            if overlap:
                out[j] += source_counts[i] * overlap / duration
    return out


def project_reported_cases(
    onset_typical: Any,
    onset_mild: Any,
    *,
    source_starts: Iterable[Any],
    source_ends: Iterable[Any],
    target_starts: Iterable[Any],
    target_ends: Iterable[Any],
    grid: AgeGrid | None = None,
    bin_to_band: np.ndarray | None = None,
    parameters: ObservationParameters,
) -> ObservationProjection:
    """Project true onsets from model bins into reported observation intervals.

    Counts are assumed uniform within each source interval for the fixed-delay
    overlap calculation.  ``bin_to_band`` is a ``(n_bins, 6)`` matrix; if it
    is omitted, it is constructed from the core2026 :class:`AgeGrid` at the
    midpoint of each source interval.
    """

    typical = _check_counts(onset_typical, name="onset_typical")
    mild = _check_counts(onset_mild, name="onset_mild")
    if typical.shape != mild.shape or typical.ndim != 2:
        raise ValueError("onset arrays must have the same shape (interval, bin)")
    source_starts = list(source_starts)
    source_ends = list(source_ends)
    target_starts = list(target_starts)
    target_ends = list(target_ends)
    all_time_values = source_starts + source_ends + target_starts + target_ends
    datetime_values = pd.to_datetime(all_time_values, errors="coerce")
    common_origin = datetime_values.min() if not datetime_values.isna().any() else None
    starts = _numeric_days(source_starts, origin=common_origin)
    ends = _numeric_days(source_ends, origin=common_origin)
    target_s = _numeric_days(target_starts, origin=common_origin)
    target_e = _numeric_days(target_ends, origin=common_origin)
    if starts.size != typical.shape[0] or ends.size != typical.shape[0]:
        raise ValueError("source interval count does not match onset arrays")
    if target_s.size != target_e.size or np.any(target_e <= target_s):
        raise ValueError("target intervals must have positive duration")
    if np.any(ends <= starts):
        raise ValueError("source intervals must have positive duration")
    if bin_to_band is None:
        if grid is None:
            grid = AgeGrid()
        midpoint_fraction = 0.5
        bin_to_band = band_matrix(grid, [(0, 1), (1, 5), (5, 10), (10, 20), (20, 60), (60, np.inf)], midpoint_fraction)
    mapping = _as_float_array(bin_to_band, name="bin_to_band", ndim=2)
    if mapping.shape != (typical.shape[1], len(AGE_BANDS)):
        raise ValueError("bin_to_band has incompatible shape")
    if np.any(mapping < 0.0):
        raise ValueError("bin_to_band must be non-negative")
    true_band = (typical + float(parameters.mild_reporting_ratio) * mild) @ mapping
    shifted = _shift_to_target(true_band, starts, ends, target_s, target_e, float(parameters.onset_to_notification_days))
    detection = np.asarray(parameters.detection_multiplier, dtype=float)
    rates = reporting_probabilities(parameters.reporting_logits)
    validate_effective_reporting_probability(rates, detection)
    if detection.size == 1:
        shifted *= float(detection.item())
    elif detection.size == target_s.size:
        shifted *= detection[:, None]
    else:
        raise ValueError("detection_multiplier must be scalar or one value per target interval")
    age_mean = shifted * rates[None, :] * float(parameters.catchment_fraction)
    return ObservationProjection(age_mean, age_mean.sum(axis=1), rates)
