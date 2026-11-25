"""Clinical outcomes cascade: confirmed cases, hospitalisations, deaths.

Interpretation limits
---------------------
The mapping converts modelled onsets to *expected* clinical outcomes under the
surveillance-reporting model.  It does not produce national fatality estimates
for countries whose mortality status is ``use_calibrated_reference_risk``; those
go through mortality_extrapolation instead.

The per-confirmed-case basis (hospitalisation_proportion, cfr) is applied only
to the *reported* (confirmed) fraction of onsets, not to all modelled onsets.
An explicit death-ascertainment parameter (default 1.0) controls what fraction
of deaths in unreported cases would be captured; 0.0 implements the
main-analysis assumption that unreported cases never contribute deaths.

Death ascertainment = proportion of the unreported-case CFR that is actually
observed in death registries.  Setting it to 0 gives the main analysis;
sensitivity runs use 0.05, 0.10.  The value is stored on every output row.

Column conventions
------------------
All output dataframes carry the provenance columns:
  profile, strategy_id, year (or age_bin), death_ascertainment
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

# Age bands matching the observation model (observation/model.py).
AGE_BANDS = ("<1", "1-4", "5-9", "10-19", "20-59", "60+")
# Public aliases used by downstream callers.
REPORTING_BANDS = AGE_BANDS
BAND_LABELS = AGE_BANDS
# Lower/upper completed-age bounds for the six reporting bands.
BAND_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 60.0), (60.0, float("inf"))
]


def reporting_probs_from_logits(logits) -> np.ndarray:
    """Convert the six fitted reporting logits to probabilities in (0, 1)."""
    return expit(np.asarray(logits, dtype=float))


def _bin_to_band_weights(grid, year_fraction: float = 0.5) -> np.ndarray:
    """Return (n_bins, 6) band-membership matrix from the age grid."""
    from ..grid import band_matrix
    return band_matrix(grid, BAND_BOUNDS, year_fraction)


def _build_age_overlap_matrix(
    severity_entries: list[dict],
    band_bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Return (n_bands, n_entries) overlap weights for severity age mapping.

    Each severity entry covers [age_start_years, age_end_years).  A band gets
    the fraction of that band's width covered by the entry.  Bands wider than
    one entry receive contributions from multiple entries.
    """
    n_bands = len(band_bounds)
    n_entries = len(severity_entries)
    weights = np.zeros((n_bands, n_entries), dtype=float)
    for j, entry in enumerate(severity_entries):
        lo = float(entry["age_start_years"])
        hi = entry.get("age_end_years", float("inf"))
        if hi is None or str(hi).strip().lower() in {"inf", "infinity", "null", "none", ""}:
            hi = float("inf")
        else:
            hi = float(hi)
        for i, (blo, bhi) in enumerate(band_bounds):
            effective_bhi = min(bhi, 100.0) if np.isinf(bhi) else bhi
            effective_hi = min(hi, 100.0) if np.isinf(hi) else hi
            overlap = max(0.0, min(effective_bhi, effective_hi) - max(blo, lo))
            band_width = max(effective_bhi - blo, 1e-12)
            weights[i, j] = overlap / band_width
    return weights


def bin_severity_weights(
    severity_entries: list[dict],
    *,
    grid=None,
) -> np.ndarray:
    """Return (n_bins,) severity weights by projecting entry age groups onto bins.

    Each entry must have a ``value`` key (set by PSA sampling) or fall back to
    ``point``.  The returned array is the per-reported-case rate at each bin.
    To obtain outcome counts multiply by confirmed_cases_per_bin.

    Parameters
    ----------
    severity_entries:
        List of severity entry dicts with age_start_years, age_end_years,
        and either 'value' (sampled) or 'point' (fixed).
    grid:
        AgeGrid; constructed from defaults if None.
    """
    if grid is None:
        from ..grid import AgeGrid
        grid = AgeGrid()
    btb = _bin_to_band_weights(grid)                             # (n_bins, 6)
    overlap = _build_age_overlap_matrix(severity_entries, BAND_BOUNDS)  # (6, n_entries)
    entry_vals = np.array(
        [float(e.get("value", e.get("point", 0.0))) for e in severity_entries],
        dtype=float,
    )
    band_prop = overlap @ entry_vals   # (6,)
    return btb @ band_prop             # (n_bins,)


def expected_confirmed_cases(
    onset_df: pd.DataFrame,
    *,
    reporting_logits,
    mild_reporting_ratio: float = 0.05,
    grid=None,
) -> pd.DataFrame:
    """Compute expected reported (confirmed) cases by age bin and year.

    Parameters
    ----------
    onset_df:
        Annual onset output, one row per (year, age_bin) with columns
        typical_onsets, mild_onsets, bin_index.  Must carry profile,
        strategy_id.
    reporting_logits:
        Six fitted reporting logits in REPORTING_BANDS order.
    mild_reporting_ratio:
        Share of mild onsets that are reported (default 0.05).
    grid:
        AgeGrid instance; constructed from defaults if None.

    Returns
    -------
    Copy of onset_df with ``reporting_prob`` and ``confirmed_cases`` added.
    """
    if grid is None:
        from ..grid import AgeGrid
        grid = AgeGrid()

    required = {"year", "age_bin", "bin_index", "typical_onsets", "mild_onsets",
                "profile", "strategy_id"}
    missing = required - set(onset_df.columns)
    if missing:
        raise ValueError(f"onset_df missing columns: {sorted(missing)}")

    probs = reporting_probs_from_logits(reporting_logits)   # (6,)
    btb = _bin_to_band_weights(grid)                        # (n_bins, 6)
    bin_prob = btb @ probs                                  # (n_bins,)

    out = onset_df.copy()
    bin_idx = out["bin_index"].to_numpy(int)
    if bin_idx.max() >= len(bin_prob):
        raise ValueError("bin_index exceeds grid size; check that grid matches onset_df")
    p = bin_prob[bin_idx]
    out["reporting_prob"] = p
    out["confirmed_cases"] = (
        out["typical_onsets"].to_numpy(float) * p
        + out["mild_onsets"].to_numpy(float) * float(mild_reporting_ratio) * p
    )
    return out


def reported_cases_from_onsets(
    onset_df: pd.DataFrame,
    reporting_logits,
    *,
    mild_reporting_ratio: float = 0.05,
    grid=None,
) -> pd.DataFrame:
    """Public alias for expected_confirmed_cases with positional logits argument."""
    return expected_confirmed_cases(
        onset_df,
        reporting_logits=reporting_logits,
        mild_reporting_ratio=mild_reporting_ratio,
        grid=grid,
    )


def apply_severity(
    confirmed_df: pd.DataFrame,
    *,
    severity_entries: list[dict],
    quantity: str,
    basis: str,
    death_ascertainment: float = 1.0,
    grid=None,
) -> pd.DataFrame:
    """Apply per-confirmed-case severity proportions to each age band.

    Parameters
    ----------
    confirmed_df:
        Output of expected_confirmed_cases (has confirmed_cases, bin_index).
    severity_entries:
        List of dicts from severity YAML, already single-draw (point or
        sampled value stored as 'value').  Each must have age_start_years,
        age_end_years, and point or value.
    quantity:
        One of hospitalisation_proportion, cfr, icu_given_hospital, los_days.
    basis:
        per_confirmed_case or per_hospitalised_case.
    death_ascertainment:
        For CFR: fraction of unreported cases' deaths captured in death
        registries.  Main analysis = 1.0.  Sensitivity = 0.05 or 0.10.
    grid:
        AgeGrid; constructed if None.

    Returns
    -------
    Copy of confirmed_df with ``outcome_count``, ``severity_prop``,
    ``quantity``, ``basis``, and ``death_ascertainment`` added.
    """
    if quantity not in ("hospitalisation_proportion", "cfr", "icu_given_hospital", "los_days"):
        raise ValueError(f"unknown quantity {quantity!r}")
    if basis not in ("per_confirmed_case", "per_hospitalised_case"):
        raise ValueError(f"unknown basis {basis!r}")

    if grid is None:
        from ..grid import AgeGrid
        grid = AgeGrid()

    bin_prop = bin_severity_weights(severity_entries, grid=grid)

    out = confirmed_df.copy()
    bin_idx = out["bin_index"].to_numpy(int)
    props = bin_prop[bin_idx]
    out["severity_prop"] = props
    out["death_ascertainment"] = float(death_ascertainment)

    if basis == "per_confirmed_case":
        out["outcome_count"] = out["confirmed_cases"].to_numpy(float) * props
        if quantity == "cfr" and float(death_ascertainment) < 1.0:
            typical = out["typical_onsets"].to_numpy(float)
            mild = out["mild_onsets"].to_numpy(float)
            rep = out["reporting_prob"].to_numpy(float)
            mild_ratio = float(out["mild_reporting_ratio"].iloc[0]
                               if "mild_reporting_ratio" in out.columns else 0.05)
            unreported = typical * (1.0 - rep) + mild * mild_ratio * (1.0 - rep)
            out["outcome_count"] = (
                out["outcome_count"].to_numpy(float)
                + unreported * props * float(death_ascertainment)
            )
    else:
        if "outcome_count" not in out.columns:
            raise ValueError(
                "per_hospitalised_case basis requires an 'outcome_count' column "
                "from a prior hospitalisation_proportion step"
            )
        out["outcome_count"] = out["outcome_count"].to_numpy(float) * props

    out["quantity"] = quantity
    out["basis"] = basis
    return out
