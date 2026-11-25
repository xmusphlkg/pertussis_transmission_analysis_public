"""Paired valuation of a nasal adjunct to retained maternal/Tdap programmes.

The caller supplies one setting, horizon, burden and hypothetical product
scenario at a time, including all available standard strategies and that
product's coverage alternatives. Draw rows must share the adopted clinical and
cost samples. ``base_cost`` includes treatment and ALL existing/additional
maternal and Tdap vaccine/service costs before adding nasal-product cost.

``nasal_doses`` contains discounted actual nasal administrations, not the
change in total vaccination receipts. A common all-in nasal price p gives
NHB = health - (base_cost + p * nasal_doses) / threshold. Standard arms have
zero nasal doses. Outcomes and costs share any one reference arm; their common
reference cancels in every strategy comparison. Functions never resample.

Production inputs contain 1,000 paired draws. ``expected_draws`` permits an
explicit alternative batch size while retaining strict two-dimensional shape
validation. Returns use lists of dictionaries for direct DataFrame conversion.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np


def _inputs(health, base_cost, nasal_doses, threshold, expected_draws):
    h, c, d = (np.asarray(x, dtype=float) for x in (health, base_cost, nasal_doses))
    if h.ndim != 2 or c.shape != h.shape or not all(h.shape):
        raise ValueError("health and base_cost must have the same nonempty (draw, strategy) shape")
    if not isinstance(expected_draws, (int, np.integer)) or isinstance(expected_draws, bool) or expected_draws < 1 or h.shape[0] != expected_draws:
        raise ValueError("draw count must equal positive expected_draws (default 1000)")
    if d.shape != (h.shape[1],):
        raise ValueError("nasal_doses must have one actual discounted dose count per strategy")
    if not all(np.isfinite(x).all() for x in (h, c, d)) or (d < 0).any():
        raise ValueError("finite outcomes/costs and finite nonnegative actual nasal doses required")
    if np.ndim(threshold) != 0 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    return h, c, d, float(threshold)


def _names(strategy_ids: Sequence[str], count):
    names = list(strategy_ids)
    if len(names) != count or len(set(names)) != count or any(not isinstance(a, str) or not a or "|" in a for a in names):
        raise ValueError("unique nonempty strategy IDs without '|' required, matching array columns")
    return names


def _tol(intercept, doses, price):
    return 128*np.finfo(float).eps*max(1., float(np.abs(intercept).max()), float(np.abs(price*doses).max()))


def _winners(names, intercept, doses, price):
    value = intercept-price*doses
    if not np.isfinite(value).all():
        raise ValueError("price is outside the finite numerical evaluation range")
    return "|".join(a for a, v in zip(names, value) if value.max()-v <= _tol(intercept, doses, price))


def _lines(strategy_ids, health, base_cost, nasal_doses, threshold, expected_draws):
    h, c, d, lam = _inputs(health, base_cost, nasal_doses, threshold, expected_draws)
    names = _names(strategy_ids, h.shape[1])
    draws = lam*h-c
    if not np.isfinite(draws).all():
        raise ValueError("nonfinite net monetary benefits")
    return names, draws.mean(axis=0), d, draws


def nhb_at_nasal_price(health, base_cost, nasal_doses, threshold, price, *, expected_draws=1000):
    """Return paired NHB array (draw, strategy); p=0 retains all base costs."""
    h, c, d, lam = _inputs(health, base_cost, nasal_doses, threshold, expected_draws)
    if np.ndim(price) != 0 or not np.isfinite(price) or price < 0:
        raise ValueError("nasal price must be finite and nonnegative")
    result = h-(c+float(price)*d[None, :])/lam
    if not np.isfinite(result).all():
        raise ValueError("nonfinite NHB at requested price")
    return result


def nasal_price_frontier(strategy_ids, health, base_cost, nasal_doses, threshold, *, expected_draws=1000):
    """Full expected-NMB upper envelope for p >= 0, preserving boundary ties.

    Row fields: price_lower, price_upper (infinity for the tail),
    optimal_strategies (pipe-separated interior optimum), ties_at_lower,
    ties_at_upper (empty for infinity). Boundaries include point-only ties.
    Differences between two actual nasal dose counts may have either sign.
    """
    names, b, d, _ = _lines(strategy_ids, health, base_cost, nasal_doses, threshold, expected_draws)
    cuts = [0.]
    for i in range(len(names)):
        for j in range(i):
            if d[i] != d[j]:
                cross = (b[i]-b[j])/(d[i]-d[j])
                if np.isfinite(cross) and cross > 0:
                    cuts.append(float(cross))
    cuts = sorted(set(cuts))+[np.inf]
    rows = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        probe = lo+max(1., abs(lo)) if np.isinf(hi) else lo+(hi-lo)/2
        best = _winners(names, b, d, probe)
        if rows and rows[-1]["optimal_strategies"] == best:
            rows[-1]["price_upper"] = hi
        else:
            rows.append(dict(price_lower=lo, price_upper=hi, optimal_strategies=best,
                             ties_at_lower=_winners(names, b, d, lo)))
    for row in rows:
        lo, hi = row["price_lower"], row["price_upper"]
        row["ties_at_upper"] = "" if np.isinf(hi) else _winners(names, b, d, hi)
        # An affine dominance inequality holds on an entire interval iff it
        # holds at both endpoints, or at the lower end and in the tail slope.
        for arm in row["optimal_strategies"].split("|"):
            i = names.index(arm)
            for p in (lo, hi):
                if np.isfinite(p):
                    values = b-p*d
                    if np.min(values[i]-values) < -2*_tol(b, d, p):
                        raise ArithmeticError("frontier interval fails dominance certificate")
            if np.isinf(hi) and d[i] != d.min():
                raise ArithmeticError("frontier tail does not have minimum dose slope")
    return rows


def nasal_price_limits(strategy_ids, health, base_cost, nasal_doses, threshold, *, expected_draws=1000):
    """Price set at which each arm attains the full-set expected-NMB maximum.

    Return one row per strategy. ``feasible`` indicates a nonempty nonnegative
    interval [price_lower, price_upper], including single-point ties. Empty
    intervals have NaN endpoints; signed_constraint_upper retains a negative
    upper constraint. ``nasal_doses`` is the actual discounted denominator.
    ties_at_lower/upper identify ALL optimal arms at finite feasible endpoints.
    Negative dose differences impose lower bounds and are never clipped.
    """
    names, b, d, _ = _lines(strategy_ids, health, base_cost, nasal_doses, threshold, expected_draws)
    rows = []
    for i, arm in enumerate(names):
        lower, upper, possible = 0., np.inf, True
        for j in range(len(names)):
            delta_b, delta_d = b[i]-b[j], d[i]-d[j]
            if delta_d > 0:
                upper = min(upper, delta_b/delta_d)
            elif delta_d < 0:
                lower = max(lower, delta_b/delta_d)
            elif delta_b < -_tol(b, d, 0.):
                possible = False
        feasible = possible and upper >= lower
        rows.append(dict(strategy_id=arm, nasal_doses=float(d[i]), feasible=bool(feasible),
            price_lower=float(lower) if feasible else np.nan,
            price_upper=float(upper) if feasible else np.nan,
            signed_constraint_upper=float(upper),
            ties_at_lower=_winners(names, b, d, lower) if feasible else "",
            ties_at_upper=_winners(names, b, d, upper) if feasible and np.isfinite(upper) else ""))
    return rows


def nasal_comparator_ceilings(strategy_ids, health, base_cost, nasal_doses, threshold,
                              same_coverage_standard: Mapping[str, str], *, expected_draws=1000):
    """Ceilings against matched Tdap and the best expected-NMB standard arm.

    Mapping keys identify nasal candidates (including any with zero receipts);
    values identify their same-coverage maternal/Tdap standard comparator.
    Every positive-nasal-dose arm must appear as a key. All remaining arms
    must have zero nasal doses and form the standard comparison set.

    Rows contain strategy_id, same_coverage_standard_strategy, nasal_doses,
    expected_nmb_advantage_vs_same_coverage, ceiling_vs_same_coverage_standard,
    same_coverage_ceiling_q025/q975 (paired conditional draws),
    best_standard_strategies, expected_nmb_advantage_vs_best_standard,
    ceiling_vs_best_standard, and price_dependence. Best-standard ceilings
    compare EXPECTED NMB, retaining all tied standard optima. Negative ceilings
    are retained. Zero-dose candidates are price-independent: a mean advantage
    >= 0 gives infinity and a negative advantage gives -infinity; quantiles
    are NaN because there is no per-administration denominator.
    """
    names, b, d, draws = _lines(strategy_ids, health, base_cost, nasal_doses, threshold, expected_draws)
    mapping = dict(same_coverage_standard)
    if not mapping or any(a not in names or ref not in names for a, ref in mapping.items()):
        raise ValueError("mapping must identify candidate and standard strategy IDs")
    standard = [i for i, a in enumerate(names) if a not in mapping]
    if not standard or any(d[i] != 0 for i in standard) or any(ref in mapping for ref in mapping.values()):
        raise ValueError("all standard arms, including matched comparators, must have zero nasal doses")
    sbest = max(b[i] for i in standard)
    ties = "|".join(names[i] for i in standard if sbest-b[i] <= _tol(b[standard], d[standard], 0.))
    rows = []
    for arm, ref in mapping.items():
        i, j = names.index(arm), names.index(ref)
        paired_gain = draws[:, i]-draws[:, j]
        matched_gain, best_gain = float(paired_gain.mean()), float(b[i]-sbest)
        if d[i] > 0:
            cap = paired_gain/d[i]
            same, best = float(cap.mean()), best_gain/d[i]
            low, high = np.quantile(cap, [.025, .975])
        else:
            same = np.inf if matched_gain >= 0 else -np.inf
            best = np.inf if best_gain >= 0 else -np.inf
            low = high = np.nan
        rows.append(dict(strategy_id=arm, same_coverage_standard_strategy=ref,
            nasal_doses=float(d[i]), expected_nmb_advantage_vs_same_coverage=matched_gain,
            ceiling_vs_same_coverage_standard=float(same),
            same_coverage_ceiling_q025=float(low), same_coverage_ceiling_q975=float(high),
            best_standard_strategies=ties, expected_nmb_advantage_vs_best_standard=best_gain,
            ceiling_vs_best_standard=float(best),
            price_dependence="per_actual_nasal_administration" if d[i] > 0 else "price_independent"))
    return rows
