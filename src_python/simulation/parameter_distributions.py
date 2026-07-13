"""Validated inverse-CDF transforms for epistemic parameter uncertainty.

The public functions in this module deliberately separate the sampling design
from the marginal parameter distributions.  A Latin-hypercube design is first
created on the unit interval and each column is then transformed by its
specified inverse CDF.  This preserves the one-dimensional stratification for
non-uniform, literature-informed priors.

Supported distribution specifications are:

``uniform``
    ``low``/``high`` (or the legacy aliases ``min``/``max``).
``beta``
    Underlying (pre-truncation) beta ``mean`` and ``sd`` on [0, 1], with
    optional truncation bounds given by ``low``/``high`` or ``min``/``max``.
``beta_pert``
    ``low``, ``mode``, ``high`` and optional PERT ``shape`` (default 4).
``truncated_normal``
    ``mean``, ``sd`` and finite bounds.
``truncated_lognormal``
    Requested post-truncation ``median``, ``log_sd`` and finite non-negative
    bounds. ``log_sd`` is the standard deviation of the underlying normal;
    its location is resolved so the truncated distribution has that median.
``complement_truncated_lognormal``
    A probability such as efficacy represented as one minus a truncated
    lognormal risk ratio.  ``median`` is on the reported probability scale,
    with ``log_sd`` applying to its complement.
``triangular`` and ``loguniform``
    ``low`` and ``high``; triangular additionally requires ``mode``.
``spike_and_slab``
    ``spike_probability``, a numeric point ``spike`` (or a nested
    distribution specification), and a nested ``slab`` specification.

A specification with bounds but no ``distribution`` remains backward
compatible and is interpreted as uniform.  Additional metadata fields such as
``note``, ``source`` and ``path`` are retained by validation and ignored by the
transform.  :func:`log_pdf` evaluates the matching normalized density (or
probability mass for a constant/spike component), so proposal generation and
Bayesian prior evaluation can share one distribution contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import special as scipy_special
from scipy.optimize import brentq
from scipy.stats import beta as beta_distribution
from scipy.stats import qmc, truncnorm


DistributionSpec: TypeAlias = Mapping[str, Any]
NamedDistributionSpecs: TypeAlias = (
    Mapping[str, DistributionSpec] | Sequence[tuple[str, DistributionSpec]]
)

SUPPORTED_DISTRIBUTIONS = frozenset(
    {
        "uniform",
        "beta",
        "beta_pert",
        "truncated_normal",
        "truncated_lognormal",
        "complement_truncated_lognormal",
        "triangular",
        "loguniform",
        "spike_and_slab",
    }
)

_DISTRIBUTION_ALIASES = {
    "beta-pert": "beta_pert",
    "pert": "beta_pert",
    "truncnorm": "truncated_normal",
    "truncated-normal": "truncated_normal",
    "truncated normal": "truncated_normal",
    "truncated-lognormal": "truncated_lognormal",
    "truncated lognormal": "truncated_lognormal",
    "truncated_log_normal": "truncated_lognormal",
    "complement-lognormal": "complement_truncated_lognormal",
    "complement_lognormal": "complement_truncated_lognormal",
    "one_minus_lognormal": "complement_truncated_lognormal",
    "log_uniform": "loguniform",
    "log-uniform": "loguniform",
    "spike-and-slab": "spike_and_slab",
    "spike and slab": "spike_and_slab",
}


def _finite_float(value: Any, *, field: str, context: str) -> float:
    """Convert a scalar to float while rejecting booleans and non-finite values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: '{field}' must be a finite number, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: '{field}' must be a finite number") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{context}: '{field}' must be finite; got {value!r}")
    return numeric


def _canonical_name(spec: Mapping[str, Any], *, context: str) -> str:
    if "distribution" not in spec:
        if any(key in spec for key in ("low", "high", "min", "max")):
            return "uniform"
        if "value" in spec:
            # Point masses are primarily useful as nested spike components.
            return "constant"
        raise ValueError(
            f"{context}: missing 'distribution'; legacy specifications must provide "
            "both 'min' and 'max' (or 'low' and 'high')"
        )
    raw_name = spec["distribution"]
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"{context}: 'distribution' must be a non-empty string")
    cleaned = raw_name.strip().lower()
    return _DISTRIBUTION_ALIASES.get(cleaned, cleaned)


def _bounds(
    spec: Mapping[str, Any],
    *,
    context: str,
    required: bool = True,
    defaults: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Read and cross-check low/high and legacy min/max bound aliases."""

    has_low = "low" in spec
    has_high = "high" in spec
    has_min = "min" in spec
    has_max = "max" in spec

    if has_low != has_high:
        missing = "high" if has_low else "low"
        raise ValueError(f"{context}: bounds must include both 'low' and 'high'; missing '{missing}'")
    if has_min != has_max:
        missing = "max" if has_min else "min"
        raise ValueError(f"{context}: legacy bounds must include both 'min' and 'max'; missing '{missing}'")

    modern: tuple[float, float] | None = None
    legacy: tuple[float, float] | None = None
    if has_low:
        modern = (
            _finite_float(spec["low"], field="low", context=context),
            _finite_float(spec["high"], field="high", context=context),
        )
    if has_min:
        legacy = (
            _finite_float(spec["min"], field="min", context=context),
            _finite_float(spec["max"], field="max", context=context),
        )

    if modern is not None and legacy is not None and modern != legacy:
        raise ValueError(
            f"{context}: 'low'/'high' and 'min'/'max' specify different bounds"
        )
    selected = modern if modern is not None else legacy
    if selected is None:
        if defaults is not None:
            selected = defaults
        elif required:
            raise ValueError(
                f"{context}: missing bounds; provide 'low' and 'high' or legacy 'min' and 'max'"
            )
        else:  # Defensive: every current optional use supplies defaults.
            raise ValueError(f"{context}: no bounds or defaults were provided")

    low, high = selected
    if not low < high:
        raise ValueError(f"{context}: lower bound must be less than upper bound; got [{low}, {high}]")
    if not np.isfinite(high - low):
        raise ValueError(f"{context}: bound width must be finite; got [{low}, {high}]")
    return low, high


def _required_float(spec: Mapping[str, Any], field: str, *, context: str) -> float:
    if field not in spec:
        raise ValueError(f"{context}: missing required field '{field}'")
    return _finite_float(spec[field], field=field, context=context)


def _nested_spec(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return validate_distribution_spec(value, context=context)
    # A numeric nested component is a convenient notation for a point mass.
    point = _finite_float(value, field="value", context=context)
    return {"distribution": "constant", "value": point}


def _support_bounds_validated(spec: Mapping[str, Any]) -> tuple[float, float]:
    """Return finite support bounds for an already-normalized specification."""

    name = str(spec["distribution"])
    if name == "constant":
        value = float(spec["value"])
        return value, value
    if name == "spike_and_slab":
        probability = float(spec["spike_probability"])
        if probability == 0.0:
            return _support_bounds_validated(spec["slab"])
        if probability == 1.0:
            return _support_bounds_validated(spec["spike"])
        spike_low, spike_high = _support_bounds_validated(spec["spike"])
        slab_low, slab_high = _support_bounds_validated(spec["slab"])
        return min(spike_low, slab_low), max(spike_high, slab_high)
    return float(spec["low"]), float(spec["high"])


def _normal_location_for_truncated_median(
    median: float,
    sd: float,
    low: float,
    high: float,
    *,
    context: str,
) -> float:
    """Resolve the untruncated normal location giving a requested truncated median."""

    if not low < median < high:
        raise ValueError(f"{context}: 'median' must lie strictly within the truncation bounds")

    def residual(location: float) -> float:
        lower_standard = (low - location) / sd
        upper_standard = (high - location) / sd
        value = float(
            truncnorm.ppf(
                0.5,
                lower_standard,
                upper_standard,
                loc=location,
                scale=sd,
            )
        )
        return value - median

    finite_span = high - low if np.isfinite(low) and np.isfinite(high) else 4.0 * sd
    width = max(sd, finite_span)
    lower = median - width
    upper = median + width
    for _ in range(20):
        lower_value = residual(lower)
        upper_value = residual(upper)
        if np.isfinite(lower_value) and np.isfinite(upper_value) and lower_value <= 0.0 <= upper_value:
            return float(brentq(residual, lower, upper, xtol=1e-13, rtol=1e-13))
        width *= 2.0
        lower = median - width
        upper = median + width
    raise ValueError(f"{context}: could not resolve a truncated distribution with the requested median")


def _log_normal_interval_probability(lower: float, upper: float) -> float:
    """Stable log P(lower <= Z <= upper) for a standard normal variable."""

    if not lower < upper:
        return -np.inf
    if lower >= 0.0:
        log_large = float(scipy_special.log_ndtr(-lower))
        log_small = float(scipy_special.log_ndtr(-upper))
    elif upper <= 0.0:
        log_large = float(scipy_special.log_ndtr(upper))
        log_small = float(scipy_special.log_ndtr(lower))
    else:
        probability = float(scipy_special.ndtr(upper) - scipy_special.ndtr(lower))
        return float(np.log(probability)) if probability > 0.0 else -np.inf
    if log_small >= log_large:
        return -np.inf
    return float(log_large + np.log1p(-np.exp(log_small - log_large)))


def validate_distribution_spec(
    spec: DistributionSpec,
    *,
    context: str = "distribution specification",
) -> dict[str, Any]:
    """Validate and return a normalized copy of a distribution specification.

    The returned dictionary retains non-distribution metadata.  Bound aliases
    are normalized by adding ``low`` and ``high`` and the canonical
    distribution name is stored in ``distribution``.  The input is never
    mutated.

    Parameters
    ----------
    spec:
        Mapping describing one of :data:`SUPPORTED_DISTRIBUTIONS`.
    context:
        Label used in validation errors, useful for identifying a parameter in
        a larger configuration.
    """

    if not isinstance(spec, Mapping):
        raise ValueError(f"{context}: specification must be a mapping")
    normalized = deepcopy(dict(spec))
    name = _canonical_name(normalized, context=context)
    if name not in SUPPORTED_DISTRIBUTIONS and name != "constant":
        supported = ", ".join(sorted(SUPPORTED_DISTRIBUTIONS))
        raise ValueError(
            f"{context}: unsupported distribution {name!r}; supported distributions are {supported}"
        )
    normalized["distribution"] = name

    if name == "constant":
        normalized["value"] = _required_float(normalized, "value", context=context)
        return normalized

    if name == "uniform":
        low, high = _bounds(normalized, context=context)
        normalized.update(low=low, high=high)
        return normalized

    if name == "beta":
        mean = _required_float(normalized, "mean", context=context)
        sd = _required_float(normalized, "sd", context=context)
        if not 0.0 < mean < 1.0:
            raise ValueError(f"{context}: beta 'mean' must lie strictly between 0 and 1")
        if sd <= 0.0:
            raise ValueError(f"{context}: beta 'sd' must be greater than zero")
        maximum_variance = mean * (1.0 - mean)
        variance = sd * sd
        if variance == 0.0:
            raise ValueError(f"{context}: beta 'sd' is too small to represent numerically")
        if variance >= maximum_variance:
            raise ValueError(
                f"{context}: beta 'sd' is too large for mean={mean}; it must be less than "
                f"{np.sqrt(maximum_variance):.12g}"
            )
        concentration = maximum_variance / variance - 1.0
        if not np.isfinite(concentration):
            raise ValueError(
                f"{context}: beta mean/sd imply non-finite shape parameters; use a larger 'sd'"
            )
        low, high = _bounds(normalized, context=context, defaults=(0.0, 1.0))
        if low < 0.0 or high > 1.0:
            raise ValueError(f"{context}: beta truncation bounds must lie within [0, 1]")
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        probability_width = float(
            beta_distribution.cdf(high, alpha, beta)
            - beta_distribution.cdf(low, alpha, beta)
        )
        if not np.isfinite(probability_width) or probability_width <= 0.0:
            raise ValueError(
                f"{context}: beta truncation interval has no numerically resolvable mass"
            )
        normalized.update(
            mean=mean,
            sd=sd,
            low=low,
            high=high,
            _alpha=alpha,
            _beta=beta,
            _log_probability_width=float(np.log(probability_width)),
        )
        return normalized

    if name == "beta_pert":
        low, high = _bounds(normalized, context=context)
        mode = _required_float(normalized, "mode", context=context)
        shape = _finite_float(normalized.get("shape", 4.0), field="shape", context=context)
        if not low <= mode <= high:
            raise ValueError(f"{context}: beta-PERT 'mode' must lie within [low, high]")
        if shape <= 0.0:
            raise ValueError(f"{context}: beta-PERT 'shape' must be greater than zero")
        mode_fraction = (mode - low) / (high - low)
        alpha = 1.0 + shape * mode_fraction
        beta = 1.0 + shape * (1.0 - mode_fraction)
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError(f"{context}: beta-PERT parameters imply non-finite shape parameters")
        normalized.update(
            low=low,
            mode=mode,
            high=high,
            shape=shape,
            _alpha=alpha,
            _beta=beta,
        )
        return normalized

    if name == "truncated_normal":
        mean = _required_float(normalized, "mean", context=context)
        sd = _required_float(normalized, "sd", context=context)
        low, high = _bounds(normalized, context=context)
        if sd <= 0.0:
            raise ValueError(f"{context}: truncated-normal 'sd' must be greater than zero")
        lower_standard = (low - mean) / sd
        upper_standard = (high - mean) / sd
        if not lower_standard < upper_standard:
            raise ValueError(
                f"{context}: truncation interval is not numerically resolvable on the normal scale"
            )
        log_probability_width = _log_normal_interval_probability(
            lower_standard, upper_standard
        )
        if not np.isfinite(log_probability_width):
            raise ValueError(
                f"{context}: truncation interval has no numerically resolvable normal mass"
            )
        normalized.update(
            mean=mean,
            sd=sd,
            low=low,
            high=high,
            _log_probability_width=log_probability_width,
        )
        return normalized

    if name == "truncated_lognormal":
        median = _required_float(normalized, "median", context=context)
        log_sd = _required_float(normalized, "log_sd", context=context)
        low, high = _bounds(normalized, context=context)
        if median <= 0.0:
            raise ValueError(f"{context}: lognormal 'median' must be greater than zero")
        if log_sd <= 0.0:
            raise ValueError(f"{context}: lognormal 'log_sd' must be greater than zero")
        if low < 0.0:
            raise ValueError(f"{context}: lognormal lower bound cannot be negative")
        log_median = np.log(median)
        log_low = -np.inf if low == 0.0 else np.log(low)
        log_high = np.log(high)
        log_location = _normal_location_for_truncated_median(
            log_median,
            log_sd,
            log_low,
            log_high,
            context=context,
        )
        lower_standard = (log_low - log_location) / log_sd
        upper_standard = (log_high - log_location) / log_sd
        if not lower_standard < upper_standard:
            raise ValueError(
                f"{context}: truncation interval is not numerically resolvable on the log scale"
            )
        log_probability_width = _log_normal_interval_probability(
            lower_standard, upper_standard
        )
        if not np.isfinite(log_probability_width):
            raise ValueError(
                f"{context}: truncation interval has no numerically resolvable lognormal mass"
            )
        normalized.update(
            median=median,
            log_sd=log_sd,
            low=low,
            high=high,
            _log_location=log_location,
            _log_probability_width=log_probability_width,
        )
        return normalized

    if name == "complement_truncated_lognormal":
        median = _required_float(normalized, "median", context=context)
        log_sd = _required_float(normalized, "log_sd", context=context)
        low, high = _bounds(normalized, context=context)
        if not 0.0 < median < 1.0:
            raise ValueError(
                f"{context}: complement-lognormal 'median' must lie strictly between 0 and 1"
            )
        if log_sd <= 0.0:
            raise ValueError(
                f"{context}: complement-lognormal 'log_sd' must be greater than zero"
            )
        if low < 0.0 or high > 1.0:
            raise ValueError(
                f"{context}: complement-lognormal bounds must lie within [0, 1]"
            )
        failure_median = 1.0 - median
        failure_low = 1.0 - high
        failure_high = 1.0 - low
        log_failure_low = -np.inf if failure_low == 0.0 else np.log(failure_low)
        log_failure_high = np.log(failure_high)
        log_failure_location = _normal_location_for_truncated_median(
            np.log(failure_median),
            log_sd,
            log_failure_low,
            log_failure_high,
            context=context,
        )
        lower_standard = (log_failure_low - log_failure_location) / log_sd
        upper_standard = (log_failure_high - log_failure_location) / log_sd
        if not lower_standard < upper_standard:
            raise ValueError(
                f"{context}: truncation interval is not numerically resolvable on the "
                "complement log scale"
            )
        log_probability_width = _log_normal_interval_probability(
            lower_standard, upper_standard
        )
        if not np.isfinite(log_probability_width):
            raise ValueError(
                f"{context}: truncation interval has no numerically resolvable complement-lognormal mass"
            )
        normalized.update(
            median=median,
            log_sd=log_sd,
            low=low,
            high=high,
            _log_failure_location=log_failure_location,
            _log_probability_width=log_probability_width,
        )
        return normalized

    if name == "triangular":
        low, high = _bounds(normalized, context=context)
        mode = _required_float(normalized, "mode", context=context)
        if not low <= mode <= high:
            raise ValueError(f"{context}: triangular 'mode' must lie within [low, high]")
        normalized.update(low=low, mode=mode, high=high)
        return normalized

    if name == "loguniform":
        low, high = _bounds(normalized, context=context)
        if low <= 0.0:
            raise ValueError(f"{context}: loguniform bounds must be strictly positive")
        normalized.update(low=low, high=high)
        return normalized

    # spike_and_slab
    probability_fields = [
        field for field in ("spike_probability", "spike_prob") if field in normalized
    ]
    if not probability_fields:
        raise ValueError(f"{context}: missing required field 'spike_probability'")
    if len(probability_fields) == 2:
        first = _finite_float(
            normalized["spike_probability"], field="spike_probability", context=context
        )
        second = _finite_float(normalized["spike_prob"], field="spike_prob", context=context)
        if first != second:
            raise ValueError(f"{context}: 'spike_probability' and 'spike_prob' disagree")
        spike_probability = first
    else:
        field = probability_fields[0]
        spike_probability = _finite_float(normalized[field], field=field, context=context)
    if not 0.0 <= spike_probability <= 1.0:
        raise ValueError(f"{context}: 'spike_probability' must lie within [0, 1]")

    if "spike" in normalized and "spike_value" in normalized:
        raise ValueError(f"{context}: provide only one of 'spike' or 'spike_value'")
    if "spike" in normalized:
        spike_value = normalized["spike"]
    elif "spike_value" in normalized:
        spike_value = normalized["spike_value"]
    else:
        raise ValueError(f"{context}: missing required nested component 'spike'")
    if "slab" not in normalized:
        raise ValueError(f"{context}: missing required nested component 'slab'")

    normalized["spike_probability"] = spike_probability
    normalized["spike"] = _nested_spec(spike_value, context=f"{context}.spike")
    normalized.pop("spike_value", None)
    normalized["slab"] = _nested_spec(normalized["slab"], context=f"{context}.slab")
    if 0.0 < spike_probability < 1.0:
        spike_low, spike_high = _support_bounds_validated(normalized["spike"])
        slab_low, slab_high = _support_bounds_validated(normalized["slab"])
        if spike_high <= slab_low:
            normalized["_mixture_order"] = "spike_first"
        elif slab_high <= spike_low:
            normalized["_mixture_order"] = "slab_first"
        else:
            raise ValueError(
                f"{context}: spike-and-slab component supports overlap; a closed-form "
                "inverse CDF is only supported for ordered non-overlapping components"
            )
    else:
        normalized["_mixture_order"] = "degenerate"
    return normalized


def _beta_shapes(mean: float, sd: float) -> tuple[float, float]:
    concentration = mean * (1.0 - mean) / (sd * sd) - 1.0
    return mean * concentration, (1.0 - mean) * concentration


def _bounded_beta_ppf(
    unit: NDArray[np.float64],
    *,
    alpha: float,
    beta: float,
    low: float,
    high: float,
    context: str,
) -> NDArray[np.float64]:
    cdf_low = float(beta_distribution.cdf(low, alpha, beta))
    cdf_high = float(beta_distribution.cdf(high, alpha, beta))
    probability_width = cdf_high - cdf_low
    if not np.isfinite(probability_width) or probability_width <= 0.0:
        raise ValueError(
            f"{context}: beta truncation interval [{low}, {high}] has no numerically "
            "resolvable probability mass"
        )
    probability = cdf_low + unit * probability_width
    values = np.asarray(beta_distribution.ppf(probability, alpha, beta), dtype=float)
    values = np.clip(values, low, high)
    values = np.where(unit == 0.0, low, values)
    values = np.where(unit == 1.0, high, values)
    return values


def _inverse_cdf_validated(
    unit: NDArray[np.float64],
    spec: Mapping[str, Any],
    *,
    context: str,
) -> NDArray[np.float64]:
    name = spec["distribution"]

    if name == "constant":
        return np.full(unit.shape, float(spec["value"]), dtype=float)

    if name == "uniform":
        low, high = float(spec["low"]), float(spec["high"])
        values = low + unit * (high - low)
        values = np.where(unit == 0.0, low, values)
        values = np.where(unit == 1.0, high, values)
        return np.clip(values, low, high)

    if name == "beta":
        alpha, beta = _beta_shapes(float(spec["mean"]), float(spec["sd"]))
        return _bounded_beta_ppf(
            unit,
            alpha=alpha,
            beta=beta,
            low=float(spec["low"]),
            high=float(spec["high"]),
            context=context,
        )

    if name == "beta_pert":
        low = float(spec["low"])
        high = float(spec["high"])
        mode = float(spec["mode"])
        shape = float(spec["shape"])
        width = high - low
        alpha = 1.0 + shape * (mode - low) / width
        beta = 1.0 + shape * (high - mode) / width
        standard = np.asarray(beta_distribution.ppf(unit, alpha, beta), dtype=float)
        values = low + width * standard
        values = np.where(unit == 0.0, low, values)
        values = np.where(unit == 1.0, high, values)
        return np.clip(values, low, high)

    if name == "truncated_normal":
        mean = float(spec["mean"])
        sd = float(spec["sd"])
        low = float(spec["low"])
        high = float(spec["high"])
        lower_standard = (low - mean) / sd
        upper_standard = (high - mean) / sd
        values = np.asarray(
            truncnorm.ppf(
                unit,
                lower_standard,
                upper_standard,
                loc=mean,
                scale=sd,
            ),
            dtype=float,
        )
        values = np.where(unit == 0.0, low, values)
        values = np.where(unit == 1.0, high, values)
        return np.clip(values, low, high)

    if name == "truncated_lognormal":
        median = float(spec["median"])
        log_sd = float(spec["log_sd"])
        low = float(spec["low"])
        high = float(spec["high"])
        log_location = float(spec["_log_location"])
        log_low = -np.inf if low == 0.0 else np.log(low)
        log_high = np.log(high)
        lower_standard = (log_low - log_location) / log_sd
        upper_standard = (log_high - log_location) / log_sd
        log_values = np.asarray(
            truncnorm.ppf(
                unit,
                lower_standard,
                upper_standard,
                loc=log_location,
                scale=log_sd,
            ),
            dtype=float,
        )
        values = np.exp(log_values)
        values = np.where(unit == 0.0, low, values)
        values = np.where(unit == 1.0, high, values)
        return np.clip(values, low, high)

    if name == "complement_truncated_lognormal":
        median = float(spec["median"])
        log_sd = float(spec["log_sd"])
        low = float(spec["low"])
        high = float(spec["high"])
        failure_median = 1.0 - median
        failure_low = 1.0 - high
        failure_high = 1.0 - low
        log_failure_low = -np.inf if failure_low == 0.0 else np.log(failure_low)
        log_failure_location = float(spec["_log_failure_location"])
        lower_standard = (log_failure_low - log_failure_location) / log_sd
        upper_standard = (np.log(failure_high) - log_failure_location) / log_sd
        log_failure = np.asarray(
            truncnorm.ppf(
                1.0 - unit,
                lower_standard,
                upper_standard,
                loc=log_failure_location,
                scale=log_sd,
            ),
            dtype=float,
        )
        values = 1.0 - np.exp(log_failure)
        values = np.where(unit == 0.0, low, values)
        values = np.where(unit == 1.0, high, values)
        return np.clip(values, low, high)

    if name == "triangular":
        low = float(spec["low"])
        mode = float(spec["mode"])
        high = float(spec["high"])
        width = high - low
        mode_fraction = (mode - low) / width
        lower = low + width * np.sqrt(unit * mode_fraction)
        upper = high - width * np.sqrt((1.0 - unit) * (1.0 - mode_fraction))
        values = np.where(unit < mode_fraction, lower, upper)
        values = np.where(unit == 0.0, low, values)
        values = np.where(unit == 1.0, high, values)
        return np.clip(values, low, high)

    if name == "loguniform":
        low = float(spec["low"])
        high = float(spec["high"])
        values = np.exp(np.log(low) + unit * (np.log(high) - np.log(low)))
        values = np.where(unit == 0.0, low, values)
        values = np.where(unit == 1.0, high, values)
        return np.clip(values, low, high)

    probability = float(spec["spike_probability"])
    spike = spec["spike"]
    slab = spec["slab"]
    if probability == 0.0:
        return _inverse_cdf_validated(unit, slab, context=f"{context}.slab")
    if probability == 1.0:
        return _inverse_cdf_validated(unit, spike, context=f"{context}.spike")

    flattened = unit.reshape(-1)
    output = np.empty(flattened.shape, dtype=float)
    if spec.get("_mixture_order") == "slab_first":
        slab_probability = 1.0 - probability
        slab_mask = flattened < slab_probability
        if np.any(slab_mask):
            conditional_slab = flattened[slab_mask] / slab_probability
            output[slab_mask] = _inverse_cdf_validated(
                conditional_slab,
                slab,
                context=f"{context}.slab",
            )
        if np.any(~slab_mask):
            conditional_spike = (flattened[~slab_mask] - slab_probability) / probability
            output[~slab_mask] = _inverse_cdf_validated(
                conditional_spike,
                spike,
                context=f"{context}.spike",
            )
    else:
        spike_mask = flattened < probability
        if np.any(spike_mask):
            conditional_spike = flattened[spike_mask] / probability
            output[spike_mask] = _inverse_cdf_validated(
                conditional_spike,
                spike,
                context=f"{context}.spike",
            )
        if np.any(~spike_mask):
            conditional_slab = (flattened[~spike_mask] - probability) / (1.0 - probability)
            output[~spike_mask] = _inverse_cdf_validated(
                conditional_slab,
                slab,
                context=f"{context}.slab",
            )
    return output.reshape(unit.shape)


def inverse_cdf(
    unit_quantiles: ArrayLike,
    spec: DistributionSpec,
    *,
    context: str = "distribution specification",
) -> float | NDArray[np.float64]:
    """Transform scalar or array-like unit quantiles using a validated inverse CDF.

    Inputs must be finite and lie in the closed unit interval.  Scalar input
    returns a Python ``float``; array-like input returns a NumPy array with the
    same shape.  Every successful result is guaranteed finite.
    """

    try:
        unit = np.asarray(unit_quantiles, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("unit quantiles must be numeric") from exc
    scalar_input = unit.ndim == 0
    if not np.all(np.isfinite(unit)):
        raise ValueError("unit quantiles must all be finite")
    if np.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("unit quantiles must all lie within [0, 1]")

    normalized = validate_distribution_spec(spec, context=context)
    values = np.asarray(
        _inverse_cdf_validated(unit, normalized, context=context),
        dtype=float,
    )
    if values.shape != unit.shape:
        raise ValueError(
            f"{context}: internal transform changed shape from {unit.shape} to {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{context}: transform produced non-finite values; check distribution parameters "
            "and truncation bounds"
        )
    if scalar_input:
        return float(values)
    return values


def inverse_cdf_from_validated_spec(
    unit_quantiles: ArrayLike,
    validated_spec: Mapping[str, Any],
    *,
    context: str = "validated distribution specification",
) -> float | NDArray[np.float64]:
    """Fast inverse CDF for a spec returned by :func:`validate_distribution_spec`.

    This avoids repeatedly resolving truncated-distribution normalization in a
    tight sampler loop.  Callers are responsible for passing an already
    validated specification; unit-quantile validation and output guarantees
    remain the same as :func:`inverse_cdf`.
    """

    try:
        unit = np.asarray(unit_quantiles, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("unit quantiles must be numeric") from exc
    scalar_input = unit.ndim == 0
    if not np.all(np.isfinite(unit)):
        raise ValueError("unit quantiles must all be finite")
    if np.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("unit quantiles must all lie within [0, 1]")
    if not isinstance(validated_spec, Mapping) or "distribution" not in validated_spec:
        raise ValueError(f"{context}: expected a validated distribution mapping")
    values = np.asarray(
        _inverse_cdf_validated(unit, validated_spec, context=context),
        dtype=float,
    )
    if values.shape != unit.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{context}: validated inverse-CDF transform produced invalid output")
    if scalar_input:
        return float(values)
    return values


def _log_pdf_validated(
    values: NDArray[np.float64],
    spec: Mapping[str, Any],
    *,
    context: str,
) -> NDArray[np.float64]:
    """Evaluate a normalized spec without repeating public validation."""

    name = str(spec["distribution"])
    output = np.full(values.shape, -np.inf, dtype=float)

    if name == "constant":
        output[values == float(spec["value"])] = 0.0
        return output

    if name == "spike_and_slab":
        probability = float(spec["spike_probability"])
        if probability == 0.0:
            return _log_pdf_validated(values, spec["slab"], context=f"{context}.slab")
        if probability == 1.0:
            return _log_pdf_validated(values, spec["spike"], context=f"{context}.spike")
        spike_logp = _log_pdf_validated(values, spec["spike"], context=f"{context}.spike")
        slab_logp = _log_pdf_validated(values, spec["slab"], context=f"{context}.slab")
        return np.logaddexp(np.log(probability) + spike_logp, np.log1p(-probability) + slab_logp)

    low = float(spec["low"])
    high = float(spec["high"])
    inside = (values >= low) & (values <= high)
    if not np.any(inside):
        return output
    x = values[inside]

    if name == "uniform":
        output[inside] = -np.log(high - low)
        return output

    if name == "beta":
        if "_alpha" in spec and "_beta" in spec:
            alpha, beta = float(spec["_alpha"]), float(spec["_beta"])
        else:  # Backward compatibility for externally prepared normalized specs.
            alpha, beta = _beta_shapes(float(spec["mean"]), float(spec["sd"]))
        log_width = float(spec.get("_log_probability_width", 0.0))
        output[inside] = (
            scipy_special.xlogy(alpha - 1.0, x)
            + scipy_special.xlog1py(beta - 1.0, -x)
            - scipy_special.betaln(alpha, beta)
            - log_width
        )
        return output

    if name == "beta_pert":
        width = high - low
        alpha = float(spec["_alpha"])
        beta = float(spec["_beta"])
        standardized = (x - low) / width
        output[inside] = (
            scipy_special.xlogy(alpha - 1.0, standardized)
            + scipy_special.xlog1py(beta - 1.0, -standardized)
            - scipy_special.betaln(alpha, beta)
            - np.log(width)
        )
        return output

    if name == "truncated_normal":
        mean = float(spec["mean"])
        sd = float(spec["sd"])
        z = (x - mean) / sd
        output[inside] = (
            -0.5 * z * z
            - np.log(sd)
            - 0.5 * np.log(2.0 * np.pi)
            - float(spec["_log_probability_width"])
        )
        return output

    if name == "truncated_lognormal":
        positive = x > 0.0
        if np.any(positive):
            log_x = np.log(x[positive])
            log_sd = float(spec["log_sd"])
            log_location = float(spec["_log_location"])
            z = (log_x - log_location) / log_sd
            local = (
                -0.5 * z * z
                - np.log(log_sd)
                - 0.5 * np.log(2.0 * np.pi)
                - float(spec["_log_probability_width"])
                - log_x
            )
            inside_positions = np.flatnonzero(inside)
            output.flat[inside_positions[positive]] = local
        return output

    if name == "complement_truncated_lognormal":
        failure = 1.0 - x
        positive = failure > 0.0
        if np.any(positive):
            log_failure = np.log(failure[positive])
            log_sd = float(spec["log_sd"])
            location = float(spec["_log_failure_location"])
            z = (log_failure - location) / log_sd
            local = (
                -0.5 * z * z
                - np.log(log_sd)
                - 0.5 * np.log(2.0 * np.pi)
                - float(spec["_log_probability_width"])
                - log_failure
            )
            inside_positions = np.flatnonzero(inside)
            output.flat[inside_positions[positive]] = local
        return output

    if name == "triangular":
        mode = float(spec["mode"])
        width = high - low
        density = np.empty(x.shape, dtype=float)
        if mode == low:
            density = 2.0 * (high - x) / (width * width)
        elif mode == high:
            density = 2.0 * (x - low) / (width * width)
        else:
            lower = 2.0 * (x - low) / (width * (mode - low))
            upper = 2.0 * (high - x) / (width * (high - mode))
            density = np.where(x <= mode, lower, upper)
        with np.errstate(divide="ignore"):
            output[inside] = np.log(density)
        return output

    if name == "loguniform":
        positive = x > 0.0
        local = np.full(x.shape, -np.inf, dtype=float)
        local[positive] = -np.log(x[positive]) - np.log(np.log(high / low))
        output[inside] = local
        return output

    raise ValueError(f"{context}: density evaluation is not implemented for {name!r}")


def log_pdf(
    values: ArrayLike,
    spec: DistributionSpec,
    *,
    context: str = "distribution specification",
) -> float | NDArray[np.float64]:
    """Evaluate the normalized log density matching :func:`inverse_cdf`.

    Values outside the configured support return ``-inf``.  A scalar input
    returns a Python ``float`` and an array-like input preserves its shape.
    Continuous densities may be infinite exactly at a support boundary (for
    example a beta density with a shape parameter below one); this is the
    mathematically correct density and callers should avoid endpoint proposal
    quantiles when a finite transformed log density is required.
    """

    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("density values must be numeric") from exc
    scalar_input = numeric.ndim == 0
    normalized = validate_distribution_spec(spec, context=context)
    output = np.asarray(
        _log_pdf_validated(numeric, normalized, context=context),
        dtype=float,
    )
    if output.shape != numeric.shape:
        raise ValueError(
            f"{context}: internal density evaluation changed shape from "
            f"{numeric.shape} to {output.shape}"
        )
    output = np.where(np.isfinite(numeric), output, -np.inf)
    if scalar_input:
        return float(output)
    return output


def log_pdf_from_validated_spec(
    values: ArrayLike,
    validated_spec: Mapping[str, Any],
    *,
    context: str = "validated distribution specification",
) -> float | NDArray[np.float64]:
    """Fast log density for a spec returned by :func:`validate_distribution_spec`."""

    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("density values must be numeric") from exc
    scalar_input = numeric.ndim == 0
    if not isinstance(validated_spec, Mapping) or "distribution" not in validated_spec:
        raise ValueError(f"{context}: expected a validated distribution mapping")
    output = np.asarray(
        _log_pdf_validated(numeric, validated_spec, context=context),
        dtype=float,
    )
    if output.shape != numeric.shape:
        raise ValueError(
            f"{context}: internal density evaluation changed shape from "
            f"{numeric.shape} to {output.shape}"
        )
    output = np.where(np.isfinite(numeric), output, -np.inf)
    if scalar_input:
        return float(output)
    return output


def _named_items(parameter_specs: NamedDistributionSpecs) -> list[tuple[str, DistributionSpec]]:
    if isinstance(parameter_specs, Mapping):
        items: list[Any] = list(parameter_specs.items())
    elif isinstance(parameter_specs, Sequence) and not isinstance(parameter_specs, (str, bytes)):
        items = list(parameter_specs)
    else:
        raise ValueError(
            "parameter_specs must be an ordered mapping or a sequence of (name, spec) pairs"
        )
    if not items:
        raise ValueError("parameter_specs must contain at least one named distribution")

    validated_items: list[tuple[str, DistributionSpec]] = []
    seen: set[str] = set()
    for position, item in enumerate(items):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError(
                f"parameter_specs item {position} must be a (name, specification) pair"
            )
        name, spec = item
        if not isinstance(name, str) or not name:
            raise ValueError(f"parameter_specs item {position} has an invalid parameter name")
        if name in seen:
            raise ValueError(f"parameter_specs contains duplicate parameter name {name!r}")
        seen.add(name)
        validated_items.append((name, spec))
    return validated_items


def latin_hypercube_draw_table(
    parameter_specs: NamedDistributionSpecs,
    n_draws: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return named Latin-hypercube draws transformed to the requested marginals.

    Mapping insertion order (or sequence order) is preserved exactly in the
    returned :class:`pandas.DataFrame`.  Given the same specifications,
    ``n_draws`` and seed, output is reproducible.
    """

    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(n_draws, Integral):
        raise ValueError("n_draws must be a positive integer")
    n = int(n_draws)
    if n <= 0:
        raise ValueError("n_draws must be a positive integer")

    items = _named_items(parameter_specs)
    normalized = [
        (name, validate_distribution_spec(spec, context=f"parameter {name!r}"))
        for name, spec in items
    ]
    try:
        sampler = qmc.LatinHypercube(d=len(normalized), seed=seed)
        unit_draws = sampler.random(n=n)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not initialize Latin-hypercube sampler: {exc}") from exc

    columns = {
        name: inverse_cdf(unit_draws[:, column], spec, context=f"parameter {name!r}")
        for column, (name, spec) in enumerate(normalized)
    }
    return pd.DataFrame(columns, columns=[name for name, _ in normalized])


# Concise alias for callers that use "draws" rather than "draw table" wording.
latin_hypercube_draws = latin_hypercube_draw_table


__all__ = [
    "SUPPORTED_DISTRIBUTIONS",
    "inverse_cdf",
    "latin_hypercube_draw_table",
    "latin_hypercube_draws",
    "validate_distribution_spec",
]
