"""Load and validate the two economics YAML files.

Schemas
-------
core2026-severity-priors-v1
    countries.<C>.status; entries list with quantity, basis, age bounds,
    distribution, params, point, source.
core2026-costs-v1
    countries.<C>.items list, .macro thresholds;
    global.disability_weights, global.life_table.

Interpretation limits
---------------------
Countries with status ``use_calibrated_reference_risk`` signal that the
clinical cascade must go through the mortality_extrapolation path; the
economics layer does not apply per-confirmed-case CFRs to them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SEVERITY_SCHEMA = "core2026-severity-priors-v1"
COSTS_SCHEMA = "core2026-costs-v1"

VALID_QUANTITIES = frozenset(
    {"hospitalisation_proportion", "cfr", "icu_given_hospital", "los_days"}
)
VALID_BASES = frozenset({"per_confirmed_case", "per_hospitalised_case"})
VALID_SEVERITY_DISTRIBUTIONS = frozenset({"beta", "lognormal", "fixed"})

VALID_COST_ITEMS = frozenset({
    "vaccine_price_maternal", "vaccine_price_adolescent",
    "delivery_maternal", "delivery_adolescent",
    "outpatient_episode", "inpatient_episode", "icu_day", "inpatient_day",
})
VALID_COST_DISTRIBUTIONS = frozenset({"gamma", "lognormal", "fixed"})

STATUS_LITERATURE = "literature_prior"
STATUS_CALIBRATED = "use_calibrated_reference_risk"
# Status labels written by the evidence ledger; both carry an entries list
# (possibly empty when every quantity is borrowed through a declared rule).
STATUS_NATIONAL = "national_data"
STATUS_LITERATURE_LEDGER = "literature"
VALID_STATUSES = frozenset({STATUS_LITERATURE, STATUS_CALIBRATED,
                            STATUS_NATIONAL, STATUS_LITERATURE_LEDGER})

_INF_STRINGS = frozenset({"inf", "infinity", "null", "none", ""})


def _inf_or_float(value: Any) -> float:
    """Parse age_end_years: None / 'inf' map to float('inf')."""
    if value is None:
        return float("inf")
    s = str(value).strip().lower()
    if s in _INF_STRINGS:
        return float("inf")
    return float(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_severity_entry(entry: dict[str, Any], country: str, idx: int) -> None:
    loc = f"countries.{country}.entries[{idx}]"
    _require("quantity" in entry, f"{loc}: missing 'quantity'")
    _require(
        entry["quantity"] in VALID_QUANTITIES,
        f"{loc}: unknown quantity {entry['quantity']!r}; must be one of "
        f"{sorted(VALID_QUANTITIES)}",
    )
    _require("basis" in entry, f"{loc}: missing 'basis'")
    _require(
        entry["basis"] in VALID_BASES,
        f"{loc}: unknown basis {entry['basis']!r}",
    )
    _require("age_start_years" in entry, f"{loc}: missing 'age_start_years'")
    _require("age_end_years" in entry, f"{loc}: missing 'age_end_years'")
    lo = float(entry["age_start_years"])
    hi = _inf_or_float(entry["age_end_years"])
    _require(0.0 <= lo < hi, f"{loc}: age bounds must satisfy 0 <= start < end")
    _require("distribution" in entry, f"{loc}: missing 'distribution'")
    dist = entry["distribution"]
    _require(
        dist in VALID_SEVERITY_DISTRIBUTIONS,
        f"{loc}: unknown distribution {dist!r}; must be one of "
        f"{sorted(VALID_SEVERITY_DISTRIBUTIONS)}",
    )
    _require("point" in entry, f"{loc}: missing point estimate")
    _require(float(entry["point"]) >= 0.0, f"{loc}: point must be non-negative")
    _require(
        "source" in entry and str(entry["source"]).strip(),
        f"{loc}: non-empty 'source' is required",
    )
    if dist != "fixed":
        _require(
            "params" in entry and isinstance(entry["params"], dict),
            f"{loc}: params dict required for distribution {dist!r}",
        )


def load_severity_priors(path: str | Path) -> dict[str, Any]:
    """Load and validate a core2026-severity-priors-v1 YAML.

    Returns the raw parsed document.  Callers use
    ``doc["countries"][country]["entries"]`` for per-age severity draws and
    ``doc["countries"][country]["status"]`` to detect reference-risk countries.

    Raises
    ------
    ValueError
        If the schema tag is missing or wrong, or any structural constraint
        fails.
    """
    doc = yaml.safe_load(Path(path).read_text())
    _require(isinstance(doc, dict), "severity YAML must be a mapping")
    _require(
        doc.get("schema") == SEVERITY_SCHEMA,
        f"expected schema {SEVERITY_SCHEMA!r}, got {doc.get('schema')!r}",
    )
    _require(
        "countries" in doc and isinstance(doc["countries"], dict),
        "severity YAML requires a 'countries' mapping",
    )
    for country, spec in doc["countries"].items():
        _require(isinstance(spec, dict), f"countries.{country} must be a mapping")
        status = spec.get("status", "")
        _require(
            status in VALID_STATUSES,
            f"countries.{country}.status must be one of {sorted(VALID_STATUSES)!r},"
            f" got {status!r}",
        )
        if status == STATUS_CALIBRATED:
            continue  # no entry list required for reference-risk countries
        if status == STATUS_LITERATURE_LEDGER and "entries" not in spec:
            spec["entries"] = []  # every quantity borrowed via declared gaps
        _require(
            "entries" in spec and isinstance(spec["entries"], list),
            f"countries.{country}: entries list required",
        )
        for i, entry in enumerate(spec["entries"]):
            _validate_severity_entry(entry, country, i)
    return doc


def _validate_cost_item(item: dict[str, Any], country: str, idx: int) -> None:
    loc = f"countries.{country}.items[{idx}]"
    _require("item" in item, f"{loc}: missing 'item'")
    _require(
        item["item"] in VALID_COST_ITEMS,
        f"{loc}: unknown item {item['item']!r}; must be one of "
        f"{sorted(VALID_COST_ITEMS)}",
    )
    has_intl = "value_intl_2024" in item
    has_usd = "value_usd_2024" in item
    _require(
        has_intl or has_usd,
        f"{loc}: at least one of value_intl_2024 or value_usd_2024 is required",
    )
    _require("distribution" in item, f"{loc}: missing 'distribution'")
    dist = item["distribution"]
    _require(
        dist in VALID_COST_DISTRIBUTIONS,
        f"{loc}: unknown distribution {dist!r}; must be one of "
        f"{sorted(VALID_COST_DISTRIBUTIONS)}",
    )
    if dist != "fixed":
        _require(
            "params" in item and isinstance(item["params"], dict),
            f"{loc}: params dict required for distribution {dist!r}",
        )


def load_costs(path: str | Path) -> dict[str, Any]:
    """Load and validate a core2026-costs-v1 YAML.

    Returns the raw parsed document.  The preferred monetary unit is
    ``value_intl_2024`` (2024 international dollars); ``value_usd_2024`` is
    accepted when intl is absent, but callers must track which unit is active.
    """
    doc = yaml.safe_load(Path(path).read_text())
    _require(isinstance(doc, dict), "costs YAML must be a mapping")
    _require(
        doc.get("schema") == COSTS_SCHEMA,
        f"expected schema {COSTS_SCHEMA!r}, got {doc.get('schema')!r}",
    )
    _require(
        "countries" in doc and isinstance(doc["countries"], dict),
        "costs YAML requires a 'countries' mapping",
    )
    for country, spec in doc["countries"].items():
        _require(isinstance(spec, dict), f"countries.{country} must be a mapping")
        _require(
            "items" in spec and isinstance(spec["items"], list),
            f"countries.{country}: items list required",
        )
        for i, item in enumerate(spec["items"]):
            _validate_cost_item(item, country, i)
    glb = doc.get("global", {})
    if "life_table" in glb:
        lt = glb["life_table"]
        _require(isinstance(lt, list) or (isinstance(lt, dict) and "csv" in lt),
                 "global.life_table must be a list of {age, le} rows or a mapping with 'csv'")
    return doc
