"""Loading and validating core2026 configuration files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config" / "core2026"
PRIORS_PATH = CONFIG_DIR / "priors.yaml"

DISTRIBUTION_FIELDS = {
    "lognormal": ("median", "log_sd"),
    "beta": ("a", "b"),
    "normal": ("mean", "sd"),
    "uniform": ("lower", "upper"),
    "random_walk_logit": ("first_band_mean_logit", "first_band_sd", "step_sd", "bands"),
    "lognormal_from_timeline": (),
}
VERIFICATION_LEVELS = {"fulltext", "citation", "author"}


@dataclass(frozen=True)
class PriorSpec:
    name: str
    scope: str
    estimated: bool
    distribution: str | None
    settings: dict[str, Any]
    value: Any
    bounds: tuple[float, float] | None
    sources: tuple[str, ...]


def config_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_priors(path: Path = PRIORS_PATH) -> tuple[dict[str, PriorSpec], dict[str, dict]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    references = raw.get("references", {})
    for key, ref in references.items():
        if ref.get("verification") not in VERIFICATION_LEVELS:
            raise ValueError(f"reference {key} has invalid verification status")
    specs: dict[str, PriorSpec] = {}
    for name, entry in raw["parameters"].items():
        scope = entry.get("scope")
        if scope not in {"structural", "observation"}:
            raise ValueError(f"{name}: scope must be structural or observation")
        estimated = bool(entry.get("estimated"))
        dist = entry.get("distribution")
        if estimated:
            if dist not in DISTRIBUTION_FIELDS:
                raise ValueError(f"{name}: unknown distribution {dist!r}")
            missing = [f for f in DISTRIBUTION_FIELDS[dist] if f not in entry]
            if missing:
                raise ValueError(f"{name}: missing {missing}")
        elif "value" not in entry:
            raise ValueError(f"{name}: fixed parameter needs a value")
        bounds = entry.get("bounds")
        if bounds is not None:
            lo, hi = float(bounds[0]), float(bounds[1])
            if not lo < hi:
                raise ValueError(f"{name}: invalid bounds")
            bounds = (lo, hi)
        sources = tuple(entry.get("sources", []))
        unknown = [s for s in sources if s not in references]
        if unknown:
            raise ValueError(f"{name}: unknown sources {unknown}")
        settings = {k: v for k, v in entry.items() if k in DISTRIBUTION_FIELDS.get(dist or "", ())}
        specs[name] = PriorSpec(
            name=name,
            scope=scope,
            estimated=estimated,
            distribution=dist,
            settings=settings,
            value=entry.get("value"),
            bounds=bounds,
            sources=sources,
        )
    return specs, references
