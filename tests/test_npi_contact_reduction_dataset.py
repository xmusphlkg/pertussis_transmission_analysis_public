from __future__ import annotations

import pandas as pd

from src_python.simulation.common import load_configs
from src_python.utils.io import project_path


def test_npi_contact_reduction_timeline_covers_model_countries() -> None:
    configs = load_configs()
    countries = set(configs["countries"])
    timeline = pd.read_csv(project_path("data/raw/covid_npi_contact_reduction_timeline.csv"))

    assert countries.issubset(set(timeline["country"]))


def test_npi_contact_reduction_timeline_has_valid_periods_and_bounds() -> None:
    timeline = pd.read_csv(project_path("data/raw/covid_npi_contact_reduction_timeline.csv"))

    required = {
        "country",
        "iso3",
        "period_start",
        "period_end",
        "baseline_contact_reduction",
        "contact_reduction_mean",
        "contact_reduction_lower",
        "contact_reduction_upper",
        "ramp_days",
        "evidence_strength",
        "notes",
    }
    assert required.issubset(timeline.columns)

    starts = pd.to_datetime(timeline["period_start"], errors="coerce")
    ends = pd.to_datetime(timeline["period_end"], errors="coerce")
    assert starts.notna().all()
    assert ends.notna().all()
    assert ends.ge(starts).all()

    baseline = timeline["baseline_contact_reduction"].astype(float)
    lower = timeline["contact_reduction_lower"].astype(float)
    mean = timeline["contact_reduction_mean"].astype(float)
    upper = timeline["contact_reduction_upper"].astype(float)
    assert lower.between(0.0, 1.0).all()
    assert mean.between(0.0, 1.0).all()
    assert upper.between(0.0, 1.0).all()
    assert lower.le(mean).all()
    assert mean.le(upper).all()
    assert timeline["ramp_days"].astype(float).ge(0.0).all()
    assert baseline.between(0.0, 1.0).all()
