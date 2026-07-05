from __future__ import annotations

import pandas as pd

from src_python.model.parameters import PreparedParameters
from src_python.simulation.common import load_configs
from src_python.simulation.common import make_config
from src_python.utils.io import project_path


def test_diagnostic_standard_timeline_covers_model_countries() -> None:
    configs = load_configs()
    countries = set(configs["countries"])
    timeline = pd.read_csv(project_path("data/raw/pertussis_diagnostic_standards_timeline.csv"))

    assert countries.issubset(set(timeline["country"]))


def test_diagnostic_standard_timeline_has_valid_priors_and_sources() -> None:
    timeline = pd.read_csv(project_path("data/raw/pertussis_diagnostic_standards_timeline.csv"))
    sources = pd.read_csv(project_path("data/raw/pertussis_diagnostic_standards_sources.csv"))
    source_ids = set(sources["source_id"])

    required = {
        "country",
        "period_start",
        "period_end",
        "relative_detection_prior_mean",
        "relative_detection_prior_lower",
        "relative_detection_prior_upper",
        "source_ids",
    }
    assert required.issubset(timeline.columns)

    starts = pd.to_datetime(timeline["period_start"], errors="coerce")
    ends = pd.to_datetime(timeline["period_end"], errors="coerce")
    assert starts.notna().all()
    assert ends.notna().all()
    assert ends.ge(starts).all()

    lower = timeline["relative_detection_prior_lower"].astype(float)
    mean = timeline["relative_detection_prior_mean"].astype(float)
    upper = timeline["relative_detection_prior_upper"].astype(float)
    assert lower.gt(0.0).all()
    assert lower.le(mean).all()
    assert mean.le(upper).all()

    referenced = {
        source_id
        for raw in timeline["source_ids"].astype(str)
        for source_id in raw.split(";")
    }
    assert referenced.issubset(source_ids)


def test_diagnostic_standard_notes_do_not_encode_transmission_dynamics() -> None:
    timeline = pd.read_csv(project_path("data/raw/pertussis_diagnostic_standards_timeline.csv"))
    text = (
        timeline["case_definition_or_reporting_change"].fillna("").astype(str)
        + " "
        + timeline["notes"].fillna("").astype(str)
    ).str.lower()

    for forbidden in ("immunity debt", "reduced circulation", "true infections"):
        assert not text.str.contains(forbidden, regex=False).any()


def test_diagnostic_standard_timeline_is_wired_into_country_config() -> None:
    config = make_config(country_profile="China", load_calibration=False)
    variation = config["diagnostic_reporting_time_variation"]

    assert variation["enabled"] is True
    assert variation["country"] == "China"
    assert variation["periods"]


def test_diagnostic_standard_multiplier_uses_calendar_periods() -> None:
    config = make_config(country_profile="China", load_calibration=False)
    config["calendar"]["analysis_start_date"] = "2024-01-01"
    config["simulation"]["start_time"] = 0.0
    config["simulation"]["end_time"] = 365.0

    params = PreparedParameters.from_config(config, analysis="test", scenario="diagnostic_standard")

    # 2024-01-01 to 2024-03-31: early resurgence passive (mean=2.00)
    assert params.diagnostic_reporting_multiplier_at(0.0) == 2.00
    # 2024-04-01 onwards: enhanced PCR screening resurgence, detection-only prior
    assert params.diagnostic_reporting_multiplier_at(100.0) == 2.25
    assert (params.reporting_rate_at(100.0) >= params.reporting_rate).all()


def test_diagnostic_standard_adjustment_can_be_disabled() -> None:
    config = make_config(
        country_profile="China",
        load_calibration=False,
        config_overrides={"observation_model": {"diagnostic_standards": {"enabled": False}}},
    )
    params = PreparedParameters.from_config(config, analysis="test", scenario="diagnostic_standard_disabled")

    assert "diagnostic_reporting_time_variation" not in config
    assert params.diagnostic_reporting_multiplier_at(0.0) == 1.0
