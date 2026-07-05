from __future__ import annotations

import pytest

from src_python.simulation.common import make_intervention_config
from src_python.simulation.run_routine_timeliness_sensitivity import (
    _apply_timeliness,
    _parse_schedule_age_months,
    timeliness_target_distribution_from_schedule,
)


def test_parse_schedule_age_pattern_handles_week_month_and_year_tokens() -> None:
    ages = _parse_schedule_age_months("W6;W10;W14;M18;Y6;Y12")

    assert ages[0] == pytest.approx(1.38, rel=0.01)
    assert ages[1] == pytest.approx(2.30, rel=0.01)
    assert ages[2] == pytest.approx(3.22, rel=0.01)
    assert ages[-1] == pytest.approx(144.0)


def test_timeliness_target_distribution_is_schedule_relative_for_infants() -> None:
    early = {"metadata": {"routine_age_pattern": "W6;W10;W14;M18"}}
    later = {"metadata": {"routine_age_pattern": "M3;M5;M12;Y5"}}

    early_distribution = timeliness_target_distribution_from_schedule(early)["infant_3_11m"]
    later_distribution = timeliness_target_distribution_from_schedule(later)["infant_3_11m"]

    assert early_distribution["recent"] > 0.80
    assert later_distribution.get("recent", 0.0) == pytest.approx(0.0)
    assert later_distribution["dose2_recent"] > later_distribution["dose1_recent"]


def test_apply_timeliness_uses_country_schedule_metadata() -> None:
    south_africa_config, _ = make_intervention_config("current", country_profile="South_Africa")
    sweden_config, _ = make_intervention_config("current", country_profile="Sweden")

    south_africa = _apply_timeliness(south_africa_config)
    sweden = _apply_timeliness(sweden_config)

    south_africa_target = south_africa["routine_vaccination"]["target_origin_distribution_by_age"]["infant_3_11m"]
    sweden_target = sweden["routine_vaccination"]["target_origin_distribution_by_age"]["infant_3_11m"]

    assert south_africa["metadata"]["timeliness_definition"] == "schedule_relative"
    assert south_africa_target["recent"] > 0.80
    assert sweden_target.get("recent", 0.0) == pytest.approx(0.0)
    assert sweden_target["dose2_recent"] > 0.70
