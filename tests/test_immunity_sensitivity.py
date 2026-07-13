from __future__ import annotations

import pytest

from src_python.simulation.run_immunity_sensitivity import (
    IMMUNITY_STRUCTURES,
    _apply_immunity_structure,
)


def _config() -> dict:
    return {
        "natural_history": {
            "recovered_immunity_duration": 90.0,
            "R_to_W_duration": 100.0,
            "W_to_S_duration": 200.0,
            "vaccine_protection_duration": 400.0,
        },
        "immunity_model": {
            "boosting_enabled": True,
            "waned_vaccine_duration": 800.0,
        },
    }


def test_no_boosting_uses_single_stage_with_matched_total_mean() -> None:
    resolved = _apply_immunity_structure(
        _config(),
        "sirws_no_boosting",
        IMMUNITY_STRUCTURES["sirws_no_boosting"],
    )

    assert resolved["immunity_model"]["boosting_enabled"] is False
    assert resolved["natural_history"]["recovered_immunity_duration"] == pytest.approx(300.0)


def test_short_vaccine_immunity_changes_both_stages_to_three_year_total() -> None:
    resolved = _apply_immunity_structure(
        _config(),
        "short_vaccine_immunity",
        IMMUNITY_STRUCTURES["short_vaccine_immunity"],
    )

    assert resolved["natural_history"]["vaccine_protection_duration"] == pytest.approx(547.5)
    assert resolved["immunity_model"]["waned_vaccine_duration"] == pytest.approx(547.5)
