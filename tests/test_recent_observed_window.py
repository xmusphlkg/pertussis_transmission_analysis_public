from __future__ import annotations

import pandas as pd

from src_python.calibration.calibrate_baseline import retain_recent_observed_window


def test_recent_window_uses_interval_midpoint_at_cross_year_boundary() -> None:
    observed = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2019-12-30", "2020-01-06", "2021-01-01"]),
            "period_end": pd.to_datetime(["2020-01-06", "2020-01-13", "2021-01-08"]),
            "reported_cases": [85.0, 10.0, 20.0],
        }
    )

    retained = retain_recent_observed_window(observed, recent_years=2)

    assert retained["reported_cases"].tolist() == [85.0, 10.0, 20.0]


def test_recent_window_rejects_invalid_calendar_intervals() -> None:
    observed = pd.DataFrame(
        {
            "period_start": ["not-a-date"],
            "period_end": ["2020-01-02"],
            "reported_cases": [1.0],
        }
    )

    try:
        retain_recent_observed_window(observed, recent_years=2)
    except ValueError as exc:
        assert "invalid dates" in str(exc)
    else:
        raise AssertionError("invalid interval should be rejected")
