from __future__ import annotations

import pandas as pd
import pytest

from src_python.data.build_country_inputs import (
    _annualized_incidence_table,
    _normalize_incidence_frame,
)


def test_known_source_date_mismatches_use_explicit_resolution() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2025-01-01", "2026-01-01", "2016-04-01"],
            "Year": [2025, 2025, 2017],
            "Month": [1, 1, 4],
            "Week": [None, None, None],
            "Disease": ["Pertussis"] * 3,
            "Cases": [5611, 971, 73],
            "Country": ["CN", "CN", "NZ"],
            "URL": [""] * 3,
            "source_sheet": ["CN", "CN", "NZ"],
        }
    )

    normalized = _normalize_incidence_frame(raw)

    assert normalized["period_start"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2026-01-01",
        "2017-04-01",
    ]
    assert normalized["period_date_resolution"].tolist() == [
        "consistent_year_month",
        "explicit_source_date_override",
        "explicit_year_month_override",
    ]
    assert normalized["source_year"].tolist() == [2025, 2025, 2017]
    assert normalized["source_month"].tolist() == [1, 1, 4]
    assert normalized["Year"].tolist() == [2025, 2026, 2017]
    assert normalized["Month"].tolist() == [1, 1, 4]
    assert not normalized.duplicated(["Country", "period_start", "period_end"]).any()
    annual_cases = (
        _annualized_incidence_table(normalized)
        .set_index("Year")["Cases"]
        .to_dict()
    )
    assert annual_cases == {2017: 73, 2025: 5611, 2026: 971}


def test_unknown_date_mismatch_is_rejected_for_manual_review() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2022-05-01"],
            "Year": [2021],
            "Month": [5],
            "Disease": ["Pertussis"],
            "Cases": [1],
            "Country": ["CN"],
        }
    )

    with pytest.raises(ValueError, match="Unresolved monthly incidence"):
        _normalize_incidence_frame(raw)
