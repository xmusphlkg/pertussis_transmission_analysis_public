from __future__ import annotations

import numpy as np
import pandas as pd

from src_python.validation.audit_observation_sources import (
    classify_source_reconciliation,
    country_source_summary,
)


def test_source_reconciliation_classifies_scale_mismatch_and_contract() -> None:
    source = pd.DataFrame(
        {
            "config_key": ["A", "A", "B"],
            "year": [2022, 2023, 2023],
            "native_cases": [100.0, 40.0, 10.0],
            "native_intervals": [12, 12, 12],
            "age_known_cases": [np.nan, 100.0, np.nan],
            "who_cases": [105.0, np.nan, np.nan],
        }
    )

    result = classify_source_reconciliation(source)

    assert result.loc[0, "reconciliation_status"] == "compatible_scale"
    assert result.loc[1, "reconciliation_status"] == "material_scale_mismatch"
    assert result.loc[2, "reconciliation_status"] == "no_national_comparator"
    assert "no_absolute_burden_claim" in result.loc[1, "model_use_contract"]


def test_country_summary_requires_three_compatible_comparisons() -> None:
    source = pd.DataFrame(
        {
            "config_key": ["A", "A", "A", "B"],
            "year": [2021, 2022, 2023, 2023],
            "native_cases": [100.0, 110.0, 90.0, 40.0],
            "native_intervals": [12, 12, 12, 12],
            "age_known_cases": [np.nan, np.nan, np.nan, np.nan],
            "who_cases": [100.0, 100.0, 100.0, 100.0],
        }
    )

    summary = country_source_summary(classify_source_reconciliation(source)).set_index(
        "country"
    )

    assert bool(summary.loc["A", "absolute_burden_claim_allowed"])
    assert not bool(summary.loc["B", "absolute_burden_claim_allowed"])
