from __future__ import annotations

import numpy as np

from src_python.simulation.run_individual_stochastic_toy import (
    SETTINGS,
    _setting_share_array,
    _synthetic_setting_shares,
    build_tables,
)


def test_synthetic_setting_shares_are_normalized_and_household_heavy_for_infants() -> None:
    infant_split = _synthetic_setting_shares("young_adult_18_39y", "infant_0_2m")
    adult_split = _synthetic_setting_shares("young_adult_18_39y", "middle_adult_40_64y")

    assert set(infant_split) == set(SETTINGS)
    assert np.isclose(sum(infant_split.values()), 1.0)
    assert np.isclose(sum(adult_split.values()), 1.0)
    assert infant_split["home"] > adult_split["home"]
    assert adult_split["work"] > infant_split["work"]


def test_setting_share_array_has_expected_dimensions() -> None:
    shares = _setting_share_array()

    assert shares.shape == (8, 8, len(SETTINGS))
    assert np.allclose(shares.sum(axis=2), 1.0)


def test_build_tables_smoke_run_uses_contactdata_setting_matrices() -> None:
    replicates, summary, contact_audit, setting_decomposition = build_tables(
        countries=("Australia",),
        replicates=2,
        population_size=700,
        seed=9,
    )

    assert len(replicates) == 2 * 3
    assert len(summary) == 3
    assert len(contact_audit) == 1
    assert len(setting_decomposition) == 64
    assert contact_audit["raw_setting_matrix_available"].all()
    assert contact_audit["processed_setting_matrix_available"].all()
    assert summary["setting_matrix_available"].all()
    assert set(setting_decomposition["setting_decomposition"]) == {"contactdata_location_specific_8group_matrices"}
    assert "structural_sensitivity_caveat" in summary.columns
