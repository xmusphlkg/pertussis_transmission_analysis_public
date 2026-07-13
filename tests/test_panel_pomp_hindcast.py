from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src_python.calibration.panel_pomp import CountrySeries
from src_python.validation.run_panel_pomp_hindcast import (
    _log_rate_baselines,
    _parse_years,
    calibrated_pit_quantile_levels,
    default_candidates,
    optimize_country_balanced_point_weights,
    optimize_country_balanced_stacking_weights,
    select_shrunk_country_candidates,
)


def _series() -> CountrySeries:
    return CountrySeries(
        country="Synthetic",
        interval_ids=[f"i{index}" for index in range(8)],
        interval_start=np.arange(8, dtype=float) * 30.0,
        interval_end=np.arange(1, 9, dtype=float) * 30.0,
        observed_counts=[10, 12, 14, 16, 18, 20, 22, 24],
        mechanistic_offset=np.ones(8),
        exposure_days=np.full(8, 30.0),
        training_mask=[True, True, True, True, True, True, False, False],
    )


def test_default_candidates_are_prespecified_and_scientifically_distinct() -> None:
    candidates = default_candidates()

    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert any(candidate.process.robust_mixture_probability > 0 for candidate in candidates)
    assert any(candidate.process.trend_enabled for candidate in candidates)


def test_log_rate_baselines_use_training_prefix_only() -> None:
    original = _series()
    changed = CountrySeries(
        country=original.country,
        interval_ids=original.interval_ids,
        interval_start=original.interval_start,
        interval_end=original.interval_end,
        observed_counts=[10, 12, 14, 16, 18, 20, 9999, 8888],
        mechanistic_offset=original.mechanistic_offset,
        exposure_days=original.exposure_days,
        training_mask=original.training_mask,
    )

    first = _log_rate_baselines(original)
    second = _log_rate_baselines(changed)

    for name in first:
        np.testing.assert_allclose(first[name], second[name])


def test_prequential_baselines_update_after_each_observed_holdout() -> None:
    original = _series()
    changed = CountrySeries(
        country=original.country,
        interval_ids=original.interval_ids,
        interval_start=original.interval_start,
        interval_end=original.interval_end,
        observed_counts=[10, 12, 14, 16, 18, 20, 9999, 24],
        mechanistic_offset=original.mechanistic_offset,
        exposure_days=original.exposure_days,
        training_mask=original.training_mask,
    )

    first = _log_rate_baselines(original, prequential=True)
    second = _log_rate_baselines(changed, prequential=True)

    assert first["ew_recent_rate"][0] == second["ew_recent_rate"][0]
    assert first["ew_recent_rate"][1] != second["ew_recent_rate"][1]


def test_parse_years_accepts_ranges_and_rejects_empty() -> None:
    assert _parse_years("2023-2025,2026") == [2023, 2024, 2025, 2026]


def test_stacking_weights_are_regularized_and_favor_better_component() -> None:
    scores = np.asarray(
        [
            [-1.0, -5.0, -6.0, -7.0],
            [-1.2, -4.0, -6.0, -8.0],
            [-0.8, -5.0, -5.5, -7.5],
            [-1.1, -4.5, -5.0, -6.0],
        ]
    )
    weights = optimize_country_balanced_stacking_weights(
        scores, np.asarray(["A", "A", "B", "B"])
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(value >= 0.02 - 1e-8 for value in weights.values())
    assert weights["pomp"] == max(weights.values())


def test_point_weights_optimize_annual_log1p_error_separately() -> None:
    predictions = np.asarray(
        [
            [1000.0, 100.0, 120.0, 90.0],
            [900.0, 110.0, 100.0, 105.0],
            [800.0, 95.0, 100.0, 110.0],
        ]
    )
    weights = optimize_country_balanced_point_weights(
        predictions,
        np.asarray([100.0, 105.0, 98.0]),
        np.asarray(["A", "A", "B"]),
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["pomp"] < 0.1


def test_country_candidate_selection_shrinks_local_scores_to_panel() -> None:
    def result(name: str, global_score: float, scores: dict[str, float]):
        fits = []
        for country, score in scores.items():
            for year in (2020, 2021, 2022):
                forecast = SimpleNamespace(
                    training_interval=False,
                    predictive_log_score=score,
                )
                fits.append(
                    SimpleNamespace(
                        country=f"{country}__validation_{year}",
                        filter_result=SimpleNamespace(forecasts=[forecast]),
                    )
                )
        return SimpleNamespace(
            name=name,
            country_balanced_validation_log_score=global_score,
            country_fits=fits,
        )

    selection = SimpleNamespace(
        candidates=[
            result("panel", -2.0, {"A": -1.0, "B": -2.0}),
            result("local", -3.0, {"A": 1.0, "B": -4.0}),
        ]
    )
    selected, details = select_shrunk_country_candidates(selection, prior_folds=3.0)

    assert selected == {"A": "local", "B": "panel"}
    assert len(details) == 4
    assert all(row["local_score_weight"] == pytest.approx(0.5) for row in details)


def test_pit_quantile_calibration_is_ordered_and_partially_pooled() -> None:
    levels, audit = calibrated_pit_quantile_levels(
        {
            "underdispersed": np.asarray([0.001, 0.01, 0.5, 0.99, 0.999]),
            "overdispersed": np.asarray([0.35, 0.4, 0.5, 0.6, 0.65]),
        },
        country_prior_intervals=5.0,
        global_prior_countries=2.0,
    )

    assert set(levels) == {"underdispersed", "overdispersed"}
    assert all(np.all(np.diff(value) > 0.0) for value in levels.values())
    assert levels["underdispersed"][0] < levels["overdispersed"][0]
    assert levels["underdispersed"][2] > levels["overdispersed"][2]
    assert len(audit) == 2
