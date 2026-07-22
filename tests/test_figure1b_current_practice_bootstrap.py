from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src_python.simulation.run_figure1b_current_practice_bootstrap as bootstrap


def test_interval_summary_contains_all_current_practice_endpoints() -> None:
    rows = []
    for country_index, country in enumerate(("A", "B"), start=1):
        for replicate in range(1, 101):
            rows.append(
                {
                    "country": country,
                    "bootstrap_replicate": replicate,
                    "reports_rate_per_100k": 10 * country_index + replicate,
                    "symptomatic_rate_per_100k": 100 * country_index + 2 * replicate,
                    "infections_rate_per_100k": 500 * country_index + 4 * replicate,
                }
            )

    intervals, stability = bootstrap._summarise_intervals(
        pd.DataFrame(rows), countries=["A", "B"], stability_blocks=5
    )

    assert len(intervals) == 6
    assert len(stability) == 6
    assert set(intervals["outcome"]) == {"Reports", "Symptomatic", "Infections"}
    assert intervals["bootstrap_replicates"].eq(100).all()
    assert intervals["interval_type"].eq(bootstrap.INTERVAL_TYPE).all()
    assert intervals["confidence_interval_method"].eq(
        bootstrap.INTERVAL_METHOD
    ).all()
    assert (intervals["rate_q025"] < intervals["rate_q975"]).all()
    assert np.isfinite(stability["maximum_endpoint_mcse"]).all()
    assert np.isfinite(stability["tail_probability_mcse"]).all()


def test_draw_contract_has_one_column_per_figure1b_endpoint() -> None:
    assert bootstrap.DRAW_COLUMNS == {
        "Reports": "reports_rate_per_100k",
        "Symptomatic": "symptomatic_rate_per_100k",
        "Infections": "infections_rate_per_100k",
    }
    assert set(bootstrap.OUTCOME_COLUMNS) == set(bootstrap.DRAW_COLUMNS)


def test_bootstrap_conditions_on_the_accepted_fitted_latent_path() -> None:
    observed = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "period_end": pd.to_datetime(["2021-01-01", "2022-01-01"]),
        }
    )
    config = {
        "transmission": {
            "log_beta_time_variation": {
                "enabled": True,
                "periods": [
                    {"start_date": "2020-01-01", "log_multiplier": 0.25},
                    {"start_date": "2021-01-01", "log_multiplier": -0.40},
                ],
            }
        }
    }
    np.testing.assert_allclose(
        bootstrap._fitted_process_state_values(config, observed),
        np.asarray([0.25, -0.40]),
    )
    source = inspect.getsource(bootstrap._run_current_practice_task)
    assert "_fitted_process_state_values" in source
    assert "_draw_bounded_ar1_path" not in source
    assert "conditional parametric bootstrap" in bootstrap.INTERVAL_BASIS.lower()
    assert "fitted annual latent transmission path" in bootstrap.INTERVAL_BASIS.lower()

    config["transmission"]["log_beta_time_variation"]["enabled"] = False
    with pytest.raises(RuntimeError, match="accepted fitted latent path"):
        bootstrap._fitted_process_state_values(config, observed)
