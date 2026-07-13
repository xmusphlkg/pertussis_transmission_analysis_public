from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src_python.model.observables import CaseExposure, ObservationPlan
from src_python.simulation import run_bayesian_uncertainty as bayes
from src_python.simulation.common import make_config


def _three_interval_exposure(age_groups: tuple[str, ...]) -> CaseExposure:
    n_age = len(age_groups)
    plan = ObservationPlan(
        interval_ids=("a", "b", "c"),
        interval_start_times=np.array([0.0, 1.0, 2.0]),
        interval_end_times=np.array([1.0, 2.0, 3.0]),
        segment_start_times=np.array([0.0, 1.0, 2.0]),
        segment_end_times=np.array([1.0, 2.0, 3.0]),
        segment_interval_index=np.array([0, 1, 2], dtype=np.int64),
        diagnostic_multipliers=np.ones(3),
        reporting_trend_multipliers=np.ones(3),
        endpoint_times=np.array([0.0, 1.0, 2.0, 3.0]),
        age_groups=age_groups,
    )
    return CaseExposure(
        plan=plan,
        case_counts=np.tile(np.array([[10.0], [20.0], [30.0]]), (1, n_age)),
        case_event_definition="infection_destined_symptomatic",
        numerical_diagnostics={},
    )


def test_reporting_only_proposal_reuses_biological_exposure(monkeypatch) -> None:
    config = make_config(country_profile="Australia", load_calibration=False)
    age_groups = tuple(str(record["label"]) for record in config["age_groups"])
    exposure = _three_interval_exposure(age_groups)
    base_rates = np.array(
        [float(record["reporting_rate"]) for record in config["age_groups"]],
        dtype=float,
    )
    calls = 0

    def fake_exposure(*args, **kwargs):
        nonlocal calls
        calls += 1
        return exposure, base_rates

    monkeypatch.setattr(bayes, "run_prepared_case_exposure", fake_exposure)
    observed = pd.DataFrame({"reported_cases": [2.0, 4.0, 6.0]})
    cache: dict[object, object] = {}

    first = bayes._log_likelihood(config, observed, "Australia", 50.0, _cache=cache)
    reporting_proposal = deepcopy(config)
    reporting_proposal["reporting_multiplier"] = (
        float(config.get("reporting_multiplier", 1.0)) * 1.5
    )
    second = bayes._log_likelihood(
        reporting_proposal,
        observed,
        "Australia",
        50.0,
        _cache=cache,
    )

    assert calls == 1
    assert np.isfinite(first)
    assert np.isfinite(second)
    assert first != second


def test_biological_proposal_invalidates_exposure_cache(monkeypatch) -> None:
    config = make_config(country_profile="Australia", load_calibration=False)
    age_groups = tuple(str(record["label"]) for record in config["age_groups"])
    exposure = _three_interval_exposure(age_groups)
    base_rates = np.array(
        [float(record["reporting_rate"]) for record in config["age_groups"]],
        dtype=float,
    )
    calls = 0

    def fake_exposure(*args, **kwargs):
        nonlocal calls
        calls += 1
        return exposure, base_rates

    monkeypatch.setattr(bayes, "run_prepared_case_exposure", fake_exposure)
    observed = pd.DataFrame({"reported_cases": [2.0, 4.0, 6.0]})
    cache: dict[object, object] = {}

    bayes._log_likelihood(config, observed, "Australia", 50.0, _cache=cache)
    biological_proposal = deepcopy(config)
    biological_proposal["transmission"]["beta_S"] *= 1.01
    bayes._log_likelihood(
        biological_proposal,
        observed,
        "Australia",
        50.0,
        _cache=cache,
    )

    assert calls == 2


def test_observation_plan_change_invalidates_exposure_cache(monkeypatch) -> None:
    config = make_config(country_profile="Australia", load_calibration=False)
    age_groups = tuple(str(record["label"]) for record in config["age_groups"])
    exposure = _three_interval_exposure(age_groups)
    base_rates = np.array(
        [float(record["reporting_rate"]) for record in config["age_groups"]],
        dtype=float,
    )
    calls = 0

    def fake_exposure(*args, **kwargs):
        nonlocal calls
        calls += 1
        return exposure, base_rates

    monkeypatch.setattr(bayes, "run_prepared_case_exposure", fake_exposure)
    observed = pd.DataFrame({"reported_cases": [2.0, 4.0, 6.0]})
    cache: dict[object, object] = {}
    bayes._log_likelihood(config, observed, "Australia", 50.0, _cache=cache)

    changed = deepcopy(config)
    changed["diagnostic_reporting_time_variation"] = {
        "enabled": True,
        "periods": [
            {
                "start_date": "2020-01-01",
                "end_date": "2020-01-31",
                "multiplier": 0.8,
            }
        ],
    }
    bayes._log_likelihood(changed, observed, "Australia", 50.0, _cache=cache)

    assert calls == 2


def test_case_exposure_cache_is_bounded(monkeypatch) -> None:
    config = make_config(country_profile="Australia", load_calibration=False)
    age_groups = tuple(str(record["label"]) for record in config["age_groups"])
    exposure = _three_interval_exposure(age_groups)
    base_rates = np.array(
        [float(record["reporting_rate"]) for record in config["age_groups"]],
        dtype=float,
    )
    monkeypatch.setattr(
        bayes,
        "run_prepared_case_exposure",
        lambda *args, **kwargs: (exposure, base_rates),
    )
    observed = pd.DataFrame({"reported_cases": [2.0, 4.0, 6.0]})
    cache: dict[object, object] = {}

    for index in range(bayes.MAX_CASE_EXPOSURE_CACHE_ENTRIES + 5):
        proposal = deepcopy(config)
        proposal["transmission"]["beta_S"] *= 1.0 + 0.001 * index
        bayes._log_likelihood(
            proposal,
            observed,
            "Australia",
            50.0,
            _cache=cache,
        )

    exposure_entries = [
        key
        for key in cache
        if isinstance(key, tuple) and key and key[0] == "case_exposure"
    ]
    assert len(exposure_entries) == bayes.MAX_CASE_EXPOSURE_CACHE_ENTRIES
