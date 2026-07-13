from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src_python.validation.run_state_space_hyperparameter_selection import (
    build_hyperparameter_grid,
    candidate_identifier,
    configured_current_hyperparameters,
    leave_one_country_out_selection,
    log1p_interval_score,
    production_selection_decision,
    rank_candidates,
    add_selection_scores,
    validate_publication_countries,
)
from src_python.simulation.common import load_configs


def _candidate(
    rho: float,
    innovation_sd: float,
    dispersion: float,
    *,
    current: bool = False,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_identifier(rho, innovation_sd, dispersion),
        "rho": rho,
        "innovation_sd": innovation_sd,
        "dispersion": dispersion,
        "is_current_configuration": current,
    }


def _fold_row(
    candidate: dict[str, object],
    country: str,
    fold: int,
    *,
    nll: float,
    naive_nll: float = 7.0,
    error: float = 0.05,
    naive_error: float = 0.20,
    lower: float = 8.0,
    upper: float = 12.0,
) -> dict[str, object]:
    observed = 10.0
    width = upper - lower
    return {
        **candidate,
        "country": country,
        "fold": fold,
        "test_year": 2023 + fold,
        "training_end_year": 2022 + fold,
        "evidence_cutoff_date": f"{2023 + fold}-01-01",
        "resistance_evidence_anchor_year": 2022 + fold,
        "resistance_timeline_applied": True,
        "resistance_evidence_years": str(2022 + fold),
        "diagnostic_periods_known_at_origin": 0,
        "future_evidence_used": False,
        "test_observation_intervals": 1,
        "optimizer_success": True,
        "observed_cases": observed,
        "predicted_cases": 10.0,
        "naive_predicted_cases": 13.0,
        "absolute_log1p_error": error,
        "naive_absolute_log1p_error": naive_error,
        "negative_binomial_nll": nll,
        "naive_negative_binomial_nll": naive_nll,
        "process_mixture_negative_binomial_nll": nll,
        "absolute_log1p_error_improvement_over_naive": naive_error - error,
        "negative_binomial_nll_improvement_over_naive": naive_nll - nll,
        "process_mixture_negative_binomial_nll_improvement_over_naive": (
            naive_nll - nll
        ),
        "smape": 0.0,
        "nb95_interval_coverage": 1.0,
        "naive_nb95_interval_coverage": 1.0,
        "nb95_interval_mean_width": width,
        "naive_nb95_interval_mean_width": 5.0,
        "nb95_interval_mean_relative_width": width / 10.0,
        "naive_nb95_interval_mean_relative_width": 0.5,
        "process_predictive_mean_q025": lower,
        "process_predictive_mean_median": 10.0,
        "process_predictive_mean_q975": upper,
        "observation_predictive_q025": lower,
        "observation_predictive_median": 10.0,
        "observation_predictive_q975": upper,
        "observation_predictive_95_coverage": lower <= observed <= upper,
        "observation_predictive_95_width": width,
        "observation_predictive_95_relative_width": width / observed,
        "observation_predictive_95_log1p_width": float(
            np.log1p(upper) - np.log1p(lower)
        ),
        "laplace_boundary_rejection_fraction": 0.0,
        "predictive_process_draws": 8,
        "predictive_measurement_draws": 32,
        "posterior_rank": 6,
        "posterior_dimension": 6,
        "posterior_condition_number": 10.0,
        "process_nfev": 12,
        "state_coordinate_uncertainty_mode": "map_conditioned",
        "predictive_uncertainty_scope": (
            "conditional_on_MAP_state_with_future_AR1_process_and_NB2_measurement"
        ),
        "posterior_coverage_claimed": False,
    }


def _folds(
    candidates: list[dict[str, object]],
    countries: list[str],
    nll_by_candidate_country: dict[tuple[str, str], float],
    *,
    folds: int = 3,
) -> pd.DataFrame:
    rows = [
        _fold_row(
            candidate,
            country,
            fold,
            nll=nll_by_candidate_country[(str(candidate["candidate_id"]), country)],
            error=0.05
            if nll_by_candidate_country[(str(candidate["candidate_id"]), country)] <= 5.0
            else 0.10,
        )
        for candidate in candidates
        for country in countries
        for fold in range(1, folds + 1)
    ]
    return add_selection_scores(pd.DataFrame(rows))


def test_joint_grid_contains_cartesian_product_and_current_candidate() -> None:
    grid = build_hyperparameter_grid([0.2, 0.5], [0.3, 0.6], [20.0, 50.0])

    assert len(grid) == 8
    assert sum(bool(row["is_current_configuration"]) for row in grid) == 1
    assert {
        (row["rho"], row["innovation_sd"], row["dispersion"]) for row in grid
    } == {
        (rho, sd, k)
        for rho in (0.2, 0.5)
        for sd in (0.3, 0.6)
        for k in (20.0, 50.0)
    }
    current = configured_current_hyperparameters()
    flagged = next(row for row in grid if row["is_current_configuration"])
    assert (flagged["rho"], flagged["innovation_sd"], flagged["dispersion"]) == current


@pytest.mark.parametrize(
    "rho, sd, dispersion",
    [([-0.1], [0.3], [50.0]), ([0.5], [0.0], [50.0]), ([0.5], [0.3], [-1.0])],
)
def test_joint_grid_rejects_invalid_axes(rho, sd, dispersion) -> None:
    with pytest.raises(ValueError):
        build_hyperparameter_grid(rho, sd, dispersion)


def test_hyperparameter_selection_rejects_exploratory_country() -> None:
    configs = load_configs()

    with pytest.raises(ValueError, match="outside the prespecified publication set"):
        validate_publication_countries(["Australia", "South_Africa"], configs)


def test_log1p_interval_score_rewards_sharp_covering_interval() -> None:
    sharp = float(log1p_interval_score(10.0, 8.0, 12.0))
    wide = float(log1p_interval_score(10.0, 1.0, 100.0))
    miss = float(log1p_interval_score(20.0, 8.0, 12.0))

    assert sharp < wide
    assert sharp < miss


def test_loco_selection_does_not_use_heldout_country_outcome() -> None:
    first = _candidate(0.2, 0.3, 20.0)
    second = _candidate(0.8, 0.6, 100.0)
    candidates = [first, second]
    countries = ["A", "B", "C"]
    scores = {
        (str(first["candidate_id"]), "A"): 1.0,
        (str(first["candidate_id"]), "B"): 1.0,
        (str(first["candidate_id"]), "C"): 6.0,
        (str(second["candidate_id"]), "A"): 2.0,
        (str(second["candidate_id"]), "B"): 2.0,
        (str(second["candidate_id"]), "C"): 0.5,
    }
    folds = _folds(candidates, countries, scores)

    evaluation, _, rankings = leave_one_country_out_selection(
        folds, candidates, {country: 3 for country in countries}
    )

    heldout_c = evaluation.loc[evaluation["heldout_country"].eq("C")].iloc[0]
    assert heldout_c["selected_candidate_id"] == first["candidate_id"]
    tuning_c = rankings.loc[
        rankings["heldout_country"].eq("C") & rankings["selected"].astype(bool)
    ].iloc[0]
    assert tuning_c["candidate_id"] == first["candidate_id"]


def test_ranking_uses_future_process_mixture_log_score_not_point_mean_nll() -> None:
    mixture_better = _candidate(0.2, 0.3, 20.0)
    point_better = _candidate(0.8, 0.6, 100.0)
    rows = []
    for candidate in (mixture_better, point_better):
        for fold in (1, 2):
            row = _fold_row(candidate, "A", fold, nll=2.0)
            if candidate is mixture_better:
                row["negative_binomial_nll"] = 100.0
                row["process_mixture_negative_binomial_nll"] = 1.0
            else:
                row["negative_binomial_nll"] = 1.0
                row["process_mixture_negative_binomial_nll"] = 2.0
            rows.append(row)
    folds = add_selection_scores(pd.DataFrame(rows))

    ranking = rank_candidates(
        folds, [mixture_better, point_better], {"A": 2}
    )

    assert ranking.loc[ranking["selected"].astype(bool), "candidate_id"].iloc[0] == (
        mixture_better["candidate_id"]
    )


def test_supported_replacement_requires_and_passes_independent_checks() -> None:
    current = _candidate(0.5, 0.6, 50.0, current=True)
    improved = _candidate(0.2, 0.3, 20.0)
    candidates = [current, improved]
    countries = ["A", "B", "C"]
    scores = {
        (str(current["candidate_id"]), country): 6.0 for country in countries
    } | {
        (str(improved["candidate_id"]), country): 5.0 for country in countries
    }
    folds = _folds(candidates, countries, scores)
    expected = {country: 3 for country in countries}
    ranking = rank_candidates(folds, candidates, expected)
    evaluation, selected_folds, _ = leave_one_country_out_selection(
        folds, candidates, expected
    )

    decision = production_selection_decision(
        folds,
        candidates,
        expected,
        ranking,
        evaluation,
        selected_folds,
    ).iloc[0]

    assert decision["recommended_candidate_id"] == improved["candidate_id"]
    assert bool(decision["loco_validation_supports_selection"])
    assert bool(decision["loco_validation_supports_replacement"])
    assert decision["production_action"] == "recommend_config_change_for_manual_review"
    assert not bool(decision["configuration_was_modified"])


def test_two_country_smoke_run_cannot_support_configuration_change() -> None:
    current = _candidate(0.5, 0.6, 50.0, current=True)
    improved = _candidate(0.2, 0.3, 20.0)
    candidates = [current, improved]
    countries = ["A", "B"]
    scores = {
        (str(current["candidate_id"]), country): 6.0 for country in countries
    } | {
        (str(improved["candidate_id"]), country): 5.0 for country in countries
    }
    folds = _folds(candidates, countries, scores)
    expected = {country: 3 for country in countries}
    ranking = rank_candidates(folds, candidates, expected)
    evaluation, selected_folds, _ = leave_one_country_out_selection(
        folds, candidates, expected
    )

    decision = production_selection_decision(
        folds,
        candidates,
        expected,
        ranking,
        evaluation,
        selected_folds,
    ).iloc[0]

    assert not bool(decision["loco_validation_supports_replacement"])
    assert "insufficient_independent_countries" in decision["decision_reasons"]
    assert decision["production_action"] == "do_not_change_configuration"
    assert decision["recommended_candidate_id"] == ""
    assert decision["provisional_all_country_best_candidate_id"] == improved[
        "candidate_id"
    ]
