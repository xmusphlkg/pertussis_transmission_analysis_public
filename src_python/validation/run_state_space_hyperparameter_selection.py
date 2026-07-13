"""Cross-country selection of fixed state-space hyperparameters.

The annual latent-transmission model currently treats the AR(1) persistence,
innovation scale, and NB2 dispersion as fixed.  This runner evaluates a small
joint grid without estimating those quantities from the same country's held-
out outcome used to judge them:

1. run identical rolling one-year-ahead folds for every country/candidate;
2. for each held-out country, rank candidates using *other* countries only;
3. evaluate the selected candidate once on the held-out country, against both
   the exposure-adjusted last-year forecast and the current fixed candidate;
4. select a provisional production candidate using all countries only after
   the leave-one-country-out (LOCO) evaluation has been recorded.

The script never edits ``config/model_settings.yaml``.  A replacement is only
supported when prespecified adequacy, sharpness, stability, sample-size, and
out-of-country improvement checks all pass.  Smaller smoke runs remain useful
for checking execution but are labelled insufficient for configuration choice.
"""

from __future__ import annotations

import argparse
from itertools import product
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src_python.simulation.common import load_configs, publication_country_names
from src_python.utils.io import project_path, write_dataframe
from src_python.utils.parallel import parallel_map
from src_python.validation.run_calibration_hindcast import (
    MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH,
    MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE,
    _aggregate_hindcast_metrics,
    expected_hindcast_fold_counts,
    hindcast_execution_failures,
    run_country_hindcast,
)


DEFAULT_RHO_VALUES = (0.20, 0.50, 0.80)
DEFAULT_INNOVATION_SD_VALUES = (0.30, 0.60)
DEFAULT_DISPERSION_VALUES = (20.0, 50.0, 100.0)

INTERVAL_ALPHA = 0.05
MINIMUM_INDEPENDENT_COUNTRIES = 3
MINIMUM_EVALUATION_FOLDS_PER_COUNTRY = 3
MINIMUM_HELDOUT_COUNTRY_PASS_FRACTION = 2.0 / 3.0
# A production recommendation is for one fixed candidate, so every outer LOCO
# split must independently select that same candidate. A weaker majority rule
# would validate a mixture of tuning decisions rather than the recommended
# fixed hyperparameter triple.
MINIMUM_SELECTION_STABILITY_FRACTION = 1.00

FAILURE_COLUMNS = (
    "candidate_id",
    "country",
    "rho",
    "innovation_sd",
    "dispersion",
    "failure_type",
    "failure_message",
)


def configured_current_hyperparameters(
    configs: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Read the production baseline directly from the calibration config."""

    resolved = load_configs() if configs is None else configs
    calibration = resolved["baseline"]["calibration"]
    process = calibration["process_model"]
    current = (
        float(process["ar1_rho"]),
        float(process["log_beta_innovation_sd"]),
        float(calibration["dispersion"]),
    )
    rho, innovation_sd, dispersion = current
    if not 0.0 <= rho < 0.99 or innovation_sd <= 0.0 or dispersion <= 0.0:
        raise ValueError("Configured state-space hyperparameters are invalid")
    return current


def _number_token(value: float) -> str:
    token = f"{float(value):.8g}"
    return token.replace("-", "m").replace(".", "p").replace("+", "")


def candidate_identifier(rho: float, innovation_sd: float, dispersion: float) -> str:
    """Return a stable, filesystem-safe identifier for one grid point."""

    return (
        f"rho_{_number_token(rho)}__sd_{_number_token(innovation_sd)}"
        f"__k_{_number_token(dispersion)}"
    )


def build_hyperparameter_grid(
    rho_values: Sequence[float],
    innovation_sd_values: Sequence[float],
    dispersion_values: Sequence[float],
    *,
    current_configuration: tuple[float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Validate and construct the full factorial hyperparameter grid (not OAT)."""

    if not rho_values or not innovation_sd_values or not dispersion_values:
        raise ValueError("Every hyperparameter axis must contain at least one value")
    rhos = [float(value) for value in rho_values]
    innovation_sds = [float(value) for value in innovation_sd_values]
    dispersions = [float(value) for value in dispersion_values]
    if any(not np.isfinite(value) or not 0.0 <= value < 0.99 for value in rhos):
        raise ValueError("AR(1) rho grid values must be finite and lie in [0, 0.99)")
    if any(not np.isfinite(value) or value <= 0.0 for value in innovation_sds):
        raise ValueError("Innovation-SD grid values must be finite and positive")
    if any(not np.isfinite(value) or value <= 0.0 for value in dispersions):
        raise ValueError("NB2 dispersion grid values must be finite and positive")

    current_rho, current_innovation_sd, current_dispersion = (
        configured_current_hyperparameters()
        if current_configuration is None
        else tuple(float(value) for value in current_configuration)
    )
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for rho, innovation_sd, dispersion in product(rhos, innovation_sds, dispersions):
        key = (rho, innovation_sd, dispersion)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "candidate_id": candidate_identifier(*key),
                "rho": rho,
                "innovation_sd": innovation_sd,
                "dispersion": dispersion,
                "is_current_configuration": bool(
                    np.isclose(rho, current_rho)
                    and np.isclose(innovation_sd, current_innovation_sd)
                    and np.isclose(dispersion, current_dispersion)
                ),
            }
        )
    return candidates


def log1p_interval_score(
    observed: float | np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    *,
    alpha: float = INTERVAL_ALPHA,
) -> np.ndarray:
    """Central interval score after log1p transformation.

    This is a proper score for the reported central interval and jointly
    penalizes width and misses.  The log1p scale prevents high-incidence
    countries from dominating solely because their count scale is larger.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("Interval-score alpha must lie in (0, 1)")
    y = np.log1p(np.asarray(observed, dtype=float))
    low = np.log1p(np.asarray(lower, dtype=float))
    high = np.log1p(np.asarray(upper, dtype=float))
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("Interval-score inputs must be finite")
    if np.any(np.asarray(observed, dtype=float) < 0.0) or np.any(
        np.asarray(lower, dtype=float) < 0.0
    ):
        raise ValueError("Interval-score counts must be non-negative")
    if np.any(high < low):
        raise ValueError("Interval-score upper bounds must not be below lower bounds")
    return (
        high
        - low
        + (2.0 / alpha) * (low - y) * (y < low)
        + (2.0 / alpha) * (y - high) * (y > high)
    )


def add_selection_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add scale-balanced proper-score columns to successful fold rows."""

    out = frame.copy()
    if out.empty:
        return out
    required = {
        "observed_cases",
        "observation_predictive_q025",
        "observation_predictive_q975",
        "negative_binomial_nll",
        "process_mixture_negative_binomial_nll",
        "naive_negative_binomial_nll",
        "test_observation_intervals",
    }
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Hyperparameter fold rows are missing score columns: {missing}")
    intervals = pd.to_numeric(
        out["test_observation_intervals"], errors="raise"
    ).to_numpy(dtype=float)
    if np.any(intervals <= 0.0):
        raise ValueError("Every hindcast fold must contain a positive interval count")
    out["conditional_mean_negative_binomial_nll_per_interval"] = (
        pd.to_numeric(out["negative_binomial_nll"], errors="raise").to_numpy(dtype=float)
        / intervals
    )
    out["negative_binomial_nll_per_interval"] = (
        pd.to_numeric(
            out["process_mixture_negative_binomial_nll"], errors="raise"
        ).to_numpy(dtype=float)
        / intervals
    )
    out["naive_negative_binomial_nll_per_interval"] = (
        pd.to_numeric(
            out["naive_negative_binomial_nll"], errors="raise"
        ).to_numpy(dtype=float)
        / intervals
    )
    out["observation_predictive_95_log1p_interval_score"] = log1p_interval_score(
        pd.to_numeric(out["observed_cases"], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(
            out["observation_predictive_q025"], errors="raise"
        ).to_numpy(dtype=float),
        pd.to_numeric(
            out["observation_predictive_q975"], errors="raise"
        ).to_numpy(dtype=float),
    )
    return out


def _run_grid_task(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload["candidate"]
    country = str(payload["country"])
    try:
        frame = run_country_hindcast(
            country,
            minimum_training_years=int(payload["minimum_training_years"]),
            max_folds=payload["max_folds"],
            predictive_draws=int(payload["predictive_draws"]),
            dispersion=float(candidate["dispersion"]),
            process_overrides={
                "ar1_rho": float(candidate["rho"]),
                "log_beta_innovation_sd": float(candidate["innovation_sd"]),
            },
            state_coordinate_uncertainty_mode="map_conditioned",
        )
        frame = frame.assign(
            candidate_id=str(candidate["candidate_id"]),
            rho=float(candidate["rho"]),
            innovation_sd=float(candidate["innovation_sd"]),
            dispersion=float(candidate["dispersion"]),
            is_current_configuration=bool(candidate["is_current_configuration"]),
        )
        return {"fold_rows": frame.to_dict(orient="records"), "failure": None}
    except Exception as error:  # retain all other candidates and expose the failure
        return {
            "fold_rows": [],
            "failure": {
                "candidate_id": str(candidate["candidate_id"]),
                "country": country,
                "rho": float(candidate["rho"]),
                "innovation_sd": float(candidate["innovation_sd"]),
                "dispersion": float(candidate["dispersion"]),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            },
        }


def run_joint_grid(
    countries: Sequence[str],
    candidates: Sequence[dict[str, Any]],
    *,
    n_jobs: int | None,
    minimum_training_years: int,
    max_folds: int | None,
    predictive_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every country/candidate task, preserving partial failures."""

    tasks = [
        {
            "candidate": dict(candidate),
            "country": country,
            "minimum_training_years": int(minimum_training_years),
            "max_folds": max_folds,
            "predictive_draws": int(predictive_draws),
        }
        for candidate in candidates
        for country in countries
    ]
    outputs = parallel_map(
        _run_grid_task,
        tasks,
        desc="state_space_joint_hyperparameter_grid",
        n_jobs=n_jobs,
    )
    fold_records = [row for output in outputs for row in output["fold_rows"]]
    failure_records = [
        output["failure"] for output in outputs if output["failure"] is not None
    ]
    folds = add_selection_scores(pd.DataFrame(fold_records))
    failures = pd.DataFrame(failure_records, columns=FAILURE_COLUMNS)
    return folds, failures


def _country_balanced_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise")
    country_means = values.groupby(frame["country"].astype(str)).mean()
    return float(country_means.mean())


def candidate_subset_metrics(
    frame: pd.DataFrame,
    expected_fold_counts: dict[str, int],
) -> dict[str, Any]:
    """Summarize one candidate on a named country subset without pooling scale."""

    countries = list(expected_fold_counts)
    subset = (
        frame.loc[frame["country"].astype(str).isin(countries)].copy()
        if "country" in frame.columns
        else frame.iloc[0:0].copy()
    )
    failures = hindcast_execution_failures(subset, expected_fold_counts)
    basic = _aggregate_hindcast_metrics(subset)
    expected_total = int(sum(expected_fold_counts.values()))
    metrics: dict[str, Any] = {
        "country_count": int(len(countries)),
        "expected_fold_count": expected_total,
        "completed_fold_count": int(len(subset)),
        "execution_complete": bool(not failures and expected_total > 0),
        "execution_failures": "; ".join(failures),
        **basic,
    }
    balanced_columns = (
        "negative_binomial_nll_per_interval",
        "naive_negative_binomial_nll_per_interval",
        "absolute_log1p_error",
        "naive_absolute_log1p_error",
        "observation_predictive_95_log1p_interval_score",
    )
    for column in balanced_columns:
        name = f"country_balanced_mean_{column}"
        metrics[name] = (
            _country_balanced_mean(subset, column)
            if not subset.empty and column in subset.columns
            else np.nan
        )
    metrics["country_balanced_nb_nll_improvement_over_naive"] = float(
        metrics["country_balanced_mean_naive_negative_binomial_nll_per_interval"]
        - metrics["country_balanced_mean_negative_binomial_nll_per_interval"]
    )
    metrics["country_balanced_log1p_error_improvement_over_naive"] = float(
        metrics["country_balanced_mean_naive_absolute_log1p_error"]
        - metrics["country_balanced_mean_absolute_log1p_error"]
    )
    return metrics


def _selection_eligibility_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(metrics["execution_complete"]):
        reasons.append("execution_incomplete")
    required_finite = (
        "country_balanced_mean_negative_binomial_nll_per_interval",
        "country_balanced_mean_absolute_log1p_error",
        "country_balanced_mean_observation_predictive_95_log1p_interval_score",
        "observation_predictive_95_coverage",
        "median_observation_predictive_95_log1p_width",
    )
    if any(not np.isfinite(float(metrics[name])) for name in required_finite):
        reasons.append("non_finite_selection_metric")
        return reasons
    if (
        float(metrics["observation_predictive_95_coverage"])
        < MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE
    ):
        reasons.append("observation_predictive_undercoverage")
    if (
        float(metrics["median_observation_predictive_95_log1p_width"])
        > MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH
    ):
        reasons.append("observation_predictive_interval_non_discriminating")
    return reasons


def rank_candidates(
    folds: pd.DataFrame,
    candidates: Sequence[dict[str, Any]],
    expected_fold_counts: dict[str, int],
) -> pd.DataFrame:
    """Rank candidates on a tuning-country subset using a fixed rule.

    Eligibility (execution, >=50% empirical coverage, <=100-fold log1p width)
    is checked before ranking.  Eligible candidates are ordered by the Monte
    Carlo future-process-mixture NB log score, then the central-interval score,
    point error, interval width, and a deterministic ID.  Every mean is first
    computed within country so countries have equal weight despite different
    count scales and fold counts.
    """

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_folds = (
            folds.loc[folds["candidate_id"].astype(str).eq(candidate_id)].copy()
            if "candidate_id" in folds.columns
            else folds.iloc[0:0].copy()
        )
        metrics = candidate_subset_metrics(candidate_folds, expected_fold_counts)
        reasons = _selection_eligibility_reasons(metrics)
        rows.append(
            {
                **candidate,
                **metrics,
                "selection_eligible": bool(not reasons),
                "selection_ineligibility_reasons": "; ".join(reasons),
                "selection_rule": (
                    "eligibility_then_country_balanced_process_mixture_NB_NLL_then_log1p_95pct_"
                    "interval_score_then_absolute_log1p_error_then_width"
                ),
                "selection_score_type": (
                    "Monte_Carlo_future_AR1_process_mixture_NB2_log_score"
                ),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(
        by=[
            "selection_eligible",
            "country_balanced_mean_negative_binomial_nll_per_interval",
            "country_balanced_mean_observation_predictive_95_log1p_interval_score",
            "country_balanced_mean_absolute_log1p_error",
            "median_observation_predictive_95_log1p_width",
            "candidate_id",
        ],
        ascending=[False, True, True, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["selection_rank"] = np.arange(1, len(ranking) + 1, dtype=int)
    ranking["selected"] = False
    eligible_positions = ranking.index[ranking["selection_eligible"].astype(bool)]
    if len(eligible_positions):
        ranking.loc[eligible_positions[0], "selected"] = True
    return ranking


def build_candidate_country_summary(
    folds: pd.DataFrame,
    candidates: Sequence[dict[str, Any]],
    expected_fold_counts: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_folds = (
            folds.loc[folds["candidate_id"].astype(str).eq(candidate_id)].copy()
            if "candidate_id" in folds.columns
            else folds.iloc[0:0].copy()
        )
        for country, expected in expected_fold_counts.items():
            metrics = candidate_subset_metrics(candidate_folds, {country: expected})
            reasons = _selection_eligibility_reasons(metrics)
            rows.append(
                {
                    **candidate,
                    "country": country,
                    **metrics,
                    "coverage_sharpness_eligible": bool(not reasons),
                    "eligibility_reasons": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def _evaluation_reasons(
    metrics: dict[str, Any],
    *,
    expected_fold_count: int,
) -> list[str]:
    reasons = _selection_eligibility_reasons(metrics)
    if expected_fold_count < MINIMUM_EVALUATION_FOLDS_PER_COUNTRY:
        reasons.append("insufficient_heldout_folds")
    if np.isfinite(float(metrics["country_balanced_nb_nll_improvement_over_naive"])) and (
        float(metrics["country_balanced_nb_nll_improvement_over_naive"]) < 0.0
    ):
        reasons.append("negative_binomial_score_not_better_than_naive")
    if np.isfinite(
        float(metrics["country_balanced_log1p_error_improvement_over_naive"])
    ) and float(metrics["country_balanced_log1p_error_improvement_over_naive"]) < 0.0:
        reasons.append("absolute_log1p_error_not_better_than_naive")
    return list(dict.fromkeys(reasons))


def _current_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [candidate for candidate in candidates if candidate["is_current_configuration"]]
    if len(matches) > 1:
        raise RuntimeError("Joint grid contains duplicate current-configuration candidates")
    return matches[0] if matches else None


def leave_one_country_out_selection(
    folds: pd.DataFrame,
    candidates: Sequence[dict[str, Any]],
    expected_fold_counts: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune outside each country and evaluate only after candidate selection."""

    evaluation_rows: list[dict[str, Any]] = []
    selected_fold_frames: list[pd.DataFrame] = []
    all_tuning_rankings: list[pd.DataFrame] = []
    baseline = _current_candidate(candidates)
    for heldout_country, heldout_expected in expected_fold_counts.items():
        tuning_counts = {
            country: count
            for country, count in expected_fold_counts.items()
            if country != heldout_country
        }
        tuning_ranking = rank_candidates(folds, candidates, tuning_counts)
        tuning_ranking = tuning_ranking.assign(heldout_country=heldout_country)
        all_tuning_rankings.append(tuning_ranking)
        selected = tuning_ranking.loc[tuning_ranking["selected"].astype(bool)]
        if selected.empty:
            evaluation_rows.append(
                {
                    "heldout_country": heldout_country,
                    "tuning_country_count": len(tuning_counts),
                    "heldout_expected_fold_count": int(heldout_expected),
                    "selected_candidate_id": "",
                    "selection_available": False,
                    "heldout_execution_complete": False,
                    "heldout_evaluation_pass": False,
                    "heldout_evaluation_reasons": "no_eligible_tuning_candidate",
                    "current_configuration_present": baseline is not None,
                }
            )
            continue
        chosen = selected.iloc[0]
        chosen_id = str(chosen["candidate_id"])
        chosen_folds = folds.loc[
            folds["candidate_id"].astype(str).eq(chosen_id)
            & folds["country"].astype(str).eq(str(heldout_country))
        ].copy()
        chosen_folds["loco_selected_for_country"] = heldout_country
        selected_fold_frames.append(chosen_folds)
        evaluation_metrics = candidate_subset_metrics(
            chosen_folds, {heldout_country: heldout_expected}
        )
        reasons = _evaluation_reasons(
            evaluation_metrics, expected_fold_count=int(heldout_expected)
        )
        row: dict[str, Any] = {
            "heldout_country": heldout_country,
            "tuning_country_count": len(tuning_counts),
            "heldout_expected_fold_count": int(heldout_expected),
            "selected_candidate_id": chosen_id,
            "selected_rho": float(chosen["rho"]),
            "selected_innovation_sd": float(chosen["innovation_sd"]),
            "selected_dispersion": float(chosen["dispersion"]),
            "selection_available": True,
            "tuning_selection_rank": int(chosen["selection_rank"]),
            "tuning_country_balanced_mean_nb_nll_per_interval": float(
                chosen["country_balanced_mean_negative_binomial_nll_per_interval"]
            ),
            "tuning_country_balanced_mean_interval_score": float(
                chosen[
                    "country_balanced_mean_observation_predictive_95_log1p_interval_score"
                ]
            ),
            "heldout_execution_complete": bool(evaluation_metrics["execution_complete"]),
            "heldout_mean_nb_nll_per_interval": float(
                evaluation_metrics[
                    "country_balanced_mean_negative_binomial_nll_per_interval"
                ]
            ),
            "heldout_naive_mean_nb_nll_per_interval": float(
                evaluation_metrics[
                    "country_balanced_mean_naive_negative_binomial_nll_per_interval"
                ]
            ),
            "heldout_nb_nll_improvement_over_naive": float(
                evaluation_metrics["country_balanced_nb_nll_improvement_over_naive"]
            ),
            "heldout_mean_absolute_log1p_error": float(
                evaluation_metrics["country_balanced_mean_absolute_log1p_error"]
            ),
            "heldout_naive_mean_absolute_log1p_error": float(
                evaluation_metrics[
                    "country_balanced_mean_naive_absolute_log1p_error"
                ]
            ),
            "heldout_log1p_error_improvement_over_naive": float(
                evaluation_metrics[
                    "country_balanced_log1p_error_improvement_over_naive"
                ]
            ),
            "heldout_observation_predictive_95_coverage": float(
                evaluation_metrics["observation_predictive_95_coverage"]
            ),
            "heldout_median_predictive_log1p_width": float(
                evaluation_metrics["median_observation_predictive_95_log1p_width"]
            ),
            "heldout_mean_log1p_interval_score": float(
                evaluation_metrics[
                    "country_balanced_mean_observation_predictive_95_log1p_interval_score"
                ]
            ),
            "heldout_evaluation_pass": bool(not reasons),
            "heldout_evaluation_reasons": "; ".join(reasons),
            "current_configuration_present": baseline is not None,
        }
        if baseline is not None:
            baseline_folds = folds.loc[
                folds["candidate_id"].astype(str).eq(str(baseline["candidate_id"]))
                & folds["country"].astype(str).eq(str(heldout_country))
            ].copy()
            baseline_metrics = candidate_subset_metrics(
                baseline_folds, {heldout_country: heldout_expected}
            )
            row.update(
                {
                    "current_candidate_id": str(baseline["candidate_id"]),
                    "heldout_current_mean_nb_nll_per_interval": float(
                        baseline_metrics[
                            "country_balanced_mean_negative_binomial_nll_per_interval"
                        ]
                    ),
                    "heldout_current_mean_absolute_log1p_error": float(
                        baseline_metrics["country_balanced_mean_absolute_log1p_error"]
                    ),
                    "heldout_nb_nll_improvement_over_current": float(
                        baseline_metrics[
                            "country_balanced_mean_negative_binomial_nll_per_interval"
                        ]
                        - evaluation_metrics[
                            "country_balanced_mean_negative_binomial_nll_per_interval"
                        ]
                    ),
                    "heldout_log1p_error_improvement_over_current": float(
                        baseline_metrics["country_balanced_mean_absolute_log1p_error"]
                        - evaluation_metrics["country_balanced_mean_absolute_log1p_error"]
                    ),
                    "heldout_current_execution_complete": bool(
                        baseline_metrics["execution_complete"]
                    ),
                }
            )
        evaluation_rows.append(row)

    evaluation = pd.DataFrame(evaluation_rows)
    selected_folds = (
        pd.concat(selected_fold_frames, ignore_index=True)
        if selected_fold_frames
        else folds.iloc[0:0].copy()
    )
    tuning_rankings = (
        pd.concat(all_tuning_rankings, ignore_index=True)
        if all_tuning_rankings
        else pd.DataFrame()
    )
    return evaluation, selected_folds, tuning_rankings


def production_selection_decision(
    folds: pd.DataFrame,
    candidates: Sequence[dict[str, Any]],
    expected_fold_counts: dict[str, int],
    all_country_ranking: pd.DataFrame,
    loco_evaluation: pd.DataFrame,
    selected_folds: pd.DataFrame,
    *,
    task_failure_count: int = 0,
    current_configuration: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """Apply conservative rules for a recommendation; never mutate config."""

    current_rho, current_innovation_sd, current_dispersion = (
        configured_current_hyperparameters()
        if current_configuration is None
        else tuple(float(value) for value in current_configuration)
    )
    reasons: list[str] = []
    selected = all_country_ranking.loc[
        all_country_ranking["selected"].astype(bool)
    ] if not all_country_ranking.empty else pd.DataFrame()
    baseline = _current_candidate(candidates)
    if selected.empty:
        reasons.append("no_eligible_all_country_candidate")
        recommended: pd.Series | None = None
    else:
        recommended = selected.iloc[0]
    if baseline is None:
        reasons.append("current_configuration_missing_from_grid")
    expected_countries = [
        country for country, count in expected_fold_counts.items() if int(count) > 0
    ]
    if len(expected_countries) < MINIMUM_INDEPENDENT_COUNTRIES:
        reasons.append("insufficient_independent_countries")
    if any(
        int(expected_fold_counts[country]) < MINIMUM_EVALUATION_FOLDS_PER_COUNTRY
        for country in expected_countries
    ):
        reasons.append("insufficient_evaluation_folds_per_country")
    if task_failure_count:
        reasons.append("one_or_more_grid_tasks_failed")

    evaluation_available = (
        not loco_evaluation.empty
        and len(loco_evaluation) == len(expected_fold_counts)
        and bool(loco_evaluation["selection_available"].fillna(False).all())
        and bool(loco_evaluation["heldout_execution_complete"].fillna(False).all())
    )
    if not evaluation_available:
        reasons.append("loco_evaluation_incomplete")

    aggregate_metrics = candidate_subset_metrics(selected_folds, expected_fold_counts)
    pass_fraction = (
        float(loco_evaluation["heldout_evaluation_pass"].fillna(False).mean())
        if not loco_evaluation.empty
        else 0.0
    )
    if pass_fraction < MINIMUM_HELDOUT_COUNTRY_PASS_FRACTION:
        reasons.append("heldout_country_pass_fraction_below_threshold")
    if np.isfinite(
        float(aggregate_metrics["country_balanced_nb_nll_improvement_over_naive"])
    ) and float(aggregate_metrics["country_balanced_nb_nll_improvement_over_naive"]) < 0.0:
        reasons.append("loco_negative_binomial_score_not_better_than_naive")
    if np.isfinite(
        float(aggregate_metrics["country_balanced_log1p_error_improvement_over_naive"])
    ) and float(aggregate_metrics["country_balanced_log1p_error_improvement_over_naive"]) < 0.0:
        reasons.append("loco_absolute_log1p_error_not_better_than_naive")
    if np.isfinite(float(aggregate_metrics["observation_predictive_95_coverage"])) and (
        float(aggregate_metrics["observation_predictive_95_coverage"])
        < MINIMUM_OBSERVATION_PREDICTIVE_COVERAGE
    ):
        reasons.append("loco_observation_predictive_undercoverage")
    if np.isfinite(
        float(aggregate_metrics["median_observation_predictive_95_log1p_width"])
    ) and (
        float(aggregate_metrics["median_observation_predictive_95_log1p_width"])
        > MAXIMUM_OBSERVATION_PREDICTIVE_LOG1P_WIDTH
    ):
        reasons.append("loco_observation_predictive_interval_non_discriminating")

    provisional_id = str(recommended["candidate_id"]) if recommended is not None else ""
    stability_fraction = (
        float(
            loco_evaluation["selected_candidate_id"]
            .astype(str)
            .eq(provisional_id)
            .mean()
        )
        if provisional_id and not loco_evaluation.empty
        else 0.0
    )
    if stability_fraction < MINIMUM_SELECTION_STABILITY_FRACTION:
        reasons.append("all_country_candidate_not_stable_across_loco_tuning_sets")

    mean_nb_improvement_current = np.nan
    mean_error_improvement_current = np.nan
    if baseline is not None and not loco_evaluation.empty:
        for column in (
            "heldout_nb_nll_improvement_over_current",
            "heldout_log1p_error_improvement_over_current",
        ):
            if column not in loco_evaluation.columns:
                reasons.append("loco_current_configuration_comparison_incomplete")
                break
        else:
            mean_nb_improvement_current = float(
                pd.to_numeric(
                    loco_evaluation["heldout_nb_nll_improvement_over_current"],
                    errors="coerce",
                ).mean()
            )
            mean_error_improvement_current = float(
                pd.to_numeric(
                    loco_evaluation["heldout_log1p_error_improvement_over_current"],
                    errors="coerce",
                ).mean()
            )
            if not np.isfinite(mean_nb_improvement_current) or mean_nb_improvement_current < 0.0:
                reasons.append("loco_nb_score_not_better_than_current_configuration")
            if (
                not np.isfinite(mean_error_improvement_current)
                or mean_error_improvement_current < 0.0
            ):
                reasons.append("loco_point_error_not_better_than_current_configuration")

    reasons = list(dict.fromkeys(reasons))
    validation_supports_selection = bool(not reasons)
    recommended_is_current = bool(
        recommended is not None and bool(recommended["is_current_configuration"])
    )
    replacement_supported = bool(validation_supports_selection and not recommended_is_current)
    if replacement_supported:
        production_action = "recommend_config_change_for_manual_review"
    elif validation_supports_selection and recommended_is_current:
        production_action = "retain_current_configuration"
    else:
        production_action = "do_not_change_configuration"
    recommendation_id = provisional_id if validation_supports_selection else ""

    return pd.DataFrame(
        [
            {
                "production_action": production_action,
                "recommended_candidate_id": recommendation_id,
                "recommended_rho": (
                    float(recommended["rho"])
                    if recommended is not None and validation_supports_selection
                    else np.nan
                ),
                "recommended_innovation_sd": (
                    float(recommended["innovation_sd"])
                    if recommended is not None and validation_supports_selection
                    else np.nan
                ),
                "recommended_dispersion": (
                    float(recommended["dispersion"])
                    if recommended is not None and validation_supports_selection
                    else np.nan
                ),
                "recommended_is_current_configuration": bool(
                    recommended_is_current and validation_supports_selection
                ),
                "provisional_all_country_best_candidate_id": provisional_id,
                "provisional_all_country_best_rho": (
                    float(recommended["rho"]) if recommended is not None else np.nan
                ),
                "provisional_all_country_best_innovation_sd": (
                    float(recommended["innovation_sd"])
                    if recommended is not None
                    else np.nan
                ),
                "provisional_all_country_best_dispersion": (
                    float(recommended["dispersion"])
                    if recommended is not None
                    else np.nan
                ),
                "recommendation_status": (
                    "independently_supported_fixed_grid_selection"
                    if validation_supports_selection
                    else "provisional_only_validation_insufficient_or_failed"
                ),
                "loco_validation_supports_selection": validation_supports_selection,
                "loco_validation_supports_replacement": replacement_supported,
                "decision_reasons": "; ".join(reasons),
                "independent_country_count": int(len(expected_countries)),
                "minimum_independent_countries_required": MINIMUM_INDEPENDENT_COUNTRIES,
                "minimum_evaluation_folds_per_country": MINIMUM_EVALUATION_FOLDS_PER_COUNTRY,
                "heldout_country_pass_fraction": pass_fraction,
                "minimum_heldout_country_pass_fraction": MINIMUM_HELDOUT_COUNTRY_PASS_FRACTION,
                "selection_stability_fraction": stability_fraction,
                "minimum_selection_stability_fraction": MINIMUM_SELECTION_STABILITY_FRACTION,
                "loco_country_balanced_nb_nll_improvement_over_naive": float(
                    aggregate_metrics["country_balanced_nb_nll_improvement_over_naive"]
                ),
                "loco_country_balanced_log1p_error_improvement_over_naive": float(
                    aggregate_metrics[
                        "country_balanced_log1p_error_improvement_over_naive"
                    ]
                ),
                "loco_observation_predictive_95_coverage": float(
                    aggregate_metrics["observation_predictive_95_coverage"]
                ),
                "loco_median_predictive_log1p_width": float(
                    aggregate_metrics[
                        "median_observation_predictive_95_log1p_width"
                    ]
                ),
                "loco_mean_nb_nll_improvement_over_current": mean_nb_improvement_current,
                "loco_mean_log1p_error_improvement_over_current": mean_error_improvement_current,
                "current_rho": current_rho,
                "current_innovation_sd": current_innovation_sd,
                "current_dispersion": current_dispersion,
                "grid_candidate_count": int(len(candidates)),
                "expected_map_fit_count": int(
                    len(candidates) * sum(expected_fold_counts.values())
                ),
                "completed_map_fit_count": int(len(folds)),
                "failed_country_candidate_task_count": int(task_failure_count),
                "configuration_was_modified": False,
                "validation_design": (
                    "leave_one_country_out_candidate_selection_with_rolling_origin_folds"
                ),
                "predictive_uncertainty_scope": (
                    "conditional_on_MAP_state_with_future_AR1_process_and_NB2_measurement"
                ),
                "posterior_coverage_claimed": False,
                "primary_selection_score": (
                    "Monte_Carlo_future_AR1_process_mixture_NB2_log_score"
                ),
                "interpretation": (
                    "fixed-hyperparameter grid validation; overlapping expanding-window "
                    "folds are country-balanced and are not treated as independent replicates; "
                    "conditional predictive intervals are hyperparameter-comparison diagnostics, "
                    "not posterior credible intervals"
                ),
            }
        ]
    )


def _validate_countries(countries: Iterable[str], configs: dict[str, Any]) -> list[str]:
    requested = list(dict.fromkeys(str(country) for country in countries))
    known = set(configs.get("countries", {}))
    unknown = sorted(set(requested).difference(known))
    if unknown:
        raise ValueError(f"Unknown country profile(s): {unknown}")
    if not requested:
        raise ValueError("At least one country is required")
    return requested


def validate_publication_countries(
    countries: Iterable[str],
    configs: dict[str, Any],
) -> list[str]:
    requested = _validate_countries(countries, configs)
    publication = publication_country_names(configs)
    outside = sorted(set(requested).difference(publication))
    if outside:
        raise ValueError(
            "Hyperparameter selection excludes countries outside the prespecified "
            "publication set: " + ", ".join(outside)
        )
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select fixed annual state-space hyperparameters with joint-grid, "
            "rolling-origin, leave-one-country-out validation"
        )
    )
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--rho-values", nargs="+", type=float, default=DEFAULT_RHO_VALUES)
    parser.add_argument(
        "--innovation-sd-values",
        nargs="+",
        type=float,
        default=DEFAULT_INNOVATION_SD_VALUES,
    )
    parser.add_argument(
        "--dispersion-values",
        nargs="+",
        type=float,
        default=DEFAULT_DISPERSION_VALUES,
    )
    parser.add_argument("--minimum-training-years", type=int, default=4)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=2,
        help="Use the most recent N folds per country; default 2 controls cost.",
    )
    parser.add_argument("--predictive-draws", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--output-stem",
        default="state_space_hyperparameter_selection",
    )
    parser.add_argument(
        "--require-replacement-support",
        action="store_true",
        help="Exit non-zero after writing outputs unless independent checks support a change.",
    )
    args = parser.parse_args()
    if args.minimum_training_years < 2:
        raise ValueError("minimum-training-years must be at least two")
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("max-folds must be at least one")
    if args.predictive_draws < 8:
        raise ValueError("predictive-draws must be at least eight")

    configs = load_configs()
    publication_countries = publication_country_names(configs)
    countries = validate_publication_countries(
        args.countries or publication_countries,
        configs,
    )
    current_configuration = configured_current_hyperparameters(configs)
    candidates = build_hyperparameter_grid(
        args.rho_values,
        args.innovation_sd_values,
        args.dispersion_values,
        current_configuration=current_configuration,
    )
    expected_counts = expected_hindcast_fold_counts(
        countries,
        minimum_training_years=args.minimum_training_years,
        max_folds=args.max_folds,
    )
    folds, failures = run_joint_grid(
        countries,
        candidates,
        n_jobs=args.n_jobs,
        minimum_training_years=args.minimum_training_years,
        max_folds=args.max_folds,
        predictive_draws=args.predictive_draws,
    )
    country_summary = build_candidate_country_summary(
        folds, candidates, expected_counts
    )
    all_country_ranking = rank_candidates(folds, candidates, expected_counts)
    loco_evaluation, selected_folds, loco_rankings = leave_one_country_out_selection(
        folds, candidates, expected_counts
    )
    decision = production_selection_decision(
        folds,
        candidates,
        expected_counts,
        all_country_ranking,
        loco_evaluation,
        selected_folds,
        task_failure_count=len(failures),
        current_configuration=current_configuration,
    )

    output_dir = project_path("outputs", "tables")
    outputs = {
        "fold_scores": folds,
        "task_failures": failures,
        "candidate_country_summary": country_summary,
        "all_country_ranking": all_country_ranking,
        "loco_tuning_rankings": loco_rankings,
        "loco_evaluation": loco_evaluation,
        "loco_selected_fold_scores": selected_folds,
        "selection_decision": decision,
    }
    for suffix, frame in outputs.items():
        write_dataframe(frame, output_dir / f"{args.output_stem}_{suffix}.csv")

    if args.require_replacement_support and not bool(
        decision.iloc[0]["loco_validation_supports_replacement"]
    ):
        raise RuntimeError(
            "State-space hyperparameter replacement is not independently supported: "
            + str(decision.iloc[0]["decision_reasons"])
        )


if __name__ == "__main__":
    main()
