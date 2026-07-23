from __future__ import annotations

"""Fail-closed publication audit for the Figure 2c bootstrap confidence interval."""

from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    current_run_metadata,
    load_configs,
    publication_country_names,
    validate_run_metadata,
    write_run_metadata,
)
from src_python.simulation.run_figure2c_parametric_bootstrap import (
    BOOTSTRAP_DATA_GENERATION,
    BOOTSTRAP_REFIT,
    CONFIDENCE_INTERVAL_METHOD,
    DEFAULT_MAX_TAIL_PROBABILITY_MCSE,
    DRAW_PATH,
    FIT_DIAGNOSTIC_PATH,
    INTERVAL_BASIS,
    INTERVAL_PATH,
    INTERVAL_TYPE,
    INTERVENTION_STRATEGIES,
    PRIMARY_REDUCTION,
    STATISTICAL_TARGET,
    STABILITY_PATH,
    STEM,
    VARIED_ESTIMATION_COMPONENTS,
    FIXED_REFERENCE_INPUTS,
)
from src_python.utils.io import project_path, write_dataframe


AUDIT_STEM = "figure2c_parametric_bootstrap_quality_audit"
OUTPUT_PATH = project_path("outputs", "tables", f"{AUDIT_STEM}.csv")
AUDITED_ARTIFACT_PATHS = (DRAW_PATH, INTERVAL_PATH, FIT_DIAGNOSTIC_PATH, STABILITY_PATH)


def _add(
    rows: list[dict[str, Any]],
    *,
    category: str,
    check: str,
    passed: bool,
    details: str,
    severity: str = "fatal",
) -> None:
    rows.append(
        {
            "category": category,
            "check": check,
            "status": "pass" if bool(passed) else "fail",
            "severity": severity,
            "details": details,
        }
    )


def _finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    if any(column not in frame.columns for column in columns):
        return False
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def _exact_integer_values(values: pd.Series) -> tuple[pd.Series, bool]:
    """Parse identifiers without accepting fractions, NaN, or infinities."""

    numeric = pd.to_numeric(values, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    integral = finite & np.isclose(
        numeric.to_numpy(dtype=float),
        np.rint(numeric.to_numpy(dtype=float)),
        rtol=0.0,
        atol=1e-10,
    )
    if not bool(integral.all()):
        return pd.Series(index=values.index, dtype="int64"), False
    return pd.Series(
        np.rint(numeric.to_numpy(dtype=float)).astype(np.int64),
        index=values.index,
    ), True


def _audit_frames(
    *,
    source_metadata: dict[str, Any],
    draws: pd.DataFrame,
    intervals: pd.DataFrame,
    fits: pd.DataFrame,
    stability: pd.DataFrame,
    countries: list[str],
    requested_replicates: int,
    minimum_successful_replicates: int,
    minimum_success_fraction: float,
    maximum_tail_probability_mcse: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_strategies = set(INTERVENTION_STRATEGIES)
    expected_cells = len(countries) * len(expected_strategies)

    _add(
        rows,
        category="route",
        check="quality_bypass_disabled",
        passed="allow_quality_bypass" not in source_metadata
        and "skip_quality_gate" not in source_metadata,
        details="The canonical runner exposes no quality-bypass option.",
    )
    metadata_contract_ok = bool(
        source_metadata.get("analysis_role")
        == "publication_estimation_confidence_interval"
        and source_metadata.get("publication_path") is True
        and source_metadata.get("figure2c_interval_source") is True
        and source_metadata.get("statistical_target") == STATISTICAL_TARGET
        and source_metadata.get("confidence_interval_method")
        == CONFIDENCE_INTERVAL_METHOD
        and source_metadata.get("bootstrap_data_generation")
        == BOOTSTRAP_DATA_GENERATION
        and source_metadata.get("bootstrap_refit") == BOOTSTRAP_REFIT
        and tuple(source_metadata.get("varied_estimation_components", ()))
        == VARIED_ESTIMATION_COMPONENTS
        and tuple(source_metadata.get("fixed_reference_inputs", ()))
        == FIXED_REFERENCE_INPUTS
    )
    _add(
        rows,
        category="route",
        check="publication_estimation_ci_metadata_contract",
        passed=metadata_contract_ok,
        details=(
            f"analysis_role={source_metadata.get('analysis_role')}, "
            f"publication_path={source_metadata.get('publication_path')}, "
            "figure2c_interval_source="
            f"{source_metadata.get('figure2c_interval_source')}"
        ),
    )
    interval_types = set(intervals.get("interval_type", pd.Series(dtype=str)).astype(str))
    _add(
        rows,
        category="estimand",
        check="interval_type_is_parametric_bootstrap_confidence_interval",
        passed=interval_types == {INTERVAL_TYPE},
        details=f"observed={sorted(interval_types)}, expected={INTERVAL_TYPE!r}",
    )
    bases = intervals.get("interval_basis", pd.Series(dtype=str)).astype(str)
    normalized_bases = bases.str.lower()
    basis_ok = bool(
        len(bases)
        and bases.eq(INTERVAL_BASIS).all()
        and normalized_bases.str.contains("parametric bootstrap", regex=False).all()
        and normalized_bases.str.contains("fully refitted", regex=False).all()
        and normalized_bases.str.contains(r"ar\(1\)", regex=True).all()
        and normalized_bases.str.contains("nb2", regex=False).all()
    )
    _add(
        rows,
        category="estimand",
        check="interval_basis_matches_marginal_refit_bootstrap",
        passed=basis_ok,
        details=f"unique_basis_count={bases.nunique(dropna=False)}",
    )
    forbidden_claim = normalized_bases.str.contains(
        "posterior|credible|prediction interval|predictive interval", regex=True
    ).any() or any(
        token in " ".join(interval_types).lower()
        for token in ("posterior", "credible", "prediction", "predictive")
    )
    _add(
        rows,
        category="estimand",
        check="no_posterior_or_prediction_interval_claim",
        passed=not bool(forbidden_claim)
        and source_metadata.get("posterior_credible_interval") is False
        and source_metadata.get("future_observation_prediction_interval") is False,
        details=(
            "posterior_credible_interval="
            f"{source_metadata.get('posterior_credible_interval')}, "
            "future_observation_prediction_interval="
            f"{source_metadata.get('future_observation_prediction_interval')}"
        ),
    )

    interval_country = set(intervals.get("country", pd.Series(dtype=str)).astype(str))
    interval_strategy = set(intervals.get("strategy", pd.Series(dtype=str)).astype(str))
    coverage_ok = interval_country == set(countries) and interval_strategy == expected_strategies
    _add(
        rows,
        category="coverage",
        check="country_and_strategy_coverage",
        passed=coverage_ok,
        details=(
            f"countries={sorted(interval_country)}, strategies={sorted(interval_strategy)}"
        ),
    )
    interval_unique = not intervals.duplicated(["country", "strategy"]).any() if {
        "country",
        "strategy",
    }.issubset(intervals.columns) else False
    _add(
        rows,
        category="coverage",
        check="expected_country_strategy_rows",
        passed=len(intervals) == expected_cells and interval_unique,
        details=f"rows={len(intervals)}, expected={expected_cells}, unique={interval_unique}",
    )

    fit_columns = {"country", "bootstrap_replicate", "success", "attempt", "seed"}
    expected_fit_pairs = {
        (str(country), replicate)
        for country in countries
        for replicate in range(1, int(requested_replicates) + 1)
    }
    fit_work = pd.DataFrame()
    fit_ids_valid = False
    observed_fit_pairs: set[tuple[str, int]] = set()
    if fit_columns.issubset(fits.columns):
        fit_ids, fit_ids_valid = _exact_integer_values(fits["bootstrap_replicate"])
        if fit_ids_valid:
            fit_work = fits.assign(
                _country=fits["country"].astype(str),
                _replicate=fit_ids,
            )
            observed_fit_pairs = set(
                zip(fit_work["_country"], fit_work["_replicate"], strict=True)
            )
    fit_complete = bool(
        fit_ids_valid
        and len(fits) == len(expected_fit_pairs)
        and len(observed_fit_pairs) == len(fits)
        and observed_fit_pairs == expected_fit_pairs
    )
    _add(
        rows,
        category="refit",
        check="fit_diagnostics_complete",
        passed=fit_complete,
        details=(
            f"rows={len(fits)}, expected={len(countries) * int(requested_replicates)}, "
            f"exact_country_replicate_pairs={len(observed_fit_pairs)}/{len(expected_fit_pairs)}, "
            f"missing_columns={sorted(fit_columns.difference(fits.columns))}"
        ),
    )
    _add(
        rows,
        category="refit",
        check="fit_country_replicate_grid_exact",
        passed=fit_complete,
        details=(
            "Expected every publication country exactly once at every "
            f"bootstrap_replicate=1..{requested_replicates}; "
            f"observed_pairs={len(observed_fit_pairs)}, "
            f"expected_pairs={len(expected_fit_pairs)}"
        ),
    )
    if fit_complete:
        fit_success = fit_work.assign(
            success=fit_work["success"].astype(bool)
        ).groupby("_country")["success"].agg(["sum", "count"])
    else:
        fit_success = pd.DataFrame(columns=["sum", "count"])
    successes_complete = set(fit_success.index.astype(str)) == set(countries)
    min_success = bool(
        successes_complete
        and fit_success["sum"].ge(int(minimum_successful_replicates)).all()
    )
    success_fraction = bool(
        successes_complete
        and (fit_success["sum"] / int(requested_replicates))
        .ge(float(minimum_success_fraction))
        .all()
    )
    _add(
        rows,
        category="refit",
        check="minimum_successful_replicates",
        passed=min_success,
        details=(
            f"successes={fit_success['sum'].to_dict() if successes_complete else {}}, "
            f"minimum={minimum_successful_replicates}"
        ),
    )
    _add(
        rows,
        category="refit",
        check="bootstrap_refit_success_fraction",
        passed=success_fraction,
        details=(
            f"success_fraction="
            f"{(fit_success['sum'] / int(requested_replicates)).to_dict() if successes_complete else {}}, "
            f"minimum={minimum_success_fraction}"
        ),
    )

    quantile_columns = [
        "reduction_q025",
        "reduction_q25",
        "reduction_median",
        "reduction_q75",
        "reduction_q975",
    ]
    ordered = False
    positive_width = False
    if _finite(intervals, quantile_columns):
        q = intervals.loc[:, quantile_columns].to_numpy(dtype=float)
        ordered = bool((np.diff(q, axis=1) >= 0.0).all())
        positive_width = bool((q[:, -1] - q[:, 0] > 0.0).all())
    _add(
        rows,
        category="intervals",
        check="interval_quantiles_ordered",
        passed=ordered,
        details=f"finite={_finite(intervals, quantile_columns)}",
    )
    _add(
        rows,
        category="intervals",
        check="interval_width_positive",
        passed=positive_width,
        details=f"cells={len(intervals)}",
    )
    stability_finite = _finite(
        stability,
        ["tail_probability_mcse", "maximum_endpoint_mcse"],
    )
    observed_max_mcse = (
        float(pd.to_numeric(stability["tail_probability_mcse"]).max())
        if stability_finite and len(stability)
        else float("nan")
    )
    stability_complete = bool(
        len(stability) == expected_cells
        and {"country", "strategy"}.issubset(stability.columns)
        and not stability.duplicated(["country", "strategy"]).any()
    )
    _add(
        rows,
        category="intervals",
        check="tail_probability_mcse_below_threshold",
        passed=stability_complete
        and stability_finite
        and observed_max_mcse <= float(maximum_tail_probability_mcse),
        details=(
            f"observed_max={observed_max_mcse}, "
            f"threshold={maximum_tail_probability_mcse}; outcome-scale "
            "endpoint sensitivity is retained diagnostically but is not a "
            "scale-invariant quality gate"
        ),
    )

    paired_columns = {
        "country",
        "bootstrap_replicate",
        "strategy",
        "current_rate",
        "intervention_rate",
        PRIMARY_REDUCTION,
    }
    pairing_ok = False
    strategy_design_ok = False
    if paired_columns.issubset(draws.columns) and len(draws) and fit_complete:
        draw_ids, draw_ids_valid = _exact_integer_values(
            draws["bootstrap_replicate"]
        )
        draw_work = draws.assign(
            _country=draws["country"].astype(str),
            _replicate=draw_ids,
            _strategy=draws["strategy"].astype(str),
        )
        draw_numeric = draws.loc[
            :, ["current_rate", "intervention_rate", PRIMARY_REDUCTION]
        ].apply(pd.to_numeric, errors="coerce")
        recalculated = (
            draw_numeric["current_rate"] - draw_numeric["intervention_rate"]
        ) / draw_numeric["current_rate"]
        successful_fit_pairs = set(
            zip(
                fit_work.loc[fit_work["success"].astype(bool), "_country"],
                fit_work.loc[fit_work["success"].astype(bool), "_replicate"],
                strict=True,
            )
        )
        observed_draw_pairs = set(
            zip(draw_work["_country"], draw_work["_replicate"], strict=True)
        )
        strategy_blocks = draw_work.groupby(
            ["_country", "_replicate"], sort=False
        )["_strategy"].agg(lambda values: frozenset(values))
        strategy_design_ok = bool(
            draw_ids_valid
            and observed_draw_pairs == successful_fit_pairs
            and len(strategy_blocks) == len(successful_fit_pairs)
            and all(block == expected_strategies for block in strategy_blocks)
            and not draws.duplicated(
                ["country", "bootstrap_replicate", "strategy"]
            ).any()
            and len(draws)
            == len(successful_fit_pairs) * len(expected_strategies)
        )
        current_ranges = draw_work.assign(
            _current_rate=draw_numeric["current_rate"]
        ).groupby(["_country", "_replicate"])["_current_rate"].agg(
            lambda values: float(values.max() - values.min())
        )
        pairing_ok = bool(
            draw_ids_valid
            and np.isfinite(draw_numeric.to_numpy(dtype=float)).all()
            and draw_numeric["current_rate"].gt(0.0).all()
            and np.allclose(
                recalculated.to_numpy(dtype=float),
                draw_numeric[PRIMARY_REDUCTION].to_numpy(dtype=float),
                rtol=1e-8,
                atol=1e-10,
            )
            and not draws.duplicated(
                ["country", "bootstrap_replicate", "strategy"]
            ).any()
            and set(draws["strategy"].astype(str)) == expected_strategies
            and strategy_design_ok
            and current_ranges.le(1e-10).all()
        )
    _add(
        rows,
        category="pairing",
        check="successful_replicate_strategy_grid_exact",
        passed=strategy_design_ok,
        details=(
            "Every successful country-replicate must contain each of the "
            f"{len(expected_strategies)} programme strategies exactly once."
        ),
    )
    _add(
        rows,
        category="pairing",
        check="paired_current_intervention",
        passed=pairing_ok and source_metadata.get("paired_scenario_contrast") is True,
        details=(
            f"draw_rows={len(draws)}, paired_metadata="
            f"{source_metadata.get('paired_scenario_contrast')}"
        ),
    )

    return rows


def main(*, fail_on_warnings: bool = True) -> pd.DataFrame:
    warnings_are_fatal = True
    configs = load_configs()
    countries = publication_country_names(configs)
    settings = (
        configs["baseline"]
        .get("bayesian_uncertainty", {})
        .get("figure2c_parametric_bootstrap_confidence_interval", {})
    )
    source_metadata = validate_run_metadata(STEM)
    for path in AUDITED_ARTIFACT_PATHS:
        if not path.exists():
            raise FileNotFoundError(path)
    draws = pd.read_csv(DRAW_PATH)
    intervals = pd.read_csv(INTERVAL_PATH)
    fits = pd.read_csv(FIT_DIAGNOSTIC_PATH)
    stability = pd.read_csv(STABILITY_PATH)
    rows = _audit_frames(
        source_metadata=source_metadata,
        draws=draws,
        intervals=intervals,
        fits=fits,
        stability=stability,
        countries=countries,
        requested_replicates=int(settings["replicates_per_country"]),
        minimum_successful_replicates=int(settings["minimum_successful_replicates"]),
        minimum_success_fraction=float(settings["minimum_success_fraction"]),
        maximum_tail_probability_mcse=float(
            settings.get(
                "maximum_tail_probability_mcse",
                DEFAULT_MAX_TAIL_PROBABILITY_MCSE,
            )
        ),
    )
    audit = pd.DataFrame(rows)
    write_dataframe(audit, OUTPUT_PATH)
    failed = audit.loc[
        audit["severity"].isin(["fatal", "warning"])
        & ~audit["status"].eq("pass")
    ]
    metadata = current_run_metadata(AUDIT_STEM, row_counts={"audit_checks": len(audit)}) | {
        "analysis_role": "publication_estimation_confidence_interval_quality_audit",
        "publication_path": True,
        "figure2c_interval_source": False,
        "audits_figure2c_interval_source": True,
        "passed": failed.empty,
        "warnings_are_fatal": warnings_are_fatal,
        "figure2c_source_stem": STEM,
    }
    write_run_metadata(AUDIT_STEM, metadata)
    if not failed.empty:
        raise RuntimeError(
            "Figure 2 bootstrap confidence-interval audit failed: "
            + ", ".join(failed["check"].astype(str))
        )
    return audit


if __name__ == "__main__":
    main()
