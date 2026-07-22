"""Validate the active point-estimate/POMP publication output contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.model.outputs import GREGORIAN_YEAR_DAYS
from src_python.simulation.common import (
    file_sha256,
    load_configs,
    publication_country_names,
    validate_run_metadata,
)
from src_python.simulation.run_joint_psa_rank_acceptability import PROGRAMME_ONLY_STRATEGIES
from src_python.simulation.run_resistance_management_psa import (
    EXPECTED_PARAMETER_NAMES as RESISTANCE_MANAGEMENT_PARAMETER_NAMES,
    PEP_RESTORATION_STRATA,
    SAMPLE_DESIGN as RESISTANCE_MANAGEMENT_SAMPLE_DESIGN,
    UNCERTAINTY_SCHEMA_VERSION as RESISTANCE_MANAGEMENT_SCHEMA_VERSION,
)
from src_python.utils.io import project_path, read_table
from src_python.utils.validation import (
    CORE_OUTPUT_STEMS,
    PUBLICATION_METADATA_STEMS,
    PUBLICATION_REQUIRED_TABLES,
    validate_baseline_outputs,
    validate_population_conservation,
)
from manuscript_notes.validate_figure2_parent_metadata import (
    validate_figure2_parent_metadata,
)


# One publication inventory is shared by the generic and manuscript-specific
# validators so retired research outputs cannot drift back into the release gate.
ACTIVE_PUBLICATION_METADATA_STEMS = PUBLICATION_METADATA_STEMS

FIGURE2_PARENT_ARTIFACTS = {
    "reference": (
        "figure2_programme_reference",
        "outputs/summaries/figure2_programme_reference_summary.csv",
        "summary",
    ),
    "bootstrap": (
        "figure2c_parametric_bootstrap",
        "outputs/tables/figure2c_programme_paired_bootstrap_draws.csv",
        "paired_bootstrap_draws",
    ),
    "selected_input_rank": (
        "joint_psa_rank_acceptability",
        "outputs/tables/joint_psa_under18_programme_rank_samples.csv",
        "under18_programme_rank_samples",
    ),
    "production_intervention": (
        "intervention_scenarios",
        "outputs/summaries/intervention_scenarios_summary.csv",
        None,
    ),
}
FIGURE2_DERIVED_SOURCE_TABLES = (
    "outputs/tables/figure2a_delivery_lever_contrast.csv",
    "outputs/tables/figure2b_reference_choice_fragility.csv",
    "outputs/tables/figure2b_fixed_reference_regret_draws.csv",
    "outputs/tables/figure2c_programme_effect_matrix.csv",
    "outputs/tables/figure2_selected_input_rank1_counts.csv",
)
FIGURE2_PROVENANCE_PATH = "outputs/tables/figure2_parent_provenance.csv"
FIGURE2_PARENT_BUNDLE_COLUMN = "figure2_parent_bundle_sha256"

# The annualized-rate and accumulated-total contrasts are normalized from
# separate ODE outputs.  Their dimensionless relative reductions can therefore
# differ by about 3e-9 from solver and floating-point error in production.
ANNUALIZED_TOTAL_CONTRAST_ATOL = 5e-9
ANNUALIZED_TOTAL_CONTRAST_RTOL = 1e-12


def _lancet_decimal(value: float, digits: int = 1) -> str:
    """Format a number using the Lancet midline decimal and minus sign."""

    return f"{float(value):.{digits}f}".replace("-", "−").replace(".", "·")


def _small_number_word(value: int) -> str:
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    return words.get(int(value), str(int(value)))


def _table_interval_value(value: float) -> str:
    """Match Table 1 precision: retain two decimals only for sub-unit limits."""

    digits = 2 if abs(float(value)) < 1.0 else 1
    return _lancet_decimal(value, digits)


def _require_text_tokens(text: str, tokens: list[str], *, label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{label} is inconsistent with current outputs: {missing}")


def _validate_figure2b_interpretation_language(interpretation: pd.Series) -> None:
    """Require design-frequency wording and reject inferential Figure 2b labels."""

    labels = interpretation.fillna("").astype(str).str.strip()
    has_selected_input = labels.str.contains(
        r"\bselected[- ]input\b", case=False, regex=True
    )
    has_design_frequency = labels.str.contains(
        r"\bdesign[- ]frequenc(?:y|ies)\b", case=False, regex=True
    )
    has_selected_parameter = labels.str.contains(
        r"\bselected[- ]parameters?\b", case=False, regex=True
    )
    has_probability = labels.str.contains(
        r"\bprobabilit(?:y|ies)\b", case=False, regex=True
    )
    has_inferential_target = labels.str.contains(
        r"\b(?:posterior|credible|confidence|prediction|predictive)\b",
        case=False,
        regex=True,
    )
    invalid = (
        labels.eq("")
        | ~has_selected_input
        | ~has_design_frequency
        | has_selected_parameter
        | has_probability
        | has_inferential_target
    )
    if invalid.any():
        examples = labels.loc[invalid].drop_duplicates().tolist()[:3]
        raise AssertionError(
            "Figure 2b interpretation must use selected-input design-frequency "
            "language and must not use selected-parameter, probability, or "
            "inferential-target language; "
            f"invalid values: {examples}"
        )


def _figure2b_headline_tokens(
    decision_fragility: pd.DataFrame,
) -> dict[str, object]:
    """Derive Figure 2b Results/Summary tokens from the current source table."""

    required = {
        "country",
        "prespecified_setting_count",
        "reference_choice_retained_count",
        "regret_percentage_points_q95",
    }
    missing = required.difference(decision_fragility.columns)
    if missing:
        raise AssertionError(
            "Figure 2b headline source is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = decision_fragility.loc[:, sorted(required)].copy()
    frame["country"] = frame["country"].astype(str).str.strip()
    if (
        len(frame) < 3
        or frame["country"].eq("").any()
        or frame["country"].duplicated().any()
    ):
        raise AssertionError(
            "Figure 2b headline source requires at least three uniquely named profiles."
        )

    for column in (
        "prespecified_setting_count",
        "reference_choice_retained_count",
        "regret_percentage_points_q95",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise AssertionError(
                f"Figure 2b headline source contains non-finite values in {column}."
            )

    setting_counts = frame["prespecified_setting_count"].unique()
    if (
        len(setting_counts) != 1
        or setting_counts[0] <= 0
        or not float(setting_counts[0]).is_integer()
    ):
        raise AssertionError(
            "Figure 2b headline source must use one positive integer setting count."
        )
    setting_count = int(setting_counts[0])
    retained = frame["reference_choice_retained_count"]
    if (
        (retained < 0).any()
        or (retained > setting_count).any()
        or not np.equal(retained, np.floor(retained)).all()
        or frame["regret_percentage_points_q95"].lt(0.0).any()
    ):
        raise AssertionError(
            "Figure 2b headline retention counts or q95 regrets are invalid."
        )
    frame["reference_choice_retained_count"] = retained.astype(int)

    retention_order = frame.sort_values(
        ["reference_choice_retained_count", "country"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    regret_order = frame.sort_values(
        ["regret_percentage_points_q95", "country"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    lowest_retention = retention_order.iloc[0]
    second_largest_regret = regret_order.iloc[1]
    if not (
        float(regret_order.iloc[0]["regret_percentage_points_q95"])
        > float(second_largest_regret["regret_percentage_points_q95"])
        > float(lowest_retention["regret_percentage_points_q95"])
        and int(second_largest_regret["reference_choice_retained_count"])
        > int(lowest_retention["reference_choice_retained_count"])
        and str(second_largest_regret["country"])
        != str(lowest_retention["country"])
    ):
        raise AssertionError(
            "Figure 2b current source does not support the prespecified dynamic "
            "retention-versus-consequence contrast."
        )

    lowest_two = retention_order.iloc[:2]
    largest_three = regret_order.iloc[:3]
    retained_min = int(retention_order["reference_choice_retained_count"].min())
    retained_max = int(retention_order["reference_choice_retained_count"].max())

    retention_token = (
        "Retention was lowest in "
        + str(lowest_two.iloc[0]["country"])
        + " ("
        + str(int(lowest_two.iloc[0]["reference_choice_retained_count"]))
        + "/"
        + str(setting_count)
        + ") and "
        + str(lowest_two.iloc[1]["country"])
        + " ("
        + str(int(lowest_two.iloc[1]["reference_choice_retained_count"]))
        + "/"
        + str(setting_count)
        + ")"
    )

    def regret_token(*, figure_reference: bool) -> str:
        ending = "; figure 2B)" if figure_reference else ")"
        return (
            "The 95th-percentile fixed-reference regret was largest in "
            + str(largest_three.iloc[0]["country"])
            + " ("
            + _lancet_decimal(
                largest_three.iloc[0]["regret_percentage_points_q95"], 1
            )
            + " percentage points of current-practice burden), "
            + str(largest_three.iloc[1]["country"])
            + " ("
            + _lancet_decimal(
                largest_three.iloc[1]["regret_percentage_points_q95"], 1
            )
            + "), and "
            + str(largest_three.iloc[2]["country"])
            + " ("
            + _lancet_decimal(
                largest_three.iloc[2]["regret_percentage_points_q95"], 1
            )
            + ending
        )

    contrast_token = (
        "Retention frequency and high-tail consequence therefore supplied different "
        "information: "
        + str(second_largest_regret["country"])
        + " retained its reference choice in "
        + str(int(second_largest_regret["reference_choice_retained_count"]))
        + "/"
        + str(setting_count)
        + " settings but had the second-largest 95th-percentile regret, whereas "
        + str(lowest_retention["country"])
        + " had the lowest retention but a lower 95th-percentile regret of "
        + _lancet_decimal(lowest_retention["regret_percentage_points_q95"], 1)
    )

    return {
        "setting_count": setting_count,
        "retained_min": retained_min,
        "retained_max": retained_max,
        "results_tokens": [
            "Across the "
            + str(setting_count)
            + " prespecified selected-input settings, the locked reference choice "
            "remained lowest-burden in "
            + str(retained_min)
            + "–"
            + str(retained_max)
            + " settings per profile",
            retention_token,
            regret_token(figure_reference=True),
            contrast_token,
            "These are deterministic summaries of the prespecified design, not "
            "selection probabilities, confidence intervals, or expected regret",
        ],
        "summary_tokens": [
            "The locked reference choice was retained in "
            + str(retained_min)
            + "–"
            + str(retained_max)
            + " of "
            + str(setting_count)
            + " selected-input settings per profile",
            regret_token(figure_reference=False),
            "these are deterministic design summaries, not probabilities or expected regret",
        ],
    }


def _validate_figure2c_estimation_ci_language(
    interval_type: pd.Series, interval_method: pd.Series
) -> None:
    """Fail closed unless Figure 2c labels identify estimation confidence intervals."""

    labels = interval_type.fillna("").astype(str).str.strip()
    methods = interval_method.fillna("").astype(str).str.strip()
    has_paired = labels.str.contains(r"\bpaired\b", case=False, regex=True)
    has_full_refit = labels.str.contains(
        r"\bfull[- ]refit\b", case=False, regex=True
    )
    has_parametric_bootstrap = labels.str.contains(
        r"\bparametric[- ]bootstrap\b", case=False, regex=True
    )
    has_95_percent = labels.str.contains(r"\b95\s*%", case=False, regex=True)
    has_estimation_ci = labels.str.contains(
        r"\bestimation confidence intervals?\b", case=False, regex=True
    )
    has_forbidden_target = labels.str.contains(
        r"\b(?:posterior|credible|prediction|predictive|bayesian)\b",
        case=False,
        regex=True,
    )
    invalid_label = (
        labels.eq("")
        | ~has_paired
        | ~has_full_refit
        | ~has_parametric_bootstrap
        | ~has_95_percent
        | ~has_estimation_ci
        | has_forbidden_target
    )
    invalid_method = methods.ne("percentile_parametric_bootstrap")
    if len(labels) != len(methods) or invalid_label.any() or invalid_method.any():
        examples = labels.loc[invalid_label].drop_duplicates().tolist()[:3]
        invalid_methods = methods.loc[invalid_method].drop_duplicates().tolist()[:3]
        raise AssertionError(
            "Figure 2c intervals must be labelled as paired full-refit "
            "parametric-bootstrap 95% estimation confidence intervals, must use "
            "the percentile_parametric_bootstrap method, and must not use "
            "Bayesian, posterior, credible, prediction, or predictive language; "
            f"invalid labels: {examples}; invalid methods: {invalid_methods}"
        )


def _metadata_integer(metadata: dict, field: str, *, label: str) -> int:
    """Read an integer-valued metadata field without silently truncating it."""

    value = metadata.get(field)
    if isinstance(value, bool):
        raise AssertionError(f"{label} metadata have invalid {field}={value!r}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"{label} metadata are missing a valid integer {field}."
        ) from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise AssertionError(f"{label} metadata have invalid {field}={value!r}.")
    return int(numeric)


def _validate_figure2c_successful_refit_gate(
    programme_effects: pd.DataFrame,
    bootstrap_metadata: dict,
    *,
    expected_countries: set[str],
) -> tuple[int, int]:
    """Validate Figure 2c refit counts against its run-specific metadata gate."""

    requested = _metadata_integer(
        bootstrap_metadata,
        "replicates_requested_per_country",
        label="Figure 2c bootstrap",
    )
    minimum = _metadata_integer(
        bootstrap_metadata,
        "minimum_successful_replicates",
        label="Figure 2c bootstrap",
    )
    if requested < 1 or minimum < 1 or minimum > requested:
        raise AssertionError(
            "Figure 2c bootstrap metadata have an invalid requested/minimum "
            "successful-refit gate."
        )

    metadata_countries = tuple(
        str(country) for country in bootstrap_metadata.get("countries", ())
    )
    if (
        len(metadata_countries) != len(expected_countries)
        or len(set(metadata_countries)) != len(metadata_countries)
        or set(metadata_countries) != expected_countries
    ):
        raise AssertionError(
            "Figure 2c bootstrap metadata do not contain the exact publication "
            "countries."
        )
    metadata_success = bootstrap_metadata.get("successful_replicates_by_country")
    if not isinstance(metadata_success, dict) or set(metadata_success) != set(
        metadata_countries
    ):
        raise AssertionError(
            "Figure 2c bootstrap metadata have incomplete country successful-refit "
            "counts."
        )
    successful_by_country = {
        country: _metadata_integer(
            metadata_success,
            country,
            label="Figure 2c country successful-refit",
        )
        for country in metadata_countries
    }
    failed_gate = {
        country: count
        for country, count in successful_by_country.items()
        if count < minimum or count > requested
    }
    if failed_gate:
        raise AssertionError(
            "Figure 2c bootstrap metadata fail the country successful-refit gate: "
            f"{failed_gate}."
        )

    if "successful_bootstrap_replicates" not in programme_effects.columns:
        raise AssertionError(
            "Figure 2c source data are missing successful_bootstrap_replicates."
        )
    source_counts = pd.to_numeric(
        programme_effects["successful_bootstrap_replicates"], errors="raise"
    )
    if (
        not np.isfinite(source_counts).all()
        or not np.equal(source_counts, np.floor(source_counts)).all()
    ):
        raise AssertionError(
            "Figure 2c source data contain invalid successful-refit counts."
        )
    source = programme_effects.assign(
        _country=programme_effects["country"].astype(str),
        _successful_refits=source_counts.astype(int),
    )
    if set(source["_country"]) != expected_countries:
        raise AssertionError(
            "Figure 2c source data do not contain the exact publication countries."
        )
    source_unique_counts = source.groupby("_country")["_successful_refits"].nunique()
    if not source_unique_counts.eq(1).all():
        raise AssertionError(
            "Figure 2c successful-refit counts are inconsistent within a country."
        )
    source_by_country = (
        source.groupby("_country")["_successful_refits"].first().to_dict()
    )
    if source_by_country != successful_by_country:
        raise AssertionError(
            "Figure 2c source successful-refit counts do not match bootstrap metadata."
        )

    observed_counts = tuple(successful_by_country.values())
    return min(observed_counts), max(observed_counts)


def _integer_range_text(minimum: int, maximum: int) -> str:
    """Format an observed integer min-max range for manuscript prose."""

    return str(minimum) if minimum == maximum else f"{minimum}–{maximum}"


def _validate_figure2c_reported_refit_range(
    figure_legend_text: str,
    supplement_text: str,
    *,
    observed_minimum: int,
    observed_maximum: int,
) -> None:
    """Require Figure 2c prose to report the observed refit-count range."""

    expected_range = _integer_range_text(observed_minimum, observed_maximum)
    legend_ranges = re.findall(
        r"The intervals used ([0-9]+(?:–[0-9]+)?) successful refits per profile",
        figure_legend_text,
    )
    supplement_ranges = re.findall(
        r"2·5th and 97·5th percentiles of "
        r"([0-9]+(?:–[0-9]+)?) successful refits per profile",
        supplement_text,
    )
    if legend_ranges != [expected_range]:
        raise AssertionError(
            "Main-manuscript Figure 2 legend does not report the observed "
            f"successful-refit range {expected_range}."
        )
    if supplement_ranges != [expected_range]:
        raise AssertionError(
            "Supplementary Figure 2 methods do not report the observed "
            f"successful-refit range {expected_range}."
        )
    _require_text_tokens(
        figure_legend_text,
        [
            "brackets show paired full-refit parametric-bootstrap 95% "
            "estimation confidence intervals",
            "are estimation confidence intervals for fitted scenario contrasts, "
            "not posterior credible intervals or future-observation prediction "
            "intervals",
        ],
        label="Main-manuscript Figure 2 estimation-CI contract",
    )
    _require_text_tokens(
        supplement_text,
        [
            "The cell brackets are paired full-refit parametric-bootstrap 95% "
            "estimation confidence intervals",
            "These panel c intervals quantify uncertainty in the fitted scenario "
            "contrasts and are neither posterior credible intervals nor "
            "future-outbreak prediction intervals",
        ],
        label="Supplementary Figure 2 estimation-CI contract",
    )


def _markdown_section(text: str, start: str, end: str) -> str:
    start_marker = f"## {start}"
    end_marker = f"## {end}"
    if start_marker not in text or end_marker not in text:
        raise AssertionError(f"Cannot locate manuscript section {start!r} to {end!r}.")
    return text.split(start_marker, 1)[1].split(end_marker, 1)[0]


def validate_main_manuscript_key_numbers() -> None:
    """Tie every headline manuscript number to the active output tables."""

    main_path = project_path("manuscript", "submission_ready", "main_manuscript.md")
    supplement_path = project_path(
        "manuscript", "submission_ready", "supplementary_material.md"
    )
    conference_abstract_path = project_path(
        "manuscript",
        "submission_ready",
        "pediatrics_conference_abstract_english.md",
    )
    main_text = Path(main_path).read_text(encoding="utf-8")
    supplement_text = Path(supplement_path).read_text(encoding="utf-8")
    conference_abstract_text = Path(conference_abstract_path).read_text(
        encoding="utf-8"
    )
    summary_text = _markdown_section(main_text, "Summary", "Research in context")
    results_text = _markdown_section(main_text, "Results", "Discussion")
    table_text = _markdown_section(main_text, "Tables", "Figure legends")
    figure_legend_text = _markdown_section(main_text, "Figure legends", "References")

    regional_incidence = read_table(
        project_path("data", "processed", "who_pertussis_region_incidence.csv")
    )
    required_regional_columns = {
        "region",
        "year",
        "reported_incidence_per_million",
    }
    missing_regional_columns = required_regional_columns.difference(
        regional_incidence.columns
    )
    if missing_regional_columns:
        raise AssertionError(
            "Figure 1a regional-incidence input is missing columns: "
            + ", ".join(sorted(missing_regional_columns))
        )
    global_incidence = regional_incidence.loc[
        regional_incidence["region"].astype(str).eq("Global")
        & pd.to_numeric(regional_incidence["year"], errors="coerce").isin([2021, 2024])
    ].copy()
    if (
        len(global_incidence) != 2
        or set(pd.to_numeric(global_incidence["year"], errors="raise").astype(int))
        != {2021, 2024}
        or global_incidence["year"].duplicated().any()
    ):
        raise AssertionError(
            "Figure 1a does not contain one Global incidence row for both 2021 and 2024."
        )
    global_incidence_per_100k = (
        global_incidence.assign(
            year=pd.to_numeric(global_incidence["year"], errors="raise").astype(int),
            incidence_per_100k=pd.to_numeric(
                global_incidence["reported_incidence_per_million"], errors="raise"
            )
            / 10.0,
        )
        .set_index("year")["incidence_per_100k"]
    )
    if not np.isfinite(global_incidence_per_100k).all():
        raise AssertionError("Figure 1a Global incidence contains non-finite values.")
    _require_text_tokens(
        results_text,
        [
            "Reported pertussis incidence rebounded after the low-circulation pandemic "
            "years, rising globally from "
            + _lancet_decimal(global_incidence_per_100k.loc[2021], 2)
            + " per 100 000 population in 2021 to "
            + _lancet_decimal(global_incidence_per_100k.loc[2024], 1)
            + " per 100 000 in 2024"
        ],
        label="Main-manuscript Figure 1a results",
    )

    prediction_intervals = read_table(
        project_path("outputs", "tables", "panel_pomp_rolling_hindcast_intervals.csv")
    )
    prediction_folds = read_table(
        project_path("outputs", "tables", "panel_pomp_rolling_hindcast_folds.csv")
    )
    prediction_summary = read_table(
        project_path("outputs", "tables", "panel_pomp_rolling_hindcast_summary.csv")
    )
    prediction_gate = read_table(
        project_path("outputs", "tables", "panel_pomp_rolling_hindcast_gate.csv")
    ).iloc[0]
    overall_prediction = prediction_summary.loc[
        prediction_summary["scope"].astype(str).eq("overall")
    ].iloc[0]
    block_folds = read_table(
        project_path("outputs", "tables", "panel_pomp_block_stress_folds.csv")
    )
    block_gate = read_table(
        project_path("outputs", "tables", "panel_pomp_block_stress_gate.csv")
    ).iloc[0]
    block_year_coverage = block_folds.groupby("test_year")[
        "model_annual_95_covered"
    ].mean()
    if not np.isclose(
        float(block_year_coverage.loc[2024]),
        float(block_year_coverage.loc[2025]),
        rtol=0.0,
        atol=5e-4,
    ):
        raise AssertionError(
            "The manuscript's shared 2024/2025 annual-block coverage is no longer valid."
        )

    prediction_tokens = [
        f"{len(prediction_folds)} country-year folds and {len(prediction_intervals)} held-out reporting intervals",
        "Country-balanced empirical coverage of nominal 95% predictive intervals was "
        + _lancet_decimal(prediction_gate["primary_predictive_95_coverage"], 3),
        "unweighted interval-level diagnostic coverage of "
        + _lancet_decimal(
            prediction_gate["model_interval_95_coverage_diagnostic"], 3
        ),
        "mean log1p interval width of "
        + _lancet_decimal(prediction_gate["primary_predictive_mean_log1p_width"], 3),
        "mean log score of "
        + _lancet_decimal(overall_prediction["model_mean_log_score_per_interval"], 3),
        "mean absolute log1p error of "
        + _lancet_decimal(overall_prediction["model_mean_absolute_log1p_error"], 3),
        _lancet_decimal(
            overall_prediction["seasonal_naive_mean_log_score_per_interval"], 3
        ),
        _lancet_decimal(
            overall_prediction["ew_recent_rate_mean_log_score_per_interval"], 3
        ),
        _lancet_decimal(
            overall_prediction["damped_log_trend_mean_log_score_per_interval"], 3
        ),
        _lancet_decimal(
            overall_prediction["seasonal_naive_mean_absolute_log1p_error"], 3
        ),
        _lancet_decimal(
            overall_prediction["ew_recent_rate_mean_absolute_log1p_error"], 3
        ),
        _lancet_decimal(
            overall_prediction["damped_log_trend_mean_absolute_log1p_error"], 3
        ),
        "lowest country-specific interval coverage ("
        + _lancet_decimal(prediction_gate["minimum_country_interval_95_coverage"], 3)
        + ")",
        "joint annual coverage of "
        + _lancet_decimal(block_gate["model_annual_95_coverage_diagnostic"], 3)
        + " overall and "
        + _lancet_decimal(block_year_coverage.loc[2024], 3)
        + " in both 2024 and 2025",
    ]
    _require_text_tokens(
        results_text, prediction_tokens, label="Main-manuscript predictive results"
    )

    priorities = read_table(
        project_path("outputs", "tables", "table1_profile_programme_priorities.csv")
    )
    current_burden = pd.to_numeric(
        priorities["current_practice_cases_per_100k_under18"], errors="raise"
    )
    delivery_contrast = read_table(
        project_path("outputs", "tables", "figure2a_delivery_lever_contrast.csv")
    )
    decision_fragility = read_table(
        project_path("outputs", "tables", "figure2b_reference_choice_fragility.csv")
    )
    regret_draws = read_table(
        project_path("outputs", "tables", "figure2b_fixed_reference_regret_draws.csv")
    )
    programme_effects = read_table(
        project_path("outputs", "tables", "figure2c_programme_effect_matrix.csv")
    )
    rank1_counts = read_table(
        project_path("outputs", "tables", "figure2_selected_input_rank1_counts.csv")
    )

    def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
        missing = columns.difference(frame.columns)
        if missing:
            raise AssertionError(
                f"{label} source data are missing columns: "
                + ", ".join(sorted(missing))
            )

    def as_boolean(values: pd.Series, label: str) -> pd.Series:
        parsed = values.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
        if parsed.isna().any():
            raise AssertionError(f"{label} contains values other than true or false.")
        return parsed.astype(bool)

    priorities_with_country = priorities.assign(
        _country=priorities["programme_profile"]
        .astype(str)
        .str.replace(" ", "_", regex=False)
    )
    expected_countries = set(priorities_with_country["_country"].astype(str))
    priority_leaders = priorities_with_country.set_index("_country")[
        "lowest_burden_programme_only_strategy"
    ]
    priority_current = pd.to_numeric(
        priorities_with_country.set_index("_country")[
            "current_practice_cases_per_100k_under18"
        ],
        errors="raise",
    )

    burden_layers = read_table(
        project_path("outputs", "tables", "figure1b_burden_layers.csv")
    )
    require_columns(
        burden_layers,
        {
            "country",
            "outcome",
            "rate_per_100k",
            "interval_applies",
            "interval_lower_per_100k",
            "interval_upper_per_100k",
            "bootstrap_replicates",
            "interval_type",
            "interval_basis",
            "confidence_interval_method",
        },
        "Figure 1b burden layers",
    )
    expected_burden_outcomes = {"Reports", "Symptomatic", "Infections"}
    if (
        len(burden_layers) != len(expected_countries) * len(expected_burden_outcomes)
        or set(burden_layers["country"].astype(str)) != expected_countries
        or set(burden_layers["outcome"].astype(str)) != expected_burden_outcomes
        or burden_layers.duplicated(["country", "outcome"]).any()
    ):
        raise AssertionError("Figure 1b burden layers are not a complete 9 x 3 matrix.")
    for column in (
        "rate_per_100k",
        "interval_lower_per_100k",
        "interval_upper_per_100k",
        "bootstrap_replicates",
    ):
        burden_layers[column] = pd.to_numeric(burden_layers[column], errors="raise")
    if (
        not np.isfinite(burden_layers["rate_per_100k"]).all()
        or burden_layers["rate_per_100k"].le(0.0).any()
        or not np.isfinite(
            burden_layers[
                ["interval_lower_per_100k", "interval_upper_per_100k"]
            ].to_numpy(dtype=float)
        ).all()
        or burden_layers["interval_lower_per_100k"].lt(0.0).any()
        or burden_layers["interval_lower_per_100k"]
        .gt(burden_layers["interval_upper_per_100k"])
        .any()
        or burden_layers["bootstrap_replicates"].lt(1000).any()
    ):
        raise AssertionError("Figure 1b burden layers contain invalid rates or CIs.")
    if not as_boolean(
        burden_layers["interval_applies"], "Figure 1b interval-applies flag"
    ).all():
        raise AssertionError("Figure 1b does not apply its CIs to all three layers.")
    if (
        set(burden_layers["interval_type"].astype(str))
        != {"95% parametric-bootstrap confidence interval"}
        or set(burden_layers["confidence_interval_method"].astype(str))
        != {"percentile_parametric_bootstrap"}
        or not burden_layers["interval_basis"]
        .astype(str)
        .str.lower()
        .str.contains("conditional")
        .all()
        or not burden_layers["interval_basis"]
        .astype(str)
        .str.lower()
        .str.contains("fully refitted")
        .all()
    ):
        raise AssertionError(
            "Figure 1b CIs do not satisfy the conditional full-refit estimation contract."
        )

    figure1b_intervals = read_table(
        project_path(
            "outputs",
            "summaries",
            "figure1b_current_practice_conditional_confidence_intervals.csv",
        )
    )
    require_columns(
        figure1b_intervals,
        {
            "country",
            "outcome",
            "rate_q025",
            "rate_q975",
            "bootstrap_replicates",
            "interval_type",
            "confidence_interval_method",
        },
        "Figure 1b interval summary",
    )
    figure1b_intervals = figure1b_intervals.assign(
        country=figure1b_intervals["country"]
        .astype(str)
        .str.replace(" ", "_", regex=False)
    )
    if (
        len(figure1b_intervals) != len(burden_layers)
        or set(figure1b_intervals["country"].astype(str)) != expected_countries
        or set(figure1b_intervals["outcome"].astype(str))
        != expected_burden_outcomes
        or figure1b_intervals.duplicated(["country", "outcome"]).any()
    ):
        raise AssertionError("Figure 1b interval summary is not a complete 9 x 3 matrix.")
    interval_check = burden_layers.merge(
        figure1b_intervals,
        on=["country", "outcome"],
        how="left",
        validate="one_to_one",
        suffixes=("_figure", "_source"),
    )
    for figure_column, source_column in (
        ("interval_lower_per_100k", "rate_q025"),
        ("interval_upper_per_100k", "rate_q975"),
        ("bootstrap_replicates_figure", "bootstrap_replicates_source"),
    ):
        if not np.allclose(
            pd.to_numeric(interval_check[figure_column], errors="raise"),
            pd.to_numeric(interval_check[source_column], errors="raise"),
            rtol=0.0,
            atol=1e-9,
        ):
            raise AssertionError(
                f"Figure 1b displayed CIs disagree with their source in {figure_column}."
            )
    if (
        not interval_check["interval_type_figure"]
        .astype(str)
        .eq(interval_check["interval_type_source"].astype(str))
        .all()
        or not interval_check["confidence_interval_method_figure"]
        .astype(str)
        .eq(interval_check["confidence_interval_method_source"].astype(str))
        .all()
    ):
        raise AssertionError("Figure 1b displayed CI labels disagree with their source.")
    baseline_pediatric = read_table(
        project_path("outputs", "tables", "lancet_baseline_pediatric_burden.csv")
    )
    require_columns(
        baseline_pediatric,
        {
            "country",
            "primary_cases_per_100k",
            "child_adolescent_reported_cases_per_100k",
            "child_adolescent_infections_per_100k",
        },
        "Baseline paediatric burden",
    )
    baseline_pediatric = baseline_pediatric.assign(
        country=baseline_pediatric["country"]
        .astype(str)
        .str.replace(" ", "_", regex=False)
    )
    if (
        len(baseline_pediatric) != len(expected_countries)
        or set(baseline_pediatric["country"].astype(str)) != expected_countries
        or baseline_pediatric["country"].duplicated().any()
    ):
        raise AssertionError(
            "Baseline paediatric burden is not one row per publication profile."
        )
    burden_wide = burden_layers.pivot(
        index="country", columns="outcome", values="rate_per_100k"
    ).sort_index()
    baseline_indexed = baseline_pediatric.set_index("country").sort_index()
    burden_crosswalk = {
        "Reports": "child_adolescent_reported_cases_per_100k",
        "Symptomatic": "primary_cases_per_100k",
        "Infections": "child_adolescent_infections_per_100k",
    }
    for burden_outcome, baseline_column in burden_crosswalk.items():
        baseline_values = pd.to_numeric(
            baseline_indexed[baseline_column], errors="raise"
        )
        if not np.allclose(
            burden_wide[burden_outcome],
            baseline_values,
            rtol=0.0,
            atol=1e-9,
        ):
            raise AssertionError(
                f"Figure 1b {burden_outcome} rates disagree with current baseline "
                "paediatric burden."
            )

    def burden_rates(outcome: str) -> pd.Series:
        rows = burden_layers.loc[
            burden_layers["outcome"].astype(str).eq(outcome), "rate_per_100k"
        ]
        if len(rows) != len(expected_countries):
            raise AssertionError(f"Incomplete Figure 1b burden layer for {outcome}.")
        return rows

    reported_rates = burden_rates("Reports")
    symptomatic_rates = burden_rates("Symptomatic")
    infection_rates = burden_rates("Infections")

    def burden_extrema(outcome: str) -> tuple[pd.Series, pd.Series]:
        rows = burden_layers.loc[
            burden_layers["outcome"].astype(str).eq(outcome)
        ].copy()
        if len(rows) != len(expected_countries):
            raise AssertionError(f"Incomplete Figure 1b extrema for {outcome}.")
        return (
            rows.loc[rows["rate_per_100k"].idxmin()],
            rows.loc[rows["rate_per_100k"].idxmax()],
        )

    reported_low, reported_high = burden_extrema("Reports")
    symptomatic_low, symptomatic_high = burden_extrema("Symptomatic")
    infection_low, infection_high = burden_extrema("Infections")
    lowest_profiles = {
        str(reported_low["country"]),
        str(symptomatic_low["country"]),
        str(infection_low["country"]),
    }
    highest_profiles = {
        str(reported_high["country"]),
        str(symptomatic_high["country"]),
        str(infection_high["country"]),
    }
    if len(lowest_profiles) != 1 or len(highest_profiles) != 1:
        raise AssertionError(
            "Figure 1b layers no longer share one lowest and one highest profile."
        )
    lowest_profile = next(iter(lowest_profiles)).replace("_", " ")
    highest_profile = next(iter(highest_profiles)).replace("_", " ")
    _require_text_tokens(
        results_text,
        [
            "cross-profile median annualised indices among people younger than 18 "
            "years were "
            + _lancet_decimal(reported_rates.median(), 1)
            + " reported cases, "
            + _lancet_decimal(symptomatic_rates.median(), 1)
            + " symptomatic cases, and "
            + _lancet_decimal(infection_rates.median(), 1)
            + " infections per 100 000",
            "Across all three layers, "
            + lowest_profile
            + " had the lowest and "
            + highest_profile
            + " the highest point estimates",
            "Reported-case indices were "
            + _lancet_decimal(reported_low["rate_per_100k"], 1)
            + " (profile-specific conditional parametric-bootstrap 95% estimation CI "
            + _lancet_decimal(reported_low["interval_lower_per_100k"], 1)
            + "–"
            + _lancet_decimal(reported_low["interval_upper_per_100k"], 1)
            + ") and "
            + _lancet_decimal(reported_high["rate_per_100k"], 1)
            + " ("
            + _lancet_decimal(reported_high["interval_lower_per_100k"], 1)
            + "–"
            + _lancet_decimal(reported_high["interval_upper_per_100k"], 1)
            + "), respectively",
            "Corresponding symptomatic-case indices were "
            + _lancet_decimal(symptomatic_low["rate_per_100k"], 1)
            + " ("
            + _lancet_decimal(symptomatic_low["interval_lower_per_100k"], 1)
            + "–"
            + _lancet_decimal(symptomatic_low["interval_upper_per_100k"], 1)
            + ") and "
            + _lancet_decimal(symptomatic_high["rate_per_100k"], 1)
            + " ("
            + _lancet_decimal(symptomatic_high["interval_lower_per_100k"], 1)
            + "–"
            + _lancet_decimal(symptomatic_high["interval_upper_per_100k"], 1)
            + "), and infection indices were "
            + _lancet_decimal(infection_low["rate_per_100k"], 1)
            + " ("
            + _lancet_decimal(infection_low["interval_lower_per_100k"], 1)
            + "–"
            + _lancet_decimal(infection_low["interval_upper_per_100k"], 1)
            + ") and "
            + _lancet_decimal(infection_high["rate_per_100k"], 1)
            + " ("
            + _lancet_decimal(infection_high["interval_lower_per_100k"], 1)
            + "–"
            + _lancet_decimal(infection_high["interval_upper_per_100k"], 1)
            + ") per 100 000",
            "The symptomatic-case IQR was "
            + _lancet_decimal(symptomatic_rates.quantile(0.25), 1)
            + "–"
            + _lancet_decimal(symptomatic_rates.quantile(0.75), 1)
            + " per 100 000 across the nine profiles (figure 1C; table 1)",
        ],
        label="Main-manuscript Figure 1b results",
    )

    # Figure 2a: two production-runtime levers against one current comparator.
    required_delivery_columns = {
        "country",
        "configured_lever",
        "configured_lever_label",
        "deterministic_primary_cases_per_100k",
        "current_practice_cases_per_100k",
        "deterministic_relative_reduction",
        "deterministic_relative_reduction_percent",
        "coverage_floor_applied",
        "timeliness_applied",
    }
    require_columns(delivery_contrast, required_delivery_columns, "Figure 2a")
    expected_levers = {"coverage_floor_only", "timeliness_only"}
    if (
        len(delivery_contrast) != len(priorities) * len(expected_levers)
        or set(delivery_contrast["country"].astype(str)) != expected_countries
        or set(delivery_contrast["configured_lever"].astype(str)) != expected_levers
        or delivery_contrast.duplicated(["country", "configured_lever"]).any()
    ):
        raise AssertionError("Figure 2a is not a complete 9 x 2 delivery-lever contrast.")
    for column in (
        "deterministic_primary_cases_per_100k",
        "current_practice_cases_per_100k",
        "deterministic_relative_reduction",
        "deterministic_relative_reduction_percent",
    ):
        delivery_contrast[column] = pd.to_numeric(
            delivery_contrast[column], errors="raise"
        )
        if not np.isfinite(delivery_contrast[column]).all():
            raise AssertionError(f"Figure 2a contains non-finite values in {column}.")
    if (
        delivery_contrast["deterministic_primary_cases_per_100k"].le(0.0).any()
        or delivery_contrast["current_practice_cases_per_100k"].le(0.0).any()
    ):
        raise AssertionError("Figure 2a contains non-positive case-rate denominators.")
    delivery_recalculated = 1.0 - (
        delivery_contrast["deterministic_primary_cases_per_100k"]
        / delivery_contrast["current_practice_cases_per_100k"]
    )
    if not np.allclose(
        delivery_contrast["deterministic_relative_reduction"],
        delivery_recalculated,
        rtol=1e-12,
        atol=1e-12,
    ) or not np.allclose(
        delivery_contrast["deterministic_relative_reduction_percent"],
        100.0 * delivery_recalculated,
        rtol=1e-12,
        atol=1e-10,
    ):
        raise AssertionError("Figure 2a relative reductions fail denominator arithmetic.")
    coverage_flags = as_boolean(
        delivery_contrast["coverage_floor_applied"],
        "Figure 2a coverage-floor flag",
    )
    timeliness_flags = as_boolean(
        delivery_contrast["timeliness_applied"],
        "Figure 2a timeliness flag",
    )
    is_coverage = delivery_contrast["configured_lever"].astype(str).eq(
        "coverage_floor_only"
    )
    if not (
        coverage_flags.eq(is_coverage).all()
        and timeliness_flags.eq(~is_coverage).all()
    ):
        raise AssertionError("Figure 2a lever flags disagree with the configured contrast.")
    for country, group in delivery_contrast.groupby("country", sort=False):
        if not np.allclose(
            group["current_practice_cases_per_100k"],
            float(priority_current.loc[country]),
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(
                f"Figure 2a does not share the Table 1 current comparator for {country}."
            )
    delivery_paired = delivery_contrast.pivot(
        index="country",
        columns="configured_lever",
        values="deterministic_relative_reduction_percent",
    )
    coverage_effects = delivery_paired["coverage_floor_only"]
    timeliness = delivery_paired["timeliness_only"]
    timeliness_advantage = timeliness - coverage_effects
    if not timeliness_advantage.gt(0.0).all():
        raise AssertionError("Figure 2a no longer favours timeliness in all nine profiles.")

    # Figure 2c: the complete six-strategy effect matrix and its marginal CIs.
    required_effect_columns = {
        "country",
        "strategy",
        "strategy_label",
        "burden_rank",
        "reference_leader",
        "deterministic_primary_cases_per_100k",
        "current_practice_cases_per_100k",
        "deterministic_absolute_reduction_per_100k",
        "deterministic_relative_reduction",
        "bootstrap_relative_reduction_median",
        "bootstrap_relative_reduction_q025",
        "bootstrap_relative_reduction_q25",
        "bootstrap_relative_reduction_q75",
        "bootstrap_relative_reduction_q975",
        "successful_bootstrap_replicates",
        "confidence_interval_type",
        "confidence_interval_method",
    }
    require_columns(programme_effects, required_effect_columns, "Figure 2c")
    if (
        len(programme_effects) != len(priorities) * len(PROGRAMME_ONLY_STRATEGIES)
        or set(programme_effects["country"].astype(str)) != expected_countries
        or set(programme_effects["strategy"].astype(str))
        != set(PROGRAMME_ONLY_STRATEGIES)
        or programme_effects.duplicated(["country", "strategy"]).any()
    ):
        raise AssertionError("Figure 2c source data are not a complete 9 x 6 matrix.")
    effect_numeric_columns = (
        "burden_rank",
        "deterministic_primary_cases_per_100k",
        "current_practice_cases_per_100k",
        "deterministic_absolute_reduction_per_100k",
        "deterministic_relative_reduction",
        "bootstrap_relative_reduction_median",
        "bootstrap_relative_reduction_q025",
        "bootstrap_relative_reduction_q25",
        "bootstrap_relative_reduction_q75",
        "bootstrap_relative_reduction_q975",
        "successful_bootstrap_replicates",
    )
    for column in effect_numeric_columns:
        programme_effects[column] = pd.to_numeric(
            programme_effects[column], errors="raise"
        )
        if not np.isfinite(programme_effects[column]).all():
            raise AssertionError(f"Figure 2c contains non-finite values in {column}.")
    if (
        programme_effects["deterministic_primary_cases_per_100k"].le(0.0).any()
        or programme_effects["current_practice_cases_per_100k"].le(0.0).any()
    ):
        raise AssertionError("Figure 2c contains non-positive case-rate denominators.")
    matrix_recalculated_reduction = 1.0 - (
        programme_effects["deterministic_primary_cases_per_100k"]
        / programme_effects["current_practice_cases_per_100k"]
    )
    matrix_recalculated_absolute = (
        programme_effects["current_practice_cases_per_100k"]
        - programme_effects["deterministic_primary_cases_per_100k"]
    )
    if not np.allclose(
        programme_effects["deterministic_relative_reduction"],
        matrix_recalculated_reduction,
        rtol=1e-12,
        atol=1e-12,
    ) or not np.allclose(
        programme_effects["deterministic_absolute_reduction_per_100k"],
        matrix_recalculated_absolute,
        rtol=1e-12,
        atol=1e-10,
    ):
        raise AssertionError("Figure 2c effects fail common-denominator arithmetic.")
    reference_leaders = as_boolean(
        programme_effects["reference_leader"], "Figure 2c reference-leader flag"
    )
    programme_effects = programme_effects.assign(_reference_leader=reference_leaders)
    selected_rows = []
    for country, group in programme_effects.groupby("country", sort=False):
        if set(group["strategy"].astype(str)) != set(PROGRAMME_ONLY_STRATEGIES):
            raise AssertionError(f"Figure 2 strategy set is incomplete for {country}.")
        if not np.allclose(
            group["current_practice_cases_per_100k"],
            float(priority_current.loc[country]),
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(
                f"Figure 2c strategies do not share the Table 1 comparator for {country}."
            )
        observed_ranks = group["burden_rank"].to_numpy(dtype=float)
        if set(observed_ranks) != set(range(1, len(PROGRAMME_ONLY_STRATEGIES) + 1)):
            raise AssertionError(f"Figure 2c burden ranks are incomplete for {country}.")
        ordered = group.sort_values("deterministic_primary_cases_per_100k")
        if not np.array_equal(
            ordered["burden_rank"].to_numpy(dtype=int),
            np.arange(1, len(PROGRAMME_ONLY_STRATEGIES) + 1),
        ):
            raise AssertionError(f"Figure 2c burden ranks disagree with rates for {country}.")
        leader_rows = group.loc[group["_reference_leader"]]
        if len(leader_rows) != 1 or int(leader_rows.iloc[0]["burden_rank"]) != 1:
            raise AssertionError(
                f"Figure 2c does not mark exactly one rank-1 reference choice for {country}."
            )
        leader = leader_rows.iloc[0]
        if str(priority_leaders.loc[country]) != str(leader["strategy_label"]):
            raise AssertionError(
                f"Figure 2c reference choice and Table 1 disagree for {country}."
            )
        selected_rows.append(leader)
    selected_effects = pd.DataFrame(selected_rows)
    interval_columns = [
        "bootstrap_relative_reduction_q025",
        "bootstrap_relative_reduction_q25",
        "bootstrap_relative_reduction_median",
        "bootstrap_relative_reduction_q75",
        "bootstrap_relative_reduction_q975",
    ]
    interval_values = programme_effects[interval_columns].to_numpy(dtype=float)
    if (np.diff(interval_values, axis=1) < -1e-12).any():
        raise AssertionError("Figure 2c bootstrap quantiles are not ordered.")
    _validate_figure2c_estimation_ci_language(
        programme_effects["confidence_interval_type"],
        programme_effects["confidence_interval_method"],
    )
    figure2c_bootstrap_metadata = validate_run_metadata(
        "figure2c_parametric_bootstrap"
    )
    observed_refit_minimum, observed_refit_maximum = (
        _validate_figure2c_successful_refit_gate(
            programme_effects,
            figure2c_bootstrap_metadata,
            expected_countries=expected_countries,
        )
    )
    intervals_include_zero = (
        programme_effects["bootstrap_relative_reduction_q025"].le(0.0)
        & programme_effects["bootstrap_relative_reduction_q975"].ge(0.0)
    )
    if int(intervals_include_zero.sum()) != 17:
        raise AssertionError("Figure 2c no longer has 17 marginal intervals including zero.")
    if not selected_effects["bootstrap_relative_reduction_q025"].gt(0.0).all():
        raise AssertionError("A Figure 2c reference-leader interval now includes zero.")

    leader_counts = selected_effects["strategy_label"].astype(str).value_counts()
    infant_leader_profiles = set(
        selected_effects.loc[
            selected_effects["strategy_label"].astype(str).eq(
                "Infant-exposure package"
            ),
            "country",
        ].astype(str)
    )
    if (
        int(leader_counts.get("Routine timeliness", 0)) != 8
        or int(leader_counts.get("Infant-exposure package", 0)) != 1
        or infant_leader_profiles != {"China"}
    ):
        raise AssertionError("Figure 2 locked reference programme leaders changed unexpectedly.")

    paired_status = priorities["paired_interval_status"].astype(str)
    expected_status = np.where(
        pd.to_numeric(
            priorities["runner_up_excess_cases_per_100k_under18_q025"],
            errors="raise",
        )
        > 0.0,
        "above_zero",
        "includes_zero",
    )
    if not np.array_equal(paired_status.to_numpy(), expected_status):
        raise AssertionError("Table 1 paired interval status disagrees with its lower limit.")
    includes_zero_profiles = set(
        priorities.loc[
            paired_status.eq("includes_zero"), "programme_profile"
        ].astype(str)
    )
    if includes_zero_profiles != {"Thailand", "United Kingdom"}:
        raise AssertionError("Figure 2 paired leader-runner interval classification changed.")
    above_zero_count = int(paired_status.eq("above_zero").sum())

    # Figure 2b: complete rank counts, fixed-reference regret arithmetic, and summary.
    required_rank_columns = {
        "country",
        "strategy",
        "rank1_count",
        "prespecified_setting_count",
        "rank1_fraction_for_arithmetic_check",
        "interpretation",
    }
    require_columns(rank1_counts, required_rank_columns, "Figure 2 rank-1 audit")
    _validate_figure2b_interpretation_language(rank1_counts["interpretation"])
    if (
        len(rank1_counts) != len(priorities) * len(PROGRAMME_ONLY_STRATEGIES)
        or set(rank1_counts["country"].astype(str)) != expected_countries
        or set(rank1_counts["strategy"].astype(str))
        != set(PROGRAMME_ONLY_STRATEGIES)
        or rank1_counts.duplicated(["country", "strategy"]).any()
    ):
        raise AssertionError("Figure 2 rank-1 counts are not a complete 9 x 6 matrix.")
    for column in (
        "rank1_count",
        "prespecified_setting_count",
        "rank1_fraction_for_arithmetic_check",
    ):
        rank1_counts[column] = pd.to_numeric(rank1_counts[column], errors="raise")
    if (
        not rank1_counts["prespecified_setting_count"].eq(128).all()
        or not rank1_counts["rank1_count"].between(0, 128).all()
        or not np.allclose(
            rank1_counts["rank1_count"],
            np.round(rank1_counts["rank1_count"]),
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            rank1_counts["rank1_fraction_for_arithmetic_check"],
            rank1_counts["rank1_count"] / 128.0,
            rtol=0.0,
            atol=1e-12,
        )
        or not rank1_counts.groupby("country")["rank1_count"].sum().eq(128).all()
    ):
        raise AssertionError("Figure 2 rank-1 count arithmetic is invalid.")

    required_regret_columns = {
        "country",
        "selected_input_setting",
        "locked_reference_strategy",
        "setting_specific_winning_strategy",
        "reference_choice_retained",
        "current_practice_cases_per_100k",
        "locked_reference_strategy_cases_per_100k",
        "setting_specific_best_cases_per_100k",
        "fixed_reference_regret_cases_per_100k",
        "fixed_reference_regret_relative_to_current",
        "fixed_reference_regret_percentage_points",
    }
    require_columns(regret_draws, required_regret_columns, "Figure 2b regret-draw")
    if (
        len(regret_draws) != len(priorities) * 128
        or set(regret_draws["country"].astype(str)) != expected_countries
        or regret_draws.duplicated(["country", "selected_input_setting"]).any()
    ):
        raise AssertionError("Figure 2b does not contain one row per profile and setting.")
    regret_draws["selected_input_setting"] = pd.to_numeric(
        regret_draws["selected_input_setting"], errors="raise"
    )
    for country, group in regret_draws.groupby("country", sort=False):
        if set(group["selected_input_setting"].to_numpy(dtype=int)) != set(
            range(1, 129)
        ):
            raise AssertionError(f"Figure 2b settings are incomplete for {country}.")
        locked_choices = set(group["locked_reference_strategy"].astype(str))
        if len(locked_choices) != 1:
            raise AssertionError(f"Figure 2b fixed choice varies within {country}.")
        locked_choice = next(iter(locked_choices))
        expected_choice = str(
            programme_effects.loc[
                programme_effects["country"].astype(str).eq(country)
                & programme_effects["_reference_leader"],
                "strategy",
            ].iloc[0]
        )
        if locked_choice != expected_choice:
            raise AssertionError(
                f"Figure 2b fixed choice and Figure 2c reference leader disagree for {country}."
            )
    if not set(regret_draws["setting_specific_winning_strategy"].astype(str)).issubset(
        set(PROGRAMME_ONLY_STRATEGIES)
    ):
        raise AssertionError("Figure 2b contains a winning strategy outside the ranking set.")
    regret_numeric_columns = (
        "current_practice_cases_per_100k",
        "locked_reference_strategy_cases_per_100k",
        "setting_specific_best_cases_per_100k",
        "fixed_reference_regret_cases_per_100k",
        "fixed_reference_regret_relative_to_current",
        "fixed_reference_regret_percentage_points",
    )
    for column in regret_numeric_columns:
        regret_draws[column] = pd.to_numeric(regret_draws[column], errors="raise")
        if not np.isfinite(regret_draws[column]).all():
            raise AssertionError(f"Figure 2b contains non-finite values in {column}.")
    if regret_draws["current_practice_cases_per_100k"].le(0.0).any():
        raise AssertionError("Figure 2b contains non-positive current-practice denominators.")
    recalculated_regret_cases = (
        regret_draws["locked_reference_strategy_cases_per_100k"]
        - regret_draws["setting_specific_best_cases_per_100k"]
    )
    recalculated_regret_relative = (
        recalculated_regret_cases
        / regret_draws["current_practice_cases_per_100k"]
    )
    if (
        (recalculated_regret_cases < -1e-10).any()
        or not np.allclose(
            regret_draws["fixed_reference_regret_cases_per_100k"],
            recalculated_regret_cases,
            rtol=1e-12,
            atol=1e-10,
        )
        or not np.allclose(
            regret_draws["fixed_reference_regret_relative_to_current"],
            recalculated_regret_relative,
            rtol=1e-12,
            atol=1e-12,
        )
        or not np.allclose(
            regret_draws["fixed_reference_regret_percentage_points"],
            100.0 * recalculated_regret_relative,
            rtol=1e-12,
            atol=1e-10,
        )
    ):
        raise AssertionError("Figure 2b fixed-reference regret arithmetic is invalid.")
    regret_retained = as_boolean(
        regret_draws["reference_choice_retained"],
        "Figure 2b reference-choice-retained flag",
    )
    expected_retained = (
        regret_draws["locked_reference_strategy"].astype(str)
        == regret_draws["setting_specific_winning_strategy"].astype(str)
    )
    if not (
        regret_retained.eq(expected_retained).all()
        and regret_retained.eq(recalculated_regret_cases.abs().le(1e-10)).all()
    ):
        raise AssertionError("Figure 2b retention flags disagree with setting-specific regret.")

    raw_winner_counts = (
        regret_draws.groupby(
            ["country", "setting_specific_winning_strategy"], as_index=False
        )
        .size()
        .rename(
            columns={
                "setting_specific_winning_strategy": "strategy",
                "size": "raw_rank1_count",
            }
        )
    )
    rank_count_comparison = rank1_counts.merge(
        raw_winner_counts,
        on=["country", "strategy"],
        how="left",
        validate="one_to_one",
    )
    rank_count_comparison["raw_rank1_count"] = rank_count_comparison[
        "raw_rank1_count"
    ].fillna(0)
    if not np.array_equal(
        rank_count_comparison["rank1_count"].to_numpy(dtype=int),
        rank_count_comparison["raw_rank1_count"].to_numpy(dtype=int),
    ):
        raise AssertionError("Figure 2 rank-1 counts disagree with the regret-draw winners.")

    required_fragility_columns = {
        "country",
        "locked_reference_strategy",
        "prespecified_setting_count",
        "reference_choice_retained_count",
        "reference_choice_retained_fraction_for_arithmetic_check",
        "positive_regret_count",
        "regret_percentage_points_median",
        "regret_percentage_points_q75",
        "regret_percentage_points_q95",
        "regret_percentage_points_q975",
        "regret_percentage_points_max",
        "regret_cases_per_100k_q95",
        "interpretation",
    }
    require_columns(decision_fragility, required_fragility_columns, "Figure 2b")
    _validate_figure2b_interpretation_language(decision_fragility["interpretation"])
    if (
        len(decision_fragility) != len(priorities)
        or set(decision_fragility["country"].astype(str)) != expected_countries
        or decision_fragility["country"].duplicated().any()
    ):
        raise AssertionError("Figure 2b fragility summary is not one row per profile.")
    fragility_numeric_columns = tuple(required_fragility_columns - {
        "country",
        "locked_reference_strategy",
        "interpretation",
    })
    for column in fragility_numeric_columns:
        decision_fragility[column] = pd.to_numeric(
            decision_fragility[column], errors="raise"
        )
        if not np.isfinite(decision_fragility[column]).all():
            raise AssertionError(f"Figure 2b contains non-finite values in {column}.")
    if not decision_fragility["prespecified_setting_count"].eq(128).all():
        raise AssertionError("Figure 2b fragility summaries are not based on 128 settings.")
    raw_fragility_rows = []
    for country, group in regret_draws.groupby("country", sort=False):
        pp = group["fixed_reference_regret_percentage_points"].to_numpy(dtype=float)
        cases = group["fixed_reference_regret_cases_per_100k"].to_numpy(dtype=float)
        raw_fragility_rows.append(
            {
                "country": country,
                "locked_reference_strategy_check": str(
                    group["locked_reference_strategy"].iloc[0]
                ),
                "reference_choice_retained_count_check": int(
                    regret_retained.loc[group.index].sum()
                ),
                "reference_choice_retained_fraction_for_arithmetic_check_check": float(
                    regret_retained.loc[group.index].mean()
                ),
                "positive_regret_count_check": int((pp > 1e-10).sum()),
                "regret_percentage_points_median_check": float(np.quantile(pp, 0.5)),
                "regret_percentage_points_q75_check": float(np.quantile(pp, 0.75)),
                "regret_percentage_points_q95_check": float(np.quantile(pp, 0.95)),
                "regret_percentage_points_q975_check": float(np.quantile(pp, 0.975)),
                "regret_percentage_points_max_check": float(pp.max()),
                "regret_cases_per_100k_q95_check": float(np.quantile(cases, 0.95)),
            }
        )
    raw_fragility = pd.DataFrame(raw_fragility_rows)
    fragility_check = decision_fragility.merge(
        raw_fragility, on="country", how="left", validate="one_to_one"
    )
    if not (
        fragility_check["locked_reference_strategy"].astype(str).eq(
            fragility_check["locked_reference_strategy_check"].astype(str)
        ).all()
        and fragility_check["reference_choice_retained_count"].eq(
            fragility_check["reference_choice_retained_count_check"]
        ).all()
        and fragility_check["positive_regret_count"].eq(
            fragility_check["positive_regret_count_check"]
        ).all()
    ):
        raise AssertionError("Figure 2b fragility counts disagree with raw regret draws.")
    for column in (
        "reference_choice_retained_fraction_for_arithmetic_check",
        "regret_percentage_points_median",
        "regret_percentage_points_q75",
        "regret_percentage_points_q95",
        "regret_percentage_points_q975",
        "regret_percentage_points_max",
        "regret_cases_per_100k_q95",
    ):
        if not np.allclose(
            fragility_check[column],
            fragility_check[f"{column}_check"],
            rtol=1e-12,
            atol=1e-10,
        ):
            raise AssertionError(
                f"Figure 2b fragility summary disagrees with raw draws for {column}."
            )

    figure2b_headline = _figure2b_headline_tokens(decision_fragility)

    matrix_point_percent = 100.0 * programme_effects[
        "deterministic_relative_reduction"
    ]
    if (
        round(float(matrix_point_percent.min()), 1) != -6.3
        or round(float(matrix_point_percent.max()), 1) != 27.0
        or round(float(coverage_effects.median()), 1) != -0.5
        or round(float(timeliness.median()), 1) != 15.4
    ):
        raise AssertionError("Figure 2a/c headline effect values changed unexpectedly.")

    programme_tokens = [
        "In the complete strategy–profile matrix, routine timeliness was the locked "
        "lowest-burden programme in eight profiles and the infant-exposure package "
        "was lowest in China",
        "Locked point reductions ranged from "
        + _lancet_decimal(matrix_point_percent.min(), 1)
        + "% to "
        + _lancet_decimal(matrix_point_percent.max(), 1)
        + "% across the 54 cells; "
        + str(int(intervals_include_zero.sum()))
        + " full-refit 95% estimation confidence intervals included zero, whereas all nine "
        "black-outlined reference-leader intervals were positive",
        "The paired leader–runner intervals were above zero in "
        + _small_number_word(above_zero_count)
        + " profiles and included zero in Thailand and the UK",
        "routine timeliness reduced the conditional symptomatic-case index among people "
        "younger than 18 years by a median "
        + _lancet_decimal(timeliness.median(), 1)
        + "% (IQR "
        + _lancet_decimal(timeliness.quantile(0.25), 1)
        + "–"
        + _lancet_decimal(timeliness.quantile(0.75), 1)
        + "), whereas the nominal coverage-floor-only contrast changed it by "
        + _lancet_decimal(coverage_effects.median(), 1)
        + "% ("
        + _lancet_decimal(coverage_effects.quantile(0.25), 1)
        + " to "
        + _lancet_decimal(coverage_effects.quantile(0.75), 1)
        + ")",
        "Timeliness produced the larger reduction in all nine profiles, with a median "
        "within-profile advantage of "
        + _lancet_decimal(timeliness_advantage.median(), 1)
        + " percentage points (IQR "
        + _lancet_decimal(timeliness_advantage.quantile(0.25), 1)
        + "–"
        + _lancet_decimal(timeliness_advantage.quantile(0.75), 1)
        + "; figure 2A)",
        *figure2b_headline["results_tokens"],
    ]
    _require_text_tokens(
        results_text, programme_tokens, label="Main-manuscript programme results"
    )

    endpoint_effects = read_table(
        project_path("outputs", "tables", "figure3a_endpoint_effect_matrix.csv")
    )
    endpoint_outcomes = (
        "Infant cases",
        "Infant hospitalisations",
        "Infant deaths",
        "Children cases",
        "Adolescent cases",
        "All <18 cases",
    )
    require_columns(
        endpoint_effects,
        {"country", "strategy", "outcome", "relative_case_reduction"},
        "Figure 3a",
    )
    if (
        len(endpoint_effects)
        != len(expected_countries)
        * len(PROGRAMME_ONLY_STRATEGIES)
        * len(endpoint_outcomes)
        or set(endpoint_effects["country"].astype(str)) != expected_countries
        or set(endpoint_effects["strategy"].astype(str))
        != set(PROGRAMME_ONLY_STRATEGIES)
        or set(endpoint_effects["outcome"].astype(str)) != set(endpoint_outcomes)
        or endpoint_effects.duplicated(["country", "strategy", "outcome"]).any()
    ):
        raise AssertionError("Figure 3a is not a complete 9 x 6 x 6 effect matrix.")
    endpoint_effects["relative_case_reduction"] = pd.to_numeric(
        endpoint_effects["relative_case_reduction"], errors="raise"
    )
    if not np.isfinite(endpoint_effects["relative_case_reduction"]).all():
        raise AssertionError("Figure 3a contains non-finite relative reductions.")

    def endpoint_median_percent(strategy: str, outcome: str) -> float:
        rows = endpoint_effects.loc[
            endpoint_effects["strategy"].astype(str).eq(strategy)
            & endpoint_effects["outcome"].astype(str).eq(outcome),
            "relative_case_reduction",
        ]
        if len(rows) != len(expected_countries):
            raise AssertionError(
                f"Incomplete Figure 3a results for {strategy}, {outcome}."
            )
        return 100.0 * float(rows.median())

    timeliness_endpoint_medians = {
        outcome: endpoint_median_percent("timeliness_only", outcome)
        for outcome in endpoint_outcomes
    }
    pregnancy_endpoint_medians = {
        outcome: endpoint_median_percent("pregnancy_tdap_scaleup", outcome)
        for outcome in endpoint_outcomes
    }
    infant_exposure_endpoint_medians = {
        outcome: endpoint_median_percent("maternal_immunization", outcome)
        for outcome in endpoint_outcomes
    }
    if abs(pregnancy_endpoint_medians["Adolescent cases"]) >= 0.05:
        raise AssertionError(
            "Pregnancy Tdap adolescent-case median is no longer approximately zero."
        )
    figure3a_tokens = [
        "Routine timeliness reduced infant cases, infant hospitalisations, infant "
        "deaths, child cases, adolescent cases, and all-<18 cases by cross-profile "
        "medians of "
        + ", ".join(
            _lancet_decimal(timeliness_endpoint_medians[outcome], 1) + "%"
            for outcome in endpoint_outcomes[:-1]
        )
        + ", and "
        + _lancet_decimal(timeliness_endpoint_medians[endpoint_outcomes[-1]], 1)
        + "%",
        "Pregnancy Tdap scale-up had modest median effects on child, adolescent, "
        "and all-<18 cases ("
        + _lancet_decimal(pregnancy_endpoint_medians["Children cases"], 1)
        + "%, approximately 0%, and "
        + _lancet_decimal(pregnancy_endpoint_medians["All <18 cases"], 1)
        + "%) but larger effects on infant cases, hospitalisations, and deaths ("
        + _lancet_decimal(pregnancy_endpoint_medians["Infant cases"], 1)
        + "%, "
        + _lancet_decimal(
            pregnancy_endpoint_medians["Infant hospitalisations"], 1
        )
        + "%, and "
        + _lancet_decimal(pregnancy_endpoint_medians["Infant deaths"], 1)
        + "%)",
        "The infant-exposure package showed the same endpoint asymmetry, with median "
        "reductions of "
        + _lancet_decimal(infant_exposure_endpoint_medians["Infant cases"], 1)
        + "%, "
        + _lancet_decimal(
            infant_exposure_endpoint_medians["Infant hospitalisations"], 1
        )
        + "%, and "
        + _lancet_decimal(infant_exposure_endpoint_medians["Infant deaths"], 1)
        + "% for the three infant outcomes but "
        + _lancet_decimal(infant_exposure_endpoint_medians["Children cases"], 1)
        + "%, "
        + _lancet_decimal(infant_exposure_endpoint_medians["Adolescent cases"], 1)
        + "%, and "
        + _lancet_decimal(infant_exposure_endpoint_medians["All <18 cases"], 1)
        + "% for child, adolescent, and all-<18 cases",
    ]
    _require_text_tokens(
        results_text, figure3a_tokens, label="Main-manuscript Figure 3a results"
    )

    endpoint_gaps = read_table(
        project_path(
            "outputs", "tables", "figure3b_infant_to_child_adolescent_gap.csv"
        )
    )
    require_columns(
        endpoint_gaps,
        {"country", "strategy", "infant_minus_overall_gap_pp"},
        "Figure 3b",
    )
    if (
        len(endpoint_gaps)
        != len(expected_countries) * len(PROGRAMME_ONLY_STRATEGIES)
        or set(endpoint_gaps["country"].astype(str)) != expected_countries
        or set(endpoint_gaps["strategy"].astype(str))
        != set(PROGRAMME_ONLY_STRATEGIES)
        or endpoint_gaps.duplicated(["country", "strategy"]).any()
    ):
        raise AssertionError("Figure 3b is not a complete 9 x 6 endpoint-gap matrix.")
    endpoint_gaps["infant_minus_overall_gap_pp"] = pd.to_numeric(
        endpoint_gaps["infant_minus_overall_gap_pp"], errors="raise"
    )
    if not np.isfinite(endpoint_gaps["infant_minus_overall_gap_pp"]).all():
        raise AssertionError("Figure 3b contains non-finite endpoint gaps.")

    def endpoint_gap_median(strategy: str) -> float:
        rows = endpoint_gaps.loc[
            endpoint_gaps["strategy"].astype(str).eq(strategy),
            "infant_minus_overall_gap_pp",
        ]
        if len(rows) != len(expected_countries):
            raise AssertionError(f"Incomplete Figure 3b results for {strategy}.")
        return float(rows.median())

    gap_tokens = [
        "The median profile-specific advantage for infant-case reduction over "
        "all-<18-case reduction was "
        + _lancet_decimal(endpoint_gap_median("maternal_immunization"), 1)
        + " percentage points for the infant-exposure package, "
        + _lancet_decimal(endpoint_gap_median("timeliness_only"), 1)
        + " for routine timeliness, and "
        + _lancet_decimal(endpoint_gap_median("pregnancy_tdap_scaleup"), 1)
        + " for pregnancy Tdap scale-up",
    ]

    adolescent_effects = read_table(
        project_path(
            "outputs", "tables", "figure3d_adolescent_booster_profile_effects.csv"
        )
    )
    require_columns(
        adolescent_effects,
        {
            "country",
            "adolescent_effect_rank",
            "all_under18_symptomatic_case_reduction",
            "adolescent_case_reduction",
        },
        "Figure 3c",
    )
    if (
        len(adolescent_effects) != len(expected_countries)
        or set(adolescent_effects["country"].astype(str)) != expected_countries
        or adolescent_effects["country"].duplicated().any()
    ):
        raise AssertionError("Figure 3c is not one row per publication profile.")
    for column in (
        "adolescent_effect_rank",
        "all_under18_symptomatic_case_reduction",
        "adolescent_case_reduction",
    ):
        adolescent_effects[column] = pd.to_numeric(
            adolescent_effects[column], errors="raise"
        )
        if not np.isfinite(adolescent_effects[column]).all():
            raise AssertionError(f"Figure 3c contains non-finite values in {column}.")
    if set(adolescent_effects["adolescent_effect_rank"].astype(int)) != set(
        range(1, len(expected_countries) + 1)
    ):
        raise AssertionError("Figure 3c adolescent-effect ranks are incomplete.")
    all_under18_adolescent_median = 100.0 * float(
        adolescent_effects["all_under18_symptomatic_case_reduction"].median()
    )
    adolescent_case_median = 100.0 * float(
        adolescent_effects["adolescent_case_reduction"].median()
    )
    if (
        abs(all_under18_adolescent_median) >= 0.05
        or abs(adolescent_case_median) >= 0.05
        or int(
            adolescent_effects["all_under18_symptomatic_case_reduction"].lt(0).sum()
        )
        < 3
        or int(adolescent_effects["adolescent_case_reduction"].lt(0).sum()) < 3
    ):
        raise AssertionError(
            "Figure 3c no longer supports approximately zero medians with several "
            "slightly unfavourable profile contrasts."
        )

    def adolescent_profile(country: str) -> pd.Series:
        rows = adolescent_effects.loc[
            adolescent_effects["country"].astype(str).eq(country)
        ]
        if len(rows) != 1:
            raise AssertionError(f"Incomplete Figure 3c result for {country}.")
        return rows.iloc[0]

    thailand_adolescent = adolescent_profile("Thailand")
    brazil_adolescent = adolescent_profile("Brazil")
    if (
        int(thailand_adolescent["adolescent_effect_rank"]) != 1
        or int(brazil_adolescent["adolescent_effect_rank"]) != 2
    ):
        raise AssertionError("Figure 3c Thailand/Brazil adolescent ranks changed.")
    adolescent_tokens = [
        "median reductions were approximately zero across all nine profiles",
        "the Thailand and Brazil profiles had all-<18/adolescent-case reductions of "
        + _lancet_decimal(
            100.0
            * thailand_adolescent["all_under18_symptomatic_case_reduction"],
            1,
        )
        + "%/"
        + _lancet_decimal(
            100.0 * thailand_adolescent["adolescent_case_reduction"], 1
        )
        + "% and "
        + _lancet_decimal(
            100.0 * brazil_adolescent["all_under18_symptomatic_case_reduction"],
            1,
        )
        + "%/"
        + _lancet_decimal(
            100.0 * brazil_adolescent["adolescent_case_reduction"], 1
        )
        + "%, respectively",
        "whereas several profiles had negligible or slightly unfavourable "
        "finite-horizon contrasts",
    ]
    _require_text_tokens(
        results_text,
        gap_tokens + adolescent_tokens,
        label="Main-manuscript Figure 3b/c results",
    )

    management = read_table(
        project_path(
            "outputs", "tables", "figure4a_resistance_guided_vs_timeliness.csv"
        )
    )
    vaccine_targets = read_table(
        project_path(
            "outputs", "tables", "figure4b_vaccine_setting_residual_index.csv"
        )
    )

    def management_rows(outcome: str) -> pd.DataFrame:
        rows = management.loc[management["outcome"].astype(str).eq(outcome)].copy()
        if len(rows) != len(priorities):
            raise AssertionError(f"Incomplete Figure 4a results for {outcome}.")
        return rows

    def management_stat(outcome: str, column: str) -> float:
        values = pd.to_numeric(management_rows(outcome)[column], errors="raise").unique()
        if len(values) != 1:
            raise AssertionError(f"Figure 4a summary field {column} varies within {outcome}.")
        return float(values[0])

    def vaccine_stat(setting: str, outcome: str, column: str) -> float:
        rows = vaccine_targets.loc[
            vaccine_targets["vaccine_setting"].astype(str).eq(setting)
            & vaccine_targets["outcome"].astype(str).eq(outcome)
        ]
        if len(rows) != len(priorities):
            raise AssertionError(
                f"Incomplete Figure 4b results for {setting}, {outcome}."
            )
        values = pd.to_numeric(rows[column], errors="raise").unique()
        if len(values) != 1:
            raise AssertionError(
                f"Figure 4b summary field {column} varies within {setting}, {outcome}."
            )
        return float(values[0])

    all_under18_guided = management_stat(
        "All <18 cases", "profiles_guided_lower"
    )
    child_guided = management_stat("Children cases", "profiles_guided_lower")
    adolescent_guided = management_stat(
        "Adolescent cases", "profiles_guided_lower"
    )
    infant_guided = management_stat("Infant cases", "profiles_guided_lower")
    infant_hospital_guided = management_stat(
        "Infant hospitalisations", "profiles_guided_lower"
    )
    infant_death_guided = management_stat(
        "Infant deaths", "profiles_guided_lower"
    )
    resistant_guided = management_stat(
        "Resistant infections", "profiles_guided_lower"
    )

    no_vaccine_outcomes = (
        "All <18 cases",
        "Infant hospitalisations",
        "Adolescent cases",
        "Resistant infections",
    )
    target_settings = (
        "infection_blocking",
        "transmission_blocking",
        "next_generation",
    )
    target_outcomes = (
        "All <18 cases",
        "Infant hospitalisations",
        "Adolescent cases",
    )

    no_vaccine_medians = {
        outcome: 100.0
        * vaccine_stat("no_vaccine", outcome, "median_residual_index")
        for outcome in no_vaccine_outcomes
    }
    target_medians = {
        (setting, outcome): 100.0
        * vaccine_stat(setting, outcome, "median_residual_index")
        for setting in target_settings
        for outcome in target_outcomes
    }
    adolescent_above = {
        setting: int(
            vaccine_stat(setting, "Adolescent cases", "profiles_above_current")
        )
        for setting in target_settings
    }
    all_under18_above = {
        setting: int(
            vaccine_stat(setting, "All <18 cases", "profiles_above_current")
        )
        for setting in target_settings
    }
    if any(all_under18_above.values()):
        raise AssertionError(
            "Figure 4b future targets unexpectedly exceed current all-<18 burden."
        )

    mechanism_tokens = [
        "lower all-<18, child, and adolescent case indices in "
        + _small_number_word(int(all_under18_guided))
        + ", "
        + _small_number_word(int(child_guided))
        + ", and "
        + _small_number_word(int(adolescent_guided))
        + " of nine profiles, respectively",
        "lower infant-case and infant-hospitalisation indices in only "
        + _small_number_word(int(infant_guided)),
        "Infant-death and all-age resistant-infection indices were lower in "
        + _small_number_word(int(infant_death_guided))
        + " and all nine profiles, respectively",
        "Cross-profile median guided burdens were "
        + _lancet_decimal(
            100.0 * management_stat("All <18 cases", "median_residual_index"), 1
        )
        + "% of routine for all-<18 cases and "
        + _lancet_decimal(
            100.0
            * management_stat("Resistant infections", "median_residual_index"),
            1,
        )
        + "% for resistant infections, compared with "
        + _lancet_decimal(
            100.0 * management_stat("Infant cases", "median_residual_index"), 1
        )
        + "% for infant cases and "
        + _lancet_decimal(
            100.0
            * management_stat(
                "Infant hospitalisations", "median_residual_index"
            ),
            1,
        )
        + "% for infant hospitalisations",
        "Under the immediate no-vaccine counterfactual, median residual burdens were "
        + _lancet_decimal(no_vaccine_medians["All <18 cases"], 1)
        + "% for all-<18 cases, "
        + _lancet_decimal(no_vaccine_medians["Infant hospitalisations"], 1)
        + "% for infant hospitalisations, "
        + _lancet_decimal(no_vaccine_medians["Adolescent cases"], 1)
        + "% for adolescent cases, and "
        + _lancet_decimal(no_vaccine_medians["Resistant infections"], 1)
        + "% for all-age resistant infections",
        "median residual all-<18 burdens were "
        + _lancet_decimal(
            target_medians[("infection_blocking", "All <18 cases")], 1
        )
        + "% under infection blocking, "
        + _lancet_decimal(
            target_medians[("transmission_blocking", "All <18 cases")], 1
        )
        + "% under transmission blocking, and "
        + _lancet_decimal(
            target_medians[("next_generation", "All <18 cases")], 1
        )
        + "% under the high-blocking target",
        "corresponding infant-hospitalisation residuals were "
        + _lancet_decimal(
            target_medians[
                ("infection_blocking", "Infant hospitalisations")
            ],
            1,
        )
        + "%, "
        + _lancet_decimal(
            target_medians[
                ("transmission_blocking", "Infant hospitalisations")
            ],
            1,
        )
        + "%, and "
        + _lancet_decimal(
            target_medians[("next_generation", "Infant hospitalisations")], 1
        )
        + "%",
        "Median adolescent-case residuals were "
        + _lancet_decimal(
            target_medians[("infection_blocking", "Adolescent cases")], 1
        )
        + "%, "
        + _lancet_decimal(
            target_medians[("transmission_blocking", "Adolescent cases")], 1
        )
        + "%, and "
        + _lancet_decimal(
            target_medians[("next_generation", "Adolescent cases")], 1
        )
        + "%",
        "adolescent burden nevertheless exceeded current practice in "
        + _small_number_word(adolescent_above["infection_blocking"])
        + ", "
        + _small_number_word(adolescent_above["transmission_blocking"])
        + ", and "
        + _small_number_word(adolescent_above["next_generation"])
        + " of nine profiles",
        "whereas all-<18 burden was below current practice in every profile under all three targets",
    ]
    if not (
        int(all_under18_guided) == 5
        and int(child_guided) == 5
        and int(adolescent_guided) == 6
        and int(infant_guided) == int(infant_hospital_guided) == 3
        and int(infant_death_guided) == 8
        and int(resistant_guided) == 9
    ):
        raise AssertionError("Figure 4a endpoint-direction counts changed unexpectedly.")
    _require_text_tokens(
        results_text, mechanism_tokens, label="Main-manuscript mechanism results"
    )

    profile_labels = {"United Kingdom": "UK", "United States": "USA"}
    expected_rows = []
    for row in priorities.itertuples(index=False):
        effect_interval = (
            _lancet_decimal(row.cases_averted_per_100k_under18, 1)
            + " ("
            + _table_interval_value(row.cases_averted_per_100k_under18_q025)
            + "–"
            + _table_interval_value(row.cases_averted_per_100k_under18_q975)
            + ")"
        )
        margin_interval = (
            _lancet_decimal(row.runner_up_excess_cases_per_100k_under18, 1)
            + " ("
            + _table_interval_value(
                row.runner_up_excess_cases_per_100k_under18_q025
            )
            + "–"
            + _table_interval_value(
                row.runner_up_excess_cases_per_100k_under18_q975
            )
            + ")"
        )
        expected_rows.append(
            "| "
            + profile_labels.get(str(row.programme_profile), str(row.programme_profile))
            + " | "
            + _lancet_decimal(row.current_practice_cases_per_100k_under18, 1)
            + " | "
            + str(row.lowest_burden_programme_only_strategy)
            + " | "
            + effect_interval
            + " | "
            + _lancet_decimal(row.reduction_percent, 1)
            + " | "
            + str(row.second_ranked_programme_only_strategy)
            + " | "
            + margin_interval
            + " |"
        )
    _require_text_tokens(table_text, expected_rows, label="Main-manuscript Table 1")
    _require_text_tokens(
        table_text,
        ["paired full-refit parametric-bootstrap 95% estimation confidence intervals"],
        label="Main-manuscript Table 1 footnote",
    )

    summary_tokens = [
        f"Across {len(prediction_intervals)} held-out intervals",
        "country-balanced empirical coverage of nominal 95% intervals was "
        + _lancet_decimal(prediction_gate["primary_predictive_95_coverage"], 3),
        "mean absolute log1p error was "
        + _lancet_decimal(overall_prediction["model_mean_absolute_log1p_error"], 3),
        "Mean log score was "
        + _lancet_decimal(overall_prediction["model_mean_log_score_per_interval"], 3),
        "Routine timeliness reduced the conditional <18 index by a median "
        + _lancet_decimal(timeliness.median(), 1)
        + "% (IQR "
        + _lancet_decimal(timeliness.quantile(0.25), 1)
        + "–"
        + _lancet_decimal(timeliness.quantile(0.75), 1)
        + "), whereas the nominal coverage-floor-only contrast changed it by "
        + _lancet_decimal(coverage_effects.median(), 1)
        + "% ("
        + _lancet_decimal(coverage_effects.quantile(0.25), 1)
        + " to "
        + _lancet_decimal(coverage_effects.quantile(0.75), 1)
        + "); timeliness produced the larger reduction in all nine profiles",
        *figure2b_headline["summary_tokens"],
        "one-year block coverage was "
        + _lancet_decimal(block_gate["model_annual_95_coverage_diagnostic"], 3),
    ]
    _require_text_tokens(summary_text, summary_tokens, label="Main-manuscript Summary")
    conference_abstract_tokens = [
        str(len(prediction_folds))
        + " country-year folds and "
        + str(len(prediction_intervals))
        + " held-out intervals",
        "Country-balanced empirical coverage of nominal 95% predictive intervals was "
        + _lancet_decimal(prediction_gate["primary_predictive_95_coverage"], 3),
        "Mean log score was "
        + _lancet_decimal(overall_prediction["model_mean_log_score_per_interval"], 3)
        + ", compared with "
        + _lancet_decimal(
            overall_prediction["seasonal_naive_mean_log_score_per_interval"], 3
        )
        + " for seasonal-naive, "
        + _lancet_decimal(
            overall_prediction["ew_recent_rate_mean_log_score_per_interval"], 3
        )
        + " for exponentially weighted recent-rate, and "
        + _lancet_decimal(
            overall_prediction["damped_log_trend_mean_log_score_per_interval"], 3
        )
        + " for damped log-trend forecasts; corresponding mean absolute log1p "
        "errors were "
        + _lancet_decimal(
            overall_prediction["model_mean_absolute_log1p_error"], 3
        )
        + ", "
        + _lancet_decimal(
            overall_prediction["seasonal_naive_mean_absolute_log1p_error"], 3
        )
        + ", "
        + _lancet_decimal(
            overall_prediction["ew_recent_rate_mean_absolute_log1p_error"], 3
        )
        + ", and "
        + _lancet_decimal(
            overall_prediction["damped_log_trend_mean_absolute_log1p_error"], 3
        ),
        "China had interval coverage of "
        + _lancet_decimal(prediction_gate["minimum_country_interval_95_coverage"], 3)
        + ", and the annual block stress test failed (joint coverage "
        + _lancet_decimal(block_gate["model_annual_95_coverage_diagnostic"], 3)
        + ")",
        "routine timeliness was lowest-burden in eight profiles and the "
        "infant-exposure package led only in China",
        "Routine timeliness reduced the conditional pooled <18 index by a median "
        + _lancet_decimal(timeliness.median(), 1)
        + "% (IQR "
        + _lancet_decimal(timeliness.quantile(0.25), 1)
        + "–"
        + _lancet_decimal(timeliness.quantile(0.75), 1)
        + "), whereas the nominal coverage-floor-only contrast changed it by "
        + _lancet_decimal(coverage_effects.median(), 1)
        + "% ("
        + _lancet_decimal(coverage_effects.quantile(0.25), 1)
        + " to "
        + _lancet_decimal(coverage_effects.quantile(0.75), 1)
        + ")",
        "Across 128 prespecified selected-input settings, the locked reference "
        "choice was retained in 67–128 settings per profile",
        "the 95th-percentile fixed-reference regret was largest in Brazil, Sweden, "
        "and Thailand",
        "resistance-guided management produced lower all-<18 burden in "
        + _small_number_word(int(all_under18_guided))
        + " of nine profiles",
    ]
    _require_text_tokens(
        conference_abstract_text,
        conference_abstract_tokens,
        label="English conference-abstract Results",
    )
    _require_text_tokens(
        figure_legend_text,
        [
            "profile-specific conditional full-refit parametric-bootstrap 95% "
            "estimation confidence intervals for all three indices",
            "These intervals quantify fitted conditional estimands, not marginal "
            "epidemic-trajectory distributions, posterior credible intervals, or "
            "future-observation prediction intervals",
        ],
        label="Main-manuscript Figure 1b legend",
    )
    _require_text_tokens(
        figure_legend_text,
        [
            "Locked relative reductions in the annualised symptomatic-case index among people younger than 18 years under a nominal coverage-floor-only contrast and routine timeliness",
            "Both contrasts use the same production-runtime current-practice denominator",
            "Fixed-reference decision fragility across 128 prespecified selected-input settings",
            "The vertical axis shows the empirical 95th percentile of fixed-reference regret",
            "These design summaries are neither probabilities, confidence intervals, nor expected regret",
            "Locked relative reduction for every profile–strategy combination",
            "brackets show paired full-refit parametric-bootstrap 95% estimation confidence intervals in percentage points",
            "the black outline marks the lowest-burden strategy at the locked reference estimate",
        ],
        label="Main-manuscript Figure 2 legend",
    )
    _validate_figure2c_reported_refit_range(
        figure_legend_text,
        supplement_text,
        observed_minimum=observed_refit_minimum,
        observed_maximum=observed_refit_maximum,
    )
    _require_text_tokens(
        supplement_text,
        [
            "2023-26; "
            + str(len(prediction_folds))
            + " country-year folds; "
            + str(len(prediction_intervals))
            + " intervals"
        ],
        label="Supplementary validation summary",
    )


def _validate_release_summary_window(stem: str) -> None:
    baseline = load_configs()["baseline"]
    expected_start = str(baseline.get("calendar", {}).get("analysis_start_date", ""))
    expected_years = (
        float(baseline["simulation"]["end_time"] - baseline["simulation"]["start_time"])
        / GREGORIAN_YEAR_DAYS
    )
    validate_run_metadata(stem)
    summary = read_table(project_path("outputs", "summaries", f"{stem}_summary.csv"))
    if summary.empty:
        raise AssertionError(f"{stem} summary is empty.")
    if "calendar_start_date" not in summary.columns:
        raise AssertionError(f"{stem} summary is missing calendar_start_date.")
    starts = set(summary["calendar_start_date"].astype(str))
    if starts != {expected_start}:
        raise AssertionError(
            f"{stem} uses calendar_start_date values {sorted(starts)}, expected {expected_start}."
        )
    years = pd.to_numeric(summary["analysis_years"], errors="coerce")
    if years.isna().any() or not np.allclose(
        years.to_numpy(dtype=float), expected_years, rtol=0.0, atol=1e-6
    ):
        observed = sorted(set(float(value) for value in years.dropna().round(6)))
        raise AssertionError(
            f"{stem} analysis_years values {observed}, expected {expected_years:.6f}."
        )


def validate_release_output_windows() -> None:
    for stem in CORE_OUTPUT_STEMS:
        summary_path = project_path("outputs", "summaries", f"{stem}_summary.csv")
        if not summary_path.exists():
            raise AssertionError(f"Missing required core summary output: {summary_path}")
        _validate_release_summary_window(stem)


def _figure2_parent_bundle_sha256(parent_digests: dict[str, str]) -> str:
    expected_parents = tuple(FIGURE2_PARENT_ARTIFACTS)
    if len(parent_digests) != len(expected_parents) or set(parent_digests) != set(
        expected_parents
    ):
        raise AssertionError(
            "Figure 2 parent bundle does not contain the four canonical parents "
            "exactly once."
        )
    normalized: list[str] = []
    for parent in expected_parents:
        digest = str(parent_digests[parent]).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise AssertionError(
                f"Figure 2 parent {parent} has an invalid SHA-256 digest."
            )
        normalized.append(f"{parent}={digest}")
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _required_metadata_text(metadata: dict, field: str, *, label: str) -> str:
    value = metadata.get(field)
    if value is None or not str(value).strip():
        raise AssertionError(f"Figure 2 {label} metadata are missing {field}.")
    return str(value)


def validate_figure2_derived_provenance(
    parent_metadata: dict[str, dict],
) -> str:
    """Bind every Figure 2 derived table to the four current parent files."""

    expected_rows: dict[str, dict[str, str]] = {}
    parent_digests: dict[str, str] = {}
    for parent, (stem, relative_path, metadata_digest_key) in (
        FIGURE2_PARENT_ARTIFACTS.items()
    ):
        if stem not in parent_metadata:
            raise AssertionError(f"Figure 2 current metadata are missing parent {stem}.")
        metadata = parent_metadata[stem]
        artifact_path = project_path(relative_path)
        if not artifact_path.exists():
            raise AssertionError(f"Missing Figure 2 parent artifact: {artifact_path}")
        observed_digest = file_sha256(artifact_path)
        if metadata_digest_key is not None:
            recorded_digests = metadata.get("output_artifact_sha256")
            if not isinstance(recorded_digests, dict):
                raise AssertionError(
                    f"Figure 2 {parent} metadata do not record output artifact hashes."
                )
            recorded_digest = str(recorded_digests.get(metadata_digest_key, ""))
            if recorded_digest != observed_digest:
                raise AssertionError(
                    f"Figure 2 {parent} parent file does not match its current metadata digest."
                )
        git = metadata.get("git")
        if not isinstance(git, dict):
            raise AssertionError(f"Figure 2 {parent} metadata are missing git provenance.")
        expected_rows[parent] = {
            "config_hash": _required_metadata_text(
                metadata, "config_hash", label=parent
            ),
            "source_code_hash": _required_metadata_text(
                metadata, "source_code_hash", label=parent
            ),
            "git_commit": _required_metadata_text(git, "commit", label=parent),
            "artifact_sha256": observed_digest,
            "generated_at_utc": _required_metadata_text(
                metadata, "generated_at_utc", label=parent
            ),
        }
        parent_digests[parent] = observed_digest

    for field in ("config_hash", "source_code_hash", "git_commit"):
        if len({row[field] for row in expected_rows.values()}) != 1:
            raise AssertionError(
                f"Figure 2 current parents do not share one {field}."
            )

    provenance_path = project_path(FIGURE2_PROVENANCE_PATH)
    if not provenance_path.exists():
        raise AssertionError(f"Missing Figure 2 derived provenance: {provenance_path}")
    provenance = read_table(provenance_path)
    required_columns = {
        "parent",
        "config_hash",
        "source_code_hash",
        "git_commit",
        "artifact_sha256",
        "generated_at_utc",
        "estimand_or_role",
        FIGURE2_PARENT_BUNDLE_COLUMN,
    }
    missing_columns = required_columns.difference(provenance.columns)
    if missing_columns:
        raise AssertionError(
            "Figure 2 derived provenance is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    expected_parent_names = set(FIGURE2_PARENT_ARTIFACTS)
    observed_parent_names = set(provenance["parent"].astype(str))
    if (
        len(provenance) != len(expected_parent_names)
        or provenance["parent"].astype(str).duplicated().any()
        or observed_parent_names != expected_parent_names
    ):
        raise AssertionError(
            "Figure 2 derived provenance must contain exactly the four canonical parents."
        )
    indexed = provenance.assign(parent=provenance["parent"].astype(str)).set_index(
        "parent"
    )
    for parent, expected in expected_rows.items():
        observed = indexed.loc[parent]
        for field, expected_value in expected.items():
            if str(observed[field]) != expected_value:
                raise AssertionError(
                    f"Figure 2 derived provenance for {parent} has stale {field}."
                )
        if not str(observed["estimand_or_role"]).strip():
            raise AssertionError(
                f"Figure 2 derived provenance for {parent} has no estimand or role."
            )

    bundle_sha256 = _figure2_parent_bundle_sha256(parent_digests)
    if set(provenance[FIGURE2_PARENT_BUNDLE_COLUMN].astype(str)) != {
        bundle_sha256
    }:
        raise AssertionError(
            "Figure 2 derived provenance has a stale four-parent bundle digest."
        )
    for relative_path in FIGURE2_DERIVED_SOURCE_TABLES:
        path = project_path(relative_path)
        if not path.exists():
            raise AssertionError(f"Missing Figure 2 derived source table: {path}")
        table = read_table(path)
        if table.empty:
            raise AssertionError(f"Figure 2 derived source table is empty: {path}")
        if FIGURE2_PARENT_BUNDLE_COLUMN not in table.columns:
            raise AssertionError(
                f"Figure 2 derived source table is not parent-bound: {path}"
            )
        if set(table[FIGURE2_PARENT_BUNDLE_COLUMN].astype(str)) != {
            bundle_sha256
        }:
            raise AssertionError(
                f"Figure 2 derived source table has a stale parent bundle: {path}"
            )
    return bundle_sha256


def validate_active_publication_outputs() -> None:
    # This semantic gate is intentionally stricter than freshness alone: the
    # Figure 2c parent must be the audited full-refit estimation-CI route.
    figure2_parent_metadata = validate_figure2_parent_metadata()
    validate_figure2_derived_provenance(figure2_parent_metadata)
    active_metadata: dict[str, dict] = {}
    for stem in ACTIVE_PUBLICATION_METADATA_STEMS:
        active_metadata[stem] = validate_run_metadata(stem)
        for relative_path in PUBLICATION_REQUIRED_TABLES.get(stem, ()):
            path = project_path(relative_path)
            if not path.exists() and not path.with_suffix(".parquet").exists():
                raise AssertionError(f"Missing required publication table for {stem}: {path}")
            if read_table(path).empty:
                raise AssertionError(f"Required publication table is empty for {stem}: {path}")
    validate_resistance_management_psa(active_metadata["resistance_management_psa"])


def _strict_boolean_series(values: pd.Series, *, label: str) -> pd.Series:
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if parsed.isna().any():
        raise AssertionError(f"{label} contains values other than true or false.")
    return parsed.astype(bool)


def validate_resistance_management_psa(metadata: dict | None = None) -> None:
    """Require the complete independent 2D management design and its consumption."""

    if metadata is None:
        metadata = validate_run_metadata("resistance_management_psa")
    configs = load_configs()
    countries = tuple(publication_country_names(configs))
    parameters = tuple(RESISTANCE_MANAGEMENT_PARAMETER_NAMES)
    strata = tuple(label for label, _ in PEP_RESTORATION_STRATA)
    expected_metadata = {
        "sample_size_requested": 128,
        "sample_seed": 20260522,
        "sample_design": RESISTANCE_MANAGEMENT_SAMPLE_DESIGN,
        "uncertainty_schema_version": RESISTANCE_MANAGEMENT_SCHEMA_VERSION,
        "analysis_horizon_years": 5,
        "comparator": "routine_timeliness",
        "programme_base_in_both_arms": "routine_timeliness",
        "contrast": (
            "routine_timeliness_plus_guided_management_vs_"
            "routine_timeliness_alone"
        ),
        "paired_across_pep_restoration_strata": True,
        "pep_restoration_probability_assigned": False,
        "figure2b_excluded": True,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise AssertionError(
                f"Resistance-management PSA metadata have the wrong {field}."
            )
    if tuple(str(value) for value in metadata.get("countries", ())) != countries:
        raise AssertionError(
            "Resistance-management PSA metadata do not contain the exact publication profiles."
        )
    if tuple(str(value) for value in metadata.get("sampled_parameters", ())) != parameters:
        raise AssertionError(
            "Resistance-management PSA metadata do not contain exactly the two locked inputs."
        )
    scopes = metadata.get("parameter_time_scopes")
    if not isinstance(scopes, dict) or scopes != {
        parameter: "prospective_implementation" for parameter in parameters
    }:
        raise AssertionError(
            "Resistance-management PSA inputs are not both prospective implementation inputs."
        )
    distributions = metadata.get("parameter_distributions")
    if (
        not isinstance(distributions, dict)
        or len(distributions) != len(parameters)
        or set(distributions) != set(parameters)
    ):
        raise AssertionError(
            "Resistance-management PSA metadata do not record the exact two distributions."
        )
    if tuple(metadata.get("deterministic_structural_strata", ())) != strata:
        raise AssertionError(
            "Resistance-management PSA metadata do not contain both deterministic PEP strata."
        )

    expected_rows = {
        "parameter_samples": 128,
        "comparator_rows": len(countries),
        "guided_rows": len(countries) * 128 * len(strata),
        "effect_samples": len(countries) * 128 * len(strata),
        "summary_rows": (len(countries) + 1) * len(strata),
        "activity_audit_rows": len(countries) * len(strata) * len(parameters),
    }
    row_counts = metadata.get("row_counts")
    if not isinstance(row_counts, dict) or any(
        int(row_counts.get(key, -1)) != expected
        for key, expected in expected_rows.items()
    ):
        raise AssertionError(
            "Resistance-management PSA metadata do not prove the complete 2D paired design."
        )

    samples = read_table(
        project_path(
            "outputs", "tables", "resistance_management_psa_parameter_samples.csv"
        )
    )
    expected_sample_columns = {
        "psa_sample_id",
        "sample_design",
        "uncertainty_schema_version",
        *parameters,
    }
    if set(samples.columns) != expected_sample_columns or len(samples) != 128:
        raise AssertionError(
            "Resistance-management PSA parameter table is not the exact 128 x 2 design."
        )
    sample_ids = pd.to_numeric(samples["psa_sample_id"], errors="raise")
    if not np.array_equal(sample_ids.to_numpy(dtype=float), np.arange(1, 129)):
        raise AssertionError(
            "Resistance-management PSA parameter IDs are not exactly 1 through 128."
        )
    if set(samples["sample_design"].astype(str)) != {
        RESISTANCE_MANAGEMENT_SAMPLE_DESIGN
    }:
        raise AssertionError("Resistance-management PSA has the wrong sample design.")
    sample_schema = pd.to_numeric(
        samples["uncertainty_schema_version"], errors="raise"
    )
    if not sample_schema.eq(RESISTANCE_MANAGEMENT_SCHEMA_VERSION).all():
        raise AssertionError("Resistance-management PSA has the wrong schema version.")
    registry_parameters = configs["parameter_distributions"][
        "resistance_management_psa"
    ]["parameters"]
    for parameter in parameters:
        values = pd.to_numeric(samples[parameter], errors="raise")
        spec = registry_parameters[parameter]
        if (
            not np.isfinite(values).all()
            or values.nunique() != 128
            or not values.between(float(spec["low"]), float(spec["high"])).all()
        ):
            raise AssertionError(
                f"Resistance-management PSA samples are invalid for {parameter}."
            )

    draws = read_table(
        project_path(
            "outputs", "tables", "resistance_management_psa_effect_samples.csv"
        )
    )
    application_columns = {
        "baseline_treatment_rate_symptomatic",
        "full_guided_treatment_rate_symptomatic",
        "applied_treatment_rate_symptomatic",
        "baseline_pep_coverage",
        "applied_pep_coverage",
        "baseline_pep_effectiveness_resistant",
        "full_guided_pep_effectiveness_resistant",
        "applied_pep_effectiveness_resistant",
    }
    comparator_columns = {
        "timeliness_total_child_adolescent_cases",
        "timeliness_annualized_child_adolescent_cases_per_100k",
        "timeliness_total_infant_cases",
        "timeliness_annualized_infant_cases_per_100k",
        "timeliness_resistant_infections",
    }
    guided_columns = {
        "guided_total_child_adolescent_cases",
        "guided_annualized_child_adolescent_cases_per_100k",
        "guided_total_infant_cases",
        "guided_annualized_infant_cases_per_100k",
        "guided_resistant_infections",
    }
    endpoint_columns = {
        "under18_cases": (
            "guided_total_child_adolescent_cases",
            "timeliness_total_child_adolescent_cases",
        ),
        "infant_cases": (
            "guided_total_infant_cases",
            "timeliness_total_infant_cases",
        ),
        "resistant_infections": (
            "guided_resistant_infections",
            "timeliness_resistant_infections",
        ),
    }
    derived_endpoint_columns = {
        column
        for endpoint in endpoint_columns
        for column in (
            f"relative_reduction_{endpoint}_vs_timeliness",
            f"guided_lower_{endpoint}",
        )
    }
    required_draw_columns = {
        "country",
        "psa_sample_id",
        "pep_restoration",
        "pep_restored",
        "sample_design",
        "uncertainty_schema_version",
        *parameters,
        *application_columns,
        *comparator_columns,
        *guided_columns,
        *derived_endpoint_columns,
    }
    missing_draw_columns = required_draw_columns.difference(draws.columns)
    if missing_draw_columns:
        raise AssertionError(
            "Resistance-management PSA effect table is missing columns: "
            + ", ".join(sorted(missing_draw_columns))
        )
    if (
        len(draws) != expected_rows["effect_samples"]
        or draws.duplicated(["country", "psa_sample_id", "pep_restoration"]).any()
        or set(draws["country"].astype(str)) != set(countries)
        or set(draws["pep_restoration"].astype(str)) != set(strata)
    ):
        raise AssertionError(
            "Resistance-management PSA effect table is not the complete profile-sample-stratum crossing."
        )
    draw_ids = pd.to_numeric(draws["psa_sample_id"], errors="raise")
    block_sizes = draws.assign(_sample_id=draw_ids).groupby(
        ["country", "pep_restoration"], dropna=False
    )["_sample_id"].agg(["size", "nunique", "min", "max"])
    if not (
        block_sizes["size"].eq(128).all()
        and block_sizes["nunique"].eq(128).all()
        and block_sizes["min"].eq(1).all()
        and block_sizes["max"].eq(128).all()
    ):
        raise AssertionError(
            "Resistance-management PSA has incomplete 128-setting structural strata."
        )
    if set(draws["sample_design"].astype(str)) != {
        RESISTANCE_MANAGEMENT_SAMPLE_DESIGN
    } or not pd.to_numeric(
        draws["uncertainty_schema_version"], errors="raise"
    ).eq(
        RESISTANCE_MANAGEMENT_SCHEMA_VERSION
    ).all():
        raise AssertionError(
            "Resistance-management PSA effect table has stale design metadata."
        )
    sample_lookup = samples.set_index("psa_sample_id")
    for parameter in parameters:
        observed = pd.to_numeric(draws[parameter], errors="raise").to_numpy()
        expected = draw_ids.map(sample_lookup[parameter]).to_numpy(dtype=float)
        if not np.allclose(observed, expected, rtol=1e-12, atol=1e-12):
            raise AssertionError(
                f"Resistance-management PSA effect rows do not consume {parameter} samples."
            )

    pep_restored = _strict_boolean_series(
        draws["pep_restored"], label="Resistance-management pep_restored"
    )
    expected_restored = draws["pep_restoration"].astype(str).map(
        {label: restored for label, restored in PEP_RESTORATION_STRATA}
    )
    if expected_restored.isna().any() or not pep_restored.equals(
        expected_restored.astype(bool)
    ):
        raise AssertionError(
            "Resistance-management PSA PEP-restoration labels and flags disagree."
        )
    uptake = pd.to_numeric(
        draws["resistance_management_uptake"], errors="raise"
    )
    reach = pd.to_numeric(
        draws["resistance_management_pep_reach_multiplier"], errors="raise"
    )
    baseline_treatment = pd.to_numeric(
        draws["baseline_treatment_rate_symptomatic"], errors="raise"
    )
    full_treatment = pd.to_numeric(
        draws["full_guided_treatment_rate_symptomatic"], errors="raise"
    )
    expected_treatment = baseline_treatment + uptake * (
        full_treatment - baseline_treatment
    )
    baseline_coverage = pd.to_numeric(draws["baseline_pep_coverage"], errors="raise")
    expected_coverage = np.minimum(1.0, baseline_coverage * reach)
    baseline_effectiveness = pd.to_numeric(
        draws["baseline_pep_effectiveness_resistant"], errors="raise"
    )
    full_effectiveness = pd.to_numeric(
        draws["full_guided_pep_effectiveness_resistant"], errors="raise"
    )
    expected_effectiveness = np.where(
        pep_restored,
        baseline_effectiveness
        + uptake * (full_effectiveness - baseline_effectiveness),
        baseline_effectiveness,
    )
    consumption_checks = (
        (
            "uptake treatment application",
            draws["applied_treatment_rate_symptomatic"],
            expected_treatment,
        ),
        ("PEP reach application", draws["applied_pep_coverage"], expected_coverage),
        (
            "uptake PEP-restoration application",
            draws["applied_pep_effectiveness_resistant"],
            expected_effectiveness,
        ),
    )
    for label, observed, expected in consumption_checks:
        if not np.allclose(
            pd.to_numeric(observed, errors="raise"),
            np.asarray(expected, dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(f"Resistance-management PSA failed {label}.")
    if any(
        draws.groupby("country", dropna=False)[column].nunique(dropna=False).max()
        != 1
        for column in comparator_columns
    ):
        raise AssertionError(
            "Resistance-management PSA did not retain one paired timeliness comparator per profile."
        )

    burden_columns = tuple(sorted(comparator_columns | guided_columns))
    numeric_burdens: dict[str, pd.Series] = {}
    for column in burden_columns:
        values = pd.to_numeric(draws[column], errors="raise")
        if not np.isfinite(values).all() or values.lt(0.0).any():
            raise AssertionError(
                f"Resistance-management PSA contains invalid raw burdens in {column}."
            )
        numeric_burdens[column] = values
    for _, comparator_column in endpoint_columns.values():
        if numeric_burdens[comparator_column].le(0.0).any():
            raise AssertionError(
                "Resistance-management PSA contains a non-positive timeliness "
                f"endpoint denominator in {comparator_column}."
            )

    for endpoint, (guided_column, comparator_column) in endpoint_columns.items():
        guided_values = numeric_burdens[guided_column]
        comparator_values = numeric_burdens[comparator_column]
        expected_relative = 1.0 - guided_values / comparator_values
        expected_lower = guided_values < comparator_values
        relative_column = f"relative_reduction_{endpoint}_vs_timeliness"
        lower_column = f"guided_lower_{endpoint}"
        observed_relative = pd.to_numeric(draws[relative_column], errors="raise")
        observed_lower = _strict_boolean_series(
            draws[lower_column],
            label=f"Resistance-management {lower_column}",
        )
        if (
            not np.isfinite(observed_relative).all()
            or not np.allclose(
                observed_relative,
                expected_relative,
                rtol=1e-12,
                atol=1e-12,
            )
            or not observed_lower.equals(expected_lower.astype(bool))
        ):
            raise AssertionError(
                "Resistance-management PSA endpoint arithmetic is invalid for "
                f"{endpoint}."
            )

    annualized_endpoint_columns = {
        "under18_cases": (
            "guided_annualized_child_adolescent_cases_per_100k",
            "timeliness_annualized_child_adolescent_cases_per_100k",
        ),
        "infant_cases": (
            "guided_annualized_infant_cases_per_100k",
            "timeliness_annualized_infant_cases_per_100k",
        ),
    }
    for endpoint, (guided_rate_column, comparator_rate_column) in (
        annualized_endpoint_columns.items()
    ):
        guided_total_column, comparator_total_column = endpoint_columns[endpoint]
        comparator_rate = numeric_burdens[comparator_rate_column]
        if comparator_rate.le(0.0).any():
            raise AssertionError(
                "Resistance-management PSA contains a non-positive annualized "
                f"timeliness denominator in {comparator_rate_column}."
            )
        relative_from_rates = (
            1.0
            - numeric_burdens[guided_rate_column]
            / comparator_rate
        )
        relative_from_totals = (
            1.0
            - numeric_burdens[guided_total_column]
            / numeric_burdens[comparator_total_column]
        )
        if not np.allclose(
            relative_from_rates,
            relative_from_totals,
            rtol=ANNUALIZED_TOTAL_CONTRAST_RTOL,
            atol=ANNUALIZED_TOTAL_CONTRAST_ATOL,
        ):
            raise AssertionError(
                "Resistance-management PSA annualized and total burden contrasts "
                f"disagree for {endpoint}."
            )

    summary = read_table(
        project_path(
            "outputs", "summaries", "resistance_management_psa_summary.csv"
        )
    )
    expected_summary_numeric_columns = {
        column
        for endpoint in endpoint_columns
        for column in (
            f"median_relative_reduction_{endpoint}_vs_timeliness",
            f"q025_relative_reduction_{endpoint}_vs_timeliness",
            f"q975_relative_reduction_{endpoint}_vs_timeliness",
            f"fraction_design_draws_guided_lower_{endpoint}",
        )
    }
    expected_summary_columns = {
        "country",
        "pep_restoration",
        "psa_samples",
        "interpretation",
        *expected_summary_numeric_columns,
    }
    if (
        set(summary.columns) != expected_summary_columns
        or len(summary) != expected_rows["summary_rows"]
        or summary.duplicated(["country", "pep_restoration"]).any()
        or set(summary["country"].astype(str))
        != set(countries) | {"All_countries_pooled"}
        or set(summary["pep_restoration"].astype(str)) != set(strata)
        or not pd.to_numeric(summary["psa_samples"], errors="raise").eq(128).all()
        or not summary["interpretation"]
        .astype(str)
        .str.contains("not a posterior probability or Figure 2b input", case=False)
        .all()
    ):
        raise AssertionError(
            "Resistance-management PSA summary is incomplete or misinterprets design frequencies."
        )

    expected_summary_rows: list[dict[str, object]] = []

    def append_expected_summary(
        country: str,
        pep_restoration: str,
        group: pd.DataFrame,
    ) -> None:
        row: dict[str, object] = {
            "country": country,
            "pep_restoration": pep_restoration,
            "psa_samples": int(group["psa_sample_id"].nunique()),
        }
        for endpoint in endpoint_columns:
            values = pd.to_numeric(
                group[f"relative_reduction_{endpoint}_vs_timeliness"],
                errors="raise",
            )
            flags = _strict_boolean_series(
                group[f"guided_lower_{endpoint}"],
                label=(
                    "Resistance-management summary source "
                    f"guided_lower_{endpoint}"
                ),
            )
            row[f"median_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.median()
            )
            row[f"q025_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.quantile(0.025)
            )
            row[f"q975_relative_reduction_{endpoint}_vs_timeliness"] = float(
                values.quantile(0.975)
            )
            row[f"fraction_design_draws_guided_lower_{endpoint}"] = float(
                flags.mean()
            )
        expected_summary_rows.append(row)

    for pep_restoration in strata:
        stratum = draws.loc[
            draws["pep_restoration"].astype(str).eq(pep_restoration)
        ]
        for country, group in stratum.groupby("country", sort=True):
            append_expected_summary(str(country), pep_restoration, group)
        append_expected_summary("All_countries_pooled", pep_restoration, stratum)

    expected_summary = pd.DataFrame(expected_summary_rows)
    observed_summary = summary.copy()
    observed_summary["country"] = observed_summary["country"].astype(str)
    observed_summary["pep_restoration"] = observed_summary[
        "pep_restoration"
    ].astype(str)
    summary_check = observed_summary.merge(
        expected_summary,
        on=["country", "pep_restoration"],
        how="outer",
        suffixes=("", "_expected"),
        validate="one_to_one",
        indicator=True,
    )
    if not summary_check["_merge"].eq("both").all():
        raise AssertionError(
            "Resistance-management PSA summary does not contain the exact "
            "country-stratum and pooled groups."
        )
    if not np.array_equal(
        pd.to_numeric(summary_check["psa_samples"], errors="raise").to_numpy(
            dtype=int
        ),
        pd.to_numeric(
            summary_check["psa_samples_expected"], errors="raise"
        ).to_numpy(dtype=int),
    ):
        raise AssertionError(
            "Resistance-management PSA summary sample counts disagree with raw effects."
        )
    for column in sorted(expected_summary_numeric_columns):
        observed = pd.to_numeric(summary_check[column], errors="raise")
        expected = pd.to_numeric(
            summary_check[f"{column}_expected"], errors="raise"
        )
        if (
            not np.isfinite(observed).all()
            or not np.allclose(observed, expected, rtol=1e-12, atol=1e-12)
        ):
            raise AssertionError(
                "Resistance-management PSA summary disagrees with raw effects for "
                f"{column}."
            )

    activity_records = metadata.get("activity_audit")
    if not isinstance(activity_records, list):
        raise AssertionError("Resistance-management PSA metadata lack an activity audit.")
    activity = pd.DataFrame(activity_records)
    required_activity_columns = {
        "country",
        "pep_restoration",
        "parameter",
        "unique_values",
        "sampled_range",
        "configured_target_range",
        "application_verified",
        "parameter_active",
    }
    if required_activity_columns.difference(activity.columns):
        raise AssertionError("Resistance-management PSA activity audit is incomplete.")
    if (
        len(activity) != expected_rows["activity_audit_rows"]
        or activity.duplicated(["country", "pep_restoration", "parameter"]).any()
        or set(activity["country"].astype(str)) != set(countries)
        or set(activity["pep_restoration"].astype(str)) != set(strata)
        or set(activity["parameter"].astype(str)) != set(parameters)
        or not pd.to_numeric(activity["unique_values"], errors="raise").eq(128).all()
        or not pd.to_numeric(activity["sampled_range"], errors="raise").gt(0.0).all()
        or not pd.to_numeric(activity["configured_target_range"], errors="raise")
        .gt(0.0)
        .all()
        or not activity["application_verified"].map(lambda value: value is True).all()
        or not activity["parameter_active"].map(lambda value: value is True).all()
    ):
        raise AssertionError(
            "Resistance-management PSA activity audit does not prove both inputs active in both strata."
        )


def validate_under18_programme_psa() -> None:
    """Require the complete six-input programme-only design for the primary endpoint."""

    configs = load_configs()
    countries = tuple(publication_country_names(configs))
    strategies = tuple(PROGRAMME_ONLY_STRATEGIES)
    samples = read_table(
        project_path("outputs", "tables", "joint_psa_under18_programme_rank_samples.csv")
    )
    expected_rows = 128 * len(countries) * len(strategies)
    if len(samples) != expected_rows:
        raise AssertionError(
            f"Primary-endpoint joint PSA has {len(samples)} rows, expected {expected_rows}."
        )
    if set(samples["country"].astype(str)) != set(countries):
        raise AssertionError("Primary-endpoint joint PSA country set does not match publication profiles.")
    if set(samples["strategy"].astype(str)) != set(strategies):
        raise AssertionError("Primary-endpoint joint PSA programme strategy set is incomplete.")
    if samples["psa_sample_id"].nunique() != 128:
        raise AssertionError("Primary-endpoint joint PSA does not contain 128 unique paired samples.")
    if samples.duplicated(["psa_sample_id", "country", "strategy"]).any():
        raise AssertionError("Primary-endpoint joint PSA contains duplicate sample-country-strategy cells.")
    group_sizes = samples.groupby(["psa_sample_id", "country"], dropna=False).size()
    if not group_sizes.eq(len(strategies)).all():
        raise AssertionError("Primary-endpoint joint PSA has incomplete within-sample strategy comparisons.")
    ranks = pd.to_numeric(samples["rank"], errors="coerce")
    if ranks.isna().any() or not ranks.between(1, len(strategies)).all():
        raise AssertionError("Primary-endpoint joint PSA ranks are missing or outside the programme set.")
    rank_blocks = samples.assign(_rank=ranks).groupby(
        ["psa_sample_id", "country"], dropna=False
    )["_rank"].agg(["nunique", "min", "max", lambda x: int(x.eq(1).sum())])
    rank_blocks.columns = ["nunique", "min", "max", "rank_one_count"]
    if not (
        rank_blocks["nunique"].eq(len(strategies)).all()
        and rank_blocks["min"].eq(1).all()
        and rank_blocks["max"].eq(len(strategies)).all()
        and rank_blocks["rank_one_count"].eq(1).all()
    ):
        raise AssertionError(
            "Primary-endpoint joint PSA does not contain one complete ranking per profile-setting block."
        )

    summary = read_table(
        project_path(
            "outputs",
            "summaries",
            "joint_psa_under18_programme_rank_acceptability_summary.csv",
        )
    )
    expected_summary_countries = set(countries) | {"All_countries_pooled"}
    if set(summary["country"].astype(str)) != expected_summary_countries:
        raise AssertionError("Primary-endpoint rank-frequency summary has the wrong country set.")
    if set(summary["strategy"].astype(str)) != set(strategies):
        raise AssertionError("Primary-endpoint rank-frequency summary has the wrong strategy set.")
    if not pd.to_numeric(summary["n_psa_samples"], errors="coerce").eq(128).all():
        raise AssertionError("Primary-endpoint rank-frequency summary is not based on 128 samples.")
    frequency = pd.to_numeric(summary["frequency_rank_1"], errors="coerce")
    if frequency.isna().any() or not frequency.between(0.0, 1.0).all():
        raise AssertionError("Primary-endpoint rank-one frequencies are invalid.")
    sums = summary.assign(_frequency=frequency).groupby("country")["_frequency"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-10):
        raise AssertionError("Primary-endpoint rank-one frequencies do not sum to one.")
    raw_rank_one = (
        samples.assign(_rank=ranks)
        .loc[lambda x: x["_rank"].eq(1)]
        .groupby(["country", "strategy"], as_index=False)
        .size()
        .rename(columns={"size": "raw_rank_one_count"})
    )
    country_summary = summary.loc[
        summary["country"].astype(str).isin(countries)
    ].copy()
    country_summary["summary_rank_one_count"] = (
        pd.to_numeric(country_summary["frequency_rank_1"], errors="raise") * 128.0
    )
    comparison = country_summary.merge(
        raw_rank_one,
        on=["country", "strategy"],
        how="left",
        validate="one_to_one",
    )
    comparison["raw_rank_one_count"] = comparison["raw_rank_one_count"].fillna(0)
    if len(comparison) != len(countries) * len(strategies) or not np.allclose(
        comparison["summary_rank_one_count"],
        comparison["raw_rank_one_count"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError(
            "Primary-endpoint rank summary is stale relative to the raw 128-setting ranks."
        )
    _validate_figure2b_interpretation_language(summary["interpretation"])


def main() -> None:
    validate_population_conservation()
    validate_baseline_outputs()
    validate_release_output_windows()
    validate_active_publication_outputs()
    validate_under18_programme_psa()
    validate_main_manuscript_key_numbers()
    print("Active publication validation checks passed.")


if __name__ == "__main__":
    main()
