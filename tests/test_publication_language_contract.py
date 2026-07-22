from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from manuscript_notes import validate_publication_outputs as validator


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_figure2b_interpretation_accepts_selected_input_design_frequency() -> None:
    interpretation = pd.Series(
        [
            "Selected-input deterministic design frequency; non-inferential design summary.",
            "Selected input design-frequency across prespecified settings.",
        ]
    )

    validator._validate_figure2b_interpretation_language(interpretation)


@pytest.mark.parametrize(
    "interpretation",
    [
        "Selected-parameter deterministic design frequency.",
        "Selected-input deterministic frequency.",
        "Deterministic design frequency across prespecified settings.",
        "Selected-input design frequency; not a posterior probability.",
        "Selected-input design frequencies are not selection probabilities.",
        "Selected-input deterministic design frequency with 95% confidence intervals.",
        "",
    ],
)
def test_figure2b_interpretation_rejects_ambiguous_language(
    interpretation: str,
) -> None:
    with pytest.raises(AssertionError, match="selected-input design-frequency"):
        validator._validate_figure2b_interpretation_language(
            pd.Series([interpretation])
        )


def _figure2b_headline_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": "Profile_E",
                "prespecified_setting_count": 200,
                "reference_choice_retained_count": 200,
                "regret_percentage_points_q95": 0.0,
            },
            {
                "country": "Profile_C",
                "prespecified_setting_count": 200,
                "reference_choice_retained_count": 150,
                "regret_percentage_points_q95": 20.04,
            },
            {
                "country": "Profile_A",
                "prespecified_setting_count": 200,
                "reference_choice_retained_count": 190,
                "regret_percentage_points_q95": 4.04,
            },
            {
                "country": "Profile_D",
                "prespecified_setting_count": 200,
                "reference_choice_retained_count": 120,
                "regret_percentage_points_q95": 25.04,
            },
            {
                "country": "Profile_B",
                "prespecified_setting_count": 200,
                "reference_choice_retained_count": 80,
                "regret_percentage_points_q95": 2.04,
            },
        ]
    )


def test_figure2b_headline_tokens_are_fully_data_derived() -> None:
    headline = validator._figure2b_headline_tokens(_figure2b_headline_fixture())

    assert headline["setting_count"] == 200
    assert headline["retained_min"] == 80
    assert headline["retained_max"] == 200
    assert headline["results_tokens"] == [
        "Across the 200 prespecified selected-input settings, the locked reference "
        "choice remained lowest-burden in 80–200 settings per profile",
        "Retention was lowest in Profile_B (80/200) and Profile_D (120/200)",
        "The 95th-percentile fixed-reference regret was largest in Profile_D "
        "(25·0 percentage points of current-practice burden), Profile_C (20·0), "
        "and Profile_A (4·0; figure 2B)",
        "Retention frequency and high-tail consequence therefore supplied different "
        "information: Profile_C retained its reference choice in 150/200 settings "
        "but had the second-largest 95th-percentile regret, whereas Profile_B had "
        "the lowest retention but a lower 95th-percentile regret of 2·0",
        "These are deterministic summaries of the prespecified design, not selection "
        "probabilities, confidence intervals, or expected regret",
    ]
    assert headline["summary_tokens"] == [
        "The locked reference choice was retained in 80–200 of 200 selected-input "
        "settings per profile",
        "The 95th-percentile fixed-reference regret was largest in Profile_D "
        "(25·0 percentage points of current-practice burden), Profile_C (20·0), "
        "and Profile_A (4·0)",
        "these are deterministic design summaries, not probabilities or expected regret",
    ]


def test_figure2b_headline_tokens_are_independent_of_row_order() -> None:
    frame = _figure2b_headline_fixture()
    expected = validator._figure2b_headline_tokens(frame)
    observed = validator._figure2b_headline_tokens(
        frame.sample(frac=1.0, random_state=20260721).reset_index(drop=True)
    )

    assert observed == expected


def test_figure2b_headline_requires_a_meaningful_retention_consequence_contrast(
) -> None:
    frame = _figure2b_headline_fixture()
    frame.loc[
        frame["country"].eq("Profile_C"), "reference_choice_retained_count"
    ] = 70

    with pytest.raises(AssertionError, match="retention-versus-consequence contrast"):
        validator._figure2b_headline_tokens(frame)


def test_figure2c_validator_accepts_estimation_confidence_interval_language() -> None:
    validator._validate_figure2c_estimation_ci_language(
        pd.Series(
            [
                "paired full-refit parametric-bootstrap 95% estimation confidence interval",
                "Paired full refit parametric bootstrap 95% estimation confidence intervals",
            ]
        ),
        pd.Series(
            ["percentile_parametric_bootstrap", "percentile_parametric_bootstrap"]
        ),
    )


def _figure2c_refit_fixture(
    successful_by_country: dict[str, int],
    *,
    requested: int = 1200,
    minimum: int = 1000,
) -> tuple[pd.DataFrame, dict]:
    effects = pd.DataFrame(
        [
            {
                "country": country,
                "strategy": f"strategy_{strategy}",
                "successful_bootstrap_replicates": successful,
            }
            for country, successful in successful_by_country.items()
            for strategy in range(6)
        ]
    )
    metadata = {
        "countries": list(successful_by_country),
        "replicates_requested_per_country": requested,
        "minimum_successful_replicates": minimum,
        "successful_replicates_by_country": successful_by_country,
    }
    return effects, metadata


def test_figure2c_successful_refit_gate_uses_metadata_not_observed_hardcode() -> None:
    effects, metadata = _figure2c_refit_fixture(
        {"Country_A": 1007, "Country_B": 1193}
    )

    observed_range = validator._validate_figure2c_successful_refit_gate(
        effects,
        metadata,
        expected_countries={"Country_A", "Country_B"},
    )

    assert observed_range == (1007, 1193)


def test_figure2c_successful_refit_gate_rejects_country_below_metadata_minimum(
) -> None:
    effects, metadata = _figure2c_refit_fixture(
        {"Country_A": 999, "Country_B": 1193}
    )

    with pytest.raises(AssertionError, match="country successful-refit gate"):
        validator._validate_figure2c_successful_refit_gate(
            effects,
            metadata,
            expected_countries={"Country_A", "Country_B"},
        )


def test_figure2c_successful_refit_gate_rejects_source_metadata_mismatch() -> None:
    effects, metadata = _figure2c_refit_fixture(
        {"Country_A": 1007, "Country_B": 1193}
    )
    effects.loc[
        effects["country"].eq("Country_A"), "successful_bootstrap_replicates"
    ] = 1008

    with pytest.raises(AssertionError, match="do not match bootstrap metadata"):
        validator._validate_figure2c_successful_refit_gate(
            effects,
            metadata,
            expected_countries={"Country_A", "Country_B"},
        )


def _figure2c_prose(refit_range: str) -> tuple[str, str]:
    legend = (
        "Cell brackets show paired full-refit parametric-bootstrap 95% estimation "
        "confidence intervals in percentage points. "
        f"The intervals used {refit_range} successful refits per profile and are "
        "estimation confidence intervals for fitted scenario contrasts, not "
        "posterior credible intervals or future-observation prediction intervals."
    )
    supplement = (
        "The cell brackets are paired full-refit parametric-bootstrap 95% "
        "estimation confidence intervals calculated from the 2·5th and 97·5th "
        f"percentiles of {refit_range} successful refits per profile. These panel c "
        "intervals quantify uncertainty in the fitted scenario contrasts and are "
        "neither posterior credible intervals nor future-outbreak prediction intervals."
    )
    return legend, supplement


def test_figure2c_prose_reports_dynamic_observed_refit_range() -> None:
    legend, supplement = _figure2c_prose("1007–1193")

    validator._validate_figure2c_reported_refit_range(
        legend,
        supplement,
        observed_minimum=1007,
        observed_maximum=1193,
    )


def test_figure2c_prose_rejects_stale_refit_range() -> None:
    legend, supplement = _figure2c_prose("1007–1193")
    stale_legend = legend.replace("1007–1193", "1007–1192")

    with pytest.raises(AssertionError, match="observed successful-refit range"):
        validator._validate_figure2c_reported_refit_range(
            stale_legend,
            supplement,
            observed_minimum=1007,
            observed_maximum=1193,
        )


def test_figure2c_prose_keeps_non_bayesian_estimation_ci_contract() -> None:
    legend, supplement = _figure2c_prose("1007–1193")
    ambiguous_supplement = supplement.replace(
        "These panel c intervals quantify uncertainty in the fitted scenario "
        "contrasts and are neither posterior credible intervals nor future-outbreak "
        "prediction intervals.",
        "These panel c intervals quantify uncertainty.",
    )

    with pytest.raises(AssertionError, match="estimation-CI contract"):
        validator._validate_figure2c_reported_refit_range(
            legend,
            ambiguous_supplement,
            observed_minimum=1007,
            observed_maximum=1193,
        )


@pytest.mark.parametrize(
    ("interval_type", "interval_method"),
    [
        (
            "full-refit parametric-bootstrap 95% estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired parametric-bootstrap 95% estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit 95% estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 90% estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 95% confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 95% posterior estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 95% credible estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 95% prediction estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 95% predictive estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 95% Bayesian estimation confidence interval",
            "percentile_parametric_bootstrap",
        ),
        (
            "paired full-refit parametric-bootstrap 95% estimation confidence interval",
            "bayesian_posterior_quantile",
        ),
    ],
)
def test_figure2c_validator_rejects_non_estimation_or_forbidden_interval_language(
    interval_type: str, interval_method: str
) -> None:
    with pytest.raises(AssertionError, match="estimation confidence intervals"):
        validator._validate_figure2c_estimation_ci_language(
            pd.Series([interval_type]), pd.Series([interval_method])
        )


def test_figure2_r_sources_emit_locked_uncertainty_and_design_language() -> None:
    data_source = _read("scripts_R/figures/figure_2/data.R")
    source_data = _read("scripts_R/figures/figure_2/source_data.R")
    compact_data_source = " ".join(
        data_source.replace('"', "").replace(",", "").split()
    )

    assert (
        "paired full-refit parametric-bootstrap 95% estimation confidence interval"
        in compact_data_source
    )
    assert "selected-input deterministic design-frequency" in data_source.lower()
    assert source_data.lower().count("selected-input deterministic design-frequency") >= 3
    assert "probabilit" not in source_data.lower()


def test_active_sources_use_selected_input_language() -> None:
    source_paths = (
        "manuscript/submission_ready/main_manuscript.md",
        "manuscript/submission_ready/supplementary_material.md",
        "manuscript/appendix_templates/supplementary_methods.md",
        "manuscript/appendix_templates/supplementary_tables.md",
        "manuscript_notes/pre_submission_reviewer_issue_closure_20260714.md",
        "manuscript_notes/final_pre_submission_reviewer_reassessment_20260714.md",
    )

    for relative_path in source_paths:
        source_text = _read(relative_path).lower()
        for forbidden in ("selected-parameter", "selected parameter"):
            assert forbidden not in source_text, relative_path


def test_resistance_management_is_an_independent_two_input_analysis() -> None:
    methods_phrase = (
        "An independent five-year, two-input selected-input analysis sampled "
        "resistance-management uptake and guided-pathway PEP reach"
    )
    for relative_path in (
        "manuscript/appendix_templates/supplementary_methods.md",
        "manuscript/submission_ready/supplementary_material.md",
    ):
        assert methods_phrase in _read(relative_path), relative_path

    table_phrase = (
        "the independent two-input resistance-management analysis of uptake and "
        "guided-pathway PEP reach"
    )
    for relative_path in (
        "manuscript/appendix_templates/supplementary_tables.md",
        "manuscript/submission_ready/supplementary_material.md",
    ):
        assert table_phrase in _read(relative_path), relative_path


def test_resistance_strata_use_not_restored_naming() -> None:
    for relative_path in (
        "ANALYSIS_PROTOCOL.md",
        "manuscript/figure_source_data_manifest.md",
    ):
        source_text = _read(relative_path)
        assert "restored and not-restored" in source_text, relative_path
        assert "restored and non-restored" not in source_text, relative_path


def test_figure2c_sources_report_estimation_confidence_intervals() -> None:
    interval_phrase = (
        "paired full-refit parametric-bootstrap 95% estimation confidence intervals"
    )
    for relative_path in (
        "manuscript/submission_ready/main_manuscript.md",
        "manuscript/submission_ready/cover_letter.md",
        "manuscript/figure_source_data_manifest.md",
        "manuscript/appendix_templates/supplementary_methods.md",
        "manuscript/submission_ready/supplementary_material.md",
        "public_release/SOURCE_DATA_MANIFEST.md",
    ):
        assert interval_phrase in _read(relative_path), relative_path


def test_table1_footnote_reports_estimation_confidence_intervals() -> None:
    table_text = validator._markdown_section(
        _read("manuscript/submission_ready/main_manuscript.md"),
        "Tables",
        "Figure legends",
    )
    assert (
        "paired full-refit parametric-bootstrap 95% estimation confidence intervals"
        in table_text
    )


def test_main_manuscript_results_match_current_canonical_tables() -> None:
    validator.validate_main_manuscript_key_numbers()
