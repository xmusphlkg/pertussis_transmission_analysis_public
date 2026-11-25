"""Observation model for the core2026 pertussis simulator.

The transmission kernel records biological onsets by model age bin.  This
package contains the deliberately separate layer that turns those onsets into
reported surveillance counts and age compositions.  Keeping this boundary
explicit means reporting parameters can be changed during inference without
rerunning the transmission simulation.
"""

from .model import (
    AGE_BANDS,
    EXCLUDED_AGE_YEARS,
    AgeCompositionObservations,
    CaseObservations,
    ObservationParameters,
    InvalidReportingProbability,
    validate_effective_reporting_probability,
    ObservationProjection,
    SourceAgeYear,
    dirichlet_multinomial_nll,
    expected_source_group_counts,
    model_band_matrix,
    load_age_compositions,
    load_case_observations,
    logit_random_walk_nll,
    negative_binomial_nll,
    project_reported_cases,
    reporting_probabilities,
    source_age_years,
)

__all__ = [
    "AGE_BANDS",
    "EXCLUDED_AGE_YEARS",
    "AgeCompositionObservations",
    "CaseObservations",
    "ObservationParameters",
    "InvalidReportingProbability",
    "validate_effective_reporting_probability",
    "ObservationProjection",
    "SourceAgeYear",
    "dirichlet_multinomial_nll",
    "expected_source_group_counts",
    "model_band_matrix",
    "load_age_compositions",
    "load_case_observations",
    "logit_random_walk_nll",
    "negative_binomial_nll",
    "project_reported_cases",
    "reporting_probabilities",
    "source_age_years",
]
