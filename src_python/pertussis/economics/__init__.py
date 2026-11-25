"""Economic evaluation layer for the core2026 pertussis model.

Modules
-------
loader   : YAML loading and schema validation (core2026-severity-priors-v1,
           core2026-costs-v1).
outcomes : Age-bin to severity-group mapping, expected reported cases,
           hospitalisations, and deaths from modelled onsets.
costing  : Vaccine costs, treatment costs, DALYs (YLL + YLD) with discounting.
psa      : Probabilistic sensitivity analysis, ICER, NMB, CEAC, and
           price-threshold analysis.

Interpretation limits
---------------------
* All monetary outputs are in 2024 international dollars unless the caller
  supplies only value_usd_2024 entries, in which case USD is used throughout.
* Severity proportions are per confirmed (reported) case.  They are never
  applied directly to modelled onsets; the observation-model reporting
  probability mediates the conversion.  The ``death_ascertainment`` parameter
  scales the CFR to account for deaths not captured in surveillance.
* Countries with status ``use_calibrated_reference_risk`` in their severity
  config must be handled through the existing health_outcomes / mortality_
  extrapolation path; this package does not override that path.
"""
from .loader import load_severity_priors, load_costs
from .outcomes import (
    bin_severity_weights,
    reported_cases_from_onsets,
    apply_severity,
    REPORTING_BANDS,
    BAND_LABELS,
)
from .costing import (
    discount_factor,
    yll_per_death,
    yll_per_death_by_age_bin,
    yld_per_case,
    compute_dalys,
    vaccine_costs_from_receipts,
    treatment_costs_from_outcomes,
)
from .psa import (
    sample_entries,
    sample_cost_draws,
    compute_incremental,
    net_monetary_benefit,
    run_psa,
)

__all__ = [
    "load_severity_priors",
    "load_costs",
    "bin_severity_weights",
    "reported_cases_from_onsets",
    "apply_severity",
    "REPORTING_BANDS",
    "BAND_LABELS",
    "discount_factor",
    "yll_per_death",
    "yll_per_death_by_age_bin",
    "yld_per_case",
    "compute_dalys",
    "vaccine_costs_from_receipts",
    "treatment_costs_from_outcomes",
    "sample_entries",
    "sample_cost_draws",
    "compute_incremental",
    "net_monetary_benefit",
    "run_psa",
]
