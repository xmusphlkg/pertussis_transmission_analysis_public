"""Within-year age, reporting and clinical-risk allocation for monthly onsets.

Risks, source choice and bin-level fallback are held fixed as in the annual
estimator. Cohort age composition follows the model's linear Lexis mapping.
"""
from types import SimpleNamespace

import numpy as np

from ..grid import KIND_COHORT
from .joint_age_severity import joint_risk_weights


def clinical_risk_endpoints(grid, groups, logits, own_entries, donor_entries,
                            own_samples, donor_samples, use_own, bin_risk, **kwargs):
    """Return joint draw/bin/group weights at calendar-year start and end.

    All cohort bins after the first contain two completed-age years, with
    the older share changing linearly from 0 to 1. The first cohort bin and
    monthly/adult bins do not move within the year. Entry samples, donor
    selection, fallback risks and optional cost/YLL modifiers are the same at
    both endpoints and must be paired across strategies and months.
    """
    endpoints = []
    for older in (False, True):
        lower, upper = grid.lower_years.copy(), grid.upper_years.copy()
        for b in np.flatnonzero(grid.kind == KIND_COHORT):
            age = int(grid.cohort_age[b])
            if age == grid.cohort_first:
                continue
            lower[b], upper[b] = ((float(age), float(age + 1)) if older
                                  else (float(age - 1), float(age)))
        proxy = SimpleNamespace(n_bins=grid.n_bins, lower_years=lower, upper_years=upper)
        endpoints.append(joint_risk_weights(proxy, groups, logits, own_entries, donor_entries,
                                           own_samples, donor_samples, use_own, bin_risk, **kwargs))
    return tuple(endpoints)


def monthly_clinical_outcomes(base_onsets, year_fractions, younger_weights, older_weights):
    """Return month × draw × group outcomes from nonnegative reported-risk onsets.

    ``base_onsets`` is typical + mild_reporting_ratio * mild in each month/bin;
    endpoint weights already combine reporting probability, clinical risk,
    group allocation and any requested cost/YLL multiplier. Discount and
    strategy contrasts must be applied by the caller.
    """
    base = np.asarray(base_onsets, float)
    fractions = np.asarray(year_fractions, float)
    younger, older = np.asarray(younger_weights, float), np.asarray(older_weights, float)
    if (base.ndim != 2 or fractions.shape != (base.shape[0],) or younger.ndim != 3
            or older.shape != younger.shape or younger.shape[1] != base.shape[1]
            or 0 in younger.shape or not np.isfinite(base).all() or (base < 0).any()
            or not np.isfinite(fractions).all() or ((fractions < 0) | (fractions > 1)).any()
            or not np.isfinite(younger).all() or not np.isfinite(older).all()
            or (younger < 0).any() or (older < 0).any()):
        raise ValueError('invalid monthly onsets, year fractions or clinical weights')
    return (np.einsum('mb,nbg->mng', base * (1 - fractions[:, None]), younger, optimize=True)
            + np.einsum('mb,nbg->mng', base * fractions[:, None], older, optimize=True))
