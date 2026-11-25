"""Age-resolved onset export and conditional mortality post-processing.

Risks refer to modelled typical/mild onsets, never to notified cases. Mortality
is attributed to the onset cohort and does not feed back into transmission.
No country fatality probabilities or vaccination-specific severity effects are
supplied by this module.
"""
import numpy as np
import pandas as pd

from ..states import ACC_TYPICAL, ACC_MILD, ACC_INFECTIONS, NA

RISK_APPLICATIONS = ('matched', 'extrapolated')


def annual_age_onsets(result, grid, calendar):
    """Export all model bins, retaining monthly infant age resolution.

    Cohort/adult labels are model bins, not exact completed-age bands.
    January boundary records are required; counters remain in onset-age bins.
    """
    days = np.asarray(result.record_days)
    records = np.asarray(result.acc_records)
    if (len(days) < 2 or records.shape != (len(days), grid.n_bins, NA)
            or not np.isfinite(records).all()):
        raise ValueError('complete age-specific accumulator records are required')
    years = []
    for day in days:
        date = calendar.dates[int(day)]
        if str(date)[5:] != '01-01':
            raise ValueError('records must be January boundaries')
        years.append(int(str(date)[:4]))
    if not np.all(np.diff(years) == 1):
        raise ValueError('records must cover consecutive years')
    increments = np.diff(records, axis=0)
    if np.any(increments < 0):
        raise ValueError('cumulative counts must not decrease')
    rows = []
    for i, year in enumerate(years[:-1]):
        for b, label in enumerate(grid.labels()):
            typical, mild = increments[i, b, ACC_TYPICAL], increments[i, b, ACC_MILD]
            rows.append(dict(year=year, age_bin=label, bin_index=b,
                age_bin_kind=int(grid.kind[b]), typical_onsets=typical,
                mild_onsets=mild, all_onsets=typical+mild,
                infections=increments[i, b, ACC_INFECTIONS]))
    return pd.DataFrame(rows)


INFANT_BINS = 12
VALUE_COLUMNS = ('typical_onsets', 'mild_onsets', 'all_onsets', 'infections')
AGE_IDENTITY = ('age_bin', 'bin_index', 'age_bin_kind')


def infant_risk_age_label(start, end):
    """Label a risk age group by its own monthly bounds, never a shared name."""
    return f'm{int(start):02d}_m{int(end)-1:02d}'


def infant_risk_age_groups(onsets, *, early_age_end_months, keys=None):
    """Aggregate the 12 monthly infant bins into one risk source's own groups.

    A borrowed risk set resolves infant age only at its own published boundary
    (2 months for Australia, 3 for England), so each source is aggregated
    separately and never pooled into a single reconciled split. Adult bins have
    no borrowed risk and must be excluded by the caller rather than zero-filled.
    All 12 monthly bins are required per key group; the output keeps the group
    bounds and sets bin_index to the group index so that infant-only
    comparisons continue to select bin_index < 12.
    """
    if (not np.isfinite(early_age_end_months)
            or int(early_age_end_months) != early_age_end_months
            or not 1 <= int(early_age_end_months) <= INFANT_BINS-1):
        raise ValueError('early age endpoint must be an integer month from 1 to 11')
    cut = int(early_age_end_months)
    required = {'bin_index', 'typical_onsets', 'mild_onsets'}
    if required-set(onsets.columns):
        raise ValueError('monthly infant bins with typical and mild onsets are required')
    if onsets.empty or onsets.bin_index.isna().any():
        raise ValueError('nonmissing infant onset rows are required')
    if onsets.bin_index.ge(INFANT_BINS).any():
        raise ValueError('subset to infant bins; adult borrowed risks are not available')
    if keys is None:
        keys = [c for c in onsets.columns
                if c not in VALUE_COLUMNS+AGE_IDENTITY]
    if not keys:
        raise ValueError('explicit grouping keys are required')
    if onsets[keys].isna().any().any():
        raise ValueError('onset grouping keys cannot be missing')
    counts = onsets[['typical_onsets', 'mild_onsets']].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('onsets must be finite and nonnegative')
    rows = []
    for key, frame in onsets.groupby(list(keys), dropna=False, sort=False):
        if sorted(frame.bin_index.tolist()) != list(range(INFANT_BINS)):
            raise ValueError('all 12 infant months are required once per key group')
        meta = dict(zip(list(keys), key if isinstance(key, tuple) else (key,)))
        for index, (lo, hi) in enumerate([(0, cut), (cut, INFANT_BINS)]):
            part = frame[frame.bin_index.ge(lo) & frame.bin_index.lt(hi)]
            rows.append(dict(meta, bin_index=index, risk_age=index,
                age_bin=infant_risk_age_label(lo, hi), age_start_months=lo,
                age_end_months_exclusive=hi,
                typical_onsets=float(part.typical_onsets.sum()),
                mild_onsets=float(part.mild_onsets.sum()),
                source_infant_months=int(hi-lo)))
    out = pd.DataFrame(rows)
    out['all_onsets'] = out.typical_onsets+out.mild_onsets
    return out


def project_deaths(onsets, risks, *, risk_application=None):
    """Apply one explicit risk draw to age/year/country onset rows.

    Call separately for each paired risk draw. Same risks apply to every arm.
    A missing age/year probability raises, including for zero-onset rows.
    For infant-only estimates, explicitly subset onsets before calling; do not
    encode missing adult risks as zeros. A risk source must establish the
    mapping to each model bin before producing this input table.

    `risk_application` is declared either as a single-valued risk column or as
    this argument. 'matched' requires locally calibrated risks, that is
    geographic_scope equal to the model profile. 'extrapolated' borrows another
    country's calibrated risks and additionally requires `risk_source_profile`
    naming that source, which must differ from the profile and agree with
    geographic_scope. Both labels are retained on every output row so that a
    borrowed-risk death count can never be read as local evidence.
    """
    keys = ['profile', 'year', 'age_bin']
    probabilities = ['p_death_given_typical', 'p_death_given_mild']
    metadata = ['source', 'risk_set_id', 'denominator', 'geographic_scope']
    if set(keys+probabilities+metadata)-set(risks.columns):
        raise ValueError('risk table requires probabilities, source and denominator metadata')
    if 'risk_application' in risks.columns:
        if risk_application is not None:
            raise ValueError('declare risk application once, as a column or an argument')
        declared = set(risks.risk_application.dropna().astype(str).str.strip())
        if len(declared) != 1 or len(risks.risk_application.dropna()) != len(risks):
            raise ValueError('one nonmissing risk application label per risk table')
        application = declared.pop()
    else:
        application = 'matched' if risk_application is None else str(risk_application)
    if application not in RISK_APPLICATIONS:
        raise ValueError(f'risk application must be one of {RISK_APPLICATIONS}')
    if risks.empty or risks[keys].isna().any().any() or risks.duplicated(keys).any():
        raise ValueError('risk table must contain unique nonmissing age/year/profile keys')
    if risks.risk_set_id.nunique() != 1:
        raise ValueError('apply one risk draw at a time, paired across strategies')
    if (risks[metadata].isna().any().any() or
            risks[metadata].astype(str).apply(lambda col: col.str.strip().eq('')).any().any()):
        raise ValueError('risk provenance cannot be missing')
    if not risks.denominator.eq('modelled_course_specific_onsets').all():
        raise ValueError('notified-case fatality cannot be applied to modelled onsets')
    if application == 'matched':
        if not risks.geographic_scope.eq(risks.profile).all():
            raise ValueError('risk geography must match the model profile')
    else:
        if 'risk_source_profile' not in risks.columns:
            raise ValueError('extrapolated risks must name their risk_source_profile')
        source = risks.risk_source_profile
        if (source.isna().any() or source.astype(str).str.strip().eq('').any()
                or source.nunique() != 1):
            raise ValueError('extrapolated risks require one nonmissing risk source country')
        if not source.eq(risks.geographic_scope).all():
            raise ValueError('risk source country must equal the calibrated risk geography')
        if source.eq(risks.profile).any():
            raise ValueError('extrapolation requires a risk source other than the model profile')
    p = risks[probabilities].to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('death probabilities must be finite and in [0, 1]')
    if set(keys+['strategy_id', 'typical_onsets', 'mild_onsets'])-set(onsets.columns):
        raise ValueError('age-specific typical and mild onsets with strategy keys are required')
    if onsets[keys].isna().any().any():
        raise ValueError('onset keys cannot be missing')
    counts = onsets[['typical_onsets', 'mild_onsets']].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('onsets must be finite and nonnegative')
    carried = keys+probabilities+metadata
    if application != 'matched':
        carried = carried+['risk_source_profile']
    out = onsets.merge(risks[carried], on=keys,
                       how='left', validate='many_to_one', indicator=True)
    if not out['_merge'].eq('both').all():
        raise ValueError('missing mortality risks; missing is not zero')
    out['expected_deaths'] = (out.typical_onsets*out.p_death_given_typical
                              + out.mild_onsets*out.p_death_given_mild)
    out['death_time_basis'] = 'onset_cohort'
    out['risk_application'] = application
    if application == 'matched':
        out['risk_source_profile'] = out.geographic_scope
    return out.drop(columns='_merge')
