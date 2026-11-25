"""Reject observation variants not represented by the projection evaluator."""
from pathlib import Path

import pandas as pd

from src_python.pertussis.config import REPO_ROOT

from src_python.pertussis.inference.structure_variants import get_variant


def _has_bad_japan_age_fingerprint(path, first, last):
    """Recognise copied/subsetted versions of the known bad age table."""
    if not path.is_file():
        return False
    columns=['country','year','source_age_group','reported_cases']
    try:
        frame=pd.read_csv(path, usecols=columns)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    signature={2018:[508,75,567,4241,2875],2019:[761,95,785,5685,4305],
               2020:[143,14,173,778,585],2021:[16,12,176,112,52],
               2022:[14,13,104,82,44],2023:[20,15,201,280,106]}
    labels=['00-01','01-04','05-09','10-14','15-19']
    for year,counts in signature.items():
        if (first is not None and year<int(first)) or (last is not None and year>int(last)):
            continue
        part=frame[(frame.country=='Japan')&(frame.year==year)]
        if part.source_age_group.duplicated().any():
            continue
        values=part.set_index('source_age_group').reported_cases
        if all(k in values.index and values[k]==v for k,v in zip(labels,counts)):
            return True
    return False


def validate_projection_variant(spec, row=None, *, posterior=False):
    """Fail before simulation if time-varying reporting would be discarded.

    A prior diagnostic can record its variant at the top level, in settings,
    or in its fitted row. An explicit school-shift coordinate is also enough
    to require a time-varying observation model, even when numerically zero.
    """
    settings=spec.get('settings') or {}
    # Japan 2018--2023 values in the historical shared file have known
    # age-column mislabelling; a converged optimiser cannot correct it.
    if spec.get('country') == 'Japan':
        default_age=REPO_ROOT/'data/processed/pertussis_age_case_observations_source_audited.csv'
        age_path=Path(settings.get('age_observations_path') or default_age)
        if not age_path.is_absolute():
            age_path=REPO_ROOT/age_path
        first=settings.get('train_start_year')
        last=settings.get('train_end_year')
        overlaps=(first is None or int(first)<=2023) and (last is None or int(last)>=2018)
        if overlaps and (age_path.resolve()==default_age.resolve() or _has_bad_japan_age_fingerprint(age_path,first,last)):
            raise ValueError('Japan fit uses the known mislabelled 2018–2023 age input; refit with a corrected source')
    values=dict(row) if row is not None else {}
    names=[]
    for value in (spec.get('variant'),settings.get('variant'),values.get('variant')):
        if value is not None and not pd.isna(value):
            names.append(get_variant(str(value)).name)
    if 's5_school_detection' in names or ('school_detection_logit_shift' in values and pd.notna(values['school_detection_logit_shift'])):
        raise ValueError('time-varying school reporting is not supported by the projection evaluator')
    explicit=get_variant(str(spec['variant'])).name if spec.get('variant') else 'baseline'
    setting=get_variant(str(settings.get('variant') or 'baseline')).name
    row_variant=get_variant(str(values['variant'])).name if values.get('variant') is not None and not pd.isna(values['variant']) else 'baseline'
    if explicit!='baseline' and explicit not in (setting,row_variant):
        raise ValueError('top-level structural variant is not represented in fit settings or row')
    if posterior and 'exponential_vaccine_waning' in names and spec.get('structure')!='exponential':
        raise ValueError('exponential posterior variant disagrees with stored waning structure')
    if posterior and any(n not in ('baseline','exponential_vaccine_waning') for n in names):
        raise ValueError('nonbaseline posterior variants require a variant-aware objective replay')
