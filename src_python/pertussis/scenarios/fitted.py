"""Load a fitted country model for prospective projection.

Shared by the programme-versus-none runner and the unified common-strategy
runner so both rebuild exactly the same inputs, parameters and history.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..inputs.bundle import BundleSettings, build_inputs, programme_path
from ..inputs.programme import SeriesSegment, load_programme
from ..model.simulate import Simulator
from ..inference.structure_variants import get_variant
from ..states import MODEL_VERSION
from .run_resource_ensemble import _parameters

REPORTING_COLUMNS = ['reporting_logit_<1', 'reporting_logit_1-4', 'reporting_logit_5-9',
                     'reporting_logit_10-19', 'reporting_logit_20-59', 'reporting_logit_60+',
                     'breakthrough_reporting_ratio']


def hold_terminal_coverage(programme, end_year):
    """Hold coverage series that stop in 2050 at their 2050 value to ``end_year``.

    A long-horizon sensitivity carries terminal assumptions forward; it does not
    extrapolate trends or touch historical schedules.
    """
    if end_year <= 2050:
        return programme
    for series in programme.coverage_series.values():
        if not series.derived and max(s.last for s in series.segments) == 2050:
            value = series.value(2050, programme)
            series.segments.append(SeriesSegment(2051, end_year, 'value', value,
                                                 ('diagnostic_hold_2050_coverage',), 'D'))
            series._cache.clear()
    return programme


def build_fitted_inputs(spec, end_year):
    """Model inputs for a saved run specification, extended to ``end_year``."""
    s = spec['settings']
    bundle = BundleSettings(history_start_year=s['history_start_year'], spinup_years=s['spinup_years'],
                            reference_year=s['reference_year'], last_year=end_year)
    if s.get('childhood_coverage_variant') == 'england_cover_cohort':
        from ..inputs.england_cover import load_england_cover_programme
        programme = load_england_cover_programme()
    else:
        programme = load_programme(Path(s['programme_path']) if s.get('programme_path')
                                   else programme_path(spec['country']))
    inputs, _ = build_inputs(spec['country'], bundle, programme=hold_terminal_coverage(programme, end_year))
    if s.get('childhood_coverage_variant') == 'england_cover_cohort':
        inputs.metadata.update(childhood_coverage_variant='england_cover_cohort')
    return inputs


def fit_parameters(spec, row):
    """Build projection parameters, retaining the fitted structural variant.

    A disabled coordinate can remain in a fit row as a prior-valued nuisance
    coordinate. It must not silently reactivate the corresponding mechanism.
    Older fit manifests may record an explicitly selected variant only on the
    fit row, while their settings contain the default ``baseline`` value.
    """
    values = dict(row)
    setting_name = spec['settings'].get('variant') or 'baseline'
    setting_variant = get_variant(setting_name)
    row_name = values.get('variant')
    row_variant = None if row_name is None or pd.isna(row_name) else get_variant(str(row_name))
    if row_variant is not None and setting_variant.name != 'baseline' and row_variant.name != setting_variant.name:
        raise ValueError('fit row and run settings specify different structural variants')
    variant = row_variant or setting_variant
    if variant.name != 'baseline' and variant.vaccine_waning_shape != spec['structure']:
        raise ValueError('fit structural variant and projection waning shape disagree')
    overrides = dict(spec['settings']['model_overrides'])
    if variant.disable_natural_boosting and 'boosting_kappa' in overrides:
        raise ValueError('model override conflicts with disabled natural boosting')
    params = replace(_parameters(values, vaccine_waning_shape=spec['structure']), **overrides)
    return replace(params, boosting_kappa=0.) if variant.disable_natural_boosting else params


def validate_fit_model(spec):
    """Prevent silently projecting a fit from a different protection model."""
    if spec.get('model_version') != MODEL_VERSION:
        raise ValueError(
            f"Fit uses {spec.get('model_version', 'legacy replacement protection')}; "
            f"current model is {MODEL_VERSION}. Refit or explicitly validate and "
            "migrate the complete historical trajectory before projection.")


def load_fit(fit_dir):
    """(spec, best fit row) of a saved fit directory."""
    fit_dir = Path(fit_dir)
    spec = json.loads((fit_dir / 'run_spec.json').read_text())
    validate_fit_model(spec)
    best = json.loads((fit_dir / 'complete.json').read_text())['best_start']
    fits = pd.read_csv(fit_dir / 'fit_summary.csv')
    return spec, fits[fits.start_index.eq(best)].iloc[0]


def history_to(inputs, params, start_year):
    """Spin-up plus history up to 1 January of ``start_year``."""
    sim = Simulator(inputs)
    y0, acc0 = sim.initial_state()
    return sim.run(params, y0, acc0, 0, inputs.calendar.year_start_day(start_year))


def resample_rows(frame, n, seed):
    """Weighted resampling with replacement; equal weights if no weight column."""
    w = frame['weight'].to_numpy(float) if 'weight' in frame.columns else np.ones(len(frame))
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('posterior weights must be finite, nonnegative and not all zero')
    rng = np.random.default_rng(seed)
    return rng.choice(len(frame), size=n, replace=True, p=w / w.sum())
