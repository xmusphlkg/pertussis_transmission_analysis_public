"""Borrow a reference country's calibrated course risks for countries without
matched national fatality evidence.

The six migration countries have no age- and case-definition-matched national
fatality or ascertainment evidence. This module applies the already calibrated
Australian and English conditional death risks to their modelled onsets. The
result is an explicit METHOD EXTRAPOLATION: it is not national evidence, not a
country fatality estimate, and not a posterior over a country's deaths.

Each reference source resolves infant age only at its own published boundary
(2 months for Australia, 3 for England), so onsets are aggregated separately
per source and the two sources are reported separately. They are never pooled,
averaged or reconciled into a single death number. Every output row keeps the
risk source, its risk scenario and draw identity so no borrowed death count can
be read as local evidence.

Reference risks are themselves conditional on one reference-country transmission
trajectory; that trajectory is recorded in the run metadata, and its draw
identity is carried as risk_source_* columns, never merged with the modelled
country's own draw identifiers.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .health_comparison import cumulative_contrasts
from .health_outcomes import INFANT_BINS, infant_risk_age_groups, infant_risk_age_label, project_deaths

CYCLE = 'outputs/core2026/health_outcomes_20261026/current_cycle_all_points/conditional_mortality'
RISK_COLUMNS = ['risk_scenario', 'risk_draw_id', 'risk_age', 'age_start_months',
                'age_end_months_exclusive', 'p_death_given_typical', 'p_death_given_mild',
                'death_ascertainment', 'prior_shape']
# Trajectory identity of the reference fit, kept distinct from the modelled
# country's own draw identity.
TRAJECTORY_COLUMNS = {'draw_id': 'risk_source_draw_id', 'structure_id': 'risk_source_structure_id',
                      'maternal_variant': 'risk_source_maternal_variant',
                      'start_year': 'risk_source_start_year'}
REFERENCE_SOURCES = {
    'Australia': dict(
        risk_draws=f'{CYCLE}/fitted_cycle2025_australia_main/risk_draws.csv.gz',
        # The Australian risk table indexes risk_age positionally; its monthly
        # boundary lives in the observation file it was calibrated against.
        age_bounds='data/processed/health_outcomes_20261026/australia_infant_deaths.csv',
        run_spec='outputs/core2026/health_outcomes_20261026/current_cycle_all_points/run_spec.json',
        source=None),
    'England': dict(
        risk_draws=f'{CYCLE}/fitted_cycle2025_england_main/risk_draws.csv.gz',
        age_bounds=None,  # the English risk table publishes its own bounds
        run_spec=f'{CYCLE}/fitted_cycle2025_england_main/run_spec.json',
        source='data/processed/health_outcomes_20261026/england_linked_infant_deaths.csv'
               ' (UKHSA_linked_enhanced_HES_ONS)'),
}


def _australia_age_bounds(path):
    """Positional risk_age to published monthly bounds, as calibrated.

    The calibration sorts its fitted observation rows by age_start_months and
    uses that order as risk_age, so the same ordering is reproduced here.
    """
    obs = pd.read_csv(path)
    fit = obs[obs.role.eq('fit') & obs.stream.eq('ACR')].sort_values('age_start_months')
    bounds = fit[['age_start_months', 'age_end_months_exclusive']].drop_duplicates()
    bounds = bounds.sort_values('age_start_months').reset_index(drop=True)
    bounds['risk_age'] = range(len(bounds))
    return bounds, str(fit.source.iloc[0])


def load_reference_risks(source_profile, *, repo_root=REPO_ROOT):
    """Tidy one reference country's calibrated risk draws with provenance.

    Returns the draws and a metadata dictionary recording the reference files,
    their hashes and the reference transmission trajectory the risks condition
    on. The two infant risk groups of one draw are dependent by construction,
    so callers must subsample draw identifiers, never single age rows.
    """
    if source_profile not in REFERENCE_SOURCES:
        raise ValueError(f'no calibrated reference risks for {source_profile}')
    spec = REFERENCE_SOURCES[source_profile]
    root = Path(repo_root)
    paths = {name: root/spec[name] for name in ['risk_draws', 'age_bounds', 'run_spec'] if spec[name]}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'missing reference mortality inputs: {missing}')
    risks = pd.read_csv(paths['risk_draws'])
    source = spec['source']
    if spec['age_bounds']:
        bounds, source = _australia_age_bounds(paths['age_bounds'])
        if {'age_start_months', 'age_end_months_exclusive'} & set(risks.columns):
            raise ValueError('reference risks already publish bounds; remove the override')
        risks = risks.merge(bounds, on='risk_age', how='left', validate='many_to_one')
    for column, renamed in TRAJECTORY_COLUMNS.items():
        if column in risks.columns:
            risks = risks.rename(columns={column: renamed})
    trajectory = {renamed: sorted(set(risks[renamed].tolist()))
                  for renamed in TRAJECTORY_COLUMNS.values() if renamed in risks.columns}
    if any(len(v) != 1 for v in trajectory.values()):
        raise ValueError('borrowed risks must come from one reference trajectory')
    if set(RISK_COLUMNS)-set(risks.columns):
        raise ValueError('reference risk draws miss required risk columns')
    if risks[RISK_COLUMNS].isna().any().any():
        raise ValueError('reference risk draws cannot be missing')
    if 'geographic_scope' in risks.columns:
        if not risks.geographic_scope.eq(source_profile).all():
            raise ValueError('reference risk geography does not match its source country')
    else:
        risks['geographic_scope'] = source_profile
    if 'denominator' in risks.columns:
        if not risks.denominator.eq('modelled_course_specific_onsets').all():
            raise ValueError('reference risks must be conditional on modelled onsets')
    else:
        risks['denominator'] = 'modelled_course_specific_onsets'
    if 'source' not in risks.columns:
        if not source:
            raise ValueError('a reference risk source citation is required')
        risks['source'] = source
    groups = risks[['risk_age', 'age_start_months', 'age_end_months_exclusive']].drop_duplicates()
    groups = groups.sort_values('risk_age').reset_index(drop=True)
    starts = groups.age_start_months.tolist()
    ends = groups.age_end_months_exclusive.tolist()
    if (len(groups) < 2 or groups.risk_age.tolist() != list(range(len(groups)))
            or starts[0] != 0 or ends[-1] != INFANT_BINS or starts[1:] != ends[:-1]):
        raise ValueError('reference risk ages must partition the 12 infant months in order')
    if len(groups) != 2:
        raise ValueError('this adapter maps one early/later infant boundary per source')
    counts = risks.groupby(['risk_scenario', 'risk_age']).risk_draw_id.nunique()
    if counts.nunique() != 1 or risks.duplicated(['risk_scenario', 'risk_age', 'risk_draw_id']).any():
        raise ValueError('every reference risk scenario needs the same unique draw identifiers')
    reference = {k: v[0] for k, v in trajectory.items()}
    specification = json.loads(paths['run_spec'].read_text()) if 'run_spec' in paths else {}
    if not reference:
        # The English risk table omits trajectory columns; its conditioning
        # trajectory is recorded in the calibration run specification.
        declared = specification.get('trajectory', {})
        reference = {renamed: declared[column] for column, renamed in TRAJECTORY_COLUMNS.items()
                     if column in declared}
    if not reference:
        raise ValueError('the reference trajectory behind borrowed risks must be recorded')
    metadata = dict(risk_source_profile=source_profile,
        early_age_end_months=int(ends[0]), risk_age_groups_months=[[int(a), int(b)]
            for a, b in zip(starts, ends)],
        risk_scenarios=sorted(set(risks.risk_scenario)), available_risk_draws=int(counts.iloc[0]),
        reference_trajectory=reference,
        reference_inference=specification.get('inference') or specification.get('interpretation'),
        source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths.values()},
        interpretation='conditional reference-country course risks borrowed for a country'
                       ' without matched national fatality evidence')
    return risks.sort_values(['risk_scenario', 'risk_draw_id', 'risk_age']).reset_index(drop=True), metadata


def subsample_risk_draws(risks, size, seed):
    """Keep whole risk draws, paired across the dependent infant age groups."""
    available = sorted(set(risks.risk_draw_id))
    if size < 1 or int(size) != size:
        raise ValueError('positive integer risk subsample size required')
    if size > len(available):
        raise ValueError('requested more risk draws than the reference posterior provides')
    chosen = np.random.default_rng(seed).choice(np.asarray(available), size=int(size), replace=False)
    out = risks[risks.risk_draw_id.isin(set(chosen.tolist()))]
    if out.risk_draw_id.nunique() != size:
        raise ValueError('risk subsample lost draw identifiers')
    return out.reset_index(drop=True)


def extrapolated_death_contrasts(age, receipts, config, risks, metadata, *,
                                 size, seed, horizons=None):
    """Paired strategy death contrasts under borrowed course risks.

    Monthly infant onsets are aggregated to the reference source's own boundary
    and each kept risk draw is applied through `project_deaths`, so every row
    carries risk_application='extrapolated' and its risk source. Risk scenarios
    and draws stay separate columns; nothing is averaged over them.
    """
    source_profile = metadata['risk_source_profile']
    infant = age[age.bin_index.lt(INFANT_BINS)]
    grouped = infant_risk_age_groups(infant, early_age_end_months=metadata['early_age_end_months'])
    if set(grouped.profile) == {source_profile}:
        raise ValueError('a matched country must use its own calibrated risks')
    years = sorted(set(grouped.year))
    labels = grouped[['risk_age', 'age_start_months', 'age_end_months_exclusive']].drop_duplicates()
    kept = subsample_risk_draws(risks, size, seed)
    settings = dict(config, cumulative_horizons_years=horizons or config['cumulative_horizons_years'])
    frames = []
    for (scenario, draw_id), draw in kept.groupby(['risk_scenario', 'risk_draw_id'], sort=True):
        table = draw.merge(labels, on=['risk_age', 'age_start_months', 'age_end_months_exclusive'],
                           how='inner', validate='one_to_one')
        if len(table) != len(labels):
            raise ValueError('reference risk ages do not match the aggregated onset groups')
        table = table.assign(key=1).merge(pd.DataFrame(dict(year=years, key=1)), on='key').drop(columns='key')
        table = table.assign(profile=grouped.profile.iloc[0],
            age_bin=[infant_risk_age_label(lo, hi) for lo, hi in
                     zip(table.age_start_months, table.age_end_months_exclusive)],
            risk_source_profile=source_profile,
            risk_set_id=f'{source_profile}:{scenario}:{draw_id}',
            risk_application='extrapolated')
        deaths = project_deaths(grouped, table.drop(columns=['risk_age', 'age_start_months',
            'age_end_months_exclusive']))
        frames.append(deaths.assign(risk_scenario=scenario, risk_draw_id=draw_id))
    deaths = pd.concat(frames, ignore_index=True)
    out = cumulative_contrasts(deaths, receipts, settings, 'expected_deaths', infant_only=True)
    out['risk_source_profile'] = source_profile
    out['risk_application'] = 'extrapolated'
    out['risk_early_age_end_months'] = metadata['early_age_end_months']
    out['mortality_evidence'] = 'borrowed_reference_country_risk_not_national_evidence'
    return out
