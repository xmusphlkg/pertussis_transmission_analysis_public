"""Summarise the completed BPZE1 policy experiment with paired contrasts.

Every product profile keeps its own decision set. Clinical anchor endpoints,
effect allocation, transport and duration are named conditions; conditional
clinical/cost intervals are computed from the retained 1,000 paired draws.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_bpze1_policy_extension_20261107 as workflow
from scripts.analyse_mucosal_vaccine_extension_20261107 import describe, marginal_components

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/core2026/02_sensitivity/mucosal_vaccines/bpze1_policy_20261107'
PRIMARY = ('reported_uniform', 'included_base')
YEARS = tuple(range(2027, 2057))
CASE_AGES = ('infant_0', 'all_ages', 'adolescent_10_19')
AGE_METRICS = ('reported_cases_averted', 'hospitalisations_averted', 'deaths_averted', 'yll_averted')


def primary(frame):
    return frame.loc[frame.burden.eq(PRIMARY[0]) & frame.mortality.eq(PRIMARY[1])].copy()


def run_design(manifest):
    """Load the named BPZE1 design used by the retained policy run."""
    path = ROOT / 'config/extensions/bpze1_policy_20261107.json'
    return path, workflow.load_design(path)


def validate_policy_grid(catalog, arms, standard_catalog):
    """Require both routes and every original feasible maternal/nasal cell."""
    expected = workflow.policy_catalog(standard_catalog)
    keys = ['route', 'maternal_coverage', 'nasal_coverage']
    if catalog.strategy_id.duplicated().any() or catalog.duplicated(keys).any():
        raise ValueError('Unique policy cells and strategy IDs required')
    if len(arms) != len(set(arms)) or not set(catalog.strategy_id) <= set(arms):
        raise ValueError('Unique complete paired strategy set required')
    standard = [a for a in arms if a not in set(catalog.strategy_id)]
    if set(standard) != set(standard_catalog.strategy_id):
        raise ValueError('All original standard strategies required')
    actual = catalog.set_index(keys).sort_index()
    target = expected.set_index(keys).sort_index()
    if not actual.index.equals(target.index):
        raise ValueError('Both routes and the complete original maternal/nasal coverage grid required')
    for name in ('matched_standard_strategy', 'target_coverage_standard_strategy'):
        if not actual[name].equals(target[name]):
            raise ValueError('Product background comparator differs: ' + name)
    np.testing.assert_allclose(actual.tdap_coverage, target.tdap_coverage, rtol=0., atol=1e-12)
    if not set(catalog.matched_standard_strategy) <= set(standard):
        raise ValueError('Every product comparator must be a standard strategy')
    return [arms.index(a) for a in standard]


def annual_arrays(frame, arms, metrics, *, age_groups=None):
    """Read complete annual outcome levels; absent cells never become zeros."""
    keys = ['strategy_id', 'year'] + ([] if age_groups is None else ['age_group'])
    selected = frame if age_groups is None else frame.loc[frame.age_group.isin(age_groups)]
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate annual outcome cells')
    grid = [arms, YEARS] + ([] if age_groups is None else [age_groups])
    index = pd.MultiIndex.from_product(grid, names=keys)
    table = selected.set_index(keys).reindex(index)
    result = {}
    shape = (len(arms), len(YEARS)) + (() if age_groups is None else (len(age_groups),))
    for metric in metrics:
        values = table[metric].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Complete finite nonnegative annual levels required: ' + metric)
        result[metric] = values.reshape(shape)
    return result


def age_effect_table(frame, arms):
    keys = ['horizon', 'age_group', 'strategy_id']
    selected = frame.loc[frame.age_group.isin(('infant_0', 'all_ages'))]
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate infant/all-age effect cells')
    index = pd.MultiIndex.from_product([(20, 30), ('infant_0', 'all_ages'), arms], names=keys)
    table = selected.set_index(keys).reindex(index)
    if not np.isfinite(table[list(AGE_METRICS)].to_numpy(float)).all():
        raise ValueError('Complete finite infant/all-age effects required')
    return table


def standard_draw_identity(cache, arms, standard_ids, reference=None):
    """Check row-for-row pairing across profiles using every standard component."""
    if len(arms) != len(set(arms)) or not set(standard_ids) <= set(arms):
        raise ValueError('Complete uniquely indexed standards required for cross-profile pairing')
    indices = [arms.index(s) for s in standard_ids]
    current = {}
    for key, array in cache.items():
        values = array[indices] if array.ndim == 1 else array[:, indices]
        if not np.isfinite(values).all():
            raise ValueError('Finite standard components required for cross-profile pairing')
        current[key] = values.copy()
    if reference is not None:
        if set(current) != set(reference):
            raise ValueError('Cross-profile standard component keys differ')
        for key in current:
            np.testing.assert_allclose(current[key], reference[key], rtol=64*np.finfo(float).eps,
                atol=1e-10, err_msg='Cross-profile standard draw identity: ' + key)
    return current


def mechanism_policy(catalog, standard_catalog, route):
    """Fix baseline maternal coverage and the first available nasal target >=70%."""
    current = standard_catalog.loc[standard_catalog.strategy_id.eq('current')]
    if len(current) != 1:
        raise ValueError('One current strategy required for mechanism comparison')
    maternal = float(current.iloc[0].maternal_coverage)
    if not np.isclose(maternal, catalog.maternal_coverage.min()):
        raise ValueError('Baseline maternal coverage must be the lowest retained level')
    target = max(.7, float(current.iloc[0].adolescent_coverage))
    options = catalog.loc[catalog.route.eq(route) & np.isclose(catalog.maternal_coverage, maternal)
                          & (catalog.nasal_coverage >= target-1e-12)].sort_values('nasal_coverage')
    if options.empty or options.duplicated('nasal_coverage').any():
        raise ValueError('Unique feasible nasal targets required for mechanism comparison')
    return options.iloc[0]


def mechanism_tables(folder, countries, profiles):
    """Compare mechanism allocation at fixed clinical anchor, bridge and policy.

    This routine reads completed local economic blocks, so it can also verify a
    finished country while other countries are running. The summary calls it
    with the exact countries and profiles in the completed run manifest.
    """
    selected_profiles = [p for p in profiles if p['clinical_anchor_label'] == 'point'
        and p['mean_years'] == 3 and p['bridge_strength'] in (.25, .5, 1.)
        and p['allocation'] in (0., .5, 1.)]
    conditions = [(p['bridge_strength'], p['allocation']) for p in selected_profiles]
    if len(conditions) != len(set(conditions)):
        raise ValueError('One profile per fixed bridge/mechanism condition required')
    response, paired, vectors = [], [], {}
    checks = dict(mechanism_response_rows=0, mechanism_paired_comparisons=0,
                  cross_profile_standard_identity_checks=0)
    for country in countries:
        standard_catalog = pd.read_csv(workflow.adopted.OUT / country / 'strategy_catalog.csv')
        standard_ids = standard_catalog.strategy_id.tolist()
        standard_reference = None
        clinical_reference = None
        source_reference = None
        for profile in selected_profiles:
            pid = profile['candidate_id']
            local = Path(folder) / country / 'profiles' / pid
            meta = json.loads((local/'economic_metadata.json').read_text())
            if (meta.get('status') != 'completed_paired_policy_evaluation' or
                    meta['country'] != country or meta['profile_id'] != pid or meta['clinical_cost_draws'] != 1000):
                raise ValueError('Completed matching economic block required for mechanism analysis')
            if clinical_reference is None:
                clinical_reference = meta['clinical_calculation']
            elif meta['clinical_calculation'] != clinical_reference:
                raise ValueError('Cross-profile clinical sampler settings differ')
            def verified_csv(name):
                return pd.read_csv(local/name)
            catalog = pd.read_csv(local/'product_catalog.csv')
            projection = json.loads((local/'projection_metadata.json').read_text())
            paired_path = local/'raw/paired_components.npz'
            with np.load(paired_path, allow_pickle=False) as archive:
                arms = archive['strategy_ids'].astype(str).tolist()
                cache = {key: archive[key] for key in archive.files if key != 'strategy_ids'}
            validate_policy_grid(catalog, arms, standard_catalog)
            new_standard = standard_draw_identity(cache, arms, standard_ids, standard_reference)
            if standard_reference is not None:
                checks['cross_profile_standard_identity_checks'] += len(cache)
            else:
                standard_reference = new_standard
            age = age_effect_table(primary(verified_csv('age_outcomes.csv.gz')), arms)
            annual = annual_arrays(verified_csv('annual_cases.csv.gz'), arms,
                                   ('reported_cases', 'symptomatic_cases'), age_groups=CASE_AGES)
            indices = {arm: i for i, arm in enumerate(arms)}
            current = indices['current']
            for route in ('standalone', 'coadmin'):
                policy = mechanism_policy(catalog, standard_catalog, route)
                candidate = indices[policy.strategy_id]
                for horizon in (20, 30):
                    record = dict(country=country, **workflow.profile_tags(profile), horizon=horizon,
                        route=route, strategy_id=policy.strategy_id, comparison='versus_current',
                        maternal_coverage=float(policy.maternal_coverage), nasal_coverage=float(policy.nasal_coverage),
                        tdap_coverage=float(policy.tdap_coverage), matched_standard_strategy=policy.matched_standard_strategy,
                        target_coverage_standard_strategy=policy.target_coverage_standard_strategy,
                        matched_background_definition=('same maternal coverage with baseline Tdap'
                            if route == 'standalone' else 'same maternal and target Tdap coverage'))
                    values = marginal_components(cache, candidate, current, horizon, *PRIMARY)
                    for metric, value in values.items():
                        record.update(describe(value, metric))
                    for ag in ('infant_0', 'all_ages'):
                        for metric in AGE_METRICS:
                            record[f'{ag}_{metric}'] = float(age.loc[(horizon, ag, policy.strategy_id), metric] -
                                                            age.loc[(horizon, ag, 'current'), metric])
                    for g, ag in enumerate(CASE_AGES):
                        for metric, array in annual.items():
                            value = float((array[current, :horizon, g]-array[candidate, :horizon, g]).sum())
                            key = f'{ag}_{metric}_averted'
                            if key in record:
                                np.testing.assert_allclose(record[key], value, rtol=1e-10, atol=1e-7)
                            record[key] = value
                        label = {'infant_0': 'infant', 'all_ages': 'all_age',
                                 'adolescent_10_19': 'adolescent'}[ag]
                        baseline = float(annual['reported_cases'][current, :horizon, g].sum())
                        if baseline <= 0:
                            raise ValueError('Positive current age-specific cases required for percentages')
                        record[f'{label}_cases_current'] = baseline
                        record[f'{label}_cases_averted_percent_of_current'] = (
                            100.*record[f'{ag}_reported_cases_averted']/baseline)
                    response.append(record)
                    vectors[(country, profile['bridge_strength'], route, horizon, profile['allocation'])] = (record, values)
        # Explicitly compare the two allocation endpoints within each country.
        for bridge in (.25, .5, 1.):
            for route in ('standalone', 'coadmin'):
                for horizon in (20, 30):
                    low = vectors.get((country, bridge, route, horizon, 0.))
                    high = vectors.get((country, bridge, route, horizon, 1.))
                    if low is None or high is None:
                        continue  # A selected-profile pilot can contain one allocation only.
                    left, lv = low; right, rv = high
                    fixed = ('observed_auc_ratio', 'reference_residual_potential', 'protected_state_residual',
                             'mean_years', 'maternal_coverage', 'nasal_coverage', 'tdap_coverage',
                             'infant_cases_current', 'adolescent_cases_current', 'all_age_cases_current')
                    for key in fixed:
                        numerical = key.endswith('_cases_current')
                        np.testing.assert_allclose(left[key], right[key],
                                                   rtol=64*np.finfo(float).eps if numerical else 0.,
                                                   atol=1e-10 if numerical else 1e-12,
                                                   err_msg='Mechanism comparison changes ' + key)
                    record = {key: left[key] for key in ('country', 'bridge_strength', 'horizon', 'route', *fixed)}
                    record.update(contrast='allocation_0_minus_allocation_1',
                        allocation_0_profile_id=left['profile_id'], allocation_1_profile_id=right['profile_id'],
                        allocation_0_strategy_id=left['strategy_id'], allocation_1_strategy_id=right['strategy_id'],
                        clinical_anchor_label='point', comparison='difference_in_benefits_and_costs_vs_current',
                        matched_standard_strategy=left['matched_standard_strategy'],
                        matched_background_definition=left['matched_background_definition'])
                    if left['matched_standard_strategy'] != right['matched_standard_strategy']:
                        raise ValueError('Mechanism comparison changes the matched standard background')
                    for metric in lv:
                        record.update(describe(lv[metric]-rv[metric], metric))
                    for key in left:
                        if key.startswith(CASE_AGES) and key.endswith('_averted'):
                            record[key] = left[key]-right[key]
                        if key.endswith('_cases_averted_percent_of_current'):
                            record[key] = left[key]-right[key]
                    paired.append(record)
    checks['mechanism_response_rows'] = len(response)
    checks['mechanism_paired_comparisons'] = len(paired)
    return pd.DataFrame(response), pd.DataFrame(paired), checks


def summary(folder, *, check=False):
    folder = Path(folder).resolve()
    workflow.check_run(folder)
    manifest = json.loads((folder / 'metadata.json').read_text())
    if manifest['status'] not in ('completed_all', 'completed_economics'):
        raise ValueError('Completed policy economics required')
    config_path, config = run_design(manifest)
    profiles = {p['candidate_id']: p for p in config['profiles']}
    if (not manifest['countries'] or len(set(manifest['countries'])) != len(manifest['countries'])
            or not manifest['profile_ids'] or len(set(manifest['profile_ids'])) != len(manifest['profile_ids'])
            or not set(manifest['profile_ids']) <= set(profiles)):
        raise ValueError('Unique nonempty countries and configured profiles required')
    marginal, interaction, route_differences, windows, best = [], [], [], [], []
    checks = dict(price_reconstructions=0, paired_route_comparisons=0, maternal_interactions=0)
    for country in manifest['countries']:
        standard_catalog = pd.read_csv(workflow.adopted.OUT / country / 'strategy_catalog.csv')
        for pid in manifest['profile_ids']:
            local = folder / country / 'profiles' / pid
            catalog = pd.read_csv(local / 'product_catalog.csv')
            age = primary(pd.read_csv(local / 'age_outcomes.csv.gz'))
            annual = pd.read_csv(local / 'annual_cases.csv.gz')
            receipts = pd.read_csv(local / 'annual_receipts_and_infections.csv')
            caps = primary(pd.read_csv(local / 'comparator_ceilings.csv'))
            high = primary(pd.read_csv(local / 'highest_reference_summary.csv'))
            with np.load(local / 'raw/paired_components.npz', allow_pickle=False) as archive:
                arms = archive['strategy_ids'].astype(str).tolist()
                cache = {k: archive[k] for k in archive.files if k != 'strategy_ids'}
            indices = {a: i for i, a in enumerate(arms)}
            standard = validate_policy_grid(catalog, arms, standard_catalog)
            ledger = annual_arrays(receipts, arms, ('nasal_doses', 'infections'))
            case_levels = annual_arrays(annual, arms, ('reported_cases', 'symptomatic_cases'), age_groups=CASE_AGES)
            age_effects = age_effect_table(age, arms)
            dose = ledger['nasal_doses']
            if np.any(dose[standard]):
                raise ValueError('Invalid actual nasal dose ledger')
            profile = workflow.profile_tags(profiles[pid])
            common = dict(country=country, **profile)
            for horizon in (20, 30):
                discounted = dose[:, :horizon] @ (1.03 ** -np.arange(horizon))
                top = high.loc[high.horizon.eq(horizon), 'threshold_intl2024']
                if top.empty or top.nunique() != 1:
                    raise ValueError('One maximum reference value required, allowing source aliases')
                lam = float(top.iloc[0])
                if not np.isfinite(lam) or lam <= 0 or lam != caps.loc[caps.horizon.eq(horizon), 'threshold_intl2024'].max():
                    raise ValueError('Positive maximum sourced reference value required')
                effects = cache[f'yll_{horizon}'] + cache[f'yld_{horizon}_{PRIMARY[0]}'][None, :]
                costs = cache[f'cost_{horizon}']
                if (effects.shape != (1000, len(arms)) or costs.shape != effects.shape
                        or not np.isfinite(effects).all() or not np.isfinite(costs).all()):
                    raise ValueError('Finite paired 1000-draw economic components required')
                std_nmb = (lam * effects[:, standard] - costs[:, standard]).mean(axis=0)
                std_best = standard[int(np.argmax(std_nmb))]
                matches = {}
                for row in catalog.to_dict('records'):
                    sid = row['strategy_id']
                    i, j = indices[sid], indices[row['matched_standard_strategy']]
                    if discounted[i] <= 0:
                        raise ValueError('Positive actual nasal doses required for every product arm')
                    tags = dict(common, horizon=horizon, strategy_id=sid, route=row['route'],
                        maternal_coverage=row['maternal_coverage'], nasal_coverage=row['nasal_coverage'],
                        tdap_coverage=row['tdap_coverage'], matched_standard_strategy=row['matched_standard_strategy'])
                    values = marginal_components(cache, i, j, horizon, *PRIMARY)
                    matches[sid] = values
                    record = dict(tags)
                    for key, v in values.items():
                        record.update(describe(v, key))
                    record['discounted_nasal_doses'] = float(discounted[i])
                    marginal.append(record)
                    actual = (lam * (effects[:, i] - effects[:, std_best]) -
                              (costs[:, i] - costs[:, std_best])).mean() / discounted[i]
                    cell = caps.loc[caps.strategy_id.eq(sid) & caps.horizon.eq(horizon) &
                                    caps.threshold_intl2024.eq(lam)]
                    if cell.empty:
                        raise ValueError('Maximum-reference price cell required')
                    np.testing.assert_allclose(actual, cell.ceiling_vs_best_standard.to_numpy(), rtol=1e-10, atol=1e-7)
                    checks['price_reconstructions'] += 1
                # Route contrasts use the same maternal/nasal coverage and paired draws.
                for (m, a), group in catalog.groupby(['maternal_coverage', 'nasal_coverage'], sort=False):
                    if set(group.route) != {'standalone', 'coadmin'} or len(group) != 2:
                        raise ValueError('Exactly two routes required per policy cell')
                    route = group.set_index('route')
                    i = indices[route.loc['coadmin', 'strategy_id']]
                    j = indices[route.loc['standalone', 'strategy_id']]
                    record = dict(common, horizon=horizon, maternal_coverage=m, nasal_coverage=a,
                                  contrast='coadmin minus standalone health benefit; coadmin minus standalone cost')
                    for key, v in marginal_components(cache, i, j, horizon, *PRIMARY).items():
                        record.update(describe(v, key))
                    route_differences.append(record)
                    checks['paired_route_comparisons'] += 1
                # Difference in nasal gains at each higher maternal level vs its baseline.
                for (route, a), group in catalog.groupby(['route', 'nasal_coverage'], sort=False):
                    group = group.sort_values('maternal_coverage')
                    low = group.iloc[0]
                    for row in group.iloc[1:].itertuples():
                        record = dict(common, horizon=horizon, route=route, nasal_coverage=a,
                            maternal_lower=low.maternal_coverage, maternal_higher=row.maternal_coverage)
                        for key in matches[row.strategy_id]:
                            v = matches[row.strategy_id][key] - matches[low.strategy_id][key]
                            record.update(describe(v, key))
                        interaction.append(record)
                        checks['maternal_interactions'] += 1
                # Select the maximum price candidate separately in each delivery route.
                local_caps = caps.loc[caps.horizon.eq(horizon) & caps.threshold_intl2024.eq(lam)]
                for route in ('standalone', 'coadmin'):
                    candidates = local_caps.loc[local_caps.route.eq(route)].sort_values(
                        ['ceiling_vs_best_standard', 'strategy_id'], ascending=[False, True])
                    selected = candidates.iloc[0]
                    row = catalog.set_index('strategy_id').loc[selected.strategy_id]
                    i, j = indices[selected.strategy_id], std_best
                    record = dict(common, horizon=horizon, route=route,
                        threshold_intl2024=lam, strategy_id=selected.strategy_id,
                        best_standard_strategy=arms[j], maternal_coverage=row.maternal_coverage,
                        nasal_coverage=row.nasal_coverage, tdap_coverage=row.tdap_coverage,
                        maximum_nasal_price_vs_best_standard=float(selected.ceiling_vs_best_standard),
                        ceiling_vs_matched_background=float(selected.ceiling_vs_same_coverage_standard),
                        discounted_nasal_doses=float(discounted[i]))
                    for key, v in marginal_components(cache, i, j, horizon, *PRIMARY).items():
                        record.update(describe(v, key))
                    benefit = effects[:, i] - effects[:, j]
                    extra_cost = costs[:, i] - costs[:, j]
                    record.update(describe(benefit - extra_cost / lam, 'nhb_before_nasal_price'))
                    record.update(describe((lam * benefit - extra_cost) / discounted[i], 'fixed_comparator_price'))
                    for ag in ('infant_0', 'all_ages'):
                        for metric in AGE_METRICS:
                            record[f'{ag}_{metric}'] = float(age_effects.loc[(horizon, ag, selected.strategy_id), metric] -
                                                            age_effects.loc[(horizon, ag, arms[j]), metric])
                    best.append(record)
            # Paired case windows reveal changes after the first twenty years.
            for row in catalog.itertuples():
                i, j = indices[row.strategy_id], indices[row.matched_standard_strategy]
                for lo, hi in ((0, 20), (20, 30), (0, 30)):
                    record = dict(common, route=row.route, strategy_id=row.strategy_id,
                        matched_standard_strategy=row.matched_standard_strategy,
                        maternal_coverage=row.maternal_coverage, nasal_coverage=row.nasal_coverage,
                        followup_start=lo+1, followup_end=hi)
                    for g, ag in enumerate(CASE_AGES):
                        for metric in ('reported_cases', 'symptomatic_cases'):
                            values = case_levels[metric]
                            record[f'{ag}_{metric}_averted'] = float((values[j, lo:hi, g]-values[i, lo:hi, g]).sum())
                    record['infections_averted'] = float((ledger['infections'][j, lo:hi]-ledger['infections'][i, lo:hi]).sum())
                    windows.append(record)
    frames = {
        'highest_reference_policy_summary.csv': pd.DataFrame(best),
        'matched_background_contrasts.csv.gz': pd.DataFrame(marginal),
        'maternal_interactions.csv.gz': pd.DataFrame(interaction),
        'paired_route_comparisons.csv.gz': pd.DataFrame(route_differences),
        'case_time_windows.csv.gz': pd.DataFrame(windows),
    }
    mechanism, mechanism_contrasts, mechanism_checks = mechanism_tables(
        folder, manifest['countries'], [profiles[pid] for pid in manifest['profile_ids']])
    frames['mechanism_response.csv'] = mechanism
    frames['mechanism_paired_contrasts.csv'] = mechanism_contrasts
    checks.update(mechanism_checks)
    reference = config['primary_reference_profile']
    frames['reference_profile_policy_summary.csv'] = frames['highest_reference_policy_summary.csv'].loc[
        lambda d: d.profile_id.eq(reference)].copy()
    target = folder / 'followup'
    if not check:
        target.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        payload = frame.to_csv(index=False).encode()
        if name.endswith('.gz'):
            payload = gzip.compress(payload, mtime=0)
        p = target / name
        if check:
            if not p.is_file() or p.read_bytes() != payload:
                raise ValueError('Followup result differs: ' + str(p))
        else:
            p.write_bytes(payload)
    meta = dict(status='completed_bpze1_policy_followup', countries=manifest['countries'],
        profile_ids=manifest['profile_ids'], reference_profile_id=reference, checks=checks,
        analysis_scope='All original standard options, both nasal routes, full maternal coverage grid',
        uncertainty='1000 paired clinical/cost draws conditional on each stated clinical-anchor, transport, allocation and duration profile; anchor-endpoint profiles are sensitivity analyses',
        prices='2024 international dollars per actual nasal dose; standalone price covers standalone delivery, coadmin price covers incremental nasal delivery',
        infant_age='infant_0 is age below one year',
        mechanism_analysis='Point clinical AUC anchor, duration 3 years, bridge strengths 0.25/0.5/1; fixed baseline maternal coverage and lowest available nasal coverage >= max(70%, baseline adolescent coverage). Allocation 0 minus 1 contrasts retain paired clinical/cost rows after checking shared sampler sources/settings and standard-component identity at floating-point precision (rtol=64*machine_epsilon, atol=1e-10). All mechanism arm effects are versus current provision; costs exclude nasal price.',
        inputs={'policy_manifest': str((folder / 'metadata.json').relative_to(ROOT)),
                'configuration': str(config_path.relative_to(ROOT))})
    p = target / 'metadata.json'
    payload = json.dumps(meta, indent=2, allow_nan=False) + '\n'
    if check:
        if p.read_text() != payload:
            raise ValueError('Followup metadata differs')
    else:
        p.write_text(payload)
    print(json.dumps(dict(status='passed' if check else 'completed', **checks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    summary(args.input_dir, check=args.check)


if __name__ == '__main__':
    main()
