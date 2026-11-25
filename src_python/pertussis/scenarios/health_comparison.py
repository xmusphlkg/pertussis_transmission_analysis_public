"""Paired, complete-window health contrasts; never pool independent draws."""
import numpy as np
import pandas as pd

PAIR_KEYS = ['profile', 'maternal_variant', 'budget_doses_per_1000']
OPTIONAL_KEYS = ['draw_id', 'structure_id', 'start_year', 'risk_draw_id', 'risk_scenario']
ARMS = {'baseline', 'maternal_scaleup', 'adolescent_one_time_booster'}


def comparison_keys(frame):
    return PAIR_KEYS + [c for c in OPTIONAL_KEYS if c in frame]


def validate_panel(frame, keys, value_columns, start, end):
    if frame.empty or frame[keys + ['strategy_id', 'year', 'bin_index']].isna().any().any():
        raise ValueError('complete nonmissing paired panel required')
    if set(frame.strategy_id) != ARMS:
        raise ValueError('baseline and both strategies are required')
    if frame.duplicated(keys + ['strategy_id', 'year', 'bin_index']).any():
        raise ValueError('duplicate age/strategy/year rows; retain draw identifiers')
    counts = frame[value_columns].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('outcomes must be finite and nonnegative')
    expected = pd.MultiIndex.from_product([sorted(ARMS), range(start, end),
        sorted(frame.bin_index.unique())], names=['strategy_id', 'year', 'bin_index'])
    actual = pd.MultiIndex.from_frame(frame[['strategy_id', 'year', 'bin_index']])
    if len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError('incomplete or extraneous years/ages in paired outcome panel')


def cumulative_contrasts(age, receipts, config, value_column, *, infant_only=False):
    """Nested horizons, actual receipts, signed differences and defined ratios.

    The age universe must be exported by the caller. Infant-only deaths may
    explicitly use 12 bins; a zero event is a row, never an absent row.
    """
    rows = []
    keys = comparison_keys(age)
    for key, frame in age.groupby(keys, dropna=False):
        meta = dict(zip(keys, key))
        start = int(meta.get('start_year', config['start_year']))
        end_limit = int(config['end_year_exclusive'])
        validate_panel(frame, keys, [value_column], start, end_limit)
        matching = receipts
        for column, value in meta.items():
            if column in receipts:
                matching = matching[matching[column].eq(value)]
            elif column not in {'risk_draw_id', 'risk_scenario'}:
                raise ValueError(f'receipts missing pairing coordinate {column}')
        if matching.duplicated(['strategy_id', 'year']).any():
            raise ValueError('duplicate annual receipt rows')
        expected = pd.MultiIndex.from_product([sorted(ARMS-{'baseline'}), range(start, end_limit)])
        actual = pd.MultiIndex.from_frame(matching[['strategy_id', 'year']])
        if len(actual) != len(expected) or len(expected.difference(actual)):
            raise ValueError('incomplete annual receipt ledger')
        if not np.isfinite(matching.additional_receipts).all() or (matching.additional_receipts < 0).any():
            raise ValueError('receipts must be finite and nonnegative')
        groups = {'under1': frame.bin_index.lt(12)} if infant_only else {
            'all_ages': np.ones(len(frame), dtype=bool), 'under3m': frame.bin_index.lt(3),
            'under1': frame.bin_index.lt(12)}
        for years in config['cumulative_horizons_years']:
            end = start + years
            if end > end_limit:
                continue
            dose_rows = matching[matching.year.ge(start) & matching.year.lt(end)]
            dose = dose_rows.groupby('strategy_id').additional_receipts.sum()
            equal = np.isclose(dose.iloc[0], dose.iloc[1], rtol=1e-9, atol=1e-7)
            for group, mask in groups.items():
                totals = frame[mask & frame.year.ge(start) & frame.year.lt(end)].groupby('strategy_id')[value_column].sum()
                if set(totals.index) != ARMS:
                    raise ValueError('requested age group has no complete paired data')
                baseline = totals['baseline']
                common = dict(meta, horizon_years=years, start_year=start, end_year=end-1,
                    outcome=value_column, age_group=group, equal_actual_receipts=bool(equal))
                for strategy, burden in totals.items():
                    used = float(dose.get(strategy, 0.))
                    rows.append(dict(common, strategy_id=strategy, comparator='baseline',
                        cumulative_burden=burden, comparator_burden=baseline, averted=baseline-burden,
                        relative_reduction=(baseline-burden)/baseline if baseline else np.nan,
                        additional_receipts=used,
                        averted_per_1000_receipts=(baseline-burden)*1000/used if used else np.nan))
                m, a = totals['maternal_scaleup'], totals['adolescent_one_time_booster']
                rows.append(dict(common, strategy_id='maternal_scaleup', comparator='adolescent_one_time_booster',
                    cumulative_burden=m, comparator_burden=a, averted=a-m,
                    relative_reduction=(a-m)/a if a else np.nan,
                    additional_receipts=np.nan, averted_per_1000_receipts=np.nan))
    return pd.DataFrame(rows)


def nonnegative_linear_interval(intercept, slope):
    """Exact intersection in [0,1] where every a_i + r*b_i is nonnegative.

    An empty interval is (NaN, NaN). Endpoints are included, allowing weak
    dominance and isolated ties. This is a condition on r, not its probability.
    """
    a,b=np.asarray(intercept,float),np.asarray(slope,float)
    if a.shape!=b.shape or a.size==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite, nonempty matching arrays required')
    lower,upper=0.,1.
    for intercept_i,slope_i in zip(a.flat,b.flat):
        if slope_i>0:lower=max(lower,-intercept_i/slope_i)
        elif slope_i<0:upper=min(upper,-intercept_i/slope_i)
        elif intercept_i<0:return np.nan,np.nan
    return (float(lower),float(upper)) if lower<=upper else (np.nan,np.nan)


def mortality_rank_bounds(age, config, *, mild_risk_ratio=0., early_age_end_months=2):
    """Partial identification without a country-specific absolute fatality risk.

    For constant-within-age risks, D_A-D_M = q_y*(A_y-M_y)+q_o*(A_o-M_o),
    The age split defaults to2 months (Australia); use3 for England.
    Report the positive q_y/q_o threshold.
    The unrestricted bounds instead normalize risk weights to sum to one and
    allow one nonnegative weight per infant month and course. They are units
    of onsets per normalized risk weight, NOT deaths or uncertainty intervals.
    Additional bounds fix the declared mild/typical risk ratio and vary only
    monthly age risks, optionally requiring them to be nonincreasing with age.
    """
    if not np.isfinite(mild_risk_ratio) or not 0 <= mild_risk_ratio <= 1:
        raise ValueError('mild risk ratio must lie in [0, 1]')
    if early_age_end_months not in range(1,12):
        raise ValueError('early age endpoint must be an integer month from1 to11')
    early_age_end_months=int(early_age_end_months)
    rows = []
    keys = comparison_keys(age)
    for key, frame in age.groupby(keys, dropna=False):
        meta = dict(zip(keys, key))
        start = int(meta.get('start_year', config['start_year']))
        validate_panel(frame, keys, ['typical_onsets', 'mild_onsets'], start, config['end_year_exclusive'])
        infant = frame[frame.bin_index.lt(12)]
        if set(infant.bin_index) != set(range(12)):
            raise ValueError('all 12 infant months are required for mortality bounds')
        for h in config['cumulative_horizons_years']:
            if start+h > config['end_year_exclusive']:
                continue
            total = infant[infant.year.lt(start+h)].groupby(['strategy_id', 'bin_index'])[['typical_onsets', 'mild_onsets']].sum()
            delta = total.loc['adolescent_one_time_booster']-total.loc['maternal_scaleup']
            effective = delta.typical_onsets + mild_risk_ratio*delta.mild_onsets
            young=float(effective.iloc[:early_age_end_months].sum())
            older=float(effective.iloc[early_age_end_months:].sum())
            threshold = -older/young if young != 0 else np.nan
            low, high = float(delta.to_numpy().min()), float(delta.to_numpy().max())
            status = ('maternal_weakly_dominates' if low >= 0 else
                      'adolescent_weakly_dominates' if high <= 0 else 'risk_dependent')
            fixed_low,fixed_high=float(effective.min()),float(effective.max())
            # Extreme points of {q0>=...>=q11>=0, sum(q)=1} put equal
            # weight on the first k months. This is an explicit shape
            # sensitivity, not a claim that the data establish monotonicity.
            prefix_means=np.cumsum(effective.to_numpy())/np.arange(1,13)
            monotone_low,monotone_high=float(prefix_means.min()),float(prefix_means.max())
            course_conditions={}
            a=delta.typical_onsets.to_numpy();b=delta.mild_onsets.to_numpy()
            for name,aa,bb in [('monthly',a,b),('nonincreasing_age',np.cumsum(a),np.cumsum(b))]:
                for strategy,sign in [('maternal',1),('adolescent',-1)]:
                    rlo,rhi=nonnegative_linear_interval(sign*aa,sign*bb)
                    course_conditions[f'{strategy}_{name}_course_ratio_lower']=rlo
                    course_conditions[f'{strategy}_{name}_course_ratio_upper']=rhi
            def order(lower,upper):
                return ('maternal_weakly_dominates' if lower>=0 else
                    'adolescent_weakly_dominates' if upper<=0 else 'risk_dependent')
            rows.append(dict(meta, start_year=start, horizon_years=h, mild_risk_ratio=mild_risk_ratio,
                early_age_end_months=early_age_end_months,
                early_advantage=young, later_advantage=older,
                positive_early_late_risk_threshold=threshold if threshold > 0 else np.nan,
                maternal_preferred_above_threshold=bool(young > 0) if threshold > 0 else None,
                normalized_advantage_lower=low, normalized_advantage_upper=high,
                risk_independent_order=status,
                fixed_ratio_monthly_lower=fixed_low,fixed_ratio_monthly_upper=fixed_high,
                fixed_ratio_monthly_order=order(fixed_low,fixed_high),
                fixed_ratio_nonincreasing_age_lower=monotone_low,fixed_ratio_nonincreasing_age_upper=monotone_high,
                fixed_ratio_nonincreasing_age_order=order(monotone_low,monotone_high),
                fixed_ratio_two_group_order=order(min(young,older),max(young,older)),**course_conditions))
    return pd.DataFrame(rows)
