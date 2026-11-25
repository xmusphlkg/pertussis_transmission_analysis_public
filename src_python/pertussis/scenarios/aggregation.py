"""Age-group and time-window aggregation of annual age-bin onsets.

One definition of age groups and windows for every outcome, so the infant,
adolescent and whole-population results are read off the same tables.  Model
cohort bins hold two completed ages, so groups are mapped with the Lexis weights
of ``grid.band_matrix`` (mid-year); this is approximate at the 10- and 20-year
edges and says so in the group labels.
"""
import numpy as np
import pandas as pd

from ..grid import AgeGrid, band_matrix

AGE_GROUPS = {'infant_0': (0., 1.), 'child_1_9': (1., 10.), 'adolescent_10_19': (10., 20.),
              'adult_20plus': (20., np.inf), 'all_ages': (0., np.inf)}
WINDOWS_20Y = {'short_2027_2031': (2027, 2031), 'medium_2032_2036': (2032, 2036),
               'long_2037_2046': (2037, 2046), 'all_2027_2046': (2027, 2046)}
WINDOWS_30Y_EXTRA = {'extended_2047_2056': (2047, 2056), 'all_2027_2056': (2027, 2056)}
VALUE_COLUMNS = ('typical_onsets', 'mild_onsets', 'all_onsets', 'infections')


def age_group_matrix(grid=None, groups=AGE_GROUPS):
    grid = grid or AgeGrid()
    return grid, list(groups), band_matrix(grid, [groups[g] for g in groups], 0.5)


def aggregate_onsets(onsets, *, windows=WINDOWS_20Y, groups=AGE_GROUPS, keys=('profile', 'strategy_id')):
    """Sum onsets over (age group x window); returns long table with undiscounted counts.

    ``onsets`` is the output of ``annual_age_onsets`` with strategy columns.
    Every year of each requested window must be present for every key group.
    """
    grid, names, mat = age_group_matrix(None, groups)
    if onsets.empty or onsets[list(keys)].isna().any().any():
        raise ValueError('nonempty onsets with nonmissing grouping keys are required')
    rows = []
    for key, frame in onsets.groupby(list(keys), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        years = set(frame.year.unique())
        for wname, (y0, y1) in windows.items():
            need = set(range(y0, y1 + 1))
            if not need <= years:
                raise ValueError(f'window {wname} needs years {sorted(need - years)} not in the projection')
            part = frame[frame.year.between(y0, y1)]
            index = pd.MultiIndex.from_product([range(y0, y1 + 1), range(grid.n_bins)],
                                               names=['year', 'bin_index'])
            cells = pd.MultiIndex.from_frame(part[['year', 'bin_index']])
            if cells.has_duplicates or len(cells) != len(index) or not cells.isin(index).all():
                raise ValueError(f'window {wname} requires exactly one onset row per year and age bin')
            values = part[list(VALUE_COLUMNS)].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('onset values must be finite and nonnegative')
            by_bin = part.groupby('bin_index')[list(VALUE_COLUMNS)].sum().reindex(range(grid.n_bins))
            if by_bin.isna().any().any():
                raise ValueError('incomplete age-bin onsets')
            sums = mat.T @ by_bin.to_numpy(float)
            for gi, gname in enumerate(names):
                rows.append(dict(zip(keys, key), window=wname, age_group=gname,
                                 **dict(zip(VALUE_COLUMNS, sums[gi]))))
    return pd.DataFrame(rows)
