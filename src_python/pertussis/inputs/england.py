"""England regional inputs; UK mixing/NPI and early uptake remain explicit proxies."""
from copy import deepcopy
import numpy as np
import pandas as pd
from ..config import REPO_ROOT, CONFIG_DIR
from .demography import Demography
from .programme import CoverageSeries, SeriesSegment, load_programme

DATA=REPO_ROOT/'data/processed/health_outcomes_20261026'


def load_england_demography():
    pop=pd.read_csv(DATA/'england_population.csv')
    births=pd.read_csv(DATA/'england_births.csv')
    if set(pop.geography)!= {'England'} or set(births.geography)!= {'England'}:
        raise ValueError('England demography contains another geography')
    years=np.arange(1971,2051)
    matrix=pop.pivot(index='year',columns='age',values='population').reindex(index=years,columns=range(101))
    b=births.set_index('year').births.reindex(years)
    return Demography('GB-ENG',years,matrix.to_numpy(float),b.to_numpy(float))


def load_england_programme():
    p=deepcopy(load_programme(CONFIG_DIR/'programmes/United_Kingdom.yaml'))
    p.country='England';p.status='development_regional_history'
    # Inherited GBR WUENIC is a declared childhood-coverage proxy; never label
    # those observations England-specific. Maternal observations are England.
    monthly=pd.read_csv(DATA/'england_maternal_coverage_monthly.csv')
    dates=pd.to_datetime(monthly.period_start)
    observed=monthly.assign(year=dates.dt.year,days=(pd.to_datetime(monthly.period_end)-dates).dt.days)
    annual=observed.groupby('year').apply(lambda g:np.average(g.coverage,weights=g.days),include_groups=False)
    p.sources['england_immform']={'citation':'UKHSA monthly prenatal coverage April2016–March2026',
        'url':str(DATA/'england_maternal_coverage_monthly.csv'),'ledger':'health_outcomes_20261026'}
    old=p.coverage_series['maternal']
    segments=[SeriesSegment(1970,2011,'value',0.,('england_immform',),'A')]
    for year in range(2012,2051):
        if year<2016:value=old.value(year,p);grade='D'
        elif year<=2025:value=float(annual.loc[year]);grade='A' if year>2016 else 'D'
        else:value=.73;grade='D'
        segments.append(SeriesSegment(year,year,'value',value,('england_immform',),grade))
    p.coverage_series['maternal']=CoverageSeries('maternal',segments,p.iso3)
    return p


def apply_observed_maternal(inputs,programme):
    """Birth-month uptake directly maps to entering newborns; no added lag."""
    rows=pd.read_csv(DATA/'england_maternal_coverage_monthly.csv')
    for r in rows.itertuples():
        start=np.datetime64(r.period_start,'D');end=np.datetime64(r.period_end,'D')
        period=next((p for p in programme.maternal if p.active(pd.Timestamp(r.period_start).date())),None)
        if period is None:continue
        mask=(inputs.calendar.dates>=start)&(inputs.calendar.dates<end)
        inputs.maternal_effective[mask]=r.coverage*period.effective_share
