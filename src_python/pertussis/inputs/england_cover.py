"""Explicit England COVER sensitivity, preserving the inherited-proxy baseline."""
import pandas as pd
from .england import DATA, load_england_programme
from .programme import CoverageSeries, SeriesSegment


def load_england_cover_programme():
    p=load_england_programme()
    frame=pd.read_csv(DATA/'england_cover_candidate_annual.csv')
    p.sources['england_cover_cohorts']={'citation':'UKHSA COVER England milestone cohorts mapped to scheduled dose dates',
        'url':str(DATA/'england_cover_candidate_annual.csv'),'ledger':'health_outcomes_20261026'}
    for name,group in frame.groupby('series'):
        old=p.coverage_series[name]
        mapped=group.set_index('year').coverage
        segments=[]
        for year in range(1970,2051):
            value=float(mapped.loc[year]) if year in mapped.index else old.value(year,p)
            segments.append(SeriesSegment(year,year,'value',value,('england_cover_cohorts',),'D'))
        p.coverage_series[name]=CoverageSeries(name,segments,p.iso3)
    # A different geography's first-dose proxy can lie below England's
    # observed completion rate. Bound dose1 below by observed completion;
    # never reduce the observed third-dose completion to match that proxy.
    first=p.coverage_series['dtp1']
    p.coverage_series['dtp1']=CoverageSeries('dtp1',[
        SeriesSegment(y,y,'value',max(first.value(y,p),p.coverage('dtp3',y)),('england_cover_cohorts',),'D')
        for y in range(1970,2051)],p.iso3)
    p.status='development_England_COVER_cohort_mapping'
    return p


def build_england_cover_inputs(settings):
    from .bundle import build_inputs
    inputs,audit=build_inputs('England',settings,programme=load_england_cover_programme())
    inputs.metadata.update(childhood_coverage_variant='england_cover_cohort',
        childhood_coverage_proxy='England COVER completion and preschool; dose1 max(GBR proxy, England dose3), dose2 derived; 18m assumed',
        childhood_coverage_mapping='milestone cohorts shifted to scheduled 4m/40m, day-weighted years; last cohort carried forward')
    return inputs,audit
