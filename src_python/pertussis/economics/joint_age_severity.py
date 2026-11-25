"""Joint age, reporting and clinical risk within annual model age bins.

Age is uniformly distributed inside each bin at the year midpoint, matching
AgeGrid's completed-age mapper. Infant bins keep the pre-existing within-month
clinical average: reporting and the requested age groups do not split those
bins. Optional segment modifiers support age-scoped unit costs and YLL.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from ..grid import OPEN_BIN_SPREAD_TO
from .evaluate import BAND_BOUNDS, _upper


def joint_risk_weights(grid, groups, logits, own_entries, donor_entries,
                       own_samples, donor_samples, use_own, bin_risk, *,
                       segment_multiplier=None, extra_boundaries=()):
    """Return draw × bin × group E[report_probability * clinical_risk * group].

    Preserve the existing bin-level own/donor choice and, where a selected
    source covers only part of a bin, use the existing bin-average risk for
    the uncovered portion. All risk boundaries, completed-age report bands,
    and group boundaries are integrated jointly over the continuous bin span.
    """
    own_samples=np.asarray(own_samples,float);donor_samples=np.asarray(donor_samples,float)
    bin_risk=np.asarray(bin_risk,float);use_own=np.asarray(use_own,bool)
    n=bin_risk.shape[0];ng=len(groups)
    if bin_risk.shape!=(n,grid.n_bins) or use_own.shape!=(grid.n_bins,):
        raise ValueError('clinical risk dimensions do not match age grid')
    if own_samples.shape!=(n,len(own_entries)) or donor_samples.shape!=(n,len(donor_entries)):
        raise ValueError('clinical entry samples do not match source entries')
    probs=expit(np.asarray(logits,float))
    if probs.shape!=(len(BAND_BOUNDS),) or not np.isfinite(probs).all():
        raise ValueError('six finite reporting logits required')
    group_ranges=list(groups.values())
    result=np.zeros((n,grid.n_bins,ng),float)
    for b in range(grid.n_bins):
        low=float(grid.lower_years[b]);up=float(grid.upper_years[b])
        if not np.isfinite(up):up=OPEN_BIN_SPREAD_TO
        if up<=low:raise ValueError('invalid age bin bounds')
        entries,samples=(own_entries,own_samples) if use_own[b] else (donor_entries,donor_samples)
        if b<12 and segment_multiplier is None:
            # Month-specific clinical risks already integrate any submonth
            # boundaries, while reporting and requested age are constant.
            ages=int(np.floor((low+up)/2))
            g=np.asarray([a<=ages<z for a,z in group_ranges],float)
            result[:,b,:]=bin_risk[:,b,None]*(probs[0]*g[None,:])
            continue
        cuts={low,up}
        cuts.update(float(a) for a in range(int(np.floor(low))+1,int(np.ceil(up))) if low<a<up)
        for x in extra_boundaries:
            if low<float(x)<up:cuts.add(float(x))
        for e in entries:
            for x in (float(e['age_start_years']),_upper(e['age_end_years'])):
                if low<x<up:cuts.add(x)
        boundaries=sorted(cuts)
        for a,z in zip(boundaries[:-1],boundaries[1:]):
            mid=(a+z)/2;age=int(np.floor(mid));fraction=(z-a)/(up-low)
            report_indices=[i for i,(x,y) in enumerate(BAND_BOUNDS) if x<=age<y]
            if len(report_indices)!=1:raise ValueError('ambiguous reporting band')
            gg=np.asarray([x<=age<y for x,y in group_ranges],float)
            hits=[j for j,e in enumerate(entries) if float(e['age_start_years'])<=mid<_upper(e['age_end_years'])]
            if len(hits)>1:raise ValueError('overlapping clinical severity entries')
            risk=samples[:,hits[0]] if hits else bin_risk[:,b]
            multiplier=np.asarray(1.0 if segment_multiplier is None else segment_multiplier(b,mid),dtype=float)
            if (multiplier.ndim != 0 and multiplier.shape != (n,)) or np.any(~np.isfinite(multiplier)) or np.any(multiplier<0):
                raise ValueError('invalid age-specific segment multiplier')
            result[:,b,:]+=(risk*multiplier)[:,None]*(fraction*probs[report_indices[0]]*gg[None,:])
    return result
