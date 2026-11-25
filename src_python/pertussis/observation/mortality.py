"""Sparse death observations conditional on latent disease-course exposure.

A death observation identifies q * death_ascertainment * latent_exposure.
It does not separately identify these factors or a vaccine-specific residual
severity effect. Censored counts retain the published interval; surveillance
and registry observations must be fitted as alternative streams, not added.
"""
import numpy as np
from scipy.special import gammaln, logsumexp, gammainc, xlogy
from scipy.stats import gamma, poisson


def _bounds(lower, upper):
    if (not np.isfinite([lower, upper]).all() or lower < 0 or upper < lower
            or lower != int(lower) or upper != int(upper)):
        raise ValueError('death count bounds must be ordered nonnegative integers')
    return np.arange(int(lower), int(upper)+1)


def interval_poisson_loglik(lower, upper, mean):
    counts = _bounds(lower, upper)
    if not np.isfinite(mean) or mean < 0:
        raise ValueError('expected deaths must be finite and nonnegative')
    return float(logsumexp(poisson.logpmf(counts, mean)))


def risk_posterior_draws(lower, upper, exposure, *, ascertainment=1.,
                         size=256, seed=0, prior_shape=.5, prior_rate=0.):
    """Truncated Gamma mixture for q in (0,1), given interval-censored deaths.

    Prior density proportional to q**(prior_shape-1)*exp(-prior_rate*q).
    Default is a proper power prior on [0,1]; for known exposure its shape
    matches the Poisson Jeffreys prior. The interval likelihood is the SUM of
    Poisson probabilities, not a midpoint count and not a uniform draw of k.
    Death ascertainment and the mild/typical risk ratio are fixed sensitivity
    assumptions, not estimated by these sparse data. Exposure refers to
    modelled typical + ratio*mild onsets in the same geography/age/time window.
    """
    counts = _bounds(lower, upper)
    if (not np.isfinite([exposure, ascertainment, prior_shape, prior_rate]).all()
            or exposure <= 0 or not 0 < ascertainment <= 1 or prior_shape <= 0
            or prior_rate < 0 or size < 1 or int(size) != size):
        raise ValueError('positive exposure/shape/size and valid ascertainment/rate required')
    e = exposure*ascertainment
    rate = e + prior_rate
    shapes = counts + prior_shape
    masses = gammainc(shapes, rate)
    logw = (counts*np.log(e) - gammaln(counts+1) + gammaln(shapes)
            - shapes*np.log(rate) + np.log(masses))
    weights = np.exp(logw-logsumexp(logw))
    rng = np.random.default_rng(seed)
    components = rng.choice(len(counts), size=int(size), p=weights)
    uniforms = rng.uniform(np.nextafter(0., 1.), 1., int(size))
    draws = gamma.ppf(uniforms*masses[components], shapes[components], scale=1/rate)
    if not np.isfinite(draws).all() or (draws <= 0).any() or (draws > 1).any():
        raise ArithmeticError('invalid truncated mortality posterior draws')
    return draws


def joint_risk_posterior_draws(lower, upper, exposure, *, total_lower, total_upper,
                               ascertainment=1., size=256, seed=0,
                               prior_shape=.5, prior_rate=0., return_counts=False):
    """Joint censored-age risks with a published sum constraint.

    Each permissible integer age-count vector contributes its joint Poisson
    probability. The total is a constraint on those same deaths, not another
    independent likelihood term. Marginal age risks become dependent.
    """
    from itertools import product
    lo=np.atleast_1d(lower);hi=np.atleast_1d(upper);ex=np.atleast_1d(exposure).astype(float)
    if lo.shape!=hi.shape or lo.shape!=ex.shape or lo.ndim!=1 or len(lo)==0:
        raise ValueError('aligned one-dimensional count bounds and exposures required')
    _bounds(total_lower,total_upper)
    choices=[_bounds(l,h) for l,h in zip(lo,hi)]
    if np.prod([len(c) for c in choices])>100000:
        raise ValueError('censored count grid too large for exact enumeration')
    if (not np.isfinite(ex).all() or np.any(ex<=0)
            or not np.isfinite([ascertainment,prior_shape,prior_rate]).all()
            or not 0<ascertainment<=1 or prior_shape<=0 or prior_rate<0
            or size<1 or int(size)!=size):
        raise ValueError('invalid mortality exposure or prior')
    counts=np.asarray([c for c in product(*choices) if total_lower<=sum(c)<=total_upper],dtype=int)
    if counts.size==0:raise ValueError('age count intervals contradict their total')
    e=ex*ascertainment;rate=e+prior_rate;shape=counts+prior_shape
    mass=gammainc(shape,rate)
    with np.errstate(divide='ignore'):
        logw=(counts*np.log(e)-gammaln(counts+1)+gammaln(shape)-shape*np.log(rate)+np.log(mass)).sum(axis=1)
    norm=logsumexp(logw)
    if not np.isfinite(norm):raise ArithmeticError('all constrained count components underflow')
    rng=np.random.default_rng(seed)
    components=rng.choice(len(counts),size=int(size),p=np.exp(logw-norm))
    u=rng.uniform(np.nextafter(0.,1.),1.,(int(size),len(ex)))
    q=gamma.ppf(u*mass[components],shape[components],scale=1/rate)
    if not np.isfinite(q).all() or (q<=0).any() or (q>1).any():
        raise ArithmeticError('invalid constrained risk draws')
    return (q,counts[components]) if return_counts else q


def coarsened_two_age_risk_draws(counts, exposures, *, ascertainment=1., size=256,
                                 seed=0, prior_shape=.5, prior_rate=0.):
    """Exact two-age risk posterior when each year's deaths are only a total.

    Y_t ~ Poisson(a * (E_t0*q0 + E_t1*q1)). The unknown age of death is
    integrated out, never imputed as an observed zero in the older group.
    Expanding the product of count polynomials yields a finite mixture of
    independent truncated Gamma pairs; dependence is retained via the shared
    mixture component. Identifiability depends on changes in exposure ratios.
    Independent q^(shape-1)*exp(-rate*q) priors are restricted to (0,1).
    """
    y=np.atleast_1d(counts).astype(float); e=np.asarray(exposures,dtype=float)
    if (y.ndim!=1 or len(y)==0 or e.shape!=(len(y),2)
            or not np.isfinite(y).all() or (y<0).any() or (y!=np.floor(y)).any()
            or not np.isfinite(e).all() or (e<0).any() or (e.sum(axis=0)<=0).any()
            or not np.isfinite([ascertainment,prior_shape,prior_rate]).all()
            or not 0<ascertainment<=1 or prior_shape<=0 or prior_rate<0
            or size<1 or int(size)!=size):
        raise ValueError('integer deaths, two nonnegative exposure columns and valid prior required')
    if y.sum()>2000:raise ValueError('exact mixture intended for sparse death observations')
    e=e*ascertainment; logcoef=np.array([0.])
    for d, row in zip(y.astype(int),e):
        k=np.arange(d+1)
        term=xlogy(k,row[0])+xlogy(d-k,row[1])-gammaln(k+1)-gammaln(d-k+1)
        updated=np.full(len(logcoef)+d,-np.inf)
        for j,v in enumerate(term):updated[j:j+len(logcoef)]=np.logaddexp(updated[j:j+len(logcoef)],logcoef+v)
        logcoef=updated
    n=int(y.sum());k=np.arange(n+1)
    shape=np.column_stack((k+prior_shape,n-k+prior_shape));rate=e.sum(axis=0)+prior_rate
    mass=gammainc(shape,rate)
    with np.errstate(divide='ignore'):
        logw=logcoef+(gammaln(shape)-shape*np.log(rate)+np.log(mass)).sum(axis=1)
    norm=logsumexp(logw)
    if not np.isfinite(norm):raise ValueError('deaths incompatible with zero exposure or numerical underflow')
    rng=np.random.default_rng(seed)
    selected=rng.choice(n+1,size=int(size),p=np.exp(logw-norm))
    u=rng.uniform(np.nextafter(0.,1.),1.,(int(size),2))
    q=gamma.ppf(u*mass[selected],shape[selected],scale=1/rate)
    if not np.isfinite(q).all() or (q<=0).any() or (q>1).any():
        raise ArithmeticError('invalid coarsened risk draws')
    return q


def matched_infant_exposure(onsets, observation, *, mild_risk_ratio=0.):
    """Validate source coverage; approximate calendar deaths by onset cohorts.

    This annual implementation requires an explicit zero-delay assumption for
    calendar-year deaths. It never silently treats diagnosis dates as onsets.
    """
    if observation['geographic_scope'] != observation['profile']:
        raise ValueError('mortality observation geography must match model profile')
    if set(onsets.profile) != {observation['profile']} or set(onsets.strategy_id) != {'baseline'}:
        raise ValueError('one matching baseline profile is required')
    if observation['year_mapping'] not in {
            'zero_delay_calendar_to_onset_assumption',
            'zero_delay_confirmation_to_onset_assumption'}:
        raise ValueError('explicit death-year to onset-year mapping is required')
    lo, hi = observation['age_start_months'], observation['age_end_months_exclusive']
    if not (0 <= lo < hi <= 12) or lo != int(lo) or hi != int(hi):
        raise ValueError('infant age interval must align with monthly model bins')
    start, end = int(observation['start_year']), int(observation['end_year_exclusive'])
    frame = onsets[onsets.year.ge(start) & onsets.year.lt(end) &
                   onsets.bin_index.ge(lo) & onsets.bin_index.lt(hi)]
    if (end <= start or len(frame) != (end-start)*(hi-lo) or
            frame.duplicated(['year', 'bin_index']).any()):
        raise ValueError('incomplete or duplicated mortality exposure window')
    values = frame[['typical_onsets', 'mild_onsets']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or not 0 <= mild_risk_ratio <= 1:
        raise ValueError('invalid onsets or mild risk ratio')
    return float(frame.typical_onsets.sum()+mild_risk_ratio*frame.mild_onsets.sum())
