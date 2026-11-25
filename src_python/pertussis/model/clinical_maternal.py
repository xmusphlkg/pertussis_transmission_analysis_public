"""Conditional mapping from passive-protection mechanisms to clinical age windows.

Uses the forward kernel's state transitions under shared constant exposure.
The clinical summary constrains a disease-risk combination, not the separately
identified susceptibility, symptom or duration parameters. Infant-vaccinated
cases are excluded beginning seven days after their first dose, as in Walker.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from ..inputs.synthetic import synthetic_inputs
from ..states import NC, NA, M, S0, DOSE1, PRODUCT_AP
from .kernel import rhs, apply_dose


@lru_cache(maxsize=1)
def _assay_inputs():
    inp=synthetic_inputs(first_year=2020,n_years=1,spinup_years=0,nudging=False,vaccination=False)
    inp.births_per_day[:]=0
    inp.contacts_home[:]=0
    inp.contacts_nonhome[:]=0
    inp.maternal_effective[:]=0
    return inp


def clinical_generator(params, exposure_per_day=1e-5):
    """Generator including the same latent onset and passive stages as the kernel."""
    if not np.isfinite(exposure_per_day) or exposure_per_day <= 0:
        raise ValueError('exposure must be finite and positive')
    inp=_assay_inputs(); t=inp.kernel_tables(); nb=inp.grid.n_bins
    p=replace(params,importation_per_million_per_year=exposure_per_day*1e6*365.25)
    P=p.kernel_vector(0.); sig,pi=p.sigma_pi()
    A=np.zeros((NC+1,NC+1))
    for j in range(NC):
        y=np.zeros((nb,NC)); y[0,j]=1
        dy=np.zeros_like(y); da=np.zeros((nb,NA))
        ng=t['c_home'].shape[1]
        rhs(y,dy,da,0.,0,P,sig,pi,t['births'],t['mat_eff'],t['npi_r'],t['imp_mult'],
            t['season_on'],t['year_frac'],t['month_idx'],t['year_idx'],t['nudge_target'],
            t['primary_product'],t['c_home'],t['c_non'],t['group'],t['age_rate'],
            t['contact_mult'],np.zeros(nb),np.zeros(ng),np.zeros(ng),np.zeros(ng))
        A[:NC,j]=dy[0]; A[NC,j]=da[0,0]
    return A


def clinical_window_ve(params, *, windows=(60.,90.), effective_share=.9,
                       first_dose_day=56., infant_coverage=.95, exposure_per_day=1e-5):
    if not (0<=effective_share<=1 and 0<=infant_coverage<=1):
        raise ValueError('shares must be in [0,1]')
    if first_dose_day <= 0 or min(windows)<=0:
        raise ValueError('age windows and first dose day must be positive')
    A=clinical_generator(params,exposure_per_day)
    z=np.zeros((NC+1,2)); z[S0,0]=1; z[M,1]=effective_share; z[S0,1]=1-effective_share
    before=expm(A*first_dose_day)@z
    dosed=before.copy()
    for j in range(2):
        y=dosed[:NC,j].reshape(1,NC).copy()
        apply_dose(y,0,DOSE1,1.,PRODUCT_AP)
        dosed[:NC,j]=y[0]
    output={}
    for T in windows:
        uncensored=(expm(A*T)@z)[NC]
        if T<=first_dose_day:
            receipt=uncensored
        else:
            # Case eligibility in the clinical study, not permanent loss of
            # protection or a new population-wide vaccination schedule.
            receipt=(expm(A*(min(T,first_dose_day+7)-first_dose_day))@dosed)[NC]
        risk=(1-infant_coverage)*uncensored+infant_coverage*receipt
        output[float(T)]=float(1-risk[1]/risk[0])
    return output


def fit_clinical_target(base, *, target=.862, sigma=1., window=60., **assay):
    """Fit pi only conditional on duration, stages, susceptibility and assay."""
    if not 0<=target<=1 or not 0<sigma<=1:
        raise ValueError('target and sigma must be probabilities, sigma positive')
    def error(pi):
        return clinical_window_ve(replace(base,sigma_maternal=sigma,pi_maternal=pi),
            windows=(window,),**assay)[window]-target
    hi,lo=error(0.),error(1.)
    if hi < 0 or lo > 0:
        return None
    pi=brentq(error,0.,1.,xtol=1e-11)
    return replace(base,sigma_maternal=sigma,pi_maternal=pi)
