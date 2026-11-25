"""Biological parameters and their conversion to kernel arrays.

``ModelParameters`` uses interpretable units (years, days, probabilities).
``kernel_arrays`` converts them to per-day rates, the per-compartment
susceptibility (``sigma``) and typical-course probability (``pi``) vectors,
and the transmission scale ``q`` implied by ``R0`` (see ``ngm.py``).

Protection is leaky: a state with relative susceptibility ``sigma`` is infected
at ``sigma * lambda``; the infection follows the typical (clinically classic,
reported at the full band rate) course with probability ``pi`` and the mild
course otherwise.  Vaccine effectiveness against typical disease is therefore
``1 - sigma * pi``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np

from ..states import (
    D1,
    D2,
    MV1,
    MV2,
    NC, TW1, TW2, AP11, AP22, WP11, WP22,
    MATERNAL_STAGES,
    PA1,
    PA2,
    PW1,
    PW2,
    SA,
    SW,
    T1,
    T2,
    M,
    S0,
)
from . import kernel as K

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class ModelParameters:
    # Transmission.
    R0: float = 10.0
    seasonal_amplitude: float = 0.15
    seasonal_peak: float = 0.6            # fraction of the calendar year
    npi_scale: float = 1.0                # multiplies the NPI reduction timeline
    importation_per_million_per_year: float = 10.0
    rho_mild: float = 0.5
    # Natural history (days).
    latent_days: float = 8.0
    infectious_days_typical: float = 21.0
    infectious_days_mild: float = 14.0
    # Immunity durations (years; Erlang-2 means for staged states).
    natural_immunity_years: float = 30.0
    boosting_kappa: float = 2.0
    wP_years: float = 10.0
    aP_years: float = 6.0
    tdap_years: float = 3.0
    maternal_days: float = 90.0
    maternal_waning_stages: int = 1
    d1_protection_years: float = 1.0
    d2_protection_years: float = 2.0
    # Structural switch (not estimated): "erlang2" or "exponential" waning of
    # wP, aP and Tdap protection with the same mean duration.
    vaccine_waning_shape: str = "erlang2"
    # Relative susceptibility (sigma) and typical-course probability (pi).
    sigma_maternal: float = 0.3
    pi_maternal: float = 0.3
    sigma_d1: float = 1.0
    pi_d1: float = 0.5
    sigma_d2: float = 1.0
    pi_d2: float = 0.2
    sigma_wP: float = 0.17
    sigma_aP: float = 0.3
    sigma_tdap: float = 0.3
    pi_protected: float = 0.15
    pi_wP_waned: float = 0.3
    pi_aP_waned: float = 0.5
    # Demography.
    nudge_years: float = 0.25

    def with_updates(self, **values: float) -> "ModelParameters":
        unknown = set(values) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"unknown parameters: {sorted(unknown)}")
        return replace(self, **values)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def sigma_pi(self) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.zeros(NC)
        pi = np.zeros(NC)
        for state, s, p in (
            (M, self.sigma_maternal, self.pi_maternal),
            (MV1, self.sigma_maternal, self.pi_maternal),
            (MV2, self.sigma_maternal, self.pi_maternal),
            (S0, 1.0, 1.0),
            (D1, self.sigma_d1, self.pi_d1),
            (D2, self.sigma_d2, self.pi_d2),
            (PW1, self.sigma_wP, self.pi_protected),
            (PW2, self.sigma_wP, self.pi_protected),
            (PA1, self.sigma_aP, self.pi_protected),
            (PA2, self.sigma_aP, self.pi_protected),
            (T1, self.sigma_tdap, self.pi_protected),
            (T2, self.sigma_tdap, self.pi_protected),
            (SW, 1.0, self.pi_wP_waned),
            (SA, 1.0, self.pi_aP_waned),
        ):
            if not (0.0 <= s and 0.0 <= p <= 1.0):
                raise ValueError("sigma must be >= 0 and pi within [0, 1]")
            sigma[state] = s
            pi[state] = p
        for stage in MATERNAL_STAGES[1:]:
            for state in stage:
                sigma[state] = self.sigma_maternal
                pi[state] = self.pi_maternal
        sigma[TW1:TW2+1] = self.sigma_tdap
        sigma[AP11:AP22+1] = min(self.sigma_aP, self.sigma_tdap)
        sigma[WP11:WP22+1] = min(self.sigma_wP, self.sigma_tdap)
        pi[TW1:] = self.pi_protected
        return sigma, pi

    def kernel_vector(self, q: float) -> np.ndarray:
        positive = (
            "latent_days", "infectious_days_typical", "infectious_days_mild",
            "natural_immunity_years", "wP_years", "aP_years", "tdap_years",
            "maternal_days", "d1_protection_years", "d2_protection_years", "nudge_years",
        )
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        P = np.zeros(K.N_PARAMS)
        P[K.P_Q] = q
        P[K.P_SEAS_AMP] = self.seasonal_amplitude
        P[K.P_SEAS_PEAK] = self.seasonal_peak
        P[K.P_NPI_EPS] = self.npi_scale
        P[K.P_IOTA] = self.importation_per_million_per_year / 1e6 / DAYS_PER_YEAR
        P[K.P_LATENT] = 1.0 / self.latent_days
        P[K.P_REC_T] = 1.0 / self.infectious_days_typical
        P[K.P_REC_M] = 1.0 / self.infectious_days_mild
        P[K.P_RHO_M] = self.rho_mild
        P[K.P_OMEGA_M] = 1.0 / self.maternal_days
        if self.maternal_waning_stages not in (1, 2, 4):
            raise ValueError("maternal_waning_stages must be 1, 2 or 4")
        P[K.P_MAT_STAGES] = self.maternal_waning_stages
        if self.vaccine_waning_shape not in {"erlang2", "exponential"}:
            raise ValueError("vaccine_waning_shape must be erlang2 or exponential")
        stages = 1.0 if self.vaccine_waning_shape == "exponential" else 2.0
        P[K.P_WP_STAGE] = stages / (self.wP_years * DAYS_PER_YEAR)
        P[K.P_AP_STAGE] = stages / (self.aP_years * DAYS_PER_YEAR)
        P[K.P_T_STAGE] = stages / (self.tdap_years * DAYS_PER_YEAR)
        P[K.P_VAC_SHAPE] = stages
        P[K.P_NAT_STAGE] = 2.0 / (self.natural_immunity_years * DAYS_PER_YEAR)
        P[K.P_KAPPA] = self.boosting_kappa
        P[K.P_D1_WANE] = 1.0 / (self.d1_protection_years * DAYS_PER_YEAR)
        P[K.P_D2_WANE] = 1.0 / (self.d2_protection_years * DAYS_PER_YEAR)
        P[K.P_NUDGE] = 1.0 / (self.nudge_years * DAYS_PER_YEAR)
        return P
