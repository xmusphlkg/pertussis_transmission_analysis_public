"""Basic reproduction number and the transmission scale ``q``.

With force of infection ``lambda_g = q * sum_h C_gh * I_h / N_h`` and a fully
naive population (every infection typical, mean infectious period ``D``), the
next-generation matrix is ``K_gh = q * D * C_gh * N_g / N_h``.  It is similar to
``C`` (``K = q D diag(N) C diag(N)^-1``), so ``R0 = q * D * rho(C)`` and
``q = R0 / (D * rho(C))`` without needing the population.  ``C`` is the total
(home + non-home) per-participant contact matrix of the reference year without
NPI reductions; seasonality averages to one.
"""

from __future__ import annotations

import numpy as np


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix, dtype=float)))))


def q_from_r0(r0: float, contacts: np.ndarray, infectious_days: float) -> float:
    if r0 <= 0 or infectious_days <= 0:
        raise ValueError("R0 and infectious period must be positive")
    rho = spectral_radius(contacts)
    if rho <= 0:
        raise ValueError("contact matrix has zero spectral radius")
    return float(r0 / (infectious_days * rho))


def r0_from_q(q: float, contacts: np.ndarray, infectious_days: float) -> float:
    return float(q * infectious_days * spectral_radius(contacts))
