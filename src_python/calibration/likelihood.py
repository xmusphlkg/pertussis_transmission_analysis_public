from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def negative_binomial_nll(observed: np.ndarray, mean: np.ndarray, dispersion: float) -> float:
    observed = np.asarray(observed, dtype=float)
    mean = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    r = max(float(dispersion), 1e-9)
    p = r / (r + mean)
    log_prob = (
        gammaln(observed + r)
        - gammaln(r)
        - gammaln(observed + 1.0)
        + r * np.log(p)
        + observed * np.log1p(-p)
    )
    return float(-np.sum(log_prob))
