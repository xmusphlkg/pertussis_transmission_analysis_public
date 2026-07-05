from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_beta_grid_chunk(payload: dict[str, Any]) -> list[tuple[int, float]]:
    """Evaluate a chunk of beta-grid posterior values in an importable worker."""
    from src_python.simulation import run_bayesian_uncertainty as bayes

    indices = np.asarray(payload["indices"], dtype=int)
    grid_values = np.asarray(payload["grid_values"], dtype=float)
    calibrated_start = np.asarray(payload["calibrated_start"], dtype=float)
    runtime_base = payload["runtime_base"]
    observed = pd.DataFrame(payload["observed"])
    settings = payload["settings"]
    country = str(payload["country"])

    out: list[tuple[int, float]] = []
    for idx, x in zip(indices, grid_values):
        vector = calibrated_start.copy()
        vector[bayes.PARAMETER_INDEX_BY_SAMPLE["beta_S"]] = float(x)
        lp, _ = bayes._log_posterior(
            vector,
            runtime_base,
            observed,
            country,
            settings,
            _cache=None,
        )
        out.append((int(idx), float(lp)))
    return out
