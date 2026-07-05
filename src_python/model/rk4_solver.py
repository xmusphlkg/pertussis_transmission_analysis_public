"""Fast fixed-step RK4 solver for MCMC likelihood evaluation.

Strategy
--------
scipy.integrate.solve_ivp (RK45) has significant Python-level overhead per
step: adaptive step-size control, dense output bookkeeping, and repeated
Python→C boundary crossings.  For MCMC we only need the *final* state (or a
coarse annual snapshot) and can tolerate a small fixed-step error in exchange
for a 5-10× wall-clock speedup.

This module provides:
  - ``rk4_solve_mcmc``: pure-Python RK4 that calls the existing ``rhs``
    function (including full SIRWS dynamics).  Drop-in replacement for
    solve_ivp in the MCMC log-likelihood.
  - ``solve_rk4``: lightweight solve_ivp-compatible wrapper.
  - ``apply_mcmc_solver_overrides``: injects fast-solver settings into config
    for MCMC runs (currently uses scipy RK45 for best performance).

Step-size selection
-------------------
The ODE has a fastest time-scale of ~1/gamma_sym ≈ 1/21 days⁻¹ ≈ 21 days.
A fixed step of 1 day gives ~21 steps per fastest time-scale, which is well
within RK4 stability for this system.  We use dt=1.0 day for the analysis
window and dt=2.0 days for the burn-in (where we only need the equilibrium).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src_python.model.compartments import StateIndex
from src_python.model.ode_system import rhs
from src_python.model.parameters import PreparedParameters


# ---------------------------------------------------------------------------
# Pure-Python RK4 (calls existing rhs, no Numba dependency)
# ---------------------------------------------------------------------------

def rk4_solve_mcmc(
    params: PreparedParameters,
    index: StateIndex,
    y0: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float = 1.0,
    t_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 integrator using the existing Python rhs.

    Returns
    -------
    t_out : 1-D array of output times
    y_out : 2-D array (n_states, n_times)
    """
    t = t_start
    y = y0.copy()
    n_steps = max(1, int(np.ceil((t_end - t_start) / dt)))
    actual_dt = (t_end - t_start) / n_steps

    if t_eval is None:
        # Only return the final state
        for _ in range(n_steps):
            k1 = rhs(t, y, params, index)
            k2 = rhs(t + 0.5 * actual_dt, y + 0.5 * actual_dt * k1, params, index)
            k3 = rhs(t + 0.5 * actual_dt, y + 0.5 * actual_dt * k2, params, index)
            k4 = rhs(t + actual_dt, y + actual_dt * k3, params, index)
            y = y + (actual_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y = np.maximum(y, 0.0)
            t += actual_dt
        return np.array([t_end]), y[:, np.newaxis]

    # Return states at requested t_eval points
    eval_set = set(float(te) for te in t_eval)
    t_out: list[float] = []
    y_out: list[np.ndarray] = []

    for _ in range(n_steps):
        k1 = rhs(t, y, params, index)
        k2 = rhs(t + 0.5 * actual_dt, y + 0.5 * actual_dt * k1, params, index)
        k3 = rhs(t + 0.5 * actual_dt, y + 0.5 * actual_dt * k2, params, index)
        k4 = rhs(t + actual_dt, y + actual_dt * k3, params, index)
        y = y + (actual_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y = np.maximum(y, 0.0)
        t += actual_dt
        if any(abs(t - te) < 0.5 * actual_dt for te in eval_set):
            t_out.append(t)
            y_out.append(y.copy())

    if not t_out:
        t_out = [t_end]
        y_out = [y]

    return np.array(t_out), np.column_stack(y_out)


# ---------------------------------------------------------------------------
# Lightweight solve_ivp-compatible wrapper for MCMC use
# ---------------------------------------------------------------------------

class _RK4Result:
    """Minimal result object matching scipy solve_ivp interface."""
    __slots__ = ("t", "y", "success", "message")

    def __init__(self, t: np.ndarray, y: np.ndarray):
        self.t = t
        self.y = y
        self.success = True
        self.message = "RK4 fixed-step integration completed."


def solve_rk4(
    params: PreparedParameters,
    index: StateIndex,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    dt: float = 1.0,
) -> _RK4Result:
    """Drop-in replacement for solve_ivp using fixed-step RK4.

    Parameters
    ----------
    params, index : model parameters and state index
    y0 : initial state vector
    t_span : (t_start, t_end)
    t_eval : optional output times (if None, only final state returned)
    dt : fixed step size in days (default 1.0)
    """
    t_arr, y_arr = rk4_solve_mcmc(params, index, y0, t_span[0], t_span[1], dt=dt, t_eval=t_eval)
    return _RK4Result(t_arr, y_arr)


# ---------------------------------------------------------------------------
# Config helper: inject fast-solver settings for MCMC
# ---------------------------------------------------------------------------

MCMC_SOLVER_OVERRIDES: dict[str, Any] = {
    "burn_in_years": 3,        # short burn-in: equilibrium reached in ~2-3y
    "output_time_step": 30,    # monthly output during MCMC (annual cases only needed)
    "solver_method": "RK45",   # scipy RK45 is faster than our Python RK4
    "rtol": 1e-3,              # relaxed tolerance for MCMC (sufficient for likelihood)
    "atol": 1e-5,
}


def apply_mcmc_solver_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of config with fast MCMC solver settings applied."""
    out = deepcopy(config)
    sim = out.setdefault("simulation", {})
    for key, value in MCMC_SOLVER_OVERRIDES.items():
        sim[key] = value
    return out
