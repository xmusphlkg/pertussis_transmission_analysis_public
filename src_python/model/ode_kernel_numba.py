"""Numba-accelerated ODE inner loops for the pertussis transmission model.

Strategy: Rather than reimplementing the entire ODE RHS in Numba (which would
diverge from the Python reference and miss features), we accelerate only the
computationally intensive inner loops:
  1. Force of infection computation (contact matrix × infectious pressure)
  2. Demographic aging (compartment-wise flow between age groups)
  3. WPP population nudging
  4. Resistance prevalence anchoring

The SIRWS immune boosting dynamics (R→W→S with re-exposure boosting W→R)
and disease progression remain in the Python-level ``ode_system.rhs()`` to
ensure exact consistency with the reference implementation.

The outer control flow (parameter access, WPP interpolation, PEP activation,
routine vaccination) remains in Python for correctness and maintainability.

Usage:
    The accelerated functions are automatically used by the ODE system when
    Numba is available. No user action required.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        """Fallback: return the function unchanged if Numba is not installed."""
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True, fastmath=True)
def compute_infectious_pressure(
    state: np.ndarray,
    n_age: int,
    n_comp: int,
    n_origins: int,
    infectious_S_sym_indices: np.ndarray,
    infectious_S_asym_indices: np.ndarray,
    infectious_R_sym_indices: np.ndarray,
    infectious_R_asym_indices: np.ndarray,
    treated_S_indices: np.ndarray,
    treated_R_indices: np.ndarray,
    infectiousness_mult: np.ndarray,
    relative_infectiousness_asymptomatic: float,
    treated_inf_relative_S: float,
    treated_inf_relative_R: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-age infectious pressure for S and R strains.

    This is the innermost hot loop: n_age × n_origins iterations with
    array indexing. Numba compilation gives ~20-30x speedup here.
    """
    pressure_S = np.zeros(n_age)
    pressure_R = np.zeros(n_age)
    population = np.maximum(state.sum(axis=1), 1.0)

    for ai in range(n_age):
        pop = population[ai]
        for oi in range(n_origins):
            inf_mult = infectiousness_mult[oi]
            # Sensitive strain
            i_sym_S = state[ai, infectious_S_sym_indices[oi]]
            i_asym_S = state[ai, infectious_S_asym_indices[oi]]
            t_S = state[ai, treated_S_indices[oi]]
            pressure_S[ai] += inf_mult * (
                i_sym_S + relative_infectiousness_asymptomatic * i_asym_S + treated_inf_relative_S * t_S
            )
            # Resistant strain
            i_sym_R = state[ai, infectious_R_sym_indices[oi]]
            i_asym_R = state[ai, infectious_R_asym_indices[oi]]
            t_R = state[ai, treated_R_indices[oi]]
            pressure_R[ai] += inf_mult * (
                i_sym_R + relative_infectiousness_asymptomatic * i_asym_R + treated_inf_relative_R * t_R
            )
        pressure_S[ai] /= pop
        pressure_R[ai] /= pop

    return pressure_S, pressure_R


@njit(cache=True, fastmath=True)
def apply_aging(
    dy: np.ndarray,
    state: np.ndarray,
    aging_rates: np.ndarray,
    n_age: int,
    n_comp: int,
    maternal_exit_age_mask: np.ndarray,
    maternal_index: int,
    susceptible_index: int,
) -> None:
    """Apply demographic aging between age groups.

    Handles maternal protection exit: when aging out of maternal-exit age
    groups, M_protected compartment flows into S instead of carrying forward.
    """
    for ai in range(n_age - 1):
        rate = aging_rates[ai]
        for ci in range(n_comp):
            flow = rate * state[ai, ci]
            dy[ai, ci] -= flow
            if maternal_exit_age_mask[ai] and ci == maternal_index:
                # Maternal protection exits to S
                dy[ai + 1, susceptible_index] += flow
            else:
                dy[ai + 1, ci] += flow
    # Oldest group exits
    rate_last = aging_rates[n_age - 1]
    for ci in range(n_comp):
        dy[n_age - 1, ci] -= rate_last * state[n_age - 1, ci]


@njit(cache=True, fastmath=True)
def apply_wpp_nudging(
    dy: np.ndarray,
    state: np.ndarray,
    target_population: np.ndarray,
    nudge_rate_per_day: float,
    n_age: int,
    n_comp: int,
    susceptible_index: int,
) -> None:
    """Apply WPP population nudging toward target trajectory."""
    for ai in range(n_age):
        current_pop = 0.0
        for ci in range(n_comp):
            current_pop += state[ai, ci]
        current_pop = max(current_pop, 1e-12)
        correction = nudge_rate_per_day * (target_population[ai] - current_pop)
        if abs(correction) < 1e-12:
            continue
        for ci in range(n_comp):
            share = state[ai, ci] / current_pop
            dy[ai, ci] += correction * share


@njit(cache=True, fastmath=True)
def apply_resistance_anchor(
    dy: np.ndarray,
    state: np.ndarray,
    rate: float,
    target: float,
    n_age: int,
    n_origins: int,
    exposed_S_indices: np.ndarray,
    exposed_R_indices: np.ndarray,
    infectious_S_sym_indices: np.ndarray,
    infectious_R_sym_indices: np.ndarray,
    infectious_S_asym_indices: np.ndarray,
    infectious_R_asym_indices: np.ndarray,
    treated_S_indices: np.ndarray,
    treated_R_indices: np.ndarray,
) -> None:
    """Anchor resistance prevalence toward target during dynamics."""
    for oi in range(n_origins):
        pairs = [
            (exposed_S_indices[oi], exposed_R_indices[oi]),
            (infectious_S_sym_indices[oi], infectious_R_sym_indices[oi]),
            (infectious_S_asym_indices[oi], infectious_R_asym_indices[oi]),
            (treated_S_indices[oi], treated_R_indices[oi]),
        ]
        for s_idx, r_idx in pairs:
            for ai in range(n_age):
                total = state[ai, s_idx] + state[ai, r_idx]
                desired_r = target * total
                flow = rate * (desired_r - state[ai, r_idx])
                dy[ai, s_idx] -= flow
                dy[ai, r_idx] += flow


# ---------------------------------------------------------------------------
# Pre-computed index cache for use by the ODE system
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from src_python.model.compartments import (
    COMPARTMENTS,
    VACCINE_ORIGINS,
    StateIndex,
    compartment_index,
    exposed_name,
    infectious_name,
    susceptible_name,
    treated_name,
)


@dataclass(frozen=True)
class CompartmentIndexCache:
    """Pre-computed integer index arrays for Numba kernels.

    Built once per parameter set, then reused for every RHS evaluation.
    """
    n_origins: int
    susceptible_indices: np.ndarray
    exposed_S_indices: np.ndarray
    exposed_R_indices: np.ndarray
    infectious_S_sym_indices: np.ndarray
    infectious_S_asym_indices: np.ndarray
    infectious_R_sym_indices: np.ndarray
    infectious_R_asym_indices: np.ndarray
    treated_S_indices: np.ndarray
    treated_R_indices: np.ndarray
    recovered_index: int
    maternal_index: int
    susceptible_index: int

    @classmethod
    def build(cls) -> "CompartmentIndexCache":
        n_origins = len(VACCINE_ORIGINS)
        return cls(
            n_origins=n_origins,
            susceptible_indices=np.array([compartment_index(susceptible_name(o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            exposed_S_indices=np.array([compartment_index(exposed_name("S", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            exposed_R_indices=np.array([compartment_index(exposed_name("R", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            infectious_S_sym_indices=np.array([compartment_index(infectious_name("S", "sym", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            infectious_S_asym_indices=np.array([compartment_index(infectious_name("S", "asym", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            infectious_R_sym_indices=np.array([compartment_index(infectious_name("R", "sym", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            infectious_R_asym_indices=np.array([compartment_index(infectious_name("R", "asym", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            treated_S_indices=np.array([compartment_index(treated_name("S", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            treated_R_indices=np.array([compartment_index(treated_name("R", o)) for o in VACCINE_ORIGINS], dtype=np.int64),
            recovered_index=compartment_index("R_natural"),
            maternal_index=compartment_index("M_protected"),
            susceptible_index=compartment_index("S"),
        )


# Module-level singleton (built once on first import)
_INDEX_CACHE: CompartmentIndexCache | None = None


def get_index_cache() -> CompartmentIndexCache:
    """Get or build the compartment index cache."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = CompartmentIndexCache.build()
    return _INDEX_CACHE
