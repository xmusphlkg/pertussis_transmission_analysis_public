"""Tau-leaping stochastic layer for resistance strain dynamics.

The deterministic ODE produces unrealistically fast resistance fixation because
it cannot represent genetic drift, founder effects, or stochastic extinction of
rare strains. This module adds a tau-leaping stochastic correction to the
resistance dynamics while keeping the rest of the ODE deterministic.

Approach:
    After each ODE time step (dt), we apply a stochastic correction to the
    S/R strain balance. The correction is drawn from a binomial process that
    represents the finite-population sampling of new infections:

    1. Compute the deterministic expected new infections by each strain
    2. Draw actual new infections from Binomial(N_new, p_R) where p_R is the
       expected resistant fraction of new infections
    3. Apply the difference (stochastic - deterministic) as a correction

    This preserves the total infection count (mass conservation) while adding
    realistic stochastic fluctuations to the strain composition.

    For small populations or low prevalence, this naturally slows resistance
    fixation because random drift can temporarily reverse selection. For large
    populations, the stochastic correction becomes negligible relative to the
    deterministic dynamics, recovering the ODE behavior.

References:
    - Gillespie, D.T. (2001). Approximate accelerated stochastic simulation
      of chemically reacting systems. J Chem Phys 115(4):1716-1733.
    - Cao, Y., Gillespie, D.T., Petzold, L.R. (2006). Efficient step size
      selection for the tau-leaping simulation method. J Chem Phys 124:044109.
    - Colijn, C. et al. (2009). The dynamics of resistance in pertussis.
      Proc R Soc B 276:2209-2216.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src_python.model.compartments import (
    COMPARTMENTS,
    STRAINS,
    VACCINE_ORIGINS,
    StateIndex,
    exposed_name,
    infectious_name,
    treated_name,
)


@dataclass(frozen=True)
class StochasticResistanceConfig:
    """Configuration for the tau-leaping resistance stochastic layer."""
    enabled: bool = True
    # Effective population size for the strain competition process.
    # Smaller values = more drift = slower fixation.
    # This represents the number of independent transmission events per time step
    # that determine strain composition, NOT the total population.
    # For pertussis with ~0.3% prevalence in a 27M population, the effective
    # infectious population is ~80,000. But transmission chains are clustered
    # (household SAR ~0.8), so the effective number of independent events is
    # much smaller. A value of 500-5000 is appropriate.
    effective_population_size: int = 2000
    # Minimum strain fraction below which stochastic extinction is possible
    min_strain_fraction: float = 0.001
    # Random seed (offset per country/scenario)
    random_seed: int = 20260515

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StochasticResistanceConfig":
        if not data:
            return cls(enabled=False)
        return cls(
            enabled=bool(data.get("enabled", True)),
            effective_population_size=max(10, int(data.get("effective_population_size", 2000))),
            min_strain_fraction=float(data.get("min_strain_fraction", 0.001)),
            random_seed=int(data.get("random_seed", 20260515)),
        )


def apply_stochastic_resistance_step(
    y: np.ndarray,
    index: StateIndex,
    *,
    config: StochasticResistanceConfig,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> np.ndarray:
    """Apply one tau-leaping stochastic correction to resistance dynamics.

    This should be called after each ODE integration step. It redistributes
    infected individuals between S and R strains based on a stochastic draw,
    preserving total infected counts per (origin, symptom_status, age_group).

    Parameters
    ----------
    y : flat state vector (after ODE step, non-negative)
    index : StateIndex for reshaping
    config : stochastic resistance configuration
    rng : numpy random generator
    dt : time step size in days (used to scale the effective population)

    Returns
    -------
    Modified state vector with stochastic strain rebalancing applied.
    """
    if not config.enabled:
        return y

    state = np.maximum(index.reshape(y.copy()), 0.0)
    c = {name: COMPARTMENTS.index(name) for name in COMPARTMENTS}

    # Scale effective population by time step (larger dt = more events = less noise)
    n_eff = max(10, int(config.effective_population_size * dt))

    for origin in VACCINE_ORIGINS:
        # For each paired (S, R) compartment set, apply stochastic rebalancing
        pairs = [
            (exposed_name("S", origin), exposed_name("R", origin)),
            (infectious_name("S", "sym", origin), infectious_name("R", "sym", origin)),
            (infectious_name("S", "asym", origin), infectious_name("R", "asym", origin)),
            (treated_name("S", origin), treated_name("R", origin)),
        ]
        for s_name, r_name in pairs:
            s_idx = c[s_name]
            r_idx = c[r_name]

            for age_idx in range(index.n_age):
                s_count = state[age_idx, s_idx]
                r_count = state[age_idx, r_idx]
                total = s_count + r_count

                if total < 1.0:
                    continue

                # Current resistant fraction
                p_r = r_count / total

                # Skip if already fixed (within numerical tolerance)
                if p_r < config.min_strain_fraction or p_r > (1.0 - config.min_strain_fraction):
                    continue

                # Draw stochastic resistant count from Binomial(n_eff, p_r)
                # then scale back to the actual population
                n_draw = min(n_eff, max(10, int(total)))
                stochastic_r_count = rng.binomial(n_draw, p_r)
                stochastic_p_r = stochastic_r_count / n_draw

                # Apply the stochastic fraction to the actual counts
                new_r = total * stochastic_p_r
                new_s = total * (1.0 - stochastic_p_r)

                state[age_idx, r_idx] = new_r
                state[age_idx, s_idx] = new_s

    return index.flatten(state)


def solve_with_stochastic_resistance(
    params: "PreparedParameters",
    index: StateIndex,
    *,
    stochastic_config: StochasticResistanceConfig | None = None,
    seed_offset: int = 0,
) -> Any:
    """Solve the ODE model with tau-leaping stochastic resistance corrections.

    Uses the standard ODE solver but applies stochastic strain rebalancing
    at each output time step. This is a hybrid deterministic-stochastic approach
    that preserves the computational efficiency of the ODE while adding
    realistic resistance dynamics.

    Parameters
    ----------
    params : PreparedParameters
    index : StateIndex
    stochastic_config : configuration for stochastic layer (None = use config)
    seed_offset : offset added to random seed for reproducibility across runs

    Returns
    -------
    solve_ivp-compatible result object
    """
    from src_python.model.outputs import initial_state, rebalance_resistant_prevalence, solve_model
    from src_python.model.parameters import PreparedParameters

    if stochastic_config is None:
        stochastic_config = StochasticResistanceConfig.from_dict(
            params.raw.get("stochastic_resistance")
        )

    if not stochastic_config.enabled:
        # Fall back to deterministic solver
        return solve_model(params, index)

    # Use the standard solver but with segmented integration
    # Apply stochastic correction at each output time step
    from scipy.integrate import solve_ivp
    from src_python.model.ode_system import rhs
    from src_python.model.rk4_solver import solve_rk4

    sim = params.raw["simulation"]
    output_time_step = float(sim.get("output_time_step", sim.get("time_step", 7.0)))
    start_time = float(sim["start_time"])
    end_time = float(sim["end_time"])
    solver_method = str(sim.get("solver_method", "RK45"))

    # Generate output times
    t_eval = np.arange(start_time, end_time + output_time_step, output_time_step)
    t_eval = t_eval[t_eval <= end_time]
    if len(t_eval) > 0 and t_eval[-1] < end_time:
        t_eval = np.append(t_eval, end_time)

    # Initialize
    y0 = initial_state(params, index)

    # Burn-in (deterministic, no stochastic correction needed for equilibration)
    burn_in_years = float(sim.get("burn_in_years", 0.0))
    if burn_in_years > 0:
        burn_t_start = start_time - burn_in_years * 365.0
        burn_solution = solve_ivp(
            fun=lambda t, y: rhs(t, y, params, index),
            t_span=(burn_t_start, start_time),
            y0=y0,
            t_eval=[start_time],
            method=solver_method,
            rtol=float(sim.get("rtol", 1e-5)),
            atol=float(sim.get("atol", 1e-7)),
        )
        if not burn_solution.success:
            raise RuntimeError(f"Burn-in failed: {burn_solution.message}")
        y0 = np.maximum(burn_solution.y[:, -1], 0.0)
        if params.resistance.get("rebalance_after_burn_in", True):
            y0 = rebalance_resistant_prevalence(y0, params, index)

    # Analysis period: integrate segment by segment with stochastic corrections
    rng = np.random.default_rng(stochastic_config.random_seed + seed_offset)
    rtol = float(sim.get("rtol", 1e-5))
    atol = float(sim.get("atol", 1e-7))

    t_out = [start_time]
    y_out = [y0.copy()]
    current_y = y0.copy()

    for i in range(1, len(t_eval)):
        t_start_seg = t_eval[i - 1]
        t_end_seg = t_eval[i]
        dt_seg = t_end_seg - t_start_seg

        # Integrate one segment deterministically
        seg_sol = solve_ivp(
            fun=lambda t, y: rhs(t, y, params, index),
            t_span=(t_start_seg, t_end_seg),
            y0=current_y,
            t_eval=[t_end_seg],
            method=solver_method,
            rtol=rtol,
            atol=atol,
        )
        if not seg_sol.success:
            # If segment fails, try with tighter tolerances
            seg_sol = solve_ivp(
                fun=lambda t, y: rhs(t, y, params, index),
                t_span=(t_start_seg, t_end_seg),
                y0=current_y,
                t_eval=[t_end_seg],
                method="LSODA",
                rtol=rtol * 0.1,
                atol=atol * 0.1,
            )
            if not seg_sol.success:
                raise RuntimeError(
                    f"Segment integration failed at t={t_start_seg}: {seg_sol.message}"
                )

        current_y = np.maximum(seg_sol.y[:, -1], 0.0)

        # Apply stochastic resistance correction
        current_y = apply_stochastic_resistance_step(
            current_y, index,
            config=stochastic_config,
            rng=rng,
            dt=dt_seg,
        )

        t_out.append(t_end_seg)
        y_out.append(current_y.copy())

    # Build result object compatible with solve_ivp interface
    from src_python.model.rk4_solver import _RK4Result
    result = _RK4Result(
        t=np.array(t_out),
        y=np.column_stack(y_out),
    )
    return result
