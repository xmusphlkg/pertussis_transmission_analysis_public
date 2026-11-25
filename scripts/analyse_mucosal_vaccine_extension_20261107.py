"""Paired health and cost contrasts used by the intranasal policy analysis."""
import numpy as np


def describe(values, prefix):
    values = np.asarray(values, dtype=float)
    if values.shape != (1000,) or not np.isfinite(values).all():
        raise ValueError("Finite paired 1000-draw vector required")
    lo, hi = np.quantile(values, [.025, .975])
    return {prefix+"_mean": float(values.mean()), prefix+"_q025": float(lo), prefix+"_q975": float(hi)}


def marginal_components(cache, candidate, matched, horizon, burden, mortality):
    """Benefit-oriented health differences and candidate-minus-standard costs."""
    yll = cache[f"yll_{horizon}"][:, candidate]-cache[f"yll_{horizon}"][:, matched]
    if mortality == "borrowed_noninfant":
        yll = yll+cache[f"extra_yll_{horizon}"][:, candidate]-cache[f"extra_yll_{horizon}"][:, matched]
    yld = np.full(1000, cache[f"yld_{horizon}_{burden}"][candidate]-cache[f"yld_{horizon}_{burden}"][matched])
    cost = cache[f"cost_{horizon}"][:, candidate]-cache[f"cost_{horizon}"][:, matched]
    vaccine = (cache[f"vaccine_{horizon}"][:, candidate]-cache[f"vaccine_{horizon}"][:, matched]).sum(axis=1)
    delivery = (cache[f"delivery_{horizon}"][:, candidate]-cache[f"delivery_{horizon}"][:, matched]).sum(axis=1)
    return dict(dalys_averted=yll+yld, yll_averted=yll, yld_averted=yld,
                non_nasal_incremental_cost=cost, standard_vaccine_incremental_cost=vaccine,
                standard_delivery_incremental_cost=delivery, incremental_treatment_cost=cost-vaccine-delivery)
