"""Resource-comparable intervention definitions for the core2026 model.

This module deliberately contains no transmission code.  It turns a common
incremental dose budget into auditable, paired programme perturbations.  The
forward simulator remains responsible for applying the resulting programme
mutation to a birth-cohort history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


Target = Literal["maternal", "routine", "timing", "adolescent"]


@dataclass(frozen=True)
class BudgetSpec:
    """A transparent dose budget for one prospective year.

    ``doses_per_1000`` is a scenario unit, not a cost estimate.  The
    denominator defines the common dose mass for a profile (the active
    runner uses annual births); the strategy-specific eligible denominator
    is stored on :class:`StrategySpec` and converts that mass to coverage.
    """

    budget_id: str
    doses_per_1000: float
    denominator_population: float

    def __post_init__(self) -> None:
        if not self.budget_id:
            raise ValueError("budget_id must be non-empty")
        if not np.isfinite(self.doses_per_1000) or self.doses_per_1000 < 0:
            raise ValueError("doses_per_1000 must be finite and non-negative")
        if not np.isfinite(self.denominator_population) or self.denominator_population <= 0:
            raise ValueError("denominator_population must be finite and positive")

    @property
    def total_doses(self) -> float:
        return float(self.doses_per_1000 * self.denominator_population / 1000.0)


@dataclass(frozen=True)
class StrategySpec:
    """One mechanism-level programme perturbation.

    ``one_time`` is required for the adolescent primary mechanism.  A timing
    shift changes the delivery delay distribution while preserving the dose
    count and therefore cannot consume a second hidden intervention budget.
    """

    strategy_id: str
    target: Target
    eligible_population: float
    baseline_coverage: float
    one_time: bool = False
    dose_age_years: float | None = None
    timing_shift_months: int = 0
    # Infant dose kind used by the routine pathway (0=dose 1, 1=dose 2,
    # 2=dose 3, 3=toddler booster).  Keeping this explicit prevents a
    # programme with an 18-month booster from silently being mutated through
    # the primary-series table.
    routine_dose_kind: int | None = None
    evidence_class: str = "policy_scenario"

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if not np.isfinite(self.eligible_population) or self.eligible_population <= 0:
            raise ValueError("eligible_population must be finite and positive")
        if not np.isfinite(self.baseline_coverage) or not 0 <= self.baseline_coverage <= 1:
            raise ValueError("baseline_coverage must lie in [0, 1]")
        if not isinstance(self.timing_shift_months, int):
            raise TypeError("timing_shift_months must be an integer number of months")
        if self.target == "adolescent":
            if not self.one_time:
                raise ValueError("the primary adolescent strategy must be one-time")
            if self.dose_age_years is None or not np.isfinite(self.dose_age_years):
                raise ValueError("the adolescent strategy needs a finite dose age")
            if self.dose_age_years < 2.5:
                raise ValueError("adolescent dose age must be at least 2.5 years")
        elif self.one_time:
            raise ValueError("one_time is reserved for the adolescent strategy")
        if self.target != "timing" and self.timing_shift_months != 0:
            raise ValueError("timing_shift_months is only valid for the timing strategy")
        if self.routine_dose_kind is not None and self.routine_dose_kind not in range(4):
            raise ValueError("routine_dose_kind must be one of 0, 1, 2, 3 or None")


@dataclass(frozen=True)
class Allocation:
    """Auditable allocation of a fixed budget to one strategy."""

    budget_id: str
    strategy_id: str
    target: Target
    budget_doses: float
    used_doses: float
    residual_doses: float
    eligible_population: float
    baseline_coverage: float
    coverage_increment: float
    resulting_coverage: float
    one_time: bool
    dose_age_years: float | None
    timing_shift_months: int
    evidence_class: str
    routine_dose_kind: int | None = None

    @property
    def dose_fraction_used(self) -> float:
        return float(self.used_doses / self.budget_doses) if self.budget_doses else 1.0


def allocate_fixed_budget(budget: BudgetSpec, strategy: StrategySpec) -> Allocation:
    """Allocate the same budget to a strategy without exceeding coverage 100%.

    The budget is shared across strategies in a comparison.  If a strategy
    saturates its eligible population, the unspent remainder is retained in
    the audit row; a caller may then either reject the comparison or report
    the residual explicitly.  It must never silently top up that strategy.
    """

    # A timing intervention reuses the same dose stream and changes only its
    # delivery time.  It therefore consumes the declared delivery budget but
    # has no coverage increment and cannot saturate.
    if strategy.target == "timing":
        used = float(budget.total_doses)
        residual = 0.0
        increment = 0.0
        result = float(strategy.baseline_coverage)
    else:
        available = strategy.eligible_population * (1.0 - strategy.baseline_coverage)
        used = min(float(budget.total_doses), float(available))
        residual = float(budget.total_doses - used)
        increment = used / strategy.eligible_population
        result = float(strategy.baseline_coverage + increment)
    if residual < -1e-9 or result > 1.0 + 1e-9:
        raise AssertionError("budget allocation violated the coverage bounds")
    return Allocation(
        budget_id=budget.budget_id,
        strategy_id=strategy.strategy_id,
        target=strategy.target,
        budget_doses=float(budget.total_doses),
        used_doses=float(used),
        residual_doses=max(0.0, residual),
        eligible_population=float(strategy.eligible_population),
        baseline_coverage=float(strategy.baseline_coverage),
        coverage_increment=float(increment),
        resulting_coverage=float(min(1.0, result)),
        one_time=bool(strategy.one_time),
        dose_age_years=strategy.dose_age_years,
        timing_shift_months=int(strategy.timing_shift_months),
        routine_dose_kind=strategy.routine_dose_kind,
        evidence_class=strategy.evidence_class)


def allocate_strategy_set(
    budget: BudgetSpec,
    strategies: Iterable[StrategySpec],
    *,
    require_full_budget: bool = True,
) -> tuple[Allocation, ...]:
    """Allocate and validate a paired strategy set under one fixed budget."""

    rows = tuple(allocate_fixed_budget(budget, strategy) for strategy in strategies)
    if not rows:
        raise ValueError("at least one strategy is required")
    if len({row.budget_id for row in rows}) != 1:
        raise ValueError("all strategies must use the same budget")
    if require_full_budget:
        saturated = [row.strategy_id for row in rows if row.residual_doses > 1e-8]
        if saturated:
            raise ValueError(
                "fixed-budget comparison has residual doses for saturated strategies: "
                + ", ".join(saturated)
            )
    return rows


def allocation_table(allocations: Iterable[Allocation]) -> list[dict[str, object]]:
    """Return stable, serialisable rows for a scenario audit table."""

    return [
        {
            "budget_id": row.budget_id,
            "strategy_id": row.strategy_id,
            "target": row.target,
            "budget_doses": row.budget_doses,
            "used_doses": row.used_doses,
            "residual_doses": row.residual_doses,
            "eligible_population": row.eligible_population,
            "baseline_coverage": row.baseline_coverage,
            "coverage_increment": row.coverage_increment,
            "resulting_coverage": row.resulting_coverage,
            "one_time": row.one_time,
            "dose_age_years": row.dose_age_years,
            "timing_shift_months": row.timing_shift_months,
            "routine_dose_kind": row.routine_dose_kind,
            "evidence_class": row.evidence_class,
        }
        for row in allocations
    ]

