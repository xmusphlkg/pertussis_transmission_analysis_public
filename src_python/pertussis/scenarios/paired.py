"""Paired prospective simulations for resource-constrained contrasts.

The important invariant in this module is that every strategy starts from one
historical state and one accumulator state.  Programme mutations are applied
only after ``horizon_start_year``; consequently a difference between two rows
is a prospective counterfactual difference and not a difference in historical
calibration.

The mutators operate on encoded :class:`~pertussis.model.tables.ModelInputs`
tables rather than reparsing programme YAML.  This keeps the history frozen and
makes the runner useful for both real country bundles and small synthetic
fixtures.  The mutation contract is deliberately explicit: maternal and
routine scale-up add an incremental coverage fraction, a timing strategy moves
the encoded infant schedule while preserving its fractions, and the adolescent
strategy appends one cohort event per prospective birth year.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from ..model.parameters import ModelParameters
from ..model.simulate import SimulationResult, Simulator
from ..model.tables import DoseEvents, ModelInputs
from ..states import ACC_TYPICAL, N_INFANT_DOSE_KINDS, TDAP_BOOSTER
from .resource_constrained import Allocation, BudgetSpec, StrategySpec, allocate_strategy_set


@dataclass(frozen=True)
class EndpointTotals:
    """Cumulative typical-course onsets during one prospective horizon."""

    infant_0_2m_cases: float
    infant_under1_cases: float
    age1_17_cases: float

    def as_dict(self) -> dict[str, float]:
        return {
            "infant_0_2m_cases": float(self.infant_0_2m_cases),
            "infant_under1_cases": float(self.infant_under1_cases),
            "age1_17_cases": float(self.age1_17_cases),
        }


@dataclass
class PairedScenarioRun:
    """Results and provenance for one paired strategy set."""

    profile: str | None
    budget: BudgetSpec
    horizon_start_year: int
    horizon_end_year: int
    baseline_endpoints: EndpointTotals
    rows: tuple[dict[str, object], ...]
    history_state: np.ndarray
    history_acc: np.ndarray
    baseline_result: SimulationResult
    scenario_results: dict[str, SimulationResult]

    def records(self) -> list[dict[str, object]]:
        """Return independent serialisable row dictionaries."""

        return [dict(row) for row in self.rows]

    def to_dataframe(self):
        """Return a pandas table without making pandas a core dependency."""

        import pandas as pd

        return pd.DataFrame(self.records())


@dataclass(frozen=True)
class PreparedPairedHistory:
    """Cached boundary state and baseline for one parameter/structure draw.

    The baseline does not depend on the incremental budget.  Keeping it with
    the historical state avoids re-integrating the same 1970--2027 trajectory
    once for every budget in an ensemble, while preserving the exact paired
    starting state used by every counterfactual.
    """

    start_day: int
    end_day: int
    history_state: np.ndarray
    history_acc: np.ndarray
    baseline_result: SimulationResult
    baseline_endpoints: EndpointTotals


def prepare_paired_history(
    inputs: ModelInputs,
    params: ModelParameters,
    *,
    horizon_start_year: int,
    horizon_end_year: int,
) -> PreparedPairedHistory:
    """Integrate history and baseline once for reuse across budgets."""

    start_day, end_day = _prospective_days(inputs, horizon_start_year, horizon_end_year)
    history_sim = Simulator(inputs)
    y0, acc0 = history_sim.initial_state()
    history = history_sim.run(params, y0, acc0, 0, start_day)
    record_days = np.array([inputs.calendar.year_start_day(year)
                            for year in range(horizon_start_year, horizon_end_year + 1)])
    baseline_result = history_sim.run(params, history.y, history.acc, start_day, end_day,
                                     record_days=record_days)
    baseline_endpoints = _endpoint_totals(baseline_result, history.acc, inputs.grid)
    return PreparedPairedHistory(
        start_day=start_day,
        end_day=end_day,
        history_state=np.array(history.y, copy=True),
        history_acc=np.array(history.acc, copy=True),
        baseline_result=baseline_result,
        baseline_endpoints=baseline_endpoints,
    )


def _prospective_days(inputs: ModelInputs, start_year: int, end_year: int) -> tuple[int, int]:
    """Return the half-open day interval ``[start, end)`` for a horizon."""

    cal = inputs.calendar
    if end_year <= start_year:
        raise ValueError("horizon_end_year must be greater than horizon_start_year")
    try:
        start = cal.year_start_day(start_year)
        end = cal.year_start_day(end_year)
    except ValueError as exc:
        raise ValueError("prospective horizon lies outside the input calendar") from exc
    return start, end


def _copy_events_with_adolescent_doses(
    inputs: ModelInputs,
    allocation: Allocation,
    start_year: int,
    end_year: int,
) -> DoseEvents:
    """Apply an adolescent coverage increment to each prospective cohort.

    A country may already have an adolescent ``tdap_booster`` event (Australia
    is the important example).  In that case the resource increment must be
    added to the existing cohort fraction.  Appending a second event on the
    same day would apply two sequential boosters and would make the nominal
    dose budget larger than the allocation.  Countries without an existing
    event receive one new event per prospective cohort.
    """

    if allocation.dose_age_years is None:
        raise ValueError("adolescent strategy needs dose_age_years")
    grid, cal = inputs.grid, inputs.calendar
    age = float(allocation.dose_age_years)
    k = int(np.floor(age + 0.5))
    if not grid.cohort_first <= k <= grid.cohort_last:
        raise ValueError(f"adolescent dose age {age} is outside the cohort grid")
    frac_of_cohort = float(allocation.coverage_increment)
    if frac_of_cohort < 0 or frac_of_cohort > 1:
        raise ValueError("adolescent coverage increment must lie in [0, 1]")
    f = age + 0.5 - k
    days = list(np.asarray(inputs.events.day, dtype=np.int64))
    bins = list(np.asarray(inputs.events.bin, dtype=np.int64))
    kinds = list(np.asarray(inputs.events.kind, dtype=np.int64))
    fracs = list(np.asarray(inputs.events.fraction, dtype=float))
    for year in range(start_year, end_year):
        day = cal.year_start_day(year) + int(round(f * 365.25))
        if day >= cal.n_days:
            continue
        target_bin = grid.cohort_bin(k)
        matches = [
            i for i, (d, b, kind) in enumerate(zip(days, bins, kinds))
            if d == day and b == target_bin and kind == TDAP_BOOSTER
        ]
        if matches:
            # The programme encoder emits one event per (day, bin, kind).  If
            # a caller supplied duplicate rows, update the first row and leave
            # the others untouched so this function never changes history.
            i = matches[0]
            fracs[i] = min(1.0, fracs[i] + frac_of_cohort)
        else:
            days.append(day)
            bins.append(target_bin)
            kinds.append(TDAP_BOOSTER)
            fracs.append(frac_of_cohort)
    return DoseEvents(
        np.asarray(days, dtype=np.int64),
        np.asarray(bins, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64),
        np.asarray(fracs, dtype=float),
    )


def _shift_infant_schedule(
    infant: np.ndarray,
    month_start: int,
    shift: int,
    fraction: float = 1.0,
    dose_kind: int | None = None,
) -> np.ndarray:
    """Shift a fraction of the encoded infant schedule after ``month_start``.

    A shift is applied to the selected dose kind, or to every kind when
    ``dose_kind`` is ``None``.  ``fraction`` makes the
    operational change resource-comparable: a 10-dose-per-1000 budget shifts
    only the corresponding share of routine doses, while preserving the total
    number of doses.  Entries that would fall outside the 0--23 month grid are
    retained in their original slot rather than silently discarded.
    """

    out = np.array(infant, dtype=float, copy=True)
    if shift == 0:
        return out
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("timing shift fraction must lie in [0, 1]")
    n_monthly = out.shape[1]
    for mi in range(max(0, month_start), out.shape[0]):
        kinds = range(out.shape[2]) if dose_kind is None else (dose_kind,)
        for kind in kinds:
            original = out[mi, :, kind].copy()
            moved = np.zeros_like(original)
            for source in range(n_monthly):
                target = source + int(shift)
                keep = original[source] * (1.0 - fraction)
                if 0 <= target < n_monthly:
                    moved[target] += original[source] * fraction
                    moved[source] += keep
                else:
                    moved[source] += original[source]
            out[mi, :, kind] = np.clip(moved, 0.0, 1.0)
    return out


def mutate_inputs_for_allocation(
    inputs: ModelInputs,
    allocation: Allocation,
    *,
    horizon_start_year: int,
    horizon_end_year: int,
) -> ModelInputs:
    """Create a counterfactual input bundle without changing historical rows.

    ``allocation.coverage_increment`` is the incremental fraction that the
    fixed budget buys.  Maternal protection is scaled on the effective
    protected-birth fraction, while routine scale-up mutates the explicitly
    selected infant dose kind.  These operations preserve the encoded
    historical table and avoid treating raw coverage as effective coverage.
    """

    start_day, _ = _prospective_days(inputs, horizon_start_year, horizon_end_year)
    start_month = inputs.calendar.month_idx[start_day]
    maternal = np.array(inputs.maternal_effective, dtype=float, copy=True)
    infant = np.array(inputs.infant_dose, dtype=float, copy=True)
    events = inputs.events
    target = allocation.target
    increment = float(allocation.coverage_increment)
    if target == "maternal":
        if allocation.baseline_coverage > 0:
            scale = allocation.resulting_coverage / allocation.baseline_coverage
            maternal[start_day:] = np.clip(maternal[start_day:] * scale, 0.0, 1.0)
        else:
            # The core programme convention uses an effective maternal share
            # of 0.9.  This fallback is used only when no maternal programme
            # exists in the baseline profile, so no empirical share can be
            # inferred from the encoded table.
            maternal[start_day:] = np.clip(maternal[start_day:] + 0.9 * increment, 0.0, 1.0)
    elif target == "routine":
        dose_kind = 2 if allocation.routine_dose_kind is None else allocation.routine_dose_kind
        if allocation.baseline_coverage > 0:
            scale = allocation.resulting_coverage / allocation.baseline_coverage
            infant[start_month:, :, dose_kind] = np.clip(infant[start_month:, :, dose_kind] * scale, 0.0, 1.0)
        else:
            infant[start_month:, :, dose_kind] = np.clip(
                infant[start_month:, :, dose_kind] + increment, 0.0, 1.0
            )
    elif target == "timing":
        if allocation.baseline_coverage <= 0.0:
            raise ValueError("timing strategy needs a positive baseline coverage")
        # Timing uses doses that would have been delivered anyway, so its
        # allocator deliberately reports no coverage increment.  The fraction
        # of the routine stream that can be moved earlier is instead the
        # budget's delivered-dose share among the baseline routine stream.
        stream_doses = allocation.eligible_population * allocation.baseline_coverage
        shifted_fraction = min(
            1.0,
            max(0.0, allocation.used_doses / stream_doses),
        ) if stream_doses > 0 else 0.0
        infant = _shift_infant_schedule(
            infant,
            start_month,
            allocation.timing_shift_months,
            shifted_fraction,
            dose_kind=(2 if allocation.routine_dose_kind is None else allocation.routine_dose_kind),
        )
    elif target == "adolescent":
        events = _copy_events_with_adolescent_doses(
            inputs, allocation, horizon_start_year, horizon_end_year
        )
    else:  # pragma: no cover - Allocation validates target at construction time.
        raise ValueError(f"unknown strategy target {target!r}")
    metadata = deepcopy(inputs.metadata)
    metadata.update(
        {
            "counterfactual_strategy_id": allocation.strategy_id,
            "counterfactual_horizon_start_year": horizon_start_year,
            "counterfactual_horizon_end_year": horizon_end_year,
            "counterfactual_budget_id": allocation.budget_id,
        }
    )
    return replace(inputs, maternal_effective=maternal, infant_dose=infant, events=events, metadata=metadata)


def _endpoint_totals(result: SimulationResult, acc_before: np.ndarray, grid) -> EndpointTotals:
    """Aggregate typical onsets into the three declared age endpoints."""

    by_bin = np.asarray(result.acc[:, ACC_TYPICAL] - acc_before[:, ACC_TYPICAL], dtype=float)
    monthly = np.arange(grid.n_monthly)
    infant_0_2m = float(by_bin[monthly[:3]].sum())
    infant_under1 = float(by_bin[monthly[:12]].sum())
    older_bins = list(monthly[12:])
    older_bins.extend(grid.cohort_bin(age) for age in range(2, 18))
    age1_17 = float(by_bin[np.asarray(older_bins, dtype=int)].sum())
    return EndpointTotals(infant_0_2m, infant_under1, age1_17)


def annual_endpoint_totals(result: SimulationResult, grid) -> list[dict[str, float]]:
    """Annual increments from boundary records, preserving the full-horizon sum.

    The paired runner records each January boundary. This permits rollout and
    later-window comparisons without restarting the counterfactual history.
    """
    if len(result.record_days) < 2 or not np.isfinite(result.acc_records).all():
        raise ValueError("complete annual accumulator records are required")
    rows = []
    for i in range(len(result.record_days)-1):
        endpoint = _endpoint_totals(
            replace(result, acc=result.acc_records[i+1]), result.acc_records[i], grid
        )
        rows.append({"interval_index": i, **endpoint.as_dict()})
    return rows


def _record(
    *,
    profile: str | None,
    budget: BudgetSpec,
    allocation: Allocation | None,
    totals: EndpointTotals,
    baseline: EndpointTotals,
    horizon_start_year: int,
    horizon_end_year: int,
) -> dict[str, object]:
    if allocation is None:
        strategy_id, target, used, increment, resulting, residual = "baseline", "baseline", 0.0, 0.0, 0.0, 0.0
        one_time, age, shift, claim = False, None, 0, "historical_baseline"
    else:
        strategy_id, target = allocation.strategy_id, allocation.target
        used, increment, resulting, residual = (
            allocation.used_doses,
            allocation.coverage_increment,
            allocation.resulting_coverage,
            allocation.residual_doses,
        )
        one_time, age, shift, claim = (
            allocation.one_time,
            allocation.dose_age_years,
            allocation.timing_shift_months,
            allocation.evidence_class,
        )
    def rel(base: float, value: float) -> float:
        return float((base - value) / base) if base > 0 else float("nan")
    horizon_years = horizon_end_year - horizon_start_year
    return {
        "profile": profile,
        "strategy_id": strategy_id,
        "target": target,
        "budget_id": budget.budget_id,
        "budget_doses_per_1000": budget.doses_per_1000,
        "annual_doses_used": used,
        "horizon_doses_used": used * horizon_years,
        "residual_doses": residual,
        "coverage_increment": increment,
        "resulting_coverage": resulting,
        "one_time": one_time,
        "dose_age_years": age,
        "timing_shift_months": shift,
        "routine_dose_kind": (allocation.routine_dose_kind if allocation is not None else None),
        "timed_fraction_of_routine_stream": (
            float(min(1.0, allocation.used_doses / (
                allocation.eligible_population * max(allocation.baseline_coverage, 1e-12)
            )))
            if allocation is not None and allocation.target == "timing" else 0.0
        ),
        "claim_label": claim,
        "horizon_start_year": horizon_start_year,
        "horizon_end_year": horizon_end_year,
        **totals.as_dict(),
        "infant_0_2m_absolute_difference_vs_baseline": totals.infant_0_2m_cases - baseline.infant_0_2m_cases,
        "infant_under1_absolute_difference_vs_baseline": totals.infant_under1_cases - baseline.infant_under1_cases,
        "age1_17_absolute_difference_vs_baseline": totals.age1_17_cases - baseline.age1_17_cases,
        "infant_0_2m_relative_reduction_vs_baseline": rel(baseline.infant_0_2m_cases, totals.infant_0_2m_cases),
        "infant_under1_relative_reduction_vs_baseline": rel(baseline.infant_under1_cases, totals.infant_under1_cases),
        "age1_17_relative_reduction_vs_baseline": rel(baseline.age1_17_cases, totals.age1_17_cases),
        # These endpoints require a separate clinical cascade and therefore
        # remain explicit nulls in the primary runner output.
        "infant_hospitalisation_index": None,
        "infant_death_index": None,
    }


def run_paired_scenarios(
    inputs: ModelInputs,
    params: ModelParameters,
    budget: BudgetSpec,
    strategies: Iterable[StrategySpec],
    *,
    horizon_start_year: int,
    horizon_end_year: int,
    profile: str | None = None,
    require_full_budget: bool = True,
    prepared_history: PreparedPairedHistory | None = None,
) -> PairedScenarioRun:
    """Run baseline and resource-comparable prospective strategies in pairs.

    The historical integration is performed once.  Every prospective run is
    then started with copies of that exact ``y`` and ``acc`` boundary state.
    ``horizon_end_year`` is exclusive, so a 2027--2035 horizon is passed as
    ``horizon_start_year=2027, horizon_end_year=2036``.
    """

    if prepared_history is None:
        prepared_history = prepare_paired_history(
            inputs,
            params,
            horizon_start_year=horizon_start_year,
            horizon_end_year=horizon_end_year,
        )
    start_day, end_day = prepared_history.start_day, prepared_history.end_day
    allocations = allocate_strategy_set(
        budget, tuple(strategies), require_full_budget=require_full_budget
    )
    baseline_result = prepared_history.baseline_result
    baseline_totals = prepared_history.baseline_endpoints
    rows = [
        _record(
            profile=profile,
            budget=budget,
            allocation=None,
            totals=baseline_totals,
            baseline=baseline_totals,
            horizon_start_year=horizon_start_year,
            horizon_end_year=horizon_end_year,
        )
    ]
    scenario_results: dict[str, SimulationResult] = {}
    for allocation in allocations:
        scenario_inputs = mutate_inputs_for_allocation(
            inputs,
            allocation,
            horizon_start_year=horizon_start_year,
            horizon_end_year=horizon_end_year,
        )
        scenario_sim = Simulator(scenario_inputs)
        result = scenario_sim.run(
            params,
            np.array(prepared_history.history_state, copy=True),
            np.array(prepared_history.history_acc, copy=True),
            start_day,
            end_day,
            record_days=baseline_result.record_days,
        )
        scenario_results[allocation.strategy_id] = result
        totals = _endpoint_totals(result, prepared_history.history_acc, inputs.grid)
        rows.append(
            _record(
                profile=profile,
                budget=budget,
                allocation=allocation,
                totals=totals,
                baseline=baseline_totals,
                horizon_start_year=horizon_start_year,
                horizon_end_year=horizon_end_year,
            )
        )
    return PairedScenarioRun(
        profile=profile,
        budget=budget,
        horizon_start_year=horizon_start_year,
        horizon_end_year=horizon_end_year,
        baseline_endpoints=baseline_totals,
        rows=tuple(rows),
        history_state=np.array(prepared_history.history_state, copy=True),
        history_acc=np.array(prepared_history.history_acc, copy=True),
        baseline_result=baseline_result,
        scenario_results=scenario_results,
    )

