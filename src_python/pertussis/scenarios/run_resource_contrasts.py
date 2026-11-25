"""Command-line runner for paired, resource-constrained core2026 contrasts.

The runner intentionally separates three objects that are often conflated in
model papers: the historical programme, the prospective perturbation, and the
parameter source.  A baseline history is integrated once.  Every intervention
then starts from the same state and cumulative onset counters.

The default ``prior_mode`` is a development mode for checking the scenario
contract.  It is written into the manifest and must not be described as a
country-specific posterior estimate.  A fitted parameter JSON can be supplied
once a country fit has passed the inference checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..inference.core2026 import Core2026Inference, InferenceSettings, STRUCTURAL_NAMES
from ..inputs.bundle import BundleSettings, build_inputs
from ..inputs.programme import load_programme
from ..model.parameters import ModelParameters
from .paired import run_paired_scenarios
from .resource_constrained import BudgetSpec, StrategySpec, allocation_table


PRIMARY_PROFILES = ("United_Kingdom", "Australia")
ALL_PROFILES = (
    "United_Kingdom", "Australia", "Brazil", "China", "Japan",
    "New_Zealand", "Sweden", "Thailand",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_mode_parameters(country: str) -> tuple[ModelParameters, dict[str, float]]:
    """Return a transparent prior-mode parameter set for a country.

    The observation model is not used in the forward run.  Constructing the
    inference object only supplies the country-specific seasonal-peak prior and
    the shared structural prior definitions.
    """

    inference = Core2026Inference(
        country,
        InferenceSettings(n_starts=1, max_iterations=1),
    )
    values = inference.space.natural(inference.space.prior_mode())
    params = ModelParameters(
        R0=values["R0"],
        seasonal_amplitude=values["seasonal_amplitude"],
        seasonal_peak=values["seasonal_peak"],
        npi_scale=values["npi_scale"],
        importation_per_million_per_year=values["importation_per_million_per_year"],
        natural_immunity_years=values["natural_immunity_years"],
        boosting_kappa=values["boosting_kappa"],
        wP_years=values["wP_years"],
        aP_years=values["aP_years"],
        tdap_years=values["tdap_years"],
        sigma_aP=values["sigma_aP"],
        sigma_tdap=values["sigma_aP"],
        pi_aP_waned=values["pi_aP_waned"],
    )
    return params, {name: float(values[name]) for name in STRUCTURAL_NAMES}


def _json_parameters(path: Path) -> tuple[ModelParameters, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("parameters", payload)
    missing = [name for name in STRUCTURAL_NAMES if name not in values]
    if missing:
        raise ValueError(f"parameter JSON is missing structural names: {missing}")
    params = ModelParameters(
        **{name: float(values[name]) for name in (
            "R0", "seasonal_amplitude", "seasonal_peak", "npi_scale",
            "importation_per_million_per_year", "natural_immunity_years",
            "boosting_kappa", "wP_years", "aP_years", "tdap_years",
            "sigma_aP", "pi_aP_waned",
        )},
        sigma_tdap=float(values["sigma_aP"]),
    )
    return params, {name: float(values[name]) for name in STRUCTURAL_NAMES}


def _strategy_specs(
    inputs,
    country: str,
    budget: BudgetSpec,
    *,
    prospective_year: int = 2027,
    programme=None,
    adolescent_age: float = 12.0,
) -> tuple[StrategySpec, ...]:
    """Build one strategy set with the same budget denominator contract.

    The maternal and routine denominators are annual births.  The adolescent
    denominator is the modelled population in the selected adolescent cohort at the first
    prospective month.  The timing shift preserves dose count and therefore
    uses the routine denominator only for the audit row.
    """

    cal, grid = inputs.calendar, inputs.grid
    start = cal.year_start_day(prospective_year)
    end = cal.year_start_day(prospective_year + 1)
    annual_births = float(inputs.births_per_day[start:end].sum())
    # Use the cohort actually targeted by the national comparison.
    if not np.isfinite(adolescent_age) or not 10 <= adolescent_age <= 19:
        raise ValueError('adolescent target age must lie in [10, 19] years')
    target_bin = grid.cohort_bin(int(np.floor(adolescent_age + .5)))
    adolescent_population = float(inputs.nudge_target[inputs.calendar.month_idx[start], target_bin])
    if annual_births <= 0 or adolescent_population <= 0:
        raise ValueError(f"invalid denominators for {country}: births={annual_births}, adolescent_population={adolescent_population}")

    if programme is not None:
        if programme.iso3 != inputs.metadata['iso3']:
            raise ValueError('strategy programme and model geography differ')
    elif country == "England":
        if inputs.metadata.get('childhood_coverage_variant')=='england_cover_cohort':
            from ..inputs.england_cover import load_england_cover_programme
            programme=load_england_cover_programme()
        else:
            from ..inputs.england import load_england_programme
            programme = load_england_programme()
    else:
        programme = load_programme(REPO_ROOT / "config" / "core2026" / "programmes" / f"{country}.yaml")
    # Maternal effective protection is encoded as coverage multiplied by the
    # programme's effective share.  Recover the raw coverage from the active
    # period rather than baking a universal 0.9 multiplier into the runner.
    active_maternal = next(
        (period for period in programme.maternal if period.active(np.datetime64(f"{prospective_year}-01-01").astype(object))),
        None,
    )
    if active_maternal is None:
        maternal = 0.0
    else:
        maternal = float(programme.coverage(active_maternal.coverage, prospective_year))
    # Target the named routine component that is active in the programme.  A
    # country with an 18-month booster uses that component; other profiles use
    # third-dose completion.  The selected kind is carried into the allocator
    # so the mutation and denominator refer to the same dose.
    routine_kind = 3 if "booster_18m" in programme.coverage_series else 2
    routine_series = "booster_18m" if routine_kind == 3 else "dtp3"
    routine = float(programme.coverage(routine_series, prospective_year))
    # A programme at another age is retained, but is not baseline uptake of
    # this target cohort. In particular Sweden's age-15 booster is not age 12.
    k = int(np.floor(adolescent_age + .5))
    event_day = start + int(round((adolescent_age + .5 - k) * 365.25))
    event_date = cal.dates[event_day].astype(object)
    active_adolescent = [d for d in programme.cohort_doses
        if d.kind == 'tdap_booster' and np.isclose(d.age, adolescent_age)
        and d.active(event_date)
        and d.applies_to_birth_year(prospective_year-int(np.floor(adolescent_age+.5)))]
    if len(active_adolescent) > 1:
        raise ValueError('multiple adolescent programme streams at the target age')
    adolescent = float(programme.coverage(active_adolescent[0].coverage, prospective_year)) if active_adolescent else 0.0
    # The strategy set uses each target's actual baseline coverage.  The fixed
    # budget is converted to a coverage increment by the allocator.
    return (
        StrategySpec("maternal_scaleup", "maternal", annual_births, maternal),
        StrategySpec("routine_scaleup", "routine", annual_births, routine, routine_dose_kind=routine_kind),
        StrategySpec("routine_timing_shift_minus_1_month", "timing", annual_births, routine, timing_shift_months=-1, routine_dose_kind=routine_kind),
        StrategySpec("adolescent_one_time_booster", "adolescent", adolescent_population, adolescent, one_time=True, dose_age_years=adolescent_age),
    )


def run_profile(
    country: str,
    output_dir: Path,
    *,
    horizon_start: int = 2027,
    horizon_end: int = 2036,
    budget_levels: tuple[tuple[str, float], ...] = (("low", 10.0), ("medium", 20.0), ("high", 30.0)),
    parameter_json: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if horizon_end <= horizon_start:
        raise ValueError("horizon_end must be greater than horizon_start")
    settings = BundleSettings(
        history_start_year=1970,
        spinup_years=70,
        last_year=horizon_end,
        reference_year=2019,
    )
    programme_path = REPO_ROOT / "config" / "core2026" / "programmes" / f"{country}.yaml"
    if parameter_json is None:
        params, parameter_values = _prior_mode_parameters(country)
        parameter_source = "prior_mode"
    else:
        params, parameter_values = _json_parameters(parameter_json)
        parameter_source = str(parameter_json)
    inputs, audit = build_inputs(country, settings, programme=load_programme(programme_path))
    # Build the budget set from the actual profile denominators.  All strategy
    # rows then receive precisely the same nominal dose budget.
    dummy = BudgetSpec("low", 1.0, 1000.0)
    strategies = _strategy_specs(inputs, country, dummy, prospective_year=horizon_start)
    rows, allocations = [], []
    for budget_id, doses_per_1000 in budget_levels:
        denominator = float(strategies[0].eligible_population)
        budget = BudgetSpec(budget_id, doses_per_1000, denominator)
        # Recreate StrategySpec with the budget's common scale only for the
        # audit.  Allocation itself uses each strategy's own denominator.
        run = run_paired_scenarios(
            inputs,
            params,
            budget,
            strategies,
            horizon_start_year=horizon_start,
            horizon_end_year=horizon_end,
            profile=country,
            require_full_budget=True,
        )
        rows.extend(run.records())
        # Allocation fields are stored in the run rows, which keeps this CSV
        # directly traceable to the outcome table.
        allocations.extend([row for row in run.records() if row["strategy_id"] != "baseline"])
    result = pd.DataFrame(rows)
    allocation_frame = pd.DataFrame(allocations)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / f"{country}_paired_results.csv", index=False)
    allocation_frame.to_csv(output_dir / f"{country}_allocation_audit.csv", index=False)
    audit.to_csv(output_dir / f"{country}_programme_audit.csv", index=False)
    manifest = {
        "analysis_id": "endpoint_specific_resource_contrasts_v1",
        "model_family": "core2026",
        "country": country,
        "parameter_source": parameter_source,
        "parameters": parameter_values,
        "horizon_start_year": horizon_start,
        "horizon_end_year_exclusive": horizon_end,
        "budget_levels": [{"budget_id": k, "doses_per_1000": v} for k, v in budget_levels],
        "budget_denominator": "annual_births_common_dose_mass",
        "common_dose_mass_across_strategies": True,
        "programme_path": str(programme_path),
        "programme_sha256": _sha256(programme_path),
        "shared_historical_state": True,
        "typical_onset_endpoint": True,
    }
    (output_dir / f"{country}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result, allocation_frame, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", default=list(PRIMARY_PROFILES), choices=ALL_PROFILES)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "core2026" / "resource_contrasts_pilot")
    parser.add_argument("--horizon-start", type=int, default=2027)
    parser.add_argument("--horizon-end", type=int, default=2036, help="exclusive end year")
    parser.add_argument("--parameter-json", type=Path)
    args = parser.parse_args(argv)
    for country in args.countries:
        result, allocation, manifest = run_profile(
            country,
            args.output_dir,
            horizon_start=args.horizon_start,
            horizon_end=args.horizon_end,
            parameter_json=args.parameter_json,
        )
        print(json.dumps({"country": country, "rows": len(result), "allocation_rows": len(allocation), "parameter_source": manifest["parameter_source"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
