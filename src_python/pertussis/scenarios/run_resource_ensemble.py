"""Run a small paired prior/structure ensemble for the resource contrasts.

This is an uncertainty screen, not a posterior probability over models.  The
output explicitly records the parameter source and draw so ranking
frequencies cannot be mistaken for calibrated country probabilities.  A
country-specific M3 MAP table or posterior-draw table can be supplied to
replace the prior draws.  This keeps the observation model in the inference
pipeline while retaining the paired prospective comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..inference.core2026 import Core2026Inference, InferenceSettings
from ..inputs.bundle import BundleSettings, build_inputs
from ..inputs.programme import load_programme
from ..model.parameters import ModelParameters
from .calibration import (
    effective_sample_size,
    normalized_tempered_weights,
    score_observation_model,
)
from .paired import prepare_paired_history, run_paired_scenarios, annual_endpoint_totals
from .resource_constrained import BudgetSpec
from .decision import benefit_decisions, summarize_benefit_decisions
from .run_resource_contrasts import (
    ALL_PROFILES,
    PRIMARY_PROFILES,
    STRUCTURAL_NAMES,
    _strategy_specs,
)


def _fitted_parameter_path(country: str, directory: Path) -> Path:
    """Resolve an M3 output directory to the country posterior-draw table."""

    candidates = (
        directory / country / f"{country}_posterior_draws.csv",
        directory / country / f"{country}_posterior_samples.csv",
        directory / f"{country}_posterior_draws.csv",
        directory / f"{country}_posterior_samples.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no posterior-draw table for {country!r} under {directory}; "
        f"looked for {[str(path) for path in candidates]}"
    )


def _map_parameter_path(country: str, directory: Path) -> Path:
    """Resolve an M3 output directory to the convergence summary."""

    candidates = (
        directory / country / "convergence_summary.csv",
        directory / "convergence_summary.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no M3 convergence summary for {country!r} under {directory}; "
        f"looked for {[str(path) for path in candidates]}"
    )


def _validate_structural_frame(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    """Return finite structural columns, failing loudly on stale output."""

    missing = [name for name in STRUCTURAL_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing structural parameter columns: {missing}")
    out = frame.copy()
    for name in STRUCTURAL_NAMES:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.loc[np.isfinite(out[list(STRUCTURAL_NAMES)]).all(axis=1)].copy()
    if out.empty:
        raise ValueError(f"{path} has no finite structural parameter rows")
    return out


def _parameter_draws(
    country: str,
    inference: Core2026Inference,
    rng: np.random.Generator,
    n_draws: int,
    *,
    posterior_dir: Path | None = None,
    map_dir: Path | None = None,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Load M3 draws/MAP values or fall back to the declared prior.

    Posterior rows are sampled uniformly from the retained table.  The caller
    records the source and row metadata, so this function never silently turns
    a posterior table into prior uncertainty or vice versa.  MCMC convergence
    diagnostics are copied into the manifest when present; they are not
    promoted to a quality claim by this scenario runner.
    """

    if posterior_dir is not None and map_dir is not None:
        raise ValueError("posterior_dir and map_dir are mutually exclusive")
    if posterior_dir is not None:
        path = _fitted_parameter_path(country, posterior_dir)
        frame = _validate_structural_frame(pd.read_csv(path), path=path)
        if "weight" in frame.columns:
            weights = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(float)
            weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
            if weights.sum() > 0.0:
                weights = weights / weights.sum()
                indices = rng.choice(len(frame), size=n_draws, replace=True, p=weights)
            else:
                indices = rng.integers(0, len(frame), size=n_draws)
        else:
            indices = rng.integers(0, len(frame), size=n_draws)
        draws = []
        for index in indices:
            row = frame.iloc[int(index)]
            names = getattr(inference.space, "names", STRUCTURAL_NAMES)
            values = {
                name: float(row[name]) for name in names
                if name in row.index and pd.notna(row[name])
            }
            values.update({name: float(row[name]) for name in STRUCTURAL_NAMES})
            if "log_posterior" in row.index and pd.notna(row["log_posterior"]):
                values["_log_posterior"] = float(row["log_posterior"])
            values["_source_row"] = int(index)
            draws.append(values)
        metadata: dict[str, object] = {
            "parameter_source": "m3_posterior_draw",
            "parameter_source_path": str(path),
            "parameter_source_rows": int(len(frame)),
            "parameter_sampling": "importance_weighted" if "weight" in frame.columns else "uniform_rows",
        }
        if "weight" in frame.columns:
            finite_weights = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(float)
            finite_weights = finite_weights[np.isfinite(finite_weights) & (finite_weights >= 0.0)]
            if finite_weights.size and finite_weights.sum() > 0.0:
                normalized = finite_weights / finite_weights.sum()
                metadata["importance_effective_sample_size"] = float(1.0 / np.sum(normalized ** 2))
                metadata["importance_max_weight"] = float(np.max(normalized))
        diagnostics_name = path.name.replace("_posterior_draws.csv", "_mcmc_diagnostics.csv").replace(
            "_posterior_samples.csv", "_mcmc_diagnostics.csv"
        )
        diagnostics = path.with_name(diagnostics_name)
        if diagnostics.exists():
            diagnostic_frame = pd.read_csv(diagnostics)
            if not diagnostic_frame.empty:
                metadata["mcmc_diagnostics_path"] = str(diagnostics)
                for column in (
                    "mean_acceptance", "max_split_rhat", "max_per_walker_split_rhat",
                    "min_effective_draws", "max_autocorrelation_steps",
                ):
                    if column in diagnostic_frame.columns and pd.notna(diagnostic_frame.iloc[0][column]):
                        metadata[f"mcmc_{column}"] = float(diagnostic_frame.iloc[0][column])
        return draws, metadata
    if map_dir is not None:
        path = _map_parameter_path(country, map_dir)
        frame = _validate_structural_frame(pd.read_csv(path), path=path)
        if "selected" in frame.columns:
            selected = frame[frame["selected"].astype(str).str.lower().isin({"true", "1"})]
            if not selected.empty:
                frame = selected
        if "success" in frame.columns:
            successful = frame[frame["success"].astype(str).str.lower().isin({"true", "1"})]
            if not successful.empty:
                frame = successful
        if "objective" in frame.columns:
            frame = frame.sort_values("objective", kind="stable")
        row = frame.iloc[0]
        names = getattr(inference.space, "names", STRUCTURAL_NAMES)
        values = {
            name: float(row[name]) for name in names
            if name in row.index and pd.notna(row[name])
        }
        values.update({name: float(row[name]) for name in STRUCTURAL_NAMES})
        draws = [dict(values) for _ in range(n_draws)]
        return draws, {
            "parameter_source": "m3_map",
            "parameter_source_path": str(path),
            "parameter_source_rows": int(len(frame)),
        }
    draws = []
    for _ in range(n_draws):
        values = inference.space.natural(inference.space.sample_prior(rng))
        draws.append({name: float(value) for name, value in values.items()})
    return draws, {"parameter_source": "independent_prior_draw", "parameter_source_rows": n_draws}


def _parameters(values: dict[str, float], *, vaccine_waning_shape: str) -> ModelParameters:
    return ModelParameters(
        R0=float(values["R0"]),
        seasonal_amplitude=float(values["seasonal_amplitude"]),
        seasonal_peak=float(values["seasonal_peak"]),
        npi_scale=float(values["npi_scale"]),
        importation_per_million_per_year=float(values["importation_per_million_per_year"]),
        natural_immunity_years=float(values["natural_immunity_years"]),
        boosting_kappa=float(values["boosting_kappa"]),
        wP_years=float(values["wP_years"]),
        aP_years=float(values["aP_years"]),
        tdap_years=float(values["tdap_years"]),
        sigma_aP=float(values["sigma_aP"]),
        sigma_tdap=float(values["sigma_aP"]),
        pi_aP_waned=float(values["pi_aP_waned"]),
        vaccine_waning_shape=vaccine_waning_shape,
    )


def _decision_metrics(result: pd.DataFrame) -> pd.DataFrame:
    endpoints = {
        "infant_0_2m_cases": "infant_0_2m_cases",
        "age1_17_cases": "age1_17_cases",
    }
    rows: list[dict[str, object]] = []
    scenario = result[result.strategy_id != "baseline"].copy()
    for (profile, structure, draw, budget), group in scenario.groupby(
        ["profile", "structure_id", "draw_id", "budget_id"], sort=True
    ):
        for endpoint, column in endpoints.items():
            ordered = group.sort_values(column, kind="stable")
            best = float(ordered.iloc[0][column])
            for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
                regret = float((row[column] - best) / best) if best > 0 else np.nan
                rows.append({
                    "profile": profile,
                    "structure_id": structure,
                    "draw_id": int(draw),
                    "budget_id": budget,
                    "endpoint": endpoint,
                    "strategy_id": row.strategy_id,
                    "rank": rank,
                    "relative_regret": regret,
                    "near_equivalent_1pct": bool(regret <= 0.01),
                    "cases": float(row[column]),
                    "calibration_weight": float(row["calibration_weight"])
                    if "calibration_weight" in row.index and pd.notna(row["calibration_weight"])
                    else np.nan,
                })
    return pd.DataFrame(rows)


def run_ensemble(
    countries: tuple[str, ...],
    output_dir: Path,
    *,
    n_draws: int = 8,
    seed: int = 20261024,
    horizon_start: int = 2027,
    horizon_end: int = 2036,
    structures: tuple[str, ...] = ("erlang2", "exponential"),
    posterior_dir: Path | None = None,
    map_dir: Path | None = None,
    calibration_power: float | None = None,
    calibration_train_start: int = 2015,
    calibration_train_end: int = 2018,
    calibration_holdout_years: tuple[int, ...] = (2019,),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    annual_rows: list[dict[str, object]] = []
    calibration_records: list[dict[str, object]] = []
    calibration_inferences: dict[tuple[str, str], Core2026Inference] = {}
    parameter_meta_by_country: dict[str, dict[str, object]] = {}
    for country_index, country in enumerate(countries):
        inference = Core2026Inference(country, InferenceSettings(n_starts=1, max_iterations=1))
        settings = BundleSettings(
            history_start_year=1970,
            spinup_years=70,
            last_year=horizon_end,
            reference_year=2019,
        )
        programme_path = REPO_ROOT / "config" / "core2026" / "programmes" / f"{country}.yaml"
        inputs, _ = build_inputs(country, settings, programme=load_programme(programme_path))
        strategies = _strategy_specs(
            inputs,
            country,
            BudgetSpec("contract", 1.0, 1000.0),
            prospective_year=horizon_start,
        )
        rng = np.random.default_rng(seed + country_index)
        parameter_draws, parameter_meta = _parameter_draws(
            country,
            inference,
            rng,
            n_draws,
            posterior_dir=posterior_dir,
            map_dir=map_dir,
        )
        parameter_meta_by_country[country] = dict(parameter_meta)
        for draw_id, draw_values in enumerate(parameter_draws):
            structural = {name: float(draw_values[name]) for name in STRUCTURAL_NAMES}
            for structure_id in structures:
                if calibration_power is not None:
                    required_observation = (
                        "reporting_logit_<1", "reporting_logit_1-4", "reporting_logit_5-9",
                        "reporting_logit_10-19", "reporting_logit_20-59", "reporting_logit_60+",
                        "breakthrough_reporting_ratio", "nb_size", "dirichlet_concentration",
                    )
                    missing_observation = [name for name in required_observation if name not in draw_values]
                    if missing_observation:
                        raise ValueError(
                            "observation-aware calibration requires natural observation coordinates; "
                            f"missing for {country} draw {draw_id}: {missing_observation}"
                        )
                    variant = "exponential_vaccine_waning" if structure_id == "exponential" else "baseline"
                    key = (country, structure_id)
                    if key not in calibration_inferences:
                        calibration_inferences[key] = Core2026Inference(
                            country,
                            InferenceSettings(n_starts=1, max_iterations=1),
                            variant=variant,
                        )
                    score = score_observation_model(
                        calibration_inferences[key],
                        draw_values,
                        train_start_year=calibration_train_start,
                        train_end_year=calibration_train_end,
                        holdout_years=calibration_holdout_years,
                    )
                    calibration_records.append({
                        "profile": country,
                        "structure_id": structure_id,
                        "draw_id": draw_id,
                        "score": score,
                    })
                params = _parameters(structural, vaccine_waning_shape=structure_id)
                prepared_history = prepare_paired_history(
                    inputs,
                    params,
                    horizon_start_year=horizon_start,
                    horizon_end_year=horizon_end,
                )
                for budget_id, doses_per_1000 in (("low", 10.0), ("medium", 20.0), ("high", 30.0)):
                    # The common budget is defined on annual births.  Each
                    # strategy then converts the same dose mass to its own
                    # target-specific coverage increment in the allocator.
                    budget = BudgetSpec(
                        budget_id,
                        doses_per_1000,
                        float(strategies[0].eligible_population),
                    )
                    run = run_paired_scenarios(
                        inputs,
                        params,
                        budget,
                        strategies,
                        horizon_start_year=horizon_start,
                        horizon_end_year=horizon_end,
                        profile=country,
                        require_full_budget=True,
                        prepared_history=prepared_history,
                    )
                    frame = run.to_dataframe()
                    frame["draw_id"] = draw_id
                    frame["structure_id"] = structure_id
                    frame["parameter_source"] = parameter_meta["parameter_source"]
                    if "_source_row" in draw_values:
                        frame["parameter_source_row"] = int(draw_values["_source_row"])
                    if "_log_posterior" in draw_values:
                        frame["parameter_log_posterior"] = float(draw_values["_log_posterior"])
                    for name, value in structural.items():
                        frame[f"param_{name}"] = value
                    all_rows.append(frame)
                    for strategy, trajectory in {"baseline": run.baseline_result, **run.scenario_results}.items():
                        for annual in annual_endpoint_totals(trajectory, inputs.grid):
                            interval = annual.pop("interval_index")
                            annual_rows.append({"profile": country, "structure_id": structure_id,
                                "draw_id": draw_id, "budget_id": budget_id,
                                "strategy_id": strategy, "year": horizon_start + interval, **annual})
    result = pd.concat(all_rows, ignore_index=True)
    calibration_ess: dict[str, float] = {}
    if calibration_power is not None:
        if not calibration_records:
            raise RuntimeError("calibration requested but no observation scores were produced")
        score_frame = pd.DataFrame(calibration_records)
        for country, group in score_frame.groupby("profile", sort=False):
            weights = normalized_tempered_weights(
                [row.score for row in group.itertuples(index=False)],
                power=float(calibration_power),
            )
            score_frame.loc[group.index, "calibration_weight"] = weights
            calibration_ess[country] = effective_sample_size(weights)
        score_frame["score_total_nll"] = score_frame["score"].map(lambda x: float(x.total_nll))
        score_frame["score_age_nll"] = score_frame["score"].map(lambda x: float(x.age_nll))
        score_frame["calibration_train_nll"] = score_frame["score"].map(lambda x: float(x.train_nll))
        score_frame["calibration_holdout_nll"] = score_frame["score"].map(lambda x: float(x.holdout_nll))
        score_frame["calibration_train_case_years"] = score_frame["score"].map(lambda x: ",".join(map(str, x.train_case_years)))
        score_frame["calibration_holdout_case_years"] = score_frame["score"].map(lambda x: ",".join(map(str, x.holdout_case_years)))
        score_frame = score_frame.drop(columns=["score"])
        result = result.merge(
            score_frame,
            on=["profile", "structure_id", "draw_id"],
            how="left",
            validate="many_to_one",
        )
    # Add dose-normalised effects only after all paired baselines are present.
    # The endpoint differences are cumulative over the full prospective
    # horizon, so the denominator must use the matching horizon dose count,
    # not the annual allocation.  Using annual doses would inflate effects by
    # the number of projection years.
    for endpoint in ("infant_0_2m", "age1_17"):
        diff = f"{endpoint}_absolute_difference_vs_baseline"
        doses = result["horizon_doses_used"].replace(0.0, np.nan)
        result[f"{endpoint}_cases_averted_per_1000_doses"] = -result[diff] / doses * 1000.0
    decision = _decision_metrics(result)
    if calibration_power is not None:
        decision["weighted_rank1"] = decision["calibration_weight"] * (decision["rank"] == 1)
        decision["weighted_top2"] = decision["calibration_weight"] * (decision["rank"] <= 2)
        decision["weighted_near"] = decision["calibration_weight"] * decision["near_equivalent_1pct"]
    summary = (
        decision.groupby(["profile", "budget_id", "endpoint", "strategy_id"], as_index=False)
        .agg(
            ensemble_replicates=("rank", "size"),
            ensemble_draws=("draw_id", "nunique"),
            structures=("structure_id", "nunique"),
            rank1_frequency=("rank", lambda x: float(np.mean(np.asarray(x) == 1))),
            top2_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 2))),
            near_equivalent_frequency=("near_equivalent_1pct", "mean"),
            median_relative_regret=("relative_regret", "median"),
            **({
                "weighted_rank1_frequency": ("weighted_rank1", "sum"),
                "weighted_top2_frequency": ("weighted_top2", "sum"),
                "weighted_near_equivalent_frequency": ("weighted_near", "sum"),
            } if calibration_power is not None else {}),
        )
    )
    result.to_csv(output_dir / "resource_contrast_ensemble_results.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(output_dir / "resource_contrast_annual_endpoints.csv", index=False)
    decision.to_csv(output_dir / "resource_contrast_ensemble_decision_metrics.csv", index=False)
    summary.to_csv(output_dir / "resource_contrast_ensemble_summary.csv", index=False)
    benefit = pd.concat([benefit_decisions(result, include_timing=flag)
                         for flag in (False, True)], ignore_index=True)
    benefit.to_csv(output_dir / "benefit_decisions.csv", index=False)
    summarize_benefit_decisions(benefit).to_csv(output_dir / "benefit_summary.csv", index=False)
    source_by_country = {}
    for country_index, country in enumerate(countries):
        # Reconstruct only the source metadata for the manifest.  The actual
        # draw selection is already recorded row-wise in the result table.
        source_by_country[country] = str(
            result.loc[result["profile"].eq(country), "parameter_source"].dropna().iloc[0]
        )
    manifest = {
        "analysis_id": "endpoint_specific_resource_contrasts_v1",
        "model_family": "core2026",
        "profiles": list(countries),
        "n_parameter_draws_per_country": n_draws,
        "structures": list(structures),
        "seed": seed,
        "parameter_source": (
            "m3_posterior_draws" if posterior_dir is not None else
            "m3_map" if map_dir is not None else "independent_prior_draws"
        ),
        "parameter_source_by_country": source_by_country,
        "parameter_metadata_by_country": parameter_meta_by_country,
        "posterior_draws_are_diagnostic": bool(posterior_dir is not None),
        "observation_calibration": {
            "enabled": calibration_power is not None,
            "likelihood_power": None if calibration_power is None else float(calibration_power),
            "train_start_year": int(calibration_train_start),
            "train_end_year": int(calibration_train_end),
            "holdout_years": [int(year) for year in calibration_holdout_years],
            "effective_sample_size_by_profile": calibration_ess,
            "score_definition": "core2026 NB2 interval likelihood plus age-composition Dirichlet-multinomial; weights use training years only",
            "interpretation": "tempered likelihood calibration screen, not a converged posterior",
        },
        "horizon_start_year": horizon_start,
        "horizon_end_year_exclusive": horizon_end,
        "budgets_doses_per_1000": {"low": 10, "medium": 20, "high": 30},
        "budget_denominator": "annual_births_common_dose_mass",
        "common_dose_mass_across_strategies": True,
        "ranking_frequency_interpretation": "ensemble proportion, not posterior probability",
        "benefit_decision_contract": {
            "primary_set": "three additional-dose strategies plus no-change baseline",
            "timing_set": "all-four sensitivity; rescheduled doses are not additional doses",
            "loss_denominator": "best positive cases averted versus baseline",
            "no_positive_benefit": "undefined",
            "thresholds": [0.05, 0.10, 0.20],
        },
        "historical_state_reused_across_budgets": True,
    }
    (output_dir / "resource_contrast_ensemble_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", default=list(PRIMARY_PROFILES), choices=ALL_PROFILES)
    parser.add_argument("--n-draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "core2026" / "resource_contrasts_ensemble_all8")
    parser.add_argument("--horizon-start", type=int, default=2027)
    parser.add_argument("--horizon-end", type=int, default=2036, help="exclusive end year")
    parser.add_argument(
        "--posterior-dir", type=Path,
        help="M3 output directory containing country posterior-draw CSVs",
    )
    parser.add_argument(
        "--map-dir", type=Path,
        help="M3 output directory containing country convergence_summary.csv",
    )
    parser.add_argument(
        "--calibration-power", type=float, default=None,
        help="enable observation-aware tempered likelihood weighting (for example 0.01); weights use 2015-2018 and hold out 2019",
    )
    parser.add_argument("--calibration-train-start", type=int, default=2015)
    parser.add_argument("--calibration-train-end", type=int, default=2018)
    parser.add_argument("--calibration-holdout-years", type=int, nargs="+", default=[2019])
    args = parser.parse_args(argv)
    _, summary = run_ensemble(
        tuple(args.countries),
        args.output_dir,
        n_draws=args.n_draws,
        seed=args.seed,
        horizon_start=args.horizon_start,
        horizon_end=args.horizon_end,
        posterior_dir=args.posterior_dir,
        map_dir=args.map_dir,
        calibration_power=args.calibration_power,
        calibration_train_start=args.calibration_train_start,
        calibration_train_end=args.calibration_train_end,
        calibration_holdout_years=tuple(args.calibration_holdout_years),
    )
    print(json.dumps({"rows": int(len(summary)), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
