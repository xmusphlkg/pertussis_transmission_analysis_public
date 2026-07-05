from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import (
    load_configs,
    make_intervention_config,
    run_scenario_list,
)
from src_python.simulation.run_routine_timeliness_sensitivity import _apply_timeliness
from src_python.utils.io import project_path, write_dataframe


NEAR_TERM_END_DATE = "2029-12-31"
NEAR_TERM_END_DAYS = 365.0 * 5.0
ROUTINE_DELIVERY_DELAY_REDUCTION = 0.25
REFERENCE_SCENARIO = "no_shock_current_delivery"

NPI_MEAN_OVERRIDES = {
    "transmission": {
        "npi_contact_reduction": {
            "enabled": True,
            "reduction_column": "contact_reduction_mean",
        }
    }
}

SCENARIO_DEFINITIONS = (
    {
        "scenario": REFERENCE_SCENARIO,
        "contact_assumption": "none",
        "routine_delay_reduction": 0.0,
        "timeliness_recovery": False,
        "note": "Counterfactual without COVID-19 contact reduction and without routine-delivery delay.",
    },
    {
        "scenario": "npi_contact_shock_only",
        "contact_assumption": "mean",
        "routine_delay_reduction": 0.0,
        "timeliness_recovery": False,
        "note": "COVID-19 NPI contact-reduction shock only, with routine delivery unchanged.",
    },
    {
        "scenario": "routine_delay_only",
        "contact_assumption": "none",
        "routine_delay_reduction": ROUTINE_DELIVERY_DELAY_REDUCTION,
        "timeliness_recovery": False,
        "note": "Reduced routine-vaccination delivery during the COVID-19 calendar window, without contact reduction.",
    },
    {
        "scenario": "combined_shock_delayed_recovery",
        "contact_assumption": "mean",
        "routine_delay_reduction": ROUTINE_DELIVERY_DELAY_REDUCTION,
        "timeliness_recovery": False,
        "note": "NPI contact reduction plus reduced routine-vaccination delivery during the COVID-19 calendar window.",
    },
    {
        "scenario": "combined_shock_timeliness_recovery",
        "contact_assumption": "mean",
        "routine_delay_reduction": ROUTINE_DELIVERY_DELAY_REDUCTION,
        "timeliness_recovery": True,
        "note": "Combined COVID-19 shock with post-shock routine-timeliness recovery from 2025 onward.",
    },
    {
        "scenario": "baseline_screened_contact_shock",
        "contact_assumption": "baseline_screened",
        "routine_delay_reduction": 0.0,
        "timeliness_recovery": False,
        "note": "Submitted baseline contact-reduction setting after calibration-screening country peak timing.",
    },
)


def _set_near_term_runtime(config: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    out.setdefault("calendar", {})["analysis_end_date"] = NEAR_TERM_END_DATE
    out.setdefault("simulation", {})["end_time"] = NEAR_TERM_END_DAYS
    out["simulation"]["output_time_step"] = 30.0
    out["simulation"]["rtol"] = max(float(out["simulation"].get("rtol", 1e-5)), 1e-4)
    out["simulation"]["atol"] = max(float(out["simulation"].get("atol", 1e-7)), 1e-6)
    return out


def _remove_npi_contact_shock(config: dict[str, Any]) -> None:
    transmission = config.setdefault("transmission", {})
    transmission.pop("npi_contact_reduction_periods", None)
    settings = transmission.get("npi_contact_reduction", {})
    if isinstance(settings, dict):
        transmission["npi_contact_reduction"] = {**settings, "enabled": False}


def _apply_routine_delivery_delay_from_npi_periods(config: dict[str, Any], reduction: float) -> None:
    if reduction <= 0.0:
        return
    periods = config.get("transmission", {}).get("npi_contact_reduction_periods", [])
    if not periods:
        return

    routine = config.setdefault("routine_vaccination", {})
    routine["delivery_shock_periods"] = [
        {
            "start_date": period["start_date"],
            "end_date": period["end_date"],
            "reduction": float(reduction),
            "ramp_days": float(period.get("ramp_days", 180.0)),
            "source": "covid_npi_period",
        }
        for period in periods
    ]
    routine["delivery_shock_note"] = (
        "Reduced routine-vaccination target-relaxation flow during configured "
        "COVID-19/NPI periods, with linear recovery over each period ramp."
    )


def _build_country_config(
    country: str,
    definition: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    contact_assumption = str(definition["contact_assumption"])
    config_overrides = NPI_MEAN_OVERRIDES if contact_assumption in {"mean", "none"} else None
    config, vaccine_name = make_intervention_config(
        "current",
        country_profile=country,
        config_overrides=config_overrides,
    )
    config = _set_near_term_runtime(config)
    _apply_routine_delivery_delay_from_npi_periods(config, float(definition["routine_delay_reduction"]))
    if contact_assumption == "none":
        _remove_npi_contact_shock(config)
    if bool(definition["timeliness_recovery"]):
        config = _apply_timeliness(config)
    return config, vaccine_name


def _build_scenarios(configs: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    resistance_name = configs["baseline"].get("baseline_resistance_scenario", "country_timeline")
    for country in configs["countries"]:
        for definition in SCENARIO_DEFINITIONS:
            config, vaccine_name = _build_country_config(country, definition)
            contact_assumption = str(definition["contact_assumption"])
            contact_shock_applied = contact_assumption in {"mean", "baseline_screened"}
            routine_delay = float(definition["routine_delay_reduction"])
            scenarios.append(
                {
                    "config": config,
                    "analysis": "shock_recovery_sensitivity",
                    "scenario": str(definition["scenario"]),
                    "vaccine_scenario": vaccine_name,
                    "resistance_scenario": resistance_name,
                    "intervention": "current",
                    "metadata": {
                        "country": country,
                        "contact_assumption": contact_assumption,
                        "contact_shock_applied": bool(contact_shock_applied),
                        "routine_delivery_delay_applied": bool(routine_delay > 0.0),
                        "routine_delivery_delay_reduction": routine_delay if routine_delay > 0.0 else np.nan,
                        "timeliness_recovery_applied": bool(definition["timeliness_recovery"]),
                        "shock_recovery_window": "COVID-19 burn-in shock with 2025-2029 recovery readout",
                        "implementation_note": str(definition["note"]),
                    },
                }
            )
    return scenarios


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _iqr_text(series: pd.Series) -> str:
    values = _numeric(series).dropna()
    if values.empty:
        return ""
    return f"{values.quantile(0.25):.4g} to {values.quantile(0.75):.4g}"


def _add_reference_changes(country: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "child_adolescent_cases": "annualized_child_adolescent_cases_per_100k",
        "infant_cases": "annualized_infant_cases_per_100k",
        "total_infections": "annualized_infections_per_100k",
    }
    out = country.copy()
    reference = out.loc[out["scenario"].eq(REFERENCE_SCENARIO), ["country", *metrics.values()]].rename(
        columns={value: f"{key}_reference" for key, value in metrics.items()}
    )
    out = out.merge(reference, on="country", how="left")
    delayed = out.loc[
        out["scenario"].eq("combined_shock_delayed_recovery"),
        ["country", *metrics.values()],
    ].rename(columns={value: f"{key}_delayed_recovery" for key, value in metrics.items()})
    out = out.merge(delayed, on="country", how="left")
    for key, column in metrics.items():
        value = _numeric(out[column])
        reference_value = _numeric(out[f"{key}_reference"])
        delayed_value = _numeric(out[f"{key}_delayed_recovery"])
        recovery_comparison = out["scenario"].isin(
            ["combined_shock_delayed_recovery", "combined_shock_timeliness_recovery"]
        )
        out[f"relative_change_{key}_vs_no_shock"] = np.where(
            reference_value > 0.0,
            value / reference_value - 1.0,
            np.nan,
        )
        out[f"relative_reduction_{key}_vs_delayed_recovery"] = np.where(
            recovery_comparison & (delayed_value > 0.0),
            1.0 - value / delayed_value,
            np.nan,
        )
    return out


def _summarize(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = summary.copy()
    country_columns = [
        "country",
        "scenario",
        "contact_assumption",
        "contact_shock_applied",
        "routine_delivery_delay_applied",
        "routine_delivery_delay_reduction",
        "timeliness_recovery_applied",
        "annualized_child_adolescent_cases_per_100k",
        "annualized_infant_cases_per_100k",
        "annualized_infections_per_100k",
        "resistant_fraction_end",
        "implementation_note",
    ]
    country = data[country_columns].copy()
    country = _add_reference_changes(country)

    rows: list[dict[str, Any]] = []
    for scenario, group in country.groupby("scenario", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "countries": int(group["country"].nunique()),
                "contact_assumption": first.get("contact_assumption", ""),
                "contact_shock_applied": bool(first.get("contact_shock_applied", False)),
                "routine_delivery_delay_applied": bool(first.get("routine_delivery_delay_applied", False)),
                "routine_delivery_delay_reduction": first.get("routine_delivery_delay_reduction", np.nan),
                "timeliness_recovery_applied": bool(first.get("timeliness_recovery_applied", False)),
                "median_child_adolescent_cases_per_100k_5y": float(
                    _numeric(group["annualized_child_adolescent_cases_per_100k"]).median(skipna=True)
                ),
                "median_infant_cases_per_100k_5y": float(
                    _numeric(group["annualized_infant_cases_per_100k"]).median(skipna=True)
                ),
                "median_relative_change_child_adolescent_cases_vs_no_shock": float(
                    _numeric(group["relative_change_child_adolescent_cases_vs_no_shock"]).median(skipna=True)
                ),
                "iqr_relative_change_child_adolescent_cases_vs_no_shock": _iqr_text(
                    group["relative_change_child_adolescent_cases_vs_no_shock"]
                ),
                "median_relative_change_infant_cases_vs_no_shock": float(
                    _numeric(group["relative_change_infant_cases_vs_no_shock"]).median(skipna=True)
                ),
                "iqr_relative_change_infant_cases_vs_no_shock": _iqr_text(
                    group["relative_change_infant_cases_vs_no_shock"]
                ),
                "median_relative_reduction_child_adolescent_cases_vs_delayed_recovery": float(
                    _numeric(group["relative_reduction_child_adolescent_cases_vs_delayed_recovery"]).median(skipna=True)
                ),
                "median_relative_reduction_infant_cases_vs_delayed_recovery": float(
                    _numeric(group["relative_reduction_infant_cases_vs_delayed_recovery"]).median(skipna=True)
                ),
                "median_end_resistant_fraction_5y": float(
                    _numeric(group["resistant_fraction_end"]).median(skipna=True)
                ),
                "implementation_note": first.get("implementation_note", ""),
            }
        )
    return pd.DataFrame(rows), country


def main(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = load_configs()
    timeseries, summary = run_scenario_list(
        _build_scenarios(configs),
        stem="shock_recovery_sensitivity",
        reference_scenario=REFERENCE_SCENARIO,
        n_jobs=n_jobs,
    )
    table, country = _summarize(summary)
    write_dataframe(table, project_path("outputs", "tables", "shock_recovery_sensitivity.csv"))
    write_dataframe(country, project_path("outputs", "tables", "shock_recovery_sensitivity_country.csv"))
    return timeseries, summary, table, country


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COVID-19 shock and routine-delivery recovery sensitivity scenarios.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
