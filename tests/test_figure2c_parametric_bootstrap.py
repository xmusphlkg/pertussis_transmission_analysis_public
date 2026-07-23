from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src_python.simulation.audit_figure2c_bootstrap_ci as audit
import src_python.simulation.run_figure2c_parametric_bootstrap as bootstrap


def _formal_programme_configs() -> dict:
    return {
        "baseline": {
            "bayesian_uncertainty": {
                "figure2c_joint_credible_interval": {
                    "intervention_priors": {
                        "adolescent_coverage_floor": {
                            "low": 0.01,
                            "mode": 0.02,
                            "high": 0.03,
                        },
                        "maternal_coverage_floor": {
                            "low": 0.04,
                            "mode": 0.05,
                            "high": 0.06,
                        },
                    }
                }
            }
        },
        "interventions": {
            "adolescent_booster": {
                "coverage_min_updates": {"adolescent_10_17y": 0.90}
            },
            "pregnancy_tdap_scaleup": {
                "coverage_min_updates": {"infant_0_2m": 0.75},
                "vaccine_overrides": {
                    "maternal_VE_sus": 0.55,
                    "maternal_VE_sym": 0.92,
                },
                "natural_history_overrides": {
                    "maternal_protection_duration": 180.0
                },
            },
            "cocooning_adjunct": {
                "coverage_min_updates": {"young_adult_18_39y": 0.55},
                "contact_matrix_reduction": {"reduction_fraction": 0.15},
            },
            "maternal_immunization": {
                "coverage_min_updates": {
                    "infant_0_2m": 0.75,
                    "young_adult_18_39y": 0.55,
                },
                "vaccine_overrides": {
                    "maternal_VE_sus": 0.55,
                    "maternal_VE_sym": 0.92,
                },
                "natural_history_overrides": {
                    "maternal_protection_duration": 180.0
                },
                "contact_matrix_reduction": {"reduction_fraction": 0.15},
            },
            "targeted_pep_high_risk": {
                "pep_updates": {"coverage_household_contacts": 0.45}
            },
        },
    }


def _use_temporary_checkpoint_bundle(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    paths = {
        "metadata": tmp_path / "metadata" / "checkpoint.json",
        "draws": tmp_path / "simulations" / "checkpoint_draws.parquet",
        "fits": tmp_path / "simulations" / "checkpoint_fits.parquet",
        "archive": tmp_path / "archive",
    }
    monkeypatch.setattr(bootstrap, "CHECKPOINT_METADATA_PATH", paths["metadata"])
    monkeypatch.setattr(bootstrap, "CHECKPOINT_DRAW_PATH", paths["draws"])
    monkeypatch.setattr(bootstrap, "CHECKPOINT_FIT_PATH", paths["fits"])
    monkeypatch.setattr(bootstrap, "CHECKPOINT_ARCHIVE_ROOT", paths["archive"])
    paths["metadata"].parent.mkdir(parents=True)
    paths["draws"].parent.mkdir(parents=True)
    return paths


def test_figure2c_contract_is_estimation_confidence_interval_only() -> None:
    assert bootstrap.ANALYSIS_ROLE == "publication_estimation_confidence_interval"
    assert bootstrap.PUBLICATION_PATH is True
    assert bootstrap.FIGURE2C_INTERVAL_SOURCE is True
    assert bootstrap.STATISTICAL_TARGET == "frequentist_confidence_interval"
    assert bootstrap.CONFIDENCE_INTERVAL_METHOD == "percentile_parametric_bootstrap"
    assert bootstrap.VARIED_ESTIMATION_COMPONENTS == (
        "annual_AR1_latent_transmission_path",
        "NB2_surveillance_observations",
        "state_space_MAP_refit",
    )
    assert bootstrap.FIXED_REFERENCE_INPUTS == (
        "biological_parameters",
        "intervention_definitions",
        "AR1_hyperparameters",
        "NB2_dispersion",
    )
    parameters = inspect.signature(bootstrap.run_parametric_bootstrap).parameters
    assert "posterior_samples" not in parameters
    assert "posterior_draws" not in parameters
    assert "intervention_uncertainty" not in parameters


def test_fixed_programme_references_ignore_legacy_prior_modes() -> None:
    first_configs = _formal_programme_configs()
    second_configs = deepcopy(first_configs)
    second_configs["baseline"]["bayesian_uncertainty"][
        "figure2c_joint_credible_interval"
    ]["intervention_priors"]["adolescent_coverage_floor"]["mode"] = 0.99
    second_configs["baseline"]["bayesian_uncertainty"][
        "figure2c_joint_credible_interval"
    ]["intervention_priors"]["maternal_coverage_floor"]["mode"] = 0.98

    first = bootstrap._fixed_intervention_reference_values(first_configs)
    second = bootstrap._fixed_intervention_reference_values(second_configs)

    assert first == second
    assert first == {
        "adolescent_coverage_floor": 0.90,
        "maternal_coverage_floor": 0.75,
        "young_adult_coverage_floor": 0.55,
        "contact_reduction_fraction": 0.15,
        "targeted_pep_coverage": 0.45,
        "maternal_protection_duration_days": 180.0,
        "maternal_VE_sus": 0.55,
        "maternal_VE_sym": 0.92,
    }


def test_sample_row_uses_formal_fixed_programme_references() -> None:
    calibrated = {
        "vaccine": {"VE_sus": 0.4, "VE_inf": 0.2, "VE_dur": 730.0},
        "transmission": {
            "beta_S": 0.3,
            "relative_infectiousness_asymptomatic": 0.5,
            "fitness_R": 0.9,
            "log_beta_time_variation": {
                "periods": [
                    {"start_date": "2020-01-01", "log_multiplier": 0.1}
                ]
            },
        },
        "natural_history": {
            "infectious_duration_symptomatic": 21.0,
            "infectious_duration_asymptomatic": 17.0,
        },
        "initial_conditions": {"initial_resistance_prevalence": 0.02},
        "reporting_multiplier": 1.1,
    }

    row = bootstrap._sample_row_from_refit(
        "A",
        7,
        calibrated,
        _formal_programme_configs(),
    ).iloc[0]

    assert row["intervention_uncertainty_adolescent_coverage_floor"] == 0.90
    assert row["intervention_uncertainty_maternal_coverage_floor"] == 0.75
    assert row["intervention_uncertainty_targeted_pep_coverage"] == 0.45


def test_fixed_programme_references_reject_inconsistent_composite_definition() -> None:
    configs = _formal_programme_configs()
    configs["interventions"]["maternal_immunization"][
        "coverage_min_updates"
    ]["infant_0_2m"] = 0.70

    with pytest.raises(ValueError, match="definitions disagree for maternal coverage"):
        bootstrap._fixed_intervention_reference_values(configs)


def test_refit_contract_rejects_parameter_or_hyperparameter_scope_expansion() -> None:
    valid = SimpleNamespace(
        posterior_coordinate_names=(
            "log_beta_S",
            "log_reporting_multiplier",
            "log_beta_process_2020",
        ),
        posterior_map_vector=np.zeros(3),
        posterior_covariance=np.eye(3),
        posterior_lower_bounds=np.full(3, -1.0),
        posterior_upper_bounds=np.full(3, 1.0),
        measurement_dispersion=50.0,
        process_ar1_rho=0.5,
        process_innovation_sd=0.6,
    )
    bootstrap._assert_estimation_ci_refit_contract(
        valid,
        expected_process_years=np.asarray([2020]),
        dispersion=50.0,
        rho=0.5,
        innovation_sd=0.6,
    )

    sampled_biology = SimpleNamespace(
        **{
            **vars(valid),
            "posterior_coordinate_names": (
                *valid.posterior_coordinate_names,
                "VE_inf",
            ),
        }
    )
    with pytest.raises(RuntimeError, match="estimation-CI contract"):
        bootstrap._assert_estimation_ci_refit_contract(
            sampled_biology,
            expected_process_years=np.asarray([2020]),
            dispersion=50.0,
            rho=0.5,
            innovation_sd=0.6,
        )

    sampled_dispersion = SimpleNamespace(
        **{**vars(valid), "measurement_dispersion": 35.0}
    )
    with pytest.raises(RuntimeError, match="estimation-CI contract"):
        bootstrap._assert_estimation_ci_refit_contract(
            sampled_dispersion,
            expected_process_years=np.asarray([2020]),
            dispersion=50.0,
            rho=0.5,
            innovation_sd=0.6,
        )


@pytest.mark.parametrize(
    "replacement",
    [
        {
            "posterior_coordinate_names": (
                "log_beta_S",
                "log_reporting_multiplier",
            ),
            "posterior_map_vector": np.zeros(2),
            "posterior_covariance": np.eye(2),
            "posterior_lower_bounds": np.full(2, -1.0),
            "posterior_upper_bounds": np.full(2, 1.0),
        },
        {
            "posterior_coordinate_names": (
                "log_beta_S",
                "log_reporting_multiplier",
                "log_beta_process_2021",
            ),
        },
        {"posterior_covariance": np.eye(2)},
    ],
)
def test_refit_contract_requires_exact_annual_coordinates_and_dimensions(
    replacement: dict[str, object],
) -> None:
    valid = {
        "posterior_coordinate_names": (
            "log_beta_S",
            "log_reporting_multiplier",
            "log_beta_process_2020",
        ),
        "posterior_map_vector": np.zeros(3),
        "posterior_covariance": np.eye(3),
        "posterior_lower_bounds": np.full(3, -1.0),
        "posterior_upper_bounds": np.full(3, 1.0),
        "measurement_dispersion": 50.0,
        "process_ar1_rho": 0.5,
        "process_innovation_sd": 0.6,
    }
    invalid = SimpleNamespace(**{**valid, **replacement})

    with pytest.raises(RuntimeError, match="estimation-CI contract"):
        bootstrap._assert_estimation_ci_refit_contract(
            invalid,
            expected_process_years=np.asarray([2020]),
            dispersion=50.0,
            rho=0.5,
            innovation_sd=0.6,
        )


def test_legacy_unconsumed_figure2c_quality_setting_fails_closed() -> None:
    configs = {
        "baseline": {
            "bayesian_uncertainty": {
                "figure2c_parametric_bootstrap_confidence_interval": {
                    "maximum_endpoint_mcse": 0.01
                }
            }
        }
    }

    with pytest.raises(ValueError, match="unconsumed legacy setting"):
        bootstrap._bootstrap_settings(configs)


def test_incompatible_checkpoint_bundle_is_archived_without_data_loss(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _use_temporary_checkpoint_bundle(tmp_path, monkeypatch)
    old_metadata = json.dumps(
        {"countries": ["B"], "replicates": 100, "seed": 7, "maxiter": 30}
    ).encode()
    old_draws = b"old checkpoint draws"
    old_fits = b"old checkpoint fits"
    paths["metadata"].write_bytes(old_metadata)
    paths["draws"].write_bytes(old_draws)
    paths["fits"].write_bytes(old_fits)

    archive_dir = bootstrap._validate_or_create_checkpoint_metadata(
        countries=["A"],
        replicates=100,
        seed=7,
        maxiter=30,
    )

    assert archive_dir is not None
    assert paths["metadata"].exists()
    assert json.loads(paths["metadata"].read_text()) == {
        "countries": ["A"],
        "replicates": 100,
        "seed": 7,
        "maxiter": 30,
    }
    assert not paths["draws"].exists()
    assert not paths["fits"].exists()
    assert (archive_dir / paths["metadata"].name).read_bytes() == old_metadata
    assert (archive_dir / paths["draws"].name).read_bytes() == old_draws
    assert (archive_dir / paths["fits"].name).read_bytes() == old_fits
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    assert "design fields do not match" in manifest["reason"]
    assert len(manifest["files"]) == 3
    draws, fits = bootstrap._load_checkpoint()
    assert draws.empty and fits.empty


def test_matching_checkpoint_is_reused_without_archive(tmp_path, monkeypatch) -> None:
    paths = _use_temporary_checkpoint_bundle(tmp_path, monkeypatch)
    kwargs = {
        "countries": ["A"],
        "replicates": 100,
        "seed": 7,
        "maxiter": 30,
    }

    assert bootstrap._validate_or_create_checkpoint_metadata(**kwargs) is None
    paths["draws"].write_bytes(b"draws")
    paths["fits"].write_bytes(b"fits")
    assert bootstrap._validate_or_create_checkpoint_metadata(**kwargs) is None
    assert not paths["archive"].exists()


@pytest.mark.parametrize("case", ["orphan", "corrupt", "temporary"])
def test_incomplete_checkpoint_states_are_archived(
    tmp_path,
    monkeypatch,
    case: str,
) -> None:
    paths = _use_temporary_checkpoint_bundle(tmp_path, monkeypatch)
    if case == "orphan":
        paths["draws"].write_bytes(b"orphan draws")
    elif case == "corrupt":
        paths["metadata"].write_text("{not-json", encoding="utf-8")
        paths["fits"].write_bytes(b"orphan fits")
    else:
        paths["metadata"].write_text(
            json.dumps(
                {"countries": ["A"], "replicates": 100, "seed": 7, "maxiter": 30}
            ),
            encoding="utf-8",
        )
        temporary = paths["draws"].with_name(paths["draws"].name + ".tmp.parquet")
        temporary.write_bytes(b"interrupted temporary")

    archive_dir = bootstrap._validate_or_create_checkpoint_metadata(
        countries=["A"],
        replicates=100,
        seed=7,
        maxiter=30,
    )

    assert archive_dir is not None
    assert (archive_dir / "manifest.json").exists()
    assert json.loads(paths["metadata"].read_text()) == {
        "countries": ["A"],
        "replicates": 100,
        "seed": 7,
        "maxiter": 30,
    }


def test_checkpoint_archive_move_failure_rolls_back_original_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _use_temporary_checkpoint_bundle(tmp_path, monkeypatch)
    originals = {
        paths["metadata"]: json.dumps(
            {"countries": ["A"], "replicates": 100, "seed": 6, "maxiter": 30}
        ).encode(),
        paths["draws"]: b"draws",
        paths["fits"]: b"fits",
    }
    for path, content in originals.items():
        path.write_bytes(content)

    original_replace = Path.replace
    source_paths = set(originals)
    calls = 0

    def fail_on_second_source_move(self: Path, target: Path) -> Path:
        nonlocal calls
        if self in source_paths:
            calls += 1
            if calls == 2:
                raise OSError("simulated move failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_on_second_source_move)
    with pytest.raises(RuntimeError, match="Could not archive"):
        bootstrap._validate_or_create_checkpoint_metadata(
            countries=["A"],
            replicates=100,
            seed=7,
            maxiter=30,
        )

    assert all(path.read_bytes() == content for path, content in originals.items())


def test_figure2c_input_validation_requires_calibrations_and_frontier(
    tmp_path,
    monkeypatch,
) -> None:
    frontier = tmp_path / "decision_frontier.csv"
    frontier.write_text("country,strategy\nA,timeliness_only\n", encoding="utf-8")
    validation_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(bootstrap, "FRONTIER_PATH", frontier)
    monkeypatch.setattr(
        bootstrap,
        "validate_calibration_artifacts",
        lambda countries, **_kwargs: validation_calls.append(tuple(countries)),
    )

    bootstrap._validate_figure2c_inputs(["A"])
    assert validation_calls == [("A",)]

    frontier.unlink()
    with pytest.raises(FileNotFoundError):
        bootstrap._validate_figure2c_inputs(["A"])


def test_bounded_ar1_path_is_reproducible_and_respects_support() -> None:
    years = np.arange(2018, 2026)
    first = bootstrap._draw_bounded_ar1_path(
        np.random.default_rng(101),
        years,
        rho=0.5,
        innovation_sd=0.6,
        bound=1.25,
    )
    second = bootstrap._draw_bounded_ar1_path(
        np.random.default_rng(101),
        years,
        rho=0.5,
        innovation_sd=0.6,
        bound=1.25,
    )

    assert np.array_equal(first, second)
    assert len(first) == len(years)
    assert np.max(np.abs(first)) <= 1.25


def test_bootstrap_retry_budget_exceeds_production_floor() -> None:
    process = {"max_nfev": 256}

    assert bootstrap._bootstrap_optimizer_max_nfev(process, maxiter=12) == 256
    assert bootstrap._bootstrap_optimizer_max_nfev(process, maxiter=32) == 256
    assert bootstrap._bootstrap_optimizer_max_nfev(process, maxiter=64) == 512
    assert bootstrap._bootstrap_optimizer_max_nfev(process, maxiter=128) == 1024


def test_synthetic_observation_allocation_preserves_likelihood_group_totals() -> None:
    observed = pd.DataFrame(
        {
            "observed_year": [2020, 2020, 2021],
            "period_start": pd.to_datetime(
                ["2020-01-01", "2020-07-01", "2021-01-01"]
            ),
            "period_end": pd.to_datetime(
                ["2020-07-01", "2021-01-01", "2022-01-01"]
            ),
            "interval_days": [182.0, 184.0, 365.0],
            "reported_cases": [10, 20, 30],
            "likelihood_group_id": ["2020", "2020", "2021"],
            "reporting_frequency": ["semiannual", "semiannual", "annual"],
        }
    )

    synthetic, generated = bootstrap._synthetic_observed_frame(
        observed,
        np.array([50.0, 80.0]),
        dispersion=25.0,
        rng=np.random.default_rng(42),
    )
    regrouped = bootstrap.grouped_likelihood_observations(synthetic)

    assert synthetic["reported_cases"].ge(0).all()
    assert np.array_equal(regrouped["reported_cases"].to_numpy(dtype=int), generated)


def test_merge_results_replaces_failed_attempt_and_its_stale_draws() -> None:
    old_fits = pd.DataFrame(
        [
            {
                "country": "A",
                "bootstrap_replicate": 1,
                "success": False,
                "attempt": 1,
            }
        ]
    )
    old_draws = pd.DataFrame(
        [
            {
                "country": "A",
                "bootstrap_replicate": 1,
                "strategy": "stale",
            }
        ]
    )
    replacement = bootstrap.BootstrapResult(
        diagnostic={
            "country": "A",
            "bootstrap_replicate": 1,
            "success": True,
            "attempt": 2,
        },
        draws=[
            {
                "country": "A",
                "bootstrap_replicate": 1,
                "strategy": "timeliness_only",
            }
        ],
    )

    draws, fits = bootstrap._merge_results(old_draws, old_fits, [replacement])

    assert len(fits) == 1
    assert bool(fits.iloc[0]["success"])
    assert int(fits.iloc[0]["attempt"]) == 2
    assert draws["strategy"].tolist() == ["timeliness_only"]


def test_interval_summary_is_percentile_bootstrap_ci(monkeypatch) -> None:
    strategies = list(bootstrap.INTERVENTION_STRATEGIES)
    point_estimates = pd.DataFrame(
        {
            "country": "A",
            "strategy": strategies,
            "reduction_point_estimate": np.linspace(0.10, 0.35, len(strategies)),
        }
    )
    monkeypatch.setattr(
        bootstrap,
        "_frontier_point_estimates",
        lambda _countries: point_estimates,
    )
    rows = []
    for strategy_index, strategy in enumerate(strategies):
        for replicate in range(1, 101):
            reduction = 0.05 * strategy_index + replicate / 1000.0
            rows.append(
                {
                    "country": "A",
                    "strategy": strategy,
                    "strategy_label": strategy,
                    "bootstrap_replicate": replicate,
                    bootstrap.PRIMARY_REDUCTION: reduction,
                    "current_rate": 100.0 + replicate,
                    "intervention_rate": (100.0 + replicate) * (1.0 - reduction),
                }
            )

    intervals, stability = bootstrap._summarise_intervals(
        pd.DataFrame(rows), countries=["A"], stability_blocks=5
    )

    assert len(intervals) == len(strategies)
    assert intervals["bootstrap_replicates"].eq(100).all()
    assert intervals["interval_type"].eq(bootstrap.INTERVAL_TYPE).all()
    assert intervals["confidence_interval_method"].eq(
        "percentile_parametric_bootstrap"
    ).all()
    assert intervals["reduction_point_estimate"].notna().all()
    assert (intervals["reduction_q025"] < intervals["reduction_q975"]).all()
    assert np.isfinite(stability["maximum_endpoint_mcse"]).all()
    assert np.isfinite(stability["tail_probability_mcse"]).all()
    assert stability["tail_probability_mcse"].max() < 0.05


def test_bootstrap_audit_accepts_only_complete_ci_route(tmp_path, monkeypatch) -> None:
    strategies = list(bootstrap.INTERVENTION_STRATEGIES)
    fits = pd.DataFrame(
        {
            "country": ["A", "A"],
            "bootstrap_replicate": [1, 2],
            "success": [True, True],
            "attempt": [1, 1],
            "seed": [11, 12],
        }
    )
    draw_rows = []
    for replicate in (1, 2):
        for strategy_index, strategy in enumerate(strategies):
            reduction = 0.1 + strategy_index / 100.0 + replicate / 1000.0
            draw_rows.append(
                {
                    "country": "A",
                    "bootstrap_replicate": replicate,
                    "strategy": strategy,
                    "current_rate": 100.0,
                    "intervention_rate": 100.0 * (1.0 - reduction),
                    bootstrap.PRIMARY_REDUCTION: reduction,
                }
            )
    draws = pd.DataFrame(draw_rows)
    intervals = pd.DataFrame(
        {
            "country": "A",
            "strategy": strategies,
            "reduction_q025": 0.05,
            "reduction_q25": 0.10,
            "reduction_median": 0.15,
            "reduction_q75": 0.20,
            "reduction_q975": 0.25,
            "interval_type": bootstrap.INTERVAL_TYPE,
            "interval_basis": bootstrap.INTERVAL_BASIS,
        }
    )
    stability = pd.DataFrame(
        {
            "country": "A",
            "strategy": strategies,
            "maximum_endpoint_mcse": 0.001,
            "tail_probability_mcse": 0.004,
        }
    )
    paths = tuple(tmp_path / name for name in ("draws.csv", "intervals.csv", "fits.csv", "stability.csv"))
    for path in paths:
        path.write_text(path.stem, encoding="utf-8")
    monkeypatch.setattr(audit, "AUDITED_ARTIFACT_PATHS", paths)
    metadata = {
        "analysis_role": "publication_estimation_confidence_interval",
        "publication_path": True,
        "figure2c_interval_source": True,
        "statistical_target": bootstrap.STATISTICAL_TARGET,
        "confidence_interval_method": bootstrap.CONFIDENCE_INTERVAL_METHOD,
        "bootstrap_data_generation": bootstrap.BOOTSTRAP_DATA_GENERATION,
        "bootstrap_refit": bootstrap.BOOTSTRAP_REFIT,
        "varied_estimation_components": list(bootstrap.VARIED_ESTIMATION_COMPONENTS),
        "fixed_reference_inputs": list(bootstrap.FIXED_REFERENCE_INPUTS),
        "paired_scenario_contrast": True,
        "posterior_credible_interval": False,
        "future_observation_prediction_interval": False,
    }

    rows = audit._audit_frames(
        source_metadata=metadata,
        draws=draws,
        intervals=intervals,
        fits=fits,
        stability=stability,
        countries=["A"],
        requested_replicates=2,
        minimum_successful_replicates=2,
        minimum_success_fraction=1.0,
        maximum_tail_probability_mcse=0.005,
    )
    assert all(row["status"] == "pass" for row in rows)

    mislabeled = intervals.copy()
    mislabeled["interval_type"] = "95% posterior credible interval"
    failed = audit._audit_frames(
        source_metadata=metadata,
        draws=draws,
        intervals=mislabeled,
        fits=fits,
        stability=stability,
        countries=["A"],
        requested_replicates=2,
        minimum_successful_replicates=2,
        minimum_success_fraction=1.0,
        maximum_tail_probability_mcse=0.005,
    )
    statuses = {row["check"]: row["status"] for row in failed}
    assert statuses["interval_type_is_parametric_bootstrap_confidence_interval"] == "fail"
    assert statuses["no_posterior_or_prediction_interval_claim"] == "fail"


def _complete_audit_design_inputs(
    tmp_path: Path,
    monkeypatch,
    *,
    countries: tuple[str, ...] = ("A", "B"),
    requested_replicates: int = 2,
) -> dict[str, object]:
    strategies = list(bootstrap.INTERVENTION_STRATEGIES)
    fits = pd.DataFrame(
        [
            {
                "country": country,
                "bootstrap_replicate": replicate,
                "success": True,
                "attempt": 1,
                "seed": 100 * country_index + replicate,
            }
            for country_index, country in enumerate(countries, start=1)
            for replicate in range(1, requested_replicates + 1)
        ]
    )
    draws = pd.DataFrame(
        [
            {
                "country": country,
                "bootstrap_replicate": replicate,
                "strategy": strategy,
                "current_rate": 100.0,
                "intervention_rate": 100.0 * (1.0 - reduction),
                bootstrap.PRIMARY_REDUCTION: reduction,
            }
            for country in countries
            for replicate in range(1, requested_replicates + 1)
            for strategy_index, strategy in enumerate(strategies)
            for reduction in [
                0.1 + strategy_index / 100.0 + replicate / 1000.0
            ]
        ]
    )
    intervals = pd.DataFrame(
        [
            {
                "country": country,
                "strategy": strategy,
                "reduction_q025": 0.05,
                "reduction_q25": 0.10,
                "reduction_median": 0.15,
                "reduction_q75": 0.20,
                "reduction_q975": 0.25,
                "interval_type": bootstrap.INTERVAL_TYPE,
                "interval_basis": bootstrap.INTERVAL_BASIS,
            }
            for country in countries
            for strategy in strategies
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "country": country,
                "strategy": strategy,
                "maximum_endpoint_mcse": 0.001,
                "tail_probability_mcse": 0.004,
            }
            for country in countries
            for strategy in strategies
        ]
    )
    paths = tuple(tmp_path / name for name in ("draws.csv", "intervals.csv", "fits.csv", "stability.csv"))
    for path in paths:
        path.write_text(path.stem, encoding="utf-8")
    monkeypatch.setattr(audit, "AUDITED_ARTIFACT_PATHS", paths)
    metadata = {
        "analysis_role": "publication_estimation_confidence_interval",
        "publication_path": True,
        "figure2c_interval_source": True,
        "statistical_target": bootstrap.STATISTICAL_TARGET,
        "confidence_interval_method": bootstrap.CONFIDENCE_INTERVAL_METHOD,
        "bootstrap_data_generation": bootstrap.BOOTSTRAP_DATA_GENERATION,
        "bootstrap_refit": bootstrap.BOOTSTRAP_REFIT,
        "varied_estimation_components": list(
            bootstrap.VARIED_ESTIMATION_COMPONENTS
        ),
        "fixed_reference_inputs": list(bootstrap.FIXED_REFERENCE_INPUTS),
        "paired_scenario_contrast": True,
        "posterior_credible_interval": False,
        "future_observation_prediction_interval": False,
    }
    return {
        "source_metadata": metadata,
        "draws": draws,
        "intervals": intervals,
        "fits": fits,
        "stability": stability,
        "countries": list(countries),
        "requested_replicates": requested_replicates,
        "minimum_successful_replicates": requested_replicates,
        "minimum_success_fraction": 1.0,
        "maximum_tail_probability_mcse": 0.005,
    }


def test_bootstrap_audit_rejects_noncanonical_country_replicate_grid(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = _complete_audit_design_inputs(tmp_path, monkeypatch)
    fits = inputs["fits"].copy()
    fits.loc[
        fits["country"].eq("B") & fits["bootstrap_replicate"].eq(2),
        "bootstrap_replicate",
    ] = 3
    inputs["fits"] = fits

    rows = audit._audit_frames(**inputs)
    statuses = {row["check"]: row["status"] for row in rows}

    assert statuses["fit_diagnostics_complete"] == "fail"
    assert statuses["fit_country_replicate_grid_exact"] == "fail"


def test_bootstrap_audit_rejects_draws_for_wrong_successful_replicate(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = _complete_audit_design_inputs(tmp_path, monkeypatch)
    draws = inputs["draws"].copy()
    draws.loc[
        draws["country"].eq("B") & draws["bootstrap_replicate"].eq(2),
        "bootstrap_replicate",
    ] = 3
    inputs["draws"] = draws

    rows = audit._audit_frames(**inputs)
    statuses = {row["check"]: row["status"] for row in rows}

    assert statuses["successful_replicate_strategy_grid_exact"] == "fail"
    assert statuses["paired_current_intervention"] == "fail"


def test_bootstrap_audit_rejects_incomplete_six_strategy_block(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = _complete_audit_design_inputs(tmp_path, monkeypatch)
    draws = inputs["draws"].copy()
    inputs["draws"] = draws.drop(index=draws.index[0]).reset_index(drop=True)

    rows = audit._audit_frames(**inputs)
    statuses = {row["check"]: row["status"] for row in rows}

    assert statuses["successful_replicate_strategy_grid_exact"] == "fail"
