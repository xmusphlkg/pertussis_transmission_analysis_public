from __future__ import annotations

import numpy as np
import pandas as pd

import src_python.simulation.run_hierarchical_joint_posterior as joint
import src_python.simulation.run_hierarchical_joint_smc as smc


def test_bayesian_joint_runners_are_explicitly_nonpublication_research() -> None:
    assert joint.ANALYSIS_ROLE == "optional_nonpublication_legacy_research"
    assert joint.PUBLICATION_PATH is False
    assert joint.FIGURE2C_INTERVAL_SOURCE is False
    assert joint.DEFAULT_STEM == "bayesian_uncertainty_joint_research"
    for module in (joint, smc):
        description = " ".join(module.__doc__.split())
        assert "not a Figure 2c interval source" in description
        assert "not part of the publication pipeline" in description


def test_hierarchical_runners_only_consume_shared_registry_priors() -> None:
    assert joint.REGISTRY_PRIOR_PARAMETER_NAMES == joint.SHARED_PARAMETER_NAMES
    assert (
        joint.CALIBRATION_STATE_PRIOR_PARAMETER_NAMES
        == joint.LOCAL_PARAMETER_NAMES
    )
    assert not set(joint.REGISTRY_PRIOR_PARAMETER_NAMES) & set(
        joint.CALIBRATION_STATE_PRIOR_PARAMETER_NAMES
    )
    assert "beta_S" in joint.CALIBRATION_STATE_PRIOR_PARAMETER_NAMES
    assert "reporting_multiplier" in joint.CALIBRATION_STATE_PRIOR_PARAMETER_NAMES
    assert smc.REGISTRY_PRIOR_PARAMETER_NAMES == joint.REGISTRY_PRIOR_PARAMETER_NAMES
    assert (
        smc.CALIBRATION_STATE_PRIOR_PARAMETER_NAMES
        == joint.CALIBRATION_STATE_PRIOR_PARAMETER_NAMES
    )


def test_recommended_run_rejects_impossible_global_tail_support() -> None:
    with np.testing.assert_raises_regex(ValueError, "mathematically impossible"):
        joint.run_hierarchical_joint_posterior(
            adaptation_draws=32,
            refinement_draws=32,
            final_draws=1999,
            state_adaptation_candidates=64,
            outer_adaptation_state_candidates=64,
            state_candidates=64,
            require_recommended=True,
        )


def test_posterior_relevant_indices_reach_requested_mass() -> None:
    weight = np.asarray([0.50, 0.30, 0.15, 0.05])
    selected = joint._posterior_relevant_indices(weight, mass=0.90)
    assert set(selected) == {0, 1, 2}
    assert weight[selected].sum() >= 0.90


def test_local_weight_summary_normalizes_exact_log_weights() -> None:
    candidates = np.column_stack((np.arange(64, dtype=float), np.zeros(64)))
    raw = np.full(64, -20.0)
    raw[:4] = np.log(np.asarray([1.0, 2.0, 3.0, 4.0]))
    ess, maximum, _pareto, _tail, probability = joint._local_weight_summary(
        candidates, raw
    )
    assert np.isclose(probability.sum(), 1.0)
    assert np.isclose(maximum, probability.max())
    assert np.isclose(ess, 1.0 / np.sum(probability**2))


def test_apply_shared_structure_updates_history_without_touching_local_scale() -> None:
    runtime = {
        "vaccine": {"VE_sus": 0.1, "VE_inf": 0.2, "VE_dur": 0.3},
        "transmission": {
            "beta_S": 0.4,
            "relative_infectiousness_asymptomatic": 0.5,
            "fitness_R": 1.0,
        },
        "natural_history": {
            "infectious_duration_symptomatic": 14.0,
            "infectious_duration_asymptomatic": 10.0,
        },
        joint.PROSPECTIVE_POLICY_KEY: {
            "history_config": {
                "vaccine": {"VE_sus": 0.1, "VE_inf": 0.2, "VE_dur": 0.3},
                "transmission": {
                    "beta_S": 0.4,
                    "relative_infectiousness_asymptomatic": 0.5,
                    "fitness_R": 1.0,
                },
                "natural_history": {
                    "infectious_duration_symptomatic": 14.0,
                    "infectious_duration_asymptomatic": 10.0,
                },
            }
        },
    }
    sample = {
        "VE_sus": 0.6,
        "VE_inf": 0.7,
        "VE_dur": 0.8,
        "relative_infectiousness_asymptomatic": 0.4,
        "infectious_duration_symptomatic": 12.0,
        "infectious_duration_asymptomatic": 9.0,
        "fitness_R": 1.1,
    }
    updated = joint._apply_shared_structure(runtime, sample)
    assert updated["transmission"]["beta_S"] == 0.4
    assert updated["vaccine"]["VE_sus"] == 0.6
    history = updated[joint.PROSPECTIVE_POLICY_KEY]["history_config"]
    assert history["vaccine"]["VE_inf"] == 0.7
    assert history["transmission"]["fitness_R"] == 1.1
    assert runtime["vaccine"]["VE_sus"] == 0.1


def test_stage_checkpoint_reuses_completed_chunk(monkeypatch, tmp_path) -> None:
    context = joint.CountryStateContext(
        country="Testland",
        runtime_reference={},
        prior_base={},
        observed=pd.DataFrame(),
        priors={},
        structural_start=np.zeros(9),
        coordinate_names=("log_beta_S", "log_reporting_multiplier"),
        state_mean=np.zeros(2),
        state_covariance=np.eye(2),
        lower=np.full(2, -10.0),
        upper=np.full(2, 10.0),
        process_rho=0.5,
        process_innovation_sd=0.6,
        beta_prior_sd=1.0,
        reporting_prior_sd=0.5,
        dispersion=50.0,
        historical_end_year=2024,
        forecast_end_year=2024,
    )

    def fake_inner(_context, _vector, **_kwargs):
        candidates = np.zeros((64, 2), dtype=float)
        log_density = np.zeros(64, dtype=float)
        raw = np.zeros(64, dtype=float)
        diagnostic = {
            "country": "Testland",
            "log_marginal_likelihood": 0.0,
            "state_effective_sample_size": 64.0,
            "state_maximum_weight": 1.0 / 64.0,
            "state_pareto_k": 0.0,
            "state_minimum_tail_ess": 2.0,
            "state_in_bounds_fraction": 1.0,
        }
        return 0.0, candidates, log_density, raw, diagnostic

    monkeypatch.setattr(joint, "_conditional_state_importance", fake_inner)
    vectors = np.zeros((2, 9), dtype=float)
    first = joint._evaluate_stage(
        [context],
        vectors,
        np.zeros(2),
        np.zeros(2),
        state_adaptation_candidates=64,
        state_candidates=64,
        state_covariance_scales=(1.0,),
        optimizer_maxiter=1,
        adaptive_rounds=0,
        localized_rounds=0,
        n_jobs=1,
        seed=7,
        retain_state_candidates=True,
        checkpoint_dir=tmp_path,
    )
    assert np.allclose(first.global_weight, 0.5)
    assert len(list(tmp_path.glob("*.joblib"))) == 2

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("completed checkpoint was not reused")

    monkeypatch.setattr(joint, "_conditional_state_importance", should_not_run)
    second = joint._evaluate_stage(
        [context],
        vectors,
        np.zeros(2),
        np.zeros(2),
        state_adaptation_candidates=64,
        state_candidates=64,
        state_covariance_scales=(1.0,),
        optimizer_maxiter=1,
        adaptive_rounds=0,
        localized_rounds=0,
        n_jobs=1,
        seed=7,
        retain_state_candidates=True,
        checkpoint_dir=tmp_path,
    )
    assert np.allclose(second.global_weight, first.global_weight)
