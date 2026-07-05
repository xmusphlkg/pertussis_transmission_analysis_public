from __future__ import annotations

from src_python.simulation import run_all


def test_run_all_publication_diagnostics_match_makefile_simulation_layer(monkeypatch) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def recorder(name: str):
        def inner(*args: object, **kwargs: object) -> None:
            calls.append((name, args, kwargs))

        return inner

    for name in (
        "run_baseline",
        "run_vaccines",
        "run_resistance",
        "run_reporting",
        "run_countries",
        "run_heatmap",
        "run_fitness_grid",
        "run_interventions",
        "run_routine_timeliness",
        "run_sensitivity",
        "run_immunity",
        "run_resistance_fitness",
        "run_bayesian_uncertainty",
        "run_hindcast",
        "run_calibration_diagnostics",
        "run_age_pattern_sensitivity",
        "run_resistance_mechanism",
        "run_program_portfolio_factorial",
        "run_infant_contact_sensitivity",
        "run_maternal_duration_sensitivity",
        "run_shock_recovery_sensitivity",
        "run_temporal_assumption_sensitivity",
        "run_treatment_implementation_sensitivity",
        "run_individual_stochastic_toy",
        "run_joint_psa",
        "write_manuscript_tables",
    ):
        monkeypatch.setattr(run_all, name, recorder(name))

    monkeypatch.setattr(run_all, "load_configs", lambda: {"countries": {"A": {}, "B": {}}})
    monkeypatch.setattr(run_all, "SELECTED_STRATEGIES", ("current", "timeliness_only"))

    run_all.main(n_jobs=7, include_bayesian=True, include_publication_diagnostics=True)

    names = [name for name, _, _ in calls]
    publication_start = names.index("run_hindcast")
    assert names.index("run_bayesian_uncertainty") < publication_start
    assert names[publication_start:] == [
        "run_hindcast",
        "run_calibration_diagnostics",
        "run_age_pattern_sensitivity",
        "run_resistance_mechanism",
        "run_program_portfolio_factorial",
        "run_infant_contact_sensitivity",
        "run_maternal_duration_sensitivity",
        "run_shock_recovery_sensitivity",
        "run_temporal_assumption_sensitivity",
        "run_treatment_implementation_sensitivity",
        "run_individual_stochastic_toy",
        "run_joint_psa",
        "run_fitness_grid",
        "write_manuscript_tables",
    ]

    deterministic_grid_call = [
        kwargs
        for name, _, kwargs in calls
        if name == "run_fitness_grid" and kwargs.get("posterior_draws") == 0
    ][0]
    assert deterministic_grid_call == {"n_jobs": 7, "posterior_draws": 0}

    publication_grid_call = [
        kwargs
        for name, _, kwargs in calls
        if name == "run_fitness_grid" and kwargs.get("run_deterministic_grid") is False
    ][0]
    assert publication_grid_call == {
        "n_jobs": 7,
        "posterior_draws": 100,
        "posterior_batch_size": 32,
        "psa_benefit_samples": -1,
        "psa_batch_size": 1,
        "run_deterministic_grid": False,
    }

    joint_psa_call = [kwargs for name, _, kwargs in calls if name == "run_joint_psa"][0]
    assert joint_psa_call == {
        "sample_size": 128,
        "seed": 20260521,
        "countries": ("A", "B"),
        "strategies": ("current", "timeliness_only"),
        "n_jobs": 7,
        "sample_batch_size": 8,
        "resume": True,
        "smoke_runtime": False,
        "keep_timeseries": False,
    }
