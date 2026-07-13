from __future__ import annotations

from src_python.simulation.common import _scenario_execution_n_jobs
from src_python.utils import parallel


def test_explicit_nested_worker_allocation_overrides_environment_default(monkeypatch) -> None:
    monkeypatch.setenv("PERTUSSIS_N_JOBS", "100")
    monkeypatch.setattr(parallel, "available_cpus", lambda: 128)

    assert parallel.resolve_n_jobs(20) == 20
    assert parallel.resolve_n_jobs(None) == 100


def test_worker_allocation_is_capped_by_available_cpus(monkeypatch) -> None:
    monkeypatch.delenv("PERTUSSIS_N_JOBS", raising=False)
    monkeypatch.setattr(parallel, "available_cpus", lambda: 12)

    assert parallel.resolve_n_jobs(100) == 12


def test_top_level_environment_budget_precedes_scenario_default(monkeypatch) -> None:
    monkeypatch.setenv("PERTUSSIS_N_JOBS", "64")
    scenarios = [{"config": {"simulation": {"n_jobs": -1}}}]

    # None is deliberately forwarded so parallel.resolve_n_jobs applies the
    # environment cap instead of treating the YAML -1 as an explicit request.
    assert _scenario_execution_n_jobs(scenarios, None) is None
    monkeypatch.setattr(parallel, "available_cpus", lambda: 128)
    assert parallel.resolve_n_jobs(
        _scenario_execution_n_jobs(scenarios, None)
    ) == 64


def test_explicit_scenario_budget_precedes_environment(monkeypatch) -> None:
    monkeypatch.setenv("PERTUSSIS_N_JOBS", "64")
    scenarios = [{"config": {"simulation": {"n_jobs": -1}}}]

    assert _scenario_execution_n_jobs(scenarios, 20) == 20


def test_scenario_default_is_used_without_environment_budget(monkeypatch) -> None:
    monkeypatch.delenv("PERTUSSIS_N_JOBS", raising=False)
    scenarios = [{"config": {"simulation": {"n_jobs": -1}}}]

    assert _scenario_execution_n_jobs(scenarios, None) == -1
