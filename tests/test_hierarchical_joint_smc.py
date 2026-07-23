from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

import src_python.simulation.run_hierarchical_joint_smc as smc


def test_reflection_stays_in_box_and_is_periodic() -> None:
    values = np.asarray([[-3.0, 7.0], [0.25, 0.75], [5.0, -4.0]])
    reflected = smc._reflect_into_box(
        values,
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
    )
    assert np.logical_and(reflected >= 0.0, reflected <= 1.0).all()
    assert np.allclose(reflected[1], values[1])
    assert np.allclose(reflected[0], [1.0, 1.0])


def test_gaussian_kernel_matches_quadratic_form() -> None:
    values = np.asarray([[1.0, 2.0], [2.0, 0.0]])
    result = smc._log_gaussian_kernel(values, np.zeros(2), np.diag([2.0, 0.5]))
    expected = np.asarray([-0.5 * (2.0 + 2.0), -0.5 * 8.0])
    assert np.allclose(result, expected)


def test_regularized_covariance_retains_global_geometry() -> None:
    local = np.column_stack((np.linspace(-0.01, 0.01, 64), np.zeros(64)))
    global_covariance = np.diag([4.0, 9.0])
    result = smc._regularized_covariance(
        local,
        global_covariance,
        floor_fraction=0.25,
    )
    assert np.all(np.linalg.eigvalsh(result) > 0.0)
    assert result[0, 0] >= 1.0
    assert result[1, 1] >= 2.25


def test_tempered_guided_geometry_connects_prior_and_laplace() -> None:
    prior = smc.PriorGeometry(np.zeros(2), np.eye(2), np.eye(2))
    laplace = smc.PriorGeometry(
        np.asarray([2.0, -2.0]),
        np.diag([4.0, 9.0]),
        np.diag([0.25, 1.0 / 9.0]),
    )
    at_prior = smc._tempered_guided_geometry(prior, laplace, 0.0)
    at_laplace = smc._tempered_guided_geometry(prior, laplace, 1.0)
    halfway = smc._tempered_guided_geometry(prior, laplace, 0.25)
    assert np.allclose(at_prior.mean, prior.mean)
    assert np.allclose(at_prior.covariance, prior.covariance)
    assert np.allclose(at_laplace.mean, laplace.mean)
    assert np.allclose(at_laplace.covariance, laplace.covariance)
    assert 0.0 < np.linalg.norm(halfway.mean) < np.linalg.norm(laplace.mean)
    assert smc._guided_geometry_weight(0.0) == 0.0
    assert smc._guided_geometry_weight(1.0) == 1.0
    assert 0.0 < smc._guided_geometry_weight(0.25) < 1.0


def test_short_mode_bank_schedule_rotates_every_secondary_kernel() -> None:
    cycles = [
        smc._stage_move_cycle(stage, mode_bank_active=True)[:3]
        for stage in range(1, 4)
    ]
    assert all(cycle[0] == "mode_bank" for cycle in cycles)
    flattened = [kind for cycle in cycles for kind in cycle[1:]]
    assert set(flattened) == {"local", "guided", "joint"}
    fractions = smc._mean_move_fractions(3, mode_bank_active=True)
    assert np.isclose(fractions["mode_bank"], 1.0 / 3.0)
    assert np.isclose(sum(fractions.values()), 1.0)


def test_short_pre_activation_schedule_rotates_every_kernel() -> None:
    cycles = [
        smc._stage_move_cycle(stage, mode_bank_active=False)[:2]
        for stage in range(1, 4)
    ]
    flattened = [kind for cycle in cycles for kind in cycle]
    assert set(flattened) == {"local", "guided", "joint"}
    assert all(flattened.count(kind) == 2 for kind in set(flattened))
    fractions = smc._mean_move_fractions(2, mode_bank_active=False)
    assert all(
        np.isclose(fractions[kind], 1.0 / 3.0)
        for kind in ("local", "guided", "joint")
    )


def test_gaussian_mixture_density_is_symmetric_for_symmetric_components() -> None:
    values = np.asarray([[-1.0], [1.0], [0.0]])
    density = smc._log_gaussian_mixture_density(
        values,
        [np.asarray([-1.0]), np.asarray([1.0])],
        [np.eye(1), np.eye(1)],
    )
    assert np.isfinite(density).all()
    assert np.isclose(density[0], density[1])


def test_bounded_state_transform_round_trip_and_jacobian() -> None:
    lower = np.asarray([-3.0, 0.5, -5.0])
    upper = np.asarray([2.0, 4.5, 5.0])
    values = np.asarray(
        [
            [-2.5, 1.0, -4.0],
            [0.0, 2.0, 0.0],
            [1.5, 4.0, 4.0],
        ]
    )
    transformed, forward_jacobian = smc._state_to_unbounded(
        values, lower, upper
    )
    recovered, inverse_jacobian = smc._state_from_unbounded(
        transformed, lower, upper
    )
    assert np.allclose(recovered, values)
    assert np.allclose(forward_jacobian, inverse_jacobian)


def test_process_support_widens_only_latent_coordinates() -> None:
    @dataclass(frozen=True)
    class Context:
        lower: np.ndarray
        upper: np.ndarray

    context = Context(
        np.asarray([-4.0, -3.0, -2.5, -2.5]),
        np.asarray([4.0, 3.0, 2.5, 2.5]),
    )
    widened = smc._with_process_support(context, 5.0)
    assert np.allclose(widened.lower[:2], context.lower[:2])
    assert np.allclose(widened.upper[:2], context.upper[:2])
    assert np.allclose(widened.lower[2:], -5.0)
    assert np.allclose(widened.upper[2:], 5.0)


def test_student_t_mode_bank_density_is_symmetric_and_draws_are_finite() -> None:
    covariance = np.diag([0.5, 2.0])
    mode_bank = smc.ModeBank(
        means=(np.asarray([-1.0, 0.0]), np.asarray([1.0, 0.0])),
        covariances=(covariance, covariance),
        precisions=(np.linalg.inv(covariance), np.linalg.inv(covariance)),
        log_determinants=(
            float(np.linalg.slogdet(covariance)[1]),
            float(np.linalg.slogdet(covariance)[1]),
        ),
        cholesky_factors=(
            np.linalg.cholesky(covariance),
            np.linalg.cholesky(covariance),
        ),
        degrees_of_freedom=7.0,
        source_stems=("left", "right"),
        dimension=2,
    )
    values = np.asarray([[-1.0, 0.5], [1.0, 0.5], [0.0, 0.0]])
    density = smc._mode_bank_log_density(values, mode_bank)
    draws = smc._draw_mode_bank(mode_bank, 128, np.random.default_rng(91))
    assert np.isfinite(density).all()
    assert np.isclose(density[0], density[1])
    assert draws.shape == (128, 2)
    assert np.isfinite(draws).all()


def test_mode_bank_move_updates_complete_state_with_exact_mh(monkeypatch) -> None:
    count = 32
    structural = np.zeros((count, 9), dtype=float)
    structural[:, smc.IMPORTANCE_NUISANCE_INDICES] = np.linspace(
        -0.2, 0.2, count
    )[:, None]
    state = np.column_stack(
        (np.linspace(-0.4, 0.4, count), np.linspace(0.4, -0.4, count))
    )
    context = SimpleNamespace(
        country="Testland",
        lower=np.full(2, -2.0),
        upper=np.full(2, 2.0),
        coordinate_names=("log_beta", "log_reporting"),
        prior_base={},
        priors={},
    )
    geometry = smc.PriorGeometry(np.zeros(2), np.eye(2), np.zeros((2, 2)))
    cloud = smc.ParticleCloud(
        structural=structural.copy(),
        state_by_country={"Testland": state.copy()},
        structural_log_prior=np.zeros(count),
        state_log_prior_by_country={"Testland": np.zeros(count)},
        potential_by_country={"Testland": np.zeros(count)},
        weights=np.full(count, 1.0 / count),
        island=np.ones(count, dtype=int),
        ancestor=np.arange(count),
    )
    coordinates, _ = smc._cloud_mode_coordinates(cloud, [context])
    covariance = np.eye(coordinates.shape[1]) * 0.02
    mode_bank = smc.ModeBank(
        means=(np.mean(coordinates, axis=0),),
        covariances=(covariance,),
        precisions=(np.linalg.inv(covariance),),
        log_determinants=(float(np.linalg.slogdet(covariance)[1]),),
        cholesky_factors=(np.linalg.cholesky(covariance),),
        degrees_of_freedom=7.0,
        source_stems=("pilot",),
        dimension=coordinates.shape[1],
    )
    monkeypatch.setattr(
        smc,
        "_log_nuisance_prior_transformed",
        lambda vector, prior_base, priors: 0.0,
    )

    def flat_potential(context, structural, states, geometry, *, n_jobs, active=None):
        output = np.full(len(structural), -np.inf)
        selected = np.ones(len(structural), dtype=bool) if active is None else active
        output[np.asarray(selected, dtype=bool)] = 0.0
        return output

    monkeypatch.setattr(smc, "_evaluate_country_potential", flat_potential)
    acceptance = smc._joint_mode_bank_independence_move(
        cloud,
        [context],
        {"Testland": geometry},
        mode_bank,
        temperature=1.0,
        n_jobs=1,
        rng=np.random.default_rng(101),
    )
    assert 0.0 < acceptance <= 1.0
    assert not np.allclose(cloud.structural, structural)
    assert not np.allclose(cloud.state_by_country["Testland"], state)
    assert np.logical_and(
        cloud.state_by_country["Testland"] > context.lower,
        cloud.state_by_country["Testland"] < context.upper,
    ).all()


def test_mode_bank_dispatch_can_keep_independent_pilot_bank_fixed(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fixed(*args, **kwargs):
        calls.append("fixed")
        return 0.25

    def crossfit(*args, **kwargs):
        calls.append("crossfit")
        return 0.50

    monkeypatch.setattr(smc, "_joint_mode_bank_independence_move", fixed)
    monkeypatch.setattr(smc, "_joint_crossfit_mode_bank_move", crossfit)
    common = {
        "cloud": object(),
        "contexts": [],
        "geometries": {},
        "mode_bank": object(),
        "temperature": 1.0,
        "n_jobs": 1,
        "rng": np.random.default_rng(12),
    }

    assert smc._joint_mode_bank_move(
        **common, crossfit_current_population=False
    ) == 0.25
    assert smc._joint_mode_bank_move(
        **common, crossfit_current_population=True
    ) == 0.50
    assert calls == ["fixed", "crossfit"]


def test_joint_differential_move_updates_shared_and_state_blocks(monkeypatch) -> None:
    count = 8
    structural = np.zeros((count, 9), dtype=float)
    structural[:, 2] = np.linspace(-1.0, 1.0, count)
    state = np.column_stack(
        (-structural[:, 2], np.linspace(-0.5, 0.5, count))
    )
    context = SimpleNamespace(
        country="Testland",
        lower=np.full(2, -10.0),
        upper=np.full(2, 10.0),
        prior_base={},
        priors={},
    )
    geometry = smc.PriorGeometry(np.zeros(2), np.eye(2), np.zeros((2, 2)))
    cloud = smc.ParticleCloud(
        structural=structural.copy(),
        state_by_country={"Testland": state.copy()},
        structural_log_prior=np.zeros(count),
        state_log_prior_by_country={"Testland": np.zeros(count)},
        potential_by_country={"Testland": np.zeros(count)},
        weights=np.full(count, 1.0 / count),
        island=np.ones(count, dtype=int),
        ancestor=np.arange(count),
    )

    monkeypatch.setattr(
        smc,
        "_log_nuisance_prior_transformed",
        lambda vector, prior_base, priors: 0.0,
    )

    def flat_potential(context, structural, states, geometry, *, n_jobs, active=None):
        output = np.full(len(structural), -np.inf)
        selected = np.ones(len(structural), dtype=bool) if active is None else active
        output[np.asarray(selected, dtype=bool)] = 0.0
        return output

    monkeypatch.setattr(smc, "_evaluate_country_potential", flat_potential)
    acceptance = smc._joint_differential_evolution_move(
        cloud,
        [context],
        {"Testland": geometry},
        temperature=1.0,
        scale=1.0,
        n_jobs=1,
        rng=np.random.default_rng(7),
    )

    assert acceptance.overall == 1.0
    assert acceptance.local == 1.0
    assert acceptance.global_jump == 1.0
    assert not np.allclose(cloud.structural[:, 2], structural[:, 2])
    assert not np.allclose(cloud.state_by_country["Testland"], state)
    assert np.isfinite(cloud.potential_by_country["Testland"]).all()
    assert np.logical_and(
        cloud.state_by_country["Testland"] >= context.lower,
        cloud.state_by_country["Testland"] <= context.upper,
    ).all()


def test_adaptive_temperature_preserves_each_island_ess_floor() -> None:
    island = np.repeat([1, 2], 64)
    weights = np.tile(np.full(64, 1.0 / 64.0), 2)
    potential = np.concatenate(
        (np.linspace(-30.0, 0.0, 64), np.linspace(-20.0, 0.0, 64))
    )
    temperature, updated, ess = smc._find_next_temperature(
        weights,
        potential,
        island,
        0.0,
        0.70,
    )
    assert 0.0 < temperature < 1.0
    assert np.all(ess >= 0.70 * 64 - 1e-6)
    for positions in smc._island_indices(island):
        assert np.isclose(updated[positions].sum(), 1.0)


def test_full_temperature_is_taken_when_weights_are_stable() -> None:
    island = np.repeat([1, 2], 32)
    weights = np.tile(np.full(32, 1.0 / 32.0), 2)
    potential = np.zeros(64)
    temperature, updated, ess = smc._find_next_temperature(
        weights,
        potential,
        island,
        0.2,
        0.70,
    )
    assert temperature == 1.0
    assert np.allclose(ess, 32.0)
    assert np.allclose(updated, weights)
