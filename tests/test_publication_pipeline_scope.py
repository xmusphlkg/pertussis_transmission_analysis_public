from __future__ import annotations

from pathlib import Path

import pytest

from src_python.simulation import run_bayesian_uncertainty as uncertainty
from src_python.simulation.common import publication_country_names


def _touch(path: Path, value: str = "stale") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_publication_country_names_applies_prespecified_exclusions() -> None:
    configs = {
        "countries": {"Australia": {}, "South_Africa": {}, "Japan": {}},
        "baseline": {
            "bayesian_uncertainty": {
                "publication_country_exclusions": {"South_Africa": "too few intervals"}
            }
        },
    }

    assert publication_country_names(configs) == ["Australia", "Japan"]


def test_stem_cleanup_removes_only_that_stem_and_excluded_country_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        uncertainty,
        "project_path",
        lambda *parts: tmp_path.joinpath(*parts),
    )
    selected_grid = _touch(
        tmp_path / "outputs/metadata/beta_grid_pilot/Australia_grid.csv"
    )
    selected_grid_parquet = _touch(
        tmp_path / "outputs/metadata/beta_grid_pilot/Australia_grid.parquet"
    )
    excluded_grid = _touch(
        tmp_path / "outputs/metadata/beta_grid_pilot/South_Africa_grid.csv"
    )
    progress = _touch(
        tmp_path / "outputs/metadata/mcmc_progress_pilot/South_Africa_chain01.txt"
    )
    stale_stage = _touch(
        tmp_path
        / "outputs/metadata/pilot_smc_stage_history_South_Africa_chain01.csv"
    )
    stale_posterior = _touch(
        tmp_path / "outputs/simulations/pilot_posterior_samples.parquet"
    )
    stale_posterior_csv = _touch(
        tmp_path / "outputs/simulations/pilot_posterior_samples.csv"
    )
    other_progress = _touch(
        tmp_path / "outputs/metadata/mcmc_progress_other/Australia_chain01.txt"
    )
    other_stage = _touch(
        tmp_path / "outputs/metadata/other_smc_stage_history_Australia_chain01.csv"
    )
    other_posterior = _touch(
        tmp_path / "outputs/simulations/other_posterior_samples.parquet"
    )

    uncertainty._clear_previous_sampler_artifacts(
        ["Australia"],
        sampler="beta_grid",
        output_stem="pilot",
        reuse_valid_beta_grid=True,
    )

    assert selected_grid.exists()
    assert selected_grid_parquet.exists()
    assert not excluded_grid.exists()
    assert not progress.exists()
    assert not stale_stage.exists()
    assert not stale_posterior.exists()
    assert not stale_posterior_csv.exists()
    assert other_progress.exists()
    assert other_stage.exists()
    assert other_posterior.exists()


def test_stem_cleanup_rejects_path_components(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single safe path component"):
        uncertainty._clear_previous_sampler_artifacts(
            ["Australia"],
            sampler="state_space_exact_importance_cut",
            output_stem=str(tmp_path / "unsafe"),
        )


def test_retired_full_joint_output_stem_is_rejected_before_a_run() -> None:
    with pytest.raises(ValueError, match="mislabeled a legacy research route"):
        uncertainty.main(output_stem="bayesian_uncertainty_full_joint")


@pytest.mark.parametrize(
    "retired_stem",
    [
        "bayesian_uncertainty_figure2c_conditional",
        "bayesian_uncertainty_figure2c_joint",
    ],
)
def test_retired_figure2c_bayesian_stems_are_rejected_before_a_run(
    retired_stem: str,
) -> None:
    with pytest.raises(ValueError, match="mislabeled a legacy research route"):
        uncertainty.main(output_stem=retired_stem)
