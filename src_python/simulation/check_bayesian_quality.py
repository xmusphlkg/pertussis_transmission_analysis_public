"""Fail-fast quality gate for Bayesian posterior pipeline artifacts.

The sampler intentionally writes diagnostic artifacts even when a posterior
does not meet its quality thresholds.  That is useful for tuning, but a
publication pipeline must not interpret a warning-only sampler exit as a
successful posterior and launch downstream predictive simulations.
"""

from __future__ import annotations

import argparse
from typing import Any

from src_python.simulation.common import validate_run_metadata


def posterior_quality_failures(
    metadata: dict[str, Any],
    *,
    expected_sampler: str | None = None,
    expected_chains: int | None = None,
    require_recommended: bool = False,
) -> list[str]:
    """Return publication-blocking posterior metadata failures."""

    failures: list[str] = []
    sampler = str(metadata.get("sampler", "")).strip().lower()
    if expected_sampler is not None and sampler != str(expected_sampler).strip().lower():
        failures.append(
            f"sampler={sampler or 'missing'} does not match expected sampler={expected_sampler}"
        )

    if expected_chains is not None:
        try:
            observed_chains = int(metadata.get("n_chains", -1))
        except (TypeError, ValueError):
            observed_chains = -1
        if observed_chains != int(expected_chains):
            failures.append(
                f"n_chains={observed_chains} does not match expected n_chains={int(expected_chains)}"
            )

    summary = metadata.get("convergence_summary")
    if not isinstance(summary, dict) or not summary:
        failures.append("convergence_summary is missing")
        return failures

    try:
        total = int(summary.get("n_parameters_total", 0))
        converged = int(summary.get("n_parameters_converged", 0))
    except (TypeError, ValueError):
        total = 0
        converged = 0
    if total <= 0:
        failures.append("convergence_summary reports no posterior parameters")
    if not bool(summary.get("all_converged", False)) or (total > 0 and converged != total):
        failures.append(
            "minimum posterior quality failed "
            f"({converged}/{total} parameters; worst_rhat={summary.get('worst_rhat')}; "
            f"min_bulk_ess={summary.get('min_bulk_ess')}; "
            f"min_tail_ess={summary.get('min_tail_ess')})"
        )

    if require_recommended:
        try:
            recommended = int(summary.get("n_parameters_recommended_converged", 0))
        except (TypeError, ValueError):
            recommended = 0
        if not bool(summary.get("all_recommended_converged", False)) or (
            total > 0 and recommended != total
        ):
            failures.append(
                "recommended publication posterior quality failed "
                f"({recommended}/{total} parameters)"
            )
    return failures


def check_bayesian_quality(
    stem: str,
    *,
    expected_sampler: str | None = None,
    expected_chains: int | None = None,
    require_recommended: bool = False,
) -> dict[str, Any]:
    """Validate metadata structure and quality, raising on any blocking failure."""

    metadata = validate_run_metadata(stem)
    failures = posterior_quality_failures(
        metadata,
        expected_sampler=expected_sampler,
        expected_chains=expected_chains,
        require_recommended=require_recommended,
    )
    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(f"Bayesian posterior quality gate failed for {stem}:\n  - {details}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail unless a Bayesian posterior artifact meets pipeline quality requirements."
    )
    parser.add_argument("--stem", required=True, help="Posterior run-metadata stem")
    parser.add_argument("--expected-sampler", default=None)
    parser.add_argument("--expected-chains", type=int, default=None)
    parser.add_argument("--require-recommended", action="store_true")
    args = parser.parse_args()

    try:
        metadata = check_bayesian_quality(
            args.stem,
            expected_sampler=args.expected_sampler,
            expected_chains=args.expected_chains,
            require_recommended=bool(args.require_recommended),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    summary = metadata["convergence_summary"]
    print(
        f"Bayesian posterior quality gate passed for {args.stem}: "
        f"{summary['n_parameters_converged']}/{summary['n_parameters_total']} parameters."
    )


if __name__ == "__main__":
    main()
