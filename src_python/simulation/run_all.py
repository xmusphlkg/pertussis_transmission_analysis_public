from __future__ import annotations

import argparse
import os

from src_python.simulation.common import (
    load_configs,
    publication_country_names,
    write_manuscript_tables,
)
from src_python.simulation.run_age_pattern_sensitivity import main as run_age_pattern_sensitivity
from src_python.simulation.run_bayesian_uncertainty import (
    DEFAULT_OUTPUT_STEM as BAYESIAN_RESEARCH_OUTPUT_STEM,
    main as run_bayesian_uncertainty,
)
from src_python.simulation.run_baseline import main as run_baseline
from src_python.simulation.run_calibration_diagnostics import run as run_calibration_diagnostics
from src_python.simulation.run_country_scenarios import main as run_countries
from src_python.simulation.run_fitness_grid import main as run_fitness_grid
from src_python.simulation.run_heatmap_grid import main as run_heatmap
from src_python.simulation.run_immunity_sensitivity import main as run_immunity
from src_python.simulation.run_individual_stochastic_toy import main as run_individual_stochastic_toy
from src_python.simulation.run_infant_contact_sensitivity import main as run_infant_contact_sensitivity
from src_python.simulation.run_intervention_scenarios import main as run_interventions
from src_python.simulation.run_joint_psa_rank_acceptability import SELECTED_STRATEGIES, run_joint_psa
from src_python.simulation.run_maternal_duration_sensitivity import main as run_maternal_duration_sensitivity
from src_python.simulation.run_program_portfolio_factorial import main as run_program_portfolio_factorial
from src_python.simulation.run_reporting_scenarios import main as run_reporting
from src_python.simulation.run_resistance_fitness_sensitivity import main as run_resistance_fitness
from src_python.simulation.run_resistance_hindcast import run as run_hindcast
from src_python.simulation.run_resistance_mechanism_decomposition import main as run_resistance_mechanism
from src_python.simulation.run_resistance_scenarios import main as run_resistance
from src_python.simulation.run_routine_timeliness_sensitivity import main as run_routine_timeliness
from src_python.simulation.run_sensitivity import main as run_sensitivity
from src_python.simulation.run_shock_recovery_sensitivity import main as run_shock_recovery_sensitivity
from src_python.simulation.run_temporal_assumption_sensitivity import main as run_temporal_assumption_sensitivity
from src_python.simulation.run_treatment_implementation_sensitivity import (
    main as run_treatment_implementation_sensitivity,
)
from src_python.simulation.run_vaccine_scenarios import main as run_vaccines


BAYESIAN_FIXED_PARAMETERS = (
    "reporting_multiplier",
    "VE_sus",
    "VE_inf",
    "VE_dur",
    "relative_infectiousness_asymptomatic",
    "fitness_R",
)
PUBLICATION_JOINT_PSA_SAMPLES = 128
PUBLICATION_JOINT_PSA_SEED = 20260521
PUBLICATION_JOINT_PSA_BATCH_SIZE = 8
PUBLICATION_FITNESS_POSTERIOR_DRAWS = 100
PUBLICATION_FITNESS_POSTERIOR_BATCH_SIZE = 32
PUBLICATION_FITNESS_PSA_BENEFIT_SAMPLES = -1
PUBLICATION_FITNESS_PSA_BATCH_SIZE = 1


def main(
    n_jobs: int | None = None,
    include_bayesian: bool = False,
    include_publication_diagnostics: bool = False,
) -> None:
    if n_jobs is not None:
        os.environ["PERTUSSIS_N_JOBS"] = str(n_jobs)

    run_baseline()
    run_vaccines()
    run_resistance()
    run_reporting()
    run_countries()
    run_heatmap()
    run_fitness_grid(n_jobs=n_jobs, posterior_draws=0)
    run_interventions()
    run_routine_timeliness(n_jobs=n_jobs)
    run_sensitivity()
    run_immunity()
    run_resistance_fitness(n_jobs=n_jobs)
    if include_bayesian:
        run_bayesian_uncertainty(
            n_jobs=n_jobs,
            solver_mode="calibration",
            sampler="state_space_exact_importance_cut",
            warmup=0,
            draws=128,
            n_chains=4,
            fix_durations=False,
            fixed_parameters=(),
            importance_nuisance_draws=128,
            output_stem=BAYESIAN_RESEARCH_OUTPUT_STEM,
        )
    if include_publication_diagnostics:
        configs = load_configs()
        countries = tuple(publication_country_names(configs))
        run_hindcast(n_jobs=n_jobs)
        run_calibration_diagnostics(countries)
        run_age_pattern_sensitivity()
        run_resistance_mechanism(n_jobs=n_jobs)
        run_program_portfolio_factorial(n_jobs=n_jobs)
        run_infant_contact_sensitivity(n_jobs=n_jobs)
        run_maternal_duration_sensitivity(n_jobs=n_jobs)
        run_shock_recovery_sensitivity(n_jobs=n_jobs)
        run_temporal_assumption_sensitivity(n_jobs=n_jobs)
        run_treatment_implementation_sensitivity(n_jobs=n_jobs)
        run_individual_stochastic_toy()
        run_joint_psa(
            sample_size=PUBLICATION_JOINT_PSA_SAMPLES,
            seed=PUBLICATION_JOINT_PSA_SEED,
            countries=countries,
            strategies=SELECTED_STRATEGIES,
            n_jobs=n_jobs,
            sample_batch_size=PUBLICATION_JOINT_PSA_BATCH_SIZE,
            resume=True,
            smoke_runtime=False,
            keep_timeseries=False,
        )
        run_fitness_grid(
            n_jobs=n_jobs,
            posterior_draws=PUBLICATION_FITNESS_POSTERIOR_DRAWS,
            posterior_batch_size=PUBLICATION_FITNESS_POSTERIOR_BATCH_SIZE,
            psa_benefit_samples=PUBLICATION_FITNESS_PSA_BENEFIT_SAMPLES,
            psa_batch_size=PUBLICATION_FITNESS_PSA_BATCH_SIZE,
            run_deterministic_grid=False,
        )
        write_manuscript_tables()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the core pertussis simulation suite.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional parallel worker cap for all scenario stages.",
    )
    parser.add_argument(
        "--include-bayesian",
        action="store_true",
        help=(
            "Also run the optional nonpublication conditional state-space research "
            "route with exact-target importance correction. This does not generate "
            "Figure 2c intervals."
        ),
    )
    parser.add_argument(
        "--include-publication-diagnostics",
        action="store_true",
        help=(
            "Also run the simulation-only publication diagnostics subset of "
            "make publication-data. This does not run manuscript_notes publication "
            "table scripts or the rolling-origin release gate; requires existing "
            "optional research uncertainty samples unless "
            "--include-bayesian is also used."
        ),
    )
    args = parser.parse_args()
    main(
        n_jobs=args.n_jobs,
        include_bayesian=args.include_bayesian,
        include_publication_diagnostics=args.include_publication_diagnostics,
    )
