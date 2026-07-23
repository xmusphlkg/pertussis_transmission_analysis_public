from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_r_figure2_uncertainty_registry_schema_is_strict() -> None:
    expression = r"""
    source('scripts_R/figures/figure_2/data.R')

    stopifnot(identical(
      figure_2_validate_uncertainty_registry_schema(list(schema_version = 1L)),
      1L
    ))
    stopifnot(identical(
      figure_2_validate_uncertainty_registry_schema(list(schema_version = 1)),
      1L
    ))

    invalid_registries <- list(
      missing = list(),
      string = list(schema_version = '1'),
      logical = list(schema_version = TRUE),
      zero = list(schema_version = 0L),
      future = list(schema_version = 2L)
    )
    for (registry in invalid_registries) {
      message <- tryCatch(
        {
          figure_2_validate_uncertainty_registry_schema(registry)
          NA_character_
        },
        error = function(error) conditionMessage(error)
      )
      stopifnot(
        !is.na(message),
        grepl('schema_version', message, fixed = TRUE)
      )
    }
    """

    subprocess.run(
        ["Rscript", "-e", expression],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_r_figure2_full_three_panel_contract_and_fail_closed_audits() -> None:
    expression = r"""
    source('scripts_R/lib/bootstrap.R')
    source('scripts_R/figures/figure_2/data.R')

    contract <- figure_2_publication_contract()
    countries <- contract$countries
    strategies <- contract$strategies
    labels <- figure_2_strategy_labels()
    n_country <- length(countries)
    n_strategy <- length(strategies)

    current_values <- setNames(seq(250, 1050, length.out = n_country), countries)
    reference <- tidyr::expand_grid(
      country = countries,
      strategy = c('current', strategies)
    ) %>%
      mutate(
        country_index = match(country, countries),
        strategy_index = match(strategy, strategies),
        leader_index = ((country_index - 1L) %% n_strategy) + 1L,
        deterministic_effect = if_else(
          strategy == 'current',
          0,
          80 - 3 * (strategy_index - leader_index)^2 + 0.01 * strategy_index
        ),
        annualized_child_adolescent_cases_per_100k =
          unname(current_values[country]) - deterministic_effect,
        relative_reduction_child_adolescent_cases =
          1 - annualized_child_adolescent_cases_per_100k / unname(current_values[country])
      ) %>%
      select(
        country,
        strategy,
        annualized_child_adolescent_cases_per_100k,
        relative_reduction_child_adolescent_cases
      )

    bootstrap <- tidyr::expand_grid(
      country = countries,
      bootstrap_replicate = seq_len(contract$minimum_successful_replicates),
      strategy = strategies
    ) %>%
      mutate(
        country_index = match(country, countries),
        strategy_index = match(strategy, strategies),
        leader_index = ((country_index - 1L) %% n_strategy) + 1L,
        current_rate = unname(current_values[country]) +
          0.05 * sin(bootstrap_replicate / 7),
        deterministic_effect = 80 - 3 * (strategy_index - leader_index)^2 +
          0.01 * strategy_index,
        intervention_rate = current_rate - deterministic_effect +
          0.15 * cos((bootstrap_replicate + strategy_index) / 11),
        relative_reduction_child_adolescent_cases =
          1 - intervention_rate / current_rate
      ) %>%
      select(
        country,
        bootstrap_replicate,
        strategy,
        current_rate,
        intervention_rate,
        relative_reduction_child_adolescent_cases
      )

    intervention <- tidyr::expand_grid(
      country = countries,
      scenario = c('current', 'higher_child_coverage')
    ) %>%
      mutate(
        country_index = match(country, countries),
        coverage_effect = if_else(
          scenario == 'current',
          0,
          2 + 0.1 * country_index
        ),
        annualized_child_adolescent_cases_per_100k =
          unname(current_values[country]) - coverage_effect,
        relative_reduction_child_adolescent_cases =
          1 - annualized_child_adolescent_cases_per_100k /
            unname(current_values[country])
      ) %>%
      select(
        country,
        scenario,
        annualized_child_adolescent_cases_per_100k,
        relative_reduction_child_adolescent_cases
      )

    rank <- tidyr::expand_grid(
      country = countries,
      psa_sample_id = seq_len(contract$rank_samples),
      strategy = strategies
    ) %>%
      mutate(
        country_index = match(country, countries),
        strategy_index = match(strategy, strategies),
        rank = ((strategy_index + psa_sample_id + country_index - 3L) %%
          n_strategy) + 1L,
        setting_current_rate = unname(current_values[country]) +
          0.1 * sin(psa_sample_id / 7),
        annualized_child_adolescent_cases_per_100k =
          setting_current_rate + rank,
        relative_reduction_under18_cases_vs_current =
          1 - annualized_child_adolescent_cases_per_100k /
            setting_current_rate,
        total_child_adolescent_cases =
          1e6 * annualized_child_adolescent_cases_per_100k
      ) %>%
      select(
        country, psa_sample_id, strategy, rank,
        total_child_adolescent_cases,
        annualized_child_adolescent_cases_per_100k,
        relative_reduction_under18_cases_vs_current
      )

    shared <- list(
      git = list(commit = 'commit-current'),
      generated_at_utc = '2026-07-17T00:00:00+00:00',
      countries = countries
    )
    reference_metadata <- utils::modifyList(shared, list(
      common_current_practice_denominator = TRUE,
      runtime_contract = 'production runtime; no diagnostic solver coarsening',
      programme_only_strategies = strategies,
      row_counts = list(summary = nrow(reference))
    ))
    bootstrap_metadata <- utils::modifyList(shared, list(
      strategies = strategies,
      statistical_target = 'frequentist_confidence_interval',
      confidence_interval_method = 'percentile_parametric_bootstrap',
      interval_type = '95% parametric-bootstrap confidence interval',
      interval_basis = 'Paired marginal parametric bootstrap; state-space model fully refitted.',
      bootstrap_data_generation = 'marginal_AR1_process_plus_NB2_measurement',
      bootstrap_refit = 'country_state_space_MAP_full_refit_per_replicate',
      analysis_role = 'publication_estimation_confidence_interval',
      varied_estimation_components = c(
        'annual_AR1_latent_transmission_path',
        'NB2_surveillance_observations',
        'state_space_MAP_refit'
      ),
      publication_path = TRUE,
      figure2c_interval_source = TRUE,
      posterior_credible_interval = FALSE,
      future_observation_prediction_interval = FALSE,
      fixed_reference_inputs = c(
        'biological_parameters', 'intervention_definitions',
        'AR1_hyperparameters', 'NB2_dispersion'
      ),
      replicates_requested_per_country = contract$bootstrap_replicates,
      minimum_successful_replicates = contract$minimum_successful_replicates,
      row_counts = list(paired_bootstrap_draws = nrow(bootstrap))
    ))
    rank_metadata <- utils::modifyList(shared, list(
      programme_only_strategies = strategies,
      figure2b_parameter_names = c(
        'infant_contact_multiplier', 'VE_inf_baseline',
        'relative_infectiousness_asymptomatic',
        'infectious_duration_asymptomatic', 'fitness_R',
        'PEP_coverage_multiplier'
      ),
      parameter_time_scopes = c(
        infant_contact_multiplier = 'structural_all_time',
        VE_inf_baseline = 'structural_all_time',
        relative_infectiousness_asymptomatic = 'structural_all_time',
        infectious_duration_asymptomatic = 'structural_all_time',
        fitness_R = 'structural_all_time',
        PEP_coverage_multiplier = 'prospective_implementation'
      ),
      excluded_dead_dimensions = 'resistance_management_uptake',
      run_status = 'complete',
      sample_size_requested = contract$rank_samples,
      completed_sample_count = contract$rank_samples,
      row_counts = list(under18_programme_rank_samples = nrow(rank)),
      completion_audit = list(
        exact_requested_samples_per_profile = TRUE,
        all_profiles_complete = TRUE,
        exact_one_rank1_strategy_per_profile_sample = TRUE,
        shared_sample_ids_across_profiles = TRUE,
        parameter_sample_ids_match_rank_parent = TRUE,
        parameter_sample_values_match_rank_parent = TRUE,
        ranks_recomputed_from_unrounded_burdens = TRUE
      )
    ))
    intervention_metadata <- utils::modifyList(shared, list(
      row_counts = list(summary = nrow(intervention))
    ))
    inputs <- list(
      reference = reference,
      bootstrap = bootstrap,
      rank = rank,
      intervention = intervention,
      metadata = list(
        reference = reference_metadata,
        bootstrap = bootstrap_metadata,
        rank = rank_metadata,
        intervention = intervention_metadata
      )
    )

    prepared <- prepare_figure_2_data(inputs)
    stopifnot(nrow(prepared$panel_a) == n_country * 2L)
    stopifnot(nrow(prepared$panel_b) == n_country)
    stopifnot(nrow(prepared$panel_c) == n_country * n_strategy)
    stopifnot(nrow(prepared$programme_effects) == n_country * n_strategy)
    stopifnot(setequal(
      as.character(unique(prepared$panel_a$scenario)),
      c('coverage_floor_only', 'timeliness_only')
    ))
    stopifnot(all(prepared$panel_b$setting_count == contract$rank_samples))
    stopifnot(all(
      prepared$panel_b$reference_choice_retained >= 0L &
        prepared$panel_b$reference_choice_retained <= contract$rank_samples
    ))
    stopifnot(all(dplyr::near(
      prepared$panel_b$retained_fraction,
      prepared$panel_b$reference_choice_retained / contract$rank_samples
    )))
    stopifnot(all(prepared$panel_b$regret_pp_q95 >= 0))
    stopifnot(all(
      prepared$panel_c %>% group_by(country) %>%
        summarise(n = sum(preferred_in_program_only)) %>% pull(n) == 1L
    ))
    stopifnot(nrow(prepared$regret_draws) == n_country * contract$rank_samples)
    stopifnot(all(
      prepared$rank_counts %>% group_by(country) %>%
        summarise(n = sum(count)) %>% pull(n) == 128L
    ))

    source('scripts_R/figures/figure_2/panels.R')
    source('scripts_R/figures/figure_2/assemble.R')
    panels <- plot_figure_2_panels(prepared)
    stopifnot(setequal(names(panels), c('a', 'b', 'c')))
    stopifnot(all(vapply(panels, inherits, logical(1), what = 'ggplot')))
    stopifnot(inherits(assemble_figure_2(panels), 'patchwork'))

    source('scripts_R/figures/figure_2/source_data.R')
    stopifnot(identical(
      unname(figure_2_source_filenames[['decision_fragility']]),
      'figure2b_reference_choice_fragility.csv'
    ))
    source_data <- make_figure_2_source_data(prepared)
    stopifnot(nrow(source_data$delivery_lever_contrast) == n_country * 2L)
    stopifnot(nrow(source_data$decision_fragility) == n_country)
    stopifnot(nrow(source_data$regret_draws) == n_country * contract$rank_samples)
    stopifnot(nrow(source_data$effect_matrix) == n_country * n_strategy)
    stopifnot(nrow(source_data$rank1_counts) == n_country * n_strategy)
    stopifnot(nrow(source_data$provenance) == 4L)
    stopifnot(all(c("git_commit", "generated_at_utc", "estimand_or_role") %in%
      names(source_data$provenance)))
    stopifnot(all(
      source_data$rank1_counts %>% group_by(country) %>%
        summarise(n = sum(rank1_count)) %>% pull(n) == 128L
    ))

    source('scripts_R/figures/figure_2/render.R')
    transaction_dir <- tempfile('figure2-transaction-')
    dir.create(transaction_dir)
    written_source <- write_figure_2_source_data(
      source_data,
      output_dir = transaction_dir
    )
    stopifnot(length(written_source) == 6L, all(file.exists(written_source)))
    final_paths <- file.path(transaction_dir, c('one.txt', 'two.txt'))
    staged_paths <- file.path(transaction_dir, c('one.tmp', 'two.tmp'))
    names(final_paths) <- names(staged_paths) <- c('one', 'two')
    writeLines(c('old one'), final_paths[[1]])
    writeLines(c('old two'), final_paths[[2]])
    writeLines(c('new one'), staged_paths[[1]])
    writeLines(c('new two'), staged_paths[[2]])
    replace_figure_2_outputs_transactionally(
      staged_paths,
      final_paths,
      '.test'
    )
    stopifnot(
      identical(readLines(final_paths[[1]]), 'new one'),
      identical(readLines(final_paths[[2]]), 'new two'),
      all(as.character(file.info(final_paths)$mode) == '664'),
      !any(file.exists(paste0(final_paths, '.backup.test')))
    )
    unlink(transaction_dir, recursive = TRUE)

    mixed_denominator <- inputs
    bad_row <- mixed_denominator$reference$country == countries[[1]] &
      mixed_denominator$reference$strategy == strategies[[1]]
    mixed_denominator$reference$relative_reduction_child_adolescent_cases[bad_row] <-
      mixed_denominator$reference$relative_reduction_child_adolescent_cases[bad_row] + 0.1
    stopifnot(tryCatch({
      prepare_figure_2_data(mixed_denominator)
      FALSE
    }, error = function(e) grepl('1 - burden/current', conditionMessage(e), fixed = TRUE)))

    partial_rank <- inputs
    partial_rank$rank <- partial_rank$rank[-1, ]
    partial_rank$metadata$rank$row_counts$under18_programme_rank_samples <- nrow(partial_rank$rank)
    stopifnot(tryCatch({
      prepare_figure_2_data(partial_rank)
      FALSE
    }, error = function(e) TRUE))

    legacy_seven_input_rank <- inputs
    legacy_seven_input_rank$metadata$rank$figure2b_parameter_names <- c(
      legacy_seven_input_rank$metadata$rank$figure2b_parameter_names,
      'resistance_management_uptake'
    )
    stopifnot(tryCatch({
      prepare_figure_2_data(legacy_seven_input_rank)
      FALSE
    }, error = function(e) grepl('six-input scope contract', conditionMessage(e), fixed = TRUE)))

    posterior_interval <- inputs
    posterior_interval$metadata$bootstrap$statistical_target <-
      'posterior_credible_interval'
    posterior_interval$metadata$bootstrap$posterior_credible_interval <- TRUE
    stopifnot(tryCatch({
      prepare_figure_2_data(posterior_interval)
      FALSE
    }, error = function(e) grepl('full-refit bootstrap', conditionMessage(e), fixed = TRUE)))

    mixed_production_current <- inputs
    bad_current <- mixed_production_current$intervention$country == countries[[1]] &
      mixed_production_current$intervention$scenario == 'current'
    mixed_production_current$intervention$annualized_child_adolescent_cases_per_100k[bad_current] <-
      mixed_production_current$intervention$annualized_child_adolescent_cases_per_100k[bad_current] + 1
    stopifnot(tryCatch({
      prepare_figure_2_data(mixed_production_current)
      FALSE
    }, error = function(e) grepl('same current-practice denominator', conditionMessage(e), fixed = TRUE)))

    rounded_rank_mismatch <- inputs
    rounded_rank_mismatch$rank$rank[[1]] <- 6L
    stopifnot(tryCatch({
      prepare_figure_2_data(rounded_rank_mismatch)
      FALSE
    }, error = function(e) grepl('unrounded burdens', conditionMessage(e), fixed = TRUE)))
    """

    subprocess.run(
        ["Rscript", "-e", expression],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
