## Figure 2 data preparation ---------------------------------------------------

figure_2_strategy_order <- function() {
  c(
    "timeliness_only",
    "maternal_immunization",
    "pregnancy_tdap_scaleup",
    "adolescent_booster",
    "cocooning_adjunct",
    "targeted_pep_high_risk"
  )
}

figure_2_strategy_labels <- function() {
  c(
    timeliness_only = "Routine schedule timeliness",
    maternal_immunization = "Infant-exposure package",
    pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up",
    adolescent_booster = "Adolescent booster",
    cocooning_adjunct = "Close-contact adult adjunct",
    targeted_pep_high_risk = "Targeted high-risk PEP"
  )
}

figure_2_strategy_short_labels <- function() {
  c(
    timeliness_only = "Routine schedule timeliness",
    maternal_immunization = "Infant-exposure package",
    pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up",
    adolescent_booster = "Adolescent booster scale-up",
    cocooning_adjunct = "Close-contact adult adjunct",
    targeted_pep_high_risk = "Targeted high-risk PEP"
  )
}

figure_2_strategy_colours <- function() {
  c(
    timeliness_only = manuscript_discrete_core[["blue"]],
    maternal_immunization = manuscript_discrete_core[["green"]],
    pregnancy_tdap_scaleup = manuscript_discrete_core[["purple"]],
    adolescent_booster = manuscript_discrete_core[["orange"]],
    cocooning_adjunct = manuscript_discrete_core[["deep_teal"]],
    targeted_pep_high_risk = manuscript_discrete_core[["vermillion"]]
  )
}

figure_2_timeliness_scenario_order <- function() {
  c(
    "coverage_floor_only",
    "timeliness_only"
  )
}

figure_2_timeliness_scenario_labels <- function() {
  c(
    coverage_floor_only = "Coverage floor only",
    timeliness_only = "Timeliness only"
  )
}

figure_2_normalize_country <- function(x) {
  stringr::str_replace_all(as.character(x), " ", "_")
}

figure_2_validate_uncertainty_registry_schema <- function(distributions) {
  if (!is.list(distributions)) {
    stop(
      "Figure 2 requires parameter_distributions.yaml to contain a mapping.",
      call. = FALSE
    )
  }
  schema_version <- distributions$schema_version
  valid <- is.numeric(schema_version) &&
    !is.logical(schema_version) &&
    length(schema_version) == 1L &&
    !is.na(schema_version) &&
    is.finite(schema_version) &&
    schema_version == 1 &&
    schema_version == as.integer(schema_version)
  if (!isTRUE(valid)) {
    stop(
      paste0(
        "Figure 2 requires parameter_distributions.yaml schema_version ",
        "to be the numeric scalar integer 1."
      ),
      call. = FALSE
    )
  }
  as.integer(schema_version)
}

figure_2_publication_contract <- function() {
  settings_path <- model_path("config", "model_settings.yaml")
  profiles_path <- model_path("config", "country_profiles.yaml")
  distributions_path <- model_path("config", "parameter_distributions.yaml")
  if (any(!file.exists(c(settings_path, profiles_path, distributions_path)))) {
    stop("Figure 2 requires the current settings, profiles, and uncertainty registry.", call. = FALSE)
  }
  settings <- yaml::read_yaml(settings_path)
  profiles <- names(yaml::read_yaml(profiles_path))
  distributions <- yaml::read_yaml(distributions_path)
  figure_2_validate_uncertainty_registry_schema(distributions)
  exclusions <- settings$runtime$bayesian_uncertainty$publication_country_exclusions
  exclusions <- if (is.null(exclusions)) character() else names(exclusions)
  bootstrap <- settings$runtime$bayesian_uncertainty$figure2c_parametric_bootstrap_confidence_interval
  rank_samples <- as.integer(distributions$joint_rank_psa$sample_size)
  if (length(rank_samples) != 1L || is.na(rank_samples) || rank_samples != 128L) {
    stop("Figure 2 requires exactly 128 prespecified selected-input settings.", call. = FALSE)
  }
  list(
    countries = figure_2_normalize_country(setdiff(profiles, exclusions)),
    strategies = figure_2_strategy_order(),
    timeliness_scenarios = figure_2_timeliness_scenario_order(),
    bootstrap_replicates = as.integer(bootstrap$replicates_per_country),
    minimum_successful_replicates = as.integer(bootstrap$minimum_successful_replicates),
    rank_samples = rank_samples
  )
}

figure_2_read_metadata <- function(stem) {
  path <- model_path("outputs", "metadata", paste0(stem, "_run_metadata.json"))
  if (!file.exists(path)) {
    stop("Missing Figure 2 parent metadata: ", path, call. = FALSE)
  }
  jsonlite::read_json(path, simplifyVector = TRUE)
}

figure_2_validate_current_parent_metadata <- function() {
  validator <- model_path("manuscript_notes", "validate_figure2_parent_metadata.py")
  python <- model_path(".venv", "bin", "python")
  if (!file.exists(python)) python <- Sys.which("python3")
  if (!nzchar(python) || !file.exists(validator)) {
    stop("Figure 2 parent-metadata validator is unavailable.", call. = FALSE)
  }
  output <- system2(python, validator, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(
      "Figure 2 parent metadata are invalid: ",
      paste(output, collapse = "\n"),
      call. = FALSE
    )
  }
  invisible(output)
}

figure_2_optional_metadata_value <- function(x, key) {
  value <- x[[key]]
  if (is.null(value) || length(value) != 1L || is.na(value) || !nzchar(as.character(value))) {
    return(NA_character_)
  }
  as.character(value)
}

validate_figure_2_parent_metadata <- function(inputs, contract) {
  parents <- inputs$metadata
  required_parents <- c("reference", "bootstrap", "rank", "intervention")
  if (is.null(parents) || !setequal(names(parents), required_parents)) {
    stop(
      "Figure 2 requires reference, bootstrap, rank, and intervention parent metadata.",
      call. = FALSE
    )
  }

  commits <- vapply(parents[required_parents], function(parent) {
    figure_2_optional_metadata_value(parent$git, "commit")
  }, character(1))

  reference <- parents$reference
  bootstrap <- parents$bootstrap
  rank <- parents$rank
  intervention <- parents$intervention
  metadata_countries <- lapply(parents[c("reference", "bootstrap", "rank")], function(parent) {
    figure_2_normalize_country(parent$countries)
  })
  if (any(!vapply(
    metadata_countries,
    function(values) setequal(values, contract$countries),
    logical(1)
  ))) {
    stop("Figure 2 parent metadata do not contain the publication-profile set.", call. = FALSE)
  }
  metadata_strategies <- list(
    reference = as.character(reference$programme_only_strategies),
    bootstrap = as.character(bootstrap$strategies),
    rank = as.character(rank$programme_only_strategies)
  )
  if (any(!vapply(
    metadata_strategies,
    function(values) setequal(values, contract$strategies),
    logical(1)
  ))) {
    stop("Figure 2 parent metadata do not contain the same six strategies.", call. = FALSE)
  }
  expected_rank_parameters <- c(
    "infant_contact_multiplier",
    "VE_inf_baseline",
    "relative_infectiousness_asymptomatic",
    "infectious_duration_asymptomatic",
    "fitness_R",
    "PEP_coverage_multiplier"
  )
  rank_parameters <- as.character(rank$figure2b_parameter_names)
  rank_scopes <- unlist(rank$parameter_time_scopes, use.names = TRUE)
  expected_rank_scopes <- stats::setNames(
    c(rep("structural_all_time", 5L), "prospective_implementation"),
    expected_rank_parameters
  )
  if (!setequal(rank_parameters, expected_rank_parameters) ||
      !setequal(names(rank_scopes), expected_rank_parameters) ||
      !identical(
        unname(as.character(rank_scopes[expected_rank_parameters])),
        unname(expected_rank_scopes)
      ) ||
      !identical(
        as.character(rank$excluded_dead_dimensions),
        "resistance_management_uptake"
      )) {
    stop(
      "Figure 2b rank metadata do not implement the locked six-input scope contract.",
      call. = FALSE
    )
  }
  if (!isTRUE(reference$common_current_practice_denominator) ||
      !stringr::str_detect(
        stringr::str_to_lower(as.character(reference$runtime_contract)),
        "production runtime"
      )) {
    stop("Figure 2 reference rows are not locked to one production runtime.", call. = FALSE)
  }
  if (!identical(as.character(bootstrap$statistical_target),
                 "frequentist_confidence_interval") ||
      !identical(as.character(bootstrap$confidence_interval_method),
                 "percentile_parametric_bootstrap") ||
      !identical(as.character(bootstrap$interval_type),
                 "95% parametric-bootstrap confidence interval") ||
      !identical(as.character(bootstrap$bootstrap_data_generation),
                 "marginal_AR1_process_plus_NB2_measurement") ||
      !identical(as.character(bootstrap$bootstrap_refit),
                 "country_state_space_MAP_full_refit_per_replicate") ||
      !identical(as.character(bootstrap$analysis_role),
                 "publication_estimation_confidence_interval") ||
      !identical(
        as.character(bootstrap$varied_estimation_components),
        c(
          "annual_AR1_latent_transmission_path",
          "NB2_surveillance_observations",
          "state_space_MAP_refit"
        )
      ) ||
      !identical(as.logical(bootstrap$publication_path), TRUE) ||
      !identical(as.logical(bootstrap$figure2c_interval_source), TRUE) ||
      !identical(as.logical(bootstrap$posterior_credible_interval), FALSE) ||
      !identical(as.logical(bootstrap$future_observation_prediction_interval), FALSE) ||
      !setequal(
        as.character(bootstrap$fixed_reference_inputs),
        c(
          "biological_parameters",
          "intervention_definitions",
          "AR1_hyperparameters",
          "NB2_dispersion"
        )
      ) ||
      !stringr::str_detect(
        stringr::str_to_lower(as.character(bootstrap$interval_basis)),
        "fully refitted"
      )) {
    stop("Figure 2 intervals are not the audited paired full-refit bootstrap.", call. = FALSE)
  }
  if (as.integer(reference$row_counts$summary) !=
        length(contract$countries) * (length(contract$strategies) + 1L) ||
      as.integer(intervention$row_counts$summary) <
        length(contract$countries) * 2L ||
      as.integer(bootstrap$replicates_requested_per_country) !=
        contract$bootstrap_replicates ||
      as.integer(bootstrap$minimum_successful_replicates) !=
        contract$minimum_successful_replicates) {
    stop("Figure 2 parent metadata do not match the configured design sizes.", call. = FALSE)
  }
  if (!identical(as.character(rank$run_status), "complete") ||
      as.integer(rank$sample_size_requested) != contract$rank_samples ||
      as.integer(rank$completed_sample_count) != contract$rank_samples ||
      !isTRUE(rank$completion_audit$exact_requested_samples_per_profile) ||
      !isTRUE(rank$completion_audit$all_profiles_complete) ||
      !isTRUE(rank$completion_audit$exact_one_rank1_strategy_per_profile_sample) ||
      !isTRUE(rank$completion_audit$shared_sample_ids_across_profiles) ||
      !isTRUE(rank$completion_audit$parameter_sample_ids_match_rank_parent) ||
      !isTRUE(rank$completion_audit$parameter_sample_values_match_rank_parent) ||
      !isTRUE(rank$completion_audit$ranks_recomputed_from_unrounded_burdens)) {
    stop(
      "Figure 2 selected-input rank parent is not a completed 128-setting run.",
      call. = FALSE
    )
  }

  if (!is.null(inputs$paths)) {
    for (name in required_parents) {
      path <- inputs$paths[[name]]
      if (is.null(path) || !file.exists(path)) {
        stop("Missing Figure 2 parent artifact for ", name, ".", call. = FALSE)
      }
    }
  }

  list(
    git_commit = commits,
    reference_generated_at_utc = as.character(reference$generated_at_utc),
    bootstrap_generated_at_utc = as.character(bootstrap$generated_at_utc),
    rank_generated_at_utc = as.character(rank$generated_at_utc),
    intervention_generated_at_utc = as.character(intervention$generated_at_utc)
  )
}

load_figure_2_inputs <- function() {
  figure_2_validate_current_parent_metadata()
  paths <- list(
    reference = model_path(
      "outputs", "summaries", "figure2_programme_reference_summary.csv"
    ),
    bootstrap = model_path(
      "outputs", "tables", "figure2c_programme_paired_bootstrap_draws.csv"
    ),
    rank = model_path(
      "outputs", "tables", "joint_psa_under18_programme_rank_samples.csv"
    ),
    intervention = model_path(
      "outputs", "summaries", "intervention_scenarios_summary.csv"
    )
  )
  missing <- names(paths)[!vapply(paths, file.exists, logical(1))]
  if (length(missing) > 0L) {
    stop("Missing Figure 2 parent artifact(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }
  list(
    reference = readr::read_csv(paths$reference, show_col_types = FALSE),
    bootstrap = readr::read_csv(paths$bootstrap, show_col_types = FALSE),
    rank = readr::read_csv(paths$rank, show_col_types = FALSE),
    intervention = readr::read_csv(paths$intervention, show_col_types = FALSE),
    metadata = list(
      reference = figure_2_read_metadata("figure2_programme_reference"),
      bootstrap = figure_2_read_metadata("figure2c_parametric_bootstrap"),
      rank = figure_2_read_metadata("joint_psa_rank_acceptability"),
      intervention = figure_2_read_metadata("intervention_scenarios")
    ),
    paths = paths
  )
}

figure_2_quantiles <- function(x) {
  values <- stats::quantile(
    x,
    probs = c(0.025, 0.25, 0.5, 0.75, 0.975),
    names = FALSE,
    type = 7,
    na.rm = FALSE
  )
  tibble::tibble(
    lower = values[[1]],
    q25 = values[[2]],
    median = values[[3]],
    q75 = values[[4]],
    upper = values[[5]]
  )
}

figure_2_assert_close <- function(observed, expected, message, tolerance = 1e-9) {
  ok <- is.finite(observed) & is.finite(expected) &
    abs(observed - expected) <= tolerance * pmax(1, abs(expected))
  if (!all(ok)) stop(message, call. = FALSE)
  invisible(TRUE)
}

prepare_figure_2_data <- function(inputs = load_figure_2_inputs()) {
  contract <- figure_2_publication_contract()
  provenance <- validate_figure_2_parent_metadata(inputs, contract)
  strategies <- contract$strategies
  labels <- figure_2_strategy_labels()
  short_labels <- figure_2_strategy_short_labels()
  colours <- figure_2_strategy_colours()
  countries <- contract$countries

  require_columns(
    inputs$reference,
    c(
      "country", "strategy", "annualized_child_adolescent_cases_per_100k",
      "relative_reduction_child_adolescent_cases"
    ),
    "figure2_programme_reference_summary.csv"
  )
  reference <- inputs$reference %>%
    transmute(
      country = figure_2_normalize_country(country),
      strategy = as.character(strategy),
      primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k),
      stored_relative_reduction = as.numeric(relative_reduction_child_adolescent_cases)
    )
  expected_reference_strategies <- c("current", strategies)
  if (nrow(reference) != length(countries) * length(expected_reference_strategies) ||
      anyDuplicated(reference[c("country", "strategy")]) > 0L ||
      !setequal(unique(reference$country), countries) ||
      !setequal(unique(reference$strategy), expected_reference_strategies) ||
      any(!is.finite(reference$primary_cases_per_100k)) ||
      any(reference$primary_cases_per_100k <= 0)) {
    stop("Figure 2 reference parent is not one complete 7-row block per profile.", call. = FALSE)
  }
  current <- reference %>%
    filter(strategy == "current") %>%
    transmute(country, current_primary_cases_per_100k = primary_cases_per_100k)
  programme <- reference %>%
    filter(strategy %in% strategies) %>%
    select(-stored_relative_reduction) %>%
    left_join(
      reference %>% select(country, strategy, stored_relative_reduction),
      by = c("country", "strategy")
    ) %>%
    left_join(current, by = "country") %>%
    mutate(
      relative_reduction = 1 - primary_cases_per_100k / current_primary_cases_per_100k,
      strategy_order = match(strategy, strategies)
    )
  figure_2_assert_close(
    programme$stored_relative_reduction,
    programme$relative_reduction,
    "Figure 2 stored reductions do not equal 1 - burden/current."
  )

  ranked <- programme %>%
    arrange(country, primary_cases_per_100k, strategy_order) %>%
    group_by(country) %>%
    mutate(burden_rank = row_number()) %>%
    ungroup() %>%
    arrange(country, desc(relative_reduction), strategy_order) %>%
    group_by(country) %>%
    mutate(reduction_rank = row_number()) %>%
    ungroup()
  if (any(ranked$burden_rank != ranked$reduction_rank)) {
    stop("Figure 2 argmin burden and argmax reduction disagree.", call. = FALSE)
  }
  leaders <- ranked %>%
    filter(burden_rank == 1L) %>%
    transmute(
      country,
      leader_strategy = strategy,
      leader_cases_per_100k = primary_cases_per_100k,
      current_primary_cases_per_100k,
      leader_relative_reduction = relative_reduction
    )
  runners <- ranked %>%
    filter(burden_rank == 2L) %>%
    transmute(
      country,
      runner_strategy = strategy,
      runner_cases_per_100k = primary_cases_per_100k
    )
  selection <- leaders %>%
    inner_join(runners, by = "country") %>%
    mutate(
      point_effect = current_primary_cases_per_100k - leader_cases_per_100k,
      point_margin = runner_cases_per_100k - leader_cases_per_100k
    )
  if (nrow(selection) != length(countries) ||
      any(selection$point_effect < 0) || any(selection$point_margin < 0)) {
    stop("Figure 2 reference leader/runner selection is incomplete or invalid.", call. = FALSE)
  }

  timeliness_scenarios <- contract$timeliness_scenarios
  timeliness_labels <- figure_2_timeliness_scenario_labels()
  require_columns(
    inputs$intervention,
    c(
      "country", "scenario", "annualized_child_adolescent_cases_per_100k",
      "relative_reduction_child_adolescent_cases"
    ),
    "intervention_scenarios_summary.csv"
  )
  coverage_raw <- inputs$intervention %>%
    transmute(
      country = figure_2_normalize_country(country),
      scenario = as.character(scenario),
      primary_cases_per_100k = as.numeric(
        annualized_child_adolescent_cases_per_100k
      ),
      stored_relative_reduction = as.numeric(
        relative_reduction_child_adolescent_cases
      )
    ) %>%
    filter(
      country %in% countries,
      scenario %in% c("current", "higher_child_coverage")
    )
  if (nrow(coverage_raw) != length(countries) * 2L ||
      anyDuplicated(coverage_raw[c("country", "scenario")]) > 0L ||
      !setequal(unique(coverage_raw$country), countries) ||
      !setequal(
        unique(coverage_raw$scenario),
        c("current", "higher_child_coverage")
      ) ||
      any(!is.finite(coverage_raw$primary_cases_per_100k)) ||
      any(coverage_raw$primary_cases_per_100k <= 0)) {
    stop(
      "Figure 2 intervention parent lacks complete production current and coverage-floor arms.",
      call. = FALSE
    )
  }
  coverage_current <- coverage_raw %>%
    filter(scenario == "current") %>%
    transmute(country, current_primary_cases_per_100k = primary_cases_per_100k)
  figure_2_assert_close(
    coverage_current$current_primary_cases_per_100k,
    current$current_primary_cases_per_100k[
      match(coverage_current$country, current$country)
    ],
    paste0(
      "Figure 2 production coverage and timeliness parents do not share ",
      "the same current-practice denominator."
    )
  )
  coverage_programme <- coverage_raw %>%
    filter(scenario == "higher_child_coverage") %>%
    left_join(current, by = "country") %>%
    mutate(
      relative_reduction = 1 -
        primary_cases_per_100k / current_primary_cases_per_100k
    )
  figure_2_assert_close(
    coverage_programme$stored_relative_reduction,
    coverage_programme$relative_reduction,
    "Figure 2 coverage-floor reductions do not equal 1 - burden/current."
  )

  timeliness_programme <- bind_rows(
    coverage_programme %>%
      transmute(
        country,
        scenario = "coverage_floor_only",
        primary_cases_per_100k,
        current_primary_cases_per_100k,
        relative_reduction,
        coverage_floor_applied = TRUE,
        timeliness_applied = FALSE,
        parent_artifact = "intervention_scenarios_summary.csv"
      ),
    programme %>%
      filter(strategy == "timeliness_only") %>%
      transmute(
        country,
        scenario = "timeliness_only",
        primary_cases_per_100k,
        current_primary_cases_per_100k,
        relative_reduction,
        coverage_floor_applied = FALSE,
        timeliness_applied = TRUE,
        parent_artifact = "figure2_programme_reference_summary.csv"
      )
  ) %>%
    mutate(
      scenario_order = match(scenario, timeliness_scenarios),
      scenario_label = unname(timeliness_labels[scenario])
    )
  if (nrow(timeliness_programme) !=
        length(countries) * length(timeliness_scenarios) ||
      anyDuplicated(timeliness_programme[c("country", "scenario")]) > 0L ||
      anyNA(timeliness_programme$scenario_order) ||
      anyNA(timeliness_programme$scenario_label)) {
    stop("Figure 2 delivery-lever contrast is incomplete.", call. = FALSE)
  }

  require_columns(
    inputs$bootstrap,
    c(
      "country", "bootstrap_replicate", "strategy", "current_rate",
      "intervention_rate", "relative_reduction_child_adolescent_cases"
    ),
    "figure2c_programme_paired_bootstrap_draws.csv"
  )
  draws <- inputs$bootstrap %>%
    transmute(
      country = figure_2_normalize_country(country),
      bootstrap_replicate = as.integer(bootstrap_replicate),
      strategy = as.character(strategy),
      current_rate = as.numeric(current_rate),
      intervention_rate = as.numeric(intervention_rate),
      stored_relative_reduction = as.numeric(relative_reduction_child_adolescent_cases)
    ) %>%
    filter(country %in% countries, strategy %in% strategies)
  if (!setequal(unique(draws$country), countries) ||
      !setequal(unique(draws$strategy), strategies) ||
      any(!is.finite(as.matrix(draws %>% select(current_rate, intervention_rate)))) ||
      any(draws$current_rate <= 0) || any(draws$intervention_rate < 0)) {
    stop("Figure 2 bootstrap draws are incomplete or non-finite.", call. = FALSE)
  }
  figure_2_assert_close(
    draws$stored_relative_reduction,
    1 - draws$intervention_rate / draws$current_rate,
    "Figure 2 bootstrap relative reductions failed their paired arithmetic audit."
  )
  replicate_audit <- draws %>%
    group_by(country, bootstrap_replicate) %>%
    summarise(
      rows = n(),
      strategies = n_distinct(strategy),
      current_range = max(current_rate) - min(current_rate),
      .groups = "drop"
    )
  if (any(replicate_audit$rows != length(strategies)) ||
      any(replicate_audit$strategies != length(strategies)) ||
      any(replicate_audit$current_range > 1e-10)) {
    stop("Figure 2 bootstrap strategies are not paired within replicate.", call. = FALSE)
  }
  replicate_counts <- replicate_audit %>% count(country, name = "n_bootstrap")
  if (any(replicate_counts$n_bootstrap < contract$minimum_successful_replicates) ||
      any(replicate_counts$n_bootstrap > contract$bootstrap_replicates)) {
    stop("Figure 2 bootstrap replicate counts violate the configured audit gate.", call. = FALSE)
  }
  recorded_bootstrap_rows <- as.integer(inputs$metadata$bootstrap$row_counts$paired_bootstrap_draws)
  if (length(recorded_bootstrap_rows) != 1L ||
      is.na(recorded_bootstrap_rows) || recorded_bootstrap_rows != nrow(draws)) {
    stop("Figure 2 bootstrap row count does not match metadata.", call. = FALSE)
  }

  programme_effects <- draws %>%
    mutate(absolute_reduction = current_rate - intervention_rate) %>%
    group_by(country, strategy) %>%
    summarise(
      lower = stats::quantile(absolute_reduction, 0.025, names = FALSE),
      q25 = stats::quantile(absolute_reduction, 0.25, names = FALSE),
      median = stats::quantile(absolute_reduction, 0.50, names = FALSE),
      q75 = stats::quantile(absolute_reduction, 0.75, names = FALSE),
      upper = stats::quantile(absolute_reduction, 0.975, names = FALSE),
      relative_lower = stats::quantile(
        stored_relative_reduction, 0.025, names = FALSE
      ),
      relative_q25 = stats::quantile(
        stored_relative_reduction, 0.25, names = FALSE
      ),
      relative_median = stats::quantile(
        stored_relative_reduction, 0.50, names = FALSE
      ),
      relative_q75 = stats::quantile(
        stored_relative_reduction, 0.75, names = FALSE
      ),
      relative_upper = stats::quantile(
        stored_relative_reduction, 0.975, names = FALSE
      ),
      .groups = "drop"
    ) %>%
    left_join(replicate_counts, by = "country") %>%
    left_join(
      programme %>%
        transmute(
          country,
          strategy,
          point = current_primary_cases_per_100k - primary_cases_per_100k,
          deterministic_relative_reduction = relative_reduction,
          primary_cases_per_100k,
          current_primary_cases_per_100k
        ),
      by = c("country", "strategy")
    ) %>%
    mutate(
      strategy_label = unname(labels[strategy]),
      confidence_interval_type = paste0(
        "paired full-refit parametric-bootstrap 95% estimation ",
        "confidence interval"
      ),
      confidence_interval_method = "percentile_parametric_bootstrap"
    )

  require_columns(
    inputs$rank,
    c(
      "country", "psa_sample_id", "strategy", "rank",
      "total_child_adolescent_cases",
      "annualized_child_adolescent_cases_per_100k",
      "relative_reduction_under18_cases_vs_current"
    ),
    "joint_psa_under18_programme_rank_samples.csv"
  )
  raw_sample_id <- as.numeric(inputs$rank$psa_sample_id)
  raw_rank <- as.numeric(inputs$rank$rank)
  if (any(!is.finite(raw_sample_id)) ||
      any(abs(raw_sample_id - round(raw_sample_id)) > 1e-10) ||
      any(!is.finite(raw_rank)) ||
      any(abs(raw_rank - round(raw_rank)) > 1e-10)) {
    stop("Figure 2 rank and sample identifiers must be integer-like.", call. = FALSE)
  }
  rank_samples <- inputs$rank %>%
    transmute(
      country = figure_2_normalize_country(country),
      psa_sample_id = as.integer(round(raw_sample_id)),
      strategy = as.character(strategy),
      rank = as.integer(round(raw_rank)),
      total_child_adolescent_cases = as.numeric(total_child_adolescent_cases),
      primary_cases_per_100k = as.numeric(
        annualized_child_adolescent_cases_per_100k
      ),
      relative_reduction = as.numeric(
        relative_reduction_under18_cases_vs_current
      ),
      current_primary_cases_per_100k =
        primary_cases_per_100k / (1 - relative_reduction)
    ) %>%
    filter(country %in% countries, strategy %in% strategies)
  if (any(!is.finite(rank_samples$total_child_adolescent_cases)) ||
      any(rank_samples$total_child_adolescent_cases < 0) ||
      any(!is.finite(rank_samples$primary_cases_per_100k)) ||
      any(rank_samples$primary_cases_per_100k <= 0) ||
      any(!is.finite(rank_samples$relative_reduction)) ||
      any(rank_samples$relative_reduction >= 1)) {
    stop("Figure 2 rank burdens must be finite and non-negative.", call. = FALSE)
  }
  expected_rank_rows <- length(countries) * contract$rank_samples * length(strategies)
  rank_block_audit <- rank_samples %>%
    group_by(country, psa_sample_id) %>%
    summarise(
      rows = n(),
      strategies = n_distinct(strategy),
      ranks = n_distinct(rank),
      rank_one = sum(rank == 1L),
      rank_min = min(rank),
      rank_max = max(rank),
      current_range = max(current_primary_cases_per_100k) -
        min(current_primary_cases_per_100k),
      current_mean = mean(current_primary_cases_per_100k),
      .groups = "drop"
    )
  if (nrow(rank_samples) != expected_rank_rows ||
      !setequal(unique(rank_samples$country), countries) ||
      !setequal(unique(rank_samples$strategy), strategies) ||
      nrow(rank_block_audit) != length(countries) * contract$rank_samples ||
      any(rank_block_audit$rows != length(strategies)) ||
      any(rank_block_audit$strategies != length(strategies)) ||
      any(rank_block_audit$ranks != length(strategies)) ||
      any(rank_block_audit$rank_one != 1L) ||
      any(rank_block_audit$rank_min != 1L) ||
      any(rank_block_audit$rank_max != length(strategies)) ||
      any(rank_block_audit$current_range >
            1e-8 * pmax(1, rank_block_audit$current_mean))) {
    stop("Figure 2 rank parent is not a complete 9 x 128 x 6 design.", call. = FALSE)
  }
  samples_per_profile <- rank_block_audit %>% count(country, name = "samples")
  if (any(samples_per_profile$samples != contract$rank_samples)) {
    stop("Figure 2 rank parent does not contain exactly 128 settings per profile.", call. = FALSE)
  }
  sample_id_sets <- split(rank_samples$psa_sample_id, rank_samples$country) %>%
    lapply(function(values) sort(unique(values)))
  reference_sample_ids <- sample_id_sets[[1]]
  if (!all(vapply(sample_id_sets, identical, logical(1), reference_sample_ids))) {
    stop("Figure 2 profiles do not share the same 128 selected-input settings.", call. = FALSE)
  }
  rank_samples <- rank_samples %>%
    group_by(country, psa_sample_id) %>%
    mutate(
      recomputed_unrounded_rank = as.integer(base::rank(
        total_child_adolescent_cases,
        ties.method = "min"
      )),
      recomputed_rate_rank = as.integer(base::rank(
        primary_cases_per_100k,
        ties.method = "min"
      ))
    ) %>%
    ungroup()
  if (any(rank_samples$rank != rank_samples$recomputed_unrounded_rank) ||
      any(rank_samples$rank != rank_samples$recomputed_rate_rank)) {
    stop(
      paste0(
        "Figure 2 stored ranks disagree with ranks recomputed from unrounded ",
        "burdens (total or annualised)."
      ),
      call. = FALSE
    )
  }
  recorded_rank_rows <- as.integer(
    inputs$metadata$rank$row_counts$under18_programme_rank_samples
  )
  if (length(recorded_rank_rows) != 1L || is.na(recorded_rank_rows) ||
      recorded_rank_rows != nrow(rank_samples)) {
    stop("Figure 2 rank row count does not match completed-run metadata.", call. = FALSE)
  }
  rank_counts <- rank_samples %>%
    filter(rank == 1L) %>%
    count(country, strategy, name = "count") %>%
    tidyr::complete(
      country = countries,
      strategy = strategies,
      fill = list(count = 0L)
    ) %>%
    mutate(count = as.integer(count))
  count_audit <- rank_counts %>%
    group_by(country) %>%
    summarise(total = sum(count), .groups = "drop")
  if (any(count_audit$total != contract$rank_samples)) {
    stop("Figure 2 rank-1 counts do not sum to exactly 128 per profile.", call. = FALSE)
  }

  # Keep country axes aligned with Figure 1b and the other main figures.
  profile_order <- main_figure_country_order(
    stringr::str_replace_all(countries, "_", " ")
  )
  if (length(profile_order) != length(main_figure_country_label_levels) ||
      !setequal(profile_order, stringr::str_replace_all(countries, "_", " "))) {
    stop("Figure 2 countries do not match the fixed main-figure order.", call. = FALSE)
  }

  country_short_labels <- c(
    Australia = "AUS",
    Brazil = "Brazil",
    China = "China",
    Japan = "Japan",
    New_Zealand = "NZ",
    Sweden = "Sweden",
    Thailand = "Thailand",
    United_Kingdom = "UK",
    United_States = "USA"
  )

  panel_a <- timeliness_programme %>%
    mutate(
      country_label = stringr::str_replace_all(country, "_", " "),
      profile_factor = factor(country_label, levels = rev(profile_order)),
      scenario = factor(scenario, levels = timeliness_scenarios),
      relative_reduction_percent = 100 * relative_reduction
    ) %>%
    transmute(
      country,
      country_label,
      profile_factor,
      scenario,
      scenario_label,
      relative_reduction,
      relative_reduction_percent,
      primary_cases_per_100k,
      current_primary_cases_per_100k,
      coverage_floor_applied,
      timeliness_applied,
      parent_artifact
    )

  reference_choices <- selection %>%
    transmute(country, reference_strategy = leader_strategy)
  regret_draws <- rank_samples %>%
    inner_join(reference_choices, by = "country") %>%
    group_by(country, psa_sample_id, reference_strategy) %>%
    summarise(
      current_primary_cases_per_100k = mean(current_primary_cases_per_100k),
      reference_primary_cases_per_100k =
        primary_cases_per_100k[strategy == first(reference_strategy)][[1]],
      best_primary_cases_per_100k = min(primary_cases_per_100k),
      winning_strategy = strategy[which.min(total_child_adolescent_cases)][[1]],
      reference_choice_retained =
        rank[strategy == first(reference_strategy)][[1]] == 1L,
      .groups = "drop"
    ) %>%
    mutate(
      regret_cases_per_100k = pmax(
        0,
        reference_primary_cases_per_100k - best_primary_cases_per_100k
      ),
      regret_relative_to_current =
        regret_cases_per_100k / current_primary_cases_per_100k,
      regret_percentage_points = 100 * regret_relative_to_current
    )
  if (nrow(regret_draws) != length(countries) * contract$rank_samples ||
      any(!is.finite(regret_draws$regret_percentage_points)) ||
      any(regret_draws$regret_percentage_points < 0) ||
      any(regret_draws$regret_cases_per_100k < 0) ||
      any(
        regret_draws$reference_choice_retained &
          regret_draws$regret_cases_per_100k > 1e-8
      ) ||
      any(
        !regret_draws$reference_choice_retained &
          regret_draws$regret_cases_per_100k <= 1e-8
      )) {
    stop("Figure 2 fixed-reference regret calculation is incomplete or invalid.", call. = FALSE)
  }

  panel_b <- regret_draws %>%
    group_by(country, reference_strategy) %>%
    summarise(
      setting_count = n(),
      retained_fraction = mean(reference_choice_retained),
      reference_choice_retained = sum(reference_choice_retained),
      positive_regret_count = sum(regret_percentage_points > 1e-10),
      regret_pp_median = stats::median(regret_percentage_points),
      regret_pp_q75 = stats::quantile(
        regret_percentage_points, 0.75, names = FALSE
      ),
      regret_pp_q95 = stats::quantile(
        regret_percentage_points, 0.95, names = FALSE
      ),
      regret_pp_q975 = stats::quantile(
        regret_percentage_points, 0.975, names = FALSE
      ),
      regret_pp_max = max(regret_percentage_points),
      regret_cases_per_100k_q95 = stats::quantile(
        regret_cases_per_100k, 0.95, names = FALSE
      ),
      .groups = "drop"
    ) %>%
    mutate(
      country_label = stringr::str_replace_all(country, "_", " "),
      country_short_label = unname(country_short_labels[country]),
      profile_factor = factor(country_label, levels = rev(profile_order)),
      reference_strategy_label = unname(short_labels[reference_strategy])
    ) %>%
    transmute(
      country,
      country_label,
      country_short_label,
      profile_factor,
      reference_strategy,
      reference_strategy_label,
      setting_count,
      reference_choice_retained,
      retained_fraction,
      positive_regret_count,
      regret_pp_median,
      regret_pp_q75,
      regret_pp_q95,
      regret_pp_q975,
      regret_pp_max,
      regret_cases_per_100k_q95
    )

  panel_c <- programme_effects %>%
    left_join(
      ranked %>% select(country, strategy, burden_rank),
      by = c("country", "strategy")
    ) %>%
    mutate(
      country_label = stringr::str_replace_all(country, "_", " "),
      profile_factor = factor(country_label, levels = rev(profile_order)),
      strategy = factor(strategy, levels = strategies),
      point_percent = 100 * deterministic_relative_reduction,
      interval_lower_percent = 100 * relative_lower,
      interval_upper_percent = 100 * relative_upper,
      display_point_percent = if_else(
        abs(point_percent) < 0.05, 0, point_percent
      ),
      display_interval_lower_percent = if_else(
        abs(interval_lower_percent) < 0.05, 0, interval_lower_percent
      ),
      display_interval_upper_percent = if_else(
        abs(interval_upper_percent) < 0.05, 0, interval_upper_percent
      ),
      effect_label = paste0(
        format_lancet_fixed(display_point_percent, 1), "%\n[",
        format_lancet_fixed(display_interval_lower_percent, 1), ", ",
        format_lancet_fixed(display_interval_upper_percent, 1), "]"
      ),
      effect_text_colour = if_else(
        deterministic_relative_reduction >= 0.14,
        "white",
        lancet_text_colour
      ),
      preferred_in_program_only = burden_rank == 1L
    ) %>%
    transmute(
      country,
      country_label,
      profile_factor,
      strategy,
      strategy_label = unname(short_labels[as.character(strategy)]),
      primary_cases_per_100k,
      current_primary_cases_per_100k,
      deterministic_relative_reduction,
      point_percent,
      relative_lower,
      relative_median,
      relative_upper,
      interval_lower_percent,
      interval_upper_percent,
      effect_label,
      effect_text_colour,
      preferred_in_program_only,
      burden_rank,
      n_bootstrap,
      confidence_interval_type,
      confidence_interval_method
    )
  selected_cell_audit <- panel_c %>%
    group_by(country) %>%
    summarise(selected = sum(preferred_in_program_only), .groups = "drop")
  if (nrow(panel_c) != length(countries) * length(strategies) ||
      any(selected_cell_audit$selected != 1L)) {
    stop("Figure 2 effect matrix does not contain one reference leader per profile.", call. = FALSE)
  }

  list(
    panel_a = panel_a,
    panel_b = panel_b,
    panel_c = panel_c,
    programme_effects = programme_effects,
    regret_draws = regret_draws,
    rank_counts = rank_counts,
    rank_samples = rank_samples,
    profile_order = profile_order,
    strategy_order = strategies,
    strategy_labels = labels,
    strategy_short_labels = short_labels,
    strategy_colours = colours,
    strategy_colors = colours,
    rank_setting_count = contract$rank_samples,
    provenance = provenance,
    interval_note = paste0(
      "Panel c brackets are paired full-refit parametric-bootstrap 95% ",
      "estimation confidence intervals calculated by the percentile method."
    ),
    sensitivity_note = paste0(
      "Panel b reports selected-input deterministic design-frequency summaries ",
      "across 128 prespecified settings; retention and regret are non-inferential ",
      "design quantities."
    )
  )
}
