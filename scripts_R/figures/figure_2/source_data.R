## Figure 2 source data --------------------------------------------------------

figure_2_source_filenames <- c(
  delivery_lever_contrast = "figure2a_delivery_lever_contrast.csv",
  decision_fragility = "figure2b_reference_choice_fragility.csv",
  regret_draws = "figure2b_fixed_reference_regret_draws.csv",
  effect_matrix = "figure2c_programme_effect_matrix.csv",
  rank1_counts = "figure2_selected_input_rank1_counts.csv",
  provenance = "figure2_parent_provenance.csv"
)

figure_2_parent_bundle_sha256 <- function(provenance) {
  required_parents <- c(
    "reference",
    "bootstrap",
    "selected_input_rank",
    "production_intervention"
  )
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("Figure 2 bundle provenance requires the R package 'digest'.", call. = FALSE)
  }
  if (nrow(provenance) != length(required_parents) ||
      anyDuplicated(as.character(provenance$parent)) ||
      !setequal(as.character(provenance$parent), required_parents)) {
    stop("Figure 2 bundle provenance requires exactly four named parents.", call. = FALSE)
  }
  ordered <- provenance[match(required_parents, provenance$parent), , drop = FALSE]
  artifact_sha256 <- as.character(ordered$artifact_sha256)
  valid_sha256 <- !is.na(artifact_sha256) &
    grepl("^[[:xdigit:]]{64}$", artifact_sha256)
  if (!all(valid_sha256)) {
    stop("Figure 2 bundle provenance contains an invalid parent SHA-256.", call. = FALSE)
  }
  payload <- paste0(required_parents, "=", tolower(artifact_sha256), collapse = "\n")
  digest::digest(payload, algo = "sha256", serialize = FALSE)
}

make_figure_2_source_data <- function(data) {
  provenance <- tibble::tibble(
    parent = c(
      "reference",
      "bootstrap",
      "selected_input_rank",
      "production_intervention"
    ),
    config_hash = data$provenance$config_hash,
    source_code_hash = data$provenance$source_code_hash,
    git_commit = data$provenance$git_commit,
    artifact_sha256 = unname(
      data$provenance$artifact_sha256[
        c("reference", "bootstrap", "rank", "intervention")
      ]
    ),
    generated_at_utc = c(
      data$provenance$reference_generated_at_utc,
      data$provenance$bootstrap_generated_at_utc,
      data$provenance$rank_generated_at_utc,
      data$provenance$intervention_generated_at_utc
    ),
    estimand_or_role = c(
      paste0(
        "Common-production-runtime current comparator, routine-timeliness ",
        "contrast, and six-strategy locked estimates"
      ),
      paste0(
        "Paired full-refit parametric-bootstrap 95% estimation confidence ",
        "intervals for all 54 profile-strategy contrasts"
      ),
      paste0(
        "Selected-input deterministic design-frequency summary of reference-choice ",
        "retention and fixed-reference regret across 128 prespecified settings"
      ),
      paste0(
        "Production-runtime current comparator and nominal coverage-floor-only ",
        "contrast"
      )
    )
  )
  parent_bundle_sha256 <- figure_2_parent_bundle_sha256(provenance)

  effect_matrix <- data$programme_effects %>%
    left_join(
      data$panel_c %>%
        transmute(
          country,
          strategy = as.character(strategy),
          burden_rank,
          reference_leader = preferred_in_program_only
        ),
      by = c("country", "strategy")
    ) %>%
    transmute(
      country,
      strategy,
      strategy_label,
      burden_rank,
      reference_leader,
      deterministic_primary_cases_per_100k = primary_cases_per_100k,
      current_practice_cases_per_100k = current_primary_cases_per_100k,
      deterministic_absolute_reduction_per_100k = point,
      deterministic_relative_reduction,
      bootstrap_relative_reduction_median = relative_median,
      bootstrap_relative_reduction_q025 = relative_lower,
      bootstrap_relative_reduction_q25 = relative_q25,
      bootstrap_relative_reduction_q75 = relative_q75,
      bootstrap_relative_reduction_q975 = relative_upper,
      bootstrap_absolute_reduction_median = median,
      bootstrap_absolute_reduction_q025 = lower,
      bootstrap_absolute_reduction_q25 = q25,
      bootstrap_absolute_reduction_q75 = q75,
      bootstrap_absolute_reduction_q975 = upper,
      successful_bootstrap_replicates = n_bootstrap,
      confidence_interval_type,
      confidence_interval_method
    )

  source_data <- list(
    delivery_lever_contrast = data$panel_a %>%
      transmute(
        country,
        configured_lever = as.character(scenario),
        configured_lever_label = scenario_label,
        deterministic_primary_cases_per_100k = primary_cases_per_100k,
        current_practice_cases_per_100k = current_primary_cases_per_100k,
        deterministic_relative_reduction = relative_reduction,
        deterministic_relative_reduction_percent = relative_reduction_percent,
        coverage_floor_applied,
        timeliness_applied,
        parent_artifact
      ),
    decision_fragility = data$panel_b %>%
      transmute(
        country,
        locked_reference_strategy = reference_strategy,
        locked_reference_strategy_label = reference_strategy_label,
        prespecified_setting_count = setting_count,
        reference_choice_retained_count = reference_choice_retained,
        reference_choice_retained_fraction_for_arithmetic_check = retained_fraction,
        positive_regret_count,
        regret_percentage_points_median = regret_pp_median,
        regret_percentage_points_q75 = regret_pp_q75,
        regret_percentage_points_q95 = regret_pp_q95,
        regret_percentage_points_q975 = regret_pp_q975,
        regret_percentage_points_max = regret_pp_max,
        regret_cases_per_100k_q95,
        interpretation = paste0(
          "Selected-input deterministic design-frequency summaries of ",
          "reference-choice retention and fixed-reference regret across ",
          "prespecified settings; non-inferential design quantities"
        )
      ),
    regret_draws = data$regret_draws %>%
      transmute(
        country,
        selected_input_setting = psa_sample_id,
        locked_reference_strategy = reference_strategy,
        setting_specific_winning_strategy = winning_strategy,
        reference_choice_retained,
        current_practice_cases_per_100k = current_primary_cases_per_100k,
        locked_reference_strategy_cases_per_100k =
          reference_primary_cases_per_100k,
        setting_specific_best_cases_per_100k = best_primary_cases_per_100k,
        fixed_reference_regret_cases_per_100k = regret_cases_per_100k,
        fixed_reference_regret_relative_to_current =
          regret_relative_to_current,
        fixed_reference_regret_percentage_points = regret_percentage_points
      ),
    effect_matrix = effect_matrix,
    rank1_counts = data$rank_counts %>%
      transmute(
        country,
        strategy,
        rank1_count = count,
        prespecified_setting_count = data$rank_setting_count,
        rank1_fraction_for_arithmetic_check = count / data$rank_setting_count,
        interpretation = paste0(
          "Selected-input deterministic design-frequency count retained as an ",
          "audit extract across prespecified settings"
        )
      ),
    provenance = provenance
  )
  purrr::map(
    source_data,
    ~ dplyr::mutate(
      .x,
      figure2_parent_bundle_sha256 = .env$parent_bundle_sha256
    )
  )
}

write_figure_2_source_data <- function(
  source_data,
  output_dir = model_path("outputs", "tables"),
  filenames = figure_2_source_filenames
) {
  if (!setequal(names(source_data), names(filenames))) {
    stop(
      "Figure 2 source-data names do not match the locked filename map.",
      call. = FALSE
    )
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- file.path(output_dir, unname(filenames[names(source_data)]))
  names(paths) <- names(source_data)
  purrr::iwalk(source_data, function(data, name) {
    readr::write_csv(data, paths[[name]])
  })
  invisible(paths)
}
