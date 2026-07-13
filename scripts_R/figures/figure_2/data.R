## Figure 2 data preparation ---------------------------------------------------

figure_2_strategy_order <- function() {
  c(
    "maternal_immunization",
    "adolescent_booster",
    "cocooning_adjunct",
    "targeted_pep_high_risk",
    "pregnancy_tdap_scaleup",
    "timeliness_only"
  )
}

figure_2_winning_strategy_group_order <- function() {
  c("timeliness_only", "maternal_immunization", "targeted_pep_high_risk")
}

figure_2_strategy_labels <- function() {
  c(
    higher_child_coverage = "Nominal coverage floor",
    timeliness_only = "Routine schedule timeliness",
    maternal_immunization = "Infant-exposure package",
    cocooning_adjunct = "Close-contact adult adjuncts",
    targeted_pep_high_risk = "Targeted high-risk PEP",
    adolescent_booster = "Adolescent booster scale-up",
    pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up"
  )
}

figure_2_strategy_overview_labels <- function() {
  c(
    higher_child_coverage = "Coverage",
    maternal_immunization = "Infant-exposure\npackage",
    timeliness_only = "Routine schedule\ntimeliness",
    cocooning_adjunct = "Close-contact\nadult adjuncts",
    targeted_pep_high_risk = "Targeted high-risk\nPEP",
    adolescent_booster = "Adolescent booster\nscale-up",
    pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up"
  )
}

figure_2_strategy_heatmap_labels <- function() {
  c(
    higher_child_coverage = "Coverage",
    maternal_immunization = "Infant-exposure\npackage",
    timeliness_only = "Routine\nschedule\ntimeliness",
    cocooning_adjunct = "Close-contact\nadult\nadjuncts",
    targeted_pep_high_risk = "Targeted\nhigh-risk\nPEP",
    adolescent_booster = "Adolescent\nbooster\nscale-up",
    pregnancy_tdap_scaleup = "Pregnancy\nTdap\nscale-up"
  )
}

figure_2_selected_strategy_labels <- figure_2_strategy_overview_labels

figure_2_publication_country_contract <- function() {
  settings_path <- model_path("config", "model_settings.yaml")
  profiles_path <- model_path("config", "country_profiles.yaml")
  if (!file.exists(settings_path) || !file.exists(profiles_path)) {
    stop("Figure 2 requires current model settings and country profiles.", call. = FALSE)
  }
  settings <- yaml::read_yaml(settings_path)
  exclusions <- settings$runtime$bayesian_uncertainty$publication_country_exclusions
  exclusions <- if (is.null(exclusions)) character() else names(exclusions)
  profiles <- names(yaml::read_yaml(profiles_path))
  unknown <- setdiff(exclusions, profiles)
  if (length(unknown) > 0L) {
    stop("Unknown publication-country exclusions: ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  list(
    profiles = profiles,
    exclusions = exclusions,
    publication_countries = setdiff(profiles, exclusions)
  )
}

validate_figure_2_country_contract <- function(
  frontier_countries,
  configured_profiles,
  publication_country_exclusions,
  publication_countries
) {
  normalize <- function(x) stringr::str_replace_all(as.character(x), " ", "_")
  frontier_countries <- unique(normalize(frontier_countries))
  configured_profiles <- normalize(configured_profiles)
  publication_country_exclusions <- normalize(publication_country_exclusions)
  publication_countries <- normalize(publication_countries)
  expected <- setdiff(configured_profiles, publication_country_exclusions)
  if (!setequal(publication_countries, expected)) {
    stop("Figure 2 publication-country configuration is inconsistent.", call. = FALSE)
  }
  missing <- setdiff(publication_countries, frontier_countries)
  unknown <- setdiff(frontier_countries, configured_profiles)
  if (length(missing) > 0L || length(unknown) > 0L) {
    stop(
      "Figure 2 frontier country mismatch; missing: ", paste(missing, collapse = ", "),
      "; unknown: ", paste(unknown, collapse = ", "), call. = FALSE
    )
  }
  publication_countries
}

require_current_figure_2_python_gate <- function() {
  configured <- Sys.getenv("PERTUSSIS_PYTHON", unset = "")
  candidates <- unique(c(
    configured,
    model_path(".venv", "bin", "python"),
    Sys.which("python3"),
    Sys.which("python")
  ))
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  if (length(candidates) == 0L) {
    stop("Figure 2 requires a project Python interpreter.", call. = FALSE)
  }
  output <- suppressWarnings(system2(
    candidates[[1]],
    c("-m", "src_python.validation.publication_gate"),
    stdout = TRUE,
    stderr = TRUE
  ))
  status <- attr(output, "status") %||% 0L
  if (!identical(as.integer(status), 0L)) {
    stop("Figure 2 predictive publication gate failed: ", paste(output, collapse = "\n"), call. = FALSE)
  }
  invisible(TRUE)
}

# Backward-compatible spelling used by the lightweight R contract test.
require_current_figure2_python_gate <- require_current_figure_2_python_gate

load_figure_2_inputs <- function() {
  require_current_figure_2_python_gate()
  contract <- figure_2_publication_country_contract()
  list(
    frontier = read_table("lancet_child_adolescent_decision_frontier.csv"),
    configured_country_profiles = contract$profiles,
    publication_country_exclusions = contract$exclusions,
    publication_countries = contract$publication_countries
  )
}

prepare_figure_2_data <- function(inputs = load_figure_2_inputs()) {
  strategy_order <- figure_2_strategy_order()
  strategy_labels <- figure_2_strategy_labels()
  strategy_overview_labels <- figure_2_strategy_overview_labels()
  strategy_heatmap_labels <- figure_2_strategy_heatmap_labels()
  selected_strategy_labels <- figure_2_selected_strategy_labels()
  winning_strategy_group_order <- unique(c(
    figure_2_winning_strategy_group_order(), strategy_order
  ))
  rank_levels <- 1:6
  rank_labels <- c("1st", "2nd", "3rd", "4th", "5th", "6th")
  rank_colours <- setNames(
    grDevices::colorRampPalette(manuscript_continuous_burden)(length(rank_labels)),
    rank_labels
  )

  require_columns(
    inputs$frontier,
    c(
      "country", "strategy", "optimization_constraint", "primary_case_reduction",
      "primary_cases_per_100k", "current_primary_cases_per_100k",
      "implementation_intensity", "non_dominated_lancet_outcome"
    ),
    "lancet_child_adolescent_decision_frontier.csv"
  )
  frontier_raw <- inputs$frontier %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label_text = format_country(country),
      strategy = as.character(strategy),
      strategy_label = coalesce(strategy_labels[strategy], strategy),
      primary_case_reduction = as.numeric(primary_case_reduction),
      primary_cases_per_100k = as.numeric(primary_cases_per_100k),
      current_primary_cases_per_100k = as.numeric(current_primary_cases_per_100k),
      implementation_intensity = as.numeric(implementation_intensity),
      non_dominated_lancet_outcome = as.logical(non_dominated_lancet_outcome)
    )
  publication_countries <- validate_figure_2_country_contract(
    unique(frontier_raw$country),
    inputs$configured_country_profiles,
    inputs$publication_country_exclusions,
    inputs$publication_countries
  )
  program_frontier <- frontier_raw %>%
    filter(
      optimization_constraint == "program_only",
      country %in% publication_countries,
      strategy %in% strategy_order
    )
  expected_rows <- length(publication_countries) * length(strategy_order)
  if (nrow(program_frontier) != expected_rows ||
      anyDuplicated(program_frontier[c("country", "strategy")]) > 0L) {
    stop("Figure 2 requires one deterministic row per publication country and programme strategy.", call. = FALSE)
  }
  if (any(!is.finite(program_frontier$primary_case_reduction)) ||
      any(!is.finite(program_frontier$primary_cases_per_100k))) {
    stop("Figure 2 received non-finite deterministic scenario effects.", call. = FALSE)
  }

  program_frontier <- program_frontier %>%
    mutate(
      strategy_overview_plot = factor(
        strategy,
        levels = rev(strategy_order),
        labels = strategy_overview_labels[rev(strategy_order)]
      )
    ) %>%
    arrange(country, primary_cases_per_100k, implementation_intensity, strategy) %>%
    group_by(country) %>%
    mutate(
      decision_rank = row_number(),
      best_cases_per_100k = first(primary_cases_per_100k),
      best_primary_case_reduction = first(primary_case_reduction),
      margin_to_best_cases_per_100k = primary_cases_per_100k - best_cases_per_100k,
      margin_to_best_percentage_points = 100 * (best_primary_case_reduction - primary_case_reduction),
      near_best_5_cases = margin_to_best_cases_per_100k <= 5,
      cases_averted_per_100k = current_primary_cases_per_100k - primary_cases_per_100k
    ) %>%
    ungroup()

  selected_program <- program_frontier %>%
    arrange(country, decision_rank) %>%
    group_by(country, country_label_text) %>%
    summarise(
      current_cases_per_100k = first(current_primary_cases_per_100k),
      winning_strategy = first(strategy),
      winning_strategy_label = first(strategy_label),
      winning_strategy_short = first(as.character(strategy_overview_plot)),
      winning_reduction = first(primary_case_reduction),
      winning_cases_per_100k = first(primary_cases_per_100k),
      runner_up_strategy = nth(strategy, 2),
      runner_up_strategy_label = nth(strategy_label, 2),
      runner_up_reduction = nth(primary_case_reduction, 2),
      runner_up_cases_per_100k = nth(primary_cases_per_100k, 2),
      winner_margin_reduction = winning_reduction - runner_up_reduction,
      winner_margin_cases_per_100k = runner_up_cases_per_100k - winning_cases_per_100k,
      uncertainty_interval_type = "none; conditional scenario point estimate",
      uncertainty_interval_basis = "deterministic within-profile contrast",
      .groups = "drop"
    ) %>%
    mutate(
      selected_cases_averted_per_100k = current_cases_per_100k - winning_cases_per_100k,
      selected_strategy_text = stringr::str_replace_all(winning_strategy_short, "\n", " "),
      runner_up_excess_label = paste0("+", format_lancet_fixed(winner_margin_cases_per_100k, digits = 1)),
      winning_strategy_group = factor(winning_strategy, levels = winning_strategy_group_order)
    )
  heatmap_country_order <- selected_program %>%
    arrange(winning_strategy_group, desc(winner_margin_cases_per_100k), country_label_text) %>%
    pull(country_label_text)
  heatmap_group_sizes <- selected_program %>%
    count(winning_strategy_group, name = "profiles") %>%
    filter(!is.na(winning_strategy_group)) %>%
    arrange(winning_strategy_group)
  heatmap_group_separators <- if (nrow(heatmap_group_sizes) > 1L) {
    length(heatmap_country_order) -
      cumsum(heatmap_group_sizes$profiles)[-nrow(heatmap_group_sizes)] + 0.5
  } else {
    numeric()
  }
  selected_program <- selected_program %>%
    mutate(country_label_margin = factor(country_label_text, levels = rev(heatmap_country_order)))
  selected_strategy_order <- strategy_order[strategy_order %in% selected_program$winning_strategy]
  selected_strategy_legend_labels <- stringr::str_replace_all(
    selected_strategy_labels[selected_strategy_order], "\n", " "
  )

  strategy_distribution <- program_frontier %>%
    group_by(strategy, strategy_label, strategy_overview_plot) %>%
    summarise(
      median_reduction = median(primary_case_reduction),
      q25_reduction = interval_quantile(primary_case_reduction, 0.25),
      q75_reduction = interval_quantile(primary_case_reduction, 0.75),
      min_reduction = min(primary_case_reduction),
      max_reduction = max(primary_case_reduction),
      countries_ranked_first = sum(decision_rank == 1),
      countries_non_dominated = sum(non_dominated_lancet_outcome),
      .groups = "drop"
    )
  program_heatmap <- program_frontier %>%
    mutate(
      country_label = factor(country_label_text, levels = rev(heatmap_country_order)),
      strategy_axis = factor(strategy, levels = strategy_order),
      primary_case_reduction_display = primary_case_reduction,
      effect_label = lancet_percent(primary_case_reduction, accuracy = 0.1),
      effect_text_colour = if_else(primary_case_reduction_display >= 0.28, "white", lancet_text_colour),
      preferred_in_program_only = decision_rank == 1
    )
  rank_distribution <- program_frontier %>%
    count(strategy, decision_rank, name = "profile_count") %>%
    tidyr::complete(strategy = strategy_order, decision_rank = rank_levels, fill = list(profile_count = 0L)) %>%
    mutate(
      strategy_label = strategy_labels[strategy],
      rank_label = factor(decision_rank, levels = rank_levels, labels = rank_labels),
      strategy_y = as.numeric(factor(strategy, levels = rev(strategy_order))),
      segment_text_colour = if_else(decision_rank %in% c(1L, 6L), "white", lancet_text_colour)
    ) %>%
    arrange(strategy, decision_rank) %>%
    group_by(strategy) %>%
    mutate(
      xmin = cumsum(lag(profile_count, default = 0L)),
      xmax = xmin + profile_count,
      xmid = (xmin + xmax) / 2,
      segment_label = if_else(profile_count > 0, as.character(profile_count), NA_character_),
      rank_total_profiles = sum(profile_count)
    ) %>%
    ungroup()

  list(
    strategy_order = strategy_order,
    strategy_labels = strategy_labels,
    strategy_overview_labels = strategy_overview_labels,
    strategy_heatmap_labels = strategy_heatmap_labels,
    rank_labels = rank_labels,
    rank_colours = rank_colours,
    publication_countries = publication_countries,
    publication_country_exclusions = inputs$publication_country_exclusions,
    primary_interval_legend_title = "Conditional scenario\nreduction (%)",
    program_frontier = program_frontier,
    selected_program = selected_program,
    heatmap_country_order = heatmap_country_order,
    heatmap_group_sizes = heatmap_group_sizes,
    heatmap_group_separators = heatmap_group_separators,
    selected_strategy_order = selected_strategy_order,
    selected_strategy_legend_labels = selected_strategy_legend_labels,
    strategy_distribution = strategy_distribution,
    program_heatmap = program_heatmap,
    rank_distribution = rank_distribution
  )
}
